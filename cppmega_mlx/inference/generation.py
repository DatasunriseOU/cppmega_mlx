"""Eager MLX generation loops for the Mac-local inference path."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Literal, cast

import mlx.core as mx

from cppmega_mlx.inference.engine import (
    ContiguousKVCache,
    ContiguousKVCacheConfig,
    make_contiguous_kv_cache,
)
from cppmega_mlx.inference.sampling import sample_next_token

GenerationFinishReason = Literal["eos", "length"]


@dataclass(frozen=True)
class GenerationChunk:
    """One generated-token event from the local MLX streaming loop."""

    token_ids: mx.array
    tokens: mx.array
    text: str | list[str] | None = None
    finish_reason: GenerationFinishReason | None = None


def _model_max_seq_length(model: Any) -> int | None:
    config = getattr(model, "config", None)
    max_seq_length = getattr(config, "max_seq_length", None)
    if max_seq_length is None:
        return None
    max_seq_length_int = int(max_seq_length)
    if max_seq_length_int <= 0:
        raise ValueError("model.config.max_seq_length must be positive")
    return max_seq_length_int


def _validate_logits_shape(logits: mx.array, tokens: mx.array) -> None:
    if len(logits.shape) == 4:
        raise ValueError(
            "MTP/draft logits with shape (batch, depth, sequence, vocab) are not "
            "supported by standard next-token inference"
        )
    if len(logits.shape) != 3:
        raise ValueError("model logits must have shape (batch, sequence, vocab)")
    if logits.shape[0] != tokens.shape[0]:
        raise ValueError("model logits batch size must match current tokens")
    if logits.shape[1] != tokens.shape[1]:
        raise ValueError("model logits sequence length must match current tokens")


def next_token_logits(model_output: Any, tokens: mx.array) -> mx.array:
    """Return standard next-token logits from a model output.

    The Stream I eager path intentionally accepts only the plain inference
    contract: one ``(batch, sequence, vocab)`` tensor. Structured outputs and
    MTP/draft tensors are rejected until the speculative/self-spec paths land.
    """

    if isinstance(model_output, tuple | list):
        raise ValueError(
            "MTP/draft tuple outputs are not supported by standard next-token "
            "inference"
        )
    if isinstance(model_output, dict):
        raise ValueError(
            "structured model outputs are not supported by standard next-token "
            "inference; pass plain logits"
        )
    if not isinstance(model_output, mx.array):
        raise TypeError("model output must be an mlx.core.array of logits")

    _validate_logits_shape(model_output, tokens)
    return model_output[:, -1, :]


def generate_tokens(
    model: Any,
    prompt_ids: mx.array,
    *,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = 1.0,
    rng_key: Any | None = None,
) -> mx.array:
    """Generate tokens by recomputing ``model(tokens)`` on the full prefix.

    This is intentionally the small MLX-native Stream I bootstrap path: no KV
    cache, no paged serving, and no per-row EOS masking. It mirrors nanochat's
    eager no-cache loop closely enough to lock sampling/decode semantics before
    the larger cache and serving ports land.
    """
    if len(prompt_ids.shape) != 2:
        raise ValueError("prompt_ids must have shape (batch, sequence)")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    max_seq_length = _model_max_seq_length(model)
    if max_seq_length is not None and prompt_ids.shape[1] > max_seq_length:
        raise ValueError("prompt_ids already exceed model.config.max_seq_length")
    if max_new_tokens == 0:
        return prompt_ids

    tokens = prompt_ids
    key = rng_key
    for _ in range(max_new_tokens):
        if max_seq_length is not None and tokens.shape[1] >= max_seq_length:
            raise ValueError("generation would exceed model.config.max_seq_length")

        step_logits = next_token_logits(model(tokens), tokens)

        step_key = None
        if key is not None:
            key, step_key = mx.random.split(key, 2)

        next_token = sample_next_token(
            step_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            rng_key=step_key,
        ).astype(tokens.dtype)
        tokens = mx.concatenate([tokens, next_token], axis=1)

        if eos_token_id is not None:
            eos_matches = cast(mx.array, next_token[:, 0] == eos_token_id)
            if bool(mx.all(eos_matches)):
                break

    return tokens


def generate_tokens_with_kv_cache(
    model: Any,
    prompt_ids: mx.array,
    *,
    max_new_tokens: int,
    cache: ContiguousKVCache | None = None,
    cache_config: ContiguousKVCacheConfig | None = None,
    num_layers: int | None = None,
    num_kv_heads: int | None = None,
    head_dim: int | None = None,
    max_seq_len: int | None = None,
    dtype: mx.Dtype | None = None,
    quantized: bool = False,
    kv_bits: int = 4,
    kv_group_size: int = 64,
    eos_token_id: int | None = None,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = 1.0,
    rng_key: Any | None = None,
) -> mx.array:
    """Generate with one prompt prefill and one-token KV-cache decode steps.

    This is the Mac-local contiguous-cache path matching nanochat's serving
    contract at the generation-loop seam. It does not implement paged serving,
    prompt caching, streaming, or model-integrated attention cache plumbing.
    """

    if len(prompt_ids.shape) != 2:
        raise ValueError("prompt_ids must have shape (batch, sequence)")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    max_seq_length = _model_max_seq_length(model)
    if max_seq_length is not None and prompt_ids.shape[1] > max_seq_length:
        raise ValueError("prompt_ids already exceed model.config.max_seq_length")
    if max_new_tokens == 0:
        return prompt_ids

    tokens = prompt_ids
    kv_cache = _resolve_kv_cache(
        cache=cache,
        cache_config=cache_config,
        batch_size=int(prompt_ids.shape[0]),
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        dtype=dtype,
        quantized=quantized,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
    )

    key = rng_key
    step_logits = next_token_logits(model(tokens, kv_cache=kv_cache), tokens)
    for step in range(max_new_tokens):
        if max_seq_length is not None and tokens.shape[1] >= max_seq_length:
            raise ValueError("generation would exceed model.config.max_seq_length")

        step_key = None
        if key is not None:
            key, step_key = mx.random.split(key, 2)

        next_token = sample_next_token(
            step_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            rng_key=step_key,
        ).astype(tokens.dtype)
        tokens = mx.concatenate([tokens, next_token], axis=1)

        if eos_token_id is not None:
            eos_matches = cast(mx.array, next_token[:, 0] == eos_token_id)
            if bool(mx.all(eos_matches)):
                break

        if step + 1 >= max_new_tokens:
            break
        if max_seq_length is not None and tokens.shape[1] >= max_seq_length:
            raise ValueError("generation would exceed model.config.max_seq_length")

        step_logits = next_token_logits(model(next_token, kv_cache=kv_cache), next_token)

    return tokens


def stream_generate_tokens(
    model: Any,
    prompt_ids: mx.array,
    *,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = 1.0,
    rng_key: Any | None = None,
    use_kv_cache: bool = False,
    cache: ContiguousKVCache | None = None,
    cache_config: ContiguousKVCacheConfig | None = None,
    num_layers: int | None = None,
    num_kv_heads: int | None = None,
    head_dim: int | None = None,
    max_seq_len: int | None = None,
    dtype: mx.Dtype | None = None,
    quantized: bool = False,
    kv_bits: int = 4,
    kv_group_size: int = 64,
    decode_token: Callable[[int], str] | None = None,
) -> Iterator[GenerationChunk]:
    """Yield generated tokens one step at a time.

    This is a local Stream I compatibility seam over the existing eager and
    contiguous-KV generation loops. Batch rows follow the same all-rows-EOS
    stop rule as ``generate_tokens`` and ``generate_tokens_with_kv_cache``.
    """

    if len(prompt_ids.shape) != 2:
        raise ValueError("prompt_ids must have shape (batch, sequence)")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if max_new_tokens == 0:
        return

    max_seq_length = _model_max_seq_length(model)
    if max_seq_length is not None and prompt_ids.shape[1] > max_seq_length:
        raise ValueError("prompt_ids already exceed model.config.max_seq_length")

    if use_kv_cache:
        yield from _stream_generate_tokens_with_kv_cache(
            model,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            rng_key=rng_key,
            cache=cache,
            cache_config=cache_config,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            dtype=dtype,
            quantized=quantized,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            decode_token=decode_token,
            max_seq_length=max_seq_length,
        )
        return

    if cache is not None or cache_config is not None or any(
        value is not None for value in (num_layers, num_kv_heads, head_dim, max_seq_len, dtype)
    ):
        raise ValueError("KV-cache configuration requires use_kv_cache=True")
    if quantized:
        raise ValueError("quantized KV-cache streaming requires use_kv_cache=True")

    key = rng_key
    tokens = prompt_ids
    for step in range(max_new_tokens):
        if max_seq_length is not None and tokens.shape[1] >= max_seq_length:
            raise ValueError("generation would exceed model.config.max_seq_length")

        step_logits = next_token_logits(model(tokens), tokens)
        key, step_key = _split_generation_key(key)
        next_token = sample_next_token(
            step_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            rng_key=step_key,
        ).astype(tokens.dtype)
        tokens = mx.concatenate([tokens, next_token], axis=1)
        yield _make_generation_chunk(
            token_ids=next_token,
            tokens=tokens,
            eos_token_id=eos_token_id,
            is_last_step=step + 1 >= max_new_tokens,
            decode_token=decode_token,
        )
        if eos_token_id is not None and _all_rows_match_token(next_token, eos_token_id):
            break


def _resolve_kv_cache(
    *,
    cache: ContiguousKVCache | None,
    cache_config: ContiguousKVCacheConfig | None,
    batch_size: int,
    num_layers: int | None,
    num_kv_heads: int | None,
    head_dim: int | None,
    max_seq_len: int | None,
    dtype: mx.Dtype | None,
    quantized: bool,
    kv_bits: int,
    kv_group_size: int,
) -> ContiguousKVCache:
    if cache is not None:
        if cache_config is not None or any(
            value is not None
            for value in (num_layers, num_kv_heads, head_dim, max_seq_len, dtype)
        ):
            raise ValueError("pass either cache or cache configuration, not both")
        if cache.config.batch_size != batch_size:
            raise ValueError("cache batch_size must match prompt_ids batch size")
        return cache

    if cache_config is not None:
        if any(value is not None for value in (num_layers, num_kv_heads, head_dim)):
            raise ValueError("pass either cache_config or shape kwargs, not both")
        if cache_config.batch_size != batch_size:
            raise ValueError("cache_config batch_size must match prompt_ids batch size")
        return make_contiguous_kv_cache(cache_config)

    return make_contiguous_kv_cache(
        num_layers=num_layers,
        batch_size=batch_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        dtype=dtype,
        quantized=quantized,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
    )


def _stream_generate_tokens_with_kv_cache(
    model: Any,
    prompt_ids: mx.array,
    *,
    max_new_tokens: int,
    eos_token_id: int | None,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    rng_key: Any | None,
    cache: ContiguousKVCache | None,
    cache_config: ContiguousKVCacheConfig | None,
    num_layers: int | None,
    num_kv_heads: int | None,
    head_dim: int | None,
    max_seq_len: int | None,
    dtype: mx.Dtype | None,
    quantized: bool,
    kv_bits: int,
    kv_group_size: int,
    decode_token: Callable[[int], str] | None,
    max_seq_length: int | None,
) -> Iterator[GenerationChunk]:
    tokens = prompt_ids
    kv_cache = _resolve_kv_cache(
        cache=cache,
        cache_config=cache_config,
        batch_size=int(prompt_ids.shape[0]),
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        dtype=dtype,
        quantized=quantized,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
    )

    key = rng_key
    step_logits = next_token_logits(model(tokens, kv_cache=kv_cache), tokens)
    for step in range(max_new_tokens):
        if max_seq_length is not None and tokens.shape[1] >= max_seq_length:
            raise ValueError("generation would exceed model.config.max_seq_length")

        key, step_key = _split_generation_key(key)
        next_token = sample_next_token(
            step_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            rng_key=step_key,
        ).astype(tokens.dtype)
        tokens = mx.concatenate([tokens, next_token], axis=1)
        yield _make_generation_chunk(
            token_ids=next_token,
            tokens=tokens,
            eos_token_id=eos_token_id,
            is_last_step=step + 1 >= max_new_tokens,
            decode_token=decode_token,
        )
        if eos_token_id is not None and _all_rows_match_token(next_token, eos_token_id):
            break
        if step + 1 >= max_new_tokens:
            break
        if max_seq_length is not None and tokens.shape[1] >= max_seq_length:
            raise ValueError("generation would exceed model.config.max_seq_length")
        step_logits = next_token_logits(model(next_token, kv_cache=kv_cache), next_token)


def _split_generation_key(key: Any | None) -> tuple[Any | None, Any | None]:
    if key is None:
        return None, None
    next_key, step_key = mx.random.split(key, 2)
    return next_key, step_key


def _make_generation_chunk(
    *,
    token_ids: mx.array,
    tokens: mx.array,
    eos_token_id: int | None,
    is_last_step: bool,
    decode_token: Callable[[int], str] | None,
) -> GenerationChunk:
    next_ids = _token_ids_by_row(token_ids)
    finish_reason: GenerationFinishReason | None = None
    if eos_token_id is not None and all(token_id == eos_token_id for token_id in next_ids):
        finish_reason = "eos"
    elif is_last_step:
        finish_reason = "length"
    text: str | list[str] | None = None
    if decode_token is not None:
        decoded = [decode_token(token_id) for token_id in next_ids]
        text = decoded[0] if len(decoded) == 1 else decoded
    return GenerationChunk(
        token_ids=token_ids,
        tokens=tokens,
        text=text,
        finish_reason=finish_reason,
    )


def _token_ids_by_row(token_ids: mx.array) -> list[int]:
    if len(token_ids.shape) != 2 or token_ids.shape[1] != 1:
        raise ValueError("streaming generation expects token_ids with shape (batch, 1)")
    return [int(token_ids[row, 0].item()) for row in range(int(token_ids.shape[0]))]


def _all_rows_match_token(token_ids: mx.array, token_id: int) -> bool:
    if len(token_ids.shape) != 2 or token_ids.shape[1] != 1:
        raise ValueError("streaming generation expects token_ids with shape (batch, 1)")
    matches = cast(mx.array, token_ids[:, 0] == token_id)
    return bool(mx.all(matches))

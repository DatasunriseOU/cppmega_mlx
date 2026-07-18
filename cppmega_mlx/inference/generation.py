"""Eager MLX generation loops for the Mac-local inference path."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import mlx.core as mx

from cppmega_mlx.data.graph_packet import GraphBatch
from cppmega_mlx.inference.engine import (
    ContiguousKVCache,
    ContiguousKVCacheConfig,
    PromptCacheEntry,
    clone_contiguous_kv_cache,
    kv_cache_position,
    make_contiguous_kv_cache,
)
from cppmega_mlx.inference.constraints import LogitsProcessor
from cppmega_mlx.inference.sampling import sample_next_token
from cppmega_mlx.inference.speculative_decode import speculative_acceptance

GenerationFinishReason = Literal["eos", "length"]
ModelKwargs = Mapping[str, Any]

_ZERO_GENERATED_MODEL_KWARGS = frozenset(
    {
        "structure_ids",
        "dep_levels",
        "ast_depth_ids",
        "sibling_index_ids",
        "node_type_ids",
        # Generated tokens have no source-domain provenance until a new
        # document-sidecar record is materialized.
        "domain_ids",
        "role_ids",
        "entity_ids",
        "scope_ids",
        "source_doc_ids",
        "source_identity_ids",
        "confidence_ids",
    }
)
_REPEAT_GENERATED_MODEL_KWARGS = frozenset({"document_ids", "platform_ids"})
_GRAPH_BIAS_MODEL_KWARGS = frozenset({"block_bias", "edge_kind_bias"})
_NON_SEQUENCE_ALIGNED_MODEL_KWARGS = frozenset(
    {
        "domain_edges",
        "build_edges",
        "shell_edges",
        "diagnostic_edges",
        "cross_domain_edges",
        "token_domain_edges",
        "token_build_edges",
        "token_shell_edges",
        "token_diagnostic_edges",
        "token_cross_domain_edges",
        "graph_batch",
    }
)
_SEQUENCE_ALIGNED_MODEL_KWARGS = (
    _ZERO_GENERATED_MODEL_KWARGS
    | _REPEAT_GENERATED_MODEL_KWARGS
    | _GRAPH_BIAS_MODEL_KWARGS
)
_PROMPT_CACHE_UNSAFE_ROUTE_ROLES = frozenset({"mamba3", "m2rnn", "engram"})


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

    return _standard_generation_logits(model_output, tokens)[:, -1, :]


def generate_tokens(
    model: Any,
    prompt_ids: mx.array,
    *,
    max_new_tokens: int,
    model_kwargs: ModelKwargs | None = None,
    eos_token_id: int | None = None,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = 1.0,
    rng_key: Any | None = None,
    logits_processors: Sequence[LogitsProcessor] | None = None,
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

        step_logits = next_token_logits(
            model(tokens, **_model_kwargs_for_prefix(model_kwargs, tokens)),
            tokens,
        )

        step_key = None
        if key is not None:
            key, step_key = mx.random.split(key, 2)

        next_token = sample_next_token(
            step_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            rng_key=step_key,
            tokens=tokens,
            logits_processors=logits_processors,
        ).astype(tokens.dtype)
        tokens = mx.concatenate([tokens, next_token], axis=1)

        if eos_token_id is not None:
            eos_matches = cast(mx.array, next_token[:, 0] == eos_token_id)
            if bool(mx.all(eos_matches)):
                break

    return tokens


def generate_tokens_speculative(
    target_model: Any,
    draft_model: Any,
    prompt_ids: mx.array,
    *,
    max_new_tokens: int,
    draft_window: int = 4,
    model_kwargs: ModelKwargs | None = None,
    draft_model_kwargs: ModelKwargs | None = None,
    eos_token_id: int | None = None,
    temperature: float = 1.0,
    rng_key: Any | None = None,
) -> mx.array:
    """Generate with vanilla speculative decoding on the eager MLX path.

    The draft model proposes up to ``draft_window`` tokens, then the target
    model verifies those tokens with one full-prefix forward and the existing
    Leviathan-style acceptance-rejection helper. This scoped Stream I slice is
    deliberately batch=1 and no-KV; paged attention, EAGLE, and MTP
    self-speculation remain separate rows.
    """

    if len(prompt_ids.shape) != 2:
        raise ValueError("prompt_ids must have shape (batch, sequence)")
    if int(prompt_ids.shape[0]) != 1:
        raise ValueError("speculative generation currently supports batch=1")
    if int(prompt_ids.shape[1]) <= 0:
        raise ValueError("prompt_ids must contain at least one token")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if draft_window <= 0:
        raise ValueError("draft_window must be positive")
    if max_new_tokens == 0:
        return prompt_ids

    target_max_seq_length = _model_max_seq_length(target_model)
    draft_max_seq_length = _model_max_seq_length(draft_model)
    _validate_prefix_fits_max_length(prompt_ids, target_max_seq_length)
    _validate_prefix_fits_max_length(prompt_ids, draft_max_seq_length)

    tokens = prompt_ids
    generated = 0
    key = rng_key
    resolved_draft_model_kwargs = draft_model_kwargs
    if resolved_draft_model_kwargs is None:
        resolved_draft_model_kwargs = model_kwargs

    while generated < max_new_tokens:
        remaining = max_new_tokens - generated
        append_limit = remaining
        window = min(draft_window, remaining)
        target_slots = _available_generation_slots(tokens, target_max_seq_length)
        if target_slots is not None:
            window = min(window, target_slots)
            append_limit = min(append_limit, target_slots)
        draft_slots = _available_generation_slots(tokens, draft_max_seq_length)
        if draft_slots is not None:
            window = min(window, draft_slots)
        draft_tokens, draft_logits, key = _propose_speculative_draft_window(
            draft_model,
            tokens,
            draft_window=window,
            model_kwargs=resolved_draft_model_kwargs,
            eos_token_id=eos_token_id,
            temperature=temperature,
            rng_key=key,
            max_seq_length=draft_max_seq_length,
        )
        candidate = mx.concatenate([tokens, draft_tokens[None, :].astype(tokens.dtype)], axis=1)
        _validate_prefix_fits_max_length(candidate, target_max_seq_length)

        target_logits = _standard_generation_logits(
            target_model(
                candidate,
                **_model_kwargs_for_prefix(model_kwargs, candidate),
            ),
            candidate,
        )
        start = int(tokens.shape[1]) - 1
        verifier_logits = target_logits[0, start : start + int(draft_tokens.shape[0]) + 1, :]

        key, acceptance_key = _split_generation_key(key)
        accepted, _n_accepted, next_token = speculative_acceptance(
            draft_logits,
            verifier_logits,
            draft_tokens,
            temperature=temperature,
            rng_key=acceptance_key,
        )
        append_tokens, found_eos = _speculative_append_tokens(
            accepted,
            next_token,
            remaining=append_limit,
            eos_token_id=eos_token_id,
        )
        tokens = mx.concatenate([tokens, append_tokens[None, :].astype(tokens.dtype)], axis=1)
        generated += int(append_tokens.shape[0])
        if found_eos:
            break

    return tokens


def generate_tokens_mtp_self_speculative(
    model: Any,
    prompt_ids: mx.array,
    *,
    max_new_tokens: int,
    draft_window: int | None = None,
    model_kwargs: ModelKwargs | None = None,
    eos_token_id: int | None = None,
    temperature: float = 1.0,
    rng_key: Any | None = None,
) -> mx.array:
    """Generate with a model-owned MTP head as the speculative drafter.

    This is the scoped FastMTP-aligned Stream I path: the same model computes
    the last verified hidden state, its attached ``mtp_head`` drafts a bounded
    token window, and the model verifies the candidate prefix with one normal
    target forward. It is deliberately eager batch=1 and no-KV.
    """

    if len(prompt_ids.shape) != 2:
        raise ValueError("prompt_ids must have shape (batch, sequence)")
    if int(prompt_ids.shape[0]) != 1:
        raise ValueError("MTP self-speculative generation currently supports batch=1")
    if int(prompt_ids.shape[1]) <= 0:
        raise ValueError("prompt_ids must contain at least one token")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if max_new_tokens == 0:
        return prompt_ids

    mtp_head = _resolve_mtp_draft_head(model)
    trained_depth = _mtp_head_trained_depth(mtp_head)
    window_limit = trained_depth if draft_window is None else draft_window
    if window_limit <= 0:
        raise ValueError("draft_window must be positive")
    if window_limit > trained_depth:
        raise ValueError("draft_window must not exceed model.mtp_head.config.depth")

    max_seq_length = _model_max_seq_length(model)
    _validate_prefix_fits_max_length(prompt_ids, max_seq_length)

    tokens = prompt_ids
    generated = 0
    key = rng_key
    while generated < max_new_tokens:
        remaining = max_new_tokens - generated
        append_limit = remaining
        window = min(window_limit, remaining)
        target_slots = _available_generation_slots(tokens, max_seq_length)
        if target_slots is not None:
            window = min(window, target_slots)
            append_limit = min(append_limit, target_slots)

        draft_tokens, draft_logits, key = _propose_mtp_self_speculative_window(
            model,
            mtp_head,
            tokens,
            draft_window=window,
            model_kwargs=model_kwargs,
            eos_token_id=eos_token_id,
            temperature=temperature,
            rng_key=key,
        )
        candidate = mx.concatenate([tokens, draft_tokens[None, :].astype(tokens.dtype)], axis=1)
        _validate_prefix_fits_max_length(candidate, max_seq_length)

        target_logits = _standard_generation_logits(
            model(
                candidate,
                **_model_kwargs_for_prefix(model_kwargs, candidate),
            ),
            candidate,
        )
        start = int(tokens.shape[1]) - 1
        verifier_logits = target_logits[0, start : start + int(draft_tokens.shape[0]) + 1, :]

        key, acceptance_key = _split_generation_key(key)
        accepted, _n_accepted, next_token = speculative_acceptance(
            draft_logits,
            verifier_logits,
            draft_tokens,
            temperature=temperature,
            rng_key=acceptance_key,
        )
        append_tokens, found_eos = _speculative_append_tokens(
            accepted,
            next_token,
            remaining=append_limit,
            eos_token_id=eos_token_id,
        )
        tokens = mx.concatenate([tokens, append_tokens[None, :].astype(tokens.dtype)], axis=1)
        generated += int(append_tokens.shape[0])
        if found_eos:
            break

    return tokens


def generate_tokens_with_kv_cache(
    model: Any,
    prompt_ids: mx.array,
    *,
    max_new_tokens: int,
    model_kwargs: ModelKwargs | None = None,
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
    logits_processors: Sequence[LogitsProcessor] | None = None,
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
    step_logits = next_token_logits(
        model(
            tokens,
            kv_cache=kv_cache,
            **_model_kwargs_for_prefix(model_kwargs, tokens),
        ),
        tokens,
    )
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
            tokens=tokens,
            logits_processors=logits_processors,
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

        step_logits = next_token_logits(
            model(
                next_token,
                kv_cache=kv_cache,
                **_model_kwargs_for_generated_step(
                    model_kwargs,
                    next_token,
                    start=int(tokens.shape[1]) - 1,
                    key_length=int(tokens.shape[1]),
                ),
            ),
            next_token,
        )

    return tokens


def build_prompt_cache(
    model: Any,
    prompt_ids: mx.array,
    *,
    model_kwargs: ModelKwargs | None = None,
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
) -> PromptCacheEntry:
    """Prefill and package a reusable contiguous-KV prompt prefix."""

    if len(prompt_ids.shape) != 2:
        raise ValueError("prompt_ids must have shape (batch, sequence)")
    if int(prompt_ids.shape[1]) <= 0:
        raise ValueError("prompt_ids must contain at least one token")

    max_seq_length = _model_max_seq_length(model)
    if max_seq_length is not None and prompt_ids.shape[1] > max_seq_length:
        raise ValueError("prompt_ids already exceed model.config.max_seq_length")
    _require_prompt_cache_safe_model(model)

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
    if kv_cache_position(kv_cache) != 0:
        raise RuntimeError("prompt cache build requires an empty contiguous KV cache")

    next_logits = next_token_logits(
        model(
            prompt_ids,
            kv_cache=kv_cache,
            **_model_kwargs_for_prefix(model_kwargs, prompt_ids),
        ),
        prompt_ids,
    )
    return PromptCacheEntry(
        prompt_ids=mx.array(prompt_ids),
        cache=kv_cache,
        next_logits=mx.array(next_logits),
    )


def generate_tokens_with_prompt_cache(
    model: Any,
    prompt_ids: mx.array,
    *,
    prompt_cache: PromptCacheEntry,
    max_new_tokens: int,
    model_kwargs: ModelKwargs | None = None,
    eos_token_id: int | None = None,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = 1.0,
    rng_key: Any | None = None,
    logits_processors: Sequence[LogitsProcessor] | None = None,
) -> mx.array:
    """Generate by cloning a reusable contiguous-KV prefix cache.

    The cache entry must match the start of ``prompt_ids`` exactly. Any suffix
    in ``prompt_ids`` is decoded through the cloned cache before new tokens are
    sampled. Paged attention, quantized attention, and safety for SSM or
    sliding-window routes are intentionally outside this narrow Stream I slice.
    """

    if len(prompt_ids.shape) != 2:
        raise ValueError("prompt_ids must have shape (batch, sequence)")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    _validate_prompt_cache_prefix(prompt_ids, prompt_cache)
    if max_new_tokens == 0:
        return prompt_ids

    max_seq_length = _model_max_seq_length(model)
    if max_seq_length is not None and prompt_ids.shape[1] > max_seq_length:
        raise ValueError("prompt_ids already exceed model.config.max_seq_length")
    _require_prompt_cache_safe_model(model)

    tokens = prompt_ids
    kv_cache = clone_contiguous_kv_cache(prompt_cache.cache)
    prefix_length = int(prompt_cache.prompt_ids.shape[1])
    suffix = prompt_ids[:, prefix_length:]
    if suffix.shape[1] == 0:
        step_logits = prompt_cache.next_logits
    else:
        step_logits = next_token_logits(
            model(
                suffix,
                kv_cache=kv_cache,
                **_model_kwargs_for_slice(
                    model_kwargs,
                    start=prefix_length,
                    tokens=suffix,
                ),
            ),
            suffix,
        )

    key = rng_key
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
            tokens=tokens,
            logits_processors=logits_processors,
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

        step_logits = next_token_logits(
            model(
                next_token,
                kv_cache=kv_cache,
                **_model_kwargs_for_generated_step(
                    model_kwargs,
                    next_token,
                    start=int(tokens.shape[1]) - 1,
                    key_length=int(tokens.shape[1]),
                ),
            ),
            next_token,
        )

    return tokens


def stream_generate_tokens(
    model: Any,
    prompt_ids: mx.array,
    *,
    max_new_tokens: int,
    model_kwargs: ModelKwargs | None = None,
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
    logits_processors: Sequence[LogitsProcessor] | None = None,
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
            model_kwargs=model_kwargs,
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
            logits_processors=logits_processors,
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

        step_logits = next_token_logits(
            model(tokens, **_model_kwargs_for_prefix(model_kwargs, tokens)),
            tokens,
        )
        key, step_key = _split_generation_key(key)
        next_token = sample_next_token(
            step_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            rng_key=step_key,
            tokens=tokens,
            logits_processors=logits_processors,
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


def _standard_generation_logits(model_output: Any, tokens: mx.array) -> mx.array:
    if isinstance(model_output, tuple | list):
        if (
            len(model_output) == 2
            and isinstance(model_output[0], mx.array)
            and model_output[1] is None
        ):
            model_output = model_output[0]
        else:
            raise ValueError(
                "MTP/draft tuple outputs are not supported by standard next-token "
                "inference; only the DenseCppLM (logits, None) inference contract "
                "is accepted"
            )
    if isinstance(model_output, dict):
        raise ValueError(
            "structured model outputs are not supported by standard next-token "
            "inference; pass plain logits"
        )
    if not isinstance(model_output, mx.array):
        raise TypeError("model output must be an mlx.core.array of logits")

    _validate_logits_shape(model_output, tokens)
    return model_output


def _model_kwargs_for_prefix(
    model_kwargs: ModelKwargs | None,
    tokens: mx.array,
) -> dict[str, Any]:
    if not model_kwargs:
        return {}
    batch_size, sequence_length = _batch_sequence(tokens)
    repository_window = _repository_graph_model_kwargs_for_window(
        model_kwargs,
        batch_size=batch_size,
        total_token_count=sequence_length,
        window_start=0,
        window_end=sequence_length,
    )
    if repository_window is not None:
        return repository_window
    return {
        name: _align_model_kwarg_for_prefix(name, value, tokens)
        for name, value in model_kwargs.items()
    }


def _model_kwargs_for_slice(
    model_kwargs: ModelKwargs | None,
    *,
    start: int,
    tokens: mx.array,
) -> dict[str, Any]:
    if not model_kwargs:
        return {}
    batch_size, sequence_length = _batch_sequence(tokens)
    repository_window = _repository_graph_model_kwargs_for_window(
        model_kwargs,
        batch_size=batch_size,
        total_token_count=start + sequence_length,
        window_start=start,
        window_end=start + sequence_length,
    )
    if repository_window is not None:
        return repository_window
    out: dict[str, Any] = {}
    for name, value in model_kwargs.items():
        _validate_model_kwarg_tensor(name, value)
        if name == "graph_batch":
            if start != 0:
                raise ValueError(
                    "static graph_batch cannot align an offset cache slice"
                )
            out[name] = _static_graph_batch_for_prefix(
                value,
                sequence_length=sequence_length,
            )
            continue
        if name in _GRAPH_BIAS_MODEL_KWARGS:
            out[name] = _graph_bias_for_window(
                name,
                value,
                batch_size=batch_size,
                query_start=start,
                query_length=sequence_length,
                key_length=start + sequence_length,
            )
            continue
        if not _sequence_aligned_model_kwarg(name, value, batch_size):
            out[name] = value
            continue
        available = max(0, int(value.shape[1]) - start)
        if available >= sequence_length:
            out[name] = value[:, start : start + sequence_length, ...]
            continue
        chunks = []
        if available > 0:
            chunks.append(value[:, start:, ...])
        chunks.append(
            _generated_model_kwarg_tail(
                name,
                value,
                batch_size=batch_size,
                sequence_length=sequence_length - available,
            )
        )
        out[name] = mx.concatenate(chunks, axis=1) if len(chunks) > 1 else chunks[0]
    return out


def _model_kwargs_for_generated_step(
    model_kwargs: ModelKwargs | None,
    tokens: mx.array,
    *,
    start: int | None = None,
    key_length: int | None = None,
) -> dict[str, Any]:
    if not model_kwargs:
        return {}
    batch_size, sequence_length = _batch_sequence(tokens)
    if "graph_batch" in model_kwargs and callable(
        getattr(model_kwargs["graph_batch"], "prompt_graph_window", None)
    ):
        if start is None or key_length is None:
            raise ValueError(
                "repository graph generated-step alignment requires absolute "
                "query start and key length"
            )
        repository_window = _repository_graph_model_kwargs_for_window(
            model_kwargs,
            batch_size=batch_size,
            total_token_count=key_length,
            window_start=start,
            window_end=key_length,
        )
        if repository_window is None:
            raise RuntimeError(
                "repository graph batch did not produce a generation window"
            )
        return repository_window
    out: dict[str, Any] = {}
    for name, value in model_kwargs.items():
        _validate_model_kwarg_tensor(name, value)
        if name == "graph_batch":
            raise ValueError(
                "static graph_batch generated-step alignment requires a "
                "PromptGraphArtifact-backed graph batch"
            )
        if name in _GRAPH_BIAS_MODEL_KWARGS:
            if start is None or key_length is None:
                raise ValueError(
                    "graph bias generated-step alignment requires absolute "
                    "query start and key length"
                )
            out[name] = _graph_bias_for_window(
                name,
                value,
                batch_size=batch_size,
                query_start=start,
                query_length=sequence_length,
                key_length=key_length,
            )
            continue
        if _sequence_aligned_model_kwarg(name, value, batch_size):
            out[name] = _generated_model_kwarg_tail(
                name,
                value,
                batch_size=batch_size,
                sequence_length=sequence_length,
            )
        else:
            out[name] = value
    return out


def _align_model_kwarg_for_prefix(
    name: str,
    value: Any,
    tokens: mx.array,
) -> Any:
    _validate_model_kwarg_tensor(name, value)
    batch_size, sequence_length = _batch_sequence(tokens)
    if name == "graph_batch":
        return _static_graph_batch_for_prefix(value, sequence_length=sequence_length)
    if name in _GRAPH_BIAS_MODEL_KWARGS:
        return _graph_bias_for_window(
            name,
            value,
            batch_size=batch_size,
            query_start=0,
            query_length=sequence_length,
            key_length=sequence_length,
        )
    if not _sequence_aligned_model_kwarg(name, value, batch_size):
        return value

    current_length = int(value.shape[1])
    if current_length == sequence_length:
        return value
    if current_length > sequence_length:
        return value[:, :sequence_length, ...]
    tail = _generated_model_kwarg_tail(
        name,
        value,
        batch_size=batch_size,
        sequence_length=sequence_length - current_length,
    )
    return mx.concatenate([value, tail], axis=1)


def _validate_model_kwarg_tensor(name: str, value: Any) -> None:
    if name == "graph_batch":
        if not isinstance(value, GraphBatch):
            raise TypeError(
                "model_kwargs['graph_batch'] must be a GraphBatch, got "
                f"{type(value).__name__}"
            )
        return
    if not isinstance(value, mx.array):
        raise TypeError(f"model_kwargs[{name!r}] must be an mlx.core.array")


def _repository_graph_model_kwargs_for_window(
    model_kwargs: ModelKwargs,
    *,
    batch_size: int,
    total_token_count: int,
    window_start: int,
    window_end: int,
) -> dict[str, Any] | None:
    graph_batch = model_kwargs.get("graph_batch")
    window_builder = getattr(graph_batch, "prompt_graph_window", None)
    if window_builder is None:
        return None
    if not isinstance(graph_batch, GraphBatch) or not callable(window_builder):
        raise TypeError(
            "repository graph_batch must be a GraphBatch with prompt_graph_window"
        )
    if batch_size != 1:
        raise ValueError(
            "repository prompt graph generation currently requires batch=1"
        )

    window_graph, graph_inputs = window_builder(
        total_token_count=total_token_count,
        window_start=window_start,
        window_end=window_end,
    )
    if not isinstance(window_graph, GraphBatch):
        raise TypeError("prompt_graph_window must return a GraphBatch")
    side_channels = getattr(graph_inputs, "side_channels", None)
    token_count = getattr(graph_inputs, "token_count", None)
    if not isinstance(side_channels, Mapping) or int(token_count) != (
        window_end - window_start
    ):
        raise ValueError(
            "prompt_graph_window returned malformed token-aligned side channels"
        )

    out: dict[str, Any] = {}
    for name, value in model_kwargs.items():
        if name == "graph_batch":
            out[name] = window_graph
            continue
        if name in _GRAPH_BIAS_MODEL_KWARGS:
            raise ValueError(
                "repository graph_batch cannot be combined with fixed graph biases"
            )
        _validate_model_kwarg_tensor(name, value)
        if name in side_channels:
            out[name] = mx.array([side_channels[name]], dtype=value.dtype)
            continue
        out[name] = _model_kwarg_for_absolute_window(
            name,
            value,
            batch_size=batch_size,
            window_start=window_start,
            window_end=window_end,
        )
    return out


def _model_kwarg_for_absolute_window(
    name: str,
    value: mx.array,
    *,
    batch_size: int,
    window_start: int,
    window_end: int,
) -> mx.array:
    if not _sequence_aligned_model_kwarg(name, value, batch_size):
        return value
    source_length = int(value.shape[1])
    if source_length >= window_end:
        return value[:, window_start:window_end, ...]

    available = max(0, source_length - window_start)
    chunks: list[mx.array] = []
    if available > 0:
        chunks.append(value[:, window_start:, ...])
    chunks.append(
        _generated_model_kwarg_tail(
            name,
            value,
            batch_size=batch_size,
            sequence_length=(window_end - window_start) - available,
        )
    )
    return mx.concatenate(chunks, axis=1) if len(chunks) > 1 else chunks[0]


def _static_graph_batch_for_prefix(
    graph_batch: GraphBatch,
    *,
    sequence_length: int,
) -> GraphBatch:
    source_length = max(
        (
            int(mx.max(chunk_ends).item())
            for chunk_ends in graph_batch.chunk_ends
            if int(chunk_ends.shape[0]) > 0
        ),
        default=0,
    )
    if source_length <= 0:
        raise ValueError("graph_batch has no source token spans")
    if sequence_length > source_length:
        raise ValueError(
            "static graph_batch cannot extend beyond its source sequence; "
            "use a PromptGraphArtifact-backed repository graph batch"
        )
    return graph_batch.input_aligned(
        source_sequence_length=source_length,
        input_sequence_length=sequence_length,
    )


def _graph_bias_for_window(
    name: str,
    value: mx.array,
    *,
    batch_size: int,
    query_start: int,
    query_length: int,
    key_length: int,
) -> mx.array:
    """Align a square prompt graph bias to a query/key attention window."""

    if len(value.shape) != 3 or int(value.shape[1]) != int(value.shape[2]):
        raise ValueError(
            f"model_kwargs[{name!r}] must have shape (B,S,S), got "
            f"{tuple(value.shape)}"
        )
    if int(value.shape[0]) not in (1, batch_size):
        raise ValueError(
            f"model_kwargs[{name!r}] batch dimension must be 1 or "
            f"{batch_size}, got {value.shape[0]}"
        )
    if query_start < 0:
        raise ValueError("graph bias query_start must be non-negative")
    if query_length <= 0 or key_length <= 0:
        raise ValueError(
            "graph bias query and key lengths must be positive, got "
            f"{query_length}/{key_length}"
        )
    if key_length < query_start + query_length:
        raise ValueError(
            "graph bias key length must cover the query window, got "
            f"key_length={key_length} query_end={query_start + query_length}"
        )

    source_length = int(value.shape[1])
    source_query_end = min(source_length, query_start + query_length)
    source_key_end = min(source_length, key_length)
    if query_start < source_query_end and source_key_end > 0:
        window = value[:, query_start:source_query_end, :source_key_end]
    else:
        window = value[:, :0, :0]
    return _pad_graph_bias(
        window,
        query_length=query_length,
        key_length=key_length,
    )


def _pad_graph_bias(
    value: mx.array,
    *,
    query_length: int,
    key_length: int,
) -> mx.array:
    current_query = int(value.shape[1])
    current_key = int(value.shape[2])
    if current_query > query_length or current_key > key_length:
        value = value[:, :query_length, :key_length]
        current_query = int(value.shape[1])
        current_key = int(value.shape[2])
    if current_query == query_length and current_key == key_length:
        return value
    return mx.pad(
        value,
        [
            (0, 0),
            (0, query_length - current_query),
            (0, key_length - current_key),
        ],
    )


def _sequence_aligned_model_kwarg(
    name: str,
    value: mx.array,
    batch_size: int,
) -> bool:
    if name in _NON_SEQUENCE_ALIGNED_MODEL_KWARGS:
        return False
    shape = value.shape
    if not shape or int(shape[0]) != batch_size:
        return False
    if name == "platform_ids":
        return len(shape) == 3
    return name in _SEQUENCE_ALIGNED_MODEL_KWARGS and len(shape) >= 2


def _generated_model_kwarg_tail(
    name: str,
    value: mx.array,
    *,
    batch_size: int,
    sequence_length: int,
) -> mx.array:
    if sequence_length <= 0:
        return value[:, :0, ...]
    tail_shape = (batch_size, sequence_length, *tuple(value.shape[2:]))
    if name in _REPEAT_GENERATED_MODEL_KWARGS and int(value.shape[1]) > 0:
        return mx.broadcast_to(value[:, -1:, ...], tail_shape)
    return mx.zeros(tail_shape, dtype=value.dtype)


def _batch_sequence(tokens: mx.array) -> tuple[int, int]:
    if len(tokens.shape) != 2:
        raise ValueError("tokens must have shape (batch, sequence)")
    return int(tokens.shape[0]), int(tokens.shape[1])


def _validate_prompt_cache_prefix(
    prompt_ids: mx.array,
    prompt_cache: PromptCacheEntry,
) -> None:
    if not isinstance(prompt_cache, PromptCacheEntry):
        raise TypeError("prompt_cache must be a PromptCacheEntry")
    if prompt_ids.shape[0] != prompt_cache.prompt_ids.shape[0]:
        raise ValueError("prompt_ids batch size must match prompt_cache")
    prefix_length = int(prompt_cache.prompt_ids.shape[1])
    if int(prompt_ids.shape[1]) < prefix_length:
        raise ValueError("prompt_ids must include the full prompt_cache prefix")
    prefix_matches = cast(mx.array, prompt_ids[:, :prefix_length] == prompt_cache.prompt_ids)
    if not bool(mx.all(prefix_matches)):
        raise ValueError("prompt_ids must start with prompt_cache.prompt_ids")


def _require_prompt_cache_safe_model(model: Any) -> None:
    route_roles = getattr(model, "route_roles", None)
    if route_roles is None:
        return
    roles = tuple(str(role) for role in route_roles)
    unsafe = tuple(role for role in roles if role in _PROMPT_CACHE_UNSAFE_ROUTE_ROLES)
    if unsafe:
        joined = ", ".join(unsafe)
        raise ValueError(
            "prompt cache is only validated for attention/stateless routes; "
            f"route roles requiring their own state cache are present: {joined}"
        )


def _propose_speculative_draft_window(
    draft_model: Any,
    tokens: mx.array,
    *,
    draft_window: int,
    model_kwargs: ModelKwargs | None,
    eos_token_id: int | None,
    temperature: float,
    rng_key: Any | None,
    max_seq_length: int | None,
) -> tuple[mx.array, mx.array, Any | None]:
    draft_tokens: list[mx.array] = []
    draft_logits: list[mx.array] = []
    draft_prefix = tokens
    key = rng_key

    for _ in range(draft_window):
        if max_seq_length is not None and draft_prefix.shape[1] >= max_seq_length:
            raise ValueError("draft generation would exceed model.config.max_seq_length")
        step_logits = next_token_logits(
            draft_model(
                draft_prefix,
                **_model_kwargs_for_prefix(model_kwargs, draft_prefix),
            ),
            draft_prefix,
        )
        key, step_key = _split_generation_key(key)
        next_token = sample_next_token(
            step_logits,
            temperature=temperature,
            rng_key=step_key,
        ).astype(tokens.dtype)
        draft_logits.append(step_logits[0])
        draft_tokens.append(next_token[:, 0])
        draft_prefix = mx.concatenate([draft_prefix, next_token], axis=1)
        if eos_token_id is not None and _all_rows_match_token(next_token, eos_token_id):
            break

    return (
        mx.concatenate(draft_tokens, axis=0).astype(tokens.dtype),
        mx.stack(draft_logits, axis=0),
        key,
    )


def _speculative_append_tokens(
    accepted: mx.array,
    next_token: mx.array,
    *,
    remaining: int,
    eos_token_id: int | None,
) -> tuple[mx.array, bool]:
    proposed = mx.concatenate([accepted, next_token], axis=0)[:remaining]
    if eos_token_id is None:
        return proposed, False

    for idx in range(int(proposed.shape[0])):
        if int(proposed[idx].item()) == eos_token_id:
            return proposed[: idx + 1], True
    return proposed, False


def _resolve_mtp_draft_head(model: Any) -> Any:
    mtp_head = getattr(model, "mtp_head", None)
    if mtp_head is None:
        raise ValueError("model.mtp_head is required for MTP self-speculative generation")
    if not callable(getattr(model, "decoder_hidden_states", None)):
        raise TypeError("MTP self-speculative generation requires model.decoder_hidden_states")
    return mtp_head


def _mtp_head_trained_depth(mtp_head: Any) -> int:
    config = getattr(mtp_head, "config", None)
    depth = getattr(config, "depth", None)
    if depth is None:
        raise ValueError("model.mtp_head.config.depth is required")
    depth_int = int(depth)
    if depth_int <= 0:
        raise ValueError("model.mtp_head.config.depth must be positive")
    return depth_int


def _propose_mtp_self_speculative_window(
    model: Any,
    mtp_head: Any,
    tokens: mx.array,
    *,
    draft_window: int,
    model_kwargs: ModelKwargs | None,
    eos_token_id: int | None,
    temperature: float,
    rng_key: Any | None,
) -> tuple[mx.array, mx.array, Any | None]:
    hidden_states = model.decoder_hidden_states(
        tokens,
        **_model_kwargs_for_prefix(model_kwargs, tokens),
    )
    if not isinstance(hidden_states, mx.array) or hidden_states.ndim != 3:
        raise TypeError("model.decoder_hidden_states must return (batch, sequence, hidden)")
    if hidden_states.shape[:2] != tokens.shape:
        raise ValueError("decoder hidden states prefix shape must match tokens")

    last_hidden = hidden_states[:, -1:, :]
    last_token_ids = tokens[:, -1:]
    draft_method = getattr(mtp_head, "draft", None)
    if callable(draft_method):
        draft_callable = cast(
            Callable[..., tuple[mx.array, mx.array, Any | None]],
            draft_method,
        )
        draft_rows, draft_logit_rows, key = draft_callable(
            last_hidden,
            last_token_ids,
            num_draft_tokens=draft_window,
            temperature=temperature,
            rng_key=rng_key,
        )
    else:
        draft_rows, draft_logit_rows, key = _draft_with_minimal_mtp_head(
            mtp_head,
            last_hidden,
            last_token_ids,
            num_draft_tokens=draft_window,
            temperature=temperature,
            rng_key=rng_key,
        )
    _validate_mtp_draft_rows(draft_rows, draft_logit_rows, draft_window=draft_window)

    draft_tokens = draft_rows[0].astype(tokens.dtype)
    draft_logits = draft_logit_rows[0]
    if eos_token_id is not None:
        for idx in range(int(draft_tokens.shape[0])):
            if int(draft_tokens[idx].item()) == eos_token_id:
                return draft_tokens[: idx + 1], draft_logits[: idx + 1], key
    return draft_tokens, draft_logits, key


def _draft_with_minimal_mtp_head(
    mtp_head: Any,
    last_hidden: mx.array,
    last_token_ids: mx.array,
    *,
    num_draft_tokens: int,
    temperature: float,
    rng_key: Any | None,
) -> tuple[mx.array, mx.array, Any | None]:
    required = (
        "token_embedding",
        "hidden_norm",
        "embedding_norm",
        "proj",
        "shared_block",
        "output_norm",
        "lm_head",
    )
    missing = [name for name in required if not callable(getattr(mtp_head, name, None))]
    if missing:
        raise TypeError(f"model.mtp_head is missing draft components: {', '.join(missing)}")

    h = last_hidden
    token_ids = last_token_ids
    draft_tokens: list[mx.array] = []
    draft_logits: list[mx.array] = []
    key = rng_key
    for _ in range(num_draft_tokens):
        token_emb = mtp_head.token_embedding(token_ids)
        h_mtp = mtp_head.proj(
            mx.concatenate(
                [mtp_head.hidden_norm(h), mtp_head.embedding_norm(token_emb)],
                axis=-1,
            )
        )
        h = mtp_head.output_norm(mtp_head.shared_block(h_mtp))
        step_logits = mtp_head.lm_head(h)[:, -1, :]
        key, step_key = _split_generation_key(key)
        next_token = sample_next_token(
            step_logits,
            temperature=temperature,
            rng_key=step_key,
        ).astype(last_token_ids.dtype)
        draft_logits.append(step_logits)
        draft_tokens.append(next_token)
        token_ids = next_token

    return mx.concatenate(draft_tokens, axis=1), mx.stack(draft_logits, axis=1), key


def _validate_mtp_draft_rows(
    draft_tokens: mx.array,
    draft_logits: mx.array,
    *,
    draft_window: int,
) -> None:
    if not isinstance(draft_tokens, mx.array):
        raise TypeError("MTP draft tokens must be an mlx.core.array")
    if not isinstance(draft_logits, mx.array):
        raise TypeError("MTP draft logits must be an mlx.core.array")
    if draft_tokens.shape != (1, draft_window):
        raise ValueError("MTP draft tokens must have shape (1, draft_window)")
    if len(draft_logits.shape) != 3 or draft_logits.shape[:2] != (1, draft_window):
        raise ValueError("MTP draft logits must have shape (1, draft_window, vocab)")


def _validate_prefix_fits_max_length(tokens: mx.array, max_seq_length: int | None) -> None:
    if max_seq_length is not None and tokens.shape[1] > max_seq_length:
        raise ValueError("tokens exceed model.config.max_seq_length")


def _available_generation_slots(tokens: mx.array, max_seq_length: int | None) -> int | None:
    if max_seq_length is None:
        return None
    slots = max_seq_length - int(tokens.shape[1])
    if slots <= 0:
        raise ValueError("generation would exceed model.config.max_seq_length")
    return slots


def _stream_generate_tokens_with_kv_cache(
    model: Any,
    prompt_ids: mx.array,
    *,
    max_new_tokens: int,
    model_kwargs: ModelKwargs | None,
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
    logits_processors: Sequence[LogitsProcessor] | None,
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
    step_logits = next_token_logits(
        model(
            tokens,
            kv_cache=kv_cache,
            **_model_kwargs_for_prefix(model_kwargs, tokens),
        ),
        tokens,
    )
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
            tokens=tokens,
            logits_processors=logits_processors,
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
        step_logits = next_token_logits(
            model(
                next_token,
                kv_cache=kv_cache,
                **_model_kwargs_for_generated_step(
                    model_kwargs,
                    next_token,
                    start=int(tokens.shape[1]) - 1,
                    key_length=int(tokens.shape[1]),
                ),
            ),
            next_token,
        )


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

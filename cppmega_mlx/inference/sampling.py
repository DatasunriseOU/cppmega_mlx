"""Sampling helpers ported from nanochat generation semantics."""

from __future__ import annotations

from typing import Any, cast

import mlx.core as mx


def sample_next_token(
    logits: mx.array,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = 1.0,
    rng_key: Any | None = None,
) -> mx.array:
    """Sample one next token from ``(batch, vocab)`` logits and return ``(batch, 1)``.

    This mirrors nanochat's Torch helper for the Mac-local MLX path:
    ``temperature=0`` is greedy, positive ``top_k`` narrows candidates, and
    ``top_p`` applies nucleus filtering after optional top-k.
    """
    if len(logits.shape) != 2:
        raise ValueError("logits must have shape (batch, vocab)")
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative")
    if top_p is None:
        top_p = 1.0
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")

    if temperature == 0.0:
        return mx.argmax(logits, axis=-1, keepdims=True)

    scaled_logits = logits / temperature
    batch_size, vocab_size = scaled_logits.shape
    candidate_idx = mx.broadcast_to(
        mx.arange(vocab_size, dtype=mx.int32)[None, :],
        (batch_size, vocab_size),
    )
    candidate_logits = scaled_logits

    if top_k is not None and top_k > 0:
        k = min(top_k, vocab_size)
        candidate_idx = mx.argpartition(-scaled_logits, kth=k - 1, axis=-1)[
            ..., :k
        ].astype(mx.int32)
        candidate_logits = mx.take_along_axis(scaled_logits, candidate_idx, axis=-1)

    if top_p < 1.0:
        sorted_order = mx.argsort(-candidate_logits, axis=-1).astype(mx.int32)
        sorted_logits = mx.take_along_axis(candidate_logits, sorted_order, axis=-1)
        sorted_probs = mx.softmax(sorted_logits, axis=-1)
        keep_mask = mx.cumsum(sorted_probs, axis=-1) <= top_p
        first_token_positions = cast(
            mx.array,
            mx.arange(sorted_logits.shape[-1], dtype=mx.int32) == 0,
        )
        first_token_mask = mx.broadcast_to(first_token_positions[None, :], sorted_logits.shape)
        keep_mask = keep_mask | first_token_mask
        neg_inf = mx.full(sorted_logits.shape, float("-inf"), dtype=sorted_logits.dtype)
        filtered_logits = mx.where(keep_mask, sorted_logits, neg_inf)
        sampled_sorted_pos = mx.random.categorical(
            filtered_logits,
            axis=-1,
            num_samples=1,
            key=rng_key,
        ).astype(mx.int32)
        sampled_candidate_pos = mx.take_along_axis(
            sorted_order,
            sampled_sorted_pos,
            axis=-1,
        )
    else:
        sampled_candidate_pos = mx.random.categorical(
            candidate_logits,
            axis=-1,
            num_samples=1,
            key=rng_key,
        ).astype(mx.int32)

    return mx.take_along_axis(candidate_idx, sampled_candidate_pos, axis=-1)

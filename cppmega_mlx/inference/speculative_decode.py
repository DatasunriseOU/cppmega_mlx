"""Acceptance sampling helpers for MLX speculative decoding.

The functions here mirror nanochat's pure tensor acceptance routines while
staying local to the MLX inference path. They do not depend on KV-cache or
runtime engine code.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from cppmega_mlx.inference.sampling import sample_next_token


def _validate_unbatched_inputs(
    draft_logits: mx.array,
    target_logits: mx.array,
    draft_tokens: mx.array,
) -> int:
    if len(draft_logits.shape) != 2:
        raise ValueError("draft_logits must have shape (K, vocab)")
    if len(target_logits.shape) != 2:
        raise ValueError("target_logits must have shape (K + 1, vocab)")
    if len(draft_tokens.shape) != 1:
        raise ValueError("draft_tokens must have shape (K,)")

    k = draft_tokens.shape[0]
    if draft_logits.shape[0] != k:
        raise ValueError(
            f"draft_logits has {draft_logits.shape[0]} positions but expected {k}"
        )
    if target_logits.shape[0] != k + 1:
        raise ValueError(
            f"target_logits has {target_logits.shape[0]} positions but expected {k + 1}"
        )
    if draft_logits.shape[1] != target_logits.shape[1]:
        raise ValueError("draft_logits and target_logits must use the same vocab size")
    return k


def _validate_batched_inputs(
    draft_logits: mx.array,
    target_logits: mx.array,
    draft_tokens: mx.array,
) -> tuple[int, int]:
    if len(draft_logits.shape) != 3:
        raise ValueError("draft_logits must have shape (batch, K, vocab)")
    if len(target_logits.shape) != 3:
        raise ValueError("target_logits must have shape (batch, K + 1, vocab)")
    if len(draft_tokens.shape) != 2:
        raise ValueError("draft_tokens must have shape (batch, K)")

    batch_size, k = draft_tokens.shape
    if draft_logits.shape[:2] != (batch_size, k):
        raise ValueError(
            "draft_logits leading dimensions must match draft_tokens shape"
        )
    if target_logits.shape[:2] != (batch_size, k + 1):
        raise ValueError(
            "target_logits leading dimensions must be (batch, K + 1)"
        )
    if draft_logits.shape[2] != target_logits.shape[2]:
        raise ValueError("draft_logits and target_logits must use the same vocab size")
    return batch_size, k


def _softmax_with_temperature(logits: mx.array, temperature: float) -> mx.array:
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative")
    if temperature > 0.0:
        return mx.softmax(logits / temperature, axis=-1)
    return mx.softmax(logits, axis=-1)


def _sample_from_probs(probs: mx.array, rng_key: Any | None) -> mx.array:
    logits = mx.where(
        probs > 0.0,
        mx.log(probs),
        mx.full(probs.shape, float("-inf"), dtype=probs.dtype),
    )
    return sample_next_token(logits[None, :], rng_key=rng_key).reshape((1,))


def _split_keys(rng_key: Any | None, count: int) -> list[Any | None]:
    if rng_key is None:
        return [None] * count
    return [key for key in mx.random.split(rng_key, count)]


def speculative_acceptance(
    draft_logits: mx.array,
    target_logits: mx.array,
    draft_tokens: mx.array,
    temperature: float = 1.0,
    *,
    rng_key: Any | None = None,
) -> tuple[mx.array, int, mx.array]:
    """Standard Leviathan-style speculative decoding acceptance.

    For each draft token, accept with probability
    ``min(1, p_target(token) / p_draft(token))`` until the first rejection. At
    rejection, sample from the normalized positive residual
    ``max(0, p_target - p_draft)``. If all draft tokens are accepted, sample the
    next token from the target distribution at position ``K``.
    """
    k = _validate_unbatched_inputs(draft_logits, target_logits, draft_tokens)
    draft_probs = _softmax_with_temperature(draft_logits, temperature)
    target_probs = _softmax_with_temperature(target_logits, temperature)
    accept_key, sample_key = _split_keys(rng_key, 2)
    random_acceptance = mx.random.uniform(shape=(k,), key=accept_key)

    n_accepted = 0
    for i in range(k):
        token = int(draft_tokens[i].item())
        p_draft = float(draft_probs[i, token].item())
        if p_draft == 0.0:
            break

        p_target = float(target_probs[i, token].item())
        acceptance_prob = min(1.0, p_target / p_draft)
        if float(random_acceptance[i].item()) < acceptance_prob:
            n_accepted += 1
        else:
            break

    if n_accepted == k:
        return draft_tokens[:k], n_accepted, _sample_from_probs(target_probs[k], sample_key)

    residual = mx.maximum(
        target_probs[n_accepted] - draft_probs[n_accepted],
        mx.array(0.0, dtype=target_probs.dtype),
    )
    residual_sum = float(mx.sum(residual).item())
    if residual_sum > 0.0:
        next_token = _sample_from_probs(residual / residual_sum, sample_key)
    else:
        next_token = _sample_from_probs(target_probs[n_accepted], sample_key)
    return draft_tokens[:n_accepted], n_accepted, next_token


def typical_acceptance(
    draft_logits: mx.array,
    target_logits: mx.array,
    draft_tokens: mx.array,
    threshold: float = 0.5,
    *,
    rng_key: Any | None = None,
) -> tuple[mx.array, int, mx.array]:
    """Simplified typical acceptance sampling.

    Accepts each draft token while ``p_target(token) / p_draft(token)`` is at
    least ``threshold``. At the first rejection, or after all draft tokens are
    accepted, the next token is sampled directly from the target distribution.
    """
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")

    k = _validate_unbatched_inputs(draft_logits, target_logits, draft_tokens)
    draft_probs = mx.softmax(draft_logits, axis=-1)
    target_probs = mx.softmax(target_logits, axis=-1)

    n_accepted = 0
    for i in range(k):
        token = int(draft_tokens[i].item())
        p_draft = float(draft_probs[i, token].item())
        if p_draft <= 0.0:
            break
        p_target = float(target_probs[i, token].item())
        if p_target / p_draft >= threshold:
            n_accepted += 1
        else:
            break

    sample_pos = k if n_accepted == k else n_accepted
    return (
        draft_tokens[:n_accepted],
        n_accepted,
        _sample_from_probs(target_probs[sample_pos], rng_key),
    )


def speculative_acceptance_batch(
    draft_logits: mx.array,
    target_logits: mx.array,
    draft_tokens: mx.array,
    temperature: float = 1.0,
    *,
    rng_key: Any | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    """Batched wrapper for standard speculative acceptance.

    Each batch row is processed independently. Accepted-token rows are padded
    with ``-1`` beyond each row's accepted prefix.
    """
    batch_size, k = _validate_batched_inputs(draft_logits, target_logits, draft_tokens)
    row_keys = _split_keys(rng_key, batch_size)
    accepted_rows: list[list[int]] = []
    n_accepted_rows: list[int] = []
    next_tokens: list[int] = []

    for b in range(batch_size):
        accepted, n_accepted, next_token = speculative_acceptance(
            draft_logits[b],
            target_logits[b],
            draft_tokens[b],
            temperature=temperature,
            rng_key=row_keys[b],
        )
        row = [-1] * k
        for i in range(n_accepted):
            row[i] = int(accepted[i].item())
        accepted_rows.append(row)
        n_accepted_rows.append(n_accepted)
        next_tokens.append(int(next_token[0].item()))

    return (
        mx.array(accepted_rows, dtype=mx.int32),
        mx.array(n_accepted_rows, dtype=mx.int32),
        mx.array(next_tokens, dtype=mx.int32),
    )


def typical_acceptance_batch(
    draft_logits: mx.array,
    target_logits: mx.array,
    draft_tokens: mx.array,
    threshold: float = 0.5,
    *,
    rng_key: Any | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    """Batched wrapper for typical speculative acceptance."""
    batch_size, k = _validate_batched_inputs(draft_logits, target_logits, draft_tokens)
    row_keys = _split_keys(rng_key, batch_size)
    accepted_rows: list[list[int]] = []
    n_accepted_rows: list[int] = []
    next_tokens: list[int] = []

    for b in range(batch_size):
        accepted, n_accepted, next_token = typical_acceptance(
            draft_logits[b],
            target_logits[b],
            draft_tokens[b],
            threshold=threshold,
            rng_key=row_keys[b],
        )
        row = [-1] * k
        for i in range(n_accepted):
            row[i] = int(accepted[i].item())
        accepted_rows.append(row)
        n_accepted_rows.append(n_accepted)
        next_tokens.append(int(next_token[0].item()))

    return (
        mx.array(accepted_rows, dtype=mx.int32),
        mx.array(n_accepted_rows, dtype=mx.int32),
        mx.array(next_tokens, dtype=mx.int32),
    )


__all__ = [
    "speculative_acceptance",
    "speculative_acceptance_batch",
    "typical_acceptance",
    "typical_acceptance_batch",
]

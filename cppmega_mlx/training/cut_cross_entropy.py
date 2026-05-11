"""MLX-native chunked linear cross-entropy.

Apple's `cut_cross_entropy` (https://github.com/apple/ml-cross-entropy) avoids
materializing the full ``[N, V]`` logits tensor when computing
``cross_entropy(e @ c.T, targets)``. The upstream package ships a CUDA Triton
kernel and a ``torch_compile`` reference path; both target torch tensors and
the Triton kernel rejects MacOS at runtime
(``RuntimeError: CCE does not support MacOS``).

This module is the MLX-native equivalent of the chunked algorithm. We chunk
along the row axis ``N = B * T``: each chunk materializes only a
``[chunk_rows, V]`` logits tile, computes the row-wise log-softmax loss,
forces an :func:`mx.eval`, and frees the tile before moving to the next
chunk. The math is unchanged versus the materialized path:

* forward: ``-(e[i] @ c[t_i]) + logsumexp(e[i] @ c.T)`` per row.
* backward: ``softmax(e[i] @ c.T) - one_hot(t_i, V)`` is the gradient w.r.t.
  the chunk's logits; we contract with ``c`` for ``de`` and accumulate
  ``grad_logits.T @ e[i]`` into ``dc`` slot-by-slot.

Public surface:

* :func:`linear_cross_entropy` -- chunked forward (works with ``mx.grad`` but
  the backward pass keeps every chunk's activations live, so no backward
  memory savings -- this matches Apple's ``torch_compile`` reference).
* :func:`linear_cross_entropy_value_and_grad` -- eager chunked
  forward+backward, returning ``(loss, de, dc)``. Runs outside MLX autograd
  so each chunk's tile is actually freed mid-backward; integrates with
  optimizer steps via :func:`mx.tree_unflatten` style updates.
* :func:`materialized_cross_entropy` -- reference path that builds the full
  ``[N, V]`` logits tensor (parity baseline for tests/benches).

All three accept 2-D ``e`` ``(N, D)`` or 3-D ``e`` ``(B, T, D)``; the
classifier ``c`` is ``(V, D)`` and ``targets`` is ``(N,)`` or ``(B, T)``.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

DEFAULT_CHUNK_ROWS = 256
"""Default rows-per-chunk; chosen so a 65536-vocab fp32 tile stays ~64 MiB."""

_VALID_REDUCTIONS = ("mean", "sum", "none")


def _flatten_inputs(
    e: mx.array, targets: mx.array
) -> tuple[mx.array, mx.array, tuple[int, ...]]:
    if e.ndim < 2:
        raise ValueError(f"e must have rank >= 2, got shape {e.shape}")
    if targets.ndim != e.ndim - 1:
        raise ValueError(
            f"targets rank must be e.ndim - 1; got e.shape={e.shape}, "
            f"targets.shape={targets.shape}"
        )
    flat_e = e.reshape(-1, e.shape[-1])
    flat_t = targets.reshape(-1)
    return flat_e, flat_t, tuple(targets.shape)


def _validate_reduction(reduction: str) -> str:
    if reduction not in _VALID_REDUCTIONS:
        raise ValueError(
            f"reduction must be one of {_VALID_REDUCTIONS}, got {reduction!r}"
        )
    return reduction


def _chunked_forward(
    e: mx.array,
    c: mx.array,
    targets: mx.array,
    *,
    chunk: int,
    reduction: str,
    eval_chunks: bool,
) -> mx.array:
    """Chunked forward pass with per-chunk eval to bound peak memory."""

    n_rows = e.shape[0]
    chunk = max(1, min(int(chunk), n_rows))
    pieces: list[mx.array] = []
    for start in range(0, n_rows, chunk):
        stop = min(start + chunk, n_rows)
        logits = (e[start:stop] @ c.T).astype(mx.float32)
        chunk_loss = nn.losses.cross_entropy(
            logits, targets[start:stop], reduction="none"
        )
        if reduction == "none":
            if eval_chunks:
                mx.eval(chunk_loss)
            pieces.append(chunk_loss)
        else:
            chunk_scalar = chunk_loss.sum().astype(mx.float32)
            if eval_chunks:
                mx.eval(chunk_scalar)
            pieces.append(chunk_scalar)

    if reduction == "none":
        return mx.concatenate(pieces, axis=0)

    total = pieces[0]
    for piece in pieces[1:]:
        total = total + piece
    if reduction == "mean":
        return total / mx.array(float(n_rows), dtype=mx.float32)
    return total


def linear_cross_entropy(
    e: mx.array,
    c: mx.array,
    targets: mx.array,
    *,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    reduction: str = "mean",
    eval_chunks: bool = True,
) -> mx.array:
    """Chunked forward equivalent of ``cross_entropy(e @ c.T, targets)``.

    By default the forward pass evaluates each ``[chunk_rows, V]`` tile in
    isolation, so forward peak memory tracks the chunk size rather than the
    full ``[N, V]`` logits tensor. Set ``eval_chunks=False`` when calling from
    ``mx.compile`` because MLX forbids explicit evaluation inside transformed
    functions. ``mx.grad`` applied to this function still works but keeps
    activations for every chunk live, so no backward memory savings; use
    :func:`linear_cross_entropy_value_and_grad` for chunked backward.
    """

    reduction = _validate_reduction(reduction)
    flat_e, flat_t, target_shape = _flatten_inputs(e, targets)
    loss = _chunked_forward(
        flat_e,
        c,
        flat_t,
        chunk=chunk_rows,
        reduction=reduction,
        eval_chunks=eval_chunks,
    )
    if reduction == "none":
        return loss.reshape(target_shape)
    return loss


def linear_cross_entropy_value_and_grad(
    e: mx.array,
    c: mx.array,
    targets: mx.array,
    *,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> tuple[mx.array, mx.array, mx.array]:
    """Chunked forward + backward, returning ``(loss, de, dc)`` eagerly.

    Unlike :func:`mx.grad` over :func:`linear_cross_entropy`, this function
    runs the loop outside any MLX trace: each chunk evaluates eagerly and the
    previous chunk's logits/probs/one-hot tensors are freed before the next
    chunk allocates. Targets receive no gradient.

    Returns:
        Tuple of ``(loss, de, dc)`` where ``loss`` is a scalar mean CE in
        fp32, ``de`` matches ``e`` in shape and dtype, and ``dc`` matches
        ``c`` in shape and dtype.
    """

    flat_e, flat_t, _ = _flatten_inputs(e, targets)
    n_rows, _ = flat_e.shape
    v = c.shape[0]
    chunk = max(1, min(int(chunk_rows), n_rows))
    cls = mx.arange(v, dtype=flat_t.dtype)
    scale = mx.array(1.0 / float(n_rows), dtype=mx.float32)

    de = mx.zeros_like(flat_e)
    dc = mx.zeros_like(c)
    total = mx.array(0.0, dtype=mx.float32)
    for start in range(0, n_rows, chunk):
        stop = min(start + chunk, n_rows)
        e_chunk = flat_e[start:stop]
        t_chunk = flat_t[start:stop]
        logits = (e_chunk @ c.T).astype(mx.float32)
        lse = mx.logsumexp(logits, axis=-1)
        gathered = mx.take_along_axis(logits, t_chunk[:, None], axis=-1).squeeze(-1)
        total = total + (lse - gathered).sum()

        probs = mx.softmax(logits, axis=-1)
        one_hot = (t_chunk[:, None] == cls[None, :]).astype(probs.dtype)
        grad_logits = ((probs - one_hot) * scale).astype(e.dtype)
        de_chunk = grad_logits @ c
        dc_partial = grad_logits.T @ e_chunk
        dc = dc + dc_partial
        de = mx.slice_update(de, de_chunk, mx.array([start, 0]), axes=(0, 1))
        # Force a per-chunk eval so MLX frees the [chunk, V] tile before we
        # build the next one. This is the load-bearing call: without it, MLX
        # accumulates a graph spanning every chunk and peak memory grows
        # linearly with N/chunk.
        mx.eval(de, dc, total)

    loss = total / mx.array(float(n_rows), dtype=mx.float32)
    de = de.reshape(e.shape)
    return loss, de, dc


def materialized_cross_entropy(
    e: mx.array,
    c: mx.array,
    targets: mx.array,
    *,
    reduction: str = "mean",
) -> mx.array:
    """Reference path that materializes the full ``[N, V]`` logits tensor."""

    reduction = _validate_reduction(reduction)
    flat_e, flat_t, target_shape = _flatten_inputs(e, targets)
    logits = (flat_e @ c.T).astype(mx.float32)
    loss = nn.losses.cross_entropy(logits, flat_t, reduction="none")
    if reduction == "none":
        return loss.reshape(target_shape)
    if reduction == "sum":
        return loss.sum()
    return loss.mean()


__all__ = [
    "DEFAULT_CHUNK_ROWS",
    "linear_cross_entropy",
    "linear_cross_entropy_value_and_grad",
    "materialized_cross_entropy",
]

"""V7-N05: pure-MLX reference for the DSA split-K indexer KL loss.

The fused TileLang kernel lives in
``cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py``; the inner KL
math (``sum_j p_j * (log(p_j+eps) - log(q_j+eps))``) is implemented
inside the Path C DSL there but had no host-side reference, making
parity tests fragile.

This module provides the numerical reference: a pure
``mlx.core`` implementation that takes the same (Q, K, IndexScores,
IndexMask) inputs and returns the per-position KL divergence + the
final scalar loss. It is slower than the kernel but easy to read and
deterministic across devices.

Used by tests/test_dsa_indexer_kl_term.py to lock the kernel against
a known-good baseline.
"""

from __future__ import annotations

import mlx.core as mx

EPS = 1e-9


def reference_indexer_kl_loss(
    Q: mx.array, K: mx.array,
    IndexScores: mx.array,
    *,
    softmax_scale: float,
    loss_coeff: float,
    IndexMask: mx.array | None = None,
    causal: bool = True,
) -> tuple[mx.array, mx.array]:
    """Compute the indexer KL loss without TileLang.

    Args:
        Q: (B, Sq, AH, AD) fp16/bf16/fp32 query.
        K: (B, Sk, AH, AD) fp16/bf16/fp32 key.
        IndexScores: (B, Sq, Sk) head-0 logits (the index head only).
        softmax_scale: 1/sqrt(AD) factor multiplied into the attention
            logits before softmax.
        loss_coeff: scalar multiplier on the mean per-position loss.
        IndexMask: optional (B, Sq, Sk) additive mask (-inf where the
            indexer should skip). Same role as the sparse_loss branch
            in the kernel.
        causal: when True (default), positions j > i are masked out so
            sq attends only to sk in [0, sq].

    Returns:
        (per_position_loss, scalar_loss) — per_position_loss is
        (B, Sq) fp32; scalar_loss is the 0-d fp32 reduction
        ``mean(per_position_loss) * loss_coeff``.
    """
    if Q.shape[0] != K.shape[0]:
        raise ValueError("Q and K must share batch dim")
    if Q.shape[2] != K.shape[2] or Q.shape[3] != K.shape[3]:
        raise ValueError("Q and K must share (AH, AD)")
    B, Sq, AH, AD = (int(Q.shape[0]), int(Q.shape[1]),
                      int(Q.shape[2]), int(Q.shape[3]))
    Sk = int(K.shape[1])

    # Cast to fp32 for the soft path so log/exp don't blow up.
    Qf = Q.astype(mx.float32)
    Kf = K.astype(mx.float32)
    # (B, AH, Sq, Sk) attention logits.
    # einsum: bsh d, bth d → b h s t
    logits = mx.einsum("bshd,bthd->bhst", Qf, Kf) * float(softmax_scale)

    # Causal mask.
    if causal:
        # Build (Sq, Sk) lower-triangular mask: keep j<=i.
        i = mx.arange(Sq).reshape(Sq, 1)
        j = mx.arange(Sk).reshape(1, Sk)
        keep = (j <= i)
        big_neg = mx.full(logits.shape, -1e9, dtype=mx.float32)
        logits = mx.where(keep[None, None, :, :], logits, big_neg)

    # Heads-summed softmax over Sk: average attention across heads
    # (the kernel does the same per-tile soft-max).
    # softmax_attn: (B, Sq, Sk) — average over AH heads.
    head_softmax = mx.softmax(logits, axis=-1)
    softmax_attn = mx.mean(head_softmax, axis=1)  # (B, Sq, Sk)

    # IndexScores softmax (the q in KL(p||q)).
    idx_logits = IndexScores.astype(mx.float32)
    if IndexMask is not None:
        idx_logits = idx_logits + IndexMask.astype(mx.float32)
    if causal:
        idx_logits = mx.where(keep[None, :, :], idx_logits, big_neg[:, 0, :, :])
    softmax_idx = mx.softmax(idx_logits, axis=-1)  # (B, Sq, Sk)

    # Per-position KL: sum_j p_j * (log(p_j+eps) - log(q_j+eps)).
    log_p = mx.log(softmax_attn + EPS)
    log_q = mx.log(softmax_idx + EPS)
    per_pos = mx.sum(softmax_attn * (log_p - log_q), axis=-1)  # (B, Sq)

    scalar = mx.mean(per_pos) * float(loss_coeff)
    return per_pos, scalar


__all__ = ["reference_indexer_kl_loss", "EPS"]

"""Pure-MLX reference for sparse multi-latent-attention.

This module implements the parity oracle for cppmega's sparse-MLA (DeepSeek-V3
style with index gating per query token). It mirrors the math of the TileLang
sparse-MLA forward/backward pair at
``cppmega/megatron/sparse_mla_ops/tilelang_sparse_mla_fwd.py`` and ``..._bwd.py``
but uses only mx.core ops so it stays differentiable through MLX's autograd.

Algorithm sketch (per query position s_i, kv_group g_i, batch b_i):

1. For each query head h in g_i's group, gather KV[b_i, indices[b_i, s_i, g_i, :], g_i, :].
2. Compute Q @ K^T over the gathered indices (shape [topk]).
3. Mask invalid indices (sentinel == -1) to -inf in the score row.
4. Softmax over the topk axis with sm_scale; output = sum(p * V).
5. KV is packed: V uses the leading ``d_v`` channels, the tail dims are extra
   QK channels (the MLA "tail_dim" / RoPE/NoPE split).

The reference does not chunk over topk; for the parity oracle we trade
peak memory for code clarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import mlx.core as mx


_INVALID_INDEX_SENTINEL = -1


@dataclass(frozen=True)
class SparseMLAShapes:
    """Resolved shape descriptors for sparse-MLA tensors."""

    batch: int
    seq_len: int
    seq_len_kv: int
    heads: int
    kv_group: int
    head_kv: int
    d_v: int
    qk_dim: int
    tail_dim: int
    topk: int


def _resolve_shapes(
    q: mx.array,
    kv: mx.array,
    indices: mx.array,
    *,
    d_v: int | None,
) -> SparseMLAShapes:
    if q.ndim != 4:
        raise ValueError(f"q must be 4D [B,S,H,D_qk], got shape {q.shape}")
    if kv.ndim != 4:
        raise ValueError(f"kv must be 4D [B,Skv,G,D_qk], got shape {kv.shape}")
    if indices.ndim != 4:
        raise ValueError(f"indices must be 4D [B,S,G,topk], got shape {indices.shape}")

    batch, seq_len, heads, qk_dim = q.shape
    batch_kv, seq_len_kv, kv_group, kv_dim = kv.shape
    if batch_kv != batch:
        raise ValueError(f"q.batch={batch} != kv.batch={batch_kv}")
    if kv_dim != qk_dim:
        raise ValueError(f"q last-dim {qk_dim} != kv last-dim {kv_dim}")
    if heads % kv_group != 0:
        raise ValueError(f"heads {heads} not divisible by kv_group {kv_group}")
    head_kv = heads // kv_group

    b_i, s_i, g_i, topk = indices.shape
    if (b_i, s_i, g_i) != (batch, seq_len, kv_group):
        raise ValueError(
            f"indices shape {indices.shape} does not match (B,S,G,topk)=("
            f"{batch},{seq_len},{kv_group},*)"
        )

    if d_v is None:
        d_v = qk_dim
    if not (0 < d_v <= qk_dim):
        raise ValueError(f"d_v must be in (0, {qk_dim}], got {d_v}")
    tail_dim = qk_dim - d_v

    return SparseMLAShapes(
        batch=batch,
        seq_len=seq_len,
        seq_len_kv=seq_len_kv,
        heads=heads,
        kv_group=kv_group,
        head_kv=head_kv,
        d_v=d_v,
        qk_dim=qk_dim,
        tail_dim=tail_dim,
        topk=topk,
    )


def _gather_kv(kv: mx.array, indices: mx.array, *, seq_len_kv: int) -> mx.array:
    """Gather KV by per-query topk indices.

    Args:
        kv: [B, Skv, G, D_qk]
        indices: [B, S, G, topk] int32, sentinel == -1 for invalid

    Returns:
        gathered: [B, S, G, topk, D_qk]
    """
    safe_indices = mx.maximum(indices, mx.array(0, dtype=indices.dtype))

    batch, seq_len, kv_group, topk = indices.shape
    batch_idx = mx.arange(batch, dtype=mx.int32).reshape(batch, 1, 1, 1)
    batch_idx = mx.broadcast_to(batch_idx, (batch, seq_len, kv_group, topk))
    group_idx = mx.arange(kv_group, dtype=mx.int32).reshape(1, 1, kv_group, 1)
    group_idx = mx.broadcast_to(group_idx, (batch, seq_len, kv_group, topk))
    gathered = kv[batch_idx, safe_indices, group_idx]  # [B, S, G, topk, D_qk]
    return gathered


def sparse_mla_attention_reference(
    q: mx.array,
    kv: mx.array,
    indices: mx.array,
    *,
    sm_scale: float | None = None,
    d_v: int | None = None,
    return_lse: bool = False,
) -> mx.array | Tuple[mx.array, mx.array]:
    """Pure-MLX sparse multi-latent-attention reference.

    Args:
        q: Query tensor with shape [B, S, H, qk_dim]. ``qk_dim = d_v + tail_dim``.
        kv: Packed KV tensor with shape [B, Skv, G, qk_dim]. The first ``d_v``
            channels supply V; all ``qk_dim`` channels participate in QK^T.
        indices: Per-token top-k KV positions with shape [B, S, G, topk] int.
            Sentinel ``-1`` masks an entry (its softmax weight becomes zero).
        sm_scale: Softmax scale. Defaults to ``1 / sqrt(qk_dim)``.
        d_v: Optional value head dimension. Defaults to ``qk_dim``.
        return_lse: If True also return the log-sum-exp tensor used for the
            backward pass.

    Returns:
        ``out`` shaped ``[B, S, H, d_v]``. If ``return_lse`` is True, returns
        ``(out, lse)`` where ``lse`` has shape ``[B, S, H]`` (in fp32).
    """

    shapes = _resolve_shapes(q, kv, indices, d_v=d_v)
    qk_dim = shapes.qk_dim
    d_v_resolved = shapes.d_v
    if sm_scale is None:
        sm_scale = qk_dim ** -0.5

    # Gather KV: [B, S, G, topk, D_qk]
    indices_i32 = indices.astype(mx.int32)
    gathered = _gather_kv(kv, indices_i32, seq_len_kv=shapes.seq_len_kv)
    # Build mask for invalid indices: True -> valid.
    valid_mask = indices_i32 != _INVALID_INDEX_SENTINEL  # [B, S, G, topk]

    # Reshape q to [B, S, G, head_kv, D_qk] so each kv group's heads are grouped.
    head_kv = shapes.head_kv
    q_grouped = q.reshape(shapes.batch, shapes.seq_len, shapes.kv_group, head_kv, qk_dim)

    # Promote to fp32 for stable softmax / matmul reduction.
    q_fp32 = q_grouped.astype(mx.float32)
    kv_fp32 = gathered.astype(mx.float32)

    # scores = einsum("bsghd,bsgkd->bsghk", q, kv) — but we use plain matmul.
    # q_fp32: [B, S, G, head_kv, D_qk]
    # kv_fp32: [B, S, G, topk, D_qk]
    scores = mx.matmul(q_fp32, mx.swapaxes(kv_fp32, -1, -2))  # [B,S,G,head_kv,topk]
    scores = scores * sm_scale

    # Apply mask broadcasting over heads: valid_mask is [B,S,G,topk] -> add new head axis.
    mask = valid_mask[:, :, :, None, :]
    neg_inf = mx.array(-mx.inf, dtype=mx.float32)
    scores = mx.where(mask, scores, neg_inf)

    # Stable softmax: subtract max along topk axis.
    m_i = mx.max(scores, axis=-1, keepdims=True)  # [B,S,G,head_kv,1]
    # Where every entry is -inf (no valid KV), set m_i to 0 so we avoid -inf - -inf = nan.
    has_any_valid = mx.any(valid_mask, axis=-1, keepdims=True)[:, :, :, None, :]
    m_i_clean = mx.where(has_any_valid, m_i, mx.zeros_like(m_i))
    scores_shifted = scores - m_i_clean
    exp_scores = mx.exp(scores_shifted)
    # Force masked positions to 0 explicitly (helps when max was -inf).
    exp_scores = mx.where(mask, exp_scores, mx.zeros_like(exp_scores))
    sumexp = mx.sum(exp_scores, axis=-1, keepdims=True)  # [B,S,G,head_kv,1]
    # Avoid divide-by-zero when no valid index — output for that token is 0.
    safe_sumexp = mx.where(sumexp > 0, sumexp, mx.ones_like(sumexp))
    probs = exp_scores / safe_sumexp  # [B,S,G,head_kv,topk]

    # V is the first d_v channels of gathered.
    v_fp32 = kv_fp32[..., :d_v_resolved]  # [B,S,G,topk,d_v]
    out_fp32 = mx.matmul(probs, v_fp32)  # [B,S,G,head_kv,d_v]
    # Zero out positions with no valid index.
    out_fp32 = out_fp32 * has_any_valid.astype(mx.float32)

    out = out_fp32.reshape(shapes.batch, shapes.seq_len, shapes.heads, d_v_resolved)
    out = out.astype(q.dtype)

    if not return_lse:
        return out
    # lse = m_i + log(sumexp), keep fp32. Reshape to [B,S,H].
    lse = (m_i_clean + mx.log(safe_sumexp))
    lse = lse.reshape(shapes.batch, shapes.seq_len, shapes.heads)
    return out, lse


def sparse_mla_attention(
    q: mx.array,
    kv: mx.array,
    indices: mx.array,
    *,
    sm_scale: float | None = None,
    d_v: int | None = None,
) -> mx.array:
    """Convenience entry point used by callers that don't need lse."""

    return sparse_mla_attention_reference(
        q,
        kv,
        indices,
        sm_scale=sm_scale,
        d_v=d_v,
        return_lse=False,
    )


__all__ = [
    "SparseMLAShapes",
    "sparse_mla_attention",
    "sparse_mla_attention_reference",
]

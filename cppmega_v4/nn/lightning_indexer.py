"""ROI 7 — DSA Lightning Indexer (V3.2-style), fp32 scaffold.

Reference: ``~/sources/rent_kernels/DeepSeek-V3.2-Exp/inference/model.py``
``class Indexer`` (lines 435-487). The upstream impl uses FP8 quantization
(``act_quant``, ``fp8_index``) and a CUDA fused indexer-logit kernel; this
scaffold ports the **logical structure** to fp32 MLX so the rest of the
stack can wire against the correct interface. A future ROI replaces the
fp32 GEMM with a fused Metal/TileLang kernel without changing this API.

Important V3.2 gotcha preserved: RoPE in the indexer is **non-interleaved**
(MLA RoPE is interleaved). The caller supplies the non-interleaved cos/sin
table; we do not flip the axis here.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass(frozen=True)
class LightningIndexerConfig:
    """V3.2 Indexer config — names mirror upstream ``ModelArgs``."""

    hidden_size: int
    n_heads: int
    head_dim: int = 32           # V3.2 default — small head dim for FP8 GEMM
    rope_head_dim: int = 16      # split of head_dim that gets RoPE
    q_lora_rank: int = 1536      # input dimension of wq_b
    index_topk: int = 64
    softmax_scale: float | None = None
    norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.head_dim <= 0 or self.rope_head_dim <= 0:
            raise ValueError("head_dim and rope_head_dim must be positive")
        if self.rope_head_dim > self.head_dim:
            raise ValueError("rope_head_dim must be <= head_dim")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.index_topk <= 0:
            raise ValueError("index_topk must be positive")


def _apply_non_interleaved_rope(
    x: mx.array, cos: mx.array, sin: mx.array
) -> mx.array:
    """Non-interleaved RoPE: rotate (x[..., :d], x[..., d:2d]) pairs.

    Per V3.2 README: the indexer RoPE is **non-interleaved** (in contrast to
    MLA's interleaved variant). For input shape ``[..., 2*d]``, rotate the
    first-d and second-d halves as a 2D plane.

    cos/sin are expected to be shape ``(T, d)`` and are broadcast across any
    leading batch/head axes between the time axis and the trailing rope axis.
    """
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"rope dim must be even, got {x.shape[-1]}")
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    cos_b = cos.astype(x.dtype)
    sin_b = sin.astype(x.dtype)
    # Reshape cos/sin to broadcast across all axes between dim 0 (time) and
    # the trailing rope axis. x has shape (..., T, ..., 2*d); we assume T is
    # axis 1 and rope is axis -1, with optional head axes in between.
    # Insert singleton axes for each axis between T (axis 1 of x) and rope axis.
    if x.ndim > 2:
        extra_axes = x.ndim - 2  # axes between T (idx 1) and rope (idx -1)
        # cos: (T, d) -> add (extra_axes - 1) singletons before the rope axis
        # so final shape is (1, T, 1, ..., 1, d).
        cos_b = cos_b.reshape(*([1] * 1), cos_b.shape[0], *([1] * (extra_axes - 1)), cos_b.shape[1])
        sin_b = sin_b.reshape(*([1] * 1), sin_b.shape[0], *([1] * (extra_axes - 1)), sin_b.shape[1])
    return mx.concatenate([x1 * cos_b - x2 * sin_b, x2 * cos_b + x1 * sin_b], axis=-1)


class LightningIndexer(nn.Module):
    """Top-k indexer for sparse MLA decode (V3.2-style, fp32 scaffold).

    Forward signature is intentionally close to the upstream class:
        ``forward(x, qr, freqs_cis, mask=None) -> topk_indices``

    Inputs:
        x:         [B, T, hidden_size] — hidden states; used to derive K.
        qr:        [B, T, q_lora_rank] — query LoRA-reduced features.
        freqs_cis: (cos, sin) tuple, each [T, rope_head_dim/2 + ?]; we cover
                   the full rope_head_dim via the two-half non-interleaved
                   rotation above (so cos/sin should be [..., rope_head_dim/2]).
        mask:      optional additive mask [B, T, T_kv] applied to scores.

    Returns:
        topk_indices: [B, T, index_topk] int32 — KV positions per query.
    """

    def __init__(self, config: LightningIndexerConfig):
        super().__init__()
        self.config = config
        self.wq_b = nn.Linear(config.q_lora_rank, config.n_heads * config.head_dim, bias=False)
        self.wk = nn.Linear(config.hidden_size, config.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(config.head_dim, eps=config.norm_eps)
        self.weights_proj = nn.Linear(config.hidden_size, config.n_heads, bias=False)

    def __call__(
        self,
        x: mx.array,
        qr: mx.array,
        freqs_cis: tuple[mx.array, mx.array],
        mask: mx.array | None = None,
    ) -> mx.array:
        cfg = self.config
        batch, seq, _ = x.shape
        cos, sin = freqs_cis

        q = self.wq_b(qr).reshape(batch, seq, cfg.n_heads, cfg.head_dim)
        q_pe = q[..., : cfg.rope_head_dim]
        q_nope = q[..., cfg.rope_head_dim :]
        # Non-interleaved RoPE on the rope_head_dim slice.
        q_pe = _apply_non_interleaved_rope(q_pe, cos, sin)
        q = mx.concatenate([q_pe, q_nope], axis=-1)

        k = self.k_norm(self.wk(x))
        k_pe = k[..., : cfg.rope_head_dim]
        k_nope = k[..., cfg.rope_head_dim :]
        # k has shape [B, T, head_dim]; broadcast for RoPE on the slice.
        k_pe = _apply_non_interleaved_rope(k_pe, cos, sin)
        k = mx.concatenate([k_pe, k_nope], axis=-1)

        # weights_proj scaling matches upstream pattern.
        weights = self.weights_proj(x) * (cfg.n_heads ** -0.5)
        sm_scale = cfg.softmax_scale if cfg.softmax_scale is not None else cfg.head_dim ** -0.5

        # Indexer score: per (B, T_q), sum over heads of q[..., h, :] . k . weights[..., h]
        # q: [B, T_q, H, D]; k: [B, T_kv, D]; weights: [B, T_q, H]
        # Score[b, t_q, t_kv] = sum_h weights[b, t_q, h] * sum_d q[b, t_q, h, d] * k[b, t_kv, d]
        # = einsum: 'bqhd,bkd,bqh->bqk'
        scores = mx.einsum("bqhd,bkd,bqh->bqk", q, k, weights) * sm_scale
        if mask is not None:
            scores = scores + mask
        # Top-k indices over the kv axis.
        topk = min(cfg.index_topk, scores.shape[-1])
        return mx.stop_gradient(
            mx.argpartition(-scores, topk - 1, axis=-1)[..., :topk]
        ).astype(mx.int32)


__all__ = ["LightningIndexer", "LightningIndexerConfig"]

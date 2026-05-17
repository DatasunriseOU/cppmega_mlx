"""ROI 8/9 — V4 sparse attention research scaffolds.

These are intentionally **minimal API surfaces** for the V4 sparse-attention
families. The full implementations need substantial research/spike effort and
land in follow-up cycles. Until then, each class accepts the right inputs
and produces correctly-shaped output by **delegating to the standard dense
SDPA path** (mathematically equivalent to no sparsity = full attention),
making the rest of the v4 stack runnable end-to-end.

ROI 8 — NSA (Native Sparse Attention, arxiv 2502.11089):
    Three-branch sparse attention: Compress + Select + Sliding. Used in V4
    alongside DSA for long-context training. Reference algorithm in the
    paper; reference Triton/TileLang in upstream community ports
    (fla-org/native-sparse-attention if it lands).

ROI 9 — CSA / HCA hybrid attention stack (V4):
    CSA = Compressed Sparse Attention (m-token KV compression + Lightning
    Indexer top-k); HCA = Heavily Compressed Attention. Combined with mHC
    residual streams. Reference: arxiv 2512.24880 (mHC paper) + V4-Pro
    config dump (``feimatrix``). Full V4 tech report not publicly released
    as of May 2026.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


# ----------------------------------------------------------------------------
# ROI 8 — NSA scaffold
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class NativeSparseAttentionConfig:
    """NSA config (arxiv 2502.11089)."""

    hidden_size: int
    num_heads: int
    head_dim: int
    compress_block_size: int = 64    # coarse-block size for Compress branch
    select_topk: int = 16            # top-k for Select branch
    sliding_window: int = 512        # window for Sliding branch
    norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.hidden_size <= 0 or self.num_heads <= 0 or self.head_dim <= 0:
            raise ValueError("hidden_size, num_heads, head_dim must be positive")
        if self.hidden_size != self.num_heads * self.head_dim:
            raise ValueError("hidden_size must equal num_heads * head_dim")
        if self.compress_block_size <= 0 or self.select_topk <= 0 or self.sliding_window <= 0:
            raise ValueError("compress_block_size / select_topk / sliding_window must be positive")


class NativeSparseAttention(nn.Module):
    """NSA scaffold: three-branch (Compress + Select + Sliding) sparse attention.

    Current behavior: falls back to dense causal SDPA so the stack runs end-to-end.
    Per-branch implementations land in a follow-up ROI.
    """

    def __init__(self, config: NativeSparseAttentionConfig):
        super().__init__()
        self.config = config
        d = config.hidden_size
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.o_proj.weight = mx.zeros_like(self.o_proj.weight)  # identity at init
        self.norm = nn.RMSNorm(d, eps=config.norm_eps)
        # Per-branch gates land in follow-up; scaffold uses a single combined route.
        self.branch_gate = nn.Linear(d, 3, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim != 3 or x.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"x must be [B, S, {self.config.hidden_size}], got {x.shape}"
            )
        cfg = self.config
        b, s, _ = x.shape
        q = self.q_proj(x).reshape(b, s, cfg.num_heads, cfg.head_dim)
        k = self.k_proj(x).reshape(b, s, cfg.num_heads, cfg.head_dim)
        v = self.v_proj(x).reshape(b, s, cfg.num_heads, cfg.head_dim)
        # Transpose to [B, H, S, D] for SDPA.
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        # Dense causal SDPA — the three sparse branches collapse to this in
        # the scaffold. Real NSA replaces this with the Compress+Select+Sliding mix.
        scale = cfg.head_dim ** -0.5
        scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * scale
        # Causal mask.
        causal = mx.tril(mx.ones((s, s), dtype=mx.bool_))
        scores = mx.where(causal, scores, mx.full(scores.shape, -1e9, dtype=scores.dtype))
        weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        out = mx.matmul(weights, v)
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(b, s, cfg.hidden_size)
        return self.norm(self.o_proj(out))


# ----------------------------------------------------------------------------
# ROI 9 — CSA / HCA hybrid scaffold
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class CsaHcaHybridConfig:
    """V4 hybrid attention config (arxiv 2512.24880)."""

    hidden_size: int
    num_heads: int
    head_dim: int
    m_token_compression: int = 4      # CSA: KV compression ratio
    heavy_compression: int = 16       # HCA: more aggressive compression
    norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.hidden_size <= 0 or self.num_heads <= 0 or self.head_dim <= 0:
            raise ValueError("hidden_size, num_heads, head_dim must be positive")
        if self.hidden_size != self.num_heads * self.head_dim:
            raise ValueError("hidden_size must equal num_heads * head_dim")
        if self.m_token_compression <= 0 or self.heavy_compression <= 0:
            raise ValueError("compression ratios must be positive")


class CsaHcaHybridAttention(nn.Module):
    """CSA + HCA hybrid scaffold. Falls back to dense causal SDPA today."""

    def __init__(self, config: CsaHcaHybridConfig):
        super().__init__()
        self.config = config
        d = config.hidden_size
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.o_proj.weight = mx.zeros_like(self.o_proj.weight)
        self.norm = nn.RMSNorm(d, eps=config.norm_eps)

    def __call__(self, x: mx.array) -> mx.array:
        # Reuse the NSA scaffold's dense-SDPA fallback for now.
        if x.ndim != 3 or x.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"x must be [B, S, {self.config.hidden_size}], got {x.shape}"
            )
        cfg = self.config
        b, s, _ = x.shape
        q = self.q_proj(x).reshape(b, s, cfg.num_heads, cfg.head_dim)
        k = self.k_proj(x).reshape(b, s, cfg.num_heads, cfg.head_dim)
        v = self.v_proj(x).reshape(b, s, cfg.num_heads, cfg.head_dim)
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        scale = cfg.head_dim ** -0.5
        scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * scale
        causal = mx.tril(mx.ones((s, s), dtype=mx.bool_))
        scores = mx.where(causal, scores, mx.full(scores.shape, -1e9, dtype=scores.dtype))
        weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        out = mx.matmul(weights, v)
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(b, s, cfg.hidden_size)
        return self.norm(self.o_proj(out))


__all__ = [
    "CsaHcaHybridAttention",
    "CsaHcaHybridConfig",
    "NativeSparseAttention",
    "NativeSparseAttentionConfig",
]

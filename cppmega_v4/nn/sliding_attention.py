"""GQA with sliding-window mask — Gemma 4-style 5:1 sliding-vs-global slot.

Thin wrapper around ``mx.fast.scaled_dot_product_attention``. The block
shape mirrors a standard GQA layer: q/k/v projections with asymmetric
head counts, optional RMSNorm on Q/K, RoPE, then SDPA with a sliding
causal mask. Used by the Gemma 4 and Arcee Trinity Large presets in the
``cppmega_v4.architectures`` registry.

The SDPA call itself is unchanged — only the mask is replaced. For a
window of size ``W``, position ``i`` attends to keys ``j`` in
``[max(0, i-W+1), i]``. This is the standard Gemma-3 / Mistral
sliding-attention pattern.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class GQASlidingConfig:
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    sliding_window_size: int = 4096
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    qk_norm: bool = True


def _sliding_causal_mask(seq_len: int, window: int, dtype: mx.Dtype) -> mx.array:
    """Lower-triangular mask that also forbids keys older than ``window``."""
    i = mx.arange(seq_len)[:, None]
    j = mx.arange(seq_len)[None, :]
    keep = (j <= i) & (j > i - window)
    neg = mx.full((seq_len, seq_len), -1e9, dtype=dtype)
    zero = mx.zeros((seq_len, seq_len), dtype=dtype)
    return mx.where(keep, zero, neg)


class GQAWithSlidingWindowBlock(nn.Module):
    """Causal GQA with a fixed sliding-window receptive field."""

    def __init__(self, cfg: GQASlidingConfig):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden_size
        nh = cfg.num_attention_heads
        nkv = cfg.num_key_value_heads
        d = cfg.head_dim
        if nh % nkv != 0:
            raise ValueError(
                f"num_attention_heads ({nh}) must be divisible by "
                f"num_key_value_heads ({nkv})"
            )
        self.q_proj = nn.Linear(H, nh * d, bias=False)
        self.k_proj = nn.Linear(H, nkv * d, bias=False)
        self.v_proj = nn.Linear(H, nkv * d, bias=False)
        self.o_proj = nn.Linear(nh * d, H, bias=False)
        if cfg.qk_norm:
            self.q_norm = nn.RMSNorm(d, eps=cfg.rms_norm_eps)
            self.k_norm = nn.RMSNorm(d, eps=cfg.rms_norm_eps)
        else:
            self.q_norm = self.k_norm = None
        self.rope = nn.RoPE(d, base=cfg.rope_theta)
        # Zero-init out so the block is identity at init.
        self.o_proj.weight = mx.zeros_like(self.o_proj.weight)

    def __call__(self, x: mx.array) -> mx.array:
        B, S, _ = x.shape
        nh = self.cfg.num_attention_heads
        nkv = self.cfg.num_key_value_heads
        d = self.cfg.head_dim
        q = self.q_proj(x).reshape(B, S, nh, d).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, S, nkv, d).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, S, nkv, d).transpose(0, 2, 1, 3)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = self.rope(q)
        k = self.rope(k)
        if nkv != nh:
            repeats = nh // nkv
            k = mx.repeat(k, repeats=repeats, axis=1)
            v = mx.repeat(v, repeats=repeats, axis=1)
        mask = _sliding_causal_mask(S, self.cfg.sliding_window_size, q.dtype)
        scale = 1.0 / (d ** 0.5)
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, S, nh * d)
        return self.o_proj(out)


__all__ = ["GQASlidingConfig", "GQAWithSlidingWindowBlock"]

"""Coarse Causal Attention (CCA) — ZAYA1's compressed-context attention.

Each query attends to two key/value streams:

  - **fine**: the last ``fine_window`` tokens, full resolution.
  - **coarse**: a mean-pool of every ``coarse_block_size`` consecutive
    tokens before the fine window, providing long-range context at a
    compression ratio of ``coarse_block_size``.

The two streams are concatenated along the key dimension; standard
softmax attention is applied over the union. The causal property is
preserved because (a) the fine stream is masked to a sliding window
ending at position ``i`` and (b) only coarse blocks whose **last**
token lies at or before ``i - fine_window`` contribute keys.

This is a thin first-class brick wrapper: it gives the planner a real
brick to plan around (``cca_attention`` kind) and gives the model
factory a working forward path. The Apple Metal kernel for it lands
later — for now compute goes through ``mx.fast.scaled_dot_product_attention``.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class CCAAttentionConfig:
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    fine_window: int = 256
    coarse_block_size: int = 16
    rms_norm_eps: float = 1e-6


class CCAAttentionBlock(nn.Module):
    """Compressed-context causal attention."""

    def __init__(self, cfg: CCAAttentionConfig):
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
        if cfg.coarse_block_size < 1:
            raise ValueError("coarse_block_size must be ≥ 1")
        if cfg.fine_window < 1:
            raise ValueError("fine_window must be ≥ 1")
        self.q_proj = nn.Linear(H, nh * d, bias=False)
        self.k_proj = nn.Linear(H, nkv * d, bias=False)
        self.v_proj = nn.Linear(H, nkv * d, bias=False)
        self.o_proj = nn.Linear(nh * d, H, bias=False)
        self.q_norm = nn.RMSNorm(d, eps=cfg.rms_norm_eps)
        self.k_norm = nn.RMSNorm(d, eps=cfg.rms_norm_eps)
        self.o_proj.weight = mx.zeros_like(self.o_proj.weight)

    def _coarse_pool(self, x: mx.array) -> mx.array:
        """Mean-pool x: [B, nkv, S, d] -> [B, nkv, S_coarse, d]."""
        B, nkv, S, d = x.shape
        bs = self.cfg.coarse_block_size
        if S < bs:
            return mx.zeros((B, nkv, 0, d), dtype=x.dtype)
        usable = (S // bs) * bs
        trimmed = x[:, :, :usable, :].reshape(B, nkv, usable // bs, bs, d)
        return trimmed.mean(axis=3)

    def __call__(self, x: mx.array) -> mx.array:
        B, S, _ = x.shape
        nh = self.cfg.num_attention_heads
        nkv = self.cfg.num_key_value_heads
        d = self.cfg.head_dim
        bs = self.cfg.coarse_block_size
        fw = self.cfg.fine_window

        q = self.q_proj(x).reshape(B, S, nh, d).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, S, nkv, d).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, S, nkv, d).transpose(0, 2, 1, 3)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Fine stream — last fw tokens, GQA-broadcast to nh heads.
        if nkv != nh:
            reps = nh // nkv
            k_full = mx.repeat(k, repeats=reps, axis=1)
            v_full = mx.repeat(v, repeats=reps, axis=1)
        else:
            k_full = k
            v_full = v

        # Coarse stream on the original GQA-shaped k, v.
        k_coarse_kv = self._coarse_pool(k)
        v_coarse_kv = self._coarse_pool(v)
        if k_coarse_kv.shape[2] > 0 and nkv != nh:
            reps = nh // nkv
            k_coarse = mx.repeat(k_coarse_kv, repeats=reps, axis=1)
            v_coarse = mx.repeat(v_coarse_kv, repeats=reps, axis=1)
        else:
            k_coarse = k_coarse_kv
            v_coarse = v_coarse_kv

        Sc = k_coarse.shape[2]
        # Concatenated keys/values: [B, nh, S+Sc, d] with coarse first.
        if Sc > 0:
            keys = mx.concatenate([k_coarse, k_full], axis=2)
            vals = mx.concatenate([v_coarse, v_full], axis=2)
        else:
            keys = k_full
            vals = v_full

        # Mask:
        #   - coarse block c covers fine positions [c*bs, c*bs + bs - 1];
        #     a query at position i may attend to it iff its last token
        #     index (c*bs + bs - 1) <= i - fw  →  c <= (i - fw - bs + 1) / bs
        #   - fine key at position j is reachable iff (i - fw) < j <= i.
        i_idx = mx.arange(S)[:, None]                       # [S, 1]
        if Sc > 0:
            c_idx = mx.arange(Sc)[None, :]                  # [1, Sc]
            coarse_keep = c_idx <= ((i_idx - fw - bs + 1) // bs)
        j_idx = mx.arange(S)[None, :]                       # [1, S]
        fine_keep = (j_idx <= i_idx) & (j_idx > i_idx - fw)
        if Sc > 0:
            keep = mx.concatenate([coarse_keep, fine_keep], axis=1)
        else:
            keep = fine_keep
        neg = mx.full(keep.shape, -1e9, dtype=q.dtype)
        zero = mx.zeros(keep.shape, dtype=q.dtype)
        mask = mx.where(keep, zero, neg)

        scale = 1.0 / (d ** 0.5)
        out = mx.fast.scaled_dot_product_attention(
            q, keys, vals, scale=scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, S, nh * d)
        return self.o_proj(out)


__all__ = ["CCAAttentionBlock", "CCAAttentionConfig"]

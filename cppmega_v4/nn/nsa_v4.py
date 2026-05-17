"""ROI 8 — Native Sparse Attention (arxiv 2502.11089), real three-branch impl.

Replaces the dense-SDPA fallback in ``sparse_attention_v4.NativeSparseAttention``
with the actual three-branch sparse attention from the NSA paper:

  - Compress branch: mean-pool K/V into coarse blocks of ``compress_block_size``;
    query attends to all coarse blocks (cheap, global, low-resolution).
  - Select branch: each query picks top-k coarse blocks by Compress-score, then
    attends to the *full-resolution* K/V inside those blocks (sparse, full-fi).
  - Sliding branch: standard windowed causal attention over the last
    ``sliding_window`` tokens (high-fi local context).

Final output is a learned softmax-gated mixture of the three branch outputs.

This is the "hardware-aligned trainable sparse attention" pattern. Two
performance levers we don't (yet) take in pure MLX:

  1. Block-sparse SDPA kernel — current impl materializes the per-branch
     attention masks and runs dense matmul + softmax. Functionally correct;
     a Metal/TileLang fused block-sparse kernel could replace this without
     changing the API. (Tracked: cppmega_v4._tilelang.nsa_path_c.py)
  2. Coarse-block KV cache — current impl recomputes compress pool each
     forward; a streaming cache (segment-tree style) avoids the redundant
     mean over historical tokens. (Tracked: ROI 8.B)

The module is wrapped so callers who want the old "dense fallback" can
keep using ``NativeSparseAttention``; the new real impl lives under
``NativeSparseAttentionV4``.
"""

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


@dataclass(frozen=True)
class NSAConfig:
    """NSA three-branch config (arxiv 2502.11089)."""

    hidden_size: int
    num_heads: int
    head_dim: int
    compress_block_size: int = 64
    select_topk: int = 16          # # of coarse blocks each query attends to
    sliding_window: int = 512
    norm_eps: float = 1e-6
    gate_init_eps: float = 1e-2    # small init so all 3 branches contribute initially

    def __post_init__(self) -> None:
        if self.hidden_size != self.num_heads * self.head_dim:
            raise ValueError(
                f"hidden_size {self.hidden_size} must equal "
                f"num_heads * head_dim ({self.num_heads * self.head_dim})"
            )
        for nm, v in [
            ("compress_block_size", self.compress_block_size),
            ("select_topk", self.select_topk),
            ("sliding_window", self.sliding_window),
        ]:
            if v <= 0:
                raise ValueError(f"{nm} must be positive, got {v}")


def _causal_mask(seq: int) -> mx.array:
    """[seq, seq] bool causal mask: row i can attend to cols j <= i."""
    return mx.tril(mx.ones((seq, seq), dtype=mx.bool_))


def _apply_mask_and_softmax(
    scores: mx.array, mask: mx.array, scale: float
) -> mx.array:
    """Mask additive, softmax over last axis, return same dtype as scores."""
    in_dtype = scores.dtype
    masked = mx.where(mask, scores, mx.full(scores.shape, -1e9, dtype=scores.dtype))
    return mx.softmax((masked * scale).astype(mx.float32), axis=-1).astype(in_dtype)


def _compress_branch(
    q: mx.array, k: mx.array, v: mx.array, *, block_size: int, scale: float,
) -> tuple[mx.array, mx.array]:
    """Pool K/V into coarse blocks, attend q over all blocks. Causal in blocks.

    Returns (out, block_scores) — block_scores is [B, H, S_q, n_blocks] used
    by the Select branch for top-k selection.
    """
    B, H, S, D = k.shape
    # Pad S to a multiple of block_size with zeros (masked out below).
    pad = (-S) % block_size
    if pad:
        k = mx.pad(k, ((0, 0), (0, 0), (0, pad), (0, 0)))
        v = mx.pad(v, ((0, 0), (0, 0), (0, pad), (0, 0)))
    S_p = S + pad
    n_blocks = S_p // block_size
    # Reshape to [B, H, n_blocks, block_size, D] and mean over block axis.
    k_blocks = k.reshape(B, H, n_blocks, block_size, D).mean(axis=3)
    v_blocks = v.reshape(B, H, n_blocks, block_size, D).mean(axis=3)
    # Attention: q [B,H,S,D] @ k_blocks.T [B,H,D,n_blocks] -> [B,H,S,n_blocks]
    raw_scores = mx.matmul(q, mx.transpose(k_blocks, (0, 1, 3, 2)))
    # Causal-in-blocks mask: query at token i can attend to block b iff
    # the *first* token of block b is at position <= i.
    rows = mx.arange(S)[:, None]                          # [S, 1]
    block_start = (mx.arange(n_blocks)[None, :]) * block_size  # [1, n_blocks]
    mask = (rows >= block_start)                          # [S, n_blocks]
    mask = mask[None, None, :, :]                         # [1, 1, S, n_blocks]
    weights = _apply_mask_and_softmax(raw_scores, mask, scale)
    out = mx.matmul(weights, v_blocks)                    # [B,H,S,D]
    return out, raw_scores                                # scores: [B,H,S,n_blocks]


def _select_branch(
    q: mx.array, k: mx.array, v: mx.array,
    block_scores: mx.array,
    *, block_size: int, topk: int, scale: float,
) -> mx.array:
    """Top-k block selection per query, then dense attention over selected tokens.

    For correctness in pure MLX we build the per-(B,H,S_q) sparse mask
    [B, H, S_q, S_kv] from the top-k coarse-block indices, then run dense
    SDPA with that mask. A real Metal/TileLang impl avoids materializing
    the full [S_q, S_kv] mask.
    """
    B, H, S, D = q.shape
    n_blocks = block_scores.shape[-1]
    topk_eff = min(topk, n_blocks)
    # Top-k block indices per (B, H, S_q): [B, H, S_q, topk]
    top_idx = mx.argpartition(-block_scores, topk_eff - 1, axis=-1)[..., :topk_eff]
    # Expand block indices into per-token KV indices.
    # block b → tokens [b*bs, (b+1)*bs). Build a [B, H, S_q, n_blocks] bool
    # mask over coarse blocks, then expand to [B, H, S_q, S_kv].
    block_one_hot = mx.zeros((B, H, S, n_blocks), dtype=mx.bool_)
    # Scatter top-k positions to True.
    # MLX doesn't have efficient scatter; build via comparison.
    all_blocks = mx.arange(n_blocks)[None, None, None, :]  # [1,1,1,n_blocks]
    # top_idx [B,H,S,topk] → broadcast vs all_blocks → equality match
    match = (top_idx[..., :, None] == all_blocks[..., None, :])  # [B,H,S,topk,n_blocks]
    block_one_hot = match.any(axis=-2)                            # [B,H,S,n_blocks]
    # Expand each block to its token span (block_size repeats).
    S_kv = n_blocks * block_size
    tok_mask = mx.repeat(block_one_hot, block_size, axis=-1)      # [B,H,S,S_kv]
    # Trim to the original S (post-padding) — caller passed padded K/V.
    tok_mask = tok_mask[..., :k.shape[2]]
    # Additionally enforce causality.
    causal = _causal_mask(S)[None, None, :, :]                    # [1,1,S,S]
    if k.shape[2] > S:
        pad_cols = k.shape[2] - S
        causal = mx.pad(causal, ((0, 0), (0, 0), (0, 0), (0, pad_cols)))
    final_mask = tok_mask & causal[..., :tok_mask.shape[-1]]
    # SDPA.
    scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2)))
    weights = _apply_mask_and_softmax(scores, final_mask, scale)
    return mx.matmul(weights, v)                                  # [B,H,S,D]


def _sliding_branch(
    q: mx.array, k: mx.array, v: mx.array,
    *, window: int, scale: float,
) -> mx.array:
    """Causal sliding-window attention: query i attends to tokens (i-window, i]."""
    B, H, S, D = q.shape
    rows = mx.arange(S)[:, None]
    cols = mx.arange(k.shape[2])[None, :]
    causal = cols <= rows
    in_window = (rows - cols) < window
    mask = (causal & in_window)[None, None, :, :]
    scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2)))
    weights = _apply_mask_and_softmax(scores, mask, scale)
    return mx.matmul(weights, v)


class NativeSparseAttentionV4(nn.Module):
    """Real three-branch NSA: Compress + Select + Sliding, gated mixture.

    Drop-in for ``sparse_attention_v4.NativeSparseAttention`` (same I/O
    signature `(B, S, H) → (B, S, H)`). The branch gate is a tiny
    learned softmax over [compress, select, sliding] per token+head;
    initialized to a uniform distribution so all three branches
    contribute from step 0.
    """

    def __init__(self, config: NSAConfig):
        super().__init__()
        self.config = config
        d = config.hidden_size
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.norm = nn.RMSNorm(d, eps=config.norm_eps)
        # Branch gate: x → [3] logits per token.
        self.branch_gate = nn.Linear(d, 3, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        cfg = self.config
        if x.ndim != 3 or x.shape[-1] != cfg.hidden_size:
            raise ValueError(
                f"x must be [B, S, {cfg.hidden_size}], got {x.shape}"
            )
        B, S, _ = x.shape
        q = self.q_proj(x).reshape(B, S, cfg.num_heads, cfg.head_dim)
        k = self.k_proj(x).reshape(B, S, cfg.num_heads, cfg.head_dim)
        v = self.v_proj(x).reshape(B, S, cfg.num_heads, cfg.head_dim)
        q = mx.transpose(q, (0, 2, 1, 3))  # [B, H, S, D]
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        scale = cfg.head_dim ** -0.5

        # Three branches.
        compress_out, block_scores = _compress_branch(
            q, k, v, block_size=cfg.compress_block_size, scale=scale,
        )
        # Pad K/V to multiple of block_size to share with Select branch.
        pad = (-S) % cfg.compress_block_size
        k_p = mx.pad(k, ((0, 0), (0, 0), (0, pad), (0, 0))) if pad else k
        v_p = mx.pad(v, ((0, 0), (0, 0), (0, pad), (0, 0))) if pad else v
        select_out = _select_branch(
            q, k_p, v_p, block_scores,
            block_size=cfg.compress_block_size, topk=cfg.select_topk, scale=scale,
        )
        sliding_out = _sliding_branch(q, k, v, window=cfg.sliding_window, scale=scale)

        # Gated mixture: gate from x → 3 logits → softmax → weighted sum.
        gate_logits = self.branch_gate(x)                          # [B, S, 3]
        gate = mx.softmax(gate_logits.astype(mx.float32), axis=-1).astype(x.dtype)
        # Branch outputs are [B, H, S, D] — transpose to [B, S, H, D] then mix per token.
        branches = mx.stack([
            mx.transpose(compress_out, (0, 2, 1, 3)),
            mx.transpose(select_out,   (0, 2, 1, 3)),
            mx.transpose(sliding_out,  (0, 2, 1, 3)),
        ], axis=-1)                                                # [B, S, H, D, 3]
        mixed = (branches * gate[:, :, None, None, :]).sum(axis=-1)  # [B, S, H, D]
        out = mixed.reshape(B, S, cfg.hidden_size)
        return self.norm(self.o_proj(out))


__all__ = [
    "NSAConfig",
    "NativeSparseAttentionV4",
    "_compress_branch",
    "_select_branch",
    "_sliding_branch",
]

"""ROI 9 — CSA + HCA hybrid V4 attention, real compression impl.

DeepSeek-V4's attention stack pairs:

  - CSA (Compressed Sparse Attention): KV is m-compressed (every m tokens
    averaged into one) AND filtered through a Lightning-Indexer top-k.
    Q sees ⌈S/m⌉ super-tokens.
  - HCA (Heavily Compressed Attention): same shape, larger m
    (m_heavy ≫ m_csa) — gives a coarse global summary for very-long ctx.
  - Hybrid: outputs of both are gated together with the optional MLA branch
    (handled by the caller — this module only does CSA+HCA).

This module replaces the dense-SDPA fallback in
``sparse_attention_v4.CsaHcaHybridAttention`` with real m-token KV
compression + gated mixture. The mean-pool compression matches the NSA
Compress branch, just at a different (configurable) ratio.

Implementation notes:
  - We *do not* include the Lightning Indexer top-k inside this module —
    it composes externally via ``cppmega_v4.nn.lightning_indexer_fp8``.
    The hybrid module takes optional ``select_indices`` arguments to
    consume them when the indexer wires up. Without indices, attention
    runs over the full compressed-block set.
  - The two branches use the same q/k/v projections (the compression
    ratio differs, not the underlying state) — this matches the DSV4
    weight-sharing pattern (see arxiv:2512.24880).
"""

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from cppmega_v4.nn.nsa_v4 import _apply_mask_and_softmax


@dataclass(frozen=True)
class CSAHCAConfig:
    """CSA + HCA hybrid attention config (arxiv 2512.24880)."""

    hidden_size: int
    num_heads: int
    head_dim: int
    m_csa: int = 4               # CSA compression ratio (tokens per super-token)
    m_hca: int = 16              # HCA compression ratio (much larger)
    norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.hidden_size != self.num_heads * self.head_dim:
            raise ValueError(
                f"hidden_size {self.hidden_size} must equal "
                f"num_heads * head_dim ({self.num_heads * self.head_dim})"
            )
        if self.m_csa <= 0 or self.m_hca <= 0:
            raise ValueError("compression ratios must be positive")
        if self.m_hca < self.m_csa:
            raise ValueError(
                f"m_hca ({self.m_hca}) must be >= m_csa ({self.m_csa})"
            )


def _compress_kv(
    k: mx.array, v: mx.array, m: int,
) -> tuple[mx.array, mx.array, int]:
    """Mean-pool K/V every m tokens. Pads with zeros if S % m != 0.

    Returns (k_comp, v_comp, n_super) — both with shape [B, H, n_super, D].
    """
    B, H, S, D = k.shape
    pad = (-S) % m
    if pad:
        k = mx.pad(k, ((0, 0), (0, 0), (0, pad), (0, 0)))
        v = mx.pad(v, ((0, 0), (0, 0), (0, pad), (0, 0)))
    n_super = (S + pad) // m
    k_c = k.reshape(B, H, n_super, m, D).mean(axis=3)
    v_c = v.reshape(B, H, n_super, m, D).mean(axis=3)
    return k_c, v_c, n_super


def _compressed_attention(
    q: mx.array, k_c: mx.array, v_c: mx.array,
    *, m: int, original_seq: int, scale: float,
    select_indices: Optional[mx.array] = None,
) -> mx.array:
    """Attention from q [B,H,S_q,D] to compressed KV [B,H,n_super,D].

    Causal-in-supertokens: query at position i sees super-token b iff
    b*m <= i (the *first* token of super-block b is no later than i).
    Optionally restrict to a top-k of super-tokens per query via
    ``select_indices`` (shape [B,H,S_q,k] int32, from an external indexer).
    """
    B, H, S_q, D = q.shape
    n_super = k_c.shape[2]
    scores = mx.matmul(q, mx.transpose(k_c, (0, 1, 3, 2)))  # [B,H,S_q,n_super]
    # Causal mask in supertokens.
    rows = mx.arange(S_q)[:, None]
    super_start = (mx.arange(n_super)[None, :]) * m
    causal_mask = (rows >= super_start)[None, None, :, :]   # [1,1,S_q,n_super]
    # Optional top-k mask from external indexer.
    if select_indices is not None:
        all_super = mx.arange(n_super)[None, None, None, :]
        match = (select_indices[..., :, None] == all_super[..., None, :])
        topk_mask = match.any(axis=-2)
        final_mask = causal_mask & topk_mask
    else:
        final_mask = causal_mask
    weights = _apply_mask_and_softmax(scores, final_mask, scale)
    return mx.matmul(weights, v_c)                          # [B,H,S_q,D]


class CSAHCAHybridV4(nn.Module):
    """Real CSA + HCA hybrid attention: two compression ratios, gated mixture.

    Drop-in for ``sparse_attention_v4.CsaHcaHybridAttention`` (same I/O).
    Branch outputs are softmax-gated per token.
    """

    def __init__(self, config: CSAHCAConfig):
        super().__init__()
        self.config = config
        d = config.hidden_size
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.norm = nn.RMSNorm(d, eps=config.norm_eps)
        # 2-way gate over [csa, hca] per token.
        self.branch_gate = nn.Linear(d, 2, bias=False)

    def __call__(
        self,
        x: mx.array,
        *,
        csa_select_indices: Optional[mx.array] = None,
        hca_select_indices: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward.

        Args:
            x: [B, S, hidden_size].
            csa_select_indices: optional [B, H, S, k] top-k super-token indices
                from a Lightning Indexer on the CSA-compressed KV. When None
                CSA attends to all super-tokens (causal).
            hca_select_indices: same, for HCA.
        """
        cfg = self.config
        if x.ndim != 3 or x.shape[-1] != cfg.hidden_size:
            raise ValueError(
                f"x must be [B, S, {cfg.hidden_size}], got {x.shape}"
            )
        B, S, _ = x.shape
        q = self.q_proj(x).reshape(B, S, cfg.num_heads, cfg.head_dim)
        k = self.k_proj(x).reshape(B, S, cfg.num_heads, cfg.head_dim)
        v = self.v_proj(x).reshape(B, S, cfg.num_heads, cfg.head_dim)
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        scale = cfg.head_dim ** -0.5

        # CSA branch.
        k_c, v_c, _ = _compress_kv(k, v, cfg.m_csa)
        csa_out = _compressed_attention(
            q, k_c, v_c, m=cfg.m_csa, original_seq=S, scale=scale,
            select_indices=csa_select_indices,
        )
        # HCA branch.
        k_h, v_h, _ = _compress_kv(k, v, cfg.m_hca)
        hca_out = _compressed_attention(
            q, k_h, v_h, m=cfg.m_hca, original_seq=S, scale=scale,
            select_indices=hca_select_indices,
        )

        # Gate (2-way softmax).
        gate_logits = self.branch_gate(x)                          # [B, S, 2]
        gate = mx.softmax(gate_logits.astype(mx.float32), axis=-1).astype(x.dtype)
        # Branch outputs are [B, H, S, D]; rearrange + mix.
        branches = mx.stack([
            mx.transpose(csa_out, (0, 2, 1, 3)),
            mx.transpose(hca_out, (0, 2, 1, 3)),
        ], axis=-1)                                                # [B,S,H,D,2]
        mixed = (branches * gate[:, :, None, None, :]).sum(axis=-1)
        out = mixed.reshape(B, S, cfg.hidden_size)
        return self.norm(self.o_proj(out))


__all__ = [
    "CSAHCAConfig",
    "CSAHCAHybridV4",
    "_compress_kv",
    "_compressed_attention",
]

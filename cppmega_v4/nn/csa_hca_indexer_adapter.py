"""Adapter: Lightning Indexer top-k tokens → CSA/HCA super-token indices.

LightningIndexerFP8 (cppmega_v4/nn/lightning_indexer_fp8.py) returns
``[B, S_q, top_k_tokens]`` int32 token-level indices ("top-k positions
to attend to"). CSAHCAHybridV4 (cppmega_v4/nn/csa_hca_v4.py) expects
``[B, H, S_q, k_super]`` int32 *super-token* indices for the
``csa_select_indices`` / ``hca_select_indices`` arguments — one super-
token is ``m_csa`` (or ``m_hca``) consecutive tokens.

This module provides the projection:

  super_idx[b, h, s, k_super] = token_idx[b, s, k_token] // m

with optional dedup-and-pad so each query sees ``k_super`` *distinct*
super-tokens (Lightning's top-k may collapse onto the same super-block
when ``m > 1``).

Public API:
    apply_indexer_to_csa_hca(indexer, csa_hca, x, qr, freqs_cis,
                             *, k_super_csa=None, k_super_hca=None,
                             num_heads=None) → out: [B, S, hidden_size]

The function wires the indexer's output through CSAHCAHybridV4 with
both select-indices populated. ``k_super_csa`` defaults to
``indexer.config.index_topk // m_csa`` (rounded down, min 1); same for HCA.
``num_heads`` defaults to ``csa_hca.config.num_heads`` (broadcast over H).
"""

from typing import Optional

import mlx.core as mx

from cppmega_v4.nn.csa_hca_v4 import CSAHCAHybridV4
from cppmega_v4.nn.lightning_indexer import LightningIndexer
from cppmega_v4.nn.lightning_indexer_fp8 import LightningIndexerFP8


def _tokens_to_super_indices(
    token_indices: mx.array,  # [B, S_q, top_k_tokens] int32
    m: int,
    k_super: int,
    num_heads: int,
) -> mx.array:
    """Map per-token top-k indices to per-(super-token) top-k indices.

    Steps:
      1. Divide each token index by m → super-token index.
      2. Per (b, s), dedup adjacent super-token indices to avoid attending
         to the same super-token twice when several top-tokens collapse
         onto the same block. (Simple O(top_k) pass; we keep first-seen.)
      3. If fewer than k_super distinct super-tokens were found, pad with
         the last-seen (the dedup outcome — repeated indices in the final
         mask are harmless because the OR'd token_mask in CSA's select
         branch is idempotent).
      4. Truncate to k_super entries per (b, s).
      5. Broadcast across num_heads → [B, H, S, k_super].
    """
    if m <= 0:
        raise ValueError(f"m must be positive, got {m}")
    if k_super <= 0:
        raise ValueError(f"k_super must be positive, got {k_super}")
    if num_heads <= 0:
        raise ValueError(f"num_heads must be positive, got {num_heads}")

    # 1. Token → super-token via floor-div by m.
    super_tok = token_indices.astype(mx.int32) // m   # [B, S, top_k_tokens]

    # 2-4. Dedup via Python loop on small last dim. top_k is typically
    # small (≤64) so this is cheap and exact.
    B, S, top_k_tokens = super_tok.shape
    arr = mx.array(super_tok)  # ensure eager
    # Build numpy view for the dedup pass (tiny array, no perf concern).
    import numpy as np
    arr_np = np.array(arr)
    out_np = np.zeros((B, S, k_super), dtype=np.int32)
    for b in range(B):
        for s in range(S):
            seen: list[int] = []
            for k in range(top_k_tokens):
                idx = int(arr_np[b, s, k])
                if idx not in seen:
                    seen.append(idx)
                    if len(seen) == k_super:
                        break
            # Pad with last-seen if we ran out of unique indices.
            if not seen:
                seen = [0]
            while len(seen) < k_super:
                seen.append(seen[-1])
            out_np[b, s, :] = seen[:k_super]
    super_dedup = mx.array(out_np)   # [B, S, k_super]

    # 5. Broadcast across H → [B, H, S, k_super].
    out = mx.broadcast_to(
        super_dedup[:, None, :, :], (B, num_heads, S, k_super),
    )
    return out


def apply_indexer_to_csa_hca(
    indexer: LightningIndexer | LightningIndexerFP8,
    csa_hca: CSAHCAHybridV4,
    x: mx.array,
    qr: mx.array,
    freqs_cis: tuple[mx.array, mx.array],
    *,
    mask: Optional[mx.array] = None,
    k_super_csa: Optional[int] = None,
    k_super_hca: Optional[int] = None,
    num_heads: Optional[int] = None,
) -> mx.array:
    """Run Lightning Indexer, project its top-k to CSA+HCA super-tokens,
    then call CSAHCAHybridV4 with both select-indices populated.

    Args:
        indexer: an instantiated LightningIndexer / LightningIndexerFP8.
        csa_hca: an instantiated CSAHCAHybridV4.
        x:        [B, S, hidden_size] hidden states.
        qr:       [B, S, q_lora_rank] LoRA-reduced query features.
        freqs_cis: (cos, sin) tuple for the indexer's RoPE; each
                   ``[S, rope_head_dim/2]``.
        mask:     optional additive mask for the indexer's logits.
        k_super_csa / k_super_hca: number of super-tokens to select per
            (b, h, s); defaults to ``index_topk // m`` (min 1).
        num_heads: number of heads to broadcast indices across; defaults
            to ``csa_hca.config.num_heads``.

    Returns:
        out: [B, S, hidden_size] — CSAHCAHybridV4 output.
    """
    cfg = csa_hca.config
    if num_heads is None:
        num_heads = cfg.num_heads

    token_indices = indexer(x, qr, freqs_cis, mask=mask)  # [B, S, top_k_tokens]
    idx_topk = indexer.config.index_topk

    if k_super_csa is None:
        k_super_csa = max(1, idx_topk // cfg.m_csa)
    if k_super_hca is None:
        k_super_hca = max(1, idx_topk // cfg.m_hca)

    csa_sel = _tokens_to_super_indices(token_indices, cfg.m_csa,
                                        k_super_csa, num_heads)
    hca_sel = _tokens_to_super_indices(token_indices, cfg.m_hca,
                                        k_super_hca, num_heads)

    return csa_hca(x, csa_select_indices=csa_sel, hca_select_indices=hca_sel)


__all__ = ["apply_indexer_to_csa_hca", "_tokens_to_super_indices"]

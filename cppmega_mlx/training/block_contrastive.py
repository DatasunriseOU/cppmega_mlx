"""Block-contrastive loss for the graph-supervised DSA indexer.

The indexer should pull a query block's representation toward the blocks it
*truly depends on* (call/type edges) and push it away from confusable negatives:

* **positives**: the dependency (call-edge destination) blocks for the query.
* **hard negatives**: blocks that *look* relevant but are not true dependencies —
  ``same-name / wrong-scope`` lookalikes and ``high-dense-attn-no-edge`` blocks
  (the dense attention attends there strongly yet there is no graph edge). These
  are exactly the cases the graph prior must correct.

The loss is an InfoNCE / softmax-over-candidates contrastive objective on the
indexer scores: for each (query block, positive block) pair, maximise the
positive's indexer score relative to the pooled positive + hard-negative set.

RULE #1: FAIL LOUD on shape / empty-candidate violations. No silent skip of a
query that has positives — if a query has positives we score it or raise.
"""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
import numpy as np

_NEG_INF = -1e9


def hard_negative_blocks(
    dense_attn_blocks: mx.array,
    edge_targets: mx.array,
    *,
    per_query: int,
    name_collision_blocks: Sequence[Sequence[int]] | None = None,
) -> list[list[list[int]]]:
    """Mine hard-negative blocks per (batch, query block).

    Two families, per the recipe:

    * ``high-dense-attn-no-edge``: blocks with the largest dense-attention mass
      that are NOT true edge targets (the indexer must learn these are not deps).
    * ``same-name / wrong-scope``: caller-supplied ``name_collision_blocks[t]``
      lookalike blocks (also excluded if they are true edges).

    Args:
        dense_attn_blocks: ``(B, Tq, Sblk)`` dense-attention block distribution.
        edge_targets: ``(B, Tq, Sblk)`` {0,1} true edge mask.
        per_query: number of high-attn negatives to mine per query.
        name_collision_blocks: optional ``Tq``-long list of lookalike block ids.

    Returns:
        ``negatives[b][t]`` -> sorted list of hard-negative block indices.
    """

    if dense_attn_blocks.ndim != 3:
        raise ValueError(
            f"hard_negative_blocks: dense_attn_blocks must be (B,Tq,Sblk), got "
            f"{tuple(dense_attn_blocks.shape)}"
        )
    if tuple(edge_targets.shape) != tuple(dense_attn_blocks.shape):
        raise ValueError(
            f"hard_negative_blocks: edge_targets {tuple(edge_targets.shape)} must "
            f"match dense_attn_blocks {tuple(dense_attn_blocks.shape)}"
        )
    if per_query < 0:
        raise ValueError(f"hard_negative_blocks: per_query must be >=0, got {per_query}")
    B, Tq, Sblk = (int(x) for x in dense_attn_blocks.shape)
    dense = np.asarray(dense_attn_blocks.astype(mx.float32))
    edges = np.asarray(edge_targets) > 0
    out: list[list[list[int]]] = []
    for b in range(B):
        per_t: list[list[int]] = []
        for t in range(Tq):
            negs: set[int] = set()
            # high-dense-attn-no-edge
            mass = dense[b, t].copy()
            mass[edges[b, t]] = -np.inf
            mass[t] = -np.inf  # never self
            if per_query > 0:
                order = np.argsort(-mass)
                for blk in order[:per_query].tolist():
                    if np.isfinite(mass[blk]):
                        negs.add(int(blk))
            # same-name / wrong-scope lookalikes
            if name_collision_blocks is not None:
                if len(name_collision_blocks) != Tq:
                    raise ValueError(
                        "hard_negative_blocks: name_collision_blocks must have one "
                        f"entry per query ({Tq}), got {len(name_collision_blocks)}"
                    )
                for blk in name_collision_blocks[t]:
                    bi = int(blk)
                    if not (0 <= bi < Sblk):
                        raise ValueError(
                            f"hard_negative_blocks: collision block {bi} out of range"
                        )
                    if not edges[b, t, bi] and bi != t:
                        negs.add(bi)
            per_t.append(sorted(negs))
        out.append(per_t)
    return out


def block_contrastive_loss(
    indexer_scores: mx.array,
    edge_targets: mx.array,
    negatives: Sequence[Sequence[Sequence[int]]],
    *,
    temperature: float = 1.0,
    loss_coeff: float = 1.0,
) -> mx.array:
    """InfoNCE block-contrastive loss over (query, positive dep, hard negs).

    For each query block ``t`` with positive set ``P`` (true edge blocks) and a
    hard-negative set ``N`` (from :func:`hard_negative_blocks`), the candidate
    pool is ``P ∪ N``. For each positive ``p`` the loss is
    ``-log softmax(score_p / tau)`` over the pool. Averaged over all positives.

    Args:
        indexer_scores: ``(B, Tq, Sblk)`` indexer logits.
        edge_targets: ``(B, Tq, Sblk)`` {0,1} positives.
        negatives: ``negatives[b][t]`` hard-negative block lists.
        temperature: softmax temperature ``tau``.
        loss_coeff: scalar multiplier.

    Returns:
        0-d fp32 scalar. Raises if NO query has any positive (degenerate batch).
    """

    if indexer_scores.ndim != 3:
        raise ValueError(
            f"block_contrastive_loss: indexer_scores must be (B,Tq,Sblk), got "
            f"{tuple(indexer_scores.shape)}"
        )
    if tuple(edge_targets.shape) != tuple(indexer_scores.shape):
        raise ValueError(
            f"block_contrastive_loss: edge_targets {tuple(edge_targets.shape)} must "
            f"match indexer_scores {tuple(indexer_scores.shape)}"
        )
    if temperature <= 0:
        raise ValueError(f"block_contrastive_loss: temperature must be >0, got {temperature}")
    B, Tq, Sblk = (int(x) for x in indexer_scores.shape)
    if len(negatives) != B:
        raise ValueError(
            f"block_contrastive_loss: negatives outer length {len(negatives)} must "
            f"match batch {B}"
        )
    logits_all = indexer_scores.astype(mx.float32) / float(temperature)
    edges = np.asarray(edge_targets) > 0

    per_positive_losses: list[mx.array] = []
    for b in range(B):
        if len(negatives[b]) != Tq:
            raise ValueError(
                f"block_contrastive_loss: negatives[{b}] length {len(negatives[b])} "
                f"must match Tq {Tq}"
            )
        for t in range(Tq):
            pos_blocks = np.nonzero(edges[b, t])[0].tolist()
            if not pos_blocks:
                continue
            neg_blocks = [int(n) for n in negatives[b][t]]
            pool = sorted(set(pos_blocks) | set(neg_blocks))
            pool_idx = mx.array(np.asarray(pool, dtype=np.int32))
            pool_logits = logits_all[b, t][pool_idx]  # (|pool|,)
            log_denom = mx.logsumexp(pool_logits, axis=-1)  # scalar
            pos_pos = [pool.index(p) for p in pos_blocks]
            pos_idx = mx.array(np.asarray(pos_pos, dtype=np.int32))
            pos_logits = logits_all[b, t][mx.array(np.asarray(pos_blocks, dtype=np.int32))]
            per_positive_losses.append(mx.mean(log_denom - pos_logits))
    if not per_positive_losses:
        raise ValueError(
            "block_contrastive_loss: no query block has any positive edge; cannot "
            "form a contrastive pair"
        )
    stacked = mx.stack(per_positive_losses)
    return mx.mean(stacked) * float(loss_coeff)


__all__ = [
    "hard_negative_blocks",
    "block_contrastive_loss",
]

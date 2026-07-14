"""Loss terms + metrics for the graph-supervised DSA dependency indexer.

The total indexer objective is

    L = KL(dense_attn_blocks || softmax(indexer))         # warm-up to dense
      + lambda_bce * BCE(indexer, call_edge_block_pairs)   # graph supervision
      + lambda_cov * coverage_hinge(top-k, true edges)     # NSA coverage

plus the block-contrastive term (see :mod:`block_contrastive`).

Each term is a separate, composable function so a training loop can schedule
them (e.g. anneal the KL warm-up down while ramping the BCE supervision up).

* **KL warm-up** distills the dense attention distribution into the indexer so
  the cheap indexer learns to mimic full attention early (DeepSeek DSA recipe).
* **BCE graph supervision** is GraphCodeBERT-style edge supervision: the indexer
  logits at true call-edge block pairs should be high, non-edges low.
* **Coverage hinge** penalises any true call-edge block that the top-k selection
  *misses*, guaranteeing the dependency is reachable through the sparse route.

Metrics: ``recall_at_k`` for call edges and ``dense_attn_topk_overlap``.

RULE #1: every function FAILS LOUD with WHERE + WHAT on shape violations. No
silent fallback, no clamping to hide a bad shape.
"""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
import numpy as np

from cppmega_mlx.data.batch import batch_values_are_prevalidated

_EPS = 1e-9
_NEG_INF = -1e9


def _check_block_scores(scores: mx.array, *, where: str) -> tuple[int, int, int]:
    if scores.ndim != 3:
        raise ValueError(
            f"{where}: block scores must be (B, Tq, Sblk), got {tuple(scores.shape)}"
        )
    return (int(scores.shape[0]), int(scores.shape[1]), int(scores.shape[2]))


def indexer_kl_warmup_loss(
    dense_attn_blocks: mx.array,
    indexer_scores: mx.array,
    *,
    mask: mx.array | None = None,
    loss_coeff: float = 1.0,
) -> mx.array:
    """KL(dense_attn_blocks || softmax(indexer_scores)) averaged over queries.

    Args:
        dense_attn_blocks: ``(B, Tq, Sblk)`` block-aggregated dense-attention
            probability distribution (rows sum to 1; the teacher ``p``).
        indexer_scores: ``(B, Tq, Sblk)`` indexer logits (the student ``q``;
            ``-inf`` entries are masked out of the softmax).
        mask: optional ``(B, Tq)`` 1/0 mask of queries that contribute.
        loss_coeff: scalar multiplier.

    Returns:
        0-d fp32 scalar ``mean_q KL(p_q || q_q) * loss_coeff``.
    """

    B, Tq, Sblk = _check_block_scores(indexer_scores, where="indexer_kl_warmup_loss")
    if tuple(dense_attn_blocks.shape) != (B, Tq, Sblk):
        raise ValueError(
            f"indexer_kl_warmup_loss: dense_attn_blocks {tuple(dense_attn_blocks.shape)}"
            f" must match indexer_scores ({B},{Tq},{Sblk})"
        )
    p = dense_attn_blocks.astype(mx.float32)
    q_log = _masked_log_softmax(indexer_scores.astype(mx.float32))
    log_p = mx.log(p + _EPS)
    per_pos = mx.sum(p * (log_p - q_log), axis=-1)  # (B, Tq)
    return _reduce_per_query(per_pos, mask) * float(loss_coeff)


def _masked_log_softmax(logits: mx.array) -> mx.array:
    """log-softmax over the last axis, treating ``<= _NEG_INF/2`` as masked."""

    valid = logits > (_NEG_INF / 2.0)
    neg = mx.array(_NEG_INF, dtype=mx.float32)
    safe = mx.where(valid, logits, neg)
    m = mx.max(safe, axis=-1, keepdims=True)
    m = mx.where(m > (_NEG_INF / 2.0), m, mx.zeros_like(m))
    shifted = safe - m
    exp = mx.where(valid, mx.exp(shifted), mx.zeros_like(shifted))
    denom = mx.sum(exp, axis=-1, keepdims=True)
    safe_denom = mx.where(denom > 0, denom, mx.ones_like(denom))
    return shifted - mx.log(safe_denom)


def _reduce_per_query(per_pos: mx.array, mask: mx.array | None) -> mx.array:
    if mask is None:
        return mx.mean(per_pos)
    if tuple(mask.shape) != tuple(per_pos.shape):
        raise ValueError(
            f"mask shape {tuple(mask.shape)} must match per-query "
            f"{tuple(per_pos.shape)}"
        )
    m = mask.astype(mx.float32)
    denom = mx.maximum(mx.sum(m), mx.array(1.0, dtype=mx.float32))
    return mx.sum(per_pos * m) / denom


def indexer_edge_bce_loss(
    indexer_scores: mx.array,
    edge_targets: mx.array,
    *,
    pair_mask: mx.array | None = None,
    pos_weight: float = 1.0,
    loss_coeff: float = 1.0,
) -> mx.array:
    """Binary cross-entropy of the indexer on true call-edge block pairs.

    Args:
        indexer_scores: ``(B, Tq, Sblk)`` indexer logits (``-inf`` => skipped).
        edge_targets: ``(B, Tq, Sblk)`` {0,1} targets — 1 at true call-edge
            block pairs, 0 elsewhere.
        pos_weight: up-weights positive (edge) pairs (graphs are sparse).
        loss_coeff: scalar multiplier.

    Returns:
        0-d fp32 scalar mean BCE over the *valid* (non-masked) pairs.
    """

    B, Tq, Sblk = _check_block_scores(indexer_scores, where="indexer_edge_bce_loss")
    if tuple(edge_targets.shape) != (B, Tq, Sblk):
        raise ValueError(
            f"indexer_edge_bce_loss: edge_targets {tuple(edge_targets.shape)} must "
            f"match indexer_scores ({B},{Tq},{Sblk})"
        )
    logits = indexer_scores.astype(mx.float32)
    valid = (logits > (_NEG_INF / 2.0)).astype(mx.float32)
    targets = edge_targets.astype(mx.float32)
    if not batch_values_are_prevalidated() and bool(
        mx.any((targets != 0) & (targets != 1)).item()
    ):
        raise ValueError("indexer_edge_bce_loss: edge_targets must contain only 0/1")
    if pair_mask is not None:
        if tuple(pair_mask.shape) != (B, Tq, Sblk):
            raise ValueError(
                f"indexer_edge_bce_loss: pair_mask {tuple(pair_mask.shape)} must "
                f"match indexer_scores ({B},{Tq},{Sblk})"
            )
        mask = pair_mask.astype(mx.float32)
        if not batch_values_are_prevalidated() and bool(
            mx.any((mask != 0) & (mask != 1)).item()
        ):
            raise ValueError("indexer_edge_bce_loss: pair_mask must contain only 0/1")
        if not batch_values_are_prevalidated() and bool(
            mx.any((targets > 0) & (mask <= 0)).item()
        ):
            raise ValueError(
                "indexer_edge_bce_loss: positive edge outside pair_mask"
            )
        valid = valid * mask
    # numerically-stable BCE-with-logits, weighted on positives.
    max0 = mx.maximum(logits, mx.zeros_like(logits))
    log1pexp = mx.log1p(mx.exp(-mx.abs(logits)))
    bce = max0 - logits * targets + log1pexp
    weight = mx.where(targets > 0, mx.array(float(pos_weight), dtype=mx.float32), mx.array(1.0, dtype=mx.float32))
    bce = bce * weight * valid
    denom = mx.maximum(mx.sum(valid * weight), mx.array(1.0, dtype=mx.float32))
    return (mx.sum(bce) / denom) * float(loss_coeff)


def indexer_coverage_hinge_loss(
    indexer_scores: mx.array,
    edge_targets: mx.array,
    *,
    pair_mask: mx.array | None = None,
    topk: int,
    margin: float = 1.0,
    loss_coeff: float = 1.0,
) -> mx.array:
    """Hinge penalty when a true edge block ranks below the top-k threshold.

    For every query with at least one true edge block, the threshold is the
    score of the k-th highest-scoring block. Any true edge block whose score is
    below ``threshold + margin`` is penalised. This pushes every true call-edge
    block into (or near the boundary of) the top-k selection — i.e. *coverage*.

    Args:
        indexer_scores: ``(B, Tq, Sblk)`` logits.
        edge_targets: ``(B, Tq, Sblk)`` {0,1} true call-edge block pairs.
        topk: the selection budget k.
        margin: hinge margin.
        loss_coeff: scalar multiplier.
    """

    B, Tq, Sblk = _check_block_scores(indexer_scores, where="indexer_coverage_hinge_loss")
    if tuple(edge_targets.shape) != (B, Tq, Sblk):
        raise ValueError(
            f"indexer_coverage_hinge_loss: edge_targets {tuple(edge_targets.shape)} "
            f"must match indexer_scores ({B},{Tq},{Sblk})"
        )
    if topk < 1:
        raise ValueError(f"indexer_coverage_hinge_loss: topk must be >=1, got {topk}")
    logits = indexer_scores.astype(mx.float32)
    targets = edge_targets.astype(mx.float32)
    if not batch_values_are_prevalidated() and bool(
        mx.any((targets != 0) & (targets != 1)).item()
    ):
        raise ValueError(
            "indexer_coverage_hinge_loss: edge_targets must contain only 0/1"
        )
    if pair_mask is not None:
        if tuple(pair_mask.shape) != (B, Tq, Sblk):
            raise ValueError(
                f"indexer_coverage_hinge_loss: pair_mask "
                f"{tuple(pair_mask.shape)} must match indexer_scores "
                f"({B},{Tq},{Sblk})"
            )
        mask = pair_mask.astype(mx.float32)
        if not batch_values_are_prevalidated() and bool(
            mx.any((mask != 0) & (mask != 1)).item()
        ):
            raise ValueError(
                "indexer_coverage_hinge_loss: pair_mask must contain only 0/1"
            )
        if not batch_values_are_prevalidated() and bool(
            mx.any((targets > 0) & (mask <= 0)).item()
        ):
            raise ValueError(
                "indexer_coverage_hinge_loss: positive edge outside pair_mask"
            )
        logits = mx.where(
            mask > 0,
            logits,
            mx.full(logits.shape, _NEG_INF, dtype=mx.float32),
        )
    # The selection boundary is the highest score that is NOT selected, i.e. the
    # (k+1)-th largest score. A true-edge block is "covered" iff its score
    # strictly exceeds that boundary. We penalise true edges whose score is below
    # ``boundary + margin`` so they get pushed into the top-k. When k >= Sblk
    # everything is selected -> no boundary, no penalty.
    if topk >= Sblk:
        return mx.array(0.0, dtype=mx.float32)
    topk_vals = mx.topk(logits, topk + 1, axis=-1)  # (B, Tq, k+1)
    boundary = mx.min(topk_vals, axis=-1, keepdims=True)  # (B, Tq, 1)
    deficit = mx.maximum(
        boundary + float(margin) - logits, mx.zeros_like(logits)
    )
    penalty = deficit * targets
    denom = mx.maximum(mx.sum(targets), mx.array(1.0, dtype=mx.float32))
    return (mx.sum(penalty) / denom) * float(loss_coeff)


def apply_graph_indexer_bias(
    indexer_scores: mx.array,
    graph_bias: mx.array,
    *,
    beta: mx.array | float = 1.0,
) -> mx.array:
    """Add a fixed graph prior to indexer logits before loss/top-k selection.

    This is the training-side equivalent of the DSA inference formula used by
    ``sparse_mla.lightning_indexer_scores``:

        I_final[b, t, s] = I_neural[b, t, s] + beta * S_graph[b, t, s]

    Masked logits (``<= -1e9/2``) stay masked even when ``graph_bias`` is large,
    so a graph prior cannot resurrect a causally invalid or otherwise forbidden
    block.

    Args:
        indexer_scores: ``(B, Tq, Sblk)`` learned/neural indexer logits.
        graph_bias: ``(Tq, Sblk)``, ``(1, Tq, Sblk)``, or ``(B, Tq, Sblk)``
            fixed route prior from ``code_graph_routes.build_attention_bias``.
        beta: scalar learnable/fixed graph-prior weight.
    """

    B, Tq, Sblk = _check_block_scores(
        indexer_scores, where="apply_graph_indexer_bias"
    )
    if graph_bias.ndim == 2:
        if tuple(graph_bias.shape) != (Tq, Sblk):
            raise ValueError(
                "apply_graph_indexer_bias: 2-D graph_bias must be "
                f"({Tq},{Sblk}), got {tuple(graph_bias.shape)}"
            )
        bias = mx.broadcast_to(graph_bias[None, :, :], (B, Tq, Sblk))
    elif graph_bias.ndim == 3:
        gB, gTq, gSblk = (int(x) for x in graph_bias.shape)
        if (gTq, gSblk) != (Tq, Sblk):
            raise ValueError(
                "apply_graph_indexer_bias: graph_bias trailing shape must be "
                f"({Tq},{Sblk}), got {tuple(graph_bias.shape)}"
            )
        if gB == B:
            bias = graph_bias
        elif gB == 1:
            bias = mx.broadcast_to(graph_bias, (B, Tq, Sblk))
        else:
            raise ValueError(
                "apply_graph_indexer_bias: graph_bias batch must be 1 or "
                f"{B}, got {gB}"
            )
    else:
        raise ValueError(
            "apply_graph_indexer_bias: graph_bias must be 2-D or 3-D, got "
            f"{tuple(graph_bias.shape)}"
        )

    beta_arr = (
        beta
        if isinstance(beta, mx.array)
        else mx.array(float(beta), dtype=mx.float32)
    )
    if beta_arr.ndim != 0:
        raise ValueError(
            "apply_graph_indexer_bias: beta must be a scalar, got "
            f"{tuple(beta_arr.shape)}"
        )
    logits = indexer_scores.astype(mx.float32)
    valid = logits > (_NEG_INF / 2.0)
    biased = logits + beta_arr.astype(mx.float32) * bias.astype(mx.float32)
    return mx.where(valid, biased, mx.array(_NEG_INF, dtype=mx.float32))


def select_graph_biased_topk(
    indexer_scores: mx.array,
    *,
    graph_bias: mx.array | None = None,
    beta: mx.array | float = 1.0,
    topk: int,
    local_window: int = 0,
    num_sinks: int = 0,
    causal: bool = True,
) -> tuple[mx.array, mx.array]:
    """Apply optional graph bias, then select DSA top-k block indices.

    Returns ``(selected_indices, final_scores)``.  The selected indices are the
    same sentinel ``-1`` format as
    :func:`cppmega_mlx.nn.sparse_mla.indexer_topk_indices`.
    """

    final_scores = (
        indexer_scores
        if graph_bias is None
        else apply_graph_indexer_bias(indexer_scores, graph_bias, beta=beta)
    )
    from cppmega_mlx.nn.sparse_mla import indexer_topk_indices

    selected = indexer_topk_indices(
        final_scores,
        topk=topk,
        local_window=local_window,
        num_sinks=num_sinks,
        causal=causal,
    )
    return selected, final_scores


def recall_at_k(
    indexer_scores: mx.array,
    edge_targets: mx.array,
    *,
    topk: int,
) -> float:
    """Fraction of true call-edge block pairs whose block lands in the top-k.

    Averaged over queries that have at least one true edge.
    """

    B, Tq, Sblk = _check_block_scores(indexer_scores, where="recall_at_k")
    if tuple(edge_targets.shape) != (B, Tq, Sblk):
        raise ValueError(
            f"recall_at_k: edge_targets {tuple(edge_targets.shape)} must match "
            f"indexer_scores ({B},{Tq},{Sblk})"
        )
    k = min(int(topk), Sblk)
    logits = np.asarray(indexer_scores.astype(mx.float32))
    targets = np.asarray(edge_targets) > 0
    hits = 0
    total = 0
    for b in range(B):
        for t in range(Tq):
            true_blocks = np.nonzero(targets[b, t])[0]
            if true_blocks.size == 0:
                continue
            row = logits[b, t]
            # top-k indices by score
            if row.size <= k:
                topk_idx = set(np.nonzero(row > (_NEG_INF / 2.0))[0].tolist())
            else:
                part = np.argpartition(-row, k - 1)[:k]
                topk_idx = set(int(i) for i in part if row[i] > (_NEG_INF / 2.0))
            for blk in true_blocks.tolist():
                total += 1
                if blk in topk_idx:
                    hits += 1
    if total == 0:
        raise ValueError("recall_at_k: no true edges present; cannot compute recall")
    return hits / total


def dense_attn_topk_overlap(
    indexer_scores: mx.array,
    dense_attn_blocks: mx.array,
    *,
    topk: int,
) -> float:
    """Mean Jaccard overlap between the indexer's top-k and dense-attn's top-k.

    Measures how well the cheap indexer reproduces where the full attention
    actually puts its mass (the KL warm-up target, as a hard set metric).
    """

    B, Tq, Sblk = _check_block_scores(indexer_scores, where="dense_attn_topk_overlap")
    if tuple(dense_attn_blocks.shape) != (B, Tq, Sblk):
        raise ValueError(
            f"dense_attn_topk_overlap: dense_attn_blocks {tuple(dense_attn_blocks.shape)}"
            f" must match indexer_scores ({B},{Tq},{Sblk})"
        )
    k = min(int(topk), Sblk)
    idx = np.asarray(indexer_scores.astype(mx.float32))
    dense = np.asarray(dense_attn_blocks.astype(mx.float32))

    def _topk_set(row: np.ndarray) -> set[int]:
        if row.size <= k:
            return set(np.nonzero(row > (_NEG_INF / 2.0))[0].tolist())
        part = np.argpartition(-row, k - 1)[:k]
        return set(int(i) for i in part)

    overlaps: list[float] = []
    for b in range(B):
        for t in range(Tq):
            a = _topk_set(idx[b, t])
            c = _topk_set(dense[b, t])
            if not a and not c:
                continue
            union = a | c
            overlaps.append(len(a & c) / max(1, len(union)))
    if not overlaps:
        raise ValueError("dense_attn_topk_overlap: no comparable queries")
    return float(np.mean(overlaps))


def total_indexer_loss(
    indexer_scores: mx.array,
    *,
    dense_attn_blocks: mx.array | None = None,
    edge_targets: mx.array | None = None,
    edge_pair_mask: mx.array | None = None,
    topk: int,
    kl_coeff: float = 1.0,
    bce_coeff: float = 1.0,
    coverage_coeff: float = 1.0,
    pos_weight: float = 1.0,
    margin: float = 1.0,
    query_mask: mx.array | None = None,
) -> tuple[mx.array, dict[str, mx.array]]:
    """Combine the indexer loss terms; returns ``(total, components)``.

    Terms are only included when their target is supplied (KL needs
    ``dense_attn_blocks``; BCE / coverage need ``edge_targets``).
    """

    components: dict[str, mx.array] = {}
    total = mx.array(0.0, dtype=mx.float32)
    if dense_attn_blocks is not None and kl_coeff:
        kl = indexer_kl_warmup_loss(
            dense_attn_blocks, indexer_scores, mask=query_mask, loss_coeff=kl_coeff
        )
        components["kl"] = kl
        total = total + kl
    if edge_targets is not None and bce_coeff:
        bce = indexer_edge_bce_loss(
            indexer_scores,
            edge_targets,
            pair_mask=edge_pair_mask,
            pos_weight=pos_weight,
            loss_coeff=bce_coeff,
        )
        components["bce"] = bce
        total = total + bce
    if edge_targets is not None and coverage_coeff:
        cov = indexer_coverage_hinge_loss(
            indexer_scores,
            edge_targets,
            pair_mask=edge_pair_mask,
            topk=topk,
            margin=margin,
            loss_coeff=coverage_coeff,
        )
        components["coverage"] = cov
        total = total + cov
    components["total"] = total
    return total, components


def edge_targets_from_candidates(
    candidates: Sequence[Sequence[int]],
    *,
    num_blocks: int,
    batch: int = 1,
) -> mx.array:
    """Build a ``(batch, num_blocks, num_blocks)`` {0,1} edge-target tensor.

    ``candidates[t]`` lists the true destination blocks for query block ``t``
    (as produced by ``code_graph_routes.build_block_candidates``). Broadcast
    across ``batch``.
    """

    if len(candidates) != num_blocks:
        raise ValueError(
            f"edge_targets_from_candidates: expected {num_blocks} query rows, got "
            f"{len(candidates)}"
        )
    mat = np.zeros((num_blocks, num_blocks), dtype=np.float32)
    for t, dests in enumerate(candidates):
        for s in dests:
            if not (0 <= int(s) < num_blocks):
                raise ValueError(
                    f"edge_targets_from_candidates: dest block {s} out of range "
                    f"[0,{num_blocks})"
                )
            mat[t, int(s)] = 1.0
    out = np.broadcast_to(mat[None], (batch, num_blocks, num_blocks)).copy()
    return mx.array(out)


__all__ = [
    "indexer_kl_warmup_loss",
    "indexer_edge_bce_loss",
    "indexer_coverage_hinge_loss",
    "apply_graph_indexer_bias",
    "select_graph_biased_topk",
    "recall_at_k",
    "dense_attn_topk_overlap",
    "total_indexer_loss",
    "edge_targets_from_candidates",
]

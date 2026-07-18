"""Tests for the graph-supervised indexer loss terms + metrics."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from cppmega_mlx.training.block_contrastive import (
    block_contrastive_loss,
    hard_negative_blocks,
)
from cppmega_mlx.training.indexer_losses import (
    apply_graph_indexer_bias,
    dense_attn_topk_overlap,
    edge_targets_from_candidates,
    indexer_coverage_hinge_loss,
    indexer_edge_bce_loss,
    indexer_kl_warmup_loss,
    recall_at_k,
    select_graph_biased_topk,
    total_indexer_loss,
)
from cppmega_mlx.data.graph_packet import EdgeIndex, GraphPacket
from cppmega_mlx.nn.code_graph_routes import (
    GraphRouteConfig,
    build_attention_bias,
    build_block_candidates,
)


def test_edge_targets_from_candidates():
    et = edge_targets_from_candidates([[2], [], [], [1]], num_blocks=4, batch=1)
    assert tuple(et.shape) == (1, 4, 4)
    e = np.asarray(et)
    assert e[0, 0, 2] == 1.0
    assert e[0, 3, 1] == 1.0
    assert e.sum() == 2.0


def test_kl_warmup_reduces_divergence_over_steps():
    # The indexer (student) starts random; KL warm-up should pull its softmax
    # toward the fixed dense-attention teacher, lowering KL over steps.
    np.random.seed(0)
    B, Tq, Sblk = 2, 5, 6
    teacher = mx.softmax(
        mx.array(np.random.randn(B, Tq, Sblk).astype(np.float32)), axis=-1
    )
    W = mx.array(np.random.randn(B, Tq, Sblk).astype(np.float32) * 0.1)

    def loss_fn(W):
        return indexer_kl_warmup_loss(teacher, W)

    kl0 = float(loss_fn(W))
    for _ in range(60):
        _, grad = mx.value_and_grad(loss_fn)(W)
        W = W - 1.0 * grad
        mx.eval(W)
    kl1 = float(loss_fn(W))
    assert kl1 < kl0 * 0.25  # divergence collapses substantially
    assert kl1 >= 0.0


def test_recall_at_k_improves_with_graph_supervision():
    # Synthetic graph: true edge blocks must rank above non-edges after training
    # the indexer with BCE + coverage. This is the indexer-recall@k claim.
    np.random.seed(1)
    Tq = Sblk = 6
    W = mx.array(np.random.randn(Tq, Sblk).astype(np.float32) * 0.1)
    et = edge_targets_from_candidates(
        [[3], [4], [], [5], [], [1]], num_blocks=6, batch=1
    )

    def loss_fn(W):
        s = W[None]
        return indexer_edge_bce_loss(s, et, pos_weight=3.0) + (
            indexer_coverage_hinge_loss(s, et, topk=2)
        )

    r0 = recall_at_k(W[None], et, topk=2)
    for _ in range(80):
        _, grad = mx.value_and_grad(loss_fn)(W)
        W = W - 0.5 * grad
        mx.eval(W)
    r1 = recall_at_k(W[None], et, topk=2)
    assert r1 > r0
    assert r1 == 1.0  # every true call-edge block lands in the top-k


def test_coverage_hinge_zero_when_edges_in_topk():
    # Construct logits where the true edge IS the top scorer -> no coverage loss.
    s = mx.array(
        np.array([[[5.0, 0.0, 0.0, 0.0]]], dtype=np.float32)
    )  # (1,1,4)
    # one query row (matches s), 4 dest blocks; true edge at block 0.
    et = mx.array(np.array([[[1.0, 0.0, 0.0, 0.0]]], dtype=np.float32))
    cov = indexer_coverage_hinge_loss(s, et, topk=1, margin=1.0)
    assert float(cov) == 0.0


def test_graph_pair_mask_rejects_positive_edge_outside_valid_pairs() -> None:
    scores = mx.zeros((1, 2, 2), dtype=mx.float32)
    targets = mx.zeros_like(scores).at[0, 1, 1].add(1.0)
    pair_mask = mx.zeros_like(scores)

    with pytest.raises(ValueError, match="positive edge outside pair_mask"):
        indexer_edge_bce_loss(scores, targets, pair_mask=pair_mask)


def test_coverage_hinge_excludes_invalid_pairs_from_topk_boundary() -> None:
    scores = mx.array([[[2.0, 1.0, 100.0, 100.0]]], dtype=mx.float32)
    targets = mx.array([[[1.0, 0.0, 0.0, 0.0]]], dtype=mx.float32)
    pair_mask = mx.array([[[1.0, 1.0, 0.0, 0.0]]], dtype=mx.float32)

    loss = indexer_coverage_hinge_loss(
        scores,
        targets,
        pair_mask=pair_mask,
        topk=1,
        margin=1.0,
    )

    assert float(loss.item()) == 0.0


def test_apply_graph_indexer_bias_broadcasts_and_preserves_masks():
    scores = mx.array(
        np.array(
            [
                [
                    [0.0, 1.0, -1e9],
                    [0.0, 1.0, 2.0],
                    [0.0, 1.0, 2.0],
                ],
                [
                    [3.0, 2.0, -1e9],
                    [3.0, 2.0, 1.0],
                    [3.0, 2.0, 1.0],
                ],
            ],
            dtype=np.float32,
        )
    )
    graph_bias = mx.array(
        np.array(
            [
                [0.0, 0.0, 100.0],
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ],
            dtype=np.float32,
        )
    )

    biased = apply_graph_indexer_bias(scores, graph_bias, beta=10.0)
    out = np.asarray(biased)
    assert tuple(biased.shape) == (2, 3, 3)
    assert out[0, 1, 0] == 10.0
    assert out[1, 2, 1] == 22.0
    # A positive graph prior must not resurrect causally/explicitly masked slots.
    assert out[0, 0, 2] < -1e8
    assert out[1, 0, 2] < -1e8


def test_select_graph_biased_topk_uses_code_graph_routes_for_recall():
    packet = GraphPacket(
        edges={
            "call": EdgeIndex.from_pairs(
                [[0, 4], [1, 5], [6, 2]], relation="call", num_nodes=8
            )
        },
        num_nodes=8,
    )
    cfg = GraphRouteConfig(num_blocks=4, relations=("call",), normalize="binary")
    graph_bias = build_attention_bias(packet, config=cfg)
    candidates = build_block_candidates(packet, config=cfg)
    targets = edge_targets_from_candidates(candidates, num_blocks=4, batch=1)

    # Flat neural scores select block 0 by deterministic argsort.  The graph
    # prior should move rows 0 and 3 to their true dependency blocks: 2 and 1.
    scores = mx.zeros((1, 4, 4), dtype=mx.float32)
    selected_off, final_off = select_graph_biased_topk(
        scores, topk=1, causal=False
    )
    selected_on, final_on = select_graph_biased_topk(
        scores, graph_bias=graph_bias, beta=20.0, topk=1, causal=False
    )

    assert recall_at_k(final_off, targets, topk=1) == 0.0
    assert recall_at_k(final_on, targets, topk=1) == 1.0
    off = np.asarray(selected_off)
    on = np.asarray(selected_on)
    assert off[0, 0, 0] == 0
    assert on[0, 0, 0] == 2
    assert on[0, 3, 0] == 1


def test_graph_biased_scores_feed_total_indexer_loss_and_beta_grad():
    packet = GraphPacket(
        edges={
            "call": EdgeIndex.from_pairs(
                [[0, 4], [1, 5], [6, 2]], relation="call", num_nodes=8
            )
        },
        num_nodes=8,
    )
    cfg = GraphRouteConfig(num_blocks=4, relations=("call",), normalize="binary")
    graph_bias = build_attention_bias(packet, config=cfg)
    targets = edge_targets_from_candidates(
        build_block_candidates(packet, config=cfg), num_blocks=4, batch=1
    )
    scores = mx.zeros((1, 4, 4), dtype=mx.float32)

    def loss_fn(beta):
        biased = apply_graph_indexer_bias(scores, graph_bias, beta=beta)
        total, _ = total_indexer_loss(
            biased,
            edge_targets=targets,
            topk=1,
            bce_coeff=1.0,
            coverage_coeff=1.0,
        )
        return total

    beta0 = mx.array(0.0, dtype=mx.float32)
    loss0, grad = mx.value_and_grad(loss_fn)(beta0)
    loss1 = loss_fn(mx.array(10.0, dtype=mx.float32))
    assert np.isfinite(float(loss0))
    assert np.isfinite(float(grad))
    assert float(grad) < 0.0
    assert float(loss1) < float(loss0)


def test_dense_attn_topk_overlap_perfect():
    s = mx.array(np.array([[[3.0, 2.0, 1.0, 0.0]]], dtype=np.float32))
    dense = mx.softmax(s, axis=-1)
    # top-2 of both is {0,1}: perfect overlap.
    assert dense_attn_topk_overlap(s, dense, topk=2) == 1.0


def test_total_indexer_loss_sums_components():
    np.random.seed(2)
    B, Tq, Sblk = 1, 4, 4
    s = mx.array(np.random.randn(B, Tq, Sblk).astype(np.float32))
    dense = mx.softmax(
        mx.array(np.random.randn(B, Tq, Sblk).astype(np.float32)), axis=-1
    )
    et = edge_targets_from_candidates([[2], [], [], [1]], num_blocks=4, batch=1)
    total, comps = total_indexer_loss(
        s, dense_attn_blocks=dense, edge_targets=et, topk=2
    )
    assert set(comps) == {"kl", "bce", "coverage", "total"}
    recon = float(comps["kl"]) + float(comps["bce"]) + float(comps["coverage"])
    assert abs(float(total) - recon) < 1e-5


def test_block_contrastive_separates_positive_from_hard_negatives():
    # high-dense-attn-no-edge block must be down-ranked vs the true dependency.
    np.random.seed(3)
    B, Tq, Sblk = 1, 5, 5
    dense = mx.softmax(
        mx.array(np.random.randn(B, Tq, Sblk).astype(np.float32)), axis=-1
    )
    # 5 query rows (== num_blocks); true deps for rows 0,1; rest have none.
    et = edge_targets_from_candidates(
        [[2], [4], [], [], []], num_blocks=5, batch=1
    )
    W = mx.array(np.random.randn(B, Tq, Sblk).astype(np.float32) * 0.1)
    negs = hard_negative_blocks(dense, et, per_query=2)

    def loss_fn(W):
        return block_contrastive_loss(W, et, negs, temperature=1.0)

    l0 = float(loss_fn(W))
    for _ in range(50):
        _, grad = mx.value_and_grad(loss_fn)(W)
        W = W - 0.5 * grad
        mx.eval(W)
    l1 = float(loss_fn(W))
    assert l1 < l0  # positives pulled above hard negatives


def test_shape_mismatch_raises():
    s = mx.zeros((1, 3, 4))
    bad = mx.zeros((1, 3, 5))
    with pytest.raises(ValueError):
        indexer_edge_bce_loss(s, bad)
    with pytest.raises(ValueError):
        indexer_kl_warmup_loss(bad, s)

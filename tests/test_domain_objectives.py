from __future__ import annotations

import mlx.core as mx

from cppmega_mlx.training.domain_objectives import (
    cross_domain_retrieval_ranking_loss,
    domain_edge_bce_loss,
    opener_domain_classification_loss,
)


def test_opener_domain_classification_loss_rewards_correct_domain():
    targets = mx.array([[2, 0, 3]], dtype=mx.int32)
    opener_mask = mx.array([[1, 0, 1]], dtype=mx.int32)
    good = mx.array(
        [
            [
                [0.0, 0.0, 5.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 5.0],
            ]
        ],
        dtype=mx.float32,
    )
    bad = mx.zeros_like(good)

    assert float(opener_domain_classification_loss(good, targets, opener_mask).item()) < float(
        opener_domain_classification_loss(bad, targets, opener_mask).item()
    )


def test_domain_edge_bce_loss_is_finite_and_respects_mask():
    logits = mx.array([[[2.0, -2.0], [-1.0, 1.0]]], dtype=mx.float32)
    targets = mx.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=mx.float32)
    mask = mx.array([[[1, 1], [1, 0]]], dtype=mx.int32)

    loss = domain_edge_bce_loss(logits, targets, edge_mask=mask, pos_weight=2.0)

    assert float(loss.item()) >= 0.0


def test_cross_domain_retrieval_ranking_loss_prefers_positive_over_negative():
    positives = mx.array([[[1]]], dtype=mx.int32)
    negatives = mx.array([[[2]]], dtype=mx.int32)
    good_scores = mx.array([[[0.0, 5.0, 0.0]]], dtype=mx.float32)
    bad_scores = mx.array([[[0.0, 0.0, 5.0]]], dtype=mx.float32)

    good = cross_domain_retrieval_ranking_loss(good_scores, positives, negatives)
    bad = cross_domain_retrieval_ranking_loss(bad_scores, positives, negatives)

    assert float(good.item()) < float(bad.item())

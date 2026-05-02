from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from cppmega_mlx.models.tiny_lm import TinyLM, TinyLMConfig
from cppmega_mlx.training.mtp import (
    DEFAULT_MTP_DECAY,
    DEFAULT_MTP_DEPTH,
    DEFAULT_MTP_LAMBDA,
    MTPLossConfig,
    MinimalMTPHead,
    compute_mtp_step_weights,
    compute_weighted_mtp_loss,
    mtp_loss_for_model,
    next_token_and_mtp_loss,
    roll_and_mask_mtp_ids,
    roll_and_mask_mtp_labels,
)


class CountingSharedBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def __call__(self, hidden_states: mx.array) -> mx.array:
        self.calls += 1
        return hidden_states


def _tiny_model() -> TinyLM:
    return TinyLM(
        TinyLMConfig(
            vocab_size=16,
            hidden_size=8,
            num_layers=1,
            num_heads=2,
            ffn_hidden_size=16,
            max_seq_length=8,
            structure_vocab_size=8,
        )
    )


def _flat_grads(grads: dict[str, object]) -> dict[str, np.ndarray]:
    mx.eval(grads)
    return {name: np.array(value) for name, value in tree_flatten(grads)}


def test_mtp_defaults_match_gb10_k2_contract() -> None:
    config = MTPLossConfig()

    assert config.depth == DEFAULT_MTP_DEPTH == 2
    assert config.decay == DEFAULT_MTP_DECAY == 0.6
    assert config.loss_weight == DEFAULT_MTP_LAMBDA == 0.3

    weights = compute_mtp_step_weights(config.depth, config.decay)
    mx.eval(weights)

    np.testing.assert_allclose(np.array(weights), np.array([0.625, 0.375]), rtol=1e-7)
    assert math.isclose(float(weights.sum().item()), 1.0, rel_tol=1e-6)


def test_roll_and_mask_static_labels_and_teacher_ids() -> None:
    targets = mx.array([[10, 11, 12, 13], [20, 21, 22, 23]], dtype=mx.int32)

    labels = roll_and_mask_mtp_labels(targets, depth=2)
    teacher_ids = roll_and_mask_mtp_ids(targets, depth=2)
    mx.eval(*labels, *teacher_ids)

    np.testing.assert_array_equal(
        np.array(labels[0]),
        np.array([[11, 12, 13, -1], [21, 22, 23, -1]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.array(labels[1]),
        np.array([[12, 13, -1, -1], [22, 23, -1, -1]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.array(teacher_ids[0]),
        np.array([[11, 12, 13, 0], [21, 22, 23, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.array(teacher_ids[1]),
        np.array([[12, 13, 0, 0], [22, 23, 0, 0]], dtype=np.int32),
    )
    assert labels[0].shape == targets.shape
    assert labels[1].shape == targets.shape
    assert teacher_ids[0].shape == targets.shape
    assert teacher_ids[1].shape == targets.shape


def test_weighted_mtp_loss_and_lambda_composition() -> None:
    per_depth = (
        mx.array(2.0, dtype=mx.float32),
        mx.array(4.0, dtype=mx.float32),
    )

    mtp_loss, weights = compute_weighted_mtp_loss(per_depth, decay=0.6)
    total = next_token_and_mtp_loss(
        mx.array(5.0, dtype=mx.float32),
        mtp_loss,
        loss_weight=0.3,
    )
    mx.eval(mtp_loss, total, weights)

    assert math.isclose(float(mtp_loss.item()), 2.75, rel_tol=1e-6)
    assert math.isclose(float(total.item()), 5.825, rel_tol=1e-6)
    np.testing.assert_allclose(np.array(weights), np.array([0.625, 0.375]), rtol=1e-7)


def test_minimal_mtp_head_tracks_per_depth_losses_and_keeps_inference_logits_sane() -> None:
    model = _tiny_model()
    inputs = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
    targets = mx.array([[2, 3, 4, 5, 6]], dtype=mx.int32)

    before_logits = model(inputs)
    hidden_states = model.token_embedding(targets)
    head = MinimalMTPHead(model.token_embedding, model.lm_head)
    mtp_loss, per_depth, weights = head.loss(hidden_states, targets)
    after_logits = model(inputs)
    mx.eval(before_logits, after_logits, mtp_loss, weights, *per_depth)

    assert len(per_depth) == 2
    assert before_logits.shape == (1, 5, model.config.vocab_size)
    assert after_logits.shape == before_logits.shape
    np.testing.assert_allclose(np.array(after_logits), np.array(before_logits), rtol=0, atol=0)
    assert all(math.isfinite(float(loss.item())) for loss in per_depth)
    assert all(float(loss.item()) > 0 for loss in per_depth)
    assert math.isfinite(float(mtp_loss.item()))
    assert float(mtp_loss.item()) > 0
    np.testing.assert_allclose(np.array(weights), np.array([0.625, 0.375]), rtol=1e-7)


def test_minimal_mtp_head_recurs_one_shared_block_for_k2() -> None:
    model = _tiny_model()
    targets = mx.array([[2, 3, 4, 5, 6]], dtype=mx.int32)
    hidden_states = model.token_embedding(targets)
    shared_block = CountingSharedBlock()
    head = MinimalMTPHead(
        model.token_embedding,
        model.lm_head,
        shared_block=shared_block,
    )

    logits_by_depth = head(hidden_states, targets)
    mx.eval(*logits_by_depth)

    assert shared_block.calls == 2
    assert len(logits_by_depth) == 2
    assert logits_by_depth[0].shape == (1, 5, model.config.vocab_size)
    assert logits_by_depth[1].shape == (1, 5, model.config.vocab_size)


def test_mtp_loss_for_model_reports_total_and_per_depth_metrics() -> None:
    model = _tiny_model()
    targets = mx.array([[2, 3, 4, 5, 6], [7, 8, 9, 10, 11]], dtype=mx.int32)

    metrics = mtp_loss_for_model(model, targets)
    mx.eval(
        metrics.next_token_loss,
        metrics.mtp_loss,
        metrics.total_loss,
        metrics.depth_weights,
        *metrics.per_depth_losses,
    )

    assert len(metrics.per_depth_losses) == 2
    assert metrics.loss_weight == 0.3
    assert math.isclose(
        float(metrics.total_loss.item()),
        0.3 * float(metrics.mtp_loss.item()),
        rel_tol=1e-6,
    )
    assert math.isclose(float(metrics.depth_weights.sum().item()), 1.0, rel_tol=1e-6)


def test_mtp_head_supports_gradients_to_head_parameters() -> None:
    model = _tiny_model()
    targets = mx.array([[2, 3, 4, 5, 6], [7, 8, 9, 10, 11]], dtype=mx.int32)
    head = MinimalMTPHead(model.token_embedding, model.lm_head)

    def loss_fn(module: MinimalMTPHead) -> mx.array:
        hidden_states = module.token_embedding(targets)
        loss, _, _ = module.loss(hidden_states, targets)
        return loss

    loss, grads = nn.value_and_grad(head, loss_fn)(head)
    grad_arrays = _flat_grads(grads)
    mx.eval(loss)

    assert math.isfinite(float(loss.item()))
    assert "token_embedding.weight" in grad_arrays
    assert "lm_head.weight" in grad_arrays
    assert np.max(np.abs(grad_arrays["token_embedding.weight"])) > 0
    assert np.max(np.abs(grad_arrays["lm_head.weight"])) > 0

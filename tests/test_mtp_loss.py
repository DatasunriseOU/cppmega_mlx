from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from cppmega_mlx.data.batch import synthetic_token_batch
from cppmega_mlx.models.hybrid_lm import HybridTinyConfig, HybridTinyLM
from cppmega_mlx.models.tiny_lm import TinyLM, TinyLMConfig
from cppmega_mlx.training.loss import (
    next_token_cross_entropy,
    next_token_cross_entropy_mtp_loss,
    next_token_cross_entropy_with_mtp,
)
from cppmega_mlx.training.loop import make_adamw, one_step_train
from cppmega_mlx.training.mtp import (
    DEFAULT_MTP_DECAY,
    DEFAULT_MTP_DEPTH,
    DEFAULT_MTP_LAMBDA,
    MTPLossConfig,
    MinimalMTPHead,
    attach_mtp_head,
    compute_mtp_step_weights,
    compute_weighted_mtp_loss,
    get_or_attach_mtp_head,
    mtp_cross_entropy_from_logits,
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


def _hybrid_tiny_model() -> HybridTinyLM:
    return HybridTinyLM(
        HybridTinyConfig(
            vocab_size=16,
            hidden_size=8,
            pattern="A",
            depth=1,
            dsa_a_layer_ranks=(0,),
            num_attention_heads=2,
            max_seq_length=8,
            structure_components="all",
            structure_num_categories=16,
            structure_max_dep_level=16,
            structure_max_ast_depth=16,
            structure_max_sibling_index=16,
            structure_num_node_types=16,
            structure_bottleneck_dim=4,
            mamba_expand=1,
            mamba_head_dim=4,
            mamba_state_dim=4,
            mamba_groups=1,
            mamba_chunk_size=4,
            moe_num_experts=2,
            moe_top_k=1,
            moe_expert_hidden_size=16,
            moe_shared_expert_hidden_size=8,
            m2rnn_k_head_dim=2,
            m2rnn_v_head_dim=2,
            m2rnn_num_v_heads=1,
            m2rnn_num_f_heads=1,
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


def test_roll_and_mask_respects_packed_document_boundaries() -> None:
    targets = mx.array([[2, 3, 4, 5]], dtype=mx.int32)
    document_ids = mx.array([[0, 0, 1, 1]], dtype=mx.int32)

    labels = roll_and_mask_mtp_labels(targets, depth=2, document_ids=document_ids)
    teacher_ids = roll_and_mask_mtp_ids(targets, depth=2, document_ids=document_ids)
    mx.eval(*labels, *teacher_ids)

    np.testing.assert_array_equal(
        np.array(labels[0]),
        np.array([[3, -1, 5, -1]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.array(labels[1]),
        np.array([[-1, -1, -1, -1]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.array(teacher_ids[0]),
        np.array([[3, 0, 5, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.array(teacher_ids[1]),
        np.array([[0, 0, 0, 0]], dtype=np.int32),
    )


def test_roll_and_mask_rejects_mismatched_document_ids_shape() -> None:
    targets = mx.array([[2, 3, 4, 5]], dtype=mx.int32)
    bad_document_ids = mx.array([[0, 0, 1]], dtype=mx.int32)

    with pytest.raises(ValueError, match="document_ids shape"):
        roll_and_mask_mtp_labels(targets, depth=2, document_ids=bad_document_ids)
    with pytest.raises(ValueError, match="document_ids shape"):
        roll_and_mask_mtp_ids(targets, depth=2, document_ids=bad_document_ids)


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


def test_minimal_mtp_head_aliases_main_embedding_and_lm_head_without_copy() -> None:
    model = _tiny_model()
    head = MinimalMTPHead(model.token_embedding, model.lm_head)

    assert head.token_embedding is model.token_embedding
    assert head.lm_head is model.lm_head
    assert head.token_embedding.weight is model.token_embedding.weight
    assert head.lm_head.weight is model.lm_head.weight


def test_attach_mtp_head_persists_model_owned_state_and_reuses_aliases() -> None:
    model = _tiny_model()
    head = attach_mtp_head(model)

    assert model.mtp_head is head
    assert get_or_attach_mtp_head(model) is head
    assert head.token_embedding is model.token_embedding
    assert head.lm_head is model.lm_head
    assert head.token_embedding.weight is model.token_embedding.weight
    assert head.lm_head.weight is model.lm_head.weight


def test_get_or_attach_mtp_head_rejects_config_drift_after_attachment() -> None:
    model = _tiny_model()
    head = attach_mtp_head(model, config=MTPLossConfig(depth=2, decay=0.6, loss_weight=0.3))

    assert get_or_attach_mtp_head(model, config=head.config) is head
    try:
        get_or_attach_mtp_head(model, config=MTPLossConfig(depth=3, decay=0.6, loss_weight=0.3))
    except ValueError as exc:
        assert "config does not match" in str(exc)
    else:  # pragma: no cover - keeps the regression failure explicit.
        raise AssertionError("MTP head config drift must fail closed")


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


def test_training_loss_with_mtp_composes_ntp_lambda_mtp_and_metrics() -> None:
    model = _tiny_model()
    batch = mx.array([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]], dtype=mx.int32)

    default_loss, default_ntokens = next_token_cross_entropy(model, batch)
    total_loss, ntokens, metrics = next_token_cross_entropy_with_mtp(model, batch)
    mx.eval(
        default_loss,
        default_ntokens,
        total_loss,
        ntokens,
        metrics.next_token_loss,
        metrics.mtp_loss,
        metrics.total_loss,
        metrics.depth_weights,
        *metrics.per_depth_losses,
    )

    assert len(metrics.per_depth_losses) == 2
    assert metrics.loss_weight == DEFAULT_MTP_LAMBDA == 0.3
    assert int(ntokens.item()) == int(default_ntokens.item())
    assert math.isclose(
        float(metrics.next_token_loss.item()),
        float(default_loss.item()),
        rel_tol=1e-6,
    )
    assert math.isclose(
        float(total_loss.item()),
        float(metrics.total_loss.item()),
        rel_tol=1e-6,
    )
    assert math.isclose(
        float(total_loss.item()),
        float(default_loss.item()) + 0.3 * float(metrics.mtp_loss.item()),
        rel_tol=1e-6,
    )
    np.testing.assert_allclose(
        np.array(metrics.depth_weights),
        np.array([0.625, 0.375]),
        rtol=1e-7,
    )


def test_training_loss_with_mtp_uses_decoder_hidden_states_not_embedding_proxy() -> None:
    model = _tiny_model()
    batch = mx.array([[1, 2, 3, 4, 5, 6]], dtype=mx.int32)

    _, _, metrics = next_token_cross_entropy_with_mtp(model, batch)
    proxy_head = MinimalMTPHead(model.token_embedding, model.lm_head)
    proxy_hidden = model.token_embedding(batch[:, :-1])
    proxy_mtp_loss, _, _ = proxy_head.loss(proxy_hidden, batch[:, 1:])
    mx.eval(metrics.mtp_loss, proxy_mtp_loss)

    assert not math.isclose(
        float(metrics.mtp_loss.item()),
        float(proxy_mtp_loss.item()),
        rel_tol=1e-5,
        abs_tol=1e-5,
    )


def test_training_loss_with_mtp_threads_hybrid_structure_hidden_states() -> None:
    mx.random.seed(91)
    model = _hybrid_tiny_model()
    batch = synthetic_token_batch(
        batch_size=1,
        seq_length=6,
        vocab_size=model.config.vocab_size,
        seed=92,
        include_structure=True,
    )
    model.structure_embedding.stacked_emb.weight = mx.ones_like(
        model.structure_embedding.stacked_emb.weight
    )
    model.structure_embedding.up_proj.weight = mx.ones_like(
        model.structure_embedding.up_proj.weight
    )
    without_structure = batch.tokens
    with_structure = batch.as_dict()

    _, _, plain_metrics = next_token_cross_entropy_with_mtp(model, without_structure)
    _, _, structured_metrics = next_token_cross_entropy_with_mtp(model, with_structure)
    mx.eval(plain_metrics.mtp_loss, structured_metrics.mtp_loss)

    assert not math.isclose(
        float(plain_metrics.mtp_loss.item()),
        float(structured_metrics.mtp_loss.item()),
        rel_tol=1e-5,
        abs_tol=1e-5,
    )


def test_training_loss_with_mtp_masks_packed_document_boundaries() -> None:
    mx.random.seed(121)
    model = _hybrid_tiny_model()
    tokens = mx.array([[1, 2, 3, 4, 5, 6]], dtype=mx.int32)
    packed_document_ids = mx.array([[0, 0, 0, 1, 1, 1]], dtype=mx.int32)

    packed_total, _, packed_metrics = next_token_cross_entropy_with_mtp(
        model,
        {
            "tokens": tokens,
            "document_ids": packed_document_ids,
        },
    )
    unpacked_total, _, unpacked_metrics = next_token_cross_entropy_with_mtp(model, tokens)
    mx.eval(
        packed_total,
        unpacked_total,
        packed_metrics.next_token_loss,
        packed_metrics.mtp_loss,
        packed_metrics.total_loss,
        *packed_metrics.per_depth_losses,
        unpacked_metrics.mtp_loss,
    )

    assert len(packed_metrics.per_depth_losses) == 2
    assert math.isfinite(float(packed_total.item()))
    assert math.isfinite(float(packed_metrics.next_token_loss.item()))
    assert math.isfinite(float(packed_metrics.mtp_loss.item()))
    assert math.isclose(
        float(packed_total.item()),
        float(packed_metrics.total_loss.item()),
        rel_tol=1e-6,
    )
    assert not math.isclose(
        float(packed_metrics.mtp_loss.item()),
        float(unpacked_metrics.mtp_loss.item()),
        rel_tol=1e-5,
        abs_tol=1e-5,
    )
    assert not math.isclose(
        float(packed_total.item()),
        float(unpacked_total.item()),
        rel_tol=1e-5,
        abs_tol=1e-5,
    )


def test_training_loss_with_mtp_reuses_model_head_and_detaches_cuda_parity_paths() -> None:
    model = _tiny_model()
    attach_mtp_head(model)
    batch = mx.array([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]], dtype=mx.int32)

    _, next_token_grads = nn.value_and_grad(model, next_token_cross_entropy)(model, batch)
    loss, grads = nn.value_and_grad(model, next_token_cross_entropy_with_mtp)(model, batch)
    first_head = model.mtp_head
    second_loss, _, _ = next_token_cross_entropy_with_mtp(model, batch)
    mx.eval(loss[0], second_loss)
    next_token_grad_arrays = _flat_grads(next_token_grads)
    grad_arrays = _flat_grads(grads)

    assert model.mtp_head is first_head
    assert math.isfinite(float(loss[0].item()))
    assert math.isfinite(float(second_loss.item()))
    assert "mtp_head.proj.weight" in grad_arrays
    assert "mtp_head.shared_block.up.weight" in grad_arrays
    assert "mtp_head.shared_block.down.weight" in grad_arrays
    assert np.max(np.abs(grad_arrays["mtp_head.proj.weight"])) > 0
    assert np.max(np.abs(grad_arrays["mtp_head.shared_block.up.weight"])) > 0
    assert np.max(np.abs(grad_arrays["mtp_head.shared_block.down.weight"])) > 0
    np.testing.assert_allclose(
        grad_arrays["lm_head.weight"],
        next_token_grad_arrays["lm_head.weight"],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        grad_arrays["layers.0.attn.query_proj.weight"],
        next_token_grad_arrays["layers.0.attn.query_proj.weight"],
        rtol=1e-6,
        atol=1e-6,
    )


def test_one_step_train_accepts_mtp_pair_loss_after_head_attachment() -> None:
    model = _tiny_model()
    attach_mtp_head(model)
    optimizer = make_adamw(learning_rate=1e-3)
    batch = mx.array([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]], dtype=mx.int32)

    result = one_step_train(
        model,
        optimizer,
        batch,
        loss_fn=next_token_cross_entropy_mtp_loss,
    )

    assert math.isfinite(result.loss)
    assert result.ntokens == 10
    assert result.tokens_per_second > 0


def test_training_loss_with_mtp_depth_zero_disables_mtp_side_loss() -> None:
    model = _tiny_model()
    batch = mx.array([[1, 2, 3, 4, 5, 6]], dtype=mx.int32)

    default_loss, default_ntokens = next_token_cross_entropy(model, batch)
    total_loss, ntokens, metrics = next_token_cross_entropy_with_mtp(
        model,
        batch,
        config=MTPLossConfig(depth=0),
    )
    logits = model(batch[:, :-1])
    mx.eval(
        default_loss,
        default_ntokens,
        total_loss,
        ntokens,
        metrics.mtp_loss,
        metrics.depth_weights,
        logits,
    )

    assert metrics.per_depth_losses == ()
    assert metrics.depth_weights.shape == (0,)
    assert math.isclose(float(metrics.mtp_loss.item()), 0.0, abs_tol=1e-7)
    assert math.isclose(
        float(total_loss.item()),
        float(default_loss.item()),
        rel_tol=1e-6,
    )
    assert int(ntokens.item()) == int(default_ntokens.item())
    assert logits.shape == (1, 5, model.config.vocab_size)
    assert np.isfinite(np.array(logits)).all()


def test_mtp_head_detaches_lm_head_but_trains_teacher_embedding() -> None:
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
    assert np.max(np.abs(grad_arrays["lm_head.weight"])) == 0


def test_mtp_cross_entropy_ignores_wrapped_static_labels() -> None:
    logits = mx.array(
        [
            [
                [0.0, 6.0, -1.0],
                [4.0, -2.0, 0.5],
                [9.0, -7.0, 3.0],
            ]
        ],
        dtype=mx.float32,
    )
    labels = mx.array([[1, -1, -1]], dtype=mx.int32)
    all_ignored = mx.full(labels.shape, -1, dtype=mx.int32)

    loss = mtp_cross_entropy_from_logits(logits, labels)
    expected = nn.losses.cross_entropy(
        logits[:, :1, :],
        labels[:, :1],
        reduction="mean",
    )
    ignored_loss = mtp_cross_entropy_from_logits(logits, all_ignored)
    mx.eval(loss, expected, ignored_loss)

    assert math.isclose(float(loss.item()), float(expected.item()), rel_tol=1e-6)
    assert math.isclose(float(ignored_loss.item()), 0.0, abs_tol=1e-7)

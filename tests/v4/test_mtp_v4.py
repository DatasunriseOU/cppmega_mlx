"""Tests for cppmega_v4.nn.mtp_v4 — DeepSeek-V3 SequentialMTPHead plugin."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from cppmega_mlx.training.mtp import (
    MinimalMTPHead,
    MTPLossConfig,
    compute_weighted_mtp_loss,
)
from cppmega_v4.nn.mtp_v4 import (
    SequentialMTPDepthBlock,
    SequentialMTPHead,
    attach_sequential_mtp_head,
)


def _emb_and_head(vocab=16, hidden=8) -> tuple[nn.Embedding, nn.Linear]:
    return nn.Embedding(vocab, hidden), nn.Linear(hidden, vocab, bias=False)


# ----- depth-block primitive -----


def test_depth_block_forward_shape():
    block = SequentialMTPDepthBlock(hidden_size=8)
    h = mx.random.normal((1, 4, 8))
    emb = mx.random.normal((1, 4, 8))
    out = block(h, emb)
    assert out.shape == (1, 4, 8)


def test_depth_block_rejects_zero_hidden():
    with pytest.raises(ValueError, match="hidden_size must be positive"):
        SequentialMTPDepthBlock(0)


# ----- D distinct blocks -----


def test_head_owns_d_distinct_depth_blocks():
    emb, lm = _emb_and_head()
    head = SequentialMTPHead(emb, lm, config=MTPLossConfig(depth=3))
    assert len(head.depth_blocks) == 3
    ids = {id(b) for b in head.depth_blocks}
    assert len(ids) == 3, "all depth blocks must be distinct module instances"


def test_head_zero_depth_returns_empty_tuple():
    emb, lm = _emb_and_head()
    head = SequentialMTPHead(emb, lm, config=MTPLossConfig(depth=0))
    target = mx.array([[0, 1, 2, 3]])
    h = emb(target)
    assert head(h, target) == ()


# ----- shared weights with model -----


def test_attach_aliases_token_embedding_and_lm_head():
    """Head must reuse model's existing embedding + lm_head modules."""

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_embedding = nn.Embedding(16, 8)
            self.lm_head = nn.Linear(8, 16, bias=False)

    model = _Model()
    head = attach_sequential_mtp_head(model, config=MTPLossConfig(depth=2))
    assert head.token_embedding is model.token_embedding
    assert head.lm_head is model.lm_head
    assert model.mtp_head is head


def test_attach_rejects_missing_embedding():
    class _Bad(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = nn.Linear(8, 16, bias=False)

    with pytest.raises(TypeError, match="token_embedding"):
        attach_sequential_mtp_head(_Bad())


def test_attach_rejects_missing_lm_head():
    class _Bad(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_embedding = nn.Embedding(16, 8)

    with pytest.raises(TypeError, match="lm_head"):
        attach_sequential_mtp_head(_Bad())


# ----- forward shape contract -----


def test_forward_returns_one_logits_tensor_per_depth():
    emb, lm = _emb_and_head(vocab=16, hidden=8)
    head = SequentialMTPHead(emb, lm, config=MTPLossConfig(depth=4))
    target = mx.array([[5, 1, 2, 3]])
    h = mx.random.normal((1, 4, 8))
    logits_by_depth = head(h, target)
    assert len(logits_by_depth) == 4
    for logits in logits_by_depth:
        assert logits.shape == (1, 4, 16)


def test_forward_rejects_wrong_rank():
    emb, lm = _emb_and_head()
    head = SequentialMTPHead(emb, lm, config=MTPLossConfig(depth=1))
    with pytest.raises(ValueError, match="hidden_states must be shaped"):
        head(mx.random.normal((1, 4)), mx.array([[0, 1, 2, 3]]))


def test_forward_rejects_shape_mismatch():
    emb, lm = _emb_and_head()
    head = SequentialMTPHead(emb, lm, config=MTPLossConfig(depth=1))
    with pytest.raises(ValueError, match="must match"):
        head(mx.random.normal((1, 4, 8)), mx.array([[0, 1, 2]]))


# ----- loss surface -----


def test_loss_returns_per_depth_and_weighted_total():
    emb, lm = _emb_and_head(vocab=16, hidden=8)
    head = SequentialMTPHead(emb, lm, config=MTPLossConfig(depth=3, decay=0.6))
    target = mx.array([[1, 2, 3, 4, 5]])
    h = mx.random.normal((1, 5, 8))
    mtp_loss, per_depth, depth_weights = head.loss(h, target)
    assert len(per_depth) == 3
    assert depth_weights.shape == (3,)
    # Reproduce the weighted sum and compare.
    expected, _ = compute_weighted_mtp_loss(per_depth, decay=0.6)
    np.testing.assert_allclose(
        np.array(mtp_loss), np.array(expected), atol=1e-6, rtol=0
    )


def test_loss_with_document_ids_runs_without_nan():
    emb, lm = _emb_and_head(vocab=16, hidden=8)
    head = SequentialMTPHead(emb, lm, config=MTPLossConfig(depth=2))
    target = mx.array([[1, 2, 3, 4]])
    doc_ids = mx.array([[0, 0, 1, 1]], dtype=mx.int32)
    h = mx.random.normal((1, 4, 8))
    mtp_loss, per_depth, _ = head.loss(h, target, document_ids=doc_ids)
    assert not bool(mx.any(mx.isnan(mtp_loss)).item())
    for pd in per_depth:
        assert not bool(mx.any(mx.isnan(pd)).item())


# ----- difference from MinimalMTPHead (one shared block vs D distinct) -----


def test_sequential_has_more_params_than_minimal_for_same_depth():
    """V3 sequential should own D copies of the transformer kernel, minimal owns 1."""
    emb, lm = _emb_and_head(vocab=16, hidden=8)
    cfg = MTPLossConfig(depth=3)
    minimal = MinimalMTPHead(emb, lm, config=cfg)
    sequential = SequentialMTPHead(emb, lm, config=cfg)

    def _count_floats(params):
        total = 0
        if isinstance(params, dict):
            for v in params.values():
                total += _count_floats(v)
        elif isinstance(params, list):
            for v in params:
                total += _count_floats(v)
        elif isinstance(params, mx.array):
            total += int(params.size)
        return total

    minimal_n = _count_floats(minimal.trainable_parameters())
    sequential_n = _count_floats(sequential.trainable_parameters())
    assert sequential_n > minimal_n


# ----- gradient flow -----


def test_grad_flows_into_each_depth_block_independently():
    """Backward on a single depth's logits should hit only that depth's block."""
    emb, lm = _emb_and_head(vocab=16, hidden=8)
    head = SequentialMTPHead(emb, lm, config=MTPLossConfig(depth=2))
    target = mx.array([[1, 2, 3, 4]])
    h = mx.random.normal((1, 4, 8))

    def loss_only_depth_0(params):
        head.update(params)
        logits = head(h, target)
        return mx.mean(mx.square(logits[0]))

    grads = mx.grad(loss_only_depth_0)(head.trainable_parameters())
    block0 = grads["depth_blocks"][0]
    block1 = grads["depth_blocks"][1]
    # Depth 0's block must have non-zero gradients.
    assert float(mx.max(mx.abs(block0["proj"]["weight"])).item()) > 0.0
    # Depth 1's block must have zero gradients (output not used).
    assert float(mx.max(mx.abs(block1["proj"]["weight"])).item()) == 0.0


def test_grad_does_not_flow_through_hidden_states():
    """hidden_states is stop-gradiented inside the head."""
    emb, lm = _emb_and_head(vocab=16, hidden=8)
    head = SequentialMTPHead(emb, lm, config=MTPLossConfig(depth=1))
    target = mx.array([[0, 1, 2, 3]])

    def loss_wrt_hidden(h_arg):
        logits = head(h_arg, target)
        return mx.mean(mx.square(logits[0]))

    h = mx.random.normal((1, 4, 8))
    grad_h = mx.grad(loss_wrt_hidden)(h)
    # stop_gradient inside the head should null out the grad here.
    np.testing.assert_array_equal(np.array(grad_h), np.zeros((1, 4, 8)))


def test_grad_through_shared_token_embedding():
    """Gradient flows back through the aliased token_embedding."""
    emb, lm = _emb_and_head(vocab=16, hidden=8)
    head = SequentialMTPHead(emb, lm, config=MTPLossConfig(depth=2))
    target = mx.array([[1, 2, 3, 4]])
    h = mx.random.normal((1, 4, 8))

    def loss_fn(params):
        head.update(params)
        logits = head(h, target)
        return sum((mx.mean(mx.square(lg)) for lg in logits), start=mx.array(0.0))

    grads = mx.grad(loss_fn)(head.trainable_parameters())
    # token_embedding weight should have a non-zero gradient.
    assert float(mx.max(mx.abs(grads["token_embedding"]["weight"])).item()) > 0.0

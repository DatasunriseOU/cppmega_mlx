"""Tests for cppmega_v4.nn.moe_v4 — V4 MoE plugin (aux-loss-free, sqrtsoftplus)."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from cppmega_mlx.nn.moe import MoEConfig, ReferenceMoE
from cppmega_v4.nn.moe_v4 import V4MoE, V4MoEConfig, _score_logits


def _tiny_config(**overrides) -> V4MoEConfig:
    base = dict(
        d_model=8,
        num_experts=4,
        top_k=2,
        expert_hidden_size=16,
        shared_expert_hidden_size=None,
        activation="swiglu",
    )
    base.update(overrides)
    return V4MoEConfig(**base)


# ----- scoring -----


def test_score_logits_softmax_matches_mlx():
    logits = mx.random.normal((3, 4))
    got = _score_logits(logits, "softmax")
    want = mx.softmax(logits, axis=-1)
    np.testing.assert_allclose(np.array(got), np.array(want), atol=1e-6)


def test_score_logits_sigmoid_matches_mlx():
    logits = mx.random.normal((3, 4))
    got = _score_logits(logits, "sigmoid")
    want = mx.sigmoid(logits)
    np.testing.assert_allclose(np.array(got), np.array(want), atol=1e-6)


def test_score_logits_sqrtsoftplus():
    logits = mx.array([[1.0, -1.0, 0.0, 2.0]])
    got = np.array(_score_logits(logits, "sqrtsoftplus"))
    want = np.sqrt(np.log1p(np.exp(np.array(logits))))
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


def test_unknown_scoring_rejected():
    with pytest.raises(ValueError, match="unsupported scoring"):
        V4MoEConfig(d_model=8, scoring="bogus")  # type: ignore[arg-type]


# ----- backward-compat (softmax + no bias = original ReferenceMoE behavior) -----


def test_v4moe_softmax_no_bias_matches_reference_top_indices():
    """V4MoE in legacy mode picks the same top-k experts as ReferenceMoE."""
    cfg = _tiny_config(scoring="softmax", aux_loss_free=False)
    mx.random.seed(11)
    v4 = V4MoE(cfg)
    # Build a reference with identical config and copy gate weights.
    ref = ReferenceMoE(cfg.as_moe_config())
    ref.router.gate.weight = v4.gate.weight
    for v_e, r_e in zip(v4.experts, ref.experts, strict=True):
        v_e.up_proj.weight = r_e.up_proj.weight
        v_e.down_proj.weight = r_e.down_proj.weight
        if hasattr(v_e, "gate_proj"):
            v_e.gate_proj.weight = r_e.gate_proj.weight

    x = mx.random.normal((1, 4, cfg.d_model))
    v4_out = v4(x)
    ref_out = ref(x)
    # Top-k indices must agree (the scoring/selection path is identical here).
    np.testing.assert_array_equal(
        np.array(v4_out.router.top_indices), np.array(ref_out.router.top_indices)
    )


# ----- aux-loss-free bias -----


def test_expert_bias_initialized_only_when_enabled():
    cfg_off = _tiny_config(aux_loss_free=False)
    cfg_on = _tiny_config(aux_loss_free=True)
    assert not hasattr(V4MoE(cfg_off), "expert_bias")
    moe = V4MoE(cfg_on)
    assert hasattr(moe, "expert_bias")
    assert moe.expert_bias.shape == (cfg_on.num_experts,)
    np.testing.assert_allclose(np.array(moe.expert_bias), np.zeros(cfg_on.num_experts))


def test_aux_loss_zero_when_aux_loss_free():
    cfg = _tiny_config(aux_loss_free=True, scoring="sigmoid")
    moe = V4MoE(cfg)
    x = mx.random.normal((1, 4, cfg.d_model))
    out = moe(x)
    assert float(out.router.aux_loss.item()) == 0.0


def test_aux_loss_nonzero_when_legacy_balancing():
    cfg = _tiny_config(aux_loss_free=False, scoring="softmax")
    moe = V4MoE(cfg)
    x = mx.random.normal((2, 8, cfg.d_model))
    out = moe(x)
    # softmax + non-trivial input → aux_loss should be > 0 (matches ReferenceMoE).
    assert float(out.router.aux_loss.item()) > 0.0


def test_bias_update_moves_underloaded_up_and_overloaded_down():
    cfg = _tiny_config(aux_loss_free=True, bias_update_rate=0.1)
    moe = V4MoE(cfg)
    # Synthetic load: expert 0 underloaded, expert 2 overloaded.
    load = mx.array([0.0, 0.25, 1.0, 0.25])
    initial = np.array(moe.expert_bias).copy()
    moe.update_bias_after_step(load)
    after = np.array(moe.expert_bias)
    assert after[0] - initial[0] == pytest.approx(0.1)   # under -> +rate
    assert after[2] - initial[2] == pytest.approx(-0.1)  # over -> -rate


def test_bias_update_noop_when_disabled():
    cfg = _tiny_config(aux_loss_free=False)
    moe = V4MoE(cfg)
    load = mx.array([0.5, 0.5, 0.0, 0.0])
    moe.update_bias_after_step(load)  # must not raise; must not create attr
    assert not hasattr(moe, "expert_bias")


def test_bias_update_rejects_wrong_shape():
    cfg = _tiny_config(aux_loss_free=True)
    moe = V4MoE(cfg)
    with pytest.raises(ValueError, match="router_load must be shape"):
        moe.update_bias_after_step(mx.array([0.1, 0.2]))  # wrong size


def test_bias_affects_top_k_selection_not_weights():
    """V3 invariant: bias shifts which experts are picked, not their weights."""
    cfg = _tiny_config(aux_loss_free=True, bias_update_rate=0.0)
    mx.random.seed(33)
    moe = V4MoE(cfg)
    x = mx.random.normal((1, 1, cfg.d_model))

    out_zero_bias = moe(x)
    # Set bias to a huge positive on expert 0, huge negative everywhere else.
    moe.expert_bias = mx.array([10.0, -10.0, -10.0, -10.0])
    out_biased = moe(x)

    # Now expert 0 should be in the top-k selection.
    selected = np.array(out_biased.router.top_indices).flatten().tolist()
    assert 0 in selected
    # And the corresponding weight should be the raw (pre-bias) score, NOT
    # influenced by the +10 bias (other than via selection).
    biased_weights = np.array(out_biased.router.top_weights).flatten()
    assert np.all(biased_weights >= 0.0)
    assert np.all(biased_weights <= 1.0 + 1e-5)


# ----- sqrtsoftplus end-to-end forward -----


def test_v4moe_sqrtsoftplus_forward_runs_and_no_nan():
    cfg = _tiny_config(scoring="sqrtsoftplus", aux_loss_free=True)
    moe = V4MoE(cfg)
    x = mx.random.normal((2, 4, cfg.d_model))
    out = moe(x)
    assert out.output.shape == x.shape
    assert not bool(mx.any(mx.isnan(out.output)).item())
    assert not bool(mx.any(mx.isinf(out.output)).item())


def test_v4moe_sigmoid_forward_runs_and_no_nan():
    cfg = _tiny_config(scoring="sigmoid", aux_loss_free=True)
    moe = V4MoE(cfg)
    x = mx.random.normal((2, 4, cfg.d_model))
    out = moe(x)
    assert out.output.shape == x.shape
    assert not bool(mx.any(mx.isnan(out.output)).item())


def test_v4moe_with_shared_expert_adds_shared_output():
    cfg = _tiny_config(shared_expert_hidden_size=16, scoring="sqrtsoftplus")
    moe = V4MoE(cfg)
    x = mx.random.normal((1, 4, cfg.d_model))
    out = moe(x)
    assert out.shared_output is not None
    assert out.shared_output.shape == x.shape
    np.testing.assert_allclose(
        np.array(out.output), np.array(out.routed_output + out.shared_output), atol=1e-5
    )


# ----- node-limited routing -----


def test_node_limited_routing_validates_divisibility():
    with pytest.raises(ValueError, match="divisible"):
        V4MoEConfig(
            d_model=8, num_experts=5, top_k=2, expert_hidden_size=16, node_limited_routing=2
        )


def test_node_limited_routing_runs_and_routes_within_groups():
    cfg = _tiny_config(num_experts=4, top_k=2, node_limited_routing=2)
    moe = V4MoE(cfg)
    x = mx.random.normal((1, 3, cfg.d_model))
    out = moe(x)
    assert out.output.shape == x.shape
    # No assertion on which specific experts get picked; just no crash + no NaN.
    assert not bool(mx.any(mx.isnan(out.output)).item())


# ----- gradient flow -----


def test_gradient_flows_to_gate_and_experts():
    cfg = _tiny_config(scoring="sqrtsoftplus", aux_loss_free=True)
    moe = V4MoE(cfg)
    x = mx.random.normal((1, 2, cfg.d_model))

    def loss_fn(params):
        moe.update(params)
        out = moe(x)
        return mx.mean(mx.square(out.output))

    params = moe.trainable_parameters()
    grads = mx.grad(loss_fn)(params)
    # Gate weight must have a non-zero gradient.
    gate_grad = grads["gate"]["weight"]
    assert float(mx.max(mx.abs(gate_grad)).item()) > 0.0
    # And expert_bias must NOT appear in the trainable grad tree (it's frozen).
    assert "expert_bias" not in grads

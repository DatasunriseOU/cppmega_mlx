"""V7-E06: MoE inference path — routed selection at gen time."""

from __future__ import annotations

import mlx.core as mx
import pytest

from cppmega_v4.nn.moe_v4 import V4MoE, V4MoEConfig


def test_v7_e06_inference_router_selects_distinct_experts():
    """Forward 16 distinct tokens through eval-mode MoE → router
    picks at least 2 distinct experts across the run."""
    cfg = V4MoEConfig(
        d_model=128, num_experts=8, top_k=2,
        expert_hidden_size=128, aux_loss_free=False,
    )
    moe = V4MoE(cfg)
    moe.eval()
    # 16 single-token forwards.
    selections = set()
    for step in range(16):
        x = mx.random.normal(shape=(1, 1, cfg.d_model),
                              key=mx.random.key(step))
        out = moe(x)
        for e in out.router.top_indices.flatten().tolist():
            selections.add(int(e))
    assert len(selections) >= 2, (
        f"router collapsed to single expert at inference: {selections}"
    )


def test_v7_e06_inference_aux_loss_is_zero():
    """eval-mode MoE: aux_loss should be 0 (no balancing on inference)."""
    cfg = V4MoEConfig(
        d_model=64, num_experts=4, top_k=2,
        expert_hidden_size=64, aux_loss_free=True,  # forces aux=0
    )
    moe = V4MoE(cfg)
    moe.eval()
    x = mx.random.normal(shape=(1, 4, cfg.d_model),
                          key=mx.random.key(0))
    out = moe(x)
    assert float(out.router.aux_loss.item()) == 0.0


def test_v7_e06_inference_router_bias_does_not_change():
    """eval-mode does not call update_bias_after_step → expert_bias
    stays untouched across forwards (different inputs)."""
    cfg = V4MoEConfig(
        d_model=64, num_experts=4, top_k=2,
        expert_hidden_size=64, aux_loss_free=True,
    )
    moe = V4MoE(cfg)
    moe.eval()
    bias_before = mx.array(moe.expert_bias)
    for step in range(8):
        x = mx.random.normal(shape=(1, 2, cfg.d_model),
                              key=mx.random.key(step + 1))
        _ = moe(x)
    bias_after = moe.expert_bias
    assert mx.allclose(bias_before, bias_after, atol=1e-12)


def test_v7_e06_inference_deterministic_outputs_same_seed():
    """Same input → bit-identical output across two forwards (the
    router has no randomness at inference)."""
    cfg = V4MoEConfig(
        d_model=64, num_experts=4, top_k=2,
        expert_hidden_size=64, aux_loss_free=False,
    )
    moe = V4MoE(cfg)
    moe.eval()
    x = mx.random.normal(shape=(1, 4, cfg.d_model),
                          key=mx.random.key(42))
    out1 = moe(x).output
    out2 = moe(x).output
    assert mx.allclose(out1, out2, atol=1e-6)

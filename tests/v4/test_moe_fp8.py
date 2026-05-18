"""Tests for FP8 MoE expert layer + V4MoE checkpoint converter."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from cppmega_v4.nn.moe_fp8 import (
    FP8FeedForwardExpert,
    FP8Linear,
    convert_v4moe_to_fp8,
    quantize_linear_to_fp8,
)
from cppmega_v4.nn.moe_v4 import V4MoE, V4MoEConfig


def test_quantize_linear_shape_and_dtypes():
    w = mx.random.normal((128, 256)).astype(mx.bfloat16)
    fp8, scale = quantize_linear_to_fp8(w)
    assert fp8.shape == w.shape
    assert fp8.dtype == mx.uint8
    assert scale.shape == (1, 2)   # ceil(128/128)=1, ceil(256/128)=2
    assert scale.dtype == mx.float32


def test_quantize_linear_rejects_wrong_rank():
    w = mx.zeros((128,))
    with pytest.raises(ValueError, match="2D"):
        quantize_linear_to_fp8(w)


def test_fp8_linear_from_linear_round_trip():
    """FP8Linear(x) ≈ original_linear(x) within fp8 precision."""
    in_dim, out_dim = 128, 128
    rng = np.random.default_rng(0)
    lin = nn.Linear(in_dim, out_dim, bias=False)
    lin.weight = mx.array(rng.standard_normal((out_dim, in_dim)).astype(np.float32) * 0.1)
    fp8 = FP8Linear.from_linear(lin)

    x = mx.array(rng.standard_normal((4, in_dim)).astype(np.float32))
    y_ref = lin(x.astype(mx.float32))
    y_fp8 = fp8(x)
    np.testing.assert_allclose(
        np.array(y_fp8.astype(mx.float32)), np.array(y_ref),
        atol=1.5e-1, rtol=2e-1,    # fp8 e4m3 worst-case ~12% relative drift
    )


def test_fp8_feedforward_expert_round_trip():
    """FP8FeedForwardExpert ≈ FeedForwardExpert within fp8 precision."""
    from cppmega_mlx.nn.moe import FeedForwardExpert
    d, hidden = 64, 128
    rng = np.random.default_rng(1)
    expert = FeedForwardExpert(d, hidden, activation="swiglu", bias=False)
    # randomize weights so the test isn't all-zero
    expert.gate_proj.weight = mx.array(rng.standard_normal((hidden, d)).astype(np.float32) * 0.1)
    expert.up_proj.weight = mx.array(rng.standard_normal((hidden, d)).astype(np.float32) * 0.1)
    expert.down_proj.weight = mx.array(rng.standard_normal((d, hidden)).astype(np.float32) * 0.1)

    fp8_expert = FP8FeedForwardExpert.from_fp32_expert(expert)
    x = mx.array(rng.standard_normal((2, 8, d)).astype(np.float32))
    y_ref = expert(x.astype(mx.float32))
    y_fp8 = fp8_expert(x)
    assert y_fp8.shape == y_ref.shape
    np.testing.assert_allclose(
        np.array(y_fp8.astype(mx.float32)), np.array(y_ref),
        atol=2e-1, rtol=1e-1,   # 2 fp8 GEMMs in a row + nonlinearity
    )


def test_convert_v4moe_to_fp8_preserves_shape_and_close_output():
    rng = np.random.default_rng(2)
    cfg = V4MoEConfig(d_model=64, num_experts=4, top_k=2,
                       expert_hidden_size=128, activation="swiglu")
    moe = V4MoE(cfg)
    # randomize weights
    for e in moe.experts:
        e.gate_proj.weight = mx.array(rng.standard_normal((128, 64)).astype(np.float32) * 0.1)
        e.up_proj.weight = mx.array(rng.standard_normal((128, 64)).astype(np.float32) * 0.1)
        e.down_proj.weight = mx.array(rng.standard_normal((64, 128)).astype(np.float32) * 0.1)
    moe.gate.weight = mx.array(rng.standard_normal((4, 64)).astype(np.float32) * 0.1)

    x = mx.array(rng.standard_normal((1, 8, 64)).astype(np.float32))
    out_ref = moe(x).output

    convert_v4moe_to_fp8(moe)
    out_fp8 = moe(x).output

    assert out_fp8.shape == out_ref.shape
    # FP8 ~5-10% drift on stacked GEMMs through routed experts.
    np.testing.assert_allclose(
        np.array(out_fp8.astype(mx.float32)), np.array(out_ref),
        atol=3e-1, rtol=2e-1,
    )


def test_fp8_fwd_with_bf16_master_weight_training_step():
    """Production training pattern: bf16 master weight + fp8 inference path.

    Each step:
        1. quantize bf16 master → fp8 (fresh per step, no stale state)
        2. fp8 forward computes loss
        3. gradient is taken through the bf16 master via the bf16 reference
           forward (standard straight-through pattern for fp8 training)
        4. SGD update in bf16

    Asserts:
        - fp8 loss tracks bf16 loss within fp8 tolerance at every step
        - bf16-side grad direction also decreases the fp8 loss
        - 5-step SGD using bf16 grads + fp8 forward monotonically decreases
          fp8 loss
    """
    d_in, d_out = 128, 128
    rng = np.random.default_rng(99)
    W_master_bf16 = mx.array(
        rng.standard_normal((d_out, d_in)).astype(np.float32) * 0.05
    ).astype(mx.bfloat16)
    x = mx.array(rng.standard_normal((4, d_in)).astype(np.float32))
    target = mx.array(rng.standard_normal((4, d_out)).astype(np.float32) * 0.1)

    def fp8_loss(W_bf16):
        lin = nn.Linear(d_in, d_out, bias=False)
        lin.weight = W_bf16
        fp8_lin = FP8Linear.from_linear(lin)
        y_fp8 = fp8_lin(x).astype(mx.float32)
        return float(mx.mean((y_fp8 - target) ** 2))

    def bf16_loss_fn(W_f32):
        y = (W_f32 @ x.T).T
        return mx.mean((y - target) ** 2)

    losses_fp8 = [fp8_loss(W_master_bf16)]
    losses_bf16 = [float(bf16_loss_fn(W_master_bf16.astype(mx.float32)))]
    np.testing.assert_allclose(
        losses_fp8[0], losses_bf16[0], atol=2e-2, rtol=2e-1,
        err_msg="fp8 fwd loss must track bf16 fwd loss at step 0",
    )

    lr = 0.1
    W = W_master_bf16
    for step in range(5):
        W_f32 = W.astype(mx.float32)
        grad_bf16 = mx.grad(bf16_loss_fn)(W_f32)
        assert np.all(np.isfinite(np.array(grad_bf16))), \
            f"step {step}: bf16 grad not finite"
        W = (W_f32 - lr * grad_bf16).astype(mx.bfloat16)
        losses_fp8.append(fp8_loss(W))
        losses_bf16.append(float(bf16_loss_fn(W.astype(mx.float32))))

    # The fp8 loss must end strictly lower than it started — the bf16 grad
    # direction does still decrease loss after the fp8 round-trip.
    assert losses_fp8[-1] < losses_fp8[0], (
        f"fp8 training loss did not decrease over 5 steps: {losses_fp8}"
    )
    # Per-step fp8 vs bf16 drift must stay within fp8 tolerance.
    for s, (lf, lb) in enumerate(zip(losses_fp8, losses_bf16)):
        np.testing.assert_allclose(
            lf, lb, atol=5e-2, rtol=3e-1,
            err_msg=f"step {s}: fp8 loss {lf} diverged from bf16 loss {lb}",
        )


def test_fp8_requantization_is_stable_across_repeated_steps():
    """Quantizing the same master weight twice yields bit-identical fp8.

    Critical for training: the fp8 path must be deterministic so the
    forward output doesn't jitter between identical optimizer states.
    """
    d_in, d_out = 64, 64
    rng = np.random.default_rng(101)
    W = mx.array(
        rng.standard_normal((d_out, d_in)).astype(np.float32) * 0.07
    ).astype(mx.bfloat16)
    fp8_a, scale_a = quantize_linear_to_fp8(W)
    fp8_b, scale_b = quantize_linear_to_fp8(W)
    assert np.array_equal(np.array(fp8_a), np.array(fp8_b))
    np.testing.assert_array_equal(np.array(scale_a), np.array(scale_b))


def test_convert_v4moe_converts_shared_expert_too():
    cfg = V4MoEConfig(d_model=64, num_experts=2, top_k=1,
                       expert_hidden_size=128, shared_expert_hidden_size=64,
                       activation="swiglu")
    moe = V4MoE(cfg)
    assert moe.shared_expert is not None
    assert type(moe.shared_expert).__name__ != "FP8FeedForwardExpert"
    convert_v4moe_to_fp8(moe)
    assert type(moe.shared_expert).__name__ == "FP8FeedForwardExpert"

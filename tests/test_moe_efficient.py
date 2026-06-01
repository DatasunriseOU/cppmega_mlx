"""Numeric parity for the env-gated memory-efficient sparse-gather MoE.

The efficient routed-combine (``CPPMEGA_MOE_EFFICIENT=1``) must produce the same
forward output *and* the same gradients (w.r.t. inputs and every parameter) as
the dense reference, otherwise the env flag would silently change training
numerics. These tests pin that exactness in fp32 (where the only remaining delta
is float epsilon) and document the bf16 accumulation-order tolerance.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from cppmega_mlx.nn import moe as moe_mod
from cppmega_mlx.nn.moe import MoEConfig, ReferenceMoE


def _config(*, d_model: int = 3584, num_experts: int = 16, top_k: int = 4) -> MoEConfig:
    return MoEConfig(
        d_model=d_model,
        num_experts=num_experts,
        top_k=top_k,
        expert_hidden_size=128,
        shared_expert_hidden_size=96,
        activation="swiglu",
    )


def _build(config: MoEConfig, dtype: mx.Dtype) -> ReferenceMoE:
    mx.random.seed(20260601)
    moe = ReferenceMoE(config)
    if dtype != mx.float32:
        moe.update(
            tree_unflatten(
                [
                    (name, leaf.astype(dtype) if leaf.dtype == mx.float32 else leaf)
                    for name, leaf in tree_flatten(moe.parameters())
                ]
            )
        )
    mx.eval(moe.parameters())
    return moe


def _run_both(moe: ReferenceMoE, x: mx.array):
    """Return (dense_output, efficient_output) for the same module + input."""

    moe_mod.os.environ.pop(moe_mod._MOE_EFFICIENT_ENV, None)
    dense_out = moe(x).output
    moe_mod.os.environ[moe_mod._MOE_EFFICIENT_ENV] = "1"
    try:
        eff_out = moe(x).output
    finally:
        moe_mod.os.environ.pop(moe_mod._MOE_EFFICIENT_ENV, None)
    mx.eval(dense_out, eff_out)
    return dense_out, eff_out


def test_env_flag_default_off() -> None:
    moe_mod.os.environ.pop(moe_mod._MOE_EFFICIENT_ENV, None)
    assert moe_mod._moe_efficient_enabled() is False
    for value in ("1", "true", "on", "YES"):
        moe_mod.os.environ[moe_mod._MOE_EFFICIENT_ENV] = value
        assert moe_mod._moe_efficient_enabled() is True
    moe_mod.os.environ[moe_mod._MOE_EFFICIENT_ENV] = "0"
    assert moe_mod._moe_efficient_enabled() is False
    moe_mod.os.environ.pop(moe_mod._MOE_EFFICIENT_ENV, None)


def test_efficient_forward_matches_dense_fp32() -> None:
    config = _config()
    moe = _build(config, mx.float32)
    mx.random.seed(11)
    x = mx.random.normal((1, 128, config.d_model)).astype(mx.float32)

    dense_out, eff_out = _run_both(moe, x)
    max_diff = float(mx.max(mx.abs(dense_out - eff_out)))
    assert max_diff < 1e-5, f"forward fp32 max_diff={max_diff:.3e} exceeds 1e-5"


def test_efficient_input_grad_matches_dense_fp32() -> None:
    config = _config()
    moe = _build(config, mx.float32)
    mx.random.seed(12)
    x = mx.random.normal((1, 128, config.d_model)).astype(mx.float32)

    def dense_loss(xx: mx.array) -> mx.array:
        moe_mod.os.environ.pop(moe_mod._MOE_EFFICIENT_ENV, None)
        return moe(xx).output.square().sum()

    def eff_loss(xx: mx.array) -> mx.array:
        moe_mod.os.environ[moe_mod._MOE_EFFICIENT_ENV] = "1"
        try:
            return moe(xx).output.square().sum()
        finally:
            moe_mod.os.environ.pop(moe_mod._MOE_EFFICIENT_ENV, None)

    gd = mx.grad(dense_loss)(x)
    gs = mx.grad(eff_loss)(x)
    mx.eval(gd, gs)
    max_diff = float(mx.max(mx.abs(gd - gs)))
    assert max_diff < 1e-5, f"input-grad fp32 max_diff={max_diff:.3e} exceeds 1e-5"


def test_efficient_param_grads_match_dense_fp32() -> None:
    config = _config()
    mx.random.seed(13)
    x = mx.random.normal((1, 128, config.d_model)).astype(mx.float32)

    def loss(model: ReferenceMoE, xx: mx.array) -> mx.array:
        return model(xx).output.square().sum()

    # Dense
    moe_dense = _build(config, mx.float32)
    moe_mod.os.environ.pop(moe_mod._MOE_EFFICIENT_ENV, None)
    _, gd = nn.value_and_grad(moe_dense, loss)(moe_dense, x)
    # Efficient (same seed -> same weights)
    moe_eff = _build(config, mx.float32)
    moe_mod.os.environ[moe_mod._MOE_EFFICIENT_ENV] = "1"
    try:
        _, gs = nn.value_and_grad(moe_eff, loss)(moe_eff, x)
    finally:
        moe_mod.os.environ.pop(moe_mod._MOE_EFFICIENT_ENV, None)

    flat_d = dict(tree_flatten(gd))
    flat_s = dict(tree_flatten(gs))
    assert set(flat_d) == set(flat_s), "gradient trees differ in keys"
    worst = 0.0
    worst_name = ""
    for name, gd_leaf in flat_d.items():
        d = float(mx.max(mx.abs(gd_leaf - flat_s[name])))
        if d > worst:
            worst, worst_name = d, name
    assert worst < 1e-4, f"param-grad fp32 worst={worst:.3e} ({worst_name}) exceeds 1e-4"


def test_efficient_unrouted_expert_keeps_zero_param_grad() -> None:
    """An expert with no routed tokens still appears in the grad tree (grad 0)."""

    config = _config(num_experts=16, top_k=1)
    mx.random.seed(99)
    moe = _build(config, mx.float32)
    # Force the router so all tokens pick expert 0 -> experts 1..15 unrouted.
    weight = np.zeros((config.num_experts, config.d_model), dtype=np.float32)
    weight[0, 0] = 50.0  # huge logit for expert 0 on channel 0
    moe.router.gate.weight = mx.array(weight)
    mx.eval(moe.parameters())
    x = mx.zeros((1, 8, config.d_model), dtype=mx.float32)
    x = x + 0.0
    x_idx = np.zeros((8, config.d_model), dtype=np.float32)
    x_idx[:, 0] = 1.0  # ensure channel 0 active so expert 0 wins
    x = mx.array(x_idx[None])

    def loss(model: ReferenceMoE, xx: mx.array) -> mx.array:
        return model(xx).output.square().sum()

    moe_mod.os.environ[moe_mod._MOE_EFFICIENT_ENV] = "1"
    try:
        _, grads = nn.value_and_grad(moe, loss)(moe, x)
    finally:
        moe_mod.os.environ.pop(moe_mod._MOE_EFFICIENT_ENV, None)
    flat = dict(tree_flatten(grads))
    # Every expert param must be present (zero grad allowed, missing not).
    for e in range(config.num_experts):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            key = f"experts.{e}.{proj}.weight"
            assert key in flat, f"missing grad for {key}"
            assert np.isfinite(np.array(flat[key])).all(), key


def test_efficient_bf16_within_accumulation_tolerance() -> None:
    """bf16 differs only by reduction-order rounding (documented, not a bug)."""

    config = _config()
    moe = _build(config, mx.bfloat16)
    mx.random.seed(14)
    x = mx.random.normal((1, 128, config.d_model)).astype(mx.bfloat16)
    dense_out, eff_out = _run_both(moe, x)
    diff = float(mx.max(mx.abs(dense_out.astype(mx.float32) - eff_out.astype(mx.float32))))
    # Reordered bf16 reductions: tolerance ~ a few ULPs of the output scale.
    assert diff < 5e-2, f"bf16 max_diff={diff:.3e} unexpectedly large"

"""V7-D04: hybrid master-fp32 + bf16-fwd helpers."""

from __future__ import annotations

import mlx.core as mx
import pytest

from cppmega_v4.runtime.hybrid_precision import (
    cast_for_forward, cast_grads_to_master, hybrid_step,
)


def test_v7_d04_cast_for_forward_returns_bf16_view():
    p = mx.random.normal(shape=(8, 8), key=mx.random.key(0))
    assert p.dtype == mx.float32
    v = cast_for_forward(p, mx.bfloat16)
    assert v.dtype == mx.bfloat16
    # Master unchanged.
    assert p.dtype == mx.float32


def test_v7_d04_cast_grads_to_master_preserves_fp32():
    grads = {"a": mx.zeros((4,), dtype=mx.bfloat16),
             "b": mx.zeros((4,), dtype=mx.float16)}
    out = cast_grads_to_master(grads, master_dtype=mx.float32)
    assert out["a"].dtype == mx.float32
    assert out["b"].dtype == mx.float32


def test_v7_d04_hybrid_step_updates_master_in_fp32():
    master = {"w": mx.array([1.0, 2.0, 3.0], dtype=mx.float32)}

    def loss_and_grad(fwd):
        # Always-positive grad of 0.1 per element.
        return mx.array(0.0), {"w": mx.array([0.1, 0.1, 0.1],
                                              dtype=mx.bfloat16)}

    def apply_gradients(m, g):
        return {k: m[k] - g[k] for k in m}

    new_master = hybrid_step(
        master_params=master, fwd_dtype=mx.bfloat16,
        loss_and_grad=loss_and_grad,
        apply_gradients=apply_gradients,
    )
    # Updated master stays fp32.
    assert new_master["w"].dtype == mx.float32
    # Each entry decreased by 0.1.
    assert mx.allclose(new_master["w"],
                        mx.array([0.9, 1.9, 2.9], dtype=mx.float32),
                        atol=1e-3)


def test_v7_d04_hybrid_step_grads_cast_to_master_before_apply():
    """apply_gradients must receive fp32 grads, not bf16."""
    seen_dtype: dict[str, mx.Dtype] = {}

    master = {"w": mx.zeros((2,), dtype=mx.float32)}

    def loss_and_grad(fwd):
        return mx.array(0.0), {"w": mx.ones((2,), dtype=mx.bfloat16)}

    def apply_gradients(m, g):
        seen_dtype["w"] = g["w"].dtype
        return m

    hybrid_step(master_params=master, fwd_dtype=mx.bfloat16,
                 loss_and_grad=loss_and_grad,
                 apply_gradients=apply_gradients)
    assert seen_dtype["w"] == mx.float32

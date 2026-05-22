"""V7-D03: LossScaler unit + integration tests."""

from __future__ import annotations

import mlx.core as mx
import pytest

from cppmega_v4.runtime.loss_scaler import LossScaler


def _grads_finite() -> dict:
    return {"w": mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)}


def _grads_with_inf() -> dict:
    return {"w": mx.array([[float("inf"), 0.0], [0.0, 0.0]],
                          dtype=mx.float32)}


def _grads_with_nan() -> dict:
    return {"w": mx.array([[float("nan"), 0.0], [0.0, 0.0]],
                          dtype=mx.float32)}


def test_v7_d03_static_scaler_init_and_scale():
    s = LossScaler(mode="static", init_scale=128.0)
    assert s.scale == 128.0
    grads = _grads_finite()
    scaled = s.scale_grads(grads)
    # Each entry multiplied by 128.
    assert float(scaled["w"][0, 0].item()) == pytest.approx(128.0)


def test_v7_d03_dynamic_scaler_halves_on_overflow():
    s = LossScaler(mode="dynamic", init_scale=512.0,
                    backoff_factor=0.5, min_scale=1.0)
    assert s.scale == 512.0
    _, overflow = s.unscale_and_check(_grads_with_inf())
    assert overflow is True
    s.update(overflow=True)
    assert s.scale == 256.0
    assert s.overflow_count == 1


def test_v7_d03_dynamic_scaler_doubles_after_growth_interval():
    s = LossScaler(mode="dynamic", init_scale=8.0,
                    growth_factor=2.0, growth_interval=3,
                    max_scale=1024.0)
    for _ in range(3):
        s.update(overflow=False)
    assert s.scale == 16.0
    for _ in range(3):
        s.update(overflow=False)
    assert s.scale == 32.0


def test_v7_d03_static_mode_counts_overflow_but_does_not_adjust():
    s = LossScaler(mode="static", init_scale=64.0)
    s.update(overflow=True)
    assert s.scale == 64.0  # unchanged
    assert s.overflow_count == 1


def test_v7_d03_unscale_inverts_scaling():
    s = LossScaler(mode="static", init_scale=4.0)
    grads = _grads_finite()
    scaled = s.scale_grads(grads)
    unscaled, overflow = s.unscale_and_check(scaled)
    assert overflow is False
    assert mx.allclose(unscaled["w"], grads["w"], atol=1e-6)


def test_v7_d03_nan_grads_also_detected_as_overflow():
    s = LossScaler(mode="dynamic", init_scale=128.0)
    _, overflow = s.unscale_and_check(_grads_with_nan())
    assert overflow is True


def test_v7_d03_dynamic_scaler_respects_min_max_bounds():
    s = LossScaler(mode="dynamic", init_scale=2.0,
                    backoff_factor=0.5, min_scale=1.0,
                    growth_factor=2.0, max_scale=4.0,
                    growth_interval=1)
    # Slam against min via repeated overflow.
    for _ in range(10):
        s.update(overflow=True)
    assert s.scale == 1.0
    # Slam against max via repeated clean steps.
    for _ in range(10):
        s.update(overflow=False)
    assert s.scale == 4.0


def test_v7_d03_snapshot_shape():
    s = LossScaler(mode="dynamic", init_scale=128.0)
    snap = s.snapshot()
    for k in ("mode", "scale", "overflow_count",
              "clean_steps_since_overflow"):
        assert k in snap
    assert snap["mode"] == "dynamic"
    assert snap["scale"] == 128.0

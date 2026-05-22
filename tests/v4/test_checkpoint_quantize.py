"""V7-C06: int8 + fp16 checkpoint compression helpers."""

from __future__ import annotations

import mlx.core as mx
import pytest

from cppmega_v4.runtime.checkpoint_quantize import (
    cast_fp16, dequantize_int8, quantize_int8, uncast_to_fp32,
)


def test_v7_c06_int8_round_trip_bounded_error():
    t = mx.random.normal(shape=(32, 32), key=mx.random.key(0))
    q, scale = quantize_int8(t)
    assert q.dtype == mx.int8
    deq = dequantize_int8(q, scale)
    # Quantisation error bounded by scale (one int8 step ≈ scale/2).
    err = float(mx.max(mx.abs(t - deq)).item())
    assert err <= scale * 1.0 + 1e-6


def test_v7_c06_int8_size_savings_4x():
    """fp32 → int8 should be 4x smaller in byte count."""
    t = mx.zeros((128,), dtype=mx.float32)
    q, _ = quantize_int8(t)
    # fp32 = 4 bytes, int8 = 1 byte; per-element 4x.
    assert q.dtype.size * 4 == 4  # int8 size * 4 == fp32 size


def test_v7_c06_fp16_round_trip_close_to_fp32():
    t = mx.random.normal(shape=(64,), key=mx.random.key(1))
    h = cast_fp16(t)
    back = uncast_to_fp32(h)
    # fp16 to fp32 round trip within fp16 precision.
    err = float(mx.max(mx.abs(t - back)).item())
    assert err < 1e-2


def test_v7_c06_int8_handles_zero_tensor():
    t = mx.zeros((4, 4), dtype=mx.float32)
    q, scale = quantize_int8(t)
    deq = dequantize_int8(q, scale)
    assert mx.allclose(deq, t, atol=1e-9)


def test_v7_c06_int8_handles_large_dynamic_range():
    t = mx.array([[-1000.0, 0.0, 1000.0], [-0.001, 0.0, 0.001]])
    q, scale = quantize_int8(t)
    deq = dequantize_int8(q, scale)
    # Large dynamic range: error bounded by scale (8/127 ≈ ).
    assert float(mx.max(mx.abs(t - deq)).item()) < scale * 1.1

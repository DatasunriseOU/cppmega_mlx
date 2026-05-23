"""V7-N06: dDequantizeBlockwise round-trip parity with the bnb LUT."""

from __future__ import annotations

import mlx.core as mx
import pytest

from cppmega_mlx.training._quantize_8bit import (
    DEFAULT_BLOCK_SIZE, QUANT_SCHEME_DYNAMIC,
    create_dynamic_map,
    dequantize_blockwise, dequantize_dynamic_lut_blockwise,
    quantize_blockwise, quantize_dynamic_lut_blockwise,
)


def test_create_dynamic_map_has_256_entries_in_unit_range():
    lut = create_dynamic_map()
    assert lut.shape == (256,)
    assert lut.dtype == mx.float32
    # Canonical bnb map covers [-1, 1].
    assert float(mx.min(lut).item()) >= -1.0 - 1e-6
    assert float(mx.max(lut).item()) <= 1.0 + 1e-6
    # Contains both 0 and 1 boundary entries.
    assert float(mx.min(mx.abs(lut)).item()) == 0.0


def test_dynamic_lut_round_trip_preserves_magnitude_within_5pct():
    # 4 blocks × 256 entries = 1024 elements.
    rng = mx.random.normal(shape=(1024,), key=mx.random.key(0xBEEF))
    qdata, absmax = quantize_dynamic_lut_blockwise(rng, DEFAULT_BLOCK_SIZE)
    assert qdata.dtype == mx.uint8
    assert qdata.shape == rng.shape
    assert absmax.shape == (1024 // DEFAULT_BLOCK_SIZE,)
    deq = dequantize_dynamic_lut_blockwise(qdata, absmax)
    mse = float(mx.mean((deq - rng) ** 2).item())
    rng_max = float(mx.max(mx.abs(rng)).item())
    # Round-trip RMSE within 5% of dynamic range — tighter than the
    # symmetric int8 path's bound near zero.
    assert mse ** 0.5 < 0.05 * rng_max


def test_dispatcher_routes_dynamic_scheme_through_lut():
    rng = mx.random.normal(shape=(256,), key=mx.random.key(0x42))
    q_d, am_d = quantize_blockwise(
        rng, scheme=QUANT_SCHEME_DYNAMIC)
    q_lut, am_lut = quantize_dynamic_lut_blockwise(rng)
    assert mx.array_equal(q_d, q_lut).item()
    assert mx.array_equal(am_d, am_lut).item()
    deq_d = dequantize_blockwise(q_d, am_d,
                                   scheme=QUANT_SCHEME_DYNAMIC)
    deq_lut = dequantize_dynamic_lut_blockwise(q_lut, am_lut)
    assert mx.array_equal(deq_d, deq_lut).item()


def test_zero_inputs_round_trip_to_zero():
    z = mx.zeros((256,), dtype=mx.float32)
    q, am = quantize_dynamic_lut_blockwise(z)
    deq = dequantize_dynamic_lut_blockwise(q, am)
    assert float(mx.max(mx.abs(deq)).item()) < 1e-6

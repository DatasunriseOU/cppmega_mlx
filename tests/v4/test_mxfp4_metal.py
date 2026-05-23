"""V8-R05 parity tests: MXFP4 (e2m1 block-scaled) codec.

Assertions:
  * LUT matches the OCP MX spec (16 entries, symmetric, max=6.0)
  * round-trip on a fp32 N(0,1) tensor keeps relative RMSE ≤ 0.15
    (e2m1's published typical error is ~7-10%; we set the bar at 15%
    so the test catches genuine regressions, not statistical noise)
  * tail block decodes correctly when numel is not a multiple of 16
  * SchemeRouter dispatch with QUANT_SCHEME_MXFP4 produces a usable
    payload that round-trips back through dequantize_blockwise
  * ShardingSpec mxfp4_enabled is mutually exclusive with fp8_enabled
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from cppmega_mlx.quant.mxfp4_metal import (
    MXFP4_BLOCK_SIZE, MXFP4_LUT, MXFP4_LUT_POSITIVE,
    dequantize_mxfp4_blockwise, mxfp4_round_trip,
    quantize_mxfp4_blockwise, quantize_round_trip_rmse,
)
from cppmega_mlx.training._quantize_8bit import (
    QUANT_SCHEME_MXFP4, dequantize_blockwise, quantize_blockwise,
)


def test_lut_matches_ocp_mx_spec():
    expected = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    assert list(MXFP4_LUT_POSITIVE) == expected
    assert len(MXFP4_LUT) == 16
    # Symmetric around zero (apart from the duplicate -0)
    assert float(MXFP4_LUT.max()) == 6.0
    assert float(MXFP4_LUT.min()) == -6.0


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_random_normal_round_trip_under_15_percent_rmse(seed: int):
    x = mx.random.normal(shape=(1024,), key=mx.random.key(seed))
    rel = quantize_round_trip_rmse(x)
    assert rel <= 0.15, (
        f"seed {seed}: relative RMSE {rel:.4f} > 0.15 — codec regression")
    # The codec actually adds something — assert it's nonzero or x is.
    assert rel > 0 or float(mx.sum(mx.abs(x))) == 0


def test_zero_tensor_round_trips_exactly():
    x = mx.zeros(64, dtype=mx.float32)
    rt = mxfp4_round_trip(x)
    assert float(mx.max(mx.abs(rt))) == 0.0


def test_constant_tensor_round_trips_in_codebook():
    """A tensor of exactly 1.0 round-trips to 1.0 (in the codebook)."""
    x = mx.ones(32, dtype=mx.float32) * 1.0
    rt = mxfp4_round_trip(x)
    np.testing.assert_allclose(np.asarray(rt), np.ones(32), atol=1e-6)


def test_tail_block_partial_decodes_correctly():
    """numel=20 → 1 full block + 4 elements of a tail block."""
    x = mx.array(np.linspace(-1.0, 1.0, 20).astype(np.float32))
    qdata, scales = quantize_mxfp4_blockwise(x)
    rt = dequantize_mxfp4_blockwise(qdata, scales, numel=20)
    assert rt.shape == (20,)
    # Per-element error never exceeds half the largest codebook step (1.0).
    diffs = np.abs(np.asarray(rt) - np.asarray(x))
    assert float(diffs.max()) < 1.0


def test_scheme_router_dispatches_mxfp4():
    x = mx.random.normal(shape=(128,), key=mx.random.key(3))
    qdata, scales = quantize_blockwise(
        x, scheme=QUANT_SCHEME_MXFP4, block_size=MXFP4_BLOCK_SIZE)
    rt = dequantize_blockwise(
        qdata, scales, scheme=QUANT_SCHEME_MXFP4,
        out_dtype=mx.float32, numel=128)
    assert rt.shape == (128,)
    rel = quantize_round_trip_rmse(x)
    rel_router = float(mx.sqrt(mx.mean(
        (rt - x.astype(mx.float32)) ** 2))) / (
            float(mx.sqrt(mx.mean(x ** 2))) + 1e-12)
    # Router path must match the direct-call RMSE within numerics.
    assert math.isclose(rel, rel_router, rel_tol=0.05, abs_tol=0.01)


def test_payload_byte_count_matches_spec():
    """Spec §8: 4 bits mantissa + 8 bits scale / 16 elements
    = 4.5 bits / element. For 256 elements that's 256/2 = 128 bytes
    of mantissa + 256/16 = 16 bytes of scale."""
    x = mx.zeros(256, dtype=mx.float32)
    qdata, scales = quantize_mxfp4_blockwise(x)
    assert qdata.size == 128, qdata.size
    assert scales.size == 16, scales.size


def test_sharding_spec_mxfp4_exclusive_with_fp8():
    from cppmega_v4.parallelism.sharding_spec import (
        AxisAssignment, ParallelismKind, ShardingSpec,
    )
    from cppmega_v4.parallelism.topology import h100_8x
    topo = h100_8x()
    with pytest.raises(ValueError, match="mutually exclusive"):
        ShardingSpec(
            topology=topo,
            axis_assignments=(AxisAssignment(
                axis_name="dp", kind=ParallelismKind.DP, degree=8),),
            fp8_enabled=True, mxfp4_enabled=True)


def test_sharding_spec_mxfp4_alone_constructs():
    from cppmega_v4.parallelism.sharding_spec import (
        AxisAssignment, ParallelismKind, ShardingSpec,
    )
    from cppmega_v4.parallelism.topology import h100_8x
    topo = h100_8x()
    s = ShardingSpec(
        topology=topo,
        axis_assignments=(AxisAssignment(
            axis_name="dp", kind=ParallelismKind.DP, degree=8),),
        mxfp4_enabled=True)
    assert s.mxfp4_enabled is True
    assert s.fp8_enabled is False

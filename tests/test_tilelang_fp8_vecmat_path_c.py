"""Tests for the Path C TileLang DSL FP8 vecmat reducer.

Path C targets the same M=1, transpose-B vecmat contract as Path B's
hand-written ``fp8_scaled_vecmat`` MSL kernel. These tests keep the DSL kernel
checked in and record the current Metal lowering gap: TileLang emits scalar
FP8 byte decodes and shuffle-style allreduce code, not Path B's packed uint32
loads plus literal ``simd_sum``.
"""

from __future__ import annotations

import pytest

from cppmega_mlx.nn._tilelang.fp8_vecmat_path_c import (
    FP8VecmatPathCStatus,
    fp8_vecmat_msl_features,
    fp8_vecmat_path_c_status,
    lower_fp8_vecmat_msl,
)


def test_status_reports_available_or_explains_why() -> None:
    status = fp8_vecmat_path_c_status()
    assert isinstance(status, FP8VecmatPathCStatus)
    assert isinstance(status.available, bool)
    assert isinstance(status.reason, str) and status.reason
    assert status.transpose_B is True
    assert status.m_equals_1 is True


def test_lowered_scalar_reducer_contains_kernel_and_fp8_decode() -> None:
    msl = lower_fp8_vecmat_msl(N=128, K=128)
    features = fp8_vecmat_msl_features(msl)
    assert features["kernel_void"] >= 1
    assert features["fp8_e4m3_decode_helper"] >= 1
    assert features["scalar_fp8_byte_decode"] >= 1
    assert "threadIdx" in msl or "thread_position" in msl


def test_lowered_scalar_reducer_records_packed_load_gap() -> None:
    msl = lower_fp8_vecmat_msl(N=128, K=128)
    features = fp8_vecmat_msl_features(msl)
    assert features["reinterpret_cast"] == 0
    assert features["device_const_uint"] == 0
    assert features["simd_sum"] == 0
    assert features["fp8_e4m3_decode_helper"] >= 1


def test_vectorized_probe_lowers_but_still_lacks_packed_uint_loads() -> None:
    msl = lower_fp8_vecmat_msl(N=128, K=128, vectorized_loads=True)
    features = fp8_vecmat_msl_features(msl)
    assert features["kernel_void"] >= 1
    assert features["fp8_e4m3_decode_helper"] >= 1
    assert features["reinterpret_cast"] == 0
    assert features["device_const_uint"] == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"N": 0, "K": 128},
        {"N": 128, "K": 0},
        {"N": 128, "K": 128, "outputs_per_block": 0},
        {"N": 128, "K": 128, "reduce_threads": 0},
        {"N": 128, "K": 128, "vec": 0},
    ],
)
def test_invalid_shapes_raise(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        lower_fp8_vecmat_msl(**kwargs)

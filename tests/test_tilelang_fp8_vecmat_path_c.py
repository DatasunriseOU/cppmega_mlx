# pyright: reportMissingImports=false
"""Tests for the Path C TileLang DSL FP8 vecmat reducer.

Path C targets the same M=1, transpose-B vecmat contract as Path B's
hand-written ``fp8_scaled_vecmat`` MSL kernel. These tests keep the DSL kernel
checked in and assert the default Metal lowering uses Path B-style packed
uint32 loads with a LUT decode. The single-warp allreduce should lower to
literal Metal ``simd_sum``.
"""

from __future__ import annotations

import numpy as np
import pytest
from typing import Any, cast

import mlx.core as mx

from cppmega_mlx.nn._tilelang.fp8_msl_kernels import fp8_scaled_vecmat
from cppmega_mlx.nn._tilelang.fp8_vecmat_path_c import (
    FP8VecmatPathCStatus,
    _fp8_vecmat_kernel_for,
    canonical_vecmat_runtime_body,
    fp8_scaled_vecmat_path_c,
    fp8_vecmat_msl_blockers,
    fp8_vecmat_msl_features,
    fp8_vecmat_path_c_status,
    lower_fp8_vecmat_msl,
)
from cppmega_mlx.nn._tilelang._msl_transform import (
    _assert_path_c_metal_fp8_intrinsics_registered,
)


def _metal_available() -> bool:
    metal = getattr(mx, "metal", None)
    return mx.default_device() == mx.gpu and metal is not None and metal.is_available()


def test_status_reports_available_or_explains_why() -> None:
    status = fp8_vecmat_path_c_status()
    assert isinstance(status, FP8VecmatPathCStatus)
    assert isinstance(status.available, bool)
    assert isinstance(status.reason, str) and status.reason
    assert status.transpose_B is True
    assert status.m_equals_1 is True


def test_lowered_default_reducer_contains_kernel_and_packed_lut_decode() -> None:
    msl = lower_fp8_vecmat_msl(N=128, K=128)
    features = fp8_vecmat_msl_features(msl)
    assert features["kernel_void"] >= 1
    assert features["fp8_e4m3_lut"] > 0
    assert features["metal_fp8_dot4_helper"] == 0
    assert features["scalar_fp8_byte_decode_calls"] == 0
    assert "thread_position_in_grid.x" in msl


def test_lowered_default_reducer_uses_packed_uint_loads_and_simd_sum() -> None:
    msl = lower_fp8_vecmat_msl(N=128, K=128)
    features = fp8_vecmat_msl_features(msl)
    assert features["reinterpret_cast"] > 0
    assert features["device_const_uint"] > 0
    assert features["simd_sum"] > 0
    assert features["simd_shuffle_down"] == 0
    assert features["fp8_e4m3_lut"] > 0


def test_lowered_default_reducer_uses_per_row_b_scale() -> None:
    msl = lower_fp8_vecmat_msl(N=128, K=128)
    assert "B_scale[0]" not in msl
    assert "B_scale[" in msl


def test_lowered_default_reducer_reports_path_b_fast_path_ready() -> None:
    msl = lower_fp8_vecmat_msl(N=128, K=128)
    blockers = fp8_vecmat_msl_blockers(msl)
    features = blockers["generated_features"]

    assert blockers["path_b_fast_path_ready"] is True
    assert blockers["missing"] == []
    assert features["simd_shuffle_down"] == 0
    assert features["simd_sum"] > 0
    assert features["reinterpret_cast"] > 0
    assert features["device_const_uint"] > 0
    assert features["fp8_e4m3_lut"] > 0
    assert features["metal_fp8_dot4_helper"] == 0
    assert features["scalar_fp8_byte_decode_calls"] == 0


def test_runtime_body_keeps_path_b_vecmat_hot_loop_and_scale_modes() -> None:
    per_row = canonical_vecmat_runtime_body(N=4096, K=4096, scale_w_per_row=True)
    scalar = canonical_vecmat_runtime_body(N=4096, K=4096, scale_w_per_row=False)
    features = fp8_vecmat_msl_features(per_row)

    assert "uint row = gid / 32u" in per_row
    assert features["simd_sum"] == 1
    assert features["reinterpret_cast"] == 2
    assert features["device_const_uint"] >= 2
    assert features["fp8_e4m3_lut"] >= 8
    assert features["scalar_fp8_byte_decode_calls"] == 0
    assert "device const uint* A4 = reinterpret_cast<device const uint*>(A)" in per_row
    assert (
        "device const uint* B4 = reinterpret_cast<device const uint*>(B + row_offset)"
        in per_row
    )
    assert "B_scale[row]" in per_row
    assert "B_scale[0]" not in per_row
    assert "B_scale[0]" in scalar
    assert "B_scale[row]" not in scalar


@pytest.mark.skipif(not _metal_available(), reason="Metal unavailable")
def test_runtime_kernel_uses_canonical_input_order_for_fast_tuple_dispatch() -> None:
    _kernel, _lowering, input_names, output_shape, _grid, threadgroup = (
        _fp8_vecmat_kernel_for(
            24,
            64,
            1,
            32,
            4,
            True,
        )
    )
    assert input_names == ["A", "A_scale", "B", "B_scale"]
    assert output_shape == (24,)
    assert threadgroup == (32, 1, 1)
    assert "thread_position_in_grid.x" in _lowering.body
    assert "blockIdx" not in _lowering.body
    assert "__tvm_fp8_e4m3_dot4_packed" not in _lowering.body
    assert "simd_sum(sum)" in _lowering.body


def test_fp8_e4m3_dot4_intrinsic_is_registered() -> None:
    """Fix-1 + Fix-A trip-wire: ``tirx.metal.fp8_e4m3_dot4`` must exist.

    The Path C FP8 vecmat macro PrimFunc emits a ``T.metal_fp8_e4m3_dot4``
    call which is lowered to the ``tirx.metal.fp8_e4m3_dot4`` op. If the
    op is not registered (Grok-D P0 from the 2026-05-06 audit), every
    Path C FP8 kernel silently falls back to scalar decode and CI stays
    green. This test makes that regression a hard failure on hosts with
    TVM available.

    Skips on hosts where TVM is not importable so CI without libz3 stays
    informative rather than red.
    """

    pytest.importorskip("tvm")
    try:
        from tvm.ir import Op  # type: ignore
    except Exception:
        try:
            from tilelang.tvm.ir import Op  # type: ignore
        except Exception as exc:
            pytest.skip(f"TVM Op import unavailable: {exc}")

    _assert_path_c_metal_fp8_intrinsics_registered()
    op = Op.get("tirx.metal.fp8_e4m3_dot4")
    assert op is not None, (
        "tirx.metal.fp8_e4m3_dot4 must be registered for Path C FP8 macro "
        "lowering. See cppmega_mlx.nn._tilelang._msl_transform."
        "_register_path_c_metal_fp8_intrinsics."
    )


def test_vectorized_probe_remains_scalar_fallback() -> None:
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
def test_invalid_shapes_raise(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        lower_fp8_vecmat_msl(**cast(Any, kwargs))


@pytest.mark.skipif(not _metal_available(), reason="Metal unavailable")
def test_path_c_vecmat_matches_path_b_scalar_scale() -> None:
    rng = np.random.default_rng(23)
    N, K = 24, 64
    x = mx.array((rng.standard_normal((K,)) * 0.1).astype(np.float32))
    W = mx.array((rng.standard_normal((N, K)) * 0.1).astype(np.float32))
    x_fp8 = mx.to_fp8(x)
    W_fp8 = mx.to_fp8(W)
    sx = mx.array([1.25], dtype=mx.float32)
    sw = mx.array([0.75], dtype=mx.float32)
    mx.eval(x_fp8, W_fp8, sx, sw)

    path_b = fp8_scaled_vecmat(x_fp8, W_fp8, scale_x=sx, scale_w=sw)
    path_c = fp8_scaled_vecmat_path_c(x_fp8, W_fp8, scale_x=sx, scale_w=sw)
    if path_c is None:
        pytest.skip("Path C TileLang/Metal dispatch unavailable")

    mx.eval(path_b, path_c)
    np.testing.assert_allclose(
        np.asarray(path_c), np.asarray(path_b), rtol=1e-5, atol=1e-5
    )


@pytest.mark.skipif(not _metal_available(), reason="Metal unavailable")
def test_path_c_vecmat_matches_path_b_per_row_scale() -> None:
    rng = np.random.default_rng(24)
    N, K = 24, 64
    x = mx.array((rng.standard_normal((K,)) * 0.1).astype(np.float32))
    W = mx.array((rng.standard_normal((N, K)) * 0.1).astype(np.float32))
    x_fp8 = mx.to_fp8(x)
    W_fp8 = mx.to_fp8(W)
    sx = mx.array([1.5], dtype=mx.float32)
    sw = mx.array(rng.uniform(0.5, 2.0, size=N).astype(np.float32))
    mx.eval(x_fp8, W_fp8, sx, sw)

    path_b = fp8_scaled_vecmat(x_fp8, W_fp8, scale_x=sx, scale_w=sw)
    path_c = fp8_scaled_vecmat_path_c(x_fp8, W_fp8, scale_x=sx, scale_w=sw)
    if path_c is None:
        pytest.skip("Path C TileLang/Metal dispatch unavailable")

    mx.eval(path_b, path_c)
    np.testing.assert_allclose(
        np.asarray(path_c), np.asarray(path_b), rtol=1e-5, atol=1e-5
    )


@pytest.mark.skipif(not _metal_available(), reason="Metal unavailable")
def test_path_c_vecmat_rejects_invalid_shapes() -> None:
    x_fp8 = mx.zeros((33,), dtype=mx.uint8)
    W_fp8 = mx.zeros((8, 33), dtype=mx.uint8)
    with pytest.raises(ValueError, match="multiple of 4"):
        fp8_scaled_vecmat_path_c(x_fp8, W_fp8, scale_x=1.0, scale_w=1.0)

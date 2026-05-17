# pyright: reportMissingImports=false
"""Parity + status tests for FP8 sparse-MLA reference and Path C surfaces.

The historical direct-MSL Path B kernel is retired. Callable prepared-buffer
Path C surfaces must route through TileLang/tvm-ffi, and float-carrier callers
fall back to the explicit MLX FP8 reference unless forced.

These tests exercise:

1. The direct-MSL status surface reports an explicit unsupported reason.
2. ``sparse_mla_fp8_apply`` falls back to the pure-MLX FP8 reference.
3. ``force_metal=True`` raises with the direct-MSL blocker.
4. The pure-MLX FP8 reference matches a "dequantize-then-BF16" parity oracle
   exactly.
5. The FP8 reference matches the original BF16 reference within FP8 noise
   tolerance (rtol=5e-3 on small inputs).
6. The MXFP8 quantized_matmul side-path runs and produces finite outputs.
7. MLX autograd flows backward through the FP8 path cleanly.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
from pathlib import Path
from typing import cast

import numpy as np
import pytest

import mlx.core as mx

from cppmega_mlx.nn._tilelang.sparse_mla_fp8 import (  # noqa: E402
    SparseMLAFp8MetalStatus,
    _from_fp8_with_scale,
    _to_fp8_with_per_tensor_scale,
    sparse_mla_fp8_apply,
    sparse_mla_fp8_bwd_metal,
    sparse_mla_fp8_fwd_metal,
    sparse_mla_fp8_metal_status,
    sparse_mla_fp8_reference,
    sparse_mla_quantized_matmul_reference,
)
from cppmega_mlx.nn._tilelang.fp8_msl_kernels import fp8_scaled_vecmat  # noqa: E402
from cppmega_mlx.nn._tilelang.sparse_mla_fp8_path_c import (  # noqa: E402
    SparseMLAFp8PathCDirectError,
    SparseMLAFp8IndexedQKReducePathCStatus,
    SparseMLAFp8QKReducePathCStatus,
    SparseMLAFp8PathCStatus,
    _to_fp8_with_per_token_scale,
    fp8_sparse_mla_indexed_qk_reduce_msl_features,
    fp8_sparse_mla_indexed_qk_reduce_path_c,
    fp8_sparse_mla_indexed_qk_reduce_path_c_status,
    fp8_sparse_mla_qk_reduce_msl_features,
    fp8_sparse_mla_qk_reduce_path_c,
    fp8_sparse_mla_qk_reduce_path_c_status,
    fp8_sparse_mla_qk_reduce_sync_plan,
    fp8_sparse_mla_qk_msl_features,
    fp8_sparse_mla_qk_path_c_status,
    fp8_sparse_mla_qk_scaled_matmul_probe_status,
    lower_fp8_sparse_mla_indexed_qk_reduce_msl,
    lower_fp8_sparse_mla_qk_reduce_msl,
    lower_fp8_sparse_mla_qk_msl,
    sparse_mla_fp8_bwd_path_c,
    sparse_mla_fp8_path_c_apply_from_float,
    sparse_mla_fp8_path_c_apply_prepared_float,
)
from cppmega_mlx.nn.attention import (  # noqa: E402
    causal_sparse_indices,
    sparse_indices_from_attention_mask,
)
from cppmega_mlx.nn.sparse_mla import sparse_mla_attention_reference  # noqa: E402


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _requires_native_tilelang_graph_outputs() -> None:
    if not mx.metal.is_available():
        pytest.skip("requires MLX Metal")


def _materialized_owner_buffer(shape: tuple[int, ...], dtype: mx.Dtype) -> mx.array:
    out = mx.zeros(shape, dtype=dtype)
    mx.eval(out)
    return out


def _make_inputs(
    *,
    batch: int = 1,
    seq_len: int = 4,
    heads: int = 2,
    kv_group: int = 1,
    qk_dim: int = 32,
    d_v: int = 16,
    topk: int = 8,
    seed: int = 0,
    scale: float = 0.1,
):
    rng = np.random.default_rng(seed)
    q = mx.array(
        (rng.standard_normal((batch, seq_len, heads, qk_dim)) * scale).astype(
            np.float32
        )
    )
    kv = mx.array(
        (rng.standard_normal((batch, seq_len, kv_group, qk_dim)) * scale).astype(
            np.float32
        )
    )
    ind_np = np.tile(
        np.arange(topk, dtype=np.int32).reshape(1, 1, 1, topk),
        (batch, seq_len, kv_group, 1),
    )
    # Mark the second half of the topk as masked (-1 sentinel).
    ind_np[:, :, :, topk // 2 :] = -1
    indices = mx.array(ind_np)
    return q, kv, indices, d_v


def _feature_int(features: dict[str, int | bool | str], key: str) -> int:
    value = features[key]
    assert isinstance(value, int) and not isinstance(value, bool)
    return value


def _array_only(value: object) -> mx.array:
    assert not isinstance(value, tuple)
    return cast(mx.array, value)


def _load_fp8_bench_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "bench_tilelang_sparse_mla_fp8.py"
    )
    spec = importlib.util.spec_from_file_location(
        "bench_tilelang_sparse_mla_fp8_for_test", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _finite_positive_float(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _indexed_qk_score_oracle_np(
    q_fp8: mx.array,
    q_scale: mx.array,
    kv_fp8: mx.array,
    kv_scale: mx.array,
    indices: mx.array,
    *,
    sm_scale: float,
) -> np.ndarray:
    q = mx.from_fp8(q_fp8, dtype=mx.float32) * q_scale.astype(mx.float32)[..., None]
    kv = mx.from_fp8(kv_fp8, dtype=mx.float32) * kv_scale.astype(mx.float32)[..., None]
    mx.eval(q, kv, indices)

    q_np = np.asarray(q).astype(np.float32)
    kv_np = np.asarray(kv).astype(np.float32)
    indices_np = np.asarray(indices).astype(np.int32)
    batch, seq_len, heads, k_dim = q_np.shape
    kv_group = kv_np.shape[2]
    topk = indices_np.shape[-1]
    head_kv = heads // kv_group
    scores = np.full((batch, seq_len, heads, topk), -np.inf, dtype=np.float32)
    for b in range(batch):
        for s in range(seq_len):
            for h in range(heads):
                group = h // head_kv
                for col in range(topk):
                    kv_pos = int(indices_np[b, s, group, col])
                    if 0 <= kv_pos < kv_np.shape[1]:
                        scores[b, s, h, col] = float(
                            np.dot(
                                q_np[b, s, h, :k_dim], kv_np[b, kv_pos, group, :k_dim]
                            )
                            * sm_scale
                        )
    return scores


# ---------------------------------------------------------------------------
# Status surface
# ---------------------------------------------------------------------------


def test_fp8_metal_status_reports_available() -> None:
    status = sparse_mla_fp8_metal_status()
    assert isinstance(status, SparseMLAFp8MetalStatus)
    assert status.available is False
    assert status.reason
    if mx.metal.is_available():
        assert "direct-MSL Path B is retired" in status.reason


def test_fp8_metal_status_with_arrays_validates_dispatcher_path() -> None:
    q, kv, indices, _ = _make_inputs()
    status = sparse_mla_fp8_metal_status(q, kv, indices)
    assert status.available is False
    assert status.reason


def test_fp8_sparse_mla_path_c_status_reports_dispatchable_qk_reducer() -> None:
    status = fp8_sparse_mla_qk_path_c_status()
    assert isinstance(status, SparseMLAFp8PathCStatus)
    assert status.m == 1
    assert status.n == 16
    assert status.k == 64
    assert status.transpose_B is True
    assert status.reason
    if not status.available:
        return
    # This status is intentionally reducer-only.  It must not satisfy the
    # full Sparse-MLA Path C dispatch gate.
    assert status.features["dispatch_surface"] == "qk_reduce"
    assert status.features.get("full_fwd_bwd_available") is not True
    assert status.features["legacy_fp8_scaled_matmul_probe_available"] is False
    assert "legacy_fp8_scaled_matmul_probe_reason" in status.features
    assert status.features["runnable_qk_reduce_available"] is True
    assert status.features["qk_shape"] == "m1_n_topk_k"
    assert status.features["signature_has_A_scale"] is True
    assert status.features["signature_has_B_scale"] is True
    assert _feature_int(status.features, "A_scale_refs") >= 1
    assert _feature_int(status.features, "B_scale_refs") >= 1
    assert _feature_int(status.features, "scalar_fp8_byte_decode_calls") == 0
    assert _feature_int(status.features, "metal_fp8_dot4_helper") >= 1
    assert (
        _feature_int(
            status.features,
            "legacy_fp8_scaled_matmul_probe_simdgroup_multiply_accumulate",
        )
        == 0
    )
    if status.features["legacy_fp8_scaled_matmul_probe_float_a_val"]:
        assert "scalar fallback" in status.features["legacy_fp8_scaled_matmul_probe_reason"]
    if status.features["legacy_fp8_scaled_matmul_probe_float_b_val"]:
        assert "scalar fallback" in status.features["legacy_fp8_scaled_matmul_probe_reason"]


def test_fp8_sparse_mla_path_c_legacy_scaled_matmul_probe_visible_and_fail_closed() -> (
    None
):
    status = fp8_sparse_mla_qk_scaled_matmul_probe_status()
    assert isinstance(status, SparseMLAFp8PathCStatus)
    assert status.m == 1
    assert status.n == 16
    assert status.k == 64
    assert status.transpose_B is True
    assert status.reason
    assert status.available is False
    if not status.features:
        return
    assert _feature_int(status.features, "simdgroup_multiply_accumulate") == 0
    if status.features["float_a_val"] or status.features["float_b_val"]:
        assert "scalar fallback" in status.reason
    assert "M=1/topk" in status.reason or "scalar fallback" in status.reason


def test_fp8_sparse_mla_path_c_scale_semantics_fail_closed() -> None:
    status = fp8_sparse_mla_qk_path_c_status()
    if not status.features:
        assert status.available is False
        return
    scale_refs_present = bool(status.features["A_scale_refs"]) and bool(
        status.features["B_scale_refs"]
    )
    scale_signature_present = bool(status.features["signature_has_A_scale"]) and bool(
        status.features["signature_has_B_scale"]
    )
    if status.available:
        assert scale_refs_present and scale_signature_present
        assert _feature_int(status.features, "scalar_fp8_byte_decode_calls") == 0
    else:
        assert (
            scale_refs_present and scale_signature_present
        ) or "scale operands disappeared" in status.reason


def test_fp8_sparse_mla_path_c_square_control_lowers_to_scale_aware_fast_path() -> None:
    status = fp8_sparse_mla_qk_path_c_status(
        M=32,
        N=32,
        K=64,
        BM=32,
        BN=32,
        BK=64,
        a_scale_size=1,
        b_scale_size=32,
    )
    if not status.available:
        assert status.reason
        return
    assert _feature_int(status.features, "simdgroup_multiply_accumulate") == 0
    assert _feature_int(status.features, "A_scale_refs") >= 1
    assert _feature_int(status.features, "B_scale_refs") >= 1
    assert status.features["signature_has_A_scale"] is True
    assert status.features["signature_has_B_scale"] is True


def test_fp8_sparse_mla_path_c_lowered_features_are_reported() -> None:
    msl = lower_fp8_sparse_mla_qk_msl(
        M=32, N=32, K=64, BM=32, BN=32, BK=64, b_scale_size=32
    )
    features = fp8_sparse_mla_qk_msl_features(msl)
    assert _feature_int(features, "kernel_void") >= 1
    assert _feature_int(features, "fp8_e4m3_decode_helper") >= 1
    assert _feature_int(features, "simdgroup_multiply_accumulate") == 0
    assert _feature_int(features, "A_scale_refs") >= 1
    assert _feature_int(features, "B_scale_refs") >= 1


def test_fp8_sparse_mla_path_c_qk_reduce_status_reports_available() -> None:
    status = fp8_sparse_mla_qk_reduce_path_c_status(N=16, K=64)
    assert isinstance(status, SparseMLAFp8QKReducePathCStatus)
    assert status.n == 16
    assert status.k == 64
    assert status.outputs_per_block == 16
    assert status.reduce_threads == 32
    assert status.vec == 4
    if not status.available:
        assert status.reason
        return
    assert status.features["signature_has_A_scale"] is True
    assert status.features["signature_has_B_scale"] is True
    assert _feature_int(status.features, "A_scale_refs") >= 1
    assert _feature_int(status.features, "B_scale_refs") >= 1
    assert status.features["sync_plan_strategy"] == "simdgroup_async"
    assert status.features["sync_plan_barrier_count"] == 0
    assert status.features["sync_plan_reduction_isolated"] is True


def test_fp8_sparse_mla_path_c_qk_reduce_status_preserves_explicit_schedule() -> None:
    status = fp8_sparse_mla_qk_reduce_path_c_status(
        N=16,
        K=64,
        outputs_per_block=4,
        reduce_threads=4,
        vec=4,
    )
    assert status.outputs_per_block == 4
    assert status.reduce_threads == 4
    assert status.vec == 4
    plan = fp8_sparse_mla_qk_reduce_sync_plan(
        N=16,
        K=64,
        outputs_per_block=4,
        reduce_threads=4,
        vec=4,
    )
    assert plan.strategy == "threadgroup_sync"
    if not status.available:
        assert status.reason
        return
    assert status.features["sync_plan_strategy"] == "threadgroup_sync"
    assert _feature_int(status.features, "sync_plan_barrier_count") >= 1


def test_fp8_sparse_mla_path_c_qk_reduce_lowered_features_are_reported() -> None:
    msl = lower_fp8_sparse_mla_qk_reduce_msl(N=16, K=64)
    features = fp8_sparse_mla_qk_reduce_msl_features(msl)
    assert _feature_int(features, "kernel_void") >= 1
    assert _feature_int(features, "fp8_e4m3_decode_helper") >= 1
    assert features["signature_has_A_scale"] is True
    assert features["signature_has_B_scale"] is True
    assert _feature_int(features, "A_scale_refs") >= 1
    assert _feature_int(features, "B_scale_refs") >= 1
    assert features["per_row_B_scale"] is True
    assert _feature_int(features, "scalar_fp8_byte_decode_calls") == 0
    assert (
        _feature_int(features, "simd_sum")
        + _feature_int(features, "simd_shuffle_down")
        + _feature_int(features, "tvm_thread_allreduce")
    ) >= 1
    assert _feature_int(features, "reinterpret_cast") == 0
    assert _feature_int(features, "device_const_uint") == 0
    assert _feature_int(features, "fp8_e4m3_lut") == 0
    assert _feature_int(features, "metal_fp8_dot4_helper") >= 1
    tuned_msl = lower_fp8_sparse_mla_qk_reduce_msl(
        N=16,
        K=64,
        outputs_per_block=16,
        reduce_threads=32,
        vec=4,
    )
    assert "reduced[0] = simd_sum(accum[0])" in tuned_msl
    assert not re.search(r"C\[[^\n;]*=\s*[^\n;]*simd_sum\(", tuned_msl)


def test_fp8_sparse_mla_path_c_qk_reduce_matches_dequant_oracle() -> None:
    status = fp8_sparse_mla_qk_reduce_path_c_status(N=16, K=64)
    if not status.available:
        pytest.skip(status.reason)

    q, kv, _indices, _d_v = _make_inputs(
        seq_len=16, heads=2, qk_dim=64, topk=16, scale=0.1
    )
    q_fp8, q_scale = _to_fp8_with_per_tensor_scale(q)
    kv_fp8, kv_scale = _to_fp8_with_per_tensor_scale(kv)
    mx.eval(q_fp8, q_scale, kv_fp8, kv_scale)

    A_fp8 = mx.contiguous(q_fp8[0, 0, 0, :].reshape((1, 64)))
    A_scale = mx.contiguous(q_scale[0, 0, 0].reshape((1,)))
    B_fp8 = mx.contiguous(kv_fp8[0, :16, 0, :])
    B_scale = mx.contiguous(kv_scale[0, :16, 0])
    out = fp8_sparse_mla_qk_reduce_path_c(A_fp8, A_scale, B_fp8, B_scale)
    assert out is not None

    oracle = mx.matmul(
        mx.from_fp8(A_fp8, dtype=mx.float32),
        mx.swapaxes(mx.from_fp8(B_fp8, dtype=mx.float32), 0, 1),
    )
    oracle = (
        oracle
        * A_scale.reshape((1, 1)).astype(mx.float32)
        * B_scale.reshape((1, 16)).astype(mx.float32)
    )
    mx.eval(out, oracle)
    np.testing.assert_allclose(
        np.asarray(out).astype(np.float32),
        np.asarray(oracle).astype(np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_fp8_sparse_mla_path_c_qk_reduce_supports_scalar_b_scale_without_broadcast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = fp8_sparse_mla_qk_reduce_path_c_status(N=16, K=64)
    if not status.available:
        pytest.skip(status.reason)

    import cppmega_mlx.nn._tilelang.sparse_mla_fp8_path_c as path_c_module

    q, kv, _indices, _d_v = _make_inputs(
        seq_len=16, heads=2, qk_dim=64, topk=16, scale=0.1, seed=11
    )
    q_fp8, q_scale = _to_fp8_with_per_tensor_scale(q)
    kv_fp8, kv_scale = _to_fp8_with_per_tensor_scale(kv)
    mx.eval(q_fp8, q_scale, kv_fp8, kv_scale)

    A_fp8 = mx.contiguous(q_fp8[0, 0, 0, :].reshape((1, 64)))
    A_scale = mx.contiguous(q_scale[0, 0, 0].reshape((1,)))
    B_fp8 = mx.contiguous(kv_fp8[0, :16, 0, :])
    B_scale_scalar = mx.contiguous(kv_scale[0, 0, 0].reshape((1,)))

    def fail_hidden_broadcast(*args: object, **kwargs: object) -> mx.array:
        del args, kwargs
        raise AssertionError("scalar B_scale must not be broadcast in Python")

    def fail_hidden_compact(*args: object, **kwargs: object) -> mx.array:
        del args, kwargs
        raise AssertionError("Path C dispatch must not compact inputs in Python")

    with monkeypatch.context() as mp:
        mp.setattr(path_c_module.mx, "ones", fail_hidden_broadcast)
        mp.setattr(path_c_module.mx, "contiguous", fail_hidden_compact)
        out = fp8_sparse_mla_qk_reduce_path_c(A_fp8, A_scale, B_fp8, B_scale_scalar)
    assert out is not None

    oracle = mx.matmul(
        mx.from_fp8(A_fp8, dtype=mx.float32),
        mx.swapaxes(mx.from_fp8(B_fp8, dtype=mx.float32), 0, 1),
    )
    oracle = (
        oracle
        * A_scale.reshape((1, 1)).astype(mx.float32)
        * B_scale_scalar.reshape((1, 1)).astype(mx.float32)
    )
    path_b = fp8_scaled_vecmat(
        A_fp8.reshape((64,)),
        B_fp8,
        scale_x=A_scale,
        scale_w=B_scale_scalar,
    ).reshape((1, 16))
    mx.eval(out, oracle, path_b)
    np.testing.assert_allclose(
        np.asarray(out).astype(np.float32),
        np.asarray(oracle).astype(np.float32),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(out).astype(np.float32),
        np.asarray(path_b).astype(np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_fp8_sparse_mla_path_c_indexed_qk_reduce_status_and_features() -> None:
    status = fp8_sparse_mla_indexed_qk_reduce_path_c_status(
        batch=1,
        seq_len=2,
        heads=2,
        seq_len_kv=4,
        kv_group=1,
        topk=4,
        K=64,
    )
    assert isinstance(status, SparseMLAFp8IndexedQKReducePathCStatus)
    assert status.head_kv == 2
    assert status.outputs_per_block == 2
    assert status.reduce_threads == 8
    assert status.vec == 4
    if not mx.metal.is_available():
        assert status.available is False
        assert status.reason
        return
    if not status.available:
        assert status.reason
        return
    assert status.features["qk_shape"] == "indexed_b_s_h_topk_k"
    assert status.features["signature_has_q_scale"] is True
    assert status.features["signature_has_kv_scale"] is True
    assert status.features["signature_has_indices"] is True
    assert status.features["signature_has_sm_scale"] is True
    assert status.features["invalid_index_guard"] is True
    assert _feature_int(status.features, "q_scale_refs") >= 1
    assert _feature_int(status.features, "kv_scale_refs") >= 1
    assert _feature_int(status.features, "indices_refs") >= 1
    assert _feature_int(status.features, "sm_scale_refs") >= 1
    assert _feature_int(status.features, "scalar_fp8_byte_decode_calls") == 0
    assert _feature_int(status.features, "reinterpret_cast") == 0
    assert _feature_int(status.features, "device_const_uint") == 0
    assert _feature_int(status.features, "fp8_e4m3_lut") == 0
    assert _feature_int(status.features, "metal_fp8_dot4_helper") >= 1


def test_fp8_sparse_mla_path_c_indexed_qk_reduce_status_tunes_default_topk16_schedule() -> (
    None
):
    status = fp8_sparse_mla_indexed_qk_reduce_path_c_status(
        batch=1,
        seq_len=1,
        heads=4,
        seq_len_kv=64,
        kv_group=1,
        topk=16,
        K=64,
    )
    assert status.outputs_per_block == 16
    assert status.reduce_threads == 32
    assert status.vec == 4
    if not status.available:
        assert status.reason


def test_fp8_sparse_mla_path_c_indexed_qk_reduce_lowered_features_are_reported() -> (
    None
):
    status = fp8_sparse_mla_indexed_qk_reduce_path_c_status(
        batch=1,
        seq_len=2,
        heads=2,
        seq_len_kv=4,
        kv_group=1,
        topk=4,
        K=64,
    )
    if not status.available:
        pytest.skip(status.reason)

    msl = lower_fp8_sparse_mla_indexed_qk_reduce_msl(
        batch=1,
        seq_len=2,
        heads=2,
        seq_len_kv=4,
        kv_group=1,
        topk=4,
        K=64,
    )
    features = fp8_sparse_mla_indexed_qk_reduce_msl_features(msl)
    assert _feature_int(features, "kernel_void") >= 1
    assert features["qk_shape"] == "indexed_b_s_h_topk_k"
    assert features["signature_has_q_scale"] is True
    assert features["signature_has_kv_scale"] is True
    assert features["signature_has_indices"] is True
    assert features["signature_has_sm_scale"] is True
    assert features["invalid_index_guard"] is True
    assert _feature_int(features, "q_scale_refs") >= 1
    assert _feature_int(features, "kv_scale_refs") >= 1
    assert _feature_int(features, "indices_refs") >= 1
    assert _feature_int(features, "sm_scale_refs") >= 1
    assert _feature_int(features, "scalar_fp8_byte_decode_calls") == 0
    assert _feature_int(features, "reinterpret_cast") == 0
    assert _feature_int(features, "device_const_uint") == 0
    assert _feature_int(features, "fp8_e4m3_lut") == 0
    assert _feature_int(features, "metal_fp8_dot4_helper") >= 1
    assert (
        _feature_int(features, "simd_sum")
        + _feature_int(features, "simd_shuffle_down")
        + _feature_int(features, "tvm_thread_allreduce")
    ) >= 1


def test_fp8_sparse_mla_path_c_indexed_qk_reduce_matches_path_b_index_contract() -> (
    None
):
    status = fp8_sparse_mla_indexed_qk_reduce_path_c_status(
        batch=1,
        seq_len=4,
        heads=2,
        seq_len_kv=4,
        kv_group=1,
        topk=4,
        K=64,
        outputs_per_block=2,
        reduce_threads=32,
        vec=4,
    )
    if not status.available:
        pytest.skip(status.reason)

    q, kv, indices, _d_v = _make_inputs(
        seq_len=4, heads=2, qk_dim=64, topk=4, scale=0.1, seed=21
    )
    q_fp8, q_scale = _to_fp8_with_per_tensor_scale(q)
    kv_fp8, kv_scale = _to_fp8_with_per_tensor_scale(kv)
    q_fp8 = mx.contiguous(q_fp8)
    q_scale = mx.contiguous(q_scale)
    kv_fp8 = mx.contiguous(kv_fp8)
    kv_scale = mx.contiguous(kv_scale)
    indices = mx.contiguous(indices)
    sm_scale = 0.125
    mx.eval(q_fp8, q_scale, kv_fp8, kv_scale, indices)

    out = fp8_sparse_mla_indexed_qk_reduce_path_c(
        q_fp8,
        q_scale,
        kv_fp8,
        kv_scale,
        indices,
        sm_scale=sm_scale,
        outputs_per_block=2,
        reduce_threads=32,
        vec=4,
    )
    assert out is not None
    mx.eval(out)

    oracle = _indexed_qk_score_oracle_np(
        q_fp8, q_scale, kv_fp8, kv_scale, indices, sm_scale=sm_scale
    )
    actual = np.asarray(out).astype(np.float32)
    indices_np = np.asarray(indices).astype(np.int32)
    head_kv = actual.shape[2] // indices_np.shape[2]
    invalid = np.repeat(indices_np == -1, repeats=head_kv, axis=2)
    assert invalid.shape == actual.shape
    assert np.all(actual[invalid] <= -3.0e38)
    np.testing.assert_allclose(actual[~invalid], oracle[~invalid], rtol=1e-5, atol=1e-5)


def test_fp8_sparse_mla_path_c_indexed_qk_reduce_masks_oob_indices() -> None:
    status = fp8_sparse_mla_indexed_qk_reduce_path_c_status(
        batch=1,
        seq_len=4,
        heads=2,
        seq_len_kv=4,
        kv_group=1,
        topk=4,
        K=64,
        outputs_per_block=2,
        reduce_threads=32,
        vec=4,
    )
    if not status.available:
        pytest.skip(status.reason)

    q, kv, indices, _d_v = _make_inputs(
        seq_len=4, heads=2, qk_dim=64, topk=4, scale=0.1, seed=23
    )
    indices_np = np.asarray(indices).astype(np.int32)
    indices_np[0, 0, 0, 1] = 99
    indices = mx.array(indices_np, dtype=mx.int32)
    q_fp8, q_scale = _to_fp8_with_per_tensor_scale(q)
    kv_fp8, kv_scale = _to_fp8_with_per_tensor_scale(kv)
    q_fp8 = mx.contiguous(q_fp8)
    q_scale = mx.contiguous(q_scale)
    kv_fp8 = mx.contiguous(kv_fp8)
    kv_scale = mx.contiguous(kv_scale)
    indices = mx.contiguous(indices)
    sm_scale = 0.125
    mx.eval(q_fp8, q_scale, kv_fp8, kv_scale, indices)

    out = fp8_sparse_mla_indexed_qk_reduce_path_c(
        q_fp8,
        q_scale,
        kv_fp8,
        kv_scale,
        indices,
        sm_scale=sm_scale,
        outputs_per_block=2,
        reduce_threads=32,
        vec=4,
    )
    assert out is not None
    mx.eval(out)

    oracle = _indexed_qk_score_oracle_np(
        q_fp8, q_scale, kv_fp8, kv_scale, indices, sm_scale=sm_scale
    )
    actual = np.asarray(out).astype(np.float32)
    indices_np = np.asarray(indices).astype(np.int32)
    head_kv = actual.shape[2] // indices_np.shape[2]
    invalid = np.repeat(
        (indices_np < 0) | (indices_np >= kv.shape[1]), repeats=head_kv, axis=2
    )
    assert invalid.shape == actual.shape
    assert np.all(actual[invalid] <= -3.0e38)
    np.testing.assert_allclose(actual[~invalid], oracle[~invalid], rtol=1e-5, atol=1e-5)


def test_fp8_sparse_mla_bench_strict_gate_enforces_path_c_perf_and_parity() -> None:
    bench = _load_fp8_bench_module()
    payload = {
        "metal_status": {"available": True, "dispatch_reason": "ok"},
        "path_c_tilelang_qk_reduce_status": {"available": True, "reason": "ok"},
        "path_c_tilelang_indexed_qk_reduce_status": {"available": True, "reason": "ok"},
        "parity": {
            "path_c_qk_reduce_vs_oracle": {"max_abs_err": 0.0},
            "path_c_qk_reduce_vs_path_b_qk_vecmat": {"max_abs_err": 0.0},
            "path_c_indexed_qk_reduce_vs_oracle": {
                "max_abs_err": 1e-6,
                "invalid_mismatch_count": 0,
            },
        },
        "ratios": {
            "path_c_qk_reduce_over_path_b_qk_vecmat": 0.99,
            "path_c_indexed_qk_reduce_over_path_b_fwd": 1.0,
        },
    }
    assert bench._strict_failures(payload, max_abs_err=1e-5, max_ratio=1.0) == []

    payload["ratios"]["path_c_qk_reduce_over_path_b_qk_vecmat"] = 1.01
    payload["parity"]["path_c_indexed_qk_reduce_vs_oracle"][
        "invalid_mismatch_count"
    ] = 1
    failures = bench._strict_failures(payload, max_abs_err=1e-5, max_ratio=1.0)
    assert any(
        "path_c_qk_reduce_over_path_b_qk_vecmat" in failure for failure in failures
    )
    assert any("invalid_mismatch_count=1" in failure for failure in failures)


def test_fp8_sparse_mla_full_dispatch_gate_rejects_reducer_only_status() -> None:
    bench = _load_fp8_bench_module()
    payload = {
        "path_c_tilelang_qk_status": {
            "available": True,
            "reason": "reducer-only",
            "features": {
                "dispatch_surface": "qk_reduce",
                "runnable_qk_reduce_available": True,
            },
        }
    }

    failures = bench._full_dispatch_strict_failures(
        payload,
        status_key="path_c_tilelang_qk_status",
        label="FP8",
    )
    assert failures == [
        "path_c_tilelang_qk_status.features.dispatch_surface='qk_reduce' "
        "is not full_fwd_bwd Path C FP8 dispatch",
        "path_c_tilelang_qk_status.features.full_fwd_bwd_available is not true",
    ]


def test_fp8_sparse_mla_full_dispatch_gate_rejects_reducer_only_e8m0_status() -> None:
    bench = _load_fp8_bench_module()
    payload = {
        "path_c_tilelang_e8m0_qk_status": {
            "available": True,
            "reason": "reducer-only",
            "features": {
                "dispatch_surface": "qk_reduce",
                "runnable_qk_reduce_available": True,
            },
        }
    }

    failures = bench._full_dispatch_strict_failures(
        payload,
        status_key="path_c_tilelang_e8m0_qk_status",
        label="blockscaled",
    )
    assert failures == [
        "path_c_tilelang_e8m0_qk_status.features.dispatch_surface='qk_reduce' "
        "is not full_fwd_bwd Path C blockscaled dispatch",
        "path_c_tilelang_e8m0_qk_status.features.full_fwd_bwd_available is not true",
    ]


def test_fp8_sparse_mla_full_dispatch_gate_requires_full_fwd_bwd_available_flag() -> (
    None
):
    bench = _load_fp8_bench_module()
    payload = {
        "path_c_tilelang_qk_status": {
            "available": True,
            "reason": "surface-only",
            "features": {
                "dispatch_surface": "full_fwd_bwd",
                "runnable_qk_reduce_available": True,
            },
        }
    }

    assert bench._full_dispatch_strict_failures(
        payload,
        status_key="path_c_tilelang_qk_status",
        label="FP8",
    ) == ["path_c_tilelang_qk_status.features.full_fwd_bwd_available is not true"]


def test_fp8_sparse_mla_checked_receipt_keeps_path_c_qk_claims_honest() -> None:
    receipt_path = (
        Path(__file__).resolve().parents[1]
        / "bench"
        / "tilelang_ports"
        / "sparse_mla_fp8.json"
    )
    payload = json.loads(receipt_path.read_text())

    assert payload["shape"]["q_shape"] == [1, 64, 4, 64]
    assert payload["shape"]["kv_shape"] == [1, 64, 1, 64]
    assert payload["shape"]["indices_shape"] == [1, 64, 1, 16]
    assert payload["strict"]["passed"] is False
    assert payload["strict"]["scope"] == "full_path_c_dispatch"
    assert payload["strict"]["failures"] == [
        "path_c_tilelang_qk_status.features.dispatch_surface='qk_reduce' "
        "is not full_fwd_bwd Path C FP8 dispatch",
        "path_c_tilelang_qk_status.features.full_fwd_bwd_available is not true",
    ]
    qk_reducer_strict = payload["qk_reducer_strict"]
    assert qk_reducer_strict["scope"] == "qk_reducer_dispatch"
    assert qk_reducer_strict["passed"] is True
    assert qk_reducer_strict["failures"] == []

    qk_status = payload["path_c_tilelang_qk_reduce_status"]
    indexed_status = payload["path_c_tilelang_indexed_qk_reduce_status"]
    dispatch_status = payload["path_c_tilelang_qk_status"]
    scaled_matmul_probe = payload["path_c_tilelang_qk_scaled_matmul_probe_status"]
    assert dispatch_status["available"] is True
    assert dispatch_status["features"]["dispatch_surface"] == "qk_reduce"
    assert dispatch_status["features"].get("full_fwd_bwd_available") is not True
    assert dispatch_status["features"]["runnable_qk_reduce_available"] is True
    assert (
        dispatch_status["features"]["legacy_fp8_scaled_matmul_probe_available"] is False
    )
    assert scaled_matmul_probe["available"] is False
    assert scaled_matmul_probe["m"] == 1
    assert scaled_matmul_probe["n"] == 16
    assert scaled_matmul_probe["k"] == 64
    assert scaled_matmul_probe["features"]["simdgroup_multiply_accumulate"] == 0
    if (
        scaled_matmul_probe["features"]["float_a_val"]
        or scaled_matmul_probe["features"]["float_b_val"]
    ):
        assert "scalar fallback" in scaled_matmul_probe["reason"]
    assert qk_status["available"] is True
    assert indexed_status["available"] is True
    assert qk_status["features"]["qk_shape"] == "m1_n_topk_k"
    assert indexed_status["features"]["qk_shape"] == "indexed_b_s_h_topk_k"
    assert indexed_status["features"]["invalid_index_guard"] is True

    parity = payload["parity"]
    assert parity["path_c_qk_reduce_vs_oracle"]["max_abs_err"] <= 1e-5
    assert parity["path_c_qk_reduce_vs_path_b_qk_vecmat"]["max_abs_err"] <= 1e-5
    indexed_parity = parity["path_c_indexed_qk_reduce_vs_oracle"]
    assert indexed_parity["max_abs_err"] <= 1e-5
    assert indexed_parity["invalid_mismatch_count"] == 0

    ratios = payload["ratios"]
    assert 0.0 < ratios["path_c_qk_reduce_over_path_b_qk_vecmat"] <= 1.0
    assert _finite_positive_float(ratios["path_c_indexed_qk_reduce_over_path_b_fwd"])


def test_fp8_fwd_metal_is_explicitly_unsupported() -> None:
    q, kv, indices, d_v = _make_inputs()
    result = sparse_mla_fp8_fwd_metal(q, kv, indices, d_v=d_v)
    assert result is None
    assert sparse_mla_fp8_metal_status(q, kv, indices).available is False


def test_fp8_bwd_metal_is_explicitly_unsupported() -> None:
    q, kv, indices, d_v = _make_inputs()
    d_out = mx.zeros((1, 4, 2, d_v), dtype=mx.float32)
    grads = sparse_mla_fp8_bwd_metal(q, kv, d_out, indices, d_v=d_v)
    assert grads is None


def test_fp8_path_c_bwd_owner_outputs_must_be_passed_as_a_pair() -> None:
    _requires_native_tilelang_graph_outputs()

    q, kv, indices, d_v = _make_inputs(scale=0.1)
    q_fp8 = mx.zeros(tuple(q.shape), dtype=mx.uint8)
    kv_fp8 = mx.zeros(tuple(kv.shape), dtype=mx.uint8)
    q_scale = mx.ones(tuple(q.shape[:-1]), dtype=mx.float32)
    kv_scale = mx.ones(tuple(kv.shape[:-1]), dtype=mx.float32)
    d_out = mx.zeros(tuple(q.shape[:3]) + (d_v,), dtype=mx.float32)
    dq_buffer = _materialized_owner_buffer(tuple(q.shape), mx.float32)

    with pytest.raises(
        SparseMLAFp8PathCDirectError,
        match="requires both dq_buffer and dkv_buffer",
    ):
        sparse_mla_fp8_bwd_path_c(
            q_fp8,
            q_scale,
            kv_fp8,
            kv_scale,
            d_out,
            indices,
            sm_scale=q.shape[-1] ** -0.5,
            d_v=d_v,
            dq_buffer=dq_buffer,
        )


def test_fp8_path_c_bwd_owner_outputs_run_through_atomic_tvm_ffi() -> None:
    _requires_native_tilelang_graph_outputs()

    q, kv, indices, d_v = _make_inputs(scale=0.1)
    q_fp8 = mx.zeros(tuple(q.shape), dtype=mx.uint8)
    kv_fp8 = mx.zeros(tuple(kv.shape), dtype=mx.uint8)
    q_scale = mx.ones(tuple(q.shape[:-1]), dtype=mx.float32)
    kv_scale = mx.ones(tuple(kv.shape[:-1]), dtype=mx.float32)
    d_out = mx.zeros(tuple(q.shape[:3]) + (d_v,), dtype=mx.float32)
    dq_buffer = _materialized_owner_buffer(tuple(q.shape), mx.float32)
    dkv_buffer = _materialized_owner_buffer(tuple(kv.shape), mx.float32)

    result = sparse_mla_fp8_bwd_path_c(
        q_fp8,
        q_scale,
        kv_fp8,
        kv_scale,
        d_out,
        indices,
        sm_scale=q.shape[-1] ** -0.5,
        d_v=d_v,
        dq_buffer=dq_buffer,
        dkv_buffer=dkv_buffer,
    )
    assert result == (dq_buffer, dkv_buffer)

    forced_result = sparse_mla_fp8_bwd_path_c(
        q_fp8,
        q_scale,
        kv_fp8,
        kv_scale,
        d_out,
        indices,
        sm_scale=q.shape[-1] ** -0.5,
        d_v=d_v,
        force_path_c=True,
        dq_buffer=dq_buffer,
        dkv_buffer=dkv_buffer,
    )
    assert forced_result == (dq_buffer, dkv_buffer)
    mx.eval(dq_buffer, dkv_buffer)
    assert np.all(np.isfinite(np.asarray(dq_buffer)))
    assert np.all(np.isfinite(np.asarray(dkv_buffer)))


def test_fp8_path_c_bwd_without_owner_outputs_clears_atomic_dkv() -> None:
    _requires_native_tilelang_graph_outputs()

    q, kv, indices, d_v = _make_inputs(scale=0.1)
    q = q.astype(mx.bfloat16)
    kv = kv.astype(mx.bfloat16)
    q_fp8, q_scale = _to_fp8_with_per_token_scale(q)
    kv_fp8, kv_scale = _to_fp8_with_per_token_scale(kv)
    d_out = mx.array(
        np.linspace(
            -0.25,
            0.25,
            num=int(q.shape[0] * q.shape[1] * q.shape[2] * d_v),
            dtype=np.float32,
        ).reshape(tuple(q.shape[:3]) + (d_v,))
    )

    dq_buffer = _materialized_owner_buffer(tuple(q.shape), mx.float32)
    dkv_buffer = _materialized_owner_buffer(tuple(kv.shape), mx.float32)
    owner_result = sparse_mla_fp8_bwd_path_c(
        q_fp8,
        q_scale,
        kv_fp8,
        kv_scale,
        d_out,
        indices,
        sm_scale=q.shape[-1] ** -0.5,
        d_v=d_v,
        force_path_c=True,
        dq_buffer=dq_buffer,
        dkv_buffer=dkv_buffer,
    )
    assert owner_result == (dq_buffer, dkv_buffer)
    mx.eval(dq_buffer, dkv_buffer)

    poison = mx.full(tuple(kv.shape), float("nan"), dtype=mx.float32)
    mx.eval(poison)
    del poison

    result = sparse_mla_fp8_bwd_path_c(
        q_fp8,
        q_scale,
        kv_fp8,
        kv_scale,
        d_out,
        indices,
        sm_scale=q.shape[-1] ** -0.5,
        d_v=d_v,
        force_path_c=True,
    )

    assert result is not None
    dq, dkv = result
    assert tuple(dq.shape) == tuple(q.shape)
    assert tuple(dkv.shape) == tuple(kv.shape)
    assert dq.dtype == mx.float32
    assert dkv.dtype == mx.float32
    mx.eval(dq, dkv)
    assert np.all(np.isfinite(np.asarray(dq)))
    assert np.all(np.isfinite(np.asarray(dkv)))
    np.testing.assert_allclose(
        np.asarray(dq), np.asarray(dq_buffer), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(dkv), np.asarray(dkv_buffer), rtol=1e-5, atol=1e-5
    )


def test_fp8_path_c_prepared_float_vjp_uses_native_graph_outputs() -> None:
    _requires_native_tilelang_graph_outputs()

    q, kv, indices, d_v = _make_inputs(scale=0.1)
    q = q.astype(mx.bfloat16)
    kv = kv.astype(mx.bfloat16)
    q_fp8 = mx.zeros(tuple(q.shape), dtype=mx.uint8)
    kv_fp8 = mx.zeros(tuple(kv.shape), dtype=mx.uint8)
    q_scale = mx.ones(tuple(q.shape[:-1]), dtype=mx.float32)
    kv_scale = mx.ones(tuple(kv.shape[:-1]), dtype=mx.float32)

    def loss_fn(q_in: mx.array, kv_in: mx.array) -> mx.array:
        out = sparse_mla_fp8_path_c_apply_prepared_float(
            q_in,
            kv_in,
            q_fp8,
            q_scale,
            kv_fp8,
            kv_scale,
            indices,
            sm_scale=q.shape[-1] ** -0.5,
            d_v=d_v,
            force_path_c=True,
        )
        return mx.sum(out.astype(mx.float32))

    loss, grads = mx.value_and_grad(loss_fn, argnums=(0, 1))(q, kv)
    dq, dkv = grads
    mx.eval(loss, dq, dkv)
    assert tuple(dq.shape) == tuple(q.shape)
    assert tuple(dkv.shape) == tuple(kv.shape)
    assert dq.dtype == q.dtype
    assert dkv.dtype == kv.dtype
    assert np.all(np.isfinite(np.asarray(dq.astype(mx.float32))))
    assert np.all(np.isfinite(np.asarray(dkv.astype(mx.float32))))


def test_fp8_path_c_materialized_owner_buffer_helper_returns_mlx_array() -> None:
    _requires_native_tilelang_graph_outputs()

    out = _materialized_owner_buffer((2, 3), mx.float32)
    assert isinstance(out, mx.array)
    assert tuple(out.shape) == (2, 3)
    assert out.dtype == mx.float32


def _assert_fp8_path_c_bwd_runs_with_owner_buffers(
    q_fp8: mx.array,
    q_scale: mx.array,
    kv_fp8: mx.array,
    kv_scale: mx.array,
    d_out: mx.array,
    indices: mx.array,
    *,
    sm_scale: float,
    d_v: int,
    dq_buffer: mx.array,
    dkv_buffer: mx.array,
    causal: bool = False,
) -> None:
    _requires_native_tilelang_graph_outputs()

    forced_result = sparse_mla_fp8_bwd_path_c(
        q_fp8,
        q_scale,
        kv_fp8,
        kv_scale,
        d_out,
        indices,
        sm_scale=sm_scale,
        d_v=d_v,
        force_path_c=True,
        causal=causal,
        dq_buffer=dq_buffer,
        dkv_buffer=dkv_buffer,
    )
    assert forced_result == (dq_buffer, dkv_buffer)
    result = sparse_mla_fp8_bwd_path_c(
        q_fp8,
        q_scale,
        kv_fp8,
        kv_scale,
        d_out,
        indices,
        sm_scale=sm_scale,
        d_v=d_v,
        causal=causal,
        dq_buffer=dq_buffer,
        dkv_buffer=dkv_buffer,
    )
    assert result == (dq_buffer, dkv_buffer)
    mx.eval(dq_buffer, dkv_buffer)
    assert np.all(np.isfinite(np.asarray(dq_buffer)))
    assert np.all(np.isfinite(np.asarray(dkv_buffer)))


def _fake_fp8_path_c_bwd_owner_output_route(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[tuple[object, ...], tuple[mx.array, mx.array]]]:
    import cppmega_mlx.nn._tilelang.sparse_mla_fp8_path_c as path_c_module

    calls: list[tuple[tuple[object, ...], tuple[mx.array, mx.array]]] = []

    def fake_tvm_ffi_kernel_for(*args: object, **kwargs: object):
        del args, kwargs

        def fake_kernel(*kernel_args: object, out: tuple[mx.array, mx.array]):
            calls.append((kernel_args, out))
            return out

        return fake_kernel

    monkeypatch.setattr(path_c_module, "can_run_metal", lambda: True)
    monkeypatch.setattr(
        path_c_module,
        "_fp8_bwd_tvm_ffi_kernel_for",
        fake_tvm_ffi_kernel_for,
    )
    monkeypatch.setattr(
        path_c_module,
        "_clear_fp8_bwd_dkv_buffer",
        lambda buffer: buffer,
    )
    monkeypatch.setattr(
        path_c_module.mx.fast,
        "metal_kernel",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("owner-output backward must not build mx.fast wrapper")
        ),
    )
    return calls


def test_fp8_apply_force_metal_raises_direct_msl_blocker() -> None:
    q, kv, indices, d_v = _make_inputs()
    with pytest.raises(RuntimeError, match="direct-MSL Path B is retired"):
        sparse_mla_fp8_apply(q, kv, indices, d_v=d_v, force_metal=True)


def test_retired_fp8_direct_msl_backward_has_no_partial_route() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "cppmega_mlx"
        / "nn"
        / "_tilelang"
        / "sparse_mla_fp8.py"
    ).read_text(encoding="utf-8")

    assert "_FP8_BWD_KERNEL_SOURCE" not in source
    assert "sparse_mla_fp8_bwd_metal_impl" not in source
    assert "_reduce_dkv_partial_fp32" not in source
    assert "dkv_partial" not in source


# ---------------------------------------------------------------------------
# FP8 helper round-trip
# ---------------------------------------------------------------------------


def test_to_fp8_per_tensor_scale_roundtrip_recovers_within_noise() -> None:
    rng = np.random.default_rng(42)
    x = mx.array(rng.standard_normal((4, 8, 16)).astype(np.float32))
    fp8, scale = _to_fp8_with_per_tensor_scale(x)
    rec = _from_fp8_with_scale(fp8, scale, dtype=mx.float32)
    mx.eval(rec)
    # FP8 e4m3 has 3 mantissa bits — expect ~3% rel error after recovery.
    err = (rec - x).abs().max().item()
    rel = err / (x.abs().max().item() + 1e-9)
    assert rel < 0.05, f"FP8 roundtrip rel err {rel:.4e} exceeded 5%"
    assert scale.shape == x.shape[:-1]
    assert scale.dtype == mx.float32


# ---------------------------------------------------------------------------
# Forward parity
# ---------------------------------------------------------------------------


def test_fp8_apply_matches_reference_within_fp8_tolerance() -> None:
    """``sparse_mla_fp8_apply`` falls back to the pure-MLX FP8 reference."""

    q, kv, indices, d_v = _make_inputs(scale=0.1)
    out_apply = _array_only(sparse_mla_fp8_apply(q, kv, indices, d_v=d_v))
    out_ref = _array_only(sparse_mla_fp8_reference(q, kv, indices, d_v=d_v))
    mx.eval(out_apply, out_ref)
    np.testing.assert_allclose(
        np.asarray(out_apply).astype(np.float32),
        np.asarray(out_ref).astype(np.float32),
        rtol=5e-3,
        atol=5e-3,
    )


def test_fp8_path_b_forward_is_not_dispatchable() -> None:
    """Retired direct-MSL FP8 forward reports unavailable instead of dispatching."""

    q, kv, indices, d_v = _make_inputs(scale=0.1)
    result = sparse_mla_fp8_fwd_metal(q, kv, indices, d_v=d_v)
    assert result is None


def test_fp8_path_c_forward_uses_tvm_ffi_owner_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cppmega_mlx.nn._tilelang.sparse_mla_fp8_path_c as path_c_module

    q, kv, indices, d_v = _make_inputs(scale=0.1)
    q_fp8 = mx.zeros(tuple(q.shape), dtype=mx.uint8)
    kv_fp8 = mx.zeros(tuple(kv.shape), dtype=mx.uint8)
    q_scale = mx.ones(tuple(q.shape[:-1]), dtype=mx.float32)
    kv_scale = mx.ones(tuple(kv.shape[:-1]), dtype=mx.float32)
    out = mx.zeros(tuple(q.shape[:3]) + (d_v,), dtype=mx.float16)
    lse = mx.zeros(tuple(q.shape[:3]), dtype=mx.float32)
    calls: list[tuple[tuple[object, ...], tuple[mx.array, mx.array]]] = []

    def fake_tvm_ffi_kernel_for(*args: object, **kwargs: object):
        del args, kwargs

        def fake_kernel(*kernel_args: object, out: tuple[mx.array, mx.array]):
            calls.append((kernel_args, out))
            return out

        return fake_kernel

    def fail_legacy_kernel(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("owner-output forward must not build mx.fast wrapper")

    monkeypatch.setattr(path_c_module, "can_run_metal", lambda: True)
    monkeypatch.setattr(path_c_module, "_fp8_apply_tvm_ffi_kernel_for", fake_tvm_ffi_kernel_for)
    monkeypatch.setattr(path_c_module, "_fp8_apply_kernel_for", fail_legacy_kernel)
    monkeypatch.setattr(path_c_module.mx.fast, "metal_kernel", fail_legacy_kernel)

    result = path_c_module.sparse_mla_fp8_path_c_apply(
        q_fp8,
        q_scale,
        kv_fp8,
        kv_scale,
        indices,
        sm_scale=q.shape[-1] ** -0.5,
        d_v=d_v,
        return_lse=True,
        force_path_c=True,
        out=out,
        lse=lse,
    )

    assert result == (out, lse)
    assert len(calls) == 1
    kernel_args, owner_outputs = calls[0]
    assert kernel_args[0].shape == (q_fp8.size,)
    assert kernel_args[1].shape == (q_scale.size,)
    assert kernel_args[2].shape == (kv_fp8.size,)
    assert kernel_args[3].shape == (kv_scale.size,)
    assert kernel_args[4].shape == (indices.size,)
    assert owner_outputs[0].shape == (out.size,)
    assert owner_outputs[1].shape == (lse.size,)


def test_fp8_path_c_forward_owner_output_abi_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cppmega_mlx.nn._tilelang.sparse_mla_fp8_path_c as path_c_module

    q, kv, indices, d_v = _make_inputs(scale=0.1)
    q_fp8 = mx.zeros(tuple(q.shape), dtype=mx.uint8)
    kv_fp8 = mx.zeros(tuple(kv.shape), dtype=mx.uint8)
    q_scale = mx.ones(tuple(q.shape[:-1]), dtype=mx.float32)
    kv_scale = mx.ones(tuple(kv.shape[:-1]), dtype=mx.float32)
    out = mx.zeros(tuple(q.shape[:3]) + (d_v,), dtype=mx.int32)
    lse = mx.zeros(tuple(q.shape[:3]), dtype=mx.float32)

    monkeypatch.setattr(path_c_module, "can_run_metal", lambda: True)
    with pytest.raises(ValueError, match="requires both out and lse"):
        path_c_module.sparse_mla_fp8_path_c_apply(
            q_fp8,
            q_scale,
            kv_fp8,
            kv_scale,
            indices,
            sm_scale=q.shape[-1] ** -0.5,
            d_v=d_v,
            out=out,
        )
    with pytest.raises(TypeError, match="output_dtype must be float32, float16, or bfloat16"):
        path_c_module.sparse_mla_fp8_path_c_apply(
            q_fp8,
            q_scale,
            kv_fp8,
            kv_scale,
            indices,
            sm_scale=q.shape[-1] ** -0.5,
            d_v=d_v,
            out=out,
            lse=lse,
        )


def test_fp8_path_b_backward_is_not_dispatchable() -> None:
    """Retired direct-MSL FP8 backward reports unavailable instead of dispatching."""

    q, kv, indices, d_v = _make_inputs(scale=0.1)

    rng = np.random.default_rng(31)
    d_out = mx.array(
        (rng.standard_normal(tuple(q.shape[:3]) + (d_v,)) * 0.1).astype(np.float32)
    )

    grads = sparse_mla_fp8_bwd_metal(q, kv, d_out, indices, d_v=d_v)
    assert grads is None


def test_fp8_path_c_backward_parity_route_uses_atomic_tvm_ffi() -> None:
    q, kv, indices, d_v = _make_inputs(scale=0.1)
    q_fp8 = mx.zeros(tuple(q.shape), dtype=mx.uint8)
    kv_fp8 = mx.zeros(tuple(kv.shape), dtype=mx.uint8)
    q_scale = mx.ones(tuple(q.shape[:-1]), dtype=mx.float32)
    kv_scale = mx.ones(tuple(kv.shape[:-1]), dtype=mx.float32)

    rng = np.random.default_rng(117)
    d_out = mx.array(
        (rng.standard_normal(tuple(q.shape[:3]) + (d_v,)) * 0.1).astype(np.float32)
    )
    dq_buffer = mx.zeros(q.shape, dtype=mx.float32)
    dkv_buffer = mx.zeros(kv.shape, dtype=mx.float32)

    _assert_fp8_path_c_bwd_runs_with_owner_buffers(
        q_fp8,
        q_scale,
        kv_fp8,
        kv_scale,
        d_out,
        indices,
        sm_scale=q.shape[-1] ** -0.5,
        d_v=d_v,
        dq_buffer=dq_buffer,
        dkv_buffer=dkv_buffer,
    )


def test_fp8_per_token_quant_producer_uses_single_tvm_ffi_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _requires_native_tilelang_graph_outputs()

    q, _kv, _indices, _d_v = _make_inputs(scale=0.1)

    def fail_to_fp8(*args, **kwargs):
        del args, kwargs
        raise AssertionError(
            "producer must not materialize scaled tensor via mx.to_fp8"
        )

    monkeypatch.setattr(mx, "to_fp8", fail_to_fp8)

    q_fp8, q_scale = _to_fp8_with_per_token_scale(q)
    mx.eval(q_fp8, q_scale)

    assert q_fp8.dtype == mx.uint8
    assert q_scale.dtype == mx.float32
    assert tuple(q_fp8.shape) == tuple(q.shape)
    assert tuple(q_scale.shape) == tuple(q.shape[:-1])
    assert np.all(np.isfinite(np.asarray(q_scale)))


def test_fp8_per_token_quant_producer_has_no_scaled_tensor_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cppmega_mlx.nn._tilelang.sparse_mla_fp8_path_c as path_c_module

    q, _kv, _indices, _d_v = _make_inputs(scale=0.1)

    def fail_to_fp8(*args, **kwargs):
        del args, kwargs
        raise AssertionError("producer fallback must not call mx.to_fp8")

    monkeypatch.setattr(mx, "to_fp8", fail_to_fp8)
    monkeypatch.setattr(
        path_c_module, "_to_fp8_with_per_token_scale_metal", lambda x: None
    )

    with pytest.raises(
        RuntimeError, match="must not materialize a full-size scaled tensor"
    ):
        _to_fp8_with_per_token_scale(q)


def test_fp8_path_c_backward_accepts_bf16_dout_without_host_cast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q, kv, indices, d_v = _make_inputs(scale=0.1)
    q_fp8 = mx.zeros(tuple(q.shape), dtype=mx.uint8)
    kv_fp8 = mx.zeros(tuple(kv.shape), dtype=mx.uint8)
    q_scale = mx.ones(tuple(q.shape[:-1]), dtype=mx.float32)
    kv_scale = mx.ones(tuple(kv.shape[:-1]), dtype=mx.float32)

    rng = np.random.default_rng(119)
    d_out = mx.array(
        (rng.standard_normal(tuple(q.shape[:3]) + (d_v,)) * 0.1).astype(np.float32)
    ).astype(mx.bfloat16)
    dq_buffer = mx.zeros(q.shape, dtype=mx.float32)
    dkv_buffer = mx.zeros(kv.shape, dtype=mx.float32)
    calls = _fake_fp8_path_c_bwd_owner_output_route(monkeypatch)

    grads = sparse_mla_fp8_bwd_path_c(
        q_fp8,
        q_scale,
        kv_fp8,
        kv_scale,
        d_out,
        indices,
        sm_scale=q.shape[-1] ** -0.5,
        d_v=d_v,
        force_path_c=True,
        dq_buffer=dq_buffer,
        dkv_buffer=dkv_buffer,
    )
    assert grads is not None
    dq_path_c, dkv_path_c = grads
    mx.eval(dq_path_c, dkv_path_c)

    assert calls[0][0][4].shape == (d_out.size,)
    assert calls[0][0][4].dtype == mx.bfloat16
    assert dq_path_c.dtype == mx.float32
    assert dkv_path_c.dtype == mx.float32
    assert np.all(np.isfinite(np.asarray(dq_path_c)))
    assert np.all(np.isfinite(np.asarray(dkv_path_c)))


def test_fp8_path_c_backward_uses_tvm_ffi_owner_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cppmega_mlx.nn._tilelang.sparse_mla_fp8_path_c as path_c_module

    q, kv, indices, d_v = _make_inputs(scale=0.1, topk=4)
    q_fp8 = mx.zeros(tuple(q.shape), dtype=mx.uint8)
    kv_fp8 = mx.zeros(tuple(kv.shape), dtype=mx.uint8)
    q_scale = mx.ones(tuple(q.shape[:-1]), dtype=mx.float32)
    kv_scale = mx.ones(tuple(kv.shape[:-1]), dtype=mx.float32)
    d_out = mx.zeros(tuple(q.shape[:3]) + (d_v,), dtype=mx.float32)
    dq_buffer = mx.zeros(q.shape, dtype=mx.float32)
    dkv_buffer = mx.zeros(kv.shape, dtype=mx.float32)
    calls: list[tuple[tuple[object, ...], tuple[mx.array, mx.array]]] = []
    clear_calls: list[mx.array] = []

    def fake_tvm_ffi_kernel_for(*args: object, **kwargs: object):
        del args, kwargs

        def fake_kernel(*kernel_args: object, out: tuple[mx.array, mx.array]):
            calls.append((kernel_args, out))
            return out

        return fake_kernel

    def fake_clear(buffer: mx.array) -> mx.array:
        clear_calls.append(buffer)
        return buffer

    def fail_mx_fast(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("owner-output backward must not build mx.fast wrapper")

    monkeypatch.setattr(path_c_module, "can_run_metal", lambda: True)
    monkeypatch.setattr(
        path_c_module,
        "_fp8_bwd_tvm_ffi_kernel_for",
        fake_tvm_ffi_kernel_for,
    )
    monkeypatch.setattr(path_c_module, "_clear_fp8_bwd_dkv_buffer", fake_clear)
    monkeypatch.setattr(path_c_module.mx.fast, "metal_kernel", fail_mx_fast)

    grads = sparse_mla_fp8_bwd_path_c(
        q_fp8,
        q_scale,
        kv_fp8,
        kv_scale,
        d_out,
        indices,
        sm_scale=q.shape[-1] ** -0.5,
        d_v=d_v,
        force_path_c=True,
        dq_buffer=dq_buffer,
        dkv_buffer=dkv_buffer,
    )

    assert grads == (dq_buffer, dkv_buffer)
    assert clear_calls == [dkv_buffer]
    assert len(calls) == 1
    kernel_args, owner_outputs = calls[0]
    assert kernel_args[0].shape == (q_fp8.size,)
    assert kernel_args[1].shape == (q_scale.size,)
    assert kernel_args[2].shape == (kv_fp8.size,)
    assert kernel_args[3].shape == (kv_scale.size,)
    assert kernel_args[4].shape == (d_out.size,)
    assert kernel_args[5].shape == (indices.size,)
    assert owner_outputs[0].shape == (dq_buffer.size,)
    assert owner_outputs[1].shape == (dkv_buffer.size,)


def test_fp8_path_c_backward_clears_dkv_before_atomic_route() -> None:
    q, kv, indices, d_v = _make_inputs(scale=0.1, topk=4)
    q_fp8 = mx.zeros(tuple(q.shape), dtype=mx.uint8)
    kv_fp8 = mx.zeros(tuple(kv.shape), dtype=mx.uint8)
    q_scale = mx.ones(tuple(q.shape[:-1]), dtype=mx.float32)
    kv_scale = mx.ones(tuple(kv.shape[:-1]), dtype=mx.float32)
    d_out = mx.zeros(tuple(q.shape[:3]) + (d_v,), dtype=mx.float32)

    dq_buffer = mx.ones(q.shape, dtype=mx.float32)
    dkv_buffer = mx.ones(kv.shape, dtype=mx.float32)
    _assert_fp8_path_c_bwd_runs_with_owner_buffers(
        q_fp8,
        q_scale,
        kv_fp8,
        kv_scale,
        d_out,
        indices,
        sm_scale=q.shape[-1] ** -0.5,
        d_v=d_v,
        dq_buffer=dq_buffer,
        dkv_buffer=dkv_buffer,
    )
    mx.eval(dq_buffer, dkv_buffer)
    np.testing.assert_allclose(
        np.asarray(dq_buffer).astype(np.float32),
        np.zeros(tuple(q.shape), dtype=np.float32),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(dkv_buffer).astype(np.float32),
        np.zeros(tuple(kv.shape), dtype=np.float32),
        atol=1e-7,
    )


def test_fp8_path_c_causal_backward_skips_partial_reduce() -> None:
    q, kv, _indices, d_v = _make_inputs(scale=0.1, topk=4)
    indices = causal_sparse_indices(
        q.shape[0],
        q.shape[1],
        kv.shape[2],
        4,
    )
    q_fp8 = mx.zeros(tuple(q.shape), dtype=mx.uint8)
    kv_fp8 = mx.zeros(tuple(kv.shape), dtype=mx.uint8)
    q_scale = mx.ones(tuple(q.shape[:-1]), dtype=mx.float32)
    kv_scale = mx.ones(tuple(kv.shape[:-1]), dtype=mx.float32)

    rng = np.random.default_rng(121)
    d_out = mx.array(
        (rng.standard_normal(tuple(q.shape[:3]) + (d_v,)) * 0.1).astype(np.float32)
    )

    dq_buffer = mx.zeros(q.shape, dtype=mx.float32)
    dkv_buffer = mx.zeros(kv.shape, dtype=mx.float32)
    _assert_fp8_path_c_bwd_runs_with_owner_buffers(
        q_fp8,
        q_scale,
        kv_fp8,
        kv_scale,
        d_out,
        indices,
        sm_scale=q.shape[-1] ** -0.5,
        d_v=d_v,
        causal=True,
        dq_buffer=dq_buffer,
        dkv_buffer=dkv_buffer,
    )


def test_fp8_path_c_explicit_mask_backward_skips_partial_reduce() -> None:
    q, kv, _indices, d_v = _make_inputs(scale=0.1, topk=2)
    explicit_mask = mx.array(
        [
            [
                [True, False, False, False],
                [True, True, False, False],
                [False, False, True, False],
                [False, False, True, True],
            ]
        ],
        dtype=mx.bool_,
    )
    indices = sparse_indices_from_attention_mask(
        explicit_mask,
        batch_size=q.shape[0],
        seq_length=q.shape[1],
        kv_group=kv.shape[2],
        topk=2,
        key_length=kv.shape[1],
    )
    q_fp8 = mx.zeros(tuple(q.shape), dtype=mx.uint8)
    kv_fp8 = mx.zeros(tuple(kv.shape), dtype=mx.uint8)
    q_scale = mx.ones(tuple(q.shape[:-1]), dtype=mx.float32)
    kv_scale = mx.ones(tuple(kv.shape[:-1]), dtype=mx.float32)

    rng = np.random.default_rng(122)
    d_out = mx.array(
        (rng.standard_normal(tuple(q.shape[:3]) + (d_v,)) * 0.1).astype(np.float32)
    )

    dq_buffer = mx.zeros(q.shape, dtype=mx.float32)
    dkv_buffer = mx.zeros(kv.shape, dtype=mx.float32)
    _assert_fp8_path_c_bwd_runs_with_owner_buffers(
        q_fp8,
        q_scale,
        kv_fp8,
        kv_scale,
        d_out,
        indices,
        sm_scale=q.shape[-1] ** -0.5,
        d_v=d_v,
        causal=True,
        dq_buffer=dq_buffer,
        dkv_buffer=dkv_buffer,
    )


def test_fp8_reference_matches_bf16_within_fp8_tolerance() -> None:
    """FP8 reference vs BF16 reference, with small-magnitude inputs.

    With std=0.1 inputs the FP8 e4m3 mantissa noise is small enough to clear
    the rtol=5e-3 tolerance from the task brief.
    """

    q, kv, indices, d_v = _make_inputs(scale=0.1)
    out_fp8 = _array_only(sparse_mla_fp8_reference(q, kv, indices, d_v=d_v))
    out_bf = _array_only(sparse_mla_attention_reference(q, kv, indices, d_v=d_v))
    mx.eval(out_fp8, out_bf)
    out_fp8_np = np.asarray(out_fp8.astype(mx.float32))
    out_bf_np = np.asarray(out_bf.astype(mx.float32))
    np.testing.assert_allclose(out_fp8_np, out_bf_np, rtol=5e-3, atol=5e-3)


def test_fp8_reference_with_lse_returns_pair() -> None:
    q, kv, indices, d_v = _make_inputs()
    result = sparse_mla_fp8_reference(q, kv, indices, d_v=d_v, return_lse=True)
    assert isinstance(result, tuple) and len(result) == 2
    out, lse = result
    mx.eval(out, lse)
    assert out.shape[-1] == d_v
    assert lse.shape == out.shape[:-1]


def test_quantized_matmul_reference_matches_bf16_within_tight_tolerance() -> None:
    """The mxfp8 hand-built path uses regular matmul on dequantized tensors,
    so it should agree with the BF16 reference at fp32 precision."""

    q, kv, indices, d_v = _make_inputs()
    out_qm = _array_only(sparse_mla_quantized_matmul_reference(q, kv, indices, d_v=d_v))
    out_bf = _array_only(sparse_mla_attention_reference(q, kv, indices, d_v=d_v))
    mx.eval(out_qm, out_bf)
    out_qm_np = np.asarray(out_qm.astype(mx.float32))
    out_bf_np = np.asarray(out_bf.astype(mx.float32))
    np.testing.assert_allclose(out_qm_np, out_bf_np, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Backward parity
# ---------------------------------------------------------------------------


def test_fp8_reference_backward_matches_bf16_over_recovered_inputs() -> None:
    """Gradient parity within FP8 noise tolerance.

    We compare the FP8 reference's gradients against a BF16 reference taken
    over the *same* dequantized Q/KV. The two paths should match to within
    FP8 noise (rtol=1e-2 per the task brief).
    """

    q, kv, indices, d_v = _make_inputs(scale=0.1)
    indices_bound = indices

    # Build the dequantized Q/KV the FP8 reference produces internally so we
    # can run the BF16 reference on the same recovered tensors as a parity
    # oracle.
    fp8_q, q_scale = _to_fp8_with_per_tensor_scale(q)
    fp8_kv, kv_scale = _to_fp8_with_per_tensor_scale(kv)
    q_rec = _from_fp8_with_scale(fp8_q, q_scale, dtype=q.dtype)
    kv_rec = _from_fp8_with_scale(fp8_kv, kv_scale, dtype=kv.dtype)
    mx.eval(q_rec, kv_rec)

    def fp8_loss(q_in, kv_in):
        out = _array_only(sparse_mla_fp8_reference(q_in, kv_in, indices_bound, d_v=d_v))
        return mx.sum(out * out)

    def bf16_loss_on_recovered(q_in, kv_in):
        out = _array_only(
            sparse_mla_attention_reference(q_in, kv_in, indices_bound, d_v=d_v)
        )
        return mx.sum(out * out)

    fp8_grad = mx.grad(fp8_loss, argnums=(0, 1))(q, kv)
    bf_grad = mx.grad(bf16_loss_on_recovered, argnums=(0, 1))(q_rec, kv_rec)
    mx.eval(*fp8_grad, *bf_grad)

    # The FP8 grads should be finite and match the BF16 dequant-oracle grads
    # within FP8 noise tolerance.
    for g in fp8_grad:
        g_np = np.asarray(g)
        assert np.all(np.isfinite(g_np)), "FP8 grads must be finite"

    fp8_dq, fp8_dkv = (np.asarray(g) for g in fp8_grad)
    bf_dq, bf_dkv = (np.asarray(g) for g in bf_grad)
    np.testing.assert_allclose(fp8_dq, bf_dq, rtol=1e-2, atol=5e-3)
    np.testing.assert_allclose(fp8_dkv, bf_dkv, rtol=1e-2, atol=5e-3)


def test_fp8_reference_backward_through_apply_finite() -> None:
    """``sparse_mla_fp8_apply`` is the production entry. Backward should flow
    cleanly without NaN/Inf even when half the topk slots are masked."""

    q, kv, indices, d_v = _make_inputs(scale=0.1)

    def loss(q_in, kv_in):
        out = _array_only(sparse_mla_fp8_apply(q_in, kv_in, indices, d_v=d_v))
        return mx.sum(out * out)

    grads = mx.grad(loss, argnums=(0, 1))(q, kv)
    mx.eval(*grads)
    for g in grads:
        g_np = np.asarray(g)
        assert np.all(np.isfinite(g_np)), "apply backward grads must be finite"


def test_fp8_path_c_float_wrapper_refuses_hidden_quantization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cppmega_mlx.nn._tilelang.sparse_mla_fp8_path_c as path_c_module

    q, kv, indices, d_v = _make_inputs(scale=0.1)

    def fail_quantize(x: mx.array) -> tuple[mx.array, mx.array]:
        raise AssertionError(f"unexpected wrapper quantization for {tuple(x.shape)}")

    monkeypatch.setattr(path_c_module, "_to_fp8_with_per_token_scale", fail_quantize)

    with pytest.raises(RuntimeError, match="requires prepared FP8 buffers"):
        sparse_mla_fp8_path_c_apply_from_float(
            q,
            kv,
            indices,
            sm_scale=q.shape[-1] ** -0.5,
            d_v=d_v,
            force_path_c=True,
        )


def test_fp8_path_c_prepared_float_wrapper_does_not_quantize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cppmega_mlx.nn._tilelang.sparse_mla_fp8_path_c as path_c_module

    q, kv, indices, d_v = _make_inputs(scale=0.1)
    q = q.astype(mx.bfloat16)
    kv = kv.astype(mx.bfloat16)
    q_fp8 = mx.zeros(tuple(q.shape), dtype=mx.uint8)
    kv_fp8 = mx.zeros(tuple(kv.shape), dtype=mx.uint8)
    q_scale = mx.ones(tuple(q.shape[:-1]), dtype=mx.float32)
    kv_scale = mx.ones(tuple(kv.shape[:-1]), dtype=mx.float32)
    calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def fail_quantize(x: mx.array) -> tuple[mx.array, mx.array]:
        raise AssertionError(f"unexpected wrapper quantization for {tuple(x.shape)}")

    def fake_apply(
        q_fp8_in: mx.array,
        q_scale_in: mx.array,
        kv_fp8_in: mx.array,
        kv_scale_in: mx.array,
        indices_in: mx.array,
        *,
        sm_scale: float,
        d_v: int | None = None,
        sinks: mx.array | None = None,
        return_lse: bool = False,
        force_path_c: bool = False,
        output_dtype: mx.Dtype | None = None,
    ) -> mx.array:
        del q_scale_in, kv_scale_in, indices_in, sm_scale, sinks, return_lse
        assert force_path_c is True
        assert output_dtype == q.dtype
        calls.append((tuple(q_fp8_in.shape), tuple(kv_fp8_in.shape)))
        return mx.zeros(
            (
                q_fp8_in.shape[0],
                q_fp8_in.shape[1],
                q_fp8_in.shape[2],
                d_v or q_fp8_in.shape[-1],
            ),
            dtype=q.dtype,
        )

    monkeypatch.setattr(path_c_module, "_to_fp8_with_per_token_scale", fail_quantize)
    monkeypatch.setattr(path_c_module, "sparse_mla_fp8_path_c_apply", fake_apply)

    out = sparse_mla_fp8_path_c_apply_prepared_float(
        q,
        kv,
        q_fp8,
        q_scale,
        kv_fp8,
        kv_scale,
        indices,
        sm_scale=q.shape[-1] ** -0.5,
        d_v=d_v,
        force_path_c=True,
    )
    mx.eval(out)

    assert calls == [(tuple(q.shape), tuple(kv.shape))]
    assert tuple(out.shape) == tuple(q.shape[:3]) + (d_v,)


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------


def test_module_public_exports_present() -> None:
    from cppmega_mlx.nn._tilelang import sparse_mla_fp8 as module

    expected = {
        "SparseMLAFp8MetalStatus",
        "sparse_mla_fp8_apply",
        "sparse_mla_fp8_bwd_metal",
        "sparse_mla_fp8_fwd_metal",
        "sparse_mla_fp8_metal_status",
        "sparse_mla_fp8_reference",
        "sparse_mla_quantized_matmul_reference",
    }
    assert expected.issubset(set(module.__all__))

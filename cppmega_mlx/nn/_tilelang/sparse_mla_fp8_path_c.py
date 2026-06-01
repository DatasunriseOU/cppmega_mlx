# pyright: reportInvalidTypeForm=false, reportMissingImports=false
"""Path C FP8 Sparse-MLA TileLang DSL surfaces.

This module owns the FP8 Sparse-MLA Path C kernels over the *prepared* ABI:
``q_fp8/q_scale/kv_fp8/kv_scale/indices`` are existing GPU buffers already
created by the upstream graph. The public ``sparse_mla_fp8_path_c_apply`` does
not quantize, gather, cast, or allocate staging tensors; if callers have
float carriers they should route through the higher-level graph planner so the
producer emits FP8 buffers directly, or fall back to the explicit MLX
reference.

This module owns the prepared-buffer FP8 Sparse-MLA forward and backward
TileLang surfaces.  The historical direct-MSL Path B wrapper in
``sparse_mla_fp8.py`` is retired and now reports explicit unsupported status.
Path C also exposes two lower-level QK status/probe surfaces:

* ``T.fp8_scaled_matmul`` probe/status glue. Current apple-head TileLang can
  lower square 32x32 FP8 matmul to the Metal simdgroup path with explicit scale
  loads, but the literal Sparse-MLA QK shape (M=1 query row against top-k
  transposed KV rows) is still fail-closed because it scalarizes or drops scale
  operands.
* A real-shape ``@T.prim_func`` reducer for ``A_fp8(1, K) @ B_fp8(N, K).T``.
  It lowers through TileLang and dispatches via tvm-ffi, preserves scalar A
  scale plus scalar/per-row B scale semantics, and is benchmarked against the
  reference as the current runnable FP8 Path C QK tile.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from collections.abc import Callable, Mapping
import os
from typing import Any, cast

import mlx.core as mx

from cppmega_mlx.nn._tilelang import _msl_transform
from cppmega_mlx.nn._tilelang._async_barrier_plan import (
    MetalReductionSyncPlan,
    plan_metal_path_c_reduction_sync,
)
from cppmega_mlx.nn._tilelang._engine_dispatch import dispatch_lower
from cppmega_mlx.nn._tilelang._msl_transform import (
    MSLDispatchUnsupported,
    can_run_metal,
)


TILELANG_METAL_FP8_SPARSE_MLA_TARGET = "metal"
SPARSE_MLA_FP8_BWD_ENV = "CPPMEGA_SPARSE_MLA_FP8_BWD"

# TileLang resolves these globals while decorating nested @T.prim_func kernels.
# Defaults keep pyright aligned with the runtime-specialized values.
_SMFP8_M = 1
_SMFP8_N = 16
_SMFP8_K = 64
_SMFP8_BM = 1
_SMFP8_BN = 16
_SMFP8_BK = 64
_SMFP8_SA = 1
_SMFP8_SB = 16
_SMFP8_B_SHAPE = (16, 64)
_SMFP8_B_SHARED_SHAPE = (16, 64)
_SMFP8_TRANSPOSE_B = True
_SMFP8_NUM_STAGES = 0

_SMFP8_QKR_N = 16
_SMFP8_QKR_K = 64
_SMFP8_QKR_NP = 8
_SMFP8_QKR_RT = 16
_SMFP8_QKR_VEC = 4
_SMFP8_QKR_BLOCK_K = _SMFP8_QKR_RT * _SMFP8_QKR_VEC
_SMFP8_QKR_K_WORDS = _SMFP8_QKR_K // 4
_SMFP8_QKR_BS = 16

_SMFP8_QKR_DEFAULT_OUTPUTS_PER_BLOCK = 2
_SMFP8_QKR_DEFAULT_REDUCE_THREADS = 8
_SMFP8_QKR_DEFAULT_VEC = 4
_SMFP8_QKR_TUNED_OUTPUTS_PER_BLOCK = 16
_SMFP8_QKR_TUNED_REDUCE_THREADS = 32
_SMFP8_QKR_TUNED_VEC = 4
_SMFP8_INVALID_SCORE_SENTINEL = -3.4028234663852886e38

_SMFP8_IQKR_B = 1
_SMFP8_IQKR_S = 1
_SMFP8_IQKR_H = 1
_SMFP8_IQKR_SKV = 16
_SMFP8_IQKR_G = 1
_SMFP8_IQKR_HEAD_KV = 1
_SMFP8_IQKR_TOPK = 16
_SMFP8_IQKR_K = 64
_SMFP8_IQKR_LANES = _SMFP8_IQKR_B * _SMFP8_IQKR_S * _SMFP8_IQKR_H
_SMFP8_IQKR_Q_SIZE = _SMFP8_IQKR_LANES * _SMFP8_IQKR_K
_SMFP8_IQKR_KV_SIZE = _SMFP8_IQKR_B * _SMFP8_IQKR_SKV * _SMFP8_IQKR_G * _SMFP8_IQKR_K
_SMFP8_IQKR_Q_SCALE_SIZE = _SMFP8_IQKR_B * _SMFP8_IQKR_S * _SMFP8_IQKR_H
_SMFP8_IQKR_KV_SCALE_SIZE = _SMFP8_IQKR_B * _SMFP8_IQKR_SKV * _SMFP8_IQKR_G
_SMFP8_IQKR_IDX_SIZE = _SMFP8_IQKR_B * _SMFP8_IQKR_S * _SMFP8_IQKR_G * _SMFP8_IQKR_TOPK
_SMFP8_IQKR_OUT_SIZE = _SMFP8_IQKR_LANES * _SMFP8_IQKR_TOPK
_SMFP8_IQKR_NP = _SMFP8_QKR_DEFAULT_OUTPUTS_PER_BLOCK
_SMFP8_IQKR_RT = _SMFP8_QKR_DEFAULT_REDUCE_THREADS
_SMFP8_IQKR_VEC = _SMFP8_QKR_DEFAULT_VEC
_SMFP8_IQKR_BLOCK_K = _SMFP8_IQKR_RT * _SMFP8_IQKR_VEC
_SMFP8_IQKR_K_WORDS = _SMFP8_IQKR_K // 4
_SMFP8_IQKR_INDEX_DTYPE = "int32"

_SMFP8_APPLY_B = 1
_SMFP8_APPLY_S = 1
_SMFP8_APPLY_H = 1
_SMFP8_APPLY_SKV = 16
_SMFP8_APPLY_G = 1
_SMFP8_APPLY_HEAD_KV = 1
_SMFP8_APPLY_TOPK = 16
_SMFP8_APPLY_K = 64
_SMFP8_APPLY_DV = 64
_SMFP8_APPLY_THREADS = 16
_SMFP8_APPLY_LOG_THREADS = 4
_SMFP8_APPLY_LANES = _SMFP8_APPLY_B * _SMFP8_APPLY_S * _SMFP8_APPLY_H
_SMFP8_APPLY_Q_SIZE = _SMFP8_APPLY_LANES * _SMFP8_APPLY_K
_SMFP8_APPLY_KV_SIZE = (
    _SMFP8_APPLY_B * _SMFP8_APPLY_SKV * _SMFP8_APPLY_G * _SMFP8_APPLY_K
)
_SMFP8_APPLY_Q_SCALE_SIZE = _SMFP8_APPLY_LANES
_SMFP8_APPLY_KV_SCALE_SIZE = _SMFP8_APPLY_B * _SMFP8_APPLY_SKV * _SMFP8_APPLY_G
_SMFP8_APPLY_IDX_SIZE = (
    _SMFP8_APPLY_B * _SMFP8_APPLY_S * _SMFP8_APPLY_G * _SMFP8_APPLY_TOPK
)
_SMFP8_APPLY_OUT_SIZE = _SMFP8_APPLY_LANES * _SMFP8_APPLY_DV
_SMFP8_APPLY_LSE_SIZE = _SMFP8_APPLY_LANES
_SMFP8_APPLY_OUT_DTYPE = "float32"
_SMFP8_APPLY_INDEX_DTYPE = "int32"

_SMFP8_BWD_B = 1
_SMFP8_BWD_S = 1
_SMFP8_BWD_H = 1
_SMFP8_BWD_SKV = 16
_SMFP8_BWD_G = 1
_SMFP8_BWD_HEAD_KV = 1
_SMFP8_BWD_TOPK = 16
_SMFP8_BWD_K = 64
_SMFP8_BWD_DV = 64
_SMFP8_BWD_THREADS = 16
_SMFP8_BWD_LOG_THREADS = 4
_SMFP8_BWD_LANES = _SMFP8_BWD_B * _SMFP8_BWD_S * _SMFP8_BWD_H
_SMFP8_BWD_Q_SIZE = _SMFP8_BWD_LANES * _SMFP8_BWD_K
_SMFP8_BWD_KV_SIZE = _SMFP8_BWD_B * _SMFP8_BWD_SKV * _SMFP8_BWD_G * _SMFP8_BWD_K
_SMFP8_BWD_Q_SCALE_SIZE = _SMFP8_BWD_LANES
_SMFP8_BWD_KV_SCALE_SIZE = _SMFP8_BWD_B * _SMFP8_BWD_SKV * _SMFP8_BWD_G
_SMFP8_BWD_IDX_SIZE = _SMFP8_BWD_B * _SMFP8_BWD_S * _SMFP8_BWD_G * _SMFP8_BWD_TOPK
_SMFP8_BWD_DOUT_SIZE = _SMFP8_BWD_LANES * _SMFP8_BWD_DV
_SMFP8_BWD_DOUT_DTYPE = "float32"
_SMFP8_BWD_INDEX_DTYPE = "int32"
_SMFP8_BWD_CLEAR_BATCH = _SMFP8_BWD_B
_SMFP8_BWD_CLEAR_SEQ_LEN_KV = _SMFP8_BWD_SKV
_SMFP8_BWD_CLEAR_KV_GROUP = _SMFP8_BWD_G
_SMFP8_BWD_CLEAR_K = _SMFP8_BWD_K
_SMFP8_BWD_CLEAR_TOTAL = _SMFP8_BWD_B * _SMFP8_BWD_SKV * _SMFP8_BWD_G * _SMFP8_BWD_K
_SMFP8_BWD_CLEAR_THREADS = 256
_SMFP8_PER_TOKEN_QUANT_THREADS = 256
_SMFP8_PTQ_ROWS = 1
_SMFP8_PTQ_K = 64
_SMFP8_PTQ_INPUT_DTYPE = "float32"
_SMFP8_PREPARE_Q_ROWS = 1
_SMFP8_PREPARE_KV_ROWS = 1
_SMFP8_PREPARE_ROWS = _SMFP8_PREPARE_Q_ROWS + _SMFP8_PREPARE_KV_ROWS
_SMFP8_PREPARE_K = 64
_SMFP8_PREPARE_INPUT_DTYPE = "float32"
_SMFP8_PREPARE_STORAGE_DTYPE = "uint8"


@dataclass(frozen=True)
class SparseMLAFp8PathCStatus:
    """Lowering status for the Path C TileLang FP8 Sparse-MLA QK tile."""

    available: bool
    reason: str
    features: dict[str, int | bool | str]
    target: str = TILELANG_METAL_FP8_SPARSE_MLA_TARGET
    m: int = 1
    n: int = 16
    k: int = 64
    transpose_B: bool = True


@dataclass(frozen=True)
class SparseMLAFp8QKReducePathCStatus:
    """Runtime/lowering status for the real-shape Path C FP8 QK reducer."""

    available: bool
    reason: str
    features: dict[str, int | bool | str]
    target: str = TILELANG_METAL_FP8_SPARSE_MLA_TARGET
    n: int = 16
    k: int = 64
    outputs_per_block: int = _SMFP8_QKR_DEFAULT_OUTPUTS_PER_BLOCK
    reduce_threads: int = _SMFP8_QKR_DEFAULT_REDUCE_THREADS
    vec: int = _SMFP8_QKR_DEFAULT_VEC


@dataclass(frozen=True)
class SparseMLAFp8IndexedQKReducePathCStatus:
    """Runtime/lowering status for indexed full-shape Path C FP8 QK scores."""

    available: bool
    reason: str
    features: dict[str, int | bool | str]
    target: str = TILELANG_METAL_FP8_SPARSE_MLA_TARGET
    batch: int = 1
    seq_len: int = 1
    heads: int = 1
    seq_len_kv: int = 16
    kv_group: int = 1
    head_kv: int = 1
    topk: int = 16
    k: int = 64
    outputs_per_block: int = _SMFP8_QKR_DEFAULT_OUTPUTS_PER_BLOCK
    reduce_threads: int = _SMFP8_QKR_DEFAULT_REDUCE_THREADS
    vec: int = _SMFP8_QKR_DEFAULT_VEC


class SparseMLAFp8PathCDirectError(RuntimeError):
    """Raised when a prepared-buffer tvm-ffi owner-output path cannot run."""


def _emit_fp8_apply_runtime_buffer(
    probe: Callable[[Mapping[str, Any]], None] | None,
    *,
    name: str,
    tensor: mx.array,
) -> None:
    if probe is None:
        return
    probe(
        {
            "name": name,
            "tensor": tensor,
            "producer_owner": "sparse_mla_fp8_path_c_apply",
            "producer_stage": "sparse_mla_fp8_apply",
        }
    )


def _index_dtype_name(indices: mx.array, *, op_name: str) -> str:
    if indices.dtype == mx.int32:
        return "int32"
    mx_int64 = getattr(mx, "int64", None)
    if mx_int64 is not None and indices.dtype == mx_int64:
        return "int64"
    raise TypeError(
        f"{op_name} requires mx.int32 or mx.int64 indices; got {indices.dtype}. "
        "Path C will not cast or copy sparse index tensors at the wrapper boundary."
    )


def _tilelang_available() -> tuple[bool, str]:
    try:
        import tilelang  # noqa: F401
        from tilelang import tvm as _tvm  # noqa: F401
        import tilelang.language as _T  # noqa: F401
    except Exception as exc:  # pragma: no cover - hosts without TileLang
        return False, f"tilelang import failed: {exc}"
    return True, "tilelang importable"


def _validate_shape(
    *,
    M: int,
    N: int,
    K: int,
    BM: int,
    BN: int,
    BK: int,
    a_scale_size: int,
    b_scale_size: int,
) -> None:
    values = {
        "M": M,
        "N": N,
        "K": K,
        "BM": BM,
        "BN": BN,
        "BK": BK,
        "a_scale_size": a_scale_size,
        "b_scale_size": b_scale_size,
    }
    bad = {name: value for name, value in values.items() if value <= 0}
    if bad:
        raise ValueError(f"FP8 Sparse-MLA Path C shape values must be positive: {bad}")


def _validate_reduce_shape(
    *,
    N: int,
    K: int,
    outputs_per_block: int,
    reduce_threads: int,
    vec: int,
) -> None:
    values = {
        "N": N,
        "K": K,
        "outputs_per_block": outputs_per_block,
        "reduce_threads": reduce_threads,
        "vec": vec,
    }
    bad = {name: value for name, value in values.items() if value <= 0}
    if bad:
        raise ValueError(
            f"FP8 Sparse-MLA Path C reducer shape values must be positive: {bad}"
        )


def _resolve_qk_reduce_schedule(
    *,
    N: int,
    K: int,
    outputs_per_block: int,
    reduce_threads: int,
    vec: int,
) -> tuple[int, int, int]:
    """Route the legacy bench/default Sparse-MLA tile to the profiled fast schedule."""

    if (
        N == 16
        and K == 64
        and outputs_per_block == _SMFP8_QKR_DEFAULT_OUTPUTS_PER_BLOCK
        and reduce_threads == _SMFP8_QKR_DEFAULT_REDUCE_THREADS
        and vec == _SMFP8_QKR_DEFAULT_VEC
    ):
        return (
            _SMFP8_QKR_TUNED_OUTPUTS_PER_BLOCK,
            _SMFP8_QKR_TUNED_REDUCE_THREADS,
            _SMFP8_QKR_TUNED_VEC,
        )
    return outputs_per_block, reduce_threads, vec


def fp8_sparse_mla_qk_reduce_sync_plan(
    *,
    N: int = 16,
    K: int = 64,
    outputs_per_block: int = _SMFP8_QKR_DEFAULT_OUTPUTS_PER_BLOCK,
    reduce_threads: int = _SMFP8_QKR_DEFAULT_REDUCE_THREADS,
    vec: int = _SMFP8_QKR_DEFAULT_VEC,
) -> MetalReductionSyncPlan:
    """Return the planned sync strategy for the FP8 Path C QK reducer."""

    outputs_per_block, reduce_threads, vec = _resolve_qk_reduce_schedule(
        N=N,
        K=K,
        outputs_per_block=outputs_per_block,
        reduce_threads=reduce_threads,
        vec=vec,
    )
    return plan_metal_path_c_reduction_sync(
        outputs_per_block=outputs_per_block,
        reduce_threads=reduce_threads,
        vec=vec,
        k_extent=K,
    )


def _validate_indexed_reduce_shape(
    *,
    batch: int,
    seq_len: int,
    heads: int,
    seq_len_kv: int,
    kv_group: int,
    topk: int,
    K: int,
    outputs_per_block: int,
    reduce_threads: int,
    vec: int,
) -> int:
    values = {
        "batch": batch,
        "seq_len": seq_len,
        "heads": heads,
        "seq_len_kv": seq_len_kv,
        "kv_group": kv_group,
        "topk": topk,
        "K": K,
        "outputs_per_block": outputs_per_block,
        "reduce_threads": reduce_threads,
        "vec": vec,
    }
    bad = {name: value for name, value in values.items() if value <= 0}
    if bad:
        raise ValueError(
            f"FP8 Sparse-MLA Path C indexed reducer values must be positive: {bad}"
        )
    if heads % kv_group != 0:
        raise ValueError(
            f"heads must be divisible by kv_group for Sparse-MLA grouping: {heads=} {kv_group=}"
        )
    return heads // kv_group


def make_fp8_sparse_mla_qk_kernel(
    *,
    M: int = 1,
    N: int = 16,
    K: int = 64,
    BM: int = 1,
    BN: int = 16,
    BK: int = 64,
    a_scale_size: int = 1,
    b_scale_size: int = 16,
    transpose_B: bool = True,
    num_stages: int = 0,
) -> Any:
    """Build the QK tile used by FP8 Sparse-MLA.

    ``M`` is the number of query rows, ``N`` is the sparse top-k tile, and
    ``B`` is transposed as ``(N, K)`` to match Path B/audiohacking vecmat scale
    semantics: A scale is per query row or scalar, B scale is per gathered KV row
    or scalar.
    """

    _validate_shape(
        M=M,
        N=N,
        K=K,
        BM=BM,
        BN=BN,
        BK=BK,
        a_scale_size=a_scale_size,
        b_scale_size=b_scale_size,
    )

    import tilelang.language as T

    T = cast(Any, T)

    b_shape = (N, K) if transpose_B else (K, N)
    shared_b_shape = (BN, BK) if transpose_B else (BK, BN)

    g = globals()
    g.update(
        _SMFP8_M=M,
        _SMFP8_N=N,
        _SMFP8_K=K,
        _SMFP8_BM=BM,
        _SMFP8_BN=BN,
        _SMFP8_BK=BK,
        _SMFP8_SA=a_scale_size,
        _SMFP8_SB=b_scale_size,
        _SMFP8_B_SHAPE=b_shape,
        _SMFP8_B_SHARED_SHAPE=shared_b_shape,
        _SMFP8_TRANSPOSE_B=transpose_B,
        _SMFP8_NUM_STAGES=num_stages,
    )

    @T.prim_func
    def fp8_sparse_mla_qk_kernel(
        A_fp8: T.Tensor((_SMFP8_M, _SMFP8_K), "float8_e4m3"),
        A_scale: T.Tensor((_SMFP8_SA,), "float32"),
        B_fp8: T.Tensor(_SMFP8_B_SHAPE, "float8_e4m3"),
        B_scale: T.Tensor((_SMFP8_SB,), "float32"),
        C: T.Tensor((_SMFP8_M, _SMFP8_N), "float32"),
    ):
        with T.Kernel(
            T.ceildiv(_SMFP8_N, _SMFP8_BN),
            T.ceildiv(_SMFP8_M, _SMFP8_BM),
            threads=128,
        ) as (bx, by):
            A_shared = T.alloc_shared(
                (_SMFP8_BM, _SMFP8_BK), "float8_e4m3", scope="shared"
            )
            B_shared = T.alloc_shared(
                _SMFP8_B_SHARED_SHAPE, "float8_e4m3", scope="shared"
            )
            C_local = T.alloc_fragment((_SMFP8_BM, _SMFP8_BN), "float32")
            T.clear(C_local)
            for ko in T.Pipelined(
                T.ceildiv(_SMFP8_K, _SMFP8_BK), num_stages=_SMFP8_NUM_STAGES
            ):
                T.copy(A_fp8[by * _SMFP8_BM, ko * _SMFP8_BK], A_shared)
                if _SMFP8_TRANSPOSE_B:
                    T.copy(B_fp8[bx * _SMFP8_BN, ko * _SMFP8_BK], B_shared)
                else:
                    T.copy(B_fp8[ko * _SMFP8_BK, bx * _SMFP8_BN], B_shared)
                T.fp8_scaled_matmul(
                    A_shared,
                    A_scale,
                    B_shared,
                    B_scale,
                    C_local,
                    transpose_B=_SMFP8_TRANSPOSE_B,
                )
            T.copy(C_local, C[by * _SMFP8_BM, bx * _SMFP8_BN])

    return fp8_sparse_mla_qk_kernel


def lower_fp8_sparse_mla_qk_msl(
    *,
    M: int = 1,
    N: int = 16,
    K: int = 64,
    BM: int = 1,
    BN: int = 16,
    BK: int = 64,
    a_scale_size: int = 1,
    b_scale_size: int = 16,
    transpose_B: bool = True,
    num_stages: int = 0,
    target: str = TILELANG_METAL_FP8_SPARSE_MLA_TARGET,
) -> str:
    """Lower the Path C FP8 Sparse-MLA QK probe and return MSL source."""

    import tilelang
    from tilelang import tvm

    prim = make_fp8_sparse_mla_qk_kernel(
        M=M,
        N=N,
        K=K,
        BM=BM,
        BN=BN,
        BK=BK,
        a_scale_size=a_scale_size,
        b_scale_size=b_scale_size,
        transpose_B=transpose_B,
        num_stages=num_stages,
    )
    artifact = tilelang.lower(prim, target=tvm.target.Target(target))
    if hasattr(artifact, "kernel_source"):
        return str(artifact.kernel_source)
    rt_mod = getattr(artifact, "rt_mod", None)
    if rt_mod is not None and hasattr(rt_mod, "get_source"):
        return str(rt_mod.get_source())
    return str(artifact)


def fp8_sparse_mla_qk_msl_features(msl: str) -> dict[str, int | bool | str]:
    """Return source markers used to guard Path C scale and fast-path semantics."""

    signature, body = _kernel_signature_and_body_for_feature_counts(msl)
    lowered = body.lower()
    return {
        "kernel_void": msl.count("kernel void"),
        "simdgroup_multiply_accumulate": msl.count("simdgroup_multiply_accumulate"),
        "simdgroup_load": msl.count("simdgroup_load"),
        "simdgroup_store": msl.count("simdgroup_store"),
        "fp8_e4m3_decode_helper": msl.count("__tvm_fp8_e4m3_to_half"),
        "A_scale_refs": body.count("A_scale["),
        "B_scale_refs": body.count("B_scale["),
        "signature_has_A_scale": "A_scale" in signature,
        "signature_has_B_scale": "B_scale" in signature,
        "float_a_val": "float a_val" in lowered,
        "float_b_val": "float b_val" in lowered,
        "threadgroup_half": "threadgroup half" in lowered,
    }


_FP8_SPARSE_MLA_QK_MSL_FEATURE_DEFAULTS: dict[str, int | bool | str] = {
    "kernel_void": 0,
    "simdgroup_multiply_accumulate": 0,
    "simdgroup_load": 0,
    "simdgroup_store": 0,
    "fp8_e4m3_decode_helper": 0,
    "A_scale_refs": 0,
    "B_scale_refs": 0,
    "signature_has_A_scale": False,
    "signature_has_B_scale": False,
    "float_a_val": False,
    "float_b_val": False,
    "threadgroup_half": False,
}


def _kernel_body_for_feature_counts(msl: str) -> str:
    _signature, body = _kernel_signature_and_body_for_feature_counts(msl)
    return body


def _kernel_signature_and_body_for_feature_counts(msl: str) -> tuple[str, str]:
    # RULE #1: `_split_kernel_msl` already raises a clear RuntimeError with
    # where+what ("missing 'kernel void'", "unbalanced parens/braces") when the
    # MSL is malformed. Do NOT swallow that into a crude string-split fallback —
    # the degraded parse would yield WRONG feature counts and silently mask a
    # genuinely-malformed kernel in the downstream Path C feature guards.
    _prelude, sig_text, body_text = _msl_transform._split_kernel_msl(msl)
    return sig_text, body_text


def _prefix_feature_keys(
    prefix: str,
    features: dict[str, int | bool | str],
) -> dict[str, int | bool | str]:
    return {f"{prefix}{key}": value for key, value in features.items()}


def fp8_sparse_mla_qk_scaled_matmul_probe_status(
    *,
    M: int = 1,
    N: int = 16,
    K: int = 64,
    BM: int = 1,
    BN: int = 16,
    BK: int = 64,
    a_scale_size: int = 1,
    b_scale_size: int = 16,
    transpose_B: bool = True,
    num_stages: int = 0,
    target: str = TILELANG_METAL_FP8_SPARSE_MLA_TARGET,
) -> SparseMLAFp8PathCStatus:
    """Probe the legacy ``T.fp8_scaled_matmul`` Sparse-MLA QK lowering only."""

    ok, reason = _tilelang_available()
    if not ok:
        return SparseMLAFp8PathCStatus(
            available=False,
            reason=reason,
            features={},
            target=target,
            m=M,
            n=N,
            k=K,
            transpose_B=transpose_B,
        )

    try:
        msl = lower_fp8_sparse_mla_qk_msl(
            M=M,
            N=N,
            K=K,
            BM=BM,
            BN=BN,
            BK=BK,
            a_scale_size=a_scale_size,
            b_scale_size=b_scale_size,
            transpose_B=transpose_B,
            num_stages=num_stages,
            target=target,
        )
    except Exception as exc:
        return SparseMLAFp8PathCStatus(
            available=False,
            reason=f"TileLang Metal lowering failed for FP8 Sparse-MLA QK shape: {type(exc).__name__}: {exc}",
            features={},
            target=target,
            m=M,
            n=N,
            k=K,
            transpose_B=transpose_B,
        )

    features = fp8_sparse_mla_qk_msl_features(msl)
    has_fast_path = bool(features["simdgroup_multiply_accumulate"])
    has_scale_refs = bool(features["A_scale_refs"]) and bool(features["B_scale_refs"])
    has_scale_signature = bool(features["signature_has_A_scale"]) and bool(
        features["signature_has_B_scale"]
    )
    has_scalar_fallback = bool(features["float_a_val"]) or bool(features["float_b_val"])
    if (
        has_fast_path
        and has_scale_refs
        and has_scale_signature
        and not has_scalar_fallback
    ):
        return SparseMLAFp8PathCStatus(
            available=True,
            reason=(
                "TileLang Path C FP8 Sparse-MLA QK probe lowers through "
                "T.fp8_scaled_matmul to Metal simdgroup MMA with scale loads"
            ),
            features=features,
            target=target,
            m=M,
            n=N,
            k=K,
            transpose_B=transpose_B,
        )

    blockers: list[str] = []
    if not has_fast_path:
        blockers.append("no simdgroup_multiply_accumulate")
    if not has_scale_refs or not has_scale_signature:
        blockers.append("scale operands disappeared from emitted MSL")
    if has_scalar_fallback:
        blockers.append("scalar fallback markers present")
    if M < 8 or BM < 8:
        blockers.append(
            "Sparse-MLA M=1/topk tile violates current Metal FP8 simdgroup tile constraints"
        )
    return SparseMLAFp8PathCStatus(
        available=False,
        reason="TileLang Path C FP8 Sparse-MLA QK is not safe to dispatch: "
        + "; ".join(blockers),
        features=features,
        target=target,
        m=M,
        n=N,
        k=K,
        transpose_B=transpose_B,
    )


def fp8_sparse_mla_qk_path_c_status(
    *,
    M: int = 1,
    N: int = 16,
    K: int = 64,
    BM: int = 1,
    BN: int = 16,
    BK: int = 64,
    a_scale_size: int = 1,
    b_scale_size: int = 16,
    transpose_B: bool = True,
    num_stages: int = 0,
    target: str = TILELANG_METAL_FP8_SPARSE_MLA_TARGET,
) -> SparseMLAFp8PathCStatus:
    """Availability probe for the dispatchable FP8 Sparse-MLA Path C QK tile."""

    probe_status = fp8_sparse_mla_qk_scaled_matmul_probe_status(
        M=M,
        N=N,
        K=K,
        BM=BM,
        BN=BN,
        BK=BK,
        a_scale_size=a_scale_size,
        b_scale_size=b_scale_size,
        transpose_B=transpose_B,
        num_stages=num_stages,
        target=target,
    )

    if M == 1 and BM == 1 and transpose_B:
        reducer_status = fp8_sparse_mla_qk_reduce_path_c_status(
            N=N,
            K=K,
            outputs_per_block=_SMFP8_QKR_DEFAULT_OUTPUTS_PER_BLOCK,
            reduce_threads=_SMFP8_QKR_DEFAULT_REDUCE_THREADS,
            vec=_SMFP8_QKR_DEFAULT_VEC,
            target=target,
        )
        legacy_features = _prefix_feature_keys(
            "legacy_fp8_scaled_matmul_probe_",
            {
                **_FP8_SPARSE_MLA_QK_MSL_FEATURE_DEFAULTS,
                **probe_status.features,
            },
        )
        if reducer_status.available:
            return SparseMLAFp8PathCStatus(
                available=True,
                reason=(
                    "TileLang Path C FP8 Sparse-MLA QK dispatches through the "
                    "real M=1/topk reducer; T.fp8_scaled_matmul remains probe-only "
                    "for this shape"
                ),
                features={
                    **reducer_status.features,
                    "dispatch_surface": "qk_reduce",
                    "runnable_qk_reduce_available": True,
                    "runnable_qk_reduce_reason": reducer_status.reason,
                    "legacy_fp8_scaled_matmul_probe_available": bool(
                        probe_status.available
                    ),
                    "legacy_fp8_scaled_matmul_probe_reason": probe_status.reason,
                    **legacy_features,
                },
                target=target,
                m=M,
                n=N,
                k=K,
                transpose_B=transpose_B,
            )
        if probe_status.available:
            return SparseMLAFp8PathCStatus(
                available=True,
                reason=probe_status.reason,
                features={
                    **probe_status.features,
                    "dispatch_surface": "fp8_scaled_matmul",
                    "runnable_qk_reduce_available": False,
                    "runnable_qk_reduce_reason": reducer_status.reason,
                    "legacy_fp8_scaled_matmul_probe_available": True,
                    "legacy_fp8_scaled_matmul_probe_reason": probe_status.reason,
                },
                target=target,
                m=M,
                n=N,
                k=K,
                transpose_B=transpose_B,
            )
        features = {
            "dispatch_surface": "unavailable",
            "runnable_qk_reduce_available": False,
            "runnable_qk_reduce_reason": reducer_status.reason,
            "legacy_fp8_scaled_matmul_probe_available": False,
            "legacy_fp8_scaled_matmul_probe_reason": probe_status.reason,
            **legacy_features,
        }
        return SparseMLAFp8PathCStatus(
            available=False,
            reason=(
                "TileLang Path C FP8 Sparse-MLA QK has no safe dispatch surface: "
                f"qk_reduce={reducer_status.reason}; "
                f"T.fp8_scaled_matmul={probe_status.reason}"
            ),
            features=features,
            target=target,
            m=M,
            n=N,
            k=K,
            transpose_B=transpose_B,
        )

    if probe_status.available:
        return SparseMLAFp8PathCStatus(
            available=True,
            reason=probe_status.reason,
            features={
                **probe_status.features,
                "dispatch_surface": "fp8_scaled_matmul",
                "legacy_fp8_scaled_matmul_probe_available": True,
                "legacy_fp8_scaled_matmul_probe_reason": probe_status.reason,
            },
            target=target,
            m=M,
            n=N,
            k=K,
            transpose_B=transpose_B,
        )
    features = {
        **probe_status.features,
        "dispatch_surface": "unavailable",
        "legacy_fp8_scaled_matmul_probe_available": False,
        "legacy_fp8_scaled_matmul_probe_reason": probe_status.reason,
    }
    return SparseMLAFp8PathCStatus(
        available=False,
        reason=probe_status.reason,
        features=features,
        target=target,
        m=M,
        n=N,
        k=K,
        transpose_B=transpose_B,
    )


def make_fp8_sparse_mla_qk_reduce_kernel(
    *,
    N: int,
    K: int,
    outputs_per_block: int = _SMFP8_QKR_DEFAULT_OUTPUTS_PER_BLOCK,
    reduce_threads: int = _SMFP8_QKR_DEFAULT_REDUCE_THREADS,
    vec: int = _SMFP8_QKR_DEFAULT_VEC,
    b_scale_size: int | None = None,
) -> Any:
    """Build the real Sparse-MLA FP8 QK tile as a TileLang reducer.

    This intentionally does not use ``T.fp8_scaled_matmul``.  The current
    Metal lowering rejects or scalarizes the ``M=1`` Sparse-MLA shape there,
    while the reducer below matches Path B's QK tile contract directly:

    * ``A_fp8`` is a single query row ``(1, K)`` in e4m3 byte storage.
    * ``B_fp8`` is gathered/transposed KV rows ``(N, K)``.
    * ``B_scale`` is scalar or per gathered KV row; both stay as caller-owned
      buffers without Python broadcast staging.
    """

    _validate_reduce_shape(
        N=N,
        K=K,
        outputs_per_block=outputs_per_block,
        reduce_threads=reduce_threads,
        vec=vec,
    )
    b_scale_extent = N if b_scale_size is None else int(b_scale_size)
    if b_scale_extent not in (1, N):
        raise ValueError(
            f"B_scale must be scalar or per-row for FP8 QK reducer; got {b_scale_extent=} {N=}"
        )

    import tilelang.language as T

    T = cast(Any, T)

    block_k = reduce_threads * vec
    g = globals()
    g.update(
        _SMFP8_QKR_N=N,
        _SMFP8_QKR_K=K,
        _SMFP8_QKR_NP=outputs_per_block,
        _SMFP8_QKR_RT=reduce_threads,
        _SMFP8_QKR_VEC=vec,
        _SMFP8_QKR_BLOCK_K=block_k,
        _SMFP8_QKR_K_WORDS=K // 4,
        _SMFP8_QKR_BS=b_scale_extent,
    )

    if vec == 4 and K % 4 == 0:

        @T.prim_func
        def fp8_sparse_mla_qk_reduce(
            A_fp8: T.Tensor((1, _SMFP8_QKR_K), "float8_e4m3"),
            A_scale: T.Tensor((1,), "float32"),
            B_fp8: T.Tensor((_SMFP8_QKR_N, _SMFP8_QKR_K), "float8_e4m3"),
            B_scale: T.Tensor((_SMFP8_QKR_BS,), "float32"),
            C: T.Tensor((1, _SMFP8_QKR_N), "float32"),
        ):
            with T.Kernel(
                T.ceildiv(_SMFP8_QKR_N, _SMFP8_QKR_NP),
                threads=(_SMFP8_QKR_RT, _SMFP8_QKR_NP),
            ) as bx:
                accum = T.alloc_local((1,), "float32")
                reduced = T.alloc_local((1,), "float32")
                kr = T.get_thread_binding(0)
                ni = T.get_thread_binding(1)
                col = bx * _SMFP8_QKR_NP + ni
                T.clear(accum)
                for ko in T.serial(T.ceildiv(_SMFP8_QKR_K_WORDS, _SMFP8_QKR_RT)):
                    i = ko * _SMFP8_QKR_RT + kr
                    if col < _SMFP8_QKR_N and i < _SMFP8_QKR_K_WORDS:
                        accum[0] += T.metal_fp8_e4m3_dot4(
                            T.access_ptr(A_fp8[0, 0], "r", extent=_SMFP8_QKR_K),
                            T.access_ptr(B_fp8[col, 0], "r", extent=_SMFP8_QKR_K),
                            i,
                            i,
                        )
                for out_lane in T.unroll(_SMFP8_QKR_NP):
                    if ni == out_lane:
                        with T.attr(
                            T.comm_reducer(lambda x, y: x + y, [T.cast(0, "float32")]),
                            "reduce_scope",
                            T.reinterpret(T.uint64(0), dtype="handle"),
                        ):
                            T.evaluate(
                                T.tvm_thread_allreduce(
                                    T.uint32(1),
                                    accum[0],
                                    True,
                                    reduced[0],
                                    kr,
                                    dtype="handle",
                                )
                            )
                        if kr == 0 and col < _SMFP8_QKR_N:
                            if _SMFP8_QKR_BS == 1:
                                C[0, col] = reduced[0] * A_scale[0] * B_scale[0]
                            else:
                                C[0, col] = reduced[0] * A_scale[0] * B_scale[col]

    else:

        @T.prim_func
        def fp8_sparse_mla_qk_reduce(
            A_fp8: T.Tensor((1, _SMFP8_QKR_K), "float8_e4m3"),
            A_scale: T.Tensor((1,), "float32"),
            B_fp8: T.Tensor((_SMFP8_QKR_N, _SMFP8_QKR_K), "float8_e4m3"),
            B_scale: T.Tensor((_SMFP8_QKR_BS,), "float32"),
            C: T.Tensor((1, _SMFP8_QKR_N), "float32"),
        ):
            with T.Kernel(
                T.ceildiv(_SMFP8_QKR_N, _SMFP8_QKR_NP),
                threads=(_SMFP8_QKR_RT, _SMFP8_QKR_NP),
            ) as bx:
                accum = T.alloc_local((1,), "float32")
                reduced = T.alloc_local((1,), "float32")
                kr = T.get_thread_binding(0)
                ni = T.get_thread_binding(1)
                col = bx * _SMFP8_QKR_NP + ni
                T.clear(accum)
                for ko in T.serial(T.ceildiv(_SMFP8_QKR_K, _SMFP8_QKR_BLOCK_K)):
                    for v in T.serial(_SMFP8_QKR_VEC):
                        k = ko * _SMFP8_QKR_BLOCK_K + kr * _SMFP8_QKR_VEC + v
                        if col < _SMFP8_QKR_N and k < _SMFP8_QKR_K:
                            accum[0] += T.cast(A_fp8[0, k], "float32") * T.cast(
                                B_fp8[col, k], "float32"
                            )
                for out_lane in T.unroll(_SMFP8_QKR_NP):
                    if ni == out_lane:
                        with T.attr(
                            T.comm_reducer(lambda x, y: x + y, [T.cast(0, "float32")]),
                            "reduce_scope",
                            T.reinterpret(T.uint64(0), dtype="handle"),
                        ):
                            T.evaluate(
                                T.tvm_thread_allreduce(
                                    T.uint32(1),
                                    accum[0],
                                    True,
                                    reduced[0],
                                    kr,
                                    dtype="handle",
                                )
                            )
                        if kr == 0 and col < _SMFP8_QKR_N:
                            if _SMFP8_QKR_BS == 1:
                                C[0, col] = reduced[0] * A_scale[0] * B_scale[0]
                            else:
                                C[0, col] = reduced[0] * A_scale[0] * B_scale[col]

    try:
        from tilelang.transform.simplify import apply_simplify

        return apply_simplify(fp8_sparse_mla_qk_reduce)
    except (ImportError, ModuleNotFoundError):
        # ACCEPTABLE: the simplify pass is a correctness-preserving optimization;
        # if it is absent from this TileLang build, the unsimplified kernel is
        # functionally identical. RULE #1: do NOT swallow a real crash *inside*
        # the pass (only the import is feature-absent) — that now propagates.
        return fp8_sparse_mla_qk_reduce


# --- Metal-4 cooperative-tensor (matmul2d) grouped QK reducer -----------------
#
# The M=1 ``make_fp8_sparse_mla_qk_reduce_kernel`` above is a single query row
# against the gathered top-k KV rows (scalar dot4 / allreduce). When several
# query heads in the same KV group (``head_kv`` rows) share one gathered KV
# matrix, the QK tile becomes a real ``[head_kv, K] @ [topk, K]^T`` GEMM whose
# M=head_kv, N=topk, K=qk_dim. For shapes that admit a legal Metal-4
# cooperative tile (M%16, N%32, K%16; see ``_coop_tile_for``), routing this
# grouped tile through the cooperative-tensor ``mpp::tensor_ops::matmul2d``
# GEMM (the same keystone emission #6 fp8_matmul uses) beats the per-row dot4
# reductions. The dot4 M=1 reducer remains the explicit shape-correct path for
# group shapes no cooperative tile divides (head_kv<16 / topk<32). This is NOT
# a silent fallback: the dispatcher in ``fp8_sparse_mla_qk_reduce_grouped_path_c``
# selects by ``_coop_tile_for`` and RAISES on genuinely unsupported shapes.

# TileLang resolves these module globals while decorating the nested grouped
# matmul2d @T.prim_func. Defaults keep pyright aligned with runtime values.
_SMFP8_QKMM_M = 16
_SMFP8_QKMM_N = 32
_SMFP8_QKMM_K = 64
_SMFP8_QKMM_BM = 16
_SMFP8_QKMM_BN = 32
_SMFP8_QKMM_BK = 32
_SMFP8_QKMM_THREADS = 32
_SMFP8_QKMM_BS = 1


def _aligned_contiguous(array: mx.array) -> mx.array:
    """Return a freshly-allocated, row-major, 64-byte-aligned copy of ``array``.

    The tvm-ffi DLPack import enforces ``require_alignment=64``. A 1-row slice
    (e.g. ``A_fp8[row:row+1, :]``) inherits a non-aligned byte offset from its
    base buffer; ``mx.contiguous`` alone may preserve that offset. Materializing
    through a fresh elementwise op forces MLX to allocate a new aligned buffer.
    """

    # A view/slice can carry a non-64-aligned byte offset from its parent
    # buffer. Force a genuine elementwise materialization (uint8 bit-or with 0 /
    # numeric add with 0) so MLX allocates a NEW buffer at offset 0, then make it
    # contiguous. ``mx.contiguous`` alone is a no-op on an already-contiguous
    # sub-offset slice and would preserve the bad offset.
    if array.dtype == mx.uint8:
        fresh = mx.bitwise_or(array, mx.array(0, dtype=mx.uint8))
    else:
        fresh = array + mx.array(0, dtype=array.dtype)
    return mx.contiguous(fresh)


def make_fp8_sparse_mla_qk_reduce_matmul2d_kernel(
    *,
    M: int,
    N: int,
    K: int,
    BM: int,
    BN: int,
    BK: int,
    threads: int,
    b_scale_size: int | None = None,
) -> Any:
    """Build the grouped FP8 QK tile as a Metal-4 cooperative-tensor GEMM.

    Computes ``C[M, N] = (A_fp8[M, K] @ B_fp8[N, K]^T) * A_scale * B_scale``
    where ``M = head_kv`` query rows share the gathered top-k KV matrix.

    The FP8 ``e4m3`` tiles are dequantized to ``half`` into ``shared``
    cooperative-input buffers (each byte routed through an explicit ``float32``
    scratch so the Metal backend emits the per-element ``__tvm_fp8_e4m3_to_half``
    LUT decode rather than a mis-lowered ``(half4)(uchar4)`` integer cast). The
    accumulator ``C_shared`` lives in ``shared`` scope, which is what triggers
    the cooperative-tensor ``matmul2d`` lowering in ``gemm_metal.py``. Scalar A
    scale and scalar/per-row B scale are applied to the shared accumulator once
    after the K reduction.

    ``BM/BN/BK/threads`` MUST be a legal cooperative tile (per ``_coop_tile_for``)
    and must evenly divide ``M/N/K`` — the cooperative kernel does not bounds-
    guard a partial output tile. The caller guarantees this via shape-correct
    dispatch; here a violated precondition RAISES (RULE #1: no silent gate).
    """

    if M <= 0 or N <= 0 or K <= 0:
        raise ValueError(
            f"grouped FP8 QK matmul2d requires positive M/N/K; got M={M} N={N} K={K}"
        )
    if M % BM or N % BN or K % BK:
        raise ValueError(
            "grouped FP8 QK matmul2d tile does not divide the GEMM extents: "
            f"M={M} N={N} K={K} BM={BM} BN={BN} BK={BK}"
        )
    if BM % 16 or BN % 32 or BK % 16:
        raise ValueError(
            "grouped FP8 QK matmul2d tile violates the Metal-4 cooperative "
            f"constraint (BM%16, BN%32, BK%16): BM={BM} BN={BN} BK={BK}"
        )
    b_scale_extent = N if b_scale_size is None else int(b_scale_size)
    if b_scale_extent not in (1, N):
        raise ValueError(
            f"B_scale must be scalar or per-row for grouped FP8 QK matmul2d; "
            f"got {b_scale_extent=} {N=}"
        )

    from tilelang import language as T
    from tvm.target import Target

    T = cast(Any, T)

    g = globals()
    g.update(
        T=T,
        Target=Target,
        _SMFP8_QKMM_M=int(M),
        _SMFP8_QKMM_N=int(N),
        _SMFP8_QKMM_K=int(K),
        _SMFP8_QKMM_BM=int(BM),
        _SMFP8_QKMM_BN=int(BN),
        _SMFP8_QKMM_BK=int(BK),
        _SMFP8_QKMM_THREADS=int(threads),
        _SMFP8_QKMM_BS=int(b_scale_extent),
    )

    @T.prim_func
    def fp8_sparse_mla_qk_reduce_matmul2d(
        A_fp8: T.Tensor((_SMFP8_QKMM_M, _SMFP8_QKMM_K), "float8_e4m3"),
        A_scale: T.Tensor((1,), "float32"),
        B_fp8: T.Tensor((_SMFP8_QKMM_N, _SMFP8_QKMM_K), "float8_e4m3"),
        B_scale: T.Tensor((_SMFP8_QKMM_BS,), "float32"),
        C: T.Tensor((_SMFP8_QKMM_M, _SMFP8_QKMM_N), "float32"),
    ):
        with T.Kernel(
            T.ceildiv(_SMFP8_QKMM_N, _SMFP8_QKMM_BN),
            T.ceildiv(_SMFP8_QKMM_M, _SMFP8_QKMM_BM),
            threads=_SMFP8_QKMM_THREADS,
        ) as (bx, by):
            A_shared = T.alloc_shared((_SMFP8_QKMM_BM, _SMFP8_QKMM_BK), "float16", scope="shared")
            B_shared = T.alloc_shared((_SMFP8_QKMM_BN, _SMFP8_QKMM_BK), "float16", scope="shared")
            C_shared = T.alloc_shared((_SMFP8_QKMM_BM, _SMFP8_QKMM_BN), "float32", scope="shared")
            T.clear(C_shared)
            for ko in T.serial(T.ceildiv(_SMFP8_QKMM_K, _SMFP8_QKMM_BK)):
                for i, kk in T.Parallel(_SMFP8_QKMM_BM, _SMFP8_QKMM_BK):
                    a_val = T.alloc_var("float32")
                    a_val = T.cast(A_fp8[by * _SMFP8_QKMM_BM + i, ko * _SMFP8_QKMM_BK + kk], "float32")
                    A_shared[i, kk] = T.cast(a_val, "float16")
                for j, kk in T.Parallel(_SMFP8_QKMM_BN, _SMFP8_QKMM_BK):
                    b_val = T.alloc_var("float32")
                    b_val = T.cast(B_fp8[bx * _SMFP8_QKMM_BN + j, ko * _SMFP8_QKMM_BK + kk], "float32")
                    B_shared[j, kk] = T.cast(b_val, "float16")
                T.gemm(A_shared, B_shared, C_shared, transpose_B=True)
            sa = A_scale[0]
            for i, j in T.Parallel(_SMFP8_QKMM_BM, _SMFP8_QKMM_BN):
                col = bx * _SMFP8_QKMM_BN + j
                if _SMFP8_QKMM_BS == 1:
                    C_shared[i, j] = C_shared[i, j] * sa * B_scale[0]
                else:
                    C_shared[i, j] = C_shared[i, j] * sa * B_scale[col]
            T.copy(C_shared, C[by * _SMFP8_QKMM_BM, bx * _SMFP8_QKMM_BN])

    return fp8_sparse_mla_qk_reduce_matmul2d


@lru_cache(maxsize=128)
def _qk_reduce_matmul2d_kernel_for(
    M: int,
    N: int,
    K: int,
    BM: int,
    BN: int,
    BK: int,
    threads: int,
    b_scale_size: int,
) -> Any:
    """Build and cache the tvm-ffi cooperative-tensor grouped QK reducer."""

    import tilelang

    prim = make_fp8_sparse_mla_qk_reduce_matmul2d_kernel(
        M=M,
        N=N,
        K=K,
        BM=BM,
        BN=BN,
        BK=BK,
        threads=threads,
        b_scale_size=b_scale_size,
    )
    return tilelang.compile(
        prim,
        target=_msl_transform._as_metal_target(TILELANG_METAL_FP8_SPARSE_MLA_TARGET),
        execution_backend="tvm_ffi",
        out_idx=4,
    )


def fp8_sparse_mla_qk_reduce_grouped_path_c(
    A_fp8: mx.array,
    A_scale: mx.array,
    B_fp8: mx.array,
    B_scale: mx.array,
    out: mx.array,
) -> mx.array:
    """Run the grouped FP8 Sparse-MLA QK tile with explicit matmul2d/dot4 dispatch.

    ``A_fp8`` is ``(M=head_kv, K)`` e4m3 uint8 storage (the query rows of one KV
    group), ``B_fp8`` is the gathered ``(N=topk, K)`` KV rows, ``out`` is the
    caller-owned ``(M, N)`` fp32 score tile.

    Shape-correct dispatch (RULE #1 -- no silent fallback):

    * When ``_coop_tile_for(M, N, K)`` returns a legal cooperative tile, the
      grouped tile runs the Metal-4 cooperative-tensor ``matmul2d`` GEMM.
    * Otherwise (e.g. head_kv<16 or topk<32, no tile divides the shape), each of
      the ``M`` query rows is routed through the proven M=1 dot4 reducer
      (``fp8_sparse_mla_qk_reduce_path_c``). This is the explicit shape-correct
      path, not a hidden gate.

    A genuinely unsupported shape (non-positive extents, K not divisible by 4 so
    even the dot4 reducer cannot lower) RAISES with where+what.
    """

    if not can_run_metal():
        raise RuntimeError(
            "fp8_sparse_mla_qk_reduce_grouped_path_c: MLX Metal backend "
            "unavailable; the FP8 Sparse-MLA QK tile cannot dispatch."
        )
    if A_fp8.ndim != 2 or B_fp8.ndim != 2:
        raise ValueError(
            "fp8_sparse_mla_qk_reduce_grouped_path_c: A_fp8/B_fp8 must be 2D; "
            f"got A={tuple(A_fp8.shape)} B={tuple(B_fp8.shape)}"
        )
    if A_fp8.dtype != mx.uint8 or B_fp8.dtype != mx.uint8:
        raise ValueError(
            "fp8_sparse_mla_qk_reduce_grouped_path_c: A_fp8/B_fp8 must be uint8 "
            f"e4m3 storage; got {A_fp8.dtype}, {B_fp8.dtype}"
        )
    M = int(A_fp8.shape[0])
    K = int(A_fp8.shape[1])
    N = int(B_fp8.shape[0])
    if M <= 0 or N <= 0 or K <= 0 or int(B_fp8.shape[1]) != K:
        raise ValueError(
            "fp8_sparse_mla_qk_reduce_grouped_path_c: A/B shape mismatch: "
            f"A={tuple(A_fp8.shape)} B={tuple(B_fp8.shape)}"
        )
    if A_scale.dtype != mx.float32 or B_scale.dtype != mx.float32:
        raise TypeError(
            "fp8_sparse_mla_qk_reduce_grouped_path_c: A_scale/B_scale must be "
            f"float32; got {A_scale.dtype}, {B_scale.dtype}"
        )
    if A_scale.size != 1:
        raise ValueError(
            "fp8_sparse_mla_qk_reduce_grouped_path_c: A_scale must be a single "
            f"FP32 scale; got shape {tuple(A_scale.shape)}"
        )
    if B_scale.size not in (1, N):
        raise ValueError(
            "fp8_sparse_mla_qk_reduce_grouped_path_c: B_scale must be scalar or "
            f"N={N} row scales; got shape {tuple(B_scale.shape)}"
        )
    if tuple(out.shape) != (M, N) or out.dtype != mx.float32:
        raise ValueError(
            "fp8_sparse_mla_qk_reduce_grouped_path_c: out must be fp32 shape "
            f"({M}, {N}); got shape {tuple(out.shape)} dtype {out.dtype}"
        )

    from cppmega_mlx.nn._tilelang.fp8_matmul_path_c import _coop_tile_for

    coop = _coop_tile_for(M, N, K)
    b_scale_extent = 1 if int(B_scale.size) == 1 else N
    if coop is not None:
        bm, bn, bk, threads = coop
        A_scale_1d = mx.contiguous(A_scale.reshape((1,)))
        B_scale_1d = mx.contiguous(B_scale.reshape((B_scale.size,)))
        try:
            kernel = _qk_reduce_matmul2d_kernel_for(
                M, N, K, bm, bn, bk, threads, b_scale_extent
            )
        except Exception as exc:
            # RULE #1: a cooperative-tensor compile failure on a shape the tile
            # divides is a real bug, not a "not applicable" signal -- raise it,
            # do NOT silently drop to dot4.
            raise RuntimeError(
                "fp8_sparse_mla_qk_reduce_grouped_path_c: cooperative-tensor "
                f"matmul2d compile failed for M={M} N={N} K={K} tile={coop} "
                f"({type(exc).__name__}: {exc})."
            ) from exc
        returned = kernel(A_fp8, A_scale_1d, B_fp8, B_scale_1d, out)
        result = returned[0] if isinstance(returned, (list, tuple)) else returned
        return cast(mx.array, result)

    # No legal cooperative tile divides this grouped shape: route each query row
    # through the M=1 dot4 reducer. Explicit shape-correct dispatch, not a gate.
    if K % 4 != 0:
        raise ValueError(
            "fp8_sparse_mla_qk_reduce_grouped_path_c: shape M={} N={} K={} admits "
            "no cooperative tile and K is not a multiple of 4, so the dot4 reducer "
            "cannot lower either -- unsupported.".format(M, N, K)
        )
    # The M=1 dot4 reducer dispatches through tvm-ffi which requires 64-byte
    # aligned, contiguous buffers; re-materialize fresh aligned copies so a
    # non-aligned slice/view does not crash the DLPack import.
    B_fp8_a = _aligned_contiguous(B_fp8)
    B_scale_a = _aligned_contiguous(B_scale.reshape((B_scale.size,)))
    A_scale_a = _aligned_contiguous(A_scale.reshape((1,)))
    mx.eval(B_fp8_a, B_scale_a, A_scale_a)
    for row in range(M):
        a_row = _aligned_contiguous(A_fp8[row : row + 1, :])
        mx.eval(a_row)
        row_scores = fp8_sparse_mla_qk_reduce_path_c(
            a_row,
            A_scale_a,
            B_fp8_a,
            B_scale_a,
        )
        if row_scores is None:
            raise RuntimeError(
                "fp8_sparse_mla_qk_reduce_grouped_path_c: dot4 reducer returned "
                f"None for grouped fallback row {row} (M={M} N={N} K={K})."
            )
        out[row : row + 1, :] = row_scores.reshape((1, N)).astype(mx.float32)
    return out


def lower_fp8_sparse_mla_qk_reduce_msl(
    *,
    N: int = 16,
    K: int = 64,
    outputs_per_block: int = _SMFP8_QKR_DEFAULT_OUTPUTS_PER_BLOCK,
    reduce_threads: int = _SMFP8_QKR_DEFAULT_REDUCE_THREADS,
    vec: int = _SMFP8_QKR_DEFAULT_VEC,
    b_scale_size: int | None = None,
    target: str = TILELANG_METAL_FP8_SPARSE_MLA_TARGET,
) -> str:
    """Lower the real-shape Path C FP8 Sparse-MLA QK reducer to MSL."""

    import tilelang
    from tilelang import tvm

    prim = make_fp8_sparse_mla_qk_reduce_kernel(
        N=N,
        K=K,
        outputs_per_block=outputs_per_block,
        reduce_threads=reduce_threads,
        vec=vec,
        b_scale_size=b_scale_size,
    )
    artifact = tilelang.lower(prim, target=tvm.target.Target(target))
    if hasattr(artifact, "kernel_source"):
        return str(artifact.kernel_source)
    rt_mod = getattr(artifact, "rt_mod", None)
    if rt_mod is not None and hasattr(rt_mod, "get_source"):
        return str(rt_mod.get_source())
    return str(artifact)


def fp8_sparse_mla_qk_reduce_msl_features(msl: str) -> dict[str, int | bool | str]:
    """Return source markers for the runnable FP8 QK reducer."""

    signature, body = _kernel_signature_and_body_for_feature_counts(msl)
    lowered = body.lower()
    scalar_decode_sites = body.count("__tvm_fp8_e4m3_to_half(")
    return {
        "kernel_void": msl.count("kernel void"),
        "fp8_e4m3_decode_helper": msl.count("__tvm_fp8_e4m3_to_half"),
        "scalar_fp8_byte_decode": scalar_decode_sites,
        "scalar_fp8_byte_decode_calls": scalar_decode_sites,
        "tvm_thread_allreduce": body.count("tvm_thread_allreduce"),
        "simd_sum": body.count("simd_sum"),
        "simd_shuffle_down": body.count("simd_shuffle_down"),
        "A_scale_refs": body.count("A_scale["),
        "B_scale_refs": body.count("B_scale["),
        "signature_has_A_scale": "A_scale" in signature,
        "signature_has_B_scale": "B_scale" in signature,
        "per_row_B_scale": body.count("B_scale[") > body.count("B_scale[0]"),
        "reinterpret_cast": body.count("reinterpret_cast"),
        "device_const_uint": body.count("device const uint"),
        "uchar4": lowered.count("uchar4"),
        "fp8_e4m3_lut": body.count("fp8_e4m3fn_lut"),
        "metal_fp8_dot4_helper": msl.count("__tvm_fp8_e4m3_dot4_packed"),
        "threadgroup_half": "threadgroup half" in lowered,
        "qk_shape": "m1_n_topk_k",
    }


def make_fp8_sparse_mla_indexed_qk_reduce_kernel(
    *,
    batch: int,
    seq_len: int,
    heads: int,
    seq_len_kv: int,
    kv_group: int,
    topk: int,
    K: int,
    outputs_per_block: int = _SMFP8_QKR_DEFAULT_OUTPUTS_PER_BLOCK,
    reduce_threads: int = _SMFP8_QKR_DEFAULT_REDUCE_THREADS,
    vec: int = _SMFP8_QKR_DEFAULT_VEC,
    index_dtype: str = "int32",
) -> Any:
    """Build a full-shape indexed FP8 QK score reducer.

    This removes the old host pre-gather blocker for Path C QK experiments:
    the kernel consumes full ``q_fp8``, ``kv_fp8``, ``indices`` and scale
    buffers, then writes ``scores[B, S, H, TOPK]`` directly.
    """

    head_kv = _validate_indexed_reduce_shape(
        batch=batch,
        seq_len=seq_len,
        heads=heads,
        seq_len_kv=seq_len_kv,
        kv_group=kv_group,
        topk=topk,
        K=K,
        outputs_per_block=outputs_per_block,
        reduce_threads=reduce_threads,
        vec=vec,
    )
    if vec != 4 or K % 4 != 0:
        raise ValueError(
            "FP8 Sparse-MLA Path C indexed reducer requires packed dot4 lowering "
            f"(vec=4 and K % 4 == 0); got {vec=} {K=}"
        )

    import tilelang.language as T

    T = cast(Any, T)

    block_k = reduce_threads * vec
    lanes = batch * seq_len * heads
    q_size = lanes * K
    kv_size = batch * seq_len_kv * kv_group * K
    q_scale_size = batch * seq_len * heads
    kv_scale_size = batch * seq_len_kv * kv_group
    idx_size = batch * seq_len * kv_group * topk
    out_size = lanes * topk
    g = globals()
    g.update(
        _SMFP8_IQKR_B=batch,
        _SMFP8_IQKR_S=seq_len,
        _SMFP8_IQKR_H=heads,
        _SMFP8_IQKR_SKV=seq_len_kv,
        _SMFP8_IQKR_G=kv_group,
        _SMFP8_IQKR_HEAD_KV=head_kv,
        _SMFP8_IQKR_TOPK=topk,
        _SMFP8_IQKR_K=K,
        _SMFP8_IQKR_LANES=lanes,
        _SMFP8_IQKR_Q_SIZE=q_size,
        _SMFP8_IQKR_KV_SIZE=kv_size,
        _SMFP8_IQKR_Q_SCALE_SIZE=q_scale_size,
        _SMFP8_IQKR_KV_SCALE_SIZE=kv_scale_size,
        _SMFP8_IQKR_IDX_SIZE=idx_size,
        _SMFP8_IQKR_OUT_SIZE=out_size,
        _SMFP8_IQKR_NP=outputs_per_block,
        _SMFP8_IQKR_RT=reduce_threads,
        _SMFP8_IQKR_VEC=vec,
        _SMFP8_IQKR_BLOCK_K=block_k,
        _SMFP8_IQKR_K_WORDS=K // 4,
        _SMFP8_IQKR_INDEX_DTYPE=index_dtype,
    )

    @T.prim_func
    def fp8_sparse_mla_indexed_qk_reduce(
        q_fp8: T.Tensor(
            (
                _SMFP8_IQKR_B,
                _SMFP8_IQKR_S,
                _SMFP8_IQKR_H,
                _SMFP8_IQKR_K,
            ),
            "float8_e4m3",
        ),
        q_scale: T.Tensor(
            (_SMFP8_IQKR_B, _SMFP8_IQKR_S, _SMFP8_IQKR_H), "float32"
        ),
        kv_fp8: T.Tensor(
            (
                _SMFP8_IQKR_B,
                _SMFP8_IQKR_SKV,
                _SMFP8_IQKR_G,
                _SMFP8_IQKR_K,
            ),
            "float8_e4m3",
        ),
        kv_scale: T.Tensor(
            (_SMFP8_IQKR_B, _SMFP8_IQKR_SKV, _SMFP8_IQKR_G), "float32"
        ),
        indices: T.Tensor(
            (_SMFP8_IQKR_B, _SMFP8_IQKR_S, _SMFP8_IQKR_G, _SMFP8_IQKR_TOPK),
            _SMFP8_IQKR_INDEX_DTYPE,
        ),
        sm_scale_buf: T.Tensor((1,), "float32"),
        scores: T.Tensor(
            (_SMFP8_IQKR_B, _SMFP8_IQKR_S, _SMFP8_IQKR_H, _SMFP8_IQKR_TOPK),
            "float32",
        ),
    ):
        with T.Kernel(
            _SMFP8_IQKR_LANES,
            T.ceildiv(_SMFP8_IQKR_TOPK, _SMFP8_IQKR_NP),
            threads=(_SMFP8_IQKR_RT, _SMFP8_IQKR_NP),
        ) as (lane_gid, topk_block):
            accum = T.alloc_local((1,), "float32")
            reduced = T.alloc_local((1,), "float32")
            gather_idx = T.alloc_local((1,), "int32")
            kr = T.get_thread_binding(0)
            ni = T.get_thread_binding(1)
            topk_col = topk_block * _SMFP8_IQKR_NP + ni
            h = lane_gid % _SMFP8_IQKR_H
            bs = lane_gid // _SMFP8_IQKR_H
            b = bs // _SMFP8_IQKR_S
            s = bs - b * _SMFP8_IQKR_S
            group = h // _SMFP8_IQKR_HEAD_KV
            q_base = lane_gid * _SMFP8_IQKR_K
            T.clear(accum)
            if topk_col < _SMFP8_IQKR_TOPK:
                gather_idx[0] = T.cast(indices[b, s, group, topk_col], "int32")
                if gather_idx[0] >= 0 and gather_idx[0] < _SMFP8_IQKR_SKV:
                    kv_base = (
                        (b * _SMFP8_IQKR_SKV + gather_idx[0]) * _SMFP8_IQKR_G + group
                    ) * _SMFP8_IQKR_K
                    for ko in T.serial(T.ceildiv(_SMFP8_IQKR_K_WORDS, _SMFP8_IQKR_RT)):
                        i = ko * _SMFP8_IQKR_RT + kr
                        if i < _SMFP8_IQKR_K_WORDS:
                            accum[0] += T.metal_fp8_e4m3_dot4(
                                T.tvm_access_ptr(
                                    T.type_annotation("float8_e4m3"),
                                    q_fp8.data,
                                    q_base,
                                    _SMFP8_IQKR_K,
                                    1,
                                ),
                                T.tvm_access_ptr(
                                    T.type_annotation("float8_e4m3"),
                                    kv_fp8.data,
                                    kv_base,
                                    _SMFP8_IQKR_K,
                                    1,
                                ),
                                i,
                                i,
                            )
            for out_lane in T.unroll(_SMFP8_IQKR_NP):
                if ni == out_lane:
                    with T.attr(
                        T.comm_reducer(lambda x, y: x + y, [T.cast(0, "float32")]),
                        "reduce_scope",
                        T.reinterpret(T.uint64(0), dtype="handle"),
                    ):
                        T.evaluate(
                            T.tvm_thread_allreduce(
                                T.uint32(1),
                                accum[0],
                                True,
                                reduced[0],
                                kr,
                                dtype="handle",
                            )
                        )
                    if kr == 0 and topk_col < _SMFP8_IQKR_TOPK:
                        gather_idx[0] = T.cast(indices[b, s, group, topk_col], "int32")
                        if gather_idx[0] < 0 or gather_idx[0] >= _SMFP8_IQKR_SKV:
                            scores[b, s, h, topk_col] = T.float32(
                                _SMFP8_INVALID_SCORE_SENTINEL
                            )
                        else:
                            scores[b, s, h, topk_col] = (
                                reduced[0]
                                * q_scale[b, s, h]
                                * kv_scale[b, gather_idx[0], group]
                                * sm_scale_buf[0]
                            )

    try:
        from tilelang.transform.simplify import apply_simplify

        return apply_simplify(fp8_sparse_mla_indexed_qk_reduce)
    except (ImportError, ModuleNotFoundError):
        # ACCEPTABLE: simplify is a correctness-preserving optimization; if the
        # pass is absent the unsimplified kernel is functionally identical.
        # RULE #1: a real crash inside the pass now propagates (not swallowed).
        return fp8_sparse_mla_indexed_qk_reduce


def lower_fp8_sparse_mla_indexed_qk_reduce_msl(
    *,
    batch: int = 1,
    seq_len: int = 1,
    heads: int = 1,
    seq_len_kv: int = 16,
    kv_group: int = 1,
    topk: int = 16,
    K: int = 64,
    outputs_per_block: int = _SMFP8_QKR_DEFAULT_OUTPUTS_PER_BLOCK,
    reduce_threads: int = _SMFP8_QKR_DEFAULT_REDUCE_THREADS,
    vec: int = _SMFP8_QKR_DEFAULT_VEC,
    target: str = TILELANG_METAL_FP8_SPARSE_MLA_TARGET,
) -> str:
    """Lower the indexed full-shape FP8 QK reducer to MSL."""

    import tilelang
    from tilelang import tvm

    prim = make_fp8_sparse_mla_indexed_qk_reduce_kernel(
        batch=batch,
        seq_len=seq_len,
        heads=heads,
        seq_len_kv=seq_len_kv,
        kv_group=kv_group,
        topk=topk,
        K=K,
        outputs_per_block=outputs_per_block,
        reduce_threads=reduce_threads,
        vec=vec,
    )
    artifact = tilelang.lower(prim, target=tvm.target.Target(target))
    if hasattr(artifact, "kernel_source"):
        return str(artifact.kernel_source)
    rt_mod = getattr(artifact, "rt_mod", None)
    if rt_mod is not None and hasattr(rt_mod, "get_source"):
        return str(rt_mod.get_source())
    return str(artifact)


def fp8_sparse_mla_indexed_qk_reduce_msl_features(
    msl: str,
) -> dict[str, int | bool | str]:
    """Return source markers for the indexed full-shape FP8 QK reducer."""

    signature, body = _kernel_signature_and_body_for_feature_counts(msl)
    lowered = body.lower()
    scalar_decode_sites = body.count("__tvm_fp8_e4m3_to_half(")
    return {
        "kernel_void": msl.count("kernel void"),
        "fp8_e4m3_decode_helper": msl.count("__tvm_fp8_e4m3_to_half"),
        "scalar_fp8_byte_decode": scalar_decode_sites,
        "scalar_fp8_byte_decode_calls": scalar_decode_sites,
        "tvm_thread_allreduce": body.count("tvm_thread_allreduce"),
        "simd_sum": body.count("simd_sum"),
        "simd_shuffle_down": body.count("simd_shuffle_down"),
        "q_scale_refs": body.count("q_scale["),
        "kv_scale_refs": body.count("kv_scale["),
        "indices_refs": body.count("indices["),
        "sm_scale_refs": body.count("sm_scale_buf["),
        "signature_has_q_scale": "q_scale" in signature,
        "signature_has_kv_scale": "kv_scale" in signature,
        "signature_has_indices": "indices" in signature,
        "signature_has_sm_scale": "sm_scale_buf" in signature,
        "invalid_index_guard": "-3.402823" in msl
        or "-INFINITY" in msl
        or "-1.0f/0.0f" in msl,
        "reinterpret_cast": body.count("reinterpret_cast"),
        "device_const_uint": body.count("device const uint"),
        "uchar4": lowered.count("uchar4"),
        "fp8_e4m3_lut": body.count("fp8_e4m3fn_lut"),
        "metal_fp8_dot4_helper": msl.count("__tvm_fp8_e4m3_dot4_packed"),
        "threadgroup_half": "threadgroup half" in lowered,
        "qk_shape": "indexed_b_s_h_topk_k",
    }


@lru_cache(maxsize=128)
def _qk_reduce_kernel_for(
    N: int,
    K: int,
    outputs_per_block: int,
    reduce_threads: int,
    vec: int,
    b_scale_size: int,
) -> tuple[Any, _msl_transform.TileLangMSLLowering, list[str]]:
    """Build and cache the tvm-ffi dispatchable TileLang QK reducer."""

    prim = make_fp8_sparse_mla_qk_reduce_kernel(
        N=N,
        K=K,
        outputs_per_block=outputs_per_block,
        reduce_threads=reduce_threads,
        vec=vec,
        b_scale_size=b_scale_size,
    )
    lowering = cast(
        _msl_transform.TileLangMSLLowering,
        dispatch_lower(
            prim,
            target=TILELANG_METAL_FP8_SPARSE_MLA_TARGET,
            return_msl=True,
        ),
    )
    input_names = [name for name in lowering.buffer_param_names if name != "C"]
    if set(input_names) != {"A_fp8", "A_scale", "B_fp8", "B_scale"}:
        raise MSLDispatchUnsupported(
            "unexpected TileLang QK reducer buffer signature: "
            + ", ".join(lowering.buffer_param_names)
        )
    import tilelang

    kernel = tilelang.compile(
        prim,
        target=_msl_transform._as_metal_target(TILELANG_METAL_FP8_SPARSE_MLA_TARGET),
        execution_backend="tvm_ffi",
        out_idx=4,
    )
    return kernel, lowering, input_names


@lru_cache(maxsize=128)
def _indexed_qk_reduce_kernel_for(
    batch: int,
    seq_len: int,
    heads: int,
    seq_len_kv: int,
    kv_group: int,
    topk: int,
    K: int,
    outputs_per_block: int,
    reduce_threads: int,
    vec: int,
    index_dtype: str,
) -> tuple[Any, _msl_transform.TileLangMSLLowering, list[str]]:
    """Build and cache the tvm-ffi dispatchable indexed TileLang QK reducer."""

    prim = make_fp8_sparse_mla_indexed_qk_reduce_kernel(
        batch=batch,
        seq_len=seq_len,
        heads=heads,
        seq_len_kv=seq_len_kv,
        kv_group=kv_group,
        topk=topk,
        K=K,
        outputs_per_block=outputs_per_block,
        reduce_threads=reduce_threads,
        vec=vec,
        index_dtype=index_dtype,
    )
    lowering = cast(
        _msl_transform.TileLangMSLLowering,
        dispatch_lower(
            prim,
            target=TILELANG_METAL_FP8_SPARSE_MLA_TARGET,
            return_msl=True,
        ),
    )
    input_names = [name for name in lowering.buffer_param_names if name != "scores"]
    if set(input_names) != {
        "q_fp8",
        "q_scale",
        "kv_fp8",
        "kv_scale",
        "indices",
        "sm_scale_buf",
    }:
        raise MSLDispatchUnsupported(
            "unexpected TileLang indexed QK reducer buffer signature: "
            + ", ".join(lowering.buffer_param_names)
        )
    import tilelang

    kernel = tilelang.compile(
        prim,
        target=_msl_transform._as_metal_target(TILELANG_METAL_FP8_SPARSE_MLA_TARGET),
        execution_backend="tvm_ffi",
        out_idx=6,
    )
    return kernel, lowering, input_names


def _grid_for_lowering(
    lowering: _msl_transform.TileLangMSLLowering,
) -> tuple[int, int, int]:
    return (
        max(1, lowering.grid[0] * lowering.threadgroup[0]),
        max(1, lowering.grid[1] * lowering.threadgroup[1]),
        max(1, lowering.grid[2] * lowering.threadgroup[2]),
    )


def _normalize_qk_reduce_inputs(
    A_fp8: mx.array,
    A_scale: mx.array,
    B_fp8: mx.array,
    B_scale: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array, int, int, int]:
    if A_fp8.ndim != 2 or A_fp8.shape[0] != 1:
        raise ValueError(f"A_fp8 must have shape (1, K); got {tuple(A_fp8.shape)}")
    if B_fp8.ndim != 2:
        raise ValueError(f"B_fp8 must have shape (N, K); got {tuple(B_fp8.shape)}")
    if A_fp8.dtype != mx.uint8 or B_fp8.dtype != mx.uint8:
        raise ValueError(
            f"A_fp8/B_fp8 must be mx.uint8 e4m3 storage; got {A_fp8.dtype}, {B_fp8.dtype}"
        )
    n = int(B_fp8.shape[0])
    k = int(A_fp8.shape[1])
    if n <= 0 or k <= 0 or int(B_fp8.shape[1]) != k:
        raise ValueError(
            f"A_fp8/B_fp8 shape mismatch: A={tuple(A_fp8.shape)}, B={tuple(B_fp8.shape)}"
        )
    if A_scale.size != 1:
        raise ValueError(
            f"A_scale must contain exactly one FP32 scale; got shape {tuple(A_scale.shape)}"
        )
    if B_scale.size not in (1, n):
        raise ValueError(
            f"B_scale must contain one scalar scale or N={n} row scales; got shape {tuple(B_scale.shape)}"
        )

    if A_scale.dtype != mx.float32 or B_scale.dtype != mx.float32:
        raise TypeError(
            f"A_scale/B_scale must be float32; got {A_scale.dtype}, {B_scale.dtype}"
        )
    A_scale_1d = A_scale.reshape((1,))
    B_scale_1d = B_scale.reshape((B_scale.size,))
    return (
        A_fp8,
        A_scale_1d,
        B_fp8,
        B_scale_1d,
        n,
        k,
        int(B_scale_1d.size),
    )


def _normalize_indexed_qk_reduce_inputs(
    q_fp8: mx.array,
    q_scale: mx.array,
    kv_fp8: mx.array,
    kv_scale: mx.array,
    indices: mx.array,
) -> tuple[
    mx.array, mx.array, mx.array, mx.array, mx.array, int, int, int, int, int, int, int
]:
    if q_fp8.ndim != 4:
        raise ValueError(
            f"q_fp8 must have shape (B, S, H, K); got {tuple(q_fp8.shape)}"
        )
    if kv_fp8.ndim != 4:
        raise ValueError(
            f"kv_fp8 must have shape (B, S_kv, G, K); got {tuple(kv_fp8.shape)}"
        )
    if indices.ndim != 4:
        raise ValueError(
            f"indices must have shape (B, S, G, TOPK); got {tuple(indices.shape)}"
        )
    if q_fp8.dtype != mx.uint8 or kv_fp8.dtype != mx.uint8:
        raise ValueError(
            f"q_fp8/kv_fp8 must be mx.uint8 e4m3 storage; got {q_fp8.dtype}, {kv_fp8.dtype}"
        )
    _index_dtype_name(indices, op_name="FP8 Sparse-MLA indexed QK reducer")

    batch, seq_len, heads, k = (int(x) for x in q_fp8.shape)
    kv_batch, seq_len_kv, kv_group, kv_k = (int(x) for x in kv_fp8.shape)
    idx_batch, idx_seq, idx_group, topk = (int(x) for x in indices.shape)
    if (kv_batch, idx_batch) != (batch, batch) or idx_seq != seq_len:
        raise ValueError(
            "q_fp8/kv_fp8/indices batch or sequence mismatch: "
            f"q={tuple(q_fp8.shape)} kv={tuple(kv_fp8.shape)} indices={tuple(indices.shape)}"
        )
    if kv_k != k:
        raise ValueError(
            f"q_fp8/kv_fp8 K mismatch: q={tuple(q_fp8.shape)} kv={tuple(kv_fp8.shape)}"
        )
    if idx_group != kv_group:
        raise ValueError(
            f"indices kv_group mismatch: indices={tuple(indices.shape)} kv={tuple(kv_fp8.shape)}"
        )
    _validate_indexed_reduce_shape(
        batch=batch,
        seq_len=seq_len,
        heads=heads,
        seq_len_kv=seq_len_kv,
        kv_group=kv_group,
        topk=topk,
        K=k,
        outputs_per_block=1,
        reduce_threads=1,
        vec=1,
    )
    if q_scale.dtype != mx.float32 or kv_scale.dtype != mx.float32:
        raise TypeError(
            f"q_scale/kv_scale must be float32; got {q_scale.dtype}, {kv_scale.dtype}"
        )
    if tuple(q_scale.shape) != (batch, seq_len, heads):
        raise ValueError(
            f"q_scale must have shape {(batch, seq_len, heads)}; got {tuple(q_scale.shape)}"
        )
    if tuple(kv_scale.shape) != (batch, seq_len_kv, kv_group):
        raise ValueError(
            f"kv_scale must have shape {(batch, seq_len_kv, kv_group)}; got {tuple(kv_scale.shape)}"
        )
    return (
        q_fp8,
        q_scale,
        kv_fp8,
        kv_scale,
        indices,
        batch,
        seq_len,
        heads,
        seq_len_kv,
        kv_group,
        topk,
        k,
    )


def fp8_sparse_mla_qk_reduce_path_c(
    A_fp8: mx.array,
    A_scale: mx.array,
    B_fp8: mx.array,
    B_scale: mx.array,
    *,
    outputs_per_block: int = _SMFP8_QKR_DEFAULT_OUTPUTS_PER_BLOCK,
    reduce_threads: int = _SMFP8_QKR_DEFAULT_REDUCE_THREADS,
    vec: int = _SMFP8_QKR_DEFAULT_VEC,
) -> mx.array | None:
    """Run the real-shape Path C FP8 Sparse-MLA QK reducer.

    Returns a ``(1, N)`` fp32 score tile, or ``None`` when Metal/TileLang is
    unavailable. Shape/type mismatches raise ``ValueError`` to avoid silent
    fallback in correctness tests. Inputs must already be compact prepared
    buffers; the wrapper does not call ``mx.contiguous`` to repair views.
    """

    if not can_run_metal():
        return None
    (
        A_fp8_u8,
        A_scale_f32,
        B_fp8_u8,
        B_scale_f32,
        n,
        k,
        b_scale_size,
    ) = _normalize_qk_reduce_inputs(A_fp8, A_scale, B_fp8, B_scale)
    outputs_per_block, reduce_threads, vec = _resolve_qk_reduce_schedule(
        N=n,
        K=k,
        outputs_per_block=outputs_per_block,
        reduce_threads=reduce_threads,
        vec=vec,
    )
    try:
        kernel, _lowering, _input_names = _qk_reduce_kernel_for(
            n,
            k,
            outputs_per_block,
            reduce_threads,
            vec,
            b_scale_size,
        )
    except (MSLDispatchUnsupported, SparseMLAFp8PathCDirectError):
        # ACCEPTABLE: explicit "this specific kernel can't take this case"
        # signal (documented feature-absent) -> None feeds documented routing.
        return None
    except Exception as exc:
        # RULE #1: any OTHER compile/lowering crash of the FP8 QK reducer is a
        # real bug, not a "not applicable" signal. Availability is probed by
        # fp8_sparse_mla_qk_reduce_path_c_status; here raise with where+what.
        raise RuntimeError(
            f"fp8_sparse_mla_qk_reduce_path_c: FP8 QK reducer kernel "
            f"compile/lowering failed for n={n} k={k} "
            f"({type(exc).__name__}: {exc})."
        ) from exc

    try:
        returned = kernel(A_fp8_u8, A_scale_f32, B_fp8_u8, B_scale_f32)
    except Exception as exc:
        # RULE #1: a kernel launch/writeback crash is a real bug, not "not
        # applicable". Raise with where+what instead of silently None.
        raise RuntimeError(
            f"fp8_sparse_mla_qk_reduce_path_c: FP8 QK reducer kernel "
            f"launch/writeback failed ({type(exc).__name__}: {exc})."
        ) from exc
    if isinstance(returned, (list, tuple)):
        if len(returned) != 1:
            return None
        return cast(mx.array, returned[0])
    return cast(mx.array, returned)


def fp8_sparse_mla_indexed_qk_reduce_path_c(
    q_fp8: mx.array,
    q_scale: mx.array,
    kv_fp8: mx.array,
    kv_scale: mx.array,
    indices: mx.array,
    *,
    sm_scale: float,
    outputs_per_block: int = _SMFP8_QKR_DEFAULT_OUTPUTS_PER_BLOCK,
    reduce_threads: int = _SMFP8_QKR_DEFAULT_REDUCE_THREADS,
    vec: int = _SMFP8_QKR_DEFAULT_VEC,
) -> mx.array | None:
    """Run indexed full-shape Path C FP8 QK scores.

    Returns ``scores[B, S, H, TOPK]`` in fp32. Invalid indices are written as a
    finite fp32-min sentinel because the current TileLang Metal path cannot
    lower ``T.infinity``. Inputs must already be compact prepared buffers; the
    wrapper does not call ``mx.contiguous`` to repair views.
    """

    if not can_run_metal():
        return None
    (
        q_fp8_u8,
        q_scale_f32,
        kv_fp8_u8,
        kv_scale_f32,
        indices_i32,
        batch,
        seq_len,
        heads,
        seq_len_kv,
        kv_group,
        topk,
        k,
    ) = _normalize_indexed_qk_reduce_inputs(q_fp8, q_scale, kv_fp8, kv_scale, indices)
    outputs_per_block, reduce_threads, vec = _resolve_qk_reduce_schedule(
        N=topk,
        K=k,
        outputs_per_block=outputs_per_block,
        reduce_threads=reduce_threads,
        vec=vec,
    )
    try:
        kernel, _lowering, _input_names = _indexed_qk_reduce_kernel_for(
            batch,
            seq_len,
            heads,
            seq_len_kv,
            kv_group,
            topk,
            k,
            outputs_per_block,
            reduce_threads,
            vec,
            _index_dtype_name(
                indices_i32,
                op_name="FP8 Sparse-MLA indexed QK reducer",
            ),
        )
    except (MSLDispatchUnsupported, SparseMLAFp8PathCDirectError):
        # ACCEPTABLE: explicit "this specific kernel can't take this case"
        # signal (documented feature-absent) -> None feeds documented routing.
        return None
    except Exception as exc:
        # RULE #1: any OTHER compile/lowering crash of the indexed FP8 QK
        # reducer is a real bug, not a "not applicable" signal. Availability is
        # probed by fp8_sparse_mla_indexed_qk_reduce_path_c_status; raise here.
        raise RuntimeError(
            f"fp8_sparse_mla_indexed_qk_reduce_path_c: indexed FP8 QK reducer "
            f"kernel compile/lowering failed for batch={batch} seq_len={seq_len} "
            f"heads={heads} topk={topk} k={k} "
            f"({type(exc).__name__}: {exc})."
        ) from exc

    sm_scale_buf = mx.array([float(sm_scale)], dtype=mx.float32)
    try:
        returned = kernel(
            q_fp8_u8,
            q_scale_f32,
            kv_fp8_u8,
            kv_scale_f32,
            indices_i32,
            sm_scale_buf,
        )
    except Exception as exc:
        # RULE #1: a kernel launch/writeback crash is a real bug, not "not
        # applicable". Raise with where+what instead of silently None.
        raise RuntimeError(
            f"fp8_sparse_mla_indexed_qk_reduce_path_c: indexed FP8 QK reducer "
            f"kernel launch/writeback failed ({type(exc).__name__}: {exc})."
        ) from exc
    if isinstance(returned, (list, tuple)):
        if len(returned) != 1:
            return None
        return cast(mx.array, returned[0])
    return cast(mx.array, returned)


def _threads_for_topk(topk: int) -> int:
    threads = min(64, max(1, int(topk)))
    power = 1
    while (power << 1) <= threads:
        power <<= 1
    return max(1, power)


def _validate_fp8_apply_inputs(
    q_fp8: mx.array,
    q_scale: mx.array,
    kv_fp8: mx.array,
    kv_scale: mx.array,
    indices: mx.array,
    *,
    d_v: int | None,
) -> tuple[int, int, int, int, int, int, int, int, int, int]:
    if q_fp8.ndim != 4:
        raise ValueError(
            f"q_fp8 must have shape (B, S, H, K); got {tuple(q_fp8.shape)}"
        )
    if kv_fp8.ndim != 4:
        raise ValueError(
            f"kv_fp8 must have shape (B, S_kv, G, K); got {tuple(kv_fp8.shape)}"
        )
    if indices.ndim != 4:
        raise ValueError(
            f"indices must have shape (B, S, G, TOPK); got {tuple(indices.shape)}"
        )
    if q_fp8.dtype != mx.uint8 or kv_fp8.dtype != mx.uint8:
        raise TypeError(
            f"q_fp8/kv_fp8 must be uint8 FP8 storage; got {q_fp8.dtype}, {kv_fp8.dtype}"
        )
    if q_scale.dtype != mx.float32 or kv_scale.dtype != mx.float32:
        raise TypeError(
            f"q_scale/kv_scale must be float32; got {q_scale.dtype}, {kv_scale.dtype}"
        )
    _index_dtype_name(indices, op_name="FP8 Sparse-MLA Path C apply")

    batch, seq_len, heads, qk_dim = (int(x) for x in q_fp8.shape)
    kv_batch, seq_len_kv, kv_group, kv_dim = (int(x) for x in kv_fp8.shape)
    idx_batch, idx_seq, idx_group, topk = (int(x) for x in indices.shape)
    if kv_batch != batch or idx_batch != batch or idx_seq != seq_len:
        raise ValueError(
            "q_fp8/kv_fp8/indices batch or sequence mismatch: "
            f"q={tuple(q_fp8.shape)} kv={tuple(kv_fp8.shape)} indices={tuple(indices.shape)}"
        )
    if kv_dim != qk_dim:
        raise ValueError(
            f"q_fp8/kv_fp8 K mismatch: q={tuple(q_fp8.shape)} kv={tuple(kv_fp8.shape)}"
        )
    if idx_group != kv_group:
        raise ValueError(
            f"indices kv_group mismatch: indices={tuple(indices.shape)} kv={tuple(kv_fp8.shape)}"
        )
    if heads % kv_group != 0:
        raise ValueError(f"heads {heads} must be divisible by kv_group {kv_group}")
    if tuple(q_scale.shape) != (batch, seq_len, heads):
        raise ValueError(
            f"q_scale must have shape {(batch, seq_len, heads)}; got {tuple(q_scale.shape)}"
        )
    if tuple(kv_scale.shape) != (batch, seq_len_kv, kv_group):
        raise ValueError(
            f"kv_scale must have shape {(batch, seq_len_kv, kv_group)}; got {tuple(kv_scale.shape)}"
        )
    d_v_resolved = qk_dim if d_v is None else int(d_v)
    if d_v_resolved <= 0 or d_v_resolved > qk_dim:
        raise ValueError(f"d_v must be in (0, {qk_dim}], got {d_v_resolved}")
    return (
        batch,
        seq_len,
        heads,
        seq_len_kv,
        kv_group,
        heads // kv_group,
        topk,
        qk_dim,
        d_v_resolved,
        _threads_for_topk(topk),
    )


def _make_fp8_sparse_mla_apply_kernel(
    *,
    batch: int,
    seq_len: int,
    heads: int,
    seq_len_kv: int,
    kv_group: int,
    head_kv: int,
    topk: int,
    K: int,
    d_v: int,
    threads: int,
    out_dtype: str,
    index_dtype: str = "int32",
) -> Any:
    import tilelang.language as T
    from tilelang.tileop.metal_quant import fp8_e4m3fn_to_float

    T = cast(Any, T)

    lanes = batch * seq_len * heads
    g = globals()
    g.update(
        _SMFP8_APPLY_B=batch,
        _SMFP8_APPLY_S=seq_len,
        _SMFP8_APPLY_H=heads,
        _SMFP8_APPLY_SKV=seq_len_kv,
        _SMFP8_APPLY_G=kv_group,
        _SMFP8_APPLY_HEAD_KV=head_kv,
        _SMFP8_APPLY_TOPK=topk,
        _SMFP8_APPLY_K=K,
        _SMFP8_APPLY_DV=d_v,
        _SMFP8_APPLY_THREADS=threads,
        _SMFP8_APPLY_LOG_THREADS=threads.bit_length() - 1,
        _SMFP8_APPLY_LANES=lanes,
        _SMFP8_APPLY_Q_SIZE=lanes * K,
        _SMFP8_APPLY_KV_SIZE=batch * seq_len_kv * kv_group * K,
        _SMFP8_APPLY_Q_SCALE_SIZE=lanes,
        _SMFP8_APPLY_KV_SCALE_SIZE=batch * seq_len_kv * kv_group,
        _SMFP8_APPLY_IDX_SIZE=batch * seq_len * kv_group * topk,
        _SMFP8_APPLY_OUT_SIZE=lanes * d_v,
        _SMFP8_APPLY_LSE_SIZE=lanes,
        _SMFP8_APPLY_OUT_DTYPE=out_dtype,
        _SMFP8_APPLY_INDEX_DTYPE=index_dtype,
    )

    @T.prim_func
    def fp8_sparse_mla_apply_kernel(
        q_fp8: T.Tensor((_SMFP8_APPLY_Q_SIZE,), "uint8"),
        q_scale: T.Tensor((_SMFP8_APPLY_Q_SCALE_SIZE,), "float32"),
        kv_fp8: T.Tensor((_SMFP8_APPLY_KV_SIZE,), "uint8"),
        kv_scale: T.Tensor((_SMFP8_APPLY_KV_SCALE_SIZE,), "float32"),
        indices: T.Tensor((_SMFP8_APPLY_IDX_SIZE,), _SMFP8_APPLY_INDEX_DTYPE),
        sm_scale_buf: T.Tensor((1,), "float32"),
        sinks: T.Tensor((_SMFP8_APPLY_H,), "float32"),
        has_sinks: T.Tensor((1,), "int32"),
        out: T.Tensor((_SMFP8_APPLY_OUT_SIZE,), _SMFP8_APPLY_OUT_DTYPE),
        lse: T.Tensor((_SMFP8_APPLY_LSE_SIZE,), "float32"),
    ):
        with T.Kernel(_SMFP8_APPLY_LANES, threads=_SMFP8_APPLY_THREADS) as bx:
            lane = T.get_thread_binding()
            scores = T.alloc_shared((_SMFP8_APPLY_TOPK,), "float32", scope="shared")
            reduce_buf = T.alloc_shared(
                (_SMFP8_APPLY_THREADS,), "float32", scope="shared"
            )
            acc = T.alloc_local((1,), "float32")
            local = T.alloc_local((1,), "float32")
            inv_sum = T.alloc_local((1,), "float32")
            stride = T.alloc_local((1,), "int32")
            gather_idx = T.alloc_local((1,), "int32")

            h = bx % _SMFP8_APPLY_H
            b = bx // (_SMFP8_APPLY_H * _SMFP8_APPLY_S)
            gidx = h // _SMFP8_APPLY_HEAD_KV
            q_row_base = bx * _SMFP8_APPLY_K
            q_scale_idx = bx
            kv_b_base = b * (_SMFP8_APPLY_SKV * _SMFP8_APPLY_G * _SMFP8_APPLY_K)
            kv_scale_b_base = b * (_SMFP8_APPLY_SKV * _SMFP8_APPLY_G)
            idx_base = (
                (bx // _SMFP8_APPLY_H) * _SMFP8_APPLY_G + gidx
            ) * _SMFP8_APPLY_TOPK
            out_row = bx * _SMFP8_APPLY_DV
            sm_scale = sm_scale_buf[0]

            for k_top in T.serial(lane, _SMFP8_APPLY_TOPK, step=_SMFP8_APPLY_THREADS):
                gather_idx[0] = T.cast(indices[idx_base + k_top], "int32")
                if gather_idx[0] < 0 or gather_idx[0] >= _SMFP8_APPLY_SKV:
                    scores[k_top] = T.float32(_SMFP8_INVALID_SCORE_SENTINEL)
                else:
                    acc[0] = 0.0
                    kv_row_base = (
                        kv_b_base
                        + (gather_idx[0] * _SMFP8_APPLY_G + gidx) * _SMFP8_APPLY_K
                    )
                    kv_scale_idx = (
                        kv_scale_b_base + gather_idx[0] * _SMFP8_APPLY_G + gidx
                    )
                    for d in T.serial(_SMFP8_APPLY_K):
                        acc[0] = acc[0] + fp8_e4m3fn_to_float(
                            q_fp8[q_row_base + d]
                        ) * fp8_e4m3fn_to_float(kv_fp8[kv_row_base + d])
                    scores[k_top] = (
                        acc[0]
                        * q_scale[q_scale_idx]
                        * kv_scale[kv_scale_idx]
                        * sm_scale
                    )
            T.sync_threads()

            local[0] = T.float32(_SMFP8_INVALID_SCORE_SENTINEL)
            for k_top in T.serial(lane, _SMFP8_APPLY_TOPK, step=_SMFP8_APPLY_THREADS):
                if scores[k_top] > local[0]:
                    local[0] = scores[k_top]
            reduce_buf[lane] = local[0]
            T.sync_threads()
            for round_id in T.serial(_SMFP8_APPLY_LOG_THREADS):
                stride[0] = T.shift_right(_SMFP8_APPLY_THREADS, round_id + 1)
                if lane < stride[0]:
                    if reduce_buf[lane + stride[0]] > reduce_buf[lane]:
                        reduce_buf[lane] = reduce_buf[lane + stride[0]]
                T.sync_threads()
            local[0] = reduce_buf[0]
            if has_sinks[0] != 0:
                if sinks[h] > local[0]:
                    local[0] = sinks[h]
            row_max = local[0]

            for k_top in T.serial(lane, _SMFP8_APPLY_TOPK, step=_SMFP8_APPLY_THREADS):
                if scores[k_top] == T.float32(_SMFP8_INVALID_SCORE_SENTINEL):
                    scores[k_top] = 0.0
                else:
                    scores[k_top] = T.exp(scores[k_top] - row_max)
            T.sync_threads()

            local[0] = 0.0
            for k_top in T.serial(lane, _SMFP8_APPLY_TOPK, step=_SMFP8_APPLY_THREADS):
                local[0] = local[0] + scores[k_top]
            reduce_buf[lane] = local[0]
            T.sync_threads()
            for round_id in T.serial(_SMFP8_APPLY_LOG_THREADS):
                stride[0] = T.shift_right(_SMFP8_APPLY_THREADS, round_id + 1)
                if lane < stride[0]:
                    reduce_buf[lane] = reduce_buf[lane] + reduce_buf[lane + stride[0]]
                T.sync_threads()
            local[0] = reduce_buf[0]
            if has_sinks[0] != 0:
                local[0] = local[0] + T.exp(sinks[h] - row_max)

            inv_sum[0] = 0.0
            if local[0] > 0.0:
                inv_sum[0] = 1.0 / local[0]

            for d in T.serial(lane, _SMFP8_APPLY_DV, step=_SMFP8_APPLY_THREADS):
                acc[0] = 0.0
                for k_top in T.serial(_SMFP8_APPLY_TOPK):
                    gather_idx[0] = T.cast(indices[idx_base + k_top], "int32")
                    if gather_idx[0] >= 0 and gather_idx[0] < _SMFP8_APPLY_SKV:
                        kv_row_base = (
                            kv_b_base
                            + (gather_idx[0] * _SMFP8_APPLY_G + gidx) * _SMFP8_APPLY_K
                        )
                        kv_scale_idx = (
                            kv_scale_b_base + gather_idx[0] * _SMFP8_APPLY_G + gidx
                        )
                        acc[0] = (
                            acc[0]
                            + scores[k_top]
                            * fp8_e4m3fn_to_float(kv_fp8[kv_row_base + d])
                            * kv_scale[kv_scale_idx]
                        )
                out[out_row + d] = T.cast(
                    acc[0] * inv_sum[0], _SMFP8_APPLY_OUT_DTYPE
                )

            if lane == 0:
                if local[0] > 0.0:
                    lse[bx] = row_max + T.log(local[0])
                else:
                    lse[bx] = 0.0

    try:
        from tilelang.transform.simplify import apply_simplify

        return apply_simplify(fp8_sparse_mla_apply_kernel)
    except (ImportError, ModuleNotFoundError):
        # ACCEPTABLE: simplify is a correctness-preserving optimization; if the
        # pass is absent the unsimplified kernel is functionally identical.
        # RULE #1: a real crash inside the pass now propagates (not swallowed).
        return fp8_sparse_mla_apply_kernel


def _make_fp8_sparse_mla_bwd_kernel(
    *,
    batch: int,
    seq_len: int,
    heads: int,
    seq_len_kv: int,
    kv_group: int,
    head_kv: int,
    topk: int,
    K: int,
    d_v: int,
    threads: int,
    d_out_dtype: str,
    index_dtype: str = "int32",
) -> Any:
    import tilelang.language as T
    from tilelang.tileop.metal_quant import fp8_e4m3fn_to_float

    T = cast(Any, T)

    lanes = batch * seq_len * heads
    g = globals()
    g.update(
        _SMFP8_BWD_B=batch,
        _SMFP8_BWD_S=seq_len,
        _SMFP8_BWD_H=heads,
        _SMFP8_BWD_SKV=seq_len_kv,
        _SMFP8_BWD_G=kv_group,
        _SMFP8_BWD_HEAD_KV=head_kv,
        _SMFP8_BWD_TOPK=topk,
        _SMFP8_BWD_K=K,
        _SMFP8_BWD_DV=d_v,
        _SMFP8_BWD_THREADS=threads,
        _SMFP8_BWD_LOG_THREADS=threads.bit_length() - 1,
        _SMFP8_BWD_LANES=lanes,
        _SMFP8_BWD_Q_SIZE=lanes * K,
        _SMFP8_BWD_KV_SIZE=batch * seq_len_kv * kv_group * K,
        _SMFP8_BWD_Q_SCALE_SIZE=lanes,
        _SMFP8_BWD_KV_SCALE_SIZE=batch * seq_len_kv * kv_group,
        _SMFP8_BWD_IDX_SIZE=batch * seq_len * kv_group * topk,
        _SMFP8_BWD_DOUT_SIZE=lanes * d_v,
        _SMFP8_BWD_DOUT_DTYPE=d_out_dtype,
        _SMFP8_BWD_CLEAR_TOTAL=batch * seq_len_kv * kv_group * K,
        _SMFP8_BWD_INDEX_DTYPE=index_dtype,
    )

    @T.prim_func
    def fp8_sparse_mla_bwd_kernel(
        q_fp8: T.Tensor((_SMFP8_BWD_Q_SIZE,), "uint8"),
        q_scale: T.Tensor((_SMFP8_BWD_Q_SCALE_SIZE,), "float32"),
        kv_fp8: T.Tensor((_SMFP8_BWD_KV_SIZE,), "uint8"),
        kv_scale: T.Tensor((_SMFP8_BWD_KV_SCALE_SIZE,), "float32"),
        d_out: T.Tensor((_SMFP8_BWD_DOUT_SIZE,), _SMFP8_BWD_DOUT_DTYPE),
        indices: T.Tensor((_SMFP8_BWD_IDX_SIZE,), _SMFP8_BWD_INDEX_DTYPE),
        sm_scale_buf: T.Tensor((1,), "float32"),
        dq_dequant: T.Tensor((_SMFP8_BWD_Q_SIZE,), "float32"),
        dkv_dequant: T.Tensor((_SMFP8_BWD_KV_SIZE,), "float32"),
    ):
        with T.Kernel(_SMFP8_BWD_LANES, threads=_SMFP8_BWD_THREADS) as bx:
            lane = T.get_thread_binding()
            scores = T.alloc_shared((_SMFP8_BWD_TOPK,), "float32", scope="shared")
            p = T.alloc_shared((_SMFP8_BWD_TOPK,), "float32", scope="shared")
            dp = T.alloc_shared((_SMFP8_BWD_TOPK,), "float32", scope="shared")
            ds = T.alloc_shared((_SMFP8_BWD_TOPK,), "float32", scope="shared")
            reduce_buf = T.alloc_shared(
                (_SMFP8_BWD_THREADS,), "float32", scope="shared"
            )
            acc = T.alloc_local((1,), "float32")
            local = T.alloc_local((1,), "float32")
            inv_sum = T.alloc_local((1,), "float32")
            stride = T.alloc_local((1,), "int32")
            gather_idx = T.alloc_local((1,), "int32")
            kv_row_base_local = T.alloc_local((1,), "int32")
            kv_scale_idx_local = T.alloc_local((1,), "int32")
            dkv_idx_local = T.alloc_local((1,), "int32")

            h = bx % _SMFP8_BWD_H
            b = bx // (_SMFP8_BWD_H * _SMFP8_BWD_S)
            gidx = h // _SMFP8_BWD_HEAD_KV
            q_row_base = bx * _SMFP8_BWD_K
            q_scale_idx = bx
            d_out_row = bx * _SMFP8_BWD_DV
            kv_b_base = b * (_SMFP8_BWD_SKV * _SMFP8_BWD_G * _SMFP8_BWD_K)
            kv_scale_b_base = b * (_SMFP8_BWD_SKV * _SMFP8_BWD_G)
            idx_base = ((bx // _SMFP8_BWD_H) * _SMFP8_BWD_G + gidx) * _SMFP8_BWD_TOPK
            sm_scale = sm_scale_buf[0]

            for k_top in T.serial(lane, _SMFP8_BWD_TOPK, step=_SMFP8_BWD_THREADS):
                gather_idx[0] = T.cast(indices[idx_base + k_top], "int32")
                if gather_idx[0] < 0 or gather_idx[0] >= _SMFP8_BWD_SKV:
                    scores[k_top] = T.float32(_SMFP8_INVALID_SCORE_SENTINEL)
                else:
                    acc[0] = 0.0
                    kv_row_base_local[0] = (
                        kv_b_base + (gather_idx[0] * _SMFP8_BWD_G + gidx) * _SMFP8_BWD_K
                    )
                    kv_scale_idx_local[0] = (
                        kv_scale_b_base + gather_idx[0] * _SMFP8_BWD_G + gidx
                    )
                    for d in T.serial(_SMFP8_BWD_K):
                        acc[0] = acc[0] + fp8_e4m3fn_to_float(
                            q_fp8[q_row_base + d]
                        ) * fp8_e4m3fn_to_float(kv_fp8[kv_row_base_local[0] + d])
                    scores[k_top] = (
                        acc[0]
                        * q_scale[q_scale_idx]
                        * kv_scale[kv_scale_idx_local[0]]
                        * sm_scale
                    )
            T.sync_threads()

            local[0] = T.float32(_SMFP8_INVALID_SCORE_SENTINEL)
            for k_top in T.serial(lane, _SMFP8_BWD_TOPK, step=_SMFP8_BWD_THREADS):
                if scores[k_top] > local[0]:
                    local[0] = scores[k_top]
            reduce_buf[lane] = local[0]
            T.sync_threads()
            for round_id in T.serial(_SMFP8_BWD_LOG_THREADS):
                stride[0] = T.shift_right(_SMFP8_BWD_THREADS, round_id + 1)
                if lane < stride[0]:
                    if reduce_buf[lane + stride[0]] > reduce_buf[lane]:
                        reduce_buf[lane] = reduce_buf[lane + stride[0]]
                T.sync_threads()
            local[0] = reduce_buf[0]
            row_max = local[0]

            for k_top in T.serial(lane, _SMFP8_BWD_TOPK, step=_SMFP8_BWD_THREADS):
                if scores[k_top] == T.float32(_SMFP8_INVALID_SCORE_SENTINEL):
                    p[k_top] = 0.0
                else:
                    p[k_top] = T.exp(scores[k_top] - row_max)
            T.sync_threads()

            local[0] = 0.0
            for k_top in T.serial(lane, _SMFP8_BWD_TOPK, step=_SMFP8_BWD_THREADS):
                local[0] = local[0] + p[k_top]
            reduce_buf[lane] = local[0]
            T.sync_threads()
            for round_id in T.serial(_SMFP8_BWD_LOG_THREADS):
                stride[0] = T.shift_right(_SMFP8_BWD_THREADS, round_id + 1)
                if lane < stride[0]:
                    reduce_buf[lane] = reduce_buf[lane] + reduce_buf[lane + stride[0]]
                T.sync_threads()
            local[0] = reduce_buf[0]
            inv_sum[0] = 0.0
            if local[0] > 0.0:
                inv_sum[0] = 1.0 / local[0]

            for k_top in T.serial(lane, _SMFP8_BWD_TOPK, step=_SMFP8_BWD_THREADS):
                p[k_top] = p[k_top] * inv_sum[0]
            T.sync_threads()

            for k_top in T.serial(lane, _SMFP8_BWD_TOPK, step=_SMFP8_BWD_THREADS):
                gather_idx[0] = T.cast(indices[idx_base + k_top], "int32")
                if gather_idx[0] < 0 or gather_idx[0] >= _SMFP8_BWD_SKV:
                    dp[k_top] = 0.0
                else:
                    acc[0] = 0.0
                    kv_row_base_local[0] = (
                        kv_b_base + (gather_idx[0] * _SMFP8_BWD_G + gidx) * _SMFP8_BWD_K
                    )
                    kv_scale_idx_local[0] = (
                        kv_scale_b_base + gather_idx[0] * _SMFP8_BWD_G + gidx
                    )
                    for d in T.serial(_SMFP8_BWD_DV):
                        acc[0] = acc[0] + fp8_e4m3fn_to_float(
                            kv_fp8[kv_row_base_local[0] + d]
                        ) * kv_scale[kv_scale_idx_local[0]] * T.cast(
                            d_out[d_out_row + d], "float32"
                        )
                    dp[k_top] = acc[0]
            T.sync_threads()

            local[0] = 0.0
            for k_top in T.serial(lane, _SMFP8_BWD_TOPK, step=_SMFP8_BWD_THREADS):
                local[0] = local[0] + p[k_top] * dp[k_top]
            reduce_buf[lane] = local[0]
            T.sync_threads()
            for round_id in T.serial(_SMFP8_BWD_LOG_THREADS):
                stride[0] = T.shift_right(_SMFP8_BWD_THREADS, round_id + 1)
                if lane < stride[0]:
                    reduce_buf[lane] = reduce_buf[lane] + reduce_buf[lane + stride[0]]
                T.sync_threads()
            local[0] = reduce_buf[0]

            for k_top in T.serial(lane, _SMFP8_BWD_TOPK, step=_SMFP8_BWD_THREADS):
                ds[k_top] = p[k_top] * (dp[k_top] - local[0])
            T.sync_threads()

            for d in T.serial(lane, _SMFP8_BWD_K, step=_SMFP8_BWD_THREADS):
                acc[0] = 0.0
                for k_top in T.serial(_SMFP8_BWD_TOPK):
                    gather_idx[0] = T.cast(indices[idx_base + k_top], "int32")
                    if gather_idx[0] >= 0 and gather_idx[0] < _SMFP8_BWD_SKV:
                        kv_row_base_local[0] = (
                            kv_b_base
                            + (gather_idx[0] * _SMFP8_BWD_G + gidx) * _SMFP8_BWD_K
                        )
                        kv_scale_idx_local[0] = (
                            kv_scale_b_base + gather_idx[0] * _SMFP8_BWD_G + gidx
                        )
                        acc[0] = (
                            acc[0]
                            + ds[k_top]
                            * fp8_e4m3fn_to_float(kv_fp8[kv_row_base_local[0] + d])
                            * kv_scale[kv_scale_idx_local[0]]
                        )
                dq_dequant[q_row_base + d] = acc[0] * sm_scale

            for kd in T.serial(
                lane,
                _SMFP8_BWD_TOPK * _SMFP8_BWD_K,
                step=_SMFP8_BWD_THREADS,
            ):
                k_top = kd // _SMFP8_BWD_K
                d = kd % _SMFP8_BWD_K
                gather_idx[0] = T.cast(indices[idx_base + k_top], "int32")
                if gather_idx[0] >= 0 and gather_idx[0] < _SMFP8_BWD_SKV:
                    qv = (
                        fp8_e4m3fn_to_float(q_fp8[q_row_base + d])
                        * q_scale[q_scale_idx]
                    )
                    acc[0] = sm_scale * ds[k_top] * qv
                    if d < _SMFP8_BWD_DV:
                        acc[0] = (
                            acc[0]
                            + p[k_top] * T.cast(d_out[d_out_row + d], "float32")
                        )
                    dkv_idx_local[0] = (
                        (
                            b * _SMFP8_BWD_SKV * _SMFP8_BWD_G
                            + gather_idx[0] * _SMFP8_BWD_G
                            + gidx
                        )
                        * _SMFP8_BWD_K
                        + d
                    )
                    T.atomic_add(
                        dkv_dequant[dkv_idx_local[0]],
                        acc[0],
                        memory_order="relaxed",
                    )

    try:
        from tilelang.transform.simplify import apply_simplify

        return apply_simplify(fp8_sparse_mla_bwd_kernel)
    except (ImportError, ModuleNotFoundError):
        # ACCEPTABLE: simplify is a correctness-preserving optimization; if the
        # pass is absent the unsimplified kernel is functionally identical.
        # RULE #1: a real crash inside the pass now propagates (not swallowed).
        return fp8_sparse_mla_bwd_kernel


def _make_fp8_bwd_clear_dkv_kernel(
    *,
    batch: int,
    seq_len_kv: int,
    kv_group: int,
    K: int,
    threads: int,
) -> Any:
    import tilelang.language as T

    T = cast(Any, T)
    total = int(batch) * int(seq_len_kv) * int(kv_group) * int(K)
    g = globals()
    g.update(
        _SMFP8_BWD_CLEAR_TOTAL=total,
        _SMFP8_BWD_CLEAR_THREADS=int(threads),
    )

    @T.prim_func
    def fp8_sparse_mla_bwd_clear_dkv_kernel(
        dkv_dequant: T.Tensor((_SMFP8_BWD_CLEAR_TOTAL,), "float32"),
    ):
        with T.Kernel(
            T.ceildiv(_SMFP8_BWD_CLEAR_TOTAL, _SMFP8_BWD_CLEAR_THREADS),
            threads=_SMFP8_BWD_CLEAR_THREADS,
        ) as bx:
            lane = T.get_thread_binding()
            elem = bx * _SMFP8_BWD_CLEAR_THREADS + lane
            if elem < _SMFP8_BWD_CLEAR_TOTAL:
                dkv_dequant[elem] = 0.0

    try:
        from tilelang.transform.simplify import apply_simplify

        return apply_simplify(fp8_sparse_mla_bwd_clear_dkv_kernel)
    except (ImportError, ModuleNotFoundError):
        # ACCEPTABLE: simplify is a correctness-preserving optimization; if the
        # pass is absent the unsimplified kernel is functionally identical.
        # RULE #1: a real crash inside the pass now propagates (not swallowed).
        return fp8_sparse_mla_bwd_clear_dkv_kernel


@lru_cache(maxsize=128)
def _fp8_apply_kernel_for(
    batch: int,
    seq_len: int,
    heads: int,
    seq_len_kv: int,
    kv_group: int,
    head_kv: int,
    topk: int,
    K: int,
    d_v: int,
    threads: int,
    out_dtype: str,
) -> tuple[Any, _msl_transform.TileLangMSLLowering, list[str]]:
    del (
        batch,
        seq_len,
        heads,
        seq_len_kv,
        kv_group,
        head_kv,
        topk,
        K,
        d_v,
        threads,
        out_dtype,
    )
    raise MSLDispatchUnsupported(
        "legacy sparse_mla_fp8 Path C no-owner MLX fast wrapper is disabled; "
        "use sparse_mla_fp8_path_c_apply_direct with caller-owned outputs or "
        "dispatch the compiled TileLang tvm-ffi kernel through graph outputs"
    )


@lru_cache(maxsize=128)
def _fp8_apply_tvm_ffi_kernel_for(
    batch: int,
    seq_len: int,
    heads: int,
    seq_len_kv: int,
    kv_group: int,
    head_kv: int,
    topk: int,
    K: int,
    d_v: int,
    threads: int,
    out_dtype: str,
    index_dtype: str,
) -> Any:
    """Compile the FP8 prepared forward kernel for caller-owned outputs."""

    import tilelang

    prim = _make_fp8_sparse_mla_apply_kernel(
        batch=batch,
        seq_len=seq_len,
        heads=heads,
        seq_len_kv=seq_len_kv,
        kv_group=kv_group,
        head_kv=head_kv,
        topk=topk,
        K=K,
        d_v=d_v,
        threads=threads,
        out_dtype=out_dtype,
        index_dtype=index_dtype,
    )
    return tilelang.compile(
        prim,
        target=_msl_transform._as_metal_target(TILELANG_METAL_FP8_SPARSE_MLA_TARGET),
        execution_backend="tvm_ffi",
        out_idx=[8, 9],
    )


@lru_cache(maxsize=128)
def _fp8_bwd_tvm_ffi_kernel_for(
    batch: int,
    seq_len: int,
    heads: int,
    seq_len_kv: int,
    kv_group: int,
    head_kv: int,
    topk: int,
    K: int,
    d_v: int,
    threads: int,
    d_out_dtype: str,
    index_dtype: str,
) -> Any:
    """Compile the FP8 prepared backward kernel for caller-owned outputs."""

    import tilelang

    prim = _make_fp8_sparse_mla_bwd_kernel(
        batch=batch,
        seq_len=seq_len,
        heads=heads,
        seq_len_kv=seq_len_kv,
        kv_group=kv_group,
        head_kv=head_kv,
        topk=topk,
        K=K,
        d_v=d_v,
        threads=threads,
        d_out_dtype=d_out_dtype,
        index_dtype=index_dtype,
    )
    return tilelang.compile(
        prim,
        target=_msl_transform._as_metal_target(TILELANG_METAL_FP8_SPARSE_MLA_TARGET),
        execution_backend="tvm_ffi",
        out_idx=[7, 8],
    )


@lru_cache(maxsize=128)
def _fp8_bwd_clear_dkv_tvm_ffi_kernel_for(
    batch: int,
    seq_len_kv: int,
    kv_group: int,
    K: int,
    threads: int = _SMFP8_BWD_CLEAR_THREADS,
) -> Any:
    """Compile the owner-output dKV clear kernel used before atomic scatter."""

    import tilelang

    prim = _make_fp8_bwd_clear_dkv_kernel(
        batch=batch,
        seq_len_kv=seq_len_kv,
        kv_group=kv_group,
        K=K,
        threads=threads,
    )
    return tilelang.compile(
        prim,
        target=_msl_transform._as_metal_target(TILELANG_METAL_FP8_SPARSE_MLA_TARGET),
        execution_backend="tvm_ffi",
        out_idx=0,
    )


def _validate_fp8_bwd_owner_outputs(
    dq_buffer: mx.array | None,
    dkv_buffer: mx.array | None,
    *,
    dq_shape: tuple[int, int, int, int],
    dkv_shape: tuple[int, int, int, int],
) -> tuple[mx.array, mx.array]:
    if dq_buffer is None or dkv_buffer is None:
        raise SparseMLAFp8PathCDirectError(
            "sparse_mla_fp8_bwd_path_c requires caller-owned dq_buffer and "
            "dkv_buffer; no-owner backward would allocate large gradient "
            "outputs and is fail-closed"
        )
    if not isinstance(dq_buffer, mx.array):
        raise TypeError(
            f"dq_buffer must be an mlx.core.array; got {type(dq_buffer).__name__}"
        )
    if not isinstance(dkv_buffer, mx.array):
        raise TypeError(
            f"dkv_buffer must be an mlx.core.array; got {type(dkv_buffer).__name__}"
        )
    if tuple(dq_buffer.shape) != dq_shape or dq_buffer.dtype != mx.float32:
        raise ValueError(
            "dq_buffer must be the final float32 Path C gradient buffer "
            f"with shape {dq_shape}; got shape {tuple(dq_buffer.shape)} "
            f"and dtype {dq_buffer.dtype}"
        )
    if tuple(dkv_buffer.shape) != dkv_shape or dkv_buffer.dtype != mx.float32:
        raise ValueError(
            "dkv_buffer must be the final float32 Path C gradient buffer "
            f"with shape {dkv_shape}; got shape {tuple(dkv_buffer.shape)} "
            f"and dtype {dkv_buffer.dtype}"
        )
    return dq_buffer, dkv_buffer


def _owner_output_tuple(
    value: object,
    *,
    expected: tuple[mx.array, ...],
    op_name: str,
) -> tuple[mx.array, ...]:
    if len(expected) == 1 and value is expected[0]:
        return expected
    if isinstance(value, (list, tuple)) and len(value) == len(expected):
        if all(got is want for got, want in zip(value, expected, strict=True)):
            return expected
    raise SparseMLAFp8PathCDirectError(
        f"{op_name} did not return caller-owned outputs"
    )


def _owner_output_graph_tuple(
    value: object,
    *,
    expected: tuple[mx.array, ...],
    op_name: str,
) -> tuple[mx.array, ...]:
    if len(expected) == 1 and isinstance(value, mx.array):
        out = (value,)
    elif isinstance(value, (list, tuple)) and len(value) == len(expected):
        out = tuple(cast(mx.array, item) for item in value)
    else:
        raise SparseMLAFp8PathCDirectError(
            f"{op_name} did not return {len(expected)} graph outputs"
        )
    for got, want in zip(out, expected, strict=True):
        if not isinstance(got, mx.array):
            raise SparseMLAFp8PathCDirectError(
                f"{op_name} returned non-array graph output"
            )
        if tuple(got.shape) != tuple(want.shape) or got.dtype != want.dtype:
            raise SparseMLAFp8PathCDirectError(
                f"{op_name} returned graph output with shape/dtype "
                f"{tuple(got.shape)}/{got.dtype}, expected "
                f"{tuple(want.shape)}/{want.dtype}"
            )
    return out


def _flat_1d_view(array: mx.array) -> mx.array:
    # reshape preserves the non-contiguity of prepared FP8 inputs
    # (q_fp8/kv_fp8 produced by per-token/tensor scaling), and tvm-ffi DLPack
    # rejects non-contiguous tensors.  mx.contiguous is a no-op when the array
    # is already contiguous and preserves dtype, so the 1-D view is always a
    # contiguous DLPack-exportable buffer.
    return mx.contiguous(array.reshape((int(array.size),)))


def _clear_fp8_bwd_dkv_buffer(dkv_buffer: mx.array) -> mx.array:
    total = int(dkv_buffer.size)
    if total <= 0:
        return dkv_buffer
    shape = tuple(int(dim) for dim in dkv_buffer.shape)
    if len(shape) != 4:
        raise SparseMLAFp8PathCDirectError(
            "direct tvm-ffi FP8 Sparse-MLA backward dKV clear expected a "
            f"4D dKV buffer, got shape {shape}"
        )
    kernel = _fp8_bwd_clear_dkv_tvm_ffi_kernel_for(
        shape[0],
        shape[1],
        shape[2],
        shape[3],
        min(_SMFP8_BWD_CLEAR_THREADS, max(1, total)),
    )
    flat = _flat_1d_view(dkv_buffer)
    returned = kernel(out=flat)
    # The clear kernel is an internal in-place side effect.  The public
    # backward kernel below still validates strict caller-owned outputs, but
    # MLX may hand back a fresh array wrapper for this single-output clear even
    # when it writes the supplied owner buffer.
    (cleared_flat,) = _owner_output_graph_tuple(
        returned,
        expected=(flat,),
        op_name="direct tvm-ffi FP8 Sparse-MLA backward dKV clear",
    )
    return cleared_flat.reshape(shape)


def _dispatch_fp8_bwd_owner_output_path_c(
    *,
    q_fp8: mx.array,
    q_scale: mx.array,
    kv_fp8: mx.array,
    kv_scale: mx.array,
    d_out: mx.array,
    indices: mx.array,
    sm_scale_buf: mx.array,
    batch: int,
    seq_len: int,
    heads: int,
    seq_len_kv: int,
    kv_group: int,
    head_kv: int,
    topk: int,
    K: int,
    d_v: int,
    threads: int,
    d_out_dtype: str,
    dq_buffer: mx.array | None,
    dkv_buffer: mx.array | None,
    force_path_c: bool,
) -> tuple[mx.array, mx.array] | None:
    """Dispatch FP8 backward through tvm-ffi into caller-owned outputs."""

    dq_shape = (batch, seq_len, heads, K)
    dkv_shape = (batch, seq_len_kv, kv_group, K)
    index_dtype = _index_dtype_name(indices, op_name="FP8 Sparse-MLA Path C backward")
    dkv_needs_clear = True
    graph_output_route = False
    if dq_buffer is None and dkv_buffer is None:
        # GRAPH-OUTPUT route: no caller-owned buffers were supplied.  Allocating
        # the dkv output via tvm-ffi out_idx hands back uninitialized (NaN-poison)
        # memory, and the bwd kernel's T.atomic_add scatters *accumulating* into
        # dkv -- so the buffer MUST be pre-zeroed.  Allocate the gradient buffers
        # here (mx.zeros, native graph outputs) and route through the same
        # owner-output dispatch with a zeroed dkv, making the graph route
        # self-clearing regardless of allocator state.
        graph_output_route = True
        dq_owner = mx.zeros(dq_shape, dtype=mx.float32)
        dkv_owner = mx.zeros(dkv_shape, dtype=mx.float32)
        dkv_needs_clear = False
    elif (dq_buffer is None) != (dkv_buffer is None):
        raise SparseMLAFp8PathCDirectError(
            "sparse_mla_fp8_bwd_path_c owner-output route requires both "
            "dq_buffer and dkv_buffer"
        )
    else:
        try:
            dq_owner, dkv_owner = _validate_fp8_bwd_owner_outputs(
                dq_buffer,
                dkv_buffer,
                dq_shape=dq_shape,
                dkv_shape=dkv_shape,
            )
        except SparseMLAFp8PathCDirectError as exc:
            if force_path_c:
                raise RuntimeError(str(exc)) from exc
            return None
    try:
        kernel = _fp8_bwd_tvm_ffi_kernel_for(
            batch,
            seq_len,
            heads,
            seq_len_kv,
            kv_group,
            head_kv,
            topk,
            K,
            d_v,
            threads,
            d_out_dtype,
            index_dtype,
        )
        # Both routes now dispatch into pre-zeroed owner buffers via out=:
        # the graph route zeroed dq/dkv at allocation above; the owner route
        # zero-clears the caller's dkv via the dedicated clear kernel.  This
        # guarantees the bwd kernel's accumulating atomic_add into dkv always
        # starts from zero instead of NaN-poison freed memory.
        if dkv_needs_clear:
            dkv_owner = _clear_fp8_bwd_dkv_buffer(cast(mx.array, dkv_owner))
        dq_flat = _flat_1d_view(cast(mx.array, dq_owner))
        dkv_flat = _flat_1d_view(cast(mx.array, dkv_owner))
        returned = kernel(
            _flat_1d_view(q_fp8),
            _flat_1d_view(q_scale),
            _flat_1d_view(kv_fp8),
            _flat_1d_view(kv_scale),
            _flat_1d_view(d_out),
            _flat_1d_view(indices),
            sm_scale_buf,
            out=(dq_flat, dkv_flat),
        )
    except Exception as exc:
        if force_path_c:
            route_label = (
                "graph-output (self-allocated zeroed buffers)"
                if graph_output_route
                else "caller-owned"
            )
            raise RuntimeError(
                f"sparse_mla_fp8_bwd_path_c: direct tvm-ffi {route_label} "
                f"dispatch failed: {type(exc).__name__}: {exc}"
            ) from exc
        return None
    dq_flat, dkv_flat = _owner_output_graph_tuple(
        returned,
        expected=(cast(mx.array, dq_flat), cast(mx.array, dkv_flat)),
        op_name="direct tvm-ffi FP8 Sparse-MLA backward",
    )
    return dq_flat.reshape(dq_shape), dkv_flat.reshape(dkv_shape)


def _tilelang_float_dtype(dtype: mx.Dtype, *, name: str = "dtype") -> str:
    if dtype == mx.float32:
        return "float32"
    if dtype == mx.float16:
        return "float16"
    if dtype == mx.bfloat16:
        return "bfloat16"
    raise TypeError(f"{name} must be float32, float16, or bfloat16; got {dtype}")


def _mx_float_dtype(dtype: mx.Dtype | None, *, default: mx.Dtype) -> mx.Dtype:
    resolved = default if dtype is None else dtype
    _tilelang_float_dtype(resolved, name="output_dtype")
    return resolved


def _validate_fp8_apply_owner_outputs(
    out: mx.array,
    lse: mx.array,
    *,
    batch: int,
    seq_len: int,
    heads: int,
    d_v: int,
    out_dtype: mx.Dtype,
) -> tuple[mx.array, mx.array]:
    if not isinstance(out, mx.array):
        raise TypeError(f"out must be an mlx.core.array; got {type(out).__name__}")
    if not isinstance(lse, mx.array):
        raise TypeError(f"lse must be an mlx.core.array; got {type(lse).__name__}")
    expected_out_shape = (batch, seq_len, heads, d_v)
    expected_lse_shape = (batch, seq_len, heads)
    if tuple(out.shape) != expected_out_shape:
        raise ValueError(
            f"out must have shape {expected_out_shape}; got {tuple(out.shape)}"
        )
    if tuple(lse.shape) != expected_lse_shape:
        raise ValueError(
            f"lse must have shape {expected_lse_shape}; got {tuple(lse.shape)}"
        )
    if out.dtype != out_dtype:
        raise TypeError(f"out must be {out_dtype}; got {out.dtype}")
    if lse.dtype != mx.float32:
        raise TypeError(f"lse must be mx.float32; got {lse.dtype}")
    return out, lse


def _make_fp8_per_token_quant_kernel(
    *,
    rows: int,
    K: int,
    in_dtype: str,
) -> Any:
    import tilelang.language as T
    from tilelang.tileop.metal_quant import float_to_fp8_e4m3fn_bits

    T = cast(Any, T)
    g = globals()
    g.update(
        _SMFP8_PTQ_ROWS=int(rows),
        _SMFP8_PTQ_K=int(K),
        _SMFP8_PTQ_INPUT_DTYPE=str(in_dtype),
    )

    @T.prim_func
    def fp8_per_token_quant(
        x: T.Tensor((_SMFP8_PTQ_ROWS * _SMFP8_PTQ_K,), _SMFP8_PTQ_INPUT_DTYPE),
        fp8: T.Tensor((_SMFP8_PTQ_ROWS * _SMFP8_PTQ_K,), "uint8"),
        scale: T.Tensor((_SMFP8_PTQ_ROWS,), "float32"),
    ):
        with T.Kernel(_SMFP8_PTQ_ROWS, threads=_SMFP8_PER_TOKEN_QUANT_THREADS) as row:
            x_abs = T.alloc_fragment((_SMFP8_PTQ_K,), "float32")
            row_amax = T.alloc_fragment((1,), "float32")
            base = row * _SMFP8_PTQ_K
            for k in T.Parallel(_SMFP8_PTQ_K):
                x_abs[k] = T.abs(T.cast(x[base + k], "float32"))
            T.reduce_max(x_abs, row_amax, dim=0, clear=True)
            row_scale = T.max(
                row_amax[0] * T.cast(1.0 / 448.0, "float32"),
                T.cast(1.0e-12, "float32"),
            )
            if T.get_thread_binding(0) == 0:
                scale[row] = row_scale
            for k in T.Parallel(_SMFP8_PTQ_K):
                normalized = T.alloc_var("float32")
                normalized = T.cast(x[base + k], "float32") / row_scale
                normalized = T.max(normalized, T.cast(-448.0, "float32"))
                normalized = T.min(normalized, T.cast(448.0, "float32"))
                fp8[base + k] = float_to_fp8_e4m3fn_bits(normalized)

    return fp8_per_token_quant


def make_fp8_sparse_mla_prepare_kernel(
    *,
    q_rows: int,
    kv_rows: int,
    K: int,
    in_dtype: str,
    storage_dtype: str = "uint8",
) -> Any:
    """Build the FP8 prepare producer PrimFunc for fused train-block schedules.

    ``post_y`` is a flat q-then-kv carrier. The kernel emits first-class
    prepared Sparse-MLA FP8 buffers and per-token scales so the downstream
    apply node can stay inside the same TileLang/TVM region.
    """

    if q_rows <= 0 or kv_rows <= 0 or K <= 0:
        raise ValueError("q_rows, kv_rows, and K must be positive")
    if storage_dtype not in {"uint8", "float8_e4m3"}:
        raise ValueError(
            "storage_dtype must be 'uint8' or 'float8_e4m3'; "
            f"got {storage_dtype!r}"
        )

    import tilelang.language as T
    from tilelang.tileop.metal_quant import float_to_fp8_e4m3fn_bits

    T = cast(Any, T)
    g = globals()
    g.update(
        _SMFP8_PREPARE_Q_ROWS=int(q_rows),
        _SMFP8_PREPARE_KV_ROWS=int(kv_rows),
        _SMFP8_PREPARE_ROWS=int(q_rows) + int(kv_rows),
        _SMFP8_PREPARE_K=int(K),
        _SMFP8_PREPARE_INPUT_DTYPE=str(in_dtype),
        _SMFP8_PREPARE_STORAGE_DTYPE=str(storage_dtype),
    )

    def encode_fp8(normalized):
        if storage_dtype == "uint8":
            return float_to_fp8_e4m3fn_bits(normalized)
        return T.cast(normalized, "float8_e4m3")

    @T.prim_func
    def fp8_sparse_mla_prepare(
        post_y: T.Tensor(
            (_SMFP8_PREPARE_ROWS * _SMFP8_PREPARE_K,),
            _SMFP8_PREPARE_INPUT_DTYPE,
        ),
        q_fp8: T.Tensor(
            (_SMFP8_PREPARE_Q_ROWS * _SMFP8_PREPARE_K,),
            _SMFP8_PREPARE_STORAGE_DTYPE,
        ),
        q_scale: T.Tensor((_SMFP8_PREPARE_Q_ROWS,), "float32"),
        kv_fp8: T.Tensor(
            (_SMFP8_PREPARE_KV_ROWS * _SMFP8_PREPARE_K,),
            _SMFP8_PREPARE_STORAGE_DTYPE,
        ),
        kv_scale: T.Tensor((_SMFP8_PREPARE_KV_ROWS,), "float32"),
    ):
        with T.Kernel(_SMFP8_PREPARE_ROWS, threads=_SMFP8_PER_TOKEN_QUANT_THREADS) as row:
            x_abs = T.alloc_fragment((_SMFP8_PREPARE_K,), "float32")
            row_amax = T.alloc_fragment((1,), "float32")
            base = row * _SMFP8_PREPARE_K
            for k in T.Parallel(_SMFP8_PREPARE_K):
                x_abs[k] = T.abs(T.cast(post_y[base + k], "float32"))
            T.reduce_max(x_abs, row_amax, dim=0, clear=True)
            row_scale = T.max(
                row_amax[0] * T.cast(1.0 / 448.0, "float32"),
                T.cast(1.0e-12, "float32"),
            )
            if row < _SMFP8_PREPARE_Q_ROWS:
                if T.get_thread_binding(0) == 0:
                    q_scale[row] = row_scale
                for k in T.Parallel(_SMFP8_PREPARE_K):
                    normalized = T.alloc_var("float32")
                    normalized = T.cast(post_y[base + k], "float32") / row_scale
                    normalized = T.max(normalized, T.cast(-448.0, "float32"))
                    normalized = T.min(normalized, T.cast(448.0, "float32"))
                    q_fp8[base + k] = encode_fp8(normalized)
            else:
                kv_row = row - _SMFP8_PREPARE_Q_ROWS
                kv_base = kv_row * _SMFP8_PREPARE_K
                if T.get_thread_binding(0) == 0:
                    kv_scale[kv_row] = row_scale
                for k in T.Parallel(_SMFP8_PREPARE_K):
                    normalized = T.alloc_var("float32")
                    normalized = T.cast(post_y[base + k], "float32") / row_scale
                    normalized = T.max(normalized, T.cast(-448.0, "float32"))
                    normalized = T.min(normalized, T.cast(448.0, "float32"))
                    kv_fp8[kv_base + k] = encode_fp8(normalized)

    return fp8_sparse_mla_prepare


@lru_cache(maxsize=128)
def _fp8_per_token_quant_tvm_ffi_kernel_for(
    rows: int,
    K: int,
    in_dtype: str,
) -> Any:
    import tilelang

    prim = _make_fp8_per_token_quant_kernel(
        rows=rows,
        K=K,
        in_dtype=in_dtype,
    )
    return tilelang.compile(
        prim,
        target=_msl_transform._as_metal_target(TILELANG_METAL_FP8_SPARSE_MLA_TARGET),
        execution_backend="tvm_ffi",
        out_idx=[1, 2],
    )


def _to_fp8_with_per_token_scale_metal(x: mx.array) -> tuple[mx.array, mx.array] | None:
    if not can_run_metal():
        return None
    K = int(x.shape[-1])
    if x.dtype not in {mx.float32, mx.float16, mx.bfloat16}:
        raise TypeError(f"FP8 producer input must be floating, got {x.dtype}")
    rows = 1
    for dim in x.shape[:-1]:
        rows *= int(dim)
    in_dtype = _tilelang_float_dtype(x.dtype, name="FP8 producer input")
    x_flat = x.reshape((rows * K,))
    try:
        kernel = _fp8_per_token_quant_tvm_ffi_kernel_for(rows, K, in_dtype)
        returned = kernel(x_flat)
    except Exception as exc:
        # RULE #1: this is the Metal FP8 per-token producer on a Metal host
        # (can_run_metal() True was asserted above). A compile/launch failure
        # here is a real bug; returning None makes the caller re-raise a generic
        # "requires native tvm-ffi producer" guard that hides the real cause.
        # Raise with where+what instead.
        raise RuntimeError(
            f"_to_fp8_with_per_token_scale_metal: Metal FP8 per-token quant "
            f"producer kernel compile/launch failed for rows={rows} K={K} "
            f"in_dtype={in_dtype} ({type(exc).__name__}: {exc})."
        ) from exc
    if not isinstance(returned, (list, tuple)) or len(returned) != 2:
        return None
    fp8_flat = cast(mx.array, returned[0])
    scale_flat = cast(mx.array, returned[1])
    return fp8_flat.reshape(x.shape), scale_flat.reshape(x.shape[:-1])


def sparse_mla_fp8_path_c_apply_direct(
    q_fp8: mx.array,
    q_scale: mx.array,
    kv_fp8: mx.array,
    kv_scale: mx.array,
    indices: mx.array,
    *,
    sm_scale: float,
    out: mx.array,
    lse: mx.array,
    d_v: int | None = None,
    sinks: mx.array | None = None,
    output_dtype: mx.Dtype | None = None,
) -> tuple[mx.array, mx.array]:
    """Run FP8 Sparse-MLA forward through tvm-ffi into caller-owned outputs."""

    if not can_run_metal():
        raise SparseMLAFp8PathCDirectError("MLX Metal backend is unavailable")
    (
        batch,
        seq_len,
        heads,
        seq_len_kv,
        kv_group,
        head_kv,
        topk,
        K,
        d_v_resolved,
        threads,
    ) = _validate_fp8_apply_inputs(
        q_fp8,
        q_scale,
        kv_fp8,
        kv_scale,
        indices,
        d_v=d_v,
    )
    out_buf, lse_buf = _validate_fp8_apply_owner_outputs(
        out,
        lse,
        batch=batch,
        seq_len=seq_len,
        heads=heads,
        d_v=d_v_resolved,
        out_dtype=_mx_float_dtype(output_dtype, default=out.dtype),
    )
    out_dtype = _tilelang_float_dtype(out_buf.dtype, name="output_dtype")
    try:
        index_dtype = _index_dtype_name(indices, op_name="FP8 Sparse-MLA Path C apply")
        kernel = _fp8_apply_tvm_ffi_kernel_for(
            batch,
            seq_len,
            heads,
            seq_len_kv,
            kv_group,
            head_kv,
            topk,
            K,
            d_v_resolved,
            threads,
            out_dtype,
            index_dtype,
        )
    except Exception as exc:
        raise SparseMLAFp8PathCDirectError(
            f"direct tvm-ffi FP8 Sparse-MLA forward compile failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    sm_scale_buf = mx.array([float(sm_scale)], dtype=mx.float32)
    if sinks is None:
        sinks_buf = mx.zeros((heads,), dtype=mx.float32)
        has_sinks_buf = mx.array([0], dtype=mx.int32)
    else:
        if not isinstance(sinks, mx.array):
            raise TypeError("sinks must be an mlx.core.array")
        if sinks.shape != (heads,):
            raise ValueError(f"sinks must have shape ({heads},), got {sinks.shape}")
        if sinks.dtype != mx.float32:
            raise ValueError("sinks must be float32")
        sinks_buf = sinks
        has_sinks_buf = mx.array([1], dtype=mx.int32)

    try:
        out_flat = _flat_1d_view(out_buf)
        lse_flat = _flat_1d_view(lse_buf)
        returned = kernel(
            _flat_1d_view(q_fp8),
            _flat_1d_view(q_scale),
            _flat_1d_view(kv_fp8),
            _flat_1d_view(kv_scale),
            _flat_1d_view(indices),
            sm_scale_buf,
            _flat_1d_view(sinks_buf),
            has_sinks_buf,
            out=(out_flat, lse_flat),
        )
    except Exception as exc:
        try:
            from tilelang.contrib.mlx_interop import DLPackInteropError
        except Exception:  # pragma: no cover - only when TileLang import itself is broken
            DLPackInteropError = ()  # type: ignore[assignment]
        if isinstance(exc, DLPackInteropError):
            raise
        raise SparseMLAFp8PathCDirectError(
            f"direct tvm-ffi FP8 Sparse-MLA forward dispatch failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    _owner_output_tuple(
        returned,
        expected=(out_flat, lse_flat),
        op_name="direct tvm-ffi FP8 Sparse-MLA forward",
    )
    return out_buf, lse_buf


def _sparse_mla_fp8_apply_cuda_eager(
    q_fp8: mx.array,
    q_scale: mx.array,
    kv_fp8: mx.array,
    kv_scale: mx.array,
    indices: mx.array,
    *,
    sm_scale: float,
    d_v: int | None,
    sinks: mx.array | None,
    output_dtype: mx.Dtype | None,
) -> tuple[mx.array, mx.array] | None:
    """Run the prepared-buffer FP8 Sparse-MLA apply on CUDA -> ``(out, lse)``."""

    from cppmega_mlx.nn._tilelang._cuda_eager import (
        cuda_eager_available,
        fp8_sparse_mla_apply_cuda_eager,
    )

    cuda_ok, _reason = cuda_eager_available()
    if not cuda_ok:
        return None
    (
        batch,
        seq_len,
        heads,
        seq_len_kv,
        kv_group,
        head_kv,
        topk,
        K,
        d_v_resolved,
        threads,
    ) = _validate_fp8_apply_inputs(
        q_fp8,
        q_scale,
        kv_fp8,
        kv_scale,
        indices,
        d_v=d_v,
    )
    output_dtype_resolved = _mx_float_dtype(output_dtype, default=mx.float16)
    out_dtype = _tilelang_float_dtype(output_dtype_resolved, name="output_dtype")
    if sinks is not None:
        if not isinstance(sinks, mx.array):
            raise TypeError("sinks must be an mlx.core.array")
        if sinks.shape != (heads,):
            raise ValueError(f"sinks must have shape ({heads},), got {sinks.shape}")
        if sinks.dtype != mx.float32:
            raise ValueError("sinks must be float32")
    index_dtype = _index_dtype_name(indices, op_name="FP8 Sparse-MLA Path C apply")
    return fp8_sparse_mla_apply_cuda_eager(
        q_fp8,
        q_scale,
        kv_fp8,
        kv_scale,
        indices,
        sm_scale=sm_scale,
        sinks=sinks,
        batch=batch,
        seq_len=seq_len,
        heads=heads,
        seq_len_kv=seq_len_kv,
        kv_group=kv_group,
        head_kv=head_kv,
        topk=topk,
        K=K,
        d_v=d_v_resolved,
        threads=threads,
        out_dtype=out_dtype,
        index_dtype=index_dtype,
    )


def sparse_mla_fp8_path_c_apply(
    q_fp8: mx.array,
    q_scale: mx.array,
    kv_fp8: mx.array,
    kv_scale: mx.array,
    indices: mx.array,
    *,
    sm_scale: float,
    d_v: int | None = None,
    sinks: mx.array | None = None,
    return_lse: bool = False,
    force_path_c: bool = False,
    out: mx.array | None = None,
    lse: mx.array | None = None,
    output_dtype: mx.Dtype | None = None,
    runtime_buffer_probe: Callable[[Mapping[str, Any]], None] | None = None,
) -> mx.array | tuple[mx.array, mx.array] | None:
    """Run fused FP8 Sparse-MLA Path C over prepared GPU buffers.

    The function consumes existing FP8/scales buffers directly. It deliberately
    does not quantize float tensors, cast scales, pre-gather KV, or materialize
    a score tensor in Python.
    """

    if (out is None) != (lse is None):
        raise ValueError(
            "sparse_mla_fp8_path_c_apply owner-output route requires both "
            "out and lse buffers"
        )
    if out is not None and lse is not None:
        direct_out, direct_lse = sparse_mla_fp8_path_c_apply_direct(
            q_fp8,
            q_scale,
            kv_fp8,
            kv_scale,
            indices,
            sm_scale=sm_scale,
            out=out,
            lse=lse,
            d_v=d_v,
            sinks=sinks,
            output_dtype=output_dtype,
        )
        if return_lse:
            return direct_out, direct_lse
        return direct_out

    if not can_run_metal():
        # CUDA EAGER branch: run the prepared-buffer FP8 Sparse-MLA apply
        # prim_func with target="cuda" over the same uint8/fp32 buffers.
        cuda_result = _sparse_mla_fp8_apply_cuda_eager(
            q_fp8,
            q_scale,
            kv_fp8,
            kv_scale,
            indices,
            sm_scale=sm_scale,
            d_v=d_v,
            sinks=sinks,
            output_dtype=output_dtype,
        )
        if cuda_result is not None:
            cuda_out, cuda_lse = cuda_result
            _emit_fp8_apply_runtime_buffer(
                runtime_buffer_probe,
                name="lse",
                tensor=cuda_lse,
            )
            if return_lse:
                return cuda_out, cuda_lse
            return cuda_out
        if force_path_c:
            raise RuntimeError(
                "sparse_mla_fp8_path_c_apply: MLX Metal backend is unavailable"
            )
        return None
    (
        batch,
        seq_len,
        heads,
        seq_len_kv,
        kv_group,
        head_kv,
        topk,
        K,
        d_v_resolved,
        threads,
    ) = _validate_fp8_apply_inputs(
        q_fp8,
        q_scale,
        kv_fp8,
        kv_scale,
        indices,
        d_v=d_v,
    )
    output_dtype_resolved = _mx_float_dtype(output_dtype, default=mx.float16)
    out_dtype = _tilelang_float_dtype(output_dtype_resolved, name="output_dtype")
    try:
        index_dtype = _index_dtype_name(indices, op_name="FP8 Sparse-MLA Path C apply")
        kernel = _fp8_apply_tvm_ffi_kernel_for(
            batch,
            seq_len,
            heads,
            seq_len_kv,
            kv_group,
            head_kv,
            topk,
            K,
            d_v_resolved,
            threads,
            out_dtype,
            index_dtype,
        )
        sm_scale_buf = mx.array([float(sm_scale)], dtype=mx.float32)
        if sinks is None:
            sinks_buf = mx.zeros((heads,), dtype=mx.float32)
            has_sinks_buf = mx.array([0], dtype=mx.int32)
        else:
            if not isinstance(sinks, mx.array):
                raise TypeError("sinks must be an mlx.core.array")
            if sinks.shape != (heads,):
                raise ValueError(f"sinks must have shape ({heads},), got {sinks.shape}")
            if sinks.dtype != mx.float32:
                raise ValueError("sinks must be float32")
            sinks_buf = sinks
            has_sinks_buf = mx.array([1], dtype=mx.int32)
        _emit_fp8_apply_runtime_buffer(
            runtime_buffer_probe,
            name="sparse_mla_sm_scale",
            tensor=sm_scale_buf,
        )
        _emit_fp8_apply_runtime_buffer(
            runtime_buffer_probe,
            name="sparse_mla_sinks",
            tensor=sinks_buf,
        )
        _emit_fp8_apply_runtime_buffer(
            runtime_buffer_probe,
            name="sparse_mla_has_sinks",
            tensor=has_sinks_buf,
        )
        returned = kernel(
            _flat_1d_view(q_fp8),
            _flat_1d_view(q_scale),
            _flat_1d_view(kv_fp8),
            _flat_1d_view(kv_scale),
            _flat_1d_view(indices),
            sm_scale_buf,
            _flat_1d_view(sinks_buf),
            has_sinks_buf,
        )
    except Exception as exc:
        if force_path_c:
            raise RuntimeError(
                "sparse_mla_fp8_path_c_apply: native TileLang tvm-ffi "
                f"graph-output dispatch failed: {type(exc).__name__}: {exc}"
            ) from exc
        return None
    if not isinstance(returned, (list, tuple)) or len(returned) != 2:
        if force_path_c:
            raise RuntimeError(
                "sparse_mla_fp8_path_c_apply: native TileLang tvm-ffi "
                "graph-output dispatch did not return out/lse"
        )
        return None
    _emit_fp8_apply_runtime_buffer(
        runtime_buffer_probe,
        name="lse",
        tensor=cast(mx.array, returned[1]),
    )
    direct_out = cast(mx.array, returned[0]).reshape(
        (batch, seq_len, heads, d_v_resolved)
    )
    direct_lse = cast(mx.array, returned[1]).reshape((batch, seq_len, heads))
    if return_lse:
        return direct_out, direct_lse
    return direct_out


def sparse_mla_fp8_bwd_path_c(
    q_fp8: mx.array,
    q_scale: mx.array,
    kv_fp8: mx.array,
    kv_scale: mx.array,
    d_out: mx.array,
    indices: mx.array,
    *,
    sm_scale: float,
    d_v: int | None = None,
    force_path_c: bool = False,
    causal: bool = False,
    dq_buffer: mx.array | None = None,
    dkv_buffer: mx.array | None = None,
) -> tuple[mx.array, mx.array] | None:
    """Run the TileLang Path C FP8 Sparse-MLA backward over prepared buffers."""

    if not can_run_metal():
        if force_path_c:
            raise RuntimeError(
                "sparse_mla_fp8_bwd_path_c: MLX Metal backend is unavailable"
            )
        return None
    (
        batch,
        seq_len,
        heads,
        seq_len_kv,
        kv_group,
        head_kv,
        topk,
        K,
        d_v_resolved,
        threads,
    ) = _validate_fp8_apply_inputs(
        q_fp8,
        q_scale,
        kv_fp8,
        kv_scale,
        indices,
        d_v=d_v,
    )
    expected_d_out_shape = (batch, seq_len, heads, d_v_resolved)
    if tuple(d_out.shape) != expected_d_out_shape:
        raise ValueError(
            f"d_out must have shape {expected_d_out_shape}; got {tuple(d_out.shape)}"
        )
    sm_scale_buf = mx.array([float(sm_scale)], dtype=mx.float32)
    d_out_dtype = _tilelang_float_dtype(d_out.dtype)
    del causal  # Sparse indices define the scatter pattern; kept for API compatibility.
    bwd_result = _dispatch_fp8_bwd_owner_output_path_c(
        q_fp8=q_fp8,
        q_scale=q_scale,
        kv_fp8=kv_fp8,
        kv_scale=kv_scale,
        d_out=d_out,
        indices=indices,
        sm_scale_buf=sm_scale_buf,
        batch=batch,
        seq_len=seq_len,
        heads=heads,
        seq_len_kv=seq_len_kv,
        kv_group=kv_group,
        head_kv=head_kv,
        topk=topk,
        K=K,
        d_v=d_v_resolved,
        threads=threads,
        d_out_dtype=d_out_dtype,
        dq_buffer=dq_buffer,
        dkv_buffer=dkv_buffer,
        force_path_c=force_path_c,
    )
    return bwd_result


def _to_fp8_with_per_token_scale(x: mx.array) -> tuple[mx.array, mx.array]:
    """Quantize a producer tensor to e4m3 + per-row scale in one Metal pass."""

    if x.ndim < 2:
        raise ValueError(f"FP8 producer input must be at least 2D, got {x.shape}")
    if x.size == 0:
        return mx.zeros(x.shape, dtype=mx.uint8), mx.ones(
            x.shape[:-1], dtype=mx.float32
        )
    metal_result = _to_fp8_with_per_token_scale_metal(x)
    if metal_result is not None:
        return metal_result
    # CUDA EAGER branch: Metal is unavailable (e.g. gb10 / sm_121). Run the
    # equivalent per-token FP8 producer as a TileLang-CUDA kernel that yields
    # the SAME prepared (uint8 e4m3 fp8, fp32 per-row scale) buffers the FP8
    # Sparse-MLA apply consumer expects — no full-size scaled-tensor
    # materialization, so the producer-owned contract is preserved.
    if not can_run_metal():
        cuda_result = _to_fp8_with_per_token_scale_cuda(x)
        if cuda_result is not None:
            return cuda_result
    raise RuntimeError(
        "_to_fp8_with_per_token_scale requires the native TileLang tvm-ffi "
        "per-token FP8 producer graph path; "
        "Path C must consume prepared q_fp8/q_scale/kv_fp8/kv_scale buffers "
        "and must not materialize a full-size scaled tensor fallback"
    )


def _to_fp8_with_per_token_scale_cuda(x: mx.array) -> tuple[mx.array, mx.array] | None:
    """TileLang-CUDA EAGER per-token FP8 producer (uint8 e4m3 + fp32 scale)."""

    from cppmega_mlx.nn._tilelang._cuda_eager import (
        cuda_eager_available,
        fp8_per_token_quant_cuda_eager,
    )

    cuda_ok, _reason = cuda_eager_available()
    if not cuda_ok:
        return None
    if x.dtype not in {mx.float32, mx.float16, mx.bfloat16}:
        raise TypeError(f"FP8 producer input must be floating, got {x.dtype}")
    return fp8_per_token_quant_cuda_eager(x)


def _prepared_fp8_bwd_ste(
    q: mx.array,
    kv: mx.array,
    q_fp8: mx.array,
    q_scale: mx.array,
    kv_fp8: mx.array,
    kv_scale: mx.array,
    d_out: mx.array,
    indices: mx.array,
    *,
    sm_scale: float,
    d_v: int | None,
    force_path_c: bool,
    causal: bool,
) -> tuple[mx.array, mx.array] | None:
    """Run the Path C FP8 sparse-MLA backward over per-token prepared buffers."""

    # CUDA EAGER branch: the Path C FP8 backward is a Metal/tvm-ffi owner-output
    # kernel that cannot dispatch on a non-Metal host (e.g. gb10 / sm_121).
    # Route to the memory-safe reference backward over the SAME prepared FP8
    # values so training gradients stay finite (mirrors c9f1a97's non-FP8
    # Sparse-MLA "fwd CUDA-eager, bwd reference VJP" design). Guarded so Metal
    # hosts keep the native Path C backward byte-for-byte.
    if not can_run_metal():
        try:
            from cppmega_mlx.nn._tilelang._cuda_eager import cuda_eager_available

            cuda_ok, _reason = cuda_eager_available()
        except Exception:
            cuda_ok = False
        if cuda_ok:
            return _prepared_fp8_reference_bwd_ste(
                q,
                kv,
                q_fp8,
                q_scale,
                kv_fp8,
                kv_scale,
                d_out,
                indices,
                sm_scale=sm_scale,
                d_v=d_v,
            )

    result = sparse_mla_fp8_bwd_path_c(
        q_fp8,
        q_scale,
        kv_fp8,
        kv_scale,
        d_out,
        indices,
        sm_scale=sm_scale,
        d_v=d_v,
        force_path_c=force_path_c,
        causal=causal,
    )
    if result is None:
        return None
    q_dtype = q.dtype
    kv_dtype = kv.dtype
    del q, kv
    dq, dkv = result
    if dq.dtype != q_dtype:
        dq = dq.astype(q_dtype)
    if dkv.dtype != kv_dtype:
        dkv = dkv.astype(kv_dtype)
    return dq, dkv


def _prepared_fp8_reference_bwd_ste(
    q: mx.array,
    kv: mx.array,
    q_fp8: mx.array,
    q_scale: mx.array,
    kv_fp8: mx.array,
    kv_scale: mx.array,
    d_out: mx.array,
    indices: mx.array,
    *,
    sm_scale: float,
    d_v: int | None,
) -> tuple[mx.array, mx.array]:
    """Memory-safe Path B backward over the same prepared FP8 values."""

    from cppmega_mlx.nn.sparse_mla import sparse_mla_attention_reference

    q_ref = (
        mx.from_fp8(q_fp8, dtype=mx.float32) * q_scale.astype(mx.float32)[..., None]
    ).astype(q.dtype)
    kv_ref = (
        mx.from_fp8(kv_fp8, dtype=mx.float32) * kv_scale.astype(mx.float32)[..., None]
    ).astype(kv.dtype)

    def _ref_apply(q_in: mx.array, kv_in: mx.array) -> mx.array:
        out = sparse_mla_attention_reference(
            q_in,
            kv_in,
            indices,
            sm_scale=sm_scale,
            d_v=d_v,
            return_lse=False,
        )
        if isinstance(out, tuple):
            return out[0]
        return out

    _, vjps = mx.vjp(_ref_apply, (q_ref, kv_ref), (d_out,))
    dq = vjps[0]
    dkv = vjps[1]
    if dq.dtype != q.dtype:
        dq = dq.astype(q.dtype)
    if dkv.dtype != kv.dtype:
        dkv = dkv.astype(kv.dtype)
    return dq, dkv


def _prepared_fp8_backward_uses_path_c(
    *,
    force_path_c: bool,
    force_backward_path_c: bool | None,
) -> bool:
    route = os.environ.get(SPARSE_MLA_FP8_BWD_ENV, "").strip().lower()
    # CUDA EAGER branch: the FP8 Sparse-MLA Path C backward is a Metal/tvm-ffi
    # owner-output kernel. On a non-Metal host (e.g. gb10 / sm_121) it cannot
    # dispatch, so unless the caller *explicitly* forces the Path C backward we
    # take the memory-safe reference backward over the same prepared FP8 values
    # (mirrors c9f1a97's non-FP8 Sparse-MLA "fwd CUDA-eager, bwd reference VJP"
    # design). An explicit ``force_backward_path_c=True`` / env still wins so the
    # failure surfaces rather than being silently downgraded.
    if not can_run_metal():
        try:
            from cppmega_mlx.nn._tilelang._cuda_eager import cuda_eager_available

            cuda_ok, _reason = cuda_eager_available()
        except Exception:
            cuda_ok = False
        if cuda_ok:
            if force_backward_path_c is True:
                return True
            if route in {"path_c", "c"}:
                return True
            if route in {"path_b", "b"}:
                return False
            if route:
                raise ValueError(
                    f"{SPARSE_MLA_FP8_BWD_ENV} must be path_b or path_c; got {route!r}"
                )
            return False
    if force_backward_path_c is not None:
        return bool(force_backward_path_c)
    if route in {"path_c", "c"}:
        return True
    if route in {"path_b", "b"}:
        return False
    if route:
        raise ValueError(
            f"{SPARSE_MLA_FP8_BWD_ENV} must be path_b or path_c; got {route!r}"
        )
    return force_path_c


def sparse_mla_fp8_path_c_apply_prepared_float(
    q: mx.array,
    kv: mx.array,
    q_fp8: mx.array,
    q_scale: mx.array,
    kv_fp8: mx.array,
    kv_scale: mx.array,
    indices: mx.array,
    *,
    sm_scale: float,
    d_v: int | None = None,
    sinks: mx.array | None = None,
    force_path_c: bool = False,
    causal: bool = False,
    output_dtype: mx.Dtype | None = None,
    runtime_buffer_probe: Callable[[Mapping[str, Any]], None] | None = None,
    force_backward_path_c: bool | None = None,
) -> mx.array:
    """Differentiable owner wrapper for prepared-buffer FP8 Path C apply.

    ``q_fp8/q_scale/kv_fp8/kv_scale`` must already be produced by the caller.
    The VJP is defined at the float producer boundary so training gradients
    flow back to Q/KV projections instead of stopping at the uint8 FP8 storage
    tensors.
    """

    from cppmega_mlx.nn._tilelang._sparse_mla_v32_fused import (
        v32_fused_apply,
        v32_fused_bwd,
        v32_fused_enabled,
    )

    use_v32_fused = v32_fused_enabled()

    @mx.custom_function
    def _apply(
        q_in: mx.array,
        kv_in: mx.array,
    ) -> mx.array:
        if use_v32_fused:
            # Env-gated real fused DeepSeek-V3.2 Sparse-MLA forward over bf16
            # q/kv/indices (O(seq*topk), full online softmax + LSE). RULE #1:
            # v32_fused_apply RAISES loudly on any kernel failure; we never
            # silently keep the reference here while claiming the fused path ran.
            return v32_fused_apply(
                q_in,
                kv_in,
                indices,
                sm_scale=sm_scale,
                return_lse=False,
            )
        apply_kwargs: dict[str, Any] = {
            "sm_scale": sm_scale,
            "d_v": d_v,
            "sinks": sinks,
            "force_path_c": force_path_c,
            "output_dtype": _mx_float_dtype(output_dtype, default=q_in.dtype),
        }
        if runtime_buffer_probe is not None:
            apply_kwargs["runtime_buffer_probe"] = runtime_buffer_probe
        out = sparse_mla_fp8_path_c_apply(
            q_fp8,
            q_scale,
            kv_fp8,
            kv_scale,
            indices,
            **apply_kwargs,
        )
        if out is None:
            if force_path_c:
                raise RuntimeError(
                    "sparse_mla_fp8_path_c_apply_prepared_float: "
                    "Path C forward unavailable"
                )
            from cppmega_mlx.nn._tilelang.sparse_mla_fp8 import (
                sparse_mla_fp8_reference,
            )

            return sparse_mla_fp8_reference(
                q_in,
                kv_in,
                indices,
                sm_scale=sm_scale,
                d_v=d_v,
                return_lse=False,
            )
        if isinstance(out, tuple):
            out = out[0]
        return out

    @_apply.vjp
    def _apply_vjp(primals, cotangent, output):  # noqa: ARG001
        q_in, kv_in = primals
        if sinks is not None:
            raise RuntimeError(
                "sparse_mla_fp8_path_c_apply_prepared_float: sinks backward is not "
                "implemented"
            )
        if use_v32_fused:
            # Env-gated real fused v32 Sparse-MLA backward (exercises
            # T.atomic_addx4 dKV scatter). Recompute fwd to obtain LSE the bwd
            # needs, then run the fused dq/dkv. RULE #1: RAISES on failure.
            fwd_out, fwd_lse = v32_fused_apply(
                q_in,
                kv_in,
                indices,
                sm_scale=sm_scale,
                return_lse=True,
            )
            dq, dkv = v32_fused_bwd(
                q_in,
                kv_in,
                fwd_out,
                cotangent,
                indices,
                fwd_lse,
                sm_scale=sm_scale,
            )
            if dq.dtype != q_in.dtype:
                dq = dq.astype(q_in.dtype)
            if dkv.dtype != kv_in.dtype:
                dkv = dkv.astype(kv_in.dtype)
            return dq, dkv
        if _prepared_fp8_backward_uses_path_c(
            force_path_c=force_path_c,
            force_backward_path_c=force_backward_path_c,
        ):
            grads = _prepared_fp8_bwd_ste(
                q_in,
                kv_in,
                q_fp8,
                q_scale,
                kv_fp8,
                kv_scale,
                cotangent,
                indices,
                sm_scale=sm_scale,
                d_v=d_v,
                force_path_c=force_path_c,
                causal=causal,
            )
        else:
            grads = _prepared_fp8_reference_bwd_ste(
                q_in,
                kv_in,
                q_fp8,
                q_scale,
                kv_fp8,
                kv_scale,
                cotangent,
                indices,
                sm_scale=sm_scale,
                d_v=d_v,
            )
        if grads is None:
            if force_path_c:
                raise RuntimeError(
                    "sparse_mla_fp8_path_c_apply_prepared_float: "
                    "Path C backward unavailable"
                )
            from cppmega_mlx.nn._tilelang.sparse_mla_fp8 import (
                sparse_mla_fp8_reference,
            )

            def _ref_apply(q_ref: mx.array, kv_ref: mx.array) -> mx.array:
                return sparse_mla_fp8_reference(
                    q_ref,
                    kv_ref,
                    indices,
                    sm_scale=sm_scale,
                    d_v=d_v,
                    return_lse=False,
                )

            _, vjps = mx.vjp(_ref_apply, (q_in, kv_in), (cotangent,))
            return vjps[0], vjps[1]
        dq, dkv = grads
        if dq.dtype != q_in.dtype:
            dq = dq.astype(q_in.dtype)
        if dkv.dtype != kv_in.dtype:
            dkv = dkv.astype(kv_in.dtype)
        return dq, dkv

    return _apply(q, kv)


def sparse_mla_fp8_path_c_apply_from_float(
    q: mx.array,
    kv: mx.array,
    indices: mx.array,
    *,
    sm_scale: float,
    d_v: int | None = None,
    sinks: mx.array | None = None,
    force_path_c: bool = False,
) -> mx.array:
    """Fail-closed compatibility hook for float Q/KV callers.

    Path C's public Sparse-MLA FP8 ABI is prepared-buffer only. Quantizing
    float Q/KV here would hide large staging tensors behind a wrapper boundary
    and break the fusion contract. Callers that need autograd over float
    producer tensors must pass existing FP8/scales buffers through
    ``sparse_mla_fp8_path_c_apply_prepared_float``.
    """

    del q, kv, indices, sm_scale, d_v, sinks, force_path_c
    raise RuntimeError(
        "sparse_mla_fp8_path_c_apply_from_float requires prepared FP8 buffers; "
        "use sparse_mla_fp8_path_c_apply_prepared_float with existing "
        "q_fp8/q_scale/kv_fp8/kv_scale buffers instead of materializing them "
        "inside the Path C wrapper"
    )


def fp8_sparse_mla_qk_reduce_path_c_status(
    *,
    N: int = 16,
    K: int = 64,
    outputs_per_block: int = _SMFP8_QKR_DEFAULT_OUTPUTS_PER_BLOCK,
    reduce_threads: int = _SMFP8_QKR_DEFAULT_REDUCE_THREADS,
    vec: int = _SMFP8_QKR_DEFAULT_VEC,
    target: str = TILELANG_METAL_FP8_SPARSE_MLA_TARGET,
) -> SparseMLAFp8QKReducePathCStatus:
    """Return whether the real-shape FP8 QK reducer can dispatch."""

    outputs_per_block, reduce_threads, vec = _resolve_qk_reduce_schedule(
        N=N,
        K=K,
        outputs_per_block=outputs_per_block,
        reduce_threads=reduce_threads,
        vec=vec,
    )
    ok, reason = _tilelang_available()
    if not ok:
        return SparseMLAFp8QKReducePathCStatus(
            available=False,
            reason=reason,
            features={},
            target=target,
            n=N,
            k=K,
            outputs_per_block=outputs_per_block,
            reduce_threads=reduce_threads,
            vec=vec,
        )
    if not can_run_metal():
        return SparseMLAFp8QKReducePathCStatus(
            available=False,
            reason="MLX Metal backend is not available on the default GPU device",
            features={},
            target=target,
            n=N,
            k=K,
            outputs_per_block=outputs_per_block,
            reduce_threads=reduce_threads,
            vec=vec,
        )
    try:
        kernel, lowering, _ = _qk_reduce_kernel_for(
            N, K, outputs_per_block, reduce_threads, vec, N
        )
        del kernel
        features = fp8_sparse_mla_qk_reduce_msl_features(lowering.msl_text)
        sync_plan = fp8_sparse_mla_qk_reduce_sync_plan(
            N=N,
            K=K,
            outputs_per_block=outputs_per_block,
            reduce_threads=reduce_threads,
            vec=vec,
        )
        features.update(sync_plan.as_feature_dict())
    except Exception as exc:
        return SparseMLAFp8QKReducePathCStatus(
            available=False,
            reason=f"TileLang/MLX lowering failed for FP8 Sparse-MLA QK reducer: {type(exc).__name__}: {exc}",
            features={},
            target=target,
            n=N,
            k=K,
            outputs_per_block=outputs_per_block,
            reduce_threads=reduce_threads,
            vec=vec,
        )

    has_scale_refs = bool(features["A_scale_refs"]) and bool(features["B_scale_refs"])
    has_scale_signature = bool(features["signature_has_A_scale"]) and bool(
        features["signature_has_B_scale"]
    )
    has_reduce = bool(
        features["simd_sum"]
        or features["simd_shuffle_down"]
        or features["tvm_thread_allreduce"]
    )
    if has_scale_refs and has_scale_signature and has_reduce:
        return SparseMLAFp8QKReducePathCStatus(
            available=True,
            reason=(
                "TileLang Path C FP8 Sparse-MLA real QK reducer is dispatchable "
                "for M=1/topk with per-row B scales through tvm-ffi "
                "owner-output dispatch; "
                f"sync plan: {sync_plan.strategy}"
            ),
            features=features,
            target=target,
            n=N,
            k=K,
            outputs_per_block=outputs_per_block,
            reduce_threads=reduce_threads,
            vec=vec,
        )

    blockers: list[str] = []
    if not has_scale_refs or not has_scale_signature:
        blockers.append("scale operands missing from emitted MSL")
    if not has_reduce:
        blockers.append("thread reduction missing from emitted MSL")
    return SparseMLAFp8QKReducePathCStatus(
        available=False,
        reason="TileLang Path C FP8 Sparse-MLA real QK reducer is not safe to dispatch: "
        + "; ".join(blockers),
        features=features,
        target=target,
        n=N,
        k=K,
        outputs_per_block=outputs_per_block,
        reduce_threads=reduce_threads,
        vec=vec,
    )


def fp8_sparse_mla_indexed_qk_reduce_path_c_status(
    *,
    batch: int = 1,
    seq_len: int = 1,
    heads: int = 1,
    seq_len_kv: int = 16,
    kv_group: int = 1,
    topk: int = 16,
    K: int = 64,
    outputs_per_block: int = _SMFP8_QKR_DEFAULT_OUTPUTS_PER_BLOCK,
    reduce_threads: int = _SMFP8_QKR_DEFAULT_REDUCE_THREADS,
    vec: int = _SMFP8_QKR_DEFAULT_VEC,
    target: str = TILELANG_METAL_FP8_SPARSE_MLA_TARGET,
) -> SparseMLAFp8IndexedQKReducePathCStatus:
    """Return whether the indexed full-shape FP8 QK reducer can dispatch."""

    outputs_per_block, reduce_threads, vec = _resolve_qk_reduce_schedule(
        N=topk,
        K=K,
        outputs_per_block=outputs_per_block,
        reduce_threads=reduce_threads,
        vec=vec,
    )
    try:
        head_kv = _validate_indexed_reduce_shape(
            batch=batch,
            seq_len=seq_len,
            heads=heads,
            seq_len_kv=seq_len_kv,
            kv_group=kv_group,
            topk=topk,
            K=K,
            outputs_per_block=outputs_per_block,
            reduce_threads=reduce_threads,
            vec=vec,
        )
    except ValueError as exc:
        return SparseMLAFp8IndexedQKReducePathCStatus(
            available=False,
            reason=str(exc),
            features={},
            target=target,
            batch=batch,
            seq_len=seq_len,
            heads=heads,
            seq_len_kv=seq_len_kv,
            kv_group=kv_group,
            head_kv=1,
            topk=topk,
            k=K,
            outputs_per_block=outputs_per_block,
            reduce_threads=reduce_threads,
            vec=vec,
        )

    ok, reason = _tilelang_available()
    if not ok:
        return SparseMLAFp8IndexedQKReducePathCStatus(
            available=False,
            reason=reason,
            features={},
            target=target,
            batch=batch,
            seq_len=seq_len,
            heads=heads,
            seq_len_kv=seq_len_kv,
            kv_group=kv_group,
            head_kv=head_kv,
            topk=topk,
            k=K,
            outputs_per_block=outputs_per_block,
            reduce_threads=reduce_threads,
            vec=vec,
        )
    if not can_run_metal():
        return SparseMLAFp8IndexedQKReducePathCStatus(
            available=False,
            reason="MLX Metal backend is not available on the default GPU device",
            features={},
            target=target,
            batch=batch,
            seq_len=seq_len,
            heads=heads,
            seq_len_kv=seq_len_kv,
            kv_group=kv_group,
            head_kv=head_kv,
            topk=topk,
            k=K,
            outputs_per_block=outputs_per_block,
            reduce_threads=reduce_threads,
            vec=vec,
        )
    try:
        kernel, lowering, _ = _indexed_qk_reduce_kernel_for(
            batch,
            seq_len,
            heads,
            seq_len_kv,
            kv_group,
            topk,
            K,
            outputs_per_block,
            reduce_threads,
            vec,
            "int32",
        )
        del kernel
        features = fp8_sparse_mla_indexed_qk_reduce_msl_features(lowering.msl_text)
        sync_plan = fp8_sparse_mla_qk_reduce_sync_plan(
            N=topk,
            K=K,
            outputs_per_block=outputs_per_block,
            reduce_threads=reduce_threads,
            vec=vec,
        )
        features.update(sync_plan.as_feature_dict())
    except Exception as exc:
        return SparseMLAFp8IndexedQKReducePathCStatus(
            available=False,
            reason=f"TileLang/MLX lowering failed for indexed FP8 Sparse-MLA QK reducer: {type(exc).__name__}: {exc}",
            features={},
            target=target,
            batch=batch,
            seq_len=seq_len,
            heads=heads,
            seq_len_kv=seq_len_kv,
            kv_group=kv_group,
            head_kv=head_kv,
            topk=topk,
            k=K,
            outputs_per_block=outputs_per_block,
            reduce_threads=reduce_threads,
            vec=vec,
        )

    has_scales = bool(features["q_scale_refs"]) and bool(features["kv_scale_refs"])
    has_inputs = (
        bool(features["signature_has_q_scale"])
        and bool(features["signature_has_kv_scale"])
        and bool(features["signature_has_indices"])
        and bool(features["signature_has_sm_scale"])
    )
    has_reduce = bool(
        features["simd_sum"]
        or features["simd_shuffle_down"]
        or features["tvm_thread_allreduce"]
    )
    has_mask = bool(features["invalid_index_guard"])
    has_packed_hot_loop = (
        int(features["scalar_fp8_byte_decode_calls"]) == 0
        and int(features["metal_fp8_dot4_helper"]) >= 1
    )
    if has_scales and has_inputs and has_reduce and has_mask and has_packed_hot_loop:
        return SparseMLAFp8IndexedQKReducePathCStatus(
            available=True,
            reason=(
                "TileLang Path C FP8 Sparse-MLA indexed QK reducer is dispatchable "
                "without host pre-gather through tvm-ffi owner-output dispatch "
                "and uses packed FP8 dot4 decode; "
                f"sync plan: {sync_plan.strategy}"
            ),
            features=features,
            target=target,
            batch=batch,
            seq_len=seq_len,
            heads=heads,
            seq_len_kv=seq_len_kv,
            kv_group=kv_group,
            head_kv=head_kv,
            topk=topk,
            k=K,
            outputs_per_block=outputs_per_block,
            reduce_threads=reduce_threads,
            vec=vec,
        )

    blockers: list[str] = []
    if not has_scales or not has_inputs:
        blockers.append("indexed/scaled operands missing from emitted MSL")
    if not has_reduce:
        blockers.append("thread reduction missing from emitted MSL")
    if not has_mask:
        blockers.append("invalid-index mask missing from emitted MSL")
    if not has_packed_hot_loop:
        blockers.append("packed FP8 dot4 hot loop missing from emitted MSL")
    return SparseMLAFp8IndexedQKReducePathCStatus(
        available=False,
        reason="TileLang Path C FP8 Sparse-MLA indexed QK reducer is not safe to dispatch: "
        + "; ".join(blockers),
        features=features,
        target=target,
        batch=batch,
        seq_len=seq_len,
        heads=heads,
        seq_len_kv=seq_len_kv,
        kv_group=kv_group,
        head_kv=head_kv,
        topk=topk,
        k=K,
        outputs_per_block=outputs_per_block,
        reduce_threads=reduce_threads,
        vec=vec,
    )


__all__ = [
    "SparseMLAFp8IndexedQKReducePathCStatus",
    "SparseMLAFp8PathCDirectError",
    "SparseMLAFp8QKReducePathCStatus",
    "SparseMLAFp8PathCStatus",
    "TILELANG_METAL_FP8_SPARSE_MLA_TARGET",
    "fp8_sparse_mla_indexed_qk_reduce_msl_features",
    "fp8_sparse_mla_indexed_qk_reduce_path_c",
    "fp8_sparse_mla_indexed_qk_reduce_path_c_status",
    "fp8_sparse_mla_qk_reduce_msl_features",
    "fp8_sparse_mla_qk_reduce_path_c",
    "fp8_sparse_mla_qk_reduce_grouped_path_c",
    "make_fp8_sparse_mla_qk_reduce_matmul2d_kernel",
    "fp8_sparse_mla_qk_reduce_path_c_status",
    "fp8_sparse_mla_qk_msl_features",
    "fp8_sparse_mla_qk_path_c_status",
    "fp8_sparse_mla_qk_scaled_matmul_probe_status",
    "lower_fp8_sparse_mla_indexed_qk_reduce_msl",
    "lower_fp8_sparse_mla_qk_reduce_msl",
    "lower_fp8_sparse_mla_qk_msl",
    "make_fp8_sparse_mla_prepare_kernel",
    "make_fp8_sparse_mla_indexed_qk_reduce_kernel",
    "make_fp8_sparse_mla_qk_reduce_kernel",
    "make_fp8_sparse_mla_qk_kernel",
    "sparse_mla_fp8_bwd_path_c",
    "sparse_mla_fp8_path_c_apply",
    "sparse_mla_fp8_path_c_apply_direct",
    "sparse_mla_fp8_path_c_apply_from_float",
    "sparse_mla_fp8_path_c_apply_prepared_float",
]

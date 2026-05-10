# pyright: reportInvalidTypeForm=false, reportMissingImports=false
# TODO(migration-phase-3-fp8): unified `_engine_dispatch.dispatch_lower` flip
# deferred to wave-6. The FP8 reducers depend on `tirx.metal.fp8_e4m3_dot4`
# which is not yet registered in the in-tree TileLang/TVM build (see status
# note below); dispatching them through the unified engine surfaces the same
# AttributeError. The broader migration also needs `simdgroup_a_fp8` /
# `simdgroup_b_fp8` Fragment factories (extension of #08 simdgroup_a/b/c
# factories at `tilelang/language/extern.py`). Track wave-6.
"""Path C FP8 Sparse-MLA TileLang DSL surfaces.

This module owns the FP8 Sparse-MLA Path C kernels over the *prepared* ABI:
``q_fp8/q_scale/kv_fp8/kv_scale/indices`` are existing GPU buffers already
created by the upstream graph. The public ``sparse_mla_fp8_path_c_apply`` does
not quantize, gather, cast, or allocate staging tensors; if callers have
float carriers they should route through the higher-level graph planner so the
producer emits FP8 buffers directly, or use the existing Path B wrapper.

Runtime status: currently broken — ``tirx.metal.fp8_e4m3_dot4`` is not
registered in the in-tree TileLang/TVM build, so dispatching the lowered FP8
reducers raises ``AttributeError: Operator tirx.metal.fp8_e4m3_dot4 is not
registered``. See agent-D report
``reports/2026-05-06-tilelang-tvm-review/agent-D-planning-vs-reality/grok__design__20260506T171408.md``
finding #1.

Path B ships the full direct-MSL FP8 Sparse-MLA forward/backward kernels in
``sparse_mla_fp8.py``.  This module owns the Path C QK tile needed by that
forward path and exposes two separate TileLang surfaces:

* ``T.fp8_scaled_matmul`` probe/status glue. Current apple-head TileLang can
  lower square 32x32 FP8 matmul to the Metal simdgroup path with explicit scale
  loads, but the literal Sparse-MLA QK shape (M=1 query row against top-k
  transposed KV rows) is still fail-closed because it scalarizes or drops scale
  operands.
* A real-shape ``@T.prim_func`` reducer for ``A_fp8(1, K) @ B_fp8(N, K).T``.
  It lowers through TileLang to MSL, dispatches via ``mx.fast.metal_kernel``,
  preserves scalar A scale plus per-row B scale semantics, and is benchmarked
  against Path B as the current runnable FP8 Path C QK tile.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast

import mlx.core as mx

from cppmega_mlx.nn._tilelang import _msl_transform
from cppmega_mlx.nn._tilelang._msl_transform import (
    MSLDispatchUnsupported,
    can_run_metal,
    lower_tilelang_to_msl_inline,
)


TILELANG_METAL_FP8_SPARSE_MLA_TARGET = "metal"

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
_SMFP8_APPLY_KV_SIZE = _SMFP8_APPLY_B * _SMFP8_APPLY_SKV * _SMFP8_APPLY_G * _SMFP8_APPLY_K
_SMFP8_APPLY_Q_SCALE_SIZE = _SMFP8_APPLY_LANES
_SMFP8_APPLY_KV_SCALE_SIZE = _SMFP8_APPLY_B * _SMFP8_APPLY_SKV * _SMFP8_APPLY_G
_SMFP8_APPLY_IDX_SIZE = _SMFP8_APPLY_B * _SMFP8_APPLY_S * _SMFP8_APPLY_G * _SMFP8_APPLY_TOPK
_SMFP8_APPLY_OUT_SIZE = _SMFP8_APPLY_LANES * _SMFP8_APPLY_DV
_SMFP8_APPLY_LSE_SIZE = _SMFP8_APPLY_LANES


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
        raise ValueError(f"FP8 Sparse-MLA Path C reducer shape values must be positive: {bad}")


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
        raise ValueError(f"FP8 Sparse-MLA Path C indexed reducer values must be positive: {bad}")
    if heads % kv_group != 0:
        raise ValueError(f"heads must be divisible by kv_group for Sparse-MLA grouping: {heads=} {kv_group=}")
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
            A_shared = T.alloc_shared((_SMFP8_BM, _SMFP8_BK), "float8_e4m3", scope="shared")
            B_shared = T.alloc_shared(_SMFP8_B_SHARED_SHAPE, "float8_e4m3", scope="shared")
            C_local = T.alloc_fragment((_SMFP8_BM, _SMFP8_BN), "float32")
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(_SMFP8_K, _SMFP8_BK), num_stages=_SMFP8_NUM_STAGES):
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


def _kernel_body_for_feature_counts(msl: str) -> str:
    _signature, body = _kernel_signature_and_body_for_feature_counts(msl)
    return body


def _kernel_signature_and_body_for_feature_counts(msl: str) -> tuple[str, str]:
    try:
        _prelude, sig_text, body_text = _msl_transform._split_kernel_msl(msl)
    except Exception:
        fallback = msl.split("kernel void", 1)[-1] if "kernel void" in msl else msl
        return fallback.split("{", 1)[0], fallback
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
    has_scale_signature = bool(features["signature_has_A_scale"]) and bool(features["signature_has_B_scale"])
    has_scalar_fallback = bool(features["float_a_val"]) or bool(features["float_b_val"])
    if has_fast_path and has_scale_refs and has_scale_signature and not has_scalar_fallback:
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
        blockers.append("Sparse-MLA M=1/topk tile violates current Metal FP8 simdgroup tile constraints")
    return SparseMLAFp8PathCStatus(
        available=False,
        reason="TileLang Path C FP8 Sparse-MLA QK is not safe to dispatch: " + "; ".join(blockers),
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
            probe_status.features,
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
                    "legacy_fp8_scaled_matmul_probe_available": bool(probe_status.available),
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
) -> Any:
    """Build the real Sparse-MLA FP8 QK tile as a TileLang reducer.

    This intentionally does not use ``T.fp8_scaled_matmul``.  The current
    Metal lowering rejects or scalarizes the ``M=1`` Sparse-MLA shape there,
    while the reducer below matches Path B's QK tile contract directly:

    * ``A_fp8`` is a single query row ``(1, K)`` in e4m3 byte storage.
    * ``B_fp8`` is gathered/transposed KV rows ``(N, K)``.
    * ``B_scale`` is per gathered KV row, matching Sparse-MLA FP8 scale use.
    """

    _validate_reduce_shape(
        N=N,
        K=K,
        outputs_per_block=outputs_per_block,
        reduce_threads=reduce_threads,
        vec=vec,
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
    )

    if vec == 4 and K % 4 == 0:

        @T.prim_func
        def fp8_sparse_mla_qk_reduce(
            A_fp8: T.Tensor((1, _SMFP8_QKR_K), "float8_e4m3"),
            A_scale: T.Tensor((1,), "float32"),
            B_fp8: T.Tensor((_SMFP8_QKR_N, _SMFP8_QKR_K), "float8_e4m3"),
            B_scale: T.Tensor((_SMFP8_QKR_N,), "float32"),
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
                    C[0, col] = reduced[0] * A_scale[0] * B_scale[col]

    else:

        @T.prim_func
        def fp8_sparse_mla_qk_reduce(
            A_fp8: T.Tensor((1, _SMFP8_QKR_K), "float8_e4m3"),
            A_scale: T.Tensor((1,), "float32"),
            B_fp8: T.Tensor((_SMFP8_QKR_N, _SMFP8_QKR_K), "float8_e4m3"),
            B_scale: T.Tensor((_SMFP8_QKR_N,), "float32"),
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
                    C[0, col] = reduced[0] * A_scale[0] * B_scale[col]

    try:
        from tilelang.transform.simplify import apply_simplify

        return apply_simplify(fp8_sparse_mla_qk_reduce)
    except Exception:
        return fp8_sparse_mla_qk_reduce


def lower_fp8_sparse_mla_qk_reduce_msl(
    *,
    N: int = 16,
    K: int = 64,
    outputs_per_block: int = _SMFP8_QKR_DEFAULT_OUTPUTS_PER_BLOCK,
    reduce_threads: int = _SMFP8_QKR_DEFAULT_REDUCE_THREADS,
    vec: int = _SMFP8_QKR_DEFAULT_VEC,
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
    )

    @T.prim_func
    def fp8_sparse_mla_indexed_qk_reduce(
        q_fp8: T.Tensor((_SMFP8_IQKR_Q_SIZE,), "float8_e4m3"),
        q_scale: T.Tensor((_SMFP8_IQKR_Q_SCALE_SIZE,), "float32"),
        kv_fp8: T.Tensor((_SMFP8_IQKR_KV_SIZE,), "float8_e4m3"),
        kv_scale: T.Tensor((_SMFP8_IQKR_KV_SCALE_SIZE,), "float32"),
        indices: T.Tensor((_SMFP8_IQKR_IDX_SIZE,), "int32"),
        sm_scale_buf: T.Tensor((1,), "float32"),
        scores: T.Tensor((_SMFP8_IQKR_OUT_SIZE,), "float32"),
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
            q_scale_idx = lane_gid
            idx_base = ((b * _SMFP8_IQKR_S + s) * _SMFP8_IQKR_G + group) * _SMFP8_IQKR_TOPK
            out_base = lane_gid * _SMFP8_IQKR_TOPK
            T.clear(accum)
            if topk_col < _SMFP8_IQKR_TOPK:
                gather_idx[0] = indices[idx_base + topk_col]
                if gather_idx[0] >= 0 and gather_idx[0] < _SMFP8_IQKR_SKV:
                    kv_base = ((b * _SMFP8_IQKR_SKV + gather_idx[0]) * _SMFP8_IQKR_G + group) * _SMFP8_IQKR_K
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
                gather_idx[0] = indices[idx_base + topk_col]
                if gather_idx[0] < 0 or gather_idx[0] >= _SMFP8_IQKR_SKV:
                    scores[out_base + topk_col] = T.float32(_SMFP8_INVALID_SCORE_SENTINEL)
                else:
                    kv_scale_idx = (b * _SMFP8_IQKR_SKV + gather_idx[0]) * _SMFP8_IQKR_G + group
                    scores[out_base + topk_col] = (
                        reduced[0]
                        * q_scale[q_scale_idx]
                        * kv_scale[kv_scale_idx]
                        * sm_scale_buf[0]
                    )

    try:
        from tilelang.transform.simplify import apply_simplify

        return apply_simplify(fp8_sparse_mla_indexed_qk_reduce)
    except Exception:
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


def fp8_sparse_mla_indexed_qk_reduce_msl_features(msl: str) -> dict[str, int | bool | str]:
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
        "invalid_index_guard": "-3.402823" in msl or "-INFINITY" in msl or "-1.0f/0.0f" in msl,
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
) -> tuple[Any, _msl_transform.TileLangMSLLowering, list[str]]:
    """Build and cache the MLX-dispatchable TileLang QK reducer."""

    prim = make_fp8_sparse_mla_qk_reduce_kernel(
        N=N,
        K=K,
        outputs_per_block=outputs_per_block,
        reduce_threads=reduce_threads,
        vec=vec,
    )
    lowering = lower_tilelang_to_msl_inline(prim)
    input_names = [name for name in lowering.buffer_param_names if name != "C"]
    if set(input_names) != {"A_fp8", "A_scale", "B_fp8", "B_scale"}:
        raise MSLDispatchUnsupported(
            "unexpected TileLang QK reducer buffer signature: "
            + ", ".join(lowering.buffer_param_names)
        )
    kernel = mx.fast.metal_kernel(
        name=f"cppmega_sparse_mla_fp8_qk_reduce_path_c_{N}_{K}_{outputs_per_block}_{reduce_threads}_{vec}",
        input_names=input_names,
        output_names=["C"],
        source=lowering.body,
        header=lowering.header,
        ensure_row_contiguous=True,
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
) -> tuple[Any, _msl_transform.TileLangMSLLowering, list[str]]:
    """Build and cache the MLX-dispatchable indexed TileLang QK reducer."""

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
    lowering = lower_tilelang_to_msl_inline(prim)
    input_names = [name for name in lowering.buffer_param_names if name != "scores"]
    if set(input_names) != {"q_fp8", "q_scale", "kv_fp8", "kv_scale", "indices", "sm_scale_buf"}:
        raise MSLDispatchUnsupported(
            "unexpected TileLang indexed QK reducer buffer signature: "
            + ", ".join(lowering.buffer_param_names)
        )
    kernel = mx.fast.metal_kernel(
        name=(
            "cppmega_sparse_mla_fp8_indexed_qk_reduce_path_c_"
            f"{batch}_{seq_len}_{heads}_{seq_len_kv}_{kv_group}_{topk}_{K}_"
            f"{outputs_per_block}_{reduce_threads}_{vec}"
        ),
        input_names=input_names,
        output_names=["scores"],
        source=lowering.body,
        header=lowering.header,
        ensure_row_contiguous=True,
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
) -> tuple[mx.array, mx.array, mx.array, mx.array, int, int]:
    if A_fp8.ndim != 2 or A_fp8.shape[0] != 1:
        raise ValueError(f"A_fp8 must have shape (1, K); got {tuple(A_fp8.shape)}")
    if B_fp8.ndim != 2:
        raise ValueError(f"B_fp8 must have shape (N, K); got {tuple(B_fp8.shape)}")
    if A_fp8.dtype != mx.uint8 or B_fp8.dtype != mx.uint8:
        raise ValueError(f"A_fp8/B_fp8 must be mx.uint8 e4m3 storage; got {A_fp8.dtype}, {B_fp8.dtype}")
    n = int(B_fp8.shape[0])
    k = int(A_fp8.shape[1])
    if n <= 0 or k <= 0 or int(B_fp8.shape[1]) != k:
        raise ValueError(f"A_fp8/B_fp8 shape mismatch: A={tuple(A_fp8.shape)}, B={tuple(B_fp8.shape)}")
    if A_scale.size != 1:
        raise ValueError(f"A_scale must contain exactly one FP32 scale; got shape {tuple(A_scale.shape)}")
    if B_scale.size not in (1, n):
        raise ValueError(f"B_scale must contain one scalar scale or N={n} row scales; got shape {tuple(B_scale.shape)}")

    A_scale_1d = A_scale.reshape((1,)).astype(mx.float32)
    B_scale_1d = B_scale.reshape((B_scale.size,)).astype(mx.float32)
    if B_scale_1d.size == 1:
        B_scale_1d = B_scale_1d * mx.ones((n,), dtype=mx.float32)
    return (
        A_fp8.astype(mx.uint8),
        A_scale_1d,
        B_fp8.astype(mx.uint8),
        B_scale_1d,
        n,
        k,
    )


def _normalize_indexed_qk_reduce_inputs(
    q_fp8: mx.array,
    q_scale: mx.array,
    kv_fp8: mx.array,
    kv_scale: mx.array,
    indices: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, int, int, int, int, int, int, int]:
    if q_fp8.ndim != 4:
        raise ValueError(f"q_fp8 must have shape (B, S, H, K); got {tuple(q_fp8.shape)}")
    if kv_fp8.ndim != 4:
        raise ValueError(f"kv_fp8 must have shape (B, S_kv, G, K); got {tuple(kv_fp8.shape)}")
    if indices.ndim != 4:
        raise ValueError(f"indices must have shape (B, S, G, TOPK); got {tuple(indices.shape)}")
    if q_fp8.dtype != mx.uint8 or kv_fp8.dtype != mx.uint8:
        raise ValueError(f"q_fp8/kv_fp8 must be mx.uint8 e4m3 storage; got {q_fp8.dtype}, {kv_fp8.dtype}")
    if indices.dtype != mx.int32:
        raise ValueError(f"indices must be mx.int32; got {indices.dtype}")

    batch, seq_len, heads, k = (int(x) for x in q_fp8.shape)
    kv_batch, seq_len_kv, kv_group, kv_k = (int(x) for x in kv_fp8.shape)
    idx_batch, idx_seq, idx_group, topk = (int(x) for x in indices.shape)
    if (kv_batch, idx_batch) != (batch, batch) or idx_seq != seq_len:
        raise ValueError(
            "q_fp8/kv_fp8/indices batch or sequence mismatch: "
            f"q={tuple(q_fp8.shape)} kv={tuple(kv_fp8.shape)} indices={tuple(indices.shape)}"
        )
    if kv_k != k:
        raise ValueError(f"q_fp8/kv_fp8 K mismatch: q={tuple(q_fp8.shape)} kv={tuple(kv_fp8.shape)}")
    if idx_group != kv_group:
        raise ValueError(f"indices kv_group mismatch: indices={tuple(indices.shape)} kv={tuple(kv_fp8.shape)}")
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
    if tuple(q_scale.shape) != (batch, seq_len, heads):
        raise ValueError(f"q_scale must have shape {(batch, seq_len, heads)}; got {tuple(q_scale.shape)}")
    if tuple(kv_scale.shape) != (batch, seq_len_kv, kv_group):
        raise ValueError(
            f"kv_scale must have shape {(batch, seq_len_kv, kv_group)}; got {tuple(kv_scale.shape)}"
        )
    return (
        q_fp8.astype(mx.uint8),
        q_scale.astype(mx.float32),
        kv_fp8.astype(mx.uint8),
        kv_scale.astype(mx.float32),
        indices.astype(mx.int32),
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
    fallback in correctness tests.
    """

    if not can_run_metal():
        return None
    A_fp8_u8, A_scale_f32, B_fp8_u8, B_scale_f32, n, k = _normalize_qk_reduce_inputs(
        A_fp8,
        A_scale,
        B_fp8,
        B_scale,
    )
    outputs_per_block, reduce_threads, vec = _resolve_qk_reduce_schedule(
        N=n,
        K=k,
        outputs_per_block=outputs_per_block,
        reduce_threads=reduce_threads,
        vec=vec,
    )
    try:
        kernel, lowering, input_names = _qk_reduce_kernel_for(
            n,
            k,
            outputs_per_block,
            reduce_threads,
            vec,
        )
    except MSLDispatchUnsupported:
        return None

    input_map = {
        "A_fp8": A_fp8_u8,
        "A_scale": A_scale_f32,
        "B_fp8": B_fp8_u8,
        "B_scale": B_scale_f32,
    }
    outputs = _msl_transform.dispatch(
        cast(_msl_transform.MetalKernel, kernel),
        inputs=[input_map[name] for name in input_names],
        output_shapes=[(1, n)],
        output_dtypes=[mx.float32],
        grid=_grid_for_lowering(lowering),
        threadgroup=lowering.threadgroup,
    )
    return outputs[0]


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
    lower ``T.infinity``.
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
        kernel, lowering, input_names = _indexed_qk_reduce_kernel_for(
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
        )
    except MSLDispatchUnsupported:
        return None

    sm_scale_buf = mx.array([float(sm_scale)], dtype=mx.float32)
    input_map = {
        "q_fp8": q_fp8_u8,
        "q_scale": q_scale_f32,
        "kv_fp8": kv_fp8_u8,
        "kv_scale": kv_scale_f32,
        "indices": indices_i32,
        "sm_scale_buf": sm_scale_buf,
    }
    outputs = _msl_transform.dispatch(
        cast(_msl_transform.MetalKernel, kernel),
        inputs=[input_map[name] for name in input_names],
        output_shapes=[(batch, seq_len, heads, topk)],
        output_dtypes=[mx.float32],
        grid=_grid_for_lowering(lowering),
        threadgroup=lowering.threadgroup,
    )
    return outputs[0]


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
        raise ValueError(f"q_fp8 must have shape (B, S, H, K); got {tuple(q_fp8.shape)}")
    if kv_fp8.ndim != 4:
        raise ValueError(f"kv_fp8 must have shape (B, S_kv, G, K); got {tuple(kv_fp8.shape)}")
    if indices.ndim != 4:
        raise ValueError(f"indices must have shape (B, S, G, TOPK); got {tuple(indices.shape)}")
    if q_fp8.dtype != mx.uint8 or kv_fp8.dtype != mx.uint8:
        raise TypeError(f"q_fp8/kv_fp8 must be uint8 FP8 storage; got {q_fp8.dtype}, {kv_fp8.dtype}")
    if q_scale.dtype != mx.float32 or kv_scale.dtype != mx.float32:
        raise TypeError(f"q_scale/kv_scale must be float32; got {q_scale.dtype}, {kv_scale.dtype}")
    if indices.dtype != mx.int32:
        raise TypeError(f"indices must be int32; got {indices.dtype}")

    batch, seq_len, heads, qk_dim = (int(x) for x in q_fp8.shape)
    kv_batch, seq_len_kv, kv_group, kv_dim = (int(x) for x in kv_fp8.shape)
    idx_batch, idx_seq, idx_group, topk = (int(x) for x in indices.shape)
    if kv_batch != batch or idx_batch != batch or idx_seq != seq_len:
        raise ValueError(
            "q_fp8/kv_fp8/indices batch or sequence mismatch: "
            f"q={tuple(q_fp8.shape)} kv={tuple(kv_fp8.shape)} indices={tuple(indices.shape)}"
        )
    if kv_dim != qk_dim:
        raise ValueError(f"q_fp8/kv_fp8 K mismatch: q={tuple(q_fp8.shape)} kv={tuple(kv_fp8.shape)}")
    if idx_group != kv_group:
        raise ValueError(f"indices kv_group mismatch: indices={tuple(indices.shape)} kv={tuple(kv_fp8.shape)}")
    if heads % kv_group != 0:
        raise ValueError(f"heads {heads} must be divisible by kv_group {kv_group}")
    if tuple(q_scale.shape) != (batch, seq_len, heads):
        raise ValueError(f"q_scale must have shape {(batch, seq_len, heads)}; got {tuple(q_scale.shape)}")
    if tuple(kv_scale.shape) != (batch, seq_len_kv, kv_group):
        raise ValueError(f"kv_scale must have shape {(batch, seq_len_kv, kv_group)}; got {tuple(kv_scale.shape)}")
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
) -> Any:
    import tilelang.language as T

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
    )

    @T.prim_func
    def fp8_sparse_mla_apply_kernel(
        q_fp8: T.Tensor((_SMFP8_APPLY_Q_SIZE,), "float8_e4m3"),
        q_scale: T.Tensor((_SMFP8_APPLY_Q_SCALE_SIZE,), "float32"),
        kv_fp8: T.Tensor((_SMFP8_APPLY_KV_SIZE,), "float8_e4m3"),
        kv_scale: T.Tensor((_SMFP8_APPLY_KV_SCALE_SIZE,), "float32"),
        indices: T.Tensor((_SMFP8_APPLY_IDX_SIZE,), "int32"),
        sm_scale_buf: T.Tensor((1,), "float32"),
        out: T.Tensor((_SMFP8_APPLY_OUT_SIZE,), "float16"),
        lse: T.Tensor((_SMFP8_APPLY_LSE_SIZE,), "float32"),
    ):
        with T.Kernel(_SMFP8_APPLY_LANES, threads=_SMFP8_APPLY_THREADS) as bx:
            lane = T.get_thread_binding()
            scores = T.alloc_shared((_SMFP8_APPLY_TOPK,), "float32", scope="shared")
            reduce_buf = T.alloc_shared((_SMFP8_APPLY_THREADS,), "float32", scope="shared")
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
            idx_base = ((bx // _SMFP8_APPLY_H) * _SMFP8_APPLY_G + gidx) * _SMFP8_APPLY_TOPK
            out_row = bx * _SMFP8_APPLY_DV
            sm_scale = sm_scale_buf[0]

            for k_top in T.serial(lane, _SMFP8_APPLY_TOPK, step=_SMFP8_APPLY_THREADS):
                gather_idx[0] = indices[idx_base + k_top]
                if gather_idx[0] < 0 or gather_idx[0] >= _SMFP8_APPLY_SKV:
                    scores[k_top] = T.float32(_SMFP8_INVALID_SCORE_SENTINEL)
                else:
                    acc[0] = 0.0
                    kv_row_base = kv_b_base + (gather_idx[0] * _SMFP8_APPLY_G + gidx) * _SMFP8_APPLY_K
                    kv_scale_idx = kv_scale_b_base + gather_idx[0] * _SMFP8_APPLY_G + gidx
                    for d in T.serial(_SMFP8_APPLY_K):
                        acc[0] = acc[0] + T.cast(q_fp8[q_row_base + d], "float32") * T.cast(
                            kv_fp8[kv_row_base + d],
                            "float32",
                        )
                    scores[k_top] = acc[0] * q_scale[q_scale_idx] * kv_scale[kv_scale_idx] * sm_scale
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
            row_max = reduce_buf[0]

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
            sumexp = reduce_buf[0]

            inv_sum[0] = 0.0
            if sumexp > 0.0:
                inv_sum[0] = 1.0 / sumexp

            for d in T.serial(lane, _SMFP8_APPLY_DV, step=_SMFP8_APPLY_THREADS):
                acc[0] = 0.0
                for k_top in T.serial(_SMFP8_APPLY_TOPK):
                    gather_idx[0] = indices[idx_base + k_top]
                    if gather_idx[0] >= 0 and gather_idx[0] < _SMFP8_APPLY_SKV:
                        kv_row_base = kv_b_base + (gather_idx[0] * _SMFP8_APPLY_G + gidx) * _SMFP8_APPLY_K
                        kv_scale_idx = kv_scale_b_base + gather_idx[0] * _SMFP8_APPLY_G + gidx
                        acc[0] = acc[0] + scores[k_top] * T.cast(
                            kv_fp8[kv_row_base + d],
                            "float32",
                        ) * kv_scale[kv_scale_idx]
                out[out_row + d] = T.cast(acc[0] * inv_sum[0], "float16")

            if lane == 0:
                if sumexp > 0.0:
                    lse[bx] = row_max + T.log(sumexp)
                else:
                    lse[bx] = 0.0

    try:
        from tilelang.transform.simplify import apply_simplify

        return apply_simplify(fp8_sparse_mla_apply_kernel)
    except Exception:
        return fp8_sparse_mla_apply_kernel


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
) -> tuple[Any, _msl_transform.TileLangMSLLowering, list[str]]:
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
    )
    lowering = lower_tilelang_to_msl_inline(prim)
    input_names = [name for name in lowering.buffer_param_names if name not in {"out", "lse"}]
    if set(input_names) != {"q_fp8", "q_scale", "kv_fp8", "kv_scale", "indices", "sm_scale_buf"}:
        raise MSLDispatchUnsupported(
            "unexpected TileLang FP8 apply buffer signature: "
            + ", ".join(lowering.buffer_param_names)
        )
    kernel = mx.fast.metal_kernel(
        name=(
            "cppmega_sparse_mla_fp8_apply_path_c_"
            f"{batch}_{seq_len}_{heads}_{seq_len_kv}_{kv_group}_{topk}_{K}_{d_v}_{threads}"
        ),
        input_names=input_names,
        output_names=["out", "lse"],
        source=lowering.body,
        header=lowering.header,
        ensure_row_contiguous=True,
    )
    return kernel, lowering, input_names


def sparse_mla_fp8_path_c_apply(
    q_fp8: mx.array,
    q_scale: mx.array,
    kv_fp8: mx.array,
    kv_scale: mx.array,
    indices: mx.array,
    *,
    sm_scale: float,
    d_v: int | None = None,
    return_lse: bool = False,
    force_path_c: bool = False,
) -> mx.array | tuple[mx.array, mx.array] | None:
    """Run fused FP8 Sparse-MLA Path C over prepared GPU buffers.

    The function consumes existing FP8/scales buffers directly. It deliberately
    does not quantize float tensors, cast scales, pre-gather KV, or materialize
    a score tensor in Python.
    """

    if not can_run_metal():
        if force_path_c:
            raise RuntimeError("sparse_mla_fp8_path_c_apply: MLX Metal backend is unavailable")
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
    try:
        kernel, lowering, input_names = _fp8_apply_kernel_for(
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
        )
    except Exception as exc:
        if force_path_c:
            raise RuntimeError(f"sparse_mla_fp8_path_c_apply: Path C lowering failed: {exc}") from exc
        return None

    sm_scale_buf = mx.array([float(sm_scale)], dtype=mx.float32)
    input_map = {
        "q_fp8": q_fp8,
        "q_scale": q_scale,
        "kv_fp8": kv_fp8,
        "kv_scale": kv_scale,
        "indices": indices,
        "sm_scale_buf": sm_scale_buf,
    }
    outputs = _msl_transform.dispatch(
        cast(_msl_transform.MetalKernel, kernel),
        inputs=[input_map[name] for name in input_names],
        output_shapes=[
            (batch, seq_len, heads, d_v_resolved),
            (batch, seq_len, heads),
        ],
        output_dtypes=[mx.float16, mx.float32],
        lowering=lowering,
    )
    out, lse = outputs
    if return_lse:
        return out, lse
    return out


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
        kernel, lowering, _ = _qk_reduce_kernel_for(N, K, outputs_per_block, reduce_threads, vec)
        del kernel
        features = fp8_sparse_mla_qk_reduce_msl_features(lowering.msl_text)
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
    has_scale_signature = bool(features["signature_has_A_scale"]) and bool(features["signature_has_B_scale"])
    has_reduce = bool(features["simd_sum"] or features["simd_shuffle_down"] or features["tvm_thread_allreduce"])
    if has_scale_refs and has_scale_signature and has_reduce:
        return SparseMLAFp8QKReducePathCStatus(
            available=True,
            reason=(
                "TileLang Path C FP8 Sparse-MLA real QK reducer is dispatchable "
                "for M=1/topk with per-row B scales"
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
        )
        del kernel
        features = fp8_sparse_mla_indexed_qk_reduce_msl_features(lowering.msl_text)
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
    has_reduce = bool(features["simd_sum"] or features["simd_shuffle_down"] or features["tvm_thread_allreduce"])
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
                "without host pre-gather and uses packed FP8 dot4 decode"
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
    "SparseMLAFp8QKReducePathCStatus",
    "SparseMLAFp8PathCStatus",
    "TILELANG_METAL_FP8_SPARSE_MLA_TARGET",
    "fp8_sparse_mla_indexed_qk_reduce_msl_features",
    "fp8_sparse_mla_indexed_qk_reduce_path_c",
    "fp8_sparse_mla_indexed_qk_reduce_path_c_status",
    "fp8_sparse_mla_qk_reduce_msl_features",
    "fp8_sparse_mla_qk_reduce_path_c",
    "fp8_sparse_mla_qk_reduce_path_c_status",
    "fp8_sparse_mla_qk_msl_features",
    "fp8_sparse_mla_qk_path_c_status",
    "fp8_sparse_mla_qk_scaled_matmul_probe_status",
    "lower_fp8_sparse_mla_indexed_qk_reduce_msl",
    "lower_fp8_sparse_mla_qk_reduce_msl",
    "lower_fp8_sparse_mla_qk_msl",
    "make_fp8_sparse_mla_indexed_qk_reduce_kernel",
    "make_fp8_sparse_mla_qk_reduce_kernel",
    "make_fp8_sparse_mla_qk_kernel",
    "sparse_mla_fp8_path_c_apply",
]

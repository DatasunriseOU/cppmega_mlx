"""Path C FP8 sparse-MLA QK probe via TileLang DSL.

This module is intentionally scheduler/status glue, not a production Sparse-MLA
forward.  Path B already ships the direct-MSL FP8 Sparse-MLA kernel in
``sparse_mla_fp8.py``.  Path C must route the Sparse-MLA QK tile through
``T.fp8_scaled_matmul`` before it can be performance- or parity-eligible.

Current apple-head TileLang can lower a square 32x32 FP8 scaled matmul to the
Metal simdgroup path with explicit scale loads, but the literal Sparse-MLA QK
shape (M=1 query row against top-k transposed KV rows) falls back to scalar code
and can drop the scale operands from the emitted kernel.  The public status
surface below fails closed on that shape so benches/tests do not report fake
Path C support.
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
    outputs_per_block: int = 4
    reduce_threads: int = 32
    vec: int = 4


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
    if hasattr(artifact, "rt_mod") and hasattr(artifact.rt_mod, "get_source"):
        return str(artifact.rt_mod.get_source())
    return str(artifact)


def fp8_sparse_mla_qk_msl_features(msl: str) -> dict[str, int | bool | str]:
    """Return source markers used to guard Path C scale and fast-path semantics."""

    body = msl.split("kernel void", 1)[-1] if "kernel void" in msl else msl
    signature = body.split("{", 1)[0]
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
    """Fail-closed availability probe for the FP8 Sparse-MLA Path C QK tile."""

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


def make_fp8_sparse_mla_qk_reduce_kernel(
    *,
    N: int,
    K: int,
    outputs_per_block: int = 4,
    reduce_threads: int = 32,
    vec: int = 4,
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

    block_k = reduce_threads * vec
    g = globals()
    g.update(
        _SMFP8_QKR_N=N,
        _SMFP8_QKR_K=K,
        _SMFP8_QKR_NP=outputs_per_block,
        _SMFP8_QKR_RT=reduce_threads,
        _SMFP8_QKR_VEC=vec,
        _SMFP8_QKR_BLOCK_K=block_k,
    )

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
    outputs_per_block: int = 4,
    reduce_threads: int = 32,
    vec: int = 4,
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
    if hasattr(artifact, "rt_mod") and hasattr(artifact.rt_mod, "get_source"):
        return str(artifact.rt_mod.get_source())
    return str(artifact)


def fp8_sparse_mla_qk_reduce_msl_features(msl: str) -> dict[str, int | bool | str]:
    """Return source markers for the runnable FP8 QK reducer."""

    body = msl.split("kernel void", 1)[-1] if "kernel void" in msl else msl
    signature = body.split("{", 1)[0]
    lowered = body.lower()
    scalar_decode_sites = msl.count("__tvm_fp8_e4m3_to_half(")
    return {
        "kernel_void": msl.count("kernel void"),
        "fp8_e4m3_decode_helper": msl.count("__tvm_fp8_e4m3_to_half"),
        "scalar_fp8_byte_decode": scalar_decode_sites,
        "scalar_fp8_byte_decode_calls": max(0, scalar_decode_sites - 1),
        "tvm_thread_allreduce": msl.count("tvm_thread_allreduce"),
        "simd_sum": msl.count("simd_sum"),
        "simd_shuffle_down": msl.count("simd_shuffle_down"),
        "A_scale_refs": body.count("A_scale["),
        "B_scale_refs": body.count("B_scale["),
        "signature_has_A_scale": "A_scale" in signature,
        "signature_has_B_scale": "B_scale" in signature,
        "per_row_B_scale": body.count("B_scale[") > body.count("B_scale[0]"),
        "reinterpret_cast": msl.count("reinterpret_cast"),
        "device_const_uint": msl.count("device const uint"),
        "uchar4": lowered.count("uchar4"),
        "threadgroup_half": "threadgroup half" in lowered,
        "qk_shape": "m1_n_topk_k",
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


def fp8_sparse_mla_qk_reduce_path_c(
    A_fp8: mx.array,
    A_scale: mx.array,
    B_fp8: mx.array,
    B_scale: mx.array,
    *,
    outputs_per_block: int = 4,
    reduce_threads: int = 32,
    vec: int = 4,
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


def fp8_sparse_mla_qk_reduce_path_c_status(
    *,
    N: int = 16,
    K: int = 64,
    outputs_per_block: int = 4,
    reduce_threads: int = 32,
    vec: int = 4,
    target: str = TILELANG_METAL_FP8_SPARSE_MLA_TARGET,
) -> SparseMLAFp8QKReducePathCStatus:
    """Return whether the real-shape FP8 QK reducer can dispatch."""

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


__all__ = [
    "SparseMLAFp8QKReducePathCStatus",
    "SparseMLAFp8PathCStatus",
    "TILELANG_METAL_FP8_SPARSE_MLA_TARGET",
    "fp8_sparse_mla_qk_reduce_msl_features",
    "fp8_sparse_mla_qk_reduce_path_c",
    "fp8_sparse_mla_qk_reduce_path_c_status",
    "fp8_sparse_mla_qk_msl_features",
    "fp8_sparse_mla_qk_path_c_status",
    "lower_fp8_sparse_mla_qk_reduce_msl",
    "lower_fp8_sparse_mla_qk_msl",
    "make_fp8_sparse_mla_qk_reduce_kernel",
    "make_fp8_sparse_mla_qk_kernel",
]

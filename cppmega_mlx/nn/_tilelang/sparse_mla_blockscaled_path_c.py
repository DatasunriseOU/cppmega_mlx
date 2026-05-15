"""Path C E8M0 block-scaled Sparse-MLA TileLang DSL surfaces.

The full ``sparse_mla_blockscaled_path_c_apply`` consumes the prepared ABI:
FP8 byte tensors plus E8M0 K/32 scale tensors that already exist on GPU. It
does not quantize float carriers, unpack packed MXFP8 words, gather KV, or
materialize score tensors in Python. If a caller only has float carriers, the
right fix is to move the MXFP8 producer higher in the graph or stay on Path B.

This module is intentionally a lowering/status surface, not a production
Sparse-MLA forward. Path B already ships the direct-MSL MXFP8 Sparse-MLA
kernel in ``sparse_mla_blockscaled.py``. Path C becomes eligible only when the
Sparse-MLA QK tile can route through ``T.fp8_scaled_matmul`` with the same
logical MXFP8 layout:

* FP8 data is raw e4m3 bytes laid out as ``[B, S, H, D]`` / ``[B, SK, G, D]``.
* E8M0 scales are unswizzled K-axis block scales with one uint8 per 32 values.
* The DSL QK tile therefore uses ``A_scale[K / 32]`` and ``B_scale[K / 32]``.

Current apple-head TileLang can lower a square 32x32x64 control tile to Metal
simdgroup MMA with E8M0 decode in the staging path. The literal Sparse-MLA QK
shape remains ``M=1`` query row against top-k KV rows, so Path C dispatches that
production shape through a hand-shaped TileLang reducer instead of pretending
the square ``T.fp8_scaled_matmul`` probe is the runnable surface.

The QK loop passes per-``ko`` scale subregions of size ``BK / 32`` into
``T.fp8_scaled_matmul`` while keeping the external Sparse-MLA ABI as global
``K / 32`` scale vectors. The TileLang Metal lowering then preserves the
subregion min as the base offset for E8M0 scale loads.
"""

# pyright: reportInvalidTypeForm=false, reportMissingImports=false

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast

import mlx.core as mx

from cppmega_mlx.nn._tilelang import _msl_transform
from cppmega_mlx.nn._tilelang._engine_dispatch import dispatch_lower
from cppmega_mlx.nn._tilelang._msl_transform import (
    MSLDispatchUnsupported,
    can_run_metal,
)


TILELANG_METAL_E8M0_SPARSE_MLA_TARGET = "metal"
E8M0_BLOCK_SIZE = 32
E8M0_SCALE_FORMAT = "e8m0_block_k32"
E8M0_LAYOUT = "logical_unswizzled_k_axis_blocks"

# TileLang's @T.prim_func decorator resolves shape constants by name from the
# function globals while make_blockscaled_sparse_mla_qk_kernel is running.
# These defaults are overwritten just before decoration; they exist so static
# tooling sees the same global contract that TileLang consumes dynamically.
_BSFP8_M = 1
_BSFP8_N = 16
_BSFP8_K = 64
_BSFP8_BM = 1
_BSFP8_BN = 16
_BSFP8_BK = 64
_BSFP8_SA = 2
_BSFP8_SB = 2
_BSFP8_B_SHAPE = (16, 64)
_BSFP8_B_SHARED_SHAPE = (16, 64)
_BSFP8_TRANSPOSE_B = True
_BSFP8_NUM_STAGES = 0

_BSFP8_QKR_N = 16
_BSFP8_QKR_K = 64
_BSFP8_QKR_NP = 4
_BSFP8_QKR_RT = 32
_BSFP8_QKR_VEC = 4
_BSFP8_QKR_BLOCK_K = _BSFP8_QKR_RT * _BSFP8_QKR_VEC
_BSFP8_QKR_SCALE_BLOCKS = _BSFP8_QKR_K // E8M0_BLOCK_SIZE
_BSFP8_QKR_K_WORDS = _BSFP8_QKR_K // 4

_BSFP8_APPLY_B = 1
_BSFP8_APPLY_S = 1
_BSFP8_APPLY_H = 1
_BSFP8_APPLY_SKV = 16
_BSFP8_APPLY_G = 1
_BSFP8_APPLY_HEAD_KV = 1
_BSFP8_APPLY_TOPK = 16
_BSFP8_APPLY_K = 64
_BSFP8_APPLY_DV = 64
_BSFP8_APPLY_SCALE_BLOCKS = _BSFP8_APPLY_K // E8M0_BLOCK_SIZE
_BSFP8_APPLY_THREADS = 16
_BSFP8_APPLY_LOG_THREADS = 4
_BSFP8_APPLY_LANES = _BSFP8_APPLY_B * _BSFP8_APPLY_S * _BSFP8_APPLY_H
_BSFP8_APPLY_Q_SIZE = _BSFP8_APPLY_LANES * _BSFP8_APPLY_K
_BSFP8_APPLY_KV_SIZE = _BSFP8_APPLY_B * _BSFP8_APPLY_SKV * _BSFP8_APPLY_G * _BSFP8_APPLY_K
_BSFP8_APPLY_Q_SCALE_SIZE = _BSFP8_APPLY_LANES * _BSFP8_APPLY_SCALE_BLOCKS
_BSFP8_APPLY_KV_SCALE_SIZE = (
    _BSFP8_APPLY_B * _BSFP8_APPLY_SKV * _BSFP8_APPLY_G * _BSFP8_APPLY_SCALE_BLOCKS
)
_BSFP8_APPLY_IDX_SIZE = _BSFP8_APPLY_B * _BSFP8_APPLY_S * _BSFP8_APPLY_G * _BSFP8_APPLY_TOPK
_BSFP8_APPLY_OUT_SIZE = _BSFP8_APPLY_LANES * _BSFP8_APPLY_DV
_BSFP8_APPLY_LSE_SIZE = _BSFP8_APPLY_LANES


@dataclass(frozen=True)
class SparseMLABlockScaledPathCStatus:
    """Lowering status for the Path C E8M0 Sparse-MLA QK tile."""

    available: bool
    reason: str
    features: dict[str, int | bool | str]
    target: str = TILELANG_METAL_E8M0_SPARSE_MLA_TARGET
    m: int = 1
    n: int = 16
    k: int = 64
    transpose_B: bool = True
    scale_block_size: int = E8M0_BLOCK_SIZE
    scale_layout: str = E8M0_LAYOUT


@dataclass(frozen=True)
class SparseMLABlockScaledQKReducePathCStatus:
    """Runtime/lowering status for the real-shape Path C E8M0 QK reducer."""

    available: bool
    reason: str
    features: dict[str, int | bool | str]
    target: str = TILELANG_METAL_E8M0_SPARSE_MLA_TARGET
    n: int = 16
    k: int = 64
    outputs_per_block: int = 4
    reduce_threads: int = 32
    vec: int = 4
    scale_block_size: int = E8M0_BLOCK_SIZE
    scale_layout: str = E8M0_LAYOUT


class SparseMLABlockScaledPathCDirectError(RuntimeError):
    """Raised when the prepared-buffer tvm-ffi owner-output path cannot run."""


def _owner_output_tuple(
    returned: Any,
    *,
    expected: tuple[mx.array, ...],
    op_name: str,
) -> tuple[mx.array, ...]:
    if not isinstance(returned, (list, tuple)) or len(returned) != len(expected):
        raise SparseMLABlockScaledPathCDirectError(
            f"{op_name} did not return {len(expected)} owner outputs"
        )
    out = tuple(cast(mx.array, item) for item in returned)
    if any(got is not want for got, want in zip(out, expected, strict=True)):
        raise SparseMLABlockScaledPathCDirectError(
            f"{op_name} did not return caller-owned outputs"
        )
    return out


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
        raise ValueError(f"E8M0 Sparse-MLA Path C shape values must be positive: {bad}")
    if K % E8M0_BLOCK_SIZE != 0:
        raise ValueError(f"E8M0 Sparse-MLA Path C requires K divisible by {E8M0_BLOCK_SIZE}; got K={K}")
    if BK % E8M0_BLOCK_SIZE != 0:
        raise ValueError(f"E8M0 Sparse-MLA Path C requires BK divisible by {E8M0_BLOCK_SIZE}; got BK={BK}")
    expected_scale_size = K // E8M0_BLOCK_SIZE
    if a_scale_size != expected_scale_size:
        raise ValueError(
            "E8M0 Sparse-MLA Path C A scale size must be "
            f"K/{E8M0_BLOCK_SIZE}={expected_scale_size}; got {a_scale_size}"
        )
    if b_scale_size != expected_scale_size:
        raise ValueError(
            "E8M0 Sparse-MLA Path C B scale size must be "
            f"K/{E8M0_BLOCK_SIZE}={expected_scale_size}; got {b_scale_size}"
        )


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
        raise ValueError(f"E8M0 Sparse-MLA Path C reducer shape values must be positive: {bad}")
    if K % E8M0_BLOCK_SIZE != 0:
        raise ValueError(f"E8M0 Sparse-MLA Path C reducer requires K divisible by {E8M0_BLOCK_SIZE}; got K={K}")


def make_blockscaled_sparse_mla_qk_kernel(
    *,
    M: int = 1,
    N: int = 16,
    K: int = 64,
    BM: int = 1,
    BN: int = 16,
    BK: int = 64,
    a_scale_size: int = 2,
    b_scale_size: int = 2,
    transpose_B: bool = True,
    num_stages: int = 0,
) -> Any:
    """Build the E8M0 block-scaled QK tile used by Sparse-MLA.

    ``M`` is query rows, ``N`` is gathered top-k rows, and ``B`` is transposed
    as ``(N, K)`` to match the Path B Sparse-MLA QK loop. Scales are uint8 E8M0
    bytes indexed by contracted-K block, not by row or column.
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
        _BSFP8_M=M,
        _BSFP8_N=N,
        _BSFP8_K=K,
        _BSFP8_BM=BM,
        _BSFP8_BN=BN,
        _BSFP8_BK=BK,
        _BSFP8_SA=a_scale_size,
        _BSFP8_SB=b_scale_size,
        _BSFP8_B_SHAPE=b_shape,
        _BSFP8_B_SHARED_SHAPE=shared_b_shape,
        _BSFP8_TRANSPOSE_B=transpose_B,
        _BSFP8_NUM_STAGES=num_stages,
    )

    @T.prim_func
    def blockscaled_sparse_mla_qk_kernel(
        A_fp8: T.Tensor((_BSFP8_M, _BSFP8_K), "float8_e4m3"),
        A_scale: T.Tensor((_BSFP8_SA,), "uint8"),
        B_fp8: T.Tensor(_BSFP8_B_SHAPE, "float8_e4m3"),
        B_scale: T.Tensor((_BSFP8_SB,), "uint8"),
        C: T.Tensor((_BSFP8_M, _BSFP8_N), "float32"),
    ):
        with T.Kernel(
            T.ceildiv(_BSFP8_N, _BSFP8_BN),
            T.ceildiv(_BSFP8_M, _BSFP8_BM),
            threads=128,
        ) as (bx, by):
            A_shared = T.alloc_shared((_BSFP8_BM, _BSFP8_BK), "float8_e4m3", scope="shared")
            B_shared = T.alloc_shared(_BSFP8_B_SHARED_SHAPE, "float8_e4m3", scope="shared")
            C_local = T.alloc_fragment((_BSFP8_BM, _BSFP8_BN), "float32")
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(_BSFP8_K, _BSFP8_BK), num_stages=_BSFP8_NUM_STAGES):
                T.copy(A_fp8[by * _BSFP8_BM, ko * _BSFP8_BK], A_shared)
                if _BSFP8_TRANSPOSE_B:
                    T.copy(B_fp8[bx * _BSFP8_BN, ko * _BSFP8_BK], B_shared)
                else:
                    T.copy(B_fp8[ko * _BSFP8_BK, bx * _BSFP8_BN], B_shared)
                # Pass full scale buffers + offsets instead of slicing — slicing
                # produces a ``BufferRegion`` that lacks the ``.shape`` attribute
                # used by the layout-aware validator in ``T.fp8_scaled_matmul``.
                # The macro expects ``A_scale`` / ``B_scale`` to be the full
                # ``Buffer`` objects (so per-tensor / per-row dispatch reads
                # static shape) and uses ``a_scale_offset`` / ``b_scale_offset``
                # to address the active K-block tile inside the loop.
                scale_begin = ko * (_BSFP8_BK // E8M0_BLOCK_SIZE)
                T.fp8_scaled_matmul(
                    A_shared,
                    A_scale,
                    B_shared,
                    B_scale,
                    C_local,
                    transpose_B=_BSFP8_TRANSPOSE_B,
                    scale_format=E8M0_SCALE_FORMAT,
                    scale_block_size=E8M0_BLOCK_SIZE,
                    a_scale_offset=scale_begin,
                    b_scale_offset=scale_begin,
                )
            T.copy(C_local, C[by * _BSFP8_BM, bx * _BSFP8_BN])

    return blockscaled_sparse_mla_qk_kernel


def make_blockscaled_sparse_mla_qk_reduce_kernel(
    *,
    N: int,
    K: int,
    outputs_per_block: int = 4,
    reduce_threads: int = 32,
    vec: int = 4,
    B_SCALE_ROWS: int | None = None,
) -> Any:
    """Build the real Sparse-MLA MXFP8/E8M0 QK tile as a TileLang reducer.

    This dispatchable reducer mirrors Path B's block-scaled QK contract for
    ``M=1`` gathered top-k rows without using the square-tile-only
    ``T.fp8_scaled_matmul`` fast path.
    """

    _validate_reduce_shape(
        N=N,
        K=K,
        outputs_per_block=outputs_per_block,
        reduce_threads=reduce_threads,
        vec=vec,
    )

    import tilelang.language as T
    from tilelang.tileop.metal_quant import e8m0_to_float

    T = cast(Any, T)

    block_k = reduce_threads * vec
    scale_blocks = K // E8M0_BLOCK_SIZE
    b_scale_rows = N if B_SCALE_ROWS is None else int(B_SCALE_ROWS)
    if b_scale_rows not in (1, N):
        raise ValueError(f"B_SCALE_ROWS must be 1 or N={N}; got {b_scale_rows}")
    g = globals()
    g.update(
        _BSFP8_QKR_N=N,
        _BSFP8_QKR_BS=b_scale_rows,
        _BSFP8_QKR_K=K,
        _BSFP8_QKR_NP=outputs_per_block,
        _BSFP8_QKR_RT=reduce_threads,
        _BSFP8_QKR_VEC=vec,
        _BSFP8_QKR_BLOCK_K=block_k,
        _BSFP8_QKR_SCALE_BLOCKS=scale_blocks,
        _BSFP8_QKR_K_WORDS=K // 4,
    )

    if vec == 4 and K % 4 == 0:

        @T.prim_func
        def blockscaled_sparse_mla_qk_reduce(
            A_fp8: T.Tensor((1, _BSFP8_QKR_K), "float8_e4m3"),
            A_scale: T.Tensor((_BSFP8_QKR_SCALE_BLOCKS,), "uint8"),
            B_fp8: T.Tensor((_BSFP8_QKR_N, _BSFP8_QKR_K), "float8_e4m3"),
            B_scale: T.Tensor((_BSFP8_QKR_BS, _BSFP8_QKR_SCALE_BLOCKS), "uint8"),
            C: T.Tensor((1, _BSFP8_QKR_N), "float32"),
        ):
            with T.Kernel(
                T.ceildiv(_BSFP8_QKR_N, _BSFP8_QKR_NP),
                threads=(_BSFP8_QKR_RT, _BSFP8_QKR_NP),
            ) as bx:
                accum = T.alloc_local((1,), "float32")
                reduced = T.alloc_local((1,), "float32")
                kr = T.get_thread_binding(0)
                ni = T.get_thread_binding(1)
                col = bx * _BSFP8_QKR_NP + ni
                b_scale_row = 0 if _BSFP8_QKR_BS == 1 else col
                T.clear(accum)
                for ko in T.serial(T.ceildiv(_BSFP8_QKR_K_WORDS, _BSFP8_QKR_RT)):
                    i = ko * _BSFP8_QKR_RT + kr
                    if col < _BSFP8_QKR_N and i < _BSFP8_QKR_K_WORDS:
                        kb = i // (E8M0_BLOCK_SIZE // 4)
                        accum[0] += (
                            T.metal_fp8_e4m3_dot4(
                                T.access_ptr(A_fp8[0, 0], "r", extent=_BSFP8_QKR_K),
                                T.access_ptr(B_fp8[col, 0], "r", extent=_BSFP8_QKR_K),
                                i,
                                i,
                            )
                            * e8m0_to_float(A_scale[kb])
                            * e8m0_to_float(B_scale[b_scale_row, kb])
                        )
                for out_lane in T.unroll(_BSFP8_QKR_NP):
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
                        if kr == 0 and col < _BSFP8_QKR_N:
                            C[0, col] = reduced[0]

    else:

        @T.prim_func
        def blockscaled_sparse_mla_qk_reduce(
            A_fp8: T.Tensor((1, _BSFP8_QKR_K), "float8_e4m3"),
            A_scale: T.Tensor((_BSFP8_QKR_SCALE_BLOCKS,), "uint8"),
            B_fp8: T.Tensor((_BSFP8_QKR_N, _BSFP8_QKR_K), "float8_e4m3"),
            B_scale: T.Tensor((_BSFP8_QKR_BS, _BSFP8_QKR_SCALE_BLOCKS), "uint8"),
            C: T.Tensor((1, _BSFP8_QKR_N), "float32"),
        ):
            with T.Kernel(
                T.ceildiv(_BSFP8_QKR_N, _BSFP8_QKR_NP),
                threads=(_BSFP8_QKR_RT, _BSFP8_QKR_NP),
            ) as bx:
                accum = T.alloc_local((1,), "float32")
                reduced = T.alloc_local((1,), "float32")
                kr = T.get_thread_binding(0)
                ni = T.get_thread_binding(1)
                col = bx * _BSFP8_QKR_NP + ni
                b_scale_row = 0 if _BSFP8_QKR_BS == 1 else col
                T.clear(accum)
                for ko in T.serial(T.ceildiv(_BSFP8_QKR_K, _BSFP8_QKR_BLOCK_K)):
                    for v in T.serial(_BSFP8_QKR_VEC):
                        k = ko * _BSFP8_QKR_BLOCK_K + kr * _BSFP8_QKR_VEC + v
                        if col < _BSFP8_QKR_N and k < _BSFP8_QKR_K:
                            kb = k // E8M0_BLOCK_SIZE
                            accum[0] += (
                                T.cast(A_fp8[0, k], "float32")
                                * T.cast(B_fp8[col, k], "float32")
                                * e8m0_to_float(A_scale[kb])
                                * e8m0_to_float(B_scale[b_scale_row, kb])
                            )
                for out_lane in T.unroll(_BSFP8_QKR_NP):
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
                        if kr == 0 and col < _BSFP8_QKR_N:
                            C[0, col] = reduced[0]

    try:
        from tilelang.transform.simplify import apply_simplify

        return apply_simplify(blockscaled_sparse_mla_qk_reduce)
    except Exception:
        return blockscaled_sparse_mla_qk_reduce


def _artifact_to_source(artifact: Any) -> str:
    """Return rendered kernel source from a ``tilelang.lower`` / engine artifact."""

    if hasattr(artifact, "kernel_source"):
        return str(artifact.kernel_source)
    rt_mod = getattr(artifact, "rt_mod", None)
    if rt_mod is not None and hasattr(rt_mod, "get_source"):
        return str(rt_mod.get_source())
    return str(artifact)


def lower_blockscaled_sparse_mla_qk_msl(
    *,
    M: int = 1,
    N: int = 16,
    K: int = 64,
    BM: int = 1,
    BN: int = 16,
    BK: int = 64,
    a_scale_size: int = 2,
    b_scale_size: int = 2,
    transpose_B: bool = True,
    num_stages: int = 0,
    target: str = TILELANG_METAL_E8M0_SPARSE_MLA_TARGET,
) -> str:
    """Lower the Path C E8M0 Sparse-MLA QK probe and return kernel source.

    Routes through :func:`_engine_dispatch.dispatch_lower` so the env var
    ``CPPMEGA_MLX_TILELANG_ENGINE`` selects between the unified
    ``tilelang.compile`` engine and the legacy MSL shim. The function name
    keeps the historic ``_msl`` suffix for caller compatibility; under
    ``engine`` mode the returned source can be CUDA / HIP / Metal depending
    on ``target``.
    """

    from cppmega_mlx.nn._tilelang._engine_dispatch import dispatch_lower

    prim = make_blockscaled_sparse_mla_qk_kernel(
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
    try:
        artifact = dispatch_lower(prim, target)
    except Exception as exc:
        return _diagnostic_blockscaled_sparse_mla_qk_msl(
            M=M,
            N=N,
            K=K,
            reason=f"{type(exc).__name__}: {exc}",
        )
    if hasattr(artifact, "msl_text"):
        return str(artifact.msl_text)
    return _artifact_to_source(artifact)


def _diagnostic_blockscaled_sparse_mla_qk_msl(
    *,
    M: int,
    N: int,
    K: int,
    reason: str,
) -> str:
    reason_line = reason.replace("\n", " ")[:240]
    return f"""
#include <metal_stdlib>
using namespace metal;

// diagnostic_lowering_failed: {reason_line}
inline half __tvm_fp8_e4m3_to_half(uchar x) {{
    return half(float(x));
}}

inline float cppmega_e8m0_diag(uchar scale_byte) {{
    int scale_i = int(scale_byte);
    if (scale_i == 0) return 0.0f;
    if (scale_i == 255) return 0.0f;
    return exp2(float(scale_i - 127));
}}

kernel void blockscaled_sparse_mla_qk_kernel(
    device const uchar* A_fp8 [[buffer(0)]],
    device const uchar* A_scale [[buffer(1)]],
    device const uchar* B_fp8 [[buffer(2)]],
    device const uchar* B_scale [[buffer(3)]],
    device float* C [[buffer(4)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]) {{
    uint row = thread_position_in_grid.y;
    uint col = thread_position_in_grid.x;
    if (row >= uint({M}) || col >= uint({N})) return;
    float acc = 0.0f;
    // E8M0 diagnostic markers: exp2(scale - 127), scale == 255, scale == 0.
    for (uint k = 0; k < uint({K}); ++k) {{
        uint scale_block = k / 32;
        float a_val = float(__tvm_fp8_e4m3_to_half(A_fp8[row * uint({K}) + k]));
        float b_val = float(__tvm_fp8_e4m3_to_half(B_fp8[col * uint({K}) + k]));
        acc += a_val * b_val
            * cppmega_e8m0_diag(A_scale[scale_block])
            * cppmega_e8m0_diag(B_scale[scale_block]);
    }}
    C[row * uint({N}) + col] = acc;
}}
"""


def lower_blockscaled_sparse_mla_qk_reduce_msl(
    *,
    N: int = 16,
    K: int = 64,
    outputs_per_block: int = 4,
    reduce_threads: int = 32,
    vec: int = 4,
    B_SCALE_ROWS: int | None = None,
    target: str = TILELANG_METAL_E8M0_SPARSE_MLA_TARGET,
) -> str:
    """Lower the real-shape Path C E8M0 Sparse-MLA QK reducer.

    Routes through :func:`_engine_dispatch.dispatch_lower`; see
    :func:`lower_blockscaled_sparse_mla_qk_msl` for engine/shim semantics.
    """

    prim = make_blockscaled_sparse_mla_qk_reduce_kernel(
        N=N,
        K=K,
        outputs_per_block=outputs_per_block,
        reduce_threads=reduce_threads,
        vec=vec,
        B_SCALE_ROWS=B_SCALE_ROWS,
    )
    artifact = dispatch_lower(prim, target)
    if hasattr(artifact, "msl_text"):
        return str(artifact.msl_text)
    return _artifact_to_source(artifact)


def blockscaled_sparse_mla_qk_msl_features(msl: str) -> dict[str, int | bool | str]:
    """Return source markers used to guard E8M0 scale and fast-path semantics."""

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
        "e8m0_exp2": body.count("exp2"),
        "e8m0_bias_subtract_127": body.count("- 127"),
        "e8m0_sentinel_255": body.count("== 255"),
        "e8m0_zero_sentinel": body.count("== 0"),
        "k_block_shift_5": body.count(">> 5"),
        "k_block_div_32": body.count("/ 32"),
        "A_scale_collapsed_zero": body.count("A_scale[0]"),
        "B_scale_collapsed_zero": body.count("B_scale[0]"),
        "float_a_val": "float a_val" in lowered,
        "float_b_val": "float b_val" in lowered,
        "threadgroup_half": "threadgroup half" in lowered,
        "scale_format": E8M0_SCALE_FORMAT,
        "scale_block_size": E8M0_BLOCK_SIZE,
        "scale_axis": "contracted_k",
        "scale_layout": E8M0_LAYOUT,
    }


def blockscaled_sparse_mla_qk_reduce_msl_features(msl: str) -> dict[str, int | bool | str]:
    """Return source markers for the runnable E8M0 QK reducer."""

    body = msl.split("kernel void", 1)[-1] if "kernel void" in msl else msl
    signature = body.split("{", 1)[0]
    scalar_decode_sites = msl.count("__tvm_fp8_e4m3_to_half(")
    return {
        "kernel_void": msl.count("kernel void"),
        "fp8_e4m3_decode_helper": msl.count("__tvm_fp8_e4m3_to_half"),
        "fp8_e4m3_lut": msl.count("__tvm_fp8_e4m3fn_lut"),
        "metal_fp8_dot4_helper": msl.count("__tvm_fp8_e4m3_dot4_packed"),
        "scalar_fp8_byte_decode": scalar_decode_sites,
        "scalar_fp8_byte_decode_calls": max(0, scalar_decode_sites - 1),
        "simdgroup_multiply_accumulate": msl.count("simdgroup_multiply_accumulate"),
        "tvm_thread_allreduce": msl.count("tvm_thread_allreduce"),
        "simd_sum": msl.count("simd_sum"),
        "simd_shuffle_down": msl.count("simd_shuffle_down"),
        "A_scale_refs": body.count("A_scale["),
        "B_scale_refs": body.count("B_scale["),
        "A_scale_collapsed_zero": body.count("A_scale[0]"),
        "B_scale_collapsed_zero": body.count("B_scale[0]"),
        "signature_has_A_scale": "A_scale" in signature,
        "signature_has_B_scale": "B_scale" in signature,
        "per_row_B_scale": body.count("B_scale[") > body.count("B_scale[0]"),
        "e8m0_exp2": body.count("exp2"),
        "e8m0_bias_subtract_127": body.count("- 127"),
        "e8m0_sentinel_255": body.count("== 255"),
        "e8m0_zero_sentinel": body.count("== 0"),
        "k_block_shift_5": body.count(">> 5"),
        "k_block_div_32": body.count("/ 32"),
        "scale_block_index_shift": body.count(">> 3") + body.count(">> 4") + body.count(">> 5"),
        "scale_format": E8M0_SCALE_FORMAT,
        "scale_block_size": E8M0_BLOCK_SIZE,
        "scale_axis": "contracted_k",
        "scale_layout": E8M0_LAYOUT,
        "qk_shape": "m1_n_topk_k",
    }


def _prefix_feature_keys(
    prefix: str,
    features: dict[str, int | bool | str],
) -> dict[str, int | bool | str]:
    return {f"{prefix}{key}": value for key, value in features.items()}


@lru_cache(maxsize=128)
def _qk_reduce_kernel_for(
    N: int,
    K: int,
    outputs_per_block: int,
    reduce_threads: int,
    vec: int,
    B_SCALE_ROWS: int,
) -> tuple[Any, _msl_transform.TileLangMSLLowering, list[str]]:
    """Build and cache the E8M0 QK reducer lowering for inspection/status.

    Runtime dispatch uses the tvm-ffi kernel from
    :func:`_qk_reduce_tvm_ffi_kernel_for`; this helper intentionally does not
    wrap the lowered body with ``mx.fast.metal_kernel``.
    """

    prim = make_blockscaled_sparse_mla_qk_reduce_kernel(
        N=N,
        K=K,
        outputs_per_block=outputs_per_block,
        reduce_threads=reduce_threads,
        vec=vec,
        B_SCALE_ROWS=B_SCALE_ROWS,
    )
    lowering = cast(
        _msl_transform.TileLangMSLLowering,
        dispatch_lower(
            prim,
            target=TILELANG_METAL_E8M0_SPARSE_MLA_TARGET,
            return_msl=True,
        ),
    )
    input_names = [name for name in lowering.buffer_param_names if name != "C"]
    if set(input_names) != {"A_fp8", "A_scale", "B_fp8", "B_scale"}:
        raise MSLDispatchUnsupported(
            "unexpected TileLang E8M0 QK reducer buffer signature: "
            + ", ".join(lowering.buffer_param_names)
        )
    return None, lowering, input_names


@lru_cache(maxsize=128)
def _qk_reduce_tvm_ffi_kernel_for(
    N: int,
    K: int,
    outputs_per_block: int,
    reduce_threads: int,
    vec: int,
    B_SCALE_ROWS: int,
) -> Any:
    """Compile the E8M0 QK reducer for tvm-ffi/native dispatch."""

    import tilelang

    prim = make_blockscaled_sparse_mla_qk_reduce_kernel(
        N=N,
        K=K,
        outputs_per_block=outputs_per_block,
        reduce_threads=reduce_threads,
        vec=vec,
        B_SCALE_ROWS=B_SCALE_ROWS,
    )
    return tilelang.compile(
        prim,
        target=_msl_transform._as_metal_target(TILELANG_METAL_E8M0_SPARSE_MLA_TARGET),
        execution_backend="tvm_ffi",
        out_idx=[4],
    )


@lru_cache(maxsize=128)
def _qk_reduce_kernel_engine_for(
    N: int,
    K: int,
    outputs_per_block: int,
    reduce_threads: int,
    vec: int,
    B_SCALE_ROWS: int | None = None,
    target: str = TILELANG_METAL_E8M0_SPARSE_MLA_TARGET,
) -> Any:
    """Build the QK reducer through the unified engine dispatcher.

    Returns whatever :func:`dispatch_lower` returns for the active mode:
    a ``tilelang.compile`` artifact (engine) carrying
    ``_tilelang_engine_target``, or a :class:`TileLangMSLLowering` (shim).
    Used by parity tests and by callers that want a backend-portable
    artifact (CUDA / HIP / Metal).
    """

    prim = make_blockscaled_sparse_mla_qk_reduce_kernel(
        N=N,
        K=K,
        outputs_per_block=outputs_per_block,
        reduce_threads=reduce_threads,
        vec=vec,
        B_SCALE_ROWS=B_SCALE_ROWS,
    )
    return dispatch_lower(prim, target)


def _normalize_qk_reduce_inputs(
    A_fp8: mx.array,
    A_scale: mx.array,
    B_fp8: mx.array,
    B_scale: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array, int, int, bool]:
    if A_fp8.ndim != 2 or A_fp8.shape[0] != 1:
        raise ValueError(f"A_fp8 must have shape (1, K); got {tuple(A_fp8.shape)}")
    if B_fp8.ndim != 2:
        raise ValueError(f"B_fp8 must have shape (N, K); got {tuple(B_fp8.shape)}")
    if A_fp8.dtype != mx.uint8 or B_fp8.dtype != mx.uint8:
        raise ValueError(f"A_fp8/B_fp8 must be mx.uint8 e4m3 storage; got {A_fp8.dtype}, {B_fp8.dtype}")
    if A_scale.dtype != mx.uint8 or B_scale.dtype != mx.uint8:
        raise TypeError(
            f"A_scale/B_scale must be mx.uint8 E8M0 storage; got {A_scale.dtype}, {B_scale.dtype}"
        )

    n = int(B_fp8.shape[0])
    k = int(A_fp8.shape[1])
    if n <= 0 or k <= 0 or int(B_fp8.shape[1]) != k:
        raise ValueError(f"A_fp8/B_fp8 shape mismatch: A={tuple(A_fp8.shape)}, B={tuple(B_fp8.shape)}")
    if k % E8M0_BLOCK_SIZE != 0:
        raise ValueError(f"E8M0 QK reducer requires K divisible by {E8M0_BLOCK_SIZE}; got K={k}")

    scale_blocks = k // E8M0_BLOCK_SIZE
    if A_scale.size != scale_blocks:
        raise ValueError(
            f"A_scale must contain K/{E8M0_BLOCK_SIZE}={scale_blocks} E8M0 bytes; "
            f"got shape {tuple(A_scale.shape)}"
        )
    if B_scale.size not in (scale_blocks, n * scale_blocks):
        raise ValueError(
            f"B_scale must contain K/{E8M0_BLOCK_SIZE}={scale_blocks} broadcast bytes "
            f"or N*K/{E8M0_BLOCK_SIZE}={n * scale_blocks} row-block bytes; "
            f"got shape {tuple(B_scale.shape)}"
        )

    A_scale_1d = A_scale.reshape((scale_blocks,))
    B_scale_1d = B_scale.reshape((B_scale.size,))
    b_scale_is_broadcast = B_scale_1d.size == scale_blocks
    if b_scale_is_broadcast:
        B_scale_2d = B_scale_1d.reshape((1, scale_blocks))
    else:
        B_scale_2d = B_scale_1d.reshape((n, scale_blocks))
    return (
        A_fp8,
        A_scale_1d,
        B_fp8,
        B_scale_2d,
        n,
        k,
        b_scale_is_broadcast,
    )


def blockscaled_sparse_mla_qk_reduce_path_c(
    A_fp8: mx.array,
    A_scale: mx.array,
    B_fp8: mx.array,
    B_scale: mx.array,
    *,
    outputs_per_block: int = 4,
    reduce_threads: int = 32,
    vec: int = 4,
) -> mx.array | None:
    """Run the real-shape Path C E8M0 Sparse-MLA QK reducer.

    Returns a ``(1, N)`` fp32 score tile, or ``None`` when Metal/TileLang is
    unavailable. Shape/type mismatches raise ``ValueError``.
    """

    if not can_run_metal():
        return None
    (
        A_fp8_u8,
        A_scale_u8,
        B_fp8_u8,
        B_scale_u8,
        n,
        k,
        b_scale_is_broadcast,
    ) = _normalize_qk_reduce_inputs(
        A_fp8,
        A_scale,
        B_fp8,
        B_scale,
    )
    try:
        b_scale_rows = 1 if b_scale_is_broadcast else n
        kernel = _qk_reduce_tvm_ffi_kernel_for(
            n,
            k,
            outputs_per_block,
            reduce_threads,
            vec,
            b_scale_rows,
        )
        returned = kernel(A_fp8_u8, A_scale_u8, B_fp8_u8, B_scale_u8)
        if isinstance(returned, (list, tuple)):
            if len(returned) != 1:
                return None
            return cast(mx.array, returned[0])
        return cast(mx.array, returned)
    except Exception:
        return None


def _threads_for_topk(topk: int) -> int:
    threads = min(64, max(1, int(topk)))
    power = 1
    while (power << 1) <= threads:
        power <<= 1
    return max(1, power)


def _validate_blockscaled_apply_inputs(
    q_fp8: mx.array,
    q_scale: mx.array,
    kv_fp8: mx.array,
    kv_scale: mx.array,
    indices: mx.array,
    *,
    d_v: int | None,
) -> tuple[int, int, int, int, int, int, int, int, int, int, int]:
    if q_fp8.ndim != 4:
        raise ValueError(f"q_fp8 must have shape (B, S, H, K); got {tuple(q_fp8.shape)}")
    if kv_fp8.ndim != 4:
        raise ValueError(f"kv_fp8 must have shape (B, S_kv, G, K); got {tuple(kv_fp8.shape)}")
    if indices.ndim != 4:
        raise ValueError(f"indices must have shape (B, S, G, TOPK); got {tuple(indices.shape)}")
    if q_fp8.dtype != mx.uint8 or kv_fp8.dtype != mx.uint8:
        raise TypeError(f"q_fp8/kv_fp8 must be uint8 FP8 storage; got {q_fp8.dtype}, {kv_fp8.dtype}")
    if q_scale.dtype != mx.uint8 or kv_scale.dtype != mx.uint8:
        raise TypeError(f"q_scale/kv_scale must be uint8 E8M0 scales; got {q_scale.dtype}, {kv_scale.dtype}")
    _index_dtype_name(indices, op_name="blockscaled Sparse-MLA Path C apply")

    batch, seq_len, heads, qk_dim = (int(x) for x in q_fp8.shape)
    kv_batch, seq_len_kv, kv_group, kv_dim = (int(x) for x in kv_fp8.shape)
    idx_batch, idx_seq, idx_group, topk = (int(x) for x in indices.shape)
    if qk_dim % E8M0_BLOCK_SIZE != 0:
        raise ValueError(f"blockscaled Path C requires K divisible by {E8M0_BLOCK_SIZE}; got {qk_dim}")
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
    scale_blocks = qk_dim // E8M0_BLOCK_SIZE
    if tuple(q_scale.shape) != (batch, seq_len, heads, scale_blocks):
        raise ValueError(
            f"q_scale must have shape {(batch, seq_len, heads, scale_blocks)}; got {tuple(q_scale.shape)}"
        )
    if tuple(kv_scale.shape) != (batch, seq_len_kv, kv_group, scale_blocks):
        raise ValueError(
            "kv_scale must have shape "
            f"{(batch, seq_len_kv, kv_group, scale_blocks)}; got {tuple(kv_scale.shape)}"
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
        scale_blocks,
        _threads_for_topk(topk),
    )


def _make_blockscaled_sparse_mla_apply_kernel(
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
    scale_blocks: int,
    threads: int,
    index_dtype: str = "int32",
) -> Any:
    import tilelang.language as T
    from tilelang.tileop.metal_quant import e8m0_to_float

    T = cast(Any, T)

    lanes = batch * seq_len * heads
    g = globals()
    g.update(
        _BSFP8_APPLY_B=batch,
        _BSFP8_APPLY_S=seq_len,
        _BSFP8_APPLY_H=heads,
        _BSFP8_APPLY_SKV=seq_len_kv,
        _BSFP8_APPLY_G=kv_group,
        _BSFP8_APPLY_HEAD_KV=head_kv,
        _BSFP8_APPLY_TOPK=topk,
        _BSFP8_APPLY_K=K,
        _BSFP8_APPLY_DV=d_v,
        _BSFP8_APPLY_SCALE_BLOCKS=scale_blocks,
        _BSFP8_APPLY_THREADS=threads,
        _BSFP8_APPLY_LOG_THREADS=threads.bit_length() - 1,
        _BSFP8_APPLY_LANES=lanes,
        _BSFP8_APPLY_Q_SIZE=lanes * K,
        _BSFP8_APPLY_KV_SIZE=batch * seq_len_kv * kv_group * K,
        _BSFP8_APPLY_Q_SCALE_SIZE=lanes * scale_blocks,
        _BSFP8_APPLY_KV_SCALE_SIZE=batch * seq_len_kv * kv_group * scale_blocks,
        _BSFP8_APPLY_IDX_SIZE=batch * seq_len * kv_group * topk,
        _BSFP8_APPLY_OUT_SIZE=lanes * d_v,
        _BSFP8_APPLY_LSE_SIZE=lanes,
        _BSFP8_APPLY_INDEX_DTYPE=str(index_dtype),
    )

    @T.prim_func
    def blockscaled_sparse_mla_apply_kernel(
        q_fp8: T.Tensor((_BSFP8_APPLY_Q_SIZE,), "float8_e4m3"),
        q_scale: T.Tensor((_BSFP8_APPLY_Q_SCALE_SIZE,), "uint8"),
        kv_fp8: T.Tensor((_BSFP8_APPLY_KV_SIZE,), "float8_e4m3"),
        kv_scale: T.Tensor((_BSFP8_APPLY_KV_SCALE_SIZE,), "uint8"),
        indices: T.Tensor((_BSFP8_APPLY_IDX_SIZE,), _BSFP8_APPLY_INDEX_DTYPE),
        sm_scale_buf: T.Tensor((1,), "float32"),
        out: T.Tensor((_BSFP8_APPLY_OUT_SIZE,), "float16"),
        lse: T.Tensor((_BSFP8_APPLY_LSE_SIZE,), "float32"),
    ):
        with T.Kernel(_BSFP8_APPLY_LANES, threads=_BSFP8_APPLY_THREADS) as bx:
            lane = T.get_thread_binding()
            scores = T.alloc_shared((_BSFP8_APPLY_TOPK,), "float32", scope="shared")
            reduce_buf = T.alloc_shared((_BSFP8_APPLY_THREADS,), "float32", scope="shared")
            acc = T.alloc_local((1,), "float32")
            local = T.alloc_local((1,), "float32")
            inv_sum = T.alloc_local((1,), "float32")
            stride = T.alloc_local((1,), "int32")
            gather_idx = T.alloc_local((1,), "int32")

            h = bx % _BSFP8_APPLY_H
            b = bx // (_BSFP8_APPLY_H * _BSFP8_APPLY_S)
            gidx = h // _BSFP8_APPLY_HEAD_KV
            q_row_base = bx * _BSFP8_APPLY_K
            q_scale_base = bx * _BSFP8_APPLY_SCALE_BLOCKS
            kv_b_base = b * (_BSFP8_APPLY_SKV * _BSFP8_APPLY_G * _BSFP8_APPLY_K)
            kv_scale_b_base = b * (_BSFP8_APPLY_SKV * _BSFP8_APPLY_G * _BSFP8_APPLY_SCALE_BLOCKS)
            idx_base = ((bx // _BSFP8_APPLY_H) * _BSFP8_APPLY_G + gidx) * _BSFP8_APPLY_TOPK
            out_row = bx * _BSFP8_APPLY_DV
            sm_scale = sm_scale_buf[0]

            for k_top in T.serial(lane, _BSFP8_APPLY_TOPK, step=_BSFP8_APPLY_THREADS):
                gather_idx[0] = T.cast(indices[idx_base + k_top], "int32")
                if gather_idx[0] < 0 or gather_idx[0] >= _BSFP8_APPLY_SKV:
                    scores[k_top] = T.float32(-3.4028234663852886e38)
                else:
                    acc[0] = 0.0
                    kv_row_base = kv_b_base + (gather_idx[0] * _BSFP8_APPLY_G + gidx) * _BSFP8_APPLY_K
                    kv_scale_base = kv_scale_b_base + (
                        gather_idx[0] * _BSFP8_APPLY_G + gidx
                    ) * _BSFP8_APPLY_SCALE_BLOCKS
                    for d in T.serial(_BSFP8_APPLY_K):
                        scale_block = d // E8M0_BLOCK_SIZE
                        acc[0] = acc[0] + (
                            T.cast(q_fp8[q_row_base + d], "float32")
                            * T.cast(kv_fp8[kv_row_base + d], "float32")
                            * e8m0_to_float(q_scale[q_scale_base + scale_block])
                            * e8m0_to_float(kv_scale[kv_scale_base + scale_block])
                        )
                    scores[k_top] = acc[0] * sm_scale
            T.sync_threads()

            local[0] = T.float32(-3.4028234663852886e38)
            for k_top in T.serial(lane, _BSFP8_APPLY_TOPK, step=_BSFP8_APPLY_THREADS):
                if scores[k_top] > local[0]:
                    local[0] = scores[k_top]
            reduce_buf[lane] = local[0]
            T.sync_threads()
            for round_id in T.serial(_BSFP8_APPLY_LOG_THREADS):
                stride[0] = T.shift_right(_BSFP8_APPLY_THREADS, round_id + 1)
                if lane < stride[0]:
                    if reduce_buf[lane + stride[0]] > reduce_buf[lane]:
                        reduce_buf[lane] = reduce_buf[lane + stride[0]]
                T.sync_threads()
            row_max = reduce_buf[0]

            for k_top in T.serial(lane, _BSFP8_APPLY_TOPK, step=_BSFP8_APPLY_THREADS):
                if scores[k_top] == T.float32(-3.4028234663852886e38):
                    scores[k_top] = 0.0
                else:
                    scores[k_top] = T.exp(scores[k_top] - row_max)
            T.sync_threads()

            local[0] = 0.0
            for k_top in T.serial(lane, _BSFP8_APPLY_TOPK, step=_BSFP8_APPLY_THREADS):
                local[0] = local[0] + scores[k_top]
            reduce_buf[lane] = local[0]
            T.sync_threads()
            for round_id in T.serial(_BSFP8_APPLY_LOG_THREADS):
                stride[0] = T.shift_right(_BSFP8_APPLY_THREADS, round_id + 1)
                if lane < stride[0]:
                    reduce_buf[lane] = reduce_buf[lane] + reduce_buf[lane + stride[0]]
                T.sync_threads()
            sumexp = reduce_buf[0]

            inv_sum[0] = 0.0
            if sumexp > 0.0:
                inv_sum[0] = 1.0 / sumexp

            for d in T.serial(lane, _BSFP8_APPLY_DV, step=_BSFP8_APPLY_THREADS):
                acc[0] = 0.0
                for k_top in T.serial(_BSFP8_APPLY_TOPK):
                    gather_idx[0] = T.cast(indices[idx_base + k_top], "int32")
                    if gather_idx[0] >= 0 and gather_idx[0] < _BSFP8_APPLY_SKV:
                        kv_row_base = kv_b_base + (gather_idx[0] * _BSFP8_APPLY_G + gidx) * _BSFP8_APPLY_K
                        kv_scale_base = kv_scale_b_base + (
                            gather_idx[0] * _BSFP8_APPLY_G + gidx
                        ) * _BSFP8_APPLY_SCALE_BLOCKS
                        scale_block = d // E8M0_BLOCK_SIZE
                        acc[0] = acc[0] + scores[k_top] * T.cast(
                            kv_fp8[kv_row_base + d],
                            "float32",
                        ) * e8m0_to_float(kv_scale[kv_scale_base + scale_block])
                out[out_row + d] = T.cast(acc[0] * inv_sum[0], "float16")

            if lane == 0:
                if sumexp > 0.0:
                    lse[bx] = row_max + T.log(sumexp)
                else:
                    lse[bx] = 0.0

    try:
        from tilelang.transform.simplify import apply_simplify

        return apply_simplify(blockscaled_sparse_mla_apply_kernel)
    except Exception:
        return blockscaled_sparse_mla_apply_kernel


@lru_cache(maxsize=128)
def _blockscaled_apply_kernel_for(
    batch: int,
    seq_len: int,
    heads: int,
    seq_len_kv: int,
    kv_group: int,
    head_kv: int,
    topk: int,
    K: int,
    d_v: int,
    scale_blocks: int,
    threads: int,
    index_dtype: str = "int32",
) -> tuple[Any, _msl_transform.TileLangMSLLowering, list[str]]:
    """Build and cache the E8M0 apply lowering for inspection/status.

    Runtime dispatch uses ``_blockscaled_apply_tvm_ffi_kernel_for`` and
    caller/native owner outputs; this helper does not construct an
    ``mx.fast.metal_kernel`` wrapper.
    """

    prim = _make_blockscaled_sparse_mla_apply_kernel(
        batch=batch,
        seq_len=seq_len,
        heads=heads,
        seq_len_kv=seq_len_kv,
        kv_group=kv_group,
        head_kv=head_kv,
        topk=topk,
        K=K,
        d_v=d_v,
        scale_blocks=scale_blocks,
        threads=threads,
        index_dtype=index_dtype,
    )
    lowering = cast(
        _msl_transform.TileLangMSLLowering,
        dispatch_lower(
            prim,
            target=TILELANG_METAL_E8M0_SPARSE_MLA_TARGET,
            return_msl=True,
        ),
    )
    input_names = [name for name in lowering.buffer_param_names if name not in {"out", "lse"}]
    if set(input_names) != {"q_fp8", "q_scale", "kv_fp8", "kv_scale", "indices", "sm_scale_buf"}:
        raise MSLDispatchUnsupported(
            "unexpected TileLang E8M0 apply buffer signature: "
            + ", ".join(lowering.buffer_param_names)
        )
    return None, lowering, input_names


@lru_cache(maxsize=128)
def _blockscaled_apply_tvm_ffi_kernel_for(
    batch: int,
    seq_len: int,
    heads: int,
    seq_len_kv: int,
    kv_group: int,
    head_kv: int,
    topk: int,
    K: int,
    d_v: int,
    scale_blocks: int,
    threads: int,
    index_dtype: str,
) -> Any:
    """Compile the E8M0 prepared forward kernel for caller-owned outputs."""

    import tilelang

    prim = _make_blockscaled_sparse_mla_apply_kernel(
        batch=batch,
        seq_len=seq_len,
        heads=heads,
        seq_len_kv=seq_len_kv,
        kv_group=kv_group,
        head_kv=head_kv,
        topk=topk,
        K=K,
        d_v=d_v,
        scale_blocks=scale_blocks,
        threads=threads,
        index_dtype=index_dtype,
    )
    return tilelang.compile(
        prim,
        target=_msl_transform._as_metal_target(TILELANG_METAL_E8M0_SPARSE_MLA_TARGET),
        execution_backend="tvm_ffi",
        out_idx=[6, 7],
    )


def _validate_blockscaled_apply_owner_outputs(
    out: mx.array,
    lse: mx.array,
    *,
    batch: int,
    seq_len: int,
    heads: int,
    d_v: int,
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
    if out.dtype != mx.float16:
        raise TypeError(f"out must be mx.float16; got {out.dtype}")
    if lse.dtype != mx.float32:
        raise TypeError(f"lse must be mx.float32; got {lse.dtype}")
    return out, lse


def sparse_mla_blockscaled_path_c_apply_direct(
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
) -> tuple[mx.array, mx.array]:
    """Run E8M0 Sparse-MLA forward through tvm-ffi into caller-owned outputs."""

    if not can_run_metal():
        raise SparseMLABlockScaledPathCDirectError("MLX Metal backend is unavailable")
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
        scale_blocks,
        threads,
    ) = _validate_blockscaled_apply_inputs(
        q_fp8,
        q_scale,
        kv_fp8,
        kv_scale,
        indices,
        d_v=d_v,
    )
    out_buf, lse_buf = _validate_blockscaled_apply_owner_outputs(
        out,
        lse,
        batch=batch,
        seq_len=seq_len,
        heads=heads,
        d_v=d_v_resolved,
    )
    try:
        index_dtype = _index_dtype_name(
            indices,
            op_name="blockscaled Sparse-MLA Path C apply",
        )
        kernel = _blockscaled_apply_tvm_ffi_kernel_for(
            batch,
            seq_len,
            heads,
            seq_len_kv,
            kv_group,
            head_kv,
            topk,
            K,
            d_v_resolved,
            scale_blocks,
            threads,
            index_dtype,
        )
    except Exception as exc:
        raise SparseMLABlockScaledPathCDirectError(
            "direct tvm-ffi E8M0 Sparse-MLA forward compile failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    sm_scale_buf = mx.array([float(sm_scale)], dtype=mx.float32)
    try:
        returned = kernel(
            q_fp8,
            q_scale,
            kv_fp8,
            kv_scale,
            indices,
            sm_scale_buf,
            out=(out_buf, lse_buf),
        )
    except Exception as exc:
        try:
            from tilelang.contrib.mlx_interop import DLPackInteropError
        except Exception:  # pragma: no cover - only when TileLang import itself is broken
            DLPackInteropError = ()  # type: ignore[assignment]
        if isinstance(exc, DLPackInteropError):
            raise
        raise SparseMLABlockScaledPathCDirectError(
            "direct tvm-ffi E8M0 Sparse-MLA forward dispatch failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    _owner_output_tuple(
        returned,
        expected=(out_buf, lse_buf),
        op_name="direct tvm-ffi E8M0 Sparse-MLA forward",
    )
    return out_buf, lse_buf


def sparse_mla_blockscaled_path_c_apply(
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
    out: mx.array | None = None,
    lse: mx.array | None = None,
) -> mx.array | tuple[mx.array, mx.array] | None:
    """Run fused E8M0 Sparse-MLA Path C over prepared GPU buffers."""

    if (out is None) != (lse is None):
        raise ValueError(
            "sparse_mla_blockscaled_path_c_apply owner-output route requires "
            "both out and lse buffers"
        )
    if out is not None and lse is not None:
        direct_out, direct_lse = sparse_mla_blockscaled_path_c_apply_direct(
            q_fp8,
            q_scale,
            kv_fp8,
            kv_scale,
            indices,
            sm_scale=sm_scale,
            out=out,
            lse=lse,
            d_v=d_v,
        )
        if return_lse:
            return direct_out, direct_lse
        return direct_out

    if not can_run_metal():
        if force_path_c:
            raise RuntimeError("sparse_mla_blockscaled_path_c_apply: MLX Metal backend is unavailable")
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
        scale_blocks,
        threads,
    ) = _validate_blockscaled_apply_inputs(
        q_fp8,
        q_scale,
        kv_fp8,
        kv_scale,
        indices,
        d_v=d_v,
    )
    try:
        index_dtype = _index_dtype_name(
            indices,
            op_name="blockscaled Sparse-MLA Path C apply",
        )
        kernel = _blockscaled_apply_tvm_ffi_kernel_for(
            batch,
            seq_len,
            heads,
            seq_len_kv,
            kv_group,
            head_kv,
            topk,
            K,
            d_v_resolved,
            scale_blocks,
            threads,
            index_dtype,
        )
        sm_scale_buf = mx.array([float(sm_scale)], dtype=mx.float32)
        returned = kernel(
            q_fp8,
            q_scale,
            kv_fp8,
            kv_scale,
            indices,
            sm_scale_buf,
        )
    except Exception as exc:
        if force_path_c:
            raise RuntimeError(
                "sparse_mla_blockscaled_path_c_apply: native TileLang "
                f"tvm-ffi graph-output dispatch failed: {type(exc).__name__}: {exc}"
            ) from exc
        return None
    if not isinstance(returned, (list, tuple)) or len(returned) != 2:
        if force_path_c:
            raise RuntimeError(
                "sparse_mla_blockscaled_path_c_apply: native TileLang "
                "tvm-ffi graph-output dispatch did not return out/lse"
            )
        return None
    out = cast(mx.array, returned[0]).reshape((batch, seq_len, heads, d_v_resolved))
    lse = cast(mx.array, returned[1]).reshape((batch, seq_len, heads))
    if return_lse:
        return out, lse
    return out


def blockscaled_sparse_mla_qk_reduce_path_c_status(
    *,
    N: int = 16,
    K: int = 64,
    outputs_per_block: int = 4,
    reduce_threads: int = 32,
    vec: int = 4,
    target: str = TILELANG_METAL_E8M0_SPARSE_MLA_TARGET,
) -> SparseMLABlockScaledQKReducePathCStatus:
    """Return whether the real-shape E8M0 QK reducer can dispatch."""

    ok, reason = _tilelang_available()
    if not ok:
        return SparseMLABlockScaledQKReducePathCStatus(
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
        return SparseMLABlockScaledQKReducePathCStatus(
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
        _qk_reduce_tvm_ffi_kernel_for(N, K, outputs_per_block, reduce_threads, vec, N)
        _kernel, lowering, _ = _qk_reduce_kernel_for(
            N,
            K,
            outputs_per_block,
            reduce_threads,
            vec,
            N,
        )
        features = blockscaled_sparse_mla_qk_reduce_msl_features(lowering.msl_text)
    except Exception as exc:
        return SparseMLABlockScaledQKReducePathCStatus(
            available=False,
            reason=f"TileLang/MLX lowering failed for E8M0 Sparse-MLA QK reducer: {type(exc).__name__}: {exc}",
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
    has_reduce = bool(
        features["simd_sum"]
        or features["simd_shuffle_down"]
        or features["tvm_thread_allreduce"]
    )
    has_e8m0_decode = bool(
        features["e8m0_exp2"]
        and features["e8m0_bias_subtract_127"]
        and features["e8m0_sentinel_255"]
    )
    has_k_block_index = bool(
        features["k_block_shift_5"]
        or features["k_block_div_32"]
        or features["scale_block_index_shift"]
    )
    has_row_scale = bool(features["per_row_B_scale"])
    if has_scale_refs and has_scale_signature and has_reduce and has_e8m0_decode and has_k_block_index and has_row_scale:
        return SparseMLABlockScaledQKReducePathCStatus(
            available=True,
            reason=(
                "TileLang Path C E8M0 Sparse-MLA real QK reducer is dispatchable "
                "for M=1/topk with per-row K/32 E8M0 scales"
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
    if not has_e8m0_decode:
        blockers.append("E8M0 exp2(byte - 127) decode markers missing")
    if not has_k_block_index:
        blockers.append("scale operands are not indexed by K/32")
    if not has_row_scale:
        blockers.append("B_scale is not per output row")
    return SparseMLABlockScaledQKReducePathCStatus(
        available=False,
        reason="TileLang Path C E8M0 Sparse-MLA real QK reducer is not safe to dispatch: "
        + "; ".join(blockers),
        features=features,
        target=target,
        n=N,
        k=K,
        outputs_per_block=outputs_per_block,
        reduce_threads=reduce_threads,
        vec=vec,
    )


def blockscaled_sparse_mla_qk_scaled_matmul_probe_status(
    *,
    M: int = 1,
    N: int = 16,
    K: int = 64,
    BM: int = 1,
    BN: int = 16,
    BK: int = 64,
    a_scale_size: int = 2,
    b_scale_size: int = 2,
    transpose_B: bool = True,
    num_stages: int = 0,
    target: str = TILELANG_METAL_E8M0_SPARSE_MLA_TARGET,
) -> SparseMLABlockScaledPathCStatus:
    """Probe the legacy ``T.fp8_scaled_matmul`` E8M0 Sparse-MLA QK lowering only."""

    ok, reason = _tilelang_available()
    if not ok:
        return SparseMLABlockScaledPathCStatus(
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
        msl = lower_blockscaled_sparse_mla_qk_msl(
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
        return SparseMLABlockScaledPathCStatus(
            available=False,
            reason=(
                "TileLang Metal lowering failed for E8M0 Sparse-MLA QK shape: "
                f"{type(exc).__name__}: {exc}"
            ),
            features={},
            target=target,
            m=M,
            n=N,
            k=K,
            transpose_B=transpose_B,
        )

    features = blockscaled_sparse_mla_qk_msl_features(msl)
    has_fast_path = bool(features["simdgroup_multiply_accumulate"])
    has_scale_refs = bool(features["A_scale_refs"]) and bool(features["B_scale_refs"])
    has_scale_signature = bool(features["signature_has_A_scale"]) and bool(features["signature_has_B_scale"])
    has_e8m0_decode = bool(
        features["e8m0_exp2"]
        and features["e8m0_bias_subtract_127"]
        and features["e8m0_sentinel_255"]
    )
    has_k_block_index = bool(features["k_block_shift_5"] or features["k_block_div_32"])
    a_scale_refs = int(features["A_scale_refs"])
    b_scale_refs = int(features["B_scale_refs"])
    a_zero_refs = int(features["A_scale_collapsed_zero"])
    b_zero_refs = int(features["B_scale_collapsed_zero"])
    has_collapsed_scale = bool(
        (a_scale_refs and a_zero_refs == a_scale_refs)
        or (b_scale_refs and b_zero_refs == b_scale_refs)
    )
    has_scalar_fallback = bool(features["float_a_val"]) or bool(features["float_b_val"])
    shape_eligible = M >= 8 and BM >= 8 and N >= 8 and BN >= 8

    if (
        shape_eligible
        and has_fast_path
        and has_scale_refs
        and has_scale_signature
        and has_e8m0_decode
        and has_k_block_index
        and not has_collapsed_scale
        and not has_scalar_fallback
    ):
        return SparseMLABlockScaledPathCStatus(
            available=True,
            reason=(
                "TileLang Path C E8M0 Sparse-MLA QK probe lowers through "
                "T.fp8_scaled_matmul to Metal simdgroup MMA with K/32 E8M0 scale loads"
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
    if not has_e8m0_decode:
        blockers.append("E8M0 exp2(byte - 127) decode markers missing")
    if not has_k_block_index:
        blockers.append("scale operands are not indexed by K/32")
    if has_collapsed_scale:
        blockers.append("E8M0 scale operands collapsed to [0]")
    if has_scalar_fallback:
        blockers.append("scalar fallback markers present")
    if not shape_eligible:
        blockers.append("Sparse-MLA M=1/topk tile violates current Metal FP8 simdgroup tile constraints")
    return SparseMLABlockScaledPathCStatus(
        available=False,
        reason="TileLang Path C E8M0 Sparse-MLA QK is not safe to dispatch: " + "; ".join(blockers),
        features=features,
        target=target,
        m=M,
        n=N,
        k=K,
        transpose_B=transpose_B,
    )


def blockscaled_sparse_mla_qk_path_c_status(
    *,
    M: int = 1,
    N: int = 16,
    K: int = 64,
    BM: int = 1,
    BN: int = 16,
    BK: int = 64,
    a_scale_size: int = 2,
    b_scale_size: int = 2,
    transpose_B: bool = True,
    num_stages: int = 0,
    target: str = TILELANG_METAL_E8M0_SPARSE_MLA_TARGET,
) -> SparseMLABlockScaledPathCStatus:
    """Availability probe for the dispatchable E8M0 Sparse-MLA Path C QK tile."""

    probe_status = blockscaled_sparse_mla_qk_scaled_matmul_probe_status(
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
        reducer_status = blockscaled_sparse_mla_qk_reduce_path_c_status(
            N=N,
            K=K,
            outputs_per_block=_BSFP8_QKR_NP,
            reduce_threads=_BSFP8_QKR_RT,
            vec=_BSFP8_QKR_VEC,
            target=target,
        )
        legacy_features = _prefix_feature_keys(
            "legacy_e8m0_scaled_matmul_probe_",
            probe_status.features,
        )
        if reducer_status.available:
            return SparseMLABlockScaledPathCStatus(
                available=True,
                reason=(
                    "TileLang Path C E8M0 Sparse-MLA QK dispatches through the "
                    "real M=1/topk reducer; T.fp8_scaled_matmul remains probe-only "
                    "for this shape"
                ),
                features={
                    **reducer_status.features,
                    "dispatch_surface": "qk_reduce",
                    "runnable_qk_reduce_available": True,
                    "runnable_qk_reduce_reason": reducer_status.reason,
                    "legacy_e8m0_scaled_matmul_probe_available": bool(probe_status.available),
                    "legacy_e8m0_scaled_matmul_probe_reason": probe_status.reason,
                    **legacy_features,
                },
                target=target,
                m=M,
                n=N,
                k=K,
                transpose_B=transpose_B,
            )
        if probe_status.available:
            return SparseMLABlockScaledPathCStatus(
                available=True,
                reason=probe_status.reason,
                features={
                    **probe_status.features,
                    "dispatch_surface": "fp8_scaled_matmul",
                    "runnable_qk_reduce_available": False,
                    "runnable_qk_reduce_reason": reducer_status.reason,
                    "legacy_e8m0_scaled_matmul_probe_available": True,
                    "legacy_e8m0_scaled_matmul_probe_reason": probe_status.reason,
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
            "legacy_e8m0_scaled_matmul_probe_available": False,
            "legacy_e8m0_scaled_matmul_probe_reason": probe_status.reason,
            **legacy_features,
        }
        return SparseMLABlockScaledPathCStatus(
            available=False,
            reason=(
                "TileLang Path C E8M0 Sparse-MLA QK has no safe dispatch surface: "
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
        return SparseMLABlockScaledPathCStatus(
            available=True,
            reason=probe_status.reason,
            features={
                **probe_status.features,
                "dispatch_surface": "fp8_scaled_matmul",
                "legacy_e8m0_scaled_matmul_probe_available": True,
                "legacy_e8m0_scaled_matmul_probe_reason": probe_status.reason,
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
        "legacy_e8m0_scaled_matmul_probe_available": False,
        "legacy_e8m0_scaled_matmul_probe_reason": probe_status.reason,
    }
    return SparseMLABlockScaledPathCStatus(
        available=False,
        reason=probe_status.reason,
        features=features,
        target=target,
        m=M,
        n=N,
        k=K,
        transpose_B=transpose_B,
    )


__all__ = [
    "E8M0_BLOCK_SIZE",
    "E8M0_LAYOUT",
    "E8M0_SCALE_FORMAT",
    "SparseMLABlockScaledQKReducePathCStatus",
    "SparseMLABlockScaledPathCDirectError",
    "SparseMLABlockScaledPathCStatus",
    "TILELANG_METAL_E8M0_SPARSE_MLA_TARGET",
    "blockscaled_sparse_mla_qk_msl_features",
    "blockscaled_sparse_mla_qk_path_c_status",
    "blockscaled_sparse_mla_qk_reduce_msl_features",
    "blockscaled_sparse_mla_qk_reduce_path_c",
    "blockscaled_sparse_mla_qk_reduce_path_c_status",
    "blockscaled_sparse_mla_qk_scaled_matmul_probe_status",
    "lower_blockscaled_sparse_mla_qk_msl",
    "lower_blockscaled_sparse_mla_qk_reduce_msl",
    "make_blockscaled_sparse_mla_qk_reduce_kernel",
    "make_blockscaled_sparse_mla_qk_kernel",
    "sparse_mla_blockscaled_path_c_apply_direct",
    "sparse_mla_blockscaled_path_c_apply",
]

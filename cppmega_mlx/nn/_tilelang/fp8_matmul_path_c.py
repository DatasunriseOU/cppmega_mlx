# pyright: reportInvalidTypeForm=false, reportMissingImports=false
"""Prepared-buffer dense FP8 matmul through TileLang Path C.

This module exposes the MLX-callable half of the dense FP8 Path C route:
``A_fp8(M,K) @ B_fp8(N,K).T -> C(M,N)`` where the caller already owns FP8
GPU buffers and fp32 scale tensors. It deliberately does not quantize,
dequantize, copy, or cast large tensors at the wrapper boundary.

The production entry point is the owner-output form
``fp8_scaled_matmul_path_c(..., out=existing_array)``. The no-``out`` helper
is retired because any allocation-backed wrapper would hide ownership and
data-movement semantics at the Python boundary.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from cppmega_mlx.nn._tilelang import _msl_transform
from cppmega_mlx.nn._tilelang._cuda_eager import cuda_eager_available
from cppmega_mlx.nn._tilelang._msl_transform import can_run_metal


TILELANG_METAL_MATMUL_TARGET = "metal -thread_warp_size=32"
FP8_PATH_C_LEGACY_MLX_FAST_ENV = "CPPMEGA_FP8_PATH_C_LEGACY_MLX_FAST"

_FP8_MM_M = 16
_FP8_MM_N = 32
_FP8_MM_K = 32
_FP8_MM_BM = 16
_FP8_MM_BN = 32
_FP8_MM_BK = 32
_FP8_MM_THREADS = 32
_FP8_MM_NUM_STAGES = 0
_FP8_MM_C_DTYPE = "float32"


def _coop_tile_for(M: int, N: int, K: int) -> tuple[int, int, int, int] | None:
    """Pick a Metal-4 cooperative-tensor (matmul2d) tile (BM, BN, BK, threads).

    The cooperative ``mpp::tensor_ops::matmul2d`` micro-tile is 16x32x16: the
    selector in ``src/backend/metal/op/gemm.cc`` only emits ``matmul2d`` when
    ``BM % 16 == 0``, ``BN % 32 == 0``, ``BK % 16 == 0`` and the threadgroup's
    warp count (``threads / 32``) partitions cleanly into ``BM/16 x BN/32``
    warp tiles. These tiles were measured on M4 Max to beat both the legacy
    LUT-dot4 Path C accumulation and the pure-MLX FP8 reference across the
    production matmul shapes (128^3..1024^3); see the bench receipt.

    Returns ``None`` when no legal cooperative tile cleanly divides ``M/N/K``
    (e.g. tiny or non-aligned shapes like 16x16x32 or 8x12x16). The cooperative
    matmul2d kernel does not bounds-guard a partial output tile, so launching it
    on an undividable shape returns numerically WRONG (column-shuffled) output.
    Callers must route a ``None`` result to the legacy dot4 kernel, which is
    correct for any shape -- this is shape-correct dispatch, NOT a silent
    gate-off (RULE #1): the production matmul shapes always yield a real tile.
    """
    # Ordered best-first per shape (measured on M4 Max). Each entry keeps the
    # cooperative-tensor threadgroup memory under the 32 KiB Metal budget that
    # the tvm-ffi runtime enforces:
    #   shmem ~= BM*BK*2 (half A) + BN*BK*2 (half B) + BM*BN*4 (fp32 C) + pad.
    candidates = (
        (32, 64, 32, 128),   # ~16 KiB; best for 256/1024^3
        (32, 32, 32, 64),    # ~8 KiB;  best for 512^3
        (32, 64, 64, 128),   # ~24 KiB
        (16, 32, 32, 32),    # ~4 KiB
        (16, 32, 16, 32),    # smallest legal cooperative tile
    )
    shmem_budget = 32 * 1024
    for bm, bn, bk, threads in candidates:
        if M % bm or N % bn or K % bk:
            continue
        if (M // bm) <= 0 or (N // bn) <= 0:
            continue
        num_warps = threads // 32
        max_m = bm // 16
        max_n = bn // 32
        if max_m <= 0 or max_n <= 0:
            continue
        # warp partition must divide num_warps into the BM/16 x BN/32 grid.
        if num_warps != max_m * max_n:
            continue
        # Conservative shared-memory estimate (padded for matmul2d layout).
        shmem = bm * bk * 2 + bn * bk * 2 + bm * bn * 4 + 2 * 1024
        if shmem > shmem_budget:
            continue
        return bm, bn, bk, threads
    # No legal cooperative tile divides this shape: signal the caller to use the
    # shape-agnostic dot4 kernel instead of emitting a wrong partial-tile GEMM.
    return None


@dataclass(frozen=True)
class FP8MatmulPathCStatus:
    available: bool
    reason: str
    target: str = TILELANG_METAL_MATMUL_TARGET
    dispatch_surface: str = "tvm_ffi_owner_output"
    consumes_prepared_fp8_buffers: bool = True
    training_surface: bool = False


class FP8MatmulPathCDirectError(RuntimeError):
    """Raised when the owner-output tvm-ffi path cannot run safely."""


class FP8MatmulPathCLegacyError(RuntimeError):
    """Raised when callers request the retired no-out allocation path."""


def _raise_owner_output_required(op_name: str) -> None:
    raise FP8MatmulPathCLegacyError(
        f"{op_name}: no-out Path C dispatch is retired. The only supported "
        "FP8 Path C matmul route is tvm-ffi owner-output dispatch; pass "
        "out=existing_mx_array. The old mx.fast.metal_kernel allocation "
        f"path is not re-enabled by {FP8_PATH_C_LEGACY_MLX_FAST_ENV} because "
        "it would allocate an output outside the caller-owned buffer contract."
    )


def _fp8_scaled_matmul_kernel_template(
    A_fp8: T.Tensor((_FP8_MM_M, _FP8_MM_K), "float8_e4m3"),  # type: ignore[name-defined]  # noqa: F821
    A_scale: T.Tensor((1,), "float32"),  # type: ignore[name-defined]  # noqa: F821
    B_fp8: T.Tensor((_FP8_MM_N, _FP8_MM_K), "float8_e4m3"),  # type: ignore[name-defined]  # noqa: F821
    B_scale: T.Tensor((1,), "float32"),  # type: ignore[name-defined]  # noqa: F821
    C: T.Tensor((_FP8_MM_M, _FP8_MM_N), _FP8_MM_C_DTYPE),  # type: ignore[name-defined]  # noqa: F821
):
    """Legacy LUT-dot4 dense FP8 matmul (per-output-cell scalar accumulation).

    This is the original Path C emission: one Metal thread owns one output
    cell and walks K via ``T.metal_fp8_e4m3_dot4`` packed LUT decode. It is
    ~1.7x slower than the pure-MLX FP8 reference at the production matmul
    shapes, but it is the path the tvm-ffi owner-output runtime can launch
    today, so it remains the safe owner-output fallback.

    The faster Metal-4 cooperative-tensor ``matmul2d`` emission lives in
    ``_fp8_scaled_matmul2d_kernel_template`` and is selected by
    ``_make_owner_output_matmul_kernel`` whenever the active TileLang backend
    can launch cooperative-tensor kernels.
    """
    with T.Kernel(  # type: ignore[name-defined]  # noqa: F821
        T.ceildiv(_FP8_MM_N, _FP8_MM_BN),  # type: ignore[name-defined]  # noqa: F821
        T.ceildiv(_FP8_MM_M, _FP8_MM_BM),  # type: ignore[name-defined]  # noqa: F821
        threads=(_FP8_MM_BN, _FP8_MM_BM),
    ) as (bx, by):
        T.fp8_scaled_matmul(  # type: ignore[name-defined]  # noqa: F821
            A_fp8,
            A_scale,
            B_fp8,
            B_scale,
            C,
            transpose_B=True,
            target=Target("metal"),  # type: ignore[name-defined]  # noqa: F821
            a_scale_offset=0,
            b_scale_offset=0,
            c_row_offset=by * _FP8_MM_BM,
            c_col_offset=bx * _FP8_MM_BN,
            outputs_per_block=_FP8_MM_BN,
        )


def _fp8_scaled_matmul2d_kernel_template(
    A_fp8: T.Tensor((_FP8_MM_M, _FP8_MM_K), "float8_e4m3"),  # type: ignore[name-defined]  # noqa: F821
    A_scale: T.Tensor((1,), "float32"),  # type: ignore[name-defined]  # noqa: F821
    B_fp8: T.Tensor((_FP8_MM_N, _FP8_MM_K), "float8_e4m3"),  # type: ignore[name-defined]  # noqa: F821
    B_scale: T.Tensor((1,), "float32"),  # type: ignore[name-defined]  # noqa: F821
    C: T.Tensor((_FP8_MM_M, _FP8_MM_N), _FP8_MM_C_DTYPE),  # type: ignore[name-defined]  # noqa: F821
):
    """Dense FP8 matmul via the Metal-4 cooperative-tensor ``matmul2d`` GEMM.

    Instead of the legacy per-output-cell ``T.metal_fp8_e4m3_dot4`` (LUT
    scalar accumulation), this dequantizes the FP8 ``e4m3`` tiles to half
    into ``shared`` cooperative-input buffers and runs the cooperative
    ``mpp::tensor_ops::matmul2d`` GEMM (triggered because the accumulator
    ``Cs`` is in ``shared`` scope). The per-tensor scales are applied to the
    shared accumulator once after the K reduction.

    The FP8->half dequant routes each byte through an explicit ``float32``
    scratch var so the Metal backend emits the per-element
    ``__tvm_fp8_e4m3_to_half`` decode. A direct vectorized ``T.cast`` of an
    FP8 buffer to ``half4`` mis-lowers to a raw ``(half4)(uchar4)`` integer
    cast (the LUT decode is skipped), which is numerically wrong; the scalar
    scratch breaks that vectorization while keeping ``T.Parallel`` thread
    coverage.
    """
    with T.Kernel(  # type: ignore[name-defined]  # noqa: F821
        T.ceildiv(_FP8_MM_N, _FP8_MM_BN),  # type: ignore[name-defined]  # noqa: F821
        T.ceildiv(_FP8_MM_M, _FP8_MM_BM),  # type: ignore[name-defined]  # noqa: F821
        threads=_FP8_MM_THREADS,
    ) as (bx, by):
        A_shared = T.alloc_shared((_FP8_MM_BM, _FP8_MM_BK), "float16", scope="shared")  # type: ignore[name-defined]  # noqa: F821
        B_shared = T.alloc_shared((_FP8_MM_BN, _FP8_MM_BK), "float16", scope="shared")  # type: ignore[name-defined]  # noqa: F821
        C_shared = T.alloc_shared((_FP8_MM_BM, _FP8_MM_BN), _FP8_MM_C_DTYPE, scope="shared")  # type: ignore[name-defined]  # noqa: F821
        T.clear(C_shared)
        for ko in T.serial(T.ceildiv(_FP8_MM_K, _FP8_MM_BK)):  # type: ignore[name-defined]  # noqa: F821
            for i, kk in T.Parallel(_FP8_MM_BM, _FP8_MM_BK):  # type: ignore[name-defined]  # noqa: F821
                a_val = T.alloc_var("float32")  # type: ignore[name-defined]  # noqa: F821
                a_val = T.cast(A_fp8[by * _FP8_MM_BM + i, ko * _FP8_MM_BK + kk], "float32")  # type: ignore[name-defined]  # noqa: F821
                A_shared[i, kk] = T.cast(a_val, "float16")  # type: ignore[name-defined]  # noqa: F821
            for j, kk in T.Parallel(_FP8_MM_BN, _FP8_MM_BK):  # type: ignore[name-defined]  # noqa: F821
                b_val = T.alloc_var("float32")  # type: ignore[name-defined]  # noqa: F821
                b_val = T.cast(B_fp8[bx * _FP8_MM_BN + j, ko * _FP8_MM_BK + kk], "float32")  # type: ignore[name-defined]  # noqa: F821
                B_shared[j, kk] = T.cast(b_val, "float16")  # type: ignore[name-defined]  # noqa: F821
            T.gemm(A_shared, B_shared, C_shared, transpose_B=True)  # type: ignore[name-defined]  # noqa: F821
        sa = A_scale[0]
        sb = B_scale[0]
        for i, j in T.Parallel(_FP8_MM_BM, _FP8_MM_BN):  # type: ignore[name-defined]  # noqa: F821
            C_shared[i, j] = C_shared[i, j] * sa * sb
        T.copy(C_shared, C[by * _FP8_MM_BM, bx * _FP8_MM_BN])  # type: ignore[name-defined]  # noqa: F821


def _fp8_scaled_matmul2d_cuda_kernel_template(
    A_fp8: T.Tensor((_FP8_MM_M, _FP8_MM_K), "float8_e4m3"),  # type: ignore[name-defined]  # noqa: F821
    A_scale: T.Tensor((1,), "float32"),  # type: ignore[name-defined]  # noqa: F821
    B_fp8: T.Tensor((_FP8_MM_N, _FP8_MM_K), "float8_e4m3"),  # type: ignore[name-defined]  # noqa: F821
    B_scale: T.Tensor((1,), "float32"),  # type: ignore[name-defined]  # noqa: F821
    C: T.Tensor((_FP8_MM_M, _FP8_MM_N), _FP8_MM_C_DTYPE),  # type: ignore[name-defined]  # noqa: F821
):
    """CUDA twin of :func:`_fp8_scaled_matmul2d_kernel_template` (gb10 / sm_121).

    IDENTICAL FP8 e4m3 dequant -> half -> ``T.gemm`` math; the ONLY deltas are
    the codegen-shape choices the CUDA GEMM backend requires where the Metal
    cooperative ``matmul2d`` path differed (mirrors the F2 forward CUDA port,
    :func:`mamba3_chunked_scan_core.chunk_scan_fwd_cuda_prim`):

      * the C accumulator is a REGISTER FRAGMENT (``T.alloc_fragment``), not a
        ``shared`` buffer. The CUDA ``T.gemm`` lowering asserts the C operand is
        a fragment (``local_buf must be a fragment, but got shared``); the Metal
        cooperative path deliberately staged ``Cs`` in shared to TRIGGER the
        ``mpp::tensor_ops::matmul2d`` simdgroup GEMM. A separate ``C_shared``
        stays ONLY as the epilogue copy-out staging buffer.
      * the post-K scale multiply runs on the fragment under ``T.Parallel``
        (lowers identically per the F2 fragment elementwise rule).
      * the epilogue global-store copy carries ``disable_tma=True`` (on sm_121
        the TMA tensormap descriptor mis-aligns at these tile dims ->
        "Invalid TMA descriptor arguments"; the cp.async path is correct).

    The FP8->half dequant is byte-identical to the Metal template: each e4m3 byte
    is routed through an explicit ``float32`` scratch var so the CUDA backend
    emits the NATIVE ``__nv_fp8_e4m3 -> half`` decode (``codegen_cuda`` ->
    ``__nv_fp8..._e4m3``), NOT a vectorized ``(half4)(uchar4)`` integer reinterpret
    that would skip the float8 decode and be numerically wrong (RULE #1: the
    scalar scratch keeps the decode honest on CUDA exactly as it does on Metal).

    ``A_shared``/``B_shared`` stay plain ``scope="shared"`` (already CUDA-correct,
    NOT ``shared.dyn``); there is no ``make_swizzled_layout`` annotation (the
    CUDA GEMM lowering picks its own ldmatrix/cp.async layout).
    """
    with T.Kernel(  # type: ignore[name-defined]  # noqa: F821
        T.ceildiv(_FP8_MM_N, _FP8_MM_BN),  # type: ignore[name-defined]  # noqa: F821
        T.ceildiv(_FP8_MM_M, _FP8_MM_BM),  # type: ignore[name-defined]  # noqa: F821
        threads=_FP8_MM_THREADS,
    ) as (bx, by):
        A_shared = T.alloc_shared((_FP8_MM_BM, _FP8_MM_BK), "float16", scope="shared")  # type: ignore[name-defined]  # noqa: F821
        B_shared = T.alloc_shared((_FP8_MM_BN, _FP8_MM_BK), "float16", scope="shared")  # type: ignore[name-defined]  # noqa: F821
        C_frag = T.alloc_fragment((_FP8_MM_BM, _FP8_MM_BN), _FP8_MM_C_DTYPE)  # type: ignore[name-defined]  # noqa: F821
        C_shared = T.alloc_shared((_FP8_MM_BM, _FP8_MM_BN), _FP8_MM_C_DTYPE, scope="shared")  # type: ignore[name-defined]  # noqa: F821
        T.clear(C_frag)
        A_fp8_local = T.alloc_fragment((_FP8_MM_BM, _FP8_MM_BK), "float8_e4m3")  # type: ignore[name-defined]  # noqa: F821
        B_fp8_local = T.alloc_fragment((_FP8_MM_BN, _FP8_MM_BK), "float8_e4m3")  # type: ignore[name-defined]  # noqa: F821
        for ko in T.serial(T.ceildiv(_FP8_MM_K, _FP8_MM_BK)):  # type: ignore[name-defined]  # noqa: F821
            # Stage the e4m3 bytes into register fragments first (disable_tma on
            # the global load), then cast element-wise. On CUDA codegen the prior
            # ``T.alloc_var("float32")`` scratch lowered to a non-assignable
            # ``float[1]`` array (lvalue error); the per-element fragment->fragment
            # ``T.cast(e4m3 -> float16)`` here lowers to the NATIVE
            # ``__nv_fp8_e4m3 -> half`` decode (codegen_cuda ``GetTileLangFP8Type``
            # -> ``fp8_e4*_t``), NOT a vectorized ``(half4)(uchar4)`` reinterpret —
            # the per-element scalar cast keeps the decode honest (RULE #1) while
            # producing a valid CUDA lvalue. (F2 forward CUDA twin pattern: copy
            # global->local, then operate on the fragment under T.Parallel.)
            T.copy(  # type: ignore[name-defined]  # noqa: F821
                A_fp8[by * _FP8_MM_BM : by * _FP8_MM_BM + _FP8_MM_BM,
                      ko * _FP8_MM_BK : ko * _FP8_MM_BK + _FP8_MM_BK],
                A_fp8_local,
                disable_tma=True,
            )
            T.copy(  # type: ignore[name-defined]  # noqa: F821
                B_fp8[bx * _FP8_MM_BN : bx * _FP8_MM_BN + _FP8_MM_BN,
                      ko * _FP8_MM_BK : ko * _FP8_MM_BK + _FP8_MM_BK],
                B_fp8_local,
                disable_tma=True,
            )
            for i, kk in T.Parallel(_FP8_MM_BM, _FP8_MM_BK):  # type: ignore[name-defined]  # noqa: F821
                A_shared[i, kk] = T.cast(A_fp8_local[i, kk], "float16")  # type: ignore[name-defined]  # noqa: F821
            for j, kk in T.Parallel(_FP8_MM_BN, _FP8_MM_BK):  # type: ignore[name-defined]  # noqa: F821
                B_shared[j, kk] = T.cast(B_fp8_local[j, kk], "float16")  # type: ignore[name-defined]  # noqa: F821
            T.gemm(A_shared, B_shared, C_frag, transpose_B=True)  # type: ignore[name-defined]  # noqa: F821
        sa = A_scale[0]
        sb = B_scale[0]
        for i, j in T.Parallel(_FP8_MM_BM, _FP8_MM_BN):  # type: ignore[name-defined]  # noqa: F821
            C_frag[i, j] = C_frag[i, j] * sa * sb
        T.copy(C_frag, C_shared)  # type: ignore[name-defined]  # noqa: F821
        T.copy(C_shared, C[by * _FP8_MM_BM, bx * _FP8_MM_BN], disable_tma=True)  # type: ignore[name-defined]  # noqa: F821


def _make_scaled_matmul_kernel(
    *,
    M: int,
    N: int,
    K: int,
    BM: int,
    BN: int,
    BK: int,
    num_stages: int,
    c_dtype: str = "float32",
) -> Any:
    """Build the legacy LUT-dot4 owner-output prim (unchanged emission)."""
    import tilelang
    from tilelang import language as T
    from tvm.target import Target

    globals().update(
        T=T,
        Target=Target,
        _FP8_MM_M=int(M),
        _FP8_MM_N=int(N),
        _FP8_MM_K=int(K),
        _FP8_MM_BM=int(BM),
        _FP8_MM_BN=int(BN),
        _FP8_MM_BK=int(BK),
        _FP8_MM_NUM_STAGES=int(num_stages),
        _FP8_MM_C_DTYPE=str(c_dtype),
    )
    return tilelang.language.prim_func(_fp8_scaled_matmul_kernel_template)


def _make_scaled_matmul2d_kernel(
    *,
    M: int,
    N: int,
    K: int,
    num_stages: int = 0,
    c_dtype: str = "float32",
) -> Any:
    """Build the Metal-4 cooperative-tensor (``matmul2d``) owner-output prim.

    The cooperative path imposes its own block / warp constraints
    (``BM % 16``, ``BN % 32``, ``BK % 16``, warp partition); the legacy dot4
    ``BM/BN/BK`` arguments are not necessarily legal cooperative tiles, so a
    measured-best cooperative tile is derived from the GEMM extents.

    Returns ``None`` when ``_coop_tile_for`` reports that no legal cooperative
    tile divides ``M/N/K`` -- the caller must then use the shape-agnostic dot4
    kernel (the cooperative kernel would emit a wrong partial-tile result).
    """
    import tilelang
    from tilelang import language as T
    from tvm.target import Target

    tile = _coop_tile_for(int(M), int(N), int(K))
    if tile is None:
        return None
    bm, bn, bk, threads = tile

    globals().update(
        T=T,
        Target=Target,
        _FP8_MM_M=int(M),
        _FP8_MM_N=int(N),
        _FP8_MM_K=int(K),
        _FP8_MM_BM=int(bm),
        _FP8_MM_BN=int(bn),
        _FP8_MM_BK=int(bk),
        _FP8_MM_THREADS=int(threads),
        _FP8_MM_NUM_STAGES=int(num_stages),
        _FP8_MM_C_DTYPE=str(c_dtype),
    )
    return tilelang.language.prim_func(_fp8_scaled_matmul2d_kernel_template)


def _make_scaled_matmul2d_cuda_kernel(
    *,
    M: int,
    N: int,
    K: int,
    num_stages: int = 0,
    c_dtype: str = "float32",
) -> Any:
    """Build the CUDA (sm_121) dequant->half->``T.gemm`` owner-output prim.

    Same cooperative-tile selection (:func:`_coop_tile_for`) as the Metal
    builder -- every production matmul shape (M=16384; mlp/attn N/K) yields a
    legal tile -- but binds the CUDA twin template
    (:func:`_fp8_scaled_matmul2d_cuda_kernel_template`, register-fragment C).

    Returns ``None`` when ``_coop_tile_for`` reports that no legal tile divides
    ``M/N/K`` (RULE #1: the caller RAISES on a CUDA host rather than emitting a
    wrong partial-tile GEMM -- there is no Metal LUT/dot4 sibling to fall to on
    CUDA; the legacy dot4 intrinsic is Metal-only).
    """
    import tilelang
    from tilelang import language as T
    from tvm.target import Target

    tile = _coop_tile_for(int(M), int(N), int(K))
    if tile is None:
        return None
    bm, bn, bk, threads = tile

    globals().update(
        T=T,
        Target=Target,
        _FP8_MM_M=int(M),
        _FP8_MM_N=int(N),
        _FP8_MM_K=int(K),
        _FP8_MM_BM=int(bm),
        _FP8_MM_BN=int(bn),
        _FP8_MM_BK=int(bk),
        _FP8_MM_THREADS=int(threads),
        _FP8_MM_NUM_STAGES=int(num_stages),
        _FP8_MM_C_DTYPE=str(c_dtype),
    )
    return tilelang.language.prim_func(_fp8_scaled_matmul2d_cuda_kernel_template)


def _resolve_fp8_compile_target(target: Any) -> Any:
    """Resolve the ``target`` kwarg threaded through the FP8 matmul builders.

    The dequant->half->``T.gemm`` prim BODY is target-agnostic plain TileLang;
    only the compile-site target object differs between Metal (Apple) and CUDA
    (gb10 / sm_121). Verbatim port of
    :func:`mamba3_chunked_scan_core._resolve_chunked_compile_target`:

      * ``None`` / ``"metal..."`` -> Metal target (back-compat default; every
        existing Metal caller is unchanged).
      * ``"cuda..."`` (or a CUDA ``tvm.target.Target``) -> CUDA target so the
        SAME prim lowers through TileLang's CUDA backend.

    RULE #1: an unrecognized target spec RAISES (no silent Metal default for a
    CUDA host); the Metal-vs-CUDA selection is an explicit gate.
    """
    if target is None:
        return _msl_transform._as_metal_target(TILELANG_METAL_MATMUL_TARGET)
    if isinstance(target, str):
        spec = target.strip()
        if spec.startswith("cuda"):
            return _msl_transform._as_cuda_target(spec)
        if spec.startswith("metal"):
            return _msl_transform._as_metal_target(spec)
        raise ValueError(
            "fp8_scaled_matmul_path_c: unrecognized target spec "
            f"{target!r}; expected 'metal...'/'cuda...' (RULE #1: no default)"
        )
    kind = str(getattr(getattr(target, "kind", None), "name", "")).lower()
    if "cuda" in kind:
        return _msl_transform._as_cuda_target(target)
    return _msl_transform._as_metal_target(target)


FP8_PATH_C_MATMUL2D_ENV = "CPPMEGA_FP8_PATH_C_MATMUL2D"


def _matmul2d_owner_output_enabled() -> bool:
    """Whether to route the owner-output GEMM through Metal-4 ``matmul2d``.

    matmul2d is the **default and only** owner-output GEMM path (RULE #1: the
    clear path; no silent gate-off to dot4). The cooperative-tensor ``matmul2d``
    kernel is numerically exact (maxdiff 0 vs the FP8 reference) and is launched
    correctly by the production **tvm-ffi** owner-output route on Apple M4+:
    verified end-to-end through ``fp8_scaled_matmul_path_c_direct`` /
    ``NativeTileLangKernel`` at 128^3..1024^3 (maxdiff 0.00000, up to ~4.4x
    faster than the legacy LUT-dot4 accumulation at 1024^3).

    This requires a TileLang build whose tvm-ffi Metal runtime launches the
    cooperative-tensor kernel with the recovered launch config (the
    ``_restore_metal_device_mod`` / ``_metal_launch_config`` recovery in
    ``tilelang/jit/adapter/tvm_ffi.py``) and commits + syncs the owner output
    buffer. Earlier the route was gated OFF because that runtime path collapsed
    the launch config to a degenerate ``(1,1,1) x (1,1,1)`` grid and the owner
    output came back zeroed; both are fixed in the live build, so the gate is
    removed and matmul2d is unconditional.

    ``CPPMEGA_FP8_PATH_C_MATMUL2D`` is retained only as an explicit emergency
    opt-OUT (set it to ``0``/``false``/``off`` to force the legacy dot4 kernel
    on a host whose tvm-ffi runtime regresses). It is **not** consulted to
    silently fall back; matmul2d failures RAISE.
    """
    import os

    val = os.environ.get(FP8_PATH_C_MATMUL2D_ENV, "").strip().lower()
    if val in {"0", "false", "no", "off"}:
        return False
    return True


_FP8_MATMUL_TVM_FFI_KERNEL_CACHE: dict[
    tuple[int, int, int, int, int, int, int, str, str],
    Any,
] = {}
_FP8_MATMUL_TVM_FFI_KERNEL_CACHE_LOCK = threading.RLock()


def _fp8_matmul_tvm_ffi_kernel_for(
    *,
    M: int,
    N: int,
    K: int,
    BM: int,
    BN: int,
    BK: int,
    num_stages: int,
    c_dtype: str,
    target: Any = None,
) -> Any:
    resolved_target = _resolve_fp8_compile_target(target)
    kind = str(getattr(getattr(resolved_target, "kind", None), "name", "")).lower()
    cache_key = (
        int(M),
        int(N),
        int(K),
        int(BM),
        int(BN),
        int(BK),
        int(num_stages),
        str(c_dtype),
        kind or "metal",
    )
    with _FP8_MATMUL_TVM_FFI_KERNEL_CACHE_LOCK:
        cached = _FP8_MATMUL_TVM_FFI_KERNEL_CACHE.get(cache_key)
        if cached is not None:
            return cached

    import tilelang

    # CUDA (gb10 / sm_121): the dequant->half->T.gemm prim is the ONE fp8 path.
    # Select the CUDA twin (register-fragment C) and compile with the F2 sm_121
    # escape hatch pass_configs so the GEMM operand copies use cp.async instead
    # of TMA (the tensormap descriptor mis-aligns at these tile dims ->
    # "Invalid TMA descriptor arguments" at RUN). RULE #1: there is NO Metal
    # LUT/dot4 sibling on CUDA, so an undividable shape or a compile failure
    # RAISES (caught + re-raised as FP8MatmulPathCDirectError by the caller); it
    # does NOT silently fall back to dot4, to bf16, or to a degraded precision.
    if "cuda" in kind:
        cuda_prim = _make_scaled_matmul2d_cuda_kernel(
            M=M,
            N=N,
            K=K,
            num_stages=num_stages,
            c_dtype=c_dtype,
        )
        if cuda_prim is None:
            raise FP8MatmulPathCDirectError(
                f"fp8_scaled_matmul_path_c (cuda): no legal cooperative tile "
                f"divides M={M} N={N} K={K}; the CUDA dequant->T.gemm kernel "
                "cannot tile this shape and there is no Metal-only dot4 sibling "
                "on CUDA (RULE #1: no silent partial-tile / fallback)."
            )
        # sm_121 (Blackwell): disable the TMA + warp-specialized lowering so the
        # GEMM operand copies use plain cp.async/vectorized loads (the same
        # dequant->T.gemm math; TileLang's documented escape hatch). Mirrors
        # compile_chunk_scan_fwd_metal (mamba3_chunked_scan_core.py). RULE #1:
        # explicit per-target codegen choice, not a silent fallback.
        kernel = tilelang.compile(
            cuda_prim,
            target=resolved_target,
            execution_backend="tvm_ffi",
            out_idx=-1,
            pass_configs={
                "tl.disable_tma_lower": True,
                "tl.disable_warp_specialized": True,
            },
        )
        with _FP8_MATMUL_TVM_FFI_KERNEL_CACHE_LOCK:
            _FP8_MATMUL_TVM_FFI_KERNEL_CACHE[cache_key] = kernel
        return kernel

    target = resolved_target

    # RULE #1: the Metal-4 cooperative-tensor matmul2d emission is the clear,
    # default owner-output GEMM path. It is launched correctly by the production
    # tvm-ffi owner-output route on Apple M4+ (verified end-to-end via
    # fp8_scaled_matmul_path_c_direct / NativeTileLangKernel): numerically exact
    # (maxdiff 0.00000 vs the FP8 reference) and up to ~4.4x faster than the
    # legacy LUT-dot4 accumulation at 1024^3. A matmul2d *compile* failure RAISES
    # here (caught and re-raised as FP8MatmulPathCDirectError by the caller) --
    # it does NOT silently fall back to dot4.
    #
    # The one shape-correct exception: when M/N/K admit no legal cooperative
    # tile (`_make_scaled_matmul2d_kernel` -> None, e.g. tiny/non-aligned shapes
    # like 16x16x32), the cooperative kernel cannot tile the output without
    # emitting a wrong partial-tile result, so the shape-agnostic dot4 kernel is
    # used. That is shape-correct dispatch, not a silent gate-off: every
    # production matmul shape (128^3..1024^3) yields a real cooperative tile and
    # runs matmul2d. dot4 is also used when the caller explicitly opts out via
    # CPPMEGA_FP8_PATH_C_MATMUL2D=0.
    coop_prim = None
    if _matmul2d_owner_output_enabled():
        coop_prim = _make_scaled_matmul2d_kernel(
            M=M,
            N=N,
            K=K,
            num_stages=num_stages,
            c_dtype=c_dtype,
        )
    if coop_prim is not None:
        kernel = tilelang.compile(
            coop_prim,
            target=target,
            execution_backend="tvm_ffi",
            out_idx=-1,
        )
    else:
        prim = _make_scaled_matmul_kernel(
            M=M,
            N=N,
            K=K,
            BM=BM,
            BN=BN,
            BK=BK,
            num_stages=num_stages,
            c_dtype=c_dtype,
        )
        kernel = tilelang.compile(
            prim,
            target=target,
            execution_backend="tvm_ffi",
            out_idx=-1,
        )
    with _FP8_MATMUL_TVM_FFI_KERNEL_CACHE_LOCK:
        _FP8_MATMUL_TVM_FFI_KERNEL_CACHE[cache_key] = kernel
    return kernel


def _resolve_scalar_scale(
    scale: mx.array | float,
    *,
    name: str,
    allow_python_scalar: bool = True,
) -> mx.array:
    if isinstance(scale, (int, float)):
        if not allow_python_scalar:
            raise TypeError(
                f"fp8_scaled_matmul_path_c direct owner-output route requires {name} "
                "as an existing mx.float32 shape (1,) tensor; Python scalars would "
                "allocate a new MLX tensor at the wrapper boundary"
            )
        return mx.array([float(scale)], dtype=mx.float32)
    if scale.ndim == 1 and scale.dtype == mx.float32 and scale.size == 1:
        return scale
    raise ValueError(
        f"fp8_scaled_matmul_path_c: expected scalar {name} as mx.float32 shape (1,); "
        f"got shape={tuple(scale.shape)} dtype={scale.dtype}"
    )


def _normalize_inputs(
    A_fp8: mx.array,
    B_fp8: mx.array,
    scale_a: mx.array | float,
    scale_b: mx.array | float,
    allow_python_scalar_scales: bool = True,
) -> tuple[mx.array, mx.array, mx.array, mx.array, int, int, int]:
    if A_fp8.ndim != 2 or B_fp8.ndim != 2:
        raise ValueError(
            f"fp8_scaled_matmul_path_c expects 2D A/B; got "
            f"A.ndim={A_fp8.ndim}, B.ndim={B_fp8.ndim}"
        )
    if A_fp8.dtype != mx.uint8 or B_fp8.dtype != mx.uint8:
        raise ValueError(
            f"fp8_scaled_matmul_path_c expects mx.uint8 e4m3 storage; "
            f"got {A_fp8.dtype}, {B_fp8.dtype}"
        )
    M, K = A_fp8.shape
    N, K_b = B_fp8.shape
    if K != K_b:
        raise ValueError(f"fp8_scaled_matmul_path_c shape mismatch: A=({M},{K}), B=({N},{K_b})")
    if K % 4 != 0:
        raise ValueError(f"fp8_scaled_matmul_path_c requires K multiple of 4; got K={K}")
    return (
        A_fp8,
        _resolve_scalar_scale(
            scale_a,
            name="scale_a",
            allow_python_scalar=allow_python_scalar_scales,
        ),
        B_fp8,
        _resolve_scalar_scale(
            scale_b,
            name="scale_b",
            allow_python_scalar=allow_python_scalar_scales,
        ),
        int(M),
        int(N),
        int(K),
    )


def _tilelang_output_dtype_for_mlx(dtype: Any, *, op_name: str) -> str:
    if dtype == mx.float32:
        return "float32"
    if dtype == mx.float16:
        return "float16"
    mx_bfloat16 = getattr(mx, "bfloat16", None)
    if mx_bfloat16 is not None and dtype == mx_bfloat16:
        raise ValueError(
            f"{op_name}: mx.bfloat16 owner-output is not supported by the "
            "current TileLang Metal ABI because codegen emits MSL `bfloat`; "
            "use mx.float32/mx.float16 or fix TileLang CodeGenMetal first"
        )
    raise ValueError(
        f"{op_name}: out dtype must be mx.float32 or mx.float16; got {dtype}"
    )


def _validate_owner_output(out: mx.array, *, M: int, N: int) -> tuple[mx.array, str]:
    if not isinstance(out, mx.array):
        raise TypeError(
            f"fp8_scaled_matmul_path_c: out must be an mlx.core.array; "
            f"got {type(out).__name__}"
        )
    if out.shape != (M, N):
        raise ValueError(
            f"fp8_scaled_matmul_path_c: out shape must be ({M}, {N}); "
            f"got {tuple(out.shape)}"
        )
    return out, _tilelang_output_dtype_for_mlx(
        out.dtype,
        op_name="fp8_scaled_matmul_path_c",
    )


def _require_fp8_host_for_target(target: Any) -> str:
    """Resolve the host gate for the requested ``target`` (RULE #1: explicit).

    Returns the resolved target kind (``"cuda"``/``"metal"``). On a CUDA target
    a CUDA host (torch.cuda + tilelang, via :func:`cuda_eager_available`) is
    REQUIRED; on a Metal target a working MLX Metal backend is REQUIRED. An
    unmet host requirement RAISES :class:`FP8MatmulPathCDirectError` with
    where+what -- it never silently selects the other backend.
    """
    resolved = _resolve_fp8_compile_target(target)
    kind = str(getattr(getattr(resolved, "kind", None), "name", "")).lower()
    if "cuda" in kind:
        ok, reason = cuda_eager_available()
        if not ok:
            raise FP8MatmulPathCDirectError(
                f"fp8_scaled_matmul_path_c (cuda target): CUDA host unavailable: "
                f"{reason}"
            )
        return "cuda"
    if not can_run_metal():
        raise FP8MatmulPathCDirectError("MLX Metal unavailable")
    return "metal"


def fp8_scaled_matmul_path_c_direct(
    A_fp8: mx.array,
    B_fp8: mx.array,
    *,
    scale_a: mx.array | float,
    scale_b: mx.array | float,
    out: mx.array,
    BM: int = 16,
    BN: int = 16,
    BK: int = 32,
    num_stages: int = 0,
    target: Any = None,
) -> mx.array:
    """Run dense FP8 Path C through tvm-ffi into a caller-owned MLX output.

    ``target`` selects the codegen backend: ``None``/``"metal..."`` keeps the
    Metal cooperative ``matmul2d`` route (Apple, unchanged); ``"cuda..."``
    selects the CUDA register-fragment dequant->``T.gemm`` twin (gb10 / sm_121).
    RULE #1: on a CUDA host the cuda prim is the ONE fp8 path -- a
    compile/dispatch failure RAISES with where+what, never a silent fall to the
    Metal LUT route, to bf16, or to degraded precision.
    """

    _require_fp8_host_for_target(target)
    A, A_scale, B, B_scale, M, N, K = _normalize_inputs(
        A_fp8,
        B_fp8,
        scale_a,
        scale_b,
        allow_python_scalar_scales=False,
    )
    C, c_dtype = _validate_owner_output(out, M=M, N=N)
    try:
        kernel = _fp8_matmul_tvm_ffi_kernel_for(
            M=M,
            N=N,
            K=K,
            BM=int(BM),
            BN=int(BN),
            BK=int(BK),
            num_stages=int(num_stages),
            c_dtype=c_dtype,
            target=target,
        )
    except FP8MatmulPathCDirectError:
        raise
    except Exception as exc:
        raise FP8MatmulPathCDirectError(
            f"direct tvm-ffi FP8 matmul compile failed: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        returned = kernel(A, A_scale, B, B_scale, C)
    except Exception as exc:
        try:
            from tilelang.contrib.mlx_interop import DLPackInteropError
        except Exception:  # pragma: no cover - only when TileLang import itself is broken
            DLPackInteropError = ()  # type: ignore[assignment]
        if isinstance(exc, DLPackInteropError):
            raise
        raise FP8MatmulPathCDirectError(
            f"direct tvm-ffi FP8 matmul dispatch failed: {type(exc).__name__}: {exc}"
        ) from exc
    if returned is not C:
        raise FP8MatmulPathCDirectError(
            "direct tvm-ffi FP8 matmul did not return the caller-owned output"
        )
    return C


def fp8_scaled_matmul_path_c(
    A_fp8: mx.array,
    B_fp8: mx.array,
    *,
    scale_a: mx.array | float,
    scale_b: mx.array | float,
    BM: int = 16,
    BN: int = 16,
    BK: int = 32,
    num_stages: int = 0,
    out: mx.array | None = None,
    target: Any = None,
) -> mx.array | None:
    """Run dense FP8 Path C matmul over prepared GPU buffers.

    ``A_fp8`` is ``(M,K)`` uint8 e4m3 storage and ``B_fp8`` is transposed
    ``(N,K)`` storage. Scales are scalar fp32 only in this first prepared-buffer
    surface. ``target`` selects the codegen backend (``None``/``"metal..."`` =>
    Metal cooperative matmul2d; ``"cuda..."`` => CUDA fragment-C dequant->T.gemm
    twin). When ``out`` is provided, runs the direct tvm-ffi owner-output route
    and returns that same object. Without ``out``, this function fails
    explicitly: there is no non-owner-output Path C dispatch surface.
    """

    if out is not None:
        return fp8_scaled_matmul_path_c_direct(
            A_fp8,
            B_fp8,
            scale_a=scale_a,
            scale_b=scale_b,
            out=out,
            BM=BM,
            BN=BN,
            BK=BK,
            num_stages=num_stages,
            target=target,
        )

    _raise_owner_output_required("fp8_scaled_matmul_path_c")


def fp8_matmul_path_c_status(target: Any = None) -> FP8MatmulPathCStatus:
    """Report whether the dense FP8 Path C matmul is dispatchable on this host.

    ``target`` selects which backend's availability is reported:
      * ``None``/``"metal..."`` -> requires a working MLX Metal backend (Apple).
      * ``"cuda..."`` -> requires a CUDA host (torch.cuda + tilelang) via
        :func:`cuda_eager_available`; on gb10 / sm_121 this returns available
        (the CUDA fragment-C dequant->T.gemm twin), NOT "MLX Metal unavailable".

    RULE #1: the cuda branch never reports the Metal-unavailable reason on a
    CUDA host, and the metal branch is unchanged on Apple.
    """
    resolved = _resolve_fp8_compile_target(target)
    kind = str(getattr(getattr(resolved, "kind", None), "name", "")).lower()
    if "cuda" in kind:
        ok, reason = cuda_eager_available()
        if not ok:
            return FP8MatmulPathCStatus(False, reason, target="cuda")
        return FP8MatmulPathCStatus(
            True,
            "dense FP8 Path C prepared-buffer owner-output matmul is "
            "dispatchable (cuda fragment-C dequant->T.gemm, sm_121)",
            target="cuda",
        )
    if not can_run_metal():
        return FP8MatmulPathCStatus(False, "MLX Metal unavailable")
    try:
        import tilelang  # noqa: F401
    except Exception as exc:
        return FP8MatmulPathCStatus(False, f"tilelang import failed: {exc}")
    return FP8MatmulPathCStatus(
        True,
        "dense FP8 Path C prepared-buffer owner-output matmul is dispatchable",
    )


__all__ = [
    "FP8MatmulPathCDirectError",
    "FP8MatmulPathCLegacyError",
    "FP8MatmulPathCStatus",
    "FP8_PATH_C_LEGACY_MLX_FAST_ENV",
    "FP8_PATH_C_MATMUL2D_ENV",
    "TILELANG_METAL_MATMUL_TARGET",
    "fp8_matmul_path_c_status",
    "fp8_scaled_matmul_path_c_direct",
    "fp8_scaled_matmul_path_c",
]


def fp8_scaled_matmul_path_c_cuda_prim(
    *,
    M: int,
    N: int,
    K: int,
    num_stages: int = 0,
    c_dtype: str = "float32",
) -> Any:
    """Public alias: build the CUDA (sm_121) dequant->half->``T.gemm`` prim.

    Thin wrapper over :func:`_make_scaled_matmul2d_cuda_kernel` exposed under
    the name the round-3 port plan references. Returns the compiled-ready
    ``@T.prim_func`` (register-fragment C, plain ``shared`` operands, the
    scalar-scratch e4m3->half decode); ``None`` when no legal cooperative tile
    divides ``M/N/K`` (the caller RAISES on a CUDA host -- RULE #1). Compile it
    via :func:`tilelang.compile` with ``target=_as_cuda_target("cuda")`` and
    ``pass_configs={"tl.disable_tma_lower": True, "tl.disable_warp_specialized":
    True}`` -- exactly what :func:`_fp8_matmul_tvm_ffi_kernel_for` does on the
    cuda branch.
    """
    return _make_scaled_matmul2d_cuda_kernel(
        M=M,
        N=N,
        K=K,
        num_stages=num_stages,
        c_dtype=c_dtype,
    )


__all__.append("fp8_scaled_matmul_path_c_cuda_prim")

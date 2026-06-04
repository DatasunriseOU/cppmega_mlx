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


def _native_fp8_tile_for(M: int, N: int, K: int) -> tuple[int, int, int, int] | None:
    """Pick a tile (BM, BN, BK, threads) for the NATIVE fp8-input e4m3 MMA path.

    Unlike the Metal cooperative ``matmul2d`` selector (:func:`_coop_tile_for`),
    this targets the gb10/sm_121 CUDA ``T.gemm`` SM120 fp8 MMA dispatch
    (``SM120_16x8x32_TN`` -> ``mma.sync.aligned.kind::f8f6f4.m16n8k32...
    .e4m3.e4m3.f32``). The hardware atom is m16n8k32, so the gating constraints
    are (proven on hardware, res_r2mma.feasibility):

      * TN layout (transpose_B=True) -- handled by the kernel, not the tile.
      * K % 32 == 0 (the fp8 MMA is K=32-deep). HARD FLOOR; BK must be a
        multiple of 32 and K a multiple of BK.
      * BM % 16 == 0 and BN % 8 == 0 (the m16n8 output fragment shape); we use
        the larger, warp-tile-friendly BM % 16, BN % 16 to map cleanly onto
        128/256-thread warp grids.

    The candidate tiles are ordered best-first per the research recommendation:
    start from the proven tilelang fp8 example geometry (128x128x64, 128 threads)
    which divides every prod GEMM (M=16384; N in {3584,10752,37888}; K in
    {3584,18944}); fall back to smaller tiles for the narrow SSD F2 shape and any
    residual sub-128 dim. Threadgroup shared memory is the e4m3 operands (1 byte)
    plus the fp32 C staging -- far under budget, so the dominant constraint is
    divisibility, not shmem.

    Returns ``None`` when no legal native-fp8 tile divides ``M/N/K`` (e.g. a shape
    whose K is not a multiple of 32). RULE #1: the caller RAISES on ``None`` (no
    silent partial-tile / no fp16-decode fallback) -- a K%32!=0 shape simply
    cannot run the native fp8 MMA and must surface that, not silently degrade.
    """
    if K % 32 != 0:
        # The fp8 m16n8k32 MMA hard-requires K (and thus BK) a multiple of 32.
        # Surface the violation to the caller (it RAISES) rather than picking an
        # illegal tile that would mis-tile the K reduction (RULE #1).
        return None
    candidates = (
        (128, 128, 64, 128),  # research-recommended fp8 example geometry
        (128, 128, 32, 128),  # smaller BK (K=32-deep amortized less)
        (128, 64, 64, 128),   # narrower N
        (64, 64, 64, 128),    # SSD F2 tile (64x64x64) and small dims
        (64, 64, 32, 128),
        (32, 64, 32, 128),    # current untuned R2 tile (kept as a sweep point)
        (16, 32, 32, 64),     # smallest legal native-fp8 tile (m16n8 fragment)
    )
    for bm, bn, bk, threads in candidates:
        if bk % 32 != 0:
            continue
        if M % bm or N % bn or K % bk:
            continue
        if bm % 16 or bn % 8:
            continue
        if (M // bm) <= 0 or (N // bn) <= 0:
            continue
        return bm, bn, bk, threads
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


def _fp8_scaled_matmul2d_cuda_native_kernel_template(
    A_fp8: T.Tensor((_FP8_MM_M, _FP8_MM_K), "float8_e4m3"),  # type: ignore[name-defined]  # noqa: F821
    A_scale: T.Tensor((1,), "float32"),  # type: ignore[name-defined]  # noqa: F821
    B_fp8: T.Tensor((_FP8_MM_N, _FP8_MM_K), "float8_e4m3"),  # type: ignore[name-defined]  # noqa: F821
    B_scale: T.Tensor((1,), "float32"),  # type: ignore[name-defined]  # noqa: F821
    C: T.Tensor((_FP8_MM_M, _FP8_MM_N), _FP8_MM_C_DTYPE),  # type: ignore[name-defined]  # noqa: F821
):
    """NATIVE fp8-input e4m3 tensor-core MMA on gb10 / sm_121 (lever r2-native-fp8mma).

    This is the lever-1 rewrite of :func:`_fp8_scaled_matmul2d_cuda_kernel_template`.
    The slow twin DECODES each e4m3 byte to ``float16`` in register fragments and
    runs ``tl.mma_sync<kFloat16,kFloat16,kFloat32,16,8,16>`` -- an fp16-MMA on
    decoded values (the 0.18-0.42x memory-lever). THIS template instead keeps the
    e4m3 operands as ``float8_e4m3`` all the way into ``T.gemm``, so TileLang's
    CUDA GEMM backend dispatches the SM120 fp8 MMA atom
    (``src/backend/cuda/op/gemm.cc`` lines 60/74 accept fp8-input GEMM ->
    ``gemm_mma.h`` ``#if __CUDA_ARCH_LIST__ >= 1200`` ->
    ``TL_DISPATCH_MMA_TEMPLATE(fp8_e4_t, fp8_e4_t, float, SM120_16x8x32_TN)`` ->
    CuTe ``SM120_16x8x32_TN`` -> the real Blackwell PTX
    ``mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32``). The
    generated CUDA emits ``tl::mma_sync<kFloat8_e4m3,kFloat8_e4m3,kFloat32,16,8,32>``
    over ``fp8_e4_t`` register operands -- a REAL fp8-input MMA, fp32 accumulate,
    ZERO fp16 decode (PROVEN on the live gb10 sm_121 GPU, res_r2mma.feasibility=GO;
    on-hardware compile+run gave maxabs-err 0.0229 vs an fp16 reference = honest
    fp8 quantization noise, RULE #1 clean).

    Deltas vs the fp16-decode twin (the ONLY changes -- everything else identical):

      * ``A_shared``/``B_shared`` are ``float8_e4m3`` (NOT ``float16``); the e4m3
        bytes are copied straight from global into the e4m3 shared buffers
        (``disable_tma=True``), with NO ``A_fp8_local``/``B_fp8_local`` fragment
        stage and NO ``T.cast(e4m3 -> float16)`` Parallel loops.
      * ``T.gemm(A_shared_e4m3, B_shared_e4m3, C_frag_fp32, transpose_B=True)``
        now dispatches the native fp8 MMA (the fp8+fp8+fp32 branch of gemm.cc)
        instead of the fp16 MMA.

    Unchanged: the fp32 ``C_frag`` register accumulator (the fp8 MMA accumulates
    in fp32 -- do NOT switch to fp16 accumulate; it loses range), the post-K
    ``sa*sb`` scale epilogue, the ``C_shared`` copy-out staging, and the
    ``disable_tma=True`` on every copy (sm_121 TMA tensormap mis-aligns at these
    tile dims -> "Invalid TMA descriptor arguments"; the cp.async path is correct).
    The compile site (:func:`_fp8_matmul_tvm_ffi_kernel_for`) keeps the same
    ``tl.disable_tma_lower`` / ``tl.disable_warp_specialized`` pass_configs.

    HARD constraints (gemm.cc CheckWgmma fp8 branch + the m16n8k32 MMA shape),
    enforced by :func:`_native_fp8_tile_for` (tile=None -> caller RAISES, RULE #1):
    transpose_B=True (TN) and K % 32 == 0. All prod shapes satisfy them.
    """
    with T.Kernel(  # type: ignore[name-defined]  # noqa: F821
        T.ceildiv(_FP8_MM_N, _FP8_MM_BN),  # type: ignore[name-defined]  # noqa: F821
        T.ceildiv(_FP8_MM_M, _FP8_MM_BM),  # type: ignore[name-defined]  # noqa: F821
        threads=_FP8_MM_THREADS,
    ) as (bx, by):
        # NATIVE-fp8: the shared operands stay e4m3 (no float16 decode buffers).
        A_shared = T.alloc_shared((_FP8_MM_BM, _FP8_MM_BK), "float8_e4m3", scope="shared")  # type: ignore[name-defined]  # noqa: F821
        B_shared = T.alloc_shared((_FP8_MM_BN, _FP8_MM_BK), "float8_e4m3", scope="shared")  # type: ignore[name-defined]  # noqa: F821
        C_frag = T.alloc_fragment((_FP8_MM_BM, _FP8_MM_BN), _FP8_MM_C_DTYPE)  # type: ignore[name-defined]  # noqa: F821
        C_shared = T.alloc_shared((_FP8_MM_BM, _FP8_MM_BN), _FP8_MM_C_DTYPE, scope="shared")  # type: ignore[name-defined]  # noqa: F821
        T.clear(C_frag)
        for ko in T.serial(T.ceildiv(_FP8_MM_K, _FP8_MM_BK)):  # type: ignore[name-defined]  # noqa: F821
            # Copy the e4m3 bytes global->shared DIRECTLY (disable_tma on sm_121).
            # NO fp16 decode: the e4m3 operands feed T.gemm unchanged so the SM120
            # fp8 MMA atom fires. This is the whole point of the lever (RULE #1: a
            # real fp8-input MMA, verified by the kFloat8_e4m3 dtype in the
            # generated CUDA, not a relabeled fp16 path).
            T.copy(  # type: ignore[name-defined]  # noqa: F821
                A_fp8[by * _FP8_MM_BM : by * _FP8_MM_BM + _FP8_MM_BM,
                      ko * _FP8_MM_BK : ko * _FP8_MM_BK + _FP8_MM_BK],
                A_shared,
                disable_tma=True,
            )
            T.copy(  # type: ignore[name-defined]  # noqa: F821
                B_fp8[bx * _FP8_MM_BN : bx * _FP8_MM_BN + _FP8_MM_BN,
                      ko * _FP8_MM_BK : ko * _FP8_MM_BK + _FP8_MM_BK],
                B_shared,
                disable_tma=True,
            )
            # transpose_B=True keeps the TN layout the SM120 fp8 MMA requires.
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


def _make_scaled_matmul2d_cuda_native_kernel(
    *,
    M: int,
    N: int,
    K: int,
    num_stages: int = 0,
    c_dtype: str = "float32",
    BM: int | None = None,
    BN: int | None = None,
    BK: int | None = None,
    threads: int | None = None,
) -> Any:
    """Build the gb10 (sm_121) NATIVE fp8-input e4m3 ``T.gemm`` owner-output prim.

    Binds :func:`_fp8_scaled_matmul2d_cuda_native_kernel_template` (e4m3 shared
    operands fed straight into ``T.gemm`` -> SM120 fp8 MMA). Tile selection uses
    :func:`_native_fp8_tile_for` (K%32==0 + m16n8 divisibility), NOT the Metal
    cooperative selector -- the fp8 MMA wants larger BM/BN than the narrow Metal
    matmul2d micro-tile. An explicit (BM,BN,BK,threads) override is accepted so
    the microbench / a tile sweep can probe a specific geometry; the override is
    VALIDATED (K%32==0, divisibility, m16n8) and RAISES on an illegal tile rather
    than silently mis-tiling (RULE #1).

    Returns ``None`` when no legal native-fp8 tile divides ``M/N/K`` (e.g. K not a
    multiple of 32). RULE #1: the caller RAISES on ``None`` -- the native fp8 MMA
    is the ONE path when selected; it never falls back to the fp16-decode twin or
    to bf16. (For an explicit override, an illegal tile RAISES here directly.)
    """
    import tilelang
    from tilelang import language as T
    from tvm.target import Target

    override = (BM, BN, BK, threads)
    if any(v is not None for v in override):
        if any(v is None for v in override):
            raise ValueError(
                "fp8_scaled_matmul_path_c (cuda native): a tile override must "
                "specify all of BM, BN, BK, threads together (RULE #1: no "
                f"half-specified tile); got BM={BM} BN={BN} BK={BK} threads={threads}"
            )
        bm, bn, bk, thr = int(BM), int(BN), int(BK), int(threads)
        # Validate the override against the fp8 MMA hard constraints (the same
        # gates _native_fp8_tile_for enforces). An illegal override RAISES.
        if bk % 32 != 0 or K % bk:
            raise ValueError(
                "fp8_scaled_matmul_path_c (cuda native): illegal BK override "
                f"BK={bk} for K={K}; the e4m3 m16n8k32 MMA requires BK%32==0 and "
                "K%BK==0 (RULE #1: K%32 is a hard floor, no silent re-tile)."
            )
        if bm % 16 or bn % 8 or M % bm or N % bn:
            raise ValueError(
                "fp8_scaled_matmul_path_c (cuda native): illegal BM/BN override "
                f"BM={bm} BN={bn} for M={M} N={N}; require BM%16==0, BN%8==0, "
                "M%BM==0, N%BN==0 (RULE #1: no wrong partial-tile)."
            )
        tile = (bm, bn, bk, thr)
    else:
        tile = _native_fp8_tile_for(int(M), int(N), int(K))
        if tile is None:
            return None
    bm, bn, bk, thr = tile

    globals().update(
        T=T,
        Target=Target,
        _FP8_MM_M=int(M),
        _FP8_MM_N=int(N),
        _FP8_MM_K=int(K),
        _FP8_MM_BM=int(bm),
        _FP8_MM_BN=int(bn),
        _FP8_MM_BK=int(bk),
        _FP8_MM_THREADS=int(thr),
        _FP8_MM_NUM_STAGES=int(num_stages),
        _FP8_MM_C_DTYPE=str(c_dtype),
    )
    return tilelang.language.prim_func(_fp8_scaled_matmul2d_cuda_native_kernel_template)


# Lever fp8-mma-tune: a CLOSURE-parameterized, PIPELINED native-fp8 prim builder
# the autotune harness drives. It is a STRICT SUPERSET of
# :func:`_make_scaled_matmul2d_cuda_native_kernel` whose params (BM,BN,BK,threads,
# num_stages,enable_rasteration) are bound as Python closure locals (NOT
# ``globals().update`` mutation of the module template), so the tuner can vary the
# tile per call without racing the shared module globals. The KERNEL BODY is the
# verified SM120 fp8-input e4m3 MMA: e4m3 shared operands feed ``T.gemm`` directly
# (-> ``tl::mma_sync<kFloat8_e4m3,kFloat8_e4m3,kFloat32,16,8,32>``), fp32 ``C_frag``
# accumulate, the ``sa*sb`` epilogue, and ``disable_tma=True`` on every copy
# (sm_121 TMA tensormap mis-aligns -> "Invalid TMA descriptor arguments").
#
# DELTAS vs the (still-default) module template
# (:func:`_fp8_scaled_matmul2d_cuda_native_kernel_template`) -- which stays
# BYTE-IDENTICAL so the production default + the Metal route are untouched:
#   * the K loop is ``T.Pipelined(..., num_stages=num_stages)`` instead of
#     ``T.serial(...)`` -- so ``num_stages`` is now LIVE (the module template's
#     ``T.serial`` makes num_stages a NO-OP); num_stages=0 lowers to an
#     un-pipelined single-buffer loop (parity with the current default).
#   * ``threads`` parameterizes ``T.Kernel`` (the module template hard-binds
#     ``_FP8_MM_THREADS``).
#   * an OPTIONAL ``T.use_swizzle(panel_size=10, enable=enable_rasteration)``
#     L2 rasterization hint (off by default -> emission parity with the module
#     template when enable_rasteration=False).
# RULE #1: this is the ONE tunable path; an illegal tile RAISES in the builder
# (validated below, same gates as :func:`_native_fp8_tile_for`); a pipelining/
# swizzle codegen failure RAISES at compile -- it never silently degrades to the
# un-pipelined template or to bf16.


def _make_scaled_matmul2d_cuda_native_tunable(
    *,
    M: int,
    N: int,
    K: int,
    block_M: int,
    block_N: int,
    block_K: int,
    num_stages: int,
    threads: int,
    c_dtype: str = "float32",
    enable_rasteration: bool = False,
) -> Any:
    """Build a PIPELINED, closure-parameterized native fp8-input e4m3 ``T.gemm`` prim.

    All of ``block_M/block_N/block_K``, ``num_stages``, ``threads`` and
    ``enable_rasteration`` are required (the autotuner supplies every axis); they
    are captured as Python closure locals and validated against the fp8 m16n8k32
    MMA hard constraints (BK%32==0, K%BK==0, BM%16==0, BN%8==0, M%BM==0, N%BN==0)
    -- an illegal tile RAISES here (RULE #1: no silent re-tile / partial-tile).

    The emitted body is identical math to the verified native template; only the
    K loop (``T.Pipelined``), the ``T.Kernel`` ``threads``, and the optional
    ``T.use_swizzle`` differ (see the module comment above). Returns a
    ``@T.prim_func`` ready for :func:`tilelang.compile` with the same
    ``tl.disable_tma_lower`` / ``tl.disable_warp_specialized`` pass_configs.
    """
    import tilelang  # noqa: F401
    from tilelang import language as T

    bm, bn, bk = int(block_M), int(block_N), int(block_K)
    thr, ns = int(threads), int(num_stages)
    M_i, N_i, K_i = int(M), int(N), int(K)
    c_dt = str(c_dtype)
    rasterize = bool(enable_rasteration)

    # Validate the tile against the e4m3 m16n8k32 MMA gates (RULE #1: RAISE, the
    # autotuner is expected to filter illegal configs out BEFORE calling this, so
    # reaching here with an illegal tile is a real bug, not a sweep miss).
    if bk % 32 != 0 or K_i % bk:
        raise ValueError(
            "fp8_scaled_matmul_path_c (cuda native tunable): illegal BK "
            f"block_K={bk} for K={K_i}; the e4m3 m16n8k32 MMA requires "
            "block_K%32==0 and K%block_K==0 (RULE #1: K%32 is a hard floor)."
        )
    if bm % 16 or bn % 8 or M_i % bm or N_i % bn:
        raise ValueError(
            "fp8_scaled_matmul_path_c (cuda native tunable): illegal BM/BN "
            f"block_M={bm} block_N={bn} for M={M_i} N={N_i}; require "
            "block_M%16==0, block_N%8==0, M%block_M==0, N%block_N==0 "
            "(RULE #1: no wrong partial-tile)."
        )
    if thr <= 0 or thr % 32 != 0:
        raise ValueError(
            "fp8_scaled_matmul_path_c (cuda native tunable): illegal threads="
            f"{thr}; require a positive multiple of 32 (warp granularity)."
        )
    if ns < 0:
        raise ValueError(
            "fp8_scaled_matmul_path_c (cuda native tunable): illegal "
            f"num_stages={ns}; require >= 0 (0 = un-pipelined single buffer)."
        )

    @T.prim_func
    def _native_fp8_tunable(
        A_fp8: T.Tensor((M_i, K_i), "float8_e4m3"),
        A_scale: T.Tensor((1,), "float32"),
        B_fp8: T.Tensor((N_i, K_i), "float8_e4m3"),
        B_scale: T.Tensor((1,), "float32"),
        C: T.Tensor((M_i, N_i), c_dt),
    ):
        with T.Kernel(
            T.ceildiv(N_i, bn),
            T.ceildiv(M_i, bm),
            threads=thr,
        ) as (bx, by):
            # NATIVE-fp8: shared operands stay e4m3 (no float16 decode buffers) so
            # T.gemm dispatches the SM120 fp8 MMA atom (verified kFloat8_e4m3).
            A_shared = T.alloc_shared((bm, bk), "float8_e4m3", scope="shared")
            B_shared = T.alloc_shared((bn, bk), "float8_e4m3", scope="shared")
            C_frag = T.alloc_fragment((bm, bn), c_dt)
            C_shared = T.alloc_shared((bm, bn), c_dt, scope="shared")
            T.clear(C_frag)
            # Optional L2 rasterization hint (off by default -> emission parity).
            T.use_swizzle(panel_size=10, enable=rasterize)
            # PIPELINED K loop: num_stages is LIVE here (the default module
            # template uses T.serial, which makes num_stages inert).
            for ko in T.Pipelined(T.ceildiv(K_i, bk), num_stages=ns):
                T.copy(
                    A_fp8[by * bm : by * bm + bm, ko * bk : ko * bk + bk],
                    A_shared,
                    disable_tma=True,
                )
                T.copy(
                    B_fp8[bx * bn : bx * bn + bn, ko * bk : ko * bk + bk],
                    B_shared,
                    disable_tma=True,
                )
                # transpose_B=True keeps the TN layout the SM120 fp8 MMA requires.
                T.gemm(A_shared, B_shared, C_frag, transpose_B=True)
            sa = A_scale[0]
            sb = B_scale[0]
            for i, j in T.Parallel(bm, bn):
                C_frag[i, j] = C_frag[i, j] * sa * sb
            T.copy(C_frag, C_shared)
            T.copy(C_shared, C[by * bm, bx * bn], disable_tma=True)

    return _native_fp8_tunable


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

# Lever r2-native-fp8mma: route the gb10 / sm_121 CUDA fp8 GEMM through the
# NATIVE fp8-input e4m3 MMA prim (_fp8_scaled_matmul2d_cuda_native_kernel_template)
# instead of the fp16-decode twin (_fp8_scaled_matmul2d_cuda_kernel_template).
# DEFAULT: native (the real fp8-input MMA is the intended clear path on sm_121).
# Set CPPMEGA_FP8_PATH_C_CUDA_NATIVE=0 to A/B against the original correct-but-slow
# fp16-decode prim (e.g. to compare TFLOPs). This is an explicit A/B opt-out, NOT
# a silent fallback: a native-prim compile/dispatch FAILURE still RAISES (RULE #1);
# the env only chooses which CUDA prim is built, never papers over a broken one.
FP8_PATH_C_CUDA_NATIVE_ENV = "CPPMEGA_FP8_PATH_C_CUDA_NATIVE"


def _cuda_native_fp8_enabled() -> bool:
    """Whether the gb10 CUDA branch builds the NATIVE fp8-input MMA prim.

    Native (real e4m3 tensor-core MMA, SM120_16x8x32_TN) is the DEFAULT clear
    path on sm_121 (res_r2mma.feasibility=GO, proven on-hardware). The env
    ``CPPMEGA_FP8_PATH_C_CUDA_NATIVE`` is an explicit A/B opt-OUT only: set it to
    ``0``/``false``/``off`` to build the original fp16-decode twin instead (for a
    side-by-side TFLOPs comparison). RULE #1: this never silently degrades -- a
    native-prim failure RAISES; the env merely selects which prim is compiled.
    """
    import os

    val = os.environ.get(FP8_PATH_C_CUDA_NATIVE_ENV, "").strip().lower()
    if val in {"0", "false", "no", "off"}:
        return False
    return True


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
        # Lever r2-native-fp8mma: DEFAULT to the native fp8-input e4m3 MMA prim
        # (real SM120 tensor-core fp8 MMA). CPPMEGA_FP8_PATH_C_CUDA_NATIVE=0 opts
        # back to the fp16-decode twin for an explicit A/B. RULE #1: either prim's
        # compile/dispatch failure RAISES (re-raised as FP8MatmulPathCDirectError
        # by the caller); the env only selects which CUDA prim is built.
        if _cuda_native_fp8_enabled():
            cuda_prim = _make_scaled_matmul2d_cuda_native_kernel(
                M=M,
                N=N,
                K=K,
                num_stages=num_stages,
                c_dtype=c_dtype,
            )
            cuda_prim_kind = "native fp8-input e4m3 MMA (SM120_16x8x32_TN)"
            no_tile_detail = (
                "no legal native-fp8 tile divides it (the e4m3 m16n8k32 MMA "
                "requires K%32==0 + m16n8 divisibility)"
            )
        else:
            cuda_prim = _make_scaled_matmul2d_cuda_kernel(
                M=M,
                N=N,
                K=K,
                num_stages=num_stages,
                c_dtype=c_dtype,
            )
            cuda_prim_kind = "fp16-decode dequant->T.gemm twin (A/B opt-out)"
            no_tile_detail = "no legal cooperative tile divides it"
        if cuda_prim is None:
            raise FP8MatmulPathCDirectError(
                f"fp8_scaled_matmul_path_c (cuda, {cuda_prim_kind}): "
                f"{no_tile_detail} for M={M} N={N} K={K}; the CUDA fp8 T.gemm "
                "kernel cannot tile this shape and there is no Metal-only dot4 "
                "sibling on CUDA (RULE #1: no silent partial-tile / fallback)."
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


def fp8_scaled_matmul_path_c_cuda_native_prim(
    *,
    M: int,
    N: int,
    K: int,
    num_stages: int = 0,
    c_dtype: str = "float32",
    BM: int | None = None,
    BN: int | None = None,
    BK: int | None = None,
    threads: int | None = None,
) -> Any:
    """Public alias: build the gb10 (sm_121) NATIVE fp8-input e4m3 ``T.gemm`` prim.

    Thin wrapper over :func:`_make_scaled_matmul2d_cuda_native_kernel` -- the
    lever-r2-native-fp8mma kernel whose e4m3 shared operands feed ``T.gemm``
    directly, so the CUDA backend dispatches the SM120 fp8 MMA atom
    (``SM120_16x8x32_TN`` -> ``mma.sync...kind::f8f6f4.m16n8k32...e4m3.e4m3.f32``)
    -- a REAL fp8-input tensor-core MMA, fp32 accumulate, NO fp16 decode.

    Returns the compiled-ready ``@T.prim_func``; ``None`` when no legal native-fp8
    tile divides ``M/N/K`` (the caller RAISES on a CUDA host -- RULE #1). Optional
    ``BM/BN/BK/threads`` pin a specific tile for a sweep (validated; illegal ->
    RAISE). Compile it via :func:`tilelang.compile` with
    ``target=_as_cuda_target("cuda")`` and ``pass_configs={"tl.disable_tma_lower":
    True, "tl.disable_warp_specialized": True}`` -- exactly what
    :func:`_fp8_matmul_tvm_ffi_kernel_for` does on the cuda native branch.

    VERIFY-GATE (RULE #1): the GB10 phase must confirm the generated CUDA emits
    ``tl::mma_sync<kFloat8_e4m3,kFloat8_e4m3,kFloat32,16,8,32>`` (NOT kFloat16) --
    that proves it is a real fp8-input MMA, not a relabeled fp16 path.
    """
    return _make_scaled_matmul2d_cuda_native_kernel(
        M=M,
        N=N,
        K=K,
        num_stages=num_stages,
        c_dtype=c_dtype,
        BM=BM,
        BN=BN,
        BK=BK,
        threads=threads,
    )


__all__.append("fp8_scaled_matmul_path_c_cuda_native_prim")
__all__.append("FP8_PATH_C_CUDA_NATIVE_ENV")


def fp8_scaled_matmul_path_c_cuda_native_tunable_prim(
    *,
    M: int,
    N: int,
    K: int,
    block_M: int,
    block_N: int,
    block_K: int,
    num_stages: int,
    threads: int,
    c_dtype: str = "float32",
    enable_rasteration: bool = False,
) -> Any:
    """Public alias: build the PIPELINED, closure-parameterized native fp8 prim.

    Thin wrapper over :func:`_make_scaled_matmul2d_cuda_native_tunable` -- the
    lever-fp8-mma-tune kernel the autotune harness (``scratch/fp8_mma_autotune.py``)
    sweeps. Every tile axis (``block_M/block_N/block_K``, ``num_stages``,
    ``threads``, ``enable_rasteration``) is required and validated; an illegal tile
    RAISES (RULE #1). The kernel body is the verified SM120 fp8-input e4m3 MMA
    (``SM120_16x8x32_TN`` -> ``mma.sync...kind::f8f6f4.m16n8k32...e4m3.e4m3.f32``),
    fp32 accumulate, but with a LIVE ``T.Pipelined(num_stages)`` K loop and a
    parameterized ``threads`` (the default
    :func:`fp8_scaled_matmul_path_c_cuda_native_prim` uses ``T.serial`` so its
    num_stages is inert). Compile via :func:`tilelang.compile` with
    ``target=_as_cuda_target("cuda")`` and ``pass_configs={"tl.disable_tma_lower":
    True, "tl.disable_warp_specialized": True}``.

    VERIFY-GATE (RULE #1): the GB10 phase must confirm the generated CUDA still
    emits ``tl::mma_sync<kFloat8_e4m3,kFloat8_e4m3,kFloat32,16,8,32>`` (NOT
    kFloat16) after pipelining -- that proves the pipelined loop did not perturb
    the SM120 fp8 MMA dispatch back to an fp16 path.
    """
    return _make_scaled_matmul2d_cuda_native_tunable(
        M=M,
        N=N,
        K=K,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        num_stages=num_stages,
        threads=threads,
        c_dtype=c_dtype,
        enable_rasteration=enable_rasteration,
    )


__all__.append("fp8_scaled_matmul_path_c_cuda_native_tunable_prim")

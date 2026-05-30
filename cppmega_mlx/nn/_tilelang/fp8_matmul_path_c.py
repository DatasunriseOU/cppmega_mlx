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


def _coop_tile_for(M: int, N: int, K: int) -> tuple[int, int, int, int]:
    """Pick a Metal-4 cooperative-tensor (matmul2d) tile (BM, BN, BK, threads).

    The cooperative ``mpp::tensor_ops::matmul2d`` micro-tile is 16x32x16: the
    selector in ``src/backend/metal/op/gemm.cc`` only emits ``matmul2d`` when
    ``BM % 16 == 0``, ``BN % 32 == 0``, ``BK % 16 == 0`` and the threadgroup's
    warp count (``threads / 32``) partitions cleanly into ``BM/16 x BN/32``
    warp tiles. These tiles were measured on M4 Max to beat both the legacy
    LUT-dot4 Path C accumulation and the pure-MLX FP8 reference across the
    production matmul shapes (128^3..1024^3); see the bench receipt.
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
    # Fallback: smallest legal cooperative tile (single warp, 16x32x16).
    return 16, 32, 16, 32


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
    """
    import tilelang
    from tilelang import language as T
    from tvm.target import Target

    bm, bn, bk, threads = _coop_tile_for(int(M), int(N), int(K))

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


FP8_PATH_C_MATMUL2D_ENV = "CPPMEGA_FP8_PATH_C_MATMUL2D"


def _matmul2d_owner_output_enabled() -> bool:
    """Whether to route the owner-output GEMM through Metal-4 ``matmul2d``.

    The cooperative-tensor ``matmul2d`` kernel is numerically exact (maxdiff 0
    vs the FP8 reference) and 1.5x-10x faster than both the legacy LUT-dot4
    accumulation and the pure-MLX FP8 reference on M4+ when launched via a
    Metal runtime that supports cooperative-tensor kernels (verified through
    TileLang's torch / ``torch.mps`` execution backend).

    It is OFF by default because the owner-output contract here dispatches via
    the **tvm-ffi** execution backend, and the tvm-ffi Metal runtime in the
    pinned TileLang build cannot launch cooperative-tensor kernels: such
    kernels lower to a host module the adapter cannot introspect
    (``device_mod is None``), so the recovered launch config collapses to a
    degenerate ``(1,1,1) x (1,1,1)`` grid and the kernel returns numerically
    wrong output. Until that runtime path lands, the safe owner-output route
    stays on the dot4 kernel. Set ``CPPMEGA_FP8_PATH_C_MATMUL2D=1`` to opt in
    once the active TileLang build can launch cooperative kernels through
    tvm-ffi.
    """
    import os

    val = os.environ.get(FP8_PATH_C_MATMUL2D_ENV, "").strip().lower()
    return val in {"1", "true", "yes", "on"}


_FP8_MATMUL_TVM_FFI_KERNEL_CACHE: dict[
    tuple[int, int, int, int, int, int, int, str],
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
) -> Any:
    cache_key = (
        int(M),
        int(N),
        int(K),
        int(BM),
        int(BN),
        int(BK),
        int(num_stages),
        str(c_dtype),
    )
    with _FP8_MATMUL_TVM_FFI_KERNEL_CACHE_LOCK:
        cached = _FP8_MATMUL_TVM_FFI_KERNEL_CACHE.get(cache_key)
        if cached is not None:
            return cached

    import tilelang

    target = _msl_transform._as_metal_target(TILELANG_METAL_MATMUL_TARGET)

    # Prefer the Metal-4 cooperative-tensor matmul2d emission when explicitly
    # enabled (and the active TileLang build can launch cooperative kernels
    # through tvm-ffi): on M4+ it is numerically exact (maxdiff 0 vs the FP8
    # reference) and 1.5x-10x faster than both the legacy LUT-dot4
    # accumulation and the pure-MLX FP8 reference across 128^3..1024^3.
    # Otherwise dispatch the dot4 owner-output kernel, which the pinned
    # tvm-ffi Metal runtime launches correctly, so the owner-output contract
    # never returns numerically wrong output. See _matmul2d_owner_output_enabled.
    kernel = None
    if _matmul2d_owner_output_enabled():
        try:
            coop_prim = _make_scaled_matmul2d_kernel(
                M=M,
                N=N,
                K=K,
                num_stages=num_stages,
                c_dtype=c_dtype,
            )
            kernel = tilelang.compile(
                coop_prim,
                target=target,
                execution_backend="tvm_ffi",
                out_idx=-1,
            )
        except Exception:
            kernel = None

    if kernel is None:
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
) -> mx.array:
    """Run dense FP8 Path C through tvm-ffi into a caller-owned MLX output."""

    if not can_run_metal():
        raise FP8MatmulPathCDirectError("MLX Metal unavailable")
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
        )
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
) -> mx.array | None:
    """Run dense FP8 Path C matmul over prepared GPU buffers.

    ``A_fp8`` is ``(M,K)`` uint8 e4m3 storage and ``B_fp8`` is transposed
    ``(N,K)`` storage. Scales are scalar fp32 only in this first prepared-buffer
    surface. When ``out`` is provided, runs the direct tvm-ffi owner-output
    route and returns that same object. Without ``out``, this function fails
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
        )

    _raise_owner_output_required("fp8_scaled_matmul_path_c")


def fp8_matmul_path_c_status() -> FP8MatmulPathCStatus:
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

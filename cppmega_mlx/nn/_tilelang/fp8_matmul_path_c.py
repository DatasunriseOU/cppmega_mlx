# pyright: reportInvalidTypeForm=false, reportMissingImports=false
"""Prepared-buffer dense FP8 matmul through TileLang Path C.

This module exposes the MLX-callable half of the dense FP8 Path C route:
``A_fp8(M,K) @ B_fp8(N,K).T -> C(M,N)`` where the caller already owns FP8
GPU buffers and fp32 scale tensors. It deliberately does not quantize,
dequantize, copy, or cast large tensors at the wrapper boundary.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, cast

import mlx.core as mx

from cppmega_mlx.nn._tilelang import _msl_transform
from cppmega_mlx.nn._tilelang._msl_transform import (
    MSLDispatchUnsupported,
    can_run_metal,
    lower_tilelang_to_msl_inline,
)


TILELANG_METAL_MATMUL_TARGET = "metal -thread_warp_size=32"

_FP8_MM_M = 16
_FP8_MM_N = 16
_FP8_MM_K = 32
_FP8_MM_BM = 16
_FP8_MM_BN = 16
_FP8_MM_BK = 32
_FP8_MM_NUM_STAGES = 0


@dataclass(frozen=True)
class FP8MatmulPathCStatus:
    available: bool
    reason: str
    target: str = TILELANG_METAL_MATMUL_TARGET
    dispatch_surface: str = "mlx_fast_metal_kernel"
    consumes_prepared_fp8_buffers: bool = True
    training_surface: bool = False


def _fp8_scaled_matmul_kernel_template(
    A_fp8: T.Tensor((_FP8_MM_M, _FP8_MM_K), "float8_e4m3"),  # type: ignore[name-defined]  # noqa: F821
    A_scale: T.Tensor((1,), "float32"),  # type: ignore[name-defined]  # noqa: F821
    B_fp8: T.Tensor((_FP8_MM_N, _FP8_MM_K), "float8_e4m3"),  # type: ignore[name-defined]  # noqa: F821
    B_scale: T.Tensor((1,), "float32"),  # type: ignore[name-defined]  # noqa: F821
    C: T.Tensor((_FP8_MM_M, _FP8_MM_N), "float32"),  # type: ignore[name-defined]  # noqa: F821
):
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


def _make_scaled_matmul_kernel(
    *,
    M: int,
    N: int,
    K: int,
    BM: int,
    BN: int,
    BK: int,
    num_stages: int,
) -> Any:
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
    )
    return tilelang.language.prim_func(_fp8_scaled_matmul_kernel_template)


_FP8_MATMUL_KERNEL_CACHE: dict[
    tuple[int, int, int, int, int, int, int],
    tuple[
        Any,
        _msl_transform.TileLangMSLLowering,
        list[str],
        tuple[int, int],
        tuple[int, int, int],
        tuple[int, int, int],
    ],
] = {}
_FP8_MATMUL_KERNEL_CACHE_LOCK = threading.RLock()


def _grid_for_lowering(lowering: _msl_transform.TileLangMSLLowering) -> tuple[int, int, int]:
    return _msl_transform.metal_grid_for_lowering(lowering)


def _fp8_matmul_kernel_for(
    *,
    M: int,
    N: int,
    K: int,
    BM: int,
    BN: int,
    BK: int,
    num_stages: int,
) -> tuple[
    Any,
    _msl_transform.TileLangMSLLowering,
    list[str],
    tuple[int, int],
    tuple[int, int, int],
    tuple[int, int, int],
]:
    cache_key = (int(M), int(N), int(K), int(BM), int(BN), int(BK), int(num_stages))
    with _FP8_MATMUL_KERNEL_CACHE_LOCK:
        cached = _FP8_MATMUL_KERNEL_CACHE.get(cache_key)
        if cached is not None:
            return cached

    prim = _make_scaled_matmul_kernel(
        M=M,
        N=N,
        K=K,
        BM=BM,
        BN=BN,
        BK=BK,
        num_stages=num_stages,
    )
    lowering = lower_tilelang_to_msl_inline(
        prim,
        target=TILELANG_METAL_MATMUL_TARGET,
    )
    tilelang_input_names = [name for name in lowering.buffer_param_names if name != "C"]
    if set(tilelang_input_names) != {"A_fp8", "A_scale", "B_fp8", "B_scale"}:
        raise MSLDispatchUnsupported(
            "unexpected TileLang FP8 matmul buffer signature: "
            + ", ".join(lowering.buffer_param_names)
        )
    input_names = ["A_fp8", "A_scale", "B_fp8", "B_scale"]
    output_shape = (int(M), int(N))
    kernel = mx.fast.metal_kernel(
        name=f"cppmega_fp8_matmul_path_c_{M}_{N}_{K}_{BM}_{BN}_{BK}_{num_stages}",
        input_names=input_names,
        output_names=["C"],
        source=lowering.body,
        header=lowering.header,
        ensure_row_contiguous=True,
    )
    result = (
        kernel,
        lowering,
        input_names,
        output_shape,
        _grid_for_lowering(lowering),
        lowering.threadgroup,
    )
    with _FP8_MATMUL_KERNEL_CACHE_LOCK:
        _FP8_MATMUL_KERNEL_CACHE[cache_key] = result
    return result


def _resolve_scalar_scale(scale: mx.array | float, *, name: str) -> mx.array:
    if isinstance(scale, (int, float)):
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
        _resolve_scalar_scale(scale_a, name="scale_a"),
        B_fp8,
        _resolve_scalar_scale(scale_b, name="scale_b"),
        int(M),
        int(N),
        int(K),
    )


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
) -> mx.array | None:
    """Run dense FP8 Path C matmul over prepared GPU buffers.

    ``A_fp8`` is ``(M,K)`` uint8 e4m3 storage and ``B_fp8`` is transposed
    ``(N,K)`` storage. Scales are scalar fp32 only in this first prepared-buffer
    surface. Returns ``None`` when TileLang/Metal cannot dispatch.
    """

    if not can_run_metal():
        return None
    A, A_scale, B, B_scale, M, N, K = _normalize_inputs(A_fp8, B_fp8, scale_a, scale_b)
    try:
        kernel, _lowering, input_names, output_shape, grid, threadgroup = _fp8_matmul_kernel_for(
            M=M,
            N=N,
            K=K,
            BM=int(BM),
            BN=int(BN),
            BK=int(BK),
            num_stages=int(num_stages),
        )
    except MSLDispatchUnsupported:
        return None

    if input_names != ["A_fp8", "A_scale", "B_fp8", "B_scale"]:
        raise RuntimeError(f"fp8_scaled_matmul_path_c: unexpected input order {input_names!r}")
    outputs = cast(_msl_transform.MetalKernel, kernel)(
        inputs=(A, A_scale, B, B_scale),
        template=None,
        output_shapes=(output_shape,),
        output_dtypes=(mx.float32,),
        grid=grid,
        threadgroup=threadgroup,
        stream=mx.gpu,
    )
    return outputs[0]


def fp8_matmul_path_c_status() -> FP8MatmulPathCStatus:
    if not can_run_metal():
        return FP8MatmulPathCStatus(False, "MLX Metal unavailable")
    try:
        _fp8_matmul_kernel_for(M=16, N=16, K=32, BM=16, BN=16, BK=32, num_stages=0)
    except Exception as exc:
        return FP8MatmulPathCStatus(False, f"{type(exc).__name__}: {exc}")
    return FP8MatmulPathCStatus(True, "dense FP8 Path C prepared-buffer matmul is dispatchable")


__all__ = [
    "FP8MatmulPathCStatus",
    "TILELANG_METAL_MATMUL_TARGET",
    "fp8_matmul_path_c_status",
    "fp8_scaled_matmul_path_c",
]

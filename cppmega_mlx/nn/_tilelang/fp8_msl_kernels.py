"""Retired FP8 direct-MSL compatibility helpers.

This module used to host vendored e4m3fn direct-MSL kernels for encode,
decode, scaled matmul, and M=1 vecmat. P2 production cleanup retired that
surface: no raw Metal source is constructed here, and ``fp8_msl_status()``
reports unavailable even on Metal-capable machines.

The public helper names remain as pure-MLX reference/oracle utilities because
tests and bench harnesses still need a stable FP8 math contract while the real
framework route lives in ``fp8_matmul_path_c.py`` and ``fp8_vecmat_path_c.py``.
These helpers are deliberately not a production acceleration path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import mlx.core as mx


__license_notice__ = (
    "Historical FP8 direct-MSL kernels in "
    "cppmega_mlx/nn/_tilelang/fp8_msl_kernels.py were ported from "
    "AppMana/mps-fp8-for-torch-and-comfyui-python-package (commit a902571e, "
    "Apache 2.0) and audiohacking/fp8-mps-metal (commit d4fbd40c, MIT). "
    "The current module no longer embeds or dispatches those sources."
)

_RETIRED_REASON = (
    "FP8 direct-MSL Path B is retired for production cleanup: this module no "
    "longer constructs or dispatches vendored Metal kernels. Use "
    "fp8_matmul_path_c.py or fp8_vecmat_path_c.py for the TileLang/tvm-ffi "
    "owner-output route, or these helpers only as pure-MLX reference oracles."
)


@dataclass(frozen=True)
class FP8MSLKernelStatus:
    """Runtime status of the retired FP8 direct-MSL compatibility surface."""

    available: bool
    reason: str
    dispatch_surface: str = "retired_direct_msl_pure_mlx_reference"
    normal_path_available: bool = False
    normal_path_reason: str = (
        "TileLang/tvm-ffi owner-output replacements live in fp8_matmul_path_c.py "
        "and fp8_vecmat_path_c.py."
    )


def fp8_msl_status() -> FP8MSLKernelStatus:
    """Report that the historical direct-MSL FP8 kernels are retired."""

    return FP8MSLKernelStatus(available=False, reason=_RETIRED_REASON)


def _check_dtype(name: str, arr: mx.array, expected: mx.Dtype) -> None:
    if arr.dtype != expected:
        raise TypeError(
            f"fp8_msl_kernels.{name}: expected dtype {expected}, got {arr.dtype}"
        )


def fp8_to_half(fp8_uchar: mx.array) -> mx.array:
    """Dequantize uint8-packed e4m3fn FP8 storage to fp16 via MLX."""

    _check_dtype("fp8_to_half", fp8_uchar, mx.uint8)
    if fp8_uchar.size == 0:
        return mx.zeros(fp8_uchar.shape, dtype=mx.float16)
    return mx.from_fp8(fp8_uchar, dtype=mx.float16)


def half_to_fp8(half_arr: mx.array) -> mx.array:
    """Quantize an fp16 tensor to e4m3fn FP8 bytes via MLX."""

    _check_dtype("half_to_fp8", half_arr, mx.float16)
    if half_arr.size == 0:
        return mx.zeros(half_arr.shape, dtype=mx.uint8)
    return mx.to_fp8(half_arr.astype(mx.float32))


def _resolve_scale(
    scale: mx.array | float, *, length: int, name: str
) -> Tuple[mx.array, int]:
    """Normalize ``scale`` to a 1D fp32 array; return ``(scale_array, mode)``."""

    if isinstance(scale, (int, float)):
        return mx.array([float(scale)], dtype=mx.float32), 0
    if scale.size == 1:
        return scale.reshape(1).astype(mx.float32), 0
    if scale.size == length:
        return scale.reshape(length).astype(mx.float32), 1
    raise ValueError(
        f"fp8_msl_kernels: expected {name} to have size 1 (per-tensor) or "
        f"{length} (per-channel), got size {scale.size}"
    )


def fp8_scaled_matmul_raw(
    A_fp8: mx.array,
    B_fp8: mx.array,
    *,
    scale_a: mx.array | float,
    scale_b: mx.array | float,
) -> mx.array:
    """Reference scaled FP8 matmul.

    ``A_fp8`` is ``(M, K)`` uint8 e4m3fn storage and ``B_fp8`` is ``(N, K)``
    transposed storage. The output is ``(M, N)`` fp32.
    """

    _check_dtype("fp8_scaled_matmul_raw[A]", A_fp8, mx.uint8)
    _check_dtype("fp8_scaled_matmul_raw[B]", B_fp8, mx.uint8)
    if A_fp8.ndim != 2 or B_fp8.ndim != 2:
        raise ValueError(
            f"fp8_scaled_matmul_raw expects 2D inputs; got A.ndim={A_fp8.ndim}, "
            f"B.ndim={B_fp8.ndim}"
        )
    M, K = A_fp8.shape
    N, K_b = B_fp8.shape
    if K != K_b:
        raise ValueError(
            f"fp8_scaled_matmul_raw shape mismatch: A is (M={M}, K={K}), "
            f"B is (N={N}, K={K_b})"
        )

    scale_a_arr, _ = _resolve_scale(scale_a, length=M, name="scale_a")
    scale_b_arr, _ = _resolve_scale(scale_b, length=N, name="scale_b")
    a_full = mx.from_fp8(A_fp8, dtype=mx.float32)
    b_full = mx.from_fp8(B_fp8, dtype=mx.float32)
    out = mx.matmul(a_full, mx.swapaxes(b_full, 0, 1))

    if scale_a_arr.size == M:
        out = out * scale_a_arr.reshape(M, 1)
    else:
        out = out * scale_a_arr[0]
    if scale_b_arr.size == N:
        out = out * scale_b_arr.reshape(1, N)
    else:
        out = out * scale_b_arr[0]
    return out


def fp8_scaled_vecmat(
    x_fp8: mx.array,
    W_fp8: mx.array,
    *,
    scale_x: mx.array | float,
    scale_w: mx.array | float,
) -> mx.array:
    """Reference vector x FP8-matrix scaled multiply."""

    _check_dtype("fp8_scaled_vecmat[x]", x_fp8, mx.uint8)
    _check_dtype("fp8_scaled_vecmat[W]", W_fp8, mx.uint8)
    if x_fp8.ndim != 1 or W_fp8.ndim != 2:
        raise ValueError(
            f"fp8_scaled_vecmat expects 1D x and 2D W; got x.ndim={x_fp8.ndim}, "
            f"W.ndim={W_fp8.ndim}"
        )
    (K,) = x_fp8.shape
    N, K_w = W_fp8.shape
    if K != K_w:
        raise ValueError(
            f"fp8_scaled_vecmat shape mismatch: x is (K={K},), W is (N={N}, K={K_w})"
        )
    if K % 4 != 0:
        raise ValueError(
            f"fp8_scaled_vecmat: K must be a multiple of 4 (Path C packed dot4 "
            f"contract); got K={K}"
        )

    scale_x_arr, _ = _resolve_scale(scale_x, length=1, name="scale_x")
    scale_w_arr, _ = _resolve_scale(scale_w, length=N, name="scale_w")
    x_full = mx.from_fp8(x_fp8, dtype=mx.float32)
    w_full = mx.from_fp8(W_fp8, dtype=mx.float32)
    out = mx.matmul(w_full, x_full) * scale_x_arr[0]
    if scale_w_arr.size == N:
        return out * scale_w_arr.reshape(N)
    return out * scale_w_arr[0]


@mx.custom_function
def fp8_scaled_matmul(
    A_fp8: mx.array,
    B_fp8: mx.array,
    scale_a: mx.array,
    scale_b: mx.array,
) -> mx.array:
    """Differentiable reference scaled FP8 matmul."""

    return fp8_scaled_matmul_raw(A_fp8, B_fp8, scale_a=scale_a, scale_b=scale_b)


@fp8_scaled_matmul.vjp
def _fp8_scaled_matmul_vjp(primals, cotangent, output):  # noqa: ARG001
    A_fp8, B_fp8, scale_a, scale_b = primals
    a_full = mx.from_fp8(A_fp8, dtype=mx.float32)
    b_full = mx.from_fp8(B_fp8, dtype=mx.float32)

    sa = scale_a.reshape(-1)
    sb = scale_b.reshape(-1)
    a_scaled = a_full * (sa.reshape(-1, 1) if sa.size != 1 else sa[0])
    b_scaled = b_full * (sb.reshape(-1, 1) if sb.size != 1 else sb[0])

    grad_a = mx.matmul(cotangent, b_scaled)
    grad_b = mx.matmul(mx.swapaxes(cotangent, 0, 1), a_scaled)
    grad_A_fp8 = mx.zeros_like(A_fp8)
    grad_B_fp8 = mx.zeros_like(B_fp8)

    if sa.size == 1:
        grad_scale_a = mx.sum(grad_a * a_full, axis=None).reshape(scale_a.shape)
    else:
        grad_scale_a = mx.sum(grad_a * a_full, axis=1).reshape(scale_a.shape)
    if sb.size == 1:
        grad_scale_b = mx.sum(grad_b * b_full, axis=None).reshape(scale_b.shape)
    else:
        grad_scale_b = mx.sum(grad_b * b_full, axis=1).reshape(scale_b.shape)

    return (grad_A_fp8, grad_B_fp8, grad_scale_a, grad_scale_b)


__all__ = [
    "FP8MSLKernelStatus",
    "__license_notice__",
    "fp8_msl_status",
    "fp8_scaled_matmul",
    "fp8_scaled_matmul_raw",
    "fp8_scaled_vecmat",
    "fp8_to_half",
    "half_to_fp8",
]

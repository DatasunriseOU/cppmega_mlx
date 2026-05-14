"""Unsupported fused dynamic-LUT optimizer-kernel seams.

The dynamic Adam8bit/Lion8bit fast paths used to compile direct-MSL kernels via
``mx.fast.metal_kernel``. There is no native MLX fused 8-bit optimizer API with
the same zero-copy contract today, so these symbols now report an explicit
unsupported status. The optimizers preserve behavior by using the native MLX
dynamic-LUT quantize/dequantize path.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


FUSED_BLOCK_SIZE = 256
"""Historical fused-kernel block size; still matches the 8-bit codec layout."""


class FusedOptimizerKernelUnsupported(RuntimeError):
    """Raised when a caller explicitly invokes an unsupported fused kernel."""


@dataclass(frozen=True)
class FusedOptimizerKernelStatus:
    available: bool
    reason: str


_ADAM_UNSUPPORTED_REASON = (
    "Adam8bit dynamic-LUT fused direct-MSL optimizer kernel is unsupported: "
    "MLX exposes custom Metal through mx.fast.metal_kernel, not a native "
    "zero-copy fused 8-bit optimizer API. Use the native MLX unfused "
    "optimizer path."
)

_LION_UNSUPPORTED_REASON = (
    "Lion8bit dynamic-LUT fused direct-MSL optimizer kernel is unsupported: "
    "MLX exposes custom Metal through mx.fast.metal_kernel, not a native "
    "zero-copy fused 8-bit optimizer API. Use the native MLX unfused "
    "optimizer path."
)


def fused_adam8bit_dynamic_status() -> FusedOptimizerKernelStatus:
    """Return the availability status for the dynamic Adam8bit fast path."""

    return FusedOptimizerKernelStatus(False, _ADAM_UNSUPPORTED_REASON)


def fused_lion8bit_dynamic_status() -> FusedOptimizerKernelStatus:
    """Return the availability status for the dynamic Lion8bit fast path."""

    return FusedOptimizerKernelStatus(False, _LION_UNSUPPORTED_REASON)


def fused_adam8bit_dynamic_step(
    param: mx.array,
    grad: mx.array,
    m_quant: mx.array,
    m_absmax: mx.array,
    v_quant: mx.array,
    v_absmax: mx.array,
    *,
    learning_rate: mx.array,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step: mx.array,
    bias_correction: bool,
    block_size: int = FUSED_BLOCK_SIZE,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Reject the removed direct-MSL fused dynamic Adam8bit path."""

    raise FusedOptimizerKernelUnsupported(_ADAM_UNSUPPORTED_REASON)


def fused_lion8bit_dynamic_step(
    param: mx.array,
    grad: mx.array,
    m_quant: mx.array,
    m_absmax: mx.array,
    *,
    learning_rate: mx.array,
    beta1: float,
    beta2: float,
    weight_decay: float,
    block_size: int = FUSED_BLOCK_SIZE,
) -> tuple[mx.array, mx.array, mx.array]:
    """Reject the removed direct-MSL fused dynamic Lion8bit path."""

    raise FusedOptimizerKernelUnsupported(_LION_UNSUPPORTED_REASON)


__all__ = [
    "FUSED_BLOCK_SIZE",
    "FusedOptimizerKernelStatus",
    "FusedOptimizerKernelUnsupported",
    "fused_adam8bit_dynamic_status",
    "fused_adam8bit_dynamic_step",
    "fused_lion8bit_dynamic_status",
    "fused_lion8bit_dynamic_step",
]

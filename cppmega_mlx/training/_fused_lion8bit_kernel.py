"""Unsupported fused Lion8bit optimizer-kernel seam.

The previous implementation built a forward-only direct-MSL kernel with
``mx.fast.metal_kernel``. That is not a native MLX optimizer API and should not
sit on the required training optimizer path. Lion8bit continues to work through
the native MLX quantize/dequantize implementation.
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


_UNSUPPORTED_REASON = (
    "Lion8bit fused direct-MSL optimizer kernel is unsupported: MLX exposes "
    "custom Metal through mx.fast.metal_kernel, not a native zero-copy fused "
    "8-bit optimizer API. Use the native MLX unfused optimizer path."
)


def fused_lion8bit_status() -> FusedOptimizerKernelStatus:
    """Return the availability status for the fused Lion8bit fast path."""

    return FusedOptimizerKernelStatus(False, _UNSUPPORTED_REASON)


def fused_lion8bit_step(
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
    """Reject the removed direct-MSL fused Lion8bit path.

    The arguments are accepted only to preserve the old callable surface. The
    function raises before inspecting, copying, or casting any tensor.
    """

    raise FusedOptimizerKernelUnsupported(_UNSUPPORTED_REASON)


__all__ = [
    "FUSED_BLOCK_SIZE",
    "FusedOptimizerKernelStatus",
    "FusedOptimizerKernelUnsupported",
    "fused_lion8bit_status",
    "fused_lion8bit_step",
]

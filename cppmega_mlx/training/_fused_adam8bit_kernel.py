"""Unsupported fused Adam8bit optimizer-kernel seam.

The previous implementation built a forward-only direct-MSL kernel with
``mx.fast.metal_kernel``. That path is not a native MLX optimizer API and it
does not satisfy this repo's training-kernel policy for required optimizer
updates. The production optimizer now routes through the native MLX
quantize/dequantize path in :mod:`cppmega_mlx.training._quantize_8bit`.

This module intentionally keeps the public fused-step symbol so older callers
get an explicit unsupported error instead of a hidden fallback or a surprise
kernel compile.
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
    "Adam8bit fused direct-MSL optimizer kernel is unsupported: MLX exposes "
    "custom Metal through mx.fast.metal_kernel, not a native zero-copy fused "
    "8-bit optimizer API. Use the native MLX unfused optimizer path."
)


def fused_adam8bit_status() -> FusedOptimizerKernelStatus:
    """Return the availability status for the fused Adam8bit fast path."""

    return FusedOptimizerKernelStatus(False, _UNSUPPORTED_REASON)


def fused_adam8bit_step(
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
    """Reject the removed direct-MSL fused Adam8bit path.

    The arguments are accepted only to preserve the old callable surface. The
    function raises before inspecting, copying, or casting any tensor.
    """

    raise FusedOptimizerKernelUnsupported(_UNSUPPORTED_REASON)


__all__ = [
    "FUSED_BLOCK_SIZE",
    "FusedOptimizerKernelStatus",
    "FusedOptimizerKernelUnsupported",
    "fused_adam8bit_status",
    "fused_adam8bit_step",
]

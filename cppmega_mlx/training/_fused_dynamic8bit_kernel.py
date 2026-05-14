"""Native fused dynamic-LUT optimizer kernel wrappers."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from cppmega_mlx.training._quantize_8bit import _get_lut
from cppmega_mlx.training.native_optim import (
    fused_adam8bit_step as _native_fused_adam8bit_step,
    fused_lion8bit_step as _native_fused_lion8bit_step,
    status as _native_status,
)


FUSED_BLOCK_SIZE = 256
"""Native fused-kernel block size; still matches the 8-bit codec layout."""


class FusedOptimizerKernelUnsupported(RuntimeError):
    """Raised when a caller explicitly invokes an unsupported fused kernel."""


@dataclass(frozen=True)
class FusedOptimizerKernelStatus:
    available: bool
    reason: str


def _status() -> FusedOptimizerKernelStatus:
    native = _native_status()
    return FusedOptimizerKernelStatus(
        bool(native.get("available")),
        str(native.get("reason")),
    )


def fused_adam8bit_dynamic_status() -> FusedOptimizerKernelStatus:
    """Return the availability status for the dynamic Adam8bit fast path."""

    return _status()


def fused_lion8bit_dynamic_status() -> FusedOptimizerKernelStatus:
    """Return the availability status for the dynamic Lion8bit fast path."""

    return _status()


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
    """Run the native fused dynamic-LUT Adam8bit kernel."""

    status = _status()
    if not status.available:
        raise FusedOptimizerKernelUnsupported(status.reason)
    if block_size != FUSED_BLOCK_SIZE:
        raise NotImplementedError(
            f"block_size={block_size} not supported by the fused kernel; "
            f"only block_size={FUSED_BLOCK_SIZE} is wired through."
        )
    outputs = _native_fused_adam8bit_step(
        param,
        grad,
        m_quant,
        m_absmax,
        v_quant,
        v_absmax,
        learning_rate,
        step,
        _get_lut(),
        True,
        float(beta1),
        float(beta2),
        float(eps),
        float(weight_decay),
        bool(bias_correction),
    )
    return tuple(outputs)  # type: ignore[return-value]


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
    """Run the native fused dynamic-LUT Lion8bit kernel."""

    status = _status()
    if not status.available:
        raise FusedOptimizerKernelUnsupported(status.reason)
    if block_size != FUSED_BLOCK_SIZE:
        raise NotImplementedError(
            f"block_size={block_size} not supported by the fused kernel; "
            f"only block_size={FUSED_BLOCK_SIZE} is wired through."
        )
    outputs = _native_fused_lion8bit_step(
        param,
        grad,
        m_quant,
        m_absmax,
        learning_rate,
        _get_lut(),
        True,
        float(beta1),
        float(beta2),
        float(weight_decay),
    )
    return tuple(outputs)  # type: ignore[return-value]


__all__ = [
    "FUSED_BLOCK_SIZE",
    "FusedOptimizerKernelStatus",
    "FusedOptimizerKernelUnsupported",
    "fused_adam8bit_dynamic_status",
    "fused_adam8bit_dynamic_step",
    "fused_lion8bit_dynamic_status",
    "fused_lion8bit_dynamic_step",
]

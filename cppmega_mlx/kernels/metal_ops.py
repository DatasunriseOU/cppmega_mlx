"""Safe seams for optional MLX custom Metal kernels.

These ops are prototypes only. They must keep a pure-MLX implementation and
must not become training-critical until differentiable kernels define VJPs/JVPs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, cast

import mlx.core as mx


Backend = Literal["auto", "mlx", "metal"]
MetalKernel = Callable[..., list[mx.array]]


class MetalKernelUnsupported(RuntimeError):
    """Raised when the caller explicitly requests an unavailable Metal path."""


@dataclass(frozen=True)
class MetalKernelStatus:
    available: bool
    reason: str


@dataclass(frozen=True)
class TrainingKernelStatus:
    in_tree: bool
    source_pinned: bool
    license_covered: bool
    fallback_covered: bool
    parity_covered: bool
    hotspot_evidence: bool
    vjp_covered: bool
    jvp_covered: bool
    training_safe: bool
    differentiable: bool
    reason: str
    fallback_backend: Backend

    def __post_init__(self) -> None:
        if not self.training_safe:
            return

        missing = []
        if not self.in_tree:
            missing.append("in-tree ownership")
        if not self.source_pinned:
            missing.append("source pin")
        if not self.license_covered:
            missing.append("license coverage")
        if not self.fallback_covered:
            missing.append("pure-MLX fallback coverage")
        if not self.parity_covered:
            missing.append("fallback/Metal parity coverage")
        if not self.hotspot_evidence:
            missing.append("profiled hotspot evidence")
        if not self.differentiable:
            missing.append("custom differentiation coverage")
        if not self.vjp_covered:
            missing.append("VJP/backward parity coverage")
        if self.fallback_backend != "mlx":
            missing.append("pure-MLX fallback backend")
        if missing:
            raise ValueError(
                "training-safe Metal kernels require " + ", ".join(missing)
            )


_SUPPORTED_METAL_DTYPES = {mx.float32, mx.float16, mx.bfloat16}


def can_run_metal() -> bool:
    """Return whether this process has an MLX GPU-backed Metal device."""

    metal = getattr(mx, "metal", None)
    return mx.default_device() == mx.gpu and metal is not None and metal.is_available()


def _metal_kernel_constructor() -> Callable[..., MetalKernel] | None:
    fast = getattr(mx, "fast", None)
    metal_kernel = getattr(fast, "metal_kernel", None)
    if metal_kernel is None:
        return None
    return cast(Callable[..., MetalKernel], metal_kernel)


def metal_kernel_status(x: mx.array | None = None) -> MetalKernelStatus:
    """Explain whether the optional prototype Metal path is eligible."""

    if not can_run_metal():
        return MetalKernelStatus(False, "MLX Metal backend is not available on the default GPU device")
    if _metal_kernel_constructor() is None:
        return MetalKernelStatus(False, "MLX mx.fast.metal_kernel API is not available")
    if x is not None and x.dtype not in _SUPPORTED_METAL_DTYPES:
        return MetalKernelStatus(False, f"unsupported dtype for prototype Metal kernel: {x.dtype}")
    if x is not None and x.size == 0:
        return MetalKernelStatus(False, "empty tensors use the pure MLX fallback")
    if _squared_relu_kernel is None:
        return MetalKernelStatus(False, "prototype Metal kernel was not constructed")
    return MetalKernelStatus(True, "Metal kernel path is available")


def squared_relu_training_status() -> TrainingKernelStatus:
    """Return the training policy for the prototype ``squared_relu`` kernel."""

    return TrainingKernelStatus(
        in_tree=True,
        source_pinned=True,
        license_covered=True,
        fallback_covered=True,
        parity_covered=True,
        hotspot_evidence=False,
        vjp_covered=False,
        jvp_covered=False,
        training_safe=False,
        differentiable=False,
        reason=(
            "prototype Metal squared_relu is forward-only; training paths must "
            "use the pure MLX fallback until an in-tree custom_function VJP/JVP "
            "is defined, parity remains covered, and hotspot evidence exists"
        ),
        fallback_backend="mlx",
    )


def squared_relu_reference(x: mx.array) -> mx.array:
    """Pure MLX fallback/reference for ``relu(x) ** 2``."""

    relu = mx.maximum(x, mx.array(0, dtype=x.dtype))
    return relu * relu


def _reject_forward_only_training_kernel() -> None:
    status = squared_relu_training_status()
    if not status.training_safe:
        raise MetalKernelUnsupported(status.reason)


def squared_relu(x: mx.array, *, backend: Backend = "auto", training: bool = False) -> mx.array:
    """Compute ``relu(x) ** 2`` through a safe optional Metal seam.

    ``backend="mlx"`` always uses the pure MLX fallback. ``backend="metal"``
    fails closed when Metal is not eligible. ``backend="auto"`` uses Metal only
    when the local process and input dtype are supported and ``training`` is
    false, otherwise it falls back to the reference implementation.

    ``training=True`` makes the differentiability policy explicit: forward-only
    Metal kernels are never inserted into a training graph. ``backend="auto"``
    and ``backend="mlx"`` use the pure MLX reference, while ``backend="metal"``
    raises until the kernel has a custom ``mx.custom_function`` VJP/JVP.
    """

    if backend not in {"auto", "mlx", "metal"}:
        raise ValueError(f"unknown backend {backend!r}; expected 'auto', 'mlx', or 'metal'")
    if training:
        if backend == "metal":
            _reject_forward_only_training_kernel()
        return squared_relu_reference(x)
    if backend == "mlx":
        return squared_relu_reference(x)

    status = metal_kernel_status(x)
    if not status.available:
        if backend == "metal":
            raise MetalKernelUnsupported(status.reason)
        return squared_relu_reference(x)

    return _squared_relu_metal(x)


def _make_squared_relu_kernel() -> MetalKernel | None:
    if not can_run_metal():
        return None
    metal_kernel = _metal_kernel_constructor()
    if metal_kernel is None:
        return None

    source = """
        uint elem = thread_position_in_grid.x;
        T value = x[elem];
        T zero = static_cast<T>(0.0f);
        T relu = value > zero ? value : zero;
        y[elem] = relu * relu;
    """
    return cast(
        MetalKernel,
        metal_kernel(
            name="cppmega_squared_relu_forward",
            input_names=["x"],
            output_names=["y"],
            source=source,
            ensure_row_contiguous=True,
        ),
    )


_squared_relu_kernel = _make_squared_relu_kernel()


def _squared_relu_metal(x: mx.array) -> mx.array:
    if _squared_relu_kernel is None:
        raise MetalKernelUnsupported("prototype Metal kernel was not constructed")

    threads = min(256, int(x.size))
    return _squared_relu_kernel(
        inputs=[x],
        template=[("T", x.dtype)],
        grid=(x.size, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
        stream=mx.gpu,
    )[0]

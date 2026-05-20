"""Path C tensor taps for explicit forward/backward buffer ownership."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import mlx.core as mx


PathCBufferProbe = Callable[[Mapping[str, Any]], None]


def emit_and_tap_path_c_tensor(
    tensor: mx.array,
    *,
    probe: PathCBufferProbe | None,
    event: Mapping[str, Any],
    capture_gradient: bool = True,
) -> mx.array:
    """Emit a zero-copy Path C buffer event and tap its VJP cotangent."""

    if probe is None:
        return tensor
    logical_names = tuple(str(name) for name in event.get("logical_names", ()))
    if not logical_names:
        return tensor
    forward_event = dict(event)
    forward_event["tensor"] = tensor
    forward_event.setdefault("phase", "forward")
    probe(forward_event)
    if not capture_gradient or not bool(getattr(probe, "capture_gradients", True)):
        return tensor
    if not _dtype_can_carry_gradient(tensor):
        return tensor

    gradient_names = tuple(
        name if name.endswith("_grad") else f"{name}_grad"
        for name in logical_names
    )
    gradient_event = {
        key: value
        for key, value in forward_event.items()
        if key not in {"tensor", "logical_names", "phase"}
    }

    @mx.custom_function
    def _tap(x: mx.array) -> mx.array:
        return x

    @_tap.vjp
    def _tap_vjp(
        primals: tuple[mx.array, ...],
        cotangent: mx.array,
        output: mx.array,
    ) -> tuple[mx.array]:
        del primals, output
        if isinstance(cotangent, mx.array):
            probe(
                {
                    **gradient_event,
                    "name": f"{event.get('name', 'tensor')}_grad",
                    "logical_names": gradient_names,
                    "tensor": cotangent,
                    "phase": "value_and_grad",
                    "source_phase": "backward",
                }
            )
        return (cotangent,)

    return _tap(tensor)


def _dtype_can_carry_gradient(tensor: mx.array) -> bool:
    return tensor.dtype in {mx.float16, mx.float32, mx.bfloat16}


__all__ = ["PathCBufferProbe", "emit_and_tap_path_c_tensor"]

"""DLPack zero-copy boundary helpers between MLX and TileLang/TVM.

V4 fusion regions are compiled by ``cppmega_mlx.runtime.path_c_fusion`` into
TileLang/TVM-FFI kernels. At the boundary between fused regions and the
surrounding MLX graph we must avoid host copies — MLX arrays and TVM NDArrays
both support ``__dlpack__`` / ``from_dlpack`` (Metal-resident capsules per the
mlx#2848 PoC), so the bridge is a thin call-through.

The helpers are deliberately tolerant:
  - If TVM/TileLang isn't importable, we expose ``dlpack_available()`` False
    and the path_c_fusion auto-planner can fall back to per-brick execution.
  - If a tensor's device isn't supported by the consumer (e.g. CUDA→MLX),
    the wrapped capsule call raises and the planner downgrades to a host
    copy via numpy as the last resort.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import mlx.core as mx


@lru_cache(maxsize=1)
def dlpack_available() -> bool:
    """Return True if both MLX and TVM can round-trip a DLPack capsule."""
    if not (hasattr(mx, "from_dlpack") and hasattr(mx.array(0.0), "__dlpack__")):
        return False
    try:
        import tvm  # noqa: F401
        from tvm.runtime import from_dlpack as _tvm_from_dlpack  # noqa: F401
    except Exception:
        return False
    return True


def mlx_to_tilelang(arr: mx.array) -> Any:
    """Wrap an MLX array as a TVM NDArray via DLPack — zero-copy on Metal.

    Raises ``RuntimeError`` if TVM is not importable in this process.
    Raises whatever DLPack raises on incompatible devices.
    """
    try:
        from tvm.runtime import from_dlpack as _tvm_from_dlpack
    except ImportError as exc:
        raise RuntimeError(
            "tvm not importable; install tilelang/tvm or guard the call site "
            "with dlpack_available()"
        ) from exc
    return _tvm_from_dlpack(arr.__dlpack__())


def tilelang_to_mlx(nda: Any) -> mx.array:
    """Wrap a TVM NDArray back to MLX via DLPack — zero-copy on Metal.

    Accepts any object with ``__dlpack__`` (TVM NDArray, raw capsule, etc.)
    and lets MLX validate compatibility.
    """
    return mx.from_dlpack(nda)


def host_copy_fallback(arr: Any) -> mx.array:
    """Last-resort host copy via numpy when DLPack handoff is unavailable.

    Use only after ``dlpack_available()`` returns False or after a DLPack
    call raises on a device mismatch. Pays a full host round-trip in
    exchange for guaranteed correctness.
    """
    import numpy as _np

    if isinstance(arr, mx.array):
        return arr
    if hasattr(arr, "numpy"):
        return mx.array(arr.numpy())
    return mx.array(_np.asarray(arr))


__all__ = [
    "dlpack_available",
    "host_copy_fallback",
    "mlx_to_tilelang",
    "tilelang_to_mlx",
]

"""V7-D01: fp8 hardware/runtime probe + reason string for UI banner.

mlx exposes mx.float8 only when built against fp8-capable hardware.
This probe returns a single dict the UI / RPC can surface as a
banner explaining whether fp8 is available and, if not, WHY.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx


def probe_fp8() -> dict[str, Any]:
    """Return {available: bool, reason: str, dtype_name: str|None}.

    available=True only when mx exposes a usable fp8 dtype constant
    AND a small alloc on that dtype succeeds without raising. On
    failure the reason string is what the UI shows verbatim.
    """
    fp8 = getattr(mx, "float8", None)
    if fp8 is None:
        return {
            "available": False,
            "reason": ("mx.float8 not present in this MLX build; "
                       "fp8 requires hardware support."),
            "dtype_name": None,
        }
    try:
        _ = mx.zeros((4,), dtype=fp8)
        mx.eval(_)
        return {
            "available": True,
            "reason": "mx.float8 alloc succeeded.",
            "dtype_name": str(fp8),
        }
    except Exception as exc:
        return {
            "available": False,
            "reason": (
                f"mx.float8 exists but alloc failed: "
                f"{type(exc).__name__}: {exc}"),
            "dtype_name": str(fp8),
        }


__all__ = ["probe_fp8"]

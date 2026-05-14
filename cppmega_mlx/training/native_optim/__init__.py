"""Native MLX optimizer extension bindings."""

from __future__ import annotations

try:
    from cppmega_mlx.training.native_optim._ext import (  # type: ignore[attr-defined]
        fused_adam8bit_step,
        fused_lion8bit_step,
        status,
    )
except Exception as exc:  # pragma: no cover - exercised on non-extension builds.
    _IMPORT_ERROR = exc

    def status() -> dict[str, object]:
        return {
            "available": False,
            "reason": f"native optimizer extension unavailable: {_IMPORT_ERROR}",
        }

    def fused_adam8bit_step(*args: object, **kwargs: object) -> object:
        raise RuntimeError(status()["reason"])

    def fused_lion8bit_step(*args: object, **kwargs: object) -> object:
        raise RuntimeError(status()["reason"])


__all__ = ["fused_adam8bit_step", "fused_lion8bit_step", "status"]

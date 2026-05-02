"""Opt-in MLX memory-limit planning helpers.

These helpers compute conservative wired and Metal allocator limits from a
known total-memory byte count. They do not change process or system limits
unless ``apply_memory_limit_plan`` is called explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_WIRED_RATIO = 0.70
DEFAULT_METAL_RATIO = 0.85


@dataclass(frozen=True)
class MemoryLimitPlan:
    """Suggested MLX memory limits in bytes."""

    total_bytes: int
    wired_ratio: float
    metal_ratio: float
    wired_limit_bytes: int
    metal_limit_bytes: int

    def to_dict(self) -> dict[str, int | float]:
        return {
            "total_bytes": self.total_bytes,
            "wired_ratio": self.wired_ratio,
            "metal_ratio": self.metal_ratio,
            "wired_limit_bytes": self.wired_limit_bytes,
            "metal_limit_bytes": self.metal_limit_bytes,
        }


@dataclass(frozen=True)
class AppliedMemoryLimits:
    """Result from an explicit memory-limit application attempt."""

    plan: MemoryLimitPlan
    applied: bool
    previous_wired_limit_bytes: int | None = None
    previous_metal_limit_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.plan.to_dict(),
            "applied": self.applied,
            "previous_wired_limit_bytes": self.previous_wired_limit_bytes,
            "previous_metal_limit_bytes": self.previous_metal_limit_bytes,
        }


def _validate_total_bytes(total_bytes: int) -> int:
    if not isinstance(total_bytes, int):
        raise TypeError("total_bytes must be an integer byte count")
    if total_bytes <= 0:
        raise ValueError("total_bytes must be positive")
    return total_bytes


def _validate_ratio(value: float, *, name: str) -> float:
    if not isinstance(value, int | float):
        raise TypeError(f"{name} must be a numeric ratio")
    ratio = float(value)
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"{name} must be > 0 and < 1")
    return ratio


def memory_limit_plan(
    total_bytes: int,
    *,
    wired_ratio: float = DEFAULT_WIRED_RATIO,
    metal_ratio: float = DEFAULT_METAL_RATIO,
) -> MemoryLimitPlan:
    """Compute suggested MLX wired and Metal memory limits.

    The returned plan is only arithmetic. It does not call MLX and does not
    claim to prevent every out-of-memory or kernel-panic failure mode.
    """

    total = _validate_total_bytes(total_bytes)
    wired = _validate_ratio(wired_ratio, name="wired_ratio")
    metal = _validate_ratio(metal_ratio, name="metal_ratio")
    return MemoryLimitPlan(
        total_bytes=total,
        wired_ratio=wired,
        metal_ratio=metal,
        wired_limit_bytes=int(total * wired),
        metal_limit_bytes=int(total * metal),
    )


def device_total_memory_bytes(mx_module: Any | None = None) -> int | None:
    """Return ``mlx.core.device_info()['memory_size']`` when available."""

    mx = _load_mlx_core() if mx_module is None else mx_module
    device_info = getattr(mx, "device_info", None)
    if device_info is None:
        return None
    info = device_info()
    if not isinstance(info, dict):
        return None
    memory_size = info.get("memory_size")
    return None if memory_size is None else int(memory_size)


def apply_memory_limit_plan(
    plan: MemoryLimitPlan,
    *,
    mx_module: Any | None = None,
    apply: bool = False,
) -> AppliedMemoryLimits:
    """Apply a precomputed plan to MLX only when ``apply=True``.

    ``mx.set_wired_limit`` and ``mx.metal.set_memory_limit`` both return the
    previous limit in current MLX releases. Tests can pass a fake ``mx_module``
    to verify behavior without touching process-global MLX limits.
    """

    if not isinstance(plan, MemoryLimitPlan):
        raise TypeError("plan must be a MemoryLimitPlan")
    if not apply:
        return AppliedMemoryLimits(plan=plan, applied=False)

    mx = _load_mlx_core() if mx_module is None else mx_module
    set_wired_limit = getattr(mx, "set_wired_limit", None)
    metal = getattr(mx, "metal", None)
    set_metal_limit = getattr(metal, "set_memory_limit", None)
    if set_wired_limit is None:
        raise RuntimeError("mlx.core.set_wired_limit is unavailable")
    if set_metal_limit is None:
        raise RuntimeError("mlx.core.metal.set_memory_limit is unavailable")

    previous_wired = int(set_wired_limit(plan.wired_limit_bytes))
    previous_metal = int(set_metal_limit(plan.metal_limit_bytes))
    return AppliedMemoryLimits(
        plan=plan,
        applied=True,
        previous_wired_limit_bytes=previous_wired,
        previous_metal_limit_bytes=previous_metal,
    )


def _load_mlx_core() -> Any:
    import mlx.core as mx

    return mx


__all__ = [
    "AppliedMemoryLimits",
    "DEFAULT_METAL_RATIO",
    "DEFAULT_WIRED_RATIO",
    "MemoryLimitPlan",
    "apply_memory_limit_plan",
    "device_total_memory_bytes",
    "memory_limit_plan",
]

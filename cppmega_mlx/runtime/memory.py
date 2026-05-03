"""Opt-in MLX memory-limit planning helpers.

These helpers compute conservative wired and Metal allocator limits from a
known total-memory byte count. They do not change process or system limits
unless ``apply_memory_limit_plan`` is called explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import platform
from typing import Any, Callable, Mapping, Sequence, cast

DEFAULT_WIRED_RATIO = 0.70
DEFAULT_METAL_RATIO = 0.85
DEFAULT_PEAK_MEMORY_RATIO = 0.75
M06_MEMORY_RECEIPT_SCOPE = "local_mlx_m06_memory"
M06_REQUIRED_STEPS = 100


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
    metal_limit_api_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.plan.to_dict(),
            "applied": self.applied,
            "previous_wired_limit_bytes": self.previous_wired_limit_bytes,
            "previous_metal_limit_bytes": self.previous_metal_limit_bytes,
            "metal_limit_api_path": self.metal_limit_api_path,
        }


@dataclass(frozen=True)
class MemoryLimitApiStatus:
    """Available MLX memory-limit APIs for a local process."""

    wired_limit_available: bool
    root_memory_limit_available: bool
    metal_memory_limit_available: bool
    preferred_memory_limit_api_path: str | None
    supported_memory_limit_api_paths: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "wired_limit_available": self.wired_limit_available,
            "root_memory_limit_available": self.root_memory_limit_available,
            "metal_memory_limit_available": self.metal_memory_limit_available,
            "preferred_memory_limit_api_path": self.preferred_memory_limit_api_path,
            "supported_memory_limit_api_paths": list(
                self.supported_memory_limit_api_paths
            ),
        }


@dataclass(frozen=True)
class RuntimeStackEvidence:
    """MLX/Metal/device stack captured with a memory receipt."""

    mlx_version: str | None
    mlx_metal_version: str | None
    platform_system: str
    platform_release: str
    macos_version: str | None
    machine: str
    default_device: str | None
    metal_available: bool | None
    device_info: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mlx_version": self.mlx_version,
            "mlx_metal_version": self.mlx_metal_version,
            "platform_system": self.platform_system,
            "platform_release": self.platform_release,
            "macos_version": self.macos_version,
            "machine": self.machine,
            "default_device": self.default_device,
            "metal_available": self.metal_available,
            "device_info": dict(self.device_info),
        }


@dataclass(frozen=True)
class ClearCacheEvent:
    """Evidence that ``mx.clear_cache`` ran after a training step."""

    step: int
    every_steps: int
    api_path: str
    cache_memory_before_bytes: int | None = None
    cache_memory_after_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "every_steps": self.every_steps,
            "api_path": self.api_path,
            "cache_memory_before_bytes": self.cache_memory_before_bytes,
            "cache_memory_after_bytes": self.cache_memory_after_bytes,
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


def _validate_positive_int(value: int, *, name: str) -> int:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


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


def should_clear_cache_after_step(step: int, every_steps: int | None) -> bool:
    """Return whether an M0.6 cache-clear cadence fires after ``step``."""

    step = _validate_positive_int(step, name="step")
    if every_steps is None:
        return False
    every = _validate_positive_int(every_steps, name="every_steps")
    return step % every == 0


def maybe_clear_cache_after_step(
    step: int,
    every_steps: int | None,
    *,
    mx_module: Any | None = None,
    synchronize: bool = True,
) -> ClearCacheEvent | None:
    """Call ``mx.clear_cache`` when a configured cadence fires.

    The helper is intentionally fail-closed: if cadence is configured and the
    current step should clear cache, missing ``mx.clear_cache`` is an error
    rather than silent non-evidence.
    """

    if not should_clear_cache_after_step(step, every_steps):
        return None
    if every_steps is None:
        return None
    checked_step = _validate_positive_int(step, name="step")
    checked_every_steps = _validate_positive_int(every_steps, name="every_steps")

    mx = _load_mlx_core() if mx_module is None else mx_module
    clear_cache = getattr(mx, "clear_cache", None)
    if not callable(clear_cache):
        raise RuntimeError("mlx.core.clear_cache is unavailable")

    before = _optional_int_call(mx, "get_cache_memory")
    cast(Callable[[], Any], clear_cache)()
    if synchronize:
        synchronize_fn = getattr(mx, "synchronize", None)
        if callable(synchronize_fn):
            cast(Callable[[], Any], synchronize_fn)()
    after = _optional_int_call(mx, "get_cache_memory")
    return ClearCacheEvent(
        step=checked_step,
        every_steps=checked_every_steps,
        api_path="mx.clear_cache",
        cache_memory_before_bytes=before,
        cache_memory_after_bytes=after,
    )


def peak_memory_threshold_bytes(
    total_bytes: int,
    *,
    peak_ratio: float = DEFAULT_PEAK_MEMORY_RATIO,
) -> int:
    """Return the strict peak-memory threshold for an M0.6-style receipt."""

    total = _validate_total_bytes(total_bytes)
    ratio = _validate_ratio(peak_ratio, name="peak_ratio")
    return int(total * ratio)


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


def _optional_int_call(obj: Any, name: str) -> int | None:
    fn = getattr(obj, name, None)
    if not callable(fn):
        return None
    try:
        return int(cast(Callable[[], Any], fn)())
    except Exception:
        return None


def memory_limit_api_status(mx_module: Any | None = None) -> MemoryLimitApiStatus:
    """Inspect supported MLX memory-limit APIs without mutating limits."""

    mx = _load_mlx_core() if mx_module is None else mx_module
    root_memory_limit_available = callable(getattr(mx, "set_memory_limit", None))
    metal = getattr(mx, "metal", None)
    metal_memory_limit_available = callable(getattr(metal, "set_memory_limit", None))
    supported_paths: list[str] = []
    if metal_memory_limit_available:
        supported_paths.append("mx.metal.set_memory_limit")
    if root_memory_limit_available:
        supported_paths.append("mx.set_memory_limit")
    return MemoryLimitApiStatus(
        wired_limit_available=callable(getattr(mx, "set_wired_limit", None)),
        root_memory_limit_available=root_memory_limit_available,
        metal_memory_limit_available=metal_memory_limit_available,
        preferred_memory_limit_api_path=supported_paths[0] if supported_paths else None,
        supported_memory_limit_api_paths=tuple(supported_paths),
    )


def runtime_stack_evidence(mx_module: Any | None = None) -> RuntimeStackEvidence:
    """Capture the local MLX/Metal/device stack for memory receipts."""

    mx = _load_mlx_core() if mx_module is None else mx_module
    metal = getattr(mx, "metal", None)
    default_device_fn = getattr(mx, "default_device", None)
    default_device = None
    if callable(default_device_fn):
        try:
            default_device = str(cast(Callable[[], Any], default_device_fn)())
        except Exception:
            default_device = None
    metal_available = None
    metal_is_available = getattr(metal, "is_available", None)
    if callable(metal_is_available):
        try:
            metal_available = bool(cast(Callable[[], Any], metal_is_available)())
        except Exception:
            metal_available = None
    return RuntimeStackEvidence(
        mlx_version=_package_version("mlx"),
        mlx_metal_version=_package_version("mlx-metal"),
        platform_system=platform.system(),
        platform_release=platform.release(),
        macos_version=platform.mac_ver()[0] or None,
        machine=platform.machine(),
        default_device=default_device,
        metal_available=metal_available,
        device_info=_device_info_mapping(mx),
    )


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _device_info_mapping(mx: Any) -> Mapping[str, Any]:
    device_info = getattr(mx, "device_info", None)
    if not callable(device_info):
        return {}
    try:
        info = cast(Callable[[], Any], device_info)()
    except Exception:
        return {}
    return info if isinstance(info, Mapping) else {}


def apply_memory_limit_plan(
    plan: MemoryLimitPlan,
    *,
    mx_module: Any | None = None,
    apply: bool = False,
) -> AppliedMemoryLimits:
    """Apply a precomputed plan to MLX only when ``apply=True``.

    ``mx.set_wired_limit`` and the root or compatibility memory-limit setter
    return the previous limit in current MLX releases. Tests can pass a fake
    ``mx_module`` to verify behavior without touching process-global limits.
    """

    if not isinstance(plan, MemoryLimitPlan):
        raise TypeError("plan must be a MemoryLimitPlan")
    if not apply:
        return AppliedMemoryLimits(plan=plan, applied=False)

    mx = _load_mlx_core() if mx_module is None else mx_module
    set_wired_limit = getattr(mx, "set_wired_limit", None)
    root_set_memory_limit = getattr(mx, "set_memory_limit", None)
    metal = getattr(mx, "metal", None)
    metal_set_memory_limit = getattr(metal, "set_memory_limit", None)
    if not callable(set_wired_limit):
        raise RuntimeError("mlx.core.set_wired_limit is unavailable")
    if callable(metal_set_memory_limit):
        set_memory_limit = metal_set_memory_limit
        memory_limit_api_path = "mx.metal.set_memory_limit"
    elif callable(root_set_memory_limit):
        set_memory_limit = root_set_memory_limit
        memory_limit_api_path = "mx.set_memory_limit"
    else:
        raise RuntimeError("no supported MLX memory limit setter is available")

    typed_set_wired_limit = cast(Callable[[int], int], set_wired_limit)
    typed_set_memory_limit = cast(Callable[[int], int], set_memory_limit)
    previous_wired = int(typed_set_wired_limit(plan.wired_limit_bytes))
    previous_metal = int(typed_set_memory_limit(plan.metal_limit_bytes))
    return AppliedMemoryLimits(
        plan=plan,
        applied=True,
        previous_wired_limit_bytes=previous_wired,
        previous_metal_limit_bytes=previous_metal,
        metal_limit_api_path=memory_limit_api_path,
    )


def _applied_memory_limits_meet_gate(
    applied_limits: AppliedMemoryLimits | None,
    api_status: MemoryLimitApiStatus,
) -> bool:
    if applied_limits is None or not applied_limits.applied:
        return False
    if applied_limits.previous_wired_limit_bytes is None:
        return False
    if applied_limits.previous_metal_limit_bytes is None:
        return False
    if not api_status.wired_limit_available:
        return False
    return (
        applied_limits.metal_limit_api_path
        in api_status.supported_memory_limit_api_paths
    )


def _memory_limit_plan_ratios_meet_gate(plan: MemoryLimitPlan) -> bool:
    return (
        plan.wired_ratio == DEFAULT_WIRED_RATIO
        and plan.metal_ratio == DEFAULT_METAL_RATIO
    )


def _runtime_stack_meets_gate(
    runtime_stack: RuntimeStackEvidence | None,
    total_bytes: int,
) -> bool:
    if runtime_stack is None:
        return False
    device_info = runtime_stack.device_info
    try:
        memory_size = int(device_info.get("memory_size", 0))
    except (TypeError, ValueError):
        return False
    return (
        runtime_stack.mlx_version is not None
        and runtime_stack.mlx_metal_version is not None
        and runtime_stack.platform_system == "Darwin"
        and runtime_stack.machine == "arm64"
        and runtime_stack.metal_available is True
        and bool(runtime_stack.default_device)
        and memory_size == total_bytes
        and bool(device_info.get("device_name"))
    )


def _normalize_clear_cache_events(
    clear_cache_events: Sequence[ClearCacheEvent | Mapping[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    if clear_cache_events is None:
        return None
    normalized: list[dict[str, Any]] = []
    for event in clear_cache_events:
        if isinstance(event, ClearCacheEvent):
            normalized.append(event.to_dict())
        elif isinstance(event, Mapping):
            normalized.append(dict(event))
        else:
            raise TypeError(
                "clear_cache_events entries must be ClearCacheEvent or mappings"
            )
    return normalized


def _clear_cache_event_steps(
    normalized_events: Sequence[Mapping[str, Any]] | None,
) -> list[int]:
    if normalized_events is None:
        return []
    steps: list[int] = []
    for event in normalized_events:
        step = event.get("step")
        if not isinstance(step, int) or step <= 0:
            return []
        steps.append(step)
    return steps


def _clear_cache_event_api_paths(
    normalized_events: Sequence[Mapping[str, Any]] | None,
) -> list[str]:
    if normalized_events is None:
        return []
    paths: list[str] = []
    for event in normalized_events:
        api_path = event.get("api_path")
        if not isinstance(api_path, str) or not api_path:
            return []
        paths.append(api_path)
    return paths


def _clear_cache_event_cadences(
    normalized_events: Sequence[Mapping[str, Any]] | None,
) -> list[int]:
    if normalized_events is None:
        return []
    cadences: list[int] = []
    for event in normalized_events:
        every_steps = event.get("every_steps")
        if not isinstance(every_steps, int) or every_steps <= 0:
            return []
        cadences.append(every_steps)
    return cadences


def _run_command_meets_gate(
    run_command: Sequence[str] | None,
    required_tokens: Sequence[str],
) -> bool:
    if run_command is None:
        return False
    command_parts = set(run_command)
    return all(token in command_parts for token in required_tokens)


def memory_limit_receipt(
    *,
    total_bytes: int,
    peak_memory_bytes: int | None,
    measured_steps: int | None,
    clear_cache_every_steps: int | None,
    clear_cache_event_count: int | None = None,
    clear_cache_events: Sequence[ClearCacheEvent | Mapping[str, Any]] | None = None,
    model_profile: str | None = None,
    optimizer_name: str | None = None,
    grad_checkpoint_enabled: bool | None = None,
    plan: MemoryLimitPlan | None = None,
    applied_limits: AppliedMemoryLimits | None = None,
    api_status: MemoryLimitApiStatus | None = None,
    runtime_stack: RuntimeStackEvidence | None = None,
    mx_module: Any | None = None,
    peak_ratio: float = DEFAULT_PEAK_MEMORY_RATIO,
    required_steps: int = M06_REQUIRED_STEPS,
    required_model_profile: str = "local_gb10_quarter",
    required_optimizer_name: str = "AdamW",
    run_command: Sequence[str] | None = None,
    required_run_command_tokens: Sequence[str] = (
        "local_gb10_quarter",
        "--grad-checkpoint",
        "--apply-memory-limit-plan",
        "--clear-cache-every-steps",
    ),
    full_acceptance_claim: bool = False,
    issue_id: str = "cppmega-mlx-t8f.6",
    receipt_scope: str = M06_MEMORY_RECEIPT_SCOPE,
    source: str | None = None,
    blockers: Sequence[Mapping[str, Any]] = (),
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """Build a conservative M0.6 memory receipt.

    The receipt separates arithmetic/profile evidence from the full milestone
    gate. A peak below threshold is not enough to claim M0.6 unless the caller
    also records the required 100-step full-model run and clear-cache cadence.
    """

    total = _validate_total_bytes(total_bytes)
    threshold = peak_memory_threshold_bytes(total, peak_ratio=peak_ratio)
    if plan is None:
        plan = memory_limit_plan(total)
    if plan.total_bytes != total:
        raise ValueError("plan.total_bytes must match total_bytes")
    if api_status is None:
        api_status = memory_limit_api_status(mx_module)
    if runtime_stack is None:
        runtime_stack = runtime_stack_evidence(mx_module)
    if applied_limits is not None and applied_limits.plan != plan:
        raise ValueError("applied_limits.plan must match plan")
    if required_steps <= 0:
        raise ValueError("required_steps must be positive")
    if measured_steps is not None and measured_steps < 0:
        raise ValueError("measured_steps must be non-negative")
    if clear_cache_every_steps is not None and clear_cache_every_steps <= 0:
        raise ValueError("clear_cache_every_steps must be positive when provided")
    if clear_cache_event_count is not None and clear_cache_event_count < 0:
        raise ValueError("clear_cache_event_count must be non-negative when provided")
    if run_command is not None and not all(
        isinstance(part, str) for part in run_command
    ):
        raise TypeError("run_command entries must be strings")
    normalized_clear_cache_events = _normalize_clear_cache_events(clear_cache_events)
    if (
        normalized_clear_cache_events is not None
        and clear_cache_event_count is not None
        and len(normalized_clear_cache_events) != clear_cache_event_count
    ):
        raise ValueError("clear_cache_event_count must match clear_cache_events length")

    peak_below_threshold = None
    if peak_memory_bytes is not None:
        if peak_memory_bytes < 0:
            raise ValueError("peak_memory_bytes must be non-negative when provided")
        peak_below_threshold = peak_memory_bytes < threshold

    measured_steps_meet_gate = (
        measured_steps is not None and measured_steps >= required_steps
    )
    clear_cache_cadence_recorded = clear_cache_every_steps is not None
    expected_clear_cache_event_count = None
    expected_clear_cache_steps: list[int] = []
    if measured_steps is not None and clear_cache_every_steps is not None:
        expected_clear_cache_event_count = measured_steps // clear_cache_every_steps
        expected_clear_cache_steps = list(
            range(clear_cache_every_steps, measured_steps + 1, clear_cache_every_steps)
        )
    clear_cache_event_steps = _clear_cache_event_steps(normalized_clear_cache_events)
    clear_cache_event_api_paths = _clear_cache_event_api_paths(
        normalized_clear_cache_events
    )
    clear_cache_event_cadences = _clear_cache_event_cadences(
        normalized_clear_cache_events
    )
    clear_cache_count_meets_gate = (
        expected_clear_cache_event_count is not None
        and expected_clear_cache_event_count > 0
        and clear_cache_event_count is not None
        and clear_cache_event_count == expected_clear_cache_event_count
    )
    clear_cache_event_sequence_meets_gate = (
        bool(expected_clear_cache_steps)
        and clear_cache_event_steps == expected_clear_cache_steps
        and clear_cache_event_api_paths
        == ["mx.clear_cache"] * len(expected_clear_cache_steps)
        and clear_cache_event_cadences
        == [clear_cache_every_steps] * len(expected_clear_cache_steps)
    )
    clear_cache_events_meet_gate = (
        clear_cache_count_meets_gate and clear_cache_event_sequence_meets_gate
    )
    memory_limits_applied_meet_gate = _applied_memory_limits_meet_gate(
        applied_limits,
        api_status,
    )
    memory_limit_ratios_meet_gate = _memory_limit_plan_ratios_meet_gate(plan)
    model_profile_meets_gate = model_profile == required_model_profile
    optimizer_meets_gate = optimizer_name == required_optimizer_name
    grad_checkpoint_meets_gate = grad_checkpoint_enabled is True
    run_command_meets_gate = _run_command_meets_gate(
        run_command,
        required_run_command_tokens,
    )
    runtime_stack_meets_gate = _runtime_stack_meets_gate(runtime_stack, total)
    full_gate_satisfied = (
        full_acceptance_claim
        and peak_below_threshold is True
        and measured_steps_meet_gate
        and clear_cache_cadence_recorded
        and clear_cache_events_meet_gate
        and memory_limits_applied_meet_gate
        and memory_limit_ratios_meet_gate
        and model_profile_meets_gate
        and optimizer_meets_gate
        and grad_checkpoint_meets_gate
        and run_command_meets_gate
        and runtime_stack_meets_gate
    )

    return {
        "receipt_schema_version": 2,
        "receipt_scope": receipt_scope,
        "issue": {"id": issue_id},
        "status": "accepted" if full_gate_satisfied else "partial",
        "full_m0_6_acceptance_claim": full_gate_satisfied,
        "total_memory_bytes": total,
        "peak_threshold_ratio": peak_ratio,
        "peak_threshold_bytes": threshold,
        "peak_memory_bytes": peak_memory_bytes,
        "peak_memory_below_threshold": peak_below_threshold,
        "required_steps": required_steps,
        "measured_steps": measured_steps,
        "measured_steps_meet_gate": measured_steps_meet_gate,
        "required_model_profile": required_model_profile,
        "model_profile": model_profile,
        "model_profile_meets_gate": model_profile_meets_gate,
        "required_optimizer_name": required_optimizer_name,
        "optimizer_name": optimizer_name,
        "optimizer_meets_gate": optimizer_meets_gate,
        "grad_checkpoint_enabled": grad_checkpoint_enabled,
        "grad_checkpoint_meets_gate": grad_checkpoint_meets_gate,
        "run_command": list(run_command) if run_command is not None else None,
        "run_command_recorded": run_command is not None,
        "required_run_command_tokens": list(required_run_command_tokens),
        "run_command_meets_gate": run_command_meets_gate,
        "runtime_stack_recorded": runtime_stack is not None,
        "runtime_stack_meets_gate": runtime_stack_meets_gate,
        "runtime_stack": runtime_stack.to_dict() if runtime_stack is not None else None,
        "clear_cache_every_steps": clear_cache_every_steps,
        "clear_cache_cadence_recorded": clear_cache_cadence_recorded,
        "expected_clear_cache_event_count": expected_clear_cache_event_count,
        "expected_clear_cache_steps": expected_clear_cache_steps,
        "clear_cache_event_count": clear_cache_event_count,
        "clear_cache_event_steps": clear_cache_event_steps,
        "clear_cache_event_api_paths": clear_cache_event_api_paths,
        "clear_cache_event_cadences": clear_cache_event_cadences,
        "clear_cache_event_count_meets_gate": clear_cache_count_meets_gate,
        "clear_cache_event_sequence_meets_gate": clear_cache_event_sequence_meets_gate,
        "clear_cache_events": normalized_clear_cache_events,
        "clear_cache_events_meet_gate": clear_cache_events_meet_gate,
        "memory_limits_applied_meet_gate": memory_limits_applied_meet_gate,
        "required_wired_ratio": DEFAULT_WIRED_RATIO,
        "required_metal_ratio": DEFAULT_METAL_RATIO,
        "memory_limit_ratios_meet_gate": memory_limit_ratios_meet_gate,
        "memory_limit_plan": plan.to_dict(),
        "memory_limit_api_status": api_status.to_dict(),
        "applied_memory_limits": (
            applied_limits.to_dict() if applied_limits is not None else None
        ),
        "memory_profile_source": source,
        "blockers": list(blockers),
        "notes": list(notes),
    }


def _load_mlx_core() -> Any:
    import mlx.core as mx

    return mx


__all__ = [
    "AppliedMemoryLimits",
    "ClearCacheEvent",
    "DEFAULT_METAL_RATIO",
    "DEFAULT_PEAK_MEMORY_RATIO",
    "DEFAULT_WIRED_RATIO",
    "M06_MEMORY_RECEIPT_SCOPE",
    "M06_REQUIRED_STEPS",
    "MemoryLimitApiStatus",
    "MemoryLimitPlan",
    "RuntimeStackEvidence",
    "apply_memory_limit_plan",
    "device_total_memory_bytes",
    "maybe_clear_cache_after_step",
    "memory_limit_api_status",
    "memory_limit_receipt",
    "memory_limit_plan",
    "peak_memory_threshold_bytes",
    "runtime_stack_evidence",
    "should_clear_cache_after_step",
]

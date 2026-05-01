"""Lightweight MLX profiling helpers for train/eval steps."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import mlx.core as mx


JsonScalar = bool | int | float | str | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]


def _json_safe(value: Any) -> JsonValue:
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_safe(item) for item in value]
    return str(value)


def _json_safe_mapping(value: Mapping[str, Any]) -> JsonObject:
    return {str(key): _json_safe(item) for key, item in value.items()}


def _optional_memory_bytes(name: str) -> tuple[int | None, str | None]:
    getter = getattr(mx, name, None)
    if getter is None:
        return None, f"mlx.core.{name} unavailable"
    try:
        return int(getter()), None
    except Exception as exc:  # pragma: no cover - backend/version dependent.
        return None, f"mlx.core.{name} failed: {exc}"


def reset_peak_memory() -> bool:
    """Reset MLX peak-memory accounting when the runtime exposes it."""

    reset = getattr(mx, "reset_peak_memory", None)
    if reset is None:
        return False
    try:
        reset()
    except Exception:  # pragma: no cover - backend/version dependent.
        return False
    return True


def synchronize() -> bool:
    """Synchronize the default MLX stream when supported."""

    sync = getattr(mx, "synchronize", None)
    if sync is None:
        return False
    sync()
    return True


def evaluate(*args: Any) -> bool:
    """Evaluate MLX arrays/pytrees when any arguments were provided."""

    if not args:
        return False
    mx.eval(*args)
    return True


@dataclass(frozen=True)
class ProfileContext:
    """Route/backend/device metadata attached to each measured profile scope."""

    route: str | None = None
    backend: str | None = None
    device: str | None = None
    model_route: str | None = None
    route_plan: Mapping[str, Any] = field(default_factory=dict)
    backend_plan: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> JsonObject:
        payload: dict[str, Any] = {
            "route": self.route,
            "backend": self.backend,
            "device": self.device,
            "model_route": self.model_route,
            "route_plan": dict(self.route_plan),
            "backend_plan": dict(self.backend_plan),
        }
        return _json_safe_mapping(
            {key: value for key, value in payload.items() if value not in (None, {}, ())}
        )


def profile_context(
    *,
    route: str | None = None,
    backend: str | None = None,
    device: str | None = None,
    model_route: str | None = None,
    route_plan: Mapping[str, Any] | None = None,
    backend_plan: Mapping[str, Any] | None = None,
) -> JsonObject:
    """Return JSON-safe route/backend/device context for profile metadata."""

    return ProfileContext(
        route=route,
        backend=backend,
        device=device,
        model_route=model_route,
        route_plan=route_plan or {},
        backend_plan=backend_plan or {},
    ).to_dict()


@dataclass(frozen=True)
class MemorySnapshot:
    """MLX allocator memory counters in bytes, feature-detected at runtime."""

    active_bytes: int | None
    peak_bytes: int | None
    cache_bytes: int | None
    available: bool
    errors: tuple[str, ...] = ()

    @classmethod
    def read(cls) -> "MemorySnapshot":
        active, active_error = _optional_memory_bytes("get_active_memory")
        peak, peak_error = _optional_memory_bytes("get_peak_memory")
        cache, cache_error = _optional_memory_bytes("get_cache_memory")
        errors = tuple(
            error
            for error in (active_error, peak_error, cache_error)
            if error is not None
        )
        return cls(
            active_bytes=active,
            peak_bytes=peak,
            cache_bytes=cache,
            available=any(value is not None for value in (active, peak, cache)),
            errors=errors,
        )

    def to_dict(self) -> JsonObject:
        return {
            "active_bytes": self.active_bytes,
            "peak_bytes": self.peak_bytes,
            "cache_bytes": self.cache_bytes,
            "available": self.available,
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class ProfileMetrics:
    """JSON-serializable timing and memory metrics for one measured scope."""

    label: str
    seconds: float
    tokens: int | None
    tokens_per_second: float | None
    memory: MemorySnapshot
    peak_memory_reset: bool
    synchronized: bool
    evaluated: bool
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def peak_memory_bytes(self) -> int | None:
        return self.memory.peak_bytes

    @property
    def active_memory_bytes(self) -> int | None:
        return self.memory.active_bytes

    @property
    def cache_memory_bytes(self) -> int | None:
        return self.memory.cache_bytes

    def to_dict(self) -> JsonObject:
        return {
            "label": self.label,
            "seconds": _json_safe(self.seconds),
            "wall_time_s": _json_safe(self.seconds),
            "elapsed_wall_time_s": _json_safe(self.seconds),
            "tokens": self.tokens,
            "tokens_per_second": _json_safe(self.tokens_per_second),
            "peak_memory_bytes": self.peak_memory_bytes,
            "active_memory_bytes": self.active_memory_bytes,
            "cache_memory_bytes": self.cache_memory_bytes,
            "memory": self.memory.to_dict(),
            "peak_memory_reset": self.peak_memory_reset,
            "synchronized": self.synchronized,
            "evaluated": self.evaluated,
            "extra": _json_safe_mapping(self.extra),
        }


@dataclass(frozen=True)
class HotspotEvidence:
    """One measured hotspot candidate for a future custom kernel."""

    name: str
    seconds: float
    total_seconds: float
    calls: int = 1
    source: str = "profile_step"
    tokens: int | None = None
    tokens_per_second: float | None = None
    route: str | None = None
    backend: str | None = None
    operation: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("hotspot name must be non-empty")
        if not self.source:
            raise ValueError("hotspot source must be non-empty")
        if not math.isfinite(self.seconds) or self.seconds < 0:
            raise ValueError("hotspot seconds must be a finite non-negative value")
        if not math.isfinite(self.total_seconds) or self.total_seconds < 0:
            raise ValueError("hotspot total_seconds must be a finite non-negative value")
        if self.total_seconds < self.seconds:
            raise ValueError("hotspot total_seconds must be >= seconds")
        if self.calls <= 0:
            raise ValueError("hotspot calls must be positive")

    @property
    def fraction(self) -> float:
        if self.total_seconds <= 0:
            return 0.0
        return self.seconds / self.total_seconds

    def to_dict(self) -> JsonObject:
        return {
            "name": self.name,
            "seconds": _json_safe(self.seconds),
            "total_seconds": _json_safe(self.total_seconds),
            "fraction": _json_safe(self.fraction),
            "calls": self.calls,
            "source": self.source,
            "tokens": self.tokens,
            "tokens_per_second": _json_safe(self.tokens_per_second),
            "route": self.route,
            "backend": self.backend,
            "operation": self.operation,
            "extra": _json_safe_mapping(self.extra),
        }


@dataclass(frozen=True)
class KernelAdoptionAssessment:
    """Fail-closed custom-kernel adoption verdict backed by profile evidence."""

    candidate_kernel: str
    allowed: bool
    reason: str
    evidence: tuple[HotspotEvidence, ...]
    min_hotspot_fraction: float
    min_hotspot_seconds: float
    min_samples: int

    @property
    def top_hotspot(self) -> HotspotEvidence | None:
        if not self.evidence:
            return None
        return max(self.evidence, key=lambda item: (item.seconds, item.fraction))

    def to_dict(self) -> JsonObject:
        top_hotspot = self.top_hotspot
        return {
            "candidate_kernel": self.candidate_kernel,
            "allowed": self.allowed,
            "reason": self.reason,
            "min_hotspot_fraction": _json_safe(self.min_hotspot_fraction),
            "min_hotspot_seconds": _json_safe(self.min_hotspot_seconds),
            "min_samples": self.min_samples,
            "sample_count": len(self.evidence),
            "top_hotspot": top_hotspot.to_dict() if top_hotspot is not None else None,
            "summary": summarize_hotspots(self.evidence),
        }


class KernelAdoptionBlocked(RuntimeError):
    """Raised when a custom kernel is requested without sufficient evidence."""


def _mapping_value(value: Mapping[str, Any], key: str) -> Any:
    item = value.get(key)
    if item is not None:
        return item
    context = value.get("context")
    if isinstance(context, Mapping):
        return context.get(key)
    return None


def hotspot_from_profile_metrics(
    metrics: ProfileMetrics,
    *,
    name: str | None = None,
    source: str = "profile_step",
    total_seconds: float | None = None,
    calls: int = 1,
    extra: Mapping[str, Any] | None = None,
) -> HotspotEvidence:
    """Convert a measured scope into a hotspot evidence record."""

    evidence_extra: dict[str, Any] = dict(metrics.extra)
    if extra:
        evidence_extra.update(extra)
    hotspot_name = name or str(_mapping_value(evidence_extra, "operation") or metrics.label)
    route = _mapping_value(evidence_extra, "route")
    backend = _mapping_value(evidence_extra, "backend")
    operation = _mapping_value(evidence_extra, "operation")
    return HotspotEvidence(
        name=hotspot_name,
        seconds=metrics.seconds,
        total_seconds=metrics.seconds if total_seconds is None else total_seconds,
        calls=calls,
        source=source,
        tokens=metrics.tokens,
        tokens_per_second=metrics.tokens_per_second,
        route=str(route) if route is not None else None,
        backend=str(backend) if backend is not None else None,
        operation=str(operation) if operation is not None else None,
        extra=evidence_extra,
    )


def summarize_hotspots(
    evidence: Iterable[HotspotEvidence],
    *,
    top_n: int = 5,
) -> JsonObject:
    """Return a stable JSON-safe summary of measured hotspot evidence."""

    if top_n <= 0:
        raise ValueError("top_n must be positive")
    rows = sorted(tuple(evidence), key=lambda item: (item.seconds, item.fraction), reverse=True)
    total_hotspot_seconds = sum(item.seconds for item in rows)
    max_total_seconds = max((item.total_seconds for item in rows), default=0.0)
    top_rows = rows[:top_n]
    return {
        "count": len(rows),
        "total_hotspot_seconds": _json_safe(total_hotspot_seconds),
        "max_total_seconds": _json_safe(max_total_seconds),
        "top_n": top_n,
        "hotspots": [item.to_dict() for item in top_rows],
    }


def assess_kernel_adoption(
    candidate_kernel: str,
    evidence: Iterable[HotspotEvidence],
    *,
    min_hotspot_fraction: float = 0.10,
    min_hotspot_seconds: float = 0.001,
    min_samples: int = 1,
) -> KernelAdoptionAssessment:
    """Assess whether custom-kernel work may proceed from measured evidence."""

    if not candidate_kernel:
        raise ValueError("candidate_kernel must be non-empty")
    if not math.isfinite(min_hotspot_fraction) or min_hotspot_fraction < 0:
        raise ValueError("min_hotspot_fraction must be a finite non-negative value")
    if not math.isfinite(min_hotspot_seconds) or min_hotspot_seconds < 0:
        raise ValueError("min_hotspot_seconds must be a finite non-negative value")
    if min_samples <= 0:
        raise ValueError("min_samples must be positive")

    rows = tuple(evidence)
    if len(rows) < min_samples:
        return KernelAdoptionAssessment(
            candidate_kernel=candidate_kernel,
            allowed=False,
            reason=(
                f"blocked: need at least {min_samples} profile sample(s), "
                f"got {len(rows)}"
            ),
            evidence=rows,
            min_hotspot_fraction=min_hotspot_fraction,
            min_hotspot_seconds=min_hotspot_seconds,
            min_samples=min_samples,
        )

    top_hotspot = max(rows, key=lambda item: (item.seconds, item.fraction))
    if top_hotspot.seconds < min_hotspot_seconds:
        return KernelAdoptionAssessment(
            candidate_kernel=candidate_kernel,
            allowed=False,
            reason=(
                "blocked: top hotspot "
                f"{top_hotspot.name!r} measured {top_hotspot.seconds:.6f}s, "
                f"below required {min_hotspot_seconds:.6f}s"
            ),
            evidence=rows,
            min_hotspot_fraction=min_hotspot_fraction,
            min_hotspot_seconds=min_hotspot_seconds,
            min_samples=min_samples,
        )
    if top_hotspot.fraction < min_hotspot_fraction:
        return KernelAdoptionAssessment(
            candidate_kernel=candidate_kernel,
            allowed=False,
            reason=(
                "blocked: top hotspot "
                f"{top_hotspot.name!r} accounts for {top_hotspot.fraction:.3f} "
                f"of profile time, below required {min_hotspot_fraction:.3f}"
            ),
            evidence=rows,
            min_hotspot_fraction=min_hotspot_fraction,
            min_hotspot_seconds=min_hotspot_seconds,
            min_samples=min_samples,
        )

    return KernelAdoptionAssessment(
        candidate_kernel=candidate_kernel,
        allowed=True,
        reason=(
            "allowed: measured hotspot "
            f"{top_hotspot.name!r} accounts for {top_hotspot.fraction:.3f} "
            f"of profile time across {len(rows)} sample(s)"
        ),
        evidence=rows,
        min_hotspot_fraction=min_hotspot_fraction,
        min_hotspot_seconds=min_hotspot_seconds,
        min_samples=min_samples,
    )


def require_kernel_hotspot_evidence(
    candidate_kernel: str,
    evidence: Iterable[HotspotEvidence],
    *,
    min_hotspot_fraction: float = 0.10,
    min_hotspot_seconds: float = 0.001,
    min_samples: int = 1,
) -> KernelAdoptionAssessment:
    """Return the adoption assessment or fail closed with an explicit reason."""

    assessment = assess_kernel_adoption(
        candidate_kernel,
        evidence,
        min_hotspot_fraction=min_hotspot_fraction,
        min_hotspot_seconds=min_hotspot_seconds,
        min_samples=min_samples,
    )
    if not assessment.allowed:
        raise KernelAdoptionBlocked(assessment.reason)
    return assessment


class StepProfiler:
    """Context manager for synchronized MLX step timing and memory sampling."""

    def __init__(
        self,
        label: str,
        *,
        tokens: int | None = None,
        eval_args: tuple[Any, ...] = (),
        reset_peak: bool = True,
        sync: bool = True,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        if tokens is not None and tokens < 0:
            raise ValueError("tokens must be non-negative")
        self.label = label
        self.tokens = tokens
        self.reset_peak = reset_peak
        self.sync = sync
        self.extra = _json_safe_mapping(extra or {})
        self._eval_args = list(eval_args)
        self._start: float | None = None
        self._peak_memory_reset = False
        self._start_synchronized = False
        self._metrics: ProfileMetrics | None = None

    @property
    def metrics(self) -> ProfileMetrics:
        if self._metrics is None:
            raise RuntimeError("profile metrics are unavailable before context exit")
        return self._metrics

    def add_eval_args(self, *args: Any) -> None:
        """Add arrays/pytrees to force with mx.eval when the scope exits."""

        self._eval_args.extend(args)

    def __enter__(self) -> "StepProfiler":
        if self.reset_peak:
            self._peak_memory_reset = reset_peak_memory()
        if self.sync:
            self._start_synchronized = synchronize()
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> None:
        if exc_type is not None:
            return None
        if self._start is None:
            raise RuntimeError("profile context was not entered")

        evaluated = evaluate(*self._eval_args)
        end_synchronized = synchronize() if self.sync else False
        elapsed = time.perf_counter() - self._start
        tokens_per_second = (
            self.tokens / elapsed
            if self.tokens is not None and elapsed > 0
            else None
        )
        self._metrics = ProfileMetrics(
            label=self.label,
            seconds=elapsed,
            tokens=self.tokens,
            tokens_per_second=tokens_per_second,
            memory=MemorySnapshot.read(),
            peak_memory_reset=self._peak_memory_reset,
            synchronized=self._start_synchronized or end_synchronized,
            evaluated=evaluated,
            extra=self.extra,
        )
        return None


def profile_step(
    label: str = "step",
    *,
    tokens: int | None = None,
    eval_args: tuple[Any, ...] = (),
    reset_peak: bool = True,
    sync: bool = True,
    extra: Mapping[str, Any] | None = None,
) -> StepProfiler:
    """Return a profiling context manager for one train/eval step."""

    return StepProfiler(
        label,
        tokens=tokens,
        eval_args=eval_args,
        reset_peak=reset_peak,
        sync=sync,
        extra=extra,
    )


__all__ = [
    "HotspotEvidence",
    "KernelAdoptionAssessment",
    "KernelAdoptionBlocked",
    "MemorySnapshot",
    "ProfileMetrics",
    "ProfileContext",
    "StepProfiler",
    "assess_kernel_adoption",
    "evaluate",
    "hotspot_from_profile_metrics",
    "profile_context",
    "profile_step",
    "require_kernel_hotspot_evidence",
    "reset_peak_memory",
    "summarize_hotspots",
    "synchronize",
]

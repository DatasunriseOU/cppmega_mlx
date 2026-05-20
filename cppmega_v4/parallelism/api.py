"""GUI-facing public API for the distributed planning layer.

One-shot ``verify_distributed_plan(build_spec, sharding)`` returns the
:class:`DistributedMemoryReport` AND every fired :class:`Gotcha`, plus
an elapsed-ms timer for the real-time GUI inner loop.

Designed to run in <100 ms per call on any of the 12 architecture
presets at production dim_envs (B=1, S=4096, H=4096) on any topology —
the test suite enforces this.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from cppmega_v4.buildspec import ModelBuildSpec
from cppmega_v4.parallelism.distributed_memory import (
    DistributedMemoryReport,
    estimate_distributed_memory,
)
from cppmega_v4.parallelism.gotcha_checker import (
    Gotcha,
    GotchaSeverity,
    check_gotchas,
)
from cppmega_v4.parallelism.sharding_spec import ShardingSpec


@dataclass(frozen=True)
class DistributedVerificationResult:
    """One-shot result of :func:`verify_distributed_plan`."""

    memory: DistributedMemoryReport
    gotchas: tuple[Gotcha, ...]
    elapsed_ms: float

    @property
    def has_errors(self) -> bool:
        return any(g.severity is GotchaSeverity.ERROR for g in self.gotchas)

    @property
    def errors(self) -> tuple[Gotcha, ...]:
        return tuple(
            g for g in self.gotchas if g.severity is GotchaSeverity.ERROR
        )

    @property
    def warnings(self) -> tuple[Gotcha, ...]:
        return tuple(
            g for g in self.gotchas if g.severity is GotchaSeverity.WARNING
        )

    def summary(self) -> dict[str, Any]:
        """Compact dict for GUI rendering."""
        return {
            "errors":   len(self.errors),
            "warnings": len(self.warnings),
            "memory":   self.memory.summary(),
            "fits":     self.memory.fits_on_topology(),
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


def verify_distributed_plan(
    build_spec: ModelBuildSpec,
    sharding: ShardingSpec,
    *,
    training: bool = True,
) -> DistributedVerificationResult:
    """One-shot verification: memory + gotchas.

    Args:
      build_spec: post-rewrite ModelBuildSpec (apply MTPRewriter first if MTP).
      sharding: which DeviceTopology + ParallelismKind strategy to model.
      training: when False, gradients / optimizer state are 0; kv_cache
        becomes meaningful.

    Returns:
      DistributedVerificationResult carrying the full memory report,
      the tuple of every fired Gotcha, and an elapsed_ms timer.
    """
    t0 = time.perf_counter()
    memory = estimate_distributed_memory(build_spec, sharding, training=training)
    gotchas = check_gotchas(sharding, build_spec)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return DistributedVerificationResult(
        memory=memory, gotchas=gotchas, elapsed_ms=elapsed_ms,
    )


__all__ = [
    "DistributedVerificationResult",
    "verify_distributed_plan",
]

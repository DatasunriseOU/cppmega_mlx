"""Pipeline orchestrator — walk a list of stages, collect StageResults.

Loads pipeline.yaml manifests, validates stage names, then dispatches
to :data:`cppmega_v4.runner.stages.STAGE_REGISTRY`.

Stops on the first failed stage unless ``continue_on_failure=True`` is
set in the manifest. Skipped stages don't count as failures.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner.stages import (
    SMOKE_STAGES,
    STAGE_REGISTRY,
    StageContext,
    StageResult,
)


@dataclass
class Pipeline:
    """Declarative manifest: which stages, in order, with options."""

    stages: tuple[str, ...]
    stage_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    continue_on_failure: bool = False

    def __post_init__(self) -> None:
        unknown = [s for s in self.stages if s not in STAGE_REGISTRY]
        if unknown:
            raise ValueError(
                f"unknown stage(s) {unknown!r}; "
                f"available: {sorted(STAGE_REGISTRY)}"
            )

    @classmethod
    def smoke(cls) -> Pipeline:
        return cls(stages=SMOKE_STAGES)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Pipeline:
        data = yaml.safe_load(Path(path).read_text())
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Pipeline:
        stages = data.get("stages")
        if not stages:
            raise ValueError("pipeline manifest missing 'stages'")
        return cls(
            stages=tuple(stages),
            stage_options=dict(data.get("stage_options", {})),
            continue_on_failure=bool(data.get("continue_on_failure", False)),
        )


@dataclass
class PipelineReport:
    """Final pipeline run rollup."""

    stages: list[StageResult]
    overall_status: str
    total_elapsed_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages": [s.to_dict() for s in self.stages],
            "overall_status": self.overall_status,
            "total_elapsed_ms": round(self.total_elapsed_ms, 3),
        }


def run_pipeline(spec: VerifyParams, pipeline: Pipeline) -> PipelineReport:
    """Run ``pipeline`` over ``spec`` and return the report.

    Stops on the first failed stage unless ``continue_on_failure`` is set.
    """
    t0 = time.perf_counter()
    ctx = StageContext(spec=spec, options=pipeline.stage_options)
    results: list[StageResult] = []
    for name in pipeline.stages:
        stage = STAGE_REGISTRY[name]
        result = stage(ctx)
        results.append(result)
        if result.status == "fail" and not pipeline.continue_on_failure:
            # Mark every subsequent stage as skipped for visibility.
            for remaining in pipeline.stages[len(results):]:
                results.append(StageResult(
                    name=remaining, status="skipped", elapsed_ms=0.0,
                ))
            break
    overall = "ok"
    if any(r.status == "fail" for r in results):
        overall = "fail"
    elapsed = (time.perf_counter() - t0) * 1000.0
    return PipelineReport(
        stages=results, overall_status=overall, total_elapsed_ms=elapsed,
    )

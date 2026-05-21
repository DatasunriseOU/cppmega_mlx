"""``cppmega-run`` pipeline runner.

See ``VisualBuilderPlan.md`` §5.3 for the design.

Stage F-A.2 surface:
  - stages: STAGE_REGISTRY (12 built-in stages, SMOKE_STAGES, FULL_STAGES)
  - pipeline: Pipeline / PipelineReport / run_pipeline
  - cli: ``cppmega-run`` entry point exposed via console-script
"""

from __future__ import annotations

from cppmega_v4.runner.pipeline import (
    Pipeline,
    PipelineReport,
    run_pipeline,
)
from cppmega_v4.runner.stages import (
    FULL_STAGES,
    SMOKE_STAGES,
    STAGE_REGISTRY,
    StageContext,
    StageResult,
)

__all__ = [
    "FULL_STAGES",
    "Pipeline",
    "PipelineReport",
    "SMOKE_STAGES",
    "STAGE_REGISTRY",
    "StageContext",
    "StageResult",
    "run_pipeline",
]

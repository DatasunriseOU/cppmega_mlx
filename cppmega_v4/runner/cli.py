"""``cppmega-run`` — single CLI entry exposing the pipeline runner.

Usage:
    cppmega-run spec.json                          # smoke pipeline, human
    cppmega-run spec.json --json                   # smoke pipeline, machine
    cppmega-run spec.json --stages all             # full 11-stage pipeline
    cppmega-run spec.json --stages parse,verify_build_spec,resolve_shapes
    cppmega-run spec.json --pipeline pipeline.yaml
    cppmega-run spec.json --pipeline pipeline.yaml --continue-on-failure

Exit code 0 iff all stages either ``ok`` or ``skipped``. Exit 1 on any
``fail``. Exit 2 on invalid input (spec parse / pipeline parse / argument
errors).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from pydantic import ValidationError

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner.pipeline import (
    Pipeline,
    PipelineReport,
    run_pipeline,
)
from cppmega_v4.runner.stages import FULL_STAGES, SMOKE_STAGES, STAGE_REGISTRY


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cppmega-run",
        description="Run the cppmega Visual Builder pipeline on a model_spec.json",
    )
    p.add_argument("spec", help="Path to model_spec.json (VerifyParams shape)")
    p.add_argument("--pipeline", help="Path to pipeline.yaml manifest")
    p.add_argument(
        "--stages",
        help="Comma-separated stage list OR 'smoke' / 'all'. "
             "Overrides --pipeline stage list.",
    )
    p.add_argument(
        "--continue-on-failure", action="store_true",
        help="Do not stop on first failed stage.",
    )
    p.add_argument("--json", action="store_true",
                   help="Emit JSON report to stdout.")
    return p


def _resolve_pipeline(args: argparse.Namespace) -> Pipeline:
    if args.pipeline:
        pipe = Pipeline.from_yaml(args.pipeline)
    else:
        pipe = Pipeline.smoke()
    if args.stages:
        stages = _expand_stage_list(args.stages)
        pipe = Pipeline(
            stages=stages,
            stage_options=pipe.stage_options,
            continue_on_failure=pipe.continue_on_failure,
        )
    if args.continue_on_failure:
        pipe = Pipeline(
            stages=pipe.stages,
            stage_options=pipe.stage_options,
            continue_on_failure=True,
        )
    return pipe


def _expand_stage_list(arg: str) -> tuple[str, ...]:
    if arg == "smoke":
        return SMOKE_STAGES
    if arg == "all":
        return FULL_STAGES
    stages = tuple(s.strip() for s in arg.split(",") if s.strip())
    unknown = [s for s in stages if s not in STAGE_REGISTRY]
    if unknown:
        raise SystemExit(
            f"cppmega-run: unknown stage(s) {unknown!r}; "
            f"available: {sorted(STAGE_REGISTRY)}"
        )
    return stages


def _load_spec(path: str) -> VerifyParams:
    raw = json.loads(Path(path).read_text())
    return VerifyParams.model_validate(raw)


def _print_human(report: PipelineReport) -> None:
    for s in report.stages:
        marker = {"ok": "PASS", "fail": "FAIL", "skipped": "SKIP"}[s.status]
        line = f"  [{marker}] {s.name:<22} {s.elapsed_ms:8.2f} ms"
        if s.error:
            line += f"  error={s.error.get('type','?')}: {s.error.get('detail','?')[:80]}"
        print(line)
    print(f"  overall: {report.overall_status} "
          f"total {report.total_elapsed_ms:.2f} ms")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        spec = _load_spec(args.spec)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"cppmega-run: failed to read spec: {exc}", file=sys.stderr)
        return 2
    except ValidationError as exc:
        print(f"cppmega-run: invalid spec: {exc}", file=sys.stderr)
        return 2

    try:
        pipeline = _resolve_pipeline(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"cppmega-run: pipeline error: {exc}", file=sys.stderr)
        return 2

    report = run_pipeline(spec, pipeline)

    if args.json:
        json.dump(report.to_dict(), sys.stdout, indent=2)
        print()
    else:
        _print_human(report)

    return 0 if report.overall_status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

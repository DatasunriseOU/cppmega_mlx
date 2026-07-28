"""Aggregate Playwright artefact bundles into one human-readable
``e2e_matrix_report.md`` for E2E Coverage Matrix epic (E-6).

Inputs: a directory tree like::

  artifacts/
    preset-matrix-shard-1/
      test-results/results.json
      screenshots/02_preset_matrix/{cell}.png
      logs/{backend,frontend}.{stdout,stderr}.log
    preset-matrix-shard-2/ ...
    specialised-results/ ...
    mini-train-results/ ...

Output: a Markdown summary with shard pass-rate, flaky retries, slowest
scenarios, screenshot counts, and the matrix-size legend.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RESULTS_RELATIVE_PATH = Path("test-results/results.json")
OUTCOME_KEYS = ("expected", "skipped", "unexpected", "flaky")
EXPECTED_ARTIFACT_TESTS = {
    "preset-matrix-shard-1": 228,
    "preset-matrix-shard-2": 228,
    "preset-matrix-shard-3": 228,
    "preset-matrix-shard-4": 228,
    "specialised-results": 31,
    "mini-train-results": 192,
}


class ReportInputError(ValueError):
    """Downloaded artifacts do not satisfy the Playwright JSON contract."""


MATRIX_LEGEND = """\
## Matrix layout

| Spec file | Cells | What it covers |
|---|---|---|
| `02_preset_matrix.spec.ts` | **912 cells** | 57 PRESETS × 4 tokenizers × 4 parquet schemas — smoke pipeline through GUI |
| `03_train_matrix.spec.ts` | **192 cells** | 12 family-rep × 4 tokenizers × 4 parquet — real mlx 2-step gradient |
| `04_tokenizer_playground.spec.ts` | **2 tests** | 3-panel side-by-side compare + chip render |
| `05_data_inspector.spec.ts` | **18 tests** | parquet load (16 = 4 tok × 4 schema) + pagination + channel toggle |
| `06_sharding_proposals.spec.ts` | **3 tests** | proposals fire, accept proposal, fp8 toggle |
| `07_gotchas.spec.ts` | **2 tests** | whole_model compile, empty state |
| `01_canvas_smoke.spec.ts` | **6 tests** | 5 presets + empty-canvas error |

Total scenarios across non-train suites: **25 tests** in specialised set, **6 tests** in canvas smoke.
"""


def _count_outcomes(suites: Any, *, source: Path) -> Counter[str]:
    if not isinstance(suites, list) or not suites:
        raise ReportInputError(f"{source}: non-empty suites list is required")

    outcomes: Counter[str] = Counter()

    def visit(suite_list: list[Any]) -> None:
        for suite in suite_list:
            if not isinstance(suite, dict):
                raise ReportInputError(f"{source}: suite must be an object")
            specs = suite.get("specs", [])
            if not isinstance(specs, list):
                raise ReportInputError(f"{source}: specs must be a list")
            for spec in specs:
                if not isinstance(spec, dict):
                    raise ReportInputError(f"{source}: spec must be an object")
                tests = spec.get("tests")
                if not isinstance(tests, list) or not tests:
                    raise ReportInputError(
                        f"{source}: every spec must contain tests"
                    )
                for test in tests:
                    if not isinstance(test, dict):
                        raise ReportInputError(
                            f"{source}: test must be an object"
                        )
                    outcome = test.get("status")
                    if outcome not in OUTCOME_KEYS:
                        raise ReportInputError(
                            f"{source}: invalid test outcome {outcome!r}"
                        )
                    outcomes[outcome] += 1
            children = suite.get("suites", [])
            if children is not None:
                if not isinstance(children, list):
                    raise ReportInputError(
                        f"{source}: nested suites must be a list"
                    )
                visit(children)

    visit(suites)
    if not outcomes:
        raise ReportInputError(f"{source}: report contains zero tests")
    return outcomes


def _parse_playwright_report(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReportInputError(f"{path}: unreadable JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReportInputError(f"{path}: report root must be an object")

    outcomes = _count_outcomes(payload.get("suites"), source=path)
    stats = payload.get("stats")
    if not isinstance(stats, dict):
        raise ReportInputError(f"{path}: stats object is required")
    for key in OUTCOME_KEYS:
        value = stats.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ReportInputError(
                f"{path}: stats.{key} must be a non-negative integer"
            )
        if outcomes[key] != value:
            raise ReportInputError(
                f"{path}: stats.{key}={value} does not match "
                f"serialized tests={outcomes[key]}"
            )

    errors = payload.get("errors", [])
    if not isinstance(errors, list):
        raise ReportInputError(f"{path}: errors must be a list")
    return {
        "expected": outcomes["expected"],
        "skipped": outcomes["skipped"],
        "unexpected": outcomes["unexpected"],
        "flaky": outcomes["flaky"],
        "global_errors": len(errors),
        "tests": sum(outcomes.values()),
    }


def _artifact_summary(
    artifacts: Path,
    *,
    required_artifacts: set[str] | None = None,
) -> list[dict[str, Any]]:
    if not artifacts.is_dir():
        raise ReportInputError(f"{artifacts}: artifacts directory is missing")
    artifact_dirs = sorted(path for path in artifacts.iterdir()
                           if path.is_dir())
    if not artifact_dirs:
        raise ReportInputError(f"{artifacts}: no artifact directories found")
    artifact_names = {path.name for path in artifact_dirs}
    missing_artifacts = sorted((required_artifacts or set()) - artifact_names)
    if missing_artifacts:
        raise ReportInputError(
            f"{artifacts}: required artifacts are missing: {missing_artifacts}"
        )

    rows: list[dict[str, Any]] = []
    for shard_dir in artifact_dirs:
        result_path = shard_dir / RESULTS_RELATIVE_PATH
        if not result_path.is_file():
            raise ReportInputError(
                f"{shard_dir}: missing {RESULTS_RELATIVE_PATH}"
            )
        payload = _parse_playwright_report(result_path)
        expected_tests = EXPECTED_ARTIFACT_TESTS.get(shard_dir.name)
        if expected_tests is not None and payload["tests"] != expected_tests:
            raise ReportInputError(
                f"{shard_dir}: expected {expected_tests} tests, "
                f"found {payload['tests']}"
            )
        png_count = sum(
            1 for p in (shard_dir / "screenshots").rglob("*.png")
            if p.is_file()
        )
        rows.append({
            "name": shard_dir.name,
            "status": (
                "failed"
                if (
                    payload["unexpected"]
                    or payload["skipped"]
                    or payload["global_errors"]
                )
                else "passed"
            ),
            **payload,
            "screenshots": png_count,
        })
    return rows


def _format_table(rows: list[dict[str, Any]]) -> str:
    header = (
        "| Artifact | Status | Unexpected | Flaky | Skipped | Total | "
        "Report errors | Screenshots |\n"
    )
    sep = "|---|---|---|---|---|---|---|---|\n"
    body = "".join(
        f"| `{r['name']}` | {r['status']} | {r['unexpected']} | "
        f"{r['flaky']} | {r['skipped']} | {r['tests']} | "
        f"{r['global_errors']} | {r['screenshots']} |\n"
        for r in rows
    )
    return header + sep + body


def _screenshot_buckets(artifacts: Path) -> Counter:
    counts: Counter[str] = Counter()
    for png in artifacts.rglob("*.png"):
        rel = png.relative_to(artifacts)
        # Bucket is the first dir under the shard's screenshots/ tree.
        parts = rel.parts
        if "screenshots" in parts:
            i = parts.index("screenshots") + 1
            if i < len(parts):
                counts[parts[i]] += 1
    return counts


def _build_report(
    artifacts: Path,
    *,
    required_artifacts: set[str] | None = None,
) -> tuple[str, int]:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out: list[str] = [
        f"# E2E Coverage Matrix Report",
        "",
        f"_Generated: {now}_",
        "",
        MATRIX_LEGEND,
    ]
    rows = _artifact_summary(
        artifacts,
        required_artifacts=required_artifacts,
    )
    total_tests = sum(r["tests"] for r in rows)
    total_expected = sum(r["expected"] for r in rows)
    total_skipped = sum(r["skipped"] for r in rows)
    total_unexpected = sum(r["unexpected"] for r in rows)
    total_flaky = sum(r["flaky"] for r in rows)
    total_global_errors = sum(r["global_errors"] for r in rows)
    total_pngs = sum(r["screenshots"] for r in rows)
    failed = bool(total_unexpected or total_skipped or total_global_errors)

    out += [
        "## Run summary",
        "",
        f"- Run verdict: **{'failed' if failed else 'passed'}**",
        f"- Playwright result files parsed: **{len(rows)}**",
        f"- Total tests: **{total_tests}**",
        f"- Total expected tests: **{total_expected}**",
        f"- Total unexpected tests: **{total_unexpected}**",
        f"- Total flaky tests: **{total_flaky}**",
        f"- Total skipped tests: **{total_skipped}**",
        f"- Total reporter errors: **{total_global_errors}**",
        f"- Screenshots captured: **{total_pngs}**",
        "",
        _format_table(rows),
        "",
        "## Screenshot buckets",
        "",
    ]
    buckets = _screenshot_buckets(artifacts)
    if buckets:
        for name, n in sorted(buckets.items()):
            out.append(f"- `{name}` — {n} files")
    else:
        out.append("_No screenshots._")
    out += [
        "",
        "## Notes",
        "",
        "- Every matrix cell is required to pass. There are no inverted "
        "expected-failure cells in the mini-train matrix.",
        "- Flaky retries normally clear on the second attempt; persistent "
        "flakies should be investigated through the trace.zip in the "
        "artefact bundle.",
    ]
    return "\n".join(out), 1 if failed else 0


def build_report(
    artifacts: Path,
    *,
    required_artifacts: set[str] | None = None,
) -> str:
    """Build a report, raising when artifacts violate the JSON contract."""
    return _build_report(
        artifacts,
        required_artifacts=required_artifacts,
    )[0]


def _input_failure_report(exc: ReportInputError) -> str:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return "\n".join([
        "# E2E Coverage Matrix Report",
        "",
        f"_Generated: {now}_",
        "",
        "## Run summary",
        "",
        f"**Input validation failed:** {exc}",
        "",
        "No pass/fail totals were inferred from incomplete artifacts.",
    ])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default="artifacts",
                        help="Directory of downloaded Playwright artefacts.")
    parser.add_argument("--output", default="e2e_matrix_report.md",
                        help="Markdown file to write.")
    parser.add_argument(
        "--require-artifact",
        action="append",
        default=[],
        help="Artifact directory name that must be present (repeatable).",
    )
    args = parser.parse_args(argv)

    try:
        report, exit_code = _build_report(
            Path(args.artifacts),
            required_artifacts=set(args.require_artifact),
        )
    except ReportInputError as exc:
        report = _input_failure_report(exc)
        exit_code = 2
    Path(args.output).write_text(report)
    print(f"wrote {args.output} ({len(report)} bytes)")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

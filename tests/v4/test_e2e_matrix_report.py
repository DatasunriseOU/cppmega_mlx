"""E-6 matrix-report tests — workflow shape + report generator."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


WORKFLOW = Path(".github/workflows/e2e-matrix.yml")
REPORT_SCRIPT = Path("tools/build_e2e_matrix_report.py")
PLAYWRIGHT_CONFIG = Path("vbgui/e2e/playwright.config.ts")


# ---------------------------------------------------------------------------
# GitHub Actions workflow shape
# ---------------------------------------------------------------------------


def test_workflow_file_exists():
    assert WORKFLOW.is_file()


def test_workflow_triggers_on_push_pr_and_schedule():
    text = WORKFLOW.read_text()
    assert "push:" in text
    assert "pull_request:" in text
    assert "schedule:" in text
    assert "cron:" in text
    assert "workflow_dispatch:" in text


def test_workflow_runs_preset_matrix_in_shards():
    text = WORKFLOW.read_text()
    assert "shard: [1, 2, 3, 4]" in text
    assert "--shard=${{ matrix.shard }}/4" in text


def test_workflow_uploads_artifacts():
    text = WORKFLOW.read_text()
    assert "actions/upload-artifact" in text
    assert "e2e/test-results" in text
    assert "e2e/screenshots" in text


def test_workflow_runs_mini_train_only_on_macos_nightly():
    text = WORKFLOW.read_text()
    assert "runs-on: [self-hosted, macOS, ARM64, cppmega-mlx-macos]" in text
    assert "github.event_name == 'schedule'" in text


def test_workflow_invokes_matrix_report_step():
    text = WORKFLOW.read_text()
    assert "tools/build_e2e_matrix_report.py" in text
    assert "GITHUB_STEP_SUMMARY" in text
    assert "--require-artifact preset-matrix-shard-1" in text
    assert "--require-artifact specialised-results" in text
    assert "--require-artifact mini-train-results" in text


def test_workflow_emits_machine_readable_playwright_results():
    text = PLAYWRIGHT_CONFIG.read_text()
    assert '"github"' in text
    assert '"json"' in text
    assert 'outputFile: "test-results/results.json"' in text


def test_matrix_report_waits_for_nightly_mini_train():
    text = WORKFLOW.read_text()
    assert "needs: [preset-matrix, specialised, mini-train-matrix]" in text


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------


def test_report_script_exists():
    assert REPORT_SCRIPT.is_file()


def _write_playwright_results(
    artifact: Path,
    *,
    expected: int = 1,
    skipped: int = 0,
    unexpected: int = 0,
    flaky: int = 0,
    errors: list[dict[str, str]] | None = None,
) -> None:
    results = artifact / "test-results"
    results.mkdir(parents=True)
    total = expected + skipped + unexpected + flaky
    tests = []
    outcomes = (
        ["expected"] * expected
        + ["skipped"] * skipped
        + ["unexpected"] * unexpected
        + ["flaky"] * flaky
    )
    for i, outcome in enumerate(outcomes):
        tests.append({
            "timeout": 30_000,
            "annotations": [],
            "expectedStatus": "passed",
            "projectId": "chromium",
            "projectName": "chromium",
            "results": [{
                "workerIndex": 0,
                "parallelIndex": 0,
                "status": "failed" if outcome == "unexpected" else "passed",
                "duration": 10,
                "errors": [],
                "stdout": [],
                "stderr": [],
                "retry": 0,
                "startTime": "2026-07-28T00:00:00.000Z",
                "annotations": [],
                "attachments": [],
            }],
            "status": outcome,
        })
    payload = {
        "config": {"rootDir": "/repo/vbgui/e2e"},
        "suites": [{
            "title": "matrix.spec.ts",
            "file": "scenarios/matrix.spec.ts",
            "column": 0,
            "line": 0,
            "specs": [{
                "title": "matrix cell",
                "ok": unexpected == 0,
                "tags": [],
                "tests": tests,
                "id": "matrix-cell",
                "file": "scenarios/matrix.spec.ts",
                "line": 1,
                "column": 1,
            }],
        }],
        "errors": errors or [],
        "stats": {
            "startTime": "2026-07-28T00:00:00.000Z",
            "duration": total * 10,
            "expected": expected,
            "skipped": skipped,
            "unexpected": unexpected,
            "flaky": flaky,
        },
    }
    (results / "results.json").write_text(json.dumps(payload))


def test_report_fails_closed_on_missing_artifacts(tmp_path: Path):
    out = tmp_path / "report.md"
    rc = subprocess.run(
        [sys.executable, str(REPORT_SCRIPT),
         "--artifacts", str(tmp_path / "nonexistent"),
         "--output", str(out)],
        capture_output=True, text=True,
    ).returncode
    assert rc != 0
    text = out.read_text()
    assert "E2E Coverage Matrix Report" in text
    assert "input validation failed" in text.lower()


def test_report_fails_closed_when_required_artifact_was_not_downloaded(
    tmp_path: Path,
):
    artifacts = tmp_path / "artifacts"
    _write_playwright_results(
        artifacts / "preset-matrix-shard-1",
        expected=1,
    )
    out = tmp_path / "report.md"
    rc = subprocess.run(
        [
            sys.executable,
            str(REPORT_SCRIPT),
            "--artifacts",
            str(artifacts),
            "--output",
            str(out),
            "--require-artifact",
            "preset-matrix-shard-1",
            "--require-artifact",
            "specialised-results",
        ],
        capture_output=True,
        text=True,
    ).returncode
    assert rc != 0
    text = out.read_text()
    assert "input validation failed" in text.lower()
    assert "specialised-results" in text


def test_report_aggregates_playwright_json(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    artifact = artifacts / "custom-results"
    _write_playwright_results(artifact, expected=10, flaky=1)
    shots = artifacts / "custom-results" / "screenshots" / "02_preset_matrix"
    shots.mkdir(parents=True)
    (shots / "cell.png").write_bytes(b"\x89PNG")

    out = tmp_path / "report.md"
    rc = subprocess.run(
        [sys.executable, str(REPORT_SCRIPT),
         "--artifacts", str(artifacts), "--output", str(out)],
        capture_output=True, text=True,
    ).returncode
    assert rc == 0
    text = out.read_text()
    assert "Playwright result files parsed: **1**" in text
    assert "Total tests: **11**" in text
    assert "Total flaky tests: **1**" in text
    assert "`02_preset_matrix`" in text
    assert "1 " in text  # screenshot count


def test_report_handles_multiple_shards(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    for i in (1, 2, 3, 4):
        _write_playwright_results(
            artifacts / f"custom-shard-{i}",
            expected=3,
        )
    out = tmp_path / "report.md"
    subprocess.run([sys.executable, str(REPORT_SCRIPT),
                   "--artifacts", str(artifacts), "--output", str(out)],
                   check=True)
    text = out.read_text()
    assert "Playwright result files parsed: **4**" in text
    assert "Total tests: **12**" in text


@pytest.mark.parametrize("case", ["missing", "malformed"])
def test_report_fails_closed_on_unreadable_result_contract(
    tmp_path: Path,
    case: str,
):
    artifact = tmp_path / "artifacts" / "preset-matrix-shard-1"
    results = artifact / "test-results"
    results.mkdir(parents=True)
    if case == "malformed":
        (results / "results.json").write_text("{not-json")
    out = tmp_path / "report.md"
    rc = subprocess.run(
        [sys.executable, str(REPORT_SCRIPT),
         "--artifacts", str(tmp_path / "artifacts"),
         "--output", str(out)],
        capture_output=True, text=True,
    ).returncode
    assert rc != 0
    assert "input validation failed" in out.read_text().lower()


def test_report_returns_failure_when_playwright_has_unexpected_tests(
    tmp_path: Path,
):
    artifacts = tmp_path / "artifacts"
    _write_playwright_results(
        artifacts / "mini-train-results",
        expected=191,
        unexpected=1,
    )
    out = tmp_path / "report.md"
    rc = subprocess.run(
        [sys.executable, str(REPORT_SCRIPT),
         "--artifacts", str(artifacts), "--output", str(out)],
        capture_output=True, text=True,
    ).returncode
    assert rc != 0
    text = out.read_text()
    assert "Total unexpected tests: **1**" in text
    assert "`mini-train-results` | failed | 1" in text


def test_report_returns_failure_when_playwright_skipped_a_test(
    tmp_path: Path,
):
    artifacts = tmp_path / "artifacts"
    _write_playwright_results(
        artifacts / "custom-results",
        expected=10,
        skipped=1,
    )
    out = tmp_path / "report.md"
    rc = subprocess.run(
        [
            sys.executable,
            str(REPORT_SCRIPT),
            "--artifacts",
            str(artifacts),
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
    ).returncode
    assert rc != 0
    text = out.read_text()
    assert "Total skipped tests: **1**" in text
    assert "`custom-results` | failed" in text


def test_report_fails_closed_on_partial_known_matrix_artifact(
    tmp_path: Path,
):
    artifacts = tmp_path / "artifacts"
    _write_playwright_results(
        artifacts / "mini-train-results",
        expected=191,
    )
    out = tmp_path / "report.md"
    rc = subprocess.run(
        [
            sys.executable,
            str(REPORT_SCRIPT),
            "--artifacts",
            str(artifacts),
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
    ).returncode
    assert rc != 0
    text = out.read_text()
    assert "input validation failed" in text.lower()
    assert "expected 192 tests, found 191" in text


def test_report_documents_matrix_sizes():
    """Smoke: report explains the 912 + 192 + 25 + 6 layout for readers."""
    artifacts_root = Path("/tmp/_report_doc_test_artifacts")
    if artifacts_root.exists():
        import shutil
        shutil.rmtree(artifacts_root)
    artifacts_root.mkdir()
    _write_playwright_results(
        artifacts_root / "documentation-results",
        expected=1,
    )
    out = artifacts_root / "out.md"
    subprocess.run([sys.executable, str(REPORT_SCRIPT),
                   "--artifacts", str(artifacts_root), "--output", str(out)],
                   check=True)
    text = out.read_text()
    assert "912 cells" in text
    assert "192 cells" in text
    assert "25 tests" in text
    assert "6 tests" in text

"""E-6 matrix-report tests — workflow shape + report generator."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


WORKFLOW = Path(".github/workflows/e2e-matrix.yml")
REPORT_SCRIPT = Path("tools/build_e2e_matrix_report.py")


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
    assert "macos-latest" in text
    assert "github.event_name == 'schedule'" in text


def test_workflow_invokes_matrix_report_step():
    text = WORKFLOW.read_text()
    assert "tools/build_e2e_matrix_report.py" in text
    assert "GITHUB_STEP_SUMMARY" in text


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------


def test_report_script_exists():
    assert REPORT_SCRIPT.is_file()


def test_report_emits_stub_on_missing_artifacts(tmp_path: Path):
    out = tmp_path / "report.md"
    rc = subprocess.run(
        [sys.executable, str(REPORT_SCRIPT),
         "--artifacts", str(tmp_path / "nonexistent"),
         "--output", str(out)],
        capture_output=True, text=True,
    ).returncode
    assert rc == 0
    text = out.read_text()
    assert "E2E Coverage Matrix Report" in text
    assert "No artefacts" in text


def test_report_aggregates_last_run_json(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    shard = artifacts / "preset-matrix-shard-1" / "test-results"
    shard.mkdir(parents=True)
    (shard / ".last-run.json").write_text(json.dumps({
        "status": "passed",
        "failedTests": [],
        "flakyTests": [{"id": "x"}],
    }))
    shots = artifacts / "preset-matrix-shard-1" / "screenshots" / "02_preset_matrix"
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
    assert "Playwright run files parsed: **1**" in text
    assert "Total flaky retries (across shards): **1**" in text
    assert "`02_preset_matrix`" in text
    assert "1 " in text  # screenshot count


def test_report_handles_multiple_shards(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    for i in (1, 2, 3, 4):
        d = artifacts / f"preset-matrix-shard-{i}" / "test-results"
        d.mkdir(parents=True)
        (d / ".last-run.json").write_text(json.dumps({
            "status": "passed", "failedTests": [], "flakyTests": [],
        }))
    out = tmp_path / "report.md"
    subprocess.run([sys.executable, str(REPORT_SCRIPT),
                   "--artifacts", str(artifacts), "--output", str(out)],
                   check=True)
    text = out.read_text()
    assert "Playwright run files parsed: **4**" in text


def test_report_documents_matrix_sizes():
    """Smoke: report explains the 912 + 192 + 25 + 6 layout for readers."""
    artifacts_root = Path("/tmp/_report_doc_test_artifacts")
    if artifacts_root.exists():
        import shutil
        shutil.rmtree(artifacts_root)
    artifacts_root.mkdir()
    out = artifacts_root / "out.md"
    subprocess.run([sys.executable, str(REPORT_SCRIPT),
                   "--artifacts", str(artifacts_root), "--output", str(out)],
                   check=True)
    text = out.read_text()
    assert "912 cells" in text
    assert "192 cells" in text
    assert "25 tests" in text
    assert "6 tests" in text

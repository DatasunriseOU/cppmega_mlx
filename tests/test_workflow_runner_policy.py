from __future__ import annotations

import re
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HOSTED_RUNNER = re.compile(
    r"^\s*runs-on:\s*.*(?:ubuntu|macos|windows)-latest\s*$",
    re.MULTILINE,
)
JOB_BLOCK = re.compile(
    r"(?ms)^  (?P<name>[A-Za-z0-9_-]+):\n(?P<body>.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)"
)
TRUSTED_PR_GUARDS = (
    "github.event.pull_request.head.repo.full_name == github.repository",
    "github.event.pull_request.head.repo.fork == false",
)
NON_PR_ONLY_GUARD = (
    "github.event_name == 'schedule' || "
    "github.event_name == 'workflow_dispatch'"
)
ACTION_USE = re.compile(
    r"uses:\s*['\"]?(?P<action>actions/[A-Za-z0-9_.-]+)"
    r"@(?P<ref>[^\s'\"#]+)"
)


def test_workflows_do_not_use_github_hosted_runners() -> None:
    violations = []
    for workflow in sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml")):
        if HOSTED_RUNNER.search(workflow.read_text(encoding="utf-8")):
            violations.append(workflow.relative_to(REPO_ROOT).as_posix())

    assert not violations, f"GitHub-hosted runners are forbidden: {violations}"


def test_pull_requests_cannot_execute_on_persistent_self_hosted_runners() -> None:
    violations = []
    for workflow in sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml")):
        text = workflow.read_text(encoding="utf-8")
        if not re.search(r"(?m)^  pull_request:\s*$", text):
            continue
        jobs = text.partition("\njobs:\n")[2]
        for match in JOB_BLOCK.finditer(jobs):
            body = match.group("body")
            if "runs-on: [self-hosted" not in body:
                continue
            guarded = any(marker in body for marker in TRUSTED_PR_GUARDS)
            guarded = guarded or NON_PR_ONLY_GUARD in body
            if not guarded:
                violations.append(
                    f"{workflow.relative_to(REPO_ROOT).as_posix()}:{match.group('name')}"
                )

    assert not violations, (
        "pull_request jobs may not execute untrusted code on persistent "
        f"self-hosted runners: {violations}"
    )


def test_persistent_pr_ci_actions_are_pinned_to_commits() -> None:
    violations = []
    for name in ("ci-self-hosted.yml", "e2e-matrix.yml"):
        workflow = (
            REPO_ROOT / ".github" / "workflows" / name
        ).read_text(encoding="utf-8")
        violations.extend(
            f"{name}:{match.group('action')}@{match.group('ref')}"
            for match in ACTION_USE.finditer(workflow)
            if re.fullmatch(r"[0-9a-f]{40}", match.group("ref")) is None
        )

    assert not violations, f"mutable action references are forbidden: {violations}"


def test_macos_e2e_uses_an_isolated_job_venv() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "e2e-matrix.yml").read_text(
        encoding="utf-8"
    )

    assert "pip install --upgrade pip" not in workflow
    assert 'job_venv="$RUNNER_TEMP/cppmega-mlx-e2e-venv"' not in workflow
    assert workflow.count('mktemp -d "$RUNNER_TEMP/cppmega-mlx-e2e-') == 3
    assert workflow.count("-m venv --system-site-packages") == 3
    assert "--no-build-isolation" not in workflow
    assert workflow.count('-e ".[gui,parquet,widget]"') == 3
    assert workflow.count('echo "VBGUI_E2E_PYTHON=$job_venv/bin/python"') == 3
    assert workflow.count('rm -rf "$VBGUI_E2E_VENV"') == 3


def test_build_backend_declares_mlx_imported_by_setup_py() -> None:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert "mlx>=0.31" in config["build-system"]["requires"]

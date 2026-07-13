from __future__ import annotations

import re
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HOSTED_RUNNER = re.compile(
    r"^\s*runs-on:\s*.*(?:ubuntu|macos|windows)-latest\s*$",
    re.MULTILINE,
)


def test_workflows_do_not_use_github_hosted_runners() -> None:
    violations = []
    for workflow in sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml")):
        if HOSTED_RUNNER.search(workflow.read_text(encoding="utf-8")):
            violations.append(workflow.relative_to(REPO_ROOT).as_posix())

    assert not violations, f"GitHub-hosted runners are forbidden: {violations}"


def test_macos_e2e_uses_an_isolated_job_venv() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "e2e-matrix.yml").read_text(
        encoding="utf-8"
    )

    assert "pip install --upgrade pip" not in workflow
    assert workflow.count("-m venv --system-site-packages") == 3
    assert workflow.count("--no-build-isolation -e") == 3
    assert workflow.count('echo "VBGUI_E2E_PYTHON=$job_venv/bin/python"') == 3


def test_build_backend_declares_mlx_imported_by_setup_py() -> None:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert "mlx>=0.31" in config["build-system"]["requires"]

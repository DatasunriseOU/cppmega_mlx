from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HOSTED_RUNNER = re.compile(
    r"\b(?:ubuntu|macos|windows)-(?:latest|[0-9]+(?:\.[0-9]+)?)\b",
    re.IGNORECASE,
)
RUNS_ON = re.compile(r"(?m)^\s*runs-on:\s*(?P<labels>.+?)\s*$")
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
    r"uses:\s*['\"]?(?P<action>(?!\./)[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)"
    r"@(?P<ref>[^\s'\"#]+)"
)


def test_workflows_do_not_use_github_hosted_runners() -> None:
    violations = []
    for workflow in sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml")):
        text = workflow.read_text(encoding="utf-8")
        if HOSTED_RUNNER.search(text):
            violations.append(f"{workflow.relative_to(REPO_ROOT).as_posix()}:hosted")
        violations.extend(
            f"{workflow.relative_to(REPO_ROOT).as_posix()}:{match.group('labels')}"
            for match in RUNS_ON.finditer(text)
            if "self-hosted" not in match.group("labels")
        )

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


def test_all_self_hosted_workflow_actions_are_pinned_to_commits() -> None:
    violations = []
    for path in sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml")):
        workflow = path.read_text(encoding="utf-8")
        violations.extend(
            f"{path.name}:{match.group('action')}@{match.group('ref')}"
            for match in ACTION_USE.finditer(workflow)
            if re.fullmatch(r"[0-9a-f]{40}", match.group("ref")) is None
        )

    assert not violations, f"mutable action references are forbidden: {violations}"


def test_cuda_lane_does_not_request_a_nonexistent_dev_extra() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "cuda-lane.yml").read_text(
        encoding="utf-8"
    )

    assert '.[dev]' not in workflow
    assert "CPPMEGA_CUDA_PYTHON" in workflow
    assert "check_cuda_environment.py" in workflow
    assert "pip check" in workflow
    assert "--collect-only" in workflow
    assert "setup-python" not in workflow
    assert "uv sync" in workflow
    assert "--locked" in workflow
    assert "--no-sources" in workflow
    assert "--group dev" in workflow
    assert "--extra path-c" in workflow
    assert "UV_PROJECT_ENVIRONMENT" in workflow
    assert "--python \"$CPPMEGA_CUDA_PYTHON\"" in workflow
    assert "pyvenv.cfg" in workflow
    assert "base_prefix" in workflow
    assert "GITHUB_WORKSPACE" in workflow
    assert "uv lock --check" in workflow
    assert 'uv pip check --python "$CPPMEGA_CUDA_PYTHON"' in workflow
    assert '"$CPPMEGA_CUDA_PYTHON" -m pip check' not in workflow
    assert "Upload CUDA environment receipt" in workflow
    assert "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02" in workflow


def test_macos_abi_probe_binds_the_runner_environment_explicitly() -> None:
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "ci-self-hosted.yml"
    ).read_text(encoding="utf-8")

    assert 'scripts/check_mlx_abi.py \\' in workflow
    assert '--python "${python_bin}"' in workflow
    assert "--env-root /Volumes/external/sources/.venvs/cppmega.mlx" in workflow


def test_tcsh_gitlink_has_a_canonical_submodule_mapping() -> None:
    gitmodules = (REPO_ROOT / ".gitmodules").read_text(encoding="utf-8")

    assert '[submodule "vendor/tree-sitter-tcsh"]' in gitmodules
    assert "path = vendor/tree-sitter-tcsh" in gitmodules
    assert "url = https://github.com/j3bit/tree-sitter-tcsh.git" in gitmodules


def test_macos_e2e_uses_an_isolated_job_venv() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "e2e-matrix.yml").read_text(
        encoding="utf-8"
    )

    assert "pip install --upgrade pip" not in workflow
    assert 'job_venv="$RUNNER_TEMP/cppmega-mlx-e2e-venv"' not in workflow
    assert workflow.count('mktemp -d "$RUNNER_TEMP/cppmega-mlx-e2e-') == 3
    assert "--system-site-packages" not in workflow
    assert workflow.count('"$base_python" -m venv "$job_venv"') == 3
    assert workflow.count('echo "PYTHONPATH=" >> "$GITHUB_ENV"') == 3
    assert workflow.count('echo "PYTHONNOUSERSITE=1" >> "$GITHUB_ENV"') == 3
    assert "--no-build-isolation" not in workflow
    assert workflow.count("run: scripts/install_self_hosted_e2e_python.sh") == 3
    assert workflow.count('echo "VBGUI_E2E_PYTHON=$job_venv/bin/python"') == 3
    assert workflow.count('rm -rf "$VBGUI_E2E_VENV"') == 3
    assert workflow.count(
        "base_python=/Volumes/external/sources/.venvs/cppmega.mlx/bin/python"
    ) == 3
    assert "/Volumes/external/sources/cppmega.mlx/.venv/bin/python" not in workflow

    pull_request_paths = workflow.split("  schedule:", 1)[0].split(
        "  pull_request:", 1
    )[1]
    assert '      - "cppmega_mlx/**"' in pull_request_paths
    assert '      - "tests/fixtures/parquet/**"' in pull_request_paths
    assert '      - "tests/fixtures/tokenizers/**"' in pull_request_paths

    installer = (
        REPO_ROOT / "scripts" / "install_self_hosted_e2e_python.sh"
    ).read_text(encoding="utf-8")
    assert '-e "$CPPMEGA_MLX_LM_CHECKOUT"' in installer
    assert '-e "$repo_root[gui,parquet,widget]"' in installer
    assert '"mlx==0.32.0"' in installer
    assert installer.index('"mlx==0.32.0"') < installer.index(
        '-e "$repo_root[gui,parquet,widget]"'
    )
    assert installer.index('-e "$repo_root[gui,parquet,widget]"') < installer.index(
        '-e "$CPPMEGA_MLX_LM_CHECKOUT"'
    )
    assert '--no-deps \\\n  -e "$CPPMEGA_MLX_LM_CHECKOUT"' in installer
    assert 'rev-parse HEAD' in installer
    assert 'diff --cached --quiet' in installer
    assert "native optimizer extension unavailable" in installer
    for module in (
        "bailing_hybrid",
        "deepseek_v4",
        "gemma4_assistant",
        "mistral4",
        "nemotron_h",
        "turbo_cache",
    ):
        assert module in installer


def test_macos_e2e_jobs_use_run_scoped_ports() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "e2e-matrix.yml").read_text(
        encoding="utf-8"
    )

    assert workflow.count("Allocate isolated E2E ports") == 3
    assert workflow.count("GITHUB_RUN_ID % 1000") == 3
    assert workflow.count('echo "VBGUI_E2E_BACKEND_PORT=') == 3
    assert workflow.count('echo "VBGUI_E2E_FRONTEND_PORT=') == 3


def test_preset_matrix_is_really_sharded_and_bounded() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "e2e-matrix.yml").read_text(
        encoding="utf-8"
    )
    jobs = workflow.partition("\njobs:\n")[2]
    preset_job = next(
        match.group("body")
        for match in JOB_BLOCK.finditer(jobs)
        if match.group("name") == "preset-matrix"
    )

    assert "shard: [1, 2, 3, 4]" in preset_job
    assert "--fully-parallel" in preset_job
    assert "--global-timeout=720000" in preset_job
    assert "--shard=${{ matrix.shard }}/4" in preset_job
    assert "timeout-minutes: 20" in preset_job
    assert "timeout-minutes: 40" not in preset_job
    assert 'cache: "npm"' not in workflow
    assert "cache-dependency-path:" not in workflow


def test_core_self_hosted_jobs_use_the_shared_receipted_lane_runner() -> None:
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "ci-self-hosted.yml"
    ).read_text(encoding="utf-8")

    assert workflow.count("scripts/run_self_hosted_ci.py lane") == 2
    assert "--lane macos-mlx" in workflow
    assert "--lane linux-portable" in workflow
    assert "python_bin=/Volumes/external/sources/.venvs/cppmega.mlx/bin/python" in workflow
    assert "/Volumes/external/sources/cppmega.mlx/.venv/bin/python" not in workflow
    assert "--bootstrap-portable" in workflow
    assert workflow.count("if-no-files-found: error") == 2
    assert workflow.count("retention-days: 14") == 2


def test_build_backend_declares_mlx_imported_by_setup_py() -> None:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert "mlx==0.32.0" in config["build-system"]["requires"]
    assert "mlx==0.32.0" in config["project"]["dependencies"]


def test_dev_environment_declares_pandas_for_expert_sft_data() -> None:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = config["dependency-groups"]["dev"]

    assert any(
        requirement.startswith("pandas>=2.2")
        for requirement in requirements
    )
    assert any(requirement.startswith("pyarrow>=21") for requirement in requirements)
    assert "datasketch==1.10.0" in requirements
    assert "libclang==18.1.1" in requirements
    assert any(requirement.startswith("nanobind>=2.4") for requirement in requirements)


def test_mlx_lm_source_is_pinned_to_the_cppmega_integration_commit() -> None:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requirement = next(
        item
        for item in config["project"]["dependencies"]
        if item.startswith("mlx-lm @ ")
    )

    assert requirement == (
        "mlx-lm @ git+https://github.com/DatasunriseOU/mlx-lm.git"
        "@8618587943181787d33bac4468d3088e80202b3f"
    )
    assert "uv" not in config.get("tool", {})


def test_package_discovery_is_limited_to_owned_python_trees() -> None:
    setup = (REPO_ROOT / "setup.py").read_text(encoding="utf-8")

    assert "packages=_local_packages()" in setup
    assert "[tool.setuptools.packages.find]" not in (
        REPO_ROOT / "pyproject.toml"
    ).read_text(encoding="utf-8")


def test_path_c_extra_binds_exact_tilelang_stack_and_compatible_z3() -> None:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    path_c = config["project"]["optional-dependencies"]["path-c"]
    marker = (
        "(platform_system == 'Darwin' and platform_machine == 'arm64') or "
        "(platform_system == 'Linux' and platform_machine == 'x86_64')"
    )
    assert "cython>=3.0,<4" in path_c
    assert (
        "tilelang @ git+https://github.com/DatasunriseOU/tilelang.git"
        "@a760fe587995def0f3108ee204be453d87467c5d ; "
        f"{marker}"
    ) in path_c
    assert (
        "apache-tvm-ffi @ git+https://github.com/DatasunriseOU/tvm-ffi.git"
        "@e4353339293459e3e8a393afc1b6a6a869e75b13 ; "
        f"{marker}"
    ) in path_c
    assert "z3-solver>=4.15,<4.15.5" in config["project"]["dependencies"]

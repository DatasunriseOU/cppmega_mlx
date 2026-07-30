#!/usr/bin/env python3
"""Run and orchestrate cppmega.mlx CI on the repository's own machines."""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import json
import os
import platform
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


SCHEMA_VERSION = "cppmega-mlx.self-hosted-ci.v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = REPO_ROOT / "scripts" / "self_hosted_hosts.json"
PORTABLE_DEPENDENCIES = (
    "pytest",
    "numpy",
    "pyarrow",
    "tokenizers",
    "zstandard",
)
MACOS_TESTS = (
    "tests/test_ast_fim.py",
    "tests/test_audit_sidecar_parquet.py",
    "tests/test_atomic_identity_publication.py",
    "tests/test_case5_domain_ingestion.py",
    "tests/test_case6_sidecar_manifest_contract.py",
    "tests/test_clang_usr_identity.py",
    "tests/test_convert_megatron_dense500m_torchdist_to_mlx.py",
    "tests/test_cpp_jsonl_generation_compile_eval.py",
    "tests/test_data_package_imports.py",
    "tests/test_domain_graph_routes.py",
    "tests/test_domain_sidecar_parquet.py",
    "tests/test_eval_domain_routed_codegen.py",
    "tests/test_graph_recipe.py",
    "tests/test_graphql_pr_stream.py",
    "tests/test_gharchive_pr_store_loader.py",
    "tests/test_inference_generation.py",
    "tests/test_inference_repository_prompt_graph.py",
    "tests/test_materialize_megatron_dependency_provenance.py",
    "tests/test_megatron_indexed.py",
    "tests/test_objective_schedule.py",
    "tests/test_pack_enriched_rows.py",
    "tests/test_packer_edge_remap.py",
    "tests/test_route_packed_source_docs.py",
    "tests/test_production_objective_mixer.py",
    "tests/test_production_megatron_bundle.py",
    "tests/test_pr_completion_receipt.py",
    "tests/test_pr_export_batches.py",
    "tests/test_pr_ingest_graphql_stream.py",
    "tests/test_pr_store_wal.py",
    "tests/test_prompt_graph.py",
    "tests/test_prompt_graph_index.py",
    "tests/test_stage1_combined_graph_objective.py",
    "tests/test_stage1_graph_domain_production.py",
    "tests/test_train_eval_graph_routes.py",
    "tests/test_streaming_conveyor_progress.py",
    "tests/test_streaming_conveyor_revision.py",
    "tests/test_streaming_reindex_run_checked.py",
    "tests/test_tokenizer_contract.py",
    "tests/test_train_stage1_smoke.py",
    "tests/test_verified_sidecar_download_verification.py",
    "tests/test_verified_sidecar_manifest_selection.py",
    "tests/test_process_commits_fail_loud.py",
    "tests/test_repair_packed_document_boundaries.py",
    "tests/test_self_hosted_ci.py",
    "tests/test_workflow_runner_policy.py",
    "tests/test_build_repo_list.py",
    "tests/test_h4_concurrency.py",
)
LINUX_TESTS = (
    "tests/test_audit_sidecar_parquet.py",
    "tests/test_case6_sidecar_manifest_contract.py",
    "tests/test_data_package_imports.py",
    "tests/test_domain_sidecar_parquet.py",
    "tests/test_graphql_pr_stream.py",
    "tests/test_gharchive_pr_store_loader.py",
    "tests/test_h4_concurrency.py",
    "tests/test_packer_edge_remap.py",
    "tests/test_production_megatron_bundle.py",
    "tests/test_route_packed_source_docs.py",
    "tests/test_pr_completion_receipt.py",
    "tests/test_pr_export_batches.py",
    "tests/test_pr_ingest_graphql_stream.py",
    "tests/test_pr_store_wal.py",
    "tests/test_streaming_conveyor_progress.py",
    "tests/test_streaming_conveyor_revision.py",
    "tests/test_streaming_reindex_run_checked.py",
    "tests/test_tokenizer_contract.py",
    "tests/test_verified_sidecar_download_verification.py",
    "tests/test_verified_sidecar_manifest_selection.py",
    "tests/test_process_commits_fail_loud.py",
    "tests/test_repair_packed_document_boundaries.py",
    "tests/test_self_hosted_ci.py",
    "tests/test_workflow_runner_policy.py",
    "tests/test_build_repo_list.py",
)


class SelfHostedCIError(RuntimeError):
    """Raised for a fail-closed runner or orchestration error."""


@dataclasses.dataclass(frozen=True)
class LaneSpec:
    name: str
    system: str
    machines: tuple[str, ...]
    tests: tuple[str, ...]
    default_timeout_seconds: int
    pytest_timeout_seconds: int
    pytest_args: tuple[str, ...] = ()
    portable: bool = False


LANES = {
    "macos-mlx": LaneSpec(
        name="macos-mlx",
        system="Darwin",
        machines=("arm64",),
        tests=MACOS_TESTS,
        default_timeout_seconds=1_500,
        pytest_timeout_seconds=1_200,
    ),
    "linux-portable": LaneSpec(
        name="linux-portable",
        system="Linux",
        machines=("x86_64", "amd64"),
        tests=LINUX_TESTS,
        default_timeout_seconds=900,
        pytest_timeout_seconds=600,
        pytest_args=("--noconftest", "-m", "not mlx_runtime"),
        portable=True,
    ),
}


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _run_id() -> str:
    github_run = os.environ.get("GITHUB_RUN_ID")
    github_attempt = os.environ.get("GITHUB_RUN_ATTEMPT")
    if github_run:
        suffix = f"-{github_attempt}" if github_attempt else ""
        return f"github-{github_run}{suffix}"
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"direct-{stamp}-{uuid.uuid4().hex[:8]}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in value)
    cleaned = cleaned.strip("-")
    if not cleaned:
        raise SelfHostedCIError(f"identifier has no safe characters: {value!r}")
    return cleaned


def _command_text(command: Sequence[str]) -> str:
    return shlex.join(str(part) for part in command)


def _git_output(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ("git", "-C", str(repo_root), *args),
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _source_commit(repo_root: Path) -> str:
    supplied = os.environ.get("CPPMEGA_SOURCE_COMMIT")
    if supplied:
        return supplied
    return _git_output(repo_root, "rev-parse", "HEAD") or "unknown"


def _source_dirty(repo_root: Path) -> bool | None:
    if os.environ.get("CPPMEGA_SOURCE_COMMIT") and _git_output(
        repo_root, "rev-parse", "--verify", "HEAD"
    ) is None:
        # Direct orchestration initializes an index from an exact git archive.
        return False
    status = _git_output(repo_root, "status", "--porcelain")
    return None if status is None else bool(status)


def _host_snapshot() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "github_runner_name": os.environ.get("RUNNER_NAME"),
        "github_runner_os": os.environ.get("RUNNER_OS"),
        "github_runner_arch": os.environ.get("RUNNER_ARCH"),
    }


def _tail(path: Path, line_count: int = 80) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-line_count:])


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGKILL)
    else:
        process.kill()
    process.wait(timeout=10)


def run_step(
    *,
    name: str,
    command: Sequence[str],
    cwd: Path,
    log_path: Path,
    timeout_seconds: float,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run one step with a hard timeout and a dedicated combined log."""

    if timeout_seconds <= 0:
        raise SelfHostedCIError(f"step {name!r} has no time remaining")
    started_at = _utc_now()
    started = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[self-hosted-ci] start {name}: {_command_text(command)}", flush=True)
    timed_out = False
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            tuple(str(part) for part in command),
            cwd=cwd,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            exit_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process_group(process)
            exit_code = 124
    duration = round(time.monotonic() - started, 3)
    status = "timed_out" if timed_out else ("passed" if exit_code == 0 else "failed")
    print(
        f"[self-hosted-ci] {status} {name} in {duration:.3f}s "
        f"(log: {log_path})",
        flush=True,
    )
    if exit_code != 0:
        tail = _tail(log_path)
        if tail:
            print(f"[self-hosted-ci] {name} log tail:\n{tail}", file=sys.stderr)
    return {
        "name": name,
        "command": [str(part) for part in command],
        "status": status,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "duration_seconds": duration,
        "timeout_seconds": timeout_seconds,
        "log": log_path.name,
    }


def _resolve_executable(value: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.is_absolute() or "/" in value:
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise SelfHostedCIError(f"Python executable is unavailable: {candidate}")
        return str(candidate)
    resolved = shutil.which(value)
    if resolved is None:
        raise SelfHostedCIError(f"Python executable is unavailable on PATH: {value}")
    return resolved


def _validate_lane_platform(spec: LaneSpec) -> None:
    actual_system = platform.system()
    actual_machine = platform.machine().lower()
    allowed = {machine.lower() for machine in spec.machines}
    if actual_system != spec.system or actual_machine not in allowed:
        raise SelfHostedCIError(
            f"lane {spec.name!r} requires {spec.system}/{','.join(spec.machines)}; "
            f"host is {actual_system}/{platform.machine()}"
        )


def _source_check_command(repo_root: Path) -> tuple[str, ...]:
    inside = _git_output(repo_root, "rev-parse", "--is-inside-work-tree")
    if inside != "true":
        raise SelfHostedCIError(f"source tree has no Git index: {repo_root}")
    head = _git_output(repo_root, "rev-parse", "--verify", "HEAD")
    if head is None:
        return ("git", "diff", "--cached", "--check")
    return ("git", "diff", "--check")


def _source_bound_environment(
    repo_root: Path,
    *,
    base_environment: dict[str, str] | None = None,
) -> dict[str, str]:
    """Run lane subprocesses against this checkout, never an editable sibling."""

    environment = dict(os.environ if base_environment is None else base_environment)
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        environment.pop(key, None)
    environment.update(
        {
            "CPPMEGA_SOURCE_ROOT": str(repo_root),
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
            "PYTHONPATH": str(repo_root),
        }
    )
    return environment


def _source_import_probe_command(
    python: str,
    repo_root: Path,
) -> tuple[str, ...]:
    """Return a child-process probe for actual project import provenance."""

    probe = r'''
import importlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
project_modules = ("cppmega_mlx", "scripts")

def inside_root(value: Path) -> bool:
    try:
        value.relative_to(root)
    except ValueError:
        return False
    return True

if root not in {Path(entry or ".").resolve() for entry in sys.path}:
    raise SystemExit(f"reviewed root is absent from sys.path: {root}")

resolved = {}
for name in project_modules:
    module = importlib.import_module(name)
    candidates = []
    origin = getattr(module, "__file__", None)
    if origin and origin not in {"built-in", "frozen"}:
        candidates.append(Path(origin).resolve())
    candidates.extend(
        Path(entry).resolve()
        for entry in getattr(module, "__path__", ())
    )
    if not candidates:
        raise SystemExit(f"project module has no source location: {name}")
    foreign = [str(value) for value in candidates if not inside_root(value)]
    if foreign:
        raise SystemExit(
            f"{name} resolved outside reviewed root {root}: {foreign}"
        )
    resolved[name] = [str(value) for value in candidates]

print(json.dumps({"root": str(root), "imports": resolved}, sort_keys=True))
'''.strip()
    return (python, "-c", probe, str(repo_root))


def run_lane(args: argparse.Namespace) -> int:
    spec = LANES[args.lane]
    repo_root = Path(args.repo_root).resolve()
    receipt_dir = Path(args.receipt_dir).resolve()
    receipt_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or _run_id()
    started_at = _utc_now()
    started = time.monotonic()
    timeout_seconds = args.timeout_seconds or spec.default_timeout_seconds
    deadline = started + timeout_seconds
    steps: list[dict[str, Any]] = []
    status = "failed"
    error: str | None = None
    bootstrap_dir: Path | None = None
    effective_python = args.python

    def remaining(step_limit: int) -> float:
        return max(0.0, min(float(step_limit), deadline - time.monotonic()))

    try:
        if not repo_root.is_dir():
            raise SelfHostedCIError(f"repository root is unavailable: {repo_root}")
        _validate_lane_platform(spec)
        effective_python = _resolve_executable(args.python)
        lane_environment = _source_bound_environment(repo_root)
        if args.bootstrap_portable and not spec.portable:
            raise SelfHostedCIError("--bootstrap-portable is valid only for linux-portable")

        preflight = run_step(
            name="python-preflight",
            command=(
                effective_python,
                "-c",
                (
                    "import platform,sys; "
                    "assert sys.version_info >= (3,11), sys.version; "
                    "print(platform.platform()); print(sys.version)"
                ),
            ),
            cwd=repo_root,
            log_path=receipt_dir / "python-preflight.log",
            timeout_seconds=remaining(30),
            env=lane_environment,
        )
        steps.append(preflight)
        if preflight["exit_code"] != 0:
            status = preflight["status"]
            return 124 if preflight["timed_out"] else 1

        if args.bootstrap_portable:
            bootstrap_parent = Path(
                os.environ.get("RUNNER_TEMP", tempfile.gettempdir())
            ).resolve()
            bootstrap_parent.mkdir(parents=True, exist_ok=True)
            bootstrap_dir = Path(
                tempfile.mkdtemp(prefix="cppmega-mlx-portable-", dir=bootstrap_parent)
            )
            create_venv = run_step(
                name="create-portable-venv",
                command=(effective_python, "-m", "venv", str(bootstrap_dir)),
                cwd=repo_root,
                log_path=receipt_dir / "create-portable-venv.log",
                timeout_seconds=remaining(120),
                env=lane_environment,
            )
            steps.append(create_venv)
            if create_venv["exit_code"] != 0:
                status = create_venv["status"]
                return 124 if create_venv["timed_out"] else 1
            effective_python = str(bootstrap_dir / "bin" / "python")
            install = run_step(
                name="install-portable-dependencies",
                command=(
                    effective_python,
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    *PORTABLE_DEPENDENCIES,
                ),
                cwd=repo_root,
                log_path=receipt_dir / "install-portable-dependencies.log",
                timeout_seconds=remaining(300),
                env=lane_environment,
            )
            steps.append(install)
            if install["exit_code"] != 0:
                status = install["status"]
                return 124 if install["timed_out"] else 1

        source_probe = run_step(
            name="source-import-provenance",
            command=_source_import_probe_command(effective_python, repo_root),
            cwd=repo_root,
            log_path=receipt_dir / "source-import-provenance.log",
            timeout_seconds=remaining(30),
            env=lane_environment,
        )
        steps.append(source_probe)
        if source_probe["exit_code"] != 0:
            status = source_probe["status"]
            return 124 if source_probe["timed_out"] else 1

        pytest_command = [
            effective_python,
            "-m",
            "pytest",
            "-q",
            *spec.pytest_args,
            *spec.tests,
        ]
        pytest_step = run_step(
            name="pytest",
            command=pytest_command,
            cwd=repo_root,
            log_path=receipt_dir / "pytest.log",
            timeout_seconds=remaining(spec.pytest_timeout_seconds),
            env=lane_environment,
        )
        steps.append(pytest_step)
        if pytest_step["exit_code"] != 0:
            status = pytest_step["status"]
            return 124 if pytest_step["timed_out"] else 1

        if spec.portable:
            compile_step = run_step(
                name="compile-portable-sources",
                command=(
                    effective_python,
                    "-m",
                    "compileall",
                    "-q",
                    "scripts",
                    "tools/clang_indexer",
                ),
                cwd=repo_root,
                log_path=receipt_dir / "compile-portable-sources.log",
                timeout_seconds=remaining(120),
                env=lane_environment,
            )
            steps.append(compile_step)
            if compile_step["exit_code"] != 0:
                status = compile_step["status"]
                return 124 if compile_step["timed_out"] else 1

        source_check = run_step(
            name="source-whitespace-check",
            command=_source_check_command(repo_root),
            cwd=repo_root,
            log_path=receipt_dir / "source-whitespace-check.log",
            timeout_seconds=remaining(60),
            env=lane_environment,
        )
        steps.append(source_check)
        if source_check["exit_code"] != 0:
            status = source_check["status"]
            return 124 if source_check["timed_out"] else 1

        status = "passed"
        return 0
    except (OSError, SelfHostedCIError, subprocess.SubprocessError) as exc:
        error = str(exc)
        print(f"[self-hosted-ci] lane failed closed: {error}", file=sys.stderr)
        return 1
    finally:
        if bootstrap_dir is not None:
            shutil.rmtree(bootstrap_dir, ignore_errors=True)
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "kind": "lane",
            "run_id": run_id,
            "lane": spec.name,
            "status": status,
            "error": error,
            "started_at": started_at,
            "completed_at": _utc_now(),
            "duration_seconds": round(time.monotonic() - started, 3),
            "timeout_seconds": timeout_seconds,
            "source_commit": _source_commit(repo_root),
            "source_dirty": _source_dirty(repo_root),
            "repo_root": str(repo_root),
            "host": _host_snapshot(),
            "python": effective_python,
            "bootstrap_portable": bool(args.bootstrap_portable),
            "steps": steps,
        }
        _write_json(receipt_dir / "receipt.json", receipt)
        print(f"[self-hosted-ci] receipt: {receipt_dir / 'receipt.json'}", flush=True)


def _load_inventory(path: Path) -> dict[str, Any]:
    try:
        inventory = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SelfHostedCIError(f"cannot read inventory {path}: {exc}") from exc
    if inventory.get("schema_version") != 1:
        raise SelfHostedCIError("inventory schema_version must be 1")
    hosts = inventory.get("hosts")
    if not isinstance(hosts, list) or not hosts:
        raise SelfHostedCIError("inventory must contain a non-empty hosts list")
    forbidden_keys = {
        "password",
        "passwd",
        "token",
        "secret",
        "credential",
        "private_key",
    }

    def inspect_keys(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower() in forbidden_keys:
                    raise SelfHostedCIError(
                        f"credentials are forbidden in host inventory: {key}"
                    )
                inspect_keys(child)
        elif isinstance(value, list):
            for child in value:
                inspect_keys(child)

    inspect_keys(inventory)
    seen: set[str] = set()
    for host in hosts:
        if not isinstance(host, dict):
            raise SelfHostedCIError("every inventory host must be an object")
        host_id = host.get("id")
        if not isinstance(host_id, str) or _safe_name(host_id) != host_id:
            raise SelfHostedCIError(f"invalid host id: {host_id!r}")
        if host_id in seen:
            raise SelfHostedCIError(f"duplicate host id: {host_id}")
        seen.add(host_id)
        lane = host.get("lane")
        if lane is not None and lane not in LANES:
            raise SelfHostedCIError(f"host {host_id} has unknown lane: {lane!r}")
        if host.get("required") and lane not in LANES:
            raise SelfHostedCIError(f"required host {host_id} has no supported lane")
        if lane in LANES:
            timeout_seconds = host.get("timeout_seconds")
            if not isinstance(timeout_seconds, int) or timeout_seconds <= 0:
                raise SelfHostedCIError(
                    f"host {host_id} must have a positive integer timeout_seconds"
                )
        if host.get("transport") not in {"local", "ssh"}:
            raise SelfHostedCIError(f"host {host_id} has unsupported transport")
    return inventory


def _ssh_base(host: dict[str, Any], connect_timeout: int) -> list[str]:
    target = host.get("ssh_target")
    if not isinstance(target, str) or not target:
        raise SelfHostedCIError(f"host {host['id']} has no ssh_target")
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "PasswordAuthentication=no",
        "-o",
        "KbdInteractiveAuthentication=no",
        "-o",
        f"ConnectTimeout={connect_timeout}",
        "-o",
        "ConnectionAttempts=1",
        target,
    ]


def classify_ssh_failure(stderr: str) -> str:
    lowered = stderr.lower()
    if "host key verification failed" in lowered or "remote host identification" in lowered:
        return "host_key_verification_failed"
    if "permission denied" in lowered:
        return "ssh_authentication_failed"
    if "connection refused" in lowered:
        return "ssh_connection_refused"
    if "no route to host" in lowered or "network is unreachable" in lowered:
        return "network_unreachable"
    if "timed out" in lowered or "operation timed out" in lowered:
        return "ssh_timeout"
    return "ssh_probe_failed"


def _parse_probe_payload(stdout: str) -> dict[str, Any]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise SelfHostedCIError("host probe returned no JSON")
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise SelfHostedCIError("host probe returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise SelfHostedCIError("host probe JSON is not an object")
    return payload


def _probe_payload_command(python_executable: str) -> list[str]:
    code = (
        "import json,platform,sys; "
        "print(json.dumps({'hostname':platform.node(),"
        "'system':platform.system(),'release':platform.release(),"
        "'machine':platform.machine(),'python':platform.python_version(),"
        "'python_ok':sys.version_info >= (3,11)}))"
    )
    return [python_executable, "-c", code]


def _platform_matches(host: dict[str, Any], observed: dict[str, Any]) -> bool:
    expected_system = str(host.get("system", "")).lower()
    expected_machines = {
        str(value).lower() for value in host.get("machines", []) if value
    }
    actual_system = str(observed.get("system", "")).lower()
    actual_machine = str(observed.get("machine", "")).lower()
    return (
        actual_system == expected_system
        and actual_machine in expected_machines
        and observed.get("python_ok") is True
    )


def probe_host(host: dict[str, Any], *, connect_timeout: int) -> dict[str, Any]:
    started = time.monotonic()
    probe: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "host-probe",
        "host_id": host["id"],
        "address": host.get("address"),
        "transport": host["transport"],
        "lane": host.get("lane"),
        "required": bool(host.get("required")),
        "eligible": host.get("lane") in LANES,
        "github": host.get("github"),
        "started_at": _utc_now(),
        "available": False,
        "status": "unavailable",
        "detail": None,
        "observed": None,
    }
    try:
        if host["transport"] == "local":
            python_executable = _resolve_executable(str(host.get("python", sys.executable)))
            result = subprocess.run(
                _probe_payload_command(python_executable),
                check=False,
                capture_output=True,
                text=True,
                timeout=connect_timeout + 5,
            )
        else:
            address = str(host.get("address", ""))
            port = int(host.get("ssh_port", 22))
            with socket.create_connection((address, port), timeout=connect_timeout):
                pass
            remote_python = str(host.get("python", "python3"))
            remote_command = _command_text(_probe_payload_command(remote_python))
            result = subprocess.run(
                (*_ssh_base(host, connect_timeout), remote_command),
                check=False,
                capture_output=True,
                text=True,
                timeout=connect_timeout + 8,
            )
        if result.returncode != 0:
            detail = (
                classify_ssh_failure(result.stderr)
                if host["transport"] == "ssh"
                else "local_probe_failed"
            )
            probe["detail"] = detail
            probe["stderr_tail"] = "\n".join(result.stderr.splitlines()[-5:])
        else:
            observed = _parse_probe_payload(result.stdout)
            probe["observed"] = observed
            if not _platform_matches(host, observed):
                probe["detail"] = "platform_or_python_mismatch"
            elif host.get("lane") not in LANES:
                probe["available"] = True
                probe["status"] = "not_eligible"
                probe["detail"] = host.get("reason") or "no supported test lane"
            else:
                probe["available"] = True
                probe["status"] = "available"
                probe["detail"] = "probe_passed"
    except socket.timeout:
        probe["detail"] = "tcp_timeout"
    except ConnectionRefusedError:
        probe["detail"] = "ssh_connection_refused"
    except subprocess.TimeoutExpired:
        probe["detail"] = "probe_timeout"
    except (OSError, SelfHostedCIError, ValueError) as exc:
        probe["detail"] = f"probe_error: {exc}"
    probe["completed_at"] = _utc_now()
    probe["duration_seconds"] = round(time.monotonic() - started, 3)
    return probe


def _run_checked(
    command: Sequence[str],
    *,
    timeout: float,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    stdout: Any = subprocess.PIPE,
    stderr: Any = subprocess.PIPE,
    stdin: Any = None,
    text: bool = True,
) -> subprocess.CompletedProcess[Any]:
    result = subprocess.run(
        tuple(str(part) for part in command),
        cwd=cwd,
        env=env,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        text=text,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        stdout_value = result.stdout if isinstance(result.stdout, str) else ""
        stderr_value = result.stderr if isinstance(result.stderr, str) else ""
        raise SelfHostedCIError(
            f"command failed ({result.returncode}): {_command_text(command)}\n"
            f"{stdout_value[-2000:]}{stderr_value[-2000:]}"
        )
    return result


def _initialize_archive_index(staged_root: Path) -> None:
    _run_checked(("git", "init", "-q", str(staged_root)), timeout=30)
    _run_checked(("git", "-C", str(staged_root), "add", "-f", "-A"), timeout=120)
    _run_checked(
        (
            "git",
            "-C",
            str(staged_root),
            "-c",
            "user.name=cppmega-self-hosted-ci",
            "-c",
            "user.email=runner@cppmega.invalid",
            "commit",
            "--quiet",
            "--no-gpg-sign",
            "-m",
            "Direct orchestration source snapshot",
        ),
        timeout=120,
    )


def _extract_local_archive(repo_root: Path, commit: str, staged_root: Path) -> None:
    archive = subprocess.Popen(
        ("git", "-C", str(repo_root), "archive", "--format=tar", commit),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert archive.stdout is not None
    extract = subprocess.run(
        ("tar", "-xf", "-", "-C", str(staged_root)),
        stdin=archive.stdout,
        capture_output=True,
        timeout=180,
        check=False,
    )
    archive.stdout.close()
    archive_stderr = archive.communicate(timeout=30)[1]
    if archive.returncode != 0 or extract.returncode != 0:
        raise SelfHostedCIError(
            "cannot stage local source archive: "
            f"git={archive.returncode} tar={extract.returncode} "
            f"{archive_stderr.decode(errors='replace')[-1000:]} "
            f"{extract.stderr.decode(errors='replace')[-1000:]}"
        )
    _initialize_archive_index(staged_root)


def _lane_command(
    host: dict[str, Any],
    *,
    repo_root: str,
    receipt_dir: str,
    run_id: str,
) -> list[str]:
    command = [
        str(host.get("python", "python3")),
        "scripts/run_self_hosted_ci.py",
        "lane",
        "--lane",
        str(host["lane"]),
        "--repo-root",
        repo_root,
        "--receipt-dir",
        receipt_dir,
        "--run-id",
        run_id,
        "--timeout-seconds",
        str(int(host["timeout_seconds"])),
    ]
    if LANES[str(host["lane"])].portable:
        command.append("--bootstrap-portable")
    return command


def _fallback_host_receipt(
    host: dict[str, Any],
    *,
    run_id: str,
    status: str,
    error: str,
    commit: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "lane",
        "run_id": run_id,
        "lane": host.get("lane"),
        "status": status,
        "error": error,
        "source_commit": commit,
        "host": {"inventory_id": host["id"], "address": host.get("address")},
        "started_at": _utc_now(),
        "completed_at": _utc_now(),
        "steps": [],
    }


def _run_local_host(
    host: dict[str, Any],
    *,
    repo_root: Path,
    commit: str,
    output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    staged_root = Path(tempfile.mkdtemp(prefix="cppmega-mlx-ci-local-"))
    transport_log = output_dir / "transport.log"
    try:
        _extract_local_archive(repo_root, commit, staged_root)
        command = _lane_command(
            host,
            repo_root=str(staged_root),
            receipt_dir=str(output_dir),
            run_id=run_id,
        )
        env = dict(os.environ)
        env["CPPMEGA_SOURCE_COMMIT"] = commit
        with transport_log.open("w", encoding="utf-8") as log_handle:
            result = subprocess.run(
                command,
                cwd=staged_root,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=int(host["timeout_seconds"]) + 30,
                check=False,
            )
        receipt_path = output_dir / "receipt.json"
        if not receipt_path.is_file():
            raise SelfHostedCIError("local lane did not produce receipt.json")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["inventory_host_id"] = host["id"]
        receipt["transport_log"] = transport_log.name
        _write_json(receipt_path, receipt)
        if result.returncode != 0 and receipt.get("status") == "passed":
            raise SelfHostedCIError(
                f"local lane exited {result.returncode} but receipt says passed"
            )
        return receipt
    except subprocess.TimeoutExpired as exc:
        receipt = _fallback_host_receipt(
            host,
            run_id=run_id,
            status="timed_out",
            error=f"local orchestration timeout: {exc}",
            commit=commit,
        )
        _write_json(output_dir / "receipt.json", receipt)
        return receipt
    except (OSError, SelfHostedCIError, json.JSONDecodeError) as exc:
        receipt = _fallback_host_receipt(
            host,
            run_id=run_id,
            status="failed",
            error=str(exc),
            commit=commit,
        )
        _write_json(output_dir / "receipt.json", receipt)
        return receipt
    finally:
        shutil.rmtree(staged_root, ignore_errors=True)


def _stage_remote_archive(
    host: dict[str, Any],
    *,
    repo_root: Path,
    commit: str,
    connect_timeout: int,
) -> str:
    remote_script = """
set -euo pipefail
workdir="$(mktemp -d /tmp/cppmega-mlx-ci.XXXXXX)"
tar -xf - -C "$workdir"
git init -q "$workdir"
git -C "$workdir" add -f -A
git -C "$workdir" \
  -c user.name=cppmega-self-hosted-ci \
  -c user.email=runner@cppmega.invalid \
  commit --quiet --no-gpg-sign -m 'Direct orchestration source snapshot'
printf '%s\\n' "$workdir"
""".strip()
    archive = subprocess.Popen(
        ("git", "-C", str(repo_root), "archive", "--format=tar", commit),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert archive.stdout is not None
    result = subprocess.run(
        (*_ssh_base(host, connect_timeout), "bash -lc " + shlex.quote(remote_script)),
        stdin=archive.stdout,
        capture_output=True,
        timeout=180,
        check=False,
    )
    archive.stdout.close()
    archive_stderr = archive.communicate(timeout=30)[1]
    if archive.returncode != 0 or result.returncode != 0:
        raise SelfHostedCIError(
            "cannot stage remote source archive: "
            f"git={archive.returncode} ssh={result.returncode} "
            f"{archive_stderr.decode(errors='replace')[-1000:]} "
            f"{result.stderr.decode(errors='replace')[-1000:]}"
        )
    remote_root = result.stdout.decode(encoding="utf-8", errors="replace").strip()
    if not remote_root.startswith("/tmp/cppmega-mlx-ci.") or "\n" in remote_root:
        raise SelfHostedCIError(f"remote staging returned unsafe path: {remote_root!r}")
    return remote_root


def _extract_receipt_archive(archive_path: Path, output_dir: Path) -> None:
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise SelfHostedCIError(f"unsafe receipt archive path: {member.name}")
            if member.issym() or member.islnk():
                raise SelfHostedCIError(f"receipt archive contains link: {member.name}")
            if not member.isfile() and not member.isdir():
                raise SelfHostedCIError(
                    f"receipt archive contains special file: {member.name}"
                )
        archive.extractall(output_dir, members=members)


def _run_remote_host(
    host: dict[str, Any],
    *,
    repo_root: Path,
    commit: str,
    output_dir: Path,
    run_id: str,
    connect_timeout: int,
) -> dict[str, Any]:
    remote_root: str | None = None
    transport_log = output_dir / "transport.log"
    try:
        remote_root = _stage_remote_archive(
            host,
            repo_root=repo_root,
            commit=commit,
            connect_timeout=connect_timeout,
        )
        command = _lane_command(
            host,
            repo_root=remote_root,
            receipt_dir=f"{remote_root}/receipts",
            run_id=run_id,
        )
        remote_script = (
            "set -euo pipefail; "
            f"cd {shlex.quote(remote_root)}; "
            f"export CPPMEGA_SOURCE_COMMIT={shlex.quote(commit)}; "
            f"exec {_command_text(command)}"
        )
        with transport_log.open("w", encoding="utf-8") as log_handle:
            result = subprocess.run(
                (
                    *_ssh_base(host, connect_timeout),
                    "bash -lc " + shlex.quote(remote_script),
                ),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                timeout=int(host["timeout_seconds"]) + 30,
                check=False,
            )
        collect_script = (
            f"set -euo pipefail; test -f {shlex.quote(remote_root)}/receipts/receipt.json; "
            f"tar -C {shlex.quote(remote_root)}/receipts -czf - ."
        )
        archive_path = output_dir / "receipts.tar.gz"
        with archive_path.open("wb") as archive_handle:
            _run_checked(
                (
                    *_ssh_base(host, connect_timeout),
                    "bash -lc " + shlex.quote(collect_script),
                ),
                timeout=60,
                stdout=archive_handle,
                stderr=subprocess.PIPE,
                text=False,
            )
        _extract_receipt_archive(archive_path, output_dir)
        archive_path.unlink()
        receipt_path = output_dir / "receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["inventory_host_id"] = host["id"]
        receipt["transport_log"] = transport_log.name
        _write_json(receipt_path, receipt)
        if result.returncode != 0 and receipt.get("status") == "passed":
            raise SelfHostedCIError(
                f"remote lane exited {result.returncode} but receipt says passed"
            )
        return receipt
    except subprocess.TimeoutExpired as exc:
        receipt = _fallback_host_receipt(
            host,
            run_id=run_id,
            status="timed_out",
            error=f"remote orchestration timeout: {exc}",
            commit=commit,
        )
        _write_json(output_dir / "receipt.json", receipt)
        return receipt
    except (OSError, SelfHostedCIError, json.JSONDecodeError, tarfile.TarError) as exc:
        receipt = _fallback_host_receipt(
            host,
            run_id=run_id,
            status="failed",
            error=str(exc),
            commit=commit,
        )
        _write_json(output_dir / "receipt.json", receipt)
        return receipt
    finally:
        if remote_root is not None:
            cleanup = f"rm -rf -- {shlex.quote(remote_root)}"
            subprocess.run(
                (*_ssh_base(host, connect_timeout), cleanup),
                check=False,
                capture_output=True,
                timeout=connect_timeout + 10,
            )


def _select_hosts(
    hosts: list[dict[str, Any]],
    *,
    host_ids: set[str],
    lanes: set[str],
) -> list[dict[str, Any]]:
    known_ids = {str(host["id"]) for host in hosts}
    unknown = host_ids - known_ids
    if unknown:
        raise SelfHostedCIError(f"unknown host ids: {sorted(unknown)}")
    selected = []
    for host in hosts:
        if host_ids and host["id"] not in host_ids:
            continue
        if lanes and host.get("lane") not in lanes:
            continue
        selected.append(host)
    if not selected:
        raise SelfHostedCIError("host selection is empty")
    return selected


def _print_probe_summary(probes: Iterable[dict[str, Any]]) -> None:
    for probe in probes:
        github = probe.get("github") or {}
        runner = github.get("runner_name", "-")
        print(
            "[self-hosted-ci] "
            f"host={probe['host_id']} lane={probe.get('lane') or '-'} "
            f"github_runner={runner} status={probe['status']} "
            f"detail={probe.get('detail') or '-'}"
        )


def orchestrate(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    inventory_path = Path(args.inventory).resolve()
    inventory = _load_inventory(inventory_path)
    selected = _select_hosts(
        inventory["hosts"],
        host_ids=set(args.host or ()),
        lanes=set(args.lane or ()),
    )
    commit = _git_output(repo_root, "rev-parse", args.ref)
    if commit is None and args.ref == "HEAD":
        commit = os.environ.get("CPPMEGA_SOURCE_COMMIT")
    if commit is None:
        raise SelfHostedCIError(f"cannot resolve Git ref {args.ref!r}")
    run_id = args.run_id or _run_id()
    receipt_base = (
        Path(args.receipt_dir).resolve()
        if args.receipt_dir
        else Path(tempfile.gettempdir()) / "cppmega-mlx-self-hosted"
    )
    receipt_root = receipt_base / run_id
    receipt_root.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now()
    started = time.monotonic()

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(selected)) as pool:
        future_hosts = {
            pool.submit(probe_host, host, connect_timeout=args.connect_timeout): host
            for host in selected
        }
        probes = [future.result() for future in future_hosts]
    probes.sort(key=lambda item: item["host_id"])
    for probe in probes:
        host_dir = receipt_root / _safe_name(str(probe["host_id"]))
        _write_json(host_dir / "probe.json", probe)
    _print_probe_summary(probes)

    host_by_id = {str(host["id"]): host for host in selected}
    required_unavailable = [
        probe
        for probe in probes
        if probe["required"] and probe["eligible"] and not probe["available"]
    ]
    runnable = [
        probe for probe in probes if probe["eligible"] and probe["available"]
    ]
    status = "dry_run" if args.dry_run or args.probe_only else "pending"
    results: list[dict[str, Any]] = []
    exit_code = 0
    if required_unavailable:
        status = "blocked_unavailable_hosts"
        exit_code = 2
    elif args.dry_run or args.probe_only:
        status = "dry_run_passed"
    elif not runnable:
        status = "blocked_no_runnable_hosts"
        exit_code = 2
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(runnable)) as pool:
            future_runs: dict[concurrent.futures.Future[dict[str, Any]], str] = {}
            for probe in runnable:
                host = host_by_id[probe["host_id"]]
                host_output = receipt_root / _safe_name(str(host["id"]))
                runner: Callable[..., dict[str, Any]] = (
                    _run_local_host
                    if host["transport"] == "local"
                    else _run_remote_host
                )
                kwargs: dict[str, Any] = {
                    "repo_root": repo_root,
                    "commit": commit,
                    "output_dir": host_output,
                    "run_id": run_id,
                }
                if host["transport"] == "ssh":
                    kwargs["connect_timeout"] = args.connect_timeout
                future_runs[pool.submit(runner, host, **kwargs)] = host["id"]
            for future, host_id in future_runs.items():
                result = future.result()
                result["inventory_host_id"] = host_id
                results.append(result)
        results.sort(key=lambda item: item["inventory_host_id"])
        failed = [result for result in results if result.get("status") != "passed"]
        status = "failed" if failed else "passed"
        exit_code = 1 if failed else 0

    orchestration_receipt = {
        "schema_version": SCHEMA_VERSION,
        "kind": "orchestration",
        "run_id": run_id,
        "status": status,
        "dry_run": bool(args.dry_run or args.probe_only),
        "started_at": started_at,
        "completed_at": _utc_now(),
        "duration_seconds": round(time.monotonic() - started, 3),
        "repository": inventory.get("repository"),
        "inventory": str(inventory_path),
        "source_commit": commit,
        "source_dirty": _source_dirty(repo_root),
        "probes": probes,
        "results": results,
        "required_unavailable_hosts": [
            probe["host_id"] for probe in required_unavailable
        ],
    }
    _write_json(receipt_root / "orchestration.json", orchestration_receipt)
    print(f"[self-hosted-ci] orchestration receipt: {receipt_root / 'orchestration.json'}")
    return exit_code


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    lane = subparsers.add_parser("lane", help="run one test lane on this host")
    lane.add_argument("--lane", required=True, choices=sorted(LANES))
    lane.add_argument("--repo-root", default=str(REPO_ROOT))
    lane.add_argument("--receipt-dir", required=True)
    lane.add_argument("--python", default=sys.executable)
    lane.add_argument("--timeout-seconds", type=int)
    lane.add_argument("--run-id")
    lane.add_argument("--bootstrap-portable", action="store_true")
    lane.set_defaults(handler=run_lane)

    orchestrator = subparsers.add_parser(
        "orchestrate",
        help="probe and dispatch the matrix directly, without GitHub Actions",
    )
    orchestrator.add_argument("--inventory", default=str(DEFAULT_INVENTORY))
    orchestrator.add_argument("--repo-root", default=str(REPO_ROOT))
    orchestrator.add_argument("--receipt-dir")
    orchestrator.add_argument("--ref", default="HEAD")
    orchestrator.add_argument("--run-id")
    orchestrator.add_argument("--host", action="append")
    orchestrator.add_argument("--lane", action="append", choices=sorted(LANES))
    orchestrator.add_argument("--connect-timeout", type=int, default=5)
    orchestrator.add_argument("--dry-run", action="store_true")
    orchestrator.add_argument("--probe-only", action="store_true")
    orchestrator.set_defaults(handler=orchestrate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, SelfHostedCIError, subprocess.SubprocessError) as exc:
        print(f"[self-hosted-ci] fatal: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

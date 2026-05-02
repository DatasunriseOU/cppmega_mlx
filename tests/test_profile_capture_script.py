from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "profile_capture.py"

SCRIPT_SPEC = importlib.util.spec_from_file_location("profile_capture", SCRIPT)
assert SCRIPT_SPEC is not None
assert SCRIPT_SPEC.loader is not None
profile_capture = importlib.util.module_from_spec(SCRIPT_SPEC)
sys.modules[SCRIPT_SPEC.name] = profile_capture
SCRIPT_SPEC.loader.exec_module(profile_capture)


def run_profile_capture(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def test_help_lists_capture_flags() -> None:
    result = run_profile_capture("--help")

    assert result.returncode == 0
    assert "--trace-path" in result.stdout
    assert "--dry-run" in result.stdout
    assert "--timeout-s" in result.stdout


def test_dry_run_receipt_has_wave14_guardrails(tmp_path: Path) -> None:
    trace_path = tmp_path / "unit.gputrace"
    result = run_profile_capture(
        "--dry-run",
        "--trace-path",
        str(trace_path),
        "--",
        sys.executable,
        "-c",
        "print('not run')",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["kind"] == "cppmega_mlx_metal_capture_receipt"
    assert payload["schema_version"] == 1
    assert payload["status"] == "dry_run"
    assert payload["capture_requested"] is True
    assert payload["capture_started"] is False
    assert payload["capture_stopped"] is False
    assert payload["capture_api_available"] is None
    assert payload["local_only"] is True
    assert payload["trainable_metal_kernel_claim"] is False
    assert payload["custom_kernel_adopted"] is False
    assert payload["trace_path"] == str(trace_path)
    assert payload["command_status"] == "not_run"
    assert payload["command"]["argv"] == [
        sys.executable,
        "-c",
        "print('not run')",
    ]
    assert payload["policy"]["profile_gate_doc"] == "docs/profile_kernel_gate.md"
    assert payload["policy"]["metal_kernel_policy_doc"] == "docs/metal_kernel_policy.md"
    assert payload["policy"]["trainable_metal_kernel_claim"] is False


def test_missing_capture_api_fails_closed_without_running_command(tmp_path: Path) -> None:
    def forbidden_runner(
        argv: list[str],
        timeout_s: float,
    ) -> Any:
        raise AssertionError(f"command should not run: {argv!r} {timeout_s!r}")

    exit_code, payload = profile_capture.run_profile_capture(
        [
            "--trace-path",
            str(tmp_path / "missing.gputrace"),
            "--",
            sys.executable,
            "-c",
            "print('blocked')",
        ],
        metal_module=SimpleNamespace(),
        command_runner=forbidden_runner,
    )

    assert exit_code == 2
    assert payload["status"] == "error"
    assert payload["capture_api_available"] is False
    assert payload["capture_started"] is False
    assert payload["capture_stopped"] is False
    assert payload["command_status"] == "not_run"
    assert "start_capture unavailable" in payload["error"]
    assert "stop_capture unavailable" in payload["error"]


def test_capture_stops_after_successful_fake_command(tmp_path: Path) -> None:
    calls: list[tuple[str, str | None]] = []

    def start_capture(path: str) -> None:
        calls.append(("start", path))

    def stop_capture() -> None:
        calls.append(("stop", None))

    def runner(argv: list[str], timeout_s: float) -> Any:
        return profile_capture.CommandResult(
            argv=tuple(argv),
            status="ok",
            returncode=0,
            elapsed_s=0.01,
            stdout_tail="ok\n",
        )

    trace_path = tmp_path / "success.gputrace"
    exit_code, payload = profile_capture.run_profile_capture(
        ["--trace-path", str(trace_path), "--", "fake-bench", "--json"],
        metal_module=SimpleNamespace(
            start_capture=start_capture,
            stop_capture=stop_capture,
        ),
        command_runner=runner,
    )

    assert exit_code == 0
    assert calls == [("start", str(trace_path)), ("stop", None)]
    assert payload["status"] == "ok"
    assert payload["capture_started"] is True
    assert payload["capture_stopped"] is True
    assert payload["capture_api_available"] is True
    assert payload["command_status"] == "ok"
    assert payload["command"]["argv"] == ["fake-bench", "--json"]
    assert payload["local_only"] is True
    assert payload["trainable_metal_kernel_claim"] is False


def test_capture_stops_after_failed_fake_command(tmp_path: Path) -> None:
    calls: list[str] = []

    def runner(argv: list[str], timeout_s: float) -> Any:
        return profile_capture.CommandResult(
            argv=tuple(argv),
            status="failed",
            returncode=7,
            elapsed_s=0.01,
            stderr_tail="boom\n",
        )

    exit_code, payload = profile_capture.run_profile_capture(
        ["--trace-path", str(tmp_path / "failed.gputrace"), "--", "fake-bench"],
        metal_module=SimpleNamespace(
            start_capture=lambda path: calls.append(f"start:{path}"),
            stop_capture=lambda: calls.append("stop"),
        ),
        command_runner=runner,
    )

    assert exit_code == 7
    assert calls == [f"start:{tmp_path / 'failed.gputrace'}", "stop"]
    assert payload["status"] == "command_failed"
    assert payload["capture_started"] is True
    assert payload["capture_stopped"] is True
    assert payload["command_status"] == "failed"
    assert payload["command"]["returncode"] == 7


def test_stop_capture_failure_fails_closed_after_command(tmp_path: Path) -> None:
    def stop_capture() -> None:
        raise RuntimeError("xcode capture finalization failed")

    def runner(argv: list[str], timeout_s: float) -> Any:
        return profile_capture.CommandResult(
            argv=tuple(argv),
            status="ok",
            returncode=0,
            elapsed_s=0.01,
        )

    exit_code, payload = profile_capture.run_profile_capture(
        ["--trace-path", str(tmp_path / "stop-fail.gputrace"), "--", "fake-bench"],
        metal_module=SimpleNamespace(
            start_capture=lambda path: None,
            stop_capture=stop_capture,
        ),
        command_runner=runner,
    )

    assert exit_code == 2
    assert payload["status"] == "error"
    assert payload["capture_started"] is True
    assert payload["capture_stopped"] is False
    assert payload["command_status"] == "ok"
    assert "stop_capture failed" in payload["error"]


def test_invalid_trace_suffix_fails_before_capture() -> None:
    exit_code, payload = profile_capture.run_profile_capture(
        ["--trace-path", "/tmp/not-a-trace.json", "--", "fake-bench"],
        metal_module=SimpleNamespace(
            start_capture=lambda path: (_ for _ in ()).throw(AssertionError(path)),
            stop_capture=lambda: None,
        ),
    )

    assert exit_code == 2
    assert payload["status"] == "error"
    assert payload["capture_started"] is False
    assert payload["command_status"] == "not_run"
    assert payload["error"] == "trace-path must end with .gputrace"


def test_existing_trace_path_fails_before_capture(tmp_path: Path) -> None:
    trace_path = tmp_path / "exists.gputrace"
    trace_path.write_text("old trace placeholder")

    exit_code, payload = profile_capture.run_profile_capture(
        ["--trace-path", str(trace_path), "--", "fake-bench"],
        metal_module=SimpleNamespace(
            start_capture=lambda path: (_ for _ in ()).throw(AssertionError(path)),
            stop_capture=lambda: None,
        ),
    )

    assert exit_code == 2
    assert payload["status"] == "error"
    assert payload["capture_started"] is False
    assert payload["command_status"] == "not_run"
    assert payload["error"] == "trace-path already exists"

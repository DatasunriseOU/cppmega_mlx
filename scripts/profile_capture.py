#!/usr/bin/env python3
"""Fail-closed MLX Metal capture wrapper for local profiling receipts."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Sequence, cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RECEIPT_SCHEMA_VERSION = 1
DEFAULT_TIMEOUT_S = 60.0
TEXT_TAIL_CHARS = 4000
OFFICIAL_CAPTURE_API = (
    "mlx.core.metal.start_capture(path: str) and "
    "mlx.core.metal.stop_capture()"
)
PROFILE_GATE_DOC = "docs/profile_kernel_gate.md"
METAL_POLICY_DOC = "docs/metal_kernel_policy.md"

JsonScalar = bool | int | float | str | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]
CommandRunner = Callable[[Sequence[str], float], "CommandResult"]


@dataclass(frozen=True)
class CaptureApi:
    start_capture: Callable[[str], None]
    stop_capture: Callable[[], None]


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    status: str
    returncode: int | None
    elapsed_s: float | None
    stdout_tail: str = ""
    stderr_tail: str = ""
    error: str | None = None

    def to_dict(self) -> JsonObject:
        return {
            "argv": list(self.argv),
            "display": shlex.join(self.argv),
            "status": self.status,
            "returncode": self.returncode,
            "elapsed_s": self.elapsed_s,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "error": self.error,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a short local command under MLX Metal capture and emit a "
            "fail-closed JSON receipt."
        ),
    )
    parser.add_argument(
        "--trace-path",
        type=Path,
        default=None,
        help="Output .gputrace path. Defaults to a unique /tmp path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Emit the planned capture receipt without importing or calling MLX capture APIs.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help="Command timeout in seconds.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help=(
            "Command to run after '--'. If omitted, a tiny bench_tiny.py smoke "
            "command is used."
        ),
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def default_trace_path() -> Path:
    return Path("/tmp") / f"cppmega_mlx_profile_capture_{time.time_ns()}.gputrace"


def default_smoke_command() -> tuple[str, ...]:
    return (
        sys.executable,
        str(ROOT / "scripts" / "bench_tiny.py"),
        "--json",
        "--no-compile",
        "--batch-size",
        "1",
        "--seq-len",
        "4",
        "--vocab-size",
        "32",
        "--d-model",
        "8",
        "--n-heads",
        "1",
        "--n-layers",
        "1",
        "--mlp-dim",
        "16",
        "--dtype",
        "float32",
        "--steps",
        "1",
        "--warmup-steps",
        "0",
    )


def clean_command(raw_command: Sequence[str]) -> tuple[str, ...]:
    command = tuple(raw_command)
    if command and command[0] == "--":
        command = command[1:]
    return command or default_smoke_command()


def text_tail(value: str | bytes | None, *, limit: int = TEXT_TAIL_CHARS) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    return value[-limit:]


def resolve_capture_api(metal_module: Any | None = None) -> tuple[CaptureApi | None, str | None]:
    if metal_module is None:
        try:
            mx = import_module("mlx.core")
        except Exception as exc:  # pragma: no cover - version/env dependent.
            return None, f"failed to import mlx.core: {exc}"
        metal_module = getattr(mx, "metal", None)

    if metal_module is None:
        return None, "mlx.core.metal unavailable"

    start_capture_obj = getattr(metal_module, "start_capture", None)
    stop_capture_obj = getattr(metal_module, "stop_capture", None)
    missing: list[str] = []
    if not callable(start_capture_obj):
        missing.append("mlx.core.metal.start_capture unavailable")
    if not callable(stop_capture_obj):
        missing.append("mlx.core.metal.stop_capture unavailable")
    if missing:
        return None, "; ".join(missing)
    start_capture = cast(Callable[[str], None], start_capture_obj)
    stop_capture = cast(Callable[[], None], stop_capture_obj)
    return CaptureApi(start_capture=start_capture, stop_capture=stop_capture), None


def run_subprocess_command(argv: Sequence[str], timeout_s: float) -> CommandResult:
    started = time.monotonic()
    command = tuple(argv)
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            argv=command,
            status="timeout",
            returncode=None,
            elapsed_s=time.monotonic() - started,
            stdout_tail=text_tail(exc.stdout),
            stderr_tail=text_tail(exc.stderr),
            error=f"command timed out after {timeout_s:.3f}s",
        )
    except OSError as exc:
        return CommandResult(
            argv=command,
            status="error",
            returncode=None,
            elapsed_s=time.monotonic() - started,
            error=str(exc),
        )

    status = "ok" if result.returncode == 0 else "failed"
    return CommandResult(
        argv=command,
        status=status,
        returncode=result.returncode,
        elapsed_s=time.monotonic() - started,
        stdout_tail=text_tail(result.stdout),
        stderr_tail=text_tail(result.stderr),
    )


def base_receipt(
    *,
    trace_path: Path,
    command: Sequence[str],
    dry_run: bool,
    timeout_s: float,
) -> JsonObject:
    command_result = CommandResult(
        argv=tuple(command),
        status="not_run",
        returncode=None,
        elapsed_s=None,
    )
    return {
        "kind": "cppmega_mlx_metal_capture_receipt",
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "pending",
        "error": None,
        "dry_run": dry_run,
        "capture_requested": True,
        "capture_started": False,
        "capture_stopped": False,
        "capture_api_available": None,
        "capture_api": OFFICIAL_CAPTURE_API,
        "trace_path": str(trace_path),
        "timeout_s": timeout_s,
        "local_only": True,
        "trainable_metal_kernel_claim": False,
        "custom_kernel_adopted": False,
        "command_status": command_result.status,
        "command": command_result.to_dict(),
        "policy": {
            "profile_gate_doc": PROFILE_GATE_DOC,
            "metal_kernel_policy_doc": METAL_POLICY_DOC,
            "local_only": True,
            "trainable_metal_kernel_claim": False,
            "custom_kernel_adopted": False,
        },
    }


def set_command_receipt(receipt: JsonObject, result: CommandResult) -> None:
    receipt["command_status"] = result.status
    receipt["command"] = result.to_dict()


def fail_receipt(
    receipt: JsonObject,
    *,
    error: str,
    status: str = "error",
    command_result: CommandResult | None = None,
) -> tuple[int, JsonObject]:
    receipt["status"] = status
    receipt["error"] = error
    if command_result is not None:
        set_command_receipt(receipt, command_result)
    return 2, receipt


def exit_code_for_command(result: CommandResult) -> int:
    if result.status == "failed":
        return result.returncode if result.returncode is not None else 1
    if result.status == "timeout":
        return 124
    if result.status == "error":
        return 2
    return 0


def run_profile_capture(
    argv: Sequence[str] | None = None,
    *,
    metal_module: Any | None = None,
    command_runner: CommandRunner = run_subprocess_command,
) -> tuple[int, JsonObject]:
    args = parse_args(argv)
    trace_path = args.trace_path or default_trace_path()
    command = clean_command(args.command)
    timeout_s = float(args.timeout_s)
    receipt = base_receipt(
        trace_path=trace_path,
        command=command,
        dry_run=bool(args.dry_run),
        timeout_s=timeout_s,
    )

    if timeout_s <= 0:
        return fail_receipt(receipt, error="timeout-s must be positive")
    if trace_path.suffix != ".gputrace":
        return fail_receipt(receipt, error="trace-path must end with .gputrace")
    if args.dry_run:
        receipt["status"] = "dry_run"
        receipt["capture_api_available"] = None
        return 0, receipt

    api, api_error = resolve_capture_api(metal_module)
    if api is None:
        receipt["capture_api_available"] = False
        return fail_receipt(
            receipt,
            error=f"MLX Metal capture API unavailable: {api_error}",
        )
    receipt["capture_api_available"] = True

    if trace_path.exists():
        return fail_receipt(receipt, error="trace-path already exists")
    if not trace_path.parent.exists():
        return fail_receipt(receipt, error="trace-path parent directory does not exist")

    try:
        api.start_capture(str(trace_path))
    except Exception as exc:  # pragma: no cover - backend/Xcode dependent.
        return fail_receipt(
            receipt,
            error=f"MLX Metal start_capture failed: {exc}",
        )
    receipt["capture_started"] = True

    command_result: CommandResult
    stop_error: str | None = None
    try:
        try:
            command_result = command_runner(command, timeout_s)
        except Exception as exc:  # pragma: no cover - defensive injection guard.
            command_result = CommandResult(
                argv=command,
                status="error",
                returncode=None,
                elapsed_s=None,
                error=f"command runner failed: {exc}",
            )
    finally:
        try:
            api.stop_capture()
            receipt["capture_stopped"] = True
        except Exception as exc:  # pragma: no cover - backend/Xcode dependent.
            stop_error = str(exc)

    set_command_receipt(receipt, command_result)
    if stop_error is not None:
        return fail_receipt(
            receipt,
            error=f"MLX Metal stop_capture failed: {stop_error}",
            command_result=command_result,
        )

    if command_result.status == "ok":
        receipt["status"] = "ok"
        return 0, receipt

    receipt["status"] = f"command_{command_result.status}"
    receipt["error"] = command_result.error or f"command {command_result.status}"
    return exit_code_for_command(command_result), receipt


def main(argv: Sequence[str] | None = None) -> int:
    exit_code, receipt = run_profile_capture(argv)
    print(json.dumps(receipt, sort_keys=True))
    error = receipt.get("error")
    if exit_code != 0 and isinstance(error, str):
        print(error, file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

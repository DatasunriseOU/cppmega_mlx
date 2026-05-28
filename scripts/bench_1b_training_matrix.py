#!/usr/bin/env python3
"""Run the final local 1B-class Path B/C training matrix.

The actual training cell is delegated to ``scripts/m04_train_step.py`` so this
harness does not fork a second training implementation. This file owns the
matrix dimensions, fresh subprocess isolation, cold/warm TileLang cache setup,
and Markdown/CSV receipt shape required by ``ml_optim_plan.md`` P12.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TARGET_PARQUET = (
    ROOT
    / "data"
    / "parquet_samples"
    / "gb10"
    / "clang_semantic_4k_v10"
    / "val_00000.parquet"
)
DEFAULT_OUT = Path("/tmp/cppmega_1b_path_matrix.md")
DEFAULT_CSV = Path("/tmp/cppmega_1b_path_matrix.csv")
DEFAULT_JSON = Path("/tmp/cppmega_1b_path_matrix.json")
DEFAULT_WORK_DIR = Path("/tmp/cppmega_1b_path_matrix_cells")
DEFAULT_CACHE_DIR = Path("/tmp/cppmega_1b_path_matrix_tilelang_cache")

DTYPE_CHOICES = ("bf16", "fp8")
OPTIMIZER_CHOICES = (
    "adamw",
    "adam8bit",
    "lion",
    "lion8bit",
    "muon",
    "muon_adamw",
    "muon_int8",
)
PATH_CHOICES = ("path_b", "path_c_cold", "path_c_warm")
STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_UNSUPPORTED = "unsupported"
STATUS_NOT_APPLICABLE = "not_applicable"
STATUS_PLANNED = "planned"
MAMBA3_PATH_C_BWD_ENV = "CPPMEGA_MAMBA3_PATH_C_BWD"
SPARSE_MLA_FP8_ROUTE_ENV = "CPPMEGA_SPARSE_MLA_FP8_ROUTE"
SPARSE_MLA_FP8_BWD_ENV = "CPPMEGA_SPARSE_MLA_FP8_BWD"
FAILURE_REASON_KEYS = (
    "status_reason",
    "failure_reason",
    "reason",
    "error",
    "message",
    "title",
)


@dataclass(frozen=True)
class MatrixCell:
    dtype: str
    optimizer: str
    path: str
    dtype_arg: str
    cli_optimizer: str
    supported: bool
    unsupported_reason: str | None
    output_json: Path
    command: tuple[str, ...]
    env: dict[str, str]
    cache_mode: str
    cache_dir: Path | None

    @property
    def case_id(self) -> str:
        return f"{self.dtype}_{self.optimizer}_{self.path}"


@dataclass
class CellResult:
    case_id: str
    dtype: str
    optimizer: str
    path: str
    status: str
    command: str
    cppmega_sha: str
    tilelang_sha: str
    mlx_sha: str
    mlx_version: str
    cache_state: dict[str, Any]
    cli_optimizer: str | None = None
    optimizer_key: str | None = None
    optimizer_name: str | None = None
    optimizer_class: str | None = None
    optimizer_source: str | None = None
    steps_completed: int | None = None
    first_step_sec: float | None = None
    median_step_sec: float | None = None
    tok_sec: float | None = None
    step_sec: float | None = None
    compile_time_s: float | None = None
    peak_memory_bytes: int | None = None
    peak_memory_gb: float | None = None
    active_memory_gb: float | None = None
    cache_memory_gb: float | None = None
    profiling_trace_path: str | None = None
    profiling_trace_captured: bool = False
    profiling_capture_receipt_path: str | None = None
    profiling_capture_status: str | None = None
    selected_schedule: dict[str, Any] = field(default_factory=dict)
    proof_result: dict[str, Any] = field(default_factory=dict)
    pass_fail_reason: str | None = None
    receipt_path: str | None = None
    returncode: int | None = None
    duration_s: float | None = None

    def to_row(self, *, max_reason_chars: int | None = None) -> dict[str, Any]:
        reason = self.pass_fail_reason
        if max_reason_chars is not None:
            reason = _short_text(reason, max_reason_chars)
        return {
            "case_id": self.case_id,
            "dtype": self.dtype,
            "optimizer": self.optimizer,
            "path": self.path,
            "status": self.status,
            "cli_optimizer": self.cli_optimizer,
            "optimizer_key": self.optimizer_key,
            "optimizer_name": self.optimizer_name,
            "optimizer_class": self.optimizer_class,
            "optimizer_source": self.optimizer_source,
            "steps_completed": self.steps_completed,
            "first_step_sec": self.first_step_sec,
            "median_step_sec": self.median_step_sec,
            "tok_sec": self.tok_sec,
            "step_sec": self.step_sec,
            "compile_time_s": self.compile_time_s,
            "peak_memory_gb": self.peak_memory_gb,
            "active_memory_gb": self.active_memory_gb,
            "cache_memory_gb": self.cache_memory_gb,
            "profiling_trace_path": self.profiling_trace_path,
            "profiling_trace_captured": self.profiling_trace_captured,
            "profiling_capture_receipt_path": self.profiling_capture_receipt_path,
            "profiling_capture_status": self.profiling_capture_status,
            "cache_hit": self.cache_state.get("cache_hit"),
            "selected_schedule": json.dumps(self.selected_schedule, sort_keys=True),
            "proof_result": json.dumps(self.proof_result, sort_keys=True),
            "pass_fail_reason": reason,
            "cppmega_sha": self.cppmega_sha,
            "tilelang_sha": self.tilelang_sha,
            "mlx_sha": self.mlx_sha,
            "mlx_version": self.mlx_version,
            "command": self.command,
            "receipt_path": self.receipt_path,
            "returncode": self.returncode,
            "duration_s": self.duration_s,
        }


def parse_csv_list(spec: str, choices: tuple[str, ...]) -> tuple[str, ...]:
    values: list[str] = []
    for raw in spec.split(","):
        value = raw.strip().lower()
        if not value:
            continue
        if value not in choices:
            raise SystemExit(
                f"unknown value {value!r}; choices: {', '.join(choices)}"
            )
        values.append(value)
    if not values:
        raise SystemExit("expected at least one value")
    return tuple(values)


def _receipt_failure_reason(receipt: dict[str, Any]) -> str | None:
    blockers = receipt.get("blockers")
    if isinstance(blockers, list):
        for blocker in blockers:
            if isinstance(blocker, dict):
                for key in FAILURE_REASON_KEYS:
                    value = blocker.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
            elif isinstance(blocker, str) and blocker.strip():
                return blocker.strip()
    for key in FAILURE_REASON_KEYS:
        value = receipt.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the P12 local 1B-class dtype/optimizer/path matrix.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--dtypes", default=",".join(DTYPE_CHOICES))
    parser.add_argument("--optimizers", default=",".join(OPTIMIZER_CHOICES))
    parser.add_argument("--paths", default=",".join(PATH_CHOICES))
    parser.add_argument(
        "--mamba3-bwd",
        choices=("path_b", "path_c"),
        default="path_b",
        help=(
            "Backward route for Mamba3 inside Path C cells. The default keeps "
            "the performance-safe Path C forward + Path B backward route; use "
            "--mamba3-bwd path_c only for an explicit full-Path-C experiment."
        ),
    )
    parser.add_argument("--fresh-process", action="store_true", default=False)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--tilelang-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument(
        "--m04-memory-cap-gib",
        type=float,
        default=None,
        help=(
            "Forward an explicit MLX memory-limit plan to each m04 cell. "
            "The cap is applied as 0.99 * GiB for both wired and Metal limits "
            "so benchmarked cells stay below the requested ceiling."
        ),
    )
    parser.add_argument(
        "--use-path-c-direct-chain-runtime",
        action="store_true",
        help=(
            "Forward --use-path-c-direct-chain-runtime to Path C m04 cells so "
            "the matrix measures the opt-in direct-chain critical-path route "
            "instead of the split/generated Path C fallback."
        ),
    )
    parser.add_argument(
        "--use-path-c-fused-train-block-runtime",
        action="store_true",
        help=(
            "Forward --use-path-c-fused-train-block-runtime to Path C m04 cells. "
            "This opt-in measures the monolithic generated fused train-block "
            "critical path and is intentionally separate from the default "
            "matrix because Metal pipeline compilation can exceed the memory cap."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write planned Markdown/CSV/JSON receipts without executing cells.",
    )
    parser.add_argument(
        "--reuse-existing-ok",
        action="store_true",
        help=(
            "Reuse existing per-cell m04 JSON receipts whose status is ok; "
            "missing or failed cells still execute."
        ),
    )
    parser.add_argument(
        "--completed-receipt-exit-grace-s",
        type=float,
        default=0.0,
        help=(
            "If a cell has already written a valid status=ok JSON receipt but "
            "the subprocess does not exit, wait this many seconds and then "
            "terminate it. Disabled by default."
        ),
    )
    parser.add_argument(
        "--capture-profiles",
        action="store_true",
        help=(
            "Run each cell under scripts/profile_capture.py and attach the "
            "resulting MLX Metal capture receipt to the matrix row."
        ),
    )
    parser.add_argument(
        "--profile-trace-dir",
        type=Path,
        default=None,
        help=(
            "Directory for per-cell .gputrace outputs. Defaults to "
            "<work-dir>/profiles when --capture-profiles is enabled."
        ),
    )
    parser.add_argument(
        "--profile-capture-timeout-s",
        type=float,
        default=60.0,
        help="Timeout forwarded to scripts/profile_capture.py for each cell.",
    )
    return parser


def dtype_optimizer_mapping(
    dtype: str,
    optimizer: str,
    *,
    path: str,
) -> tuple[str, str, bool, str | None]:
    optimizer_map = {
        "adamw": "adamw",
        "adam8bit": "adam8bit",
        "lion": "lion",
        "lion8bit": "lion8bit",
        "muon": "muon",
        "muon_adamw": "muon_adamw",
        "muon_int8": "int8",
    }
    cli_optimizer = optimizer_map.get(optimizer)
    if cli_optimizer is None:
        return "bfloat16", optimizer, False, f"unsupported optimizer {optimizer!r}"
    if dtype == "bf16":
        return "bfloat16", cli_optimizer, True, None
    if dtype == "fp8":
        return ("fp8_path_b" if path == "path_b" else "fp8_path_c"), cli_optimizer, True, None
    return "bfloat16", optimizer, False, f"unsupported dtype/optimizer pair {dtype}/{optimizer}"


def path_env_and_support(
    *,
    dtype: str,
    path: str,
    cache_dir: Path,
    mamba3_bwd: str,
) -> tuple[dict[str, str], bool, str | None, str, Path | None]:
    if dtype == "fp8" and path == "path_b":
        return (
            {
                "CPPMEGA_KERNEL_PATH": "path_b",
                "CPPMEGA_KERNEL_PATH__MAMBA3_MIMO": "path_b",
                "CPPMEGA_KERNEL_PATH__M2RNN": "path_b",
                "CPPMEGA_KERNEL_PATH__SPARSE_MLA": "path_b",
                SPARSE_MLA_FP8_ROUTE_ENV: "path_b",
            },
            True,
            None,
            "not_applicable",
            None,
        )
    if path == "path_b":
        return (
            {
                "CPPMEGA_KERNEL_PATH": "path_b",
                "CPPMEGA_KERNEL_PATH__MAMBA3_MIMO": "path_b",
                "CPPMEGA_KERNEL_PATH__M2RNN": "path_b",
                "CPPMEGA_KERNEL_PATH__SPARSE_MLA": "path_b",
            },
            True,
            None,
            "not_applicable",
            None,
        )
    if path == "path_c_cold":
        env = {
            "CPPMEGA_KERNEL_PATH": "path_c",
            "CPPMEGA_KERNEL_PATH__MAMBA3_MIMO": "path_c",
            "CPPMEGA_KERNEL_PATH__M2RNN": "path_c",
            "CPPMEGA_KERNEL_PATH__SPARSE_MLA": "path_c",
            MAMBA3_PATH_C_BWD_ENV: mamba3_bwd,
        }
        if dtype == "fp8":
            env[SPARSE_MLA_FP8_ROUTE_ENV] = "path_c"
            env[SPARSE_MLA_FP8_BWD_ENV] = "path_c"
        return (env, True, None, "cold", cache_dir)
    if path == "path_c_warm":
        env = {
            "CPPMEGA_KERNEL_PATH": "path_c",
            "CPPMEGA_KERNEL_PATH__MAMBA3_MIMO": "path_c",
            "CPPMEGA_KERNEL_PATH__M2RNN": "path_c",
            "CPPMEGA_KERNEL_PATH__SPARSE_MLA": "path_c",
            MAMBA3_PATH_C_BWD_ENV: mamba3_bwd,
        }
        if dtype == "fp8":
            env[SPARSE_MLA_FP8_ROUTE_ENV] = "path_c"
            env[SPARSE_MLA_FP8_BWD_ENV] = "path_c"
        return (env, True, None, "warm", cache_dir)
    return {}, False, f"unknown path {path!r}", "not_applicable", None


def build_cell(
    *,
    dtype: str,
    optimizer: str,
    path: str,
    args: argparse.Namespace,
) -> MatrixCell:
    dtype_arg, cli_optimizer, supported, unsupported_reason = dtype_optimizer_mapping(
        dtype,
        optimizer,
        path=path,
    )
    cache_dir = args.tilelang_cache_dir / f"{dtype}_{optimizer}"
    path_env, path_supported, path_reason, cache_mode, cache_path = path_env_and_support(
        dtype=dtype,
        path=path,
        cache_dir=cache_dir,
        mamba3_bwd=args.mamba3_bwd,
    )
    if not path_supported:
        supported = False
        unsupported_reason = path_reason
    output_json = args.work_dir / f"{dtype}_{optimizer}_{path}.json"
    command = (
        sys.executable,
        "scripts/m04_train_step.py",
        "--model-profile",
        "local_gb10_quarter",
        "--data-path",
        str(TARGET_PARQUET.relative_to(ROOT)),
        "--data-format",
        "parquet",
        "--token-key",
        "token_ids",
        "--steps",
        str(args.steps),
        "--batch-size",
        str(args.batch_size),
        "--seq-len",
        str(args.block_size),
        "--dtype",
        dtype_arg,
        "--optimizer",
        cli_optimizer,
        "--optimizer-quant-scheme",
        "dynamic_int8_v1",
        "--lr",
        "1e-4",
        "--grad-checkpoint",
        "--output",
        str(output_json),
        "--json",
    )
    if path.startswith("path_c") and bool(args.use_path_c_direct_chain_runtime):
        command += ("--use-path-c-direct-chain-runtime",)
    if path.startswith("path_c") and bool(
        args.use_path_c_fused_train_block_runtime
    ):
        command += ("--use-path-c-fused-train-block-runtime",)
    if args.m04_memory_cap_gib is not None:
        if args.m04_memory_cap_gib <= 0:
            raise SystemExit("--m04-memory-cap-gib must be positive")
        command += (
            "--memory-limit-total-bytes",
            str(int(float(args.m04_memory_cap_gib) * (1024**3))),
            "--memory-limit-wired-ratio",
            "0.99",
            "--memory-limit-metal-ratio",
            "0.99",
            "--apply-memory-limit-plan",
        )
    return MatrixCell(
        dtype=dtype,
        optimizer=optimizer,
        path=path,
        dtype_arg=dtype_arg,
        cli_optimizer=cli_optimizer,
        supported=supported,
        unsupported_reason=unsupported_reason,
        output_json=output_json,
        command=command,
        env=path_env,
        cache_mode=cache_mode,
        cache_dir=cache_path,
    )


def build_cells(args: argparse.Namespace) -> list[MatrixCell]:
    dtypes = parse_csv_list(args.dtypes, DTYPE_CHOICES)
    optimizers = parse_csv_list(args.optimizers, OPTIMIZER_CHOICES)
    paths = parse_csv_list(args.paths, PATH_CHOICES)
    cells = [
        build_cell(dtype=dtype, optimizer=optimizer, path=path, args=args)
        for dtype in dtypes
        for optimizer in optimizers
        for path in paths
    ]
    if args.max_cells is not None:
        cells = cells[: args.max_cells]
    return cells


def run_capture(command: list[str], *, cwd: Path) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def git_sha(cwd: Path) -> str:
    return run_capture(["git", "rev-parse", "--short", "HEAD"], cwd=cwd) or "unknown"


def tilelang_root_from_env() -> Path | None:
    for key in ("TILELANG_DEV_BUILD_ROOT", "TVM_LIBRARY_PATH"):
        raw = os.environ.get(key)
        if not raw:
            continue
        for token in raw.split(os.pathsep):
            path = Path(token)
            candidates = [path, path.parent]
            if path.name == "build":
                candidates.append(path.parent)
            for candidate in candidates:
                if (candidate / "tilelang").exists() and (candidate / ".git").exists():
                    return candidate
    sibling = ROOT.parent / "tilelang"
    if (sibling / "tilelang").exists() and (sibling / ".git").exists():
        return sibling
    return None


def mlx_version_and_sha() -> tuple[str, str]:
    code = (
        "import mlx.core as mx; "
        "v=getattr(mx, '__version__', '') or ''; "
        "print(v)"
    )
    version = run_capture([sys.executable, "-c", code], cwd=ROOT)
    sha = "unknown"
    if "+" in version:
        sha = version.rsplit("+", 1)[-1]
    return version or "unknown", sha


def software_identity() -> dict[str, str]:
    tilelang_root = tilelang_root_from_env()
    mlx_version, mlx_sha = mlx_version_and_sha()
    return {
        "cppmega_sha": git_sha(ROOT),
        "tilelang_sha": git_sha(tilelang_root) if tilelang_root else "unknown",
        "tilelang_root": str(tilelang_root) if tilelang_root else "unknown",
        "mlx_version": mlx_version,
        "mlx_sha": mlx_sha,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def cache_file_count(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file())


def reused_cache_state(cell: MatrixCell) -> dict[str, Any]:
    file_count = cache_file_count(cell.cache_dir)
    return {
        "cache_mode": cell.cache_mode,
        "cache_dir": str(cell.cache_dir) if cell.cache_dir else None,
        "cache_files_before": None,
        "cache_files_after": file_count if cell.cache_dir is not None else None,
        "cache_hit": (
            cell.cache_mode == "warm" and file_count > 0
            if cell.cache_dir is not None
            else None
        ),
        "reused_existing_receipt": True,
    }


def prepare_cache(cell: MatrixCell, *, fresh_process: bool) -> dict[str, Any]:
    if cell.cache_dir is None:
        return {
            "cache_mode": cell.cache_mode,
            "cache_dir": None,
            "cache_files_before": None,
            "cache_files_after": None,
            "cache_hit": None,
        }
    if cell.cache_mode == "cold" and fresh_process and cell.cache_dir.exists():
        shutil.rmtree(cell.cache_dir)
    cell.cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = cell.cache_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    before = cache_file_count(cell.cache_dir)
    return {
        "cache_mode": cell.cache_mode,
        "cache_dir": str(cell.cache_dir),
        "cache_tmp_dir": str(tmp_dir),
        "cache_files_before": before,
        "cache_files_after": None,
        "cache_hit": cell.cache_mode == "warm" and before > 0,
    }


def command_string(command: tuple[str, ...]) -> str:
    return " ".join(command)


def profile_capture_paths(
    cell: MatrixCell,
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    trace_dir = args.profile_trace_dir or (args.work_dir / "profiles")
    return (
        trace_dir / f"{cell.case_id}.gputrace",
        trace_dir / f"{cell.case_id}_capture.json",
    )


def profile_capture_command(
    command: tuple[str, ...],
    *,
    trace_path: Path,
    timeout_s: float,
) -> tuple[str, ...]:
    return (
        sys.executable,
        "scripts/profile_capture.py",
        "--trace-path",
        str(trace_path),
        "--timeout-s",
        f"{float(timeout_s):g}",
        "--",
        *command,
    )


def profile_capture_receipt_from_stdout(stdout: str) -> dict[str, Any] | None:
    for raw_line in reversed(stdout.splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(payload, dict)
            and payload.get("kind") == "cppmega_mlx_metal_capture_receipt"
        ):
            return payload
    return None


def profile_capture_receipt_from_process(
    process: subprocess.CompletedProcess[str],
    *,
    trace_path: Path,
    capture_receipt_path: Path,
) -> dict[str, Any]:
    receipt = profile_capture_receipt_from_stdout(process.stdout or "")
    if receipt is None:
        receipt = {
            "kind": "cppmega_mlx_metal_capture_receipt",
            "status": "missing_capture_receipt",
            "error": "scripts/profile_capture.py did not emit a JSON receipt",
            "trace_path": str(trace_path),
            "capture_started": False,
            "capture_stopped": False,
            "command_status": "unknown",
            "returncode": process.returncode,
        }
    capture_receipt_path.parent.mkdir(parents=True, exist_ok=True)
    capture_receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def existing_receipt_is_ok(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return isinstance(receipt, dict) and receipt.get("status") == "ok"


def unsupported_result(cell: MatrixCell, identity: dict[str, str]) -> CellResult:
    status = (
        STATUS_NOT_APPLICABLE
        if cell.dtype == "fp8" and cell.path == "path_b"
        else STATUS_UNSUPPORTED
    )
    return CellResult(
        case_id=cell.case_id,
        dtype=cell.dtype,
        optimizer=cell.optimizer,
        path=cell.path,
        status=status,
        command=command_string(cell.command),
        cppmega_sha=identity["cppmega_sha"],
        tilelang_sha=identity["tilelang_sha"],
        mlx_sha=identity["mlx_sha"],
        mlx_version=identity["mlx_version"],
        cache_state={
            "cache_mode": cell.cache_mode,
            "cache_dir": str(cell.cache_dir) if cell.cache_dir else None,
            "cache_hit": None,
        },
        cli_optimizer=cell.cli_optimizer,
        pass_fail_reason=cell.unsupported_reason,
    )


def planned_result(cell: MatrixCell, identity: dict[str, str]) -> CellResult:
    return CellResult(
        case_id=cell.case_id,
        dtype=cell.dtype,
        optimizer=cell.optimizer,
        path=cell.path,
        status=STATUS_PLANNED,
        command=command_string(cell.command),
        cppmega_sha=identity["cppmega_sha"],
        tilelang_sha=identity["tilelang_sha"],
        mlx_sha=identity["mlx_sha"],
        mlx_version=identity["mlx_version"],
        cache_state={
            "cache_mode": cell.cache_mode,
            "cache_dir": str(cell.cache_dir) if cell.cache_dir else None,
            "cache_hit": None,
        },
        cli_optimizer=cell.cli_optimizer,
        pass_fail_reason="dry-run plan only; cell not executed",
        receipt_path=str(cell.output_json),
    )


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _first_string(*values: Any) -> str | None:
    for value in values:
        text = _string_or_none(value)
        if text is not None:
            return text
    return None


def _truthy_flag(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    if isinstance(value, int):
        return value == 1
    return False


def profiling_payload_from_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    profiling = receipt.get("profiling") if isinstance(receipt, dict) else {}
    if not isinstance(profiling, dict):
        profiling = {}
    trace_path = _first_string(
        receipt.get("profiling_trace_path"),
        receipt.get("profile_trace_path"),
        receipt.get("trace_path"),
        receipt.get("xctrace_path"),
        profiling.get("profiling_trace_path"),
        profiling.get("profile_trace_path"),
        profiling.get("trace_path"),
        profiling.get("xctrace_path"),
    )
    capture_receipt_path = _first_string(
        receipt.get("profiling_capture_receipt_path"),
        receipt.get("profile_capture_receipt_path"),
        receipt.get("capture_receipt_path"),
        profiling.get("profiling_capture_receipt_path"),
        profiling.get("profile_capture_receipt_path"),
        profiling.get("capture_receipt_path"),
    )
    capture_status = _first_string(
        receipt.get("profiling_capture_status"),
        receipt.get("profile_capture_status"),
        receipt.get("capture_status"),
        profiling.get("profiling_capture_status"),
        profiling.get("profile_capture_status"),
        profiling.get("capture_status"),
    )
    return {
        "profiling_trace_path": trace_path,
        "profiling_trace_captured": bool(
            trace_path or _truthy_flag(receipt.get("profiling_trace_captured"))
        ),
        "profiling_capture_receipt_path": capture_receipt_path,
        "profiling_capture_status": capture_status,
    }


def profiling_payload_from_capture_receipt(
    receipt: dict[str, Any],
    *,
    requested_trace_path: Path | None,
    capture_receipt_path: Path | None,
) -> dict[str, Any]:
    capture_status = _first_string(receipt.get("status"))
    trace_path = _first_string(
        receipt.get("trace_path"),
        str(requested_trace_path) if requested_trace_path is not None else None,
    )
    captured = bool(
        capture_status == "ok"
        and trace_path
        and _truthy_flag(receipt.get("capture_started"))
        and _truthy_flag(receipt.get("capture_stopped"))
    )
    return {
        "profiling_trace_path": trace_path if captured else None,
        "profiling_trace_captured": captured,
        "profiling_capture_receipt_path": (
            str(capture_receipt_path) if capture_receipt_path is not None else None
        ),
        "profiling_capture_status": capture_status,
    }


def selected_schedule_from_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    dispatch = list((receipt.get("training") or {}).get("kernel_dispatch") or [])
    kernel_counts: dict[str, int] = {}
    path_counts: dict[str, int] = {}
    op_kernel: dict[str, str] = {}
    for entry in dispatch:
        kernel = str(entry.get("kernel_used") or "")
        path = str(entry.get("path") or "")
        op = str(entry.get("op_name") or "")
        if kernel:
            kernel_counts[kernel] = kernel_counts.get(kernel, 0) + 1
        if path:
            path_counts[path] = path_counts.get(path, 0) + 1
        if op and kernel:
            op_kernel[op] = kernel
    return {
        "kernel_counts": dict(sorted(kernel_counts.items())),
        "path_counts": dict(sorted(path_counts.items())),
        "op_kernel": dict(sorted(op_kernel.items())),
        "path_c_fusion": path_c_fusion_summary_from_receipt(receipt),
    }


def path_c_fusion_payload_from_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    for section_name in ("training", "workload"):
        section = receipt.get(section_name)
        if not isinstance(section, dict):
            continue
        route = section.get("fp8_path_c_training_route")
        if isinstance(route, dict) and isinstance(route.get("path_c_fusion"), dict):
            return route["path_c_fusion"]
    return {}


def path_c_fusion_summary_from_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    fusion = path_c_fusion_payload_from_receipt(receipt)
    if not fusion:
        return {}
    production = fusion.get("production_schedule")
    if not isinstance(production, dict):
        production = {}
    contract = fusion.get("schedule_contract")
    if not isinstance(contract, dict):
        contract = {}
    runtime_binding = fusion.get("runtime_training_binding")
    if not isinstance(runtime_binding, dict):
        runtime_binding = {}
    direct_chained = fusion.get("direct_chained_fusion")
    if not isinstance(direct_chained, dict):
        direct_chained = {}
    direct_chain_binding = direct_chained.get("runtime_binding")
    if not isinstance(direct_chain_binding, dict):
        direct_chain_binding = {}
    direct_chain_training_contract = direct_chained.get("training_runtime_contract")
    if not isinstance(direct_chain_training_contract, dict):
        direct_chain_training_contract = {}
    direct_chain_training_ready = bool(
        direct_chain_training_contract.get("training_runtime_available")
        or direct_chain_training_contract.get("critical_path_ready")
    )
    runtime_uses_fused_train_block = bool(
        runtime_binding.get("runtime_uses_fused_train_block")
        or direct_chain_training_ready
    )
    runtime_binding_status = (
        "ok"
        if direct_chain_training_ready
        else direct_chain_training_contract.get("status")
        if direct_chain_binding.get("runtime_uses_direct_fusion_chain")
        else runtime_binding.get("status")
    )
    return {
        "mode": fusion.get("mode"),
        "status": fusion.get("status"),
        "schedule_name": fusion.get("schedule_name"),
        "schedule_status": fusion.get("schedule_status"),
        "single_kernel_fused": fusion.get("single_kernel_fused"),
        "default_allowed": fusion.get("default_allowed"),
        "schedule_blockers": [
            item.get("kind")
            for item in fusion.get("schedule_blockers", [])
            if isinstance(item, dict)
        ],
        "production_schedule_id": production.get("schedule_id"),
        "implementation_kind": production.get("implementation_kind"),
        "production_fragments_complete": production.get(
            "production_fragments_complete"
        ),
        "real_abi_contract_complete": production.get(
            "real_abi_contract_complete"
        ),
        "missing_real_abi_inputs": production.get("missing_real_abi_inputs"),
        "schedule_contract_status": contract.get("status"),
        "runtime_binding_status": runtime_binding_status,
        "runtime_uses_fused_train_block": runtime_uses_fused_train_block,
        "runtime_uses_direct_fusion_chain": direct_chain_binding.get(
            "runtime_uses_direct_fusion_chain"
        ),
        "direct_chain_status": direct_chained.get("status"),
        "direct_chain_segment_count": direct_chained.get("segment_count"),
        "direct_chain_runtime_binding_status": direct_chain_binding.get("status"),
        "direct_chain_training_runtime_status": (
            direct_chain_training_contract.get("status")
        ),
        "direct_chain_training_runtime_available": (
            direct_chain_training_contract.get("training_runtime_available")
        ),
        "required_bank_buffers": runtime_binding.get("required_bank_buffers"),
        "missing_bank_buffers": runtime_binding.get("missing_bank_buffers"),
    }


def proof_result_from_receipt(receipt: dict[str, Any], *, path: str) -> dict[str, Any]:
    training = receipt.get("training") if isinstance(receipt, dict) else {}
    route = {}
    if isinstance(training, dict):
        route = training.get("fp8_path_c_training_route") or {}
    fusion_summary = path_c_fusion_summary_from_receipt(receipt)
    route_reports_fused = bool(
        isinstance(route, dict) and route.get("fused_train_block_runtime_available")
    )
    fusion_reports_fused = bool(fusion_summary.get("runtime_uses_fused_train_block"))
    fused_train_block_runtime_available = bool(
        route_reports_fused and fusion_reports_fused
    )
    return {
        "path": path,
        "proof_source": (
            "per-kernel runtime dispatch log plus Path C fusion planner metadata; "
            "fused schedule native compile is a separate compile receipt"
        ),
        "path_c_requested": path.startswith("path_c"),
        "fp8_path_c_route_status": route.get("status") if isinstance(route, dict) else None,
        "kernel_surface_available": (
            route.get("kernel_surface_available") if isinstance(route, dict) else None
        ),
        "path_c_fusion": fusion_summary,
        "runtime_uses_fused_train_block": fused_train_block_runtime_available,
        "fused_train_block_runtime_available": (
            fused_train_block_runtime_available
        ),
    }


def extract_result(
    *,
    cell: MatrixCell,
    identity: dict[str, str],
    cache_state: dict[str, Any],
    process: subprocess.CompletedProcess[str],
    duration_s: float,
    profile_trace_path: Path | None = None,
    profile_capture_receipt_path: Path | None = None,
    profile_capture_receipt: dict[str, Any] | None = None,
) -> CellResult:
    cache_after = cache_file_count(cell.cache_dir)
    if cell.cache_dir is not None:
        cache_state["cache_files_after"] = cache_after
        if cell.cache_mode == "warm":
            cache_files_before = cache_state.get("cache_files_before")
            cache_state["cache_hit"] = bool(
                (cache_files_before if cache_files_before is not None else cache_after)
                > 0
            )
        elif cell.cache_mode == "cold":
            cache_state["cache_hit"] = False

    receipt: dict[str, Any] = {}
    if cell.output_json.exists():
        try:
            receipt = json.loads(cell.output_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            receipt = {}
    status = STATUS_OK if process.returncode == 0 and receipt.get("status") == "ok" else STATUS_FAILED
    timing = receipt.get("timing") if isinstance(receipt, dict) else {}
    memory = receipt.get("memory") if isinstance(receipt, dict) else {}
    training = receipt.get("training") if isinstance(receipt, dict) else {}
    step_times = list(timing.get("step_times_s") or []) if isinstance(timing, dict) else []
    first_step_sec = float(step_times[0]) if step_times else None
    compile_time_s = first_step_sec
    steady_times = [float(value) for value in step_times[1:]] if len(step_times) > 1 else []
    step_sec = (sum(steady_times) / len(steady_times)) if steady_times else (
        float(timing.get("mean_step_time_s")) if isinstance(timing, dict) and timing.get("mean_step_time_s") is not None else None
    )
    sorted_steady_times = sorted(steady_times)
    median_step_sec = (
        sorted_steady_times[len(sorted_steady_times) // 2]
        if sorted_steady_times
        else first_step_sec
    )
    tok_sec = (
        float(timing.get("tokens_per_second"))
        if isinstance(timing, dict) and timing.get("tokens_per_second") is not None
        else None
    )
    peak_bytes = (
        int(memory.get("peak_memory_bytes"))
        if isinstance(memory, dict) and memory.get("peak_memory_bytes") is not None
        else None
    )
    memory_after = memory.get("after") if isinstance(memory, dict) else {}
    active_bytes = (
        int(memory_after.get("active_memory_bytes"))
        if isinstance(memory_after, dict)
        and memory_after.get("active_memory_bytes") is not None
        else None
    )
    cache_bytes = (
        int(memory_after.get("cache_memory_bytes"))
        if isinstance(memory_after, dict)
        and memory_after.get("cache_memory_bytes") is not None
        else None
    )
    reason = None
    if status != STATUS_OK:
        reason = (
            _receipt_failure_reason(receipt)
            or process.stderr.strip()
            or process.stdout.strip()
            or f"cell exited {process.returncode}"
        )
    elif isinstance(training, dict) and training.get("all_finite") is False:
        status = STATUS_FAILED
        reason = "training reported non-finite values"
    else:
        reason = "ok"
    optimizer_payload = (
        training.get("optimizer")
        if isinstance(training, dict) and isinstance(training.get("optimizer"), dict)
        else {}
    )
    profiling_payload = profiling_payload_from_receipt(receipt)
    if profile_capture_receipt is not None:
        capture_profiling_payload = profiling_payload_from_capture_receipt(
            profile_capture_receipt,
            requested_trace_path=profile_trace_path,
            capture_receipt_path=profile_capture_receipt_path,
        )
        if not profiling_payload["profiling_trace_captured"]:
            profiling_payload = capture_profiling_payload
        else:
            profiling_payload["profiling_capture_receipt_path"] = (
                profiling_payload["profiling_capture_receipt_path"]
                or capture_profiling_payload["profiling_capture_receipt_path"]
            )
            profiling_payload["profiling_capture_status"] = (
                profiling_payload["profiling_capture_status"]
                or capture_profiling_payload["profiling_capture_status"]
            )
    steps_completed = (
        int(training.get("steps_completed"))
        if isinstance(training, dict) and training.get("steps_completed") is not None
        else None
    )
    executed_command = (
        tuple(str(part) for part in process.args) if process.args else cell.command
    )
    return CellResult(
        case_id=cell.case_id,
        dtype=cell.dtype,
        optimizer=cell.optimizer,
        path=cell.path,
        status=status,
        command=command_string(executed_command),
        cppmega_sha=identity["cppmega_sha"],
        tilelang_sha=identity["tilelang_sha"],
        mlx_sha=identity["mlx_sha"],
        mlx_version=identity["mlx_version"],
        cache_state=cache_state,
        cli_optimizer=cell.cli_optimizer,
        optimizer_key=(
            str(optimizer_payload.get("key"))
            if optimizer_payload.get("key") is not None
            else cell.cli_optimizer
        ),
        optimizer_name=(
            str(optimizer_payload.get("name"))
            if optimizer_payload.get("name") is not None
            else None
        ),
        optimizer_class=(
            str(optimizer_payload.get("class"))
            if optimizer_payload.get("class") is not None
            else None
        ),
        optimizer_source=(
            str(optimizer_payload.get("source"))
            if optimizer_payload.get("source") is not None
            else None
        ),
        steps_completed=steps_completed,
        first_step_sec=first_step_sec,
        median_step_sec=median_step_sec,
        tok_sec=tok_sec,
        step_sec=step_sec,
        compile_time_s=compile_time_s,
        peak_memory_bytes=peak_bytes,
        peak_memory_gb=(peak_bytes / (1024**3)) if peak_bytes is not None else None,
        active_memory_gb=(
            active_bytes / (1024**3) if active_bytes is not None else None
        ),
        cache_memory_gb=(
            cache_bytes / (1024**3) if cache_bytes is not None else None
        ),
        profiling_trace_path=profiling_payload["profiling_trace_path"],
        profiling_trace_captured=profiling_payload["profiling_trace_captured"],
        profiling_capture_receipt_path=profiling_payload[
            "profiling_capture_receipt_path"
        ],
        profiling_capture_status=profiling_payload["profiling_capture_status"],
        selected_schedule=selected_schedule_from_receipt(receipt),
        proof_result=proof_result_from_receipt(receipt, path=cell.path),
        pass_fail_reason=reason,
        receipt_path=str(cell.output_json),
        returncode=process.returncode,
        duration_s=duration_s,
    )


def run_cell(cell: MatrixCell, *, args: argparse.Namespace, identity: dict[str, str]) -> CellResult:
    if not cell.supported:
        return unsupported_result(cell, identity)
    if args.dry_run:
        return planned_result(cell, identity)
    if bool(args.reuse_existing_ok) and existing_receipt_is_ok(cell.output_json):
        return extract_result(
            cell=cell,
            identity=identity,
            cache_state=reused_cache_state(cell),
            process=subprocess.CompletedProcess(cell.command, 0, "", ""),
            duration_s=0.0,
        )
    cache_state = prepare_cache(cell, fresh_process=bool(args.fresh_process))
    env = os.environ.copy()
    env.update(cell.env)
    if cell.cache_dir is not None:
        env["TILELANG_CACHE_DIR"] = str(cell.cache_dir)
        env["TILELANG_TMP_DIR"] = str(cell.cache_dir / "tmp")
        env.pop("TILELANG_DISABLE_CACHE", None)
    cell.output_json.parent.mkdir(parents=True, exist_ok=True)
    profile_trace_path: Path | None = None
    profile_capture_receipt_path: Path | None = None
    command = list(cell.command)
    completed_receipt_exit_grace_s = float(args.completed_receipt_exit_grace_s)
    if bool(args.capture_profiles):
        profile_trace_path, profile_capture_receipt_path = profile_capture_paths(
            cell,
            args,
        )
        profile_trace_path.parent.mkdir(parents=True, exist_ok=True)
        profile_capture_receipt_path.parent.mkdir(parents=True, exist_ok=True)
        command = list(
            profile_capture_command(
                cell.command,
                trace_path=profile_trace_path,
                timeout_s=float(args.profile_capture_timeout_s),
            )
        )
        completed_receipt_exit_grace_s = 0.0
    start = time.perf_counter()
    process, completed_after_ok_receipt = run_cell_process(
        command,
        cwd=ROOT,
        env=env,
        output_json=cell.output_json,
        completed_receipt_exit_grace_s=completed_receipt_exit_grace_s,
    )
    duration = time.perf_counter() - start
    profile_capture_receipt: dict[str, Any] | None = None
    if profile_trace_path is not None and profile_capture_receipt_path is not None:
        profile_capture_receipt = profile_capture_receipt_from_process(
            process,
            trace_path=profile_trace_path,
            capture_receipt_path=profile_capture_receipt_path,
        )
    if completed_after_ok_receipt:
        cache_state["terminated_after_ok_receipt"] = True
        cache_state["completed_receipt_exit_grace_s"] = completed_receipt_exit_grace_s
    return extract_result(
        cell=cell,
        identity=identity,
        cache_state=cache_state,
        process=process,
        duration_s=duration,
        profile_trace_path=profile_trace_path,
        profile_capture_receipt_path=profile_capture_receipt_path,
        profile_capture_receipt=profile_capture_receipt,
    )


def run_cell_process(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    output_json: Path,
    completed_receipt_exit_grace_s: float = 0.0,
) -> tuple[subprocess.CompletedProcess[str], bool]:
    with tempfile.TemporaryFile("w+", encoding="utf-8") as stdout_file, tempfile.TemporaryFile(
        "w+",
        encoding="utf-8",
    ) as stderr_file:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            text=True,
            stdout=stdout_file,
            stderr=stderr_file,
        )
        grace_started: float | None = None
        terminated_after_ok_receipt = False
        forced_returncode: int | None = None
        while True:
            returncode = process.poll()
            if returncode is not None:
                break
            if completed_receipt_exit_grace_s > 0 and existing_receipt_is_ok(output_json):
                now = time.perf_counter()
                if grace_started is None:
                    grace_started = now
                elif now - grace_started >= completed_receipt_exit_grace_s:
                    terminated_after_ok_receipt = True
                    process.terminate()
                    try:
                        process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    forced_returncode = 0
                    break
            else:
                grace_started = None
            time.sleep(1.0)
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read()
        stderr = stderr_file.read()
        if terminated_after_ok_receipt:
            stderr += (
                "\nbench harness terminated subprocess after status=ok receipt "
                f"remained alive for {completed_receipt_exit_grace_s:.1f}s"
            )
        return (
            subprocess.CompletedProcess(
                command,
                forced_returncode if forced_returncode is not None else process.returncode,
                stdout or "",
                stderr or "",
            ),
            terminated_after_ok_receipt,
        )


def write_csv(path: Path, results: list[CellResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [result.to_row(max_reason_chars=2000) for result in results]
    fieldnames = list(rows[0]) if rows else list(CellResult("", "", "", "", "", "", "", "", "", "", {}).to_row())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _short_text(value: str | None, limit: int = 500) -> str | None:
    if value is None or len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


def write_markdown(path: Path, *, results: list[CellResult], identity: dict[str, str], command: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# cppmega 1B Path Matrix",
        "",
        f"- Command: `{command}`",
        f"- cppmega SHA: `{identity['cppmega_sha']}`",
        f"- TileLang SHA: `{identity['tilelang_sha']}`",
        f"- MLX SHA: `{identity['mlx_sha']}`",
        f"- MLX version: `{identity['mlx_version']}`",
        "",
        "| dtype | optimizer | path | status | tok/s | step/s | compile s | peak GB | cache hit | profile trace | reason |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for result in results:
        step_per_second = (1.0 / result.step_sec) if result.step_sec else None
        lines.append(
            "| {dtype} | {optimizer} | {path} | {status} | {tok} | {step} | {compile} | {peak} | {cache} | {trace} | {reason} |".format(
                dtype=result.dtype,
                optimizer=result.optimizer,
                path=result.path,
                status=result.status,
                tok=_fmt(result.tok_sec),
                step=_fmt(step_per_second),
                compile=_fmt(result.compile_time_s),
                peak=_fmt(result.peak_memory_gb),
                cache=_fmt(result.cache_state.get("cache_hit")),
                trace=(
                    _short_text(result.profiling_trace_path, 500) or ""
                ).replace("|", "/"),
                reason=(_short_text(result.pass_fail_reason, 500) or "").replace("|", "/"),
            )
        )
    lines.extend(
        [
            "",
            "## Cell Commands",
            "",
        ]
    )
    for result in results:
        lines.append(f"- `{result.case_id}`: `{result.command}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cells = build_cells(args)
    identity = software_identity()
    results: list[CellResult] = []
    for index, cell in enumerate(cells, start=1):
        print(f"[{index}/{len(cells)}] {cell.case_id}", flush=True)
        results.append(run_cell(cell, args=args, identity=identity))
    command = command_string(tuple(sys.argv))
    write_markdown(args.out, results=results, identity=identity, command=command)
    write_csv(args.csv, results)
    write_json(
        args.json,
        {
            "schema_version": 1,
            "scope": "cppmega_1b_path_matrix",
            "command": command,
            "config": {
                "batch_size": args.batch_size,
                "block_size": args.block_size,
                "steps": args.steps,
                "dtypes": list(parse_csv_list(args.dtypes, DTYPE_CHOICES)),
                "optimizers": list(parse_csv_list(args.optimizers, OPTIMIZER_CHOICES)),
                "paths": list(parse_csv_list(args.paths, PATH_CHOICES)),
                "mamba3_bwd": args.mamba3_bwd,
                "fresh_process": bool(args.fresh_process),
                "dry_run": bool(args.dry_run),
                "reuse_existing_ok": bool(args.reuse_existing_ok),
            },
            "software": identity,
            "results": [result.to_row() for result in results],
        },
    )
    failures = [result for result in results if result.status == STATUS_FAILED]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

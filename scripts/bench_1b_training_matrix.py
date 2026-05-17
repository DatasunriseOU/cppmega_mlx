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

DTYPE_CHOICES = ("bf16", "fp8", "int8")
OPTIMIZER_CHOICES = ("adamw", "lion", "muon", "muon_adamw")
PATH_CHOICES = ("path_b", "path_c_cold", "path_c_warm")
STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_UNSUPPORTED = "unsupported"
STATUS_NOT_APPLICABLE = "not_applicable"
STATUS_PLANNED = "planned"
MAMBA3_PATH_C_BWD_ENV = "CPPMEGA_MAMBA3_PATH_C_BWD"
SPARSE_MLA_FP8_ROUTE_ENV = "CPPMEGA_SPARSE_MLA_FP8_ROUTE"


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
    parser.add_argument("--fresh-process", action="store_true", default=False)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--tilelang-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--max-cells", type=int, default=None)
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
    return parser


def dtype_optimizer_mapping(
    dtype: str,
    optimizer: str,
    *,
    path: str,
) -> tuple[str, str, bool, str | None]:
    if dtype == "bf16":
        return "bfloat16", optimizer, True, None
    if dtype == "fp8":
        return ("fp8_path_b" if path == "path_b" else "fp8_path_c"), optimizer, True, None
    if dtype == "int8":
        if optimizer in {"muon", "muon_adamw"}:
            return "bfloat16", "int8", True, None
        if optimizer == "adamw":
            return "bfloat16", "adam8bit", True, None
        if optimizer == "lion":
            return "bfloat16", "lion8bit", True, None
    return "bfloat16", optimizer, False, f"unsupported dtype/optimizer pair {dtype}/{optimizer}"


def path_env_and_support(
    *,
    dtype: str,
    path: str,
    cache_dir: Path,
) -> tuple[dict[str, str], bool, str | None, str, Path | None]:
    if dtype == "fp8" and path == "path_b":
        return (
            {
                "CPPMEGA_KERNEL_PATH": "auto",
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
            {"CPPMEGA_KERNEL_PATH": "auto"},
            True,
            None,
            "not_applicable",
            None,
        )
    if path == "path_c_cold":
        env = {
            "CPPMEGA_KERNEL_PATH": "path_c",
            MAMBA3_PATH_C_BWD_ENV: "path_b",
        }
        if dtype == "fp8":
            env[SPARSE_MLA_FP8_ROUTE_ENV] = "path_c"
        return (env, True, None, "cold", cache_dir)
    if path == "path_c_warm":
        env = {
            "CPPMEGA_KERNEL_PATH": "path_c",
            MAMBA3_PATH_C_BWD_ENV: "path_b",
        }
        if dtype == "fp8":
            env[SPARSE_MLA_FP8_ROUTE_ENV] = "path_c"
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
    }


def proof_result_from_receipt(receipt: dict[str, Any], *, path: str) -> dict[str, Any]:
    training = receipt.get("training") if isinstance(receipt, dict) else {}
    route = {}
    if isinstance(training, dict):
        route = training.get("fp8_path_c_training_route") or {}
    return {
        "path": path,
        "proof_source": "per-kernel TileLang route receipts plus runtime dispatch log",
        "path_c_requested": path.startswith("path_c"),
        "fp8_path_c_route_status": route.get("status") if isinstance(route, dict) else None,
        "kernel_surface_available": (
            route.get("kernel_surface_available") if isinstance(route, dict) else None
        ),
    }


def extract_result(
    *,
    cell: MatrixCell,
    identity: dict[str, str],
    cache_state: dict[str, Any],
    process: subprocess.CompletedProcess[str],
    duration_s: float,
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
            receipt.get("status_reason")
            or receipt.get("failure_reason")
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
    steps_completed = (
        int(training.get("steps_completed"))
        if isinstance(training, dict) and training.get("steps_completed") is not None
        else None
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
    start = time.perf_counter()
    process = subprocess.run(
        list(cell.command),
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    duration = time.perf_counter() - start
    return extract_result(
        cell=cell,
        identity=identity,
        cache_state=cache_state,
        process=process,
        duration_s=duration,
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
        "| dtype | optimizer | path | status | tok/s | step/s | compile s | peak GB | cache hit | reason |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for result in results:
        step_per_second = (1.0 / result.step_sec) if result.step_sec else None
        lines.append(
            "| {dtype} | {optimizer} | {path} | {status} | {tok} | {step} | {compile} | {peak} | {cache} | {reason} |".format(
                dtype=result.dtype,
                optimizer=result.optimizer,
                path=result.path,
                status=result.status,
                tok=_fmt(result.tok_sec),
                step=_fmt(step_per_second),
                compile=_fmt(result.compile_time_s),
                peak=_fmt(result.peak_memory_gb),
                cache=_fmt(result.cache_state.get("cache_hit")),
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

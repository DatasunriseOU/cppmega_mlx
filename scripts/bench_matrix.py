#!/usr/bin/env python3
"""Run a small matrix of comparable MLX training microbenchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bench_tiny import (  # noqa: E402
    MODEL_ROUTES,
    BenchConfig,
    default_hardware_label,
    dry_run_payload,
    run_benchmark,
)


@dataclass(frozen=True)
class MatrixProfile:
    name: str
    vocab_size: int
    d_model: int
    n_heads: int
    n_layers: int
    mlp_dim: int


PROFILES: dict[str, MatrixProfile] = {
    "smoke": MatrixProfile(
        name="smoke",
        vocab_size=32,
        d_model=8,
        n_heads=1,
        n_layers=1,
        mlp_dim=16,
    ),
    "tiny": MatrixProfile(
        name="tiny",
        vocab_size=2048,
        d_model=128,
        n_heads=4,
        n_layers=2,
        mlp_dim=512,
    ),
    "hybrid-smoke": MatrixProfile(
        name="hybrid-smoke",
        vocab_size=32,
        d_model=8,
        n_heads=1,
        n_layers=4,
        mlp_dim=16,
    ),
}

ROUTE_ALIASES = {
    "plain": ("tiny", False),
    "structure": ("tiny", True),
    "mamba3": ("hybrid-m", True),
    "m2rnn": ("hybrid-r", True),
    "hybrid-aemr": ("hybrid", True),
}
ROUTES = (*ROUTE_ALIASES, *MODEL_ROUTES)
ROUTE_ALIAS_HELP = {
    "mamba3": "hybrid-m",
    "m2rnn": "hybrid-r",
    "hybrid-aemr": "hybrid",
}

BENCH_RECEIPT_SCHEMA_VERSION = 1
BASELINE_ARCHIVE_SCHEMA_VERSION = 1
BASELINE_ARCHIVE_KIND = "cppmega.mlx.local_m4_benchmark_baselines"
BASELINE_RECORD_SCHEMA_VERSION = 1
BASELINE_RECORD_KIND = "cppmega.mlx.local_m4_benchmark_baseline_record"
MATCHED_RUN_GUARD = (
    "compare only against matched rows with identical profile, route, "
    "workload_key, and software_key"
)
MATRIX_MATCHED_RUN_POLICY = (
    "Matrix output is evidence inventory only; make M4 Max vs GB10 "
    "claims only after both hardware labels have rows with identical "
    "comparison_key.workload and comparison_key.software values."
)
REQUIRED_RECEIPT_FIELDS = (
    "hardware_label",
    "software.mlx_version",
    "software.mlx_lm_version",
    "software.mlx_metal_version",
    "software.default_device",
    "software.device_name",
    "route",
    "seq_len",
    "batch_size",
    "warmup_steps",
    "measured_steps",
    "compile",
    "timing.wall_time_s",
    "timing.tokens_per_second_or_step_time",
)
COMPARE_LINE_FIELDS = (
    "hardware_label",
    "dtype",
    "batch_size",
    "seq_len",
    "warmup_steps",
    "measured_steps",
    "compile",
    "include_structure",
    "tokens_per_second",
    "peak_memory_bytes",
)
LOCAL_BASELINE_POLICY = (
    "This archive is a local M4 baseline/regression ledger only. It does not "
    "contain a GB10 parity claim; run scripts/compare_bench_rows.py on matched "
    "M4 and GB10 rows before reporting any cross-host ratio."
)
LOCAL_RECEIPT_SCOPE = "local_only"
LOCAL_ONLY_RECEIPT_POLICY = (
    "Single-host matrix receipt only; not M4-vs-GB10 parity evidence. "
    "Cross-host ratios require matched M4 and GB10 rows with identical "
    "comparison_key.workload and comparison_key.software values."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a comparable M4/GB10 matrix over scripts/bench_tiny.py.",
        epilog=(
            "Matched-run guard: do not claim M4-vs-GB10 parity unless both hosts "
            "use identical profile, route, dtype, batch, seq, compile mode, "
            "warmup, measured steps, data contract, and software-stack fields."
        ),
    )
    parser.add_argument(
        "--hardware-label",
        default=default_hardware_label(),
        help="Human label for this host, e.g. 'M4 Max' or 'GB10'.",
    )
    parser.add_argument(
        "--batch-sizes",
        default="1,2",
        help="Comma-separated batch sizes.",
    )
    parser.add_argument(
        "--seq-lens",
        default="32,64",
        help="Comma-separated sequence lengths.",
    )
    parser.add_argument(
        "--profiles",
        default="smoke",
        help=f"Comma-separated model profiles. Available: {','.join(sorted(PROFILES))}.",
    )
    parser.add_argument(
        "--routes",
        default="plain",
        help=(
            "Comma-separated model routes. Aliases: plain,structure,mamba3,m2rnn,"
            "hybrid-aemr. "
            f"Model routes: {','.join(MODEL_ROUTES)}."
        ),
    )
    parser.add_argument("--dtype", default=BenchConfig.dtype)
    parser.add_argument("--lr", type=float, default=BenchConfig.learning_rate)
    parser.add_argument("--warmup-steps", type=int, default=BenchConfig.warmup_steps)
    parser.add_argument("--steps", type=int, default=BenchConfig.steps)
    parser.add_argument("--seed", type=int, default=BenchConfig.seed)
    parser.add_argument(
        "--compile-modes",
        default="compiled",
        help="Comma-separated compile modes: compiled,eager.",
    )
    parser.add_argument(
        "--auto-wired-limit",
        action="store_true",
        help="Forward --auto-wired-limit to each tiny benchmark case.",
    )
    parser.add_argument(
        "--wired-limit-bytes",
        type=int,
        default=None,
        help="Forward --wired-limit-bytes to each tiny benchmark case.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit one JSON summary object. This is the default human-readable mode.",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Emit one JSON object per benchmark case.",
    )
    parser.add_argument(
        "--dry-run-json",
        action="store_true",
        help="Validate and emit the planned matrix without running benchmarks.",
    )
    parser.add_argument(
        "--archive-baseline",
        type=Path,
        default=None,
        help=(
            "Append this matrix summary to a schema-versioned local M4 baseline "
            "archive JSON file. The archive is for regression tracking only and "
            "does not make GB10 parity claims."
        ),
    )
    parser.add_argument(
        "--baseline-note",
        default="",
        help="Optional note stored with --archive-baseline records.",
    )
    return parser


def _csv_items(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("comma-separated option must contain at least one value")
    return items


def _csv_ints(value: str, *, name: str) -> list[int]:
    values = [int(item) for item in _csv_items(value)]
    if any(item <= 0 for item in values):
        raise ValueError(f"{name} values must be > 0")
    return values


def _profiles(value: str) -> list[MatrixProfile]:
    profiles: list[MatrixProfile] = []
    for name in _csv_items(value):
        if name not in PROFILES:
            raise ValueError(f"unknown profile {name!r}; expected one of {sorted(PROFILES)}")
        profiles.append(PROFILES[name])
    return profiles


def _routes(value: str) -> list[str]:
    routes = _csv_items(value)
    unknown = sorted(set(routes) - set(ROUTES))
    if unknown:
        raise ValueError(f"unknown route(s) {unknown}; expected {list(ROUTES)}")
    return routes


def _route_config(route: str) -> tuple[str, bool]:
    if route in ROUTE_ALIASES:
        return ROUTE_ALIASES[route]
    return route, route.startswith("hybrid")


def _route_alias_metadata(route: str, model_route: str) -> dict[str, Any]:
    return {
        "requested_route": route,
        "route_alias": route if route != model_route else None,
        "resolved_model_route": model_route,
        "is_route_alias": route != model_route,
    }


def _compile_modes(value: str) -> list[bool]:
    modes: list[bool] = []
    for mode in _csv_items(value):
        if mode == "compiled":
            modes.append(True)
        elif mode == "eager":
            modes.append(False)
        else:
            raise ValueError("compile modes must be 'compiled' or 'eager'")
    return modes


def iter_configs(args: argparse.Namespace) -> list[tuple[str, str, str, BenchConfig]]:
    batches = _csv_ints(args.batch_sizes, name="batch_sizes")
    seq_lens = _csv_ints(args.seq_lens, name="seq_lens")
    profiles = _profiles(args.profiles)
    routes = _routes(args.routes)
    compile_modes = _compile_modes(args.compile_modes)
    configs: list[tuple[str, str, str, BenchConfig]] = []
    for batch_size, seq_len, profile, route, compile_mode in product(
        batches,
        seq_lens,
        profiles,
        routes,
        compile_modes,
    ):
        compile_label = "compiled" if compile_mode else "eager"
        model_route, include_structure = _route_config(route)
        case_id = (
            f"{profile.name}-route_{route}-b{batch_size}-s{seq_len}-"
            f"{args.dtype}-{compile_label}"
        )
        configs.append(
            (
                case_id,
                profile.name,
                route,
                BenchConfig(
                    hardware_label=args.hardware_label,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    vocab_size=profile.vocab_size,
                    d_model=profile.d_model,
                    n_heads=profile.n_heads,
                    n_layers=profile.n_layers,
                    mlp_dim=profile.mlp_dim,
                    dtype=args.dtype,
                    learning_rate=args.lr,
                    warmup_steps=args.warmup_steps,
                    steps=args.steps,
                    seed=args.seed,
                    compile=compile_mode,
                    include_structure=include_structure,
                    model_route=model_route,
                    auto_wired_limit=args.auto_wired_limit,
                    wired_limit_bytes=args.wired_limit_bytes,
                ),
            )
        )
    return configs


def _augment_metrics(
    *,
    case_id: str,
    profile: str,
    route: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    device = metrics.get("device") or {}
    run_metadata = metrics.get("run_metadata") or {}
    framework_metadata = run_metadata.get("framework") or {}
    matched_run = metrics.get("matched_run") or {}
    profile_hooks = metrics.get("profile")
    case_metrics = {key: value for key, value in metrics.items() if key != "profile"}
    matched_key = {
        "profile": profile,
        "route": route,
        **(matched_run.get("key") or {}),
    }
    mlx_device_info = framework_metadata.get("mlx_device_info") or device.get("mlx_device_info") or {}
    metal = framework_metadata.get("metal") or device.get("metal")
    framework_name = "mlx" if (framework_metadata.get("mlx") or device.get("mlx")) else None
    software_key = {
        "framework": framework_name,
        "backend": metrics.get("backend"),
        "execution_backend": metrics.get("backend"),
        "framework_backend": "metal" if metal else framework_name,
        "python_version": framework_metadata.get("python") or device.get("python"),
        "platform": framework_metadata.get("platform") or device.get("platform"),
        "machine": framework_metadata.get("machine") or device.get("machine"),
        "mlx_version": framework_metadata.get("mlx") or device.get("mlx"),
        "mlx_lm_version": framework_metadata.get("mlx_lm") or device.get("mlx_lm"),
        "mlx_metal_version": framework_metadata.get("mlx_metal") or device.get("mlx_metal"),
        "default_device": framework_metadata.get("default_device") or device.get("default_device"),
        "device_name": mlx_device_info.get("device_name"),
        "metal": metal,
    }
    comparison_key = {
        "schema_version": 1,
        "workload": matched_key,
        "software": software_key,
    }
    timing = _timing_receipt(metrics)
    alias_metadata = _route_alias_metadata(route, str(metrics.get("model_route") or ""))
    bench_receipt = {
        "schema_version": BENCH_RECEIPT_SCHEMA_VERSION,
        "receipt_scope": LOCAL_RECEIPT_SCOPE,
        "local_only": True,
        "gb10_parity_claim": False,
        "hardware_label": metrics.get("hardware_label"),
        "profile": profile,
        "route": route,
        "model_route": metrics.get("model_route"),
        **alias_metadata,
        "seq_len": metrics.get("seq_len"),
        "batch_size": metrics.get("batch_size"),
        "dtype": metrics.get("dtype"),
        "warmup_steps": timing["warmup_steps"],
        "measured_steps": timing["measured_steps"],
        "compile": timing["compile"],
        "tokens_per_second": timing["tokens_per_second"],
        "mean_step_time_s": timing["mean_step_time_s"],
        "wall_time_s": timing["wall_time_s"],
        "mean_wall_time_s": timing["mean_wall_time_s"],
        "total_wall_time_s": timing["total_wall_time_s"],
        "median_step_time_s": timing["median_step_time_s"],
        "device": {
            "default_device": software_key["default_device"],
            "device_name": software_key["device_name"],
            "platform": software_key["platform"],
            "machine": software_key["machine"],
            "metal": software_key["metal"],
        },
        "software": software_key,
        "workload": matched_key,
        "timing": timing,
        "comparison_key": comparison_key,
        "matched_run_guard": MATCHED_RUN_GUARD,
        "parity_claim_policy": MATRIX_MATCHED_RUN_POLICY,
        "local_only_policy": LOCAL_ONLY_RECEIPT_POLICY,
    }
    matrix_matched_run = {
        **matched_run,
        "key": matched_key,
        "receipt_scope": LOCAL_RECEIPT_SCOPE,
        "local_only": True,
        "gb10_parity_claim": False,
        "guard": (
            "Compare M4 Max and GB10 only when both rows were collected "
            "with identical comparison_key.workload and "
            "comparison_key.software."
        ),
    }
    matrix_run_metadata = {
        **run_metadata,
        "matched_run": matrix_matched_run,
    }
    return {
        **case_metrics,
        "case_id": case_id,
        "receipt_schema_version": BENCH_RECEIPT_SCHEMA_VERSION,
        "receipt_scope": LOCAL_RECEIPT_SCOPE,
        "local_only": True,
        "gb10_parity_claim": False,
        "profile": profile,
        "route": route,
        **alias_metadata,
        "mlx_version": device.get("mlx"),
        "mlx_lm_version": device.get("mlx_lm"),
        "workload_key": matched_key,
        "software_key": software_key,
        "comparison_key": comparison_key,
        "matched_run_key": matched_key,
        "matched_run_guard": MATCHED_RUN_GUARD,
        "bench_receipt": bench_receipt,
        "profile_hooks": profile_hooks,
        "matched_run": matrix_matched_run,
        "run_metadata": matrix_run_metadata,
    }


def _timing_receipt(metrics: dict[str, Any]) -> dict[str, Any]:
    profile = metrics.get("profile") or {}
    measured = (profile.get("scopes") or {}).get("measured_steps") or {}
    synchronized = measured.get("synchronized")
    if synchronized is None and metrics.get("status") == "ok":
        synchronized = True
    tokens_per_second = metrics.get("tokens_per_second")
    mean_step_time_s = metrics.get("mean_step_time_s")
    median_step_time_s = metrics.get("median_step_time_s")
    step_times_s = list(metrics.get("step_times_s") or [])
    wall_time_s = metrics.get("wall_time_s")
    if wall_time_s is None:
        wall_time_s = mean_step_time_s
    mean_wall_time_s = metrics.get("mean_wall_time_s")
    if mean_wall_time_s is None:
        mean_wall_time_s = wall_time_s
    total_wall_time_s = metrics.get("total_wall_time_s")
    if total_wall_time_s is None and step_times_s:
        total_wall_time_s = sum(step_times_s)
    return {
        "tokens_per_step": metrics.get("tokens_per_step"),
        "warmup_steps": metrics.get("warmup_steps"),
        "measured_steps": metrics.get("measured_steps"),
        "compile": metrics.get("compile"),
        "first_call_time_s": metrics.get("first_call_time_s"),
        "compile_time_s": metrics.get("compile_time_s"),
        "mean_step_time_s": mean_step_time_s,
        "wall_time_s": wall_time_s,
        "mean_wall_time_s": mean_wall_time_s,
        "total_wall_time_s": total_wall_time_s,
        "median_step_time_s": median_step_time_s,
        "tokens_per_second": tokens_per_second,
        "tokens_per_second_or_step_time": (
            tokens_per_second is not None
            or mean_step_time_s is not None
            or median_step_time_s is not None
        ),
        "warmup_step_times_s": list(metrics.get("warmup_step_times_s") or []),
        "step_times_s": step_times_s,
        "synchronized_timing": synchronized,
        "timing_method": (
            "wall-clock timing around MLX train steps with mx.eval outputs and "
            "mx.synchronize before reporting; compile first-call time is separate"
        ),
    }


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    cases = []
    for case_id, profile, route, config in iter_configs(args):
        metrics = dry_run_payload(config) if args.dry_run_json else run_benchmark(config)
        cases.append(
            _augment_metrics(
                case_id=case_id,
                profile=profile,
                route=route,
                metrics=metrics,
            )
        )
    return {
        "schema_version": BENCH_RECEIPT_SCHEMA_VERSION,
        "receipt_schema_version": BENCH_RECEIPT_SCHEMA_VERSION,
        "receipt_scope": LOCAL_RECEIPT_SCOPE,
        "local_only": True,
        "gb10_parity_claim": False,
        "status": "dry_run" if args.dry_run_json else "ok",
        "hardware_label": args.hardware_label,
        "case_count": len(cases),
        "profiles": sorted({case["profile"] for case in cases}),
        "routes": sorted({case["route"] for case in cases}),
        "matched_run_guard": "GB10 parity requires matched rows, not max-throughput cherry-picks.",
        "matched_run_policy": MATRIX_MATCHED_RUN_POLICY,
        "parity_claim_policy": MATRIX_MATCHED_RUN_POLICY,
        "local_only_policy": LOCAL_ONLY_RECEIPT_POLICY,
        "required_receipt_fields": list(REQUIRED_RECEIPT_FIELDS),
        "cases": cases,
    }


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _archive_row(case: dict[str, Any]) -> dict[str, Any]:
    receipt = case["bench_receipt"]
    return {
        "schema_version": BASELINE_RECORD_SCHEMA_VERSION,
        "kind": BASELINE_RECORD_KIND,
        "receipt_scope": LOCAL_RECEIPT_SCOPE,
        "local_only": True,
        "gb10_parity_claim": False,
        "case_id": case["case_id"],
        "status": case["status"],
        "hardware_label": case["hardware_label"],
        "profile": case["profile"],
        "route": case["route"],
        "comparison_key": case["comparison_key"],
        "workload_key": case["workload_key"],
        "software_key": case["software_key"],
        "bench_receipt": receipt,
        "metrics": {
            "tokens_per_second": case.get("tokens_per_second"),
            "mean_step_time_s": case.get("mean_step_time_s"),
            "median_step_time_s": case.get("median_step_time_s"),
            "wall_time_s": case.get("wall_time_s"),
            "total_wall_time_s": case.get("total_wall_time_s"),
            "peak_memory_bytes": case.get("peak_memory_bytes"),
        },
        "guards": {
            "matched_run_guard": MATCHED_RUN_GUARD,
            "parity_claim_policy": LOCAL_BASELINE_POLICY,
            "local_only_policy": LOCAL_ONLY_RECEIPT_POLICY,
            "local_only": True,
            "gb10_parity_claim": False,
        },
    }


def build_baseline_record(
    summary: dict[str, Any],
    *,
    note: str = "",
    recorded_at_utc: str | None = None,
) -> dict[str, Any]:
    recorded_at = recorded_at_utc or _utc_now()
    return {
        "schema_version": BASELINE_RECORD_SCHEMA_VERSION,
        "kind": BASELINE_RECORD_KIND,
        "receipt_scope": LOCAL_RECEIPT_SCOPE,
        "local_only": True,
        "gb10_parity_claim": False,
        "recorded_at_utc": recorded_at,
        "hardware_label": summary["hardware_label"],
        "status": summary["status"],
        "case_count": summary["case_count"],
        "profiles": summary["profiles"],
        "routes": summary["routes"],
        "note": note,
        "source": {
            "script": "scripts/bench_matrix.py",
            "receipt_schema_version": BENCH_RECEIPT_SCHEMA_VERSION,
            "matrix_schema_version": summary["schema_version"],
        },
        "compare_line_contract": {
            "source": "scripts/bench_tiny.py --compare-line",
            "fields": list(COMPARE_LINE_FIELDS),
            "stability": "append-only by explicit migration; do not reorder fields",
        },
        "guards": {
            "matched_run_guard": summary["matched_run_guard"],
            "matched_run_policy": summary["matched_run_policy"],
            "parity_claim_policy": LOCAL_BASELINE_POLICY,
            "local_only_policy": LOCAL_ONLY_RECEIPT_POLICY,
            "local_only": True,
            "gb10_parity_claim": False,
        },
        "rows": [_archive_row(case) for case in summary["cases"]],
    }


def _new_archive(created_at_utc: str) -> dict[str, Any]:
    return {
        "schema_version": BASELINE_ARCHIVE_SCHEMA_VERSION,
        "kind": BASELINE_ARCHIVE_KIND,
        "receipt_scope": LOCAL_RECEIPT_SCOPE,
        "local_only": True,
        "gb10_parity_claim": False,
        "created_at_utc": created_at_utc,
        "updated_at_utc": created_at_utc,
        "records": [],
        "guards": {
            "parity_claim_policy": LOCAL_BASELINE_POLICY,
            "local_only_policy": LOCAL_ONLY_RECEIPT_POLICY,
            "local_only": True,
            "gb10_parity_claim": False,
            "matched_run_guard": MATCHED_RUN_GUARD,
        },
    }


def _validate_local_only_guards(
    obj: dict[str, Any],
    *,
    label: str,
    require_scope: bool = True,
) -> None:
    if require_scope and obj.get("receipt_scope") != LOCAL_RECEIPT_SCOPE:
        raise ValueError(f"{label} receipt_scope must be {LOCAL_RECEIPT_SCOPE!r}")
    if obj.get("local_only") is not True:
        raise ValueError(f"{label} local_only must be true")
    if obj.get("gb10_parity_claim") is not False:
        raise ValueError(f"{label} gb10_parity_claim must be false")


def _validate_existing_archive(archive: dict[str, Any]) -> None:
    _validate_local_only_guards(archive, label="baseline archive")
    guards = archive.get("guards")
    if not isinstance(guards, dict):
        raise ValueError("baseline archive guards must be an object")
    _validate_local_only_guards(guards, label="baseline archive guards", require_scope=False)
    records = archive.get("records")
    if not isinstance(records, list):
        raise ValueError("baseline archive records must be a list")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"baseline archive record {index} must be an object")
        _validate_local_only_guards(record, label=f"baseline archive record {index}")
        record_guards = record.get("guards")
        if not isinstance(record_guards, dict):
            raise ValueError(f"baseline archive record {index} guards must be an object")
        _validate_local_only_guards(
            record_guards,
            label=f"baseline archive record {index} guards",
            require_scope=False,
        )
        rows = record.get("rows")
        if not isinstance(rows, list):
            raise ValueError(f"baseline archive record {index} rows must be a list")
        for row_index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(
                    f"baseline archive record {index} row {row_index} must be an object"
                )
            _validate_local_only_guards(
                row,
                label=f"baseline archive record {index} row {row_index}",
            )
            row_guards = row.get("guards")
            if not isinstance(row_guards, dict):
                raise ValueError(
                    f"baseline archive record {index} row {row_index} guards must be an object"
                )
            _validate_local_only_guards(
                row_guards,
                label=f"baseline archive record {index} row {row_index} guards",
                require_scope=False,
            )
            receipt = row.get("bench_receipt")
            if not isinstance(receipt, dict):
                raise ValueError(
                    f"baseline archive record {index} row {row_index} bench_receipt "
                    "must be an object"
                )
            _validate_local_only_guards(
                receipt,
                label=f"baseline archive record {index} row {row_index} bench_receipt",
            )


def _load_archive(path: Path, *, now_utc: str) -> dict[str, Any]:
    if not path.exists():
        return _new_archive(now_utc)
    with path.open("r", encoding="utf-8") as fh:
        archive = json.load(fh)
    if archive.get("schema_version") != BASELINE_ARCHIVE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported baseline archive schema_version {archive.get('schema_version')!r}"
        )
    if archive.get("kind") != BASELINE_ARCHIVE_KIND:
        raise ValueError(f"unsupported baseline archive kind {archive.get('kind')!r}")
    _validate_existing_archive(archive)
    return archive


def write_baseline_archive(
    path: Path,
    summary: dict[str, Any],
    *,
    note: str = "",
    recorded_at_utc: str | None = None,
) -> dict[str, Any]:
    now_utc = recorded_at_utc or _utc_now()
    archive = _load_archive(path, now_utc=now_utc)
    record = build_baseline_record(summary, note=note, recorded_at_utc=now_utc)
    archive["records"].append(record)
    archive["updated_at_utc"] = now_utc
    archive["guards"] = {
        **(archive.get("guards") or {}),
        "parity_claim_policy": LOCAL_BASELINE_POLICY,
        "local_only_policy": LOCAL_ONLY_RECEIPT_POLICY,
        "local_only": True,
        "gb10_parity_claim": False,
        "matched_run_guard": MATCHED_RUN_GUARD,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(archive, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return archive


def _compact_backend_summary(case: dict[str, Any]) -> str:
    summary = case.get("backend_summary") or {}
    if not summary:
        return "none"
    return ",".join(f"{key}:{summary[key]}" for key in sorted(summary))


def _device_name(case: dict[str, Any]) -> str:
    device = case.get("device") or {}
    info = device.get("mlx_device_info") or {}
    return str(info.get("device_name") or device.get("default_device") or "unknown")


def print_human(summary: dict[str, Any]) -> None:
    print("cppmega.mlx MLX benchmark matrix")
    print(f"status: {summary['status']}")
    print(f"hardware_label: {summary['hardware_label']}")
    print(f"case_count: {summary['case_count']}")
    print(f"receipt_scope: {summary['receipt_scope']}")
    print(f"local_only: {summary['local_only']}")
    print(f"gb10_parity_claim: {summary['gb10_parity_claim']}")
    print(f"local_only_policy: {summary['local_only_policy']}")
    print(f"matched_run_guard: {summary['matched_run_guard']}")
    print(
        "case_id status tokens_per_second mean_step_time_s median_step_time_s "
        "compile_time_s first_call_time_s peak_memory_bytes "
        "profile route model_route route_symbols backend backend_summary device_name "
        "dtype batch_size seq_len compile"
    )
    for case in summary["cases"]:
        route_plan = case.get("route_plan") or {}
        print(
            f"{case['case_id']} {case['status']} {case.get('tokens_per_second')} "
            f"{case.get('mean_step_time_s')} {case.get('median_step_time_s')} "
            f"{case.get('compile_time_s')} {case.get('first_call_time_s')} "
            f"{case.get('peak_memory_bytes')} {case['profile']} {case['route']} "
            f"{case.get('model_route')} {route_plan.get('route_symbols')} "
            f"{case.get('backend')} {_compact_backend_summary(case)} "
            f"{json.dumps(_device_name(case))} {case['dtype']} {case['batch_size']} "
            f"{case['seq_len']} {case['compile']}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.auto_wired_limit and args.wired_limit_bytes is not None:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": "--auto-wired-limit and --wired-limit-bytes are mutually exclusive",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    try:
        summary = run_matrix(args)
        if args.archive_baseline is not None:
            archive = write_baseline_archive(
                args.archive_baseline,
                summary,
                note=args.baseline_note,
            )
            summary["baseline_archive"] = {
                "path": str(args.archive_baseline),
                "schema_version": archive["schema_version"],
                "kind": archive["kind"],
                "record_count": len(archive["records"]),
                "parity_claim_policy": LOCAL_BASELINE_POLICY,
                "local_only_policy": LOCAL_ONLY_RECEIPT_POLICY,
                "receipt_scope": LOCAL_RECEIPT_SCOPE,
                "local_only": True,
                "gb10_parity_claim": False,
            }
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2, sort_keys=True))
        return 2

    if args.jsonl:
        for case in summary["cases"]:
            print(json.dumps(case, sort_keys=True))
    elif args.json or args.dry_run_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_human(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

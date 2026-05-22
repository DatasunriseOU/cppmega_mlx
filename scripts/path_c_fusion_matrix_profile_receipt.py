#!/usr/bin/env python3
"""Emit a Path C fusion matrix/profile receipt from a benchmark matrix report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import m04_train_step  # noqa: E402


DEFAULT_MATRIX_REPORT = ROOT / "reports" / "cppmega_1b_path_matrix.json"
DEFAULT_OUTPUT = ROOT / "reports" / "path_c_fusion_matrix_profile_receipt.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Derive the production Path C fusion matrix/profile receipt from "
            "a 1B dtype/optimizer benchmark matrix report."
        ),
    )
    parser.add_argument("--matrix-report", type=Path, default=DEFAULT_MATRIX_REPORT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--compile-receipt",
        type=Path,
        default=m04_train_step.PATH_C_FUSION_COMPILE_RECEIPT_PATH,
        help="Compile receipt used to infer the production schedule id/name.",
    )
    parser.add_argument(
        "--schedule-id",
        help="Override the schedule id instead of reading the compile receipt.",
    )
    parser.add_argument(
        "--schedule-name",
        help="Override the schedule name instead of reading the compile receipt.",
    )
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _schedule_from_args(args: argparse.Namespace) -> tuple[str, str]:
    if args.schedule_id and args.schedule_name:
        return str(args.schedule_id), str(args.schedule_name)
    payload = m04_train_step.path_c_fusion_payload(
        compile_receipt_path=args.compile_receipt,
    )
    schedule = payload.get("production_schedule", {})
    if not isinstance(schedule, dict):
        raise ValueError("path_c_fusion_payload did not return a production schedule")
    schedule_id = args.schedule_id or schedule.get("schedule_id")
    schedule_name = args.schedule_name or schedule.get("schedule_name")
    if not schedule_id or not schedule_name:
        raise ValueError(
            "schedule id/name are required; pass --schedule-id and --schedule-name "
            "or provide a compile receipt with a production schedule"
        )
    return str(schedule_id), str(schedule_name)


def build_matrix_profile_receipt(
    *,
    matrix_report_path: Path,
    schedule_id: str,
    schedule_name: str,
) -> dict[str, Any]:
    matrix_report = _read_json(matrix_report_path)
    return m04_train_step.path_c_fusion_matrix_profile_receipt_from_report(
        matrix_report,
        schedule_id=schedule_id,
        schedule_name=schedule_name,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    schedule_id, schedule_name = _schedule_from_args(args)
    receipt = build_matrix_profile_receipt(
        matrix_report_path=args.matrix_report,
        schedule_id=schedule_id,
        schedule_name=schedule_name,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.out)
    return 0 if receipt.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())

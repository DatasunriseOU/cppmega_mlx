#!/usr/bin/env python3
"""Archive one benchmark baseline row without running benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, NoReturn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cppmega_mlx.training.baselines import (  # noqa: E402
    BASELINE_ARCHIVE_KIND,
    BASELINE_INDEX_KIND,
    PARITY_EVIDENCE_POLICY,
    BaselineValidationError,
    archive_baseline_row,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and archive one benchmark JSON row using the local baseline "
            "helper. This command does not run benchmarks or make performance "
            "claims."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a JSON object benchmark row, or '-' to read from stdin.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where the baseline row and index.json will be written.",
    )
    return parser


def load_row(input_path: str) -> dict[str, Any]:
    try:
        text = sys.stdin.read() if input_path == "-" else Path(input_path).read_text(encoding="utf-8")
        payload = json.loads(text)
    except OSError as exc:
        raise BaselineValidationError(f"failed to read input JSON: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise BaselineValidationError(f"failed to parse input JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise BaselineValidationError("input JSON must be one benchmark row object")
    return payload


def build_receipt(archived: dict[str, Any]) -> dict[str, Any]:
    entry = archived["entry"]
    return {
        "status": "ok",
        "kind": "cppmega.mlx.benchmark_baseline_archive_receipt",
        "schema_version": 1,
        "archive_kind": BASELINE_ARCHIVE_KIND,
        "index_kind": BASELINE_INDEX_KIND,
        "path": archived["path"],
        "index_path": archived["index_path"],
        "filename": archived["filename"],
        "entry": entry,
        "local_only": entry["local_only"],
        "gb10_parity_claim": entry["gb10_parity_claim"],
        "parity_evidence_policy": PARITY_EVIDENCE_POLICY,
    }


def print_error(message: str) -> None:
    print(
        json.dumps(
            {
                "status": "error",
                "error": message,
                "parity_evidence_policy": PARITY_EVIDENCE_POLICY,
            },
            sort_keys=True,
        ),
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        row = load_row(args.input)
        archived = archive_baseline_row(args.output_dir, row)
    except BaselineValidationError as exc:
        print_error(str(exc))
        return 2

    print(json.dumps(build_receipt(archived), sort_keys=True))
    return 0


def _main() -> NoReturn:
    raise SystemExit(main())


if __name__ == "__main__":
    _main()

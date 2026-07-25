#!/usr/bin/env python3
"""Thin wrapper: run the CI enriched stream via tokenize_ci_enriched.py.

This is a convenience entrypoint equivalent to:
    python scripts/streaming_conveyor.py --streams ci

It calls tokenize_ci_enriched.py with proper args, reading from
outputs/ci_enriched/ and publishing to outputs/reindexed_ci_<ts>_code/<bucket>/.

Usage:
    python scripts/run_ci_conveyor.py
    python scripts/run_ci_conveyor.py --input outputs/ci_enriched/ --seq-lengths 1024,2048,4096
    python scripts/run_ci_conveyor.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_TOKENIZE_CI = _SCRIPT_DIR / "tokenize_ci_enriched.py"
_DEFAULT_INPUT = _REPO_ROOT / "outputs" / "ci_enriched"
_DEFAULT_OUTPUT = _REPO_ROOT / "outputs"

# Use the project venv python if available, else sys.executable.
_VENV_PYTHON = _REPO_ROOT / ".venv" / "bin" / "python"
_PYTHON = str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the CI enriched pipeline: tokenize CI JSONL, route to length "
            "buckets, and pack into zstd parquet.\n\n"
            "Thin wrapper around tokenize_ci_enriched.py with conveyor-compatible "
            "defaults."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", default=str(_DEFAULT_INPUT),
        help="Input directory with CI enriched JSONL files. "
             f"Default: {_DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output", default=str(_DEFAULT_OUTPUT),
        help="Output root for packed parquet. "
             f"Default: {_DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--seq-lengths", default="1024,2048,4096",
        help="Comma-separated sequence length buckets (default: 1024,2048,4096).",
    )
    parser.add_argument(
        "--tokenizer-path", default=None,
        help="Path to tokenizer.json (auto-resolved if not provided).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Tokenization batch size (default: 256).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Process and classify but do not write output files.",
    )
    parser.add_argument(
        "--max-docs", type=int, default=0,
        help="Maximum number of documents to process (0 = all).",
    )
    args = parser.parse_args()

    if not _TOKENIZE_CI.exists():
        print(f"ERROR: {_TOKENIZE_CI} not found", file=sys.stderr)
        return 1

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input directory does not exist: {input_path}", file=sys.stderr)
        return 1

    cmd = [
        _PYTHON, str(_TOKENIZE_CI),
        "--input", str(input_path),
        "--output", str(Path(args.output)),
        "--seq-lengths", args.seq_lengths,
        "--batch-size", str(args.batch_size),
    ]
    if args.tokenizer_path:
        cmd += ["--tokenizer-path", args.tokenizer_path]
    if args.dry_run:
        cmd.append("--dry-run")
    if args.max_docs > 0:
        cmd += ["--max-docs", str(args.max_docs)]

    print(f"[run_ci_conveyor] {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())

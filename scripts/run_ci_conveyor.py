#!/usr/bin/env python3
"""Thin wrapper: run the CI enriched stream via tokenize_ci_enriched.py.

This is a convenience entrypoint equivalent to:
    python scripts/streaming_conveyor.py --streams ci

It calls tokenize_ci_enriched.py with proper args, reading from
outputs/ci_enriched/ and publishing to outputs/reindexed_ci_<ts>_ci/<bucket>/.

Usage:
    python scripts/run_ci_conveyor.py
    python scripts/run_ci_conveyor.py --input outputs/ci_enriched/
    python scripts/run_ci_conveyor.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_TOKENIZE_CI = _SCRIPT_DIR / "tokenize_ci_enriched.py"
_DEFAULT_INPUT = _REPO_ROOT / "outputs" / "ci_enriched"
_DEFAULT_OUTPUT = _REPO_ROOT / "outputs"
_DEFAULT_SEQ_LENGTHS = "1024,2048,4096,8192,16384"
_MANIFEST_SCHEMA = "cppmega_ci_fixed_buckets_manifest_v3"

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
        "--seq-lengths", default=_DEFAULT_SEQ_LENGTHS,
        help=(
            "Comma-separated sequence length buckets "
            f"(production default and requirement: {_DEFAULT_SEQ_LENGTHS})."
        ),
    )
    parser.add_argument(
        "--tokenizer-path", default=None,
        help="Path to tokenizer.json (auto-resolved if not provided).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Bounded tokenization batch size (default: 16).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Process and classify but do not write output files.",
    )
    parser.add_argument(
        "--max-docs", type=int, default=0,
        help="Maximum number of documents to process (0 = all).",
    )
    parser.add_argument(
        "--expected-code-revision",
        default=None,
        help=(
            "Required exact clean cppmega.mlx Git commit for a production run."
        ),
    )
    parser.add_argument(
        "--ci-log-completion-receipt",
        default=None,
        help=(
            "Path to cppmega_ci_log_extraction_v1 receipt. Defaults to "
            "INPUT/ci_logs_enriched.completion.json."
        ),
    )
    args = parser.parse_args()
    if not args.dry_run and args.seq_lengths != _DEFAULT_SEQ_LENGTHS:
        parser.error(
            f"production CI requires --seq-lengths {_DEFAULT_SEQ_LENGTHS}"
        )
    if args.max_docs > 0 and not args.dry_run:
        parser.error("--max-docs is allowed only with --dry-run")
    if not args.dry_run and args.expected_code_revision is None:
        parser.error("--expected-code-revision is required for production")

    if not _TOKENIZE_CI.exists():
        print(f"ERROR: {_TOKENIZE_CI} not found", file=sys.stderr)
        return 1

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input directory does not exist: {input_path}", file=sys.stderr)
        return 1

    run_id = f"{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}-{os.getpid()}"
    cmd = [
        _PYTHON, str(_TOKENIZE_CI),
        "--input", str(input_path),
        "--output", str(Path(args.output)),
        "--seq-lengths", args.seq_lengths,
        "--batch-size", str(args.batch_size),
        "--run-id", run_id,
    ]
    completion_receipt = (
        Path(args.ci_log_completion_receipt)
        if args.ci_log_completion_receipt
        else input_path / "ci_logs_enriched.completion.json"
    )
    if args.tokenizer_path:
        cmd += ["--tokenizer-path", args.tokenizer_path]
    if args.dry_run:
        cmd.extend(["--dry-run", "--allow-empty-buckets"])
    else:
        cmd += [
            "--expected-code-revision",
            args.expected_code_revision,
            "--ci-log-completion-receipt",
            str(completion_receipt),
        ]
    if args.max_docs > 0:
        cmd += ["--max-docs", str(args.max_docs)]

    print(f"[run_ci_conveyor] {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0 or args.dry_run:
        return result.returncode

    output_dir = Path(args.output) / f"reindexed_ci_{run_id}_ci"
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        print(
            f"ERROR: CI tokenizer returned success without {manifest_path}",
            file=sys.stderr,
        )
        return 1
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        print(f"ERROR: invalid CI manifest {manifest_path}: {error}", file=sys.stderr)
        return 1
    counters = manifest.get("counters")
    producer = manifest.get("producer")
    revision = producer.get("code_revision") if isinstance(producer, dict) else None
    source_inventory = manifest.get("source_inventory")
    source_completion = manifest.get("source_completion")
    if (
        manifest.get("schema") != _MANIFEST_SCHEMA
        or manifest.get("kind") != "ci"
        or manifest.get("seq_lengths") != [1024, 2048, 4096, 8192, 16384]
        or manifest.get("verification", {}).get("fixed_width_all_rows") is not True
        or manifest.get("verification", {}).get("unexpected_rejects") != 0
        or manifest.get("verification", {}).get("packing_overflow_docs") != 0
        or not isinstance(counters, dict)
        or counters.get("input_docs") != counters.get("tokenized_docs")
        or counters.get("source_tokens") != counters.get("fragment_tokens")
        or not isinstance(revision, dict)
        or revision.get("schema") != "cppmega_ci_code_revision_v2"
        or revision.get("git_commit") != args.expected_code_revision
        or not isinstance(revision.get("source_tree_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", revision["source_tree_sha256"]) is None
        or not isinstance(source_inventory, list)
        or len(source_inventory) != 2
        or any(not isinstance(entry, dict) for entry in source_inventory)
        or [entry.get("name") for entry in source_inventory if isinstance(entry, dict)]
        != ["ci_logs_enriched.jsonl", "ci_paired_enriched.jsonl"]
        or not isinstance(source_completion, dict)
        or source_completion.get("schema") != "cppmega_ci_log_extraction_v1"
        or source_completion.get("status") != "complete"
        or source_completion.get("unresolved_count") != 0
        or not all(
            isinstance(source_completion.get(name), int)
            and not isinstance(source_completion.get(name), bool)
            and source_completion[name] >= 0
            for name in (
                "unique_job_count",
                "fetched_count",
                "expired_count",
                "too_short_count",
            )
        )
        or source_completion["unique_job_count"]
        != source_completion["fetched_count"]
        + source_completion["expired_count"]
        + source_completion["too_short_count"]
    ):
        print(
            f"ERROR: CI manifest did not satisfy production closure: {manifest_path}",
            file=sys.stderr,
        )
        return 1
    print(f"[run_ci_conveyor] verified manifest {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

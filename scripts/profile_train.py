#!/usr/bin/env python3
"""Profile the packed-row data-preparation path that feeds training.

This profiler measures the throughput and packing efficiency of the whole-document
packer (``pack_documents`` / ``pack_parquet_dataset``) for a chosen ``--seqlen``.
It is intentionally scoped to the data-prep stage owned here; it does NOT import
or drive the model/training runner.

Reported per-stage timings and aggregate packing stats let us see how much wall
time the offline packer costs before a training run and how well rows are filled
(slack / padding) at a given sequence length. It fails loud on bad inputs rather
than degrading silently.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from typing import Any, Iterator

from cppmega_mlx.data.packing import PackingStrategy
from scripts.nanochat_data.pack_enriched_rows import (
    NUM_DOCS_COLUMN,
    SLACK_TOKENS_COLUMN,
    TRAINED_TOKEN_COUNT_COLUMN,
    VALID_TOKEN_COUNT_COLUMN,
    pack_documents,
    read_tokenized_documents,
)


@contextmanager
def _timed(stage: str, sink: dict[str, float]) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        sink[stage] = time.perf_counter() - start


def profile_packing(
    input_path: str | os.PathLike[str],
    *,
    seqlen: int,
    pad_token_id: int = 0,
    strategy: PackingStrategy = "best_fit",
    repeat: int = 1,
) -> dict[str, Any]:
    """Profile read + pack for ``input_path`` at the requested ``seqlen``.

    ``repeat`` re-runs only the pack step (docs are read once) so packer timing
    can be averaged. Returns a JSON-serializable profile dict.
    """

    if seqlen <= 0:
        raise ValueError(f"--seqlen must be > 0, got {seqlen}")
    if repeat < 1:
        raise ValueError(f"--repeat must be >= 1, got {repeat}")

    timings: dict[str, float] = {}
    with _timed("read_documents_s", timings):
        docs = read_tokenized_documents(input_path)
    if not docs:
        raise ValueError(f"no tokenized documents read from {input_path}")

    input_token_total = sum(doc.token_count for doc in docs)
    oversized_docs = sum(1 for doc in docs if doc.token_count > seqlen)

    pack_times: list[float] = []
    packed_rows: list[dict[str, Any]] = []
    overflow: list[dict[str, Any]] = []
    for _ in range(repeat):
        run: dict[str, float] = {}
        with _timed("pack_s", run):
            packed_rows, overflow = pack_documents(
                docs,
                target_length=int(seqlen),
                pad_token_id=int(pad_token_id),
                strategy=strategy,
            )
        pack_times.append(run["pack_s"])
    timings["pack_s_mean"] = sum(pack_times) / len(pack_times)
    timings["pack_s_min"] = min(pack_times)

    valid_tokens = sum(int(row[VALID_TOKEN_COUNT_COLUMN]) for row in packed_rows)
    trained_tokens = sum(int(row[TRAINED_TOKEN_COUNT_COLUMN]) for row in packed_rows)
    slack_tokens = sum(int(row[SLACK_TOKENS_COLUMN]) for row in packed_rows)
    row_capacity = sum(len(row["input_ids"]) for row in packed_rows)
    docs_per_row = [int(row[NUM_DOCS_COLUMN]) for row in packed_rows]

    return {
        "input_path": str(input_path),
        "seqlen": int(seqlen),
        "strategy": strategy,
        "repeat": int(repeat),
        "input_docs": len(docs),
        "input_token_total": int(input_token_total),
        "oversized_docs_kept_whole": int(oversized_docs),
        "packed_rows": len(packed_rows),
        "overflow_docs_recorded": len(overflow),
        "valid_tokens": int(valid_tokens),
        "trained_tokens": int(trained_tokens),
        "slack_tokens": int(slack_tokens),
        "row_capacity_tokens": int(row_capacity),
        "fill_ratio": (valid_tokens / row_capacity) if row_capacity else 0.0,
        "mean_docs_per_row": (
            sum(docs_per_row) / len(docs_per_row) if docs_per_row else 0.0
        ),
        "max_docs_per_row": max(docs_per_row, default=0),
        "tokens_per_s": (
            input_token_total / timings["pack_s_mean"]
            if timings["pack_s_mean"] > 0
            else None
        ),
        "timings_s": timings,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile the packed-row data-prep path for a given seqlen."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input parquet file or directory of per-document tokenized shards.",
    )
    parser.add_argument(
        "--seqlen",
        type=int,
        required=True,
        help="Fixed packed-row length to profile (e.g. 4096 or 8192).",
    )
    parser.add_argument(
        "--pad-token-id",
        type=int,
        default=0,
        help="Token ID used for right padding (default: 0).",
    )
    parser.add_argument(
        "--strategy",
        choices=("best_fit", "sequential"),
        default="best_fit",
        help="Whole-document packing strategy to profile (default: best_fit).",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat the pack step N times and report mean/min timing.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    profile = profile_packing(
        args.input,
        seqlen=args.seqlen,
        pad_token_id=args.pad_token_id,
        strategy=args.strategy,
        repeat=args.repeat,
    )
    print(json.dumps(profile, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["profile_packing"]

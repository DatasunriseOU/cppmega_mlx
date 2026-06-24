#!/usr/bin/env python3
"""Build a fixed-seqlen packed-row parquet dataset from tokenized enriched docs.

Thin, fail-loud CLI front-end over
``scripts.nanochat_data.pack_enriched_rows.pack_parquet_dataset`` that pins the
packed row length to a single ``--seqlen`` (e.g. 4096 or 8192) and enforces the
whole-function packing invariant end to end:

  * Whole documents are concatenated in dependency-first (topological) order and
    every per-doc structure (chunk_starts/ends, call/type edge endpoints,
    changed_chunk_ids/spans) is remapped into block-coordinate space so the
    edges keep pointing at the right tokens/chunks after packing.
  * A document longer than ``--seqlen`` is NEVER split or truncated: it is kept
    whole in its own packed row and also recorded in the overflow sidecar.

This module deliberately adds no new packing logic; it reuses the materializer
so there is exactly one code path that produces packed rows.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow.parquet as pq  # type: ignore[import-not-found]

from cppmega_mlx.data.packing import PackingStrategy
from scripts.nanochat_data.pack_enriched_rows import (
    pack_documents,
    pack_parquet_dataset,
    read_tokenized_documents,
    rows_to_table,
    write_overflow_records,
)


def build_packed_mixed(
    sources: list[tuple[str, str, int]],
    output_path: str | os.PathLike[str],
    *,
    seqlen: int,
    pad_token_id: int = 0,
    strategy: PackingStrategy = "best_fit",
    overflow_output: str | os.PathLike[str] | None = None,
    row_group_size: int = 1024,
) -> dict[str, object]:
    """Pack a token-budgeted MIX of several tokenized-enriched sources.

    ``sources`` is a list of ``(name, input_path, token_budget)`` triples. Each
    source is streamed whole-document until its ``token_budget`` is met (docs are
    never split), then all docs across sources are packed together through the
    single ``pack_documents`` materializer path so mixing changes only WHICH docs
    share a row, not how rows are built. Raises on a non-positive ``seqlen``
    rather than clamping; raises if any source yields zero docs.
    """

    if seqlen <= 0:
        raise ValueError(f"--seqlen must be > 0, got {seqlen}")
    if not sources:
        raise ValueError("at least one --source is required")

    all_docs = []
    signature_to_id: dict[str, int] = {}
    per_source: dict[str, dict[str, int]] = {}
    for name, input_path, token_budget in sources:
        if token_budget <= 0:
            raise ValueError(f"source {name!r} token_budget must be > 0, got {token_budget}")
        docs = read_tokenized_documents(
            input_path,
            token_budget=token_budget,
            start_source_doc_index=len(all_docs),
            signature_to_id=signature_to_id,
        )
        if not docs:
            raise ValueError(f"source {name!r} at {input_path} yielded zero documents")
        toks = sum(doc.token_count for doc in docs)
        per_source[name] = {"docs": len(docs), "tokens": toks}
        all_docs.extend(docs)

    packed_rows, overflow = pack_documents(
        all_docs,
        target_length=int(seqlen),
        pad_token_id=int(pad_token_id),
        strategy=strategy,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        rows_to_table(packed_rows), output, row_group_size=int(row_group_size)
    )
    if overflow_output is not None:
        write_overflow_records(overflow, overflow_output)

    return {
        "input_docs": len(all_docs),
        "packed_rows": len(packed_rows),
        "overflow_docs": len(overflow),
        "per_source": per_source,
    }


def build_packed_seqlen(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    seqlen: int,
    pad_token_id: int = 0,
    strategy: PackingStrategy = "best_fit",
    overflow_output: str | os.PathLike[str] | None = None,
    row_group_size: int = 1024,
) -> dict[str, int]:
    """Pack a tokenized-enriched dataset into fixed ``seqlen`` packed rows.

    ``seqlen`` is the fixed packed-row length (the packer's ``target_length``).
    Documents that exceed ``seqlen`` are emitted whole in their own rows and
    recorded in ``overflow_output`` when provided. Raises on a non-positive
    ``seqlen`` rather than silently clamping.
    """

    if seqlen <= 0:
        raise ValueError(f"--seqlen must be > 0, got {seqlen}")

    return pack_parquet_dataset(
        input_path,
        output_path,
        target_length=int(seqlen),
        pad_token_id=int(pad_token_id),
        strategy=strategy,
        overflow_output=overflow_output,
        row_group_size=int(row_group_size),
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a fixed-seqlen packed-row parquet dataset from tokenized "
            "enriched documents (whole-document, no-truncation packing)."
        )
    )
    parser.add_argument(
        "--input",
        default="",
        help="Single input parquet file or directory of per-document shards.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="NAME=PATH:BUDGET",
        help=(
            "Repeatable mixed source as NAME=PATH:TOKEN_BUDGET. When any --source "
            "is given, sources are token-budgeted, streamed whole-document, and "
            "packed together (mutually exclusive with --input)."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output parquet path for fixed-seqlen packed rows.",
    )
    parser.add_argument(
        "--seqlen",
        type=int,
        required=True,
        help="Fixed packed-row length, typically 4096 or 8192.",
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
        help="Whole-document packing strategy (default: best_fit).",
    )
    parser.add_argument(
        "--overflow-output",
        default="",
        help=(
            "Optional JSONL or parquet path recording oversized docs. Oversized "
            "docs are still emitted whole in their own packed rows."
        ),
    )
    parser.add_argument(
        "--row-group-size",
        type=int,
        default=1024,
        help="Output parquet row group size (default: 1024).",
    )
    return parser


def _parse_source(spec: str) -> tuple[str, str, int]:
    if "=" not in spec:
        raise ValueError(f"--source must be NAME=PATH:BUDGET, got {spec!r}")
    name, rest = spec.split("=", 1)
    if ":" not in rest:
        raise ValueError(f"--source must be NAME=PATH:BUDGET, got {spec!r}")
    path, budget_str = rest.rsplit(":", 1)
    return name.strip(), path.strip(), int(budget_str)


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.source and args.input:
        raise ValueError("--source and --input are mutually exclusive")
    if args.source:
        sources = [_parse_source(spec) for spec in args.source]
        summary: dict[str, object] = build_packed_mixed(
            sources,
            args.output,
            seqlen=args.seqlen,
            pad_token_id=args.pad_token_id,
            strategy=args.strategy,
            overflow_output=args.overflow_output or None,
            row_group_size=args.row_group_size,
        )
    else:
        if not args.input:
            raise ValueError("either --input or at least one --source is required")
        summary = build_packed_seqlen(
            args.input,
            args.output,
            seqlen=args.seqlen,
            pad_token_id=args.pad_token_id,
            strategy=args.strategy,
            overflow_output=args.overflow_output or None,
            row_group_size=args.row_group_size,
        )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["build_packed_mixed", "build_packed_seqlen"]

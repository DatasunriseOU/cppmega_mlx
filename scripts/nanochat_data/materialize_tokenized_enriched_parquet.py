#!/usr/bin/env python3
"""Add token-level enriched columns to an existing enriched parquet dataset.

Reads local parquet shards, preserves existing columns, and writes a sibling
dataset with token_ids plus token-level enriched metadata columns materialized
offline. This is intended for A/B dataloader benchmarking against the legacy
char-level parquet path without round-tripping through JSONL again.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow as pa  # type: ignore[import-not-found]
import pyarrow.parquet as pq  # type: ignore[import-not-found]

from cppmega_mlx.tokenizer.cpp_tokenizer import load_cppmega_tokenizer
from cppmega_mlx.data.nanochat_pipeline.tokenized_enriched import materialize_tokenized_enriched_batch
from cppmega_mlx.data.nanochat_pipeline.tokenized_enriched_schema import TOKENIZED_ENRICHED_COLUMNS


def _decode_json_like(value):
    while isinstance(value, str) and value:
        decoded = json.loads(value)
        if decoded == value:
            break
        value = decoded
    return value


def _list_parquet_files(input_dir: str) -> list[Path]:
    return sorted(
        Path(input_dir).glob("*.parquet"),
        key=lambda p: p.name,
    )


def _table_to_docs(table: pa.Table) -> list[dict]:
    columns = {name: table.column(name).to_pylist() for name in table.column_names}
    num_rows = table.num_rows
    docs = []
    for i in range(num_rows):
        row = {name: values[i] for name, values in columns.items()}
        for json_list_field in ("chunk_boundaries", "call_edges", "type_edges"):
            value = row.get(json_list_field)
            if value is None:
                continue
            value = _decode_json_like(value)
            if isinstance(value, list):
                value = [_decode_json_like(item) for item in value]
            row[json_list_field] = value
        for json_obj_field in ("platform_info", "language_info", "build_info"):
            value = row.get(json_obj_field)
            if value is None:
                continue
            row[json_obj_field] = _decode_json_like(value)
        docs.append(row)
    return docs


def _merge_table_with_tokenized(
    table: pa.Table,
    tokenized_rows: list[dict],
) -> pa.Table:
    base_columns = {name: table.column(name) for name in table.column_names}
    for column_name in TOKENIZED_ENRICHED_COLUMNS:
        base_columns[column_name] = pa.array(
            [row.get(column_name, []) for row in tokenized_rows]
        )
    return pa.table(base_columns)


def _copy_metadata_files(input_dir: str, output_dir: str) -> None:
    for name in ("_COMPLETE",):
        src = Path(input_dir) / name
        if src.exists():
            Path(output_dir, name).write_text(src.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Only process the first N parquet shards (0 = all).",
    )
    parser.add_argument(
        "--tokenizer-path",
        default="",
        help="Optional tokenizer.json path used for token materialization.",
    )
    parser.add_argument(
        "--row-group-size",
        type=int,
        default=1024,
        help="Row group size for rewritten output shards.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    parquet_files = _list_parquet_files(args.input_dir)
    if args.max_files > 0:
        parquet_files = parquet_files[: args.max_files]
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {args.input_dir}")

    tokenizer_path = args.tokenizer_path
    if not tokenizer_path:
        raise ValueError("--tokenizer-path is required for parquet materialization")
    tokenizer = load_cppmega_tokenizer(tokenizer_path)

    for idx, src_path in enumerate(parquet_files, 1):
        dst_path = Path(args.output_dir) / src_path.name
        print(f"[{idx}/{len(parquet_files)}] {src_path.name}")
        table = pq.read_table(src_path)
        docs = _table_to_docs(table)
        tokenized_rows = materialize_tokenized_enriched_batch(docs, tokenizer)
        merged = _merge_table_with_tokenized(table, tokenized_rows)
        pq.write_table(
            merged,
            dst_path,
            row_group_size=args.row_group_size,
            compression="snappy",
        )

    _copy_metadata_files(args.input_dir, args.output_dir)
    print(f"Wrote tokenized parquet dataset to {args.output_dir}")


if __name__ == "__main__":
    main()

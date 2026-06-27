#!/usr/bin/env python3
"""Audit cppmega packed parquet sidecar coverage and shape correctness.

The conveyor emits parquet first, then Megatron sidecar indexed data.  This
script is the fail-closed gate before uploading parquet shards to shared object
storage: it checks every selected parquet shard, validates token-aligned sidecar
lengths and graph/span endpoints, and writes JSON/Markdown receipts.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import traceback
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq


TOKEN_COLUMNS = (
    "input_ids",
    "target_ids",
    "loss_mask",
    "doc_ids",
)

SOURCE_COLUMNS = (
    "source_doc_ids",
    "source_doc_token_lengths",
    "source_platform_ids",
    "source_repo_stable_ids",
    "source_filepath_stable_ids",
    "source_file_local_commit_indices",
)

TOKEN_ALIGNED_FIELDS = (
    "token_platform_ids",
    "token_structure_ids",
    "token_dep_levels",
    "token_ast_depth",
    "token_sibling_index",
    "token_ast_node_type",
    "token_symbol_ids",
    "token_call_targets",
    "token_type_refs",
    "token_def_use",
    "token_change_mask_pre",
    "token_change_mask_post",
    "hunk_id_per_token",
    "edit_op_per_token",
)

CHUNK_ALIGNED_FIELDS = (
    "token_chunk_starts",
    "token_chunk_ends",
    "token_chunk_kinds",
    "token_chunk_dep_levels",
)

LIST_FIELDS = (
    "platform_ids",
    "token_call_edges",
    "token_type_edges",
    "changed_chunk_ids",
    "changed_chunk_spans",
)

ALL_FIELDS = (*TOKEN_COLUMNS, *SOURCE_COLUMNS, *TOKEN_ALIGNED_FIELDS, *CHUNK_ALIGNED_FIELDS, *LIST_FIELDS)


@dataclass
class FieldStats:
    rows_present: int = 0
    rows_nonempty: int = 0
    rows_nonzero: int = 0
    slots_total: int = 0
    slots_nonzero: int = 0
    bad_length_rows: int = 0
    bad_value_rows: int = 0
    missing_files: int = 0

    def add(self, other: "FieldStats") -> None:
        self.rows_present += other.rows_present
        self.rows_nonempty += other.rows_nonempty
        self.rows_nonzero += other.rows_nonzero
        self.slots_total += other.slots_total
        self.slots_nonzero += other.slots_nonzero
        self.bad_length_rows += other.bad_length_rows
        self.bad_value_rows += other.bad_value_rows
        self.missing_files += other.missing_files

    def as_dict(self, rows_total: int, files_total: int) -> dict[str, Any]:
        def pct(num: int, den: int) -> float:
            return round(100.0 * num / den, 6) if den else 0.0

        return {
            "rows_present": self.rows_present,
            "rows_present_pct": pct(self.rows_present, rows_total),
            "rows_nonempty": self.rows_nonempty,
            "rows_nonempty_pct": pct(self.rows_nonempty, rows_total),
            "rows_nonzero": self.rows_nonzero,
            "rows_nonzero_pct": pct(self.rows_nonzero, rows_total),
            "slots_total": self.slots_total,
            "slots_nonzero": self.slots_nonzero,
            "slots_nonzero_pct": pct(self.slots_nonzero, self.slots_total),
            "bad_length_rows": self.bad_length_rows,
            "bad_value_rows": self.bad_value_rows,
            "missing_files": self.missing_files,
            "missing_files_pct": pct(self.missing_files, files_total),
        }


@dataclass
class AuditStats:
    files: int = 0
    rows: int = 0
    capacity_tokens: int = 0
    valid_tokens: int = 0
    trained_tokens: int = 0
    slack_tokens: int = 0
    bad_rows: int = 0
    bad_files: int = 0
    edge_count: dict[str, int] = field(default_factory=lambda: {"token_call_edges": 0, "token_type_edges": 0})
    field_stats: dict[str, FieldStats] = field(default_factory=lambda: {name: FieldStats() for name in ALL_FIELDS})
    errors: list[str] = field(default_factory=list)

    def add(self, other: "AuditStats") -> None:
        self.files += other.files
        self.rows += other.rows
        self.capacity_tokens += other.capacity_tokens
        self.valid_tokens += other.valid_tokens
        self.trained_tokens += other.trained_tokens
        self.slack_tokens += other.slack_tokens
        self.bad_rows += other.bad_rows
        self.bad_files += other.bad_files
        for key, value in other.edge_count.items():
            self.edge_count[key] = self.edge_count.get(key, 0) + value
        for name, stats in other.field_stats.items():
            self.field_stats[name].add(stats)
        self.errors.extend(other.errors)

    def as_dict(self) -> dict[str, Any]:
        return {
            "files": self.files,
            "rows": self.rows,
            "capacity_tokens": self.capacity_tokens,
            "valid_tokens": self.valid_tokens,
            "trained_tokens": self.trained_tokens,
            "slack_tokens": self.slack_tokens,
            "pad_tokens": self.capacity_tokens - self.valid_tokens,
            "pad_pct": round(100.0 * (self.capacity_tokens - self.valid_tokens) / self.capacity_tokens, 6)
            if self.capacity_tokens
            else 0.0,
            "bad_rows": self.bad_rows,
            "bad_files": self.bad_files,
            "edge_count": self.edge_count,
            "fields": {
                name: stats.as_dict(self.rows, self.files)
                for name, stats in sorted(self.field_stats.items())
            },
            "errors": self.errors[:2000],
        }


def _flatten_edge(edge: Any) -> tuple[int | None, int | None]:
    if isinstance(edge, dict):
        return _as_int(edge.get("from")), _as_int(edge.get("to"))
    if isinstance(edge, (list, tuple)) and len(edge) >= 2:
        return _as_int(edge[0]), _as_int(edge[1])
    return None, None


def _as_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _nonzero_count(values: Any, *, hunk: bool = False) -> int:
    if values is None:
        return 0
    out = 0
    for value in values:
        iv = _as_int(value)
        if iv is None:
            continue
        if hunk:
            out += int(iv >= 0)
        else:
            out += int(iv != 0)
    return out


def _len_or_none(values: Any) -> int | None:
    if values is None:
        return None
    try:
        return len(values)
    except TypeError:
        return None


def _list_lengths(column: Any) -> np.ndarray:
    lengths = pc.fill_null(pc.list_value_length(column), 0).combine_chunks()
    return np.asarray(lengths.to_numpy(zero_copy_only=False), dtype=np.int64)


def _column_numpy(column: Any) -> np.ndarray:
    arr = column.combine_chunks()
    return np.asarray(arr.to_numpy(zero_copy_only=False))


def _flat_numpy(column: Any) -> np.ndarray:
    flat = pc.list_flatten(column).combine_chunks()
    if len(flat) == 0:
        return np.asarray([], dtype=np.int64)
    return np.asarray(flat.to_numpy(zero_copy_only=False))


def _rows_with_flat_mask(lengths: np.ndarray, flat_mask: np.ndarray) -> np.ndarray:
    rows = np.zeros(len(lengths), dtype=bool)
    if len(lengths) == 0 or len(flat_mask) == 0:
        return rows
    nonempty = lengths > 0
    if not np.any(nonempty):
        return rows
    starts = np.empty(len(lengths), dtype=np.int64)
    starts[0] = 0
    if len(lengths) > 1:
        np.cumsum(lengths[:-1], out=starts[1:])
    reduced = np.add.reduceat(flat_mask.astype(np.int8), starts[nonempty])
    rows[nonempty] = reduced > 0
    return rows


def _add_numeric_list_stats(
    *,
    stats: AuditStats,
    field: str,
    column: Any,
    expected_lengths: np.ndarray | int | None,
    hunk: bool = False,
    row_bad: np.ndarray | None = None,
) -> np.ndarray:
    fs = stats.field_stats[field]
    lengths = _list_lengths(column)
    fs.rows_present += len(lengths)
    nonempty = lengths > 0
    fs.rows_nonempty += int(np.count_nonzero(nonempty))
    fs.slots_total += int(lengths.sum())

    flat = _flat_numpy(column)
    if len(flat):
        if hunk:
            nz_mask = flat >= 0
        else:
            nz_mask = flat != 0
        fs.slots_nonzero += int(np.count_nonzero(nz_mask))
        fs.rows_nonzero += int(np.count_nonzero(_rows_with_flat_mask(lengths, nz_mask)))

    if expected_lengths is not None:
        if isinstance(expected_lengths, int):
            bad_len = lengths != expected_lengths
        else:
            bad_len = lengths != expected_lengths
        bad_len_count = int(np.count_nonzero(bad_len))
        fs.bad_length_rows += bad_len_count
        if row_bad is not None and bad_len_count:
            row_bad |= bad_len

    return lengths


def _add_generic_list_stats(stats: AuditStats, field: str, column: Any) -> np.ndarray:
    fs = stats.field_stats[field]
    lengths = _list_lengths(column)
    fs.rows_present += len(lengths)
    nonempty = lengths > 0
    count = int(np.count_nonzero(nonempty))
    fs.rows_nonempty += count
    fs.rows_nonzero += count
    slots = int(lengths.sum())
    fs.slots_total += slots
    fs.slots_nonzero += slots
    return lengths


def _count_bad_flat_values(
    *,
    stats: AuditStats,
    field: str,
    column: Any,
    lengths: np.ndarray,
    bad_mask: np.ndarray,
    row_bad: np.ndarray,
) -> None:
    rows = _rows_with_flat_mask(lengths, bad_mask)
    count = int(np.count_nonzero(rows))
    if count:
        stats.field_stats[field].bad_value_rows += count
        row_bad |= rows


def _iter_or_empty(values: Any) -> Any:
    if values is None:
        return ()
    return values


def _edge_stats_and_validate(
    *,
    stats: AuditStats,
    field: str,
    rows: list[Any],
    n_chunks: np.ndarray,
    row_bad: np.ndarray,
) -> None:
    fs = stats.field_stats[field]
    fs.rows_present += len(rows)
    for idx, edges in enumerate(rows):
        edges = _iter_or_empty(edges)
        if len(edges):
            fs.rows_nonempty += 1
            fs.rows_nonzero += 1
        fs.slots_total += len(edges)
        fs.slots_nonzero += len(edges)
        stats.edge_count[field] = stats.edge_count.get(field, 0) + len(edges)
        limit = int(n_chunks[idx]) if idx < len(n_chunks) else 0
        for edge in edges:
            src, dst = _flatten_edge(edge)
            if src is None or dst is None or src < 0 or dst < 0 or src >= limit or dst >= limit:
                row_bad[idx] = True
                fs.bad_value_rows += 1
                break


def _audit_file(path_str: str, kind: str, bucket: str, vocab_size: int | None) -> dict[str, Any]:
    path = Path(path_str)
    stats = AuditStats(files=1)
    try:
        pf = pq.ParquetFile(path)
        top_level_names = set(pf.schema_arrow.names)
        cols = [name for name in ALL_FIELDS if name in top_level_names]
        cols += [
            name
            for name in ("valid_token_count", "trained_token_count", "slack_tokens")
            if name in top_level_names
        ]
        table = pf.read(columns=cols)
    except Exception as exc:
        stats.bad_files += 1
        stats.errors.append(f"{path}: read failed: {type(exc).__name__}: {exc}")
        return {
            "kind": kind,
            "bucket": bucket,
            "path": path_str,
            "stats": stats.as_dict(),
        }

    names = set(cols)
    for name in ALL_FIELDS:
        if name not in names:
            stats.field_stats[name].missing_files += 1

    expected_len = int(bucket) if bucket.isdigit() else None
    stats.rows = table.num_rows
    row_bad = np.zeros(stats.rows, dtype=bool)

    try:
        if "input_ids" not in names:
            stats.bad_files += 1
            stats.bad_rows = stats.rows
            stats.errors.append(f"{path}: missing input_ids")
            return {
                "kind": kind,
                "bucket": bucket,
                "path": path_str,
                "stats": stats.as_dict(),
            }

        input_ids = table.column("input_ids")
        input_lengths = _add_numeric_list_stats(
            stats=stats,
            field="input_ids",
            column=input_ids,
            expected_lengths=expected_len,
            row_bad=row_bad,
        )
        stats.capacity_tokens += int(input_lengths.sum())

        valid = (
            _column_numpy(table.column("valid_token_count")).astype(np.int64)
            if "valid_token_count" in names
            else input_lengths
        )
        trained = (
            _column_numpy(table.column("trained_token_count")).astype(np.int64)
            if "trained_token_count" in names
            else valid
        )
        slack = (
            _column_numpy(table.column("slack_tokens")).astype(np.int64)
            if "slack_tokens" in names
            else np.maximum(0, input_lengths - valid)
        )
        stats.valid_tokens += int(valid.sum())
        stats.trained_tokens += int(trained.sum())
        stats.slack_tokens += int(slack.sum())
        bad_counts = (valid < 0) | (valid > input_lengths) | (trained < 0) | (trained > input_lengths)
        if np.any(bad_counts):
            row_bad |= bad_counts
            stats.errors.append(f"{path}: {int(np.count_nonzero(bad_counts))} rows have invalid valid/trained counts")

        if vocab_size is not None:
            flat_input = _flat_numpy(input_ids)
            if len(flat_input):
                bad_vocab = (flat_input < 0) | (flat_input >= vocab_size)
                _count_bad_flat_values(
                    stats=stats,
                    field="input_ids",
                    column=input_ids,
                    lengths=input_lengths,
                    bad_mask=bad_vocab,
                    row_bad=row_bad,
                )

        for name in ("target_ids", "loss_mask", "doc_ids"):
            if name in names:
                _add_numeric_list_stats(
                    stats=stats,
                    field=name,
                    column=table.column(name),
                    expected_lengths=input_lengths,
                    row_bad=row_bad,
                )

        for name in TOKEN_ALIGNED_FIELDS:
            if name in names:
                _add_numeric_list_stats(
                    stats=stats,
                    field=name,
                    column=table.column(name),
                    expected_lengths=input_lengths,
                    hunk=(name == "hunk_id_per_token"),
                    row_bad=row_bad,
                )

        for name in SOURCE_COLUMNS:
            if name in names:
                _add_generic_list_stats(stats, name, table.column(name))

        n_chunks = (
            _add_numeric_list_stats(
                stats=stats,
                field="token_chunk_starts",
                column=table.column("token_chunk_starts"),
                expected_lengths=None,
            )
            if "token_chunk_starts" in names
            else np.zeros(stats.rows, dtype=np.int64)
        )
        for name in ("token_chunk_ends", "token_chunk_kinds", "token_chunk_dep_levels"):
            if name in names:
                _add_numeric_list_stats(
                    stats=stats,
                    field=name,
                    column=table.column(name),
                    expected_lengths=n_chunks,
                    row_bad=row_bad,
                )

        if "token_chunk_starts" in names:
            starts = _flat_numpy(table.column("token_chunk_starts"))
            if len(starts):
                bad_starts = (starts < 0) | (starts > np.repeat(input_lengths, n_chunks))
                _count_bad_flat_values(
                    stats=stats,
                    field="token_chunk_starts",
                    column=table.column("token_chunk_starts"),
                    lengths=n_chunks,
                    bad_mask=bad_starts,
                    row_bad=row_bad,
                )
        if "token_chunk_ends" in names:
            ends = _flat_numpy(table.column("token_chunk_ends"))
            if len(ends):
                bad_ends = (ends < 0) | (ends > np.repeat(input_lengths, n_chunks))
                _count_bad_flat_values(
                    stats=stats,
                    field="token_chunk_ends",
                    column=table.column("token_chunk_ends"),
                    lengths=n_chunks,
                    bad_mask=bad_ends,
                    row_bad=row_bad,
                )

        if "platform_ids" in names:
            _add_numeric_list_stats(
                stats=stats,
                field="platform_ids",
                column=table.column("platform_ids"),
                expected_lengths=None,
            )

        if "changed_chunk_ids" in names:
            _add_numeric_list_stats(
                stats=stats,
                field="changed_chunk_ids",
                column=table.column("changed_chunk_ids"),
                expected_lengths=None,
            )
            for row_idx, values in enumerate(table.column("changed_chunk_ids").to_pylist()):
                limit = int(n_chunks[row_idx]) if row_idx < len(n_chunks) else 0
                for value in _iter_or_empty(values):
                    idx = _as_int(value)
                    if idx is None or idx < 0 or idx >= limit:
                        row_bad[row_idx] = True
                        stats.field_stats["changed_chunk_ids"].bad_value_rows += 1
                        break

        if "changed_chunk_spans" in names:
            span_rows = table.column("changed_chunk_spans").to_pylist()
            _add_generic_list_stats(stats, "changed_chunk_spans", table.column("changed_chunk_spans"))
            for row_idx, spans in enumerate(span_rows):
                for span in _iter_or_empty(spans):
                    if isinstance(span, dict):
                        start, end = _as_int(span.get("start")), _as_int(span.get("end"))
                    elif isinstance(span, (list, tuple)) and len(span) >= 2:
                        start, end = _as_int(span[0]), _as_int(span[1])
                    else:
                        start = end = None
                    if start is None or end is None or start < 0 or end < start or end > input_lengths[row_idx]:
                        row_bad[row_idx] = True
                        stats.field_stats["changed_chunk_spans"].bad_value_rows += 1
                        break

        for name in ("token_call_edges", "token_type_edges"):
            if name in names:
                _edge_stats_and_validate(
                    stats=stats,
                    field=name,
                    rows=table.column(name).to_pylist(),
                    n_chunks=n_chunks,
                    row_bad=row_bad,
                )

    except Exception as exc:
        stats.bad_files += 1
        stats.errors.append(f"{path}: batch audit failed: {type(exc).__name__}: {exc}")
        stats.errors.extend(traceback.format_exc().splitlines()[-8:])

    stats.bad_rows = int(np.count_nonzero(row_bad))
    return {
        "kind": kind,
        "bucket": bucket,
        "path": path_str,
        "stats": stats.as_dict(),
    }


def _discover(root: Path, kind: str, buckets: set[str] | None) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for path in sorted(root.glob("*/*.parquet")):
        bucket = path.parent.name
        if buckets is not None and bucket not in buckets:
            continue
        out.append((str(path), kind, bucket))
    return out


def _rollup(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = AuditStats()
    by_kind: dict[str, AuditStats] = {}
    by_bucket: dict[str, AuditStats] = {}
    by_kind_bucket: dict[str, AuditStats] = {}
    bad_files: list[str] = []

    for result in results:
        kind = result["kind"]
        bucket = result["bucket"]
        stats = _stats_from_dict(result["stats"])
        total.add(stats)
        by_kind.setdefault(kind, AuditStats()).add(stats)
        by_bucket.setdefault(bucket, AuditStats()).add(stats)
        by_kind_bucket.setdefault(f"{kind}/{bucket}", AuditStats()).add(stats)
        if result["stats"]["bad_files"] or result["stats"]["bad_rows"]:
            bad_files.append(result["path"])

    return {
        "total": total.as_dict(),
        "by_kind": {k: v.as_dict() for k, v in sorted(by_kind.items())},
        "by_bucket": {k: v.as_dict() for k, v in sorted(by_bucket.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0])},
        "by_kind_bucket": {k: v.as_dict() for k, v in sorted(by_kind_bucket.items())},
        "bad_files": bad_files[:10000],
    }


def _stats_from_dict(data: dict[str, Any]) -> AuditStats:
    stats = AuditStats()
    stats.files = data["files"]
    stats.rows = data["rows"]
    stats.capacity_tokens = data["capacity_tokens"]
    stats.valid_tokens = data["valid_tokens"]
    stats.trained_tokens = data["trained_tokens"]
    stats.slack_tokens = data["slack_tokens"]
    stats.bad_rows = data["bad_rows"]
    stats.bad_files = data["bad_files"]
    stats.edge_count = dict(data["edge_count"])
    stats.errors = list(data.get("errors") or [])
    for name, fdata in data["fields"].items():
        fs = FieldStats()
        for key in (
            "rows_present",
            "rows_nonempty",
            "rows_nonzero",
            "slots_total",
            "slots_nonzero",
            "bad_length_rows",
            "bad_value_rows",
            "missing_files",
        ):
            setattr(fs, key, int(fdata[key]))
        stats.field_stats[name] = fs
    return stats


def _write_md(report: dict[str, Any], path: Path) -> None:
    def fmt_int(value: int) -> str:
        return f"{value:,}"

    lines = ["# Sidecar Parquet Audit", ""]
    total = report["total"]
    lines.extend(
        [
            "## Totals",
            "",
            "| metric | value |",
            "|---|---:|",
            f"| files | {fmt_int(total['files'])} |",
            f"| rows | {fmt_int(total['rows'])} |",
            f"| capacity tokens | {fmt_int(total['capacity_tokens'])} |",
            f"| valid tokens | {fmt_int(total['valid_tokens'])} |",
            f"| trained tokens | {fmt_int(total['trained_tokens'])} |",
            f"| pad pct | {total['pad_pct']} |",
            f"| bad files | {fmt_int(total['bad_files'])} |",
            f"| bad rows | {fmt_int(total['bad_rows'])} |",
            "",
            "## By Kind",
            "",
            "| kind | files | rows | valid tokens | capacity tokens | bad rows |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for kind, stats in report["by_kind"].items():
        lines.append(
            f"| {kind} | {fmt_int(stats['files'])} | {fmt_int(stats['rows'])} | "
            f"{fmt_int(stats['valid_tokens'])} | {fmt_int(stats['capacity_tokens'])} | {fmt_int(stats['bad_rows'])} |"
        )
    lines.extend(["", "## By Kind/Bucket", "", "| split | files | rows | valid tokens | capacity tokens | pad pct | bad rows |", "|---|---:|---:|---:|---:|---:|---:|"])
    for key, stats in report["by_kind_bucket"].items():
        lines.append(
            f"| {key} | {fmt_int(stats['files'])} | {fmt_int(stats['rows'])} | "
            f"{fmt_int(stats['valid_tokens'])} | {fmt_int(stats['capacity_tokens'])} | {stats['pad_pct']} | {fmt_int(stats['bad_rows'])} |"
        )
    lines.extend(["", "## Field Coverage And Correctness", "", "| field | rows present % | rows nonempty % | rows nonzero % | slots nonzero % | bad length rows | bad value rows | missing files |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for field, stats in total["fields"].items():
        lines.append(
            f"| `{field}` | {stats['rows_present_pct']} | {stats['rows_nonempty_pct']} | "
            f"{stats['rows_nonzero_pct']} | {stats['slots_nonzero_pct']} | "
            f"{fmt_int(stats['bad_length_rows'])} | {fmt_int(stats['bad_value_rows'])} | {fmt_int(stats['missing_files'])} |"
        )
    if report["bad_files"]:
        lines.extend(["", "## Bad Files", ""])
        lines.extend(f"- `{item}`" for item in report["bad_files"][:200])
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--code-root", type=Path, default=Path("outputs/reindexed"))
    ap.add_argument("--commit-root", type=Path, default=Path("outputs/reindexed_commits"))
    ap.add_argument("--pr-root", type=Path, default=Path("outputs/reindexed_pr"))
    ap.add_argument("--buckets", default="", help="Comma-separated buckets to include, e.g. 1024,2048")
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    ap.add_argument("--vocab-size", type=int, default=65536)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/sidecar_audit"))
    ap.add_argument("--fail-on-bad", action="store_true")
    args = ap.parse_args()

    buckets = {item.strip() for item in args.buckets.split(",") if item.strip()} or None
    files: list[tuple[str, str, str]] = []
    files.extend(_discover(args.code_root, "code", buckets))
    files.extend(_discover(args.commit_root, "commits", buckets))
    files.extend(_discover(args.pr_root, "pr", buckets))
    if not files:
        raise SystemExit("no parquet files selected")

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(_audit_file, path, kind, bucket, args.vocab_size)
            for path, kind, bucket in files
        ]
        for idx, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if idx % 100 == 0 or idx == len(futures):
                print(f"audited {idx}/{len(futures)} parquet files", flush=True)

    report = _rollup(results)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "sidecar_parquet_audit.json").write_text(json.dumps(report, indent=2))
    _write_md(report, args.out_dir / "sidecar_parquet_audit.md")
    print(json.dumps({
        "out_dir": str(args.out_dir),
        "files": report["total"]["files"],
        "valid_tokens": report["total"]["valid_tokens"],
        "capacity_tokens": report["total"]["capacity_tokens"],
        "bad_files": report["total"]["bad_files"],
        "bad_rows": report["total"]["bad_rows"],
    }, indent=2))
    if args.fail_on_bad and (report["total"]["bad_files"] or report["total"]["bad_rows"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

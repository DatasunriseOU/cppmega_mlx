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
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from cppmega_mlx.data.domain_schema import DomainKind, DomainRoleKind, ParseConfidence
from cppmega_mlx.data.tokenizer_contract import DOMAIN_DELIMITER_TOKEN_IDS


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
    "token_domain_ids",
    "token_role_ids",
    "token_entity_ids",
    "token_scope_ids",
    "token_source_doc_ids",
    "token_confidence_ids",
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
    "token_domain_edges",
    "token_build_edges",
    "token_shell_edges",
    "token_diagnostic_edges",
    "token_cross_domain_edges",
    "changed_chunk_ids",
    "changed_chunk_spans",
)

ALL_FIELDS = (*TOKEN_COLUMNS, *SOURCE_COLUMNS, *TOKEN_ALIGNED_FIELDS, *CHUNK_ALIGNED_FIELDS, *LIST_FIELDS)

_DIAGNOSTIC_DOMAIN_IDS = {
    int(DomainKind.COMPILER_DIAGNOSTIC),
    int(DomainKind.BUILD_DIAGNOSTIC),
    int(DomainKind.COMPILER_ERROR),
    int(DomainKind.BUILD_ERROR),
    int(DomainKind.LINKER_ERROR),
    int(DomainKind.TEST_OUTPUT),
    int(DomainKind.TOOL_OUTPUT),
}

_DIAGNOSTIC_DELIMITER_IDS = np.asarray(
    [
        DOMAIN_DELIMITER_TOKEN_IDS[name]
        for name in (
            "COMPILER_DIAGNOSTIC_START",
            "COMPILER_DIAGNOSTIC_END",
            "BUILD_DIAGNOSTIC_START",
            "BUILD_DIAGNOSTIC_END",
            "COMPILER_ERROR_START",
            "COMPILER_ERROR_END",
            "BUILD_ERROR_START",
            "BUILD_ERROR_END",
            "LINKER_ERROR_START",
            "LINKER_ERROR_END",
            "TEST_OUTPUT_START",
            "TEST_OUTPUT_END",
            "TOOL_OUTPUT_START",
            "TOOL_OUTPUT_END",
        )
    ],
    dtype=np.int64,
)


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
    edge_count: dict[str, int] = field(
        default_factory=lambda: {
            "token_call_edges": 0,
            "token_type_edges": 0,
            "token_domain_edges": 0,
            "token_build_edges": 0,
            "token_shell_edges": 0,
            "token_diagnostic_edges": 0,
            "token_cross_domain_edges": 0,
        }
    )
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


def _flatten_edge_triple(edge: Any) -> tuple[int | None, int | None, int | None]:
    if isinstance(edge, dict):
        return (
            _as_int(edge.get("from", edge.get("src"))),
            _as_int(edge.get("to", edge.get("dst"))),
            _as_int(edge.get("kind")),
        )
    if isinstance(edge, (list, tuple)) and len(edge) >= 3:
        return _as_int(edge[0]), _as_int(edge[1]), _as_int(edge[2])
    return None, None, None


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


def _validate_packed_document_boundaries(
    *,
    stats: AuditStats,
    loss_mask_col: Any,
    doc_ids_col: Any,
    source_doc_token_lengths_col: Any,
    input_lengths: np.ndarray,
    valid: np.ndarray,
    row_bad: np.ndarray,
) -> None:
    """Validate boundaries against independent logical-document lengths.

    ``doc_ids`` cannot be its own oracle: an old producer assigned the same ID
    to distinct functions sharing file provenance, and a mask derived from those
    IDs looked internally consistent while still training across documents.
    ``source_doc_token_lengths`` is the independent packed-document contract.
    """
    doc_rows = doc_ids_col.to_pylist()
    mask_rows = loss_mask_col.to_pylist()
    source_length_rows = source_doc_token_lengths_col.to_pylist()
    bad_doc_ids = 0
    bad_masks = 0

    for row_index, (capacity, valid_count) in enumerate(zip(input_lengths, valid)):
        capacity = int(capacity)
        valid_count = int(valid_count)
        raw_lengths = source_length_rows[row_index]
        if raw_lengths is None:
            row_bad[row_index] = True
            stats.errors.append(f"row {row_index}: missing source_doc_token_lengths")
            continue
        logical_lengths = [int(value) for value in raw_lengths]
        if (
            (valid_count > 0 and not logical_lengths)
            or any(length <= 0 for length in logical_lengths)
            or sum(logical_lengths) != valid_count
        ):
            row_bad[row_index] = True
            stats.field_stats["source_doc_token_lengths"].bad_value_rows += 1
            stats.errors.append(
                f"row {row_index}: source_doc_token_lengths={logical_lengths!r} "
                f"do not partition valid_token_count={valid_count}"
            )
            continue

        expected_doc_ids: list[int] = []
        expected_loss_mask: list[int] = []
        for local_doc_id, length in enumerate(logical_lengths, start=1):
            expected_doc_ids.extend([local_doc_id] * length)
            expected_loss_mask.extend([1] * (length - 1))
            expected_loss_mask.append(0)
        pad_doc_id = len(logical_lengths) if logical_lengths else 0
        expected_doc_ids.extend([pad_doc_id] * (capacity - valid_count))
        expected_loss_mask.extend([0] * (capacity - valid_count))

        actual_doc_ids = [int(value) for value in (doc_rows[row_index] or [])]
        actual_loss_mask = [int(value) for value in (mask_rows[row_index] or [])]
        if actual_doc_ids != expected_doc_ids:
            row_bad[row_index] = True
            bad_doc_ids += 1
        if actual_loss_mask != expected_loss_mask:
            row_bad[row_index] = True
            bad_masks += 1

    if bad_doc_ids:
        stats.field_stats["doc_ids"].bad_value_rows += bad_doc_ids
    if bad_masks:
        stats.field_stats["loss_mask"].bad_value_rows += bad_masks


def _validate_target_ids_shift(
    *,
    stats: AuditStats,
    input_ids_col: Any,
    target_ids_col: Any,
    input_lengths: np.ndarray,
    row_bad: np.ndarray,
    pad_token_id: int = 0,
) -> None:
    """Check `target_ids == input_ids[1:] + PAD` for each fixed-width row."""
    rows = len(input_lengths)
    if rows == 0:
        return
    lengths = input_lengths.astype(np.int64)
    n = int(lengths.sum())
    input_flat = _flat_numpy(input_ids_col)
    target_flat = _flat_numpy(target_ids_col)
    if len(input_flat) != n or len(target_flat) != n:
        return

    expected = np.full(n, pad_token_id, dtype=np.asarray(target_flat).dtype)
    nonempty = lengths > 0
    if np.any(nonempty):
        starts = np.zeros(rows, dtype=np.int64)
        if rows > 1:
            np.cumsum(lengths[:-1], out=starts[1:])
        ends = starts + lengths
        for start, end in zip(starts[nonempty], ends[nonempty]):
            if end - start > 1:
                expected[start:end - 1] = input_flat[start + 1:end]
    mismatch = target_flat != expected
    bad_rows = _rows_with_flat_mask(lengths, mismatch)
    count = int(np.count_nonzero(bad_rows))
    if count:
        stats.field_stats["target_ids"].bad_value_rows += count
        row_bad |= bad_rows


def _validate_trained_count_against_loss_mask(
    *,
    stats: AuditStats,
    trained: np.ndarray,
    loss_mask_col: Any,
    input_lengths: np.ndarray,
    row_bad: np.ndarray,
) -> None:
    """Check `trained_token_count` is exactly the positive loss-mask sum."""
    rows = len(input_lengths)
    if rows == 0:
        return
    lengths = input_lengths.astype(np.int64)
    lm_flat = _flat_numpy(loss_mask_col)
    if len(lm_flat) != int(lengths.sum()) or len(trained) != rows:
        return

    expected = np.zeros(rows, dtype=np.int64)
    nonempty = lengths > 0
    if np.any(nonempty):
        starts = np.zeros(rows, dtype=np.int64)
        if rows > 1:
            np.cumsum(lengths[:-1], out=starts[1:])
        reduced = np.add.reduceat((lm_flat > 0).astype(np.int64), starts[nonempty])
        expected[nonempty] = reduced

    mismatch = trained.astype(np.int64) != expected
    count = int(np.count_nonzero(mismatch))
    if count:
        row_bad |= mismatch
        stats.errors.append(
            f"{count} rows have trained_token_count != sum(loss_mask)"
        )


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


def _edge_triple_stats_and_validate(
    *,
    stats: AuditStats,
    field: str,
    rows: list[Any],
    token_lengths: np.ndarray,
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
        limit = int(token_lengths[idx]) if idx < len(token_lengths) else 0
        for edge in edges:
            src, dst, kind = _flatten_edge_triple(edge)
            if (
                src is None
                or dst is None
                or kind is None
                or src < 0
                or dst < 0
                or kind < 0
                or src >= limit
                or dst >= limit
            ):
                row_bad[idx] = True
                fs.bad_value_rows += 1
                break


def _validate_domain_delimiter_sidecars(
    *,
    stats: AuditStats,
    table: Any,
    names: set[str],
    input_ids_col: Any,
    input_lengths: np.ndarray,
    row_bad: np.ndarray,
) -> None:
    flat_input = _flat_numpy(input_ids_col)
    if len(flat_input) == 0:
        return
    delimiter_ids = np.asarray(sorted(DOMAIN_DELIMITER_TOKEN_IDS.values()), dtype=flat_input.dtype)
    has_delimiter_flat = np.isin(flat_input, delimiter_ids)
    if not np.any(has_delimiter_flat):
        return
    rows_with_delimiter = _rows_with_flat_mask(input_lengths, has_delimiter_flat)
    if "token_domain_ids" not in names or "token_role_ids" not in names:
        row_bad |= rows_with_delimiter
        missing = [
            name
            for name in ("token_domain_ids", "token_role_ids")
            if name not in names
        ]
        stats.errors.append(
            f"{int(np.count_nonzero(rows_with_delimiter))} rows contain domain delimiter tokens "
            f"but missing sidecars: {missing}"
        )
        return
    role_flat = _flat_numpy(table.column("token_role_ids"))
    domain_flat = _flat_numpy(table.column("token_domain_ids"))
    if len(role_flat) != len(flat_input) or len(domain_flat) != len(flat_input):
        return
    bad_delimiter_sidecars = has_delimiter_flat & (
        (role_flat != int(DomainRoleKind.DELIMITER)) | (domain_flat == 0)
    )
    count = int(np.count_nonzero(_rows_with_flat_mask(input_lengths, bad_delimiter_sidecars)))
    if count:
        stats.field_stats["token_role_ids"].bad_value_rows += count
        row_bad |= _rows_with_flat_mask(input_lengths, bad_delimiter_sidecars)


def _validate_diagnostic_rows_have_edges_or_raw(
    *,
    stats: AuditStats,
    table: Any,
    names: set[str],
    input_ids_col: Any,
    input_lengths: np.ndarray,
    row_bad: np.ndarray,
) -> None:
    """Fail diagnostic/error rows without parsed edges unless marked RAW.

    Diagnostics are world observations, not comments. A structured diagnostic
    row should carry token_diagnostic_edges. If a parser could not recover
    edges, the row must explicitly say so by marking tokens
    ParseConfidence.RAW; otherwise a silent parser miss would look certified.
    """

    flat_input = _flat_numpy(input_ids_col)
    if len(flat_input) == 0:
        return

    diagnostic_flat = np.isin(flat_input.astype(np.int64), _DIAGNOSTIC_DELIMITER_IDS)
    if "token_domain_ids" in names:
        domain_flat = _flat_numpy(table.column("token_domain_ids"))
        if len(domain_flat) == len(flat_input):
            diagnostic_flat |= np.isin(domain_flat.astype(np.int64), list(_DIAGNOSTIC_DOMAIN_IDS))

    diagnostic_rows = _rows_with_flat_mask(input_lengths, diagnostic_flat)
    if not np.any(diagnostic_rows):
        return

    if "token_diagnostic_edges" in names:
        edge_lengths = _list_lengths(table.column("token_diagnostic_edges"))
        has_edges = edge_lengths > 0
    else:
        edge_lengths = np.zeros(len(input_lengths), dtype=np.int64)
        has_edges = np.zeros(len(input_lengths), dtype=bool)

    if "token_confidence_ids" in names:
        conf_flat = _flat_numpy(table.column("token_confidence_ids"))
        if len(conf_flat) == len(flat_input):
            raw_rows = _rows_with_flat_mask(
                input_lengths,
                conf_flat.astype(np.int64) == int(ParseConfidence.RAW),
            )
        else:
            raw_rows = np.zeros(len(input_lengths), dtype=bool)
    else:
        raw_rows = np.zeros(len(input_lengths), dtype=bool)

    bad = diagnostic_rows & ~has_edges & ~raw_rows
    count = int(np.count_nonzero(bad))
    if count:
        row_bad |= bad
        stats.field_stats["token_diagnostic_edges"].bad_value_rows += count
        stats.errors.append(
            f"{count} diagnostic/error rows have no token_diagnostic_edges "
            "and no explicit ParseConfidence.RAW"
        )


def _audit_table(
    *,
    path: Path,
    table: object,
    names: set[str],
    expected_len: int | None,
    vocab_size: int | None,
) -> AuditStats:
    """Audit one bounded parquet row group and return additive statistics."""

    stats = AuditStats()
    stats.rows = table.num_rows
    row_bad = np.zeros(stats.rows, dtype=bool)

    try:
        if "input_ids" not in names:
            stats.bad_rows = stats.rows
            stats.errors.append(f"{path}: missing input_ids")
            return stats

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

        _validate_domain_delimiter_sidecars(
            stats=stats,
            table=table,
            names=names,
            input_ids_col=input_ids,
            input_lengths=input_lengths,
            row_bad=row_bad,
        )
        _validate_diagnostic_rows_have_edges_or_raw(
            stats=stats,
            table=table,
            names=names,
            input_ids_col=input_ids,
            input_lengths=input_lengths,
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

        # Derive both doc_ids and loss_mask from logical source-document lengths.
        # Never trust doc_ids as the oracle for its own boundary correctness.
        if all(name in names for name in ("loss_mask", "doc_ids", "source_doc_token_lengths")):
            _validate_packed_document_boundaries(
                stats=stats,
                loss_mask_col=table.column("loss_mask"),
                doc_ids_col=table.column("doc_ids"),
                source_doc_token_lengths_col=table.column("source_doc_token_lengths"),
                input_lengths=input_lengths,
                valid=valid,
                row_bad=row_bad,
            )

        if "target_ids" in names:
            _validate_target_ids_shift(
                stats=stats,
                input_ids_col=input_ids,
                target_ids_col=table.column("target_ids"),
                input_lengths=input_lengths,
                row_bad=row_bad,
            )

        if "trained_token_count" in names and "loss_mask" in names:
            _validate_trained_count_against_loss_mask(
                stats=stats,
                trained=trained,
                loss_mask_col=table.column("loss_mask"),
                input_lengths=input_lengths,
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

        if "token_source_doc_ids" in names:
            source_rows = table.column("token_source_doc_ids").to_pylist()
            bad_source_rows = 0
            for row_index, valid_count in enumerate(valid):
                source_ids = [int(value) for value in (source_rows[row_index] or [])]
                prefix = source_ids[: int(valid_count)]
                if len(prefix) != int(valid_count) or any(value <= 0 for value in prefix):
                    row_bad[row_index] = True
                    bad_source_rows += 1
            if bad_source_rows:
                stats.field_stats["token_source_doc_ids"].bad_value_rows += (
                    bad_source_rows
                )
                stats.errors.append(
                    f"{bad_source_rows} rows: token_source_doc_ids must "
                    "be positive within valid_token_count"
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
        if "token_chunk_starts" in names and "token_chunk_ends" in names:
            start_rows = table.column("token_chunk_starts").to_pylist()
            end_rows = table.column("token_chunk_ends").to_pylist()
            for row_idx, (row_starts, row_ends) in enumerate(
                zip(start_rows, end_rows, strict=True)
            ):
                if len(row_starts or []) != len(row_ends or []):
                    continue
                if any(
                    int(start) >= int(end)
                    for start, end in zip(row_starts or [], row_ends or [], strict=True)
                ):
                    row_bad[row_idx] = True
                    stats.field_stats["token_chunk_ends"].bad_value_rows += 1

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

        for name in (
            "token_domain_edges",
            "token_build_edges",
            "token_shell_edges",
            "token_diagnostic_edges",
            "token_cross_domain_edges",
        ):
            if name in names:
                _edge_triple_stats_and_validate(
                    stats=stats,
                    field=name,
                    rows=table.column(name).to_pylist(),
                    token_lengths=input_lengths,
                    row_bad=row_bad,
                )

    except Exception as exc:
        # Fail loud: a crash mid-audit cannot certify a shard. Re-raise with
        # where+what instead of recording partial stats and returning "ok-ish".
        raise RuntimeError(
            f"{path}: audit failed: {type(exc).__name__}: {exc}"
        ) from exc

    stats.bad_rows = int(np.count_nonzero(row_bad))
    return stats


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
    except Exception as exc:
        raise RuntimeError(
            f"{path}: parquet open failed: {type(exc).__name__}: {exc}"
        ) from exc

    names = set(cols)
    for name in ALL_FIELDS:
        if name not in names:
            stats.field_stats[name].missing_files += 1

    if "input_ids" not in names:
        stats.rows = int(pf.metadata.num_rows)
        stats.bad_files = 1
        stats.bad_rows = stats.rows
        stats.errors.append(f"{path}: missing input_ids")
    else:
        expected_len = int(bucket) if bucket.isdigit() else None
        for row_group_idx in range(pf.metadata.num_row_groups):
            try:
                table = pf.read_row_group(row_group_idx, columns=cols)
            except Exception as exc:
                raise RuntimeError(
                    f"{path}#row_group{row_group_idx}: parquet read failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            stats.add(
                _audit_table(
                    path=path,
                    table=table,
                    names=names,
                    expected_len=expected_len,
                    vocab_size=vocab_size,
                )
            )

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
    ap.add_argument(
        "--allow-bad",
        action="store_true",
        help=(
            "Escape hatch: do NOT block on bad files/rows. By default this gate "
            "is FAIL-CLOSED -- any bad file or bad row exits non-zero so the "
            "upload is blocked. Pass this flag to opt out explicitly."
        ),
    )
    ap.add_argument(
        "--fail-on-bad",
        action="store_true",
        help="Deprecated no-op: failing on bad files/rows is now the default.",
    )
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
    has_bad = bool(report["total"]["bad_files"] or report["total"]["bad_rows"])
    if has_bad and not args.allow_bad:
        # FAIL-CLOSED default: block the upload by exiting non-zero.
        print(
            "AUDIT FAILED (fail-closed): "
            f"bad_files={report['total']['bad_files']} "
            f"bad_rows={report['total']['bad_rows']}. "
            "Upload must be blocked. Pass --allow-bad to override explicitly.",
            flush=True,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

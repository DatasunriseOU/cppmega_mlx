#!/usr/bin/env python3
"""Pack tokenized enriched parquet documents into fixed-length training rows.

This is the cppmega-local port of nanochat's offline packed-row materializer.
The input rows must already contain tokenized enriched columns such as
``token_ids`` and token-aligned structure/AST/chunk metadata. This stage does
not parse C++; it only repacks whole enriched documents into dense LM rows.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, SupportsInt

import pyarrow as pa  # type: ignore[import-not-found]
import pyarrow.parquet as pq  # type: ignore[import-not-found]

from cppmega_mlx.data.nanochat_pipeline.packed_rows_schema import (
    CHANGED_CHUNK_IDS_COLUMN,
    CHANGED_CHUNK_SPANS_COLUMN,
    DOC_IDS_COLUMN,
    INPUT_IDS_COLUMN,
    LOSS_MASK_COLUMN,
    NUM_DOCS_COLUMN,
    PACKED_ROWS_CHUNK_METADATA_COLUMNS,
    PACKED_ROWS_DENSE_FALLBACK_FILL_VALUES,
    PACKED_ROWS_TOKEN_METADATA_COLUMNS,
    PACK_ID_COLUMN,
    ROW_PLATFORM_IDS_COLUMN,
    SOURCE_DOC_IDS_COLUMN,
    SOURCE_FILEPATH_STABLE_IDS_COLUMN,
    SOURCE_FILE_LOCAL_COMMIT_INDICES_COLUMN,
    SOURCE_REPO_STABLE_IDS_COLUMN,
    TARGET_IDS_COLUMN,
    TOKEN_PLATFORM_IDS_COLUMN,
    VALID_TOKEN_COUNT_COLUMN,
)
from cppmega_mlx.data.nanochat_pipeline.tokenized_enriched_schema import (
    AUTHOR_TIMESTAMP_COLUMN,
    COMMIT_HASH_COLUMN,
    COMMIT_TIMESTAMP_COLUMN,
    EDIT_OP_PER_TOKEN_COLUMN,
    FILEPATH_COLUMN,
    FILEPATH_STABLE_ID_COLUMN,
    FILE_LOCAL_COMMIT_INDEX_COLUMN,
    HAS_AMBIGUOUS_RECONSTRUCTION_COLUMN,
    HAS_RENAME_AMBIGUITY_COLUMN,
    HUNK_ID_PER_TOKEN_COLUMN,
    IS_MERGE_COMMIT_COLUMN,
    PARENT_COUNT_COLUMN,
    PARENT_HASHES_COLUMN,
    PLATFORM_IDS_COLUMN,
    RAW_COMMIT_CHRONOLOGY_COLUMNS,
    REPO_COLUMN,
    REPO_STABLE_ID_COLUMN,
    TIMESTAMP_COLUMN,
    TOKEN_AST_DEPTH_COLUMN,
    TOKEN_AST_NODE_TYPE_COLUMN,
    TOKEN_CALL_EDGES_COLUMN,
    TOKEN_CALL_TARGETS_COLUMN,
    TOKEN_CHANGE_MASK_POST_COLUMN,
    TOKEN_CHANGE_MASK_PRE_COLUMN,
    TOKEN_CHUNK_DEP_LEVELS_COLUMN,
    TOKEN_CHUNK_ENDS_COLUMN,
    TOKEN_CHUNK_KINDS_COLUMN,
    TOKEN_CHUNK_STARTS_COLUMN,
    TOKEN_DEF_USE_COLUMN,
    TOKEN_DEP_LEVELS_COLUMN,
    TOKEN_IDS_COLUMN,
    TOKEN_SIBLING_INDEX_COLUMN,
    TOKEN_STRUCTURE_IDS_COLUMN,
    TOKEN_SYMBOL_IDS_COLUMN,
    TOKEN_TYPE_EDGES_COLUMN,
    TOKEN_TYPE_REFS_COLUMN,
)
from cppmega_mlx.data.nanochat_pipeline.platform_vocab import MAX_PLATFORM_IDS
from cppmega_mlx.data.packing import PackingStrategy

TRAINED_TOKEN_COUNT_COLUMN = "trained_token_count"
SLACK_TOKENS_COLUMN = "slack_tokens"
SOURCE_DOC_INDICES_COLUMN = SOURCE_DOC_IDS_COLUMN
SOURCE_DOC_TOKEN_LENGTHS_COLUMN = "source_doc_token_lengths"
SOURCE_PLATFORM_IDS_COLUMN = "source_platform_ids"

PACKED_TOKEN_METADATA_COLUMNS = PACKED_ROWS_TOKEN_METADATA_COLUMNS
PACKED_CHUNK_METADATA_COLUMNS = PACKED_ROWS_CHUNK_METADATA_COLUMNS
PACKED_ROW_PROVENANCE_COLUMNS = tuple(
    column for column in RAW_COMMIT_CHRONOLOGY_COLUMNS if column != TIMESTAMP_COLUMN
) + (TIMESTAMP_COLUMN,)

CHANGED_CHUNK_SPAN_STRUCT = pa.struct(
    [
        pa.field("start", pa.uint32()),
        pa.field("end", pa.uint32()),
    ]
)

EDGE_STRUCT = pa.struct(
    [
        pa.field("from", pa.uint16()),
        pa.field("to", pa.uint16()),
    ]
)

PACKED_ROW_SCHEMA = pa.schema(
    [
        pa.field(PACK_ID_COLUMN, pa.int64()),
        pa.field(VALID_TOKEN_COUNT_COLUMN, pa.int32()),
        pa.field(TRAINED_TOKEN_COUNT_COLUMN, pa.int32()),
        pa.field(NUM_DOCS_COLUMN, pa.int32()),
        pa.field(SLACK_TOKENS_COLUMN, pa.int32()),
        pa.field(INPUT_IDS_COLUMN, pa.list_(pa.uint32())),
        pa.field(TARGET_IDS_COLUMN, pa.list_(pa.uint32())),
        pa.field(LOSS_MASK_COLUMN, pa.list_(pa.uint8())),
        pa.field(DOC_IDS_COLUMN, pa.list_(pa.uint32())),
        pa.field(SOURCE_DOC_INDICES_COLUMN, pa.list_(pa.int64())),
        pa.field(SOURCE_DOC_TOKEN_LENGTHS_COLUMN, pa.list_(pa.int32())),
        pa.field(SOURCE_PLATFORM_IDS_COLUMN, pa.list_(pa.list_(pa.uint16()))),
        pa.field(SOURCE_REPO_STABLE_IDS_COLUMN, pa.list_(pa.string())),
        pa.field(SOURCE_FILEPATH_STABLE_IDS_COLUMN, pa.list_(pa.string())),
        pa.field(SOURCE_FILE_LOCAL_COMMIT_INDICES_COLUMN, pa.list_(pa.int32())),
        pa.field(ROW_PLATFORM_IDS_COLUMN, pa.list_(pa.uint16())),
        pa.field(REPO_COLUMN, pa.string()),
        pa.field(FILEPATH_COLUMN, pa.string()),
        pa.field(COMMIT_HASH_COLUMN, pa.string()),
        pa.field(TIMESTAMP_COLUMN, pa.string()),
        pa.field(PARENT_HASHES_COLUMN, pa.list_(pa.string())),
        pa.field(PARENT_COUNT_COLUMN, pa.int32()),
        pa.field(IS_MERGE_COMMIT_COLUMN, pa.bool_()),
        pa.field(AUTHOR_TIMESTAMP_COLUMN, pa.string()),
        pa.field(COMMIT_TIMESTAMP_COLUMN, pa.string()),
        pa.field(REPO_STABLE_ID_COLUMN, pa.string()),
        pa.field(FILEPATH_STABLE_ID_COLUMN, pa.string()),
        pa.field(FILE_LOCAL_COMMIT_INDEX_COLUMN, pa.int32()),
        pa.field(HAS_AMBIGUOUS_RECONSTRUCTION_COLUMN, pa.bool_()),
        pa.field(HAS_RENAME_AMBIGUITY_COLUMN, pa.bool_()),
        pa.field(TOKEN_PLATFORM_IDS_COLUMN, pa.list_(pa.uint16())),
        pa.field(TOKEN_STRUCTURE_IDS_COLUMN, pa.list_(pa.uint8())),
        pa.field(TOKEN_DEP_LEVELS_COLUMN, pa.list_(pa.uint16())),
        pa.field(TOKEN_AST_DEPTH_COLUMN, pa.list_(pa.int32())),
        pa.field(TOKEN_SIBLING_INDEX_COLUMN, pa.list_(pa.int32())),
        pa.field(TOKEN_AST_NODE_TYPE_COLUMN, pa.list_(pa.int32())),
        pa.field(TOKEN_SYMBOL_IDS_COLUMN, pa.list_(pa.uint32())),
        pa.field(TOKEN_CALL_TARGETS_COLUMN, pa.list_(pa.uint32())),
        pa.field(TOKEN_TYPE_REFS_COLUMN, pa.list_(pa.uint32())),
        pa.field(TOKEN_DEF_USE_COLUMN, pa.list_(pa.uint8())),
        pa.field(TOKEN_CHANGE_MASK_PRE_COLUMN, pa.list_(pa.uint8())),
        pa.field(TOKEN_CHANGE_MASK_POST_COLUMN, pa.list_(pa.uint8())),
        pa.field(HUNK_ID_PER_TOKEN_COLUMN, pa.list_(pa.int32())),
        pa.field(EDIT_OP_PER_TOKEN_COLUMN, pa.list_(pa.uint8())),
        pa.field(TOKEN_CHUNK_STARTS_COLUMN, pa.list_(pa.uint32())),
        pa.field(TOKEN_CHUNK_ENDS_COLUMN, pa.list_(pa.uint32())),
        pa.field(TOKEN_CHUNK_KINDS_COLUMN, pa.list_(pa.uint8())),
        pa.field(TOKEN_CHUNK_DEP_LEVELS_COLUMN, pa.list_(pa.uint16())),
        pa.field(TOKEN_CALL_EDGES_COLUMN, pa.list_(EDGE_STRUCT)),
        pa.field(TOKEN_TYPE_EDGES_COLUMN, pa.list_(EDGE_STRUCT)),
        pa.field(CHANGED_CHUNK_IDS_COLUMN, pa.list_(pa.uint32())),
        pa.field(CHANGED_CHUNK_SPANS_COLUMN, pa.list_(CHANGED_CHUNK_SPAN_STRUCT)),
    ]
)


@dataclass(frozen=True)
class NormalizedDoc:
    """A normalized per-document row ready for offline packing."""

    source_doc_index: int
    token_ids: list[int]
    token_meta: dict[str, list[int]]
    chunk_starts: list[int]
    chunk_ends: list[int]
    chunk_kinds: list[int]
    chunk_dep_levels: list[int]
    call_edges: list[dict[str, int]]
    type_edges: list[dict[str, int]]
    platform_ids: list[int]
    changed_chunk_ids: list[int]
    changed_chunk_spans: list[tuple[int, int]]
    chronology: dict[str, Any]

    @property
    def token_count(self) -> int:
        return len(self.token_ids)

    def to_overflow_record(self) -> dict[str, Any]:
        record = {
            "source_doc_index": int(self.source_doc_index),
            "token_count": int(self.token_count),
            TOKEN_IDS_COLUMN: list(self.token_ids),
            PLATFORM_IDS_COLUMN: list(self.platform_ids),
            TOKEN_CHUNK_STARTS_COLUMN: list(self.chunk_starts),
            TOKEN_CHUNK_ENDS_COLUMN: list(self.chunk_ends),
            TOKEN_CHUNK_KINDS_COLUMN: list(self.chunk_kinds),
            TOKEN_CHUNK_DEP_LEVELS_COLUMN: list(self.chunk_dep_levels),
            TOKEN_CALL_EDGES_COLUMN: [dict(edge) for edge in self.call_edges],
            TOKEN_TYPE_EDGES_COLUMN: [dict(edge) for edge in self.type_edges],
            CHANGED_CHUNK_IDS_COLUMN: list(self.changed_chunk_ids),
            CHANGED_CHUNK_SPANS_COLUMN: [
                {"start": int(start), "end": int(end)}
                for start, end in self.changed_chunk_spans
            ],
        }
        record.update(self.chronology)
        for column, values in self.token_meta.items():
            record[column] = list(values)
        return record


@dataclass
class PackBin:
    """One fixed-length output row under construction."""

    docs: list[NormalizedDoc] = field(default_factory=list)
    used_tokens: int = 0

    def can_fit(self, token_count: int, target_length: int) -> bool:
        return self.used_tokens + token_count <= target_length

    def add(self, doc: NormalizedDoc) -> None:
        self.docs.append(doc)
        self.used_tokens += doc.token_count


def _doc_has_temporal_chronology(doc: NormalizedDoc) -> bool:
    return any(
        doc.chronology.get(column) not in (None, "")
        for column in (
            REPO_COLUMN,
            FILEPATH_COLUMN,
            COMMIT_HASH_COLUMN,
            TIMESTAMP_COLUMN,
            REPO_STABLE_ID_COLUMN,
            FILEPATH_STABLE_ID_COLUMN,
            FILE_LOCAL_COMMIT_INDEX_COLUMN,
        )
    )


def _doc_commit_window_key(doc: NormalizedDoc) -> tuple[str, str, int] | None:
    repo_stable_id = doc.chronology.get(REPO_STABLE_ID_COLUMN)
    filepath_stable_id = doc.chronology.get(FILEPATH_STABLE_ID_COLUMN)
    file_local_commit_index = doc.chronology.get(FILE_LOCAL_COMMIT_INDEX_COLUMN)
    if repo_stable_id in (None, "") or filepath_stable_id in (None, ""):
        return None
    if file_local_commit_index is None:
        return None
    return (
        str(repo_stable_id),
        str(filepath_stable_id),
        int(file_local_commit_index),
    )


def _docs_form_strict_commit_window(docs: list[NormalizedDoc]) -> bool:
    if len(docs) <= 1:
        return True
    keys = [_doc_commit_window_key(doc) for doc in docs]
    if any(key is None for key in keys):
        return False
    repo_file_pairs = {(key[0], key[1]) for key in keys if key is not None}
    if len(repo_file_pairs) != 1:
        return False
    indices = sorted(key[2] for key in keys if key is not None)
    if len(set(indices)) != len(indices):
        return False
    expected = list(range(indices[0], indices[0] + len(indices)))
    return indices == expected


def _pack_bin_accepts_doc(bin_docs: list[NormalizedDoc], candidate: NormalizedDoc) -> bool:
    if not bin_docs:
        return True
    existing_has_temporal = any(_doc_has_temporal_chronology(doc) for doc in bin_docs)
    candidate_has_temporal = _doc_has_temporal_chronology(candidate)
    if not existing_has_temporal and not candidate_has_temporal:
        return True
    return _docs_form_strict_commit_window([*bin_docs, candidate])


def _order_docs_for_row(docs: list[NormalizedDoc]) -> list[NormalizedDoc]:
    if len(docs) > 1 and _docs_form_strict_commit_window(docs):
        return sorted(
            docs,
            key=lambda doc: (
                _coerce_optional_int(doc.chronology.get(FILE_LOCAL_COMMIT_INDEX_COLUMN))
                or 0,
                doc.source_doc_index,
            ),
        )
    return sorted(docs, key=lambda doc: doc.source_doc_index)


def _shared_chronology_for_docs(docs: list[NormalizedDoc]) -> dict[str, Any]:
    shared: dict[str, Any] = {}
    for column in PACKED_ROW_PROVENANCE_COLUMNS:
        values = [doc.chronology.get(column) for doc in docs]
        first = values[0] if values else None
        shared[column] = first if all(value == first for value in values) else None
    return shared


def _merged_platform_ids_for_docs(docs: list[NormalizedDoc]) -> list[int]:
    ids = sorted(
        {
            int(platform_id)
            for doc in docs
            for platform_id in doc.platform_ids
            if int(platform_id) > 0
        }
    )
    if len(ids) > MAX_PLATFORM_IDS:
        raise ValueError(
            f"packed row platform_ids has {len(ids)} unique IDs; "
            f"MAX_PLATFORM_IDS={MAX_PLATFORM_IDS}"
        )
    return ids


def _coerce_optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes, bytearray, int, bool)):
        return int(value)
    if isinstance(value, float):
        return int(value)
    if isinstance(value, SupportsInt):
        return int(value)
    return None


def _as_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    return [int(item) for item in value]


def _normalize_edge_list(value: Any) -> list[dict[str, int]]:
    edges: list[dict[str, int]] = []
    for item in value or []:
        if isinstance(item, dict):
            src = int(item.get("from", 0))
            dst = int(item.get("to", 0))
        else:
            src = int(item[0])
            dst = int(item[1])
        edges.append({"from": src, "to": dst})
    return edges


def _normalize_changed_chunk_spans(value: Any) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for item in value or []:
        if isinstance(item, dict):
            start = int(item.get("start", 0))
            end = int(item.get("end", 0))
        else:
            start = int(item[0])
            end = int(item[1])
        spans.append((start, end))
    return spans


def _validate_chunk_graph_references(
    *,
    source_doc_index: int,
    chunk_count: int,
    call_edges: list[dict[str, int]],
    type_edges: list[dict[str, int]],
    changed_chunk_ids: list[int],
) -> None:
    has_graph_metadata = bool(call_edges or type_edges or changed_chunk_ids)
    if has_graph_metadata and chunk_count == 0:
        raise ValueError(
            "graph/chunk metadata requires non-empty token_chunk_* layout "
            f"for source_doc_index={source_doc_index}"
        )

    for column, edges in (
        (TOKEN_CALL_EDGES_COLUMN, call_edges),
        (TOKEN_TYPE_EDGES_COLUMN, type_edges),
    ):
        for edge in edges:
            src = int(edge["from"])
            dst = int(edge["to"])
            if not (0 <= src < chunk_count and 0 <= dst < chunk_count):
                raise ValueError(
                    f"{column} edge out of range for source_doc_index={source_doc_index}: "
                    f"got ({src}, {dst}) with chunk_count={chunk_count}"
                )

    for chunk_id in changed_chunk_ids:
        if not (0 <= int(chunk_id) < chunk_count):
            raise ValueError(
                f"{CHANGED_CHUNK_IDS_COLUMN} out of range for "
                f"source_doc_index={source_doc_index}: got {chunk_id} "
                f"with chunk_count={chunk_count}"
            )


def _normalize_chronology(record: dict[str, Any]) -> dict[str, Any]:
    chronology: dict[str, Any] = {
        REPO_COLUMN: record.get(REPO_COLUMN),
        FILEPATH_COLUMN: record.get(FILEPATH_COLUMN),
        COMMIT_HASH_COLUMN: record.get(COMMIT_HASH_COLUMN),
        TIMESTAMP_COLUMN: record.get(TIMESTAMP_COLUMN),
        PARENT_HASHES_COLUMN: [
            str(item) for item in (record.get(PARENT_HASHES_COLUMN) or [])
        ],
        AUTHOR_TIMESTAMP_COLUMN: record.get(AUTHOR_TIMESTAMP_COLUMN),
        COMMIT_TIMESTAMP_COLUMN: record.get(COMMIT_TIMESTAMP_COLUMN),
        REPO_STABLE_ID_COLUMN: record.get(REPO_STABLE_ID_COLUMN),
        FILEPATH_STABLE_ID_COLUMN: record.get(FILEPATH_STABLE_ID_COLUMN),
    }
    parent_count = record.get(PARENT_COUNT_COLUMN)
    if parent_count is None:
        parent_count = len(chronology[PARENT_HASHES_COLUMN])
    chronology[PARENT_COUNT_COLUMN] = int(parent_count)
    chronology[IS_MERGE_COMMIT_COLUMN] = bool(
        record.get(IS_MERGE_COMMIT_COLUMN, chronology[PARENT_COUNT_COLUMN] > 1)
    )
    file_local_index = record.get(FILE_LOCAL_COMMIT_INDEX_COLUMN)
    chronology[FILE_LOCAL_COMMIT_INDEX_COLUMN] = (
        int(file_local_index) if file_local_index is not None else None
    )
    chronology[HAS_AMBIGUOUS_RECONSTRUCTION_COLUMN] = bool(
        record.get(HAS_AMBIGUOUS_RECONSTRUCTION_COLUMN, False)
    )
    chronology[HAS_RENAME_AMBIGUITY_COLUMN] = bool(
        record.get(HAS_RENAME_AMBIGUITY_COLUMN, False)
    )
    return chronology


def _normalize_token_meta(
    record: dict[str, Any],
    *,
    token_count: int,
    source_doc_index: int,
) -> dict[str, list[int]]:
    token_meta: dict[str, list[int]] = {}
    for column in PACKED_TOKEN_METADATA_COLUMNS:
        values = _as_int_list(record.get(column))
        if values and len(values) != token_count:
            raise ValueError(
                f"{column} length mismatch for source_doc_index={source_doc_index}: "
                f"expected {token_count}, got {len(values)}"
            )
        token_meta[column] = values
    return token_meta


def _normalize_chunk_meta(
    record: dict[str, Any],
    *,
    source_doc_index: int,
    token_count: int,
) -> tuple[
    list[int],
    list[int],
    list[int],
    list[int],
    list[dict[str, int]],
    list[dict[str, int]],
    list[int],
    list[tuple[int, int]],
]:
    starts = _as_int_list(record.get(TOKEN_CHUNK_STARTS_COLUMN))
    ends = _as_int_list(record.get(TOKEN_CHUNK_ENDS_COLUMN))
    kinds = _as_int_list(record.get(TOKEN_CHUNK_KINDS_COLUMN))
    dep_levels = _as_int_list(record.get(TOKEN_CHUNK_DEP_LEVELS_COLUMN))

    nonempty = [values for values in (starts, ends, kinds, dep_levels) if values]
    if nonempty:
        expected = len(nonempty[0])
        for column_name, values in (
            (TOKEN_CHUNK_STARTS_COLUMN, starts),
            (TOKEN_CHUNK_ENDS_COLUMN, ends),
            (TOKEN_CHUNK_KINDS_COLUMN, kinds),
            (TOKEN_CHUNK_DEP_LEVELS_COLUMN, dep_levels),
        ):
            if len(values) != expected:
                raise ValueError(
                    f"{column_name} length mismatch for source_doc_index={source_doc_index}: "
                    f"expected {expected}, got {len(values)}"
                )
        for start, end in zip(starts, ends):
            if not (0 <= start < end <= token_count):
                raise ValueError(
                    f"chunk span out of range for source_doc_index={source_doc_index}: "
                    f"got ({start}, {end}) with token_count={token_count}"
                )

    call_edges = _normalize_edge_list(record.get(TOKEN_CALL_EDGES_COLUMN))
    type_edges = _normalize_edge_list(record.get(TOKEN_TYPE_EDGES_COLUMN))
    changed_chunk_ids = _as_int_list(record.get(CHANGED_CHUNK_IDS_COLUMN))
    changed_chunk_spans = _normalize_changed_chunk_spans(
        record.get(CHANGED_CHUNK_SPANS_COLUMN)
    )
    if len(changed_chunk_ids) != len(changed_chunk_spans):
        raise ValueError(
            f"{CHANGED_CHUNK_IDS_COLUMN}/{CHANGED_CHUNK_SPANS_COLUMN} length mismatch "
            f"for source_doc_index={source_doc_index}: expected equal lengths, got "
            f"{len(changed_chunk_ids)} and {len(changed_chunk_spans)}"
        )
    _validate_chunk_graph_references(
        source_doc_index=source_doc_index,
        chunk_count=len(starts),
        call_edges=call_edges,
        type_edges=type_edges,
        changed_chunk_ids=changed_chunk_ids,
    )
    for start, end in changed_chunk_spans:
        if not (0 <= start < end <= token_count):
            raise ValueError(
                f"{CHANGED_CHUNK_SPANS_COLUMN} out of range for source_doc_index={source_doc_index}: "
                f"got ({start}, {end}) with token_count={token_count}"
            )
    return (
        starts,
        ends,
        kinds,
        dep_levels,
        call_edges,
        type_edges,
        changed_chunk_ids,
        changed_chunk_spans,
    )


def normalize_document_record(
    record: dict[str, Any],
    *,
    source_doc_index: int,
) -> NormalizedDoc:
    """Validate and normalize one input parquet record."""

    token_ids = _as_int_list(record.get(TOKEN_IDS_COLUMN))
    if not token_ids:
        raise ValueError(
            f"Missing or empty {TOKEN_IDS_COLUMN} for source_doc_index={source_doc_index}"
        )

    token_meta = _normalize_token_meta(
        record,
        token_count=len(token_ids),
        source_doc_index=source_doc_index,
    )
    (
        chunk_starts,
        chunk_ends,
        chunk_kinds,
        chunk_dep_levels,
        call_edges,
        type_edges,
        changed_chunk_ids,
        changed_chunk_spans,
    ) = _normalize_chunk_meta(
        record,
        source_doc_index=source_doc_index,
        token_count=len(token_ids),
    )

    return NormalizedDoc(
        source_doc_index=int(source_doc_index),
        token_ids=token_ids,
        token_meta=token_meta,
        chunk_starts=chunk_starts,
        chunk_ends=chunk_ends,
        chunk_kinds=chunk_kinds,
        chunk_dep_levels=chunk_dep_levels,
        call_edges=call_edges,
        type_edges=type_edges,
        platform_ids=_as_int_list(record.get(PLATFORM_IDS_COLUMN)),
        changed_chunk_ids=changed_chunk_ids,
        changed_chunk_spans=changed_chunk_spans,
        chronology=_normalize_chronology(record),
    )


def _list_input_files(input_path: str | os.PathLike[str]) -> list[Path]:
    path = Path(input_path)
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(
            child for child in path.iterdir() if child.is_file() and child.suffix == ".parquet"
        )
        if not files:
            raise FileNotFoundError(f"No parquet files found under {path}")
        return files
    raise FileNotFoundError(f"Input path does not exist: {path}")


def read_tokenized_documents(input_path: str | os.PathLike[str]) -> list[NormalizedDoc]:
    """Read tokenized per-document parquet rows from a file or directory."""

    docs: list[NormalizedDoc] = []
    source_doc_index = 0
    for path in _list_input_files(input_path):
        parquet_file = pq.ParquetFile(path)
        available = set(parquet_file.schema_arrow.names)
        if TOKEN_IDS_COLUMN not in available:
            raise ValueError(f"{path} is missing required column {TOKEN_IDS_COLUMN}")

        selected_columns = [
            column
            for column in (
                TOKEN_IDS_COLUMN,
                PLATFORM_IDS_COLUMN,
                *RAW_COMMIT_CHRONOLOGY_COLUMNS,
                *PACKED_TOKEN_METADATA_COLUMNS,
                *PACKED_CHUNK_METADATA_COLUMNS,
            )
            if column in available
        ]

        for batch in parquet_file.iter_batches(columns=selected_columns, batch_size=1024):
            for record in batch.to_pylist():
                docs.append(
                    normalize_document_record(
                        record,
                        source_doc_index=source_doc_index,
                    )
                )
                source_doc_index += 1
    return docs


def _pack_docs_best_fit(
    docs: list[NormalizedDoc],
    *,
    target_length: int,
) -> tuple[list[PackBin], list[dict[str, Any]]]:
    remaining = sorted(docs, key=lambda doc: doc.source_doc_index)
    bins: list[PackBin] = []
    overflow: list[dict[str, Any]] = []
    while remaining:
        current = PackBin()
        next_remaining: list[NormalizedDoc] = []
        for doc in remaining:
            if doc.token_count > target_length:
                overflow.append(doc.to_overflow_record())
            else:
                next_remaining.append(doc)
        remaining = next_remaining
        if not remaining:
            break

        while remaining:
            capacity = target_length - current.used_tokens
            candidate_pos: int | None = None
            candidate_len = -1
            candidate_source = 0
            for pos, doc in enumerate(remaining):
                if not current.can_fit(doc.token_count, target_length):
                    continue
                if not _pack_bin_accepts_doc(current.docs, doc):
                    continue
                if doc.token_count > candidate_len or (
                    doc.token_count == candidate_len
                    and doc.source_doc_index < candidate_source
                ):
                    candidate_pos = pos
                    candidate_len = doc.token_count
                    candidate_source = doc.source_doc_index
            if candidate_pos is None or candidate_len > capacity:
                break
            current.add(remaining.pop(candidate_pos))
        if current.docs:
            bins.append(current)
        else:
            # All remaining docs were topology-incompatible with this row shape.
            # Emit the next one alone to guarantee progress.
            doc = remaining.pop(0)
            solo = PackBin()
            solo.add(doc)
            bins.append(solo)
    return bins, overflow


def _pack_docs_sequential(
    docs: list[NormalizedDoc],
    *,
    target_length: int,
) -> tuple[list[PackBin], list[dict[str, Any]]]:
    bins: list[PackBin] = []
    overflow: list[dict[str, Any]] = []
    current = PackBin()
    for doc in sorted(docs, key=lambda item: item.source_doc_index):
        if doc.token_count > target_length:
            overflow.append(doc.to_overflow_record())
            continue
        if current.docs and not current.can_fit(doc.token_count, target_length):
            bins.append(current)
            current = PackBin()
        if current.docs and not _pack_bin_accepts_doc(current.docs, doc):
            bins.append(current)
            current = PackBin()
        current.add(doc)
        if current.used_tokens == target_length:
            bins.append(current)
            current = PackBin()
    if current.docs:
        bins.append(current)
    return bins, overflow


def _pad(values: list[int], target_length: int, *, pad_value: int = 0) -> list[int]:
    return list(values) + [pad_value] * max(target_length - len(values), 0)


def _target_ids_for_packed_tokens(
    concatenated_tokens: list[int],
    *,
    target_length: int,
    pad_token_id: int,
) -> list[int]:
    return _pad(concatenated_tokens[1:], target_length, pad_value=pad_token_id)


def _loss_mask_for_packed_docs(
    doc_ids: list[int],
    *,
    target_length: int,
) -> list[int]:
    keep: list[int] = []
    for pos in range(len(doc_ids)):
        if pos + 1 >= len(doc_ids):
            keep.append(0)
        else:
            keep.append(1 if doc_ids[pos] == doc_ids[pos + 1] else 0)
    return _pad(keep, target_length, pad_value=0)


def _materialize_packed_row(
    docs: list[NormalizedDoc],
    *,
    target_length: int,
    pad_token_id: int,
    pack_id: int,
) -> dict[str, Any]:
    ordered_docs = _order_docs_for_row(docs)
    concatenated_tokens: list[int] = []
    doc_ids: list[int] = []
    source_doc_indices: list[int] = []
    source_doc_token_lengths: list[int] = []
    source_platform_ids: list[list[int]] = []
    source_repo_stable_ids: list[str | None] = []
    source_filepath_stable_ids: list[str | None] = []
    source_file_local_commit_indices: list[int | None] = []
    chronology = _shared_chronology_for_docs(ordered_docs)

    token_meta_acc: dict[str, list[int]] = {
        column: [] for column in PACKED_TOKEN_METADATA_COLUMNS
    }
    chunk_starts: list[int] = []
    chunk_ends: list[int] = []
    chunk_kinds: list[int] = []
    chunk_dep_levels: list[int] = []
    call_edges: list[dict[str, int]] = []
    type_edges: list[dict[str, int]] = []
    changed_chunk_ids: list[int] = []
    changed_chunk_spans: list[dict[str, int]] = []

    token_offset = 0
    chunk_offset = 0
    for doc_id, doc in enumerate(ordered_docs, start=1):
        concatenated_tokens.extend(doc.token_ids)
        doc_ids.extend([doc_id] * doc.token_count)
        source_doc_indices.append(doc.source_doc_index)
        source_doc_token_lengths.append(doc.token_count)
        source_platform_ids.append(list(doc.platform_ids))
        source_repo_stable_ids.append(doc.chronology.get(REPO_STABLE_ID_COLUMN))
        source_filepath_stable_ids.append(doc.chronology.get(FILEPATH_STABLE_ID_COLUMN))
        source_file_local_commit_indices.append(
            doc.chronology.get(FILE_LOCAL_COMMIT_INDEX_COLUMN)
        )

        for column in PACKED_TOKEN_METADATA_COLUMNS:
            values = doc.token_meta[column]
            if values:
                token_meta_acc[column].extend(values)
            else:
                fill = PACKED_ROWS_DENSE_FALLBACK_FILL_VALUES.get(column, 0)
                token_meta_acc[column].extend([int(fill)] * doc.token_count)

        if doc.chunk_starts:
            chunk_starts.extend(token_offset + value for value in doc.chunk_starts)
            chunk_ends.extend(token_offset + value for value in doc.chunk_ends)
            chunk_kinds.extend(doc.chunk_kinds)
            chunk_dep_levels.extend(doc.chunk_dep_levels)
            call_edges.extend(
                {
                    "from": chunk_offset + int(edge["from"]),
                    "to": chunk_offset + int(edge["to"]),
                }
                for edge in doc.call_edges
            )
            type_edges.extend(
                {
                    "from": chunk_offset + int(edge["from"]),
                    "to": chunk_offset + int(edge["to"]),
                }
                for edge in doc.type_edges
            )
            changed_chunk_ids.extend(chunk_offset + value for value in doc.changed_chunk_ids)
            changed_chunk_spans.extend(
                {
                    "start": token_offset + int(start),
                    "end": token_offset + int(end),
                }
                for start, end in doc.changed_chunk_spans
            )
            chunk_offset += len(doc.chunk_starts)

        token_offset += doc.token_count

    valid_token_count = len(concatenated_tokens)
    trained_token_count = sum(
        _loss_mask_for_packed_docs(doc_ids, target_length=valid_token_count)
    )
    slack_tokens = target_length - valid_token_count
    pad_doc_id = len(ordered_docs) if ordered_docs else 0

    row: dict[str, Any] = {
        PACK_ID_COLUMN: int(pack_id),
        VALID_TOKEN_COUNT_COLUMN: int(valid_token_count),
        TRAINED_TOKEN_COUNT_COLUMN: int(trained_token_count),
        NUM_DOCS_COLUMN: int(len(ordered_docs)),
        SLACK_TOKENS_COLUMN: int(slack_tokens),
        INPUT_IDS_COLUMN: _pad(concatenated_tokens, target_length, pad_value=pad_token_id),
        TARGET_IDS_COLUMN: _target_ids_for_packed_tokens(
            concatenated_tokens,
            target_length=target_length,
            pad_token_id=pad_token_id,
        ),
        LOSS_MASK_COLUMN: _loss_mask_for_packed_docs(
            doc_ids,
            target_length=target_length,
        ),
        DOC_IDS_COLUMN: _pad(doc_ids, target_length, pad_value=pad_doc_id),
        SOURCE_DOC_INDICES_COLUMN: source_doc_indices,
        SOURCE_DOC_TOKEN_LENGTHS_COLUMN: source_doc_token_lengths,
        SOURCE_PLATFORM_IDS_COLUMN: source_platform_ids,
        SOURCE_REPO_STABLE_IDS_COLUMN: source_repo_stable_ids,
        SOURCE_FILEPATH_STABLE_IDS_COLUMN: source_filepath_stable_ids,
        SOURCE_FILE_LOCAL_COMMIT_INDICES_COLUMN: source_file_local_commit_indices,
        ROW_PLATFORM_IDS_COLUMN: _merged_platform_ids_for_docs(ordered_docs),
        CHANGED_CHUNK_IDS_COLUMN: changed_chunk_ids,
        CHANGED_CHUNK_SPANS_COLUMN: changed_chunk_spans,
        TOKEN_CHUNK_STARTS_COLUMN: chunk_starts,
        TOKEN_CHUNK_ENDS_COLUMN: chunk_ends,
        TOKEN_CHUNK_KINDS_COLUMN: chunk_kinds,
        TOKEN_CHUNK_DEP_LEVELS_COLUMN: chunk_dep_levels,
        TOKEN_CALL_EDGES_COLUMN: call_edges,
        TOKEN_TYPE_EDGES_COLUMN: type_edges,
    }
    for column in PACKED_ROW_PROVENANCE_COLUMNS:
        value = chronology.get(column) if chronology else None
        if column == PARENT_HASHES_COLUMN:
            row[column] = [str(item) for item in (value or [])]
        elif column in (PARENT_COUNT_COLUMN, FILE_LOCAL_COMMIT_INDEX_COLUMN):
            row[column] = int(value) if value is not None else None
        elif column in (
            IS_MERGE_COMMIT_COLUMN,
            HAS_AMBIGUOUS_RECONSTRUCTION_COLUMN,
            HAS_RENAME_AMBIGUITY_COLUMN,
        ):
            row[column] = bool(value) if value is not None else None
        else:
            row[column] = value
    for column in PACKED_TOKEN_METADATA_COLUMNS:
        pad_value = int(PACKED_ROWS_DENSE_FALLBACK_FILL_VALUES.get(column, 0))
        row[column] = _pad(token_meta_acc[column], target_length, pad_value=pad_value)
    return row


def pack_documents(
    docs: list[NormalizedDoc],
    *,
    target_length: int,
    pad_token_id: int = 0,
    strategy: PackingStrategy = "best_fit",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pack whole documents into fixed-length rows with no truncation."""

    if target_length <= 0:
        raise ValueError("target_length must be > 0")
    if strategy not in {"best_fit", "sequential"}:
        raise ValueError("strategy must be 'best_fit' or 'sequential'")

    if strategy == "sequential":
        bins, overflow = _pack_docs_sequential(docs, target_length=target_length)
    else:
        bins, overflow = _pack_docs_best_fit(docs, target_length=target_length)

    packed_rows = [
        _materialize_packed_row(
            pack.docs,
            target_length=target_length,
            pad_token_id=pad_token_id,
            pack_id=pack_id,
        )
        for pack_id, pack in enumerate(bins)
    ]
    return packed_rows, overflow


def rows_to_table(rows: list[dict[str, Any]]) -> pa.Table:
    """Convert materialized packed rows into a fixed-schema Arrow table."""

    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        normalized_rows.append(
            {
                PACK_ID_COLUMN: int(row.get(PACK_ID_COLUMN, 0)),
                VALID_TOKEN_COUNT_COLUMN: int(row.get(VALID_TOKEN_COUNT_COLUMN, 0)),
                TRAINED_TOKEN_COUNT_COLUMN: int(row.get(TRAINED_TOKEN_COUNT_COLUMN, 0)),
                NUM_DOCS_COLUMN: int(row.get(NUM_DOCS_COLUMN, 0)),
                SLACK_TOKENS_COLUMN: int(row.get(SLACK_TOKENS_COLUMN, 0)),
                INPUT_IDS_COLUMN: _as_int_list(row.get(INPUT_IDS_COLUMN)),
                TARGET_IDS_COLUMN: _as_int_list(row.get(TARGET_IDS_COLUMN)),
                LOSS_MASK_COLUMN: _as_int_list(row.get(LOSS_MASK_COLUMN)),
                DOC_IDS_COLUMN: _as_int_list(row.get(DOC_IDS_COLUMN)),
                SOURCE_DOC_INDICES_COLUMN: _as_int_list(row.get(SOURCE_DOC_INDICES_COLUMN)),
                SOURCE_DOC_TOKEN_LENGTHS_COLUMN: _as_int_list(
                    row.get(SOURCE_DOC_TOKEN_LENGTHS_COLUMN)
                ),
                SOURCE_PLATFORM_IDS_COLUMN: [
                    _as_int_list(item)
                    for item in row.get(SOURCE_PLATFORM_IDS_COLUMN, [])
                ],
                SOURCE_REPO_STABLE_IDS_COLUMN: [
                    None if item is None else str(item)
                    for item in row.get(SOURCE_REPO_STABLE_IDS_COLUMN, [])
                ],
                SOURCE_FILEPATH_STABLE_IDS_COLUMN: [
                    None if item is None else str(item)
                    for item in row.get(SOURCE_FILEPATH_STABLE_IDS_COLUMN, [])
                ],
                SOURCE_FILE_LOCAL_COMMIT_INDICES_COLUMN: [
                    int(item) if item is not None else None
                    for item in row.get(SOURCE_FILE_LOCAL_COMMIT_INDICES_COLUMN, [])
                ],
                ROW_PLATFORM_IDS_COLUMN: _as_int_list(row.get(ROW_PLATFORM_IDS_COLUMN)),
                REPO_COLUMN: row.get(REPO_COLUMN),
                FILEPATH_COLUMN: row.get(FILEPATH_COLUMN),
                COMMIT_HASH_COLUMN: row.get(COMMIT_HASH_COLUMN),
                TIMESTAMP_COLUMN: row.get(TIMESTAMP_COLUMN),
                PARENT_HASHES_COLUMN: [
                    str(item) for item in row.get(PARENT_HASHES_COLUMN, [])
                ],
                PARENT_COUNT_COLUMN: _coerce_optional_int(row.get(PARENT_COUNT_COLUMN)),
                IS_MERGE_COMMIT_COLUMN: (
                    bool(row.get(IS_MERGE_COMMIT_COLUMN))
                    if row.get(IS_MERGE_COMMIT_COLUMN) is not None
                    else None
                ),
                AUTHOR_TIMESTAMP_COLUMN: row.get(AUTHOR_TIMESTAMP_COLUMN),
                COMMIT_TIMESTAMP_COLUMN: row.get(COMMIT_TIMESTAMP_COLUMN),
                REPO_STABLE_ID_COLUMN: row.get(REPO_STABLE_ID_COLUMN),
                FILEPATH_STABLE_ID_COLUMN: row.get(FILEPATH_STABLE_ID_COLUMN),
                FILE_LOCAL_COMMIT_INDEX_COLUMN: _coerce_optional_int(
                    row.get(FILE_LOCAL_COMMIT_INDEX_COLUMN)
                ),
                HAS_AMBIGUOUS_RECONSTRUCTION_COLUMN: (
                    bool(row.get(HAS_AMBIGUOUS_RECONSTRUCTION_COLUMN))
                    if row.get(HAS_AMBIGUOUS_RECONSTRUCTION_COLUMN) is not None
                    else None
                ),
                HAS_RENAME_AMBIGUITY_COLUMN: (
                    bool(row.get(HAS_RENAME_AMBIGUITY_COLUMN))
                    if row.get(HAS_RENAME_AMBIGUITY_COLUMN) is not None
                    else None
                ),
                TOKEN_PLATFORM_IDS_COLUMN: _as_int_list(
                    row.get(TOKEN_PLATFORM_IDS_COLUMN)
                ),
                TOKEN_STRUCTURE_IDS_COLUMN: _as_int_list(
                    row.get(TOKEN_STRUCTURE_IDS_COLUMN)
                ),
                TOKEN_DEP_LEVELS_COLUMN: _as_int_list(row.get(TOKEN_DEP_LEVELS_COLUMN)),
                TOKEN_AST_DEPTH_COLUMN: _as_int_list(row.get(TOKEN_AST_DEPTH_COLUMN)),
                TOKEN_SIBLING_INDEX_COLUMN: _as_int_list(
                    row.get(TOKEN_SIBLING_INDEX_COLUMN)
                ),
                TOKEN_AST_NODE_TYPE_COLUMN: _as_int_list(
                    row.get(TOKEN_AST_NODE_TYPE_COLUMN)
                ),
                TOKEN_SYMBOL_IDS_COLUMN: _as_int_list(row.get(TOKEN_SYMBOL_IDS_COLUMN)),
                TOKEN_CALL_TARGETS_COLUMN: _as_int_list(
                    row.get(TOKEN_CALL_TARGETS_COLUMN)
                ),
                TOKEN_TYPE_REFS_COLUMN: _as_int_list(row.get(TOKEN_TYPE_REFS_COLUMN)),
                TOKEN_DEF_USE_COLUMN: _as_int_list(row.get(TOKEN_DEF_USE_COLUMN)),
                TOKEN_CHANGE_MASK_PRE_COLUMN: _as_int_list(
                    row.get(TOKEN_CHANGE_MASK_PRE_COLUMN)
                ),
                TOKEN_CHANGE_MASK_POST_COLUMN: _as_int_list(
                    row.get(TOKEN_CHANGE_MASK_POST_COLUMN)
                ),
                HUNK_ID_PER_TOKEN_COLUMN: _as_int_list(row.get(HUNK_ID_PER_TOKEN_COLUMN)),
                EDIT_OP_PER_TOKEN_COLUMN: _as_int_list(row.get(EDIT_OP_PER_TOKEN_COLUMN)),
                TOKEN_CHUNK_STARTS_COLUMN: _as_int_list(
                    row.get(TOKEN_CHUNK_STARTS_COLUMN)
                ),
                TOKEN_CHUNK_ENDS_COLUMN: _as_int_list(row.get(TOKEN_CHUNK_ENDS_COLUMN)),
                TOKEN_CHUNK_KINDS_COLUMN: _as_int_list(row.get(TOKEN_CHUNK_KINDS_COLUMN)),
                TOKEN_CHUNK_DEP_LEVELS_COLUMN: _as_int_list(
                    row.get(TOKEN_CHUNK_DEP_LEVELS_COLUMN)
                ),
                CHANGED_CHUNK_IDS_COLUMN: _as_int_list(row.get(CHANGED_CHUNK_IDS_COLUMN)),
                CHANGED_CHUNK_SPANS_COLUMN: [
                    {
                        "start": int(span["start"]),
                        "end": int(span["end"]),
                    }
                    for span in row.get(CHANGED_CHUNK_SPANS_COLUMN, [])
                ],
                TOKEN_CALL_EDGES_COLUMN: [
                    {
                        "from": int(edge["from"]),
                        "to": int(edge["to"]),
                    }
                    for edge in row.get(TOKEN_CALL_EDGES_COLUMN, [])
                ],
                TOKEN_TYPE_EDGES_COLUMN: [
                    {
                        "from": int(edge["from"]),
                        "to": int(edge["to"]),
                    }
                    for edge in row.get(TOKEN_TYPE_EDGES_COLUMN, [])
                ],
            }
        )
    return pa.Table.from_pylist(normalized_rows, schema=PACKED_ROW_SCHEMA)


def _empty_overflow_table() -> pa.Table:
    return pa.Table.from_pylist(
        [],
        schema=pa.schema(
            [
                pa.field("source_doc_index", pa.int64()),
                pa.field("token_count", pa.int32()),
                pa.field(TOKEN_IDS_COLUMN, pa.list_(pa.uint32())),
                pa.field(REPO_COLUMN, pa.string()),
                pa.field(FILEPATH_COLUMN, pa.string()),
                pa.field(COMMIT_HASH_COLUMN, pa.string()),
            ]
        ),
    )


def write_overflow_records(
    overflow_records: list[dict[str, Any]],
    output_path: str | os.PathLike[str],
) -> None:
    """Write overflow docs as JSONL or parquet."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as handle:
            for record in overflow_records:
                handle.write(json.dumps(record, sort_keys=True))
                handle.write("\n")
        return

    if overflow_records:
        table = pa.Table.from_pylist(overflow_records)
    else:
        table = _empty_overflow_table()
    pq.write_table(table, path)


def pack_parquet_dataset(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    target_length: int,
    pad_token_id: int = 0,
    strategy: PackingStrategy = "best_fit",
    overflow_output: str | os.PathLike[str] | None = None,
    row_group_size: int = 1024,
) -> dict[str, int]:
    """Pack a parquet dataset end to end and write packed rows."""

    docs = read_tokenized_documents(input_path)
    packed_rows, overflow = pack_documents(
        docs,
        target_length=target_length,
        pad_token_id=pad_token_id,
        strategy=strategy,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(rows_to_table(packed_rows), output, row_group_size=row_group_size)

    if overflow_output is not None:
        write_overflow_records(overflow, overflow_output)

    return {
        "input_docs": len(docs),
        "packed_rows": len(packed_rows),
        "overflow_docs": len(overflow),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline no-crop packer for tokenized enriched parquet rows."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input parquet file or directory of per-document tokenized parquet shards.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output parquet path for fixed-length packed rows.",
    )
    parser.add_argument(
        "--target-length",
        type=int,
        required=True,
        help="Fixed packed row length, typically 4096 or 8192.",
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
        help="Packing strategy for whole-document rows (default: best_fit).",
    )
    parser.add_argument(
        "--overflow-output",
        default="",
        help="Optional JSONL or parquet path for oversized docs that did not fit.",
    )
    parser.add_argument(
        "--row-group-size",
        type=int,
        default=1024,
        help="Output parquet row group size (default: 1024).",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    summary = pack_parquet_dataset(
        args.input,
        args.output,
        target_length=args.target_length,
        pad_token_id=args.pad_token_id,
        strategy=args.strategy,
        overflow_output=args.overflow_output or None,
        row_group_size=args.row_group_size,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "DOC_IDS_COLUMN",
    "INPUT_IDS_COLUMN",
    "LOSS_MASK_COLUMN",
    "NUM_DOCS_COLUMN",
    "PACK_ID_COLUMN",
    "SOURCE_DOC_INDICES_COLUMN",
    "TARGET_IDS_COLUMN",
    "VALID_TOKEN_COUNT_COLUMN",
    "NormalizedDoc",
    "normalize_document_record",
    "pack_documents",
    "pack_parquet_dataset",
    "read_tokenized_documents",
    "rows_to_table",
    "write_overflow_records",
]

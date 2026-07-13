"""Build typed CodePacket / CommitPacket objects from the v12/packed sources.

The two inputs are:

  (a) an ``LMTokenBatch`` — the dense, already-windowed tensors (tokens, targets,
      loss_mask, document_ids, and the structure/ast side channels) the trainer
      consumes; and
  (b) a ``pyarrow.Table`` row-group following the v12/packed parquet schema — the
      ragged source columns (token semantics, graph edges, chunk metadata,
      provenance, temporal/diff columns) keyed by the
      ``packed_rows_schema`` / ``tokenized_enriched_schema`` constants and the
      ``parquet_dataset`` group constants.

Column mapping uses the group constants from ``parquet_dataset.py`` so the builder
stays in lock-step with the reader's aliases.  Absent optional columns become
``None`` (and are recorded in ``metadata['absent_columns']``) — they are NEVER
fabricated.  Any shape mismatch FAILS LOUD (RULE #1).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import mlx.core as mx
import numpy as np

from cppmega_mlx.data.batch import LMTokenBatch
from cppmega_mlx.data.code_packet import CodePacket
from cppmega_mlx.data.commit_packet import CommitPacket
from cppmega_mlx.data.domain_packet import DomainEdgeIndex
from cppmega_mlx.data.graph_packet import EdgeIndex
from cppmega_mlx.data.parquet_dataset import (
    _TOKEN_CHUNK_METADATA_COLUMNS,
    _TOKEN_GRAPH_METADATA_COLUMNS,
    _TOKEN_SEMANTIC_METADATA_COLUMNS,
    _TOKEN_TEMPORAL_METADATA_COLUMNS,
    _normalize_edge_pairs,
)
from cppmega_mlx.data.nanochat_pipeline.tokenized_enriched_schema import (
    CHANGED_CHUNK_IDS_COLUMN,
    CHANGED_CHUNK_SPANS_COLUMN,
    COMMIT_HASH_COLUMN,
    EDIT_OP_PER_TOKEN_COLUMN,
    FILEPATH_COLUMN,
    HUNK_ID_PER_TOKEN_COLUMN,
    REPO_COLUMN,
    TIMESTAMP_COLUMN,
    TOKEN_CALL_EDGES_COLUMN,
    TOKEN_CALL_TARGETS_COLUMN,
    TOKEN_CHANGE_MASK_POST_COLUMN,
    TOKEN_CHANGE_MASK_PRE_COLUMN,
    TOKEN_CHUNK_DEP_LEVELS_COLUMN,
    TOKEN_CHUNK_ENDS_COLUMN,
    TOKEN_CHUNK_KINDS_COLUMN,
    TOKEN_CHUNK_STARTS_COLUMN,
    TOKEN_DEF_USE_COLUMN,
    TOKEN_BUILD_EDGES_COLUMN,
    TOKEN_CONFIDENCE_IDS_COLUMN,
    TOKEN_CROSS_DOMAIN_EDGES_COLUMN,
    TOKEN_DIAGNOSTIC_EDGES_COLUMN,
    TOKEN_DOMAIN_EDGES_COLUMN,
    TOKEN_DOMAIN_IDS_COLUMN,
    TOKEN_ENTITY_IDS_COLUMN,
    TOKEN_ROLE_IDS_COLUMN,
    TOKEN_SCOPE_IDS_COLUMN,
    TOKEN_SHELL_EDGES_COLUMN,
    TOKEN_SOURCE_DOC_IDS_COLUMN,
    TOKEN_SYMBOL_IDS_COLUMN,
    TOKEN_TYPE_EDGES_COLUMN,
    TOKEN_TYPE_REFS_COLUMN,
)


# Semantic parquet column -> CodePacket field name.
_SEMANTIC_COLUMN_TO_FIELD: Mapping[str, str] = {
    TOKEN_SYMBOL_IDS_COLUMN: "symbol_ids",
    TOKEN_CALL_TARGETS_COLUMN: "call_targets",
    TOKEN_TYPE_REFS_COLUMN: "type_refs",
    TOKEN_DEF_USE_COLUMN: "def_use",
}
assert set(_SEMANTIC_COLUMN_TO_FIELD) == set(_TOKEN_SEMANTIC_METADATA_COLUMNS)

# Chunk parquet column -> CodePacket field name.
_CHUNK_COLUMN_TO_FIELD: Mapping[str, str] = {
    TOKEN_CHUNK_STARTS_COLUMN: "chunk_starts",
    TOKEN_CHUNK_ENDS_COLUMN: "chunk_ends",
    TOKEN_CHUNK_KINDS_COLUMN: "chunk_kinds",
    TOKEN_CHUNK_DEP_LEVELS_COLUMN: "chunk_dep_levels",
}
assert set(_CHUNK_COLUMN_TO_FIELD) == set(_TOKEN_CHUNK_METADATA_COLUMNS)

# Graph parquet column -> (CodePacket field name, relation label).
_GRAPH_COLUMN_TO_FIELD: Mapping[str, tuple[str, str]] = {
    TOKEN_CALL_EDGES_COLUMN: ("call_edges", "call"),
    TOKEN_TYPE_EDGES_COLUMN: ("type_edges", "type"),
}
assert set(_GRAPH_COLUMN_TO_FIELD) == set(_TOKEN_GRAPH_METADATA_COLUMNS)

_DOMAIN_TOKEN_COLUMN_TO_FIELD: Mapping[str, str] = {
    TOKEN_DOMAIN_IDS_COLUMN: "domain_ids",
    TOKEN_ROLE_IDS_COLUMN: "role_ids",
    TOKEN_ENTITY_IDS_COLUMN: "entity_ids",
    TOKEN_SCOPE_IDS_COLUMN: "scope_ids",
    TOKEN_SOURCE_DOC_IDS_COLUMN: "source_doc_ids",
    TOKEN_CONFIDENCE_IDS_COLUMN: "confidence_ids",
}

_DOMAIN_EDGE_COLUMN_TO_FIELD: Mapping[str, str] = {
    TOKEN_DOMAIN_EDGES_COLUMN: "domain_edges",
    TOKEN_BUILD_EDGES_COLUMN: "build_edges",
    TOKEN_SHELL_EDGES_COLUMN: "shell_edges",
    TOKEN_DIAGNOSTIC_EDGES_COLUMN: "diagnostic_edges",
    TOKEN_CROSS_DOMAIN_EDGES_COLUMN: "cross_domain_edges",
}

# Temporal token-level parquet column -> CommitPacket field name.
_TEMPORAL_COLUMN_TO_FIELD: Mapping[str, str] = {
    TOKEN_CHANGE_MASK_PRE_COLUMN: "change_mask_pre",
    TOKEN_CHANGE_MASK_POST_COLUMN: "change_mask_post",
    HUNK_ID_PER_TOKEN_COLUMN: "hunk_ids",
    EDIT_OP_PER_TOKEN_COLUMN: "edit_ops",
}
assert set(_TEMPORAL_COLUMN_TO_FIELD) == set(_TOKEN_TEMPORAL_METADATA_COLUMNS)


def _table_columns(table: Any) -> dict[str, list[Any]]:
    """Normalize a pyarrow Table (or column->list mapping) to plain Python lists."""

    if isinstance(table, Mapping):
        return {str(name): list(values) for name, values in table.items()}
    column_names = getattr(table, "column_names", None)
    if column_names is None:
        raise TypeError(
            "code_packet_builder expects a pyarrow.Table or column mapping, got "
            f"{type(table).__name__}"
        )
    return {str(name): table[name].to_pylist() for name in column_names}


def _row_count(columns: Mapping[str, list[Any]]) -> int:
    lengths = {len(values) for values in columns.values()}
    if len(lengths) > 1:
        raise ValueError(
            f"row-group columns have inconsistent row counts: "
            f"{ {name: len(v) for name, v in columns.items()} }"
        )
    return next(iter(lengths)) if lengths else 0


def _int_vector(value: Any, *, where: str) -> mx.array:
    if value is None:
        raise ValueError(f"{where}: cannot build int vector from None")
    arr = np.asarray(value)
    if arr.ndim != 1:
        if arr.size == 0:
            arr = arr.reshape(0)
        else:
            raise ValueError(
                f"{where}: expected a 1-D token-aligned sequence, got shape "
                f"{tuple(arr.shape)}"
            )
    return mx.array(arr.astype(np.int32))


def _symbol_id_vector(value: Any, *, where: str) -> mx.array:
    if value is None:
        raise ValueError(f"{where}: cannot build symbol ID vector from None")
    arr = np.asarray(value, dtype=object)
    if arr.ndim != 1:
        if arr.size == 0:
            arr = arr.reshape(0)
        else:
            raise ValueError(
                f"{where}: expected a 1-D token-aligned sequence, got shape "
                f"{tuple(arr.shape)}"
            )
    values = [int(item) for item in arr.tolist()]
    if any(item < 0 or item > np.iinfo(np.uint64).max for item in values):
        raise ValueError(f"{where}: symbol IDs must fit unsigned 64-bit")
    return mx.array(np.asarray(values, dtype=np.uint64))


def _str_scalar(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _build_edge_index(
    raw: Any, *, relation: str, num_nodes: int | None
) -> EdgeIndex | None:
    pairs = _normalize_edge_pairs(raw)
    # An absent column yields raw is None -> [] ; keep an empty (but present) edge
    # set distinct from "column absent" which the caller handles before calling.
    return EdgeIndex.from_pairs(pairs, relation=relation, num_nodes=num_nodes)


def _normalize_edge_triples(raw: Any) -> list[tuple[int, int, int]]:
    if raw is None:
        return []
    triples: list[tuple[int, int, int]] = []
    for edge in raw:
        if hasattr(edge, "as_py"):
            edge = edge.as_py()
        if isinstance(edge, Mapping):
            src = edge.get("from", edge.get("src"))
            dst = edge.get("to", edge.get("dst"))
            kind = edge.get("kind")
        elif isinstance(edge, (list, tuple, np.ndarray)) and len(edge) >= 3:
            src, dst, kind = edge[0], edge[1], edge[2]
        else:
            raise ValueError(
                "domain edge triple must be {from,to,kind}/{src,dst,kind} or "
                f"length-3 sequence, got {type(edge).__name__}"
            )
        if src is None or dst is None or kind is None:
            raise ValueError(f"domain edge triple has missing value: {edge!r}")
        src_i = int(src)
        dst_i = int(dst)
        kind_i = int(kind)
        if src_i < 0 or dst_i < 0 or kind_i < 0:
            raise ValueError(f"domain edge triple must be non-negative: {edge!r}")
        triples.append((src_i, dst_i, kind_i))
    return triples


def _build_domain_edge_index(raw: Any, *, num_tokens: int) -> DomainEdgeIndex:
    edge_index = DomainEdgeIndex.from_triples(_normalize_edge_triples(raw))
    if edge_index.num_edges:
        max_endpoint = int(max(mx.max(edge_index.src).item(), mx.max(edge_index.dst).item()))
        if max_endpoint >= num_tokens:
            raise ValueError(
                f"domain edge endpoint {max_endpoint} outside token range 0..{num_tokens - 1}"
            )
    return edge_index


def build_code_packet_from_row(
    *,
    token_ids: mx.array,
    columns: Mapping[str, list[Any]],
    row_index: int,
    target_ids: mx.array | None = None,
    loss_mask: mx.array | None = None,
    document_ids: mx.array | None = None,
    structure_ids: mx.array | None = None,
    ast_depth: mx.array | None = None,
    sibling_index: mx.array | None = None,
    ast_node_type: mx.array | None = None,
    dep_levels: mx.array | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> CodePacket:
    """Build one CodePacket from dense per-row tensors + a parquet row's columns."""

    absent: list[str] = []
    present: list[str] = []

    semantic_kwargs: dict[str, mx.array] = {}
    for column, field_name in _SEMANTIC_COLUMN_TO_FIELD.items():
        if column not in columns:
            absent.append(column)
            continue
        vector_builder = (
            _int_vector if column == TOKEN_DEF_USE_COLUMN else _symbol_id_vector
        )
        semantic_kwargs[field_name] = vector_builder(
            columns[column][row_index], where=f"{column}[row={row_index}]"
        )
        present.append(column)

    chunk_kwargs: dict[str, mx.array] = {}
    for column, field_name in _CHUNK_COLUMN_TO_FIELD.items():
        if column not in columns:
            absent.append(column)
            continue
        chunk_kwargs[field_name] = _int_vector(
            columns[column][row_index], where=f"{column}[row={row_index}]"
        )
        present.append(column)

    num_chunks = (
        int(chunk_kwargs["chunk_starts"].shape[0])
        if "chunk_starts" in chunk_kwargs
        else None
    )

    edge_kwargs: dict[str, EdgeIndex] = {}
    for column, (field_name, relation) in _GRAPH_COLUMN_TO_FIELD.items():
        if column not in columns:
            absent.append(column)
            continue
        edge_kwargs[field_name] = _build_edge_index(
            columns[column][row_index], relation=relation, num_nodes=num_chunks
        )
        present.append(column)

    domain_token_kwargs: dict[str, mx.array] = {}
    for column, field_name in _DOMAIN_TOKEN_COLUMN_TO_FIELD.items():
        if column not in columns:
            absent.append(column)
            continue
        domain_token_kwargs[field_name] = _int_vector(
            columns[column][row_index], where=f"{column}[row={row_index}]"
        )
        present.append(column)

    domain_edge_kwargs: dict[str, DomainEdgeIndex] = {}
    num_tokens = int(token_ids.shape[-1])
    for column, field_name in _DOMAIN_EDGE_COLUMN_TO_FIELD.items():
        if column not in columns:
            absent.append(column)
            continue
        domain_edge_kwargs[field_name] = _build_domain_edge_index(
            columns[column][row_index],
            num_tokens=num_tokens,
        )
        present.append(column)

    metadata: dict[str, Any] = {
        "absent_columns": tuple(absent),
        "present_columns": tuple(present),
        "row_index": int(row_index),
    }
    if extra_metadata:
        metadata.update(dict(extra_metadata))

    return CodePacket(
        token_ids=token_ids,
        target_ids=target_ids,
        loss_mask=loss_mask,
        document_ids=document_ids,
        repo=_str_scalar(columns[REPO_COLUMN][row_index]) if REPO_COLUMN in columns else None,
        filepath=_str_scalar(columns[FILEPATH_COLUMN][row_index])
        if FILEPATH_COLUMN in columns
        else None,
        commit_or_ref=_str_scalar(columns[COMMIT_HASH_COLUMN][row_index])
        if COMMIT_HASH_COLUMN in columns
        else None,
        structure_ids=structure_ids,
        ast_depth=ast_depth,
        sibling_index=sibling_index,
        ast_node_type=ast_node_type,
        dep_levels=dep_levels,
        call_edges=edge_kwargs.get("call_edges"),
        type_edges=edge_kwargs.get("type_edges"),
        metadata=metadata,
        **semantic_kwargs,
        **domain_token_kwargs,
        **domain_edge_kwargs,
        **chunk_kwargs,
    )


def _row_tensor(batch_field: mx.array | None, row: int) -> mx.array | None:
    if batch_field is None:
        return None
    if batch_field.ndim < 1:
        raise ValueError(
            f"LMTokenBatch field expected >=1-D, got shape {tuple(batch_field.shape)}"
        )
    return batch_field[row]


def build_code_packets(
    batch: LMTokenBatch,
    table: Any,
    *,
    row_indices: Sequence[int] | None = None,
) -> list[CodePacket]:
    """Build one CodePacket per batch row from an LMTokenBatch + parquet row-group.

    ``row_indices`` maps batch row ``i`` to its source row in ``table``; defaults
    to ``range(batch_size)`` which requires the table row-count to match the batch.
    """

    if not isinstance(batch, LMTokenBatch):
        raise TypeError(
            f"build_code_packets expects an LMTokenBatch, got {type(batch).__name__}"
        )
    columns = _table_columns(table)
    table_rows = _row_count(columns)
    batch_size = int(batch.tokens.shape[0])

    if row_indices is None:
        if columns and table_rows != batch_size:
            raise ValueError(
                f"row-group row count {table_rows} != batch size {batch_size}; pass "
                "row_indices to map batch rows to table rows explicitly"
            )
        row_indices = tuple(range(batch_size))
    else:
        row_indices = tuple(int(i) for i in row_indices)
        if len(row_indices) != batch_size:
            raise ValueError(
                f"row_indices length {len(row_indices)} != batch size {batch_size}"
            )

    # Structure/ast channels come from LMTokenBatch's typed/family side channels.
    syntax = (batch.side_channels or {}).get("semantic_graph", {})

    packets: list[CodePacket] = []
    for batch_row, table_row in enumerate(row_indices):
        if columns and not (0 <= table_row < table_rows):
            raise ValueError(
                f"row_indices[{batch_row}]={table_row} out of range for "
                f"{table_rows} table rows"
            )
        packets.append(
            build_code_packet_from_row(
                token_ids=batch.tokens[batch_row],
                columns=columns if columns else {},
                row_index=table_row if columns else 0,
                target_ids=_row_tensor(batch.target_tokens, batch_row),
                loss_mask=_row_tensor(batch.loss_mask, batch_row),
                document_ids=_row_tensor(batch.document_ids, batch_row),
                structure_ids=_row_tensor(batch.structure_ids, batch_row),
                ast_depth=_row_tensor(batch.ast_depth_ids, batch_row),
                sibling_index=_row_tensor(batch.sibling_index_ids, batch_row),
                ast_node_type=_row_tensor(batch.node_type_ids, batch_row),
                dep_levels=_row_tensor(batch.dep_levels, batch_row),
                extra_metadata={"batch_row": batch_row, "syntax_present": tuple(syntax)},
            )
        )
    return packets


def build_commit_packet_from_row(
    *,
    columns: Mapping[str, list[Any]],
    row_index: int,
    pre_token_ids: mx.array | None = None,
    post_token_ids: mx.array | None = None,
    diff_token_ids: mx.array | None = None,
    commit_msg: mx.array | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> CommitPacket:
    """Build one CommitPacket from temporal/diff columns of a parquet row.

    The post-state token ids default to the row's token ids when not supplied so
    the token-level diff channels (hunk/edit-op/change_mask_post) have a reference
    to validate against.
    """

    absent: list[str] = []
    present: list[str] = []

    temporal_kwargs: dict[str, mx.array] = {}
    for column, field_name in _TEMPORAL_COLUMN_TO_FIELD.items():
        if column not in columns:
            absent.append(column)
            continue
        temporal_kwargs[field_name] = _int_vector(
            columns[column][row_index], where=f"{column}[row={row_index}]"
        )
        present.append(column)

    changed_chunk_ids = None
    changed_chunk_spans = None
    if CHANGED_CHUNK_IDS_COLUMN in columns and CHANGED_CHUNK_SPANS_COLUMN in columns:
        raw_ids = columns[CHANGED_CHUNK_IDS_COLUMN][row_index]
        raw_spans = columns[CHANGED_CHUNK_SPANS_COLUMN][row_index]
        ids_arr = np.asarray(raw_ids if raw_ids is not None else [], dtype=np.int32).reshape(-1)
        span_pairs = _normalize_edge_pairs(raw_spans)
        spans_arr = (
            np.asarray(span_pairs, dtype=np.int32)
            if span_pairs
            else np.zeros((0, 2), dtype=np.int32)
        )
        changed_chunk_ids = mx.array(ids_arr)
        changed_chunk_spans = mx.array(spans_arr)
        present.extend([CHANGED_CHUNK_IDS_COLUMN, CHANGED_CHUNK_SPANS_COLUMN])
    else:
        if CHANGED_CHUNK_IDS_COLUMN not in columns:
            absent.append(CHANGED_CHUNK_IDS_COLUMN)
        if CHANGED_CHUNK_SPANS_COLUMN not in columns:
            absent.append(CHANGED_CHUNK_SPANS_COLUMN)

    metadata: dict[str, Any] = {
        "absent_columns": tuple(absent),
        "present_columns": tuple(present),
        "row_index": int(row_index),
    }
    if extra_metadata:
        metadata.update(dict(extra_metadata))

    return CommitPacket(
        pre_token_ids=pre_token_ids,
        post_token_ids=post_token_ids,
        diff_token_ids=diff_token_ids,
        commit_msg=commit_msg,
        changed_chunk_ids=changed_chunk_ids,
        changed_chunk_spans=changed_chunk_spans,
        repo=_str_scalar(columns[REPO_COLUMN][row_index]) if REPO_COLUMN in columns else None,
        filepath=_str_scalar(columns[FILEPATH_COLUMN][row_index])
        if FILEPATH_COLUMN in columns
        else None,
        commit_or_ref=_str_scalar(columns[COMMIT_HASH_COLUMN][row_index])
        if COMMIT_HASH_COLUMN in columns
        else (
            _str_scalar(columns[TIMESTAMP_COLUMN][row_index])
            if TIMESTAMP_COLUMN in columns
            else None
        ),
        metadata=metadata,
        **temporal_kwargs,
    )


__all__ = [
    "build_code_packet_from_row",
    "build_code_packets",
    "build_commit_packet_from_row",
]

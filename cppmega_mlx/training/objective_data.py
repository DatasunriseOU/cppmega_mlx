"""Typed tokenized-enriched rows -> production objective sources."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import mlx.core as mx
import numpy as np

from cppmega_mlx.data.code_packet import CodePacket
from cppmega_mlx.data.code_packet_builder import (
    build_code_packet_from_row,
    build_commit_packet_from_row,
)
from cppmega_mlx.data.nanochat_pipeline.tokenized_enriched_schema import (
    COMMIT_MSG_TOKEN_IDS_COLUMN,
    DIFF_TOKEN_IDS_COLUMN,
    IFIM_INSTRUCTION_TOKEN_IDS_COLUMN,
    POST_TOKEN_IDS_COLUMN,
    PRE_TOKEN_IDS_COLUMN,
    PLATFORM_IDS_COLUMN,
    TOKEN_AST_DEPTH_COLUMN,
    TOKEN_AST_NODE_TYPE_COLUMN,
    TOKEN_DEP_LEVELS_COLUMN,
    TOKEN_IDS_COLUMN,
    TOKEN_SIBLING_INDEX_COLUMN,
    TOKEN_STRUCTURE_IDS_COLUMN,
    TOKENIZED_ENRICHED_CHUNK_COLUMNS,
    TOKENIZED_ENRICHED_GRAPH_COLUMNS,
    TOKENIZED_ENRICHED_DOMAIN_GRAPH_COLUMNS,
    TOKENIZED_ENRICHED_DOMAIN_TOKEN_COLUMNS,
    TOKENIZED_ENRICHED_SEMANTIC_COLUMNS,
    TOKENIZED_ENRICHED_TEMPORAL_TOKEN_COLUMNS,
    SOURCE_IDENTITY_REGISTRY_COLUMN,
)
from cppmega_mlx.data.graph_packet import EdgeIndex, GraphPacket
from cppmega_mlx.data.symbol_identity import SYMBOL_IDENTITIES_COLUMN
from cppmega_mlx.training.objective_mixer import ObjectiveSource

_CHUNK_GRAPH_FIELDS = {"call": "call_edges", "type": "type_edges"}
_TOKEN_GRAPH_FIELDS = {
    "domain": "domain_edges",
    "build": "build_edges",
    "shell": "shell_edges",
    "diagnostic": "diagnostic_edges",
    "cross_domain": "cross_domain_edges",
}

OBJECTIVE_SECTION_COLUMNS = (
    IFIM_INSTRUCTION_TOKEN_IDS_COLUMN,
    COMMIT_MSG_TOKEN_IDS_COLUMN,
    PRE_TOKEN_IDS_COLUMN,
    POST_TOKEN_IDS_COLUMN,
    DIFF_TOKEN_IDS_COLUMN,
)

OBJECTIVE_REQUIRED_SOURCE_COLUMNS = (
    TOKEN_IDS_COLUMN,
    TOKEN_STRUCTURE_IDS_COLUMN,
    TOKEN_DEP_LEVELS_COLUMN,
    TOKEN_AST_DEPTH_COLUMN,
    TOKEN_SIBLING_INDEX_COLUMN,
    TOKEN_AST_NODE_TYPE_COLUMN,
    *TOKENIZED_ENRICHED_SEMANTIC_COLUMNS,
    *TOKENIZED_ENRICHED_CHUNK_COLUMNS,
    *TOKENIZED_ENRICHED_GRAPH_COLUMNS,
)

OBJECTIVE_SOURCE_COLUMNS = (
    *OBJECTIVE_REQUIRED_SOURCE_COLUMNS,
    *OBJECTIVE_SECTION_COLUMNS,
    PLATFORM_IDS_COLUMN,
    *TOKENIZED_ENRICHED_DOMAIN_TOKEN_COLUMNS,
    *TOKENIZED_ENRICHED_DOMAIN_GRAPH_COLUMNS,
    *TOKENIZED_ENRICHED_TEMPORAL_TOKEN_COLUMNS,
    SYMBOL_IDENTITIES_COLUMN,
    SOURCE_IDENTITY_REGISTRY_COLUMN,
)

OBJECTIVE_MEGATRON_REQUIRED_SOURCE_COLUMNS = tuple(
    column
    for column in OBJECTIVE_SOURCE_COLUMNS
    if column not in OBJECTIVE_SECTION_COLUMNS
)


def require_objective_source_columns(columns: Sequence[str]) -> None:
    missing = sorted(set(OBJECTIVE_REQUIRED_SOURCE_COLUMNS) - set(columns))
    if missing:
        raise ValueError(
            "tokenized objective source is missing required typed columns: "
            + ", ".join(missing)
        )


def require_megatron_objective_source_columns(columns: Sequence[str]) -> None:
    missing = sorted(set(OBJECTIVE_MEGATRON_REQUIRED_SOURCE_COLUMNS) - set(columns))
    if missing:
        raise ValueError(
            "Megatron objective materialization is missing required typed "
            "columns: " + ", ".join(missing)
        )


def _i32(values: Any, *, where: str) -> mx.array:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{where}: expected a 1-D token vector, got {array.shape}")
    return mx.array(array.astype(np.int32))


def _optional_i32(row: Mapping[str, Any], column: str) -> mx.array | None:
    values = row.get(column)
    if values is None:
        return None
    return _i32(values, where=column)


def objective_source_from_tokenized_row(
    row: Mapping[str, Any],
    *,
    source_index: int,
) -> ObjectiveSource:
    """Build code and, when complete, commit views from one typed source row."""

    columns = {str(name): [value] for name, value in row.items()}
    token_ids = _i32(row.get(TOKEN_IDS_COLUMN), where=TOKEN_IDS_COLUMN)
    raw_document_ids = row.get("doc_ids")
    document_ids = (
        [1] * int(token_ids.shape[0])
        if raw_document_ids is None
        else [int(value) for value in raw_document_ids]
    )
    if len(document_ids) != int(token_ids.shape[0]):
        raise ValueError(
            f"doc_ids length {len(document_ids)} != token_ids length "
            f"{int(token_ids.shape[0])}"
        )
    if any(value <= 0 for value in document_ids):
        raise ValueError("doc_ids must be positive for every source token")
    code_packet = build_code_packet_from_row(
        token_ids=token_ids,
        columns=columns,
        row_index=0,
        document_ids=mx.array(np.asarray(document_ids, dtype=np.int32)),
        structure_ids=_optional_i32(row, TOKEN_STRUCTURE_IDS_COLUMN),
        dep_levels=_optional_i32(row, TOKEN_DEP_LEVELS_COLUMN),
        ast_depth=_optional_i32(row, TOKEN_AST_DEPTH_COLUMN),
        sibling_index=_optional_i32(row, TOKEN_SIBLING_INDEX_COLUMN),
        ast_node_type=_optional_i32(row, TOKEN_AST_NODE_TYPE_COLUMN),
        extra_metadata={
            "source_index": int(source_index),
            "platform_ids": row.get(PLATFORM_IDS_COLUMN),
            SOURCE_IDENTITY_REGISTRY_COLUMN: row.get(SOURCE_IDENTITY_REGISTRY_COLUMN),
            SYMBOL_IDENTITIES_COLUMN: row.get(SYMBOL_IDENTITIES_COLUMN),
            "token_change_mask_pre": row.get("token_change_mask_pre"),
            "token_change_mask_post": row.get("token_change_mask_post"),
        },
    )

    def has_values(column: str) -> bool:
        value = row.get(column)
        if value is None:
            return False
        array = np.asarray(value)
        return array.ndim == 1 and int(array.size) > 0

    has_any_commit_section = any(
        has_values(column)
        for column in (
            COMMIT_MSG_TOKEN_IDS_COLUMN,
            PRE_TOKEN_IDS_COLUMN,
            POST_TOKEN_IDS_COLUMN,
            DIFF_TOKEN_IDS_COLUMN,
        )
    )
    # These masks annotate the assembled code document, not the independently
    # tokenized pre/post sections. Attaching them to CommitPacket would falsely
    # claim token alignment and either raise or corrupt temporal supervision.
    commit_columns = {
        name: values
        for name, values in columns.items()
        if name not in TOKENIZED_ENRICHED_TEMPORAL_TOKEN_COLUMNS
    }
    commit_packet = (
        build_commit_packet_from_row(columns=commit_columns, row_index=0)
        if has_any_commit_section
        else None
    )
    return ObjectiveSource(code_packet=code_packet, commit_packet=commit_packet)


def token_graph_for_aligned_example(
    packet: CodePacket,
    *,
    input_length: int,
) -> GraphPacket | None:
    """Expand chunk-index graph edges to every valid token pair in each span."""

    targets, pair_mask, relation_pairs = _graph_arrays(
        packet,
        input_length=input_length,
        upstream_pair_mask=None,
        relations=None,
    )
    if not targets.any():
        return None
    edges: dict[str, EdgeIndex] = {}
    for relation, pairs in relation_pairs.items():
        if pairs:
            edges[relation] = EdgeIndex.from_pairs(
                pairs, relation=relation, num_nodes=input_length
            )
    if not edges:
        return None
    return GraphPacket(edges=edges, num_nodes=input_length)


def _graph_arrays(
    packet: CodePacket,
    *,
    input_length: int,
    upstream_pair_mask: Any | None,
    relations: Sequence[str] | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[tuple[int, int]]]]:
    if input_length < 1:
        raise ValueError(f"graph input_length must be >=1, got {input_length}")
    if packet.document_ids is None:
        raise ValueError("graph supervision requires token-aligned document_ids")

    token_count = int(packet.token_ids.shape[0])
    if input_length > token_count:
        raise ValueError(
            f"graph input_length {input_length} exceeds token count {token_count}"
        )
    supported_relations = set(_CHUNK_GRAPH_FIELDS) | set(_TOKEN_GRAPH_FIELDS)
    selected_relations = supported_relations if relations is None else set(relations)
    unknown_relations = sorted(selected_relations - supported_relations)
    if unknown_relations:
        raise ValueError(f"unsupported graph relations: {unknown_relations}")

    if relations is not None:
        missing_relations = sorted(
            relation
            for relation in selected_relations
            if getattr(
                packet,
                _CHUNK_GRAPH_FIELDS.get(relation, _TOKEN_GRAPH_FIELDS.get(relation)),
            )
            is None
        )
        if missing_relations:
            raise ValueError(
                "graph supervision is missing required relation sidecars: "
                + ", ".join(missing_relations)
            )
    chunk_edges = {
        relation: getattr(packet, field)
        for relation, field in _CHUNK_GRAPH_FIELDS.items()
        if relation in selected_relations and getattr(packet, field) is not None
    }
    starts: list[int] = []
    ends: list[int] = []
    if chunk_edges:
        if packet.chunk_starts is None or packet.chunk_ends is None:
            raise ValueError(
                "chunk graph supervision requires chunk_starts and chunk_ends"
            )
        starts = [int(value) for value in np.asarray(packet.chunk_starts).tolist()]
        ends = [int(value) for value in np.asarray(packet.chunk_ends).tolist()]
        if len(starts) != len(ends):
            raise ValueError(
                f"chunk_starts length {len(starts)} != chunk_ends length {len(ends)}"
            )
        for chunk_index, (start, end) in enumerate(zip(starts, ends, strict=True)):
            if not 0 <= start < end <= token_count:
                raise ValueError(
                    f"chunk {chunk_index} span [{start}, {end}) is outside "
                    f"0..{token_count}"
                )

    document_ids = np.asarray(packet.document_ids).reshape(-1)
    if document_ids.shape != (token_count,):
        raise ValueError(
            f"document_ids shape {document_ids.shape} != token shape ({token_count},)"
        )
    document_ids = document_ids[:input_length].astype(np.int64)
    if np.any(document_ids <= 0):
        raise ValueError("graph document_ids must be positive on every input token")

    positions = np.arange(input_length, dtype=np.int64)
    pair_mask = positions[:, None] >= positions[None, :]
    pair_mask &= document_ids[:, None] == document_ids[None, :]
    if upstream_pair_mask is not None:
        upstream = np.asarray(upstream_pair_mask)
        if upstream.shape != (input_length, input_length):
            raise ValueError(
                f"upstream graph pair mask shape {upstream.shape} != "
                f"({input_length}, {input_length})"
            )
        if not np.all(np.isin(upstream, (0, 1, False, True))):
            raise ValueError("upstream graph pair mask must be binary")
        pair_mask &= upstream.astype(bool)

    targets = np.zeros((input_length, input_length), dtype=np.float32)
    relation_pairs: dict[str, list[tuple[int, int]]] = {}
    for relation, edge in chunk_edges.items():
        expanded: list[tuple[int, int]] = []
        for source_chunk, destination_chunk in edge.to_pairs():
            if not (
                0 <= source_chunk < len(starts) and 0 <= destination_chunk < len(starts)
            ):
                raise ValueError(
                    f"{relation} edge ({source_chunk}, {destination_chunk}) is "
                    f"outside chunk layout of size {len(starts)}"
                )
            source_end = min(ends[source_chunk], input_length)
            destination_end = min(ends[destination_chunk], input_length)
            for source_token in range(starts[source_chunk], source_end):
                for destination_token in range(
                    starts[destination_chunk], destination_end
                ):
                    if pair_mask[source_token, destination_token]:
                        targets[source_token, destination_token] = 1.0
                        expanded.append((source_token, destination_token))
        relation_pairs[relation] = expanded

    for relation, field in _TOKEN_GRAPH_FIELDS.items():
        if relation not in selected_relations:
            continue
        edge = getattr(packet, field)
        if edge is None:
            continue
        expanded = []
        for source_token, destination_token, _kind in edge.to_triples():
            if not (
                0 <= source_token < token_count and 0 <= destination_token < token_count
            ):
                raise ValueError(
                    f"{relation} edge ({source_token}, {destination_token}) is "
                    f"outside token layout of size {token_count}"
                )
            if (
                source_token < input_length
                and destination_token < input_length
                and pair_mask[source_token, destination_token]
            ):
                targets[source_token, destination_token] = 1.0
                expanded.append((source_token, destination_token))
        relation_pairs[relation] = expanded
    return targets, pair_mask.astype(np.float32), relation_pairs


def graph_targets_and_pair_mask(
    packet: CodePacket,
    *,
    input_length: int,
    relations: Sequence[str] | None = None,
    upstream_pair_mask: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return dense graph targets and the exact eligible token-pair mask."""

    targets, pair_mask, _relations = _graph_arrays(
        packet,
        input_length=input_length,
        upstream_pair_mask=upstream_pair_mask,
        relations=relations,
    )
    return targets, pair_mask


__all__ = [
    "OBJECTIVE_SECTION_COLUMNS",
    "OBJECTIVE_SOURCE_COLUMNS",
    "OBJECTIVE_REQUIRED_SOURCE_COLUMNS",
    "OBJECTIVE_MEGATRON_REQUIRED_SOURCE_COLUMNS",
    "graph_targets_and_pair_mask",
    "objective_source_from_tokenized_row",
    "require_megatron_objective_source_columns",
    "require_objective_source_columns",
    "token_graph_for_aligned_example",
]

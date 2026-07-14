"""Typed tokenized-enriched rows -> production objective sources."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np

from cppmega_mlx.data.code_packet import CodePacket
from cppmega_mlx.data.code_packet_builder import (
    build_code_packet_from_row,
    build_commit_packet_from_row,
    build_commit_packets_from_packed_row,
)
from cppmega_mlx.data.nanochat_pipeline.packed_rows_schema import (
    INPUT_IDS_COLUMN,
    NUM_DOCS_COLUMN,
    PACKED_ROWS_OBJECTIVE_SOURCE_COLUMNS,
    PACKED_ROWS_TOKEN_ALIGNED_COLUMNS,
    SOURCE_IFIM_INSTRUCTION_TOKEN_IDS_COLUMN,
    VALID_TOKEN_COUNT_COLUMN,
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
from cppmega_mlx.data.fim import INSERTED_TOKEN_SOURCE_INDEX
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
OBJECTIVE_GRAPH_RELATION_COLUMNS = {
    "call": "token_call_edges",
    "type": "token_type_edges",
    "domain": "token_domain_edges",
    "build": "token_build_edges",
    "shell": "token_shell_edges",
    "diagnostic": "token_diagnostic_edges",
    "cross_domain": "token_cross_domain_edges",
}
OBJECTIVE_CHUNK_ROUTE_COLUMNS = (
    "token_chunk_starts",
    "token_chunk_ends",
    "token_chunk_kinds",
    "token_chunk_dep_levels",
)
OBJECTIVE_ROUTE_COLUMNS = (
    *OBJECTIVE_GRAPH_RELATION_COLUMNS.values(),
    *OBJECTIVE_CHUNK_ROUTE_COLUMNS,
)
OBJECTIVE_ROUTE_MAPPING_SCHEMA = "cppmega_exact_source_route_remap_v1"
OBJECTIVE_ROUTE_RECEIPT_SCHEMA = "cppmega_objective_route_receipt_v1"
OBJECTIVE_ROUTE_RETENTION_SCHEMA = "cppmega_objective_route_retention_v1"
OBJECTIVE_ROUTE_COUNT_FIELDS = (
    "source_edges",
    "retained_edges",
    "dropped_unmapped_edges",
    "dropped_noncausal_routes",
    "dropped_duplicate_routes",
    "excluded_edges",
)

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
    available = set(columns)
    required = set(OBJECTIVE_MEGATRON_REQUIRED_SOURCE_COLUMNS)
    if TOKEN_IDS_COLUMN not in available:
        if {INPUT_IDS_COLUMN, VALID_TOKEN_COUNT_COLUMN} <= available:
            required.remove(TOKEN_IDS_COLUMN)
        else:
            required.add(TOKEN_IDS_COLUMN)
    missing = sorted(required - available)
    if missing:
        raise ValueError(
            "Megatron objective materialization is missing required typed "
            "columns: " + ", ".join(missing)
        )


def normalize_megatron_objective_source_row(
    row: Mapping[str, Any],
    *,
    source_index: int,
) -> dict[str, Any]:
    """Adapt one fixed-length packed row to the typed objective-row contract."""

    normalized = dict(row)
    if TOKEN_IDS_COLUMN in normalized:
        return normalized
    if INPUT_IDS_COLUMN not in normalized or VALID_TOKEN_COUNT_COLUMN not in normalized:
        raise ValueError(
            "Megatron objective source requires token_ids or packed "
            "input_ids + valid_token_count"
        )

    packed_tokens = list(normalized[INPUT_IDS_COLUMN] or [])
    raw_valid = normalized[VALID_TOKEN_COUNT_COLUMN]
    if isinstance(raw_valid, bool):
        raise ValueError("valid_token_count must be an integer, not bool")
    valid = int(raw_valid)
    if not 0 < valid <= len(packed_tokens):
        raise ValueError(
            f"valid_token_count {valid} is outside packed input length "
            f"{len(packed_tokens)}"
        )

    for column in PACKED_ROWS_TOKEN_ALIGNED_COLUMNS:
        if column not in normalized:
            continue
        values = normalized[column]
        if values is None:
            raise ValueError(f"packed token-aligned column {column} is null")
        vector = list(values)
        if len(vector) != len(packed_tokens):
            raise ValueError(
                f"packed token-aligned column {column} length {len(vector)} != "
                f"input_ids length {len(packed_tokens)}"
            )
        normalized[column] = vector[:valid]
    normalized[TOKEN_IDS_COLUMN] = packed_tokens[:valid]

    packed_instructions = normalized.get(SOURCE_IFIM_INSTRUCTION_TOKEN_IDS_COLUMN)
    if packed_instructions is not None:
        instructions = list(packed_instructions)
        raw_num_docs = normalized.get(NUM_DOCS_COLUMN, len(instructions))
        if isinstance(raw_num_docs, bool):
            raise ValueError("num_docs must be an integer, not bool")
        num_docs = int(raw_num_docs)
        if num_docs != len(instructions):
            raise ValueError(
                f"{SOURCE_IFIM_INSTRUCTION_TOKEN_IDS_COLUMN} count "
                f"{len(instructions)} != num_docs {num_docs}"
            )
        # A packet-wide instruction cannot be bound to one constituent without
        # changing the CodePacket ABI. Preserve IFIM only when that binding is
        # exact; multi-document packs remain eligible for the other objectives.
        if num_docs == 1 and instructions[0]:
            normalized[IFIM_INSTRUCTION_TOKEN_IDS_COLUMN] = list(instructions[0])

    return normalized


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

    row = normalize_megatron_objective_source_row(row, source_index=source_index)
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
    packed_commit_packets = (
        build_commit_packets_from_packed_row(columns, row_index=0)
        if any(column in row for column in PACKED_ROWS_OBJECTIVE_SOURCE_COLUMNS)
        else []
    )
    if packed_commit_packets:
        commit_packet = packed_commit_packets[source_index % len(packed_commit_packets)]
    elif has_any_commit_section:
        commit_packet = build_commit_packet_from_row(
            columns=commit_columns,
            row_index=0,
        )
    else:
        commit_packet = None
    return ObjectiveSource(code_packet=code_packet, commit_packet=commit_packet)


def empty_objective_routes() -> dict[str, list[Any]]:
    """Return the complete empty route row used for explicit exclusions."""

    return {column: [] for column in OBJECTIVE_ROUTE_COLUMNS}


@dataclass(frozen=True)
class ObjectiveRouteRemap:
    columns: dict[str, list[Any]]
    receipt: dict[str, object]


def _empty_relation_counts(*, source_edges: int = 0) -> dict[str, int]:
    counts = {field: 0 for field in OBJECTIVE_ROUTE_COUNT_FIELDS}
    counts["source_edges"] = source_edges
    return counts


def _route_vector(packet: CodePacket, field: str, *, where: str) -> list[int] | None:
    value = getattr(packet, field)
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(
            f"{where}: CodePacket.{field} must be one-dimensional, got {array.shape}"
        )
    return [int(item) for item in array.tolist()]


def remap_objective_routes(
    packet: CodePacket,
    *,
    source_token_indices: Sequence[int],
    where: str,
    require_sidecars: bool = False,
    mode: str = "source_token_remap",
) -> ObjectiveRouteRemap:
    """Remap chunk spans and graph endpoints through an exact token transform.

    ``source_token_indices`` maps each output token to one source token, with
    ``-1`` reserved for synthetic markers. Source chunks are split whenever a
    transform makes their output positions or source positions discontinuous.
    Only causal edges with two mapped endpoints are emitted.
    """

    source_tokens = np.asarray(packet.token_ids)
    if source_tokens.ndim != 1:
        raise ValueError(
            f"{where}: route remapping requires one-dimensional CodePacket tokens, "
            f"got {source_tokens.shape}"
        )
    source_length = int(source_tokens.shape[0])
    source_map = [int(index) for index in source_token_indices]
    inverse: dict[int, int] = {}
    for output_index, source_index in enumerate(source_map):
        if source_index == INSERTED_TOKEN_SOURCE_INDEX:
            continue
        if not 0 <= source_index < source_length:
            raise ValueError(
                f"{where}: source-token route index {source_index} at output token "
                f"{output_index} is outside 0..{source_length - 1}"
            )
        if source_index in inverse:
            raise ValueError(
                f"{where}: source token {source_index} maps to multiple output tokens; "
                "graph routes would be ambiguous"
            )
        inverse[source_index] = output_index

    chunk_fields = {
        field: _route_vector(packet, field, where=where)
        for field in ("chunk_starts", "chunk_ends", "chunk_kinds", "chunk_dep_levels")
    }
    relation_fields = {
        **_CHUNK_GRAPH_FIELDS,
        **_TOKEN_GRAPH_FIELDS,
    }
    if require_sidecars:
        missing = [
            f"CodePacket.{field}"
            for field, values in chunk_fields.items()
            if values is None
        ]
        missing.extend(
            f"CodePacket.{field}"
            for field in relation_fields.values()
            if getattr(packet, field) is None
        )
        if missing:
            raise ValueError(
                f"{where}: production route remapping is missing required sidecars: "
                + ", ".join(missing)
            )

    present_chunk_fields = {
        field: values for field, values in chunk_fields.items() if values is not None
    }
    if present_chunk_fields and len(present_chunk_fields) != len(chunk_fields):
        missing = sorted(set(chunk_fields) - set(present_chunk_fields))
        raise ValueError(
            f"{where}: chunk route sidecars must be all present or all absent; "
            f"missing {missing}"
        )
    if not present_chunk_fields:
        for relation, field in _CHUNK_GRAPH_FIELDS.items():
            edge = getattr(packet, field)
            if edge is not None and edge.to_pairs():
                raise ValueError(
                    f"{where}: {relation} routes require complete chunk sidecars"
                )
        chunks: list[tuple[int, int, int, int, int]] = []
        source_chunk_count = 0
    else:
        starts = present_chunk_fields["chunk_starts"]
        ends = present_chunk_fields["chunk_ends"]
        kinds = present_chunk_fields["chunk_kinds"]
        dep_levels = present_chunk_fields["chunk_dep_levels"]
        assert starts is not None
        assert ends is not None
        assert kinds is not None
        assert dep_levels is not None
        lengths = {len(starts), len(ends), len(kinds), len(dep_levels)}
        if len(lengths) != 1:
            raise ValueError(
                f"{where}: chunk route sidecar lengths disagree: "
                f"starts={len(starts)}, ends={len(ends)}, kinds={len(kinds)}, "
                f"dep_levels={len(dep_levels)}"
            )
        source_chunk_count = len(starts)
        previous_end = 0
        for chunk_index, (start, end) in enumerate(zip(starts, ends, strict=True)):
            if not 0 <= start < end <= source_length:
                raise ValueError(
                    f"{where}: source chunk {chunk_index} span [{start}, {end}) is "
                    f"outside 0..{source_length}"
                )
            if chunk_index and start < previous_end:
                raise ValueError(
                    f"{where}: source chunk {chunk_index} starts at {start} before "
                    f"the previous chunk ends at {previous_end}"
                )
            previous_end = end

        chunks = []
        for chunk_index, (start, end, kind, dep_level) in enumerate(
            zip(starts, ends, kinds, dep_levels, strict=True)
        ):
            mapped = sorted(
                (inverse[source_index], source_index)
                for source_index in range(start, end)
                if source_index in inverse
            )
            if not mapped:
                continue
            fragment_start = mapped[0][0]
            previous_output, previous_source = mapped[0]
            for output_index, source_index in mapped[1:]:
                if (
                    output_index != previous_output + 1
                    or source_index != previous_source + 1
                ):
                    chunks.append(
                        (
                            fragment_start,
                            previous_output + 1,
                            int(kind),
                            int(dep_level),
                            chunk_index,
                        )
                    )
                    fragment_start = output_index
                previous_output = output_index
                previous_source = source_index
            chunks.append(
                (
                    fragment_start,
                    previous_output + 1,
                    int(kind),
                    int(dep_level),
                    chunk_index,
                )
            )
        chunks.sort(key=lambda chunk: (chunk[0], chunk[1], chunk[4]))
        for previous, current in zip(chunks, chunks[1:]):
            if (
                current[0] < previous[1]
            ):  # pragma: no cover - unique inverse proves this
                raise AssertionError(f"{where}: remapped chunk fragments overlap")
        if len(chunks) > np.iinfo(np.uint16).max + 1:
            raise ValueError(
                f"{where}: {len(chunks)} remapped chunks exceed uint16 graph endpoints"
            )

    result = empty_objective_routes()
    result["token_chunk_starts"] = [chunk[0] for chunk in chunks]
    result["token_chunk_ends"] = [chunk[1] for chunk in chunks]
    result["token_chunk_kinds"] = [chunk[2] for chunk in chunks]
    result["token_chunk_dep_levels"] = [chunk[3] for chunk in chunks]
    output_chunks_by_source: dict[int, list[int]] = {
        chunk_index: [] for chunk_index in range(source_chunk_count)
    }
    for output_chunk, chunk in enumerate(chunks):
        output_chunks_by_source[chunk[4]].append(output_chunk)

    output_starts = result["token_chunk_starts"]
    relation_receipts: dict[str, dict[str, int]] = {}
    for relation, field in _CHUNK_GRAPH_FIELDS.items():
        edge = getattr(packet, field)
        if edge is None:
            relation_receipts[relation] = _empty_relation_counts()
            continue
        source_pairs = edge.to_pairs()
        counts = _empty_relation_counts(source_edges=len(source_pairs))
        remapped: set[tuple[int, int]] = set()
        for source_chunk, destination_chunk in source_pairs:
            if not (
                0 <= source_chunk < source_chunk_count
                and 0 <= destination_chunk < source_chunk_count
            ):
                raise ValueError(
                    f"{where}: {relation} edge ({source_chunk}, {destination_chunk}) "
                    f"is outside {source_chunk_count} source chunks"
                )
            source_outputs = output_chunks_by_source[source_chunk]
            destination_outputs = output_chunks_by_source[destination_chunk]
            if not source_outputs or not destination_outputs:
                counts["dropped_unmapped_edges"] += 1
                continue
            for output_source in source_outputs:
                for output_destination in destination_outputs:
                    if (
                        output_starts[output_destination]
                        <= output_starts[output_source]
                    ):
                        route = (output_source, output_destination)
                        if route in remapped:
                            counts["dropped_duplicate_routes"] += 1
                        else:
                            remapped.add(route)
                    else:
                        counts["dropped_noncausal_routes"] += 1
        result[OBJECTIVE_GRAPH_RELATION_COLUMNS[relation]] = [
            {"from": source, "to": destination}
            for source, destination in sorted(remapped)
        ]
        counts["retained_edges"] = len(remapped)
        relation_receipts[relation] = counts

    for relation, field in _TOKEN_GRAPH_FIELDS.items():
        edge = getattr(packet, field)
        if edge is None:
            relation_receipts[relation] = _empty_relation_counts()
            continue
        source_triples = edge.to_triples()
        counts = _empty_relation_counts(source_edges=len(source_triples))
        remapped_triples: set[tuple[int, int, int]] = set()
        for source_token, destination_token, kind in source_triples:
            if not (
                0 <= source_token < source_length
                and 0 <= destination_token < source_length
            ):
                raise ValueError(
                    f"{where}: {relation} edge ({source_token}, {destination_token}) "
                    f"is outside {source_length} source tokens"
                )
            output_source = inverse.get(source_token)
            output_destination = inverse.get(destination_token)
            if output_source is None or output_destination is None:
                counts["dropped_unmapped_edges"] += 1
                continue
            if output_destination <= output_source:
                route = (output_source, output_destination, int(kind))
                if route in remapped_triples:
                    counts["dropped_duplicate_routes"] += 1
                else:
                    remapped_triples.add(route)
            else:
                counts["dropped_noncausal_routes"] += 1
        result[OBJECTIVE_GRAPH_RELATION_COLUMNS[relation]] = [
            {"from": source, "to": destination, "kind": kind}
            for source, destination, kind in sorted(remapped_triples)
        ]
        counts["retained_edges"] = len(remapped_triples)
        relation_receipts[relation] = counts
    return ObjectiveRouteRemap(
        columns=result,
        receipt={
            "schema": OBJECTIVE_ROUTE_RECEIPT_SCHEMA,
            "mode": mode,
            "source_chunks": source_chunk_count,
            "retained_chunks": len(chunks),
            "dropped_chunks": sum(
                not output_chunks for output_chunks in output_chunks_by_source.values()
            ),
            "excluded_chunks": 0,
            "relations": relation_receipts,
        },
    )


def exclude_objective_routes(
    packet: CodePacket | None,
    *,
    where: str,
    reason: str,
    require_sidecars: bool = False,
) -> ObjectiveRouteRemap:
    """Explicitly exclude routes when an objective has no exact source-token map."""

    if packet is None:
        source_chunks = 0
        source_relations = {
            relation: _empty_relation_counts()
            for relation in OBJECTIVE_GRAPH_RELATION_COLUMNS
        }
    else:
        identity = remap_objective_routes(
            packet,
            source_token_indices=range(int(packet.token_ids.shape[0])),
            where=where,
            require_sidecars=require_sidecars,
            mode="identity_validation",
        )
        source_chunks = int(identity.receipt["source_chunks"])
        raw_relations = identity.receipt["relations"]
        assert isinstance(raw_relations, Mapping)
        source_relations = {}
        for relation in OBJECTIVE_GRAPH_RELATION_COLUMNS:
            raw_counts = raw_relations[relation]
            assert isinstance(raw_counts, Mapping)
            source_edges = int(raw_counts["source_edges"])
            counts = _empty_relation_counts(source_edges=source_edges)
            counts["excluded_edges"] = source_edges
            source_relations[relation] = counts
    return ObjectiveRouteRemap(
        columns=empty_objective_routes(),
        receipt={
            "schema": OBJECTIVE_ROUTE_RECEIPT_SCHEMA,
            "mode": "excluded",
            "reason": reason,
            "source_chunks": source_chunks,
            "retained_chunks": 0,
            "dropped_chunks": 0,
            "excluded_chunks": source_chunks,
            "relations": source_relations,
        },
    )


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
    "OBJECTIVE_CHUNK_ROUTE_COLUMNS",
    "OBJECTIVE_GRAPH_RELATION_COLUMNS",
    "OBJECTIVE_SECTION_COLUMNS",
    "OBJECTIVE_SOURCE_COLUMNS",
    "OBJECTIVE_REQUIRED_SOURCE_COLUMNS",
    "OBJECTIVE_MEGATRON_REQUIRED_SOURCE_COLUMNS",
    "OBJECTIVE_ROUTE_COLUMNS",
    "OBJECTIVE_ROUTE_COUNT_FIELDS",
    "OBJECTIVE_ROUTE_MAPPING_SCHEMA",
    "OBJECTIVE_ROUTE_RECEIPT_SCHEMA",
    "OBJECTIVE_ROUTE_RETENTION_SCHEMA",
    "ObjectiveRouteRemap",
    "empty_objective_routes",
    "exclude_objective_routes",
    "graph_targets_and_pair_mask",
    "normalize_megatron_objective_source_row",
    "objective_source_from_tokenized_row",
    "remap_objective_routes",
    "require_megatron_objective_source_columns",
    "require_objective_source_columns",
    "token_graph_for_aligned_example",
]

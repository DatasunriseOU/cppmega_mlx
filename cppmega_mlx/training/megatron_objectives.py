"""Pre-materialize realized objectives for Megatron's causal dataset API."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cppmega_mlx.data.code_packet import CodePacket
from cppmega_mlx.data.domain_schema import (
    DOMAIN_DELIMITER_ROLES,
    DomainKind,
    DomainRoleKind,
    ParseConfidence,
)
from cppmega_mlx.data.fim import INSERTED_TOKEN_SOURCE_INDEX
from cppmega_mlx.data.source_identity import (
    MAX_ROW_LOCAL_DOC_ID,
    MAX_SOURCE_ID,
    validate_source_identity_registry,
)
from cppmega_mlx.data.symbol_identity import (
    SYMBOL_IDENTITIES_COLUMN,
    SymbolIdentityRegistry,
)
from cppmega_mlx.data.tokenizer_contract import DOMAIN_DELIMITER_TOKEN_IDS
from cppmega_mlx.training.objective_mixer import (
    GraphAuxLossConfig,
    ObjectiveSource,
    RealizedObjective,
)
from cppmega_mlx.training.objective_data import (
    OBJECTIVE_CHUNK_ROUTE_COLUMNS,
    OBJECTIVE_GRAPH_RELATION_COLUMNS,
    OBJECTIVE_ROUTE_MAPPING_SCHEMA,
    OBJECTIVE_ROUTE_RETENTION_SCHEMA,
    exclude_objective_routes,
    remap_objective_routes,
)
from cppmega_mlx.training.objectives import SOURCE_TOKEN_INDICES_METADATA_KEY
from cppmega_mlx.training.task_mixer import TaskKind

OBJECTIVE_CONTRACT_SCHEMA = "cppmega_pre_materialized_objectives_v1"
OBJECTIVE_MATERIALIZATION_ARTIFACT_SCHEMA = (
    "cppmega_objective_materialization_artifact_v1"
)
OBJECTIVE_MATERIALIZATION_ARTIFACT_NAME = "objective_materialization.json"
OBJECTIVE_TOKEN_SIDE_CHANNELS: tuple[tuple[str, str], ...] = (
    ("loss_mask", "uint8"),
    ("doc_ids", "uint32"),
    ("token_domain_ids", "uint16"),
    ("token_role_ids", "uint16"),
    ("token_entity_ids", "uint32"),
    ("token_scope_ids", "uint32"),
    ("token_source_doc_ids", "uint32"),
    ("token_source_identity_ids", "uint64"),
    ("token_confidence_ids", "uint8"),
    ("token_structure_ids", "uint8"),
    ("token_dep_levels", "uint16"),
    ("token_ast_depth", "uint16"),
    ("token_sibling_index", "uint16"),
    ("token_ast_node_type", "uint16"),
    ("token_symbol_ids", "uint64"),
    ("token_call_targets", "uint64"),
    ("token_type_refs", "uint64"),
    ("token_def_use", "uint8"),
    ("token_change_mask_pre", "uint8"),
    ("token_change_mask_post", "uint8"),
)
OBJECTIVE_GRAPH_SIDECARS: tuple[tuple[str, str, str], ...] = (
    ("token_call_edges", "edge_pairs", "int32"),
    ("token_type_edges", "edge_pairs", "int32"),
    ("token_domain_edges", "edge_triples", "int32"),
    ("token_build_edges", "edge_triples", "int32"),
    ("token_shell_edges", "edge_triples", "int32"),
    ("token_diagnostic_edges", "edge_triples", "int32"),
    ("token_cross_domain_edges", "edge_triples", "int32"),
    ("token_chunk_starts", "ragged_1d", "uint32"),
    ("token_chunk_ends", "ragged_1d", "uint32"),
    ("token_chunk_kinds", "ragged_1d", "uint8"),
    ("token_chunk_dep_levels", "ragged_1d", "uint16"),
)
OBJECTIVE_KIND_IDS = {
    TaskKind.CAUSAL_LM.value: 1,
    TaskKind.FIM.value: 2,
    TaskKind.AST_FIM.value: 3,
    TaskKind.IFIM.value: 4,
    TaskKind.COMMIT_DIFF.value: 5,
    TaskKind.PRE_TO_POST.value: 6,
    TaskKind.SYMBOL_RECOVERY.value: 7,
    TaskKind.TYPE_RECOVERY.value: 8,
    TaskKind.CALLEE_RECOVERY.value: 9,
}
_ALIGNED_TASKS = frozenset(
    {
        TaskKind.CAUSAL_LM,
    }
)
_TRANSFORMED_CODE_TASKS = frozenset(
    {
        TaskKind.FIM,
        TaskKind.AST_FIM,
        TaskKind.IFIM,
        TaskKind.SYMBOL_RECOVERY,
        TaskKind.TYPE_RECOVERY,
        TaskKind.CALLEE_RECOVERY,
    }
)
_COMMIT_TASKS = frozenset({TaskKind.COMMIT_DIFF, TaskKind.PRE_TO_POST})

_TOKEN_SIDECARS = {
    "token_structure_ids": "structure_ids",
    "token_dep_levels": "dep_levels",
    "token_ast_depth": "ast_depth",
    "token_sibling_index": "sibling_index",
    "token_ast_node_type": "ast_node_type",
    "token_symbol_ids": "symbol_ids",
    "token_call_targets": "call_targets",
    "token_type_refs": "type_refs",
    "token_def_use": "def_use",
    "token_domain_ids": "domain_ids",
    "token_role_ids": "role_ids",
    "token_entity_ids": "entity_ids",
    "token_scope_ids": "scope_ids",
    "token_source_doc_ids": "source_doc_ids",
    "token_source_identity_ids": "source_identity_ids",
    "token_confidence_ids": "confidence_ids",
}
SOURCE_IDENTITY_REGISTRY_COLUMN = "source_identity_registry"
_GRAPH_RELATION_COLUMNS = dict(OBJECTIVE_GRAPH_RELATION_COLUMNS)
_DOMAIN_DELIMITER_BY_ID: dict[int, tuple[int, str, int]] = {}
for _domain, (_start_role, _end_role) in DOMAIN_DELIMITER_ROLES.items():
    _start_id = int(DOMAIN_DELIMITER_TOKEN_IDS[_start_role])
    _end_id = int(DOMAIN_DELIMITER_TOKEN_IDS[_end_role])
    _DOMAIN_DELIMITER_BY_ID[_start_id] = (int(_domain), "start", _end_id)
    _DOMAIN_DELIMITER_BY_ID[_end_id] = (int(_domain), "end", _end_id)


def objective_route_mapping_contract() -> dict[str, object]:
    """Return the converter-visible policy for graph routes after transforms."""

    return {
        "schema": OBJECTIVE_ROUTE_MAPPING_SCHEMA,
        "source_token_sentinel": INSERTED_TOKEN_SOURCE_INDEX,
        "identity_objectives": [TaskKind.CAUSAL_LM.value],
        "source_token_remap_objectives": [
            task.value for task in TaskKind if task in _TRANSFORMED_CODE_TASKS
        ],
        "explicit_exclusions": {
            TaskKind.COMMIT_DIFF.value: (
                "independently_tokenized_commit_sections_have_no_exact_source_map"
            ),
            TaskKind.PRE_TO_POST.value: (
                "independently_tokenized_commit_sections_have_no_exact_source_map"
            ),
        },
        "chunk_fragmentation": "split_output_and_source_discontinuities_v1",
        "edge_filter": "mapped_endpoints_causal_order_v1",
        "retained_relation_columns": dict(OBJECTIVE_GRAPH_RELATION_COLUMNS),
        "retained_chunk_columns": list(OBJECTIVE_CHUNK_ROUTE_COLUMNS),
    }


def _ints(value: Any, *, where: str) -> list[int]:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{where} must be one-dimensional, got {array.shape}")
    return [int(item) for item in array.tolist()]


def _packet_vector(packet: CodePacket, field: str, *, length: int) -> list[int]:
    value = getattr(packet, field)
    if value is None:
        raise ValueError(
            f"pre-materialized aligned objective requires CodePacket.{field}"
        )
    values = _ints(value, where=f"CodePacket.{field}")
    if len(values) != length:
        raise ValueError(
            f"CodePacket.{field} length {len(values)} != materialized token "
            f"length {length}"
        )
    return values


def _optional_metadata_vector(
    packet: CodePacket, field: str, *, length: int
) -> list[int]:
    value = packet.metadata.get(field)
    if value is None:
        return [0] * length
    values = _ints(value, where=f"CodePacket.metadata[{field!r}]")
    if not values:
        return [0] * length
    if len(values) != length:
        raise ValueError(
            f"typed metadata {field} length {len(values)} != materialized "
            f"token length {length}"
        )
    return values


def _validated_source_token_indices(
    *,
    metadata: Mapping[str, object],
    packet: CodePacket,
    transformed_tokens: Sequence[int],
    objective_kind: str,
) -> list[int]:
    raw_indices = metadata.get(SOURCE_TOKEN_INDICES_METADATA_KEY)
    if not isinstance(raw_indices, (tuple, list)):
        raise ValueError(
            f"{objective_kind}: missing exact {SOURCE_TOKEN_INDICES_METADATA_KEY}"
        )
    source_indices = [int(index) for index in raw_indices]
    if len(source_indices) != len(transformed_tokens):
        raise ValueError(
            f"{objective_kind}: source-token map length {len(source_indices)} != "
            f"transformed token length {len(transformed_tokens)}"
        )
    source_tokens = _ints(packet.token_ids, where="CodePacket.token_ids")
    for output_index, source_index in enumerate(source_indices):
        if source_index == INSERTED_TOKEN_SOURCE_INDEX:
            continue
        if not 0 <= source_index < len(source_tokens):
            raise ValueError(
                f"{objective_kind}: source-token map index {source_index} at "
                f"output token {output_index} is outside {len(source_tokens)} tokens"
            )
        if int(transformed_tokens[output_index]) != source_tokens[source_index]:
            raise ValueError(
                f"{objective_kind}: transformed token {output_index} does not match "
                f"CodePacket token {source_index}"
            )

    raw_span = metadata.get("source_document_span")
    if not isinstance(raw_span, (tuple, list)) or len(raw_span) != 2:
        raise ValueError(f"{objective_kind}: source_document_span must be a pair")
    start, end = (int(value) for value in raw_span)
    if not 0 <= start < end <= len(source_tokens):
        raise ValueError(
            f"{objective_kind}: source_document_span [{start}, {end}) is outside "
            f"{len(source_tokens)} CodePacket tokens"
        )
    mapped = sorted(
        index for index in source_indices if index != INSERTED_TOKEN_SOURCE_INDEX
    )
    if mapped != list(range(start, end)):
        raise ValueError(
            f"{objective_kind}: source-token map must cover source_document_span "
            "exactly once"
        )
    return source_indices


def _source_vector(
    packet: CodePacket,
    field: str,
    *,
    source_length: int,
    required: bool,
) -> list[int] | None:
    value = getattr(packet, field)
    if value is None:
        if required:
            raise ValueError(
                "pre-materialized production transformed objective requires "
                f"CodePacket.{field}"
            )
        return None
    return _packet_vector(packet, field, length=source_length)


def _permute_source_vector(
    source_values: Sequence[int] | None,
    source_indices: Sequence[int],
    *,
    inserted_value: int = 0,
) -> list[int]:
    if source_values is None:
        return [inserted_value] * len(source_indices)
    return [
        inserted_value
        if source_index == INSERTED_TOKEN_SOURCE_INDEX
        else int(source_values[source_index])
        for source_index in source_indices
    ]


def _permute_provenance_vector(
    source_values: Sequence[int],
    source_indices: Sequence[int],
    *,
    where: str,
) -> list[int]:
    """Copy mapped values and assign insertions to their nearest source token."""

    result: list[int | None] = [
        None
        if source_index == INSERTED_TOKEN_SOURCE_INDEX
        else int(source_values[source_index])
        for source_index in source_indices
    ]
    mapped_positions = [
        index for index, value in enumerate(result) if value is not None
    ]
    if not mapped_positions:
        raise ValueError(f"{where}: source-token map contains no original tokens")
    previous: list[int | None] = [None] * len(result)
    next_position: list[int | None] = [None] * len(result)
    last: int | None = None
    for index, value in enumerate(result):
        if value is not None:
            last = index
        previous[index] = last
    last = None
    for index in range(len(result) - 1, -1, -1):
        if result[index] is not None:
            last = index
        next_position[index] = last
    for index, value in enumerate(result):
        if value is not None:
            continue
        left = previous[index]
        right = next_position[index]
        if right is None or (left is not None and index - left <= right - index):
            source_position = left
        else:
            source_position = right
        if source_position is None:  # pragma: no cover - mapped_positions proves this
            raise AssertionError(f"{where}: cannot resolve inserted-token provenance")
        result[index] = result[source_position]
    return [int(value) for value in result if value is not None]


def _contiguous_document_id_order(values: Sequence[int], *, where: str) -> list[int]:
    if not values:
        raise ValueError(f"{where}: doc_ids must be non-empty")
    order: list[int] = []
    closed: set[int] = set()
    previous: int | None = None
    for index, raw_value in enumerate(values):
        value = int(raw_value)
        if value <= 0:
            raise ValueError(f"{where}: doc_ids[{index}] must be positive, got {value}")
        if value != previous:
            if value in closed:
                raise ValueError(
                    f"{where}: document ID {value} is reused non-contiguously"
                )
            if previous is not None:
                closed.add(previous)
            order.append(value)
            previous = value
    return order


def _normalize_attention_document_ids(
    values: Sequence[int], *, where: str
) -> list[int]:
    """Renumber contiguous attention segments to row-local IDs ``1..N``."""

    order = _contiguous_document_id_order(values, where=where)
    remap = {document_id: index + 1 for index, document_id in enumerate(order)}
    normalized = [remap[int(value)] for value in values]
    return normalized


def _source_platform_bags_by_document(
    packet: CodePacket,
    document_ids: Sequence[int],
    *,
    required: bool,
) -> dict[int, list[int]]:
    order = _contiguous_document_id_order(
        document_ids,
        where="CodePacket.document_ids",
    )
    raw_bags = packet.metadata.get("source_platform_ids")
    if raw_bags is None:
        raw_union = packet.metadata.get("platform_ids")
        union = [] if raw_union is None else _ints(raw_union, where="platform_ids")
        if not union:
            if required:
                raise ValueError(
                    "pre-materialized production objective requires source_platform_ids"
                )
            return {}
        if len(order) != 1 and required:
            raise ValueError(
                "pre-materialized production multi-document objective requires "
                "exact source_platform_ids bags"
            )
        raw_bags = [union for _ in order]
    if not isinstance(raw_bags, (list, tuple)) or len(raw_bags) != len(order):
        raise ValueError(
            "source_platform_ids count must equal the number of source documents: "
            f"bags={0 if not isinstance(raw_bags, (list, tuple)) else len(raw_bags)}, "
            f"documents={len(order)}"
        )
    result: dict[int, list[int]] = {}
    for document_id, raw_bag in zip(order, raw_bags, strict=True):
        if not isinstance(raw_bag, (list, tuple)):
            raise ValueError("every source_platform_ids bag must be a list")
        bag = _ints(raw_bag, where=f"source_platform_ids[{document_id}]")
        if required and not bag:
            raise ValueError("production source_platform_ids bags must be non-empty")
        if any(not 0 < value <= np.iinfo(np.uint16).max for value in bag):
            raise ValueError("source_platform_ids values must be positive uint16")
        result[document_id] = bag
    return result


def _platform_bags_for_output_documents(
    document_ids: Sequence[int],
    source_bags: Mapping[int, Sequence[int]],
    *,
    required: bool,
    where: str,
) -> list[list[int]]:
    order = _contiguous_document_id_order(document_ids, where=where)
    if not source_bags:
        if required:
            raise ValueError(f"{where}: source platform bags are missing")
        return []
    missing = [document_id for document_id in order if document_id not in source_bags]
    if missing:
        raise ValueError(f"{where}: source platform bags missing documents {missing}")
    return [[int(value) for value in source_bags[document_id]] for document_id in order]


def _domain_stack_defaults(
    token_ids: Sequence[int],
    *,
    where: str,
) -> tuple[list[int], list[int], list[int]]:
    domains: list[int] = []
    roles: list[int] = []
    confidence: list[int] = []
    stack: list[tuple[int, int]] = []
    for index, token_id in enumerate(token_ids):
        marker = _DOMAIN_DELIMITER_BY_ID.get(int(token_id))
        if marker is None:
            domains.append(stack[-1][0] if stack else int(DomainKind.UNKNOWN))
            roles.append(int(DomainRoleKind.NONE))
            confidence.append(int(ParseConfidence.ABSENT))
            continue
        domain_id, direction, expected_close = marker
        domains.append(domain_id)
        roles.append(int(DomainRoleKind.DELIMITER))
        confidence.append(int(ParseConfidence.EXACT))
        if direction == "start":
            stack.append((domain_id, expected_close))
        elif not stack or stack[-1] != (domain_id, int(token_id)):
            raise ValueError(
                f"{where}: mismatched domain closer {token_id} at token {index}"
            )
        else:
            stack.pop()
    if stack:
        raise ValueError(f"{where}: unclosed domain delimiter stack {stack}")
    return domains, roles, confidence


def _bind_domain_stack_sidecars(
    token_ids: Sequence[int],
    source_indices: Sequence[int],
    source_domain_ids: Sequence[int] | None,
    source_role_ids: Sequence[int] | None,
    source_confidence_ids: Sequence[int] | None,
    *,
    where: str,
) -> tuple[list[int], list[int], list[int]]:
    """Bind output domains to delimiters while retaining mapped annotations.

    UNKNOWN source domains inherit the active output wrapper. Any non-zero
    source domain must already agree with that wrapper.
    """

    if len(token_ids) != len(source_indices):
        raise ValueError(f"{where}: source-token map must be token aligned")
    domains, roles, confidence = _domain_stack_defaults(token_ids, where=where)
    source_vectors = (
        ("domain_ids", source_domain_ids),
        ("role_ids", source_role_ids),
        ("confidence_ids", source_confidence_ids),
    )
    for output_index, source_index in enumerate(source_indices):
        if source_index == INSERTED_TOKEN_SOURCE_INDEX:
            continue
        if source_index < 0:
            raise ValueError(
                f"{where}: invalid source-token map index {source_index} at "
                f"output token {output_index}"
            )
        for field, values in source_vectors:
            if values is not None and not 0 <= source_index < len(values):
                raise ValueError(
                    f"{where}: source {field} index {source_index} at output token "
                    f"{output_index} is outside {len(values)} values"
                )
        marker = _DOMAIN_DELIMITER_BY_ID.get(int(token_ids[output_index]))
        if marker is not None:
            expected_domain, _direction, _expected_close = marker
            if (
                source_domain_ids is not None
                and int(source_domain_ids[source_index]) != expected_domain
            ):
                raise ValueError(
                    f"{where}: source delimiter {token_ids[output_index]} requires "
                    f"domain {expected_domain}, got "
                    f"{source_domain_ids[source_index]}"
                )
            if source_role_ids is not None and int(
                source_role_ids[source_index]
            ) != int(DomainRoleKind.DELIMITER):
                raise ValueError(
                    f"{where}: source delimiter {token_ids[output_index]} requires "
                    "DELIMITER role"
                )
            if source_confidence_ids is not None and int(
                source_confidence_ids[source_index]
            ) != int(ParseConfidence.EXACT):
                raise ValueError(
                    f"{where}: source delimiter {token_ids[output_index]} requires "
                    "EXACT confidence"
                )
            continue
        if source_domain_ids is not None:
            source_domain = int(source_domain_ids[source_index])
            expected_domain = domains[output_index]
            if source_domain not in {int(DomainKind.UNKNOWN), expected_domain}:
                raise ValueError(
                    f"{where}: source token {source_index} has domain "
                    f"{source_domain}, incompatible with active output domain "
                    f"{expected_domain}"
                )
        if source_role_ids is not None:
            roles[output_index] = int(source_role_ids[source_index])
        if source_confidence_ids is not None:
            confidence[output_index] = int(source_confidence_ids[source_index])
    return domains, roles, confidence


def _validate_domain_stack_sidecars(
    token_ids: Sequence[int],
    domain_ids: Sequence[int],
    role_ids: Sequence[int],
    confidence_ids: Sequence[int],
    *,
    where: str,
) -> None:
    if not (len(token_ids) == len(domain_ids) == len(role_ids) == len(confidence_ids)):
        raise ValueError(f"{where}: domain sidecars must be token aligned")
    stack: list[tuple[int, int]] = []
    for index, (token_id, domain_id, role_id, confidence_id) in enumerate(
        zip(token_ids, domain_ids, role_ids, confidence_ids, strict=True)
    ):
        marker = _DOMAIN_DELIMITER_BY_ID.get(int(token_id))
        if marker is None:
            expected_domain = stack[-1][0] if stack else int(DomainKind.UNKNOWN)
            if int(domain_id) != expected_domain:
                raise ValueError(
                    f"{where}: token {index} requires domain {expected_domain}, "
                    f"got {domain_id}"
                )
            if int(role_id) == int(DomainRoleKind.DELIMITER):
                raise ValueError(
                    f"{where}: non-delimiter token {index} has DELIMITER role"
                )
            continue
        expected_domain, direction, expected_close = marker
        if int(domain_id) != expected_domain:
            raise ValueError(
                f"{where}: delimiter {token_id} requires domain {expected_domain}, "
                f"got {domain_id}"
            )
        if int(role_id) != int(DomainRoleKind.DELIMITER):
            raise ValueError(f"{where}: delimiter {token_id} requires DELIMITER role")
        if int(confidence_id) != int(ParseConfidence.EXACT):
            raise ValueError(f"{where}: delimiter {token_id} requires EXACT confidence")
        if direction == "start":
            stack.append((expected_domain, expected_close))
        elif not stack or stack[-1] != (expected_domain, int(token_id)):
            raise ValueError(
                f"{where}: mismatched domain closer {token_id} at token {index}"
            )
        else:
            stack.pop()
    if stack:
        raise ValueError(f"{where}: unclosed domain delimiter stack {stack}")


@dataclass(frozen=True)
class MaterializedMegatronDocument:
    objective_kind: str
    token_ids: list[int]
    loss_mask: list[int]
    graph_edge_count: int
    row: dict[str, object]
    route_receipt: Mapping[str, object] | None = None


def _source_identity_registry_for_ids(
    packet: CodePacket,
    source_identity_ids: Sequence[int],
    *,
    required: bool,
) -> list[dict[str, int | str]]:
    referenced = sorted({int(value) for value in source_identity_ids if int(value) > 0})
    raw_registry = packet.metadata.get(SOURCE_IDENTITY_REGISTRY_COLUMN)
    if raw_registry is None:
        if required:
            raise ValueError(
                "pre-materialized production objective requires "
                "CodePacket.metadata['source_identity_registry']"
            )
        return []
    if not isinstance(raw_registry, Sequence) or isinstance(raw_registry, (str, bytes)):
        raise ValueError("source_identity_registry must be a sequence of records")
    validated = validate_source_identity_registry(
        raw_registry,
        referenced_ids=referenced,
    )
    return [validated[identity_id].as_dict() for identity_id in referenced]


def _symbol_identity_records(packet: CodePacket) -> list[dict[str, object]]:
    raw_records = packet.metadata.get(SYMBOL_IDENTITIES_COLUMN, [])
    registry = SymbolIdentityRegistry()
    registry.register_records(raw_records, source="pre-materialized objective")
    used_ids: set[int] = set()
    for field in ("symbol_ids", "call_targets", "type_refs"):
        value = getattr(packet, field)
        if value is not None:
            used_ids.update(
                item for item in _ints(value, where=f"CodePacket.{field}") if item != 0
            )
    registry.require_ids(used_ids, source="pre-materialized objective")
    return registry.records(used_ids)


def _canonical_contract_sha256(contract: Mapping[str, object]) -> str:
    encoded = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_objective_materialization_artifact(
    output_dir: str | Path,
    *,
    contract: Mapping[str, object],
    parquet_paths: Sequence[str | Path],
) -> Path:
    """Write the canonical CASE1-to-Megatron materialization handoff."""

    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    payload = dict(contract)
    if payload.get("schema") != OBJECTIVE_CONTRACT_SCHEMA:
        raise ValueError(
            f"objective contract schema must be {OBJECTIVE_CONTRACT_SCHEMA!r}"
        )
    totals = payload.get("totals")
    if not isinstance(totals, Mapping):
        raise ValueError("objective contract totals must be an object")
    documents = totals.get("samples")
    if isinstance(documents, bool) or not isinstance(documents, int) or documents < 1:
        raise ValueError("objective contract totals.samples must be a positive integer")

    materialization = payload.get("materialization")
    expected_materialization = {
        "format": "shifted_lm_document_v1",
        "token_column": "input_ids",
        "loss_mask_column": "loss_mask",
        "length_column": "valid_token_count",
        "objective_column": "objective_kind",
        "document_id_column": "doc_ids",
        "source_document_id_column": "token_source_doc_ids",
    }
    if materialization != expected_materialization:
        raise ValueError("objective contract materialization semantics drifted")
    graph = payload.get("graph_auxiliary")
    if not isinstance(graph, Mapping):
        raise ValueError("objective contract graph_auxiliary must be an object")
    pair_mask = graph.get("pair_mask")
    expansion = graph.get("chunk_edge_expansion")
    if pair_mask != "causal_same_document_upstream_v1":
        raise ValueError("objective contract graph pair mask semantics drifted")
    if expansion != "cartesian_token_spans_v1":
        raise ValueError("objective contract graph span expansion semantics drifted")
    if graph.get("route_mapping") != objective_route_mapping_contract():
        raise ValueError("objective contract route remapping semantics drifted")
    route_retention = graph.get("route_retention")
    if (
        not isinstance(route_retention, Mapping)
        or route_retention.get("schema") != OBJECTIVE_ROUTE_RETENTION_SCHEMA
        or not isinstance(route_retention.get("by_objective"), Mapping)
        or not route_retention["by_objective"]
    ):
        raise ValueError("objective contract route retention receipt is invalid")
    relations = graph.get("relations")
    if (
        not isinstance(relations, list)
        or not relations
        or not all(isinstance(relation, str) and relation for relation in relations)
    ):
        raise ValueError("objective contract graph relations must be non-empty strings")

    resolved_shards = sorted(Path(path).resolve() for path in parquet_paths)
    if not resolved_shards or len(set(resolved_shards)) != len(resolved_shards):
        raise ValueError("objective materialization needs unique parquet shards")
    for path in resolved_shards:
        if path.parent != root or path.suffix != ".parquet" or not path.is_file():
            raise ValueError(
                "objective parquet shards must be existing .parquet files directly "
                "inside output_dir"
            )
    existing_shards = sorted(root.glob("*.parquet"))
    if existing_shards != resolved_shards:
        unbound = sorted(
            path.name for path in set(existing_shards) - set(resolved_shards)
        )
        raise ValueError(f"output_dir contains unbound parquet shards: {unbound}")

    contract_path = root / "objective_contract.json"
    contract_tmp = contract_path.with_suffix(".json.tmp")
    contract_tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    contract_tmp.replace(contract_path)
    shard_records = [
        {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in resolved_shards
    ]
    converter = {
        "split": "all",
        "token_column": "input_ids",
        "length_column": "valid_token_count",
        "side_channels": [
            {"column": column, "dtype": dtype}
            for column, dtype in OBJECTIVE_TOKEN_SIDE_CHANNELS
        ],
        "graph_sidecars": [
            {"column": column, "kind": kind, "dtype": dtype}
            for column, kind, dtype in OBJECTIVE_GRAPH_SIDECARS
        ],
        "source_platform_sidecar": "require",
        "graph_relations": list(relations),
        "graph_pair_mask": pair_mask,
        "chunk_edge_expansion": expansion,
    }
    artifact = {
        "schema": OBJECTIVE_MATERIALIZATION_ARTIFACT_SCHEMA,
        "documents": documents,
        "objective_contract": {
            "path": contract_path.name,
            "sha256": _canonical_contract_sha256(payload),
            "size_bytes": contract_path.stat().st_size,
            "file_sha256": _file_sha256(contract_path),
        },
        "parquet_shards": shard_records,
        "converter": converter,
    }
    artifact["artifact_set_sha256"] = _canonical_contract_sha256(artifact)
    artifact_path = root / OBJECTIVE_MATERIALIZATION_ARTIFACT_NAME
    artifact_tmp = artifact_path.with_suffix(".json.tmp")
    artifact_tmp.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifact_tmp.replace(artifact_path)
    return artifact_path


def materialize_megatron_document(
    realized: RealizedObjective,
    source: ObjectiveSource,
    *,
    require_production_sidecars: bool = False,
) -> MaterializedMegatronDocument:
    """Convert aligned ObjectiveExample tensors to one shifted-LM document.

    Megatron derives labels by shifting a token document. The only lossless
    representation is therefore ``[input_ids[0], *target_ids]``; this function
    first proves every intermediate target agrees with the next input token.
    """

    example = realized.example
    inputs = _ints(example.input_ids, where=f"{realized.task.value}.input_ids")
    targets = _ints(example.target_ids, where=f"{realized.task.value}.target_ids")
    mask = _ints(example.loss_mask, where=f"{realized.task.value}.loss_mask")
    if not inputs or len(inputs) != len(targets) or len(mask) != len(inputs):
        raise ValueError(
            f"{realized.task.value}: input/target/loss vectors must have one "
            f"positive shared length, got {len(inputs)}/{len(targets)}/{len(mask)}"
        )
    if inputs[1:] != targets[:-1]:
        raise ValueError(
            f"{realized.task.value}: objective is not a shifted-LM sequence; "
            "target_ids[:-1] must equal input_ids[1:]"
        )
    if any(value not in (0, 1) for value in mask) or not any(mask):
        raise ValueError(
            f"{realized.task.value}: loss_mask must be binary with at least one "
            "trained token"
        )
    tokens = [inputs[0], *targets]
    materialized_mask = [*mask, 0]
    aligned = realized.task in _ALIGNED_TASKS
    packet = source.code_packet
    if aligned and packet is None:
        raise ValueError(
            f"{realized.task.value}: aligned objective requires ObjectiveSource.code_packet"
        )

    row: dict[str, object] = {
        "input_ids": tokens,
        "loss_mask": materialized_mask,
        "valid_token_count": len(tokens),
        "objective_kind": realized.task.value,
        "doc_ids": [1] * len(tokens),
        "source_platform_ids": [],
    }
    route_receipt: Mapping[str, object] | None = None
    packet_document_ids: list[int] | None = None
    source_platform_bags: dict[int, list[int]] = {}
    if packet is not None:
        packet_length = int(packet.token_ids.shape[0])
        packet_document_ids = (
            [1] * packet_length
            if packet.document_ids is None
            else _packet_vector(packet, "document_ids", length=packet_length)
        )
        source_platform_bags = _source_platform_bags_by_document(
            packet,
            packet_document_ids,
            required=require_production_sidecars,
        )

    if aligned:
        assert packet is not None
        assert packet_document_ids is not None
        packet_tokens = _ints(packet.token_ids, where="CodePacket.token_ids")
        if packet_tokens != tokens:
            raise ValueError(
                f"{realized.task.value}: aligned materialized tokens differ from "
                "CodePacket.token_ids"
            )
        row["doc_ids"] = list(packet_document_ids)
        row["source_platform_ids"] = _platform_bags_for_output_documents(
            row["doc_ids"],  # type: ignore[arg-type]
            source_platform_bags,
            required=require_production_sidecars,
            where=f"{realized.task.value}.doc_ids",
        )
        for column, field in _TOKEN_SIDECARS.items():
            if column == "token_source_identity_ids" and getattr(packet, field) is None:
                if require_production_sidecars:
                    raise ValueError(
                        "pre-materialized production objective requires "
                        "CodePacket.source_identity_ids"
                    )
                row[column] = [0] * len(tokens)
            else:
                row[column] = _packet_vector(packet, field, length=len(tokens))
        domains, roles, confidence = _bind_domain_stack_sidecars(
            tokens,
            list(range(len(tokens))),
            row["token_domain_ids"],
            row["token_role_ids"],
            row["token_confidence_ids"],
            where=realized.task.value,
        )
        row["token_domain_ids"] = domains
        row["token_role_ids"] = roles
        row["token_confidence_ids"] = confidence
        _validate_domain_stack_sidecars(
            tokens,
            row["token_domain_ids"],
            row["token_role_ids"],
            row["token_confidence_ids"],
            where=realized.task.value,
        )
        if require_production_sidecars and any(
            int(value) <= 0 for value in row["token_source_doc_ids"]
        ):
            raise ValueError(
                "pre-materialized production objective requires positive "
                "token_source_doc_ids"
            )
        if require_production_sidecars and any(
            int(value) <= 0 for value in row["token_source_identity_ids"]
        ):
            raise ValueError(
                "pre-materialized production objective requires positive "
                "token_source_identity_ids"
            )
        row[SOURCE_IDENTITY_REGISTRY_COLUMN] = _source_identity_registry_for_ids(
            packet,
            row["token_source_identity_ids"],
            required=require_production_sidecars,
        )
        row[SYMBOL_IDENTITIES_COLUMN] = _symbol_identity_records(packet)
        row["token_change_mask_pre"] = _optional_metadata_vector(
            packet, "token_change_mask_pre", length=len(tokens)
        )
        row["token_change_mask_post"] = _optional_metadata_vector(
            packet, "token_change_mask_post", length=len(tokens)
        )
        route_remap = remap_objective_routes(
            packet,
            source_token_indices=range(len(tokens)),
            where=realized.task.value,
            require_sidecars=require_production_sidecars,
            mode="identity",
        )
        row.update(route_remap.columns)
        route_receipt = route_remap.receipt
    elif realized.task in _TRANSFORMED_CODE_TASKS:
        if packet is None:
            raise ValueError(
                f"{realized.task.value}: transformed code objective requires "
                "ObjectiveSource.code_packet"
            )
        source_length = int(packet.token_ids.shape[0])
        assert packet_document_ids is not None
        source_indices = _validated_source_token_indices(
            metadata=example.metadata,
            packet=packet,
            transformed_tokens=tokens,
            objective_kind=realized.task.value,
        )
        row["doc_ids"] = _permute_provenance_vector(
            packet_document_ids,
            source_indices,
            where=f"{realized.task.value}.doc_ids",
        )
        row["source_platform_ids"] = _platform_bags_for_output_documents(
            row["doc_ids"],  # type: ignore[arg-type]
            source_platform_bags,
            required=require_production_sidecars,
            where=f"{realized.task.value}.doc_ids",
        )

        special_domain_columns = {
            "token_domain_ids",
            "token_role_ids",
            "token_confidence_ids",
            "token_source_doc_ids",
            "token_source_identity_ids",
        }
        for column, field in _TOKEN_SIDECARS.items():
            if column in special_domain_columns:
                continue
            source_values = _source_vector(
                packet,
                field,
                source_length=source_length,
                required=require_production_sidecars,
            )
            row[column] = _permute_source_vector(source_values, source_indices)

        source_domains = _source_vector(
            packet,
            "domain_ids",
            source_length=source_length,
            required=require_production_sidecars,
        )
        source_roles = _source_vector(
            packet,
            "role_ids",
            source_length=source_length,
            required=require_production_sidecars,
        )
        source_confidence = _source_vector(
            packet,
            "confidence_ids",
            source_length=source_length,
            required=require_production_sidecars,
        )
        domains, roles, confidence = _bind_domain_stack_sidecars(
            tokens,
            source_indices,
            source_domains,
            source_roles,
            source_confidence,
            where=realized.task.value,
        )
        _validate_domain_stack_sidecars(
            tokens,
            domains,
            roles,
            confidence,
            where=realized.task.value,
        )
        row["token_domain_ids"] = domains
        row["token_role_ids"] = roles
        row["token_confidence_ids"] = confidence

        source_doc_ids = _source_vector(
            packet,
            "source_doc_ids",
            source_length=source_length,
            required=require_production_sidecars,
        )
        if source_doc_ids is None:
            row["token_source_doc_ids"] = [0] * len(tokens)
        else:
            row["token_source_doc_ids"] = _permute_provenance_vector(
                source_doc_ids,
                source_indices,
                where=f"{realized.task.value}.token_source_doc_ids",
            )
            if any(
                not 0 < value <= MAX_ROW_LOCAL_DOC_ID
                for value in row["token_source_doc_ids"]
            ):
                raise ValueError(
                    f"{realized.task.value}: transformed token_source_doc_ids "
                    "must remain positive uint32 values"
                )

        source_identity_ids = _source_vector(
            packet,
            "source_identity_ids",
            source_length=source_length,
            required=require_production_sidecars,
        )
        if source_identity_ids is None:
            row["token_source_identity_ids"] = [0] * len(tokens)
        else:
            row["token_source_identity_ids"] = _permute_provenance_vector(
                source_identity_ids,
                source_indices,
                where=f"{realized.task.value}.token_source_identity_ids",
            )
            if any(
                not 0 < value <= MAX_SOURCE_ID
                for value in row["token_source_identity_ids"]
            ):
                raise ValueError(
                    f"{realized.task.value}: transformed "
                    "token_source_identity_ids must remain positive uint64 values"
                )
            if len(set(row["token_source_identity_ids"])) != 1:
                raise ValueError(
                    f"{realized.task.value}: transformed objective requires "
                    "exactly one positive physical source identity"
                )
        row[SOURCE_IDENTITY_REGISTRY_COLUMN] = _source_identity_registry_for_ids(
            packet,
            row["token_source_identity_ids"],
            required=require_production_sidecars,
        )
        row[SYMBOL_IDENTITIES_COLUMN] = _symbol_identity_records(packet)
        for column in ("token_change_mask_pre", "token_change_mask_post"):
            source_values = _optional_metadata_vector(
                packet,
                column,
                length=source_length,
            )
            row[column] = _permute_source_vector(source_values, source_indices)
        route_remap = remap_objective_routes(
            packet,
            source_token_indices=source_indices,
            where=realized.task.value,
            require_sidecars=require_production_sidecars,
        )
        row.update(route_remap.columns)
        route_receipt = route_remap.receipt
    elif realized.task in _COMMIT_TASKS:
        commit_source_indices = example.metadata.get(SOURCE_TOKEN_INDICES_METADATA_KEY)
        if (
            not isinstance(commit_source_indices, (tuple, list))
            or len(commit_source_indices) != len(tokens)
            or any(
                int(index) != INSERTED_TOKEN_SOURCE_INDEX
                for index in commit_source_indices
            )
        ):
            raise ValueError(
                f"{realized.task.value}: CommitPacket tokens must use only the "
                "inserted-token source sentinel"
            )
        zeros = [0] * len(tokens)
        for column in _TOKEN_SIDECARS:
            row[column] = list(zeros)
        domains, roles, confidence = _domain_stack_defaults(
            tokens,
            where=realized.task.value,
        )
        _validate_domain_stack_sidecars(
            tokens,
            domains,
            roles,
            confidence,
            where=realized.task.value,
        )
        row["token_domain_ids"] = domains
        row["token_role_ids"] = roles
        row["token_confidence_ids"] = confidence

        source_document_id = 0
        source_identity_id = 0
        if packet is not None and packet.source_doc_ids is not None:
            source_doc_ids = _ints(
                packet.source_doc_ids,
                where="CodePacket.source_doc_ids",
            )
            if any(not 0 < value <= MAX_ROW_LOCAL_DOC_ID for value in source_doc_ids):
                raise ValueError(
                    f"{realized.task.value}: commit source_doc_ids must be "
                    "positive uint32 values"
                )
            if len(set(source_doc_ids)) != 1:
                raise ValueError(
                    f"{realized.task.value}: commit objective requires exactly "
                    "one source document identity"
                )
            source_document_id = source_doc_ids[0]
        elif require_production_sidecars:
            raise ValueError(
                f"{realized.task.value}: production commit objective requires "
                "CodePacket.source_doc_ids"
            )
        if packet is not None and packet.source_identity_ids is not None:
            source_identity_ids = _ints(
                packet.source_identity_ids,
                where="CodePacket.source_identity_ids",
            )
            if any(not 0 < value <= MAX_SOURCE_ID for value in source_identity_ids):
                raise ValueError(
                    f"{realized.task.value}: commit physical source identities "
                    "must be positive uint64 values"
                )
            if len(set(source_identity_ids)) != 1:
                raise ValueError(
                    f"{realized.task.value}: commit objective requires exactly "
                    "one physical source identity"
                )
            source_identity_id = source_identity_ids[0]
        elif require_production_sidecars:
            raise ValueError(
                f"{realized.task.value}: production commit objective requires "
                "CodePacket.source_identity_ids"
            )
        row["token_source_doc_ids"] = [source_document_id] * len(tokens)
        row["token_source_identity_ids"] = [source_identity_id] * len(tokens)
        if (
            packet is not None
            and packet_document_ids is not None
            and packet.source_doc_ids is not None
        ):
            packet_source_doc_ids = _packet_vector(
                packet,
                "source_doc_ids",
                length=len(packet_document_ids),
            )
            attention_document_ids = {
                packet_document_ids[index]
                for index, value in enumerate(packet_source_doc_ids)
                if value == source_document_id
            }
            if not attention_document_ids:
                raise ValueError(
                    f"{realized.task.value}: commit source document has no "
                    "attention document mapping"
                )
            mapped_bags = [
                source_platform_bags[document_id]
                for document_id in sorted(attention_document_ids)
                if document_id in source_platform_bags
            ]
            if len(mapped_bags) != len(attention_document_ids):
                if require_production_sidecars:
                    raise ValueError(
                        f"{realized.task.value}: commit source platform mapping "
                        "is incomplete"
                    )
            elif mapped_bags:
                first_bag = mapped_bags[0]
                if any(bag != first_bag for bag in mapped_bags[1:]):
                    raise ValueError(
                        f"{realized.task.value}: commit source spans attention "
                        "documents with conflicting platform bags"
                    )
                row["source_platform_ids"] = [list(first_bag)]
        row[SOURCE_IDENTITY_REGISTRY_COLUMN] = (
            []
            if packet is None
            else _source_identity_registry_for_ids(
                packet,
                row["token_source_identity_ids"],
                required=require_production_sidecars,
            )
        )
        row[SYMBOL_IDENTITIES_COLUMN] = []
        row["token_change_mask_pre"] = list(zeros)
        row["token_change_mask_post"] = list(zeros)
        route_remap = exclude_objective_routes(
            packet,
            where=realized.task.value,
            reason=("independently_tokenized_commit_sections_have_no_exact_source_map"),
            require_sidecars=require_production_sidecars,
        )
        row.update(route_remap.columns)
        route_receipt = route_remap.receipt
    else:  # pragma: no cover - TaskKind exhaustiveness guard
        raise ValueError(f"unsupported materialized objective {realized.task.value}")

    row["doc_ids"] = _normalize_attention_document_ids(
        row["doc_ids"],  # type: ignore[arg-type]
        where=f"{realized.task.value}.doc_ids",
    )
    graph_edge_count = sum(
        len(row[column]) for column in _GRAPH_RELATION_COLUMNS.values()
    )
    return MaterializedMegatronDocument(
        objective_kind=realized.task.value,
        token_ids=tokens,
        loss_mask=materialized_mask,
        graph_edge_count=graph_edge_count,
        row=row,
        route_receipt=route_receipt,
    )


def build_pre_materialized_objective_contract(
    documents: Sequence[MaterializedMegatronDocument],
    *,
    rates: Mapping[TaskKind | str, float],
    seed: int,
    quota_window_samples: int,
    graph_config: GraphAuxLossConfig,
    graph_weight: float,
) -> dict[str, object]:
    """Build the exact receipt consumed by root Megatron conversion."""

    from cppmega_mlx.training.objective_contract_accumulator import (
        ObjectiveContractAccumulator,
    )

    accumulator = ObjectiveContractAccumulator(
        rates=rates,
        seed=seed,
        quota_window_samples=quota_window_samples,
        graph_config=graph_config,
        graph_weight=graph_weight,
    )
    accumulator.validate_sample_count(len(documents))
    for document in documents:
        accumulator.add(document)
    return accumulator.finalize()


__all__ = [
    "MaterializedMegatronDocument",
    "OBJECTIVE_CONTRACT_SCHEMA",
    "OBJECTIVE_GRAPH_SIDECARS",
    "OBJECTIVE_KIND_IDS",
    "OBJECTIVE_MATERIALIZATION_ARTIFACT_NAME",
    "OBJECTIVE_MATERIALIZATION_ARTIFACT_SCHEMA",
    "OBJECTIVE_TOKEN_SIDE_CHANNELS",
    "build_pre_materialized_objective_contract",
    "materialize_megatron_document",
    "objective_route_mapping_contract",
    "write_objective_materialization_artifact",
]

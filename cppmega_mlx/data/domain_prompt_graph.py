"""Frozen typed prompt graphs for local domain-routed generation evals.

The builder reuses the production domain parsers and token materializer. It
constructs one token-aligned prompt graph from explicit typed/plain parts and
never infers a missing cross-domain route.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Any

from cppmega_mlx.data.domain_ingestion import parse_domain_document
from cppmega_mlx.data.domain_schema import (
    DomainEdgeKind,
    DomainKind,
    DomainRoleKind,
    ParseConfidence,
    domain_edge_family,
    validate_domain_edge_kind,
)
from cppmega_mlx.data.nanochat_pipeline import tokenized_enriched_schema as schema
from cppmega_mlx.data.nanochat_pipeline.tokenized_enriched import (
    materialize_tokenized_enriched_batch,
)
from cppmega_mlx.data.prompt_graph import (
    GENERATED_TOKEN_SIDECAR_DEFAULTS,
    TOKEN_SIDECAR_NAMES,
)
from cppmega_mlx.data.source_identity import source_identity


DOMAIN_PROMPT_GRAPH_SCHEMA = "cppmega_domain_prompt_graph_v1"
DOMAIN_PROMPT_GRAPH_WINDOW_SCHEMA = "cppmega_domain_prompt_graph_window_v1"

EVAL_TOKEN_SIDECARS = (
    schema.TOKEN_DOMAIN_IDS_COLUMN,
    schema.TOKEN_ROLE_IDS_COLUMN,
    schema.TOKEN_ENTITY_IDS_COLUMN,
    schema.TOKEN_SCOPE_IDS_COLUMN,
    schema.TOKEN_SOURCE_DOC_IDS_COLUMN,
    schema.TOKEN_SOURCE_IDENTITY_IDS_COLUMN,
    schema.TOKEN_CONFIDENCE_IDS_COLUMN,
)
EVAL_EDGE_SIDECARS = (
    schema.TOKEN_DOMAIN_EDGES_COLUMN,
    schema.TOKEN_BUILD_EDGES_COLUMN,
    schema.TOKEN_SHELL_EDGES_COLUMN,
    schema.TOKEN_DIAGNOSTIC_EDGES_COLUMN,
    schema.TOKEN_CROSS_DOMAIN_EDGES_COLUMN,
)
DOMAIN_PROMPT_GRAPH_EDGE_FAMILIES = (
    "domain",
    "build",
    "shell",
    "diagnostic",
    "cross_domain",
)
_EDGE_COLUMN_FAMILIES = {
    schema.TOKEN_DOMAIN_EDGES_COLUMN: "domain",
    schema.TOKEN_BUILD_EDGES_COLUMN: "build",
    schema.TOKEN_SHELL_EDGES_COLUMN: "shell",
    schema.TOKEN_DIAGNOSTIC_EDGES_COLUMN: "diagnostic",
    schema.TOKEN_CROSS_DOMAIN_EDGES_COLUMN: "cross_domain",
}
_MODEL_SIDECAR_COLUMNS = {
    "domain_ids": schema.TOKEN_DOMAIN_IDS_COLUMN,
    "role_ids": schema.TOKEN_ROLE_IDS_COLUMN,
    "entity_ids": schema.TOKEN_ENTITY_IDS_COLUMN,
    "scope_ids": schema.TOKEN_SCOPE_IDS_COLUMN,
    "source_doc_ids": schema.TOKEN_SOURCE_DOC_IDS_COLUMN,
    "confidence_ids": schema.TOKEN_CONFIDENCE_IDS_COLUMN,
}


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _sha_json(value: Any) -> str:
    return sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _normalize_required_edge_families(
    values: Sequence[Any] | None,
    *,
    context: str,
) -> tuple[str, ...]:
    """Normalize the edge families a frozen publisher promises to retain."""

    raw = () if values is None else values
    if isinstance(raw, (str, bytes)):
        raise ValueError(f"{context}: required_edge_families must be a sequence")
    try:
        names = tuple(str(value) for value in raw)
    except TypeError as exc:
        raise ValueError(
            f"{context}: required_edge_families must be a sequence"
        ) from exc
    if len(set(names)) != len(names):
        raise ValueError(f"{context}: required_edge_families must not contain duplicates")
    unknown = set(names) - set(DOMAIN_PROMPT_GRAPH_EDGE_FAMILIES)
    if unknown:
        raise ValueError(
            f"{context}: unknown required edge families {sorted(unknown)}"
        )
    return tuple(
        family
        for family in DOMAIN_PROMPT_GRAPH_EDGE_FAMILIES
        if family in names
    )


def _domain_marker_name(domain: DomainKind) -> str:
    return "CPP_CODE" if domain == DomainKind.CPP else domain.name


def _domain_from_value(value: Any, *, where: str) -> DomainKind:
    try:
        if isinstance(value, str) and not value.isdigit():
            name = "CPP" if value.upper() == "CPP_CODE" else value.upper()
            domain = DomainKind[name]
        else:
            domain = DomainKind(int(value))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{where}: unknown domain {value!r}") from exc
    if domain == DomainKind.UNKNOWN:
        raise ValueError(f"{where}: UNKNOWN must be represented by an untyped part")
    return domain


def _role_from_value(value: Any, *, where: str) -> DomainRoleKind:
    try:
        if isinstance(value, str) and not value.isdigit():
            return DomainRoleKind[value.upper()]
        return DomainRoleKind(int(value))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{where}: unknown domain role {value!r}") from exc


def _edge_kind_from_value(value: Any, *, where: str) -> DomainEdgeKind:
    try:
        if isinstance(value, str) and not value.isdigit():
            return DomainEdgeKind[value.upper()]
        return DomainEdgeKind(int(value))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{where}: unknown domain edge kind {value!r}") from exc


@dataclass(frozen=True)
class DomainPromptPart:
    text: str
    domain: DomainKind | None
    path: str
    source_doc_id: int

    @classmethod
    def from_mapping(
        cls,
        row: Mapping[str, Any],
        *,
        index: int,
        context: str,
    ) -> "DomainPromptPart":
        text = row.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError(f"{context}: prompt part {index} needs non-empty text")
        raw_domain = row.get("domain")
        domain = (
            None
            if raw_domain in (None, "", "UNKNOWN", 0)
            else _domain_from_value(
                raw_domain,
                where=f"{context}: prompt part {index}",
            )
        )
        path = row.get("path", f"prompt-part-{index}.txt")
        if not isinstance(path, str) or not path:
            raise ValueError(f"{context}: prompt part {index} path must be non-empty")
        source_doc_id = row.get("source_doc_id", index + 1)
        if (
            isinstance(source_doc_id, bool)
            or not isinstance(source_doc_id, int)
            or source_doc_id <= 0
        ):
            raise ValueError(
                f"{context}: prompt part {index} source_doc_id must be positive"
            )
        return cls(
            text=text,
            domain=domain,
            path=path,
            source_doc_id=source_doc_id,
        )

    def rendered_text(self) -> str:
        if self.domain is None:
            return self.text
        marker = _domain_marker_name(self.domain)
        return f"<{marker}_START>{self.text}<{marker}_END>"


@dataclass(frozen=True)
class DomainPromptGraphWindow:
    side_channels: Mapping[str, list[int]]
    edge_sidecars: Mapping[str, list[dict[str, int]]]
    receipt: Mapping[str, Any]
    token_count: int
    required_edge_families: tuple[str, ...] = ()

    def dense_relation_attention_bias(
        self,
        *,
        relation_weights: Mapping[str, float] | None = None,
    ) -> list[list[float]]:
        weights = {family: 1.0 for family in _EDGE_COLUMN_FAMILIES.values()}
        if relation_weights is not None:
            unknown = set(relation_weights) - set(weights)
            if unknown:
                raise ValueError(
                    f"unknown domain prompt relation weights: {sorted(unknown)}"
                )
            weights.update(
                {str(key): float(value) for key, value in relation_weights.items()}
            )
        for family, weight in weights.items():
            if not math.isfinite(weight) or weight == 0.0:
                raise ValueError(
                    f"domain prompt relation {family} weight must be finite/nonzero"
                )
        bias = [
            [0.0 for _ in range(self.token_count)]
            for _ in range(self.token_count)
        ]
        for column, family in _EDGE_COLUMN_FAMILIES.items():
            for edge in self.edge_sidecars[column]:
                bias[int(edge["from"])][int(edge["to"])] += weights[family]
        return bias

    def dense_edge_kind_attention_bias(
        self,
        *,
        edge_kind_weights: Mapping[int, float] | None = None,
        default_weight: float = 1.0,
    ) -> list[list[float]]:
        default_weight = float(default_weight)
        if not math.isfinite(default_weight) or default_weight == 0.0:
            raise ValueError(
                "domain prompt edge-kind default weight must be finite/nonzero"
            )
        weights = {
            int(kind): float(weight)
            for kind, weight in (edge_kind_weights or {}).items()
        }
        for kind, weight in weights.items():
            validate_domain_edge_kind(kind)
            if not math.isfinite(weight) or weight == 0.0:
                raise ValueError(
                    f"domain prompt edge kind {kind} weight must be finite/nonzero"
                )
        bias = [
            [0.0 for _ in range(self.token_count)]
            for _ in range(self.token_count)
        ]
        for column in EVAL_EDGE_SIDECARS:
            for edge in self.edge_sidecars[column]:
                kind = int(edge["kind"])
                bias[int(edge["from"])][int(edge["to"])] += weights.get(
                    kind,
                    default_weight,
                )
        return bias

    def edge_kind_route_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for column in EVAL_EDGE_SIDECARS:
            for edge in self.edge_sidecars[column]:
                kind = int(edge["kind"])
                counts[kind] = counts.get(kind, 0) + 1
        return dict(sorted(counts.items()))


@dataclass(frozen=True)
class DomainPromptGraph:
    token_ids: tuple[int, ...]
    eval_sidecars: Mapping[str, tuple[Any, ...]]
    side_channels: Mapping[str, list[int]]
    part_ranges: tuple[tuple[int, int], ...]
    receipt: Mapping[str, Any]
    required_edge_families: tuple[str, ...] = ()
    schema: str = DOMAIN_PROMPT_GRAPH_SCHEMA

    def validate(self) -> None:
        if self.schema != DOMAIN_PROMPT_GRAPH_SCHEMA:
            raise ValueError(f"unsupported domain prompt graph schema {self.schema!r}")
        if not self.token_ids:
            raise ValueError("domain prompt graph has no token ids")
        token_count = len(self.token_ids)
        if set(self.eval_sidecars) != set((*EVAL_TOKEN_SIDECARS, *EVAL_EDGE_SIDECARS)):
            raise ValueError(
                "domain prompt graph eval sidecars mismatch: "
                f"actual={sorted(self.eval_sidecars)}"
            )
        for column in EVAL_TOKEN_SIDECARS:
            values = self.eval_sidecars[column]
            if len(values) != token_count:
                raise ValueError(
                    f"domain prompt graph {column} length {len(values)} != {token_count}"
                )
        for value in self.eval_sidecars[schema.TOKEN_DOMAIN_IDS_COLUMN]:
            DomainKind(int(value))
        for value in self.eval_sidecars[schema.TOKEN_ROLE_IDS_COLUMN]:
            DomainRoleKind(int(value))
        for value in self.eval_sidecars[schema.TOKEN_CONFIDENCE_IDS_COLUMN]:
            ParseConfidence(int(value))
        if any(
            int(value) <= 0
            for value in self.eval_sidecars[schema.TOKEN_SOURCE_DOC_IDS_COLUMN]
        ):
            raise ValueError("domain prompt graph source_doc_ids must be positive")
        if any(
            int(value) <= 0
            for value in self.eval_sidecars[
                schema.TOKEN_SOURCE_IDENTITY_IDS_COLUMN
            ]
        ):
            raise ValueError(
                "domain prompt graph source_identity_ids must be positive"
            )

        for column, family in _EDGE_COLUMN_FAMILIES.items():
            for edge in self.eval_sidecars[column]:
                if not isinstance(edge, Mapping):
                    raise ValueError(f"domain prompt graph {column} edges must be objects")
                src = int(edge["from"])
                dst = int(edge["to"])
                kind = int(edge["kind"])
                if not (0 <= src < token_count and 0 <= dst < token_count):
                    raise ValueError(
                        f"domain prompt graph {column} edge {src}->{dst} is out of bounds"
                    )
                if domain_edge_family(kind) != family:
                    raise ValueError(
                        f"domain prompt graph {column} edge kind {kind} is not {family}"
                    )

        if set(self.side_channels) != set(TOKEN_SIDECAR_NAMES):
            raise ValueError("domain prompt graph model side-channel names mismatch")
        for name, values in self.side_channels.items():
            if len(values) != token_count:
                raise ValueError(
                    f"domain prompt graph model side channel {name} length mismatch"
                )
        for start, end in self.part_ranges:
            if not 0 <= int(start) < int(end) <= token_count:
                raise ValueError(
                    f"domain prompt graph part range [{start},{end}) is invalid"
                )
        normalized_required = _normalize_required_edge_families(
            self.required_edge_families,
            context="domain prompt graph",
        )
        receipt_required = _normalize_required_edge_families(
            self.receipt.get("required_edge_families", ()),
            context="domain prompt graph receipt",
        )
        if receipt_required != normalized_required:
            raise ValueError(
                "domain prompt graph receipt required_edge_families mismatch: "
                f"actual={normalized_required} receipt={receipt_required}"
            )
        actual_edge_counts = self.edge_counts
        receipt_edge_counts = self.receipt.get("edge_counts")
        if not isinstance(receipt_edge_counts, Mapping):
            raise ValueError("domain prompt graph receipt edge_counts must be an object")
        normalized_receipt_counts = {
            family: int(receipt_edge_counts.get(family, -1))
            for family in DOMAIN_PROMPT_GRAPH_EDGE_FAMILIES
        }
        if normalized_receipt_counts != actual_edge_counts:
            raise ValueError(
                "domain prompt graph receipt edge_counts mismatch: "
                f"actual={actual_edge_counts} receipt={normalized_receipt_counts}"
            )
        missing_required = {
            family: actual_edge_counts[family]
            for family in normalized_required
            if actual_edge_counts[family] <= 0
        }
        if missing_required:
            raise ValueError(
                "domain prompt graph required edge families are empty: "
                f"{missing_required}"
            )
        if self.receipt.get("schema") != DOMAIN_PROMPT_GRAPH_SCHEMA:
            raise ValueError("domain prompt graph receipt schema mismatch")
        if int(self.receipt.get("token_count", -1)) != token_count:
            raise ValueError("domain prompt graph receipt token count mismatch")

    @property
    def edge_counts(self) -> dict[str, int]:
        return {
            _EDGE_COLUMN_FAMILIES[column]: len(self.eval_sidecars[column])
            for column in EVAL_EDGE_SIDECARS
        }

    def model_inputs(
        self,
        *,
        total_token_count: int,
        window_start: int,
        window_end: int,
    ) -> DomainPromptGraphWindow:
        self.validate()
        prompt_count = len(self.token_ids)
        if total_token_count < prompt_count:
            raise ValueError(
                "total_token_count cannot be shorter than the domain prompt graph"
            )
        if not 0 <= window_start < window_end <= total_token_count:
            raise ValueError(
                f"invalid domain prompt window [{window_start},{window_end})"
            )
        generated_count = total_token_count - prompt_count
        domains = self.side_channels["domain_ids"]
        roles = self.side_channels["role_ids"]
        anchors = [
            index
            for index, (domain, role) in enumerate(zip(domains, roles, strict=True))
            if int(domain) != int(DomainKind.UNKNOWN)
            and int(role) != int(DomainRoleKind.DELIMITER)
        ]
        if not anchors:
            raise ValueError("domain prompt graph has no typed generation anchor")
        anchor = anchors[-1]
        generated_values = dict(GENERATED_TOKEN_SIDECAR_DEFAULTS)
        for name in (
            "structure_ids",
            "dep_levels",
            "ast_depth_ids",
            "sibling_index_ids",
            "node_type_ids",
            "domain_ids",
        ):
            generated_values[name] = int(self.side_channels[name][anchor])
        generated_values["role_ids"] = int(DomainRoleKind.NONE)
        generated_values["confidence_ids"] = int(ParseConfidence.HEURISTIC)
        generated_values["source_doc_ids"] = 0
        side_channels = {
            name: (
                list(values) + [int(generated_values[name])] * generated_count
            )[window_start:window_end]
            for name, values in self.side_channels.items()
        }

        visible_edges: dict[str, list[dict[str, int]]] = {
            column: [] for column in EVAL_EDGE_SIDECARS
        }
        for column in EVAL_EDGE_SIDECARS:
            for edge in self.eval_sidecars[column]:
                src = int(edge["from"])
                dst = int(edge["to"])
                if (
                    window_start <= src < window_end
                    and window_start <= dst < window_end
                ):
                    visible_edges[column].append(
                        {
                            "from": src - window_start,
                            "to": dst - window_start,
                            "kind": int(edge["kind"]),
                        }
                    )
        receipt = {
            "schema": DOMAIN_PROMPT_GRAPH_WINDOW_SCHEMA,
            "artifact_sha256": self.receipt["artifact_sha256"],
            "window_start": window_start,
            "window_end": window_end,
            "total_token_count": total_token_count,
            "token_count": window_end - window_start,
            "generated_token_count": generated_count,
            "required_edge_families": list(self.required_edge_families),
            "edge_counts": {
                _EDGE_COLUMN_FAMILIES[column]: len(visible_edges[column])
                for column in EVAL_EDGE_SIDECARS
            },
        }
        missing_required = {
            family: receipt["edge_counts"][family]
            for family in self.required_edge_families
            if receipt["edge_counts"][family] <= 0
        }
        if missing_required:
            raise ValueError(
                "domain prompt graph generation window dropped required edge "
                f"families: {missing_required}"
            )
        return DomainPromptGraphWindow(
            side_channels=side_channels,
            edge_sidecars=visible_edges,
            receipt=receipt,
            token_count=window_end - window_start,
            required_edge_families=self.required_edge_families,
        )


def render_domain_prompt_graph_spec(
    spec: Mapping[str, Any],
    *,
    context: str = "domain prompt graph",
) -> str:
    raw_parts = spec.get("parts")
    if not isinstance(raw_parts, Sequence) or isinstance(raw_parts, (str, bytes)):
        raise ValueError(f"{context}: parts must be a sequence")
    parts = [
        DomainPromptPart.from_mapping(row, index=index, context=context)
        for index, row in enumerate(raw_parts)
        if isinstance(row, Mapping)
    ]
    if len(parts) != len(raw_parts) or not parts:
        raise ValueError(f"{context}: every prompt part must be an object")
    return "".join(part.rendered_text() for part in parts)


def _plain_part_row(
    tokenizer: Any,
    part: DomainPromptPart,
) -> dict[str, Any]:
    ids = list(tokenizer.encode(part.text))
    bos_id = int(tokenizer.bos_token_id)
    token_ids = [bos_id, *ids]
    count = len(token_ids)
    identity_id = int(
        source_identity(
            {
                "repo": "evals/domain-prompt",
                "source_path": part.path,
                "text": part.text,
            }
        ).source_identity_id
    )
    row: dict[str, Any] = {
        schema.TOKEN_IDS_COLUMN: token_ids,
        schema.TOKEN_DOMAIN_IDS_COLUMN: [int(DomainKind.UNKNOWN)] * count,
        schema.TOKEN_ROLE_IDS_COLUMN: [int(DomainRoleKind.NONE)] * count,
        schema.TOKEN_ENTITY_IDS_COLUMN: [0] * count,
        schema.TOKEN_SCOPE_IDS_COLUMN: [0] * count,
        schema.TOKEN_SOURCE_DOC_IDS_COLUMN: [part.source_doc_id] * count,
        schema.TOKEN_SOURCE_IDENTITY_IDS_COLUMN: [identity_id] * count,
        schema.TOKEN_CONFIDENCE_IDS_COLUMN: [int(ParseConfidence.HEURISTIC)] * count,
    }
    row.update({column: [] for column in EVAL_EDGE_SIDECARS})
    return row


def _typed_part_row(
    tokenizer: Any,
    part: DomainPromptPart,
    *,
    context: str,
) -> dict[str, Any]:
    assert part.domain is not None
    parsed = parse_domain_document(
        part.path,
        part.text,
        source_doc_id=part.source_doc_id,
        provenance={
            "repo": "evals/domain-prompt",
            "filepath": part.path,
            "context": context,
        },
    )
    if parsed.domain != part.domain:
        raise ValueError(
            f"{context}: part {part.path!r} parsed as {parsed.domain.name}, "
            f"expected {part.domain.name}"
        )
    return materialize_tokenized_enriched_batch(
        [parsed.to_enriched_document()],
        tokenizer,
        num_threads=1,
    )[0]


def _anchor_for_cross_edge(
    *,
    sidecars: Mapping[str, list[Any]],
    part_ranges: Sequence[tuple[int, int]],
    part_index: int,
    role: DomainRoleKind,
    occurrence: int,
    context: str,
) -> int:
    if not 0 <= part_index < len(part_ranges):
        raise ValueError(f"{context}: cross-domain part index {part_index} is invalid")
    start, end = part_ranges[part_index]
    candidates = [
        index
        for index in range(start, end)
        if int(sidecars[schema.TOKEN_ROLE_IDS_COLUMN][index]) == int(role)
    ]
    if not candidates:
        raise ValueError(
            f"{context}: part {part_index} has no token with role {role.name}"
        )
    resolved = occurrence if occurrence >= 0 else len(candidates) + occurrence
    if not 0 <= resolved < len(candidates):
        raise ValueError(
            f"{context}: role occurrence {occurrence} is outside {len(candidates)} matches"
        )
    return candidates[resolved]


def build_domain_prompt_graph(
    tokenizer: Any,
    spec: Mapping[str, Any],
    *,
    context: str = "domain prompt graph",
) -> DomainPromptGraph:
    raw_parts = spec.get("parts")
    if not isinstance(raw_parts, Sequence) or isinstance(raw_parts, (str, bytes)):
        raise ValueError(f"{context}: parts must be a sequence")
    parts: list[DomainPromptPart] = []
    for index, raw_part in enumerate(raw_parts):
        if not isinstance(raw_part, Mapping):
            raise ValueError(f"{context}: prompt part {index} must be an object")
        parts.append(
            DomainPromptPart.from_mapping(raw_part, index=index, context=context)
        )
    if not parts:
        raise ValueError(f"{context}: prompt graph needs at least one part")
    required_edge_families = _normalize_required_edge_families(
        spec.get("required_edge_families", ()),
        context=context,
    )

    merged_token_ids: list[int] = []
    merged_sidecars: dict[str, list[Any]] = {
        column: [] for column in (*EVAL_TOKEN_SIDECARS, *EVAL_EDGE_SIDECARS)
    }
    part_ranges: list[tuple[int, int]] = []
    for part_index, part in enumerate(parts):
        row = (
            _plain_part_row(tokenizer, part)
            if part.domain is None
            else _typed_part_row(tokenizer, part, context=context)
        )
        local_ids = [int(value) for value in row[schema.TOKEN_IDS_COLUMN]]
        if not local_ids or local_ids[0] != int(tokenizer.bos_token_id):
            raise ValueError(f"{context}: part {part_index} is missing BOS")
        local_start = 0 if part_index == 0 else 1
        global_start = len(merged_token_ids)
        merged_token_ids.extend(local_ids[local_start:])
        for column in EVAL_TOKEN_SIDECARS:
            values = list(row[column])
            if len(values) != len(local_ids):
                raise ValueError(
                    f"{context}: part {part_index} {column} is not token-aligned"
                )
            merged_sidecars[column].extend(values[local_start:])
        for column in EVAL_EDGE_SIDECARS:
            for raw_edge in row.get(column, []):
                src = int(raw_edge["from"])
                dst = int(raw_edge["to"])
                if src < local_start or dst < local_start:
                    raise ValueError(
                        f"{context}: part {part_index} edge references its skipped BOS"
                    )
                merged_sidecars[column].append(
                    {
                        "from": global_start + src - local_start,
                        "to": global_start + dst - local_start,
                        "kind": int(raw_edge["kind"]),
                    }
                )
        part_ranges.append((global_start, len(merged_token_ids)))

    raw_cross_edges = spec.get("cross_domain_edges", ())
    if not isinstance(raw_cross_edges, Sequence) or isinstance(
        raw_cross_edges,
        (str, bytes),
    ):
        raise ValueError(f"{context}: cross_domain_edges must be a sequence")
    for edge_index, raw_edge in enumerate(raw_cross_edges):
        if not isinstance(raw_edge, Mapping):
            raise ValueError(f"{context}: cross-domain edge {edge_index} must be an object")
        where = f"{context}: cross-domain edge {edge_index}"
        try:
            from_part = int(raw_edge["from_part"])
            to_part = int(raw_edge["to_part"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{where}: from_part/to_part must be integers") from exc
        from_role = _role_from_value(raw_edge.get("from_role"), where=where)
        to_role = _role_from_value(raw_edge.get("to_role"), where=where)
        kind = validate_domain_edge_kind(
            _edge_kind_from_value(
                raw_edge.get("kind", int(DomainEdgeKind.EMBEDDED_DOMAIN)),
                where=where,
            ),
            family="cross_domain",
        )
        src = _anchor_for_cross_edge(
            sidecars=merged_sidecars,
            part_ranges=part_ranges,
            part_index=from_part,
            role=from_role,
            occurrence=int(raw_edge.get("from_occurrence", 0)),
            context=where,
        )
        dst = _anchor_for_cross_edge(
            sidecars=merged_sidecars,
            part_ranges=part_ranges,
            part_index=to_part,
            role=to_role,
            occurrence=int(raw_edge.get("to_occurrence", 0)),
            context=where,
        )
        domains = merged_sidecars[schema.TOKEN_DOMAIN_IDS_COLUMN]
        if int(domains[src]) == int(domains[dst]):
            raise ValueError(f"{where}: endpoints must belong to distinct domains")
        merged_sidecars[schema.TOKEN_CROSS_DOMAIN_EDGES_COLUMN].append(
            {"from": src, "to": dst, "kind": int(kind)}
        )

    eval_sidecars = {
        column: tuple(values)
        for column, values in merged_sidecars.items()
    }
    token_count = len(merged_token_ids)
    side_channels: dict[str, list[int]] = {
        name: [0] * token_count for name in TOKEN_SIDECAR_NAMES
    }
    for name, column in _MODEL_SIDECAR_COLUMNS.items():
        side_channels[name] = [int(value) for value in eval_sidecars[column]]

    core = {
        "token_ids": merged_token_ids,
        "eval_sidecars": {
            column: list(values) for column, values in sorted(eval_sidecars.items())
        },
        "part_ranges": [list(item) for item in part_ranges],
        "rendered_prompt": "".join(part.rendered_text() for part in parts),
        "required_edge_families": list(required_edge_families),
    }
    receipt = {
        "schema": DOMAIN_PROMPT_GRAPH_SCHEMA,
        "token_count": token_count,
        "part_count": len(parts),
        "part_domains": [
            None if part.domain is None else part.domain.name for part in parts
        ],
        "required_edge_families": list(required_edge_families),
        "edge_counts": {
            _EDGE_COLUMN_FAMILIES[column]: len(eval_sidecars[column])
            for column in EVAL_EDGE_SIDECARS
        },
        "rendered_prompt_sha256": sha256(
            core["rendered_prompt"].encode("utf-8")
        ).hexdigest(),
        "artifact_sha256": _sha_json(core),
    }
    graph = DomainPromptGraph(
        token_ids=tuple(merged_token_ids),
        eval_sidecars=eval_sidecars,
        side_channels=side_channels,
        part_ranges=tuple(part_ranges),
        receipt=receipt,
        required_edge_families=required_edge_families,
    )
    graph.validate()
    return graph


def domain_prompt_graph_from_frozen(
    token_ids: Sequence[Any],
    sidecars: Mapping[str, Sequence[Any]],
    *,
    part_ranges: Sequence[Sequence[int]] | None = None,
    required_edge_families: Sequence[Any] | None = None,
    context: str = "frozen domain prompt graph",
) -> DomainPromptGraph:
    normalized_sidecars: dict[str, tuple[Any, ...]] = {}
    for column in (*EVAL_TOKEN_SIDECARS, *EVAL_EDGE_SIDECARS):
        raw = sidecars.get(column)
        if raw is None:
            raise ValueError(f"{context}: missing sidecar {column}")
        if column in EVAL_EDGE_SIDECARS:
            normalized_sidecars[column] = tuple(
                {
                    "from": int(edge["from"]),
                    "to": int(edge["to"]),
                    "kind": int(edge["kind"]),
                }
                for edge in raw
            )
        else:
            normalized_sidecars[column] = tuple(int(value) for value in raw)
    normalized_ids = tuple(int(value) for value in token_ids)
    token_count = len(normalized_ids)
    ranges = (
        ((0, token_count),)
        if part_ranges is None
        else tuple((int(row[0]), int(row[1])) for row in part_ranges)
    )
    normalized_required = _normalize_required_edge_families(
        required_edge_families,
        context=context,
    )
    side_channels = {name: [0] * token_count for name in TOKEN_SIDECAR_NAMES}
    for name, column in _MODEL_SIDECAR_COLUMNS.items():
        side_channels[name] = [
            int(value) for value in normalized_sidecars[column]
        ]
    core = {
        "token_ids": list(normalized_ids),
        "eval_sidecars": {
            column: list(values)
            for column, values in sorted(normalized_sidecars.items())
        },
        "part_ranges": [list(item) for item in ranges],
        "required_edge_families": list(normalized_required),
    }
    receipt = {
        "schema": DOMAIN_PROMPT_GRAPH_SCHEMA,
        "token_count": token_count,
        "part_count": len(ranges),
        "part_domains": [],
        "required_edge_families": list(normalized_required),
        "edge_counts": {
            _EDGE_COLUMN_FAMILIES[column]: len(normalized_sidecars[column])
            for column in EVAL_EDGE_SIDECARS
        },
        "rendered_prompt_sha256": None,
        "artifact_sha256": _sha_json(core),
    }
    graph = DomainPromptGraph(
        token_ids=normalized_ids,
        eval_sidecars=normalized_sidecars,
        side_channels=side_channels,
        part_ranges=ranges,
        receipt=receipt,
        required_edge_families=normalized_required,
    )
    graph.validate()
    return graph


__all__ = [
    "DOMAIN_PROMPT_GRAPH_SCHEMA",
    "DOMAIN_PROMPT_GRAPH_WINDOW_SCHEMA",
    "DomainPromptGraph",
    "DomainPromptGraphWindow",
    "DomainPromptPart",
    "DOMAIN_PROMPT_GRAPH_EDGE_FAMILIES",
    "EVAL_EDGE_SIDECARS",
    "EVAL_TOKEN_SIDECARS",
    "build_domain_prompt_graph",
    "domain_prompt_graph_from_frozen",
    "render_domain_prompt_graph_spec",
]

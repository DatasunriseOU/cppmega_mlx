"""Canonical bounded objective scheduling and graph-capability receipts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

from cppmega_mlx.data.code_packet import CodePacket
from cppmega_mlx.data.commit_packet import CommitPacket
from cppmega_mlx.training.objective_contract_accumulator import (
    count_configured_graph_positive_edges,
)
from cppmega_mlx.training.objective_mixer import (
    EligibilityAwareTaskMixer,
    ObjectiveQuotaUnsatisfiedError,
    ObjectiveSource,
    RealizedObjective,
    SourceInput,
)
from cppmega_mlx.training.objective_data import OBJECTIVE_GRAPH_RELATION_COLUMNS
from cppmega_mlx.training.objective_data import (
    exclude_objective_routes,
    remap_objective_routes,
)
from cppmega_mlx.training.objectives import SOURCE_TOKEN_INDICES_METADATA_KEY
from cppmega_mlx.training.task_mixer import TaskKind

OBJECTIVE_SOURCE_SELECTION_SCHEMA = "cppmega_objective_source_selection_v3"
OBJECTIVE_SOURCE_RESUME_SCHEMA = "cppmega_objective_source_resume_v1"
OBJECTIVE_SCHEDULE_WINDOW_SCHEMA = "cppmega_objective_schedule_window_v1"
OBJECTIVE_SCHEDULE_RECEIPT_SCHEMA = "cppmega_objective_schedule_v1"
OBJECTIVE_SCHEDULE_ALGORITHM = (
    "bounded_eligibility_bipartite_graph_capability_v1"
)
GRAPH_ELIGIBILITY_RECEIPT_SCHEMA = "cppmega_objective_graph_eligibility_v1"

_TRANSFORMED_TASKS = frozenset(
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
_GRAPH_PACKET_FIELDS = {
    "call": "call_edges",
    "type": "type_edges",
    "domain": "domain_edges",
    "build": "build_edges",
    "shell": "shell_edges",
    "diagnostic": "diagnostic_edges",
    "cross_domain": "cross_domain_edges",
}


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def canonical_schedule_receipt_sha256(windows: Sequence[Mapping[str, object]]) -> str:
    """Hash the ordered canonical window receipts."""

    return _canonical_sha256(list(windows))


def canonical_window_quotas(
    rates: Mapping[TaskKind | str, float | str],
    window_size: int,
) -> dict[str, int]:
    """Compute the exact Hamilton quota map used by the typed mixer."""

    if window_size < 1:
        raise ValueError("window_size must be positive")
    parsed: dict[TaskKind, Fraction] = {}
    for raw_task, raw_rate in rates.items():
        task = raw_task if isinstance(raw_task, TaskKind) else TaskKind(raw_task)
        rate = (
            raw_rate
            if isinstance(raw_rate, Fraction)
            else Fraction(str(raw_rate))
        )
        if rate < 0:
            raise ValueError(f"objective rate for {task.value} is negative")
        parsed[task] = parsed.get(task, Fraction()) + rate
    total = sum(parsed.values(), Fraction())
    if not parsed or total <= 0:
        raise ValueError("objective rates must contain a positive total")
    normalized = {task: rate / total for task, rate in parsed.items()}
    raw = {task: rate * window_size for task, rate in normalized.items()}
    quotas = {task: value.numerator // value.denominator for task, value in raw.items()}
    remaining = window_size - sum(quotas.values())
    task_order = {task: index for index, task in enumerate(TaskKind)}
    ranked = sorted(
        normalized,
        key=lambda task: (-(raw[task] - quotas[task]), task_order[task]),
    )
    for task in ranked[:remaining]:
        quotas[task] += 1
    return {task.value: quotas[task] for task in normalized}


def _receipt_mapping(value: object, *, where: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be an object")
    return value


def _receipt_int(
    value: object,
    *,
    where: str,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{where} must be an integer >= {minimum}")
    return value


def validate_objective_source_selection_receipt(
    receipt: Mapping[str, object],
    *,
    output_samples: int,
    quota_window_samples: int,
    graph_relations: Sequence[str],
    expected_window_quotas: Mapping[str, int] | None = None,
) -> None:
    """Validate the canonical bounded schedule receipt without filling defaults."""

    expected_keys = {
        "schema",
        "algorithm",
        "output_samples",
        "source_rows_consumed",
        "unused_buffered_sources",
        "quota_window_samples",
        "quota_lookahead_samples",
        "max_source_pool_samples",
        "max_source_pool_observed",
        "required_graph_relations",
        "windows",
        "windows_sha256",
        "resume",
        "schedule",
    }
    if set(receipt) != expected_keys:
        raise ValueError(
            "objective source selection keys must be exactly "
            f"{sorted(expected_keys)}"
        )
    if receipt.get("schema") != OBJECTIVE_SOURCE_SELECTION_SCHEMA:
        raise ValueError("objective source selection schema is invalid")
    if receipt.get("algorithm") != OBJECTIVE_SCHEDULE_ALGORITHM:
        raise ValueError("objective source selection algorithm is invalid")
    if receipt.get("output_samples") != output_samples:
        raise ValueError("objective source selection output_samples drifted")
    if receipt.get("quota_window_samples") != quota_window_samples:
        raise ValueError("objective source selection quota window drifted")
    if output_samples % quota_window_samples:
        raise ValueError("objective source selection requires complete quota windows")
    lookahead = _receipt_int(
        receipt.get("quota_lookahead_samples"),
        where="source_selection.quota_lookahead_samples",
    )
    max_pool = _receipt_int(
        receipt.get("max_source_pool_samples"),
        where="source_selection.max_source_pool_samples",
        minimum=quota_window_samples,
    )
    if max_pool != quota_window_samples + lookahead:
        raise ValueError("objective source selection max pool does not bind lookahead")
    max_observed = _receipt_int(
        receipt.get("max_source_pool_observed"),
        where="source_selection.max_source_pool_observed",
        minimum=quota_window_samples,
    )
    if max_observed > max_pool:
        raise ValueError("objective source selection exceeded its bounded pool")
    consumed = _receipt_int(
        receipt.get("source_rows_consumed"),
        where="source_selection.source_rows_consumed",
        minimum=output_samples,
    )
    unused = _receipt_int(
        receipt.get("unused_buffered_sources"),
        where="source_selection.unused_buffered_sources",
    )
    if consumed != output_samples + unused:
        raise ValueError("source rows consumed must equal output plus buffered sources")
    relations = tuple(graph_relations)
    if receipt.get("required_graph_relations") != list(relations):
        raise ValueError("objective source selection graph relations drifted")

    raw_windows = receipt.get("windows")
    if not isinstance(raw_windows, list):
        raise ValueError("objective source selection windows must be a list")
    if len(raw_windows) != output_samples // quota_window_samples:
        raise ValueError("objective source selection window count is invalid")
    if receipt.get("windows_sha256") != canonical_schedule_receipt_sha256(
        raw_windows
    ):
        raise ValueError("objective source selection window digest is invalid")
    schedule = _receipt_mapping(receipt.get("schedule"), where="source_selection.schedule")
    if schedule != {
        "schema": OBJECTIVE_SCHEDULE_RECEIPT_SCHEMA,
        "algorithm": OBJECTIVE_SCHEDULE_ALGORITHM,
        "windows_sha256": receipt["windows_sha256"],
    }:
        raise ValueError("objective source selection schedule binding is invalid")

    selected_sources: set[int] = set()
    previous_consumed = 0
    expected_window_keys = {
        "schema",
        "algorithm",
        "start_step",
        "output_samples",
        "source_pool_samples",
        "source_rows_consumed",
        "selected_source_indices",
        "task_counts",
        "assignments",
        "graph_positive_assignments",
        "graph_positive_edges",
    }
    assignment_keys = {
        "source_index",
        "source_pool_index",
        "task",
        "graph_eligibility",
    }
    graph_keys = {
        "schema",
        "objective",
        "eligible",
        "reason",
        "positive_edges",
        "relations",
        "route_mode",
        "route_receipt",
    }
    for window_index, raw_window in enumerate(raw_windows):
        window = _receipt_mapping(
            raw_window,
            where=f"source_selection.windows[{window_index}]",
        )
        if set(window) != expected_window_keys:
            raise ValueError(f"objective schedule window {window_index} keys are invalid")
        if window.get("schema") != OBJECTIVE_SCHEDULE_WINDOW_SCHEMA:
            raise ValueError(f"objective schedule window {window_index} schema is invalid")
        if window.get("algorithm") != OBJECTIVE_SCHEDULE_ALGORITHM:
            raise ValueError(
                f"objective schedule window {window_index} algorithm is invalid"
            )
        if window.get("start_step") != window_index * quota_window_samples:
            raise ValueError(f"objective schedule window {window_index} start step drifted")
        if window.get("output_samples") != quota_window_samples:
            raise ValueError(f"objective schedule window {window_index} size drifted")
        pool_samples = _receipt_int(
            window.get("source_pool_samples"),
            where=f"source_selection.windows[{window_index}].source_pool_samples",
            minimum=quota_window_samples,
        )
        if pool_samples > max_pool:
            raise ValueError(f"objective schedule window {window_index} pool is unbounded")
        window_consumed = _receipt_int(
            window.get("source_rows_consumed"),
            where=f"source_selection.windows[{window_index}].source_rows_consumed",
            minimum=previous_consumed,
        )
        if window_consumed > consumed:
            raise ValueError(f"objective schedule window {window_index} over-consumed")
        previous_consumed = window_consumed
        raw_assignments = window.get("assignments")
        if not isinstance(raw_assignments, list) or len(raw_assignments) != quota_window_samples:
            raise ValueError(f"objective schedule window {window_index} assignments are invalid")
        assignments = [
            _receipt_mapping(
                raw_assignment,
                where=(
                    f"source_selection.windows[{window_index}].assignments["
                    f"{assignment_index}]"
                ),
            )
            for assignment_index, raw_assignment in enumerate(raw_assignments)
        ]
        if any(set(assignment) != assignment_keys for assignment in assignments):
            raise ValueError(f"objective schedule window {window_index} assignment keys drifted")
        source_indices = [
            _receipt_int(
                assignment.get("source_index"),
                where=(
                    f"source_selection.windows[{window_index}].assignments.source_index"
                ),
            )
            for assignment in assignments
        ]
        if len(set(source_indices)) != quota_window_samples:
            raise ValueError(f"objective schedule window {window_index} reuses a source")
        if selected_sources.intersection(source_indices):
            raise ValueError("objective source selection reuses a source across windows")
        selected_sources.update(source_indices)
        if window.get("selected_source_indices") != source_indices:
            raise ValueError(f"objective schedule window {window_index} selection drifted")
        task_counts = Counter(str(assignment.get("task")) for assignment in assignments)
        if window.get("task_counts") != dict(sorted(task_counts.items())):
            raise ValueError(f"objective schedule window {window_index} task counts drifted")
        if expected_window_quotas is not None and task_counts != Counter(
            expected_window_quotas
        ):
            raise ValueError(f"objective schedule window {window_index} quotas drifted")

        positive_assignments = 0
        positive_edges = 0
        for assignment_index, assignment in enumerate(assignments):
            pool_index = _receipt_int(
                assignment.get("source_pool_index"),
                where=(
                    f"source_selection.windows[{window_index}].assignments["
                    f"{assignment_index}].source_pool_index"
                ),
            )
            if pool_index >= pool_samples:
                raise ValueError(
                    f"objective schedule window {window_index} pool index is invalid"
                )
            task = assignment.get("task")
            graph_receipt = assignment.get("graph_eligibility")
            if not relations:
                if graph_receipt is not None:
                    raise ValueError("graph eligibility must be absent without relations")
                continue
            graph = _receipt_mapping(
                graph_receipt,
                where=(
                    f"source_selection.windows[{window_index}].assignments["
                    f"{assignment_index}].graph_eligibility"
                ),
            )
            graph_fields = set(graph)
            if graph_fields != graph_keys and graph_fields != graph_keys | {"detail"}:
                raise ValueError("objective graph eligibility receipt keys are invalid")
            if graph.get("schema") != GRAPH_ELIGIBILITY_RECEIPT_SCHEMA:
                raise ValueError("objective graph eligibility receipt schema is invalid")
            if graph.get("objective") != task:
                raise ValueError("objective graph eligibility task binding drifted")
            if graph.get("relations") != list(relations):
                raise ValueError("objective graph eligibility relations drifted")
            edges = _receipt_int(
                graph.get("positive_edges"),
                where="objective graph eligibility positive_edges",
            )
            eligible = graph.get("eligible")
            if not isinstance(eligible, bool) or eligible != (edges > 0):
                raise ValueError("objective graph eligibility verdict is inconsistent")
            reason = graph.get("reason")
            if eligible:
                if reason is not None:
                    raise ValueError("graph-positive assignment must not have a reason")
                positive_assignments += 1
                positive_edges += edges
            elif not isinstance(reason, str) or not reason:
                raise ValueError("graph-ineligible assignment requires an explicit reason")
            route_mode = graph.get("route_mode")
            route_receipt = graph.get("route_receipt")
            if route_mode == "unavailable":
                if reason != "missing_exact_source_token_route_map" or route_receipt is not None:
                    raise ValueError("unavailable graph route receipt is inconsistent")
            else:
                route = _receipt_mapping(
                    route_receipt,
                    where="objective graph eligibility route_receipt",
                )
                if route.get("mode") != route_mode:
                    raise ValueError("objective graph route mode binding drifted")
            if task in {TaskKind.COMMIT_DIFF.value, TaskKind.PRE_TO_POST.value} and (
                eligible
                or route_mode != "excluded"
                or reason != "exact_source_route_map_unavailable"
            ):
                raise ValueError("commit objective graph eligibility must be excluded")
        if window.get("graph_positive_assignments") != positive_assignments:
            raise ValueError(f"objective schedule window {window_index} graph count drifted")
        if window.get("graph_positive_edges") != positive_edges:
            raise ValueError(f"objective schedule window {window_index} edge count drifted")
        if relations and positive_assignments < 1:
            raise ValueError(f"objective schedule window {window_index} has no graph-positive assignment")

    if previous_consumed != consumed:
        raise ValueError("final objective schedule window consumption drifted")
    resume = _receipt_mapping(receipt.get("resume"), where="source_selection.resume")
    if resume.get("schema") != OBJECTIVE_SOURCE_RESUME_SCHEMA:
        raise ValueError("objective source resume schema is invalid")
    if resume.get("cursor_semantics") != (
        "replay_buffered_rows_then_continue_after_last_yielded_v1"
    ):
        raise ValueError("objective source resume semantics are invalid")
    last_cursor = _receipt_mapping(
        resume.get("last_yielded_cursor"),
        where="source_selection.resume.last_yielded_cursor",
    )
    if last_cursor.get("source_index") != consumed - 1:
        raise ValueError("objective source resume cursor is not the final consumed row")
    buffered = resume.get("buffered_source_cursors")
    if not isinstance(buffered, list) or len(buffered) != unused:
        raise ValueError("objective source buffered resume cursors are invalid")


def _objective_source(source: SourceInput) -> ObjectiveSource:
    if isinstance(source, ObjectiveSource):
        return source
    if isinstance(source, CodePacket):
        return ObjectiveSource(code_packet=source)
    if isinstance(source, CommitPacket):
        return ObjectiveSource(commit_packet=source)
    raise TypeError(f"unsupported objective source {type(source).__name__}")


def _code_packet_for_task(
    source: SourceInput,
    task: TaskKind,
) -> CodePacket | None:
    if task in _COMMIT_TASKS:
        return None
    if isinstance(source, ObjectiveSource):
        return source.code_packet
    return source if isinstance(source, CodePacket) else None


def source_has_graph_candidate(
    source: SourceInput,
    task: TaskKind,
    *,
    graph_relations: Sequence[str],
) -> bool:
    """Cheap pre-match filter; final eligibility is always post-materialization."""

    packet = _code_packet_for_task(source, task)
    if packet is None or task in _COMMIT_TASKS:
        return False
    for relation in graph_relations:
        field = _GRAPH_PACKET_FIELDS[relation]
        edge = getattr(packet, field)
        if edge is None:
            continue
        if relation in {"call", "type"}:
            if edge.to_pairs():
                return True
        elif edge.to_triples():
            return True
    return False


def _unavailable_graph_receipt(
    realized: RealizedObjective,
    *,
    graph_relations: Sequence[str],
    reason: str,
    detail: str | None = None,
) -> dict[str, object]:
    receipt: dict[str, object] = {
        "schema": GRAPH_ELIGIBILITY_RECEIPT_SCHEMA,
        "objective": realized.task.value,
        "eligible": False,
        "reason": reason,
        "positive_edges": 0,
        "relations": list(graph_relations),
        "route_mode": "unavailable",
        "route_receipt": None,
    }
    if detail is not None:
        receipt["detail"] = detail
    return receipt


def assess_graph_positive_capability(
    realized: RealizedObjective,
    source: SourceInput,
    *,
    graph_relations: Sequence[str],
    require_route_sidecars: bool = False,
) -> dict[str, object]:
    """Return an explicit receipt for the realized objective's graph capability."""

    relations = tuple(graph_relations)
    if not relations:
        raise ValueError("graph capability assessment requires graph relations")
    unknown = sorted(set(relations) - set(OBJECTIVE_GRAPH_RELATION_COLUMNS))
    if unknown:
        raise ValueError(f"unsupported graph relations: {unknown}")

    if realized.task in _TRANSFORMED_TASKS:
        raw_map = realized.example.metadata.get("source_token_indices")
        expected_length = int(realized.example.input_ids.shape[0]) + 1
        if not isinstance(raw_map, (tuple, list)) or len(raw_map) != expected_length:
            return _unavailable_graph_receipt(
                realized,
                graph_relations=relations,
                reason="missing_exact_source_token_route_map",
                detail=(
                    f"expected {expected_length} source-token entries, got "
                    f"{0 if not isinstance(raw_map, (tuple, list)) else len(raw_map)}"
                ),
            )

    source_object = _objective_source(source)
    packet = _code_packet_for_task(source_object, realized.task)
    if realized.task in _COMMIT_TASKS:
        route_remap = exclude_objective_routes(
            packet,
            where=realized.task.value,
            reason="independently_tokenized_commit_sections_have_no_exact_source_map",
            require_sidecars=require_route_sidecars,
        )
        route_receipt = route_remap.receipt
    else:
        if packet is None:
            return _unavailable_graph_receipt(
                realized,
                graph_relations=relations,
                reason="missing_exact_source_token_route_map",
                detail="code objective has no CodePacket",
            )
        raw_map = realized.example.metadata.get(SOURCE_TOKEN_INDICES_METADATA_KEY)
        if not isinstance(raw_map, (tuple, list)):
            return _unavailable_graph_receipt(
                realized,
                graph_relations=relations,
                reason="missing_exact_source_token_route_map",
            )
        source_token_indices = [int(value) for value in raw_map]
        try:
            route_remap = remap_objective_routes(
                packet,
                source_token_indices=source_token_indices,
                where=realized.task.value,
                require_sidecars=require_route_sidecars,
                mode=(
                    "identity"
                    if realized.task is TaskKind.CAUSAL_LM
                    else "source_token_remap"
                ),
            )
        except ValueError as error:
            if realized.task not in _TRANSFORMED_TASKS:
                raise
            return _unavailable_graph_receipt(
                realized,
                graph_relations=relations,
                reason="missing_exact_source_token_route_map",
                detail=str(error),
            )
        route_receipt = route_remap.receipt
    if not isinstance(route_receipt, Mapping):
        return _unavailable_graph_receipt(
            realized,
            graph_relations=relations,
            reason="missing_exact_route_receipt",
        )
    route_mode = route_receipt.get("mode")
    if realized.task in _COMMIT_TASKS:
        return {
            "schema": GRAPH_ELIGIBILITY_RECEIPT_SCHEMA,
            "objective": realized.task.value,
            "eligible": False,
            "reason": "exact_source_route_map_unavailable",
            "positive_edges": 0,
            "relations": list(relations),
            "route_mode": route_mode,
            "route_receipt": dict(route_receipt),
        }
    if realized.task in _TRANSFORMED_TASKS and route_mode != "source_token_remap":
        return {
            "schema": GRAPH_ELIGIBILITY_RECEIPT_SCHEMA,
            "objective": realized.task.value,
            "eligible": False,
            "reason": "exact_source_route_map_unavailable",
            "positive_edges": 0,
            "relations": list(relations),
            "route_mode": route_mode,
            "route_receipt": dict(route_receipt),
        }

    assert packet is not None
    raw_map = realized.example.metadata.get(SOURCE_TOKEN_INDICES_METADATA_KEY)
    if not isinstance(raw_map, (tuple, list)):
        return _unavailable_graph_receipt(
            realized,
            graph_relations=relations,
            reason="missing_exact_source_token_route_map",
        )
    source_token_indices = [int(value) for value in raw_map]
    source_length = int(packet.token_ids.shape[0])
    if packet.document_ids is None:
        source_document_ids = [1] * source_length
    else:
        source_document_ids = [int(value) for value in packet.document_ids.tolist()]
    output_document_ids: list[int | None] = []
    for source_index in source_token_indices:
        if source_index < 0:
            output_document_ids.append(None)
        elif source_index >= source_length:
            return _unavailable_graph_receipt(
                realized,
                graph_relations=relations,
                reason="missing_exact_source_token_route_map",
                detail=f"source index {source_index} is outside {source_length}",
            )
        else:
            output_document_ids.append(source_document_ids[source_index])
    for index, value in enumerate(output_document_ids):
        if value is not None:
            continue
        left = next(
            (output_document_ids[position] for position in range(index - 1, -1, -1)
             if output_document_ids[position] is not None),
            None,
        )
        right = next(
            (
                output_document_ids[position]
                for position in range(index + 1, len(output_document_ids))
                if output_document_ids[position] is not None
            ),
            None,
        )
        output_document_ids[index] = left if left is not None else right
    if any(value is None for value in output_document_ids):
        return _unavailable_graph_receipt(
            realized,
            graph_relations=relations,
            reason="missing_exact_source_token_route_map",
        )
    from cppmega_mlx.training.megatron_objectives import MaterializedMegatronDocument

    inputs = [int(value) for value in realized.example.input_ids.tolist()]
    targets = [int(value) for value in realized.example.target_ids.tolist()]
    mask = [int(value) for value in realized.example.loss_mask.tolist()]
    document = MaterializedMegatronDocument(
        objective_kind=realized.task.value,
        token_ids=[inputs[0], *targets],
        loss_mask=[*mask, 0],
        graph_edge_count=sum(
            len(route_remap.columns[column])
            for column in OBJECTIVE_GRAPH_RELATION_COLUMNS.values()
        ),
        row={
            "doc_ids": [int(value) for value in output_document_ids],
            **route_remap.columns,
        },
        route_receipt=route_receipt,
    )
    positive_edges = count_configured_graph_positive_edges(
        document,
        relations=relations,
    )
    eligible = positive_edges > 0
    return {
        "schema": GRAPH_ELIGIBILITY_RECEIPT_SCHEMA,
        "objective": realized.task.value,
        "eligible": eligible,
        "reason": (
            None
            if eligible
            else "no_configured_graph_positive_causal_same_document_pair"
        ),
        "positive_edges": int(positive_edges),
        "relations": list(relations),
        "route_mode": route_mode,
        "route_receipt": dict(route_receipt),
    }


def validate_graph_receipt_against_document(
    receipt: Mapping[str, object],
    document: object,
    *,
    graph_relations: Sequence[str],
) -> None:
    """Ensure the probe receipt and final materialized document are identical."""

    positive_edges = count_configured_graph_positive_edges(
        document,
        relations=graph_relations,
    )
    if receipt.get("positive_edges") != positive_edges:
        raise RuntimeError(
            "objective graph eligibility receipt disagrees with final materialization: "
            f"receipt={receipt.get('positive_edges')}, document={positive_edges}"
        )
    if receipt.get("eligible") != (positive_edges > 0):
        raise RuntimeError(
            "objective graph eligibility verdict disagrees with final materialization"
        )
    route_receipt = getattr(document, "route_receipt", None)
    route_mode = receipt.get("route_mode")
    if isinstance(route_receipt, Mapping) and route_receipt.get("mode") != route_mode:
        raise RuntimeError(
            "objective graph route mode disagrees with final materialization: "
            f"receipt={route_mode}, document={route_receipt.get('mode')}"
        )


@dataclass(frozen=True)
class ScheduledObjective:
    source_index: int
    source: SourceInput
    realized: RealizedObjective
    graph_eligibility: Mapping[str, object] | None


@dataclass(frozen=True)
class ObjectiveScheduleWindow:
    assignments: tuple[ScheduledObjective, ...]
    receipt: dict[str, object]


class CanonicalObjectivePlanner:
    """Own source-pool lookahead, exact quotas, and graph-positive selection."""

    def __init__(
        self,
        *,
        mixer: EligibilityAwareTaskMixer,
        source_iter: Iterator[SourceInput],
        quota_window_samples: int,
        quota_lookahead_samples: int,
        graph_relations: Sequence[str] = (),
        require_route_sidecars: bool = False,
    ) -> None:
        if quota_window_samples < 1:
            raise ValueError("quota_window_samples must be >=1")
        if quota_lookahead_samples < 0:
            raise ValueError("quota_lookahead_samples must be >=0")
        relations = tuple(graph_relations)
        unknown = sorted(set(relations) - set(OBJECTIVE_GRAPH_RELATION_COLUMNS))
        if unknown:
            raise ValueError(f"unsupported graph relations: {unknown}")
        if len(set(relations)) != len(relations):
            raise ValueError("graph relations must not contain duplicates")
        self.mixer = mixer
        self.source_iter = iter(source_iter)
        self.quota_window_samples = int(quota_window_samples)
        self.quota_lookahead_samples = int(quota_lookahead_samples)
        self.graph_relations = relations
        self.require_route_sidecars = require_route_sidecars
        self.max_source_pool_samples = (
            self.quota_window_samples + self.quota_lookahead_samples
        )
        self._source_pool: list[SourceInput] = []
        self._source_indices: list[int] = []
        self._source_cursor_pool: list[dict[str, int]] = []
        self._source_rows_consumed = 0
        self._max_source_pool_observed = 0
        self._last_yielded_cursor: dict[str, int] | None = None
        self._windows: list[dict[str, object]] = []

    @property
    def source_rows_consumed(self) -> int:
        return self._source_rows_consumed

    @property
    def unused_buffered_sources(self) -> int:
        return len(self._source_pool)

    @property
    def last_yielded_cursor(self) -> dict[str, int] | None:
        return (
            None
            if self._last_yielded_cursor is None
            else dict(self._last_yielded_cursor)
        )

    @property
    def windows(self) -> tuple[dict[str, object], ...]:
        return tuple(self._windows)

    def _append_source(self) -> None:
        source = next(self.source_iter)
        raw_cursor = getattr(self.source_iter, "last_cursor", None)
        cursor = (
            {"source_index": self._source_rows_consumed}
            if raw_cursor is None
            else dict(raw_cursor)
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in cursor.values()
        ):
            raise ValueError("objective source cursor values must be non-negative integers")
        if int(cursor.get("source_index", -1)) != self._source_rows_consumed:
            raise ValueError(
                "objective source iterator cursor is not aligned to yielded rows: "
                f"cursor={cursor.get('source_index')}, "
                f"expected={self._source_rows_consumed}"
            )
        self._source_pool.append(source)
        self._source_indices.append(self._source_rows_consumed)
        self._source_cursor_pool.append(cursor)
        self._last_yielded_cursor = cursor
        self._source_rows_consumed += 1
        self._max_source_pool_observed = max(
            self._max_source_pool_observed,
            len(self._source_pool),
        )

    def _ensure_window_pool(self) -> None:
        while len(self._source_pool) < self.quota_window_samples:
            self._append_source()

    @staticmethod
    def _receipt_key(item: RealizedObjective) -> tuple[object, ...]:
        raw_map = item.example.metadata.get("source_token_indices")
        return (
            item.source_index,
            item.task.value,
            tuple(int(value) for value in item.example.input_ids.tolist()),
            tuple(raw_map) if isinstance(raw_map, (tuple, list)) else None,
        )

    def plan_window(self, *, start_step: int = 0) -> ObjectiveScheduleWindow:
        self._ensure_window_pool()
        graph_receipts: dict[tuple[object, ...], dict[str, object]] = {}

        def assess(source: SourceInput, item: RealizedObjective) -> bool:
            key = self._receipt_key(item)
            receipt = graph_receipts.get(key)
            if receipt is None:
                receipt = assess_graph_positive_capability(
                    item,
                    source,
                    graph_relations=self.graph_relations,
                    require_route_sidecars=self.require_route_sidecars,
                )
                graph_receipts[key] = receipt
            return bool(receipt["eligible"])

        while True:
            try:
                kwargs: dict[str, object] = {}
                if self.graph_relations:
                    kwargs = {
                        "required_realized_assignment": assess,
                        "candidate_assignment": (
                            lambda source, task: source_has_graph_candidate(
                                source,
                                task,
                                graph_relations=self.graph_relations,
                            )
                        ),
                    }
                realized = self.mixer.materialize_window_from_pool(
                    self._source_pool,
                    output_count=self.quota_window_samples,
                    start_step=start_step,
                    **kwargs,
                )
                break
            except ObjectiveQuotaUnsatisfiedError as error:
                if len(self._source_pool) >= self.max_source_pool_samples:
                    raise ObjectiveQuotaUnsatisfiedError(
                        "objective quota remained unsatisfied at the bounded "
                        "lookahead limit: "
                        f"start_step={start_step}, pool={len(self._source_pool)}, "
                        f"window={self.quota_window_samples}, "
                        f"lookahead={self.quota_lookahead_samples}; {error}"
                    ) from error
                self._append_source()

        selected_local_indices = [item.source_index for item in realized]
        if len(selected_local_indices) != self.quota_window_samples or len(
            set(selected_local_indices)
        ) != self.quota_window_samples:
            raise RuntimeError(
                "canonical objective planner selected the wrong number of sources: "
                f"selected={selected_local_indices}, "
                f"expected={self.quota_window_samples}"
            )

        assignments: list[ScheduledObjective] = []
        for item in realized:
            source = self._source_pool[item.source_index]
            graph_receipt = None
            if self.graph_relations:
                key = self._receipt_key(item)
                graph_receipt = graph_receipts.get(key)
                if graph_receipt is None:
                    assess(source, item)
                    graph_receipt = graph_receipts[key]
            assignments.append(
                ScheduledObjective(
                    source_index=self._source_indices[item.source_index],
                    source=source,
                    realized=item,
                    graph_eligibility=graph_receipt,
                )
            )

        task_counts = Counter(
            getattr(getattr(item.realized, "task", None), "value", "unknown")
            for item in assignments
        )
        graph_positive_assignments = sum(
            bool(item.graph_eligibility and item.graph_eligibility["eligible"])
            for item in assignments
        )
        graph_positive_edges = sum(
            int(item.graph_eligibility["positive_edges"])
            for item in assignments
            if item.graph_eligibility is not None
        )
        window_receipt: dict[str, object] = {
            "schema": OBJECTIVE_SCHEDULE_WINDOW_SCHEMA,
            "algorithm": OBJECTIVE_SCHEDULE_ALGORITHM,
            "start_step": int(start_step),
            "output_samples": self.quota_window_samples,
            "source_pool_samples": len(self._source_pool),
            "source_rows_consumed": self._source_rows_consumed,
            "selected_source_indices": [item.source_index for item in assignments],
            "task_counts": dict(
                sorted(task_counts.items(), key=lambda item: item[0])
            ),
            "assignments": [
                {
                    "source_index": item.source_index,
                    "source_pool_index": item.realized.source_index,
                    "task": getattr(
                        getattr(item.realized, "task", None),
                        "value",
                        "unknown",
                    ),
                    "graph_eligibility": (
                        None
                        if item.graph_eligibility is None
                        else dict(item.graph_eligibility)
                    ),
                }
                for item in assignments
            ],
            "graph_positive_assignments": int(graph_positive_assignments),
            "graph_positive_edges": int(graph_positive_edges),
        }
        if self.graph_relations and graph_positive_assignments < 1:
            raise RuntimeError(
                "canonical objective planner returned a window without a "
                "graph-positive assignment"
            )
        self._windows.append(window_receipt)

        selected = set(selected_local_indices)
        retained = [
            (source, source_index, cursor)
            for local_index, (source, source_index, cursor) in enumerate(
                zip(
                    self._source_pool,
                    self._source_indices,
                    self._source_cursor_pool,
                    strict=True,
                )
            )
            if local_index not in selected
        ]
        self._source_pool = [source for source, _index, _cursor in retained]
        self._source_indices = [index for _source, index, _cursor in retained]
        self._source_cursor_pool = [cursor for _source, _index, cursor in retained]
        return ObjectiveScheduleWindow(tuple(assignments), window_receipt)

    def source_selection_receipt(self, *, output_samples: int) -> dict[str, object]:
        if output_samples < 1:
            raise ValueError("source selection output_samples must be positive")
        if self._last_yielded_cursor is None:
            raise RuntimeError("objective planner consumed no source rows")
        windows = [dict(window) for window in self._windows]
        return {
            "schema": OBJECTIVE_SOURCE_SELECTION_SCHEMA,
            "algorithm": OBJECTIVE_SCHEDULE_ALGORITHM,
            "output_samples": int(output_samples),
            "source_rows_consumed": self._source_rows_consumed,
            "unused_buffered_sources": len(self._source_pool),
            "quota_window_samples": self.quota_window_samples,
            "quota_lookahead_samples": self.quota_lookahead_samples,
            "max_source_pool_samples": self.max_source_pool_samples,
            "max_source_pool_observed": self._max_source_pool_observed,
            "required_graph_relations": list(self.graph_relations),
            "windows": windows,
            "windows_sha256": canonical_schedule_receipt_sha256(windows),
            "resume": {
                "schema": OBJECTIVE_SOURCE_RESUME_SCHEMA,
                "cursor_semantics": (
                    "replay_buffered_rows_then_continue_after_last_yielded_v1"
                ),
                "last_yielded_cursor": dict(self._last_yielded_cursor),
                "buffered_source_cursors": [
                    dict(cursor) for cursor in self._source_cursor_pool
                ],
            },
            "schedule": {
                "schema": OBJECTIVE_SCHEDULE_RECEIPT_SCHEMA,
                "algorithm": OBJECTIVE_SCHEDULE_ALGORITHM,
                "windows_sha256": canonical_schedule_receipt_sha256(windows),
            },
        }


__all__ = [
    "CanonicalObjectivePlanner",
    "GRAPH_ELIGIBILITY_RECEIPT_SCHEMA",
    "OBJECTIVE_SCHEDULE_ALGORITHM",
    "OBJECTIVE_SCHEDULE_RECEIPT_SCHEMA",
    "OBJECTIVE_SCHEDULE_WINDOW_SCHEMA",
    "OBJECTIVE_SOURCE_RESUME_SCHEMA",
    "OBJECTIVE_SOURCE_SELECTION_SCHEMA",
    "ObjectiveScheduleWindow",
    "ScheduledObjective",
    "assess_graph_positive_capability",
    "canonical_window_quotas",
    "canonical_schedule_receipt_sha256",
    "source_has_graph_candidate",
    "validate_graph_receipt_against_document",
    "validate_objective_source_selection_receipt",
]

"""Incremental accounting for pre-materialized Megatron objectives."""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
import math

from cppmega_mlx.training.megatron_objectives import (
    MaterializedMegatronDocument,
    OBJECTIVE_CONTRACT_SCHEMA,
    OBJECTIVE_KIND_IDS,
    objective_route_mapping_contract,
)
from cppmega_mlx.data.batch import (
    LOSS_MASK_ALIGNMENT_SOURCE_TOKEN_PREDICTS_NEXT_V1,
)
from cppmega_mlx.training.objective_data import (
    OBJECTIVE_GRAPH_RELATION_COLUMNS,
    OBJECTIVE_ROUTE_COUNT_FIELDS,
    OBJECTIVE_ROUTE_RECEIPT_SCHEMA,
    OBJECTIVE_ROUTE_RETENTION_SCHEMA,
)
from cppmega_mlx.training.objective_mixer import GraphAuxLossConfig
from cppmega_mlx.training.task_mixer import TaskKind

_KNOWN_TASKS = frozenset(task.value for task in TaskKind)
_REQUIRED_PRODUCTION_TASKS = frozenset(
    {
        TaskKind.CAUSAL_LM.value,
        TaskKind.FIM.value,
        TaskKind.AST_FIM.value,
        TaskKind.IFIM.value,
        TaskKind.COMMIT_DIFF.value,
        TaskKind.PRE_TO_POST.value,
    }
)
_GRAPH_RELATION_COLUMNS = {
    "call": "token_call_edges",
    "type": "token_type_edges",
    "domain": "token_domain_edges",
    "build": "token_build_edges",
    "shell": "token_shell_edges",
    "diagnostic": "token_diagnostic_edges",
    "cross_domain": "token_cross_domain_edges",
}
_CHUNK_RELATIONS = frozenset({"call", "type"})


def _fraction_string(value: Fraction) -> str:
    return (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )


def _exact_rates(
    rates: Mapping[TaskKind | str, float],
) -> tuple[tuple[str, ...], dict[str, Fraction]]:
    by_task: dict[str, Fraction] = {}
    seen: set[str] = set()
    for key, value in rates.items():
        task = key.value if isinstance(key, TaskKind) else str(key)
        if task in seen:
            raise ValueError(f"duplicate objective rate for {task}")
        seen.add(task)
        if task not in _KNOWN_TASKS:
            raise ValueError(f"unknown objective rate {task!r}")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError(
                f"objective rate for {task} must be finite and non-negative"
            )
        fraction = Fraction(str(numeric))
        if fraction > 0:
            by_task[task] = fraction
    raw = [
        (task.value, by_task[task.value]) for task in TaskKind if task.value in by_task
    ]
    if not raw:
        raise ValueError("objective contract rates have no positive task")
    missing = sorted(_REQUIRED_PRODUCTION_TASKS - {task for task, _value in raw})
    if missing:
        raise ValueError(
            f"objective contract is missing production objectives: {missing}"
        )
    if not math.isclose(
        math.fsum(float(value) for _task, value in raw),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("objective contract rates must sum to 1")
    total = sum((value for _task, value in raw), Fraction())
    return tuple(task for task, _value in raw), {
        task: value / total for task, value in raw
    }


def _hamilton(
    task_order: Sequence[str], rates: Mapping[str, Fraction], size: int
) -> dict[str, int]:
    raw = {task: rates[task] * size for task in task_order}
    quotas = {task: math.floor(value) for task, value in raw.items()}
    remaining = size - sum(quotas.values())
    order = {task: index for index, task in enumerate(task_order)}
    ranked = sorted(
        task_order,
        key=lambda task: (-(raw[task] - quotas[task]), order[task]),
    )
    for task in ranked[:remaining]:
        quotas[task] += 1
    return quotas


def _merged_intervals(
    intervals: Iterable[tuple[int, int]], *, upper_bound: int
) -> Iterable[tuple[int, int]]:
    clipped = sorted(
        (max(0, start), min(end, upper_bound))
        for start, end in intervals
        if max(0, start) < min(end, upper_bound)
    )
    if not clipped:
        return
    merged_start, merged_end = clipped[0]
    for start, end in clipped[1:]:
        if start <= merged_end:
            merged_end = max(merged_end, end)
            continue
        yield merged_start, merged_end
        merged_start, merged_end = start, end
    yield merged_start, merged_end


def count_configured_graph_positive_edges(
    document: MaterializedMegatronDocument,
    *,
    relations: Sequence[str],
) -> int:
    """Count unique causal same-document graph pairs without expanding rectangles."""

    input_length = len(document.token_ids) - 1
    doc_ids = document.row.get("doc_ids")
    if not isinstance(doc_ids, list) or len(doc_ids) < input_length:
        raise ValueError("materialized graph accounting requires aligned doc_ids")

    positions_by_document: defaultdict[int, list[int]] = defaultdict(list)
    for position in range(input_length):
        document_id = int(doc_ids[position])
        if document_id > 0:
            positions_by_document[document_id].append(position)

    starts_at: defaultdict[int, list[tuple[int, int, int]]] = defaultdict(list)
    ends_at: defaultdict[int, list[int]] = defaultdict(list)
    direct_by_query: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
    interval_id = 0

    for relation in relations:
        column = _GRAPH_RELATION_COLUMNS[relation]
        edges = document.row.get(column)
        if not isinstance(edges, list):
            raise ValueError(
                f"materialized {relation} graph column {column} must be a list"
            )
        if relation not in _CHUNK_RELATIONS:
            for edge in edges:
                query = int(edge["from"])
                key = int(edge["to"])
                if 0 <= key <= query < input_length:
                    direct_by_query[query].append((key, key + 1))
            continue

        starts = document.row.get("token_chunk_starts")
        ends = document.row.get("token_chunk_ends")
        if not isinstance(starts, list) or not isinstance(ends, list):
            raise ValueError("chunk graph accounting requires token_chunk_starts/ends")
        if len(starts) != len(ends):
            raise ValueError("materialized chunk starts/ends length mismatch")
        for edge in edges:
            source = int(edge["from"])
            destination = int(edge["to"])
            if not (0 <= source < len(starts) and 0 <= destination < len(starts)):
                raise ValueError("materialized chunk edge endpoint is invalid")
            query_start = max(0, int(starts[source]))
            query_end = min(int(ends[source]), input_length)
            key_start = max(0, int(starts[destination]))
            key_end = min(int(ends[destination]), input_length)
            if query_start >= query_end or key_start >= key_end:
                continue
            starts_at[query_start].append((interval_id, key_start, key_end))
            ends_at[query_end].append(interval_id)
            interval_id += 1

    active_intervals: dict[int, tuple[int, int]] = {}
    positive_edges = 0
    for query in range(input_length):
        for expired in ends_at.get(query, ()):
            active_intervals.pop(expired, None)
        for added, key_start, key_end in starts_at.get(query, ()):
            active_intervals[added] = (key_start, key_end)

        document_id = int(doc_ids[query])
        if document_id <= 0:
            continue
        intervals: Iterable[tuple[int, int]] = active_intervals.values()
        direct = direct_by_query.get(query)
        if direct:
            intervals = (*intervals, *direct)
        positions = positions_by_document[document_id]
        for start, end in _merged_intervals(intervals, upper_bound=query + 1):
            positive_edges += bisect_left(positions, end) - bisect_left(
                positions, start
            )
    return positive_edges


class ObjectiveContractAccumulator:
    """Build an objective contract in O(tasks) corpus-level state."""

    def __init__(
        self,
        *,
        rates: Mapping[TaskKind | str, float],
        seed: int,
        quota_window_samples: int,
        graph_config: GraphAuxLossConfig,
        graph_weight: float,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("objective contract seed must be a non-negative integer")
        if quota_window_samples < 1:
            raise ValueError("objective quota window must contain at least one sample")
        if not math.isfinite(graph_weight) or graph_weight <= 0.0:
            raise ValueError("graph auxiliary global weight must be > 0")
        if graph_weight != graph_config.global_weight:
            raise ValueError(
                "graph auxiliary global weight differs from GraphAuxLossConfig: "
                f"{graph_weight} != {graph_config.global_weight}"
            )
        unknown_relations = sorted(
            set(graph_config.relations) - set(_GRAPH_RELATION_COLUMNS)
        )
        if unknown_relations:
            raise ValueError(
                f"unsupported Megatron graph auxiliary relations: {unknown_relations}"
            )
        if graph_config.bce_weight <= 0.0 or graph_config.coverage_weight <= 0.0:
            raise ValueError(
                "Megatron graph auxiliary contract requires positive BCE and "
                "coverage weights"
            )

        task_order, exact_rates = _exact_rates(rates)
        self._task_order = task_order
        self._exact_rates = exact_rates
        self._seed = seed
        self._quota_window_samples = quota_window_samples
        self._graph_config = graph_config
        self._window_quotas = _hamilton(task_order, exact_rates, quota_window_samples)
        self._realized = {
            task: {"samples": 0, "input_tokens": 0, "loss_tokens": 0}
            for task in task_order
        }
        self._actual: Counter[str] = Counter()
        self._window_actual: Counter[str] = Counter()
        self._window_positive_edges = 0
        self._samples = 0
        self._eligible_graph_samples = 0
        self._positive_edges = 0
        self._route_receipt_samples = 0
        self._missing_route_receipts = 0
        self._route_retention = {
            task: {
                "samples": 0,
                "modes": Counter(),
                "source_chunks": 0,
                "retained_chunks": 0,
                "dropped_chunks": 0,
                "excluded_chunks": 0,
                "relations": {
                    relation: Counter() for relation in OBJECTIVE_GRAPH_RELATION_COLUMNS
                },
            }
            for task in task_order
        }
        self._finalized = False

    @property
    def samples(self) -> int:
        return self._samples

    def validate_sample_count(self, samples: int) -> dict[str, int]:
        """Validate complete quota windows before reading document payloads."""

        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
            raise ValueError("cannot build objective contract for no documents")
        if samples % self._quota_window_samples:
            raise ValueError(
                "document count must contain complete objective quota windows"
            )
        windows = samples // self._quota_window_samples
        planned = {task: count * windows for task, count in self._window_quotas.items()}
        zero_quota = [task for task, count in planned.items() if count == 0]
        if zero_quota:
            raise ValueError(f"configured objectives received zero quota: {zero_quota}")
        return planned

    def add(self, document: MaterializedMegatronDocument) -> None:
        if self._finalized:
            raise RuntimeError("cannot add documents after contract finalization")
        task = document.objective_kind
        if task not in self._realized:
            raise ValueError(f"realized objective has no configured quota: {task!r}")

        row = self._realized[task]
        row["samples"] += 1
        row["input_tokens"] += len(document.token_ids) - 1
        row["loss_tokens"] += sum(document.loss_mask[:-1])
        self._actual[task] += 1
        self._window_actual[task] += 1
        self._samples += 1

        if document.route_receipt is None:
            self._missing_route_receipts += 1
        else:
            self._add_route_receipt(task, document.route_receipt)

        graph_edges = count_configured_graph_positive_edges(
            document,
            relations=self._graph_config.relations,
        )
        if graph_edges > 0:
            self._eligible_graph_samples += 1
            self._positive_edges += graph_edges
            self._window_positive_edges += graph_edges

        if self._samples % self._quota_window_samples == 0:
            if self._window_actual != Counter(self._window_quotas):
                raise ValueError(
                    "realized objective window differs from deterministic quotas: "
                    f"realized={dict(self._window_actual)}, "
                    f"planned={self._window_quotas}"
                )
            if self._window_positive_edges < 1:
                raise ValueError(
                    "objective quota window has no materialized graph-positive "
                    "causal same-document pair"
                )
            self._window_actual.clear()
            self._window_positive_edges = 0

    def _add_route_receipt(
        self,
        task: str,
        receipt: Mapping[str, object],
    ) -> None:
        mode = receipt.get("mode")
        expected_mode = (
            "identity"
            if task == TaskKind.CAUSAL_LM.value
            else "excluded"
            if task in {TaskKind.COMMIT_DIFF.value, TaskKind.PRE_TO_POST.value}
            else "source_token_remap"
        )
        if receipt.get("schema") != OBJECTIVE_ROUTE_RECEIPT_SCHEMA:
            raise ValueError(f"{task}: objective route receipt schema is invalid")
        if mode != expected_mode:
            raise ValueError(
                f"{task}: objective route receipt mode {mode!r} != {expected_mode!r}"
            )
        expected_keys = {
            "schema",
            "mode",
            "source_chunks",
            "retained_chunks",
            "dropped_chunks",
            "excluded_chunks",
            "relations",
        }
        if mode == "excluded":
            expected_keys.add("reason")
            expected_reason = objective_route_mapping_contract()["explicit_exclusions"]
            assert isinstance(expected_reason, Mapping)
            if receipt.get("reason") != expected_reason[task]:
                raise ValueError(f"{task}: objective route exclusion reason is invalid")
        if set(receipt) != expected_keys:
            raise ValueError(
                f"{task}: objective route receipt keys must be {sorted(expected_keys)}"
            )

        aggregate = self._route_retention[task]
        aggregate["samples"] += 1
        aggregate["modes"][str(mode)] += 1
        for field in (
            "source_chunks",
            "retained_chunks",
            "dropped_chunks",
            "excluded_chunks",
        ):
            value = receipt.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{task}: route receipt {field} must be non-negative")
            aggregate[field] += value

        relations = receipt.get("relations")
        if not isinstance(relations, Mapping) or set(relations) != set(
            OBJECTIVE_GRAPH_RELATION_COLUMNS
        ):
            raise ValueError(f"{task}: objective route receipt relations are invalid")
        for relation in OBJECTIVE_GRAPH_RELATION_COLUMNS:
            counts = relations[relation]
            if not isinstance(counts, Mapping) or set(counts) != set(
                OBJECTIVE_ROUTE_COUNT_FIELDS
            ):
                raise ValueError(
                    f"{task}: objective route receipt {relation} counts are invalid"
                )
            for field in OBJECTIVE_ROUTE_COUNT_FIELDS:
                value = counts[field]
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(
                        f"{task}: route receipt {relation}.{field} must be non-negative"
                    )
                aggregate["relations"][relation][field] += value
        self._route_receipt_samples += 1

    def _route_retention_contract(self) -> dict[str, object] | None:
        if self._route_receipt_samples and self._missing_route_receipts:
            raise ValueError("objective route receipts are partially missing")
        if not self._route_receipt_samples:
            return None
        if self._route_receipt_samples != self._samples:
            raise ValueError("objective route receipt count differs from samples")
        by_objective: dict[str, object] = {}
        for task in self._task_order:
            aggregate = self._route_retention[task]
            by_objective[task] = {
                "samples": aggregate["samples"],
                "modes": dict(sorted(aggregate["modes"].items())),
                "source_chunks": aggregate["source_chunks"],
                "retained_chunks": aggregate["retained_chunks"],
                "dropped_chunks": aggregate["dropped_chunks"],
                "excluded_chunks": aggregate["excluded_chunks"],
                "relations": {
                    relation: {
                        field: int(aggregate["relations"][relation][field])
                        for field in OBJECTIVE_ROUTE_COUNT_FIELDS
                    }
                    for relation in OBJECTIVE_GRAPH_RELATION_COLUMNS
                },
            }
        return {
            "schema": OBJECTIVE_ROUTE_RETENTION_SCHEMA,
            "by_objective": by_objective,
        }

    def finalize(self) -> dict[str, object]:
        if self._finalized:
            raise RuntimeError("objective contract was already finalized")
        planned = self.validate_sample_count(self._samples)
        if self._actual != Counter(planned):
            raise ValueError(
                "realized objective mix differs from deterministic quotas: "
                f"realized={dict(self._actual)}, planned={planned}"
            )
        if self._eligible_graph_samples < 1 or self._positive_edges < 1:
            raise ValueError(
                "configured graph auxiliary objective has no eligible materialized "
                "samples"
            )

        total_input = sum(row["input_tokens"] for row in self._realized.values())
        total_loss = sum(row["loss_tokens"] for row in self._realized.values())
        graph_config = self._graph_config
        route_retention = self._route_retention_contract()
        graph_auxiliary: dict[str, object] = {
            "relations": list(graph_config.relations),
            "eligible_samples": self._eligible_graph_samples,
            "positive_edges": self._positive_edges,
            "global_weight": _fraction_string(
                Fraction(str(graph_config.global_weight))
            ),
            "indexer_weight": _fraction_string(
                Fraction(str(graph_config.indexer_weight))
            ),
            "layer_weight": _fraction_string(Fraction(str(graph_config.layer_weight))),
            "layer_reduction": "sum",
            "bce_weight": _fraction_string(Fraction(str(graph_config.bce_weight))),
            "coverage_weight": _fraction_string(
                Fraction(str(graph_config.coverage_weight))
            ),
            "topk": graph_config.topk,
            "pos_weight": _fraction_string(Fraction(str(graph_config.pos_weight))),
            "margin": _fraction_string(Fraction(str(graph_config.margin))),
            "included_in_total_loss": True,
            "runtime": "megatron_dsa_indexer_v1",
            "pair_mask": "causal_same_document_upstream_v1",
            "chunk_edge_expansion": "cartesian_token_spans_v1",
            "route_mapping": objective_route_mapping_contract(),
        }
        if route_retention is not None:
            graph_auxiliary["route_retention"] = route_retention
        contract: dict[str, object] = {
            "schema": OBJECTIVE_CONTRACT_SCHEMA,
            "algorithm": "hamilton_eligibility_bipartite_v1",
            "seed": int(self._seed),
            "quota_window_samples": int(self._quota_window_samples),
            "task_order": list(self._task_order),
            "objective_ids": {
                task: OBJECTIVE_KIND_IDS[task] for task in self._task_order
            },
            "configured_rates": {
                task: _fraction_string(self._exact_rates[task])
                for task in self._task_order
            },
            "planned_samples": planned,
            "realized": {task: dict(self._realized[task]) for task in self._task_order},
            "totals": {
                "samples": self._samples,
                "input_tokens": total_input,
                "loss_tokens": total_loss,
            },
            "typed_sources": {
                "ifim_instruction": "ifim_instruction_token_ids",
                "commit_message": "commit_msg_token_ids",
                "diff": "diff_token_ids",
                "pre": "pre_token_ids",
                "post": "post_token_ids",
                "missing_fields": "ineligible",
                "rendered_text_parsing": False,
            },
            "graph_auxiliary": graph_auxiliary,
            "materialization": {
                "format": "shifted_lm_document_v1",
                "token_column": "input_ids",
                "loss_mask_column": "loss_mask",
                "loss_mask_alignment": (
                    LOSS_MASK_ALIGNMENT_SOURCE_TOKEN_PREDICTS_NEXT_V1
                ),
                "length_column": "valid_token_count",
                "objective_column": "objective_kind",
                "document_id_column": "doc_ids",
                "source_document_id_column": "token_source_doc_ids",
            },
        }
        self._finalized = True
        return contract


__all__ = [
    "ObjectiveContractAccumulator",
    "count_configured_graph_positive_edges",
]

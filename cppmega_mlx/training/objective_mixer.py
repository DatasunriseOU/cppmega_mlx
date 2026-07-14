"""Eligibility-aware production objective scheduling and exact accounting."""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import mlx.core as mx
import numpy as np

from cppmega_mlx.data.ast_fim import (
    eligible_ast_chunk_indices,
    logical_document_spans,
)
from cppmega_mlx.data.code_packet import CodePacket
from cppmega_mlx.data.commit_packet import CommitPacket
from cppmega_mlx.data.fim import FIMSpecialTokenInput
from cppmega_mlx.data.graph_packet import GraphPacket
from cppmega_mlx.training.indexer_losses import total_indexer_loss
from cppmega_mlx.training.objectives import ObjectiveExample
from cppmega_mlx.training.task_mixer import TaskKind, TaskMixer, normalize_rates


@dataclass(frozen=True)
class RealizedObjective:
    task: TaskKind
    example: ObjectiveExample
    ineligible: Mapping[TaskKind, str]
    source_index: int


@dataclass(frozen=True)
class ObjectiveSource:
    """One upstream document with independently optional code/commit views."""

    code_packet: CodePacket | None = None
    commit_packet: CommitPacket | None = None

    def __post_init__(self) -> None:
        if self.code_packet is None and self.commit_packet is None:
            raise ValueError("ObjectiveSource requires a code or commit packet")


SourceInput = CodePacket | CommitPacket | ObjectiveSource


def _array_length(value: mx.array | None) -> int:
    return 0 if value is None else int(value.shape[0])


def _eligibility_reason(
    task: TaskKind,
    packet: CodePacket | CommitPacket,
    *,
    max_input_tokens: int | None = None,
) -> str | None:
    if task in {
        TaskKind.CAUSAL_LM,
        TaskKind.FIM,
        TaskKind.AST_FIM,
        TaskKind.IFIM,
        TaskKind.SYMBOL_RECOVERY,
        TaskKind.TYPE_RECOVERY,
        TaskKind.CALLEE_RECOVERY,
    }:
        if not isinstance(packet, CodePacket):
            return "requires CodePacket"
        token_count = int(packet.token_ids.shape[-1])
        document_lengths = [
            end - start
            for start, end, _document_id in logical_document_spans(packet)
        ]
        if task is TaskKind.CAUSAL_LM:
            reason = (
                None
                if any(length >= 2 for length in document_lengths)
                else "requires a logical document with at least 2 tokens"
            )
            estimated_input = token_count - 1
            return _length_reason(reason, estimated_input, max_input_tokens)
        eligible_document_lengths = [
            length for length in document_lengths if length >= 3
        ]
        if not eligible_document_lengths:
            return "requires a logical document with at least 3 tokens"
        selected_token_count = max(eligible_document_lengths)
        if task is TaskKind.FIM:
            return _length_reason(None, selected_token_count + 3, max_input_tokens)
        if task is TaskKind.AST_FIM:
            missing = [
                name
                for name in (
                    "chunk_starts",
                    "chunk_ends",
                    "chunk_kinds",
                    "chunk_dep_levels",
                )
                if _array_length(getattr(packet, name)) == 0
            ]
            if missing:
                reason = "missing " + ", ".join(missing)
            elif not eligible_ast_chunk_indices(packet):
                reason = "no interior clang chunk with non-empty context"
            else:
                reason = None
            return _length_reason(reason, selected_token_count + 3, max_input_tokens)
        if task is TaskKind.IFIM:
            instruction_count = _array_length(packet.ifim_instruction_token_ids)
            reason = (
                None
                if instruction_count > 0
                else "missing or empty ifim_instruction_token_ids"
            )
            return _length_reason(
                reason,
                selected_token_count + instruction_count + 4,
                max_input_tokens,
            )
        field_name = {
            TaskKind.SYMBOL_RECOVERY: "symbol_ids",
            TaskKind.TYPE_RECOVERY: "type_refs",
            TaskKind.CALLEE_RECOVERY: "call_targets",
        }[task]
        values = getattr(packet, field_name)
        if values is None:
            return f"missing {field_name}"
        if not np.any(np.asarray(values) != 0):
            return f"{field_name} has no non-zero span"
        return _length_reason(None, token_count + 4, max_input_tokens)

    if not isinstance(packet, CommitPacket):
        return "requires CommitPacket"
    if task is TaskKind.COMMIT_DIFF:
        required = ("commit_msg", "diff_token_ids")
    elif task is TaskKind.PRE_TO_POST:
        required = ("commit_msg", "pre_token_ids", "post_token_ids")
    else:  # pragma: no cover - TaskKind exhaustiveness guard
        return f"unsupported task {task.value}"
    missing = [name for name in required if _array_length(getattr(packet, name)) == 0]
    reason = None if not missing else "missing or empty " + ", ".join(missing)
    if task is TaskKind.COMMIT_DIFF:
        estimated_input = _array_length(packet.commit_msg) + _array_length(
            packet.diff_token_ids
        ) + 4
    else:
        estimated_input = (
            _array_length(packet.commit_msg)
            + _array_length(packet.pre_token_ids)
            + _array_length(packet.post_token_ids)
            + 7
        )
    return _length_reason(reason, estimated_input, max_input_tokens)


def _length_reason(
    existing_reason: str | None,
    input_tokens: int,
    max_input_tokens: int | None,
) -> str | None:
    if existing_reason is not None:
        return existing_reason
    if max_input_tokens is not None and input_tokens > max_input_tokens:
        return (
            f"objective input length {input_tokens} exceeds configured maximum "
            f"{max_input_tokens}"
        )
    return None


class EligibilityAwareTaskMixer:
    """Realize exact deterministic task quotas over a finite source window."""

    def __init__(
        self,
        rates: Mapping[TaskKind | str, float] | None = None,
        *,
        seed: int,
        stage: str = "stage1",
        special_token_ids: FIMSpecialTokenInput = None,
        spm_rate: float = 0.5,
        max_input_tokens: int | None = None,
    ) -> None:
        self._rates = normalize_rates(rates, stage=stage)
        self._tasks = tuple(task for task, rate in self._rates.items() if rate > 0.0)
        self._seed = seed
        if max_input_tokens is not None and max_input_tokens < 1:
            raise ValueError("max_input_tokens must be >=1 when configured")
        self._max_input_tokens = max_input_tokens
        self._builder = TaskMixer(
            self._rates,
            seed=seed,
            special_token_ids=special_token_ids,
            spm_rate=spm_rate,
        )

    @property
    def rates(self) -> dict[TaskKind, float]:
        return dict(self._rates)

    def _step_rng(self, step_index: int) -> random.Random:
        derived = (self._seed * 0x9E3779B1 + int(step_index)) & 0xFFFFFFFFFFFFFFFF
        return random.Random(derived)

    def _eligibility(
        self, source: SourceInput
    ) -> tuple[tuple[TaskKind, ...], dict[TaskKind, str]]:
        eligible: list[TaskKind] = []
        ineligible: dict[TaskKind, str] = {}
        for task in self._tasks:
            packet = self._packet_for_task(source, task)
            if packet is None:
                family = (
                    "commit_packet"
                    if task
                    in {
                        TaskKind.COMMIT_DIFF,
                        TaskKind.PRE_TO_POST,
                    }
                    else "code_packet"
                )
                ineligible[task] = f"missing ObjectiveSource.{family}"
                continue
            reason = _eligibility_reason(
                task, packet, max_input_tokens=self._max_input_tokens
            )
            if reason is None:
                eligible.append(task)
            else:
                ineligible[task] = reason
        return tuple(eligible), ineligible

    @staticmethod
    def _packet_for_task(
        source: SourceInput, task: TaskKind
    ) -> CodePacket | CommitPacket | None:
        if not isinstance(source, ObjectiveSource):
            return source
        if task in {TaskKind.COMMIT_DIFF, TaskKind.PRE_TO_POST}:
            return source.commit_packet
        return source.code_packet

    def materialize(
        self,
        packet: SourceInput,
        *,
        step_index: int,
    ) -> RealizedObjective:
        eligible, ineligible = self._eligibility(packet)
        if not eligible:
            details = "; ".join(
                f"{task.value}: {reason}" for task, reason in ineligible.items()
            )
            raise ValueError(f"no eligible objective at step {step_index}: {details}")
        rng = self._step_rng(step_index)
        task = rng.choices(
            eligible,
            weights=[self._rates[candidate] for candidate in eligible],
            k=1,
        )[0]
        task_packet = self._packet_for_task(packet, task)
        assert task_packet is not None
        example = self._builder.build(task, task_packet, rng=rng)
        self._require_realized_kind(task, example)
        return RealizedObjective(
            task=task,
            example=example,
            ineligible=ineligible,
            source_index=0,
        )

    @staticmethod
    def _require_realized_kind(
        task: TaskKind, example: ObjectiveExample
    ) -> None:
        if example.objective != task.value:
            raise RuntimeError(
                f"selected {task.value} but builder realized {example.objective}; "
                "objective quotas and telemetry must record the realized kind"
            )

    def quotas(self, window_size: int) -> dict[TaskKind, int]:
        if window_size < 1:
            raise ValueError(f"objective quota window must be >=1, got {window_size}")
        decimal_rates = {task: Fraction(str(self._rates[task])) for task in self._tasks}
        exact_total = sum(decimal_rates.values(), Fraction())
        exact_rates = {task: rate / exact_total for task, rate in decimal_rates.items()}
        raw = {task: exact_rates[task] * window_size for task in self._tasks}
        quotas = {
            task: value.numerator // value.denominator for task, value in raw.items()
        }
        remaining = window_size - sum(quotas.values())
        task_order = {task: index for index, task in enumerate(self._tasks)}
        ranked = sorted(
            self._tasks,
            key=lambda task: (-(raw[task] - quotas[task]), task_order[task]),
        )
        for task in ranked[:remaining]:
            quotas[task] += 1
        return quotas

    def materialize_window(
        self,
        packets: Sequence[SourceInput],
        *,
        start_step: int = 0,
    ) -> list[RealizedObjective]:
        if not packets:
            return []
        quotas = self.quotas(len(packets))
        eligibility = [self._eligibility(packet) for packet in packets]
        eligible_counts = Counter(
            task for eligible, _ in eligibility for task in eligible
        )

        impossible = [
            task for task, quota in quotas.items() if eligible_counts[task] < quota
        ]
        if impossible:
            details = "; ".join(
                f"{task.value} quota={quotas[task]} eligible={eligible_counts[task]}"
                for task in impossible
            )
            raise ValueError(f"objective quota is not satisfiable: {details}")

        rng = self._step_rng(start_step)
        packet_ties = [rng.random() for _ in packets]
        packet_order = sorted(
            range(len(packets)),
            key=lambda index: (len(eligibility[index][0]), packet_ties[index], index),
        )
        slots = [task for task in self._tasks for _ in range(quotas[task])]
        rng.shuffle(slots)
        slot_owner: list[int | None] = [None] * len(slots)

        def assign(packet_index: int, seen: set[int]) -> bool:
            eligible = set(eligibility[packet_index][0])
            for slot_index, task in enumerate(slots):
                if slot_index in seen or task not in eligible:
                    continue
                seen.add(slot_index)
                owner = slot_owner[slot_index]
                if owner is None or assign(owner, seen):
                    slot_owner[slot_index] = packet_index
                    return True
            return False

        for packet_index in packet_order:
            if not assign(packet_index, set()):
                quota_text = ", ".join(
                    f"{task.value}={quota}" for task, quota in quotas.items()
                )
                raise ValueError(
                    "objective quota matching failed despite aggregate eligibility; "
                    f"quotas: {quota_text}; source_index={packet_index}"
                )

        task_by_packet = {
            packet_index: slots[slot_index]
            for slot_index, packet_index in enumerate(slot_owner)
            if packet_index is not None
        }
        realized: list[RealizedObjective] = []
        for source_index, packet in enumerate(packets):
            task = task_by_packet[source_index]
            step_index = start_step + source_index
            build_rng = self._step_rng(step_index)
            task_packet = self._packet_for_task(packet, task)
            assert task_packet is not None
            example = self._builder.build(task, task_packet, rng=build_rng)
            self._require_realized_kind(task, example)
            realized.append(
                RealizedObjective(
                    task=task,
                    example=example,
                    ineligible=eligibility[source_index][1],
                    source_index=source_index,
                )
            )
        assert Counter(item.task for item in realized) == Counter(quotas)
        return realized


@dataclass
class _ObjectiveTotals:
    samples: int = 0
    input_tokens: int = 0
    loss_tokens: int = 0
    loss_sum: float = 0.0


class ObjectiveAccounting:
    """Accumulate exact sample/token counts and token-weighted objective loss."""

    def __init__(
        self,
        configured_rates: Mapping[TaskKind | str, float] | None = None,
    ) -> None:
        self._totals: dict[TaskKind, _ObjectiveTotals] = {}
        self._rates = (
            None if configured_rates is None else normalize_rates(configured_rates)
        )

    def record(
        self,
        task: TaskKind,
        example: ObjectiveExample,
        *,
        loss: mx.array | float,
    ) -> None:
        input_tokens = int(example.input_ids.shape[0])
        loss_tokens = int(np.asarray(example.loss_mask).astype(np.int64).sum())
        if loss_tokens < 1:
            raise ValueError(f"{task.value}: objective example has no loss tokens")
        scalar_loss = float(loss.item()) if isinstance(loss, mx.array) else float(loss)
        if not math.isfinite(scalar_loss):
            raise ValueError(
                f"{task.value}: objective loss must be finite, got {scalar_loss}"
            )
        totals = self._totals.setdefault(task, _ObjectiveTotals())
        totals.samples += 1
        totals.input_tokens += input_tokens
        totals.loss_tokens += loss_tokens
        totals.loss_sum += scalar_loss * loss_tokens

    def report(self) -> dict[str, dict[str, float | int]]:
        all_samples = sum(item.samples for item in self._totals.values())
        all_input_tokens = sum(item.input_tokens for item in self._totals.values())
        all_loss_tokens = sum(item.loss_tokens for item in self._totals.values())
        report: dict[str, dict[str, float | int]] = {}
        for task in sorted(self._totals, key=lambda item: item.value):
            totals = self._totals[task]
            row: dict[str, float | int] = {
                "samples": totals.samples,
                "input_tokens": totals.input_tokens,
                "loss_tokens": totals.loss_tokens,
                "loss_sum": totals.loss_sum,
                "mean_loss": totals.loss_sum / totals.loss_tokens,
                "sample_rate": totals.samples / all_samples,
                "input_token_rate": totals.input_tokens / all_input_tokens,
                "loss_token_rate": totals.loss_tokens / all_loss_tokens,
            }
            if self._rates is not None:
                row["configured_rate"] = self._rates.get(task, 0.0)
                row["sample_rate_drift"] = row["sample_rate"] - row["configured_rate"]
            report[task.value] = row
        return report


@dataclass(frozen=True)
class GraphAuxLossConfig:
    relations: tuple[str, ...] = ("call", "type")
    topk: int = 8
    global_weight: float = 1.0
    indexer_weight: float = 0.001
    layer_weight: float = 1.0
    bce_weight: float = 1.0
    coverage_weight: float = 1.0
    pos_weight: float = 1.0
    margin: float = 1.0

    def __post_init__(self) -> None:
        if not self.relations or any(
            not isinstance(relation, str) or not relation for relation in self.relations
        ):
            raise ValueError("graph auxiliary relations must contain non-empty strings")
        if len(set(self.relations)) != len(self.relations):
            raise ValueError("graph auxiliary relations must not contain duplicates")
        if self.topk < 1:
            raise ValueError(f"graph auxiliary topk must be >=1, got {self.topk}")
        for name in (
            "global_weight",
            "indexer_weight",
            "layer_weight",
            "bce_weight",
            "coverage_weight",
            "pos_weight",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"graph auxiliary {name} must be positive")
        if not math.isfinite(float(self.margin)) or self.margin < 0.0:
            raise ValueError("graph auxiliary margin must be non-negative")


def _graph_targets(
    graph: GraphPacket,
    *,
    relations: Sequence[str],
    batch: int,
    queries: int,
    keys: int,
) -> mx.array:
    if queries != keys:
        raise ValueError(
            "graph auxiliary loss requires square node-aligned indexer scores, "
            f"got queries={queries}, keys={keys}"
        )
    if graph.num_nodes is None:
        raise ValueError("graph auxiliary loss requires GraphPacket.num_nodes")
    if int(graph.num_nodes) != queries:
        raise ValueError(
            f"graph num_nodes={graph.num_nodes} does not match indexer size={queries}"
        )
    targets = np.zeros((batch, queries, keys), dtype=np.float32)
    edge_count = 0
    for relation in relations:
        edge = graph.edge(relation)
        if edge is None:
            raise ValueError(
                f"graph auxiliary relation {relation!r} is configured but absent"
            )
        for source, destination in edge.to_pairs():
            if not (0 <= source < queries and 0 <= destination < keys):
                raise ValueError(
                    f"graph edge {relation} ({source}, {destination}) is outside "
                    f"indexer shape ({queries}, {keys})"
                )
            targets[:, source, destination] = 1.0
            edge_count += 1
    if edge_count == 0:
        raise ValueError(
            "graph auxiliary loss has no real edges for configured relations "
            f"{tuple(relations)}"
        )
    return mx.array(targets)


def compute_graph_auxiliary_loss(
    indexer_scores: Sequence[mx.array],
    graph: GraphPacket,
    config: GraphAuxLossConfig,
) -> tuple[mx.array, dict[str, float | int]]:
    if not indexer_scores:
        raise ValueError(
            "graph auxiliary loss configured but indexer scores are absent"
        )
    layer_losses: list[mx.array] = []
    bce_total = 0.0
    coverage_total = 0.0
    for layer_index, scores in enumerate(indexer_scores):
        if scores.ndim != 3:
            raise ValueError(
                f"graph indexer layer {layer_index} must be (B,Q,K), got "
                f"{tuple(scores.shape)}"
            )
        batch, queries, keys = (int(value) for value in scores.shape)
        targets = _graph_targets(
            graph,
            relations=config.relations,
            batch=batch,
            queries=queries,
            keys=keys,
        )
        layer_loss, components = total_indexer_loss(
            scores,
            edge_targets=targets,
            topk=config.topk,
            kl_coeff=0.0,
            bce_coeff=config.bce_weight,
            coverage_coeff=config.coverage_weight,
            pos_weight=config.pos_weight,
            margin=config.margin,
        )
        layer_losses.append(layer_loss)
        bce_total += float(components.get("bce", mx.array(0.0)).item())
        coverage_total += float(components.get("coverage", mx.array(0.0)).item())
    total = (
        mx.sum(mx.stack(layer_losses))
        * config.global_weight
        * config.indexer_weight
        * config.layer_weight
    )
    layer_count = len(layer_losses)
    return total, {
        "graph_indexer_layers": layer_count,
        "graph_indexer_bce": bce_total / layer_count,
        "graph_indexer_coverage": coverage_total / layer_count,
        "graph_indexer_global_weight": config.global_weight,
        "graph_indexer_indexer_weight": config.indexer_weight,
        "graph_indexer_layer_weight": config.layer_weight,
        "graph_indexer_layer_reduction": "sum",
    }


def graph_auxiliary_loss_from_targets(
    indexer_scores: Sequence[mx.array],
    edge_targets: mx.array,
    pair_mask: mx.array,
    config: GraphAuxLossConfig,
) -> mx.array:
    """Differentiable graph loss for fixed-shape production batches."""

    if not indexer_scores:
        raise ValueError(
            "graph auxiliary loss configured but indexer scores are absent"
        )
    if edge_targets.ndim != 3 or pair_mask.shape != edge_targets.shape:
        raise ValueError(
            "graph auxiliary targets/pair_mask must share (B,Q,K) shape; got "
            f"targets={tuple(edge_targets.shape)} mask={tuple(pair_mask.shape)}"
        )
    losses: list[mx.array] = []
    for layer_index, scores in enumerate(indexer_scores):
        if scores.shape != edge_targets.shape:
            raise ValueError(
                f"graph indexer layer {layer_index} shape {tuple(scores.shape)} "
                f"!= targets {tuple(edge_targets.shape)}"
            )
        loss, _ = total_indexer_loss(
            scores,
            edge_targets=edge_targets,
            edge_pair_mask=pair_mask,
            topk=config.topk,
            kl_coeff=0.0,
            bce_coeff=config.bce_weight,
            coverage_coeff=config.coverage_weight,
            pos_weight=config.pos_weight,
            margin=config.margin,
        )
        losses.append(loss)
    return (
        mx.sum(mx.stack(losses))
        * config.global_weight
        * config.indexer_weight
        * config.layer_weight
    )


def combine_lm_and_aux_losses(
    lm_loss: mx.array,
    auxiliary_losses: Mapping[str, mx.array],
    weights: Mapping[str, float],
) -> tuple[mx.array, dict[str, float]]:
    total = lm_loss
    metrics: dict[str, float] = {"loss_lm": float(lm_loss.item())}
    for name, weight in weights.items():
        if weight < 0.0:
            raise ValueError(f"auxiliary loss weight {name!r} must be non-negative")
        if weight == 0.0:
            continue
        if name not in auxiliary_losses:
            raise ValueError(
                f"auxiliary loss {name!r} has configured weight {weight} but no value"
            )
        loss = auxiliary_losses[name]
        total = total + float(weight) * loss
        metrics[f"loss_{name}"] = float(loss.item())
        metrics[f"loss_{name}_weight"] = float(weight)
    metrics["loss_total"] = float(total.item())
    return total, metrics


def production_training_loss(
    model: Any,
    input_ids: mx.array,
    targets: mx.array,
    loss_mask: mx.array,
    *,
    side_channels: Mapping[str, mx.array],
    block_bias: mx.array | None,
    graph_targets: mx.array | None,
    graph_pair_mask: mx.array | None,
    graph_config: GraphAuxLossConfig | None,
    graph_weight: float,
) -> tuple[mx.array, mx.array, mx.array]:
    """Compute the differentiated LM + configured graph auxiliary objective."""

    if not math.isfinite(float(graph_weight)) or graph_weight <= 0.0:
        raise ValueError("graph auxiliary global weight must be finite and positive")
    if graph_config is None or graph_targets is None or graph_pair_mask is None:
        raise ValueError(
            "production graph objective requires config/targets/pair_mask"
        )
    if graph_weight != graph_config.global_weight:
        raise ValueError(
            "graph auxiliary global weight differs from GraphAuxLossConfig: "
            f"{graph_weight} != {graph_config.global_weight}"
        )
    if block_bias is None:
        raise ValueError("production graph objective requires graph route block_bias")
    model_config = getattr(model, "config", None)
    if (
        model_config is None
        or getattr(model_config, "attention_mode", None) != "dsa"
        or getattr(model_config, "require_graph_routes", None) is not True
    ):
        raise ValueError(
            "production graph objective requires active fail-closed DSA graph routes"
        )
    if float(getattr(model_config, "structure_residual_scale", 0.0)) <= 0.0:
        raise ValueError(
            "production graph objective requires active structure residual routing"
        )
    required_structure_channels = {
        "structure_ids",
        "dep_levels",
        "ast_depth_ids",
        "sibling_index_ids",
        "node_type_ids",
    }
    missing_structure_channels = sorted(
        name
        for name in required_structure_channels
        if side_channels.get(name) is None
    )
    if missing_structure_channels:
        raise ValueError(
            "production graph objective is missing required structure sidecars: "
            + ", ".join(missing_structure_channels)
        )
    _, lm_loss = model(
        input_ids,
        targets=targets,
        loss_mask=loss_mask,
        block_bias=block_bias,
        **side_channels,
    )
    if lm_loss is None:  # pragma: no cover - targets make this unreachable
        raise RuntimeError("model returned no LM loss despite supplied targets")
    graph_loss = graph_auxiliary_loss_from_targets(
        model.indexer_scores(
            input_ids,
            block_bias=block_bias,
            **side_channels,
        ),
        graph_targets,
        graph_pair_mask,
        graph_config,
    )
    return lm_loss + graph_loss, lm_loss, graph_loss


__all__ = [
    "EligibilityAwareTaskMixer",
    "GraphAuxLossConfig",
    "ObjectiveAccounting",
    "ObjectiveSource",
    "RealizedObjective",
    "combine_lm_and_aux_losses",
    "compute_graph_auxiliary_loss",
    "graph_auxiliary_loss_from_targets",
    "production_training_loss",
]

"""Eligibility-aware production objective scheduling and exact accounting."""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import mlx.core as mx
import numpy as np

from cppmega_mlx.data.ast_fim import (
    domain_preserving_document_spans,
    has_safe_physical_middle,
    eligible_ast_chunk_indices,
    logical_document_spans,
    span_has_single_physical_source,
)
from cppmega_mlx.data.code_packet import CodePacket
from cppmega_mlx.data.commit_packet import CommitPacket
from cppmega_mlx.data.fim import (
    FIMSpecialTokenInput,
    UnsupportedDomainDelimiterStructureError,
)
from cppmega_mlx.data.graph_packet import GraphPacket
from cppmega_mlx.data.nanochat_pipeline.platform_vocab import MAX_PLATFORM_IDS
from cppmega_mlx.data.source_identity import MAX_ROW_LOCAL_DOC_ID, MAX_SOURCE_ID
from cppmega_mlx.training.indexer_losses import total_indexer_loss
from cppmega_mlx.training.objectives import (
    ObjectiveExample,
    ifim_instruction_ids_by_document,
)
from cppmega_mlx.training.task_mixer import TaskKind, TaskMixer, normalize_rates


PACKED_COMMIT_BINDING_METADATA_KEY = "packed_constituent_binding"
PACKED_COMMIT_BINDING_SCHEMA = "cppmega_packed_commit_constituent_v1"
_PACKED_COMMIT_BINDING_FIELDS = {
    "schema",
    "constituent_index",
    "token_start",
    "token_end",
    "attention_document_id",
    "source_document_id",
    "source_identity_id",
    "platform_ids",
}


@dataclass(frozen=True)
class RealizedObjective:
    task: TaskKind
    example: ObjectiveExample
    ineligible: Mapping[TaskKind, str]
    source_index: int
    selected_packet: CodePacket | CommitPacket | None = None
    selected_packet_index: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.selected_packet_index, bool)
            or not isinstance(self.selected_packet_index, int)
            or self.selected_packet_index < 0
        ):
            raise ValueError("selected_packet_index must be a non-negative integer")


@dataclass(frozen=True)
class ObjectiveSource:
    """One upstream document with independently optional code/commit views."""

    code_packet: CodePacket | None = None
    commit_packet: CommitPacket | None = None
    commit_candidates: tuple[CommitPacket, ...] = ()

    def __post_init__(self) -> None:
        if (
            self.code_packet is None
            and self.commit_packet is None
            and not self.commit_candidates
        ):
            raise ValueError("ObjectiveSource requires a code or commit packet")
        if any(
            not isinstance(packet, CommitPacket) for packet in self.commit_candidates
        ):
            raise TypeError(
                "ObjectiveSource.commit_candidates must contain CommitPacket values"
            )
        if self.commit_candidates:
            if self.commit_packet is None:
                raise ValueError(
                    "ObjectiveSource.commit_candidates require commit_packet to name "
                    "the canonical first candidate"
                )
            if self.commit_packet is not self.commit_candidates[0]:
                raise ValueError(
                    "ObjectiveSource.commit_packet must be the first commit candidate"
                )


SourceInput = CodePacket | CommitPacket | ObjectiveSource


def _array_length(value: mx.array | None) -> int:
    return 0 if value is None else int(value.shape[0])


def _packet_vector_values(packet: CodePacket, field: str) -> list[int] | None:
    raw = getattr(packet, field)
    if raw is None:
        return None
    values = np.asarray(raw)
    token_count = int(packet.token_ids.shape[-1])
    if values.ndim != 1 or len(values) != token_count:
        raise ValueError(
            f"CodePacket.{field} must have shape ({token_count},), got {values.shape}"
        )
    return [int(value) for value in values.tolist()]


def validated_packed_commit_binding(
    source: ObjectiveSource,
    packet: CommitPacket,
) -> dict[str, int | str | list[int]] | None:
    """Validate and return one packed commit candidate's exact constituent binding."""

    if source.commit_candidates and not any(
        packet is candidate for candidate in source.commit_candidates
    ):
        raise ValueError("selected commit packet is not one of the source candidates")
    raw_binding = packet.metadata.get(PACKED_COMMIT_BINDING_METADATA_KEY)
    code = source.code_packet
    if raw_binding is None:
        if code is None:
            return None
        source_docs = _packet_vector_values(code, "source_doc_ids")
        source_identities = _packet_vector_values(code, "source_identity_ids")
        if source_docs is None or source_identities is None:
            return None
        constituents = set(zip(source_docs, source_identities, strict=True))
        if len(constituents) > 1:
            raise ValueError(
                "multi-constituent commit candidate requires an exact packed "
                "constituent binding"
            )
        return None
    if not isinstance(raw_binding, Mapping):
        raise ValueError("packed commit constituent binding must be a mapping")
    if set(raw_binding) != _PACKED_COMMIT_BINDING_FIELDS:
        raise ValueError(
            "packed commit constituent binding fields are invalid: "
            f"{sorted(raw_binding)}"
        )
    if raw_binding.get("schema") != PACKED_COMMIT_BINDING_SCHEMA:
        raise ValueError("packed commit constituent binding schema is invalid")
    if code is None:
        raise ValueError("packed commit constituent binding requires a CodePacket")

    integer_fields = (
        "constituent_index",
        "token_start",
        "token_end",
        "attention_document_id",
        "source_document_id",
        "source_identity_id",
    )
    integers: dict[str, int] = {}
    for field in integer_fields:
        value = raw_binding.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"packed commit binding {field} must be an integer")
        integers[field] = value
    if integers["constituent_index"] < 0:
        raise ValueError("packed commit constituent_index must be non-negative")
    packet_source_index = packet.metadata.get("source_index")
    if (
        isinstance(packet_source_index, bool)
        or not isinstance(packet_source_index, int)
        or packet_source_index != integers["constituent_index"]
    ):
        raise ValueError(
            "packed commit binding constituent_index does not match its typed sections"
        )
    token_count = int(code.token_ids.shape[-1])
    start = integers["token_start"]
    end = integers["token_end"]
    if not 0 <= start < end <= token_count:
        raise ValueError(
            f"packed commit token span [{start}, {end}) is outside 0..{token_count}"
        )
    attention_document_id = integers["attention_document_id"]
    source_document_id = integers["source_document_id"]
    source_identity_id = integers["source_identity_id"]
    if attention_document_id <= 0:
        raise ValueError("packed commit attention_document_id must be positive")
    if not 0 < source_document_id <= MAX_ROW_LOCAL_DOC_ID:
        raise ValueError("packed commit source_document_id must be positive uint32")
    if not 0 < source_identity_id <= MAX_SOURCE_ID:
        raise ValueError("packed commit source_identity_id must be positive uint64")

    document_ids = _packet_vector_values(code, "document_ids")
    source_doc_ids = _packet_vector_values(code, "source_doc_ids")
    source_identity_ids = _packet_vector_values(code, "source_identity_ids")
    if document_ids is None or source_doc_ids is None or source_identity_ids is None:
        raise ValueError(
            "packed commit binding requires token-aligned document_ids, "
            "source_doc_ids, and source_identity_ids"
        )
    expected_positions = [
        index
        for index, value in enumerate(document_ids)
        if value == attention_document_id
    ]
    if expected_positions != list(range(start, end)):
        raise ValueError(
            "packed commit token span does not exactly match its attention document"
        )
    if any(value != source_document_id for value in source_doc_ids[start:end]):
        raise ValueError("packed commit token span crosses source document identities")
    if any(value != source_identity_id for value in source_identity_ids[start:end]):
        raise ValueError("packed commit token span crosses physical source identities")

    raw_platform_ids = raw_binding.get("platform_ids")
    if not isinstance(raw_platform_ids, list):
        raise ValueError("packed commit platform_ids must be a list")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in raw_platform_ids
    ):
        raise ValueError("packed commit platform_ids must contain integers")
    platform_ids = list(raw_platform_ids)
    if any(not 0 < value <= np.iinfo(np.uint16).max for value in platform_ids):
        raise ValueError("packed commit platform_ids must be positive uint16")
    if len(platform_ids) > MAX_PLATFORM_IDS:
        raise ValueError(
            f"packed commit platform_ids exceed MAX_PLATFORM_IDS={MAX_PLATFORM_IDS}"
        )
    if not platform_ids:
        raise ValueError("packed commit platform_ids must be non-empty")
    raw_bags = code.metadata.get("source_platform_ids")
    if not isinstance(raw_bags, (list, tuple)):
        raise ValueError("packed commit binding requires source_platform_ids")
    document_order: list[int] = []
    for document_id in document_ids:
        if not document_order or document_order[-1] != document_id:
            if document_id in document_order:
                raise ValueError("CodePacket.document_ids are not contiguous")
            document_order.append(document_id)
    try:
        bag_index = document_order.index(attention_document_id)
    except ValueError as exc:  # pragma: no cover - span validation proves membership
        raise ValueError(
            "packed commit attention document has no platform bag"
        ) from exc
    if bag_index != integers["constituent_index"]:
        raise ValueError(
            "packed commit constituent_index does not match its attention document"
        )
    if len(raw_bags) != len(document_order):
        raise ValueError(
            "source_platform_ids count does not match packed attention documents"
        )
    raw_constituent_bag = raw_bags[bag_index]
    if not isinstance(raw_constituent_bag, (list, tuple)) or any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in raw_constituent_bag
    ):
        raise ValueError("packed commit source platform bag must contain integers")
    if list(raw_constituent_bag) != platform_ids:
        raise ValueError("packed commit platform bag does not match its constituent")

    return {
        "schema": PACKED_COMMIT_BINDING_SCHEMA,
        **integers,
        "platform_ids": platform_ids,
    }


def _transformed_document_physical_identity_reason(
    packet: CodePacket,
    document_spans,
) -> str | None:
    """Validate provenance and require one source-local middle candidate.

    A packed code document may intentionally contain cross-file context.  The
    unsafe condition is a FIM middle that crosses physical sources, not a
    context packet that contains more than one source identity.
    """

    if packet.source_identity_ids is None:
        return None
    source_identity_ids = np.asarray(packet.source_identity_ids)
    token_count = int(packet.token_ids.shape[-1])
    if source_identity_ids.ndim != 1 or len(source_identity_ids) != token_count:
        return (
            "requires one token-aligned physical source identity per code token; "
            f"got shape {source_identity_ids.shape} for {token_count} tokens"
        )
    for region in document_spans:
        if region.content_end - region.content_start < 3:
            continue
        try:
            if has_safe_physical_middle(packet, region):
                return None
        except ValueError as exc:
            return str(exc)
    return (
        "no logical document has a physical source run with non-empty "
        "prefix/middle/suffix context"
    )


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
        document_spans = logical_document_spans(packet)
        document_lengths = [end - start for start, end, _document_id in document_spans]
        if task is TaskKind.CAUSAL_LM:
            reason = (
                None
                if any(length >= 2 for length in document_lengths)
                else "requires a logical document with at least 2 tokens"
            )
            estimated_input = token_count - 1
            return _length_reason(reason, estimated_input, max_input_tokens)
        try:
            transform_regions = domain_preserving_document_spans(packet)
        except UnsupportedDomainDelimiterStructureError as exc:
            return str(exc)
        if task in {TaskKind.FIM, TaskKind.AST_FIM, TaskKind.IFIM}:
            eligible_regions = [
                region
                for region in transform_regions
                if region.content_end - region.content_start >= 3
            ]
            if not eligible_regions:
                return "requires a supported domain interior with at least 3 tokens"
            selected_token_count = max(
                region.document_end - region.document_start
                for region in eligible_regions
            )
            physical_reason = _transformed_document_physical_identity_reason(
                packet,
                eligible_regions,
            )
            if physical_reason is not None:
                return physical_reason
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
            try:
                instruction_bindings = ifim_instruction_ids_by_document(packet)
            except ValueError as exc:
                return str(exc)
            eligible_instruction_regions = [
                region
                for region in eligible_regions
                if region.document_id in instruction_bindings
                and has_safe_physical_middle(packet, region)
            ]
            instruction_count = max(
                (
                    len(instruction_bindings[region.document_id])
                    for region in eligible_instruction_regions
                ),
                default=0,
            )
            reason = (
                None
                if eligible_instruction_regions and instruction_count > 0
                else (
                    "missing or empty source_ifim_instruction_token_ids "
                    "per-document IFIM instruction binding"
                )
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
        markers = [int(value) for value in np.asarray(values).reshape(-1).tolist()]
        candidate_regions: list[tuple[int, int]] = []
        index = 0
        while index < len(markers):
            if markers[index] == 0:
                index += 1
                continue
            end = index + 1
            while end < len(markers) and markers[end] == markers[index]:
                end += 1
            for region in transform_regions:
                if (
                    region.content_start <= index < end <= region.content_end
                    and span_has_single_physical_source(
                        packet,
                        start=index,
                        end=end,
                    )
                ):
                    candidate_regions.append(
                        (region.document_start, region.document_end)
                    )
                    break
            index = end
        if not candidate_regions:
            return (
                f"{field_name} has no non-zero span inside a supported domain interior"
            )
        selected_token_count = max(end - start for start, end in candidate_regions)
        return _length_reason(
            None,
            selected_token_count + 4,
            max_input_tokens,
        )

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
        estimated_input = (
            _array_length(packet.commit_msg) + _array_length(packet.diff_token_ids) + 6
        )
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


class ObjectiveQuotaUnsatisfiedError(ValueError):
    """The current bounded source pool cannot realize an objective quota window."""


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
            packets = self._packets_for_task(source, task)
            if not packets:
                family = (
                    "commit_packet/commit_candidates"
                    if task
                    in {
                        TaskKind.COMMIT_DIFF,
                        TaskKind.PRE_TO_POST,
                    }
                    else "code_packet"
                )
                ineligible[task] = f"missing ObjectiveSource.{family}"
                continue
            reasons = [
                self._packet_eligibility_reason(source, task, packet)
                for packet in packets
            ]
            if any(reason is None for reason in reasons):
                eligible.append(task)
            else:
                unique_reasons = tuple(dict.fromkeys(str(reason) for reason in reasons))
                ineligible[task] = "; ".join(unique_reasons)
        return tuple(eligible), ineligible

    @staticmethod
    def _packets_for_task(
        source: SourceInput, task: TaskKind
    ) -> tuple[CodePacket | CommitPacket, ...]:
        if not isinstance(source, ObjectiveSource):
            return (source,)
        if task in {TaskKind.COMMIT_DIFF, TaskKind.PRE_TO_POST}:
            if source.commit_candidates:
                return source.commit_candidates
            return () if source.commit_packet is None else (source.commit_packet,)
        return () if source.code_packet is None else (source.code_packet,)

    def _packet_eligibility_reason(
        self,
        source: SourceInput,
        task: TaskKind,
        packet: CodePacket | CommitPacket,
    ) -> str | None:
        if isinstance(source, ObjectiveSource) and isinstance(packet, CommitPacket):
            try:
                validated_packed_commit_binding(source, packet)
            except ValueError as exc:
                return str(exc)
        return _eligibility_reason(
            task,
            packet,
            max_input_tokens=self._max_input_tokens,
        )

    def _select_packet_for_task(
        self,
        source: SourceInput,
        task: TaskKind,
        *,
        rng: random.Random,
    ) -> CodePacket | CommitPacket:
        eligible_packets = [
            packet
            for packet in self._packets_for_task(source, task)
            if self._packet_eligibility_reason(source, task, packet) is None
        ]
        if not eligible_packets:  # pragma: no cover - guarded by assignment matching
            raise RuntimeError(f"selected task {task.value} has no eligible packet")
        if len(eligible_packets) == 1:
            return eligible_packets[0]
        return eligible_packets[rng.randrange(len(eligible_packets))]

    @classmethod
    def _selected_packet_index(
        cls,
        source: SourceInput,
        task: TaskKind,
        selected_packet: CodePacket | CommitPacket,
    ) -> int:
        for index, packet in enumerate(cls._packets_for_task(source, task)):
            if packet is selected_packet:
                return index
        raise RuntimeError(
            f"selected {task.value} packet is not bound to its ObjectiveSource"
        )

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
        task_packet = self._select_packet_for_task(packet, task, rng=rng)
        example = self._builder.build(task, task_packet, rng=rng)
        self._require_realized_kind(task, example)
        return RealizedObjective(
            task=task,
            example=example,
            ineligible=ineligible,
            source_index=0,
            selected_packet=task_packet,
            selected_packet_index=self._selected_packet_index(packet, task, task_packet),
        )

    @staticmethod
    def _require_realized_kind(task: TaskKind, example: ObjectiveExample) -> None:
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
        return self.materialize_window_from_pool(
            packets,
            output_count=len(packets),
            start_step=start_step,
        )

    def materialize_window_from_pool(
        self,
        packets: Sequence[SourceInput],
        *,
        output_count: int,
        start_step: int = 0,
        required_assignment: Callable[[SourceInput, TaskKind], bool] | None = None,
        required_realized_assignment: Callable[
            [SourceInput, RealizedObjective], bool
        ]
        | None = None,
        candidate_assignment: Callable[[SourceInput, TaskKind], bool] | None = None,
    ) -> list[RealizedObjective]:
        """Select and realize one exact quota window from a bounded source pool."""

        if required_assignment is not None and required_realized_assignment is not None:
            raise ValueError(
                "required_assignment and required_realized_assignment are mutually "
                "exclusive"
            )
        if candidate_assignment is not None and required_realized_assignment is None:
            raise ValueError(
                "candidate_assignment requires required_realized_assignment"
            )

        if output_count < 0:
            raise ValueError(f"output_count must be >=0, got {output_count}")
        if output_count > len(packets):
            raise ValueError(
                "objective source pool is smaller than the requested output window: "
                f"pool={len(packets)}, output_count={output_count}"
            )
        if output_count == 0:
            return []
        if not packets:
            return []
        quotas = self.quotas(output_count)
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
            raise ObjectiveQuotaUnsatisfiedError(
                f"objective quota is not satisfiable: {details}"
            )

        rng = self._step_rng(start_step)
        packet_ties = [rng.random() for _ in packets]
        slots = [task for task in self._tasks for _ in range(quotas[task])]
        slot_ties = [rng.random() for _ in slots]
        candidate_packets = [
            [
                packet_index
                for packet_index, (eligible, _ineligible) in enumerate(eligibility)
                if task in eligible
            ]
            for task in slots
        ]
        for candidates in candidate_packets:
            candidates.sort(
                key=lambda packet_index: (
                    len(eligibility[packet_index][0]),
                    packet_ties[packet_index],
                    packet_index,
                )
            )
        slot_order = sorted(
            range(len(slots)),
            key=lambda slot_index: (
                len(candidate_packets[slot_index]),
                slot_ties[slot_index],
                slot_index,
            ),
        )

        def match(
            forced_assignment: tuple[int, int] | None,
        ) -> tuple[list[int | None] | None, int | None]:
            packet_owner: list[int | None] = [None] * len(packets)
            locked_packet = locked_slot = None
            if forced_assignment is not None:
                locked_packet, locked_slot = forced_assignment
                packet_owner[locked_packet] = locked_slot

            def assign(slot_index: int, seen_packets: set[int]) -> bool:
                for packet_index in candidate_packets[slot_index]:
                    if packet_index == locked_packet or packet_index in seen_packets:
                        continue
                    seen_packets.add(packet_index)
                    owner = packet_owner[packet_index]
                    if owner is None or assign(owner, seen_packets):
                        packet_owner[packet_index] = slot_index
                        return True
                return False

            for slot_index in slot_order:
                if slot_index == locked_slot:
                    continue
                if not assign(slot_index, set()):
                    return None, slot_index
            return packet_owner, None

        slot_indices_by_task: dict[TaskKind, list[int]] = {}
        for slot_index, task in enumerate(slots):
            slot_indices_by_task.setdefault(task, []).append(slot_index)

        def forced_candidate_pairs(
            predicate: Callable[[SourceInput, TaskKind], bool],
        ) -> list[tuple[int, int]]:
            # Enumerate forced candidate edges, not complete assignments.  Each edge
            # gets at most one augmenting-path matching attempt below, keeping
            # the search polynomial in the bounded source/slot graph.
            pairs: list[tuple[int, int]] = []
            for packet_index, packet in enumerate(packets):
                for task in eligibility[packet_index][0]:
                    task_slots = slot_indices_by_task.get(task)
                    if task_slots is None or not predicate(packet, task):
                        continue
                    pairs.extend(
                        (packet_index, slot_index) for slot_index in task_slots
                    )
            pairs.sort(
                key=lambda pair: (
                    len(candidate_packets[pair[1]]),
                    len(eligibility[pair[0]][0]),
                    packet_ties[pair[0]],
                    pair[0],
                    pair[1],
                )
            )
            return pairs

        forced_candidates: list[tuple[int, int] | None]
        if required_realized_assignment is not None:
            forced_candidates = [None]
            forced_candidates.extend(
                forced_candidate_pairs(
                    lambda packet, task: (
                        candidate_assignment is None
                        or candidate_assignment(packet, task)
                    )
                )
            )
        elif required_assignment is None:
            forced_candidates = [None]
        else:
            forced_candidates = forced_candidate_pairs(required_assignment)
            if not forced_candidates:
                raise ObjectiveQuotaUnsatisfiedError(
                    "objective source pool has no eligible assignment satisfying "
                    "the required auxiliary constraint: "
                    f"pool={len(packets)}, output_count={output_count}"
                )

        def realize(packet_owner: Sequence[int | None]) -> list[RealizedObjective]:
            task_by_packet = {
                packet_index: slots[slot_index]
                for packet_index, slot_index in enumerate(packet_owner)
                if slot_index is not None
            }
            realized: list[RealizedObjective] = []
            for output_index, source_index in enumerate(sorted(task_by_packet)):
                packet = packets[source_index]
                task = task_by_packet[source_index]
                step_index = start_step + output_index
                build_rng = self._step_rng(step_index)
                task_packet = self._select_packet_for_task(
                    packet,
                    task,
                    rng=build_rng,
                )
                example = self._builder.build(task, task_packet, rng=build_rng)
                self._require_realized_kind(task, example)
                realized.append(
                    RealizedObjective(
                        task=task,
                        example=example,
                        ineligible=eligibility[source_index][1],
                        source_index=source_index,
                        selected_packet=task_packet,
                        selected_packet_index=self._selected_packet_index(
                            packet,
                            task,
                            task_packet,
                        ),
                    )
                )
            assert Counter(item.task for item in realized) == Counter(quotas)
            return realized

        failed_slot = None
        matched_assignment = False
        attempted_assignments: set[tuple[int | None, ...]] = set()
        for forced_assignment in forced_candidates:
            packet_owner, failed_slot = match(forced_assignment)
            if packet_owner is None:
                continue
            signature = tuple(packet_owner)
            if signature in attempted_assignments:
                continue
            attempted_assignments.add(signature)
            matched_assignment = True
            realized = realize(packet_owner)
            if required_realized_assignment is None or any(
                required_realized_assignment(packets[item.source_index], item)
                for item in realized
            ):
                return realized

        if required_realized_assignment is not None and matched_assignment:
            raise ObjectiveQuotaUnsatisfiedError(
                "objective quota matching produced no realized assignment "
                "satisfying the required auxiliary constraint: "
                f"pool={len(packets)}, output_count={output_count}"
            )
        quota_text = ", ".join(
            f"{task.value}={quota}" for task, quota in quotas.items()
        )
        failed_task = "unknown" if failed_slot is None else slots[failed_slot].value
        raise ObjectiveQuotaUnsatisfiedError(
            "objective quota matching failed despite aggregate eligibility; "
            f"quotas: {quota_text}; slot={failed_task}; "
            f"pool={len(packets)}; output_count={output_count}; "
            f"required_assignment={required_assignment is not None}; "
            f"required_realized_assignment={required_realized_assignment is not None}"
        )


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
    bias_beta: float = 1.0

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
        if not math.isfinite(float(self.bias_beta)) or self.bias_beta <= 0.0:
            raise ValueError("graph auxiliary bias_beta must be positive")


@dataclass(frozen=True)
class GraphAuxLossBreakdown:
    """Differentiable graph-loss terms after all configured weights."""

    total: mx.array
    edge_bce: mx.array
    ranking: mx.array
    layer_count: int


@dataclass(frozen=True)
class ProductionTrainingLossBreakdown:
    """Array-valued LM + graph objective metrics from one decoder forward."""

    total: mx.array
    lm_ce: mx.array
    graph_total: mx.array
    graph_edge_bce: mx.array
    graph_ranking: mx.array
    graph_positive_pairs: mx.array
    ntokens: mx.array
    graph_layer_count: int


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

    return graph_auxiliary_loss_breakdown_from_targets(
        indexer_scores,
        edge_targets,
        pair_mask,
        config,
    ).total


def graph_auxiliary_loss_breakdown_from_targets(
    indexer_scores: Sequence[mx.array],
    edge_targets: mx.array,
    pair_mask: mx.array,
    config: GraphAuxLossConfig,
) -> GraphAuxLossBreakdown:
    """Return weighted BCE/ranking terms without detaching their MLX graph."""

    if not indexer_scores:
        raise ValueError(
            "graph auxiliary loss configured but indexer scores are absent"
        )
    if edge_targets.ndim != 3 or pair_mask.shape != edge_targets.shape:
        raise ValueError(
            "graph auxiliary targets/pair_mask must share (B,Q,K) shape; got "
            f"targets={tuple(edge_targets.shape)} mask={tuple(pair_mask.shape)}"
        )
    edge_bce_losses: list[mx.array] = []
    ranking_losses: list[mx.array] = []
    for layer_index, scores in enumerate(indexer_scores):
        if scores.shape != edge_targets.shape:
            raise ValueError(
                f"graph indexer layer {layer_index} shape {tuple(scores.shape)} "
                f"!= targets {tuple(edge_targets.shape)}"
            )
        _loss, components = total_indexer_loss(
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
        edge_bce_losses.append(components["bce"])
        ranking_losses.append(components["coverage"])
    scale = (
        float(config.global_weight)
        * float(config.indexer_weight)
        * float(config.layer_weight)
    )
    edge_bce = mx.sum(mx.stack(edge_bce_losses)) * scale
    ranking = mx.sum(mx.stack(ranking_losses)) * scale
    return GraphAuxLossBreakdown(
        total=edge_bce + ranking,
        edge_bce=edge_bce,
        ranking=ranking,
        layer_count=len(indexer_scores),
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
    document_ids: mx.array | None,
    block_bias: mx.array | None,
    edge_kind_bias: mx.array | None = None,
    graph_targets: mx.array | None,
    graph_pair_mask: mx.array | None,
    graph_config: GraphAuxLossConfig | None,
    graph_weight: float,
) -> tuple[mx.array, mx.array, mx.array]:
    """Compute the differentiated LM + configured graph auxiliary objective."""

    breakdown = production_training_loss_breakdown(
        model,
        input_ids,
        targets,
        loss_mask,
        side_channels=side_channels,
        document_ids=document_ids,
        block_bias=block_bias,
        edge_kind_bias=edge_kind_bias,
        graph_targets=graph_targets,
        graph_pair_mask=graph_pair_mask,
        graph_config=graph_config,
        graph_weight=graph_weight,
    )
    return breakdown.total, breakdown.lm_ce, breakdown.graph_total


def _validate_scheduled_loss_receipt(
    input_ids: mx.array,
    *,
    objective: TaskKind | str | None,
    schedule_assignments: Sequence[Mapping[str, object]] | None,
    graph_relations: Sequence[str],
    graph_config: GraphAuxLossConfig | None,
    graph_targets: mx.array | None,
    graph_pair_mask: mx.array | None,
    graph_weight: float,
    require_schedule_receipt: bool,
) -> bool:
    """Validate control-plane bindings before entering the differentiated path."""

    graph_positive = False
    if schedule_assignments is None:
        if require_schedule_receipt:
            raise ValueError(
                "scheduled production loss requires a canonical schedule receipt"
            )
    else:
        batch_size = int(input_ids.shape[0])
        if len(schedule_assignments) != batch_size:
            raise ValueError(
                "canonical schedule assignment count does not match loss batch: "
                f"{len(schedule_assignments)} != {batch_size}"
            )
        from cppmega_mlx.training.objective_schedule import (
            validate_objective_assignment_receipt,
        )

        expected_task = (
            objective.value if isinstance(objective, TaskKind) else objective
        )
        for index, assignment in enumerate(schedule_assignments):
            validate_objective_assignment_receipt(
                assignment,
                graph_relations=graph_relations,
            )
            if expected_task is not None and assignment.get("task") != expected_task:
                raise ValueError(
                    "canonical schedule task differs from the loss batch: "
                    f"row={index}, receipt={assignment.get('task')!r}, "
                    f"loss={expected_task!r}"
                )
            graph_receipt = assignment.get("graph_eligibility")
            graph_positive = graph_positive or bool(
                isinstance(graph_receipt, Mapping)
                and graph_receipt.get("eligible") is True
            )

    graph_values = (graph_config, graph_targets, graph_pair_mask)
    if graph_positive and (
        graph_weight <= 0.0 or any(value is None for value in graph_values)
    ):
        raise ValueError(
            "graph-positive canonical assignment requires a positive graph weight "
            "and complete graph config/targets/pair_mask"
        )
    if schedule_assignments is not None and graph_relations and (
        graph_weight <= 0.0 or any(value is None for value in graph_values)
    ):
        raise ValueError(
            "canonical graph schedule requires a positive graph weight and complete "
            "graph config/targets/pair_mask"
        )
    return graph_positive


def scheduled_production_training_loss_breakdown(
    model: Any,
    input_ids: mx.array,
    targets: mx.array,
    loss_mask: mx.array,
    *,
    objective: TaskKind | str | None = None,
    schedule_assignments: Sequence[Mapping[str, object]] | None = None,
    side_channels: Mapping[str, mx.array],
    document_ids: mx.array | None,
    block_bias: mx.array | None,
    edge_kind_bias: mx.array | None = None,
    graph_targets: mx.array | None,
    graph_pair_mask: mx.array | None,
    graph_config: GraphAuxLossConfig | None,
    graph_weight: float,
    graph_relations: Sequence[str] = (),
    require_schedule_receipt: bool = False,
    allow_route_free_smoke: bool = False,
) -> ProductionTrainingLossBreakdown:
    """Run the one scheduled Stage-1 total-loss path and return its breakdown.

    The schedule is validated before the decoder is entered.  Production graph
    calls then delegate to the single differentiated LM+graph implementation;
    the route-free branch is reserved for the tiny legacy smoke and is explicit
    so it cannot masquerade as graph-conditioned production training.
    """

    graph_positive = _validate_scheduled_loss_receipt(
        input_ids,
        objective=objective,
        schedule_assignments=schedule_assignments,
        graph_relations=graph_relations,
        graph_config=graph_config,
        graph_targets=graph_targets,
        graph_pair_mask=graph_pair_mask,
        graph_weight=graph_weight,
        require_schedule_receipt=require_schedule_receipt,
    )

    model_config = getattr(model, "config", None)
    graph_routes_active = bool(
        model_config is not None
        and getattr(model_config, "graph_routes_enabled", False)
        and getattr(model_config, "require_graph_routes", False)
    )
    if graph_routes_active:
        return production_training_loss_breakdown(
            model,
            input_ids,
            targets,
            loss_mask,
            side_channels=side_channels,
            document_ids=document_ids,
            block_bias=block_bias,
            edge_kind_bias=edge_kind_bias,
            graph_targets=graph_targets,
            graph_pair_mask=graph_pair_mask,
            graph_config=graph_config,
            graph_weight=graph_weight,
        )

    if not allow_route_free_smoke:
        raise ValueError(
            "scheduled production loss requires active fail-closed graph routes"
        )
    if graph_positive or any(
        value is not None for value in (graph_config, graph_targets, graph_pair_mask)
    ):
        raise ValueError(
            "route-free smoke cannot bypass a scheduled graph objective"
        )
    if graph_weight != 0.0:
        raise ValueError("route-free smoke requires graph_weight=0")
    result = model(
        input_ids,
        targets=targets,
        loss_mask=loss_mask,
        **dict(side_channels),
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError("route-free smoke model returned an invalid loss contract")
    _logits, lm_loss = result
    if not isinstance(lm_loss, mx.array):
        raise RuntimeError("route-free smoke model returned a non-array loss")
    zero = lm_loss * mx.array(0.0, dtype=lm_loss.dtype)
    return ProductionTrainingLossBreakdown(
        total=lm_loss,
        lm_ce=lm_loss,
        graph_total=zero,
        graph_edge_bce=zero,
        graph_ranking=zero,
        graph_positive_pairs=mx.array(0.0, dtype=mx.float32),
        ntokens=mx.sum(mx.asarray(loss_mask).astype(mx.float32)),
        graph_layer_count=0,
    )


def scheduled_production_training_loss(
    model: Any,
    input_ids: mx.array,
    targets: mx.array,
    loss_mask: mx.array,
    *,
    objective: TaskKind | str | None = None,
    schedule_assignments: Sequence[Mapping[str, object]] | None = None,
    side_channels: Mapping[str, mx.array],
    document_ids: mx.array | None,
    block_bias: mx.array | None,
    edge_kind_bias: mx.array | None = None,
    graph_targets: mx.array | None,
    graph_pair_mask: mx.array | None,
    graph_config: GraphAuxLossConfig | None,
    graph_weight: float,
    graph_relations: Sequence[str] = (),
    require_schedule_receipt: bool = False,
    allow_route_free_smoke: bool = False,
) -> tuple[mx.array, mx.array, mx.array]:
    """Return the canonical scheduled total, LM, and graph losses."""

    breakdown = scheduled_production_training_loss_breakdown(
        model,
        input_ids,
        targets,
        loss_mask,
        objective=objective,
        schedule_assignments=schedule_assignments,
        side_channels=side_channels,
        document_ids=document_ids,
        block_bias=block_bias,
        edge_kind_bias=edge_kind_bias,
        graph_targets=graph_targets,
        graph_pair_mask=graph_pair_mask,
        graph_config=graph_config,
        graph_weight=graph_weight,
        graph_relations=graph_relations,
        require_schedule_receipt=require_schedule_receipt,
        allow_route_free_smoke=allow_route_free_smoke,
    )
    return breakdown.total, breakdown.lm_ce, breakdown.graph_total


def production_training_loss_breakdown(
    model: Any,
    input_ids: mx.array,
    targets: mx.array,
    loss_mask: mx.array,
    *,
    side_channels: Mapping[str, mx.array],
    document_ids: mx.array | None,
    block_bias: mx.array | None,
    edge_kind_bias: mx.array | None,
    graph_targets: mx.array | None,
    graph_pair_mask: mx.array | None,
    graph_config: GraphAuxLossConfig | None,
    graph_weight: float,
) -> ProductionTrainingLossBreakdown:
    """Compose route-conditioned CE and optional DSA indexer supervision.

    Dense GQA and DSA share one route-conditioned decoder invocation.  GQA uses
    the additive dense graph prior while retaining the full causal key set; DSA
    may additionally request the debiased neural-indexer auxiliary loss.  A
    caller that supplies only route tensors gets a route-conditioned LM loss;
    a partial or mode-incompatible indexer objective raises instead of being
    silently discarded.
    """

    if not math.isfinite(float(graph_weight)) or graph_weight < 0.0:
        raise ValueError("graph auxiliary global weight must be finite and non-negative")
    model_config = getattr(model, "config", None)
    aux_values = (graph_config, graph_targets, graph_pair_mask)
    graph_aux_requested = any(value is not None for value in aux_values)
    if graph_aux_requested and any(value is None for value in aux_values):
        raise ValueError(
            "production graph objective requires config/targets/pair_mask together"
        )
    if graph_aux_requested:
        assert graph_config is not None
        assert graph_targets is not None
        assert graph_pair_mask is not None
        if graph_weight <= 0.0:
            raise ValueError(
                "graph auxiliary global weight must be finite and positive when "
                "indexer supervision is configured"
            )
        if graph_weight != graph_config.global_weight:
            raise ValueError(
                "graph auxiliary global weight differs from GraphAuxLossConfig: "
                f"{graph_weight} != {graph_config.global_weight}"
            )
    elif graph_weight != 0.0:
        raise ValueError(
            "graph auxiliary global weight must be zero when indexer supervision "
            "is not configured"
        )
    if block_bias is None:
        raise ValueError("production graph objective requires graph route block_bias")
    if document_ids is None:
        raise ValueError("production graph objective requires document_ids")
    if tuple(document_ids.shape) != tuple(input_ids.shape):
        raise ValueError(
            "production graph objective document_ids must match input_ids shape "
            f"{tuple(input_ids.shape)}, got {tuple(document_ids.shape)}"
        )
    if (
        model_config is None
        or getattr(model_config, "attention_mode", None) not in {"gqa", "dsa"}
        or getattr(model_config, "require_graph_routes", None) is not True
        or getattr(model_config, "graph_routes_enabled", None) is not True
    ):
        raise ValueError(
            "production graph objective requires active fail-closed DSA graph routes "
            "or active dense GQA graph routes"
        )
    attention_mode = str(getattr(model_config, "attention_mode"))
    if attention_mode == "gqa" and graph_aux_requested:
        raise ValueError(
            "production indexer supervision requires attention_mode='dsa'; "
            "dense GQA uses additive graph bias while retaining all eligible tokens"
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
        name for name in required_structure_channels if side_channels.get(name) is None
    )
    if missing_structure_channels:
        raise ValueError(
            "production graph objective is missing required structure sidecars: "
            + ", ".join(missing_structure_channels)
        )
    if edge_kind_bias is None:
        raise ValueError(
            "production graph objective requires an explicit graph edge-kind prior"
        )
    if tuple(edge_kind_bias.shape) != tuple(block_bias.shape):
        raise ValueError(
            "production graph edge-kind prior must match relation prior shape "
            f"{tuple(block_bias.shape)}, got {tuple(edge_kind_bias.shape)}"
        )
    if graph_aux_requested:
        assert graph_config is not None
        model_beta = getattr(model_config, "graph_attention_bias_beta", None)
        if model_beta is None or float(model_beta) != float(graph_config.bias_beta):
            raise ValueError(
                "production graph bias_beta differs between objective recipe and "
                f"model config: {graph_config.bias_beta} != {model_beta}"
            )
    decoder_forward = getattr(model, "decoder_hidden_states", None)
    if not callable(decoder_forward):
        raise TypeError(
            "production graph objective requires model.decoder_hidden_states"
        )
    if attention_mode == "dsa":
        decoder_result = decoder_forward(
            input_ids,
            document_ids=document_ids,
            block_bias=block_bias,
            edge_kind_bias=edge_kind_bias,
            return_indexer_scores=True,
            **side_channels,
        )
        if not isinstance(decoder_result, tuple) or len(decoder_result) != 2:
            raise RuntimeError(
                "production graph decoder did not return hidden states and indexer scores"
            )
        hidden_states, indexer_scores = decoder_result
        if not isinstance(hidden_states, mx.array) or not isinstance(indexer_scores, tuple):
            raise RuntimeError(
                "production graph decoder returned an invalid hidden/indexer contract"
            )
        if graph_aux_requested:
            graph_supervision_scores = getattr(model, "graph_supervision_scores", None)
            if not callable(graph_supervision_scores):
                raise TypeError(
                    "production graph objective requires model.graph_supervision_scores"
                )
            learned_indexer_scores = graph_supervision_scores(
                indexer_scores,
                input_ids=input_ids,
                document_ids=document_ids,
                block_bias=block_bias,
                edge_kind_bias=edge_kind_bias,
            )
            if not isinstance(learned_indexer_scores, tuple):
                raise RuntimeError(
                    "production graph supervision returned an invalid score contract"
                )
        else:
            learned_indexer_scores = ()
    elif attention_mode == "gqa":
        hidden_states = decoder_forward(
            input_ids,
            document_ids=document_ids,
            block_bias=block_bias,
            edge_kind_bias=edge_kind_bias,
            **side_channels,
        )
        if not isinstance(hidden_states, mx.array):
            raise RuntimeError(
                "production GQA decoder returned an invalid hidden-state contract"
            )
        learned_indexer_scores = ()
    else:  # pragma: no cover - config validation should make this unreachable
        raise ValueError(f"unsupported production graph attention mode {attention_mode!r}")

    if bool(getattr(model_config, "chunked_ce", False)):
        chunked_cross_entropy = getattr(model, "_chunked_cross_entropy", None)
        if not callable(chunked_cross_entropy):
            raise TypeError("chunked CE requires model._chunked_cross_entropy")
        lm_loss = chunked_cross_entropy(
            hidden_states,
            targets,
            loss_mask,
            int(getattr(model_config, "ce_chunk_size")),
        )
    else:
        lm_head = getattr(model, "lm_head", None)
        cross_entropy = getattr(model, "_cross_entropy", None)
        if not callable(lm_head) or not callable(cross_entropy):
            raise TypeError("production graph objective requires LM head/CE helpers")
        logits = lm_head(hidden_states)
        lm_loss = cross_entropy(logits, targets, loss_mask)

    if graph_aux_requested:
        assert graph_config is not None
        assert graph_targets is not None
        assert graph_pair_mask is not None
        graph_breakdown = graph_auxiliary_loss_breakdown_from_targets(
            learned_indexer_scores,
            graph_targets,
            graph_pair_mask,
            graph_config,
        )
        graph_positive_pairs = mx.sum(graph_targets.astype(mx.float32))
    else:
        zero = lm_loss * mx.array(0.0, dtype=lm_loss.dtype)
        graph_breakdown = GraphAuxLossBreakdown(
            total=zero,
            edge_bce=zero,
            ranking=zero,
            layer_count=0,
        )
        graph_positive_pairs = mx.array(0.0, dtype=mx.float32)
    ntokens = mx.sum(loss_mask.astype(mx.float32))
    return ProductionTrainingLossBreakdown(
        total=lm_loss + graph_breakdown.total,
        lm_ce=lm_loss,
        graph_total=graph_breakdown.total,
        graph_edge_bce=graph_breakdown.edge_bce,
        graph_ranking=graph_breakdown.ranking,
        graph_positive_pairs=graph_positive_pairs,
        ntokens=ntokens,
        graph_layer_count=graph_breakdown.layer_count,
    )


__all__ = [
    "EligibilityAwareTaskMixer",
    "GraphAuxLossBreakdown",
    "GraphAuxLossConfig",
    "ObjectiveAccounting",
    "ObjectiveSource",
    "PACKED_COMMIT_BINDING_METADATA_KEY",
    "PACKED_COMMIT_BINDING_SCHEMA",
    "ProductionTrainingLossBreakdown",
    "RealizedObjective",
    "combine_lm_and_aux_losses",
    "compute_graph_auxiliary_loss",
    "graph_auxiliary_loss_breakdown_from_targets",
    "graph_auxiliary_loss_from_targets",
    "production_training_loss",
    "production_training_loss_breakdown",
    "scheduled_production_training_loss",
    "scheduled_production_training_loss_breakdown",
    "validated_packed_commit_binding",
]

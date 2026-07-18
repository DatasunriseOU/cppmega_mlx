from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from cppmega_mlx.data.code_packet import CodePacket
from cppmega_mlx.data.commit_packet import CommitPacket
from cppmega_mlx.data.graph_packet import EdgeIndex
from cppmega_mlx.training.objective_mixer import (
    EligibilityAwareTaskMixer,
    ObjectiveSource,
)
from cppmega_mlx.training.objective_schedule import (
    CanonicalObjectivePlanner,
    GRAPH_ELIGIBILITY_RECEIPT_SCHEMA,
    assess_graph_positive_capability,
)
from cppmega_mlx.training.task_mixer import TaskKind
from scripts import materialize_megatron_objectives as materializer
from scripts.train_eval_stage1 import _objective_batches


def _arr(values: list[int]) -> mx.array:
    return mx.array(np.asarray(values, dtype=np.int32))


def _code_packet(*, call_edge: tuple[int, int] | None = None) -> CodePacket:
    token_count = 8
    zeros = _arr([0] * token_count)
    return CodePacket(
        token_ids=_arr(list(range(100, 100 + token_count))),
        document_ids=_arr([1] * token_count),
        ifim_instruction_token_ids=_arr([1201, 1202]),
        structure_ids=_arr([1] * token_count),
        dep_levels=zeros,
        ast_depth=zeros,
        sibling_index=zeros,
        ast_node_type=_arr([1] * token_count),
        symbol_ids=zeros,
        call_targets=zeros,
        type_refs=zeros,
        def_use=zeros,
        domain_ids=zeros,
        role_ids=zeros,
        entity_ids=zeros,
        scope_ids=zeros,
        confidence_ids=_arr([1] * token_count),
        source_doc_ids=_arr([1] * token_count),
        source_identity_ids=_arr([1] * token_count),
        chunk_starts=_arr([0, 2, 5]),
        chunk_ends=_arr([2, 5, 8]),
        chunk_kinds=_arr([1, 1, 1]),
        chunk_dep_levels=_arr([0, 0, 0]),
        call_edges=EdgeIndex.from_pairs(
            [] if call_edge is None else [call_edge],
            relation="call",
            num_nodes=3,
        ),
        type_edges=EdgeIndex.from_pairs([], relation="type", num_nodes=3),
        metadata={"platform_ids": [2]},
    )


def _typed_sources() -> list[ObjectiveSource]:
    commit = CommitPacket(
        pre_token_ids=_arr([10, 11]),
        post_token_ids=_arr([20, 21]),
        diff_token_ids=_arr([30, 31]),
        commit_msg=_arr([40, 41]),
    )
    return [
        ObjectiveSource(code_packet=_code_packet()),
        ObjectiveSource(code_packet=_code_packet(), commit_packet=commit),
        ObjectiveSource(code_packet=_code_packet(call_edge=(1, 0))),
    ]


class _Sink:
    def add(self, _value) -> None:
        return None


def test_materializer_and_runner_share_canonical_bounded_schedule_receipt() -> None:
    rates = {TaskKind.CAUSAL_LM: 0.5, TaskKind.COMMIT_DIFF: 0.5}
    sources = _typed_sources()

    materialized = materializer._materialize_stream(
        mixer=EligibilityAwareTaskMixer(rates, seed=23),
        source_iter=iter(sources),
        accumulator=_Sink(),
        writer=_Sink(),
        samples=2,
        quota_window_samples=2,
        quota_lookahead_samples=1,
        graph_relations=("call",),
        require_route_sidecars=False,
    )
    runner_receipts: list[dict[str, object]] = []
    batches = _objective_batches(
        iter(_typed_sources()),
        EligibilityAwareTaskMixer(rates, seed=23),
        batch_size=1,
        seq_len=64,
        quota_window_samples=2,
        quota_lookahead_samples=1,
        seed=23,
        graph_relations=("call",),
        require_route_sidecars=False,
        schedule_receipts=runner_receipts,
    )
    runner_batches = [next(batches), next(batches)]

    assert materialized["windows"] == runner_receipts
    assert all(
        batch.schedule_window_receipt == runner_receipts[0]
        for batch in runner_batches
    )
    window = runner_receipts[0]
    assert window["selected_source_indices"] == [1, 2]
    assert window["graph_positive_assignments"] == 1
    assignments = window["assignments"]
    assert isinstance(assignments, list)
    graph_assignment = next(
        row for row in assignments if row["source_index"] == 2
    )
    assert graph_assignment["task"] == TaskKind.CAUSAL_LM.value
    assert graph_assignment["graph_eligibility"]["eligible"] is True


def test_graph_capability_receipts_use_realized_fim_ifim_routes_and_commit_exclusion() -> None:
    code_source = ObjectiveSource(code_packet=_code_packet(call_edge=(1, 0)))
    for task in (TaskKind.FIM, TaskKind.IFIM):
        realized = EligibilityAwareTaskMixer({task: 1.0}, seed=23).materialize(
            code_source,
            step_index=0,
        )
        receipt = assess_graph_positive_capability(
            realized,
            code_source,
            graph_relations=("call",),
            require_route_sidecars=False,
        )
        assert receipt["schema"] == GRAPH_ELIGIBILITY_RECEIPT_SCHEMA
        assert receipt["eligible"] is True
        assert receipt["route_mode"] == "source_token_remap"
        assert receipt["positive_edges"] > 0

    commit_source = _typed_sources()[1]
    realized_commit = EligibilityAwareTaskMixer(
        {TaskKind.COMMIT_DIFF: 1.0}, seed=23
    ).materialize(commit_source, step_index=0)
    receipt = assess_graph_positive_capability(
        realized_commit,
        commit_source,
        graph_relations=("call",),
        require_route_sidecars=False,
    )
    assert receipt["schema"] == GRAPH_ELIGIBILITY_RECEIPT_SCHEMA
    assert receipt["eligible"] is False
    assert receipt["route_mode"] == "excluded"
    assert receipt["reason"] == "exact_source_route_map_unavailable"
    assert receipt["route_receipt"]["mode"] == "excluded"


@pytest.mark.parametrize("graph_task", [TaskKind.FIM, TaskKind.IFIM])
def test_canonical_planner_accepts_transformed_graph_positive_assignment(
    graph_task: TaskKind,
) -> None:
    planner = CanonicalObjectivePlanner(
        mixer=EligibilityAwareTaskMixer(
            {graph_task: 0.5, TaskKind.COMMIT_DIFF: 0.5},
            seed=23,
        ),
        source_iter=iter(_typed_sources()),
        quota_window_samples=2,
        quota_lookahead_samples=1,
        graph_relations=("call",),
        require_route_sidecars=False,
    )

    window = planner.plan_window(start_step=0)

    graph_assignment = next(
        assignment
        for assignment in window.assignments
        if assignment.realized.task is graph_task
    )
    assert graph_assignment.source_index == 2
    assert graph_assignment.graph_eligibility["eligible"] is True
    assert graph_assignment.graph_eligibility["route_mode"] == "source_token_remap"
    assert graph_assignment.graph_eligibility["positive_edges"] == 6


def test_transformed_objective_without_exact_route_map_is_explicitly_ineligible() -> None:
    realized = SimpleNamespace(
        task=TaskKind.FIM,
        example=SimpleNamespace(
            input_ids=_arr([1, 2]),
            metadata={},
        ),
    )

    receipt = assess_graph_positive_capability(
        realized,  # type: ignore[arg-type]
        ObjectiveSource(code_packet=_code_packet(call_edge=(1, 0))),
        graph_relations=("call",),
        require_route_sidecars=False,
    )

    assert receipt["eligible"] is False
    assert receipt["reason"] == "missing_exact_source_token_route_map"
    assert receipt["route_mode"] == "unavailable"
    assert receipt["route_receipt"] is None

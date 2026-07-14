from __future__ import annotations

from cppmega_mlx.training.megatron_objectives import (
    MaterializedMegatronDocument,
    build_pre_materialized_objective_contract,
)
from cppmega_mlx.training.objective_contract_accumulator import (
    ObjectiveContractAccumulator,
    count_configured_graph_positive_edges,
)
from cppmega_mlx.training.objective_mixer import (
    EligibilityAwareTaskMixer,
    GraphAuxLossConfig,
)
from cppmega_mlx.training.task_mixer import STAGE1_DEFAULT_RATES, TaskKind


def _document(
    task: TaskKind,
    *,
    graph: bool = False,
) -> MaterializedMegatronDocument:
    token_ids = [101, 102, 103, 104]
    loss_mask = [1, 1, 1, 0]
    return MaterializedMegatronDocument(
        objective_kind=task.value,
        token_ids=token_ids,
        loss_mask=loss_mask,
        graph_edge_count=int(graph),
        row={
            "doc_ids": [1, 1, 1, 1],
            "token_chunk_starts": [0, 2],
            "token_chunk_ends": [2, 3],
            "token_call_edges": [{"from": 1, "to": 0}] if graph else [],
            "token_type_edges": [],
        },
    )


def test_incremental_contract_exactly_matches_reference_builder() -> None:
    mixer = EligibilityAwareTaskMixer(STAGE1_DEFAULT_RATES, seed=17)
    quotas = mixer.quotas(60)
    documents = [
        _document(task, graph=task is TaskKind.CAUSAL_LM and index == 0)
        for task in TaskKind
        for index in range(quotas[task])
    ]
    graph_config = GraphAuxLossConfig(
        relations=("call", "type"),
        global_weight=1.0,
        indexer_weight=0.001,
        layer_weight=1.0,
        bce_weight=0.10,
        coverage_weight=0.05,
    )

    reference = build_pre_materialized_objective_contract(
        documents,
        rates=mixer.rates,
        seed=17,
        quota_window_samples=60,
        graph_config=graph_config,
        graph_weight=1.0,
    )
    accumulator = ObjectiveContractAccumulator(
        rates=mixer.rates,
        seed=17,
        quota_window_samples=60,
        graph_config=graph_config,
        graph_weight=1.0,
    )
    for document in documents:
        accumulator.add(document)

    assert accumulator.finalize() == reference


def test_full_span_chunk_edge_counts_without_cartesian_pair_storage() -> None:
    input_length = 20_000
    document = MaterializedMegatronDocument(
        objective_kind=TaskKind.CAUSAL_LM.value,
        token_ids=[1] * (input_length + 1),
        loss_mask=[1] * input_length + [0],
        graph_edge_count=1,
        row={
            "doc_ids": [7] * (input_length + 1),
            "token_chunk_starts": [0],
            "token_chunk_ends": [input_length],
            "token_call_edges": [{"from": 0, "to": 0}],
            "token_type_edges": [],
        },
    )

    assert (
        count_configured_graph_positive_edges(
            document,
            relations=("call", "type"),
        )
        == input_length * (input_length + 1) // 2
    )

from __future__ import annotations

from dataclasses import replace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from cppmega_mlx.data.code_packet import CodePacket
from cppmega_mlx.data.commit_packet import CommitPacket
from cppmega_mlx.data.domain_packet import DomainEdgeIndex
from cppmega_mlx.data.graph_packet import EdgeIndex
from cppmega_mlx.data.tokenizer_contract import DOMAIN_DELIMITER_TOKEN_IDS
from cppmega_mlx.models.dense_cpp_lm import DenseCppLM, DenseCppLMConfig
from cppmega_mlx.training.objective_mixer import (
    EligibilityAwareTaskMixer,
    GraphAuxLossConfig,
    ObjectiveSource,
    production_training_loss,
)
from cppmega_mlx.training.task_mixer import TaskKind
from scripts.train_eval_stage1 import ObjectiveBatch, _materialize_batch


def _arr(values: list[int]) -> mx.array:
    return mx.array(np.asarray(values, dtype=np.int32))


def _code_packet() -> CodePacket:
    tokens = [
        2,
        DOMAIN_DELIMITER_TOKEN_IDS["CPP_CODE_START"],
        100,
        101,
        102,
        103,
        104,
        DOMAIN_DELIMITER_TOKEN_IDS["CPP_CODE_END"],
    ]
    zeros = _arr([0] * len(tokens))
    return CodePacket(
        token_ids=_arr(tokens),
        document_ids=_arr([1] * len(tokens)),
        ifim_instruction_token_ids=_arr([1201, 1202]),
        structure_ids=_arr([1] * len(tokens)),
        dep_levels=zeros,
        ast_depth=zeros,
        sibling_index=zeros,
        ast_node_type=_arr([1] * len(tokens)),
        symbol_ids=_arr([0, 0, 7, 7, 0, 0, 0, 0]),
        call_targets=_arr([0, 0, 0, 0, 6, 6, 0, 0]),
        type_refs=_arr([0, 0, 0, 0, 8, 8, 0, 0]),
        def_use=zeros,
        domain_ids=zeros,
        role_ids=zeros,
        entity_ids=zeros,
        scope_ids=zeros,
        confidence_ids=_arr([1] * len(tokens)),
        call_edges=EdgeIndex.from_pairs([], relation="call", num_nodes=3),
        type_edges=EdgeIndex.from_pairs([], relation="type", num_nodes=3),
        domain_edges=DomainEdgeIndex.empty(),
        build_edges=DomainEdgeIndex.empty(),
        shell_edges=DomainEdgeIndex.empty(),
        diagnostic_edges=DomainEdgeIndex.empty(),
        cross_domain_edges=DomainEdgeIndex.empty(),
        chunk_starts=_arr([0, 2, 5]),
        chunk_ends=_arr([2, 5, 8]),
        chunk_kinds=_arr([1, 1, 1]),
        chunk_dep_levels=_arr([0, 0, 0]),
    )


def _commit_packet() -> CommitPacket:
    return CommitPacket(
        pre_token_ids=_arr([10, 11]),
        post_token_ids=_arr([20, 21]),
        diff_token_ids=_arr([30, 31]),
        commit_msg=_arr([40, 41]),
    )


def _route_batches(
    task: TaskKind,
) -> tuple[ObjectiveBatch, ObjectiveBatch, int, int]:
    packet = _code_packet()
    source = ObjectiveSource(code_packet=packet)
    realized = EligibilityAwareTaskMixer({task: 1.0}, seed=37).materialize(
        source, step_index=0
    )
    source_map = list(realized.example.metadata["source_token_indices"][:-1])
    mapped = [
        (output_index, source_index)
        for output_index, source_index in enumerate(source_map)
        if source_index >= 0
    ]
    assert len(mapped) >= 2
    key_output, key_source = mapped[0]
    query_output, query_source = mapped[-1]
    assert query_output > key_output
    routed_packet = replace(
        packet,
        domain_edges=DomainEdgeIndex.from_triples(
            [(query_source, key_source, 60)]
        ),
    )
    routed = _materialize_batch(
        task,
        [(realized.example, ObjectiveSource(code_packet=routed_packet))],
        seq_len=32,
        graph_relations=("domain",),
    )
    graphless = _materialize_batch(
        task,
        [(realized.example, source)],
        seq_len=32,
        graph_relations=("domain",),
    )
    return routed, graphless, query_output, key_output


@pytest.mark.parametrize(
    "task",
    (
        TaskKind.CAUSAL_LM,
        TaskKind.FIM,
        TaskKind.IFIM,
        TaskKind.SYMBOL_RECOVERY,
    ),
)
def test_valid_objective_route_changes_remapped_bias_and_targets(task: TaskKind) -> None:
    routed, graphless, query_output, key_output = _route_batches(task)
    mx.eval(
        routed.block_bias,
        graphless.block_bias,
        routed.graph_targets,
        routed.graph_pair_mask,
    )

    assert float(routed.block_bias[0, query_output, key_output].item()) > 0.0
    assert float(routed.graph_targets[0, query_output, key_output].item()) == 1.0
    assert float(routed.graph_pair_mask[0, query_output, key_output].item()) == 1.0
    assert routed.graph_samples == 1
    assert routed.graph_edges == 1
    assert routed.graph_route_exclusion_reason is None
    relation_receipt = routed.graph_route_receipts[0]["relations"]["domain"]
    assert relation_receipt["retained_edges"] == 1
    assert float(
        mx.sum(mx.abs(routed.block_bias - graphless.block_bias)).item()
    ) > 0.0


def test_remapped_route_changes_dsa_loss_and_gradients() -> None:
    routed, graphless, _query_output, _key_output = _route_batches(TaskKind.FIM)
    config = DenseCppLMConfig(
        vocab_size=65536,
        hidden_size=16,
        depth=1,
        ffn_hidden_size=32,
        max_seq_length=32,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=4,
        attention_mode="dsa",
        attention_sparse_topk=2,
        indexer_heads=2,
        indexer_dim=4,
        indexer_local_window=0,
        indexer_num_sinks=0,
        graph_routes_enabled=True,
        require_graph_routes=True,
        ngram_hash_enabled=False,
    )
    graph_config = GraphAuxLossConfig(
        relations=("domain",),
        topk=2,
        global_weight=1.0,
        indexer_weight=1.0,
        layer_weight=1.0,
        bce_weight=1.0,
        coverage_weight=0.5,
    )
    mx.random.seed(47)
    model = DenseCppLM(config)

    def loss(batch: ObjectiveBatch) -> tuple[mx.array, mx.array, mx.array]:
        return production_training_loss(
            model,
            batch.input_ids,
            batch.targets,
            batch.loss_mask,
            side_channels=batch.side_channels,
            document_ids=batch.document_ids,
            block_bias=batch.block_bias,
            edge_kind_bias=batch.edge_kind_bias,
            graph_targets=batch.graph_targets,
            graph_pair_mask=batch.graph_pair_mask,
            graph_config=graph_config,
            graph_weight=graph_config.global_weight,
        )

    routed_loss_and_grad = nn.value_and_grad(model, lambda: loss(routed))
    graphless_loss_and_grad = nn.value_and_grad(model, lambda: loss(graphless))
    routed_losses, routed_gradients = routed_loss_and_grad()
    graphless_losses, graphless_gradients = graphless_loss_and_grad()
    routed_total, _routed_lm, routed_graph = routed_losses
    graphless_total, _graphless_lm, graphless_graph = graphless_losses
    routed_arrays = {
        name: value
        for name, value in tree_flatten(routed_gradients)
        if isinstance(value, mx.array)
    }
    graphless_arrays = {
        name: value
        for name, value in tree_flatten(graphless_gradients)
        if isinstance(value, mx.array)
    }
    assert routed_arrays.keys() == graphless_arrays.keys()
    gradient_delta = mx.sum(
        mx.stack(
            [
                mx.sum(mx.abs(routed_arrays[name] - graphless_arrays[name]))
                for name in routed_arrays
            ]
        )
    )
    mx.eval(
        routed_total,
        graphless_total,
        routed_graph,
        graphless_graph,
        gradient_delta,
    )

    assert float(routed_graph.item()) > 0.0
    assert float(graphless_graph.item()) == 0.0
    assert abs(float(routed_total.item()) - float(graphless_total.item())) > 1e-6
    assert float(gradient_delta.item()) > 1e-6


def test_commit_repair_routes_are_explicitly_excluded_with_zero_tensors() -> None:
    packet = replace(
        _code_packet(),
        domain_edges=DomainEdgeIndex.from_triples([(5, 2, 60)]),
    )
    commit = _commit_packet()
    source = ObjectiveSource(code_packet=packet, commit_packet=commit)
    realized = EligibilityAwareTaskMixer(
        {TaskKind.COMMIT_DIFF: 1.0}, seed=37
    ).materialize(source, step_index=0)
    batch = _materialize_batch(
        TaskKind.COMMIT_DIFF,
        [(realized.example, source)],
        seq_len=32,
        graph_relations=("domain",),
    )
    mx.eval(batch.block_bias, batch.edge_kind_bias, batch.graph_targets, batch.graph_pair_mask)

    assert float(mx.sum(mx.abs(batch.block_bias)).item()) == 0.0
    assert float(mx.sum(mx.abs(batch.edge_kind_bias)).item()) == 0.0
    assert float(mx.sum(batch.graph_targets).item()) == 0.0
    assert float(mx.sum(batch.graph_pair_mask).item()) == 0.0
    assert batch.graph_samples == 0
    assert batch.graph_edges == 0
    assert (
        batch.graph_route_exclusion_reason
        == "independently_tokenized_commit_sections_have_no_exact_source_map"
    )
    relation_receipt = batch.graph_route_receipts[0]["relations"]["domain"]
    assert relation_receipt["source_edges"] == 1
    assert relation_receipt["excluded_edges"] == 1


def test_malformed_excluded_commit_route_fails_closed() -> None:
    packet = replace(
        _code_packet(),
        domain_edges=DomainEdgeIndex.from_triples([(99, 2, 60)]),
    )
    malformed_source = ObjectiveSource(code_packet=packet, commit_packet=_commit_packet())
    realized = EligibilityAwareTaskMixer(
        {TaskKind.COMMIT_DIFF: 1.0}, seed=37
    ).materialize(malformed_source, step_index=0)
    with pytest.raises(
        ValueError, match=r"domain edge \(99, 2\) is outside 8 source tokens"
    ):
        _materialize_batch(
            TaskKind.COMMIT_DIFF,
            [(realized.example, malformed_source)],
            seq_len=32,
            graph_relations=("domain",),
        )

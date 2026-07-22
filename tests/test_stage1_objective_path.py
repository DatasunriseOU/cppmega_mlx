"""Focused regressions for the single Stage-1 objective path."""

from __future__ import annotations

import ast
import itertools
from collections import Counter
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from cppmega_mlx.data.code_packet import CodePacket
from cppmega_mlx.data.commit_packet import CommitPacket
from cppmega_mlx.data.domain_packet import DomainEdgeIndex
from cppmega_mlx.data.graph_packet import EdgeIndex
from cppmega_mlx.models.dense_cpp_lm import DenseCppLM, DenseCppLMConfig
from cppmega_mlx.training.objective_mixer import (
    EligibilityAwareTaskMixer,
    GraphAuxLossConfig,
    ObjectiveSource,
    scheduled_production_training_loss,
)
from cppmega_mlx.training.task_mixer import TaskKind
from scripts.train_eval_stage1 import _materialize_batch, _objective_batches
from scripts.train_stage1 import (
    _loss_for_step,
    build_train_step,
    materialize_steps,
)


ROOT = Path(__file__).resolve().parents[1]


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
        call_edges=EdgeIndex.from_pairs(
            [] if call_edge is None else [call_edge],
            relation="call",
            num_nodes=3,
        ),
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


def _all_repair_rates() -> dict[TaskKind, float]:
    return {
        TaskKind.CAUSAL_LM: 0.2,
        TaskKind.FIM: 0.2,
        TaskKind.IFIM: 0.2,
        TaskKind.COMMIT_DIFF: 0.2,
        TaskKind.PRE_TO_POST: 0.2,
    }


def test_train_stage1_mixer_realizes_fim_ifim_and_both_commit_repairs() -> None:
    # Distinct source identities model a packed multi-constituent code row. The
    # standalone commit view must remain usable without an artificial binding.
    values = dict(_code_packet().__dict__)
    values["document_ids"] = _arr([1] * 8)
    values["source_doc_ids"] = _arr([10, 10, 10, 10, 20, 20, 20, 20])
    values["source_identity_ids"] = _arr([11, 11, 11, 11, 22, 22, 22, 22])
    packed_code = CodePacket(**values)

    steps = materialize_steps(
        EligibilityAwareTaskMixer(_all_repair_rates(), seed=23),
        [packed_code],
        [_commit_packet()],
        num_steps=10,
    )

    assert Counter(step.objective for step in steps) == {
        task.value: 2 for task in _all_repair_rates()
    }
    assert all(step.input_ids.shape[0] == 1 for step in steps)


def test_train_stage1_graph_required_step_rejects_missing_graph_batch() -> None:
    packet = _code_packet()
    example = EligibilityAwareTaskMixer(
        {TaskKind.CAUSAL_LM: 1.0}, seed=7
    ).materialize(ObjectiveSource(code_packet=packet), step_index=0).example
    step = build_train_step(example.objective, example, packet)
    model = DenseCppLM(
        DenseCppLMConfig(
            vocab_size=65536,
            hidden_size=16,
            depth=1,
            ffn_hidden_size=32,
            max_seq_length=32,
            num_query_heads=2,
            num_kv_heads=1,
            head_dim=8,
            graph_routes_enabled=True,
            require_graph_routes=True,
            ngram_hash_enabled=False,
        )
    )

    # Remove the typed graph route after objective selection. Production must
    # reject the batch instead of silently turning graph supervision off.
    missing_graph_step = replace(step, graph_batch=None)
    with pytest.raises(ValueError, match="requires graph sidecars"):
        _loss_for_step(model, missing_graph_step, channels_on=True)


def test_mixed_objective_batches_reach_one_loss_composition() -> None:
    plain = ObjectiveSource(code_packet=_code_packet())
    graph = ObjectiveSource(code_packet=_code_packet(call_edge=(1, 0)))
    repair = ObjectiveSource(
        code_packet=_code_packet(),
        commit_packet=_commit_packet(),
    )
    batches = _objective_batches(
        itertools.cycle((plain, repair, graph)),
        EligibilityAwareTaskMixer(_all_repair_rates(), seed=23, max_input_tokens=32),
        batch_size=1,
        seq_len=32,
        quota_window_samples=5,
        quota_lookahead_samples=10,
        seed=23,
        graph_relations=("call",),
        require_route_sidecars=True,
    )
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
    model = DenseCppLM(config)
    graph_config = GraphAuxLossConfig(
        relations=("call",),
        topk=2,
        global_weight=1.0,
        indexer_weight=1.0,
        layer_weight=1.0,
        bce_weight=1.0,
        coverage_weight=0.5,
    )

    seen: dict[TaskKind, tuple[float, float, str | None]] = {}
    for _ in range(5):
        batch = next(batches)
        total, lm_loss, graph_loss = scheduled_production_training_loss(
            model,
            batch.input_ids,
            batch.targets,
            batch.loss_mask,
            objective=batch.task,
            schedule_assignments=batch.schedule_assignment_receipts,
            side_channels=batch.side_channels,
            document_ids=batch.document_ids,
            block_bias=batch.block_bias,
            edge_kind_bias=batch.edge_kind_bias,
            graph_targets=batch.graph_targets,
            graph_pair_mask=batch.graph_pair_mask,
            graph_config=graph_config,
            graph_weight=graph_config.global_weight,
            graph_relations=graph_config.relations,
            require_schedule_receipt=True,
        )
        mx.eval(total, lm_loss, graph_loss)
        total_value = float(total.item())
        lm_value = float(lm_loss.item())
        graph_value = float(graph_loss.item())
        assert total_value == pytest.approx(lm_value + graph_value, rel=1e-6)
        seen[batch.task] = (
            graph_value,
            float(mx.sum(batch.graph_targets).item()),
            batch.graph_route_exclusion_reason,
        )

    assert set(seen) == set(_all_repair_rates())
    assert any(seen[task][0] > 0.0 for task in (TaskKind.FIM, TaskKind.IFIM))
    for task in (TaskKind.COMMIT_DIFF, TaskKind.PRE_TO_POST):
        assert seen[task][0] == 0.0
        assert seen[task][1] == 0.0
        assert seen[task][2] is not None


def test_route_batch_materialization_fails_closed_on_missing_graph_sidecar() -> None:
    packet = replace(_code_packet(call_edge=(1, 0)), type_edges=None)
    source = ObjectiveSource(code_packet=packet)
    realized = EligibilityAwareTaskMixer({TaskKind.FIM: 1.0}, seed=23).materialize(
        source, step_index=0
    )

    with pytest.raises(ValueError, match="missing required sidecars"):
        _materialize_batch(
            TaskKind.FIM,
            [(realized.example, source)],
            seq_len=32,
            graph_relations=("call",),
            require_route_sidecars=True,
        )


def _called_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return names


def test_stage1_runners_use_the_mixer_and_do_not_dispatch_objective_builders() -> None:
    forbidden = {
        "TaskMixer",
        "build_fim",
        "build_ifim",
        "build_commit_diff",
        "build_pre_to_post",
    }
    train_stage1_calls = _called_names(ROOT / "scripts" / "train_stage1.py")
    train_eval_calls = _called_names(ROOT / "scripts" / "train_eval_stage1.py")

    assert "EligibilityAwareTaskMixer" in train_stage1_calls
    assert "EligibilityAwareTaskMixer" in train_eval_calls
    assert "CanonicalObjectivePlanner" in train_stage1_calls
    assert not forbidden.intersection(train_stage1_calls)
    assert not forbidden.intersection(train_eval_calls)
    assert "scheduled_production_training_loss" in train_stage1_calls
    assert "scheduled_production_training_loss" in train_eval_calls
    assert "materialize_window_from_pool" not in (
        ROOT / "scripts" / "train_stage1.py"
    ).read_text(encoding="utf-8")
    # Bundle mode is an ingress to the hash-bound pre-materialized objective;
    # it must not become a second low-level objective dispatcher.
    source = (ROOT / "scripts" / "train_stage1.py").read_text(encoding="utf-8")
    assert "run_stage1_graph_domain_production" in source
    assert "EligibilityAwareTaskMixer" in source
    materializer = (
        ROOT / "scripts" / "materialize_megatron_objectives.py"
    ).read_text(encoding="utf-8")
    production = (
        ROOT / "cppmega_mlx" / "training" / "stage1_production.py"
    ).read_text(encoding="utf-8")
    assert "EligibilityAwareTaskMixer" in materializer
    assert "CanonicalObjectivePlanner" in materializer
    assert "scheduled_production_training_loss_breakdown" in production
    assert "source_selection_receipt" in (
        ROOT / "scripts" / "train_eval_stage1.py"
    ).read_text(encoding="utf-8")
    assert "require_production_objectives=True" in (
        ROOT / "scripts" / "train_eval_stage1.py"
    ).read_text(encoding="utf-8")
    assert "production_training_loss" not in train_eval_calls

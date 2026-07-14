"""Regression tests for the production objective mixer contract."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import mlx.core as mx
import mlx.nn as nn
import pytest

from cppmega_mlx.data.code_packet import CodePacket
from cppmega_mlx.data.code_packet_builder import build_commit_packets_from_packed_row
from cppmega_mlx.data.commit_packet import CommitPacket
from cppmega_mlx.data.domain_packet import DomainEdgeIndex
from cppmega_mlx.data.graph_packet import EdgeIndex, GraphPacket
from cppmega_mlx.data.nanochat_pipeline.packed_rows_schema import (
    SOURCE_COMMIT_MSG_TOKEN_IDS_COLUMN,
    SOURCE_DIFF_TOKEN_IDS_COLUMN,
    SOURCE_IFIM_INSTRUCTION_TOKEN_IDS_COLUMN,
    SOURCE_POST_TOKEN_IDS_COLUMN,
    SOURCE_PRE_TOKEN_IDS_COLUMN,
)
from cppmega_mlx.data.nanochat_pipeline.tokenized_enriched_schema import (
    COMMIT_MSG_TOKEN_IDS_COLUMN,
    DIFF_TOKEN_IDS_COLUMN,
    IFIM_INSTRUCTION_TOKEN_IDS_COLUMN,
    POST_TOKEN_IDS_COLUMN,
    PRE_TOKEN_IDS_COLUMN,
)
from cppmega_mlx.models.dense_cpp_lm import DenseCppLM, DenseCppLMConfig
from cppmega_mlx.training.objective_mixer import (
    EligibilityAwareTaskMixer,
    GraphAuxLossConfig,
    ObjectiveAccounting,
    ObjectiveSource,
    combine_lm_and_aux_losses,
    compute_graph_auxiliary_loss,
    graph_auxiliary_loss_from_targets,
    production_training_loss,
    RealizedObjective,
)
from cppmega_mlx.training.megatron_objectives import (
    MaterializedMegatronDocument,
    OBJECTIVE_CONTRACT_SCHEMA,
    OBJECTIVE_KIND_IDS,
    OBJECTIVE_MATERIALIZATION_ARTIFACT_SCHEMA,
    build_pre_materialized_objective_contract,
    materialize_megatron_document,
    write_objective_materialization_artifact,
)
from cppmega_mlx.training.objective_data import (
    OBJECTIVE_SECTION_COLUMNS,
    OBJECTIVE_SOURCE_COLUMNS,
    graph_targets_and_pair_mask,
    objective_source_from_tokenized_row,
    require_megatron_objective_source_columns,
    require_objective_source_columns,
)
from cppmega_mlx.training.objectives import build_causal_lm
from cppmega_mlx.training.task_mixer import TaskKind, normalize_rates
from scripts.nanochat_data.pack_enriched_rows import (
    INPUT_IDS_COLUMN,
    pack_documents,
    read_tokenized_documents,
)
from scripts.materialize_megatron_objectives import (
    deterministic_source_id,
    materialized_schema,
    padded_row,
)
from scripts.train_eval_stage1 import _objective_batches


def _arr(values: list[int]) -> mx.array:
    return mx.array(np.asarray(values, dtype=np.int32))


def _code_packet(*, instruction_ids: list[int] | None = None) -> CodePacket:
    return CodePacket(
        token_ids=_arr([100, 101, 102, 103, 104, 105, 106, 107]),
        document_ids=_arr([1] * 8),
        ifim_instruction_token_ids=(
            _arr(instruction_ids) if instruction_ids is not None else None
        ),
        chunk_starts=_arr([0, 2, 5]),
        chunk_ends=_arr([2, 5, 8]),
        chunk_kinds=_arr([1, 1, 1]),
        chunk_dep_levels=_arr([0, 0, 0]),
    )


def test_commit_packets_use_only_typed_real_fields() -> None:
    columns = {
        SOURCE_PRE_TOKEN_IDS_COLUMN: [[[10, 11]]],
        SOURCE_POST_TOKEN_IDS_COLUMN: [[[20, 21]]],
        SOURCE_DIFF_TOKEN_IDS_COLUMN: [[[30, 31, 32]]],
        SOURCE_COMMIT_MSG_TOKEN_IDS_COLUMN: [[[40, 41]]],
        "repo": ["repo"],
        "filepath": ["src/demo.cc"],
        "commit_hash": ["abc123"],
    }

    packets = build_commit_packets_from_packed_row(columns, row_index=0)

    assert len(packets) == 1
    packet = packets[0]
    assert np.asarray(packet.pre_token_ids).tolist() == [10, 11]
    assert np.asarray(packet.post_token_ids).tolist() == [20, 21]
    assert np.asarray(packet.diff_token_ids).tolist() == [30, 31, 32]
    assert np.asarray(packet.commit_msg).tolist() == [40, 41]

    missing_real_fields = {
        SOURCE_PRE_TOKEN_IDS_COLUMN: [[[10, 11]]],
        SOURCE_POST_TOKEN_IDS_COLUMN: [[[20, 21]]],
    }
    assert build_commit_packets_from_packed_row(missing_real_fields, row_index=0) == []


def test_typed_tokenized_row_builds_independent_code_and_commit_views() -> None:
    row = {
        "token_ids": [1, 2, 3, 4],
        "token_structure_ids": [1, 1, 1, 1],
        "token_dep_levels": [0, 0, 0, 0],
        "token_ast_depth": [0, 1, 1, 0],
        "token_sibling_index": [0, 0, 1, 0],
        "token_ast_node_type": [1, 2, 2, 1],
        "token_symbol_ids": [0, 0, 0, 0],
        "token_call_targets": [0, 0, 0, 0],
        "token_type_refs": [0, 0, 0, 0],
        "token_def_use": [0, 0, 0, 0],
        "token_chunk_starts": [0],
        "token_chunk_ends": [4],
        "token_chunk_kinds": [3],
        "token_chunk_dep_levels": [0],
        "token_call_edges": [],
        "token_type_edges": [],
        IFIM_INSTRUCTION_TOKEN_IDS_COLUMN: [50, 51],
        COMMIT_MSG_TOKEN_IDS_COLUMN: [40, 41],
        PRE_TOKEN_IDS_COLUMN: [10, 11],
        POST_TOKEN_IDS_COLUMN: [20, 21],
        DIFF_TOKEN_IDS_COLUMN: [30, 31],
        "token_change_mask_pre": [0, 1, 0, 0],
        "token_change_mask_post": [0, 0, 1, 0],
        "hunk_id_per_token": [0, 1, 1, 0],
        "edit_op_per_token": [0, 2, 1, 0],
    }
    require_objective_source_columns(tuple(row))

    source = objective_source_from_tokenized_row(row, source_index=9)

    assert np.asarray(source.code_packet.ifim_instruction_token_ids).tolist() == [
        50,
        51,
    ]
    assert np.asarray(source.commit_packet.diff_token_ids).tolist() == [30, 31]
    assert source.commit_packet.change_mask_pre is None
    assert source.commit_packet.change_mask_post is None


def test_task_specific_columns_are_optional_until_that_objective_is_selected() -> None:
    row = {
        "token_ids": [1, 2, 3, 4],
        "token_structure_ids": [1, 1, 1, 1],
        "token_dep_levels": [0, 0, 0, 0],
        "token_ast_depth": [0, 1, 1, 0],
        "token_sibling_index": [0, 0, 1, 0],
        "token_ast_node_type": [1, 2, 2, 1],
        "token_symbol_ids": [0, 0, 0, 0],
        "token_call_targets": [0, 0, 0, 0],
        "token_type_refs": [0, 0, 0, 0],
        "token_def_use": [0, 0, 0, 0],
        "token_chunk_starts": [0, 1],
        "token_chunk_ends": [1, 4],
        "token_chunk_kinds": [3, 3],
        "token_chunk_dep_levels": [0, 0],
        "token_call_edges": [],
        "token_type_edges": [],
    }
    require_objective_source_columns(tuple(row))

    source = objective_source_from_tokenized_row(row, source_index=9)

    assert source.code_packet is not None
    assert source.commit_packet is None
    assert np.asarray(source.code_packet.document_ids).tolist() == [1, 1, 1, 1]
    causal = EligibilityAwareTaskMixer({TaskKind.CAUSAL_LM: 1.0}, seed=3)
    assert causal.materialize(source, step_index=0).task is TaskKind.CAUSAL_LM
    with pytest.raises(ValueError, match="ifim_instruction_token_ids"):
        EligibilityAwareTaskMixer({TaskKind.IFIM: 1.0}, seed=3).materialize(
            source, step_index=0
        )

    row[COMMIT_MSG_TOKEN_IDS_COLUMN] = [40, 41]
    row[DIFF_TOKEN_IDS_COLUMN] = [30, 31]
    partial_commit = objective_source_from_tokenized_row(row, source_index=10)
    assert partial_commit.commit_packet is not None
    assert partial_commit.commit_packet.pre_token_ids is None
    commit_diff = EligibilityAwareTaskMixer({TaskKind.COMMIT_DIFF: 1.0}, seed=3)
    assert commit_diff.materialize(partial_commit, step_index=0).task is TaskKind.COMMIT_DIFF
    with pytest.raises(ValueError, match="pre_token_ids.*post_token_ids"):
        EligibilityAwareTaskMixer({TaskKind.PRE_TO_POST: 1.0}, seed=3).materialize(
            partial_commit, step_index=0
        )


def test_megatron_materialization_source_schema_fails_closed_on_missing_sidecars() -> (
    None
):
    with pytest.raises(ValueError, match="token_domain_ids"):
        require_megatron_objective_source_columns(
            (
                "token_ids",
                "ifim_instruction_token_ids",
                "commit_msg_token_ids",
                "pre_token_ids",
                "post_token_ids",
                "diff_token_ids",
            )
        )


def test_megatron_source_schema_does_not_require_every_task_section() -> None:
    columns = tuple(
        column
        for column in OBJECTIVE_SOURCE_COLUMNS
        if column not in OBJECTIVE_SECTION_COLUMNS
    )

    require_megatron_objective_source_columns(columns)


def test_commit_section_schema_round_trips_through_packer(tmp_path) -> None:
    shard = tmp_path / "commit_sections.parquet"
    pq.write_table(
        pa.table(
            {
                "token_ids": [[1, 2, 3]],
                IFIM_INSTRUCTION_TOKEN_IDS_COLUMN: [[50, 51]],
                COMMIT_MSG_TOKEN_IDS_COLUMN: [[40, 41]],
                DIFF_TOKEN_IDS_COLUMN: [[30, 31]],
                PRE_TOKEN_IDS_COLUMN: [[10, 11]],
                POST_TOKEN_IDS_COLUMN: [[20, 21]],
            }
        ),
        shard,
    )

    docs = read_tokenized_documents(shard)
    rows, overflow = pack_documents(
        docs,
        target_length=8,
        pad_token_id=0,
        strategy="sequential",
    )

    assert overflow == []
    assert rows[0][INPUT_IDS_COLUMN][:3] == [1, 2, 3]
    assert rows[0][SOURCE_IFIM_INSTRUCTION_TOKEN_IDS_COLUMN] == [[50, 51]]
    assert rows[0][SOURCE_COMMIT_MSG_TOKEN_IDS_COLUMN] == [[40, 41]]
    assert rows[0][SOURCE_DIFF_TOKEN_IDS_COLUMN] == [[30, 31]]
    assert rows[0][SOURCE_PRE_TOKEN_IDS_COLUMN] == [[10, 11]]
    assert rows[0][SOURCE_POST_TOKEN_IDS_COLUMN] == [[20, 21]]


def test_rendered_source_text_does_not_make_ifim_eligible() -> None:
    mixer = EligibilityAwareTaskMixer(
        {TaskKind.IFIM: 0.5, TaskKind.AST_FIM: 0.5},
        seed=7,
    )
    packet = CodePacket(
        **{
            **_code_packet().__dict__,
            "metadata": {"source_text": "/** @brief fabricated wrapper */"},
        }
    )

    realized = mixer.materialize(packet, step_index=0)

    assert realized.task is TaskKind.AST_FIM
    assert realized.example.objective == "ast_fim"
    assert TaskKind.IFIM in realized.ineligible
    assert "ifim_instruction_token_ids" in realized.ineligible[TaskKind.IFIM]


def test_empty_chunk_metadata_does_not_make_ast_fim_eligible() -> None:
    packet = CodePacket(
        token_ids=_arr([100, 101, 102]),
        chunk_starts=_arr([]),
        chunk_ends=_arr([]),
        chunk_kinds=_arr([]),
        chunk_dep_levels=_arr([]),
    )
    mixer = EligibilityAwareTaskMixer({TaskKind.AST_FIM: 1.0}, seed=9)

    with pytest.raises(ValueError, match="no eligible objective.*chunk"):
        mixer.materialize(packet, step_index=0)


def test_no_interior_chunk_does_not_consume_ast_fim_quota_as_plain_fim() -> None:
    packet = CodePacket(
        token_ids=_arr([100, 101, 102, 103]),
        chunk_starts=_arr([0]),
        chunk_ends=_arr([4]),
        chunk_kinds=_arr([1]),
        chunk_dep_levels=_arr([0]),
    )

    with pytest.raises(ValueError, match="no eligible objective.*interior"):
        EligibilityAwareTaskMixer({TaskKind.AST_FIM: 1.0}, seed=9).materialize(
            packet, step_index=0
        )


def test_true_ifim_uses_typed_instruction_tokens_not_ast_fim() -> None:
    mixer = EligibilityAwareTaskMixer({TaskKind.IFIM: 1.0}, seed=11)

    realized = mixer.materialize(_code_packet(instruction_ids=[201, 202]), step_index=3)

    assert realized.task is TaskKind.IFIM
    assert realized.example.objective == "ifim"
    assert realized.example.metadata["fim_kind"] == "ifim"
    assert realized.example.metadata["chunk_index"] is None


def test_eligibility_aware_window_hits_exact_deterministic_quotas() -> None:
    commit = CommitPacket(
        pre_token_ids=_arr([10, 11]),
        post_token_ids=_arr([20, 21]),
        diff_token_ids=_arr([30, 31]),
        commit_msg=_arr([40, 41]),
    )
    packets = [
        _code_packet(instruction_ids=[201, 202]),
        _code_packet(instruction_ids=[203]),
        _code_packet(instruction_ids=[204]),
        _code_packet(instruction_ids=[205]),
        commit,
        commit,
        commit,
        commit,
    ]
    rates = {
        TaskKind.CAUSAL_LM: 0.25,
        TaskKind.FIM: 0.25,
        TaskKind.COMMIT_DIFF: 0.25,
        TaskKind.PRE_TO_POST: 0.25,
    }
    mixer = EligibilityAwareTaskMixer(rates, seed=17)

    first = mixer.materialize_window(packets, start_step=100)
    second = mixer.materialize_window(packets, start_step=100)

    assert [item.task for item in first] == [item.task for item in second]
    assert {task: sum(item.task is task for item in first) for task in rates} == {
        task: 2 for task in rates
    }


def test_hamilton_quotas_use_the_same_exact_decimal_math_as_receipts() -> None:
    rates = {
        TaskKind.CAUSAL_LM: 0.012896825396825396,
        TaskKind.FIM: 0.7837301587301587,
        TaskKind.AST_FIM: 0.20337301587301587,
    }

    quotas = EligibilityAwareTaskMixer(rates, seed=17).quotas(42)

    assert quotas == {
        TaskKind.CAUSAL_LM: 0,
        TaskKind.FIM: 33,
        TaskKind.AST_FIM: 9,
    }


def test_eligibility_aware_window_fails_when_quota_is_impossible() -> None:
    mixer = EligibilityAwareTaskMixer(
        {TaskKind.CAUSAL_LM: 0.5, TaskKind.COMMIT_DIFF: 0.5}, seed=19
    )

    with pytest.raises(ValueError, match="commit_diff.*quota"):
        mixer.materialize_window([_code_packet(), _code_packet()])


def test_production_batch_window_preserves_exact_task_quotas() -> None:
    code = CodePacket(
        **{
            **_code_packet(instruction_ids=[201, 202]).__dict__,
            "structure_ids": _arr([1] * 8),
            "dep_levels": _arr([0] * 8),
            "ast_depth": _arr([0] * 8),
            "sibling_index": _arr([0] * 8),
            "ast_node_type": _arr([1] * 8),
                "call_edges": EdgeIndex.from_pairs([(1, 0)], relation="call", num_nodes=3),
                "type_edges": EdgeIndex.from_pairs([], relation="type", num_nodes=3),
        }
    )
    commit = CommitPacket(
        pre_token_ids=_arr([10, 11]),
        post_token_ids=_arr([20, 21]),
        diff_token_ids=_arr([30, 31]),
        commit_msg=_arr([40, 41]),
    )
    source = ObjectiveSource(code_packet=code, commit_packet=commit)
    rates = {
        TaskKind.CAUSAL_LM: 0.2,
        TaskKind.FIM: 0.2,
        TaskKind.IFIM: 0.2,
        TaskKind.COMMIT_DIFF: 0.2,
        TaskKind.PRE_TO_POST: 0.2,
    }
    batches = _objective_batches(
        itertools.cycle([source]),
        EligibilityAwareTaskMixer(rates, seed=23, max_input_tokens=32),
        batch_size=2,
        seq_len=32,
        quota_window_samples=10,
        seed=23,
    )

    window = [next(batches) for _ in range(5)]

    assert Counter(batch.task for batch in window) == Counter(rates.keys())
    assert all(batch.input_ids.shape == (2, 32) for batch in window)
    causal = next(batch for batch in window if batch.task is TaskKind.CAUSAL_LM)
    assert causal.graph_samples == 2
    assert float(mx.sum(causal.graph_targets).item()) == 12.0
    pair_mask = np.asarray(causal.graph_pair_mask)
    assert pair_mask[0, 1, 0] == 1.0
    assert pair_mask[0, 0, 1] == 0.0


def test_objective_accounting_records_exact_samples_tokens_and_loss() -> None:
    example = build_causal_lm(CodePacket(token_ids=_arr([10, 11, 12, 13])))
    accounting = ObjectiveAccounting()

    accounting.record(TaskKind.CAUSAL_LM, example, loss=mx.array(2.0))
    report = accounting.report()

    row = report[TaskKind.CAUSAL_LM.value]
    assert row["samples"] == 1
    assert row["input_tokens"] == 3
    assert row["loss_tokens"] == 3
    assert row["loss_sum"] == 6.0
    assert row["mean_loss"] == 2.0


def test_objective_accounting_rejects_non_finite_loss() -> None:
    example = build_causal_lm(CodePacket(token_ids=_arr([10, 11, 12])))

    with pytest.raises(ValueError, match="finite"):
        ObjectiveAccounting().record(TaskKind.CAUSAL_LM, example, loss=float("nan"))


def test_graph_auxiliary_loss_enters_total_loss() -> None:
    graph = GraphPacket(
        edges={
            "call": EdgeIndex.from_pairs([(0, 2), (1, 3)], relation="call", num_nodes=4)
        },
        num_nodes=4,
    )
    indexer_scores = [mx.zeros((1, 4, 4), dtype=mx.float32)]
    cfg = GraphAuxLossConfig(
        relations=("call",),
        topk=1,
        bce_weight=1.0,
        coverage_weight=0.25,
    )

    aux_loss, aux_metrics = compute_graph_auxiliary_loss(indexer_scores, graph, cfg)
    total, metrics = combine_lm_and_aux_losses(
        mx.array(1.0, dtype=mx.float32),
        {"graph_indexer": aux_loss},
        {"graph_indexer": 2.0},
    )

    assert float(aux_loss.item()) > 0.0
    assert aux_metrics["graph_indexer_layers"] == 1
    assert abs(float(total.item()) - (1.0 + 2.0 * float(aux_loss.item()))) < 1e-6
    assert metrics["loss_graph_indexer_weight"] == 2.0


@pytest.mark.parametrize(
    "field",
    (
        "global_weight",
        "indexer_weight",
        "layer_weight",
        "bce_weight",
        "coverage_weight",
        "pos_weight",
    ),
)
def test_graph_weight_semantics_require_positive_coefficients(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        GraphAuxLossConfig(**{field: 0.0})


def test_graph_token_span_expansion_honors_edge_doc_and_upstream_masks() -> None:
    packet = CodePacket(
        token_ids=_arr([10, 11, 12, 13, 14, 15]),
        document_ids=_arr([101, 101, 202, 202, 202, 202]),
        chunk_starts=_arr([0, 2, 4]),
        chunk_ends=_arr([2, 4, 6]),
        chunk_kinds=_arr([1, 1, 1]),
        chunk_dep_levels=_arr([0, 0, 0]),
        call_edges=EdgeIndex(
            src=_arr([2, 0]),
            dst=_arr([1, 2]),
            mask=_arr([1, 0]),
            relation="call",
            num_nodes=3,
        ),
    )
    upstream = np.ones((6, 6), dtype=np.uint8)
    upstream[5, 3] = 0

    targets, pair_mask = graph_targets_and_pair_mask(
        packet,
        input_length=6,
        relations=("call",),
        upstream_pair_mask=upstream,
    )

    expected_edges = {(4, 2), (4, 3), (5, 2)}
    assert set(zip(*np.nonzero(targets))) == expected_edges
    assert pair_mask[1, 0] == 1
    assert pair_mask[2, 1] == 0
    assert pair_mask[5, 3] == 0
    assert np.all(targets <= pair_mask)


def test_graph_direct_token_relations_use_the_same_document_and_upstream_mask() -> None:
    packet = CodePacket(
        token_ids=_arr([10, 11, 12, 13]),
        document_ids=_arr([1, 1, 2, 2]),
        domain_edges=DomainEdgeIndex.from_triples(
            [(3, 2, 20), (2, 1, 20), (1, 0, 20)]
        ),
    )
    upstream = np.ones((4, 4), dtype=np.uint8)
    upstream[1, 0] = 0

    targets, pair_mask = graph_targets_and_pair_mask(
        packet,
        input_length=4,
        relations=("domain",),
        upstream_pair_mask=upstream,
    )

    assert set(zip(*np.nonzero(targets))) == {(3, 2)}
    assert pair_mask[2, 1] == 0
    assert pair_mask[1, 0] == 0
    assert np.all(targets <= pair_mask)


def test_materialized_causal_document_preserves_unique_document_ids() -> None:
    zeros = _arr([0, 0, 0, 0])
    packet = CodePacket(
        token_ids=_arr([10, 11, 12, 13]),
        document_ids=_arr([41, 41, 42, 42]),
        structure_ids=_arr([1, 1, 1, 1]),
        dep_levels=zeros,
        ast_depth=zeros,
        sibling_index=zeros,
        ast_node_type=_arr([1, 1, 1, 1]),
        symbol_ids=zeros,
        call_targets=zeros,
        type_refs=zeros,
        def_use=zeros,
        domain_ids=zeros,
        role_ids=zeros,
        entity_ids=zeros,
        scope_ids=zeros,
        source_doc_ids=zeros,
        confidence_ids=_arr([1, 1, 1, 1]),
        chunk_starts=_arr([0, 2]),
        chunk_ends=_arr([2, 4]),
        chunk_kinds=_arr([1, 1]),
        chunk_dep_levels=_arr([0, 0]),
        metadata={
            "token_change_mask_pre": [0, 0, 0, 0],
            "token_change_mask_post": [0, 0, 0, 0],
        },
    )
    realized = RealizedObjective(
        task=TaskKind.CAUSAL_LM,
        example=build_causal_lm(packet),
        ineligible={},
        source_index=0,
    )

    document = materialize_megatron_document(
        realized,
        ObjectiveSource(code_packet=packet),
    )

    assert document.row["doc_ids"] == [41, 41, 42, 42]
    assert np.asarray(realized.example.loss_mask).tolist() == [1, 0, 1]
    assert document.row["loss_mask"] == [1, 0, 1, 0]


def test_permuted_objective_preserves_one_stable_source_identity() -> None:
    packet = CodePacket(
        token_ids=_arr([10, 11, 12, 13, 14, 15]),
        document_ids=_arr([1, 1, 1, 1, 1, 1]),
        source_doc_ids=_arr([77, 77, 77, 77, 77, 77]),
        metadata={"platform_ids": [2]},
    )
    source = ObjectiveSource(code_packet=packet)
    realized = EligibilityAwareTaskMixer(
        {TaskKind.FIM: 1.0}, seed=47
    ).materialize(source, step_index=0)

    document = materialize_megatron_document(
        realized,
        source,
        require_production_sidecars=True,
    )

    assert set(document.row["token_source_doc_ids"]) == {77}
    assert set(document.row["doc_ids"]) == {1}


def test_permuted_objective_selects_exactly_one_logical_document() -> None:
    packet = CodePacket(
        token_ids=_arr([10, 11, 12, 13, 14, 15]),
        document_ids=_arr([1, 1, 1, 2, 2, 2]),
        source_doc_ids=_arr([77, 77, 77, 88, 88, 88]),
        metadata={"platform_ids": [2]},
    )
    source = ObjectiveSource(code_packet=packet)
    realized = EligibilityAwareTaskMixer(
        {TaskKind.FIM: 1.0}, seed=53
    ).materialize(source, step_index=0)

    document = materialize_megatron_document(
        realized,
        source,
        require_production_sidecars=True,
    )

    selected_span = realized.example.metadata["source_document_span"]
    selected_source = 77 if selected_span == (0, 3) else 88
    assert set(document.row["token_source_doc_ids"]) == {selected_source}
    selected_tokens = {10, 11, 12} if selected_source == 77 else {13, 14, 15}
    assert set(document.token_ids) & {10, 11, 12, 13, 14, 15} <= selected_tokens


def test_real_dsa_forward_and_backward_include_graph_auxiliary_loss() -> None:
    model = DenseCppLM(
        DenseCppLMConfig(
            vocab_size=128,
            hidden_size=32,
            depth=1,
            ffn_hidden_size=64,
            max_seq_length=8,
            num_query_heads=4,
            num_kv_heads=2,
            head_dim=8,
            attention_mode="dsa",
            attention_sparse_topk=2,
            indexer_heads=2,
            indexer_dim=4,
            indexer_local_window=0,
            indexer_num_sinks=0,
            require_graph_routes=True,
            ngram_hash_enabled=False,
            structure_residual_scale=1.0,
            platform_residual_scale=0.0,
        )
    )
    input_ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    targets = mx.array([[2, 3, 4, 5]], dtype=mx.int32)
    loss_mask = mx.ones((1, 4), dtype=mx.float32)
    graph_targets = mx.zeros((1, 4, 4), dtype=mx.float32)
    graph_targets = graph_targets.at[0, 2, 0].add(1.0)
    pair_mask = mx.ones((1, 4, 4), dtype=mx.float32)
    config = GraphAuxLossConfig(
        relations=("call",),
        topk=1,
        global_weight=0.5,
        indexer_weight=0.25,
        layer_weight=0.5,
        bce_weight=1.0,
        coverage_weight=0.25,
    )
    structure_sidecars = {
        name: mx.zeros_like(input_ids)
        for name in (
            "structure_ids",
            "dep_levels",
            "ast_depth_ids",
            "sibling_index_ids",
            "node_type_ids",
        )
    }

    total, lm_loss, graph_loss = production_training_loss(
        model,
        input_ids,
        targets,
        loss_mask,
        side_channels=structure_sidecars,
        block_bias=graph_targets,
        graph_targets=graph_targets,
        graph_pair_mask=pair_mask,
        graph_config=config,
        graph_weight=0.5,
    )
    mx.eval(total, lm_loss, graph_loss)

    assert float(graph_loss.item()) > 0.0
    assert abs(float(total.item()) - float((lm_loss + graph_loss).item())) < 1e-6
    unweighted_graph_loss = graph_auxiliary_loss_from_targets(
        model.indexer_scores(
            input_ids,
            block_bias=graph_targets,
            **structure_sidecars,
        ),
        graph_targets,
        pair_mask,
        GraphAuxLossConfig(
            relations=("call",),
            topk=1,
            global_weight=1.0,
            indexer_weight=1.0,
            layer_weight=1.0,
            bce_weight=1.0,
            coverage_weight=0.25,
        ),
    )
    mx.eval(unweighted_graph_loss)
    assert float(graph_loss.item()) == pytest.approx(
        0.5 * 0.25 * 0.5 * float(unweighted_graph_loss.item()),
        rel=1e-5,
    )

    loss_and_grad = nn.value_and_grad(
        model,
        lambda: production_training_loss(
            model,
            input_ids,
            targets,
            loss_mask,
            side_channels=structure_sidecars,
            block_bias=graph_targets,
            graph_targets=graph_targets,
            graph_pair_mask=pair_mask,
            graph_config=config,
            graph_weight=0.5,
        ),
    )
    (differentiated_loss, differentiated_lm, differentiated_graph), gradients = (
        loss_and_grad()
    )
    mx.eval(
        differentiated_loss,
        differentiated_lm,
        differentiated_graph,
        gradients,
    )
    assert mx.isfinite(differentiated_loss).item()
    assert float(differentiated_graph.item()) > 0.0


def test_production_loss_rejects_non_finite_graph_weight_before_forward() -> None:
    values = mx.array([[1, 2]], dtype=mx.int32)

    with pytest.raises(ValueError, match="finite"):
        production_training_loss(
            None,
            values,
            values,
            mx.ones_like(values),
            side_channels={},
            block_bias=None,
            graph_targets=None,
            graph_pair_mask=None,
            graph_config=None,
            graph_weight=float("nan"),
        )


@pytest.mark.parametrize(
    ("attention_mode", "require_graph_routes", "structure_scale", "message"),
    (
        ("gqa", True, 1.0, "active fail-closed DSA graph routes"),
        ("dsa", False, 1.0, "active fail-closed DSA graph routes"),
        ("dsa", True, 0.0, "active structure residual routing"),
    ),
)
def test_production_graph_loss_rejects_disabled_routes(
    attention_mode: str,
    require_graph_routes: bool,
    structure_scale: float,
    message: str,
) -> None:
    values = mx.array([[1, 2]], dtype=mx.int32)
    model = SimpleNamespace(
        config=SimpleNamespace(
            attention_mode=attention_mode,
            require_graph_routes=require_graph_routes,
            structure_residual_scale=structure_scale,
        )
    )

    with pytest.raises(ValueError, match=message):
        production_training_loss(
            model,
            values,
            values,
            mx.ones_like(values),
            side_channels={},
            block_bias=mx.zeros((1, 2, 2)),
            graph_targets=mx.zeros((1, 2, 2)),
            graph_pair_mask=mx.ones((1, 2, 2)),
            graph_config=GraphAuxLossConfig(relations=("call",)),
            graph_weight=1.0,
        )


def test_production_graph_loss_rejects_missing_structure_sidecars() -> None:
    values = mx.array([[1, 2]], dtype=mx.int32)
    model = SimpleNamespace(
        config=SimpleNamespace(
            attention_mode="dsa",
            require_graph_routes=True,
            structure_residual_scale=1.0,
        )
    )

    with pytest.raises(ValueError, match="missing required structure sidecars"):
        production_training_loss(
            model,
            values,
            values,
            mx.ones_like(values),
            side_channels={"structure_ids": values},
            block_bias=mx.zeros((1, 2, 2)),
            graph_targets=mx.zeros((1, 2, 2)),
            graph_pair_mask=mx.ones((1, 2, 2)),
            graph_config=GraphAuxLossConfig(relations=("call",)),
            graph_weight=1.0,
        )


def test_megatron_document_materializes_real_commit_diff_not_post_tokens() -> None:
    commit = CommitPacket(
        pre_token_ids=_arr([10, 11]),
        post_token_ids=_arr([20, 21]),
        diff_token_ids=_arr([30, 31, 32]),
        commit_msg=_arr([40, 41]),
    )
    mixer = EligibilityAwareTaskMixer({TaskKind.COMMIT_DIFF: 1.0}, seed=31)
    realized = mixer.materialize(commit, step_index=0)

    document = materialize_megatron_document(
        realized, ObjectiveSource(commit_packet=commit)
    )

    assert document.objective_kind == "commit_diff"
    assert document.token_ids[:8] == [17, 40, 41, 18, 15, 30, 31, 32]
    assert 20 not in document.token_ids and 21 not in document.token_ids
    assert document.loss_mask[-1] == 0
    assert sum(document.loss_mask[:-1]) == int(
        np.asarray(realized.example.loss_mask).sum()
    )


def test_megatron_document_rejects_non_shifted_targets() -> None:
    example = build_causal_lm(CodePacket(token_ids=_arr([10, 11, 12, 13])))
    tampered = type(example)(
        input_ids=example.input_ids,
        target_ids=_arr([99, 12, 13]),
        loss_mask=example.loss_mask,
        objective=example.objective,
        metadata=example.metadata,
    )
    realized = RealizedObjective(
        task=TaskKind.CAUSAL_LM,
        example=tampered,
        ineligible={},
        source_index=0,
    )

    with pytest.raises(ValueError, match="shifted-LM"):
        materialize_megatron_document(
            realized,
            ObjectiveSource(code_packet=CodePacket(token_ids=_arr([10, 11, 12, 13]))),
        )


def test_pre_materialized_contract_matches_exact_realized_schedule() -> None:
    n = 8
    zeros = _arr([0] * n)
    code = CodePacket(
        token_ids=_arr([100, 101, 102, 103, 104, 105, 106, 107]),
        document_ids=_arr([1] * n),
        ifim_instruction_token_ids=_arr([201, 202]),
        structure_ids=_arr([1] * n),
        dep_levels=zeros,
        ast_depth=zeros,
        sibling_index=zeros,
        ast_node_type=_arr([1] * n),
        symbol_ids=zeros,
        call_targets=zeros,
        type_refs=zeros,
        def_use=zeros,
        domain_ids=zeros,
        role_ids=zeros,
        entity_ids=zeros,
        scope_ids=zeros,
        source_doc_ids=_arr([1] * n),
        confidence_ids=_arr([1] * n),
        chunk_starts=_arr([0, 2, 6]),
        chunk_ends=_arr([2, 6, 8]),
        chunk_kinds=_arr([1, 1, 1]),
        chunk_dep_levels=_arr([0, 0, 0]),
        call_edges=EdgeIndex.from_pairs([(1, 0)], relation="call", num_nodes=3),
        metadata={
            "platform_ids": [2, 62],
            "token_change_mask_pre": [0] * n,
            "token_change_mask_post": [0] * n,
        },
    )
    commit = CommitPacket(
        pre_token_ids=_arr([10, 11]),
        post_token_ids=_arr([20, 21]),
        diff_token_ids=_arr([30, 31]),
        commit_msg=_arr([40, 41]),
    )
    source = ObjectiveSource(code_packet=code, commit_packet=commit)
    rates = {
        TaskKind.CAUSAL_LM: 1 / 6,
        TaskKind.FIM: 1 / 6,
        TaskKind.AST_FIM: 1 / 6,
        TaskKind.IFIM: 1 / 6,
        TaskKind.COMMIT_DIFF: 1 / 6,
        TaskKind.PRE_TO_POST: 1 / 6,
    }
    mixer = EligibilityAwareTaskMixer(rates, seed=37, max_input_tokens=32)
    realized = mixer.materialize_window([source] * 6)
    documents = [materialize_megatron_document(item, source) for item in realized]

    contract = build_pre_materialized_objective_contract(
        documents,
        rates=mixer.rates,
        seed=37,
        quota_window_samples=6,
        graph_config=GraphAuxLossConfig(
            relations=("call", "type"),
            topk=8,
            bce_weight=0.10,
            coverage_weight=0.05,
        ),
        graph_weight=1.0,
    )

    assert contract["schema"] == OBJECTIVE_CONTRACT_SCHEMA
    assert contract["planned_samples"] == {task.value: 1 for task in rates}
    assert contract["realized"] == {
        task.value: {
            "samples": 1,
            "input_tokens": sum(
                len(document.token_ids) - 1
                for document in documents
                if document.objective_kind == task.value
            ),
            "loss_tokens": sum(
                sum(document.loss_mask[:-1])
                for document in documents
                if document.objective_kind == task.value
            ),
        }
        for task in rates
    }
    assert contract["typed_sources"]["diff"] == "diff_token_ids"
    assert contract["typed_sources"]["rendered_text_parsing"] is False
    assert contract["objective_ids"] == {
        task.value: OBJECTIVE_KIND_IDS[task.value] for task in rates
    }
    assert contract["graph_auxiliary"]["included_in_total_loss"] is True
    assert contract["graph_auxiliary"]["global_weight"] == "1"
    assert contract["graph_auxiliary"]["indexer_weight"] == "1/1000"
    assert contract["graph_auxiliary"]["layer_weight"] == "1"
    assert contract["graph_auxiliary"]["layer_reduction"] == "sum"
    assert contract["graph_auxiliary"]["pair_mask"] == (
        "causal_same_document_upstream_v1"
    )
    assert contract["graph_auxiliary"]["chunk_edge_expansion"] == (
        "cartesian_token_spans_v1"
    )
    assert contract["graph_auxiliary"]["positive_edges"] == 8
    assert contract["materialization"] == {
        "format": "shifted_lm_document_v1",
        "token_column": "input_ids",
        "loss_mask_column": "loss_mask",
        "length_column": "valid_token_count",
        "objective_column": "objective_kind",
        "document_id_column": "doc_ids",
        "source_document_id_column": "token_source_doc_ids",
    }


def test_pre_materialized_contract_counts_the_configured_graph_relation() -> None:
    documents = [
        MaterializedMegatronDocument(
            objective_kind=task.value,
            token_ids=[1, 2, 3],
            loss_mask=[1, 1, 0],
            graph_edge_count=0,
            row={
                "token_call_edges": [],
                "token_type_edges": [],
                "token_domain_edges": (
                    [{"from": 1, "to": 0, "kind": 1}]
                    if task is TaskKind.CAUSAL_LM
                    else []
                ),
                "token_build_edges": [],
                "token_shell_edges": [],
                "token_diagnostic_edges": [],
                "token_cross_domain_edges": [],
                "doc_ids": [1, 1, 1],
                "token_chunk_starts": [],
                "token_chunk_ends": [],
            },
        )
        for task in (
            TaskKind.CAUSAL_LM,
            TaskKind.FIM,
            TaskKind.AST_FIM,
            TaskKind.IFIM,
            TaskKind.COMMIT_DIFF,
            TaskKind.PRE_TO_POST,
        )
    ]
    rates = {document.objective_kind: 1 / 6 for document in documents}

    contract = build_pre_materialized_objective_contract(
        documents,
        rates=rates,
        seed=41,
        quota_window_samples=6,
        graph_config=GraphAuxLossConfig(
            relations=("domain",),
            topk=2,
            bce_weight=0.1,
            coverage_weight=0.05,
        ),
        graph_weight=1.0,
    )

    assert contract["graph_auxiliary"]["relations"] == ["domain"]
    assert contract["graph_auxiliary"]["eligible_samples"] == 1
    assert contract["graph_auxiliary"]["positive_edges"] == 1


def test_pre_materialized_contract_rejects_zero_quota_objective() -> None:
    tasks = (
        TaskKind.CAUSAL_LM,
        TaskKind.FIM,
        TaskKind.AST_FIM,
        TaskKind.IFIM,
        TaskKind.COMMIT_DIFF,
        TaskKind.PRE_TO_POST,
    )
    documents = [
        MaterializedMegatronDocument(
            objective_kind=task.value,
            token_ids=[1, 2],
            loss_mask=[1, 0],
            graph_edge_count=1 if task is TaskKind.CAUSAL_LM else 0,
            row={
                "token_call_edges": (
                    [{"from": 0, "to": 0}] if task is TaskKind.CAUSAL_LM else []
                ),
                "token_type_edges": [],
            },
        )
        for task in tasks[:-1]
    ]

    with pytest.raises(ValueError, match="zero quota.*pre_to_post"):
        build_pre_materialized_objective_contract(
            documents,
            rates={task: 1 / 6 for task in tasks},
            seed=43,
            quota_window_samples=5,
            graph_config=GraphAuxLossConfig(
                relations=("call",),
                topk=2,
                bce_weight=0.1,
                coverage_weight=0.05,
            ),
            graph_weight=1.0,
        )


def test_materialized_document_serializes_with_production_arrow_schema() -> None:
    row: dict[str, object] = {
        "input_ids": [10, 11],
        "valid_token_count": 2,
        "objective_kind": "causal_lm",
        "loss_mask": [1, 0],
        "doc_ids": [1, 1],
        "source_platform_ids": [[2, 62]],
        "token_call_edges": [{"from": 0, "to": 0}],
        "token_type_edges": [],
        "token_domain_edges": [],
        "token_build_edges": [],
        "token_shell_edges": [],
        "token_diagnostic_edges": [],
        "token_cross_domain_edges": [],
        "token_chunk_starts": [0],
        "token_chunk_ends": [2],
        "token_chunk_kinds": [1],
        "token_chunk_dep_levels": [0],
    }
    for column in (
        "token_domain_ids",
        "token_role_ids",
        "token_entity_ids",
        "token_scope_ids",
        "token_source_doc_ids",
        "token_confidence_ids",
        "token_structure_ids",
        "token_dep_levels",
        "token_ast_depth",
        "token_sibling_index",
        "token_ast_node_type",
        "token_symbol_ids",
        "token_call_targets",
        "token_type_refs",
        "token_def_use",
        "token_change_mask_pre",
        "token_change_mask_post",
    ):
        row[column] = [0, 0]
    document = MaterializedMegatronDocument(
        objective_kind="causal_lm",
        token_ids=[10, 11],
        loss_mask=[1, 0],
        graph_edge_count=1,
        row=row,
    )

    table = pa.Table.from_pylist(
        [padded_row(document, capacity=4)], schema=materialized_schema()
    )

    assert table.num_rows == 1
    assert table["input_ids"][0].as_py() == [10, 11, 0, 0]
    assert table["loss_mask"][0].as_py() == [1, 0, 0, 0]
    assert table["objective_kind"][0].as_py() == "causal_lm"


def test_canonical_objective_artifact_binds_contract_shards_and_converter(
    tmp_path: Path,
) -> None:
    contract = {
        "schema": OBJECTIVE_CONTRACT_SCHEMA,
        "totals": {"samples": 2},
        "materialization": {
            "format": "shifted_lm_document_v1",
            "token_column": "input_ids",
            "loss_mask_column": "loss_mask",
            "length_column": "valid_token_count",
            "objective_column": "objective_kind",
            "document_id_column": "doc_ids",
            "source_document_id_column": "token_source_doc_ids",
        },
        "graph_auxiliary": {
            "relations": ["call", "type"],
            "pair_mask": "causal_same_document_upstream_v1",
            "chunk_edge_expansion": "cartesian_token_spans_v1",
        },
    }
    first = tmp_path / "objectives_00000.parquet"
    second = tmp_path / "objectives_00001.parquet"
    first.write_bytes(b"first shard")
    second.write_bytes(b"second shard")

    artifact_path = write_objective_materialization_artifact(
        tmp_path,
        contract=contract,
        parquet_paths=[second, first],
    )

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    canonical_contract = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    assert artifact["schema"] == OBJECTIVE_MATERIALIZATION_ARTIFACT_SCHEMA
    assert artifact["documents"] == 2
    assert artifact["objective_contract"]["path"] == "objective_contract.json"
    assert artifact["objective_contract"]["sha256"] == hashlib.sha256(
        canonical_contract
    ).hexdigest()
    contract_bytes = (tmp_path / "objective_contract.json").read_bytes()
    assert artifact["objective_contract"]["size_bytes"] == len(contract_bytes)
    assert artifact["objective_contract"]["file_sha256"] == hashlib.sha256(
        contract_bytes
    ).hexdigest()
    assert [row["path"] for row in artifact["parquet_shards"]] == [
        first.name,
        second.name,
    ]
    assert artifact["converter"]["side_channels"][1] == {
        "column": "doc_ids",
        "dtype": "uint16",
    }
    assert {
        "column": "token_source_doc_ids",
        "dtype": "uint32",
    } in artifact["converter"]["side_channels"]
    assert artifact["converter"]["graph_pair_mask"] == (
        "causal_same_document_upstream_v1"
    )
    assert artifact["converter"]["chunk_edge_expansion"] == (
        "cartesian_token_spans_v1"
    )
    artifact_set_payload = dict(artifact)
    artifact_set_sha256 = artifact_set_payload.pop("artifact_set_sha256")
    assert artifact_set_sha256 == hashlib.sha256(
        json.dumps(
            artifact_set_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    assert json.loads((tmp_path / "objective_contract.json").read_text()) == contract


def test_objective_order_and_source_ids_ignore_mapping_encounter_order() -> None:
    forward = {
        TaskKind.CAUSAL_LM: 0.5,
        TaskKind.FIM: 0.3,
        TaskKind.IFIM: 0.2,
    }
    reverse = dict(reversed(tuple(forward.items())))

    assert tuple(normalize_rates(forward)) == tuple(normalize_rates(reverse))
    assert EligibilityAwareTaskMixer(forward, seed=29).quotas(10) == (
        EligibilityAwareTaskMixer(reverse, seed=29).quotas(10)
    )
    assert OBJECTIVE_KIND_IDS == {
        "causal_lm": 1,
        "fim": 2,
        "ast_fim": 3,
        "ifim": 4,
        "commit_diff": 5,
        "pre_to_post": 6,
        "symbol_recovery": 7,
        "type_recovery": 8,
        "callee_recovery": 9,
    }
    source_signatures = ("repo-a\0path\0commit", "repo-b\0path\0commit")
    forward_ids = {
        signature: deterministic_source_id(signature)
        for signature in source_signatures
    }
    reverse_ids = {
        signature: deterministic_source_id(signature)
        for signature in reversed(source_signatures)
    }
    assert forward_ids == reverse_ids
    assert len(set(forward_ids.values())) == len(source_signatures)


def test_anonymous_tokenized_source_ids_ignore_encounter_order(tmp_path: Path) -> None:
    forward_dir = tmp_path / "forward"
    reverse_dir = tmp_path / "reverse"
    forward_dir.mkdir()
    reverse_dir.mkdir()
    token_rows = [[101, 102], [201, 202]]
    pq.write_table(
        pa.table({"token_ids": token_rows}),
        forward_dir / "rows.parquet",
    )
    pq.write_table(
        pa.table({"token_ids": list(reversed(token_rows))}),
        reverse_dir / "rows.parquet",
    )

    def identities(path: Path) -> dict[tuple[int, ...], int]:
        return {
            tuple(document.token_ids): document.stable_doc_id
            for document in read_tokenized_documents(path)
        }

    assert identities(forward_dir) == identities(reverse_dir)
    assert len(set(identities(forward_dir).values())) == len(token_rows)


def test_graph_supervision_rejects_missing_configured_relation_sidecar() -> None:
    packet = CodePacket(
        token_ids=_arr([10, 11, 12, 13]),
        document_ids=_arr([1, 1, 1, 1]),
        chunk_starts=_arr([0, 2]),
        chunk_ends=_arr([2, 4]),
        chunk_kinds=_arr([1, 1]),
        chunk_dep_levels=_arr([0, 0]),
        call_edges=EdgeIndex.from_pairs([], relation="call", num_nodes=2),
    )

    with pytest.raises(ValueError, match="missing required relation sidecars: type"):
        graph_targets_and_pair_mask(
            packet,
            input_length=4,
            relations=("call", "type"),
        )

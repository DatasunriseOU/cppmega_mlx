from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pytest
from mlx.utils import tree_flatten

from cppmega_mlx.data.batch import LMTokenBatch
from cppmega_mlx.data.domain_schema import DomainEdgeKind
from cppmega_mlx.data.graph_packet import EdgeIndex, GraphBatch, GraphPacket
from cppmega_mlx.models.dense_cpp_lm import DenseCppLM
from cppmega_mlx.training.compiled import CompiledPretrainingStep
from cppmega_mlx.training.objective_mixer import (
    graph_auxiliary_loss_breakdown_from_targets,
)
from cppmega_mlx.training.stage1_production import (
    Stage1ProductionObjective,
    _stage1_objective_batch,
    stage1_production_config,
)


def _model_config():
    return stage1_production_config(
        attention_mode="dsa",
        vocab_size=64,
        hidden_size=32,
        depth=1,
        ffn_hidden_size=64,
        max_seq_length=6,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=8,
        indexer_heads=2,
        indexer_dim=4,
        indexer_local_window=0,
        indexer_num_sinks=0,
        ngram_hash_enabled=False,
        platform_residual_scale=0.0,
    )


def _production_batch_rows(
    pairs_by_row: tuple[tuple[tuple[int, int], ...], ...],
    *,
    relation: str = "domain",
    chunk_spans: tuple[tuple[int, int], ...] | None = None,
    document_rows: tuple[tuple[int, ...], ...] | None = None,
    attention_rows: tuple[tuple[int, ...], ...] | None = None,
) -> LMTokenBatch:
    row_count = len(pairs_by_row)
    tokens = mx.broadcast_to(
        mx.array([[2, 3, 5, 7, 11, 13]], dtype=mx.int32),
        (row_count, 6),
    )
    spans = chunk_spans or tuple((index, index + 1) for index in range(6))
    starts = mx.array([start for start, _end in spans], dtype=mx.int32)
    ends = mx.array([end for _start, end in spans], dtype=mx.int32)
    graph = GraphBatch(
        graphs=tuple(
            GraphPacket(
                edges={
                    relation: EdgeIndex.from_pairs(
                        pairs,
                        relation=relation,
                        num_nodes=len(spans),
                    )
                },
                num_nodes=len(spans),
            )
            for pairs in pairs_by_row
        ),
        chunk_starts=tuple(starts for _ in pairs_by_row),
        chunk_ends=tuple(ends for _ in pairs_by_row),
        chunk_kinds=tuple(mx.ones(starts.shape, dtype=mx.int32) for _ in pairs_by_row),
        chunk_dep_levels=tuple(
            mx.zeros(starts.shape, dtype=mx.int32) for _ in pairs_by_row
        ),
        edge_kinds=tuple(
            {
                relation: mx.array(
                    [int(DomainEdgeKind.DIAG_PRIMARY_LOCATION)] * len(pairs),
                    dtype=mx.int32,
                )
            }
            for pairs in pairs_by_row
        ),
    )
    structure = mx.ones(tokens.shape, dtype=mx.int32)
    if document_rows is None:
        document_rows = tuple((0, 0, 0, 1, 1, 1) for _ in pairs_by_row)
    return LMTokenBatch(
        tokens=tokens,
        attention_mask=(
            None
            if attention_rows is None
            else mx.array(attention_rows, dtype=mx.float32)
        ),
        loss_mask=mx.ones(tokens.shape, dtype=mx.float32),
        document_ids=mx.array(document_rows, dtype=mx.int32),
        structure_ids=structure,
        dep_levels=structure,
        ast_depth_ids=structure,
        sibling_index_ids=structure,
        node_type_ids=structure,
        domain_ids=mx.full(tokens.shape, 3, dtype=mx.int32),
        role_ids=mx.full(tokens.shape, 2, dtype=mx.int32),
        confidence_ids=mx.ones(tokens.shape, dtype=mx.int32),
        graph_batch=graph,
    )


def _production_batch(*, with_edge: bool) -> LMTokenBatch:
    pairs = ((4, 3),) if with_edge else ()
    return _production_batch_rows((pairs,))


class _CountingDenseCppLM(DenseCppLM):
    def __init__(self) -> None:
        super().__init__(_model_config())
        self.decoder_invocations = 0

    def decoder_hidden_states(self, *args, **kwargs):
        self.decoder_invocations += 1
        return super().decoder_hidden_states(*args, **kwargs)


def _gradient_l1(gradients: object, *name_fragments: str) -> float:
    arrays = [
        value
        for name, value in tree_flatten(gradients)
        if isinstance(value, mx.array)
        and any(fragment in name for fragment in name_fragments)
    ]
    assert arrays, name_fragments
    values = [mx.sum(mx.abs(value.astype(mx.float32))) for value in arrays]
    mx.eval(*values)
    return sum(float(value.item()) for value in values)


def test_stage1_combined_loss_uses_one_forward_and_exact_decomposition() -> None:
    objective = Stage1ProductionObjective()
    batch = _production_batch(with_edge=True)
    model = _CountingDenseCppLM()

    objective.validate_batch(batch)
    breakdown = objective.loss_breakdown(model, batch)
    mx.eval(
        breakdown.total,
        breakdown.lm_ce,
        breakdown.graph_total,
        breakdown.graph_edge_bce,
        breakdown.graph_ranking,
        breakdown.graph_positive_pairs,
    )

    assert model.decoder_invocations == 1
    assert float(breakdown.graph_positive_pairs.item()) == 1.0
    assert float(breakdown.graph_total.item()) > 0.0
    assert float(breakdown.graph_edge_bce.item()) > 0.0
    assert float(breakdown.graph_ranking.item()) >= 0.0
    assert float(breakdown.graph_total.item()) == pytest.approx(
        float((breakdown.graph_edge_bce + breakdown.graph_ranking).item()),
        rel=1e-6,
    )
    assert float(breakdown.total.item()) == pytest.approx(
        float((breakdown.lm_ce + breakdown.graph_total).item()),
        rel=1e-6,
    )
    assert objective.receipt()["single_decoder_forward"] is True


def test_stage1_self_chunk_targets_are_intersected_with_causal_pair_mask() -> None:
    batch = _production_batch_rows(
        (((0, 0),),),
        relation="call",
        chunk_spans=((1, 4),),
        document_rows=((0, 0, 0, 0, 0, 0),),
    )

    values = _stage1_objective_batch(batch)
    mx.eval(values.graph_targets, values.graph_pair_mask)

    assert float(values.relation_bias[0, 1, 2].item()) > 0.0
    assert float(values.graph_pair_mask[0, 1, 2].item()) == 0.0
    assert float(values.graph_targets[0, 1, 2].item()) == 0.0
    assert float(values.graph_targets[0, 2, 1].item()) == 1.0


def test_stage1_graphless_row_is_excluded_from_pair_mask_and_loss() -> None:
    values = _stage1_objective_batch(_production_batch_rows((((4, 3),), ())))
    baseline_scores = mx.zeros_like(values.graph_targets)
    changed_graphless_scores = mx.concatenate(
        (
            baseline_scores[:1],
            mx.full(baseline_scores[1:].shape, 9.0, dtype=mx.float32),
        ),
        axis=0,
    )
    baseline = graph_auxiliary_loss_breakdown_from_targets(
        (baseline_scores,),
        values.graph_targets,
        values.graph_pair_mask,
        Stage1ProductionObjective().graph_config,
    )
    changed = graph_auxiliary_loss_breakdown_from_targets(
        (changed_graphless_scores,),
        values.graph_targets,
        values.graph_pair_mask,
        Stage1ProductionObjective().graph_config,
    )
    mx.eval(values.graph_pair_mask, baseline.total, changed.total)

    assert float(mx.sum(values.graph_pair_mask[1]).item()) == 0.0
    assert float(changed.total.item()) == float(baseline.total.item())


def test_stage1_padding_pairs_are_excluded_from_graph_loss() -> None:
    values = _stage1_objective_batch(
        _production_batch_rows(
            (((2, 1),),),
            document_rows=((1, 1, 1, 0, 0, 0),),
            attention_rows=((1, 1, 1, 0, 0, 0),),
        )
    )
    baseline_scores = mx.zeros_like(values.graph_targets)
    padding_scores = baseline_scores.at[:, 3:, :].add(11.0)
    padding_scores = padding_scores.at[:, :, 3:].add(11.0)
    baseline = graph_auxiliary_loss_breakdown_from_targets(
        (baseline_scores,),
        values.graph_targets,
        values.graph_pair_mask,
        Stage1ProductionObjective().graph_config,
    )
    changed = graph_auxiliary_loss_breakdown_from_targets(
        (padding_scores,),
        values.graph_targets,
        values.graph_pair_mask,
        Stage1ProductionObjective().graph_config,
    )
    mx.eval(values.graph_pair_mask, baseline.total, changed.total)

    assert float(mx.sum(values.graph_pair_mask[:, 3:, :]).item()) == 0.0
    assert float(mx.sum(values.graph_pair_mask[:, :, 3:]).item()) == 0.0
    assert float(changed.total.item()) == pytest.approx(float(baseline.total.item()))


def test_stage1_graphless_batch_is_lm_only_and_updates_optimizer() -> None:
    objective = Stage1ProductionObjective()
    batch = _production_batch(with_edge=False)
    model = DenseCppLM(_model_config())

    objective.validate_batch(batch)
    breakdown = objective.loss_breakdown(model, batch)
    mx.eval(
        breakdown.total,
        breakdown.lm_ce,
        breakdown.graph_total,
        breakdown.graph_edge_bce,
        breakdown.graph_ranking,
        breakdown.graph_positive_pairs,
    )

    assert float(breakdown.graph_total.item()) == 0.0
    assert float(breakdown.graph_edge_bce.item()) == 0.0
    assert float(breakdown.graph_ranking.item()) == 0.0
    assert float(breakdown.graph_positive_pairs.item()) == 0.0
    assert float(breakdown.total.item()) == float(breakdown.lm_ce.item())

    before = model.token_embedding.weight
    stepper = CompiledPretrainingStep(
        model,
        optim.SGD(learning_rate=1e-3),
        loss_fn=objective,
        compile=True,
    )
    metrics = stepper(batch)
    update_l1 = mx.sum(mx.abs(model.token_embedding.weight - before))
    mx.eval(update_l1)

    assert metrics.updated is True
    assert metrics.ntokens == 5
    assert float(update_l1.item()) > 0.0


def test_stage1_combined_loss_reaches_model_and_graph_indexer_parameters() -> None:
    objective = Stage1ProductionObjective()
    batch = _production_batch(with_edge=True)
    model = DenseCppLM(_model_config())
    objective.validate_batch(batch)

    loss_and_grad = nn.value_and_grad(
        model,
        lambda: objective.loss_breakdown(model, batch).total,
    )
    loss, gradients = loss_and_grad()
    mx.eval(loss, gradients)

    assert mx.isfinite(loss).item()
    assert _gradient_l1(gradients, "token_embedding", "ffn") > 0.0
    assert _gradient_l1(
        gradients,
        "index_q_proj",
        "index_k_proj",
        "index_head_weights",
    ) > 0.0

    step_model = DenseCppLM(_model_config())
    stepper = CompiledPretrainingStep(
        step_model,
        optim.SGD(learning_rate=1e-3),
        loss_fn=objective,
        compile=True,
    )
    metrics = stepper(batch)
    assert metrics.updated is True
    assert metrics.ntokens == 5
    assert metrics.loss > 0.0

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
    stage1_production_batch_receipt,
    stage1_production_config,
)


def _model_config(*, attention_mode: str | None = "dsa"):
    kwargs: dict[str, object] = dict(
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
    if attention_mode is not None:
        kwargs["attention_mode"] = attention_mode
    return stage1_production_config(**kwargs)


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
    document_ids = mx.array(document_rows, dtype=mx.int32)
    attention_mask = (
        mx.ones(tokens.shape, dtype=mx.float32)
        if attention_rows is None
        else mx.array(attention_rows, dtype=mx.float32)
    )
    valid = attention_mask.astype(mx.bool_)
    loss_mask = mx.concatenate(
        [
            (
                (document_ids[:, :-1] == document_ids[:, 1:])
                & valid[:, :-1]
                & valid[:, 1:]
            ).astype(mx.float32),
            valid[:, -1:].astype(mx.float32),
        ],
        axis=1,
    )
    return LMTokenBatch(
        tokens=tokens,
        attention_mask=None if attention_rows is None else attention_mask,
        loss_mask=loss_mask,
        document_ids=document_ids,
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


def _gradient_array(gradients: object, name_fragment: str) -> mx.array:
    arrays = [
        value
        for name, value in tree_flatten(gradients)
        if isinstance(value, mx.array) and name_fragment in name
    ]
    assert len(arrays) == 1, (name_fragment, len(arrays))
    return arrays[0]


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
    dsa_receipt = objective.receipt(attention_mode="dsa")
    assert dsa_receipt["single_decoder_forward"] is True
    assert dsa_receipt["graph_auxiliary_enabled"] is True
    assert dsa_receipt["route_only"] is False


def test_stage1_gqa_graph_route_changes_logits_loss_and_gradients() -> None:
    objective = Stage1ProductionObjective()
    graph_batch = _production_batch(with_edge=True)
    empty_batch = _production_batch(with_edge=False)
    mx.random.seed(211)
    model = DenseCppLM(_model_config(attention_mode="gqa"))
    graph_values = _stage1_objective_batch(graph_batch)
    empty_values = _stage1_objective_batch(empty_batch)

    graph_logits = model.logits(
        graph_values.input_ids,
        document_ids=graph_values.document_ids,
        block_bias=graph_values.relation_bias,
        edge_kind_bias=graph_values.edge_kind_bias,
        **graph_values.side_channels,
    )
    empty_logits = model.logits(
        empty_values.input_ids,
        document_ids=empty_values.document_ids,
        block_bias=empty_values.relation_bias,
        edge_kind_bias=empty_values.edge_kind_bias,
        **empty_values.side_channels,
    )
    graph_loss_and_grad = nn.value_and_grad(
        model,
        lambda: objective.loss_breakdown(model, graph_batch).total,
    )
    empty_loss_and_grad = nn.value_and_grad(
        model,
        lambda: objective.loss_breakdown(model, empty_batch).total,
    )
    graph_loss, graph_gradients = graph_loss_and_grad()
    empty_loss, empty_gradients = empty_loss_and_grad()
    graph_q_grad = _gradient_array(
        graph_gradients, "layers.0.attention.q_proj.weight"
    )
    empty_q_grad = _gradient_array(
        empty_gradients, "layers.0.attention.q_proj.weight"
    )
    mx.eval(
        graph_logits,
        empty_logits,
        graph_loss,
        empty_loss,
        graph_q_grad,
        empty_q_grad,
    )

    assert float(mx.sum(mx.abs(graph_logits - empty_logits)).item()) > 1e-5
    assert abs(float(graph_loss.item()) - float(empty_loss.item())) > 1e-6
    assert float(mx.sum(mx.abs(graph_q_grad - empty_q_grad)).item()) > 1e-6
    graph_breakdown = objective.loss_breakdown(model, graph_batch)
    mx.eval(graph_breakdown.graph_total, graph_breakdown.graph_positive_pairs)
    assert float(graph_breakdown.graph_total.item()) == 0.0
    assert float(graph_breakdown.graph_positive_pairs.item()) == 0.0
    assert graph_breakdown.graph_layer_count == 0


def test_stage1_default_gqa_receipt_separates_route_signal_from_graph_auxiliary() -> None:
    objective = Stage1ProductionObjective()
    batch = _production_batch(with_edge=True)
    model = DenseCppLM(_model_config(attention_mode=None))

    assert model.config.attention_mode == "gqa"
    breakdown = objective.loss_breakdown(model, batch)
    batch_receipt = stage1_production_batch_receipt(batch, config=model.config)
    loss_receipt = objective.receipt(attention_mode=model.config.attention_mode)
    mx.eval(
        breakdown.graph_total,
        breakdown.graph_positive_pairs,
        breakdown.graph_edge_bce,
        breakdown.graph_ranking,
    )

    assert float(breakdown.graph_total.item()) == 0.0
    assert float(breakdown.graph_positive_pairs.item()) == 0.0
    assert float(breakdown.graph_edge_bce.item()) == 0.0
    assert float(breakdown.graph_ranking.item()) == 0.0
    assert breakdown.graph_layer_count == 0
    assert batch_receipt["graph_route_positive_pairs"] == 1
    assert batch_receipt["graph_supervision_positive_pairs"] == 0
    assert "graph_positive_pairs" not in batch_receipt
    assert loss_receipt["name"] == "gqa_route_only_lm_ce"
    assert loss_receipt["formula"] == "lm_ce"
    assert loss_receipt["graph_auxiliary_enabled"] is False
    assert loss_receipt["graph_supervision"] == "none"
    assert loss_receipt["route_only"] is True


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
    assert metrics.ntokens == 4
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
    assert metrics.ntokens == 4
    assert metrics.loss > 0.0


def test_stage1_graph_loss_trains_neural_indexer_not_fixed_route_beta() -> None:
    objective = Stage1ProductionObjective()
    batch = _production_batch(with_edge=True)
    model = DenseCppLM(_model_config())
    objective.validate_batch(batch)

    loss_and_grad = nn.value_and_grad(
        model,
        lambda: objective.loss_breakdown(model, batch).graph_total,
    )
    graph_loss, gradients = loss_and_grad()
    mx.eval(graph_loss, gradients)

    assert float(graph_loss.item()) > 0.0
    assert _gradient_l1(
        gradients,
        "index_q_proj",
        "index_k_proj",
        "index_head_weights",
    ) > 0.0
    # ``index_beta`` is a fixed checkpoint-visible route coefficient. It is
    # deliberately absent from MLX's trainable gradient tree so AdamW cannot
    # move it through weight decay; older implementations exposed a zero
    # gradient entry instead.
    beta_gradients = [
        value
        for name, value in tree_flatten(gradients)
        if isinstance(value, mx.array) and "index_beta" in name
    ]
    assert not beta_gradients


def test_stage1_graph_loss_excludes_fixed_relation_and_kind_priors() -> None:
    objective = Stage1ProductionObjective()
    batch = _production_batch(with_edge=True)
    model = DenseCppLM(_model_config())
    attention = model.layers[0].attention
    attention.index_q_proj.weight = mx.zeros_like(attention.index_q_proj.weight)
    attention.index_k_proj.weight = mx.zeros_like(attention.index_k_proj.weight)
    attention.index_beta = mx.array([4.0], dtype=mx.float32)
    values = _stage1_objective_batch(batch)

    decoder_result = model.decoder_hidden_states(
        values.input_ids,
        document_ids=values.document_ids,
        block_bias=values.relation_bias,
        edge_kind_bias=values.edge_kind_bias,
        return_indexer_scores=True,
        **values.side_channels,
    )
    assert isinstance(decoder_result, tuple)
    _hidden_states, final_scores = decoder_result
    learned_scores = model.graph_supervision_scores(
        final_scores,
        input_ids=values.input_ids,
        document_ids=values.document_ids,
        block_bias=values.relation_bias,
        edge_kind_bias=values.edge_kind_bias,
    )
    raw_breakdown = graph_auxiliary_loss_breakdown_from_targets(
        final_scores,
        values.graph_targets,
        values.graph_pair_mask,
        objective.graph_config,
    )
    learned_breakdown = graph_auxiliary_loss_breakdown_from_targets(
        learned_scores,
        values.graph_targets,
        values.graph_pair_mask,
        objective.graph_config,
    )
    production_breakdown = objective.loss_breakdown(model, batch)
    mx.eval(
        final_scores,
        learned_scores,
        raw_breakdown.total,
        learned_breakdown.total,
        production_breakdown.graph_total,
    )

    eligible = values.graph_targets > 0
    route_bias = values.relation_bias + values.edge_kind_bias
    expected_delta = mx.where(eligible, 4.0 * route_bias, mx.zeros_like(route_bias))
    observed_delta = mx.where(
        eligible,
        final_scores[0] - learned_scores[0],
        mx.zeros_like(route_bias),
    )
    mx.eval(expected_delta, observed_delta)

    assert float(mx.sum(mx.abs(observed_delta - expected_delta)).item()) == pytest.approx(
        0.0,
        abs=1e-6,
    )
    assert float(raw_breakdown.total.item()) != pytest.approx(
        float(learned_breakdown.total.item()),
        abs=1e-6,
    )
    assert float(production_breakdown.graph_total.item()) == pytest.approx(
        float(learned_breakdown.total.item()),
        rel=1e-6,
    )

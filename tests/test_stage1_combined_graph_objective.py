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
from cppmega_mlx.training.stage1_production import (
    Stage1ProductionObjective,
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


def _production_batch(*, with_edge: bool) -> LMTokenBatch:
    tokens = mx.array([[2, 3, 5, 7, 11, 13]], dtype=mx.int32)
    pairs = [(4, 3)] if with_edge else []
    graph = GraphBatch(
        graphs=(
            GraphPacket(
                edges={
                    "domain": EdgeIndex.from_pairs(
                        pairs,
                        relation="domain",
                        num_nodes=6,
                    )
                },
                num_nodes=6,
            ),
        ),
        chunk_starts=(mx.arange(6, dtype=mx.int32),),
        chunk_ends=(mx.arange(1, 7, dtype=mx.int32),),
        chunk_kinds=(mx.ones((6,), dtype=mx.int32),),
        chunk_dep_levels=(mx.zeros((6,), dtype=mx.int32),),
        edge_kinds=(
            {
                "domain": mx.array(
                    [int(DomainEdgeKind.DIAG_PRIMARY_LOCATION)] if with_edge else [],
                    dtype=mx.int32,
                )
            },
        ),
    )
    structure = mx.ones(tokens.shape, dtype=mx.int32)
    return LMTokenBatch(
        tokens=tokens,
        loss_mask=mx.ones(tokens.shape, dtype=mx.float32),
        document_ids=mx.array([[0, 0, 0, 1, 1, 1]], dtype=mx.int32),
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


def test_stage1_combined_loss_rejects_a_no_edge_production_batch() -> None:
    objective = Stage1ProductionObjective()

    with pytest.raises(ValueError, match="nonzero graph targets"):
        objective.validate_batch(_production_batch(with_edge=False))


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

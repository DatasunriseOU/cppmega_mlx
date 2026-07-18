from __future__ import annotations

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pytest
from mlx.utils import tree_flatten

from cppmega_mlx.data.graph_packet import EdgeIndex, GraphBatch, GraphPacket
from cppmega_mlx.models.dense_cpp_lm import DenseCppLM, DenseCppLMConfig
from cppmega_mlx.nn.code_graph_routes import (
    apply_graph_route_prior,
    build_token_graph_biases,
    remove_graph_route_prior,
)
from cppmega_mlx.training.indexer_losses import (
    apply_graph_indexer_bias,
    indexer_edge_bce_loss,
    remove_graph_indexer_bias,
)


def _config(*, beta: float = 4.0, topk: int = 1) -> DenseCppLMConfig:
    return DenseCppLMConfig(
        vocab_size=32,
        hidden_size=16,
        depth=1,
        ffn_hidden_size=32,
        max_seq_length=8,
        num_query_heads=2,
        num_kv_heads=1,
        head_dim=8,
        attention_mode="dsa",
        attention_sparse_topk=topk,
        indexer_heads=1,
        indexer_dim=4,
        indexer_local_window=0,
        indexer_num_sinks=0,
        require_graph_routes=True,
        graph_routes_enabled=True,
        graph_attention_bias_beta=beta,
        rope=False,
        ngram_hash_enabled=False,
        structure_residual_scale=0.0,
        platform_residual_scale=0.0,
    )


def _graph(seq_length: int, pairs: list[list[int]]) -> GraphBatch:
    edge = EdgeIndex.from_pairs(pairs, relation="call", num_nodes=seq_length)
    starts = mx.arange(seq_length, dtype=mx.int32)
    return GraphBatch(
        graphs=(
            GraphPacket(
                edges={"call": edge},
                num_nodes=seq_length,
            ),
        ),
        chunk_starts=(starts,),
        chunk_ends=(starts + 1,),
    )


def _ids() -> mx.array:
    return mx.arange(8, dtype=mx.int32)[None, :]


def _gradient_l1(gradients: object, fragment: str) -> float:
    values = [
        value
        for name, value in tree_flatten(gradients)
        if isinstance(value, mx.array) and fragment in name
    ]
    assert len(values) == 1, (fragment, len(values))
    result = mx.sum(mx.abs(values[0].astype(mx.float32)))
    mx.eval(result)
    return float(result.item())


def test_graph_batch_changes_dsa_indexer_and_selected_attention_output() -> None:
    mx.random.seed(7)
    model = DenseCppLM(_config(beta=100.0, topk=1))
    ids = _ids()
    empty = _graph(8, [])

    empty_scores = model.indexer_scores(ids, graph_batch=empty)[0]
    mx.eval(empty_scores)
    empty_row = np.asarray(empty_scores)[0, 7]
    empty_winner = int(np.argmax(empty_row))
    graph_target = int(np.argmin(empty_row))
    if graph_target == empty_winner:
        graph_target = (empty_winner + 1) % empty_row.shape[0]
    graph = _graph(8, [[7, graph_target]])

    graph_scores = model.indexer_scores(ids, graph_batch=graph)[0]
    empty_logits = model.logits(ids, graph_batch=empty)
    graph_logits = model.logits(ids, graph_batch=graph)
    mx.eval(graph_scores, empty_scores, graph_logits, empty_logits)

    graph_row = np.asarray(graph_scores)[0, 7]
    assert graph_row[graph_target] - empty_row[graph_target] == pytest.approx(100.0)
    assert int(np.argmax(graph_row)) == graph_target
    assert empty_winner != graph_target
    assert float(mx.sum(mx.abs(graph_logits - empty_logits)).item()) > 1e-5


def test_production_dsa_fails_closed_without_typed_graph_or_explicit_bias() -> None:
    model = DenseCppLM(_config())
    ids = _ids()

    with pytest.raises(RuntimeError, match="no typed GraphBatch or explicit block_bias"):
        model.logits(ids)
    with pytest.raises(RuntimeError, match="no typed GraphBatch or explicit block_bias"):
        model.indexer_scores(ids)


def test_graph_supervision_round_trip_uses_neural_score_once() -> None:
    model = DenseCppLM(_config(beta=4.0, topk=2))
    ids = _ids()
    graph = _graph(8, [[7, 1]])
    empty = _graph(8, [])

    routed = model.indexer_scores(ids, graph_batch=graph)[0]
    neural_from_model = model.graph_supervision_scores(
        (routed,),
        input_ids=ids,
        graph_batch=graph,
    )[0]
    neural_from_empty = model.indexer_scores(ids, graph_batch=empty)[0]
    relation_bias, kind_bias = build_token_graph_biases(
        graph,
        batch_size=1,
        seq_length=8,
    )
    route_bias = relation_bias + kind_bias
    direct_neural = remove_graph_route_prior(routed, route_bias, beta=4.0)
    direct_routed = apply_graph_route_prior(neural_from_empty, route_bias, beta=4.0)
    loss_neural = remove_graph_indexer_bias(routed, route_bias, beta=4.0)
    loss_routed = apply_graph_indexer_bias(
        neural_from_empty,
        route_bias,
        beta=4.0,
    )
    mx.eval(
        neural_from_model,
        neural_from_empty,
        direct_neural,
        direct_routed,
        loss_neural,
        loss_routed,
    )

    np.testing.assert_allclose(
        np.asarray(neural_from_model),
        np.asarray(neural_from_empty),
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(neural_from_model),
        np.asarray(direct_neural),
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(routed),
        np.asarray(direct_routed),
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(neural_from_model),
        np.asarray(loss_neural),
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(routed),
        np.asarray(loss_routed),
        rtol=0.0,
        atol=1e-6,
    )


def test_graph_supervised_loss_is_invariant_to_fixed_route_beta() -> None:
    mx.random.seed(9)
    model = DenseCppLM(_config(beta=1.0, topk=2))
    ids = _ids()
    graph = _graph(8, [[7, 1]])
    targets = mx.zeros((1, 8, 8), dtype=mx.float32).at[0, 7, 1].add(1.0)

    def graph_loss(beta: float) -> tuple[mx.array, mx.array]:
        model.layers[0].attention.index_beta = mx.array(
            [beta],
            dtype=mx.float32,
        )
        routed = model.indexer_scores(ids, graph_batch=graph)[0]
        neural = model.graph_supervision_scores(
            (routed,),
            input_ids=ids,
            graph_batch=graph,
        )[0]
        return indexer_edge_bce_loss(neural, targets), routed

    loss_one, routed_one = graph_loss(1.0)
    loss_eight, routed_eight = graph_loss(8.0)
    mx.eval(loss_one, loss_eight, routed_one, routed_eight)

    assert float(loss_eight.item()) == pytest.approx(
        float(loss_one.item()),
        rel=0.0,
        abs=1e-6,
    )
    assert float((routed_eight - routed_one)[0, 7, 1].item()) == pytest.approx(
        7.0
    )


def test_graph_loss_has_individual_neural_indexer_gradients() -> None:
    mx.random.seed(11)
    model = DenseCppLM(_config(beta=4.0, topk=2))
    ids = _ids()
    graph = _graph(8, [[7, 1]])
    targets = mx.zeros((1, 8, 8), dtype=mx.float32).at[0, 7, 1].add(1.0)

    def loss_fn(current: DenseCppLM) -> mx.array:
        routed = current.indexer_scores(ids, graph_batch=graph)[0]
        neural = current.graph_supervision_scores(
            (routed,),
            input_ids=ids,
            graph_batch=graph,
        )[0]
        return indexer_edge_bce_loss(neural, targets)

    loss, gradients = nn.value_and_grad(model, loss_fn)(model)
    mx.eval(loss, gradients)
    assert np.isfinite(float(loss.item()))
    for fragment in (
        "index_q_proj",
        "index_k_proj",
        "index_head_weights",
    ):
        assert _gradient_l1(gradients, fragment) > 0.0
    assert not any(
        isinstance(value, mx.array) and "index_beta" in name
        for name, value in tree_flatten(gradients)
    )


def test_fixed_route_beta_does_not_move_under_adamw_decay() -> None:
    mx.random.seed(13)
    model = DenseCppLM(_config(beta=4.0, topk=2))
    ids = _ids()
    graph = _graph(8, [[7, 1]])
    before = model.layers[0].attention.index_beta

    def loss_fn(current: DenseCppLM) -> mx.array:
        routed = current.indexer_scores(ids, graph_batch=graph)[0]
        neural = current.graph_supervision_scores(
            (routed,),
            input_ids=ids,
            graph_batch=graph,
        )[0]
        return mx.sum(neural * neural)

    loss, gradients = nn.value_and_grad(model, loss_fn)(model)
    optimizer = optim.AdamW(learning_rate=1e-2, weight_decay=0.1)
    optimizer.update(model, gradients)
    mx.eval(loss, model.parameters(), optimizer.state)
    after = model.layers[0].attention.index_beta

    np.testing.assert_array_equal(np.asarray(before), np.asarray(after))

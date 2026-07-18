from __future__ import annotations

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
import pytest
from mlx.utils import tree_flatten

from cppmega_mlx.data.batch import LMTokenBatch
from cppmega_mlx.data.domain_schema import DomainEdgeKind
from cppmega_mlx.data.graph_packet import EdgeIndex, GraphBatch, GraphPacket
from cppmega_mlx.models.dense_cpp_lm import DenseCppLM, DenseCppLMConfig
from cppmega_mlx.training.compiled import (
    CompiledPretrainingStep,
    STABLE_BATCH_KEYS,
    normalize_compiled_batch,
)


def _config(**overrides: object) -> DenseCppLMConfig:
    values: dict[str, object] = {
        "vocab_size": 64,
        "hidden_size": 32,
        "depth": 1,
        "ffn_hidden_size": 64,
        "max_seq_length": 8,
        "num_query_heads": 4,
        "num_kv_heads": 2,
        "head_dim": 8,
        "ngram_hash_enabled": False,
        "structure_residual_scale": 0.0,
        "platform_residual_scale": 0.0,
    }
    values.update(overrides)
    return DenseCppLMConfig(**values)


def _graph(
    edge: tuple[int, int],
    *,
    seq_length: int = 8,
    relation: str = "call",
    edge_kind: DomainEdgeKind | None = None,
) -> GraphBatch:
    edge_kinds = ()
    if edge_kind is not None:
        edge_kinds = ({relation: mx.array([int(edge_kind)], dtype=mx.int32)},)
    return GraphBatch(
        graphs=(
            GraphPacket(
                edges={
                    relation: EdgeIndex.from_pairs(
                        [edge], relation=relation, num_nodes=seq_length
                    )
                },
                num_nodes=seq_length,
            ),
        ),
        chunk_starts=(mx.arange(seq_length, dtype=mx.int32),),
        chunk_ends=(mx.arange(1, seq_length + 1, dtype=mx.int32),),
        edge_kinds=edge_kinds,
    )


def _batch(
    *,
    graph: GraphBatch | None = None,
    domain_id: int | None = None,
    document_ids: mx.array | None = None,
) -> LMTokenBatch:
    tokens = mx.array([[3, 5, 7, 11, 13, 17, 19, 23]], dtype=mx.int32)
    targets = mx.array([[5, 7, 11, 13, 17, 19, 23, 29]], dtype=mx.int32)
    side_channels = None
    if domain_id is not None:
        side_channels = {
            "domain_routes": {
                "domain_ids": mx.full(tokens.shape, domain_id, dtype=mx.int32),
                "role_ids": mx.full(tokens.shape, 2, dtype=mx.int32),
                "confidence_ids": mx.full(tokens.shape, 1, dtype=mx.int32),
            }
        }
    return LMTokenBatch(
        tokens=tokens,
        target_tokens=targets,
        loss_mask=mx.ones(tokens.shape, dtype=mx.float32),
        document_ids=document_ids,
        graph_batch=graph,
        side_channels=side_channels,
    )


def _parameters(model: DenseCppLM) -> dict[str, np.ndarray]:
    mx.eval(model.parameters())
    return {
        name: np.asarray(value).copy()
        for name, value in tree_flatten(model.parameters())
        if isinstance(value, mx.array)
    }


def _run(
    cfg: DenseCppLMConfig,
    batch: LMTokenBatch,
    *,
    compile: bool,
    seed: int = 41,
    initialize_domain: bool = False,
) -> tuple[float, dict[str, np.ndarray], dict[str, np.ndarray], DenseCppLM]:
    mx.random.seed(seed)
    model = DenseCppLM(cfg)
    if initialize_domain:
        assert model.domain_embedding is not None
        embedding = model.domain_embedding
        embedding.stacked_emb.weight = mx.random.normal(
            embedding.stacked_emb.weight.shape
        )
        embedding.up_proj.weight = mx.random.normal(embedding.up_proj.weight.shape)
    optimizer = optim.SGD(learning_rate=1e-2)
    mx.eval(model.state, optimizer.state)
    before = _parameters(model)
    metrics = CompiledPretrainingStep(model, optimizer, compile=compile)(batch)
    return metrics.loss, before, _parameters(model), model


def _max_update(
    before: dict[str, np.ndarray], after: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    return {name: after[name] - value for name, value in before.items()}


def _indexer_scores(model: DenseCppLM, batch: LMTokenBatch) -> tuple[mx.array, ...]:
    normalized = normalize_compiled_batch(batch, graph_routes_enabled=True)
    return model.indexer_scores(
        batch.inputs,
        block_bias=normalized["graph_attention_bias"],
        edge_kind_bias=normalized["graph_edge_kind_bias"],
        document_ids=batch.input_document_ids,
    )


def test_normalize_materializes_array_only_graph_and_flattens_domain_routes() -> None:
    normalized = normalize_compiled_batch(
        _batch(graph=_graph((7, 1)), domain_id=3),
        graph_routes_enabled=True,
    )

    assert tuple(normalized) == STABLE_BATCH_KEYS
    assert "graph_batch" not in normalized
    assert "side_channels" not in normalized
    assert tuple(normalized["graph_attention_bias"].shape) == (1, 8, 8)
    assert normalized["graph_attention_bias"].dtype == mx.float32
    assert float(mx.sum(normalized["graph_attention_bias"]).item()) == 1.0
    assert tuple(normalized["graph_edge_kind_bias"].shape) == (1, 8, 8)
    assert normalized["graph_edge_kind_bias"].dtype == mx.float32
    assert float(mx.sum(normalized["graph_edge_kind_bias"]).item()) == 0.0
    for key in ("domain_ids", "role_ids", "confidence_ids"):
        assert isinstance(normalized[key], mx.array)
        assert tuple(normalized[key].shape) == (1, 8)


def test_compiled_gqa_graph_changes_loss_and_parameter_gradients() -> None:
    cfg = _config(graph_routes_enabled=True, graph_attention_bias_beta=25.0)

    left_loss, left_before, left_after, _ = _run(
        cfg, _batch(graph=_graph((7, 1))), compile=True
    )
    right_loss, right_before, right_after, _ = _run(
        cfg, _batch(graph=_graph((7, 2))), compile=True
    )

    assert abs(left_loss - right_loss) > 1e-6
    left_update = _max_update(left_before, left_after)
    right_update = _max_update(right_before, right_after)
    assert any(
        not np.array_equal(left_update[name], right_update[name])
        for name in left_update
    )


def test_compiled_dsa_graph_changes_indexer_scores_and_training_update() -> None:
    cfg = _config(
        attention_mode="dsa",
        attention_sparse_topk=4,
        indexer_local_window=0,
        indexer_num_sinks=0,
        graph_routes_enabled=True,
        graph_attention_bias_beta=1.0,
    )

    _, left_before, left_after, left_model = _run(
        cfg, _batch(graph=_graph((7, 1))), compile=True
    )
    _, right_before, right_after, right_model = _run(
        cfg, _batch(graph=_graph((7, 2))), compile=True
    )
    left_scores = _indexer_scores(left_model, _batch(graph=_graph((7, 1))))[0]
    right_scores = _indexer_scores(right_model, _batch(graph=_graph((7, 2))))[0]
    mx.eval(left_scores, right_scores)

    assert float(left_scores[0, 7, 1].item() - right_scores[0, 7, 1].item()) == pytest.approx(1.0)
    left_update = _max_update(left_before, left_after)
    right_update = _max_update(right_before, right_after)
    assert any(
        not np.array_equal(left_update[name], right_update[name])
        for name in left_update
    )


def test_compiled_edge_kind_changes_gqa_loss_and_parameter_gradients() -> None:
    cfg = _config(graph_routes_enabled=True, graph_attention_bias_beta=10.0)
    edge = (7, 1)
    build = _batch(
        graph=_graph(
            edge,
            relation="domain",
            edge_kind=DomainEdgeKind.BUILD_TARGET_DEP,
        )
    )
    diagnostic = _batch(
        graph=_graph(
            edge,
            relation="domain",
            edge_kind=DomainEdgeKind.DIAG_PRIMARY_LOCATION,
        )
    )

    build_loss, build_before, build_after, _ = _run(cfg, build, compile=True)
    diag_loss, diag_before, diag_after, _ = _run(cfg, diagnostic, compile=True)

    assert abs(build_loss - diag_loss) > 1e-6
    build_update = _max_update(build_before, build_after)
    diag_update = _max_update(diag_before, diag_after)
    assert any(
        not np.array_equal(build_update[name], diag_update[name])
        for name in build_update
    )


def test_compiled_edge_kind_changes_dsa_indexer_scores() -> None:
    cfg = _config(
        attention_mode="dsa",
        attention_sparse_topk=4,
        indexer_local_window=0,
        indexer_num_sinks=0,
        graph_routes_enabled=True,
    )
    edge = (7, 1)
    build = _batch(
        graph=_graph(
            edge,
            relation="domain",
            edge_kind=DomainEdgeKind.BUILD_TARGET_DEP,
        )
    )
    diagnostic = _batch(
        graph=_graph(
            edge,
            relation="domain",
            edge_kind=DomainEdgeKind.DIAG_PRIMARY_LOCATION,
        )
    )

    _, _, _, build_model = _run(cfg, build, compile=True)
    _, _, _, diag_model = _run(cfg, diagnostic, compile=True)
    build_scores = _indexer_scores(build_model, build)[0]
    diag_scores = _indexer_scores(diag_model, diagnostic)[0]
    mx.eval(build_scores, diag_scores)

    assert float(
        diag_scores[0, 7, 1].item() - build_scores[0, 7, 1].item()
    ) == pytest.approx(1.0, abs=2e-4)


def test_compiled_domain_channels_change_loss_and_gradients() -> None:
    cfg = _config(domain_residual_scale=0.5)

    left_loss, left_before, left_after, _ = _run(
        cfg,
        _batch(domain_id=1),
        compile=True,
        initialize_domain=True,
    )
    right_loss, right_before, right_after, _ = _run(
        cfg,
        _batch(domain_id=2),
        compile=True,
        initialize_domain=True,
    )

    assert abs(left_loss - right_loss) > 1e-6
    left_update = _max_update(left_before, left_after)
    right_update = _max_update(right_before, right_after)
    assert any(
        name.startswith("domain_embedding.")
        and not np.array_equal(left_update[name], right_update[name])
        for name in left_update
    )


def test_graph_enabled_eager_and_compiled_step_are_numerically_aligned() -> None:
    cfg = _config(graph_routes_enabled=True, graph_attention_bias_beta=7.0)
    batch = _batch(graph=_graph((7, 1)))

    eager_loss, eager_before, eager_after, _ = _run(cfg, batch, compile=False)
    compiled_loss, compiled_before, compiled_after, _ = _run(cfg, batch, compile=True)

    assert compiled_loss == pytest.approx(eager_loss, rel=1e-5, abs=1e-6)
    eager_update = _max_update(eager_before, eager_after)
    compiled_update = _max_update(compiled_before, compiled_after)
    for name in eager_update:
        np.testing.assert_allclose(
            compiled_update[name], eager_update[name], rtol=1e-5, atol=1e-6
        )


@pytest.mark.parametrize("compile", [False, True])
def test_dsa_training_runs_two_optimizer_steps_without_mutating_module_tree(
    compile: bool,
) -> None:
    cfg = _config(
        attention_mode="dsa",
        attention_sparse_topk=4,
        indexer_local_window=0,
        indexer_num_sinks=0,
        graph_routes_enabled=True,
    )
    model = DenseCppLM(cfg)
    optimizer = optim.SGD(learning_rate=1e-2)
    step = CompiledPretrainingStep(model, optimizer, compile=compile)
    batch = _batch(graph=_graph((7, 1)))
    state_keys_before = tuple(name for name, _value in tree_flatten(model.state))

    first = step(batch)
    second = step(batch)
    state_keys_after = tuple(name for name, _value in tree_flatten(model.state))

    assert np.isfinite(first.loss)
    assert np.isfinite(second.loss)
    assert first.updated is True and second.updated is True
    assert second.step == 2
    assert state_keys_after == state_keys_before
    assert all("last_index_scores" not in name for name in state_keys_after)


def test_graph_disabled_compiled_step_rejects_graph_payload_before_trace() -> None:
    cfg = _config(graph_routes_enabled=False)
    model = DenseCppLM(cfg)
    step = CompiledPretrainingStep(
        model, optim.SGD(learning_rate=1e-2), compile=True
    )

    with pytest.raises(ValueError, match="graph data.*routes are disabled"):
        step(_batch(graph=_graph((7, 1))))

    assert step._compiled_step is None


def test_malformed_graph_endpoint_fails_before_compile_trace() -> None:
    cfg = _config(graph_routes_enabled=True)
    model = DenseCppLM(cfg)
    step = CompiledPretrainingStep(
        model, optim.SGD(learning_rate=1e-2), compile=True
    )

    with pytest.raises(ValueError, match="out of range|exceed"):
        step(_batch(graph=_graph((8, 1))))

    assert step._compiled_step is None


def test_cross_document_graph_route_fails_before_compile_trace() -> None:
    cfg = _config(graph_routes_enabled=True)
    model = DenseCppLM(cfg)
    step = CompiledPretrainingStep(
        model, optim.SGD(learning_rate=1e-2), compile=True
    )
    document_ids = mx.array([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=mx.int32)

    with pytest.raises(ValueError, match="crosses document boundary"):
        step(
            _batch(
                graph=_graph((7, 1)),
                document_ids=document_ids,
            )
        )

    assert step._compiled_step is None


def test_compiled_graph_contract_rejects_invalid_edge_kind_before_trace() -> None:
    graph = GraphBatch(
        graphs=(
            GraphPacket(
                edges={
                    "domain": EdgeIndex.from_pairs(
                        [(7, 1)], relation="domain", num_nodes=8
                    )
                },
                num_nodes=8,
            ),
        ),
        chunk_starts=(mx.arange(8, dtype=mx.int32),),
        chunk_ends=(mx.arange(1, 9, dtype=mx.int32),),
        edge_kinds=({"domain": mx.array([999], dtype=mx.int32)},),
    )
    cfg = _config(graph_routes_enabled=True)
    step = CompiledPretrainingStep(
        DenseCppLM(cfg), optim.SGD(learning_rate=1e-2), compile=True
    )

    with pytest.raises(ValueError, match="unsupported edge kind"):
        step(_batch(graph=graph))

    assert step._compiled_step is None


def test_invalid_compiled_domain_id_fails_before_compile_trace() -> None:
    cfg = _config(domain_residual_scale=0.5)
    model = DenseCppLM(cfg)
    step = CompiledPretrainingStep(
        model, optim.SGD(learning_rate=1e-2), compile=True
    )

    with pytest.raises(ValueError, match="domain_ids out of range"):
        step(_batch(domain_id=cfg.domain_num_domains))

    assert step._compiled_step is None


def test_compiled_graph_contract_rejects_dtype_change() -> None:
    cfg = _config(graph_routes_enabled=True)
    model = DenseCppLM(cfg)
    step = CompiledPretrainingStep(
        model, optim.SGD(learning_rate=1e-2), compile=True
    )
    good = normalize_compiled_batch(
        _batch(graph=_graph((7, 1))), graph_routes_enabled=True
    )
    step(good)
    bad = dict(good)
    bad["graph_attention_bias"] = good["graph_attention_bias"].astype(mx.float16)

    with pytest.raises(ValueError, match="dtype|shape/dtype/field"):
        step(bad)

    bad_kind = dict(good)
    bad_kind["graph_edge_kind_bias"] = good["graph_edge_kind_bias"].astype(mx.float16)
    with pytest.raises(ValueError, match="dtype|shape/dtype/field"):
        step(bad_kind)


@pytest.mark.parametrize("compile", [False, True])
@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    [
        (
            "graph_attention_bias",
            lambda value: value.astype(mx.float16),
            "dtype must be float32",
        ),
        (
            "graph_edge_kind_bias",
            lambda value: mx.full(value.shape, float("nan"), dtype=mx.float32),
            "non-finite",
        ),
    ],
)
def test_eager_and_compiled_steps_reject_invalid_fixed_graph_biases(
    compile: bool,
    field: str,
    replacement,
    error: str,
) -> None:
    cfg = _config(graph_routes_enabled=True)
    normalized = normalize_compiled_batch(
        _batch(graph=_graph((7, 1))),
        graph_routes_enabled=True,
    )
    bad = dict(normalized)
    bad[field] = replacement(normalized[field])
    step = CompiledPretrainingStep(
        DenseCppLM(cfg),
        optim.SGD(learning_rate=1e-2),
        compile=compile,
    )

    with pytest.raises(ValueError, match=error):
        step(bad)

    assert step.state.step == 0
    assert step._compiled_step is None

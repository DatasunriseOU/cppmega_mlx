from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest
from mlx.utils import tree_flatten

from cppmega_mlx.config.model import (
    DSAConfig,
    M2RNNConfig,
    Mamba3Config,
    Nam56RModelConfig,
)
from cppmega_mlx.data.batch import synthetic_token_batch
from cppmega_mlx.models.hybrid_lm import HybridTinyConfig, HybridTinyLM
from cppmega_mlx.recipes.nam56r import build_hybrid_tiny_config_from_nam56r
from cppmega_mlx.training.compiled import CompiledPretrainingStep
from cppmega_mlx.training.loss import (
    next_token_cross_entropy,
    next_token_cross_entropy_with_mtp,
)


def _hybrid_config(**overrides) -> HybridTinyConfig:
    params = {
        "vocab_size": 32,
        "hidden_size": 16,
        "pattern": "AEMR",
        "depth": 4,
        "dsa_a_layer_ranks": (0,),
        "num_attention_heads": 4,
        "max_seq_length": 8,
        "structure_vocab_size": 16,
    }
    params.update(overrides)
    return HybridTinyConfig(**params)


def _single_route_config(symbol: str) -> HybridTinyConfig:
    return _hybrid_config(
        pattern=symbol,
        depth=1,
        dsa_a_layer_ranks=(0,) if symbol == "A" else (),
        max_seq_length=7,
    )


def _flat_tree(tree) -> dict[str, np.ndarray]:
    mx.eval(tree)
    return {name: np.array(value) for name, value in tree_flatten(tree)}


def _assert_finite_nonzero(tree: dict[str, np.ndarray], name: str) -> None:
    assert name in tree
    assert np.isfinite(tree[name]).all(), name
    assert _max_abs(tree, name) > 0, name


def _max_abs(tree: dict[str, np.ndarray], name: str) -> float:
    return float(np.max(np.abs(tree[name])))


def _assert_adamw_state_for(
    optimizer_state: dict[str, np.ndarray],
    param_name: str,
) -> None:
    for suffix in (".m", ".v"):
        state_name = f"{param_name}{suffix}"
        assert state_name in optimizer_state
        assert np.isfinite(optimizer_state[state_name]).all(), state_name
        assert _max_abs(optimizer_state, state_name) > 0, state_name


@pytest.mark.parametrize(
    ("symbol", "backend", "param_name"),
    [
        ("A", "attention", "layers.0.block.out_proj.weight"),
        ("E", "moe", "layers.0.block.router.gate.weight"),
        ("M", "mamba3", "layers.0.block.in_proj.weight"),
        ("R", "m2rnn", "layers.0.block.state_weight"),
    ],
)
def test_hybrid_lm_single_route_compiled_eager_train_matrix_with_side_channels(
    symbol: str,
    backend: str,
    param_name: str,
) -> None:
    batch = synthetic_token_batch(
        batch_size=1,
        seq_length=5,
        vocab_size=_single_route_config(symbol).vocab_size,
        seed=171,
        include_structure=True,
    )
    assert {
        "structure_ids",
        "dep_levels",
        "ast_depth_ids",
        "sibling_index_ids",
        "node_type_ids",
    } <= set(batch.as_dict())

    def run_step(
        *, compile: bool
    ) -> tuple[float, int, np.ndarray, dict[str, np.ndarray]]:
        mx.random.seed(173)
        model = HybridTinyLM(_single_route_config(symbol))
        optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=0.0)
        before = _flat_tree(model.parameters())
        metrics = CompiledPretrainingStep(model, optimizer, compile=compile)(
            batch.as_dict()
        )
        after = _flat_tree(model.parameters())
        optimizer_state = _flat_tree(optimizer.state)
        delta = after[param_name] - before[param_name]

        assert metrics.compiled is compile
        assert metrics.updated is True
        assert metrics.step == 1
        assert metrics.ntokens == metrics.trained_tokens == 4
        assert math.isfinite(metrics.loss)
        assert metrics.loss > 0
        assert model.route_symbols == (symbol,)
        assert [layer.backend for layer in model.layers] == [backend]
        assert _max_abs({param_name: delta}, param_name) > 0, param_name
        _assert_adamw_state_for(optimizer_state, param_name)
        return metrics.loss, metrics.ntokens, delta, optimizer_state

    eager_loss, eager_ntokens, eager_delta, _ = run_step(compile=False)
    compiled_loss, compiled_ntokens, compiled_delta, _ = run_step(compile=True)

    assert eager_ntokens == compiled_ntokens == 4
    assert math.isclose(compiled_loss, eager_loss, rel_tol=1e-5, abs_tol=1e-6)
    np.testing.assert_allclose(compiled_delta, eager_delta, rtol=1e-4, atol=1e-7)


def test_hybrid_lm_single_route_losses_reach_active_route_gradients() -> None:
    # The sibling cppmega NAM layout treats A/E/M/R as distinct active layer
    # routes. A single-route LM should therefore backprop through that route's
    # own parameters, not only through embeddings or the LM head.
    cases = {
        "A": ("attention", "layers.0.block.out_proj.weight"),
        "M": ("mamba3", "layers.0.block.in_proj.weight"),
        "E": ("moe", "layers.0.block.router.gate.weight"),
        "R": ("m2rnn", "layers.0.block.state_weight"),
    }

    for offset, (symbol, (backend, grad_name)) in enumerate(cases.items()):
        mx.random.seed(101 + offset)
        model = HybridTinyLM(_single_route_config(symbol))
        batch = synthetic_token_batch(
            batch_size=2,
            seq_length=6,
            vocab_size=model.config.vocab_size,
            seed=201 + offset,
            include_structure=True,
        )
        loss_and_grad = nn.value_and_grad(model, next_token_cross_entropy)

        (loss, ntokens), grads = loss_and_grad(model, batch)
        mx.eval(loss, ntokens, grads)

        assert model.route_symbols == (symbol,)
        assert [layer.backend for layer in model.layers] == [backend]
        assert math.isfinite(float(loss.item()))
        assert float(loss.item()) > 0
        assert int(ntokens.item()) == 10

        flat_grads = _flat_tree(grads)
        _assert_finite_nonzero(flat_grads, grad_name)


def test_hybrid_lm_single_route_train_steps_update_route_specific_sentinels() -> None:
    cases = {
        "A": ("attention", "layers.0.block.out_proj.weight"),
        "M": ("mamba3", "layers.0.block.in_proj.weight"),
        "E": ("moe", "layers.0.block.router.gate.weight"),
        "R": ("m2rnn", "layers.0.block.state_weight"),
    }

    for offset, (symbol, (backend, param_name)) in enumerate(cases.items()):
        mx.random.seed(151 + offset)
        model = HybridTinyLM(_single_route_config(symbol))
        optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=0.0)
        batch = synthetic_token_batch(
            batch_size=2,
            seq_length=6,
            vocab_size=model.config.vocab_size,
            seed=251 + offset,
            include_structure=True,
        )
        before = _flat_tree(model.parameters())

        loss_and_grad = nn.value_and_grad(model, next_token_cross_entropy)
        (loss, ntokens), grads = loss_and_grad(model, batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss, ntokens)
        flat_grads = _flat_tree(grads)
        after = _flat_tree(model.parameters())
        optimizer_state = _flat_tree(optimizer.state)

        assert model.route_symbols == (symbol,)
        assert [layer.backend for layer in model.layers] == [backend]
        assert math.isfinite(float(loss.item()))
        assert float(loss.item()) > 0
        assert int(ntokens.item()) == 10
        _assert_finite_nonzero(flat_grads, param_name)
        assert (
            _max_abs({param_name: after[param_name] - before[param_name]}, param_name)
            > 0
        )
        _assert_adamw_state_for(optimizer_state, param_name)


def test_hybrid_lm_skips_causal_mask_allocation_for_non_attention_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, mx.Dtype]] = []
    original = nn.MultiHeadAttention.create_additive_causal_mask

    def recording_mask(seq_length: int, *, dtype: mx.Dtype):
        calls.append((seq_length, dtype))
        return original(seq_length, dtype=dtype)

    monkeypatch.setattr(
        nn.MultiHeadAttention,
        "create_additive_causal_mask",
        recording_mask,
    )

    input_ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    for symbol in ("M", "R", "MR"):
        model = HybridTinyLM(
            _hybrid_config(
                pattern=symbol,
                depth=len(symbol),
                dsa_a_layer_ranks=(),
                hidden_size=8,
                num_attention_heads=1,
                max_seq_length=8,
                mamba_expand=1,
                mamba_head_dim=4,
                mamba_state_dim=4,
                mamba_groups=1,
                mamba_chunk_size=4,
                m2rnn_k_head_dim=2,
                m2rnn_v_head_dim=2,
                m2rnn_num_v_heads=1,
                m2rnn_num_f_heads=1,
                m2rnn_num_weight_heads=1,
                m2rnn_chunk_size=4,
            )
        )
        out = model(input_ids)
        mx.eval(out)

    assert calls == []

    attention_model = HybridTinyLM(_single_route_config("A"))
    attention_out = attention_model(input_ids)
    mx.eval(attention_out)

    assert calls == [(4, mx.float32)]


def test_hybrid_lm_dsa_path_c_uses_sparse_causal_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cppmega_mlx.nn._tilelang.sparse_mla_fp8_path_c as fp8_path_c

    mask_calls: list[tuple[int, mx.Dtype]] = []
    apply_calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    original = nn.MultiHeadAttention.create_additive_causal_mask

    def recording_mask(seq_length: int, *, dtype: mx.Dtype):
        mask_calls.append((seq_length, dtype))
        return original(seq_length, dtype=dtype)

    def fake_apply(
        q_fp8: mx.array,
        q_scale: mx.array,
        kv_fp8: mx.array,
        kv_scale: mx.array,
        indices: mx.array,
        *,
        sm_scale: float,
        d_v: int | None = None,
        sinks: mx.array | None = None,
        return_lse: bool = False,
        force_path_c: bool = False,
        output_dtype: mx.Dtype | None = None,
    ) -> mx.array:
        assert sinks is None
        del q_scale, kv_scale, indices, sm_scale, return_lse, output_dtype
        assert force_path_c is True
        d_v_resolved = int(q_fp8.shape[-1] if d_v is None else d_v)
        apply_calls.append((tuple(q_fp8.shape), tuple(kv_fp8.shape)))
        return mx.zeros(
            (q_fp8.shape[0], q_fp8.shape[1], q_fp8.shape[2], d_v_resolved),
            dtype=mx.float16,
        )

    monkeypatch.setenv("CPPMEGA_KERNEL_PATH__SPARSE_MLA", "path_c")
    monkeypatch.setattr(
        nn.MultiHeadAttention,
        "create_additive_causal_mask",
        recording_mask,
    )
    monkeypatch.setattr(fp8_path_c, "sparse_mla_fp8_path_c_apply", fake_apply)

    model = HybridTinyLM(
        _hybrid_config(
            pattern="A",
            depth=1,
            dsa_a_layer_ranks=(0,),
            hidden_size=16,
            num_attention_heads=4,
            num_attention_kv_heads=2,
            attention_sparse_topk=2,
            max_seq_length=8,
        )
    )
    out = model(mx.array([[1, 2, 3, 4]], dtype=mx.int32))
    mx.eval(out)

    assert out.shape == (1, 4, model.config.vocab_size)
    assert mask_calls == []
    assert apply_calls == [((1, 4, 4, 4), (1, 4, 2, 4))]


def test_hybrid_lm_dsa_path_c_threads_document_mask_to_sparse_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cppmega_mlx.nn._tilelang.sparse_mla_fp8_path_c as fp8_path_c

    seen_indices: list[mx.array] = []

    def fake_apply(
        q_fp8: mx.array,
        q_scale: mx.array,
        kv_fp8: mx.array,
        kv_scale: mx.array,
        indices: mx.array,
        *,
        sm_scale: float,
        d_v: int | None = None,
        sinks: mx.array | None = None,
        return_lse: bool = False,
        force_path_c: bool = False,
        output_dtype: mx.Dtype | None = None,
    ) -> mx.array:
        del q_scale, kv_fp8, kv_scale, sm_scale, sinks, return_lse, output_dtype
        assert force_path_c is True
        seen_indices.append(indices)
        d_v_resolved = int(q_fp8.shape[-1] if d_v is None else d_v)
        return mx.zeros(
            (q_fp8.shape[0], q_fp8.shape[1], q_fp8.shape[2], d_v_resolved),
            dtype=mx.float16,
        )

    monkeypatch.setenv("CPPMEGA_KERNEL_PATH__SPARSE_MLA", "path_c")
    monkeypatch.setattr(fp8_path_c, "sparse_mla_fp8_path_c_apply", fake_apply)

    model = HybridTinyLM(
        _hybrid_config(
            pattern="A",
            depth=1,
            dsa_a_layer_ranks=(0,),
            hidden_size=16,
            num_attention_heads=4,
            num_attention_kv_heads=2,
            attention_sparse_topk=2,
            max_seq_length=8,
        )
    )
    input_ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    document_ids = mx.array([[0, 0, 1, 1]], dtype=mx.int32)

    out = model(input_ids, document_ids=document_ids)
    mx.eval(out, seen_indices[0])

    assert out.shape == (1, 4, model.config.vocab_size)
    np.testing.assert_array_equal(
        np.sort(np.array(seen_indices[0][0, :, 0, :]), axis=-1),
        np.array([[-1, 0], [0, 1], [-1, 2], [2, 3]], dtype=np.int32),
    )


def test_hybrid_lm_document_ids_mask_cross_document_attention() -> None:
    mx.random.seed(461)
    model = HybridTinyLM(_single_route_config("A"))
    prefix_a = mx.array([[1, 2, 3]], dtype=mx.int32)
    prefix_b = mx.array([[7, 8, 9]], dtype=mx.int32)
    suffix = mx.array([[4, 5]], dtype=mx.int32)
    document_ids = mx.array([[0, 0, 0, 1, 1]], dtype=mx.int32)

    out_a = model(mx.concatenate([prefix_a, suffix], axis=1), document_ids=document_ids)
    out_b = model(mx.concatenate([prefix_b, suffix], axis=1), document_ids=document_ids)
    mx.eval(out_a, out_b)

    np.testing.assert_allclose(
        np.array(out_a[:, 3:, :]),
        np.array(out_b[:, 3:, :]),
        rtol=1e-5,
        atol=1e-5,
    )
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            np.array(out_a[:, 3:, :]),
            np.array(model(mx.concatenate([prefix_b, suffix], axis=1))[:, 3:, :]),
            rtol=1e-5,
            atol=1e-5,
        )


def test_hybrid_lm_document_ids_fail_closed_on_shape_and_negative_values() -> None:
    model = HybridTinyLM(_single_route_config("A"))
    input_ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)

    with pytest.raises(ValueError, match="document_ids.*match input_ids"):
        model(input_ids, document_ids=mx.array([[0, 0, 0]], dtype=mx.int32))
    with pytest.raises(ValueError, match="document_ids.*non-negative"):
        model(input_ids, document_ids=mx.array([[0, 0, -1, 1]], dtype=mx.int32))


def test_hybrid_lm_document_ids_fail_closed_even_without_attention_routes() -> None:
    model = HybridTinyLM(
        _hybrid_config(
            pattern="M",
            depth=1,
            dsa_a_layer_ranks=(),
            hidden_size=8,
            num_attention_heads=1,
            max_seq_length=8,
            mamba_expand=1,
            mamba_head_dim=4,
            mamba_state_dim=4,
            mamba_groups=1,
            mamba_chunk_size=4,
        )
    )
    input_ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)

    with pytest.raises(ValueError, match="document_ids.*match input_ids"):
        model(input_ids, document_ids=mx.array([[0, 0, 0]], dtype=mx.int32))
    with pytest.raises(ValueError, match="document_ids.*non-negative"):
        model(input_ids, document_ids=mx.array([[0, 0, -1, 1]], dtype=mx.int32))


def test_next_token_loss_accepts_document_id_aliases_and_rejects_conflicts() -> None:
    mx.random.seed(463)
    model = HybridTinyLM(_single_route_config("A"))
    tokens = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
    document_ids = mx.array([[0, 0, 0, 1, 1]], dtype=mx.int32)

    losses: list[float] = []
    for alias in ("document_ids", "doc_ids", "packing_document_ids"):
        loss, ntokens = next_token_cross_entropy(
            model,
            {
                "tokens": tokens,
                alias: document_ids,
            },
        )
        mx.eval(loss, ntokens)
        assert int(ntokens.item()) == 4
        losses.append(float(loss.item()))

    assert losses[0] == pytest.approx(losses[1], rel=0, abs=0)
    assert losses[0] == pytest.approx(losses[2], rel=0, abs=0)

    with pytest.raises(ValueError, match="only one document-id alias"):
        next_token_cross_entropy(
            model,
            {
                "tokens": tokens,
                "document_ids": document_ids,
                "doc_ids": document_ids,
            },
        )
    with pytest.raises(ValueError, match="doc_ids.*non-negative"):
        next_token_cross_entropy(
            model,
            {
                "tokens": tokens,
                "doc_ids": mx.array([[0, 0, -1, 1, 1]], dtype=mx.int32),
            },
        )


def test_mtp_loss_threads_document_ids_through_decoder_hidden_states() -> None:
    mx.random.seed(467)
    model = HybridTinyLM(_single_route_config("A"))
    tokens = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
    document_ids = mx.array([[0, 0, 0, 1, 1]], dtype=mx.int32)

    total_loss, ntokens, metrics = next_token_cross_entropy_with_mtp(
        model,
        {
            "tokens": tokens,
            "packing_document_ids": document_ids,
        },
    )
    mx.eval(total_loss, ntokens, metrics.mtp_loss)

    assert int(ntokens.item()) == 4
    assert math.isfinite(float(total_loss.item()))
    assert math.isfinite(float(metrics.mtp_loss.item()))
    with pytest.raises(ValueError, match="packing_document_ids.*non-negative"):
        next_token_cross_entropy_with_mtp(
            model,
            {
                "tokens": tokens,
                "packing_document_ids": mx.array([[0, 0, 0, -1, 1]], dtype=mx.int32),
            },
        )


def test_hybrid_lm_r_route_updates_m2rnn_recurrence_parameters() -> None:
    mx.random.seed(419)
    model = HybridTinyLM(
        _hybrid_config(
            pattern="R",
            depth=1,
            dsa_a_layer_ranks=(),
            hidden_size=8,
            num_attention_heads=1,
            max_seq_length=8,
            m2rnn_k_head_dim=2,
            m2rnn_v_head_dim=2,
            m2rnn_num_v_heads=1,
            m2rnn_num_f_heads=1,
            m2rnn_num_weight_heads=1,
            m2rnn_chunk_size=3,
        )
    )
    optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=0.0)
    batch = synthetic_token_batch(
        batch_size=2,
        seq_length=7,
        vocab_size=model.config.vocab_size,
        seed=421,
        include_structure=True,
    )
    expected_updates = (
        "layers.0.block.in_proj.weight",
        "layers.0.block.g_norm.weight",
        "layers.0.block.out_proj.weight",
        "layers.0.block.state_weight",
        "layers.0.block.A_log",
        "layers.0.block.dt_bias",
        "layers.0.block.D",
    )
    before = _flat_tree(model.parameters())

    loss_and_grad = nn.value_and_grad(model, next_token_cross_entropy)
    (loss, ntokens), grads = loss_and_grad(model, batch)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state, loss, ntokens)
    flat_grads = _flat_tree(grads)
    after = _flat_tree(model.parameters())

    assert model.route_symbols == ("R",)
    assert [layer.backend for layer in model.layers] == ["m2rnn"]
    assert math.isfinite(float(loss.item()))
    assert float(loss.item()) > 0
    assert int(ntokens.item()) == 12
    for name in expected_updates:
        assert name in flat_grads
        assert np.isfinite(flat_grads[name]).all(), name
        assert _max_abs(flat_grads, name) > 0, name
        assert _max_abs({name: after[name] - before[name]}, name) > 0, name


def test_hybrid_lm_m_route_updates_mamba3_scan_parameters() -> None:
    mx.random.seed(431)
    model = HybridTinyLM(
        _hybrid_config(
            pattern="M",
            depth=1,
            dsa_a_layer_ranks=(),
            hidden_size=8,
            num_attention_heads=1,
            max_seq_length=8,
            mamba_expand=1,
            mamba_head_dim=4,
            mamba_state_dim=4,
            mamba_groups=1,
            mamba_mimo_rank=1,
            mamba_is_mimo=False,
            mamba_conv_kernel=3,
            mamba_chunk_size=4,
        )
    )
    optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=0.0)
    batch = synthetic_token_batch(
        batch_size=2,
        seq_length=7,
        vocab_size=model.config.vocab_size,
        seed=433,
        include_structure=True,
    )
    expected_updates = (
        "layers.0.block.in_proj.weight",
        "layers.0.block.out_proj.weight",
        "layers.0.block.conv_weight",
        "layers.0.block.conv_bias",
        "layers.0.block.dt_bias",
        "layers.0.block.B_norm_weight",
        "layers.0.block.C_norm_weight",
        "layers.0.block.B_bias",
        "layers.0.block.C_bias",
        "layers.0.block.D",
    )
    before = _flat_tree(model.parameters())

    loss_and_grad = nn.value_and_grad(model, next_token_cross_entropy)
    (loss, ntokens), grads = loss_and_grad(model, batch)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state, loss, ntokens)
    flat_grads = _flat_tree(grads)
    after = _flat_tree(model.parameters())
    optimizer_state = _flat_tree(optimizer.state)

    assert model.route_symbols == ("M",)
    assert [layer.backend for layer in model.layers] == ["mamba3"]
    assert math.isfinite(float(loss.item()))
    assert float(loss.item()) > 0
    assert int(ntokens.item()) == 12
    for name in expected_updates:
        assert name in flat_grads
        assert np.isfinite(flat_grads[name]).all(), name
        assert _max_abs(flat_grads, name) > 0, name
        assert _max_abs({name: after[name] - before[name]}, name) > 0, name
        _assert_adamw_state_for(optimizer_state, name)


def test_hybrid_lm_loss_differentiates_through_all_route_backends() -> None:
    mx.random.seed(7)
    model = HybridTinyLM(_hybrid_config())
    batch = synthetic_token_batch(
        batch_size=2,
        seq_length=7,
        vocab_size=model.config.vocab_size,
        seed=19,
        include_structure=True,
    )
    loss_and_grad = nn.value_and_grad(model, next_token_cross_entropy)

    (loss, ntokens), grads = loss_and_grad(model, batch)
    mx.eval(loss, ntokens, grads)

    assert [layer.backend for layer in model.layers] == [
        "attention",
        "moe",
        "mamba3",
        "m2rnn",
    ]
    assert math.isfinite(float(loss.item()))
    assert int(ntokens.item()) == 12

    flat_grads = _flat_tree(grads)
    for name, grad in flat_grads.items():
        assert np.isfinite(grad).all(), name

    # Representative non-zero gradients prove the LM loss reaches each route.
    assert _max_abs(flat_grads, "layers.0.block.out_proj.weight") > 0
    assert _max_abs(flat_grads, "layers.1.block.router.gate.weight") > 0
    assert _max_abs(flat_grads, "layers.1.block.experts.0.down_proj.weight") > 0
    assert _max_abs(flat_grads, "layers.1.block.shared_expert.down_proj.weight") > 0
    assert _max_abs(flat_grads, "layers.2.block.in_proj.weight") > 0
    assert _max_abs(flat_grads, "layers.3.block.state_weight") > 0


def test_nam56r_lite_recipe_instantiates_custom_routes_and_backpropagates() -> None:
    mx.random.seed(113)
    source_config = Nam56RModelConfig(
        pattern="AEMR",
        depth=4,
        hidden_size=8,
        num_attention_heads=1,
        seq_len=8,
        max_position_embeddings=8,
        dsa=DSAConfig(a_layer_ranks=(0,)),
        mamba3=Mamba3Config(
            d_model=8,
            state_dim=4,
            expand=1,
            head_dim=4,
            num_groups=1,
            is_mimo=False,
            mimo_rank=1,
            chunk_size=4,
        ),
        m2rnn=M2RNNConfig(
            d_model=8,
            k_head_dim=2,
            v_head_dim=2,
            runtime_bwd_chunk_size=4,
        ),
    )
    config = build_hybrid_tiny_config_from_nam56r(
        source_config,
        vocab_size=32,
        dsa_a_layer_ranks=(0,),
        moe_num_experts=4,
        moe_top_k=2,
        moe_expert_hidden_size=16,
        moe_shared_expert_hidden_size=8,
        m2rnn_num_v_heads=1,
        m2rnn_num_f_heads=1,
    )
    model = HybridTinyLM(config)
    batch = synthetic_token_batch(
        batch_size=2,
        seq_length=6,
        vocab_size=model.config.vocab_size,
        seed=313,
        include_structure=True,
    )
    loss_and_grad = nn.value_and_grad(model, next_token_cross_entropy)

    (loss, ntokens), grads = loss_and_grad(model, batch)
    mx.eval(loss, ntokens, grads)

    assert model.route_symbols == ("A", "E", "M", "R")
    assert [layer.backend for layer in model.layers] == [
        "attention",
        "moe",
        "mamba3",
        "m2rnn",
    ]
    assert math.isfinite(float(loss.item()))
    assert int(ntokens.item()) == 10

    flat_grads = _flat_tree(grads)
    assert _max_abs(flat_grads, "layers.2.block.in_proj.weight") > 0
    assert _max_abs(flat_grads, "layers.2.block.out_proj.weight") > 0
    assert _max_abs(flat_grads, "layers.3.block.in_proj.weight") > 0
    assert _max_abs(flat_grads, "layers.3.block.state_weight") > 0


def test_hybrid_lm_structure_embedding_receives_gradients_when_projection_enabled() -> (
    None
):
    mx.random.seed(17)
    model = HybridTinyLM(
        _hybrid_config(
            pattern="A",
            depth=1,
            dsa_a_layer_ranks=(0,),
            structure_components="all",
            structure_num_categories=16,
        )
    )
    model.structure_embedding.up_proj.weight = mx.ones_like(
        model.structure_embedding.up_proj.weight
    )
    batch = synthetic_token_batch(
        batch_size=2,
        seq_length=6,
        vocab_size=model.config.vocab_size,
        seed=31,
        include_structure=True,
    )
    loss_and_grad = nn.value_and_grad(model, next_token_cross_entropy)

    (loss, ntokens), grads = loss_and_grad(model, batch)
    mx.eval(loss, ntokens, grads)

    flat_grads = _flat_tree(grads)
    assert math.isfinite(float(loss.item()))
    assert int(ntokens.item()) == 10
    assert "structure_embedding.stacked_emb.weight" in flat_grads
    assert np.isfinite(flat_grads["structure_embedding.stacked_emb.weight"]).all()
    assert _max_abs(flat_grads, "structure_embedding.stacked_emb.weight") > 0


def test_hybrid_lm_optimizer_step_updates_representative_route_params() -> None:
    mx.random.seed(11)
    model = HybridTinyLM(_hybrid_config(moe_top_k=4))
    moe = model.layers[1].moe_block
    assert moe is not None
    optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=0.0)
    batch = synthetic_token_batch(
        batch_size=2,
        seq_length=7,
        vocab_size=model.config.vocab_size,
        seed=23,
        include_structure=True,
    )
    representative = [
        "layers.0.block.out_proj.weight",
        "layers.1.block.router.gate.weight",
        "layers.1.block.shared_expert.gate_proj.weight",
        "layers.1.block.shared_expert.up_proj.weight",
        "layers.1.block.shared_expert.down_proj.weight",
        "layers.2.block.in_proj.weight",
        "layers.3.block.state_weight",
    ]
    for expert_id in range(moe.config.num_experts):
        representative.extend(
            [
                f"layers.1.block.experts.{expert_id}.gate_proj.weight",
                f"layers.1.block.experts.{expert_id}.up_proj.weight",
                f"layers.1.block.experts.{expert_id}.down_proj.weight",
            ]
        )
    before = _flat_tree(model.parameters())

    loss_and_grad = nn.value_and_grad(model, next_token_cross_entropy)
    (loss, ntokens), grads = loss_and_grad(model, batch)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state, loss, ntokens)
    flat_grads = _flat_tree(grads)
    after = _flat_tree(model.parameters())
    optimizer_state = _flat_tree(optimizer.state)

    assert math.isfinite(float(loss.item()))
    assert int(ntokens.item()) == 12
    for name in representative:
        assert _max_abs(flat_grads, name) > 0, name
        assert _max_abs({name: after[name] - before[name]}, name) > 0, name
        if (
            ".block.router." in name
            or ".block.experts." in name
            or ".block.shared_expert." in name
        ):
            _assert_adamw_state_for(optimizer_state, name)


def test_hybrid_lm_aemr_compiled_and_eager_train_steps_match_and_update_routes() -> (
    None
):
    config = _hybrid_config(
        hidden_size=8,
        num_attention_heads=1,
        max_seq_length=8,
        moe_num_experts=4,
        moe_top_k=4,
        moe_expert_hidden_size=16,
        moe_shared_expert_hidden_size=8,
        mamba_expand=1,
        mamba_head_dim=4,
        mamba_state_dim=4,
        mamba_groups=1,
        mamba_chunk_size=4,
        m2rnn_k_head_dim=2,
        m2rnn_v_head_dim=2,
        m2rnn_num_v_heads=1,
        m2rnn_num_f_heads=1,
        m2rnn_num_weight_heads=1,
        m2rnn_chunk_size=4,
    )
    batch = synthetic_token_batch(
        batch_size=1,
        seq_length=7,
        vocab_size=config.vocab_size,
        seed=601,
        include_structure=True,
    )
    representative = (
        "layers.0.block.out_proj.weight",
        "layers.1.block.router.gate.weight",
        "layers.1.block.shared_expert.down_proj.weight",
        "layers.2.block.in_proj.weight",
        "layers.3.block.state_weight",
    )

    def run_step(
        *,
        compile: bool,
    ) -> tuple[float, int, dict[str, np.ndarray], dict[str, np.ndarray]]:
        mx.random.seed(607)
        model = HybridTinyLM(config)
        optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=0.0)
        before = _flat_tree(model.parameters())
        stepper = CompiledPretrainingStep(model, optimizer, compile=compile)
        metrics = stepper(batch.as_dict())
        after = _flat_tree(model.parameters())

        assert metrics.compiled is compile
        assert metrics.updated is True
        assert metrics.step == 1
        assert metrics.trained_tokens == metrics.ntokens == 6
        assert math.isfinite(metrics.loss)
        assert metrics.loss > 0
        assert model.route_symbols == ("A", "E", "M", "R")
        assert [layer.backend for layer in model.layers] == [
            "attention",
            "moe",
            "mamba3",
            "m2rnn",
        ]
        for name in representative:
            assert _max_abs({name: after[name] - before[name]}, name) > 0, name
        return metrics.loss, metrics.ntokens, before, after

    eager_loss, eager_ntokens, eager_before, eager_after = run_step(compile=False)
    compiled_loss, compiled_ntokens, compiled_before, compiled_after = run_step(
        compile=True
    )

    assert eager_ntokens == compiled_ntokens == 6
    assert math.isclose(compiled_loss, eager_loss, rel_tol=1e-5, abs_tol=1e-6)
    for name in representative:
        np.testing.assert_allclose(
            compiled_after[name] - compiled_before[name],
            eager_after[name] - eager_before[name],
            rtol=1e-4,
            atol=1e-7,
        )


def test_hybrid_lm_e_only_step_updates_all_routed_and_shared_experts() -> None:
    mx.random.seed(127)
    model = HybridTinyLM(
        _hybrid_config(
            pattern="E",
            depth=1,
            dsa_a_layer_ranks=(),
            hidden_size=8,
            num_attention_heads=1,
            max_seq_length=7,
            moe_num_experts=4,
            moe_top_k=4,
            moe_expert_hidden_size=16,
            moe_shared_expert_hidden_size=8,
        )
    )
    moe = model.layers[0].moe_block
    assert moe is not None
    optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=0.0)
    batch = synthetic_token_batch(
        batch_size=2,
        seq_length=6,
        vocab_size=model.config.vocab_size,
        seed=337,
        include_structure=True,
    )
    expected_updates = [
        "layers.0.block.router.gate.weight",
        "layers.0.block.shared_expert.gate_proj.weight",
        "layers.0.block.shared_expert.up_proj.weight",
        "layers.0.block.shared_expert.down_proj.weight",
    ]
    for expert_id in range(moe.config.num_experts):
        expected_updates.extend(
            [
                f"layers.0.block.experts.{expert_id}.gate_proj.weight",
                f"layers.0.block.experts.{expert_id}.up_proj.weight",
                f"layers.0.block.experts.{expert_id}.down_proj.weight",
            ]
        )
    before = _flat_tree(model.parameters())

    loss_and_grad = nn.value_and_grad(model, next_token_cross_entropy)
    (loss, ntokens), grads = loss_and_grad(model, batch)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state, loss, ntokens)
    flat_grads = _flat_tree(grads)
    after = _flat_tree(model.parameters())
    optimizer_state = _flat_tree(optimizer.state)

    assert model.route_symbols == ("E",)
    assert [layer.backend for layer in model.layers] == ["moe"]
    assert math.isfinite(float(loss.item()))
    assert int(ntokens.item()) == 10
    for name in expected_updates:
        assert _max_abs(flat_grads, name) > 0, name
        assert _max_abs({name: after[name] - before[name]}, name) > 0, name
        _assert_adamw_state_for(optimizer_state, name)


def test_hybrid_lm_train_eval_modes_preserve_logits_contract() -> None:
    mx.random.seed(13)
    model = HybridTinyLM(_hybrid_config())
    batch = synthetic_token_batch(
        batch_size=2,
        seq_length=6,
        vocab_size=model.config.vocab_size,
        seed=29,
        include_structure=True,
    )

    model.train()
    assert model.training is True
    assert all(layer.training is True for layer in model.layers)
    train_logits = model(batch.inputs, **batch.model_kwargs())

    model.eval()
    assert model.training is False
    assert all(layer.training is False for layer in model.layers)
    eval_logits = model(batch.inputs, **batch.model_kwargs())
    mx.eval(train_logits, eval_logits)

    expected = (2, 5, model.config.vocab_size)
    assert train_logits.shape == expected
    assert eval_logits.shape == expected
    assert np.isfinite(np.array(train_logits)).all()
    assert np.isfinite(np.array(eval_logits)).all()

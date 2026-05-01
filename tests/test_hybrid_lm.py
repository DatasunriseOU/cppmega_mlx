from __future__ import annotations

import math

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
import pytest

from cppmega_mlx.data.batch import synthetic_token_batch
from cppmega_mlx.models.hybrid_lm import HybridTinyConfig, HybridTinyLM
from cppmega_mlx.nn.structure_embedding import CppMegaStructureEmbedding
from cppmega_mlx.recipes.nam56r import REFERENCE_PATTERN
from cppmega_mlx.training.loop import one_step_train
from cppmega_mlx.training.loss import next_token_cross_entropy


_ROUTE_BACKENDS = {
    "A": "attention",
    "M": "mamba3",
    "E": "moe",
    "R": "m2rnn",
}


def _hybrid_config(**overrides) -> HybridTinyConfig:
    params = {
        "vocab_size": 48,
        "hidden_size": 16,
        "pattern": "AEMR",
        "depth": 4,
        "dsa_a_layer_ranks": (0,),
        "num_attention_heads": 4,
        "max_seq_length": 16,
        "structure_vocab_size": 16,
    }
    params.update(overrides)
    return HybridTinyConfig(**params)


def _small_single_route_config(symbol: str) -> HybridTinyConfig:
    return _hybrid_config(
        hidden_size=8,
        pattern=symbol,
        depth=1,
        dsa_a_layer_ranks=(0,) if symbol == "A" else (),
        num_attention_heads=1,
        max_seq_length=8,
        mamba_expand=1,
        mamba_head_dim=4,
        mamba_state_dim=4,
        mamba_groups=1,
        mamba_chunk_size=4,
        moe_num_experts=4,
        moe_top_k=4,
        moe_expert_hidden_size=16,
        moe_shared_expert_hidden_size=8,
        m2rnn_k_head_dim=2,
        m2rnn_v_head_dim=2,
        m2rnn_num_v_heads=1,
        m2rnn_num_f_heads=1,
        m2rnn_num_weight_heads=1,
        m2rnn_chunk_size=4,
    )


def test_single_route_lms_preserve_route_specific_loss_contract() -> None:
    for offset, (symbol, backend) in enumerate(_ROUTE_BACKENDS.items()):
        config = _hybrid_config(
            pattern=symbol,
            depth=1,
            dsa_a_layer_ranks=(0,) if symbol == "A" else (),
            max_seq_length=8,
        )
        model = HybridTinyLM(config)
        batch = synthetic_token_batch(
            batch_size=2,
            seq_length=6,
            vocab_size=model.config.vocab_size,
            seed=41 + offset,
            include_structure=True,
        )

        loss, ntokens = next_token_cross_entropy(model, batch)
        mx.eval(loss, ntokens)

        assert model.route_symbols == (symbol,)
        assert model.route_roles == (backend,)
        assert [layer.backend for layer in model.layers] == [backend]
        if symbol == "A":
            route_info = model.layers[0].block.route_info
            assert route_info is not None
            assert route_info.mode == "dsa"
        assert int(ntokens.item()) == 10
        assert math.isfinite(float(loss.item()))
        assert float(loss.item()) > 0


def test_single_route_blocks_emit_finite_distinguishable_route_deltas() -> None:
    deltas: dict[str, np.ndarray] = {}

    for offset, (symbol, backend) in enumerate(_ROUTE_BACKENDS.items()):
        mx.random.seed(301 + offset)
        model = HybridTinyLM(_small_single_route_config(symbol))
        batch = synthetic_token_batch(
            batch_size=2,
            seq_length=6,
            vocab_size=model.config.vocab_size,
            seed=331,
            include_structure=False,
        )
        hidden_states = model.token_embedding(batch.inputs) + model.position_embedding(
            mx.arange(batch.inputs.shape[1])[None, :]
        )
        mask = mx.zeros((batch.inputs.shape[1], batch.inputs.shape[1]), dtype=hidden_states.dtype)
        delta = model.layers[0].route_delta(hidden_states, mask)
        mx.eval(delta)

        delta_np = np.array(delta)
        assert model.route_symbols == (symbol,)
        assert [layer.backend for layer in model.layers] == [backend]
        assert delta_np.shape == np.array(hidden_states).shape
        assert np.isfinite(delta_np).all()
        assert np.max(np.abs(delta_np)) > 0, symbol
        deltas[symbol] = delta_np

    for left in _ROUTE_BACKENDS:
        for right in _ROUTE_BACKENDS:
            if left >= right:
                continue
            assert not np.allclose(deltas[left], deltas[right]), (left, right)


def test_mixed_mamba3_m2rnn_route_runs_loss_and_train_step() -> None:
    config = _hybrid_config(
        hidden_size=8,
        pattern="MR",
        depth=2,
        dsa_a_layer_ranks=(),
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
        m2rnn_chunk_size=4,
    )
    model = HybridTinyLM(config)
    batch = synthetic_token_batch(
        batch_size=2,
        seq_length=6,
        vocab_size=model.config.vocab_size,
        seed=71,
        include_structure=True,
    )

    loss, ntokens = next_token_cross_entropy(model, batch)
    mx.eval(loss, ntokens)
    train_result = one_step_train(
        model,
        optim.AdamW(learning_rate=1e-3, weight_decay=0.0),
        batch,
    )

    assert model.route_symbols == ("M", "R")
    assert model.route_roles == ("mamba3", "m2rnn")
    assert [layer.backend for layer in model.layers] == ["mamba3", "m2rnn"]
    assert int(ntokens.item()) == 10
    assert math.isfinite(float(loss.item()))
    assert float(loss.item()) > 0
    assert train_result.ntokens == 10
    assert math.isfinite(train_result.loss)
    assert train_result.loss > 0
    assert train_result.tokens_per_second > 0


def test_route_construction_from_nam56r_constants_at_tiny_scale() -> None:
    config = _hybrid_config(
        pattern=REFERENCE_PATTERN,
        depth=len(REFERENCE_PATTERN) + 1,
        dsa_a_layer_ranks=(0, 1, 2),
    )
    model = HybridTinyLM(config)

    assert "".join(model.route_symbols) == "AEMEAEMEAEMRA"
    assert model.route_roles == (
        "attention",
        "moe",
        "mamba3",
        "moe",
        "attention",
        "moe",
        "mamba3",
        "moe",
        "attention",
        "moe",
        "mamba3",
        "m2rnn",
        "attention",
    )
    assert model.pattern.dsa_layer_numbers == (1, 5, 9)
    assert model.pattern.mla_layer_numbers == (13,)
    assert [block.backend for block in model.layers] == [
        "attention",
        "moe",
        "mamba3",
        "moe",
        "attention",
        "moe",
        "mamba3",
        "moe",
        "attention",
        "moe",
        "mamba3",
        "m2rnn",
        "attention",
    ]
    first_route = model.layers[0].block.route_info
    last_route = model.layers[-1].block.route_info
    assert first_route is not None
    assert last_route is not None
    assert first_route.mode == "dsa"
    assert last_route.mode == "mla"


def test_full_aemr_route_direct_forward_exercises_all_backend_modules() -> None:
    mx.random.seed(501)
    model = HybridTinyLM(
        _hybrid_config(
            hidden_size=8,
            pattern="AEMR",
            depth=4,
            dsa_a_layer_ranks=(0,),
            num_attention_heads=1,
            max_seq_length=8,
            mamba_expand=1,
            mamba_head_dim=4,
            mamba_state_dim=4,
            mamba_groups=1,
            mamba_chunk_size=4,
            moe_top_k=4,
            moe_expert_hidden_size=16,
            moe_shared_expert_hidden_size=8,
            m2rnn_k_head_dim=2,
            m2rnn_v_head_dim=2,
            m2rnn_num_v_heads=1,
            m2rnn_num_f_heads=1,
            m2rnn_num_weight_heads=1,
            m2rnn_chunk_size=4,
        )
    )
    batch = synthetic_token_batch(
        batch_size=2,
        seq_length=6,
        vocab_size=model.config.vocab_size,
        seed=503,
        include_structure=True,
    )

    logits = model(batch.inputs, **batch.model_kwargs())
    mx.eval(logits)

    assert model.route_symbols == ("A", "E", "M", "R")
    assert model.route_roles == ("attention", "moe", "mamba3", "m2rnn")
    assert [layer.backend for layer in model.layers] == [
        "attention",
        "moe",
        "mamba3",
        "m2rnn",
    ]
    assert model.layers[0].attention_block is not None
    assert model.layers[1].moe_block is not None
    assert model.layers[2].mamba3_block is not None
    assert model.layers[3].m2rnn_block is not None
    assert logits.shape == (2, 5, model.config.vocab_size)
    assert np.isfinite(np.array(logits)).all()


def test_hybrid_tiny_lm_output_shape_and_side_channels() -> None:
    model = HybridTinyLM(_hybrid_config())
    batch = synthetic_token_batch(
        batch_size=2,
        seq_length=7,
        vocab_size=model.config.vocab_size,
        seed=23,
        include_structure=True,
    )

    logits = model(batch.inputs, **batch.model_kwargs())
    mx.eval(logits)

    assert logits.shape == (2, 6, model.config.vocab_size)
    assert np.isfinite(np.array(logits)).all()


def test_hybrid_tiny_lm_uses_source_equivalent_structure_embedding() -> None:
    model = HybridTinyLM(_hybrid_config())

    assert isinstance(model.structure_embedding, CppMegaStructureEmbedding)
    assert model.structure_embedding.active_component_names == ("structure", "dep_level")


def test_hybrid_tiny_lm_zero_init_structure_side_channels_do_not_change_logits() -> None:
    mx.random.seed(61)
    model = HybridTinyLM(
        _hybrid_config(
            pattern="A",
            depth=1,
            dsa_a_layer_ranks=(0,),
            max_seq_length=8,
            structure_components="all",
        )
    )
    batch = synthetic_token_batch(
        batch_size=2,
        seq_length=5,
        vocab_size=model.config.vocab_size,
        seed=43,
        include_structure=True,
    )

    without_structure = model(batch.inputs)
    with_structure = model(batch.inputs, **batch.model_kwargs())
    mx.eval(without_structure, with_structure)

    assert np.allclose(np.array(without_structure), np.array(with_structure))


def test_hybrid_tiny_lm_accepts_all_structure_side_channels() -> None:
    model = HybridTinyLM(
        _hybrid_config(
            pattern="A",
            depth=1,
            dsa_a_layer_ranks=(0,),
            max_seq_length=8,
            structure_components="all",
            structure_num_categories=16,
        )
    )
    batch = synthetic_token_batch(
        batch_size=2,
        seq_length=6,
        vocab_size=model.config.vocab_size,
        seed=47,
        include_structure=True,
    )

    logits = model(batch.inputs, **batch.model_kwargs())
    mx.eval(logits)

    assert model.structure_embedding.active_component_names == (
        "structure",
        "dep_level",
        "ast_depth",
        "sibling_index",
        "ast_node_type",
    )
    assert logits.shape == batch.inputs.shape + (model.config.vocab_size,)
    assert np.isfinite(np.array(logits)).all()


def test_hybrid_tiny_lm_structure_side_channels_can_affect_logits_when_enabled() -> None:
    mx.random.seed(67)
    model = HybridTinyLM(
        _hybrid_config(
            pattern="A",
            depth=1,
            dsa_a_layer_ranks=(0,),
            max_seq_length=8,
            structure_components="all",
            structure_num_categories=16,
        )
    )
    batch = synthetic_token_batch(
        batch_size=2,
        seq_length=5,
        vocab_size=model.config.vocab_size,
        seed=53,
        include_structure=True,
    )

    before = model(batch.inputs, **batch.model_kwargs())
    model.structure_embedding.stacked_emb.weight = mx.ones_like(
        model.structure_embedding.stacked_emb.weight
    )
    model.structure_embedding.up_proj.weight = mx.ones_like(
        model.structure_embedding.up_proj.weight
    )
    after = model(batch.inputs, **batch.model_kwargs())
    mx.eval(before, after)

    assert before.shape == after.shape == batch.inputs.shape + (model.config.vocab_size,)
    assert not np.allclose(np.array(before), np.array(after))


def test_hybrid_tiny_lm_requires_direct_structure_side_channels_to_match_inputs() -> None:
    mx.random.seed(71)
    model = HybridTinyLM(
        _hybrid_config(
            pattern="A",
            depth=1,
            dsa_a_layer_ranks=(0,),
            max_seq_length=8,
            structure_components="all",
            structure_num_categories=16,
        )
    )
    model.structure_embedding.stacked_emb.weight = mx.ones_like(
        model.structure_embedding.stacked_emb.weight
    )
    model.structure_embedding.up_proj.weight = mx.ones_like(
        model.structure_embedding.up_proj.weight
    )
    batch = synthetic_token_batch(
        batch_size=2,
        seq_length=6,
        vocab_size=model.config.vocab_size,
        seed=57,
        include_structure=True,
    )

    sliced_logits = model(batch.inputs, **batch.model_kwargs())
    mx.eval(sliced_logits)

    assert sliced_logits.shape == batch.inputs.shape + (model.config.vocab_size,)
    with pytest.raises(ValueError, match="structure_ids shape .* must exactly match input_ids shape"):
        model(
            batch.inputs,
            structure_ids=batch.structure_ids,
            dep_levels=batch.dep_levels,
            ast_depth_ids=batch.ast_depth_ids,
            sibling_index_ids=batch.sibling_index_ids,
            node_type_ids=batch.node_type_ids,
        )


def test_hybrid_tiny_lm_rejects_malformed_direct_structure_side_channels() -> None:
    model = HybridTinyLM(
        _hybrid_config(
            pattern="A",
            depth=1,
            dsa_a_layer_ranks=(0,),
            max_seq_length=8,
            structure_components="all",
            structure_num_categories=16,
        )
    )
    batch = synthetic_token_batch(
        batch_size=2,
        seq_length=6,
        vocab_size=model.config.vocab_size,
        seed=59,
        include_structure=True,
    )
    valid_kwargs = batch.model_kwargs()

    invalid_cases = (
        ("rank", {"structure_ids": mx.zeros((2, 5, 1), dtype=mx.int32)}),
        ("batch", {"dep_levels": mx.zeros((1, 5), dtype=mx.int32)}),
        ("sequence", {"node_type_ids": mx.zeros((2, 4), dtype=mx.int32)}),
        ("too_long", {"node_type_ids": mx.zeros((2, 6), dtype=mx.int32)}),
    )

    for label, replacement in invalid_cases:
        kwargs = {**valid_kwargs, **replacement}
        with pytest.raises(ValueError, match=r"structure_ids|dep_levels|node_type_ids"):
            model(batch.inputs, **kwargs)
        assert label


def test_hybrid_tiny_block_fails_closed_on_backend_metadata_corruption() -> None:
    model = HybridTinyLM(_small_single_route_config("A"))
    batch = synthetic_token_batch(
        batch_size=2,
        seq_length=5,
        vocab_size=model.config.vocab_size,
        seed=67,
        include_structure=False,
    )
    hidden_states = model.token_embedding(batch.inputs) + model.position_embedding(
        mx.arange(batch.inputs.shape[1])[None, :]
    )
    mask = mx.zeros((batch.inputs.shape[1], batch.inputs.shape[1]), dtype=hidden_states.dtype)

    model.layers[0].backend = "moe"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="requires backend 'attention'"):
        model.layers[0].route_delta(hidden_states, mask)

    model = HybridTinyLM(_small_single_route_config("A"))
    model.layers[0].mamba3_block = object()  # type: ignore[assignment]
    with pytest.raises(ValueError, match="unexpected route modules"):
        model.layers[0].route_delta(hidden_states, mask)


def test_hybrid_tiny_lm_wires_optional_ngram_hash_embedding() -> None:
    model = HybridTinyLM(
        _hybrid_config(
            ngram_hash_enabled=True,
            ngram_hash_orders=(2,),
            ngram_hash_heads=2,
            ngram_hash_table_size=257,
            ngram_hash_embed_dim=4,
            ngram_hash_seed=13,
        )
    )
    batch = synthetic_token_batch(
        batch_size=2,
        seq_length=6,
        vocab_size=model.config.vocab_size,
        seed=37,
        include_structure=True,
    )

    assert model.ngram_hash_embedding is not None
    zero_init_enrichment = model.ngram_hash_embedding(batch.inputs)
    logits = model(batch.inputs, **batch.model_kwargs())
    mx.eval(zero_init_enrichment, logits)

    assert zero_init_enrichment.shape == batch.inputs.shape + (model.config.hidden_size,)
    assert np.count_nonzero(np.array(zero_init_enrichment)) == 0
    assert logits.shape == batch.inputs.shape + (model.config.vocab_size,)
    assert np.isfinite(np.array(logits)).all()


def test_hybrid_tiny_lm_ngram_hash_can_affect_logits_after_projection_is_enabled() -> None:
    config = _hybrid_config(
        pattern="A",
        depth=1,
        dsa_a_layer_ranks=(0,),
        max_seq_length=8,
        ngram_hash_enabled=True,
        ngram_hash_orders=(2,),
        ngram_hash_heads=1,
        ngram_hash_table_size=257,
        ngram_hash_embed_dim=4,
        ngram_hash_seed=17,
    )
    model = HybridTinyLM(config)
    assert model.ngram_hash_embedding is not None
    batch = synthetic_token_batch(
        batch_size=2,
        seq_length=5,
        vocab_size=model.config.vocab_size,
        seed=41,
        include_structure=False,
    )

    before = model(batch.inputs)
    model.ngram_hash_embedding.out_proj.weight = mx.ones_like(
        model.ngram_hash_embedding.out_proj.weight
    )
    after = model(batch.inputs)
    mx.eval(before, after)

    assert before.shape == after.shape == batch.inputs.shape + (model.config.vocab_size,)
    assert not np.allclose(np.array(before), np.array(after))


def test_hybrid_tiny_lm_rejects_invalid_ngram_hash_config_when_enabled() -> None:
    invalid_cases = (
        {"ngram_hash_orders": ()},
        {"ngram_hash_orders": (0,)},
        {"ngram_hash_heads": 0},
        {"ngram_hash_table_size": 0},
        {"ngram_hash_embed_dim": 0},
        {"ngram_hash_dropout": 1.0},
    )

    for overrides in invalid_cases:
        try:
            _hybrid_config(ngram_hash_enabled=True, **overrides)
        except ValueError as exc:
            assert "ngram_hash" in str(exc)
        else:  # pragma: no cover - explicit failure path.
            raise AssertionError(f"expected invalid ngram_hash config for {overrides}")


def test_hybrid_tiny_lm_rejects_invalid_structure_config() -> None:
    invalid_cases = (
        {"structure_components": "core,platform"},
        {"structure_components": ""},
        {"structure_bottleneck_dim": 0},
        {"structure_num_categories": 0},
        {"structure_max_dep_level": 0},
        {"structure_max_ast_depth": 0},
        {"structure_max_sibling_index": 0},
        {"structure_num_node_types": 0},
    )

    for overrides in invalid_cases:
        with pytest.raises((TypeError, ValueError)):
            _hybrid_config(**overrides)


def test_hybrid_tiny_lm_finite_train_step() -> None:
    model = HybridTinyLM(_hybrid_config())
    optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=0.0)
    batch = synthetic_token_batch(
        batch_size=2,
        seq_length=8,
        vocab_size=model.config.vocab_size,
        seed=31,
        include_structure=True,
    )

    result = one_step_train(model, optimizer, batch)

    assert result.ntokens == 14
    assert math.isfinite(result.loss)
    assert result.loss > 0
    assert result.tokens_per_second > 0

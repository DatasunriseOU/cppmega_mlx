from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from cppmega_mlx.recipes.model_factory import (
    LOCAL_GB10_QUARTER_DSA_A_LAYER_RANKS,
    LOCAL_GB10_QUARTER_PATTERN,
    MTPProfile,
    ModelFactoryProfile,
    build_local_gb10_quarter_tiny_smoke_model,
    forward_has_finite_logits,
    get_model_profile,
    local_gb10_quarter_profile,
)


def test_local_gb10_quarter_profile_matches_m0_2_contract_without_allocation() -> None:
    profile = local_gb10_quarter_profile()

    assert profile.name == "local_gb10_quarter"
    assert profile.depth == 13
    assert profile.hidden_size == 3584
    assert profile.ffn_hidden_size == 18_944
    assert profile.num_attention_heads == 28
    assert profile.head_dim == 128
    assert profile.vocab_size == 65_536
    assert profile.pattern == LOCAL_GB10_QUARTER_PATTERN == "AEMEAEMEAEMR"
    assert profile.dsa_a_layer_ranks == LOCAL_GB10_QUARTER_DSA_A_LAYER_RANKS == (1, 2, 3)
    assert profile.mtp == MTPProfile(depth=2, beta=0.6, loss_weight=0.3)
    assert profile.expanded_pattern.source_pattern == "AEMEAEMEAEMR"
    assert "".join(profile.expanded_pattern.symbols) == "AEMEAEMEAEMRA"
    assert profile.expanded_pattern.dsa_layer_numbers == (5, 9, 13)
    assert profile.expanded_pattern.mla_layer_numbers == (1,)


def test_local_gb10_quarter_builds_valid_existing_configs_without_model_allocation() -> None:
    profile = local_gb10_quarter_profile()

    nam = profile.nam56r_config()
    hybrid = profile.hybrid_config()

    assert nam.depth == profile.depth
    assert nam.hidden_size == profile.hidden_size
    assert nam.ffn_hidden_size == profile.ffn_hidden_size
    assert nam.num_attention_heads == profile.num_attention_heads
    assert nam.head_dim == profile.head_dim
    assert nam.vocab_size == profile.vocab_size
    assert nam.moe.ffn_hidden_size == 896
    assert nam.moe.shared_expert_intermediate_size == 1024
    assert hybrid.depth == profile.depth
    assert hybrid.hidden_size == profile.hidden_size
    assert hybrid.vocab_size == profile.vocab_size
    assert hybrid.max_seq_length == 4096
    assert get_model_profile("local_gb10_quarter") == profile


def test_model_factory_validation_fails_closed_for_invalid_combos() -> None:
    with pytest.raises(ValueError, match="num_attention_heads \\* head_dim"):
        local_gb10_quarter_profile(head_dim=64)

    with pytest.raises(ValueError, match="DSA indexer"):
        local_gb10_quarter_profile(dsa_indexer_n_heads=None)

    with pytest.raises(ValueError, match="MTP enabled"):
        local_gb10_quarter_profile(mtp=MTPProfile(depth=None))

    with pytest.raises(ValueError, match="positive depth"):
        local_gb10_quarter_profile(mtp=MTPProfile(depth=0))

    with pytest.raises(ValueError, match="moe_top_k"):
        local_gb10_quarter_profile(moe_num_experts=2, moe_top_k=4)

    with pytest.raises(ValueError, match="DSA A-layer ranks"):
        local_gb10_quarter_profile(dsa_a_layer_ranks=(4,))

    with pytest.raises(ValueError, match="unknown model factory profile"):
        get_model_profile("nam56r_full")


def test_tiny_smoke_model_preserves_profile_route_and_has_finite_t512_forward() -> None:
    mx.random.seed(802)
    model = build_local_gb10_quarter_tiny_smoke_model(vocab_size=64)
    input_ids = mx.array(
        np.arange(512, dtype=np.uint32).reshape(1, 512) % model.config.vocab_size,
        dtype=mx.uint32,
    )

    logits = model(input_ids)
    mx.eval(logits)

    assert model.config.depth == 13
    assert "".join(model.route_symbols) == "AEMEAEMEAEMRA"
    assert model.pattern.dsa_layer_numbers == (5, 9, 13)
    assert model.pattern.mla_layer_numbers == (1,)
    assert logits.shape == (1, 512, model.config.vocab_size)
    assert bool(mx.all(mx.isfinite(logits)).item())
    assert math.isfinite(float(mx.mean(logits).item()))


def test_forward_has_finite_logits_helper_uses_existing_model_builder() -> None:
    model = build_local_gb10_quarter_tiny_smoke_model(
        vocab_size=32,
        depth=1,
        pattern="A",
        dsa_a_layer_ranks=(0,),
        max_seq_length=512,
    )
    input_ids = mx.zeros((1, 512), dtype=mx.uint32)

    assert forward_has_finite_logits(model, input_ids)


def test_profile_dataclass_rejects_unsupported_model_kind() -> None:
    with pytest.raises(ValueError, match="unsupported model_kind"):
        ModelFactoryProfile(
            name="bad",
            pattern="A",
            depth=1,
            hidden_size=128,
            ffn_hidden_size=256,
            num_attention_heads=1,
            head_dim=128,
            vocab_size=256,
            dsa_a_layer_ranks=(0,),
            model_kind="dense",  # type: ignore[arg-type]
        )

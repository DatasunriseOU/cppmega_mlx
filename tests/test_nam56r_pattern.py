from importlib import import_module
import sys
from pathlib import Path

import pytest

from cppmega_mlx.config.model import DEFAULT_DSA_A_LAYER_RANKS
from cppmega_mlx.config.model import (
    DSAConfig,
    M2RNNConfig,
    Mamba3Config,
    Nam56RModelConfig,
    NgramHashConfig,
    SourceStructureEnvConfig,
)
from cppmega_mlx.models.hybrid_lm import HybridTinyConfig
from cppmega_mlx.recipes.nam56r import (
    build_hybrid_tiny_config_from_nam56r,
    build_nam56r_parity_contract,
    build_nam56r_pattern,
    require_fully_native_megatron_parity,
)
from cppmega_mlx.recipes.pattern import (
    a_layer_numbers,
    expand_nam_pattern,
    expand_symbols,
    r_layer_numbers,
)


def _cppmega_source_layout():
    reference_root = Path(__file__).resolve().parents[2] / "cppmega"
    if not reference_root.exists():
        pytest.skip("../cppmega reference checkout is not present")
    sys.path.insert(0, str(reference_root))
    module = pytest.importorskip("cppmega.megatron.nam56r_layout")
    # Use dynamic import so local pyright does not require the sibling checkout.
    module = import_module(module.__name__)
    module_file = module.__file__
    assert module_file is not None
    assert Path(module_file).resolve().is_relative_to(reference_root.resolve())
    return module.load_attention_layer_numbers, module.load_dsa_a_layer_ranks


def test_expand_symbols_tiles_pattern_to_depth_with_one_based_layers():
    expanded = expand_nam_pattern("AEMR", 10)

    assert expanded.symbols == ("A", "E", "M", "R", "A", "E", "M", "R", "A", "E")
    assert expanded.layer_numbers == tuple(range(1, 11))
    assert expanded.a_layer_numbers == (1, 5, 9)
    assert expanded.r_layer_numbers == (4, 8)


def test_parser_fails_closed_on_unsupported_symbols():
    for bad in ("", "ADMR", "ADG|", "A|EMR", "A-MR"):
        with pytest.raises(ValueError):
            expand_symbols(bad, 4)

    with pytest.raises(ValueError, match="depth must be positive"):
        expand_symbols("AEMR", 0)


def test_default_nam56r_pattern_matches_cppmega_counts_and_indices():
    expanded = build_nam56r_pattern()
    attention_layers = (1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49)
    moe_layers = (
        2,
        4,
        6,
        8,
        10,
        14,
        16,
        18,
        20,
        22,
        26,
        28,
        30,
        32,
        34,
        38,
        40,
        42,
        44,
        46,
        50,
        52,
    )
    mamba3_layers = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51)
    m2rnn_layers = (12, 24, 36, 48)

    assert expanded.depth == 52
    assert expanded.counts == {"A": 13, "E": 22, "M": 13, "R": 4}
    assert expanded.role_counts == {
        "attention": 13,
        "moe": 22,
        "mamba3": 13,
        "m2rnn": 4,
    }
    assert expanded.a_layer_numbers == attention_layers
    assert expanded.moe_layer_numbers == moe_layers
    assert expanded.mamba3_layer_numbers == mamba3_layers
    assert expanded.r_layer_numbers == m2rnn_layers
    assert expanded.layer_numbers_by_role == {
        "attention": attention_layers,
        "moe": moe_layers,
        "mamba3": mamba3_layers,
        "m2rnn": m2rnn_layers,
    }
    assert expanded.layer_numbers_for_role("attention") == attention_layers
    assert expanded.layer_numbers_for_role("moe") == moe_layers
    assert expanded.layer_numbers_for_role("mamba3") == mamba3_layers
    assert expanded.layer_numbers_for_role("m2rnn") == m2rnn_layers
    assert a_layer_numbers("AEMEAEMEAEMR", 52) == expanded.a_layer_numbers
    assert r_layer_numbers("AEMEAEMEAEMR", 52) == expanded.r_layer_numbers


def test_default_routes_match_cppmega_source_rank_mapping(monkeypatch):
    load_attention_layer_numbers, load_dsa_a_layer_ranks = _cppmega_source_layout()
    monkeypatch.setenv("CPPMEGA_NEM_PATTERN", "AEMEAEMEAEMR")
    monkeypatch.setenv("CPPMEGA_LAYER_DEPTH", "52")
    monkeypatch.setenv("CPPMEGA_DSA_A_LAYER_RANKS", "1,2,3,5,6,7,9,10,11")

    source_a_layers = load_attention_layer_numbers()
    source_dsa_ranks = load_dsa_a_layer_ranks()
    source_dsa_layers = tuple(source_a_layers[rank] for rank in source_dsa_ranks)
    source_mla_layers = tuple(
        layer for rank, layer in enumerate(source_a_layers) if rank not in source_dsa_ranks
    )
    expanded = build_nam56r_pattern()

    assert expanded.a_layer_numbers == source_a_layers
    assert expanded.dsa_a_layer_ranks == source_dsa_ranks
    assert expanded.dsa_layer_numbers == source_dsa_layers
    assert expanded.mla_layer_numbers == source_mla_layers


def test_default_dsa_routes_match_cppmega_zero_based_a_indices():
    expanded = build_nam56r_pattern()

    assert expanded.dsa_a_layer_ranks == DEFAULT_DSA_A_LAYER_RANKS
    assert expanded.dsa_layer_numbers == (5, 9, 13, 21, 25, 29, 37, 41, 45)
    assert expanded.mla_layer_numbers == (1, 17, 33, 49)
    assert expanded.attention_route_for_layer(1) == "mla"
    assert expanded.attention_route_for_layer(13) == "dsa"
    assert expanded.attention_route_for_layer(12) is None


def test_dsa_zero_based_rank_validation_accepts_first_a_layer_and_rejects_bad_indices():
    expanded = expand_nam_pattern("AEMR", 8, dsa_a_layer_ranks=(0,))

    assert expanded.a_layer_numbers == (1, 5)
    assert expanded.dsa_layer_numbers == (1,)
    assert expanded.mla_layer_numbers == (5,)
    assert [layer.a_rank for layer in expanded.layers if layer.symbol == "A"] == [0, 1]

    with pytest.raises(ValueError, match="duplicate"):
        expand_nam_pattern("AEMR", 8, dsa_a_layer_ranks=(1, 1))

    with pytest.raises(ValueError, match="non-negative"):
        expand_nam_pattern("AEMR", 8, dsa_a_layer_ranks=(-1,))

    with pytest.raises(ValueError, match="exceed"):
        expand_nam_pattern("AEMEAEMEAEMR", 52, dsa_a_layer_ranks=(13,))


def test_layer_roles_distinguish_mamba3_and_m2rnn():
    expanded = expand_nam_pattern("AEMR", 4, dsa_a_layer_ranks=(0,))

    assert [layer.role for layer in expanded.layers] == ["attention", "moe", "mamba3", "m2rnn"]
    assert expanded.layers[0].attention_route == "dsa"
    assert expanded.layers[2].attention_route is None


def test_nam56r_parity_contract_fails_closed_on_native_megatron_claims():
    contract = build_nam56r_parity_contract()

    assert contract.pattern.depth == 52
    assert contract.locally_covered_roles == ("attention", "moe", "mamba3", "m2rnn")
    assert contract.custom_megatron_roles == ("mamba3", "m2rnn")
    assert contract.unsupported_megatron_symbols == ("D", "G", "|")
    assert contract.megatron_fully_native is False
    assert "does not claim fully native Megatron parity" in contract.reason
    assert "upstream-only symbols D, G, | are unsupported" in contract.reason

    with pytest.raises(NotImplementedError, match="does not claim fully native Megatron parity"):
        require_fully_native_megatron_parity()


def test_nam56r_recipe_builds_hybrid_tiny_config_with_custom_routes():
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

    tiny = build_hybrid_tiny_config_from_nam56r(
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

    assert isinstance(tiny, HybridTinyConfig)
    assert tiny.pattern == "AEMR"
    assert tiny.depth == 4
    assert tiny.expanded_pattern().mamba3_layer_numbers == (3,)
    assert tiny.expanded_pattern().r_layer_numbers == (4,)
    assert tiny.mamba_head_dim == 4
    assert tiny.m2rnn_k_head_dim == 2
    assert tiny.m2rnn_v_head_dim == 2


def test_nam56r_recipe_maps_source_env_defaults_into_hybrid_tiny_config():
    source_config = Nam56RModelConfig(
        pattern="A",
        depth=1,
        hidden_size=8,
        num_attention_heads=1,
        seq_len=8,
        max_position_embeddings=8,
        dsa=DSAConfig(a_layer_ranks=(0,)),
        source_structure_env=SourceStructureEnvConfig(enabled=True),
        ngram_hash=NgramHashConfig(enabled=True),
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

    tiny = build_hybrid_tiny_config_from_nam56r(
        source_config,
        vocab_size=32,
        moe_num_experts=4,
        moe_top_k=2,
        moe_expert_hidden_size=16,
        moe_shared_expert_hidden_size=8,
        m2rnn_num_v_heads=1,
        m2rnn_num_f_heads=1,
        ngram_hash_table_size=257,
    )

    assert tiny.structure_components == "core"
    assert tiny.structure_bottleneck_dim == 64
    assert tiny.structure_max_ast_depth == 20
    assert tiny.structure_max_sibling_index == 10
    assert tiny.structure_num_node_types == 64
    assert tiny.ngram_hash_enabled is True
    assert tiny.ngram_hash_orders == (2, 3)
    assert tiny.ngram_hash_heads == 8
    assert tiny.ngram_hash_table_size == 257
    assert tiny.ngram_hash_embed_dim == 16

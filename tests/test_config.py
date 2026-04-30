import pytest

from cppmega_mlx.config.model import (
    DEFAULT_NAM56R_DEPTH,
    DEFAULT_NAM56R_PATTERN,
    DSAConfig,
    M2RNNConfig,
    Mamba3Config,
    MoeConfig,
    Nam56RModelConfig,
    NgramHashConfig,
    SourceStructureEnvConfig,
    StructureConfig,
    VocabMetadata,
)
from cppmega_mlx.recipes.nam56r import build_nam56r_config, with_dsa_a_layer_ranks


def test_nam56r_core_defaults_match_port_contract():
    config = build_nam56r_config()

    assert config.pattern == DEFAULT_NAM56R_PATTERN == "AEMEAEMEAEMR"
    assert config.depth == DEFAULT_NAM56R_DEPTH == 52
    assert config.hidden_size == 4096
    assert config.num_attention_heads == 32
    assert config.head_dim == 128
    assert config.seq_len == 4096
    assert config.max_position_embeddings == 4096


def test_vocab_metadata_preserves_65536_and_131072_contracts():
    vocab = VocabMetadata()

    assert vocab.local_profile_vocab_size == 65_536
    assert vocab.megacpp_tokenizer_vocab_size == 131_072
    assert vocab.default_model_vocab_size == 131_072
    assert build_nam56r_config().vocab_size == 131_072


def test_moe_m2rnn_and_mamba3_defaults_are_import_safe():
    config = build_nam56r_config()

    assert config.moe == MoeConfig(
        num_experts=16,
        top_k=4,
        ffn_hidden_size=896,
        shared_expert_intermediate_size=1024,
        expert_model_parallel_size=1,
        token_dispatcher_type="alltoall",
        flex_dispatcher_backend="deepep",
        router_dtype="fp32",
        grouped_gemm=True,
        router_fusion=True,
    )
    assert config.m2rnn == M2RNNConfig(
        d_model=4096,
        k_head_dim=64,
        v_head_dim=16,
        conv_kernel=4,
        gradient_clipping=1.0,
        use_residual=True,
        a_init_min=0.0,
        a_init_max=16.0,
        dt_init_min=1e-3,
        dt_init_max=0.1,
        dt_init_floor=1e-4,
        use_xma=False,
        runtime_kernel="triton",
        runtime_save_hnew=False,
        runtime_bwd_chunk_size=64,
        runtime_fwd_autotune=False,
        runtime_fwd_num_warps=4,
        runtime_fwd_num_stages=3,
        runtime_broadcast_views=True,
        runtime_bwd_reduce_broadcast_qk=False,
    )
    assert config.mamba3 == Mamba3Config(
        d_model=4096,
        state_dim=64,
        expand=2,
        head_dim=64,
        num_groups=8,
        rope_fraction=0.5,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        a_floor=1e-4,
        is_outproj_norm=False,
        is_mimo=True,
        mimo_rank=4,
        chunk_size=64,
        recompute=True,
    )
    assert config.mamba3.native_num_heads == 64
    assert config.mamba3.author_num_heads == 128


def test_structure_defaults_match_cppmega_source_contract():
    config = build_nam56r_config()

    assert config.structure == StructureConfig(
        active_components="core",
        bottleneck_dim=64,
        num_categories=9,
        max_dep_level=16,
        max_ast_depth=64,
        max_sibling_index=64,
        num_node_types=256,
    )
    assert config.structure.component_names == ("structure", "dep_level")
    assert StructureConfig(active_components="all").component_names == (
        "structure",
        "dep_level",
        "ast_depth",
        "sibling_index",
        "ast_node_type",
    )
    assert StructureConfig(
        active_components="sibling_index,structure"
    ).component_names == ("structure", "sibling_index")


def test_config_validation_fails_closed():
    with pytest.raises(ValueError, match="hidden_size must be divisible"):
        build_nam56r_config(hidden_size=4097, num_attention_heads=32)

    assert Nam56RModelConfig(
        pattern="AEMR",
        depth=4,
        dsa=DSAConfig(a_layer_ranks=(0,)),
    ).dsa.a_layer_ranks == (0,)

    with pytest.raises(ValueError, match="invalid NAM pattern chars"):
        Nam56RModelConfig(pattern="AEMD")

    with pytest.raises(ValueError, match="DSA A-layer ranks"):
        Nam56RModelConfig(
            pattern="AEMR",
            depth=4,
            dsa=DSAConfig(a_layer_ranks=(1,)),
        )

    with pytest.raises(ValueError, match="top_k"):
        MoeConfig(num_experts=2, top_k=4)

    with pytest.raises(ValueError, match="router_dtype"):
        MoeConfig(token_dispatcher_type="flex", router_dtype=None)

    with pytest.raises(ValueError, match="indexer_dtype"):
        DSAConfig(indexer_dtype="fp8")

    with pytest.raises(ValueError, match="non-negative"):
        DSAConfig(a_layer_ranks=(-1,))

    with pytest.raises(ValueError, match="unique"):
        DSAConfig(a_layer_ranks=(1, 1))

    with pytest.raises(ValueError, match="M2RNN d_model"):
        Nam56RModelConfig(m2rnn=M2RNNConfig(d_model=1024))

    with pytest.raises(ValueError, match="unknown structure components"):
        StructureConfig(active_components="core,platform")

    with pytest.raises(ValueError, match="active_components"):
        StructureConfig(active_components="")

    with pytest.raises(ValueError, match="max_ast_depth"):
        StructureConfig(max_ast_depth=0)


def test_source_custom_embedding_env_defaults_are_explicit_and_separate():
    config = build_nam56r_config()

    assert config.source_structure_env == SourceStructureEnvConfig(
        enabled=False,
        active_components="core",
        max_ast_depth=20,
        max_sibling_index=10,
        num_node_types=64,
        bottleneck_dim=64,
    )
    assert config.ngram_hash == NgramHashConfig(
        enabled=False,
        orders=(2, 3),
        num_heads=8,
        table_size=500_000,
        embed_dim=16,
        dropout=0.0,
        offload=False,
        seed=None,
    )

    assert config.structure.max_ast_depth == 64
    assert config.structure.max_sibling_index == 64
    assert config.structure.num_node_types == 256
    assert config.source_structure_env.max_ast_depth == 20
    assert config.source_structure_env.max_sibling_index == 10
    assert config.source_structure_env.num_node_types == 64


def test_source_env_config_validation_fails_closed():
    with pytest.raises(ValueError, match="max_ast_depth"):
        SourceStructureEnvConfig(max_ast_depth=0)

    with pytest.raises(ValueError, match="unknown structure components"):
        SourceStructureEnvConfig(active_components="core,platform")

    with pytest.raises(ValueError, match="orders"):
        NgramHashConfig(orders=())

    with pytest.raises(ValueError, match="orders"):
        NgramHashConfig(orders=(0,))

    with pytest.raises(ValueError, match="table_size"):
        NgramHashConfig(table_size=0)

    with pytest.raises(ValueError, match="dropout"):
        NgramHashConfig(dropout=1.0)


def test_config_builder_rebases_component_widths_for_hidden_override():
    config = build_nam56r_config(hidden_size=2048, num_attention_heads=16)

    assert config.head_dim == 128
    assert config.m2rnn.d_model == 2048
    assert config.mamba3.d_model == 2048
    assert config.mamba3.native_num_heads == 32


def test_with_dsa_a_layer_ranks_keeps_config_immutable():
    config = build_nam56r_config()
    updated = with_dsa_a_layer_ranks(config, (0, 3))

    assert config.dsa.a_layer_ranks == (1, 2, 3, 5, 6, 7, 9, 10, 11)
    assert updated.dsa.a_layer_ranks == (0, 3)

    with pytest.raises(ValueError, match="DSA A-layer ranks"):
        with_dsa_a_layer_ranks(config, (13,))

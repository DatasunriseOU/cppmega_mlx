"""NAM56R recipe helpers for MLX-native model construction."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import NoReturn

from cppmega_mlx.config.model import (
    DEFAULT_DSA_A_LAYER_RANKS,
    DEFAULT_ATTENTION_HEADS,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_NAM56R_DEPTH,
    DEFAULT_NAM56R_PATTERN,
    M2RNNConfig,
    Mamba3Config,
    Nam56RModelConfig,
)
from cppmega_mlx.recipes.pattern import (
    AttentionRoute,
    ExpandedNamPattern,
    LayerRole,
    expand_nam_pattern,
)


UNSUPPORTED_MEGATRON_PARITY_SYMBOLS = ("D", "G", "|")
CUSTOM_MEGATRON_LAYER_ROLES: tuple[LayerRole, ...] = ("mamba3", "m2rnn")


@dataclass(frozen=True)
class Nam56RParityContract:
    """Local MLX NAM56R route coverage versus source Megatron intent."""

    pattern: ExpandedNamPattern
    locally_covered_roles: tuple[LayerRole, ...]
    custom_megatron_roles: tuple[LayerRole, ...]
    unsupported_megatron_symbols: tuple[str, ...]
    megatron_fully_native: bool

    @property
    def reason(self) -> str:
        custom_roles = ", ".join(self.custom_megatron_roles)
        unsupported_symbols = ", ".join(self.unsupported_megatron_symbols)
        return (
            "local MLX NAM56R covers A/E/M/R route placement, but does not "
            f"claim fully native Megatron parity: {custom_roles} remain custom "
            f"seams and upstream-only symbols {unsupported_symbols} are unsupported"
        )


def build_nam56r_config(**overrides) -> Nam56RModelConfig:
    """Build a validated NAM56R config without importing Megatron or MLX."""

    hidden_size = overrides.get("hidden_size", DEFAULT_HIDDEN_SIZE)
    num_attention_heads = overrides.get("num_attention_heads", DEFAULT_ATTENTION_HEADS)
    if hidden_size % num_attention_heads != 0:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    if "m2rnn" not in overrides:
        overrides["m2rnn"] = M2RNNConfig(d_model=hidden_size)
    if "mamba3" not in overrides:
        overrides["mamba3"] = Mamba3Config(d_model=hidden_size)
    return Nam56RModelConfig(**overrides)


def default_nam56r_config() -> Nam56RModelConfig:
    return build_nam56r_config()


def build_nam56r_pattern(
    config: Nam56RModelConfig | None = None,
    *,
    pattern: str | None = None,
    depth: int | None = None,
    dsa_a_layer_ranks: str | tuple[int, ...] | list[int] | None = None,
) -> ExpandedNamPattern:
    """Return the expanded default NAM56R layer plan."""

    if config is None:
        config = build_nam56r_config()
    return expand_nam_pattern(
        pattern if pattern is not None else config.pattern,
        depth if depth is not None else config.depth,
        dsa_a_layer_ranks=(
            dsa_a_layer_ranks if dsa_a_layer_ranks is not None else config.dsa.a_layer_ranks
        ),
    )


def build_nam56r_parity_contract(
    config: Nam56RModelConfig | None = None,
) -> Nam56RParityContract:
    """Describe what the local recipe covers without over-claiming parity."""

    expanded = build_nam56r_pattern(config)
    return Nam56RParityContract(
        pattern=expanded,
        locally_covered_roles=tuple(expanded.layer_numbers_by_role),
        custom_megatron_roles=CUSTOM_MEGATRON_LAYER_ROLES,
        unsupported_megatron_symbols=UNSUPPORTED_MEGATRON_PARITY_SYMBOLS,
        megatron_fully_native=False,
    )


def require_fully_native_megatron_parity(
    config: Nam56RModelConfig | None = None,
) -> NoReturn:
    """Fail closed for callers that need native Megatron parity evidence."""

    contract = build_nam56r_parity_contract(config)
    raise NotImplementedError(contract.reason)


def attention_route_for_layer(
    layer_number: int,
    config: Nam56RModelConfig | None = None,
) -> AttentionRoute | None:
    return build_nam56r_pattern(config).attention_route_for_layer(layer_number)


def with_dsa_a_layer_ranks(
    config: Nam56RModelConfig,
    dsa_a_layer_ranks: tuple[int, ...],
) -> Nam56RModelConfig:
    """Return *config* with a new DSA A-rank routing tuple."""

    return replace(config, dsa=replace(config.dsa, a_layer_ranks=dsa_a_layer_ranks))


def build_hybrid_tiny_config_from_nam56r(
    config: Nam56RModelConfig | None = None,
    **overrides,
):
    """Build the local MLX hybrid smoke config from a validated NAM56R recipe.

    Kept import-lazy so recipe/pattern helpers remain usable without importing
    MLX.  The returned config is intentionally tiny-capable: callers can
    override dimensions for local Metal smoke tests while preserving the source
    NAM56R pattern and DSA rank routing by default.
    """

    if config is None:
        config = build_nam56r_config()

    from cppmega_mlx.models.hybrid_lm import HybridTinyConfig

    structure = config.structure
    if config.source_structure_env.enabled:
        structure = replace(
            structure,
            active_components=config.source_structure_env.active_components,
            bottleneck_dim=config.source_structure_env.bottleneck_dim,
            max_ast_depth=config.source_structure_env.max_ast_depth,
            max_sibling_index=config.source_structure_env.max_sibling_index,
            num_node_types=config.source_structure_env.num_node_types,
        )

    params = {
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "pattern": config.pattern,
        "depth": config.depth,
        "dsa_a_layer_ranks": config.dsa.a_layer_ranks,
        "num_attention_heads": config.num_attention_heads,
        "max_seq_length": config.seq_len,
        "structure_components": structure.active_components,
        "structure_bottleneck_dim": structure.bottleneck_dim,
        "structure_num_categories": structure.num_categories,
        "structure_max_dep_level": structure.max_dep_level,
        "structure_max_ast_depth": structure.max_ast_depth,
        "structure_max_sibling_index": structure.max_sibling_index,
        "structure_num_node_types": structure.num_node_types,
        "moe_num_experts": config.moe.num_experts,
        "moe_top_k": config.moe.top_k,
        "moe_expert_hidden_size": config.moe.ffn_hidden_size,
        "moe_shared_expert_hidden_size": config.moe.shared_expert_intermediate_size,
        "mamba_expand": config.mamba3.expand,
        "mamba_head_dim": config.mamba3.head_dim,
        "mamba_state_dim": config.mamba3.state_dim,
        "mamba_groups": config.mamba3.num_groups,
        "mamba_mimo_rank": config.mamba3.mimo_rank,
        "mamba_is_mimo": config.mamba3.is_mimo,
        "mamba_chunk_size": config.mamba3.chunk_size,
        "mamba_rope_fraction": config.mamba3.rope_fraction,
        "m2rnn_k_head_dim": config.m2rnn.k_head_dim,
        "m2rnn_v_head_dim": config.m2rnn.v_head_dim,
        "m2rnn_chunk_size": config.m2rnn.runtime_bwd_chunk_size,
        "ngram_hash_enabled": config.ngram_hash.enabled,
        "ngram_hash_orders": config.ngram_hash.orders,
        "ngram_hash_heads": config.ngram_hash.num_heads,
        "ngram_hash_table_size": config.ngram_hash.table_size,
        "ngram_hash_embed_dim": config.ngram_hash.embed_dim,
        "ngram_hash_dropout": config.ngram_hash.dropout,
        "ngram_hash_seed": config.ngram_hash.seed,
    }
    params.update(overrides)
    return HybridTinyConfig(**params)


REFERENCE_PATTERN = DEFAULT_NAM56R_PATTERN
REFERENCE_DEPTH = DEFAULT_NAM56R_DEPTH
REFERENCE_DSA_A_LAYER_RANKS = DEFAULT_DSA_A_LAYER_RANKS

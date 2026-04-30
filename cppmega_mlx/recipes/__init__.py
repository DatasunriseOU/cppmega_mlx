"""Recipe and layer-pattern helpers."""

from cppmega_mlx.recipes.pattern import (
    ORDERED_NAM_SYMBOLS,
    SUPPORTED_NAM_SYMBOLS,
    AttentionRoute,
    ExpandedNamPattern,
    LayerRole,
    NamLayer,
    NamSymbol,
    a_layer_numbers,
    expand_nam_pattern,
    expand_symbols,
    layer_numbers_for_symbol,
    parse_nam_pattern,
    parse_rank_list,
    r_layer_numbers,
)

REFERENCE_PATTERN = "AEMEAEMEAEMR"
REFERENCE_DEPTH = 52
REFERENCE_DSA_A_LAYER_RANKS = (1, 2, 3, 5, 6, 7, 9, 10, 11)
UNSUPPORTED_MEGATRON_PARITY_SYMBOLS = ("D", "G", "|")
CUSTOM_MEGATRON_LAYER_ROLES = ("mamba3", "m2rnn")


def build_nam56r_config(**overrides):
    from cppmega_mlx.recipes.nam56r import build_nam56r_config as impl

    return impl(**overrides)


def default_nam56r_config():
    from cppmega_mlx.recipes.nam56r import default_nam56r_config as impl

    return impl()


def build_nam56r_pattern(
    config=None,
    *,
    pattern=None,
    depth=None,
    dsa_a_layer_ranks=None,
):
    from cppmega_mlx.recipes.nam56r import build_nam56r_pattern as impl

    return impl(
        config,
        pattern=pattern,
        depth=depth,
        dsa_a_layer_ranks=dsa_a_layer_ranks,
    )


def build_nam56r_parity_contract(config=None):
    from cppmega_mlx.recipes.nam56r import build_nam56r_parity_contract as impl

    return impl(config)


def require_fully_native_megatron_parity(config=None):
    from cppmega_mlx.recipes.nam56r import require_fully_native_megatron_parity as impl

    return impl(config)


def attention_route_for_layer(layer_number: int, config=None):
    from cppmega_mlx.recipes.nam56r import attention_route_for_layer as impl

    return impl(layer_number, config)


def with_dsa_a_layer_ranks(config, dsa_a_layer_ranks: tuple[int, ...]):
    from cppmega_mlx.recipes.nam56r import with_dsa_a_layer_ranks as impl

    return impl(config, dsa_a_layer_ranks)


def build_hybrid_tiny_config_from_nam56r(config=None, **overrides):
    from cppmega_mlx.recipes.nam56r import build_hybrid_tiny_config_from_nam56r as impl

    return impl(config, **overrides)

__all__ = [
    "CUSTOM_MEGATRON_LAYER_ROLES",
    "ORDERED_NAM_SYMBOLS",
    "REFERENCE_DEPTH",
    "REFERENCE_DSA_A_LAYER_RANKS",
    "REFERENCE_PATTERN",
    "SUPPORTED_NAM_SYMBOLS",
    "UNSUPPORTED_MEGATRON_PARITY_SYMBOLS",
    "AttentionRoute",
    "ExpandedNamPattern",
    "LayerRole",
    "NamLayer",
    "NamSymbol",
    "a_layer_numbers",
    "attention_route_for_layer",
    "build_hybrid_tiny_config_from_nam56r",
    "build_nam56r_config",
    "build_nam56r_parity_contract",
    "build_nam56r_pattern",
    "default_nam56r_config",
    "expand_nam_pattern",
    "expand_symbols",
    "layer_numbers_for_symbol",
    "parse_nam_pattern",
    "parse_rank_list",
    "r_layer_numbers",
    "require_fully_native_megatron_parity",
    "with_dsa_a_layer_ranks",
]

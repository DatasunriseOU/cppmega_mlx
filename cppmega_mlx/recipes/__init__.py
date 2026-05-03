"""Recipe and layer-pattern helpers."""

from importlib import import_module
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from cppmega_mlx.recipes.model_factory import (
        LOCAL_GB10_QUARTER_DEPTH,
        LOCAL_GB10_QUARTER_DSA_A_LAYER_RANKS,
        LOCAL_GB10_QUARTER_FFN_HIDDEN_SIZE,
        LOCAL_GB10_QUARTER_HEAD_DIM,
        LOCAL_GB10_QUARTER_HIDDEN_SIZE,
        LOCAL_GB10_QUARTER_MAX_SEQ_LENGTH,
        LOCAL_GB10_QUARTER_MTP_BETA,
        LOCAL_GB10_QUARTER_MTP_DEPTH,
        LOCAL_GB10_QUARTER_MTP_LAMBDA,
        LOCAL_GB10_QUARTER_NUM_HEADS,
        LOCAL_GB10_QUARTER_PATTERN,
        LOCAL_GB10_QUARTER_PROFILE,
        LOCAL_GB10_QUARTER_UPSTREAM_RECIPE_NAME,
        LOCAL_GB10_QUARTER_VOCAB_SIZE,
        MODEL_FACTORY_UPSTREAM_RECIPE_MODULE,
        MTPProfile,
        ModelFactoryProfile,
        NAM56R_FULL_DEPTH,
        NAM56R_FULL_DSA_A_LAYER_RANKS,
        NAM56R_FULL_FFN_HIDDEN_SIZE,
        NAM56R_FULL_HEAD_DIM,
        NAM56R_FULL_HIDDEN_SIZE,
        NAM56R_FULL_MAX_SEQ_LENGTH,
        NAM56R_FULL_NUM_HEADS,
        NAM56R_FULL_PATTERN,
        NAM56R_FULL_PROFILE,
        NAM56R_FULL_UPSTREAM_RECIPE_NAME,
        NAM56R_FULL_VOCAB_SIZE,
        build_local_gb10_quarter_tiny_smoke_model,
        forward_has_finite_logits,
        get_model_profile,
        local_gb10_quarter,
        local_gb10_quarter_profile,
        nam56r_full_profile,
    )

REFERENCE_PATTERN = "AEMEAEMEAEMR"
REFERENCE_DEPTH = 52
REFERENCE_DSA_A_LAYER_RANKS = (1, 2, 3, 5, 6, 7, 9, 10, 11)
UNSUPPORTED_MEGATRON_PARITY_SYMBOLS = ("D", "G", "|")
CUSTOM_MEGATRON_LAYER_ROLES = ("mamba3", "m2rnn")

_MODEL_FACTORY_EXPORTS = frozenset(
    {
        "LOCAL_GB10_QUARTER_DEPTH",
        "LOCAL_GB10_QUARTER_DSA_A_LAYER_RANKS",
        "LOCAL_GB10_QUARTER_FFN_HIDDEN_SIZE",
        "LOCAL_GB10_QUARTER_HEAD_DIM",
        "LOCAL_GB10_QUARTER_HIDDEN_SIZE",
        "LOCAL_GB10_QUARTER_MAX_SEQ_LENGTH",
        "LOCAL_GB10_QUARTER_MTP_BETA",
        "LOCAL_GB10_QUARTER_MTP_DEPTH",
        "LOCAL_GB10_QUARTER_MTP_LAMBDA",
        "LOCAL_GB10_QUARTER_NUM_HEADS",
        "LOCAL_GB10_QUARTER_PATTERN",
        "LOCAL_GB10_QUARTER_PROFILE",
        "LOCAL_GB10_QUARTER_UPSTREAM_RECIPE_NAME",
        "LOCAL_GB10_QUARTER_VOCAB_SIZE",
        "MODEL_FACTORY_UPSTREAM_RECIPE_MODULE",
        "MTPProfile",
        "ModelFactoryProfile",
        "NAM56R_FULL_DEPTH",
        "NAM56R_FULL_DSA_A_LAYER_RANKS",
        "NAM56R_FULL_FFN_HIDDEN_SIZE",
        "NAM56R_FULL_HEAD_DIM",
        "NAM56R_FULL_HIDDEN_SIZE",
        "NAM56R_FULL_MAX_SEQ_LENGTH",
        "NAM56R_FULL_NUM_HEADS",
        "NAM56R_FULL_PATTERN",
        "NAM56R_FULL_PROFILE",
        "NAM56R_FULL_UPSTREAM_RECIPE_NAME",
        "NAM56R_FULL_VOCAB_SIZE",
        "build_local_gb10_quarter_tiny_smoke_model",
        "forward_has_finite_logits",
        "get_model_profile",
        "local_gb10_quarter",
        "local_gb10_quarter_profile",
        "nam56r_full_profile",
    }
)


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


def __getattr__(name: str):
    if name in _MODEL_FACTORY_EXPORTS:
        impl = import_module("cppmega_mlx.recipes.model_factory")
        return getattr(impl, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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
    "LOCAL_GB10_QUARTER_DEPTH",
    "LOCAL_GB10_QUARTER_DSA_A_LAYER_RANKS",
    "LOCAL_GB10_QUARTER_FFN_HIDDEN_SIZE",
    "LOCAL_GB10_QUARTER_HEAD_DIM",
    "LOCAL_GB10_QUARTER_HIDDEN_SIZE",
    "LOCAL_GB10_QUARTER_MAX_SEQ_LENGTH",
    "LOCAL_GB10_QUARTER_MTP_BETA",
    "LOCAL_GB10_QUARTER_MTP_DEPTH",
    "LOCAL_GB10_QUARTER_MTP_LAMBDA",
    "LOCAL_GB10_QUARTER_NUM_HEADS",
    "LOCAL_GB10_QUARTER_PATTERN",
    "LOCAL_GB10_QUARTER_PROFILE",
    "LOCAL_GB10_QUARTER_UPSTREAM_RECIPE_NAME",
    "LOCAL_GB10_QUARTER_VOCAB_SIZE",
    "MODEL_FACTORY_UPSTREAM_RECIPE_MODULE",
    "MTPProfile",
    "ModelFactoryProfile",
    "NAM56R_FULL_DEPTH",
    "NAM56R_FULL_DSA_A_LAYER_RANKS",
    "NAM56R_FULL_FFN_HIDDEN_SIZE",
    "NAM56R_FULL_HEAD_DIM",
    "NAM56R_FULL_HIDDEN_SIZE",
    "NAM56R_FULL_MAX_SEQ_LENGTH",
    "NAM56R_FULL_NUM_HEADS",
    "NAM56R_FULL_PATTERN",
    "NAM56R_FULL_PROFILE",
    "NAM56R_FULL_UPSTREAM_RECIPE_NAME",
    "NAM56R_FULL_VOCAB_SIZE",
    "NamLayer",
    "NamSymbol",
    "a_layer_numbers",
    "attention_route_for_layer",
    "build_hybrid_tiny_config_from_nam56r",
    "build_local_gb10_quarter_tiny_smoke_model",
    "build_nam56r_config",
    "build_nam56r_parity_contract",
    "build_nam56r_pattern",
    "default_nam56r_config",
    "expand_nam_pattern",
    "expand_symbols",
    "forward_has_finite_logits",
    "get_model_profile",
    "layer_numbers_for_symbol",
    "local_gb10_quarter",
    "local_gb10_quarter_profile",
    "nam56r_full_profile",
    "parse_nam_pattern",
    "parse_rank_list",
    "r_layer_numbers",
    "require_fully_native_megatron_parity",
    "with_dsa_a_layer_ranks",
]

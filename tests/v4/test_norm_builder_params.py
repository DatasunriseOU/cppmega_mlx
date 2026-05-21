"""E7-6-builder tests: pre_norm/post_norm params actually thread into
_build_attention and _build_mlp (Spec-v2 §3.5)."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from cppmega_v4.models.unified_superblock_v4 import (
    BLOCK_BUILDERS, _make_norm,
)


HIDDEN = 32


def test_make_norm_returns_rmsnorm():
    n = _make_norm("rmsnorm", HIDDEN, 1e-6)
    assert isinstance(n, nn.RMSNorm)


def test_make_norm_returns_layernorm():
    n = _make_norm("layernorm", HIDDEN, 1e-5)
    assert isinstance(n, nn.LayerNorm)


def test_make_norm_returns_none_for_none():
    assert _make_norm("none", HIDDEN, 1e-6) is None


def test_make_norm_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown norm kind"):
        _make_norm("batchnorm", HIDDEN, 1e-6)


@pytest.mark.parametrize("pre,post", [
    ("none", "rmsnorm"),  # legacy default
    ("rmsnorm", "none"),
    ("layernorm", "rmsnorm"),
    ("rmsnorm", "rmsnorm"),
    ("layernorm", "layernorm"),
])
def test_attention_with_norm_combos_forward(pre, post):
    """Each valid norm combination produces finite forward output."""
    mlp = BLOCK_BUILDERS["attention"](HIDDEN, {
        "pre_norm": pre, "post_norm": post,
    })
    x = mx.random.normal((1, 4, HIDDEN), key=mx.random.key(0))
    y = mlp(x)
    assert y.shape == (1, 4, HIDDEN)
    assert mx.isfinite(y).all().item()


@pytest.mark.parametrize("pre,post", [
    ("none", "none"),
    ("rmsnorm", "layernorm"),
    ("layernorm", "rmsnorm"),
])
def test_mlp_with_norm_combos_forward(pre, post):
    """MLP same — pre/post wrap the activation core."""
    mlp = BLOCK_BUILDERS["mlp"](HIDDEN, {
        "intermediate_size": 64,
        "pre_norm": pre, "post_norm": post,
        "activation": "swiglu",
    })
    x = mx.random.normal((1, 4, HIDDEN), key=mx.random.key(0))
    y = mlp(x)
    assert y.shape == (1, 4, HIDDEN)
    assert mx.isfinite(y).all().item()


def test_attention_default_matches_legacy_only_post_norm():
    """Without pre_norm/post_norm params, behavior matches legacy
    (single RMSNorm at output, no pre-norm)."""
    mlp = BLOCK_BUILDERS["attention"](HIDDEN, {})
    assert mlp.pre_norm is None
    assert isinstance(mlp.norm, nn.RMSNorm)


def test_attention_layernorm_post_switch_instantiated():
    mlp = BLOCK_BUILDERS["attention"](HIDDEN, {
        "post_norm": "layernorm",
    })
    assert isinstance(mlp.norm, nn.LayerNorm)


def test_mlp_default_no_norms_legacy_behavior():
    """MLP legacy default: no pre/post norm wrappers (caller wraps)."""
    mlp = BLOCK_BUILDERS["mlp"](HIDDEN, {})
    assert mlp.pre_norm is None
    assert mlp.post_norm is None

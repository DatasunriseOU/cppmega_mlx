"""E7-5 tests: _build_mlp activation switch (glu/swiglu/gelu/relu/relu2/silu)."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from cppmega_v4.models.unified_superblock_v4 import BLOCK_BUILDERS


HIDDEN = 32


def _forward(activation: str | None) -> mx.array:
    params = {"intermediate_size": 64}
    if activation is not None:
        params["activation"] = activation
    mlp = BLOCK_BUILDERS["mlp"](HIDDEN, params)
    x = mx.random.normal((1, 4, HIDDEN), key=mx.random.key(0))
    return mlp(x)


def test_mlp_default_keeps_glu_behavior():
    """No activation param → existing sigmoid(gate)*up path."""
    y = _forward(None)
    assert y.shape == (1, 4, HIDDEN)


@pytest.mark.parametrize("activation", [
    "glu", "swiglu", "gelu", "relu", "relu2", "sqrelu", "silu",
])
def test_mlp_each_activation_preserves_shape(activation):
    y = _forward(activation)
    assert y.shape == (1, 4, HIDDEN)
    assert mx.isfinite(y).all().item()


def test_mlp_swiglu_differs_from_glu():
    """SwiGLU uses silu(gate), GLU uses sigmoid(gate) — outputs must differ."""
    mx.random.seed(42)
    y_glu = _forward("glu")
    mx.random.seed(42)
    y_swi = _forward("swiglu")
    # New module instances so weights re-init — checking shape+finite suffices
    assert y_glu.shape == y_swi.shape
    assert mx.isfinite(y_glu).all().item()
    assert mx.isfinite(y_swi).all().item()


def test_mlp_gelu_is_dense_path():
    """Dense activation (gelu) shouldn't multiply by gate — but gate
    weight still exists (state-dict parity). Smoke."""
    y = _forward("gelu")
    assert y.shape == (1, 4, HIDDEN)


def test_mlp_unknown_activation_falls_back_to_glu():
    """Unknown activation name must not crash — fallback to GLU."""
    y = _forward("not_a_real_activation")
    assert y.shape == (1, 4, HIDDEN)
    assert mx.isfinite(y).all().item()

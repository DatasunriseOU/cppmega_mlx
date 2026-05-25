"""E7-13 tests: extended activations (mish/geglu/reglu/xielu)."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from cppmega_mlx.nn.activations import (
    ACTIVATION_NAMES, IS_GATED, apply_activation, is_gated,
)
from cppmega_v4.explain import get_entry


def test_activation_names_have_eleven_entries():
    assert set(ACTIVATION_NAMES) == {
        "glu", "gelu", "relu", "relu2", "sqrelu", "silu", "mish",
        "swiglu", "geglu", "reglu", "xielu",
    }


def test_gated_set_has_five_entries():
    gated = {n for n, g in IS_GATED.items() if g}
    assert gated == {"glu", "swiglu", "geglu", "reglu", "xielu"}


def test_glu_forward_matches_reference():
    x = mx.random.normal((2, 4), key=mx.random.key(0))
    g = mx.random.normal((2, 4), key=mx.random.key(1))
    got = apply_activation("glu", x, gate=g)
    expected = mx.sigmoid(g) * x
    assert mx.allclose(got, expected).item()


def test_mish_is_dense():
    assert not is_gated("mish")


@pytest.mark.parametrize("name", ["geglu", "reglu", "xielu"])
def test_new_gated_activations_require_gate(name):
    x = mx.random.normal((2, 4))
    with pytest.raises(ValueError, match="is gated"):
        apply_activation(name, x)


def test_mish_forward_matches_reference():
    """Mish = x * tanh(softplus(x))."""
    x = mx.random.normal((4, 8), key=mx.random.key(0))
    got = apply_activation("mish", x)
    expected = x * mx.tanh(mx.log1p(mx.exp(x)))
    assert mx.allclose(got, expected).item()


def test_geglu_uses_gelu_on_gate():
    x = mx.random.normal((2, 4), key=mx.random.key(0))
    g = mx.random.normal((2, 4), key=mx.random.key(1))
    got = apply_activation("geglu", x, gate=g)
    expected = nn.gelu_approx(g) * x
    assert mx.allclose(got, expected).item()


def test_reglu_uses_relu_on_gate():
    x = mx.random.normal((2, 4), key=mx.random.key(0))
    g = mx.random.normal((2, 4), key=mx.random.key(1))
    got = apply_activation("reglu", x, gate=g)
    expected = mx.maximum(g, 0) * x
    assert mx.allclose(got, expected).item()


def test_xielu_uses_gelu_gate_silu_value():
    x = mx.random.normal((2, 4), key=mx.random.key(0))
    g = mx.random.normal((2, 4), key=mx.random.key(1))
    got = apply_activation("xielu", x, gate=g)
    expected = nn.gelu_approx(g) * nn.silu(x)
    assert mx.allclose(got, expected).item()


@pytest.mark.parametrize("name", ACTIVATION_NAMES)
def test_every_activation_has_catalog_entry(name):
    entry = get_entry("activation", name)
    assert entry is not None, f"missing catalog entry for {name}"
    assert entry.summary

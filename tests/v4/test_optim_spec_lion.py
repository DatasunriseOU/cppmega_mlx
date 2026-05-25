"""E7-12 backend tests: Lion / Lion8bit / Adam8bit registration + activation
   registry dispatch."""

from __future__ import annotations

import warnings

import mlx.core as mx
import mlx.nn as nn
import pytest

from cppmega_v4.buildspec.optim_spec import (
    LION_LR_WARN_CEILING,
    OPTIM_BUILTINS,
    OptimKind,
    OptimSpec,
    ParamGroup,
    adam8bit,
    adamw,
    lion,
    lion8bit,
    muon,
    muon_adamw_hybrid,
    sgd,
)
from cppmega_mlx.nn.activations import (
    ACTIVATION_NAMES,
    IS_GATED,
    apply_activation,
    is_gated,
)


# ---------------------------------------------------------------------------
# OptimKind enum
# ---------------------------------------------------------------------------


def test_optim_kind_has_ten_entries():
    expected = {"adamw", "muon", "muon_adamw_hybrid",
                "lion", "lion8bit", "adam8bit", "sgd",
                "adam", "adafactor", "rmsprop"}
    assert {k.value for k in OptimKind} == expected


def test_optim_builtins_registers_all_ten():
    for kind in ("adamw", "muon", "muon_adamw_hybrid",
                 "lion", "lion8bit", "adam8bit", "sgd",
                 "adam", "adafactor", "rmsprop"):
        assert kind in OPTIM_BUILTINS
        assert OPTIM_BUILTINS[kind].endswith(f":{kind}")


# ---------------------------------------------------------------------------
# Lion factory + defaults
# ---------------------------------------------------------------------------


def test_lion_factory_defaults_match_paper():
    spec = lion()
    assert spec.kind is OptimKind.LION
    assert len(spec.groups) == 1
    g = spec.groups[0]
    assert g.lr == 1e-4
    assert g.weight_decay == 0.0
    assert g.betas == (0.9, 0.99)
    assert g.matcher == "all"


def test_lion_lr_above_ceiling_emits_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lion(lr=1e-3)
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 1
    msg = str(user_warnings[0].message)
    assert "exceeds recommended ceiling" in msg
    assert "Chen et al" in msg


def test_lion_lr_at_ceiling_does_not_warn():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lion(lr=LION_LR_WARN_CEILING)
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)
                     and "Chen et al" in str(w.message)]
    assert user_warnings == []


def test_lion_requires_betas_when_constructed_directly():
    with pytest.raises(ValueError, match="LION group must declare betas"):
        OptimSpec(
            kind=OptimKind.LION,
            groups=(ParamGroup(matcher="all", lr=1e-4),),  # no betas
        )


# ---------------------------------------------------------------------------
# Lion8bit + Adam8bit factories
# ---------------------------------------------------------------------------


def test_lion8bit_factory_defaults_match_lion():
    spec = lion8bit()
    assert spec.kind is OptimKind.LION_8BIT
    g = spec.groups[0]
    assert g.lr == 1e-4
    assert g.betas == (0.9, 0.99)


def test_lion8bit_lr_above_ceiling_emits_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lion8bit(lr=2e-3)
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)
                     and "Chen et al" in str(w.message)]
    assert len(user_warnings) == 1


def test_lion8bit_requires_betas_when_constructed_directly():
    with pytest.raises(ValueError, match="LION8BIT group must declare betas"):
        OptimSpec(
            kind=OptimKind.LION_8BIT,
            groups=(ParamGroup(matcher="all", lr=1e-4),),
        )


def test_adam8bit_factory_defaults_track_adamw():
    spec = adam8bit()
    assert spec.kind is OptimKind.ADAM_8BIT
    g = spec.groups[0]
    assert g.lr == 3e-4
    assert g.betas == (0.9, 0.999)
    assert g.weight_decay == 0.01


def test_adam8bit_requires_betas():
    with pytest.raises(ValueError, match="ADAM_8BIT group must declare betas"):
        # OptimKind.ADAM_8BIT.value is "adam8bit"; ADAM_8BIT.name is
        # "ADAM_8BIT" — the message uses .name to keep the legible
        # underscore form. See: cppmega_v4/buildspec/optim_spec.py
        OptimSpec(
            kind=OptimKind.ADAM_8BIT,
            groups=(ParamGroup(matcher="all", lr=3e-4),),
        )


# ---------------------------------------------------------------------------
# Existing factories continue to work (regression)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", [adamw, muon, muon_adamw_hybrid, sgd])
def test_existing_factories_unchanged(factory):
    spec = factory()
    assert isinstance(spec, OptimSpec)
    assert len(spec.groups) >= 1


# ---------------------------------------------------------------------------
# ActivationName registry
# ---------------------------------------------------------------------------


def test_activation_names_contains_eleven_entries():
    # E7-13 extended the set from 6 → 10 (added mish + geglu/reglu/xielu), and we now added glu (11 total).
    assert set(ACTIVATION_NAMES) == {
        "glu", "gelu", "relu", "relu2", "sqrelu", "silu", "mish",
        "swiglu", "geglu", "reglu", "xielu",
    }


def test_is_gated_five_gated_six_dense():
    for name in ("glu", "swiglu", "geglu", "reglu", "xielu"):
        assert is_gated(name) is True
    for name in ("gelu", "relu", "relu2", "sqrelu", "silu", "mish"):
        assert is_gated(name) is False


def test_is_gated_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown activation 'banana'"):
        is_gated("banana")


@pytest.mark.parametrize("name", ["gelu", "relu", "relu2", "sqrelu", "silu"])
def test_apply_dense_activation_shape_preserved(name):
    x = mx.random.normal((2, 4))
    y = apply_activation(name, x)
    assert y.shape == x.shape


def test_apply_activation_silu_matches_mlx_nn():
    x = mx.random.normal((4, 8))
    got = apply_activation("silu", x)
    expected = nn.silu(x)
    assert mx.allclose(got, expected).item()


def test_apply_activation_relu2_equals_relu_squared():
    x = mx.random.normal((3, 5))
    got = apply_activation("relu2", x)
    expected = mx.square(mx.maximum(x, 0))
    assert mx.allclose(got, expected).item()


def test_apply_activation_sqrelu_equals_relu2():
    x = mx.random.normal((3, 5))
    a = apply_activation("relu2", x)
    b = apply_activation("sqrelu", x)
    assert mx.allclose(a, b).item()


def test_apply_activation_swiglu_requires_gate():
    x = mx.random.normal((2, 4))
    with pytest.raises(ValueError, match="'swiglu' is gated"):
        apply_activation("swiglu", x)


def test_apply_activation_swiglu_with_gate():
    x = mx.random.normal((2, 4))
    g = mx.random.normal((2, 4))
    y = apply_activation("swiglu", x, gate=g)
    expected = nn.silu(g) * x
    assert mx.allclose(y, expected).item()


def test_apply_activation_dense_rejects_gate():
    x = mx.random.normal((2, 4))
    g = mx.random.normal((2, 4))
    with pytest.raises(ValueError, match="'gelu' is dense"):
        apply_activation("gelu", x, gate=g)


def test_apply_activation_unknown_name():
    x = mx.random.normal((1, 1))
    with pytest.raises(ValueError, match="unknown activation 'tanh'"):
        apply_activation("tanh", x)


# ---------------------------------------------------------------------------
# End-to-end: train one step with Lion via real mlx.optimizers.Lion
# ---------------------------------------------------------------------------


def test_lion_factory_drives_real_mlx_lion_step():
    """Smoke: build a tiny linear, optimise with Lion, weight delta > 0."""
    import mlx.optimizers as optim
    spec = lion(lr=1e-4)
    g = spec.groups[0]

    model = nn.Linear(8, 4)
    opt = optim.Lion(learning_rate=g.lr, betas=g.betas,
                     weight_decay=g.weight_decay)

    before = mx.array(model.weight)

    def loss_fn(m, x, target):
        return ((m(x) - target) ** 2).mean()

    x = mx.random.normal((2, 8))
    target = mx.random.normal((2, 4))
    loss, grads = nn.value_and_grad(model, loss_fn)(model, x, target)
    mx.eval(loss, grads)
    opt.update(model, grads)
    mx.eval(model.parameters(), opt.state)

    delta = mx.linalg.norm(model.weight - before).item()
    assert delta > 1e-8, f"Lion update produced no weight delta: {delta}"
    assert mx.isfinite(loss).item()

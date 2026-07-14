"""Tests for the FPRM coupling formula and the bounded-norm stability claim."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from cppmega_mlx.nn.stable_loop import StableFixedPointLoop


class _ToyCore:
    """Minimal core: each sublayer is a fixed dense map plus its own RMSNorm."""

    def __init__(self, d_model: int, n_sublayers: int, *, seed: int = 0) -> None:
        mx.random.seed(seed)
        self.norms = [nn.RMSNorm(d_model) for _ in range(n_sublayers)]
        self._weights = [
            mx.random.normal((d_model, d_model)) * 0.5 for _ in range(n_sublayers)
        ]
        self.sublayers = [
            (lambda norm, z, ctx, _w=w: norm(z) @ _w) for w in self._weights
        ]


class _NoScalingLoop(StableFixedPointLoop):
    def _loop_step_with_scales(self, z, x, ctx, scales):
        del scales
        for sublayer, norm in zip(self.core.sublayers, self.core.norms):
            z = z + sublayer(norm, z, ctx)
        return z + x


class _BigB1Loop(StableFixedPointLoop):
    def _f_theta_with_scales(self, z, ctx, scales):
        a1, _a2, _b1, _b2 = scales
        for sublayer, norm in zip(self.core.sublayers, self.core.norms):
            z = a1 * z + 100.0 * sublayer(norm, z, ctx)
        return z


def _make_loop(
    d_model=8,
    n_sublayers=4,
    a1=0.75,
    a2=0.25,
    seed=0,
    loop_cls=StableFixedPointLoop,
):
    core = _ToyCore(d_model, n_sublayers, seed=seed)
    return loop_cls(
        core,
        d_model=d_model,
        n_sublayers=n_sublayers,
        a1_init=a1,
        a2_init=a2,
    )


def _expected_b1_b2(a1: float, a2: float, two_l: int):
    a1_pow = a1**two_l
    b2 = 1.0 - a2 * a1_pow
    b1 = b2 * (1.0 - a1) / (1.0 - a1_pow)
    return b1, b2


@pytest.mark.parametrize("a1,a2", [(0.75, 0.25), (0.5, 0.5), (0.9, 0.1), (0.3, 0.8)])
@pytest.mark.parametrize("n_sublayers", [2, 4, 6])
def test_scales_match_coupling_formula(a1, a2, n_sublayers):
    loop = _make_loop(n_sublayers=n_sublayers, a1=a1, a2=a2)
    got_a1, got_a2, got_b1, got_b2 = (float(v.item()) for v in loop.scales())

    # The realized gates should match the requested inits (sigmoid round-trip).
    assert got_a1 == pytest.approx(a1, abs=1e-5)
    assert got_a2 == pytest.approx(a2, abs=1e-5)

    exp_b1, exp_b2 = _expected_b1_b2(a1, a2, n_sublayers)
    assert got_b2 == pytest.approx(exp_b2, rel=1e-5, abs=1e-6)
    assert got_b1 == pytest.approx(exp_b1, rel=1e-5, abs=1e-6)


def test_gate_constraints_enforced():
    # Requested values round-trip through the margin-scaled sigmoid.
    loop = _make_loop(a1=0.99, a2=0.99, n_sublayers=4)
    a1, a2, b1, b2 = (float(v.item()) for v in loop.scales())
    assert 0.0 < a1 < 1.0
    assert 0.0 < a2 < 1.0
    assert b2 > 0.0
    assert b1 >= 0.0


def test_extreme_logits_keep_gates_strictly_inside_margin():
    loop = _make_loop()
    loop.logit_a1 = mx.array(1e30, dtype=mx.float32)
    loop.logit_a2 = mx.array(-1e30, dtype=mx.float32)

    a1, a2 = (float(v.item()) for v in loop.gates())

    assert 0.0 < a2 <= loop.gate_margin
    assert 1.0 - loop.gate_margin <= a1 < 1.0


def test_nonfinite_gate_parameters_raise():
    loop = _make_loop()
    loop.logit_a1 = mx.array(float("inf"), dtype=mx.float32)
    with pytest.raises(FloatingPointError, match="logit_a1"):
        loop.gates()

    loop = _make_loop()
    loop.logit_a2 = mx.array(float("nan"), dtype=mx.float32)
    with pytest.raises(FloatingPointError, match="logit_a2"):
        loop.gates()


@pytest.mark.parametrize("gate_margin", [0.0, 1e-8, 0.5, float("nan")])
def test_invalid_gate_margin_raises(gate_margin):
    core = _ToyCore(8, 4, seed=0)
    with pytest.raises(ValueError, match="gate_margin"):
        StableFixedPointLoop(
            core,
            d_model=8,
            n_sublayers=4,
            gate_margin=gate_margin,
        )


@pytest.mark.parametrize("init", [0.0, 1.0, -0.1, 1.5])
def test_invalid_gate_init_raises(init):
    with pytest.raises(ValueError):
        _make_loop(a1=init)


# TIGHT bound: with the *correct* derived coupling the fixed point sits at
# roughly input scale (empirically ratio z_norm/x_norm <= ~1.5 across seeds and
# gate settings). We assert a tight multiple so the test is NOT trivially true:
#
#   * Removing residual scaling entirely (a1=b1=a2=b2=1) makes ||z||_inf grow
#     geometrically in T (~217x at T=32) -> FAILS this bound.
#   * Keeping a1<1 but using a WRONG b1 (e.g. b1=100 instead of the derived
#     value) converges in T but to ~51x the input norm -> ALSO FAILS this bound.
#
# Only the derived coupling b1 = b2*(1-a1)/(1-a1**2L), b2 = 1-a2*a1**2L keeps
# the converged norm near input scale. See the ablation in the module docstring.
_BOUNDED_NORM_RATIO = 8.0


@pytest.mark.parametrize("T", [1, 2, 4, 8, 16, 32])
def test_activation_inf_norm_bounded_across_T(T):
    """Core stability claim: L-inf norm stays bounded (near input scale) for all T.

    With the derived coupling, a fixed point exists and ||z||_inf must NOT blow
    up as we increase the number of loop iterations T. The bound is TIGHT
    (``_BOUNDED_NORM_RATIO`` * input norm) so that a removed-or-wrong residual
    scaling would fail it (see ``test_*_is_real_*`` below for the ablations).
    Large T (>=16) is exercised explicitly via the parametrization.
    """
    d_model, n_sublayers = 8, 4
    loop = _make_loop(d_model=d_model, n_sublayers=n_sublayers, seed=7)

    mx.random.seed(123)
    x = mx.random.normal((2, 5, d_model))
    x_norm = float(mx.max(mx.abs(x)).item())

    z = loop.forward(x, x, None, training_loops=T)
    z_norm = float(mx.max(mx.abs(z)).item())

    assert mx.isfinite(z).all().item(), f"z became non-finite at T={T}"
    # TIGHT bound near input scale, independent of T. A geometric blow-up
    # (scaling removed) or a wrong-magnitude b1 both violate this.
    assert z_norm <= _BOUNDED_NORM_RATIO * x_norm + 2.0, (
        f"L-inf norm exceeded tight bound at T={T}: {z_norm} vs input {x_norm}"
    )


def _run_norm_curve(loop_cls, *, seed=7):
    """Drive an explicit scaling ablation, returning T->||z||."""
    loop = _make_loop(d_model=8, n_sublayers=4, seed=seed, loop_cls=loop_cls)
    mx.random.seed(123)
    x = mx.random.normal((2, 5, 8))
    x_norm = float(mx.max(mx.abs(x)).item())
    norms = {}
    for T in (1, 2, 4, 8, 16, 32):
        z = loop.forward(x, x, None, training_loops=T)
        norms[T] = float(mx.max(mx.abs(z)).item())
    return x_norm, norms


def test_bounded_norm_test_is_real_no_scaling_blows_up():
    """ADVERSARIAL: deleting residual scaling (a1=b1=a2=b2=1) blows the norm up.

    This proves ``test_activation_inf_norm_bounded_across_T`` is NOT trivially
    true: without the derived coupling the L-inf norm grows geometrically in T
    and exceeds the tight bound the real test asserts.
    """

    x_norm, norms = _run_norm_curve(_NoScalingLoop)
    bound = _BOUNDED_NORM_RATIO * x_norm + 2.0
    # Grows with T and busts the bound at large T.
    assert norms[32] > norms[1], norms
    assert norms[32] > bound, (
        f"ablation should bust the bound {bound:.2f} but got {norms[32]:.2f}"
    )


def test_bounded_norm_test_is_real_wrong_b1_busts_tight_bound():
    """ADVERSARIAL: keeping a1<1 but a WRONG b1 converges yet far above input.

    This proves the bound is TIGHT (not just catching divergence): a wrong b1
    still converges in T (because a1<1 contracts), but to ~50x input scale,
    which the tight bound rejects. Only the derived b1 keeps it near input.
    """

    x_norm, norms = _run_norm_curve(_BigB1Loop)
    bound = _BOUNDED_NORM_RATIO * x_norm + 2.0
    # Converges in T (a1<1 still contracts) ...
    assert abs(norms[32] - norms[16]) < 1.0, norms
    # ... but to a value far above the tight bound the real coupling satisfies.
    assert norms[32] > bound, (
        f"wrong-b1 norm {norms[32]:.2f} should exceed tight bound {bound:.2f}"
    )


def test_inf_norm_does_not_grow_monotonically_with_T():
    """Stronger bounded check: norm at large T is not wildly above small T."""
    d_model, n_sublayers = 8, 4
    loop = _make_loop(d_model=d_model, n_sublayers=n_sublayers, seed=11)
    mx.random.seed(321)
    x = mx.random.normal((2, 5, d_model))

    x_norm = float(mx.max(mx.abs(x)).item())
    norms = {}
    for T in (1, 2, 4, 8, 16, 32):
        z = loop.forward(x, x, None, training_loops=T)
        norms[T] = float(mx.max(mx.abs(z)).item())

    # The norm should CONVERGE: once T is large the value stops moving. The
    # tail values (T>=8) must agree to a tight tolerance (fixed point reached).
    tail = [norms[T] for T in (8, 16, 32)]
    assert max(tail) - min(tail) <= 1e-3, f"norm not converged in tail: {norms}"
    # And the converged value stays near input scale (tight bound), not merely
    # finite — a wrong coupling would converge far above this.
    assert max(norms.values()) <= 8.0 * x_norm + 2.0, f"norm too large: {norms}"

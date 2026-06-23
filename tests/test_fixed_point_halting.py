"""Tests for fixed-point residual decrease, halting under tau, and FPOPT."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from cppmega_mlx.nn.stable_loop import StableFixedPointLoop
from cppmega_mlx.training.fixed_point import fpopt_step


class _ContractingCore:
    """A deliberately contracting core so the loop converges to a fixed point.

    Each sublayer is identity-on-norm scaled small, so f_theta is a mild
    perturbation and the iteration mix (a2, b2) pulls the state toward a unique
    fixed point. With small a1/a2 the map is a contraction in L-inf.
    """

    def __init__(self, d_model: int, n_sublayers: int) -> None:
        self.norms = [nn.RMSNorm(d_model) for _ in range(n_sublayers)]
        # delta = 0.1 * norm(z): small, smooth, contraction-friendly.
        self.sublayers = [
            (lambda norm, z, ctx: 0.1 * norm(z)) for _ in range(n_sublayers)
        ]


def _make_converging_loop(d_model=8, n_sublayers=4):
    core = _ContractingCore(d_model, n_sublayers)
    # a1 close to 1 makes the within-block contraction gentle, so the loop
    # takes several iterations to reach the fixed point (instead of one),
    # exercising the residual-decrease trajectory.
    return StableFixedPointLoop(
        core,
        d_model=d_model,
        n_sublayers=n_sublayers,
        a1_init=0.9,
        a2_init=0.6,
        tau=0.1,
        max_loops=64,
    )


def test_residual_decreases_on_converging_map():
    loop = _make_converging_loop()
    mx.random.seed(5)
    x = mx.random.normal((2, 4, 8))

    # Start the fixed-point iteration far from the attractor (zeros) so the
    # residual must descend over multiple steps to reach the fixed point.
    z0 = mx.zeros_like(x)
    z, info = loop.forward(z0, x, None, collect_residuals=True)
    residuals = info["residuals"]

    assert len(residuals) >= 2
    # Overall the residual must trend down: the last is well below the first.
    assert residuals[-1] < residuals[0]
    # Monotone (allowing tiny numerical wiggle) decrease for a contraction.
    for prev, nxt in zip(residuals, residuals[1:]):
        assert nxt <= prev + 1e-4, f"residual increased: {residuals}"


def test_halting_triggers_under_tau():
    loop = _make_converging_loop()
    mx.random.seed(6)
    x = mx.random.normal((2, 4, 8))

    z, info = loop.forward(mx.zeros_like(x), x, None, collect_residuals=True)

    assert info["halted"] is True, info
    # The final recorded residual is the one that dipped below tau.
    assert info["residuals"][-1] < loop.tau
    assert info["steps"] <= loop.max_loops


def test_no_halt_when_tau_unreachable():
    """With a gentle contraction (a1 near 1) capped at a small max_loops, a
    tight tau is NOT reached within budget, so halting must not fire."""
    core = _ContractingCore(8, 4)
    loop = StableFixedPointLoop(
        core,
        d_model=8,
        n_sublayers=4,
        a1_init=0.95,  # very gentle within-block contraction -> slow descent
        a2_init=0.9,
        tau=1e-4,
        max_loops=3,  # too few steps to drive the residual below tau
    )
    mx.random.seed(9)
    x = mx.random.normal((2, 4, 8))
    z, info = loop.forward(mx.zeros_like(x), x, None, collect_residuals=True)
    assert all(r >= loop.tau for r in info["residuals"]), info["residuals"]
    assert info["halted"] is False
    assert info["steps"] == 3


def test_fpopt_step_reduces_residual_over_iterations():
    loop = _make_converging_loop()
    mx.random.seed(8)
    x = mx.random.normal((2, 4, 8))

    z = mx.zeros_like(x)
    residuals = []
    for _ in range(40):
        z, r = fpopt_step(loop, z, x, None, eta=1.0)
        residuals.append(r)
    assert residuals[-1] < residuals[0]
    assert residuals[-1] < loop.tau


def test_fpopt_step_invalid_eta_raises():
    loop = _make_converging_loop()
    x = mx.zeros((1, 2, 8))
    with pytest.raises(ValueError):
        fpopt_step(loop, x, x, None, eta=0.0)
    with pytest.raises(ValueError):
        fpopt_step(loop, x, x, None, eta=1.5)


def test_fpopt_damping_gamma_keeps_eta_at_one_when_gamma_one():
    """gamma == 1.0 is the paper's best: eta should remain 1.0 even on stalls."""
    loop = _make_converging_loop()
    mx.random.seed(10)
    x = mx.random.normal((2, 4, 8))
    z, info = loop.forward(
        x, x, None, collect_residuals=True, fpopt_gamma=1.0, fpopt_eta0=1.0
    )
    assert info["eta"] == pytest.approx(1.0)

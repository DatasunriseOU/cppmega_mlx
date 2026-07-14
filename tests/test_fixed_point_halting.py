"""Tests for fixed-point residual decrease, halting under tau, and FPOPT."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from cppmega_mlx.nn.stable_loop import (
    FixedPointConvergenceError,
    FixedPointConvergenceResult,
    StableFixedPointLoop,
)
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


class _OscillatingLoop(StableFixedPointLoop):
    """Deterministic period-two map used to exercise FPOPT damping."""

    def residual_map(self, z, x, ctx):
        del x, ctx
        return -z


class _NonfiniteCore:
    def __init__(self) -> None:
        self.norms = [nn.RMSNorm(2), nn.RMSNorm(2)]
        self.sublayers = [
            lambda norm, z, ctx: norm(z) * float("nan"),
            lambda norm, z, ctx: norm(z),
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


def _make_oscillating_loop(*, max_loops=4, tau=1e-6):
    return _OscillatingLoop(
        _ContractingCore(2, 2),
        d_model=2,
        n_sublayers=2,
        tau=tau,
        max_loops=max_loops,
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


def test_max_loop_exhaustion_raises_typed_error_by_default():
    """A finite but non-converged state must not escape strict inference."""
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
    with pytest.raises(FixedPointConvergenceError) as exc_info:
        loop.forward(mx.zeros_like(x), x, None, collect_residuals=True)

    result = exc_info.value.result
    assert isinstance(result, FixedPointConvergenceResult)
    assert all(r >= loop.tau for r in result.residuals)
    assert result.converged is False
    assert result.steps == 3


def test_max_loop_exhaustion_requires_explicit_best_effort():
    loop = _make_oscillating_loop(max_loops=2)
    z0 = mx.ones((1, 2))

    result = loop.forward(
        z0,
        mx.zeros_like(z0),
        None,
        return_convergence=True,
        best_effort=True,
    )

    assert isinstance(result, FixedPointConvergenceResult)
    assert result.converged is False
    assert result.steps == 2
    assert result.final_residual >= result.tau


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


def test_fpopt_stall_decays_eta_and_increases_damping():
    loop = _make_oscillating_loop(max_loops=4, tau=1e-8)
    z0 = mx.ones((1, 2))

    result = loop.forward(
        z0,
        mx.zeros_like(z0),
        None,
        fpopt_patience=1,
        fpopt_gamma=0.5,
        fpopt_eta0=1.0,
        return_convergence=True,
    )

    assert isinstance(result, FixedPointConvergenceResult)
    assert result.converged is True
    assert result.eta == pytest.approx(0.5)
    assert result.residuals[:2] == pytest.approx((2.0, 2.0), rel=1e-5)


def test_nonfinite_initial_state_raises_in_training_and_inference():
    loop = _make_converging_loop()
    bad_state = mx.array([[float("nan")] * 8])
    x = mx.zeros_like(bad_state)

    with pytest.raises(FloatingPointError, match="initial_state"):
        loop.forward(bad_state, x, None, training_loops=1)
    with pytest.raises(FloatingPointError, match="initial_state"):
        loop.forward(bad_state, x, None)


def test_nonfinite_generated_state_raises_in_training_and_inference():
    loop = StableFixedPointLoop(
        _NonfiniteCore(),
        d_model=2,
        n_sublayers=2,
        max_loops=2,
    )
    z = mx.ones((1, 2))

    with pytest.raises(FloatingPointError, match="training.output_state"):
        loop.forward(z, z, None, training_loops=1)
    with pytest.raises(FloatingPointError, match="loop_step.output_state"):
        loop.forward(z, z, None)


def test_nonfinite_residual_input_raises():
    loop = _make_converging_loop(d_model=2, n_sublayers=2)
    z = mx.ones((1, 2))
    bad_mapped_state = mx.array([[float("inf"), 0.0]])

    with pytest.raises(FloatingPointError, match="mapped_state"):
        loop.relative_residual(z, bad_mapped_state)

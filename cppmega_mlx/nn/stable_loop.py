"""FPRM stable looped core (arXiv 2606.18206), MLX reference implementation.

This module implements the *Fixed-Point Recurrent Model* (FPRM) stable looped
core described in arXiv:2606.18206. One loop iteration runs a single
weight-tied block over ``2L`` sublayers (``L`` layers, each contributing an
``{attn, FFN}`` pair). Every sublayer is pre-norm with TWO learnable scalar
gates::

    z^l = a1 * z^{l-1} + b1 * f^l(Norm(z^{l-1}))

and the iteration mix between consecutive loop steps is::

    z0_{i+1} = a2 * z^{2L}_i + b2 * x

Only ``a1`` and ``a2`` are free parameters. ``b1`` and ``b2`` are DERIVED each
step from the coupling formula (they are never learned independently)::

    b2 = 1 - a2 * a1**(2L)
    b1 = b2 * (1 - a1) / (1 - a1**(2L))

with ``0 < a1 < 1`` and ``0 < a2 < 1`` (enforced via a margin-scaled sigmoid on
the stored logits). This coupling is precisely what keeps the activation
L-infinity norm BOUNDED *near input scale* as the unrolled depth ``T`` grows —
the core stability claim of the paper.

Ablation (verified in ``tests/test_stable_loop_scaling.py``), toy core,
input ``||x||_inf == 2.29``:

* Derived coupling: ``||z||_inf`` converges to ~2.62 (~1.14x input), flat for
  all T from 8 upward.
* Remove residual scaling (``a1=b1=a2=b2=1``): ``||z||_inf`` grows
  geometrically — 8.9 -> 15 -> 27 -> 57 -> 113 -> 216 across T in {1..32}.
* Keep ``a1<1`` but a WRONG ``b1=100``: converges in T (a1 still contracts) but
  to ~118 (~51x input).

Only the derived ``b1``/``b2`` keep the converged norm near input scale; the
tight bounded-norm test rejects both ablations.

Inference uses fixed-point halting + FPOPT damping; training uses a fixed
number of loop steps (truncated-BPTT windows are driven from
``cppmega_mlx.training.fixed_point``).

This is a correctness-first reference. It does not claim production kernel
performance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import mlx.core as mx
import mlx.nn as nn

# A sublayer is a callable mapping ``(norm_module, x, ctx) -> delta``. The
# caller supplies the pre-norm module so the loop owns the residual scaling
# while the core owns the per-sublayer transform ``f^l``.
Sublayer = Callable[[nn.Module, mx.array, object], mx.array]


@dataclass(frozen=True)
class FixedPointConvergenceResult:
    """Auditable outcome of one fixed-point inference run."""

    state: mx.array
    residuals: tuple[float, ...]
    steps: int
    converged: bool
    eta: float
    tau: float

    @property
    def final_residual(self) -> float:
        """Residual observed on the final attempted iteration."""
        return self.residuals[-1]

    def to_info(self) -> dict[str, object]:
        """Return the legacy diagnostics mapping used by ``collect_residuals``."""
        return {
            "residuals": list(self.residuals),
            "steps": self.steps,
            "halted": self.converged,
            "converged": self.converged,
            "eta": self.eta,
            "mode": "inference",
            "best_effort": not self.converged,
        }


class FixedPointConvergenceError(RuntimeError):
    """Raised when fixed-point inference exhausts its iteration budget."""

    def __init__(self, result: FixedPointConvergenceResult) -> None:
        self.result = result
        super().__init__(
            "StableFixedPointLoop inference did not converge after "
            f"{result.steps} steps: final_residual={result.final_residual:.8g} "
            f">= tau={result.tau:.8g}; pass best_effort=True only when an "
            "explicitly non-converged state is acceptable"
        )


class StableLoopCore(Protocol):
    """Protocol for the weight-tied core driven by :class:`StableFixedPointLoop`.

    ``sublayers`` is the ordered list of ``2L`` callables (one per sublayer).
    Each callable receives ``(norm, x, ctx)`` and returns the pre-residual
    delta ``f^l(Norm(x))``. The core owns the sublayer weights AND the per-
    sublayer norm modules; the loop only owns the scalar gates.
    """

    sublayers: Sequence[Sublayer]
    norms: Sequence[nn.Module]


def _inf_norm(x: mx.array) -> mx.array:
    """L-infinity norm (max absolute element) as a 0-d MLX scalar."""
    return mx.max(mx.abs(x))


def _require_finite_tensor(x: mx.array, *, where: str) -> None:
    """Fail immediately when an MLX tensor contains NaN or infinity."""
    if not bool(mx.all(mx.isfinite(x)).item()):
        raise FloatingPointError(f"{where}: tensor contains non-finite values")


def _require_finite_float(value: float, *, where: str) -> None:
    if not math.isfinite(value):
        raise FloatingPointError(f"{where}: value is non-finite ({value})")


class StableFixedPointLoop(nn.Module):
    """Weight-tied stable looped core with derived residual coupling.

    Parameters
    ----------
    core:
        Object exposing ``sublayers`` (``2L`` callables) and ``norms`` (``2L``
        pre-norm modules). The core holds all transform weights; this loop
        only holds the two scalar gate logits.
    d_model:
        Hidden size (used only for validation / introspection).
    n_sublayers:
        ``2L`` — the number of sublayers per loop iteration. Must match
        ``len(core.sublayers)`` and ``len(core.norms)``.
    a1_init, a2_init:
        Initial values of the two free gates (paper defaults 0.75 / 0.25).
        Must lie strictly inside the configured gate margin.
    tau:
        Fixed-point halting threshold for the relative L-inf residual.
    max_loops:
        Maximum number of loop iterations during inference fixed-point
        iteration before non-convergence raises.
    eps:
        Numerical floor used in the relative residual denominator.
    gate_margin:
        Strict distance maintained between each realized trainable gate and
        the endpoints zero and one, including when logits saturate.
    """

    def __init__(
        self,
        core: StableLoopCore,
        d_model: int,
        n_sublayers: int,
        *,
        a1_init: float = 0.75,
        a2_init: float = 0.25,
        tau: float = 0.1,
        max_loops: int = 32,
        eps: float = 1e-6,
        gate_margin: float = 1e-4,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if n_sublayers <= 0:
            raise ValueError(f"n_sublayers must be positive, got {n_sublayers}")
        sublayers = list(getattr(core, "sublayers"))
        norms = list(getattr(core, "norms"))
        if len(sublayers) != n_sublayers:
            raise ValueError(
                f"core.sublayers has {len(sublayers)} entries but n_sublayers="
                f"{n_sublayers}"
            )
        if len(norms) != n_sublayers:
            raise ValueError(
                f"core.norms has {len(norms)} entries but n_sublayers={n_sublayers}"
            )
        if not math.isfinite(gate_margin) or not 1e-6 <= gate_margin < 0.5:
            raise ValueError(
                f"gate_margin must be finite and in [1e-6, 0.5), got {gate_margin}"
            )
        for name, value in (("a1_init", a1_init), ("a2_init", a2_init)):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
            if not gate_margin < value < 1.0 - gate_margin:
                raise ValueError(
                    f"{name} must be in ({gate_margin}, {1.0 - gate_margin}), "
                    f"got {value}"
                )
        if not math.isfinite(tau) or tau <= 0.0:
            raise ValueError(f"tau must be finite and positive, got {tau}")
        if max_loops <= 0:
            raise ValueError(f"max_loops must be positive, got {max_loops}")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError(f"eps must be finite and positive, got {eps}")

        self.core = core
        self.d_model = int(d_model)
        self.n_sublayers = int(n_sublayers)
        self.tau = float(tau)
        self.max_loops = int(max_loops)
        self.eps = float(eps)
        self.gate_margin = float(gate_margin)

        # Only a1, a2 are free. We store them as unconstrained logits and pass
        # them through a margin-scaled sigmoid so even saturated logits stay a
        # representable, strict distance from both zero and one.
        self.logit_a1 = mx.array(
            _inverse_margin_sigmoid(a1_init, self.gate_margin), dtype=mx.float32
        )
        self.logit_a2 = mx.array(
            _inverse_margin_sigmoid(a2_init, self.gate_margin), dtype=mx.float32
        )

    def gates(self) -> tuple[mx.array, mx.array]:
        """Return finite free gates strictly inside ``(0, 1)``."""
        for name, logit in (
            ("logit_a1", self.logit_a1),
            ("logit_a2", self.logit_a2),
        ):
            if logit.shape != ():
                raise ValueError(
                    f"StableFixedPointLoop.gates: {name} must be scalar, got "
                    f"shape {logit.shape}"
                )

        logits = mx.stack(
            [self.logit_a1.astype(mx.float32), self.logit_a2.astype(mx.float32)]
        )
        gates = self.gate_margin + (1.0 - 2.0 * self.gate_margin) * mx.sigmoid(logits)
        combined = mx.concatenate([logits, gates])
        if not bool(mx.all(mx.isfinite(combined)).item()):
            _require_finite_tensor(
                self.logit_a1, where="StableFixedPointLoop.gates.logit_a1"
            )
            _require_finite_tensor(
                self.logit_a2, where="StableFixedPointLoop.gates.logit_a2"
            )
            _require_finite_tensor(gates, where="StableFixedPointLoop.gates.output")
        return gates[0], gates[1]

    def scales(self) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """Return ``(a1, a2, b1, b2)`` with ``b1, b2`` derived via coupling.

        Coupling (verbatim from arXiv:2606.18206), with ``2L = n_sublayers``::

            b2 = 1 - a2 * a1**(2L)
            b1 = b2 * (1 - a1) / (1 - a1**(2L))
        """
        a1, a2 = self.gates()
        two_l = self.n_sublayers
        a1_pow = a1**two_l
        b2 = 1.0 - a2 * a1_pow
        # The gate margin keeps this denominator strictly positive even when a
        # trainable logit saturates, so the coupling remains the exact formula.
        denom = 1.0 - a1_pow
        b1 = b2 * (1.0 - a1) / denom
        _require_finite_tensor(
            mx.stack([a1, a2, b1, b2]), where="StableFixedPointLoop.scales"
        )
        return a1, a2, b1, b2

    def f_theta(self, z: mx.array, x: mx.array, ctx: object) -> mx.array:
        """Apply the ``2L`` sublayers once with derived residual scaling.

        Implements, for ``l = 1 .. 2L``::

            z^l = a1 * z^{l-1} + b1 * f^l(Norm(z^{l-1}))

        starting from ``z^0 = z``. ``x`` is unused inside one block sweep (it
        only enters the iteration mix); it is accepted so the signature matches
        the residual-map convention ``f(z; x)``.
        """
        del x  # x participates only in the inter-iteration mix, not within f.
        _require_finite_tensor(z, where="StableFixedPointLoop.f_theta.input_state")
        scales = self.scales()
        z = self._f_theta_with_scales(z, ctx, scales)
        _require_finite_tensor(z, where="StableFixedPointLoop.f_theta.output_state")
        return z

    def _f_theta_with_scales(
        self,
        z: mx.array,
        ctx: object,
        scales: tuple[mx.array, mx.array, mx.array, mx.array],
    ) -> mx.array:
        a1, _a2, b1, _b2 = scales
        sublayers = list(self.core.sublayers)
        norms = list(self.core.norms)
        for sublayer, norm in zip(sublayers, norms):
            delta = sublayer(norm, z, ctx)
            z = a1 * z + b1 * delta
        return z

    def iteration_mix(self, z_block: mx.array, x: mx.array) -> mx.array:
        """Combine the post-block state with the injected input ``x``.

        ``z0_{i+1} = a2 * z^{2L}_i + b2 * x``.
        """
        _require_finite_tensor(
            z_block, where="StableFixedPointLoop.iteration_mix.block_state"
        )
        _require_finite_tensor(x, where="StableFixedPointLoop.iteration_mix.input")
        mixed = self._iteration_mix_with_scales(z_block, x, self.scales())
        _require_finite_tensor(
            mixed, where="StableFixedPointLoop.iteration_mix.output_state"
        )
        return mixed

    @staticmethod
    def _iteration_mix_with_scales(
        z_block: mx.array,
        x: mx.array,
        scales: tuple[mx.array, mx.array, mx.array, mx.array],
    ) -> mx.array:
        _a1, a2, _b1, b2 = scales
        return a2 * z_block + b2 * x

    def loop_step(self, z: mx.array, x: mx.array, ctx: object) -> mx.array:
        """One full FPRM loop iteration: block sweep then iteration mix."""
        _require_finite_tensor(z, where="StableFixedPointLoop.loop_step.input_state")
        _require_finite_tensor(x, where="StableFixedPointLoop.loop_step.input")
        mixed = self._loop_step_with_scales(z, x, ctx, self.scales())
        _require_finite_tensor(
            mixed, where="StableFixedPointLoop.loop_step.output_state"
        )
        return mixed

    def _loop_step_with_scales(
        self,
        z: mx.array,
        x: mx.array,
        ctx: object,
        scales: tuple[mx.array, mx.array, mx.array, mx.array],
    ) -> mx.array:
        z_block = self._f_theta_with_scales(z, ctx, scales)
        return self._iteration_mix_with_scales(z_block, x, scales)

    def residual_map(self, z: mx.array, x: mx.array, ctx: object) -> mx.array:
        """The full fixed-point map ``f(z; x)`` = one loop iteration output.

        A fixed point ``z*`` satisfies ``z* = residual_map(z*; x)``.
        """
        return self.loop_step(z, x, ctx)

    def relative_residual(self, z: mx.array, f_z: mx.array) -> mx.array:
        """Relative L-inf residual ``||z - f(z)||_inf / (||f(z)||_inf + eps)``."""
        _require_finite_tensor(z, where="StableFixedPointLoop.relative_residual.state")
        _require_finite_tensor(
            f_z, where="StableFixedPointLoop.relative_residual.mapped_state"
        )
        residual = _inf_norm(z - f_z) / (_inf_norm(f_z) + self.eps)
        _require_finite_tensor(
            residual, where="StableFixedPointLoop.relative_residual.output"
        )
        return residual

    def forward(
        self,
        z0: mx.array,
        x: mx.array,
        ctx: object = None,
        *,
        training_loops: int | None = None,
        fpopt_patience: int = 5,
        fpopt_gamma: float = 1.0,
        fpopt_eta0: float = 1.0,
        collect_residuals: bool = False,
        return_convergence: bool = False,
        best_effort: bool = False,
    ) -> mx.array | tuple[mx.array, dict[str, object]] | FixedPointConvergenceResult:
        """Run the stable loop.

        Two distinct modes:

        * ``training_loops`` is not ``None`` -> run EXACTLY that many loop
          iterations (no halting, fully differentiable). This is the path used
          by truncated-BPTT windows.
        * ``training_loops`` is ``None`` -> inference fixed-point iteration with
          relative-residual halting (threshold ``tau``) and FPOPT damping.

        When ``collect_residuals`` is ``True`` the method returns
        ``(z, info)`` where ``info`` carries the per-step residual trace, the
        number of steps run, and whether halting fired under ``tau``.

        Inference raises :class:`FixedPointConvergenceError` when ``max_loops``
        is exhausted. ``best_effort=True`` is the only path that returns a
        non-converged state. ``return_convergence=True`` preserves the typed
        convergence result for higher-level model APIs.
        """
        _require_finite_tensor(z0, where="StableFixedPointLoop.forward.initial_state")
        _require_finite_tensor(x, where="StableFixedPointLoop.forward.input")
        if collect_residuals and return_convergence:
            raise ValueError(
                "collect_residuals and return_convergence are mutually exclusive"
            )
        if training_loops is not None:
            if best_effort:
                raise ValueError("best_effort is only valid for fixed-point inference")
            if return_convergence:
                raise ValueError(
                    "return_convergence is only valid for fixed-point inference"
                )
            if training_loops <= 0:
                raise ValueError(
                    f"training_loops must be positive, got {training_loops}"
                )
            z = z0
            scales = self.scales()
            residuals: list[float] = []
            for _ in range(training_loops):
                f_z = self._loop_step_with_scales(z, x, ctx, scales)
                _require_finite_tensor(
                    f_z, where="StableFixedPointLoop.training.output_state"
                )
                if collect_residuals:
                    residuals.append(float(self.relative_residual(z, f_z).item()))
                z = f_z
            if collect_residuals:
                info: dict[str, object] = {
                    "residuals": residuals,
                    "steps": training_loops,
                    "halted": False,
                    "mode": "training",
                }
                return z, info
            return z

        # Inference: fixed-point iteration with FPOPT damping + halting.
        return self._fixed_point_infer(
            z0,
            x,
            ctx,
            fpopt_patience=fpopt_patience,
            fpopt_gamma=fpopt_gamma,
            fpopt_eta0=fpopt_eta0,
            collect_residuals=collect_residuals,
            return_convergence=return_convergence,
            best_effort=best_effort,
        )

    def _fixed_point_infer(
        self,
        z0: mx.array,
        x: mx.array,
        ctx: object,
        *,
        fpopt_patience: int,
        fpopt_gamma: float,
        fpopt_eta0: float,
        collect_residuals: bool,
        return_convergence: bool,
        best_effort: bool,
    ) -> mx.array | tuple[mx.array, dict[str, object]] | FixedPointConvergenceResult:
        if fpopt_patience <= 0:
            raise ValueError(f"fpopt_patience must be positive, got {fpopt_patience}")
        if not 0.0 < fpopt_gamma <= 1.0:
            raise ValueError(f"fpopt_gamma must be in (0, 1], got {fpopt_gamma}")
        if not 0.0 < fpopt_eta0 <= 1.0:
            raise ValueError(f"fpopt_eta0 must be in (0, 1], got {fpopt_eta0}")

        z = z0
        eta = fpopt_eta0
        best_residual = math.inf
        stall = 0
        residuals = []
        halted = False
        steps = 0
        for step in range(self.max_loops):
            steps = step + 1
            f_tilde = self.residual_map(z, x, ctx)
            residual = float(self.relative_residual(z, f_tilde).item())
            _require_finite_float(
                residual, where="StableFixedPointLoop.inference.residual"
            )
            residuals.append(residual)
            if residual < self.tau:
                halted = True
                # Take the damped step then stop.
                z = eta * f_tilde + (1.0 - eta) * z
                _require_finite_tensor(
                    z, where="StableFixedPointLoop.inference.converged_state"
                )
                break
            # FPOPT damping: on a stall (no improvement), geometrically decay
            # eta. Smaller eta retains more of the old state and damps harder.
            if residual < best_residual - self.eps:
                best_residual = residual
                stall = 0
            else:
                stall += 1
                if stall >= fpopt_patience:
                    eta = eta * fpopt_gamma
                    if not math.isfinite(eta) or eta <= 0.0:
                        raise FloatingPointError(
                            "StableFixedPointLoop.inference.eta: damping decayed "
                            f"to an invalid value ({eta})"
                        )
                    stall = 0
            z = eta * f_tilde + (1.0 - eta) * z
            _require_finite_tensor(
                z, where="StableFixedPointLoop.inference.updated_state"
            )

        result = FixedPointConvergenceResult(
            state=z,
            residuals=tuple(residuals),
            steps=steps,
            converged=halted,
            eta=eta,
            tau=self.tau,
        )
        if not halted and not best_effort:
            raise FixedPointConvergenceError(result)
        if return_convergence:
            return result
        if collect_residuals:
            return z, result.to_info()
        return z

    def __call__(
        self,
        z0: mx.array,
        x: mx.array,
        ctx: object = None,
        *,
        training_loops: int | None = None,
        **kwargs: object,
    ) -> mx.array | tuple[mx.array, dict[str, object]] | FixedPointConvergenceResult:
        return self.forward(z0, x, ctx, training_loops=training_loops, **kwargs)


def _inverse_sigmoid(p: float) -> float:
    """Logit of ``p`` so ``sigmoid(logit) == p`` for ``p`` in ``(0, 1)``."""
    if not 0.0 < p < 1.0:
        # Exact endpoints have no finite logit and cannot satisfy the strict
        # gate-margin contract.
        raise ValueError(f"gate init must be in the open interval (0, 1), got {p}")
    return math.log(p / (1.0 - p))


def _inverse_margin_sigmoid(p: float, margin: float) -> float:
    unit_p = (p - margin) / (1.0 - 2.0 * margin)
    return _inverse_sigmoid(unit_p)


__all__ = [
    "FixedPointConvergenceError",
    "FixedPointConvergenceResult",
    "StableFixedPointLoop",
    "StableLoopCore",
    "Sublayer",
]

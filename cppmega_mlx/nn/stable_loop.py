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

with ``0 <= a1 < 1`` and ``0 <= a2 < 1`` (enforced via a sigmoid on the stored
logits). This coupling is precisely what keeps the activation L-infinity norm
BOUNDED *near input scale* as the unrolled depth ``T`` grows — the core
stability claim of the paper.

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
from typing import Callable, Protocol, Sequence

import mlx.core as mx
import mlx.nn as nn

# A sublayer is a callable mapping ``(norm_module, x, ctx) -> delta``. The
# caller supplies the pre-norm module so the loop owns the residual scaling
# while the core owns the per-sublayer transform ``f^l``.
Sublayer = Callable[[nn.Module, mx.array, object], mx.array]


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
        Must lie in ``[0, 1)``.
    tau:
        Fixed-point halting threshold for the relative L-inf residual.
    max_loops:
        Maximum number of loop iterations during inference fixed-point
        iteration before halting is forced.
    eps:
        Numerical floor used in the relative residual denominator.
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
                f"core.norms has {len(norms)} entries but n_sublayers="
                f"{n_sublayers}"
            )
        if not 0.0 <= a1_init < 1.0:
            raise ValueError(f"a1_init must be in [0, 1), got {a1_init}")
        if not 0.0 <= a2_init < 1.0:
            raise ValueError(f"a2_init must be in [0, 1), got {a2_init}")
        if tau <= 0.0:
            raise ValueError(f"tau must be positive, got {tau}")
        if max_loops <= 0:
            raise ValueError(f"max_loops must be positive, got {max_loops}")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.core = core
        self.d_model = int(d_model)
        self.n_sublayers = int(n_sublayers)
        self.tau = float(tau)
        self.max_loops = int(max_loops)
        self.eps = float(eps)

        # Only a1, a2 are free. We store them as unconstrained logits and pass
        # them through a sigmoid so the realized gates always satisfy the
        # 0 <= a < 1 constraint without any clamping in the forward path.
        self.logit_a1 = mx.array(_inverse_sigmoid(a1_init), dtype=mx.float32)
        self.logit_a2 = mx.array(_inverse_sigmoid(a2_init), dtype=mx.float32)

    def gates(self) -> tuple[mx.array, mx.array]:
        """Return the realized free gates ``(a1, a2)`` in ``[0, 1)``."""
        a1 = mx.sigmoid(self.logit_a1)
        a2 = mx.sigmoid(self.logit_a2)
        return a1, a2

    def scales(self) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """Return ``(a1, a2, b1, b2)`` with ``b1, b2`` derived via coupling.

        Coupling (verbatim from arXiv:2606.18206), with ``2L = n_sublayers``::

            b2 = 1 - a2 * a1**(2L)
            b1 = b2 * (1 - a1) / (1 - a1**(2L))
        """
        a1, a2 = self.gates()
        two_l = self.n_sublayers
        a1_pow = a1 ** two_l
        b2 = 1.0 - a2 * a1_pow
        # 1 - a1**(2L) is strictly positive for a1 in [0, 1); the eps floor only
        # guards the a1 -> 0 / 2L large numeric edge and never changes the
        # mathematical value within machine precision.
        denom = 1.0 - a1_pow
        b1 = b2 * (1.0 - a1) / (denom + self.eps)
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
        _a1, _a2, b1, _b2 = self.scales()
        a1, _ = self.gates()
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
        _a1, a2, _b1, b2 = self.scales()
        return a2 * z_block + b2 * x

    def loop_step(self, z: mx.array, x: mx.array, ctx: object) -> mx.array:
        """One full FPRM loop iteration: block sweep then iteration mix."""
        z_block = self.f_theta(z, x, ctx)
        return self.iteration_mix(z_block, x)

    def residual_map(self, z: mx.array, x: mx.array, ctx: object) -> mx.array:
        """The full fixed-point map ``f(z; x)`` = one loop iteration output.

        A fixed point ``z*`` satisfies ``z* = residual_map(z*; x)``.
        """
        return self.loop_step(z, x, ctx)

    def relative_residual(
        self, z: mx.array, f_z: mx.array
    ) -> mx.array:
        """Relative L-inf residual ``||z - f(z)||_inf / (||f(z)||_inf + eps)``."""
        return _inf_norm(z - f_z) / (_inf_norm(f_z) + self.eps)

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
    ) -> mx.array | tuple[mx.array, dict[str, object]]:
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
        """
        if training_loops is not None:
            if training_loops <= 0:
                raise ValueError(
                    f"training_loops must be positive, got {training_loops}"
                )
            z = z0
            residuals: list[float] = []
            for _ in range(training_loops):
                if collect_residuals:
                    f_z = self.residual_map(z, x, ctx)
                    residuals.append(float(self.relative_residual(z, f_z).item()))
                    z = f_z
                else:
                    z = self.loop_step(z, x, ctx)
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
    ) -> mx.array | tuple[mx.array, dict[str, object]]:
        if fpopt_patience <= 0:
            raise ValueError(
                f"fpopt_patience must be positive, got {fpopt_patience}"
            )
        if not 0.0 < fpopt_gamma <= 1.0:
            raise ValueError(
                f"fpopt_gamma must be in (0, 1], got {fpopt_gamma}"
            )
        if not 0.0 < fpopt_eta0 <= 1.0:
            raise ValueError(
                f"fpopt_eta0 must be in (0, 1], got {fpopt_eta0}"
            )

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
            residuals.append(residual)
            if residual < self.tau:
                halted = True
                # Take the damped step then stop.
                z = eta * f_tilde + (1.0 - eta) * z
                break
            # FPOPT damping: on a stall (no improvement) relax eta toward 1.0
            # via gamma. gamma == 1.0 keeps eta == eta0 (the paper's best).
            if residual < best_residual - self.eps:
                best_residual = residual
                stall = 0
            else:
                stall += 1
                if stall >= fpopt_patience:
                    eta = eta * fpopt_gamma
                    stall = 0
            z = eta * f_tilde + (1.0 - eta) * z

        if collect_residuals:
            info: dict[str, object] = {
                "residuals": residuals,
                "steps": steps,
                "halted": halted,
                "eta": eta,
                "mode": "inference",
            }
            return z, info
        return z

    def __call__(
        self,
        z0: mx.array,
        x: mx.array,
        ctx: object = None,
        *,
        training_loops: int | None = None,
        **kwargs: object,
    ) -> mx.array | tuple[mx.array, dict[str, object]]:
        return self.forward(z0, x, ctx, training_loops=training_loops, **kwargs)


def _inverse_sigmoid(p: float) -> float:
    """Logit of ``p`` so ``sigmoid(logit) == p`` for ``p`` in ``(0, 1)``."""
    if not 0.0 < p < 1.0:
        # Allow the paper inits (0.75, 0.25); reject exact 0/1 which have no
        # finite logit (and would violate the strict 0 <= a < 1 with a < 1).
        raise ValueError(
            f"gate init must be in the open interval (0, 1), got {p}"
        )
    return math.log(p / (1.0 - p))


__all__ = [
    "StableFixedPointLoop",
    "StableLoopCore",
    "Sublayer",
]

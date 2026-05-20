"""ModelBuildSpec — composer that unifies forward graph + loss + optimizer
plus an ordered list of graph rewrites.

Pure data layer. Rewrites are applied via :meth:`apply_rewrites`, which
returns a NEW :class:`ModelBuildSpec` (immutable / frozen dataclass).
The Rewriter protocol is defined here as a forward-ref to keep this
module decoupled from Stage C — concrete rewriters live in
``cppmega_v4.buildspec.rewriters``.

Stage B scope: composer only. The :func:`verify_build_spec` coherence
checker lives in ``cppmega_v4.buildspec.diagnostics``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from cppmega_v4.buildspec.loss_spec import LossSpec
from cppmega_v4.buildspec.optim_spec import OptimSpec
from cppmega_v4.fusion.brick_graph import BrickGraph

if TYPE_CHECKING:  # pragma: no cover
    pass


class Rewriter(Protocol):
    """A pure :class:`ModelBuildSpec` transformation.

    A rewriter takes a :class:`ModelBuildSpec` and returns a NEW one;
    must not mutate inputs. Implementations advertise their preconditions
    and postconditions so the verifier can check ordering coherence.

    Required attributes (read by the verifier):
      name: identifier for telemetry / GUI chip.
      required_preconditions: frozenset of state tokens the spec must
        carry BEFORE this rewriter runs (e.g. ``"single_head"``).
      provided_postconditions: frozenset of state tokens added after
        this rewriter runs (e.g. ``"mtp_k_heads"``).
    """

    name: str
    required_preconditions: frozenset[str]
    provided_postconditions: frozenset[str]

    def __call__(self, spec: "ModelBuildSpec") -> "ModelBuildSpec": ...


@dataclass(frozen=True)
class ModelBuildSpec:
    """Unified spec: forward graph + loss + optimizer + rewrite chain.

    Fields:
      graph: the BrickGraph (Stage A surface from cppmega_v4.fusion).
      loss:  LossSpec.
      optim: OptimSpec.
      rewrites: tuple of Rewriter to apply in order (left-to-right).
      dim_env: named-dim env for shape resolution (used by verify_build_spec
        and by Stage E build_model when running the verify_and_estimate pass).
      state_tokens: frozenset of state tokens carried by this spec —
        Rewriter preconditions are checked against this set.
        Initial set is ``frozenset({"single_head"})``.
    """

    graph: BrickGraph
    loss: LossSpec
    optim: OptimSpec
    rewrites: tuple[Rewriter, ...] = ()
    dim_env: Mapping[str, int] = field(default_factory=dict)
    state_tokens: frozenset[str] = field(
        default_factory=lambda: frozenset({"single_head"})
    )

    def __post_init__(self) -> None:
        if not isinstance(self.graph, BrickGraph):
            raise TypeError(
                f"ModelBuildSpec.graph must be BrickGraph, got "
                f"{type(self.graph).__name__}"
            )
        if not isinstance(self.loss, LossSpec):
            raise TypeError(
                f"ModelBuildSpec.loss must be LossSpec, got "
                f"{type(self.loss).__name__}"
            )
        if not isinstance(self.optim, OptimSpec):
            raise TypeError(
                f"ModelBuildSpec.optim must be OptimSpec, got "
                f"{type(self.optim).__name__}"
            )
        for r in self.rewrites:
            for required_attr in (
                "name", "required_preconditions",
                "provided_postconditions", "__call__",
            ):
                if not hasattr(r, required_attr):
                    raise TypeError(
                        f"rewriter {r!r} missing attribute {required_attr!r}"
                    )

    def replace(
        self,
        *,
        graph: BrickGraph | None = None,
        loss: LossSpec | None = None,
        optim: OptimSpec | None = None,
        rewrites: tuple[Rewriter, ...] | None = None,
        dim_env: Mapping[str, int] | None = None,
        state_tokens: frozenset[str] | None = None,
    ) -> "ModelBuildSpec":
        """Return a copy with the given fields overridden."""
        return ModelBuildSpec(
            graph=graph if graph is not None else self.graph,
            loss=loss if loss is not None else self.loss,
            optim=optim if optim is not None else self.optim,
            rewrites=rewrites if rewrites is not None else self.rewrites,
            dim_env=dim_env if dim_env is not None else dict(self.dim_env),
            state_tokens=(
                state_tokens if state_tokens is not None else self.state_tokens
            ),
        )

    def apply_rewrites(self) -> "ModelBuildSpec":
        """Apply every rewriter in declared order.

        Each rewriter is checked for satisfied preconditions; raises
        :class:`RewriteOrderError` on violation. After every rewriter
        runs, its provided postconditions are added to the spec's
        ``state_tokens``.

        Returns a new spec with empty ``rewrites`` (already consumed).
        """
        current = self
        for r in self.rewrites:
            missing = r.required_preconditions - current.state_tokens
            if missing:
                raise RewriteOrderError(
                    f"rewriter {r.name!r} requires preconditions "
                    f"{sorted(missing)} but spec only carries "
                    f"{sorted(current.state_tokens)}"
                )
            current = r(current)
            current = current.replace(
                state_tokens=current.state_tokens | r.provided_postconditions,
            )
        return current.replace(rewrites=())


class RewriteOrderError(RuntimeError):
    """Raised by :meth:`ModelBuildSpec.apply_rewrites` on bad ordering."""


__all__ = [
    "ModelBuildSpec",
    "RewriteOrderError",
    "Rewriter",
]

"""OptimSpec — declarative optimizer specification.

Carries a sequence of :class:`ParamGroup` records (matcher + hyperparams)
plus a global :class:`OptimKind` and optional gradient-clip / mixed-
precision toggles. The matcher is a small DSL: ``"all"`` matches every
parameter, ``"moe_experts"`` matches anything whose qualified name
contains ``"expert"``, ``"embeddings"`` matches embedding tensors, and
``"regex:<pattern>"`` lets the GUI build a one-off group.

Pure data — no MLX runtime. The actual optimizer instance is materialised
at :func:`build_model` time (Stage E).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Final


class OptimKind(str, Enum):
    """Recognised optimizer families."""

    ADAMW              = "adamw"
    MUON               = "muon"
    MUON_ADAMW_HYBRID  = "muon_adamw_hybrid"
    SGD                = "sgd"


_VALID_MATCHER_PREFIXES: Final[frozenset[str]] = frozenset({"regex:"})
_BUILTIN_MATCHERS: Final[frozenset[str]] = frozenset({
    "all", "moe_experts", "embeddings", "attention", "mlp", "head",
})


@dataclass(frozen=True)
class ParamGroup:
    """One optimizer parameter group.

    Fields:
      matcher: built-in name (``"all"`` / ``"moe_experts"`` / ...) or
        ``"regex:<python regex>"`` for custom selection.
      lr: learning rate (must be > 0).
      weight_decay: L2 weight decay (≥ 0).
      betas: AdamW betas (β1, β2). Required when used with ADAMW kind;
        ignored for SGD / MUON.
      ns_steps: Muon Newton-Schulz iteration count. Required when used
        with MUON kind; ignored otherwise.
    """

    matcher: str
    lr: float
    weight_decay: float = 0.01
    betas: tuple[float, float] | None = None
    ns_steps: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.matcher, str) or not self.matcher.strip():
            raise ValueError(
                "ParamGroup.matcher must be non-empty str"
            )
        if (
            self.matcher not in _BUILTIN_MATCHERS
            and not any(self.matcher.startswith(p) for p in _VALID_MATCHER_PREFIXES)
        ):
            raise ValueError(
                f"ParamGroup.matcher={self.matcher!r} must be one of "
                f"{sorted(_BUILTIN_MATCHERS)} or start with 'regex:'"
            )
        if self.lr <= 0:
            raise ValueError(f"ParamGroup.lr must be > 0, got {self.lr!r}")
        if self.weight_decay < 0:
            raise ValueError(
                f"ParamGroup.weight_decay must be ≥ 0, got {self.weight_decay!r}"
            )
        if self.betas is not None:
            if len(self.betas) != 2 or not all(0 <= b < 1 for b in self.betas):
                raise ValueError(
                    f"ParamGroup.betas must be (β1, β2) with 0 ≤ β < 1; "
                    f"got {self.betas!r}"
                )
        if self.ns_steps is not None and self.ns_steps < 1:
            raise ValueError(
                f"ParamGroup.ns_steps must be ≥ 1, got {self.ns_steps!r}"
            )


@dataclass(frozen=True)
class OptimSpec:
    """Declarative optimizer specification.

    Fields:
      kind: one of :class:`OptimKind`.
      groups: tuple of :class:`ParamGroup` (must be non-empty).
      gradient_clip_norm: optional L2 grad-clip norm; None disables.
      mixed_precision: when True, optimizer state can be fp32 while
        params are bf16.
    """

    kind: OptimKind
    groups: tuple[ParamGroup, ...]
    gradient_clip_norm: float | None = 1.0
    mixed_precision: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.kind, OptimKind):
            raise TypeError(
                f"OptimSpec.kind must be OptimKind, got {type(self.kind).__name__}"
            )
        if not self.groups:
            raise ValueError("OptimSpec.groups must not be empty")
        for g in self.groups:
            if not isinstance(g, ParamGroup):
                raise TypeError(
                    f"OptimSpec.groups entries must be ParamGroup, got "
                    f"{type(g).__name__}"
                )
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError(
                "OptimSpec.gradient_clip_norm must be > 0 or None; "
                f"got {self.gradient_clip_norm!r}"
            )
        # Per-kind sanity
        if self.kind is OptimKind.ADAMW:
            for g in self.groups:
                if g.betas is None:
                    raise ValueError(
                        f"ADAMW group must declare betas; group {g.matcher!r} "
                        "is missing them"
                    )
        elif self.kind is OptimKind.MUON:
            for g in self.groups:
                if g.ns_steps is None:
                    raise ValueError(
                        f"MUON group must declare ns_steps; group {g.matcher!r} "
                        "is missing it"
                    )


# ---------------------------------------------------------------------------
# Built-in factories
# ---------------------------------------------------------------------------


def adamw(
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.95),
    *,
    gradient_clip_norm: float | None = 1.0,
) -> OptimSpec:
    """Single-group AdamW spec applied to all parameters."""
    return OptimSpec(
        kind=OptimKind.ADAMW,
        groups=(
            ParamGroup(matcher="all", lr=lr, weight_decay=weight_decay,
                       betas=betas),
        ),
        gradient_clip_norm=gradient_clip_norm,
    )


def muon(
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    ns_steps: int = 5,
    *,
    gradient_clip_norm: float | None = 1.0,
) -> OptimSpec:
    """Single-group Muon spec — 2D-parameter optimiser, no betas."""
    return OptimSpec(
        kind=OptimKind.MUON,
        groups=(
            ParamGroup(matcher="all", lr=lr, weight_decay=weight_decay,
                       ns_steps=ns_steps),
        ),
        gradient_clip_norm=gradient_clip_norm,
    )


def muon_adamw_hybrid(
    muon_lr: float = 1e-2,
    adam_lr: float = 3e-4,
    weight_decay: float = 0.01,
    adam_betas: tuple[float, float] = (0.9, 0.95),
    ns_steps: int = 5,
    *,
    gradient_clip_norm: float | None = 1.0,
) -> OptimSpec:
    """Hybrid: 2D params on Muon, everything else (norms, biases,
    embeddings, head) on AdamW. Common pattern for >100B models.

    The first group ``moe_experts`` is intentionally on AdamW (small
    matrices per expert) — Muon is for backbone matmuls."""
    return OptimSpec(
        kind=OptimKind.MUON_ADAMW_HYBRID,
        groups=(
            ParamGroup(
                matcher="moe_experts", lr=adam_lr, weight_decay=weight_decay,
                betas=adam_betas,
            ),
            ParamGroup(
                matcher="embeddings", lr=adam_lr, weight_decay=0.0,
                betas=adam_betas,
            ),
            ParamGroup(
                matcher="head", lr=adam_lr, weight_decay=weight_decay,
                betas=adam_betas,
            ),
            ParamGroup(
                matcher="all", lr=muon_lr, weight_decay=weight_decay,
                ns_steps=ns_steps,
            ),
        ),
        gradient_clip_norm=gradient_clip_norm,
    )


def sgd(
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    *,
    gradient_clip_norm: float | None = 1.0,
) -> OptimSpec:
    """Single-group plain SGD."""
    return OptimSpec(
        kind=OptimKind.SGD,
        groups=(ParamGroup(matcher="all", lr=lr, weight_decay=weight_decay),),
        gradient_clip_norm=gradient_clip_norm,
    )


# ---------------------------------------------------------------------------
# Registry — used by Stage E / GUI dropdown
# ---------------------------------------------------------------------------


OPTIM_BUILTINS: dict[str, str] = {
    "adamw":              "cppmega_v4.buildspec.optim_spec:adamw",
    "muon":               "cppmega_v4.buildspec.optim_spec:muon",
    "muon_adamw_hybrid":  "cppmega_v4.buildspec.optim_spec:muon_adamw_hybrid",
    "sgd":                "cppmega_v4.buildspec.optim_spec:sgd",
}


__all__ = [
    "OPTIM_BUILTINS",
    "OptimKind",
    "OptimSpec",
    "ParamGroup",
    "adamw",
    "muon",
    "muon_adamw_hybrid",
    "sgd",
]

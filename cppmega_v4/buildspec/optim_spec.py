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

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Final


class OptimKind(str, Enum):
    """Recognised optimizer families."""

    ADAMW              = "adamw"
    MUON               = "muon"
    MUON_ADAMW_HYBRID  = "muon_adamw_hybrid"
    LION               = "lion"
    LION_8BIT          = "lion8bit"
    ADAM_8BIT          = "adam8bit"
    SGD                = "sgd"


# Sign-based Lion updates scale ONLY by the sign of momentum, so a large
# learning rate quickly blows up to NaN. Chen et al. 2023 recommend
# 3-10x smaller LR than AdamW; we issue a UserWarning above this ceiling.
LION_LR_WARN_CEILING: Final[float] = 5e-4


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
        elif self.kind in (OptimKind.LION, OptimKind.LION_8BIT):
            for g in self.groups:
                if g.betas is None:
                    raise ValueError(
                        f"{self.kind.value.upper()} group must declare betas; "
                        f"group {g.matcher!r} is missing them"
                    )
                if g.lr > LION_LR_WARN_CEILING:
                    warnings.warn(
                        f"{self.kind.value} group {g.matcher!r} lr={g.lr:.2e} "
                        f"exceeds recommended ceiling {LION_LR_WARN_CEILING:.0e}; "
                        "sign-based updates ignore gradient magnitude and "
                        "diverge to NaN at high LR (Chen et al. 2023).",
                        UserWarning,
                        stacklevel=3,
                    )
        elif self.kind is OptimKind.ADAM_8BIT:
            for g in self.groups:
                if g.betas is None:
                    raise ValueError(
                        f"ADAM_8BIT group must declare betas; group "
                        f"{g.matcher!r} is missing them"
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


def lion(
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    betas: tuple[float, float] = (0.9, 0.99),
    *,
    gradient_clip_norm: float | None = 1.0,
) -> OptimSpec:
    """Single-group Lion spec (Chen et al. 2023).

    Defaults match the paper: lr=1e-4 (3-10x smaller than AdamW because
    sign-based updates ignore gradient magnitude), betas=(0.9, 0.99).
    Memory footprint is half of AdamW: a single fp32 momentum buffer.

    Raises UserWarning if lr > 5e-4 (typically NaN territory).
    """
    return OptimSpec(
        kind=OptimKind.LION,
        groups=(
            ParamGroup(matcher="all", lr=lr, weight_decay=weight_decay,
                       betas=betas),
        ),
        gradient_clip_norm=gradient_clip_norm,
    )


def lion8bit(
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    betas: tuple[float, float] = (0.9, 0.99),
    *,
    gradient_clip_norm: float | None = 1.0,
) -> OptimSpec:
    """Single-group 8-bit Lion (quantized momentum) — same defaults as
    :func:`lion` but with int8 momentum + per-block fp32 absmax (block
    size 256). Use when memory-constrained beyond Lion's already-halved
    state. Runtime selects :class:`Lion8bit` over :class:`LionFP32Moments`
    via the ``kind`` field."""
    return OptimSpec(
        kind=OptimKind.LION_8BIT,
        groups=(
            ParamGroup(matcher="all", lr=lr, weight_decay=weight_decay,
                       betas=betas),
        ),
        gradient_clip_norm=gradient_clip_norm,
    )


def adam8bit(
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.999),
    *,
    gradient_clip_norm: float | None = 1.0,
) -> OptimSpec:
    """Single-group 8-bit AdamW (quantized first+second moments).

    Defaults track standard AdamW (lr=3e-4) because the quantization
    preserves the update direction; only the moment buffers shrink.
    Use when memory budget cannot accommodate full fp32 moments."""
    return OptimSpec(
        kind=OptimKind.ADAM_8BIT,
        groups=(
            ParamGroup(matcher="all", lr=lr, weight_decay=weight_decay,
                       betas=betas),
        ),
        gradient_clip_norm=gradient_clip_norm,
    )


# ---------------------------------------------------------------------------
# Registry — used by Stage E / GUI dropdown
# ---------------------------------------------------------------------------


OPTIM_BUILTINS: dict[str, str] = {
    "adamw":              "cppmega_v4.buildspec.optim_spec:adamw",
    "muon":               "cppmega_v4.buildspec.optim_spec:muon",
    "muon_adamw_hybrid":  "cppmega_v4.buildspec.optim_spec:muon_adamw_hybrid",
    "lion":               "cppmega_v4.buildspec.optim_spec:lion",
    "lion8bit":           "cppmega_v4.buildspec.optim_spec:lion8bit",
    "adam8bit":           "cppmega_v4.buildspec.optim_spec:adam8bit",
    "sgd":                "cppmega_v4.buildspec.optim_spec:sgd",
}


__all__ = [
    "LION_LR_WARN_CEILING",
    "OPTIM_BUILTINS",
    "OptimKind",
    "OptimSpec",
    "ParamGroup",
    "adam8bit",
    "adamw",
    "lion",
    "lion8bit",
    "muon",
    "muon_adamw_hybrid",
    "sgd",
]

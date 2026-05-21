"""LR Schedule factories for the Visual Builder.

A :class:`ScheduleSpec` is a pure-data descriptor for an LR schedule
shape (cosine / linear_warmup / wsd / inv_sqrt / polynomial / constant).
:meth:`ScheduleSpec.build` materialises a ``Callable[[int], float]`` —
the standard MLX optimizer ``learning_rate`` argument shape — by
sampling the underlying maths at each step index.

Tied into OptimSpec via the new ``ParamGroup.schedule`` field
(optional, falls back to constant ``lr`` when absent). The runner
materialises every group's effective learning-rate callable at
:func:`stage_train` time.

References:
  * cosine: Loshchilov & Hutter, 2016 (arXiv:1608.03983) — SGDR.
  * wsd:    DeepSeek-V2 tech report, 2024 (arXiv:2405.04434).
  * inv_sqrt: Vaswani et al., 2017 (Attention Is All You Need).
  * polynomial: standard BERT/T5 pretraining recipe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Final, Literal


ScheduleKind = Literal[
    "constant", "cosine", "linear_warmup", "wsd",
    "inv_sqrt", "polynomial",
]

SCHEDULE_KINDS: Final[tuple[str, ...]] = (
    "constant", "cosine", "linear_warmup", "wsd",
    "inv_sqrt", "polynomial",
)


@dataclass(frozen=True)
class ScheduleSpec:
    """Declarative LR schedule.

    All fields default to safe values; ``__post_init__`` enforces the
    invariants documented per kind. Use one of the module-level
    factories (cosine_annealing / linear_warmup_then_constant / wsd /
    inv_sqrt / polynomial / constant) rather than instantiating
    directly when possible — factories pre-validate per kind.

    Fields:
      kind: see :class:`ScheduleKind`.
      warmup_steps: linear warmup from 0 to base_lr over this many
        steps. 0 disables warmup. Must be ≥ 0.
      total_steps: required for cosine / wsd / polynomial. Total number
        of training steps the schedule shapes itself over. None for
        constant / linear_warmup / inv_sqrt.
      min_lr_ratio: floor as a fraction of base_lr at the end of decay.
        Used by cosine / wsd. Must satisfy 0 ≤ ratio ≤ 1.
      decay_steps: required for wsd — number of steps in the decay
        tail. Implies steady-state length = total_steps - warmup_steps
        - decay_steps.
      power: polynomial decay exponent. > 0 typically; 1.0 = linear,
        2.0 = quadratic.
    """

    kind: ScheduleKind = "constant"
    warmup_steps: int = 0
    total_steps: int | None = None
    min_lr_ratio: float = 0.1
    decay_steps: int | None = None
    power: float = 2.0

    def __post_init__(self) -> None:
        if self.kind not in SCHEDULE_KINDS:
            raise ValueError(
                f"ScheduleSpec.kind={self.kind!r} must be one of "
                f"{SCHEDULE_KINDS}"
            )
        if self.warmup_steps < 0:
            raise ValueError(
                f"warmup_steps must be ≥ 0; got {self.warmup_steps!r}"
            )
        if not (0.0 <= self.min_lr_ratio <= 1.0):
            raise ValueError(
                f"min_lr_ratio must be in [0, 1]; got {self.min_lr_ratio!r}"
            )
        if self.power <= 0:
            raise ValueError(
                f"power must be > 0; got {self.power!r}"
            )
        if self.kind in ("cosine", "wsd", "polynomial"):
            if self.total_steps is None or self.total_steps <= 0:
                raise ValueError(
                    f"{self.kind!r} schedule requires total_steps > 0; "
                    f"got {self.total_steps!r}"
                )
            if self.total_steps < self.warmup_steps:
                raise ValueError(
                    f"{self.kind!r} total_steps ({self.total_steps}) must "
                    f"be ≥ warmup_steps ({self.warmup_steps})"
                )
        if self.kind == "wsd":
            if self.decay_steps is None or self.decay_steps < 1:
                raise ValueError(
                    f"wsd schedule requires decay_steps ≥ 1; got "
                    f"{self.decay_steps!r}"
                )
            assert self.total_steps is not None  # narrowed above
            if self.warmup_steps + self.decay_steps > self.total_steps:
                raise ValueError(
                    f"wsd warmup_steps ({self.warmup_steps}) + "
                    f"decay_steps ({self.decay_steps}) must be ≤ "
                    f"total_steps ({self.total_steps})"
                )

    def build(self, base_lr: float) -> Callable[[int], float]:
        """Return ``step → lr`` callable for use as
        ``mlx.optimizers.AdamW(learning_rate=...)``."""
        if base_lr <= 0:
            raise ValueError(f"base_lr must be > 0; got {base_lr!r}")
        kind = self.kind
        warmup = self.warmup_steps
        total = self.total_steps
        floor = base_lr * self.min_lr_ratio
        decay = self.decay_steps
        power = self.power

        if kind == "constant":
            def _constant(step: int) -> float:
                return base_lr
            return _constant

        if kind == "linear_warmup":
            def _linear_warmup(step: int) -> float:
                if step < warmup:
                    return base_lr * (step / max(1, warmup))
                return base_lr
            return _linear_warmup

        if kind == "cosine":
            assert total is not None
            def _cosine(step: int) -> float:
                if step < warmup:
                    return base_lr * (step / max(1, warmup))
                # Cosine from base_lr down to floor over (total-warmup)
                progress = min(1.0, (step - warmup) / max(1, total - warmup))
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return floor + (base_lr - floor) * cosine
            return _cosine

        if kind == "wsd":
            assert total is not None and decay is not None
            steady_end = total - decay
            def _wsd(step: int) -> float:
                if step < warmup:
                    return base_lr * (step / max(1, warmup))
                if step < steady_end:
                    return base_lr
                # Linear decay from base_lr to floor over `decay` steps
                progress = min(1.0, (step - steady_end) / max(1, decay))
                return base_lr + (floor - base_lr) * progress
            return _wsd

        if kind == "inv_sqrt":
            scale = max(1, warmup)
            def _inv_sqrt(step: int) -> float:
                if step < warmup:
                    return base_lr * (step / max(1, warmup))
                return base_lr * math.sqrt(scale / max(1, step))
            return _inv_sqrt

        if kind == "polynomial":
            assert total is not None
            def _polynomial(step: int) -> float:
                if step < warmup:
                    return base_lr * (step / max(1, warmup))
                progress = min(1.0, (step - warmup) / max(1, total - warmup))
                return floor + (base_lr - floor) * ((1.0 - progress) ** power)
            return _polynomial

        raise AssertionError(f"unreachable: {kind!r}")

    def sample(self, base_lr: float, n_points: int = 50) -> list[float]:
        """Return ``n_points`` LR values sampled along the schedule.
        Used by GUI to render the mini-sparkline preview.

        If ``total_steps`` is None (constant / linear_warmup / inv_sqrt),
        samples ``max(warmup_steps, n_points)`` steps so the warmup
        ramp is visible.
        """
        fn = self.build(base_lr)
        if self.total_steps is not None:
            horizon = self.total_steps
        elif self.warmup_steps > 0:
            horizon = max(self.warmup_steps * 2, n_points)
        else:
            horizon = n_points
        step_size = max(1, horizon // n_points)
        return [fn(i * step_size) for i in range(n_points)]


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def constant() -> ScheduleSpec:
    """No-op schedule — base_lr at every step."""
    return ScheduleSpec(kind="constant")


def linear_warmup_then_constant(warmup_steps: int) -> ScheduleSpec:
    """Linear ramp from 0 to base_lr over ``warmup_steps``, then
    holds base_lr forever. Common for fine-tuning where total
    duration is open-ended."""
    if warmup_steps <= 0:
        raise ValueError(f"warmup_steps must be > 0; got {warmup_steps!r}")
    return ScheduleSpec(kind="linear_warmup", warmup_steps=warmup_steps)


def cosine_annealing(
    total_steps: int,
    min_lr_ratio: float = 0.1,
    warmup_steps: int = 0,
) -> ScheduleSpec:
    """Linear warmup + cosine decay to ``base_lr * min_lr_ratio``.
    Default LLM pretraining recipe (Chinchilla, GPT-NeoX)."""
    return ScheduleSpec(
        kind="cosine",
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=min_lr_ratio,
    )


def wsd(
    warmup_steps: int,
    decay_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> ScheduleSpec:
    """Warmup → Steady → linear Decay (DeepSeek-V2). Steady phase
    allows mid-training checkpoint reuse without re-tuning LR."""
    return ScheduleSpec(
        kind="wsd",
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        decay_steps=decay_steps,
        min_lr_ratio=min_lr_ratio,
    )


def inv_sqrt(warmup_steps: int) -> ScheduleSpec:
    """Linear warmup + 1/sqrt(step) decay (Vaswani 2017)."""
    if warmup_steps <= 0:
        raise ValueError(f"warmup_steps must be > 0; got {warmup_steps!r}")
    return ScheduleSpec(kind="inv_sqrt", warmup_steps=warmup_steps)


def polynomial(
    total_steps: int,
    power: float = 2.0,
    min_lr_ratio: float = 0.1,
    warmup_steps: int = 0,
) -> ScheduleSpec:
    """Linear warmup + polynomial decay to ``base_lr * min_lr_ratio``."""
    return ScheduleSpec(
        kind="polynomial",
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        power=power,
        min_lr_ratio=min_lr_ratio,
    )


SCHEDULE_BUILTINS: dict[str, str] = {
    "constant":      "cppmega_v4.buildspec.schedules:constant",
    "linear_warmup": "cppmega_v4.buildspec.schedules:linear_warmup_then_constant",
    "cosine":        "cppmega_v4.buildspec.schedules:cosine_annealing",
    "wsd":           "cppmega_v4.buildspec.schedules:wsd",
    "inv_sqrt":      "cppmega_v4.buildspec.schedules:inv_sqrt",
    "polynomial":    "cppmega_v4.buildspec.schedules:polynomial",
}


__all__ = [
    "SCHEDULE_BUILTINS",
    "SCHEDULE_KINDS",
    "ScheduleKind",
    "ScheduleSpec",
    "constant",
    "cosine_annealing",
    "inv_sqrt",
    "linear_warmup_then_constant",
    "polynomial",
    "wsd",
]

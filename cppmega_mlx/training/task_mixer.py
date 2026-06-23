"""Weighted, deterministic task mixer for Stage-1 multi-objective training.

The mixer draws ONE :class:`TaskKind` per source packet according to explicit
stage rates, then dispatches to the matching builder in
:mod:`cppmega_mlx.training.objectives` to produce an
:class:`~cppmega_mlx.training.objectives.ObjectiveExample`.

Determinism: every draw uses a ``random.Random`` seeded from an explicit ``seed``
arg (NEVER the global RNG / ``Math.random`` equivalent), so a fixed seed + a fixed
packet stream reproduces the exact task sequence and FIM permutations.

RULE #1 (fail fast / fail loud): the rate map must sum to ``1.0`` (within a tiny
tolerance) and contain only known TaskKinds; an out-of-spec mix RAISES.  A packet
that cannot satisfy the drawn task (wrong packet type, absent required field)
propagates the builder's RAISE — no silent skip / re-draw.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator, Mapping
from enum import Enum
from typing import Callable

from cppmega_mlx.data.ast_fim import InstructionEncoder
from cppmega_mlx.data.code_packet import CodePacket
from cppmega_mlx.data.commit_packet import CommitPacket
from cppmega_mlx.data.fim import FIMSpecialTokenInput
from cppmega_mlx.training.objectives import (
    ObjectiveExample,
    build_ast_fim,
    build_causal_lm,
    build_commit_diff,
    build_ifim,
    build_pre_to_post,
    build_recovery,
)

_RATE_SUM_TOL = 1e-6


class TaskKind(str, Enum):
    """The Stage-1 training objectives the mixer can draw."""

    CAUSAL_LM = "causal_lm"
    AST_FIM = "ast_fim"
    IFIM = "ifim"
    COMMIT_DIFF = "commit_diff"
    PRE_TO_POST = "pre_to_post"
    SYMBOL_RECOVERY = "symbol_recovery"
    TYPE_RECOVERY = "type_recovery"
    CALLEE_RECOVERY = "callee_recovery"


# Stage-1 default mix.  The four research-grounded "buckets" are causal_lm 0.5,
# ast_fim/ifim 0.2, commit 0.2, recovery 0.1; we split each bucket evenly across
# its member TaskKinds so the bucket totals match the spec exactly.
STAGE1_DEFAULT_RATES: dict[TaskKind, float] = {
    TaskKind.CAUSAL_LM: 0.5,
    TaskKind.AST_FIM: 0.1,
    TaskKind.IFIM: 0.1,
    TaskKind.COMMIT_DIFF: 0.1,
    TaskKind.PRE_TO_POST: 0.1,
    TaskKind.SYMBOL_RECOVERY: 0.1 / 3.0,
    TaskKind.TYPE_RECOVERY: 0.1 / 3.0,
    TaskKind.CALLEE_RECOVERY: 0.1 / 3.0,
}

# Named stage presets; extendable for later curricula.
STAGE_RATES: dict[str, dict[TaskKind, float]] = {
    "stage1": STAGE1_DEFAULT_RATES,
}

_CODE_TASKS = frozenset(
    {
        TaskKind.CAUSAL_LM,
        TaskKind.AST_FIM,
        TaskKind.IFIM,
        TaskKind.SYMBOL_RECOVERY,
        TaskKind.TYPE_RECOVERY,
        TaskKind.CALLEE_RECOVERY,
    }
)
_COMMIT_TASKS = frozenset({TaskKind.COMMIT_DIFF, TaskKind.PRE_TO_POST})

_RECOVERY_KIND = {
    TaskKind.SYMBOL_RECOVERY: "symbol",
    TaskKind.TYPE_RECOVERY: "type",
    TaskKind.CALLEE_RECOVERY: "callee",
}


def normalize_rates(
    rates: Mapping[TaskKind | str, float] | None,
    *,
    stage: str = "stage1",
) -> dict[TaskKind, float]:
    """Validate + canonicalize a rate map (defaults to the named stage preset).

    RAISES on unknown keys, negative rates, or a sum that is not ``1.0`` within a
    tiny tolerance — fail-loud, no silent renormalization.
    """

    if rates is None:
        if stage not in STAGE_RATES:
            raise ValueError(
                f"unknown stage {stage!r}; known: {sorted(STAGE_RATES)}"
            )
        rates = STAGE_RATES[stage]

    canonical: dict[TaskKind, float] = {}
    for key, value in rates.items():
        task = key if isinstance(key, TaskKind) else TaskKind(key)
        if value < 0.0:
            raise ValueError(f"task rate for {task.value!r} is negative: {value}")
        canonical[task] = canonical.get(task, 0.0) + float(value)

    if not canonical:
        raise ValueError("task rate map is empty")

    total = sum(canonical.values())
    if abs(total - 1.0) > _RATE_SUM_TOL:
        raise ValueError(
            f"task rates must sum to 1.0 (within {_RATE_SUM_TOL}); got {total} "
            f"for {{ {', '.join(f'{k.value}={v}' for k, v in canonical.items())} }}"
        )
    return canonical


class TaskMixer:
    """Deterministic weighted sampler over Stage-1 objectives.

    Each ``mix(...)`` / ``draw_task(...)`` call derives a per-step
    ``random.Random`` from ``(seed, step_index)`` so the task draw AND the FIM
    permutation inside a step are jointly reproducible for a fixed seed.
    """

    def __init__(
        self,
        rates: Mapping[TaskKind | str, float] | None = None,
        *,
        seed: int,
        stage: str = "stage1",
        instruction_encoder: InstructionEncoder | None = None,
        special_token_ids: FIMSpecialTokenInput = None,
        spm_rate: float = 0.5,
    ) -> None:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError(f"TaskMixer seed must be an int, got {type(seed).__name__}")
        self._rates = normalize_rates(rates, stage=stage)
        self._tasks = list(self._rates.keys())
        self._weights = [self._rates[t] for t in self._tasks]
        self._seed = seed
        self._instruction_encoder = instruction_encoder
        self._special_token_ids = special_token_ids
        self._spm_rate = spm_rate

    @property
    def rates(self) -> dict[TaskKind, float]:
        return dict(self._rates)

    def _step_rng(self, step_index: int) -> random.Random:
        # Derive a stable per-step int seed (tuple seeds are rejected on 3.14+).
        derived = (self._seed * 0x9E3779B1 + step_index) & 0xFFFFFFFFFFFFFFFF
        return random.Random(derived)

    def draw_task(self, step_index: int) -> TaskKind:
        """Draw the TaskKind for ``step_index`` (deterministic for a fixed seed)."""

        rng = self._step_rng(step_index)
        return rng.choices(self._tasks, weights=self._weights, k=1)[0]

    def build(
        self,
        task: TaskKind,
        packet: CodePacket | CommitPacket,
        *,
        rng: random.Random,
    ) -> ObjectiveExample:
        """Dispatch one packet to the builder for ``task`` using ``rng``."""

        if task in _CODE_TASKS:
            if not isinstance(packet, CodePacket):
                raise TypeError(
                    f"task {task.value!r} requires a CodePacket, got "
                    f"{type(packet).__name__}"
                )
        elif task in _COMMIT_TASKS:
            if not isinstance(packet, CommitPacket):
                raise TypeError(
                    f"task {task.value!r} requires a CommitPacket, got "
                    f"{type(packet).__name__}"
                )
        else:  # pragma: no cover - guarded by the enum
            raise ValueError(f"unhandled task kind {task!r}")

        if task is TaskKind.CAUSAL_LM:
            return build_causal_lm(packet)
        if task is TaskKind.AST_FIM:
            return build_ast_fim(
                packet,
                rng=rng,
                spm_rate=self._spm_rate,
                special_token_ids=self._special_token_ids,
            )
        if task is TaskKind.IFIM:
            if self._instruction_encoder is None:
                raise ValueError(
                    "TaskMixer drew IFIM but no instruction_encoder was provided; "
                    "supply one or set the IFIM rate to 0"
                )
            return build_ifim(
                packet,
                instruction_encoder=self._instruction_encoder,
                rng=rng,
                spm_rate=self._spm_rate,
                special_token_ids=self._special_token_ids,
            )
        if task is TaskKind.COMMIT_DIFF:
            return build_commit_diff(packet, special_token_ids=self._special_token_ids)
        if task is TaskKind.PRE_TO_POST:
            return build_pre_to_post(packet, special_token_ids=self._special_token_ids)
        # Recovery family.
        return build_recovery(packet, kind=_RECOVERY_KIND[task], rng=rng)

    def mix(
        self,
        packets: Iterable[CodePacket | CommitPacket],
        *,
        start_step: int = 0,
    ) -> Iterator[tuple[TaskKind, ObjectiveExample]]:
        """Yield ``(task, example)`` per packet, drawing a task deterministically.

        The step index advances per consumed packet (offset by ``start_step`` for
        resumable streams).  Each step's RNG drives BOTH the task draw and the
        builder, so the whole stream is reproducible for a fixed seed.
        """

        for offset, packet in enumerate(packets):
            step_index = start_step + offset
            rng = self._step_rng(step_index)
            task = rng.choices(self._tasks, weights=self._weights, k=1)[0]
            yield task, self.build(task, packet, rng=rng)


__all__ = [
    "STAGE1_DEFAULT_RATES",
    "STAGE_RATES",
    "TaskKind",
    "TaskMixer",
    "normalize_rates",
]

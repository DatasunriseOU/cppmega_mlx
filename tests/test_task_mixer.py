"""Tests for the weighted deterministic task mixer."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from cppmega_mlx.data.code_packet import CodePacket
from cppmega_mlx.data.commit_packet import CommitPacket
from cppmega_mlx.training.objectives import ObjectiveExample
from cppmega_mlx.training.task_mixer import (
    STAGE1_DEFAULT_RATES,
    TaskKind,
    TaskMixer,
    normalize_rates,
)


def _arr(values: list[int]) -> mx.array:
    return mx.array(np.asarray(values, dtype=np.int32))


def _code_packet() -> CodePacket:
    return CodePacket(
        token_ids=_arr(list(range(12))),
        chunk_starts=_arr([0, 4, 8]),
        chunk_ends=_arr([4, 8, 12]),
        chunk_kinds=_arr([1, 1, 1]),
        chunk_dep_levels=_arr([0, 0, 0]),
        symbol_ids=_arr([0, 0, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0]),
        type_refs=_arr([0, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        call_targets=_arr([0, 0, 0, 0, 0, 9, 9, 0, 0, 0, 0, 0]),
        ifim_instruction_token_ids=_arr([300, 301, 302]),
    )


# --------------------------- RATE VALIDATION --------------------- #
def test_stage1_default_rates_sum_to_one() -> None:
    canonical = normalize_rates(None, stage="stage1")
    assert abs(sum(canonical.values()) - 1.0) < 1e-9
    # Bucket checks per spec.
    assert abs(canonical[TaskKind.CAUSAL_LM] - 0.5) < 1e-9
    fim_bucket = (
        canonical[TaskKind.FIM]
        + canonical[TaskKind.AST_FIM]
        + canonical[TaskKind.IFIM]
    )
    assert abs(fim_bucket - 0.2) < 1e-9
    commit_bucket = canonical[TaskKind.COMMIT_DIFF] + canonical[TaskKind.PRE_TO_POST]
    assert abs(commit_bucket - 0.2) < 1e-9
    recovery_bucket = (
        canonical[TaskKind.SYMBOL_RECOVERY]
        + canonical[TaskKind.TYPE_RECOVERY]
        + canonical[TaskKind.CALLEE_RECOVERY]
    )
    assert abs(recovery_bucket - 0.1) < 1e-9


def test_rates_not_summing_to_one_raises() -> None:
    with pytest.raises(ValueError, match="must sum to 1.0"):
        normalize_rates({TaskKind.CAUSAL_LM: 0.5, TaskKind.AST_FIM: 0.2})


def test_negative_rate_raises() -> None:
    with pytest.raises(ValueError, match="negative"):
        normalize_rates({TaskKind.CAUSAL_LM: 1.2, TaskKind.AST_FIM: -0.2})


def test_non_finite_rate_raises() -> None:
    with pytest.raises(ValueError, match="finite"):
        normalize_rates({TaskKind.CAUSAL_LM: float("nan")})


def test_unknown_task_key_raises() -> None:
    with pytest.raises(ValueError):
        normalize_rates({"not_a_task": 1.0})


# --------------------------- DETERMINISM ------------------------- #
def test_draw_task_deterministic_for_fixed_seed() -> None:
    m1 = TaskMixer(seed=123)
    m2 = TaskMixer(seed=123)
    seq1 = [m1.draw_task(i) for i in range(50)]
    seq2 = [m2.draw_task(i) for i in range(50)]
    assert seq1 == seq2


def test_different_seed_changes_sequence() -> None:
    a = [TaskMixer(seed=1).draw_task(i) for i in range(50)]
    b = [TaskMixer(seed=2).draw_task(i) for i in range(50)]
    assert a != b


# --------------------------- RATE FIDELITY ----------------------- #
def test_draw_respects_rates_within_tolerance() -> None:
    # Use a simple two-task mix so the empirical fraction is easy to check.
    rates = {TaskKind.CAUSAL_LM: 0.7, TaskKind.AST_FIM: 0.3}
    mixer = TaskMixer(rates, seed=99)
    n = 20000
    draws = [mixer.draw_task(i) for i in range(n)]
    frac_causal = sum(1 for d in draws if d is TaskKind.CAUSAL_LM) / n
    assert abs(frac_causal - 0.7) < 0.02


# --------------------------- END-TO-END MIX ---------------------- #
def test_mix_yields_aligned_examples_and_is_deterministic() -> None:
    packets = [_code_packet() for _ in range(8)]
    mixer = TaskMixer(
        {TaskKind.CAUSAL_LM: 0.4, TaskKind.AST_FIM: 0.3, TaskKind.IFIM: 0.3},
        seed=5,
    )
    out1 = list(mixer.mix(packets))
    out2 = list(TaskMixer(
        {TaskKind.CAUSAL_LM: 0.4, TaskKind.AST_FIM: 0.3, TaskKind.IFIM: 0.3},
        seed=5,
    ).mix(packets))

    assert [t for t, _ in out1] == [t for t, _ in out2]
    for (_, ex1), (_, ex2) in zip(out1, out2):
        assert isinstance(ex1, ObjectiveExample)
        n = int(ex1.input_ids.shape[0])
        assert int(ex1.target_ids.shape[0]) == n
        assert int(ex1.loss_mask.shape[0]) == n
        assert np.asarray(ex1.input_ids).tolist() == np.asarray(ex2.input_ids).tolist()


def test_ifim_without_typed_instruction_raises() -> None:
    mixer = TaskMixer({TaskKind.IFIM: 1.0}, seed=0)
    packet = CodePacket(token_ids=_arr(list(range(12))))
    with pytest.raises(ValueError, match="ifim_instruction_token_ids"):
        list(mixer.mix([packet]))


def test_commit_task_with_code_packet_raises() -> None:
    mixer = TaskMixer({TaskKind.COMMIT_DIFF: 1.0}, seed=0)
    with pytest.raises(TypeError, match="requires a CommitPacket"):
        list(mixer.mix([_code_packet()]))


def test_mix_commit_packets() -> None:
    commits = [
        CommitPacket(
            pre_token_ids=_arr([10, 11, 12]),
            post_token_ids=_arr([20, 21, 22]),
            diff_token_ids=_arr([30, 31]),
            commit_msg=_arr([40, 41]),
        )
        for _ in range(6)
    ]
    mixer = TaskMixer(
        {TaskKind.COMMIT_DIFF: 0.5, TaskKind.PRE_TO_POST: 0.5}, seed=3
    )
    out = list(mixer.mix(commits))
    assert len(out) == 6
    for task, ex in out:
        assert task in (TaskKind.COMMIT_DIFF, TaskKind.PRE_TO_POST)
        assert int(ex.input_ids.shape[0]) == int(ex.loss_mask.shape[0])

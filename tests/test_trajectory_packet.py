"""Tests for the TrajectoryPacket contract + real golden-mini loading."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from cppmega_mlx.data.trajectory_packet import (
    Transition,
    TrajectoryPacket,
    load_golden_mini_transitions,
)

_GOLDEN_COMMITS = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "golden_mini"
    / "commits"
    / "commits.parquet"
)


def _arr(values: list[int]) -> mx.array:
    return mx.array(values, dtype=mx.int32)


# --------------------------------------------------------------------------- #
# Contract validation
# --------------------------------------------------------------------------- #
def test_transition_basic_construction() -> None:
    t = Transition(obs=_arr([1, 2, 3]), action=_arr([8]), next_obs=_arr([4, 5]))
    assert t.obs.shape == (3,)
    assert t.action.shape == (1,)
    assert t.next_obs.shape == (2,)
    # Real transitions are label-free.
    assert t.reward is None
    assert t.done is None
    assert t.is_synthetic is False


def test_transition_token_aligned_channels_validated() -> None:
    # change_mask aligns to next_obs (len 2). A len-3 mask must RAISE.
    with pytest.raises(ValueError, match="token-aligned length"):
        Transition(
            obs=_arr([1, 2, 3]),
            action=_arr([8]),
            next_obs=_arr([4, 5]),
            change_mask=_arr([1, 1, 0]),
        )


def test_transition_rejects_empty_obs() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Transition(
            obs=mx.array([], dtype=mx.int32),
            action=_arr([2]),
            next_obs=_arr([1]),
        )


def test_transition_rejects_missing_action() -> None:
    with pytest.raises(TypeError, match="Transition.action must be an mx.array"):
        Transition(obs=_arr([1]), action=None, next_obs=_arr([2]))  # type: ignore[arg-type]


def test_transition_rejects_non_integer_action() -> None:
    with pytest.raises(TypeError, match="action must contain integer token ids"):
        Transition(
            obs=_arr([1]),
            action=mx.array([0.5], dtype=mx.float32),
            next_obs=_arr([2]),
        )


def test_transition_rejects_negative_action_token() -> None:
    with pytest.raises(ValueError, match="action token ids must be non-negative"):
        Transition(obs=_arr([1]), action=_arr([-1]), next_obs=_arr([2]))


def test_real_transition_with_label_raises() -> None:
    # RULE #1: a non-synthetic transition carrying reward/done is a fabricated
    # label and MUST raise.
    with pytest.raises(ValueError, match="no fabricated labels"):
        Transition(
            obs=_arr([1, 2]),
            action=_arr([8]),
            next_obs=_arr([3]),
            reward=1.0,
            is_synthetic=False,
        )


def test_with_synthetic_control_attaches_label() -> None:
    # Synthetic labels are explicitly opt-in and clearly flagged.
    base = Transition(obs=_arr([1, 2]), action=_arr([8]), next_obs=_arr([3, 4]))
    syn = base.with_synthetic_control(reward=0.5, done=True)
    assert syn.is_synthetic is True
    assert syn.reward == 0.5
    assert syn.done is True
    assert syn.metadata["synthetic_control"] is True
    assert bool(mx.array_equal(syn.action, base.action).item())
    # The original is untouched and still label-free.
    assert base.reward is None and base.done is None


def test_trajectory_packet_requires_transitions() -> None:
    with pytest.raises(ValueError, match="at least one Transition"):
        TrajectoryPacket(transitions=())


def test_trajectory_horizon() -> None:
    steps = (
        Transition(obs=_arr([1]), action=_arr([7]), next_obs=_arr([2])),
        Transition(obs=_arr([2]), action=_arr([8]), next_obs=_arr([3])),
    )
    traj = TrajectoryPacket(transitions=steps)
    assert traj.horizon == 2
    assert len(traj) == 2
    assert traj.has_real_transitions() is True


# --------------------------------------------------------------------------- #
# Real golden-mini loading (NOT synthetic)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not _GOLDEN_COMMITS.exists(), reason="golden_mini commits fixture missing"
)
def test_load_golden_mini_real_transitions() -> None:
    trajectories = load_golden_mini_transitions(_GOLDEN_COMMITS)
    # The fixture has 2 commit pairs; only the post-edit docs yield transitions.
    assert len(trajectories) >= 1
    for traj in trajectories:
        assert isinstance(traj, TrajectoryPacket)
        assert traj.metadata["source"] == "golden_mini/commits"
        assert traj.has_real_transitions()
        for step in traj.transitions:
            # Real data: label-free, with edit supervision aligned to next_obs.
            assert step.is_synthetic is False
            assert step.reward is None
            assert step.done is None
            assert step.obs.shape[0] >= 1
            assert step.action.shape[0] >= 1
            assert step.next_obs.shape[0] >= 1
            assert step.change_mask is not None
            assert step.change_mask.shape[0] == step.next_obs.shape[0]
            # The next-observation region is the EDITED region: at least one
            # changed token is present (this is what makes it a transition).
            assert int(step.change_mask.sum().item()) >= 1


@pytest.mark.skipif(
    not _GOLDEN_COMMITS.exists(), reason="golden_mini commits fixture missing"
)
def test_load_golden_mini_pre_distinct_from_post() -> None:
    trajectories = load_golden_mini_transitions(_GOLDEN_COMMITS)
    for traj in trajectories:
        step = traj.transitions[0]
        # The pre-observation (context) and next-observation (edited region) are
        # distinct token spans of the same document.
        assert step.obs.shape[0] != step.next_obs.shape[0] or not bool(
            mx.array_equal(step.obs, step.next_obs).item()
        )

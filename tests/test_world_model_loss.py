"""Tests for the world-model loss: real next-obs learning + label provenance."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from cppmega_mlx.data.trajectory_packet import (
    Transition,
    TrajectoryPacket,
    load_golden_mini_transitions,
)
from cppmega_mlx.models.code_loop_world_model import (
    CodeLoopWorldModel,
    CodeLoopWorldModelConfig,
)
from cppmega_mlx.training.world_model import (
    WorldModelLossConfig,
    train_world_model,
    world_model_loss,
)

_GOLDEN_COMMITS = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "golden_mini"
    / "commits"
    / "commits.parquet"
)


def _real_trajectories() -> list[TrajectoryPacket]:
    return load_golden_mini_transitions(_GOLDEN_COMMITS)


def _model_for(trajectories: list[TrajectoryPacket], *, hidden: int = 48) -> CodeLoopWorldModel:
    maxlen = max(int(t.transitions[0].next_obs.shape[0]) for t in trajectories)
    maxobs = max(int(t.transitions[0].obs.shape[0]) for t in trajectories)
    cfg = CodeLoopWorldModelConfig(
        vocab_size=65536,
        hidden_size=hidden,
        num_heads=4,
        ffn_hidden_size=hidden * 2,
        max_seq_length=max(maxlen, maxobs) + 8,
        route_layers=2,
        train_loops=3,
    )
    return CodeLoopWorldModel(cfg)


# --------------------------------------------------------------------------- #
# (a) Next-obs loss DECREASES on REAL golden-mini commit transitions.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not _GOLDEN_COMMITS.exists(), reason="golden_mini commits fixture missing"
)
def test_next_obs_loss_decreases_on_real_transitions() -> None:
    mx.random.seed(0)
    trajectories = _real_trajectories()
    # All transitions used here are REAL (label-free) golden-mini commits.
    assert all(t.has_real_transitions() for t in trajectories)
    assert all(
        all(not step.is_synthetic for step in t.transitions) for t in trajectories
    )

    model = _model_for(trajectories)
    curve = train_world_model(model, trajectories, steps=40, learning_rate=3e-3)

    assert len(curve) == 40
    # Headline metric: next-observation prediction loss must drop substantially.
    assert curve[-1] < curve[0]
    assert curve[-1] < 0.5 * curve[0]


@pytest.mark.skipif(
    not _GOLDEN_COMMITS.exists(), reason="golden_mini commits fixture missing"
)
def test_real_loss_breakdown_has_no_fabricated_labels() -> None:
    # (d) NO fabricated labels: on REAL transitions the reward/done terms are
    # exactly zero and are flagged as not having run on any synthetic label.
    trajectories = _real_trajectories()
    model = _model_for(trajectories)
    for traj in trajectories:
        breakdown = world_model_loss(model, traj)
        assert breakdown.next_obs_ran_on_real is True
        assert breakdown.num_synthetic_transitions == 0
        assert breakdown.reward_ran_on_synthetic is False
        assert breakdown.done_ran_on_synthetic is False
        assert float(breakdown.reward.item()) == 0.0
        assert float(breakdown.done.item()) == 0.0


# --------------------------------------------------------------------------- #
# (e) latent-consistency + entropy-reg present (computed on real data).
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not _GOLDEN_COMMITS.exists(), reason="golden_mini commits fixture missing"
)
def test_latent_consistency_and_entropy_present() -> None:
    trajectories = _real_trajectories()
    model = _model_for(trajectories)
    breakdown = world_model_loss(model, trajectories[0])
    # Both regularizers are real finite numbers folded into the total.
    assert mx.isfinite(breakdown.latent_consistency).item()
    assert mx.isfinite(breakdown.entropy).item()
    assert float(breakdown.entropy.item()) > 0.0  # a fresh model has nonzero entropy

    # The total reflects the configured combination of all terms.
    cfg = WorldModelLossConfig()
    expected = (
        cfg.next_obs_weight * float(breakdown.next_obs.item())
        + cfg.latent_consistency_weight * float(breakdown.latent_consistency.item())
        - cfg.entropy_weight * float(breakdown.entropy.item())
        + cfg.reward_weight * float(breakdown.reward.item())
        + cfg.done_weight * float(breakdown.done.item())
    )
    assert abs(float(breakdown.total.item()) - expected) < 1e-3


# --------------------------------------------------------------------------- #
# (d) reward/done heads trained ONLY where a SYNTHETIC label is provided.
# This section uses CLEARLY-SYNTHETIC labeled data (not real commit data).
# --------------------------------------------------------------------------- #
def test_reward_done_run_only_on_synthetic_labels() -> None:
    # Build a transition from a REAL golden-mini obs/next but attach an EXPLICIT
    # synthetic control label. The reward/done terms only fire because the label
    # is flagged synthetic.
    mx.random.seed(0)
    trajectories = _real_trajectories() if _GOLDEN_COMMITS.exists() else None
    if trajectories is not None:
        base = trajectories[0].transitions[0]
        model = _model_for(trajectories)
    else:  # pragma: no cover - fixture always present in this repo
        base = Transition(
            obs=mx.array([1, 2, 3, 4], dtype=mx.int32),
            next_obs=mx.array([5, 6, 7], dtype=mx.int32),
        )
        model = CodeLoopWorldModel(
            CodeLoopWorldModelConfig(vocab_size=64, hidden_size=32, max_seq_length=64)
        )

    # SYNTHETIC label (clearly synthetic): reward=1.0, done=True.
    syn = base.with_synthetic_control(reward=1.0, done=True)
    syn_traj = TrajectoryPacket(transitions=(syn,))
    breakdown = world_model_loss(model, syn_traj)

    assert breakdown.num_synthetic_transitions == 1
    assert breakdown.num_real_transitions == 0
    assert breakdown.reward_ran_on_synthetic is True
    assert breakdown.done_ran_on_synthetic is True
    assert float(breakdown.reward.item()) > 0.0
    assert float(breakdown.done.item()) > 0.0


def test_real_transition_in_loss_never_touches_control_heads() -> None:
    # A SYNTHETIC-shaped but UNLABELED transition is treated as real and never
    # contributes a reward/done term.
    base = Transition(
        obs=mx.array([1, 2, 3, 4], dtype=mx.int32),
        next_obs=mx.array([5, 6, 7], dtype=mx.int32),
        change_mask=mx.array([1, 1, 0], dtype=mx.int32),
    )
    traj = TrajectoryPacket(transitions=(base,))
    model = CodeLoopWorldModel(
        CodeLoopWorldModelConfig(vocab_size=64, hidden_size=32, max_seq_length=64)
    )
    breakdown = world_model_loss(model, traj)
    assert breakdown.num_real_transitions == 1
    assert breakdown.reward_ran_on_synthetic is False
    assert breakdown.done_ran_on_synthetic is False
    assert float(breakdown.reward.item()) == 0.0
    assert float(breakdown.done.item()) == 0.0

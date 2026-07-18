"""Training objective + loop for the code-edit world model.

The world-model loss has four parts:

  1. ``next_obs`` — masked next-observation (post-edit) cross-entropy conditioned
     on the packet's required action.  This is the core dynamics objective: the
     rolled latent must decode the real post-edit token region.  Where a
     transition carries ``change_mask`` the loss is weighted toward the
     inserted/modified tokens (the actual edit).
  2. ``latent_consistency`` — the rolled latent should match the latent obtained
     by re-encoding the teacher next observation (a forward-prediction
     consistency term, MSE).
  3. ``entropy`` — entropy regularizer on the next-observation predictive
     distribution (subtracted with a small coefficient to discourage collapse).
  4. ``reward`` / ``done`` — control-head losses applied ONLY to transitions
     flagged ``is_synthetic=True`` with explicit labels.  RULE #1: real
     transitions carry NO reward/done label, so these terms are exactly zero
     (and contribute nothing to grads) unless a synthetic label is present.  A
     non-synthetic transition that somehow carries a label RAISES.

``WorldModelLossBreakdown`` reports which terms ran on real vs synthetic data so
the verifier can confirm "no fabricated labels".
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from cppmega_mlx.data.trajectory_packet import TrajectoryPacket, Transition
from cppmega_mlx.models.code_loop_world_model import CodeLoopWorldModel


@dataclass(frozen=True)
class WorldModelLossConfig:
    """Coefficients for the world-model loss terms."""

    next_obs_weight: float = 1.0
    changed_token_weight: float = 2.0  # extra weight on inserted/modified tokens
    latent_consistency_weight: float = 0.1
    entropy_weight: float = 1e-3
    reward_weight: float = 1.0
    done_weight: float = 1.0
    training_loops: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "next_obs_weight",
            "changed_token_weight",
            "latent_consistency_weight",
            "entropy_weight",
            "reward_weight",
            "done_weight",
        ):
            value = float(getattr(self, name))
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value}")


@dataclass
class WorldModelLossBreakdown:
    """Per-term loss values + provenance (real vs synthetic counts)."""

    total: mx.array
    next_obs: mx.array
    latent_consistency: mx.array
    entropy: mx.array
    reward: mx.array
    done: mx.array
    num_real_transitions: int
    num_synthetic_transitions: int
    next_obs_ran_on_real: bool
    reward_ran_on_synthetic: bool
    done_ran_on_synthetic: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "total": float(self.total.item()),
            "next_obs": float(self.next_obs.item()),
            "latent_consistency": float(self.latent_consistency.item()),
            "entropy": float(self.entropy.item()),
            "reward": float(self.reward.item()),
            "done": float(self.done.item()),
            "num_real_transitions": self.num_real_transitions,
            "num_synthetic_transitions": self.num_synthetic_transitions,
            "next_obs_ran_on_real": self.next_obs_ran_on_real,
            "reward_ran_on_synthetic": self.reward_ran_on_synthetic,
            "done_ran_on_synthetic": self.done_ran_on_synthetic,
        }


def _transition_next_obs_loss(
    model: CodeLoopWorldModel,
    transition: Transition,
    cfg: WorldModelLossConfig,
) -> tuple[mx.array, mx.array, mx.array]:
    """Return ``(next_obs_loss, entropy, rolled_latent)`` for one transition."""

    obs = transition.obs[None, :]  # (1, S)
    action = transition.action[None, :]  # (1, A)
    target = transition.next_obs[None, :]  # (1, L)
    length = int(transition.next_obs.shape[0])
    logits, rolled = model.predict_next(
        obs, action, length, training_loops=cfg.training_loops
    )

    token_ce = nn.losses.cross_entropy(
        logits.astype(mx.float32), target, reduction="none"
    )[0]  # (L,)

    # Per-token weight: changed tokens get changed_token_weight, others 1.0.
    if transition.change_mask is not None:
        mask = transition.change_mask.astype(mx.float32)
        weight = 1.0 + (cfg.changed_token_weight - 1.0) * mask
    else:
        weight = mx.ones_like(token_ce)
    denom = mx.maximum(weight.sum(), mx.array(1.0, dtype=mx.float32))
    next_obs_loss = (token_ce * weight).sum() / denom

    # Entropy of the predictive distribution (mean over positions).
    log_probs = logits[0].astype(mx.float32) - mx.logsumexp(
        logits[0].astype(mx.float32), axis=-1, keepdims=True
    )
    probs = mx.exp(log_probs)
    entropy = -(probs * log_probs).sum(axis=-1).mean()

    return next_obs_loss, entropy, rolled


def world_model_loss(
    model: CodeLoopWorldModel,
    trajectory: TrajectoryPacket,
    cfg: WorldModelLossConfig | None = None,
) -> WorldModelLossBreakdown:
    """Compute the full world-model loss over one trajectory.

    Sums the next-obs / latent-consistency / entropy terms over every transition;
    the reward/done terms are added ONLY for transitions flagged synthetic with a
    label.  Returns a :class:`WorldModelLossBreakdown` with provenance flags.
    """

    cfg = cfg or WorldModelLossConfig()
    transitions = list(trajectory.transitions)

    zero = mx.array(0.0, dtype=mx.float32)
    next_obs_sum = zero
    consistency_sum = zero
    entropy_sum = zero
    reward_sum = zero
    done_sum = zero

    n_real = 0
    n_syn = 0
    n_reward = 0
    n_done = 0

    for transition in transitions:
        next_obs_loss, entropy, rolled = _transition_next_obs_loss(model, transition, cfg)
        next_obs_sum = next_obs_sum + next_obs_loss
        entropy_sum = entropy_sum + entropy

        # Latent-consistency: rolled latent vs latent re-encoded from the teacher
        # next observation (stop-grad on the teacher target).
        target_latent = model.encode(transition.next_obs[None, :])
        target_latent = mx.stop_gradient(target_latent)
        consistency_sum = consistency_sum + mx.mean((rolled - target_latent) ** 2)

        if transition.is_synthetic:
            n_syn += 1
            # RULE #1: control heads run ONLY where a synthetic label exists.
            if transition.reward is not None:
                pred_r = model.reward(rolled)[0]
                tgt_r = mx.array(float(transition.reward), dtype=mx.float32)
                reward_sum = reward_sum + (pred_r - tgt_r) ** 2
                n_reward += 1
            if transition.done is not None:
                pred_d = model.done_logit(rolled)[0]
                tgt_d = mx.array(1.0 if transition.done else 0.0, dtype=mx.float32)
                done_sum = done_sum + nn.losses.binary_cross_entropy(
                    pred_d, tgt_d, with_logits=True
                )
                n_done += 1
        else:
            n_real += 1
            # Real transitions must NOT carry labels (Transition.__post_init__
            # already enforces this; assert here as a second guard).
            if transition.reward is not None or transition.done is not None:
                raise ValueError(
                    "real (non-synthetic) transition carries reward/done; "
                    "fabricated labels are forbidden (RULE #1)"
                )

    count = max(len(transitions), 1)
    next_obs_mean = next_obs_sum / count
    consistency_mean = consistency_sum / count
    entropy_mean = entropy_sum / count
    reward_mean = reward_sum / n_reward if n_reward else zero
    done_mean = done_sum / n_done if n_done else zero

    total = (
        cfg.next_obs_weight * next_obs_mean
        + cfg.latent_consistency_weight * consistency_mean
        - cfg.entropy_weight * entropy_mean
        + cfg.reward_weight * reward_mean
        + cfg.done_weight * done_mean
    )

    return WorldModelLossBreakdown(
        total=total,
        next_obs=next_obs_mean,
        latent_consistency=consistency_mean,
        entropy=entropy_mean,
        reward=reward_mean,
        done=done_mean,
        num_real_transitions=n_real,
        num_synthetic_transitions=n_syn,
        next_obs_ran_on_real=n_real > 0,
        reward_ran_on_synthetic=n_reward > 0,
        done_ran_on_synthetic=n_done > 0,
    )


def train_world_model(
    model: CodeLoopWorldModel,
    trajectories: list[TrajectoryPacket],
    *,
    steps: int,
    learning_rate: float = 1e-3,
    cfg: WorldModelLossConfig | None = None,
) -> list[float]:
    """Train the world model for ``steps`` steps; return the next-obs loss curve.

    Each step sums the world-model loss over all trajectories and applies one
    Adam update.  The returned list is the per-step MEAN next-observation loss
    (the headline metric the verifier checks for a decrease on real transitions).
    """

    import mlx.optimizers as optim

    cfg = cfg or WorldModelLossConfig()
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if not trajectories:
        raise ValueError("train_world_model requires at least one trajectory")

    optimizer = optim.Adam(learning_rate=learning_rate)

    def batch_loss(m: CodeLoopWorldModel) -> tuple[mx.array, mx.array]:
        total = mx.array(0.0, dtype=mx.float32)
        next_obs_total = mx.array(0.0, dtype=mx.float32)
        for traj in trajectories:
            breakdown = world_model_loss(m, traj, cfg)
            total = total + breakdown.total
            next_obs_total = next_obs_total + breakdown.next_obs
        n = len(trajectories)
        return total / n, next_obs_total / n

    loss_and_grad = nn.value_and_grad(model, batch_loss)

    curve: list[float] = []
    for _ in range(steps):
        (_, next_obs_mean), grads = loss_and_grad(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        curve.append(float(next_obs_mean.item()))
    return curve


__all__ = [
    "WorldModelLossBreakdown",
    "WorldModelLossConfig",
    "train_world_model",
    "world_model_loss",
]

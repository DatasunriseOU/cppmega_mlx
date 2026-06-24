"""Tests for the CodeLoopWorldModel: stable-loop dynamics + deferred rollout."""

from __future__ import annotations

import mlx.core as mx
import pytest

from cppmega_mlx.models.code_loop_world_model import (
    CodeLoopWorldModel,
    CodeLoopWorldModelConfig,
)
from cppmega_mlx.nn.stable_loop import StableFixedPointLoop


def _model(**overrides) -> CodeLoopWorldModel:
    cfg = CodeLoopWorldModelConfig(
        vocab_size=64,
        hidden_size=32,
        num_heads=4,
        ffn_hidden_size=64,
        max_seq_length=64,
        route_layers=2,
        train_loops=3,
        **overrides,
    )
    return CodeLoopWorldModel(cfg)


def test_dynamics_core_is_stable_fixed_point_loop() -> None:
    # HARD requirement (c): the dynamics core IS StableFixedPointLoop, not a
    # reimplemented loop.
    model = _model()
    assert isinstance(model.dynamics, StableFixedPointLoop)
    assert model.dynamics.n_sublayers == model.config.n_sublayers


def test_encode_shapes() -> None:
    model = _model()
    obs = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
    latent = model.encode(obs)
    assert latent.shape == (1, model.config.hidden_size)


def test_step_latent_preserves_shape() -> None:
    model = _model()
    latent = mx.zeros((2, model.config.hidden_size))
    rolled = model.step_latent(latent, training_loops=2)
    assert rolled.shape == latent.shape


def test_predict_next_shapes() -> None:
    model = _model()
    obs = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    logits, latent = model.predict_next(obs, length=6)
    assert logits.shape == (1, 6, model.config.vocab_size)
    assert latent.shape == (1, model.config.hidden_size)


def test_deferred_rollout_terminal_equals_per_step_decode() -> None:
    # HARD requirement (b): deferred rollout decodes only at the terminal step
    # and EQUALS the per-step decode at the same step (shape AND value).
    mx.random.seed(0)
    model = _model()
    obs = mx.array([[3, 1, 4, 1, 5, 9, 2, 6]], dtype=mx.int32)
    horizon, length = 4, 5

    terminal_logits, terminal_latent = model.rollout(obs, horizon=horizon, length=length)
    per_step = model.rollout_decode_all(obs, horizon=horizon, length=length)

    assert len(per_step) == horizon
    # Every per-step decode has the SAME shape as the terminal deferred decode.
    for step_logits in per_step:
        assert step_logits.shape == terminal_logits.shape
    # The terminal entry equals the deferred rollout exactly (same latent).
    assert bool(mx.allclose(terminal_logits, per_step[-1]).item())


def test_rollout_horizon_one_matches_predict_next() -> None:
    mx.random.seed(1)
    model = _model()
    obs = mx.array([[2, 4, 6, 8]], dtype=mx.int32)
    logits_rollout, _ = model.rollout(obs, horizon=1, length=4)
    logits_predict, _ = model.predict_next(obs, length=4)
    assert bool(mx.allclose(logits_rollout, logits_predict).item())


def test_control_heads_emit_scalars() -> None:
    model = _model()
    latent = mx.zeros((3, model.config.hidden_size))
    assert model.reward(latent).shape == (3,)
    assert model.done_logit(latent).shape == (3,)


def test_decode_length_validation() -> None:
    model = _model()
    latent = mx.zeros((1, model.config.hidden_size))
    with pytest.raises(ValueError, match="decode length"):
        model.decode(latent, 0)


def test_invalid_config_raises() -> None:
    with pytest.raises(ValueError, match="divisible by num_heads"):
        CodeLoopWorldModelConfig(hidden_size=33, num_heads=4)

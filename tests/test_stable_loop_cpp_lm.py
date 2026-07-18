"""Tests for the tiny FPRM stable-looped LM and its deep-supervision step."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pytest

from cppmega_mlx.models.stable_loop_cpp_lm import (
    StableLoopCppLM,
    StableLoopCppLMConfig,
    StableLoopCppLMInferenceResult,
)
from cppmega_mlx.nn.stable_loop import FixedPointConvergenceError
from cppmega_mlx.training.fixed_point import truncated_bptt_step


def _tiny_config(**kw) -> StableLoopCppLMConfig:
    base = dict(
        vocab_size=16,
        hidden_size=16,
        num_heads=2,
        ffn_hidden_size=32,
        max_seq_length=8,
        route_layers=2,
        prelude_blocks=1,
        train_loops=4,
    )
    base.update(kw)
    return StableLoopCppLMConfig(**base)


def test_forward_runs_and_shapes():
    cfg = _tiny_config()
    model = StableLoopCppLM(cfg)
    ids = mx.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
    logits = model(ids)
    mx.eval(logits)
    assert logits.shape == (2, 5, cfg.vocab_size)
    assert mx.isfinite(logits).all().item()


def test_n_sublayers_is_two_l():
    cfg = _tiny_config(route_layers=3)
    model = StableLoopCppLM(cfg)
    assert cfg.n_sublayers == 6
    assert model.loop.n_sublayers == 6
    assert len(model.route_core.sublayers) == 6
    assert len(model.route_core.norms) == 6


def test_inference_fixed_point_mode_runs():
    cfg = _tiny_config()
    model = StableLoopCppLM(cfg)
    ids = mx.array([[1, 2, 3, 4]])
    logits = model(ids, training_loops=-1)  # inference fixed-point + FPOPT
    mx.eval(logits)
    assert logits.shape == (1, 4, cfg.vocab_size)
    assert mx.isfinite(logits).all().item()


def test_inference_api_preserves_convergence_result():
    model = StableLoopCppLM(_tiny_config(tau=10.0, max_loops=2))
    ids = mx.array([[1, 2, 3, 4]])

    result = model(ids, training_loops=-1, return_convergence=True)
    mx.eval(result.logits, result.state)

    assert isinstance(result, StableLoopCppLMInferenceResult)
    assert result.convergence.converged is True
    assert result.convergence.steps == 1
    assert result.state is result.convergence.state
    assert result.logits.shape == (1, 4, model.config.vocab_size)


def test_inference_api_is_strict_unless_best_effort_is_explicit():
    mx.random.seed(17)
    model = StableLoopCppLM(_tiny_config(tau=1e-30, max_loops=1))
    ids = mx.array([[1, 2, 3, 4]])

    with pytest.raises(FixedPointConvergenceError):
        model(ids, training_loops=-1)

    result = model(
        ids,
        training_loops=-1,
        return_convergence=True,
        best_effort=True,
    )
    assert isinstance(result, StableLoopCppLMInferenceResult)
    assert result.convergence.converged is False
    assert result.convergence.steps == 1


def _loss_fn(logits: mx.array, targets: mx.array) -> mx.array:
    B, S, V = logits.shape
    return nn.losses.cross_entropy(
        logits.reshape(B * S, V), targets.reshape(B * S), reduction="mean"
    )


def test_deep_supervision_step_lowers_loss_on_memorization():
    """A tiny memorization task: the deep-supervision truncated-BPTT step must
    drive the loss down over a handful of optimizer steps."""
    mx.random.seed(0)
    cfg = _tiny_config()
    model = StableLoopCppLM(cfg)

    # Fixed tiny batch to memorize (next-token shift).
    seq = mx.array([[1, 2, 3, 4, 5, 6, 7]])
    inputs = seq[:, :-1]
    targets = seq[:, 1:]

    optimizer = optim.Adam(learning_rate=3e-2)

    seq_len = inputs.shape[1]
    mask = model._causal_mask(seq_len, mx.float32)

    def head(z: mx.array) -> mx.array:
        return model.head(z, mask)

    # Embedding + prelude build the loop input x. Pass it as a BUILDER so the
    # embedding/prelude parameters stay inside the differentiated closure and
    # receive gradients (deep supervision through the full upstream stack).
    def make_x():
        hidden = model.embed(inputs)
        return model.run_prelude(hidden, mask)

    losses = []
    for _ in range(60):
        result = truncated_bptt_step(
            model.loop,
            model,
            make_x,  # z0 builder
            make_x,  # x builder
            head,
            targets,
            _loss_fn,
            optimizer,
            mask,
            window=cfg.train_loops,
            num_windows=1,
        )
        losses.append(result["loss"])

    assert losses[-1] < losses[0], f"loss did not drop: {losses[0]} -> {losses[-1]}"
    # Memorization on a 1-sample batch should reach a clearly lower loss.
    assert losses[-1] < 0.5 * losses[0]


def test_end_to_end_value_and_grad_train_step_lowers_loss():
    """Plain end-to-end value_and_grad over the whole model (fixed train_loops)
    also lowers loss on the memorization task — exercises model.__call__."""
    mx.random.seed(1)
    cfg = _tiny_config()
    model = StableLoopCppLM(cfg)
    seq = mx.array([[2, 3, 4, 5, 6, 7, 1]])
    inputs, targets = seq[:, :-1], seq[:, 1:]

    optimizer = optim.Adam(learning_rate=3e-2)

    def loss_fn(m):
        logits = m(inputs)
        return _loss_fn(logits, targets)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    first = None
    last = None
    for step in range(40):
        loss, grads = loss_and_grad(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        if step == 0:
            first = float(loss.item())
        last = float(loss.item())
    assert last < first, f"loss did not drop: {first} -> {last}"


def test_multiple_windows_deep_supervision_sums_losses():
    cfg = _tiny_config(train_loops=2)
    model = StableLoopCppLM(cfg)
    seq = mx.array([[1, 2, 3, 4, 5]])
    inputs, targets = seq[:, :-1], seq[:, 1:]
    optimizer = optim.Adam(learning_rate=1e-2)
    seq_len = inputs.shape[1]
    mask = model._causal_mask(seq_len, mx.float32)

    def head(z):
        return model.head(z, mask)

    x = model.run_prelude(model.embed(inputs), mask)
    result = truncated_bptt_step(
        model.loop,
        model,
        x,
        x,
        head,
        targets,
        _loss_fn,
        optimizer,
        mask,
        window=2,
        num_windows=3,
    )
    assert result["num_windows"] == 3
    assert len(result["window_losses"]) == 3
    assert result["loss"] == pytest.approx(sum(result["window_losses"]), rel=1e-5)
    assert result["optimizer_updates"] == 3
    assert result["loop_iterations"] == 6
    assert result["bptt_horizon"] == 2
    assert result["schedule"] == "online_per_window"


def test_truncated_bptt_carries_each_window_endpoint_without_double_advance():
    cfg = _tiny_config(train_loops=2)
    model = StableLoopCppLM(cfg)
    ids = mx.array([[1, 2, 3, 4]])
    targets = mx.array([[2, 3, 4, 5]])
    mask = model._causal_mask(ids.shape[1], mx.float32)
    x = model.run_prelude(model.embed(ids), mask)
    optimizer = optim.SGD(learning_rate=0.0)

    def head(z):
        return model.head(z, mask)

    expected_state = model.loop.forward(x, x, mask, training_loops=6)
    result = truncated_bptt_step(
        model.loop,
        model,
        x,
        x,
        head,
        targets,
        _loss_fn,
        optimizer,
        mask,
        window=2,
        num_windows=3,
    )
    mx.eval(expected_state, result["final_state"], optimizer.state)

    assert int(optimizer.state["step"].item()) == 3
    assert result["optimizer_updates"] == 3
    assert result["loop_iterations"] == 6
    assert result["bptt_horizon"] == 2
    assert mx.allclose(result["final_state"], expected_state, atol=1e-6).item()

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from cppmega_mlx.models.tiny_lm import TinyLM, TinyLMConfig
from cppmega_mlx.training.loss import (
    next_token_cross_entropy,
    next_token_cross_entropy_with_stp,
)
from cppmega_mlx.training.stp_loss import (
    DEFAULT_STP_LAMBDA,
    DEFAULT_STP_SPANS,
    STPLossConfig,
    compute_stp_loss,
    next_token_and_stp_loss,
)


def _scalar(value: mx.array) -> float:
    mx.eval(value)
    return float(value.item())


def _hidden(points: list[list[float]]) -> mx.array:
    return mx.array([[points]], dtype=mx.float32).reshape(1, len(points), len(points[0]))


def _tiny_model() -> TinyLM:
    return TinyLM(
        TinyLMConfig(
            vocab_size=16,
            hidden_size=8,
            num_layers=1,
            num_heads=2,
            ffn_hidden_size=16,
            max_seq_length=8,
            structure_vocab_size=8,
        )
    )


def test_stp_defaults_are_opt_in_and_single_span() -> None:
    config = STPLossConfig()

    assert config.n_spans == DEFAULT_STP_SPANS == 1
    assert config.loss_weight == DEFAULT_STP_LAMBDA == 0.0


def test_stp_loss_is_near_zero_for_straight_trajectory() -> None:
    loss = compute_stp_loss(_hidden([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]))

    assert math.isclose(_scalar(loss), 0.0, abs_tol=1e-6)


def test_stp_loss_is_near_two_for_opposed_direction() -> None:
    loss = compute_stp_loss(_hidden([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]))

    assert math.isclose(_scalar(loss), 2.0, abs_tol=1e-6)


def test_stp_loss_is_near_one_for_orthogonal_bend() -> None:
    loss = compute_stp_loss(_hidden([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]))

    assert math.isclose(_scalar(loss), 1.0, abs_tol=1e-6)


def test_stp_loss_returns_zero_for_short_sequences() -> None:
    hidden_states = mx.ones((2, 2, 4), dtype=mx.float32)

    loss = compute_stp_loss(hidden_states, n_spans=3)

    assert math.isclose(_scalar(loss), 0.0, abs_tol=1e-7)


def test_stp_loss_averages_tuple_layers_and_multiple_spans() -> None:
    straight = _hidden(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
        ]
    )
    opposed = _hidden(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
        ]
    )

    loss = compute_stp_loss((straight, opposed), n_spans=2)

    assert math.isclose(_scalar(loss), 1.0, abs_tol=1e-6)


def test_stp_loss_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="hidden_states must be shaped"):
        compute_stp_loss(mx.ones((3, 4), dtype=mx.float32))
    with pytest.raises(TypeError, match="n_spans must be an integer"):
        compute_stp_loss(mx.ones((1, 3, 2), dtype=mx.float32), n_spans=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_spans must be non-negative"):
        compute_stp_loss(mx.ones((1, 3, 2), dtype=mx.float32), n_spans=-1)


def test_next_token_and_stp_loss_composes_lambda_weight() -> None:
    total = next_token_and_stp_loss(
        mx.array(5.0, dtype=mx.float32),
        mx.array(2.0, dtype=mx.float32),
        loss_weight=0.25,
    )

    assert math.isclose(_scalar(total), 5.5, abs_tol=1e-6)
    with pytest.raises(ValueError, match="loss weight must be non-negative"):
        next_token_and_stp_loss(
            mx.array(1.0, dtype=mx.float32),
            mx.array(1.0, dtype=mx.float32),
            loss_weight=-0.1,
        )


def test_training_loss_with_stp_is_opt_in_and_reports_metrics() -> None:
    mx.random.seed(171)
    model = _tiny_model()
    batch = mx.array([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]], dtype=mx.int32)

    default_loss, default_ntokens = next_token_cross_entropy(model, batch)
    total_loss, ntokens, metrics = next_token_cross_entropy_with_stp(
        model,
        batch,
        config=STPLossConfig(n_spans=2, loss_weight=0.4),
    )
    mx.eval(
        default_loss,
        default_ntokens,
        total_loss,
        ntokens,
        metrics.next_token_loss,
        metrics.stp_loss,
        metrics.total_loss,
    )

    assert metrics.n_spans == 2
    assert metrics.loss_weight == 0.4
    assert int(ntokens.item()) == int(default_ntokens.item())
    assert math.isclose(
        float(metrics.next_token_loss.item()),
        float(default_loss.item()),
        rel_tol=1e-6,
    )
    assert math.isclose(
        float(total_loss.item()),
        float(default_loss.item()) + 0.4 * float(metrics.stp_loss.item()),
        rel_tol=1e-6,
    )
    assert math.isfinite(float(metrics.stp_loss.item()))
    assert np.isfinite(np.array(model(batch[:, :-1]))).all()

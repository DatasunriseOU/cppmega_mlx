from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from cppmega_mlx.data.batch import synthetic_token_batch
from cppmega_mlx.models.tiny_lm import TinyLM, TinyLMConfig
from cppmega_mlx.training.eval import evaluate_batches


def _tiny_config() -> TinyLMConfig:
    return TinyLMConfig(
        vocab_size=48,
        hidden_size=16,
        num_layers=1,
        num_heads=4,
        ffn_hidden_size=32,
        max_seq_length=16,
        structure_vocab_size=16,
    )


def _flat_params(model: TinyLM) -> dict[str, np.ndarray]:
    mx.eval(model.parameters())
    return {name: np.array(value) for name, value in tree_flatten(model.parameters())}


def test_evaluate_batches_reports_weighted_loss_and_throughput() -> None:
    model = TinyLM(_tiny_config())
    batches = [
        synthetic_token_batch(
            batch_size=2,
            seq_length=8,
            vocab_size=model.config.vocab_size,
            seed=seed,
            include_structure=True,
        )
        for seed in (1, 2)
    ]

    metrics = evaluate_batches(model, batches)

    assert metrics.batches == 2
    assert metrics.ntokens == 28
    assert math.isfinite(metrics.loss)
    assert metrics.loss > 0
    assert metrics.seconds >= 0
    assert metrics.tokens_per_second > 0


def test_evaluate_batches_does_not_update_parameters() -> None:
    model = TinyLM(_tiny_config())
    batches = [
        synthetic_token_batch(
            batch_size=2,
            seq_length=7,
            vocab_size=model.config.vocab_size,
            seed=7,
            include_structure=True,
        )
    ]
    before = _flat_params(model)

    metrics = evaluate_batches(model, batches)
    after = _flat_params(model)

    assert metrics.ntokens == 12
    for name, expected in before.items():
        np.testing.assert_allclose(after[name], expected, rtol=0, atol=0)


def test_evaluate_batches_accepts_mapping_batches() -> None:
    model = TinyLM(_tiny_config())
    batch = synthetic_token_batch(
        batch_size=1,
        seq_length=6,
        vocab_size=model.config.vocab_size,
        seed=11,
        include_structure=True,
    ).as_dict()

    metrics = evaluate_batches(model, [batch])

    assert metrics.batches == 1
    assert metrics.ntokens == 5
    assert math.isfinite(metrics.loss)


def test_evaluate_batches_rejects_empty_iterable() -> None:
    model = TinyLM(_tiny_config())

    with pytest.raises(ValueError, match="at least one batch"):
        evaluate_batches(model, [])


def test_evaluate_batches_rejects_zero_token_loss_results() -> None:
    model = TinyLM(_tiny_config())
    batch = synthetic_token_batch(
        batch_size=1,
        seq_length=6,
        vocab_size=model.config.vocab_size,
        seed=13,
        include_structure=False,
    )

    def zero_token_loss(_model: Any, _batch: Any) -> tuple[mx.array, mx.array]:
        return mx.array(0.0, dtype=mx.float32), mx.array(0, dtype=mx.int32)

    with pytest.raises(ValueError, match="zero tokens"):
        evaluate_batches(model, [batch], loss_fn=zero_token_loss)

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

import cppmega_mlx.inference as inference
from cppmega_mlx.inference import sample_next_token


def _as_numpy(tokens: mx.array) -> np.ndarray:
    mx.eval(tokens)
    return np.array(tokens)


def test_greedy_temperature_zero_returns_argmax_with_batch_column_shape() -> None:
    logits = mx.array(
        [
            [0.0, 4.0, 1.0],
            [3.0, 2.0, 5.0],
        ],
        dtype=mx.float32,
    )

    tokens = sample_next_token(logits, temperature=0.0)

    assert tokens.shape == (2, 1)
    np.testing.assert_array_equal(_as_numpy(tokens), np.array([[1], [2]], dtype=np.uint32))


def test_rejects_invalid_temperature_and_top_p() -> None:
    logits = mx.array([[0.0, 1.0]], dtype=mx.float32)

    with pytest.raises(ValueError, match="temperature"):
        sample_next_token(logits, temperature=-0.1)
    with pytest.raises(ValueError, match="top_p"):
        sample_next_token(logits, top_p=0.0)
    with pytest.raises(ValueError, match="top_p"):
        sample_next_token(logits, top_p=1.1)


def test_rejects_non_matrix_logits() -> None:
    with pytest.raises(ValueError, match="shape"):
        sample_next_token(mx.array([0.0, 1.0], dtype=mx.float32))


def test_non_positive_top_k_disables_top_k_filter_like_nanochat() -> None:
    logits = mx.array([[0.0, 1.0, 10.0]], dtype=mx.float32)

    disabled_zero = sample_next_token(
        logits,
        top_k=0,
        rng_key=mx.random.key(7),
    )
    disabled_negative = sample_next_token(
        logits,
        top_k=-4,
        rng_key=mx.random.key(7),
    )
    no_top_k = sample_next_token(logits, top_k=None, rng_key=mx.random.key(7))

    np.testing.assert_array_equal(_as_numpy(disabled_zero), _as_numpy(no_top_k))
    np.testing.assert_array_equal(_as_numpy(disabled_negative), _as_numpy(no_top_k))


def test_top_k_one_always_keeps_only_top_token() -> None:
    logits = mx.array(
        [
            [0.0, 9.0, 8.0],
            [7.0, 3.0, 4.0],
        ],
        dtype=mx.float32,
    )

    for seed in range(5):
        tokens = sample_next_token(logits, top_k=1, rng_key=mx.random.key(seed))
        np.testing.assert_array_equal(
            _as_numpy(tokens),
            np.array([[1], [0]], dtype=np.int32),
        )


def test_tiny_top_p_keeps_first_sorted_candidate() -> None:
    logits = mx.array([[4.0, 3.0, 2.0]], dtype=mx.float32)

    tokens = sample_next_token(
        logits,
        top_p=1e-6,
        rng_key=mx.random.key(11),
    )

    np.testing.assert_array_equal(_as_numpy(tokens), np.array([[0]], dtype=np.int32))


def test_sampled_ids_stay_within_top_k_candidates() -> None:
    logits = mx.array([[10.0, 9.0, -100.0, -100.0]], dtype=mx.float32)

    observed = {
        int(_as_numpy(sample_next_token(logits, top_k=2, rng_key=mx.random.key(seed)))[0, 0])
        for seed in range(20)
    }

    assert observed <= {0, 1}


def test_top_p_runs_after_top_k_candidates() -> None:
    logits = mx.array([[100.0, 90.0, 80.0, 70.0]], dtype=mx.float32)

    tokens = sample_next_token(
        logits,
        top_k=3,
        top_p=0.7,
        rng_key=mx.random.key(19),
    )

    np.testing.assert_array_equal(_as_numpy(tokens), np.array([[0]], dtype=np.int32))


def test_seeded_sampling_is_deterministic() -> None:
    logits = mx.array([[1.0, 1.0, 1.0, 1.0]], dtype=mx.float32)

    left = sample_next_token(logits, rng_key=mx.random.key(123))
    right = sample_next_token(logits, rng_key=mx.random.key(123))

    np.testing.assert_array_equal(_as_numpy(left), _as_numpy(right))


def test_inference_root_exports_sampler() -> None:
    assert inference.sample_next_token is sample_next_token
    assert "sample_next_token" in inference.__all__

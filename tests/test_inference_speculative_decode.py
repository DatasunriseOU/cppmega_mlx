from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

import cppmega_mlx.inference as inference
from cppmega_mlx.inference import (
    speculative_acceptance,
    speculative_acceptance_batch,
    typical_acceptance,
    typical_acceptance_batch,
)


def _as_numpy(tokens: mx.array) -> np.ndarray:
    mx.eval(tokens)
    return np.array(tokens)


def test_speculative_acceptance_accepts_all_and_samples_target_tail() -> None:
    draft_logits = mx.array(
        [
            [0.0, 8.0, -4.0],
            [-3.0, 0.0, 8.0],
        ],
        dtype=mx.float32,
    )
    target_logits = mx.array(
        [
            [0.0, 9.0, -4.0],
            [-3.0, 0.0, 9.0],
            [10.0, 0.0, -2.0],
        ],
        dtype=mx.float32,
    )
    draft_tokens = mx.array([1, 2], dtype=mx.int32)

    accepted, n_accepted, next_token = speculative_acceptance(
        draft_logits,
        target_logits,
        draft_tokens,
        rng_key=mx.random.key(7),
    )

    assert n_accepted == 2
    np.testing.assert_array_equal(_as_numpy(accepted), np.array([1, 2], dtype=np.int32))
    np.testing.assert_array_equal(_as_numpy(next_token), np.array([0], dtype=np.int32))


def test_speculative_acceptance_rejects_and_samples_positive_residual() -> None:
    draft_logits = mx.array(
        [
            [10.0, 0.0, -10.0],
            [10.0, -10.0, -10.0],
        ],
        dtype=mx.float32,
    )
    target_logits = mx.array(
        [
            [-10.0, 10.0, -10.0],
            [0.0, 10.0, -10.0],
            [0.0, 0.0, 10.0],
        ],
        dtype=mx.float32,
    )
    draft_tokens = mx.array([0, 0], dtype=mx.int32)

    accepted, n_accepted, next_token = speculative_acceptance(
        draft_logits,
        target_logits,
        draft_tokens,
        rng_key=mx.random.key(13),
    )

    assert n_accepted == 0
    assert accepted.shape == (0,)
    np.testing.assert_array_equal(_as_numpy(next_token), np.array([1], dtype=np.int32))


def test_speculative_acceptance_residual_fallback_samples_target_distribution() -> None:
    draft_logits = mx.array([[float("-inf"), 0.0]], dtype=mx.float32)
    target_logits = mx.array(
        [[float("-inf"), 0.0], [8.0, 0.0]],
        dtype=mx.float32,
    )
    draft_tokens = mx.array([0], dtype=mx.int32)

    accepted, n_accepted, next_token = speculative_acceptance(
        draft_logits,
        target_logits,
        draft_tokens,
        rng_key=mx.random.key(3),
    )

    assert n_accepted == 0
    assert accepted.shape == (0,)
    np.testing.assert_array_equal(_as_numpy(next_token), np.array([1], dtype=np.int32))


def test_speculative_acceptance_temperature_zero_matches_reference_softmax_path() -> None:
    draft_logits = mx.array([[0.0, 6.0]], dtype=mx.float32)
    target_logits = mx.array([[0.0, 7.0], [8.0, 0.0]], dtype=mx.float32)
    draft_tokens = mx.array([1], dtype=mx.int32)

    accepted, n_accepted, next_token = speculative_acceptance(
        draft_logits,
        target_logits,
        draft_tokens,
        temperature=0.0,
        rng_key=mx.random.key(2),
    )

    assert n_accepted == 1
    np.testing.assert_array_equal(_as_numpy(accepted), np.array([1], dtype=np.int32))
    np.testing.assert_array_equal(_as_numpy(next_token), np.array([0], dtype=np.int32))


def test_speculative_acceptance_zero_draft_window_samples_target_head() -> None:
    draft_logits = mx.zeros((0, 3), dtype=mx.float32)
    target_logits = mx.array([[0.0, 0.0, 9.0]], dtype=mx.float32)
    draft_tokens = mx.array([], dtype=mx.int32)

    accepted, n_accepted, next_token = speculative_acceptance(
        draft_logits,
        target_logits,
        draft_tokens,
        rng_key=mx.random.key(5),
    )

    assert n_accepted == 0
    assert accepted.shape == (0,)
    np.testing.assert_array_equal(_as_numpy(next_token), np.array([2], dtype=np.int32))


def test_typical_acceptance_uses_threshold_and_samples_target_at_rejection() -> None:
    draft_logits = mx.array(
        [
            [0.0, 8.0, -4.0],
            [8.0, 0.0, -4.0],
        ],
        dtype=mx.float32,
    )
    target_logits = mx.array(
        [
            [0.0, 9.0, -4.0],
            [-2.0, 9.0, -4.0],
            [0.0, 0.0, 10.0],
        ],
        dtype=mx.float32,
    )
    draft_tokens = mx.array([1, 0], dtype=mx.int32)

    accepted, n_accepted, next_token = typical_acceptance(
        draft_logits,
        target_logits,
        draft_tokens,
        threshold=0.5,
        rng_key=mx.random.key(17),
    )

    assert n_accepted == 1
    np.testing.assert_array_equal(_as_numpy(accepted), np.array([1], dtype=np.int32))
    np.testing.assert_array_equal(_as_numpy(next_token), np.array([1], dtype=np.int32))


def test_typical_acceptance_threshold_zero_accepts_complete_window() -> None:
    draft_logits = mx.array(
        [
            [8.0, 0.0, -4.0],
            [0.0, 8.0, -4.0],
        ],
        dtype=mx.float32,
    )
    target_logits = mx.array(
        [
            [0.0, 8.0, -4.0],
            [8.0, 0.0, -4.0],
            [0.0, 0.0, 9.0],
        ],
        dtype=mx.float32,
    )
    draft_tokens = mx.array([0, 1], dtype=mx.int32)

    accepted, n_accepted, next_token = typical_acceptance(
        draft_logits,
        target_logits,
        draft_tokens,
        threshold=0.0,
        rng_key=mx.random.key(19),
    )

    assert n_accepted == 2
    np.testing.assert_array_equal(_as_numpy(accepted), np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(_as_numpy(next_token), np.array([2], dtype=np.int32))


def test_typical_acceptance_accepts_all_and_samples_tail() -> None:
    draft_logits = mx.array([[0.0, 8.0], [8.0, 0.0]], dtype=mx.float32)
    target_logits = mx.array([[0.0, 9.0], [9.0, 0.0], [0.0, 10.0]], dtype=mx.float32)
    draft_tokens = mx.array([1, 0], dtype=mx.int32)

    accepted, n_accepted, next_token = typical_acceptance(
        draft_logits,
        target_logits,
        draft_tokens,
        threshold=0.5,
        rng_key=mx.random.key(23),
    )

    assert n_accepted == 2
    np.testing.assert_array_equal(_as_numpy(accepted), np.array([1, 0], dtype=np.int32))
    np.testing.assert_array_equal(_as_numpy(next_token), np.array([1], dtype=np.int32))


def test_batched_speculative_acceptance_matches_rowwise_split_keys() -> None:
    draft_logits = mx.array(
        [
            [[0.0, 7.0, -4.0], [-4.0, 0.0, 7.0]],
            [[7.0, 0.0, -4.0], [0.0, 7.0, -4.0]],
        ],
        dtype=mx.float32,
    )
    target_logits = mx.array(
        [
            [[0.0, 8.0, -4.0], [-4.0, 0.0, 8.0], [8.0, 0.0, -4.0]],
            [[-4.0, 8.0, 0.0], [0.0, 8.0, -4.0], [0.0, -4.0, 8.0]],
        ],
        dtype=mx.float32,
    )
    draft_tokens = mx.array([[1, 2], [0, 1]], dtype=mx.int32)
    rng_key = mx.random.key(41)

    accepted, n_accepted, next_tokens = speculative_acceptance_batch(
        draft_logits,
        target_logits,
        draft_tokens,
        temperature=0.75,
        rng_key=rng_key,
    )

    expected_accepted: list[list[int]] = []
    expected_n_accepted: list[int] = []
    expected_next_tokens: list[int] = []
    for row, row_key in enumerate(mx.random.split(rng_key, 2)):
        row_accepted, row_n_accepted, row_next_token = speculative_acceptance(
            draft_logits[row],
            target_logits[row],
            draft_tokens[row],
            temperature=0.75,
            rng_key=row_key,
        )
        padded = [-1, -1]
        for i in range(row_n_accepted):
            padded[i] = int(row_accepted[i].item())
        expected_accepted.append(padded)
        expected_n_accepted.append(row_n_accepted)
        expected_next_tokens.append(int(row_next_token[0].item()))

    np.testing.assert_array_equal(_as_numpy(accepted), np.array(expected_accepted, dtype=np.int32))
    np.testing.assert_array_equal(_as_numpy(n_accepted), np.array(expected_n_accepted, dtype=np.int32))
    np.testing.assert_array_equal(_as_numpy(next_tokens), np.array(expected_next_tokens, dtype=np.int32))


def test_batched_speculative_acceptance_pads_accepted_prefixes() -> None:
    draft_logits = mx.array(
        [
            [[0.0, 8.0, -4.0], [-3.0, 0.0, 8.0]],
            [[10.0, 0.0, -10.0], [10.0, -10.0, -10.0]],
        ],
        dtype=mx.float32,
    )
    target_logits = mx.array(
        [
            [[0.0, 9.0, -4.0], [-3.0, 0.0, 9.0], [10.0, 0.0, -2.0]],
            [[-10.0, 10.0, -10.0], [0.0, 10.0, -10.0], [0.0, 0.0, 10.0]],
        ],
        dtype=mx.float32,
    )
    draft_tokens = mx.array([[1, 2], [0, 0]], dtype=mx.int32)

    accepted, n_accepted, next_tokens = speculative_acceptance_batch(
        draft_logits,
        target_logits,
        draft_tokens,
        rng_key=mx.random.key(29),
    )

    np.testing.assert_array_equal(
        _as_numpy(accepted),
        np.array([[1, 2], [-1, -1]], dtype=np.int32),
    )
    np.testing.assert_array_equal(_as_numpy(n_accepted), np.array([2, 0], dtype=np.int32))
    np.testing.assert_array_equal(_as_numpy(next_tokens), np.array([0, 1], dtype=np.int32))


def test_batched_typical_acceptance_matches_rowwise_split_keys() -> None:
    draft_logits = mx.array(
        [
            [[0.0, 8.0], [8.0, 0.0]],
            [[8.0, 0.0], [0.0, 8.0]],
        ],
        dtype=mx.float32,
    )
    target_logits = mx.array(
        [
            [[0.0, 9.0], [9.0, 0.0], [0.0, 10.0]],
            [[0.0, 9.0], [0.0, 9.0], [10.0, 0.0]],
        ],
        dtype=mx.float32,
    )
    draft_tokens = mx.array([[1, 0], [0, 1]], dtype=mx.int32)
    rng_key = mx.random.key(43)

    accepted, n_accepted, next_tokens = typical_acceptance_batch(
        draft_logits,
        target_logits,
        draft_tokens,
        threshold=0.5,
        rng_key=rng_key,
    )

    expected_accepted: list[list[int]] = []
    expected_n_accepted: list[int] = []
    expected_next_tokens: list[int] = []
    for row, row_key in enumerate(mx.random.split(rng_key, 2)):
        row_accepted, row_n_accepted, row_next_token = typical_acceptance(
            draft_logits[row],
            target_logits[row],
            draft_tokens[row],
            threshold=0.5,
            rng_key=row_key,
        )
        padded = [-1, -1]
        for i in range(row_n_accepted):
            padded[i] = int(row_accepted[i].item())
        expected_accepted.append(padded)
        expected_n_accepted.append(row_n_accepted)
        expected_next_tokens.append(int(row_next_token[0].item()))

    np.testing.assert_array_equal(_as_numpy(accepted), np.array(expected_accepted, dtype=np.int32))
    np.testing.assert_array_equal(_as_numpy(n_accepted), np.array(expected_n_accepted, dtype=np.int32))
    np.testing.assert_array_equal(_as_numpy(next_tokens), np.array(expected_next_tokens, dtype=np.int32))


def test_batched_typical_acceptance_pads_accepted_prefixes() -> None:
    draft_logits = mx.array(
        [
            [[0.0, 8.0], [8.0, 0.0]],
            [[8.0, 0.0], [0.0, 8.0]],
        ],
        dtype=mx.float32,
    )
    target_logits = mx.array(
        [
            [[0.0, 9.0], [9.0, 0.0], [0.0, 10.0]],
            [[0.0, 9.0], [0.0, 9.0], [10.0, 0.0]],
        ],
        dtype=mx.float32,
    )
    draft_tokens = mx.array([[1, 0], [0, 1]], dtype=mx.int32)

    accepted, n_accepted, next_tokens = typical_acceptance_batch(
        draft_logits,
        target_logits,
        draft_tokens,
        threshold=0.5,
        rng_key=mx.random.key(31),
    )

    np.testing.assert_array_equal(
        _as_numpy(accepted),
        np.array([[1, 0], [-1, -1]], dtype=np.int32),
    )
    np.testing.assert_array_equal(_as_numpy(n_accepted), np.array([2, 0], dtype=np.int32))
    np.testing.assert_array_equal(_as_numpy(next_tokens), np.array([1, 1], dtype=np.int32))


def test_speculative_acceptance_rejects_invalid_shapes_and_temperature() -> None:
    draft_logits = mx.array([[0.0, 1.0]], dtype=mx.float32)
    target_logits = mx.array([[0.0, 1.0], [1.0, 0.0]], dtype=mx.float32)
    draft_tokens = mx.array([1], dtype=mx.int32)

    with pytest.raises(ValueError, match="temperature"):
        speculative_acceptance(draft_logits, target_logits, draft_tokens, temperature=-1.0)
    with pytest.raises(ValueError, match="draft_tokens"):
        speculative_acceptance(draft_logits, target_logits, draft_tokens[:, None])
    with pytest.raises(ValueError, match="target_logits"):
        speculative_acceptance(draft_logits, target_logits[:1], draft_tokens)
    with pytest.raises(ValueError, match="vocab"):
        speculative_acceptance(draft_logits, target_logits[:, :1], draft_tokens)


def test_typical_acceptance_rejects_negative_threshold() -> None:
    draft_logits = mx.array([[0.0, 1.0]], dtype=mx.float32)
    target_logits = mx.array([[0.0, 1.0], [1.0, 0.0]], dtype=mx.float32)
    draft_tokens = mx.array([1], dtype=mx.int32)

    with pytest.raises(ValueError, match="threshold"):
        typical_acceptance(draft_logits, target_logits, draft_tokens, threshold=-0.1)


def test_inference_root_exports_speculative_decode_helpers() -> None:
    assert inference.speculative_acceptance is speculative_acceptance
    assert inference.speculative_acceptance_batch is speculative_acceptance_batch
    assert inference.typical_acceptance is typical_acceptance
    assert inference.typical_acceptance_batch is typical_acceptance_batch
    assert {
        "speculative_acceptance",
        "speculative_acceptance_batch",
        "typical_acceptance",
        "typical_acceptance_batch",
    } <= set(inference.__all__)

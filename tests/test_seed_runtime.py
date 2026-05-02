from __future__ import annotations

import json
import random

import mlx.core as mx
import numpy as np

from cppmega_mlx.runtime.seed import (
    capture_rng_state,
    mlx_rng_state_available,
    restore_rng_state,
    seed_all,
)


def test_rng_snapshot_roundtrips_through_json() -> None:
    seed_all(123)
    snapshot = capture_rng_state()
    encoded = json.dumps(snapshot, sort_keys=True)
    decoded = json.loads(encoded)

    result = restore_rng_state(decoded)

    assert result["python_random"] == "restored"
    assert result["numpy_random"] == "restored"
    assert snapshot["scope"] == "single_process_local"
    assert decoded["mlx_random"]["available"] in {True, False}


def test_python_and_numpy_rng_restore_determinism() -> None:
    seed_all(991)
    snapshot = capture_rng_state()
    expected_python = [random.random() for _ in range(4)]
    expected_numpy = np.random.random(4)

    _ = [random.random() for _ in range(8)]
    _ = np.random.random(8)
    restore_rng_state(snapshot)

    assert [random.random() for _ in range(4)] == expected_python
    np.testing.assert_allclose(np.random.random(4), expected_numpy, rtol=0, atol=0)


def test_mlx_rng_restore_determinism_or_reports_unavailable() -> None:
    seed_all(771)
    snapshot = capture_rng_state()
    if not mlx_rng_state_available():
        assert snapshot["mlx_random"]["available"] is False
        assert snapshot["mlx_random"]["reason"]
        result = restore_rng_state(snapshot)
        assert result["mlx_random"]["restored"] is False
        return

    expected = mx.random.uniform(shape=(4,))
    mx.eval(expected)
    _ = mx.random.uniform(shape=(8,))
    restore_result = restore_rng_state(snapshot)
    actual = mx.random.uniform(shape=(4,))
    mx.eval(actual)

    assert restore_result["mlx_random"] == {"restored": True}
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=0, atol=0)

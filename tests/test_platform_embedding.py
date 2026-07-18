from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx

from cppmega_mlx.data.platform_context import PLATFORM_VOCAB_SIZE
from cppmega_mlx.nn.platform_embedding import PlatformEmbedding


def _to_numpy(value: mx.array) -> np.ndarray:
    mx.eval(value)
    return np.asarray(value)


def test_platform_embedding_is_zero_init_and_padding_aware() -> None:
    module = PlatformEmbedding(hidden_size=6, vocab_size=PLATFORM_VOCAB_SIZE)
    platform_ids = mx.array([[2, 64, 0], [0, 0, 0]], dtype=mx.int32)

    out = module(platform_ids, target_dtype=mx.float32)

    assert out.shape == (2, 1, 6)
    assert np.count_nonzero(_to_numpy(out)) == 0

    module.embedding.weight = mx.ones_like(module.embedding.weight)
    enabled = module(platform_ids, target_dtype=mx.float32)

    np.testing.assert_allclose(_to_numpy(enabled)[0, 0], np.full((6,), 2.0))
    np.testing.assert_allclose(_to_numpy(enabled)[1, 0], np.zeros((6,)))


def test_platform_embedding_supports_token_local_platform_ids() -> None:
    module = PlatformEmbedding(hidden_size=3, vocab_size=PLATFORM_VOCAB_SIZE)
    platform_ids = mx.array(
        [[[2, 64, 0], [3, 64, 94], [0, 0, 0]]],
        dtype=mx.int32,
    )

    out = module(platform_ids, target_dtype=mx.float32)

    assert out.shape == (1, 3, 3)
    assert np.count_nonzero(_to_numpy(out)) == 0

    module.embedding.weight = mx.ones_like(module.embedding.weight)
    enabled = module(platform_ids, target_dtype=mx.float32)

    np.testing.assert_allclose(_to_numpy(enabled)[0, 0], np.full((3,), 2.0))
    np.testing.assert_allclose(_to_numpy(enabled)[0, 1], np.full((3,), 3.0))
    np.testing.assert_allclose(_to_numpy(enabled)[0, 2], np.zeros((3,)))


def test_platform_embedding_validates_platform_ids_shape_and_range() -> None:
    module = PlatformEmbedding(hidden_size=4, vocab_size=PLATFORM_VOCAB_SIZE)

    with pytest.raises(ValueError, match="platform_ids must be shaped"):
        module(mx.array([1, 2, 3], dtype=mx.int32))

    with pytest.raises(ValueError, match="non-negative"):
        module(mx.array([[1, -2]], dtype=mx.int32))


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (0.5, "fractional"),
        (float("nan"), "finite"),
        (float("inf"), "finite"),
    ],
)
def test_platform_embedding_rejects_invalid_integer_channels_before_cast(
    value: float,
    error: str,
) -> None:
    module = PlatformEmbedding(hidden_size=4, vocab_size=PLATFORM_VOCAB_SIZE)

    with pytest.raises(ValueError, match=error):
        module(mx.array([[value]], dtype=mx.float32))


def test_platform_embedding_rejects_uint64_overflow_before_cast() -> None:
    module = PlatformEmbedding(hidden_size=4, vocab_size=PLATFORM_VOCAB_SIZE)

    with pytest.raises(ValueError, match="less than vocab_size"):
        module(mx.array([[2**64 - 1]], dtype=mx.uint64))

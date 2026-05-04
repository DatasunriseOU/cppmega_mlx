from __future__ import annotations

from typing import Any, cast

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import KVCache, QuantizedKVCache

import cppmega_mlx.inference as inference
from cppmega_mlx.inference import (
    ContiguousKVCache,
    ContiguousKVCacheConfig,
    kv_cache_position,
    make_contiguous_kv_cache,
    prefill_contiguous_kv_cache,
    rollback_contiguous_kv_cache,
    trim_contiguous_kv_cache,
)


def _array_pair(
    value: tuple[mx.array, mx.array] | tuple[tuple[mx.array, ...], tuple[mx.array, ...]],
) -> tuple[mx.array, mx.array]:
    first, second = value
    assert isinstance(first, mx.array)
    assert isinstance(second, mx.array)
    return first, second


def _as_numpy(array: mx.array) -> np.ndarray:
    mx.eval(array)
    return np.array(array)


def _keys(
    sequence: int,
    *,
    batch: int = 1,
    heads: int = 2,
    head_dim: int = 32,
    start: int = 0,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    total = batch * heads * sequence * head_dim
    return (mx.arange(start, start + total, dtype=dtype) / 100.0).reshape(
        batch,
        heads,
        sequence,
        head_dim,
    )


def test_make_contiguous_kv_cache_starts_empty() -> None:
    cache = make_contiguous_kv_cache(
        num_layers=2,
        batch_size=1,
        num_kv_heads=2,
        head_dim=32,
        max_seq_len=8,
        dtype=mx.float32,
    )

    assert isinstance(cache, ContiguousKVCache)
    assert cache.config == ContiguousKVCacheConfig(
        num_layers=2,
        batch_size=1,
        num_kv_heads=2,
        head_dim=32,
        max_seq_len=8,
        dtype=mx.float32,
    )
    assert len(cache.layers) == 2
    assert all(isinstance(layer, KVCache) for layer in cache.layers)
    assert kv_cache_position(cache) == 0


def test_contiguous_kv_update_and_fetch_shapes_and_position() -> None:
    cache = make_contiguous_kv_cache(
        num_layers=1,
        batch_size=1,
        num_kv_heads=2,
        head_dim=32,
        max_seq_len=8,
        dtype=mx.float32,
    )
    keys = _keys(3)
    values = keys + 1.0

    cached_keys, cached_values = _array_pair(cache.update_and_fetch(0, keys, values))

    assert cache.position() == 3
    assert cached_keys.shape == (1, 2, 3, 32)
    assert cached_values.shape == (1, 2, 3, 32)
    np.testing.assert_allclose(_as_numpy(cached_keys), _as_numpy(keys))
    np.testing.assert_allclose(_as_numpy(cached_values), _as_numpy(values))


def test_contiguous_kv_appends_then_rolls_back_like_track_a() -> None:
    cache = make_contiguous_kv_cache(
        num_layers=1,
        batch_size=1,
        num_kv_heads=2,
        head_dim=32,
        max_seq_len=8,
    )
    first = _keys(3)
    second = _keys(2, start=first.size)
    cache.update_and_fetch(0, first, first + 1.0)
    cached_keys, _ = _array_pair(cache.update_and_fetch(0, second, second + 1.0))

    assert cache.position() == 5
    assert cached_keys.shape == (1, 2, 5, 32)
    assert rollback_contiguous_kv_cache(cache, 2) == 2
    assert cache.position() == 3

    replacement = _keys(1, start=10_000)
    cached_keys, _ = _array_pair(
        cache.update_and_fetch(0, replacement, replacement + 1.0)
    )

    assert cache.position() == 4
    np.testing.assert_allclose(
        _as_numpy(cached_keys[:, :, 3:4, :]),
        _as_numpy(replacement),
    )


def test_trim_contiguous_kv_cache_trims_all_layers() -> None:
    cache = make_contiguous_kv_cache(
        num_layers=2,
        batch_size=1,
        num_kv_heads=2,
        head_dim=32,
    )
    keys = _keys(3)
    cache.update_and_fetch(0, keys, keys + 1.0)
    cache.update_and_fetch(1, keys + 2.0, keys + 3.0)

    assert trim_contiguous_kv_cache(cache, 10) == 3
    assert cache.position() == 0


def test_prefill_contiguous_kv_cache_copies_single_batch_into_larger_batch() -> None:
    source = make_contiguous_kv_cache(
        num_layers=1,
        batch_size=1,
        num_kv_heads=2,
        head_dim=32,
        max_seq_len=8,
        dtype=mx.float32,
    )
    destination = make_contiguous_kv_cache(
        num_layers=1,
        batch_size=3,
        num_kv_heads=2,
        head_dim=32,
        max_seq_len=8,
        dtype=mx.float32,
    )
    keys = _keys(4)
    source.update_and_fetch(0, keys, keys + 1.0)

    prefill_contiguous_kv_cache(destination, source)

    assert destination.position() == 4
    state = destination.layers[0].state
    assert state is not None
    dst_keys, dst_values = state
    assert isinstance(dst_keys, mx.array)
    assert isinstance(dst_values, mx.array)
    assert dst_keys.shape == (3, 2, 4, 32)
    assert dst_values.shape == (3, 2, 4, 32)
    np.testing.assert_allclose(_as_numpy(dst_keys[0:1]), _as_numpy(keys))
    np.testing.assert_allclose(_as_numpy(dst_keys[1:2]), _as_numpy(keys))
    np.testing.assert_allclose(_as_numpy(dst_values[2:3]), _as_numpy(keys + 1.0))

    replacement = _keys(1, batch=3, start=20_000)
    destination.update_and_fetch(0, replacement, replacement + 1.0)
    assert source.position() == 4


def test_prefill_rejects_non_empty_destination_and_incompatible_shape() -> None:
    source = make_contiguous_kv_cache(
        num_layers=1,
        batch_size=1,
        num_kv_heads=2,
        head_dim=32,
    )
    destination = make_contiguous_kv_cache(
        num_layers=1,
        batch_size=1,
        num_kv_heads=2,
        head_dim=32,
    )
    other_heads = make_contiguous_kv_cache(
        num_layers=1,
        batch_size=1,
        num_kv_heads=1,
        head_dim=32,
    )
    keys = _keys(1)
    source.update_and_fetch(0, keys, keys)
    destination.update_and_fetch(0, keys, keys)

    with pytest.raises(RuntimeError, match="non-empty"):
        prefill_contiguous_kv_cache(destination, source)
    with pytest.raises(ValueError, match="num_kv_heads"):
        prefill_contiguous_kv_cache(other_heads, source)


def test_quantized_contiguous_kv_cache_uses_mlx_lm_quantized_layers() -> None:
    cache = make_contiguous_kv_cache(
        num_layers=1,
        batch_size=1,
        num_kv_heads=2,
        head_dim=32,
        max_seq_len=8,
        quantized=True,
        kv_group_size=32,
        kv_bits=4,
    )
    keys = _keys(3, head_dim=32)

    packed_keys, packed_values = cache.update_and_fetch(0, keys, keys + 1.0)
    mx.eval(*(tuple(packed_keys) + tuple(packed_values)))

    assert isinstance(cache.layers[0], QuantizedKVCache)
    assert cache.position() == 3
    assert [part.shape for part in packed_keys] == [(1, 2, 3, 4), (1, 2, 3, 1), (1, 2, 3, 1)]
    assert [part.shape for part in packed_values] == [
        (1, 2, 3, 4),
        (1, 2, 3, 1),
        (1, 2, 3, 1),
    ]


def test_quantized_prefill_preserves_quantized_meta_state() -> None:
    source = make_contiguous_kv_cache(
        num_layers=1,
        batch_size=1,
        num_kv_heads=2,
        head_dim=32,
        quantized=True,
        kv_group_size=32,
        kv_bits=4,
    )
    destination = make_contiguous_kv_cache(
        num_layers=1,
        batch_size=1,
        num_kv_heads=2,
        head_dim=32,
        quantized=True,
        kv_group_size=32,
        kv_bits=4,
    )
    source.update_and_fetch(0, _keys(2), _keys(2, start=1000))

    destination.prefill_from(source)

    assert destination.position() == 2
    assert isinstance(destination.layers[0], QuantizedKVCache)
    assert destination.layers[0].meta_state == source.layers[0].meta_state


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"num_layers": 0}, "num_layers"),
        ({"batch_size": 0}, "batch_size"),
        ({"num_kv_heads": 0}, "num_kv_heads"),
        ({"head_dim": 0}, "head_dim"),
        ({"max_seq_len": 0}, "max_seq_len"),
        ({"quantized": True, "kv_group_size": 16}, "kv_group_size"),
        ({"quantized": True, "kv_bits": 3}, "kv_bits"),
        ({"quantized": True, "head_dim": 48, "kv_group_size": 32}, "head_dim"),
    ],
)
def test_contiguous_kv_cache_config_fails_closed(
    kwargs: dict[str, int | bool],
    match: str,
) -> None:
    base: dict[str, object] = {
        "num_layers": 1,
        "batch_size": 1,
        "num_kv_heads": 2,
        "head_dim": 32,
    }
    base.update(kwargs)

    with pytest.raises(ValueError, match=match):
        ContiguousKVCacheConfig(**cast(Any, base))


def test_make_contiguous_kv_cache_rejects_mixed_config_and_kwargs() -> None:
    config = ContiguousKVCacheConfig(
        num_layers=1,
        batch_size=1,
        num_kv_heads=2,
        head_dim=32,
    )

    with pytest.raises(ValueError, match="either config or shape kwargs"):
        make_contiguous_kv_cache(config, num_layers=1)
    with pytest.raises(ValueError, match="required"):
        make_contiguous_kv_cache(num_layers=1, batch_size=1)


def test_contiguous_kv_update_fails_closed_on_bad_inputs() -> None:
    cache = make_contiguous_kv_cache(
        num_layers=1,
        batch_size=1,
        num_kv_heads=2,
        head_dim=32,
        max_seq_len=2,
        dtype=mx.float32,
    )
    keys = _keys(1)

    with pytest.raises(IndexError, match="layer_idx"):
        cache.update_and_fetch(1, keys, keys)
    with pytest.raises(TypeError, match="mlx.core.array"):
        cache.update_and_fetch(0, object(), keys)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape"):
        cache.update_and_fetch(0, mx.ones((1, 2, 32)), keys)
    with pytest.raises(ValueError, match="matching shapes"):
        cache.update_and_fetch(0, keys, mx.ones((1, 2, 2, 32)))
    with pytest.raises(ValueError, match="matching dtype"):
        cache.update_and_fetch(0, keys, keys.astype(mx.float16))
    with pytest.raises(ValueError, match="batch_size"):
        cache.update_and_fetch(0, _keys(1, batch=2), _keys(1, batch=2))
    with pytest.raises(ValueError, match="num_kv_heads"):
        cache.update_and_fetch(0, _keys(1, heads=1), _keys(1, heads=1))
    with pytest.raises(ValueError, match="head_dim"):
        cache.update_and_fetch(0, _keys(1, head_dim=16), _keys(1, head_dim=16))

    cache.update_and_fetch(0, _keys(2), _keys(2))
    with pytest.raises(RuntimeError, match="max_seq_len"):
        cache.update_and_fetch(0, _keys(1), _keys(1))


def test_contiguous_kv_position_and_rollback_fail_closed() -> None:
    cache = make_contiguous_kv_cache(
        num_layers=2,
        batch_size=1,
        num_kv_heads=2,
        head_dim=32,
    )
    keys = _keys(1)
    cache.update_and_fetch(0, keys, keys)

    with pytest.raises(RuntimeError, match="offsets are not aligned"):
        kv_cache_position(cache)
    with pytest.raises(ValueError, match="non-negative"):
        trim_contiguous_kv_cache(cache, -1)
    with pytest.raises(RuntimeError, match="below position 0"):
        rollback_contiguous_kv_cache(
            make_contiguous_kv_cache(
                num_layers=1,
                batch_size=1,
                num_kv_heads=2,
                head_dim=32,
            ),
            1,
        )


def test_inference_root_exports_contiguous_kv_helpers() -> None:
    assert inference.ContiguousKVCache is ContiguousKVCache
    assert inference.ContiguousKVCacheConfig is ContiguousKVCacheConfig
    assert inference.kv_cache_position is kv_cache_position
    assert inference.make_contiguous_kv_cache is make_contiguous_kv_cache
    assert inference.prefill_contiguous_kv_cache is prefill_contiguous_kv_cache
    assert inference.rollback_contiguous_kv_cache is rollback_contiguous_kv_cache
    assert inference.trim_contiguous_kv_cache is trim_contiguous_kv_cache
    assert {
        "ContiguousKVCache",
        "ContiguousKVCacheConfig",
        "kv_cache_position",
        "make_contiguous_kv_cache",
        "prefill_contiguous_kv_cache",
        "rollback_contiguous_kv_cache",
        "trim_contiguous_kv_cache",
    } <= set(inference.__all__)

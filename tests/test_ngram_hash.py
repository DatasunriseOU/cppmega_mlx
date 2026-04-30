import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn

from cppmega_mlx.nn.ngram_hash import CppMegaNgramHashEmbedding, pick_primes


def to_numpy(x: mx.array) -> np.ndarray:
    mx.eval(x)
    return np.asarray(x)


def test_pick_primes_matches_cppmega_fallback_and_near_target_behavior():
    assert pick_primes(3, 500_000) == (499_801, 499_819, 499_853)
    assert pick_primes(3, 257) == (257, 258, 259)


def test_ngram_hash_embedding_returns_hidden_sized_tensor_and_zero_init_projection():
    module = CppMegaNgramHashEmbedding(
        hidden_size=32,
        orders=(2, 3),
        num_heads=2,
        table_size=257,
        embed_dim=8,
        seed=7,
    )
    token_ids = mx.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=mx.int64)

    out = module(token_ids)

    assert out.shape == (2, 4, 32)
    assert np.count_nonzero(to_numpy(out)) == 0


def test_ngram_hash_indices_stay_inside_unified_table_offsets():
    module = CppMegaNgramHashEmbedding(
        hidden_size=16,
        orders=(1, 3),
        num_heads=3,
        table_size=257,
        embed_dim=4,
        seed=11,
    )
    token_ids = mx.array([[1, 5, 7, 9], [2, 4, 6, 8]], dtype=mx.int64)

    indices = module._hash_all(token_ids)
    indices_np = to_numpy(indices)
    offsets = to_numpy(module.table_offsets)
    sizes = to_numpy(module.table_sizes_t)

    assert indices.shape == (2, 6, 4)
    for table in range(module.num_tables):
        table_indices = indices_np[:, table, :]
        assert table_indices.min() >= offsets[table]
        assert table_indices.max() < offsets[table] + sizes[table]


def test_ngram_hash_has_stable_indices_when_seeded():
    first = CppMegaNgramHashEmbedding(
        hidden_size=8,
        orders=(2,),
        num_heads=2,
        table_size=257,
        embed_dim=4,
        seed=123,
    )
    second = CppMegaNgramHashEmbedding(
        hidden_size=8,
        orders=(2,),
        num_heads=2,
        table_size=257,
        embed_dim=4,
        seed=123,
    )
    token_ids = mx.array([[1, 2, 3]], dtype=mx.int64)

    assert np.array_equal(to_numpy(first._hash_all(token_ids)), to_numpy(second._hash_all(token_ids)))


def test_ngram_hash_gradients_reach_unified_table_after_projection_is_enabled():
    module = CppMegaNgramHashEmbedding(
        hidden_size=16,
        orders=(2,),
        num_heads=2,
        table_size=257,
        embed_dim=4,
        seed=3,
    )
    module.out_proj.weight = mx.ones_like(module.out_proj.weight)
    token_ids = mx.array([[1, 5, 7, 9]], dtype=mx.int64)

    def loss_fn():
        return mx.sum(module(token_ids))

    loss, grads = nn.value_and_grad(module, loss_fn)()
    mx.eval(loss, grads)

    assert "unified_table" in grads
    grad = grads["unified_table"]["weight"]
    assert grad.shape == module.unified_table.weight.shape
    assert bool(mx.any(mx.isfinite(grad)).item())
    assert float(mx.sum(mx.abs(grad)).item()) > 0.0


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"orders": ()}, "orders"),
        ({"orders": (0,)}, "positive"),
        ({"num_heads": 0}, "num_heads"),
        ({"table_size": 0}, "table_size"),
        ({"embed_dim": 0}, "embed_dim"),
    ],
)
def test_ngram_hash_validates_constructor_args(kwargs, error):
    with pytest.raises(ValueError, match=error):
        CppMegaNgramHashEmbedding(hidden_size=8, **kwargs)


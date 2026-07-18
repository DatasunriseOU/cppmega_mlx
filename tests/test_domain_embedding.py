from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn

from cppmega_mlx.nn.domain_embedding import CppMegaDomainEmbedding


def _to_numpy(value: mx.array) -> np.ndarray:
    mx.eval(value)
    return np.asarray(value)


def test_domain_embedding_is_zero_init_with_live_table_gradient() -> None:
    module = CppMegaDomainEmbedding(hidden_size=8, bottleneck_dim=4)
    domain_ids = mx.array([[1, 2]], dtype=mx.int32)
    role_ids = mx.array([[1, 6]], dtype=mx.int32)
    confidence_ids = mx.array([[4, 4]], dtype=mx.int32)

    def loss_fn(model: CppMegaDomainEmbedding) -> mx.array:
        return mx.sum(
            model(
                domain_ids=domain_ids,
                role_ids=role_ids,
                confidence_ids=confidence_ids,
            )
        )

    out = module(
        domain_ids=domain_ids,
        role_ids=role_ids,
        confidence_ids=confidence_ids,
        target_dtype=mx.float16,
    )
    loss, grads = nn.value_and_grad(module, loss_fn)(module)
    mx.eval(out, loss, grads)

    assert out.shape == (1, 2, 8)
    assert out.dtype == mx.float16
    assert np.count_nonzero(_to_numpy(out)) == 0
    assert float(mx.sum(mx.abs(grads["stacked_emb"]["weight"])).item()) > 0.0


def test_domain_embedding_accepts_missing_optional_components_but_not_all_absent() -> None:
    module = CppMegaDomainEmbedding(hidden_size=8, bottleneck_dim=4)
    out = module(
        domain_ids=mx.array([[1, 2]], dtype=mx.int32),
        role_ids=None,
        confidence_ids=None,
    )
    assert out.shape == (1, 2, 8)

    with pytest.raises(ValueError, match="all domain sidecars are absent"):
        module(domain_ids=None, role_ids=None, confidence_ids=None)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("domain_ids", 64, "domain_ids out of range"),
        ("role_ids", -1, "role_ids out of range"),
        ("confidence_ids", 8, "confidence_ids out of range"),
    ],
)
def test_domain_embedding_rejects_invalid_ids(
    field: str,
    value: int,
    message: str,
) -> None:
    module = CppMegaDomainEmbedding(hidden_size=8, bottleneck_dim=4)
    kwargs = {
        "domain_ids": mx.ones((1, 2), dtype=mx.int32),
        "role_ids": mx.ones((1, 2), dtype=mx.int32),
        "confidence_ids": mx.ones((1, 2), dtype=mx.int32),
    }
    kwargs[field] = mx.array([[value, 1]], dtype=mx.int32)

    with pytest.raises(ValueError, match=message):
        module(**kwargs)


def test_domain_embedding_rejects_mismatched_shapes() -> None:
    module = CppMegaDomainEmbedding(hidden_size=8, bottleneck_dim=4)
    with pytest.raises(ValueError, match="role_ids shape"):
        module(
            domain_ids=mx.ones((1, 2), dtype=mx.int32),
            role_ids=mx.ones((1, 3), dtype=mx.int32),
            confidence_ids=None,
        )

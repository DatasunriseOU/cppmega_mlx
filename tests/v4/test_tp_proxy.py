"""V7-B02: tensor-parallel column/row split proxy correctness."""

from __future__ import annotations

import mlx.core as mx
import pytest

from cppmega_v4.runtime.tp_proxy import (
    column_split_forward, row_split_forward,
)


def test_v7_b02_column_split_equivalent_to_full_matmul():
    W = mx.random.normal(shape=(64, 128), key=mx.random.key(0))
    x = mx.random.normal(shape=(4, 64), key=mx.random.key(1))
    full = mx.matmul(x, W)
    for tp in (1, 2, 4, 8):
        if W.shape[1] % tp != 0:
            continue
        split = column_split_forward(W, x, tp)
        assert split.shape == full.shape
        assert mx.allclose(full, split, atol=1e-4)


def test_v7_b02_row_split_equivalent_to_full_matmul():
    W = mx.random.normal(shape=(64, 32), key=mx.random.key(2))
    x = mx.random.normal(shape=(4, 64), key=mx.random.key(3))
    full = mx.matmul(x, W)
    for tp in (1, 2, 4, 8):
        if W.shape[0] % tp != 0:
            continue
        split = row_split_forward(W, x, tp)
        assert split.shape == full.shape
        assert mx.allclose(full, split, atol=1e-3)


def test_v7_b02_column_split_validates_divisibility():
    W = mx.zeros((4, 7))
    with pytest.raises(ValueError):
        column_split_forward(W, mx.zeros((1, 4)), 2)


def test_v7_b02_row_split_validates_divisibility_and_dim():
    W = mx.zeros((7, 4))
    with pytest.raises(ValueError):
        row_split_forward(W, mx.zeros((1, 7)), 2)
    with pytest.raises(ValueError):
        row_split_forward(mx.zeros((4, 4)), mx.zeros((1, 8)), 2)

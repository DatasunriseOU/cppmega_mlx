"""V7-B12: tp_proxy column/row split forward parity vs no-TP."""

from __future__ import annotations

import mlx.core as mx

from cppmega_v4.runtime.tp_proxy import (
    column_split_forward, row_split_forward,
)


def test_column_split_forward_matches_unsharded_matmul():
    mx.random.seed(0)
    in_dim, out_dim = 8, 16
    W = mx.random.normal(shape=(in_dim, out_dim), key=mx.random.key(0))
    x = mx.random.normal(shape=(2, in_dim), key=mx.random.key(1))

    y_ref = mx.matmul(x, W)
    for tp in (1, 2, 4, 8):
        y = column_split_forward(W, x, tp)
        assert mx.allclose(y, y_ref, atol=1e-6).item(), (
            f"tp={tp} column-split drifted from baseline")


def test_row_split_forward_matches_unsharded_matmul():
    mx.random.seed(1)
    in_dim, out_dim = 16, 8
    W = mx.random.normal(shape=(in_dim, out_dim), key=mx.random.key(2))
    x = mx.random.normal(shape=(2, in_dim), key=mx.random.key(3))

    y_ref = mx.matmul(x, W)
    for tp in (1, 2, 4, 8):
        y = row_split_forward(W, x, tp)
        assert mx.allclose(y, y_ref, atol=1e-6).item(), (
            f"tp={tp} row-split drifted from baseline")


def test_two_brick_chain_loss_parity():
    """ColumnParallel(W1) → RowParallel(W2) chain produces the same
    forward output as the unsharded W1@W2 matmul."""
    mx.random.seed(7)
    in_dim, mid, out_dim = 8, 16, 8
    W1 = mx.random.normal(shape=(in_dim, mid), key=mx.random.key(4))
    W2 = mx.random.normal(shape=(mid, out_dim), key=mx.random.key(5))
    x = mx.random.normal(shape=(2, in_dim), key=mx.random.key(6))
    y_ref = mx.matmul(mx.matmul(x, W1), W2)

    for tp in (1, 2, 4):
        h = column_split_forward(W1, x, tp)
        y = row_split_forward(W2, h, tp)
        assert mx.allclose(y, y_ref, atol=1e-5).item(), (
            f"tp={tp} chain drifted from baseline")

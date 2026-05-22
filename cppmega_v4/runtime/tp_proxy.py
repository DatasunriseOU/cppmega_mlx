"""V7-B02: tensor-parallel proxy — column/row split Linear.

Single-device emulation of TP-rank-N: split a Linear weight along
output (column) or input (row) axis into N chunks, run forward
through each, gather/sum back. Mathematically equivalent to the
unsharded forward — proves the split mechanic is correct on a single
device. Real multi-GPU TP collectives land under V7-B01.
"""

from __future__ import annotations

import mlx.core as mx


def column_split_forward(W: mx.array, x: mx.array,
                          tp_size: int) -> mx.array:
    """ColumnParallelLinear: W shape (in, out). Split along axis=1.
    Each shard computes x @ W_shard → concat along last axis."""
    if tp_size < 1 or W.shape[1] % tp_size != 0:
        raise ValueError("tp_size must divide W.shape[1]")
    chunk = W.shape[1] // tp_size
    pieces = [
        mx.matmul(x, W[:, r * chunk:(r + 1) * chunk])
        for r in range(tp_size)
    ]
    return mx.concatenate(pieces, axis=-1)


def row_split_forward(W: mx.array, x: mx.array,
                       tp_size: int) -> mx.array:
    """RowParallelLinear: W shape (in, out). Split along axis=0.
    Each shard computes x_shard @ W_shard → sum across ranks."""
    if tp_size < 1 or W.shape[0] % tp_size != 0:
        raise ValueError("tp_size must divide W.shape[0]")
    if x.shape[-1] != W.shape[0]:
        raise ValueError("x last dim must match W in dim")
    chunk = W.shape[0] // tp_size
    pieces = [
        mx.matmul(x[..., r * chunk:(r + 1) * chunk],
                   W[r * chunk:(r + 1) * chunk, :])
        for r in range(tp_size)
    ]
    acc = pieces[0]
    for p in pieces[1:]:
        acc = acc + p
    return acc


__all__ = ["column_split_forward", "row_split_forward"]

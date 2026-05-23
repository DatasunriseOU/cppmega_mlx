"""V7-B02 (real): tensor-parallel Megatron-style Column/Row Linear.

Two API surfaces:

  * The single-device helpers (column_split_forward / row_split_forward)
    keep the legacy proxy contract — split W on one device, fan out and
    concat / sum back. These exist for unit tests.

  * The Megatron-style ColumnParallelLinear / RowParallelLinear classes
    take a `tp_group` (mx.distributed.Group or None) and route through
    cppmega_v4.runtime.distributed.all_gather / all_reduce. When the
    distributed runtime is initialised with world_size > 1, the forward
    actually shards across ranks. When world_size == 1 the same code
    path falls back to single-device equivalence — same math, no
    collectives.
"""

from __future__ import annotations


import mlx.core as mx
import mlx.nn as nn

from cppmega_v4.runtime import distributed as _d


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


class ColumnParallelLinear(nn.Module):
    """Megatron-style column-parallel Linear.

    Output dim is sharded across ranks. Each rank stores W[:, rank*out_per_rank:
    (rank+1)*out_per_rank] and computes its local x @ W_shard. The
    forward gathers the per-rank outputs along axis=-1 so every rank
    sees the full output.

    When world_size==1 this is a normal Linear with no collectives.
    """

    def __init__(self, in_features: int, out_features: int,
                 *, bias: bool = False,
                 gather_output: bool = True) -> None:
        super().__init__()
        w = _d.world()
        tp = max(1, w.world_size)
        if out_features % tp != 0:
            raise ValueError(
                f"out_features={out_features} not divisible by tp={tp}")
        self.in_features = in_features
        self.out_features = out_features
        self.tp = tp
        self.gather_output = gather_output
        out_per_rank = out_features // tp
        # Initialise with the proper rank-local shape so the
        # parameter count is honest about sharding.
        self.weight = mx.random.normal(
            shape=(in_features, out_per_rank),
            key=mx.random.key(0xC0 + w.rank)) * (in_features ** -0.5)
        self.bias = (mx.zeros((out_per_rank,))
                     if bias else None)

    def __call__(self, x: mx.array) -> mx.array:
        y = mx.matmul(x, self.weight)
        if self.bias is not None:
            y = y + self.bias
        if not self.gather_output:
            return y
        return _d.all_gather(y, axis=-1)


class RowParallelLinear(nn.Module):
    """Megatron-style row-parallel Linear.

    Input dim is sharded across ranks. Each rank stores W[rank*in_per_rank:
    (rank+1)*in_per_rank, :]; the input is expected to already be
    sharded along its last axis. After the local matmul the result is
    all-reduced (sum) across ranks to materialise the full output.
    """

    def __init__(self, in_features: int, out_features: int,
                 *, bias: bool = False,
                 input_is_parallel: bool = True) -> None:
        super().__init__()
        w = _d.world()
        tp = max(1, w.world_size)
        if in_features % tp != 0:
            raise ValueError(
                f"in_features={in_features} not divisible by tp={tp}")
        self.in_features = in_features
        self.out_features = out_features
        self.tp = tp
        self.input_is_parallel = input_is_parallel
        in_per_rank = in_features // tp
        self.weight = mx.random.normal(
            shape=(in_per_rank, out_features),
            key=mx.random.key(0xD0 + w.rank)) * (in_features ** -0.5)
        self.bias = (mx.zeros((out_features,))
                     if bias else None)

    def __call__(self, x: mx.array) -> mx.array:
        if not self.input_is_parallel:
            # Split input along last axis to match this rank's shard.
            in_per_rank = self.in_features // self.tp
            r = _d.world().rank
            x = x[..., r * in_per_rank:(r + 1) * in_per_rank]
        y = mx.matmul(x, self.weight)
        y = _d.all_reduce(y, op="sum")
        if self.bias is not None:
            y = y + self.bias
        return y


__all__ = ["column_split_forward", "row_split_forward",
            "ColumnParallelLinear", "RowParallelLinear"]

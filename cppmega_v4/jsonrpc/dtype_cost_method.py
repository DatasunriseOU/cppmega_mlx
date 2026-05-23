"""V7-D06: dtype.cost_estimate RPC.

Honest-closure: scripts/bench_dtype_cast_cost.py produced a static
CSV/HTML report of cast_overhead_ms + fwd_ms + fwdbwd_ms per dtype,
but the UI dtype dropdown showed no numbers and the user could not
see the real cost of fp32 vs bf16 vs fp16 before clicking Train.

This method runs an inline tiny-model probe (n_iter=5 by default,
B=1 S=8 H=128) so the UI can fetch fresh numbers on mount and render
ms/token alongside each option in the dropdown.
"""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

import mlx.core as mx
import mlx.nn as nn

from cppmega_v4.jsonrpc.cache import LRUCache

DtypeName = Literal["fp32", "bf16", "fp16"]


class DtypeCostParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_iter: int = Field(5, ge=1, le=200)
    hidden: int = Field(128, ge=16, le=4096)
    batch: int = Field(1, ge=1, le=32)
    seq: int = Field(8, ge=1, le=2048)


class DtypeCostRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    dtype: DtypeName
    supported: bool
    cast_overhead_ms: float | None = None
    fwd_ms: float | None = None
    fwdbwd_ms: float | None = None
    fwd_ms_per_token: float | None = None
    fwdbwd_ms_per_token: float | None = None
    error: str | None = None


class DtypeCostResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[DtypeCostRow] = Field(default_factory=list)
    n_iter: int
    shape: list[int]


def _build_tiny(hidden: int) -> nn.Module:
    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q = nn.Linear(hidden, hidden, bias=False)
            self.up = nn.Linear(hidden, 4 * hidden, bias=False)
            self.down = nn.Linear(4 * hidden, hidden, bias=False)

        def __call__(self, x: mx.array) -> mx.array:
            return self.down(nn.silu(self.up(self.q(x))))

    return _Block()


def _bench_one(dtype: DtypeName, *, n_iter: int, hidden: int,
               batch: int, seq: int) -> DtypeCostRow:
    dtype_map = {"fp32": mx.float32, "bf16": mx.bfloat16,
                 "fp16": mx.float16}
    if dtype not in dtype_map:
        return DtypeCostRow(dtype=dtype, supported=False,
                             error=f"unknown dtype {dtype}")
    target = dtype_map[dtype]
    try:
        model = _build_tiny(hidden)
        t0 = time.perf_counter()
        model.set_dtype(target)
        mx.eval(model.parameters())
        cast_ms = (time.perf_counter() - t0) * 1000.0

        x = mx.random.normal(shape=(batch, seq, hidden),
                              key=mx.random.key(0)).astype(target)
        # Warm.
        warm = model(x)
        mx.eval(warm)
        t0 = time.perf_counter()
        for _ in range(n_iter):
            out = model(x)
            mx.eval(out)
        fwd_ms = (time.perf_counter() - t0) * 1000.0 / n_iter

        def _loss(m, _x):
            return mx.mean(m(_x).astype(mx.float32) ** 2)

        lvg = nn.value_and_grad(model, _loss)
        warm_lvg = lvg(model, x)
        mx.eval(warm_lvg)
        t0 = time.perf_counter()
        for _ in range(n_iter):
            loss, grads = lvg(model, x)
            mx.eval(loss, grads)
        fwdbwd_ms = (time.perf_counter() - t0) * 1000.0 / n_iter

        tokens = batch * seq
        return DtypeCostRow(
            dtype=dtype, supported=True,
            cast_overhead_ms=round(cast_ms, 4),
            fwd_ms=round(fwd_ms, 4),
            fwdbwd_ms=round(fwdbwd_ms, 4),
            fwd_ms_per_token=round(fwd_ms / tokens, 6),
            fwdbwd_ms_per_token=round(fwdbwd_ms / tokens, 6),
        )
    except Exception as exc:
        return DtypeCostRow(dtype=dtype, supported=False, error=str(exc))


def dtype_cost_estimate(
    params: DtypeCostParams, *, cache: LRUCache | None = None,
) -> DtypeCostResult:
    dtype_names: tuple[DtypeName, ...] = ("fp32", "bf16", "fp16")
    rows = [
        _bench_one(dt, n_iter=params.n_iter, hidden=params.hidden,
                   batch=params.batch, seq=params.seq)
        for dt in dtype_names
    ]
    return DtypeCostResult(
        rows=rows, n_iter=params.n_iter,
        shape=[params.batch, params.seq, params.hidden],
    )


__all__ = ["DtypeCostParams", "DtypeCostResult", "DtypeCostRow",
            "dtype_cost_estimate"]

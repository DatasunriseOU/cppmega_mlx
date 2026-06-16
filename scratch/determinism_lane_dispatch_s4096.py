"""Fresh-process determinism for the in-regime LANE backward at s4096.

Exercises the ACTUAL wired prod path: mamba3_mimo_apply_with_state_path_c (the
forced PATH_C apply whose .vjp is the LANE backward mamba3_mimo_bwd_path_c, the
mode AUTO selects in-regime at s4096), at the prod shape B=1 H=112 P=64 N=64
fp32. Computes all-8-grad SHA-256. Each invocation is a FRESH process; the
orchestrator compares the digests for byte-identity across N runs. memguard 70.
"""

from __future__ import annotations

import hashlib
import sys

import numpy as np
import mlx.core as mx

from cppmega_mlx.runtime.memory import (
    apply_memory_limit_plan,
    device_total_memory_bytes,
    memory_limit_plan,
)
from cppmega_mlx.nn._tilelang.mamba3_path_c import (
    mamba3_mimo_apply_with_state_path_c,
)


def _apply_memguard():
    total = device_total_memory_bytes()
    plan = memory_limit_plan(total, wired_ratio=0.70)
    apply_memory_limit_plan(plan)


def _make_inputs(seq, seed=0):
    batch, heads, headdim, state = 1, 112, 64, 64
    mx.random.seed(seed)
    shp = (batch, seq, heads, headdim)
    x = mx.random.normal(shp) * 0.1
    B = mx.random.normal((batch, seq, heads, state)) * 0.1
    C = mx.random.normal((batch, seq, heads, state)) * 0.1
    z = mx.random.normal(shp) * 0.1
    A = -mx.abs(mx.random.normal((batch, seq, heads))) * 0.5 - 0.1
    dt = mx.abs(mx.random.normal((batch, seq, heads))) * 0.05 + 0.01
    D = mx.random.normal((heads,)) * 0.1
    h0 = mx.random.normal((batch, heads, headdim, state)) * 0.1
    arrs = [x, B, C, z, A, dt, D, h0]
    mx.eval(*arrs)
    return arrs


def main():
    seq = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    _apply_memguard()
    inputs = _make_inputs(seq)

    def loss(*args):
        y, _h = mamba3_mimo_apply_with_state_path_c(*args)
        return mx.sum(y * y)

    g = mx.value_and_grad(loss, argnums=tuple(range(8)))
    _val, grads = g(*inputs)
    mx.eval(_val, *grads)
    h = hashlib.sha256()
    for arr in grads:
        h.update(np.asarray(arr, dtype=np.float32).tobytes())
    print(h.hexdigest(), flush=True)


if __name__ == "__main__":
    main()

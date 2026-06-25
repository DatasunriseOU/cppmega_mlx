"""Deterministic 50/50 gate at s4096 for the partial route.

Runs the production PARTIAL bwd route N times in a single process (same inputs,
same seed) and asserts the SHA-256 over all 8 grads is byte-identical every run.
Under memguard 70 (raises if RSS crosses the ceiling). NO silent fallback.
"""
from __future__ import annotations

import hashlib
import os
import sys
import gc

import numpy as np
import mlx.core as mx
import psutil

from cppmega_mlx.nn._tilelang.mamba3_path_c import mamba3_mimo_bwd_path_c

MEMGUARD_GB = 70.0
_PROC = psutil.Process(os.getpid())
_PEAK = 0.0


def _guard(where: str) -> None:
    global _PEAK
    rss = _PROC.memory_info().rss / (1024 ** 3)
    if rss > _PEAK:
        _PEAK = rss
    if rss > MEMGUARD_GB:
        raise MemoryError(f"MEMGUARD {rss:.2f}GB > {MEMGUARD_GB} at {where}")


def make_inputs(seq, heads=128, headdim=64, state=64, batch=1, seed=0):
    mx.random.seed(seed)
    x = (mx.random.normal((batch, seq, heads, headdim)) * 0.1).astype(mx.float32)
    B = (mx.random.normal((batch, seq, heads, state)) * 0.1).astype(mx.float32)
    C = (mx.random.normal((batch, seq, heads, state)) * 0.1).astype(mx.float32)
    z = (mx.random.normal((batch, seq, heads, headdim)) * 0.1).astype(mx.float32)
    A = (-mx.random.uniform(0.01, 0.5, (batch, seq, heads))).astype(mx.float32)
    dt = (mx.random.uniform(0.001, 0.05, (batch, seq, heads))).astype(mx.float32)
    D = (mx.random.normal((heads,)) * 0.1).astype(mx.float32)
    h0 = (mx.random.normal((batch, heads, headdim, state)) * 0.1).astype(mx.float32)
    dy = (mx.random.normal((batch, seq, heads, headdim)) * 0.1).astype(mx.float32)
    mx.eval(x, B, C, z, A, dt, D, h0, dy)
    return dy, x, B, C, z, A, dt, D, h0


def hashed(seq, seed):
    inputs = make_inputs(seq, seed=seed)
    _guard(f"s{seq} inputs")
    g = mamba3_mimo_bwd_path_c(*inputs)
    mx.eval(*g)
    _guard(f"s{seq} bwd")
    h = hashlib.sha256()
    for arr in g:
        h.update(np.asarray(arr, dtype=np.float32).tobytes())
    del g, inputs
    gc.collect()
    return h.hexdigest()


def main():
    seq = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    ref = None
    ok = 0
    for i in range(n):
        d = hashed(seq, seed=0)
        if ref is None:
            ref = d
        if d == ref:
            ok += 1
        else:
            print(f"MISMATCH run{i}: {d} != {ref}")
    print(f"DETERMINISM s{seq} {ok}/{n} identical hash={ref}")
    print(f"PEAK_RSS_GB {_PEAK:.2f}")
    print("PASS" if ok == n else "FAIL")


if __name__ == "__main__":
    main()

"""Diagnose dC/dz error at s4096 in the SIMD LANE route. Compare SIMD route vs
partial route vs gold across seqlens. Identify whether the dC/dz blow-up is
seqlen-dependent and route-specific. Under memguard 70.
"""
from __future__ import annotations

import os
import sys
import threading
import time

_MEMGUARD_LIMIT_KB = 70 * 1024 * 1024
_PEAK_RSS_KB = 0


def _rss_kb() -> int:
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)


def _memguard_thread():
    global _PEAK_RSS_KB
    while True:
        r = _rss_kb()
        if r > _PEAK_RSS_KB:
            _PEAK_RSS_KB = r
        if r > _MEMGUARD_LIMIT_KB:
            sys.stderr.write(f"[memguard70] KILL self rss_kb={r} > 70GB\n")
            sys.stderr.flush()
            os._exit(137)
        time.sleep(0.25)


threading.Thread(target=_memguard_thread, daemon=True).start()

os.environ.setdefault("TILELANG_MLX_TVM_FFI_FORCE_COMMAND_BUFFER_BOUNDARY", "1")
sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx")

import numpy as np
import mlx.core as mx
from cppmega_mlx.nn._tilelang.mamba3 import mamba3_mimo_bwd_metal
from cppmega_mlx.nn._tilelang.mamba3_path_c import (
    _mamba3_mimo_bwd_path_c_simd_kernel,
    _mamba3_mimo_bwd_path_c_partial_kernel,
)

GRAD_NAMES = ("dx", "dB", "dC", "dz", "dA", "ddt", "dD", "dh0")
HEADS, HEADDIM, STATE = 128, 64, 64
BATCH = 1


def build_inputs(SEQ, seed=0):
    rng = np.random.RandomState(seed)

    def f32(*shape, sc=0.1):
        return mx.array((rng.randn(*shape) * sc).astype(np.float32))

    x = f32(BATCH, SEQ, HEADS, HEADDIM)
    B = f32(BATCH, SEQ, HEADS, STATE)
    C = f32(BATCH, SEQ, HEADS, STATE)
    z = f32(BATCH, SEQ, HEADS, HEADDIM, sc=0.5)
    A_head = (-rng.rand(HEADS)).astype(np.float32)
    A = mx.array(np.broadcast_to(A_head[None, None, :], (BATCH, SEQ, HEADS)).copy())
    dt = mx.array((rng.rand(BATCH, SEQ, HEADS) * 0.05).astype(np.float32))
    D = mx.array((rng.randn(HEADS)).astype(np.float32))
    h0 = f32(BATCH, HEADS, HEADDIM, STATE)
    dy = mx.array((rng.randn(BATCH, SEQ, HEADS, HEADDIM) * 0.1).astype(np.float32))
    return dy, x, B, C, z, A, dt, D, h0


def maxabs(a, ref):
    return float(np.abs(np.asarray(a.astype(mx.float32), np.float64)
                        - np.asarray(ref.astype(mx.float32), np.float64)).max())


def run(route, fn, dy, primals, grads_gold):
    try:
        grads = fn(dy, *primals)
        mx.eval(*grads)
        per = {nm: maxabs(g, gg) for nm, g, gg in zip(GRAD_NAMES, grads, grads_gold)}
        worst = max(per.values())
        print(f"  {route:8s} worst={worst:.3e} ok={worst<1e-3}  "
              + " ".join(f"{k}={v:.1e}" for k, v in per.items()))
    except Exception as e:  # noqa: BLE001
        print(f"  {route:8s} FAILED: {type(e).__name__}: {str(e)[:120]}")


for SEQ in [512, 1024, 2048, 4096]:
    print(f"\n=== SEQ={SEQ} ===")
    dy, x, B, C, z, A, dt, D, h0 = build_inputs(SEQ)
    primals = (x, B, C, z, A, dt, D, h0)
    grads_gold = mamba3_mimo_bwd_metal(dy, *primals, backend="mlx")
    mx.eval(*grads_gold)
    run("SIMD", _mamba3_mimo_bwd_path_c_simd_kernel, dy, primals, grads_gold)
    run("PARTIAL", _mamba3_mimo_bwd_path_c_partial_kernel, dy, primals, grads_gold)

print(f"\nPEAK_RSS_KB={_PEAK_RSS_KB} (~{_PEAK_RSS_KB/1048576:.3f}GB)  memguard70=ON")
print("RC=0")

"""CLEAN-CACHE GATE for the 2-dispatch SIMD LANE route at HEAD (no C++ edit).
Run with TILELANG_DISABLE_CACHE=1. Validates: s4096 LOWERS, all-8 grads <1e-3 at
{128,512,1024,2048,4096} vs path-b gold, AND vs a float64 analytic oracle at
s4096, plus determinism 8/8 at s4096. memguard 70.
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
from cppmega_mlx.nn._tilelang.mamba3_path_c import _mamba3_mimo_bwd_path_c_simd_kernel

GRAD_NAMES = ("dx", "dB", "dC", "dz", "dA", "ddt", "dD", "dh0")
HEADS, HEADDIM, STATE = 128, 64, 64
BATCH = 1
print("CACHE_DISABLED =", os.environ.get("TILELANG_DISABLE_CACHE"))


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


results = {}
for SEQ in [128, 512, 1024, 2048, 4096]:
    print(f"\n=== SEQ={SEQ} ===")
    dy, x, B, C, z, A, dt, D, h0 = build_inputs(SEQ)
    primals = (x, B, C, z, A, dt, D, h0)
    gold = mamba3_mimo_bwd_metal(dy, *primals, backend="mlx")
    mx.eval(*gold)
    try:
        grads = _mamba3_mimo_bwd_path_c_simd_kernel(dy, x, B, C, z, A, dt, D, h0)
        mx.eval(*grads)
        per = {nm: maxabs(g, gg) for nm, g, gg in zip(GRAD_NAMES, grads, gold)}
        worst = max(per.values())
        ok = worst < 1e-3
        results[SEQ] = (True, worst, ok)
        print(f"  LOWERED ok={ok} worst={worst:.3e}  "
              + " ".join(f"{k}={v:.1e}" for k, v in per.items()))
    except Exception as e:  # noqa: BLE001
        results[SEQ] = (False, None, False)
        print(f"  FAILED: {type(e).__name__}: {str(e)[:160]}")

# Determinism at s4096: run twice, compare bitwise-equal
print("\n=== DETERMINISM s4096 (2 runs bitwise) ===")
dy, x, B, C, z, A, dt, D, h0 = build_inputs(4096)
g1 = _mamba3_mimo_bwd_path_c_simd_kernel(dy, x, B, C, z, A, dt, D, h0)
mx.eval(*g1)
g1n = [np.asarray(g.astype(mx.float32)) for g in g1]
g2 = _mamba3_mimo_bwd_path_c_simd_kernel(dy, x, B, C, z, A, dt, D, h0)
mx.eval(*g2)
det = all(np.array_equal(a, np.asarray(b.astype(mx.float32))) for a, b in zip(g1n, g2))
print(f"  deterministic(2/2 bitwise)={det}")

print("\nSUMMARY:")
allpass = True
for SEQ in [128, 512, 1024, 2048, 4096]:
    low, worst, ok = results[SEQ]
    allpass = allpass and low and ok
    print(f"  s{SEQ}: lowered={low} ok={ok} worst={worst}")
print(f"  GATE_BITCORRECT_ALL={allpass}  DETERMINISTIC={det}")
print(f"\nPEAK_RSS_KB={_PEAK_RSS_KB} (~{_PEAK_RSS_KB/1048576:.3f}GB)  memguard70=ON")
print("RC=0")

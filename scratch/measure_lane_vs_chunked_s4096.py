"""Head-to-head s4096: 2-dispatch SIMD LANE vs 3-dispatch PARTIAL (chunked
fallback). Wall-time both, warmup then measure. Reports us each + ratio.
memguard 70. Run at HEAD (no C++ edit) -- LANE already lowers correctly.
"""
from __future__ import annotations
import os, sys, threading, time
_LIM = 70 * 1024 * 1024
_PEAK = 0


def _rss():
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)


def _guard():
    global _PEAK
    while True:
        r = _rss()
        if r > _PEAK:
            _PEAK = r
        if r > _LIM:
            sys.stderr.write(f"[memguard70] KILL rss_kb={r}\n"); sys.stderr.flush(); os._exit(137)
        time.sleep(0.25)


threading.Thread(target=_guard, daemon=True).start()
os.environ.setdefault("TILELANG_MLX_TVM_FFI_FORCE_COMMAND_BUFFER_BOUNDARY", "1")
sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx")
import numpy as np
import mlx.core as mx
from cppmega_mlx.nn._tilelang.mamba3_path_c import (
    _mamba3_mimo_bwd_path_c_simd_kernel,
    _mamba3_mimo_bwd_path_c_partial_kernel,
)

HEADS, HEADDIM, STATE, BATCH = 128, 64, 64, 1
SEQ = 4096
rng = np.random.RandomState(0)


def f32(*shape, sc=0.1):
    return mx.array((rng.randn(*shape) * sc).astype(np.float32))


x = f32(BATCH, SEQ, HEADS, HEADDIM); B = f32(BATCH, SEQ, HEADS, STATE)
C = f32(BATCH, SEQ, HEADS, STATE); z = f32(BATCH, SEQ, HEADS, HEADDIM, sc=0.5)
A_head = (-rng.rand(HEADS)).astype(np.float32)
A = mx.array(np.broadcast_to(A_head[None, None, :], (BATCH, SEQ, HEADS)).copy())
dt = mx.array((rng.rand(BATCH, SEQ, HEADS) * 0.05).astype(np.float32))
D = mx.array((rng.randn(HEADS)).astype(np.float32))
h0 = f32(BATCH, HEADS, HEADDIM, STATE)
dy = mx.array((rng.randn(BATCH, SEQ, HEADS, HEADDIM) * 0.1).astype(np.float32))
args = (dy, x, B, C, z, A, dt, D, h0)

WARMUP, RUNS = 3, 7


def bench(fn, name):
    for _ in range(WARMUP):
        g = fn(*args); mx.eval(*g)
    ts = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        g = fn(*args); mx.eval(*g)
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    med = ts[len(ts) // 2]
    print(f"  {name:10s} median={med:.1f} us  min={ts[0]:.1f}  (runs={ts})")
    return med


print("=== s4096 LANE vs CHUNKED (warmup=3 runs=7) ===")
lane_us = bench(_mamba3_mimo_bwd_path_c_simd_kernel, "SIMD_LANE")
chunk_us = bench(_mamba3_mimo_bwd_path_c_partial_kernel, "PARTIAL")
print(f"\n  LANE={lane_us:.1f}us  CHUNKED(partial)={chunk_us:.1f}us  "
      f"speedup_lane_over_chunked={chunk_us/lane_us:.3f}x  "
      f"faster_than_chunked={lane_us < chunk_us}")
print(f"\nPEAK_RSS_KB={_PEAK} (~{_PEAK/1048576:.3f}GB)  memguard70=ON")
print("RC=0")

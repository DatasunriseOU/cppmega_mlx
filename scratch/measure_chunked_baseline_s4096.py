"""Measure the CHUNKED-FALLBACK baseline us at s4096: the 3-dispatch fp32 partial
route (_mamba3_mimo_bwd_path_c_partial_kernel) that production currently uses for
fp32. This is the 'faster-than-chunked' gate denominator. Read+probe only.
Under memguard 70.
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
        if r > _PEAK: _PEAK = r
        if r > _LIM:
            sys.stderr.write(f"[memguard70] KILL rss_kb={r}\n"); sys.stderr.flush(); os._exit(137)
        time.sleep(0.25)
threading.Thread(target=_guard, daemon=True).start()
os.environ.setdefault("TILELANG_MLX_TVM_FFI_FORCE_COMMAND_BUFFER_BOUNDARY", "1")
sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx")

import numpy as np
import mlx.core as mx
from cppmega_mlx.nn._tilelang.mamba3_path_c import _mamba3_mimo_bwd_path_c_partial_kernel

HEADS, HEADDIM, STATE, BATCH = 128, 64, 64, 1
SEQ = 4096
rng = np.random.RandomState(0)
def f32(*shape, sc=0.1): return mx.array((rng.randn(*shape) * sc).astype(np.float32))
x = f32(BATCH, SEQ, HEADS, HEADDIM); B = f32(BATCH, SEQ, HEADS, STATE)
C = f32(BATCH, SEQ, HEADS, STATE); z = f32(BATCH, SEQ, HEADS, HEADDIM, sc=0.5)
A_head = (-rng.rand(HEADS)).astype(np.float32)
A = mx.array(np.broadcast_to(A_head[None, None, :], (BATCH, SEQ, HEADS)).copy())
dt = mx.array((rng.rand(BATCH, SEQ, HEADS) * 0.05).astype(np.float32))
D = mx.array((rng.randn(HEADS)).astype(np.float32))
h0 = f32(BATCH, HEADS, HEADDIM, STATE)
dy = mx.array((rng.randn(BATCH, SEQ, HEADS, HEADDIM) * 0.1).astype(np.float32))

def run():
    g = _mamba3_mimo_bwd_path_c_partial_kernel(dy, x, B, C, z, A, dt, D, h0)
    mx.eval(*g)

# warmup
for _ in range(3): run()
N = 20
ts = []
for _ in range(N):
    t0 = time.perf_counter(); run(); ts.append((time.perf_counter() - t0) * 1e6)
ts.sort()
print(f"CHUNKED-FALLBACK (3-dispatch fp32 partial) s4096 bwd: "
      f"median={ts[N//2]:.1f}us min={ts[0]:.1f}us max={ts[-1]:.1f}us  N={N}")
print(f"PEAK_RSS_KB={_PEAK} (~{_PEAK/1048576:.3f}GB)  memguard70=ON")
print("RC=0")

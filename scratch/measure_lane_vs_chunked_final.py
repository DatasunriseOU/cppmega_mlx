"""DECISIVE same-process head-to-head at s4096: 2-dispatch SIMD LANE vs the
working CHUNKED backward route (_mamba3_chunked_backward_path_c). Both warmed up
and timed identically. Also bit-correctness of both vs path-b gold. memguard 70.
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
from cppmega_mlx.nn._tilelang.mamba3 import mamba3_mimo_bwd_metal
from cppmega_mlx.nn._tilelang.mamba3_path_c import (
    _mamba3_mimo_bwd_path_c_simd_kernel,
    _mamba3_chunked_backward_path_c,
    _mamba3_chunked_fwd_intermediates_path_c,
)

HEADS, HEADDIM, STATE, BATCH = 128, 64, 64, 1
SEQ = 4096
GRAD_NAMES = ("dx", "dB", "dC", "dz", "dA", "ddt", "dD", "dh0")
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
primals = (x, B, C, z, A, dt, D, h0)

gold = mamba3_mimo_bwd_metal(dy, *primals, backend="mlx")
mx.eval(*gold)


def maxabs(a, ref):
    return float(np.abs(np.asarray(a.astype(mx.float32), np.float64)
                        - np.asarray(ref.astype(mx.float32), np.float64)).max())


def lane_fn():
    return _mamba3_mimo_bwd_path_c_simd_kernel(dy, x, B, C, z, A, dt, D, h0)


def chunked_fn():
    cb, dA_cumsum, prev_states = _mamba3_chunked_fwd_intermediates_path_c(x, B, C, A, dt, h0)
    from cppmega_mlx.nn._tilelang.mamba3 import mamba3_mimo_reference
    y_skip, _ = mamba3_mimo_reference(x, B, C, z, A, dt, D, h0)
    return _mamba3_chunked_backward_path_c(
        dy, x, B, C, z, A, dt, D, h0,
        cb=cb, dA_cumsum=dA_cumsum, prev_states=prev_states, y=y_skip)


WARMUP, RUNS = 3, 7


def bench(fn, name):
    try:
        for _ in range(WARMUP):
            g = fn(); mx.eval(*g)
    except Exception as e:  # noqa: BLE001
        print(f"  {name:10s} FAILED warmup: {type(e).__name__}: {str(e)[:140]}")
        return None, None
    g = fn(); mx.eval(*g)
    per = {nm: maxabs(gv, gg) for nm, gv, gg in zip(GRAD_NAMES, g, gold)}
    worst = max(per.values())
    ts = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        g = fn(); mx.eval(*g)
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    med = ts[len(ts) // 2]
    print(f"  {name:10s} median={med:.1f}us min={ts[0]:.1f} bitcorrect_worst={worst:.2e} ok={worst<1e-3}")
    return med, worst


print("=== s4096 SIMD LANE vs CHUNKED (warmup=3 runs=7) ===")
lane_us, lane_w = bench(lane_fn, "SIMD_LANE")
chunk_us, chunk_w = bench(chunked_fn, "CHUNKED")
if lane_us and chunk_us:
    print(f"\n  LANE={lane_us:.1f}us  CHUNKED={chunk_us:.1f}us  "
          f"chunked/lane={chunk_us/lane_us:.3f}x  "
          f"LANE_faster_than_chunked={lane_us < chunk_us}")
print(f"\nPEAK_RSS_KB={_PEAK} (~{_PEAK/1048576:.3f}GB)  memguard70=ON")
print("RC=0")

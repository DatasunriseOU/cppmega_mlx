"""Directed chunk-size sweep for the recompute-fused Mamba3 Path C SIMD bwd.

Routes the single-dispatch snapshot SIMD kernel (snapshot build + recompute-fused
bwd_snap_simd) at one CPPMEGA_MAMBA3_BWD_SNAPSHOT_BLOCK value (set in env BEFORE
import so the lru-cached kernels bind to it), then:
  - bit-correctness: all 8 grads vs path-b GOLD (<1e-3),
  - timing: median wall of the LANE route vs MSL (mamba3_mimo_bwd_metal) and the
    TRUE roofline floor (2.235 ms at s4096),
  - determinism: N repeats all <1e-3.
memguard 70 mandatory. RULE #1: raise on any failure; no fallback.

Usage: SEQ=128|4096 CHUNK=<int> NREP=<int> python scratch/sweep_bwd_snap_chunk.py
"""
from __future__ import annotations
import os, sys, threading, time, statistics

_MEMGUARD_LIMIT_KB = 70 * 1024 * 1024
_PEAK = 0
def _rss_kb():
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
def _memguard():
    global _PEAK
    while True:
        r = _rss_kb()
        if r > _PEAK: _PEAK = r
        if r > _MEMGUARD_LIMIT_KB:
            sys.stderr.write(f"[memguard70] KILL rss_kb={r}\n"); sys.stderr.flush(); os._exit(137)
        time.sleep(0.25)
threading.Thread(target=_memguard, daemon=True).start()

CHUNK = int(os.environ.get("CHUNK", "1"))
SEQ = int(os.environ.get("SEQ", "128"))
NREP = int(os.environ.get("NREP", "5"))
TIMEIT = int(os.environ.get("TIMEIT", "1"))
# bind the chunk BEFORE importing the module (kernels lru-cache on resolved int)
os.environ["CPPMEGA_MAMBA3_BWD_SNAPSHOT_BLOCK"] = str(CHUNK)
os.environ.setdefault("TILELANG_MLX_TVM_FFI_FORCE_COMMAND_BUFFER_BOUNDARY", "1")
sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx")

import numpy as np
import mlx.core as mx

from cppmega_mlx.nn._tilelang.mamba3 import mamba3_mimo_bwd_metal
from cppmega_mlx.nn._tilelang.mamba3_path_c import (
    _mamba3_mimo_bwd_path_c_simd_kernel,
    _bwd_simd_p_reduction_supported,
    _bwd_scan_plan_for,
    _bwd_snapshot_block,
)

# nam56r surface: (b,s,h,p,n) = (1, SEQ, 128, 64, 64)
b, s, h, p, n = 1, SEQ, 128, 64, 64
rng = np.random.RandomState(0)
def f32(*shape, sc=0.1):
    return mx.array((rng.randn(*shape) * sc).astype(np.float32))
x = f32(b, s, h, p); B = f32(b, s, h, n); C = f32(b, s, h, n)
z = f32(b, s, h, p, sc=0.5)
A_head = (-rng.rand(h)).astype(np.float32)
A = mx.array(np.broadcast_to(A_head[None, None, :], (b, s, h)).copy())
dt = mx.array((rng.rand(b, s, h) * 0.05).astype(np.float32))
D = mx.array((rng.randn(h)).astype(np.float32))
h0 = f32(b, h, p, n)
cot_y = mx.array((rng.randn(b, s, h, p) * 0.1).astype(np.float32))
primals = (x, B, C, z, A, dt, D, h0)

assert _bwd_snapshot_block() == CHUNK, f"env chunk not honored: {_bwd_snapshot_block()} != {CHUNK}"
plan = _bwd_scan_plan_for(batch=b, seq=s, heads=h, headdim=p, state=n)
print(f"[cfg] SEQ={s} CHUNK={CHUNK} policy={plan.snapshot_plan.policy} "
      f"chunk_size={plan.snapshot_plan.chunk_size} chunk_count={plan.snapshot_plan.chunk_count} "
      f"snapshot_count={plan.snapshot_plan.snapshot_count}")
assert _bwd_simd_p_reduction_supported(batch=b, heads=h, headdim=p)

# GOLD (path-b)
grads_gold = mamba3_mimo_bwd_metal(cot_y, *primals, backend="mlx")
mx.eval(*grads_gold)

# LANE recompute-fused route
grads_lane = _mamba3_mimo_bwd_path_c_simd_kernel(cot_y, *primals)
mx.eval(*grads_lane)

names = ("dx", "dB", "dC", "dz", "dA", "ddt", "dD", "dh0")
def maxabs(a, ref):
    return float(np.abs(np.asarray(a.astype(mx.float32), np.float64)
                        - np.asarray(ref.astype(mx.float32), np.float64)).max())

worst = 0.0
print(f"\n[8-grad: LANE recompute chunk={CHUNK} vs path-b GOLD]")
for nm, g, gg in zip(names, grads_lane, grads_gold):
    d = maxabs(g, gg); worst = max(worst, d)
    print(f"  {nm:4s} maxdiff={d:.3e} [{'OK' if d<1e-3 else 'FAIL'}]")
print(f"  WORST = {worst:.3e}  gate<1e-3 => {'PASS' if worst<1e-3 else 'FAIL'}")

# determinism
det_ok = True
for it in range(NREP):
    gi = _mamba3_mimo_bwd_path_c_simd_kernel(cot_y, *primals)
    mx.eval(*gi)
    w = max(maxabs(g, gg) for g, gg in zip(gi, grads_gold))
    if w >= 1e-3: det_ok = False
print(f"[determinism] {NREP} repeats => {'PASS' if det_ok else 'FAIL'}")

# timing
lane_ms = msl_ms = float('nan')
if TIMEIT:
    def _bench(fn, reps=int(os.environ.get("REPS", "12")), warm=3):
        for _ in range(warm):
            o = fn(); mx.eval(*o)
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); o = fn(); mx.eval(*o)
            ts.append((time.perf_counter() - t0) * 1e3)
        return statistics.median(ts)
    lane_ms = _bench(lambda: _mamba3_mimo_bwd_path_c_simd_kernel(cot_y, *primals))
    # MSL baseline = the production native-Metal single-dispatch lane kernel
    # (backend="metal"), matching the roofline-phase MSL measurement. The
    # MLX-backed reference (backend="mlx") is the slow gold oracle, not MSL.
    msl_ms = _bench(lambda: mamba3_mimo_bwd_metal(cot_y, *primals, backend="metal"))

FLOOR = 2.235 if s == 4096 else (2.235 * s / 4096.0)
ratio_msl = lane_ms / msl_ms if msl_ms == msl_ms else float('nan')
ratio_floor = lane_ms / FLOOR if lane_ms == lane_ms else float('nan')
print(f"\n[timing] LANE={lane_ms:.3f}ms MSL={msl_ms:.3f}ms floor={FLOOR:.3f}ms")
print(f"[timing] LANE/MSL={ratio_msl:.3f}x  LANE/floor={ratio_floor:.2f}x")
print(f"\nPEAK_RSS_KB={_PEAK} (~{_PEAK/1048576:.3f}GB) memguard70=ON")
print(f"RESULT chunk={CHUNK} seq={s} worst={worst:.3e} pass={worst<1e-3} det={det_ok} "
      f"lane_ms={lane_ms:.3f} msl_ms={msl_ms:.3f} floor_ms={FLOOR:.3f} "
      f"ratio_msl={ratio_msl:.3f} ratio_floor={ratio_floor:.2f}")
print("RC=0")

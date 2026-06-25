"""GROUND TRUTH: does the in-kernel SIMD P-reduce LANE route run at s4096, is it
bit-correct vs path-b gold, and how fast vs MSL (backend=metal) and the PARTIAL
(expansion) route? Single authoritative measurement. memguard 70.

Run:  /opt/homebrew/bin/python3.13 scratch/measure_simd_lane_ground_truth.py [SEQ ...]
Default SEQ list = [4096].
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
            sys.stderr.write(f"[memguard70] KILL rss_kb={r}\n")
            sys.stderr.flush()
            os._exit(137)
        time.sleep(0.25)


threading.Thread(target=_guard, daemon=True).start()
os.environ.setdefault("TILELANG_MLX_TVM_FFI_FORCE_COMMAND_BUFFER_BOUNDARY", "1")
sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx")

import numpy as np  # noqa: E402
import mlx.core as mx  # noqa: E402
from cppmega_mlx.nn._tilelang.mamba3 import mamba3_mimo_bwd_metal  # noqa: E402
from cppmega_mlx.nn._tilelang.mamba3_path_c import (  # noqa: E402
    _mamba3_mimo_bwd_path_c_simd_kernel,
    _mamba3_mimo_bwd_path_c_partial_kernel,
)

GRAD_NAMES = ("dx", "dB", "dC", "dz", "dA", "ddt", "dD", "dh0")
HEADS, HEADDIM, STATE, BATCH = 128, 64, 64, 1


def build(SEQ, seed=0):
    rng = np.random.RandomState(seed)

    def f32(*s, sc=0.1):
        return mx.array((rng.randn(*s) * sc).astype(np.float32))

    x = f32(BATCH, SEQ, HEADS, HEADDIM)
    B = f32(BATCH, SEQ, HEADS, STATE)
    C = f32(BATCH, SEQ, HEADS, STATE)
    z = f32(BATCH, SEQ, HEADS, HEADDIM, sc=0.5)
    Ah = (-rng.rand(HEADS)).astype(np.float32)
    A = mx.array(np.broadcast_to(Ah[None, None, :], (BATCH, SEQ, HEADS)).copy())
    dt = mx.array((rng.rand(BATCH, SEQ, HEADS) * 0.05).astype(np.float32))
    D = mx.array((rng.randn(HEADS)).astype(np.float32))
    h0 = f32(BATCH, HEADS, HEADDIM, STATE)
    dy = mx.array((rng.randn(BATCH, SEQ, HEADS, HEADDIM) * 0.1).astype(np.float32))
    return dy, x, B, C, z, A, dt, D, h0


def maxabs(a, ref):
    return float(np.abs(np.asarray(a.astype(mx.float32), np.float64)
                        - np.asarray(ref.astype(mx.float32), np.float64)).max())


def bench(fn, args, name, warmup=3, runs=7):
    for _ in range(warmup):
        g = fn(*args)
        mx.eval(*g)
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        g = fn(*args)
        mx.eval(*g)
        ts.append((time.perf_counter() - t0) * 1e3)  # ms
    ts.sort()
    med = ts[len(ts) // 2]
    print(f"  {name:16s} median={med:8.3f}ms  min={ts[0]:8.3f}ms")
    return med


SEQS = [int(a) for a in sys.argv[1:]] or [4096]

for SEQ in SEQS:
    print(f"\n=========== SEQ={SEQ} (BATCH={BATCH} HEADS={HEADS} HEADDIM={HEADDIM} STATE={STATE}) ===========")
    args = build(SEQ)
    dy = args[0]
    primals = args[1:]

    # gold (path-b mlx reference)
    gold = mamba3_mimo_bwd_metal(dy, *primals, backend="mlx")
    mx.eval(*gold)

    # MSL baseline (backend=metal)
    def msl(*a):
        return mamba3_mimo_bwd_metal(a[0], *a[1:], backend="metal")

    # bit-correctness of SIMD lane route
    try:
        simd = _mamba3_mimo_bwd_path_c_simd_kernel(*args)
        mx.eval(*simd)
        worst = 0.0
        per = {}
        for nm, g, gg in zip(GRAD_NAMES, simd, gold):
            d = maxabs(g, gg)
            per[nm] = d
            worst = max(worst, d)
        bad = {k: f"{v:.2e}" for k, v in per.items() if v >= 1e-3}
        print(f"  SIMD_LANE bit-correct: worst={worst:.3e} ok(<1e-3)={worst < 1e-3}"
              + (f"  FAILING={bad}" if bad else ""))
        print(f"    per-grad: " + " ".join(f"{k}={v:.2e}" for k, v in per.items()))
        simd_runs = True
    except Exception as e:  # noqa: BLE001
        import traceback
        print(f"  SIMD_LANE FAILED: {type(e).__name__}: {str(e)[:300]}")
        for ln in traceback.format_exc().splitlines():
            if any(s in ln for s in ("int32", "2147483", "exceeds", "FlattenBuffer", "overflow")):
                print(f"      >> {ln.strip()}")
        simd_runs = False

    # timings
    msl_ms = bench(msl, args, "MSL(metal)")
    if simd_runs:
        simd_ms = bench(_mamba3_mimo_bwd_path_c_simd_kernel, args, "SIMD_LANE")
        print(f"  >>> SIMD/MSL = {simd_ms / msl_ms:.3f}x  (<1.0 means SIMD BEATS MSL)")
    try:
        part_ms = bench(_mamba3_mimo_bwd_path_c_partial_kernel, args, "PARTIAL(expand)")
        print(f"  >>> PARTIAL/MSL = {part_ms / msl_ms:.3f}x")
    except Exception as e:  # noqa: BLE001
        print(f"  PARTIAL FAILED: {type(e).__name__}: {str(e)[:200]}")

print(f"\nPEAK_RSS_KB={_PEAK} (~{_PEAK/1048576:.3f}GB)  memguard70=ON")
print("RC=0")

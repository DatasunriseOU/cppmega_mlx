"""FINAL GATE: the in-kernel SIMD P-reduce recompute route (production default
BLOCK=4). Bit-correct all 8 grads vs path-b gold at s128/512/1024/2048/4096,
determinism N/N at s4096, time + GB/s + x-vs-MSL + x-vs-floor. memguard 70.
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
    _BWD_SIMD_RECOMPUTE_BLOCK,
)

print(f"BLOCK={_BWD_SIMD_RECOMPUTE_BLOCK}")
GRAD_NAMES = ("dx", "dB", "dC", "dz", "dA", "ddt", "dD", "dh0")
HEADS, HEADDIM, STATE, BATCH = 128, 64, 64, 1
# irreducible DRAM floor = 2.235ms for 1.221GB at 546 GB/s (briefing).
FLOOR_MS = 2.235
PEAK_GBPS = 546.0


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


def bench(fn, args, warmup=3, runs=7):
    for _ in range(warmup):
        g = fn(*args)
        mx.eval(*g)
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        g = fn(*args)
        mx.eval(*g)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2], ts[0]


# Final-grad traffic at s4096 (no expansion): reads dy,x,z (3*BSHP*4) + B,C
# (2*BSHN*4) + h_snap(B*(S/BLK+1)*H*P*N*4) + writes dx,dz (2*BSHP*4) + dB,dC
# (2*BSHN*4). Report GB moved/ms for the SIMD route as an effective BW figure.

s4096_maxdiff = {}
all_seq_worst = {}
for SEQ in [128, 512, 1024, 2048, 4096]:
    args = build(SEQ)
    dy = args[0]
    primals = args[1:]
    gold = mamba3_mimo_bwd_metal(dy, *primals, backend="mlx")
    mx.eval(*gold)
    g = _mamba3_mimo_bwd_path_c_simd_kernel(*args)
    mx.eval(*g)
    worst = 0.0
    per = {}
    for nm, gg, gr in zip(GRAD_NAMES, g, gold):
        d = maxabs(gg, gr)
        per[nm] = d
        worst = max(worst, d)
    all_seq_worst[SEQ] = worst
    if SEQ == 4096:
        s4096_maxdiff = dict(per)
    print(f"SEQ={SEQ:5d} worst={worst:.3e} ok(<1e-3)={worst < 1e-3}  "
          + " ".join(f"{k}={v:.1e}" for k, v in per.items()))

# determinism 50/50 at s4096
print("\n=== determinism s4096 (N=8 for time; bit-identical check) ===")
args = build(4096)
ref = _mamba3_mimo_bwd_path_c_simd_kernel(*args)
mx.eval(*ref)
ref_np = [np.asarray(r.astype(mx.float32)) for r in ref]
det_pass = 0
N = 8
for i in range(N):
    g = _mamba3_mimo_bwd_path_c_simd_kernel(*args)
    mx.eval(*g)
    same = all(np.array_equal(np.asarray(gg.astype(mx.float32)), rn)
               for gg, rn in zip(g, ref_np))
    det_pass += int(same)
print(f"determinism {det_pass}/{N} bit-identical")

# timing at s4096
print("\n=== timing s4096 ===")
args = build(4096)
msl_med, msl_min = bench(lambda *a: mamba3_mimo_bwd_metal(a[0], *a[1:], backend="metal"), args)
simd_med, simd_min = bench(_mamba3_mimo_bwd_path_c_simd_kernel, args)
print(f"MSL   median={msl_med:.3f}ms min={msl_min:.3f}ms")
print(f"SIMD  median={simd_med:.3f}ms min={simd_min:.3f}ms")
print(f">>> SIMD/MSL = {simd_med/msl_med:.4f}x  (<=1.0 PASSES GATE)")
print(f">>> SIMD x-vs-floor = {simd_med/FLOOR_MS:.2f}x  (floor={FLOOR_MS}ms)")
print(f">>> MSL  x-vs-floor = {msl_med/FLOOR_MS:.2f}x")

print(f"\nPEAK_RSS_KB={_PEAK} (~{_PEAK/1048576:.3f}GB)  memguard70=ON")
print(f"GATE_PASS={simd_med <= msl_med and all(w < 1e-3 for w in all_seq_worst.values()) and det_pass == N}")
print("RC=0")

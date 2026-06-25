"""PARITY + BENCH: path_c_fwd_path_c_bwd vs path_c_fwd_path_b_bwd (prod MSL).

Compares ALL 8 grads (dx,dB,dC,dz,dA,ddt,dD,dh0) on the nam56r-config surface,
within the established < 1e-3 per-grad tolerance from
tests/test_mamba3_chunked_backward_b0b1b2.py (NOT loosened).

Reference (GOLD): the pure-MLX fp32 backward oracle (mamba3_mimo_bwd_metal
backend='mlx') is the step-by-step fp32 parity oracle. We also report the prod
MSL metal backward (path_c_fwd_path_b_bwd) so the chunked path is judged against
the SAME production path it would replace.

memguard 70. No fabrication. RULE #1: no silent fallback anywhere.
"""
import os
import sys
import threading
import time

_MEMGUARD_LIMIT_KB = 70 * 1024 * 1024
_PEAK = 0


def _rss_kb():
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)


def _guard():
    global _PEAK
    while True:
        r = _rss_kb()
        if r > _PEAK:
            _PEAK = r
        if r > _MEMGUARD_LIMIT_KB:
            sys.stderr.write(f"[memguard70] KILL rss_kb={r}\n")
            sys.stderr.flush()
            os._exit(137)
        time.sleep(0.25)


threading.Thread(target=_guard, daemon=True).start()
sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx")

import numpy as np
import mlx.core as mx

from cppmega_mlx.nn._tilelang.mamba3_path_c import (
    mamba3_mimo_apply_with_state_path_c_fwd_path_c_bwd,
    mamba3_mimo_apply_with_state_path_c_fwd_path_b_bwd,
)
from cppmega_mlx.nn._tilelang.mamba3 import mamba3_mimo_bwd_metal

# nam56r config-exact dims. Use seq=128 (2 chunks of 64) + b=1.
b, seq, H, P, N, chunk = 1, 128, 128, 64, 64, 64
rng = np.random.RandomState(0)


def f32(*shape, s=0.1):
    return mx.array((rng.randn(*shape) * s).astype(np.float32))


x = f32(b, seq, H, P)
B = f32(b, seq, H, N)              # per-head (G==H at this surface)
C = f32(b, seq, H, N)
z = f32(b, seq, H, P, s=0.5)
# A per-head-CONSTANT across seq (the chunked kernels' validated regime).
A_head = (-rng.rand(H)).astype(np.float32)        # (H,)
A = mx.array(np.broadcast_to(A_head[None, None, :], (b, seq, H)).copy())
dt = mx.array((rng.rand(b, seq, H) * 0.05).astype(np.float32))
D = mx.array((rng.randn(H)).astype(np.float32))
h0 = f32(b, H, P, N)
cot_y = mx.array((rng.randn(b, seq, H, P) * 0.1).astype(np.float32))

primals = (x, B, C, z, A, dt, D, h0)
prim_names = ["x", "B", "C", "z", "A", "dt", "D", "h0"]
names = ["dx", "dB", "dC", "dz", "dA", "ddt", "dD", "dh0"]


# ---- CHUNKED path (mode under test) backward via vjp(y) ----
def fwd_y_chunked(x, B, C, z, A, dt, D, h0):
    out = mamba3_mimo_apply_with_state_path_c_fwd_path_c_bwd(x, B, C, z, A, dt, D, h0)
    return out[0]


_, grads_chunked = mx.vjp(fwd_y_chunked, primals, (cot_y,))
mx.eval(*grads_chunked)

# ---- prod MSL metal backward (path_c_fwd_path_b_bwd surface) ----
grads_msl = mamba3_mimo_bwd_metal(cot_y, x, B, C, z, A, dt, D, h0, backend="metal")
mx.eval(*grads_msl)

# ---- pure-MLX fp32 GOLD oracle (the step-by-step parity reference) ----
grads_gold = mamba3_mimo_bwd_metal(cot_y, x, B, C, z, A, dt, D, h0, backend="mlx")
mx.eval(*grads_gold)


def maxdiff(a, bb):
    a = np.asarray(a, np.float64)
    bb = np.asarray(bb, np.float64)
    return float(np.abs(a - bb).max())


def relnorm(a, bb):
    a = np.asarray(a, np.float64)
    bb = np.asarray(bb, np.float64)
    den = np.abs(bb).max()
    if den == 0:
        return float(np.abs(a - bb).max())
    return float(np.abs(a - bb).max() / den)


print("\n=== PARITY: path_c_fwd_path_c_bwd vs (MSL metal | MLX fp32 GOLD) ===")
print(f"dims b={b} seq={seq} H={H} P={P} N={N} chunk={chunk}")
worst_vs_gold = 0.0
worst_vs_msl = 0.0
rows = []
for nm, gc, gm, gg in zip(names, grads_chunked, grads_msl, grads_gold):
    gc_np = np.array(gc.astype(mx.float32))
    gm_np = np.array(gm.astype(mx.float32))
    gg_np = np.array(gg.astype(mx.float32))
    d_gold = maxdiff(gc_np, gg_np)
    d_msl = maxdiff(gc_np, gm_np)
    rn_gold = relnorm(gc_np, gg_np)
    # also msl-vs-gold to show the prod path's own fp16 error floor
    d_msl_gold = maxdiff(gm_np, gg_np)
    worst_vs_gold = max(worst_vs_gold, d_gold)
    worst_vs_msl = max(worst_vs_msl, d_msl)
    rows.append((nm, d_gold, rn_gold, d_msl, d_msl_gold))
    print(f"  {nm:4s} |c-gold|={d_gold:.3e} rel={rn_gold:.3e} | "
          f"|c-msl|={d_msl:.3e} | |msl-gold|={d_msl_gold:.3e}")

print(f"\nWORST |chunked - GOLD(mlx fp32)| = {worst_vs_gold:.3e}  (gate < 1e-3)")
print(f"WORST |chunked - MSL(metal)|     = {worst_vs_msl:.3e}")
gold_pass = all(r[1] < 1e-3 for r in rows)
print(f"ALL_8_GRADS_WITHIN_1e-3_vs_GOLD = {gold_pass}")

# ----------------------------------------------------------------------------
# BENCH: end-to-end fwd+bwd wall-clock, median >= 20 iters after warmup.
# ----------------------------------------------------------------------------
ITERS = 30
WARMUP = 5


def bench_chunked():
    def f(x, B, C, z, A, dt, D, h0):
        out = mamba3_mimo_apply_with_state_path_c_fwd_path_c_bwd(x, B, C, z, A, dt, D, h0)
        return out[0]
    times = []
    for i in range(WARMUP + ITERS):
        t0 = time.perf_counter()
        _, g = mx.vjp(f, primals, (cot_y,))
        mx.eval(*g)
        t1 = time.perf_counter()
        if i >= WARMUP:
            times.append(t1 - t0)
    return times


def bench_msl():
    # path_c_fwd_path_b_bwd end-to-end: Path C fwd + MSL metal bwd.
    from cppmega_mlx.nn._tilelang.mamba3_path_c import mamba3_mimo_fwd_path_c
    times = []
    for i in range(WARMUP + ITERS):
        t0 = time.perf_counter()
        y, h_last = mamba3_mimo_fwd_path_c(x, B, C, z, A, dt, D, h0)
        g = mamba3_mimo_bwd_metal(cot_y, x, B, C, z, A, dt, D, h0, backend="metal")
        mx.eval(y, h_last, *g)
        t1 = time.perf_counter()
        if i >= WARMUP:
            times.append(t1 - t0)
    return times


t_chunked = bench_chunked()
t_msl = bench_msl()
med_chunked = float(np.median(t_chunked)) * 1e6  # us
med_msl = float(np.median(t_msl)) * 1e6
print(f"\n=== BENCH end-to-end fwd+bwd (median of {ITERS}, warmup {WARMUP}) ===")
print(f"path_c_fwd_path_c_bwd (chunked): {med_chunked:9.1f} us")
print(f"path_c_fwd_path_b_bwd (MSL)    : {med_msl:9.1f} us")
print(f"ratio chunked/MSL = {med_chunked/med_msl:.3f}x  (<=1.0 => chunked faster/equal)")
print(f"chunked_faster_or_equal = {med_chunked <= med_msl}")

print(f"\nPEAK_RSS_KB={_PEAK} (~{_PEAK/1048576:.3f}GB) memguard70=ON")
print("PARITY_BENCH_DONE")

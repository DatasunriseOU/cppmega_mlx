"""HARD GATE MEASUREMENT for the fused forward-store + in-kernel SIMD P-reduce
single-dispatch fp32 LANE backward (the MSL method realized as ONE TileLang
dispatch).

Measures + asserts, all under memguard 70:
  CORRECTNESS (hard gate): at s4096 ALL 8 grads < 1e-3 vs path-b GOLD
    (mamba3_mimo_bwd_metal backend='mlx'); + s128 + sweep {512,1024,2048}.
  LATENCY: fused single-dispatch us vs the 2-dispatch snapshot-simd LANE vs MSL
    (mamba3_mimo_bwd_metal backend='metal') at s128 + s4096; ratio fused/MSL
    must be <= 1.0x (catch-up minimum -- parity, not 1.19x).
  DETERMINISM: at s4096, 8 repeats of the fused kernel all < 1e-3 vs gold.

RULE #1: no fabrication, no loosened gate. If fused is NOT <= 1.0x MSL or any
s4096 grad >= 1e-3 -> the gate prints FAIL and the caller REVERTS.
"""
from __future__ import annotations
import os, sys, threading, time

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
            sys.stderr.write(f"[memguard70] KILL rss_kb={r} (~{r//1048576}GB)\n")
            sys.stderr.flush(); os._exit(137)
        time.sleep(0.2)
threading.Thread(target=_memguard, daemon=True).start()

os.environ.setdefault("TILELANG_MLX_TVM_FFI_FORCE_COMMAND_BUFFER_BOUNDARY", "1")
os.environ["CPPMEGA_MAMBA3_BWD_SEQ_CHUNK"] = "0"
sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx")

import numpy as np
import mlx.core as mx

from cppmega_mlx.nn._tilelang.mamba3 import mamba3_mimo_bwd_metal
from cppmega_mlx.nn._tilelang.mamba3_path_c import (
    mamba3_mimo_bwd_path_c,
    _mamba3_mimo_bwd_path_c_scratch_simd_kernel,
    _mamba3_mimo_bwd_path_c_simd_kernel,
)

H, P, N = 128, 64, 64
ITERS = int(os.environ.get("BENCH_ITERS", "10"))
WARMUP = int(os.environ.get("BENCH_WARMUP", "3"))
names = ("dx", "dB", "dC", "dz", "dA", "ddt", "dD", "dh0")

def build(seq, seed=0):
    rng = np.random.RandomState(seed)
    b = 1
    def f32(*shape, sc=0.1):
        return mx.array((rng.randn(*shape) * sc).astype(np.float32))
    x = f32(b, seq, H, P); B = f32(b, seq, H, N); C = f32(b, seq, H, N)
    z = f32(b, seq, H, P, sc=0.5)
    A_head = (-rng.rand(H)).astype(np.float32)
    A = mx.array(np.broadcast_to(A_head[None, None, :], (b, seq, H)).copy())
    dt = mx.array((rng.rand(b, seq, H) * 0.05).astype(np.float32))
    D = mx.array((rng.randn(H)).astype(np.float32))
    h0 = f32(b, H, P, N)
    cot_y = mx.array((rng.randn(b, seq, H, P) * 0.1).astype(np.float32))
    return (x, B, C, z, A, dt, D, h0), cot_y

def maxabs(a, ref):
    return float(np.abs(np.asarray(a.astype(mx.float32), np.float64)
                        - np.asarray(ref.astype(mx.float32), np.float64)).max())

def med_us(times):
    return float(np.median(np.asarray(times, np.float64))) * 1e6

def time_fn(fn):
    ts = []
    for i in range(WARMUP + ITERS):
        t0 = time.perf_counter()
        g = fn()
        mx.eval(*g)
        if i >= WARMUP:
            ts.append(time.perf_counter() - t0)
    return med_us(ts)

CORR_SEQS = [int(s) for s in os.environ.get("CORR_SEQS", "128,512,1024,2048,4096").split(",")]
LAT_SEQS = [int(s) for s in os.environ.get("LAT_SEQS", "128,4096").split(",")]

print(f"=== FUSED single-dispatch fp32 LANE bwd GATE (H={H} P={P} N={N}) "
      f"iters={ITERS} warmup={WARMUP} memguard70=ON ===")

# ---- CORRECTNESS sweep (fused vs path-b GOLD) ----
corr = {}
for seq in CORR_SEQS:
    primals, cot_y = build(seq)
    grads_gold = mamba3_mimo_bwd_metal(cot_y, *primals, backend="mlx")
    mx.eval(*grads_gold)
    grads_fused = _mamba3_mimo_bwd_path_c_scratch_simd_kernel(cot_y, *primals)
    mx.eval(*grads_fused)
    per = {nm: maxabs(g, gg) for nm, g, gg in zip(names, grads_fused, grads_gold)}
    worst = max(per.values())
    corr[seq] = (worst, per)
    flag = "PASS" if worst < 1e-3 else "FAIL"
    pk = max(_rss_kb(), _PEAK)
    print(f"[corr seq={seq:5d}] worst={worst:.3e} [{flag}] "
          f"per-grad={ {k: f'{v:.2e}' for k,v in per.items()} } peakRSS~{pk/1048576:.2f}GB")
    del primals, cot_y, grads_gold, grads_fused
    mx.clear_cache()

# ---- DETERMINISM at s4096 ----
print("\n--- DETERMINISM (fused, s4096, 8 repeats vs gold) ---")
primals, cot_y = build(4096)
grads_gold = mamba3_mimo_bwd_metal(cot_y, *primals, backend="mlx")
mx.eval(*grads_gold)
det_pass = 0
for it in range(8):
    gi = _mamba3_mimo_bwd_path_c_scratch_simd_kernel(cot_y, *primals)
    mx.eval(*gi)
    w = max(maxabs(g, gg) for g, gg in zip(gi, grads_gold))
    ok = w < 1e-3
    det_pass += int(ok)
    print(f"  run {it}: worst={w:.3e} {'OK' if ok else 'FAIL'}")
    del gi
print(f"  determinism => {det_pass}/8")
del primals, cot_y, grads_gold
mx.clear_cache()

# ---- LATENCY: fused vs 2-dispatch snapshot-simd vs MSL ----
print("\n--- LATENCY (fused single-dispatch vs 2-dispatch snapshot-simd vs MSL) ---")
lat = {}
for seq in LAT_SEQS:
    primals, cot_y = build(seq)
    # warm compile caches
    _ = _mamba3_mimo_bwd_path_c_scratch_simd_kernel(cot_y, *primals); mx.eval(*_)
    _ = _mamba3_mimo_bwd_path_c_simd_kernel(cot_y, *primals); mx.eval(*_)
    _ = mamba3_mimo_bwd_metal(cot_y, *primals, backend="metal"); mx.eval(*_)
    del _
    f_us = time_fn(lambda: _mamba3_mimo_bwd_path_c_scratch_simd_kernel(cot_y, *primals))
    d2_us = time_fn(lambda: _mamba3_mimo_bwd_path_c_simd_kernel(cot_y, *primals))
    m_us = time_fn(lambda: mamba3_mimo_bwd_metal(cot_y, *primals, backend="metal"))
    ratio_msl = f_us / m_us
    ratio_2d = f_us / d2_us
    lat[seq] = (f_us, d2_us, m_us, ratio_msl, ratio_2d)
    pk = max(_rss_kb(), _PEAK)
    print(f"[lat seq={seq:5d}] fused={f_us:9.1f}us  2disp={d2_us:9.1f}us  MSL={m_us:9.1f}us  "
          f"fused/MSL={ratio_msl:.3f}x  fused/2disp={ratio_2d:.3f}x  peakRSS~{pk/1048576:.2f}GB")
    del primals, cot_y
    mx.clear_cache()

# ---- VERDICT ----
print("\n=== GATE VERDICT ===")
s4096_worst = corr.get(4096, (float('inf'), {}))[0]
s4096_bitcorrect = s4096_worst < 1e-3
all_corr_pass = all(w < 1e-3 for (w, _) in corr.values())
f_s4096, d2_s4096, m_s4096, r_msl_s4096, r_2d_s4096 = lat.get(4096, (0,0,0,float('inf'),float('inf')))
f_s128, d2_s128, m_s128, r_msl_s128, r_2d_s128 = lat.get(128, (0,0,0,float('inf'),float('inf')))
le_msl = r_msl_s4096 <= 1.0
det_ok = det_pass == 8
print(f"  s4096 all-8 worst        = {s4096_worst:.3e}  bit-correct(<1e-3) = {s4096_bitcorrect}")
print(f"  all-corr-seqs pass       = {all_corr_pass} (seqs {CORR_SEQS})")
print(f"  determinism s4096        = {det_pass}/8 => {det_ok}")
print(f"  s128  fused={f_s128:.1f}us MSL={m_s128:.1f}us fused/MSL={r_msl_s128:.3f}x fused/2disp={r_2d_s128:.3f}x")
print(f"  s4096 fused={f_s4096:.1f}us MSL={m_s4096:.1f}us fused/MSL={r_msl_s4096:.3f}x fused/2disp={r_2d_s4096:.3f}x")
print(f"  fused <= 1.0x MSL @s4096  = {le_msl}")
gate = s4096_bitcorrect and all_corr_pass and det_ok and le_msl
print(f"\nGATE => {'PASS' if gate else 'FAIL'}")
print(f"PEAK_RSS_KB={_PEAK} (~{_PEAK/1048576:.3f}GB) memguard70=ON")
print(f"MEASURE_RESULT gate={gate} s4096_bitcorrect={s4096_bitcorrect} "
      f"le_msl={le_msl} det={det_pass}/8 fused_s4096={f_s4096:.1f} msl_s4096={m_s4096:.1f} "
      f"ratio_msl_s4096={r_msl_s4096:.3f}")
print("RC=0")

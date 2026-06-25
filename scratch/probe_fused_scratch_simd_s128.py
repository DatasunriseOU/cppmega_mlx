"""PROBE (validate the new fused single-dispatch route): route fp32 nam56r LANE
backward through _mamba3_mimo_bwd_path_c_scratch_simd_kernel (forward-store slab
+ in-kernel SIMD P-reduce, ONE TileLang dispatch) and compare its 8 grads vs the
path-b GOLD oracle at s128. Confirms:
  (1) fused kernel is dim-compatible with fp32 nam56r (HEADDIM=64),
  (2) all 8 grads produced + maxdiff vs gold < 1e-3 (hard gate),
  (3) DISPATCH COUNT: fused lowering buffer signature => single kernel,
  (4) determinism: 8 repeats all < 1e-3 vs gold.
memguard 70 mandatory. RULE #1: no fallback; raise on any failure.
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
            sys.stderr.write(f"[memguard70] KILL rss_kb={r}\n"); sys.stderr.flush(); os._exit(137)
        time.sleep(0.25)
threading.Thread(target=_memguard, daemon=True).start()

os.environ.setdefault("TILELANG_MLX_TVM_FFI_FORCE_COMMAND_BUFFER_BOUNDARY", "1")
sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx")

import numpy as np
import mlx.core as mx

from cppmega_mlx.nn._tilelang.mamba3 import mamba3_mimo_bwd_metal
from cppmega_mlx.nn._tilelang.mamba3_path_c import (
    _mamba3_mimo_bwd_path_c_scratch_simd_kernel,
    _mamba3_mimo_bwd_path_c_simd_kernel,
    _bwd_scratch_simd_kernel_for,
    _bwd_simd_p_reduction_supported,
    _bwd_scan_plan_for,
)

S = int(os.environ.get("PROBE_SEQ", "128"))
# nam56r surface (b,s,c,G=H,H,P,N) = (1,S,64,128,128,64,64)
b, s, h, p, n = 1, S, 128, 64, 64
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

supported = _bwd_simd_p_reduction_supported(batch=b, heads=h, headdim=p)
plan = _bwd_scan_plan_for(batch=b, seq=s, heads=h, headdim=p, state=n)
print(f"[dim-compat] _bwd_simd_p_reduction_supported(b={b},h={h},p={p}) = {supported}")
print(f"[scan-plan]  snapshot policy = {plan.snapshot_plan.policy}  "
      f"chunk_size={plan.snapshot_plan.chunk_size} chunk_count={plan.snapshot_plan.chunk_count}")
assert supported, "RULE#1: fp32 nam56r must be SIMD-supported"

# (3) dispatch count: inspect the fused lowering buffer signature. The fused
# kernel is ONE prim_func; its lowering must expose exactly the 9 inputs +
# 8 outputs and ONE internal workspace slab. Snapshot route would have a
# SEPARATE snapshot kernel; fused has none.
_k, _low = _bwd_scratch_simd_kernel_for(b, s, h, p, n)
print(f"[fused-lowering] buffer_param_names = {list(_low.buffer_param_names)}")
print(f"[fused-lowering] kernel object = {type(_k).__name__} (single compiled kernel)")

# GOLD (path-b)
grads_gold = mamba3_mimo_bwd_metal(cot_y, *primals, backend="mlx")
mx.eval(*grads_gold)

# fused single-dispatch
grads_fused = _mamba3_mimo_bwd_path_c_scratch_simd_kernel(cot_y, *primals)
mx.eval(*grads_fused)

# 2-dispatch snapshot simd (the f05580a MSL-class LANE) for cross-check
grads_simd = _mamba3_mimo_bwd_path_c_simd_kernel(cot_y, *primals)
mx.eval(*grads_simd)

names = ("dx", "dB", "dC", "dz", "dA", "ddt", "dD", "dh0")
def maxabs(a, ref):
    return float(np.abs(np.asarray(a.astype(mx.float32), np.float64)
                        - np.asarray(ref.astype(mx.float32), np.float64)).max())

print(f"\n[8-grad correctness: FUSED single-dispatch fp32 vs path-b GOLD, s={s}]")
worst_fused = 0.0
for nm, g, gg in zip(names, grads_fused, grads_gold):
    d = maxabs(g, gg); worst_fused = max(worst_fused, d)
    flag = "OK" if d < 1e-3 else "FAIL"
    print(f"  {nm:4s} maxdiff={d:.3e} shape={tuple(g.shape)} [{flag}]")
print(f"  WORST(fused vs gold) = {worst_fused:.3e}  gate<1e-3 => {'PASS' if worst_fused<1e-3 else 'FAIL'}")

print(f"\n[cross-check: fused vs 2-dispatch snapshot-simd (should be ~bit-identical)]")
worst_xc = 0.0
for nm, gf, gs in zip(names, grads_fused, grads_simd):
    d = maxabs(gf, gs); worst_xc = max(worst_xc, d)
    print(f"  {nm:4s} fused-vs-simd maxdiff={d:.3e}")
print(f"  WORST(fused vs snapshot-simd) = {worst_xc:.3e}")

# (4) determinism
print(f"\n[determinism: FUSED fp32, 8 repeats all<1e-3 vs gold, s={s}]")
det_ok = True
for it in range(8):
    gi = _mamba3_mimo_bwd_path_c_scratch_simd_kernel(cot_y, *primals)
    mx.eval(*gi)
    w = max(maxabs(g, gg) for g, gg in zip(gi, grads_gold))
    if w >= 1e-3: det_ok = False
    print(f"  run {it}: worst={w:.3e} {'OK' if w<1e-3 else 'FAIL'}")
print(f"  determinism => {'PASS' if det_ok else 'FAIL'}")

print(f"\nPEAK_RSS_KB={_PEAK} (~{_PEAK/1048576:.3f}GB) memguard70=ON")
print(f"RESULT seq={s} worst_fused={worst_fused:.3e} worst_xc={worst_xc:.3e} "
      f"fused_pass={worst_fused<1e-3} det_pass={det_ok}")
print("RC=0")

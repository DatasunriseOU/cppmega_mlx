"""PERF PROFILE (production MLX tvm_ffi route), nam56r dims, memguard 70.

Splits the path_c_fwd_path_c_bwd cost into measurable pieces on the SAME
production MLX route the .vjp uses (NOT the torch/mps bench the receipt was
measured on). All medians over >=20 iters after warmup. No fabrication.

Pieces measured (each its own warmed median):
  (A) chunked bwd ALONE (B2->B1->B0) via _mamba3_chunked_backward_path_c,
      using PRE-COMPUTED stash (cb/dA_cumsum/prev_states/y) so only the bwd
      kernels are timed. vs
  (A') MSL bwd ALONE mamba3_mimo_bwd_metal(backend='metal').
  (B) forward pieces:
      - lane-scan fwd alone: mamba3_mimo_fwd_path_c (this IS the path_b fwd)
      - F0/F1 chunked-fwd intermediates alone: _mamba3_chunked_fwd_intermediates_path_c
      - full path_c fwd (lane-scan + F0/F1 stash): the custom_function fwd
  (C) e2e fwd+bwd both modes (cross-check vs parity script).

RULE #1: no silent fallback; any kernel that is not eligible RAISES.
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
    mamba3_mimo_fwd_path_c,
    _mamba3_chunked_backward_path_c,
    _mamba3_chunked_fwd_intermediates_path_c,
)
from cppmega_mlx.nn._tilelang.mamba3 import mamba3_mimo_bwd_metal

# nam56r config-exact dims.
b, seq, H, P, N, chunk = 1, 128, 128, 64, 64, 64
rng = np.random.RandomState(0)


def f32(*shape, s=0.1):
    return mx.array((rng.randn(*shape) * s).astype(np.float32))


x = f32(b, seq, H, P)
B = f32(b, seq, H, N)
C = f32(b, seq, H, N)
z = f32(b, seq, H, P, s=0.5)
A_head = (-rng.rand(H)).astype(np.float32)
A = mx.array(np.broadcast_to(A_head[None, None, :], (b, seq, H)).copy())
dt = mx.array((rng.rand(b, seq, H) * 0.05).astype(np.float32))
D = mx.array((rng.randn(H)).astype(np.float32))
h0 = f32(b, H, P, N)
cot_y = mx.array((rng.randn(b, seq, H, P) * 0.1).astype(np.float32))

ITERS = 30
WARMUP = 8


def med_us(times):
    return float(np.median(np.asarray(times, np.float64))) * 1e6


def time_call(fn, label):
    """fn() must return the mx arrays to eval. Median wall us over ITERS."""
    times = []
    for i in range(WARMUP + ITERS):
        t0 = time.perf_counter()
        outs = fn()
        mx.eval(*outs)
        t1 = time.perf_counter()
        if i >= WARMUP:
            times.append(t1 - t0)
    m = med_us(times)
    print(f"  {label:42s} {m:9.1f} us")
    return m


# Pre-compute the stash ONCE (these are forward artifacts, not part of the bwd).
cb, dA_cumsum, prev_states = _mamba3_chunked_fwd_intermediates_path_c(
    x, B, C, A, dt, h0
)
mx.eval(cb, dA_cumsum, prev_states)
y_lane, h_last = mamba3_mimo_fwd_path_c(x, B, C, z, A, dt, D, h0)
mx.eval(y_lane, h_last)
y_stash = mx.contiguous(y_lane.astype(mx.float16))
mx.eval(y_stash)

print("\n=== PERF SPLIT (production MLX tvm_ffi route) ===")
print(f"dims b={b} seq={seq} H={H} P={P} N={N} chunk={chunk}, "
      f"median of {ITERS} (warmup {WARMUP})")

print("\n[1] BACKWARD ALONE (pre-stashed forward artifacts):")


def run_chunked_bwd():
    return _mamba3_chunked_backward_path_c(
        cot_y, x, B, C, z, A, dt, D, h0,
        cb=cb, dA_cumsum=dA_cumsum, prev_states=prev_states, y=y_stash,
    )


def run_msl_bwd():
    return mamba3_mimo_bwd_metal(cot_y, x, B, C, z, A, dt, D, h0, backend="metal")


m_bwd_chunked = time_call(run_chunked_bwd, "chunked bwd B2->B1->B0 (MLX route)")
m_bwd_msl = time_call(run_msl_bwd, "MSL bwd mamba3_mimo_bwd_metal(metal)")
print(f"  -> chunked-bwd-alone / MSL-bwd-alone = {m_bwd_chunked/m_bwd_msl:.3f}x")

print("\n[2] FORWARD pieces:")


def run_lane_fwd():
    return mamba3_mimo_fwd_path_c(x, B, C, z, A, dt, D, h0)


def run_f0f1():
    return _mamba3_chunked_fwd_intermediates_path_c(x, B, C, A, dt, h0)


def run_full_fwd():
    return mamba3_mimo_apply_with_state_path_c_fwd_path_c_bwd(
        x, B, C, z, A, dt, D, h0
    )


m_lane = time_call(run_lane_fwd, "lane-scan fwd (== path_b fwd)")
m_f0f1 = time_call(run_f0f1, "F0/F1 chunked-fwd intermediates")
m_full_fwd = time_call(run_full_fwd, "full path_c fwd (lane + F0/F1 stash)")
print(f"  -> redundant F0/F1 forward overhead = {m_f0f1:.1f} us "
      f"({m_f0f1/m_lane:.2f}x the lane fwd); "
      f"full_fwd/lane = {m_full_fwd/m_lane:.2f}x")

print("\n[3] e2e fwd+bwd (cross-check):")


def run_e2e_chunked():
    def f(x, B, C, z, A, dt, D, h0):
        out = mamba3_mimo_apply_with_state_path_c_fwd_path_c_bwd(
            x, B, C, z, A, dt, D, h0
        )
        return out[0]
    _, g = mx.vjp(f, (x, B, C, z, A, dt, D, h0), (cot_y,))
    return g


def run_e2e_msl():
    y, hl = mamba3_mimo_fwd_path_c(x, B, C, z, A, dt, D, h0)
    g = mamba3_mimo_bwd_metal(cot_y, x, B, C, z, A, dt, D, h0, backend="metal")
    return (y, hl, *g)


m_e2e_chunked = time_call(run_e2e_chunked, "e2e path_c_fwd_path_c_bwd (chunked)")
m_e2e_msl = time_call(run_e2e_msl, "e2e path_c_fwd_path_b_bwd (MSL)")
print(f"  -> e2e chunked/MSL = {m_e2e_chunked/m_e2e_msl:.3f}x")

print("\n=== SUMMARY (us) ===")
print(f"  bwd_chunked_alone   = {m_bwd_chunked:9.1f}")
print(f"  bwd_msl_alone       = {m_bwd_msl:9.1f}")
print(f"  fwd_lane (path_b)   = {m_lane:9.1f}")
print(f"  fwd_f0f1 (extra)    = {m_f0f1:9.1f}")
print(f"  fwd_full (lane+f0f1)= {m_full_fwd:9.1f}")
print(f"  e2e_chunked         = {m_e2e_chunked:9.1f}")
print(f"  e2e_msl             = {m_e2e_msl:9.1f}")
# accounting: e2e_chunked ~ full_fwd + bwd_chunked + vjp/stash glue
acct = m_full_fwd + m_bwd_chunked
print(f"  full_fwd+bwd_chunked= {acct:9.1f}  (e2e glue = {m_e2e_chunked-acct:+.1f})")
# if we removed the redundant F0/F1 AND used chunked bwd on top of MSL-equiv fwd:
ideal_drop_f0f1 = m_e2e_chunked - m_f0f1
print(f"  e2e_chunked - f0f1  = {ideal_drop_f0f1:9.1f}  (drop redundant fwd lever)")

print(f"\nPEAK_RSS_KB={_PEAK} (~{_PEAK/1048576:.3f}GB) memguard70=ON")
print("PERF_SPLIT_DONE")

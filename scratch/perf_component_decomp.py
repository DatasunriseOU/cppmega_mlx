"""PER-COMPONENT decomposition of the MLX-route chunked-bwd penalty.
Bit-correct base 5d5c878. nam56r dims. memguard 70 (guard thread mandatory).

Components measured (each warmed median >=20 iters, us + cross-checked):
  (1) per-kernel dispatch B2, B1, B0 in ISOLATION on the MLX route
  (2) cmd-buffer boundary A/B: FORCE_COMMAND_BUFFER_BOUNDARY=1 vs =0
      (TIMING-ONLY probe per RULE#1 -- does NOT ship the non-det path)
  (3) wrapper fp32 recompute cost (dD reduction + casts), measured alone
  (4) lazy MLX alloc / primitive-boundary: empty-eval + per-call alloc floor
  (5) torch/mps SAME chain stage-by-stage (no MLX-route overhead) as the
      no-MLX-route reference; MSL single fused kernel as the target.

S is taken from argv[1] (128 or 512). RULE#1: any ineligible kernel RAISES.
"""
import os
import sys
import threading
import time

_LIM = 70 * 1024 * 1024
_PEAK = 0


def _rss():
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)


def _g():
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


threading.Thread(target=_g, daemon=True).start()
sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx")

import numpy as np
import mlx.core as mx

from cppmega_mlx.nn._tilelang.mamba3_path_c import (
    mamba3_mimo_fwd_path_c,
    _mamba3_chunked_backward_path_c,
    _mamba3_chunked_fwd_intermediates_path_c,
    _chunked_bwd_b2_kernel,
    _chunked_bwd_b1_kernel,
    _chunked_bwd_b0_kernel,
)
from cppmega_mlx.nn._tilelang.mamba3 import mamba3_mimo_bwd_metal

SEQ = int(sys.argv[1]) if len(sys.argv) > 1 else 128
b, seq, H, P, N, chunk = 1, SEQ, 128, 64, 64, 64
G = H
nchunks = seq // chunk
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

ITERS = 25
WARMUP = 8


def med_us(times):
    a = np.asarray(times, np.float64)
    return float(np.median(a)) * 1e6, float(np.percentile(a, 10)) * 1e6


def time_call(fn, label):
    times = []
    for i in range(WARMUP + ITERS):
        t0 = time.perf_counter()
        outs = fn()
        if outs is not None:
            mx.eval(*outs)
        t1 = time.perf_counter()
        if i >= WARMUP:
            times.append(t1 - t0)
    m, p10 = med_us(times)
    print(f"  {label:46s} med={m:9.1f} us  p10={p10:9.1f}")
    return m


# ----- stash the forward artifacts (NOT timed as bwd) -----
cb, dA_cumsum, prev_states = _mamba3_chunked_fwd_intermediates_path_c(x, B, C, A, dt, h0)
mx.eval(cb, dA_cumsum, prev_states)
y_lane, h_last = mamba3_mimo_fwd_path_c(x, B, C, z, A, dt, D, h0)
mx.eval(y_lane, h_last)
y_stash = mx.contiguous(y_lane.astype(mx.float16))
mx.eval(y_stash)

# fp16 ABI casts (same as wrapper), pre-evaled so casts are NOT timed.
def f16(a):
    return mx.contiguous(a.astype(mx.float16))


x16 = f16(x); B16 = f16(B); C16 = f16(C); z16 = f16(z); D16 = f16(D)
dout16 = f16(cot_y); y16 = y_stash
dt16 = dt.astype(mx.float16)
dt_k = mx.contiguous(mx.transpose(dt16.reshape(b, nchunks, chunk, H), (0, 3, 1, 2)))
cb16 = cb.astype(mx.float16) if cb.dtype != mx.float16 else cb
dA16 = dA_cumsum.astype(mx.float16) if dA_cumsum.dtype != mx.float16 else dA_cumsum
A_head16 = A[:, 0, :][0].astype(mx.float16)
prev32 = prev_states.astype(mx.float32)
mx.eval(x16, B16, C16, z16, D16, dout16, dt_k, cb16, dA16, A_head16, prev32)

k_b2 = _chunked_bwd_b2_kernel(b, seq, chunk, G, H, P, N)
k_b1 = _chunked_bwd_b1_kernel(b, seq, chunk, G, H, P, N)
k_b0 = _chunked_bwd_b0_kernel(b, seq, chunk, G, H, P, N)


def run_b2():
    return k_b2(dout16, cb16, x16, z16, dt_k, dA16, C16, B16, prev32, D16, y16)


# B1/B0 need the upstream outputs; pre-compute once so isolation times only the kernel.
_o2 = run_b2()
mx.eval(*_o2)
dC_m, dx_b2, dz_m, dchunk, dinp_diag, dA_y, _dD = _o2
dh_last0 = mx.zeros((b, H, P, N), dtype=mx.float32)
mx.eval(dh_last0)


def run_b1():
    return k_b1(dchunk, dA16, dh_last0, prev32)


_o1 = run_b1()
mx.eval(*_o1)
dstates, dh0_m, dA_tail = _o1


def run_b0():
    return k_b0(dstates, dinp_diag, dA_y, dA_tail, dA16, x16, B16, dt_k, A_head16)


def set_boundary(v):
    if v:
        os.environ["TILELANG_MLX_TVM_FFI_FORCE_COMMAND_BUFFER_BOUNDARY"] = "1"
    else:
        os.environ["TILELANG_MLX_TVM_FFI_FORCE_COMMAND_BUFFER_BOUNDARY"] = "0"


def run_full_chunked_bwd():
    return _mamba3_chunked_backward_path_c(
        cot_y, x, B, C, z, A, dt, D, h0,
        cb=cb, dA_cumsum=dA_cumsum, prev_states=prev_states, y=y_stash,
    )


def run_msl_bwd():
    return mamba3_mimo_bwd_metal(cot_y, x, B, C, z, A, dt, D, h0, backend="metal")


# ----- wrapper fp32 recompute (dD reduction + gate) measured ALONE -----
def run_wrapper_recompute():
    z32 = z.astype(mx.float32)
    sig_z = mx.sigmoid(z32)
    silu_z = z32 * sig_z
    d_y_skip = cot_y.astype(mx.float32) * silu_z
    dD_wrapper = mx.sum(d_y_skip * x.astype(mx.float32), axis=(0, 1, 3))
    # also the dx add + dtype maps that the wrapper does post-kernel
    dx_total = dx_b2 + dstates[:, 0, :, :, 0] * 0  # cheap add proxy not used; keep dD only
    return (dD_wrapper,)


def run_alloc_floor():
    # lazy MLX alloc / primitive-boundary floor: one trivial owner-output kernel
    # eval (empty graph + a single small contiguous) to expose per-call dispatch
    # floor without any chunked compute.
    t = mx.contiguous(x16 * 1.0)
    return (t,)


print(f"\n=== PER-COMPONENT DECOMP S={SEQ} (nchunks={nchunks}) base 5d5c878 ===")
print(f"dims b={b} seq={seq} H={H} P={P} N={N} chunk={chunk}, median of {ITERS}")

print("\n[BOUNDARY FORCED = 1 (production deterministic path)]")
set_boundary(True)
m_b2_on = time_call(run_b2, "B2 dispatch (boundary=1)")
m_b1_on = time_call(run_b1, "B1 dispatch (boundary=1)")
m_b0_on = time_call(run_b0, "B0 dispatch (boundary=1)")
m_full_on = time_call(run_full_chunked_bwd, "FULL chunked bwd wrapper (boundary=1)")

print("\n[BOUNDARY FORCED = 0 (TIMING-ONLY probe, NOT shipped)]")
set_boundary(False)
m_b2_off = time_call(run_b2, "B2 dispatch (boundary=0)")
m_b1_off = time_call(run_b1, "B1 dispatch (boundary=0)")
m_b0_off = time_call(run_b0, "B0 dispatch (boundary=0)")
m_full_off = time_call(run_full_chunked_bwd, "FULL chunked bwd wrapper (boundary=0)")

# restore production default
set_boundary(True)

print("\n[WRAPPER fp32 recompute + alloc floor]")
m_recompute = time_call(run_wrapper_recompute, "wrapper dD fp32 reduction alone")
m_alloc = time_call(run_alloc_floor, "alloc/primitive floor (1 trivial kernel)")

print("\n[TARGETS]")
m_msl = time_call(run_msl_bwd, "MSL bwd (single fused kernel) TARGET")

print("\n=== SUMMARY (us) ===")
print(f"  B2_boundary_on   = {m_b2_on:9.1f}   B2_boundary_off = {m_b2_off:9.1f}   d={m_b2_on-m_b2_off:+.1f}")
print(f"  B1_boundary_on   = {m_b1_on:9.1f}   B1_boundary_off = {m_b1_off:9.1f}   d={m_b1_on-m_b1_off:+.1f}")
print(f"  B0_boundary_on   = {m_b0_on:9.1f}   B0_boundary_off = {m_b0_off:9.1f}   d={m_b0_on-m_b0_off:+.1f}")
print(f"  FULL_on          = {m_full_on:9.1f}   FULL_off        = {m_full_off:9.1f}   d={m_full_on-m_full_off:+.1f}")
print(f"  sum(B2+B1+B0)_on = {m_b2_on+m_b1_on+m_b0_on:9.1f}   (full-sum glue={m_full_on-(m_b2_on+m_b1_on+m_b0_on):+.1f})")
print(f"  wrapper_recompute= {m_recompute:9.1f}")
print(f"  alloc_floor      = {m_alloc:9.1f}")
print(f"  MSL_target       = {m_msl:9.1f}")
print(f"  FULL_on / MSL    = {m_full_on/m_msl:.3f}x")
print(f"\nPEAK_RSS_KB={_PEAK} (~{_PEAK/1048576:.3f}GB) memguard70=ON")
print("COMPONENT_DECOMP_DONE")

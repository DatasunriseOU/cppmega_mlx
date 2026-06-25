"""SMOKE: PRODUCTION-model F0 input fix (inp = x*B, drop dt) + chunked bwd 8-grads.

Checkpoint 1 (FORWARD correctness of the F0 fix): the F0 summary_states is the
per-chunk state contribution sum_l decay[l]*x[l,p]*B[l,n]. With the dt-on-input
DROPPED it must now match the PRODUCTION reference (sum_l decay*x*B) to ~1e-6, and
DIVERGE from the OLD buggy model (sum_l decay*dt*x*B) by a large gap (~0.33 scale).
decay[l] = exp(dacs[L-1]-dacs[l]) with dacs = inclusive cumsum of A[h]*dt[l].

Checkpoint 2: full chunked B2->B1->B0 backward runs and returns 8 FINITE grads.

memguard 70 mandatory. No fabrication: every number printed is a real measurement.
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
    _chunked_fwd_f0_kernel,
    _assert_per_head_constant_A,
)
from cppmega_mlx.nn._tilelang.mamba3 import mamba3_mimo_bwd_metal

# nam56r config dims (P,N,chunk). small batch + short seq for the smoke.
b, seq, H, P, N, chunk = 1, 128, 128, 64, 64, 64
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

# ---------------------------------------------------------------------------
# CHECKPOINT 1: F0 summary_states (per-chunk state contribution) vs production.
# ---------------------------------------------------------------------------
A_head16 = _assert_per_head_constant_A(A).astype(mx.float16)
x16 = mx.contiguous(x.astype(mx.float16))
B16 = mx.contiguous(B.astype(mx.float16))
C16 = mx.contiguous(C.astype(mx.float16))
dt16 = dt.astype(mx.float16)

k_f0 = _chunked_fwd_f0_kernel(b, seq, chunk, H, H, P, N)
cb, dA_cumsum, summary = k_f0(x16, B16, C16, A_head16, dt16)
mx.eval(cb, dA_cumsum, summary)
summary_np = np.asarray(summary.astype(mx.float32))  # (b, nchunks, H, P, N)

# numpy reference using the SAME fp16-cast inputs (so we isolate the MODEL diff,
# not fp16 rounding). decay = exp2(p*(dacs[L-1]-dacs[l])), p=log2(e).
LOG2E = float(np.log2(np.e))
xf = np.asarray(x16.astype(mx.float32))
Bf = np.asarray(B16.astype(mx.float32))
dtf = np.asarray(dt16.astype(mx.float32))
Af = np.asarray(A_head16.astype(mx.float32))  # (H,)

ref_prod = np.zeros((b, nchunks, H, P, N), np.float64)   # sum_l decay*x*B  (NEW model)
ref_old = np.zeros((b, nchunks, H, P, N), np.float64)    # sum_l decay*dt*x*B (OLD)
for ci in range(nchunks):
    base = ci * chunk
    a = Af[None, :] * dtf[:, base:base + chunk, :]        # (b, L, H) = A*dt
    dacs = np.cumsum(a, axis=1)                            # inclusive cumsum over l
    tail = dacs[:, -1:, :]                                 # (b,1,H)
    decay = np.exp2(LOG2E * (tail - dacs))                 # (b, L, H)
    xb = xf[:, base:base + chunk, :, :, None] * Bf[:, base:base + chunk, :, None, :]  # (b,L,H,P,N)
    dec = decay[:, :, :, None, None]
    ref_prod[:, ci] = np.sum(dec * xb, axis=1)
    ref_old[:, ci] = np.sum(dec * dtf[:, base:base + chunk, :, None, None] * xb, axis=1)

den = np.maximum(np.abs(ref_prod), 1e-6)
delta_prod_abs = float(np.max(np.abs(summary_np - ref_prod)))
delta_prod_rel = float(np.max(np.abs(summary_np - ref_prod) / den))
delta_old_abs = float(np.max(np.abs(ref_prod - ref_old)))   # model gap (NEW vs OLD)
delta_kernel_vs_old = float(np.max(np.abs(summary_np - ref_old)))

print("=== CHECKPOINT 1: F0 summary_states (inp=x*B) vs production reference ===")
print(f"  F0 summary shape={summary_np.shape}")
print(f"  |kernel - PROD(x*B)|     max_abs={delta_prod_abs:.3e}  max_rel={delta_prod_rel:.3e}")
print(f"  |kernel - OLD(dt*x*B)|   max_abs={delta_kernel_vs_old:.3e}  (kernel should NOT match OLD)")
print(f"  |PROD - OLD| model gap   max_abs={delta_old_abs:.3e}  (~scale of the dt baked-in error)")
FWD_OK = delta_prod_abs < 1e-3  # fp16 ABI -> ~1e-3 not 1e-6; report the real number
print(f"  FWD_F0_MATCHES_PROD={FWD_OK}")

# ---------------------------------------------------------------------------
# CHECKPOINT 2: full chunked fwd+bwd -> 8 grads finite. Plus dA/ddt vs GOLD.
# ---------------------------------------------------------------------------
def fwd_y(x, B, C, z, A, dt, D, h0):
    out = mamba3_mimo_apply_with_state_path_c_fwd_path_c_bwd(x, B, C, z, A, dt, D, h0)
    return out[0]


cot_y = mx.array((rng.randn(b, seq, H, P) * 0.1).astype(np.float32))
y_out, grads = mx.vjp(fwd_y, (x, B, C, z, A, dt, D, h0), (cot_y,))
mx.eval(y_out, *grads)

names = ["dx", "dB", "dC", "dz", "dA", "ddt", "dD", "dh0"]
print("\n=== CHECKPOINT 2: chunked B2->B1->B0 backward, 8 grads ===")
all_finite = True
for nm, g in zip(names, grads):
    fin = bool(mx.all(mx.isfinite(g.astype(mx.float32))))
    all_finite = all_finite and fin
    gv = np.asarray(g.astype(mx.float32))
    print(f"  {nm:4s} shape={tuple(g.shape)} dtype={g.dtype} finite={fin} "
          f"absmax={float(np.max(np.abs(gv))):.3e}")
print(f"  ALL_8_GRADS_FINITE={all_finite}  n_grads={len(grads)}")

# Extra sanity: dA/ddt vs GOLD (the dt-remap grads most likely to regress).
gold = mamba3_mimo_bwd_metal(cot_y, x, B, C, z, A, dt, D, h0)
mx.eval(*gold)
gnames = ["dx", "dB", "dC", "dz", "dA", "ddt", "dD", "dh0"]
print("\n=== EXTRA: per-grad |chunked - GOLD(mamba3_mimo_bwd_metal)| (rtol=1e-3/atol=1e-4) ===")
gate_pass = True
for nm, gc, gg in zip(gnames, grads, gold):
    a = np.asarray(gc.astype(mx.float32))
    bb = np.asarray(gg.astype(mx.float32))
    amax = float(np.max(np.abs(a - bb)))
    tol = 1e-4 + 1e-3 * float(np.max(np.abs(bb)))
    ok = amax <= tol
    gate_pass = gate_pass and ok
    print(f"  {nm:4s} max_abs_diff={amax:.3e}  tol={tol:.3e}  PASS={ok}")
print(f"  ALL_8_GRAD_GATE_PASS={gate_pass}")

print(f"\nPEAK_RSS_KB={_PEAK} (~{_PEAK/1048576:.3f}GB) memguard70=ON")
print("SMOKE_RESULT:", "OK" if (FWD_OK and all_finite) else "CHECK")

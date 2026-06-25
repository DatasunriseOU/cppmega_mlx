"""RULE#1 numeric probe (NO production edit): confirm the dz/y_skip bug.

Hypothesis: B2 reads the forward-stashed GATED y = silu(z)*(C.h + D*x) and uses
it as the dgate multiplicand, but the production dz VJP needs the PRE-GATE
y_skip = C.h + D*x. Probe: at nam56r, compute gold dz (path-b), then compute dz
two ways from the SAME stashed forward tensors:
  (A) dz_gated  = dY * y_gated  * silu'(z)   <- current B2 (BUG)
  (B) dz_yskip  = dY * y_skip   * silu'(z)   <- proposed fix
and compare both vs gold. If (B) matches gold and (A) reproduces the 1.087e-1
failure, the diagnosis is confirmed.
"""

import os, sys, threading, time

_LIM = 70 * 1024 * 1024
def _rss():
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
def _guard():
    while True:
        if _rss() > _LIM:
            sys.stderr.write(f"[memguard70] KILL rss_kb={_rss()}\n"); sys.stderr.flush(); os._exit(137)
        time.sleep(0.25)
threading.Thread(target=_guard, daemon=True).start()

sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx")

import mlx.core as mx
import numpy as np

mx.set_default_device(mx.gpu)

# nam56r shape: b=1, seq=128, H=128, P=64, N=64
b, seq, H, P, N = 1, 128, 128, 64, 64
rng = np.random.default_rng(0)


def rn(*shape):
    return mx.array(rng.standard_normal(shape).astype(np.float32) * 0.1)


x = rn(b, seq, H, P)
B = rn(b, seq, H, N)
C = rn(b, seq, H, N)
z = rn(b, seq, H, P)
A = -mx.abs(rn(b, seq, H))  # per-head-ish; bwd asserts per-head constant -> broadcast
dt = mx.abs(rn(b, seq, H)) + 0.5
D = rn(H)
h0 = rn(b, H, P, N)
dy = rn(b, seq, H, P)

# A must be per-head constant for the chunked route assert; make it constant over seq.
A = mx.broadcast_to(A[:, :1, :], (b, seq, H))
A = mx.contiguous(A)

from cppmega_mlx.nn._tilelang.mamba3 import (
    mamba3_mimo_fwd_metal,
    mamba3_mimo_bwd_metal,
)

# Gold path-b backward (the production gate VJP ground truth).
g_dx, g_dB, g_dC, g_dz, g_dA, g_ddt, g_dD, g_dh0 = mamba3_mimo_bwd_metal(
    dy, x, B, C, z, A, dt, D, h0
)
mx.eval(g_dz)

# ---- Recompute the forward recurrence in pure MLX to get y_skip AND y_gated ----
# decay_t = exp(A*dt); h_t = decay*h_{t-1} + x_t (outer) B_t ; y_raw = sum_n C.h ;
# y_skip = y_raw + D*x ; gated y = silu(z)*y_skip.
h = h0  # (b,H,P,N)
y_skip_list = []
for t in range(seq):
    decay = mx.exp(A[:, t, :] * dt[:, t, :])[:, :, None, None]  # (b,H,1,1)
    xt = x[:, t, :, :][:, :, :, None]  # (b,H,P,1)
    Bt = B[:, t, :, :][:, :, None, :]  # (b,H,1,N)
    h = decay * h + xt * Bt
    Ct = C[:, t, :, :][:, :, None, :]  # (b,H,1,N)
    y_raw = mx.sum(h * Ct, axis=-1)  # (b,H,P)
    y_skip = y_raw + D[None, :, None] * x[:, t, :, :]
    y_skip_list.append(y_skip)
y_skip = mx.stack(y_skip_list, axis=1)  # (b,seq,H,P)

sig = mx.sigmoid(z)
silu = z * sig
silu_prime = sig * (1.0 + z * (1.0 - sig))
y_gated = silu * y_skip  # == production forward y stash

# (A) current B2 (BUG): dgate uses GATED y
dz_gated = dy * y_gated * silu_prime
# (B) proposed fix: dgate uses PRE-GATE y_skip
dz_yskip = dy * y_skip * silu_prime

mx.eval(dz_gated, dz_yskip, y_gated, y_skip)


def maxabs(a, c):
    return float(mx.max(mx.abs(a - c)))


print("gold dz                 |abs| max:", float(mx.max(mx.abs(g_dz))))
print("dz_gated (BUG)  vs gold :", maxabs(dz_gated, g_dz))
print("dz_yskip (FIX)  vs gold :", maxabs(dz_yskip, g_dz))
print("y_gated vs y_skip diff  :", maxabs(y_gated, y_skip))
# Also confirm silu reconstruction y_skip = y_gated / silu (where silu!=0)
recon = y_gated / mx.where(mx.abs(silu) < 1e-6, mx.ones_like(silu), silu)
print("recon(y_gated/silu) vs y_skip:", maxabs(recon, y_skip))

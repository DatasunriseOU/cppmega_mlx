"""Standalone parity: B2->B1->B0 Metal kernels vs the MLX backward proto (GOLD).

The MLX proto (scratch/mamba3_chunked_backward_proto.py) is validated to 1.30e-4
vs the serial VJP. This drives BOTH the proto and the Metal B0/B1/B2 chain with
identical fp32 inputs and compares each grad tensor. RULE #1: a mismatch FAILS.

Run: .venv/bin/python scratch/test_b0b1b2_metal_vs_proto.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import mlx.core as mx

from mamba3_chunked_forward_proto import _segsum  # noqa
import mamba3_chunked_backward_proto as bp

from cppmega_mlx.nn._tilelang.mamba3_chunked_precompute_core import (
    build_chunk_precompute_metal, build_inter_chunk_recur_metal,
)
from cppmega_mlx.nn._tilelang.mamba3_chunked_backward_core import (
    build_chunk_scan_combine_bwd_metal, build_inter_chunk_recur_bwd_metal,
    build_chunk_precompute_bwd_metal,
)

torch.manual_seed(0)
np.random.seed(0)
import os
b = 1
S = int(os.environ.get("S", "256"))
chunk = int(os.environ.get("CHUNK", "64"))
G, H, P, N = 1, 2, 64, 16
USE_DHLAST = os.environ.get("DHLAST", "0") == "1"
nchunks = S // chunk
dev = "mps"

# ---- inputs (fp32 numpy, shared) ----
x_np = (np.random.randn(b, S, H, P) * 0.1).astype(np.float32)
B_np = (np.random.randn(b, S, G, N) * 0.1).astype(np.float32)
C_np = (np.random.randn(b, S, G, N) * 0.1).astype(np.float32)
A_np = (-np.random.rand(H)).astype(np.float32)
dt_np = (np.random.rand(b, S, H) * 0.05).astype(np.float32)
D_np = (np.random.randn(H)).astype(np.float32)
h0_np = (np.random.randn(b, H, P, N) * 0.1).astype(np.float32)
dout_np = (np.random.randn(b, S, H, P) * 0.1).astype(np.float32)

# ---- GOLD: MLX proto forward-full + backward ----
# CRITICAL: OUR recurrence is h[t]=decay*h[t-1] + dt[t]*(x outer B). The proto's
# `inp` argument is the FULL per-step input INCLUDING dt: inp = dt*(x outer B).
def mxa(a): return mx.array(a)
z_np = (np.random.randn(b, S, H, P) * 0.5).astype(np.float32)
log_decay = (mxa(A_np).reshape(1, 1, H) * mxa(dt_np)).reshape(b, S, H, 1, 1)
B_h = mx.broadcast_to(mxa(B_np).reshape(b, S, G, N)[:, :, :, None, :], (b, S, G, H // G, N)).reshape(b, S, H, N)
dt_bshp11 = mxa(dt_np)[:, :, :, None, None]
inp = dt_bshp11 * (mxa(x_np)[..., None] * B_h[:, :, :, None, :])   # (b,S,H,P,N), incl dt
C_proto = mx.broadcast_to(mxa(C_np).reshape(b, S, G, N)[:, :, :, None, :], (b, S, G, H // G, N)).reshape(b, S, H, N)
out, final_state, cache = bp.chunked_mamba3_forward_full(
    log_decay, inp, C_proto, mxa(x_np), mxa(z_np),
    mxa(D_np), mxa(h0_np), chunk_size=chunk,
)
dh_last_np = (np.random.randn(b, H, P, N) * 0.1).astype(np.float32) if USE_DHLAST else None
grads = bp.chunked_mamba3_backward(
    mxa(dout_np), cache, dh_last=(mxa(dh_last_np) if USE_DHLAST else None))
mx.eval(out, final_state, grads["log_decay"], grads["inp"], grads["C"],
        grads["x"], grads["z"], grads["D"], grads["h0"])

g_dlog = np.array(grads["log_decay"]).reshape(b, S, H)
g_dinp = np.array(grads["inp"])  # (b,S,H,P,N)
g_dC = np.array(grads["C"])      # (b,S,H,N) (proto returns per-head)
g_dx = np.array(grads["x"])      # (b,S,H,P) skip path only
g_dz = np.array(grads["z"])
g_dD = np.array(grads["D"])      # (H,P) reduced
g_dh0 = np.array(grads["h0"])    # (b,H,P,N)
y_np = np.array(cache["y"])      # (b,S,H,P) pre-gate

# proto's dinp is grad wrt inp = dt*(x outer B). Derive dx_inp / dB / ddt_inp:
#   dx_inp[s,p] = sum_n dinp[s,p,n]*dt[s]*B[s,n]
#   dB[s,n]     = sum_p dinp[s,p,n]*dt[s]*x[s,p]   (per head; group-sum over heads)
#   ddt_inp[s]  = sum_{p,n} dinp[s,p,n]*x[s,p]*B[s,n]
g_dxinp = np.einsum("bshpn,bsh,bsn->bshp", g_dinp, dt_np, B_np[:, :, 0, :])
g_dB_perhead = np.einsum("bshpn,bsh,bshp->bshn", g_dinp, dt_np, x_np)
g_dB = g_dB_perhead.sum(axis=2, keepdims=False)[:, :, None, :]  # group-sum over heads (G=1)
g_ddt_inp = np.einsum("bshpn,bshp,bsn->bsh", g_dinp, x_np, B_np[:, :, 0, :])
g_dlogdt = g_dlog  # = da (proto dlog_decay), decay-chain path only

# ---- Metal forward F0/F1 to get the cache the bwd kernels consume ----
def t(a, dt_=torch.float16): return torch.tensor(a, device=dev, dtype=dt_).contiguous()
k_f0 = build_chunk_precompute_metal(b, S, chunk, G, H, P, N)
cb = torch.zeros(b, nchunks, G, chunk, chunk, device=dev, dtype=torch.float16)
dA_cumsum = torch.zeros(b, H, nchunks, chunk, device=dev, dtype=torch.float16)
summary_states = torch.zeros(b, nchunks, H, P, N, device=dev, dtype=torch.float32)
k_f0(t(x_np), t(B_np), t(C_np), t(A_np), t(dt_np), cb, dA_cumsum, summary_states)
torch.mps.synchronize()
k_f1 = build_inter_chunk_recur_metal(b, S, chunk, G, H, P, N)
prev_states = torch.zeros(b, nchunks, H, P, N, device=dev, dtype=torch.float32)
final_state_m = torch.zeros(b, H, P, N, device=dev, dtype=torch.float32)
k_f1(summary_states.contiguous(), dA_cumsum.contiguous(), t(h0_np, torch.float32),
     prev_states, final_state_m)
torch.mps.synchronize()

from einops import rearrange
dt_k = rearrange(t(dt_np), "b (c s) hh -> b hh c s", c=nchunks).contiguous()

# ---- B2 ----
k_b2 = build_chunk_scan_combine_bwd_metal(b, S, chunk, G, H, P, N)
dC_m = torch.zeros(b, S, H, N, device=dev, dtype=torch.float32)
dx_m = torch.zeros(b, S, H, P, device=dev, dtype=torch.float32)
dz_m = torch.zeros(b, S, H, P, device=dev, dtype=torch.float32)
dchunk_states = torch.zeros(b, nchunks, H, P, N, device=dev, dtype=torch.float32)
dinp_diag = torch.zeros(b, S, H, P, N, device=dev, dtype=torch.float32)
dA_cumsum_y = torch.zeros(b, H, nchunks, chunk, device=dev, dtype=torch.float32)
dD_m = torch.zeros(H, device=dev, dtype=torch.float32)
k_b2(t(dout_np), cb.contiguous(), t(x_np), t(z_np), dt_k, dA_cumsum.contiguous(),
     t(C_np), t(B_np), prev_states.contiguous(), t(D_np), t(y_np),
     dC_m, dx_m, dz_m, dchunk_states, dinp_diag, dA_cumsum_y, dD_m)
torch.mps.synchronize()

# ---- B1 ----
k_b1 = build_inter_chunk_recur_bwd_metal(b, S, chunk, G, H, P, N)
dh_last = (torch.tensor(dh_last_np, device=dev, dtype=torch.float32).contiguous()
           if USE_DHLAST else torch.zeros(b, H, P, N, device=dev, dtype=torch.float32))
dstates = torch.zeros(b, nchunks, H, P, N, device=dev, dtype=torch.float32)
dh0_m = torch.zeros(b, H, P, N, device=dev, dtype=torch.float32)
dA_cumsum_tail = torch.zeros(b, H, nchunks, chunk, device=dev, dtype=torch.float32)
k_b1(dchunk_states.contiguous(), dA_cumsum.contiguous(), dh_last,
     prev_states.contiguous(),
     dstates, dh0_m, dA_cumsum_tail)
torch.mps.synchronize()

# ---- B0 ----
k_b0 = build_chunk_precompute_bwd_metal(b, S, chunk, G, H, P, N)
dx_full = dx_m.clone()  # B0 accumulates the inp-path into the D-skip dx from B2
dB_m = torch.zeros(b, S, H, N, device=dev, dtype=torch.float32)
dlog_decay_m = torch.zeros(b, S, H, device=dev, dtype=torch.float32)
ddt_m = torch.zeros(b, S, H, device=dev, dtype=torch.float32)
k_b0(dstates.contiguous(), dinp_diag.contiguous(), dA_cumsum_y.contiguous(),
     dA_cumsum_tail.contiguous(), dA_cumsum.contiguous(), t(x_np), t(B_np), dt_k, t(A_np),
     dx_full, dB_m, dlog_decay_m, ddt_m)
torch.mps.synchronize()

# ---- compare ----
def cmp(name, got, gold):
    got = np.asarray(got, dtype=np.float64); gold = np.asarray(gold, dtype=np.float64)
    d = float(np.abs(got - gold).max())
    print(f"  {name:14s} max|abs diff| = {d:.3e}  (gold |max|={np.abs(gold).max():.3e})")
    return d

print(f"[B0/B1/B2 vs proto] b={b} S={S} chunk={chunk} H={H} P={P} N={N}")
ds = []
ds.append(cmp("dz", dz_m.float().cpu().numpy(), g_dz))
ds.append(cmp("dx", dx_full.float().cpu().numpy(), g_dx + g_dxinp))
ds.append(cmp("dC", dC_m.float().cpu().numpy(), g_dC))
ds.append(cmp("dB", dB_m.float().cpu().numpy().sum(2, keepdims=False)[:, :, None, :], g_dB))
ds.append(cmp("dlog_decay", dlog_decay_m.float().cpu().numpy(), g_dlogdt))
# OUR-model ddt = decay-chain path (da*A) + inp path (sum dinp*x*B).
g_ddt_full = g_dlogdt * A_np.reshape(1, 1, H) + g_ddt_inp
ds.append(cmp("ddt", ddt_m.float().cpu().numpy(), g_ddt_full))
ds.append(cmp("dh0", dh0_m.float().cpu().numpy(), g_dh0))
ds.append(cmp("dD", dD_m.float().cpu().numpy(), g_dD.sum(-1) if g_dD.ndim == 2 else g_dD))
worst = max(ds)
print(f"WORST = {worst:.3e}  {'PASS' if worst < 1e-3 else 'FAIL'} (tol 1e-3)")
sys.exit(0 if worst < 1e-3 else 1)

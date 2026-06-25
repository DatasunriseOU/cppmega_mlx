"""FULL chained backward (B2->B1->B0) vs the MLX proto, with B2 = §HALFSPLIT
simdgroup-GEMM and B1/B0 = serial. The flag is set ONLY around the B2 build,
then unset (B0 correctly RAISES under the global flag — it has no GEMM path).
This is the true < 1e-3 backward gate for the simdgroup-GEMM B2. memguard 70.
Mirrors tests/test_mamba3_chunked_backward_b0b1b2.py exactly, B2 build excepted.
"""
import os, sys, threading, time
_LIM = 70 * 1024 * 1024
_PEAK = 0
def _rss():
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
def _g():
    global _PEAK
    while True:
        r = _rss()
        if r > _PEAK: _PEAK = r
        if r > _LIM:
            sys.stderr.write(f"[memguard70] KILL rss_kb={r}\n"); os._exit(137)
        time.sleep(0.25)
threading.Thread(target=_g, daemon=True).start()

sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx")
import numpy as np
import torch
from einops import rearrange
import mlx.core as mx
sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx/scratch")
import mamba3_chunked_backward_proto as bp
from cppmega_mlx.nn._tilelang.mamba3_chunked_precompute_core import (
    build_chunk_precompute_metal, build_inter_chunk_recur_metal,
)
from cppmega_mlx.nn._tilelang.mamba3_chunked_backward_core import (
    build_chunk_scan_combine_bwd_metal, build_inter_chunk_recur_bwd_metal,
    build_chunk_precompute_bwd_metal,
)

seqlen = 512
b, chunk, G, H, P, N = 1, 64, 1, 2, 64, 16
nchunks = seqlen // chunk
dev = "mps"
rng = np.random.RandomState(0)
x_np = (rng.randn(b, seqlen, H, P) * 0.1).astype(np.float32)
B_np = (rng.randn(b, seqlen, G, N) * 0.1).astype(np.float32)
C_np = (rng.randn(b, seqlen, G, N) * 0.1).astype(np.float32)
A_np = (-rng.rand(H)).astype(np.float32)
dt_np = (rng.rand(b, seqlen, H) * 0.05).astype(np.float32)
D_np = (rng.randn(H)).astype(np.float32)
h0_np = (rng.randn(b, H, P, N) * 0.1).astype(np.float32)
dout_np = (rng.randn(b, seqlen, H, P) * 0.1).astype(np.float32)
z_np = (rng.randn(b, seqlen, H, P) * 0.5).astype(np.float32)

def mxa(a): return mx.array(a)
log_decay = (mxa(A_np).reshape(1, 1, H) * mxa(dt_np)).reshape(b, seqlen, H, 1, 1)
B_h = mx.broadcast_to(mxa(B_np)[:, :, :, None, :], (b, seqlen, G, H // G, N)).reshape(b, seqlen, H, N)
inp = mxa(dt_np)[:, :, :, None, None] * (mxa(x_np)[..., None] * B_h[:, :, :, None, :])
C_proto = mx.broadcast_to(mxa(C_np)[:, :, :, None, :], (b, seqlen, G, H // G, N)).reshape(b, seqlen, H, N)
out, fs, cache = bp.chunked_mamba3_forward_full(
    log_decay, inp, C_proto, mxa(x_np), mxa(z_np), mxa(D_np), mxa(h0_np), chunk_size=chunk)
grads = bp.chunked_mamba3_backward(mxa(dout_np), cache, dh_last=None)
mx.eval(grads["log_decay"], grads["inp"], grads["C"], grads["x"], grads["z"], grads["D"], grads["h0"])
g_dz = np.array(grads["z"]); g_dC = np.array(grads["C"])
g_dx = np.array(grads["x"]); g_dh0 = np.array(grads["h0"])
g_dlog = np.array(grads["log_decay"]).reshape(b, seqlen, H)
g_dinp = np.array(grads["inp"]); g_dD = np.array(grads["D"])
y_np = np.array(cache["y"])
g_dxinp = np.einsum("bshpn,bsh,bsn->bshp", g_dinp, dt_np, B_np[:, :, 0, :])
g_dB = np.einsum("bshpn,bsh,bshp->bshn", g_dinp, dt_np, x_np).sum(2, keepdims=False)[:, :, None, :]
g_ddt = g_dlog * A_np.reshape(1, 1, H) + np.einsum("bshpn,bshp,bsn->bsh", g_dinp, x_np, B_np[:, :, 0, :])

def th(a, d=torch.float16): return torch.tensor(a, device=dev, dtype=d).contiguous()
k_f0 = build_chunk_precompute_metal(b, seqlen, chunk, G, H, P, N)
cb = torch.zeros(b, nchunks, G, chunk, chunk, device=dev, dtype=torch.float16)
dA = torch.zeros(b, H, nchunks, chunk, device=dev, dtype=torch.float16)
summ = torch.zeros(b, nchunks, H, P, N, device=dev, dtype=torch.float32)
k_f0(th(x_np), th(B_np), th(C_np), th(A_np), th(dt_np), cb, dA, summ); torch.mps.synchronize()
k_f1 = build_inter_chunk_recur_metal(b, seqlen, chunk, G, H, P, N)
prev = torch.zeros(b, nchunks, H, P, N, device=dev, dtype=torch.float32)
fst = torch.zeros(b, H, P, N, device=dev, dtype=torch.float32)
k_f1(summ.contiguous(), dA.contiguous(), th(h0_np, torch.float32), prev, fst); torch.mps.synchronize()
dt_k = rearrange(th(dt_np), "b (c s) hh -> b hh c s", c=nchunks).contiguous()

# ---- B2 with the §HALFSPLIT simdgroup-GEMM (flag ON only here) ----
os.environ["CPPMEGA_PATH_C_METAL_GEMM"] = "1"
k_b2 = build_chunk_scan_combine_bwd_metal(b, seqlen, chunk, G, H, P, N)
os.environ.pop("CPPMEGA_PATH_C_METAL_GEMM", None)  # B0/B1 serial
dC_m = torch.zeros(b, seqlen, H, N, device=dev, dtype=torch.float32)
dx_m = torch.zeros(b, seqlen, H, P, device=dev, dtype=torch.float32)
dz_m = torch.zeros(b, seqlen, H, P, device=dev, dtype=torch.float32)
dchunk = torch.zeros(b, nchunks, H, P, N, device=dev, dtype=torch.float32)
dinp_diag = torch.zeros(b, seqlen, H, P, N, device=dev, dtype=torch.float32)
dA_y = torch.zeros(b, H, nchunks, chunk, device=dev, dtype=torch.float32)
dD_m = torch.zeros(H, device=dev, dtype=torch.float32)
k_b2(th(dout_np), cb.contiguous(), th(x_np), th(z_np), dt_k, dA.contiguous(),
     th(C_np), th(B_np), prev.contiguous(), th(D_np), th(y_np),
     dC_m, dx_m, dz_m, dchunk, dinp_diag, dA_y, dD_m); torch.mps.synchronize()

k_b1 = build_inter_chunk_recur_bwd_metal(b, seqlen, chunk, G, H, P, N)
dh_last = torch.zeros(b, H, P, N, device=dev, dtype=torch.float32)
dstates = torch.zeros(b, nchunks, H, P, N, device=dev, dtype=torch.float32)
dh0_m = torch.zeros(b, H, P, N, device=dev, dtype=torch.float32)
dA_tail = torch.zeros(b, H, nchunks, chunk, device=dev, dtype=torch.float32)
k_b1(dchunk.contiguous(), dA.contiguous(), dh_last, prev.contiguous(),
     dstates, dh0_m, dA_tail); torch.mps.synchronize()

k_b0 = build_chunk_precompute_bwd_metal(b, seqlen, chunk, G, H, P, N)
dx_full = dx_m.clone()
dB_m = torch.zeros(b, seqlen, H, N, device=dev, dtype=torch.float32)
dlog_m = torch.zeros(b, seqlen, H, device=dev, dtype=torch.float32)
ddt_m = torch.zeros(b, seqlen, H, device=dev, dtype=torch.float32)
k_b0(dstates.contiguous(), dinp_diag.contiguous(), dA_y.contiguous(),
     dA_tail.contiguous(), dA.contiguous(), th(x_np), th(B_np), dt_k, th(A_np),
     dx_full, dB_m, dlog_m, ddt_m); torch.mps.synchronize()

def d(got, gold):
    return float(np.abs(np.asarray(got, np.float64) - np.asarray(gold, np.float64)).max())
diffs = {
    "dz": d(dz_m.float().cpu(), g_dz),
    "dx": d(dx_full.float().cpu(), g_dx + g_dxinp),
    "dC": d(dC_m.float().cpu(), g_dC),
    "dB": d(dB_m.float().cpu().numpy().sum(2, keepdims=False)[:, :, None, :], g_dB),
    "dlog_decay": d(dlog_m.float().cpu(), g_dlog),
    "ddt": d(ddt_m.float().cpu(), g_ddt),
    "dh0": d(dh0_m.float().cpu(), g_dh0),
    "dD": d(dD_m.float().cpu(), g_dD.sum(-1) if g_dD.ndim == 2 else g_dD),
}
worst = max(diffs.values())
print("[chained-backward B2=GEMM] per-grad max|abs|: "
      + " ".join(f"{k}={v:.2e}" for k, v in diffs.items()) + f" -> WORST={worst:.3e}")
print(f"GATE_1e-3_PASS={worst < 1e-3}")
print(f"PEAK_RSS_KB={_PEAK} (~{_PEAK/1048576:.3f}GB) memguard70=ON")
print("RC=0")

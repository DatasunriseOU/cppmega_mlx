"""Compare the MLX tvm_ffi chunked-chain (my VJP's route) vs the torch/mps route
stage-by-stage at nam56r dims, SAME inputs. Isolates whether the MLX out_idx
threading (zeroed buffers, chained handoff) diverges from the validated torch
route. memguard 70."""
import os, sys, threading, time
_LIM = 70 * 1024 * 1024; _PEAK = 0
def _rss():
    import resource; return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
def _g():
    global _PEAK
    while True:
        r = _rss(); _PEAK = max(_PEAK, r)
        if r > _LIM: os._exit(137)
        time.sleep(0.25)
threading.Thread(target=_g, daemon=True).start()
sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx")

import numpy as np
import torch
from einops import rearrange
import mlx.core as mx
from cppmega_mlx.nn._tilelang.mamba3_chunked_precompute_core import (
    build_chunk_precompute_metal, build_inter_chunk_recur_metal)
from cppmega_mlx.nn._tilelang.mamba3_chunked_backward_core import (
    build_chunk_scan_combine_bwd_metal, build_inter_chunk_recur_bwd_metal,
    build_chunk_precompute_bwd_metal)
from cppmega_mlx.nn._tilelang.mamba3_path_c import (
    _chunked_fwd_f0_kernel, _chunked_fwd_f1_kernel,
    _chunked_bwd_b2_kernel, _chunked_bwd_b1_kernel, _chunked_bwd_b0_kernel)

b, seqlen, chunk, G, H, P, N = 1, 128, 64, 128, 128, 64, 64
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

# ---------- TORCH/MPS route (validated) ----------
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
# need y from forward proto-equivalent? The torch test used cache["y"]; here build via F-path?
# The B2 needs the forward pre-gate y. Reuse the proto to get y (same as test).
sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx/scratch")
import mamba3_chunked_backward_proto as bp
def mxa(a): return mx.array(a)
log_decay = (mxa(A_np).reshape(1,1,H)*mxa(dt_np)).reshape(b,seqlen,H,1,1)
B_h = mx.broadcast_to(mxa(B_np)[:,:,:,None,:],(b,seqlen,G,H//G,N)).reshape(b,seqlen,H,N)
inp = mxa(dt_np)[:,:,:,None,None]*(mxa(x_np)[...,None]*B_h[:,:,:,None,:])
C_proto = mx.broadcast_to(mxa(C_np)[:,:,:,None,:],(b,seqlen,G,H//G,N)).reshape(b,seqlen,H,N)
_o,_f,cache = bp.chunked_mamba3_forward_full(log_decay, inp, C_proto, mxa(x_np), mxa(z_np), mxa(D_np), mxa(h0_np), chunk_size=chunk)
y_np = np.array(cache["y"])
dt_k = rearrange(th(dt_np), "b (c s) hh -> b hh c s", c=nchunks).contiguous()
k_b2 = build_chunk_scan_combine_bwd_metal(b, seqlen, chunk, G, H, P, N)
dC_t = torch.zeros(b,seqlen,H,N,device=dev,dtype=torch.float32)
dx_t = torch.zeros(b,seqlen,H,P,device=dev,dtype=torch.float32)
dz_t = torch.zeros(b,seqlen,H,P,device=dev,dtype=torch.float32)
dchunk_t = torch.zeros(b,nchunks,H,P,N,device=dev,dtype=torch.float32)
dinp_t = torch.zeros(b,seqlen,H,P,N,device=dev,dtype=torch.float32)
dAy_t = torch.zeros(b,H,nchunks,chunk,device=dev,dtype=torch.float32)
dD_t = torch.zeros(H,device=dev,dtype=torch.float32)
k_b2(th(dout_np),cb.contiguous(),th(x_np),th(z_np),dt_k,dA.contiguous(),th(C_np),th(B_np),prev.contiguous(),th(D_np),th(y_np),dC_t,dx_t,dz_t,dchunk_t,dinp_t,dAy_t,dD_t); torch.mps.synchronize()
k_b1 = build_inter_chunk_recur_bwd_metal(b, seqlen, chunk, G, H, P, N)
dhl = torch.zeros(b,H,P,N,device=dev,dtype=torch.float32)
dstates_t = torch.zeros(b,nchunks,H,P,N,device=dev,dtype=torch.float32)
dh0_t = torch.zeros(b,H,P,N,device=dev,dtype=torch.float32)
dAtail_t = torch.zeros(b,H,nchunks,chunk,device=dev,dtype=torch.float32)
k_b1(dchunk_t.contiguous(),dA.contiguous(),dhl,prev.contiguous(),dstates_t,dh0_t,dAtail_t); torch.mps.synchronize()
k_b0 = build_chunk_precompute_bwd_metal(b, seqlen, chunk, G, H, P, N)
dxfull_t = dx_t.clone()
dB_t = torch.zeros(b,seqlen,H,N,device=dev,dtype=torch.float32)
dlog_t = torch.zeros(b,seqlen,H,device=dev,dtype=torch.float32)
ddt_t = torch.zeros(b,seqlen,H,device=dev,dtype=torch.float32)
k_b0(dstates_t.contiguous(),dinp_t.contiguous(),dAy_t.contiguous(),dAtail_t.contiguous(),dA.contiguous(),th(x_np),th(B_np),dt_k,th(A_np),dxfull_t,dB_t,dlog_t,ddt_t); torch.mps.synchronize()

T = {"cb":cb,"dA":dA,"prev":prev,"y":y_np,"dC":dC_t,"dx_b2":dx_t,"dz":dz_t,"dchunk":dchunk_t,
     "dinp":dinp_t,"dAy":dAy_t,"dD":dD_t,"dstates":dstates_t,"dh0":dh0_t,"dAtail":dAtail_t,
     "dxfull":dxfull_t,"dB":dB_t,"dlog":dlog_t,"ddt":ddt_t}
Tn = {k:(v.float().cpu().numpy() if isinstance(v,torch.Tensor) else np.asarray(v)) for k,v in T.items()}

# ---------- MLX tvm_ffi route (my VJP's exact builders & call order) ----------
def f16(a): return mx.contiguous(a.astype(mx.float16))
x16=f16(mxa(x_np)); B16=f16(mxa(B_np)); C16=f16(mxa(C_np)); z16=f16(mxa(z_np)); D16=f16(mxa(D_np))
dout16=f16(mxa(dout_np)); y16=f16(mxa(y_np)); A16=mxa(A_np).astype(mx.float16)
dt16=mxa(dt_np).astype(mx.float16)
dt_k_mlx=mx.contiguous(mx.transpose(dt16.reshape(b,nchunks,chunk,H),(0,3,1,2)))
mf0=_chunked_fwd_f0_kernel(b,seqlen,chunk,G,H,P,N)
cb_m,dA_m,summ_m=mf0(x16,B16,C16,A16,dt16)
mf1=_chunked_fwd_f1_kernel(b,seqlen,chunk,G,H,P,N)
prev_m,_fst=mf1(summ_m,dA_m,mxa(h0_np).astype(mx.float32))
mb2=_chunked_bwd_b2_kernel(b,seqlen,chunk,G,H,P,N)
cb16=cb_m.astype(mx.float16); dA16=dA_m.astype(mx.float16); prev32=prev_m.astype(mx.float32)
dC_m,dxb2_m,dz_m,dchunk_m,dinp_m,dAy_m,dD_m=mb2(dout16,cb16,x16,z16,dt_k_mlx,dA16,C16,B16,prev32,D16,y16)
mb1=_chunked_bwd_b1_kernel(b,seqlen,chunk,G,H,P,N)
dhl_m=mx.zeros((b,H,P,N),dtype=mx.float32)
dstates_m,dh0_m,dAtail_m=mb1(dchunk_m,dA16,dhl_m,prev32)
mb0=_chunked_bwd_b0_kernel(b,seqlen,chunk,G,H,P,N)
dxb0_m,dB_m,dlog_m,ddt_m=mb0(dstates_m,dinp_m,dAy_m,dAtail_m,dA16,x16,B16,dt_k_mlx,A16)
dxfull_m=dxb2_m+dxb0_m
mx.eval(cb_m,dA_m,prev_m,dC_m,dxb2_m,dz_m,dchunk_m,dinp_m,dAy_m,dD_m,dstates_m,dh0_m,dAtail_m,dxfull_m,dB_m,dlog_m,ddt_m)
Mn={"cb":np.array(cb_m.astype(mx.float32)),"dA":np.array(dA_m.astype(mx.float32)),
    "prev":np.array(prev_m.astype(mx.float32)),"y":np.array(y16.astype(mx.float32)),
    "dC":np.array(dC_m.astype(mx.float32)),"dx_b2":np.array(dxb2_m.astype(mx.float32)),
    "dz":np.array(dz_m.astype(mx.float32)),"dchunk":np.array(dchunk_m.astype(mx.float32)),
    "dinp":np.array(dinp_m.astype(mx.float32)),"dAy":np.array(dAy_m.astype(mx.float32)),
    "dD":np.array(dD_m.astype(mx.float32)),"dstates":np.array(dstates_m.astype(mx.float32)),
    "dh0":np.array(dh0_m.astype(mx.float32)),"dAtail":np.array(dAtail_m.astype(mx.float32)),
    "dxfull":np.array(dxfull_m.astype(mx.float32)),"dB":np.array(dB_m.astype(mx.float32)),
    "dlog":np.array(dlog_m.astype(mx.float32)),"ddt":np.array(ddt_m.astype(mx.float32))}

print(f"[MLX route vs TORCH route] nam56r dims, stage-by-stage max|abs| diff:")
for k in ["cb","dA","prev","dC","dx_b2","dz","dchunk","dinp","dAy","dD","dstates","dh0","dAtail","dxfull","dB","dlog","ddt"]:
    a=Mn[k]; bb=Tn[k]
    if a.shape!=bb.shape:
        print(f"  {k:8s} SHAPE MISMATCH mlx={a.shape} torch={bb.shape}"); continue
    dd=float(np.abs(a.astype(np.float64)-bb.astype(np.float64)).max())
    tag=" <-- DIVERGES" if dd>1e-2 else ""
    print(f"  {k:8s} maxdiff={dd:.3e}  mlx|max|={np.abs(a).max():.3e} torch|max|={np.abs(bb).max():.3e}{tag}")
print(f"PEAK_RSS_KB={_PEAK}")

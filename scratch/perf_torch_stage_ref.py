"""torch/mps SAME B2->B1->B0 chain, stage-by-stage timed (no MLX-route overhead),
to isolate the MLX-route penalty PER kernel. Plus MSL single fused. memguard 70.
base 5d5c878. Uses the torch builders (build_*_bwd_metal) on dev=mps.
"""
import os, sys, threading, time
_LIM = 70 * 1024 * 1024; _PEAK = 0
def _g():
    global _PEAK
    import resource
    while True:
        r = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
        if r > _PEAK: _PEAK = r
        if r > _LIM: os._exit(137)
        time.sleep(0.25)
threading.Thread(target=_g, daemon=True).start()
sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx")
sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx/scratch")
import numpy as np, torch
from einops import rearrange
import mlx.core as mx
import mamba3_chunked_backward_proto as bp
from cppmega_mlx.nn._tilelang.mamba3_chunked_precompute_core import (
    build_chunk_precompute_metal, build_inter_chunk_recur_metal)
from cppmega_mlx.nn._tilelang.mamba3_chunked_backward_core import (
    build_chunk_scan_combine_bwd_metal, build_inter_chunk_recur_bwd_metal,
    build_chunk_precompute_bwd_metal)

SEQ = int(sys.argv[1]) if len(sys.argv) > 1 else 128
b, seqlen, chunk, G, H, P, N = 1, SEQ, 64, 128, 128, 64, 64
nchunks = seqlen // chunk; dev = "mps"
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
def th(a, d=torch.float16): return torch.tensor(a, device=dev, dtype=d).contiguous()
def mxa(a): return mx.array(a)
# forward stash via proto for y
log_decay = (mxa(A_np).reshape(1,1,H)*mxa(dt_np)).reshape(b,seqlen,H,1,1)
B_h = mx.broadcast_to(mxa(B_np)[:,:,:,None,:],(b,seqlen,G,H//G,N)).reshape(b,seqlen,H,N)
inp = mxa(dt_np)[:,:,:,None,None]*(mxa(x_np)[...,None]*B_h[:,:,:,None,:])
C_proto = mx.broadcast_to(mxa(C_np)[:,:,:,None,:],(b,seqlen,G,H//G,N)).reshape(b,seqlen,H,N)
_o,_f,cache = bp.chunked_mamba3_forward_full(log_decay, inp, C_proto, mxa(x_np), mxa(z_np), mxa(D_np), mxa(h0_np), chunk_size=chunk)
y_np = np.array(cache["y"])
# F0/F1
k_f0 = build_chunk_precompute_metal(b, seqlen, chunk, G, H, P, N)
cb = torch.zeros(b,nchunks,G,chunk,chunk,device=dev,dtype=torch.float16)
dA = torch.zeros(b,H,nchunks,chunk,device=dev,dtype=torch.float16)
summ = torch.zeros(b,nchunks,H,P,N,device=dev,dtype=torch.float32)
k_f0(th(x_np),th(B_np),th(C_np),th(A_np),th(dt_np),cb,dA,summ); torch.mps.synchronize()
k_f1 = build_inter_chunk_recur_metal(b, seqlen, chunk, G, H, P, N)
prev = torch.zeros(b,nchunks,H,P,N,device=dev,dtype=torch.float32)
fst = torch.zeros(b,H,P,N,device=dev,dtype=torch.float32)
k_f1(summ.contiguous(),dA.contiguous(),th(h0_np,torch.float32),prev,fst); torch.mps.synchronize()
dt_k = rearrange(th(dt_np),"b (c s) hh -> b hh c s",c=nchunks).contiguous()
k_b2 = build_chunk_scan_combine_bwd_metal(b, seqlen, chunk, G, H, P, N)
k_b1 = build_inter_chunk_recur_bwd_metal(b, seqlen, chunk, G, H, P, N)
k_b0 = build_chunk_precompute_bwd_metal(b, seqlen, chunk, G, H, P, N)

def alloc_b2():
    return (torch.zeros(b,seqlen,H,N,device=dev,dtype=torch.float32),
            torch.zeros(b,seqlen,H,P,device=dev,dtype=torch.float32),
            torch.zeros(b,seqlen,H,P,device=dev,dtype=torch.float32),
            torch.zeros(b,nchunks,H,P,N,device=dev,dtype=torch.float32),
            torch.zeros(b,seqlen,H,P,N,device=dev,dtype=torch.float32),
            torch.zeros(b,H,nchunks,chunk,device=dev,dtype=torch.float32),
            torch.zeros(H,device=dev,dtype=torch.float32))

dC,dx,dz,dchunk,dinp,dAy,dD = alloc_b2()
k_b2(th(dout_np),cb.contiguous(),th(x_np),th(z_np),dt_k,dA.contiguous(),th(C_np),th(B_np),prev.contiguous(),th(D_np),th(y_np),dC,dx,dz,dchunk,dinp,dAy,dD); torch.mps.synchronize()
dhl = torch.zeros(b,H,P,N,device=dev,dtype=torch.float32)
dstates = torch.zeros(b,nchunks,H,P,N,device=dev,dtype=torch.float32)
dh0 = torch.zeros(b,H,P,N,device=dev,dtype=torch.float32)
dAtail = torch.zeros(b,H,nchunks,chunk,device=dev,dtype=torch.float32)
k_b1(dchunk.contiguous(),dA.contiguous(),dhl,prev.contiguous(),dstates,dh0,dAtail); torch.mps.synchronize()

def med_us(ts): return float(np.median(np.asarray(ts,np.float64)))*1e6
def time_b2():
    ts=[]
    for i in range(33):
        oc,ox,oz,odc,odi,oay,odd = alloc_b2()
        torch.mps.synchronize(); t0=time.perf_counter()
        k_b2(th(dout_np),cb.contiguous(),th(x_np),th(z_np),dt_k,dA.contiguous(),th(C_np),th(B_np),prev.contiguous(),th(D_np),th(y_np),oc,ox,oz,odc,odi,oay,odd)
        torch.mps.synchronize();
        if i>=8: ts.append(time.perf_counter()-t0)
    return med_us(ts)
def time_b1():
    ts=[]
    for i in range(33):
        ds=torch.zeros(b,nchunks,H,P,N,device=dev,dtype=torch.float32)
        d0=torch.zeros(b,H,P,N,device=dev,dtype=torch.float32)
        dat=torch.zeros(b,H,nchunks,chunk,device=dev,dtype=torch.float32)
        torch.mps.synchronize(); t0=time.perf_counter()
        k_b1(dchunk.contiguous(),dA.contiguous(),dhl,prev.contiguous(),ds,d0,dat)
        torch.mps.synchronize()
        if i>=8: ts.append(time.perf_counter()-t0)
    return med_us(ts)
def time_b0():
    ts=[]
    for i in range(33):
        dxf=dx.clone(); dB=torch.zeros(b,seqlen,H,N,device=dev,dtype=torch.float32)
        dlog=torch.zeros(b,seqlen,H,device=dev,dtype=torch.float32)
        ddt=torch.zeros(b,seqlen,H,device=dev,dtype=torch.float32)
        torch.mps.synchronize(); t0=time.perf_counter()
        k_b0(dstates.contiguous(),dinp.contiguous(),dAy.contiguous(),dAtail.contiguous(),dA.contiguous(),th(x_np),th(B_np),dt_k,th(A_np),dxf,dB,dlog,ddt)
        torch.mps.synchronize()
        if i>=8: ts.append(time.perf_counter()-t0)
    return med_us(ts)

print(f"\n=== TORCH/MPS stage-by-stage (no MLX route) S={SEQ} nchunks={nchunks} ===")
tb2=time_b2(); tb1=time_b1(); tb0=time_b0()
print(f"  B2 (torch/mps) = {tb2:9.1f} us")
print(f"  B1 (torch/mps) = {tb1:9.1f} us")
print(f"  B0 (torch/mps) = {tb0:9.1f} us")
print(f"  sum B2+B1+B0   = {tb2+tb1+tb0:9.1f} us")
print(f"PEAK_RSS_KB={_PEAK} (~{_PEAK/1048576:.3f}GB) memguard70=ON")
print("TORCH_STAGE_DONE")

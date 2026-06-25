"""Decisive: are MLX tvm_ffi out_idx owner buffers zeroed before atomic_add?
Run B2 twice (compact out_idx) and compare dD/dinp across runs; then run with
explicit pre-zeroed out= buffers and compare to torch gold. memguard 70."""
import os, sys, threading, time
_LIM=70*1024*1024; _P=0
def _r():
    import resource; return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//1024)
def _g():
    global _P
    while True:
        r=_r(); _P=max(_P,r)
        if r>_LIM: os._exit(137)
        time.sleep(0.25)
threading.Thread(target=_g,daemon=True).start()
sys.path.insert(0,"/Volumes/external/sources/cppmega.mlx")
sys.path.insert(0,"/Volumes/external/sources/cppmega.mlx/scratch")
import numpy as np, torch
from einops import rearrange
import mlx.core as mx
import mamba3_chunked_backward_proto as bp
from cppmega_mlx.nn._tilelang.mamba3_chunked_precompute_core import build_chunk_precompute_metal, build_inter_chunk_recur_metal
from cppmega_mlx.nn._tilelang.mamba3_chunked_backward_core import build_chunk_scan_combine_bwd_metal
from cppmega_mlx.nn._tilelang.mamba3_path_c import _chunked_fwd_f0_kernel, _chunked_fwd_f1_kernel, _chunked_bwd_b2_kernel

b,seqlen,chunk,G,H,P,N=1,128,64,128,128,64,64
nchunks=seqlen//chunk; dev="mps"
rng=np.random.RandomState(0)
x_np=(rng.randn(b,seqlen,H,P)*0.1).astype(np.float32)
B_np=(rng.randn(b,seqlen,G,N)*0.1).astype(np.float32)
C_np=(rng.randn(b,seqlen,G,N)*0.1).astype(np.float32)
A_np=(-rng.rand(H)).astype(np.float32)
dt_np=(rng.rand(b,seqlen,H)*0.05).astype(np.float32)
D_np=(rng.randn(H)).astype(np.float32)
h0_np=(rng.randn(b,H,P,N)*0.1).astype(np.float32)
dout_np=(rng.randn(b,seqlen,H,P)*0.1).astype(np.float32)
z_np=(rng.randn(b,seqlen,H,P)*0.5).astype(np.float32)
def mxa(a): return mx.array(a)
log_decay=(mxa(A_np).reshape(1,1,H)*mxa(dt_np)).reshape(b,seqlen,H,1,1)
B_h=mx.broadcast_to(mxa(B_np)[:,:,:,None,:],(b,seqlen,G,H//G,N)).reshape(b,seqlen,H,N)
inp=mxa(dt_np)[:,:,:,None,None]*(mxa(x_np)[...,None]*B_h[:,:,:,None,:])
C_proto=mx.broadcast_to(mxa(C_np)[:,:,:,None,:],(b,seqlen,G,H//G,N)).reshape(b,seqlen,H,N)
_o,_f,cache=bp.chunked_mamba3_forward_full(log_decay,inp,C_proto,mxa(x_np),mxa(z_np),mxa(D_np),mxa(h0_np),chunk_size=chunk)
y_np=np.array(cache["y"])

# torch gold for dD/dinp
def th(a,d=torch.float16): return torch.tensor(a,device=dev,dtype=d).contiguous()
k_f0=build_chunk_precompute_metal(b,seqlen,chunk,G,H,P,N)
cb=torch.zeros(b,nchunks,G,chunk,chunk,device=dev,dtype=torch.float16)
dA=torch.zeros(b,H,nchunks,chunk,device=dev,dtype=torch.float16)
summ=torch.zeros(b,nchunks,H,P,N,device=dev,dtype=torch.float32)
k_f0(th(x_np),th(B_np),th(C_np),th(A_np),th(dt_np),cb,dA,summ); torch.mps.synchronize()
k_f1=build_inter_chunk_recur_metal(b,seqlen,chunk,G,H,P,N)
prev=torch.zeros(b,nchunks,H,P,N,device=dev,dtype=torch.float32)
fst=torch.zeros(b,H,P,N,device=dev,dtype=torch.float32)
k_f1(summ.contiguous(),dA.contiguous(),th(h0_np,torch.float32),prev,fst); torch.mps.synchronize()
dt_k=rearrange(th(dt_np),"b (c s) hh -> b hh c s",c=nchunks).contiguous()
k_b2=build_chunk_scan_combine_bwd_metal(b,seqlen,chunk,G,H,P,N)
dC_t=torch.zeros(b,seqlen,H,N,device=dev,dtype=torch.float32); dx_t=torch.zeros(b,seqlen,H,P,device=dev,dtype=torch.float32)
dz_t=torch.zeros(b,seqlen,H,P,device=dev,dtype=torch.float32); dchunk_t=torch.zeros(b,nchunks,H,P,N,device=dev,dtype=torch.float32)
dinp_t=torch.zeros(b,seqlen,H,P,N,device=dev,dtype=torch.float32); dAy_t=torch.zeros(b,H,nchunks,chunk,device=dev,dtype=torch.float32)
dD_t=torch.zeros(H,device=dev,dtype=torch.float32)
k_b2(th(dout_np),cb.contiguous(),th(x_np),th(z_np),dt_k,dA.contiguous(),th(C_np),th(B_np),prev.contiguous(),th(D_np),th(y_np),dC_t,dx_t,dz_t,dchunk_t,dinp_t,dAy_t,dD_t); torch.mps.synchronize()
dD_gold=dD_t.float().cpu().numpy(); dinp_gold=dinp_t.float().cpu().numpy()

# MLX route inputs
def f16(a): return mx.contiguous(a.astype(mx.float16))
x16=f16(mxa(x_np)); B16=f16(mxa(B_np)); C16=f16(mxa(C_np)); z16=f16(mxa(z_np)); D16=f16(mxa(D_np))
dout16=f16(mxa(dout_np)); y16=f16(mxa(y_np)); A16=mxa(A_np).astype(mx.float16); dt16=mxa(dt_np).astype(mx.float16)
dt_k_mlx=mx.contiguous(mx.transpose(dt16.reshape(b,nchunks,chunk,H),(0,3,1,2)))
mf0=_chunked_fwd_f0_kernel(b,seqlen,chunk,G,H,P,N); cb_m,dA_m,summ_m=mf0(x16,B16,C16,A16,dt16)
mf1=_chunked_fwd_f1_kernel(b,seqlen,chunk,G,H,P,N); prev_m,_=mf1(summ_m,dA_m,mxa(h0_np).astype(mx.float32))
cb16=cb_m.astype(mx.float16); dA16=dA_m.astype(mx.float16); prev32=prev_m.astype(mx.float32)
mb2=_chunked_bwd_b2_kernel(b,seqlen,chunk,G,H,P,N)

def runc():
    o=mb2(dout16,cb16,x16,z16,dt_k_mlx,dA16,C16,B16,prev32,D16,y16)
    mx.eval(*o); return o
o1=runc(); o2=runc()
dD1=np.array(o1[6].astype(mx.float32)); dD2=np.array(o2[6].astype(mx.float32))
dinp1=np.array(o1[4].astype(mx.float32)); dinp2=np.array(o2[4].astype(mx.float32))
print(f"[compact out_idx] dD run1-vs-run2 maxdiff={np.abs(dD1-dD2).max():.3e} (nonzero => GARBAGE owner buf)")
print(f"[compact out_idx] dinp run1-vs-run2 maxdiff={np.abs(dinp1-dinp2).max():.3e}")
print(f"[compact out_idx] dD-vs-gold={np.abs(dD1-dD_gold).max():.3e}  dinp-vs-gold={np.abs(dinp1-dinp_gold).max():.3e}")

# explicit pre-zeroed out= buffers
outs=[mx.zeros((b,seqlen,G,N),mx.float32),mx.zeros((b,seqlen,H,P),mx.float32),mx.zeros((b,seqlen,H,P),mx.float32),
      mx.zeros((b,nchunks,H,P,N),mx.float32),mx.zeros((b,seqlen,H,P,N),mx.float32),mx.zeros((b,H,nchunks,chunk),mx.float32),
      mx.zeros((H,),mx.float32)]
try:
    oz=mb2(dout16,cb16,x16,z16,dt_k_mlx,dA16,C16,B16,prev32,D16,y16,out=outs)
    mx.eval(*(oz if isinstance(oz,(list,tuple)) else [oz]))
    dDz=np.array(outs[6].astype(mx.float32)); dinpz=np.array(outs[4].astype(mx.float32))
    print(f"[explicit zeroed out=] dD-vs-gold={np.abs(dDz-dD_gold).max():.3e}  dinp-vs-gold={np.abs(dinpz-dinp_gold).max():.3e}")
except Exception as e:
    print(f"[explicit out=] FAILED: {type(e).__name__}: {e}")
print(f"PEAK_RSS_KB={_P}")

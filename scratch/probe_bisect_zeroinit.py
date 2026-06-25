"""PROBE/BISECT: vary the fused B2+B1 kernel's zero_init_output_positions and
measure dstates/dh0 determinism vs a pure-MLX B1 reference over repeats. memguard 70.

Variants (zero_init set, ABSOLUTE output ordinals among out_idx; out_idx order =
[dC,dx,dz,dchunk,dinp,dA_y,dD,dstates,dh0,dA_tail] -> ordinals 0..9):
  A: [6]            (PRODUCTION: dD only)               -> expect RACE
  B: []             (pin NOTHING, kernel owns all init) -> test
  C: [6,7,8]        (dD + dstates + dh0 blitted)        -> test
  D: [6,9]          (dD + dA_tail)                      -> test
The B1 grid stores dstates/dh0 as FULL single-writer; if blitting them removes the
race the bug is a missing blit; if SUPPRESSING the blit removes it, the bridge is
auto-zeroing (atomic-detected) and racing the kernel writes.
"""
from __future__ import annotations
import os, sys, threading, time
_MEMGUARD_LIMIT_KB = 70 * 1024 * 1024
_PEAK = 0
def _rss_kb():
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
def _guard():
    global _PEAK
    while True:
        r = _rss_kb()
        if r > _PEAK: _PEAK = r
        if r > _MEMGUARD_LIMIT_KB:
            sys.stderr.write(f"[memguard70] KILL rss_kb={r}\n"); sys.stderr.flush(); os._exit(137)
        time.sleep(0.25)
threading.Thread(target=_guard, daemon=True).start()
sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx")
sys.path.insert(0, "/Volumes/external/sources/tilelang")
import numpy as np
import mlx.core as mx
import tilelang
from cppmega_mlx.runtime.path_c_backward_fusion_search import (
    _build_eval_inputs, _maxabs, splice_prims, _SHARED_BUFFER_NAMES)
from cppmega_mlx.nn._tilelang.mamba3_path_c import (
    _force_chunked_command_buffer_boundary, _assert_per_head_constant_A, _CHUNKED_METAL_TARGET)
from cppmega_mlx.nn._tilelang.mamba3_chunked_backward_core import (
    chunk_scan_combine_bwd_metal_prim, inter_chunk_recur_bwd_metal_prim)

_force_chunked_command_buffer_boundary()
S=int(os.environ.get("PROBE_SEQ","4096")); H=int(os.environ.get("PROBE_H","128")); N=int(os.environ.get("PROBE_N","64"))
REPEATS=int(os.environ.get("PROBE_REPEATS","10"))
ZSET=os.environ.get("ZSET","6")  # comma list or empty
zinit=[int(t) for t in ZSET.split(",") if t.strip()!=""]
dims=(1,S,64,H,H,64,N); batch,heads,headdim,state=1,H,64,N; chunk=64; nchunks=S//chunk; G=H; L=chunk
inp=_build_eval_inputs(dims)
x,B,C,z,A,dt,D,h0=inp["primals"]; dy=inp["dy"]
cb=inp["cb"]; dA_cumsum=inp["dA_cumsum"]; prev_states=inp["prev_states"]; y=inp["y"]
def f16(a): return mx.contiguous(a.astype(mx.float16))
x16=f16(x);B16=f16(B);C16=f16(C);z16=f16(z);D16=f16(D);dout16=f16(dy);y16=f16(y)
dt16=dt.astype(mx.float16); dt_k=mx.contiguous(mx.transpose(dt16.reshape(batch,nchunks,chunk,heads),(0,3,1,2)))
cb16=cb.astype(mx.float16);dA16=dA_cumsum.astype(mx.float16);A_head16=_assert_per_head_constant_A(A).astype(mx.float16);prev32=prev_states.astype(mx.float32)

b2=chunk_scan_combine_bwd_metal_prim(batch,S,chunk,G,heads,headdim,state)
b1=inter_chunk_recur_bwd_metal_prim(batch,S,chunk,G,heads,headdim,state)
out_idx=[11,12,13,14,15,16,17,19,20,21]
fused=splice_prims([b2,b1],_SHARED_BUFFER_NAMES,"b2b1_bisect",zero_init_output_positions=zinit)
k=tilelang.compile(fused,target=_CHUNKED_METAL_TARGET,execution_backend="tvm_ffi",out_idx=out_idx)
dh_last=mx.zeros((batch,heads,headdim,state),dtype=mx.float32)
tail=dA16[:,:,:,L-1].astype(mx.float32);decay=mx.exp(tail)
def b1_ref(dchunk):
    dch=dchunk.astype(mx.float32); g=mx.zeros((batch,heads,headdim,state),dtype=mx.float32)
    ds=[None]*nchunks
    for cc in range(nchunks-1,-1,-1):
        ds[cc]=g; g=decay[:,:,cc][:,:,None,None]*g+dch[:,cc]
    return mx.stack(ds,axis=1),g
print(f"# ZSET={zinit} s={S} H={H} N={N} nchunks={nchunks} repeats={REPEATS}")
worst_ds=0.0;worst_dh0=0.0;nfail=0
for r in range(REPEATS):
    out=k(dout16,cb16,x16,z16,dt_k,dA16,C16,B16,prev32,D16,y16,dh_last)
    dstates=out[7]; dh0_m=out[8]; dchunk=out[3]
    mx.eval(dstates,dh0_m,dchunk)
    rds,rdh0=b1_ref(dchunk); mx.eval(rds,rdh0)
    dds=_maxabs(dstates.astype(mx.float32),rds); ddh0=_maxabs(dh0_m.astype(mx.float32),rdh0)
    worst_ds=max(worst_ds,dds);worst_dh0=max(worst_dh0,ddh0)
    if dds>1e-3 or ddh0>1e-3: nfail+=1
    print(f"  run {r:2d}: dstates={dds:.3e} dh0={ddh0:.3e}",flush=True)
print(f"SUMMARY ZSET={zinit}: worst_dstates={worst_ds:.3e} worst_dh0={worst_dh0:.3e} fails={nfail}/{REPEATS} "
      f"{'CLEAN' if nfail==0 else 'RACE'}")

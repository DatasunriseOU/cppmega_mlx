"""PROBE/BISECT end-to-end: replace the fused B2+B1 path with the SEPARATE
3-dispatch path (B2 -> standalone []-pinned B1 -> B0), then run the full chunked
backward vs path-b GOLD across REPEATS in-process. memguard 70. NOT a production edit.

This isolates the SPLICE as the race source: if the unfused path is CLEAN and the
fused path RACES (same prims, same fp16 casts), the bug is the splice's zero-init /
command-buffer ordering, NOT any fp16 narrowing.
"""
from __future__ import annotations
import os, sys, threading, time
_MEMGUARD_LIMIT_KB=70*1024*1024;_PEAK=0
def _rss_kb():
    import resource;return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//1024)
def _guard():
    global _PEAK
    while True:
        r=_rss_kb()
        if r>_PEAK:_PEAK=r
        if r>_MEMGUARD_LIMIT_KB:
            sys.stderr.write(f"[memguard70] KILL rss_kb={r}\n");sys.stderr.flush();os._exit(137)
        time.sleep(0.25)
threading.Thread(target=_guard,daemon=True).start()
sys.path.insert(0,"/Volumes/external/sources/cppmega.mlx");sys.path.insert(0,"/Volumes/external/sources/tilelang")
import numpy as np
import mlx.core as mx
import cppmega_mlx.nn._tilelang.mamba3_path_c as M
from cppmega_mlx.runtime.path_c_backward_fusion_search import _build_eval_inputs,_maxabs,_GRAD_NAMES

S=int(os.environ.get("PROBE_SEQ","4096"));H=int(os.environ.get("PROBE_H","128"));N=int(os.environ.get("PROBE_N","64"))
REPEATS=int(os.environ.get("PROBE_REPEATS","12"))

# Build a SEPARATE-DISPATCH replacement for the fused kernel: run B2 alone, then the
# standalone []-pinned B1 alone, returning the SAME 10-tuple the wrapper unpacks.
def unfused_b2b1(b,s,c,g,h,p,n):
    k_b2=M._chunked_bwd_b2_kernel(b,s,c,g,h,p,n)
    k_b1=M._chunked_bwd_b1_kernel(b,s,c,g,h,p,n)
    def call(dout16,cb16,x16,z16,dt_k,dA16,C16,B16,prev32,D16,y16,dh_last):
        dC_m,dx_b2,dz_m,dchunk,dinp_diag,dA_y,dD_m=k_b2(dout16,cb16,x16,z16,dt_k,dA16,C16,B16,prev32,D16,y16)
        mx.eval(dchunk)  # force B2 completion before B1 reads dchunk (separate dispatch)
        dstates,dh0_m,dA_tail_k=k_b1(dchunk,dA16,dh_last,prev32)
        return (dC_m,dx_b2,dz_m,dchunk,dinp_diag,dA_y,dD_m,dstates,dh0_m,dA_tail_k)
    return call
M._chunked_bwd_b2b1_fused_kernel=unfused_b2b1

M._force_chunked_command_buffer_boundary()
dims=(1,S,64,H,H,64,N)
inp=_build_eval_inputs(dims)
print(f"# UNFUSED 3-dispatch s={S} H={H} N={N} nchunks={S//64} repeats={REPEATS}")
worst_overall=0.0;nfail=0
for r in range(REPEATS):
    g=M._mamba3_chunked_backward_path_c(inp["dy"],*inp["primals"],cb=inp["cb"],dA_cumsum=inp["dA_cumsum"],prev_states=inp["prev_states"],y=inp["y"])
    mx.eval(*g)
    diffs={nm:_maxabs(gc,gg) for nm,gc,gg in zip(_GRAD_NAMES,g,inp["grads_gold"])}
    w=max(diffs.values());worst_overall=max(worst_overall,w)
    if w>=1e-3:nfail+=1
    print(f"  run {r:2d}: WORST={w:.3e} {'PASS' if w<1e-3 else 'FAIL'}",flush=True)
print(f"SUMMARY UNFUSED: worst={worst_overall:.3e} fails={nfail}/{REPEATS} {'CLEAN' if nfail==0 else 'RACE'}")

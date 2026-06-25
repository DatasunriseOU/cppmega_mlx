"""PROBE/BISECT end-to-end: monkeypatch the PRODUCTION fused B2+B1 kernel builder
to vary zero_init_output_positions, then run the FULL _mamba3_chunked_backward_path_c
vs path-b GOLD across REPEATS in-process. memguard 70. NOT a production edit.

ZSET selects the fused zero-init ordinals (out_idx order = dC,dx,dz,dchunk,dinp,dA_y,dD,dstates,dh0,dA_tail = 0..9):
  6      : PRODUCTION (dD only)
  (empty): pin nothing
  6,7,8  : dD + dstates + dh0
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
import tilelang
import cppmega_mlx.nn._tilelang.mamba3_path_c as M
from cppmega_mlx.runtime.path_c_backward_fusion_search import (_build_eval_inputs,_maxabs,_GRAD_NAMES,splice_prims,_SHARED_BUFFER_NAMES)
from cppmega_mlx.nn._tilelang.mamba3_chunked_backward_core import (chunk_scan_combine_bwd_metal_prim,inter_chunk_recur_bwd_metal_prim)

S=int(os.environ.get("PROBE_SEQ","4096"));H=int(os.environ.get("PROBE_H","128"));N=int(os.environ.get("PROBE_N","64"))
REPEATS=int(os.environ.get("PROBE_REPEATS","10"))
ZSET=os.environ.get("ZSET","6");zinit=[int(t) for t in ZSET.split(",") if t.strip()!=""]

# monkeypatch the fused builder to use OUR zinit
def patched_fused(b,s,c,g,h,p,n):
    b2=chunk_scan_combine_bwd_metal_prim(b,s,c,g,h,p,n)
    b1=inter_chunk_recur_bwd_metal_prim(b,s,c,g,h,p,n)
    out_idx=[11,12,13,14,15,16,17,19,20,21]
    fused=splice_prims([b2,b1],_SHARED_BUFFER_NAMES,"b2b1_fused_live",zero_init_output_positions=zinit)
    return tilelang.compile(fused,target=M._CHUNKED_METAL_TARGET,execution_backend="tvm_ffi",out_idx=out_idx)
M._chunked_bwd_b2b1_fused_kernel=patched_fused

M._force_chunked_command_buffer_boundary()
dims=(1,S,64,H,H,64,N)
inp=_build_eval_inputs(dims)
print(f"# ZSET={zinit} s={S} H={H} N={N} nchunks={S//64} repeats={REPEATS}")
worst_overall=0.0;nfail=0
for r in range(REPEATS):
    g=M._mamba3_chunked_backward_path_c(inp["dy"],*inp["primals"],cb=inp["cb"],dA_cumsum=inp["dA_cumsum"],prev_states=inp["prev_states"],y=inp["y"])
    mx.eval(*g)
    diffs={nm:_maxabs(gc,gg) for nm,gc,gg in zip(_GRAD_NAMES,g,inp["grads_gold"])}
    w=max(diffs.values());worst_overall=max(worst_overall,w)
    if w>=1e-3:nfail+=1
    print(f"  run {r:2d}: WORST={w:.3e} {'PASS' if w<1e-3 else 'FAIL'}  "+" ".join(f"{nm}={diffs[nm]:.2e}" for nm in _GRAD_NAMES),flush=True)
print(f"SUMMARY ZSET={zinit}: worst={worst_overall:.3e} fails={nfail}/{REPEATS} {'CLEAN' if nfail==0 else 'RACE'}")

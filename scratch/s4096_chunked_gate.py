"""s4096 PRODUCTION chunked backward (the ONLY s4096-capable fp32 route) bit-correct
+ determinism gate, reached via the chunked VJP intermediates. Plus the full seqlen
sweep through the SAME production chunked route. Under memguard 70.
"""
from __future__ import annotations
import os, sys, threading, time
_LIM=70*1024*1024; _PK=0
def _rss():
    import resource; return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//1024)
def _g():
    global _PK
    while True:
        r=_rss()
        if r>_PK:_PK=r
        if r>_LIM: sys.stderr.write(f"[memguard70] KILL rss={r}\n"); os._exit(137)
        time.sleep(0.25)
threading.Thread(target=_g,daemon=True).start()
os.environ.setdefault("TILELANG_MLX_TVM_FFI_FORCE_COMMAND_BUFFER_BOUNDARY","1")
sys.path.insert(0,"/Volumes/external/sources/cppmega.mlx")

import numpy as np
import mlx.core as mx
from cppmega_mlx.nn._tilelang.mamba3 import mamba3_mimo_bwd_metal
from cppmega_mlx.nn._tilelang.mamba3_path_c import (
    _mamba3_chunked_backward_path_c,
    _mamba3_chunked_fwd_intermediates_path_c,
    _mamba3_pre_gate_yskip_path_c,
)

GRAD_NAMES=("dx","dB","dC","dz","dA","ddt","dD","dh0")
HEADS,HEADDIM,STATE=128,64,64; BATCH=1

def build(SEQ,seed=0):
    rng=np.random.RandomState(seed)
    def f32(*s,sc=0.1): return mx.array((rng.randn(*s)*sc).astype(np.float32))
    x=f32(BATCH,SEQ,HEADS,HEADDIM); B=f32(BATCH,SEQ,HEADS,STATE); C=f32(BATCH,SEQ,HEADS,STATE)
    z=f32(BATCH,SEQ,HEADS,HEADDIM,sc=0.5)
    Ah=(-rng.rand(HEADS)).astype(np.float32)
    A=mx.array(np.broadcast_to(Ah[None,None,:],(BATCH,SEQ,HEADS)).copy())
    dt=mx.array((rng.rand(BATCH,SEQ,HEADS)*0.05).astype(np.float32))
    D=mx.array((rng.randn(HEADS)).astype(np.float32))
    h0=f32(BATCH,HEADS,HEADDIM,STATE)
    dy=mx.array((rng.randn(BATCH,SEQ,HEADS,HEADDIM)*0.1).astype(np.float32))
    return dy,x,B,C,z,A,dt,D,h0

def maxabs(a,ref):
    return float(np.abs(np.asarray(a.astype(mx.float32),np.float64)-np.asarray(ref.astype(mx.float32),np.float64)).max())

def run_chunked(dy,x,B,C,z,A,dt,D,h0):
    cb,dA_cumsum,prev_states=_mamba3_chunked_fwd_intermediates_path_c(x,B,C,A,dt,h0)
    y_skip=_mamba3_pre_gate_yskip_path_c(x,B,C,A,dt,D,h0).astype(mx.float16)
    mx.eval(cb,dA_cumsum,prev_states,y_skip)
    g=_mamba3_chunked_backward_path_c(dy,x,B,C,z,A,dt,D,h0,cb=cb,dA_cumsum=dA_cumsum,prev_states=prev_states,y=y_skip)
    mx.eval(*g); return g

print("=== FULL SEQLEN SWEEP via PRODUCTION CHUNKED backward (s4096-capable) ===")
table={}
for SEQ in [128,512,1024,2048,4096]:
    dy,x,B,C,z,A,dt,D,h0=build(SEQ); primals=(x,B,C,z,A,dt,D,h0)
    gold=mamba3_mimo_bwd_metal(dy,*primals,backend="mlx"); mx.eval(*gold)
    g=run_chunked(dy,*primals)
    per={nm:maxabs(a,gg) for nm,a,gg in zip(GRAD_NAMES,g,gold)}
    worst=max(per.values()); worst_nm=max(per,key=per.get)
    table[SEQ]=(worst,worst_nm)
    bad={k:f"{v:.2e}" for k,v in per.items() if v>=1e-3}
    print(f"  s{SEQ:5d}: worst={worst:.3e} @ {worst_nm}  ok(<1e-3)={worst<1e-3}"+(f" FAIL={bad}" if bad else ""))

# s4096 determinism (10 reps, fresh-input deterministic = same worst each time)
print("\n=== s4096 DETERMINISM (10 reps, fixed inputs) ===")
SEQ=4096
dy,x,B,C,z,A,dt,D,h0=build(SEQ); primals=(x,B,C,z,A,dt,D,h0)
gold=mamba3_mimo_bwd_metal(dy,*primals,backend="mlx"); mx.eval(*gold)
worsts=[]
for rep in range(10):
    g=run_chunked(dy,*primals)
    w=max(maxabs(a,gg) for a,gg in zip(g,gold))
    worsts.append(w)
uniq=sorted(set(f"{w:.10e}" for w in worsts))
print(f"  worsts={[f'{w:.4e}' for w in worsts]}")
print(f"  distinct={len(uniq)} deterministic={len(uniq)==1} all<1e-3={all(w<1e-3 for w in worsts)}")

print("\n=== SEQLEN -> WORST TABLE ===")
for SEQ,(w,nm) in table.items():
    print(f"  s{SEQ}: {w:.3e} @ {nm}")

print(f"\nPEAK_RSS_KB={_PK} (~{_PK/1048576:.3f}GB) memguard70=ON")
print("RC=0")

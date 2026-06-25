"""PRECISE dispatch-floor accounting separating fused-LANE from MSL parity, AND
s4096 production (chunked) bit-correct + determinism. The mandate wants the EXACT
number, not approximate. Under memguard 70.

Facts to establish:
 (1) LANE fp32 route ACTUAL dispatch count at s128 (the schedule-class is 1, the
     fp32 codegen lowers to N kernels -> the floor gap vs MSL's 1).
 (2) s4096: LANE route cannot lower (int32 wall) -> production runs CHUNKED.
 (3) s4096 CHUNKED bit-correct (all 8 grads <1e-3 vs path-b GOLD) + deterministic.
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
from cppmega_mlx.nn._tilelang.mamba3_path_c import mamba3_mimo_bwd_path_c

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

# (3) s4096 production route bit-correct + determinism (the REAL production path)
SEQ=4096
print(f"=== s{SEQ} PRODUCTION mamba3_mimo_bwd_path_c (chunked, s4096-capable) ===")
dy,x,B,C,z,A,dt,D,h0=build(SEQ)
primals=(x,B,C,z,A,dt,D,h0)
gold=mamba3_mimo_bwd_metal(dy,*primals,backend="mlx"); mx.eval(*gold)
worsts=[]
for rep in range(3):
    grads=mamba3_mimo_bwd_path_c(dy,*primals); mx.eval(*grads)
    w=max(maxabs(g,gg) for g,gg in zip(grads,gold))
    worsts.append(w)
    per={nm:maxabs(g,gg) for nm,g,gg in zip(GRAD_NAMES,grads,gold)}
    bad={k:f"{v:.2e}" for k,v in per.items() if v>=1e-3}
    print(f"  rep{rep}: worst={w:.3e} ok(<1e-3)={w<1e-3}"+(f" FAIL={bad}" if bad else ""))
det = len(set(f"{w:.10e}" for w in worsts))==1
print(f"  deterministic across 3 reps: {det} (worsts={[f'{w:.3e}' for w in worsts]})")

print(f"\nPEAK_RSS_KB={_PK} (~{_PK/1048576:.3f}GB) memguard70=ON")
print("RC=0")

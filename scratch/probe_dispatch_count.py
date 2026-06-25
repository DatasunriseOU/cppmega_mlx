"""Confirm dispatch count: count distinct compiled TileLang kernel invocations
for the fp32 SIMD-fused route vs the 3-kernel partial route at nam56r.
We instrument by counting tilelang.compile cache entries / kernel call sites.
memguard 70.
"""
from __future__ import annotations
import os, sys, threading, time
_LIM = 70*1024*1024; _PEAK=0
def _rss():
    import resource; return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//1024)
def _mg():
    global _PEAK
    while True:
        r=_rss()
        if r>_PEAK:_PEAK=r
        if r>_LIM: os._exit(137)
        time.sleep(0.25)
threading.Thread(target=_mg,daemon=True).start()
os.environ.setdefault("TILELANG_MLX_TVM_FFI_FORCE_COMMAND_BUFFER_BOUNDARY","1")
sys.path.insert(0,"/Volumes/external/sources/cppmega.mlx")

import numpy as np, mlx.core as mx
import cppmega_mlx.nn._tilelang.mamba3_path_c as M

b,s,h,p,n = 1,128,128,64,64
rng=np.random.RandomState(0)
def f32(*sh,sc=0.1): return mx.array((rng.randn(*sh)*sc).astype(np.float32))
x=f32(b,s,h,p);B=f32(b,s,h,n);C=f32(b,s,h,n);z=f32(b,s,h,p,sc=0.5)
A=mx.array(np.broadcast_to((-rng.rand(h)).astype(np.float32)[None,None,:],(b,s,h)).copy())
dt=mx.array((rng.rand(b,s,h)*0.05).astype(np.float32));D=mx.array((rng.randn(h)).astype(np.float32))
h0=f32(b,h,p,n);cot=mx.array((rng.randn(b,s,h,p)*0.1).astype(np.float32))
prim=(x,B,C,z,A,dt,D,h0)

# Count actual kernel __call__ dispatches by wrapping tilelang compiled kernels.
import tilelang
orig_compile = tilelang.compile
counters = {"simd": [], "partial": []}
active = {"tag": None}
class _CountWrap:
    def __init__(self, k, name): self._k=k; self._name=name
    def __call__(self,*a,**kw):
        t=active["tag"]
        if t: counters[t].append(self._name)
        return self._k(*a,**kw)
    def __getattr__(self,n): return getattr(self._k,n)
def _patched(func,*a,**kw):
    k=orig_compile(func,*a,**kw)
    nm=getattr(getattr(func,'__name__',None),'__str__',lambda:'?')()
    try: nm=func.__name__ if hasattr(func,'__name__') else str(getattr(func,'name','?'))
    except Exception: nm='?'
    return _CountWrap(k, nm)
tilelang.compile=_patched

# warm compile both (compiles cached), then count call dispatches per route.
active["tag"]="simd"
g1=M._mamba3_mimo_bwd_path_c_simd_kernel(cot,*prim); mx.eval(*g1)
active["tag"]="partial"
g2=M._mamba3_mimo_bwd_path_c_partial_kernel(cot,*prim); mx.eval(*g2)
active["tag"]=None

print("[SIMD-fused route]  kernel dispatches:", counters["simd"])
print("[3-kernel partial]  kernel dispatches:", counters["partial"])
print(f"SIMD dispatch_count = {len(counters['simd'])}")
print(f"PARTIAL dispatch_count = {len(counters['partial'])}")
print(f"PEAK_RSS_KB={_PEAK} memguard70=ON")
print("RC=0")

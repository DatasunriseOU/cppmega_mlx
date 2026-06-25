"""Run ONLY the B2 dispatch (tested-S512) a fixed number of times for xctrace /
Metal System Trace capture. memguard 70. No fabrication."""
from __future__ import annotations
import os, sys, threading, time
import numpy as np

_LIM = 70 * 1024 * 1024
_PEAK = 0
def _rss():
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
def _guard():
    global _PEAK
    while True:
        r = _rss()
        if r > _PEAK: _PEAK = r
        if r > _LIM:
            sys.stderr.write(f"[memguard70] KILL rss_kb={r}\n"); sys.stderr.flush(); os._exit(137)
        time.sleep(0.25)
threading.Thread(target=_guard, daemon=True).start()

sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx")
import torch
from cppmega_mlx.nn._tilelang.mamba3_chunked_backward_core import (
    build_chunk_scan_combine_bwd_metal,
)

DEV = "mps"
b, S, L, G, H, P, N = 1, 512, 64, 1, 2, 64, 16
nchunks = S // L
rng = np.random.RandomState(0)
def th(a, d=torch.float32): return torch.tensor(a, device=DEV, dtype=d).contiguous()

dout = th(rng.randn(b,S,H,P)*0.1, torch.float16)
cb = th(rng.randn(b,nchunks,G,L,L)*0.1, torch.float16)
x = th(rng.randn(b,S,H,P)*0.1, torch.float16)
z = th(rng.randn(b,S,H,P)*0.5, torch.float16)
dt = th(rng.rand(b,H,nchunks,L)*0.05, torch.float16)
dA = th(rng.randn(b,H,nchunks,L)*0.1, torch.float16)
C = th(rng.randn(b,S,G,N)*0.1, torch.float16)
Bt = th(rng.randn(b,S,G,N)*0.1, torch.float16)
prev = th(rng.randn(b,nchunks,H,P,N)*0.1, torch.float32)
D = th(rng.randn(H), torch.float16)
y = th(rng.randn(b,S,H,P)*0.1, torch.float16)

k = build_chunk_scan_combine_bwd_metal(b, S, L, G, H, P, N)

def alloc():
    return (torch.zeros(b,S,H,N,device=DEV,dtype=torch.float32),
            torch.zeros(b,S,H,P,device=DEV,dtype=torch.float32),
            torch.zeros(b,S,H,P,device=DEV,dtype=torch.float32),
            torch.zeros(b,nchunks,H,P,N,device=DEV,dtype=torch.float32),
            torch.zeros(b,S,H,P,N,device=DEV,dtype=torch.float32),
            torch.zeros(b,H,nchunks,L,device=DEV,dtype=torch.float32),
            torch.zeros(H,device=DEV,dtype=torch.float32))

ITERS = int(os.environ.get("B2_ITERS", "10"))
for _ in range(2):  # warmup
    outs = alloc()
    k(dout, cb, x, z, dt, dA, C, Bt, prev, D, y, *outs)
    torch.mps.synchronize()
t0 = time.perf_counter()
for _ in range(ITERS):
    outs = alloc()
    k(dout, cb, x, z, dt, dA, C, Bt, prev, D, y, *outs)
    torch.mps.synchronize()
dt_us = (time.perf_counter()-t0)/ITERS*1e6
print(f"B2 tested-S512 mean over {ITERS} iters = {dt_us:.1f}us")
print(f"PEAK_RSS_KB={_PEAK} (~{_PEAK/1048576:.3f}GB) memguard70=ON")
print("RC=0")

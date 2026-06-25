"""PROBE: isolate WHICH fused-kernel output races at s4096. Recompute the B1
reverse scan (dstates, dh0) in pure fp32 MLX from the kernel's OWN dchunk_states
+ stashed prev_states/dA_cumsum, and compare against the fused kernel's dstates/dh0
across many in-process repeats. memguard 70. NOT a production edit.

If dstates/dh0 from the kernel diverge from the MLX reference intermittently -> the
fused B1 grid output races. If they always match but the final grads still flip ->
the race is downstream (B0 read ordering / dA_tail).
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
from cppmega_mlx.runtime.path_c_backward_fusion_search import _build_eval_inputs, _maxabs, _GRAD_NAMES
from cppmega_mlx.nn._tilelang.mamba3_path_c import (
    _chunked_bwd_b2b1_fused_kernel, _force_chunked_command_buffer_boundary,
    _path_c_chunk_size_for, _assert_per_head_constant_A,
)

_force_chunked_command_buffer_boundary()
S = int(os.environ.get("PROBE_SEQ", "4096"))
H = int(os.environ.get("PROBE_H", "128"))
N = int(os.environ.get("PROBE_N", "64"))
REPEATS = int(os.environ.get("PROBE_REPEATS", "12"))
dims = (1, S, 64, H, H, 64, N)
inp = _build_eval_inputs(dims)
batch, heads, headdim, state = 1, H, 64, N
chunk = 64
nchunks = S // chunk
G = H

x, B, C, z, A, dt, D, h0 = inp["primals"]
dy = inp["dy"]
cb = inp["cb"]; dA_cumsum = inp["dA_cumsum"]; prev_states = inp["prev_states"]; y = inp["y"]

def f16(a): return mx.contiguous(a.astype(mx.float16))
x16=f16(x); B16=f16(B); C16=f16(C); z16=f16(z); D16=f16(D); dout16=f16(dy); y16=f16(y)
dt16=dt.astype(mx.float16)
dt_k=mx.contiguous(mx.transpose(dt16.reshape(batch,nchunks,chunk,heads),(0,3,1,2)))
cb16=cb.astype(mx.float16); dA16=dA_cumsum.astype(mx.float16)
A_head16=_assert_per_head_constant_A(A).astype(mx.float16)
prev32=prev_states.astype(mx.float32)

k = _chunked_bwd_b2b1_fused_kernel(batch,S,chunk,G,heads,headdim,state)

# Pure-MLX B1 reference reverse scan from a FIXED dchunk_states (deterministic):
# we capture dchunk from the FIRST kernel run, then reference dstates/dh0 are a
# function of (dchunk, dA_cumsum tail, prev_states). decay=exp(tail).
L = chunk
tail = dA16[:, :, :, L-1].astype(mx.float32)            # (b,H,nchunks)
decay = mx.exp(tail)                                     # (b,H,nchunks)
def b1_ref(dchunk):
    # dchunk: (b,nchunks,H,P,N) fp32. g carried reverse; dstates[c]=g; g=decay[c]*g+dchunk[c]
    dch = dchunk.astype(mx.float32)
    g = mx.zeros((batch,heads,headdim,state), dtype=mx.float32)  # dh_last=0
    dstates_ref = [None]*nchunks
    for cc in range(nchunks-1, -1, -1):
        dstates_ref[cc] = g
        dec = decay[:, :, cc][:, :, None, None]  # (b,H,1,1)
        g = dec * g + mx.transpose(dch[:, cc], (0,1,2,3))  # dch[:,cc]=(b,H,P,N)
    dh0_ref = g
    ds = mx.stack(dstates_ref, axis=1)  # (b,nchunks,H,P,N)
    return ds, dh0_ref

dh_last = mx.zeros((batch,heads,headdim,state),dtype=mx.float32)
ref_ds = None; ref_dh0 = None
print(f"# s={S} H={H} N={N} nchunks={nchunks} repeats={REPEATS}")
for r in range(REPEATS):
    out = k(dout16,cb16,x16,z16,dt_k,dA16,C16,B16,prev32,D16,y16,dh_last)
    (dC_m,dx_b2,dz_m,dchunk,dinp_diag,dA_y,_dD,dstates,dh0_m,_tail)=out
    mx.eval(dstates, dh0_m, dchunk)
    if ref_ds is None:
        ref_ds, ref_dh0 = b1_ref(dchunk)
        mx.eval(ref_ds, ref_dh0)
    d_ds = _maxabs(dstates.astype(mx.float32), ref_ds)
    d_dh0 = _maxabs(dh0_m.astype(mx.float32), ref_dh0)
    # also re-derive ref from THIS run's dchunk to detect dchunk races
    ds2, dh02 = b1_ref(dchunk); mx.eval(ds2, dh02)
    d_ds_self = _maxabs(dstates.astype(mx.float32), ds2)
    d_dh0_self = _maxabs(dh0_m.astype(mx.float32), dh02)
    print(f"run {r:2d}: kernel-dstates vs MLX-B1(firstdchunk)={d_ds:.3e}  dh0={d_dh0:.3e} | "
          f"vs MLX-B1(thisdchunk) dstates={d_ds_self:.3e} dh0={d_dh0_self:.3e}", flush=True)

"""LOCAL Apple-Metal numeric A/B for the §METAL-RETILE sub_chunks=2 body.

Compares the byte-identical SERIAL prod Metal prim vs the BATCHED Metal GEMM prim
built with sub_chunks=2 (the L_sub=L/2 re-tile carry), at a SMALL in-budget bounded
shape (L=32 -> L_sub=16, P=N=32, HPC=1) that fits Apple's 32 KB pool. This is the
ONLY shape at which the retile carry can be NUMERICALLY validated on the Apple GPU:
prod L=64 via this retile is 68096 B (HPC=1) / 99328 B (HPC=2) >> 32 KB, so the
build gate RAISES (honest NO-GO) and never dispatches over-budget (RULE #1).

Validates: the sub_chunks=2 body RUNS on Apple GPU and ALL grads parity <1e-3 vs the
serial prod prim (the inter-sub-chunk dchunk additive + dinp first-order cross carry +
dC_diag/dseg diag00/diag11/cross10 block recombination are numerically clean).
"""
import os
import time

import numpy as np
import mlx.core as mx  # noqa: F401  (forces Metal availability)
import tilelang
import tilelang.language as T  # noqa: F401
import torch

from cppmega_mlx.nn._tilelang.mamba3_chunked_backward_core import (
    chunk_scan_combine_bwd_metal_prim,
    chunk_scan_combine_bwd_metal_gemm_prim_batched,
)

DEV = "mps"
B, S, CH, G, H, P, N = 1, 64, 32, 2, 4, 32, 32  # L=32 -> sub_chunks=2 -> L_sub=16
HPC = 1
SUB = 2
nchunks = S // CH

rng = np.random.RandomState(0)
def f16(shape):
    return torch.tensor((rng.randn(*shape) * 0.1).astype(np.float32), dtype=torch.float16, device=DEV)
def f32(shape):
    return torch.tensor((rng.randn(*shape) * 0.1).astype(np.float32), dtype=torch.float32, device=DEV)

dout = f16((B, S, H, P)); cb = f16((B, nchunks, G, CH, CH)); x = f16((B, S, H, P))
z = f16((B, S, H, P)); dt = f16((B, H, nchunks, CH)); dA = f16((B, H, nchunks, CH))
C = f16((B, S, G, N)); Bm = f16((B, S, G, N)); prev = f32((B, nchunks, H, P, N))
D = f16((H,)); y = f16((B, S, H, P))

def zeros():
    return (torch.zeros(B, S, H, N, dtype=torch.float32, device=DEV),
            torch.zeros(B, S, H, P, dtype=torch.float32, device=DEV),
            torch.zeros(B, S, H, P, dtype=torch.float32, device=DEV),
            torch.zeros(B, nchunks, H, P, N, dtype=torch.float32, device=DEV),
            torch.zeros(B, S, H, P, N, dtype=torch.float32, device=DEV),
            torch.zeros(B, H, nchunks, CH, dtype=torch.float32, device=DEV),
            torch.zeros(H, dtype=torch.float32, device=DEV))

OUT = [11, 12, 13, 14, 15, 16, 17]
def compile_prim(prim):
    return tilelang.compile(prim, out_idx=OUT, target=None)
def run(kern):
    o = zeros()
    kern(dout, cb, x, z, dt, dA, C, Bm, prev, D, y, *o)
    torch.mps.synchronize()
    return o
def timeit(kern, iters=30):
    run(kern); torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        o = zeros()
        kern(dout, cb, x, z, dt, dA, C, Bm, prev, D, y, *o)
    torch.mps.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3

print(f"[METAL-RETILE] dims B={B} S={S} chunk={CH} G={G} H={H} P={P} N={N} HPC={HPC} SUB={SUB}")
serial_prim = chunk_scan_combine_bwd_metal_prim(B, S, CH, G, H, P, N)
retile_prim = chunk_scan_combine_bwd_metal_gemm_prim_batched(
    B, S, CH, G, H, P, N, heads_per_cta=HPC, sub_chunks=SUB)

k_serial = compile_prim(serial_prim)
k_retile = compile_prim(retile_prim)

o_s = run(k_serial)
o_b = run(k_retile)
names = ["dC", "dx", "dz", "dchunk", "dinp", "dA_y", "dD"]
maxabs = {nm: float((a - b).abs().max().cpu()) for nm, a, b in zip(names, o_s, o_b)}

t_s = timeit(k_serial)
t_b = timeit(k_retile)
speedup = t_s / t_b if t_b > 0 else float("nan")
verdict = "GO" if t_b < t_s else "NO-GO"
worst = max(maxabs.values())
parity = "PASS" if worst < 1e-3 else "FAIL"

print(f"[METAL-RETILE] serial={t_s:.3f}ms  retile(sub2)={t_b:.3f}ms  speedup={speedup:.3f}x  {verdict}")
print(f"[METAL-RETILE] parity {parity} worst={worst:.2e}  "
      + "  ".join(f"{k}={v:.2e}" for k, v in maxabs.items()))
print("RETILE_RESULT_JSON " + str({
    "serial_ms": round(t_s, 4), "retile_ms": round(t_b, 4),
    "speedup": round(speedup, 4), "verdict": verdict, "sub_chunks": SUB, "HPC": HPC,
    "parity": parity, "maxabs": {k: f"{v:.2e}" for k, v in maxabs.items()},
}))

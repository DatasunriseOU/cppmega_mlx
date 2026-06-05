"""LEAN gb10 A/B: B2 v1-threaded vs §27 single-tile GEMM vs batched-large-tile.

Skips the slow prod-shape autograd gold (the full parity gate is the separate
probe). This isolates the B2 TIMING + the batched-vs-{v1,§27} grad self-consistency
(batched vs v1 max|abs| on the GEMM-able outputs). prod cfg, bs1 or bs4.
RULE #1: build/run errors propagate.
"""
import os
import sys
import time

import numpy as np
import torch

import tilelang as tl
from cppmega_mlx.nn._tilelang.mamba3_chunked_backward_core import (
    chunk_scan_combine_bwd_cuda_prim,
    chunk_scan_combine_bwd_cuda_prim_gemm,
    chunk_scan_combine_bwd_cuda_prim_gemm_batched,
    _b2_batched_heads_per_cta,
)
from cppmega_mlx.nn._tilelang.mamba3_chunked_scan_core import (
    _resolve_chunked_compile_target as _rct,
)

DEV = "cuda"
BS = 4 if "--bs4" in sys.argv else 1
HPC_REQ = int(os.environ.get("CPPMEGA_PATH_C_B2_HEADS_PER_CTA", "2"))
b, S, chunk, G, H, P, N = BS, 4096, 64, 8, 112, 64, 64
nchunks = S // chunk
HPC = _b2_batched_heads_per_cta(H, H // G, HPC_REQ)
print(f"[B2-AB] dev={torch.cuda.get_device_name(0)} bs={b} S={S} c={chunk} G={G} "
      f"H={H} P={P} N={N} HPC={HPC}")

rng = np.random.RandomState(0)
def f16(*sh):
    return torch.tensor((rng.randn(*sh) * 0.1).astype(np.float32), device=DEV, dtype=torch.float16).contiguous()
def f32(*sh):
    return torch.tensor((rng.randn(*sh) * 0.1).astype(np.float32), device=DEV, dtype=torch.float32).contiguous()

dout = f16(b, S, H, P); cb = f16(b, nchunks, G, chunk, chunk); x = f16(b, S, H, P)
z = f16(b, S, H, P); dt = f16(b, H, nchunks, chunk); dA = f16(b, H, nchunks, chunk)
C = f16(b, S, G, N); Bm = f16(b, S, G, N); prev = f32(b, nchunks, H, P, N)
D = f16(H); y = f16(b, S, H, P)
INP = (dout, cb, x, z, dt, dA, C, Bm, prev, D, y)

def outs():
    return [torch.zeros(b, S, H, N, device=DEV, dtype=torch.float32),
            torch.zeros(b, S, H, P, device=DEV, dtype=torch.float32),
            torch.zeros(b, S, H, P, device=DEV, dtype=torch.float32),
            torch.zeros(b, nchunks, H, P, N, device=DEV, dtype=torch.float32),
            torch.zeros(b, S, H, P, N, device=DEV, dtype=torch.float32),
            torch.zeros(b, H, nchunks, chunk, device=DEV, dtype=torch.float32),
            torch.zeros(H, device=DEV, dtype=torch.float32)]

_tgt = _rct("cuda")
_pc = {"tl.disable_tma_lower": True, "tl.disable_warp_specialized": True}
OUT = [11, 12, 13, 14, 15, 16, 17]

def build(prim):
    return tl.compile(prim, out_idx=OUT, target=_tgt, pass_configs=_pc)

def run(k, o):
    for t in o:
        t.zero_()
    k(*INP, *o)

def timeit(k, n=20):
    o = outs()
    run(k, o); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        run(k, o)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3, o

print("[B2-AB] building v1-threaded ..."); k1 = build(chunk_scan_combine_bwd_cuda_prim(b, S, chunk, G, H, P, N))
print("[B2-AB] building §27 single-tile gemm ..."); kg = build(chunk_scan_combine_bwd_cuda_prim_gemm(b, S, chunk, G, H, P, N))
print(f"[B2-AB] building batched HPC={HPC} ..."); kb = build(chunk_scan_combine_bwd_cuda_prim_gemm_batched(b, S, chunk, G, H, P, N, heads_per_cta=HPC))

t1, o1 = timeit(k1)
tg, og = timeit(kg)
tb, ob = timeit(kb)

names = ["dC", "dx", "dz", "dchunk", "dinp", "dA_y", "dD"]
eq_b_v1 = {nm: float((a - c).abs().max().cpu()) for nm, a, c in zip(names, ob, o1)}
worst = max(eq_b_v1.values())

print(f"\n[B2-AB] MEASURED v1_threaded={t1:.3f}ms  §27_single_tile_gemm={tg:.3f}ms  "
      f"batched={tb:.3f}ms  HPC={HPC} bs={b}")
print(f"[B2-AB] batched_vs_v1={t1/tb:.3f}x  batched_vs_§27={tg/tb:.3f}x  "
      f"§27_vs_v1={t1/tg:.3f}x")
print(f"[B2-AB] verdict_vs_v1={'GO' if tb < t1 else 'NO-GO'}  "
      f"verdict_vs_§27={'GO' if tb < tg else 'NO-GO'}")
print(f"[B2-AB] batched-vs-v1 max|abs| worst={worst:.2e}  "
      + "  ".join(f"{k}={v:.2e}" for k, v in eq_b_v1.items()))
print("B2_AB_JSON " + str({
    "bs": b, "HPC": HPC, "v1_ms": round(t1, 4), "s27_gemm_ms": round(tg, 4),
    "batched_ms": round(tb, 4), "batched_vs_v1": round(t1/tb, 4),
    "batched_vs_s27": round(tg/tb, 4), "s27_vs_v1": round(t1/tg, 4),
    "worst_vs_v1": f"{worst:.2e}", "maxabs": {k: f"{v:.2e}" for k, v in eq_b_v1.items()},
}))

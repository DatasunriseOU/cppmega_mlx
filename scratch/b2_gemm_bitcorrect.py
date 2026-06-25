"""B2-ONLY bit-correctness: serial Metal B2 vs the §HALFSPLIT simdgroup-GEMM B2.

Builds BOTH B2 prims on the SAME random inputs (tested-S512 dims), runs each,
and reports per-output max|abs| diff. The serial prim is the parity reference
(the de-funnel base, WORST 3.84e-04 vs the MLX proto). A simdgroup-GEMM B2 that
matches the serial prim to well below the 1e-3 backward gate is bit-correct.
memguard 70. No fabrication: real Metal dispatch via tilelang JITKernel.
"""
import os, sys, threading, time
_LIM = 70 * 1024 * 1024
_PEAK = 0
def _rss():
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
def _g():
    global _PEAK
    while True:
        r = _rss()
        if r > _PEAK: _PEAK = r
        if r > _LIM:
            sys.stderr.write(f"[memguard70] KILL rss_kb={r}\n"); os._exit(137)
        time.sleep(0.25)
threading.Thread(target=_g, daemon=True).start()

sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx")
import numpy as np
import torch
from einops import rearrange
from cppmega_mlx.nn._tilelang.mamba3_chunked_backward_core import (
    build_chunk_scan_combine_bwd_metal,
)

b, seqlen, chunk, G, H, P, N = 1, 512, 64, 1, 2, 64, 16
nchunks = seqlen // chunk
dev = "mps"
rng = np.random.RandomState(0)
x_np = (rng.randn(b, seqlen, H, P) * 0.1).astype(np.float32)
B_np = (rng.randn(b, seqlen, G, N) * 0.1).astype(np.float32)
C_np = (rng.randn(b, seqlen, G, N) * 0.1).astype(np.float32)
A_np = (-rng.rand(H)).astype(np.float32)
dt_np = (rng.rand(b, seqlen, H) * 0.05).astype(np.float32)
D_np = (rng.randn(H)).astype(np.float32)
dout_np = (rng.randn(b, seqlen, H, P) * 0.1).astype(np.float32)
z_np = (rng.randn(b, seqlen, H, P) * 0.5).astype(np.float32)
y_np = (rng.randn(b, seqlen, H, P) * 0.1).astype(np.float32)
prev_np = (rng.randn(b, nchunks, H, P, N) * 0.1).astype(np.float32)
cb_np = (rng.randn(b, nchunks, G, chunk, chunk) * 0.1).astype(np.float32)
dA_np = (-rng.rand(b, H, nchunks, chunk) * 2.0).astype(np.float32)

def th(a, d=torch.float16):
    return torch.tensor(a, device=dev, dtype=d).contiguous()

dt_k = rearrange(th(dt_np), "b (c s) hh -> b hh c s", c=nchunks).contiguous()

def run_b2(gemm: bool):
    if gemm:
        os.environ["CPPMEGA_PATH_C_METAL_GEMM"] = "1"
    else:
        os.environ.pop("CPPMEGA_PATH_C_METAL_GEMM", None)
    k = build_chunk_scan_combine_bwd_metal(b, seqlen, chunk, G, H, P, N)
    dC = torch.zeros(b, seqlen, H, N, device=dev, dtype=torch.float32)
    dx = torch.zeros(b, seqlen, H, P, device=dev, dtype=torch.float32)
    dz = torch.zeros(b, seqlen, H, P, device=dev, dtype=torch.float32)
    dchunk = torch.zeros(b, nchunks, H, P, N, device=dev, dtype=torch.float32)
    dinp = torch.zeros(b, seqlen, H, P, N, device=dev, dtype=torch.float32)
    dA_y = torch.zeros(b, H, nchunks, chunk, device=dev, dtype=torch.float32)
    dD = torch.zeros(H, device=dev, dtype=torch.float32)
    k(th(dout_np), th(cb_np), th(x_np), th(z_np), dt_k, th(dA_np),
      th(C_np), th(B_np), th(prev_np, torch.float32), th(D_np), th(y_np),
      dC, dx, dz, dchunk, dinp, dA_y, dD)
    torch.mps.synchronize()
    return dict(dC=dC, dx=dx, dz=dz, dchunk=dchunk, dinp=dinp, dA_y=dA_y, dD=dD)

ser = run_b2(False)
gem = run_b2(True)

def d(a, bb):
    return float(np.abs(a.float().cpu().numpy().astype(np.float64)
                        - bb.float().cpu().numpy().astype(np.float64)).max())

diffs = {k: d(ser[k], gem[k]) for k in ser}
worst = max(diffs.values())
print("[B2 serial vs §HALFSPLIT-GEMM] per-output max|abs|: "
      + " ".join(f"{k}={v:.3e}" for k, v in diffs.items()))
print(f"WORST_B2_GEMM_vs_SERIAL = {worst:.3e}")
print(f"PEAK_RSS_KB={_PEAK} (~{_PEAK/1048576:.3f}GB) memguard70=ON")
print("RC=0")

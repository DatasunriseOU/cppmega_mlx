"""MEASURED B2-only A/B: serial Metal B2 vs the §DCHUNK-ONLY simdgroup-GEMM B2.

Reuses the production bench harness's input builders + forward prereqs, times
ONLY the B2 (chunk_scan_combine_bwd) dispatch on both tested shapes, serial vs
GEMM. Isolates B2 so B0 (which raises under the global GEMM flag) is not built.
memguard 70. Reports median us, effective GFLOP/s for the dchunk_states GEMM.
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
sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx/scratch")
import numpy as np
import torch
from einops import rearrange

from cppmega_mlx.nn._tilelang.mamba3_chunked_backward_core import (
    build_chunk_scan_combine_bwd_metal,
)
import bench_mamba3_chunked_vs_msl_bwd as H

ITERS = int(os.environ.get("BENCH_ITERS", "50"))
WARMUP = int(os.environ.get("BENCH_WARMUP", "10"))
DEV = "mps"


def time_b2(b, seqlen, chunk, G, H_, P, N, gemm):
    if gemm:
        os.environ["CPPMEGA_PATH_C_METAL_GEMM"] = "1"
    else:
        os.environ.pop("CPPMEGA_PATH_C_METAL_GEMM", None)
    d = H.build_inputs(b, seqlen, chunk, G, H_, P, N, seed=0)
    pre = H.forward_prereqs(d, b, seqlen, chunk, G, H_, P, N)
    nchunks = seqlen // chunk
    th = H.th
    dt_k = rearrange(th(d["dt"], torch.float16), "b (c s) hh -> b hh c s", c=nchunks).contiguous()
    k_b2 = build_chunk_scan_combine_bwd_metal(b, seqlen, chunk, G, H_, P, N)
    dout_t = th(d["dout"], torch.float16); cb = pre["cb"]; x_t = th(d["x"], torch.float16)
    z_t = th(d["z"], torch.float16); dA = pre["dA"]; C_t = th(d["C"], torch.float16)
    B_t = th(d["B"], torch.float16); prev = pre["prev"]; D_t = th(d["D"], torch.float16)
    y_t = th(pre["y"], torch.float16)

    out_bufs = [torch.zeros(*s, device=DEV, dtype=torch.float32) for s in
                [(b, seqlen, H_, N), (b, seqlen, H_, P), (b, seqlen, H_, P),
                 (b, nchunks, H_, P, N), (b, seqlen, H_, P, N), (b, H_, nchunks, chunk), (H_,)]]
    cb_c = cb.contiguous(); dA_c = dA.contiguous(); prev_c = prev.contiguous()

    def run():
        for o in out_bufs:
            o.zero_()
        k_b2(dout_t, cb_c, x_t, z_t, dt_k, dA_c,
             C_t, B_t, prev_c, D_t, y_t, *out_bufs)

    for _ in range(WARMUP):
        run()
    torch.mps.synchronize()
    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        run()
        torch.mps.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2] * 1e6


SHAPES = {
    "tested-S512": dict(b=1, seqlen=512, chunk=64, G=1, H_=2, P=64, N=16),
    "nam56r-H128": dict(b=1, seqlen=512, chunk=64, G=8, H_=128, P=64, N=64),
}

print(f"ITERS={ITERS} WARMUP={WARMUP}")
for name, sh in SHAPES.items():
    ser = time_b2(**sh, gemm=False)
    gem = time_b2(**sh, gemm=True)
    nchunks = sh["seqlen"] // sh["chunk"]
    # dchunk_states GEMM FLOPs: per (b,chunk,head): 2*P*N*L MACs*2
    flops = 2 * sh["b"] * nchunks * sh["H_"] * sh["P"] * sh["N"] * sh["chunk"]
    g_gflops = flops / (gem * 1e-6) / 1e9
    s_gflops = flops / (ser * 1e-6) / 1e9
    print(f"[{name}] B2 serial={ser:.1f}us  GEMM={gem:.1f}us  speedup={ser/gem:.3f}x  "
          f"| dchunk GFLOP/s serial={s_gflops:.1f} GEMM={g_gflops:.1f}")

print(f"\nPEAK_RSS_KB={_PEAK} (~{_PEAK/1048576:.3f}GB) memguard70=ON")
print("RC=0")

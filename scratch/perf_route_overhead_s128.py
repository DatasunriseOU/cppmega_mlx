"""ROUTE-OVERHEAD A/B at the SAME parity dims (S=128, G=H=128): torch/mps
build_*_metal route (the route the 0.98x receipt was measured on) vs the
production MLX tvm_ffi route. Same logical bwd problem, same dims, so the
delta isolates the route overhead. memguard 70.

Reuses the torch/mps Path-A runner from bench_mamba3_chunked_vs_msl_bwd.py.
"""
import os
import sys
import threading
import time

_MEMGUARD_LIMIT_KB = 70 * 1024 * 1024
_PEAK = 0


def _rss_kb():
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)


def _guard():
    global _PEAK
    while True:
        r = _rss_kb()
        if r > _PEAK:
            _PEAK = r
        if r > _MEMGUARD_LIMIT_KB:
            sys.stderr.write(f"[memguard70] KILL rss_kb={r}\n")
            sys.stderr.flush()
            os._exit(137)
        time.sleep(0.25)


threading.Thread(target=_guard, daemon=True).start()
sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx")
sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx/scratch")

import numpy as np

# Import the torch/mps bench building blocks (Path A = build_*_metal torch route).
import bench_mamba3_chunked_vs_msl_bwd as bench

# Parity-surface dims: b=1, S=128 (2 chunks of 64), G==H==128, P=64, N=64.
b, seqlen, chunk, G, H, P, N = 1, 128, 64, 128, 128, 64, 64
ITERS = 30
WARMUP = 8


def _median_us(times):
    return float(np.median(np.asarray(times, np.float64))) * 1e6


print("\n=== ROUTE-OVERHEAD A/B (S=128 parity dims) ===")
print(f"dims b={b} S={seqlen} chunk={chunk} G={G} H={H} P={P} N={N} "
      f"median of {ITERS} (warmup {WARMUP})")

d = bench.build_inputs(b, seqlen, chunk, G, H, P, N)
pre = bench.forward_prereqs(d, b, seqlen, chunk, G, H, P, N)
run_b2, run_b1, run_b0, run_all = bench.make_pathA_runners(
    d, pre, b, seqlen, chunk, G, H, P, N
)

print("\n[torch/mps build_*_metal route — the receipt route]")
a_us = bench.time_runner(run_all, ITERS, WARMUP, "A=chained B2->B1->B0 (torch/mps)")
b2_us = bench.time_runner(run_b2, ITERS, WARMUP, "  B2")
b1_us = bench.time_runner(run_b1, ITERS, WARMUP, "  B1")
b0_us = bench.time_runner(run_b0, ITERS, WARMUP, "  B0")

runB = bench.make_pathB_runner(d, b, seqlen, G, H, P, N)
bmsl_us = bench.time_runner(runB, ITERS, WARMUP, "B=MSL mamba3_mimo_bwd_metal")

print(f"\n  torch/mps: A(chunked)={a_us:.1f}us  B(MSL)={bmsl_us:.1f}us  "
      f"A/B={a_us/bmsl_us:.3f}x  (B2={b2_us:.1f} B1={b1_us:.1f} B0={b0_us:.1f})")

print(f"\nPEAK_RSS_KB={_PEAK} (~{_PEAK/1048576:.3f}GB) memguard70=ON")
print("ROUTE_OVERHEAD_DONE")

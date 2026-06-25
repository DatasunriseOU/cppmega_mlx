"""Partial-route gate harness for the int64 unblock at s4096.

Measures the production PARTIAL bwd route (mamba3_mimo_bwd_path_c -> fp32 LANE ->
_mamba3_mimo_bwd_path_c_partial_kernel) against the path-b GOLD oracle
(mamba3_mimo_bwd_metal backend='mlx', the step-by-step fp32 reference).

Enforces an in-process 70 GB RSS guard (the mandated memguard 70): every dispatch
checks peak RSS and ABORTS (raises) if it crosses the ceiling -- no silent fallback.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time

import numpy as np
import mlx.core as mx
import psutil

from cppmega_mlx.nn._tilelang.mamba3 import mamba3_mimo_bwd_metal
from cppmega_mlx.nn._tilelang.mamba3_path_c import mamba3_mimo_bwd_path_c

MEMGUARD_GB = 70.0
_PROC = psutil.Process(os.getpid())
_PEAK_RSS = 0.0


def _rss_gb() -> float:
    return _PROC.memory_info().rss / (1024 ** 3)


def _guard(where: str) -> None:
    global _PEAK_RSS
    rss = _rss_gb()
    if rss > _PEAK_RSS:
        _PEAK_RSS = rss
    if rss > MEMGUARD_GB:
        raise MemoryError(
            f"MEMGUARD: RSS {rss:.2f} GB > {MEMGUARD_GB} GB at {where} -- ABORT"
        )


GRAD_NAMES = ("dx", "dB", "dC", "dz", "dA", "ddt", "dD", "dh0")


def make_inputs(seq: int, heads: int = 128, headdim: int = 64, state: int = 64,
                batch: int = 1, seed: int = 0, dtype=mx.float32):
    mx.random.seed(seed)
    x = (mx.random.normal((batch, seq, heads, headdim)) * 0.1).astype(dtype)
    B = (mx.random.normal((batch, seq, heads, state)) * 0.1).astype(dtype)
    C = (mx.random.normal((batch, seq, heads, state)) * 0.1).astype(dtype)
    z = (mx.random.normal((batch, seq, heads, headdim)) * 0.1).astype(dtype)
    A = (-mx.random.uniform(0.01, 0.5, (batch, seq, heads))).astype(dtype)
    dt = (mx.random.uniform(0.001, 0.05, (batch, seq, heads))).astype(dtype)
    D = (mx.random.normal((heads,)) * 0.1).astype(dtype)
    h0 = (mx.random.normal((batch, heads, headdim, state)) * 0.1).astype(dtype)
    dy = (mx.random.normal((batch, seq, heads, headdim)) * 0.1).astype(dtype)
    mx.eval(x, B, C, z, A, dt, D, h0, dy)
    return dy, x, B, C, z, A, dt, D, h0


def run_partial(inputs):
    grads = mamba3_mimo_bwd_path_c(*inputs)
    mx.eval(*grads)
    return grads


def run_gold(inputs):
    grads = mamba3_mimo_bwd_metal(*inputs, backend="mlx")
    mx.eval(*grads)
    return grads


def maxdiff(a, b):
    an = np.asarray(a, dtype=np.float32)
    bn = np.asarray(b, dtype=np.float32)
    return float(np.max(np.abs(an - bn)))


def correctness(seq: int, seed: int = 0):
    inputs = make_inputs(seq, seed=seed)
    _guard(f"corr s{seq} inputs")
    pg = run_partial(inputs)
    _guard(f"corr s{seq} partial")
    gg = run_gold(inputs)
    _guard(f"corr s{seq} gold")
    diffs = {n: maxdiff(p, g) for n, p, g in zip(GRAD_NAMES, pg, gg)}
    worst = max(diffs.values())
    del pg, gg, inputs
    gc.collect()
    return diffs, worst


def measure_ms(seq: int, repeats: int = 8, seed: int = 0):
    inputs = make_inputs(seq, seed=seed)
    _guard(f"meas s{seq} inputs")
    # warmup (compile + cache)
    g = run_partial(inputs)
    mx.eval(*g)
    _guard(f"meas s{seq} warmup")
    times = []
    for i in range(repeats):
        t0 = time.perf_counter()
        g = run_partial(inputs)
        mx.eval(*g)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)
        _guard(f"meas s{seq} iter{i}")
        del g
    times.sort()
    median = times[len(times) // 2]
    del inputs
    gc.collect()
    return median, times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["smoke", "corr", "measure", "sweep", "determinism"])
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.mode == "smoke":
        diffs, worst = correctness(args.seq, args.seed)
        print("SMOKE_DIFFS", {k: f"{v:.3e}" for k, v in diffs.items()})
        print("SMOKE_WORST", f"{worst:.3e}")
        print("LOWERED_OK" if worst == worst else "NAN")
        print(f"PEAK_RSS_GB {_PEAK_RSS:.2f}")
        print("PASS" if worst < 1e-3 else "FAIL")

    elif args.mode == "corr":
        diffs, worst = correctness(args.seq, args.seed)
        print(f"CORR s{args.seq}", {k: f"{v:.3e}" for k, v in diffs.items()})
        print(f"CORR_WORST s{args.seq} {worst:.3e}")
        print(f"PEAK_RSS_GB {_PEAK_RSS:.2f}")
        print("PASS" if worst < 1e-3 else "FAIL")

    elif args.mode == "measure":
        median, times = measure_ms(args.seq, args.repeats, args.seed)
        print(f"MEASURE s{args.seq} median_ms {median:.2f} "
              f"all_ms {[round(t,2) for t in times]}")
        print(f"PEAK_RSS_GB {_PEAK_RSS:.2f}")

    elif args.mode == "sweep":
        for s in (512, 1024, 2048):
            diffs, worst = correctness(s, args.seed)
            print(f"SWEEP s{s} worst {worst:.3e} "
                  f"{'PASS' if worst < 1e-3 else 'FAIL'} "
                  f"diffs {{{', '.join(f'{k}={v:.2e}' for k,v in diffs.items())}}}")
        print(f"PEAK_RSS_GB {_PEAK_RSS:.2f}")

    elif args.mode == "determinism":
        # single fresh-process run: hash all 8 grads
        inputs = make_inputs(args.seq, seed=args.seed)
        g = run_partial(inputs)
        import hashlib
        h = hashlib.sha256()
        for arr in g:
            h.update(np.asarray(arr, dtype=np.float32).tobytes())
        print(f"DETERMINISM_HASH s{args.seq} {h.hexdigest()}")
        print(f"PEAK_RSS_GB {_PEAK_RSS:.2f}")


if __name__ == "__main__":
    main()

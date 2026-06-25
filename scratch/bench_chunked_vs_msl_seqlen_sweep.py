"""SEQLEN SWEEP: chunked Path-C bwd e2e vs production MSL bwd e2e.

For seqlen in {128, 512, 1024, 2048, 4096} (b=1, H=128, P=64, N=64, chunk=64 ->
nchunks=seqlen/64) measure the MEDIAN (>=15 iters, warmup) wall-clock of:
  (1) chunked bwd e2e = path_c_fwd_path_c_bwd via mx.vjp  (direct-pipeline+fusion)
  (2) MSL    bwd e2e = mamba3_mimo_fwd_path_c + mamba3_mimo_bwd_metal(backend=metal)
                       with CPPMEGA_MAMBA3_BWD_SEQ_CHUNK=0.

This is the SAME production MLX tvm_ffi route the .vjp uses (the parity script
parity_path_c_chunked_bwd.py structure, extended to a seqlen sweep). Reports the
chunked/MSL ratio at each seqlen and identifies the CROSSOVER seqlen where
chunked <= MSL (if any).

At the largest feasible seqlen (and at the crossover if found) VALIDATE bit-correct:
all 8 grads vs path-b GOLD (mamba3_mimo_bwd_metal backend='mlx', fp32 oracle) < 1e-3.

memguard 70 MANDATORY: a self-imposed 70GB RSS killer thread stays alive the whole
run; per-seqlen peak RSS is tracked. If a seqlen approaches/OOMs the 70GB guard we
report the max feasible seqlen + its ratio and DO NOT remove the guard.

RULE #1: no fabrication (every us a real timed dispatch), no silent fallback (any
ineligible kernel RAISES).
"""
from __future__ import annotations

import os
import sys
import threading
import time

# disable MSL internal seq-chunking BEFORE importing the module (env read at import)
os.environ["CPPMEGA_MAMBA3_BWD_SEQ_CHUNK"] = "0"

_MEMGUARD_LIMIT_KB = 70 * 1024 * 1024  # 70 GiB
_PEAK = 0


def _rss_kb():
    import resource
    # macOS ru_maxrss is bytes -> KB
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)


def _guard():
    global _PEAK
    while True:
        r = _rss_kb()
        if r > _PEAK:
            _PEAK = r
        if r > _MEMGUARD_LIMIT_KB:
            sys.stderr.write(f"[memguard70] KILL self rss_kb={r} (~{r // 1048576}GB) > 70GB\n")
            sys.stderr.flush()
            os._exit(137)
        time.sleep(0.25)


threading.Thread(target=_guard, daemon=True).start()
sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx")

import numpy as np  # noqa: E402
import mlx.core as mx  # noqa: E402

from cppmega_mlx.nn._tilelang.mamba3_path_c import (  # noqa: E402
    mamba3_mimo_apply_with_state_path_c_fwd_path_c_bwd,
    mamba3_mimo_fwd_path_c,
)
from cppmega_mlx.nn._tilelang.mamba3 import mamba3_mimo_bwd_metal  # noqa: E402

# nam56r config-exact dims (b=1, H=128, P=64, N=64, chunk=64). G==H at this surface.
B_BATCH, H, P, N, CHUNK = 1, 128, 64, 64, 64
SEQLENS = [int(s) for s in os.environ.get("SWEEP_SEQLENS", "128,512,1024,2048,4096").split(",")]
ITERS = int(os.environ.get("BENCH_ITERS", "15"))
WARMUP = int(os.environ.get("BENCH_WARMUP", "5"))
NAMES = ["dx", "dB", "dC", "dz", "dA", "ddt", "dD", "dh0"]


def build_inputs(seq, seed=0):
    rng = np.random.RandomState(seed)
    b = B_BATCH

    def f32(*shape, s=0.1):
        return mx.array((rng.randn(*shape) * s).astype(np.float32))

    x = f32(b, seq, H, P)
    B = f32(b, seq, H, N)
    C = f32(b, seq, H, N)
    z = f32(b, seq, H, P, s=0.5)
    A_head = (-rng.rand(H)).astype(np.float32)  # per-head-constant across seq
    A = mx.array(np.broadcast_to(A_head[None, None, :], (b, seq, H)).copy())
    dt = mx.array((rng.rand(b, seq, H) * 0.05).astype(np.float32))
    D = mx.array((rng.randn(H)).astype(np.float32))
    h0 = f32(b, H, P, N)
    cot_y = mx.array((rng.randn(b, seq, H, P) * 0.1).astype(np.float32))
    return (x, B, C, z, A, dt, D, h0), cot_y


def med_us(times):
    return float(np.median(np.asarray(times, np.float64))) * 1e6


def maxdiff(a, bb):
    a = np.asarray(a, np.float64)
    bb = np.asarray(bb, np.float64)
    return float(np.abs(a - bb).max())


def parity(primals, cot_y):
    """Return (worst_vs_gold, per-grad dict, all_pass). Chunked vjp vs fp32 GOLD."""
    def fwd_y_chunked(x, B, C, z, A, dt, D, h0):
        out = mamba3_mimo_apply_with_state_path_c_fwd_path_c_bwd(x, B, C, z, A, dt, D, h0)
        return out[0]

    _, grads_chunked = mx.vjp(fwd_y_chunked, primals, (cot_y,))
    mx.eval(*grads_chunked)
    x, B, C, z, A, dt, D, h0 = primals
    grads_gold = mamba3_mimo_bwd_metal(cot_y, x, B, C, z, A, dt, D, h0, backend="mlx")
    mx.eval(*grads_gold)

    per = {}
    worst = 0.0
    for nm, gc, gg in zip(NAMES, grads_chunked, grads_gold):
        d = maxdiff(np.array(gc.astype(mx.float32)), np.array(gg.astype(mx.float32)))
        per[nm] = d
        worst = max(worst, d)
    return worst, per, all(v < 1e-3 for v in per.values())


def bench_chunked(primals, cot_y):
    def f(x, B, C, z, A, dt, D, h0):
        out = mamba3_mimo_apply_with_state_path_c_fwd_path_c_bwd(x, B, C, z, A, dt, D, h0)
        return out[0]
    times = []
    for i in range(WARMUP + ITERS):
        t0 = time.perf_counter()
        _, g = mx.vjp(f, primals, (cot_y,))
        mx.eval(*g)
        t1 = time.perf_counter()
        if i >= WARMUP:
            times.append(t1 - t0)
    return med_us(times)


def bench_msl(primals, cot_y):
    x, B, C, z, A, dt, D, h0 = primals
    times = []
    for i in range(WARMUP + ITERS):
        t0 = time.perf_counter()
        y, h_last = mamba3_mimo_fwd_path_c(x, B, C, z, A, dt, D, h0)
        g = mamba3_mimo_bwd_metal(cot_y, x, B, C, z, A, dt, D, h0, backend="metal")
        mx.eval(y, h_last, *g)
        t1 = time.perf_counter()
        if i >= WARMUP:
            times.append(t1 - t0)
    return med_us(times)


def main():
    global _PEAK
    print(f"=== SEQLEN SWEEP chunked-vs-MSL bwd e2e (b={B_BATCH} H={H} P={P} N={N} "
          f"chunk={CHUNK}) median of {ITERS} (warmup {WARMUP}) memguard70=ON ===")
    print(f"SEQLENS={SEQLENS}  CPPMEGA_MAMBA3_BWD_SEQ_CHUNK={os.environ.get('CPPMEGA_MAMBA3_BWD_SEQ_CHUNK')}")
    rows = []
    max_feasible = None
    for seq in SEQLENS:
        nchunks = seq // CHUNK
        rss_before = _rss_kb()
        try:
            primals, cot_y = build_inputs(seq)
            c_us = bench_chunked(primals, cot_y)
            m_us = bench_msl(primals, cot_y)
        except Exception as e:  # report and stop — do NOT silently degrade
            print(f"[seq={seq} nchunks={nchunks}] RAISED during bench: {type(e).__name__}: {e}")
            print(f"  -> max feasible seqlen = {max_feasible}")
            break
        ratio = c_us / m_us
        rss_after = _rss_kb()
        peak_here = max(rss_after, _PEAK)
        rows.append(dict(seq=seq, nchunks=nchunks, chunked_us=c_us, msl_us=m_us,
                         ratio=ratio, peak_rss_gb=peak_here / 1048576.0))
        max_feasible = seq
        print(f"[seq={seq:5d} nchunks={nchunks:3d}] chunked={c_us:10.1f}us  MSL={m_us:10.1f}us  "
              f"chunked/MSL={ratio:6.3f}x  peakRSS~{peak_here/1048576.0:.2f}GB "
              f"(rss_before~{rss_before/1048576.0:.2f}GB)")
        # free before next seqlen
        del primals, cot_y
        mx.clear_cache()

    print("\n--- RATIO TABLE (seq -> chunked us / MSL us / ratio / peakRSS) ---")
    for r in rows:
        print(f"  seq={r['seq']:5d} nchunks={r['nchunks']:3d}  chunked={r['chunked_us']:10.1f}us  "
              f"MSL={r['msl_us']:10.1f}us  ratio={r['ratio']:.3f}x  peakRSS={r['peak_rss_gb']:.2f}GB")

    # crossover: first seqlen where chunked <= MSL (ratio <= 1.0)
    crossover = None
    for r in rows:
        if r["ratio"] <= 1.0:
            crossover = r["seq"]
            break
    print(f"\nCROSSOVER seqlen (chunked <= MSL): {crossover if crossover is not None else 'NONE'}")
    print(f"MAX FEASIBLE seqlen under memguard70: {max_feasible}")

    # bit-correct validation at largest feasible (and crossover if distinct)
    val_seqs = []
    if max_feasible is not None:
        val_seqs.append(max_feasible)
    if crossover is not None and crossover not in val_seqs:
        val_seqs.append(crossover)
    bit_results = {}
    for vs in val_seqs:
        primals, cot_y = build_inputs(vs)
        worst, per, ok = parity(primals, cot_y)
        bit_results[vs] = (worst, ok)
        per_s = " ".join(f"{k}={v:.2e}" for k, v in per.items())
        print(f"\nBIT-CORRECT @ seq={vs}: WORST |chunked-GOLD|={worst:.3e}  ALL_8<1e-3={ok}")
        print(f"  per-grad: {per_s}")
        del primals, cot_y
        mx.clear_cache()

    print(f"\nPEAK_RSS_KB={_PEAK} (~{_PEAK / 1048576.0:.3f}GB) memguard70=ON")
    print("SWEEP_DONE RC=0")


if __name__ == "__main__":
    main()

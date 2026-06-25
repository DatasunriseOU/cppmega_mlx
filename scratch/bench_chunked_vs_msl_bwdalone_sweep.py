"""SEQLEN SWEEP (BWD-ALONE): chunked Path-C bwd vs MSL bwd, forward pre-stashed.

Complements bench_chunked_vs_msl_seqlen_sweep.py (which times e2e fwd+bwd). Here
the forward artifacts (cb/dA_cumsum/prev_states/y_skip) are computed ONCE per
seqlen OUTSIDE the timed region (they are forward outputs, not part of the bwd),
so ONLY the backward kernels are timed:
  (1) chunked bwd ALONE = _mamba3_chunked_backward_path_c(..., stash)
  (2) MSL bwd ALONE     = mamba3_mimo_bwd_metal(backend='metal')  [SEQ_CHUNK=0]

This is the perf_mlxroute_split.py [1] section extended to the seqlen sweep, to
reconcile the early "ratio shrinks with seqlen" data (that was bwd-alone) and to
answer the crossover question for the backward in isolation.

memguard 70 stays alive; per-seqlen peak RSS tracked. RULE #1: no fabrication, no
silent fallback (ineligible kernel RAISES).
"""
from __future__ import annotations

import os
import sys
import threading
import time

os.environ["CPPMEGA_MAMBA3_BWD_SEQ_CHUNK"] = "0"

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
            sys.stderr.write(f"[memguard70] KILL self rss_kb={r} (~{r // 1048576}GB) > 70GB\n")
            sys.stderr.flush()
            os._exit(137)
        time.sleep(0.25)


threading.Thread(target=_guard, daemon=True).start()
sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx")

import numpy as np  # noqa: E402
import mlx.core as mx  # noqa: E402

from cppmega_mlx.nn._tilelang.mamba3_path_c import (  # noqa: E402
    mamba3_mimo_fwd_path_c,
    _mamba3_chunked_backward_path_c,
    _mamba3_chunked_fwd_intermediates_path_c,
    _mamba3_pre_gate_yskip_path_c,
)
from cppmega_mlx.nn._tilelang.mamba3 import mamba3_mimo_bwd_metal  # noqa: E402

B_BATCH, H, P, N, CHUNK = 1, 128, 64, 64, 64
SEQLENS = [int(s) for s in os.environ.get("SWEEP_SEQLENS", "128,512,1024,2048,4096").split(",")]
ITERS = int(os.environ.get("BENCH_ITERS", "15"))
WARMUP = int(os.environ.get("BENCH_WARMUP", "5"))


def build_inputs(seq, seed=0):
    rng = np.random.RandomState(seed)
    b = B_BATCH

    def f32(*shape, s=0.1):
        return mx.array((rng.randn(*shape) * s).astype(np.float32))

    x = f32(b, seq, H, P)
    B = f32(b, seq, H, N)
    C = f32(b, seq, H, N)
    z = f32(b, seq, H, P, s=0.5)
    A_head = (-rng.rand(H)).astype(np.float32)
    A = mx.array(np.broadcast_to(A_head[None, None, :], (b, seq, H)).copy())
    dt = mx.array((rng.rand(b, seq, H) * 0.05).astype(np.float32))
    D = mx.array((rng.randn(H)).astype(np.float32))
    h0 = f32(b, H, P, N)
    cot_y = mx.array((rng.randn(b, seq, H, P) * 0.1).astype(np.float32))
    return (x, B, C, z, A, dt, D, h0), cot_y


def med_us(times):
    return float(np.median(np.asarray(times, np.float64))) * 1e6


def main():
    print(f"=== BWD-ALONE SEQLEN SWEEP (forward pre-stashed) (b={B_BATCH} H={H} P={P} "
          f"N={N} chunk={CHUNK}) median of {ITERS} (warmup {WARMUP}) memguard70=ON ===")
    rows = []
    max_feasible = None
    for seq in SEQLENS:
        nchunks = seq // CHUNK
        try:
            primals, cot_y = build_inputs(seq)
            x, B, C, z, A, dt, D, h0 = primals
            # forward stash (NOT timed)
            cb, dA_cumsum, prev_states = _mamba3_chunked_fwd_intermediates_path_c(x, B, C, A, dt, h0)
            mx.eval(cb, dA_cumsum, prev_states)
            y_skip = _mamba3_pre_gate_yskip_path_c(x, B, C, A, dt, D, h0)
            y_stash = mx.contiguous(y_skip.astype(mx.float16))
            mx.eval(y_stash)

            def run_chunked():
                return _mamba3_chunked_backward_path_c(
                    cot_y, x, B, C, z, A, dt, D, h0,
                    cb=cb, dA_cumsum=dA_cumsum, prev_states=prev_states, y=y_stash,
                )

            def run_msl():
                return mamba3_mimo_bwd_metal(cot_y, x, B, C, z, A, dt, D, h0, backend="metal")

            tc = []
            for i in range(WARMUP + ITERS):
                t0 = time.perf_counter()
                g = run_chunked()
                mx.eval(*g)
                if i >= WARMUP:
                    tc.append(time.perf_counter() - t0)
            tm = []
            for i in range(WARMUP + ITERS):
                t0 = time.perf_counter()
                g = run_msl()
                mx.eval(*g)
                if i >= WARMUP:
                    tm.append(time.perf_counter() - t0)
            c_us, m_us = med_us(tc), med_us(tm)
        except Exception as e:
            print(f"[seq={seq} nchunks={nchunks}] RAISED: {type(e).__name__}: {e}")
            print(f"  -> max feasible seqlen = {max_feasible}")
            break
        ratio = c_us / m_us
        peak_here = max(_rss_kb(), _PEAK)
        rows.append(dict(seq=seq, nchunks=nchunks, c=c_us, m=m_us, ratio=ratio,
                         peak_gb=peak_here / 1048576.0))
        max_feasible = seq
        print(f"[seq={seq:5d} nchunks={nchunks:3d}] chunked_bwd={c_us:10.1f}us  MSL_bwd={m_us:10.1f}us  "
              f"chunked/MSL={ratio:6.3f}x  peakRSS~{peak_here/1048576.0:.2f}GB")
        del primals, cot_y, cb, dA_cumsum, prev_states, y_stash
        mx.clear_cache()

    print("\n--- BWD-ALONE RATIO TABLE ---")
    for r in rows:
        print(f"  seq={r['seq']:5d} nchunks={r['nchunks']:3d}  chunked_bwd={r['c']:10.1f}us  "
              f"MSL_bwd={r['m']:10.1f}us  ratio={r['ratio']:.3f}x  peakRSS={r['peak_gb']:.2f}GB")
    crossover = next((r["seq"] for r in rows if r["ratio"] <= 1.0), None)
    print(f"\nBWD-ALONE CROSSOVER (chunked_bwd <= MSL_bwd): {crossover if crossover is not None else 'NONE'}")
    print(f"MAX FEASIBLE seqlen: {max_feasible}")
    print(f"PEAK_RSS_KB={_PEAK} (~{_PEAK/1048576.0:.3f}GB) memguard70=ON")
    print("BWDALONE_SWEEP_DONE RC=0")


if __name__ == "__main__":
    main()

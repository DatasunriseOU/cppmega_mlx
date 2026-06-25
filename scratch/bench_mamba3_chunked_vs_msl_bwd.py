"""MEASURED A/B bench: chunked Path-C backward (B2->B1->B0) vs production MSL
mamba3_mimo_bwd_metal, SAME inputs, warmed + median over >=20 iters.

RULE #1: no fabrication, no silent fallback. If a Metal kernel is not eligible
we RAISE (we never silently time a pure-MLX path and call it "MSL").

Path A = chained chunked backward via the REAL build_*_bwd_metal builders (the
exact protocol the passing test tests/test_mamba3_chunked_backward_b0b1b2.py
exercises). Forward prerequisites (cb, dA, prev, summ, y) are computed ONCE
outside the timed region (they are forward artifacts, not part of the bwd).

Path B = production MSL mamba3_mimo_bwd_metal forced onto the NON-chunked Metal
kernel (CPPMEGA_MAMBA3_BWD_SEQ_CHUNK=0, backend='metal'), B/C broadcast
groups->heads so it is the SAME logical problem.

Self-imposed 70GB RSS watchdog (memguard 70) + peak-RSS tracking.
"""
from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np

# --- memguard 70: self-imposed 70GB RSS killer for THIS process -------------
_MEMGUARD_LIMIT_KB = 70 * 1024 * 1024  # 70 GiB
_PEAK_RSS_KB = 0


def _rss_kb() -> int:
    import resource
    # macOS ru_maxrss is bytes; convert to KB.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)


def _memguard_thread():
    global _PEAK_RSS_KB
    while True:
        r = _rss_kb()
        if r > _PEAK_RSS_KB:
            _PEAK_RSS_KB = r
        if r > _MEMGUARD_LIMIT_KB:
            sys.stderr.write(
                f"[memguard70] KILL self rss_kb={r} (~{r//1048576}GB) > 70GB\n"
            )
            sys.stderr.flush()
            os._exit(137)
        time.sleep(0.25)


threading.Thread(target=_memguard_thread, daemon=True).start()

sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx")
sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx/scratch")

import torch  # noqa: E402
from einops import rearrange  # noqa: E402

import mlx.core as mx  # noqa: E402

assert torch.backends.mps.is_available(), "Metal (mps) backend required"

from cppmega_mlx.nn._tilelang.mamba3_chunked_precompute_core import (  # noqa: E402
    build_chunk_precompute_metal,
    build_inter_chunk_recur_metal,
)
from cppmega_mlx.nn._tilelang.mamba3_chunked_backward_core import (  # noqa: E402
    build_chunk_scan_combine_bwd_metal,
    build_inter_chunk_recur_bwd_metal,
    build_chunk_precompute_bwd_metal,
)
import mamba3_chunked_backward_proto as bp  # noqa: E402

DEV = "mps"
ITERS = int(os.environ.get("BENCH_ITERS", "30"))
WARMUP = int(os.environ.get("BENCH_WARMUP", "5"))


def _median_us(times):
    return float(np.median(np.asarray(times, np.float64)) * 1e6)


def th(a, d=torch.float32):
    return torch.tensor(a, device=DEV, dtype=d).contiguous()


def build_inputs(b, seqlen, chunk, G, H, P, N, seed=0):
    rng = np.random.RandomState(seed)
    return dict(
        x=(rng.randn(b, seqlen, H, P) * 0.1).astype(np.float32),
        B=(rng.randn(b, seqlen, G, N) * 0.1).astype(np.float32),
        C=(rng.randn(b, seqlen, G, N) * 0.1).astype(np.float32),
        A=(-rng.rand(H)).astype(np.float32),
        dt=(rng.rand(b, seqlen, H) * 0.05).astype(np.float32),
        D=(rng.randn(H)).astype(np.float32),
        h0=(rng.randn(b, H, P, N) * 0.1).astype(np.float32),
        dout=(rng.randn(b, seqlen, H, P) * 0.1).astype(np.float32),
        z=(rng.randn(b, seqlen, H, P) * 0.5).astype(np.float32),
    )


def forward_prereqs(d, b, seqlen, chunk, G, H, P, N):
    """Compute the forward artifacts the chunked bwd consumes (cb, dA, prev, summ,
    y). NOT part of the timed bwd region."""
    nchunks = seqlen // chunk
    # proto forward for y (the D-skip residual target the bwd reads)
    def mxa(a):
        return mx.array(a)
    log_decay = (mxa(d["A"]).reshape(1, 1, H) * mxa(d["dt"])).reshape(b, seqlen, H, 1, 1)
    B_h = mx.broadcast_to(
        mxa(d["B"])[:, :, :, None, :], (b, seqlen, G, H // G, N)
    ).reshape(b, seqlen, H, N)
    inp = mxa(d["dt"])[:, :, :, None, None] * (mxa(d["x"])[..., None] * B_h[:, :, :, None, :])
    C_proto = mx.broadcast_to(
        mxa(d["C"])[:, :, :, None, :], (b, seqlen, G, H // G, N)
    ).reshape(b, seqlen, H, N)
    out, fs, cache = bp.chunked_mamba3_forward_full(
        log_decay, inp, C_proto, mxa(d["x"]), mxa(d["z"]), mxa(d["D"]), mxa(d["h0"]),
        chunk_size=chunk,
    )
    y_np = np.array(cache["y"])

    k_f0 = build_chunk_precompute_metal(b, seqlen, chunk, G, H, P, N)
    cb = torch.zeros(b, nchunks, G, chunk, chunk, device=DEV, dtype=torch.float16)
    dA = torch.zeros(b, H, nchunks, chunk, device=DEV, dtype=torch.float16)
    summ = torch.zeros(b, nchunks, H, P, N, device=DEV, dtype=torch.float32)
    k_f0(th(d["x"], torch.float16), th(d["B"], torch.float16), th(d["C"], torch.float16),
         th(d["A"], torch.float16), th(d["dt"], torch.float16), cb, dA, summ)
    torch.mps.synchronize()
    k_f1 = build_inter_chunk_recur_metal(b, seqlen, chunk, G, H, P, N)
    prev = torch.zeros(b, nchunks, H, P, N, device=DEV, dtype=torch.float32)
    fst = torch.zeros(b, H, P, N, device=DEV, dtype=torch.float32)
    k_f1(summ.contiguous(), dA.contiguous(), th(d["h0"], torch.float32), prev, fst)
    torch.mps.synchronize()
    return dict(cb=cb, dA=dA, prev=prev, summ=summ, y=y_np, nchunks=nchunks)


def make_pathA_runners(d, pre, b, seqlen, chunk, G, H, P, N):
    """Return (run_b2, run_b1, run_b0, run_all). Each runs its kernel on fresh
    pre-zeroed buffers and synchronizes. Buffers/handoffs are allocated here once
    where they are pure outputs; chained handoffs are produced by the prior stage."""
    nchunks = pre["nchunks"]
    cb, dA, prev, y = pre["cb"], pre["dA"], pre["prev"], pre["y"]
    dt_k = rearrange(th(d["dt"], torch.float16), "b (c s) hh -> b hh c s", c=nchunks).contiguous()
    dt_k_f32 = rearrange(th(d["dt"], torch.float32), "b (c s) hh -> b hh c s", c=nchunks).contiguous()

    k_b2 = build_chunk_scan_combine_bwd_metal(b, seqlen, chunk, G, H, P, N)
    k_b1 = build_inter_chunk_recur_bwd_metal(b, seqlen, chunk, G, H, P, N)
    k_b0 = build_chunk_precompute_bwd_metal(b, seqlen, chunk, G, H, P, N)

    # B2 inputs (constant across iters)
    dout_t = th(d["dout"]); x_t = th(d["x"]); z_t = th(d["z"]); C_t = th(d["C"])
    B_t = th(d["B"]); D_t = th(d["D"]); y_t = th(y)
    A_t = th(d["A"])

    def alloc_b2_out():
        return (
            torch.zeros(b, seqlen, H, N, device=DEV, dtype=torch.float32),       # dC
            torch.zeros(b, seqlen, H, P, device=DEV, dtype=torch.float32),       # dx
            torch.zeros(b, seqlen, H, P, device=DEV, dtype=torch.float32),       # dz
            torch.zeros(b, nchunks, H, P, N, device=DEV, dtype=torch.float32),   # dchunk
            torch.zeros(b, seqlen, H, P, N, device=DEV, dtype=torch.float32),    # dinp_diag
            torch.zeros(b, H, nchunks, chunk, device=DEV, dtype=torch.float32),  # dA_y
            torch.zeros(H, device=DEV, dtype=torch.float32),                     # dD
        )

    def run_b2():
        dC_m, dx_m, dz_m, dchunk, dinp_diag, dA_y, dD_m = alloc_b2_out()
        k_b2(dout_t, cb.contiguous(), x_t, z_t, dt_k, dA.contiguous(),
             C_t, B_t, prev.contiguous(), D_t, y_t,
             dC_m, dx_m, dz_m, dchunk, dinp_diag, dA_y, dD_m)
        torch.mps.synchronize()
        return dx_m, dchunk, dinp_diag, dA_y

    # produce a real dchunk/dinp_diag/dA_y once for B1/B0 timing inputs
    _dx_seed, dchunk_seed, dinp_seed, dA_y_seed = run_b2()

    def run_b1():
        dh_last = torch.zeros(b, H, P, N, device=DEV, dtype=torch.float32)
        dstates = torch.zeros(b, nchunks, H, P, N, device=DEV, dtype=torch.float32)
        dh0_m = torch.zeros(b, H, P, N, device=DEV, dtype=torch.float32)
        dA_tail = torch.zeros(b, H, nchunks, chunk, device=DEV, dtype=torch.float32)
        k_b1(dchunk_seed.contiguous(), dA.contiguous(), dh_last, prev.contiguous(),
             dstates, dh0_m, dA_tail)
        torch.mps.synchronize()
        return dstates, dA_tail

    dstates_seed, dA_tail_seed = run_b1()

    def run_b0():
        dx_full = _dx_seed.clone()
        dB_m = torch.zeros(b, seqlen, H, N, device=DEV, dtype=torch.float32)
        dlog_m = torch.zeros(b, seqlen, H, device=DEV, dtype=torch.float32)
        ddt_m = torch.zeros(b, seqlen, H, device=DEV, dtype=torch.float32)
        k_b0(dstates_seed.contiguous(), dinp_seed.contiguous(), dA_y_seed.contiguous(),
             dA_tail_seed.contiguous(), dA.contiguous(), x_t, B_t, dt_k_f32, A_t,
             dx_full, dB_m, dlog_m, ddt_m)
        torch.mps.synchronize()
        return dx_full

    def run_all():
        # full chained B2 -> B1 -> B0 (every iter, fresh buffers, single sync at end)
        dC_m, dx_m, dz_m, dchunk, dinp_diag, dA_y, dD_m = alloc_b2_out()
        k_b2(dout_t, cb.contiguous(), x_t, z_t, dt_k, dA.contiguous(),
             C_t, B_t, prev.contiguous(), D_t, y_t,
             dC_m, dx_m, dz_m, dchunk, dinp_diag, dA_y, dD_m)
        dh_last = torch.zeros(b, H, P, N, device=DEV, dtype=torch.float32)
        dstates = torch.zeros(b, nchunks, H, P, N, device=DEV, dtype=torch.float32)
        dh0_m = torch.zeros(b, H, P, N, device=DEV, dtype=torch.float32)
        dA_tail = torch.zeros(b, H, nchunks, chunk, device=DEV, dtype=torch.float32)
        k_b1(dchunk.contiguous(), dA.contiguous(), dh_last, prev.contiguous(),
             dstates, dh0_m, dA_tail)
        dx_full = dx_m.clone()
        dB_m = torch.zeros(b, seqlen, H, N, device=DEV, dtype=torch.float32)
        dlog_m = torch.zeros(b, seqlen, H, device=DEV, dtype=torch.float32)
        ddt_m = torch.zeros(b, seqlen, H, device=DEV, dtype=torch.float32)
        k_b0(dstates.contiguous(), dinp_diag.contiguous(), dA_y.contiguous(),
             dA_tail.contiguous(), dA.contiguous(), x_t, B_t, dt_k_f32, A_t,
             dx_full, dB_m, dlog_m, ddt_m)
        torch.mps.synchronize()

    return run_b2, run_b1, run_b0, run_all


def make_pathB_runner(d, b, seqlen, G, H, P, N):
    """Production MSL mamba3_mimo_bwd_metal, NON-chunked Metal kernel forced."""
    os.environ["CPPMEGA_MAMBA3_BWD_SEQ_CHUNK"] = "0"  # disable internal seq-chunking
    from cppmega_mlx.nn._tilelang.mamba3 import (
        mamba3_mimo_bwd_metal,
        mamba3_mimo_metal_status,
        _mamba3_mimo_bwd_metal_kernel,
    )

    def mxf(a):
        return mx.array(a.astype(np.float32))

    # broadcast B/C groups -> heads so MSL sees the SAME logical problem
    B_h = np.broadcast_to(d["B"][:, :, :, None, :], (b, seqlen, G, H // G, N)).reshape(b, seqlen, H, N)
    C_h = np.broadcast_to(d["C"][:, :, :, None, :], (b, seqlen, G, H // G, N)).reshape(b, seqlen, H, N)
    # MSL kernel computes log_decay = A * dt with A of shape (b, seq, H). The
    # chunked path uses log_decay = A_perhead.reshape(1,1,H) * dt. Broadcast the
    # per-head A to (b, seq, H) so BOTH paths share the SAME log-decay (apples-to-
    # apples) — NOT a fabrication, it is the identical mathematical input.
    A_bsh = np.broadcast_to(d["A"][None, None, :], (b, seqlen, H)).astype(np.float32)
    dy = mxf(d["dout"]); x = mxf(d["x"]); Bm = mxf(np.ascontiguousarray(B_h))
    Cm = mxf(np.ascontiguousarray(C_h)); z = mxf(d["z"]); A = mxf(np.ascontiguousarray(A_bsh))
    dt = mxf(d["dt"]); D = mxf(d["D"]); h0 = mxf(d["h0"])

    st = mamba3_mimo_metal_status(x)
    if not st.available:
        raise RuntimeError(f"RULE#1: MSL bwd Metal kernel NOT available: {st}")
    # RULE#1 probe: prove the non-chunked Metal kernel actually returns (not None)
    probe = _mamba3_mimo_bwd_metal_kernel(dy, x, Bm, Cm, z, A, dt, D, h0)
    if probe is None:
        raise RuntimeError("RULE#1: non-chunked MSL Metal bwd returned None (not eligible)")

    def run():
        grads = mamba3_mimo_bwd_metal(dy, x, Bm, Cm, z, A, dt, D, h0, backend="metal")
        mx.eval(*grads)
        return grads

    return run


def time_runner(run, iters, warmup, label):
    for _ in range(warmup):
        run()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        run()
        times.append(time.perf_counter() - t0)
    us = _median_us(times)
    print(f"  [{label}] median={us:.1f}us over {iters} iters (min={min(times)*1e6:.1f} max={max(times)*1e6:.1f})")
    return us


def bench_config(name, b, seqlen, chunk, G, H, P, N):
    print(f"\n===== CONFIG {name}: b={b} S={seqlen} chunk={chunk} G={G} H={H} P={P} N={N} =====")
    d = build_inputs(b, seqlen, chunk, G, H, P, N)
    pre = forward_prereqs(d, b, seqlen, chunk, G, H, P, N)
    run_b2, run_b1, run_b0, run_all = make_pathA_runners(d, pre, b, seqlen, chunk, G, H, P, N)

    print("-- Path A: chunked B2->B1->B0 --")
    a_us = time_runner(run_all, ITERS, WARMUP, "A=chained B2->B1->B0")
    b2_us = time_runner(run_b2, ITERS, WARMUP, "  B2 (chunk_scan_combine_bwd / P1-class dstates GEMM)")
    b1_us = time_runner(run_b1, ITERS, WARMUP, "  B1 (inter_chunk_recur_bwd)")
    b0_us = time_runner(run_b0, ITERS, WARMUP, "  B0 (chunk_precompute_bwd)")

    print("-- Path B: production MSL mamba3_mimo_bwd_metal (non-chunked Metal) --")
    runB = make_pathB_runner(d, b, seqlen, G, H, P, N)
    b_us = time_runner(runB, ITERS, WARMUP, "B=MSL mamba3_mimo_bwd_metal")

    ratio = a_us / b_us
    # NOTE: in this chunked design B2=chunk_scan_combine_bwd is the P1-class dstates
    # GEMM (dstates=dot(dout,c)); the prompt labels it "B0" as the P1 hotspot. Report
    # both names. dominator = max of the three stages.
    stage = {"B2_scan_combine": b2_us, "B1_inter_recur": b1_us, "B0_precompute": b0_us}
    dom = max(stage, key=stage.get)
    print(f"\n  RESULT {name}: A={a_us:.1f}us B={b_us:.1f}us  A/B={ratio:.2f}x")
    print(f"  stage split: B2={b2_us:.1f}us B1={b1_us:.1f}us B0={b0_us:.1f}us  dominator={dom}")
    return dict(name=name, A_us=a_us, B_us=b_us, ratio=ratio,
                B2_us=b2_us, B1_us=b1_us, B0_us=b0_us, dominator=dom,
                b2_is_p1_hotspot=(dom == "B2_scan_combine"))


def main():
    results = []
    # (1) tested clean A/B
    results.append(bench_config("tested-S512", 1, 512, 64, 1, 2, 64, 16))
    # (2) nam56r config-exact: chunk=64, nh=128, head_dim=64(P), state_dim=64(N),
    #     num_groups=8(G). Use S=512 so nchunks=8 like the tested config.
    results.append(bench_config("nam56r-H128", 1, 512, 64, 8, 128, 64, 64))

    print("\n\n========== SUMMARY ==========")
    for r in results:
        print(f"{r['name']}: A={r['A_us']:.1f}us B={r['B_us']:.1f}us A/B={r['ratio']:.2f}x | "
              f"B2={r['B2_us']:.1f} B1={r['B1_us']:.1f} B0={r['B0_us']:.1f} dom={r['dominator']} "
              f"b2_is_P1_hotspot={r['b2_is_p1_hotspot']}")
    print(f"\nPEAK_RSS_KB={_PEAK_RSS_KB} (~{_PEAK_RSS_KB/1048576:.3f}GB)  memguard70=ON")
    print("RC=0")


if __name__ == "__main__":
    main()

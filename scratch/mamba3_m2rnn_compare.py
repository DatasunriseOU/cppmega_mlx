"""LEVER 5 — mamba3/m2rnn OURS-vs-cppmega per-region compare bench (gb10 / sm_121).

WHY THIS EXISTS
---------------
The SAME mamba3 + m2rnn model is faster in cppmega than in cppmega.mlx path_c
(docs/RELAX-GRAPH-VS-MEGATRON.md §17: path_c GO ≈907 tok/s @8L / ≈298 @28L; the
SAME model in cppmega hits ~3692 tok/s @4437 ms/iter on gb10, tensorwise). The
prior read-only investigation ranked the cause as a RUNTIME/EXECUTION-ENGINE gap
(torch+CUDA-graph device-resident step vs Relax-VM many-launch + an abstract numpy
backward/adam/loss), NOT the SSD scan itself. THIS bench measures that claim
HONESTLY by timing, at the SAME problem config, the per-region cost of:

  * OUR path_c implementation — the gridded CUDA SSD kernels that actually run the
    mamba3 forward/backward on gb10:
        F0 = mamba3_chunk_precompute        (grid, no scan dep)
        F1 = mamba3_inter_chunk_recur       (O(S/C) sequential recurrence)
        F2 = mamba3_chunk_scan_combine      (grid scan+combine -> y)
        B2 = mamba3_chunk_scan_combine_bwd  (grid)
        B1 = mamba3_inter_chunk_recur_bwd   (O(S/C) reverse recurrence)
        B0 = mamba3_chunk_precompute_bwd    (grid)
    plus the HOST/LAUNCH-STAGING overhead between regions (the §17 suspect).

  * The cppmega implementation of the SAME mamba3 model — the Megatron SSD Triton
    kernel ``mamba_chunk_scan_combined`` (mamba_ssm.ops.triton.ssd_combined), which
    is the EXACT op cppmega/megatron/mamba3_mixer.py drives. We time its fwd and
    its autograd backward (same chunked-SSD math, fused Triton, CUDA-graph-friendly).

  * The cppmega M2RNN — the fused Triton ``m2rnn_scan_triton``
    (cppmega/megatron/m2rnn_triton.py), fwd + autograd backward, at the SAME
    (B,S,H,K,V) shape the model uses.

  * OUR M2RNN — HONEST NOTE (RULE #1): cppmega_mlx M2RNN Path-C
    (cppmega_mlx/nn/_tilelang/m2rnn_path_c.py) is a METAL-ONLY MSL-through-MLX
    surface; it has NO CUDA twin. So on gb10 there is no "ours-m2rnn CUDA kernel"
    to time. We DO NOT fabricate one. We report ours-m2rnn = NOT-RUNNABLE-ON-CUDA
    with the reason, and we time the cppmega m2rnn alone so the GB10 phase still
    sees the absolute m2rnn region cost (and can compare it to the mamba region
    cost to confirm the §17 finding that mamba/m2rnn are a MINORITY of the iter).

WHAT THIS LOCATES
-----------------
A machine-parseable RESULT json with, per region, the MEASURED median ms/call for
OURS and (where runnable) cppmega, the ours/cppmega RATIO, and the host-staging
delta (the wall the chained ours-fwd / ours-bwd spends OUTSIDE the kernels — the
launch-overhead term the §17 hypothesis blames). This pins, with numbers, WHERE
path_c loses time vs cppmega on the identical model:
  - if ours-F2 ≈ cppmega-fwd-kernel but the ours-chain wall ≫ sum(F0+F1+F2),
    the loser is HOST/LAUNCH STAGING (H1/H4), not the scan;
  - if ours-F2 ≫ cppmega-fwd, the loser is the KERNEL (refutes the §17 claim).
RULE #1: every printed number is MEASURED on this box or explicitly labelled
ABSENT (NOT-RUNNABLE) — no extrapolated/fabricated speedup, no silent fallback.
Any compile/run failure RAISES with WHERE+WHAT and is recorded as a hard error in
the RESULT (never swallowed into a degraded path).

RUN (gb10, single-owner; ensure exclusive GPU ownership + >105 GB free first):
  ensure_nvrtc_builtins_path  (cppmega_mlx/_gb10_nvrtc_env.py) BEFORE torch import
  PYTHONPATH=<cppmega_mlx>:<tvm/python>:<tvm-ffi/python> \
  TVM_LIBRARY_PATH=<tilelang/build/lib> \
  /home/dave/cppmega-venv/bin/python scratch/mamba3_m2rnn_compare.py [--prod] [--bs4] \
      [--no-mamba-cppmega] [--no-m2rnn-cppmega] [--steps N]

  --prod   local_gb10_quarter mamba tile: S=4096 c=64 g=8 H=112 P=64 N=64 (the §17
           config). Without --prod a small fast cfg is used for off-box smoke.
  --bs4    micro-batch 1 -> 4 (the Megatron 16384-tok/step batch axis). Requires
           --prod (RULE #1: never silently ignored).
"""

# CRITICAL: no ``from __future__ import annotations`` is needed here (this is a
# plain bench, not a TileLang prim_func module), but we also DO NOT import the
# scan-core modules' prim builders at module top — they are imported lazily inside
# the timed functions exactly as the probe scripts do, so a missing tvm/torch on
# this Mac fails at the call site with WHERE+WHAT, not at import.

import json
import os
import sys
import time
import traceback


# --------------------------------------------------------------------------- #
# Problem configs. The mamba tile mirrors scratch/probe_chunked_*_cuda_gb10.py
# (the §17 prod cfg). The m2rnn shape is the SAME (B,S) with the model's head/dim
# layout so the m2rnn region cost is comparable to the mamba region cost.
# --------------------------------------------------------------------------- #
def mamba_cfg(prod: bool, batch: int) -> dict:
    if prod:
        # local_gb10_quarter: S=4096 c=64 H=112 P=64 N=64 g=8 (== §17 / the probes)
        return dict(batch=batch, seqlen=4096, chunk=64, ngroups=8, nheads=112,
                    headdim=64, dstate=64)
    # small smoke cfg (fast compile/run for off-box plumbing checks)
    return dict(batch=batch, seqlen=512, chunk=64, ngroups=1, nheads=8,
                headdim=64, dstate=16)


def m2rnn_cfg(prod: bool, batch: int) -> dict:
    """M2RNN shape at the SAME (B,S). H/K/V picked to be a comparable working set
    to the mamba region (the model's R layers run alongside the M layers). These
    are bench dims, NOT a parity contract — labelled MEASURED-as-configured."""
    if prod:
        return dict(batch=batch, seqlen=4096, heads=8, k_dim=128, v_dim=128)
    return dict(batch=batch, seqlen=512, heads=4, k_dim=64, v_dim=64)


# --------------------------------------------------------------------------- #
# Timing helper (warm + median of N device-synced dispatches). Mirrors the probe
# scripts' _time so the numbers are directly comparable to §15/§17.
# --------------------------------------------------------------------------- #
def _time_cuda(fn, n: int = 20, warm: int = 3):
    import torch
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return {
        "median_ms": ts[len(ts) // 2],
        "min_ms": ts[0],
        "p90_ms": ts[min(len(ts) - 1, int(0.9 * len(ts)))],
        "n": n,
    }


# =========================================================================== #
# OUR path_c arm — the gridded CUDA SSD forward F0/F1/F2 + backward B0/B1/B2.
# Reuses the EXACT reference algebra + invocation from
# scratch/probe_chunked_scan_cuda_gb10.py and probe_chunked_backward_cuda_gb10.py
# (imported as modules so there is ONE source of truth for the kernel call ABI —
# RULE #1: no second/divergent copy of the kernel-call contract).
# =========================================================================== #
def bench_ours_mamba(cfg: dict, *, run_backward: bool = True) -> dict:
    """Time OUR F0/F1/F2 (+ B0/B1/B2) gridded CUDA kernels at ``cfg``.

    Returns per-region median ms + the CHAINED fwd/bwd wall (host-staged, the
    launch-overhead term §17 blames). RAISES with WHERE+WHAT on any compile/run
    failure (recorded as a hard error by the caller)."""
    import numpy as np
    import torch
    from einops import rearrange

    # Reuse the probe references (ONE source of truth for the call ABI).
    from cppmega_mlx.nn._tilelang.mamba3_chunked_precompute_core import (
        build_chunk_precompute_metal, build_inter_chunk_recur_metal,
        chunk_precompute_fwd_grid, inter_chunk_recur_fwd_grid,
    )
    from cppmega_mlx.nn._tilelang.mamba3_chunked_scan_core import (
        build_chunk_scan_combine_metal, chunk_scan_fwd_grid,
    )
    # The probe modules carry the validated eager_precompute / serial_y / VJP gold.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from probe_chunked_scan_cuda_gb10 import eager_precompute  # noqa: E402
    from probe_chunked_backward_cuda_gb10 import serial_y  # noqa: E402

    DEV = "cuda"
    b = cfg["batch"]; S = cfg["seqlen"]; chunk = cfg["chunk"]
    G = cfg["ngroups"]; H = cfg["nheads"]; P = cfg["headdim"]; N = cfg["dstate"]
    nchunks = S // chunk
    out = {"cfg": dict(cfg), "regions": {}, "grids": {}}
    out["grids"]["F0"] = chunk_precompute_fwd_grid(b, S, chunk, G, H, P, N)[0]
    out["grids"]["F1"] = inter_chunk_recur_fwd_grid(b, S, chunk, G, H, P, N)[0]
    out["grids"]["F2"] = chunk_scan_fwd_grid(b, S, chunk, G, H, P, N)[0]

    # ---- seeded inputs (fp16, matching the probe / the fp16 forward cache) ----
    torch.manual_seed(0)
    C = (torch.randn(b, S, G, N, device=DEV) * 0.1).half()
    Bmat = (torch.randn(b, S, G, N, device=DEV) * 0.1).half()
    x = (torch.randn(b, S, H, P, device=DEV) * 0.1).half()
    A = -torch.rand(H, device=DEV).half()
    dt = (torch.rand(b, S, H, device=DEV) * 0.05).half()
    D = torch.randn(H, device=DEV).half()
    h0 = (torch.randn(b, H, P, N, device=DEV) * 0.1).half()
    dt_k = rearrange(dt, "b (c s) hh -> b hh c s", c=nchunks).contiguous()

    # ===================== FORWARD F0 / F1 / F2 =====================
    bd = N if N >= 16 else 16
    k_f0 = build_chunk_precompute_metal(b, S, chunk, G, H, P, N, target="cuda")
    k_f1 = build_inter_chunk_recur_metal(b, S, chunk, G, H, P, N, target="cuda")
    k_f2 = build_chunk_scan_combine_metal(b, S, chunk, G, H, P, N, target="cuda",
                                          block_Dstate=bd)

    cb = torch.zeros(b, nchunks, G, chunk, chunk, device=DEV, dtype=torch.float16)
    dA = torch.zeros(b, H, nchunks, chunk, device=DEV, dtype=torch.float16)
    summ = torch.zeros(b, nchunks, H, P, N, device=DEV, dtype=torch.float32)
    prev = torch.zeros(b, nchunks, H, P, N, device=DEV, dtype=torch.float32)
    fst = torch.zeros(b, H, P, N, device=DEV, dtype=torch.float32)
    y_out = torch.zeros(b, S, H, P, device=DEV, dtype=torch.float16)

    def run_f0():
        k_f0(x.contiguous(), Bmat.contiguous(), C.contiguous(), A.contiguous(),
             dt.contiguous(), cb, dA, summ)

    def run_f1():
        k_f1(summ.contiguous(), dA.contiguous(), h0.float().contiguous(), prev, fst)

    def run_f2():
        k_f2(cb.contiguous(), x.contiguous(), dt_k.contiguous(), dA.contiguous(),
             C.contiguous(), prev.contiguous(), D.contiguous(), y_out)

    # individual region medians
    run_f0(); torch.cuda.synchronize()
    out["regions"]["F0"] = _time_cuda(run_f0)
    run_f1(); torch.cuda.synchronize()
    out["regions"]["F1"] = _time_cuda(run_f1)
    run_f2(); torch.cuda.synchronize()
    out["regions"]["F2"] = _time_cuda(run_f2)

    # CHAINED forward wall (F0->F1->F2 back-to-back, host-staged exactly as the
    # graph drives them) — the host/launch-staging term. If this exceeds
    # F0+F1+F2 medians the gap is launch/host overhead, NOT the kernels.
    def run_fwd_chain():
        run_f0(); run_f1(); run_f2()
    run_fwd_chain(); torch.cuda.synchronize()
    out["fwd_chain"] = _time_cuda(run_fwd_chain)
    out["fwd_region_sum_ms"] = (out["regions"]["F0"]["median_ms"]
                                + out["regions"]["F1"]["median_ms"]
                                + out["regions"]["F2"]["median_ms"])
    out["fwd_host_stage_ms"] = out["fwd_chain"]["median_ms"] - out["fwd_region_sum_ms"]

    if not run_backward:
        return out

    # ===================== BACKWARD B2 / B1 / B0 =====================
    from cppmega_mlx.nn._tilelang.mamba3_chunked_backward_core import (
        build_chunk_scan_combine_bwd_metal, build_inter_chunk_recur_bwd_metal,
        build_chunk_precompute_bwd_metal,
    )
    z = (torch.randn(b, S, H, P, device=DEV) * 0.5).half()
    dout = (torch.randn(b, S, H, P, device=DEV) * 0.1).half()
    # y the backward consumes (the F2-delta ungated SSD output, reused, no replay).
    y_t = serial_y(cb, x, dt_k, dA, C, prev, D, chunk)

    k_b2 = build_chunk_scan_combine_bwd_metal(b, S, chunk, G, H, P, N, target="cuda")
    dC_m = torch.zeros(b, S, H, N, device=DEV, dtype=torch.float32)
    dx_m = torch.zeros(b, S, H, P, device=DEV, dtype=torch.float32)
    dz_m = torch.zeros(b, S, H, P, device=DEV, dtype=torch.float32)
    dchunk = torch.zeros(b, nchunks, H, P, N, device=DEV, dtype=torch.float32)
    dinp_diag = torch.zeros(b, S, H, P, N, device=DEV, dtype=torch.float32)
    dA_y = torch.zeros(b, H, nchunks, chunk, device=DEV, dtype=torch.float32)
    dD_m = torch.zeros(H, device=DEV, dtype=torch.float32)

    def run_b2():
        dC_m.zero_(); dx_m.zero_(); dz_m.zero_(); dchunk.zero_()
        dinp_diag.zero_(); dA_y.zero_(); dD_m.zero_()
        k_b2(dout, cb.contiguous(), x, z, dt_k, dA.contiguous(),
             C, Bmat, prev.contiguous(), D, y_t,
             dC_m, dx_m, dz_m, dchunk, dinp_diag, dA_y, dD_m)
    run_b2(); torch.cuda.synchronize()
    out["regions"]["B2"] = _time_cuda(run_b2)

    k_b1 = build_inter_chunk_recur_bwd_metal(b, S, chunk, G, H, P, N, target="cuda")
    dh_last = torch.zeros(b, H, P, N, device=DEV, dtype=torch.float32)
    dstates = torch.zeros(b, nchunks, H, P, N, device=DEV, dtype=torch.float32)
    dh0_m = torch.zeros(b, H, P, N, device=DEV, dtype=torch.float32)
    dA_tail = torch.zeros(b, H, nchunks, chunk, device=DEV, dtype=torch.float32)

    def run_b1():
        dstates.zero_(); dh0_m.zero_(); dA_tail.zero_()
        k_b1(dchunk.contiguous(), dA.contiguous(), dh_last.contiguous(),
             prev.contiguous(), dstates, dh0_m, dA_tail)
    run_b1(); torch.cuda.synchronize()
    out["regions"]["B1"] = _time_cuda(run_b1)

    k_b0 = build_chunk_precompute_bwd_metal(b, S, chunk, G, H, P, N, target="cuda")
    dx_full = dx_m.clone()
    dB_m = torch.zeros(b, S, H, N, device=DEV, dtype=torch.float32)
    dlog_m = torch.zeros(b, S, H, device=DEV, dtype=torch.float32)
    ddt_m = torch.zeros(b, S, H, device=DEV, dtype=torch.float32)

    def run_b0():
        dx_full.copy_(dx_m)
        dB_m.zero_(); dlog_m.zero_(); ddt_m.zero_()
        k_b0(dstates.contiguous(), dinp_diag.contiguous(), dA_y.contiguous(),
             dA_tail.contiguous(), dA.contiguous(), x, Bmat, dt_k, A,
             dx_full, dB_m, dlog_m, ddt_m)
    run_b0(); torch.cuda.synchronize()
    out["regions"]["B0"] = _time_cuda(run_b0)

    # CHAINED backward wall (B2->B1->B0) — the §17 447.8 ms chain analogue.
    def run_bwd_chain():
        run_b2(); run_b1(); run_b0()
    run_bwd_chain(); torch.cuda.synchronize()
    out["bwd_chain"] = _time_cuda(run_bwd_chain)
    out["bwd_region_sum_ms"] = (out["regions"]["B2"]["median_ms"]
                                + out["regions"]["B1"]["median_ms"]
                                + out["regions"]["B0"]["median_ms"])
    out["bwd_host_stage_ms"] = out["bwd_chain"]["median_ms"] - out["bwd_region_sum_ms"]
    return out


# =========================================================================== #
# cppmega arm — the SAME mamba3 model op: Megatron SSD Triton
# mamba_chunk_scan_combined (the exact kernel cppmega/megatron/mamba3_mixer.py
# drives). fwd + autograd backward, SAME (B,S,H,P,N,chunk) config as ours.
# =========================================================================== #
def bench_cppmega_mamba(cfg: dict) -> dict:
    """Time cppmega's mamba_chunk_scan_combined fwd + autograd bwd at ``cfg``.

    Same chunked-SSD math as OUR F0/F1/F2; this is the apples-to-apples kernel the
    §17 gap is measured against. RAISES (recorded as hard error) if mamba_ssm is
    not installed — RULE #1: we do NOT silently skip it, we surface WHY."""
    import torch
    try:
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    except Exception as exc:  # surface WHERE+WHAT — do not degrade silently
        raise RuntimeError(
            "bench_cppmega_mamba: cannot import mamba_ssm.ops.triton.ssd_combined "
            f"(the Megatron SSD kernel cppmega drives): {exc!r}. Install mamba_ssm "
            "on gb10 to run the cppmega mamba arm; this is NOT a fallback path."
        ) from exc

    DEV = "cuda"
    b = cfg["batch"]; S = cfg["seqlen"]; chunk = cfg["chunk"]
    G = cfg["ngroups"]; H = cfg["nheads"]; P = cfg["headdim"]; N = cfg["dstate"]
    out = {"cfg": dict(cfg), "regions": {}}

    torch.manual_seed(0)
    # mamba_chunk_scan_combined shapes (mamba_ssm convention):
    #   x:(b,S,H,P) dt:(b,S,H) A:(H,) B:(b,S,G,N) C:(b,S,G,N) D:(H,)
    x = torch.randn(b, S, H, P, device=DEV, dtype=torch.float16, requires_grad=True)
    dt = (torch.rand(b, S, H, device=DEV, dtype=torch.float16) * 0.05).requires_grad_(True)
    A = -torch.rand(H, device=DEV, dtype=torch.float32)
    Bm = torch.randn(b, S, G, N, device=DEV, dtype=torch.float16, requires_grad=True)
    Cm = torch.randn(b, S, G, N, device=DEV, dtype=torch.float16, requires_grad=True)
    D = torch.randn(H, device=DEV, dtype=torch.float32)

    def run_fwd():
        return mamba_chunk_scan_combined(x, dt, A, Bm, Cm, chunk, D=D)

    y = run_fwd()
    torch.cuda.synchronize()
    out["regions"]["fwd"] = _time_cuda(run_fwd)

    # backward: time the autograd VJP of the fused kernel (the same grad the
    # cppmega training step computes). Recompute fwd each iter so grad graph is live.
    dy = torch.randn_like(y)

    def run_bwd():
        for t in (x, dt, Bm, Cm):
            if t.grad is not None:
                t.grad = None
        yy = mamba_chunk_scan_combined(x, dt, A, Bm, Cm, chunk, D=D)
        yy.backward(dy)

    run_bwd(); torch.cuda.synchronize()
    out["regions"]["bwd_with_recompute_fwd"] = _time_cuda(run_bwd)
    # NOTE labelled: cppmega "bwd" here INCLUDES a forward recompute (autograd needs
    # a live graph); ours-bwd reuses the forward cache (no replay). We record both
    # the combined number AND derive a fwd-subtracted bwd estimate for honesty.
    out["bwd_minus_fwd_ms"] = (out["regions"]["bwd_with_recompute_fwd"]["median_ms"]
                               - out["regions"]["fwd"]["median_ms"])
    return out


# =========================================================================== #
# cppmega M2RNN arm — the fused Triton m2rnn_scan_triton
# (cppmega/megatron/m2rnn_triton.py). fwd + autograd bwd at the m2rnn shape.
# =========================================================================== #
def bench_cppmega_m2rnn(cfg: dict) -> dict:
    """Time cppmega's fused Triton M2RNN fwd + autograd bwd at ``cfg``.

    RAISES (hard error) if triton / the cppmega package is not importable on the
    box — RULE #1: surface WHY, never silently skip."""
    import torch
    # cppmega original repo must be importable (it lives beside cppmega.mlx).
    cppmega_root = "/Volumes/external/sources/cppmega"
    if os.path.isdir(cppmega_root) and cppmega_root not in sys.path:
        sys.path.insert(0, cppmega_root)
    # gb10 path (the cppmega checkout may live elsewhere there); env override:
    gb10_root = os.environ.get("CPPMEGA_ORIG_ROOT")
    if gb10_root and gb10_root not in sys.path:
        sys.path.insert(0, gb10_root)
    try:
        from cppmega.megatron.m2rnn_triton import m2rnn_scan_triton
    except Exception as exc:
        raise RuntimeError(
            "bench_cppmega_m2rnn: cannot import cppmega.megatron.m2rnn_triton."
            f"m2rnn_scan_triton: {exc!r}. Set CPPMEGA_ORIG_ROOT to the cppmega "
            "original checkout on gb10 (and ensure triton is installed). NOT a "
            "fallback path."
        ) from exc

    DEV = "cuda"
    b = cfg["batch"]; S = cfg["seqlen"]; Hh = cfg["heads"]
    K = cfg["k_dim"]; V = cfg["v_dim"]
    out = {"cfg": dict(cfg), "regions": {}}

    torch.manual_seed(0)
    # m2rnn_scan_triton shapes: q:(b,S,n_q,K) k:(b,S,n_k,K) v:(b,S,n_v,V)
    #   W:(n_w,V,V) xf:(b,S,n_f). Use n_q=n_k=n_v=n_w=n_f=Hh (the simple square case).
    q = torch.randn(b, S, Hh, K, device=DEV, dtype=torch.float32, requires_grad=True)
    k = torch.randn(b, S, Hh, K, device=DEV, dtype=torch.float32, requires_grad=True)
    v = torch.randn(b, S, Hh, V, device=DEV, dtype=torch.float32, requires_grad=True)
    W = torch.randn(Hh, V, V, device=DEV, dtype=torch.float32, requires_grad=True)
    xf = torch.randn(b, S, Hh, device=DEV, dtype=torch.float32, requires_grad=True)

    def run_fwd():
        y, _ = m2rnn_scan_triton(q, k, v, W, xf)
        return y

    y = run_fwd()
    torch.cuda.synchronize()
    out["regions"]["fwd"] = _time_cuda(run_fwd)

    dy = torch.randn_like(y)

    def run_bwd():
        for t in (q, k, v, W, xf):
            if t.grad is not None:
                t.grad = None
        yy, _ = m2rnn_scan_triton(q, k, v, W, xf)
        yy.backward(dy)

    run_bwd(); torch.cuda.synchronize()
    out["regions"]["bwd_with_recompute_fwd"] = _time_cuda(run_bwd)
    out["bwd_minus_fwd_ms"] = (out["regions"]["bwd_with_recompute_fwd"]["median_ms"]
                               - out["regions"]["fwd"]["median_ms"])
    return out


# =========================================================================== #
# Cross-arm attribution: compute ours-vs-cppmega ratios + the host-stage finding.
# =========================================================================== #
def attribute(ours_m: dict, cpp_m: dict | None, cpp_r: dict | None) -> dict:
    """Build the per-region ours/cppmega comparison + the verdict the GB10 phase
    reads. Every ratio is MEASURED/MEASURED; missing arms are recorded as None
    with a reason (RULE #1: no fabricated cross-arm number)."""
    a = {"per_region_ms": {}, "ours_vs_cppmega": {}, "host_stage": {}, "verdict": {}}

    # ours mamba region medians
    for r, d in ours_m.get("regions", {}).items():
        a["per_region_ms"][f"ours_{r}"] = round(d["median_ms"], 4)
    a["per_region_ms"]["ours_fwd_chain"] = round(ours_m["fwd_chain"]["median_ms"], 4)
    if "bwd_chain" in ours_m:
        a["per_region_ms"]["ours_bwd_chain"] = round(ours_m["bwd_chain"]["median_ms"], 4)

    # ours forward = F0+F1+F2 region sum; cppmega forward = its fused kernel.
    ours_fwd_sum = ours_m.get("fwd_region_sum_ms")
    a["host_stage"]["ours_fwd_region_sum_ms"] = round(ours_fwd_sum, 4)
    a["host_stage"]["ours_fwd_chain_ms"] = round(ours_m["fwd_chain"]["median_ms"], 4)
    a["host_stage"]["ours_fwd_host_stage_ms"] = round(ours_m["fwd_host_stage_ms"], 4)
    if "bwd_chain" in ours_m:
        a["host_stage"]["ours_bwd_region_sum_ms"] = round(ours_m["bwd_region_sum_ms"], 4)
        a["host_stage"]["ours_bwd_chain_ms"] = round(ours_m["bwd_chain"]["median_ms"], 4)
        a["host_stage"]["ours_bwd_host_stage_ms"] = round(ours_m["bwd_host_stage_ms"], 4)

    if cpp_m is not None:
        cpp_fwd = cpp_m["regions"]["fwd"]["median_ms"]
        a["per_region_ms"]["cppmega_mamba_fwd"] = round(cpp_fwd, 4)
        a["per_region_ms"]["cppmega_mamba_bwd_with_recompute_fwd"] = round(
            cpp_m["regions"]["bwd_with_recompute_fwd"]["median_ms"], 4)
        a["per_region_ms"]["cppmega_mamba_bwd_minus_fwd"] = round(
            cpp_m["bwd_minus_fwd_ms"], 4)
        # ours-forward (whole chunked chain) vs cppmega-fused-forward
        if ours_fwd_sum and cpp_fwd > 0:
            a["ours_vs_cppmega"]["fwd_chain_vs_fused"] = round(
                ours_m["fwd_chain"]["median_ms"] / cpp_fwd, 4)
            a["ours_vs_cppmega"]["fwd_kernels_only_vs_fused"] = round(
                ours_fwd_sum / cpp_fwd, 4)
        if "bwd_chain" in ours_m and cpp_m["bwd_minus_fwd_ms"] > 0:
            a["ours_vs_cppmega"]["bwd_chain_vs_fused"] = round(
                ours_m["bwd_chain"]["median_ms"] / cpp_m["bwd_minus_fwd_ms"], 4)
    else:
        a["ours_vs_cppmega"]["mamba"] = None
        a["verdict"]["cppmega_mamba"] = "ABSENT: mamba_ssm not run (see hard_errors)"

    if cpp_r is not None:
        a["per_region_ms"]["cppmega_m2rnn_fwd"] = round(
            cpp_r["regions"]["fwd"]["median_ms"], 4)
        a["per_region_ms"]["cppmega_m2rnn_bwd_minus_fwd"] = round(
            cpp_r["bwd_minus_fwd_ms"], 4)
    else:
        a["verdict"]["cppmega_m2rnn"] = "ABSENT: m2rnn_scan_triton not run (see hard_errors)"

    # OURS m2rnn — HONEST: no CUDA twin exists (Metal-only Path-C). Never fabricated.
    a["verdict"]["ours_m2rnn"] = (
        "NOT-RUNNABLE-ON-CUDA: cppmega_mlx M2RNN Path-C "
        "(cppmega_mlx/nn/_tilelang/m2rnn_path_c.py) is Metal/MSL-through-MLX only; "
        "there is NO CUDA twin to time on gb10. ours-m2rnn region cost is therefore "
        "UNMEASURABLE on this box (do not fabricate). The cppmega m2rnn region cost "
        "above is the absolute reference; the §17 finding it confirms is that "
        "mamba+m2rnn are a MINORITY of the iter."
    )

    # The §17 verdict: is the loser the KERNEL or the HOST/LAUNCH STAGING?
    hs = ours_m.get("fwd_host_stage_ms", 0.0)
    fwd_chain = ours_m["fwd_chain"]["median_ms"]
    if fwd_chain > 0:
        frac = hs / fwd_chain
        a["host_stage"]["ours_fwd_host_stage_fraction"] = round(frac, 4)
        a["verdict"]["fwd_loser"] = (
            "HOST/LAUNCH-STAGING dominates the forward chain wall "
            f"({frac*100:.1f}% of fwd_chain is staging, not kernel)"
            if frac > 0.25 else
            "KERNELS dominate the forward chain wall "
            f"(staging only {frac*100:.1f}%)"
        )
    return a


# --------------------------------------------------------------------------- #
def main() -> int:
    prod = "--prod" in sys.argv
    bs4 = "--bs4" in sys.argv
    run_mamba_cpp = "--no-mamba-cppmega" not in sys.argv
    run_m2rnn_cpp = "--no-m2rnn-cppmega" not in sys.argv
    if bs4 and not prod:
        print("FAIL-LOUD: --bs4 requires --prod (the bs4 target is the prod tile); "
              "RULE #1: refusing to silently ignore --bs4")
        return 2
    batch = 4 if bs4 else 1

    print("=== mamba3/m2rnn OURS-vs-cppmega per-region compare (gb10 sm_121) ===")
    print(f"prod={prod} bs4={bs4} batch={batch} "
          f"mamba_cppmega={run_mamba_cpp} m2rnn_cppmega={run_m2rnn_cpp}")

    try:
        import torch
        print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available(),
              "dev", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
        if not torch.cuda.is_available():
            print("NO CUDA DEVICE — this bench is gb10-only (RULE #1: no CPU stub)")
            return 2
    except Exception:
        print("torch import FAILED (expected on this Mac; run on gb10):")
        traceback.print_exc()
        return 2

    mcfg = mamba_cfg(prod, batch)
    rcfg = m2rnn_cfg(prod, batch)
    hard_errors = {}

    # ---- OUR path_c mamba (always; it is the thing under test) ----
    try:
        ours_m = bench_ours_mamba(mcfg, run_backward=True)
        print("[OURS-mamba] F0/F1/F2 + B0/B1/B2 timed (medians captured)")
    except Exception as exc:  # RULE #1: surface WHERE+WHAT, record as hard error.
        traceback.print_exc()
        hard_errors["ours_mamba"] = repr(exc)
        ours_m = None

    # ---- cppmega mamba (Megatron SSD Triton) ----
    cpp_m = None
    if run_mamba_cpp:
        try:
            cpp_m = bench_cppmega_mamba(mcfg)
            print("[CPPMEGA-mamba] mamba_chunk_scan_combined fwd+bwd timed")
        except Exception as exc:
            traceback.print_exc()
            hard_errors["cppmega_mamba"] = repr(exc)

    # ---- cppmega m2rnn (fused Triton) ----
    cpp_r = None
    if run_m2rnn_cpp:
        try:
            cpp_r = bench_cppmega_m2rnn(rcfg)
            print("[CPPMEGA-m2rnn] m2rnn_scan_triton fwd+bwd timed")
        except Exception as exc:
            traceback.print_exc()
            hard_errors["cppmega_m2rnn"] = repr(exc)

    block = {
        "bench": "mamba3_m2rnn_compare",
        "prod": prod, "bs4": bs4, "batch": batch,
        "mamba_cfg": mcfg, "m2rnn_cfg": rcfg,
        "ours_mamba": ours_m,
        "cppmega_mamba": cpp_m,
        "cppmega_m2rnn": cpp_r,
        "hard_errors": hard_errors,
    }
    if ours_m is not None:
        block["attribution"] = attribute(ours_m, cpp_m, cpp_r)

    # Machine-parseable single-line RESULT (the GB10 phase greps this).
    print("\nRESULT " + json.dumps(block))

    # Human summary
    if ours_m is not None:
        att = block["attribution"]
        print("\n--- per-region medians (ms/call), MEASURED ---")
        for k, v in att["per_region_ms"].items():
            print(f"  {k:42s} {v:9.4f} ms")
        print("\n--- ours-vs-cppmega ratios (MEASURED/MEASURED) ---")
        for k, v in att["ours_vs_cppmega"].items():
            print(f"  {k:42s} {v}")
        print("\n--- §17 host-staging finding ---")
        for k, v in att["host_stage"].items():
            print(f"  {k:42s} {v}")
        for k, v in att["verdict"].items():
            print(f"  VERDICT[{k}]: {v}")
    if hard_errors:
        print("\n--- HARD ERRORS (RULE #1: surfaced, not swallowed) ---")
        for k, v in hard_errors.items():
            print(f"  {k}: {v}")

    # Exit nonzero ONLY if the thing-under-test (ours) failed; a missing cppmega
    # arm (mamba_ssm/triton absent) is a recorded ABSENCE, not a bench failure —
    # but it IS surfaced loudly above so the GB10 phase installs the dep.
    return 0 if ours_m is not None else 1


if __name__ == "__main__":
    sys.exit(main())

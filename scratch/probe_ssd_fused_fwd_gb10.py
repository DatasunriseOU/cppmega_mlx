"""GB10/sm_121 measurement + parity harness for the FUSED SSD forward (§23).

Builds the NEW fused ``ssd_fused_fwd_cuda_prim`` (F0+F1+F2 in ONE smem-resident
CUDA kernel) at the prod cfg (S=4096 c=64 g=8 H=112 P=64 N=64 bs1) and:

  1. reports whether it LAUNCHES at N=64 (the smem-fit question).
  2. times it (warm median + min over N timed dispatches, device-resident).
  3. checks PARITY of the fused output + final_state vs:
       (a) the un-fused F0->F1->F2 CUDA chain (the real production reference),
       (b) the SERIAL per-timestep reference (the contract anchor),
     over ALL elements (max|abs|), gate fp16 < 5e-4.
  4. also builds + times the un-fused F0/F1/F2 chain at the SAME cfg so we have
     the apples-to-apples comparison (fused ms vs 16.37+1.24+F2 ms).

RULE #1: every kernel goes through its ONE builder; a compile/run/parity failure
is SURFACED (traceback printed, returns False) — never swallowed, never falls
back to a serial path silently.

Run on gb10 (same env as scratch/probe_chunked_scan_cuda_gb10.py):
  PYTHONPATH=/home/dave/source/cppmega_mlx:/home/dave/source/tilelang/3rdparty/tvm/python:/home/dave/source/tilelang/3rdparty/tvm/3rdparty/tvm-ffi/python \
  TVM_LIBRARY_PATH=/home/dave/source/tilelang/build/lib \
  /home/dave/cppmega-venv/bin/python scratch/probe_ssd_fused_fwd_gb10.py [--prod]
"""

import sys
import time
import traceback

# RULE #1: fix the NVRTC builtins loader BEFORE torch/tilelang import (gb10
# sm_121 CUDA codegen needs the nvrtc builtins path for exp2/etc.). No-op +
# byte-identical to the fp8 bench's ordering.
try:
    from cppmega_mlx._gb10_nvrtc_env import ensure_nvrtc_builtins_path

    ensure_nvrtc_builtins_path()
except Exception as _e:  # surfaced, not swallowed — print so it's visible
    print(f"[nvrtc-env] ensure_nvrtc_builtins_path unavailable: {_e!r}")

import torch
from einops import rearrange, repeat

DEV = "cuda"


# --------------------------------------------------------------------------- #
# References (identical algebra to tests/test_mamba3_chunk_scan_combine_f2 and  #
# scratch/probe_chunked_scan_cuda_gb10.py — the SSD numerical ground truth).    #
# --------------------------------------------------------------------------- #
def serial_full_forward(C, Bmat, x, A, dt, h0, D):
    """SERIAL per-timestep diagonal forward over the FULL sequence (fp32).

    h[t]  = exp(A[h]*dt[t]) * h[t-1] + dt[t] * (x[t] outer B[t])
    y[t]  = sum_n h[t]*C[t] + D*x[t]   (seeded by h0). Returns (out, h_last).
    """
    batch, seqlen, ngroups, dstate = C.shape
    _, _, nheads, headdim = x.shape
    h = nheads // ngroups
    Cf = repeat(C.float(), "b l g n -> b l (g h) n", h=h)
    Bf = repeat(Bmat.float(), "b l g n -> b l (g h) n", h=h)
    xf = x.float()
    Af = A.float()
    dtf = dt.float()
    state = h0.float().clone()
    out = torch.zeros(batch, seqlen, nheads, headdim, device=x.device)
    for t in range(seqlen):
        decay = torch.exp(Af.view(1, nheads) * dtf[:, t])
        inp = dtf[:, t][:, :, None, None] * (
            xf[:, t][:, :, :, None] * Bf[:, t][:, :, None, :]
        )
        state = decay[:, :, None, None] * state + inp
        y = torch.einsum("bhpn,bhn->bhp", state, Cf[:, t])
        out[:, t] = y + D.float().view(1, nheads, 1) * xf[:, t]
    return out, state


def eager_precompute(C, Bmat, x, A, dt, h0, chunk_size):
    """F0+F1 reference -> (cb fp16, dA_cumsum fp16, prev_states fp16)."""
    batch, seqlen, ngroups, dstate = C.shape
    _, _, nheads, headdim = x.shape
    nchunks = seqlen // chunk_size
    Cf, Bf, xf, Af, dtf = C.float(), Bmat.float(), x.float(), A.float(), dt.float()
    a = Af.view(1, 1, nheads) * dtf
    a_c = rearrange(a, "b (c l) h -> b h c l", c=nchunks)
    dA_cumsum = torch.cumsum(a_c, dim=-1)
    Cc = rearrange(Cf, "b (c l) g n -> b c l g n", c=nchunks)
    Bc = rearrange(Bf, "b (c s) g n -> b c s g n", c=nchunks)
    cb = torch.einsum("bclgn,bcsgn->bcgls", Cc, Bc)
    h = nheads // ngroups
    Bexp = rearrange(Bf, "b (c s) g n -> b c s g n", c=nchunks).repeat_interleave(h, dim=3)
    xexp = rearrange(xf, "b (c s) hh p -> b c s hh p", c=nchunks)
    dtc = rearrange(dtf, "b (c s) hh -> b hh c s", c=nchunks)
    decay_states = torch.exp(dA_cumsum[:, :, :, -1:] - dA_cumsum)
    states = torch.einsum("bhcs,bhcs,bcshp,bcshn->bchpn", decay_states, dtc, xexp, Bexp)
    chunk_tail = dA_cumsum[:, :, :, -1]
    init = h0.float().unsqueeze(1)
    states_cat = torch.cat([init, states], dim=1)
    pad = torch.nn.functional.pad(chunk_tail, (1, 0))
    cc = pad.shape[-1]
    csum = torch.cumsum(pad, dim=-1)
    seg = csum[:, :, :, None] - csum[:, :, None, :]
    mask = torch.tril(torch.ones(cc, cc, device=pad.device, dtype=torch.bool))
    seg = seg.masked_fill(~mask, float("-inf"))
    decay_chunk = torch.exp(seg)
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states_cat)
    prev_states = new_states[:, :-1]
    return (
        cb.half().contiguous(),
        dA_cumsum.half().contiguous(),
        prev_states.half().contiguous(),
    )


def _time(fn, n=20):
    for _ in range(3):
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
    return ts[len(ts) // 2], ts[0]


def run_cfg(batch, seqlen, chunk, ngroups, nheads, headdim, dstate):
    from cppmega_mlx.nn._tilelang.mamba3_ssd_fused_fwd import (
        build_ssd_fused_fwd,
        ssd_fused_fwd_grid,
        ssd_fused_fwd_smem_budget_bytes,
    )
    from cppmega_mlx.nn._tilelang.mamba3_chunked_precompute_core import (
        build_chunk_precompute_metal,
        build_inter_chunk_recur_metal,
    )
    from cppmega_mlx.nn._tilelang.mamba3_chunked_scan_core import (
        build_chunk_scan_combine_metal,
    )

    nchunks = seqlen // chunk
    tg, grid = ssd_fused_fwd_grid(batch, seqlen, chunk, ngroups, nheads, headdim, dstate)
    budget = ssd_fused_fwd_smem_budget_bytes(chunk, headdim, dstate)
    print(f"\n=== cfg b={batch} S={seqlen} c={chunk} g={ngroups} H={nheads} "
          f"P={headdim} N={dstate} ===")
    print(f"[FUSED] grid={grid} tg={tg}  smem_budget={budget['total']} B "
          f"({budget['total']/1024:.2f} KiB) cap=101376 B")

    torch.manual_seed(0)
    C = (torch.randn(batch, seqlen, ngroups, dstate, device=DEV) * 0.1).half()
    Bmat = (torch.randn(batch, seqlen, ngroups, dstate, device=DEV) * 0.1).half()
    x = (torch.randn(batch, seqlen, nheads, headdim, device=DEV) * 0.1).half()
    A = -torch.rand(nheads, device=DEV).half()
    dt = (torch.rand(batch, seqlen, nheads, device=DEV) * 0.05).half()
    D = torch.randn(nheads, device=DEV).half()
    h0 = (torch.randn(batch, nheads, headdim, dstate, device=DEV) * 0.1).float()

    # ---- SERIAL ground truth (CPU, fp32) ----
    out_serial, hlast_serial = serial_full_forward(
        C.cpu(), Bmat.cpu(), x.cpu(), A.cpu(), dt.cpu(), h0.cpu(), D.cpu())

    # =================== FUSED kernel ===================
    try:
        kf = build_ssd_fused_fwd(
            batch, seqlen, chunk, ngroups, nheads, headdim, dstate, target="cuda")
    except Exception:
        print("[FUSED] COMPILE FAILED:")
        traceback.print_exc()
        return False
    print("[FUSED] COMPILE ok")

    out_f = torch.zeros(batch, seqlen, nheads, headdim, device=DEV, dtype=torch.float16)
    fst_f = torch.zeros(batch, nheads, headdim, dstate, device=DEV, dtype=torch.float32)
    fused_args = (
        x.contiguous(), Bmat.contiguous(), C.contiguous(), A.contiguous(),
        dt.contiguous(), D.contiguous(), h0.contiguous(),
    )
    try:
        kf(*fused_args, out_f, fst_f)
        torch.cuda.synchronize()
    except Exception:
        print("[FUSED] RUN FAILED (LAUNCH at N=%d):" % dstate)
        traceback.print_exc()
        return False
    print("[FUSED] RUN ok (LAUNCHES at N=%d)" % dstate)

    f_med, f_min = _time(lambda: kf(*fused_args, out_f, fst_f))
    print(f"[FUSED] TIMING median={f_med:.3f} ms/call  min={f_min:.3f} ms")

    nan_f = bool(torch.isnan(out_f).any()) or bool(torch.isnan(fst_f).any())
    fs_out_serial = float((out_f.float().cpu() - out_serial).abs().max())
    fs_hlast_serial = float((fst_f.float().cpu() - hlast_serial).abs().max())
    print(f"[FUSED vs SERIAL] out max|abs|={fs_out_serial:.3e}  "
          f"final_state max|abs|={fs_hlast_serial:.3e}  NaN={nan_f}")

    # =================== UN-FUSED F0 -> F1 -> F2 chain ===================
    try:
        k0 = build_chunk_precompute_metal(
            batch, seqlen, chunk, ngroups, nheads, headdim, dstate, target="cuda")
        k1 = build_inter_chunk_recur_metal(
            batch, seqlen, chunk, ngroups, nheads, headdim, dstate, target="cuda")
        k2 = build_chunk_scan_combine_metal(
            batch, seqlen, chunk, ngroups, nheads, headdim, dstate, target="cuda")
    except Exception:
        print("[UNFUSED] COMPILE FAILED:")
        traceback.print_exc()
        return False
    print("[UNFUSED] F0/F1/F2 COMPILE ok")

    cb = torch.zeros(batch, nchunks, ngroups, chunk, chunk, device=DEV, dtype=torch.float16)
    dA_cumsum = torch.zeros(batch, nheads, nchunks, chunk, device=DEV, dtype=torch.float16)
    summary_states = torch.zeros(batch, nchunks, nheads, headdim, dstate, device=DEV, dtype=torch.float32)
    prev_states = torch.zeros(batch, nchunks, nheads, headdim, dstate, device=DEV, dtype=torch.float32)
    final_state_u = torch.zeros(batch, nheads, headdim, dstate, device=DEV, dtype=torch.float32)
    out_u = torch.zeros(batch, seqlen, nheads, headdim, device=DEV, dtype=torch.float16)
    dt_k = rearrange(dt, "b (c s) hh -> b hh c s", c=nchunks).contiguous()

    def _run_f0():
        k0(x.contiguous(), Bmat.contiguous(), C.contiguous(), A.contiguous(),
           dt.contiguous(), cb, dA_cumsum, summary_states)

    def _run_f1():
        k1(summary_states.contiguous(), dA_cumsum.contiguous(), h0.contiguous(),
           prev_states, final_state_u)

    def _run_f2():
        k2(cb.contiguous(), x.contiguous(), dt_k.contiguous(), dA_cumsum.contiguous(),
           C.contiguous(), prev_states.contiguous(), D.contiguous(), out_u)

    try:
        _run_f0(); torch.cuda.synchronize()
        _run_f1(); torch.cuda.synchronize()
        _run_f2(); torch.cuda.synchronize()
    except Exception:
        print("[UNFUSED] RUN FAILED:")
        traceback.print_exc()
        return False
    print("[UNFUSED] F0/F1/F2 RUN ok (F2 LAUNCHES at N=%d)" % dstate)

    t0, t0min = _time(_run_f0)
    t1, t1min = _time(_run_f1)
    # F2 must be re-fed prev_states each time, but prev_states is stable post-F1.
    t2, t2min = _time(_run_f2)
    unfused_sum = t0 + t1 + t2
    print(f"[UNFUSED] F0={t0:.3f}  F1={t1:.3f}  F2={t2:.3f}  "
          f"SUM={unfused_sum:.3f} ms/call (medians)")

    # parity of UN-FUSED chain vs serial (so we know the reference itself is sound)
    uo_serial = float((out_u.float().cpu() - out_serial).abs().max())
    uh_serial = float((final_state_u.float().cpu() - hlast_serial).abs().max())
    print(f"[UNFUSED vs SERIAL] out max|abs|={uo_serial:.3e}  "
          f"final_state max|abs|={uh_serial:.3e}")

    # parity of FUSED vs UN-FUSED (the production reference) over ALL elements
    fu_out = float((out_f.float() - out_u.float()).abs().max())
    fu_fst = float((fst_f - final_state_u).abs().max())
    print(f"[FUSED vs UNFUSED] out max|abs|={fu_out:.3e}  "
          f"final_state max|abs|={fu_fst:.3e}")

    # ===== verdicts =====
    gate = 5e-4
    parity_ok = (
        fs_out_serial < gate and fs_hlast_serial < gate
        and fu_out < gate and fu_fst < gate and not nan_f
    )
    print(f"\n[SUMMARY] FUSED median={f_med:.3f} ms  UNFUSED sum={unfused_sum:.3f} ms  "
          f"speedup={unfused_sum/f_med:.2f}x  cppmega_fused_ref=3.110 ms "
          f"(fused/cppmega={f_med/3.110:.2f}x)")
    print(f"[SUMMARY] parity_all_elements={'PASS' if parity_ok else 'FAIL'} "
          f"(gate fp16<{gate})")
    print(f"[CSV] cfg=S{seqlen}c{chunk}g{ngroups}H{nheads}P{headdim}N{dstate} "
          f"fused_ms={f_med:.4f} fused_min={f_min:.4f} "
          f"f0={t0:.4f} f1={t1:.4f} f2={t2:.4f} unfused_sum={unfused_sum:.4f} "
          f"fused_vs_unfused={unfused_sum/f_med:.4f}x "
          f"fused_vs_cppmega={f_med/3.110:.4f}x "
          f"fs_out={fs_out_serial:.3e} fs_fst={fs_hlast_serial:.3e} "
          f"fu_out={fu_out:.3e} fu_fst={fu_fst:.3e} parity={'PASS' if parity_ok else 'FAIL'}")
    return parity_ok


def main():
    prod = "--prod" in sys.argv
    print("=== FUSED SSD forward probe (gb10 sm_121) ===")
    print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available(),
          "dev", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
    if not torch.cuda.is_available():
        print("NO CUDA DEVICE")
        sys.exit(2)
    if prod:
        cfgs = [dict(batch=1, seqlen=4096, chunk=64, ngroups=8, nheads=112,
                     headdim=64, dstate=64)]
    else:
        cfgs = [
            dict(batch=1, seqlen=256, chunk=64, ngroups=1, nheads=2, headdim=64, dstate=16),
            dict(batch=1, seqlen=512, chunk=64, ngroups=8, nheads=8, headdim=64, dstate=64),
        ]
    ok = True
    for cfg in cfgs:
        try:
            ok = run_cfg(**cfg) and ok
        except Exception:
            print("CFG CRASHED:")
            traceback.print_exc()
            ok = False
    print("\nALL PARITY PASS" if ok else "\nPARITY/RUN FAILURE (see above)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

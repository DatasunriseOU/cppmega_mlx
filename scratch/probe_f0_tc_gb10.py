"""GB10/sm_121 F0 tensor-core precompute probe (§26).

Builds the NEW chunk_precompute_fwd_cuda_prim (cb + summary_states T.gemm)
at prod cfg and:
  1. dumps generated CUDA, greps mma.sync (REAL sm_120 MMA, not relabeled serial).
  2. times new F0 per-call (warm median + min).
  3. parity of cb + summary_states + dA_cumsum vs:
       (a) the SERIAL F0 reference (eager precompute math),
       (b) the SERIAL Metal F0 prim built on CUDA target == NO, the serial prim is
           Metal-only; instead we compare against the eager-precompute fp32 reference
           AND re-build the *serial* prim source for byte check separately.
     over ALL elements (max|abs|), gate fp16 < 5e-4.
  4. F0->F1->F2 chain timing with the NEW F0.

RULE #1: every kernel goes through its ONE builder; failures SURFACED.
"""
import sys, time, traceback, re
try:
    from cppmega_mlx._gb10_nvrtc_env import ensure_nvrtc_builtins_path
    ensure_nvrtc_builtins_path()
except Exception as _e:
    print(f"[nvrtc-env] unavailable: {_e!r}")

import torch
from einops import rearrange, repeat
DEV = "cuda"


def eager_precompute_f0(C, Bmat, x, A, dt, chunk_size):
    """SERIAL F0 reference (fp32): cb, dA_cumsum, summary_states.
    Mirrors chunk_precompute math exactly (the contract anchor)."""
    batch, seqlen, ngroups, dstate = C.shape
    _, _, nheads, headdim = x.shape
    nchunks = seqlen // chunk_size
    h = nheads // ngroups
    Cf, Bf, xf, Af, dtf = C.float(), Bmat.float(), x.float(), A.float(), dt.float()
    a = Af.view(1, 1, nheads) * dtf
    a_c = rearrange(a, "b (c l) hh -> b hh c l", c=nchunks)
    dA_cumsum = torch.cumsum(a_c, dim=-1)                      # (b,h,c,l)
    Cc = rearrange(Cf, "b (c l) g n -> b c l g n", c=nchunks)
    Bc = rearrange(Bf, "b (c s) g n -> b c s g n", c=nchunks)
    cb = torch.einsum("bclgn,bcsgn->bcgls", Cc, Bc)           # (b,c,g,l,s)
    Bexp = Bc.repeat_interleave(h, dim=3)                     # (b,c,s,hh,n)
    xexp = rearrange(xf, "b (c s) hh p -> b c s hh p", c=nchunks)
    dtc = rearrange(dtf, "b (c s) hh -> b hh c s", c=nchunks)
    decay_states = torch.exp(dA_cumsum[:, :, :, -1:] - dA_cumsum)  # (b,h,c,s)
    # summary_states[b,c,hh,p,n] = sum_s decay*dt * x * B
    states = torch.einsum("bhcs,bhcs,bcshp,bcshn->bchpn",
                          decay_states, dtc, xexp, Bexp)
    return (cb.float().contiguous(),
            dA_cumsum.float().contiguous(),
            states.float().contiguous())


def _time(fn, n=30):
    for _ in range(5):
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
    return ts[len(ts)//2], ts[0]


def run_cfg(batch, seqlen, chunk, ngroups, nheads, headdim, dstate):
    from cppmega_mlx.nn._tilelang.mamba3_chunked_precompute_core import (
        build_chunk_precompute_metal, build_inter_chunk_recur_metal,
        chunk_precompute_fwd_cuda_prim, chunk_precompute_fwd_metal_prim,
    )
    from cppmega_mlx.nn._tilelang.mamba3_chunked_scan_core import (
        build_chunk_scan_combine_metal,
    )
    nchunks = seqlen // chunk
    print(f"\n=== cfg b={batch} S={seqlen} c={chunk} g={ngroups} H={nheads} "
          f"P={headdim} N={dstate}  nchunks={nchunks} ===")

    torch.manual_seed(0)
    C = (torch.randn(batch, seqlen, ngroups, dstate, device=DEV) * 0.1).half()
    Bmat = (torch.randn(batch, seqlen, ngroups, dstate, device=DEV) * 0.1).half()
    x = (torch.randn(batch, seqlen, nheads, headdim, device=DEV) * 0.1).half()
    A = -torch.rand(nheads, device=DEV).half()
    dt = (torch.rand(batch, seqlen, nheads, device=DEV) * 0.05).half()
    D = torch.randn(nheads, device=DEV).half()
    h0 = (torch.randn(batch, nheads, headdim, dstate, device=DEV) * 0.1).float()

    # ---- SERIAL F0 reference (fp32 einsum on the SAME fp16 inputs) ----
    cb_ref, dac_ref, ss_ref = eager_precompute_f0(
        C, Bmat, x, A, dt, chunk)

    # =================== NEW CUDA F0 (tensor-core) ===================
    try:
        k0 = build_chunk_precompute_metal(
            batch, seqlen, chunk, ngroups, nheads, headdim, dstate, target="cuda")
    except Exception:
        print("[F0-CUDA] COMPILE FAILED:")
        traceback.print_exc()
        return False
    print("[F0-CUDA] COMPILE ok")

    # ---- dump CUDA source, grep mma.sync ----
    try:
        src = k0.get_kernel_source()
        # tl::mma_sync is the codegen wrapper for the real PTX
        # mma.sync.aligned.m16n8k16 (CuTe SM80_16x8x16_F32F16F16F32_TN).
        n_mma = src.count("tl::mma_sync") + src.count("mma.sync")
        n_ldm = src.count("ptx_ldmatrix")
        n_serial_dot = len(re.findall(r"for \(int n = 0; n < %d" % dstate, src))
        print(f"[F0-CUDA] codegen: tl::mma_sync x{n_mma}  ldmatrix x{n_ldm}  "
              f"leftover_serial_N_dot x{n_serial_dot}  len={len(src)}B")
        with open("/tmp/f0_cuda_src.cu", "w") as f:
            f.write(src)
        print("[F0-CUDA] source -> /tmp/f0_cuda_src.cu")
    except Exception:
        print("[F0-CUDA] source dump FAILED:")
        traceback.print_exc()
        n_mma = -1

    cb = torch.zeros(batch, nchunks, ngroups, chunk, chunk, device=DEV, dtype=torch.float16)
    dA_cumsum = torch.zeros(batch, nheads, nchunks, chunk, device=DEV, dtype=torch.float16)
    summary_states = torch.zeros(batch, nchunks, nheads, headdim, dstate, device=DEV, dtype=torch.float32)

    def _run_f0():
        k0(x.contiguous(), Bmat.contiguous(), C.contiguous(), A.contiguous(),
           dt.contiguous(), cb, dA_cumsum, summary_states)
    try:
        _run_f0(); torch.cuda.synchronize()
    except Exception:
        print("[F0-CUDA] RUN FAILED:")
        traceback.print_exc()
        return False
    print("[F0-CUDA] RUN ok (LAUNCHES at N=%d)" % dstate)

    # ---- PARITY (all elements) of new F0 vs serial F0 reference ----
    nan0 = bool(torch.isnan(cb).any()) or bool(torch.isnan(summary_states).any()) \
        or bool(torch.isnan(dA_cumsum).any())
    cb_err = float((cb.float().cpu() - cb_ref.cpu()).abs().max())
    dac_err = float((dA_cumsum.float().cpu() - dac_ref.cpu()).abs().max())
    ss_err = float((summary_states.float().cpu() - ss_ref.cpu()).abs().max())
    print(f"[F0-CUDA vs SERIAL-ref] cb max|abs|={cb_err:.3e}  "
          f"dA_cumsum={dac_err:.3e}  summary_states={ss_err:.3e}  NaN={nan0}")

    f0_med, f0_min = _time(_run_f0)
    print(f"[F0-CUDA] TIMING median={f0_med:.4f} ms/call  min={f0_min:.4f} ms")

    # =================== SERIAL METAL F0 prim built on CUDA (the 24.57ms baseline) ===================
    # Build the *serial* metal prim explicitly on the CUDA target so we get the
    # apples-to-apples 24.57ms serial F0 on the SAME gb10 device.
    f0_serial_med = None
    cb_s_err = ss_s_err = None
    try:
        import tilelang
        from cppmega_mlx.nn._tilelang.mamba3_chunked_scan_core import _resolve_chunked_compile_target
        prim_s = chunk_precompute_fwd_metal_prim(
            batch, seqlen, chunk, ngroups, nheads, headdim, dstate)
        k0s = tilelang.compile(prim_s, out_idx=[5, 6, 7],
                               target=_resolve_chunked_compile_target("cuda"))
        cb_s = torch.zeros_like(cb); dac_s = torch.zeros_like(dA_cumsum)
        ss_s = torch.zeros_like(summary_states)
        def _run_f0s():
            k0s(x.contiguous(), Bmat.contiguous(), C.contiguous(), A.contiguous(),
                dt.contiguous(), cb_s, dac_s, ss_s)
        _run_f0s(); torch.cuda.synchronize()
        print("[F0-SERIAL on CUDA] RUN ok")
        f0_serial_med, f0_serial_min = _time(_run_f0s)
        print(f"[F0-SERIAL on CUDA] TIMING median={f0_serial_med:.4f} ms  min={f0_serial_min:.4f} ms")
        # parity new-CUDA vs serial-prim (the tightest gate: same algorithm, fp16)
        cb_s_err = float((cb.float() - cb_s.float()).abs().max())
        ss_s_err = float((summary_states.float() - ss_s.float()).abs().max())
        dac_s_err = float((dA_cumsum.float() - dac_s.float()).abs().max())
        print(f"[F0-CUDA vs F0-SERIAL prim] cb={cb_s_err:.3e}  "
              f"summary_states={ss_s_err:.3e}  dA_cumsum={dac_s_err:.3e}")
    except Exception:
        print("[F0-SERIAL on CUDA] build/run FAILED (serial prim may be Metal-only):")
        traceback.print_exc()

    # =================== F0->F1->F2 chain timing with NEW F0 ===================
    try:
        k1 = build_inter_chunk_recur_metal(
            batch, seqlen, chunk, ngroups, nheads, headdim, dstate, target="cuda")
        k2 = build_chunk_scan_combine_metal(
            batch, seqlen, chunk, ngroups, nheads, headdim, dstate, target="cuda")
    except Exception:
        print("[CHAIN] F1/F2 COMPILE FAILED:")
        traceback.print_exc()
        return False
    prev_states = torch.zeros(batch, nchunks, nheads, headdim, dstate, device=DEV, dtype=torch.float32)
    final_state = torch.zeros(batch, nheads, headdim, dstate, device=DEV, dtype=torch.float32)
    out_u = torch.zeros(batch, seqlen, nheads, headdim, device=DEV, dtype=torch.float16)
    dt_k = rearrange(dt, "b (c s) hh -> b hh c s", c=nchunks).contiguous()

    def _run_f1():
        k1(summary_states.contiguous(), dA_cumsum.contiguous(), h0.contiguous(),
           prev_states, final_state)
    def _run_f2():
        k2(cb.contiguous(), x.contiguous(), dt_k.contiguous(), dA_cumsum.contiguous(),
           C.contiguous(), prev_states.contiguous(), D.contiguous(), out_u)
    try:
        _run_f1(); torch.cuda.synchronize()
        _run_f2(); torch.cuda.synchronize()
    except Exception:
        print("[CHAIN] F1/F2 RUN FAILED:")
        traceback.print_exc()
        return False
    f1_med, f1_min = _time(_run_f1)
    f2_med, f2_min = _time(_run_f2)
    chain = f0_med + f1_med + f2_med
    print(f"[CHAIN] F0={f0_med:.4f}  F1={f1_med:.4f}  F2={f2_med:.4f}  "
          f"SUM={chain:.4f} ms (medians)")

    gate = 5e-4
    parity_ok = (cb_err < gate and ss_err < gate and dac_err < 1e-2 and not nan0)
    # NOTE: dA_cumsum is fp16-stored so its gate is looser (cumsum magnitude),
    # report it but key the gate on cb/summary_states (the GEMM outputs).
    print(f"\n[SUMMARY] new_F0={f0_med:.4f} ms  "
          f"serial_F0={'%.4f'%f0_serial_med if f0_serial_med else 'NA'} ms  "
          f"baseline_24.57ms  chain={chain:.4f} ms")
    print(f"[SUMMARY] parity cb={cb_err:.2e} ss={ss_err:.2e} -> "
          f"{'PASS' if parity_ok else 'FAIL'} (gate {gate})")
    print(f"[CSV] new_f0={f0_med:.4f} new_f0_min={f0_min:.4f} "
          f"serial_f0={f0_serial_med if f0_serial_med else -1:.4f} "
          f"f1={f1_med:.4f} f2={f2_med:.4f} chain={chain:.4f} "
          f"cb_err={cb_err:.3e} ss_err={ss_err:.3e} dac_err={dac_err:.3e} "
          f"cb_vs_serialprim={cb_s_err if cb_s_err else -1:.3e} "
          f"ss_vs_serialprim={ss_s_err if ss_s_err else -1:.3e} "
          f"mma_sync={n_mma} parity={'PASS' if parity_ok else 'FAIL'}")
    return parity_ok


def main():
    print("=== F0 tensor-core precompute probe (gb10 sm_121) ===")
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
    if not torch.cuda.is_available():
        print("NO CUDA"); sys.exit(2)
    cfg = dict(batch=1, seqlen=4096, chunk=64, ngroups=8, nheads=112,
               headdim=64, dstate=64)
    ok = False
    try:
        ok = run_cfg(**cfg)
    except Exception:
        print("CFG CRASHED:"); traceback.print_exc()
    print("\nF0 PARITY PASS" if ok else "\nF0 PARITY/RUN FAILURE")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

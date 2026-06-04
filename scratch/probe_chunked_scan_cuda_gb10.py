"""CUDA-port probe for the mamba3 chunked SSD grid kernels (gb10 / sm_121).

Compiles F0/F1/F2 for ``target="cuda"`` and runs them against the SAME serial
SSD reference the Metal parity test uses (``tests/test_mamba3_chunk_scan_combine_f2``
algebra, reproduced here standalone). Reports, per stage:
  * COMPILE ok / the exact TVM/TileLang error (RULE #1: surfaced, never swallowed)
  * RUN ok / error
  * max|abs| vs the serial reference + NaN check

Run on gb10:
  PYTHONPATH=/home/dave/source/cppmega_mlx:/home/dave/source/tilelang/3rdparty/tvm/python:/home/dave/source/tilelang/3rdparty/tvm/3rdparty/tvm-ffi/python \
  TVM_LIBRARY_PATH=/home/dave/source/tilelang/build/lib \
  /home/dave/cppmega-venv/bin/python scratch/probe_chunked_scan_cuda_gb10.py [--prod]
"""

import sys
import traceback

import torch
from einops import rearrange, repeat

DEV = "cuda"


def eager_precompute(C, Bmat, x, A, dt, h0, chunk_size):
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


def serial_forward(cb, x, dt, dA_cumsum, C, prev_states, D, chunk_size):
    batch, seqlen, ngroups, _ = C.shape
    _, _, nheads, headdim = x.shape
    _, _, nchunks, _ = dt.shape
    h = nheads // ngroups
    Cf = repeat(C.float(), "b l g n -> b l (g h) n", h=h)
    cbf = repeat(cb.float(), "b c g l s -> b c (g h) l s", h=h)
    xf = rearrange(x.float(), "b (c s) hh p -> b c s hh p", c=nchunks)
    dtf = dt.float()
    dac = dA_cumsum.float()
    psf = prev_states.float()
    Cc = rearrange(Cf, "b (c l) hh n -> b c l hh n", c=nchunks)
    out = torch.zeros(batch, nchunks, chunk_size, nheads, headdim, device=x.device)
    for c in range(nchunks):
        for l in range(chunk_size):
            acc = torch.zeros(batch, nheads, headdim, device=x.device)
            for s in range(l + 1):
                decay = torch.exp(dac[:, :, c, l] - dac[:, :, c, s])
                coef = cbf[:, c, :, l, s] * decay * dtf[:, :, c, s]
                acc = acc + coef[:, :, None] * xf[:, c, s]
            yoff = torch.einsum("bhn,bhpn->bhp", Cc[:, c, l], psf[:, c])
            yoff = yoff * torch.exp(dac[:, :, c, l])[:, :, None]
            out[:, c, l] = acc + yoff
    out = rearrange(out, "b c l hh p -> b (c l) hh p")
    Dskip = rearrange(D.float(), "hh -> hh 1")
    return out + x.float() * Dskip


def probe_f2(batch, seqlen, chunk, ngroups, nheads, headdim, dstate, block_Dstate):
    from cppmega_mlx.nn._tilelang.mamba3_chunked_scan_core import (
        build_chunk_scan_combine_metal,
        chunk_scan_fwd_grid,
    )

    total_tg, grid = chunk_scan_fwd_grid(batch, seqlen, chunk, ngroups, nheads, headdim, dstate)
    print(f"[F2] shape b={batch} s={seqlen} c={chunk} g={ngroups} h={nheads} "
          f"p={headdim} n={dstate} block_Dstate={block_Dstate} -> grid={grid} tg={total_tg}")
    torch.manual_seed(0)
    nchunks = seqlen // chunk
    C = (torch.randn(batch, seqlen, ngroups, dstate, device=DEV) * 0.1).half()
    Bmat = (torch.randn(batch, seqlen, ngroups, dstate, device=DEV) * 0.1).half()
    x = (torch.randn(batch, seqlen, nheads, headdim, device=DEV) * 0.1).half()
    A = -torch.rand(nheads, device=DEV).half()
    dt = (torch.rand(batch, seqlen, nheads, device=DEV) * 0.05).half()
    D = torch.randn(nheads, device=DEV).half()
    h0 = (torch.randn(batch, nheads, headdim, dstate, device=DEV) * 0.1).half()
    cb, dA_cumsum, prev_states = eager_precompute(C, Bmat, x, A, dt, h0, chunk)
    dt_k = rearrange(dt, "b (c s) hh -> b hh c s", c=nchunks).contiguous()

    try:
        kernel = build_chunk_scan_combine_metal(
            batch, seqlen, chunk, ngroups, nheads, headdim, dstate,
            target="cuda", block_Dstate=block_Dstate,
        )
    except Exception:
        print("[F2] COMPILE FAILED:")
        traceback.print_exc()
        return False
    print("[F2] COMPILE ok")
    out = torch.zeros(batch, seqlen, nheads, headdim, device=DEV, dtype=torch.float16)
    try:
        kernel(cb.contiguous(), x.contiguous(), dt_k.contiguous(), dA_cumsum.contiguous(),
               C.contiguous(), prev_states.contiguous(), D.contiguous(), out)
        torch.cuda.synchronize()
    except Exception:
        print("[F2] RUN FAILED:")
        traceback.print_exc()
        return False
    # --- timing: warm + median of N timed dispatches (device-resident) ---
    import time
    for _ in range(3):
        kernel(cb.contiguous(), x.contiguous(), dt_k.contiguous(), dA_cumsum.contiguous(),
               C.contiguous(), prev_states.contiguous(), D.contiguous(), out)
    torch.cuda.synchronize()
    N = 20
    ts = []
    for _ in range(N):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        kernel(cb.contiguous(), x.contiguous(), dt_k.contiguous(), dA_cumsum.contiguous(),
               C.contiguous(), prev_states.contiguous(), D.contiguous(), out)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    median_ms = ts[len(ts) // 2]
    print(f"[F2] TIMING median={median_ms:.3f} ms/call  min={ts[0]:.3f} ms (N={N}, warm)")

    ref = serial_forward(cb, x, dt_k, dA_cumsum, C, prev_states, D, chunk)
    nan = bool(torch.isnan(out).any())
    max_abs = float((out.float() - ref.float()).abs().max())
    print(f"[F2] RUN ok  NaN={nan}  max|abs diff vs serial|={max_abs:.3e}  "
          f"(gate fp16<5e-4) -> {'PASS' if (max_abs < 5e-4 and not nan) else 'FAIL'}")
    return (max_abs < 5e-4) and not nan


def probe_f0_f1(batch, seqlen, chunk, ngroups, nheads, headdim, dstate):
    from cppmega_mlx.nn._tilelang.mamba3_chunked_precompute_core import (
        build_chunk_precompute_metal,
        build_inter_chunk_recur_metal,
        chunk_precompute_fwd_grid,
        inter_chunk_recur_fwd_grid,
    )

    nchunks = seqlen // chunk
    g0 = chunk_precompute_fwd_grid(batch, seqlen, chunk, ngroups, nheads, headdim, dstate)
    g1 = inter_chunk_recur_fwd_grid(batch, seqlen, chunk, ngroups, nheads, headdim, dstate)
    print(f"[F0] grid={g0[1]} tg={g0[0]}    [F1] grid={g1[1]} tg={g1[0]}")
    torch.manual_seed(1)
    x = (torch.randn(batch, seqlen, nheads, headdim, device=DEV) * 0.1).half()
    Bmat = (torch.randn(batch, seqlen, ngroups, dstate, device=DEV) * 0.1).half()
    C = (torch.randn(batch, seqlen, ngroups, dstate, device=DEV) * 0.1).half()
    A = -torch.rand(nheads, device=DEV).half()
    dt = (torch.rand(batch, seqlen, nheads, device=DEV) * 0.05).half()
    h0 = (torch.randn(batch, nheads, headdim, dstate, device=DEV) * 0.1).half()

    # ---- F0 ----
    try:
        k0 = build_chunk_precompute_metal(
            batch, seqlen, chunk, ngroups, nheads, headdim, dstate, target="cuda")
    except Exception:
        print("[F0] COMPILE FAILED:")
        traceback.print_exc()
        return False
    print("[F0] COMPILE ok")
    cb = torch.zeros(batch, nchunks, ngroups, chunk, chunk, device=DEV, dtype=torch.float16)
    dA_cumsum = torch.zeros(batch, nheads, nchunks, chunk, device=DEV, dtype=torch.float16)
    summary_states = torch.zeros(batch, nchunks, nheads, headdim, dstate, device=DEV, dtype=torch.float32)
    try:
        k0(x.contiguous(), Bmat.contiguous(), C.contiguous(), A.contiguous(),
           dt.contiguous(), cb, dA_cumsum, summary_states)
        torch.cuda.synchronize()
    except Exception:
        print("[F0] RUN FAILED:")
        traceback.print_exc()
        return False
    print(f"[F0] RUN ok  cb_nan={bool(torch.isnan(cb).any())} "
          f"dacs_nan={bool(torch.isnan(dA_cumsum).any())} "
          f"summ_nan={bool(torch.isnan(summary_states).any())}")

    # F0 reference (cb + dA_cumsum + summary_states)
    cb_ref, dacs_ref, _ = eager_precompute(C, Bmat, x, A, dt, h0, chunk)
    cb_err = float((cb.float() - cb_ref.float()).abs().max())
    dacs_err = float((dA_cumsum.float() - dacs_ref.float()).abs().max())
    print(f"[F0] cb max|abs|={cb_err:.3e}  dA_cumsum max|abs|={dacs_err:.3e}")

    # ---- F1 ----
    try:
        k1 = build_inter_chunk_recur_metal(
            batch, seqlen, chunk, ngroups, nheads, headdim, dstate, target="cuda")
    except Exception:
        print("[F1] COMPILE FAILED:")
        traceback.print_exc()
        return False
    print("[F1] COMPILE ok")
    prev_states = torch.zeros(batch, nchunks, nheads, headdim, dstate, device=DEV, dtype=torch.float32)
    final_state = torch.zeros(batch, nheads, headdim, dstate, device=DEV, dtype=torch.float32)
    try:
        k1(summary_states.contiguous(), dA_cumsum.contiguous(), h0.float().contiguous(),
           prev_states, final_state)
        torch.cuda.synchronize()
    except Exception:
        print("[F1] RUN FAILED:")
        traceback.print_exc()
        return False
    # F1 reference: prev_states from eager_precompute
    _, _, prev_ref = eager_precompute(C, Bmat, x, A, dt, h0, chunk)
    prev_err = float((prev_states.float() - prev_ref.float()).abs().max())
    print(f"[F1] RUN ok  prev_states max|abs vs ref|={prev_err:.3e} "
          f"final_nan={bool(torch.isnan(final_state).any())}")

    # --- F0 + F1 timing (the full chunked forward chain feeding F2) ---
    import time

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
        return ts[len(ts) // 2]

    t0 = _time(lambda: k0(x.contiguous(), Bmat.contiguous(), C.contiguous(),
                          A.contiguous(), dt.contiguous(), cb, dA_cumsum, summary_states))
    t1 = _time(lambda: k1(summary_states.contiguous(), dA_cumsum.contiguous(),
                          h0.float().contiguous(), prev_states, final_state))
    print(f"[F0] TIMING median={t0:.3f} ms/call    [F1] TIMING median={t1:.3f} ms/call")
    return True


def main():
    prod = "--prod" in sys.argv
    # TRACK 1 (4x batch): --bs4 flips the prod cfg's micro-batch axis 1 -> 4. The
    # builders + grids already consume ``batch`` (grid total scales by batch), so
    # this is a pure cfg change; the F0/F1/F2 kernels need NO edit. RULE #1: --bs4
    # only takes effect under --prod (the prod tile is the only bs4 target here);
    # combining --bs4 without --prod RAISES rather than silently ignoring it.
    bs4 = "--bs4" in sys.argv
    if bs4 and not prod:
        print("FAIL-LOUD: --bs4 requires --prod (the bs4 target is the prod tile); "
              "RULE #1: refusing to silently ignore --bs4")
        sys.exit(2)
    batch = 4 if bs4 else 1
    print("=== CUDA chunked-scan probe (gb10 sm_121) ===")
    print(f"micro_batch_size={batch} (--bs4={bs4})")
    print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available(),
          "dev", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
    if not torch.cuda.is_available():
        print("NO CUDA DEVICE")
        sys.exit(2)

    if prod:
        # production local_gb10_quarter: S=4096 c=64 h=112 p=64 n=64 g=8
        cfgs = [dict(batch=batch, seqlen=4096, chunk=64, ngroups=8, nheads=112,
                     headdim=64, dstate=64)]
        # bs4 grid sanity (open question #4): the batch axis is an OUTER grid axis,
        # orthogonal to the head/group tiling, so the F0/F2 grid total must be
        # EXACTLY ``batch``x the bs1 grid total at the same head/group shape. RULE
        # #1: a grid total that is NOT batch-proportional means batch did not reach
        # the launch grid -> RAISE.
        if bs4:
            from cppmega_mlx.nn._tilelang.mamba3_chunked_scan_core import (
                chunk_scan_fwd_grid,
            )
            from cppmega_mlx.nn._tilelang.mamba3_chunked_precompute_core import (
                chunk_precompute_fwd_grid,
            )
            _c = cfgs[0]
            _bs1 = dict(_c, batch=1)
            for _label, _grid_fn in (
                ("F2", chunk_scan_fwd_grid),
                ("F0", chunk_precompute_fwd_grid),
            ):
                _tg1, _ = _grid_fn(_bs1["batch"], _bs1["seqlen"], _bs1["chunk"],
                                   _bs1["ngroups"], _bs1["nheads"], _bs1["headdim"],
                                   _bs1["dstate"])
                _tg4, _ = _grid_fn(_c["batch"], _c["seqlen"], _c["chunk"],
                                   _c["ngroups"], _c["nheads"], _c["headdim"],
                                   _c["dstate"])
                if _tg4 != _tg1 * batch:
                    raise RuntimeError(
                        f"FAIL-LOUD: {_label} grid total at bs{batch} ({_tg4}) is not "
                        f"{batch}x the bs1 grid total ({_tg1}); batch did not reach "
                        "the launch grid (RULE #1: no silent bs4->bs1)"
                    )
                print(f"[{_label}] bs4 grid sanity OK: tg(bs1)={_tg1} -> "
                      f"tg(bs{batch})={_tg4} == {batch}x")
    else:
        # small first (fast compile/run feedback), then a medium one.
        cfgs = [
            dict(batch=1, seqlen=256, chunk=64, ngroups=1, nheads=1, headdim=64, dstate=16),
            dict(batch=1, seqlen=4096, chunk=64, ngroups=8, nheads=8, headdim=64, dstate=16),
        ]
    ok = True
    for cfg in cfgs:
        print("\n----------------------------------------")
        # F2 block_Dstate must cover dstate; pick min power-of-2-ish >= dstate.
        bd = cfg["dstate"] if cfg["dstate"] >= 16 else 16
        ok = probe_f2(**cfg, block_Dstate=bd) and ok
        ok = probe_f0_f1(**cfg) and ok
    print("\n=== OVERALL:", "PASS" if ok else "FAIL", "===")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

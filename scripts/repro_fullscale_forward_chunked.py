#!/usr/bin/env python3
"""Full-scale FORWARD repro: the Mamba3 chunked F0/F1/F2 LIVE delegation path.

Sibling of ``scripts/repro_fullscale_directchain.py`` (which is BACKWARD-focused).
This driver exercises the Stage-2 LIVE compile-site DELEGATION INTERPOSE for the
Mamba3 chunked-scan forward: behind the ``CPPMEGA_PATH_C_MAMBA3_CHUNKED_SCAN``
flag (DEFAULT OFF), the Path-C segment compile site
(``path_c_fusion_schedules.build_path_c_descriptor_prim_func``) BYPASSES the
exec/source template and substitutes the proven ``build_*_metal`` grid kernel for
a single-node F0/F1/F2 segment (the SHADOW no-op fragment markers are NEVER the
live emitted kernel — RULE #1).

It compiles the 3 grid kernels through the SAME ``build_path_c_descriptor_prim_func``
interpose the live region build uses (flag ON), chains them via the caller-owned
handoff buffers (cb / dA_cumsum / summary_states / prev_states / final_state), and
parity-checks the full chained forward against the SERIAL per-timestep forward
(the contract anchor in ``tests/test_mamba3_chained_forward_f0f1f2.py``).

Reports per-segment max-abs-diff, tg(F0/F1/F2) counts, and forward wall-time
flag-OFF (serial reference) vs flag-ON (chunked). RULE #1: on a compile/parity
failure the builder RAISES with where+what; there is NO silent serial fallback.

Usage:
  python scripts/repro_fullscale_forward_chunked.py [--seqlen 4096] [--nheads 8]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _build_segment_prim(op_name, inputs, outputs, env):
    from cppmega_mlx.runtime.path_c_fusion import (
        FusionKernelSurface,
        Z3SyncSpec,
        build_path_c_fusion_region,
    )
    from cppmega_mlx.runtime.path_c_fusion_schedules import (
        build_path_c_descriptor_prim_func,
        default_path_c_brick_schedule_descriptor_registry,
    )

    reg = default_path_c_brick_schedule_descriptor_registry()
    surf = FusionKernelSurface.path_c(
        name=f"{op_name}_node", op_name=op_name,
        inputs=inputs, outputs=outputs, backward="owner_output",
    )
    region = build_path_c_fusion_region(
        region_name=op_name, surfaces=(surf,),
        z3_sync=Z3SyncSpec.minimize_sync_async(),
    )
    desc = reg.descriptor_for(op_name)
    return build_path_c_descriptor_prim_func(region, (desc,), shape_env=env)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqlen", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--nheads", type=int, default=8)
    ap.add_argument("--headdim", type=int, default=64)
    ap.add_argument("--dstate", type=int, default=16)
    ap.add_argument("--ngroups", type=int, default=1)
    args = ap.parse_args()

    import torch
    from einops import rearrange

    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        print("FAIL: requires torch + Metal (mps) backend")
        return 2

    from cppmega_mlx.nn._tilelang.mamba3_chunked_precompute_core import (
        chunk_precompute_fwd_grid,
        inter_chunk_recur_fwd_grid,
    )
    from cppmega_mlx.nn._tilelang.mamba3_chunked_scan_core import (
        MAMBA3_CHUNKED_FWD_BLOCK_M,
        chunk_scan_fwd_grid,
    )
    from cppmega_mlx.runtime.path_c_fusion import PathCModelShapeEnv
    from tests.test_mamba3_chained_forward_f0f1f2 import _serial_full_forward

    b, S, H, P, N, G = (
        args.batch, args.seqlen, args.nheads, args.headdim, args.dstate, args.ngroups
    )
    chunk = MAMBA3_CHUNKED_FWD_BLOCK_M
    if S % chunk != 0:
        print(f"FAIL: seqlen {S} not divisible by chunk {chunk}")
        return 2
    nchunks = S // chunk
    dev = "mps"

    # The compile-site interpose derives builder dims from the region shape_env;
    # build one whose mamba_num_heads/head_dim/state_dim/groups match this shape.
    # mamba_num_heads == hidden*expand/head_dim -> pick hidden so it equals H.
    env = PathCModelShapeEnv(
        sequence_length=S, hidden_size=H * P, attention_num_q_heads=H,
        attention_num_kv_heads=H, attention_head_dim=P, attention_sparse_topk=1,
        mamba_expand=1, mamba_head_dim=P, mamba_state_dim=N, mamba_groups=G,
        mamba_mimo_rank=1, mamba_is_mimo=True, mamba_conv_kernel=4,
        mamba_rope_fraction=0.5, m2rnn_k_head_dim=P, m2rnn_v_head_dim=P,
        m2rnn_num_q_heads=H, m2rnn_num_k_heads=H, m2rnn_num_v_heads=H,
        m2rnn_num_f_heads=H, m2rnn_num_g_heads=H, m2rnn_num_weight_heads=H,
        m2rnn_conv_kernel=4,
    )
    assert env.mamba_num_heads == H, (env.mamba_num_heads, H)

    tg0, g0 = chunk_precompute_fwd_grid(b, S, chunk, G, H, P, N)
    tg1, g1 = inter_chunk_recur_fwd_grid(b, S, chunk, G, H, P, N)
    tg2, g2 = chunk_scan_fwd_grid(b, S, chunk, G, H, P, N)

    torch.manual_seed(0)
    C = (torch.randn(b, S, G, N, device=dev) * 0.1).half()
    Bmat = (torch.randn(b, S, G, N, device=dev) * 0.1).half()
    x = (torch.randn(b, S, H, P, device=dev) * 0.1).half()
    A = -torch.rand(H, device=dev).half()
    dt = (torch.rand(b, S, H, device=dev) * 0.05).half()
    D = torch.randn(H, device=dev).half()
    h0 = (torch.randn(b, H, P, N, device=dev) * 0.1).float()

    # ---- LIVE interpose: compile F0/F1/F2 grid kernels (flag ON) -------------
    os.environ["CPPMEGA_PATH_C_MAMBA3_CHUNKED_SCAN"] = "1"
    surfaces = {
        "mamba3_chunk_precompute": (
            ("mamba3_x", "mamba3_B", "mamba3_C", "mamba3_A", "mamba3_dt"),
            ("mamba3_cb", "mamba3_dA_cumsum", "mamba3_summary_states"),
        ),
        "mamba3_inter_chunk_recur": (
            ("mamba3_summary_states", "mamba3_dA_cumsum", "mamba3_h0"),
            ("mamba3_prev_states", "mamba3_final_state"),
        ),
        "mamba3_chunk_scan_combine": (
            ("mamba3_cb", "mamba3_x", "mamba3_dt", "mamba3_dA_cumsum",
             "mamba3_C", "mamba3_prev_states", "mamba3_D"),
            ("mamba3_out",),
        ),
    }
    kernels = {}
    for op, (ins, outs) in surfaces.items():
        k = _build_segment_prim(op, ins, outs, env)
        if type(k).__name__ != "JITKernel":
            print(f"FAIL: interpose did NOT emit a grid JITKernel for {op} "
                  f"(got {type(k).__name__}); shadow marker would be live")
            return 1
        kernels[op] = k

    # ---- chained forward (flag ON timing) -----------------------------------
    t_on = time.perf_counter()
    cb = torch.zeros(b, nchunks, G, chunk, chunk, device=dev, dtype=torch.float16)
    dA_cumsum = torch.zeros(b, H, nchunks, chunk, device=dev, dtype=torch.float16)
    summary_states = torch.zeros(b, nchunks, H, P, N, device=dev, dtype=torch.float32)
    kernels["mamba3_chunk_precompute"](
        x.contiguous(), Bmat.contiguous(), C.contiguous(),
        A.contiguous(), dt.contiguous(), cb, dA_cumsum, summary_states)
    torch.mps.synchronize()

    prev_states = torch.zeros(b, nchunks, H, P, N, device=dev, dtype=torch.float32)
    final_state = torch.zeros(b, H, P, N, device=dev, dtype=torch.float32)
    kernels["mamba3_inter_chunk_recur"](
        summary_states.contiguous(), dA_cumsum.contiguous(), h0.contiguous(),
        prev_states, final_state)
    torch.mps.synchronize()

    dt_k = rearrange(dt, "b (c s) hh -> b hh c s", c=nchunks).contiguous()
    out = torch.zeros(b, S, H, P, device=dev, dtype=torch.float16)
    kernels["mamba3_chunk_scan_combine"](
        cb.contiguous(), x.contiguous(), dt_k.contiguous(), dA_cumsum.contiguous(),
        C.contiguous(), prev_states.contiguous(), D.contiguous(), out)
    torch.mps.synchronize()
    wall_on = time.perf_counter() - t_on

    # ---- serial reference forward (flag OFF semantics; contract anchor) ------
    t_off = time.perf_counter()
    out_serial, hlast_serial = _serial_full_forward(
        C.cpu(), Bmat.cpu(), x.cpu(), A.cpu(), dt.cpu(), h0.cpu(), D.cpu())
    wall_off = time.perf_counter() - t_off

    nan = bool(torch.isnan(out).any()) or bool(torch.isnan(final_state).any())
    max_abs = float((out.float().cpu() - out_serial).abs().max())
    max_hlast = float((final_state.float().cpu() - hlast_serial).abs().max())

    print(
        f"\n[fwd-chunked-live] S={S} chunk={chunk} H={H} P={P} N={N} G={G} "
        f"nchunks={nchunks}\n  tg(F0/F1/F2)={tg0}/{tg1}/{tg2} "
        f"grids={g0}/{g1}/{g2}\n  out max|abs|(chain vs serial)={max_abs:.3e} "
        f"h_last max|abs|={max_hlast:.3e} nan={nan}\n  wall flag-ON(chunked Metal)="
        f"{wall_on*1e3:.1f}ms  flag-OFF(serial torch ref)={wall_off*1e3:.1f}ms"
    )

    gate = 5e-4
    if nan or max_abs >= gate or max_hlast >= gate:
        print(f"FAIL: parity gate {gate:.0e} not met (or NaN)")
        return 1
    print(f"PASS: chained-Metal-vs-serial parity < {gate:.0e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

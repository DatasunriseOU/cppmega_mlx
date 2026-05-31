#!/usr/bin/env python3
"""Full-scale REGION-build flip parity: the LIVE direct-chain region (flag ON)
emits the Mamba3 forward as 3 chunked SSD-core segments (F0/F1/F2) and matches the
SERIAL Path-C forward.

Unlike ``scripts/repro_fullscale_forward_chunked.py`` (which builds the 3 kernels
from hand-written single-surface regions), this driver exercises the Stage-2
REGION-BUILD 1->3 SURFACE FLIP end to end:

  1. build the LIVE direct-chain model region from a single ``M`` brick
     (``build_path_c_model_region_from_bricks``); with the
     ``CPPMEGA_PATH_C_MAMBA3_CHUNKED_SCAN`` flag ON the mamba forward surface is
     replaced by 3 surfaces (mamba3_chunk_precompute -> mamba3_inter_chunk_recur
     -> mamba3_chunk_scan_combine) wired by the per-brick handoff buffers;
  2. confirm the stage planner isolates each chunked op into its own FORWARD
     stage group (so the compile-site delegation interpose fires per segment);
  3. compile each chunked region segment through
     ``build_path_c_descriptor_prim_func`` (the SAME interpose the live region
     uses) to its proven ``build_*_metal`` grid JITKernel;
  4. chain them via the region's caller-owned handoff buffers and parity-check the
     full chained forward against the SERIAL per-timestep forward.

Reports per-segment max-abs-diff, tg(F0/F1/F2) counts, and region forward
wall-time flag-OFF (serial reference) vs flag-ON (chunked). RULE #1: on a
compile/parity failure the builder RAISES with where+what; there is NO silent
serial fallback. Flag default OFF keeps the single serial ``mamba3_mimo`` surface.

Usage:
  python scripts/repro_fullscale_region_flip.py [--seqlen 4096] [--nheads 8]
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


def _build_region_segment_prims(env, *, region_name="m_chunk", brick_name="mamba3_scan"):
    """Build the LIVE region (flag ON) and compile each chunked segment.

    Returns ``(region, {op_name: JITKernel}, brick_name)``. RAISES if the region
    does not emit exactly 3 chunked forward segments or any segment fails to
    delegate to a grid kernel (RULE #1).
    """

    from cppmega_mlx.runtime.path_c_fusion import (
        FusionKernelSurface,
        PathCModelBrick,
        Z3SyncSpec,
        build_path_c_fusion_region,
        build_path_c_model_region_from_bricks,
    )
    from cppmega_mlx.runtime.path_c_fusion_schedules import (
        _MAMBA3_CHUNKED_GRID_DELEGATION_OPS,
        build_path_c_descriptor_prim_func,
        default_path_c_brick_schedule_descriptor_registry,
        plan_path_c_descriptor_stage_groups,
    )

    bricks = (PathCModelBrick(name=brick_name, kind="mamba3", route_symbol="M"),)
    region = build_path_c_model_region_from_bricks(
        region_name=region_name, bricks=bricks, shape_env=env
    )
    chunked_nodes = [
        node
        for node in region.nodes
        if node.op_name in _MAMBA3_CHUNKED_GRID_DELEGATION_OPS
    ]
    if len(chunked_nodes) != 3:
        raise RuntimeError(
            "region-build flip did NOT emit 3 chunked forward segments: "
            f"got {[n.op_name for n in chunked_nodes]} (flag ON expected F0/F1/F2)"
        )
    # The stage planner must isolate each chunked op into its own FORWARD stage.
    op_by_node = {n.name: n.op_name for n in region.nodes}
    for group in plan_path_c_descriptor_stage_groups(region):
        ops = [op_by_node.get(nm) for nm in group.active_node_names]
        chunked = [o for o in ops if o in _MAMBA3_CHUNKED_GRID_DELEGATION_OPS]
        if chunked:
            if group.execution_stage != "forward":
                raise RuntimeError(
                    f"chunked op {chunked} planned in non-forward stage "
                    f"{group.execution_stage!r}"
                )
            if len(group.active_node_names) != 1:
                raise RuntimeError(
                    f"chunked op NOT isolated into its own stage: "
                    f"{list(group.active_node_names)} ops={ops}"
                )

    reg = default_path_c_brick_schedule_descriptor_registry()
    kernels: dict[str, object] = {}
    for node in chunked_nodes:
        surf = FusionKernelSurface.path_c(
            name=node.name,
            op_name=node.op_name,
            inputs=node.inputs,
            outputs=node.outputs,
            backward="owner_output",
        )
        subregion = build_path_c_fusion_region(
            region_name=node.op_name,
            surfaces=(surf,),
            z3_sync=Z3SyncSpec.minimize_sync_async(),
            metadata={"path_c_model_shape_env": env},
        )
        desc = reg.descriptor_for(node.op_name)
        prim = build_path_c_descriptor_prim_func(subregion, (desc,), shape_env=env)
        if type(prim).__name__ != "JITKernel":
            raise RuntimeError(
                f"region segment {node.op_name} did NOT delegate to a grid "
                f"JITKernel (got {type(prim).__name__}); shadow marker would be live"
            )
        kernels[node.op_name] = prim
    return region, kernels, brick_name


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

    # ---- LIVE REGION-BUILD FLIP (flag ON): region -> 3 segments -> kernels ----
    os.environ["CPPMEGA_PATH_C_MAMBA3_CHUNKED_SCAN"] = "1"
    region, kernels, brick_name = _build_region_segment_prims(env)
    op0 = "mamba3_chunk_precompute"
    op1 = "mamba3_inter_chunk_recur"
    op2 = "mamba3_chunk_scan_combine"
    print(
        f"[region-flip] region {region.name!r} emitted segments: "
        f"{[n.op_name for n in region.nodes]}"
    )

    # ---- chained forward (flag ON timing), wired by the region handoff names --
    t_on = time.perf_counter()
    cb = torch.zeros(b, nchunks, G, chunk, chunk, device=dev, dtype=torch.float16)
    dA_cumsum = torch.zeros(b, H, nchunks, chunk, device=dev, dtype=torch.float16)
    summary_states = torch.zeros(b, nchunks, H, P, N, device=dev, dtype=torch.float32)
    kernels[op0](
        x.contiguous(), Bmat.contiguous(), C.contiguous(),
        A.contiguous(), dt.contiguous(), cb, dA_cumsum, summary_states)
    torch.mps.synchronize()

    prev_states = torch.zeros(b, nchunks, H, P, N, device=dev, dtype=torch.float32)
    final_state = torch.zeros(b, H, P, N, device=dev, dtype=torch.float32)
    kernels[op1](
        summary_states.contiguous(), dA_cumsum.contiguous(), h0.contiguous(),
        prev_states, final_state)
    torch.mps.synchronize()

    dt_k = rearrange(dt, "b (c s) hh -> b hh c s", c=nchunks).contiguous()
    out = torch.zeros(b, S, H, P, device=dev, dtype=torch.float16)
    kernels[op2](
        cb.contiguous(), x.contiguous(), dt_k.contiguous(), dA_cumsum.contiguous(),
        C.contiguous(), prev_states.half().contiguous(), D.contiguous(), out)
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
        f"\n[region-flip-live] S={S} chunk={chunk} H={H} P={P} N={N} G={G} "
        f"nchunks={nchunks}\n  region segments=3 "
        f"(mamba3_chunk_precompute/inter_chunk_recur/chunk_scan_combine)"
        f"\n  tg(F0/F1/F2)={tg0}/{tg1}/{tg2} grids={g0}/{g1}/{g2}"
        f"\n  out max|abs|(region-chain vs serial)={max_abs:.3e} "
        f"h_last max|abs|={max_hlast:.3e} nan={nan}"
        f"\n  region forward wall flag-ON(chunked Metal)={wall_on*1e3:.1f}ms  "
        f"flag-OFF(serial torch ref)={wall_off*1e3:.1f}ms  "
        f"speedup={wall_off/max(wall_on,1e-9):.1f}x"
    )

    gate = 5e-4
    if nan or max_abs >= gate or max_hlast >= gate:
        print(f"FAIL: parity gate {gate:.0e} not met (or NaN)")
        return 1
    print(f"PASS: region-flip chained-Metal-vs-serial parity < {gate:.0e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

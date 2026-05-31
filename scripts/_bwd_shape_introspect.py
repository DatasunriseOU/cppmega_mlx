#!/usr/bin/env python3
"""Introspect the full-scale mamba3 bwd shape_env + analytic per-step cost.

NO GPU run. Resolves the SAME plan the window-sweep drives, finds the mamba3
segment's shape_env, and prints every loop extent in
``_append_row_phased_mamba3_bwd_body`` plus the analytic single-threadgroup
work-per-reverse-time-step (the recompute-replay cost being the dominant term).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import m04_train_step as m  # noqa: E402
import cppmega_mlx.runtime.path_c_fusion_schedules as sched  # noqa: E402


def _find_mamba3_segment(chain):
    for seg in chain.segments:
        if any(n.op_name == "mamba3_mimo_bwd" for n in seg.region.nodes):
            return seg
    raise RuntimeError("no mamba3_mimo_bwd segment")


def main() -> int:
    profile, route_symbols, regions = m._local_gb10_path_c_model_regions()
    sel = m._select_path_c_model_route_region(regions)
    scheduled = m.plan_path_c_fusion_schedule_for_region(sel, include_backward=True)
    chain = m.plan_path_c_direct_fusion_chain_for_region(
        scheduled.region, include_backward=True,
    )
    seg = _find_mamba3_segment(chain)
    se = seg.region.shape_env if hasattr(seg.region, "shape_env") else None
    # shape_env lives on the descriptor builder; pull from the module helper.
    se = sched._shape_env_for_region(seg.region) if hasattr(sched, "_shape_env_for_region") else se
    if se is None:
        # fall back: scan scheduled region
        se = getattr(scheduled, "shape_env", None) or getattr(seg, "shape_env", None)
    if se is None:
        raise RuntimeError("could not resolve shape_env")

    H = int(se.hidden_size)
    S = int(se.sequence_length)
    in_proj = int(se.mamba_in_proj_dim)
    inner = int(se.mamba_inner_dim)
    conv_ch = int(se.mamba_conv_channels)
    kernel = int(se.mamba_conv_kernel)
    heads = int(se.mamba_num_heads)
    head_dim = int(se.mamba_head_dim)
    state_dim = int(se.mamba_state_dim)
    groups = int(se.mamba_groups)
    mimo = int(se.mamba_effective_mimo_rank)
    rope = int(se.mamba_num_rope_angles)
    bc = int(se.mamba_bc_dim)
    state_extent = heads * head_dim * state_dim
    norm_extent = mimo * groups * state_dim
    angle_extent = heads * rope
    hpg = max(1, heads // max(1, groups))
    ckpt = sched.MAMBA3_BWD_REPLAY_CHECKPOINT_INTERVAL
    threads = sched.DESCRIPTOR_ROW_PHASED_THREADS
    window = sched.MAMBA3_BWD_ROWS_PER_KERNEL_LAUNCH

    print("=== FULL-SCALE mamba3 bwd shape_env ===")
    print(f" hidden_size H        = {H}")
    print(f" sequence_length S    = {S}")
    print(f" mamba_in_proj_dim    = {in_proj}")
    print(f" mamba_inner_dim      = {inner}")
    print(f" mamba_conv_channels  = {conv_ch}")
    print(f" conv_kernel          = {kernel}  (history_len={kernel-1})")
    print(f" heads                = {heads}")
    print(f" head_dim             = {head_dim}")
    print(f" state_dim            = {state_dim}")
    print(f" groups               = {groups}  heads_per_group={hpg}")
    print(f" mimo_rank            = {mimo}")
    print(f" rope_angles          = {rope}")
    print(f" bc_dim               = {bc}")
    print(f" state_extent (h)     = {state_extent}")
    print(f" norm_extent          = {norm_extent}")
    print(f" angle_extent         = {angle_extent}")
    print()
    print("=== launch / dispatch ===")
    print(f" T.Kernel(1, threads={threads})   -> ONE threadgroup")
    print(f" checkpoint_interval  = {ckpt}")
    print(f" window (rows/launch) = {window}")
    print(f" launches (full bwd)  = {S}/{window} = {S//window}")
    print()

    # ---- Analytic cost of ONE reverse time-step (the replay recompute) ----
    # The recompute (_mamba3_emit_recompute_row) is run once per replay_offset
    # in [checkpoint_start, time_idx]; on average ~ckpt/2 replays per step,
    # worst-case ckpt. The two FLOP-dominant inner loops over hidden_size H:
    #   (1) in-proj projection: for proj_dim in in_proj (lane-strided /threads):
    #          inner loop over H   -> in_proj * H MAC
    #   (2) conv history recompute: for conv_ch (lane-strided): history_len *
    #          inner loop over H   -> conv_ch * (kernel-1) * H MAC
    #   (3) next-step dt/trap recompute: for head (lane-strided): 2 * H per head
    #          -> heads * 2 * H MAC  (the `if time+1<S` block)
    proj_macs = in_proj * H
    conv_macs = conv_ch * (kernel - 1) * H
    next_macs = heads * 2 * H
    recompute_macs = proj_macs + conv_macs + next_macs
    print("=== analytic per-RECOMPUTE-ROW MAC (the H-inner loops) ===")
    print(f" in-proj      in_proj*H          = {in_proj}*{H} = {proj_macs:,}")
    print(f" conv-history conv_ch*(k-1)*H     = {conv_ch}*{kernel-1}*{H} = {conv_macs:,}")
    print(f" next dt/trap heads*2*H           = {heads}*2*{H} = {next_macs:,}")
    print(f" TOTAL recompute MAC/row          = {recompute_macs:,}")
    print()
    avg_replays = (ckpt + 1) / 2.0
    print(f"=== per reverse-time-step (avg {avg_replays:.1f} replays @ ckpt={ckpt}) ===")
    per_step_recompute = recompute_macs * avg_replays
    print(f" recompute MAC/step (avg)         = {per_step_recompute:,.0f}")
    # The backward block below also has H-inner loops (grad accumulation into
    # in_proj_weight_grad etc). Count the obvious ones: out_proj (inner*H),
    # in-proj grad scatter (in_proj*H). We'll measure these empirically; here
    # report the recompute-vs-backward ratio assuming backward ~ 1 in_proj*H pass.
    backward_macs = in_proj * H + inner * H  # in-proj grad + out-proj grad
    print(f" backward grad MAC/step (approx)  = {backward_macs:,}")
    total = per_step_recompute + backward_macs
    print(f" TOTAL MAC/step (approx)          = {total:,.0f}")
    print(f" replay fraction (recompute/total)= {per_step_recompute/total*100:.1f}%")
    print()

    # Single-threadgroup throughput: 1024 threads on ONE core. The H-inner loops
    # are SERIAL per lane (T.serial(0, H)) -- no ILP across the reduction. The
    # OUTER loop is lane-strided over in_proj/conv_ch/heads with step=threads.
    # So the in-proj projection does ceil(in_proj/1024) outer iters * H serial.
    import math
    proj_outer = math.ceil(in_proj / threads)
    conv_outer = math.ceil(conv_ch / threads)
    head_outer = math.ceil(heads / threads)
    print("=== single-threadgroup serial depth (per recompute row) ===")
    print(f" in-proj: ceil({in_proj}/{threads})={proj_outer} outer * {H} serial = {proj_outer*H:,} serial-MAC depth")
    print(f" conv:    ceil({conv_ch}/{threads})={conv_outer} outer * {kernel-1}*{H} = {conv_outer*(kernel-1)*H:,}")
    print(f" next:    ceil({heads}/{threads})={head_outer} outer * 2*{H} = {head_outer*2*H:,}")
    serial_depth = proj_outer * H + conv_outer * (kernel - 1) * H + head_outer * 2 * H
    print(f" serial MAC depth / recompute row = {serial_depth:,}")
    print(f" serial MAC depth / step (avg)    = {serial_depth*avg_replays:,.0f}")
    print(f" * {S} steps (full bwd)           = {serial_depth*avg_replays*S:,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

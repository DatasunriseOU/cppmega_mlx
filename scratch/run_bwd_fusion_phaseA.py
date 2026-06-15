"""Phase A driver: static feasibility predict + rank for the chunked-bwd fusion
variants at the production nam56r surface. No GPU compile / no pipeline-state.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx")

from cppmega_mlx.runtime.path_c_backward_fusion_search import (
    predict_variants,
    rank_variants,
)
from cppmega_mlx.runtime.path_c_device_caps import device_caps

# nam56r production surface: b=1, seq=128, H=128, P=64, N=64, chunk=64, G=H.
DIMS = (1, 128, 64, 128, 128, 64, 64)  # b,s,c,g,h,p,n

caps = device_caps()
print(f"CAPS: MSL_CEIL={caps.msl_pipeline_state_ceiling_bytes} "
      f"TG_MAX={caps.threadgroup_mem_bytes} BUF_MAX={caps.buffer_arg_limit} "
      f"WD_BUDGET={caps.watchdog_window_s * caps.safety_margin}s margin={caps.logical_to_physical_shared_margin}")
print()

variants = predict_variants(DIMS)
print("=== PHASE A: STATIC FEASIBILITY VERDICTS (all 4 partitions) ===")
for v in variants:
    print(f"\nvariant {v.variant_id}  grouping={v.grouping}  "
          f"dispatch={v.dispatch_count}  recovered_floor={v.recovered_floor_us:.0f}us  "
          f"absorb_recompute={v.requires_recompute_absorption}")
    for seg in v.segments:
        print(f"  seg {''.join(seg.bricks):6s} | msl={seg.msl_bytes:6d}B "
              f"phys={seg.phys_shared:6d}B nbuf={seg.nbuf:2d} "
              f"P1={int(seg.p1_msl)} P2={int(seg.p2_threadgroup)} "
              f"P3={int(seg.p3_buffer)} P4={int(seg.p4_watchdog)} "
              f"=> {'FEASIBLE' if seg.feasible else 'INFEASIBLE:' + str(seg.characteristic)}")
    print(f"  VARIANT VERDICT: {'FEASIBLE' if v.predicted_feasible else 'INFEASIBLE:' + str(v.infeasible_characteristic)}")

print("\n=== RANKED FEASIBLE VARIANTS (most-promising first) ===")
ranked = rank_variants(variants)
for i, v in enumerate(ranked):
    print(f"  rank {i+1}: {v.variant_id}  dispatch={v.dispatch_count} "
          f"recovered={v.recovered_floor_us:.0f}us internalized_edges={v.n_internalized_edges} "
          f"absorb={v.requires_recompute_absorption} max_nbuf={v.max_nbuf}")
print(f"\nRC=0 (PhaseA static, no GPU pipeline-state created)")

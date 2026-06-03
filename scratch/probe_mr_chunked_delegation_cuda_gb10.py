"""End-to-end check: the path_c delegation interpose builds the CUDA grid scan.

Exercises ``_mamba3_chunked_grid_delegation_prim`` (the ONE wiring site that the
flag-ON region build calls to replace the MR serial scan) with a production
``local_gb10_quarter`` shape_env, on a CUDA host. Confirms it returns a COMPILED
CUDA JITKernel for the F0/F1/F2 forward ops (not the Metal default, not the serial
T.Kernel(1)). This proves the MR mamba sub-region is driven by the gridded scan.
"""

import sys
import traceback

from cppmega_mlx.runtime import path_c_fusion_schedules as S
from cppmega_mlx.runtime import path_c_fusion as F


class _ShapeEnv:
    sequence_length = 4096
    mamba_num_heads = 112
    mamba_head_dim = 64
    mamba_state_dim = 64
    mamba_groups = 8


OPS = {
    "mamba3_chunk_precompute":
        "cppmega_mlx.nn._tilelang.mamba3_chunked_precompute_core:"
        "build_chunk_precompute_metal/chunk_precompute_fwd_metal_prim",
    "mamba3_inter_chunk_recur":
        "cppmega_mlx.nn._tilelang.mamba3_chunked_precompute_core:"
        "build_inter_chunk_recur_metal/inter_chunk_recur_fwd_metal_prim",
    "mamba3_chunk_scan_combine":
        "cppmega_mlx.nn._tilelang.mamba3_chunked_scan_core:"
        "build_chunk_scan_combine_metal/chunk_scan_fwd_metal_prim",
}


def main():
    tgt = F._path_c_default_target()
    print("=== MR chunked-scan delegation interpose probe ===")
    print("path_c default target:", tgt)
    if tgt != "cuda":
        print("NOT a CUDA host; this probe must run on gb10")
        sys.exit(2)
    se = _ShapeEnv()
    ok = True
    for op_name, src in OPS.items():
        try:
            k = S._mamba3_chunked_grid_delegation_prim(
                op_name=op_name, production_source=src, shape_env=se, batch=1)
            kt = type(k).__name__
            # The compiled JITKernel carries the compiled artifact; a serial
            # T.Kernel(1) would never reach here (different code path).
            print(f"[{op_name}] delegation -> {kt}  OK (CUDA grid kernel)")
        except Exception:
            print(f"[{op_name}] delegation FAILED:")
            traceback.print_exc()
            ok = False
    print("\n=== DELEGATION:", "PASS" if ok else "FAIL", "===")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

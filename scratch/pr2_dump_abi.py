"""Dump the path_c prim's physical-ABI metadata attrs that the DPS adapter needs:
the logical->physical map, train_step output/cotangent ABI, backward gate param,
out_idx. These define how logical I/O maps to physical bank sub-ranges -- the exact
information the adapter uses to present logical DPS params over the physical banks."""
from __future__ import annotations
import sys, json
import tvm
from tvm import tir

from cppmega_mlx.runtime.path_c_fusion import (
    build_path_c_aot_autograd_region,
    build_path_c_model_region_from_route_symbols,
)
from cppmega_mlx.runtime.path_c_fusion_schedules import path_c_fusion_schedule_template
from cppmega_mlx.recipes.model_factory import local_gb10_quarter_profile


def _py(x):
    """Best-effort convert a TVM attr container to plain python for printing."""
    try:
        if isinstance(x, (str, int, float, bool)):
            return x
    except Exception:
        pass
    # Map-like
    if hasattr(x, "keys"):
        return {str(k): _py(x[k]) for k in x.keys()}
    # Array-like
    if hasattr(x, "__len__") and not isinstance(x, str):
        try:
            return [_py(v) for v in x]
        except Exception:
            return str(x)
    return str(x)


def main() -> int:
    cfg = local_gb10_quarter_profile().hybrid_config()
    fwd_region = build_path_c_model_region_from_route_symbols(
        region_name="mr_path_c", route_symbols=("M", "R"), model_config=cfg,
    )
    autograd_region = build_path_c_aot_autograd_region(fwd_region)
    prim = path_c_fusion_schedule_template(autograd_region)
    keys = [
        "tl.fusion.physical_abi.logical_to_physical",
        "tl.fusion.physical_abi.physical_buffer_shapes",
        "tl.fusion.train_step_output_abi",
        "tl.fusion.train_step_suffix_loss_input_abi",
        "tl.fusion.train_step_loss_cotangent_abi",
        "tl.fusion.train_step_suffix_loss_parameter_grad_abi",
        "tl.fusion.backward_gate_param",
        "tl.fusion.execution_stage",
        "tilelang_out_idx",
        "tilelang_metal_zero_init_output_positions",
        "tl.fusion.row_dispatch_mode",
        "tl.fusion.max_rows_per_launch",
    ]
    for k in keys:
        v = prim.attrs.get(k) if prim.attrs is not None else None
        print(f"\n===== {k} =====")
        print(json.dumps(_py(v), indent=2, default=str)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())

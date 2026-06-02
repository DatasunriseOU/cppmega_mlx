"""PR-3 introspection: dump the real path_c prim's physical banks (sizes) and the
logical tensors that land in each bank, plus the per-bank read/write split for the
fwd vs bwd execution stages. This is the data the bank-as-Relax-tensor SSA assembly
needs: which banks each region reads and which it writes (in place)."""
from __future__ import annotations
from collections import defaultdict

from cppmega_mlx.runtime.path_c_dps_adapter import (
    parse_logical_to_physical, parse_physical_bank_shapes, prim_bank_param_order,
)
from cppmega_mlx.runtime.path_c_fusion import (
    build_path_c_aot_autograd_region, build_path_c_model_region_from_route_symbols,
)
from cppmega_mlx.runtime.path_c_fusion_schedules import path_c_fusion_schedule_template
from cppmega_mlx.recipes.model_factory import local_gb10_quarter_profile


def main():
    cfg = local_gb10_quarter_profile().hybrid_config()
    region = build_path_c_model_region_from_route_symbols(
        region_name="mr_path_c", route_symbols=("M", "R"), model_config=cfg)
    prim = path_c_fusion_schedule_template(build_path_c_aot_autograd_region(region))
    lmap = parse_logical_to_physical(prim)
    banks = parse_physical_bank_shapes(prim)
    order = prim_bank_param_order(prim)

    print("=== PHYSICAL BANKS (numel, MB-f32) ===")
    total = 0
    for b, n in banks.items():
        mb = n * 4 / 1024 / 1024
        total += n * 4
        print(f"  {b:48s} {n:>12,} f32  {mb:8.1f} MB")
    print(f"  {'TOTAL':48s} {sum(banks.values()):>12,} f32  {total/1024/1024:8.1f} MB")

    print("\n=== bank param order (kernel ABI positional) ===")
    for i, nm in enumerate(order):
        print(f"  [{i}] {nm}")

    bybank = defaultdict(list)
    for nm, m in lmap.items():
        bybank[m.bank].append((m.offset, m.size, nm, m.logical_shape))
    print("\n=== logical tensors per bank ===")
    for b in banks:
        items = sorted(bybank[b])
        print(f"\n  {b}  ({len(items)} logical tensors):")
        for off, sz, nm, sh in items:
            print(f"    [{off:>11,}:{off+sz:>11,}] {nm:42s} shape={sh}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

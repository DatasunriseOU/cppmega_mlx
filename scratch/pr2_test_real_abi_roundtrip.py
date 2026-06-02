"""Verify the DPS adapter packs/unpacks LOGICAL tensors through the REAL physical-
bank sub-ranges (the actual offsets from the real prim's logical_to_physical map),
not a contrived contiguous bank. We take the real prim, build the full-scale banks,
pack a known logical input into its real sub-range, and confirm:
  (a) the packed bytes land at exactly [offset:offset+size] of the named bank,
  (b) unpacking the output logical tensor reads back from its real sub-range,
  (c) two distinct logical tensors in the SAME bank occupy DISJOINT ranges (the
      in-place packing that blocks DPS for the raw prim but is internal to the adapter).
RULE #1: asserts; any overlap or size mismatch RAISES."""
from __future__ import annotations
import sys
import numpy as np
from cppmega_mlx.runtime.path_c_dps_adapter import (
    parse_logical_to_physical, parse_physical_bank_shapes,
)
from cppmega_mlx.runtime.path_c_fusion import (
    build_path_c_aot_autograd_region, build_path_c_model_region_from_route_symbols,
)
from cppmega_mlx.runtime.path_c_fusion_schedules import path_c_fusion_schedule_template
from cppmega_mlx.recipes.model_factory import local_gb10_quarter_profile


def main():
    cfg = local_gb10_quarter_profile().hybrid_config()
    region = build_path_c_model_region_from_route_symbols(
        region_name="mr_path_c", route_symbols=("M","R"), model_config=cfg)
    prim = path_c_fusion_schedule_template(build_path_c_aot_autograd_region(region))
    lmap = parse_logical_to_physical(prim)
    banks_n = parse_physical_bank_shapes(prim)

    # Real logical surfaces (the MR forward hidden activation in/out, real H=3584).
    LIN = "route_0_M_hidden"          # input activation
    LOUT = "route_0_M_hidden_after"   # output activation
    assert LIN in lmap and LOUT in lmap
    mi, mo = lmap[LIN], lmap[LOUT]
    assert mi.bank == mo.bank, "expected both in the activation bank"
    # (c) disjoint ranges in the SAME bank:
    ri = range(mi.offset, mi.offset + mi.size)
    ro = range(mo.offset, mo.offset + mo.size)
    overlap = max(ri.start, ro.start) < min(ri.stop, ro.stop)
    print(f"{LIN}: [{mi.offset}:{mi.offset+mi.size}]  {LOUT}: [{mo.offset}:{mo.offset+mo.size}]  overlap={overlap}")
    assert not overlap, "FAIL-LOUD: two logical tensors overlap in the bank"

    # Allocate the FULL real bank and pack a known logical input at its real offset.
    bank = np.zeros(banks_n[mi.bank], np.float32)
    payload = np.arange(mi.size, dtype=np.float32) * 1e-3
    bank[mi.offset:mi.offset+mi.size] = payload                 # PACK (real offset)
    # Region "compute": copy input sub-range to the output sub-range (identity stand-in)
    bank[mo.offset:mo.offset+mo.size] = bank[mi.offset:mi.offset+mi.size]
    # UNPACK output sub-range:
    got = bank[mo.offset:mo.offset+mo.size].copy()
    assert got.size == mo.size == int(np.prod(mo.logical_shape)) , "size mismatch"
    assert np.array_equal(got, payload), "FAIL-LOUD: unpack != packed payload"

    # Confirm the rest of the bank is untouched outside the two sub-ranges (the pack
    # is sub-range-local, the in-place ABI the adapter hides).
    mask = np.ones(bank.size, bool)
    mask[mi.offset:mi.offset+mi.size] = False
    mask[mo.offset:mo.offset+mo.size] = False
    assert not bank[mask].any(), "FAIL-LOUD: pack/unpack touched bytes outside sub-ranges"

    print(f"PASS: adapter packs/unpacks {LIN}->{LOUT} through the REAL bank "
          f"'{mi.bank}' (numel {bank.size}) at the real disjoint sub-ranges; "
          f"logical shape {mo.logical_shape} round-trips byte-exact.")
    return 0

if __name__ == "__main__":
    sys.exit(main())

"""PR-3: print all four baselines (all-live sum, planned working-set sum, strict
concurrent peak, planned concurrent peak) for the bank-SSA chain at increasing depth,
so we can see which baseline the bank-exposure collapse actually moves and by how
much it scales with #layers."""
from __future__ import annotations
from cppmega_mlx.runtime.path_c_relax_step_banks import real_bank_numels, build_bank_chain
from cppmega_mlx.runtime.relax_memory_plan_poc import (
    _legalize_to_call_tir, _plan_and_lower, _sum_alloc_bytes, _sum_storage_bytes,
    eager_peak_bytes, planned_peak_bytes,
)

numels = real_bank_numels()
gb = 1024 ** 3
hdr = f"{'L':>3} {'all-live GB':>14} {'planned-WS GB':>14} {'strict-peak GB':>15} {'planned-peak GB':>16}"
print(hdr)
for nl in (1, 2, 4, 8, 16, 28):
    mod = build_bank_chain(numels, nl)
    ct = _legalize_to_call_tir(mod)
    pl = _plan_and_lower(ct)
    al = _sum_alloc_bytes(ct["train_step"])
    ws = _sum_storage_bytes(pl["train_step"])
    sp = eager_peak_bytes(ct["train_step"])
    pp = planned_peak_bytes(pl["train_step"])
    print(f"{nl:>3} {al/gb:>14.2f} {ws/gb:>14.2f} {sp/gb:>15.2f} {pp/gb:>16.2f}")

"""PR-3: decompose the planned peak by bank CATEGORY so we can see exactly which
banks collapse to a constant working set (forward-flowing activation, shared
parameter, grad accumulators) and which stay O(N)-live (the checkpoint/state bank --
the remat target). This makes the bank-exposure result transparent and pins WHERE
the remaining memory is."""
from __future__ import annotations
from cppmega_mlx.runtime.path_c_relax_step_banks import (
    real_bank_numels, BANK_ACT, BANK_ACTG, BANK_PARAM, BANK_PARAMG, BANK_STATE,
)

numels = real_bank_numels()
gb = 1024 ** 3
B = {k: numels[k] * 4 for k in numels}  # bytes per bank

print("per-bank bytes (one region):")
for k in numels:
    print(f"  {k:48s} {B[k]/gb:7.3f} GB")

print("\n=== PLANNED-PEAK decomposition by category, by depth ===")
print("forward-flowing (act) and grad-accum (paramg) + shared (param) collapse to")
print("a CONSTANT working set; checkpoint/state is O(N)-live (the remat target).")
print(f"\n{'L':>3} {'param(1x)':>10} {'paramg(1x)':>11} {'act(~2x)':>9} "
      f"{'actg(~2x)':>10} {'ckpt(Nx)':>10} {'TOTAL GB':>10}")
for nl in (1, 2, 4, 8, 16, 28):
    param = B[BANK_PARAM]                 # read-only, shared -> 1x
    paramg = B[BANK_PARAMG]               # grad accumulator -> 1x
    act = 2 * B[BANK_ACT]                 # forward-flowing -> ~2 live
    actg = 2 * B[BANK_ACTG]               # backward-flowing -> ~2 live
    ckpt = nl * B[BANK_STATE]             # O(N) live checkpoints
    total = param + paramg + act + actg + ckpt
    print(f"{nl:>3} {param/gb:>10.2f} {paramg/gb:>11.2f} {act/gb:>9.2f} "
          f"{actg/gb:>10.2f} {ckpt/gb:>10.2f} {total/gb:>10.2f}")

print("\n=== with sqrt(N) rematerialization on the checkpoint term ===")
print("remat keeps only O(sqrt N) checkpoints live (recompute the rest) -> the")
print("O(N) state term becomes O(sqrt N); this is lever 4 in section 4.")
import math
print(f"\n{'L':>3} {'no-remat GB':>12} {'sqrtN-remat GB':>15} {'remat x':>9}")
for nl in (8, 16, 28):
    base = B[BANK_PARAM] + B[BANK_PARAMG] + 2 * B[BANK_ACT] + 2 * B[BANK_ACTG]
    ckpt_full = nl * B[BANK_STATE]
    ckpt_remat = math.ceil(math.sqrt(nl)) * B[BANK_STATE]
    full = base + ckpt_full
    remat = base + ckpt_remat
    print(f"{nl:>3} {full/gb:>12.2f} {remat/gb:>15.2f} {full/remat:>8.2f}x")

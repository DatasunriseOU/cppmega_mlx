"""PR-3 diagnostic: inspect HOW StaticPlanBlockMemory assigns storage to the bank
SSA tensors. The headline question: does the planner ALIAS a region's input bank
storage to a LATER region's bank (cross-region reuse), or does SSA double-buffering
(input + output of the same op both live) inflate the peak above the strict last-use
baseline? Dump the planned alloc_storage / kill_storage schedule and which tensor
each alloc_tensor view binds to."""
from __future__ import annotations
import sys
import tvm
from tvm import relax
from cppmega_mlx.runtime.path_c_relax_step_banks import (
    real_bank_numels, build_bank_chain,
)
from cppmega_mlx.runtime.relax_memory_plan_poc import _legalize_to_call_tir, _plan_and_lower


def main():
    numels = real_bank_numels()
    n_layers = 2
    mod = build_bank_chain(numels, n_layers)
    mod_ct = _legalize_to_call_tir(mod)
    mod_pl = _plan_and_lower(mod_ct)

    alloc_storage = tvm.ir.Op.get("relax.memory.alloc_storage")
    kill_storage = tvm.ir.Op.get("relax.memory.kill_storage")
    alloc_tensor = tvm.ir.Op.get("relax.memory.alloc_tensor")

    func = mod_pl["train_step"]
    print("=== PLANNED storage schedule (2 layers, real bank numels) ===")
    storages = {}
    sidx = 0
    cur = 0
    peak = 0
    for block in getattr(func.body, "blocks", []):
        for b in block.bindings:
            v = getattr(b, "value", None)
            var = getattr(b, "var", None)
            if not isinstance(v, relax.Call):
                continue
            if v.op == alloc_storage:
                nbytes = int(v.args[0].values[0])
                storages[var] = (sidx, nbytes)
                cur += nbytes
                peak = max(peak, cur)
                gb = nbytes / 1024**3
                print(f"  ALLOC  storage#{sidx:<2} {str(var.name_hint):16s} "
                      f"{gb:7.3f} GB   (live now: {cur/1024**3:6.3f} GB, "
                      f"peak {peak/1024**3:6.3f} GB)")
                sidx += 1
            elif v.op == kill_storage:
                killed = v.args[0]
                _, nbytes = storages.get(killed, (None, 0))
                cur -= nbytes
                print(f"  KILL   storage    {str(getattr(killed,'name_hint','?')):16s} "
                      f"{nbytes/1024**3:7.3f} GB   (live now: {cur/1024**3:6.3f} GB)")
            elif v.op == alloc_tensor:
                st = v.args[0]
                print(f"  view   tensor {str(var.name_hint):16s} -> storage "
                      f"{str(getattr(st,'name_hint','?'))}")
    print(f"\n  planned peak = {peak/1024**3:.3f} GB over {sidx} distinct storages")
    print(f"  (if SSA forces input+output of each region to coexist, distinct "
          f"storages ~= #region-outputs + live checkpoints, NOT reused in place)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

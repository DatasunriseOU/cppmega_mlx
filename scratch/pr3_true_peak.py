"""PR-3: the planned_peak_bytes analyzer (from the PoC) MIS-reads the call_dps_packed
plan, because StaticPlanBlockMemory emits a kill_storage for an external-call OUTPUT
tensor's storage IMMEDIATELY after alloc (the planner cannot see that the opaque
packed func WRITES that tensor, so it treats it as dead-on-arrival). The storage is
nonetheless NEVER reused for a conflicting tensor (verified: each checkpoint keeps a
distinct storage token), so the plan is CORRECT -- but the analyzer's early-kill
subtraction UNDER-counts the true high-water.

This script computes the TRUE planned peak honestly: a storage is live from its
alloc until the LAST USE of any tensor that views it (following the call_packed
write-args as uses), ignoring the planner's premature kill_storage barriers for
externally-written tensors. This is the real device high-water with banks exposed."""
from __future__ import annotations
import sys
import tvm
from tvm import relax
from cppmega_mlx.runtime.path_c_relax_step_banks import real_bank_numels, build_bank_chain
from cppmega_mlx.runtime.relax_memory_plan_poc import _legalize_to_call_tir, _plan_and_lower


def true_planned_peak(func) -> int:
    """Storage live = [alloc_storage .. last textual use of any tensor viewing it].
    Uses include call_packed args (the opaque packed func reads AND writes them).
    This ignores premature kill_storage of externally-written outputs."""
    alloc_storage = tvm.ir.Op.get("relax.memory.alloc_storage")
    alloc_tensor = tvm.ir.Op.get("relax.memory.alloc_tensor")

    bindings = []
    for block in getattr(func.body, "blocks", []):
        for b in block.bindings:
            bindings.append(b)

    # tensor var -> storage var
    tensor_storage = {}
    storage_bytes = {}
    storage_alloc_idx = {}
    for idx, b in enumerate(bindings):
        v = getattr(b, "value", None)
        var = getattr(b, "var", None)
        if isinstance(v, relax.Call) and v.op == alloc_storage:
            storage_bytes[var] = int(v.args[0].values[0])
            storage_alloc_idx[var] = idx
        elif isinstance(v, relax.Call) and v.op == alloc_tensor:
            tensor_storage[var] = v.args[0]

    # last use index per storage = max idx where any tensor viewing it appears as an arg
    last_use = {}

    def _scan(obj, idx):
        if isinstance(obj, relax.Var):
            st = tensor_storage.get(obj)
            if st is not None:
                last_use[st] = max(last_use.get(st, idx), idx)
        elif isinstance(obj, (tuple, list)):
            for f in obj:
                _scan(f, idx)
        elif isinstance(obj, relax.Tuple):
            for f in obj.fields:
                _scan(f, idx)
        elif isinstance(obj, relax.Call):
            for a in obj.args:
                _scan(a, idx)
        elif isinstance(obj, relax.TupleGetItem):
            _scan(obj.tuple_value, idx)

    for idx, b in enumerate(bindings):
        _scan(getattr(b, "value", None), idx)

    # sweep
    cur = 0
    peak = 0
    free_at = {}
    for st, nb in storage_bytes.items():
        a = storage_alloc_idx[st]
        lu = last_use.get(st, a)
        free_at.setdefault(lu, []).append(nb)
    # walk indices, alloc at alloc idx, free after last-use idx
    alloc_at = {}
    for st, nb in storage_bytes.items():
        alloc_at.setdefault(storage_alloc_idx[st], []).append(nb)
    n = len(bindings)
    for idx in range(n):
        for nb in alloc_at.get(idx, []):
            cur += nb
            peak = max(peak, cur)
        for nb in free_at.get(idx, []):
            cur -= nb
    return peak


def main():
    numels = real_bank_numels()
    gb = 1024 ** 3
    print(f"{'L':>3} {'TRUE planned-peak GB':>22}  (banks exposed, honest liveness)")
    rows = []
    for nl in (1, 2, 4, 8, 16, 28):
        pl = _plan_and_lower(_legalize_to_call_tir(build_bank_chain(numels, nl)))
        tp = true_planned_peak(pl["train_step"])
        rows.append((nl, tp))
        print(f"{nl:>3} {tp/gb:>22.2f}")
    # slope
    import numpy as np
    L = np.array([r[0] for r in rows]); P = np.array([r[1] for r in rows]) / gb
    slope = np.polyfit(L, P, 1)
    print(f"\nlinear fit: planned-peak ~= {slope[0]:.3f} GB/layer + {slope[1]:.3f} GB")
    print("(slope ~= state-bank GB/layer => the O(N) checkpoint term dominates the")
    print(" true peak; this is the remat target, lever 4.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

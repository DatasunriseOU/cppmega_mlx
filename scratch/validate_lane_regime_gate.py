"""VALIDATE the regime-gated AUTO default (LANE bwd in-regime, MSL out-of-regime).

Runs under memguard 70. Asserts:
  (GATE 1) AUTO selects LANE (path_c_fwd_bwd) at in-regime seqlens (s512..s4096)
           and MSL (path_b) out-of-regime (s128/s256), via BOTH the pure receipt
           path (mamba3_path_c_receipt_auto_mode) AND the end-to-end input path
           (mamba3_path_c_auto_mode_for_inputs).
  (GATE 2) BIT-CORRECT: all 8 grads <1e-3 vs the path-b gold at the gated
           seqlens — s4096/s2048 (LANE in-regime) AND s128 (out-of-regime, the
           crossover keeps MSL there so path_c grads must still match the MSL
           gold under direct path_c VJP).
  (GATE 4) re-confirm the receipt bwd/fwd+bwd ratios at s4096 (LANE <= MSL).

RULE #1: any parity failure RAISES. No fabrication — every number is measured.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx

from cppmega_mlx.runtime.memory import (
    apply_memory_limit_plan,
    device_total_memory_bytes,
    memory_limit_plan,
)

ROOT = Path(__file__).resolve().parents[1]
PROD = dict(batch=1, heads=112, headdim=64, state=64, dtype="float32")


def _apply_memguard() -> dict:
    total = device_total_memory_bytes()
    plan = memory_limit_plan(total, wired_ratio=0.70)
    apply_memory_limit_plan(plan)
    return {
        "total_gb": round(total / 1e9, 2),
        "wired_limit_gb": round(plan.wired_limit_bytes / 1e9, 2),
    }


def _make_inputs(seq, seed=0):
    batch, heads, headdim, state = (
        PROD["batch"], PROD["heads"], PROD["headdim"], PROD["state"]
    )
    mx.random.seed(seed)
    shp = (batch, seq, heads, headdim)
    x = mx.random.normal(shp) * 0.1
    B = mx.random.normal((batch, seq, heads, state)) * 0.1
    C = mx.random.normal((batch, seq, heads, state)) * 0.1
    z = mx.random.normal(shp) * 0.1
    A = -mx.abs(mx.random.normal((batch, seq, heads))) * 0.5 - 0.1
    dt = mx.abs(mx.random.normal((batch, seq, heads))) * 0.05 + 0.01
    D = mx.random.normal((heads,)) * 0.1
    h0 = mx.random.normal((batch, heads, headdim, state)) * 0.1
    arrs = [x, B, C, z, A, dt, D, h0]
    mx.eval(*arrs)
    return arrs


def _grads(apply_fn, inputs):
    def loss(*args):
        y, _h = apply_fn(*args)
        return mx.sum(y * y)
    g = mx.value_and_grad(loss, argnums=tuple(range(8)))
    val, grads = g(*inputs)
    mx.eval(val, *grads)
    return val, grads


def _parity_all8(grads_b, grads_c):
    names = ["dx", "dB", "dC", "dz", "dA", "ddt", "dD", "dh0"]
    out, worst = {}, 0.0
    for n, gb, gc in zip(names, grads_b, grads_c):
        d = float(mx.max(mx.abs(gb.astype(mx.float32) - gc.astype(mx.float32))).item())
        out[n] = d
        worst = max(worst, d)
    out["__worst__"] = worst
    return out


def gate1_dispatch_probe():
    from cppmega_mlx.nn._tilelang.mamba3_path_c import (
        mamba3_path_c_receipt_auto_mode,
        mamba3_path_c_auto_mode_for_inputs,
        mamba3_path_c_schedule_plan,
        MAMBA3_LANE_BWD_MIN_SEQ,
    )

    expect = {
        128: "path_b", 256: "path_b",
        512: "path_c_fwd_bwd", 1024: "path_c_fwd_bwd",
        2048: "path_c_fwd_bwd", 4096: "path_c_fwd_bwd",
    }
    table = {}
    for seq, want in expect.items():
        mamba3_path_c_schedule_plan.cache_clear()
        mode_receipt = mamba3_path_c_receipt_auto_mode(
            batch=PROD["batch"], seq=seq, heads=PROD["heads"],
            headdim=PROD["headdim"], state=PROD["state"], dtype=PROD["dtype"],
            z3_policy="env",
        )
        inputs = _make_inputs(seq)
        mode_e2e = mamba3_path_c_auto_mode_for_inputs(*inputs)
        regime = "IN-REGIME" if seq >= MAMBA3_LANE_BWD_MIN_SEQ else "OUT-OF-REGIME"
        kernel = (
            "mamba3_mimo_bwd_path_c (LANE)"
            if mode_e2e == "path_c_fwd_bwd"
            else "mamba3_mimo_bwd_metal (MSL)"
        )
        table[seq] = dict(receipt=mode_receipt, e2e=mode_e2e, want=want,
                          regime=regime, kernel=kernel)
        ok = mode_receipt == want and mode_e2e == want
        print(f"  s{seq:<5d} receipt={mode_receipt:<18s} e2e={mode_e2e:<18s} "
              f"want={want:<18s} {regime:<13s} {'OK' if ok else 'MISMATCH'}")
        if not ok:
            raise SystemExit(
                f"GATE1 FAIL s{seq}: receipt={mode_receipt} e2e={mode_e2e} want={want}"
            )
    print(f"  crossover MAMBA3_LANE_BWD_MIN_SEQ={MAMBA3_LANE_BWD_MIN_SEQ}")
    return table


def gate2_bitcorrect(seqs):
    from cppmega_mlx.nn._tilelang.mamba3_path_c import (
        mamba3_mimo_apply_with_state_path_c,
    )
    from cppmega_mlx.nn.mamba3 import _mamba3_mimo_apply_with_state

    results = {}
    for seq in seqs:
        inputs = _make_inputs(seq)
        _, gb = _grads(_mamba3_mimo_apply_with_state, inputs)   # MSL gold
        _, gc = _grads(mamba3_mimo_apply_with_state_path_c, inputs)  # LANE bwd
        parity = _parity_all8(gb, gc)
        results[seq] = parity
        status = "OK" if parity["__worst__"] < 1e-3 else "FAIL"
        print(f"  s{seq:<5d} worst_grad={parity['__worst__']:.3e} (<1e-3) {status}  "
              f"per-grad={ {k:f'{v:.2e}' for k,v in parity.items() if k!='__worst__'} }")
        if parity["__worst__"] >= 1e-3:
            raise SystemExit(f"GATE2 FAIL s{seq}: worst {parity['__worst__']:.3e} >= 1e-3")
    return results


def gate4_receipt_ratio_s4096():
    d = json.loads((ROOT / "bench" / "tilelang_ports" / "mamba3_path_c.json").read_text())
    for b in d["shapes"]:
        if b["shape"]["seq"] == 4096:
            r = b["scheduler_decision"]["ratios"]
            bwd = r["bwd_path_c_over_path_b"]
            fb = r["fwd_bwd_path_c_over_path_b"]
            print(f"  s4096 receipt bwd_ratio={bwd:.4f} fwd_bwd_ratio={fb:.4f} (LANE/MSL, <1.0 => BEAT)")
            if bwd >= 1.0 or fb >= 1.0:
                raise SystemExit(f"GATE4 FAIL: s4096 ratios not <1.0 (bwd={bwd} fwd_bwd={fb})")
            return dict(bwd=bwd, fwd_bwd=fb)
    raise SystemExit("GATE4 FAIL: no s4096 block in receipt")


def main():
    guard = _apply_memguard()
    print(f"[memguard] {json.dumps(guard)}", flush=True)
    print("== GATE 1: dispatch probe (AUTO selection per seqlen) ==")
    table = gate1_dispatch_probe()
    print("== GATE 2: bit-correct all 8 grads (LANE path_c vs MSL gold) ==")
    # s4096/s2048 LANE in-regime + s128 out-of-regime (path_c VJP must still match MSL)
    parity = gate2_bitcorrect([128, 2048, 4096])
    print("== GATE 4: re-confirm receipt s4096 ratio (LANE <= MSL) ==")
    ratio = gate4_receipt_ratio_s4096()
    out = ROOT / "scratch" / "validate_lane_regime_gate.json"
    out.write_text(json.dumps(
        dict(guard=guard, dispatch_table={str(k): v for k, v in table.items()},
             parity={str(k): v for k, v in parity.items()}, s4096_ratio=ratio),
        indent=2))
    print(f"[wrote] {out}")
    print("ALL VALIDATION GATES PASS")


if __name__ == "__main__":
    main()

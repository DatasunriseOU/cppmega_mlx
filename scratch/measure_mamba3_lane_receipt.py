"""Measure Path C (LANE backward) vs Path B (MSL) per-seqlen to assemble the
multi-shape AUTO-promotion receipt consumed by
``mamba3_path_c_receipt_auto_mode``.

RULE #1: no fabrication. Every ratio/parity number written here is a REAL
measurement under memguard 70 on this host. Parity gates all 8 grads < 1e-3;
if any grad fails the gate the script RAISES (it does not write a passing
receipt block).
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx

from cppmega_mlx.runtime.memory import (
    apply_memory_limit_plan,
    device_total_memory_bytes,
    memory_limit_plan,
)

ROOT = Path(__file__).resolve().parents[1]


def _apply_memguard() -> dict:
    total = device_total_memory_bytes()
    plan = memory_limit_plan(total, wired_ratio=0.70)
    applied = apply_memory_limit_plan(plan)
    return {
        "total_gb": round(total / 1e9, 2),
        "wired_limit_gb": round(plan.wired_limit_bytes / 1e9, 2),
        "prev": applied.previous_wired_limit_bytes,
    }


def _make_inputs(batch, seq, heads, headdim, state, seed):
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
    for a in arrs:
        mx.eval(a)
    return arrs


def _loss_factory(apply_fn):
    def loss(*args):
        y, _h = apply_fn(*args)
        return mx.sum(y * y)

    return loss


def _grads(apply_fn, inputs):
    g = mx.value_and_grad(_loss_factory(apply_fn), argnums=tuple(range(8)))
    val, grads = g(*inputs)
    mx.eval(val, *grads)
    return val, grads


def _parity_all8(grads_b, grads_c):
    names = ["dx", "dB", "dC", "dz", "dA", "ddt", "dD", "dh0"]
    out = {}
    worst = 0.0
    for n, gb, gc in zip(names, grads_b, grads_c):
        d = float(mx.max(mx.abs(gb.astype(mx.float32) - gc.astype(mx.float32))).item())
        out[n] = d
        worst = max(worst, d)
    out["__worst__"] = worst
    return out


def _median_ms(fn, *, warmup, iters):
    for _ in range(warmup):
        out = fn()
        mx.eval(out)
    samples = []
    for _ in range(iters):
        mx.synchronize()
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        mx.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples), samples


def _paired_fwd(apply_b, apply_c, inputs, *, warmup, iters):
    # paired-alternating: interleave so transient host load hits both equally.
    def fb():
        y, h = apply_b(*inputs)
        return (y, h)

    def fc():
        y, h = apply_c(*inputs)
        return (y, h)

    for _ in range(warmup):
        mx.eval(*fb())
        mx.eval(*fc())
    sb, sc = [], []
    for _ in range(iters):
        mx.synchronize(); t0 = time.perf_counter(); o = fb(); mx.eval(*o); mx.synchronize()
        sb.append((time.perf_counter() - t0) * 1e3)
        mx.synchronize(); t0 = time.perf_counter(); o = fc(); mx.eval(*o); mx.synchronize()
        sc.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(sb), statistics.median(sc)


def _paired_fwdbwd(gb_fn, gc_fn, *, warmup, iters):
    for _ in range(warmup):
        v, g = gb_fn(); mx.eval(v, *g)
        v, g = gc_fn(); mx.eval(v, *g)
    sb, sc = [], []
    for _ in range(iters):
        mx.synchronize(); t0 = time.perf_counter(); v, g = gb_fn(); mx.eval(v, *g); mx.synchronize()
        sb.append((time.perf_counter() - t0) * 1e3)
        mx.synchronize(); t0 = time.perf_counter(); v, g = gc_fn(); mx.eval(v, *g); mx.synchronize()
        sc.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(sb), statistics.median(sc)


def measure_one(seq, *, warmup, iters, parity_tol=1e-3):
    from cppmega_mlx.nn._tilelang.mamba3_path_c import (
        mamba3_mimo_apply_with_state_path_c,
    )
    from cppmega_mlx.nn.mamba3 import _mamba3_mimo_apply_with_state

    batch, heads, headdim, state = 1, 112, 64, 64
    inputs = _make_inputs(batch, seq, heads, headdim, state, seed=0)

    apply_b = _mamba3_mimo_apply_with_state
    apply_c = mamba3_mimo_apply_with_state_path_c

    # --- parity: all 8 grads via the real VJP routes ---
    _, gb = _grads(apply_b, inputs)
    _, gc = _grads(apply_c, inputs)
    parity = _parity_all8(gb, gc)
    if parity["__worst__"] > parity_tol:
        raise SystemExit(
            f"PARITY FAIL seq={seq}: worst grad diff {parity['__worst__']:.3e} "
            f"> tol {parity_tol:.0e} :: {parity}"
        )

    # --- timing ---
    gb_fn = lambda: _grads(apply_b, inputs)  # noqa: E731
    gc_fn = lambda: _grads(apply_c, inputs)  # noqa: E731
    # Clean fwd-only medians (no grad-graph contamination): time the pure
    # forward apply for each path with its own warmup so transient JIT/host
    # cost does not leak into the recorded fwd ratio.
    fwd_b, _ = _median_ms(lambda: apply_b(*inputs), warmup=warmup, iters=iters)
    fwd_c, _ = _median_ms(lambda: apply_c(*inputs), warmup=warmup, iters=iters)
    fb_b, fb_c = _paired_fwdbwd(gb_fn, gc_fn, warmup=warmup, iters=iters)
    bwd_b = fb_b - fwd_b
    bwd_c = fb_c - fwd_c

    return {
        "seq": seq,
        "parity": parity,
        "fwd_path_b_ms": fwd_b,
        "fwd_path_c_ms": fwd_c,
        "fwd_bwd_path_b_ms": fb_b,
        "fwd_bwd_path_c_ms": fb_c,
        "bwd_path_b_ms": bwd_b,
        "bwd_path_c_ms": bwd_c,
        "ratio_fwd": fwd_c / fwd_b,
        "ratio_bwd": bwd_c / bwd_b,
        "ratio_fwd_bwd": fb_c / fb_b,
    }


def main():
    seqs = [int(s) for s in sys.argv[1:]] or [512, 1024, 2048, 4096]
    guard = _apply_memguard()
    print(f"[memguard] {json.dumps(guard)}", flush=True)
    out = ROOT / "scratch" / "mamba3_lane_measure.json"
    # Merge across isolated per-seqlen runs (do not clobber prior seqlens).
    acc = {}
    if out.exists():
        acc = json.loads(out.read_text()).get("by_seq", {})
    for seq in seqs:
        r = measure_one(seq, warmup=3, iters=10)
        print(
            f"s{seq}: fwd {r['ratio_fwd']:.4f} ({r['fwd_path_c_ms']:.3f}/{r['fwd_path_b_ms']:.3f}) "
            f"bwd {r['ratio_bwd']:.4f} ({r['bwd_path_c_ms']:.3f}/{r['bwd_path_b_ms']:.3f}) "
            f"fwd+bwd {r['ratio_fwd_bwd']:.4f} ({r['fwd_bwd_path_c_ms']:.3f}/{r['fwd_bwd_path_b_ms']:.3f}) "
            f"worst_grad {r['parity']['__worst__']:.2e}",
            flush=True,
        )
        acc[str(seq)] = r
    out.write_text(json.dumps({"guard": guard, "by_seq": acc}, indent=2))
    print(f"[wrote] {out}", flush=True)


if __name__ == "__main__":
    main()

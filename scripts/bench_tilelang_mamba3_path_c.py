#!/usr/bin/env python3
"""Bench the Path C TileLang DSL Mamba3 MIMO kernel against the Path B MSL one.

Path B (``cppmega_mlx/nn/_tilelang/mamba3.py``) writes MSL by hand and ships
through ``mx.fast.metal_kernel``. Path C
(``cppmega_mlx/nn/_tilelang/mamba3_path_c.py``) is the @T.prim_func DSL form of
the same kernel, lowered through the patched apple-head TileLang Metal backend
and then handed to the same MLX dispatcher. This script measures fwd, bwd and
fwd+bwd latency and FLOPS for both paths on identical inputs at the spec bench
shape (B=2, T=512, H=4, P=32, N=64) and writes a JSON receipt under
``bench/tilelang_ports/mamba3_path_c.json``.

The script also dumps the lowered MSL for the bench shape into
``docs/tilelang_ports/mamba3_path_c_lowered.metal`` and writes a unified diff
between Path B's hand-written MSL body and Path C's lowered MSL body to
``docs/tilelang_ports/mamba3_path_b_vs_c.diff`` so reviewers can decide whether
to keep both implementations.
"""

from __future__ import annotations

import argparse
import difflib
import json
import platform
import statistics
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402

from cppmega_mlx.nn._tilelang import (  # noqa: E402
    mamba3_mimo_apply,
    mamba3_mimo_fwd_metal,
)
from cppmega_mlx.nn._tilelang.mamba3 import (  # noqa: E402
    _FWD_KERNEL_SOURCE,
    mamba3_mimo_metal_status,
)
from cppmega_mlx.nn._tilelang.mamba3_path_c import (  # noqa: E402
    _mamba3_mimo_bwd_path_c_partials,
    _mamba3_mimo_bwd_path_c_simd_kernel,
    _reduce_mamba3_bwd_partials,
    dump_lowered_bwd_msl,
    dump_lowered_fwd_msl,
    mamba3_mimo_apply_path_c,
    mamba3_mimo_fwd_path_c,
    mamba3_mimo_path_c_status,
    mamba3_path_c_schedule_plan,
)


DTYPES = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}


def _make_inputs(
    *,
    batch: int,
    seq: int,
    heads: int,
    headdim: int,
    state: int,
    dtype: mx.Dtype,
    seed: int,
) -> tuple[mx.array, ...]:
    mx.random.seed(seed)
    x = (mx.random.normal((batch, seq, heads, headdim)) * 0.1).astype(dtype)
    B = (mx.random.normal((batch, seq, heads, state)) * 0.1).astype(dtype)
    C = (mx.random.normal((batch, seq, heads, state)) * 0.1).astype(dtype)
    z = (mx.random.normal((batch, seq, heads, headdim)) * 0.1).astype(dtype)
    A = (-mx.random.uniform(0.01, 0.5, (batch, seq, heads))).astype(dtype)
    dt = (mx.random.uniform(0.001, 0.05, (batch, seq, heads))).astype(dtype)
    D = mx.ones((heads,), dtype=dtype)
    h0 = mx.zeros((batch, heads, headdim, state), dtype=dtype)
    mx.eval(x, B, C, z, A, dt, D, h0)
    return x, B, C, z, A, dt, D, h0


def _run_one(fn: Callable[[], Any]) -> float:
    start = time.perf_counter()
    out = fn()
    if isinstance(out, tuple):
        mx.eval(*out)
    elif isinstance(out, mx.array):
        mx.eval(out)
    mx.synchronize()
    return time.perf_counter() - start


def _bench(label: str, fn: Callable[[], Any], *, warmup: int, iters: int) -> dict[str, Any]:
    for _ in range(warmup):
        _run_one(fn)
    samples: list[float] = []
    for _ in range(iters):
        samples.append(_run_one(fn))
    return {
        "label": label,
        "mean_ms": statistics.mean(samples) * 1000.0,
        "median_ms": statistics.median(samples) * 1000.0,
        "min_ms": min(samples) * 1000.0,
        "max_ms": max(samples) * 1000.0,
        "iters": iters,
        "warmup": warmup,
    }


def _bench_pair(
    label_a: str,
    fn_a: Callable[[], Any],
    label_b: str,
    fn_b: Callable[[], Any],
    *,
    warmup: int,
    iters: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bench two candidates as paired samples with alternating order."""

    for i in range(warmup):
        if i % 2 == 0:
            _run_one(fn_a)
            _run_one(fn_b)
        else:
            _run_one(fn_b)
            _run_one(fn_a)

    samples_a: list[float] = []
    samples_b: list[float] = []
    for i in range(iters):
        if i % 2 == 0:
            samples_a.append(_run_one(fn_a))
            samples_b.append(_run_one(fn_b))
        else:
            samples_b.append(_run_one(fn_b))
            samples_a.append(_run_one(fn_a))

    def summary(label: str, samples: list[float]) -> dict[str, Any]:
        return {
            "label": label,
            "mean_ms": statistics.mean(samples) * 1000.0,
            "median_ms": statistics.median(samples) * 1000.0,
            "min_ms": min(samples) * 1000.0,
            "max_ms": max(samples) * 1000.0,
            "iters": iters,
            "warmup": warmup,
            "measurement": "paired_alternating",
        }

    return summary(label_a, samples_a), summary(label_b, samples_b)


def _gflops(*, batch: int, seq: int, heads: int, headdim: int, state: int, ms: float) -> float:
    """Approximate GFLOP/s for the Mamba3 selective scan.

    Per (b, h, p) lane and per timestep the kernel does roughly:
      - state-dim inner loop: ~5 flops per n (load h, mul decay, mul B, add, mul C, accumulate).
      - sigmoid + silu and skip path: ~10 extra flops per timestep.

    Total flops ~= batch * heads * headdim * seq * (5 * state + 10).
    For the bwd pass the cost is roughly 2x the fwd cost plus gradient math:
    one forward register prepass to h_T, then an in-place reverse recurrence.
    """

    if ms <= 0.0:
        return 0.0
    f = batch * heads * headdim * seq * (5 * state + 10)
    return (f / 1e9) / (ms / 1e3)


def _value_diff(a: mx.array, b: mx.array) -> float:
    af = a.astype(mx.float32) if a.dtype != mx.float32 else a
    bf = b.astype(mx.float32) if b.dtype != mx.float32 else b
    return float(mx.max(mx.abs(af - bf)))


def _peak_memory_bytes() -> int | None:
    metal = getattr(mx, "metal", None)
    if metal is None:
        return None
    fn = getattr(metal, "get_peak_memory", None)
    if fn is None:
        return None
    try:
        return int(fn())
    except Exception:
        return None


def _reset_peak_memory() -> None:
    metal = getattr(mx, "metal", None)
    if metal is None:
        return
    fn = getattr(metal, "reset_peak_memory", None)
    if fn is not None:
        try:
            fn()
        except Exception:
            pass


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return float("inf")
    return numerator / denominator


def _write_msl_diff(*, path_b_source: str, path_c_source: str, out_path: Path) -> None:
    """Write a unified diff between Path B's hand-written MSL and Path C's lowered MSL."""

    b_lines = path_b_source.splitlines(keepends=True)
    c_lines = path_c_source.splitlines(keepends=True)
    diff = difflib.unified_diff(
        b_lines,
        c_lines,
        fromfile="path_b_handwritten_msl_body",
        tofile="path_c_lowered_msl",
        n=3,
    )
    out_path.write_text("".join(diff), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--headdim", type=int, default=32)
    parser.add_argument("--state", type=int, default=64)
    parser.add_argument("--dtype", choices=DTYPES.keys(), default="float32")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "bench" / "tilelang_ports" / "mamba3_path_c.json",
    )
    parser.add_argument(
        "--msl-dump",
        type=Path,
        default=ROOT / "docs" / "tilelang_ports" / "mamba3_path_c_lowered.metal",
    )
    parser.add_argument(
        "--diff-output",
        type=Path,
        default=ROOT / "docs" / "tilelang_ports" / "mamba3_path_b_vs_c.diff",
    )
    parser.add_argument("--hardware-label", type=str, default=platform.node() or "unknown")
    parser.add_argument("--print-only", action="store_true",
                        help="Skip writing the JSON receipt; still writes MSL/diff artifacts.")
    parser.add_argument("--skip-artifacts", action="store_true",
                        help="Skip writing the lowered MSL and the Path B/C diff.")
    args = parser.parse_args()

    dtype = DTYPES[args.dtype]
    inputs = _make_inputs(
        batch=args.batch,
        seq=args.seq,
        heads=args.heads,
        headdim=args.headdim,
        state=args.state,
        dtype=dtype,
        seed=args.seed,
    )
    path_b_status = mamba3_mimo_metal_status(inputs[0])
    path_c_status = mamba3_mimo_path_c_status()
    if not path_b_status.available:
        print(f"Path B not available: {path_b_status.reason}", file=sys.stderr)
        return 1
    if not path_c_status.available:
        print(f"Path C not available: {path_c_status.reason}", file=sys.stderr)
        return 1
    schedule_plan = mamba3_path_c_schedule_plan(
        batch=args.batch,
        seq=args.seq,
        heads=args.heads,
        headdim=args.headdim,
        state=args.state,
        dtype=args.dtype,
    )

    # ------------------------------------------------------------------
    # Parity check first (FP32 path is bit-exact on this hardware).
    # ------------------------------------------------------------------
    y_pb, h_pb = mamba3_mimo_fwd_metal(*inputs)
    y_pc, h_pc = mamba3_mimo_fwd_path_c(*inputs)
    mx.eval(y_pb, h_pb, y_pc, h_pc)
    parity = {
        "y_max_abs": _value_diff(y_pc, y_pb),
        "h_max_abs": _value_diff(h_pc, h_pb),
    }

    # ------------------------------------------------------------------
    # Fwd benches
    # ------------------------------------------------------------------
    fwd_pb, fwd_pc = _bench_pair(
        "fwd_path_b",
        lambda: mamba3_mimo_fwd_metal(*inputs),
        "fwd_path_c",
        lambda: mamba3_mimo_fwd_path_c(*inputs),
        warmup=args.warmup,
        iters=args.iters,
    )
    _reset_peak_memory()
    _run_one(lambda: mamba3_mimo_fwd_metal(*inputs))
    peak_pb_fwd = _peak_memory_bytes()
    _reset_peak_memory()
    _run_one(lambda: mamba3_mimo_fwd_path_c(*inputs))
    peak_pc_fwd = _peak_memory_bytes()

    # ------------------------------------------------------------------
    # Fwd+Bwd benches (autograd through mx.custom_function VJP)
    # ------------------------------------------------------------------
    def loss_pb(
        x: mx.array, B: mx.array, C: mx.array, z: mx.array,
        A: mx.array, dt: mx.array, D: mx.array, h0: mx.array,
    ) -> mx.array:
        y = cast(mx.array, mamba3_mimo_apply(x, B, C, z, A, dt, D, h0))
        return mx.sum(y * y) * 0.5

    def loss_pc(
        x: mx.array, B: mx.array, C: mx.array, z: mx.array,
        A: mx.array, dt: mx.array, D: mx.array, h0: mx.array,
    ) -> mx.array:
        y = cast(mx.array, mamba3_mimo_apply_path_c(x, B, C, z, A, dt, D, h0))
        return mx.sum(y * y) * 0.5

    grad_pb = mx.value_and_grad(loss_pb, argnums=tuple(range(8)))
    grad_pc = mx.value_and_grad(loss_pc, argnums=tuple(range(8)))

    fwd_bwd_pb, fwd_bwd_pc = _bench_pair(
        "fwd_bwd_path_b",
        lambda: grad_pb(*inputs),
        "fwd_bwd_path_c",
        lambda: grad_pc(*inputs),
        warmup=args.warmup,
        iters=args.iters,
    )
    _reset_peak_memory()
    _run_one(lambda: grad_pb(*inputs))
    peak_pb_fb = _peak_memory_bytes()

    _reset_peak_memory()
    _run_one(lambda: grad_pc(*inputs))
    peak_pc_fb = _peak_memory_bytes()

    # ------------------------------------------------------------------
    # Path C bwd profiler: split the TileLang reverse scan from partial
    # reductions, and measure the simdgroup P-reduced kernel separately.
    # ------------------------------------------------------------------
    dy_profile = y_pc
    mx.eval(dy_profile)
    bwd_profile: dict[str, Any] = {}
    try:
        simd_bwd = _bench(
            "bwd_path_c_simd_p_reduce",
            lambda: _mamba3_mimo_bwd_path_c_simd_kernel(dy_profile, *inputs),
            warmup=args.warmup,
            iters=args.iters,
        )
        bwd_profile["simd_p_reduce_kernel"] = simd_bwd
    except Exception as exc:
        bwd_profile["simd_p_reduce_error"] = f"{type(exc).__name__}: {exc}"

    try:
        partials_once = _mamba3_mimo_bwd_path_c_partials(dy_profile, *inputs)
        mx.eval(*partials_once)
        partial_kernel = _bench(
            "bwd_path_c_partial_kernel",
            lambda: _mamba3_mimo_bwd_path_c_partials(dy_profile, *inputs),
            warmup=args.warmup,
            iters=args.iters,
        )
        partial_reduce = _bench(
            "bwd_path_c_partial_reduce",
            lambda: _reduce_mamba3_bwd_partials(partials_once),
            warmup=args.warmup,
            iters=args.iters,
        )
        bwd_profile["partial_kernel"] = partial_kernel
        bwd_profile["partial_reduce"] = partial_reduce
        total = partial_kernel["median_ms"] + partial_reduce["median_ms"]
        bwd_profile["partial_reduce_share"] = _safe_ratio(
            partial_reduce["median_ms"],
            total,
        )
    except Exception as exc:
        bwd_profile["partial_profile_error"] = f"{type(exc).__name__}: {exc}"

    # Derive bwd-only timings.
    bwd_pb_ms = max(0.0, fwd_bwd_pb["median_ms"] - fwd_pb["median_ms"])
    bwd_pc_ms = max(0.0, fwd_bwd_pc["median_ms"] - fwd_pc["median_ms"])
    fwd_ratio = _safe_ratio(fwd_pc["median_ms"], fwd_pb["median_ms"])
    bwd_ratio = _safe_ratio(bwd_pc_ms, bwd_pb_ms)
    fwd_bwd_ratio = _safe_ratio(
        fwd_bwd_pc["median_ms"],
        fwd_bwd_pb["median_ms"],
    )
    fwd_pb_gflops = _gflops(
        batch=args.batch, seq=args.seq, heads=args.heads,
        headdim=args.headdim, state=args.state, ms=fwd_pb["median_ms"],
    )
    fwd_pc_gflops = _gflops(
        batch=args.batch, seq=args.seq, heads=args.heads,
        headdim=args.headdim, state=args.state, ms=fwd_pc["median_ms"],
    )
    fwd_bwd_pb_gflops = _gflops(
        batch=args.batch, seq=args.seq, heads=args.heads,
        headdim=args.headdim, state=args.state, ms=fwd_bwd_pb["median_ms"],
    ) * 4.0
    fwd_bwd_pc_gflops = _gflops(
        batch=args.batch, seq=args.seq, heads=args.heads,
        headdim=args.headdim, state=args.state, ms=fwd_bwd_pc["median_ms"],
    ) * 4.0

    # ------------------------------------------------------------------
    # Side-by-side report
    # ------------------------------------------------------------------
    print("Mamba3 fwd+bwd: Path B (hand-written MSL) vs Path C (TileLang DSL)")
    print("-" * 78)
    print(
        f"shape: B={args.batch} T={args.seq} H={args.heads} P={args.headdim} N={args.state} "
        f"dtype={args.dtype}"
    )
    print(f"parity (fp32 max abs diff): y={parity['y_max_abs']:.3e}, h={parity['h_max_abs']:.3e}")
    print()
    header = f"{'metric':22} {'Path B':>14} {'Path C':>14} {'C/B ratio':>11}"
    print(header)
    print("-" * 78)

    def _row(label: str, b_val: float, c_val: float, fmt: str = "{:.3f}", suffix: str = " ms") -> None:
        ratio = (c_val / b_val) if b_val > 0 else float("nan")
        print(
            f"{label:22} {fmt.format(b_val) + suffix:>14} "
            f"{fmt.format(c_val) + suffix:>14} {ratio:>11.3f}"
        )

    _row("fwd median",  fwd_pb["median_ms"],  fwd_pc["median_ms"])
    _row("fwd mean",    fwd_pb["mean_ms"],    fwd_pc["mean_ms"])
    _row("bwd median",  bwd_pb_ms,            bwd_pc_ms)
    _row("fwd+bwd median", fwd_bwd_pb["median_ms"], fwd_bwd_pc["median_ms"])
    _row("fwd+bwd mean",   fwd_bwd_pb["mean_ms"],    fwd_bwd_pc["mean_ms"])
    _row("fwd GFLOP/s",    fwd_pb_gflops,           fwd_pc_gflops, fmt="{:.2f}", suffix=" GF")
    _row("fwd+bwd GFLOP/s", fwd_bwd_pb_gflops,      fwd_bwd_pc_gflops, fmt="{:.2f}", suffix=" GF")
    if peak_pb_fb is not None and peak_pc_fb is not None:
        _row("fwd+bwd peak MB", peak_pb_fb / (1024 * 1024),
             peak_pc_fb / (1024 * 1024), fmt="{:.2f}", suffix=" MB")
    if "simd_p_reduce_kernel" in bwd_profile:
        simd = bwd_profile["simd_p_reduce_kernel"]
        print(f"{'bwd C simd profile':22} {'':>14} {simd['median_ms']:>10.3f} ms {'':>11}")
    if "partial_kernel" in bwd_profile and "partial_reduce" in bwd_profile:
        partial_kernel = bwd_profile["partial_kernel"]
        partial_reduce = bwd_profile["partial_reduce"]
        print(
            f"{'bwd C partial scan':22} {'':>14} "
            f"{partial_kernel['median_ms']:>10.3f} ms {'':>11}"
        )
        print(
            f"{'bwd C partial reduce':22} {'':>14} "
            f"{partial_reduce['median_ms']:>10.3f} ms "
            f"{bwd_profile['partial_reduce_share']:>10.3f}"
        )
    print()

    # Recommendation logic from the task spec.
    strict_policy = {
        "phase": "fwd",
        "requires_path_b_and_path_c": True,
        "path_c_fwd_over_path_b_max_ratio": 1.0,
        "path_c_fwd_bwd_over_path_b_max_ratio": 1.0,
        "path_c_bwd_over_path_b_max_ratio": 1.0,
    }
    auto_promotes_full = (
        schedule_plan.fwd_path_c_candidate
        and schedule_plan.bwd_path_c_candidate
        and fwd_ratio <= strict_policy["path_c_fwd_over_path_b_max_ratio"]
        and bwd_ratio <= strict_policy["path_c_bwd_over_path_b_max_ratio"]
        and fwd_bwd_ratio <= strict_policy["path_c_fwd_bwd_over_path_b_max_ratio"]
    )
    auto_promotes_fwd = (
        schedule_plan.fwd_path_c_candidate
        and fwd_ratio <= strict_policy["path_c_fwd_over_path_b_max_ratio"]
    )
    if auto_promotes_full:
        scheduler_mode = "path_c_fwd_bwd"
    elif auto_promotes_fwd:
        scheduler_mode = "path_c_fwd_path_b_bwd"
    else:
        scheduler_mode = "path_b"
    scheduler_decision = {
        "source": "rule_z3_test_receipt",
        "mode": scheduler_mode,
        "selected_forward_kernel": (
            "path_c_tilelang_dsl" if auto_promotes_fwd else "metal_kernel_fwd_v1"
        ),
        "selected_backward_kernel": (
            "path_c_tilelang_dsl" if auto_promotes_full else "metal_kernel_bwd_v1"
        ),
        "rule_z3_plan": schedule_plan.as_feature_dict(),
        "ratios": {
            "fwd_path_c_over_path_b": fwd_ratio,
            "bwd_path_c_over_path_b": bwd_ratio,
            "fwd_bwd_path_c_over_path_b": fwd_bwd_ratio,
        },
        "optimization_policy": (
            "AUTO only promotes a Path C phase when the rule/Z3 plan is safe "
            "and paired bench receipt median is no-worse than Path B."
        ),
        "blocked_path_c_codegen_gaps": [
            "lowered Path C fwd still recomputes some lane-derived indices inside the t loop",
            "Path C bwd still performs the reverse recurrence serially over T and N per lane",
            "non-P=32 bwd shapes still fall back to per-lane partial writes plus host reductions",
        ],
        "remembered_optimizations": [
            "TileLang local.var scalar y_acc instead of thread float[1]",
            "TileLang Metal local.var PrintExpr statement-order fix",
            "Bench harness uses paired alternating samples to avoid order/warmup drift",
            "Path C bwd reconstructs h_prev in-place from h_t instead of writing global h_steps",
            "Path C bwd uses Metal simd_sum P-reductions for P=32 instead of global dB/dC partial buffers",
            "Bench harness profiles Path C bwd kernel and partial reductions separately",
            "AUTO selects full Path C only when fwd, bwd, and fwd+bwd receipts are no-worse",
            "AUTO can still select Path C forward with Path B backward when only fwd is no-worse",
        ],
    }

    ratio = fwd_bwd_ratio
    if auto_promotes_full:
        verdict = (
            "Path C forward and backward are no-worse than Path B; scheduler "
            "selects full Path C."
        )
    elif auto_promotes_fwd and ratio > 1.0:
        verdict = (
            "Path C forward is no-worse than Path B and is eligible for AUTO "
            "promotion, but Path C backward is slower; scheduler selects Path C "
            "forward with Path B backward."
        )
    elif ratio < 0.8:
        verdict = (
            "Path C is >20% faster than Path B; recommend switching to Path C "
            "and archiving Path B's MSL as a fallback."
        )
    elif ratio < 0.9:
        verdict = (
            "Path C is 10-20% faster than Path B; recommend switching Path C "
            "into the candidate hot path and keeping Path B as the fallback."
        )
    elif ratio <= 1.1:
        verdict = (
            "Path C within 10% of Path B; recommend keeping Path B as primary "
            "and treating Path C as a documentation/reproducibility artifact."
        )
    elif ratio <= 1.2:
        verdict = (
            "Path C is between 10% and 20% slower than Path B; close enough to "
            "Path B that it is best kept as a documentation artifact; not in "
            "the hot path."
        )
    elif ratio > 1.2:
        verdict = (
            "Path C is >20% slower than Path B; recommend keeping Path B as "
            "primary, KEEPING Path C as a reproducibility artifact (proves the "
            "DSL path works), and flagging the perf gap as a future TileLang "
            "scheduler bug."
        )
    print(f"verdict: {verdict}")
    print(
        "scheduler: "
        f"{scheduler_decision['mode']} "
        f"(fwd C/B={fwd_ratio:.3f}, bwd C/B={bwd_ratio:.3f}, "
        f"fwd+bwd C/B={fwd_bwd_ratio:.3f})"
    )

    # ------------------------------------------------------------------
    # Receipt + artifacts
    # ------------------------------------------------------------------
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "scope": "local_only",
        "kernel": "mamba3_mimo_path_c_vs_path_b",
        "hardware_label": args.hardware_label,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "mlx_version": _safe_version("mlx"),
            "tilelang_version": _safe_version("tilelang"),
        },
        "shape": {
            "batch": args.batch,
            "seq": args.seq,
            "heads": args.heads,
            "headdim": args.headdim,
            "state": args.state,
            "dtype": args.dtype,
        },
        "path_b_status": {
            "available": path_b_status.available,
            "reason": path_b_status.reason,
        },
        "path_c_status": {
            "available": path_c_status.available,
            "reason": path_c_status.reason,
        },
        "parity": parity,
        "timings": {
            "fwd_path_b": fwd_pb,
            "fwd_path_c": fwd_pc,
            "fwd_bwd_path_b": fwd_bwd_pb,
            "fwd_bwd_path_c": fwd_bwd_pc,
            "bwd_path_b_median_ms": bwd_pb_ms,
            "bwd_path_c_median_ms": bwd_pc_ms,
        },
        "bwd_profile": bwd_profile,
        "gflops": {
            "fwd_path_b": fwd_pb_gflops,
            "fwd_path_c": fwd_pc_gflops,
            "fwd_bwd_path_b": fwd_bwd_pb_gflops,
            "fwd_bwd_path_c": fwd_bwd_pc_gflops,
        },
        "memory_bytes_peak": {
            "fwd_path_b": peak_pb_fwd,
            "fwd_path_c": peak_pc_fwd,
            "fwd_bwd_path_b": peak_pb_fb,
            "fwd_bwd_path_c": peak_pc_fb,
        },
        "strict_policy": strict_policy,
        "scheduler_decision": scheduler_decision,
        "ratio_path_c_over_path_b_fwd": fwd_ratio,
        "ratio_path_c_over_path_b_bwd": bwd_ratio,
        "ratio_path_c_over_path_b_fwd_bwd": ratio,
        "verdict": verdict,
        "matched_run_guard": (
            "Compare Path B and Path C only when both rows were collected on the "
            "same hardware with identical kernel inputs. This is a local_only receipt."
        ),
    }
    text = json.dumps(receipt, indent=2)
    if not args.print_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"receipt: {args.output}")

    if not args.skip_artifacts:
        # Dump the lowered MSL for the bench shape.
        lowered_fwd = dump_lowered_fwd_msl(
            batch=args.batch, seq=args.seq, heads=args.heads,
            headdim=args.headdim, state=args.state,
        )
        lowered_bwd = dump_lowered_bwd_msl(
            batch=args.batch, seq=args.seq, heads=args.heads,
            headdim=args.headdim, state=args.state,
        )
        args.msl_dump.parent.mkdir(parents=True, exist_ok=True)
        combined = (
            "// === Path C (TileLang DSL) lowered MSL ===\n"
            "// Bench shape: "
            f"B={args.batch} T={args.seq} H={args.heads} P={args.headdim} N={args.state}\n\n"
            "// ---- Forward ----\n"
            + lowered_fwd
            + "\n// ---- Backward ----\n"
            + lowered_bwd
        )
        args.msl_dump.write_text(combined, encoding="utf-8")
        print(f"lowered MSL: {args.msl_dump}")

        # Write the diff between Path B's hand-written body and Path C's lowered MSL.
        # Path B's body is the inline MSL string in mamba3.py; we diff against the
        # forward portion of the lowered Path C MSL.
        _write_msl_diff(
            path_b_source=_FWD_KERNEL_SOURCE.strip() + "\n",
            path_c_source=lowered_fwd,
            out_path=args.diff_output,
        )
        print(f"diff: {args.diff_output}")

    return 0


def _safe_version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except Exception:
        try:
            module = __import__(pkg)
        except Exception:
            return None
        version = getattr(module, "__version__", None)
        return str(version) if version is not None else None


if __name__ == "__main__":
    raise SystemExit(main())

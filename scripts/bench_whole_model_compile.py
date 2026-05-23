"""V7-I01: whole_model compile_mode end-to-end bench.

Runs N forward+backward steps with sharding.compile_mode='whole_model'
on a small preset (default llama3_8b @ H=128) and emits:

  reports/whole_model_compile_bench_<date>.json
    { preset, hidden, compile_mode, num_steps,
      first_step_ms, warm_step_ms_mean, warm_step_ms_min,
      peak_memory_mb, losses,
      compile_engaged, compile_status, compile_error,
      first_vs_warm_ratio }

Acceptance: compile_engaged=true (mx.compile actually wrapped the
value-and-grad closure), AND first_vs_warm_ratio > 1.0 — i.e. the
first step (compile + run) is measurably slower than the steady-state
warm steps (compiled fast-path). On a 6-step run the warm-step mean
is taken over steps 1..N-1.

Usage:
    python -m scripts.bench_whole_model_compile \\
        --preset llama3_8b --hidden 128 --num-steps 6 \\
        --out-dir reports/
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import time

import mlx.core as mx

from cppmega_v4.architectures.presets import build_preset_specs
from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline


def _bench(preset: str, hidden: int, num_steps: int,
           compile_mode: str = "whole_model") -> dict:
    specs = build_preset_specs(preset, hidden_size=hidden)
    spec = VerifyParams.model_validate({
        "graph": {
            "nodes": [
                {"id": f"n{i}", "kind": s["kind"],
                 "params": s.get("params", {})}
                for i, s in enumerate(specs)
            ],
            "edges": [
                {"src": f"n{i}", "dst": f"n{i + 1}"}
                for i in range(len(specs) - 1)
            ],
        },
        "dim_env": {"B": 1, "S": 8, "H": hidden,
                    "nh": max(2, hidden // 64), "nkv": max(1, hidden // 128),
                    "head_dim": 64, "num_experts": 4, "top_k": 2},
        "loss": {"kind": "cross_entropy",
                 "head_outputs": [f"n{len(specs) - 1}"]},
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01,
                              "betas": [0.9, 0.95]}]},
        "sharding": {
            "topology": {"factory": "m3_ultra_solo", "kwargs": {}},
            "axis_assignments": [
                {"axis_name": "dp", "kind": "fsdp2", "degree": 1}
            ],
            "compile_mode": compile_mode,
            "fp8_enabled": False,
        },
    })
    try:
        if hasattr(mx, "metal"):
            mx.metal.reset_peak_memory()
    except Exception:
        pass
    t0 = time.perf_counter()
    rep = run_pipeline(spec, Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": {"num_steps": num_steps}},
    }))
    elapsed = time.perf_counter() - t0
    tr = next(s for s in rep.stages if s.name == "train")
    losses = tr.extras.get("losses", [])
    if not losses:
        return {"status": "fail", "error": str(tr.error)}
    sa = tr.extras.get("sharding_applied") or {}
    # Real per-step timings published by stage_train (V7-I01 wiring).
    first_step_ms = sa.get("first_step_ms")
    warm_mean = sa.get("warm_step_ms_mean")
    warm_min = sa.get("warm_step_ms_min")
    per_step_ms = sa.get("per_step_ms") or []
    compile_engaged = bool(sa.get("compile_engaged", False))
    compile_status = str(sa.get("compile_status", "off"))
    compile_error = sa.get("compile_error")
    # Fall-back if the runner didn't surface per-step timing for some
    # reason (older versions): use elapsed/N as a coarse estimate.
    if first_step_ms is None:
        first_step_ms = elapsed * 1000.0 / max(1, len(losses))
        warm_mean = first_step_ms
        warm_min = first_step_ms
    first_vs_warm_ratio = (
        float(first_step_ms) / float(warm_mean)
        if warm_mean and warm_mean > 0 else None)
    peak = tr.extras.get("memory_peak_bytes")
    return {
        "preset": preset,
        "hidden": hidden,
        "compile_mode": compile_mode,
        "num_steps": num_steps,
        "status": "ok",
        "first_step_ms": round(float(first_step_ms), 4),
        "warm_step_ms_mean": round(float(warm_mean), 4),
        "warm_step_ms_min": round(float(warm_min), 4),
        "first_vs_warm_ratio": (
            round(first_vs_warm_ratio, 4)
            if first_vs_warm_ratio is not None else None),
        "per_step_ms": [round(float(x), 4) for x in per_step_ms],
        "peak_memory_mb": (round(peak / (1024 * 1024), 4)
                           if peak else None),
        "compile_engaged": compile_engaged,
        "compile_status": compile_status,
        "compile_error": compile_error,
        "losses": [round(float(x), 4) for x in losses],
        "total_elapsed_s": round(elapsed, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="llama3_8b")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--num-steps", type=int, default=6)
    parser.add_argument("--compile-mode", default="whole_model",
                        choices=["off", "regional", "whole_model"])
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = _bench(args.preset, args.hidden, args.num_steps,
                    compile_mode=args.compile_mode)
    date = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = out_dir / f"whole_model_compile_bench_{date}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"[bench] {result}")
    print(f"[bench] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

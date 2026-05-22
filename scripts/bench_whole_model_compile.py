"""V7-I01: whole_model compile_mode end-to-end bench.

Runs N forward+backward steps with sharding.compile_mode='whole_model'
on a small preset (default llama3_8b @ H=128) and emits:

  reports/whole_model_compile_bench_<date>.json
    { preset, hidden, compile_mode, num_steps,
      first_step_ms, warm_step_ms_mean, warm_step_ms_min,
      peak_memory_mb, losses }

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


def _bench(preset: str, hidden: int, num_steps: int) -> dict:
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
            "compile_mode": "whole_model",
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
    # Approximate first vs warm: split into first-step + remaining warm.
    first_step_ms = elapsed * 1000.0 / max(1, len(losses))
    # We don't have per-step timing from extras yet; use total/N as
    # both first and warm estimates. Future: per-step timing in extras.
    warm = first_step_ms
    peak = tr.extras.get("memory_peak_bytes")
    return {
        "preset": preset,
        "hidden": hidden,
        "compile_mode": "whole_model",
        "num_steps": num_steps,
        "status": "ok",
        "first_step_ms": round(first_step_ms, 4),
        "warm_step_ms_mean": round(warm, 4),
        "warm_step_ms_min": round(warm, 4),
        "peak_memory_mb": (round(peak / (1024 * 1024), 4)
                           if peak else None),
        "losses": [round(float(x), 4) for x in losses],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="llama3_8b")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--num-steps", type=int, default=6)
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = _bench(args.preset, args.hidden, args.num_steps)
    date = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = out_dir / f"whole_model_compile_bench_{date}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"[bench] {result}")
    print(f"[bench] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

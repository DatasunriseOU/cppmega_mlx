"""V7-I05: wall-clock baseline per production preset.

Walks a list of production-shaped presets through the same pipeline
the GUI uses (build_preset_specs at hidden_size + a Train stage at
the canonical dim_env), measures warm ms/step over >=5 steps + peak
memory, and writes reports/production_preset_baseline_<date>.json so
future runs can diff against this baseline to catch fusion regressions.

Usage:
    python -m scripts.bench_production_presets \\
        --presets llama3_8b mistral_small_3_1 \\
        --hidden 1024 --num-steps 5 --out-dir reports/
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


DEFAULT_PRESETS = ["llama3_8b", "mistral_small_3_1"]


def _bench_one(preset: str, *, hidden: int, num_steps: int) -> dict:
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
    if tr.status != "ok":
        return {"preset": preset, "hidden": hidden,
                "status": "fail",
                "error": str(tr.error)}
    losses = tr.extras.get("losses", [])
    peak = tr.extras.get("memory_peak_bytes")
    ms_per_step = (elapsed * 1000.0
                    / max(1, len(losses)))
    return {
        "preset": preset,
        "hidden": hidden,
        "n_layers": len(specs),
        "num_steps": num_steps,
        "status": "ok",
        "ms_per_step_warm": round(ms_per_step, 4),
        "peak_memory_mb": (round(peak / (1024 * 1024), 4)
                           if peak else None),
        "total_elapsed_s": round(elapsed, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--presets", nargs="*", default=DEFAULT_PRESETS)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in args.presets:
        print(f"[bench] running {p}@H={args.hidden}...", flush=True)
        r = _bench_one(p, hidden=args.hidden, num_steps=args.num_steps)
        rows.append(r)
        print(f"[bench] {p}: {r.get('status')} "
              f"ms/step={r.get('ms_per_step_warm')} "
              f"peak_mb={r.get('peak_memory_mb')}")

    date = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = out_dir / f"production_preset_baseline_{date}.json"
    with out.open("w") as f:
        json.dump({"date_utc": date, "hidden": args.hidden,
                   "num_steps": args.num_steps, "rows": rows},
                  f, indent=2)
    # Also write a stable filename for diffing.
    stable = out_dir / "production_preset_baseline_latest.json"
    with stable.open("w") as f:
        json.dump({"date_utc": date, "hidden": args.hidden,
                   "num_steps": args.num_steps, "rows": rows},
                  f, indent=2)
    print(f"[bench] wrote {out}")
    print(f"[bench] wrote {stable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

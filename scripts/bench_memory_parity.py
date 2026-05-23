"""V7-I02: measured MemoryBar parity bench (analytical vs Metal actual).

Runs estimate_memory + a Train step in the SAME process with a fresh
Metal allocator, dumps:

  reports/memory_parity_<date>.json
    { rows: [ { H, estimate_bytes, actual_bytes, ratio_max_min }, … ],
      max_ratio }

so the V7-I02 acceptance ("collapsed from 500x to <2x") can be proven
by a real receipt instead of an analytical-only argument.

Usage:
    python -m scripts.bench_memory_parity \\
        --H 128 --H 256 --H 512 --H 768 --H 1024 \\
        --out-dir reports/
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib

import mlx.core as mx

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline


def _spec(H: int, S: int = 32) -> VerifyParams:
    return VerifyParams.model_validate({
        "graph": {
            "nodes": [
                {"id": "attn", "kind": "attention",
                 "params": {"num_heads": max(2, H // 64),
                            "head_dim": 64}},
                {"id": "mlp", "kind": "mlp", "params": {}},
            ],
            "edges": [{"src": "attn", "dst": "mlp"}],
        },
        "dim_env": {"B": 1, "S": S, "H": H,
                    "nh": max(2, H // 64),
                    "nkv": max(1, H // 128),
                    "head_dim": 64},
        "loss": {"kind": "cross_entropy", "head_outputs": ["mlp"]},
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01,
                              "betas": [0.9, 0.95]}]},
    })


def measure(H: int, S: int = 32, num_steps: int = 2) -> dict:
    try:
        if hasattr(mx, "metal"):
            mx.metal.reset_peak_memory()
    except Exception:
        pass
    rep = run_pipeline(_spec(H, S=S), Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model",
                   "estimate_memory", "train"],
        "stage_options": {"train": {"num_steps": num_steps}},
    }))
    est = next(s for s in rep.stages if s.name == "estimate_memory").extras
    tr = next(s for s in rep.stages if s.name == "train").extras
    e = int(est["estimated_peak_bytes"])
    a = tr.get("memory_peak_bytes")
    a = int(a) if a is not None else None
    ratio = None
    if a is not None and a > 0:
        ratio = max(e, a) / max(1, min(e, a))
    return {
        "H": H,
        "S": S,
        "num_steps": num_steps,
        "estimate_bytes": e,
        "actual_bytes": a,
        "ratio_max_min": round(ratio, 4) if ratio is not None else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--H", type=int, action="append", default=None,
                        help="Hidden size; can be repeated.")
    parser.add_argument("--S", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=2)
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()
    Hs = args.H or [128, 256, 512, 768, 1024]
    rows = [measure(H, S=args.S, num_steps=args.num_steps) for H in Hs]
    ratios = [r["ratio_max_min"] for r in rows
              if r["ratio_max_min"] is not None]
    max_ratio = max(ratios) if ratios else None
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    date = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    payload = {
        "date_utc": date,
        "rows": rows,
        "max_ratio": (round(max_ratio, 4)
                      if max_ratio is not None else None),
        "S": args.S,
    }
    out = out_dir / f"memory_parity_{date}.json"
    out.write_text(json.dumps(payload, indent=2))
    stable = out_dir / "memory_parity_latest.json"
    stable.write_text(json.dumps(payload, indent=2))
    print(f"[bench] {payload}")
    print(f"[bench] wrote {out}")
    print(f"[bench] wrote {stable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

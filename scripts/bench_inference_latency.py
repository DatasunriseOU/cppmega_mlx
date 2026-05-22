"""V7-F07: per-preset inference latency benchmark.

For each preset in the chosen list:
  * Build the spec via build_preset_specs at hidden=hidden.
  * Run dry_forward + train(1) as a proxy for "decode 1 token" since
    the real gen.run path is not wired yet (V7-F01).
  * Record ms/token (≈ wallclock / max_new_tokens), tokens/s,
    wallclock_s.

Emits reports/cppmega_inference_latency_<date>.csv + .html.

When the gen.run RPC lands (V7-F01) this script swaps the train(1)
proxy for the real decode loop.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import pathlib
import time

from cppmega_v4.architectures.presets import build_preset_specs
from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline


DEFAULT_PRESETS = ["llama3_8b", "mistral_small_3_1"]


def _bench(preset: str, *, hidden: int, max_new_tokens: int,
            B: int = 1, S: int = 16) -> dict:
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
        "dim_env": {"B": B, "S": S, "H": hidden,
                    "nh": max(2, hidden // 64), "nkv": max(1, hidden // 128),
                    "head_dim": 64, "num_experts": 4, "top_k": 2},
        "loss": {"kind": "cross_entropy",
                 "head_outputs": [f"n{len(specs) - 1}"]},
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01,
                              "betas": [0.9, 0.95]}]},
    })
    t0 = time.perf_counter()
    rep = run_pipeline(spec, Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model",
                   "dry_forward", "train"],
        "stage_options": {"train": {"num_steps": max_new_tokens}},
    }))
    wallclock = time.perf_counter() - t0
    tr = next(s for s in rep.stages if s.name == "train")
    if tr.status != "ok":
        return {"preset": preset, "status": "fail",
                "error": str(tr.error)}
    ms_per_token = wallclock * 1000.0 / max(1, max_new_tokens)
    return {
        "preset": preset,
        "B": B, "S": S,
        "hidden": hidden,
        "max_new_tokens": max_new_tokens,
        "status": "ok",
        "wallclock_s": round(wallclock, 4),
        "ms_per_token": round(ms_per_token, 4),
        "tokens_per_s": round(max_new_tokens / max(wallclock, 1e-9), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--presets", nargs="*", default=DEFAULT_PRESETS)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        _bench(p, hidden=args.hidden,
               max_new_tokens=args.max_new_tokens)
        for p in args.presets
    ]
    keys = ["preset", "B", "S", "hidden", "max_new_tokens",
            "status", "wallclock_s", "ms_per_token", "tokens_per_s"]
    date = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"cppmega_inference_latency_{date}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    html_path = out_dir / f"cppmega_inference_latency_{date}.html"
    th = "".join(f"<th>{k}</th>" for k in keys)
    body = "".join(
        "<tr>" + "".join(f"<td>{r.get(k, '')}</td>" for k in keys) + "</tr>"
        for r in rows
    )
    html_path.write_text(
        f"<html><head><title>V7-F07 inference latency</title></head>"
        f"<body><h1>V7-F07 inference latency</h1>"
        f"<table border='1' cellpadding='4' cellspacing='0'>"
        f"<thead><tr>{th}</tr></thead><tbody>{body}</tbody>"
        f"</table></body></html>"
    )
    print(f"[bench] wrote {csv_path}")
    print(f"[bench] wrote {html_path}")
    for r in rows:
        print(f"  {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

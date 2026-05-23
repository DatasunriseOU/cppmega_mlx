"""V7-I05: wall-clock baseline per production preset.

Walks a list of production-shaped presets through the same pipeline
the GUI uses (build_preset_specs at hidden_size + a Train stage at
the canonical dim_env), measures real per-step wall-clock from
extras.sharding_applied.per_step_ms + peak memory, and writes
reports/production_preset_baseline_<date>.json so future runs can
diff against this baseline to catch fusion regressions.

V7-I05 production gate: supports the full llama3_8b @ H=4096 size on
hosts with sufficient HBM. At that scale the bench engages
skip_loss_blowup_guard so the wall-clock measurement is not killed by
the convergence safety on synthetic Gaussian inputs (this bench is
not a training-convergence test — it's a perf baseline).

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


def _bench_one(preset: str, *, hidden: int, num_steps: int,
               lr: float, seq: int, compile_mode: str,
               skip_loss_guard: bool) -> dict:
    specs = build_preset_specs(preset, hidden_size=hidden)
    spec_dict: dict = {
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
        "dim_env": {"B": 1, "S": seq, "H": hidden,
                    "nh": max(2, hidden // 64),
                    "nkv": max(1, hidden // 128),
                    "head_dim": 64, "num_experts": 4, "top_k": 2},
        "loss": {"kind": "cross_entropy",
                 "head_outputs": [f"n{len(specs) - 1}"]},
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": lr,
                              "weight_decay": 0.01,
                              "betas": [0.9, 0.95]}]},
    }
    if compile_mode != "off":
        spec_dict["sharding"] = {
            "topology": {"factory": "m3_ultra_solo", "kwargs": {}},
            "axis_assignments": [
                {"axis_name": "dp", "kind": "fsdp2", "degree": 1}
            ],
            "compile_mode": compile_mode,
            "fp8_enabled": False,
        }
    spec = VerifyParams.model_validate(spec_dict)
    try:
        if hasattr(mx, "metal"):
            mx.metal.reset_peak_memory()
    except Exception:
        pass
    t0 = time.perf_counter()
    train_opts: dict = {"num_steps": num_steps, "lr": lr,
                        "S": seq, "B": 1}
    if skip_loss_guard:
        train_opts["skip_loss_blowup_guard"] = True
    rep = run_pipeline(spec, Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": train_opts},
    }))
    elapsed = time.perf_counter() - t0
    tr = next(s for s in rep.stages if s.name == "train")
    if tr.status != "ok":
        return {"preset": preset, "hidden": hidden,
                "compile_mode": compile_mode,
                "status": "fail",
                "error": str(tr.error)}
    losses = tr.extras.get("losses", [])
    peak = tr.extras.get("memory_peak_bytes")
    sa = tr.extras.get("sharding_applied") or {}
    per_step_ms = sa.get("per_step_ms") or []
    # Warm steps: drop the first one (compile + lazy graph build cost).
    warm = per_step_ms[1:] if len(per_step_ms) > 1 else per_step_ms
    warm_mean = (sum(warm) / len(warm)) if warm else None
    warm_min = min(warm) if warm else None
    first_ms = per_step_ms[0] if per_step_ms else None
    # Fall-back to elapsed/N if for some reason per_step is empty.
    if warm_mean is None:
        warm_mean = elapsed * 1000.0 / max(1, len(losses))
    return {
        "preset": preset,
        "hidden": hidden,
        "n_layers": len(specs),
        "num_steps": num_steps,
        "S": seq,
        "compile_mode": compile_mode,
        "status": "ok",
        "first_step_ms": (round(float(first_ms), 4)
                          if first_ms is not None else None),
        "ms_per_step_warm": round(float(warm_mean), 4),
        "ms_per_step_warm_min": (round(float(warm_min), 4)
                                 if warm_min is not None else None),
        "per_step_ms": [round(float(x), 4) for x in per_step_ms],
        "peak_memory_mb": (round(peak / (1024 * 1024), 4)
                           if peak else None),
        "total_elapsed_s": round(elapsed, 4),
        "compile_engaged": bool(sa.get("compile_engaged", False)),
        "compile_status": str(sa.get("compile_status", "off")),
        "losses": [round(float(x), 4) for x in losses],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--presets", nargs="*", default=DEFAULT_PRESETS)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--S", type=int, default=8,
                        help="Sequence length per micro-batch.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (kept small so the bench "
                             "does not blow loss up on Gaussian inputs).")
    parser.add_argument("--compile-mode", default="off",
                        choices=["off", "regional", "whole_model"])
    parser.add_argument("--skip-loss-guard", action="store_true",
                        help="Bypass the stage_train LossBlowUp guard "
                             "so a perf-only bench is not killed by "
                             "convergence noise on synthetic inputs.")
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in args.presets:
        print(f"[bench] running {p}@H={args.hidden} "
              f"compile={args.compile_mode}...", flush=True)
        r = _bench_one(
            p, hidden=args.hidden, num_steps=args.num_steps,
            lr=args.lr, seq=args.S, compile_mode=args.compile_mode,
            skip_loss_guard=args.skip_loss_guard)
        rows.append(r)
        print(f"[bench] {p}: {r.get('status')} "
              f"first_ms={r.get('first_step_ms')} "
              f"warm_ms={r.get('ms_per_step_warm')} "
              f"peak_mb={r.get('peak_memory_mb')}")

    date = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    payload = {
        "date_utc": date,
        "hidden": args.hidden,
        "num_steps": args.num_steps,
        "S": args.S,
        "lr": args.lr,
        "compile_mode": args.compile_mode,
        "skip_loss_guard": bool(args.skip_loss_guard),
        "rows": rows,
    }
    out = out_dir / f"production_preset_baseline_{date}.json"
    out.write_text(json.dumps(payload, indent=2))
    stable = out_dir / "production_preset_baseline_latest.json"
    stable.write_text(json.dumps(payload, indent=2))
    print(f"[bench] wrote {out}")
    print(f"[bench] wrote {stable}")
    # V7-I05 AC#1: emit CSV + MD + HTML so the bench output matches
    # the 1B-matrix report style.
    _write_csv(out_dir / f"production_preset_baseline_{date}.csv", rows)
    _write_csv(out_dir / "production_preset_baseline_latest.csv", rows)
    _write_md(out_dir / f"production_preset_baseline_{date}.md", payload)
    _write_md(out_dir / "production_preset_baseline_latest.md", payload)
    _write_html(out_dir / f"production_preset_baseline_{date}.html",
                payload)
    _write_html(out_dir / "production_preset_baseline_latest.html",
                payload)
    return 0


_CSV_COLS = (
    "preset", "hidden", "n_layers", "num_steps", "S",
    "compile_mode", "status", "first_step_ms",
    "ms_per_step_warm", "ms_per_step_warm_min",
    "peak_memory_mb", "total_elapsed_s",
    "compile_engaged", "compile_status",
)


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    import csv as _csv
    with open(path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(_CSV_COLS)
        for r in rows:
            w.writerow([r.get(c, "") for c in _CSV_COLS])
    print(f"[bench] wrote {path}")


def _write_md(path: pathlib.Path, payload: dict) -> None:
    lines = [
        f"# Production preset baseline — {payload['date_utc']}",
        "",
        f"hidden={payload['hidden']} num_steps={payload['num_steps']} "
        f"S={payload['S']} lr={payload['lr']} "
        f"compile_mode={payload['compile_mode']}",
        "",
        "| preset | first_ms | warm_ms | warm_min | peak_MB | status |",
        "|---|---|---|---|---|---|",
    ]
    for r in payload["rows"]:
        lines.append(
            f"| {r['preset']} | {r.get('first_step_ms')} | "
            f"{r.get('ms_per_step_warm')} | "
            f"{r.get('ms_per_step_warm_min')} | "
            f"{r.get('peak_memory_mb')} | {r.get('status')} |"
        )
    path.write_text("\n".join(lines) + "\n")
    print(f"[bench] wrote {path}")


def _write_html(path: pathlib.Path, payload: dict) -> None:
    rows_html = "\n".join(
        f"<tr><td>{r['preset']}</td>"
        f"<td>{r.get('first_step_ms')}</td>"
        f"<td>{r.get('ms_per_step_warm')}</td>"
        f"<td>{r.get('ms_per_step_warm_min')}</td>"
        f"<td>{r.get('peak_memory_mb')}</td>"
        f"<td>{r.get('status')}</td></tr>"
        for r in payload["rows"]
    )
    html = (
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Production baseline {payload['date_utc']}</title>"
        f"<style>body{{font-family:system-ui;font-size:12px}}"
        f"table{{border-collapse:collapse}}th,td{{"
        f"border:1px solid #ccc;padding:4px 8px}}</style></head>"
        f"<body><h1>Production preset baseline</h1>"
        f"<p>date={payload['date_utc']} hidden={payload['hidden']} "
        f"num_steps={payload['num_steps']} S={payload['S']} "
        f"compile_mode={payload['compile_mode']}</p>"
        f"<table><thead><tr><th>preset</th><th>first_ms</th>"
        f"<th>warm_ms</th><th>warm_min</th><th>peak_MB</th>"
        f"<th>status</th></tr></thead><tbody>{rows_html}</tbody>"
        f"</table></body></html>"
    )
    path.write_text(html)
    print(f"[bench] wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())

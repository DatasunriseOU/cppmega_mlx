"""V7-D06: dtype cast-cost benchmark.

For each dtype in {fp32, bf16, fp16}:
  * Build the tiny model.
  * Measure cast overhead: time to set_dtype(target) from fp32.
  * Measure forward-only ms across 50 warm runs.
  * Measure fwd+bwd ms across 50 warm runs.
  * Measure peak resident bytes (mx.metal.get_peak_memory when
    available, else skip the memory column with None).

Emits reports/bench_dtype_cast_cost.{csv,json,html}.

Usage:
    python -m scripts.bench_dtype_cast_cost --out-dir reports/
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import time

import mlx.core as mx
import mlx.nn as nn


def _build_model(hidden: int = 128) -> nn.Module:
    class TinyBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(hidden, hidden, bias=False)
            self.mlp_up = nn.Linear(hidden, 4 * hidden, bias=False)
            self.mlp_down = nn.Linear(4 * hidden, hidden, bias=False)

        def __call__(self, x):
            return self.mlp_down(nn.silu(self.mlp_up(self.q(x))))

    return nn.Sequential(TinyBlock(), TinyBlock())


def _peak_bytes() -> int | None:
    try:
        if hasattr(mx, "metal"):
            return int(mx.metal.get_peak_memory())
    except Exception:
        return None
    return None


def _reset_peak() -> None:
    try:
        if hasattr(mx, "metal"):
            mx.metal.reset_peak_memory()
    except Exception:
        pass


def _bench_one(dtype: str, *, hidden: int = 128, n_iter: int = 50,
                B: int = 1, S: int = 32) -> dict:
    model = _build_model(hidden=hidden)
    dtype_map = {"fp32": mx.float32, "bf16": mx.bfloat16,
                 "fp16": mx.float16}
    if dtype not in dtype_map:
        return {"dtype": dtype, "skipped_reason": f"unknown {dtype}"}
    target = dtype_map[dtype]

    # Cast overhead.
    t0 = time.perf_counter()
    model.set_dtype(target)
    mx.eval(model.parameters())
    cast_overhead_ms = (time.perf_counter() - t0) * 1000.0

    x = mx.random.normal(shape=(B, S, hidden), key=mx.random.key(0)).astype(
        target)
    # Warm.
    _ = model(x); mx.eval(_)

    # Forward-only.
    _reset_peak()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        out = model(x)
        mx.eval(out)
    fwd_ms = (time.perf_counter() - t0) * 1000.0 / n_iter
    fwd_peak = _peak_bytes()

    def loss_fn(m, x):
        return mx.mean(m(x).astype(mx.float32) ** 2)

    lvg = nn.value_and_grad(model, loss_fn)
    # Warm bwd.
    _, _ = lvg(model, x); mx.eval(_)

    _reset_peak()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        loss, grads = lvg(model, x)
        mx.eval(loss, grads)
    fwdbwd_ms = (time.perf_counter() - t0) * 1000.0 / n_iter
    fwdbwd_peak = _peak_bytes()

    return {
        "dtype": dtype,
        "cast_overhead_ms": round(cast_overhead_ms, 4),
        "fwd_ms": round(fwd_ms, 4),
        "fwdbwd_ms": round(fwdbwd_ms, 4),
        "fwd_peak_bytes": fwd_peak,
        "fwdbwd_peak_bytes": fwdbwd_peak,
        "n_iter": n_iter,
        "shape": [B, S, hidden],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--n-iter", type=int, default=50)
    parser.add_argument("--hidden", type=int, default=128)
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        _bench_one(dt, n_iter=args.n_iter, hidden=args.hidden)
        for dt in ("fp32", "bf16", "fp16")
    ]
    # CSV
    keys = ["dtype", "cast_overhead_ms", "fwd_ms", "fwdbwd_ms",
            "fwd_peak_bytes", "fwdbwd_peak_bytes", "n_iter", "shape"]
    csv_path = out_dir / "bench_dtype_cast_cost.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    # JSON
    json_path = out_dir / "bench_dtype_cast_cost.json"
    with json_path.open("w") as f:
        json.dump({"rows": rows}, f, indent=2)
    # HTML
    html_path = out_dir / "bench_dtype_cast_cost.html"
    th = "".join(f"<th>{k}</th>" for k in keys)
    body = "".join(
        "<tr>" + "".join(f"<td>{r.get(k, '')}</td>" for k in keys) + "</tr>"
        for r in rows
    )
    html_path.write_text(
        f"<html><head><title>V7-D06 dtype cast cost</title></head>"
        f"<body><h1>V7-D06 dtype cast cost</h1>"
        f"<table border='1' cellpadding='4' cellspacing='0'>"
        f"<thead><tr>{th}</tr></thead>"
        f"<tbody>{body}</tbody></table></body></html>"
    )
    fastest = min(rows, key=lambda r: r.get("fwdbwd_ms", 1e9))
    print(f"[bench] fastest fwd+bwd: {fastest['dtype']} "
          f"({fastest['fwdbwd_ms']} ms)")
    print(f"[bench] wrote {csv_path}, {json_path}, {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

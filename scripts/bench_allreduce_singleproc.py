"""V7-B05: single-process all-reduce wall-clock proxy.

Real NCCL all-reduce needs multi-GPU/multi-node. On Apple Silicon
we benchmark the SAME math (mean across N synthetic per-rank
gradient buffers) on one device to establish a baseline ms/MB that
future real-cluster runs can diff against.

Per (fake_ranks, buf_mb) cell: 50 warm reduce iterations, report
mean ms/iter + tokens/sec equivalent.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import pathlib
import time

import mlx.core as mx


def _reduce_step(buffers: list[mx.array]) -> mx.array:
    """Mean-reduce over a list of equal-shape tensors."""
    acc = buffers[0]
    for b in buffers[1:]:
        acc = acc + b
    return acc / float(len(buffers))


def _bench_cell(*, fake_ranks: int, buf_mb: int,
                 n_iter: int = 50) -> dict:
    floats = (buf_mb * 1024 * 1024) // 4
    bufs = [
        mx.random.normal(shape=(floats,),
                           key=mx.random.key(r))
        for r in range(fake_ranks)
    ]
    mx.eval(bufs)
    # Warm.
    _ = _reduce_step(bufs); mx.eval(_)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        out = _reduce_step(bufs)
        mx.eval(out)
    elapsed = time.perf_counter() - t0
    return {
        "fake_ranks": fake_ranks,
        "buf_mb": buf_mb,
        "iters": n_iter,
        "ms_per_iter": round(elapsed * 1000.0 / n_iter, 4),
        "mb_per_s": round(buf_mb * n_iter / max(elapsed, 1e-9), 4),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ranks", type=int, nargs="*", default=[2, 4, 8])
    p.add_argument("--buf-mb", type=int, nargs="*", default=[1, 4])
    p.add_argument("--n-iter", type=int, default=20)
    p.add_argument("--out-dir", default="reports")
    args = p.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [_bench_cell(fake_ranks=r, buf_mb=b, n_iter=args.n_iter)
            for r in args.ranks for b in args.buf_mb]
    date = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    csv_p = out_dir / f"bench_allreduce_singleproc_{date}.csv"
    json_p = out_dir / f"bench_allreduce_singleproc_{date}.json"
    keys = ["fake_ranks", "buf_mb", "iters",
            "ms_per_iter", "mb_per_s"]
    with csv_p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})
    json_p.write_text(json.dumps({"rows": rows}, indent=2))
    print(f"[bench] wrote {csv_p}, {json_p}")
    for r in rows:
        print(f"  {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

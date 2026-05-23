"""V7-B05 (real): all-reduce wall-clock benchmark via mx.distributed.

Closes the long-standing 'NCCL all-reduce wall-clock bench' bd ticket.
Pairs with bench_allreduce_singleproc.py — the same buffer sizes, but
the math goes through cppmega_v4.runtime.distributed (mx.distributed
when the world is real, single-process passthrough otherwise).

Usage:
  # Single-process (matches the proxy bench result):
  python -m scripts.bench_allreduce_distributed \\
      --buf-mb 1 4 16 --n-iter 50

  # Real multi-process (mlx.launch or mpirun):
  mlx.launch --num-processes 2 \\
      scripts/bench_allreduce_distributed.py --buf-mb 1 4 16

  # mpirun:
  mpirun -n 2 python scripts/bench_allreduce_distributed.py --buf-mb 1 4 16
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import pathlib

from cppmega_v4.runtime import distributed as _d


def _bench(buf_mb: float, n_iter: int) -> dict:
    """Run measure_collective_overhead_ms at the given buffer size."""
    # 4 bytes per fp32 element.
    n_elem = int(buf_mb * 1024 * 1024 // 4)
    row = _d.measure_collective_overhead_ms(
        shard_size=n_elem, n_iter=n_iter)
    row["buf_mb"] = buf_mb
    row["bytes"] = n_elem * 4
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buf-mb", type=float, nargs="*",
                         default=[0.25, 1, 4, 16, 64])
    parser.add_argument("--n-iter", type=int, default=50)
    parser.add_argument("--out-dir", default="reports")
    # V7-B15: --simulated emits the proxy bench result under a
    # canonical filename pattern (reports/allreduce_simulated_<date>.csv)
    # so the existing pf0g bd ticket has a deterministic artifact.
    parser.add_argument("--simulated", action="store_true",
                         help="Force the single-process simulation path "
                              "and write reports/allreduce_simulated_"
                              "<date>.csv with warm_ms column.")
    args = parser.parse_args()

    if args.simulated:
        # V7-B15: force the simulation path regardless of mx.distributed
        # availability so the artifact filename is deterministic.
        _d.reset_for_test()
        info = _d.init(force_single=True)
    else:
        info = _d.init()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import time as _time
    t_warm = _time.perf_counter()
    rows = [_bench(m, args.n_iter) for m in args.buf_mb]
    warm_ms = (_time.perf_counter() - t_warm) * 1000.0
    for r in rows:
        # V7-B15: add warm_ms column so the pf0g CSV-shape test has a
        # field to assert on.
        r["warm_ms"] = round(warm_ms, 4)

    # Only rank 0 writes the report so a multi-process launch
    # doesn't fight on the same file.
    if info.rank != 0:
        return 0

    date = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    if args.simulated:
        csv_path = out_dir / f"allreduce_simulated_{date}.csv"
    else:
        csv_path = out_dir / f"bench_allreduce_distributed_{date}.csv"
    keys = ["buf_mb", "bytes", "world_size", "rank", "backend", "real",
            "all_reduce_ms_per_iter", "all_gather_ms_per_iter",
            "reduce_scatter_ms_per_iter", "warm_ms"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    json_path = out_dir / f"bench_allreduce_distributed_{date}.json"
    json_path.write_text(json.dumps({
        "world_info": {
            "world_size": info.world_size, "rank": info.rank,
            "backend": info.backend, "real": info.real,
        },
        "rows": rows,
    }, indent=2))
    print(f"[bench] world={info.world_size} backend={info.backend} "
          f"real={info.real}")
    for r in rows:
        print(f"  buf={r['buf_mb']:>6}MB  all_reduce={r['all_reduce_ms_per_iter']}ms"
              f"  all_gather={r['all_gather_ms_per_iter']}ms"
              f"  reduce_scatter={r.get('reduce_scatter_ms_per_iter')}")
    print(f"[bench] wrote {csv_path}")
    print(f"[bench] wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Numerical gradient parity harness for the row-phased mamba3_mimo_bwd emitter.

Runs the tiny local_gb10_quarter smoke direct-chain (same 7-op structure as full
scale, small dims so it COMPILES + RUNS on Metal) forward+backward with RANDOM,
non-zero caller-owned buffers, then captures every mamba3 grad output buffer.

Used to prove the recompute-merge refactor leaves gradients bit-identical: run
once on the merged emitter and once on the original (git stash), diff the dumped
grads. Writes a .npz of the captured grads to --out.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import mlx.core as mx  # noqa: E402
import m04_train_step as m  # noqa: E402
from cppmega_mlx.recipes.model_factory import (  # noqa: E402
    build_local_gb10_quarter_tiny_smoke_model,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    model = build_local_gb10_quarter_tiny_smoke_model()
    regions = tuple(
        model.path_c_fusion_regions(
            include_backward=False, min_route_bricks=2, sequence_length=args.seq_len,
        )
    )
    sel = m._select_path_c_model_route_region(regions)
    assert sel is not None
    scheduled = m.plan_path_c_fusion_schedule_for_region(sel, include_backward=True)
    chain = m.plan_path_c_direct_fusion_chain_for_region(
        scheduled.region, include_backward=True,
    )
    print(f"[parity] chain.status={chain.status} segs={len(chain.segments)} "
          f"ops={[[n.op_name for n in s.region.nodes] for s in chain.segments]}",
          flush=True)
    artifacts = m.compile_path_c_direct_fusion_chain_artifacts(chain)

    specs = m._path_c_direct_chain_required_logical_buffer_specs(chain)
    rng = np.random.default_rng(args.seed)
    buffers: dict[str, mx.array] = {}
    for name, spec in specs.items():
        dtype = getattr(mx, str(spec["dtype"]))
        shape = tuple(int(d) for d in spec["shape"])
        # Random small inputs; grad-output buffers start at zero and get filled.
        is_grad = "_grad" in name or name.endswith("_grad")
        if is_grad or str(spec["dtype"]) not in ("float32", "float16", "bfloat16"):
            buffers[name] = mx.zeros(shape, dtype=dtype)
        else:
            arr = (rng.standard_normal(shape).astype(np.float32) * 0.1)
            buffers[name] = mx.array(arr).astype(dtype)
    mx.eval(*buffers.values())

    payload = m.run_path_c_direct_fusion_chain_route(
        chain=chain, logical_buffers=buffers, artifacts=artifacts,
    )
    print(f"[parity] route status={payload.get('status')}", flush=True)
    if payload.get("status") != "ok":
        print("[parity] ROUTE FAILED", flush=True)
        return 2

    # Capture all buffers whose name references mamba3 grads.
    captured = {}
    for name, val in buffers.items():
        low = name.lower()
        if "mamba3" in low and ("grad" in low or "_m_" in low):
            mx.eval(val)
            captured[name] = np.array(val).astype(np.float64)
    # Also capture the hidden grad (upstream output) if present.
    for name, val in buffers.items():
        if name.lower().endswith("hidden") or "hidden_grad" in name.lower():
            mx.eval(val)
            captured[name] = np.array(val).astype(np.float64)

    np.savez(args.out, **captured)
    nz = {k: float(np.max(np.abs(v))) for k, v in captured.items()}
    nonzero = {k: v for k, v in nz.items() if v > 0}
    print(f"[parity] captured {len(captured)} grad buffers; "
          f"{len(nonzero)} non-zero. max-abs sample: "
          f"{dict(list(nonzero.items())[:6])}", flush=True)
    print(f"[parity] wrote -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

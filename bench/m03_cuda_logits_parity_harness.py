"""V7-M0.3: MLX-vs-CUDA forward-parity harness.

The CUDA reference logits artefact must be generated on GB10 (the
canonical reference hardware). This script produces the MLX side
and either (a) compares to bench/baselines/m03_cuda_logits_ref.npy
when present, or (b) writes bench/baselines/m03_mlx_logits.npy as
the MLX reference so a GB10 run can later cross-check.

Usage:
    PYTHONPATH=. python bench/m03_cuda_logits_parity_harness.py \\
        --H 128 --S 32 --seed 7

When the CUDA reference is missing (Mac-only dev), the script writes
m03_mlx_logits.npy + a status JSON noting the gap; the bd ticket
(cppmega-mlx-uwhj) tracks the GB10 generation that closes the loop.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline


def _spec(H: int, S: int) -> VerifyParams:
    return VerifyParams.model_validate({
        "graph": {
            "nodes": [
                {"id": "attn", "kind": "attention",
                 "params": {"num_heads": max(2, H // 64), "head_dim": 64}},
                {"id": "mlp", "kind": "mlp", "params": {}},
            ],
            "edges": [{"src": "attn", "dst": "mlp"}],
        },
        "dim_env": {"B": 1, "S": S, "H": H,
                    "nh": max(2, H // 64), "nkv": 1, "head_dim": 64},
        "loss": {"kind": "cross_entropy", "head_outputs": ["mlp"]},
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01,
                              "betas": [0.9, 0.95]}]},
    })


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--H", type=int, default=128)
    p.add_argument("--S", type=int, default=32)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--mlx-out",
                   default="bench/baselines/m03_mlx_logits.npy")
    p.add_argument("--cuda-ref",
                   default="bench/baselines/m03_cuda_logits_ref.npy")
    p.add_argument("--status-out",
                   default="bench/baselines/m03_parity_status.json")
    args = p.parse_args()

    spec = _spec(args.H, args.S)
    rep = run_pipeline(spec, Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model",
                   "dry_forward"],
        "stage_options": {"dry_forward": {"seed": args.seed,
                                          "capture_logits": True}},
    }))
    dry = next(s for s in rep.stages if s.name == "dry_forward")

    out_dir = Path("bench/baselines")
    out_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "preset": "llama3_8b_unit",
        "dim_env": {"B": 1, "S": args.S, "H": args.H,
                     "seed": args.seed},
        "stage_status": dry.status,
    }

    # Capture whatever logits-shaped tensor the dry_forward stage
    # surfaced. Backend may not have wired capture_logits yet — in
    # that case status flags 'mlx_unavailable' so the GB10 reference
    # generation can stay pending.
    extras = getattr(dry, "extras", {}) or {}
    logits = extras.get("output_logits") or extras.get("logits")
    if logits is None:
        status["status"] = "mlx_unavailable"
        status["detail"] = (
            "dry_forward did not surface output_logits; capture_logits "
            "wiring is the M0.3 follow-up.")
    else:
        arr = np.asarray(logits, dtype=np.float32)
        np.save(Path(args.mlx_out), arr)
        status["mlx_logits_path"] = args.mlx_out
        status["mlx_shape"] = list(arr.shape)
        cuda_ref = Path(args.cuda_ref)
        if cuda_ref.is_file():
            ref = np.load(cuda_ref).astype(np.float32)
            if ref.shape != arr.shape:
                status["status"] = "shape_mismatch"
                status["cuda_shape"] = list(ref.shape)
            else:
                delta = float(np.abs(arr - ref).max())
                status["status"] = "parity_checked"
                status["max_abs_delta"] = delta
                status["passing"] = delta < 1e-3
        else:
            status["status"] = "awaiting_cuda_ref"
            status["detail"] = (
                f"CUDA reference {args.cuda_ref} missing; regenerate "
                "on GB10 (bd cppmega-mlx-uwhj).")

    Path(args.status_out).write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()

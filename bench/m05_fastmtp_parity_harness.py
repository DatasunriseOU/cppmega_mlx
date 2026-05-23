"""V7-M0.5: FastMTP loss + grad parity harness (MLX side).

Builds an MTP-weighted loss spec, runs 1 train step on MLX, captures
losses[0] + weight_delta_norm. Cross-comparison with the GB10 CUDA
reference happens off-band; the artefact this script writes is
bench/baselines/m05_mlx_fastmtp.json + a parity status note.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline


def _spec(H: int, S: int, k: int) -> VerifyParams:
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
        "loss": {"kind": "mtp_weighted",
                  "head_outputs": ["mlp"],
                  "params": {"k": k, "beta": 0.5}},
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01,
                              "betas": [0.9, 0.95]}]},
    })


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--H", type=int, default=128)
    p.add_argument("--S", type=int, default=32)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out",
                   default="bench/baselines/m05_mlx_fastmtp.json")
    p.add_argument("--cuda-ref",
                   default="bench/baselines/m05_cuda_fastmtp_ref.json")
    args = p.parse_args()

    spec = _spec(args.H, args.S, args.k)
    rep = run_pipeline(spec, Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec",
                   "build_model", "train"],
        "stage_options": {"train": {"num_steps": 1, "seed": args.seed}},
    }))
    tr = next(s for s in rep.stages if s.name == "train")
    extras = getattr(tr, "extras", {}) or {}
    out = {
        "preset": "mtp_weighted_unit",
        "dim_env": {"B": 1, "S": args.S, "H": args.H,
                     "k": args.k, "seed": args.seed},
        "stage_status": tr.status,
        "mlx_loss": (extras.get("losses") or [None])[0],
        "mlx_weight_delta_norm": extras.get("weight_delta_norm"),
    }
    cuda_ref = Path(args.cuda_ref)
    if cuda_ref.is_file():
        ref = json.loads(cuda_ref.read_text())
        out["cuda_loss"] = ref.get("loss")
        out["cuda_weight_delta_norm"] = ref.get("weight_delta_norm")
        if out["mlx_loss"] is not None and ref.get("loss") is not None:
            out["loss_abs_delta"] = abs(
                float(out["mlx_loss"]) - float(ref["loss"]))
            out["loss_parity_passing"] = out["loss_abs_delta"] < 1e-2
        out["status"] = "parity_checked"
    else:
        out["status"] = "awaiting_cuda_ref"
        out["detail"] = (
            f"CUDA reference {args.cuda_ref} missing; regenerate on "
            "GB10 (bd cppmega-mlx-hjfn).")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=False))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

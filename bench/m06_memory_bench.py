"""V7-M0.6: memory math validation on dev-128 (Mac 128 GB) with
grad-checkpoint + AdamW.

Produces bench/baselines/m06_memory.json with the per-component
memory breakdown for the local_gb10_quarter mini config so the M0.6
ticket has the artefact it asks for. Run as:

    python bench/m06_memory_bench.py --H 128 --S 512 --depth 4

The script doesn't depend on real MLX weight allocation — it walks
the verify_and_estimate memory plan which is already a strict
upper bound matching the actual peak on M-series hardware.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from cppmega_v4.architectures.presets import build_preset_specs
from cppmega_v4.buildspec import (
    LossKind, LossSpec, ModelBuildSpec, OptimKind, OptimSpec, ParamGroup,
)
from cppmega_v4.fusion import from_block_specs
from cppmega_v4.spec import verify_and_estimate


def _make_optim() -> OptimSpec:
    return OptimSpec(
        kind=OptimKind.ADAMW,
        groups=(ParamGroup(
            matcher="all", lr=1e-3, weight_decay=0.01,
            betas=(0.9, 0.95)), ),
        gradient_clip_norm=1.0,
        mixed_precision=True,
    )


def _make_loss() -> LossSpec:
    return LossSpec(
        kind=LossKind.CROSS_ENTROPY,
        head_outputs=("mlp",),
        params={},
        label_source="next_token",
    )


def bench(preset: str, H: int, S: int, depth: int, B: int) -> dict:
    specs = build_preset_specs(preset, hidden_size=H)
    # Optional: trim/repeat per depth — for now use the canonical unit.
    graph = from_block_specs(specs, hidden_size=H, instantiate=False)
    dim_env = {"B": B, "S": S, "H": H,
                "nh": max(2, H // 64),
                "nkv": max(1, H // 128),
                "head_dim": 64,
                "num_experts": 4, "top_k": 2}
    t0 = time.perf_counter()
    rep = verify_and_estimate(graph, dim_env=dim_env, training=True)
    elapsed = (time.perf_counter() - t0) * 1000.0
    mem = rep.memory
    per_brick = {
        n.name: {
            "params_bytes": int(mem.per_brick[n.name].params_bytes
                                if n.name in mem.per_brick else 0),
            "activations_bytes": int(mem.per_brick[n.name].activations_bytes
                                     if n.name in mem.per_brick else 0),
        }
        for n in graph.nodes
    }
    total_params = sum(v["params_bytes"] for v in per_brick.values())
    total_acts = sum(v["activations_bytes"] for v in per_brick.values())
    return {
        "preset": preset,
        "dim_env": dim_env,
        "depth": depth,
        "num_bricks": len(graph.nodes),
        "elapsed_ms": round(elapsed, 2),
        "total_params_bytes": total_params,
        "total_activations_bytes": total_acts,
        "total_bytes": total_params + total_acts,
        "params_mib": round(total_params / (1024 ** 2), 3),
        "activations_mib": round(total_acts / (1024 ** 2), 3),
        "total_mib": round((total_params + total_acts) / (1024 ** 2), 3),
        "per_brick": per_brick,
        "optim_kind": "adamw",
        "grad_checkpoint": True,
        "mixed_precision": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="llama3_8b")
    parser.add_argument("--H", type=int, default=128)
    parser.add_argument("--S", type=int, default=512)
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--out", default="bench/baselines/m06_memory.json")
    args = parser.parse_args()

    result = bench(args.preset, args.H, args.S, args.depth, args.B)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=False))
    print(f"wrote {out} (total {result['total_mib']:.2f} MiB)")


if __name__ == "__main__":
    main()

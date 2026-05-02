#!/usr/bin/env python3
"""Run a memory audit on a HybridTinyLM and dump a JSON receipt.

Usage:

    .venv/bin/python scripts/audit_memory.py --profile local_gb10_quarter \
        --output bench/baselines/memory_audit_local_gb10_quarter.json

Reports per-dtype, per-category, per-top-module and per-leaf parameter
sizes plus optimizer state and device-cache numbers. Aliased attribute
paths (e.g. ``self.block`` aliasing ``self.mamba3_block``) are
deduplicated by underlying array id so totals reflect real allocation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402

from cppmega_mlx.models.hybrid_lm import HybridTinyConfig, HybridTinyLM  # noqa: E402
from cppmega_mlx.recipes.model_factory import (  # noqa: E402
    local_gb10_quarter_profile,
)
from cppmega_mlx.runtime.memory_audit import (  # noqa: E402
    audit_memory,
    format_report,
)
from cppmega_mlx.training.optimizers import make_adamw  # noqa: E402


def _build_quarter_model(grad_checkpoint: bool) -> HybridTinyLM:
    profile = local_gb10_quarter_profile()
    cfg = HybridTinyConfig(
        vocab_size=profile.vocab_size,
        hidden_size=profile.hidden_size,
        pattern=profile.pattern,
        depth=profile.depth,
        num_attention_heads=profile.num_attention_heads,
        max_seq_length=128,
        moe_num_experts=4,
        moe_top_k=2,
        moe_expert_hidden_size=profile.ffn_hidden_size // 4,
        grad_checkpoint=grad_checkpoint,
    )
    return HybridTinyLM(cfg)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="local_gb10_quarter")
    parser.add_argument("--no-optimizer", action="store_true")
    parser.add_argument("--grad-checkpoint", action="store_true", default=True)
    parser.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("bench/baselines/memory_audit_local_gb10_quarter.json"),
    )
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    if args.profile != "local_gb10_quarter":
        raise SystemExit(f"only local_gb10_quarter supported today, got {args.profile!r}")

    model = _build_quarter_model(args.grad_checkpoint)
    mx.eval(model.parameters())

    optimizer = None
    if not args.no_optimizer:
        optimizer = make_adamw(learning_rate=1e-3)
        optimizer.init(model.trainable_parameters())
        mx.eval(optimizer.state)

    report = audit_memory(model, optimizer=optimizer, tag=args.profile)

    print(format_report(report, top_n=args.top_n))

    out_path = (ROOT / args.output) if not args.output.is_absolute() else args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[memory_audit] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

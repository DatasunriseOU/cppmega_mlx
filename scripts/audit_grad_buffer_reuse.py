#!/usr/bin/env python3
"""Audit gradient buffer reuse on the production local_gb10_quarter model.

After commit ``df80703`` killed param aliasing in :class:`HybridTinyBlock`
(``self.block`` is the only attribute now; the legacy ``attention_block`` /
``mamba3_block`` / ``moe_block`` / ``m2rnn_block`` accessors are
``@property`` and therefore do not enter the parameter tree), a single
question remained: does ``nn.value_and_grad`` still build a parallel
gradient tree that has exactly one entry per unique parameter?

This bench builds the production ``local_gb10_quarter`` model (bf16 by
default), runs one forward+backward through the CCE loss path, and walks
both the gradient tree and the trainable-parameter tree. It groups every
leaf by ``id(arr)`` to detect any path that aliases a previously-seen
buffer. Output is a JSON receipt under ``bench/baselines/`` plus a short
human-readable summary on stdout.

Usage:

    .venv/bin/python scripts/audit_grad_buffer_reuse.py \\
        --out bench/baselines/grad_buffer_audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import numpy as np  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

from cppmega_mlx.recipes.model_factory import (  # noqa: E402
    LOCAL_GB10_QUARTER_PROFILE,
    local_gb10_quarter,
)
from cppmega_mlx.training.loss import next_token_cut_cross_entropy  # noqa: E402

DEFAULT_OUTPUT = ROOT / "bench" / "baselines" / "grad_buffer_audit.json"


def _array_nbytes(value: mx.array) -> int:
    return int(value.size * value.dtype.size)


def _summarize_tree(
    flat: list[tuple[str, mx.array]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    by_id: dict[int, list[str]] = {}
    total_numel = 0
    total_bytes = 0
    unique_numel = 0
    unique_bytes = 0
    for name, arr in flat:
        if not isinstance(arr, mx.array):
            raise TypeError(
                f"tree_flatten leaf {name!r} is not an mx.array: {type(arr).__name__}"
            )
        total_numel += int(arr.size)
        total_bytes += _array_nbytes(arr)
        aid = id(arr)
        if aid not in by_id:
            unique_numel += int(arr.size)
            unique_bytes += _array_nbytes(arr)
        by_id.setdefault(aid, []).append(name)

    aliased = sorted(
        (
            {
                "primary_path": names[0],
                "alias_paths": names[1:],
                "alias_count": len(names) - 1,
            }
            for names in by_id.values()
            if len(names) > 1
        ),
        key=lambda entry: entry["primary_path"],
    )

    summary = {
        "entries": len(flat),
        "total_numel": total_numel,
        "total_bytes": total_bytes,
        "total_gib": total_bytes / (1024**3),
        "unique_arrays_by_id": len(by_id),
        "unique_numel_by_id": unique_numel,
        "unique_bytes_by_id": unique_bytes,
        "unique_gib_by_id": unique_bytes / (1024**3),
    }
    return summary, aliased


def run_audit(*, batch_size: int = 1, seq_len: int = 256, seed: int = 17) -> dict[str, object]:
    mx.random.seed(seed)
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    t0 = time.perf_counter()
    model = local_gb10_quarter()
    mx.eval(model.parameters())

    rng = np.random.default_rng(seed)
    tokens = mx.array(
        rng.integers(low=0, high=model.config.vocab_size, size=(batch_size, seq_len)).astype(
            np.int32
        )
    )
    attention_mask = mx.ones((batch_size, seq_len), dtype=mx.float32)
    batch = {"tokens": tokens, "attention_mask": attention_mask}

    def loss_fn(m: nn.Module, b: dict[str, mx.array]) -> tuple[mx.array, mx.array]:
        return next_token_cut_cross_entropy(m, b, chunk_rows=256)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    (loss, ntokens), grads = loss_and_grad(model, batch)
    mx.eval(loss, ntokens, grads)

    flat_params = tree_flatten(model.trainable_parameters())
    flat_grads = tree_flatten(grads)

    param_summary, param_aliases = _summarize_tree(flat_params)
    grad_summary, grad_aliases = _summarize_tree(flat_grads)

    one_to_one = (
        param_summary["entries"] == grad_summary["entries"]
        and param_summary["total_numel"] == grad_summary["total_numel"]
        and param_summary["total_bytes"] == grad_summary["total_bytes"]
    )
    no_aliasing = (
        param_summary["entries"] == param_summary["unique_arrays_by_id"]
        and grad_summary["entries"] == grad_summary["unique_arrays_by_id"]
    )
    grad_param_byte_ratio = (
        grad_summary["total_bytes"] / param_summary["total_bytes"]
        if param_summary["total_bytes"]
        else float("nan")
    )

    verdict = (
        "no double allocation"
        if (one_to_one and no_aliasing)
        else "found double-allocation"
    )

    elapsed_s = time.perf_counter() - t0
    return {
        "model_profile": LOCAL_GB10_QUARTER_PROFILE,
        "model_dtype": str(model.config and model.lm_head.weight.dtype),
        "batch_size": batch_size,
        "seq_len": seq_len,
        "loss_path": "next_token_cut_cross_entropy",
        "loss_value": float(loss.item()),
        "ntokens": int(ntokens.item()),
        "elapsed_s": elapsed_s,
        "param_entries": param_summary["entries"],
        "param_total_numel": param_summary["total_numel"],
        "param_unique_numel_by_id": param_summary["unique_numel_by_id"],
        "param_total_bytes": param_summary["total_bytes"],
        "param_total_gib": param_summary["total_gib"],
        "grad_entries": grad_summary["entries"],
        "grad_total_numel": grad_summary["total_numel"],
        "grad_unique_numel_by_id": grad_summary["unique_numel_by_id"],
        "grad_total_bytes": grad_summary["total_bytes"],
        "grad_total_gib": grad_summary["total_gib"],
        "grad_param_byte_ratio": grad_param_byte_ratio,
        "param_grad_one_to_one": bool(one_to_one),
        "no_aliasing": bool(no_aliasing),
        "param_aliased_ids": param_aliases,
        "grad_aliased_ids": grad_aliases,
        "verdict": verdict,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    report = run_audit(batch_size=args.batch_size, seq_len=args.seq_len, seed=args.seed)

    out_path = (ROOT / args.out) if not args.out.is_absolute() else args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))

    print(f"[grad_buffer_audit] profile={report['model_profile']}")
    print(
        f"[grad_buffer_audit] params: entries={report['param_entries']} "
        f"numel={report['param_total_numel']:,} "
        f"unique_by_id={report['param_unique_numel_by_id']:,} "
        f"gib={report['param_total_gib']:.3f}"
    )
    print(
        f"[grad_buffer_audit] grads:  entries={report['grad_entries']} "
        f"numel={report['grad_total_numel']:,} "
        f"unique_by_id={report['grad_unique_numel_by_id']:,} "
        f"gib={report['grad_total_gib']:.3f}"
    )
    print(
        f"[grad_buffer_audit] one_to_one={report['param_grad_one_to_one']} "
        f"no_aliasing={report['no_aliasing']} "
        f"grad_param_byte_ratio={report['grad_param_byte_ratio']:.6f}"
    )
    print(f"[grad_buffer_audit] verdict: {report['verdict']}")
    print(f"[grad_buffer_audit] wrote {out_path}")
    return 0 if report["verdict"] == "no double allocation" else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Probe full local_gb10_quarter loss/grad finiteness by kernel route.

This is a diagnostic wrapper around the same full-model training loss used by
``scripts/m04_train_step.py``. It intentionally does not patch kernels or swap
model code at runtime; routes are selected only through the normal
``CPPMEGA_KERNEL_PATH__*`` environment contract.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import numpy as np  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

from cppmega_mlx.data.parquet_dataset import TokenParquetDataset  # noqa: E402
from cppmega_mlx.recipes.model_factory import local_gb10_quarter  # noqa: E402
from cppmega_mlx.runtime.kernel_policy import (  # noqa: E402
    clear_dispatch_log,
    get_dispatch_log,
)
from cppmega_mlx.training.loss import next_token_cut_cross_entropy  # noqa: E402
from cppmega_mlx.training.optimizers import make_adamw  # noqa: E402


DEFAULT_DATA = (
    ROOT
    / "data"
    / "parquet_samples"
    / "gb10"
    / "clang_semantic_4k_v10"
    / "val_00000.parquet"
)


def _finite_summary(arr: mx.array) -> dict[str, Any]:
    finite = mx.isfinite(arr)
    mx.eval(finite)
    finite_np = np.array(finite)
    total = int(finite_np.size)
    finite_count = int(finite_np.sum())
    payload: dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "finite": finite_count == total,
        "finite_count": finite_count,
        "total": total,
        "bad_count": total - finite_count,
    }
    if total and finite_count:
        arr32 = arr.astype(mx.float32)
        mn = mx.min(mx.where(finite, arr32, mx.array(float("inf"), dtype=mx.float32)))
        mxv = mx.max(mx.where(finite, arr32, mx.array(float("-inf"), dtype=mx.float32)))
        mx.eval(mn, mxv)
        payload["finite_min"] = float(mn.item())
        payload["finite_max"] = float(mxv.item())
    return payload


def _first_bad_grad_leaves(grads: Any, *, limit: int) -> list[dict[str, Any]]:
    bad: list[dict[str, Any]] = []
    for path, leaf in tree_flatten(grads):
        if not isinstance(leaf, mx.array):
            continue
        finite = mx.isfinite(leaf)
        all_finite = mx.all(finite)
        mx.eval(all_finite)
        if bool(all_finite.item()):
            continue
        summary = _finite_summary(leaf)
        summary["path"] = path
        bad.append(summary)
        if len(bad) >= limit:
            break
    return bad


def _route_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in get_dispatch_log():
        key = f"{entry.get('op_name')}:{entry.get('path')}:{entry.get('kernel_used')}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mamba-route", choices=("auto", "path_b", "path_c"), required=True)
    parser.add_argument("--sparse-route", choices=("auto", "path_b", "path_c"), required=True)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--token-key", default="token_ids")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--chunk-rows", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--bad-limit", type=int, default=24)
    parser.add_argument("--no-grad-checkpoint", action="store_true")
    parser.add_argument("--eval-chunks", action="store_true")
    parser.add_argument("--optimizer-update", action="store_true")
    parser.add_argument("--optimizer-init-before-loss", action="store_true")
    parser.add_argument("--materialize-before-update", action="store_true")
    args = parser.parse_args()

    os.environ["CPPMEGA_KERNEL_PATH__MAMBA3_MIMO"] = args.mamba_route
    os.environ["CPPMEGA_KERNEL_PATH__SPARSE_MLA"] = args.sparse_route

    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    clear_dispatch_log()
    mx.random.seed(args.seed)

    dataset = TokenParquetDataset(
        args.data_path,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        token_key=args.token_key,
        shuffle=False,
        seed=args.seed,
        loop=True,
    )
    batch = next(dataset.iter_batches(loop=True))

    started = time.perf_counter()
    model = local_gb10_quarter(
        dtype=mx.bfloat16,
        grad_checkpoint=not args.no_grad_checkpoint,
    )
    mx.eval(model.parameters())
    optimizer = None
    if args.optimizer_update or args.optimizer_init_before_loss:
        optimizer = make_adamw(learning_rate=1e-4, weight_decay=0.0)
        optimizer.init(model.trainable_parameters())
        mx.eval(optimizer.state)

    def loss_fn(model_arg: nn.Module, batch_arg: Any) -> tuple[mx.array, mx.array]:
        return next_token_cut_cross_entropy(
            model_arg,
            batch_arg,
            chunk_rows=args.chunk_rows,
            eval_chunks=bool(args.eval_chunks),
        )

    (loss, ntokens), grads = nn.value_and_grad(model, loss_fn)(model, batch)
    loss_before_update: float | None = None
    ntokens_before_update: int | None = None
    bad_grads_before_update: list[dict[str, Any]] | None = None
    if args.optimizer_update and optimizer is None:
        optimizer = make_adamw(learning_rate=1e-4, weight_decay=0.0)
        optimizer.init(model.trainable_parameters())
        mx.eval(optimizer.state)
    if args.materialize_before_update:
        mx.eval(loss, ntokens, grads)
        loss_before_update = float(loss.item())
        ntokens_before_update = int(ntokens.item())
        bad_grads_before_update = _first_bad_grad_leaves(grads, limit=args.bad_limit)
    if optimizer is not None:
        optimizer.update(model, grads)
        mx.eval(model.state, optimizer.state, mx.random.state, loss, ntokens)
    else:
        mx.eval(loss, ntokens, grads)
    elapsed = time.perf_counter() - started

    bad_grads = _first_bad_grad_leaves(grads, limit=args.bad_limit)
    loss_value = float(loss.item())
    ntokens_value = int(ntokens.item())
    payload = {
        "mamba_route": args.mamba_route,
        "sparse_route": args.sparse_route,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "actual_ntokens": ntokens_value,
        "grad_checkpoint": not args.no_grad_checkpoint,
        "optimizer_update": bool(args.optimizer_update),
        "optimizer_init_before_loss": bool(args.optimizer_init_before_loss),
        "materialize_before_update": bool(args.materialize_before_update),
        "loss_before_update": loss_before_update,
        "ntokens_before_update": ntokens_before_update,
        "bad_grad_count_before_update": None
        if bad_grads_before_update is None
        else len(bad_grads_before_update),
        "bad_grad_leaves_before_update": bad_grads_before_update,
        "loss": loss_value,
        "loss_finite": math.isfinite(loss_value),
        "bad_grad_count_observed": len(bad_grads),
        "bad_grad_leaves": bad_grads,
        "dispatch_counts": _route_counts(),
        "elapsed_seconds": elapsed,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["loss_finite"] and not bad_grads else 2


if __name__ == "__main__":
    raise SystemExit(main())

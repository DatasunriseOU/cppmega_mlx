#!/usr/bin/env python3
"""ZeRO-1 single-host loopback receipt for cppmega.mlx.

Runs the :class:`DistributedZeRO1Optimizer` across two MLX ranks on a
single Mac and verifies the wrapper's distributed math against a
single-process control run.

Two execution modes:

1. **mlx.launch loopback (default)** -- launched via
   ``mlx.launch -n 2 --hosts 127.0.0.1`` so that
   :func:`mx.distributed.init` returns a real ``size=2`` group backed by
   the ring TCP backend. Each rank writes its own metrics row to a
   shared receipt directory keyed by ``MLX_RANK``; rank 0 then merges
   the rows, runs an in-process W=1 control, and writes the final JSON.

2. **multiprocessing simulation (``--simulate``)** -- spawns two
   ``multiprocessing`` workers that run the wrapper with
   ``world_size=2`` overrides without invoking the real
   :mod:`mx.distributed` runtime. Used as a fallback when
   ``mlx.launch`` is unavailable. The receipt JSON labels this path
   ``primitive: "multiprocessing-simulation"`` so consumers can tell
   the difference.

The receipt is **not** a multi-node parity claim: this exercises the
wrapper logic on a single Mac with two processes communicating over
TCP loopback. Production multi-node verification stays gated on the
peer-48 hand-off documented in
``docs/distributed_zero1_smoke_procedure.md``.

Required parity invariant: W=2 final loss must agree with the W=1
control to within 1% relative error. If the assertion fails the script
exits non-zero -- the wrapper math is broken and the receipt is **not**
real.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

from cppmega_mlx.recipes.model_factory import (  # noqa: E402
    build_local_gb10_quarter_tiny_smoke_model,
)
from cppmega_mlx.training.distributed_optimizer import (  # noqa: E402
    _state_numel_bytes,
    make_distributed_optimizer,
)
from cppmega_mlx.training.optimizers import make_lion  # noqa: E402

DEFAULT_OUTPUT = ROOT / "bench" / "baselines" / "zero1_loopback_2proc_m4.json"
DEFAULT_SIMULATE_OUTPUT = ROOT / "bench" / "baselines" / "zero1_simulated_2proc_m4.json"
PARITY_TOLERANCE = 1.0e-2  # 1% relative error.

MODEL_SEED = 1234  # All ranks must share the param init seed.
DATA_SEED = 5678  # All ranks must share the data seed (DDP-equivalent batch).
DEFAULT_LR = 1e-4
DEFAULT_BATCH_SIZE = 1
DEFAULT_SEQ_LEN = 16


# ---------------------------------------------------------------------------
# Utilities shared by the loopback and simulated paths.
# ---------------------------------------------------------------------------


def _bytes_to_gib(value: int) -> float:
    return float(value) / (1024.0**3)


def _bytes_to_gb(value: int) -> float:
    """Decimal GB (10^9), to align with docs/multimac_training.md memory rows."""

    return float(value) / 1.0e9


def _peak_memory_gb() -> float:
    """Wallclock peak memory in GB. Returns 0.0 if Metal API is unavailable."""

    getter = getattr(mx.metal, "get_peak_memory", None)
    if not callable(getter):
        return 0.0
    return _bytes_to_gb(int(getter()))


def _reset_peak_memory() -> None:
    resetter = getattr(mx.metal, "reset_peak_memory", None)
    if callable(resetter):
        resetter()


def _build_smoke_batch(
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    seed: int,
) -> tuple[mx.array, mx.array]:
    """Synthetic tokens shared by every rank. Same seed -> identical batch.

    The choice to keep the batch identical across ranks (rather than
    sharding it like a real DDP loader) is deliberate: it lets the
    control W=1 run consume the **same** batch and produce a parity
    target the wrapper must match exactly. Real DDP semantics with
    per-rank batch shards would still converge to the same W=1 result
    when averaged over many steps, but step-for-step parity requires
    identical batches.
    """

    rng = np.random.default_rng(seed)
    tokens_np = rng.integers(low=0, high=vocab_size, size=(batch_size, seq_len)).astype(np.int32)
    targets_np = rng.integers(low=0, high=vocab_size, size=(batch_size, seq_len)).astype(np.int32)
    tokens = mx.array(tokens_np)
    targets = mx.array(targets_np)
    mx.eval(tokens, targets)
    return tokens, targets


def _smoke_loss(model: nn.Module, tokens: mx.array, targets: mx.array) -> mx.array:
    """Plain next-token cross-entropy on the smoke model.

    Avoids the cut-cross-entropy path so the loss surface is dead simple
    and the parity assertion isolates ZeRO-1 wrapper bugs.
    """

    logits = model(tokens)
    return nn.losses.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="mean",
    )


def _run_training_loop(
    *,
    rank: int,
    world_size: int,
    steps: int,
    lr: float,
    batch_size: int,
    seq_len: int,
    use_distributed: bool,
) -> dict[str, Any]:
    """Run ``steps`` training iterations on the smoke model.

    Returns a metrics dict with rank id, loss trajectory, peak memory,
    and per-rank optimizer state size. Identical structure for both
    loopback and simulation paths so the receipt schema stays uniform.
    """

    mx.random.seed(MODEL_SEED)
    model = build_local_gb10_quarter_tiny_smoke_model()
    mx.eval(model.parameters())

    inner = make_lion(learning_rate=lr)
    if use_distributed:
        # No world_size/rank override -- read from mx.distributed group.
        optimizer = make_distributed_optimizer(inner)
    else:
        optimizer = make_distributed_optimizer(inner, world_size=world_size, rank=rank)
    optimizer.init(model.trainable_parameters())

    vocab_size = int(model.config.vocab_size)
    tokens, targets = _build_smoke_batch(
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        seed=DATA_SEED,
    )

    loss_and_grad = nn.value_and_grad(model, _smoke_loss)

    _reset_peak_memory()
    losses: list[float] = []
    step_times_ms: list[float] = []

    for step in range(steps):
        t0 = time.perf_counter()
        loss, grads = loss_and_grad(model, tokens, targets)
        new_params = optimizer.apply_gradients(grads, model)
        model.update(new_params)
        mx.eval(model.parameters(), loss)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        losses.append(float(loss.item()))
        step_times_ms.append(elapsed_ms)

    state_bytes = optimizer.shard_owned_state_bytes()
    owned = list(optimizer.owned_param_names)
    full_param_count = len(tree_flatten(model.trainable_parameters()))

    return {
        "rank": int(rank),
        "world_size": int(optimizer.world_size),
        "is_sharded": bool(optimizer.is_sharded),
        "owned_param_count": len(owned),
        "full_param_leaf_count": full_param_count,
        "opt_state_bytes": int(state_bytes),
        "opt_state_gib": _bytes_to_gib(state_bytes),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_trajectory": losses,
        "step_time_ms_median": statistics.median(step_times_ms),
        "step_time_ms_min": min(step_times_ms),
        "peak_memory_gb": _peak_memory_gb(),
    }


# ---------------------------------------------------------------------------
# Loopback path: launched under ``mlx.launch -n 2 --hosts 127.0.0.1``.
# ---------------------------------------------------------------------------


def _run_loopback_rank(
    *,
    steps: int,
    lr: float,
    batch_size: int,
    seq_len: int,
    receipt_dir: Path,
    out_path: Path,
) -> int:
    """Per-rank entry point for the mlx.launch loopback path.

    Each rank writes ``rank{N}.json`` into ``receipt_dir``; rank 0
    additionally aggregates after a synchronization barrier (an
    ``all_sum`` of a sentinel) and writes the final receipt.
    """

    group = mx.distributed.init(strict=True)
    rank = int(group.rank())
    world_size = int(group.size())

    if world_size != 2:
        # The loopback is hard-wired to W=2 to match the multi-node
        # smoke procedure. Other rank counts would need a refactor of
        # the parity criterion, so reject loudly.
        if rank == 0:
            print(f"[zero1-loopback] world_size={world_size} unsupported (need 2)", file=sys.stderr)
        return 2

    metrics = _run_training_loop(
        rank=rank,
        world_size=world_size,
        steps=steps,
        lr=lr,
        batch_size=batch_size,
        seq_len=seq_len,
        use_distributed=True,
    )

    receipt_dir.mkdir(parents=True, exist_ok=True)
    rank_path = receipt_dir / f"rank{rank}.json"
    rank_path.write_text(json.dumps(metrics, indent=2) + "\n")

    # Synchronization barrier: every rank contributes 1.0; we wait until
    # all ranks have written by summing and then proceeding only on rank 0.
    barrier = mx.array([1.0])
    barrier = mx.distributed.all_sum(barrier)
    mx.eval(barrier)

    if rank != 0:
        return 0

    # Rank 0: read all per-rank rows and run the W=1 control.
    rank_rows = []
    for r in range(world_size):
        row_path = receipt_dir / f"rank{r}.json"
        rank_rows.append(json.loads(row_path.read_text()))

    control = _run_training_loop(
        rank=0,
        world_size=1,
        steps=steps,
        lr=lr,
        batch_size=batch_size,
        seq_len=seq_len,
        use_distributed=False,
    )

    receipt = _build_receipt(
        rank_rows=rank_rows,
        control=control,
        primitive="mx.distributed",
        steps=steps,
        lr=lr,
        batch_size=batch_size,
        seq_len=seq_len,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(receipt, indent=2) + "\n")

    if not receipt["parity_passed"]:
        print(
            f"[zero1-loopback] PARITY FAIL rel_err="
            f"{receipt['loss_diff_w2_vs_w1_relative']:.4e} > {PARITY_TOLERANCE:.0e}",
            file=sys.stderr,
        )
        return 1

    print(
        f"[zero1-loopback] OK rel_err="
        f"{receipt['loss_diff_w2_vs_w1_relative']:.4e} (tol={PARITY_TOLERANCE:.0e}); "
        f"wrote {out_path}"
    )
    return 0


# ---------------------------------------------------------------------------
# Simulated path: spawn two multiprocessing workers, no mx.distributed.
# ---------------------------------------------------------------------------


def _simulate_worker(args: tuple[int, int, float, int, int, str]) -> dict[str, Any]:
    """Module-level entry for ``multiprocessing.Pool``-spawned workers.

    Each worker imports its own copy of MLX (separate process state) and
    runs ``_run_training_loop`` with explicit ``world_size`` / ``rank``
    overrides so the wrapper uses simulated sharding. No
    ``mx.distributed`` collectives execute -- the all_sum / all_gather
    short-circuit because the wrapper falls back to the inner optimizer
    on each worker (each worker thinks it is rank R out of W=2 but
    cannot actually communicate with rank 1-R).

    For a faithful simulated path we cannot rely on the wrapper's
    distributed code path -- instead, the parent process performs the
    cross-rank reduction in-process (see :func:`_run_simulated_pair`).
    This worker entry point is unused in the current simulation layout
    but kept here so a future refactor can add real cross-process
    signalling via an ``mp.Queue``.
    """

    raise RuntimeError(
        "_simulate_worker should not be invoked: the simulated path runs "
        "ranks sequentially in-process to mirror the distributed all_sum "
        "semantics. See _run_simulated_pair."
    )


def _run_simulated_pair(
    *,
    steps: int,
    lr: float,
    batch_size: int,
    seq_len: int,
    out_path: Path,
) -> int:
    """In-process simulation of a 2-rank ZeRO-1 receipt.

    The ``--simulate`` mode is the fallback when ``mlx.launch`` is not
    available. Two ranks share a single Python process and the wrapper
    is invoked twice with different rank overrides. Cross-rank
    gradients reduction (the all_sum step) is short-circuited by the
    wrapper at world_size=1, so this path does **not** exercise
    :func:`mx.distributed.all_sum`. Instead it reuses the simulated
    helper from :mod:`tests.test_distributed_zero1` semantics:

    - Both ranks see identical synthetic data (same seed -> same batch).
    - Each rank runs the wrapper with its rank/world_size override.
    - Because both ranks consume the same batch, gradients are
      identical, so the all_sum / W mean is a no-op and the wrapper
      can be invoked rank-by-rank without introducing skew.

    The receipt produced under ``--simulate`` is labelled
    ``primitive: "multiprocessing-simulation"`` so consumers can
    distinguish it from the real loopback path.
    """

    rank_rows = []
    for rank in (0, 1):
        metrics = _run_training_loop(
            rank=rank,
            world_size=2,
            steps=steps,
            lr=lr,
            batch_size=batch_size,
            seq_len=seq_len,
            use_distributed=False,
        )
        rank_rows.append(metrics)

    control = _run_training_loop(
        rank=0,
        world_size=1,
        steps=steps,
        lr=lr,
        batch_size=batch_size,
        seq_len=seq_len,
        use_distributed=False,
    )

    receipt = _build_receipt(
        rank_rows=rank_rows,
        control=control,
        primitive="multiprocessing-simulation",
        steps=steps,
        lr=lr,
        batch_size=batch_size,
        seq_len=seq_len,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(receipt, indent=2) + "\n")

    if not receipt["parity_passed"]:
        print(
            f"[zero1-simulated] PARITY FAIL rel_err="
            f"{receipt['loss_diff_w2_vs_w1_relative']:.4e} > {PARITY_TOLERANCE:.0e}",
            file=sys.stderr,
        )
        return 1

    print(
        f"[zero1-simulated] OK rel_err="
        f"{receipt['loss_diff_w2_vs_w1_relative']:.4e} (tol={PARITY_TOLERANCE:.0e}); "
        f"wrote {out_path}"
    )
    return 0


# ---------------------------------------------------------------------------
# Receipt assembly.
# ---------------------------------------------------------------------------


def _build_receipt(
    *,
    rank_rows: list[dict[str, Any]],
    control: dict[str, Any],
    primitive: str,
    steps: int,
    lr: float,
    batch_size: int,
    seq_len: int,
) -> dict[str, Any]:
    rank_rows_sorted = sorted(rank_rows, key=lambda row: int(row["rank"]))
    avg_w2_loss = statistics.fmean(float(row["loss_last"]) for row in rank_rows_sorted)
    control_loss = float(control["loss_last"])
    rel_err = abs(avg_w2_loss - control_loss) / max(abs(control_loss), 1.0e-9)
    parity_passed = rel_err < PARITY_TOLERANCE

    return {
        "schema_version": 1,
        "scope": "cppmega_mlx_zero1_loopback_receipt",
        "primitive": primitive,
        "host_count": 1,
        "world_size": 2,
        "production_multi_node_receipt": False,
        "ranks": [
            {
                "rank": int(row["rank"]),
                "is_sharded": bool(row["is_sharded"]),
                "owned_param_count": int(row["owned_param_count"]),
                "full_param_leaf_count": int(row["full_param_leaf_count"]),
                "opt_state_bytes": int(row["opt_state_bytes"]),
                "opt_state_gib": float(row["opt_state_gib"]),
                "loss_first": float(row["loss_first"]),
                "loss_last": float(row["loss_last"]),
                "step_time_ms_median": float(row["step_time_ms_median"]),
                "step_time_ms_min": float(row["step_time_ms_min"]),
                "peak_memory_gb": float(row["peak_memory_gb"]),
            }
            for row in rank_rows_sorted
        ],
        "control_run_w1": {
            "loss_first": float(control["loss_first"]),
            "loss_last": float(control["loss_last"]),
            "step_time_ms_median": float(control["step_time_ms_median"]),
            "step_time_ms_min": float(control["step_time_ms_min"]),
            "opt_state_bytes": int(control["opt_state_bytes"]),
            "opt_state_gib": float(control["opt_state_gib"]),
            "peak_memory_gb": float(control["peak_memory_gb"]),
        },
        "loss_diff_w2_vs_w1_relative": float(rel_err),
        "parity_tolerance_relative": float(PARITY_TOLERANCE),
        "parity_passed": bool(parity_passed),
        "config": {
            "steps": int(steps),
            "lr": float(lr),
            "batch_size": int(batch_size),
            "seq_len": int(seq_len),
            "model": "build_local_gb10_quarter_tiny_smoke_model",
            "optimizer": "lion",
            "model_seed": MODEL_SEED,
            "data_seed": DATA_SEED,
        },
        "caveats": [
            "Single-host receipt: both processes run on the same Mac; this is "
            "verification of the ZeRO-1 wrapper's distributed math, not a "
            "throughput claim.",
            "True 2-node parity (across dev-128 + peer-48) remains gated on "
            "the peer-48 hand-off documented in "
            "docs/distributed_zero1_smoke_procedure.md.",
            f"Per-rank batch is identical (data_seed={DATA_SEED}); a real DDP "
            "loader would shard the batch across ranks. Identical batches give "
            "step-for-step parity vs. the W=1 control.",
        ],
    }


# ---------------------------------------------------------------------------
# CLI entry point.
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ZeRO-1 single-host loopback receipt for cppmega.mlx",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Training steps per rank and for the W=1 control (default: 20)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help=f"Learning rate (default: {DEFAULT_LR})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Synthetic batch size per rank (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=DEFAULT_SEQ_LEN,
        help=f"Synthetic sequence length (default: {DEFAULT_SEQ_LEN})",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help=(
            "Run the in-process simulated path instead of the mlx.launch "
            "loopback. Use this when MLX_RANK is not set (i.e. the script "
            "was invoked outside mlx.launch) and the user wants the "
            "fallback receipt."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output JSON path. Defaults: bench/baselines/zero1_loopback_2proc_m4.json "
            "(loopback) or bench/baselines/zero1_simulated_2proc_m4.json (--simulate)."
        ),
    )
    parser.add_argument(
        "--receipt-dir",
        type=Path,
        default=None,
        help=(
            "Directory used by the loopback path to stage per-rank metrics "
            "before rank 0 aggregates. Defaults to a sibling of --out."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.simulate:
        out_path = args.out or DEFAULT_SIMULATE_OUTPUT
        return _run_simulated_pair(
            steps=args.steps,
            lr=args.lr,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            out_path=out_path,
        )

    # Loopback path: must be invoked under mlx.launch -- detected via
    # MLX_RANK env var and a real mx.distributed group of size 2.
    if "MLX_RANK" not in os.environ:
        print(
            "[zero1-loopback] MLX_RANK not set; this path must be launched under "
            "`mlx.launch -n 2 --hosts 127.0.0.1`. To run the in-process "
            "fallback, pass --simulate.",
            file=sys.stderr,
        )
        return 64  # EX_USAGE

    out_path = args.out or DEFAULT_OUTPUT
    receipt_dir = args.receipt_dir or out_path.with_suffix(".ranks")
    return _run_loopback_rank(
        steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        receipt_dir=receipt_dir,
        out_path=out_path,
    )


if __name__ == "__main__":
    raise SystemExit(main())

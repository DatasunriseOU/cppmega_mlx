#!/usr/bin/env python3
"""Production-shape MLX throughput sweep on local_gb10_quarter at T=4096.

This script runs a tokens/second benchmark for the full
``cppmega_mlx.recipes.model_factory.local_gb10_quarter`` profile (depth=13,
hidden=3584, ~1.985B params, MTP=2, AEMEAEMEAEMR pattern) at sequence length
4096 to match the parquet corpus shape (``clang_semantic_4k_v10`` /
``clang_commits_4k_v1``). Two optimizer configurations are exercised:

* **Lion** at the empirically-optimal LR from the Stream E TinyLM smoke:
  ``make_lion(learning_rate=3e-3, betas=(0.9, 0.99), weight_decay=0.1)``.
* **Muon+AdamW** matching the GB10 CUDA wiring:
  ``make_muon(cppmega_cuda_parity=True)``. Both groups share the parity LR.

Bench protocol per (B, optimizer) pair (T=4096 fixed):
1. ``mx.reset_peak_memory()`` and ``mx.clear_cache()`` before the run.
2. Synthetic random tokens (vocab=65536, fixed seed).
3. ``--steps`` total steps, first ``--warmup`` discarded as warm-up.
4. Capture median/p10/p90 tokens per second over the post-warmup window.
5. Stop the sweep early when ``peak_memory_gb > memory_cap_gb``.

The receipt JSON has one row per (optimizer, B) pair; the receipt is **not**
the M0.4 acceptance gate (synthetic tokens, throughput-only). A separate
Path A vs Path B comparison is appended for the winning Muon+AdamW B.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

from cppmega_mlx.recipes.model_factory import (  # noqa: E402
    LOCAL_GB10_QUARTER_DEPTH,
    LOCAL_GB10_QUARTER_FFN_HIDDEN_SIZE,
    LOCAL_GB10_QUARTER_HIDDEN_SIZE,
    LOCAL_GB10_QUARTER_NUM_HEADS,
    LOCAL_GB10_QUARTER_PATTERN,
    LOCAL_GB10_QUARTER_VOCAB_SIZE,
    local_gb10_quarter,
)
from cppmega_mlx.runtime.kernel_policy import get_dispatch_log
from cppmega_mlx.training.loss import next_token_cut_cross_entropy
from cppmega_mlx.training.optimizers import (
    MuonAdamWMulti,
    make_lion,
    make_muon,
)


DEFAULT_OUTPUT = ROOT / "bench" / "baselines" / "local_gb10_quarter_throughput_m4.json"
SEQ_LEN_REQUIRED = 4096
DEFAULT_BATCH_SIZES = (1, 2, 3, 4, 6, 8, 10, 12)
DEFAULT_STEPS = 100
DEFAULT_WARMUP = 50
DEFAULT_MEMORY_CAP_GB = 88.0
DEFAULT_VOCAB_SIZE = LOCAL_GB10_QUARTER_VOCAB_SIZE
DEFAULT_SEED = 4096

OPTIMIZER_CHOICES = ("lion", "muon_adamw")
PATH_CHOICES = ("auto", "ref", "path_b")

LION_KWARGS: dict[str, Any] = {
    "learning_rate": 3e-3,
    "betas": [0.9, 0.99],
    "weight_decay": 0.1,
}
MUON_KWARGS: dict[str, Any] = {"cppmega_cuda_parity": True}

REQUIRED_PATH_B_OPS = ("mamba3_mimo", "m2rnn")
# sparse_mla is not yet wired into the local_gb10_quarter forward (DSA mode
# is a dense placeholder in CausalSelfAttention; cf. cppmega_mlx/nn/attention.py
# AttentionRouteInfo). Path B verification therefore covers the two ops that
# DO dispatch in the live model forward: mamba3_mimo (M layers) and m2rnn
# (R layers). When sparse_mla is wired into model forward we should add it.


@dataclass(frozen=True)
class BenchRow:
    optimizer: str
    batch_size: int
    seq_len: int
    steps_total: int
    steps_warmup: int
    steps_measured: int
    tokens_per_step: int
    tokens_per_second_median: float
    tokens_per_second_p10: float
    tokens_per_second_p90: float
    peak_memory_bytes: int
    peak_memory_gb: float
    memory_cap_gb: float
    memory_cap_hit: bool
    loss_first: float
    loss_last_10_mean: float
    optimizer_state_bytes: int
    optimizer_state_gb: float
    kernel_path: str
    kernel_dispatch: tuple[dict[str, str], ...]
    path_b_dispatched: bool
    grad_checkpoint: bool
    dtype: str
    elapsed_s: float
    error: str | None
    error_type: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimizer": self.optimizer,
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "steps_total": self.steps_total,
            "steps_warmup": self.steps_warmup,
            "steps_measured": self.steps_measured,
            "tokens_per_step": self.tokens_per_step,
            "tokens_per_second_median": _json_float(self.tokens_per_second_median),
            "tokens_per_second_p10": _json_float(self.tokens_per_second_p10),
            "tokens_per_second_p90": _json_float(self.tokens_per_second_p90),
            "peak_memory_bytes": self.peak_memory_bytes,
            "peak_memory_gb": _json_float(self.peak_memory_gb),
            "memory_cap_gb": _json_float(self.memory_cap_gb),
            "memory_cap_hit": self.memory_cap_hit,
            "loss_first": _json_float(self.loss_first),
            "loss_last_10_mean": _json_float(self.loss_last_10_mean),
            "optimizer_state_bytes": self.optimizer_state_bytes,
            "optimizer_state_gb": _json_float(self.optimizer_state_gb),
            "kernel_path": self.kernel_path,
            "kernel_dispatch": [dict(item) for item in self.kernel_dispatch],
            "path_b_dispatched": self.path_b_dispatched,
            "grad_checkpoint": self.grad_checkpoint,
            "dtype": self.dtype,
            "elapsed_s": _json_float(self.elapsed_s),
            "error": self.error,
            "error_type": self.error_type,
        }


def _json_float(value: float | None) -> float | None:
    if value is None:
        return None
    return value if math.isfinite(value) else None


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    lower_idx = int(math.floor(pos))
    upper_idx = int(math.ceil(pos))
    weight = pos - lower_idx
    return float(
        sorted_values[lower_idx] * (1.0 - weight)
        + sorted_values[upper_idx] * weight
    )


def _flatten_arrays(tree: Any) -> Iterable[mx.array]:
    if isinstance(tree, dict):
        for value in tree.values():
            yield from _flatten_arrays(value)
    elif isinstance(tree, (list, tuple)):
        for value in tree:
            yield from _flatten_arrays(value)
    elif isinstance(tree, mx.array):
        yield tree


def optimizer_state_bytes(optimizer: Any) -> int:
    total = 0
    for arr in _flatten_arrays(optimizer.state):
        total += int(arr.size * arr.dtype.size)
    return total


def parse_int_list(spec: str) -> tuple[int, ...]:
    items = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        items.append(int(token))
    if not items:
        raise ValueError("expected at least one integer")
    return tuple(items)


def parse_str_list(spec: str, choices: tuple[str, ...]) -> tuple[str, ...]:
    items = []
    for token in spec.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token not in choices:
            raise ValueError(
                f"unknown value {token!r}; choices: {', '.join(choices)}"
            )
        items.append(token)
    if not items:
        raise ValueError("expected at least one value")
    return tuple(items)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Throughput sweep on local_gb10_quarter at T=4096 for the Lion and "
            "Muon+AdamW optimizers. Outputs a JSON receipt at "
            f"{DEFAULT_OUTPUT.relative_to(ROOT)}."
        )
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default=",".join(str(b) for b in DEFAULT_BATCH_SIZES),
        help=(
            "Comma-separated batch sizes. The sweep proceeds in the listed "
            "order and stops the first time peak memory exceeds the cap."
        ),
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=SEQ_LEN_REQUIRED,
        help=(
            "Sequence length. The cppmega parquet corpus is T=4096; "
            "anything else is rejected by default."
        ),
    )
    parser.add_argument(
        "--allow-non-4k-seq-len",
        action="store_true",
        help=(
            "Override the T=4096-only guard. Documentation-only knob; "
            "not used by the production sweep."
        ),
    )
    parser.add_argument(
        "--optimizers",
        type=str,
        default=",".join(OPTIMIZER_CHOICES),
        help="Comma-separated optimizer keys to bench (lion, muon_adamw).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help="Total measured steps per (optimizer, B) pair.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help="Warm-up steps discarded before tok/s statistics.",
    )
    parser.add_argument(
        "--memory-cap-gb",
        type=float,
        default=DEFAULT_MEMORY_CAP_GB,
        help="Peak-memory cap in GiB. Sweep stops when a run exceeds the cap.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=DEFAULT_VOCAB_SIZE,
        help="Synthetic-token vocab size; defaults to the profile vocab.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed used for synthetic tokens and MLX RNG.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination JSON receipt path.",
    )
    parser.add_argument(
        "--cce-chunk-rows",
        type=int,
        default=512,
        help=(
            "Cut-cross-entropy chunked logits row count. Avoids materializing "
            "the full [B*T, V] logits tensor at the lm_head."
        ),
    )
    parser.add_argument(
        "--path-b-comparison",
        action="store_true",
        default=True,
        help=(
            "Run the Path A vs Path B comparison at the winning Muon+AdamW B. "
            "On by default; pass --no-path-b-comparison to skip."
        ),
    )
    parser.add_argument(
        "--no-path-b-comparison",
        dest="path_b_comparison",
        action="store_false",
    )
    parser.add_argument(
        "--max-runtime-s",
        type=float,
        default=None,
        help="Optional global wall-time cap; abort the sweep when exceeded.",
    )
    return parser


def device_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "mlx_version": getattr(mx, "__version__", None),
        "default_device": str(mx.default_device()),
    }
    metal = getattr(mx, "metal", None)
    if metal is not None and hasattr(metal, "device_info"):
        try:
            summary["metal_device_info"] = metal.device_info()
        except Exception as exc:  # pragma: no cover - device-dependent
            summary["metal_device_info_error"] = str(exc)
    return summary


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def make_optimizer(optimizer_key: str):
    if optimizer_key == "lion":
        return make_lion(**LION_KWARGS)
    if optimizer_key == "muon_adamw":
        return make_muon(**MUON_KWARGS)
    raise ValueError(f"unknown optimizer key {optimizer_key!r}")


def loss_with_cce(model: nn.Module, batch: dict[str, mx.array], chunk_rows: int):
    return next_token_cut_cross_entropy(
        model,
        batch,
        chunk_rows=chunk_rows,
    )


def synthetic_batch(
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    seed: int,
    step: int,
) -> dict[str, mx.array]:
    rng = np.random.default_rng(seed * 1_000_003 + step)
    tokens_np = rng.integers(low=0, high=vocab_size, size=(batch_size, seq_len), dtype=np.int32)
    tokens = mx.array(tokens_np)
    attention_mask = mx.ones((batch_size, seq_len), dtype=mx.float32)
    return {"tokens": tokens, "attention_mask": attention_mask}


def reset_state() -> None:
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()


def get_peak_memory_bytes() -> int:
    fn = getattr(mx, "get_peak_memory", None)
    if fn is None:
        metal = getattr(mx, "metal", None)
        if metal is None:
            return 0
        fn = getattr(metal, "get_peak_memory", None)
        if fn is None:
            return 0
    try:
        return int(fn())
    except Exception:
        return 0


def path_b_dispatched(
    dispatch_log: Iterable[dict[str, str]],
    *,
    required_ops: tuple[str, ...] = REQUIRED_PATH_B_OPS,
) -> bool:
    seen: dict[str, bool] = {op: False for op in required_ops}
    for entry in dispatch_log:
        op = entry.get("op_name")
        kernel = entry.get("kernel_used")
        if op in seen and kernel == "metal_kernel_fwd_v1":
            seen[op] = True
    return all(seen.values())


def run_bench_one(
    *,
    optimizer_key: str,
    batch_size: int,
    seq_len: int,
    steps: int,
    warmup: int,
    vocab_size: int,
    seed: int,
    cce_chunk_rows: int,
    memory_cap_gb: float,
    kernel_path_label: str,
    grad_checkpoint: bool = True,
    log_prefix: str = "",
) -> BenchRow:
    """Run one (optimizer, B) bench and return a BenchRow."""

    print(
        f"{log_prefix}[bench] optimizer={optimizer_key} B={batch_size} "
        f"T={seq_len} steps={steps} warmup={warmup} "
        f"kernel_path={kernel_path_label}",
        flush=True,
    )

    reset_state()
    mx.random.seed(seed)
    start_wall = time.perf_counter()

    error: str | None = None
    error_type: str | None = None
    step_tps: list[float] = []
    losses: list[float] = []
    last10_mean = float("nan")
    optimizer_state_size = 0
    dispatch_snapshot: tuple[dict[str, str], ...] = ()
    peak_bytes = 0
    steps_completed = 0

    model = None
    optimizer = None
    try:
        model = local_gb10_quarter(grad_checkpoint=grad_checkpoint)
        model.set_dtype(mx.bfloat16)

        optimizer = make_optimizer(optimizer_key)
        optimizer.init(model.trainable_parameters())
        mx.eval(model.parameters(), optimizer.state)

        optimizer_state_size = optimizer_state_bytes(optimizer)

        loss_fn = lambda m, batch: loss_with_cce(m, batch, cce_chunk_rows)
        loss_and_grad = nn.value_and_grad(model, loss_fn)

        cap_bytes = int(memory_cap_gb * (1024**3))
        cap_hit_early = False

        # Run the steps.
        for step in range(steps):
            batch = synthetic_batch(
                batch_size=batch_size,
                seq_len=seq_len,
                vocab_size=vocab_size,
                seed=seed,
                step=step,
            )
            mx.synchronize()
            t0 = time.perf_counter()

            (loss, ntokens), grads = loss_and_grad(model, batch)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss, ntokens)
            mx.synchronize()
            dt = time.perf_counter() - t0

            tokens_in_step = batch_size * seq_len
            tps = tokens_in_step / dt if dt > 0 else float("inf")
            losses.append(float(loss.item()))
            if step >= warmup:
                step_tps.append(tps)
            steps_completed = step + 1

            current_peak = get_peak_memory_bytes()
            peak_bytes = max(peak_bytes, current_peak)

            if step == 0:
                dispatch_snapshot = tuple(
                    {str(k): str(v) for k, v in entry.items()}
                    for entry in get_dispatch_log()
                )
                fired_path_b = path_b_dispatched(dispatch_snapshot)
                print(
                    f"{log_prefix}[bench]   step1 loss={losses[-1]:.4f} "
                    f"tok/s={tps:.0f} peak={current_peak / (1024**3):.2f} GB "
                    f"path_b_ok={fired_path_b}",
                    flush=True,
                )
            elif (step + 1) % 10 == 0 or step + 1 == steps:
                print(
                    f"{log_prefix}[bench]   step{step + 1:>3d} loss={losses[-1]:.4f} "
                    f"tok/s={tps:.0f} peak={current_peak / (1024**3):.2f} GB",
                    flush=True,
                )

            if current_peak > cap_bytes:
                cap_hit_early = True
                print(
                    f"{log_prefix}[bench] STOP: B={batch_size} hit "
                    f"{memory_cap_gb:.0f} GB cap "
                    f"(observed {current_peak / (1024**3):.2f} GB) at step {step + 1}",
                    flush=True,
                )
                break

        if losses:
            last10 = losses[-min(10, len(losses)):]
            last10_mean = statistics.fmean(last10)

        if dispatch_snapshot == ():
            dispatch_snapshot = tuple(
                {str(k): str(v) for k, v in entry.items()}
                for entry in get_dispatch_log()
            )
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
        error_type = type(exc).__name__
        print(
            f"{log_prefix}[bench] ERROR optimizer={optimizer_key} B={batch_size}: "
            f"{error_type}: {error}",
            flush=True,
        )

    median_tps = float(statistics.median(step_tps)) if step_tps else float("nan")
    p10 = percentile(step_tps, 0.10) if step_tps else float("nan")
    p90 = percentile(step_tps, 0.90) if step_tps else float("nan")

    if peak_bytes == 0:
        peak_bytes = get_peak_memory_bytes()
    cap_hit = peak_bytes > int(memory_cap_gb * (1024**3))

    elapsed = time.perf_counter() - start_wall

    # Free model + optimizer before the next sweep entry.
    if optimizer is not None:
        del optimizer
    if model is not None:
        del model
    gc.collect()
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()

    return BenchRow(
        optimizer=optimizer_key,
        batch_size=batch_size,
        seq_len=seq_len,
        steps_total=steps,
        steps_warmup=warmup,
        steps_measured=len(step_tps),
        tokens_per_step=batch_size * seq_len,
        tokens_per_second_median=median_tps,
        tokens_per_second_p10=p10,
        tokens_per_second_p90=p90,
        peak_memory_bytes=peak_bytes,
        peak_memory_gb=peak_bytes / (1024**3),
        memory_cap_gb=memory_cap_gb,
        memory_cap_hit=cap_hit,
        loss_first=losses[0] if losses else float("nan"),
        loss_last_10_mean=last10_mean,
        optimizer_state_bytes=optimizer_state_size,
        optimizer_state_gb=optimizer_state_size / (1024**3),
        kernel_path=kernel_path_label,
        kernel_dispatch=dispatch_snapshot,
        path_b_dispatched=path_b_dispatched(dispatch_snapshot),
        grad_checkpoint=grad_checkpoint,
        dtype="bfloat16",
        elapsed_s=elapsed,
        error=error,
        error_type=error_type,
    )


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    if args.seq_len != SEQ_LEN_REQUIRED and not args.allow_non_4k_seq_len:
        raise SystemExit(
            f"--seq-len must be {SEQ_LEN_REQUIRED} (parquet shape); "
            f"override with --allow-non-4k-seq-len if intentional."
        )
    batch_sizes = parse_int_list(args.batch_sizes)
    optimizers = parse_str_list(args.optimizers, OPTIMIZER_CHOICES)
    if args.warmup >= args.steps:
        raise SystemExit("--warmup must be < --steps")

    initial_kernel_env = os.environ.get("CPPMEGA_KERNEL_PATH")
    initial_kernel_label = (initial_kernel_env or "auto").strip().lower() or "auto"
    print(
        f"[bench] sweep batch_sizes={list(batch_sizes)} optimizers={list(optimizers)} "
        f"steps={args.steps} warmup={args.warmup} "
        f"memory_cap_gb={args.memory_cap_gb} "
        f"kernel_path_env={initial_kernel_env or '<unset, defaults to auto>'}",
        flush=True,
    )

    rows: list[BenchRow] = []
    sweep_start = time.perf_counter()
    aborted = False

    for optimizer_key in optimizers:
        for B in batch_sizes:
            if (
                args.max_runtime_s is not None
                and time.perf_counter() - sweep_start > args.max_runtime_s
            ):
                print(
                    f"[bench] global wall-time cap ({args.max_runtime_s:.0f}s) hit; aborting",
                    flush=True,
                )
                aborted = True
                break
            row = run_bench_one(
                optimizer_key=optimizer_key,
                batch_size=B,
                seq_len=args.seq_len,
                steps=args.steps,
                warmup=args.warmup,
                vocab_size=args.vocab_size,
                seed=args.seed,
                cce_chunk_rows=args.cce_chunk_rows,
                memory_cap_gb=args.memory_cap_gb,
                kernel_path_label=initial_kernel_label,
            )
            rows.append(row)
            if row.error is not None:
                # OOM-shaped errors are why we stop the inner sweep.
                # Cap-already-hit returns row.memory_cap_hit=True via measurement;
                # if MLX raised earlier (allocator/pool), treat as cap-equivalent.
                print(
                    f"[bench] STOP: optimizer={optimizer_key} B={B} hit a hard error; "
                    "stopping this optimizer's sweep.",
                    flush=True,
                )
                break
            if row.memory_cap_hit:
                print(
                    f"[bench] STOP: optimizer={optimizer_key} B={B} hit "
                    f"{args.memory_cap_gb:.0f} GB cap; stopping this optimizer's sweep.",
                    flush=True,
                )
                break
        if aborted:
            break

    # Path B vs Path A comparison at the winning B for muon_adamw.
    path_comparison_rows: list[BenchRow] = []
    if args.path_b_comparison and "muon_adamw" in optimizers and not aborted:
        muon_rows = [
            r
            for r in rows
            if r.optimizer == "muon_adamw"
            and r.error is None
            and not r.memory_cap_hit
            and r.steps_measured > 0
        ]
        if muon_rows:
            best = max(muon_rows, key=lambda r: r.tokens_per_second_median)
            print(
                f"[bench] Path comparison at winning Muon+AdamW B={best.batch_size} "
                f"(median tok/s={best.tokens_per_second_median:.0f}, "
                f"peak={best.peak_memory_gb:.2f} GB)",
                flush=True,
            )
            for path_label in ("ref",):
                # Already have Path B (auto/path_b) baseline above; only
                # rerun the reference path explicitly here.
                env_backup = os.environ.get("CPPMEGA_KERNEL_PATH")
                os.environ["CPPMEGA_KERNEL_PATH"] = path_label
                try:
                    row = run_bench_one(
                        optimizer_key="muon_adamw",
                        batch_size=best.batch_size,
                        seq_len=args.seq_len,
                        steps=args.steps,
                        warmup=args.warmup,
                        vocab_size=args.vocab_size,
                        seed=args.seed,
                        cce_chunk_rows=args.cce_chunk_rows,
                        memory_cap_gb=args.memory_cap_gb,
                        kernel_path_label=path_label,
                        log_prefix="[path-cmp] ",
                    )
                    path_comparison_rows.append(row)
                finally:
                    if env_backup is None:
                        os.environ.pop("CPPMEGA_KERNEL_PATH", None)
                    else:
                        os.environ["CPPMEGA_KERNEL_PATH"] = env_backup

    elapsed_total = time.perf_counter() - sweep_start

    receipt = {
        "schema_version": 1,
        "scope": "local_gb10_quarter_throughput_m4",
        "model_profile": "local_gb10_quarter",
        "model_geometry": {
            "depth": LOCAL_GB10_QUARTER_DEPTH,
            "hidden_size": LOCAL_GB10_QUARTER_HIDDEN_SIZE,
            "ffn_hidden_size": LOCAL_GB10_QUARTER_FFN_HIDDEN_SIZE,
            "num_attention_heads": LOCAL_GB10_QUARTER_NUM_HEADS,
            "vocab_size": LOCAL_GB10_QUARTER_VOCAB_SIZE,
            "pattern": LOCAL_GB10_QUARTER_PATTERN,
        },
        "config": {
            "batch_sizes": list(batch_sizes),
            "seq_len": args.seq_len,
            "optimizers": list(optimizers),
            "steps": args.steps,
            "warmup": args.warmup,
            "memory_cap_gb": args.memory_cap_gb,
            "vocab_size": args.vocab_size,
            "seed": args.seed,
            "cce_chunk_rows": args.cce_chunk_rows,
            "grad_checkpoint": True,
            "dtype": "bfloat16",
            "kernel_path_default": initial_kernel_label,
            "lion_kwargs": LION_KWARGS,
            "muon_kwargs": MUON_KWARGS,
            "path_b_comparison_enabled": bool(args.path_b_comparison),
            "max_runtime_s": args.max_runtime_s,
        },
        "device": device_summary(),
        "git_commit": git_commit(),
        "rows": [row.to_dict() for row in rows],
        "path_comparison_rows": [row.to_dict() for row in path_comparison_rows],
        "elapsed_s": elapsed_total,
        "aborted_by_runtime_cap": aborted,
        "synthetic_tokens": True,
        "data_note": (
            "Throughput-only synthetic random tokens. The cppmega parquet "
            "corpus shape is T=4096; this receipt does not satisfy M0.4 by "
            "itself."
        ),
    }
    return receipt


def write_receipt(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def summarize_rows(receipt: dict[str, Any]) -> str:
    lines = []
    lines.append(f"sweep elapsed: {receipt['elapsed_s']:.1f}s")
    lines.append("")
    lines.append("rows:")
    lines.append(
        "  optimizer       B  median tok/s  peak GB  state GB  loss[0]  last10  cap_hit  err"
    )
    for row in receipt["rows"]:
        lines.append(
            "  {opt:<14} {B:>2}  {tps:>11.0f}  {peak:>6.2f}  {state:>7.2f}  "
            "{l0:>6.3f}  {l10:>6.3f}  {cap!s:<7}  {err}".format(
                opt=row["optimizer"],
                B=row["batch_size"],
                tps=(row["tokens_per_second_median"] or float("nan")),
                peak=(row["peak_memory_gb"] or 0.0),
                state=(row["optimizer_state_gb"] or 0.0),
                l0=(row["loss_first"] or float("nan")),
                l10=(row["loss_last_10_mean"] or float("nan")),
                cap=row["memory_cap_hit"],
                err=row["error"] or "",
            )
        )
    if receipt["path_comparison_rows"]:
        lines.append("")
        lines.append("path comparison:")
        for row in receipt["path_comparison_rows"]:
            lines.append(
                "  {path:<6} optimizer={opt} B={B} median tok/s={tps:.0f} "
                "peak={peak:.2f} GB path_b_dispatched={pb}".format(
                    path=row["kernel_path"],
                    opt=row["optimizer"],
                    B=row["batch_size"],
                    tps=(row["tokens_per_second_median"] or 0.0),
                    peak=(row["peak_memory_gb"] or 0.0),
                    pb=row["path_b_dispatched"],
                )
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = run_sweep(args)
    write_receipt(args.out, receipt)
    print(summarize_rows(receipt))
    print(f"\nwrote {args.out}")
    # Exit cleanly even on cap hit; the receipt holds the verdict.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

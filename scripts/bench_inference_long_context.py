"""Stream I KV-q4 long-context smoke benchmark.

The default benchmark uses built-in token-id NIAH and RULER-style tasks. It
exercises the repo-local contiguous QuantizedKVCache path without claiming real
long-context leaderboard quality. Real token-id tasks can be supplied as JSONL
rows using the same schema.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from cppmega_mlx.inference.engine import ContiguousKVCache, make_contiguous_kv_cache
from cppmega_mlx.inference.generation import next_token_logits
from cppmega_mlx.inference.sampling import sample_next_token
from cppmega_mlx.models.hybrid_lm import HybridTinyConfig, HybridTinyLM

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "bench" / "baselines" / "inference_long_context_smoke.json"
SUITE_CHOICES = ("niah", "ruler")
DTYPE_CHOICES = ("float32", "bfloat16")
DEFAULT_VOCAB_SIZE = 128
LARGE_MODEL_LIMIT_BYTES = 10 * 1024**3


@dataclass(frozen=True)
class LongContextTask:
    suite: Literal["niah", "ruler"]
    task_id: str
    task_class: str
    context_ids: tuple[int, ...]
    expected_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class LongContextRun:
    generated_token_ids: tuple[int, ...]
    final_position: int
    quantized_layer_count: int
    elapsed_s: float


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        payload = run_benchmark(args)
    except ValueError as exc:
        error = {
            "status": "error",
            "schema_version": 1,
            "error": str(exc),
        }
        print(json.dumps(error, indent=2, sort_keys=True))
        return 2

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"[long-context] wrote {output}")
        for row in payload["rows"]:
            print(
                f"  {row['suite']}: score={row['score']:.4f} "
                f"context={row['context_length']} kv=q{row['kv_cache']['bits']}"
            )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-jsonl")
    parser.add_argument("--suites", nargs="+", choices=SUITE_CHOICES, default=list(SUITE_CHOICES))
    parser.add_argument("--context-tokens", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measured-steps", type=int, default=3)
    parser.add_argument("--kv-bits", type=int, default=4)
    parser.add_argument("--kv-group-size", type=int, default=32)
    parser.add_argument("--quantized-kv-start", type=int, default=0)
    parser.add_argument("--dtype", choices=DTYPE_CHOICES, default="bfloat16")
    parser.add_argument("--seed", type=int, default=176)
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--json", action="store_true")
    return parser


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    selected_suites = tuple(str(suite) for suite in args.suites)
    tasks = load_tasks(
        Path(args.tasks_jsonl) if args.tasks_jsonl else None,
        selected_suites=selected_suites,
        context_tokens=int(args.context_tokens),
    )
    dtype = _dtype_from_name(str(args.dtype))
    max_context = max(len(task.context_ids) for task in tasks)
    max_seq_length = max_context + int(args.decode_tokens)
    mx.random.seed(int(args.seed))
    model = build_long_context_model(max_seq_length=max_seq_length, dtype=dtype)
    estimated_model_bytes = _estimate_model_bytes(model)
    if estimated_model_bytes >= LARGE_MODEL_LIMIT_BYTES:
        raise ValueError("long-context smoke harness unexpectedly built a large model")

    rows = [
        evaluate_task(
            model,
            task,
            decode_tokens=int(args.decode_tokens),
            warmup_steps=int(args.warmup_steps),
            measured_steps=int(args.measured_steps),
            kv_bits=int(args.kv_bits),
            kv_group_size=int(args.kv_group_size),
        )
        for task in tasks
    ]
    return {
        "status": "ok",
        "schema_version": 1,
        "receipt_scope": "local_kv_q4_long_context_smoke",
        "local_only": True,
        "gb10_parity_claim": False,
        "leaderboard_claim": False,
        "full_model_long_context_claim": False,
        "dataset_source": "jsonl_token_id_tasks"
        if args.tasks_jsonl
        else "built_in_token_id_smoke",
        "suites": list(selected_suites),
        "context_tokens": int(args.context_tokens),
        "decode_tokens": int(args.decode_tokens),
        "warmup_steps": int(args.warmup_steps),
        "measured_steps": int(args.measured_steps),
        "dtype": str(args.dtype),
        "kv_cache": {
            "quantized": True,
            "bits": int(args.kv_bits),
            "group_size": int(args.kv_group_size),
            "quantized_kv_start": int(args.quantized_kv_start),
            "mode": "contiguous_quantized_kv_cache",
            "mixed_start_threshold_exercised": False,
        },
        "model": {
            "kind": "HybridTinyLM",
            "scale": "smoke",
            "vocab_size": model.config.vocab_size,
            "hidden_size": model.config.hidden_size,
            "pattern": model.config.pattern,
            "depth": model.config.depth,
            "attention_heads": model.config.num_attention_heads,
            "dtype": str(args.dtype),
        },
        "memory_safety": {
            "estimated_model_bytes": estimated_model_bytes,
            "large_tensor_limit_bytes": LARGE_MODEL_LIMIT_BYTES,
            "full_model_materialized": False,
        },
        "hardware": _hardware_metadata(),
        "rows": rows,
        "non_claims": {
            "real_niah_ruler_leaderboard": True,
            "full_model_long_context_quality": True,
            "m4_vs_gb10_long_context_parity": True,
            "mixed_quantized_kv_start_threshold": int(args.quantized_kv_start) != 0,
        },
    }


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("context_tokens", "decode_tokens", "measured_steps"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if int(args.warmup_steps) < 0:
        raise ValueError("warmup-steps must be non-negative")
    if int(args.quantized_kv_start) < 0:
        raise ValueError("quantized-kv-start must be non-negative")
    if int(args.quantized_kv_start) != 0:
        raise ValueError(
            "this contiguous KV-q4 smoke harness only supports "
            "--quantized-kv-start 0; mixed bf16->q4 cache transition is not wired"
        )
    if int(args.context_tokens) + int(args.decode_tokens) > 4096:
        raise ValueError(
            "allocation-safe smoke context is capped at 4096 tokens; "
            "use a separate large-run workflow for real long-context receipts"
        )


def build_long_context_model(*, max_seq_length: int, dtype: mx.Dtype) -> HybridTinyLM:
    config = HybridTinyConfig(
        vocab_size=DEFAULT_VOCAB_SIZE,
        hidden_size=64,
        pattern="A",
        depth=1,
        dsa_a_layer_ranks=(),
        num_attention_heads=1,
        num_attention_kv_heads=1,
        max_seq_length=max_seq_length,
        structure_vocab_size=8,
        structure_bottleneck_dim=8,
        structure_num_categories=4,
        structure_max_dep_level=4,
        structure_max_ast_depth=4,
        structure_max_sibling_index=4,
        structure_num_node_types=8,
    )
    return HybridTinyLM(config, dtype=dtype)


def load_tasks(
    path: Path | None,
    *,
    selected_suites: tuple[str, ...],
    context_tokens: int,
) -> list[LongContextTask]:
    if path is None:
        tasks = built_in_tasks(context_tokens)
    else:
        tasks = []
        for line_no, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"{path}:{line_no}: task row must be a JSON object")
            tasks.append(parse_task(raw, source=f"{path}:{line_no}"))

    filtered = [task for task in tasks if task.suite in selected_suites]
    if not filtered:
        raise ValueError("no long-context tasks remain after suite filtering")
    return filtered


def built_in_tasks(context_tokens: int) -> list[LongContextTask]:
    if context_tokens < 4:
        raise ValueError("context-tokens must be at least 4 for built-in tasks")
    return [
        LongContextTask(
            suite="niah",
            task_id="niah_smoke_0",
            task_class="needle_in_haystack",
            context_ids=_make_niah_context(context_tokens),
            expected_token_ids=(73,),
        ),
        LongContextTask(
            suite="ruler",
            task_id="ruler_smoke_0",
            task_class="ruler_variable_tracking",
            context_ids=_make_ruler_context(context_tokens),
            expected_token_ids=(88,),
        ),
    ]


def _make_niah_context(context_tokens: int) -> tuple[int, ...]:
    context = [11] * context_tokens
    context[0] = 10
    context[context_tokens // 2] = 73
    context[-1] = 12
    return tuple(context)


def _make_ruler_context(context_tokens: int) -> tuple[int, ...]:
    base = [20 + (idx % 7) for idx in range(context_tokens)]
    base[0] = 80
    base[context_tokens // 3] = 88
    base[-1] = 81
    return tuple(base)


def parse_task(raw: dict[str, Any], *, source: str) -> LongContextTask:
    suite = raw.get("suite")
    if suite not in SUITE_CHOICES:
        raise ValueError(f"{source}: suite must be one of {', '.join(SUITE_CHOICES)}")
    context_ids = _parse_token_tuple(
        raw.get("context_ids"),
        field="context_ids",
        source=source,
    )
    expected_token_ids = _parse_token_tuple(
        raw.get("expected_token_ids"),
        field="expected_token_ids",
        source=source,
    )
    task_class = str(raw.get("task_class") or _default_task_class(str(suite)))
    return LongContextTask(
        suite=suite,
        task_id=str(raw.get("task_id", source)),
        task_class=task_class,
        context_ids=context_ids,
        expected_token_ids=expected_token_ids,
    )


def _parse_token_tuple(value: Any, *, field: str, source: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{source}: {field} must be a non-empty list")
    tokens: list[int] = []
    for token in value:
        if not isinstance(token, int) or isinstance(token, bool) or token < 0:
            raise ValueError(f"{source}: {field} must contain non-negative ints")
        if token >= DEFAULT_VOCAB_SIZE:
            raise ValueError(f"{source}: {field} token {token} exceeds smoke vocab")
        tokens.append(token)
    return tuple(tokens)


def _default_task_class(suite: str) -> str:
    if suite == "niah":
        return "needle_in_haystack"
    if suite == "ruler":
        return "ruler_variable_tracking"
    raise ValueError(f"unsupported suite {suite!r}")


def evaluate_task(
    model: HybridTinyLM,
    task: LongContextTask,
    *,
    decode_tokens: int,
    warmup_steps: int,
    measured_steps: int,
    kv_bits: int,
    kv_group_size: int,
) -> dict[str, Any]:
    for _ in range(warmup_steps):
        run_task_once(
            model,
            task,
            decode_tokens=decode_tokens,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
        )

    runs = [
        run_task_once(
            model,
            task,
            decode_tokens=decode_tokens,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
        )
        for _ in range(measured_steps)
    ]
    last_run = runs[-1]
    exact_match = last_run.generated_token_ids[: len(task.expected_token_ids)] == task.expected_token_ids
    mean_elapsed_s = statistics.fmean(run.elapsed_s for run in runs)
    processed_tokens = len(task.context_ids) + decode_tokens
    return {
        "suite": task.suite,
        "task_id": task.task_id,
        "task_class": task.task_class,
        "status": "ok",
        "metric": "exact_token_match",
        "score": 1.0 if exact_match else 0.0,
        "exact_match": exact_match,
        "context_length": len(task.context_ids),
        "generated_tokens": len(last_run.generated_token_ids),
        "expected_token_ids": list(task.expected_token_ids),
        "generated_token_ids": list(last_run.generated_token_ids),
        "kv_cache": {
            "quantized": True,
            "bits": kv_bits,
            "group_size": kv_group_size,
            "final_position": last_run.final_position,
            "quantized_layer_count": last_run.quantized_layer_count,
        },
        "timing": {
            "tokens": processed_tokens,
            "mean_elapsed_s": _json_float(mean_elapsed_s),
            "tokens_per_second": _json_float(processed_tokens / mean_elapsed_s),
            "step_times_s": [_json_float(run.elapsed_s) for run in runs],
        },
    }


def run_task_once(
    model: HybridTinyLM,
    task: LongContextTask,
    *,
    decode_tokens: int,
    kv_bits: int,
    kv_group_size: int,
) -> LongContextRun:
    cache = make_contiguous_kv_cache(
        num_layers=1,
        batch_size=1,
        num_kv_heads=1,
        head_dim=64,
        max_seq_len=len(task.context_ids) + decode_tokens,
        dtype=None,
        quantized=True,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
    )
    prompt = mx.array([task.context_ids], dtype=mx.int32)
    start = time.perf_counter()
    logits = next_token_logits(model(prompt, kv_cache=cache), prompt)
    generated: list[int] = []
    for _ in range(decode_tokens):
        next_token = sample_next_token(logits, temperature=0.0).astype(mx.int32)
        mx.eval(next_token)
        token_id = int(np.array(next_token)[0, 0])
        generated.append(token_id)
        logits = next_token_logits(model(next_token, kv_cache=cache), next_token)
    mx.eval(logits)
    elapsed_s = time.perf_counter() - start
    return LongContextRun(
        generated_token_ids=tuple(generated),
        final_position=cache.position(),
        quantized_layer_count=_count_quantized_layers(cache),
        elapsed_s=elapsed_s,
    )


def _count_quantized_layers(cache: ContiguousKVCache) -> int:
    return sum(layer.__class__.__name__ == "QuantizedKVCache" for layer in cache.layers)


def _dtype_from_name(name: str) -> mx.Dtype:
    if name == "float32":
        return mx.float32
    if name == "bfloat16":
        return mx.bfloat16
    raise ValueError(f"unsupported dtype {name!r}")


def _estimate_model_bytes(model: HybridTinyLM) -> int:
    total = 0
    for _name, value in tree_flatten(model.trainable_parameters()):
        if isinstance(value, mx.array):
            array_value = cast(mx.array, value)
            nbytes = getattr(array_value, "nbytes", None)
            if nbytes is not None:
                total += int(nbytes)
            else:
                total += int(array_value.size * array_value.itemsize)
    return total


def _json_float(value: float) -> float:
    if not np.isfinite(value):
        raise ValueError("benchmark timing produced a non-finite value")
    return float(value)


def _hardware_metadata() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "default_device": str(mx.default_device()),
        "mlx_version": getattr(mx, "__version__", None),
    }


if __name__ == "__main__":
    raise SystemExit(main())

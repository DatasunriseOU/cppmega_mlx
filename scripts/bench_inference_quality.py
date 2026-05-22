"""Stream I q4 inference quality smoke harness.

The default benchmark uses built-in token-id tasks shaped like ARC, MMLU, and
HumanEval. It validates the q4 inference path and receipt schema without
claiming real leaderboard quality. Real benchmark data can be supplied as
JSONL rows using the same token-id schema.
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from cppmega_mlx.inference.generation import generate_tokens
from cppmega_mlx.inference.quantization import quantize_module_for_inference
from cppmega_mlx.models.hybrid_lm import HybridTinyConfig, HybridTinyLM

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "bench" / "baselines" / "inference_quality_smoke.json"
SUITE_CHOICES = ("arc", "mmlu", "humaneval")
DTYPE_CHOICES = ("float32", "bfloat16")
DEFAULT_VOCAB_SIZE = 128


@dataclass(frozen=True)
class QualityTask:
    suite: Literal["arc", "mmlu", "humaneval"]
    task_id: str
    prompt_ids: tuple[int, ...]
    choice_token_ids: tuple[int, ...] = ()
    answer_index: int | None = None
    expected_token_ids: tuple[int, ...] = ()


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
        print(f"[quality] wrote {output}")
        for row in payload["suite_rows"]:
            print(f"  {row['suite']}: {row['metric']}={row['score']:.4f}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-jsonl")
    parser.add_argument("--suites", nargs="+", choices=SUITE_CHOICES, default=list(SUITE_CHOICES))
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--dtype", choices=DTYPE_CHOICES, default="bfloat16")
    parser.add_argument("--seed", type=int, default=175)
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--json", action="store_true")
    return parser


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    selected_suites = tuple(str(suite) for suite in args.suites)
    tasks = load_tasks(
        Path(args.tasks_jsonl) if args.tasks_jsonl else None,
        selected_suites=selected_suites,
    )
    dtype = _dtype_from_name(str(args.dtype))
    mx.random.seed(int(args.seed))
    model = build_quality_model(dtype=dtype)
    quantize_module_for_inference(
        model,
        bits=int(args.bits),
        group_size=int(args.group_size),
    )
    quantized_linear_modules = _count_quantized_linear_modules(model)
    if quantized_linear_modules <= 0:
        raise ValueError("q4 quality benchmark quantized zero Linear modules")

    suite_rows = [
        evaluate_suite(model, suite, tasks_for_suite)
        for suite, tasks_for_suite in _group_tasks(tasks, selected_suites).items()
    ]
    return {
        "status": "ok",
        "schema_version": 1,
        "receipt_scope": "local_inference_quality_smoke",
        "local_only": True,
        "gb10_parity_claim": False,
        "leaderboard_claim": False,
        "dataset_source": "jsonl_token_id_tasks"
        if args.tasks_jsonl
        else "built_in_token_id_smoke",
        "suites": list(selected_suites),
        "quantization": {
            "bits": int(args.bits),
            "group_size": int(args.group_size),
            "mode": "affine",
            "quantized_linear_modules": quantized_linear_modules,
            "embed_lm_head_skipped": True,
        },
        "model": {
            "kind": "HybridTinyLM",
            "scale": "smoke",
            "vocab_size": model.config.vocab_size,
            "hidden_size": model.config.hidden_size,
            "pattern": model.config.pattern,
            "depth": model.config.depth,
            "dtype": str(args.dtype),
        },
        "suite_rows": suite_rows,
        "hardware": _hardware_metadata(),
        "non_claims": {
            "real_arc_mmlu_humaneval_leaderboard": True,
            "full_q4_model_quality": True,
            "m4_vs_gb10_quality_parity": True,
        },
    }


def build_quality_model(*, dtype: mx.Dtype) -> HybridTinyLM:
    config = HybridTinyConfig(
        vocab_size=DEFAULT_VOCAB_SIZE,
        hidden_size=32,
        pattern="AE",
        depth=2,
        dsa_a_layer_ranks=(),
        num_attention_heads=4,
        max_seq_length=16,
        moe_num_experts=4,
        moe_top_k=2,
        moe_expert_hidden_size=32,
        moe_shared_expert_hidden_size=32,
        mamba_expand=1,
        mamba_head_dim=4,
        mamba_state_dim=4,
        mamba_groups=1,
        mamba_mimo_rank=1,
        mamba_is_mimo=False,
        mamba_chunk_size=8,
        m2rnn_k_head_dim=4,
        m2rnn_v_head_dim=4,
        m2rnn_num_q_heads=1,
        m2rnn_num_k_heads=1,
        m2rnn_num_v_heads=1,
        m2rnn_num_f_heads=1,
        m2rnn_num_weight_heads=1,
        m2rnn_chunk_size=8,
    )
    return HybridTinyLM(config, dtype=dtype)


def load_tasks(
    path: Path | None,
    *,
    selected_suites: tuple[str, ...],
) -> list[QualityTask]:
    if path is None:
        tasks = built_in_tasks()
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
        raise ValueError("no quality tasks remain after suite filtering")
    return filtered


def built_in_tasks() -> list[QualityTask]:
    return [
        QualityTask(
            suite="arc",
            task_id="arc_smoke_0",
            prompt_ids=(10, 11, 12),
            choice_token_ids=(21, 22, 23, 24),
            answer_index=0,
        ),
        QualityTask(
            suite="mmlu",
            task_id="mmlu_smoke_0",
            prompt_ids=(30, 31, 32),
            choice_token_ids=(41, 42, 43, 44),
            answer_index=1,
        ),
        QualityTask(
            suite="humaneval",
            task_id="humaneval_smoke_0",
            prompt_ids=(50, 51, 52),
            expected_token_ids=(61,),
        ),
    ]


def parse_task(raw: dict[str, Any], *, source: str) -> QualityTask:
    suite = raw.get("suite")
    if suite not in SUITE_CHOICES:
        raise ValueError(f"{source}: suite must be one of {', '.join(SUITE_CHOICES)}")
    task_id = str(raw.get("task_id", source))
    prompt_ids = _parse_token_tuple(raw.get("prompt_ids"), field="prompt_ids", source=source)
    if suite in {"arc", "mmlu"}:
        choice_token_ids = _parse_token_tuple(
            raw.get("choice_token_ids"),
            field="choice_token_ids",
            source=source,
        )
        answer_index = raw.get("answer_index")
        if not isinstance(answer_index, int) or isinstance(answer_index, bool):
            raise ValueError(f"{source}: answer_index must be an integer")
        if not 0 <= answer_index < len(choice_token_ids):
            raise ValueError(f"{source}: answer_index must reference choice_token_ids")
        return QualityTask(
            suite=suite,
            task_id=task_id,
            prompt_ids=prompt_ids,
            choice_token_ids=choice_token_ids,
            answer_index=answer_index,
        )

    expected_token_ids = _parse_token_tuple(
        raw.get("expected_token_ids"),
        field="expected_token_ids",
        source=source,
    )
    return QualityTask(
        suite="humaneval",
        task_id=task_id,
        prompt_ids=prompt_ids,
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


def _group_tasks(
    tasks: list[QualityTask],
    selected_suites: tuple[str, ...],
) -> dict[str, list[QualityTask]]:
    grouped = {suite: [] for suite in selected_suites}
    for task in tasks:
        grouped[task.suite].append(task)
    return {suite: suite_tasks for suite, suite_tasks in grouped.items() if suite_tasks}


def evaluate_suite(
    model: HybridTinyLM,
    suite: str,
    tasks: list[QualityTask],
) -> dict[str, Any]:
    if suite in {"arc", "mmlu"}:
        correct = sum(evaluate_multiple_choice(model, task) for task in tasks)
        metric = "accuracy"
    elif suite == "humaneval":
        correct = sum(evaluate_humaneval(model, task) for task in tasks)
        metric = "pass_at_1_exact_token_match"
    else:
        raise ValueError(f"unsupported suite {suite!r}")
    score = correct / len(tasks)
    return {
        "suite": suite,
        "status": "ok",
        "metric": metric,
        "num_tasks": len(tasks),
        "num_correct": int(correct),
        "score": float(score),
    }


def evaluate_multiple_choice(model: HybridTinyLM, task: QualityTask) -> bool:
    if task.answer_index is None or not task.choice_token_ids:
        raise ValueError(f"{task.task_id}: multiple-choice task is incomplete")
    prompt = mx.array([task.prompt_ids], dtype=mx.int32)
    logits = model(prompt)
    mx.eval(logits)
    last_logits = np.array(logits[0, -1, :])
    scores = [float(last_logits[token_id]) for token_id in task.choice_token_ids]
    predicted = int(np.argmax(np.array(scores, dtype=np.float32)))
    return predicted == task.answer_index


def evaluate_humaneval(model: HybridTinyLM, task: QualityTask) -> bool:
    if not task.expected_token_ids:
        raise ValueError(f"{task.task_id}: HumanEval task needs expected_token_ids")
    prompt = mx.array([task.prompt_ids], dtype=mx.int32)
    generated = generate_tokens(
        model,
        prompt,
        max_new_tokens=len(task.expected_token_ids),
        temperature=0.0,
    )
    mx.eval(generated)
    generated_suffix = tuple(int(value) for value in np.array(generated)[0, -len(task.expected_token_ids) :])
    return generated_suffix == task.expected_token_ids


def _count_quantized_linear_modules(model: HybridTinyLM) -> int:
    return sum(isinstance(module, nn.QuantizedLinear) for _name, module in model.named_modules())


def _dtype_from_name(name: str) -> mx.Dtype:
    if name == "float32":
        return mx.float32
    if name == "bfloat16":
        return mx.bfloat16
    raise ValueError(f"unsupported dtype {name!r}")


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


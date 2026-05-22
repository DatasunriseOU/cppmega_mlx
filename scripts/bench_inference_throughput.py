"""Stream I local inference prefill/decode throughput benchmark.

Default runs are intentionally smoke-scale: they exercise the current MLX
inference paths for Qwen3-4B-class and NAM56R-class route shapes without
allocating real multi-billion-parameter models. Full-shape runs must be
requested explicitly with ``--allow-large``.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from cppmega_mlx.inference.generation import (
    PromptCacheEntry,
    build_prompt_cache,
    generate_tokens,
    next_token_logits,
)
from cppmega_mlx.inference.sampling import sample_next_token
from cppmega_mlx.models.hybrid_lm import HybridTinyConfig, HybridTinyLM

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "bench" / "baselines" / "inference_throughput_smoke.json"
PROFILE_CHOICES = ("qwen3_4b_class", "nam56r_class")
DTYPE_CHOICES = ("float32", "bfloat16")
LARGE_MODEL_LIMIT_BYTES = 10 * 1024**3


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    profile_class: str
    target_family: str
    target_parameter_class: str
    config: HybridTinyConfig


@dataclass(frozen=True)
class TimingResult:
    tokens: int
    mean_step_time_s: float
    tokens_per_second: float
    step_times_s: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens": self.tokens,
            "mean_step_time_s": _json_float(self.mean_step_time_s),
            "tokens_per_second": _json_float(self.tokens_per_second),
            "step_times_s": [_json_float(value) for value in self.step_times_s],
        }


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
        print(f"[bench] wrote {output}")
        for row in payload["rows"]:
            print(
                f"  {row['profile']}: prefill={row['prefill']['tokens_per_second']:.2f} "
                f"tok/s decode={row['decode']['tokens_per_second']:.2f} "
                f"tok/s mode={row['decode']['mode']}"
            )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", nargs="+", choices=PROFILE_CHOICES, default=list(PROFILE_CHOICES))
    parser.add_argument("--scale", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--allow-large", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prefill-tokens", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measured-steps", type=int, default=3)
    parser.add_argument("--dtype", choices=DTYPE_CHOICES, default="bfloat16")
    parser.add_argument("--seed", type=int, default=174)
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--json", action="store_true")
    return parser


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    dtype = _dtype_from_name(args.dtype)
    mx.random.seed(int(args.seed))
    rows: list[dict[str, Any]] = []
    for profile_name in args.profiles:
        profile = build_profile_spec(
            profile_name,
            scale=args.scale,
            max_seq_length=int(args.prefill_tokens) + int(args.decode_tokens) + 1,
        )
        rows.append(
            run_profile_benchmark(
                profile,
                batch_size=int(args.batch_size),
                prefill_tokens=int(args.prefill_tokens),
                decode_tokens=int(args.decode_tokens),
                warmup_steps=int(args.warmup_steps),
                measured_steps=int(args.measured_steps),
                dtype=dtype,
                dtype_name=str(args.dtype),
                seed=int(args.seed),
                scale=str(args.scale),
            )
        )

    return {
        "status": "ok",
        "schema_version": 1,
        "receipt_scope": "local_inference_throughput_smoke"
        if args.scale == "smoke"
        else "local_inference_throughput_full",
        "local_only": True,
        "gb10_parity_claim": False,
        "full_model_throughput_claim": bool(args.scale == "full" and args.allow_large),
        "scale": args.scale,
        "batch_size": int(args.batch_size),
        "prefill_tokens": int(args.prefill_tokens),
        "decode_tokens": int(args.decode_tokens),
        "warmup_steps": int(args.warmup_steps),
        "measured_steps": int(args.measured_steps),
        "dtype": args.dtype,
        "hardware": _hardware_metadata(),
        "rows": rows,
    }


def _validate_args(args: argparse.Namespace) -> None:
    if args.scale == "full" and not args.allow_large:
        raise ValueError("full-scale inference throughput requires --allow-large")
    for name in ("batch_size", "prefill_tokens", "decode_tokens", "measured_steps"):
        if int(getattr(args, name.replace("-", "_"), getattr(args, name))) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if int(args.warmup_steps) < 0:
        raise ValueError("warmup-steps must be non-negative")


def build_profile_spec(
    profile_name: str,
    *,
    scale: str,
    max_seq_length: int,
) -> ProfileSpec:
    if scale != "smoke":
        raise ValueError(
            "full-scale profile materialization is not wired for this local harness yet; "
            "use --scale smoke for allocation-safe measurement"
        )
    if profile_name == "qwen3_4b_class":
        config = _smoke_config(
            pattern="AE",
            depth=2,
            max_seq_length=max_seq_length,
        )
        return ProfileSpec(
            name=profile_name,
            profile_class="qwen3-4b-class",
            target_family="Qwen3 dense/MoE inference class",
            target_parameter_class="4B",
            config=config,
        )
    if profile_name == "nam56r_class":
        config = _smoke_config(
            pattern="AEMR",
            depth=4,
            max_seq_length=max_seq_length,
        )
        return ProfileSpec(
            name=profile_name,
            profile_class="nam56r-class",
            target_family="NAM56R A/M/E/R hybrid route class",
            target_parameter_class="full NAM56R",
            config=config,
        )
    raise ValueError(f"unknown inference throughput profile {profile_name!r}")


def _smoke_config(
    *,
    pattern: str,
    depth: int,
    max_seq_length: int,
) -> HybridTinyConfig:
    return HybridTinyConfig(
        vocab_size=128,
        hidden_size=16,
        pattern=pattern,
        depth=depth,
        dsa_a_layer_ranks=(),
        num_attention_heads=4,
        max_seq_length=max_seq_length,
        moe_num_experts=4,
        moe_top_k=2,
        moe_expert_hidden_size=32,
        moe_shared_expert_hidden_size=16,
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


def run_profile_benchmark(
    profile: ProfileSpec,
    *,
    batch_size: int,
    prefill_tokens: int,
    decode_tokens: int,
    warmup_steps: int,
    measured_steps: int,
    dtype: mx.Dtype,
    dtype_name: str,
    seed: int,
    scale: str,
) -> dict[str, Any]:
    model = HybridTinyLM(profile.config, dtype=dtype)
    parameter_bytes = _parameter_bytes(model)
    if parameter_bytes >= LARGE_MODEL_LIMIT_BYTES:
        raise ValueError(
            f"{profile.name} estimated model bytes {parameter_bytes} exceeds "
            f"the {LARGE_MODEL_LIMIT_BYTES} byte default safety limit"
        )
    prompt = _synthetic_prompt(
        batch_size=batch_size,
        seq_len=prefill_tokens,
        vocab_size=profile.config.vocab_size,
        seed=seed + _stable_profile_offset(profile.name),
    )
    _clear_mlx_cache()
    _reset_peak_memory()

    prefill = _measure(
        lambda: model(prompt),
        tokens=batch_size * prefill_tokens,
        warmup_steps=warmup_steps,
        measured_steps=measured_steps,
    )
    decode_mode = _decode_mode(profile.config)
    decode = _measure_decode(
        model,
        prompt,
        decode_tokens=decode_tokens,
        decode_mode=decode_mode,
        warmup_steps=warmup_steps,
        measured_steps=measured_steps,
        dtype=dtype,
    )
    return {
        "profile": profile.name,
        "profile_class": profile.profile_class,
        "target_family": profile.target_family,
        "target_parameter_class": profile.target_parameter_class,
        "status": "ok",
        "actual_config": {
            "scale": scale,
            "vocab_size": profile.config.vocab_size,
            "hidden_size": profile.config.hidden_size,
            "depth": profile.config.depth,
            "pattern": profile.config.pattern,
            "max_seq_length": profile.config.max_seq_length,
            "dtype": dtype_name,
        },
        "prefill": prefill.to_dict() | {"mode": "forward_full_prompt"},
        "decode": decode.to_dict() | {"mode": decode_mode},
        "memory_safety": {
            "default_limit_bytes": LARGE_MODEL_LIMIT_BYTES,
            "estimated_model_bytes": parameter_bytes,
            "peak_memory_bytes": _get_peak_memory_bytes(),
            "large_model_allocation": False,
        },
        "route_roles": list(getattr(model, "route_roles", ())),
    }


def _measure(
    fn: Callable[[], mx.array],
    *,
    tokens: int,
    warmup_steps: int,
    measured_steps: int,
) -> TimingResult:
    times: list[float] = []
    for step in range(warmup_steps + measured_steps):
        start = time.perf_counter()
        value = fn()
        mx.eval(value)
        elapsed = time.perf_counter() - start
        if step >= warmup_steps:
            times.append(elapsed)
    return _timing_result(tokens=tokens, times=times)


def _measure_decode(
    model: HybridTinyLM,
    prompt: mx.array,
    *,
    decode_tokens: int,
    decode_mode: Literal["contiguous_kv_cache", "eager_full_prefix"],
    warmup_steps: int,
    measured_steps: int,
    dtype: mx.Dtype,
) -> TimingResult:
    batch_size = int(prompt.shape[0])
    times: list[float] = []
    for step in range(warmup_steps + measured_steps):
        if decode_mode == "contiguous_kv_cache":
            prompt_cache = build_prompt_cache(
                model,
                prompt,
                num_layers=_attention_layer_count(model),
                num_kv_heads=model.config.num_attention_kv_heads
                or model.config.num_attention_heads,
                head_dim=model.config.hidden_size // model.config.num_attention_heads,
                max_seq_len=int(prompt.shape[1]) + decode_tokens + 1,
                dtype=dtype,
            )
            mx.eval(prompt_cache.next_logits)
            start = time.perf_counter()
            value = _decode_from_prefilled_cache(
                model,
                prompt,
                prompt_cache,
                decode_tokens=decode_tokens,
            )
        else:
            start = time.perf_counter()
            value = generate_tokens(
                model,
                prompt,
                max_new_tokens=decode_tokens,
                temperature=0.0,
            )
        mx.eval(value)
        elapsed = time.perf_counter() - start
        if step >= warmup_steps:
            times.append(elapsed)
    return _timing_result(tokens=batch_size * decode_tokens, times=times)


def _decode_from_prefilled_cache(
    model: HybridTinyLM,
    prompt: mx.array,
    prompt_cache: PromptCacheEntry,
    *,
    decode_tokens: int,
) -> mx.array:
    tokens = prompt
    step_logits = prompt_cache.next_logits
    kv_cache = prompt_cache.cache
    for step in range(decode_tokens):
        next_token = sample_next_token(step_logits, temperature=0.0).astype(tokens.dtype)
        tokens = mx.concatenate([tokens, next_token], axis=1)
        if step + 1 >= decode_tokens:
            break
        step_logits = next_token_logits(model(next_token, kv_cache=kv_cache), next_token)
    return tokens


def _timing_result(*, tokens: int, times: list[float]) -> TimingResult:
    mean_s = statistics.mean(times)
    return TimingResult(
        tokens=tokens,
        mean_step_time_s=mean_s,
        tokens_per_second=tokens / max(mean_s, 1e-12),
        step_times_s=tuple(times),
    )


def _decode_mode(
    config: HybridTinyConfig,
) -> Literal["contiguous_kv_cache", "eager_full_prefix"]:
    roles = {layer.role for layer in config.expanded_pattern().layers}
    stateful_roles = {"mamba3", "m2rnn", "engram"}
    if roles.isdisjoint(stateful_roles) and "attention" in roles:
        return "contiguous_kv_cache"
    return "eager_full_prefix"


def _attention_layer_count(model: HybridTinyLM) -> int:
    return sum(1 for role in model.route_roles if role == "attention")


def _synthetic_prompt(
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    seed: int,
) -> mx.array:
    rng = np.random.default_rng(seed)
    data = rng.integers(0, vocab_size, size=(batch_size, seq_len), dtype=np.int32)
    return mx.array(data, dtype=mx.int32)


def _parameter_bytes(model: HybridTinyLM) -> int:
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


def _stable_profile_offset(profile_name: str) -> int:
    return sum(ord(char) for char in profile_name)


def _dtype_from_name(name: str) -> mx.Dtype:
    if name == "float32":
        return mx.float32
    if name == "bfloat16":
        return mx.bfloat16
    raise ValueError(f"unsupported dtype {name!r}")


def _clear_mlx_cache() -> None:
    clear_cache = getattr(mx, "clear_cache", None)
    if callable(clear_cache):
        clear_cache()


def _reset_peak_memory() -> None:
    reset = getattr(mx, "reset_peak_memory", None)
    if callable(reset):
        reset()


def _get_peak_memory_bytes() -> int | None:
    getter = getattr(mx, "get_peak_memory", None)
    if callable(getter):
        raw_value = getter()
        if isinstance(raw_value, int | float):
            return int(raw_value)
    metal = getattr(mx, "metal", None)
    getter = getattr(metal, "get_peak_memory", None)
    if callable(getter):
        raw_value = getter()
        if isinstance(raw_value, int | float):
            return int(raw_value)
    return None


def _hardware_metadata() -> dict[str, Any]:
    device_info = None
    metal = getattr(mx, "metal", None)
    if metal is not None and callable(getattr(metal, "device_info", None)):
        try:
            device_info = metal.device_info()
        except RuntimeError:
            device_info = None
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "mlx_version": getattr(mx, "__version__", None),
        "default_device": str(mx.default_device()),
        "metal_device_info": device_info,
    }


def _json_float(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value) if math.isfinite(value) else None


if __name__ == "__main__":
    raise SystemExit(main())

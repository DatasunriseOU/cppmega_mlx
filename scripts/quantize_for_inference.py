"""Inference-only quantization helper for local MLX smoke workflows.

This script wraps ``cppmega_mlx.inference.quantization`` for manifestable local
q4 inference checks. It does not load or rewrite production checkpoints yet; it
is a fail-closed CLI around the repo-local inference quantization primitives.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from cppmega_mlx.inference.quantization import (
    InferenceQuantizationConfig,
    make_quantized_kv_cache,
    quantize_module_for_inference,
    validate_kv_head_dim,
)
from cppmega_mlx.models.hybrid_lm import HybridTinyConfig, HybridTinyLM

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "bench" / "baselines" / "quantize_for_inference_smoke.json"
PRESET_CHOICES = ("smoke_attention", "smoke_hybrid")
DTYPE_CHOICES = ("float32", "bfloat16")
LARGE_MODEL_LIMIT_BYTES = 10 * 1024**3


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        payload = run_quantization(args)
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
        quant = payload["linear_quantization"]
        print(f"[quantize] wrote {output}")
        print(
            "  quantized_linear_modules="
            f"{quant['quantized_linear_modules']} remaining_linear_modules="
            f"{quant['remaining_linear_modules']}"
        )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESET_CHOICES, default="smoke_attention")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--kv-bits", type=int, default=4)
    parser.add_argument("--kv-group-size", type=int, default=32)
    parser.add_argument("--quantized-kv-start", type=int, default=0)
    parser.add_argument("--dtype", choices=DTYPE_CHOICES, default="bfloat16")
    parser.add_argument("--check-forward", action="store_true")
    parser.add_argument("--seed", type=int, default=178)
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--json", action="store_true")
    return parser


def run_quantization(args: argparse.Namespace) -> dict[str, Any]:
    quant_config = InferenceQuantizationConfig(
        bits=int(args.bits),
        group_size=int(args.group_size),
        kv_bits=int(args.kv_bits),
        kv_group_size=int(args.kv_group_size),
        quantized_kv_start=int(args.quantized_kv_start),
    )
    dtype = _dtype_from_name(str(args.dtype))
    mx.random.seed(int(args.seed))
    model = build_preset_model(str(args.preset), dtype=dtype)
    validate_kv_head_dim(model.config.hidden_size // model.config.num_attention_heads, group_size=quant_config.kv_group_size)
    pre_quant_bytes = _parameter_bytes(model)
    if pre_quant_bytes >= LARGE_MODEL_LIMIT_BYTES:
        raise ValueError("quantize_for_inference smoke preset exceeded memory limit")

    prompt = _synthetic_prompt(
        vocab_size=model.config.vocab_size,
        seq_len=min(4, model.config.max_seq_length),
        seed=int(args.seed),
    )
    source_logits = model(prompt) if args.check_forward else None
    if source_logits is not None:
        mx.eval(source_logits)

    quantize_module_for_inference(
        model,
        bits=quant_config.bits,
        group_size=quant_config.group_size,
    )
    quantized_logits = model(prompt) if args.check_forward else None
    if quantized_logits is not None:
        mx.eval(quantized_logits)

    kv_cache = make_quantized_kv_cache(
        bits=quant_config.kv_bits,
        group_size=quant_config.kv_group_size,
    )
    return {
        "status": "ok",
        "schema_version": 1,
        "receipt_scope": "local_inference_quantization_manifest",
        "local_only": True,
        "training_quantization_claim": False,
        "full_checkpoint_converter_claim": False,
        "gb10_parity_claim": False,
        "preset": str(args.preset),
        "model": {
            "kind": "HybridTinyLM",
            "scale": "smoke",
            "vocab_size": model.config.vocab_size,
            "hidden_size": model.config.hidden_size,
            "pattern": model.config.pattern,
            "depth": model.config.depth,
            "dtype": str(args.dtype),
        },
        "linear_quantization": {
            "bits": quant_config.bits,
            "group_size": quant_config.group_size,
            "mode": quant_config.mode,
            "quantized_linear_modules": _count_modules(model, nn.QuantizedLinear),
            "remaining_linear_modules": _count_modules(model, nn.Linear),
            "embed_lm_head_skipped": True,
        },
        "kv_cache": {
            "quantized": True,
            "bits": quant_config.kv_bits,
            "group_size": quant_config.kv_group_size,
            "quantized_kv_start": quant_config.quantized_kv_start,
            "class_name": kv_cache.__class__.__name__,
        },
        "forward_check": _forward_check_payload(source_logits, quantized_logits),
        "memory_safety": {
            "pre_quant_model_bytes": pre_quant_bytes,
            "post_quant_model_bytes": _parameter_bytes(model),
            "large_tensor_limit_bytes": LARGE_MODEL_LIMIT_BYTES,
            "full_model_materialized": False,
        },
        "hardware": _hardware_metadata(),
        "non_claims": {
            "training_quantization": True,
            "full_checkpoint_conversion": True,
            "m4_vs_gb10_quantization_parity": True,
        },
    }


def build_preset_model(preset: str, *, dtype: mx.Dtype) -> HybridTinyLM:
    if preset == "smoke_attention":
        config = _smoke_config(pattern="A", depth=1, max_seq_length=8)
    elif preset == "smoke_hybrid":
        config = _smoke_config(pattern="AEMR", depth=4, max_seq_length=8)
    else:
        raise ValueError(f"unsupported preset {preset!r}")
    return HybridTinyLM(config, dtype=dtype)


def _smoke_config(*, pattern: str, depth: int, max_seq_length: int) -> HybridTinyConfig:
    return HybridTinyConfig(
        vocab_size=128,
        hidden_size=64,
        pattern=pattern,
        depth=depth,
        dsa_a_layer_ranks=(),
        num_attention_heads=2,
        num_attention_kv_heads=2,
        max_seq_length=max_seq_length,
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


def _synthetic_prompt(*, vocab_size: int, seq_len: int, seed: int) -> mx.array:
    rng = np.random.default_rng(seed)
    data = rng.integers(0, vocab_size, size=(1, seq_len), dtype=np.int32)
    return mx.array(data, dtype=mx.int32)


def _forward_check_payload(
    source_logits: mx.array | None,
    quantized_logits: mx.array | None,
) -> dict[str, Any]:
    if source_logits is None or quantized_logits is None:
        return {
            "enabled": False,
            "finite": None,
            "max_abs_diff": None,
        }
    source = np.array(source_logits.astype(mx.float32))
    quantized = np.array(quantized_logits.astype(mx.float32))
    finite = bool(np.isfinite(source).all() and np.isfinite(quantized).all())
    max_abs_diff = float(np.max(np.abs(source - quantized)))
    return {
        "enabled": True,
        "finite": finite,
        "max_abs_diff": max_abs_diff,
    }


def _count_modules(model: nn.Module, cls: type[nn.Module]) -> int:
    return sum(isinstance(module, cls) for _name, module in model.named_modules())


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

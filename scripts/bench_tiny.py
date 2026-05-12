#!/usr/bin/env python3
"""Tiny MLX training benchmark for local Apple GPU bring-up.

The benchmark intentionally uses synthetic tokens and a small language-model
shape so it can run before the full cppmega model is ported. If a future lane
adds richer tiny-model APIs, this script can grow an adapter while keeping the
fallback path stable.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from functools import partial
from importlib import metadata
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402

try:
    from cppmega_mlx.data.batch import LMTokenBatch, synthetic_token_batch  # noqa: E402
    from cppmega_mlx.models.hybrid_lm import HybridTinyConfig, HybridTinyLM  # noqa: E402
    from cppmega_mlx.models.tiny_lm import TinyLM, TinyLMConfig  # noqa: E402
    from cppmega_mlx.runtime.memory import (  # noqa: E402
        apply_memory_limit_plan,
        memory_limit_plan,
    )
    from cppmega_mlx.training.loss import next_token_cross_entropy  # noqa: E402
    from cppmega_mlx.training.profile import (  # noqa: E402
        MemorySnapshot,
        profile_context,
        profile_step,
    )
except Exception:  # pragma: no cover - exercised only in partial lane checkouts.
    HybridTinyConfig = None
    HybridTinyLM = None
    LMTokenBatch = None
    MemorySnapshot = None
    TinyLM = None
    TinyLMConfig = None
    apply_memory_limit_plan = None
    memory_limit_plan = None
    profile_context = None
    profile_step = None
    synthetic_token_batch = None
    next_token_cross_entropy = None


DTYPES = {
    "float32": mx.float32,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
}
MODEL_ROUTES = ("tiny", "hybrid", "hybrid-a", "hybrid-e", "hybrid-m", "hybrid-r")
HYBRID_ROUTE_PATTERNS = {
    "hybrid": "AEMR",
    "hybrid-a": "A",
    "hybrid-e": "E",
    "hybrid-m": "M",
    "hybrid-r": "R",
}
BENCH_RECEIPT_SCHEMA_VERSION = 1
LOCAL_RECEIPT_SCOPE = "local_only"
MATCHED_RUN_GUARD = (
    "Compare M4 Max and GB10 only when both rows were collected with "
    "identical comparison_key.workload and comparison_key.software."
)
LOCAL_ONLY_RECEIPT_POLICY = (
    "Single-host tiny benchmark receipt only; not M4-vs-GB10 parity evidence. "
    "Cross-host ratios require matched M4 and GB10 rows with identical "
    "comparison_key.workload and comparison_key.software values."
)
SINGLE_HOST_PARITY_POLICY = "No GB10 parity claim from a single-host row."
AUTO_WIRED_METAL_RATIO = 0.85


@dataclass(frozen=True)
class BenchConfig:
    hardware_label: str = "local"
    batch_size: int = 2
    seq_len: int = 64
    vocab_size: int = 2048
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    mlp_dim: int = 512
    dtype: str = "bfloat16"
    learning_rate: float = 1e-3
    warmup_steps: int = 2
    steps: int = 5
    seed: int = 0
    compile: bool = True
    include_structure: bool = False
    model_route: str = "tiny"
    auto_wired_limit: bool = False
    wired_limit_bytes: int | None = None


class FallbackTinyLM(nn.Module):
    """Small causal LM used until the lane-7 tiny trainer/model is available."""

    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        mlp_dim: int,
        dtype: mx.Dtype,
    ) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.position_embed = nn.Embedding(4096, d_model)
        self.encoder = nn.TransformerEncoder(
            n_layers,
            d_model,
            n_heads,
            mlp_dim,
            dropout=0.0,
            norm_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.dtype = dtype
        self.set_dtype(dtype)
        self.eval()

    def __call__(self, tokens: mx.array) -> mx.array:
        _, seq_len = tokens.shape
        positions = mx.arange(seq_len)
        x = self.token_embed(tokens) + self.position_embed(positions)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len, self.dtype)
        x = self.encoder(x, mask)
        return self.output(self.norm(x))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark a tiny synthetic MLX training step on Apple GPU.",
    )
    parser.add_argument(
        "--hardware-label",
        default=default_hardware_label(),
        help="Human label for comparable runs, e.g. 'M4 Max' or 'GB10'.",
    )
    parser.add_argument("--batch-size", type=int, default=BenchConfig.batch_size)
    parser.add_argument("--seq-len", type=int, default=BenchConfig.seq_len)
    parser.add_argument("--vocab-size", type=int, default=BenchConfig.vocab_size)
    parser.add_argument("--d-model", type=int, default=BenchConfig.d_model)
    parser.add_argument("--n-heads", type=int, default=BenchConfig.n_heads)
    parser.add_argument("--n-layers", type=int, default=BenchConfig.n_layers)
    parser.add_argument("--mlp-dim", type=int, default=BenchConfig.mlp_dim)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default=BenchConfig.dtype)
    parser.add_argument("--lr", type=float, default=BenchConfig.learning_rate)
    parser.add_argument("--warmup-steps", type=int, default=BenchConfig.warmup_steps)
    parser.add_argument("--steps", type=int, default=BenchConfig.steps)
    parser.add_argument("--seed", type=int, default=BenchConfig.seed)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument(
        "--model-route",
        choices=MODEL_ROUTES,
        default=BenchConfig.model_route,
        help=(
            "Model route to benchmark: tiny, full hybrid AEMR, or a single "
            "hybrid backend route."
        ),
    )
    parser.add_argument(
        "--include-structure",
        action="store_true",
        help="Include cppmega structure side-channel tensors when local APIs exist.",
    )
    wired_group = parser.add_mutually_exclusive_group()
    wired_group.add_argument(
        "--auto-wired-limit",
        action="store_true",
        help=(
            "Set MLX wired memory limit to the device max recommended working "
            "set when Metal reports one."
        ),
    )
    wired_group.add_argument(
        "--wired-limit-bytes",
        type=int,
        default=None,
        help="Set an explicit MLX wired memory limit in bytes before benchmarking.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit only the metrics JSON object.",
    )
    parser.add_argument(
        "--compare-line",
        action="store_true",
        help="Emit one key=value line with the stable GB10/M4 comparison fields.",
    )
    parser.add_argument(
        "--dry-run-json",
        action="store_true",
        help="Validate configuration and emit the planned benchmark without running it.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def config_from_args(args: argparse.Namespace) -> BenchConfig:
    return BenchConfig(
        hardware_label=args.hardware_label,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_dim=args.mlp_dim,
        dtype=args.dtype,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        steps=args.steps,
        seed=args.seed,
        compile=not args.no_compile,
        include_structure=args.include_structure or args.model_route.startswith("hybrid"),
        model_route=args.model_route,
        auto_wired_limit=args.auto_wired_limit,
        wired_limit_bytes=args.wired_limit_bytes,
    )


def validate_config(config: BenchConfig) -> None:
    positive_fields = (
        "batch_size",
        "seq_len",
        "vocab_size",
        "d_model",
        "n_heads",
        "n_layers",
        "mlp_dim",
        "steps",
    )
    for field in positive_fields:
        if getattr(config, field) <= 0:
            raise ValueError(f"{field} must be > 0")
    if config.warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")
    if config.seq_len > 4096:
        raise ValueError("seq_len must be <= 4096 for the fallback positional table")
    if config.d_model % config.n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")
    if config.model_route not in MODEL_ROUTES:
        raise ValueError(f"model_route must be one of {MODEL_ROUTES}")
    if (
        config.model_route == "hybrid"
        and config.n_layers < len(HYBRID_ROUTE_PATTERNS["hybrid"])
    ):
        raise ValueError("n_layers must be >= 4 for the full hybrid AEMR route")
    if config.model_route.startswith("hybrid") and not use_project_hybrid_api():
        raise RuntimeError("project hybrid model API is not available")
    if config.wired_limit_bytes is not None and config.wired_limit_bytes < 0:
        raise ValueError("wired_limit_bytes must be >= 0")


def use_project_tiny_api() -> bool:
    return TinyLM is not None and TinyLMConfig is not None and synthetic_token_batch is not None


def use_project_hybrid_api() -> bool:
    return (
        HybridTinyLM is not None
        and HybridTinyConfig is not None
        and synthetic_token_batch is not None
    )


def fallback_synthetic_batch(config: BenchConfig) -> tuple[mx.array, mx.array]:
    shape = (config.batch_size, config.seq_len)
    tokens = mx.random.randint(0, config.vocab_size, shape)
    targets = mx.random.randint(0, config.vocab_size, shape)
    mx.eval(tokens, targets)
    return tokens, targets


def fallback_loss_fn(model: nn.Module, tokens: mx.array, targets: mx.array) -> mx.array:
    logits = model(tokens)
    return nn.losses.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="mean",
    )


def project_loss_fn(model: nn.Module, batch: Any) -> mx.array:
    if next_token_cross_entropy is None:
        raise RuntimeError("project tiny loss API is not available")
    if LMTokenBatch is not None and isinstance(batch, dict):
        batch = LMTokenBatch(**batch)
    loss, _ = next_token_cross_entropy(model, batch)
    return loss


def parameter_count(model: nn.Module) -> int:
    return _nested_parameter_count(model.parameters())


def _nested_parameter_count(tree: Any) -> int:
    if hasattr(tree, "size"):
        return int(tree.size)
    if isinstance(tree, list | tuple):
        return sum(_nested_parameter_count(value) for value in tree)
    if not isinstance(tree, dict):
        return 0
    total = 0
    for value in tree.values():
        total += _nested_parameter_count(value)
    return total


def default_hardware_label() -> str:
    try:
        if hasattr(mx, "device_info"):
            device_name = mx.device_info().get("device_name")
            if device_name:
                return str(device_name)
    except Exception:
        pass
    return platform.node() or platform.machine() or "local"


def comparable_fields(
    config: BenchConfig,
    *,
    tokens_per_second: float | None = None,
    peak_memory_bytes: int | None = None,
) -> dict[str, Any]:
    return {
        "hardware_label": config.hardware_label,
        "dtype": config.dtype,
        "batch_size": config.batch_size,
        "seq_len": config.seq_len,
        "warmup_steps": config.warmup_steps,
        "measured_steps": config.steps,
        "compile": config.compile,
        "include_structure": config.include_structure,
        "tokens_per_second": tokens_per_second,
        "peak_memory_bytes": peak_memory_bytes,
    }


def bytes_to_gib(value: int | None) -> float | None:
    return None if value is None else value / 1024**3


def hybrid_pattern_for_route(model_route: str) -> str:
    try:
        return HYBRID_ROUTE_PATTERNS[model_route]
    except KeyError as exc:
        raise ValueError(f"{model_route!r} is not a hybrid model route") from exc


def hybrid_depth_for_route(config: BenchConfig) -> int:
    pattern = hybrid_pattern_for_route(config.model_route)
    if config.model_route == "hybrid":
        return max(config.n_layers, len(pattern))
    return 1


def _small_positive_divisor(value: int, *, limit: int) -> int:
    for candidate in range(min(value, limit), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def _hybrid_head_dim(hidden_size: int) -> int:
    for candidate in (64, 32, 16, 8, 4, 2, 1):
        if hidden_size % candidate == 0:
            return candidate
    return 1


def _hybrid_m2rnn_v_head_dim(hidden_size: int) -> int:
    return max(1, min(16, _hybrid_head_dim(hidden_size)))


def hybrid_config_from_bench(config: BenchConfig) -> Any:
    if HybridTinyConfig is None:
        raise RuntimeError("project hybrid model API is not available")
    mamba_head_dim = _hybrid_head_dim(config.d_model)
    mamba_groups = _small_positive_divisor(
        max(1, config.d_model // mamba_head_dim),
        limit=4,
    )
    m2rnn_v_head_dim = _hybrid_m2rnn_v_head_dim(config.d_model)
    return HybridTinyConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.d_model,
        pattern=hybrid_pattern_for_route(config.model_route),
        depth=hybrid_depth_for_route(config),
        num_attention_heads=config.n_heads,
        max_seq_length=config.seq_len,
        structure_vocab_size=max(2, min(config.vocab_size, 32)),
        moe_num_experts=2,
        moe_top_k=1,
        moe_expert_hidden_size=config.mlp_dim,
        moe_shared_expert_hidden_size=max(config.d_model, config.mlp_dim // 2),
        mamba_expand=1,
        mamba_head_dim=mamba_head_dim,
        mamba_state_dim=4,
        mamba_groups=mamba_groups,
        mamba_chunk_size=min(max(config.seq_len, 1), 8),
        m2rnn_k_head_dim=max(1, min(16, mamba_head_dim)),
        m2rnn_v_head_dim=m2rnn_v_head_dim,
        m2rnn_num_v_heads=1,
        m2rnn_num_f_heads=1,
        m2rnn_num_g_heads=1,
        m2rnn_num_weight_heads=1,
        m2rnn_chunk_size=min(max(config.seq_len, 1), 8),
    )


def route_plan_for_model(model: nn.Module, config: BenchConfig) -> dict[str, Any]:
    route_symbols = tuple(getattr(model, "route_symbols", ()))
    route_roles = tuple(getattr(model, "route_roles", ()))
    if route_symbols or route_roles:
        return {
            "model_route": config.model_route,
            "route_symbols": "".join(str(symbol) for symbol in route_symbols),
            "route_roles": list(route_roles),
            "pattern": getattr(getattr(model, "config", None), "pattern", None)
            or hybrid_pattern_for_route(config.model_route),
        }
    return {
        "model_route": config.model_route,
        "route_symbols": "tiny",
        "route_roles": ["attention", "ffn"],
        "pattern": "tiny",
    }


def backend_plan_for_model(model: nn.Module, config: BenchConfig) -> dict[str, Any]:
    backends: list[str] = []
    attention_modes: list[str] = []
    attention_backends: list[str] = []
    attention_sparse_reference: list[bool] = []
    for layer in getattr(model, "layers", []):
        backend = getattr(layer, "backend", None)
        if backend is not None:
            backends.append(str(backend))
        elif hasattr(layer, "attn"):
            backends.append("mlx.nn.MultiHeadAttention")
        route_info = getattr(getattr(layer, "block", None), "route_info", None)
        if route_info is not None:
            attention_modes.append(str(getattr(route_info, "mode", "unknown")))
            attention_backends.append(str(getattr(route_info, "backend", "unknown")))
            attention_sparse_reference.append(bool(getattr(route_info, "sparse_reference", False)))
    if not backends and isinstance(model, FallbackTinyLM):
        backends = ["mlx.nn.TransformerEncoder"] * config.n_layers
    counts: dict[str, int] = {}
    for backend in backends:
        counts[backend] = counts.get(backend, 0) + 1
    execution_backend = "mlx"
    if attention_backends:
        execution_backend = "+".join(sorted(set(["mlx", *attention_backends])))
    return {
        "execution_backend": execution_backend,
        "layer_backends": backends,
        "backend_summary": counts,
        "attention_modes": attention_modes,
        "attention_backends": attention_backends,
        "attention_sparse_reference": attention_sparse_reference,
    }


def planned_route_metadata(config: BenchConfig, *, model: nn.Module | None = None) -> dict[str, Any]:
    if model is not None:
        route_plan = route_plan_for_model(model, config)
        backend_plan = backend_plan_for_model(model, config)
    elif config.model_route.startswith("hybrid"):
        pattern = hybrid_pattern_for_route(config.model_route)
        symbol_to_role = {
            "A": "attention",
            "E": "moe",
            "M": "mamba3",
            "R": "m2rnn",
        }
        symbol_to_backend = {
            "A": "attention",
            "E": "moe",
            "M": "mamba3",
            "R": "m2rnn",
        }
        symbols = (pattern * ((hybrid_depth_for_route(config) + len(pattern) - 1) // len(pattern)))[
            : hybrid_depth_for_route(config)
        ]
        route_plan = {
            "model_route": config.model_route,
            "route_symbols": symbols,
            "route_roles": [symbol_to_role[symbol] for symbol in symbols],
            "pattern": pattern,
        }
        backends = [symbol_to_backend[symbol] for symbol in symbols]
        backend_plan = {
            "execution_backend": "mlx+mlx.fast.sdpa" if "A" in symbols else "mlx",
            "layer_backends": backends,
            "backend_summary": {backend: backends.count(backend) for backend in sorted(set(backends))},
            "attention_modes": ["mla"] * symbols.count("A"),
            "attention_backends": ["mlx.fast.sdpa"] * symbols.count("A"),
            "attention_sparse_reference": [False] * symbols.count("A"),
        }
    else:
        route_plan = {
            "model_route": config.model_route,
            "route_symbols": "tiny",
            "route_roles": ["attention", "ffn"],
            "pattern": "tiny",
        }
        backend_plan = {
            "execution_backend": "mlx",
            "layer_backends": ["mlx.nn.MultiHeadAttention"] * config.n_layers,
            "backend_summary": {"mlx.nn.MultiHeadAttention": config.n_layers},
            "attention_modes": [],
            "attention_backends": [],
            "attention_sparse_reference": [],
        }
    return {
        "model_route": config.model_route,
        "route_plan": route_plan,
        "backend_plan": backend_plan,
        "backend": backend_plan["execution_backend"],
        "attention_modes": backend_plan["attention_modes"],
        "attention_backends": backend_plan["attention_backends"],
        "backend_summary": backend_plan["backend_summary"],
    }


def profile_scope_context(
    config: BenchConfig,
    route_metadata: dict[str, Any],
) -> dict[str, Any]:
    device = str(mx.default_device())
    if profile_context is not None:
        return profile_context(
            route=config.model_route,
            backend=str(route_metadata["backend"]),
            device=device,
            model_route=config.model_route,
            route_plan=route_metadata["route_plan"],
            backend_plan=route_metadata["backend_plan"],
        )
    return {
        "route": config.model_route,
        "backend": route_metadata["backend"],
        "device": device,
        "model_route": config.model_route,
        "route_plan": route_metadata["route_plan"],
        "backend_plan": route_metadata["backend_plan"],
    }


def memory_snapshot(*, measured: bool) -> dict[str, Any]:
    if not measured:
        active = peak = cache = None
        available = False
        errors: list[str] = []
    elif MemorySnapshot is not None:
        snapshot = MemorySnapshot.read()
        active = snapshot.active_bytes
        peak = snapshot.peak_bytes
        cache = snapshot.cache_bytes
        available = snapshot.available
        errors = list(snapshot.errors)
    else:
        active = _get_mlx_memory_bytes("get_active_memory")
        peak = _get_mlx_memory_bytes("get_peak_memory")
        cache = _get_mlx_memory_bytes("get_cache_memory")
        available = any(value is not None for value in (active, peak, cache))
        errors = [
            f"mlx.core.{name} unavailable"
            for name, value in (
                ("get_active_memory", active),
                ("get_peak_memory", peak),
                ("get_cache_memory", cache),
            )
            if value is None
        ]
    return {
        "measured": measured,
        "active_bytes": active,
        "active_gib": bytes_to_gib(active),
        "peak_bytes": peak,
        "peak_gib": bytes_to_gib(peak),
        "cache_bytes": cache,
        "cache_gib": bytes_to_gib(cache),
        "available": available,
        "errors": errors,
    }


def _get_mlx_memory_bytes(name: str) -> int | None:
    getter = getattr(mx, name, None)
    if getter is None:
        return None
    try:
        return int(getter())
    except Exception:
        return None


def reset_peak_memory() -> bool:
    reset = getattr(mx, "reset_peak_memory", None)
    if reset is None:
        return False
    try:
        reset()
    except Exception:
        return False
    return True


def metal_is_available() -> bool:
    metal = getattr(mx, "metal", None)
    if metal is None or not hasattr(metal, "is_available"):
        return False
    try:
        return bool(metal.is_available())
    except Exception:
        return False


def device_memory_limits() -> dict[str, int | None]:
    try:
        info = mx.device_info() if hasattr(mx, "device_info") else {}
    except Exception:
        info = {}
    return {
        "memory_size_bytes": _optional_int(info.get("memory_size")),
        "max_recommended_working_set_size_bytes": _optional_int(
            info.get("max_recommended_working_set_size")
        ),
    }


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def wired_limit_report(
    config: BenchConfig,
    *,
    apply: bool,
    mx_module: Any | None = None,
) -> dict[str, Any]:
    mx_target = mx_module or mx
    limits = device_memory_limits()
    if config.wired_limit_bytes is not None:
        mode = "explicit"
        requested = config.wired_limit_bytes
        plan_total = None
    elif config.auto_wired_limit:
        mode = "auto_max_recommended"
        requested = limits["max_recommended_working_set_size_bytes"]
        plan_total = limits["memory_size_bytes"]
    else:
        mode = "off"
        requested = None
        plan_total = None

    report: dict[str, Any] = {
        "mode": mode,
        "requested_bytes": requested,
        "applied_bytes": None,
        "previous_bytes": None,
        "metal_limit_bytes": None,
        "previous_metal_limit_bytes": None,
        "applied": False,
        "available": hasattr(mx_target, "set_wired_limit") and metal_is_available(),
        "memory_size_bytes": limits["memory_size_bytes"],
        "max_recommended_working_set_size_bytes": limits[
            "max_recommended_working_set_size_bytes"
        ],
        "helper": "cppmega_mlx.runtime.memory",
        "memory_limit_plan": None,
        "error": None,
    }
    if mode == "off":
        return report
    if requested is None:
        report["error"] = "device max_recommended_working_set_size is unavailable"
        return report
    if limits["memory_size_bytes"] is not None and requested >= limits["memory_size_bytes"]:
        raise ValueError("wired limit must be strictly less than device memory_size")
    if (
        limits["max_recommended_working_set_size_bytes"] is not None
        and requested > limits["max_recommended_working_set_size_bytes"]
    ):
        raise ValueError(
            "wired limit must be <= max_recommended_working_set_size reported by MLX"
        )
    if not report["available"]:
        report["error"] = "MLX wired limit is unavailable on this backend"
        return report
    if (
        requested > 0
        and memory_limit_plan is not None
        and apply_memory_limit_plan is not None
    ):
        if plan_total is None:
            plan_total = limits["memory_size_bytes"] or requested + 1
        plan = memory_limit_plan(
            plan_total,
            wired_ratio=requested / plan_total,
            metal_ratio=min(AUTO_WIRED_METAL_RATIO, 1 - (1 / plan_total)),
        )
        report["memory_limit_plan"] = plan.to_dict()
        report["metal_limit_bytes"] = plan.metal_limit_bytes
        if not apply:
            return report
        try:
            applied = apply_memory_limit_plan(plan, mx_module=mx_target, apply=True)
            report["previous_bytes"] = applied.previous_wired_limit_bytes
            report["previous_metal_limit_bytes"] = applied.previous_metal_limit_bytes
            report["applied_bytes"] = plan.wired_limit_bytes
            report["applied"] = applied.applied
        except Exception as exc:
            report["error"] = str(exc)
        return report
    if not apply:
        return report
    try:
        report["previous_bytes"] = int(mx_target.set_wired_limit(requested))
        report["applied_bytes"] = requested
        report["applied"] = True
    except Exception as exc:
        report["error"] = str(exc)
    return report


def memory_report(
    *,
    after_warmup: dict[str, Any],
    after_measured_steps: dict[str, Any],
    wired_limit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "active_bytes": after_measured_steps["active_bytes"],
        "active_gib": after_measured_steps["active_gib"],
        "peak_bytes": after_measured_steps["peak_bytes"],
        "peak_gib": after_measured_steps["peak_gib"],
        "cache_bytes": after_measured_steps["cache_bytes"],
        "cache_gib": after_measured_steps["cache_gib"],
        "after_warmup": after_warmup,
        "after_measured_steps": after_measured_steps,
        "wired_limit": wired_limit,
    }


def _profile_scope(
    label: str,
    *,
    tokens: int | None = None,
    eval_args: tuple[Any, ...] = (),
    reset_peak: bool = True,
    extra: dict[str, Any] | None = None,
) -> Any:
    if profile_step is not None:
        return profile_step(
            label,
            tokens=tokens,
            eval_args=eval_args,
            reset_peak=reset_peak,
            sync=True,
            extra=extra,
        )
    return FallbackProfileScope(
        label,
        tokens=tokens,
        eval_args=eval_args,
        reset_peak=reset_peak,
        extra=extra or {},
    )


class FallbackProfileScope:
    """Small local copy of the profile context for partial lane checkouts."""

    def __init__(
        self,
        label: str,
        *,
        tokens: int | None,
        eval_args: tuple[Any, ...],
        reset_peak: bool,
        extra: dict[str, Any],
    ) -> None:
        self.label = label
        self.tokens = tokens
        self.eval_args = list(eval_args)
        self.reset_peak = reset_peak
        self.extra = extra
        self._start = 0.0
        self._peak_memory_reset = False
        self.metrics: FallbackProfileMetrics

    def add_eval_args(self, *args: Any) -> None:
        self.eval_args.extend(args)

    def __enter__(self) -> "FallbackProfileScope":
        if self.reset_peak:
            self._peak_memory_reset = reset_peak_memory()
        mx.synchronize()
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc_type is not None:
            return None
        if self.eval_args:
            mx.eval(*self.eval_args)
        mx.synchronize()
        seconds = time.perf_counter() - self._start
        tokens_per_second = (
            self.tokens / seconds
            if self.tokens is not None and seconds > 0
            else None
        )
        self.metrics = FallbackProfileMetrics(
            label=self.label,
            seconds=seconds,
            tokens=self.tokens,
            tokens_per_second=tokens_per_second,
            memory=memory_snapshot(measured=True),
            peak_memory_reset=self._peak_memory_reset,
            evaluated=bool(self.eval_args),
            extra=self.extra,
        )
        return None


@dataclass(frozen=True)
class FallbackProfileMetrics:
    label: str
    seconds: float
    tokens: int | None
    tokens_per_second: float | None
    memory: dict[str, Any]
    peak_memory_reset: bool
    evaluated: bool
    extra: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "seconds": self.seconds,
            "wall_time_s": self.seconds,
            "elapsed_wall_time_s": self.seconds,
            "tokens": self.tokens,
            "tokens_per_second": self.tokens_per_second,
            "peak_memory_bytes": self.memory["peak_bytes"],
            "active_memory_bytes": self.memory["active_bytes"],
            "cache_memory_bytes": self.memory["cache_bytes"],
            "memory": {
                "active_bytes": self.memory["active_bytes"],
                "peak_bytes": self.memory["peak_bytes"],
                "cache_bytes": self.memory["cache_bytes"],
                "available": self.memory["available"],
                "errors": self.memory["errors"],
            },
            "peak_memory_reset": self.peak_memory_reset,
            "synchronized": True,
            "evaluated": self.evaluated,
            "extra": self.extra,
        }


def profile_helpers_metadata() -> dict[str, Any]:
    return {
        "profile_step": (
            "cppmega_mlx.training.profile.profile_step"
            if profile_step is not None
            else "scripts.bench_tiny.FallbackProfileScope"
        ),
        "memory_snapshot": (
            "cppmega_mlx.training.profile.MemorySnapshot"
            if MemorySnapshot is not None
            else "scripts.bench_tiny.memory_snapshot"
        ),
        "profile_context": (
            "cppmega_mlx.training.profile.profile_context"
            if profile_context is not None
            else "scripts.bench_tiny.profile_scope_context"
        ),
    }


def matched_run_metadata(
    config: BenchConfig,
    *,
    model_source: str | None,
    route_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    device = device_info()
    comparable = comparable_fields(config)
    route_metadata = route_metadata or planned_route_metadata(config)
    stack = {
        "python": device.get("python"),
        "platform": device.get("platform"),
        "machine": device.get("machine"),
        "mlx": device.get("mlx"),
        "mlx_lm": device.get("mlx_lm"),
        "mlx_metal": device.get("mlx_metal"),
        "default_device": device.get("default_device"),
        "mlx_device_info": device.get("mlx_device_info"),
        "metal": device.get("metal"),
    }
    workload = {
        **comparable,
        "vocab_size": config.vocab_size,
        "d_model": config.d_model,
        "n_heads": config.n_heads,
        "n_layers": config.n_layers,
        "mlp_dim": config.mlp_dim,
        "learning_rate": config.learning_rate,
        "seed": config.seed,
        "model_source": model_source,
        "model_route": config.model_route,
        "route_plan": route_metadata["route_plan"],
        "backend_plan": route_metadata["backend_plan"],
        "backend": route_metadata["backend"],
        "data_contract": "synthetic_tokens",
    }
    key = {
        key: workload[key]
        for key in (
            "dtype",
            "batch_size",
            "seq_len",
            "vocab_size",
            "d_model",
            "n_heads",
            "n_layers",
            "mlp_dim",
            "warmup_steps",
            "measured_steps",
            "compile",
            "include_structure",
            "learning_rate",
            "model_source",
            "model_route",
            "route_plan",
            "backend_plan",
            "backend",
            "data_contract",
        )
    }
    return {
        "schema_version": 1,
        "workload": workload,
        "framework": stack,
        "profile_helpers": profile_helpers_metadata(),
        "matched_run": {
            "key": key,
            "receipt_scope": LOCAL_RECEIPT_SCOPE,
            "local_only": True,
            "gb10_parity_claim": False,
            "guard": MATCHED_RUN_GUARD,
            "claim_policy": SINGLE_HOST_PARITY_POLICY,
        },
    }


def software_key_from_metadata(
    metrics: dict[str, Any],
    run_metadata: dict[str, Any],
) -> dict[str, Any]:
    device = metrics.get("device") or {}
    framework_metadata = run_metadata.get("framework") or {}
    mlx_device_info = framework_metadata.get("mlx_device_info") or device.get(
        "mlx_device_info"
    ) or {}
    metal = framework_metadata.get("metal") or device.get("metal")
    framework_name = (
        "mlx" if (framework_metadata.get("mlx") or device.get("mlx")) else None
    )
    backend = metrics.get("backend")
    return {
        "framework": framework_name,
        "backend": backend,
        "execution_backend": backend,
        "framework_backend": "metal" if metal else framework_name,
        "python_version": framework_metadata.get("python") or device.get("python"),
        "platform": framework_metadata.get("platform") or device.get("platform"),
        "machine": framework_metadata.get("machine") or device.get("machine"),
        "mlx_version": framework_metadata.get("mlx") or device.get("mlx"),
        "mlx_lm_version": framework_metadata.get("mlx_lm") or device.get("mlx_lm"),
        "mlx_metal_version": framework_metadata.get("mlx_metal")
        or device.get("mlx_metal"),
        "default_device": framework_metadata.get("default_device")
        or device.get("default_device"),
        "device_name": mlx_device_info.get("device_name"),
        "metal": metal,
    }


def timing_receipt(metrics: dict[str, Any]) -> dict[str, Any]:
    profile = metrics.get("profile") or {}
    measured = (profile.get("scopes") or {}).get("measured_steps") or {}
    synchronized = measured.get("synchronized")
    if synchronized is None and metrics.get("status") == "ok":
        synchronized = True
    tokens_per_second = metrics.get("tokens_per_second")
    mean_step_time_s = metrics.get("mean_step_time_s")
    median_step_time_s = metrics.get("median_step_time_s")
    step_times_s = list(metrics.get("step_times_s") or [])
    wall_time_s = metrics.get("wall_time_s")
    if wall_time_s is None:
        wall_time_s = mean_step_time_s
    mean_wall_time_s = metrics.get("mean_wall_time_s")
    if mean_wall_time_s is None:
        mean_wall_time_s = wall_time_s
    total_wall_time_s = metrics.get("total_wall_time_s")
    if total_wall_time_s is None and step_times_s:
        total_wall_time_s = sum(step_times_s)
    return {
        "tokens_per_step": metrics.get("tokens_per_step"),
        "warmup_steps": metrics.get("warmup_steps"),
        "measured_steps": metrics.get("measured_steps"),
        "compile": metrics.get("compile"),
        "first_call_time_s": metrics.get("first_call_time_s"),
        "compile_time_s": metrics.get("compile_time_s"),
        "mean_step_time_s": mean_step_time_s,
        "wall_time_s": wall_time_s,
        "mean_wall_time_s": mean_wall_time_s,
        "total_wall_time_s": total_wall_time_s,
        "median_step_time_s": median_step_time_s,
        "tokens_per_second": tokens_per_second,
        "tokens_per_second_or_step_time": (
            tokens_per_second is not None
            or mean_step_time_s is not None
            or median_step_time_s is not None
        ),
        "warmup_step_times_s": list(metrics.get("warmup_step_times_s") or []),
        "step_times_s": step_times_s,
        "synchronized_timing": synchronized,
        "timing_method": (
            "wall-clock timing around MLX train steps with mx.eval outputs and "
            "mx.synchronize before reporting; compile first-call time is separate"
        ),
    }


def add_receipt_metadata(metrics: dict[str, Any]) -> dict[str, Any]:
    run_metadata = metrics["run_metadata"]
    workload_key = dict(run_metadata["matched_run"]["key"])
    software_key = software_key_from_metadata(metrics, run_metadata)
    comparison_key = {
        "schema_version": BENCH_RECEIPT_SCHEMA_VERSION,
        "workload": workload_key,
        "software": software_key,
    }
    timing = timing_receipt(metrics)
    receipt = {
        "schema_version": BENCH_RECEIPT_SCHEMA_VERSION,
        "receipt_scope": LOCAL_RECEIPT_SCOPE,
        "local_only": True,
        "gb10_parity_claim": False,
        "hardware_label": metrics.get("hardware_label"),
        "model_route": metrics.get("model_route"),
        "seq_len": metrics.get("seq_len"),
        "batch_size": metrics.get("batch_size"),
        "dtype": metrics.get("dtype"),
        "warmup_steps": timing["warmup_steps"],
        "measured_steps": timing["measured_steps"],
        "compile": timing["compile"],
        "include_structure": metrics.get("include_structure"),
        "tokens_per_second": timing["tokens_per_second"],
        "mean_step_time_s": timing["mean_step_time_s"],
        "wall_time_s": timing["wall_time_s"],
        "mean_wall_time_s": timing["mean_wall_time_s"],
        "total_wall_time_s": timing["total_wall_time_s"],
        "median_step_time_s": timing["median_step_time_s"],
        "device": {
            "default_device": software_key["default_device"],
            "device_name": software_key["device_name"],
            "platform": software_key["platform"],
            "machine": software_key["machine"],
            "metal": software_key["metal"],
        },
        "software": software_key,
        "workload": workload_key,
        "timing": timing,
        "comparison_key": comparison_key,
        "matched_run_guard": MATCHED_RUN_GUARD,
        "parity_claim_policy": SINGLE_HOST_PARITY_POLICY,
        "local_only_policy": LOCAL_ONLY_RECEIPT_POLICY,
    }
    metrics.update(
        {
            "receipt_schema_version": BENCH_RECEIPT_SCHEMA_VERSION,
            "receipt_scope": LOCAL_RECEIPT_SCOPE,
            "local_only": True,
            "gb10_parity_claim": False,
            "workload_key": workload_key,
            "software_key": software_key,
            "comparison_key": comparison_key,
            "matched_run_key": workload_key,
            "bench_receipt": receipt,
            "matched_run": {
                **run_metadata["matched_run"],
                "key": workload_key,
                "receipt_scope": LOCAL_RECEIPT_SCOPE,
                "local_only": True,
                "gb10_parity_claim": False,
                "guard": MATCHED_RUN_GUARD,
                "claim_policy": SINGLE_HOST_PARITY_POLICY,
            },
        }
    )
    metrics["run_metadata"]["matched_run"] = metrics["matched_run"]
    return metrics


def make_train_step(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_callable: Callable[[nn.Module, Any], mx.array],
    batch: Any,
    *,
    compile: bool,
) -> tuple[Callable[..., mx.array], tuple[mx.array, ...]]:
    loss_and_grad = nn.value_and_grad(model, loss_callable)
    pack: Callable[[tuple[mx.array, ...]], Any]

    if isinstance(batch, dict):
        keys = tuple(batch)
        step_args = tuple(batch[key] for key in keys)

        def pack_dict(arrays: tuple[mx.array, ...]) -> dict[str, mx.array]:
            return {key: value for key, value in zip(keys, arrays)}

        pack = pack_dict
    elif isinstance(batch, tuple):
        step_args = batch

        def pack_tuple(arrays: tuple[mx.array, ...]) -> tuple[mx.array, ...]:
            return arrays

        pack = pack_tuple
    else:
        step_args = (batch,)

        def pack_single(arrays: tuple[mx.array, ...]) -> mx.array:
            return arrays[0]

        pack = pack_single

    def train_step(*arrays: mx.array) -> mx.array:
        batch_arg = pack(arrays)
        loss, grads = loss_and_grad(model, batch_arg)
        optimizer.update(model, grads)
        return loss

    if not compile:
        return train_step, step_args

    captured_state = [model.state, optimizer.state, mx.random.state]

    @partial(mx.compile, inputs=captured_state, outputs=captured_state)
    def compiled_train_step(*arrays: mx.array) -> mx.array:
        batch_arg = pack(arrays)
        loss, grads = loss_and_grad(model, batch_arg)
        optimizer.update(model, grads)
        return loss

    return compiled_train_step, step_args


def fallback_loss_callable(model: nn.Module, batch: tuple[mx.array, mx.array]) -> mx.array:
    return fallback_loss_fn(model, batch[0], batch[1])


def build_model_and_batch(config: BenchConfig) -> tuple[nn.Module, Any, str]:
    dtype = DTYPES[config.dtype]
    if config.model_route.startswith("hybrid"):
        hybrid_config_cls = HybridTinyConfig
        hybrid_lm = HybridTinyLM
        token_batch = synthetic_token_batch
        if hybrid_config_cls is None or hybrid_lm is None or token_batch is None:
            raise RuntimeError("project hybrid model API is not available")
        model = hybrid_lm(hybrid_config_from_bench(config))
        model.set_dtype(dtype)
        model.train()
        batch = token_batch(
            batch_size=config.batch_size,
            seq_length=config.seq_len,
            vocab_size=config.vocab_size,
            seed=config.seed,
            include_structure=True,
        )
        return model, batch.as_dict(), "cppmega_mlx.models.hybrid_lm"

    if use_project_tiny_api():
        tiny_lm_config = TinyLMConfig
        tiny_lm = TinyLM
        token_batch = synthetic_token_batch
        if tiny_lm_config is None or tiny_lm is None or token_batch is None:
            raise RuntimeError("project tiny model API is not available")
        tiny_config = tiny_lm_config(
            vocab_size=config.vocab_size,
            hidden_size=config.d_model,
            num_layers=config.n_layers,
            num_heads=config.n_heads,
            ffn_hidden_size=config.mlp_dim,
            max_seq_length=config.seq_len,
            structure_vocab_size=max(2, min(config.vocab_size, 32)),
        )
        model = tiny_lm(tiny_config)
        model.set_dtype(dtype)
        model.train()
        batch = token_batch(
            batch_size=config.batch_size,
            seq_length=config.seq_len,
            vocab_size=config.vocab_size,
            seed=config.seed,
            include_structure=config.include_structure,
        )
        return model, batch.as_dict(), "cppmega_mlx.models.tiny_lm"

    model = FallbackTinyLM(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        mlp_dim=config.mlp_dim,
        dtype=dtype,
    )
    tokens, targets = fallback_synthetic_batch(config)
    return model, (tokens, targets), "self_contained_fallback"


def metadata_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def device_info() -> dict[str, Any]:
    info = {
        "default_device": str(mx.default_device()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "mlx": metadata_version("mlx"),
        "mlx_lm": metadata_version("mlx-lm"),
        "mlx_metal": metadata_version("mlx-metal"),
    }
    if hasattr(mx, "device_info"):
        info["mlx_device_info"] = mx.device_info()
    metal = getattr(mx, "metal", None)
    if metal is not None:
        info["metal"] = {
            "available": metal_is_available(),
            "capture_supported": all(
                hasattr(metal, name) for name in ("start_capture", "stop_capture")
            ),
        }
    return info


def dry_run_payload(config: BenchConfig) -> dict[str, Any]:
    wired_limit = wired_limit_report(config, apply=False)
    if config.model_route.startswith("hybrid"):
        model_source = "cppmega_mlx.models.hybrid_lm"
    else:
        model_source = (
            "cppmega_mlx.models.tiny_lm"
            if use_project_tiny_api()
            else "self_contained_fallback"
        )
    route_metadata = planned_route_metadata(config)
    run_metadata = matched_run_metadata(
        config,
        model_source=model_source,
        route_metadata=route_metadata,
    )
    payload = {
        "status": "dry_run",
        "config": asdict(config),
        "tokens_per_step": config.batch_size * config.seq_len,
        "device": device_info(),
        "model_source": model_source,
        **route_metadata,
        "memory": memory_report(
            after_warmup=memory_snapshot(measured=False),
            after_measured_steps=memory_snapshot(measured=False),
            wired_limit=wired_limit,
        ),
        "profile": {
            "enabled": False,
            "helpers": profile_helpers_metadata(),
            "scopes": {},
        },
        "run_metadata": run_metadata,
        "matched_run": run_metadata["matched_run"],
    }
    payload.update(comparable_fields(config))
    return add_receipt_metadata(payload)


def run_benchmark(config: BenchConfig) -> dict[str, Any]:
    validate_config(config)
    wired_limit = wired_limit_report(config, apply=True)
    mx.random.seed(config.seed)

    model, batch, model_source = build_model_and_batch(config)
    route_metadata = planned_route_metadata(config, model=model)
    scope_context = profile_scope_context(config, route_metadata)
    optimizer = optim.AdamW(learning_rate=config.learning_rate)
    mx.eval(model.state, optimizer.state)

    if model_source in {"cppmega_mlx.models.tiny_lm", "cppmega_mlx.models.hybrid_lm"}:
        loss_callable = project_loss_fn
    else:
        loss_callable = fallback_loss_callable
    step, step_args = make_train_step(
        model,
        optimizer,
        loss_callable,
        batch,
        compile=config.compile,
    )

    profile_scopes: dict[str, dict[str, Any]] = {}
    tokens_per_step = config.batch_size * config.seq_len
    with _profile_scope(
        "first_call",
        tokens=tokens_per_step,
        eval_args=(model.state, optimizer.state),
        reset_peak=True,
        extra={"compile": config.compile, "context": scope_context},
    ) as prof:
        loss = step(*step_args)
        prof.add_eval_args(loss)
    first_call_profile = prof.metrics.to_dict()
    profile_scopes["first_call"] = first_call_profile
    first_call_time_s = float(first_call_profile["seconds"])
    compile_time_s = first_call_time_s if config.compile else 0.0

    warmup_times: list[float] = []
    if config.warmup_steps:
        with _profile_scope(
            "warmup",
            tokens=tokens_per_step * config.warmup_steps,
            eval_args=(model.state, optimizer.state),
            reset_peak=False,
            extra={"steps": config.warmup_steps, "context": scope_context},
        ) as prof:
            for _ in range(config.warmup_steps):
                start = time.perf_counter()
                loss = step(*step_args)
                mx.eval(loss, model.state, optimizer.state)
                mx.synchronize()
                warmup_times.append(time.perf_counter() - start)
            prof.add_eval_args(loss)
        warmup_profile = prof.metrics.to_dict()
        profile_scopes["warmup"] = warmup_profile
        after_warmup_memory = _memory_snapshot_from_profile(
            warmup_profile,
            measured=True,
        )
    else:
        after_warmup_memory = memory_snapshot(measured=True)

    steady_times: list[float] = []
    with _profile_scope(
        "measured_steps",
        tokens=tokens_per_step * config.steps,
        eval_args=(model.state, optimizer.state),
        reset_peak=True,
        extra={"steps": config.steps, "context": scope_context},
    ) as prof:
        for _ in range(config.steps):
            start = time.perf_counter()
            loss = step(*step_args)
            mx.eval(loss, model.state, optimizer.state)
            mx.synchronize()
            steady_times.append(time.perf_counter() - start)
        prof.add_eval_args(loss)
    measured_profile = prof.metrics.to_dict()
    profile_scopes["measured_steps"] = measured_profile
    after_measured_memory = _memory_snapshot_from_profile(measured_profile, measured=True)

    final_loss = float(loss)
    mean_step_s = statistics.fmean(steady_times)
    tokens_per_second = tokens_per_step / mean_step_s
    total_measured_wall_time_s = sum(steady_times)
    peak_memory_bytes = after_measured_memory["peak_bytes"]
    peak_memory_gib = after_measured_memory["peak_gib"]

    run_metadata = matched_run_metadata(
        config,
        model_source=model_source,
        route_metadata=route_metadata,
    )
    metrics = {
        "status": "ok",
        "config": asdict(config),
        "device": device_info(),
        "model_source": model_source,
        **route_metadata,
        "parameter_count": parameter_count(model),
        "tokens_per_step": tokens_per_step,
        "first_call_time_s": first_call_time_s,
        "compile_time_s": compile_time_s,
        "warmup_step_times_s": warmup_times,
        "step_times_s": steady_times,
        "mean_step_time_s": mean_step_s,
        "mean_wall_time_s": mean_step_s,
        "wall_time_s": mean_step_s,
        "total_wall_time_s": total_measured_wall_time_s,
        "median_step_time_s": statistics.median(steady_times),
        "tokens_per_second": tokens_per_second,
        "peak_memory_bytes": peak_memory_bytes,
        "peak_memory_gib": peak_memory_gib,
        "memory": memory_report(
            after_warmup=after_warmup_memory,
            after_measured_steps=after_measured_memory,
            wired_limit=wired_limit,
        ),
        "profile": {
            "enabled": True,
            "helpers": profile_helpers_metadata(),
            "scopes": profile_scopes,
        },
        "run_metadata": run_metadata,
        "matched_run": run_metadata["matched_run"],
        "final_loss": final_loss,
    }
    metrics.update(
        comparable_fields(
            config,
            tokens_per_second=tokens_per_second,
            peak_memory_bytes=peak_memory_bytes,
        )
    )
    return add_receipt_metadata(metrics)


def _memory_snapshot_from_profile(
    profile: dict[str, Any],
    *,
    measured: bool,
) -> dict[str, Any]:
    memory = profile.get("memory") or {}
    active = _optional_int(memory.get("active_bytes"))
    peak = _optional_int(memory.get("peak_bytes"))
    cache = _optional_int(memory.get("cache_bytes"))
    return {
        "measured": measured,
        "active_bytes": active,
        "active_gib": bytes_to_gib(active),
        "peak_bytes": peak,
        "peak_gib": bytes_to_gib(peak),
        "cache_bytes": cache,
        "cache_gib": bytes_to_gib(cache),
        "available": bool(memory.get("available")),
        "errors": list(memory.get("errors") or []),
    }


def format_compare_line(metrics: dict[str, Any]) -> str:
    keys = (
        "hardware_label",
        "dtype",
        "batch_size",
        "seq_len",
        "warmup_steps",
        "measured_steps",
        "compile",
        "include_structure",
        "tokens_per_second",
        "peak_memory_bytes",
    )
    parts: list[str] = []
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, float):
            value = f"{value:.6f}"
        parts.append(f"{key}={value}")
    return " ".join(parts)


def format_memory(value_bytes: int | None, value_gib: float | None) -> str:
    if value_bytes is None or value_gib is None:
        return "unavailable"
    return f"{value_bytes} bytes ({value_gib:.4f} GiB)"


def print_human(metrics: dict[str, Any]) -> None:
    config = metrics["config"]
    print("cppmega.mlx tiny MLX training benchmark")
    print(f"status: {metrics['status']}")
    print(f"device: {metrics['device']['default_device']}")
    if "mlx_device_info" in metrics["device"]:
        print(f"device_name: {metrics['device']['mlx_device_info'].get('device_name')}")
    print(
        "shape: "
        f"batch={config['batch_size']} seq={config['seq_len']} "
        f"vocab={config['vocab_size']} d_model={config['d_model']} "
        f"heads={config['n_heads']} layers={config['n_layers']} "
        f"mlp={config['mlp_dim']} dtype={config['dtype']}"
    )
    print(f"model_source: {metrics['model_source']}")
    print(f"parameter_count: {metrics['parameter_count']:,}")
    print(f"compile_time_s: {metrics['compile_time_s']:.6f}")
    print(f"mean_step_time_s: {metrics['mean_step_time_s']:.6f}")
    print(f"median_step_time_s: {metrics['median_step_time_s']:.6f}")
    print(f"tokens_per_second: {metrics['tokens_per_second']:.2f}")
    print(f"peak_memory: {format_memory(metrics['peak_memory_bytes'], metrics['peak_memory_gib'])}")
    memory = metrics["memory"]
    print(f"active_memory: {format_memory(memory['active_bytes'], memory['active_gib'])}")
    print(f"cache_memory: {format_memory(memory['cache_bytes'], memory['cache_gib'])}")
    wired = memory["wired_limit"]
    print(
        "wired_limit: "
        f"mode={wired['mode']} applied={wired['applied']} "
        f"requested_bytes={wired['requested_bytes']}"
    )
    print(f"final_loss: {metrics['final_loss']:.6f}")
    print("\njson:")
    print(json.dumps(metrics, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = config_from_args(args)
    try:
        validate_config(config)
        metrics = dry_run_payload(config) if args.dry_run_json else run_benchmark(config)
    except Exception as exc:
        payload = {"status": "error", "error": str(exc), "config": asdict(config)}
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2

    if args.compare_line and not (args.json or args.dry_run_json):
        print(format_compare_line(metrics))
    elif args.json or args.dry_run_json:
        print(json.dumps(metrics, indent=2, sort_keys=True))
    else:
        print_human(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

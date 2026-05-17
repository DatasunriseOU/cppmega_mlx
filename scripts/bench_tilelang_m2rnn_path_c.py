#!/usr/bin/env python3
"""Bench production-layout M2RNN Path B against TileLang Path C.

This is a focused profiling harness for the `local_gb10_quarter` R block shape:
mapped heads, packed recurrent input, and inline residual/gate post-processing.
It compares the full user-visible M2RNN output for:

* Path B: hand-written `mx.fast.metal_kernel` recurrent scan plus MLX post ops.
* Path C: TileLang -> TVM -> tvm-ffi generated fused `mapped_packed_post`.

The script also dumps the Path B source template we send to MLX and the lowered
Path C MSL kernels that TVM emits, so profiler numbers can be read together
with the generated code.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import statistics
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

import cppmega_mlx.nn._tilelang.m2rnn_path_c as m2rnn_path_c  # noqa: E402
from cppmega_mlx.nn._tilelang.m2rnn import (  # noqa: E402
    _BWD_KERNEL_SOURCE,
    _FWD_KERNEL_SOURCE,
    m2rnn_apply_with_state,
    m2rnn_fwd_metal,
)
from cppmega_mlx.nn._tilelang.m2rnn_path_c import (  # noqa: E402
    m2rnn_apply_mapped_packed_with_state_path_c,
    m2rnn_apply_mapped_packed_post_with_state_path_c,
    m2rnn_apply_post_residual_gate_path_c,
    m2rnn_mapped_packed_path_c_status,
    m2rnn_mapped_packed_post_path_c_status,
    m2rnn_post_residual_gate_path_c_status,
)


DTYPES = {
    "float32": mx.float32,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
}


def _safe_version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except Exception:
        return None


def _dtype_name(dtype: mx.Dtype) -> str:
    if dtype == mx.float32:
        return "float32"
    if dtype == mx.float16:
        return "float16"
    if dtype == mx.bfloat16:
        return "bfloat16"
    return str(dtype)


def _tl_dtype(dtype: mx.Dtype) -> str:
    name = _dtype_name(dtype)
    if name not in {"float32", "float16", "bfloat16"}:
        raise ValueError(f"unsupported dtype {dtype}")
    return name


def _make_inputs(
    *,
    batch: int,
    seq: int,
    total_heads: int,
    q_heads: int,
    k_heads: int,
    v_heads: int,
    g_heads: int,
    f_heads: int,
    w_heads: int,
    k_dim: int,
    v_dim: int,
    dtype: mx.Dtype,
    seed: int,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    mx.random.seed(seed)
    conv_dim = q_heads * k_dim + k_heads * k_dim + v_heads * v_dim
    projected_dim = conv_dim + f_heads + g_heads * v_dim

    q = (mx.random.normal((batch, seq, q_heads, k_dim)) * 0.1).astype(dtype)
    k = (mx.random.normal((batch, seq, k_heads, k_dim)) * 0.1).astype(dtype)
    v = (mx.random.normal((batch, seq, v_heads, v_dim)) * 0.1).astype(dtype)
    conv_input = mx.concatenate(
        [
            q.reshape(batch, seq, -1),
            k.reshape(batch, seq, -1),
            v.reshape(batch, seq, -1),
        ],
        axis=-1,
    )
    eye = mx.broadcast_to(mx.eye(v_dim, dtype=mx.float32)[None], (w_heads, v_dim, v_dim))
    W = (eye + mx.random.normal((w_heads, v_dim, v_dim)) * 0.03).astype(dtype)
    xf = mx.random.uniform(0.05, 0.95, (batch, seq, f_heads)).astype(dtype)
    h0 = mx.zeros((batch, total_heads, k_dim, v_dim), dtype=dtype)
    D = (mx.random.normal((total_heads, v_dim)) * 0.02).astype(dtype)
    projected = (mx.random.normal((batch, seq, projected_dim)) * 0.1).astype(dtype)
    mx.eval(conv_input, W, xf, h0, D, projected)
    return conv_input, W, xf, h0, D, projected


def _split_conv_input(
    conv_input: mx.array,
    *,
    batch: int,
    seq: int,
    q_heads: int,
    k_heads: int,
    v_heads: int,
    k_dim: int,
    v_dim: int,
) -> tuple[mx.array, mx.array, mx.array]:
    q_stop = q_heads * k_dim
    k_stop = q_stop + k_heads * k_dim
    q = conv_input[:, :, :q_stop].reshape(batch, seq, q_heads, k_dim)
    k = conv_input[:, :, q_stop:k_stop].reshape(batch, seq, k_heads, k_dim)
    v = conv_input[:, :, k_stop:].reshape(batch, seq, v_heads, v_dim)
    return q, k, v


def _expand_heads(x: mx.array, total_heads: int, axis: int) -> mx.array:
    heads = x.shape[axis]
    if heads == total_heads:
        return x
    if heads == 1:
        shape = list(x.shape)
        shape[axis] = total_heads
        return mx.broadcast_to(x, tuple(shape))
    return mx.repeat(x, repeats=total_heads // heads, axis=axis)


def _path_b_post(
    y: mx.array,
    conv_input: mx.array,
    D: mx.array,
    projected: mx.array,
    *,
    batch: int,
    seq: int,
    total_heads: int,
    q_heads: int,
    k_heads: int,
    v_heads: int,
    g_heads: int,
    k_dim: int,
    v_dim: int,
) -> mx.array:
    v_offset = q_heads * k_dim + k_heads * k_dim
    v_source = conv_input[:, :, v_offset:].reshape(batch, seq, v_heads, v_dim)
    v_broadcast = _expand_heads(v_source, total_heads, -2)
    skipped = y + v_broadcast * D.astype(y.dtype)
    g_dim = g_heads * v_dim
    g_flat = projected[:, :, -g_dim:]
    g_repeat = (
        g_flat
        if g_heads == total_heads
        else mx.repeat(g_flat, repeats=total_heads // g_heads, axis=-1)
    )
    return skipped.reshape(batch, seq, total_heads * v_dim) * nn.silu(g_repeat)


def _path_b_full(
    conv_input: mx.array,
    W: mx.array,
    xf: mx.array,
    h0: mx.array,
    D: mx.array,
    projected: mx.array,
    *,
    batch: int,
    seq: int,
    total_heads: int,
    q_heads: int,
    k_heads: int,
    v_heads: int,
    g_heads: int,
    k_dim: int,
    v_dim: int,
) -> tuple[mx.array, mx.array]:
    q, k, v = _split_conv_input(
        conv_input,
        batch=batch,
        seq=seq,
        q_heads=q_heads,
        k_heads=k_heads,
        v_heads=v_heads,
        k_dim=k_dim,
        v_dim=v_dim,
    )
    q_b = _expand_heads(q, total_heads, -2)
    k_b = _expand_heads(k, total_heads, -2)
    v_b = _expand_heads(v, total_heads, -2)
    W_b = _expand_heads(W, total_heads, 0)
    xf_b = _expand_heads(xf, total_heads, -1)
    y, h = m2rnn_apply_with_state(q_b, k_b, v_b, W_b, xf_b, h0)
    post = _path_b_post(
        y,
        conv_input,
        D,
        projected,
        batch=batch,
        seq=seq,
        total_heads=total_heads,
        q_heads=q_heads,
        k_heads=k_heads,
        v_heads=v_heads,
        g_heads=g_heads,
        k_dim=k_dim,
        v_dim=v_dim,
    )
    return post, h


def _path_b_fwd_only(
    conv_input: mx.array,
    W: mx.array,
    xf: mx.array,
    h0: mx.array,
    D: mx.array,
    projected: mx.array,
    *,
    batch: int,
    seq: int,
    total_heads: int,
    q_heads: int,
    k_heads: int,
    v_heads: int,
    g_heads: int,
    k_dim: int,
    v_dim: int,
) -> tuple[mx.array, mx.array]:
    q, k, v = _split_conv_input(
        conv_input,
        batch=batch,
        seq=seq,
        q_heads=q_heads,
        k_heads=k_heads,
        v_heads=v_heads,
        k_dim=k_dim,
        v_dim=v_dim,
    )
    y, h, _tanh_cache = m2rnn_fwd_metal(
        _expand_heads(q, total_heads, -2),
        _expand_heads(k, total_heads, -2),
        _expand_heads(v, total_heads, -2),
        _expand_heads(W, total_heads, 0),
        _expand_heads(xf, total_heads, -1),
        h0,
    )
    post = _path_b_post(
        y,
        conv_input,
        D,
        projected,
        batch=batch,
        seq=seq,
        total_heads=total_heads,
        q_heads=q_heads,
        k_heads=k_heads,
        v_heads=v_heads,
        g_heads=g_heads,
        k_dim=k_dim,
        v_dim=v_dim,
    )
    return post, h


def _path_c_full(
    conv_input: mx.array,
    W: mx.array,
    xf: mx.array,
    h0: mx.array,
    D: mx.array,
    projected: mx.array,
    *,
    q_heads: int,
    k_heads: int,
    v_heads: int,
    g_heads: int,
) -> tuple[mx.array, mx.array]:
    return m2rnn_apply_mapped_packed_post_with_state_path_c(
        conv_input,
        W,
        xf,
        h0,
        D,
        projected,
        q_heads=q_heads,
        k_heads=k_heads,
        v_heads=v_heads,
        g_heads=g_heads,
    )


def _path_c_split(
    conv_input: mx.array,
    W: mx.array,
    xf: mx.array,
    h0: mx.array,
    D: mx.array,
    projected: mx.array,
    *,
    q_heads: int,
    k_heads: int,
    v_heads: int,
    g_heads: int,
) -> tuple[mx.array, mx.array]:
    y, h = m2rnn_apply_mapped_packed_with_state_path_c(
        conv_input,
        W,
        xf,
        h0,
        q_heads=q_heads,
        k_heads=k_heads,
        v_heads=v_heads,
    )
    post = m2rnn_apply_post_residual_gate_path_c(
        y,
        conv_input,
        D,
        projected,
        q_heads=q_heads,
        k_heads=k_heads,
        v_heads=v_heads,
        g_heads=g_heads,
    )
    return post, h


def _eval_tree(value: Any) -> None:
    if isinstance(value, mx.array):
        mx.eval(value)
        return
    if isinstance(value, tuple) or isinstance(value, list):
        arrays = [x for x in _flatten(value) if isinstance(x, mx.array)]
        if arrays:
            mx.eval(*arrays)


def _flatten(value: Any) -> list[Any]:
    if isinstance(value, tuple) or isinstance(value, list):
        out: list[Any] = []
        for item in value:
            out.extend(_flatten(item))
        return out
    return [value]


def _bench(label: str, fn: Callable[[], Any], *, warmup: int, iters: int) -> dict[str, Any]:
    for _ in range(warmup):
        _eval_tree(fn())
        mx.synchronize()
    samples: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        _eval_tree(fn())
        mx.synchronize()
        samples.append(time.perf_counter() - start)
    return {
        "label": label,
        "mean_ms": statistics.mean(samples) * 1000.0,
        "median_ms": statistics.median(samples) * 1000.0,
        "min_ms": min(samples) * 1000.0,
        "max_ms": max(samples) * 1000.0,
        "iters": iters,
        "warmup": warmup,
    }


def _reset_peak_memory() -> None:
    fn = getattr(mx, "reset_peak_memory", None)
    if fn is None:
        fn = getattr(getattr(mx, "metal", None), "reset_peak_memory", None)
    if fn is not None:
        try:
            fn()
        except Exception:
            pass


def _peak_memory_mb() -> float | None:
    fn = getattr(mx, "get_peak_memory", None)
    if fn is None:
        fn = getattr(getattr(mx, "metal", None), "get_peak_memory", None)
    if fn is None:
        return None
    try:
        return float(fn()) / (1024.0 * 1024.0)
    except Exception:
        return None


def _max_abs(a: mx.array, b: mx.array) -> float:
    af = a.astype(mx.float32) if a.dtype != mx.float32 else a
    bf = b.astype(mx.float32) if b.dtype != mx.float32 else b
    return float(mx.max(mx.abs(af - bf)))


def _loss(out: tuple[mx.array, mx.array]) -> mx.array:
    y, _h = out
    yf = y.astype(mx.float32) if y.dtype != mx.float32 else y
    return mx.sum(yf * yf) * 0.5


def _source_metrics(source: str) -> dict[str, Any]:
    return {
        "line_count": len(source.splitlines()),
        "for_count": len(re.findall(r"\bfor\s*\(", source)),
        "threadgroup_barrier_count": source.count("threadgroup_barrier"),
        "simd_sum_count": source.count("simd_sum"),
        "atomic_add_count": source.count("atomic_add"),
        "exp_count": len(re.findall(r"\bexp\s*\(", source)),
        "tanh_count": len(re.findall(r"\btanh\s*\(", source)),
        "threadgroup_float_count": source.count("threadgroup float"),
        "device_buffer_count": len(re.findall(r"\bdevice\s+(?:const\s+)?[^,\n]+\\*", source)),
    }


def _dump_msl(
    path: Path,
    *,
    batch: int,
    seq: int,
    total_heads: int,
    q_heads: int,
    k_heads: int,
    v_heads: int,
    g_heads: int,
    f_heads: int,
    w_heads: int,
    k_dim: int,
    v_dim: int,
    projected_dim: int,
    carrier_dtype: str,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    conv_dim = q_heads * k_dim + k_heads * k_dim + v_heads * v_dim
    _fwd_kernel, fwd_lowering = m2rnn_path_c._mapped_packed_post_fwd_kernel_for(
        batch,
        seq,
        total_heads,
        q_heads,
        k_heads,
        v_heads,
        g_heads,
        w_heads,
        f_heads,
        k_dim,
        v_dim,
        projected_dim,
        carrier_dtype,
    )
    _recompute_post_bwd_kernel, recompute_post_bwd_lowering = (
        m2rnn_path_c._post_residual_gate_bwd_from_recurrence_kernel_for(
            batch,
            seq,
            total_heads,
            q_heads,
            k_heads,
            v_heads,
            g_heads,
            f_heads,
            k_dim,
            v_dim,
            conv_dim,
            projected_dim,
            carrier_dtype,
            "float32",
        )
    )
    _split_post_fwd_kernel, split_post_fwd_lowering = (
        m2rnn_path_c._post_residual_gate_fwd_kernel_for(
            batch,
            seq,
            total_heads,
            q_heads,
            k_heads,
            v_heads,
            g_heads,
            k_dim,
            v_dim,
            conv_dim,
            projected_dim,
            carrier_dtype,
        )
    )
    _split_post_bwd_kernel, split_post_bwd_lowering = (
        m2rnn_path_c._post_residual_gate_bwd_kernel_for(
            batch,
            seq,
            total_heads,
            q_heads,
            k_heads,
            v_heads,
            g_heads,
            k_dim,
            v_dim,
            conv_dim,
            projected_dim,
            carrier_dtype,
            "float32",
        )
    )
    _recurrent_bwd_kernel, recurrent_bwd_lowering = (
        m2rnn_path_c._mapped_packed_bwd_kernel_for(
            batch,
            seq,
            total_heads,
            q_heads,
            k_heads,
            v_heads,
            w_heads,
            f_heads,
            k_dim,
            v_dim,
            carrier_dtype,
            "float32",
            "float32",
        )
    )
    template = {
        "T_OUT": carrier_dtype,
        "BATCH": batch,
        "SEQ": seq,
        "HEADS": total_heads,
        "KDIM": k_dim,
        "VDIM": v_dim,
        "STATE_KV": k_dim * v_dim,
        "STATE_VV": v_dim * v_dim,
    }
    sections = {
        "path_b_hand_fwd_template": _FWD_KERNEL_SOURCE,
        "path_b_hand_bwd_template": _BWD_KERNEL_SOURCE,
        "path_c_generated_fused_fwd": fwd_lowering.msl_text,
        "path_c_generated_split_post_fwd": split_post_fwd_lowering.msl_text,
        "path_c_generated_split_post_bwd": split_post_bwd_lowering.msl_text,
        "path_c_generated_recompute_post_bwd_diagnostic": (
            recompute_post_bwd_lowering.msl_text
        ),
        "path_c_generated_recurrent_bwd": recurrent_bwd_lowering.msl_text,
    }
    text_parts = [
        "// === M2RNN Path B template sent to MLX and Path C generated MSL ===",
        "// Shape: "
        + json.dumps(
            {
                "batch": batch,
                "seq": seq,
                "total_heads": total_heads,
                "q_heads": q_heads,
                "k_heads": k_heads,
                "v_heads": v_heads,
                "g_heads": g_heads,
                "f_heads": f_heads,
                "w_heads": w_heads,
                "k_dim": k_dim,
                "v_dim": v_dim,
                "projected_dim": projected_dim,
                "carrier_dtype": carrier_dtype,
            },
            sort_keys=True,
        ),
        "// Path B MLX template values: " + json.dumps(template, sort_keys=True),
    ]
    for name, source in sections.items():
        text_parts.append(f"\n// ---- {name} ----\n")
        text_parts.append(source)
    path.write_text("\n".join(text_parts) + "\n", encoding="utf-8")
    return {
        "path": str(path),
        "lowerings": {
            "path_c_fused_fwd": {
                "grid": fwd_lowering.grid,
                "threadgroup": fwd_lowering.threadgroup,
            },
            "path_c_recompute_post_bwd_diagnostic": {
                "grid": recompute_post_bwd_lowering.grid,
                "threadgroup": recompute_post_bwd_lowering.threadgroup,
            },
            "path_c_split_post_fwd": {
                "grid": split_post_fwd_lowering.grid,
                "threadgroup": split_post_fwd_lowering.threadgroup,
            },
            "path_c_split_post_bwd": {
                "grid": split_post_bwd_lowering.grid,
                "threadgroup": split_post_bwd_lowering.threadgroup,
            },
            "path_c_recurrent_bwd": {
                "grid": recurrent_bwd_lowering.grid,
                "threadgroup": recurrent_bwd_lowering.threadgroup,
            },
        },
        "metrics": {name: _source_metrics(source) for name, source in sections.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--total-heads", type=int, default=4)
    parser.add_argument("--q-heads", type=int, default=1)
    parser.add_argument("--k-heads", type=int, default=1)
    parser.add_argument("--v-heads", type=int, default=2)
    parser.add_argument("--g-heads", type=int, default=4)
    parser.add_argument("--f-heads", type=int, default=2)
    parser.add_argument("--w-heads", type=int, default=1)
    parser.add_argument("--k-dim", type=int, default=64)
    parser.add_argument("--v-dim", type=int, default=16)
    parser.add_argument("--dtype", choices=DTYPES.keys(), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument(
        "--mode",
        choices=("both", "path_b", "path_c"),
        default="both",
        help="Run both paths for comparison, or isolate one path for xctrace.",
    )
    parser.add_argument(
        "--path-c-route",
        choices=("split", "fused"),
        default="split",
        help=(
            "Path C route to benchmark. split matches production nn/m2rnn.py; "
            "fused keeps the mapped_packed_post diagnostic route."
        ),
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "profiling" / "m2rnn_model_shape_path_b_vs_c.json",
    )
    parser.add_argument(
        "--msl-dump",
        type=Path,
        default=ROOT / "reports" / "profiling" / "m2rnn_model_shape_path_b_vs_c.metal",
    )
    args = parser.parse_args()

    dtype = DTYPES[args.dtype]
    carrier_dtype = _tl_dtype(dtype)
    batch = args.batch
    seq = args.seq
    total_heads = args.total_heads
    q_heads = args.q_heads
    k_heads = args.k_heads
    v_heads = args.v_heads
    g_heads = args.g_heads
    f_heads = args.f_heads
    w_heads = args.w_heads
    k_dim = args.k_dim
    v_dim = args.v_dim
    conv_dim = q_heads * k_dim + k_heads * k_dim + v_heads * v_dim
    projected_dim = conv_dim + f_heads + g_heads * v_dim

    inputs = _make_inputs(
        batch=batch,
        seq=seq,
        total_heads=total_heads,
        q_heads=q_heads,
        k_heads=k_heads,
        v_heads=v_heads,
        g_heads=g_heads,
        f_heads=f_heads,
        w_heads=w_heads,
        k_dim=k_dim,
        v_dim=v_dim,
        dtype=dtype,
        seed=args.seed,
    )
    conv_input, W, xf, h0, D, projected = inputs
    run_path_b = args.mode in {"both", "path_b"}
    run_path_c = args.mode in {"both", "path_c"}
    status = None
    if run_path_c:
        if args.path_c_route == "split":
            scan_status = m2rnn_mapped_packed_path_c_status(
                conv_input,
                W,
                xf,
                h0,
                q_heads=q_heads,
                k_heads=k_heads,
                v_heads=v_heads,
            )
            if scan_status.available:
                y_probe, _h_probe = m2rnn_apply_mapped_packed_with_state_path_c(
                    conv_input,
                    W,
                    xf,
                    h0,
                    q_heads=q_heads,
                    k_heads=k_heads,
                    v_heads=v_heads,
                )
                mx.eval(y_probe)
                post_status = m2rnn_post_residual_gate_path_c_status(
                    y_probe,
                    conv_input,
                    D,
                    projected,
                    q_heads=q_heads,
                    k_heads=k_heads,
                    v_heads=v_heads,
                    g_heads=g_heads,
                )
                status = post_status
            else:
                status = scan_status
        else:
            status = m2rnn_mapped_packed_post_path_c_status(
                conv_input,
                W,
                xf,
                h0,
                D,
                projected,
                q_heads=q_heads,
                k_heads=k_heads,
                v_heads=v_heads,
                g_heads=g_heads,
            )
        if not status.available:
            raise RuntimeError(f"M2RNN mapped packed post Path C unavailable: {status.reason}")

    b_kwargs = {
        "batch": batch,
        "seq": seq,
        "total_heads": total_heads,
        "q_heads": q_heads,
        "k_heads": k_heads,
        "v_heads": v_heads,
        "g_heads": g_heads,
        "k_dim": k_dim,
        "v_dim": v_dim,
    }
    c_kwargs = {
        "q_heads": q_heads,
        "k_heads": k_heads,
        "v_heads": v_heads,
        "g_heads": g_heads,
    }
    parity: dict[str, Any] = {}
    if run_path_b and run_path_c:
        post_b, h_b = _path_b_fwd_only(*inputs, **b_kwargs)
        path_c_fn = _path_c_split if args.path_c_route == "split" else _path_c_full
        post_c, h_c = path_c_fn(*inputs, **c_kwargs)
        mx.eval(post_b, h_b, post_c, h_c)
        parity = {
            "post_max_abs": _max_abs(post_b, post_c),
            "h_max_abs": _max_abs(h_b, h_c),
        }

    timings: dict[str, Any] = {}
    peak_memory: dict[str, float | None] = {}

    if run_path_b:
        _reset_peak_memory()
        timings["fwd_path_b"] = _bench(
            "fwd_path_b_full_post",
            lambda: _path_b_fwd_only(*inputs, **b_kwargs),
            warmup=args.warmup,
            iters=args.iters,
        )
        peak_memory["fwd_path_b"] = _peak_memory_mb()

    if run_path_c:
        path_c_fn = _path_c_split if args.path_c_route == "split" else _path_c_full
        _reset_peak_memory()
        timings["fwd_path_c"] = _bench(
            f"fwd_path_c_{args.path_c_route}",
            lambda: path_c_fn(*inputs, **c_kwargs),
            warmup=args.warmup,
            iters=args.iters,
        )
        peak_memory["fwd_path_c"] = _peak_memory_mb()

    def path_b_loss(
        conv_input_: mx.array,
        W_: mx.array,
        xf_: mx.array,
        h0_: mx.array,
        D_: mx.array,
        projected_: mx.array,
    ) -> mx.array:
        return _loss(_path_b_full(conv_input_, W_, xf_, h0_, D_, projected_, **b_kwargs))

    def path_c_loss(
        conv_input_: mx.array,
        W_: mx.array,
        xf_: mx.array,
        h0_: mx.array,
        D_: mx.array,
        projected_: mx.array,
    ) -> mx.array:
        path_c_fn = _path_c_split if args.path_c_route == "split" else _path_c_full
        return _loss(path_c_fn(conv_input_, W_, xf_, h0_, D_, projected_, **c_kwargs))

    if run_path_b:
        path_b_grad = mx.value_and_grad(path_b_loss, argnums=tuple(range(6)))
        _reset_peak_memory()
        timings["fwd_bwd_path_b"] = _bench(
            "fwd_bwd_path_b_full_post",
            lambda: path_b_grad(*inputs),
            warmup=args.warmup,
            iters=args.iters,
        )
        peak_memory["fwd_bwd_path_b"] = _peak_memory_mb()

    if run_path_c:
        path_c_grad = mx.value_and_grad(path_c_loss, argnums=tuple(range(6)))
        _reset_peak_memory()
        timings["fwd_bwd_path_c"] = _bench(
            f"fwd_bwd_path_c_{args.path_c_route}",
            lambda: path_c_grad(*inputs),
            warmup=args.warmup,
            iters=args.iters,
        )
        peak_memory["fwd_bwd_path_c"] = _peak_memory_mb()

    if run_path_c:
        msl = _dump_msl(
            args.msl_dump,
            batch=batch,
            seq=seq,
            total_heads=total_heads,
            q_heads=q_heads,
            k_heads=k_heads,
            v_heads=v_heads,
            g_heads=g_heads,
            f_heads=f_heads,
            w_heads=w_heads,
            k_dim=k_dim,
            v_dim=v_dim,
            projected_dim=projected_dim,
            carrier_dtype=carrier_dtype,
        )
    else:
        msl = {
            "path": str(args.msl_dump),
            "skipped": "Path C MSL dump skipped because --mode path_b was requested.",
            "metrics": {
                "path_b_hand_fwd_template": _source_metrics(_FWD_KERNEL_SOURCE),
                "path_b_hand_bwd_template": _source_metrics(_BWD_KERNEL_SOURCE),
            },
        }

    ratios: dict[str, float] = {}
    if run_path_b and run_path_c:
        ratios = {
            "fwd_median": timings["fwd_path_c"]["median_ms"]
            / timings["fwd_path_b"]["median_ms"],
            "fwd_bwd_median": timings["fwd_bwd_path_c"]["median_ms"]
            / timings["fwd_bwd_path_b"]["median_ms"],
        }
    receipt = {
        "schema_version": 1,
        "scope": "local_only",
        "kernel": "m2rnn_mapped_packed_post_path_c_vs_path_b",
        "hardware_label": platform.node() or "unknown",
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "mlx_version": _safe_version("mlx"),
            "tilelang_version": _safe_version("tilelang"),
        },
        "shape": {
            "batch": batch,
            "seq": seq,
            "total_heads": total_heads,
            "q_heads": q_heads,
            "k_heads": k_heads,
            "v_heads": v_heads,
            "g_heads": g_heads,
            "f_heads": f_heads,
            "w_heads": w_heads,
            "k_dim": k_dim,
            "v_dim": v_dim,
            "conv_dim": conv_dim,
            "projected_dim": projected_dim,
            "dtype": args.dtype,
        },
        "mode": args.mode,
        "path_c_route": args.path_c_route,
        "path_c_status": {
            "available": status.available if status is not None else None,
            "reason": status.reason if status is not None else "not probed in path_b-only mode",
        },
        "parity": parity,
        "timings": timings,
        "ratios_path_c_over_path_b": ratios,
        "peak_memory_mb": peak_memory,
        "msl": msl,
        "interpretation": {
            "path_b": "hand-written MSL recurrent scan plus MLX residual/gate post ops",
            "path_c": (
                "TileLang/TVM/tvm-ffi production split route"
                if args.path_c_route == "split"
                else "TileLang/TVM/tvm-ffi fused recurrent scan and residual/gate post ops"
            ),
            "matched_run_guard": "Compare only within this receipt: same process, same tensors, paired shape, same dtype and hardware.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

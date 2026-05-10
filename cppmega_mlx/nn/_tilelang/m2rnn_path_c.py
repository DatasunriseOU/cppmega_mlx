"""Path C TileLang DSL forward surface for cppmega M2RNN.

The Path B module owns the optimized hand-written MSL forward/backward pair.
This module supplies the missing Path C public apply surface by lowering a
TileLang ``@T.prim_func`` to Metal MSL and dispatching it through MLX. The
first implementation is intentionally forward-only: it computes the same
``y`` tensor as Path B, while public callers that need gradients should keep
using Path B until the backward DSL port lands.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast

import mlx.core as mx

from cppmega_mlx.nn._tilelang import _msl_transform
from cppmega_mlx.nn._tilelang._msl_transform import MSLDispatchUnsupported
from cppmega_mlx.nn._tilelang._msl_transform import lower_tilelang_to_msl_inline
from cppmega_mlx.nn._tilelang.m2rnn import (
    _validate_inputs,
    m2rnn_apply,
)


@dataclass(frozen=True)
class M2RNNPathCStatus:
    available: bool
    reason: str


def _threads_for(lanes: int) -> int:
    if lanes <= 0:
        return 1
    return min(32, lanes)


def _tl_dtype_for(dtype: mx.Dtype) -> str | None:
    if dtype == mx.float32:
        return "float32"
    if dtype == mx.float16:
        return "float16"
    return None


def _validate_same_dtype(reference: mx.array, *arrays: mx.array) -> bool:
    return all(x.dtype == reference.dtype for x in arrays)


@lru_cache(maxsize=128)
def _fwd_kernel_for(
    batch: int,
    seq: int,
    heads: int,
    k_dim: int,
    v_dim: int,
    carrier_dtype: str,
    *,
    return_msl: bool = False,
) -> tuple[Any, _msl_transform.TileLangMSLLowering]:
    import tilelang.language as T

    lanes = batch * heads * v_dim
    threads = _threads_for(lanes)
    accum_dtype = "float32"

    @T.prim_func
    def fwd(
        q: T.Tensor((batch, seq, heads, k_dim), carrier_dtype),
        k: T.Tensor((batch, seq, heads, k_dim), carrier_dtype),
        v: T.Tensor((batch, seq, heads, v_dim), carrier_dtype),
        W: T.Tensor((heads, v_dim, v_dim), carrier_dtype),
        xf: T.Tensor((batch, seq, heads), carrier_dtype),
        h0: T.Tensor((batch, heads, k_dim, v_dim), carrier_dtype),
        y: T.Tensor((batch, seq, heads, v_dim), carrier_dtype),
    ):
        with T.Kernel(T.ceildiv(lanes, threads), threads=threads) as bx:
            tid = T.get_thread_binding(0)
            lane = bx * threads + tid
            h_state = T.alloc_local((k_dim, v_dim), accum_dtype)
            h_next = T.alloc_local((k_dim, v_dim), accum_dtype)
            if lane < lanes:
                vv_out = lane % v_dim
                h = (lane // v_dim) % heads
                b = lane // (v_dim * heads)

                for kk in T.serial(k_dim):
                    for vv in T.serial(v_dim):
                        h_state[kk, vv] = T.cast(h0[b, h, kk, vv], accum_dtype)

                for t in T.serial(seq):
                    f_val = T.cast(xf[b, t, h], accum_dtype)
                    one_minus_f = 1.0 - f_val
                    y_acc = T.alloc_local((1,), accum_dtype)
                    y_acc[0] = 0.0

                    for kk in T.serial(k_dim):
                        k_val = T.cast(k[b, t, h, kk], accum_dtype)
                        q_val = T.cast(q[b, t, h, kk], accum_dtype)
                        for vv in T.serial(v_dim):
                            acc = T.alloc_local((1,), accum_dtype)
                            acc[0] = 0.0
                            for v0 in T.serial(v_dim):
                                acc[0] = acc[0] + h_state[kk, v0] * T.cast(
                                    W[h, v0, vv],
                                    accum_dtype,
                                )
                            z = acc[0] + k_val * T.cast(v[b, t, h, vv], accum_dtype)
                            tz = T.tanh(z)
                            h_next[kk, vv] = f_val * h_state[kk, vv] + one_minus_f * tz
                        y_acc[0] = y_acc[0] + q_val * h_next[kk, vv_out]

                    y[b, t, h, vv_out] = T.cast(y_acc[0], carrier_dtype)
                    for kk in T.serial(k_dim):
                        for vv in T.serial(v_dim):
                            h_state[kk, vv] = h_next[kk, vv]

    lowering = lower_tilelang_to_msl_inline(fwd)
    input_names = [name for name in lowering.buffer_param_names if name != "y"]
    if set(input_names) != {"q", "k", "v", "W", "xf", "h0"}:
        raise MSLDispatchUnsupported(
            "unexpected M2RNN Path C buffer signature: "
            + ", ".join(lowering.buffer_param_names)
        )
    kernel = mx.fast.metal_kernel(
        name=f"cppmega_m2rnn_path_c_fwd_{carrier_dtype}_{batch}_{seq}_{heads}_{k_dim}_{v_dim}",
        input_names=input_names,
        output_names=["y"],
        source=lowering.body,
        header=lowering.header,
        ensure_row_contiguous=True,
    )
    return kernel, lowering


def m2rnn_path_c_status() -> M2RNNPathCStatus:
    if not _msl_transform.can_run_metal():
        return M2RNNPathCStatus(False, "MLX Metal backend is not available")
    try:
        kernel, lowering = _fwd_kernel_for(1, 4, 2, 4, 4, "float32")
        del kernel
        source = lowering.msl_text
    except Exception as exc:
        return M2RNNPathCStatus(
            False,
            f"TileLang/MLX lowering failed for M2RNN Path C forward: {type(exc).__name__}: {exc}",
        )
    if "kernel void" not in source:
        return M2RNNPathCStatus(False, "lowered M2RNN Path C source has no kernel")
    return M2RNNPathCStatus(True, "M2RNN TileLang DSL Path C forward is dispatchable")


def m2rnn_fwd_path_c(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    W: mx.array,
    xf: mx.array,
    h0: mx.array | None = None,
) -> mx.array | None:
    if not _msl_transform.can_run_metal():
        return None
    if h0 is None:
        return None
    if not _validate_same_dtype(q, k, v, W, xf, h0):
        return None
    carrier_dtype = _tl_dtype_for(q.dtype)
    if carrier_dtype is None:
        return None
    batch, seq, heads, k_dim, v_dim = _validate_inputs(q, k, v, W, xf, h0)
    if seq == 0:
        return mx.zeros((batch, 0, heads, v_dim), dtype=q.dtype)
    try:
        kernel, lowering = _fwd_kernel_for(batch, seq, heads, k_dim, v_dim, carrier_dtype)
    except Exception:
        return None

    input_map = {
        "q": q,
        "k": k,
        "v": v,
        "W": W,
        "xf": xf,
        "h0": h0,
    }
    outputs = _msl_transform.dispatch(
        cast(_msl_transform.MetalKernel, kernel),
        inputs=[input_map[name] for name in lowering.buffer_param_names if name != "y"],
        output_shapes=[(batch, seq, heads, v_dim)],
        output_dtypes=[q.dtype],
        lowering=lowering,
    )
    return outputs[0]


def m2rnn_apply_path_c(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    W: mx.array,
    xf: mx.array,
    h0: mx.array | None = None,
    *,
    force_path_c: bool = False,
) -> mx.array:
    out = m2rnn_fwd_path_c(q, k, v, W, xf, h0)
    if out is not None:
        return out
    if force_path_c:
        raise RuntimeError(f"m2rnn_apply_path_c unavailable: {m2rnn_path_c_status().reason}")
    if h0 is None:
        raise RuntimeError(
            "m2rnn_apply_path_c fallback requires an existing h0 tensor; "
            "the adapter will not allocate one implicitly"
        )
    return m2rnn_apply(q, k, v, W, xf, h0)


__all__ = [
    "M2RNNPathCStatus",
    "m2rnn_apply_path_c",
    "m2rnn_fwd_path_c",
    "m2rnn_path_c_status",
]

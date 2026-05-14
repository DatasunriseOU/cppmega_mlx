"""Path C TileLang DSL forward/backward surface for cppmega M2RNN.

The Path B module owns the optimized hand-written MSL forward/backward pair.
This module supplies the Path C public apply surface by lowering TileLang
``@T.prim_func`` kernels to Metal MSL and dispatching them through MLX. It keeps
the same explicit tensor contract as Path B: callers provide ``h0`` up front,
forward returns TileLang-owned outputs, and backward uses explicit partial
output buffers for reductions/scratch instead of CPU staging.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast

import mlx.core as mx

from cppmega_mlx.nn._tilelang import _msl_transform
from cppmega_mlx.nn._tilelang._engine_dispatch import dispatch_lower
from cppmega_mlx.nn._tilelang._msl_transform import MSLDispatchUnsupported
from cppmega_mlx.nn._tilelang.m2rnn import (
    _validate_inputs,
)


@dataclass(frozen=True)
class M2RNNPathCStatus:
    available: bool
    reason: str


def _threads_for(lanes: int) -> int:
    if lanes <= 0:
        return 1
    return min(32, lanes)


def _threadgroup_threads_for(*lanes: int) -> int:
    target = max(1, *lanes)
    threads = 1
    while threads < target:
        threads <<= 1
    return min(256, threads)


def _tl_dtype_for(dtype: mx.Dtype) -> str | None:
    if dtype == mx.float32:
        return "float32"
    if dtype == mx.float16:
        return "float16"
    if dtype == mx.bfloat16:
        return "bfloat16"
    return None


def _validate_same_dtype(reference: mx.array, *arrays: mx.array) -> bool:
    return all(x.dtype == reference.dtype for x in arrays)


def _path_c_inputs_eligible(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    W: mx.array,
    xf: mx.array,
    h0: mx.array | None,
) -> bool:
    if not _msl_transform.can_run_metal() or h0 is None:
        return False
    if not _validate_same_dtype(q, k, v, W, xf, h0):
        return False
    if _tl_dtype_for(q.dtype) is None:
        return False
    try:
        _validate_inputs(q, k, v, W, xf, h0)
    except (TypeError, ValueError):
        return False
    return True


_FWD_OUTPUT_NAMES = ("h_last", "tanh_cache", "y")
_FWD_OUTPUT_IDX = (6, 7, 8)
_BWD_OUTPUT_NAMES = (
    "dW_partial",
    "dh0",
    "dk",
    "dq",
    "dv",
    "dxf",
    "h_steps_scratch",
)
_BWD_OUTPUT_IDX = (8, 9, 10, 11, 12, 13, 14)
_PACKED_FWD_OUTPUT_IDX = (4, 5, 6)
_PACKED_POST_FWD_OUTPUT_NAMES = ("h_last", "tanh_cache", "post")
_PACKED_POST_FWD_OUTPUT_IDX = (6, 7, 8)
_PACKED_BWD_OUTPUT_NAMES = (
    "dconv_input",
    "dW_partial",
    "dxf",
    "dh0",
    "h_steps_scratch",
)
_PACKED_BWD_OUTPUT_IDX = (6, 7, 8, 9, 10)
_POST_FWD_OUTPUT_NAMES = ("post",)
_POST_FWD_OUTPUT_IDX = (4,)
_POST_BWD_OUTPUT_NAMES = ("dy_recurrent", "dconv_input", "dD", "dprojected")
_POST_BWD_OUTPUT_IDX = (5, 6, 7, 8)
_POST_RECOMPUTE_BWD_OUTPUT_IDX = (7, 8, 9, 10)
_PACKED_FWD_K_PARALLEL_MIN_K = 4
_PACKED_BWD_K_PARALLEL_MIN_K = 4

M2RNNFwdOwnerOutputs = tuple[mx.array, mx.array, mx.array]
M2RNNBwdOwnerOutputs = tuple[
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
]
M2RNNPackedBwdOwnerOutputs = tuple[mx.array, mx.array, mx.array, mx.array, mx.array]
M2RNNPackedPostFwdOutputs = tuple[mx.array, mx.array, mx.array]
M2RNNPostBwdOwnerOutputs = tuple[mx.array, mx.array, mx.array, mx.array]


def _require_owner_array(
    op_name: str,
    name: str,
    array: mx.array,
    *,
    shape: tuple[int, ...],
    dtype: mx.Dtype,
) -> mx.array:
    if not isinstance(array, mx.array):
        raise TypeError(
            f"{op_name}: owner output {name} must be an mlx.core.array; "
            f"got {type(array).__name__}"
        )
    if tuple(array.shape) != shape:
        raise ValueError(
            f"{op_name}: owner output {name} must have shape {shape}; "
            f"got {tuple(array.shape)}"
        )
    if array.dtype != dtype:
        raise TypeError(
            f"{op_name}: owner output {name} must have dtype {dtype}; got {array.dtype}"
        )
    return array


def _packed_path_c_inputs_eligible(
    conv_input: mx.array,
    W: mx.array,
    xf: mx.array,
    h0: mx.array | None,
) -> bool:
    return m2rnn_packed_path_c_status(
        conv_input,
        W,
        xf,
        h0,
        require_backward=False,
    ).available


def _packed_path_c_inputs_well_formed(
    conv_input: mx.array,
    W: mx.array,
    xf: mx.array,
    h0: mx.array | None,
) -> bool:
    if not _msl_transform.can_run_metal() or h0 is None:
        return False
    if not _validate_same_dtype(conv_input, W, xf, h0):
        return False
    if _tl_dtype_for(conv_input.dtype) is None:
        return False
    try:
        if conv_input.ndim != 3 or W.ndim != 3 or xf.ndim != 3 or h0.ndim != 4:
            return False
        batch, seq, conv_dim = conv_input.shape
        heads, v_dim, w_v_dim = W.shape
        if v_dim != w_v_dim:
            return False
        if h0.shape[0] != batch or h0.shape[1] != heads or h0.shape[3] != v_dim:
            return False
        if xf.shape != (batch, seq, heads):
            return False
        k_dim = h0.shape[2]
        return conv_dim == heads * (2 * k_dim + v_dim)
    except (TypeError, ValueError):
        return False


def _kernel_lowering_status(
    label: str,
    kernel_factory: Any,
    *args: Any,
) -> M2RNNPathCStatus:
    try:
        _kernel, lowering = kernel_factory(*args)
    except Exception as exc:
        return M2RNNPathCStatus(
            False,
            f"{label} lowering failed: {type(exc).__name__}: {exc}",
        )
    if "kernel void" not in lowering.msl_text:
        return M2RNNPathCStatus(False, f"{label} source has no kernel")
    return M2RNNPathCStatus(True, f"{label} lowers to a Metal kernel")


def _m2rnn_fwd_owner_outputs(
    out: M2RNNFwdOwnerOutputs | None,
    *,
    batch: int,
    seq: int,
    heads: int,
    k_dim: int,
    v_dim: int,
    dtype: mx.Dtype,
) -> M2RNNFwdOwnerOutputs:
    op_name = "m2rnn_fwd_path_c"
    if out is None:
        return (
            mx.zeros((batch, seq, heads, v_dim), dtype=dtype),
            mx.zeros((batch, heads, k_dim, v_dim), dtype=dtype),
            mx.zeros((batch, seq, heads, k_dim, v_dim), dtype=dtype),
        )
    if not isinstance(out, tuple) or len(out) != 3:
        raise TypeError(
            f"{op_name}: out must be a (y, h_last, tanh_cache) owner-output tuple"
        )
    y, h_last, tanh_cache = out
    return (
        _require_owner_array(
            op_name,
            "y",
            y,
            shape=(batch, seq, heads, v_dim),
            dtype=dtype,
        ),
        _require_owner_array(
            op_name,
            "h_last",
            h_last,
            shape=(batch, heads, k_dim, v_dim),
            dtype=dtype,
        ),
        _require_owner_array(
            op_name,
            "tanh_cache",
            tanh_cache,
            shape=(batch, seq, heads, k_dim, v_dim),
            dtype=dtype,
        ),
    )


def _m2rnn_bwd_owner_outputs(
    out: M2RNNBwdOwnerOutputs | None,
    *,
    batch: int,
    seq: int,
    heads: int,
    k_dim: int,
    v_dim: int,
    dtype: mx.Dtype,
) -> M2RNNBwdOwnerOutputs:
    op_name = "m2rnn_bwd_path_c"
    if out is None:
        return (
            mx.zeros((batch, seq, heads, k_dim), dtype=dtype),
            mx.zeros((batch, seq, heads, k_dim), dtype=dtype),
            mx.zeros((batch, seq, heads, v_dim), dtype=dtype),
            mx.zeros((batch, heads, v_dim, v_dim), dtype=dtype),
            mx.zeros((batch, seq, heads), dtype=dtype),
            mx.zeros((batch, heads, k_dim, v_dim), dtype=dtype),
            mx.zeros((batch, heads, seq, k_dim, v_dim), dtype=dtype),
        )
    if not isinstance(out, tuple) or len(out) != 7:
        raise TypeError(
            "m2rnn_bwd_path_c: out must be a "
            "(dq, dk, dv, dW_partial, dxf, dh0, h_steps_scratch) "
            "owner-output tuple"
        )
    names = ("dq", "dk", "dv", "dW_partial", "dxf", "dh0", "h_steps_scratch")
    expected_shapes = (
        (batch, seq, heads, k_dim),
        (batch, seq, heads, k_dim),
        (batch, seq, heads, v_dim),
        (batch, heads, v_dim, v_dim),
        (batch, seq, heads),
        (batch, heads, k_dim, v_dim),
        (batch, heads, seq, k_dim, v_dim),
    )
    return cast(
        M2RNNBwdOwnerOutputs,
        tuple(
            _require_owner_array(op_name, name, array, shape=shape, dtype=dtype)
            for name, array, shape in zip(names, out, expected_shapes, strict=True)
        ),
    )


def _m2rnn_packed_bwd_owner_outputs(
    out: M2RNNPackedBwdOwnerOutputs | None,
    *,
    batch: int,
    seq: int,
    heads: int,
    conv_dim: int,
    k_dim: int,
    v_dim: int,
    dtype: mx.Dtype,
) -> M2RNNPackedBwdOwnerOutputs:
    op_name = "m2rnn_packed_bwd_path_c"
    if out is None:
        return (
            mx.zeros((batch, seq, conv_dim), dtype=dtype),
            mx.zeros((batch, heads, v_dim, v_dim), dtype=dtype),
            mx.zeros((batch, seq, heads), dtype=dtype),
            mx.zeros((batch, heads, k_dim, v_dim), dtype=dtype),
            mx.zeros((batch, heads, seq, k_dim, v_dim), dtype=dtype),
        )
    if not isinstance(out, tuple) or len(out) != 5:
        raise TypeError(
            "m2rnn_packed_bwd_path_c: out must be a "
            "(dconv_input, dW_partial, dxf, dh0, h_steps_scratch) "
            "owner-output tuple"
        )
    names = ("dconv_input", "dW_partial", "dxf", "dh0", "h_steps_scratch")
    expected_shapes = (
        (batch, seq, conv_dim),
        (batch, heads, v_dim, v_dim),
        (batch, seq, heads),
        (batch, heads, k_dim, v_dim),
        (batch, heads, seq, k_dim, v_dim),
    )
    return cast(
        M2RNNPackedBwdOwnerOutputs,
        tuple(
            _require_owner_array(op_name, name, array, shape=shape, dtype=dtype)
            for name, array, shape in zip(names, out, expected_shapes, strict=True)
        ),
    )


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
        h_last: T.Tensor((batch, heads, k_dim, v_dim), carrier_dtype),
        tanh_cache: T.Tensor((batch, seq, heads, k_dim, v_dim), carrier_dtype),
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
                            if vv_out == 0:
                                tanh_cache[b, t, h, kk, vv] = T.cast(
                                    tz,
                                    carrier_dtype,
                                )
                            h_next[kk, vv] = f_val * h_state[kk, vv] + one_minus_f * tz
                        y_acc[0] = y_acc[0] + q_val * h_next[kk, vv_out]

                    y[b, t, h, vv_out] = T.cast(y_acc[0], carrier_dtype)
                    for kk in T.serial(k_dim):
                        for vv in T.serial(v_dim):
                            h_state[kk, vv] = h_next[kk, vv]
                if vv_out == 0:
                    for kk in T.serial(k_dim):
                        for vv in T.serial(v_dim):
                            h_last[b, h, kk, vv] = T.cast(
                                h_state[kk, vv],
                                carrier_dtype,
                            )

    lowering = dispatch_lower(fwd, target="metal", return_msl=True)
    input_names = [
        name for name in lowering.buffer_param_names if name not in _FWD_OUTPUT_NAMES
    ]
    if set(input_names) != {"q", "k", "v", "W", "xf", "h0"}:
        raise MSLDispatchUnsupported(
            "unexpected M2RNN Path C buffer signature: "
            + ", ".join(lowering.buffer_param_names)
        )
    import tilelang

    kernel = tilelang.compile(
        fwd,
        target=_msl_transform._as_metal_target("metal"),
        execution_backend="tvm_ffi",
        out_idx=list(_FWD_OUTPUT_IDX),
    )
    return kernel, lowering


@lru_cache(maxsize=128)
def _bwd_kernel_for(
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

    lanes = batch * heads
    threads = _threads_for(lanes)
    accum_dtype = "float32"

    @T.prim_func
    def bwd(
        dy: T.Tensor((batch, seq, heads, v_dim), carrier_dtype),
        q: T.Tensor((batch, seq, heads, k_dim), carrier_dtype),
        k: T.Tensor((batch, seq, heads, k_dim), carrier_dtype),
        v: T.Tensor((batch, seq, heads, v_dim), carrier_dtype),
        W: T.Tensor((heads, v_dim, v_dim), carrier_dtype),
        xf: T.Tensor((batch, seq, heads), carrier_dtype),
        h0: T.Tensor((batch, heads, k_dim, v_dim), carrier_dtype),
        tanh_cache: T.Tensor((batch, seq, heads, k_dim, v_dim), carrier_dtype),
        dW_partial: T.Tensor((batch, heads, v_dim, v_dim), carrier_dtype),
        dh0: T.Tensor((batch, heads, k_dim, v_dim), carrier_dtype),
        dk: T.Tensor((batch, seq, heads, k_dim), carrier_dtype),
        dq: T.Tensor((batch, seq, heads, k_dim), carrier_dtype),
        dv: T.Tensor((batch, seq, heads, v_dim), carrier_dtype),
        dxf: T.Tensor((batch, seq, heads), carrier_dtype),
        h_steps_scratch: T.Tensor((batch, heads, seq, k_dim, v_dim), carrier_dtype),
    ):
        with T.Kernel(T.ceildiv(lanes, threads), threads=threads) as bx:
            tid = T.get_thread_binding(0)
            lane = bx * threads + tid
            h_state = T.alloc_local((k_dim, v_dim), accum_dtype)
            dh = T.alloc_local((k_dim, v_dim), accum_dtype)
            dz = T.alloc_local((k_dim, v_dim), accum_dtype)
            dh_next = T.alloc_local((k_dim, v_dim), accum_dtype)
            dW_acc = T.alloc_local((v_dim, v_dim), accum_dtype)
            if lane < lanes:
                h = lane % heads
                b = lane // heads

                for kk in T.serial(k_dim):
                    for vv in T.serial(v_dim):
                        h_state[kk, vv] = T.cast(h0[b, h, kk, vv], accum_dtype)
                        dh[kk, vv] = 0.0
                for v0 in T.serial(v_dim):
                    for vv in T.serial(v_dim):
                        dW_acc[v0, vv] = 0.0

                for t in T.serial(seq):
                    f_val = T.cast(xf[b, t, h], accum_dtype)
                    one_minus_f = 1.0 - f_val
                    for kk in T.serial(k_dim):
                        for vv in T.serial(v_dim):
                            h_steps_scratch[b, h, t, kk, vv] = T.cast(
                                h_state[kk, vv],
                                carrier_dtype,
                            )
                            tz = T.cast(tanh_cache[b, t, h, kk, vv], accum_dtype)
                            h_state[kk, vv] = (
                                f_val * h_state[kk, vv] + one_minus_f * tz
                            )

                for r in T.serial(seq):
                    t = seq - 1 - r
                    f_val = T.cast(xf[b, t, h], accum_dtype)
                    one_minus_f = 1.0 - f_val

                    for kk in T.serial(k_dim):
                        q_val = T.cast(q[b, t, h, kk], accum_dtype)
                        dq_acc = T.alloc_local((1,), accum_dtype)
                        dq_acc[0] = 0.0
                        for vv in T.serial(v_dim):
                            dY = T.cast(dy[b, t, h, vv], accum_dtype)
                            h_prev = T.cast(
                                h_steps_scratch[b, h, t, kk, vv],
                                accum_dtype,
                            )
                            tz = T.cast(
                                tanh_cache[b, t, h, kk, vv],
                                accum_dtype,
                            )
                            h_t = f_val * h_prev + one_minus_f * tz
                            dq_acc[0] = dq_acc[0] + dY * h_t
                            dh[kk, vv] = dh[kk, vv] + q_val * dY
                        dq[b, t, h, kk] = T.cast(dq_acc[0], carrier_dtype)

                    df_acc = T.alloc_local((1,), accum_dtype)
                    df_acc[0] = 0.0
                    for kk in T.serial(k_dim):
                        for vv in T.serial(v_dim):
                            h_prev = T.cast(
                                h_steps_scratch[b, h, t, kk, vv],
                                accum_dtype,
                            )
                            tz = T.cast(
                                tanh_cache[b, t, h, kk, vv],
                                accum_dtype,
                            )
                            dh_kv = dh[kk, vv]
                            df_acc[0] = df_acc[0] + dh_kv * (h_prev - tz)
                            dz[kk, vv] = (
                                one_minus_f * dh_kv * (1.0 - tz * tz)
                            )
                    dxf[b, t, h] = T.cast(df_acc[0], carrier_dtype)

                    for kk in T.serial(k_dim):
                        dk_acc = T.alloc_local((1,), accum_dtype)
                        dk_acc[0] = 0.0
                        for vv in T.serial(v_dim):
                            dk_acc[0] = dk_acc[0] + dz[kk, vv] * T.cast(
                                v[b, t, h, vv],
                                accum_dtype,
                            )
                        dk[b, t, h, kk] = T.cast(dk_acc[0], carrier_dtype)

                    for vv in T.serial(v_dim):
                        dv_acc = T.alloc_local((1,), accum_dtype)
                        dv_acc[0] = 0.0
                        for kk in T.serial(k_dim):
                            dv_acc[0] = dv_acc[0] + dz[kk, vv] * T.cast(
                                k[b, t, h, kk],
                                accum_dtype,
                            )
                        dv[b, t, h, vv] = T.cast(dv_acc[0], carrier_dtype)

                    for v0 in T.serial(v_dim):
                        for vv in T.serial(v_dim):
                            w_acc = T.alloc_local((1,), accum_dtype)
                            w_acc[0] = 0.0
                            for kk in T.serial(k_dim):
                                h_prev = T.cast(
                                    h_steps_scratch[b, h, t, kk, v0],
                                    accum_dtype,
                                )
                                w_acc[0] = w_acc[0] + h_prev * dz[kk, vv]
                            dW_acc[v0, vv] = dW_acc[v0, vv] + w_acc[0]

                    for kk in T.serial(k_dim):
                        for v_in in T.serial(v_dim):
                            acc = T.alloc_local((1,), accum_dtype)
                            acc[0] = f_val * dh[kk, v_in]
                            for v_out in T.serial(v_dim):
                                acc[0] = acc[0] + dz[kk, v_out] * T.cast(
                                    W[h, v_in, v_out],
                                    accum_dtype,
                                )
                            dh_next[kk, v_in] = acc[0]
                    for kk in T.serial(k_dim):
                        for vv in T.serial(v_dim):
                            dh[kk, vv] = dh_next[kk, vv]

                for kk in T.serial(k_dim):
                    for vv in T.serial(v_dim):
                        dh0[b, h, kk, vv] = T.cast(dh[kk, vv], carrier_dtype)
                for v0 in T.serial(v_dim):
                    for vv in T.serial(v_dim):
                        dW_partial[b, h, v0, vv] = T.cast(
                            dW_acc[v0, vv],
                            carrier_dtype,
                        )

    lowering = dispatch_lower(bwd, target="metal", return_msl=True)
    input_names = [
        name for name in lowering.buffer_param_names if name not in _BWD_OUTPUT_NAMES
    ]
    if set(input_names) != {"dy", "q", "k", "v", "W", "xf", "h0", "tanh_cache"}:
        raise MSLDispatchUnsupported(
            "unexpected M2RNN Path C bwd buffer signature: "
            + ", ".join(lowering.buffer_param_names)
        )
    import tilelang

    kernel = tilelang.compile(
        bwd,
        target=_msl_transform._as_metal_target("metal"),
        execution_backend="tvm_ffi",
        out_idx=list(_BWD_OUTPUT_IDX),
    )
    return kernel, lowering


@lru_cache(maxsize=128)
def _packed_fwd_k_parallel_kernel_for(
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

    del return_msl
    conv_dim = heads * (2 * k_dim + v_dim)
    k_offset = heads * k_dim
    v_offset = k_offset + heads * k_dim
    groups = batch * heads
    threads = _threadgroup_threads_for(k_dim, v_dim)
    if threads < k_dim or threads < v_dim:
        raise MSLDispatchUnsupported(
            f"packed M2RNN k-parallel Path C needs one thread per K/V lane; "
            f"got K={k_dim}, V={v_dim}, threads={threads}"
        )
    accum_dtype = "float32"

    @T.prim_func
    def fwd(
        conv_input: T.Tensor((batch, seq, conv_dim), carrier_dtype),
        W: T.Tensor((heads, v_dim, v_dim), carrier_dtype),
        xf: T.Tensor((batch, seq, heads), carrier_dtype),
        h0: T.Tensor((batch, heads, k_dim, v_dim), carrier_dtype),
        h_last: T.Tensor((batch, heads, k_dim, v_dim), carrier_dtype),
        tanh_cache: T.Tensor((batch, seq, heads, k_dim, v_dim), carrier_dtype),
        y: T.Tensor((batch, seq, heads, v_dim), carrier_dtype),
    ):
        with T.Kernel(groups, threads=threads) as group_id:
            tid = T.get_thread_binding(0)
            h = group_id % heads
            b = group_id // heads
            q_head_offset = h * k_dim
            k_head_offset = k_offset + h * k_dim
            v_head_offset = v_offset + h * v_dim
            h_row = T.alloc_local((v_dim,), accum_dtype)
            h_next = T.alloc_local((v_dim,), accum_dtype)
            w_shared = T.alloc_shared((v_dim, v_dim), accum_dtype, scope="shared")
            y_shared = T.alloc_shared((k_dim, v_dim), accum_dtype, scope="shared")

            for i in T.serial(tid, v_dim * v_dim, step=threads):
                w_shared[i // v_dim, i % v_dim] = T.cast(
                    W[h, i // v_dim, i % v_dim],
                    accum_dtype,
                )
            T.sync_threads()

            if tid < k_dim:
                for vv in T.serial(v_dim):
                    h_row[vv] = T.cast(h0[b, h, tid, vv], accum_dtype)

            for t in T.serial(seq):
                if tid < k_dim:
                    f_val = T.cast(xf[b, t, h], accum_dtype)
                    one_minus_f = 1.0 - f_val
                    k_val = T.cast(
                        conv_input[b, t, k_head_offset + tid],
                        accum_dtype,
                    )
                    q_val = T.cast(
                        conv_input[b, t, q_head_offset + tid],
                        accum_dtype,
                    )
                    for vv in T.serial(v_dim):
                        acc = T.alloc_local((1,), accum_dtype)
                        acc[0] = 0.0
                        for v0 in T.serial(v_dim):
                            acc[0] = acc[0] + h_row[v0] * w_shared[v0, vv]
                        z = acc[0] + k_val * T.cast(
                            conv_input[b, t, v_head_offset + vv],
                            accum_dtype,
                        )
                        tz = T.tanh(z)
                        tanh_cache[b, t, h, tid, vv] = T.cast(tz, carrier_dtype)
                        h_new = f_val * h_row[vv] + one_minus_f * tz
                        h_next[vv] = h_new
                        y_shared[tid, vv] = q_val * h_new
                    for vv in T.serial(v_dim):
                        h_row[vv] = h_next[vv]
                T.sync_threads()

                if tid < v_dim:
                    y_acc = T.alloc_local((1,), accum_dtype)
                    y_acc[0] = 0.0
                    for kk in T.serial(k_dim):
                        y_acc[0] = y_acc[0] + y_shared[kk, tid]
                    y[b, t, h, tid] = T.cast(y_acc[0], carrier_dtype)
                T.sync_threads()

            if tid < k_dim:
                for vv in T.serial(v_dim):
                    h_last[b, h, tid, vv] = T.cast(h_row[vv], carrier_dtype)

    lowering = dispatch_lower(fwd, target="metal", return_msl=True)
    input_names = [
        name for name in lowering.buffer_param_names if name not in _FWD_OUTPUT_NAMES
    ]
    if set(input_names) != {"conv_input", "W", "xf", "h0"}:
        raise MSLDispatchUnsupported(
            "unexpected packed M2RNN Path C k-parallel buffer signature: "
            + ", ".join(lowering.buffer_param_names)
        )
    import tilelang

    kernel = tilelang.compile(
        fwd,
        target=_msl_transform._as_metal_target("metal"),
        execution_backend="tvm_ffi",
        out_idx=list(_PACKED_FWD_OUTPUT_IDX),
    )
    return kernel, lowering


@lru_cache(maxsize=128)
def _packed_fwd_kernel_for(
    batch: int,
    seq: int,
    heads: int,
    k_dim: int,
    v_dim: int,
    carrier_dtype: str,
    *,
    return_msl: bool = False,
) -> tuple[Any, _msl_transform.TileLangMSLLowering]:
    if k_dim >= _PACKED_FWD_K_PARALLEL_MIN_K and max(k_dim, v_dim) <= 256:
        return _packed_fwd_k_parallel_kernel_for(
            batch,
            seq,
            heads,
            k_dim,
            v_dim,
            carrier_dtype,
            return_msl=return_msl,
        )

    import tilelang.language as T

    del return_msl
    conv_dim = heads * (2 * k_dim + v_dim)
    k_offset = heads * k_dim
    v_offset = k_offset + heads * k_dim
    lanes = batch * heads * v_dim
    threads = _threads_for(lanes)
    accum_dtype = "float32"

    @T.prim_func
    def fwd(
        conv_input: T.Tensor((batch, seq, conv_dim), carrier_dtype),
        W: T.Tensor((heads, v_dim, v_dim), carrier_dtype),
        xf: T.Tensor((batch, seq, heads), carrier_dtype),
        h0: T.Tensor((batch, heads, k_dim, v_dim), carrier_dtype),
        h_last: T.Tensor((batch, heads, k_dim, v_dim), carrier_dtype),
        tanh_cache: T.Tensor((batch, seq, heads, k_dim, v_dim), carrier_dtype),
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
                q_head_offset = h * k_dim
                k_head_offset = k_offset + h * k_dim
                v_head_offset = v_offset + h * v_dim

                for kk in T.serial(k_dim):
                    for vv in T.serial(v_dim):
                        h_state[kk, vv] = T.cast(h0[b, h, kk, vv], accum_dtype)

                for t in T.serial(seq):
                    f_val = T.cast(xf[b, t, h], accum_dtype)
                    one_minus_f = 1.0 - f_val
                    y_acc = T.alloc_local((1,), accum_dtype)
                    y_acc[0] = 0.0

                    for kk in T.serial(k_dim):
                        k_val = T.cast(
                            conv_input[b, t, k_head_offset + kk],
                            accum_dtype,
                        )
                        q_val = T.cast(
                            conv_input[b, t, q_head_offset + kk],
                            accum_dtype,
                        )
                        for vv in T.serial(v_dim):
                            acc = T.alloc_local((1,), accum_dtype)
                            acc[0] = 0.0
                            for v0 in T.serial(v_dim):
                                acc[0] = acc[0] + h_state[kk, v0] * T.cast(
                                    W[h, v0, vv],
                                    accum_dtype,
                                )
                            z = acc[0] + k_val * T.cast(
                                conv_input[b, t, v_head_offset + vv],
                                accum_dtype,
                            )
                            tz = T.tanh(z)
                            if vv_out == 0:
                                tanh_cache[b, t, h, kk, vv] = T.cast(
                                    tz,
                                    carrier_dtype,
                                )
                            h_next[kk, vv] = f_val * h_state[kk, vv] + one_minus_f * tz
                        y_acc[0] = y_acc[0] + q_val * h_next[kk, vv_out]

                    y[b, t, h, vv_out] = T.cast(y_acc[0], carrier_dtype)
                    for kk in T.serial(k_dim):
                        for vv in T.serial(v_dim):
                            h_state[kk, vv] = h_next[kk, vv]
                if vv_out == 0:
                    for kk in T.serial(k_dim):
                        for vv in T.serial(v_dim):
                            h_last[b, h, kk, vv] = T.cast(
                                h_state[kk, vv],
                                carrier_dtype,
                            )

    lowering = dispatch_lower(fwd, target="metal", return_msl=True)
    input_names = [
        name for name in lowering.buffer_param_names if name not in _FWD_OUTPUT_NAMES
    ]
    if set(input_names) != {"conv_input", "W", "xf", "h0"}:
        raise MSLDispatchUnsupported(
            "unexpected packed M2RNN Path C buffer signature: "
            + ", ".join(lowering.buffer_param_names)
        )
    import tilelang

    kernel = tilelang.compile(
        fwd,
        target=_msl_transform._as_metal_target("metal"),
        execution_backend="tvm_ffi",
        out_idx=list(_PACKED_FWD_OUTPUT_IDX),
    )
    return kernel, lowering


@lru_cache(maxsize=128)
def _packed_bwd_k_parallel_kernel_for(
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

    del return_msl
    conv_dim = heads * (2 * k_dim + v_dim)
    k_offset = heads * k_dim
    v_offset = k_offset + heads * k_dim
    groups = batch * heads
    threads = _threadgroup_threads_for(k_dim, v_dim)
    if threads < k_dim or threads < v_dim:
        raise MSLDispatchUnsupported(
            f"packed M2RNN k-parallel bwd Path C needs one thread per K/V lane; "
            f"got K={k_dim}, V={v_dim}, threads={threads}"
        )
    accum_dtype = "float32"

    @T.prim_func
    def bwd(
        dy: T.Tensor((batch, seq, heads, v_dim), carrier_dtype),
        conv_input: T.Tensor((batch, seq, conv_dim), carrier_dtype),
        W: T.Tensor((heads, v_dim, v_dim), carrier_dtype),
        xf: T.Tensor((batch, seq, heads), carrier_dtype),
        h0: T.Tensor((batch, heads, k_dim, v_dim), carrier_dtype),
        tanh_cache: T.Tensor((batch, seq, heads, k_dim, v_dim), carrier_dtype),
        dconv_input: T.Tensor((batch, seq, conv_dim), carrier_dtype),
        dW_partial: T.Tensor((batch, heads, v_dim, v_dim), carrier_dtype),
        dxf: T.Tensor((batch, seq, heads), carrier_dtype),
        dh0: T.Tensor((batch, heads, k_dim, v_dim), carrier_dtype),
        h_steps_scratch: T.Tensor((batch, heads, seq, k_dim, v_dim), carrier_dtype),
    ):
        with T.Kernel(groups, threads=threads) as group_id:
            tid = T.get_thread_binding(0)
            h = group_id % heads
            b = group_id // heads
            q_head_offset = h * k_dim
            k_head_offset = k_offset + h * k_dim
            v_head_offset = v_offset + h * v_dim
            h_row = T.alloc_local((v_dim,), accum_dtype)
            dh_row = T.alloc_local((v_dim,), accum_dtype)
            dz_row = T.alloc_local((v_dim,), accum_dtype)
            dh_next_row = T.alloc_local((v_dim,), accum_dtype)
            w_shared = T.alloc_shared((v_dim, v_dim), accum_dtype, scope="shared")
            dz_shared = T.alloc_shared((k_dim, v_dim), accum_dtype, scope="shared")
            h_prev_shared = T.alloc_shared((k_dim, v_dim), accum_dtype, scope="shared")
            dxf_partial = T.alloc_shared((k_dim,), accum_dtype, scope="shared")
            dW_shared = T.alloc_shared((v_dim, v_dim), accum_dtype, scope="shared")

            for i in T.serial(tid, v_dim * v_dim, step=threads):
                w_shared[i // v_dim, i % v_dim] = T.cast(
                    W[h, i // v_dim, i % v_dim],
                    accum_dtype,
                )
                dW_shared[i // v_dim, i % v_dim] = 0.0
            T.sync_threads()

            if tid < k_dim:
                for vv in T.serial(v_dim):
                    h_row[vv] = T.cast(h0[b, h, tid, vv], accum_dtype)
                    dh_row[vv] = 0.0

            for t in T.serial(seq):
                if tid < k_dim:
                    f_val = T.cast(xf[b, t, h], accum_dtype)
                    one_minus_f = 1.0 - f_val
                    for vv in T.serial(v_dim):
                        h_steps_scratch[b, h, t, tid, vv] = T.cast(
                            h_row[vv],
                            carrier_dtype,
                        )
                        tz = T.cast(tanh_cache[b, t, h, tid, vv], accum_dtype)
                        h_row[vv] = f_val * h_row[vv] + one_minus_f * tz
            T.sync_threads()

            for r in T.serial(seq):
                t = seq - 1 - r
                f_val = T.cast(xf[b, t, h], accum_dtype)
                one_minus_f = 1.0 - f_val

                if tid < k_dim:
                    q_val = T.cast(
                        conv_input[b, t, q_head_offset + tid],
                        accum_dtype,
                    )
                    dq_acc = T.alloc_local((1,), accum_dtype)
                    dq_acc[0] = 0.0
                    df_kk = T.alloc_local((1,), accum_dtype)
                    df_kk[0] = 0.0
                    for vv in T.serial(v_dim):
                        dY = T.cast(dy[b, t, h, vv], accum_dtype)
                        h_prev = T.cast(
                            h_steps_scratch[b, h, t, tid, vv],
                            accum_dtype,
                        )
                        tz = T.cast(
                            tanh_cache[b, t, h, tid, vv],
                            accum_dtype,
                        )
                        h_t = f_val * h_prev + one_minus_f * tz
                        dq_acc[0] = dq_acc[0] + dY * h_t
                        dh_row[vv] = dh_row[vv] + q_val * dY
                        df_kk[0] = df_kk[0] + dh_row[vv] * (h_prev - tz)
                        dz_val = one_minus_f * dh_row[vv] * (1.0 - tz * tz)
                        dz_row[vv] = dz_val
                        dz_shared[tid, vv] = dz_val
                        h_prev_shared[tid, vv] = h_prev
                    dconv_input[b, t, q_head_offset + tid] = T.cast(
                        dq_acc[0],
                        carrier_dtype,
                    )
                    dxf_partial[tid] = df_kk[0]

                    dk_acc = T.alloc_local((1,), accum_dtype)
                    dk_acc[0] = 0.0
                    for vv in T.serial(v_dim):
                        dk_acc[0] = dk_acc[0] + dz_row[vv] * T.cast(
                            conv_input[b, t, v_head_offset + vv],
                            accum_dtype,
                        )
                    dconv_input[b, t, k_head_offset + tid] = T.cast(
                        dk_acc[0],
                        carrier_dtype,
                    )
                T.sync_threads()

                if tid == 0:
                    df_total = T.alloc_local((1,), accum_dtype)
                    df_total[0] = 0.0
                    for kk in T.serial(k_dim):
                        df_total[0] = df_total[0] + dxf_partial[kk]
                    dxf[b, t, h] = T.cast(df_total[0], carrier_dtype)

                if tid < v_dim:
                    dv_acc = T.alloc_local((1,), accum_dtype)
                    dv_acc[0] = 0.0
                    for kk in T.serial(k_dim):
                        dv_acc[0] = dv_acc[0] + dz_shared[kk, tid] * T.cast(
                            conv_input[b, t, k_head_offset + kk],
                            accum_dtype,
                        )
                    dconv_input[b, t, v_head_offset + tid] = T.cast(
                        dv_acc[0],
                        carrier_dtype,
                    )

                for pair in T.serial(tid, v_dim * v_dim, step=threads):
                    v0 = pair // v_dim
                    vv = pair % v_dim
                    w_acc = T.alloc_local((1,), accum_dtype)
                    w_acc[0] = 0.0
                    for kk in T.serial(k_dim):
                        w_acc[0] = (
                            w_acc[0] + h_prev_shared[kk, v0] * dz_shared[kk, vv]
                        )
                    dW_shared[v0, vv] = dW_shared[v0, vv] + w_acc[0]
                T.sync_threads()

                if tid < k_dim:
                    for v_in in T.serial(v_dim):
                        acc = T.alloc_local((1,), accum_dtype)
                        acc[0] = f_val * dh_row[v_in]
                        for v_out in T.serial(v_dim):
                            acc[0] = acc[0] + dz_row[v_out] * w_shared[v_in, v_out]
                        dh_next_row[v_in] = acc[0]
                    for vv in T.serial(v_dim):
                        dh_row[vv] = dh_next_row[vv]
                T.sync_threads()

            if tid < k_dim:
                for vv in T.serial(v_dim):
                    dh0[b, h, tid, vv] = T.cast(dh_row[vv], carrier_dtype)
            for i in T.serial(tid, v_dim * v_dim, step=threads):
                dW_partial[b, h, i // v_dim, i % v_dim] = T.cast(
                    dW_shared[i // v_dim, i % v_dim],
                    carrier_dtype,
                )

    lowering = dispatch_lower(bwd, target="metal", return_msl=True)
    input_names = [
        name
        for name in lowering.buffer_param_names
        if name not in _PACKED_BWD_OUTPUT_NAMES
    ]
    if set(input_names) != {"dy", "conv_input", "W", "xf", "h0", "tanh_cache"}:
        raise MSLDispatchUnsupported(
            "unexpected packed M2RNN Path C bwd buffer signature: "
            + ", ".join(lowering.buffer_param_names)
        )
    import tilelang

    kernel = tilelang.compile(
        bwd,
        target=_msl_transform._as_metal_target("metal"),
        execution_backend="tvm_ffi",
        out_idx=list(_PACKED_BWD_OUTPUT_IDX),
    )
    return kernel, lowering


@lru_cache(maxsize=128)
def _packed_bwd_kernel_for(
    batch: int,
    seq: int,
    heads: int,
    k_dim: int,
    v_dim: int,
    carrier_dtype: str,
    *,
    return_msl: bool = False,
) -> tuple[Any, _msl_transform.TileLangMSLLowering]:
    if k_dim >= _PACKED_BWD_K_PARALLEL_MIN_K and max(k_dim, v_dim) <= 256:
        return _packed_bwd_k_parallel_kernel_for(
            batch,
            seq,
            heads,
            k_dim,
            v_dim,
            carrier_dtype,
            return_msl=return_msl,
        )

    import tilelang.language as T

    del return_msl
    conv_dim = heads * (2 * k_dim + v_dim)
    k_offset = heads * k_dim
    v_offset = k_offset + heads * k_dim
    lanes = batch * heads
    threads = _threads_for(lanes)
    accum_dtype = "float32"

    @T.prim_func
    def bwd(
        dy: T.Tensor((batch, seq, heads, v_dim), carrier_dtype),
        conv_input: T.Tensor((batch, seq, conv_dim), carrier_dtype),
        W: T.Tensor((heads, v_dim, v_dim), carrier_dtype),
        xf: T.Tensor((batch, seq, heads), carrier_dtype),
        h0: T.Tensor((batch, heads, k_dim, v_dim), carrier_dtype),
        tanh_cache: T.Tensor((batch, seq, heads, k_dim, v_dim), carrier_dtype),
        dconv_input: T.Tensor((batch, seq, conv_dim), carrier_dtype),
        dW_partial: T.Tensor((batch, heads, v_dim, v_dim), carrier_dtype),
        dxf: T.Tensor((batch, seq, heads), carrier_dtype),
        dh0: T.Tensor((batch, heads, k_dim, v_dim), carrier_dtype),
        h_steps_scratch: T.Tensor((batch, heads, seq, k_dim, v_dim), carrier_dtype),
    ):
        with T.Kernel(T.ceildiv(lanes, threads), threads=threads) as bx:
            tid = T.get_thread_binding(0)
            lane = bx * threads + tid
            h_state = T.alloc_local((k_dim, v_dim), accum_dtype)
            dh = T.alloc_local((k_dim, v_dim), accum_dtype)
            dz = T.alloc_local((k_dim, v_dim), accum_dtype)
            dh_next = T.alloc_local((k_dim, v_dim), accum_dtype)
            dW_acc = T.alloc_local((v_dim, v_dim), accum_dtype)
            if lane < lanes:
                h = lane % heads
                b = lane // heads
                q_head_offset = h * k_dim
                k_head_offset = k_offset + h * k_dim
                v_head_offset = v_offset + h * v_dim

                for kk in T.serial(k_dim):
                    for vv in T.serial(v_dim):
                        h_state[kk, vv] = T.cast(h0[b, h, kk, vv], accum_dtype)
                        dh[kk, vv] = 0.0
                for v0 in T.serial(v_dim):
                    for vv in T.serial(v_dim):
                        dW_acc[v0, vv] = 0.0

                for t in T.serial(seq):
                    f_val = T.cast(xf[b, t, h], accum_dtype)
                    one_minus_f = 1.0 - f_val
                    for kk in T.serial(k_dim):
                        for vv in T.serial(v_dim):
                            h_steps_scratch[b, h, t, kk, vv] = T.cast(
                                h_state[kk, vv],
                                carrier_dtype,
                            )
                            tz = T.cast(tanh_cache[b, t, h, kk, vv], accum_dtype)
                            h_state[kk, vv] = (
                                f_val * h_state[kk, vv] + one_minus_f * tz
                            )

                for r in T.serial(seq):
                    t = seq - 1 - r
                    f_val = T.cast(xf[b, t, h], accum_dtype)
                    one_minus_f = 1.0 - f_val

                    for kk in T.serial(k_dim):
                        q_val = T.cast(
                            conv_input[b, t, q_head_offset + kk],
                            accum_dtype,
                        )
                        dq_acc = T.alloc_local((1,), accum_dtype)
                        dq_acc[0] = 0.0
                        for vv in T.serial(v_dim):
                            dY = T.cast(dy[b, t, h, vv], accum_dtype)
                            h_prev = T.cast(
                                h_steps_scratch[b, h, t, kk, vv],
                                accum_dtype,
                            )
                            tz = T.cast(
                                tanh_cache[b, t, h, kk, vv],
                                accum_dtype,
                            )
                            h_t = f_val * h_prev + one_minus_f * tz
                            dq_acc[0] = dq_acc[0] + dY * h_t
                            dh[kk, vv] = dh[kk, vv] + q_val * dY
                        dconv_input[b, t, q_head_offset + kk] = T.cast(
                            dq_acc[0],
                            carrier_dtype,
                        )

                    df_acc = T.alloc_local((1,), accum_dtype)
                    df_acc[0] = 0.0
                    for kk in T.serial(k_dim):
                        for vv in T.serial(v_dim):
                            h_prev = T.cast(
                                h_steps_scratch[b, h, t, kk, vv],
                                accum_dtype,
                            )
                            tz = T.cast(
                                tanh_cache[b, t, h, kk, vv],
                                accum_dtype,
                            )
                            dh_kv = dh[kk, vv]
                            df_acc[0] = df_acc[0] + dh_kv * (h_prev - tz)
                            dz[kk, vv] = one_minus_f * dh_kv * (1.0 - tz * tz)
                    dxf[b, t, h] = T.cast(df_acc[0], carrier_dtype)

                    for kk in T.serial(k_dim):
                        dk_acc = T.alloc_local((1,), accum_dtype)
                        dk_acc[0] = 0.0
                        for vv in T.serial(v_dim):
                            dk_acc[0] = dk_acc[0] + dz[kk, vv] * T.cast(
                                conv_input[b, t, v_head_offset + vv],
                                accum_dtype,
                            )
                        dconv_input[b, t, k_head_offset + kk] = T.cast(
                            dk_acc[0],
                            carrier_dtype,
                        )

                    for vv in T.serial(v_dim):
                        dv_acc = T.alloc_local((1,), accum_dtype)
                        dv_acc[0] = 0.0
                        for kk in T.serial(k_dim):
                            dv_acc[0] = dv_acc[0] + dz[kk, vv] * T.cast(
                                conv_input[b, t, k_head_offset + kk],
                                accum_dtype,
                            )
                        dconv_input[b, t, v_head_offset + vv] = T.cast(
                            dv_acc[0],
                            carrier_dtype,
                        )

                    for v0 in T.serial(v_dim):
                        for vv in T.serial(v_dim):
                            w_acc = T.alloc_local((1,), accum_dtype)
                            w_acc[0] = 0.0
                            for kk in T.serial(k_dim):
                                h_prev = T.cast(
                                    h_steps_scratch[b, h, t, kk, v0],
                                    accum_dtype,
                                )
                                w_acc[0] = w_acc[0] + h_prev * dz[kk, vv]
                            dW_acc[v0, vv] = dW_acc[v0, vv] + w_acc[0]

                    for kk in T.serial(k_dim):
                        for v_in in T.serial(v_dim):
                            acc = T.alloc_local((1,), accum_dtype)
                            acc[0] = f_val * dh[kk, v_in]
                            for v_out in T.serial(v_dim):
                                acc[0] = acc[0] + dz[kk, v_out] * T.cast(
                                    W[h, v_in, v_out],
                                    accum_dtype,
                                )
                            dh_next[kk, v_in] = acc[0]
                    for kk in T.serial(k_dim):
                        for vv in T.serial(v_dim):
                            dh[kk, vv] = dh_next[kk, vv]

                for kk in T.serial(k_dim):
                    for vv in T.serial(v_dim):
                        dh0[b, h, kk, vv] = T.cast(dh[kk, vv], carrier_dtype)
                for v0 in T.serial(v_dim):
                    for vv in T.serial(v_dim):
                        dW_partial[b, h, v0, vv] = T.cast(
                            dW_acc[v0, vv],
                            carrier_dtype,
                        )

    lowering = dispatch_lower(bwd, target="metal", return_msl=True)
    input_names = [
        name
        for name in lowering.buffer_param_names
        if name not in _PACKED_BWD_OUTPUT_NAMES
    ]
    if set(input_names) != {"dy", "conv_input", "W", "xf", "h0", "tanh_cache"}:
        raise MSLDispatchUnsupported(
            "unexpected packed M2RNN Path C bwd buffer signature: "
            + ", ".join(lowering.buffer_param_names)
        )
    import tilelang

    kernel = tilelang.compile(
        bwd,
        target=_msl_transform._as_metal_target("metal"),
        execution_backend="tvm_ffi",
        out_idx=list(_PACKED_BWD_OUTPUT_IDX),
    )
    return kernel, lowering


@lru_cache(maxsize=128)
def _mapped_packed_fwd_kernel_for(
    batch: int,
    seq: int,
    total_heads: int,
    q_heads: int,
    k_heads: int,
    v_heads: int,
    w_heads: int,
    f_heads: int,
    k_dim: int,
    v_dim: int,
    carrier_dtype: str,
    *,
    return_msl: bool = False,
) -> tuple[Any, _msl_transform.TileLangMSLLowering]:
    import tilelang.language as T

    del return_msl
    conv_dim = _mapped_conv_dim(q_heads, k_heads, v_heads, k_dim, v_dim)
    k_offset = q_heads * k_dim
    v_offset = k_offset + k_heads * k_dim
    q_group = total_heads // q_heads
    k_group = total_heads // k_heads
    v_group = total_heads // v_heads
    w_group = total_heads // w_heads
    f_group = total_heads // f_heads
    lanes = batch * total_heads * v_dim
    threads = _threads_for(lanes)
    accum_dtype = "float32"

    @T.prim_func
    def fwd(
        conv_input: T.Tensor((batch, seq, conv_dim), carrier_dtype),
        W: T.Tensor((w_heads, v_dim, v_dim), carrier_dtype),
        xf: T.Tensor((batch, seq, f_heads), carrier_dtype),
        h0: T.Tensor((batch, total_heads, k_dim, v_dim), carrier_dtype),
        h_last: T.Tensor((batch, total_heads, k_dim, v_dim), carrier_dtype),
        tanh_cache: T.Tensor((batch, seq, total_heads, k_dim, v_dim), carrier_dtype),
        y: T.Tensor((batch, seq, total_heads, v_dim), carrier_dtype),
    ):
        with T.Kernel(T.ceildiv(lanes, threads), threads=threads) as bx:
            tid = T.get_thread_binding(0)
            lane = bx * threads + tid
            h_state = T.alloc_local((k_dim, v_dim), accum_dtype)
            h_next = T.alloc_local((k_dim, v_dim), accum_dtype)
            if lane < lanes:
                vv_out = lane % v_dim
                h = (lane // v_dim) % total_heads
                b = lane // (v_dim * total_heads)
                q_src = h // q_group
                k_src = h // k_group
                v_src = h // v_group
                w_src = h // w_group
                f_src = h // f_group
                q_head_offset = q_src * k_dim
                k_head_offset = k_offset + k_src * k_dim
                v_head_offset = v_offset + v_src * v_dim

                for kk in T.serial(k_dim):
                    for vv in T.serial(v_dim):
                        h_state[kk, vv] = T.cast(h0[b, h, kk, vv], accum_dtype)

                for t in T.serial(seq):
                    f_val = T.cast(xf[b, t, f_src], accum_dtype)
                    one_minus_f = 1.0 - f_val
                    y_acc = T.alloc_local((1,), accum_dtype)
                    y_acc[0] = 0.0

                    for kk in T.serial(k_dim):
                        k_val = T.cast(
                            conv_input[b, t, k_head_offset + kk],
                            accum_dtype,
                        )
                        q_val = T.cast(
                            conv_input[b, t, q_head_offset + kk],
                            accum_dtype,
                        )
                        for vv in T.serial(v_dim):
                            acc = T.alloc_local((1,), accum_dtype)
                            acc[0] = 0.0
                            for v0 in T.serial(v_dim):
                                acc[0] = acc[0] + h_state[kk, v0] * T.cast(
                                    W[w_src, v0, vv],
                                    accum_dtype,
                                )
                            z = acc[0] + k_val * T.cast(
                                conv_input[b, t, v_head_offset + vv],
                                accum_dtype,
                            )
                            tz = T.tanh(z)
                            if vv_out == 0:
                                tanh_cache[b, t, h, kk, vv] = T.cast(
                                    tz,
                                    carrier_dtype,
                                )
                            h_next[kk, vv] = f_val * h_state[kk, vv] + one_minus_f * tz
                        y_acc[0] = y_acc[0] + q_val * h_next[kk, vv_out]

                    y[b, t, h, vv_out] = T.cast(y_acc[0], carrier_dtype)
                    for kk in T.serial(k_dim):
                        for vv in T.serial(v_dim):
                            h_state[kk, vv] = h_next[kk, vv]
                if vv_out == 0:
                    for kk in T.serial(k_dim):
                        for vv in T.serial(v_dim):
                            h_last[b, h, kk, vv] = T.cast(
                                h_state[kk, vv],
                                carrier_dtype,
                            )

    lowering = dispatch_lower(fwd, target="metal", return_msl=True)
    input_names = [
        name for name in lowering.buffer_param_names if name not in _FWD_OUTPUT_NAMES
    ]
    if set(input_names) != {"conv_input", "W", "xf", "h0"}:
        raise MSLDispatchUnsupported(
            "unexpected mapped packed M2RNN Path C buffer signature: "
            + ", ".join(lowering.buffer_param_names)
        )
    import tilelang

    kernel = tilelang.compile(
        fwd,
        target=_msl_transform._as_metal_target("metal"),
        execution_backend="tvm_ffi",
        out_idx=list(_PACKED_FWD_OUTPUT_IDX),
    )
    return kernel, lowering


@lru_cache(maxsize=128)
def _mapped_packed_post_fwd_kernel_for(
    batch: int,
    seq: int,
    total_heads: int,
    q_heads: int,
    k_heads: int,
    v_heads: int,
    g_heads: int,
    w_heads: int,
    f_heads: int,
    k_dim: int,
    v_dim: int,
    projected_dim: int,
    carrier_dtype: str,
    *,
    return_msl: bool = False,
) -> tuple[Any, _msl_transform.TileLangMSLLowering]:
    import tilelang.language as T

    del return_msl
    conv_dim = _mapped_conv_dim(q_heads, k_heads, v_heads, k_dim, v_dim)
    k_offset = q_heads * k_dim
    v_offset = k_offset + k_heads * k_dim
    q_group = total_heads // q_heads
    k_group = total_heads // k_heads
    v_group = total_heads // v_heads
    g_repeat = total_heads // g_heads
    w_group = total_heads // w_heads
    f_group = total_heads // f_heads
    g_dim = g_heads * v_dim
    g_offset = projected_dim - g_dim
    features = total_heads * v_dim
    lanes = batch * features
    threads = _threads_for(lanes)
    accum_dtype = "float32"

    @T.prim_func
    def fwd(
        conv_input: T.Tensor((batch, seq, conv_dim), carrier_dtype),
        W: T.Tensor((w_heads, v_dim, v_dim), carrier_dtype),
        xf: T.Tensor((batch, seq, f_heads), carrier_dtype),
        h0: T.Tensor((batch, total_heads, k_dim, v_dim), carrier_dtype),
        D: T.Tensor((total_heads, v_dim), carrier_dtype),
        projected: T.Tensor((batch, seq, projected_dim), carrier_dtype),
        h_last: T.Tensor((batch, total_heads, k_dim, v_dim), carrier_dtype),
        tanh_cache: T.Tensor((batch, seq, total_heads, k_dim, v_dim), carrier_dtype),
        post: T.Tensor((batch, seq, features), carrier_dtype),
    ):
        with T.Kernel(T.ceildiv(lanes, threads), threads=threads) as bx:
            tid = T.get_thread_binding(0)
            lane = bx * threads + tid
            h_state = T.alloc_local((k_dim, v_dim), accum_dtype)
            h_next = T.alloc_local((k_dim, v_dim), accum_dtype)
            if lane < lanes:
                feature = lane % features
                vv_out = feature % v_dim
                h = feature // v_dim
                b = lane // features
                q_src = h // q_group
                k_src = h // k_group
                v_src = h // v_group
                w_src = h // w_group
                f_src = h // f_group
                g_flat = feature // g_repeat
                q_head_offset = q_src * k_dim
                k_head_offset = k_offset + k_src * k_dim
                v_head_offset = v_offset + v_src * v_dim
                v_index = v_head_offset + vv_out
                g_index = g_offset + g_flat
                d_val = T.cast(D[h, vv_out], accum_dtype)

                for kk in T.serial(k_dim):
                    for vv in T.serial(v_dim):
                        h_state[kk, vv] = T.cast(h0[b, h, kk, vv], accum_dtype)

                for t in T.serial(seq):
                    f_val = T.cast(xf[b, t, f_src], accum_dtype)
                    one_minus_f = 1.0 - f_val
                    y_acc = T.alloc_local((1,), accum_dtype)
                    y_acc[0] = 0.0

                    for kk in T.serial(k_dim):
                        k_val = T.cast(
                            conv_input[b, t, k_head_offset + kk],
                            accum_dtype,
                        )
                        q_val = T.cast(
                            conv_input[b, t, q_head_offset + kk],
                            accum_dtype,
                        )
                        for vv in T.serial(v_dim):
                            acc = T.alloc_local((1,), accum_dtype)
                            acc[0] = 0.0
                            for v0 in T.serial(v_dim):
                                acc[0] = acc[0] + h_state[kk, v0] * T.cast(
                                    W[w_src, v0, vv],
                                    accum_dtype,
                                )
                            z = acc[0] + k_val * T.cast(
                                conv_input[b, t, v_head_offset + vv],
                                accum_dtype,
                            )
                            tz = T.tanh(z)
                            if vv_out == 0:
                                tanh_cache[b, t, h, kk, vv] = T.cast(
                                    tz,
                                    carrier_dtype,
                                )
                            h_next[kk, vv] = f_val * h_state[kk, vv] + one_minus_f * tz
                        y_acc[0] = y_acc[0] + q_val * h_next[kk, vv_out]

                    g_val = T.cast(projected[b, t, g_index], accum_dtype)
                    sig_g = T.alloc_var(T.float32, init=0.0)
                    if g_val >= 0.0:
                        sig_g = 1.0 / (1.0 + T.exp(-g_val))
                    else:
                        sig_g = T.exp(g_val)
                        sig_g = sig_g / (1.0 + sig_g)
                    v_val = T.cast(conv_input[b, t, v_index], accum_dtype)
                    post[b, t, feature] = T.cast(
                        (y_acc[0] + v_val * d_val) * g_val * sig_g,
                        carrier_dtype,
                    )

                    for kk in T.serial(k_dim):
                        for vv in T.serial(v_dim):
                            h_state[kk, vv] = h_next[kk, vv]
                if vv_out == 0:
                    for kk in T.serial(k_dim):
                        for vv in T.serial(v_dim):
                            h_last[b, h, kk, vv] = T.cast(
                                h_state[kk, vv],
                                carrier_dtype,
                            )

    lowering = dispatch_lower(fwd, target="metal", return_msl=True)
    input_names = [
        name
        for name in lowering.buffer_param_names
        if name not in _PACKED_POST_FWD_OUTPUT_NAMES
    ]
    if set(input_names) != {"conv_input", "W", "xf", "h0", "D", "projected"}:
        raise MSLDispatchUnsupported(
            "unexpected mapped packed M2RNN inline post Path C buffer signature: "
            + ", ".join(lowering.buffer_param_names)
        )
    import tilelang

    kernel = tilelang.compile(
        fwd,
        target=_msl_transform._as_metal_target("metal"),
        execution_backend="tvm_ffi",
        out_idx=list(_PACKED_POST_FWD_OUTPUT_IDX),
    )
    return kernel, lowering


@lru_cache(maxsize=128)
def _mapped_packed_bwd_kernel_for(
    batch: int,
    seq: int,
    total_heads: int,
    q_heads: int,
    k_heads: int,
    v_heads: int,
    w_heads: int,
    f_heads: int,
    k_dim: int,
    v_dim: int,
    carrier_dtype: str,
    dy_dtype: str,
    grad_dtype: str,
    *,
    return_msl: bool = False,
) -> tuple[Any, _msl_transform.TileLangMSLLowering]:
    import tilelang.language as T

    del return_msl
    conv_dim = _mapped_conv_dim(q_heads, k_heads, v_heads, k_dim, v_dim)
    k_offset = q_heads * k_dim
    v_offset = k_offset + k_heads * k_dim
    q_group = total_heads // q_heads
    k_group = total_heads // k_heads
    v_group = total_heads // v_heads
    w_group = total_heads // w_heads
    f_group = total_heads // f_heads
    lanes = batch * total_heads
    threads = _threads_for(lanes)
    accum_dtype = "float32"

    @T.prim_func
    def bwd(
        dy: T.Tensor((batch, seq, total_heads, v_dim), dy_dtype),
        conv_input: T.Tensor((batch, seq, conv_dim), carrier_dtype),
        W: T.Tensor((w_heads, v_dim, v_dim), carrier_dtype),
        xf: T.Tensor((batch, seq, f_heads), carrier_dtype),
        h0: T.Tensor((batch, total_heads, k_dim, v_dim), carrier_dtype),
        tanh_cache: T.Tensor((batch, seq, total_heads, k_dim, v_dim), carrier_dtype),
        dconv_input: T.Tensor((batch, seq, conv_dim), grad_dtype),
        dW_partial: T.Tensor((batch, w_heads, v_dim, v_dim), grad_dtype),
        dxf: T.Tensor((batch, seq, f_heads), grad_dtype),
        dh0: T.Tensor((batch, total_heads, k_dim, v_dim), grad_dtype),
        h_steps_scratch: T.Tensor((batch, total_heads, seq, k_dim, v_dim), carrier_dtype),
    ):
        with T.Kernel(T.ceildiv(lanes, threads), threads=threads) as bx:
            tid = T.get_thread_binding(0)
            lane = bx * threads + tid
            h_state = T.alloc_local((k_dim, v_dim), accum_dtype)
            dh = T.alloc_local((k_dim, v_dim), accum_dtype)
            dz = T.alloc_local((k_dim, v_dim), accum_dtype)
            dh_next = T.alloc_local((k_dim, v_dim), accum_dtype)
            dW_acc = T.alloc_local((v_dim, v_dim), accum_dtype)
            if lane < lanes:
                h = lane % total_heads
                b = lane // total_heads
                q_src = h // q_group
                k_src = h // k_group
                v_src = h // v_group
                w_src = h // w_group
                f_src = h // f_group
                q_head_offset = q_src * k_dim
                k_head_offset = k_offset + k_src * k_dim
                v_head_offset = v_offset + v_src * v_dim

                for kk in T.serial(k_dim):
                    for vv in T.serial(v_dim):
                        h_state[kk, vv] = T.cast(h0[b, h, kk, vv], accum_dtype)
                        dh[kk, vv] = 0.0
                for v0 in T.serial(v_dim):
                    for vv in T.serial(v_dim):
                        dW_acc[v0, vv] = 0.0

                for t in T.serial(seq):
                    f_val = T.cast(xf[b, t, f_src], accum_dtype)
                    one_minus_f = 1.0 - f_val
                    for kk in T.serial(k_dim):
                        for vv in T.serial(v_dim):
                            h_steps_scratch[b, h, t, kk, vv] = T.cast(
                                h_state[kk, vv],
                                carrier_dtype,
                            )
                            tz = T.cast(tanh_cache[b, t, h, kk, vv], accum_dtype)
                            h_state[kk, vv] = f_val * h_state[kk, vv] + one_minus_f * tz

                for r in T.serial(seq):
                    t = seq - 1 - r
                    f_val = T.cast(xf[b, t, f_src], accum_dtype)
                    one_minus_f = 1.0 - f_val

                    for kk in T.serial(k_dim):
                        q_val = T.cast(
                            conv_input[b, t, q_head_offset + kk],
                            accum_dtype,
                        )
                        dq_acc = T.alloc_local((1,), accum_dtype)
                        dq_acc[0] = 0.0
                        for vv in T.serial(v_dim):
                            dY = T.cast(dy[b, t, h, vv], accum_dtype)
                            h_prev = T.cast(
                                h_steps_scratch[b, h, t, kk, vv],
                                accum_dtype,
                            )
                            tz = T.cast(
                                tanh_cache[b, t, h, kk, vv],
                                accum_dtype,
                            )
                            h_t = f_val * h_prev + one_minus_f * tz
                            dq_acc[0] = dq_acc[0] + dY * h_t
                            dh[kk, vv] = dh[kk, vv] + q_val * dY
                        T.atomic_add(
                            dconv_input[b, t, q_head_offset + kk],
                            dq_acc[0],
                            memory_order="relaxed",
                        )

                    df_acc = T.alloc_local((1,), accum_dtype)
                    df_acc[0] = 0.0
                    for kk in T.serial(k_dim):
                        for vv in T.serial(v_dim):
                            h_prev = T.cast(
                                h_steps_scratch[b, h, t, kk, vv],
                                accum_dtype,
                            )
                            tz = T.cast(
                                tanh_cache[b, t, h, kk, vv],
                                accum_dtype,
                            )
                            dh_kv = dh[kk, vv]
                            df_acc[0] = df_acc[0] + dh_kv * (h_prev - tz)
                            dz[kk, vv] = one_minus_f * dh_kv * (1.0 - tz * tz)
                    T.atomic_add(
                        dxf[b, t, f_src],
                        df_acc[0],
                        memory_order="relaxed",
                    )

                    for kk in T.serial(k_dim):
                        dk_acc = T.alloc_local((1,), accum_dtype)
                        dk_acc[0] = 0.0
                        for vv in T.serial(v_dim):
                            dk_acc[0] = dk_acc[0] + dz[kk, vv] * T.cast(
                                conv_input[b, t, v_head_offset + vv],
                                accum_dtype,
                            )
                        T.atomic_add(
                            dconv_input[b, t, k_head_offset + kk],
                            dk_acc[0],
                            memory_order="relaxed",
                        )

                    for vv in T.serial(v_dim):
                        dv_acc = T.alloc_local((1,), accum_dtype)
                        dv_acc[0] = 0.0
                        for kk in T.serial(k_dim):
                            dv_acc[0] = dv_acc[0] + dz[kk, vv] * T.cast(
                                conv_input[b, t, k_head_offset + kk],
                                accum_dtype,
                            )
                        T.atomic_add(
                            dconv_input[b, t, v_head_offset + vv],
                            dv_acc[0],
                            memory_order="relaxed",
                        )

                    for v0 in T.serial(v_dim):
                        for vv in T.serial(v_dim):
                            w_acc = T.alloc_local((1,), accum_dtype)
                            w_acc[0] = 0.0
                            for kk in T.serial(k_dim):
                                h_prev = T.cast(
                                    h_steps_scratch[b, h, t, kk, v0],
                                    accum_dtype,
                                )
                                w_acc[0] = w_acc[0] + h_prev * dz[kk, vv]
                            dW_acc[v0, vv] = dW_acc[v0, vv] + w_acc[0]

                    for kk in T.serial(k_dim):
                        for v_in in T.serial(v_dim):
                            acc = T.alloc_local((1,), accum_dtype)
                            acc[0] = f_val * dh[kk, v_in]
                            for v_out in T.serial(v_dim):
                                acc[0] = acc[0] + dz[kk, v_out] * T.cast(
                                    W[w_src, v_in, v_out],
                                    accum_dtype,
                                )
                            dh_next[kk, v_in] = acc[0]
                    for kk in T.serial(k_dim):
                        for vv in T.serial(v_dim):
                            dh[kk, vv] = dh_next[kk, vv]

                for kk in T.serial(k_dim):
                    for vv in T.serial(v_dim):
                        dh0[b, h, kk, vv] = T.cast(dh[kk, vv], grad_dtype)
                for v0 in T.serial(v_dim):
                    for vv in T.serial(v_dim):
                        T.atomic_add(
                            dW_partial[b, w_src, v0, vv],
                            dW_acc[v0, vv],
                            memory_order="relaxed",
                        )

    lowering = dispatch_lower(bwd, target="metal", return_msl=True)
    input_names = [
        name
        for name in lowering.buffer_param_names
        if name not in _PACKED_BWD_OUTPUT_NAMES
    ]
    if set(input_names) != {"dy", "conv_input", "W", "xf", "h0", "tanh_cache"}:
        raise MSLDispatchUnsupported(
            "unexpected mapped packed M2RNN Path C bwd buffer signature: "
            + ", ".join(lowering.buffer_param_names)
        )
    import tilelang

    kernel = tilelang.compile(
        bwd,
        target=_msl_transform._as_metal_target("metal"),
        execution_backend="tvm_ffi",
        out_idx=list(_PACKED_BWD_OUTPUT_IDX),
    )
    return kernel, lowering


@lru_cache(maxsize=128)
def _post_residual_gate_fwd_kernel_for(
    batch: int,
    seq: int,
    total_heads: int,
    q_heads: int,
    k_heads: int,
    v_heads: int,
    g_heads: int,
    k_dim: int,
    v_dim: int,
    conv_dim: int,
    projected_dim: int,
    carrier_dtype: str,
    *,
    return_msl: bool = False,
) -> tuple[Any, _msl_transform.TileLangMSLLowering]:
    import tilelang.language as T

    del return_msl
    v_offset = q_heads * k_dim + k_heads * k_dim
    v_group = total_heads // v_heads
    g_repeat = total_heads // g_heads
    g_dim = g_heads * v_dim
    g_offset = projected_dim - g_dim
    features = total_heads * v_dim
    lanes = batch * seq * features
    threads = _threads_for(features)
    accum_dtype = "float32"

    @T.prim_func
    def fwd(
        y: T.Tensor((batch, seq, total_heads, v_dim), carrier_dtype),
        conv_input: T.Tensor((batch, seq, conv_dim), carrier_dtype),
        D: T.Tensor((total_heads, v_dim), carrier_dtype),
        projected: T.Tensor((batch, seq, projected_dim), carrier_dtype),
        post: T.Tensor((batch, seq, features), carrier_dtype),
    ):
        with T.Kernel(T.ceildiv(lanes, threads), threads=threads) as bx:
            tid = T.get_thread_binding(0)
            lane = bx * threads + tid
            if lane < lanes:
                feature = lane % features
                t = (lane // features) % seq
                b = lane // (features * seq)
                h = feature // v_dim
                vv = feature % v_dim
                v_src = h // v_group
                g_flat = feature // g_repeat
                g_val = T.cast(projected[b, t, g_offset + g_flat], accum_dtype)
                sig_g = T.alloc_var(T.float32, init=0.0)
                if g_val >= 0.0:
                    sig_g = 1.0 / (1.0 + T.exp(-g_val))
                else:
                    sig_g = T.exp(g_val)
                    sig_g = sig_g / (1.0 + sig_g)
                y_val = T.cast(y[b, t, h, vv], accum_dtype)
                v_val = T.cast(
                    conv_input[b, t, v_offset + v_src * v_dim + vv],
                    accum_dtype,
                )
                d_val = T.cast(D[h, vv], accum_dtype)
                post[b, t, feature] = T.cast(
                    (y_val + v_val * d_val) * g_val * sig_g,
                    carrier_dtype,
                )

    lowering = dispatch_lower(fwd, target="metal", return_msl=True)
    input_names = [
        name for name in lowering.buffer_param_names if name not in _POST_FWD_OUTPUT_NAMES
    ]
    if set(input_names) != {"y", "conv_input", "D", "projected"}:
        raise MSLDispatchUnsupported(
            "unexpected M2RNN post Path C buffer signature: "
            + ", ".join(lowering.buffer_param_names)
        )
    import tilelang

    kernel = tilelang.compile(
        fwd,
        target=_msl_transform._as_metal_target("metal"),
        execution_backend="tvm_ffi",
        out_idx=list(_POST_FWD_OUTPUT_IDX),
    )
    return kernel, lowering


@lru_cache(maxsize=128)
def _post_residual_gate_bwd_kernel_for(
    batch: int,
    seq: int,
    total_heads: int,
    q_heads: int,
    k_heads: int,
    v_heads: int,
    g_heads: int,
    k_dim: int,
    v_dim: int,
    conv_dim: int,
    projected_dim: int,
    carrier_dtype: str,
    grad_dtype: str,
    *,
    return_msl: bool = False,
) -> tuple[Any, _msl_transform.TileLangMSLLowering]:
    import tilelang.language as T

    del return_msl
    v_offset = q_heads * k_dim + k_heads * k_dim
    v_group = total_heads // v_heads
    g_repeat = total_heads // g_heads
    g_dim = g_heads * v_dim
    g_offset = projected_dim - g_dim
    features = total_heads * v_dim
    lanes = batch * seq * features
    threads = _threads_for(features)
    accum_dtype = "float32"

    @T.prim_func
    def bwd(
        dpost: T.Tensor((batch, seq, features), carrier_dtype),
        y: T.Tensor((batch, seq, total_heads, v_dim), carrier_dtype),
        conv_input: T.Tensor((batch, seq, conv_dim), carrier_dtype),
        D: T.Tensor((total_heads, v_dim), carrier_dtype),
        projected: T.Tensor((batch, seq, projected_dim), carrier_dtype),
        dy_recurrent: T.Tensor((batch, seq, total_heads, v_dim), grad_dtype),
        dconv_input: T.Tensor((batch, seq, conv_dim), grad_dtype),
        dD: T.Tensor((total_heads, v_dim), grad_dtype),
        dprojected: T.Tensor((batch, seq, projected_dim), grad_dtype),
    ):
        with T.Kernel(T.ceildiv(lanes, threads), threads=threads) as bx:
            tid = T.get_thread_binding(0)
            lane = bx * threads + tid
            if lane < lanes:
                feature = lane % features
                t = (lane // features) % seq
                b = lane // (features * seq)
                h = feature // v_dim
                vv = feature % v_dim
                v_src = h // v_group
                v_index = v_offset + v_src * v_dim + vv
                g_flat = feature // g_repeat
                g_index = g_offset + g_flat
                dpost_val = T.cast(dpost[b, t, feature], accum_dtype)
                g_val = T.cast(projected[b, t, g_index], accum_dtype)
                sig_g = T.alloc_var(T.float32, init=0.0)
                if g_val >= 0.0:
                    sig_g = 1.0 / (1.0 + T.exp(-g_val))
                else:
                    sig_g = T.exp(g_val)
                    sig_g = sig_g / (1.0 + sig_g)
                silu_g = g_val * sig_g
                silu_dg = sig_g * (1.0 + g_val * (1.0 - sig_g))
                y_val = T.cast(y[b, t, h, vv], accum_dtype)
                v_val = T.cast(conv_input[b, t, v_index], accum_dtype)
                d_val = T.cast(D[h, vv], accum_dtype)
                skipped = y_val + v_val * d_val
                dskipped = dpost_val * silu_g

                dy_recurrent[b, t, h, vv] = T.cast(dskipped, grad_dtype)
                T.atomic_add(
                    dconv_input[b, t, v_index],
                    dskipped * d_val,
                    memory_order="relaxed",
                )
                T.atomic_add(
                    dD[h, vv],
                    dskipped * v_val,
                    memory_order="relaxed",
                )
                T.atomic_add(
                    dprojected[b, t, g_index],
                    dpost_val * skipped * silu_dg,
                    memory_order="relaxed",
                )

    lowering = dispatch_lower(bwd, target="metal", return_msl=True)
    input_names = [
        name
        for name in lowering.buffer_param_names
        if name not in _POST_BWD_OUTPUT_NAMES
    ]
    if set(input_names) != {"dpost", "y", "conv_input", "D", "projected"}:
        raise MSLDispatchUnsupported(
            "unexpected M2RNN post Path C bwd buffer signature: "
            + ", ".join(lowering.buffer_param_names)
        )
    import tilelang

    kernel = tilelang.compile(
        bwd,
        target=_msl_transform._as_metal_target("metal"),
        execution_backend="tvm_ffi",
        out_idx=list(_POST_BWD_OUTPUT_IDX),
    )
    return kernel, lowering


@lru_cache(maxsize=128)
def _post_residual_gate_bwd_from_recurrence_kernel_for(
    batch: int,
    seq: int,
    total_heads: int,
    q_heads: int,
    k_heads: int,
    v_heads: int,
    g_heads: int,
    f_heads: int,
    k_dim: int,
    v_dim: int,
    conv_dim: int,
    projected_dim: int,
    carrier_dtype: str,
    grad_dtype: str,
    *,
    return_msl: bool = False,
) -> tuple[Any, _msl_transform.TileLangMSLLowering]:
    import tilelang.language as T

    del return_msl
    v_offset = q_heads * k_dim + k_heads * k_dim
    q_group = total_heads // q_heads
    v_group = total_heads // v_heads
    g_repeat = total_heads // g_heads
    f_group = total_heads // f_heads
    g_dim = g_heads * v_dim
    g_offset = projected_dim - g_dim
    features = total_heads * v_dim
    lanes = batch * features
    threads = _threads_for(lanes)
    accum_dtype = "float32"

    @T.prim_func
    def bwd(
        dpost: T.Tensor((batch, seq, features), carrier_dtype),
        conv_input: T.Tensor((batch, seq, conv_dim), carrier_dtype),
        xf: T.Tensor((batch, seq, f_heads), carrier_dtype),
        h0: T.Tensor((batch, total_heads, k_dim, v_dim), carrier_dtype),
        tanh_cache: T.Tensor((batch, seq, total_heads, k_dim, v_dim), carrier_dtype),
        D: T.Tensor((total_heads, v_dim), carrier_dtype),
        projected: T.Tensor((batch, seq, projected_dim), carrier_dtype),
        dy_recurrent: T.Tensor((batch, seq, total_heads, v_dim), grad_dtype),
        dconv_input: T.Tensor((batch, seq, conv_dim), grad_dtype),
        dD: T.Tensor((total_heads, v_dim), grad_dtype),
        dprojected: T.Tensor((batch, seq, projected_dim), grad_dtype),
    ):
        with T.Kernel(T.ceildiv(lanes, threads), threads=threads) as bx:
            tid = T.get_thread_binding(0)
            lane = bx * threads + tid
            h_col = T.alloc_local((k_dim,), accum_dtype)
            if lane < lanes:
                feature = lane % features
                vv = feature % v_dim
                h = feature // v_dim
                b = lane // features
                q_src = h // q_group
                v_src = h // v_group
                f_src = h // f_group
                q_head_offset = q_src * k_dim
                v_index = v_offset + v_src * v_dim + vv
                g_flat = feature // g_repeat
                g_index = g_offset + g_flat
                d_val = T.cast(D[h, vv], accum_dtype)

                for kk in T.serial(k_dim):
                    h_col[kk] = T.cast(h0[b, h, kk, vv], accum_dtype)

                for t in T.serial(seq):
                    f_val = T.cast(xf[b, t, f_src], accum_dtype)
                    one_minus_f = 1.0 - f_val
                    y_acc = T.alloc_local((1,), accum_dtype)
                    y_acc[0] = 0.0
                    for kk in T.serial(k_dim):
                        q_val = T.cast(
                            conv_input[b, t, q_head_offset + kk],
                            accum_dtype,
                        )
                        tz = T.cast(tanh_cache[b, t, h, kk, vv], accum_dtype)
                        h_col[kk] = f_val * h_col[kk] + one_minus_f * tz
                        y_acc[0] = y_acc[0] + q_val * h_col[kk]

                    dpost_val = T.cast(dpost[b, t, feature], accum_dtype)
                    g_val = T.cast(projected[b, t, g_index], accum_dtype)
                    sig_g = T.alloc_var(T.float32, init=0.0)
                    if g_val >= 0.0:
                        sig_g = 1.0 / (1.0 + T.exp(-g_val))
                    else:
                        sig_g = T.exp(g_val)
                        sig_g = sig_g / (1.0 + sig_g)
                    silu_g = g_val * sig_g
                    silu_dg = sig_g * (1.0 + g_val * (1.0 - sig_g))
                    v_val = T.cast(conv_input[b, t, v_index], accum_dtype)
                    skipped = y_acc[0] + v_val * d_val
                    dskipped = dpost_val * silu_g

                    dy_recurrent[b, t, h, vv] = T.cast(dskipped, grad_dtype)
                    T.atomic_add(
                        dconv_input[b, t, v_index],
                        dskipped * d_val,
                        memory_order="relaxed",
                    )
                    T.atomic_add(
                        dD[h, vv],
                        dskipped * v_val,
                        memory_order="relaxed",
                    )
                    T.atomic_add(
                        dprojected[b, t, g_index],
                        dpost_val * skipped * silu_dg,
                        memory_order="relaxed",
                    )

    lowering = dispatch_lower(bwd, target="metal", return_msl=True)
    input_names = [
        name
        for name in lowering.buffer_param_names
        if name not in _POST_BWD_OUTPUT_NAMES
    ]
    if set(input_names) != {
        "dpost",
        "conv_input",
        "xf",
        "h0",
        "tanh_cache",
        "D",
        "projected",
    }:
        raise MSLDispatchUnsupported(
            "unexpected M2RNN inline post bwd buffer signature: "
            + ", ".join(lowering.buffer_param_names)
        )
    import tilelang

    kernel = tilelang.compile(
        bwd,
        target=_msl_transform._as_metal_target("metal"),
        execution_backend="tvm_ffi",
        out_idx=list(_POST_RECOMPUTE_BWD_OUTPUT_IDX),
    )
    return kernel, lowering


def m2rnn_path_c_status() -> M2RNNPathCStatus:
    if not _msl_transform.can_run_metal():
        return M2RNNPathCStatus(False, "MLX Metal backend is not available")
    probes = (
        (
            "M2RNN Path C fwd",
            _fwd_kernel_for,
            (1, 4, 2, 4, 4, "float32"),
        ),
        (
            "M2RNN Path C bwd",
            _bwd_kernel_for,
            (1, 4, 2, 4, 4, "float32"),
        ),
        (
            "packed M2RNN Path C fwd",
            _packed_fwd_kernel_for,
            (1, 4, 2, 4, 4, "float32"),
        ),
        (
            "packed M2RNN Path C bwd",
            _packed_bwd_kernel_for,
            (1, 4, 2, 4, 4, "float32"),
        ),
        (
            "packed M2RNN Path C bf16 K=16 fwd",
            _packed_fwd_kernel_for,
            (1, 4, 2, 16, 4, "bfloat16"),
        ),
        (
            "packed M2RNN Path C bf16 K=16 bwd",
            _packed_bwd_kernel_for,
            (1, 4, 2, 16, 4, "bfloat16"),
        ),
        (
            "mapped packed M2RNN Path C fwd",
            _mapped_packed_fwd_kernel_for,
            (1, 4, 4, 1, 1, 2, 1, 2, 4, 4, "float32"),
        ),
        (
            "mapped packed M2RNN Path C bwd",
            _mapped_packed_bwd_kernel_for,
            (1, 4, 4, 1, 1, 2, 1, 2, 4, 4, "float32", "float32", "float32"),
        ),
        (
            "mapped packed M2RNN Path C inline post fwd",
            _mapped_packed_post_fwd_kernel_for,
            (1, 4, 4, 1, 1, 2, 2, 1, 2, 4, 4, 22, "float32"),
        ),
        (
            "M2RNN post residual/gate Path C fwd",
            _post_residual_gate_fwd_kernel_for,
            (1, 4, 4, 1, 1, 2, 2, 4, 4, 12, 22, "float32"),
        ),
        (
            "M2RNN post residual/gate Path C bwd",
            _post_residual_gate_bwd_kernel_for,
            (1, 4, 4, 1, 1, 2, 2, 4, 4, 12, 22, "float32", "float32"),
        ),
        (
            "M2RNN inline post residual/gate Path C bwd",
            _post_residual_gate_bwd_from_recurrence_kernel_for,
            (1, 4, 4, 1, 1, 2, 2, 2, 4, 4, 12, 22, "float32", "float32"),
        ),
    )
    for label, kernel_factory, args in probes:
        status = _kernel_lowering_status(label, kernel_factory, *args)
        if not status.available:
            return M2RNNPathCStatus(
                False,
                f"TileLang/MLX lowering failed for {status.reason}",
            )
    return M2RNNPathCStatus(
        True,
        "M2RNN TileLang DSL Path C forward/backward is dispatchable, including packed bf16 K=16, mapped production head layouts, and fused post residual/gate",
    )


def m2rnn_packed_path_c_status(
    conv_input: mx.array,
    W: mx.array,
    xf: mx.array,
    h0: mx.array | None,
    *,
    require_backward: bool = True,
) -> M2RNNPathCStatus:
    if not _msl_transform.can_run_metal():
        return M2RNNPathCStatus(False, "MLX Metal backend is not available")
    if h0 is None:
        return M2RNNPathCStatus(False, "packed M2RNN Path C requires h0")
    if not _validate_same_dtype(conv_input, W, xf, h0):
        return M2RNNPathCStatus(False, "packed M2RNN Path C inputs must share dtype")
    carrier_dtype = _tl_dtype_for(conv_input.dtype)
    if carrier_dtype is None:
        return M2RNNPathCStatus(
            False,
            f"packed M2RNN Path C unsupported dtype {conv_input.dtype}",
        )
    if not _packed_path_c_inputs_well_formed(conv_input, W, xf, h0):
        return M2RNNPathCStatus(
            False,
            "packed M2RNN Path C requires conv_input=(B,S,H*(2K+V)), "
            "W=(H,V,V), xf=(B,S,H), h0=(B,H,K,V), and matching dtype",
        )
    batch, seq, heads, k_dim, v_dim, _conv_dim = _packed_shape(conv_input, W, h0)
    if seq == 0:
        return M2RNNPathCStatus(
            True,
            "packed M2RNN Path C seq=0 is handled without launching a kernel",
        )
    fwd_status = _kernel_lowering_status(
        f"packed M2RNN Path C {carrier_dtype} K={k_dim} fwd",
        _packed_fwd_kernel_for,
        batch,
        seq,
        heads,
        k_dim,
        v_dim,
        carrier_dtype,
    )
    if not fwd_status.available:
        return fwd_status
    if require_backward:
        bwd_status = _kernel_lowering_status(
            f"packed M2RNN Path C {carrier_dtype} K={k_dim} bwd",
            _packed_bwd_kernel_for,
            batch,
            seq,
            heads,
            k_dim,
            v_dim,
            carrier_dtype,
        )
        if not bwd_status.available:
            return bwd_status
    return M2RNNPathCStatus(
        True,
        f"packed M2RNN Path C {carrier_dtype} K={k_dim} is dispatchable",
    )


def m2rnn_mapped_packed_path_c_status(
    conv_input: mx.array,
    W: mx.array,
    xf: mx.array,
    h0: mx.array | None,
    *,
    q_heads: int,
    k_heads: int,
    v_heads: int,
    require_backward: bool = True,
) -> M2RNNPathCStatus:
    if not _msl_transform.can_run_metal():
        return M2RNNPathCStatus(False, "MLX Metal backend is not available")
    if h0 is None:
        return M2RNNPathCStatus(False, "mapped packed M2RNN Path C requires h0")
    if not _validate_same_dtype(conv_input, W, xf, h0):
        return M2RNNPathCStatus(False, "mapped packed M2RNN Path C inputs must share dtype")
    carrier_dtype = _tl_dtype_for(conv_input.dtype)
    if carrier_dtype is None:
        return M2RNNPathCStatus(
            False,
            f"mapped packed M2RNN Path C unsupported dtype {conv_input.dtype}",
        )
    if _is_aligned_packed_layout(
        conv_input,
        W,
        xf,
        h0,
        q_heads=q_heads,
        k_heads=k_heads,
        v_heads=v_heads,
    ):
        return m2rnn_packed_path_c_status(
            conv_input,
            W,
            xf,
            h0,
            require_backward=require_backward,
        )
    if not _mapped_packed_path_c_inputs_well_formed(
        conv_input,
        W,
        xf,
        h0,
        q_heads=q_heads,
        k_heads=k_heads,
        v_heads=v_heads,
    ):
        return M2RNNPathCStatus(
            False,
            "mapped packed M2RNN Path C requires conv_input=(B,S,Q*K+K*K+V*Vd), "
            "W=(W,Vd,Vd), xf=(B,S,F), h0=(B,H,K,Vd), H divisible by Q/K/V/W/F, "
            "and matching dtype",
        )
    batch, seq, _conv_dim = conv_input.shape
    total_heads = h0.shape[1]
    k_dim = h0.shape[2]
    v_dim = h0.shape[3]
    w_heads = W.shape[0]
    f_heads = xf.shape[-1]
    if seq == 0:
        return M2RNNPathCStatus(
            True,
            "mapped packed M2RNN Path C seq=0 is handled without launching a kernel",
        )
    fwd_status = _kernel_lowering_status(
        f"mapped packed M2RNN Path C {carrier_dtype} K={k_dim} fwd",
        _mapped_packed_fwd_kernel_for,
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
    )
    if not fwd_status.available:
        return fwd_status
    if require_backward:
        dy_dtypes = [carrier_dtype]
        if carrier_dtype != "float32":
            dy_dtypes.append("float32")
        for dy_dtype in dy_dtypes:
            bwd_status = _kernel_lowering_status(
                f"mapped packed M2RNN Path C {carrier_dtype} K={k_dim} bwd dy={dy_dtype}",
                _mapped_packed_bwd_kernel_for,
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
                dy_dtype,
                "float32",
            )
            if not bwd_status.available:
                return bwd_status
    return M2RNNPathCStatus(
        True,
        f"mapped packed M2RNN Path C {carrier_dtype} K={k_dim} is dispatchable",
    )


def _mapped_packed_post_shape(
    conv_input: mx.array,
    W: mx.array,
    xf: mx.array,
    h0: mx.array | None,
    D: mx.array,
    projected: mx.array,
    *,
    q_heads: int,
    k_heads: int,
    v_heads: int,
    g_heads: int,
) -> tuple[int, int, int, int, int, int, int, int, int]:
    if h0 is None:
        raise ValueError("h0 is required")
    if not _mapped_packed_path_c_inputs_well_formed(
        conv_input,
        W,
        xf,
        h0,
        q_heads=q_heads,
        k_heads=k_heads,
        v_heads=v_heads,
    ):
        raise ValueError(
            "conv_input/W/xf/h0 do not match mapped packed M2RNN Path C layout"
        )
    if D.ndim != 2:
        raise ValueError(f"D must be rank 2, got shape {D.shape}")
    if projected.ndim != 3:
        raise ValueError(f"projected must be rank 3, got shape {projected.shape}")
    if not _require_positive_heads(g_heads):
        raise ValueError(f"g_heads must be positive, got {g_heads}")
    batch, seq, conv_dim = conv_input.shape
    total_heads = h0.shape[1]
    k_dim = h0.shape[2]
    v_dim = h0.shape[3]
    w_heads = W.shape[0]
    f_heads = xf.shape[-1]
    if D.shape != (total_heads, v_dim):
        raise ValueError(f"D must have shape {(total_heads, v_dim)}, got {D.shape}")
    if projected.shape[0] != batch or projected.shape[1] != seq:
        raise ValueError(
            "projected must share conv_input batch/sequence dimensions, got "
            f"{projected.shape[:2]} vs {(batch, seq)}"
        )
    if total_heads % g_heads != 0:
        raise ValueError(f"g_heads={g_heads} must divide total_heads={total_heads}")
    projected_dim = projected.shape[-1]
    if projected_dim < g_heads * v_dim:
        raise ValueError(
            f"projected width {projected_dim} is too small for g_heads*V={g_heads * v_dim}"
        )
    return batch, seq, total_heads, k_dim, v_dim, conv_dim, projected_dim, w_heads, f_heads


def _mapped_packed_post_inputs_well_formed(
    conv_input: mx.array,
    W: mx.array,
    xf: mx.array,
    h0: mx.array | None,
    D: mx.array,
    projected: mx.array,
    *,
    q_heads: int,
    k_heads: int,
    v_heads: int,
    g_heads: int,
) -> bool:
    if not _msl_transform.can_run_metal() or h0 is None:
        return False
    if not _validate_same_dtype(conv_input, W, xf, h0, D, projected):
        return False
    if _tl_dtype_for(conv_input.dtype) is None:
        return False
    try:
        _mapped_packed_post_shape(
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
    except (TypeError, ValueError):
        return False
    return True


def m2rnn_mapped_packed_post_path_c_status(
    conv_input: mx.array,
    W: mx.array,
    xf: mx.array,
    h0: mx.array | None,
    D: mx.array,
    projected: mx.array,
    *,
    q_heads: int,
    k_heads: int,
    v_heads: int,
    g_heads: int,
    require_backward: bool = True,
) -> M2RNNPathCStatus:
    if not _msl_transform.can_run_metal():
        return M2RNNPathCStatus(False, "MLX Metal backend is not available")
    if h0 is None:
        return M2RNNPathCStatus(False, "mapped packed M2RNN inline post Path C requires h0")
    if not _validate_same_dtype(conv_input, W, xf, h0, D, projected):
        return M2RNNPathCStatus(
            False,
            "mapped packed M2RNN inline post Path C inputs must share dtype",
        )
    carrier_dtype = _tl_dtype_for(conv_input.dtype)
    if carrier_dtype is None:
        return M2RNNPathCStatus(
            False,
            f"mapped packed M2RNN inline post Path C unsupported dtype {conv_input.dtype}",
        )
    try:
        batch, seq, total_heads, k_dim, v_dim, conv_dim, projected_dim, w_heads, f_heads = (
            _mapped_packed_post_shape(
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
        )
    except (TypeError, ValueError) as exc:
        return M2RNNPathCStatus(
            False,
            f"mapped packed M2RNN inline post Path C shape mismatch: {exc}",
        )
    if seq == 0:
        return M2RNNPathCStatus(
            True,
            "mapped packed M2RNN inline post Path C seq=0 is handled without launching a kernel",
        )
    fwd_status = _kernel_lowering_status(
        f"mapped packed M2RNN inline post Path C {carrier_dtype} K={k_dim} fwd",
        _mapped_packed_post_fwd_kernel_for,
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
    if not fwd_status.available:
        return fwd_status
    if require_backward:
        post_bwd_status = _kernel_lowering_status(
            f"mapped packed M2RNN inline post Path C {carrier_dtype} bwd",
            _post_residual_gate_bwd_from_recurrence_kernel_for,
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
        if not post_bwd_status.available:
            return post_bwd_status
        recurrent_bwd_status = _kernel_lowering_status(
            f"mapped packed M2RNN Path C {carrier_dtype} K={k_dim} bwd dy=float32",
            _mapped_packed_bwd_kernel_for,
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
        if not recurrent_bwd_status.available:
            return recurrent_bwd_status
    return M2RNNPathCStatus(
        True,
        f"mapped packed M2RNN inline post Path C {carrier_dtype} K={k_dim} is dispatchable",
    )


def _m2rnn_fwd_path_c_full(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    W: mx.array,
    xf: mx.array,
    h0: mx.array | None = None,
    *,
    out: M2RNNFwdOwnerOutputs | None = None,
) -> tuple[mx.array, mx.array, mx.array] | None:
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
        if out is not None:
            raise RuntimeError(
                "m2rnn_fwd_path_c owner-output route is not dispatchable "
                "for seq=0; return h0 directly instead of copying it"
            )
        return (
            mx.zeros((batch, 0, heads, v_dim), dtype=q.dtype),
            h0,
            mx.zeros((batch, 0, heads, k_dim, v_dim), dtype=q.dtype),
        )
    try:
        kernel, lowering = _fwd_kernel_for(batch, seq, heads, k_dim, v_dim, carrier_dtype)
    except Exception:
        return None

    del lowering
    if out is None:
        h_last, tanh_cache, y = kernel(q, k, v, W, xf, h0)
        return y, h_last, tanh_cache

    y, h_last, tanh_cache = _m2rnn_fwd_owner_outputs(
        out,
        batch=batch,
        seq=seq,
        heads=heads,
        k_dim=k_dim,
        v_dim=v_dim,
        dtype=q.dtype,
    )
    outputs = kernel(
        q,
        k,
        v,
        W,
        xf,
        h0,
        out=(h_last, tanh_cache, y),
    )
    if not all(
        got is expected
        for got, expected in zip(outputs, (h_last, tanh_cache, y), strict=True)
    ):
        raise RuntimeError("M2RNN Path C fwd tvm-ffi did not return caller-owned outputs")
    return y, h_last, tanh_cache


def m2rnn_fwd_path_c(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    W: mx.array,
    xf: mx.array,
    h0: mx.array | None = None,
    *,
    out: M2RNNFwdOwnerOutputs | None = None,
) -> mx.array | None:
    full = _m2rnn_fwd_path_c_full(q, k, v, W, xf, h0, out=out)
    if full is None:
        return None
    return full[0]


def m2rnn_fwd_with_state_path_c(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    W: mx.array,
    xf: mx.array,
    h0: mx.array | None = None,
    *,
    out: M2RNNFwdOwnerOutputs | None = None,
) -> tuple[mx.array, mx.array] | None:
    full = _m2rnn_fwd_path_c_full(q, k, v, W, xf, h0, out=out)
    if full is None:
        return None
    y, h_last, _ = full
    return y, h_last


def _m2rnn_bwd_path_c_kernel(
    dy: mx.array,
    q: mx.array,
    k: mx.array,
    v: mx.array,
    W: mx.array,
    xf: mx.array,
    tanh_cache: mx.array,
    h0: mx.array | None = None,
    *,
    out: M2RNNBwdOwnerOutputs | None = None,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array] | None:
    if not _msl_transform.can_run_metal():
        return None
    if h0 is None:
        return None
    if not _validate_same_dtype(q, k, v, W, xf, h0, dy, tanh_cache):
        return None
    carrier_dtype = _tl_dtype_for(q.dtype)
    if carrier_dtype is None:
        return None
    batch, seq, heads, k_dim, v_dim = _validate_inputs(q, k, v, W, xf, h0)
    if dy.shape != (batch, seq, heads, v_dim):
        raise ValueError(f"dy must be {(batch, seq, heads, v_dim)}, got {dy.shape}")
    if tanh_cache.shape != (batch, seq, heads, k_dim, v_dim):
        raise ValueError(
            "tanh_cache must be "
            f"{(batch, seq, heads, k_dim, v_dim)}, got {tanh_cache.shape}"
        )
    if seq == 0:
        if out is not None:
            raise RuntimeError(
                "m2rnn_bwd_path_c owner-output route is not dispatchable "
                "for seq=0 because no TileLang kernel runs to initialize buffers"
            )
        return (
            mx.zeros_like(q),
            mx.zeros_like(k),
            mx.zeros_like(v),
            mx.zeros_like(W),
            mx.zeros_like(xf),
            mx.zeros_like(h0),
        )
    try:
        kernel, lowering = _bwd_kernel_for(batch, seq, heads, k_dim, v_dim, carrier_dtype)
    except Exception:
        return None

    del lowering
    if out is None:
        dW_partial, dh0, dk, dq, dv, dxf, _scratch = kernel(
            dy,
            q,
            k,
            v,
            W,
            xf,
            h0,
            tanh_cache,
        )
        dW = mx.sum(dW_partial, axis=0)
        return dq, dk, dv, dW, dxf, dh0

    (
        dq,
        dk,
        dv,
        dW_partial,
        dxf,
        dh0,
        h_steps_scratch,
    ) = _m2rnn_bwd_owner_outputs(
        out,
        batch=batch,
        seq=seq,
        heads=heads,
        k_dim=k_dim,
        v_dim=v_dim,
        dtype=q.dtype,
    )
    outputs = kernel(
        dy,
        q,
        k,
        v,
        W,
        xf,
        h0,
        tanh_cache,
        out=(dW_partial, dh0, dk, dq, dv, dxf, h_steps_scratch),
    )
    if not all(
        got is expected
        for got, expected in zip(
            outputs,
            (dW_partial, dh0, dk, dq, dv, dxf, h_steps_scratch),
            strict=True,
        )
    ):
        raise RuntimeError("M2RNN Path C bwd tvm-ffi did not return caller-owned outputs")
    dW_partial, dh0, dk, dq, dv, dxf, _scratch = outputs
    dW = mx.sum(dW_partial, axis=0)
    return dq, dk, dv, dW, dxf, dh0


def m2rnn_bwd_path_c(
    dy: mx.array,
    q: mx.array,
    k: mx.array,
    v: mx.array,
    W: mx.array,
    xf: mx.array,
    tanh_cache: mx.array,
    h0: mx.array | None = None,
    *,
    force_path_c: bool = False,
    out: M2RNNBwdOwnerOutputs | None = None,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    del force_path_c
    if out is None:
        grads = _m2rnn_bwd_path_c_kernel(dy, q, k, v, W, xf, tanh_cache, h0)
    else:
        grads = _m2rnn_bwd_path_c_kernel(
            dy,
            q,
            k,
            v,
            W,
            xf,
            tanh_cache,
            h0,
            out=out,
        )
    if grads is not None:
        return grads
    raise RuntimeError(f"m2rnn_bwd_path_c unavailable: {m2rnn_path_c_status().reason}")


def _packed_shape(conv_input: mx.array, W: mx.array, h0: mx.array) -> tuple[int, int, int, int, int, int]:
    batch, seq, conv_dim = conv_input.shape
    heads, v_dim, _ = W.shape
    k_dim = h0.shape[2]
    return batch, seq, heads, k_dim, v_dim, conv_dim


def _require_positive_heads(*heads: int) -> bool:
    return all(isinstance(h, int) and h > 0 for h in heads)


def _mapped_conv_dim(q_heads: int, k_heads: int, v_heads: int, k_dim: int, v_dim: int) -> int:
    return q_heads * k_dim + k_heads * k_dim + v_heads * v_dim


def _is_aligned_packed_layout(
    conv_input: mx.array,
    W: mx.array,
    xf: mx.array,
    h0: mx.array,
    *,
    q_heads: int,
    k_heads: int,
    v_heads: int,
) -> bool:
    total_heads = h0.shape[1]
    return (
        q_heads == total_heads
        and k_heads == total_heads
        and v_heads == total_heads
        and W.shape[0] == total_heads
        and xf.shape[-1] == total_heads
        and conv_input.shape[-1] == total_heads * (2 * h0.shape[2] + h0.shape[3])
    )


def _mapped_packed_path_c_inputs_well_formed(
    conv_input: mx.array,
    W: mx.array,
    xf: mx.array,
    h0: mx.array | None,
    *,
    q_heads: int,
    k_heads: int,
    v_heads: int,
) -> bool:
    if not _msl_transform.can_run_metal() or h0 is None:
        return False
    if not _validate_same_dtype(conv_input, W, xf, h0):
        return False
    if _tl_dtype_for(conv_input.dtype) is None:
        return False
    try:
        if conv_input.ndim != 3 or W.ndim != 3 or xf.ndim != 3 or h0.ndim != 4:
            return False
        if not _require_positive_heads(q_heads, k_heads, v_heads):
            return False
        batch, seq, conv_dim = conv_input.shape
        total_heads = h0.shape[1]
        k_dim = h0.shape[2]
        v_dim = h0.shape[3]
        w_heads, w_v_dim, w_v_dim_2 = W.shape
        f_heads = xf.shape[-1]
        if w_v_dim != v_dim or w_v_dim_2 != v_dim:
            return False
        if xf.shape[0] != batch or xf.shape[1] != seq:
            return False
        if h0.shape[0] != batch:
            return False
        if not _require_positive_heads(total_heads, w_heads, f_heads):
            return False
        for heads in (q_heads, k_heads, v_heads, w_heads, f_heads):
            if total_heads % heads != 0:
                return False
        return conv_dim == _mapped_conv_dim(q_heads, k_heads, v_heads, k_dim, v_dim)
    except (TypeError, ValueError):
        return False


def _m2rnn_mapped_packed_bwd_owner_outputs(
    out: M2RNNPackedBwdOwnerOutputs | None,
    *,
    batch: int,
    seq: int,
    total_heads: int,
    w_heads: int,
    f_heads: int,
    conv_dim: int,
    k_dim: int,
    v_dim: int,
    carrier_dtype: mx.Dtype,
    grad_dtype: mx.Dtype,
) -> M2RNNPackedBwdOwnerOutputs:
    op_name = "m2rnn_mapped_packed_bwd_path_c"
    if out is None:
        return (
            mx.zeros((batch, seq, conv_dim), dtype=grad_dtype),
            mx.zeros((batch, w_heads, v_dim, v_dim), dtype=grad_dtype),
            mx.zeros((batch, seq, f_heads), dtype=grad_dtype),
            mx.zeros((batch, total_heads, k_dim, v_dim), dtype=grad_dtype),
            mx.zeros((batch, total_heads, seq, k_dim, v_dim), dtype=carrier_dtype),
        )
    if not isinstance(out, tuple) or len(out) != 5:
        raise TypeError(
            "m2rnn_mapped_packed_bwd_path_c: out must be a "
            "(dconv_input, dW_partial, dxf, dh0, h_steps_scratch) "
            "owner-output tuple"
        )
    dconv_input, dW_partial, dxf, dh0, h_steps_scratch = out
    return (
        _require_owner_array(
            op_name,
            "dconv_input",
            dconv_input,
            shape=(batch, seq, conv_dim),
            dtype=grad_dtype,
        ),
        _require_owner_array(
            op_name,
            "dW_partial",
            dW_partial,
            shape=(batch, w_heads, v_dim, v_dim),
            dtype=grad_dtype,
        ),
        _require_owner_array(
            op_name,
            "dxf",
            dxf,
            shape=(batch, seq, f_heads),
            dtype=grad_dtype,
        ),
        _require_owner_array(
            op_name,
            "dh0",
            dh0,
            shape=(batch, total_heads, k_dim, v_dim),
            dtype=grad_dtype,
        ),
        _require_owner_array(
            op_name,
            "h_steps_scratch",
            h_steps_scratch,
            shape=(batch, total_heads, seq, k_dim, v_dim),
            dtype=carrier_dtype,
        ),
    )


def _post_residual_gate_shape(
    y: mx.array,
    conv_input: mx.array,
    D: mx.array,
    projected: mx.array,
    *,
    q_heads: int,
    k_heads: int,
    v_heads: int,
    g_heads: int,
) -> tuple[int, int, int, int, int, int, int]:
    if y.ndim != 4:
        raise ValueError(f"y must be rank 4, got shape {y.shape}")
    if conv_input.ndim != 3:
        raise ValueError(f"conv_input must be rank 3, got shape {conv_input.shape}")
    if D.ndim != 2:
        raise ValueError(f"D must be rank 2, got shape {D.shape}")
    if projected.ndim != 3:
        raise ValueError(f"projected must be rank 3, got shape {projected.shape}")
    if not _require_positive_heads(q_heads, k_heads, v_heads, g_heads):
        raise ValueError("q_heads, k_heads, v_heads, and g_heads must be positive")
    batch, seq, total_heads, v_dim = y.shape
    if conv_input.shape[0] != batch or conv_input.shape[1] != seq:
        raise ValueError(
            "conv_input must share y batch/sequence dimensions, got "
            f"{conv_input.shape[:2]} vs {(batch, seq)}"
        )
    if projected.shape[0] != batch or projected.shape[1] != seq:
        raise ValueError(
            "projected must share y batch/sequence dimensions, got "
            f"{projected.shape[:2]} vs {(batch, seq)}"
        )
    if D.shape != (total_heads, v_dim):
        raise ValueError(f"D must have shape {(total_heads, v_dim)}, got {D.shape}")
    if not _require_positive_heads(total_heads):
        raise ValueError(f"total_heads must be positive, got {total_heads}")
    if total_heads % v_heads != 0:
        raise ValueError(f"v_heads={v_heads} must divide total_heads={total_heads}")
    if total_heads % g_heads != 0:
        raise ValueError(f"g_heads={g_heads} must divide total_heads={total_heads}")
    conv_dim = conv_input.shape[-1]
    v_width = v_heads * v_dim
    qk_width = conv_dim - v_width
    qk_heads = q_heads + k_heads
    if qk_width <= 0 or qk_width % qk_heads != 0:
        raise ValueError(
            "conv_input width must equal q_heads*K + k_heads*K + v_heads*V"
        )
    k_dim = qk_width // qk_heads
    if k_dim <= 0:
        raise ValueError(f"k_dim must be positive, got {k_dim}")
    if conv_dim != _mapped_conv_dim(q_heads, k_heads, v_heads, k_dim, v_dim):
        raise ValueError(
            "conv_input width must equal q_heads*K + k_heads*K + v_heads*V"
        )
    projected_dim = projected.shape[-1]
    if projected_dim < g_heads * v_dim:
        raise ValueError(
            f"projected width {projected_dim} is too small for g_heads*V={g_heads * v_dim}"
        )
    return batch, seq, total_heads, k_dim, v_dim, conv_dim, projected_dim


def _post_residual_gate_inputs_well_formed(
    y: mx.array,
    conv_input: mx.array,
    D: mx.array,
    projected: mx.array,
    *,
    q_heads: int,
    k_heads: int,
    v_heads: int,
    g_heads: int,
) -> bool:
    if not _msl_transform.can_run_metal():
        return False
    if not _validate_same_dtype(y, conv_input, D, projected):
        return False
    if _tl_dtype_for(y.dtype) is None:
        return False
    try:
        _post_residual_gate_shape(
            y,
            conv_input,
            D,
            projected,
            q_heads=q_heads,
            k_heads=k_heads,
            v_heads=v_heads,
            g_heads=g_heads,
        )
    except (TypeError, ValueError):
        return False
    return True


def _m2rnn_post_bwd_owner_outputs(
    out: M2RNNPostBwdOwnerOutputs | None,
    *,
    batch: int,
    seq: int,
    total_heads: int,
    v_dim: int,
    conv_dim: int,
    projected_dim: int,
    grad_dtype: mx.Dtype,
) -> M2RNNPostBwdOwnerOutputs:
    op_name = "m2rnn_post_residual_gate_bwd_path_c"
    if out is None:
        return (
            mx.zeros((batch, seq, total_heads, v_dim), dtype=grad_dtype),
            mx.zeros((batch, seq, conv_dim), dtype=grad_dtype),
            mx.zeros((total_heads, v_dim), dtype=grad_dtype),
            mx.zeros((batch, seq, projected_dim), dtype=grad_dtype),
        )
    if not isinstance(out, tuple) or len(out) != 4:
        raise TypeError(
            "m2rnn_post_residual_gate_bwd_path_c: out must be a "
            "(dy_recurrent, dconv_input, dD, dprojected) owner-output tuple"
        )
    names = ("dy_recurrent", "dconv_input", "dD", "dprojected")
    expected_shapes = (
        (batch, seq, total_heads, v_dim),
        (batch, seq, conv_dim),
        (total_heads, v_dim),
        (batch, seq, projected_dim),
    )
    return cast(
        M2RNNPostBwdOwnerOutputs,
        tuple(
            _require_owner_array(op_name, name, array, shape=shape, dtype=grad_dtype)
            for name, array, shape in zip(names, out, expected_shapes, strict=True)
        ),
    )


def m2rnn_post_residual_gate_path_c_status(
    y: mx.array,
    conv_input: mx.array,
    D: mx.array,
    projected: mx.array,
    *,
    q_heads: int,
    k_heads: int,
    v_heads: int,
    g_heads: int,
    require_backward: bool = True,
) -> M2RNNPathCStatus:
    if not _msl_transform.can_run_metal():
        return M2RNNPathCStatus(False, "MLX Metal backend is not available")
    if not _validate_same_dtype(y, conv_input, D, projected):
        return M2RNNPathCStatus(
            False,
            "M2RNN post residual/gate Path C inputs must share dtype",
        )
    carrier_dtype = _tl_dtype_for(y.dtype)
    if carrier_dtype is None:
        return M2RNNPathCStatus(
            False,
            f"M2RNN post residual/gate Path C unsupported dtype {y.dtype}",
        )
    try:
        batch, seq, total_heads, k_dim, v_dim, conv_dim, projected_dim = (
            _post_residual_gate_shape(
                y,
                conv_input,
                D,
                projected,
                q_heads=q_heads,
                k_heads=k_heads,
                v_heads=v_heads,
                g_heads=g_heads,
            )
        )
    except (TypeError, ValueError) as exc:
        return M2RNNPathCStatus(
            False,
            f"M2RNN post residual/gate Path C shape mismatch: {exc}",
        )
    if seq == 0:
        return M2RNNPathCStatus(
            True,
            "M2RNN post residual/gate Path C seq=0 is handled without launching a kernel",
        )
    fwd_status = _kernel_lowering_status(
        f"M2RNN post residual/gate Path C {carrier_dtype} fwd",
        _post_residual_gate_fwd_kernel_for,
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
    if not fwd_status.available:
        return fwd_status
    if require_backward:
        bwd_status = _kernel_lowering_status(
            f"M2RNN post residual/gate Path C {carrier_dtype} bwd",
            _post_residual_gate_bwd_kernel_for,
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
        if not bwd_status.available:
            return bwd_status
    return M2RNNPathCStatus(
        True,
        f"M2RNN post residual/gate Path C {carrier_dtype} is dispatchable",
    )


def _m2rnn_packed_fwd_path_c_full(
    conv_input: mx.array,
    W: mx.array,
    xf: mx.array,
    h0: mx.array | None = None,
    *,
    out: M2RNNFwdOwnerOutputs | None = None,
) -> tuple[mx.array, mx.array, mx.array] | None:
    if h0 is None or not _packed_path_c_inputs_eligible(conv_input, W, xf, h0):
        return None
    carrier_dtype = _tl_dtype_for(conv_input.dtype)
    if carrier_dtype is None:
        return None
    batch, seq, heads, k_dim, v_dim, _conv_dim = _packed_shape(conv_input, W, h0)
    if seq == 0:
        if out is not None:
            raise RuntimeError(
                "m2rnn_packed_fwd_path_c owner-output route is not dispatchable "
                "for seq=0; return h0 directly instead of copying it"
            )
        return (
            mx.zeros((batch, 0, heads, v_dim), dtype=conv_input.dtype),
            h0,
            mx.zeros((batch, 0, heads, k_dim, v_dim), dtype=conv_input.dtype),
        )
    try:
        kernel, lowering = _packed_fwd_kernel_for(
            batch,
            seq,
            heads,
            k_dim,
            v_dim,
            carrier_dtype,
        )
    except Exception:
        return None

    del lowering
    if out is None:
        h_last, tanh_cache, y = kernel(conv_input, W, xf, h0)
        return y, h_last, tanh_cache

    y, h_last, tanh_cache = _m2rnn_fwd_owner_outputs(
        out,
        batch=batch,
        seq=seq,
        heads=heads,
        k_dim=k_dim,
        v_dim=v_dim,
        dtype=conv_input.dtype,
    )
    outputs = kernel(
        conv_input,
        W,
        xf,
        h0,
        out=(h_last, tanh_cache, y),
    )
    if not all(
        got is expected
        for got, expected in zip(outputs, (h_last, tanh_cache, y), strict=True)
    ):
        raise RuntimeError("Packed M2RNN Path C fwd tvm-ffi did not return caller-owned outputs")
    return y, h_last, tanh_cache


def _m2rnn_packed_bwd_path_c_kernel(
    dy: mx.array,
    conv_input: mx.array,
    W: mx.array,
    xf: mx.array,
    tanh_cache: mx.array,
    h0: mx.array | None = None,
    *,
    out: M2RNNPackedBwdOwnerOutputs | None = None,
) -> tuple[mx.array, mx.array, mx.array, mx.array] | None:
    if h0 is None or not _packed_path_c_inputs_eligible(conv_input, W, xf, h0):
        return None
    if not _validate_same_dtype(conv_input, W, xf, h0, dy, tanh_cache):
        return None
    carrier_dtype = _tl_dtype_for(conv_input.dtype)
    if carrier_dtype is None:
        return None
    batch, seq, heads, k_dim, v_dim, conv_dim = _packed_shape(conv_input, W, h0)
    if dy.shape != (batch, seq, heads, v_dim):
        raise ValueError(f"dy must be {(batch, seq, heads, v_dim)}, got {dy.shape}")
    if tanh_cache.shape != (batch, seq, heads, k_dim, v_dim):
        raise ValueError(
            "tanh_cache must be "
            f"{(batch, seq, heads, k_dim, v_dim)}, got {tanh_cache.shape}"
        )
    if seq == 0:
        if out is not None:
            raise RuntimeError(
                "m2rnn_packed_bwd_path_c owner-output route is not dispatchable "
                "for seq=0 because no TileLang kernel runs to initialize buffers"
            )
        return (
            mx.zeros_like(conv_input),
            mx.zeros_like(W),
            mx.zeros_like(xf),
            mx.zeros_like(h0),
        )
    try:
        kernel, lowering = _packed_bwd_kernel_for(
            batch,
            seq,
            heads,
            k_dim,
            v_dim,
            carrier_dtype,
        )
    except Exception:
        return None

    del lowering
    if out is None:
        dconv_input, dW_partial, dxf, dh0, _scratch = kernel(
            dy,
            conv_input,
            W,
            xf,
            h0,
            tanh_cache,
        )
        dW = mx.sum(dW_partial, axis=0)
        return dconv_input, dW, dxf, dh0

    (
        dconv_input,
        dW_partial,
        dxf,
        dh0,
        h_steps_scratch,
    ) = _m2rnn_packed_bwd_owner_outputs(
        out,
        batch=batch,
        seq=seq,
        heads=heads,
        conv_dim=conv_dim,
        k_dim=k_dim,
        v_dim=v_dim,
        dtype=conv_input.dtype,
    )
    outputs = kernel(
        dy,
        conv_input,
        W,
        xf,
        h0,
        tanh_cache,
        out=(dconv_input, dW_partial, dxf, dh0, h_steps_scratch),
    )
    if not all(
        got is expected
        for got, expected in zip(
            outputs,
            (dconv_input, dW_partial, dxf, dh0, h_steps_scratch),
            strict=True,
        )
    ):
        raise RuntimeError("Packed M2RNN Path C bwd tvm-ffi did not return caller-owned outputs")
    dconv_input, dW_partial, dxf, dh0, _scratch = outputs
    dW = mx.sum(dW_partial, axis=0)
    return dconv_input, dW, dxf, dh0


def m2rnn_packed_bwd_path_c(
    dy: mx.array,
    conv_input: mx.array,
    W: mx.array,
    xf: mx.array,
    tanh_cache: mx.array,
    h0: mx.array | None = None,
    *,
    out: M2RNNPackedBwdOwnerOutputs | None = None,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    grads = _m2rnn_packed_bwd_path_c_kernel(
        dy,
        conv_input,
        W,
        xf,
        tanh_cache,
        h0,
        out=out,
    )
    if grads is not None:
        return grads
    raise RuntimeError(f"m2rnn_packed_bwd_path_c unavailable: {m2rnn_path_c_status().reason}")


def _m2rnn_mapped_packed_fwd_path_c_full(
    conv_input: mx.array,
    W: mx.array,
    xf: mx.array,
    h0: mx.array | None = None,
    *,
    q_heads: int,
    k_heads: int,
    v_heads: int,
    out: M2RNNFwdOwnerOutputs | None = None,
) -> tuple[mx.array, mx.array, mx.array] | None:
    if h0 is None:
        return None
    if _is_aligned_packed_layout(
        conv_input,
        W,
        xf,
        h0,
        q_heads=q_heads,
        k_heads=k_heads,
        v_heads=v_heads,
    ):
        return _m2rnn_packed_fwd_path_c_full(conv_input, W, xf, h0, out=out)
    if not m2rnn_mapped_packed_path_c_status(
        conv_input,
        W,
        xf,
        h0,
        q_heads=q_heads,
        k_heads=k_heads,
        v_heads=v_heads,
        require_backward=False,
    ).available:
        return None
    carrier_dtype = _tl_dtype_for(conv_input.dtype)
    if carrier_dtype is None:
        return None
    batch, seq, conv_dim = conv_input.shape
    total_heads = h0.shape[1]
    k_dim = h0.shape[2]
    v_dim = h0.shape[3]
    w_heads = W.shape[0]
    f_heads = xf.shape[-1]
    if conv_dim != _mapped_conv_dim(q_heads, k_heads, v_heads, k_dim, v_dim):
        return None
    if seq == 0:
        if out is not None:
            raise RuntimeError(
                "m2rnn_mapped_packed_fwd_path_c owner-output route is not dispatchable "
                "for seq=0; return h0 directly instead of copying it"
            )
        return (
            mx.zeros((batch, 0, total_heads, v_dim), dtype=conv_input.dtype),
            h0,
            mx.zeros((batch, 0, total_heads, k_dim, v_dim), dtype=conv_input.dtype),
        )
    try:
        kernel, lowering = _mapped_packed_fwd_kernel_for(
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
        )
    except Exception:
        return None

    del lowering
    if out is None:
        h_last, tanh_cache, y = kernel(conv_input, W, xf, h0)
        return y, h_last, tanh_cache

    y, h_last, tanh_cache = _m2rnn_fwd_owner_outputs(
        out,
        batch=batch,
        seq=seq,
        heads=total_heads,
        k_dim=k_dim,
        v_dim=v_dim,
        dtype=conv_input.dtype,
    )
    outputs = kernel(
        conv_input,
        W,
        xf,
        h0,
        out=(h_last, tanh_cache, y),
    )
    if not all(
        got is expected
        for got, expected in zip(outputs, (h_last, tanh_cache, y), strict=True)
    ):
        raise RuntimeError("Mapped packed M2RNN Path C fwd tvm-ffi did not return caller-owned outputs")
    return y, h_last, tanh_cache


def _m2rnn_mapped_packed_bwd_path_c_kernel(
    dy: mx.array,
    conv_input: mx.array,
    W: mx.array,
    xf: mx.array,
    tanh_cache: mx.array,
    h0: mx.array | None = None,
    *,
    q_heads: int,
    k_heads: int,
    v_heads: int,
    out: M2RNNPackedBwdOwnerOutputs | None = None,
) -> tuple[mx.array, mx.array, mx.array, mx.array] | None:
    if h0 is None:
        return None
    if _is_aligned_packed_layout(
        conv_input,
        W,
        xf,
        h0,
        q_heads=q_heads,
        k_heads=k_heads,
        v_heads=v_heads,
    ):
        return _m2rnn_packed_bwd_path_c_kernel(dy, conv_input, W, xf, tanh_cache, h0, out=out)
    if not m2rnn_mapped_packed_path_c_status(
        conv_input,
        W,
        xf,
        h0,
        q_heads=q_heads,
        k_heads=k_heads,
        v_heads=v_heads,
        require_backward=True,
    ).available:
        return None
    if not _validate_same_dtype(conv_input, W, xf, h0, tanh_cache):
        return None
    carrier_dtype = _tl_dtype_for(conv_input.dtype)
    if carrier_dtype is None:
        return None
    dy_dtype = _tl_dtype_for(dy.dtype)
    if dy_dtype is None:
        return None
    batch, seq, conv_dim = conv_input.shape
    total_heads = h0.shape[1]
    k_dim = h0.shape[2]
    v_dim = h0.shape[3]
    w_heads = W.shape[0]
    f_heads = xf.shape[-1]
    if dy.shape != (batch, seq, total_heads, v_dim):
        raise ValueError(f"dy must be {(batch, seq, total_heads, v_dim)}, got {dy.shape}")
    if tanh_cache.shape != (batch, seq, total_heads, k_dim, v_dim):
        raise ValueError(
            "tanh_cache must be "
            f"{(batch, seq, total_heads, k_dim, v_dim)}, got {tanh_cache.shape}"
        )
    if seq == 0:
        if out is not None:
            raise RuntimeError(
                "m2rnn_mapped_packed_bwd_path_c owner-output route is not dispatchable "
                "for seq=0 because no TileLang kernel runs to initialize buffers"
            )
        return (
            mx.zeros(conv_input.shape, dtype=mx.float32),
            mx.zeros(W.shape, dtype=mx.float32),
            mx.zeros(xf.shape, dtype=mx.float32),
            mx.zeros(h0.shape, dtype=mx.float32),
        )
    try:
        kernel, lowering = _mapped_packed_bwd_kernel_for(
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
            dy_dtype,
            "float32",
        )
    except Exception:
        return None

    del lowering
    (
        dconv_input,
        dW_partial,
        dxf,
        dh0,
        h_steps_scratch,
    ) = _m2rnn_mapped_packed_bwd_owner_outputs(
        out,
        batch=batch,
        seq=seq,
        total_heads=total_heads,
        w_heads=w_heads,
        f_heads=f_heads,
        conv_dim=conv_dim,
        k_dim=k_dim,
        v_dim=v_dim,
        carrier_dtype=conv_input.dtype,
        grad_dtype=mx.float32,
    )
    outputs = kernel(
        dy,
        conv_input,
        W,
        xf,
        h0,
        tanh_cache,
        out=(dconv_input, dW_partial, dxf, dh0, h_steps_scratch),
    )
    if not all(
        got is expected
        for got, expected in zip(
            outputs,
            (dconv_input, dW_partial, dxf, dh0, h_steps_scratch),
            strict=True,
        )
    ):
        raise RuntimeError("Mapped packed M2RNN Path C bwd tvm-ffi did not return caller-owned outputs")
    dconv_input, dW_partial, dxf, dh0, _scratch = outputs
    dW = mx.sum(dW_partial, axis=0)
    return dconv_input, dW, dxf, dh0


def m2rnn_mapped_packed_bwd_path_c(
    dy: mx.array,
    conv_input: mx.array,
    W: mx.array,
    xf: mx.array,
    tanh_cache: mx.array,
    h0: mx.array | None = None,
    *,
    q_heads: int,
    k_heads: int,
    v_heads: int,
    out: M2RNNPackedBwdOwnerOutputs | None = None,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    grads = _m2rnn_mapped_packed_bwd_path_c_kernel(
        dy,
        conv_input,
        W,
        xf,
        tanh_cache,
        h0,
        q_heads=q_heads,
        k_heads=k_heads,
        v_heads=v_heads,
        out=out,
    )
    if grads is not None:
        return grads
    status = m2rnn_mapped_packed_path_c_status(
        conv_input,
        W,
        xf,
        h0,
        q_heads=q_heads,
        k_heads=k_heads,
        v_heads=v_heads,
    )
    raise RuntimeError(f"m2rnn_mapped_packed_bwd_path_c unavailable: {status.reason}")


def _m2rnn_mapped_packed_post_fwd_path_c_full(
    conv_input: mx.array,
    W: mx.array,
    xf: mx.array,
    h0: mx.array | None,
    D: mx.array,
    projected: mx.array,
    *,
    q_heads: int,
    k_heads: int,
    v_heads: int,
    g_heads: int,
) -> M2RNNPackedPostFwdOutputs | None:
    if h0 is None:
        return None
    if not _mapped_packed_post_inputs_well_formed(
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
    ):
        return None
    carrier_dtype = _tl_dtype_for(conv_input.dtype)
    if carrier_dtype is None:
        return None
    batch, seq, total_heads, k_dim, v_dim, _conv_dim, projected_dim, w_heads, f_heads = (
        _mapped_packed_post_shape(
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
    )
    if seq == 0:
        return (
            mx.zeros((batch, 0, total_heads * v_dim), dtype=conv_input.dtype),
            h0,
            mx.zeros((batch, 0, total_heads, k_dim, v_dim), dtype=conv_input.dtype),
        )
    try:
        kernel, lowering = _mapped_packed_post_fwd_kernel_for(
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
    except Exception:
        return None

    del lowering
    h_last, tanh_cache, post = kernel(conv_input, W, xf, h0, D, projected)
    return post, h_last, tanh_cache


def _m2rnn_inline_post_bwd_path_c_kernel(
    dpost: mx.array,
    conv_input: mx.array,
    xf: mx.array,
    h0: mx.array | None,
    tanh_cache: mx.array,
    D: mx.array,
    projected: mx.array,
    *,
    q_heads: int,
    k_heads: int,
    v_heads: int,
    g_heads: int,
    out: M2RNNPostBwdOwnerOutputs | None = None,
) -> tuple[mx.array, mx.array, mx.array, mx.array] | None:
    if h0 is None:
        return None
    if not _msl_transform.can_run_metal():
        return None
    if not _validate_same_dtype(conv_input, xf, h0, D, projected):
        return None
    if _tl_dtype_for(conv_input.dtype) is None:
        return None
    if dpost.dtype != conv_input.dtype:
        return None
    carrier_dtype = _tl_dtype_for(conv_input.dtype)
    if carrier_dtype is None:
        return None
    try:
        if conv_input.ndim != 3 or xf.ndim != 3 or h0.ndim != 4:
            return None
        if D.ndim != 2 or projected.ndim != 3:
            return None
        if not _require_positive_heads(q_heads, k_heads, v_heads, g_heads):
            return None
        batch, seq, conv_dim = conv_input.shape
        total_heads = h0.shape[1]
        k_dim = h0.shape[2]
        v_dim = h0.shape[3]
        f_heads = xf.shape[-1]
        projected_dim = projected.shape[-1]
        if h0.shape[0] != batch:
            return None
        if xf.shape[0] != batch or xf.shape[1] != seq:
            return None
        if projected.shape[0] != batch or projected.shape[1] != seq:
            return None
        if D.shape != (total_heads, v_dim):
            return None
        if not _require_positive_heads(total_heads, f_heads):
            return None
        for heads in (q_heads, k_heads, v_heads, g_heads, f_heads):
            if total_heads % heads != 0:
                return None
        if conv_dim != _mapped_conv_dim(q_heads, k_heads, v_heads, k_dim, v_dim):
            return None
        if projected_dim < g_heads * v_dim:
            return None
    except (TypeError, ValueError):
        return None
    if dpost.shape != (batch, seq, total_heads * v_dim):
        raise ValueError(
            f"dpost must be {(batch, seq, total_heads * v_dim)}, got {dpost.shape}"
        )
    if tanh_cache.shape != (batch, seq, total_heads, k_dim, v_dim):
        raise ValueError(
            "tanh_cache must be "
            f"{(batch, seq, total_heads, k_dim, v_dim)}, got {tanh_cache.shape}"
        )
    if seq == 0:
        if out is not None:
            raise RuntimeError(
                "m2rnn_inline_post_bwd_path_c owner-output route is not dispatchable "
                "for seq=0 because no TileLang kernel runs to initialize buffers"
            )
        return (
            mx.zeros((batch, seq, total_heads, v_dim), dtype=mx.float32),
            mx.zeros(conv_input.shape, dtype=mx.float32),
            mx.zeros(D.shape, dtype=mx.float32),
            mx.zeros(projected.shape, dtype=mx.float32),
        )
    try:
        kernel, lowering = _post_residual_gate_bwd_from_recurrence_kernel_for(
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
    except Exception:
        return None

    del lowering
    dy_recurrent, dconv_input, dD, dprojected = _m2rnn_post_bwd_owner_outputs(
        out,
        batch=batch,
        seq=seq,
        total_heads=total_heads,
        v_dim=v_dim,
        conv_dim=conv_dim,
        projected_dim=projected_dim,
        grad_dtype=mx.float32,
    )
    outputs = kernel(
        dpost,
        conv_input,
        xf,
        h0,
        tanh_cache,
        D,
        projected,
        out=(dy_recurrent, dconv_input, dD, dprojected),
    )
    if not all(
        got is expected
        for got, expected in zip(
            outputs,
            (dy_recurrent, dconv_input, dD, dprojected),
            strict=True,
        )
    ):
        raise RuntimeError(
            "M2RNN inline post residual/gate Path C bwd tvm-ffi did not return "
            "caller-owned outputs"
        )
    return outputs


def _m2rnn_post_residual_gate_fwd_path_c(
    y: mx.array,
    conv_input: mx.array,
    D: mx.array,
    projected: mx.array,
    *,
    q_heads: int,
    k_heads: int,
    v_heads: int,
    g_heads: int,
) -> mx.array | None:
    if not _post_residual_gate_inputs_well_formed(
        y,
        conv_input,
        D,
        projected,
        q_heads=q_heads,
        k_heads=k_heads,
        v_heads=v_heads,
        g_heads=g_heads,
    ):
        return None
    carrier_dtype = _tl_dtype_for(y.dtype)
    if carrier_dtype is None:
        return None
    batch, seq, total_heads, k_dim, v_dim, conv_dim, projected_dim = (
        _post_residual_gate_shape(
            y,
            conv_input,
            D,
            projected,
            q_heads=q_heads,
            k_heads=k_heads,
            v_heads=v_heads,
            g_heads=g_heads,
        )
    )
    if seq == 0:
        return mx.zeros((batch, 0, total_heads * v_dim), dtype=y.dtype)
    try:
        kernel, lowering = _post_residual_gate_fwd_kernel_for(
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
    except Exception:
        return None

    del lowering
    return kernel(y, conv_input, D, projected)


def _m2rnn_post_residual_gate_bwd_path_c_kernel(
    dpost: mx.array,
    y: mx.array,
    conv_input: mx.array,
    D: mx.array,
    projected: mx.array,
    *,
    q_heads: int,
    k_heads: int,
    v_heads: int,
    g_heads: int,
    out: M2RNNPostBwdOwnerOutputs | None = None,
) -> tuple[mx.array, mx.array, mx.array, mx.array] | None:
    if not _post_residual_gate_inputs_well_formed(
        y,
        conv_input,
        D,
        projected,
        q_heads=q_heads,
        k_heads=k_heads,
        v_heads=v_heads,
        g_heads=g_heads,
    ):
        return None
    if dpost.dtype != y.dtype:
        return None
    carrier_dtype = _tl_dtype_for(y.dtype)
    if carrier_dtype is None:
        return None
    batch, seq, total_heads, k_dim, v_dim, conv_dim, projected_dim = (
        _post_residual_gate_shape(
            y,
            conv_input,
            D,
            projected,
            q_heads=q_heads,
            k_heads=k_heads,
            v_heads=v_heads,
            g_heads=g_heads,
        )
    )
    if dpost.shape != (batch, seq, total_heads * v_dim):
        raise ValueError(
            f"dpost must be {(batch, seq, total_heads * v_dim)}, got {dpost.shape}"
        )
    if seq == 0:
        if out is not None:
            raise RuntimeError(
                "m2rnn_post_residual_gate_bwd_path_c owner-output route is not "
                "dispatchable for seq=0 because no TileLang kernel runs to initialize buffers"
            )
        return (
            mx.zeros(y.shape, dtype=mx.float32),
            mx.zeros(conv_input.shape, dtype=mx.float32),
            mx.zeros(D.shape, dtype=mx.float32),
            mx.zeros(projected.shape, dtype=mx.float32),
        )
    try:
        kernel, lowering = _post_residual_gate_bwd_kernel_for(
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
    except Exception:
        return None

    del lowering
    dy_recurrent, dconv_input, dD, dprojected = _m2rnn_post_bwd_owner_outputs(
        out,
        batch=batch,
        seq=seq,
        total_heads=total_heads,
        v_dim=v_dim,
        conv_dim=conv_dim,
        projected_dim=projected_dim,
        grad_dtype=mx.float32,
    )
    outputs = kernel(
        dpost,
        y,
        conv_input,
        D,
        projected,
        out=(dy_recurrent, dconv_input, dD, dprojected),
    )
    if not all(
        got is expected
        for got, expected in zip(
            outputs,
            (dy_recurrent, dconv_input, dD, dprojected),
            strict=True,
        )
    ):
        raise RuntimeError(
            "M2RNN post residual/gate Path C bwd tvm-ffi did not return "
            "caller-owned outputs"
        )
    return outputs


def m2rnn_post_residual_gate_bwd_path_c(
    dpost: mx.array,
    y: mx.array,
    conv_input: mx.array,
    D: mx.array,
    projected: mx.array,
    *,
    q_heads: int,
    k_heads: int,
    v_heads: int,
    g_heads: int,
    out: M2RNNPostBwdOwnerOutputs | None = None,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    grads = _m2rnn_post_residual_gate_bwd_path_c_kernel(
        dpost,
        y,
        conv_input,
        D,
        projected,
        q_heads=q_heads,
        k_heads=k_heads,
        v_heads=v_heads,
        g_heads=g_heads,
        out=out,
    )
    if grads is not None:
        return grads
    status = m2rnn_post_residual_gate_path_c_status(
        y,
        conv_input,
        D,
        projected,
        q_heads=q_heads,
        k_heads=k_heads,
        v_heads=v_heads,
        g_heads=g_heads,
    )
    raise RuntimeError(f"m2rnn_post_residual_gate_bwd_path_c unavailable: {status.reason}")


def _raise_path_c_unavailable() -> None:
    raise RuntimeError(f"m2rnn_apply_path_c unavailable: {m2rnn_path_c_status().reason}")


def _raise_packed_path_c_unavailable(
    conv_input: mx.array,
    W: mx.array,
    xf: mx.array,
    h0: mx.array | None,
) -> None:
    status = m2rnn_packed_path_c_status(
        conv_input,
        W,
        xf,
        h0,
        require_backward=False,
    )
    raise RuntimeError(
        f"m2rnn_apply_packed_with_state_path_c unavailable: {status.reason}"
    )


def _raise_mapped_packed_path_c_unavailable(
    conv_input: mx.array,
    W: mx.array,
    xf: mx.array,
    h0: mx.array | None,
    *,
    q_heads: int,
    k_heads: int,
    v_heads: int,
) -> None:
    status = m2rnn_mapped_packed_path_c_status(
        conv_input,
        W,
        xf,
        h0,
        q_heads=q_heads,
        k_heads=k_heads,
        v_heads=v_heads,
        require_backward=False,
    )
    raise RuntimeError(
        f"m2rnn_apply_mapped_packed_with_state_path_c unavailable: {status.reason}"
    )


def _raise_mapped_packed_post_path_c_unavailable(
    conv_input: mx.array,
    W: mx.array,
    xf: mx.array,
    h0: mx.array | None,
    D: mx.array,
    projected: mx.array,
    *,
    q_heads: int,
    k_heads: int,
    v_heads: int,
    g_heads: int,
) -> None:
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
        require_backward=False,
    )
    raise RuntimeError(
        "m2rnn_apply_mapped_packed_post_with_state_path_c unavailable: "
        f"{status.reason}"
    )


def _raise_post_residual_gate_path_c_unavailable(
    y: mx.array,
    conv_input: mx.array,
    D: mx.array,
    projected: mx.array,
    *,
    q_heads: int,
    k_heads: int,
    v_heads: int,
    g_heads: int,
) -> None:
    status = m2rnn_post_residual_gate_path_c_status(
        y,
        conv_input,
        D,
        projected,
        q_heads=q_heads,
        k_heads=k_heads,
        v_heads=v_heads,
        g_heads=g_heads,
        require_backward=False,
    )
    raise RuntimeError(
        f"m2rnn_apply_post_residual_gate_path_c unavailable: {status.reason}"
    )


def _match_primal_gradient_dtypes(
    grads: tuple[mx.array, ...],
    primals: tuple[mx.array, ...],
) -> tuple[mx.array, ...]:
    return tuple(
        grad if grad.dtype == primal.dtype else grad.astype(primal.dtype)
        for grad, primal in zip(grads, primals, strict=True)
    )


@mx.custom_function
def m2rnn_apply_packed_with_state_path_c(
    conv_input: mx.array,
    W: mx.array,
    xf: mx.array,
    h0: mx.array,
) -> tuple[mx.array, mx.array]:
    full = _m2rnn_packed_fwd_path_c_full(conv_input, W, xf, h0)
    if full is None:
        _raise_packed_path_c_unavailable(conv_input, W, xf, h0)
    y, h_last, _tanh_cache = full
    return y, h_last


@m2rnn_apply_packed_with_state_path_c.vjp
def _m2rnn_apply_packed_with_state_path_c_vjp(
    primals: tuple[mx.array, ...],
    cotangent: tuple[mx.array, mx.array],
    output: tuple[mx.array, mx.array],
) -> tuple[mx.array, ...]:
    del output
    conv_input, W, xf, h0 = primals
    dy = cotangent[0]
    full = _m2rnn_packed_fwd_path_c_full(conv_input, W, xf, h0)
    if full is None:
        _raise_packed_path_c_unavailable(conv_input, W, xf, h0)
    _y, _h_last, tanh_cache = full
    grads = m2rnn_packed_bwd_path_c(
        dy,
        conv_input,
        W,
        xf,
        tanh_cache,
        h0,
    )
    return _match_primal_gradient_dtypes(grads, primals)


@lru_cache(maxsize=128)
def _mapped_packed_apply_for_layout(q_heads: int, k_heads: int, v_heads: int) -> Any:
    @mx.custom_function
    def _apply(
        conv_input: mx.array,
        W: mx.array,
        xf: mx.array,
        h0: mx.array,
    ) -> tuple[mx.array, mx.array]:
        full = _m2rnn_mapped_packed_fwd_path_c_full(
            conv_input,
            W,
            xf,
            h0,
            q_heads=q_heads,
            k_heads=k_heads,
            v_heads=v_heads,
        )
        if full is None:
            _raise_mapped_packed_path_c_unavailable(
                conv_input,
                W,
                xf,
                h0,
                q_heads=q_heads,
                k_heads=k_heads,
                v_heads=v_heads,
            )
        y, h_last, _tanh_cache = full
        return y, h_last

    apply_any = cast(Any, _apply)

    @apply_any.vjp
    def _apply_vjp(
        primals: tuple[mx.array, ...],
        cotangent: tuple[mx.array, mx.array],
        output: tuple[mx.array, mx.array],
    ) -> tuple[mx.array, ...]:
        del output
        conv_input, W, xf, h0 = primals
        dy = cotangent[0]
        full = _m2rnn_mapped_packed_fwd_path_c_full(
            conv_input,
            W,
            xf,
            h0,
            q_heads=q_heads,
            k_heads=k_heads,
            v_heads=v_heads,
        )
        if full is None:
            _raise_mapped_packed_path_c_unavailable(
                conv_input,
                W,
                xf,
                h0,
                q_heads=q_heads,
                k_heads=k_heads,
                v_heads=v_heads,
            )
        _y, _h_last, tanh_cache = full
        grads = m2rnn_mapped_packed_bwd_path_c(
            dy,
            conv_input,
            W,
            xf,
            tanh_cache,
            h0,
            q_heads=q_heads,
            k_heads=k_heads,
            v_heads=v_heads,
        )
        return _match_primal_gradient_dtypes(grads, primals)

    return _apply


def m2rnn_apply_mapped_packed_with_state_path_c(
    conv_input: mx.array,
    W: mx.array,
    xf: mx.array,
    h0: mx.array,
    *,
    q_heads: int,
    k_heads: int,
    v_heads: int,
) -> tuple[mx.array, mx.array]:
    if _is_aligned_packed_layout(
        conv_input,
        W,
        xf,
        h0,
        q_heads=q_heads,
        k_heads=k_heads,
        v_heads=v_heads,
    ):
        return m2rnn_apply_packed_with_state_path_c(conv_input, W, xf, h0)
    apply = _mapped_packed_apply_for_layout(int(q_heads), int(k_heads), int(v_heads))
    return apply(conv_input, W, xf, h0)


@lru_cache(maxsize=128)
def _mapped_packed_post_apply_for_layout(
    q_heads: int,
    k_heads: int,
    v_heads: int,
    g_heads: int,
) -> Any:
    @mx.custom_function
    def _apply(
        conv_input: mx.array,
        W: mx.array,
        xf: mx.array,
        h0: mx.array,
        D: mx.array,
        projected: mx.array,
    ) -> tuple[mx.array, mx.array]:
        full = _m2rnn_mapped_packed_post_fwd_path_c_full(
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
        if full is None:
            _raise_mapped_packed_post_path_c_unavailable(
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
        post, h_last, _tanh_cache = full
        return post, h_last

    apply_any = cast(Any, _apply)

    @apply_any.vjp
    def _apply_vjp(
        primals: tuple[mx.array, ...],
        cotangent: tuple[mx.array, mx.array],
        output: tuple[mx.array, mx.array],
    ) -> tuple[mx.array, ...]:
        del output
        conv_input, W, xf, h0, D, projected = primals
        dpost = cotangent[0]
        full = _m2rnn_mapped_packed_post_fwd_path_c_full(
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
        if full is None:
            _raise_mapped_packed_post_path_c_unavailable(
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
        _post, _h_last, tanh_cache = full
        post_grads = _m2rnn_inline_post_bwd_path_c_kernel(
            dpost,
            conv_input,
            xf,
            h0,
            tanh_cache,
            D,
            projected,
            q_heads=q_heads,
            k_heads=k_heads,
            v_heads=v_heads,
            g_heads=g_heads,
        )
        if post_grads is None:
            _raise_mapped_packed_post_path_c_unavailable(
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
        dy_recurrent, dconv_post, dD, dprojected = post_grads
        dconv_recurrent, dW, dxf, dh0 = m2rnn_mapped_packed_bwd_path_c(
            dy_recurrent,
            conv_input,
            W,
            xf,
            tanh_cache,
            h0,
            q_heads=q_heads,
            k_heads=k_heads,
            v_heads=v_heads,
        )
        grads = (
            dconv_recurrent + dconv_post,
            dW,
            dxf,
            dh0,
            dD,
            dprojected,
        )
        return _match_primal_gradient_dtypes(grads, primals)

    return _apply


def m2rnn_apply_mapped_packed_post_with_state_path_c(
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
    apply = _mapped_packed_post_apply_for_layout(
        int(q_heads),
        int(k_heads),
        int(v_heads),
        int(g_heads),
    )
    return apply(conv_input, W, xf, h0, D, projected)


@lru_cache(maxsize=128)
def _post_residual_gate_apply_for_layout(
    q_heads: int,
    k_heads: int,
    v_heads: int,
    g_heads: int,
) -> Any:
    @mx.custom_function
    def _apply(
        y: mx.array,
        conv_input: mx.array,
        D: mx.array,
        projected: mx.array,
    ) -> mx.array:
        post = _m2rnn_post_residual_gate_fwd_path_c(
            y,
            conv_input,
            D,
            projected,
            q_heads=q_heads,
            k_heads=k_heads,
            v_heads=v_heads,
            g_heads=g_heads,
        )
        if post is None:
            _raise_post_residual_gate_path_c_unavailable(
                y,
                conv_input,
                D,
                projected,
                q_heads=q_heads,
                k_heads=k_heads,
                v_heads=v_heads,
                g_heads=g_heads,
            )
        return post

    apply_any = cast(Any, _apply)

    @apply_any.vjp
    def _apply_vjp(
        primals: tuple[mx.array, ...],
        cotangent: mx.array,
        output: mx.array,
    ) -> tuple[mx.array, ...]:
        del output
        y, conv_input, D, projected = primals
        grads = m2rnn_post_residual_gate_bwd_path_c(
            cotangent,
            y,
            conv_input,
            D,
            projected,
            q_heads=q_heads,
            k_heads=k_heads,
            v_heads=v_heads,
            g_heads=g_heads,
        )
        return _match_primal_gradient_dtypes(grads, primals)

    return _apply


def m2rnn_apply_post_residual_gate_path_c(
    y: mx.array,
    conv_input: mx.array,
    D: mx.array,
    projected: mx.array,
    *,
    q_heads: int,
    k_heads: int,
    v_heads: int,
    g_heads: int,
) -> mx.array:
    apply = _post_residual_gate_apply_for_layout(
        int(q_heads),
        int(k_heads),
        int(v_heads),
        int(g_heads),
    )
    return apply(y, conv_input, D, projected)


@mx.custom_function
def _m2rnn_apply_path_c_checked(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    W: mx.array,
    xf: mx.array,
    h0: mx.array,
) -> mx.array:
    full = _m2rnn_fwd_path_c_full(q, k, v, W, xf, h0)
    if full is None:
        _raise_path_c_unavailable()
    y, _h_last, _tanh_cache = full
    return y


@_m2rnn_apply_path_c_checked.vjp
def _m2rnn_apply_path_c_checked_vjp(
    primals: tuple[mx.array, ...],
    cotangent: mx.array,
    output: mx.array,
) -> tuple[mx.array, ...]:
    del output
    q, k, v, W, xf, h0 = primals
    full = _m2rnn_fwd_path_c_full(q, k, v, W, xf, h0)
    if full is None:
        _raise_path_c_unavailable()
    _y, _h_last, tanh_cache = full
    grads = m2rnn_bwd_path_c(
        cotangent,
        q,
        k,
        v,
        W,
        xf,
        tanh_cache,
        h0,
        force_path_c=True,
    )
    return _match_primal_gradient_dtypes(grads, primals)


@mx.custom_function
def m2rnn_apply_with_state_path_c(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    W: mx.array,
    xf: mx.array,
    h0: mx.array,
) -> tuple[mx.array, mx.array]:
    try:
        full = _m2rnn_fwd_path_c_full(q, k, v, W, xf, h0)
    except Exception as exc:
        try:
            from tilelang.contrib.mlx_interop import DLPackConversionError
        except Exception:  # pragma: no cover - only when TileLang import itself is broken
            DLPackConversionError = ()  # type: ignore[assignment]
        if isinstance(exc, DLPackConversionError):
            raise RuntimeError(
                "m2rnn_apply_with_state_path_c requires DLPack-contiguous "
                "caller-owned MLX input/output buffers; Path C will not copy "
                "or materialize broadcast/slice views implicitly"
            ) from exc
        raise
    if full is None:
        _raise_path_c_unavailable()
    y, h_last, _tanh_cache = full
    return y, h_last


@m2rnn_apply_with_state_path_c.vjp
def _m2rnn_apply_with_state_path_c_vjp(
    primals: tuple[mx.array, ...],
    cotangent: tuple[mx.array, mx.array],
    output: tuple[mx.array, mx.array],
) -> tuple[mx.array, ...]:
    del output
    q, k, v, W, xf, h0 = primals
    dy = cotangent[0]
    full = _m2rnn_fwd_path_c_full(q, k, v, W, xf, h0)
    if full is None:
        _raise_path_c_unavailable()
    _y, _h_last, tanh_cache = full
    grads = m2rnn_bwd_path_c(
        dy,
        q,
        k,
        v,
        W,
        xf,
        tanh_cache,
        h0,
        force_path_c=True,
    )
    return _match_primal_gradient_dtypes(grads, primals)


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
    if h0 is None:
        raise RuntimeError(
            "m2rnn_apply_path_c requires an existing h0 tensor; "
            "Path C will not allocate one implicitly"
        )
    if _path_c_inputs_eligible(q, k, v, W, xf, h0):
        return _m2rnn_apply_path_c_checked(q, k, v, W, xf, h0)
    raise RuntimeError(f"m2rnn_apply_path_c unavailable: {m2rnn_path_c_status().reason}")


def m2rnn_apply_with_state_path_c_or_fallback(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    W: mx.array,
    xf: mx.array,
    h0: mx.array,
    *,
    force_path_c: bool = False,
) -> tuple[mx.array, mx.array]:
    if _path_c_inputs_eligible(q, k, v, W, xf, h0):
        return m2rnn_apply_with_state_path_c(q, k, v, W, xf, h0)
    del force_path_c
    raise RuntimeError(
        f"m2rnn_apply_with_state_path_c unavailable: {m2rnn_path_c_status().reason}"
    )


__all__ = [
    "M2RNNPathCStatus",
    "m2rnn_apply_mapped_packed_post_with_state_path_c",
    "m2rnn_apply_mapped_packed_with_state_path_c",
    "m2rnn_apply_post_residual_gate_path_c",
    "m2rnn_apply_packed_with_state_path_c",
    "m2rnn_apply_with_state_path_c",
    "m2rnn_apply_with_state_path_c_or_fallback",
    "m2rnn_apply_path_c",
    "m2rnn_bwd_path_c",
    "m2rnn_fwd_path_c",
    "m2rnn_fwd_with_state_path_c",
    "m2rnn_mapped_packed_bwd_path_c",
    "m2rnn_mapped_packed_path_c_status",
    "m2rnn_mapped_packed_post_path_c_status",
    "m2rnn_post_residual_gate_bwd_path_c",
    "m2rnn_post_residual_gate_path_c_status",
    "m2rnn_packed_bwd_path_c",
    "m2rnn_packed_path_c_status",
    "m2rnn_path_c_status",
]

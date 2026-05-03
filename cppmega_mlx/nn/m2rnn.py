"""Correctness-first MLX reference for the cppmega M2RNN recurrence."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, overload

import mlx.core as mx
import mlx.nn as nn

from cppmega_mlx.nn.mamba3 import causal_depthwise_conv1d

DEFAULT_CHUNK_SIZE = 128


def _require_positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


@dataclass(frozen=True)
class M2RNNConfig:
    """Small local config for MLX smoke models and tests."""

    d_model: int
    k_head_dim: int = 64
    v_head_dim: int = 16
    num_q_heads: int = 1
    num_k_heads: int = 1
    num_v_heads: int = 4
    num_f_heads: int = 4
    num_g_heads: int = 4
    num_weight_heads: int = 1
    conv_kernel: int = 4
    chunk_size: int = DEFAULT_CHUNK_SIZE
    use_residual: bool = True
    A_init_min: float = 0.0
    A_init_max: float = 16.0
    dt_init_min: float = 1e-3
    dt_init_max: float = 0.1
    dt_init_floor: float = 1e-4

    @property
    def num_heads(self) -> int:
        return max(
            self.num_q_heads,
            self.num_k_heads,
            self.num_v_heads,
            self.num_f_heads,
            self.num_g_heads,
            self.num_weight_heads,
        )

    def __post_init__(self) -> None:
        _require_positive_int("d_model", self.d_model)
        _require_positive_int("k_head_dim", self.k_head_dim)
        _require_positive_int("v_head_dim", self.v_head_dim)
        _require_positive_int("num_q_heads", self.num_q_heads)
        _require_positive_int("num_k_heads", self.num_k_heads)
        _require_positive_int("num_v_heads", self.num_v_heads)
        _require_positive_int("num_f_heads", self.num_f_heads)
        _require_positive_int("num_g_heads", self.num_g_heads)
        _require_positive_int("num_weight_heads", self.num_weight_heads)
        _require_positive_int("conv_kernel", self.conv_kernel)
        _require_positive_int("chunk_size", self.chunk_size)

        total_heads = self.num_heads
        _require_positive_divisible(total_heads, self.num_q_heads, "num_q_heads")
        _require_positive_divisible(total_heads, self.num_k_heads, "num_k_heads")
        _require_positive_divisible(total_heads, self.num_v_heads, "num_v_heads")
        _require_positive_divisible(total_heads, self.num_f_heads, "num_f_heads")
        _require_positive_divisible(total_heads, self.num_g_heads, "num_g_heads")
        _require_positive_divisible(total_heads, self.num_weight_heads, "num_weight_heads")
        _require_nonnegative_float("A_init_min", self.A_init_min)
        _require_greater_float("A_init_max", self.A_init_max, "A_init_min", self.A_init_min)
        _require_positive_float("dt_init_min", self.dt_init_min)
        _require_greater_equal_float("dt_init_max", self.dt_init_max, "dt_init_min", self.dt_init_min)
        _require_positive_float("dt_init_floor", self.dt_init_floor)
        _require_greater_equal_float(
            "dt_init_max",
            self.dt_init_max,
            "dt_init_floor",
            self.dt_init_floor,
        )


@dataclass(frozen=True)
class M2RNNMixerState:
    """Explicit continuation state for M2RNNMixer segmented execution."""

    h: mx.array
    conv_state: mx.array


def _require_nonnegative_float(name: str, value: float) -> None:
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def _require_positive_float(name: str, value: float) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}")


def _require_greater_float(name: str, value: float, floor_name: str, floor: float) -> None:
    if value <= floor:
        raise ValueError(f"{name} must be greater than {floor_name}, got {value} <= {floor}")


def _require_greater_equal_float(name: str, value: float, floor_name: str, floor: float) -> None:
    if value < floor:
        raise ValueError(f"{name} must be >= {floor_name}, got {value} < {floor}")


def _require_rank(name: str, x: mx.array, rank: int) -> None:
    if x.ndim != rank:
        raise ValueError(f"{name} must be rank {rank}, got shape {x.shape}")


def _require_floating(name: str, x: mx.array) -> None:
    if not mx.issubdtype(x.dtype, mx.floating):
        raise TypeError(f"{name} must use a floating dtype, got {x.dtype}")


def _require_same_dtype(reference_name: str, reference: mx.array, name: str, x: mx.array) -> None:
    if x.dtype != reference.dtype:
        raise TypeError(
            f"{name} dtype {x.dtype} must match {reference_name} dtype {reference.dtype}"
        )


def _require_positive_divisible(total_heads: int, heads: int, name: str) -> None:
    if heads <= 0:
        raise ValueError(f"{name} head count must be positive, got {heads}")
    if total_heads % heads != 0:
        raise ValueError(
            f"{name} head count {heads} must divide broadcast head count {total_heads}"
        )


def _broadcast_heads(x: mx.array, total_heads: int, axis: int, name: str) -> mx.array:
    heads = x.shape[axis]
    _require_positive_divisible(total_heads, heads, name)
    if heads == total_heads:
        return x
    if heads == 1:
        target_shape = list(x.shape)
        target_shape[axis] = total_heads
        return mx.broadcast_to(x, tuple(target_shape))
    return mx.repeat(x, repeats=total_heads // heads, axis=axis)


def broadcast_m2rnn_heads(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    W: mx.array,
    xf: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Broadcast all M2RNN head axes to H=max(n_q,n_k,n_v,n_w,n_f)."""

    _require_rank("q", q, 4)
    _require_rank("k", k, 4)
    _require_rank("v", v, 4)
    _require_rank("W", W, 3)
    _require_rank("xf", xf, 3)
    _require_floating("q", q)
    _require_floating("k", k)
    _require_floating("v", v)
    _require_floating("W", W)
    _require_floating("xf", xf)
    _require_same_dtype("q", q, "k", k)
    _require_same_dtype("q", q, "v", v)
    _require_same_dtype("q", q, "W", W)
    _require_same_dtype("q", q, "xf", xf)

    batch, seq, n_q, k_dim = q.shape
    if k.shape[0] != batch or v.shape[0] != batch or xf.shape[0] != batch:
        raise ValueError("q, k, v, and xf must share batch size")
    if k.shape[1] != seq or v.shape[1] != seq or xf.shape[1] != seq:
        raise ValueError("q, k, v, and xf must share sequence length")
    if k.shape[-1] != k_dim:
        raise ValueError(f"k last dim must match q K dim {k_dim}, got {k.shape[-1]}")
    if W.shape[-2] != W.shape[-1]:
        raise ValueError(f"W must be square on its last two dims, got {W.shape}")
    if W.shape[-1] != v.shape[-1]:
        raise ValueError(f"W V dim {W.shape[-1]} must match v V dim {v.shape[-1]}")

    total_heads = max(n_q, k.shape[-2], v.shape[-2], W.shape[0], xf.shape[-1])
    return (
        _broadcast_heads(q, total_heads, -2, "q"),
        _broadcast_heads(k, total_heads, -2, "k"),
        _broadcast_heads(v, total_heads, -2, "v"),
        _broadcast_heads(W, total_heads, 0, "W"),
        _broadcast_heads(xf, total_heads, -1, "xf"),
    )


def m2rnn_softplus_decay_gate(
    f_input: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
) -> mx.array:
    """Megatron-style learnable M2RNN forget gate.

    Matches exp(-exp(A_log) * softplus(f_input + dt_bias)) and broadcasts
    projected forget heads to the learned per-state head parameters.
    """

    _require_rank("f_input", f_input, 3)
    _require_rank("A_log", A_log, 1)
    _require_rank("dt_bias", dt_bias, 1)
    _require_floating("f_input", f_input)
    _require_floating("A_log", A_log)
    _require_floating("dt_bias", dt_bias)
    if A_log.shape != dt_bias.shape:
        raise ValueError(f"A_log shape {A_log.shape} must match dt_bias shape {dt_bias.shape}")

    gate_dtype = f_input.dtype
    if f_input.shape[-1] != A_log.shape[0]:
        f_input = _broadcast_heads(f_input, A_log.shape[0], -1, "f_input")

    dt = nn.softplus(f_input.astype(mx.float32) + dt_bias.astype(mx.float32))
    log_decay = -mx.exp(A_log.astype(mx.float32)) * dt
    return mx.exp(log_decay).astype(gate_dtype)


def _initial_m2rnn_state(
    h0: mx.array | None,
    *,
    batch: int,
    heads: int,
    k_dim: int,
    v_dim: int,
    dtype: mx.Dtype,
) -> mx.array:
    shape = (batch, heads, k_dim, v_dim)
    if h0 is None:
        return mx.zeros(shape, dtype=dtype)
    if h0.shape != shape:
        raise ValueError(f"h0 must have shape {shape}, got {h0.shape}")
    _require_floating("h0", h0)
    if h0.dtype != dtype:
        raise TypeError(f"h0 dtype {h0.dtype} must match q dtype {dtype}")
    return h0


def m2rnn_scan(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    W: mx.array,
    xf: mx.array,
    *,
    h0: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Sequential MLX reference scan matching the Megatron PyTorch M2RNN seam.

    Shapes:
    q(B,S,n_q,K), k(B,S,n_k,K), v(B,S,n_v,V), W(n_w,V,V),
    xf(B,S,n_f), optional h0(B,H,K,V).  Returns
    out(B,S,H,V) and final h(B,H,K,V).
    """

    q, k, v, W, xf = broadcast_m2rnn_heads(q, k, v, W, xf)
    batch, seq, heads, k_dim = q.shape
    v_dim = v.shape[-1]

    h = _initial_m2rnn_state(
        h0,
        batch=batch,
        heads=heads,
        k_dim=k_dim,
        v_dim=v_dim,
        dtype=q.dtype,
    )

    # Match the Megatron chunk reference: keep the B*S outer products outside
    # the recurrent token loop so MLX compile can schedule that work separately.
    x_all = mx.expand_dims(k, -1) * mx.expand_dims(v, -2)
    xf_5d = xf[:, :, :, None, None]
    W_expanded = mx.expand_dims(W, 0)
    outputs: list[mx.array] = []
    for s in range(seq):
        f = xf_5d[:, s]
        h_new = mx.tanh(mx.matmul(h, W_expanded) + x_all[:, s])
        h = f * h + (1.0 - f) * h_new
        outputs.append(mx.einsum("bhk,bhkv->bhv", q[:, s], h))

    if outputs:
        out = mx.stack(outputs, axis=1)
    else:
        out = mx.zeros((batch, 0, heads, v_dim), dtype=q.dtype)
    return out, h


def chunked_m2rnn_scan(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    W: mx.array,
    xf: mx.array,
    *,
    h0: mx.array | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[mx.array, mx.array]:
    """Chunk wrapper for API parity with the CUDA/Triton design.

    This remains a correctness reference: chunks only bound Python loop slices;
    the recurrence is still sequential across all tokens.
    """

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    q, k, v, W, xf = broadcast_m2rnn_heads(q, k, v, W, xf)
    batch, seq, heads, k_dim = q.shape
    v_dim = v.shape[-1]

    h = _initial_m2rnn_state(
        h0,
        batch=batch,
        heads=heads,
        k_dim=k_dim,
        v_dim=v_dim,
        dtype=q.dtype,
    )

    # Precompute per-token rank-1 inputs before the chunk loop, matching the
    # CUDA/Triton-oriented reference decomposition in ../cppmega.
    x_all = mx.expand_dims(k, -1) * mx.expand_dims(v, -2)
    xf_5d = xf[:, :, :, None, None]
    W_expanded = mx.expand_dims(W, 0)
    outputs: list[mx.array] = []
    n_chunks = math.ceil(seq / chunk_size) if seq else 0
    for chunk in range(n_chunks):
        start = chunk * chunk_size
        end = min(start + chunk_size, seq)
        x_chunk = x_all[:, start:end]
        f_chunk = xf_5d[:, start:end]
        q_chunk = q[:, start:end]
        for t in range(end - start):
            f = f_chunk[:, t]
            h_new = mx.tanh(mx.matmul(h, W_expanded) + x_chunk[:, t])
            h = f * h + (1.0 - f) * h_new
            outputs.append(mx.einsum("bhk,bhkv->bhv", q_chunk[:, t], h))

    if outputs:
        out = mx.stack(outputs, axis=1)
    else:
        out = mx.zeros((batch, 0, heads, v_dim), dtype=q.dtype)
    return out, h


class M2RNNMixer(nn.Module):
    """Lightweight hidden-state mixer for local MLX smoke training.

    The default return remains (out, h) for existing callers.  Segmented
    continuation through the Q/K/V causal convolution must opt into
    return_state=True and pass the returned mixer_state to the suffix.
    """

    def __init__(self, config: M2RNNConfig):
        super().__init__()
        self.config = config
        self.q_dim = config.num_q_heads * config.k_head_dim
        self.k_dim = config.num_k_heads * config.k_head_dim
        self.v_dim = config.num_v_heads * config.v_head_dim
        self.conv_dim = self.q_dim + self.k_dim + self.v_dim
        self.f_dim = config.num_f_heads
        self.g_dim = config.num_g_heads * config.v_head_dim

        self.in_proj = nn.Linear(
            config.d_model,
            self.conv_dim + self.f_dim + self.g_dim,
            bias=False,
        )
        self.conv_weight = self._init_conv_weight(self.conv_dim, config.conv_kernel)
        self.conv_bias = mx.zeros((self.conv_dim,))
        self.g_norm = nn.RMSNorm(config.num_heads * config.v_head_dim)
        self.out_proj = nn.Linear(config.num_heads * config.v_head_dim, config.d_model, bias=False)
        self.state_weight = mx.broadcast_to(
            mx.eye(config.v_head_dim)[None, :, :],
            (config.num_weight_heads, config.v_head_dim, config.v_head_dim),
        )
        self.A_log = self._init_A_log(config)
        self.dt_bias = self._init_dt_bias(config)
        self.D = mx.ones((config.num_heads, config.v_head_dim)) if config.use_residual else None

    @staticmethod
    def _init_conv_weight(channels: int, kernel_size: int) -> mx.array:
        if kernel_size <= 0:
            raise ValueError(f"conv_kernel must be positive, got {kernel_size}")
        scale = math.sqrt(1 / (channels * kernel_size))
        return mx.random.uniform(-scale, scale, (channels, kernel_size, 1))

    def _empty_conv_state(self, batch: int, dtype: mx.Dtype) -> mx.array:
        return mx.zeros((batch, self.config.conv_kernel - 1, self.conv_dim), dtype=dtype)

    def _validate_conv_state(
        self,
        conv_state: mx.array,
        *,
        batch: int,
        dtype: mx.Dtype,
    ) -> mx.array:
        expected_shape = (batch, self.config.conv_kernel - 1, self.conv_dim)
        _require_rank("mixer_state.conv_state", conv_state, 3)
        _require_floating("mixer_state.conv_state", conv_state)
        if conv_state.shape != expected_shape:
            raise ValueError(
                f"mixer_state.conv_state must have shape {expected_shape}, got {conv_state.shape}"
            )
        if conv_state.dtype != dtype:
            raise TypeError(
                f"mixer_state.conv_state dtype {conv_state.dtype} must match projected dtype {dtype}"
            )
        return conv_state

    def _next_conv_state(self, conv_source: mx.array, *, batch: int, dtype: mx.Dtype) -> mx.array:
        history_len = self.config.conv_kernel - 1
        if history_len == 0:
            return self._empty_conv_state(batch, dtype)
        source_len = conv_source.shape[1]
        if source_len >= history_len:
            return conv_source[:, -history_len:, :]
        pad = mx.zeros((batch, history_len - source_len, self.conv_dim), dtype=dtype)
        return mx.concatenate([pad, conv_source], axis=1)

    @staticmethod
    def _init_A_log(config: M2RNNConfig) -> mx.array:
        A = mx.random.uniform(config.A_init_min, config.A_init_max, (config.num_heads,))
        A = mx.maximum(A, mx.array(1e-8, dtype=A.dtype))
        return mx.log(A).astype(mx.float32)

    @staticmethod
    def _init_dt_bias(config: M2RNNConfig) -> mx.array:
        dt_min = max(config.dt_init_min, config.dt_init_floor)
        dt = mx.exp(mx.random.uniform(math.log(dt_min), math.log(config.dt_init_max), (config.num_heads,)))
        return (dt + mx.log(-mx.expm1(-dt))).astype(mx.float32)

    @overload
    def __call__(
        self,
        hidden_states: mx.array,
        *,
        h0: mx.array | None = None,
        mixer_state: M2RNNMixerState | None = None,
        chunk_size: int | None = None,
        return_state: Literal[False] = False,
    ) -> tuple[mx.array, mx.array]: ...

    @overload
    def __call__(
        self,
        hidden_states: mx.array,
        *,
        h0: mx.array | None = None,
        mixer_state: M2RNNMixerState | None = None,
        chunk_size: int | None = None,
        return_state: Literal[True],
    ) -> tuple[mx.array, M2RNNMixerState]: ...

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        h0: mx.array | None = None,
        mixer_state: M2RNNMixerState | None = None,
        chunk_size: int | None = None,
        return_state: bool = False,
    ) -> tuple[mx.array, mx.array] | tuple[mx.array, M2RNNMixerState]:
        if hidden_states.ndim != 3:
            raise ValueError(f"hidden_states must be shaped (B,S,D), got {hidden_states.shape}")
        if h0 is not None and mixer_state is not None:
            raise ValueError("pass either h0 or mixer_state, not both")

        cfg = self.config
        projected = self.in_proj(hidden_states)
        conv_end = self.conv_dim
        f_end = conv_end + self.f_dim

        batch, seq, _ = hidden_states.shape
        projected_conv = projected[:, :, :conv_end]
        conv_prefix: mx.array | None = None
        scan_h0 = h0
        if mixer_state is not None:
            if not isinstance(mixer_state, M2RNNMixerState):
                raise TypeError("mixer_state must be an M2RNNMixerState")
            scan_h0 = _initial_m2rnn_state(
                mixer_state.h,
                batch=batch,
                heads=cfg.num_heads,
                k_dim=cfg.k_head_dim,
                v_dim=cfg.v_head_dim,
                dtype=projected.dtype,
            )
            conv_prefix = self._validate_conv_state(
                mixer_state.conv_state,
                batch=batch,
                dtype=projected.dtype,
            )
        conv_source = (
            projected_conv
            if conv_prefix is None or conv_prefix.shape[1] == 0
            else mx.concatenate([conv_prefix, projected_conv], axis=1)
        )
        next_conv_state = self._next_conv_state(conv_source, batch=batch, dtype=projected.dtype)
        if conv_source.shape[1] == 0:
            conv_input = mx.zeros((batch, 0, conv_end), dtype=projected.dtype)
        else:
            conv_input = causal_depthwise_conv1d(
                conv_source,
                self.conv_weight.astype(projected.dtype),
                self.conv_bias.astype(projected.dtype),
            )
        if conv_prefix is not None and conv_prefix.shape[1] > 0:
            conv_input = conv_input[:, conv_prefix.shape[1] :, :]
        conv_input = nn.silu(conv_input)

        q_end = self.q_dim
        k_end = q_end + self.k_dim
        v_end = k_end + self.v_dim
        q = conv_input[:, :, :q_end].reshape(batch, seq, cfg.num_q_heads, cfg.k_head_dim)
        k = conv_input[:, :, q_end:k_end].reshape(batch, seq, cfg.num_k_heads, cfg.k_head_dim)
        v = conv_input[:, :, k_end:v_end].reshape(batch, seq, cfg.num_v_heads, cfg.v_head_dim)
        xf = m2rnn_softplus_decay_gate(projected[:, :, conv_end:f_end], self.A_log, self.dt_bias)
        g = projected[:, :, f_end:].reshape(batch, seq, cfg.num_g_heads, cfg.v_head_dim)

        out, h = chunked_m2rnn_scan(
            q,
            k,
            v,
            self.state_weight.astype(q.dtype),
            xf,
            h0=scan_h0,
            chunk_size=cfg.chunk_size if chunk_size is None else chunk_size,
        )
        if self.D is not None:
            v_broadcast = _broadcast_heads(v, cfg.num_heads, -2, "v")
            out = out + v_broadcast * self.D.astype(out.dtype)
        out = out.reshape(batch, seq, cfg.num_heads * cfg.v_head_dim)
        g = g.reshape(batch, seq, self.g_dim)
        if cfg.num_g_heads != cfg.num_heads:
            g = mx.repeat(g, repeats=cfg.num_heads // cfg.num_g_heads, axis=-1)
        out = out * nn.silu(g).astype(out.dtype)
        out = self.g_norm(out)
        out = self.out_proj(out)
        if return_state:
            return out, M2RNNMixerState(h=h, conv_state=next_conv_state)
        return out, h


__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "M2RNNConfig",
    "M2RNNMixer",
    "M2RNNMixerState",
    "broadcast_m2rnn_heads",
    "chunked_m2rnn_scan",
    "m2rnn_softplus_decay_gate",
    "m2rnn_scan",
]

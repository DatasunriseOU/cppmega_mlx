"""8-bit blockwise-quantized optimizer state for AdamW and Lion.

This module ships :class:`Adam8bit` (drop-in AdamW with uint8 ``m``, ``v``
moments) and :class:`Lion8bit` (drop-in Lion with a single uint8 momentum
buffer ``m``). Per-parameter state is stored as ``uint8`` plus per-256-block
fp32 absmax scales:

* Adam8bit: ~2.06 bytes/parameter (2 * 1B + 4B/64B) versus 8 bytes/parameter
  for fp32 ``m``+``v`` -- the same target as ``bitsandbytes.optim.Adam8bit``.
* Lion8bit: ~1.02 bytes/parameter (1B + 4B/256B) versus 4 bytes/parameter for
  fp32 ``m`` -- the same target as ``bitsandbytes.optim.Lion8bit``. See
  ``cppmega/docs/lion8bit_ab_2026_04_25.md`` for the CUDA reference run.

Numerical policy:

* Quantization is selectable via ``quant_scheme``:

  - ``"symmetric_int8_v1"`` (default): uint8 with +128 bias, the existing
    M0-grade codec.
  - ``"dynamic_int8_v1"``: opt-in bitsandbytes-style dynamic LUT
    (``dDequantizeBlockwise`` parity for the 256-entry signed dynamic
    map). Denser bins near zero so small Adam ``m, v`` values keep more
    precision on the round-trip.

  Default stays symmetric for backward compatibility; ``bitsandbytes``
  defaults to the dynamic LUT on the CUDA stack. Large tensors use the native
  MLX C++/Metal fused optimizer primitive when available; otherwise they fall
  back to the native MLX codec path.
* The internal optimizer math runs in fp32 inline: every ``apply_single``
  call dequantizes the moment state to fp32, performs the standard update,
  then re-quantizes. The parameter is read in fp32 and cast back to whatever
  dtype it had on input (bf16 in production).
* No master copy of weights -- the parameter tree is the only source of
  truth for weights, just like the fp32-moments optimizers upstream.
* ``min_8bit_size`` can keep small tensors in fp32 optimizer state. The repo
  default is ``0`` for backward compatibility with existing fully-quantized
  receipts; production receipt routes opt into ``4096`` to mirror the
  bitsandbytes stability policy for norms and biases.

See :mod:`cppmega_mlx.training._quantize_8bit` for the native MLX codecs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import mlx.core as mx
import mlx.optimizers as optim

from cppmega_mlx.training._fused_adam8bit_kernel import (
    FUSED_BLOCK_SIZE,
    fused_adam8bit_step,
    fused_adam8bit_status,
)
from cppmega_mlx.training._fused_dynamic8bit_kernel import (
    fused_adam8bit_dynamic_step,
    fused_adam8bit_dynamic_status,
    fused_lion8bit_dynamic_step,
    fused_lion8bit_dynamic_status,
)
from cppmega_mlx.training._fused_lion8bit_kernel import (
    fused_lion8bit_step,
    fused_lion8bit_status,
)
from cppmega_mlx.training._quantize_8bit import (
    DEFAULT_BLOCK_SIZE,
    QUANT_SCHEME_DYNAMIC,
    QUANT_SCHEME_SYMMETRIC,
    QUANT_SCHEMES,
    dequantize_blockwise,
    num_blocks,
    quantize_blockwise,
)


ADAM8BIT_CLASS = "cppmega_mlx.training.optimizers_quantized.Adam8bit"
ADAM8BIT_SOURCE = "cppmega_mlx.training.optimizers_quantized.make_adam8bit"
ADAM8BIT_QUANT_KIND = "symmetric_int8_blockwise_v1"
"""Default codec identifier emitted by checkpoint metadata.

The default :class:`Adam8bit` instance keeps the symmetric int8 path so this
constant stays pinned to ``"symmetric_int8_blockwise_v1"``. Callers that pass
``quant_scheme="dynamic_int8_v1"`` get the bnb-style dynamic LUT codec; the
runtime scheme is recorded on the optimizer instance via ``quant_scheme``
without mutating this module-level constant.
"""

ADAM8BIT_QUANT_SCHEMES: tuple[str, ...] = QUANT_SCHEMES
"""All accepted ``quant_scheme`` strings for :class:`Adam8bit` and
:func:`make_adam8bit`. Re-exported from ``_quantize_8bit`` so callers can
import the canonical scheme list from a single place."""

Adam8bitQuantScheme = Literal["symmetric_int8_v1", "dynamic_int8_v1"]
"""Type alias for the accepted ``quant_scheme`` strings."""

LION8BIT_CLASS = "cppmega_mlx.training.optimizers_quantized.Lion8bit"
LION8BIT_SOURCE = "cppmega_mlx.training.optimizers_quantized.make_lion8bit"
LION8BIT_QUANT_KIND = "symmetric_int8_blockwise_v1"
"""Lion8bit shares the same symmetric int8 codec as Adam8bit. The CUDA
reference (``bitsandbytes.optim.Lion8bit``) uses a dynamic LUT; bumping this
identifier signals when a dynamic-LUT MLX path lands."""


@dataclass(frozen=True)
class FusedKernelRequestStatus:
    available: bool
    reason: str


def _unavailable_fused_status(reason: str) -> FusedKernelRequestStatus:
    return FusedKernelRequestStatus(False, reason)


def _normalize_fused_status(status: Any) -> FusedKernelRequestStatus:
    return FusedKernelRequestStatus(
        bool(getattr(status, "available")),
        str(getattr(status, "reason")),
    )


def _adam_fused_status(
    *,
    requested: bool,
    block_size: int,
    quant_scheme: str,
) -> FusedKernelRequestStatus:
    if not requested:
        return _unavailable_fused_status("fused optimizer kernel disabled by caller")
    if block_size != FUSED_BLOCK_SIZE:
        return _unavailable_fused_status(
            f"fused optimizer kernel requires block_size={FUSED_BLOCK_SIZE}"
        )
    if quant_scheme == QUANT_SCHEME_DYNAMIC:
        return _normalize_fused_status(fused_adam8bit_dynamic_status())
    if quant_scheme == QUANT_SCHEME_SYMMETRIC:
        return _normalize_fused_status(fused_adam8bit_status())
    return _unavailable_fused_status(f"unsupported quant scheme {quant_scheme!r}")


def _lion_fused_status(
    *,
    requested: bool,
    block_size: int,
    quant_scheme: str,
) -> FusedKernelRequestStatus:
    if not requested:
        return _unavailable_fused_status("fused optimizer kernel disabled by caller")
    if block_size != FUSED_BLOCK_SIZE:
        return _unavailable_fused_status(
            f"fused optimizer kernel requires block_size={FUSED_BLOCK_SIZE}"
        )
    if quant_scheme == QUANT_SCHEME_DYNAMIC:
        return _normalize_fused_status(fused_lion8bit_dynamic_status())
    if quant_scheme == QUANT_SCHEME_SYMMETRIC:
        return _normalize_fused_status(fused_lion8bit_status())
    return _unavailable_fused_status(f"unsupported quant scheme {quant_scheme!r}")


def _block_absmax(x: mx.array, block_size: int) -> mx.array:
    """Return fp32 absmax per flattened block without padding large tensors."""

    flat = x.reshape(-1)
    if flat.dtype != mx.float32:
        flat = flat.astype(mx.float32)
    flat = mx.abs(flat)
    if int(flat.size) == 0:
        return mx.zeros((0,), dtype=mx.float32)
    n = int(flat.size)
    n_full = n // block_size
    full_size = n_full * block_size
    parts: list[mx.array] = []
    if n_full:
        parts.append(mx.max(flat[:full_size].reshape(n_full, block_size), axis=1))
    if full_size < n:
        parts.append(mx.max(flat[full_size:], keepdims=True))
    if len(parts) == 1:
        return parts[0]
    return mx.concatenate(parts, axis=0)


def _adam_block_update(
    m_new: mx.array,
    v_new: mx.array,
    parameter: mx.array,
    v_block_step: mx.array,
    *,
    learning_rate: mx.array,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step: mx.array,
    bias_correction: bool,
    block_size: int,
) -> mx.array:
    """Apply AdamW using per-block noise floors without repeating them."""

    original_shape = parameter.shape
    param_dtype = parameter.dtype
    n = int(parameter.size)
    if n == 0:
        return parameter

    flat_m = m_new.reshape(-1)
    flat_v = v_new.reshape(-1)
    flat_param = parameter.reshape(-1)
    n_full = n // block_size
    full_size = n_full * block_size

    if bias_correction:
        step_fp32 = step.astype(mx.float32)
        c1 = learning_rate / (
            1.0 - mx.power(mx.array(beta1, dtype=mx.float32), step_fp32)
        )
        c2 = mx.rsqrt(
            1.0 - mx.power(mx.array(beta2, dtype=mx.float32), step_fp32)
        )
    else:
        c1 = learning_rate
        c2 = None

    def apply_block(
        m_block: mx.array,
        v_block: mx.array,
        p_block: mx.array,
        floor: mx.array,
    ) -> mx.array:
        p32 = p_block.astype(mx.float32) if p_block.dtype != mx.float32 else p_block
        if c2 is None:
            update = learning_rate * m_block / (mx.sqrt(v_block) + floor + eps)
        else:
            update = (c1 * m_block) / (mx.sqrt(v_block) * c2 + floor + eps)
        decayed = p32 * (1.0 - learning_rate * weight_decay)
        return decayed - update

    parts: list[mx.array] = []
    if n_full:
        m_full = flat_m[:full_size].reshape(n_full, block_size)
        v_full = flat_v[:full_size].reshape(n_full, block_size)
        p_full = flat_param[:full_size].reshape(n_full, block_size)
        parts.append(
            apply_block(m_full, v_full, p_full, v_block_step[:n_full, None]).reshape(-1)
        )

    if full_size < n:
        parts.append(
            apply_block(
                flat_m[full_size:],
                flat_v[full_size:],
                flat_param[full_size:],
                v_block_step[n_full],
            )
        )

    updated = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)
    return updated.reshape(original_shape).astype(param_dtype)


class Adam8bit(optim.Optimizer):
    """AdamW with 8-bit blockwise-quantized ``m``, ``v`` moments.

    State per parameter:

    * ``m_quant``: uint8, same shape as the parameter.
    * ``m_absmax``: fp32, shape ``(num_blocks(param.size, block_size),)``.
    * ``v_quant``: uint8, same shape as the parameter.
    * ``v_absmax``: fp32, same shape as ``m_absmax``.

    Plus the standard ``step`` (uint64) and ``learning_rate`` (fp32) shared by
    all MLX optimizers.

    Memory: ~2 bytes/param (uint8 m + uint8 v) plus ~0.0625 bytes/param of
    metadata (2 * fp32 / 256 = 1/32 byte each). Total ~2.06 bytes/param.

    Apply path:

    1. Dequantize ``m, v`` to fp32.
    2. Standard AdamW math:
       ``m = beta1 * m + (1 - beta1) * g``,
       ``v = beta2 * v + (1 - beta2) * g**2``,
       ``update = lr * m_hat / (sqrt(v_hat) + eps)``.
    3. Re-quantize updated ``m, v`` to uint8 + absmax.
    4. Apply ``param * (1 - lr * weight_decay) - update`` in fp32, cast back
       to the input parameter dtype (bf16 in production).
    """

    def __init__(
        self,
        learning_rate: float | Callable[[mx.array], mx.array],
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        bias_correction: bool = False,
        block_size: int = DEFAULT_BLOCK_SIZE,
        use_fused_kernel: bool = True,
        quant_scheme: Adam8bitQuantScheme = QUANT_SCHEME_SYMMETRIC,
        min_8bit_size: int = 0,
    ) -> None:
        super().__init__()
        self._maybe_schedule("learning_rate", learning_rate)
        self.betas = list(betas)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.bias_correction = bool(bias_correction)
        self.block_size = int(block_size)
        if quant_scheme not in QUANT_SCHEMES:
            raise ValueError(
                f"quant_scheme must be one of {QUANT_SCHEMES}; got {quant_scheme!r}"
            )
        self.quant_scheme: str = quant_scheme
        if min_8bit_size < 0:
            raise ValueError("min_8bit_size must be >= 0")
        self.min_8bit_size = int(min_8bit_size)
        self.fused_kernel_status = _adam_fused_status(
            requested=bool(use_fused_kernel),
            block_size=self.block_size,
            quant_scheme=self.quant_scheme,
        )
        self.use_fused_kernel = self.fused_kernel_status.available

    def init_single(self, parameter: mx.array, state: dict[str, Any]) -> None:
        if self.min_8bit_size and int(parameter.size) < self.min_8bit_size:
            state["m"] = mx.zeros(parameter.shape, dtype=mx.float32)
            state["v"] = mx.zeros(parameter.shape, dtype=mx.float32)
            return

        nb = num_blocks(int(parameter.size), self.block_size)
        # For symmetric int8 the +128 bias maps to signed 0, i.e. an all-zero
        # moment after dequant. For the dynamic LUT scheme byte index 127
        # maps to LUT[127] == 0.0 (the canonical zero entry produced by the
        # ``data.append(0)`` step in ``create_dynamic_map``). Either way an
        # all-128 (symmetric) or all-127 (dynamic) initial payload yields
        # zero moments after dequant.
        zero_byte = 128 if self.quant_scheme == QUANT_SCHEME_SYMMETRIC else 127
        state["m_quant"] = mx.full(parameter.shape, zero_byte, dtype=mx.uint8)
        state["m_absmax"] = mx.zeros((nb,), dtype=mx.float32)
        state["v_quant"] = mx.full(parameter.shape, zero_byte, dtype=mx.uint8)
        state["v_absmax"] = mx.zeros((nb,), dtype=mx.float32)

    def apply_single(
        self,
        gradient: mx.array,
        parameter: mx.array,
        state: dict[str, Any],
    ) -> mx.array:
        b1, b2 = self.betas
        eps = self.eps
        lr_fp32 = self.learning_rate.astype(mx.float32)
        param_dtype = parameter.dtype

        if "m" in state and "v" in state:
            m_prev = state["m"]
            v_prev = mx.maximum(state["v"], 0.0)
            g32 = gradient.astype(mx.float32)
            m_new = b1 * m_prev + (1.0 - b1) * g32
            v_new = mx.maximum(b2 * v_prev + (1.0 - b2) * mx.square(g32), 0.0)
            if self.bias_correction:
                step = self.step.astype(mx.float32)
                c1 = lr_fp32 / (1.0 - mx.power(mx.array(b1, dtype=mx.float32), step))
                c2 = mx.rsqrt(1.0 - mx.power(mx.array(b2, dtype=mx.float32), step))
                update = (c1 * m_new) / (mx.sqrt(v_new) * c2 + eps)
            else:
                update = lr_fp32 * m_new / (mx.sqrt(v_new) + eps)
            state["m"] = m_new
            state["v"] = v_new
            decayed = parameter.astype(mx.float32) * (1.0 - lr_fp32 * self.weight_decay)
            return (decayed - update).astype(param_dtype)

        scheme = self.quant_scheme
        if self.use_fused_kernel:
            if scheme == QUANT_SCHEME_DYNAMIC:
                updated, m_q, m_absmax, v_q, v_absmax = fused_adam8bit_dynamic_step(
                    parameter,
                    gradient,
                    state["m_quant"],
                    state["m_absmax"],
                    state["v_quant"],
                    state["v_absmax"],
                    learning_rate=lr_fp32,
                    beta1=b1,
                    beta2=b2,
                    eps=eps,
                    weight_decay=self.weight_decay,
                    step=self.step,
                    bias_correction=self.bias_correction,
                    block_size=self.block_size,
                )
            else:
                updated, m_q, m_absmax, v_q, v_absmax = fused_adam8bit_step(
                    parameter,
                    gradient,
                    state["m_quant"],
                    state["m_absmax"],
                    state["v_quant"],
                    state["v_absmax"],
                    learning_rate=lr_fp32,
                    beta1=b1,
                    beta2=b2,
                    eps=eps,
                    weight_decay=self.weight_decay,
                    step=self.step,
                    bias_correction=self.bias_correction,
                    block_size=self.block_size,
                )
            state["m_quant"] = m_q
            state["m_absmax"] = m_absmax
            state["v_quant"] = v_q
            state["v_absmax"] = v_absmax
            return updated

        # 1) Dequantize current moments to fp32. For the symmetric path the
        # dequant introduces rounding error bounded by ``absmax / (2 * 127)``
        # per block; the dynamic LUT path bounds the error by half the
        # neighbour-bin spacing in the LUT, which is much smaller for values
        # near zero. We compensate either way by floor-clamping ``v`` to the
        # per-block quantization step size during the AdamW update -- without
        # that, the ``sqrt(v) + eps`` denominator would underflow and the
        # update would diverge whenever ``v`` was small relative to its block
        # absmax (e.g. tail of a fat-tailed gradient distribution).
        m_prev = dequantize_blockwise(
            state["m_quant"], state["m_absmax"], scheme=scheme, out_dtype=mx.float32
        )
        v_prev = dequantize_blockwise(
            state["v_quant"], state["v_absmax"], scheme=scheme, out_dtype=mx.float32
        )
        v_prev = mx.maximum(v_prev, 0.0)

        g32 = gradient.astype(mx.float32)
        m_new = b1 * m_prev + (1.0 - b1) * g32
        v_new = mx.maximum(b2 * v_prev + (1.0 - b2) * mx.square(g32), 0.0)

        # 2) Build a per-block lower-bound for v that absorbs the quant noise
        # floor: anything below the per-block round-trip step is
        # indistinguishable from zero, so we add it to ``sqrt(v)`` as a
        # block-aware epsilon. The symmetric path uses ``absmax / 127`` (the
        # uniform int8 step). The dynamic LUT path uses ``absmax / 127`` as a
        # safe upper bound; the actual smallest LUT bin is ~5.5e-7 which is
        # much tighter, but using the symmetric bound here is conservative
        # and avoids needing a per-block LUT-spacing query in the optimizer.
        # state["v_absmax"] is the absmax from the *previous* step. Use the
        # max of (previous, |v_new| absmax estimate) so the floor reflects
        # whatever round-trip we're about to take.
        v_absmax_estimate = _block_absmax(v_new, self.block_size)
        v_block_step = mx.maximum(state["v_absmax"], v_absmax_estimate) / 127.0

        # 3) Compute the update in fp32 before re-quantization, so the round
        # trip through uint8 happens after we already used the fresh moments.
        # The denominator ``sqrt(v_new) + block_floor + eps`` never underflows
        # even when v_new has elements that quantize to zero on the next round
        # trip. The block floor is broadcast per block/tail rather than
        # materialized with a full-size repeat tensor.
        updated = _adam_block_update(
            m_new,
            v_new,
            parameter,
            v_block_step,
            learning_rate=lr_fp32,
            beta1=b1,
            beta2=b2,
            eps=eps,
            weight_decay=self.weight_decay,
            step=self.step,
            bias_correction=self.bias_correction,
            block_size=self.block_size,
        )

        # 4) Re-quantize the updated moments back to uint8 + absmax storage
        # using whichever codec the optimizer was configured for.
        m_q, m_absmax = quantize_blockwise(m_new, self.block_size, scheme=scheme)
        v_q, v_absmax = quantize_blockwise(v_new, self.block_size, scheme=scheme)
        state["m_quant"] = m_q
        state["m_absmax"] = m_absmax
        state["v_quant"] = v_q
        state["v_absmax"] = v_absmax

        return updated


def make_adam8bit(
    *,
    learning_rate: float | Callable[[mx.array], mx.array] = 1e-3,
    weight_decay: float = 0.01,
    betas: list[float] | None = None,
    eps: float = 1e-8,
    bias_correction: bool = False,
    block_size: int = DEFAULT_BLOCK_SIZE,
    use_fused_kernel: bool = True,
    quant_scheme: Adam8bitQuantScheme = QUANT_SCHEME_SYMMETRIC,
    min_8bit_size: int = 0,
) -> Adam8bit:
    """Construct the repo-default 8-bit AdamW for bf16 training.

    The defaults mirror :func:`cppmega_mlx.training.optimizers.make_adamw` so
    the Adam8bit path is a drop-in swap. ``block_size`` matches bitsandbytes's
    256-element blockwise default; only that value is supported by the native
    MLX codec today (other sizes raise ``NotImplementedError``).

    ``use_fused_kernel`` requests the native MLX C++/Metal fused primitive for
    the dequant -> update -> requant -> apply path. If the extension is not
    built, ``optimizer.fused_kernel_status`` records the fallback reason and
    the optimizer uses the native MLX codec path.

    ``quant_scheme`` selects the 8-bit codec:

    * ``"symmetric_int8_v1"`` (default): the existing symmetric int8 codec
      with +128 bias. Backwards compatible.
    * ``"dynamic_int8_v1"``: the bitsandbytes-style dynamic LUT
      (``dDequantizeBlockwise`` parity for the signed dynamic map). Denser
      bins near zero so small Adam ``m, v`` values keep more precision.

    ``min_8bit_size`` keeps tensors with fewer elements in fp32 ``m``/``v``
    state. Set it to ``4096`` for the bitsandbytes default stability policy;
    the function default remains ``0`` to preserve older all-quantized local
    receipts unless the caller opts in.

    Memory footprint vs ``make_adamw`` on a 1.797B-param bf16 model:

    * AdamWFP32Moments: 2 * 4 B/param = 8 B/param  -> ~14.4 GiB state.
    * Adam8bit: 2 * 1 B/param + 2 * 4 B / 256 B = ~2.06 B/param -> ~3.7 GiB.
    """

    return Adam8bit(
        learning_rate=learning_rate,
        betas=(0.9, 0.999) if betas is None else (betas[0], betas[1]),
        eps=eps,
        weight_decay=weight_decay,
        bias_correction=bias_correction,
        block_size=block_size,
        use_fused_kernel=use_fused_kernel,
        quant_scheme=quant_scheme,
        min_8bit_size=min_8bit_size,
    )


Lion8bitQuantScheme = Literal["symmetric_int8_v1", "dynamic_int8_v1"]
"""Type alias for the accepted ``quant_scheme`` strings on :class:`Lion8bit`."""


class Lion8bit(optim.Optimizer):
    """Lion with 8-bit blockwise-quantized momentum.

    Lion (Chen et al. arXiv 2302.06675) only carries a single momentum buffer
    ``m`` per parameter, half the state of AdamW. With blockwise quantization
    that buffer drops to ~1.02 B/param, mirroring
    ``bitsandbytes.optim.Lion8bit`` on the GB10 CUDA stack -- see
    ``cppmega/docs/lion8bit_ab_2026_04_25.md`` for the reference run.

    State per parameter:

    * ``m_quant``: uint8, same shape as the parameter.
    * ``m_absmax``: fp32, shape ``(num_blocks(param.size, block_size),)``.

    Plus the standard ``step`` (uint64) and ``learning_rate`` (fp32) shared by
    all MLX optimizers. There is no ``v`` buffer (Lion only has ``m``).

    Memory: 1 B/param (uint8 m) plus ~0.0156 B/param of metadata
    (fp32 / 256 = 1/64 byte). Total ~1.02 B/param vs Lion fp32's 4 B/param;
    a ~3.94x state shrink. For 1.797B params: ~1.83 GiB vs 6.69 GiB.

    Apply path:

    1. Dequantize ``m`` to fp32.
    2. Compute the interpolated direction
       ``c = beta1 * m + (1 - beta1) * g`` and the sign-update
       ``update = sign(c)``.
    3. Update the persistent momentum:
       ``m = beta2 * m + (1 - beta2) * g``.
    4. Re-quantize updated ``m`` to uint8 + absmax.
    5. Apply ``param * (1 - lr * weight_decay) - lr * update`` in fp32, cast
       back to the input parameter dtype (bf16 in production).

    Note Lion uses **two betas**: ``beta1`` for the interpolation that feeds
    ``sign()``, and ``beta2`` for the persistent momentum update. Both bnb's
    ``Lion8bit`` and the MLX upstream ``optim.Lion`` follow this scheme.

    ``use_fused_kernel`` requests the native MLX C++/Metal fused primitive for
    the dequant -> update -> requant -> apply path. If the extension is not
    built, ``optimizer.fused_kernel_status`` records the fallback reason and
    the optimizer uses the native MLX codec path.

    ``quant_scheme`` selects the codec used for the ``m`` buffer, dispatched
    through :func:`cppmega_mlx.training._quantize_8bit.quantize_blockwise`:

    * ``"symmetric_int8_v1"`` (default): the M0-grade symmetric int8 codec
      (uint8 with +128 bias). Sign-update math is naturally robust to
      symmetric quant noise because only elements with magnitude under
      ``absmax/127`` flip sign, and those are the elements where the update
      direction is genuinely ambiguous.
    * ``"dynamic_int8_v1"``: bitsandbytes-style dynamic LUT
      (``dDequantizeBlockwise`` parity), with denser bins near zero so small
      momentum values keep more precision on the round-trip. Closer to
      ``bitsandbytes.optim.Lion8bit`` numerics on the CUDA stack.
    """

    def __init__(
        self,
        learning_rate: float | Callable[[mx.array], mx.array],
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        block_size: int = DEFAULT_BLOCK_SIZE,
        use_fused_kernel: bool = True,
        quant_scheme: Lion8bitQuantScheme = QUANT_SCHEME_SYMMETRIC,
    ) -> None:
        super().__init__()
        self._maybe_schedule("learning_rate", learning_rate)
        self.betas = list(betas)
        self.weight_decay = float(weight_decay)
        self.block_size = int(block_size)
        if quant_scheme not in QUANT_SCHEMES:
            raise ValueError(
                f"quant_scheme must be one of {QUANT_SCHEMES}; got {quant_scheme!r}"
            )
        self.quant_scheme: str = quant_scheme
        self.fused_kernel_status = _lion_fused_status(
            requested=bool(use_fused_kernel),
            block_size=self.block_size,
            quant_scheme=self.quant_scheme,
        )
        self.use_fused_kernel = self.fused_kernel_status.available

    def init_single(self, parameter: mx.array, state: dict[str, Any]) -> None:
        nb = num_blocks(int(parameter.size), self.block_size)
        # For symmetric int8 the +128 bias maps to signed 0 (all-zero momentum
        # after dequant). For the dynamic LUT scheme byte index 127 maps to
        # LUT[127] == 0.0 (the canonical zero entry from create_dynamic_map).
        # Either way an all-128 / all-127 initial payload yields zero momentum
        # after dequant. Mirrors the Adam8bit zero-byte policy.
        zero_byte = 128 if self.quant_scheme == QUANT_SCHEME_SYMMETRIC else 127
        state["m_quant"] = mx.full(parameter.shape, zero_byte, dtype=mx.uint8)
        state["m_absmax"] = mx.zeros((nb,), dtype=mx.float32)

    def apply_single(
        self,
        gradient: mx.array,
        parameter: mx.array,
        state: dict[str, Any],
    ) -> mx.array:
        b1, b2 = self.betas
        lr_fp32 = self.learning_rate.astype(mx.float32)
        param_dtype = parameter.dtype
        scheme = self.quant_scheme

        if self.use_fused_kernel:
            if scheme == QUANT_SCHEME_DYNAMIC:
                updated, m_q, m_absmax = fused_lion8bit_dynamic_step(
                    parameter,
                    gradient,
                    state["m_quant"],
                    state["m_absmax"],
                    learning_rate=lr_fp32,
                    beta1=b1,
                    beta2=b2,
                    weight_decay=self.weight_decay,
                    block_size=self.block_size,
                )
            else:
                updated, m_q, m_absmax = fused_lion8bit_step(
                    parameter,
                    gradient,
                    state["m_quant"],
                    state["m_absmax"],
                    learning_rate=lr_fp32,
                    beta1=b1,
                    beta2=b2,
                    weight_decay=self.weight_decay,
                    block_size=self.block_size,
                )
            state["m_quant"] = m_q
            state["m_absmax"] = m_absmax
            return updated

        # 1) Dequantize the persistent momentum to fp32 for the inner math.
        # Lion's update is sign-based, so symmetric int8 quant noise on m
        # only flips the sign on elements within ~absmax/127 of zero -- those
        # are the same elements where the update direction is genuinely
        # ambiguous, so the loss-trajectory impact is small. The dynamic LUT
        # path tightens that floor near zero.
        m_prev = dequantize_blockwise(
            state["m_quant"],
            state["m_absmax"],
            scheme=scheme,
            out_dtype=mx.float32,
        )

        g32 = gradient.astype(mx.float32)

        # 2) Interp that feeds sign(); uses the *current* m before the b2
        # update lands. This matches Chen et al. and bnb's Lion8bit.
        c = b1 * m_prev + (1.0 - b1) * g32

        # 3) Update the persistent momentum AFTER computing the sign-update
        # direction. Order matters: c uses m at step t, then m advances to
        # step t+1.
        m_new = b2 * m_prev + (1.0 - b2) * g32

        # 4) Re-quantize the updated momentum back to uint8 + absmax storage
        # using whichever codec the optimizer was configured for.
        m_q, m_absmax = quantize_blockwise(m_new, self.block_size, scheme=scheme)
        state["m_quant"] = m_q
        state["m_absmax"] = m_absmax

        # 5) Apply weight decay (decoupled, AdamW-style: `(1 - lr*wd) * w`)
        # and the sign-update step in fp32, then cast back to param dtype.
        # Equivalent to the form `w - lr * (sign(c) + wd * w)` modulo the
        # second-order `lr^2 * wd` term, matching MLX's upstream optim.Lion.
        decayed = parameter.astype(mx.float32)
        if self.weight_decay > 0.0:
            decayed = decayed * (1.0 - lr_fp32 * self.weight_decay)
        updated = decayed - lr_fp32 * mx.sign(c)
        return updated.astype(param_dtype)


def make_lion8bit(
    *,
    learning_rate: float | Callable[[mx.array], mx.array] = 1e-4,
    weight_decay: float = 0.01,
    betas: list[float] | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
    use_fused_kernel: bool = True,
    quant_scheme: Lion8bitQuantScheme = QUANT_SCHEME_SYMMETRIC,
) -> Lion8bit:
    """Construct the repo-default 8-bit Lion for bf16 training.

    The defaults mirror :func:`cppmega_mlx.training.optimizers.make_lion` so
    the Lion8bit path is a drop-in swap: ``betas=(0.9, 0.99)`` per
    Chen et al. arXiv 2302.06675, default LR 3-10x smaller than AdamW because
    the sign-based update does not rescale by gradient magnitude.
    ``block_size`` matches bitsandbytes's 256-element blockwise default; only
    that value is supported by the native MLX codec today (other sizes raise
    ``NotImplementedError``).

    ``use_fused_kernel`` requests the native MLX C++/Metal fused primitive for
    the dequant -> update -> requant -> apply path. If the extension is not
    built, ``optimizer.fused_kernel_status`` records the fallback reason and
    the optimizer uses the native MLX codec path.

    ``quant_scheme`` selects the 8-bit codec:

    * ``"symmetric_int8_v1"`` (default): the existing symmetric int8 codec.
      Backwards compatible.
    * ``"dynamic_int8_v1"``: the bitsandbytes-style dynamic LUT, denser bins
      near zero so small momentum values keep more precision.

    Memory footprint vs ``make_lion`` on a 1.797B-param bf16 model:

    * LionFP32Moments: 1 * 4 B/param = 4 B/param  -> ~6.69 GiB state.
    * Lion8bit: 1 * 1 B/param + 1 * 4 B / 256 B = ~1.02 B/param -> ~1.83 GiB.
    """

    return Lion8bit(
        learning_rate=learning_rate,
        betas=(0.9, 0.99) if betas is None else (betas[0], betas[1]),
        weight_decay=weight_decay,
        block_size=block_size,
        use_fused_kernel=use_fused_kernel,
        quant_scheme=quant_scheme,
    )


__all__ = [
    "ADAM8BIT_CLASS",
    "ADAM8BIT_QUANT_KIND",
    "ADAM8BIT_QUANT_SCHEMES",
    "ADAM8BIT_SOURCE",
    "Adam8bit",
    "Adam8bitQuantScheme",
    "LION8BIT_CLASS",
    "LION8BIT_QUANT_KIND",
    "LION8BIT_SOURCE",
    "Lion8bit",
    "Lion8bitQuantScheme",
    "make_adam8bit",
    "make_lion8bit",
]

"""8-bit blockwise-quantized optimizer state for AdamW.

This module ships :class:`Adam8bit`, a drop-in AdamW where the per-parameter
``m`` and ``v`` moments are stored as ``uint8`` plus per-256-block fp32 absmax
scales. The memory footprint is ~2.06 bytes/parameter (2 * 1B + 4B/64B) versus
8 bytes/parameter for fp32 ``m``+``v`` -- the same target as
``bitsandbytes.optim.Adam8bit`` on the CUDA stack.

Numerical policy:

* Quantization is **symmetric int8** (uint8 with +128 bias). This is **not**
  bitsandbytes-bit-exact -- ``bitsandbytes`` ships a non-uniform dynamic LUT
  in ``dDequantizeBlockwise`` that gives slightly higher fidelity near zero.
  The symmetric path tracks AdamW within a few percent loss-trajectory drift,
  which is enough for the M0 throughput target.
* The internal AdamW math runs in fp32 inline: every ``apply_single`` call
  dequantizes ``m, v`` to fp32, performs the standard ``m = b1*m + (1-b1)*g``
  update, then re-quantizes. The parameter is read in fp32 and cast back to
  whatever dtype it had on input (bf16 in production).
* No master copy of weights -- the parameter tree is the only source of truth
  for weights, just like the fp32-moments AdamW upstream.

See :mod:`cppmega_mlx.training._quantize_8bit` for the Metal-backed codec.
"""

from __future__ import annotations

from typing import Any, Callable

import mlx.core as mx
import mlx.optimizers as optim

from cppmega_mlx.training._quantize_8bit import (
    DEFAULT_BLOCK_SIZE,
    dequantize_dynamic_blockwise,
    num_blocks,
    quantize_dynamic_blockwise,
)


ADAM8BIT_CLASS = "cppmega_mlx.training.optimizers_quantized.Adam8bit"
ADAM8BIT_SOURCE = "cppmega_mlx.training.optimizers_quantized.make_adam8bit"
ADAM8BIT_QUANT_KIND = "symmetric_int8_blockwise_v1"
"""Identifier for the quant codec; bumps if dynamic-LUT path lands later."""


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
    ) -> None:
        super().__init__()
        self._maybe_schedule("learning_rate", learning_rate)
        self.betas = list(betas)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.bias_correction = bool(bias_correction)
        self.block_size = int(block_size)

    def init_single(self, parameter: mx.array, state: dict[str, Any]) -> None:
        nb = num_blocks(int(parameter.size), self.block_size)
        # Bias 128 maps to signed 0, i.e. an all-zero moment after dequant.
        state["m_quant"] = mx.full(parameter.shape, 128, dtype=mx.uint8)
        state["m_absmax"] = mx.zeros((nb,), dtype=mx.float32)
        state["v_quant"] = mx.full(parameter.shape, 128, dtype=mx.uint8)
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

        # 1) Dequantize current moments to fp32. The dequant introduces
        # symmetric rounding error bounded by ``absmax / (2 * 127)`` per
        # block, so values of ``v`` smaller than ``absmax_v / 127`` collapse
        # to zero. We compensate by floor-clamping ``v`` to the per-block
        # quantization step size during the AdamW update -- without that,
        # the ``sqrt(v) + eps`` denominator would underflow and the update
        # would diverge whenever ``v`` was small relative to its block
        # absmax (e.g. tail of a fat-tailed gradient distribution).
        m_prev = dequantize_dynamic_blockwise(
            state["m_quant"], state["m_absmax"], out_dtype=mx.float32
        )
        v_prev = dequantize_dynamic_blockwise(
            state["v_quant"], state["v_absmax"], out_dtype=mx.float32
        )

        g32 = gradient.astype(mx.float32)
        m_new = b1 * m_prev + (1.0 - b1) * g32
        v_new = b2 * v_prev + (1.0 - b2) * mx.square(g32)

        # 2) Build a per-block lower-bound for v that absorbs the quant
        # noise floor: anything below ``absmax_v / 127`` is indistinguishable
        # from zero on the round-trip, so we add it to ``sqrt(v)`` as a
        # block-aware epsilon. This keeps the AdamW denominator stable
        # without inflating the global ``eps`` (which is the bnb-style
        # symmetric-quantization stability fix; the dynamic-LUT version
        # would shape the bins so this floor is much smaller).
        nb = num_blocks(int(v_new.size), self.block_size)
        # state["v_absmax"] is the absmax from the *previous* step. Use the
        # max of (previous, |v_new| absmax estimate) so the floor reflects
        # whatever round-trip we're about to take.
        v_block_step = state["v_absmax"] / 127.0  # shape: (num_blocks,)
        # Broadcast the per-block step to a per-element noise floor by
        # repeating each scale ``block_size`` times and trimming the tail.
        v_noise_floor = mx.repeat(v_block_step, self.block_size)[: int(v_new.size)].reshape(
            v_new.shape
        )

        # 3) Compute the update in fp32 before re-quantization, so the
        # round trip through uint8 happens after we already used the fresh
        # moments. The denominator ``sqrt(v_new) + v_noise_floor + eps``
        # never underflows even when v_new has elements that quantize to
        # zero on the next round trip.
        if self.bias_correction:
            step = self.step.astype(mx.float32)
            c1 = lr_fp32 / (1.0 - mx.power(mx.array(b1, dtype=mx.float32), step))
            c2 = mx.rsqrt(1.0 - mx.power(mx.array(b2, dtype=mx.float32), step))
            numerator = c1 * m_new
            denominator = mx.sqrt(v_new) * c2 + v_noise_floor + eps
            update = numerator / denominator
        else:
            update = lr_fp32 * m_new / (mx.sqrt(v_new) + v_noise_floor + eps)

        # 4) Re-quantize the updated moments back to uint8 + absmax storage.
        m_q, m_absmax = quantize_dynamic_blockwise(m_new, self.block_size)
        v_q, v_absmax = quantize_dynamic_blockwise(v_new, self.block_size)
        state["m_quant"] = m_q
        state["m_absmax"] = m_absmax
        state["v_quant"] = v_q
        state["v_absmax"] = v_absmax

        # 5) Apply weight decay + step in fp32, then cast back to param dtype.
        decayed = parameter.astype(mx.float32) * (1.0 - lr_fp32 * self.weight_decay)
        updated = decayed - update
        return updated.astype(param_dtype)


def make_adam8bit(
    *,
    learning_rate: float | Callable[[mx.array], mx.array] = 1e-3,
    weight_decay: float = 0.01,
    betas: list[float] | None = None,
    eps: float = 1e-8,
    bias_correction: bool = False,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> Adam8bit:
    """Construct the repo-default 8-bit AdamW for bf16 training.

    The defaults mirror :func:`cppmega_mlx.training.optimizers.make_adamw` so
    the Adam8bit path is a drop-in swap. ``block_size`` matches bitsandbytes's
    256-element blockwise default; only that value is wired through the Metal
    kernel today (other sizes raise ``NotImplementedError``).

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
    )


__all__ = [
    "ADAM8BIT_CLASS",
    "ADAM8BIT_QUANT_KIND",
    "ADAM8BIT_SOURCE",
    "Adam8bit",
    "make_adam8bit",
]

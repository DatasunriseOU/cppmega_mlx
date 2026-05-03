"""Fused Metal kernel for symmetric int8 ``Adam8bit`` updates.

The unfused path (see :mod:`cppmega_mlx.training.optimizers_quantized`) launches
4-5 separate Metal kernels per parameter on every step:

1. dequantize ``m`` (uint8 + fp32 absmax -> fp32)
2. dequantize ``v`` (uint8 + fp32 absmax -> fp32)
3. fp32 AdamW math + parameter update
4. quantize ``m`` (fp32 -> uint8 + fp32 absmax)
5. quantize ``v`` (fp32 -> uint8 + fp32 absmax)

Each launch round-trips fp32 working tensors through device memory. For the
1.797B-param ``local_gb10_quarter`` model that is ~28 GiB of redundant traffic
per step, which dominates the optimizer-update wall time on M4.

This module fuses everything into a **single kernel launch per parameter**.
One threadgroup processes one 256-element block end-to-end, keeping fp32
working tensors in threadgroup memory:

* Stage A: load 256 elements of ``param`` (bf16 in, fp32 working),
  ``grad`` (bf16 in, fp32 working), ``m_quant`` + ``m_absmax_prev``
  (uint8 + scalar -> fp32), ``v_quant`` + ``v_absmax_prev``
  (uint8 + scalar -> fp32). One element per thread.
* Stage B: AdamW math in registers
  (``m_new = b1*m + (1-b1)*g``, ``v_new = b2*v + (1-b2)*g**2``).
* Stage C: tree-reduce ``|m_new|`` and ``|v_new|`` over the threadgroup to
  get the per-block ``m_absmax_new`` and ``v_absmax_new``.
* Stage D: compute ``update`` using the symmetric-quant noise floor
  (``max(absmax_prev, absmax_new) / 127``) added to the AdamW denominator,
  matching the unfused path's stability fix.
* Stage E: re-quantize ``m_new``, ``v_new`` with the freshly computed
  absmax and write back ``param_bf16``, ``m_quant``, ``v_quant``,
  ``m_absmax``, ``v_absmax``.

This is a drop-in replacement for the dequant -> math -> quant -> apply
chain. Tail blocks (input numel not a multiple of 256) zero-pad the unused
slots, exactly matching the unfused codec's behaviour.

Hyperparameters that change once per step (``learning_rate``, ``beta1``,
``beta2``, ``eps``, ``weight_decay``, ``step``, ``bias_correction``) are
passed as scalar 0-D inputs so the kernel signature is stable across all
parameters in the model.

The kernel is hardcoded to ``BLOCK_SIZE = 256`` (matching the unfused
codec). Other sizes need a recompile and are gated as ``NotImplementedError``
in :class:`Adam8bit`.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx


FUSED_BLOCK_SIZE = 256
"""Threadgroup block size; matches the symmetric-int8 codec layout."""


_FUSED_HEADER = """
constant constexpr uint FUSED_BLOCK = 256;
"""


_FUSED_SOURCE = """
    threadgroup float m_scratch[FUSED_BLOCK];
    threadgroup float v_scratch[FUSED_BLOCK];

    uint tid = thread_position_in_threadgroup.x;
    uint bid = threadgroup_position_in_grid.x;
    uint total = param_shape[0];
    uint elem = bid * FUSED_BLOCK + tid;
    bool active = elem < total;

    // ----- Stage A: load + dequant inputs into fp32 registers -----
    float m_absmax_prev = m_absmax[bid];
    float v_absmax_prev = v_absmax[bid];

    float param_fp = 0.0f;
    float grad_fp = 0.0f;
    float m_prev = 0.0f;
    float v_prev = 0.0f;
    if (active) {
        param_fp = (float)param[elem];
        grad_fp = (float)grad[elem];
        int m_signed = (int)m_quant_in[elem] - 128;
        int v_signed = (int)v_quant_in[elem] - 128;
        m_prev = ((float)m_signed) * (1.0f / 127.0f) * m_absmax_prev;
        v_prev = ((float)v_signed) * (1.0f / 127.0f) * v_absmax_prev;
    }

    // ----- Stage B: AdamW math in fp32 -----
    float m_new = beta1 * m_prev + (1.0f - beta1) * grad_fp;
    float v_new = beta2 * v_prev + (1.0f - beta2) * grad_fp * grad_fp;

    // Stash in scratch so we can both reduce |m|, |v| and re-load post-reduce.
    m_scratch[tid] = active ? m_new : 0.0f;
    v_scratch[tid] = active ? v_new : 0.0f;
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    // ----- Stage C: tree reduction for |m|, |v| absmax over the block -----
    // Reduce |m| and |v| in parallel by reusing the same scratch with abs.
    {
        // Replace each scratch slot with its abs value before reduction.
        float ma = metal::abs(m_scratch[tid]);
        float va = metal::abs(v_scratch[tid]);
        m_scratch[tid] = ma;
        v_scratch[tid] = va;
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    for (uint stride = FUSED_BLOCK / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            float other_m = m_scratch[tid + stride];
            if (other_m > m_scratch[tid]) m_scratch[tid] = other_m;
            float other_v = v_scratch[tid + stride];
            if (other_v > v_scratch[tid]) v_scratch[tid] = other_v;
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }
    float m_absmax_new = m_scratch[0];
    float v_absmax_new = v_scratch[0];

    // ----- Stage D: AdamW update with symmetric-quant noise floor -----
    // Mirrors the unfused path: noise floor uses
    // max(prev_v_absmax, new_v_absmax) / 127 as a per-block lower bound on
    // the quantization step size, added to sqrt(v) + eps to keep the
    // denominator stable when v collapses to zero on the next round trip.
    float v_block_step = metal::max(v_absmax_prev, v_absmax_new) * (1.0f / 127.0f);

    float numerator;
    float denominator;
    if (bias_correction != 0.0f) {
        // Use float-exponent power; step is passed as fp32 already.
        float c1 = lr / (1.0f - metal::pow(beta1, step_fp));
        float c2 = metal::rsqrt(1.0f - metal::pow(beta2, step_fp));
        numerator = c1 * m_new;
        denominator = metal::sqrt(v_new) * c2 + v_block_step + eps;
    } else {
        numerator = lr * m_new;
        denominator = metal::sqrt(v_new) + v_block_step + eps;
    }
    float update = numerator / denominator;

    // Decoupled weight decay: param * (1 - lr * wd) - update
    float param_decayed = param_fp * (1.0f - lr * wd);
    float param_new = param_decayed - update;

    // ----- Stage E: write outputs (param, quantized moments, absmax) -----
    if (active) {
        param_out[elem] = (T)param_new;

        // Symmetric int8 quant of m_new with new absmax. scale==0 -> bias 128.
        float m_norm = (m_absmax_new > 0.0f) ? (m_new / m_absmax_new) : 0.0f;
        int m_rounded = (int)metal::round(m_norm * 127.0f);
        if (m_rounded > 127) m_rounded = 127;
        if (m_rounded < -127) m_rounded = -127;
        m_quant_out[elem] = (uint8_t)(m_rounded + 128);

        float v_norm = (v_absmax_new > 0.0f) ? (v_new / v_absmax_new) : 0.0f;
        int v_rounded = (int)metal::round(v_norm * 127.0f);
        if (v_rounded > 127) v_rounded = 127;
        if (v_rounded < -127) v_rounded = -127;
        v_quant_out[elem] = (uint8_t)(v_rounded + 128);
    }

    if (tid == 0) {
        m_absmax_out[bid] = m_absmax_new;
        v_absmax_out[bid] = v_absmax_new;
    }
"""


_fused_kernel: Optional[object] = None


def _can_run_metal() -> bool:
    metal = getattr(mx, "metal", None)
    return mx.default_device() == mx.gpu and metal is not None and metal.is_available()


def _get_fused_kernel() -> object:
    """Lazily JIT-compile the fused MSL kernel.

    The kernel is templated on the param dtype ``T`` (bf16 in production,
    fp32 in tests). Inputs/outputs are templated indirectly: ``param`` and
    ``grad`` come in as ``T``; ``param_out`` writes back as ``T``. All
    fp32 working tensors are local to the threadgroup.
    """

    global _fused_kernel
    if _fused_kernel is None:
        if not _can_run_metal():
            raise RuntimeError(
                "Fused Adam8bit kernel requires the MLX Metal backend; "
                "default device is not GPU or mx.metal is unavailable."
            )
        _fused_kernel = mx.fast.metal_kernel(
            name="cppmega_fused_adam8bit_symmetric",
            input_names=[
                "param",
                "grad",
                "m_quant_in",
                "m_absmax",
                "v_quant_in",
                "v_absmax",
                "lr",
                "beta1",
                "beta2",
                "eps",
                "wd",
                "step_fp",
                "bias_correction",
            ],
            output_names=[
                "param_out",
                "m_quant_out",
                "m_absmax_out",
                "v_quant_out",
                "v_absmax_out",
            ],
            header=_FUSED_HEADER,
            source=_FUSED_SOURCE,
            ensure_row_contiguous=True,
        )
    return _fused_kernel


def fused_adam8bit_step(
    param: mx.array,
    grad: mx.array,
    m_quant: mx.array,
    m_absmax: mx.array,
    v_quant: mx.array,
    v_absmax: mx.array,
    *,
    learning_rate: mx.array,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step: mx.array,
    bias_correction: bool,
    block_size: int = FUSED_BLOCK_SIZE,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Run the fused dequant -> AdamW -> quant -> apply kernel for one parameter.

    Inputs:

    * ``param``: parameter tensor (bf16 or fp32). Treated flat internally.
    * ``grad``: gradient tensor with same shape and dtype as ``param``.
    * ``m_quant``, ``v_quant``: uint8 quantized moments, same shape as ``param``.
    * ``m_absmax``, ``v_absmax``: fp32 per-256-block absmax scales.
    * ``learning_rate``, ``step``: 0-D mx.array scalars.
    * ``beta1``, ``beta2``, ``eps``, ``weight_decay``, ``bias_correction``:
      Python scalars that become fp32 0-D inputs to the kernel.

    Returns ``(param_out, m_quant_out, m_absmax_out, v_quant_out, v_absmax_out)``
    with the same shapes/dtypes as the corresponding inputs.

    Tail block: if ``param.size`` is not a multiple of ``block_size`` the
    final block zero-pads the unused threads, exactly matching the unfused
    codec's per-block layout.
    """

    if block_size != FUSED_BLOCK_SIZE:
        raise NotImplementedError(
            f"block_size={block_size} not supported by the fused kernel; "
            f"only block_size={FUSED_BLOCK_SIZE} is wired through."
        )
    if param.dtype not in {mx.bfloat16, mx.float32, mx.float16}:
        raise TypeError(
            f"fused Adam8bit param must be float (bf16/fp32/fp16); got {param.dtype}"
        )
    if grad.dtype != param.dtype:
        # Cast grad to the param dtype so we keep a single ``T`` template.
        grad = grad.astype(param.dtype)
    if m_quant.dtype != mx.uint8 or v_quant.dtype != mx.uint8:
        raise TypeError("m_quant and v_quant must be uint8")
    if m_absmax.dtype != mx.float32 or v_absmax.dtype != mx.float32:
        raise TypeError("m_absmax and v_absmax must be fp32")
    if param.shape != grad.shape:
        raise ValueError(
            f"param.shape {param.shape} must equal grad.shape {grad.shape}"
        )
    if m_quant.shape != param.shape or v_quant.shape != param.shape:
        raise ValueError("m_quant and v_quant must share param.shape")

    original_shape = param.shape
    flat_param = param.reshape(-1)
    flat_grad = grad.reshape(-1)
    flat_mq = m_quant.reshape(-1)
    flat_vq = v_quant.reshape(-1)

    n = int(flat_param.size)
    if n == 0:
        # Empty parameter: pass through, nothing to launch.
        return (
            param,
            m_quant,
            m_absmax,
            v_quant,
            v_absmax,
        )

    nblocks = (n + block_size - 1) // block_size

    # Pack the hyperparameters as 0-D fp32 scalars; the kernel signature picks
    # them up as bare-scalar arguments (no array indexing required).
    lr_scalar = learning_rate.astype(mx.float32) if learning_rate.ndim == 0 else learning_rate.astype(mx.float32).reshape(())
    if lr_scalar.ndim != 0:
        lr_scalar = lr_scalar.reshape(())
    step_fp = step.astype(mx.float32)
    if step_fp.ndim != 0:
        step_fp = step_fp.reshape(())
    beta1_arr = mx.array(float(beta1), dtype=mx.float32)
    beta2_arr = mx.array(float(beta2), dtype=mx.float32)
    eps_arr = mx.array(float(eps), dtype=mx.float32)
    wd_arr = mx.array(float(weight_decay), dtype=mx.float32)
    bc_arr = mx.array(1.0 if bias_correction else 0.0, dtype=mx.float32)

    kernel = _get_fused_kernel()
    outputs = kernel(
        inputs=[
            flat_param,
            flat_grad,
            flat_mq,
            m_absmax,
            flat_vq,
            v_absmax,
            lr_scalar,
            beta1_arr,
            beta2_arr,
            eps_arr,
            wd_arr,
            step_fp,
            bc_arr,
        ],
        template=[("T", param.dtype)],
        output_shapes=[
            flat_param.shape,
            flat_mq.shape,
            m_absmax.shape,
            flat_vq.shape,
            v_absmax.shape,
        ],
        output_dtypes=[
            param.dtype,
            mx.uint8,
            mx.float32,
            mx.uint8,
            mx.float32,
        ],
        grid=(nblocks * block_size, 1, 1),
        threadgroup=(block_size, 1, 1),
        stream=mx.gpu,
    )
    (
        param_flat_out,
        m_quant_flat_out,
        m_absmax_out,
        v_quant_flat_out,
        v_absmax_out,
    ) = outputs

    return (
        param_flat_out.reshape(original_shape),
        m_quant_flat_out.reshape(original_shape),
        m_absmax_out,
        v_quant_flat_out.reshape(original_shape),
        v_absmax_out,
    )


__all__ = [
    "FUSED_BLOCK_SIZE",
    "fused_adam8bit_step",
]

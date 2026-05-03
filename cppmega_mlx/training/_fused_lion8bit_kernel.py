"""Fused Metal kernel for symmetric int8 ``Lion8bit`` updates.

The unfused path (see :mod:`cppmega_mlx.training.optimizers_quantized`)
launches three separate Metal kernels per parameter on every step:

1. dequantize ``m`` (uint8 + fp32 absmax -> fp32)
2. fp32 Lion math (``c = b1*m + (1-b1)*g``, ``sign(c)``, ``m = b2*m + (1-b2)*g``)
   + parameter update
3. quantize ``m`` (fp32 -> uint8 + fp32 absmax)

Each launch round-trips fp32 working tensors through device memory. For the
1.797B-param ``local_gb10_quarter`` model that is a lot of redundant traffic
per step, which dominates the optimizer-update wall time on M4.

This module fuses everything into a **single kernel launch per parameter**.
One threadgroup processes one 256-element block end-to-end, keeping fp32
working tensors in threadgroup memory:

* Stage A: load 256 elements of ``param`` (bf16 in, fp32 working),
  ``grad`` (bf16 in, fp32 working) and ``m_quant`` + ``m_absmax_prev``
  (uint8 + scalar -> fp32). One element per thread.
* Stage B: Lion math in registers
  (``c = b1*m + (1-b1)*g``, ``update = sign(c)``,
  ``m_new = b2*m + (1-b2)*g``).
* Stage C: apply ``param_new = param * (1 - lr*wd) - lr * sign(c)`` and
  write back the bf16 parameter.
* Stage D: tree-reduce ``|m_new|`` over the threadgroup to produce the
  per-block ``m_absmax_new``.
* Stage E: re-quantize ``m_new`` with the freshly computed absmax and
  write back ``m_quant`` and ``m_absmax``.

This is a drop-in replacement for the dequant -> math -> quant -> apply
chain. Tail blocks (input numel not a multiple of 256) zero-pad the unused
slots, exactly matching the unfused codec's behaviour.

Hyperparameters that change once per step (``learning_rate``, ``beta1``,
``beta2``, ``weight_decay``) are passed as scalar 0-D inputs so the kernel
signature is stable across all parameters in the model.

The kernel is hardcoded to ``BLOCK_SIZE = 256`` (matching the unfused
codec). Other sizes need a recompile and are gated as ``NotImplementedError``
in :class:`Lion8bit`.

Lion's update is sign-only (no magnitude), so the per-block "noise floor"
that the Adam fused kernel uses for ``sqrt(v) + eps`` stability is not
needed here. The only place quant noise can flip the result is when an
element of ``c = b1*m + (1-b1)*g`` is within ~``absmax/127`` of zero, and
those are the elements where the update direction is genuinely ambiguous.
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

    uint tid = thread_position_in_threadgroup.x;
    uint bid = threadgroup_position_in_grid.x;
    uint total = param_shape[0];
    uint elem = bid * FUSED_BLOCK + tid;
    bool active = elem < total;

    // ----- Stage A: load + dequant inputs into fp32 registers -----
    float m_absmax_prev = m_absmax[bid];

    float param_fp = 0.0f;
    float grad_fp = 0.0f;
    float m_prev = 0.0f;
    if (active) {
        param_fp = (float)param[elem];
        grad_fp = (float)grad[elem];
        int m_signed = (int)m_quant_in[elem] - 128;
        m_prev = ((float)m_signed) * (1.0f / 127.0f) * m_absmax_prev;
    }

    // ----- Stage B: Lion math in fp32 -----
    // ``c`` feeds sign(); uses the *current* m before the b2 update lands.
    // Matches Chen et al. (arXiv 2302.06675) and bnb.optim.Lion8bit.
    float c = beta1 * m_prev + (1.0f - beta1) * grad_fp;
    // Update the persistent momentum AFTER computing c. Order matters: c
    // uses m at step t, then m advances to step t+1 with beta2.
    float m_new = beta2 * m_prev + (1.0f - beta2) * grad_fp;

    // ----- Stage C: apply parameter update with decoupled weight decay -----
    // metal::sign returns 0 for input 0, matching mx.sign and bnb's reference.
    // Decoupled weight decay (AdamW-style): ``(1 - lr*wd) * w - lr * sign(c)``.
    float sign_c = metal::sign(c);
    float param_new = param_fp * (1.0f - lr * wd) - lr * sign_c;

    if (active) {
        param_out[elem] = (T)param_new;
    }

    // Stash m_new in scratch so we can both reduce |m_new| and re-load
    // post-reduce for re-quantization.
    m_scratch[tid] = active ? m_new : 0.0f;
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    // ----- Stage D: tree reduction for |m_new| absmax over the block -----
    // Read m_new, replace scratch with |m_new| for the reduction, but keep
    // a register copy of the signed m_new for the requantize step below.
    float m_new_signed = m_scratch[tid];
    m_scratch[tid] = metal::abs(m_new_signed);
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    for (uint stride = FUSED_BLOCK / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            float other = m_scratch[tid + stride];
            if (other > m_scratch[tid]) m_scratch[tid] = other;
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }
    float m_absmax_new = m_scratch[0];

    // ----- Stage E: re-quantize m_new with new absmax and write back -----
    if (active) {
        // Symmetric int8 quant of m_new with new absmax. scale==0 -> bias 128.
        float m_norm = (m_absmax_new > 0.0f) ? (m_new_signed / m_absmax_new) : 0.0f;
        int m_rounded = (int)metal::round(m_norm * 127.0f);
        if (m_rounded > 127) m_rounded = 127;
        if (m_rounded < -127) m_rounded = -127;
        m_quant_out[elem] = (uint8_t)(m_rounded + 128);
    }

    if (tid == 0) {
        m_absmax_out[bid] = m_absmax_new;
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
                "Fused Lion8bit kernel requires the MLX Metal backend; "
                "default device is not GPU or mx.metal is unavailable."
            )
        _fused_kernel = mx.fast.metal_kernel(
            name="cppmega_fused_lion8bit_symmetric",
            input_names=[
                "param",
                "grad",
                "m_quant_in",
                "m_absmax",
                "lr",
                "beta1",
                "beta2",
                "wd",
            ],
            output_names=[
                "param_out",
                "m_quant_out",
                "m_absmax_out",
            ],
            header=_FUSED_HEADER,
            source=_FUSED_SOURCE,
            ensure_row_contiguous=True,
        )
    return _fused_kernel


def fused_lion8bit_step(
    param: mx.array,
    grad: mx.array,
    m_quant: mx.array,
    m_absmax: mx.array,
    *,
    learning_rate: mx.array,
    beta1: float,
    beta2: float,
    weight_decay: float,
    block_size: int = FUSED_BLOCK_SIZE,
) -> tuple[mx.array, mx.array, mx.array]:
    """Run the fused dequant -> Lion -> quant -> apply kernel for one parameter.

    Inputs:

    * ``param``: parameter tensor (bf16 or fp32). Treated flat internally.
    * ``grad``: gradient tensor with same shape and dtype as ``param``.
    * ``m_quant``: uint8 quantized momentum, same shape as ``param``.
    * ``m_absmax``: fp32 per-256-block absmax scales.
    * ``learning_rate``: 0-D mx.array scalar.
    * ``beta1``, ``beta2``, ``weight_decay``: Python scalars that become
      fp32 0-D inputs to the kernel.

    Returns ``(param_out, m_quant_out, m_absmax_out)`` with the same shapes
    and dtypes as the corresponding inputs.

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
            f"fused Lion8bit param must be float (bf16/fp32/fp16); got {param.dtype}"
        )
    if grad.dtype != param.dtype:
        # Cast grad to the param dtype so we keep a single ``T`` template.
        grad = grad.astype(param.dtype)
    if m_quant.dtype != mx.uint8:
        raise TypeError("m_quant must be uint8")
    if m_absmax.dtype != mx.float32:
        raise TypeError("m_absmax must be fp32")
    if param.shape != grad.shape:
        raise ValueError(
            f"param.shape {param.shape} must equal grad.shape {grad.shape}"
        )
    if m_quant.shape != param.shape:
        raise ValueError("m_quant must share param.shape")

    original_shape = param.shape
    flat_param = param.reshape(-1)
    flat_grad = grad.reshape(-1)
    flat_mq = m_quant.reshape(-1)

    n = int(flat_param.size)
    if n == 0:
        # Empty parameter: pass through, nothing to launch.
        return (param, m_quant, m_absmax)

    nblocks = (n + block_size - 1) // block_size

    # Pack the hyperparameters as 0-D fp32 scalars; the kernel signature picks
    # them up as bare-scalar arguments (no array indexing required).
    lr_scalar = learning_rate.astype(mx.float32)
    if lr_scalar.ndim != 0:
        lr_scalar = lr_scalar.reshape(())
    beta1_arr = mx.array(float(beta1), dtype=mx.float32)
    beta2_arr = mx.array(float(beta2), dtype=mx.float32)
    wd_arr = mx.array(float(weight_decay), dtype=mx.float32)

    kernel = _get_fused_kernel()
    outputs = kernel(
        inputs=[
            flat_param,
            flat_grad,
            flat_mq,
            m_absmax,
            lr_scalar,
            beta1_arr,
            beta2_arr,
            wd_arr,
        ],
        template=[("T", param.dtype)],
        output_shapes=[
            flat_param.shape,
            flat_mq.shape,
            m_absmax.shape,
        ],
        output_dtypes=[
            param.dtype,
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
    ) = outputs

    return (
        param_flat_out.reshape(original_shape),
        m_quant_flat_out.reshape(original_shape),
        m_absmax_out,
    )


__all__ = [
    "FUSED_BLOCK_SIZE",
    "fused_lion8bit_step",
]

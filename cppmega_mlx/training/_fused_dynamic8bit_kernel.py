"""Fused Metal kernels for dynamic-LUT 8-bit Adam and Lion updates.

The unfused dynamic-LUT path (see :mod:`cppmega_mlx.training.optimizers_quantized`)
launches 4-5 separate Metal kernels per parameter on every step:

1. dequantize ``m`` (uint8 LUT-index + fp32 absmax + fp32 LUT -> fp32)
2. dequantize ``v`` (uint8 LUT-index + fp32 absmax + fp32 LUT -> fp32)
3. fp32 AdamW math + parameter update (Lion: only ``m``)
4. quantize ``m`` (fp32 -> uint8 LUT-index + fp32 absmax via binary search)
5. quantize ``v`` (fp32 -> uint8 LUT-index + fp32 absmax via binary search)

Each launch round-trips fp32 working tensors through device memory plus pulls
the 256-entry LUT from constant memory on every kernel call. For the 1.797B-
param ``local_gb10_quarter`` model that's ~28 GiB of redundant traffic per step
which dominates wall time on M4 (~1041 ms/step versus ~223 ms/step for the
fused symmetric path).

This module fuses everything into a **single kernel launch per parameter**.
One threadgroup processes one 256-element block end-to-end, keeping fp32
working tensors *and the dynamic LUT* in threadgroup memory:

* Stage 0: thread ``tid`` cooperatively loads ``lut_tg[tid] = lut[tid]`` so
  every threadgroup keeps its own copy of the 256-entry fp32 LUT (1024 B).
* Stage A: load 256 elements of ``param`` (bf16 in, fp32 working),
  ``grad`` (bf16 in, fp32 working), ``m_quant`` + ``m_absmax_prev``
  (uint8 + scalar -> fp32 via ``lut[m_quant] * absmax``),
  and (Adam only) ``v_quant`` + ``v_absmax_prev``. One element per thread.
* Stage B: optimizer math in registers
  - Adam:  ``m_new = b1 * m + (1 - b1) * g``,
           ``v_new = b2 * v + (1 - b2) * g**2``.
  - Lion:  ``c     = b1 * m + (1 - b1) * g``  (feeds sign update),
           ``m_new = b2 * m + (1 - b2) * g``  (persistent momentum).
* Stage C: tree-reduce ``|m_new|`` (and ``|v_new|`` for Adam) over the
  threadgroup to get the per-block ``m_absmax_new`` (and ``v_absmax_new``).
* Stage D: compute ``update`` (Adam: AdamW with the symmetric-quant noise
  floor ``max(absmax_prev, absmax_new) / 127`` for v; Lion: ``lr * sign(c)``).
* Stage E: re-quantize via **binary search in threadgroup-resident LUT** -
  log2(256)=8 comparisons per element, all from threadgroup memory. Write
  back ``param_bf16``, ``m_quant`` (and ``v_quant`` for Adam), updated absmax.

This is a drop-in replacement for the dequant -> math -> quant -> apply
chain when ``quant_scheme == "dynamic_int8_v1"`` and ``block_size == 256``.

Tail blocks (input numel not a multiple of 256) zero-pad the unused slots,
exactly matching the unfused codec's behaviour.

Hyperparameters that change once per step (``learning_rate``, ``beta1``,
``beta2``, ``eps``, ``weight_decay``, ``step``, ``bias_correction``) are
passed as scalar 0-D inputs so the kernel signature is stable across all
parameters in the model.

The kernel is hardcoded to ``BLOCK_SIZE = 256`` (matching the codec layout).
Other sizes need a recompile and are gated as ``NotImplementedError`` in the
optimizer dispatcher.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx

from cppmega_mlx.training._quantize_8bit import _get_lut


FUSED_BLOCK_SIZE = 256
"""Threadgroup block size; matches the dynamic-LUT codec layout."""


# ---------------------------------------------------------------------------
# Common header + LUT helper.
# ---------------------------------------------------------------------------
#
# Both kernels share the same LUT-loading + binary-search helper code.

_FUSED_DYNAMIC_HEADER = """
constant constexpr uint FUSED_BLOCK = 256;
constant constexpr uint LUT_SIZE = 256;
"""


# ---------------------------------------------------------------------------
# Adam8bit fused dynamic kernel source.
# ---------------------------------------------------------------------------
#
# Layout matches _fused_adam8bit_kernel.py but swaps:
#   * dequant: from ``(qbyte - 128) / 127 * absmax`` to ``lut_tg[qbyte] * absmax``.
#   * quant:   from ``round(x / absmax * 127) + 128`` to a binary search over
#              ``lut_tg`` for the closest entry to ``x / absmax``.
# Everything else (tree-reduction for absmax, AdamW math, weight-decay apply,
# noise-floor for v) is identical.

_FUSED_ADAM_DYNAMIC_SOURCE = """
    threadgroup float lut_tg[LUT_SIZE];
    threadgroup float m_scratch[FUSED_BLOCK];
    threadgroup float v_scratch[FUSED_BLOCK];

    uint tid = thread_position_in_threadgroup.x;
    uint bid = threadgroup_position_in_grid.x;
    uint total = param_shape[0];
    uint elem = bid * FUSED_BLOCK + tid;
    bool active = elem < total;

    // ----- Stage 0: cooperatively load the 256-entry LUT into threadgroup mem -----
    lut_tg[tid] = lut[tid];
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

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
        // Dynamic LUT dequant: lut[qbyte] * absmax. lut is the 256-entry signed
        // dynamic map produced by bnb.create_dynamic_map(signed=True, ...).
        uint m_idx = (uint)m_quant_in[elem];
        uint v_idx = (uint)v_quant_in[elem];
        m_prev = lut_tg[m_idx] * m_absmax_prev;
        v_prev = lut_tg[v_idx] * v_absmax_prev;
        if (v_prev < 0.0f) v_prev = 0.0f;
    }

    // ----- Stage B: AdamW math in fp32 -----
    float m_new = beta1 * m_prev + (1.0f - beta1) * grad_fp;
    float v_new = beta2 * v_prev + (1.0f - beta2) * grad_fp * grad_fp;
    if (v_new < 0.0f) v_new = 0.0f;

    // Stash in scratch so we can both reduce |m|, |v| and re-load post-reduce.
    m_scratch[tid] = active ? m_new : 0.0f;
    v_scratch[tid] = active ? v_new : 0.0f;
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    // ----- Stage C: tree reduction for |m|, |v| absmax over the block -----
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
    // Same noise floor as the symmetric fused path: max(prev, new) / 127.
    // The dynamic LUT path technically has tighter near-zero bins (~5.5e-7)
    // but using the symmetric bound here is conservative and matches the
    // unfused dynamic chain exactly so the optimizer math stays identical.
    float v_block_step = metal::max(v_absmax_prev, v_absmax_new) * (1.0f / 127.0f);

    float numerator;
    float denominator;
    if (bias_correction != 0.0f) {
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

        // Dynamic-LUT quant of m_new with new absmax via binary search in
        // threadgroup memory. The LUT is monotone non-decreasing so we walk
        // it in log2(256)=8 comparisons, then pick the closer of [lo, lo-1].
        float m_norm = (m_absmax_new > 0.0f) ? (m_new / m_absmax_new) : 0.0f;
        if (m_norm > 1.0f) m_norm = 1.0f;
        if (m_norm < -1.0f) m_norm = -1.0f;
        uint m_lo = 0u;
        uint m_hi = LUT_SIZE - 1u;
        while (m_lo < m_hi) {
            uint mid = (m_lo + m_hi) >> 1;
            if (lut_tg[mid] < m_norm) {
                m_lo = mid + 1u;
            } else {
                m_hi = mid;
            }
        }
        uint m_best = m_lo;
        if (m_lo > 0u) {
            float d_hi = metal::abs(lut_tg[m_lo] - m_norm);
            float d_lo = metal::abs(lut_tg[m_lo - 1u] - m_norm);
            if (d_lo < d_hi) m_best = m_lo - 1u;
        }
        m_quant_out[elem] = (uint8_t)m_best;

        float v_norm = (v_absmax_new > 0.0f) ? (v_new / v_absmax_new) : 0.0f;
        if (v_norm > 1.0f) v_norm = 1.0f;
        if (v_norm < -1.0f) v_norm = -1.0f;
        uint v_lo = 0u;
        uint v_hi = LUT_SIZE - 1u;
        while (v_lo < v_hi) {
            uint mid = (v_lo + v_hi) >> 1;
            if (lut_tg[mid] < v_norm) {
                v_lo = mid + 1u;
            } else {
                v_hi = mid;
            }
        }
        uint v_best = v_lo;
        if (v_lo > 0u) {
            float d_hi = metal::abs(lut_tg[v_lo] - v_norm);
            float d_lo = metal::abs(lut_tg[v_lo - 1u] - v_norm);
            if (d_lo < d_hi) v_best = v_lo - 1u;
        }
        v_quant_out[elem] = (uint8_t)v_best;
    }

    if (tid == 0) {
        m_absmax_out[bid] = m_absmax_new;
        v_absmax_out[bid] = v_absmax_new;
    }
"""


# ---------------------------------------------------------------------------
# Lion8bit fused dynamic kernel source.
# ---------------------------------------------------------------------------
#
# Lion only carries a single momentum buffer m. The update is sign-based:
#   c     = b1 * m + (1 - b1) * g     (feeds sign update; uses *current* m)
#   m_new = b2 * m + (1 - b2) * g     (persistent momentum)
#   param = param * (1 - lr * wd) - lr * sign(c)
# Compared to Adam: no v buffer, no sqrt denominator, no noise-floor needed.

_FUSED_LION_DYNAMIC_SOURCE = """
    threadgroup float lut_tg[LUT_SIZE];
    threadgroup float m_scratch[FUSED_BLOCK];

    uint tid = thread_position_in_threadgroup.x;
    uint bid = threadgroup_position_in_grid.x;
    uint total = param_shape[0];
    uint elem = bid * FUSED_BLOCK + tid;
    bool active = elem < total;

    // ----- Stage 0: cooperatively load the 256-entry LUT into threadgroup mem -----
    lut_tg[tid] = lut[tid];
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    // ----- Stage A: load + dequant momentum into fp32 registers -----
    float m_absmax_prev = m_absmax[bid];

    float param_fp = 0.0f;
    float grad_fp = 0.0f;
    float m_prev = 0.0f;
    if (active) {
        param_fp = (float)param[elem];
        grad_fp = (float)grad[elem];
        uint m_idx = (uint)m_quant_in[elem];
        m_prev = lut_tg[m_idx] * m_absmax_prev;
    }

    // ----- Stage B: Lion math in fp32 -----
    // c uses the *current* m (before the b2 update lands); matches Chen et al.
    // and bnb's Lion8bit ordering exactly.
    float c = beta1 * m_prev + (1.0f - beta1) * grad_fp;
    float m_new = beta2 * m_prev + (1.0f - beta2) * grad_fp;

    // Stash m_new for tree-reduction over |m_new|.
    m_scratch[tid] = active ? m_new : 0.0f;
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    // ----- Stage C: tree reduction for |m| absmax over the block -----
    {
        float ma = metal::abs(m_scratch[tid]);
        m_scratch[tid] = ma;
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    for (uint stride = FUSED_BLOCK / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            float other_m = m_scratch[tid + stride];
            if (other_m > m_scratch[tid]) m_scratch[tid] = other_m;
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }
    float m_absmax_new = m_scratch[0];

    // ----- Stage D: Lion sign-update + decoupled weight decay -----
    // sign(c) is +1 for c>0, -1 for c<0, 0 for c==0 (matches mx.sign).
    float sign_c = 0.0f;
    if (c > 0.0f) sign_c = 1.0f;
    else if (c < 0.0f) sign_c = -1.0f;

    float param_decayed = param_fp;
    if (wd > 0.0f) {
        param_decayed = param_fp * (1.0f - lr * wd);
    }
    float param_new = param_decayed - lr * sign_c;

    // ----- Stage E: write outputs (param, quantized momentum, absmax) -----
    if (active) {
        param_out[elem] = (T)param_new;

        // Dynamic-LUT quant of m_new via binary search in threadgroup memory.
        float m_norm = (m_absmax_new > 0.0f) ? (m_new / m_absmax_new) : 0.0f;
        if (m_norm > 1.0f) m_norm = 1.0f;
        if (m_norm < -1.0f) m_norm = -1.0f;
        uint m_lo = 0u;
        uint m_hi = LUT_SIZE - 1u;
        while (m_lo < m_hi) {
            uint mid = (m_lo + m_hi) >> 1;
            if (lut_tg[mid] < m_norm) {
                m_lo = mid + 1u;
            } else {
                m_hi = mid;
            }
        }
        uint m_best = m_lo;
        if (m_lo > 0u) {
            float d_hi = metal::abs(lut_tg[m_lo] - m_norm);
            float d_lo = metal::abs(lut_tg[m_lo - 1u] - m_norm);
            if (d_lo < d_hi) m_best = m_lo - 1u;
        }
        m_quant_out[elem] = (uint8_t)m_best;
    }

    if (tid == 0) {
        m_absmax_out[bid] = m_absmax_new;
    }
"""


_fused_adam_dynamic_kernel: Optional[object] = None
_fused_lion_dynamic_kernel: Optional[object] = None


def _can_run_metal() -> bool:
    metal = getattr(mx, "metal", None)
    return mx.default_device() == mx.gpu and metal is not None and metal.is_available()


def _get_fused_adam_dynamic_kernel() -> object:
    """Lazily JIT-compile the fused Adam8bit dynamic-LUT MSL kernel."""

    global _fused_adam_dynamic_kernel
    if _fused_adam_dynamic_kernel is None:
        if not _can_run_metal():
            raise RuntimeError(
                "Fused Adam8bit dynamic-LUT kernel requires the MLX Metal backend; "
                "default device is not GPU or mx.metal is unavailable."
            )
        _fused_adam_dynamic_kernel = mx.fast.metal_kernel(
            name="cppmega_fused_adam8bit_dynamic",
            input_names=[
                "param",
                "grad",
                "m_quant_in",
                "m_absmax",
                "v_quant_in",
                "v_absmax",
                "lut",
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
            header=_FUSED_DYNAMIC_HEADER,
            source=_FUSED_ADAM_DYNAMIC_SOURCE,
            ensure_row_contiguous=True,
        )
    return _fused_adam_dynamic_kernel


def _get_fused_lion_dynamic_kernel() -> object:
    """Lazily JIT-compile the fused Lion8bit dynamic-LUT MSL kernel."""

    global _fused_lion_dynamic_kernel
    if _fused_lion_dynamic_kernel is None:
        if not _can_run_metal():
            raise RuntimeError(
                "Fused Lion8bit dynamic-LUT kernel requires the MLX Metal backend; "
                "default device is not GPU or mx.metal is unavailable."
            )
        _fused_lion_dynamic_kernel = mx.fast.metal_kernel(
            name="cppmega_fused_lion8bit_dynamic",
            input_names=[
                "param",
                "grad",
                "m_quant_in",
                "m_absmax",
                "lut",
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
            header=_FUSED_DYNAMIC_HEADER,
            source=_FUSED_LION_DYNAMIC_SOURCE,
            ensure_row_contiguous=True,
        )
    return _fused_lion_dynamic_kernel


def fused_adam8bit_dynamic_step(
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
    """Run the fused dequant -> AdamW -> quant -> apply kernel for one parameter
    using the dynamic LUT codec.

    Inputs:

    * ``param``: parameter tensor (bf16 or fp32). Treated flat internally.
    * ``grad``: gradient tensor with same shape and dtype as ``param``.
    * ``m_quant``, ``v_quant``: uint8 LUT-index quantized moments,
      same shape as ``param``.
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
            f"block_size={block_size} not supported by the fused dynamic kernel; "
            f"only block_size={FUSED_BLOCK_SIZE} is wired through."
        )
    if param.dtype not in {mx.bfloat16, mx.float32, mx.float16}:
        raise TypeError(
            f"fused Adam8bit param must be float (bf16/fp32/fp16); got {param.dtype}"
        )
    if grad.dtype != param.dtype:
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
        return (param, m_quant, m_absmax, v_quant, v_absmax)

    nblocks = (n + block_size - 1) // block_size

    # 0-D fp32 scalars; the kernel signature reads them as bare scalars.
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

    lut = _get_lut()  # 256 fp32 entries; module-scoped cached.

    kernel = _get_fused_adam_dynamic_kernel()
    outputs = kernel(
        inputs=[
            flat_param,
            flat_grad,
            flat_mq,
            m_absmax,
            flat_vq,
            v_absmax,
            lut,
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


def fused_lion8bit_dynamic_step(
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
    """Run the fused dequant -> Lion -> quant -> apply kernel for one parameter
    using the dynamic LUT codec.

    Inputs:

    * ``param``: parameter tensor (bf16 or fp32). Treated flat internally.
    * ``grad``: gradient tensor with same shape and dtype as ``param``.
    * ``m_quant``: uint8 LUT-index quantized momentum, same shape as ``param``.
    * ``m_absmax``: fp32 per-256-block absmax scales.
    * ``learning_rate``: 0-D mx.array scalar.
    * ``beta1``, ``beta2``, ``weight_decay``: Python scalars that become fp32
      0-D inputs to the kernel.

    Returns ``(param_out, m_quant_out, m_absmax_out)`` with the same shapes/
    dtypes as the corresponding inputs.

    Tail block: if ``param.size`` is not a multiple of ``block_size`` the
    final block zero-pads the unused threads, exactly matching the unfused
    codec's per-block layout.
    """

    if block_size != FUSED_BLOCK_SIZE:
        raise NotImplementedError(
            f"block_size={block_size} not supported by the fused dynamic kernel; "
            f"only block_size={FUSED_BLOCK_SIZE} is wired through."
        )
    if param.dtype not in {mx.bfloat16, mx.float32, mx.float16}:
        raise TypeError(
            f"fused Lion8bit param must be float (bf16/fp32/fp16); got {param.dtype}"
        )
    if grad.dtype != param.dtype:
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
        return (param, m_quant, m_absmax)

    nblocks = (n + block_size - 1) // block_size

    lr_scalar = learning_rate.astype(mx.float32) if learning_rate.ndim == 0 else learning_rate.astype(mx.float32).reshape(())
    if lr_scalar.ndim != 0:
        lr_scalar = lr_scalar.reshape(())
    beta1_arr = mx.array(float(beta1), dtype=mx.float32)
    beta2_arr = mx.array(float(beta2), dtype=mx.float32)
    wd_arr = mx.array(float(weight_decay), dtype=mx.float32)

    lut = _get_lut()

    kernel = _get_fused_lion_dynamic_kernel()
    outputs = kernel(
        inputs=[
            flat_param,
            flat_grad,
            flat_mq,
            m_absmax,
            lut,
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
    (param_flat_out, m_quant_flat_out, m_absmax_out) = outputs

    return (
        param_flat_out.reshape(original_shape),
        m_quant_flat_out.reshape(original_shape),
        m_absmax_out,
    )


__all__ = [
    "FUSED_BLOCK_SIZE",
    "fused_adam8bit_dynamic_step",
    "fused_lion8bit_dynamic_step",
]

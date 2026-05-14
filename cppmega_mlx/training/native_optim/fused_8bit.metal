#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

constant constexpr uint FUSED_BLOCK = 256;
constant constexpr uint LUT_SIZE = 256;

template <typename T>
[[kernel]] void cppmega_fused_adam8bit_symmetric(
    device const T* param [[buffer(0)]],
    device const T* grad [[buffer(1)]],
    device const uint8_t* m_quant_in [[buffer(2)]],
    device const float* m_absmax [[buffer(3)]],
    device const uint8_t* v_quant_in [[buffer(4)]],
    device const float* v_absmax [[buffer(5)]],
    device const float* lr_ptr [[buffer(6)]],
    device const float* step_ptr [[buffer(7)]],
    device T* param_out [[buffer(8)]],
    device uint8_t* m_quant_out [[buffer(9)]],
    device float* m_absmax_out [[buffer(10)]],
    device uint8_t* v_quant_out [[buffer(11)]],
    device float* v_absmax_out [[buffer(12)]],
    constant const uint& total [[buffer(13)]],
    constant const float& beta1 [[buffer(14)]],
    constant const float& beta2 [[buffer(15)]],
    constant const float& eps [[buffer(16)]],
    constant const float& wd [[buffer(17)]],
    constant const float& bias_correction [[buffer(18)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]]) {
  threadgroup float m_scratch[FUSED_BLOCK];
  threadgroup float v_scratch[FUSED_BLOCK];

  uint elem = bid * FUSED_BLOCK + tid;
  bool active = elem < total;
  float lr = lr_ptr[0];
  float step_fp = step_ptr[0];

  float m_absmax_prev = m_absmax[bid];
  float v_absmax_prev = v_absmax[bid];
  float param_fp = 0.0f;
  float grad_fp = 0.0f;
  float m_prev = 0.0f;
  float v_prev = 0.0f;
  if (active) {
    param_fp = static_cast<float>(param[elem]);
    grad_fp = static_cast<float>(grad[elem]);
    int m_signed = static_cast<int>(m_quant_in[elem]) - 128;
    int v_signed = static_cast<int>(v_quant_in[elem]) - 128;
    m_prev = static_cast<float>(m_signed) * (1.0f / 127.0f) * m_absmax_prev;
    v_prev = static_cast<float>(v_signed) * (1.0f / 127.0f) * v_absmax_prev;
    if (v_prev < 0.0f) {
      v_prev = 0.0f;
    }
  }

  float m_new = beta1 * m_prev + (1.0f - beta1) * grad_fp;
  float v_new = beta2 * v_prev + (1.0f - beta2) * grad_fp * grad_fp;
  if (v_new < 0.0f) {
    v_new = 0.0f;
  }
  m_scratch[tid] = active ? m_new : 0.0f;
  v_scratch[tid] = active ? v_new : 0.0f;
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  m_scratch[tid] = metal::abs(m_scratch[tid]);
  v_scratch[tid] = metal::abs(v_scratch[tid]);
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  for (uint stride = FUSED_BLOCK / 2u; stride > 0u; stride >>= 1) {
    if (tid < stride) {
      m_scratch[tid] = metal::max(m_scratch[tid], m_scratch[tid + stride]);
      v_scratch[tid] = metal::max(v_scratch[tid], v_scratch[tid + stride]);
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  }
  float m_absmax_new = m_scratch[0];
  float v_absmax_new = v_scratch[0];
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
  float param_new = param_fp * (1.0f - lr * wd) - numerator / denominator;

  if (active) {
    param_out[elem] = static_cast<T>(param_new);
    float m_norm = (m_absmax_new > 0.0f) ? (m_new / m_absmax_new) : 0.0f;
    int m_rounded = static_cast<int>(metal::round(m_norm * 127.0f));
    m_rounded = metal::clamp(m_rounded, -127, 127);
    m_quant_out[elem] = static_cast<uint8_t>(m_rounded + 128);

    float v_norm = (v_absmax_new > 0.0f) ? (v_new / v_absmax_new) : 0.0f;
    int v_rounded = static_cast<int>(metal::round(v_norm * 127.0f));
    v_rounded = metal::clamp(v_rounded, -127, 127);
    v_quant_out[elem] = static_cast<uint8_t>(v_rounded + 128);
  }
  if (tid == 0) {
    m_absmax_out[bid] = m_absmax_new;
    v_absmax_out[bid] = v_absmax_new;
  }
}

template <typename T>
[[kernel]] void cppmega_fused_adam8bit_dynamic(
    device const T* param [[buffer(0)]],
    device const T* grad [[buffer(1)]],
    device const uint8_t* m_quant_in [[buffer(2)]],
    device const float* m_absmax [[buffer(3)]],
    device const uint8_t* v_quant_in [[buffer(4)]],
    device const float* v_absmax [[buffer(5)]],
    device const float* lr_ptr [[buffer(6)]],
    device const float* step_ptr [[buffer(7)]],
    device const float* lut [[buffer(8)]],
    device T* param_out [[buffer(9)]],
    device uint8_t* m_quant_out [[buffer(10)]],
    device float* m_absmax_out [[buffer(11)]],
    device uint8_t* v_quant_out [[buffer(12)]],
    device float* v_absmax_out [[buffer(13)]],
    constant const uint& total [[buffer(14)]],
    constant const float& beta1 [[buffer(15)]],
    constant const float& beta2 [[buffer(16)]],
    constant const float& eps [[buffer(17)]],
    constant const float& wd [[buffer(18)]],
    constant const float& bias_correction [[buffer(19)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]]) {
  threadgroup float lut_tg[LUT_SIZE];
  threadgroup float m_scratch[FUSED_BLOCK];
  threadgroup float v_scratch[FUSED_BLOCK];

  lut_tg[tid] = lut[tid];
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  uint elem = bid * FUSED_BLOCK + tid;
  bool active = elem < total;
  float lr = lr_ptr[0];
  float step_fp = step_ptr[0];

  float m_absmax_prev = m_absmax[bid];
  float v_absmax_prev = v_absmax[bid];
  float param_fp = 0.0f;
  float grad_fp = 0.0f;
  float m_prev = 0.0f;
  float v_prev = 0.0f;
  if (active) {
    param_fp = static_cast<float>(param[elem]);
    grad_fp = static_cast<float>(grad[elem]);
    m_prev = lut_tg[static_cast<uint>(m_quant_in[elem])] * m_absmax_prev;
    v_prev = lut_tg[static_cast<uint>(v_quant_in[elem])] * v_absmax_prev;
    if (v_prev < 0.0f) {
      v_prev = 0.0f;
    }
  }

  float m_new = beta1 * m_prev + (1.0f - beta1) * grad_fp;
  float v_new = beta2 * v_prev + (1.0f - beta2) * grad_fp * grad_fp;
  if (v_new < 0.0f) {
    v_new = 0.0f;
  }
  m_scratch[tid] = active ? m_new : 0.0f;
  v_scratch[tid] = active ? v_new : 0.0f;
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  m_scratch[tid] = metal::abs(m_scratch[tid]);
  v_scratch[tid] = metal::abs(v_scratch[tid]);
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  for (uint stride = FUSED_BLOCK / 2u; stride > 0u; stride >>= 1) {
    if (tid < stride) {
      m_scratch[tid] = metal::max(m_scratch[tid], m_scratch[tid + stride]);
      v_scratch[tid] = metal::max(v_scratch[tid], v_scratch[tid + stride]);
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  }
  float m_absmax_new = m_scratch[0];
  float v_absmax_new = v_scratch[0];
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
  float param_new = param_fp * (1.0f - lr * wd) - numerator / denominator;

  if (active) {
    param_out[elem] = static_cast<T>(param_new);
    float m_norm = (m_absmax_new > 0.0f) ? (m_new / m_absmax_new) : 0.0f;
    m_norm = metal::clamp(m_norm, -1.0f, 1.0f);
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
      if (d_lo < d_hi) {
        m_best = m_lo - 1u;
      }
    }
    m_quant_out[elem] = static_cast<uint8_t>(m_best);

    float v_norm = (v_absmax_new > 0.0f) ? (v_new / v_absmax_new) : 0.0f;
    v_norm = metal::clamp(v_norm, -1.0f, 1.0f);
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
      if (d_lo < d_hi) {
        v_best = v_lo - 1u;
      }
    }
    v_quant_out[elem] = static_cast<uint8_t>(v_best);
  }
  if (tid == 0) {
    m_absmax_out[bid] = m_absmax_new;
    v_absmax_out[bid] = v_absmax_new;
  }
}

template <typename T>
[[kernel]] void cppmega_fused_lion8bit_symmetric(
    device const T* param [[buffer(0)]],
    device const T* grad [[buffer(1)]],
    device const uint8_t* m_quant_in [[buffer(2)]],
    device const float* m_absmax [[buffer(3)]],
    device const float* lr_ptr [[buffer(4)]],
    device T* param_out [[buffer(5)]],
    device uint8_t* m_quant_out [[buffer(6)]],
    device float* m_absmax_out [[buffer(7)]],
    constant const uint& total [[buffer(8)]],
    constant const float& beta1 [[buffer(9)]],
    constant const float& beta2 [[buffer(10)]],
    constant const float& wd [[buffer(11)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]]) {
  threadgroup float m_scratch[FUSED_BLOCK];
  uint elem = bid * FUSED_BLOCK + tid;
  bool active = elem < total;
  float lr = lr_ptr[0];
  float m_absmax_prev = m_absmax[bid];
  float param_fp = 0.0f;
  float grad_fp = 0.0f;
  float m_prev = 0.0f;
  float m_new = 0.0f;
  float c = 0.0f;
  if (active) {
    param_fp = static_cast<float>(param[elem]);
    grad_fp = static_cast<float>(grad[elem]);
    int m_signed = static_cast<int>(m_quant_in[elem]) - 128;
    m_prev = static_cast<float>(m_signed) * (1.0f / 127.0f) * m_absmax_prev;
    c = beta1 * m_prev + (1.0f - beta1) * grad_fp;
    m_new = beta2 * m_prev + (1.0f - beta2) * grad_fp;
    float sign_c = c > 0.0f ? 1.0f : (c < 0.0f ? -1.0f : 0.0f);
    param_out[elem] = static_cast<T>(param_fp * (1.0f - lr * wd) - lr * sign_c);
  }
  m_scratch[tid] = active ? metal::abs(m_new) : 0.0f;
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  for (uint stride = FUSED_BLOCK / 2u; stride > 0u; stride >>= 1) {
    if (tid < stride) {
      m_scratch[tid] = metal::max(m_scratch[tid], m_scratch[tid + stride]);
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  }
  float m_absmax_new = m_scratch[0];
  if (active) {
    float m_norm = (m_absmax_new > 0.0f) ? (m_new / m_absmax_new) : 0.0f;
    int m_rounded = static_cast<int>(metal::round(m_norm * 127.0f));
    m_rounded = metal::clamp(m_rounded, -127, 127);
    m_quant_out[elem] = static_cast<uint8_t>(m_rounded + 128);
  }
  if (tid == 0) {
    m_absmax_out[bid] = m_absmax_new;
  }
}

template <typename T>
[[kernel]] void cppmega_fused_lion8bit_dynamic(
    device const T* param [[buffer(0)]],
    device const T* grad [[buffer(1)]],
    device const uint8_t* m_quant_in [[buffer(2)]],
    device const float* m_absmax [[buffer(3)]],
    device const float* lr_ptr [[buffer(4)]],
    device const float* lut [[buffer(5)]],
    device T* param_out [[buffer(6)]],
    device uint8_t* m_quant_out [[buffer(7)]],
    device float* m_absmax_out [[buffer(8)]],
    constant const uint& total [[buffer(9)]],
    constant const float& beta1 [[buffer(10)]],
    constant const float& beta2 [[buffer(11)]],
    constant const float& wd [[buffer(12)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]]) {
  threadgroup float lut_tg[LUT_SIZE];
  threadgroup float m_scratch[FUSED_BLOCK];
  lut_tg[tid] = lut[tid];
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  uint elem = bid * FUSED_BLOCK + tid;
  bool active = elem < total;
  float lr = lr_ptr[0];
  float m_absmax_prev = m_absmax[bid];
  float param_fp = 0.0f;
  float grad_fp = 0.0f;
  float m_prev = 0.0f;
  float m_new = 0.0f;
  float c = 0.0f;
  if (active) {
    param_fp = static_cast<float>(param[elem]);
    grad_fp = static_cast<float>(grad[elem]);
    m_prev = lut_tg[static_cast<uint>(m_quant_in[elem])] * m_absmax_prev;
    c = beta1 * m_prev + (1.0f - beta1) * grad_fp;
    m_new = beta2 * m_prev + (1.0f - beta2) * grad_fp;
    float sign_c = c > 0.0f ? 1.0f : (c < 0.0f ? -1.0f : 0.0f);
    param_out[elem] = static_cast<T>(param_fp * (1.0f - lr * wd) - lr * sign_c);
  }
  m_scratch[tid] = active ? metal::abs(m_new) : 0.0f;
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  for (uint stride = FUSED_BLOCK / 2u; stride > 0u; stride >>= 1) {
    if (tid < stride) {
      m_scratch[tid] = metal::max(m_scratch[tid], m_scratch[tid + stride]);
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  }
  float m_absmax_new = m_scratch[0];
  if (active) {
    float m_norm = (m_absmax_new > 0.0f) ? (m_new / m_absmax_new) : 0.0f;
    m_norm = metal::clamp(m_norm, -1.0f, 1.0f);
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
      if (d_lo < d_hi) {
        m_best = m_lo - 1u;
      }
    }
    m_quant_out[elem] = static_cast<uint8_t>(m_best);
  }
  if (tid == 0) {
    m_absmax_out[bid] = m_absmax_new;
  }
}

#define instantiate_cppmega_adam(name, type)                                  \
  instantiate_kernel(                                                         \
      "cppmega_fused_adam8bit_symmetric_" #name,                              \
      cppmega_fused_adam8bit_symmetric,                                       \
      type)                                                                   \
  instantiate_kernel(                                                         \
      "cppmega_fused_adam8bit_dynamic_" #name,                                \
      cppmega_fused_adam8bit_dynamic,                                         \
      type)

#define instantiate_cppmega_lion(name, type)                                  \
  instantiate_kernel(                                                         \
      "cppmega_fused_lion8bit_symmetric_" #name,                              \
      cppmega_fused_lion8bit_symmetric,                                       \
      type)                                                                   \
  instantiate_kernel(                                                         \
      "cppmega_fused_lion8bit_dynamic_" #name,                                \
      cppmega_fused_lion8bit_dynamic,                                         \
      type)

instantiate_cppmega_adam(float32, float);
instantiate_cppmega_adam(float16, half);
instantiate_cppmega_adam(bfloat16, bfloat16_t);
instantiate_cppmega_lion(float32, float);
instantiate_cppmega_lion(float16, half);
instantiate_cppmega_lion(bfloat16, bfloat16_t);

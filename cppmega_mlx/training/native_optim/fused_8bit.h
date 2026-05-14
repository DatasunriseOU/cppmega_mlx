#pragma once

#include <string>
#include <vector>

#include "mlx/array.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

namespace cppmega_native_optim {

constexpr int kFusedBlockSize = 256;

std::string status_reason();
bool available();

std::vector<mx::array> fused_adam8bit_step(
    const mx::array& param,
    const mx::array& grad,
    const mx::array& m_quant,
    const mx::array& m_absmax,
    const mx::array& v_quant,
    const mx::array& v_absmax,
    const mx::array& learning_rate,
    const mx::array& step,
    const mx::array& lut,
    bool dynamic_lut,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    bool bias_correction,
    mx::StreamOrDevice s = {});

std::vector<mx::array> fused_lion8bit_step(
    const mx::array& param,
    const mx::array& grad,
    const mx::array& m_quant,
    const mx::array& m_absmax,
    const mx::array& learning_rate,
    const mx::array& lut,
    bool dynamic_lut,
    float beta1,
    float beta2,
    float weight_decay,
    mx::StreamOrDevice s = {});

} // namespace cppmega_native_optim

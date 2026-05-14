#include "fused_8bit.h"

#include <dlfcn.h>

#include <filesystem>
#include <stdexcept>

#include "mlx/backend/cpu/encoder.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace cppmega_native_optim {
namespace {

std::string current_binary_dir() {
  static std::string binary_dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&current_binary_dir), &info)) {
      throw std::runtime_error("Unable to resolve native optimizer binary dir");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return binary_dir;
}

std::string dtype_suffix(const mx::array& a) {
  if (a.dtype() == mx::bfloat16) {
    return "bfloat16";
  }
  if (a.dtype() == mx::float16) {
    return "float16";
  }
  if (a.dtype() == mx::float32) {
    return "float32";
  }
  throw std::runtime_error("fused 8-bit optimizer supports bf16/fp16/fp32 params");
}

void validate_common(
    const mx::array& param,
    const mx::array& grad,
    const mx::array& m_quant,
    const mx::array& m_absmax) {
  if (param.shape() != grad.shape()) {
    throw std::invalid_argument("param.shape must equal grad.shape");
  }
  if (m_quant.shape() != param.shape()) {
    throw std::invalid_argument("m_quant must share param.shape");
  }
  if (m_quant.dtype() != mx::uint8) {
    throw std::invalid_argument("m_quant must be uint8");
  }
  if (m_absmax.dtype() != mx::float32) {
    throw std::invalid_argument("m_absmax must be float32");
  }
  auto expected_blocks =
      (static_cast<int64_t>(param.size()) + kFusedBlockSize - 1) /
      kFusedBlockSize;
  if (m_absmax.size() != expected_blocks) {
    throw std::invalid_argument("m_absmax block count does not match param size");
  }
  (void)dtype_suffix(param);
}

mx::array scalar_f32(const mx::array& x, mx::StreamOrDevice s) {
  auto out = mx::astype(x, mx::float32, s);
  if (out.ndim() != 0) {
    out = mx::reshape(out, {}, s);
  }
  return out;
}

class FusedAdam8bit : public mx::Primitive {
 public:
  FusedAdam8bit(
      mx::Stream stream,
      bool dynamic_lut,
      float beta1,
      float beta2,
      float eps,
      float weight_decay,
      bool bias_correction)
      : mx::Primitive(stream),
        dynamic_lut_(dynamic_lut),
        beta1_(beta1),
        beta2_(beta2),
        eps_(eps),
        weight_decay_(weight_decay),
        bias_correction_(bias_correction ? 1.0f : 0.0f) {}

  void eval_cpu(
      const std::vector<mx::array>&,
      std::vector<mx::array>&) override {
    throw std::runtime_error("FusedAdam8bit has no CPU implementation");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "CppmegaFusedAdam8bit";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& o = static_cast<const FusedAdam8bit&>(other);
    return dynamic_lut_ == o.dynamic_lut_ && beta1_ == o.beta1_ &&
        beta2_ == o.beta2_ && eps_ == o.eps_ &&
        weight_decay_ == o.weight_decay_ &&
        bias_correction_ == o.bias_correction_;
  }

 private:
  bool dynamic_lut_;
  float beta1_;
  float beta2_;
  float eps_;
  float weight_decay_;
  float bias_correction_;
};

class FusedLion8bit : public mx::Primitive {
 public:
  FusedLion8bit(
      mx::Stream stream,
      bool dynamic_lut,
      float beta1,
      float beta2,
      float weight_decay)
      : mx::Primitive(stream),
        dynamic_lut_(dynamic_lut),
        beta1_(beta1),
        beta2_(beta2),
        weight_decay_(weight_decay) {}

  void eval_cpu(
      const std::vector<mx::array>&,
      std::vector<mx::array>&) override {
    throw std::runtime_error("FusedLion8bit has no CPU implementation");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "CppmegaFusedLion8bit";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& o = static_cast<const FusedLion8bit&>(other);
    return dynamic_lut_ == o.dynamic_lut_ && beta1_ == o.beta1_ &&
        beta2_ == o.beta2_ && weight_decay_ == o.weight_decay_;
  }

 private:
  bool dynamic_lut_;
  float beta1_;
  float beta2_;
  float weight_decay_;
};

#ifdef _METAL_
void allocate_outputs(std::vector<mx::array>& outputs) {
  for (auto& out : outputs) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
  }
}

void FusedAdam8bit::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& param = inputs[0];
  auto& grad = inputs[1];
  auto& m_quant = inputs[2];
  auto& m_absmax = inputs[3];
  auto& v_quant = inputs[4];
  auto& v_absmax = inputs[5];
  auto& lr = inputs[6];
  auto& step = inputs[7];
  const mx::array* lut = dynamic_lut_ ? &inputs[8] : nullptr;

  allocate_outputs(outputs);
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto lib = d.get_library("cppmega_native_optim", current_binary_dir());
  std::string kname = dynamic_lut_ ? "cppmega_fused_adam8bit_dynamic_"
                                   : "cppmega_fused_adam8bit_symmetric_";
  kname += dtype_suffix(param);
  auto kernel = d.get_kernel(kname, lib);

  auto& enc = mx::metal::get_command_encoder(s);
  enc.set_compute_pipeline_state(kernel);

  int arg = 0;
  enc.set_input_array(param, arg++);
  enc.set_input_array(grad, arg++);
  enc.set_input_array(m_quant, arg++);
  enc.set_input_array(m_absmax, arg++);
  enc.set_input_array(v_quant, arg++);
  enc.set_input_array(v_absmax, arg++);
  enc.set_input_array(lr, arg++);
  enc.set_input_array(step, arg++);
  if (dynamic_lut_) {
    enc.set_input_array(*lut, arg++);
  }
  enc.set_output_array(outputs[0], arg++);
  enc.set_output_array(outputs[1], arg++);
  enc.set_output_array(outputs[2], arg++);
  enc.set_output_array(outputs[3], arg++);
  enc.set_output_array(outputs[4], arg++);
  auto total = static_cast<uint32_t>(param.size());
  enc.set_bytes(total, arg++);
  enc.set_bytes(beta1_, arg++);
  enc.set_bytes(beta2_, arg++);
  enc.set_bytes(eps_, arg++);
  enc.set_bytes(weight_decay_, arg++);
  enc.set_bytes(bias_correction_, arg++);

  size_t nblocks = outputs[2].size();
  enc.dispatch_threads(
      MTL::Size(nblocks * kFusedBlockSize, 1, 1),
      MTL::Size(kFusedBlockSize, 1, 1));
}

void FusedLion8bit::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& param = inputs[0];
  auto& grad = inputs[1];
  auto& m_quant = inputs[2];
  auto& m_absmax = inputs[3];
  auto& lr = inputs[4];
  const mx::array* lut = dynamic_lut_ ? &inputs[5] : nullptr;

  allocate_outputs(outputs);
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto lib = d.get_library("cppmega_native_optim", current_binary_dir());
  std::string kname = dynamic_lut_ ? "cppmega_fused_lion8bit_dynamic_"
                                   : "cppmega_fused_lion8bit_symmetric_";
  kname += dtype_suffix(param);
  auto kernel = d.get_kernel(kname, lib);

  auto& enc = mx::metal::get_command_encoder(s);
  enc.set_compute_pipeline_state(kernel);

  int arg = 0;
  enc.set_input_array(param, arg++);
  enc.set_input_array(grad, arg++);
  enc.set_input_array(m_quant, arg++);
  enc.set_input_array(m_absmax, arg++);
  enc.set_input_array(lr, arg++);
  if (dynamic_lut_) {
    enc.set_input_array(*lut, arg++);
  }
  enc.set_output_array(outputs[0], arg++);
  enc.set_output_array(outputs[1], arg++);
  enc.set_output_array(outputs[2], arg++);
  auto total = static_cast<uint32_t>(param.size());
  enc.set_bytes(total, arg++);
  enc.set_bytes(beta1_, arg++);
  enc.set_bytes(beta2_, arg++);
  enc.set_bytes(weight_decay_, arg++);

  size_t nblocks = outputs[2].size();
  enc.dispatch_threads(
      MTL::Size(nblocks * kFusedBlockSize, 1, 1),
      MTL::Size(kFusedBlockSize, 1, 1));
}
#else
void FusedAdam8bit::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("FusedAdam8bit requires MLX Metal");
}

void FusedLion8bit::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("FusedLion8bit requires MLX Metal");
}
#endif

} // namespace

bool available() {
#ifdef _METAL_
  return true;
#else
  return false;
#endif
}

std::string status_reason() {
#ifdef _METAL_
  return "native MLX C++/Metal fused 8-bit optimizer extension available";
#else
  return "native optimizer extension was built without MLX Metal";
#endif
}

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
    mx::StreamOrDevice s) {
  validate_common(param, grad, m_quant, m_absmax);
  if (v_quant.shape() != param.shape() || v_quant.dtype() != mx::uint8) {
    throw std::invalid_argument("v_quant must be uint8 and share param.shape");
  }
  if (v_absmax.dtype() != mx::float32 || v_absmax.shape() != m_absmax.shape()) {
    throw std::invalid_argument("v_absmax must be float32 and share m_absmax.shape");
  }
  if (param.size() == 0) {
    return {param, m_quant, m_absmax, v_quant, v_absmax};
  }

  auto stream = mx::to_stream(s);
  auto p = mx::contiguous(param, false, stream);
  auto g = mx::contiguous(
      grad.dtype() == param.dtype() ? grad : mx::astype(grad, param.dtype(), stream),
      false,
      stream);
  auto mq = mx::contiguous(m_quant, false, stream);
  auto ma = mx::contiguous(m_absmax, false, stream);
  auto vq = mx::contiguous(v_quant, false, stream);
  auto va = mx::contiguous(v_absmax, false, stream);
  auto lr = scalar_f32(learning_rate, stream);
  auto st = scalar_f32(step, stream);

  std::vector<mx::array> inputs{p, g, mq, ma, vq, va, lr, st};
  if (dynamic_lut) {
    if (lut.dtype() != mx::float32 || lut.size() != 256) {
      throw std::invalid_argument("dynamic_lut=True requires a 256-entry fp32 LUT");
    }
    inputs.push_back(mx::contiguous(lut, false, stream));
  }

  return mx::array::make_arrays(
      {param.shape(), param.shape(), m_absmax.shape(), param.shape(),
       v_absmax.shape()},
      {param.dtype(), mx::uint8, mx::float32, mx::uint8, mx::float32},
      std::make_shared<FusedAdam8bit>(
          stream, dynamic_lut, beta1, beta2, eps, weight_decay, bias_correction),
      std::move(inputs));
}

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
    mx::StreamOrDevice s) {
  validate_common(param, grad, m_quant, m_absmax);
  if (param.size() == 0) {
    return {param, m_quant, m_absmax};
  }

  auto stream = mx::to_stream(s);
  auto p = mx::contiguous(param, false, stream);
  auto g = mx::contiguous(
      grad.dtype() == param.dtype() ? grad : mx::astype(grad, param.dtype(), stream),
      false,
      stream);
  auto mq = mx::contiguous(m_quant, false, stream);
  auto ma = mx::contiguous(m_absmax, false, stream);
  auto lr = scalar_f32(learning_rate, stream);

  std::vector<mx::array> inputs{p, g, mq, ma, lr};
  if (dynamic_lut) {
    if (lut.dtype() != mx::float32 || lut.size() != 256) {
      throw std::invalid_argument("dynamic_lut=True requires a 256-entry fp32 LUT");
    }
    inputs.push_back(mx::contiguous(lut, false, stream));
  }

  return mx::array::make_arrays(
      {param.shape(), param.shape(), m_absmax.shape()},
      {param.dtype(), mx::uint8, mx::float32},
      std::make_shared<FusedLion8bit>(
          stream, dynamic_lut, beta1, beta2, weight_decay),
      std::move(inputs));
}

} // namespace cppmega_native_optim

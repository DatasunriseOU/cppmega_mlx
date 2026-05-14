#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

#include <new>

#include "fused_8bit.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

mx::array unwrap_mx_array(nb::handle obj, const char* name) {
  auto* storage = nb::inst_ptr<mx::array>(obj);
  if (storage == nullptr) {
    std::string msg = std::string(name) + " must be mlx.core.array";
    throw nb::type_error(msg.c_str());
  }
  return *storage;
}

nb::object wrap_mx_array(mx::array&& array) {
  nb::object array_type = nb::module_::import_("mlx.core").attr("array");
  nb::object py_array = array_type.attr("__new__")(array_type);
  auto* storage = nb::inst_ptr<mx::array>(py_array);
  if (storage == nullptr) {
    throw std::runtime_error("failed to allocate mlx.core.array instance");
  }
  new (storage) mx::array(std::move(array));
  nb::inst_set_state(py_array, true, true);
  return py_array;
}

nb::list wrap_mx_arrays(std::vector<mx::array>&& arrays) {
  nb::list out;
  for (auto& array : arrays) {
    out.append(wrap_mx_array(std::move(array)));
  }
  return out;
}

} // namespace

NB_MODULE(_ext, m) {
  m.doc() = "cppmega native MLX optimizer kernels";
  m.def("status", []() {
    nb::dict out;
    out["available"] = cppmega_native_optim::available();
    out["reason"] = cppmega_native_optim::status_reason();
    out["block_size"] = cppmega_native_optim::kFusedBlockSize;
    return out;
  });
  m.def(
      "fused_adam8bit_step",
      [](
          nb::handle param,
          nb::handle grad,
          nb::handle m_quant,
          nb::handle m_absmax,
          nb::handle v_quant,
          nb::handle v_absmax,
          nb::handle learning_rate,
          nb::handle step,
          nb::handle lut,
          bool dynamic_lut,
          float beta1,
          float beta2,
          float eps,
          float weight_decay,
          bool bias_correction) {
        return wrap_mx_arrays(cppmega_native_optim::fused_adam8bit_step(
            unwrap_mx_array(param, "param"),
            unwrap_mx_array(grad, "grad"),
            unwrap_mx_array(m_quant, "m_quant"),
            unwrap_mx_array(m_absmax, "m_absmax"),
            unwrap_mx_array(v_quant, "v_quant"),
            unwrap_mx_array(v_absmax, "v_absmax"),
            unwrap_mx_array(learning_rate, "learning_rate"),
            unwrap_mx_array(step, "step"),
            unwrap_mx_array(lut, "lut"),
            dynamic_lut,
            beta1,
            beta2,
            eps,
            weight_decay,
            bias_correction));
      },
      "param"_a,
      "grad"_a,
      "m_quant"_a,
      "m_absmax"_a,
      "v_quant"_a,
      "v_absmax"_a,
      "learning_rate"_a,
      "step"_a,
      "lut"_a,
      "dynamic_lut"_a,
      "beta1"_a,
      "beta2"_a,
      "eps"_a,
      "weight_decay"_a,
      "bias_correction"_a);
  m.def(
      "fused_lion8bit_step",
      [](
          nb::handle param,
          nb::handle grad,
          nb::handle m_quant,
          nb::handle m_absmax,
          nb::handle learning_rate,
          nb::handle lut,
          bool dynamic_lut,
          float beta1,
          float beta2,
          float weight_decay) {
        return wrap_mx_arrays(cppmega_native_optim::fused_lion8bit_step(
            unwrap_mx_array(param, "param"),
            unwrap_mx_array(grad, "grad"),
            unwrap_mx_array(m_quant, "m_quant"),
            unwrap_mx_array(m_absmax, "m_absmax"),
            unwrap_mx_array(learning_rate, "learning_rate"),
            unwrap_mx_array(lut, "lut"),
            dynamic_lut,
            beta1,
            beta2,
            weight_decay));
      },
      "param"_a,
      "grad"_a,
      "m_quant"_a,
      "m_absmax"_a,
      "learning_rate"_a,
      "lut"_a,
      "dynamic_lut"_a,
      "beta1"_a,
      "beta2"_a,
      "weight_decay"_a);
}

#include "tilelang_mlx_tvm_ffi_bridge.h"

#if __has_include(<tilelang/contrib/mlx_tvm_ffi_c_api.h>)
#include <tilelang/contrib/mlx_tvm_ffi_c_api.h>
#elif __has_include(<contrib/mlx_tvm_ffi/mlx_tvm_ffi_c_api.h>)
#include <contrib/mlx_tvm_ffi/mlx_tvm_ffi_c_api.h>
#else
#error "TileLang MLX TVM-FFI C API header not found"
#endif

namespace {

int fill_status(CppmegaTileLangMLXTVMFFIBridgeStatus *out, size_t out_size,
                int code, const char *state, const char *reason) {
  if (out == nullptr) {
    return 1;
  }
  if (out_size < sizeof(CppmegaTileLangMLXTVMFFIBridgeStatus)) {
    return 2;
  }
  *out = CppmegaTileLangMLXTVMFFIBridgeStatus{};
  out->version = CPPMEGA_TILELANG_MLX_TVM_FFI_BRIDGE_VERSION;
  out->struct_size = sizeof(CppmegaTileLangMLXTVMFFIBridgeStatus);
  out->code = code;
  out->state = state;
  out->reason = reason;
  return code;
}

} // namespace

extern "C" CPPMEGA_TILELANG_MLX_TVM_FFI_BRIDGE_EXPORT int
cppmega_tilelang_mlx_tvm_ffi_bridge_probe(
    CppmegaTileLangMLXTVMFFIBridgeStatus *out, size_t out_size) {
  if (out == nullptr) {
    return 1;
  }
  if (out_size < sizeof(CppmegaTileLangMLXTVMFFIBridgeStatus)) {
    return 2;
  }

  TileLangMLXTVMFFICAPI api{};
  int rc = tilelang_mlx_tvm_ffi_get_c_api(TILELANG_MLX_TVM_FFI_C_API_VERSION,
                                          TILELANG_MLX_TVM_FFI_C_API_ABI_HASH,
                                          &api, sizeof(api));
  if (rc != kTileLangMLXTVMFFICApiOk) {
    return fill_status(out, out_size, rc, "blocked_tilelang_c_api_mismatch",
                       "TileLang MLX TVM-FFI C API version/hash check failed");
  }

  TileLangMLXTVMFFIStatus native_status{};
  rc = api.status(&native_status, sizeof(native_status));
  if (rc != kTileLangMLXTVMFFICApiOk) {
    return fill_status(out, out_size, rc, "blocked_tilelang_c_api_status",
                       "TileLang MLX TVM-FFI C API status call failed");
  }

  *out = CppmegaTileLangMLXTVMFFIBridgeStatus{};
  out->version = CPPMEGA_TILELANG_MLX_TVM_FFI_BRIDGE_VERSION;
  out->struct_size = sizeof(CppmegaTileLangMLXTVMFFIBridgeStatus);
  out->code = 0;
  out->state = "available";
  out->reason = "cppmega bridge linked to TileLang MLX TVM-FFI C API";
  out->tilelang_abi_hash = api.abi_hash;
  out->tilelang_header_sha256 = api.header_sha256;
  out->mlx_version = api.mlx_version;
  out->mlx_lib_sha256 = api.mlx_lib_sha256;
  out->mlx_python_bridge_sha256 = api.mlx_python_bridge_sha256;
  return 0;
}

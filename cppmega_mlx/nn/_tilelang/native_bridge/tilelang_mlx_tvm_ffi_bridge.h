#ifndef CPPMEGA_MLX_NN_TILELANG_NATIVE_BRIDGE_TILELANG_MLX_TVM_FFI_BRIDGE_H_
#define CPPMEGA_MLX_NN_TILELANG_NATIVE_BRIDGE_TILELANG_MLX_TVM_FFI_BRIDGE_H_

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) || defined(__CYGWIN__)
#ifdef CPPMEGA_TILELANG_MLX_TVM_FFI_BRIDGE_EXPORTS
#define CPPMEGA_TILELANG_MLX_TVM_FFI_BRIDGE_EXPORT __declspec(dllexport)
#else
#define CPPMEGA_TILELANG_MLX_TVM_FFI_BRIDGE_EXPORT __declspec(dllimport)
#endif
#else
#define CPPMEGA_TILELANG_MLX_TVM_FFI_BRIDGE_EXPORT                             \
  __attribute__((visibility("default")))
#endif

#define CPPMEGA_TILELANG_MLX_TVM_FFI_BRIDGE_VERSION 1u

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CppmegaTileLangMLXTVMFFIBridgeStatus {
  uint32_t version;
  uint32_t struct_size;
  int32_t code;
  const char *state;
  const char *reason;
  const char *tilelang_abi_hash;
  const char *tilelang_header_sha256;
  const char *mlx_version;
  const char *mlx_lib_sha256;
  const char *mlx_python_bridge_sha256;
} CppmegaTileLangMLXTVMFFIBridgeStatus;

CPPMEGA_TILELANG_MLX_TVM_FFI_BRIDGE_EXPORT int
cppmega_tilelang_mlx_tvm_ffi_bridge_probe(
    CppmegaTileLangMLXTVMFFIBridgeStatus *out, size_t out_size);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CPPMEGA_MLX_NN_TILELANG_NATIVE_BRIDGE_TILELANG_MLX_TVM_FFI_BRIDGE_H_

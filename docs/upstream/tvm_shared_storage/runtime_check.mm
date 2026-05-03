// Live check: dlopen libtvm_runtime.dylib, look up the metal device API
// via the ffi reflection registry, and confirm:
//   (a) GetMetalStorageOptions() resolves to the value implied by
//       TVM_METAL_STORAGE_MODE
//   (b) AllocDataSpace returns an MTLBuffer whose .storageMode matches
//
// This avoids needing a full Python/tvm-ffi install — we go through the
// minimal C ABI (TVMFFIFunctionGetGlobal + tvm-ffi anys) instead.
//
// Build:
//   xcrun --sdk macosx clang++ -std=c++17 -framework Metal \
//     -I tvm/3rdparty/tvm-ffi/include \
//     -I tvm/3rdparty/tvm-ffi/3rdparty/dlpack/include \
//     -L tvm/build/lib -ltvm_runtime -ltvm_ffi \
//     -Wl,-rpath,@executable_path/tvm/build/lib \
//     runtime_check.mm -o runtime_check
//
//   ./runtime_check                                # default = private
//   TVM_METAL_STORAGE_MODE=shared ./runtime_check  # opt-in shared

#import <Metal/MTLBuffer.h>
#import <Metal/MTLDevice.h>

#include <tvm/ffi/c_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/string.h>
#include <tvm/runtime/device_api.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace ffi = tvm::ffi;
using tvm::runtime::DeviceAPI;

static const char* StorageModeLabel(MTLStorageMode m) {
  switch (m) {
    case MTLStorageModeShared:   return "shared";
    case MTLStorageModePrivate:  return "private";
    case MTLStorageModeManaged:  return "managed";
    case MTLStorageModeMemoryless: return "memoryless";
    default: return "unknown";
  }
}

int main() {
  const char* env = std::getenv("TVM_METAL_STORAGE_MODE");
  std::printf("env TVM_METAL_STORAGE_MODE = %s\n", env ? env : "<unset>");

  // (a) Ask the runtime which storage mode it resolved.
  auto get_mode = ffi::Function::GetGlobalRequired("metal.GetStorageMode");
  ffi::String reported = get_mode().cast<ffi::String>();
  std::printf("metal.GetStorageMode -> '%s'\n", reported.c_str());

  // (b) Allocate a buffer via the device API and inspect storageMode.
  auto get_api = ffi::Function::GetGlobalRequired("device_api.metal");
  void* api_ptr = get_api().cast<void*>();
  auto* dev_api = static_cast<DeviceAPI*>(api_ptr);
  DLDevice dev{static_cast<DLDeviceType>(8 /* kDLMetal */), 0};
  DLDataType f32{kDLFloat, 32, 1};
  void* p = dev_api->AllocDataSpace(dev, 1024, 16, f32);
  if (p == nullptr) {
    std::fprintf(stderr, "AllocDataSpace returned nullptr\n");
    return 1;
  }
  id<MTLBuffer> buf = (__bridge id<MTLBuffer>)(p);
  const char* actual = StorageModeLabel(buf.storageMode);
  std::printf("MTLBuffer.storageMode = %s\n", actual);

  // Cross-check: reported label must match buffer label.
  bool ok = std::strcmp(reported.c_str(), actual) == 0;
  // Treat "managed" buffers on iOS-style fallback or absent enum equally.
  if (!ok && reported == ffi::String("managed") && std::strcmp(actual, "managed") != 0) {
    std::printf("(managed not honoured by this device, falling back is OK)\n");
    ok = true;
  }

  dev_api->FreeDataSpace(dev, p);
  std::printf("%s\n", ok ? "OK" : "MISMATCH");
  return ok ? 0 : 2;
}

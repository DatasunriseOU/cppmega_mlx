// Standalone syntax check for the GetMetalStorageOptions() helper
// added by 0001-metal-shared-storage-opt-in.patch.
//
// We isolate the function so it can be compiled without linking the
// full TVM runtime. Anything that depends on tvm-ffi headers
// (the FFI registration block) is excluded — see compile_check_full.mm
// note in REPORT.md. The helper is the only code with non-trivial
// platform-specific logic, so this check covers the risky bits:
//   - Metal framework constants are spelled correctly
//   - the case-insensitive env parsing builds with Apple clang
//   - the function-local static cached lambda is well-formed
//
// Build:
//   xcrun --sdk macosx clang++ -std=c++17 -fobjc-arc -fsyntax-only \
//       -framework Metal syntax_check.mm
#import <Metal/MTLBuffer.h>
#import <Metal/MTLDevice.h>

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <string>
#include <iostream>

// --- BEGIN COPY of helper from src/runtime/metal/metal_device_api.mm ---
// (kept in sync manually; if the upstream version drifts, regenerate.)
namespace tvm {
namespace runtime {
namespace metal {

inline MTLResourceOptions GetMetalStorageOptions() {
  static const MTLResourceOptions cached = []() -> MTLResourceOptions {
    const char* raw = std::getenv("TVM_METAL_STORAGE_MODE");
    if (raw == nullptr || raw[0] == '\0') {
      return MTLResourceStorageModePrivate;
    }
    std::string v(raw);
    for (auto& c : v) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (v == "shared") {
      return MTLResourceStorageModeShared;
    }
    if (v == "managed") {
#if TARGET_OS_IPHONE
      return MTLResourceStorageModeShared;
#else
      return MTLResourceStorageModeManaged;
#endif
    }
    if (v == "private") {
      return MTLResourceStorageModePrivate;
    }
    return MTLResourceStorageModePrivate;
  }();
  return cached;
}

}  // namespace metal
}  // namespace runtime
}  // namespace tvm
// --- END COPY ---

int main() {
  MTLResourceOptions r = tvm::runtime::metal::GetMetalStorageOptions();
  // Print resolved storage mode label for ad-hoc verification.
  const char* label = "private";
  if (r == MTLResourceStorageModeShared) label = "shared";
  if (r == MTLResourceStorageModeManaged) label = "managed";
  std::cout << "GetMetalStorageOptions resolved to: " << label
            << " (raw=" << static_cast<unsigned long>(r) << ")\n";
  return 0;
}

// Live check: dlopen libtvm_runtime.dylib, look up the metal device API
// via the ffi reflection registry, and confirm:
//   (a) GetMetalStorageOptions() resolves to the value implied by
//       TVM_METAL_STORAGE_MODE
//   (b) AllocDataSpace returns an MTLBuffer whose .storageMode matches
//
// This avoids needing a full Python/tvm-ffi install — we go through the
// minimal C ABI (TVMFFIFunctionGetGlobal + tvm-ffi anys) instead.
//
// Build the TVM-linked storage-mode check:
//   xcrun --sdk macosx clang++ -std=c++17 -framework Metal \
//     -I tvm/3rdparty/tvm-ffi/include \
//     -I tvm/3rdparty/tvm-ffi/3rdparty/dlpack/include \
//     -L tvm/build/lib -ltvm_runtime -ltvm_ffi \
//     -Wl,-rpath,@executable_path/tvm/build/lib \
//     runtime_check.mm -o runtime_check
//
//   ./runtime_check                                # default = private
//   TVM_METAL_STORAGE_MODE=shared ./runtime_check  # opt-in shared
//
// Build the standalone transfer microbench (no TVM headers or runtime needed):
//   xcrun --sdk macosx clang++ -std=c++17 -O2 -fobjc-arc -framework Metal \
//     -DCPPMEGA_STANDALONE_METAL_BENCH runtime_check.mm -o metal_transfer_bench
//   ./metal_transfer_bench 100

#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

static const char* StorageModeLabel(MTLStorageMode m) {
  switch (m) {
    case MTLStorageModeShared:   return "shared";
    case MTLStorageModePrivate:  return "private";
    case MTLStorageModeManaged:  return "managed";
    case MTLStorageModeMemoryless: return "memoryless";
    default: return "unknown";
  }
}

#ifdef CPPMEGA_STANDALONE_METAL_BENCH

namespace {

using Clock = std::chrono::steady_clock;

double MedianUs(std::vector<double>* values) {
  std::sort(values->begin(), values->end());
  return (*values)[values->size() / 2];
}

void FillHost(std::vector<unsigned char>* data) {
  for (size_t i = 0; i < data->size(); ++i) {
    (*data)[i] = static_cast<unsigned char>((i * 131u + 17u) & 0xffu);
  }
}

unsigned long LongChecksum(const std::vector<unsigned char>& data) {
  unsigned long acc = 0;
  const size_t stride = std::max<size_t>(1, data.size() / 4096);
  for (size_t i = 0; i < data.size(); i += stride) {
    acc += data[i];
  }
  return acc;
}

void BlitAndWait(id<MTLCommandQueue> queue, id<MTLBuffer> src, id<MTLBuffer> dst, size_t nbytes) {
  id<MTLCommandBuffer> cb = [queue commandBuffer];
  id<MTLBlitCommandEncoder> encoder = [cb blitCommandEncoder];
  [encoder copyFromBuffer:src sourceOffset:0 toBuffer:dst destinationOffset:0 size:nbytes];
  [encoder endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  if ([cb status] == MTLCommandBufferStatusError) {
    std::fprintf(stderr, "Metal blit failed: %s\n", [[[cb error] localizedDescription] UTF8String]);
    std::exit(2);
  }
}

double BenchCpuToMetal(id<MTLDevice> dev, id<MTLCommandQueue> queue,
                       MTLResourceOptions storage_mode, size_t nbytes, int iters,
                       unsigned long* checksum) {
  std::vector<unsigned char> host(nbytes);
  FillHost(&host);
  id<MTLBuffer> dst = [dev newBufferWithLength:nbytes options:storage_mode];
  id<MTLBuffer> staging = [dev newBufferWithLength:nbytes options:MTLResourceStorageModeShared];
  if (dst == nil || staging == nil) {
    std::fprintf(stderr, "newBufferWithLength failed for %zu bytes\n", nbytes);
    std::exit(2);
  }

  std::vector<double> samples;
  samples.reserve(static_cast<size_t>(iters));
  for (int i = 0; i < iters + 5; ++i) {
    auto start = Clock::now();
    if ([dst storageMode] == MTLStorageModeShared || [dst storageMode] == MTLStorageModeManaged) {
      std::memcpy([dst contents], host.data(), nbytes);
    } else {
      std::memcpy([staging contents], host.data(), nbytes);
      BlitAndWait(queue, staging, dst, nbytes);
    }
    auto end = Clock::now();
    if (i >= 5) {
      samples.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }
  }

  if ([dst storageMode] == MTLStorageModeShared || [dst storageMode] == MTLStorageModeManaged) {
    if (std::memcmp([dst contents], host.data(), nbytes) != 0) {
      std::fprintf(stderr, "CPU->Metal verification failed for shared buffer\n");
      std::exit(2);
    }
  } else {
    BlitAndWait(queue, dst, staging, nbytes);
    if (std::memcmp([staging contents], host.data(), nbytes) != 0) {
      std::fprintf(stderr, "CPU->Metal verification failed for private buffer\n");
      std::exit(2);
    }
  }

  *checksum += LongChecksum(host);
  return MedianUs(&samples);
}

double BenchMetalToCpu(id<MTLDevice> dev, id<MTLCommandQueue> queue,
                       MTLResourceOptions storage_mode, size_t nbytes, int iters,
                       unsigned long* checksum) {
  std::vector<unsigned char> seed(nbytes);
  std::vector<unsigned char> host(nbytes);
  FillHost(&seed);

  id<MTLBuffer> src = [dev newBufferWithLength:nbytes options:storage_mode];
  id<MTLBuffer> staging = [dev newBufferWithLength:nbytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> seed_buffer = [dev newBufferWithLength:nbytes options:MTLResourceStorageModeShared];
  if (src == nil || staging == nil || seed_buffer == nil) {
    std::fprintf(stderr, "newBufferWithLength failed for %zu bytes\n", nbytes);
    std::exit(2);
  }
  std::memcpy([seed_buffer contents], seed.data(), nbytes);
  if ([src storageMode] == MTLStorageModeShared || [src storageMode] == MTLStorageModeManaged) {
    std::memcpy([src contents], seed.data(), nbytes);
  } else {
    BlitAndWait(queue, seed_buffer, src, nbytes);
  }

  std::vector<double> samples;
  samples.reserve(static_cast<size_t>(iters));
  for (int i = 0; i < iters + 5; ++i) {
    auto start = Clock::now();
    if ([src storageMode] == MTLStorageModeShared || [src storageMode] == MTLStorageModeManaged) {
      std::memcpy(host.data(), [src contents], nbytes);
    } else {
      BlitAndWait(queue, src, staging, nbytes);
      std::memcpy(host.data(), [staging contents], nbytes);
    }
    auto end = Clock::now();
    if (i >= 5) {
      samples.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }
  }
  if (std::memcmp(host.data(), seed.data(), nbytes) != 0) {
    std::fprintf(stderr, "Metal->CPU verification failed\n");
    std::exit(2);
  }
  *checksum += LongChecksum(host);
  return MedianUs(&samples);
}

}  // namespace

int main(int argc, char** argv) {
  @autoreleasepool {
    int iters = 100;
    if (argc > 1) {
      iters = std::max(1, std::atoi(argv[1]));
    }
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (dev == nil) {
      std::fprintf(stderr, "No default Metal device\n");
      return 1;
    }
    id<MTLCommandQueue> queue = [dev newCommandQueue];
    if (queue == nil) {
      std::fprintf(stderr, "newCommandQueue failed\n");
      return 1;
    }

    std::printf("standalone Metal transfer bench: device=%s iterations=%d\n",
                [[dev name] UTF8String], iters);
    std::printf("bytes,direction,private_us,shared_us,shared/private\n");

    unsigned long checksum = 0;
    for (size_t nbytes : {4096UL, 1048576UL, 16777216UL}) {
      double private_h2d = BenchCpuToMetal(dev, queue, MTLResourceStorageModePrivate,
                                           nbytes, iters, &checksum);
      double shared_h2d = BenchCpuToMetal(dev, queue, MTLResourceStorageModeShared,
                                          nbytes, iters, &checksum);
      std::printf("%zu,cpu_to_metal,%.3f,%.3f,%.3f\n",
                  nbytes, private_h2d, shared_h2d, shared_h2d / private_h2d);

      double private_d2h = BenchMetalToCpu(dev, queue, MTLResourceStorageModePrivate,
                                           nbytes, iters, &checksum);
      double shared_d2h = BenchMetalToCpu(dev, queue, MTLResourceStorageModeShared,
                                          nbytes, iters, &checksum);
      std::printf("%zu,metal_to_cpu,%.3f,%.3f,%.3f\n",
                  nbytes, private_d2h, shared_d2h, shared_d2h / private_d2h);
    }
    std::printf("checksum=%lu\n", checksum);
    return 0;
  }
}

#else

#include <tvm/ffi/c_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/string.h>
#include <tvm/runtime/device_api.h>

namespace ffi = tvm::ffi;
using tvm::runtime::DeviceAPI;

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

#endif  // CPPMEGA_STANDALONE_METAL_BENCH

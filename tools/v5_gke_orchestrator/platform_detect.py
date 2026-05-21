#!/usr/bin/env python3
"""Per-file platform detection for C++ source code (v4 enrichment).

Python port of tools/cpp_chunker/src/platform.rs.
Scans preprocessor macros and #include directives to detect target platforms
across 6 categories: OS, RTOS, GPU/accelerator, architecture, compiler, C++ standard.

Uses compiled regex + set lookups (fast enough for per-file detection).
"""

import re
from typing import TypeAlias

PlatformInfo: TypeAlias = dict[str, list[str] | str | None]

# ---------------------------------------------------------------------------
# Pattern tables — must match platform.rs exactly
# ---------------------------------------------------------------------------

# (pattern, category, label)
MACRO_PATTERNS = [
    # ── Operating Systems ─────────────────────────────────────────────
    ("_WIN32", "os", "windows"),
    ("_WIN64", "os", "windows"),
    ("WIN32", "os", "windows"),
    ("__WINDOWS__", "os", "windows"),
    ("WINAPI", "os", "windows"),
    ("_WINRT_DLL", "os", "windows"),
    ("__WINRT__", "os", "windows"),
    ("_MSC_VER", "os", "windows"),
    ("__linux__", "os", "linux"),
    ("__linux", "os", "linux"),
    ("__gnu_linux__", "os", "linux"),
    ("__APPLE__", "os", "macos"),
    ("__MACH__", "os", "macos"),
    ("TARGET_OS_MAC", "os", "macos"),
    ("TARGET_OS_IPHONE", "os", "ios"),
    ("TARGET_OS_IOS", "os", "ios"),
    ("TARGET_OS_TV", "os", "tvos"),
    ("TARGET_OS_WATCH", "os", "watchos"),
    ("__FreeBSD__", "os", "freebsd"),
    ("__OpenBSD__", "os", "openbsd"),
    ("__NetBSD__", "os", "netbsd"),
    ("__DragonFly__", "os", "dragonfly"),
    ("__sun", "os", "solaris"),
    ("__SVR4", "os", "solaris"),
    ("__illumos__", "os", "illumos"),
    ("__sunos__", "os", "solaris"),
    ("_AIX", "os", "aix"),
    ("__hpux", "os", "hpux"),
    ("__QNX__", "os", "qnx"),
    ("__QNXNTO__", "os", "qnx"),
    ("__Fuchsia__", "os", "fuchsia"),
    ("__HAIKU__", "os", "haiku"),
    ("__serenity__", "os", "serenity"),
    ("__ANDROID__", "os", "android"),
    ("ANDROID", "os", "android"),
    ("__EMSCRIPTEN__", "os", "emscripten"),
    ("__wasm__", "os", "wasm"),
    ("__CYGWIN__", "os", "cygwin"),
    ("__MSYS__", "os", "msys"),
    ("__MINGW32__", "os", "mingw"),
    ("__MINGW64__", "os", "mingw"),
    ("__native_client__", "os", "nacl"),
    ("__ORBIS__", "os", "playstation"),
    ("__PROSPERO__", "os", "playstation"),
    ("_XBOX_ONE", "os", "xbox"),
    ("_GAMING_XBOX", "os", "xbox"),
    ("__SWITCH__", "os", "nintendo"),
    ("NN_NINTENDO_SDK", "os", "nintendo"),
    # ── RTOS ──────────────────────────────────────────────────────────
    ("FREERTOS", "rtos", "freertos"),
    ("configUSE_PREEMPTION", "rtos", "freertos"),
    ("portYIELD", "rtos", "freertos"),
    ("xTaskCreate", "rtos", "freertos"),
    ("vTaskDelay", "rtos", "freertos"),
    ("__ZEPHYR__", "rtos", "zephyr"),
    ("CONFIG_ZEPHYR", "rtos", "zephyr"),
    ("__VXWORKS__", "rtos", "vxworks"),
    ("__vxworks", "rtos", "vxworks"),
    ("TX_INCLUDE_USER_DEFINE_FILE", "rtos", "threadx"),
    ("TX_THREAD", "rtos", "threadx"),
    ("CONFIG_NUTTX_VERSION", "rtos", "nuttx"),
    ("__rtems__", "rtos", "rtems"),
    ("__ECOS", "rtos", "ecos"),
    ("MBED_CONF_RTOS_PRESENT", "rtos", "mbedos"),
    ("CH_CFG_ST_FREQUENCY", "rtos", "chibios"),
    ("__SYMBIAN32__", "rtos", "symbian"),
    ("RIOT_VERSION", "rtos", "riot"),
    ("ESP_PLATFORM", "rtos", "espidf"),
    ("ARDUINO", "rtos", "arduino"),
    ("TEENSYDUINO", "rtos", "arduino"),
    ("CMSIS_OS", "rtos", "cmsis"),
    # ── GPU / Accelerator ─────────────────────────────────────────────
    ("__CUDA_ARCH__", "gpu", "cuda"),
    ("__CUDACC__", "gpu", "cuda"),
    ("CUDA_VERSION", "gpu", "cuda"),
    ("__device__", "gpu", "cuda"),
    ("__global__", "gpu", "cuda"),
    ("__host__", "gpu", "cuda"),
    ("__shared__", "gpu", "cuda"),
    ("cudaMalloc", "gpu", "cuda"),
    ("cudaMemcpy", "gpu", "cuda"),
    ("__HIP_PLATFORM_AMD__", "gpu", "hip"),
    ("__HIP__", "gpu", "hip"),
    ("HIP_VERSION", "gpu", "hip"),
    ("__HIP_PLATFORM_NVIDIA__", "gpu", "hip"),
    ("hipMalloc", "gpu", "hip"),
    ("hipMemcpy", "gpu", "hip"),
    ("__SYCL__", "gpu", "sycl"),
    ("SYCL_LANGUAGE_VERSION", "gpu", "sycl"),
    ("__INTEL_LLVM_COMPILER", "gpu", "sycl"),
    ("cl::sycl", "gpu", "sycl"),
    ("sycl::queue", "gpu", "sycl"),
    ("__OPENCL_VERSION__", "gpu", "opencl"),
    ("CL_VERSION", "gpu", "opencl"),
    ("clCreateBuffer", "gpu", "opencl"),
    ("clEnqueueNDRangeKernel", "gpu", "opencl"),
    ("__METAL_VERSION__", "gpu", "metal"),
    ("MTLDevice", "gpu", "metal"),
    ("MTLCommandQueue", "gpu", "metal"),
    ("VK_API_VERSION", "gpu", "vulkan"),
    ("VK_VERSION", "gpu", "vulkan"),
    ("vkCreateInstance", "gpu", "vulkan"),
    ("VkDevice", "gpu", "vulkan"),
    ("_OPENACC", "gpu", "openacc"),
    ("_OPENMP", "gpu", "openmp"),
    ("ID3D12Device", "gpu", "directx"),
    ("ID3D11Device", "gpu", "directx"),
    ("D3D12_COMMAND", "gpu", "directx"),
    ("ze_driver_handle_t", "gpu", "levelzero"),
    ("zeInit", "gpu", "levelzero"),
    # ── Architecture ──────────────────────────────────────────────────
    ("__i386__", "arch", "x86"),
    ("__i386", "arch", "x86"),
    ("_M_IX86", "arch", "x86"),
    ("__i686__", "arch", "x86"),
    ("__X86__", "arch", "x86"),
    ("__x86_64__", "arch", "x64"),
    ("__x86_64", "arch", "x64"),
    ("__amd64__", "arch", "x64"),
    ("__amd64", "arch", "x64"),
    ("_M_X64", "arch", "x64"),
    ("_M_AMD64", "arch", "x64"),
    ("__arm__", "arch", "arm32"),
    ("_M_ARM", "arch", "arm32"),
    ("__ARM_ARCH", "arch", "arm32"),
    ("__thumb__", "arch", "arm32"),
    ("__aarch64__", "arch", "arm64"),
    ("_M_ARM64", "arch", "arm64"),
    ("__ARM64__", "arch", "arm64"),
    ("__riscv", "arch", "riscv"),
    ("__riscv__", "arch", "riscv"),
    ("__RISCV__", "arch", "riscv"),
    ("__mips__", "arch", "mips"),
    ("__mips", "arch", "mips"),
    ("_MIPS_ARCH", "arch", "mips"),
    ("__mips64", "arch", "mips64"),
    ("__powerpc__", "arch", "powerpc"),
    ("__ppc__", "arch", "powerpc"),
    ("__PPC__", "arch", "powerpc"),
    ("_ARCH_PPC", "arch", "powerpc"),
    ("__powerpc64__", "arch", "powerpc64"),
    ("__ppc64__", "arch", "powerpc64"),
    ("_ARCH_PPC64", "arch", "powerpc64"),
    ("__s390__", "arch", "s390"),
    ("__s390x__", "arch", "s390x"),
    ("__sparc__", "arch", "sparc"),
    ("__sparc", "arch", "sparc"),
    ("__sparc64__", "arch", "sparc64"),
    ("__wasm32__", "arch", "wasm32"),
    ("__wasm64__", "arch", "wasm64"),
    ("__loongarch__", "arch", "loongarch"),
    ("__loongarch64", "arch", "loongarch64"),
    ("__ia64__", "arch", "ia64"),
    ("_M_IA64", "arch", "ia64"),
    ("__alpha__", "arch", "alpha"),
    ("__m68k__", "arch", "m68k"),
    ("__AVR__", "arch", "avr"),
    ("__AVR", "arch", "avr"),
    ("__XTENSA__", "arch", "xtensa"),
    ("__SSE__", "arch", "x86_sse"),
    ("__SSE2__", "arch", "x86_sse2"),
    ("__SSE4_1__", "arch", "x86_sse4"),
    ("__AVX__", "arch", "x86_avx"),
    ("__AVX2__", "arch", "x86_avx2"),
    ("__AVX512F__", "arch", "x86_avx512"),
    ("__ARM_NEON", "arch", "arm_neon"),
    ("__ARM_NEON__", "arch", "arm_neon"),
    ("__ARM_SVE", "arch", "arm_sve"),
    # ── Compiler ──────────────────────────────────────────────────────
    ("__GNUC__", "compiler", "gcc"),
    ("__clang__", "compiler", "clang"),
    ("_MSC_VER", "compiler", "msvc"),
    ("__INTEL_COMPILER", "compiler", "icc"),
    ("__ICC", "compiler", "icc"),
    ("__NVCC__", "compiler", "nvcc"),
    ("__NVCOMPILER", "compiler", "nvc"),
    ("__EMSCRIPTEN__", "compiler", "emscripten"),
    ("__BORLANDC__", "compiler", "borland"),
    ("__SUNPRO_CC", "compiler", "suncc"),
    ("__xlC__", "compiler", "xlc"),
    ("__MINGW32__", "compiler", "mingw"),
    ("__MINGW64__", "compiler", "mingw"),
    ("__PGI", "compiler", "pgi"),
    ("__ARMCC_VERSION", "compiler", "armcc"),
    ("__HIPCC__", "compiler", "hipcc"),
    # ── C++ Standard feature-test macros ──────────────────────────────
    ("__cpp_concepts", "cpp_std", "c++20"),
    ("__cpp_modules", "cpp_std", "c++20"),
    ("__cpp_coroutines", "cpp_std", "c++20"),
    ("__cpp_constexpr_dynamic_alloc", "cpp_std", "c++20"),
    ("__cpp_lib_ranges", "cpp_std", "c++20"),
    ("__cpp_lib_format", "cpp_std", "c++20"),
    ("__cpp_lib_expected", "cpp_std", "c++23"),
    ("__cpp_lib_print", "cpp_std", "c++23"),
    ("__cpp_static_call_operator", "cpp_std", "c++23"),
    ("__cpp_multidimensional_subscript", "cpp_std", "c++23"),
    ("__cpp_if_consteval", "cpp_std", "c++23"),
    ("__cpp_deducing_this", "cpp_std", "c++23"),
]

# (header_pattern, category, label) — prefix or exact match on #include path
HEADER_PATTERNS = [
    # Windows
    ("windows.h", "os", "windows"),
    ("Windows.h", "os", "windows"),
    ("winnt.h", "os", "windows"),
    ("winsock2.h", "os", "windows"),
    ("winsock.h", "os", "windows"),
    ("ws2tcpip.h", "os", "windows"),
    ("direct.h", "os", "windows"),
    ("io.h", "os", "windows"),
    ("d3d11.h", "os", "windows"),
    ("d3d12.h", "os", "windows"),
    ("d3dx9.h", "os", "windows"),
    ("dxgi.h", "os", "windows"),
    ("combaseapi.h", "os", "windows"),
    ("objbase.h", "os", "windows"),
    ("atlbase.h", "os", "windows"),
    ("shlwapi.h", "os", "windows"),
    ("shlobj.h", "os", "windows"),
    # POSIX
    ("unistd.h", "os", "posix"),
    ("sys/socket.h", "os", "posix"),
    ("sys/types.h", "os", "posix"),
    ("sys/stat.h", "os", "posix"),
    ("sys/mman.h", "os", "posix"),
    ("sys/wait.h", "os", "posix"),
    ("sys/ioctl.h", "os", "posix"),
    ("pthread.h", "os", "posix"),
    ("dlfcn.h", "os", "posix"),
    ("dirent.h", "os", "posix"),
    ("termios.h", "os", "posix"),
    ("fcntl.h", "os", "posix"),
    ("poll.h", "os", "posix"),
    # Linux
    ("sys/epoll.h", "os", "linux"),
    ("linux/io_uring.h", "os", "linux"),
    ("linux/futex.h", "os", "linux"),
    ("sys/inotify.h", "os", "linux"),
    ("sys/signalfd.h", "os", "linux"),
    ("sys/eventfd.h", "os", "linux"),
    ("sys/timerfd.h", "os", "linux"),
    ("linux/", "os", "linux"),
    # macOS
    ("mach/mach.h", "os", "macos"),
    ("CoreFoundation/", "os", "macos"),
    ("Foundation/", "os", "macos"),
    ("Cocoa/", "os", "macos"),
    ("AppKit/", "os", "macos"),
    ("UIKit/", "os", "ios"),
    ("dispatch/dispatch.h", "os", "macos"),
    ("sys/sysctl.h", "os", "macos"),
    ("TargetConditionals.h", "os", "macos"),
    # BSD
    ("sys/event.h", "os", "bsd"),
    # Android
    ("android/", "os", "android"),
    ("jni.h", "os", "android"),
    # CUDA
    ("cuda.h", "gpu", "cuda"),
    ("cuda_runtime.h", "gpu", "cuda"),
    ("cuda_runtime_api.h", "gpu", "cuda"),
    ("cuda_fp16.h", "gpu", "cuda"),
    ("cuda_bf16.h", "gpu", "cuda"),
    ("cublas", "gpu", "cuda"),
    ("cudnn", "gpu", "cuda"),
    ("cufft", "gpu", "cuda"),
    ("cusparse", "gpu", "cuda"),
    ("curand", "gpu", "cuda"),
    ("cusolver", "gpu", "cuda"),
    ("nccl.h", "gpu", "cuda"),
    ("nvtx3/", "gpu", "cuda"),
    ("thrust/", "gpu", "cuda"),
    ("cub/", "gpu", "cuda"),
    ("cutlass/", "gpu", "cuda"),
    # HIP / ROCm
    ("hip/hip_runtime.h", "gpu", "hip"),
    ("hip/hip_runtime_api.h", "gpu", "hip"),
    ("hip/", "gpu", "hip"),
    ("rocblas/", "gpu", "hip"),
    ("hipblas/", "gpu", "hip"),
    ("miopen/", "gpu", "hip"),
    ("rccl/", "gpu", "hip"),
    ("rocrand/", "gpu", "hip"),
    # SYCL / oneAPI
    ("CL/sycl.hpp", "gpu", "sycl"),
    ("sycl/sycl.hpp", "gpu", "sycl"),
    ("oneapi/", "gpu", "sycl"),
    # OpenCL
    ("CL/cl.h", "gpu", "opencl"),
    ("CL/opencl.h", "gpu", "opencl"),
    ("OpenCL/", "gpu", "opencl"),
    # Metal
    ("Metal/", "gpu", "metal"),
    ("MetalKit/", "gpu", "metal"),
    ("MetalPerformanceShaders/", "gpu", "metal"),
    # Vulkan
    ("vulkan/vulkan.h", "gpu", "vulkan"),
    ("vulkan/", "gpu", "vulkan"),
    # DirectX
    ("d3d12.h", "gpu", "directx"),
    ("d3d11.h", "gpu", "directx"),
    ("d3dcompiler.h", "gpu", "directx"),
    # OpenGL
    ("GL/gl.h", "gpu", "opengl"),
    ("GL/glew.h", "gpu", "opengl"),
    ("GLES2/", "gpu", "opengl_es"),
    ("GLES3/", "gpu", "opengl_es"),
    # FreeRTOS
    ("FreeRTOS.h", "rtos", "freertos"),
    ("freertos/", "rtos", "freertos"),
    ("task.h", "rtos", "freertos"),
    # Zephyr
    ("zephyr/", "rtos", "zephyr"),
    # ARM / x86 intrinsics
    ("arm_neon.h", "arch", "arm_neon"),
    ("arm_sve.h", "arch", "arm_sve"),
    ("immintrin.h", "arch", "x86_simd"),
    ("emmintrin.h", "arch", "x86_sse2"),
    ("xmmintrin.h", "arch", "x86_sse"),
    ("smmintrin.h", "arch", "x86_sse4"),
    ("nmmintrin.h", "arch", "x86_sse4"),
    ("avxintrin.h", "arch", "x86_avx"),
    ("avx2intrin.h", "arch", "x86_avx2"),
    ("avx512fintrin.h", "arch", "x86_avx512"),
    ("wasm_simd128.h", "arch", "wasm_simd"),
    # ML frameworks
    ("NvInfer.h", "gpu", "tensorrt"),
    ("NvInferRuntime.h", "gpu", "tensorrt"),
    ("tvm/", "gpu", "tvm"),
    ("tensorflow/", "gpu", "tensorflow"),
    ("xla/", "gpu", "xla"),
]

# Build lookup structures once at import time
_MACRO_SET: dict[str, list[tuple[str, str]]] = {}  # pattern -> [(category, label), ...]
for _pat, _cat, _label in MACRO_PATTERNS:
    _MACRO_SET.setdefault(_pat, []).append((_cat, _label))

# Pre-compile word-boundary regex for each macro pattern
_MACRO_REGEXES: dict[str, re.Pattern[str]] = {}
for _pat in _MACRO_SET:
    # Word boundary: not preceded/followed by alphanumeric or underscore
    _escaped = re.escape(_pat)
    _MACRO_REGEXES[_pat] = re.compile(r'(?<![a-zA-Z0-9_])' + _escaped + r'(?![a-zA-Z0-9_])')

# Include line regex
_INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.MULTILINE)

# __cplusplus value regex
_CPLUSPLUS_RE = re.compile(r'__cplusplus\s*>=?\s*(\d+)L?\b')

# C++ standard version to label
_CPLUSPLUS_MAP = {
    199711: "c++98",
    201103: "c++11",
    201402: "c++14",
    201703: "c++17",
    202002: "c++20",
    202302: "c++23",
    202612: "c++26",
}

# Modern C++ headers → minimum standard
_STD_HEADERS = {
    # C++11
    "thread": "c++11", "mutex": "c++11", "atomic": "c++11",
    "chrono": "c++11", "regex": "c++11", "array": "c++11",
    "tuple": "c++11", "unordered_map": "c++11", "unordered_set": "c++11",
    "condition_variable": "c++11", "future": "c++11",
    "random": "c++11", "ratio": "c++11", "type_traits": "c++11",
    # C++17
    "filesystem": "c++17", "optional": "c++17", "variant": "c++17",
    "any": "c++17", "string_view": "c++17", "charconv": "c++17",
    "execution": "c++17", "memory_resource": "c++17",
    # C++20
    "ranges": "c++20", "concepts": "c++20", "coroutine": "c++20",
    "span": "c++20", "format": "c++20", "source_location": "c++20",
    "semaphore": "c++20", "latch": "c++20", "barrier": "c++20",
    "bit": "c++20", "numbers": "c++20", "compare": "c++20",
    "stop_token": "c++20", "jthread": "c++20", "syncstream": "c++20",
    # C++23
    "expected": "c++23", "print": "c++23", "stacktrace": "c++23",
    "generator": "c++23", "mdspan": "c++23", "flat_map": "c++23",
    "flat_set": "c++23", "stdfloat": "c++23",
}

_STD_ORDER = {"c++98": 0, "c++11": 1, "c++14": 2, "c++17": 3, "c++20": 4, "c++23": 5, "c++26": 6}


def detect_platforms(source: str) -> PlatformInfo | None:
    """Detect platforms from C++ source code.

    Returns dict matching PlatformInfo:
        {"os": [...], "rtos": [...], "gpu": [...], "arch": [...],
         "compiler": [...], "cpp_std": "c++17" or None}

    Returns None if no platform detected.
    """
    os_set: set[str] = set()
    rtos_set: set[str] = set()
    gpu_set: set[str] = set()
    arch_set: set[str] = set()
    compiler_set: set[str] = set()
    cpp_std_set: set[str] = set()

    cat_map = {
        "os": os_set, "rtos": rtos_set, "gpu": gpu_set,
        "arch": arch_set, "compiler": compiler_set, "cpp_std": cpp_std_set,
    }

    # Pass 1: Macro pattern matching with word boundaries
    for pat, regex in _MACRO_REGEXES.items():
        if regex.search(source):
            for cat, label in _MACRO_SET[pat]:
                cat_map[cat].add(label)

    # Pass 2: #include header matching
    for m in _INCLUDE_RE.finditer(source):
        include_path = m.group(1)
        for header, cat, label in HEADER_PATTERNS:
            if include_path == header or include_path.startswith(header):
                cat_map[cat].add(label)

        # Check standard library headers for C++ version detection
        # Strip directory prefix (e.g., "bits/stl_algobase.h" -> skip)
        base = include_path.split("/")[-1] if "/" not in include_path else None
        if base is None:
            base = include_path  # No slash, use as-is
        base_no_ext = base.replace(".h", "").replace(".hpp", "")
        if base_no_ext in _STD_HEADERS:
            cpp_std_set.add(_STD_HEADERS[base_no_ext])

    # Pass 3: __cplusplus version detection
    for m in _CPLUSPLUS_RE.finditer(source):
        val = int(m.group(1))
        # Find the matching or next-lower standard
        best = None
        for threshold, label in sorted(_CPLUSPLUS_MAP.items()):
            if val >= threshold:
                best = label
        if best:
            cpp_std_set.add(best)

    # Pick highest C++ standard
    cpp_std = None
    if cpp_std_set:
        cpp_std = max(cpp_std_set, key=lambda s: _STD_ORDER.get(s, -1))

    # Check if anything was detected
    if not any([os_set, rtos_set, gpu_set, arch_set, compiler_set, cpp_std]):
        return None

    return {
        "os": sorted(os_set),
        "rtos": sorted(rtos_set),
        "gpu": sorted(gpu_set),
        "arch": sorted(arch_set),
        "compiler": sorted(compiler_set),
        "cpp_std": cpp_std,
    }

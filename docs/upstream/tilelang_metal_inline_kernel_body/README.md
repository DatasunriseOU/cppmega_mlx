# TileLang Metal "emit kernel body inline" investigation

Status: **NO PR FILED**.
The bug described in the task does not exist in the current `metal-gemm-upstream-rebase`
branch (jorgecurious/tilelang) or the vendored TileLang/tvm submodule on commit
`0e15b27`. TileLang's Metal codegen already emits prim_func bodies directly inside
a `kernel void` function with no `inline void` helper wrapper, so no codegen change
is needed.

## What the task asked for

> Fix the Metal codegen to emit kernel body directly instead of an inline void
> wrapper when the body contains threadgroup allocations.

## What we actually found

### 1. The codegen already emits `kernel void` directly

`src/target/codegen_metal.cc::CodeGenTileLangMetal::AddFunction` (lines 60-187) and
`3rdparty/tvm/src/target/source/codegen_metal.cc::CodeGenMetal::AddFunction`
(lines 59-179) both emit:

```
kernel void <name>(
  device const T* A [[ buffer(0) ]],
  ...
  uint3 blockIdx [[threadgroup_position_in_grid]],
  uint3 threadIdx [[thread_position_in_threadgroup]]
) {
  threadgroup float buf_dyn_shmem[N];   // T.alloc_shared
  simdgroup_float8x8 rC[K];              // T.alloc_fragment
  ...
}
```

There is no `inline void` wrapper anywhere in the Metal codegen. `grep -rn "inline\b"
src/target/codegen_metal.* 3rdparty/tvm/src/target/source/codegen_metal.*` returns
zero matches. There is no `PrintFunctionSignature` override on the Metal target,
and the codegen does not call the parent `CodeGenC::AddFunction` (which is the only
path that uses `PrintFunctionSignature`).

### 2. Both `T.alloc_shared` and `T.alloc_fragment` emissions compile cleanly

We lowered two PrimFuncs and ran `xcrun --sdk macosx metal -c` against the raw
TileLang MSL output (no cppmega post-processing):

* `tilelang_alloc_shared_emitted.metal`: a 128-element add kernel with
  `T.alloc_shared`. Compiles. AIR generated.
* `tilelang_alloc_fragment_emitted.metal`: a 128x128 GEMM with
  `T.alloc_fragment` + `T.gemm` (uses `simdgroup_float8x8`). Compiles. AIR generated.

Both `.metal` files are checked in next to this README so the verification is
reproducible.

### 3. The cppmega workaround addresses a different problem

`cppmega_mlx/nn/_tilelang/_msl_transform.py::_inline_tilelang_kernel_body` is real,
but it is not patching around an `inline void` wrapper. It is adapting the
TileLang-emitted `kernel void <name>(...)` so the body can be embedded inside the
kernel `mx.fast.metal_kernel` auto-generates around the caller-supplied `source=`.
Specifically:

- MLX's `mx.fast.metal_kernel(name, source, ...)` injects `source` into an
  auto-generated kernel signature that does not include `blockIdx`/`threadIdx`
  parameters by default.
- The cppmega workaround strips TileLang's outer `kernel void {...}`, then prepends
  `uint3 blockIdx = threadgroup_position_in_grid; uint3 threadIdx =
  thread_position_in_threadgroup;` declarations inside the body so the inner code
  that references `blockIdx.x` etc. continues to work.

This is an **MLX integration concern**, not a TileLang codegen bug. There is
nothing for upstream TileLang to fix. The cppmega Path B
(`_path_b_lowering.py::transform_tilelang_kernel`) does generate an `inline void`
helper as part of its rewrite, but that is downstream of TileLang and is the
opposite direction of the fix described in the task.

## Reproducing

```bash
cd /Volumes/external/sources/cppmega.mlx
.venv/bin/python -c "
import tilelang
from tilelang import tvm
from tilelang import language as T

@T.prim_func
def add_kernel(A: T.Tensor((128,), 'float32'),
               B: T.Tensor((128,), 'float32'),
               C: T.Tensor((128,), 'float32')):
    with T.Kernel(1, threads=128) as bx:
        sA = T.alloc_shared((128,), 'float32')
        T.copy(A, sA)
        for i in T.serial(128):
            C[i] = sA[i] + B[i]

from tilelang.engine.lower import lower as tl_lower
artifact = tl_lower(add_kernel, target=tvm.target.Target('metal'))
print(artifact.kernel_source)
" > /tmp/tl.metal
# Strip the dev-root banner that the venv prints before the source
tail -n +2 /tmp/tl.metal > /tmp/tl_stripped.metal
xcrun --sdk macosx metal -c /tmp/tl_stripped.metal -o /tmp/tl.air
# Exit code 0, AIR produced.
```

## Recommendation

No upstream PR is warranted. If a future TileLang refactor introduces an
`inline void` helper path (e.g. for inter-function calls), the fix described in
the task would become applicable. Today the codegen path is already
"emit body directly inside `kernel void`", which is the correct shape.

The cppmega `_inline_tilelang_kernel_body` workaround should remain in place
because it solves the MLX integration problem, not a TileLang codegen problem.

## Working tree

Branch `cppmega/metal-emit-kernel-body-inline` was created in
`/tmp/tl_inline_kernel_body` for the investigation but no commits were made
beyond the branch checkpoint. Remote `fork` was added pointing to
`apstenku123/tilelang`. No push, no PR.

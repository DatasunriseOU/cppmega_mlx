# TileLang Metal codegen: emit Metal builtins directly

Filed PRs:

* Tilelang supermodule half: https://github.com/tile-ai/tilelang/pull/2143
* TileLang/tvm submodule mirror: https://github.com/tile-ai/tvm/pull/37

Branch: `apstenku123:cppmega/metal-emit-builtins-directly` (same name on both forks)
Stacks on: tile-ai/tilelang#2130 (jorgecurious metal-gemm-upstream-rebase)

## What this fixes

TileLang's Metal codegen names the thread/block kernel-launch parameters
using the CUDA-style identifiers `blockIdx` and `threadIdx`. The MSL output
of `lower(prim_func, target='metal')` therefore looks like:

```cpp
kernel void smoke_kernel(
    device const half4* A [[ buffer(0) ]],
    device half4* C [[ buffer(1) ]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]]
) {
    C[((((int)threadIdx.x) * 4) / 4)] = A[((((int)threadIdx.x) * 4) / 4)];
}
```

The named parameters mirror CUDA's `blockIdx.x`/`threadIdx.x`, but downstream
consumers that inline the body of `kernel void` into another kernel (e.g.
the cppmega.mlx Path C ports that splice TileLang-emitted bodies into
`mx.fast.metal_kernel` `source=` strings) end up having to:

* Inject `uint3 blockIdx = threadgroup_position_in_grid;` and
  `uint3 threadIdx = thread_position_in_threadgroup;` shims so the body's
  references still bind, then
* Regex-substitute every `((int)threadIdx.x)` etc. back to
  `((int)thread_position_in_threadgroup.x)`, then
* Drop the now-dead alias declarations.

See the canonicalization helpers in cppmega.mlx
(`cppmega_mlx/nn/_tilelang/_msl_transform.py`):

* `_metal_builtin_for_tilelang_alias`
* `_rewrite_tilelang_builtin_axis`
* `_rewrite_tilelang_builtin_axis_cast`
* `_canonicalize_tilelang_builtin_aliases`
* `_drop_alias_decl_if_unused`

The whole chain is pure overhead: every consumer either lives with the
alias or post-processes it back to the Metal builtin.

## What this PR does

In TileLang's Metal codegen, the thread/block kernel parameters are now
declared using the Metal builtin identifiers themselves:

```cpp
uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]
```

`BindThreadIndex` translates the CUDA-style `IterVar::thread_tag`
(`"blockIdx.x"`, `"threadIdx.y"`, ...) to the matching Metal builtin
reference (`threadgroup_position_in_grid.x`, ...) before recording it in
`var_idmap_`. The body therefore emits `((int)threadgroup_position_in_grid.x)`
directly. The `name_supply_` reservation also keeps the legacy
`blockIdx`/`threadIdx` names blocked so the rest of the kernel cannot
collide with them.

### Before / after MSL

Before:

```cpp
kernel void smoke_kernel(
    device const half4* A [[ buffer(0) ]],
    device half4* C [[ buffer(1) ]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]]
) {
    C[((((int)threadIdx.x) * 4) / 4)] = A[((((int)threadIdx.x) * 4) / 4)];
}
```

After:

```cpp
kernel void smoke_kernel(
    device const half4* A [[ buffer(0) ]],
    device half4* C [[ buffer(1) ]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]
) {
    C[((((int)thread_position_in_threadgroup.x) * 4) / 4)] = A[((((int)thread_position_in_threadgroup.x) * 4) / 4)];
}
```

## Files in this directory

* `0001-metal-emit-builtins-directly.patch` - the TileLang half. Touches
  `src/target/codegen_metal.cc`. ~36 LOC.
* `0002-tvm-metal-emit-builtins-directly.patch` - the vendored TVM half.
  Touches `3rdparty/tvm/src/target/source/codegen_metal.cc`. ~30 LOC.

The TileLang half is filed as one PR to `tile-ai/tilelang`. The TVM half
needs to land in `TileLang/tvm` (the vendored fork) before a TileLang
release that bumps the submodule.

## Stacking

This PR depends on `jorgecurious/tilelang:metal-gemm-upstream-rebase`
(PR tile-ai/tilelang#2130) for the simdgroup-store hardening that the
Path C ports rely on. The diff applies cleanly on top of that branch.

## Verification

* Build: TileLang `ninja -j8` succeeds against the patched
  `src/target/codegen_metal.cc` and submodule `codegen_metal.cc`.
* Smoke test: `lower(prim_func, target='metal')` no longer emits
  `int blockIdx_x = ...;`-style aliases, the Metal builtin name appears
  directly in body references.
* cppmega Path C tests still pass: the regex helpers in
  `_msl_transform.py` become no-ops because the emitted MSL already uses
  the Metal builtin names; the helpers are kept (idempotent) so the
  fallback works against unpatched TileLang releases as well.

## Risk

* Limited to the `tilelang_metal` and `metal` codegen paths in TileLang
  and the vendored TVM fork respectively. CUDA, ROCm, OpenCL, WebGPU,
  and CPU codegen are unaffected.
* MSL parameter names are user-chosen identifiers; renaming the
  parameter does not change semantics as the `[[threadgroup_position_in_grid]]`
  attribute is what binds the value. Apple's MSL spec permits the
  parameter identifier to match the attribute name.
* `name_supply_` reserves both the new Metal builtin names and the
  legacy `blockIdx`/`threadIdx` names so a future user-defined symbol
  cannot collide, and the existing assertion-based contract on the
  reservation order is preserved.

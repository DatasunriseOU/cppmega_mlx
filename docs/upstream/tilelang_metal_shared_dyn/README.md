# Metal `shared.dyn` storage scope: investigation outcome

## Original blocker (as reported)

    Fatal: Unknown storage scope `shared.dyn`

This was the error users saw in earlier TileLang builds when
``T.alloc_shared(scope="shared.dyn")`` was emitted into a kernel that the
Metal codegen reached without `MergeSharedMemoryAllocations` having
collapsed it.

## Status on apple-head: already fixed (no patch needed)

Walking the live source on `tilelang.apple-head @ 7f4a5cb8`:

* `src/target/codegen_metal.cc::PrintStorageScope` (line 493) accepts
  ``shared.dyn`` and prints it as ``threadgroup`` (identical to plain
  ``shared``).

* `tir/transforms/merge_shared_memory_allocations.cc` collapses one or
  many ``shared.dyn`` allocations into a single backing buffer
  ``buf_dyn_shmem`` whose extent is the sum of the merged extents. When
  every alloc has compile-time-constant extent, this merged buffer also
  has constant extent, and codegen emits a static-sized
  ``threadgroup half buf_dyn_shmem[N]`` declaration.

The original task's static-shape probe lowers without error — the
"Unknown storage scope" blocker doesn't reproduce on this branch:

```
$ .venv/bin/python docs/upstream/tilelang_metal_shared_dyn/test_shared_dyn_probe.py
================================================================================
Case                                     Status     Notes
================================================================================
static_size_dyn                          OK         merged into static threadgroup decl
merged_dyn                               OK         merged into static threadgroup decl
symbolic_dyn (known limitation)          FAILED     ICHECK constant_size > 0
```

## Remaining limitation: symbolic-extent shared.dyn

When the dyn-shmem extent depends on a *symbolic* dimension (e.g.
``T.alloc_shared((N,), scope="shared.dyn")`` for symbolic ``N``), the
merge pass can't fold the size to a constant. ``codegen_metal.cc::
VisitStmt_(AllocateNode)`` then trips:

    src/target/codegen_metal.cc:494: Check failed: constant_size > 0
    (0 vs. 0): Can only handle constant size stack allocation for now

This is a true Metal-codegen gap (CUDA handles it via
``extern __shared__ T buf[];`` plus ``cuLaunchKernel(..., dyn_shmem_size,
...)``). A working Metal port needs:

1. **Codegen**: when the kernel's launch params include
   ``kUseDynamicSharedMemoryTag``, emit a kernel parameter
   ``threadgroup uint8_t* buf_dyn_shmem [[threadgroup(0)]]`` instead of
   a stack-allocated array, and route the `Allocate(scope="shared.dyn")`
   to that parameter.
2. **Runtime**: `metal_module.mm::MetalWrappedFunc::operator()` extracts
   ``wl.dyn_shmem_size`` from `LaunchParamConfig::Extract(args)` (the
   field already exists in `ThreadWorkLoad`, see `thread_storage_scope.h`)
   and calls `[encoder setThreadgroupMemoryLength:dyn_shmem_size atIndex:0]`
   before `dispatchThreadgroups`.
3. **Glue**: TileLang's `LowerDeviceKernelLaunch` already adds the
   `dyn_shmem_size` argument to the host-side call when
   `kUseDynamicSharedMemoryTag` is set; this just needs to be propagated
   through TileLang's Metal launch-param packing (currently the Metal path
   only packs grid + block dims).

Total work: ~50 LOC across `codegen_metal.cc`, `metal_module.mm`, and
the launch-param packing in `engine/lower.py`. Not implemented in this
patch because:

* The originally-blocked kernel (`topk_selector`) was rewritten in
  `cppmega_mlx/nn/_tilelang/topk_selector.py` to use
  `mx.fast.metal_kernel` directly — bypassing TileLang lowering. See
  the module docstring for the rationale.
* No other Path B kernel currently in the cppmega porting plan uses
  symbolic-extent dynamic shared memory.

If a future kernel needs it, the wiring is bounded and well-understood;
file an issue against this `docs/upstream/tilelang_metal_shared_dyn/`
folder with the failing kernel as repro.

## Diff stat

This patch is intentionally a no-op. `0001-metal-shared-dyn-storage-scope.patch`
is a documentation-only artefact recording the investigation outcome
(the original blocker is already resolved in the apple-head branch).

## Test

`test_shared_dyn_probe.py` is the regression probe for both the working
and failing regimes. Run it against the built tilelang to confirm:

* `static_size_dyn` (single dyn alloc with const extent): OK
* `merged_dyn` (two dyn allocs merged by the merge pass): OK
* `symbolic_dyn` (symbolic extent → known limitation): FAILS with
  `ICHECK constant_size > 0`

## Upstream-PR readiness

Nothing to upstream. The "Unknown storage scope" surface fix is already
in TileLang. The symbolic-extent limitation is real but not on cppmega's
critical path — file as separate upstream issue when/if needed.

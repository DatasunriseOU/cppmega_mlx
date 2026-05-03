# TileLang patch: allow mixed-dtype T.gemm via Metal scalar fallback

Artifact: 0001-tilelang-allow-mixed-gemm-dtypes.patch
Authored against: tile-ai/tilelang apple-head branch (HEAD: 7f4a5cb8 "Preserve Metal reduce thread range")
Local branch: cppmega/gemm-mixed-dtype-metal @ a69d6df7

## Blocker

tilelang/tileop/gemm/gemm_base.py:88 asserted self.A.dtype == self.B.dtype,
which blocked chained mixed-precision patterns. The canonical case is the
two-step attention path:

```python
T.gemm(Q_shared, K_shared, S_local, transpose_B=True)   # fp16 × fp16 → fp32
T.gemm(S_local, V_shared, O_local)                       # fp32 × fp16 → fp32
```

The second call has A.dtype = float32 (the accumulator from the first GEMM)
and B.dtype = float16 (V_shared), tripping the assert before any backend
dispatch could choose how to handle it.

This is overly conservative. cuBLAS, CUTLASS, MPS BNNS, and MSL
simdgroup_matrix_multiply accept different precisions for A/B (or for the
accumulator C). The fp16-input/fp32-accumulator case is canonical.

## Design choice: Option 2 + 3 hybrid (Metal-target route to scalar fallback)

The task brief offered three options:

* **Option 1** — drop the assert, let backends silently produce wrong results
  if the cast isn't there.
* **Option 2** — auto-cast in the frontend.
* **Option 3** — explicit cast_dtype kwarg.

We implemented a **safer Option 2 variant scoped to Metal**:

1. Drop the strict assert in GemmBase.in_dtype (it was the load-bearing
   constraint, but each backend already knows what dtypes it accepts).
2. In the dispatcher (Gemm._select_gemm_instruction for Metal), detect
   A.dtype != B.dtype and route to GemmInst.Scalar. PR #2118 already
   added a GemmMetalScalar lowering — that path emits scalar
   T.cast(..., accum_dtype) for each element of A and B independently,
   which is exactly the correct semantics for mixed precision.
3. Update the MetalFragmentToSimdgroup transform so it does NOT promote
   the C accumulator (or fragment A) of a mixed-dtype GEMM to
   metal.simdgroup scope. Otherwise the scalar fallback would then
   dereference a simdgroup_float32 8×8 register elementwise, which the
   Metal codegen rejects.

CUDA / ROCm / Hopper / Blackwell paths still go through the C++
GemmGetGemmInst selection. Their MMA emitters use a single in_dtype
value internally (per-tilelang convention); if a backend genuinely
cannot lower a mixed-dtype gemm, the layout/emitter check raises with a
clearer message than the original frontend assert.

## Files changed

| File                                              | +/-           |
| ------------------------------------------------- | ------------- |
| tilelang/tileop/gemm/gemm_base.py                 | +22 / -1      |
| tilelang/tileop/gemm/__init__.py                  | +30 / -0      |
| tilelang/transform/metal_fragment_to_simdgroup.py | +63 / -7      |
| testing/python/metal/test_metal_codegen_linux.py  | +55 / -0      |
| **Total**                                         | **+170 / -8** |

The diff has zero C++ changes — Python only.

## Test results

### Probe: docs/upstream/test_sparse_mla_pipeline.py


Kernel                    Status     Error
======================================================================
k1_simple_gemm            OK
k2_pipelined_gemm         FAILED     InternalError: Check failed: float32x4 (unrelated 3-D pipeline issue)
k3_multi_gemm             OK         (was: AssertionError "A and B must have the same dtype")


**k3_multi_gemm now lowers OK.** This is the key probe for the
mixed-precision attention chain. k2_pipelined_gemm still fails — but
on an entirely different blocker (3-D buffers from T.Pipelined(num_stages=2)),
documented in docs/upstream/local_build_status.md issue #2.

### Upstream regression suite

* testing/python/metal/: 50 passed, 6 failed, 3 skipped (vs. baseline:
  49 passed, 6 failed, 3 skipped — net **+1 test passing**, no regressions).
  The new test test_attention_chain_mixed_dtype_metal_codegen passes;
  test_t_gemm_metal_codegen_pipelined_float32 flipped to passing
  (the conservative metal.simdgroup rewrite skip helps it too).
* testing/python/cpu/test_tilelang_cpu_tgemm.py: 11 passed (no
  regressions; CPU scalar already handles mixed dtypes).

The 6 baseline metal-test failures are pre-existing; this patch does not
touch their failure paths.

### cppmega.mlx local suite

* tests/test_tilelang_*.py: 130 passed.

## Performance / profiler readout

This patch is a correctness/codegen attempt, not a throughput optimization. I
rechecked the generated Metal source from
`docs/upstream/test_sparse_mla_pipeline.py::k3_multi_gemm` on 2026-05-03 with
the local TileLang dev tree imported from
`/private/tmp/tilelang_apple_head/tilelang` at `a69d6df7`. The TileLang
`lower()` path and the scoped pytest still pass, but `xcrun --sdk macosx metal
-c` on the generated MSL fails, so there is no honest runtime profiler number
for this artifact yet. The simple same-dtype `docs/upstream/test_metal_gemm.py`
MSL compiles with the same `xcrun` command, so this is not just a local Metal
toolchain failure.

Observed lowering:

* The first same-dtype `Q_shared x K_shared -> S_local` GEMM still attempts the
  simdgroup path: generated MSL contains one `simdgroup_multiply_accumulate`
  and two `simdgroup_load` sites.
* Because `S_local` is later consumed by the mixed-dtype scalar fallback, the
  conservative simdgroup-rewrite skip leaves it as `thread float S_local[1024]`.
  That makes the first simdgroup call invalid MSL:

```text
error: no matching function for call to 'simdgroup_multiply_accumulate'
candidate template ignored: could not match 'simdgroup_matrix<R, Cols, Rows>'
against 'float'
```

* The second mixed-dtype `S_local x V_shared -> O_local` GEMM deliberately does
  **not** use `simdgroup_multiply_accumulate`; generated MSL contains an
  explicit scalar loop:

```metal
for (int i_7 = 0; i_7 < 32; ++i_7) {
  for (int j_4 = 0; j_4 < 64; ++j_4) {
    for (int k = 0; k < 32; ++k) {
      float a_val = S_local[((i_7 * 32) + k)];
      float b_val = ((float)V_shared[((k * 64) + j_4)]);
      O_local[((i_7 * 64) + j_4)] = (O_local[((i_7 * 64) + j_4)] + (a_val * b_val));
    }
  }
}
```

So the concrete performance finding is stricter than "scalar fallback is slow":
the current patch is not profiler-ready for the chained attention probe because
the same fragment is produced by simdgroup GEMM and consumed by scalar GEMM. A
safe upstream fix has to handle that producer/consumer conflict explicitly, for
example by routing every GEMM that touches a mixed-consumed fragment through the
scalar path, or by materializing a separate scalar/cast copy for the second GEMM.
The latter is the only route with a plausible speed path. A kernel-local
workaround is to keep the second GEMM same-dtype by staging or casting `S_local`
to FP16 before `S x V` when the accuracy budget allows it; a real upstream
speedup would need a separate TileLang Metal lowering/scheduler patch that
supports mixed-input simdgroup MMA or an explicit pre-cast/fused-load strategy.

## Upstream-PR readiness

* **Independent of FP8 / TVM-storage-mode work** — touches only the
  TileLang Python frontend and a single TileLang transform. No
  3rdparty/tvm changes, no codegen_metal.cc changes.
* **Backwards compatible** — when A and B share a dtype, the
  dispatch and rewrite both produce the same IR as before.
* **Targets apple-head / Metal-dev** — applies cleanly on top of the branch
  that already contains PR #2118's GemmMetalScalar lowering.
* **Needs refresh for current public main** — on fresh `tile-ai/tilelang` main
  at `2eec5f0`, `git apply --check` fails because the public tree has drifted:
  `tilelang/transform/metal_fragment_to_simdgroup.py` is absent and the
  `test_metal_codegen_linux.py` / `tilelang/tileop/gemm/__init__.py` hunks no
  longer match. Refresh this patch against the actual upstream PR base before
  submitting it.
* **One follow-up could mirror this on CUDA / ROCm**: route to a
  scalar fallback when A.dtype != B.dtype and the chosen MMA emitter
  would refuse same-dtype inputs. Out of scope here — left for the
  backend-specific PR that implements those scalar fallbacks.

## Reproduction

```bash
cd /tmp/tilelang_apple_head/tilelang
git apply /Volumes/external/sources/cppmega.mlx/docs/upstream/tilelang_gemm_mixed_dtype/0001-tilelang-allow-mixed-gemm-dtypes.patch
ninja -C build -j$(sysctl -n hw.ncpu) tilelang_objs   # Python-only patch — no rebuild needed
.venv/bin/python /Volumes/external/sources/cppmega.mlx/docs/upstream/test_sparse_mla_pipeline.py
.venv/bin/python -m pytest testing/python/metal/test_metal_codegen_linux.py::test_attention_chain_mixed_dtype_metal_codegen -v
```

---
aspect: correctness
provider: grok
model: gpt-5-5-pro
range: main..mlx-z3-wiring
base_ref: aef115bb0fa97b68b541580d957ff6b043e76176
head_ref: 888d368715dee062c804309c99cd15d4ee55468f
timestamp: 2026-05-07T03:30:54.424509+00:00
files: ['bench/tilelang_ports/sparse_mla.json', 'bench/tilelang_ports/topk_selector.json', 'cppmega_mlx/nn/_tilelang/_msl_transform.py', 'cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py', 'cppmega_mlx/nn/_tilelang/fp8_amax.py', 'cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py', 'cppmega_mlx/nn/_tilelang/sparse_mla_path_c.py', 'cppmega_mlx/nn/_tilelang/topk_selector.py', 'tests/test_tilelang_topk.py']
---
**No findings.**

The visible chunk (bench JSON updates + the new `_preload_libz3_for_dev_tilelang()` implementation in `_msl_transform.py`) introduces no correctness bugs of the types requested (off-by-one, error swallowing, mismatched types, race conditions, null/None mishandling, edge-case regressions, etc.).

### Key observations from the diff in this chunk:
- **JSON benchmark files**: Pure data updates (version bumps, timing changes, iters/warmup reductions, and status flips from failure → success with concrete ratios). These are expected outputs from the wiring changes; no code logic altered here.
- **_msl_transform.py**:
  - New helper `_preload_libz3_for_dev_tilelang()` is defensive and well-structured.
  - Uses a class-attribute-based `_done` guard (idempotent) and a `_failed_attempts` cap (=3) as documented.
  - `failed = getattr(..., "_failed_attempts", 0)` + increment only on complete failure path — no uninitialized attribute access.
  - All `ctypes.CDLL` calls are inside `try` with bare `except OSError` (appropriate for dlopen failures; other exceptions would still surface).
  - No swallowed exceptions that mask real problems; failures are counted but preload is best-effort/silent.
  - Candidate list construction is straightforward; no off-by-one, no incorrect defaults, no None mishandling.
  - Darwin-only gating and early return patterns are clean.
  - The hard assert mentioned in the review prompt ("_path_c_rewrite_merge_round hard assert") is **not present** in this chunk, so no visibility into it.

The changes align with the review note ("libz3 preload _failed_attempts cap=3") and appear to clear previous dispatch CRITICALs by making TileLang dev builds reliably importable on macOS without side-effecting production paths.

No correctness regressions or bugs detectable in the provided chunk.

**No findings.**

This chunk (chunk 2 of 3) introduces two new files:

- `cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py`
- `cppmega_mlx/nn/_tilelang/fp8_amax.py`

Both files are large, well-structured TileLang ports with extensive comments, careful handling of Metal vs CUDA block sizing, boundary conditions, causal masking, online-softmax recurrence, register pressure awareness, and explicit OOB guards. The changes visible in this chunk introduce no obvious correctness bugs of the types requested (off-by-one, swallowed exceptions, mismatched types, race conditions, broken None handling, mishandled edge cases, or regressions).

Key areas reviewed with no issues found:

- **dsa_splitk_indexer_loss.py**:
  - `_active_sk_tiles` calculation with explicit `T.max(..., 1)` clamp to avoid skipping initialisation on edge ASq shapes.
  - Consistent `in_bounds` / `valid` predicates and `-inf` priming for index scores on head 0.
  - `USE_MD_PRE` budget gating and fallback path for M/D preloads.
  - `topk_indices` validation + `scatter_` safety in the wrapper.
  - Empty-tensor and zero-size handling.
  - Denominator guards (`denom <= 0 → 1`).
  - No swallowed exceptions; all error paths raise appropriately.

- **fp8_amax.py**:
  - `_pick_block_size` + `_bucket_n` logic with power-of-two snapping and divisibility enforcement.
  - Explicit last-block masking in amax kernel.
  - `atomic_max` usage (single thread).
  - Non-finite `amax_val` detection and loud `FloatingPointError`.
  - `inv_scale` handling (tensor vs float).
  - Contiguous handling and out= support in quantize.

No critical, high, medium, or low correctness issues detected in the visible diff chunk. All edge-case mitigations (ASq % BLOCK_SQ != 0, Sk % BLOCK_SK != 0, small shapes, AH > 64 on Metal, etc.) appear intentionally addressed.

(The CRITICALs mentioned in the wave-4 context appear to have been cleared by prior fixes, consistent with the "post fix-round-4" note.)
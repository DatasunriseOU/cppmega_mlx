---
aspect: performance
provider: grok
model: gpt-5-5-pro
range: HEAD~1..HEAD
base_ref: 44f4f888bb8f22bc9526658ac87168ee11e677cb
head_ref: 2b9310ef73638a41b8de47566c41891b55218679
timestamp: 2026-05-07T02:21:50.228102+00:00
files: ['cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py', 'cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py']
---
Here is a focused performance review of the two attached files (`dsa_splitk_indexer_loss.py` and `fp8_vecmat_path_c.py`), with emphasis on **hot-path regressions or concerns** introduced relative to the referenced Triton baseline (and prior Path B hand-written MSL in the fp8 case). I prioritized quantifiable issues around memory pressure, redundant HBM traffic, register/shared allocation, loop structure, and Metal-specific constraints. Severity is tagged **P0** (breaks production perf or correctness under load), **P1** (noticeable regression vs. Triton/Path B on hot shapes), **P2** (missed opportunity or edge-case slowdown).

### DSA Split-K Indexer-Loss (`dsa_splitk_indexer_loss.py`)

**P1: Stage-2 register pressure on Metal remains high even after "Wave-2" hoists (lines  ~480-520, ~550-570).**  
The comment at lines 480-490 and the pre-load of `M_pre`/`D_pre` (AH × BLOCK_SQ fp32 fragments) + per-sk_tile per-head recompute of `h_scores` + four large BLOCK_SQ×BLOCK_SK fp32 fragments (`h_scores`, `softmax_attn`, `softmax_idx`, `kl_term`) still risks heavy register spilling on Apple Silicon. For the Metal override (BLOCK_SQ=BLOCK_SK=32, BLOCK_D=16), this is ~16 KB fragments alone before shared staging and the online-softmax accumulators. The docstring at top (~lines 80-100) already flags the 64 KB problem at 64×64; the 32×32 reduction helps but does not eliminate spilling when AH=128 (worst-case ~128×32×4 = 16 KB just for the pre-loads).  

Compared to the Triton reference (which used smaller per-lane fragments and better warp-level scheduling), this can cause 20-50%+ slowdown on M-series due to register pressure + spill/fill traffic. The TODO at line 510 ("hoist Q out of the sk_tile loop in stage 2 too") is the exact missing piece—Q loads still happen inside the nested `sk_tile → h → d_tile` loop (see lines 530-550), re-reading the same Q tiles many times from HBM. This is a clear regression vs. the Stage-1 hoist (lines 320-340, which is good).  

**Actionable:** Fully hoist Q per (sq_block, h) into a shared buffer (similar to Stage-1's `Q_full`) and restructure the loop order (outer h or outer sk_tile with per-h Q cache). This mirrors the "Wave-2 perf #5" comment style already used elsewhere and would cut redundant HBM reads by up to SK_TILES × AH factor on early sq_blocks. Quantified impact: potentially 1.5-2× on Metal for long SK / moderate AH.

**P1: Metal block constants still too aggressive for threadgroup memory budget in Stage-2 ( `_metal_block_overrides`, lines 110-140).**  
Even at 32×32×16 the four fp32 fragments + online softmax scratch + shared Q/K staging can push close to or over the 32 KB per-threadgroup limit when combined with TileLang's internal double-buffering/pipelining (requested via `num_stages=2`). The docstring acknowledges the issue but the chosen values (32/32/16) were tuned assuming "comfortably under"; real register allocation + compiler padding often exceeds this, causing spills or compile-time fallback to slower paths. CUDA defaults (128) are fine, but the Metal path is the new hot path for mlx/cppmega.

**P2 (minor regression risk):** Causal trim (`_active_sk_tiles`, lines 360-370 and 620-630) is good, but the computation of `_max_useful_sk` uses `T.min(_max_sq_in_block, ASq - 1)` inside the kernel. On boundary sq_blocks this is cheap, but if ASq % BLOCK_SQ is large and SK_TILES is huge, the pipelined loop still launches unnecessary iterations for early blocks. The trim helps but is not as aggressive as a host-side pre-computation of per-block active tiles (which Triton wrapper often does implicitly via grid sizing). Low impact unless ASq >> SK.

**P2: No autotune / shape-adaptive BLOCK_* on CUDA (lines 20-50, `_block_constants_for_target`).**  
Static 128×128×64 matches the Triton launch but TileLang's Metal path already special-cases; the CUDA path should consider a small lru_cache-driven sweep (or at least expose BLOCK_SK as tunable) for cases where SK is not nicely divisible or when AH is small. TileLang paper results show strong gains from automated tile inference; hard-coding here loses some of that vs. pure Triton autotune.

**No major O(n²) or allocation-in-loop issues found.** The index_mask scatter (lines 780-790) is host-side and only when `sparse_loss=True` (matching upstream); the empty() fallback for non-sparse is a nice zero-fill avoidance. Caching via `@lru_cache` on `_stage*` kernels is solid. The `struct.pack/unpack` for scale_bits is a reasonable hashable-key hack.

**Overall for DSA:** The Wave-2 hoists and trims are positive steps toward closing the gap with Triton, but **Stage-2 Q-hoist + tighter Metal register tuning** are the highest-ROI remaining items. Without them, Metal dispatch will show noticeable regression vs. the original Triton path on production DSA shapes (moderate AH, long SK).

### FP8 Vecmat (`fp8_vecmat_path_c.py`)

**P1: The packed `T.metal_fp8_e4m3_dot4` fast path (lines 280-310, `_uses_fp8_dot4_packed_macro`) is gated behind a runtime `K % 4 == 0` check that is already enforced upstream, but the scalar/vectorized fallback branches (lines 250-270 and 320-350) still exist and can be hit on mis-tuned shapes.**  
The canonical fast path relies on the packed macro + `simd_sum`; the vectorized_loads=True probe is explicitly documented as "does not reliably emit packed uint32 MSL loads" (lines 170-180). If anything perturbs the `_uses_fp8_dot4_packed_macro` decision (e.g., future changes to vec/K), you silently fall back to much slower scalar FP8 byte decodes or `tvm_thread_allreduce`. The `fp8_vecmat_msl_blockers` helper (lines 430+) is excellent for debugging, but production should fail-fast or log loudly when the fast path is missed (beyond the existing one-shot warnings).

**P2: Redundant shape/resolution work and potential small allocations in the hot dispatch path (`_normalize_vecmat_inputs`, `_resolve_vecmat_scale`, lines 650-700; `_fp8_vecmat_kernel_for` lru_cache).**  
`_resolve_vecmat_scale` does reshape + possible astype on every call (even for scalar inputs). For the very hot inference path (M=1 vecmat), this is minor but unnecessary. The lru_cache on `_fp8_vecmat_kernel_for` (maxsize=128) is good, but the key does not include `vectorized_loads` (only passed to make_ but not the cache decorator), so toggling the probe can cause cache thrashing or incorrect reuse. Also, the output_shape handling (lines 720-730) has a branch for packed vs. non-packed that forces a reshape on every call—cheap, but a small hot-path tax.

**P2: PassConfig filtering and simplify fallback (lines 50-100, 390-410) add small but repeated overhead.**  
The `_filter_supported_pass_configs` probe runs under lock on first use (good), but every kernel build still pays the `apply_simplify` try/except (which warns on failure). If simplify frequently fails on this kernel (as the warning path suggests it can), you pay the exception cost + slower IR on every dispatch. The Z3-related configs are nice but the "drop_provable_bound_checks" etc. are not guaranteed to fire on the M=1 vecmat shape; measure whether they actually reduce bound-check overhead in the generated MSL.

**No large synchronous payloads, N+1, or memory-growth issues.** The `_ensure_path_c_metal_fp8_intrinsics_registered` process-cache is a solid fix from prior waves. The canonical_vecmat_runtime_body helper is useful for comparison but not on hot path.

**Overall for FP8 vecmat:** The Path C implementation is close to the hand-written Path B fast path when the dot4 macro fires, but the fallback branches and small per-call overheads (scale resolution, reshape) introduce measurable regression on the absolute hottest inference shapes (large N, K%4==0). Prioritize making the packed path the *only* path (or raise hard on fallback) and cache more aggressively.

### Cross-File / Integration Notes (P2)
- Both files use TileLang's Metal lowering through SIMDgroup paths (good for Apple), but neither exposes easy benchmarking hooks against the original Triton/Path B (e.g., a side-by-side timer in the status functions). Adding a lightweight `benchmark_vs_triton` or `msl_features` delta would help catch regressions early.
- `tilelang_supports` / `can_run_metal` gating is present and correct; no blocking calls or async issues.
- Memory footprint comments in DSA (64 KB → 16 KB) and fp8 vecmat are accurate but conservative—real TileLang emission + register allocation often exceeds documented numbers on M-series; profile with Metal GPU counters.

**Highest priority fixes (P1):**  
1. Complete the Stage-2 Q hoist in `dsa_splitk_indexer_loss.py` (mirroring Stage-1).  
2. Tighten Metal block sizes further or add dynamic register-pressure-aware overrides.  
3. Harden the FP8 packed-dot4 path to be the only production path.

These changes should eliminate the remaining performance gap vs. the Triton reference on CUDA and the hand-written MSL on Metal. The Wave-2 improvements (hoists, trims, caching, warnings) are already moving in the right direction—great progress. Let me know if you want patches or targeted benchmarks for any of these.
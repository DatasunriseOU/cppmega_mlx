---
aspect: performance
provider: grok
model: gpt-5-5-pro
range: main..mlx-z3-wiring
base_ref: aef115bb0fa97b68b541580d957ff6b043e76176
head_ref: 9098d6ac43db0cc347996c6a0bda6eee34095d27
timestamp: 2026-05-07T02:48:04.835877+00:00
files: ['cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py', 'cppmega_mlx/nn/_tilelang/fp8_amax.py', 'cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py', 'cppmega_mlx/nn/_tilelang/topk_selector.py']
---
**No findings.**

The visible chunk (primarily the new `dsa_splitk_indexer_loss.py` plus the start of `fp8_amax.py`) shows careful attention to performance details for both CUDA and Metal targets. Key hot-path elements were addressed or mitigated:

- **M_pre/D_pre 32KB budget gate** in stage 2 (`make_dsa_splitk_stage2_kernel`, lines ~650-670): `USE_MD_PRE = _MD_PRE_BYTES <= _MD_PRE_BUDGET_BYTES` with fallback to per-iteration HBM reads when over budget. This directly prevents register spilling on Metal (and large CUDA blocks), a high-impact regression risk that was proactively gated.
- **Q hoist** in stage 1 (`make_dsa_splitk_stage1_kernel`, lines ~340-360) and partial hoist in stage 2 eliminate redundant HBM loads of Q tiles across the split-K dimension — good bandwidth win.
- **Causal trim** (`_active_sk_tiles` calculation with clamp-to-1) in both stages reduces useless iterations on early query blocks.
- **Topk bounds-check** and **NaN guard for fully-masked rows** in the Torch wrapper (`dsa_splitk_indexer_loss_tilelang`, lines ~1050-1080) are correctly implemented as defensive measures without introducing measurable hot-path cost (single `max`/`min` + `any` on a small tensor).
- No O(n²) loops, no N+1 patterns, no blocking calls in async paths, no large synchronous payloads, and no obvious allocation inside tight loops visible in this chunk.
- TileLang `lru_cache` on kernel construction, contiguous checks (with no-op fast path), and power-of-two bucketing (in the sibling `fp8_amax.py`) further limit JIT and memory pressure.

The changes in this chunk of Wave-2 post fix-round-2 introduce no new performance regressions. The added memory-budget logic and hoisting improvements are net positive for the Metal hot path. 

(Chunk 2 review will cover the remainder of the files.)
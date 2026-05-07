---
aspect: correctness
provider: grok
model: gpt-5-5-pro
range: main..mlx-z3-wiring
base_ref: aef115bb0fa97b68b541580d957ff6b043e76176
head_ref: 85470b2a15362a4232215e436e6be7a005991c62
timestamp: 2026-05-07T02:14:22.747731+00:00
files: ['cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py', 'cppmega_mlx/nn/_tilelang/fp8_amax.py', 'cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py', 'cppmega_mlx/nn/_tilelang/topk_selector.py']
---
**No findings.**

All changes in the diff introduce new files or targeted fixes that align with the described conservative Z3 wiring and prior fix-rounds.

The four touched files were reviewed line-by-line against the specified correctness criteria (off-by-one, error handling, swallowed exceptions, type mismatches, races, defaults, None handling, edge cases, regressions).

- **dsa_splitk_indexer_loss.py** (new, 959 lines): Comprehensive bounds guards (`sq_idx < ASq`, `sk_idx < Sk`, `d_idx < AD`, `valid = in_bounds and (sq_idx >= sk_idx)`), explicit positive-dim checks in builders, safe `-inf`/`0` sentinel handling for online softmax, predicate guards on `IndexMask`/`IndexScores` loads, Metal block-size overrides for 32 KB limit, empty tensor for non-sparse `index_mask` to avoid zero-fill cost, contiguous enforcement, and `K > 0` guard in related Path C (cross-referenced). No off-by-one in ceildiv-style `NUM_SQ_BLOCKS` / `SK_TILES`. No swallowed exceptions. No races (caches use `lru_cache` + `threading.Lock` per MLX wiring note).

- **fp8_amax.py** (new, 459 lines): Empty-tensor fast-path, contiguous reshape, `if gi < N` guards, `T.reduce_max` + `T.atomic_max` on pre-zeroed scalar, clamp via `T.max`/`T.min`, proper `inv_scale_buf` handling for scalar/tensor input. Matches Triton contracts. No regressions.

- **fp8_vecmat_path_c.py**: Only addition is `K <= 0` guard (lines 408-409 in new diff) + `threading.Lock` for pass-config cache (lines 64, 112-116). Directly addresses "mlx K>0 + threading.Lock for caches" fix-round. The `except (AttributeError, KeyError, TypeError)` narrows prior broad `except Exception` (improved, not a swallow regression). No other behavioral changes.

- **topk_selector.py**: Symmetric `threading.Lock` addition for its pass-config cache (lines 161, 209-213). Matches "threading.Lock on caches" MLX-side wiring. No other modifications.

All referenced fix-rounds are visibly applied:
- `K>0` guard present.
- Cache locks present.
- No new null/None issues, no broken defaults, no race conditions introduced.
- Edge cases (zero/empty tensors, boundary tiles, sparse/non-sparse, Metal vs CUDA block sizes, AH>128 rejection) explicitly handled.

The diff is clean from a correctness perspective. No bugs found.
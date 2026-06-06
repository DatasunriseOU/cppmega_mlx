# §METAL-RETILE — sub_chunks=2 L-retile for the batched B2 Metal GEMM prim

Status: **carry IMPLEMENTED + z3-PROVEN + Apple-GPU numerically VALIDATED (in-budget).
Prod L=64 via this retile is an honest 32 KB NO-GO (gate RAISES, no over-budget
launch).** HPC=2 dchunk parity bug **FIXED** (8.59e-3 -> 6.03e-6).

All Apple-GPU dispatches in this campaign went through
`/Users/dave/.local/bin/safe_metal_run.sh <timeout> <cmd>` (single-owner GPU mutex +
RSS>40GB SIGKILL + wall-clock timeout before the 90s SoC watchdog window). Bounded
shapes only (B=1, S<=64). Guard log clean: every run `END rc=0`, no TIMEOUT, no
RSS_CAP_HIT, no watchdog panic. Lock released after each run.

## What was delivered

### (1) sub_chunks=2 retile BODY — `main_retile` in `chunk_scan_combine_bwd_metal_gemm_prim_batched`
`cppmega_mlx/nn/_tilelang/mamba3_chunked_backward_core.py` L3530-3891. Splits the
length-L chunk into `sc0 = rows[0, L_sub)` and `sc1 = rows[L_sub, L)` (L_sub = L/2)
so the L-indexed simdgroup operands fit Apple's 32 KB pool. Inter-sub-chunk SSD
state carried per the z3-proven identities:
- **dchunk_states ADDITIVE**: one `dchunk_frag` accumulates both L_sub-row
  `transpose_A` GEMMs (sc0 then sc1); the sc1 partial is also drained to a resident
  `Psc1[P,N]` for the dinp cross carry.
- **dinp first-order cross carry**: for `s in sc0`, local sum over (l in sc0, l>=s)
  plus `Psc1[p,n] / exp2(dacs[s]*p)`; for `s in sc1` purely local.
- **dC_diag / dseg**: the single (L,L) DYX GEMM is restructured into three L_sub
  blocks — DYX00 (diag00), DYX11 (diag11), DYX10 (cross10) — 3 inlined L_sub GEMMs;
  dseg scatters over the same three blocks.

`sub_chunks==1` returns the legacy `main` body BYTE-IDENTICAL. `sub_chunks not in
{1,2}` RAISES. `L_sub % 8 != 0` RAISES. RULE #1: ONE path past the gates, no silent
fallback.

### (2) HPC=2 dchunk parity bug FIX — §BARRIER-FIX
Root cause: head-band staging-tile REUSE race. At `HEADS_PER_CTA>=2` the next head's
refill of the shared `dY16/opA/opB/store_f32/DYX` tiles could race the previous
head's frag->shared->global drain. Fix: per-head `T.sync_threads()` fences at the
start of each head iteration in the DYX phase (A) and the dchunk_states phase (D),
plus frag->store_f32 fences before the parallel/global reads. No-op at HPC=1.

## MEASURED (guarded, local Apple Metal, bounded B=1 S=64 G=2 H=4 P=N=32)

| run | shape | serial ms | batched/retile ms | speedup | parity worst | verdict |
|-----|-------|-----------|-------------------|---------|--------------|---------|
| HPC=1 (sub=1)        | L=P=N=32         | 10.83 | 2.20 | **4.93x** | 6.03e-06 | GO PASS |
| HPC=2 (sub=1) AFTER FIX | L=P=N=32      | 10.84 | 4.37 | **2.48x** | **6.03e-06** (dchunk 6.03e-06, was 8.59e-3) | GO PASS |
| retile sub=2 HPC=1   | L=32->L_sub=16,P=N=32 | 10.83 | 2.04 | **5.30x** | 6.03e-06 (dchunk 6.03e-06, dinp 5.21e-06) | GO PASS |

All grads `<1e-3` in every run. dx/dz exactly 0.

## MSL codegen (CPU lower, no dispatch)
retile body (sub=2, L=32, L_sub=16, P=N=32, HPC=1): MSL 44284 B,
`simdgroup_multiply_accumulate: 5`, `simdgroup_float8x8: 3`,
`make_filled_simdgroup_matrix: 44`. Legacy sub=1 (HPC=2): MSL 37982 B.

## z3 (non-vacuous) — `scratch/proof_b2_batched_driver.py`
VERDICT `ALL_POSITIVES_PROVED_AND_NON_VACUOUS`. 5 metal_subchunk positives UNSAT
(associativity, dchunk_additive, dinp_firstorder_carry, dcdiag_block_split,
dseg_block_partition — all L64 nsub2), each paired with a bugged-control negative
(dropped_carry / dropped_sc1 / dropped_cross / dropped_cross10 / nocross_misses)
that is sat/miss. dchunk head-band disjointness: the `single_writer` obligation on
the dchunk/dcoff/dcdiag contractions UNSAT; the interleaved-bands negative
sat/single_writer=False.

## HONEST NO-GO — prod L=64 via retile does NOT fit 32 KB
`_metal_subchunk_smem_bytes(L_sub=32, P=N=64)`:
- HPC=1: all_static **68096 B**, store_dynamic 51712 B
- HPC=2: all_static **99328 B**, store_dynamic 82944 B

Both far exceed Apple's **32768 B** pool — dominated by the full-L `dY_band`, the
fp32 `Psc1` resident, and the fp32 `store_f32`. Even moving `store_f32` dynamic
leaves the static residual > 32 KB. So **prod L=64 via this retile is an honest
32 KB NO-GO on Apple**. The build gate RAISES `NotImplementedError` end-to-end
(verified: `~68096 B ... >= Apple's 32768 B limit`) — it NEVER launches
over-budget (RULE #1).

The carry MECHANISM is nonetheless validated correct: z3 over the reals AND a clean
Apple-GPU numeric run at the in-budget L=32->L_sub=16 shape (5.30x, 6.03e-6). To
make prod L=64 actually fit would require sub_chunks=4 (L_sub=16) PLUS shrinking the
resident set (Psc1 fp16, store dynamic, dropping the full-L dY_band to per-sub-chunk
reload) — and the nsub=4 generalized lower-triangular block carry is NOT z3-proven
here, so the prim RAISES for sub_chunks not in {1,2} rather than emit an unproven
generalization.

## Bottom line
The 4.95x B2 batched Metal speedup is REAL and parity-clean at in-budget native
L=32 (P=N=32), now at BOTH HPC=1 (4.93x) and HPC=2 (2.48x, bug fixed). The
sub_chunks=2 retile carry is implemented, z3-proven, and Apple-GPU-validated at an
in-budget shape (5.30x). It does NOT unlock prod L=64 (P=N=64) — that overflows the
32 KB pool and the gate honestly RAISES.

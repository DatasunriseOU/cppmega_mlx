"""z3/TLA proof driver for the batched B2 GEMM rewrite (§TB1).

Discharges the 3 GEMM-able batched contractions (dchunk_states transpose_A +
decay-fold, dC_off dense, dC_diag lower-tri mask) at prod dims L=N=P=64,
HEADS_PER_CTA=4, AND proves non-vacuity with two negative controls:
  (1) overlap-band tiling (m_stride < tile_m) -> single_writer MUST be False
  (2) transpose-bugged operand map -> operand_maps_match MUST be False
RAISES (RULE #1) if any positive fails or any negative spuriously passes.
"""
import json

from cppmega_mlx.nn._tilelang import _gemm_rewrite_proof as grp

grp._msl_transform.ensure_libz3_preloaded()
import z3  # noqa: E402

L = N = P = 64
HPC = 4
out = {"positives": {}, "negatives": {}}

# ---------------- POSITIVE: the 3 batched GEMM-able contractions ------------
dchunk = grp.b2_batched_dchunk_contraction(z3, chunk_size=L, headdim=P, dstate=N)
t_dchunk = grp.b2_batched_tiling(tile_m=P, tile_n=N, tile_k=L, heads_per_cta=HPC)
p_dchunk = grp.require_gemm_rewrite_proof(dchunk, t_dchunk)

dcoff = grp.b2_batched_dcoff_contraction(z3, chunk_size=L, headdim=P, dstate=N)
t_dcoff = grp.b2_batched_tiling(tile_m=L, tile_n=N, tile_k=P, heads_per_cta=HPC)
p_dcoff = grp.require_gemm_rewrite_proof(dcoff, t_dcoff)

dcdiag = grp.b2_batched_dcdiag_contraction(z3, chunk_size=L, dstate=N)
t_dcdiag = grp.b2_batched_tiling(tile_m=L, tile_n=N, tile_k=L, heads_per_cta=HPC)
p_dcdiag = grp.require_gemm_rewrite_proof(dcdiag, t_dcdiag)

for nm, pr in (("dchunk", p_dchunk), ("dcoff", p_dcoff), ("dcdiag", p_dcdiag)):
    out["positives"][nm] = {
        "z3_used": pr.z3_used,
        "z3_proved": pr.z3_proved,
        "operand_maps_match": pr.operand_maps_match,
        "mask_equiv": pr.mask_equiv,
        "scale_equiv": pr.scale_equiv,
        "single_writer": pr.single_writer,
        "k_covered": pr.k_covered,
    }
    assert pr.z3_proved, f"POSITIVE {nm} not proved: {pr.reason}"
    assert pr.single_writer, f"POSITIVE {nm} single_writer False: {pr.reason}"

# ---------------- NEGATIVE 1: overlapping head-bands (race) -----------------
# m_stride < tile_m => band b and b+1 share rows => single_writer MUST be False.
t_overlap = grp.GemmTiling(
    tile_m=P, tile_n=N, tile_k=L, m_blocks=HPC, n_blocks=1, k_steps=1,
    m_stride=P // 2, n_stride=N,
)
p_overlap = grp.prove_gemm_rewrite(dchunk, t_overlap)
out["negatives"]["overlap_band"] = {
    "z3_proved": p_overlap.z3_proved,
    "single_writer": p_overlap.single_writer,
    "reason": p_overlap.reason[:160],
}
assert not p_overlap.single_writer, "NEG overlap_band spuriously single_writer=True (VACUOUS)"
assert not p_overlap.z3_proved, "NEG overlap_band spuriously z3_proved=True (VACUOUS)"

# ---------------- NEGATIVE 2: transpose-bugged operand map ------------------
# Inject a transpose bug: make the GEMM A-address disagree with the serial
# A-address (gemm reads k*M+i where serial reads i*K+k). operand_maps_match
# MUST be False (the rewrite would compute the wrong thing).
src = grp.b2_batched_dchunk_contraction(z3, chunk_size=L, headdim=P, dstate=N)
M_, K_ = src.m_extent, src.k_extent
# Real dchunk a_addr_gemm = k*headdim+i (transpose_A). Inject the WRONG
# (un-transposed) layout i*K+k, which differs whenever i != k -> bug.
def bugged_a_gemm(i, k):
    return i * K_ + k
bad = grp.GemmContraction(
    name=src.name + "_transpose_bug",
    m_extent=src.m_extent, n_extent=src.n_extent, k_extent=src.k_extent,
    a_addr_serial=src.a_addr_serial,
    a_addr_gemm=bugged_a_gemm,
    b_addr_serial=src.b_addr_serial,
    b_addr_gemm=src.b_addr_gemm,
    mask_serial=src.mask_serial,
    mask_gemm=src.mask_gemm,
    scale_serial=src.scale_serial,
    scale_gemm=src.scale_gemm,
)
p_bad = grp.prove_gemm_rewrite(bad, t_dchunk)
out["negatives"]["transpose_bug"] = {
    "z3_proved": p_bad.z3_proved,
    "operand_maps_match": p_bad.operand_maps_match,
    "reason": p_bad.reason[:160],
}
assert not p_bad.operand_maps_match, "NEG transpose_bug spuriously operand_maps_match=True (VACUOUS)"
assert not p_bad.z3_proved, "NEG transpose_bug spuriously z3_proved=True (VACUOUS)"

out["VERDICT"] = "ALL_POSITIVES_PROVED_AND_NON_VACUOUS"
print("PROOF_RESULT_JSON_BEGIN")
print(json.dumps(out, indent=2))
print("PROOF_RESULT_JSON_END")

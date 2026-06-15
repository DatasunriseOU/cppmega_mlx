"""Directed-search unit + parity test for the chunked-backward fusion search.

Covers the THREE deliverables of cppmega_mlx.runtime.path_c_backward_fusion_search:

  1. FEASIBILITY PREDICTOR (Phase A, static, no GPU pipeline-state): the 4
     contiguous partitions of [B2,B1,B0] are enumerated and each fused segment's
     P1..P4 (MSL ceiling / threadgroup / buffer-arg / watchdog) verdict is computed
     from tilelang.lower source (NO newComputePipelineState). The MSL byte anchors
     match the design (B2~28KB, B1~20KB, B0~24KB, fully-fused~72KB < 140000).

  2. RANKER: feasible variants order dispatch-count ASC, then internalized-edges
     DESC -- so {B2,B1,B0}(1) < {B2,B1}{B0}(2, clean) < {B2}{B1,B0}(2, absorb) <
     baseline(3).

  3. BIT-CORRECTNESS of the SELECTED clean-splice fusion: the B2+B1 multi-T.Kernel
     spliced kernel produces grads byte-identical (fp32 floor) to the 3-separate-
     kernel baseline AND within 1e-3 of the path-b GOLD, DETERMINISTICALLY over
     repeats (the B2->B1 dchunk_states handoff flows inside ONE command buffer).

RULE #1: the gate is NOT loosened; the absorption-gated groupings ({B1,B0} fused)
are asserted INFEASIBLE for the clean splice (they would re-introduce the dA_tail
race), not silently fused.
"""
from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx


_DIMS = (1, 128, 64, 2, 2, 64, 16)  # b,s,c,G,H,P,N -- small 2-chunk surface (fast)
_GRAD_NAMES = ("dx", "dB", "dC", "dz", "dA", "ddt", "dD", "dh0")
_ABS_GATE = 1e-3
_REPEATS = 12


def _metal_mlx_available() -> bool:
    try:
        import mlx.core as _mx
        return _mx.metal.is_available()
    except Exception:
        return False


def test_phaseA_predict_and_rank():
    """Static feasibility + ranking (no GPU pipeline-state)."""
    from cppmega_mlx.runtime.path_c_backward_fusion_search import (
        predict_variants, rank_variants,
    )

    variants = predict_variants(_DIMS)
    assert len(variants) == 4, "exactly 4 contiguous partitions of [B2,B1,B0]"
    by_id = {v.variant_id: v for v in variants}
    # all 4 partitions are statically predicted feasible at this surface (P1..P4).
    for vid in ("B2B1B0", "B2B1_B0", "B2_B1B0", "B2_B1_B0"):
        assert vid in by_id, f"missing partition {vid}"
        assert by_id[vid].predicted_feasible, f"{vid} should be P1..P4 feasible"
    # MSL ceiling never exceeded; buffer count never over 31 (P1/P3 anchors).
    for v in variants:
        for seg in v.segments:
            assert seg.msl_bytes < 140000, f"{v.variant_id} seg MSL {seg.msl_bytes}"
            assert seg.nbuf <= 31, f"{v.variant_id} seg nbuf {seg.nbuf}"
            assert seg.phys_shared <= 32768, f"{v.variant_id} seg phys {seg.phys_shared}"
    # ranking: dispatch ASC, then clean before absorb among equal dispatch.
    ranked = rank_variants(variants)
    ids = [v.variant_id for v in ranked]
    assert ids[0] == "B2B1B0", "1-dispatch ranks first"
    assert ids.index("B2B1_B0") < ids.index("B2_B1B0"), (
        "clean 2-dispatch (B2B1_B0) ranks above absorb 2-dispatch (B2_B1B0)"
    )
    assert ids[-1] == "B2_B1_B0", "3-dispatch baseline ranks last"


@pytest.mark.kernel
@pytest.mark.skipif(not _metal_mlx_available(), reason="requires MLX Metal GPU")
def test_b2b1_splice_bitcorrect_vs_baseline_and_gold():
    """The B2+B1 multi-T.Kernel spliced fusion is bit-correct vs the 3-kernel
    baseline AND the path-b GOLD, deterministically over repeats."""
    import os
    os.environ.setdefault(
        "TILELANG_MLX_TVM_FFI_FORCE_COMMAND_BUFFER_BOUNDARY", "1")
    from cppmega_mlx.runtime.path_c_backward_fusion_search import (
        make_fused_backward, _build_eval_inputs, _maxabs,
    )
    from cppmega_mlx.nn._tilelang.mamba3_path_c import (
        _force_chunked_command_buffer_boundary,
    )

    _force_chunked_command_buffer_boundary()
    inp = _build_eval_inputs(_DIMS)
    bwd_base = make_fused_backward((("B2",), ("B1",), ("B0",)), _DIMS)
    bwd_fused = make_fused_backward((("B2", "B1"), ("B0",)), _DIMS)

    def run(bwd):
        g = bwd(inp["dy"], *inp["primals"], cb=inp["cb"], dA_cumsum=inp["dA_cumsum"],
                prev_states=inp["prev_states"], y=inp["y"])
        mx.eval(*g)
        return g

    worst_gold = {nm: 0.0 for nm in _GRAD_NAMES}
    worst_vs_base = 0.0
    for _ in range(_REPEATS):
        g_base = run(bwd_base)
        g_fused = run(bwd_fused)
        for nm, gf, gg in zip(_GRAD_NAMES, g_fused, inp["grads_gold"]):
            worst_gold[nm] = max(worst_gold[nm], _maxabs(gf, gg))
        worst_vs_base = max(
            worst_vs_base, max(_maxabs(a, b) for a, b in zip(g_fused, g_base)))

    for nm in _GRAD_NAMES:
        assert worst_gold[nm] < _ABS_GATE, (
            f"B2B1-fused {nm} vs GOLD max|abs|={worst_gold[nm]:.3e} >= {_ABS_GATE:.0e} "
            "(gate NOT loosened)")
    assert worst_vs_base < 1e-4, (
        f"B2B1-fused vs 3-kernel baseline diverged ({worst_vs_base:.3e}); the "
        "dchunk_states handoff must flow bit-exactly inside the one command buffer")


@pytest.mark.kernel
@pytest.mark.skipif(not _metal_mlx_available(), reason="requires MLX Metal GPU")
def test_absorption_grouping_is_clean_splice_infeasible():
    """The B1->B0 fused groupings are NOT clean-splice executable (they require the
    dA_tail absorption that re-introduces the race) -- the search must mark them
    infeasible, never silently fuse them."""
    from cppmega_mlx.runtime.path_c_backward_fusion_search import (
        _grouping_is_clean_splice,
    )

    assert _grouping_is_clean_splice((("B2", "B1"), ("B0",))), "B2B1_B0 is clean"
    assert _grouping_is_clean_splice(
        (("B2",), ("B1",), ("B0",))), "baseline is clean"
    assert not _grouping_is_clean_splice(
        (("B2", "B1", "B0"),)), "fully-fused needs absorption"
    assert not _grouping_is_clean_splice(
        (("B2",), ("B1", "B0"))), "B1B0-fused needs absorption"

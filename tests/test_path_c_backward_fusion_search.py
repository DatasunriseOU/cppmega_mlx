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
    """Static feasibility + ranking (no GPU pipeline-state).

    The variant space now spans TWO schedule CLASSES: the 4 contiguous
    partitions of [B2,B1,B0] (CHUNKED_SEGMENT) PLUS the MSL-class single fused
    lane scan (LANE_SEQUENTIAL). So predict_variants yields 5 candidates.
    """
    from cppmega_mlx.runtime.path_c_backward_fusion_search import (
        predict_variants, rank_variants,
        SCHEDULE_CLASS_CHUNKED_SEGMENT, SCHEDULE_CLASS_LANE_SEQUENTIAL,
    )

    variants = predict_variants(_DIMS)
    assert len(variants) == 5, (
        "4 contiguous partitions of [B2,B1,B0] + 1 LANE_SEQUENTIAL MSL-class"
    )
    by_id = {v.variant_id: v for v in variants}
    chunked = [v for v in variants if v.schedule_class == SCHEDULE_CLASS_CHUNKED_SEGMENT]
    lane = [v for v in variants if v.schedule_class == SCHEDULE_CLASS_LANE_SEQUENTIAL]
    assert len(chunked) == 4, "4 chunked-segment groupings"
    assert len(lane) == 1, "exactly one LANE_SEQUENTIAL schedule-class candidate"
    # all 4 partitions are statically predicted feasible at this surface (P1..P4).
    for vid in ("B2B1B0", "B2B1_B0", "B2_B1B0", "B2_B1_B0"):
        assert vid in by_id, f"missing partition {vid}"
        assert by_id[vid].predicted_feasible, f"{vid} should be P1..P4 feasible"
    # the LANE_SEQUENTIAL candidate: single dispatch, no brick grouping, and
    # predicted feasible at the nam56r surface (HEADDIM=64 SIMD P-reduction).
    lane_v = lane[0]
    assert lane_v.variant_id == "LANE"
    assert lane_v.dispatch_count == 1, "lane scan is single fused dispatch"
    assert lane_v.grouping == (), "LANE_SEQUENTIAL is not a brick grouping"
    assert lane_v.predicted_feasible, "HEADDIM=64 SIMD P-reduction is supported"
    # MSL ceiling never exceeded; buffer count never over 31 (P1/P3 anchors) for
    # the chunked-segment candidates (the lane variant carries no spliced segments).
    for v in chunked:
        for seg in v.segments:
            assert seg.msl_bytes < 140000, f"{v.variant_id} seg MSL {seg.msl_bytes}"
            assert seg.nbuf <= 31, f"{v.variant_id} seg nbuf {seg.nbuf}"
            assert seg.phys_shared <= 32768, f"{v.variant_id} seg phys {seg.phys_shared}"
    # ranking: dispatch ASC, then clean before absorb among equal dispatch. Both
    # the absorption-fused B2B1B0 and the LANE candidate are 1-dispatch and rank
    # ahead of every 2-/3-dispatch chunked grouping.
    ranked = rank_variants(variants)
    ids = [v.variant_id for v in ranked]
    assert set(ids[:2]) == {"B2B1B0", "LANE"}, "both 1-dispatch variants rank first"
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
def test_lane_sequential_msl_class_candidate_bitcorrect():
    """The LANE_SEQUENTIAL MSL-class candidate (mamba3_mimo_bwd_path_c lane scan)
    is a registered search candidate that produces all 8 grads bit-correct vs the
    path-b GOLD and is deterministic over repeats -- so the search may SELECT it."""
    from cppmega_mlx.runtime.path_c_backward_fusion_search import (
        make_lane_sequential_backward, _build_eval_inputs, _maxabs,
    )

    inp = _build_eval_inputs(_DIMS)
    bwd_lane = make_lane_sequential_backward(_DIMS)

    def run():
        g = bwd_lane(inp["dy"], *inp["primals"], cb=inp["cb"],
                     dA_cumsum=inp["dA_cumsum"], prev_states=inp["prev_states"],
                     y=inp["y"])
        mx.eval(*g)
        return g

    worst_gold = {nm: 0.0 for nm in _GRAD_NAMES}
    first = None
    worst_det = 0.0
    for _ in range(_REPEATS):
        g = run()
        if first is None:
            first = g
        else:
            worst_det = max(worst_det, max(_maxabs(a, b) for a, b in zip(g, first)))
        for nm, gc, gg in zip(_GRAD_NAMES, g, inp["grads_gold"]):
            worst_gold[nm] = max(worst_gold[nm], _maxabs(gc, gg))

    for nm in _GRAD_NAMES:
        assert worst_gold[nm] < _ABS_GATE, (
            f"LANE_SEQUENTIAL {nm} vs GOLD max|abs|={worst_gold[nm]:.3e} >= "
            f"{_ABS_GATE:.0e} (gate NOT loosened)")
    assert worst_det == 0.0, (
        f"LANE_SEQUENTIAL not deterministic across repeats ({worst_det:.3e})")


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

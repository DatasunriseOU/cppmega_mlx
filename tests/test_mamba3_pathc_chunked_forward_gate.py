"""Gate + descriptor tests for the Path-C mamba3 chunked-forward swap.

Covers the emitter-side wiring of the PROVEN chunked-parallel forward scan-core
(``cppmega_mlx.nn._tilelang.mamba3_chunked_scan_core``) into the live forward
emitter (``_append_row_phased_mamba3_body``):

  1. ``mamba3_chunked_forward_scan_grid`` returns the production
     ``local_gb10_quarter`` grid (S=4096, chunk=64, heads=112, head_dim=64,
     state_dim=64, groups=8) = ``(112, 4, 64)`` -> 28672 threadgroups vs the
     serial forward's 1.
  2. The RAISING gate surfaces every infeasible shape as
     ``PathCSplitInfeasible`` (RULE #1: no silent serial fallback).
  3. The non-raising classifier ``_mamba3_chunked_forward_scan_feasibility``
     reports infeasibility WITHOUT aborting (so small/non-tile test shapes keep
     emitting the serial scan, which is still the only compute path today).
  4. The emitted forward source records the chunked-grid descriptor (the live,
     asserted dispatch parameters) right at the scan-core integration site.
"""

from __future__ import annotations

import pytest

from cppmega_mlx.runtime.path_c_fusion_schedules import (
    MAMBA3_CHUNKED_FWD_SCAN_CHUNK_SIZE,
    PathCSplitInfeasible,
    _mamba3_chunked_forward_scan_feasibility,
    build_mamba3_fp8_train_acceptance_fixture_region,
    mamba3_chunked_forward_scan_block_dstate,
    mamba3_chunked_forward_scan_grid,
    mamba3_fp8_train_fusion_schedule_template,
)

# Production local_gb10_quarter mamba3 forward dims (hidden=3584, expand=2 ->
# inner=7168; head_dim=64 -> nheads=112; state_dim=64; groups=8; chunk=64;
# max_seq=4096). Matches the dims compiled+run on Metal in
# scratch/run_chunk_scan_fwd_metal_prod.py.
_PROD = dict(
    sequence_length=4096,
    batch=1,
    heads=112,
    head_dim=64,
    state_dim=64,
    groups=8,
)


def test_production_chunked_forward_grid_is_28672_threadgroups():
    total, grid = mamba3_chunked_forward_scan_grid(region_name="prod", **_PROD)
    assert grid == (112, 4, 64)
    assert total == 28672
    # The whole point of the swap: many threadgroups vs the serial forward's 1.
    assert total > 1
    assert MAMBA3_CHUNKED_FWD_SCAN_CHUNK_SIZE == 64


@pytest.mark.parametrize(
    "override,needle",
    [
        (dict(sequence_length=4095), "sequence_length not divisible"),
        (dict(head_dim=63), "head_dim not divisible"),
        (dict(groups=5), "heads not divisible by groups"),
    ],
)
def test_chunked_forward_gate_raises_on_infeasible_shape(override, needle):
    """RULE #1: infeasible shapes RAISE PathCSplitInfeasible (no fallback)."""
    kwargs = dict(_PROD)
    kwargs.update(override)
    with pytest.raises(PathCSplitInfeasible, match=needle):
        mamba3_chunked_forward_scan_grid(region_name="r", **kwargs)


def test_chunked_forward_gate_raises_on_chunk_size_mismatch():
    with pytest.raises(PathCSplitInfeasible, match="chunk_size != validated"):
        mamba3_chunked_forward_scan_grid(region_name="r", chunk_size=128, **_PROD)


def test_classifier_is_non_raising_for_small_test_shapes():
    """Non-tile-aligned test shapes classify NOT-feasible without aborting."""
    feasible, total, grid, reason, est, limit = (
        _mamba3_chunked_forward_scan_feasibility(
            sequence_length=4096,
            batch=1,
            heads=4,
            head_dim=8,  # < block_N=16 -> not feasible
            state_dim=8,
            groups=1,
        )
    )
    assert feasible is False
    assert grid is None
    assert "head_dim not divisible" in reason


def test_classifier_agrees_with_gate_on_production_shape():
    feasible, total, grid, *_ = _mamba3_chunked_forward_scan_feasibility(**_PROD)
    assert feasible is True
    assert (total, grid) == mamba3_chunked_forward_scan_grid(
        region_name="r", **_PROD
    )


def test_feasibility_and_delegation_share_the_metal_state_tile():
    assert mamba3_chunked_forward_scan_block_dstate(
        state_dim=128, target="metal"
    ) == 64


def test_emitted_forward_records_chunked_grid_descriptor():
    """The live emitter records the chunked-scan-core dispatch descriptor."""
    prim_func = mamba3_fp8_train_fusion_schedule_template(
        build_mamba3_fp8_train_acceptance_fixture_region(include_backward=True)
    )
    src = prim_func._cppmega_path_c_generated_source
    assert "# mamba3_chunked_forward_scan: grid=" in src
    assert "threadgroups=" in src
    # The serial scan must still be present (the full kernel-split swap is the
    # documented remaining work); the descriptor sits beside it, never silently
    # replacing it.
    assert "mamba3_scan_policy: external_state_recurrence" in src
    # Descriptor must be emitted immediately before the serial scan-policy line.
    lines = src.splitlines()
    desc_idx = next(
        i
        for i, line in enumerate(lines)
        if "mamba3_chunked_forward_scan: grid=" in line
    )
    scan_idx = next(
        i
        for i, line in enumerate(lines)
        if "mamba3_scan_policy: external_state_recurrence" in line
    )
    assert desc_idx < scan_idx

"""Steps 4-6 caps-wiring unit tests (pool pass, recurrent classification).

These exercise the device-caps-derived shared-scratch pooling and the structural
recurrent classification WITHOUT compiling full model kernels, so they run fast
on any backend.
"""

from __future__ import annotations

import pytest

from cppmega_mlx.runtime import path_c_fusion_schedules as sched
from cppmega_mlx.runtime.path_c_device_caps import device_caps
from cppmega_mlx.runtime.path_c_fusion import _path_c_default_target


def test_metal_pool_pass_uses_caps_not_hardcoded_constants():
    if _path_c_default_target() != "metal":
        pytest.skip("metal-only pool pass")
    import inspect

    src = inspect.getsource(sched._pool_oversized_shared_scratch_to_metal_workspace)
    # The hardcoded literal trigger / demote-target comparison must be gone (the
    # constant names may survive in an explanatory comment, but never as a live
    # ``<= _METAL_SHARED_SCRATCH_*`` comparison driving the pool decision).
    assert "<= _METAL_SHARED_SCRATCH_TRIGGER_BYTES" not in src
    assert "<= _METAL_SHARED_SCRATCH_DEMOTE_TARGET_BYTES" not in src
    assert "device_caps()" in src
    assert "logical_to_physical_shared_margin" in src


def test_metal_pool_pass_pools_oversized_and_leaves_small_untouched():
    if _path_c_default_target() != "metal":
        pytest.skip("metal-only pool pass")
    big = 524288  # 2 MiB float32
    oversized = (
        f'    _bwd_mamba3_h_next = T.alloc_shared(({big},), "float32")\n'
        f'    _bwd_mamba3_h_prev = T.alloc_shared(({big},), "float32")\n'
        '    small_a = T.alloc_shared((100,), "float32")\n'
    )
    pooled = sched._pool_oversized_shared_scratch_to_metal_workspace(oversized)
    assert "T.alloc_global(" in pooled  # oversized buffers demoted to a pool
    # A name-gated kernel whose shared already fits the cap is untouched.
    small = (
        '    _bwd_mamba3_h_next = T.alloc_shared((100,), "float32")\n'
        '    small_a = T.alloc_shared((100,), "float32")\n'
    )
    assert "T.alloc_global(" not in (
        sched._pool_oversized_shared_scratch_to_metal_workspace(small)
    )


def test_path_c_split_infeasible_message_has_where_and_what():
    exc = sched.PathCSplitInfeasible(
        "region_chain_7_8", "watchdog", 12.0, 2.5, op_name="sparse_mla_fp8_apply_bwd"
    )
    msg = str(exc)
    assert "region_chain_7_8" in msg
    assert "sparse_mla_fp8_apply_bwd" in msg
    assert "watchdog" in msg
    assert "12.0" in msg and "2.5" in msg
    assert exc.region_name == "region_chain_7_8"
    assert exc.characteristic == "watchdog"

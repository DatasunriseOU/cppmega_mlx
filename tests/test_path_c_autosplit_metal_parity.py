"""Metal golden-parity gate (design §7.1): the caps-derived auto-split must
reproduce the CURRENT hand-tuned local_gb10_quarter splits byte-for-byte.

The reference is main 9f74055's segment grouping + dispatch modes + chunk params.
This test plans the real local_gb10_quarter route region on Metal with the
auto-split enabled (default sentinels resolve from device_caps()) and asserts the
exact 12-segment plan.

CUDA monolithic-parity (§7.2) is asserted at the resolution level (caps fields)
so it can run without a CUDA device; the full CUDA plan is checked on gb10.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from cppmega_mlx.runtime import path_c_fusion_schedules as sched
from cppmega_mlx.runtime.path_c_device_caps import device_caps
from cppmega_mlx.runtime.path_c_fusion import _path_c_default_target

_SCRIPTS = str(Path(__file__).resolve().parents[1] / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)


# The hand-tuned reference (main 9f74055), in plan order.
# (ops, execution_phase, row_dispatch_mode, max_rows_per_launch, buffer_count)
_GOLDEN_SEGMENTS = [
    (("entry_rmsnorm", "mamba3_mimo"), "forward", "grid_chunks", 64, 23),
    (("residual_rmsnorm", "m2rnn"), "forward", "grid_chunks", 64, 20),
    (("residual_rmsnorm",), "forward", "grid_chunks", 64, 6),
    (("attention_qkv_projection",), "forward", "launcher_chunks", 64, 9),
    (("sparse_mla_fp8_apply",), "forward", "launcher_chunks", 64, 12),
    (("sparse_mla_fp8_apply_bwd",), "backward", "launcher_chunks", 64, 15),
    (("attention_qkv_projection_bwd",), "backward", "launcher_chunks", 64, 12),
    (("residual_rmsnorm_bwd",), "backward", "grid_chunks", 64, 9),
    (("m2rnn_bwd",), "backward", "launcher_chunks", 64, 25),
    (("residual_rmsnorm_bwd",), "backward", "grid_chunks", 64, 9),
    (("mamba3_mimo_bwd",), "backward", "launcher_chunks", 64, 30),
    (("entry_rmsnorm_bwd",), "backward", "grid_chunks", 64, 6),
]


def _plan_local_gb10_quarter_chain():
    import m04_train_step as m

    profile, syms, regions = m._local_gb10_path_c_model_regions()
    sel = m._select_path_c_model_route_region(regions)
    scheduled = m.plan_path_c_fusion_schedule_for_region(sel, include_backward=True)
    return sched.plan_path_c_direct_fusion_chain_for_region(
        scheduled.region, include_backward=True
    )


@pytest.mark.slow
def test_metal_autosplit_reproduces_hand_tuned_splits():
    if _path_c_default_target() != "metal":
        pytest.skip("Metal golden parity (run on M4 Max)")
    chain = _plan_local_gb10_quarter_chain()
    assert chain.status == "ready", chain.reason
    assert chain.max_kernel_buffers == 31  # caps.buffer_arg_limit on Metal

    actual = []
    for seg in chain.segments:
        tgt = seg.schedule_target
        actual.append(
            (
                tuple(n.op_name for n in seg.region.nodes),
                seg.execution_phase,
                getattr(tgt, "row_dispatch_mode", None),
                getattr(tgt, "max_rows_per_launch", None),
                seg.kernel_parameter_count,
            )
        )

    assert len(actual) == len(_GOLDEN_SEGMENTS), (
        f"segment count {len(actual)} != hand-tuned {len(_GOLDEN_SEGMENTS)}"
    )
    for i, (got, want) in enumerate(zip(actual, _GOLDEN_SEGMENTS)):
        assert got == want, f"seg{i}: auto-split {got} != hand-tuned {want}"


def test_metal_caps_drive_the_split_decisions():
    """The split-driving caps fields equal the hand-tuned values on Metal."""
    if _path_c_default_target() != "metal":
        pytest.skip("metal-only")
    c = device_caps()
    assert c.forward_max_segment_nodes == 2  # MTLCompilerService pipeline band
    assert c.backward_max_segment_nodes == 1  # watchdog per-op isolation
    assert c.has_command_buffer_watchdog is True
    assert c.msl_pipeline_state_ceiling_bytes == 140000
    assert c.buffer_arg_limit == 31


def test_cuda_caps_resolve_to_monolithic():
    """§7.2: CUDA caps -> no segment-node split, no row/time chunking, only the
    queried shared-mem demote. Asserted at the caps-resolution level so it runs
    without a CUDA device; the full plan is exercised on gb10."""
    from cppmega_mlx.runtime.path_c_device_presets import preset_for_identity

    gb10 = preset_for_identity(
        backend="cuda", architecture="sm_121", device_name="NVIDIA GB10"
    )
    assert gb10 is not None
    # No op-count caps -> greedy monolithic fusion (no segment split).
    assert gb10.forward_max_segment_nodes is None
    assert gb10.backward_max_segment_nodes is None
    # No watchdog -> no row/time chunking path fires.
    assert gb10.has_command_buffer_watchdog is False
    assert gb10.watchdog_window_s is None
    # No MSL pipeline-state ceiling (ptxas fails loud).
    assert gb10.compiler_shader_ceiling_bytes is None
    # If we ARE on a CUDA host, assert the live caps too.
    if _path_c_default_target() == "cuda":
        c = device_caps()
        assert c.forward_max_segment_nodes is None
        assert c.backward_max_segment_nodes is None
        assert c.has_command_buffer_watchdog is False
        assert c.msl_pipeline_state_ceiling_bytes is None

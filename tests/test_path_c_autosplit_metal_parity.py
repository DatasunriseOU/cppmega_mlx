"""Metal golden-parity gate (design §7.1): the caps-derived auto-split must
reproduce the CURRENT hand-tuned local_gb10_quarter splits byte-for-byte.

The reference is main 82b6ab0's segment grouping + dispatch modes + chunk params
(the auto-split was originally calibrated against 9f74055; main has since advanced
with the mamba3 per-op time-chunk window=2 (b18f415), the backward atomic_add ->
non-atomic RMW (20860b7), and the CUDA monolithic gate (930748a) -- all of which
the auto-split reproduces, the window via _mamba3_bwd_rows_per_kernel_launch_for_nodes
and CUDA-monolithic via caps.has_command_buffer_watchdog==False).
This test plans the real local_gb10_quarter route region on Metal with the
auto-split enabled (default sentinels resolve from device_caps()) and asserts the
exact 12-segment plan AND the per-segment time-chunk window.

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


# The hand-tuned reference (main 82b6ab0), in plan order.
# (ops, execution_phase, row_dispatch_mode, max_rows_per_launch, buffer_count,
#  rows_per_kernel_launch). The last field is the per-launch TIME-CHUNK WINDOW:
# main 82b6ab0 added a smaller window (=2) for mamba3_mimo_bwd (b18f415) because
# its pooled global reverse-scan state makes the shared 8-step command buffer trip
# the macOS GPU watchdog; every other launcher-chunked backward op keeps the
# shared default of 8. The auto-split reproduces this exactly via
# _mamba3_bwd_rows_per_kernel_launch_for_nodes.
_GOLDEN_SEGMENTS = [
    (("entry_rmsnorm", "mamba3_mimo"), "forward", "grid_chunks", 64, 23, 8),
    (("residual_rmsnorm", "m2rnn"), "forward", "grid_chunks", 64, 20, 8),
    (("residual_rmsnorm",), "forward", "grid_chunks", 64, 6, 8),
    (("attention_qkv_projection",), "forward", "launcher_chunks", 64, 9, 8),
    (("sparse_mla_fp8_apply",), "forward", "launcher_chunks", 64, 12, 8),
    (("sparse_mla_fp8_apply_bwd",), "backward", "launcher_chunks", 64, 15, 8),
    (("attention_qkv_projection_bwd",), "backward", "launcher_chunks", 64, 12, 8),
    (("residual_rmsnorm_bwd",), "backward", "grid_chunks", 64, 9, 8),
    (("m2rnn_bwd",), "backward", "launcher_chunks", 64, 25, 8),
    (("residual_rmsnorm_bwd",), "backward", "grid_chunks", 64, 9, 8),
    (("mamba3_mimo_bwd",), "backward", "launcher_chunks", 64, 30, 2),
    (("entry_rmsnorm_bwd",), "backward", "grid_chunks", 64, 6, 8),
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
        # The per-launch time-chunk window lives only on the generated PrimFunc
        # attrs (it is not recorded on the schedule_target), so read it back from
        # the compiled template to assert the mamba3 window=2 (b18f415).
        prim_func = tgt.schedule_template(seg.region)
        rows_per_kernel_launch = getattr(
            prim_func, "_cppmega_path_c_rows_per_kernel_launch", None
        )
        actual.append(
            (
                tuple(n.op_name for n in seg.region.nodes),
                seg.execution_phase,
                getattr(tgt, "row_dispatch_mode", None),
                getattr(tgt, "max_rows_per_launch", None),
                seg.kernel_parameter_count,
                rows_per_kernel_launch,
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


def test_metal_single_launcher_fullgraph_not_pooled_at_short_sequence():
    """§7.1 golden gate: the single-launcher fullgraph train block must stay a
    single fused kernel (NOT pooled) at a short sequence on Metal.

    Regression guard: the caps-derived shared-scratch pool trigger must remain the
    threadgroup cap (physical-overflow point), NOT ``cap / margin``. When step 5
    used ``cap / margin`` (8856 on M4 Max) as the trigger, the ~20 KiB logical
    single-launcher fullgraph kernel was over-pooled: its internal fusion-edge
    buffer ``entry_rmsnorm_hidden`` was demoted from ``alloc_shared`` to the
    ``alloc_global`` pool, stripping its load/store out of the entry PrimFunc, so
    tilelang's fullgraph validator rejected the missing edge and the auto-split
    SILENTLY fell back to the staged launcher (a RULE #1 violation). main 82b6ab0
    single-fuses this shape; this gate keeps that behaviour byte-for-byte.
    """
    if _path_c_default_target() != "metal":
        pytest.skip("Metal single-launcher parity (run on M4 Max)")

    import m04_train_step as m
    from cppmega_mlx.recipes.model_factory import (
        build_local_gb10_quarter_tiny_smoke_model,
    )

    model = build_local_gb10_quarter_tiny_smoke_model()
    # Short sequence (the train-step path subtracts the shift, giving 15 here) is
    # exactly where the over-pool regression bit: the fullgraph kernel's logical
    # shared total (~20 KiB) sits in the 8856..32768 band where cap/margin pools
    # but the cap-itself trigger does not.
    compiled = m.compile_path_c_fused_train_block_artifact_for_model(
        model=model,
        sequence_length=15,
        lowerer=lambda func, *, target, **kwargs: (lambda *args: None),
    )

    assert compiled["status"] == "ok", compiled.get("reason")
    plan = compiled["plan"]
    # The single-launcher fullgraph must fuse to ONE kernel and be SELECTED.
    assert plan["single_kernel_fused"] is True, plan.get(
        "single_launcher_compile_error"
    )
    assert plan["selected_runtime_artifact"] == "single_generated_launcher_chunks", (
        f"auto-split fell back to {plan['selected_runtime_artifact']!r} instead of "
        "the single fused launcher; the shared-scratch pool likely over-pooled an "
        "internal fusion-edge buffer (RULE #1: no silent fused->staged fallback). "
        f"compile error: {plan.get('single_launcher_compile_error')!r}"
    )
    assert plan["single_launcher_runtime_blocked"] is False
    # On the success path no compile error is recorded (the key is only set in the
    # except branch when the single-launcher compile raises).
    assert plan.get("single_launcher_compile_error") is None
    assert plan["single_launcher_compile_verified"] is True


def test_metal_shared_scratch_pool_trigger_is_cap_not_cap_over_margin():
    """The pool TRIGGER is the threadgroup cap; the DEMOTE TARGET is cap/margin.

    Pins the two distinct thresholds so a future edit cannot silently collapse them
    back into ``cap / margin`` (which over-pools small fullgraph kernels). A kernel
    whose logical shared scratch fits the cap must be returned UNCHANGED; a kernel
    that overflows must be pooled down to the cap/margin demote target.
    """
    if _path_c_default_target() != "metal":
        pytest.skip("metal-only")
    c = device_caps()
    cap = int(c.threadgroup_mem_bytes)
    # demote target (cap/margin) is strictly below the trigger (cap): the two are
    # NOT the same number, so the regression (trigger == target) cannot recur.
    assert c.shared_scratch_trigger_bytes < cap
    assert c.shared_scratch_trigger_bytes == int(cap / c.logical_to_physical_shared_margin)

    # A name-gated reverse-scan kernel whose residual shared scratch FITS the cap
    # is left byte-for-byte unchanged (no pool emitted).
    fits = (
        "        _bwd_mamba3_h_next = T.alloc_shared((8,), \"float32\")\n"
        "        _bwd_mamba3_h_prev = T.alloc_shared((8,), \"float32\")\n"
        "        _bwd_mamba3_h_next[0] = _bwd_mamba3_h_prev[0]\n"
    )
    assert sched._pool_oversized_shared_scratch_to_metal_workspace(fits) == fits


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

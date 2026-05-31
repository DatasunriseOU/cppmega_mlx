"""Descriptor-driven schedule planning for Path C fusion regions.

The named Mamba3 FP8 train-block target is an acceptance preset, not the source
of truth.  Schedule construction starts from a region graph, resolves each
known op through a brick descriptor, and then emits a single-entry TileLang
template for that chain.  The production schedule ID remains untrusted by
default until compile, profile, memory, and 1B matrix receipts prove it.
"""

from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import keyword
import linecache
import re
from typing import Any, cast

from cppmega_mlx.runtime.path_c_fusion import (
    CompiledPathCRegion,
    _path_c_default_target,
    FusionCompilePlan,
    FusionKernelSurface,
    FusionScheduleContractStatus,
    MAMBA3_FP8_TRAIN_REQUIRED_REAL_ABI_INPUTS,
    PathCFusionRegion,
    PathCModelShapeEnv,
    PathCSemanticGraphSideChannelBatch,
    Z3SyncSpec,
    build_path_c_aot_autograd_region,
    build_path_c_fusion_region,
    build_path_c_model_regions_from_model,
    build_mamba3_fp8_train_acceptance_fixture_region,
    compile_path_c_region,
    mark_path_c_schedule_template_for_region,
    path_c_mamba3_chunked_scan_enabled as _path_c_mamba3_chunked_scan_enabled,
    tilelang_single_entry_lowerer,
    trusted_path_c_production_schedule_ids,
)


__all__ = [
    "MAMBA3_CHUNKED_FWD_SCAN_CHUNK_SIZE",
    "mamba3_chunked_forward_scan_grid",
    "MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_ID",
    "MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_NAME",
    "MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_STATUS",
    "MAMBA3_FP8_TRAIN_BUFFER_EXTENT",
    "MAMBA3_FP8_TRAIN_PROTOTYPE_SCHEDULE_NAME",
    "MAMBA3_FP8_TRAIN_PROTOTYPE_SCHEDULE_STATUS",
    "CompiledMamba3Fp8TrainFusionSchedule",
    "Mamba3Fp8TrainFusionSchedulePlan",
    "Mamba3Fp8TrainFusionScheduleSpec",
    "DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN",
    "DESCRIPTOR_INTERNAL_BUFFER_POLICY_SCALAR_LOCAL",
    "DESCRIPTOR_EXECUTION_STAGE_ALL",
    "DESCRIPTOR_EXECUTION_STAGE_BACKWARD",
    "DESCRIPTOR_EXECUTION_STAGE_FORWARD",
    "DESCRIPTOR_LOOP_POLICY_FLAT",
    "DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN",
    "PATH_C_DESCRIPTOR_SCHEDULE_GENERATOR",
    "PathCBrickScheduleDescriptor",
    "PathCBrickScheduleFragment",
    "PathCBrickScheduleDescriptorRegistry",
    "PathCDescriptorScheduleStageGroup",
    "PathCFusionScheduleAcceptanceProfile",
    "PathCFusionScheduleChainPlan",
    "PathCFusionScheduleChainSegment",
    "PathCFusionScheduleOptimizer",
    "PathCFusionScheduleOptimizerPlan",
    "PathCFusionScheduleRegistry",
    "PathCFusionScheduleSpec",
    "PathCFusionScheduleTarget",
    "PathCSplitInfeasible",
    "build_path_c_descriptor_prim_func",
    "default_path_c_brick_schedule_descriptor_registry",
    "default_path_c_fusion_schedule_registry",
    "mamba3_fp8_train_prototype_schedule_target",
    "mamba3_fp8_train_fusion_schedule_spec",
    "mamba3_fp8_train_fusion_schedule_template",
    "mamba3_fp8_train_fusion_schedule_target",
    "mamba3_fp8_train_prototype_schedule_template",
    "make_path_c_descriptor_stage_schedule_template",
    "make_path_c_descriptor_schedule_template",
    "merge_path_c_physical_abi_for_prim_funcs",
    "path_c_fusion_schedule_spec",
    "path_c_fusion_schedule_template",
    "path_c_descriptor_stage_prim_funcs",
    "compile_mamba3_fp8_train_fusion_schedule",
    "path_c_semantic_graph_schedule_inputs",
    "plan_path_c_descriptor_phase_groups",
    "plan_path_c_descriptor_stage_groups",
    "plan_path_c_direct_fusion_chain_for_region",
    "plan_path_c_direct_fusion_chains_for_model",
    "plan_path_c_fusion_schedule_for_region",
    "plan_path_c_fusion_schedules_for_model",
    "plan_mamba3_fp8_train_fusion_schedule",
    "prototype_path_c_fusion_schedule_registry",
    "select_path_c_fusion_schedule_target",
]


MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_ID = (
    "mamba3_m2rnn_attention_fp8_train_block_fwd_bwd_v1"
)
MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_NAME = (
    "mamba3_m2rnn_attention_fp8_train_block:production_fwd_bwd_v1"
)
MAMBA3_FP8_TRAIN_PROTOTYPE_SCHEDULE_NAME = (
    "mamba3_m2rnn_attention_fp8_train_block:prototype_fwd_bwd"
)
MAMBA3_FP8_TRAIN_PROTOTYPE_SCHEDULE_STATUS = "prototype"
MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_STATUS = "ready"
PATH_C_DESCRIPTOR_SCHEDULE_GENERATOR = "dynamic_brick_descriptor_generator"
DESCRIPTOR_DEFAULT_BUFFER_EXTENT = 4
DESCRIPTOR_DEFAULT_THREADS = 256
DESCRIPTOR_ROW_PHASED_THREADS = 1024
DESCRIPTOR_INTERNAL_BUFFER_POLICY_SCALAR_LOCAL = "scalar_local"
DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN = "row_local_hidden"
DESCRIPTOR_LOOP_POLICY_FLAT = "flat"
DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN = "row_phased_hidden"
DESCRIPTOR_PHYSICAL_ABI_POLICY_DIRECT = "direct_buffers"
DESCRIPTOR_PHYSICAL_ABI_POLICY_BANKED_BY_DTYPE = "banked_by_dtype"
DESCRIPTOR_PHYSICAL_ABI_POLICY_BANKED_BY_ROLE = "banked_by_role"
DESCRIPTOR_PORTABLE_KERNEL_BUFFER_LIMIT = 31
DESCRIPTOR_SHARED_SCRATCH_SPILL_THRESHOLD_BYTES = 1024
DESCRIPTOR_SHARED_SCRATCH_BUDGET_BYTES = 8 * 1024
MAMBA3_BWD_REPLAY_CHECKPOINT_INTERVAL = 8
M2RNN_BWD_REPLAY_CHECKPOINT_INTERVAL = 1
_DESCRIPTOR_ROOT_READS_MARKER = "cppmega_path_c_root_reads"
_DESCRIPTOR_ROOT_WRITES_MARKER = "cppmega_path_c_root_writes"
_EXACT_ROW_PHASED_BACKWARD_OPS = frozenset(
    {
        "attention_qkv_projection_bwd",
        "sparse_mla_fp8_apply_bwd",
        "m2rnn_bwd",
        "mamba3_mimo_bwd",
    }
)
# Recurrent REVERSE-TIME-SCAN backward ops (mamba3_mimo_bwd / m2rnn_bwd). Each is
# a single ``for time_rev in T.serial(0, S)`` over the WHOLE sequence inside ONE
# threadgroup; under grid_chunks that is ONE Metal command buffer spanning all S
# time steps -> exceeds the macOS GPU watchdog window -> must be switched to
# launcher_chunks TIME-chunking.
#
# REPLACED (design step 6 / §2.4): the former hardcoded
# ``_TIME_CHUNKED_RECURRENT_BACKWARD_OPS = {m2rnn_bwd, mamba3_mimo_bwd}`` frozenset
# is DELETED. Recurrence is now derived STRUCTURALLY by
# ``_op_is_recurrent_state_scan`` (the op's FORWARD descriptor emits a
# ``*_state_recurrence`` codegen step). This reproduces the old membership exactly
# (verified by the membership-parity test) while removing the duplicated literal.
# Per-row-INDEPENDENT heavy BACKWARD ops that exceed the macOS GPU watchdog
# (~5-6s per command buffer) when run as ONE monolithic grid_chunks command
# buffer at full local_gb10_quarter scale (depth=13 hidden=3584 max_seq=4096).
#
# Measured (full scale, each op ISOLATED in its own segment, grid_chunks, run in
# a FRESH process so no prior GPU error pollutes the device):
#   sparse_mla_fp8_apply_bwd      monolithic grid_chunks -> ~12s  -> WATCHDOG KILL
#                                 (kIOGPUCommandBufferCallbackErrorTimeout)
#   attention_qkv_projection_bwd  monolithic grid_chunks -> ~10s  -> WATCHDOG KILL
#   residual_rmsnorm_bwd          monolithic grid_chunks -> ~0.08s -> safe
# Both heavy ops time out the watchdog even ALONE (not from summing several ops
# in one command buffer), so a fused-segment op cap alone does NOT clear them --
# each needs its single op split across multiple command buffers over its
# per-row axis. Unlike the recurrent reverse-time scans (m2rnn_bwd /
# mamba3_mimo_bwd), these ops are per-row-INDEPENDENT: each output row depends
# only on its own input row plus the shared weights, with NO carried reverse-time
# state. So launcher_chunks row-windowing splits them into K command buffers over
# rows with zero cross-launch state (see _row_phased_launcher_carry_buffers_for_nodes
# which adds carry buffers ONLY for mamba3/m2rnn, never for these). The weight
# gradients still accumulate correctly: path_c_first_row_launch zeroes the owner
# grads exactly once on the first (chunk 0, subchunk 0) launch, then every launch
# adds its rows' contribution.
# Measured row-windowed (launcher_chunks, max_rows_per_launch=64 ->
# rows_per_kernel_launch=8 -> 512 launches of 8 rows each):
#   attention_qkv_projection_bwd  ~0.42s per launch -> watchdog-safe.
# Metal-only (CUDA's compiler/scheduler has no such per-command-buffer watchdog,
# so CUDA keeps the monolithic grid_chunks path).
_ROW_CHUNKED_INDEPENDENT_BACKWARD_OPS = frozenset(
    {
        "sparse_mla_fp8_apply_bwd",
        "attention_qkv_projection_bwd",
    }
)
# Per-row-INDEPENDENT heavy FORWARD ops that exceed the macOS GPU watchdog
# (~5-6s per command buffer) when run as ONE monolithic grid_chunks command
# buffer at full local_gb10_quarter scale (depth=13 hidden=3584 max_seq=4096).
# These are the FORWARD analog of _ROW_CHUNKED_INDEPENDENT_BACKWARD_OPS.
#
# Measured at full scale, seg[2] FORWARD region ..._chain_4_6 (the fused
# residual_rmsnorm + attention_qkv_projection + sparse_mla_fp8_apply forward
# segment) runs as a monolithic grid_chunks command buffer over S=4096 and is
# killed at ~9.6s by the watchdog (kIOGPUCommandBufferCallbackErrorTimeout).
# The two heavy ops are attention_qkv_projection and sparse_mla_fp8_apply.
#
# Both are per-row-INDEPENDENT in the FORWARD direction:
#   * attention_qkv_projection is a per-token (per-row) Q/KV linear projection +
#     split-half RoPE + per-head FP8 prepare. Output row R reads input hidden
#     row R plus the shared projection weights only -- there is NO cross-row /
#     cross-time carried state. Its outputs (q_fp8/q_scale/kv_fp8/kv_scale) are
#     caller-owned full-sequence KV-history workspace buffers (see
#     _is_attention_kv_history_workspace_output): across the K row-launches every
#     row [0, S) is written into those persistent buffers.
#   * sparse_mla_fp8_apply writes a per-query-row attention output (and per-row
#     LSE). Query row R reads the SHARED full-sequence KV-history workspace
#     (kv_fp8/kv_scale, indexed by the row's top-k sparse indices, all <= R for
#     causal) plus its own q row -- again NO carried cross-row state; the KV it
#     reads is produced by attention_qkv_projection in an EARLIER segment and is
#     fully materialized before this op runs.
# So launcher_chunks row-windowing splits each into K command buffers over the
# independent ROW axis with zero cross-launch state. As with the independent
# backward ops, _row_phased_launcher_carry_buffers_for_nodes adds carry buffers
# ONLY for mamba3/m2rnn, never for these -- no carry buffers are added here.
#
# CORRECTNESS / PARITY REQUIREMENT: because sparse_mla_fp8_apply row R reads KV
# at positions written by attention_qkv_projection for OTHER rows, these two ops
# must NOT be row-windowed while fused in one segment (a row window would read
# KV positions not yet written in that window). The forward fusion cap is
# therefore lowered to 1 op for any forward segment containing a row-chunked
# forward op (see _effective_forward_max_segment_nodes_for_window), isolating
# attention_qkv_projection and sparse_mla_fp8_apply each into their OWN segment.
# attention_qkv_projection then writes the FULL KV workspace across its K
# committed launches BEFORE sparse_mla_fp8_apply's segment row-windows over the
# now-complete KV -> forward activations are bitwise unchanged by chunking.
# Metal-only (CUDA has no per-command-buffer watchdog, so CUDA keeps the
# monolithic grid_chunks forward path and greedy forward fusion).
_ROW_CHUNKED_INDEPENDENT_FORWARD_OPS = frozenset(
    {
        "attention_qkv_projection",
        "sparse_mla_fp8_apply",
    }
)
DESCRIPTOR_ROW_PHASED_BWD_SCRATCH_ABI_BUFFERS = frozenset(
    {
        "attention_hidden",
        "attention_hidden_grad",
        "hidden_after_mamba3",
        "hidden_after_mamba3_grad",
        "m2rnn_hidden",
        "m2rnn_hidden_grad",
        "m2rnn_delta",
        "m2rnn_delta_grad",
        "mamba3_delta",
        "mamba3_delta_grad",
    }
)
# Mamba3 chunked-scan F0->F1->F2 caller-owned handoff buffers (design §3.1).
# These flow segment->segment across the multi-segment chain and so must
# materialise as distinct physical device ABI buffers (mirrors
# DESCRIPTOR_ROW_PHASED_BWD_SCRATCH_ABI_BUFFERS). summary_states/prev_states are
# fp32 (resolved in _buffer_dtype). Registered into the force-spill scratch ABI
# below so the segment boundary does not alias them. The chunked-scan LIVE flip
# is flag-gated (CPPMEGA_PATH_C_MAMBA3_CHUNKED_SCAN, default OFF) — these names
# never appear in the serial mamba3_mimo surface, so OFF behaviour is unchanged.
DESCRIPTOR_MAMBA3_CHUNKED_FWD_HANDOFF_ABI_BUFFERS = frozenset(
    {
        "mamba3_cb",
        "mamba3_dA_cumsum",
        "mamba3_summary_states",
        "mamba3_prev_states",
        "mamba3_final_state",
    }
)
# Backward grad-handoff buffers (B2 -> B1 -> B0; Stage 3). Force-spilled to named
# ABI scratch (mirror of the forward set) so they land as stable named params
# across the producer/consumer backward segments. All fp32 (resolved in
# _buffer_dtype). Flag-gated (default OFF) — never appear in the serial backward.
DESCRIPTOR_MAMBA3_CHUNKED_BWD_HANDOFF_ABI_BUFFERS = frozenset(
    {
        "mamba3_dh_last",
        "mamba3_dchunk_states",
        "mamba3_dstates",
        "mamba3_dinp_diag",
        "mamba3_dA_cumsum_y",
        "mamba3_dA_cumsum_tail",
    }
)
DESCRIPTOR_ROW_PHASED_BWD_SCRATCH_ABI_CANONICALS = frozenset(
    {
        "hidden",
        "attention_hidden",
        "hidden_after_mamba3",
        "m2rnn_hidden",
        "m2rnn_delta",
        "mamba3_delta",
        "q_fp8",
        "q_scale",
        "kv_fp8",
        "kv_scale",
    }
)
DESCRIPTOR_DEFAULT_MAX_ROWS_PER_LAUNCH = 64
DESCRIPTOR_ROW_DISPATCH_GRID_CHUNKS = "grid_chunks"
DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS = "launcher_chunks"
DESCRIPTOR_DEFAULT_ROW_DISPATCH_MODE = DESCRIPTOR_ROW_DISPATCH_GRID_CHUNKS
DESCRIPTOR_ROW_CHUNK_INDEX_BUFFER = "path_c_row_chunk_index"
DESCRIPTOR_ROW_CHUNK_INDEX_PARAM = "path_c_row_chunk_index"
DESCRIPTOR_ROW_SUBCHUNK_INDEX_PARAM = "path_c_row_subchunk_index"
DESCRIPTOR_ROWS_PER_KERNEL_LAUNCH = 8
# Per-op time-chunk window override for the mamba3 reverse-time-scan backward.
#
# Every launcher-chunked backward segment processes ``rows_per_kernel_launch``
# reverse time-steps per Metal command buffer (one ``(chunk, subchunk)`` launch;
# see _path_c_segment_time_chunk_launches). The shared default of 8 steps/launch
# is watchdog-safe for the per-row-INDEPENDENT heavy ops
# (attention_qkv_projection_bwd / sparse_mla_fp8_apply_bwd, ~0.42s/launch) but
# NOT for mamba3_mimo_bwd: its reverse-scan state (1.75MB/step x 4) was pooled to
# GLOBAL device memory (commit b330bdb, forced by Metal's 32KiB threadgroup cap),
# so each reverse step is far slower and an 8-step command buffer trips the macOS
# GPU watchdog (kIOGPUCommandBufferCallbackErrorTimeout) on its FIRST launch at
# full local_gb10_quarter scale (depth=13 hidden=3584 max_seq=4096). mamba3 gets
# its OWN smaller window here.
#
# Measured isolation sweep (scripts/_bwdgate_mamba3_window_sweep.py, M4 Max, full
# local_gb10_quarter, first 4 launches; first launch carries the checkpoint-replay
# setup, ~5s GPU watchdog window):
#   window  total launches   first launch   steady/launch   first-launch fits?
#   ------  --------------   ------------   -------------   ------------------
#     8         512            (timeout)         --           NO  (kIOGPU timeout)
#     4        1024             3.63s           2.42s         yes (tight, ~1.4s margin)
#     2        2048             2.52s           1.19s         yes (~2.5s margin)
#     1        4096             1.65s           0.68s         yes (~3.3s margin)
# Per-step cost is ~linear (~0.6s/time-step) so the TOTAL mamba3 backward wall is
# roughly window-independent (~40-46 min); reducing the window only trades a fixed
# total cost for more launches with more watchdog margin. 4 is the LARGEST
# watchdog-safe window; 2 is chosen as the committed default for a comfortable
# ~50% margin during the ~61-min end-to-end run (a window-4 launch at 3.63s leaves
# little headroom if a transient GPU stall lands on it). Metal-only: CUDA has no
# per-command-buffer watchdog and keeps the shared default.
MAMBA3_BWD_ROWS_PER_KERNEL_LAUNCH = 2
DESCRIPTOR_BACKWARD_GATE_PARAM = "path_c_run_backward"
DESCRIPTOR_BACKWARD_STAGE_INDEX_PARAM = "path_c_backward_stage_index"
DESCRIPTOR_EXECUTION_STAGE_ALL = "all"
DESCRIPTOR_EXECUTION_STAGE_FORWARD = "forward"
DESCRIPTOR_EXECUTION_STAGE_BACKWARD = "backward"
DESCRIPTOR_EXECUTION_STAGES = frozenset(
    {
        DESCRIPTOR_EXECUTION_STAGE_ALL,
        DESCRIPTOR_EXECUTION_STAGE_FORWARD,
        DESCRIPTOR_EXECUTION_STAGE_BACKWARD,
    }
)
# Metal-only FORWARD fused-segment op-count cap.
#
# The greedy direct-chain planner fuses each forward segment up to the portable
# kernel-buffer limit, producing the 4-op forward mega-kernel
# ``m2rnn + residual_rmsnorm + attention_qkv_projection + sparse_mla_fp8_apply``
# (region ``..._chain_3_7``). Its generated MSL device kernel is ~176-199 KB /
# ~1600-2200 lines (dominated by attention_qkv_projection + sparse_mla_fp8_apply,
# a sparse top-k FP8 attention with RoPE + softmax). TileLang codegen + metallib
# build SUCCEED, but at runtime Metal's MTLCompilerService crashes the final
# AIR->GPU-ISA pipeline-state stage inside ``newComputePipelineState``:
#   InternalError: Check failed: (state != nullptr): ...
#   Compilation failed due to an interrupted connection:
#   XPC_ERROR_CONNECTION_INTERRUPTED. This error occurred after multiple retries.
# Measured (full local_gb10_quarter, depth=13 hidden=3584 max_seq=4096):
#   chain_0_3 (3 light fwd ops)  ~46 KB  -> pipeline OK
#   chain_7_10 (3 bwd ops)       ~116 KB -> pipeline OK
#   chain_3_7 (4 heavy fwd ops)  ~176 KB -> pipeline CRASH (newComputePipelineState)
# Capping forward segments at 2 ops splits the heavy forward chain into smaller
# pipeline-safe kernels. CUDA's compiler has no such pipeline-state limit, so this
# cap is Metal only (CUDA keeps greedy forward fusion).
#
# SUPERSEDED (design step 8 / §3.1): the planner no longer reads this literal --
# it resolves the forward op cap from ``device_caps().forward_max_segment_nodes``
# (the M4 Max preset carries 2; CUDA carries None -> monolithic), and ADDS the
# MSL-byte estimate predicate as a device-grounded backstop. The constant is kept
# only as documentation of the characterized M4 Max value.
METAL_FORWARD_MAX_SEGMENT_NODES = 2

# Metal-only BACKWARD fused-segment op-count cap.
#
# Sibling of METAL_FORWARD_MAX_SEGMENT_NODES, for the reverse-autograd backward
# phase. The greedy direct-chain planner otherwise fuses backward ops up to the
# portable buffer limit, producing the 3-op backward mega-kernel
# ``sparse_mla_fp8_apply_bwd + attention_qkv_projection_bwd + residual_rmsnorm_bwd``
# (region ``..._chain_7_10``). Measured at full local_gb10_quarter scale that
# fused segment is ONE ~115 KB grid_chunks command buffer that runs ~10-25s of
# GPU time and is killed by the macOS GPU watchdog
# (kIOGPUCommandBufferCallbackErrorTimeout) -- the FIRST backward segment to die.
# Capping backward segments at 1 op puts each backward op in its OWN segment /
# command buffer, which (a) keeps each command buffer small and (b) lets the two
# per-row-INDEPENDENT heavy ops (sparse_mla_fp8_apply_bwd,
# attention_qkv_projection_bwd; see _ROW_CHUNKED_INDEPENDENT_BACKWARD_OPS) get
# launcher_chunks row-windowing so each command buffer holds only a row window's
# worth of work (~0.42s) -- well under the watchdog. The light backward ops
# (residual_rmsnorm_bwd / entry_rmsnorm_bwd) run monolithically in ~0.08s.
#
# NOTE: an op cap alone does NOT clear sparse_mla_fp8_apply_bwd /
# attention_qkv_projection_bwd -- each times out the watchdog even ISOLATED in a
# 1-op segment, so isolation MUST be paired with row-windowing (the cap of 1
# guarantees the isolation those ops need to be row-chunked). CUDA has no such
# watchdog, so the cap is Metal-only (CUDA keeps greedy backward fusion).
#
# SUPERSEDED (design step 8 / §3.1): the planner now resolves the backward op cap
# from ``device_caps().backward_max_segment_nodes`` (the M4 Max preset carries 1
# -- the watchdog per-op isolation; CUDA carries None -> monolithic), and ADDS the
# watchdog-time estimate predicate as a device-grounded backstop. The constant is
# kept only as documentation of the characterized M4 Max value.
METAL_BACKWARD_MAX_SEGMENT_NODES = 1


class _ResolveFromTarget:
    """Sentinel: resolve a planner default from the active lowering target.

    A distinct type (not ``None``, which is a valid "no cap" value) so callers
    can explicitly request the target-resolved default or override it.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "<resolve-from-target>"


_RESOLVE_FORWARD_CAP_FROM_TARGET = _ResolveFromTarget()
_RESOLVE_BACKWARD_CAP_FROM_TARGET = _ResolveFromTarget()
# Resolve the kernel buffer-argument limit from the device-capability probe
# (caps.buffer_arg_limit): Metal -> 31 (family ABI const), CUDA -> unbounded
# sentinel. Distinct sentinel so an explicit caller-supplied limit is honored
# unchanged while the default tracks the queried/preset device cap.
_RESOLVE_BUFFER_LIMIT_FROM_CAPS = _ResolveFromTarget()
_GENERIC_MODEL_REAL_ABI_INPUT_SUFFIXES = (
    "residual_norm_weight",
    # Block A: the per-brick entry RMSNorm weight emitted by the model-
    # level surface lowering for the first in-region brick.
    "entry_rmsnorm_weight",
)
_TRAIN_STEP_SCALAR_OUTPUT_ABI_NAMES = ("loss", "ntokens")
_TRAIN_STEP_SCALAR_OUTPUT_ABI_REASON = (
    "train-step scalar ABI slots are declared, but suffix loss codegen is not "
    "fused into the descriptor body yet"
)
_TRAIN_STEP_SUFFIX_LOSS_INPUT_ABI_NAMES = (
    "target_ids",
    "target_mask",
    "final_norm_weight",
    "lm_head_weight",
)
_TRAIN_STEP_SUFFIX_LOSS_PARAMETER_GRAD_ABI_NAMES = (
    "final_norm_weight_grad",
    "lm_head_weight_grad",
)
_TRAIN_STEP_SUFFIX_LOSS_INPUT_ABI_REASON = (
    "train-step suffix loss inputs are declared for fused loss codegen; "
    "target_mask is consumed for ntokens, while full loss codegen is pending"
)
class PathCSplitInfeasible(RuntimeError):
    """A Path-C segment cannot be split to fit a hard device limit (RULE #1).

    Raised when an irreducible segment -- a single op (``end - start == 1``) that,
    after the pool/demote pass, still exceeds the threadgroup-memory cap, or
    exceeds the buffer-argument count, or (forward) exceeds the MSL pipeline-state
    ceiling, or (backward) exceeds the watchdog budget even at the minimal row
    window -- has no feasible split.  We NEVER fall back to greedy fusion, NEVER
    emit the oversized kernel, NEVER return zeros: we surface where + what failed.
    """

    def __init__(
        self,
        region_name: str,
        characteristic: str,
        estimated_value: Any,
        limit: Any,
        *,
        op_name: str = "",
    ) -> None:
        self.region_name = region_name
        self.characteristic = characteristic
        self.estimated_value = estimated_value
        self.limit = limit
        self.op_name = op_name
        op_suffix = f" op={op_name}" if op_name else ""
        super().__init__(
            f"PathCSplitInfeasible: region={region_name}{op_suffix} "
            f"{characteristic} estimate={estimated_value} exceeds device limit "
            f"{limit}; no feasible split (RULE #1: refusing to emit an over-budget "
            f"kernel or fall back to greedy fusion)"
        )


_DTYPE_NBYTES = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "float16": 2,
    "bfloat16": 2,
    "uint16": 2,
    "int16": 2,
    "float32": 4,
    "uint32": 4,
    "int32": 4,
    "float64": 8,
    "uint64": 8,
    "int64": 8,
}


def path_c_semantic_graph_schedule_inputs(
    graph: PathCSemanticGraphSideChannelBatch,
) -> dict[str, object]:
    """Return caller-owned semantic graph buffers in Path C ABI names."""

    inputs: dict[str, object] = {}
    if graph.call_edges is not None:
        inputs["path_c_semantic_call_edges"] = graph.call_edges
    if graph.call_edge_mask is not None:
        inputs["path_c_semantic_call_edge_mask"] = graph.call_edge_mask
    if graph.type_edges is not None:
        inputs["path_c_semantic_type_edges"] = graph.type_edges
    if graph.type_edge_mask is not None:
        inputs["path_c_semantic_type_edge_mask"] = graph.type_edge_mask
    return inputs


def _mamba3_fp8_train_buffer_extent() -> int:
    from cppmega_mlx.recipes.model_factory import local_gb10_quarter_profile

    return int(local_gb10_quarter_profile().max_seq_length)


MAMBA3_FP8_TRAIN_BUFFER_EXTENT = _mamba3_fp8_train_buffer_extent()

_PRODUCTION_SCHEDULE_REASON = (
    "dynamic brick descriptors can construct a single-entry TileLang/TIR "
    "schedule for the model-semantic mamba3 + residual/RMSNorm + m2rnn + "
    "attention_qkv_projection + sparse_mla_fp8_apply fwd/bwd region, but the "
    "schedule remains untrusted by default until compile, 1B matrix, profiling, "
    "memory, and cache receipts pass"
)
_MAMBA3_FP8_TRAIN_FWD_BWD_OP_SIGNATURE = (
    # Block A: the fused region applies an eager pre-block RMSNorm to
    # ``hidden_entry`` before the first brick consumes it. The bwd is
    # auto-fissioned in reverse order and runs last.
    "entry_rmsnorm",
    "mamba3_mimo",
    "residual_rmsnorm",
    "m2rnn",
    "residual_rmsnorm",
    "attention_qkv_projection",
    "sparse_mla_fp8_apply",
    "sparse_mla_fp8_apply_bwd",
    "attention_qkv_projection_bwd",
    "residual_rmsnorm_bwd",
    "m2rnn_bwd",
    "residual_rmsnorm_bwd",
    "mamba3_mimo_bwd",
    "entry_rmsnorm_bwd",
)


@dataclass(frozen=True)
class PathCBrickScheduleDescriptor:
    """Schedule metadata for one reusable Path C model brick."""

    op_name: str
    implementation_status: str
    required_codegen_steps: tuple[str, ...]
    schedule_family: str = "loop_descriptor_dataflow"
    supports_backward: bool = True
    description: str = ""
    production_source: str = ""
    production_fragment_status: str = "not_inlined"
    production_fragment_reason: str = ""
    preferred_internal_buffer_policy: str = DESCRIPTOR_INTERNAL_BUFFER_POLICY_SCALAR_LOCAL
    preferred_loop_policy: str = DESCRIPTOR_LOOP_POLICY_FLAT
    production_fragment_policy: str = ""
    production_fragment_codegen_step: str = ""
    production_fragment_inlined_reason: str = ""
    max_production_hidden_size: int | None = None
    max_production_op_occurrences: int | None = None
    max_production_op_occurrences_min_hidden_size: int | None = None
    fragment_emitter: Callable[..., Any] | None = None


@dataclass(frozen=True)
class PathCFusionScheduleSpec:
    """Schedule contract selected from a concrete Path C model region."""

    schedule_id: str
    schedule_name: str
    region_name: str
    implementation_kind: str
    implementation_status: str
    missing_reason: str
    trusted_by_default: bool
    contract_name: str
    contract_key: str
    shape_env_key: str
    op_signature: tuple[str, ...]
    required_internal_buffers: tuple[str, ...]
    required_external_buffers: tuple[str, ...]
    required_real_abi_inputs: tuple[str, ...]
    required_real_abi_input_shapes: tuple[str, ...]
    missing_real_abi_inputs: tuple[str, ...]
    real_abi_contract_complete: bool
    required_codegen_steps: tuple[str, ...]
    schedule_generator: str
    schedule_generator_status: str
    internal_buffer_policy: str
    loop_policy: str
    buffer_extent: int
    loop_extent: int
    brick_ops: tuple[str, ...]
    brick_schedule_families: tuple[str, ...]
    brick_descriptor_statuses: tuple[str, ...]
    brick_production_fragment_statuses: tuple[str, ...]
    brick_production_fragment_reasons: tuple[str, ...]
    brick_production_fragment_blockers: tuple[str, ...]
    production_fragments_complete: bool


@dataclass(frozen=True)
class Mamba3Fp8TrainFusionScheduleSpec(PathCFusionScheduleSpec):
    """Named acceptance schedule target for the 1B Path C train block."""


@dataclass(frozen=True)
class PathCFusionScheduleTarget:
    """Registry entry for a high-level Path C fused schedule pattern."""

    schedule_id: str
    schedule_name: str
    op_signature: tuple[str, ...]
    schedule_status: str
    implementation_kind: str
    missing_reason: str
    required_codegen_steps: tuple[str, ...]
    schedule_template: Callable[[Any], Any]
    required_real_abi_inputs: tuple[str, ...] = ()
    brick_descriptors: tuple[PathCBrickScheduleDescriptor, ...] = ()
    schedule_generator: str = PATH_C_DESCRIPTOR_SCHEDULE_GENERATOR
    buffer_extent: int = DESCRIPTOR_DEFAULT_BUFFER_EXTENT
    internal_buffer_policy: str = DESCRIPTOR_INTERNAL_BUFFER_POLICY_SCALAR_LOCAL
    loop_policy: str = DESCRIPTOR_LOOP_POLICY_FLAT
    physical_abi_policy: str = DESCRIPTOR_PHYSICAL_ABI_POLICY_DIRECT
    max_rows_per_launch: int | None = None
    row_dispatch_mode: str = DESCRIPTOR_DEFAULT_ROW_DISPATCH_MODE


@dataclass(frozen=True)
class PathCFusionScheduleAcceptanceProfile:
    """Metadata applied to a descriptor-built target selected from a live region."""

    op_signature: tuple[str, ...]
    schedule_id: str
    schedule_name: str
    schedule_status: str
    implementation_kind: str
    missing_reason: str
    required_codegen_steps: tuple[str, ...]
    entry_symbol: str | None = None
    required_real_abi_inputs: tuple[str, ...] = ()
    buffer_extent: int = DESCRIPTOR_DEFAULT_BUFFER_EXTENT
    required_region_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class PathCFusionScheduleOptimizerPlan:
    """Generic high-level Path C optimization plan."""

    region: PathCFusionRegion
    plan: FusionCompilePlan
    schedule_target: PathCFusionScheduleTarget | None


@dataclass(frozen=True)
class PathCFusionScheduleChainSegment:
    """One contiguous fused segment in a generic Path C schedule chain."""

    index: int
    node_start: int
    node_end: int
    region: PathCFusionRegion
    plan: FusionCompilePlan | None
    schedule_target: PathCFusionScheduleTarget | None
    kernel_parameter_count: int | None
    physical_abi_policy: str
    status: str
    reason: str
    execution_phase: str


@dataclass(frozen=True)
class PathCFusionScheduleChainPlan:
    """Generic direct-buffer chain plan for a Path C region."""

    source_region: PathCFusionRegion
    max_kernel_buffers: int
    segments: tuple[PathCFusionScheduleChainSegment, ...]
    status: str
    reason: str


@dataclass(frozen=True)
class PathCDescriptorScheduleStageGroup:
    """One descriptor-planned generated-kernel stage within a train block."""

    index: int
    execution_stage: str
    active_node_names: tuple[str, ...]
    stage_suffix: str
    row_dispatch_mode: str
    rows_per_kernel_launch: int
    reason: str


@dataclass(frozen=True)
class Mamba3Fp8TrainFusionSchedulePlan:
    """High-level planner output for the Mamba3 FP8 train-block target."""

    region: PathCFusionRegion
    plan: FusionCompilePlan
    schedule_spec: Mamba3Fp8TrainFusionScheduleSpec


@dataclass(frozen=True)
class CompiledMamba3Fp8TrainFusionSchedule:
    """Lowered named Mamba3 FP8 train-block schedule with its contract."""

    region: PathCFusionRegion
    compiled: CompiledPathCRegion
    schedule_spec: Mamba3Fp8TrainFusionScheduleSpec


@dataclass(frozen=True)
class _ScheduleNodeView:
    name: str
    op_name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    backward: str = ""


def _path_c_schedule_node_execution_phase(node: Any) -> str:
    backward = str(getattr(node, "backward", ""))
    op_name = str(getattr(node, "op_name", ""))
    # The chunked F0/F1/F2 SSD-core FORWARD ops carry NO synthesized AOT backward
    # (they own their output gradient: backward="owner_output"), but they must run
    # in the FORWARD execution stage so the compile-site delegation interpose
    # substitutes their grid kernel during the forward pass (RULE #1: the chunked
    # forward is never grouped as a backward fragment). The chunked B0/B1/B2
    # BACKWARD ops (``*_bwd``) are explicitly EXCLUDED here so they classify as
    # backward below (they too are in _MAMBA3_CHUNKED_GRID_DELEGATION_OPS for the
    # shared interpose, but they belong to the backward execution stage).
    if (
        op_name in _MAMBA3_CHUNKED_GRID_DELEGATION_OPS
        and not op_name.endswith("_bwd")
    ):
        return "forward"
    if backward == "owner_output" or op_name.endswith("_bwd"):
        return "backward"
    return "forward"


def _path_c_schedule_segment_execution_phase(nodes: Iterable[Any]) -> str:
    phases = {_path_c_schedule_node_execution_phase(node) for node in nodes}
    if not phases:
        return "empty"
    if len(phases) == 1:
        return next(iter(phases))
    return "mixed"


def _path_c_descriptor_stage_node_name(node: Any) -> str:
    name = str(getattr(node, "name", ""))
    if not name:
        raise ValueError("Path C descriptor stage node is missing a name")
    return name


def _path_c_descriptor_stage_node_op_name(node: Any) -> str:
    return str(getattr(node, "op_name", ""))


def _op_is_recurrent_state_scan(
    op_name: str,
    *,
    registry: "PathCBrickScheduleDescriptorRegistry | None" = None,
) -> bool:
    """Structural test: does ``op_name`` carry a reverse-time recurrent state scan?

    Replaces the hardcoded ``_TIME_CHUNKED_RECURRENT_BACKWARD_OPS`` /
    ``_ROW_CHUNKED_INDEPENDENT_BACKWARD_OPS`` frozensets (design §2.4). A backward
    op is a reverse-time recurrence iff its FORWARD descriptor emits a
    ``*_state_recurrence`` codegen step -- i.e. the forward pass carries scan
    state across time, so the backward is a single reverse-time scan over the
    whole sequence and must be TIME-chunked (launcher_chunks). The per-row
    INDEPENDENT heavy ops (attention_qkv_projection / sparse_mla_fp8_apply) have
    no such step and are ROW-windowed instead.

    The forward descriptor is resolved by stripping a trailing ``_bwd`` (the
    synthesized AOT-backward descriptors do not carry the recurrence step; the
    information lives on the forward descriptor that the backward differentiates).
    This reproduces the old frozenset membership EXACTLY:
    {m2rnn_bwd, mamba3_mimo_bwd} -> recurrent; everything else -> independent.
    """

    forward_op = op_name[: -len("_bwd")] if op_name.endswith("_bwd") else op_name
    reg = registry or default_path_c_brick_schedule_descriptor_registry()
    descriptor = reg.descriptor_for(forward_op)
    if descriptor is None:
        return False
    return any(
        "_state_recurrence" in step for step in descriptor.required_codegen_steps
    )


def _node_is_recurrent_state_scan(
    node: Any,
    *,
    registry: "PathCBrickScheduleDescriptorRegistry | None" = None,
) -> bool:
    return _op_is_recurrent_state_scan(
        _path_c_descriptor_stage_node_op_name(node), registry=registry
    )


def _nodes_contain_row_chunked_forward_op(nodes: Iterable[Any]) -> bool:
    """True if any node is a row-chunked-independent FORWARD op.

    These ops (:data:`_ROW_CHUNKED_INDEPENDENT_FORWARD_OPS`) must be ISOLATED in
    their own forward segment so each can be launcher_chunks row-windowed without
    a fused sibling reading partially-written cross-row workspace (see that
    frozenset's docstring for the KV-history parity requirement).
    """

    return any(
        _path_c_descriptor_stage_node_op_name(node)
        in _ROW_CHUNKED_INDEPENDENT_FORWARD_OPS
        for node in nodes
    )


def _effective_forward_max_segment_nodes_for_window(
    candidate_nodes: Iterable[Any],
    forward_max_segment_nodes: int | None,
) -> int | None:
    """Forward op cap for a candidate segment window.

    Lowers the configured ``forward_max_segment_nodes`` to 1 whenever the window
    contains a row-chunked-independent forward op so that
    ``attention_qkv_projection`` / ``sparse_mla_fp8_apply`` each land alone in
    their own segment (the watchdog isolation that lets them be row-windowed and
    that keeps sparse-MLA's KV reads bitwise-correct -- see
    :data:`_ROW_CHUNKED_INDEPENDENT_FORWARD_OPS`). Returns the unchanged cap
    otherwise; ``None`` stays ``None`` only when there is no row-chunked forward
    op (a row-chunked forward op always forces a cap of 1, even on an otherwise
    uncapped target -- but this is reached only on Metal, where the cap is set).
    """

    nodes = tuple(candidate_nodes)
    if not _nodes_contain_row_chunked_forward_op(nodes):
        return forward_max_segment_nodes
    if forward_max_segment_nodes is None:
        return 1
    return min(int(forward_max_segment_nodes), 1)


def _path_c_descriptor_stage_rows_per_kernel_launch_for_node(node: Any) -> int:
    op_name = _path_c_descriptor_stage_node_op_name(node)
    if op_name in {"attention_qkv_projection_bwd", "mamba3_mimo_bwd"}:
        return 1
    return DESCRIPTOR_ROWS_PER_KERNEL_LAUNCH


def _mamba3_bwd_rows_per_kernel_launch_for_nodes(nodes: Iterable[Any]) -> int:
    """Per-launch time-chunk window for a launcher-chunked backward segment.

    Returns :data:`MAMBA3_BWD_ROWS_PER_KERNEL_LAUNCH` (the smaller, watchdog-safe
    mamba3 window) when the segment contains ``mamba3_mimo_bwd`` -- whose pooled
    global reverse-scan state makes each reverse step expensive enough that the
    shared 8-step window trips the macOS GPU watchdog. Every other
    launcher-chunked backward op (the per-row-INDEPENDENT heavy ops and m2rnn_bwd)
    keeps the shared :data:`DESCRIPTOR_ROWS_PER_KERNEL_LAUNCH` window, which is
    already watchdog-safe for them. With the backward op cap = 1 each backward op
    is in its OWN segment, so a segment never mixes mamba3 with another op.
    """

    if any(
        _path_c_descriptor_stage_node_op_name(node) == "mamba3_mimo_bwd"
        for node in nodes
    ):
        return MAMBA3_BWD_ROWS_PER_KERNEL_LAUNCH
    return DESCRIPTOR_ROWS_PER_KERNEL_LAUNCH


def _path_c_descriptor_stage_append(
    groups: list[PathCDescriptorScheduleStageGroup],
    *,
    execution_stage: str,
    stage_index: int,
    active_node_names: Sequence[str],
    reason: str,
    row_dispatch_mode: str = DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS,
    rows_per_kernel_launch: int = DESCRIPTOR_ROWS_PER_KERNEL_LAUNCH,
) -> None:
    names = tuple(str(name) for name in active_node_names if str(name))
    if not names:
        return
    validated_stage = _validated_execution_stage(execution_stage)
    validated_mode = _validated_row_dispatch_mode(row_dispatch_mode)
    prefix = "g" if validated_stage == DESCRIPTOR_EXECUTION_STAGE_FORWARD else "b"
    groups.append(
        PathCDescriptorScheduleStageGroup(
            index=stage_index,
            execution_stage=validated_stage,
            active_node_names=names,
            stage_suffix=f"{prefix}{stage_index}",
            row_dispatch_mode=validated_mode,
            rows_per_kernel_launch=_validated_rows_per_kernel_launch(
                rows_per_kernel_launch
            ),
            reason=reason,
        )
    )


def plan_path_c_descriptor_stage_groups(
    region: Any,
) -> tuple[PathCDescriptorScheduleStageGroup, ...]:
    """Plan generated TileLang stages from a descriptor region graph.

    The caller still lowers each returned stage through the normal descriptor
    template. Keeping the grouping here makes the fused train-block assembly a
    property of the block descriptor graph instead of a runtime-script policy.
    """

    nodes = tuple(getattr(region, "nodes", ()))
    if not nodes:
        return ()

    forward_nodes = tuple(
        node
        for node in nodes
        if _path_c_schedule_node_execution_phase(node) == "forward"
    )
    backward_nodes = tuple(
        node
        for node in nodes
        if _path_c_schedule_node_execution_phase(node) == "backward"
    )

    groups: list[PathCDescriptorScheduleStageGroup] = []
    forward_index = 0
    forward_tail: list[str] = []
    for node in forward_nodes:
        name = _path_c_descriptor_stage_node_name(node)
        op_name = _path_c_descriptor_stage_node_op_name(node)
        # The chunked F0/F1/F2 forward ops are GRID-launched single-node segments
        # (own multi-grid T.Kernel) that MUST each be isolated into their own
        # forward stage so the compile-site delegation interpose (which requires
        # len(nodes) == 1) substitutes the proven build_*_metal grid kernel
        # (RULE #1: never fused into a single-T.Kernel template fragment).
        if (
            op_name in {"entry_rmsnorm", "mamba3_mimo"}
            or op_name in _MAMBA3_CHUNKED_GRID_DELEGATION_OPS
        ):
            if forward_tail:
                _path_c_descriptor_stage_append(
                    groups,
                    execution_stage=DESCRIPTOR_EXECUTION_STAGE_FORWARD,
                    stage_index=forward_index,
                    active_node_names=forward_tail,
                    reason="descriptor_fuses_forward_tail_between_heavy_blocks",
                )
                forward_index += 1
                forward_tail = []
            _path_c_descriptor_stage_append(
                groups,
                execution_stage=DESCRIPTOR_EXECUTION_STAGE_FORWARD,
                stage_index=forward_index,
                active_node_names=(name,),
                reason=f"descriptor_isolates_forward_{op_name}_block",
            )
            forward_index += 1
            continue
        forward_tail.append(name)
    if forward_tail:
        _path_c_descriptor_stage_append(
            groups,
            execution_stage=DESCRIPTOR_EXECUTION_STAGE_FORWARD,
            stage_index=forward_index,
            active_node_names=forward_tail,
            reason="descriptor_fuses_remaining_forward_blocks",
        )

    backward_consumed: set[str] = set()
    backward_index = 0
    for op_name in ("sparse_mla_fp8_apply_bwd", "attention_qkv_projection_bwd"):
        for node in backward_nodes:
            name = _path_c_descriptor_stage_node_name(node)
            if (
                name in backward_consumed
                or _path_c_descriptor_stage_node_op_name(node) != op_name
            ):
                continue
            _path_c_descriptor_stage_append(
                groups,
                execution_stage=DESCRIPTOR_EXECUTION_STAGE_BACKWARD,
                stage_index=backward_index,
                active_node_names=(name,),
                reason=f"descriptor_isolates_backward_{op_name}_fragment",
                rows_per_kernel_launch=(
                    _path_c_descriptor_stage_rows_per_kernel_launch_for_node(
                        node
                    )
                ),
            )
            backward_consumed.add(name)
            backward_index += 1

    for node in backward_nodes:
        name = _path_c_descriptor_stage_node_name(node)
        if name in backward_consumed:
            continue
        _path_c_descriptor_stage_append(
            groups,
            execution_stage=DESCRIPTOR_EXECUTION_STAGE_BACKWARD,
            stage_index=backward_index,
            active_node_names=(name,),
            reason="descriptor_keeps_backward_block_as_generated_stage",
            rows_per_kernel_launch=(
                _path_c_descriptor_stage_rows_per_kernel_launch_for_node(node)
            ),
        )
        backward_consumed.add(name)
        backward_index += 1

    return tuple(groups)


def _path_c_backward_stage_name_groups_for_nodes(
    nodes: Sequence[Any],
) -> tuple[tuple[str, ...], ...]:
    """Return descriptor-planned backward stage names for a node sequence."""

    backward_nodes = tuple(
        node
        for node in nodes
        if _path_c_schedule_node_execution_phase(node) == "backward"
    )
    groups: list[tuple[str, ...]] = []
    consumed: set[str] = set()
    for op_name in ("sparse_mla_fp8_apply_bwd", "attention_qkv_projection_bwd"):
        for node in backward_nodes:
            name = _path_c_descriptor_stage_node_name(node)
            if (
                name in consumed
                or _path_c_descriptor_stage_node_op_name(node) != op_name
            ):
                continue
            groups.append((name,))
            consumed.add(name)
    for node in backward_nodes:
        name = _path_c_descriptor_stage_node_name(node)
        if name in consumed:
            continue
        groups.append((name,))
        consumed.add(name)
    return tuple(groups)


def plan_path_c_descriptor_phase_groups(
    region: Any,
) -> tuple[PathCDescriptorScheduleStageGroup, ...]:
    """Plan one generated TileLang stage per train-block execution phase.

    This is the closest runnable form of the original Path C intent while the
    single Metal function is too large for first-compile: the descriptor graph,
    not the Python runtime, decides which blocks form the forward phase and
    which blocks form the backward phase.
    """

    nodes = tuple(getattr(region, "nodes", ()))
    if not nodes:
        return ()

    forward_names = tuple(
        _path_c_descriptor_stage_node_name(node)
        for node in nodes
        if _path_c_schedule_node_execution_phase(node) == "forward"
    )
    backward_names = tuple(
        _path_c_descriptor_stage_node_name(node)
        for node in nodes
        if _path_c_schedule_node_execution_phase(node) == "backward"
    )

    groups: list[PathCDescriptorScheduleStageGroup] = []
    _path_c_descriptor_stage_append(
        groups,
        execution_stage=DESCRIPTOR_EXECUTION_STAGE_FORWARD,
        stage_index=0,
        active_node_names=forward_names,
        reason="descriptor_fuses_forward_phase_blocks",
    )
    _path_c_descriptor_stage_append(
        groups,
        execution_stage=DESCRIPTOR_EXECUTION_STAGE_BACKWARD,
        stage_index=0,
        active_node_names=backward_names,
        reason="descriptor_fuses_backward_phase_blocks",
    )
    return tuple(groups)


def make_path_c_descriptor_stage_schedule_template(
    *,
    schedule_target: PathCFusionScheduleTarget,
    region: Any,
    abi_prim_func: Any,
    execution_stage: str,
    active_node_names: Sequence[str] | None = None,
    stage_suffix: str = "",
    row_dispatch_mode: str = DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS,
    rows_per_kernel_launch: int = DESCRIPTOR_ROWS_PER_KERNEL_LAUNCH,
) -> Callable[[Any], Any]:
    """Build a generated stage template with the train-block ABI contract.

    Runtime bank owners, compile receipts, and m04 artifact install all need to
    agree on the exact stage PrimFunc ABI. Keeping the stage-template assembly
    here makes that ABI a descriptor-scheduler product instead of a duplicated
    Python runtime policy.
    """

    suffix = f"_{stage_suffix}" if stage_suffix else ""
    stage_name = (
        f"{getattr(region, 'name', 'path_c_region')}_{execution_stage}{suffix}"
    )
    return mark_path_c_schedule_template_for_region(
        make_path_c_descriptor_schedule_template(
            schedule_target.brick_descriptors,
            entry_symbol=stage_name,
            buffer_extent=schedule_target.buffer_extent,
            shape_env=getattr(abi_prim_func, "_cppmega_path_c_shape_env", None),
            internal_buffer_policy=schedule_target.internal_buffer_policy,
            loop_policy=schedule_target.loop_policy,
            physical_abi_policy=schedule_target.physical_abi_policy,
            train_step_output_abi=bool(
                getattr(
                    schedule_target.schedule_template,
                    "_cppmega_path_c_train_step_output_abi_enabled",
                    False,
                )
            ),
            max_rows_per_launch=schedule_target.max_rows_per_launch,
            row_dispatch_mode=row_dispatch_mode,
            rows_per_kernel_launch=rows_per_kernel_launch,
            execution_stage=execution_stage,
            active_node_names=active_node_names,
        ),
        region,
        implementation_kind=schedule_target.implementation_kind,
        production_schedule_id=schedule_target.schedule_id
        if schedule_target.implementation_kind == "production"
        else "",
        required_real_abi_inputs=schedule_target.required_real_abi_inputs,
    )


def path_c_descriptor_stage_prim_funcs(
    *,
    region: Any,
    schedule_target: PathCFusionScheduleTarget,
    abi_prim_func: Any,
    groups: Sequence[PathCDescriptorScheduleStageGroup] | None = None,
) -> tuple[Any, ...]:
    """Materialise descriptor-planned stage PrimFuncs for a train block."""

    stage_groups = tuple(groups) if groups is not None else plan_path_c_descriptor_stage_groups(region)
    templates = tuple(
        make_path_c_descriptor_stage_schedule_template(
            schedule_target=schedule_target,
            region=region,
            abi_prim_func=abi_prim_func,
            execution_stage=group.execution_stage,
            active_node_names=group.active_node_names,
            stage_suffix=group.stage_suffix,
            row_dispatch_mode=group.row_dispatch_mode,
            rows_per_kernel_launch=group.rows_per_kernel_launch,
        )
        for group in stage_groups
    )
    return tuple(template(region) for template in templates)


def _shape_element_count(shape: Sequence[int]) -> int:
    size = 1
    for dim in shape:
        size *= int(dim)
    return size


def merge_path_c_physical_abi_for_prim_funcs(
    prim_funcs: Sequence[Any],
) -> tuple[dict[str, Any], dict[str, tuple[int, ...]]]:
    """Merge logical placements and max bank shapes across generated PrimFuncs."""

    merged_map: dict[str, Any] = {}
    merged_shapes: dict[str, tuple[int, ...]] = {}
    for prim_func in prim_funcs:
        merged_map.update(
            dict(getattr(prim_func, "_cppmega_path_c_physical_buffer_abi_map", {}) or {})
        )
        for raw_name, raw_shape in dict(
            getattr(prim_func, "_cppmega_path_c_physical_buffer_abi_shapes", {}) or {}
        ).items():
            name = str(raw_name)
            shape = tuple(int(dim) for dim in tuple(raw_shape))
            existing = merged_shapes.get(name)
            if existing is None or _shape_element_count(shape) > _shape_element_count(existing):
                merged_shapes[name] = shape
    return merged_map, merged_shapes


@dataclass(frozen=True)
class PathCBrickScheduleFragment:
    allocations: tuple[str, ...]
    statements: tuple[str, ...]


_ScheduleNodeFragment = PathCBrickScheduleFragment


@dataclass(frozen=True)
class _PhysicalAbiPlan:
    param_lines: tuple[str, ...]
    external_access_by_buffer: Mapping[str, str]
    physical_buffer_shapes: Mapping[str, tuple[int, ...]]
    logical_to_physical: Mapping[str, Mapping[str, Any]]


class PathCBrickScheduleDescriptorRegistry:
    """Registry mapping reusable brick op names to schedule descriptors."""

    def __init__(
        self,
        descriptors: Sequence[PathCBrickScheduleDescriptor] = (),
    ) -> None:
        self._descriptors: dict[str, PathCBrickScheduleDescriptor] = {}
        for descriptor in descriptors:
            self.register(descriptor)

    def register(
        self,
        descriptor: PathCBrickScheduleDescriptor,
    ) -> "PathCBrickScheduleDescriptorRegistry":
        if not isinstance(descriptor, PathCBrickScheduleDescriptor):
            raise TypeError("descriptor must be PathCBrickScheduleDescriptor")
        if not descriptor.op_name:
            raise ValueError("Path C brick descriptor op_name must not be empty")
        self._descriptors[descriptor.op_name] = descriptor
        return self

    def descriptor_for(self, op_name: str) -> PathCBrickScheduleDescriptor | None:
        descriptor = self._descriptors.get(op_name)
        if descriptor is not None:
            return descriptor
        if not op_name.endswith("_bwd"):
            return None
        base_op_name = op_name[: -len("_bwd")]
        base = self._descriptors.get(base_op_name)
        if base is None or not base.supports_backward:
            return None
        return PathCBrickScheduleDescriptor(
            op_name=op_name,
            implementation_status=f"{base.implementation_status}:aot_backward",
            required_codegen_steps=(f"{base_op_name}_aot_backward_descriptor",),
            schedule_family=base.schedule_family,
            supports_backward=False,
            description=f"AOT backward descriptor for {base_op_name}",
            production_source=base.production_source,
            production_fragment_status="region_fragment_inlined_unoptimized",
            production_fragment_reason=(
                "synthesized AOT backward descriptors use the generic owner-output "
                "gradient fragment; register an explicit backward descriptor before "
                "this can be treated as production-inlined"
            ),
            fragment_emitter=None,
        )

    def descriptors_for_signature(
        self,
        op_signature: Sequence[str],
    ) -> tuple[PathCBrickScheduleDescriptor, ...] | None:
        descriptors: list[PathCBrickScheduleDescriptor] = []
        for op_name in op_signature:
            descriptor = self.descriptor_for(str(op_name))
            if descriptor is None:
                return None
            descriptors.append(descriptor)
        return tuple(descriptors)


def default_path_c_brick_schedule_descriptor_registry() -> (
    PathCBrickScheduleDescriptorRegistry
):
    """Return descriptors for model bricks that can participate in Path C chains."""

    return PathCBrickScheduleDescriptorRegistry(
        (
            PathCBrickScheduleDescriptor(
                op_name="entry_rmsnorm",
                implementation_status="descriptor_codegen_ready",
                required_codegen_steps=(
                    "entry_rmsnorm_descriptor",
                    "entry_rmsnorm_pre_block_internal_buffers",
                ),
                description=(
                    "Per-brick entry RMSNorm descriptor applied to the "
                    "first in-region brick's hidden input"
                ),
                production_source=(
                    "cppmega_mlx.runtime.path_c_fusion_schedules:"
                    "_emit_entry_rmsnorm_source"
                ),
                production_fragment_status="region_fragment_inlined_unoptimized",
                production_fragment_reason=(
                    "entry RMSNorm is emitted into the fused TileLang region "
                    "with an explicit norm-weight ABI input, but it is still "
                    "scalar descriptor code rather than the production vector "
                    "RMSNorm schedule"
                ),
                preferred_internal_buffer_policy=(
                    DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN
                ),
                preferred_loop_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_codegen_step=(
                    "entry_rmsnorm_row_phased_production_fragment"
                ),
                production_fragment_inlined_reason=(
                    "row-phased descriptor codegen emits the per-brick entry "
                    "RMSNorm full-row sum-of-squares reduction, inverse RMS, "
                    "and weighted normalized output without full activation "
                    "staging"
                ),
                fragment_emitter=_emit_entry_rmsnorm_source,
            ),
            PathCBrickScheduleDescriptor(
                op_name="entry_rmsnorm_bwd",
                implementation_status="descriptor_codegen_ready",
                required_codegen_steps=(
                    "entry_rmsnorm_bwd_recompute_descriptor",
                ),
                supports_backward=False,
                description=(
                    "Per-brick entry RMSNorm recompute backward descriptor"
                ),
                production_source=(
                    "cppmega_mlx.runtime.path_c_fusion_schedules:"
                    "_append_row_phased_entry_rmsnorm_bwd_body"
                ),
                production_fragment_status="region_fragment_inlined_unoptimized",
                production_fragment_reason=(
                    "entry RMSNorm backward has an explicit descriptor; "
                    "row-phased descriptor codegen now recomputes full-hidden "
                    "forward state and accumulates norm-weight gradients, but "
                    "it is still tracked as a policy-gated production fragment "
                    "until row-local hidden scheduling is selected"
                ),
                preferred_internal_buffer_policy=(
                    DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN
                ),
                preferred_loop_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_codegen_step=(
                    "entry_rmsnorm_bwd_row_phased_recompute_fragment"
                ),
                production_fragment_inlined_reason=(
                    "row-phased descriptor codegen recomputes entry RMSNorm "
                    "state from forward inputs and accumulates norm-weight "
                    "grads without full activation staging"
                ),
                fragment_emitter=None,
            ),
            PathCBrickScheduleDescriptor(
                op_name="mamba3_mimo",
                implementation_status="descriptor_codegen_ready",
                required_codegen_steps=(
                    "mamba3_scan_descriptor",
                    "mamba3_scan_fwd_internal_buffers",
                    "mamba3_row_phased_dense_in_projection",
                    "mamba3_row_phased_causal_conv_ring_history",
                    "mamba3_row_phased_bc_norm_rope",
                    "mamba3_row_phased_state_recurrence",
                    "mamba3_row_phased_gate_out_projection",
                ),
                description="Mamba3 scan brick descriptor",
                production_source=(
                    "cppmega_mlx.nn._tilelang.mamba3_path_c:"
                    "mamba3_mimo_fwd_path_c/mamba3_mimo_bwd_path_c"
                ),
                production_fragment_status="region_fragment_inlined_unoptimized",
                production_fragment_reason=(
                    "the fused train-block region emits a Mamba3 descriptor "
                    "fragment; it only becomes production-inlined when the "
                    "row-local hidden policy is selected so the schedule can "
                    "carry scan state without staging full projected x/B/C/z/A/dt "
                    "activation tensors"
                ),
                preferred_internal_buffer_policy=(
                    DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN
                ),
                preferred_loop_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_codegen_step=(
                    "mamba3_row_phased_fused_project_conv_scan_out"
                ),
                production_fragment_inlined_reason=(
                    "row-phased descriptor codegen fuses Mamba3 dense input "
                    "projection, causal depthwise convolution, B/C norm+RoPE, "
                    "scan-state recurrence, gate, and output projection from "
                    "the block-level ABI without full activation staging"
                ),
                fragment_emitter=_emit_mamba3_mimo_source,
            ),
            PathCBrickScheduleDescriptor(
                op_name="mamba3_chunk_scan_combine",
                implementation_status="descriptor_codegen_ready",
                required_codegen_steps=(
                    "mamba3_chunk_scan_combine_descriptor",
                    "mamba3_chunk_scan_combine_grid_kernel",
                ),
                description=(
                    "Mamba3 chunked SSD scan+combine (F2) brick descriptor; "
                    "GRID-launched scan-core, delegates to "
                    "chunk_scan_fwd_metal_prim (shadow-registered, Stage 1)"
                ),
                production_source=(
                    "cppmega_mlx.nn._tilelang.mamba3_chunked_scan_core:"
                    "build_chunk_scan_combine_metal/chunk_scan_fwd_metal_prim"
                ),
                production_fragment_status="region_fragment_inlined_unoptimized",
                production_fragment_reason=(
                    "F2 scan+combine is a grid-launched kernel (own T.Kernel grid) "
                    "delegating to the proven chunk_scan_fwd_metal_prim core; it is "
                    "shadow-registered so its op-name signature resolves via select "
                    "without blocking, while the live mamba3 forward still emits the "
                    "serial scan. Chain-template wiring of the F0/F1 handoff is Stage 2"
                ),
                # F2 is a grid kernel, NOT row-phased single-thread; keep the
                # default flat loop policy so the row-phased mamba template gating
                # (_is_row_phased_mamba3) does not swallow it (design §3.2).
                preferred_loop_policy=DESCRIPTOR_LOOP_POLICY_FLAT,
                fragment_emitter=_emit_mamba3_chunk_scan_combine_source,
            ),
            PathCBrickScheduleDescriptor(
                op_name="mamba3_chunk_precompute",
                implementation_status="descriptor_codegen_ready",
                required_codegen_steps=(
                    "mamba3_chunk_precompute_descriptor",
                    "mamba3_chunk_precompute_grid_kernel",
                ),
                description=(
                    "Mamba3 chunked SSD precompute (F0) brick descriptor; "
                    "GRID-launched, NO scan dependency; delegates to "
                    "chunk_precompute_fwd_metal_prim (forms cb, dA_cumsum, "
                    "summary_states for the F1/F2 handoff)"
                ),
                production_source=(
                    "cppmega_mlx.nn._tilelang.mamba3_chunked_precompute_core:"
                    "build_chunk_precompute_metal/chunk_precompute_fwd_metal_prim"
                ),
                production_fragment_status="region_fragment_inlined_unoptimized",
                production_fragment_reason=(
                    "F0 precompute is a grid-launched kernel (own T.Kernel grid) "
                    "delegating to chunk_precompute_fwd_metal_prim; it writes the "
                    "caller-owned cb/dA_cumsum/summary_states handoff buffers the "
                    "F1 inter-chunk recurrence and F2 scan+combine consume. "
                    "Validated chained vs serial forward by "
                    "tests/test_mamba3_chained_forward_f0f1f2.py (Stage 2)"
                ),
                # F0 is a grid kernel, NOT row-phased single-thread; keep the
                # default flat loop policy so the row-phased mamba template gating
                # does not swallow it (design §3.2).
                preferred_loop_policy=DESCRIPTOR_LOOP_POLICY_FLAT,
                fragment_emitter=_emit_mamba3_chunk_precompute_source,
            ),
            PathCBrickScheduleDescriptor(
                op_name="mamba3_inter_chunk_recur",
                implementation_status="descriptor_codegen_ready",
                required_codegen_steps=(
                    "mamba3_inter_chunk_recur_descriptor",
                    "mamba3_inter_chunk_recur_grid_kernel",
                ),
                description=(
                    "Mamba3 chunked SSD inter-chunk recurrence (F1) brick "
                    "descriptor; the ONLY O(S/C) sequential stage; delegates to "
                    "inter_chunk_recur_fwd_metal_prim (summary_states + h0 -> "
                    "prev_states, final_state)"
                ),
                production_source=(
                    "cppmega_mlx.nn._tilelang.mamba3_chunked_precompute_core:"
                    "build_inter_chunk_recur_metal/inter_chunk_recur_fwd_metal_prim"
                ),
                production_fragment_status="region_fragment_inlined_unoptimized",
                production_fragment_reason=(
                    "F1 inter-chunk recurrence is a grid-launched kernel (own "
                    "T.Kernel grid, batch*nheads threadgroups; the chunk axis is "
                    "the only serial carry) delegating to "
                    "inter_chunk_recur_fwd_metal_prim; it reads the F0 "
                    "summary_states/dA_cumsum handoff plus h0 and writes the "
                    "prev_states the F2 scan+combine consumes (fp32 for precision, "
                    "design §3.3). Validated chained vs serial forward by "
                    "tests/test_mamba3_chained_forward_f0f1f2.py (Stage 2)"
                ),
                preferred_loop_policy=DESCRIPTOR_LOOP_POLICY_FLAT,
                fragment_emitter=_emit_mamba3_inter_chunk_recur_source,
            ),
            # --- Stage 3 BACKWARD chunked descriptors (B2 / B1 / B0) --- #
            # The analytic transpose of the forward F2/F1/F0 (design §2/§7 Stage 3).
            # Each is a single-entry GRID kernel; the compile-site interpose
            # substitutes the real build_*_bwd_metal grid prim when the flag is ON.
            PathCBrickScheduleDescriptor(
                op_name="mamba3_chunk_scan_combine_bwd",
                implementation_status="descriptor_codegen_ready",
                required_codegen_steps=(
                    "mamba3_chunk_scan_combine_bwd_descriptor",
                    "mamba3_chunk_scan_combine_bwd_grid_kernel",
                ),
                description=(
                    "Mamba3 chunked SSD scan+combine BACKWARD (B2) brick "
                    "descriptor; GRID-launched output/Y transpose; delegates to "
                    "chunk_scan_combine_bwd_metal_prim (dC/dx/dz/dchunk_states/"
                    "dinp/dA_cumsum_y/dD)"
                ),
                production_source=(
                    "cppmega_mlx.nn._tilelang.mamba3_chunked_backward_core:"
                    "build_chunk_scan_combine_bwd_metal/"
                    "chunk_scan_combine_bwd_metal_prim"
                ),
                production_fragment_status="region_fragment_inlined_unoptimized",
                production_fragment_reason=(
                    "B2 scan+combine backward is a grid-launched kernel (own "
                    "T.Kernel grid) delegating to chunk_scan_combine_bwd_metal_prim "
                    "— the analytic transpose of the forward F2. Validated vs the "
                    "MLX backward proto (worst grad 3.68e-4) by "
                    "scratch/test_b0b1b2_metal_vs_proto.py (Stage 3)"
                ),
                supports_backward=False,
                preferred_loop_policy=DESCRIPTOR_LOOP_POLICY_FLAT,
                fragment_emitter=_emit_mamba3_chunk_scan_combine_bwd_source,
            ),
            PathCBrickScheduleDescriptor(
                op_name="mamba3_inter_chunk_recur_bwd",
                implementation_status="descriptor_codegen_ready",
                required_codegen_steps=(
                    "mamba3_inter_chunk_recur_bwd_descriptor",
                    "mamba3_inter_chunk_recur_bwd_grid_kernel",
                ),
                description=(
                    "Mamba3 chunked SSD inter-chunk recurrence BACKWARD (B1) brick "
                    "descriptor; the NEW O(S/C) REVERSE upper-tri combiner; "
                    "delegates to inter_chunk_recur_bwd_metal_prim (reuses forward "
                    "prev_states -> dstates/dh0/dA_cumsum_tail; no 8x replay)"
                ),
                production_source=(
                    "cppmega_mlx.nn._tilelang.mamba3_chunked_backward_core:"
                    "build_inter_chunk_recur_bwd_metal/"
                    "inter_chunk_recur_bwd_metal_prim"
                ),
                production_fragment_status="region_fragment_inlined_unoptimized",
                production_fragment_reason=(
                    "B1 inter-chunk recurrence backward is the REVERSE O(S/C) "
                    "upper-tri combiner (own T.Kernel(batch,nheads) grid; the chunk "
                    "axis is the only serial REVERSE carry) — the adjoint of the "
                    "forward F1 lower-tri recurrence. REUSES the forward-"
                    "materialized prev_states (the 8x checkpoint-replay elimination, "
                    "design §3/§6). Validated by scratch/test_b0b1b2_metal_vs_proto.py"
                ),
                supports_backward=False,
                preferred_loop_policy=DESCRIPTOR_LOOP_POLICY_FLAT,
                fragment_emitter=_emit_mamba3_inter_chunk_recur_bwd_source,
            ),
            PathCBrickScheduleDescriptor(
                op_name="mamba3_chunk_precompute_bwd",
                implementation_status="descriptor_codegen_ready",
                required_codegen_steps=(
                    "mamba3_chunk_precompute_bwd_descriptor",
                    "mamba3_chunk_precompute_bwd_grid_kernel",
                ),
                description=(
                    "Mamba3 chunked SSD precompute BACKWARD (B0) brick descriptor; "
                    "GRID-launched precompute transpose; delegates to "
                    "chunk_precompute_bwd_metal_prim (decay_states transpose + dinp "
                    "assembly + cumsum/segsum VJP -> dx/dB/dlog_decay/ddt)"
                ),
                production_source=(
                    "cppmega_mlx.nn._tilelang.mamba3_chunked_backward_core:"
                    "build_chunk_precompute_bwd_metal/"
                    "chunk_precompute_bwd_metal_prim"
                ),
                production_fragment_status="region_fragment_inlined_unoptimized",
                production_fragment_reason=(
                    "B0 precompute backward is a grid-launched kernel (own T.Kernel "
                    "grid) delegating to chunk_precompute_bwd_metal_prim — the "
                    "transpose of the forward F0 precompute, assembling the final "
                    "input grads. Validated by scratch/test_b0b1b2_metal_vs_proto.py"
                ),
                supports_backward=False,
                preferred_loop_policy=DESCRIPTOR_LOOP_POLICY_FLAT,
                fragment_emitter=_emit_mamba3_chunk_precompute_bwd_source,
            ),
            PathCBrickScheduleDescriptor(
                op_name="mamba3_mimo_bwd",
                implementation_status="descriptor_codegen_ready",
                required_codegen_steps=(
                    "mamba3_mimo_bwd_descriptor",
                    "mamba3_mimo_bwd_final_gradient_owner_outputs",
                ),
                supports_backward=False,
                description="Mamba3 MIMO backward descriptor",
                production_source=(
                    "cppmega_mlx.nn._tilelang.mamba3_path_c:"
                    "mamba3_mimo_bwd_path_c"
                ),
                production_fragment_status="region_fragment_inlined_unoptimized",
                production_fragment_reason=(
                    "Mamba3 backward now has an explicit descriptor tied to the "
                    "Path C backward source and emits stage-specific "
                    "project/conv/dt/state/out gradient owner outputs, but it "
                    "is still scalar descriptor code; the existing Path C "
                    "backward kernel consumes scan-level dy/x/B/C/z/A/dt/D/h0 "
                    "tensors rather than block-level model weights"
                ),
                preferred_internal_buffer_policy=(
                    DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN
                ),
                preferred_loop_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_codegen_step=(
                    "mamba3_mimo_bwd_row_phased_weight_state_recompute"
                ),
                production_fragment_inlined_reason=(
                    "row-phased descriptor codegen recomputes Mamba3 backward "
                    "owner outputs from block-level weights, state, h0, and "
                    "row-local hidden gradients without full activation staging"
                ),
                fragment_emitter=None,
            ),
            PathCBrickScheduleDescriptor(
                op_name="residual_rmsnorm",
                implementation_status="descriptor_codegen_ready",
                required_codegen_steps=(
                    "residual_rmsnorm_descriptor",
                    "residual_rmsnorm_bridge_internal_buffers",
                ),
                description="Residual bridge plus RMSNorm descriptor",
                production_source=(
                    "cppmega_mlx.runtime.path_c_fusion_schedules:"
                    "_emit_residual_rmsnorm_source"
                ),
                production_fragment_status="region_fragment_inlined_unoptimized",
                production_fragment_reason=(
                    "residual/RMSNorm bridge is now emitted into the fused "
                    "TileLang region with an explicit norm-weight ABI input, "
                    "but it is still scalar descriptor code rather than the "
                    "production vector RMSNorm schedule"
                ),
                preferred_internal_buffer_policy=(
                    DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN
                ),
                preferred_loop_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_codegen_step=(
                    "residual_rmsnorm_row_phased_production_fragment"
                ),
                production_fragment_inlined_reason=(
                    "row-phased descriptor codegen emits the residual bridge, "
                    "full-row sum-of-squares reduction, inverse RMS, and "
                    "weighted normalized output without full activation staging"
                ),
                fragment_emitter=_emit_residual_rmsnorm_source,
            ),
            PathCBrickScheduleDescriptor(
                op_name="residual_rmsnorm_bwd",
                implementation_status="descriptor_codegen_ready",
                required_codegen_steps=(
                    "residual_rmsnorm_bwd_recompute_descriptor",
                ),
                supports_backward=False,
                description="Residual bridge plus RMSNorm recompute backward descriptor",
                production_source=(
                    "cppmega_mlx.runtime.path_c_fusion_schedules:"
                    "_append_row_phased_residual_rmsnorm_bwd_body"
                ),
                production_fragment_status="region_fragment_inlined_unoptimized",
                production_fragment_reason=(
                    "residual/RMSNorm backward has an explicit descriptor; "
                    "row-phased descriptor codegen now recomputes full-hidden "
                    "forward state and accumulates norm-weight gradients, but "
                    "it is still tracked as a policy-gated production fragment "
                    "until row-local hidden scheduling is selected"
                ),
                preferred_internal_buffer_policy=(
                    DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN
                ),
                preferred_loop_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_codegen_step=(
                    "residual_rmsnorm_bwd_row_phased_recompute_fragment"
                ),
                production_fragment_inlined_reason=(
                    "row-phased descriptor codegen recomputes residual/RMSNorm "
                    "state from forward inputs and accumulates norm-weight grads "
                    "without full activation staging"
                ),
                fragment_emitter=None,
            ),
            PathCBrickScheduleDescriptor(
                op_name="m2rnn",
                implementation_status="descriptor_codegen_ready",
                required_codegen_steps=(
                    "m2rnn_descriptor",
                    "m2rnn_packed_post_internal_buffers",
                    "m2rnn_row_phased_dense_in_projection",
                    "m2rnn_row_phased_causal_conv_ring_history",
                    "m2rnn_row_phased_state_recurrence",
                    "m2rnn_row_phased_gate_norm_out_projection",
                ),
                description="M2RNN packed post descriptor",
                production_source=(
                    "cppmega_mlx.nn._tilelang.m2rnn_path_c:"
                    "m2rnn_apply_mapped_packed_post_with_state_path_c"
                ),
                production_fragment_status="region_fragment_inlined_unoptimized",
                production_fragment_reason=(
                    "the fused train-block region emits an M2RNN descriptor "
                    "fragment; it only becomes production-inlined when the "
                    "row-local hidden policy is selected so the schedule can "
                    "carry recurrent state without staging full projected, "
                    "conv_input, or post tensors"
                ),
                preferred_internal_buffer_policy=(
                    DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN
                ),
                preferred_loop_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_codegen_step=(
                    "m2rnn_row_phased_fused_project_conv_recurrence_post"
                ),
                production_fragment_inlined_reason=(
                    "row-phased descriptor codegen fuses M2RNN dense input "
                    "projection, causal depthwise convolution, mapped state "
                    "recurrence, gate/RMSNorm, and output projection from the "
                    "block-level ABI without full activation staging"
                ),
                fragment_emitter=_emit_m2rnn_source,
            ),
            PathCBrickScheduleDescriptor(
                op_name="m2rnn_bwd",
                implementation_status="descriptor_codegen_ready",
                required_codegen_steps=(
                    "m2rnn_bwd_descriptor",
                    "m2rnn_bwd_final_gradient_owner_outputs",
                ),
                supports_backward=False,
                description="M2RNN packed backward descriptor",
                production_source=(
                    "cppmega_mlx.nn._tilelang.m2rnn_path_c:"
                    "m2rnn_mapped_packed_bwd_path_c"
                ),
                production_fragment_status="region_fragment_inlined_unoptimized",
                production_fragment_reason=(
                    "M2RNN backward now has an explicit descriptor tied to the "
                    "Path C packed backward source and emits stage-specific "
                    "project/conv/recurrent/post gradient owner outputs, but "
                    "it is still scalar descriptor code; the existing Path C "
                    "backward kernels consume mapped-packed recurrent/post "
                    "intermediates rather than block-level model weights"
                ),
                preferred_internal_buffer_policy=(
                    DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN
                ),
                preferred_loop_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_codegen_step=(
                    "m2rnn_bwd_row_phased_weight_state_recompute"
                ),
                production_fragment_inlined_reason=(
                    "row-phased descriptor codegen recomputes M2RNN backward "
                    "owner outputs from block-level projection, convolution, "
                    "state, gate, post, h0, and row-local hidden gradients "
                    "without full activation staging"
                ),
                fragment_emitter=None,
            ),
            PathCBrickScheduleDescriptor(
                op_name="attention_qkv_projection",
                implementation_status="descriptor_codegen_ready",
                required_codegen_steps=(
                    "attention_qkv_projection_descriptor",
                    "attention_qkv_projection_fp8_prepare_uint8",
                ),
                description="Attention Q/KV projection and FP8 prepare descriptor",
                production_source=(
                    "cppmega_mlx.nn.attention:"
                    "CausalSelfAttention.prepare_sparse_mla_fp8"
                ),
                production_fragment_status="region_fragment_inlined_unoptimized",
                production_fragment_reason=(
                    "the fused train-block region emits an attention projection "
                    "descriptor fragment with the real ABI, but it remains "
                    "policy-gated until row-phased hidden scheduling is selected"
                ),
                preferred_internal_buffer_policy=(
                    DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN
                ),
                preferred_loop_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_codegen_step=(
                    "attention_qkv_projection_row_phased_rope_fp8_fragment"
                ),
                production_fragment_inlined_reason=(
                    "row-phased descriptor codegen emits real q/sparse-kv "
                    "dot-products, split-half RoPE, per-head FP8 scaling, "
                    "uint8 FP8 storage, and full-window causal sparse indices "
                    "without full activation staging"
                ),
                fragment_emitter=_emit_attention_qkv_projection_source,
            ),
            PathCBrickScheduleDescriptor(
                op_name="attention_qkv_projection_bwd",
                implementation_status="descriptor_codegen_ready",
                required_codegen_steps=(
                    "attention_qkv_projection_bwd_descriptor",
                    "attention_qkv_projection_bwd_weight_bias_gradients",
                ),
                supports_backward=False,
                description="Attention Q/KV projection backward descriptor",
                production_source=(
                    "cppmega_mlx.nn.attention:"
                    "CausalSelfAttention.prepare_sparse_mla_fp8"
                ),
                production_fragment_status="region_fragment_inlined_unoptimized",
                production_fragment_reason=(
                    "attention Q/KV projection backward now has an explicit "
                    "descriptor and emits stage-specific q/kv/RoPE gradient "
                    "owner outputs, but it is still scalar descriptor code "
                    "rather than the production q/kv weight, bias, RoPE, and "
                    "FP8-prepare backward schedule"
                ),
                preferred_internal_buffer_policy=(
                    DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN
                ),
                preferred_loop_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_codegen_step=(
                    "attention_qkv_projection_bwd_row_phased_weight_bias_rope"
                ),
                production_fragment_inlined_reason=(
                    "row-phased descriptor codegen emits attention Q/KV "
                    "projection backward weight, bias, hidden, and RoPE owner "
                    "gradients from block-level ABI and row-local prepared-FP8 "
                    "gradients without full activation staging"
                ),
                fragment_emitter=None,
            ),
            PathCBrickScheduleDescriptor(
                op_name="sparse_mla_fp8_apply",
                implementation_status="descriptor_codegen_ready",
                required_codegen_steps=(
                    "sparse_mla_fp8_apply_descriptor",
                    "sparse_mla_fp8_apply_owner_output",
                    "sparse_mla_fp8_apply_softmax_lse_out_proj",
                    "sparse_mla_fp8_apply_lse_reuses_softmax_stats",
                    "sparse_mla_fp8_apply_row_topk_indices_cache",
                    "sparse_mla_fp8_apply_invalid_index_sentinel",
                ),
                supports_backward=False,
                description="Sparse MLA FP8 apply descriptor",
                production_source=(
                    "cppmega_mlx.nn._tilelang.sparse_mla_fp8_path_c:"
                    "make_fp8_sparse_mla_prepare_kernel/"
                    "sparse_mla_fp8_path_c_apply"
                ),
                production_fragment_status="region_fragment_inlined_unoptimized",
                production_fragment_reason=(
                    "the fused train-block region emits a Sparse-MLA FP8 apply "
                    "descriptor fragment; it becomes production-inlined when "
                    "row-local hidden scheduling is selected so q/kv FP8 "
                    "prepare, sparse top-k scores, softmax stats, LSE, and "
                    "attention out-projection stay in the train-block region"
                ),
                preferred_internal_buffer_policy=(
                    DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN
                ),
                preferred_loop_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_codegen_step=(
                    "sparse_mla_fp8_apply_row_phased_prepared_apply"
                ),
                production_fragment_inlined_reason=(
                    "row-phased descriptor codegen emits prepared-FP8 sparse "
                    "attention apply with score max/sumexp, row-local cached "
                    "top-k indices, weighted KV values, invalid-index score "
                    "sentinels, attention out-projection, and LSE from the same "
                    "softmax stats without full activation staging"
                ),
                fragment_emitter=_emit_sparse_mla_fp8_apply_source,
            ),
            PathCBrickScheduleDescriptor(
                op_name="sparse_mla_fp8_apply_bwd",
                implementation_status="descriptor_codegen_ready",
                required_codegen_steps=(
                    "sparse_mla_fp8_apply_bwd_descriptor",
                    "sparse_mla_fp8_apply_bwd_prepared_grad_owner_outputs",
                    "sparse_mla_fp8_apply_bwd_out_proj_grad_owner_outputs",
                ),
                supports_backward=False,
                description="Sparse MLA FP8 apply backward descriptor",
                production_source=(
                    "cppmega_mlx.nn._tilelang.sparse_mla_fp8_path_c:"
                    "sparse_mla_fp8_path_c_apply VJP"
                ),
                production_fragment_status="region_fragment_inlined_unoptimized",
                production_fragment_reason=(
                    "Sparse-MLA FP8 apply backward now has an explicit "
                    "descriptor and emits prepared q/kv FP8, scale, and "
                    "attention out-projection owner gradients inside the "
                    "train-block graph, but it is still scalar descriptor code "
                    "rather than the production softmax/out-projection VJP"
                ),
                preferred_internal_buffer_policy=(
                    DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN
                ),
                preferred_loop_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_codegen_step=(
                    "sparse_mla_fp8_apply_bwd_row_phased_prepared_grad"
                ),
                production_fragment_inlined_reason=(
                    "row-phased descriptor codegen emits Sparse-MLA apply "
                    "backward owner outputs for prepared q/kv FP8 values, "
                    "prepared scales, and attention out-projection gradients "
                    "without exposing q/kv prepared gradients as external ABI"
                ),
                fragment_emitter=None,
            ),
        )
    )


def make_path_c_descriptor_schedule_template(
    brick_descriptors: Sequence[PathCBrickScheduleDescriptor],
    *,
    entry_symbol: str | None = None,
    buffer_extent: int = DESCRIPTOR_DEFAULT_BUFFER_EXTENT,
    shape_env: PathCModelShapeEnv | None = None,
    internal_buffer_policy: str = DESCRIPTOR_INTERNAL_BUFFER_POLICY_SCALAR_LOCAL,
    loop_policy: str = DESCRIPTOR_LOOP_POLICY_FLAT,
    physical_abi_policy: str = DESCRIPTOR_PHYSICAL_ABI_POLICY_DIRECT,
    train_step_output_abi: bool = False,
    max_rows_per_launch: int | None = None,
    row_dispatch_mode: str = DESCRIPTOR_DEFAULT_ROW_DISPATCH_MODE,
    rows_per_kernel_launch: int = DESCRIPTOR_ROWS_PER_KERNEL_LAUNCH,
    execution_stage: str = DESCRIPTOR_EXECUTION_STAGE_ALL,
    active_node_names: Sequence[str] | None = None,
) -> Callable[[Any], Any]:
    """Return a schedule template generated from brick descriptors."""

    descriptors = tuple(brick_descriptors)
    if not descriptors:
        raise ValueError("descriptor schedule template requires at least one brick")
    extent = _validated_buffer_extent(buffer_extent)
    validated_internal_buffer_policy = _validated_internal_buffer_policy(
        internal_buffer_policy
    )
    validated_loop_policy = _validated_loop_policy(loop_policy)
    validated_physical_abi_policy = _validated_physical_abi_policy(
        physical_abi_policy
    )
    validated_max_rows_per_launch = _validated_max_rows_per_launch(
        max_rows_per_launch
    )
    validated_row_dispatch_mode = _validated_row_dispatch_mode(row_dispatch_mode)
    validated_rows_per_kernel_launch = _validated_rows_per_kernel_launch(
        rows_per_kernel_launch
    )
    validated_execution_stage = _validated_execution_stage(execution_stage)
    active_node_names_tuple = (
        tuple(str(name) for name in active_node_names)
        if active_node_names is not None
        else None
    )

    def descriptor_schedule_template(template_region: Any) -> Any:
        return build_path_c_descriptor_prim_func(
            template_region,
            descriptors,
            entry_symbol=entry_symbol,
            buffer_extent=extent,
            shape_env=shape_env,
            internal_buffer_policy=validated_internal_buffer_policy,
            loop_policy=validated_loop_policy,
            physical_abi_policy=validated_physical_abi_policy,
            train_step_output_abi=bool(train_step_output_abi),
            max_rows_per_launch=validated_max_rows_per_launch,
            row_dispatch_mode=validated_row_dispatch_mode,
            rows_per_kernel_launch=validated_rows_per_kernel_launch,
            execution_stage=validated_execution_stage,
            active_node_names=active_node_names_tuple,
        )

    descriptor_schedule_metadata = cast(Any, descriptor_schedule_template)
    descriptor_schedule_metadata._cppmega_path_c_schedule_generator = (
        PATH_C_DESCRIPTOR_SCHEDULE_GENERATOR
    )
    descriptor_schedule_metadata._cppmega_path_c_brick_ops = tuple(
        descriptor.op_name for descriptor in descriptors
    )
    descriptor_schedule_metadata._cppmega_path_c_buffer_extent = extent
    descriptor_schedule_metadata._cppmega_path_c_shape_env = shape_env
    descriptor_schedule_metadata._cppmega_path_c_internal_buffer_policy = (
        validated_internal_buffer_policy
    )
    descriptor_schedule_metadata._cppmega_path_c_loop_policy = validated_loop_policy
    descriptor_schedule_metadata._cppmega_path_c_physical_abi_policy = (
        validated_physical_abi_policy
    )
    descriptor_schedule_metadata._cppmega_path_c_max_rows_per_launch = (
        validated_max_rows_per_launch
    )
    descriptor_schedule_metadata._cppmega_path_c_row_dispatch_mode = (
        validated_row_dispatch_mode
    )
    descriptor_schedule_metadata._cppmega_path_c_rows_per_kernel_launch = (
        validated_rows_per_kernel_launch
    )
    descriptor_schedule_metadata._cppmega_path_c_execution_stage = (
        validated_execution_stage
    )
    descriptor_schedule_metadata._cppmega_path_c_active_node_names = (
        active_node_names_tuple or ()
    )
    descriptor_schedule_metadata._cppmega_path_c_workspace_edge_buffers = (
        ("q_fp8", "q_scale", "kv_fp8", "kv_scale")
        if _descriptor_chain_uses_kv_history_workspace(descriptors)
        else ()
    )
    descriptor_schedule_metadata._cppmega_path_c_train_step_output_abi_enabled = (
        bool(train_step_output_abi)
    )
    return descriptor_schedule_template


def mamba3_fp8_train_fusion_schedule_template(region: Any) -> Any:
    """Generate the explicit Mamba3 acceptance schedule for ``region``."""

    resolved_region = _require_path_c_region_graph(
        region,
        function_name="mamba3_fp8_train_fusion_schedule_template",
    )
    target = _profiled_descriptor_target_for_region(
        resolved_region,
        _mamba3_fp8_train_acceptance_profile(),
    )
    return target.schedule_template(resolved_region)


def mamba3_fp8_train_prototype_schedule_template(region: Any) -> Any:
    """Generate the explicit prototype Mamba3 schedule for ``region``."""

    resolved_region = _require_path_c_region_graph(
        region,
        function_name="mamba3_fp8_train_prototype_schedule_template",
    )
    target = _profiled_descriptor_target_for_region(
        resolved_region,
        _mamba3_fp8_train_prototype_profile(),
    )
    return target.schedule_template(resolved_region)


cast(
    Any,
    mamba3_fp8_train_fusion_schedule_template,
)._cppmega_path_c_workspace_edge_buffers = (
    "q_fp8",
    "q_scale",
    "kv_fp8",
    "kv_scale",
)
cast(
    Any,
    mamba3_fp8_train_prototype_schedule_template,
)._cppmega_path_c_workspace_edge_buffers = (
    "q_fp8",
    "q_scale",
    "kv_fp8",
    "kv_scale",
)


def _path_c_fp8_e4m3fn_to_float(T: Any, bits: Any) -> Any:
    bits_u = T.Cast("uint32", bits)
    abs_bits = bits_u & T.uint32(0x7F)
    sign = (bits_u >> T.uint32(7)) & T.uint32(1)
    exp_bits = (bits_u >> T.uint32(3)) & T.uint32(0xF)
    mant_bits = bits_u & T.uint32(0x7)

    mant = T.Cast("float32", mant_bits)
    subnormal = mant * T.float32(1.0 / 512.0)
    normal = (T.float32(1.0) + mant * T.float32(1.0 / 8.0)) * T.exp2(
        T.Cast("float32", T.Cast("int32", exp_bits) - T.int32(7))
    )
    value = T.if_then_else(exp_bits == T.uint32(0), subnormal, normal)
    value = T.if_then_else(abs_bits == T.uint32(0x7F), T.float32(0.0), value)
    return T.if_then_else(sign != T.uint32(0), -value, value)


def _path_c_float_to_fp8_e4m3fn_bits(T: Any, value: Any) -> Any:
    x = T.Cast("float32", value)
    finite_x = T.if_then_else(T.isfinite(x), x, T.float32(0.0))
    sign = T.if_then_else(finite_x < T.float32(0.0), T.uint32(0x80), T.uint32(0))
    ax = T.min(T.abs(finite_x), T.float32(448.0))

    subnormal_mant = T.min(
        T.Cast("uint32", T.round(ax * T.float32(512.0))),
        T.uint32(7),
    )

    normal_ax = T.max(ax, T.float32(1.0 / 64.0))
    exp_unbiased = T.Cast("int32", T.floor(T.log2(normal_ax)))
    exp_bits_i = exp_unbiased + T.int32(7)
    base = T.exp2(T.Cast("float32", exp_unbiased))
    mant = T.Cast(
        "uint32",
        T.round((normal_ax / base - T.float32(1.0)) * T.float32(8.0)),
    )

    mant_carry = mant >= T.uint32(8)
    exp_bits_i = exp_bits_i + T.if_then_else(mant_carry, T.int32(1), T.int32(0))
    exp_bits_i = T.max(T.min(exp_bits_i, T.int32(15)), T.int32(1))
    mant = T.if_then_else(mant_carry, T.uint32(0), mant)
    mant = T.if_then_else(
        exp_bits_i == T.int32(15),
        T.min(mant, T.uint32(6)),
        T.min(mant, T.uint32(7)),
    )
    normal_bits = (T.Cast("uint32", exp_bits_i) << T.uint32(3)) | mant
    magnitude = T.if_then_else(
        ax < T.float32(1.0 / 64.0),
        subnormal_mant,
        normal_bits,
    )
    return T.Cast("uint8", sign | magnitude)


def build_path_c_descriptor_prim_func(
    region: Any,
    brick_descriptors: Sequence[PathCBrickScheduleDescriptor],
    *,
    entry_symbol: str | None = None,
    buffer_extent: int = DESCRIPTOR_DEFAULT_BUFFER_EXTENT,
    shape_env: PathCModelShapeEnv | None = None,
    internal_buffer_policy: str = DESCRIPTOR_INTERNAL_BUFFER_POLICY_SCALAR_LOCAL,
    loop_policy: str = DESCRIPTOR_LOOP_POLICY_FLAT,
    physical_abi_policy: str = DESCRIPTOR_PHYSICAL_ABI_POLICY_DIRECT,
    train_step_output_abi: bool = False,
    max_rows_per_launch: int | None = None,
    row_dispatch_mode: str = DESCRIPTOR_DEFAULT_ROW_DISPATCH_MODE,
    rows_per_kernel_launch: int = DESCRIPTOR_ROWS_PER_KERNEL_LAUNCH,
    execution_stage: str = DESCRIPTOR_EXECUTION_STAGE_ALL,
    active_node_names: Sequence[str] | None = None,
) -> Any:
    """Generate a single-entry TileLang PrimFunc from a descriptor chain."""

    nodes = _node_views_for_region(region)
    descriptors = tuple(brick_descriptors)
    if len(descriptors) != len(nodes):
        raise ValueError(
            "descriptor count must match region node count: "
            f"{len(descriptors)} descriptors for {len(nodes)} nodes"
        )
    _validate_descriptors_match_nodes(nodes, descriptors)
    resolved_shape_env = shape_env or _shape_env_for_region(region)
    validated_internal_buffer_policy = _validated_internal_buffer_policy(
        internal_buffer_policy
    )
    validated_loop_policy = _validated_loop_policy(loop_policy)
    validated_physical_abi_policy = _validated_physical_abi_policy(
        physical_abi_policy
    )
    validated_max_rows_per_launch = _validated_max_rows_per_launch(
        max_rows_per_launch
    )
    validated_row_dispatch_mode = _validated_row_dispatch_mode(row_dispatch_mode)
    validated_rows_per_kernel_launch = _validated_rows_per_kernel_launch(
        rows_per_kernel_launch
    )
    validated_execution_stage = _validated_execution_stage(execution_stage)
    active_node_name_set = (
        frozenset(str(name) for name in active_node_names)
        if active_node_names is not None
        else None
    )
    descriptors = _descriptors_with_policy_fragment_statuses(
        descriptors,
        internal_buffer_policy=validated_internal_buffer_policy,
        loop_policy=validated_loop_policy,
        shape_env=resolved_shape_env,
    )
    row_chunk_count = _row_phased_chunk_count(
        loop_policy=validated_loop_policy,
        shape_env=resolved_shape_env,
        max_rows_per_launch=validated_max_rows_per_launch,
    )
    row_subchunk_count = (
        max(
            1,
            (
                int(validated_max_rows_per_launch)
                + validated_rows_per_kernel_launch
                - 1
            )
            // validated_rows_per_kernel_launch,
        )
        if (
            row_chunk_count is not None
            and validated_max_rows_per_launch is not None
            and validated_row_dispatch_mode == DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS
        )
        else None
    )
    row_phased_launcher_carry_buffers = _row_phased_launcher_carry_buffers_for_nodes(
        nodes,
        shape_env=resolved_shape_env,
        loop_policy=validated_loop_policy,
        max_rows_per_launch=validated_max_rows_per_launch,
        row_dispatch_mode=validated_row_dispatch_mode,
    )
    row_phased_launcher_carry_buffers = _append_unique_names(
        row_phased_launcher_carry_buffers,
        _row_phased_replay_buffers_for_nodes(
            nodes,
            shape_env=resolved_shape_env,
            loop_policy=validated_loop_policy,
        ),
    )
    internal_buffers = _internal_buffers_for_nodes(
        nodes,
        shape_env=resolved_shape_env,
        internal_buffer_policy=validated_internal_buffer_policy,
        loop_policy=validated_loop_policy,
    )
    dtype_by_buffer = {
        name: _buffer_dtype(name, shape_env=resolved_shape_env)
        for node in nodes
        for name in (*node.inputs, *node.outputs)
    }
    external_buffers = _external_buffers_for_nodes(nodes, internal_buffers)
    train_step_loss_source_buffers = _train_step_suffix_loss_source_buffers(
        nodes,
    )
    train_step_computed_output_buffers = _train_step_computed_output_buffers(
        declared=bool(train_step_output_abi),
        shape_env=resolved_shape_env,
        loss_source_buffers=train_step_loss_source_buffers,
    )
    train_step_output_abi_payload = _train_step_output_abi_payload(
        declared=bool(train_step_output_abi),
        computed_logical_outputs=train_step_computed_output_buffers,
    )
    train_step_suffix_loss_input_abi_payload = (
        _train_step_suffix_loss_input_abi_payload(
            declared=bool(train_step_output_abi),
        )
    )
    train_step_suffix_loss_input_buffers = tuple(
        train_step_suffix_loss_input_abi_payload["logical_inputs"]
        if train_step_suffix_loss_input_abi_payload["declared"]
        else ()
    )
    train_step_output_buffers = tuple(
        train_step_output_abi_payload["logical_outputs"]
        if train_step_output_abi_payload["declared"]
        else ()
    )
    train_step_suffix_loss_parameter_grad_buffers = (
        _train_step_suffix_loss_parameter_grad_buffers(
            declared=bool(train_step_output_abi),
        )
    )
    external_buffers_for_abi = _append_unique_names(
        external_buffers,
        (
            *row_phased_launcher_carry_buffers,
            *train_step_suffix_loss_input_buffers,
            *train_step_output_buffers,
            *train_step_suffix_loss_parameter_grad_buffers,
        ),
    )
    extent = _validated_buffer_extent(buffer_extent)
    entry_name = _safe_identifier(
        entry_symbol or getattr(region, "entry_symbol", None) or getattr(region, "name", None)
        or "path_c_descriptor_region"
    )
    internal_buffer_shapes = _internal_buffer_shapes(
        internal_buffers,
        validated_internal_buffer_policy,
        resolved_shape_env,
    )
    shape_by_buffer = {
        name: _buffer_shape(name, extent, resolved_shape_env)
        for name in external_buffers
    }
    dtype_by_buffer = dict(dtype_by_buffer)
    for name in row_phased_launcher_carry_buffers:
        dtype_by_buffer[name] = _buffer_dtype(name, shape_env=resolved_shape_env)
        shape_by_buffer[name] = _buffer_shape(name, extent, resolved_shape_env)
    for name in train_step_suffix_loss_input_buffers:
        dtype_by_buffer[name] = _buffer_dtype(name, shape_env=resolved_shape_env)
        shape_by_buffer[name] = _buffer_shape(name, extent, resolved_shape_env)
    for name in train_step_output_buffers:
        dtype_by_buffer[name] = "float32"
        shape_by_buffer[name] = (1,)
    for name in train_step_suffix_loss_parameter_grad_buffers:
        dtype_by_buffer[name] = _buffer_dtype(name, shape_env=resolved_shape_env)
        shape_by_buffer[name] = _buffer_shape(name, extent, resolved_shape_env)
    loop_extent = _descriptor_loop_extent(
        external_buffers,
        extent,
        resolved_shape_env,
    )
    physical_abi_plan = _physical_abi_plan(
        external_buffers=external_buffers_for_abi,
        shape_by_buffer=shape_by_buffer,
        dtype_by_buffer=dtype_by_buffer,
        buffer_extent=extent,
        loop_extent=loop_extent,
        shape_env=resolved_shape_env,
        physical_abi_policy=validated_physical_abi_policy,
    )
    train_step_loss_cotangent_buffers = _train_step_suffix_loss_cotangent_buffers(
        train_step_loss_source_buffers,
        physical_abi_plan,
    )
    train_step_loss_cotangent_abi_payload = (
        _train_step_loss_cotangent_abi_payload(
            source_logical_buffers=train_step_loss_source_buffers,
            logical_cotangent_buffers=train_step_loss_cotangent_buffers,
            cotangents_computed=(
                "loss" in train_step_computed_output_buffers
                and len(train_step_loss_cotangent_buffers)
                == len(train_step_loss_source_buffers)
            ),
        )
    )
    train_step_suffix_loss_parameter_grad_abi_payload = (
        _train_step_suffix_loss_parameter_grad_abi_payload(
            declared=bool(train_step_output_abi),
            logical_gradient_buffers=train_step_suffix_loss_parameter_grad_buffers,
            physical_abi_plan=physical_abi_plan,
        )
    )
    # --- Mamba3 chunked-scan LIVE delegation interpose (design §3.1/§7) ---
    # When the chunked-scan flag is ON and this segment is a single
    # GRID-launched F0/F1/F2 op, its kernel is the proven build_*_metal grid
    # prim (its own multi-grid T.Kernel cannot be a fragment in the template's
    # single T.Kernel). BYPASS the exec/source path and substitute the real grid
    # prim — the SHADOW fragment markers are NEVER emitted for a live op
    # (RULE #1). Flag OFF (live default): this branch is never taken, the serial
    # mamba3_mimo descriptor source path is byte-identical to today's behaviour.
    if (
        len(nodes) == 1
        and nodes[0].op_name in _MAMBA3_CHUNKED_GRID_DELEGATION_OPS
        and _path_c_mamba3_chunked_scan_enabled()
    ):
        delegated_prim = _mamba3_chunked_grid_delegation_prim(
            op_name=nodes[0].op_name,
            production_source=descriptors[0].production_source,
            shape_env=resolved_shape_env,
        )
        delegated_prim._cppmega_path_c_schedule_generator = (
            PATH_C_DESCRIPTOR_SCHEDULE_GENERATOR
        )
        delegated_prim._cppmega_path_c_brick_ops = (nodes[0].op_name,)
        delegated_prim._cppmega_path_c_shape_env = resolved_shape_env
        delegated_prim._cppmega_path_c_buffer_extent = extent
        delegated_prim._cppmega_path_c_execution_stage = validated_execution_stage
        delegated_prim._cppmega_path_c_mamba3_chunked_grid_delegation = (
            nodes[0].op_name
        )
        # --- Delegated named-buffer ABI (binder reconciliation) ----------------
        # The delegated prim is a COMPILED tilelang JITKernel: its ``.params`` are
        # unnamed positional ``KernelParam`` (dtype+shape), so the direct-chain
        # runtime — which binds caller-owned buffers BY NAME — cannot use them.
        # The region surface node already declares the ordered named buffers for
        # this op, and they POSITIONALLY MATCH the builder's compiled param order
        # 1:1 (verified for all 6 F0/F1/F2/B0/B1/B2 ops: node.inputs are the
        # builder's leading tensor params and node.outputs are its out_idx params,
        # in the SAME order; shapes match). Attach that ordered name list +
        # per-name shapes so path_c_kernel_buffer_order / the route arg-assembly /
        # the pre-step owner allocator can bind the handoff + region buffers
        # positionally. RULE #1: the order mirrors the prim ABI exactly; a
        # mis-ordered slot would silently mis-bind a buffer, so the count is
        # asserted to equal the kernel's device-buffer param count.
        delegated_buffer_order = (
            *(str(name) for name in nodes[0].inputs),
            *(str(name) for name in nodes[0].outputs),
        )
        # Per-name PRODUCER/CONSUMER role for this delegated segment: a name in
        # ``node.outputs`` is WRITTEN here (producer slot), a name in
        # ``node.inputs`` is READ here (consumer slot). The direct-chain owner
        # allocates each handoff buffer at its PRODUCER dtype (model policy), and
        # the route casts the bound buffer to a CONSUMER slot's dtype only when it
        # narrows for a read (e.g. F1 writes ``prev_states`` fp32; F2/B1 read it
        # fp16 — an explicit fp16 cast-copy at bind, mirroring the validated
        # kernel test's ``prev_states.half()``; the fp32 owner buffer is intact).
        delegated_output_names = frozenset(
            str(name) for name in nodes[0].outputs
        )
        delegated_buffer_roles = {
            str(name): ("output" if str(name) in delegated_output_names else "input")
            for name in delegated_buffer_order
        }
        # The compiled JITKernel's device-buffer KernelParams are the AUTHORITATIVE
        # per-slot shapes (the exact ABI the kernel binds). Pair them positionally
        # with the ordered names (scalar params, if any, are skipped — these grid
        # kernels are all-tensor).
        delegated_buffer_params = tuple(
            param
            for param in tuple(getattr(delegated_prim, "params", ()))
            if not (hasattr(param, "is_scalar") and bool(param.is_scalar()))
        )
        # Attach the named-buffer ABI only when the region surface's ordered
        # buffers RECONCILE 1:1 with the kernel's device-buffer param slots. The
        # FULL model surface declares exactly the kernel's inputs+outputs (verified
        # for all 6 F0/F1/F2/B0/B1/B2 ops), so the live chain route gets the ABI.
        # A PARTIAL/stub surface (e.g. a single placeholder output in a kernel-only
        # unit test) does not reconcile — skip the ABI attach there (the prim is
        # still a valid delegated JITKernel for kernel-parity tests). RULE #1 is
        # preserved end-to-end: with no ABI attached, the chain-route binding
        # payload reports the missing/unbound buffers LOUDLY (it never silently
        # mis-binds), and the count-mismatch detail is recorded for diagnosis.
        if len(delegated_buffer_order) == len(delegated_buffer_params):
            delegated_prim._cppmega_path_c_delegated_kernel_buffer_order = (
                delegated_buffer_order
            )
            delegated_prim._cppmega_path_c_delegated_kernel_buffer_shapes = {
                name: tuple(int(dim) for dim in tuple(getattr(param, "shape", ())))
                for name, param in zip(
                    delegated_buffer_order, delegated_buffer_params
                )
            }
            delegated_prim._cppmega_path_c_delegated_kernel_buffer_dtypes = {
                name: str(getattr(param, "dtype", "float32"))
                for name, param in zip(
                    delegated_buffer_order, delegated_buffer_params
                )
            }
            delegated_prim._cppmega_path_c_delegated_kernel_buffer_roles = (
                delegated_buffer_roles
            )
        else:
            delegated_prim._cppmega_path_c_delegated_kernel_buffer_abi_skipped = (
                f"region surface declares {len(delegated_buffer_order)} ordered "
                f"buffers but the compiled grid kernel binds "
                f"{len(delegated_buffer_params)} device-buffer params"
            )
        return delegated_prim

    source, spilled_shared_scratch = _descriptor_prim_func_source(
        entry_name=entry_name,
        nodes=nodes,
        descriptors=descriptors,
        internal_buffers=internal_buffers,
        internal_buffer_shapes=internal_buffer_shapes,
        physical_abi_plan=physical_abi_plan,
        physical_abi_policy=validated_physical_abi_policy,
        internal_buffer_policy=validated_internal_buffer_policy,
        loop_policy=validated_loop_policy,
        max_rows_per_launch=validated_max_rows_per_launch,
        row_dispatch_mode=validated_row_dispatch_mode,
        rows_per_kernel_launch=validated_rows_per_kernel_launch,
        execution_stage=validated_execution_stage,
        active_node_names=active_node_name_set,
        external_buffers=external_buffers_for_abi,
        shape_by_buffer=shape_by_buffer,
        dtype_by_buffer=dtype_by_buffer,
        buffer_extent=extent,
        loop_extent=loop_extent,
        shape_env=resolved_shape_env,
        train_step_computed_output_buffers=train_step_computed_output_buffers,
        train_step_loss_source_buffers=train_step_loss_source_buffers,
        train_step_loss_cotangent_buffers=train_step_loss_cotangent_buffers,
        train_step_loss_parameter_grad_buffers=(
            train_step_suffix_loss_parameter_grad_buffers
            if train_step_suffix_loss_parameter_grad_abi_payload["gradients_computed"]
            else ()
        ),
    )

    import tilelang.language as T
    float_to_fp8_e4m3fn_bits = lambda value: _path_c_float_to_fp8_e4m3fn_bits(
        T,
        value,
    )
    fp8_e4m3fn_to_float = lambda bits: _path_c_fp8_e4m3fn_to_float(T, bits)

    filename = f"<path_c_descriptor_schedule:{entry_name}>"
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    namespace: dict[str, Any] = {
        "T": T,
        "float_to_fp8_e4m3fn_bits": float_to_fp8_e4m3fn_bits,
        "fp8_e4m3fn_to_float": fp8_e4m3fn_to_float,
    }
    exec(compile(source, filename, "exec"), namespace)
    prim_func = namespace[entry_name]
    internal_scratch_abi_buffers = tuple(
        name
        for name, info in spilled_shared_scratch.items()
        if bool(info.get("internal_scratch_abi"))
    )
    internal_scratch_abi_aliases = {
        name: {
            "bank": str(info["bank"]),
            "offset": int(info["offset"]),
            "shape": tuple(info["shape"]),
            "dtype": str(info["dtype"]),
        }
        for name, info in spilled_shared_scratch.items()
        if bool(info.get("internal_scratch_abi"))
        and bool(info.get("coalesced_scratch_bank"))
    }
    if internal_scratch_abi_buffers:
        prim_func = prim_func.with_attr(
            "tl.fusion.internal_scratch_abi_buffers",
            json.dumps(internal_scratch_abi_buffers),
        )
    if internal_scratch_abi_aliases:
        prim_func = prim_func.with_attr(
            "tl.fusion.internal_scratch_abi_aliases",
            json.dumps(internal_scratch_abi_aliases, sort_keys=True),
        )
    prim_func = prim_func.with_attr(
        "tl.fusion.physical_abi.policy",
        validated_physical_abi_policy,
    ).with_attr(
        "tl.fusion.physical_abi.logical_to_physical",
        json.dumps(physical_abi_plan.logical_to_physical, sort_keys=True),
    ).with_attr(
        "tl.fusion.physical_abi.physical_buffer_shapes",
        json.dumps(physical_abi_plan.physical_buffer_shapes, sort_keys=True),
    ).with_attr(
        "tl.fusion.train_step_output_abi",
        json.dumps(train_step_output_abi_payload, sort_keys=True),
    ).with_attr(
        "tl.fusion.train_step_suffix_loss_input_abi",
        json.dumps(train_step_suffix_loss_input_abi_payload, sort_keys=True),
    ).with_attr(
        "tl.fusion.train_step_loss_cotangent_abi",
        json.dumps(train_step_loss_cotangent_abi_payload, sort_keys=True),
    ).with_attr(
        "tl.fusion.train_step_suffix_loss_parameter_grad_abi",
        json.dumps(
            train_step_suffix_loss_parameter_grad_abi_payload,
            sort_keys=True,
        ),
    )
    if validated_max_rows_per_launch is not None:
        prim_func = prim_func.with_attr(
            "tl.fusion.max_rows_per_launch",
            validated_max_rows_per_launch,
        ).with_attr(
            "tl.fusion.row_dispatch_mode",
            validated_row_dispatch_mode,
        )
    prim_func = prim_func.with_attr(
        "tl.fusion.execution_stage",
        validated_execution_stage,
    )
    if row_chunk_count is not None:
        prim_func = prim_func.with_attr(
            "tl.fusion.row_chunk_count",
            row_chunk_count,
        )
        if (
            validated_row_dispatch_mode
            == DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS
        ):
            prim_func = prim_func.with_attr(
                "tl.fusion.row_chunk_index_param",
                DESCRIPTOR_ROW_CHUNK_INDEX_PARAM,
            ).with_attr(
                "tl.fusion.row_subchunk_index_param",
                DESCRIPTOR_ROW_SUBCHUNK_INDEX_PARAM,
            ).with_attr(
                "tl.fusion.rows_per_kernel_launch",
                validated_rows_per_kernel_launch,
            ).with_attr(
                "tl.fusion.row_subchunk_count",
                row_subchunk_count,
            )
    if (
        validated_execution_stage == DESCRIPTOR_EXECUTION_STAGE_ALL
        and any(
            node.op_name.endswith("_bwd")
            and (active_node_name_set is None or node.name in active_node_name_set)
            for node in nodes
        )
    ):
        prim_func = prim_func.with_attr(
            "tl.fusion.backward_gate_param",
            DESCRIPTOR_BACKWARD_GATE_PARAM,
        )
    backward_stage_name_groups = (
        _path_c_backward_stage_name_groups_for_nodes(nodes)
        if (
            validated_execution_stage == DESCRIPTOR_EXECUTION_STAGE_ALL
            and validated_loop_policy == DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN
            and validated_row_dispatch_mode == DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS
            and active_node_name_set is None
        )
        else ()
    )
    if backward_stage_name_groups:
        prim_func = prim_func.with_attr(
            "tl.fusion.backward_stage_index_param",
            DESCRIPTOR_BACKWARD_STAGE_INDEX_PARAM,
        ).with_attr(
            "tl.fusion.backward_stage_count",
            len(backward_stage_name_groups),
        )
    compile_pass_configs = _descriptor_tilelang_compile_pass_configs(
        descriptors,
        loop_policy=validated_loop_policy,
    )
    if compile_pass_configs:
        prim_func = prim_func.with_attr(
            "tilelang_pass_configs",
            compile_pass_configs,
        )
    if validated_loop_policy == DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN:
        prim_func = prim_func.with_attr("tl.fusion.disable_tir_simplify", True)
    written_logical_buffers = _append_unique_names(
        tuple(
            output_name
            for node in nodes
            if (
                (active_node_name_set is None or node.name in active_node_name_set)
                and (
                    validated_execution_stage != DESCRIPTOR_EXECUTION_STAGE_FORWARD
                    or not node.op_name.endswith("_bwd")
                )
                and (
                    validated_execution_stage != DESCRIPTOR_EXECUTION_STAGE_BACKWARD
                    or node.op_name.endswith("_bwd")
                )
            )
            for output_name in node.outputs
        ),
        (
            *row_phased_launcher_carry_buffers,
            *_row_phased_replay_buffers_for_nodes(
                nodes,
                shape_env=resolved_shape_env,
                loop_policy=validated_loop_policy,
            ),
            *train_step_computed_output_buffers,
            *(
                train_step_suffix_loss_parameter_grad_buffers
                if train_step_suffix_loss_parameter_grad_abi_payload[
                    "gradients_computed"
                ]
                else ()
            ),
        ),
    )
    owner_output_param_indices = _descriptor_owner_output_param_indices(
        prim_func,
        physical_abi_plan=physical_abi_plan,
        written_logical_buffers=written_logical_buffers,
        spilled_shared_scratch=spilled_shared_scratch,
    )
    if owner_output_param_indices:
        prim_func = prim_func.with_attr(
            "tilelang_out_idx",
            list(owner_output_param_indices),
        ).with_attr(
            "tilelang_metal_zero_init_output_positions",
            [],
        )
    prim_func._cppmega_path_c_schedule_generator = PATH_C_DESCRIPTOR_SCHEDULE_GENERATOR
    prim_func._cppmega_path_c_brick_ops = tuple(
        descriptor.op_name for descriptor in descriptors
    )
    prim_func._cppmega_path_c_buffer_extent = extent
    prim_func._cppmega_path_c_shape_env = resolved_shape_env
    prim_func._cppmega_path_c_internal_buffer_policy = (
        validated_internal_buffer_policy
    )
    prim_func._cppmega_path_c_loop_policy = validated_loop_policy
    prim_func._cppmega_path_c_physical_abi_policy = validated_physical_abi_policy
    prim_func._cppmega_path_c_max_rows_per_launch = validated_max_rows_per_launch
    prim_func._cppmega_path_c_row_dispatch_mode = validated_row_dispatch_mode
    prim_func._cppmega_path_c_rows_per_kernel_launch = (
        validated_rows_per_kernel_launch
    )
    prim_func._cppmega_path_c_execution_stage = validated_execution_stage
    prim_func._cppmega_path_c_active_node_names = tuple(
        sorted(active_node_name_set or ())
    )
    prim_func._cppmega_path_c_row_chunk_count = row_chunk_count
    prim_func._cppmega_path_c_row_chunk_index_param = (
        DESCRIPTOR_ROW_CHUNK_INDEX_PARAM
        if (
            row_chunk_count is not None
            and validated_row_dispatch_mode
            == DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS
        )
        else None
    )
    prim_func._cppmega_path_c_row_subchunk_index_param = (
        DESCRIPTOR_ROW_SUBCHUNK_INDEX_PARAM
        if (
            row_chunk_count is not None
            and validated_row_dispatch_mode
            == DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS
        )
        else None
    )
    prim_func._cppmega_path_c_row_subchunk_count = row_subchunk_count
    prim_func._cppmega_path_c_backward_gate_param = (
        DESCRIPTOR_BACKWARD_GATE_PARAM
        if (
            validated_execution_stage == DESCRIPTOR_EXECUTION_STAGE_ALL
            and any(
                node.op_name.endswith("_bwd")
                and (
                    active_node_name_set is None
                    or node.name in active_node_name_set
                )
                for node in nodes
            )
        )
        else None
    )
    prim_func._cppmega_path_c_backward_stage_index_param = (
        DESCRIPTOR_BACKWARD_STAGE_INDEX_PARAM if backward_stage_name_groups else None
    )
    prim_func._cppmega_path_c_backward_stage_count = (
        len(backward_stage_name_groups) if backward_stage_name_groups else None
    )
    prim_func._cppmega_path_c_backward_stage_node_groups = backward_stage_name_groups
    prim_func._cppmega_path_c_internal_buffer_shapes = internal_buffer_shapes
    prim_func._cppmega_path_c_buffer_abi_shapes = {
        name: shape_by_buffer[name]
        for name in external_buffers_for_abi
    }
    prim_func._cppmega_path_c_physical_buffer_abi_shapes = dict(
        physical_abi_plan.physical_buffer_shapes
    )
    prim_func._cppmega_path_c_physical_buffer_abi_map = dict(
        physical_abi_plan.logical_to_physical
    )
    prim_func._cppmega_path_c_spilled_shared_scratch_shapes = dict(
        spilled_shared_scratch
    )
    prim_func._cppmega_path_c_internal_scratch_abi_buffers = (
        internal_scratch_abi_buffers
    )
    prim_func._cppmega_path_c_internal_scratch_abi_aliases = dict(
        internal_scratch_abi_aliases
    )
    prim_func._cppmega_path_c_loop_extent = loop_extent
    prim_func._cppmega_path_c_generated_source = source
    prim_func._cppmega_path_c_compile_pass_configs = compile_pass_configs
    prim_func._cppmega_path_c_train_step_output_abi = dict(
        train_step_output_abi_payload
    )
    prim_func._cppmega_path_c_train_step_suffix_loss_input_abi = dict(
        train_step_suffix_loss_input_abi_payload
    )
    prim_func._cppmega_path_c_train_step_suffix_loss_source_buffers = (
        train_step_loss_source_buffers
    )
    prim_func._cppmega_path_c_train_step_loss_cotangent_abi = dict(
        train_step_loss_cotangent_abi_payload
    )
    prim_func._cppmega_path_c_train_step_suffix_loss_parameter_grad_abi = dict(
        train_step_suffix_loss_parameter_grad_abi_payload
    )
    return prim_func


def _descriptor_tilelang_compile_pass_configs(
    descriptors: Sequence[PathCBrickScheduleDescriptor],
    *,
    loop_policy: str,
) -> dict[str, bool]:
    configs: dict[str, bool] = {}
    if loop_policy == DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN:
        configs["tl.disable_thread_storage_sync"] = True
        configs["tirx.merge_static_smem"] = False
        configs["tirx.disable_cse_tir"] = True
    if (
        loop_policy == DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN
        and any(descriptor.op_name.endswith("_bwd") for descriptor in descriptors)
    ):
        configs["tirx.disable_storage_rewrite"] = True
    return configs


def _append_unique_names(
    names: Sequence[str],
    extra_names: Sequence[str],
) -> tuple[str, ...]:
    values = [str(name) for name in names]
    seen = set(values)
    for raw_name in extra_names:
        name = str(raw_name)
        if name in seen:
            continue
        values.append(name)
        seen.add(name)
    return tuple(values)


def _descriptor_owner_output_param_indices(
    prim_func: Any,
    *,
    physical_abi_plan: "_PhysicalAbiPlan",
    written_logical_buffers: Sequence[str],
    spilled_shared_scratch: Mapping[str, Mapping[str, Any]],
) -> tuple[int, ...]:
    """Return only PrimFunc params whose caller-owned storage is written."""

    output_param_names: set[str] = set()
    for raw_name in written_logical_buffers:
        name = str(raw_name)
        placement = physical_abi_plan.logical_to_physical.get(name)
        if isinstance(placement, Mapping):
            bank_name = placement.get("bank")
            if bank_name:
                output_param_names.add(str(bank_name))
            continue
        if name in physical_abi_plan.physical_buffer_shapes:
            output_param_names.add(name)

    for info in spilled_shared_scratch.values():
        if bool(info.get("coalesced_scratch_bank")):
            bank_name = info.get("bank")
            if bank_name:
                output_param_names.add(str(bank_name))
            continue
        param_name = info.get("param_name")
        if param_name:
            output_param_names.add(str(param_name))

    if not output_param_names:
        return ()
    buffer_map = getattr(prim_func, "buffer_map", {}) or {}
    indices: list[int] = []
    for idx, param in enumerate(getattr(prim_func, "params", ())):
        buffer = buffer_map.get(param)
        if buffer is None:
            continue
        name = str(getattr(buffer, "name", None) or getattr(param, "name", param))
        if name in output_param_names:
            indices.append(idx)
    return tuple(indices)


def _row_phased_launcher_carry_buffers_for_nodes(
    nodes: Sequence[_ScheduleNodeView],
    *,
    shape_env: PathCModelShapeEnv | None,
    loop_policy: str,
    max_rows_per_launch: int | None,
    row_dispatch_mode: str,
) -> tuple[str, ...]:
    if (
        shape_env is None
        or loop_policy != DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN
        or max_rows_per_launch is None
        or row_dispatch_mode != DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS
    ):
        return ()
    carries: list[str] = []
    if any(node.op_name == "mamba3_mimo" for node in nodes):
        if int(shape_env.mamba_num_heads) * int(shape_env.mamba_num_rope_angles) > 0:
            carries.append("mamba3_angle_state")
        if max(0, int(shape_env.mamba_conv_kernel) - 1) > 0:
            carries.append("mamba3_conv_state")
    if any(node.op_name == "m2rnn" for node in nodes):
        carries.append("m2rnn_h_state")
        if max(0, int(shape_env.m2rnn_conv_kernel) - 1) > 0:
            carries.append("m2rnn_conv_state")
    return tuple(carries)


def _row_phased_replay_buffers_for_nodes(
    nodes: Sequence[_ScheduleNodeView],
    *,
    shape_env: PathCModelShapeEnv | None,
    loop_policy: str,
) -> tuple[str, ...]:
    if shape_env is None or loop_policy != DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN:
        return ()
    buffers: list[str] = []
    if any(node.op_name in {"mamba3_mimo", "mamba3_mimo_bwd"} for node in nodes):
        buffers.extend(
            (
                "mamba3_h_checkpoint",
                "mamba3_angle_checkpoint",
                "mamba3_angle_grad_state",
            )
        )
    if any(node.op_name in {"m2rnn", "m2rnn_bwd"} for node in nodes):
        buffers.append("m2rnn_h_checkpoint")
    return tuple(buffers)


def _train_step_output_abi_payload(
    *,
    declared: bool,
    computed_logical_outputs: Sequence[str] = (),
) -> dict[str, Any]:
    logical_outputs = _TRAIN_STEP_SCALAR_OUTPUT_ABI_NAMES if declared else ()
    computed_outputs = tuple(
        name
        for name in logical_outputs
        if name in {str(output) for output in computed_logical_outputs}
    )
    pending_outputs = tuple(
        name for name in logical_outputs if name not in set(computed_outputs)
    )
    outputs_computed = bool(declared and logical_outputs and not pending_outputs)
    return {
        "declared": bool(declared),
        "outputs_computed": bool(outputs_computed),
        "computed_logical_outputs": computed_outputs,
        "pending_logical_outputs": pending_outputs,
        "logical_outputs": logical_outputs,
        "reason": _TRAIN_STEP_SCALAR_OUTPUT_ABI_REASON
        if declared and not computed_outputs
        else (
            "train-step scalar ABI computes ntokens in the descriptor body, "
            "but loss remains pending fused suffix codegen"
        )
        if declared and not outputs_computed
        else (
            "train-step scalar ABI slots are generated and populated by fused "
            "suffix loss codegen"
            if declared
            else "train-step scalar ABI slots are not required for this descriptor"
        ),
    }


def _train_step_computed_output_buffers(
    *,
    declared: bool,
    shape_env: PathCModelShapeEnv | None,
    loss_source_buffers: Sequence[str] = (),
) -> tuple[str, ...]:
    if not declared or shape_env is None:
        return ()
    outputs: list[str] = []
    if loss_source_buffers and int(getattr(shape_env, "vocab_size", 0) or 0) > 0:
        outputs.append("loss")
    outputs.append("ntokens")
    return tuple(outputs)


def _train_step_suffix_loss_source_buffers(
    nodes: Sequence[_ScheduleNodeView],
) -> tuple[str, ...]:
    produced = {output for node in nodes for output in node.outputs}
    seed_gradients = sorted(
        input_name
        for node in nodes
        if node.backward == "owner_output"
        for input_name in node.inputs
        if input_name.endswith("_grad") and input_name not in produced
    )
    return tuple(
        name[: -len("_grad")]
        for name in seed_gradients
        if name.endswith("_grad")
    )


def _train_step_suffix_loss_cotangent_buffers(
    source_logical_buffers: Sequence[str],
    physical_abi_plan: _PhysicalAbiPlan,
) -> tuple[str, ...]:
    return tuple(
        cotangent_name
        for source_name in source_logical_buffers
        for cotangent_name in (f"{source_name}_grad",)
        if cotangent_name in physical_abi_plan.logical_to_physical
    )


def _train_step_loss_cotangent_abi_payload(
    *,
    source_logical_buffers: Sequence[str],
    logical_cotangent_buffers: Sequence[str],
    cotangents_computed: bool,
) -> dict[str, Any]:
    source_buffers = tuple(str(name) for name in source_logical_buffers)
    cotangent_buffers = tuple(str(name) for name in logical_cotangent_buffers)
    missing_cotangents = tuple(
        f"{name}_grad"
        for name in source_buffers
        if f"{name}_grad" not in set(cotangent_buffers)
    )
    return {
        "declared": bool(source_buffers),
        "cotangents_computed": bool(cotangents_computed and not missing_cotangents),
        "source_logical_buffers": source_buffers,
        "logical_cotangent_buffers": cotangent_buffers,
        "missing_logical_cotangent_buffers": missing_cotangents,
        "reason": (
            "train-step suffix loss cotangents are generated into backward seed buffers"
            if cotangents_computed and not missing_cotangents
            else "train-step suffix loss cotangents are not required for this descriptor"
            if not source_buffers
            else "train-step suffix loss cotangent buffers are ABI-external replay seeds"
            if not missing_cotangents
            else "train-step suffix loss cotangent seed buffers are missing from the physical ABI"
        ),
    }


def _train_step_suffix_loss_parameter_grad_buffers(
    *,
    declared: bool,
) -> tuple[str, ...]:
    return _TRAIN_STEP_SUFFIX_LOSS_PARAMETER_GRAD_ABI_NAMES if declared else ()


def _train_step_suffix_loss_parameter_grad_abi_payload(
    *,
    declared: bool,
    logical_gradient_buffers: Sequence[str],
    physical_abi_plan: _PhysicalAbiPlan,
) -> dict[str, Any]:
    parameter_buffers = ("final_norm_weight", "lm_head_weight") if declared else ()
    gradient_buffers = tuple(str(name) for name in logical_gradient_buffers)
    missing_gradients = tuple(
        name
        for name in _TRAIN_STEP_SUFFIX_LOSS_PARAMETER_GRAD_ABI_NAMES
        if name not in physical_abi_plan.logical_to_physical
    )
    gradients_computed = bool(
        declared
        and parameter_buffers
        and not missing_gradients
        and all(name in physical_abi_plan.logical_to_physical for name in parameter_buffers)
    )
    return {
        "declared": bool(declared),
        "parameter_logical_buffers": parameter_buffers,
        "logical_gradient_buffers": gradient_buffers,
        "gradients_computed": gradients_computed,
        "missing_logical_gradient_buffers": missing_gradients,
        "reason": (
            "train-step suffix loss parameter gradients are generated for "
            "final_norm_weight and lm_head_weight"
            if gradients_computed
            else "train-step suffix loss parameter gradients are not required for this descriptor"
            if not declared
            else "train-step suffix loss parameter gradient buffers are missing from the physical ABI"
        ),
    }


def _train_step_suffix_loss_input_abi_payload(
    *,
    declared: bool,
) -> dict[str, Any]:
    return {
        "declared": bool(declared),
        "logical_inputs": _TRAIN_STEP_SUFFIX_LOSS_INPUT_ABI_NAMES
        if declared
        else (),
        "reason": _TRAIN_STEP_SUFFIX_LOSS_INPUT_ABI_REASON
        if declared
        else "train-step suffix loss inputs are not required for this descriptor",
    }


def _descriptor_prim_func_source(
    *,
    entry_name: str,
    nodes: Sequence[_ScheduleNodeView],
    descriptors: Sequence[PathCBrickScheduleDescriptor],
    internal_buffers: Sequence[str],
    internal_buffer_shapes: Mapping[str, tuple[int, ...]],
    physical_abi_plan: _PhysicalAbiPlan,
    physical_abi_policy: str,
    internal_buffer_policy: str,
    loop_policy: str,
    max_rows_per_launch: int | None,
    row_dispatch_mode: str,
    rows_per_kernel_launch: int,
    execution_stage: str,
    active_node_names: frozenset[str] | None,
    external_buffers: Sequence[str],
    shape_by_buffer: Mapping[str, tuple[int, ...]],
    dtype_by_buffer: dict[str, str],
    buffer_extent: int,
    loop_extent: int,
    shape_env: PathCModelShapeEnv | None,
    train_step_computed_output_buffers: Sequence[str] = (),
    train_step_loss_source_buffers: Sequence[str] = (),
    train_step_loss_cotangent_buffers: Sequence[str] = (),
    train_step_loss_parameter_grad_buffers: Sequence[str] = (),
) -> tuple[str, Mapping[str, Mapping[str, Any]]]:
    indent = " " * 4
    param_lines = list(physical_abi_plan.param_lines)
    if not param_lines:
        param_lines = [
            f"{indent}_dummy: T.Tensor(({buffer_extent},), \"float32\"),"
        ]
    chunked_row_count = _row_phased_chunk_count(
        loop_policy=loop_policy,
        shape_env=shape_env,
        max_rows_per_launch=max_rows_per_launch,
    )
    chunked_by_launcher = (
        chunked_row_count is not None
        and row_dispatch_mode == DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS
    )
    if chunked_by_launcher:
        param_lines.append(
            f"{indent}{DESCRIPTOR_ROW_CHUNK_INDEX_PARAM}: T.int32,"
        )
        param_lines.append(
            f"{indent}{DESCRIPTOR_ROW_SUBCHUNK_INDEX_PARAM}: T.int32,"
        )
    has_backward = (
        execution_stage == DESCRIPTOR_EXECUTION_STAGE_ALL
        and any(
            node.op_name.endswith("_bwd")
            and (active_node_names is None or node.name in active_node_names)
            for node in nodes
        )
    )
    if has_backward:
        param_lines.append(
            f"{indent}{DESCRIPTOR_BACKWARD_GATE_PARAM}: T.int32,"
        )
    backward_stage_selector = bool(
        has_backward
        and chunked_by_launcher
        and loop_policy == DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN
        and active_node_names is None
    )
    backward_stage_name_groups = (
        _path_c_backward_stage_name_groups_for_nodes(nodes)
        if backward_stage_selector
        else ()
    )
    if backward_stage_selector and backward_stage_name_groups:
        param_lines.append(
            f"{indent}{DESCRIPTOR_BACKWARD_STAGE_INDEX_PARAM}: T.int32,"
        )
    active_internal_buffers = _stage_active_internal_buffers(
        nodes=nodes,
        internal_buffers=internal_buffers,
        active_node_names=active_node_names,
    )
    force_spilled_internal_buffers = _stage_boundary_internal_buffers(
        nodes=nodes,
        internal_buffers=active_internal_buffers,
        active_node_names=active_node_names,
    )
    access_by_buffer = {
        buffer_name: _internal_buffer_ref(
            buffer_name,
            internal_buffer_shapes[buffer_name],
            shape_env,
        )
        for buffer_name in internal_buffers
    }
    for buffer_name, shape in shape_by_buffer.items():
        access_by_buffer[buffer_name] = physical_abi_plan.external_access_by_buffer[
            buffer_name
        ]
    fragments = tuple(
        _descriptor_node_source(
            node=node,
            node_index=node_index,
            descriptor=descriptors[node_index],
            dtype_by_buffer=dtype_by_buffer,
            access_by_buffer=access_by_buffer,
        )
        for node_index, node in enumerate(nodes)
    )
    thread_limit = (
        DESCRIPTOR_ROW_PHASED_THREADS
        if loop_policy == DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN
        else DESCRIPTOR_DEFAULT_THREADS
    )
    thread_count = min(thread_limit, max(1, loop_extent))
    block_count = (loop_extent + thread_count - 1) // thread_count
    if loop_policy == DESCRIPTOR_LOOP_POLICY_FLAT:
        kernel_line = f"{indent}with T.Kernel({block_count}, threads={thread_count}) as bx:"
    elif chunked_by_launcher:
        kernel_line = f"{indent}with T.Kernel(1, threads={thread_count}) as chunk:"
    elif chunked_row_count is not None:
        kernel_line = (
            f"{indent}with T.Kernel({chunked_row_count}, threads={thread_count}) as chunk:"
        )
    else:
        kernel_line = f"{indent}with T.Kernel(1, threads={thread_count}):"

    body: list[str] = [
        "@T.prim_func",
        f"def {entry_name}(",
        *param_lines,
        "):",
        kernel_line,
        f"{indent * 2}# internal_buffer_policy: {internal_buffer_policy}",
        f"{indent * 2}# loop_policy: {loop_policy}",
        f"{indent * 2}# execution_stage: {execution_stage}",
    ]
    if loop_policy == DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN:
        root_regions = _root_kernel_buffer_regions(
            physical_abi_plan.physical_buffer_shapes
        )
        body.append(
            f"{indent * 2}T.reads({', '.join(root_regions)})  "
            f"# {_DESCRIPTOR_ROOT_READS_MARKER}"
        )
        body.append(
            f"{indent * 2}T.writes({', '.join(root_regions)})  "
            f"# {_DESCRIPTOR_ROOT_WRITES_MARKER}"
        )
    if max_rows_per_launch is not None:
        body.append(f"{indent * 2}# max_rows_per_launch: {max_rows_per_launch}")
        body.append(f"{indent * 2}# row_dispatch_mode: {row_dispatch_mode}")
    if loop_policy == DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN:
        body.append(f"{indent * 2}lane = T.get_thread_binding(0)")
    internal_allocator = (
        "T.alloc_shared"
        if internal_buffer_policy == DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN
        else "T.alloc_local"
    )
    for buffer_name in active_internal_buffers:
        storage_dtype = _internal_buffer_storage_dtype(
            dtype_by_buffer[buffer_name],
            allocator=internal_allocator,
        )
        body.append(
            f"{indent * 2}{_safe_identifier(buffer_name)} = "
            f"{internal_allocator}({_shape_literal(internal_buffer_shapes[buffer_name])}, "
            f"\"{storage_dtype}\")"
        )
    for node, descriptor, fragment in zip(nodes, descriptors, fragments, strict=True):
        if (
            loop_policy == DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN
            and (
                (active_node_names is None or node.name in active_node_names)
                and (
                _is_row_phased_mamba3(node, descriptor, shape_env)
                or node.op_name == "residual_rmsnorm"
                or _is_row_phased_m2rnn(node, descriptor, shape_env)
                or _is_row_phased_residual_rmsnorm_bwd(node, descriptor)
                or _is_row_phased_bwd_descriptor(node, descriptor, shape_env)
                )
            )
        ):
            continue
        if active_node_names is not None and node.name not in active_node_names:
            continue
        for allocation in fragment.allocations:
            body.append(f"{indent * 2}{allocation}")
    if not active_internal_buffers and not any(
        fragment.allocations for fragment in fragments
    ):
        body.append(f"{indent * 2}_scratch = T.alloc_local((1,), \"float32\")")
    if loop_policy == DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN:
        _append_row_phased_hidden_body(
            body,
            nodes=nodes,
            descriptors=descriptors,
            fragments=fragments,
            dtype_by_buffer=dtype_by_buffer,
            access_by_buffer=access_by_buffer,
            shape_env=shape_env,
            train_step_computed_output_buffers=train_step_computed_output_buffers,
            train_step_loss_source_buffers=train_step_loss_source_buffers,
            physical_abi_plan=physical_abi_plan,
            train_step_loss_cotangent_buffers=train_step_loss_cotangent_buffers,
            train_step_loss_parameter_grad_buffers=(
                train_step_loss_parameter_grad_buffers
            ),
            max_rows_per_launch=max_rows_per_launch,
            row_dispatch_mode=row_dispatch_mode,
            rows_per_kernel_launch=rows_per_kernel_launch,
            execution_stage=execution_stage,
            active_node_names=active_node_names,
            backward_stage_selector=bool(backward_stage_name_groups),
            row_dispatch_defined=False,
            indent=indent,
            cuda_target=(_path_c_default_target() == "cuda"),
        )
    else:
        active_items = tuple(
            (node, descriptor, fragment)
            for node, descriptor, fragment in zip(
                nodes, descriptors, fragments, strict=True
            )
            if (
                (active_node_names is None or node.name in active_node_names)
                and (
                execution_stage != DESCRIPTOR_EXECUTION_STAGE_BACKWARD
                and not node.op_name.endswith("_bwd")
                )
            )
            or (
                (active_node_names is None or node.name in active_node_names)
                and (
                execution_stage != DESCRIPTOR_EXECUTION_STAGE_FORWARD
                and node.op_name.endswith("_bwd")
                )
            )
        )
        _reject_proxy_backward_fragments(active_items)
        body.append(f"{indent * 2}tid = T.get_thread_binding(0)")
        body.append(f"{indent * 2}i = bx * {thread_count} + tid")
        body.append(f"{indent * 2}if i < {loop_extent}:")
        for node, descriptor, fragment in active_items:
            _append_descriptor_node_comments(
                body,
                node=node,
                descriptor=descriptor,
                indent=indent * 3,
            )
            for statement in fragment.statements:
                body.append(f"{indent * 3}{statement}")
        _append_train_step_suffix_scalar_outputs(
            body,
            computed_outputs=train_step_computed_output_buffers,
            loss_source_buffers=train_step_loss_source_buffers,
            physical_abi_plan=physical_abi_plan,
            train_step_loss_cotangent_buffers=train_step_loss_cotangent_buffers,
            train_step_loss_parameter_grad_buffers=(
                train_step_loss_parameter_grad_buffers
            ),
            loop_policy=loop_policy,
            shape_env=shape_env,
            indent=indent,
        )
    spilled_source, spilled_scratch = _spill_large_shared_scratch_to_abi(
        body,
        existing_parameter_count=len(param_lines),
        internal_buffer_names=frozenset(active_internal_buffers),
        force_spill_names=frozenset(force_spilled_internal_buffers),
        force_builtin_spill_names=(
            physical_abi_policy != DESCRIPTOR_PHYSICAL_ABI_POLICY_DIRECT
        ),
    )
    # CUDA-only, target/capacity-gated: any oversized T.alloc_shared scratch that
    # could NOT be spilled to ABI device-scratch params above (the portable
    # 31-buffer kernel ABI budget is exhausted on the backward direct-chain) is
    # rewritten to a global device workspace so ptxas accepts the kernel. Off
    # CUDA (e.g. Metal) this is a no-op and the source is unchanged.
    spilled_source = _demote_residual_shared_scratch_to_global(spilled_source)
    # Metal-only: pool oversized residual T.alloc_shared scratch into ONE global
    # workspace buffer.  Metal caps threadgroup memory at 32 KiB; the reverse-scan
    # backward declares multiple MB of shared.dyn, which overflows that limit and
    # crashes newComputePipelineState.  Per-buffer demotion is not viable (each
    # global buffer is a kernel argument and the kernel already uses ~28 of ~31),
    # so this pools every demoted float32 buffer into a single coalesced global
    # workspace (one extra argument).  Off Metal this is a no-op.
    spilled_source = _pool_oversized_shared_scratch_to_metal_workspace(spilled_source)
    return spilled_source, spilled_scratch


def _stage_active_internal_buffers(
    *,
    nodes: Sequence[_ScheduleNodeView],
    internal_buffers: Sequence[str],
    active_node_names: frozenset[str] | None,
) -> tuple[str, ...]:
    if active_node_names is None:
        return tuple(internal_buffers)
    internal_set = set(internal_buffers)
    touched: set[str] = set()
    for node in nodes:
        if node.name not in active_node_names:
            continue
        for buffer_name in (*node.inputs, *node.outputs):
            if buffer_name in internal_set:
                touched.add(buffer_name)
    return tuple(name for name in internal_buffers if name in touched)


def _stage_boundary_internal_buffers(
    *,
    nodes: Sequence[_ScheduleNodeView],
    internal_buffers: Sequence[str],
    active_node_names: frozenset[str] | None,
) -> tuple[str, ...]:
    if active_node_names is None:
        return ()
    internal_set = set(internal_buffers)
    producer_by_buffer: dict[str, str] = {}
    consumers_by_buffer: dict[str, set[str]] = {name: set() for name in internal_buffers}
    for node in nodes:
        for output_name in node.outputs:
            if output_name in internal_set:
                producer_by_buffer[output_name] = node.name
        for input_name in node.inputs:
            if input_name in internal_set:
                consumers_by_buffer.setdefault(input_name, set()).add(node.name)
    boundary: list[str] = []
    for buffer_name in internal_buffers:
        producer_active = producer_by_buffer.get(buffer_name) in active_node_names
        consumer_active = any(
            consumer in active_node_names
            for consumer in consumers_by_buffer.get(buffer_name, ())
        )
        if producer_active != consumer_active:
            boundary.append(buffer_name)
    return tuple(boundary)


def _internal_buffer_storage_dtype(dtype: str, *, allocator: str) -> str:
    # TVM's Metal bf16 wrapper is not valid as threadgroup storage. Internal
    # row-local scratch participates in fp32 math, so keep bf16 at ABI/model
    # boundaries and use float32 for generated shared scratch.
    if allocator == "T.alloc_shared" and str(dtype) == "bfloat16":
        return "float32"
    return str(dtype)


def _append_train_step_suffix_scalar_outputs(
    body: list[str],
    *,
    computed_outputs: Sequence[str],
    loss_source_buffers: Sequence[str],
    train_step_loss_cotangent_buffers: Sequence[str],
    train_step_loss_parameter_grad_buffers: Sequence[str],
    physical_abi_plan: _PhysicalAbiPlan,
    loop_policy: str,
    shape_env: PathCModelShapeEnv | None,
    indent: str,
) -> None:
    computed_output_set = {str(output) for output in computed_outputs}
    compute_ntokens = "ntokens" in computed_output_set
    compute_loss = "loss" in computed_output_set
    compute_cotangents = bool(train_step_loss_cotangent_buffers) and compute_loss
    compute_parameter_grads = (
        bool(train_step_loss_parameter_grad_buffers) and compute_loss
    )
    if not compute_ntokens and not compute_loss:
        return
    if shape_env is None:
        return
    if "target_mask" not in physical_abi_plan.logical_to_physical:
        return
    if compute_ntokens and "ntokens" not in physical_abi_plan.logical_to_physical:
        return
    if compute_loss and not _train_step_suffix_loss_can_emit(
        physical_abi_plan=physical_abi_plan,
        loss_source_buffers=loss_source_buffers,
    ):
        compute_loss = False
        compute_cotangents = False
        compute_parameter_grads = False
    if compute_cotangents and not _train_step_suffix_loss_cotangents_can_emit(
        physical_abi_plan=physical_abi_plan,
        loss_source_buffers=loss_source_buffers,
        cotangent_buffers=train_step_loss_cotangent_buffers,
    ):
        compute_cotangents = False
    if (
        compute_parameter_grads
        and not _train_step_suffix_loss_parameter_grads_can_emit(
            physical_abi_plan=physical_abi_plan,
            loss_source_buffers=loss_source_buffers,
            parameter_grad_buffers=train_step_loss_parameter_grad_buffers,
        )
    ):
        compute_parameter_grads = False

    sequence_length = int(shape_env.sequence_length)
    hidden_size = int(shape_env.hidden_size)
    vocab_size = int(getattr(shape_env, "vocab_size", 0) or 0)
    target_mask_ref = _physical_logical_buffer_ref(
        physical_abi_plan,
        "target_mask",
        "token_row",
    )
    target_mask_value = _physical_logical_buffer_value_expr(
        physical_abi_plan,
        "target_mask",
        "token_row",
    )
    target_id_ref = _physical_logical_buffer_ref(
        physical_abi_plan,
        "target_ids",
        "token_row",
    )
    loss_ref = _physical_logical_buffer_ref(physical_abi_plan, "loss", "0")
    ntokens_ref = _physical_logical_buffer_ref(physical_abi_plan, "ntokens", "0")
    guard = (
        "lane == 0"
        if loop_policy == DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN
        else "i == 0"
    )
    body.append(f"{indent * 2}# train_step_suffix_loss_scalar")
    if compute_loss:
        for scratch_name in (
            "train_step_suffix_loss_accum",
            "train_step_suffix_row_sum_sq",
            "train_step_suffix_inv_rms",
            "train_step_suffix_max_logit",
            "train_step_suffix_target_logit",
            "train_step_suffix_logit",
            "train_step_suffix_sum_exp",
            "train_step_suffix_hidden_value",
        ):
            body.append(
                f"{indent * 2}{scratch_name} = T.alloc_local((1,), \"float32\")"
            )
    if compute_cotangents:
        for scratch_name in (
            "train_step_suffix_seed_dot",
            "train_step_suffix_seed_grad_norm",
            "train_step_suffix_seed_softmax",
            "train_step_suffix_seed_class_grad",
            "train_step_suffix_seed_hidden_grad",
        ):
            body.append(
                f"{indent * 2}{scratch_name} = T.alloc_local((1,), \"float32\")"
            )
    if compute_parameter_grads:
        for scratch_name in (
            "train_step_suffix_param_grad_norm",
            "train_step_suffix_param_class_grad",
        ):
            body.append(
                f"{indent * 2}{scratch_name} = T.alloc_local((1,), \"float32\")"
            )
    body.append(f"{indent * 2}if {guard}:")
    if compute_ntokens:
        body.append(f"{indent * 3}# train_step_suffix_loss_ntokens")
        body.append(f"{indent * 3}{ntokens_ref} = T.cast(0.0, \"float32\")")
        body.append(
            f"{indent * 3}for token_row in T.serial(0, {sequence_length}):"
        )
        body.append(
            f"{indent * 4}{ntokens_ref} = {ntokens_ref} + "
            f"T.cast({target_mask_ref}, \"float32\")"
        )
    if compute_loss:
        body.append(f"{indent * 3}{loss_ref} = T.cast(0.0, \"float32\")")
        body.append(
            f"{indent * 3}train_step_suffix_loss_accum[0] = "
            'T.cast(0.0, "float32")'
        )
        body.append(
            f"{indent * 3}for token_row in T.serial(0, {sequence_length}):"
        )
        body.append(f"{indent * 4}if T.cast({target_mask_ref}, \"float32\") != 0.0:")
        body.append(
            f"{indent * 5}train_step_suffix_row_sum_sq[0] = "
            'T.cast(0.0, "float32")'
        )
        body.append(
            f"{indent * 5}for suffix_hidden_col in T.serial(0, {hidden_size}):"
        )
        _append_suffix_hidden_value(
            body,
            physical_abi_plan=physical_abi_plan,
            loss_source_buffers=loss_source_buffers,
            row_expr="token_row",
            hidden_expr="suffix_hidden_col",
            target="train_step_suffix_hidden_value[0]",
            indent=indent * 6,
        )
        body.append(
            f"{indent * 6}train_step_suffix_row_sum_sq[0] = "
            "train_step_suffix_row_sum_sq[0] + "
            "(train_step_suffix_hidden_value[0] * "
            "train_step_suffix_hidden_value[0])"
        )
        body.append(
            f"{indent * 5}train_step_suffix_inv_rms[0] = "
            f"T.rsqrt((train_step_suffix_row_sum_sq[0] / {float(hidden_size)}) "
            "+ 0.00001)"
        )
        body.append(
            f"{indent * 5}train_step_suffix_max_logit[0] = "
            "T.float32(-3.4028234663852886e38)"
        )
        body.append(
            f"{indent * 5}train_step_suffix_target_logit[0] = "
            'T.cast(0.0, "float32")'
        )
        body.append(f"{indent * 5}for vocab_col in T.serial(0, {vocab_size}):")
        _append_suffix_logit(
            body,
            physical_abi_plan=physical_abi_plan,
            loss_source_buffers=loss_source_buffers,
            hidden_size=hidden_size,
            row_expr="token_row",
            vocab_expr="vocab_col",
            indent=indent * 6,
        )
        body.append(
            f"{indent * 6}if train_step_suffix_logit[0] > "
            "train_step_suffix_max_logit[0]:"
        )
        body.append(
            f"{indent * 7}train_step_suffix_max_logit[0] = "
            "train_step_suffix_logit[0]"
        )
        body.append(f"{indent * 6}if vocab_col == {target_id_ref}:")
        body.append(
            f"{indent * 7}train_step_suffix_target_logit[0] = "
            "train_step_suffix_logit[0]"
        )
        body.append(
            f"{indent * 5}train_step_suffix_sum_exp[0] = "
            'T.cast(0.0, "float32")'
        )
        body.append(f"{indent * 5}for vocab_col in T.serial(0, {vocab_size}):")
        _append_suffix_logit(
            body,
            physical_abi_plan=physical_abi_plan,
            loss_source_buffers=loss_source_buffers,
            hidden_size=hidden_size,
            row_expr="token_row",
            vocab_expr="vocab_col",
            indent=indent * 6,
        )
        body.append(
            f"{indent * 6}train_step_suffix_sum_exp[0] = "
            "train_step_suffix_sum_exp[0] + "
            "T.exp(train_step_suffix_logit[0] - "
            "train_step_suffix_max_logit[0])"
        )
        body.append(
            f"{indent * 5}train_step_suffix_loss_accum[0] = "
            "train_step_suffix_loss_accum[0] + "
            f"(T.cast({target_mask_value}, \"float32\") * "
            "((T.log(train_step_suffix_sum_exp[0]) + "
            "train_step_suffix_max_logit[0]) - "
            "train_step_suffix_target_logit[0]))"
        )
        body.append(
            f"{indent * 3}{loss_ref} = train_step_suffix_loss_accum[0] / "
            f'T.max(T.cast({ntokens_ref}, "float32"), T.cast(1.0, "float32"))'
        )
    if compute_cotangents:
        _append_train_step_suffix_loss_cotangent_seeds(
            body,
            loss_source_buffers=loss_source_buffers,
            cotangent_buffers=train_step_loss_cotangent_buffers,
            physical_abi_plan=physical_abi_plan,
            sequence_length=sequence_length,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            target_mask_ref=target_mask_ref,
            target_mask_value=target_mask_value,
            target_id_ref=target_id_ref,
            ntokens_ref=ntokens_ref,
            indent=indent,
        )
    if compute_parameter_grads:
        _append_train_step_suffix_loss_parameter_grads(
            body,
            loss_source_buffers=loss_source_buffers,
            parameter_grad_buffers=train_step_loss_parameter_grad_buffers,
            physical_abi_plan=physical_abi_plan,
            sequence_length=sequence_length,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            target_mask_ref=target_mask_ref,
            target_mask_value=target_mask_value,
            target_id_ref=target_id_ref,
            ntokens_ref=ntokens_ref,
            indent=indent,
        )
    if loop_policy == DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN:
        body.append(f"{indent * 2}T.sync_threads()")


def _train_step_suffix_loss_can_emit(
    *,
    physical_abi_plan: _PhysicalAbiPlan,
    loss_source_buffers: Sequence[str],
) -> bool:
    required = {
        "loss",
        "ntokens",
        "target_ids",
        "target_mask",
        "final_norm_weight",
        "lm_head_weight",
    }
    required.update(str(name) for name in loss_source_buffers)
    return all(name in physical_abi_plan.logical_to_physical for name in required)


def _train_step_suffix_loss_cotangents_can_emit(
    *,
    physical_abi_plan: _PhysicalAbiPlan,
    loss_source_buffers: Sequence[str],
    cotangent_buffers: Sequence[str],
) -> bool:
    return (
        bool(loss_source_buffers)
        and len(loss_source_buffers) == len(cotangent_buffers)
        and all(
            str(name) in physical_abi_plan.logical_to_physical
            for name in (*loss_source_buffers, *cotangent_buffers)
        )
    )


def _train_step_suffix_loss_parameter_grads_can_emit(
    *,
    physical_abi_plan: _PhysicalAbiPlan,
    loss_source_buffers: Sequence[str],
    parameter_grad_buffers: Sequence[str],
) -> bool:
    required = {
        "final_norm_weight",
        "lm_head_weight",
        "ntokens",
        "target_ids",
        "target_mask",
        *tuple(str(name) for name in loss_source_buffers),
        *tuple(str(name) for name in parameter_grad_buffers),
    }
    return bool(loss_source_buffers) and all(
        name in physical_abi_plan.logical_to_physical for name in required
    )


def _append_train_step_suffix_loss_cotangent_seeds(
    body: list[str],
    *,
    loss_source_buffers: Sequence[str],
    cotangent_buffers: Sequence[str],
    physical_abi_plan: _PhysicalAbiPlan,
    sequence_length: int,
    hidden_size: int,
    vocab_size: int,
    target_mask_ref: str,
    target_mask_value: str,
    target_id_ref: str,
    ntokens_ref: str,
    indent: str,
) -> None:
    body.append(f"{indent * 3}# train_step_suffix_loss_cotangent_seeds")
    body.append(f"{indent * 3}for token_row in T.serial(0, {sequence_length}):")
    body.append(f"{indent * 4}for suffix_hidden_col in T.serial(0, {hidden_size}):")
    for cotangent_name in cotangent_buffers:
        cotangent_ref = _physical_logical_buffer_ref_2d(
            physical_abi_plan,
            str(cotangent_name),
            "token_row",
            "suffix_hidden_col",
        )
        body.append(f"{indent * 5}{cotangent_ref} = T.cast(0.0, \"float32\")")
    body.append(f"{indent * 4}if T.cast({target_mask_ref}, \"float32\") != 0.0:")
    body.append(
        f"{indent * 5}train_step_suffix_row_sum_sq[0] = "
        'T.cast(0.0, "float32")'
    )
    body.append(
        f"{indent * 5}for seed_hidden_dot_col in T.serial(0, {hidden_size}):"
    )
    _append_suffix_hidden_value(
        body,
        physical_abi_plan=physical_abi_plan,
        loss_source_buffers=loss_source_buffers,
        row_expr="token_row",
        hidden_expr="seed_hidden_dot_col",
        target="train_step_suffix_hidden_value[0]",
        indent=indent * 6,
    )
    body.append(
        f"{indent * 6}train_step_suffix_row_sum_sq[0] = "
        "train_step_suffix_row_sum_sq[0] + "
        "(train_step_suffix_hidden_value[0] * "
        "train_step_suffix_hidden_value[0])"
    )
    body.append(
        f"{indent * 5}train_step_suffix_inv_rms[0] = "
        f"T.rsqrt((train_step_suffix_row_sum_sq[0] / {float(hidden_size)}) "
        "+ 0.00001)"
    )
    body.append(
        f"{indent * 5}train_step_suffix_max_logit[0] = "
        "T.float32(-3.4028234663852886e38)"
    )
    body.append(f"{indent * 5}for vocab_col in T.serial(0, {vocab_size}):")
    _append_suffix_logit(
        body,
        physical_abi_plan=physical_abi_plan,
        loss_source_buffers=loss_source_buffers,
        hidden_size=hidden_size,
        row_expr="token_row",
        vocab_expr="vocab_col",
        indent=indent * 6,
        hidden_loop_name="suffix_seed_logit_hidden_col",
    )
    body.append(
        f"{indent * 6}if train_step_suffix_logit[0] > "
        "train_step_suffix_max_logit[0]:"
    )
    body.append(
        f"{indent * 7}train_step_suffix_max_logit[0] = "
        "train_step_suffix_logit[0]"
    )
    body.append(
        f"{indent * 5}train_step_suffix_sum_exp[0] = "
        'T.cast(0.0, "float32")'
    )
    body.append(f"{indent * 5}for vocab_col in T.serial(0, {vocab_size}):")
    _append_suffix_logit(
        body,
        physical_abi_plan=physical_abi_plan,
        loss_source_buffers=loss_source_buffers,
        hidden_size=hidden_size,
        row_expr="token_row",
        vocab_expr="vocab_col",
        indent=indent * 6,
        hidden_loop_name="suffix_seed_logit_hidden_col",
    )
    body.append(
        f"{indent * 6}train_step_suffix_sum_exp[0] = "
        "train_step_suffix_sum_exp[0] + "
        "T.exp(train_step_suffix_logit[0] - "
        "train_step_suffix_max_logit[0])"
    )
    body.append(
        f"{indent * 5}train_step_suffix_seed_dot[0] = "
        'T.cast(0.0, "float32")'
    )
    body.append(
        f"{indent * 5}for seed_hidden_dot_col in T.serial(0, {hidden_size}):"
    )
    _append_train_step_suffix_seed_grad_norm(
        body,
        physical_abi_plan=physical_abi_plan,
        loss_source_buffers=loss_source_buffers,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        row_expr="token_row",
        hidden_expr="seed_hidden_dot_col",
        target_id_ref=target_id_ref,
        target_mask_value=target_mask_value,
        ntokens_ref=ntokens_ref,
        indent=indent * 6,
    )
    _append_suffix_hidden_value(
        body,
        physical_abi_plan=physical_abi_plan,
        loss_source_buffers=loss_source_buffers,
        row_expr="token_row",
        hidden_expr="seed_hidden_dot_col",
        target="train_step_suffix_hidden_value[0]",
        indent=indent * 6,
    )
    seed_final_norm = _physical_logical_buffer_value_expr(
        physical_abi_plan,
        "final_norm_weight",
        "seed_hidden_dot_col",
    )
    body.append(
        f"{indent * 6}train_step_suffix_seed_dot[0] = "
        "train_step_suffix_seed_dot[0] + "
        "(train_step_suffix_seed_grad_norm[0] * "
        f"{seed_final_norm} * train_step_suffix_hidden_value[0])"
    )
    body.append(f"{indent * 5}for suffix_hidden_col in T.serial(0, {hidden_size}):")
    _append_train_step_suffix_seed_grad_norm(
        body,
        physical_abi_plan=physical_abi_plan,
        loss_source_buffers=loss_source_buffers,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        row_expr="token_row",
        hidden_expr="suffix_hidden_col",
        target_id_ref=target_id_ref,
        target_mask_value=target_mask_value,
        ntokens_ref=ntokens_ref,
        indent=indent * 6,
    )
    _append_suffix_hidden_value(
        body,
        physical_abi_plan=physical_abi_plan,
        loss_source_buffers=loss_source_buffers,
        row_expr="token_row",
        hidden_expr="suffix_hidden_col",
        target="train_step_suffix_hidden_value[0]",
        indent=indent * 6,
    )
    final_norm = _physical_logical_buffer_value_expr(
        physical_abi_plan,
        "final_norm_weight",
        "suffix_hidden_col",
    )
    body.append(
        f"{indent * 6}train_step_suffix_seed_hidden_grad[0] = "
        f"(train_step_suffix_inv_rms[0] * {final_norm} * "
        "train_step_suffix_seed_grad_norm[0]) - "
        "(train_step_suffix_hidden_value[0] * train_step_suffix_inv_rms[0] * "
        "train_step_suffix_inv_rms[0] * train_step_suffix_inv_rms[0] * "
        f"train_step_suffix_seed_dot[0] / {float(hidden_size)})"
    )
    for cotangent_name in cotangent_buffers:
        cotangent_ref = _physical_logical_buffer_ref_2d(
            physical_abi_plan,
            str(cotangent_name),
            "token_row",
            "suffix_hidden_col",
        )
        body.append(
            f"{indent * 6}{cotangent_ref} = "
            "train_step_suffix_seed_hidden_grad[0]"
        )


def _append_train_step_suffix_loss_parameter_grads(
    body: list[str],
    *,
    loss_source_buffers: Sequence[str],
    parameter_grad_buffers: Sequence[str],
    physical_abi_plan: _PhysicalAbiPlan,
    sequence_length: int,
    hidden_size: int,
    vocab_size: int,
    target_mask_ref: str,
    target_mask_value: str,
    target_id_ref: str,
    ntokens_ref: str,
    indent: str,
) -> None:
    buffer_set = {str(name) for name in parameter_grad_buffers}
    body.append(f"{indent * 3}# train_step_suffix_loss_parameter_grads")
    if "final_norm_weight_grad" in buffer_set:
        body.append(f"{indent * 3}for suffix_hidden_col in T.serial(0, {hidden_size}):")
        final_norm_grad_ref = _physical_logical_buffer_ref(
            physical_abi_plan,
            "final_norm_weight_grad",
            "suffix_hidden_col",
        )
        body.append(f"{indent * 4}{final_norm_grad_ref} = T.cast(0.0, \"float32\")")
    if "lm_head_weight_grad" in buffer_set:
        body.append(f"{indent * 3}for vocab_col in T.serial(0, {vocab_size}):")
        body.append(f"{indent * 4}for suffix_hidden_col in T.serial(0, {hidden_size}):")
        lm_head_grad_ref = _physical_logical_buffer_ref_2d(
            physical_abi_plan,
            "lm_head_weight_grad",
            "vocab_col",
            "suffix_hidden_col",
        )
        body.append(f"{indent * 5}{lm_head_grad_ref} = T.cast(0.0, \"float32\")")

    body.append(f"{indent * 3}for token_row in T.serial(0, {sequence_length}):")
    body.append(f"{indent * 4}if T.cast({target_mask_ref}, \"float32\") != 0.0:")
    body.append(
        f"{indent * 5}train_step_suffix_row_sum_sq[0] = "
        'T.cast(0.0, "float32")'
    )
    body.append(f"{indent * 5}for suffix_hidden_col in T.serial(0, {hidden_size}):")
    _append_suffix_hidden_value(
        body,
        physical_abi_plan=physical_abi_plan,
        loss_source_buffers=loss_source_buffers,
        row_expr="token_row",
        hidden_expr="suffix_hidden_col",
        target="train_step_suffix_hidden_value[0]",
        indent=indent * 6,
    )
    body.append(
        f"{indent * 6}train_step_suffix_row_sum_sq[0] = "
        "train_step_suffix_row_sum_sq[0] + "
        "(train_step_suffix_hidden_value[0] * "
        "train_step_suffix_hidden_value[0])"
    )
    body.append(
        f"{indent * 5}train_step_suffix_inv_rms[0] = "
        f"T.rsqrt((train_step_suffix_row_sum_sq[0] / {float(hidden_size)}) "
        "+ 0.00001)"
    )
    body.append(
        f"{indent * 5}train_step_suffix_max_logit[0] = "
        "T.float32(-3.4028234663852886e38)"
    )
    body.append(f"{indent * 5}for vocab_col in T.serial(0, {vocab_size}):")
    _append_suffix_logit(
        body,
        physical_abi_plan=physical_abi_plan,
        loss_source_buffers=loss_source_buffers,
        hidden_size=hidden_size,
        row_expr="token_row",
        vocab_expr="vocab_col",
        indent=indent * 6,
        hidden_loop_name="suffix_param_logit_hidden_col",
    )
    body.append(
        f"{indent * 6}if train_step_suffix_logit[0] > "
        "train_step_suffix_max_logit[0]:"
    )
    body.append(
        f"{indent * 7}train_step_suffix_max_logit[0] = "
        "train_step_suffix_logit[0]"
    )
    body.append(
        f"{indent * 5}train_step_suffix_sum_exp[0] = "
        'T.cast(0.0, "float32")'
    )
    body.append(f"{indent * 5}for vocab_col in T.serial(0, {vocab_size}):")
    _append_suffix_logit(
        body,
        physical_abi_plan=physical_abi_plan,
        loss_source_buffers=loss_source_buffers,
        hidden_size=hidden_size,
        row_expr="token_row",
        vocab_expr="vocab_col",
        indent=indent * 6,
        hidden_loop_name="suffix_param_logit_hidden_col",
    )
    body.append(
        f"{indent * 6}train_step_suffix_sum_exp[0] = "
        "train_step_suffix_sum_exp[0] + "
        "T.exp(train_step_suffix_logit[0] - "
        "train_step_suffix_max_logit[0])"
    )
    if "lm_head_weight_grad" in buffer_set:
        body.append(f"{indent * 5}for vocab_col in T.serial(0, {vocab_size}):")
        _append_train_step_suffix_param_class_grad(
            body,
            physical_abi_plan=physical_abi_plan,
            loss_source_buffers=loss_source_buffers,
            hidden_size=hidden_size,
            row_expr="token_row",
            vocab_expr="vocab_col",
            target_id_ref=target_id_ref,
            target_mask_value=target_mask_value,
            ntokens_ref=ntokens_ref,
            indent=indent * 6,
        )
        body.append(f"{indent * 6}for suffix_hidden_col in T.serial(0, {hidden_size}):")
        _append_suffix_hidden_value(
            body,
            physical_abi_plan=physical_abi_plan,
            loss_source_buffers=loss_source_buffers,
            row_expr="token_row",
            hidden_expr="suffix_hidden_col",
            target="train_step_suffix_hidden_value[0]",
            indent=indent * 7,
        )
        final_norm = _physical_logical_buffer_value_expr(
            physical_abi_plan,
            "final_norm_weight",
            "suffix_hidden_col",
        )
        lm_head_grad_ref = _physical_logical_buffer_ref_2d(
            physical_abi_plan,
            "lm_head_weight_grad",
            "vocab_col",
            "suffix_hidden_col",
        )
        body.append(
            f"{indent * 7}{lm_head_grad_ref} = {lm_head_grad_ref} + "
            "(train_step_suffix_param_class_grad[0] * "
            "train_step_suffix_hidden_value[0] * "
            f"train_step_suffix_inv_rms[0] * {final_norm})"
        )
    if "final_norm_weight_grad" in buffer_set:
        body.append(f"{indent * 5}for suffix_hidden_col in T.serial(0, {hidden_size}):")
        _append_train_step_suffix_param_grad_norm(
            body,
            physical_abi_plan=physical_abi_plan,
            loss_source_buffers=loss_source_buffers,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            row_expr="token_row",
            hidden_expr="suffix_hidden_col",
            target_id_ref=target_id_ref,
            target_mask_value=target_mask_value,
            ntokens_ref=ntokens_ref,
            indent=indent * 6,
        )
        _append_suffix_hidden_value(
            body,
            physical_abi_plan=physical_abi_plan,
            loss_source_buffers=loss_source_buffers,
            row_expr="token_row",
            hidden_expr="suffix_hidden_col",
            target="train_step_suffix_hidden_value[0]",
            indent=indent * 6,
        )
        final_norm_grad_ref = _physical_logical_buffer_ref(
            physical_abi_plan,
            "final_norm_weight_grad",
            "suffix_hidden_col",
        )
        body.append(
            f"{indent * 6}{final_norm_grad_ref} = {final_norm_grad_ref} + "
            "(train_step_suffix_param_grad_norm[0] * "
            "train_step_suffix_hidden_value[0] * "
            "train_step_suffix_inv_rms[0])"
        )


def _append_train_step_suffix_param_grad_norm(
    body: list[str],
    *,
    physical_abi_plan: _PhysicalAbiPlan,
    loss_source_buffers: Sequence[str],
    hidden_size: int,
    vocab_size: int,
    row_expr: str,
    hidden_expr: str,
    target_id_ref: str,
    target_mask_value: str,
    ntokens_ref: str,
    indent: str,
) -> None:
    body.append(
        f"{indent}train_step_suffix_param_grad_norm[0] = "
        'T.cast(0.0, "float32")'
    )
    body.append(f"{indent}for vocab_col in T.serial(0, {vocab_size}):")
    _append_train_step_suffix_param_class_grad(
        body,
        physical_abi_plan=physical_abi_plan,
        loss_source_buffers=loss_source_buffers,
        hidden_size=hidden_size,
        row_expr=row_expr,
        vocab_expr="vocab_col",
        target_id_ref=target_id_ref,
        target_mask_value=target_mask_value,
        ntokens_ref=ntokens_ref,
        indent=f"{indent}    ",
    )
    lm_head = _physical_logical_buffer_value_expr_2d(
        physical_abi_plan,
        "lm_head_weight",
        "vocab_col",
        hidden_expr,
    )
    body.append(
        f"{indent}    train_step_suffix_param_grad_norm[0] = "
        "train_step_suffix_param_grad_norm[0] + "
        f"(train_step_suffix_param_class_grad[0] * {lm_head})"
    )


def _append_train_step_suffix_param_class_grad(
    body: list[str],
    *,
    physical_abi_plan: _PhysicalAbiPlan,
    loss_source_buffers: Sequence[str],
    hidden_size: int,
    row_expr: str,
    vocab_expr: str,
    target_id_ref: str,
    target_mask_value: str,
    ntokens_ref: str,
    indent: str,
) -> None:
    _append_suffix_logit(
        body,
        physical_abi_plan=physical_abi_plan,
        loss_source_buffers=loss_source_buffers,
        hidden_size=hidden_size,
        row_expr=row_expr,
        vocab_expr=vocab_expr,
        indent=indent,
        hidden_loop_name="suffix_param_logit_hidden_col",
    )
    body.append(
        f"{indent}train_step_suffix_param_class_grad[0] = "
        "T.exp(train_step_suffix_logit[0] - train_step_suffix_max_logit[0]) / "
        "train_step_suffix_sum_exp[0]"
    )
    body.append(f"{indent}if {vocab_expr} == {target_id_ref}:")
    body.append(
        f"{indent}    train_step_suffix_param_class_grad[0] = "
        "train_step_suffix_param_class_grad[0] - T.cast(1.0, \"float32\")"
    )
    body.append(
        f"{indent}train_step_suffix_param_class_grad[0] = "
        "train_step_suffix_param_class_grad[0] * "
        f"T.cast({target_mask_value}, \"float32\") / "
        f'T.max(T.cast({ntokens_ref}, "float32"), T.cast(1.0, "float32"))'
    )


def _append_train_step_suffix_seed_grad_norm(
    body: list[str],
    *,
    physical_abi_plan: _PhysicalAbiPlan,
    loss_source_buffers: Sequence[str],
    hidden_size: int,
    vocab_size: int,
    row_expr: str,
    hidden_expr: str,
    target_id_ref: str,
    target_mask_value: str,
    ntokens_ref: str,
    indent: str,
) -> None:
    body.append(
        f"{indent}train_step_suffix_seed_grad_norm[0] = "
        'T.cast(0.0, "float32")'
    )
    body.append(f"{indent}for vocab_col in T.serial(0, {vocab_size}):")
    _append_suffix_logit(
        body,
        physical_abi_plan=physical_abi_plan,
        loss_source_buffers=loss_source_buffers,
        hidden_size=hidden_size,
        row_expr=row_expr,
        vocab_expr="vocab_col",
        indent=f"{indent}    ",
        hidden_loop_name="suffix_seed_logit_hidden_col",
    )
    body.append(
        f"{indent}    train_step_suffix_seed_softmax[0] = "
        "T.exp(train_step_suffix_logit[0] - train_step_suffix_max_logit[0]) / "
        "train_step_suffix_sum_exp[0]"
    )
    body.append(
        f"{indent}    train_step_suffix_seed_class_grad[0] = "
        "train_step_suffix_seed_softmax[0]"
    )
    body.append(f"{indent}    if vocab_col == {target_id_ref}:")
    body.append(
        f"{indent}        train_step_suffix_seed_class_grad[0] = "
        "train_step_suffix_seed_class_grad[0] - T.cast(1.0, \"float32\")"
    )
    body.append(
        f"{indent}    train_step_suffix_seed_class_grad[0] = "
        "train_step_suffix_seed_class_grad[0] * "
        f"T.cast({target_mask_value}, \"float32\") / "
        f'T.max(T.cast({ntokens_ref}, "float32"), T.cast(1.0, "float32"))'
    )
    lm_head = _physical_logical_buffer_value_expr_2d(
        physical_abi_plan,
        "lm_head_weight",
        "vocab_col",
        hidden_expr,
    )
    body.append(
        f"{indent}    train_step_suffix_seed_grad_norm[0] = "
        "train_step_suffix_seed_grad_norm[0] + "
        f"(train_step_suffix_seed_class_grad[0] * {lm_head})"
    )


def _append_suffix_hidden_value(
    body: list[str],
    *,
    physical_abi_plan: _PhysicalAbiPlan,
    loss_source_buffers: Sequence[str],
    row_expr: str,
    hidden_expr: str,
    target: str,
    indent: str,
) -> None:
    body.append(f'{indent}{target} = T.cast(0.0, "float32")')
    for source_name in loss_source_buffers:
        source_ref = _physical_logical_buffer_value_expr_2d(
            physical_abi_plan,
            str(source_name),
            row_expr,
            hidden_expr,
        )
        body.append(f"{indent}{target} = {target} + {source_ref}")


def _append_suffix_logit(
    body: list[str],
    *,
    physical_abi_plan: _PhysicalAbiPlan,
    loss_source_buffers: Sequence[str],
    hidden_size: int,
    row_expr: str,
    vocab_expr: str,
    indent: str,
    hidden_loop_name: str = "suffix_hidden_col",
) -> None:
    body.append(f'{indent}train_step_suffix_logit[0] = T.cast(0.0, "float32")')
    body.append(f"{indent}for {hidden_loop_name} in T.serial(0, {hidden_size}):")
    _append_suffix_hidden_value(
        body,
        physical_abi_plan=physical_abi_plan,
        loss_source_buffers=loss_source_buffers,
        row_expr=row_expr,
        hidden_expr=hidden_loop_name,
        target="train_step_suffix_hidden_value[0]",
        indent=f"{indent}    ",
    )
    final_norm = _physical_logical_buffer_value_expr(
        physical_abi_plan,
        "final_norm_weight",
        hidden_loop_name,
    )
    lm_head = _physical_logical_buffer_value_expr_2d(
        physical_abi_plan,
        "lm_head_weight",
        vocab_expr,
        hidden_loop_name,
    )
    body.append(
        f"{indent}    train_step_suffix_logit[0] = "
        "train_step_suffix_logit[0] + "
        "(train_step_suffix_hidden_value[0] * "
        f"train_step_suffix_inv_rms[0] * {final_norm} * {lm_head})"
    )


def _physical_logical_buffer_ref(
    physical_abi_plan: _PhysicalAbiPlan,
    logical_name: str,
    index_expr: str,
) -> str:
    info = physical_abi_plan.logical_to_physical.get(logical_name)
    if info is None:
        return f"{_safe_identifier(logical_name)}[{index_expr}]"
    size = int(info.get("size", 1) or 1)
    expr = "0" if size <= 1 else index_expr
    bank_name = str(info["bank"])
    offset = int(info.get("offset", 0) or 0)
    if offset == 0:
        return f"{bank_name}[{expr}]"
    return f"{bank_name}[{offset} + ({expr})]"


def _physical_logical_buffer_ref_2d(
    physical_abi_plan: _PhysicalAbiPlan,
    logical_name: str,
    row_expr: str,
    col_expr: str,
) -> str:
    info = physical_abi_plan.logical_to_physical.get(logical_name)
    if info is None:
        return f"{_safe_identifier(logical_name)}[{row_expr}, {col_expr}]"
    shape = tuple(int(dim) for dim in info.get("shape", ()) or ())
    logical_shape = tuple(
        int(dim)
        for dim in (info.get("logical_shape", ()) or shape)
    )
    bank_name = str(info["bank"])
    offset = int(info.get("offset", 0) or 0)
    if offset == 0 and bank_name == logical_name and len(logical_shape) == 2:
        return f"{bank_name}[{row_expr}, {col_expr}]"
    if (
        offset == 0
        and bank_name == logical_name
        and len(logical_shape) == 3
        and logical_shape[0] == 1
    ):
        return f"{bank_name}[0, {row_expr}, {col_expr}]"
    if len(logical_shape) == 2:
        stride = logical_shape[1]
    elif len(logical_shape) == 3 and logical_shape[0] == 1:
        stride = logical_shape[2]
    else:
        stride = shape[1] if len(shape) == 2 else 1
    return _physical_logical_buffer_ref(
        physical_abi_plan,
        logical_name,
        f"({row_expr}) * {stride} + ({col_expr})",
    )


def _physical_logical_buffer_value_expr(
    physical_abi_plan: _PhysicalAbiPlan,
    logical_name: str,
    index_expr: str,
) -> str:
    ref = _physical_logical_buffer_ref(physical_abi_plan, logical_name, index_expr)
    return _physical_logical_buffer_value_from_ref(
        physical_abi_plan,
        logical_name,
        ref,
    )


def _physical_logical_buffer_value_expr_2d(
    physical_abi_plan: _PhysicalAbiPlan,
    logical_name: str,
    row_expr: str,
    col_expr: str,
) -> str:
    ref = _physical_logical_buffer_ref_2d(
        physical_abi_plan,
        logical_name,
        row_expr,
        col_expr,
    )
    return _physical_logical_buffer_value_from_ref(
        physical_abi_plan,
        logical_name,
        ref,
    )


def _physical_logical_buffer_value_from_ref(
    physical_abi_plan: _PhysicalAbiPlan,
    logical_name: str,
    ref: str,
) -> str:
    info = physical_abi_plan.logical_to_physical.get(logical_name, {})
    dtype = str(info.get("dtype", "float32"))
    if dtype in {"bfloat16", "float16", "int32"}:
        return f'T.cast({ref}, "float32")'
    if dtype == "uint8":
        return f"fp8_e4m3fn_to_float({ref})"
    return ref


_ALLOC_SHARED_LINE_RE = re.compile(
    r'^(?P<indent>\s*)(?P<name>[A-Za-z_]\w*) = '
    r'T\.alloc_shared\((?P<shape>\([^)]*\)), "(?P<dtype>[^"]+)"\)$'
)


def _coalesced_scratch_bank_name(dtype: str) -> str:
    return _safe_identifier(f"path_c_{dtype}_scratch_bank")


def _root_kernel_buffer_region(buffer_name: str, shape: Sequence[int]) -> str:
    extents = ", ".join(f"0:{int(dim)}" for dim in shape)
    return f"{_safe_identifier(buffer_name)}[{extents}]"


def _root_kernel_buffer_regions(
    shapes_by_buffer: Mapping[str, Sequence[int]],
) -> list[str]:
    return [
        _root_kernel_buffer_region(buffer_name, shape)
        for buffer_name, shape in shapes_by_buffer.items()
    ]


def _append_regions_to_root_annotation(
    line: str,
    *,
    marker: str,
    extra_regions: Sequence[str],
) -> str:
    if marker not in line or not extra_regions:
        return line
    match = re.match(r"(\s*T\.(?:reads|writes)\()(.+?)(\)\s*# .*)", line)
    if match is None:
        return line
    existing = match.group(2).strip()
    regions = [existing] if existing else []
    regions.extend(extra_regions)
    return f"{match.group(1)}{', '.join(regions)}{match.group(3)}"


def _can_coalesce_spilled_scratch(candidate: Mapping[str, Any]) -> bool:
    return len(tuple(candidate["shape"])) == 1 and str(candidate["dtype"]) in {
        "float32",
        "int32",
    }


def _force_spill_shared_scratch_name(scratch_name: str) -> bool:
    return str(scratch_name).endswith(
        (
            "_mamba3_b_inv_rms",
            "_mamba3_c_inv_rms",
            "_mamba3_b_mean",
            "_mamba3_c_mean",
            "_mamba3_b_raw",
            "_mamba3_c_raw",
            "_mamba3_b_group",
            "_mamba3_c_group",
            "_mamba3_b_raw_grad",
            "_mamba3_c_raw_grad",
            "_mamba3_b_group_grad",
            "_mamba3_c_group_grad",
            "_mamba3_project_grad",
            "_mamba3_dt_vec",
            "_mamba3_a_vec",
            "_mamba3_dt_grad",
            "_mamba3_a_grad",
            "_mamba3_trap_group",
            "_mamba3_trap_grad",
            "_mamba3_next_dt_pre_vec",
            "_mamba3_next_dt_vec",
            "_mamba3_next_trap_vec",
            "_mamba3_angle_cumsum",
            "_mamba3_angle_grad",
        )
    )


def _coalesced_scratch_parameter_count(
    candidates: Sequence[Mapping[str, Any]],
) -> int:
    coalesced_banks = {
        _coalesced_scratch_bank_name(str(candidate["dtype"]))
        for candidate in candidates
        if _can_coalesce_spilled_scratch(candidate)
    }
    standalone = sum(
        1 for candidate in candidates if not _can_coalesce_spilled_scratch(candidate)
    )
    return len(coalesced_banks) + standalone


def _replace_one_dimensional_buffer_refs(
    line: str,
    *,
    source_name: str,
    target_name: str,
    target_offset: int,
) -> str:
    if f"{source_name}[" not in line:
        return line
    pieces: list[str] = []
    cursor = 0
    needle = f"{source_name}["
    while True:
        start = line.find(needle, cursor)
        if start < 0:
            pieces.append(line[cursor:])
            break
        if start > 0 and (line[start - 1].isalnum() or line[start - 1] == "_"):
            pieces.append(line[cursor : start + len(needle)])
            cursor = start + len(needle)
            continue
        depth = 1
        index_start = start + len(needle)
        end = index_start
        while end < len(line) and depth:
            char = line[end]
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
            end += 1
        if depth:
            pieces.append(line[cursor:])
            break
        index_expr = line[index_start : end - 1]
        if target_offset == 0:
            replacement = f"{target_name}[{index_expr}]"
        else:
            replacement = f"{target_name}[{target_offset} + ({index_expr})]"
        pieces.append(line[cursor:start])
        pieces.append(replacement)
        cursor = end
    return "".join(pieces)


def _spill_large_shared_scratch_to_abi(
    source_lines: Sequence[str],
    *,
    existing_parameter_count: int,
    internal_buffer_names: frozenset[str] = frozenset(),
    force_spill_names: frozenset[str] = frozenset(),
    force_builtin_spill_names: bool = True,
) -> tuple[str, Mapping[str, Mapping[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    total_shared_bytes = 0
    for line in source_lines:
        match = _ALLOC_SHARED_LINE_RE.match(line)
        if match is None:
            continue
        shape_value = ast.literal_eval(match.group("shape"))
        shape = (
            (int(shape_value),)
            if isinstance(shape_value, int)
            else tuple(int(dim) for dim in shape_value)
        )
        dtype = match.group("dtype")
        byte_count = _flattened_extent(shape) * _DTYPE_NBYTES[dtype]
        total_shared_bytes += byte_count
        scratch_name = match.group("name")
        force_scratch_abi = (
            scratch_name in force_spill_names
            or _is_row_phased_bwd_scratch_abi_buffer(scratch_name)
            or (
                force_builtin_spill_names
                and _force_spill_shared_scratch_name(scratch_name)
            )
        )
        if (
            byte_count >= DESCRIPTOR_SHARED_SCRATCH_SPILL_THRESHOLD_BYTES
            or force_scratch_abi
        ):
            candidates.append(
                {
                    "name": scratch_name,
                    "param_name": scratch_name,
                    "shape": shape,
                    "dtype": dtype,
                    "bytes": byte_count,
                    "force_scratch_abi": force_scratch_abi,
                    "internal_scratch_abi": scratch_name in internal_buffer_names,
                }
            )

    available_parameters = max(
        0,
        DESCRIPTOR_PORTABLE_KERNEL_BUFFER_LIMIT - existing_parameter_count,
    )
    remaining_shared_bytes = total_shared_bytes
    selected: list[dict[str, Any]] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            not bool(item.get("force_scratch_abi")),
            -int(item["bytes"]),
        ),
    ):
        if (
            _coalesced_scratch_parameter_count((*selected, candidate))
            > available_parameters
        ):
            if bool(candidate.get("force_scratch_abi")):
                raise ValueError(
                    "descriptor stage ABI scratch parameter budget exceeded "
                    f"while forcing {candidate['name']!r} to device ABI"
                )
            break
        if (
            not bool(candidate.get("force_scratch_abi"))
            and
            remaining_shared_bytes <= DESCRIPTOR_SHARED_SCRATCH_BUDGET_BYTES
            and int(candidate["bytes"])
            <= DESCRIPTOR_SHARED_SCRATCH_SPILL_THRESHOLD_BYTES
        ):
            break
        selected.append(candidate)
        remaining_shared_bytes -= int(candidate["bytes"])
        if (
            remaining_shared_bytes <= DESCRIPTOR_SHARED_SCRATCH_BUDGET_BYTES
            and not bool(candidate.get("force_scratch_abi"))
        ):
            break

    coalesced_offsets: dict[str, tuple[str, int]] = {}
    coalesced_shapes: dict[str, int] = {}
    for candidate in selected:
        if not _can_coalesce_spilled_scratch(candidate):
            continue
        bank_name = _coalesced_scratch_bank_name(str(candidate["dtype"]))
        offset = coalesced_shapes.get(bank_name, 0)
        coalesced_offsets[str(candidate["name"])] = (bank_name, offset)
        coalesced_shapes[bank_name] = offset + _flattened_extent(candidate["shape"])

    spilled = {
        str(candidate["name"]): {
            "dtype": str(candidate["dtype"]),
            "param_name": coalesced_offsets.get(str(candidate["name"]), ("", 0))[0]
            or str(candidate["param_name"]),
            "shape": tuple(candidate["shape"]),
            "bytes": int(candidate["bytes"]),
            "internal_scratch_abi": bool(candidate["internal_scratch_abi"]),
            "coalesced_scratch_bank": str(candidate["name"]) in coalesced_offsets,
            "bank": coalesced_offsets.get(str(candidate["name"]), ("", 0))[0],
            "offset": coalesced_offsets.get(str(candidate["name"]), ("", 0))[1],
        }
        for candidate in selected
    }
    if not spilled:
        return "\n".join(source_lines) + "\n", {}

    extra_root_regions = [
        _root_kernel_buffer_region(bank_name, (extent,))
        for bank_name, extent in coalesced_shapes.items()
    ]
    extra_root_regions.extend(
        _root_kernel_buffer_region(str(info["param_name"]), info["shape"])
        for info in spilled.values()
        if not bool(info.get("coalesced_scratch_bank"))
    )

    def rewrite_scratch_refs(line: str) -> str:
        rewritten_line = line
        for scratch_name, (bank_name, offset) in coalesced_offsets.items():
            rewritten_line = _replace_one_dimensional_buffer_refs(
                rewritten_line,
                source_name=scratch_name,
                target_name=bank_name,
                target_offset=offset,
            )
        return rewritten_line

    rewritten: list[str] = []
    signature_close_index: int | None = None
    for line in source_lines:
        match = _ALLOC_SHARED_LINE_RE.match(line)
        if match is not None and match.group("name") in spilled:
            continue
        if signature_close_index is None and line == "):":
            signature_close_index = len(rewritten)
        rewritten_line = rewrite_scratch_refs(line)
        rewritten_line = _append_regions_to_root_annotation(
            rewritten_line,
            marker=_DESCRIPTOR_ROOT_READS_MARKER,
            extra_regions=extra_root_regions,
        )
        rewritten_line = _append_regions_to_root_annotation(
            rewritten_line,
            marker=_DESCRIPTOR_ROOT_WRITES_MARKER,
            extra_regions=extra_root_regions,
        )
        rewritten.append(rewritten_line)

    if signature_close_index is None:
        raise ValueError("descriptor source did not contain a function signature close")
    param_indent = " " * 4
    coalesced_bank_dtypes = {
        bank_name: bank_name.removeprefix("path_c_").removesuffix("_scratch_bank")
        for bank_name in coalesced_shapes
    }
    coalesced_param_lines = [
        f'{param_indent}{bank_name}: T.Tensor(({extent},), '
        f'"{coalesced_bank_dtypes[bank_name]}"),'
        for bank_name, extent in coalesced_shapes.items()
    ]
    spill_param_lines = [
        f'{param_indent}{info["param_name"]}: T.Tensor({_shape_literal(info["shape"])}, '
        f'"{info["dtype"]}"),'
        for name, info in spilled.items()
        if not bool(info.get("coalesced_scratch_bank"))
    ]
    rewritten[signature_close_index:signature_close_index] = [
        *coalesced_param_lines,
        *spill_param_lines,
    ]
    return "\n".join(rewritten) + "\n", spilled


def _is_row_phased_bwd_scratch_abi_buffer(buffer_name: str) -> bool:
    """Return True for internal bwd scratch that must be a device ABI buffer."""

    name = str(buffer_name)
    if name in DESCRIPTOR_ROW_PHASED_BWD_SCRATCH_ABI_BUFFERS:
        return True
    if name in DESCRIPTOR_MAMBA3_CHUNKED_FWD_HANDOFF_ABI_BUFFERS:
        return True
    if name in DESCRIPTOR_MAMBA3_CHUNKED_BWD_HANDOFF_ABI_BUFFERS:
        return True
    if not name.endswith("_grad"):
        return False
    canonical = _canonical_buffer_name(name)
    if canonical in DESCRIPTOR_ROW_PHASED_BWD_SCRATCH_ABI_CANONICALS:
        return True
    base = name[: -len("_grad")]
    return base.endswith(("_hidden", "_delta", "_out"))


def _validated_internal_buffer_policy(policy: str) -> str:
    normalized = str(policy)
    if normalized not in {
        DESCRIPTOR_INTERNAL_BUFFER_POLICY_SCALAR_LOCAL,
        DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN,
    }:
        raise ValueError(
            "internal_buffer_policy must be one of "
            f"{DESCRIPTOR_INTERNAL_BUFFER_POLICY_SCALAR_LOCAL!r}, "
            f"{DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN!r}; "
            f"got {policy!r}"
        )
    return normalized


def _validated_loop_policy(policy: str) -> str:
    normalized = str(policy)
    if normalized not in {
        DESCRIPTOR_LOOP_POLICY_FLAT,
        DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
    }:
        raise ValueError(
            "loop_policy must be one of "
            f"{DESCRIPTOR_LOOP_POLICY_FLAT!r}, "
            f"{DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN!r}; "
            f"got {policy!r}"
        )
    return normalized


def _validated_physical_abi_policy(policy: str) -> str:
    normalized = str(policy)
    if normalized not in {
        DESCRIPTOR_PHYSICAL_ABI_POLICY_DIRECT,
        DESCRIPTOR_PHYSICAL_ABI_POLICY_BANKED_BY_DTYPE,
        DESCRIPTOR_PHYSICAL_ABI_POLICY_BANKED_BY_ROLE,
    }:
        raise ValueError(
            "physical_abi_policy must be one of "
            f"{DESCRIPTOR_PHYSICAL_ABI_POLICY_DIRECT!r}, "
            f"{DESCRIPTOR_PHYSICAL_ABI_POLICY_BANKED_BY_DTYPE!r}, "
            f"{DESCRIPTOR_PHYSICAL_ABI_POLICY_BANKED_BY_ROLE!r}; "
            f"got {policy!r}"
        )
    return normalized


def _physical_abi_plan(
    *,
    external_buffers: Sequence[str],
    shape_by_buffer: Mapping[str, tuple[int, ...]],
    dtype_by_buffer: Mapping[str, str],
    buffer_extent: int,
    loop_extent: int,
    shape_env: PathCModelShapeEnv | None,
    physical_abi_policy: str,
) -> _PhysicalAbiPlan:
    indent = " " * 4
    if physical_abi_policy == DESCRIPTOR_PHYSICAL_ABI_POLICY_DIRECT:
        return _direct_physical_abi_plan(
            external_buffers=external_buffers,
            shape_by_buffer=shape_by_buffer,
            dtype_by_buffer=dtype_by_buffer,
            loop_extent=loop_extent,
            shape_env=shape_env,
            indent=indent,
        )
    if not external_buffers:
        return _PhysicalAbiPlan((), {}, {}, {})

    bank_order: list[str] = []
    bank_totals: dict[str, int] = {}
    bank_dtypes: dict[str, str] = {}
    access_by_buffer: dict[str, str] = {}
    logical_to_physical: dict[str, Mapping[str, Any]] = {}
    for buffer_name in external_buffers:
        dtype = dtype_by_buffer[buffer_name]
        bank_name = _physical_abi_bank_name_for_buffer(
            dtype,
            buffer_name,
            physical_abi_policy=physical_abi_policy,
        )
        if bank_name not in bank_totals:
            bank_order.append(bank_name)
            bank_totals[bank_name] = 0
            bank_dtypes[bank_name] = dtype
        offset = bank_totals[bank_name]
        shape = shape_by_buffer[buffer_name]
        size = max(1, _flattened_extent(shape))
        bank_totals[bank_name] += size
        logical_ref = _loop_indexed_buffer_ref(
            buffer_name,
            shape,
            loop_extent,
            shape_env,
        )
        access_by_buffer[buffer_name] = _banked_buffer_ref(
            logical_ref,
            bank_name=bank_name,
            offset=offset,
        )
        logical_shape = _direct_logical_buffer_shape(
            buffer_name,
            shape,
            shape_env,
        )
        logical_to_physical[buffer_name] = {
            "bank": bank_name,
            "dtype": dtype,
            "offset": offset,
            "shape": shape,
            "logical_shape": logical_shape,
            "size": size,
        }

    physical_shapes = {bank_name: (bank_totals[bank_name],) for bank_name in bank_order}
    param_lines = tuple(
        f"{indent}{bank_name}: "
        f"T.Tensor({_shape_literal(physical_shapes[bank_name])}, "
        f"\"{bank_dtypes[bank_name]}\"),"
        for bank_name in bank_order
    )
    return _PhysicalAbiPlan(
        param_lines=param_lines,
        external_access_by_buffer=access_by_buffer,
        physical_buffer_shapes=physical_shapes,
        logical_to_physical=logical_to_physical,
    )


def _direct_physical_abi_plan(
    *,
    external_buffers: Sequence[str],
    shape_by_buffer: Mapping[str, tuple[int, ...]],
    dtype_by_buffer: Mapping[str, str],
    loop_extent: int,
    shape_env: PathCModelShapeEnv | None,
    indent: str,
) -> _PhysicalAbiPlan:
    direct_shape_by_buffer = {
        buffer_name: _direct_logical_buffer_shape(
            buffer_name,
            shape_by_buffer[buffer_name],
            shape_env,
        )
        for buffer_name in external_buffers
    }
    access_by_buffer = {
        buffer_name: _direct_loop_indexed_buffer_ref(
            buffer_name,
            direct_shape_by_buffer[buffer_name],
            loop_extent,
            shape_env,
        )
        for buffer_name in external_buffers
    }
    physical_shapes = {
        buffer_name: direct_shape_by_buffer[buffer_name]
        for buffer_name in external_buffers
    }
    logical_to_physical = {
        buffer_name: {
            "bank": buffer_name,
            "dtype": dtype_by_buffer[buffer_name],
            "offset": 0,
            "shape": direct_shape_by_buffer[buffer_name],
            "logical_shape": direct_shape_by_buffer[buffer_name],
            "size": _flattened_extent(direct_shape_by_buffer[buffer_name]),
        }
        for buffer_name in external_buffers
    }
    param_lines = tuple(
        f"{indent}{name}: T.Tensor({_shape_literal(direct_shape_by_buffer[name])}, "
        f"\"{dtype_by_buffer[name]}\"),"
        for name in external_buffers
    )
    return _PhysicalAbiPlan(
        param_lines=param_lines,
        external_access_by_buffer=access_by_buffer,
        physical_buffer_shapes=physical_shapes,
        logical_to_physical=logical_to_physical,
    )


def _direct_logical_buffer_shape(
    buffer_name: str,
    flat_shape: tuple[int, ...],
    shape_env: PathCModelShapeEnv | None,
) -> tuple[int, ...]:
    if shape_env is None:
        return flat_shape
    hidden = shape_env.hidden_size
    q_dim = shape_env.attention_num_q_heads * shape_env.attention_head_dim
    kv_dim = shape_env.attention_num_kv_heads * shape_env.attention_head_dim
    canonical_name = _canonical_buffer_name(buffer_name)
    if canonical_name == "mamba3_in_proj_weight":
        return (shape_env.mamba_in_proj_dim, hidden)
    if canonical_name == "mamba3_out_proj_weight":
        return (hidden, shape_env.mamba_inner_dim)
    if canonical_name == "mamba3_conv_weight":
        return (shape_env.mamba_conv_channels, shape_env.mamba_conv_kernel, 1)
    if canonical_name in {
        "mamba3_B_norm_weight",
        "mamba3_B_bias",
        "mamba3_C_norm_weight",
        "mamba3_C_bias",
    }:
        return (
            shape_env.mamba_effective_mimo_rank,
            shape_env.mamba_groups,
            shape_env.mamba_state_dim,
        )
    if canonical_name == "m2rnn_in_proj_weight":
        return (shape_env.m2rnn_in_proj_dim, hidden)
    if canonical_name == "m2rnn_conv_weight":
        return (shape_env.m2rnn_conv_dim, shape_env.m2rnn_conv_kernel, 1)
    if canonical_name == "m2rnn_state_weight":
        return (
            shape_env.m2rnn_num_weight_heads,
            shape_env.m2rnn_v_head_dim,
            shape_env.m2rnn_v_head_dim,
        )
    if canonical_name == "m2rnn_D":
        return (shape_env.m2rnn_num_heads, shape_env.m2rnn_v_head_dim)
    if canonical_name == "m2rnn_out_proj_weight":
        return (
            hidden,
            shape_env.m2rnn_num_heads * shape_env.m2rnn_v_head_dim,
        )
    if canonical_name == "attention_q_proj_weight":
        return (q_dim, hidden)
    if canonical_name == "attention_sparse_kv_proj_weight":
        return (kv_dim, hidden)
    if canonical_name == "attention_out_proj_weight":
        return (hidden, q_dim)
    if canonical_name == "final_norm_weight":
        return (hidden,)
    if canonical_name == "lm_head_weight":
        vocab = max(1, int(getattr(shape_env, "vocab_size", 0) or 0))
        return (vocab, hidden)
    sequence_hidden_names = {
        "hidden",
        "mamba3_delta",
        "m2rnn_hidden",
        "m2rnn_delta",
        "attention_hidden",
        "hidden_after_mamba3",
        "hidden_after_m2rnn",
        "attention_out",
    }
    ungrad_name = (
        str(buffer_name)[: -len("_grad")]
        if str(buffer_name).endswith("_grad")
        else str(buffer_name)
    )
    if (
        canonical_name in sequence_hidden_names
        or ungrad_name.endswith("_hidden")
        or ungrad_name.endswith("_hidden_after")
        or ungrad_name.endswith("_delta")
        or ungrad_name.endswith("_out")
    ):
        return (1, shape_env.sequence_length, hidden)
    return flat_shape


def _direct_loop_indexed_buffer_ref(
    buffer_name: str,
    shape: Sequence[int],
    loop_extent: int,
    shape_env: PathCModelShapeEnv | None,
) -> str:
    if len(tuple(shape)) > 1:
        return _row_major_buffer_ref(_safe_identifier(buffer_name), shape)
    return _loop_indexed_buffer_ref(buffer_name, shape, loop_extent, shape_env)


def _row_major_buffer_ref(name: str, shape: Sequence[int]) -> str:
    dims = tuple(int(dim) for dim in shape)
    flat_extent = _flattened_extent(dims)
    if len(dims) == 2:
        return (
            f"{name}[(i % {flat_extent}) // {dims[1]}, "
            f"(i % {flat_extent}) % {dims[1]}]"
        )
    if len(dims) == 3:
        if dims[0] == 1:
            return f"{name}[0, i // {dims[2]}, i % {dims[2]}]"
        inner = dims[1] * dims[2]
        return (
            f"{name}[(i % {flat_extent}) // {inner}, "
            f"((i % {flat_extent}) // {dims[2]}) % {dims[1]}, "
            f"(i % {flat_extent}) % {dims[2]}]"
        )
    return f"{name}[i % {flat_extent}]"


def _physical_abi_bank_name(dtype: str) -> str:
    return _safe_identifier(f"path_c_{dtype}_abi_bank")


def _physical_abi_bank_name_for_buffer(
    dtype: str,
    buffer_name: str,
    *,
    physical_abi_policy: str,
) -> str:
    if (
        physical_abi_policy != DESCRIPTOR_PHYSICAL_ABI_POLICY_BANKED_BY_ROLE
        or dtype != "float32"
    ):
        return _physical_abi_bank_name(dtype)
    return _safe_identifier(f"path_c_{dtype}_{_physical_abi_role(buffer_name)}_abi_bank")


def _physical_abi_role(buffer_name: str) -> str:
    name = str(buffer_name)
    if name.endswith("_grad"):
        base = name[: -len("_grad")]
        canonical_base = _canonical_buffer_name(base)
        if _physical_abi_activation_like(base, canonical_base):
            return "activation_gradient"
        return "parameter_gradient"
    canonical = _canonical_buffer_name(buffer_name)
    if (
        canonical.endswith("_weight")
        or canonical.endswith("_bias")
        or canonical in {
            "mamba3_D",
            "mamba3_h0",
            "m2rnn_A_log",
            "m2rnn_D",
            "m2rnn_h0",
            "sparse_mla_sinks",
            "sparse_mla_sm_scale",
        }
    ):
        return "parameter"
    if canonical in {
        "mamba_state",
        "scan_state",
        "mamba3_angle_state",
        "mamba3_angle_checkpoint",
        "mamba3_angle_grad_state",
        "mamba3_conv_state",
        "mamba3_h_checkpoint",
        "m2rnn_conv_state",
        "m2rnn_h_state",
        "m2rnn_h_checkpoint",
    }:
        return "state"
    if canonical in {
        "q_scale",
        "kv_scale",
        "lse",
    }:
        return "attention"
    return "activation"


def _physical_abi_activation_like(buffer_name: str, canonical_name: str) -> bool:
    name = str(buffer_name)
    return (
        canonical_name
        in {
            "hidden",
            "hidden_after_m2rnn",
            "attention_out",
            "mamba3_delta",
            "m2rnn_delta",
        }
        or name.endswith("_hidden")
        or name.endswith("_delta")
        or name.endswith("_out")
    )


def _banked_buffer_ref(
    logical_ref: str,
    *,
    bank_name: str,
    offset: int,
) -> str:
    match = re.fullmatch(r"[A-Za-z_]\w*\[(.+)\]", logical_ref)
    expr = match.group(1) if match is not None else "0"
    if offset == 0:
        return f"{bank_name}[{expr}]"
    return f"{bank_name}[{offset} + ({expr})]"


def _internal_buffer_shapes(
    internal_buffers: Sequence[str],
    internal_buffer_policy: str,
    shape_env: PathCModelShapeEnv | None,
) -> dict[str, tuple[int, ...]]:
    if (
        internal_buffer_policy == DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN
        and shape_env is None
        and internal_buffers
    ):
        raise ValueError(
            "row_local_hidden internal buffer policy requires a model shape_env"
        )
    if (
        internal_buffer_policy
        == DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN
        and shape_env is not None
    ):
        return {
            buffer_name: _row_local_internal_buffer_shape(buffer_name, shape_env)
            for buffer_name in internal_buffers
        }
    return {buffer_name: (1,) for buffer_name in internal_buffers}


def _row_local_internal_buffer_shape(
    buffer_name: str,
    shape_env: PathCModelShapeEnv,
) -> tuple[int, ...]:
    canonical_name = _canonical_buffer_name(buffer_name)
    if str(buffer_name).endswith("_grad") and (
        canonical_name in DESCRIPTOR_ROW_PHASED_BWD_SCRATCH_ABI_CANONICALS
        or str(buffer_name)[: -len("_grad")].endswith(("_hidden", "_delta", "_out"))
    ):
        if canonical_name == "q_fp8":
            return (
                shape_env.sequence_length
                * shape_env.attention_num_q_heads
                * shape_env.attention_head_dim,
            )
        if canonical_name == "q_scale":
            return (shape_env.sequence_length * shape_env.attention_num_q_heads,)
        if canonical_name == "kv_fp8":
            return (
                shape_env.sequence_length
                * shape_env.attention_num_kv_heads
                * shape_env.attention_head_dim,
            )
        if canonical_name == "kv_scale":
            return (shape_env.sequence_length * shape_env.attention_num_kv_heads,)
        return (shape_env.sequence_length * shape_env.hidden_size,)
    if canonical_name in {
        "mamba3_delta",
        "m2rnn_hidden",
        "m2rnn_delta",
        "attention_hidden",
        "hidden_after_mamba3",
    }:
        return (shape_env.sequence_length * shape_env.hidden_size,)
    if canonical_name == "q_fp8":
        return (
            shape_env.sequence_length
            * shape_env.attention_num_q_heads
            * shape_env.attention_head_dim,
        )
    if canonical_name == "q_scale":
        return (shape_env.sequence_length * shape_env.attention_num_q_heads,)
    if canonical_name == "indices":
        return (
            shape_env.sequence_length
            * shape_env.attention_num_kv_heads
            * shape_env.attention_sparse_topk,
        )
    if canonical_name == "kv_fp8":
        return (shape_env.attention_num_kv_heads * shape_env.attention_head_dim,)
    if canonical_name == "kv_scale":
        return (shape_env.attention_num_kv_heads,)
    if canonical_name == "lse":
        return (shape_env.attention_num_q_heads,)
    # Block A: the entry RMSNorm output (and its bwd grad slot) must be
    # full-sequence because the first in-region brick's row-phased body
    # reads the input at ``(row + 1) * H + ...`` (cross-row look-ahead)
    # during the current row's iteration. A row-local scratch would
    # silently alias all rows to the same H slots and corrupt the
    # state recurrence.
    if _is_entry_rmsnorm_output_buffer(buffer_name):
        return (shape_env.sequence_length * shape_env.hidden_size,)
    return (shape_env.hidden_size,)


def _is_entry_rmsnorm_output_buffer(buffer_name: str) -> bool:
    """Return True for the entry RMSNorm forward output / its bwd grad slot."""

    name = str(buffer_name)
    if name.endswith("_grad"):
        name = name[: -len("_grad")]
    return name.endswith("_entry_rmsnorm_hidden")


def _internal_buffer_ref(
    buffer_name: str,
    shape: Sequence[int],
    shape_env: PathCModelShapeEnv | None,
) -> str:
    name = _safe_identifier(buffer_name)
    if shape_env is not None:
        canonical_name = _canonical_buffer_name(buffer_name)
        if str(buffer_name).endswith("_grad"):
            if canonical_name == "q_fp8":
                q_width = shape_env.attention_num_q_heads * shape_env.attention_head_dim
                return (
                    f"{name}[(i // {shape_env.hidden_size}) * {q_width} + "
                    f"(i % {q_width})]"
                )
            if canonical_name == "q_scale":
                return (
                    f"{name}[(i // {shape_env.hidden_size}) * "
                    f"{shape_env.attention_num_q_heads} + "
                    f"((i % {shape_env.hidden_size}) // "
                    f"{shape_env.attention_head_dim})]"
                )
            if canonical_name == "kv_fp8":
                kv_width = (
                    shape_env.attention_num_kv_heads * shape_env.attention_head_dim
                )
                return (
                    f"{name}[(i // {shape_env.hidden_size}) * {kv_width} + "
                    f"(i % {kv_width})]"
                )
            if canonical_name == "kv_scale":
                return (
                    f"{name}[(i // {shape_env.hidden_size}) * "
                    f"{shape_env.attention_num_kv_heads} + "
                    f"(((i % {shape_env.hidden_size}) // "
                    f"{shape_env.attention_head_dim}) % "
                    f"{shape_env.attention_num_kv_heads})]"
                )
        if canonical_name == "q_fp8":
            q_width = shape_env.attention_num_q_heads * shape_env.attention_head_dim
            return (
                f"{name}[(i // {shape_env.hidden_size}) * {q_width} + "
                f"(i % {q_width})]"
            )
        if canonical_name == "q_scale":
            return (
                f"{name}[(i // {shape_env.hidden_size}) * "
                f"{shape_env.attention_num_q_heads} + "
                f"((i % {shape_env.hidden_size}) // "
                f"{shape_env.attention_head_dim})]"
            )
        if canonical_name == "indices":
            row_width = (
                shape_env.attention_num_kv_heads
                * shape_env.attention_sparse_topk
            )
            return f"{name}[(i // {shape_env.hidden_size}) * {row_width} + (i % {row_width})]"
        if canonical_name == "q_scale":
            return (
                f"{name}[(i % {shape_env.hidden_size}) // "
                f"{shape_env.attention_head_dim}]"
            )
        if canonical_name == "kv_scale":
            return (
                f"{name}[((i % {shape_env.hidden_size}) // "
                f"{shape_env.attention_head_dim}) % "
                f"{shape_env.attention_num_kv_heads}]"
            )
        if canonical_name == "kv_fp8":
            return (
                f"{name}[i % "
                f"{shape_env.attention_num_kv_heads * shape_env.attention_head_dim}]"
            )
        if canonical_name == "lse":
            return f"{name}[i % {shape_env.attention_num_q_heads}]"
    if (
        shape_env is not None
        and _flattened_extent(shape)
        == shape_env.sequence_length * shape_env.hidden_size
    ):
        # Block A: full-sequence internal buffer (e.g. entry RMSNorm
        # output). The consumer schedule indexes by absolute ``i``, so
        # the access pattern must NOT wrap modulo H.
        return f"{name}[i]"
    if shape_env is not None and _flattened_extent(shape) == shape_env.hidden_size:
        return f"{name}[i % {shape_env.hidden_size}]"
    return f"{name}[0]"


def _append_descriptor_node_comments(
    body: list[str],
    *,
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
    indent: str,
) -> None:
    body.append(f"{indent}# {node.name}: {node.op_name}")
    body.append(
        f"{indent}# {node.name} production_fragment_status: "
        f"{descriptor.production_fragment_status}"
    )
    if descriptor.production_fragment_reason:
        body.append(
            f"{indent}# {node.name} production_fragment_reason: "
            f"{descriptor.production_fragment_reason}"
        )


def _append_row_phased_hidden_body(
    body: list[str],
    *,
    nodes: Sequence[_ScheduleNodeView],
    descriptors: Sequence[PathCBrickScheduleDescriptor],
    fragments: Sequence[PathCBrickScheduleFragment],
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    shape_env: PathCModelShapeEnv | None,
    train_step_computed_output_buffers: Sequence[str],
    train_step_loss_source_buffers: Sequence[str],
    physical_abi_plan: _PhysicalAbiPlan,
    train_step_loss_cotangent_buffers: Sequence[str],
    train_step_loss_parameter_grad_buffers: Sequence[str],
    max_rows_per_launch: int | None,
    row_dispatch_mode: str,
    rows_per_kernel_launch: int,
    execution_stage: str,
    active_node_names: frozenset[str] | None,
    backward_stage_selector: bool,
    row_dispatch_defined: bool,
    indent: str,
    cuda_target: bool = False,
) -> None:
    if shape_env is None:
        raise ValueError("row_phased_hidden loop policy requires a model shape_env")
    hidden_size = int(shape_env.hidden_size)
    sequence_length = int(shape_env.sequence_length)
    thread_count = min(
        DESCRIPTOR_ROW_PHASED_THREADS,
        max(1, sequence_length * hidden_size),
    )
    chunked_rows = max_rows_per_launch is not None
    row_loop = (
        "for row in T.serial(row_chunk_start, row_chunk_stop):"
        if chunked_rows
        else f"for row in T.serial(0, {sequence_length}):"
    )
    entry_row_loop = (
        "for row in T.serial(row_chunk_start, entry_row_chunk_stop):"
        if chunked_rows
        else row_loop
    )
    row_chunk_expr = (
        DESCRIPTOR_ROW_CHUNK_INDEX_PARAM
        if row_dispatch_mode == DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS
        else "chunk"
    )
    launcher_subchunked_rows = (
        chunked_rows
        and row_dispatch_mode == DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS
    )
    row_chunk_assumptions: tuple[str, ...] = ()
    if launcher_subchunked_rows:
        chunk_count = max(
            1,
            (sequence_length + int(max_rows_per_launch) - 1)
            // int(max_rows_per_launch),
        )
        subchunk_count = max(
            1,
            (int(max_rows_per_launch) + int(rows_per_kernel_launch) - 1)
            // int(rows_per_kernel_launch),
        )
        row_chunk_assumptions = (
            f"{indent * 2}T.assume({row_chunk_expr} >= 0)",
            f"{indent * 2}T.assume({row_chunk_expr} < {chunk_count})",
            f"{indent * 2}T.assume({DESCRIPTOR_ROW_SUBCHUNK_INDEX_PARAM} >= 0)",
            f"{indent * 2}T.assume({DESCRIPTOR_ROW_SUBCHUNK_INDEX_PARAM} < {subchunk_count})",
        )
        if not row_dispatch_defined:
            body.append(
                f"{indent * 2}path_c_first_row_launch = T.if_then_else("
                f"{row_chunk_expr} == 0, T.if_then_else("
                f"{DESCRIPTOR_ROW_SUBCHUNK_INDEX_PARAM} == 0, 1, 0), 0)"
            )
    fwd_items = tuple(
        (node, descriptor, fragment)
        for node, descriptor, fragment in zip(
            nodes, descriptors, fragments, strict=True
        )
        if (
            (active_node_names is None or node.name in active_node_names)
            and (
            execution_stage != DESCRIPTOR_EXECUTION_STAGE_BACKWARD
            and not node.op_name.endswith("_bwd")
            )
        )
    )
    bwd_items = tuple(
        (node, descriptor, fragment)
        for node, descriptor, fragment in zip(
            nodes, descriptors, fragments, strict=True
        )
        if (
            (active_node_names is None or node.name in active_node_names)
            and (
            execution_stage != DESCRIPTOR_EXECUTION_STAGE_FORWARD
            and node.op_name.endswith("_bwd")
            )
        )
    )
    for node, _descriptor, _fragment in fwd_items:
        if node.op_name != "residual_rmsnorm":
            continue
        body.append(
            f"{indent * 2}{_scratch_name(node, 'row_sum_sq_partial')} = "
            f"T.alloc_shared(({thread_count},), \"float32\")"
        )
        body.append(
            f"{indent * 2}{_scratch_name(node, 'row_sum_sq')} = "
            "T.alloc_shared((1,), \"float32\")"
        )
        body.append(
            f"{indent * 2}{_scratch_name(node, 'row_inv_rms')} = "
            "T.alloc_shared((1,), \"float32\")"
        )
    for node, _descriptor, _fragment in fwd_items:
        if node.op_name != "entry_rmsnorm":
            continue
        # Block A: entry RMSNorm reuses the same row-phased reduction
        # scratch layout as residual_rmsnorm. The only difference is the
        # forward math operates on a single ``hidden`` input rather than
        # ``hidden + delta``.
        body.append(
            f"{indent * 2}{_scratch_name(node, 'row_sum_sq_partial')} = "
            f"T.alloc_shared(({thread_count},), \"float32\")"
        )
        body.append(
            f"{indent * 2}{_scratch_name(node, 'row_sum_sq')} = "
            "T.alloc_shared((1,), \"float32\")"
        )
        body.append(
            f"{indent * 2}{_scratch_name(node, 'row_inv_rms')} = "
            "T.alloc_shared((1,), \"float32\")"
        )
    for node, descriptor, _fragment in fwd_items:
        if _is_row_phased_mamba3(node, descriptor, shape_env):
            _append_row_phased_mamba3_init(
                body,
                node=node,
                dtype_by_buffer=dtype_by_buffer,
                access_by_buffer=access_by_buffer,
                shape_env=shape_env,
                thread_count=thread_count,
                indent=indent,
                launcher_chunked_rows=launcher_subchunked_rows,
                cuda_target=cuda_target,
            )
    for node, descriptor, _fragment in fwd_items:
        if _is_row_phased_m2rnn(node, descriptor, shape_env):
            _append_row_phased_m2rnn_init(
                body,
                node=node,
                dtype_by_buffer=dtype_by_buffer,
                access_by_buffer=access_by_buffer,
                shape_env=shape_env,
                thread_count=thread_count,
                indent=indent,
                launcher_chunked_rows=launcher_subchunked_rows,
            )
    for node, _descriptor, _fragment in fwd_items:
        if _is_row_phased_sparse_mla_fp8_apply(node, shape_env):
            q_dim = (
                int(shape_env.attention_num_q_heads)
                * int(shape_env.attention_head_dim)
            )
            body.append(
                f"{indent * 2}{_scratch_name(node, 'context_values')} = "
                f"T.alloc_shared(({q_dim},), \"float32\")"
            )
    for node, _descriptor, _fragment in bwd_items:
        if not _is_row_phased_residual_rmsnorm_bwd(node, _descriptor):
            continue
        body.append(
            f"{indent * 2}{_scratch_name(node, 'row_sum_sq_partial')} = "
            f"T.alloc_shared(({thread_count},), \"float32\")"
        )
        body.append(
            f"{indent * 2}{_scratch_name(node, 'row_dot_partial')} = "
            f"T.alloc_shared(({thread_count},), \"float32\")"
        )
        body.append(
            f"{indent * 2}{_scratch_name(node, 'row_sum_sq')} = "
            "T.alloc_shared((1,), \"float32\")"
        )
        body.append(
            f"{indent * 2}{_scratch_name(node, 'row_inv_rms')} = "
            "T.alloc_shared((1,), \"float32\")"
        )
        body.append(
            f"{indent * 2}{_scratch_name(node, 'row_dot')} = "
            "T.alloc_shared((1,), \"float32\")"
        )
        body.append(
            f"{indent * 2}{_scratch_name(node, 'row_norm_grad')} = "
            "T.alloc_local((1,), \"float32\")"
        )
        body.append(
            f"{indent * 2}{_scratch_name(node, 'row_total_grad')} = "
            "T.alloc_local((1,), \"float32\")"
        )
    for node, _descriptor, _fragment in bwd_items:
        if not _is_row_phased_entry_rmsnorm_bwd(node, _descriptor):
            continue
        # Block A: entry RMSNorm bwd reuses the same per-row scratch
        # layout as residual_rmsnorm_bwd. The recompute math is slightly
        # cheaper because the forward sum-of-squares is taken over
        # ``hidden`` alone (no ``+ delta`` term), but the reduction
        # bookkeeping is identical.
        body.append(
            f"{indent * 2}{_scratch_name(node, 'row_sum_sq_partial')} = "
            f"T.alloc_shared(({thread_count},), \"float32\")"
        )
        body.append(
            f"{indent * 2}{_scratch_name(node, 'row_dot_partial')} = "
            f"T.alloc_shared(({thread_count},), \"float32\")"
        )
        body.append(
            f"{indent * 2}{_scratch_name(node, 'row_sum_sq')} = "
            "T.alloc_shared((1,), \"float32\")"
        )
        body.append(
            f"{indent * 2}{_scratch_name(node, 'row_inv_rms')} = "
            "T.alloc_shared((1,), \"float32\")"
        )
        body.append(
            f"{indent * 2}{_scratch_name(node, 'row_dot')} = "
            "T.alloc_shared((1,), \"float32\")"
        )
        body.append(
            f"{indent * 2}{_scratch_name(node, 'row_norm_grad')} = "
            "T.alloc_local((1,), \"float32\")"
        )
        body.append(
            f"{indent * 2}{_scratch_name(node, 'row_total_grad')} = "
            "T.alloc_local((1,), \"float32\")"
        )
    # Block A: entry RMSNorm produces a full-sequence internal buffer that
    # downstream bricks (mamba3 in particular) read with cross-row
    # look-ahead during their row-phased pass. To make those look-aheads
    # well-defined we materialize the full-sequence entry RMSNorm output
    # in its OWN pre-pass row loop BEFORE the main brick row loop starts.
    entry_rmsnorm_fwd_items = tuple(
        (node, descriptor, fragment)
        for node, descriptor, fragment in fwd_items
        if node.op_name == "entry_rmsnorm"
    )
    other_fwd_items = tuple(
        (node, descriptor, fragment)
        for node, descriptor, fragment in fwd_items
        if node.op_name != "entry_rmsnorm"
    )

    def append_forward_phase(lines: Sequence[str]) -> None:
        if not lines:
            return
        if bwd_items:
            body.append(f"{indent * 2}if {DESCRIPTOR_BACKWARD_GATE_PARAM} != 1:")
            body.extend(f"{indent}{line}" for line in lines)
            return
        body.extend(lines)

    if entry_rmsnorm_fwd_items:
        body.append(
            f"{indent * 2}# entry_rmsnorm_policy: full_sequence_prepass"
        )
        if chunked_rows:
            body.append(f"{indent * 2}# row_dispatch_policy: chunked_rows")
            body.extend(row_chunk_assumptions)
            if launcher_subchunked_rows:
                body.append(
                    f"{indent * 2}logical_row_chunk_start = "
                    f"{row_chunk_expr} * {max_rows_per_launch}"
                )
                body.append(
                    f"{indent * 2}logical_row_chunk_stop = "
                    f"T.min(logical_row_chunk_start + "
                    f"{max_rows_per_launch}, {sequence_length})"
                )
                body.append(
                    f"{indent * 2}subchunk_row_chunk_start = "
                    f"T.min(logical_row_chunk_start + "
                    f"{DESCRIPTOR_ROW_SUBCHUNK_INDEX_PARAM} * "
                    f"{rows_per_kernel_launch}, "
                    "logical_row_chunk_stop)"
                )
                body.append(
                    f"{indent * 2}subchunk_row_chunk_stop = "
                    f"T.min(subchunk_row_chunk_start + "
                    f"{rows_per_kernel_launch}, "
                    "logical_row_chunk_stop)"
                )
                body.append(f"{indent * 2}row_chunk_start = subchunk_row_chunk_start")
                body.append(f"{indent * 2}row_chunk_stop = subchunk_row_chunk_stop")
            else:
                body.append(
                    f"{indent * 2}row_chunk_start = "
                    f"{row_chunk_expr} * {max_rows_per_launch}"
                )
                body.append(
                    f"{indent * 2}row_chunk_stop = T.min(row_chunk_start + "
                    f"{max_rows_per_launch}, {sequence_length})"
                )
            body.append(
                f"{indent * 2}entry_row_chunk_stop = "
                f"T.min(row_chunk_stop + 1, {sequence_length})"
            )
        entry_forward_body: list[str] = []
        entry_forward_body.append(
            f"{indent * 2}{entry_row_loop}"
        )
        for node, descriptor, _fragment in entry_rmsnorm_fwd_items:
            _append_row_phased_entry_rmsnorm_body(
                entry_forward_body,
                node=node,
                descriptor=descriptor,
                dtype_by_buffer=dtype_by_buffer,
                access_by_buffer=access_by_buffer,
                hidden_size=hidden_size,
                thread_count=thread_count,
                indent=indent,
            )
        append_forward_phase(entry_forward_body)
    if other_fwd_items:
        if chunked_rows and not entry_rmsnorm_fwd_items:
            body.append(f"{indent * 2}# row_dispatch_policy: chunked_rows")
            body.extend(row_chunk_assumptions)
            if launcher_subchunked_rows:
                body.append(
                    f"{indent * 2}logical_row_chunk_start = "
                    f"{row_chunk_expr} * {max_rows_per_launch}"
                )
                body.append(
                    f"{indent * 2}logical_row_chunk_stop = "
                    f"T.min(logical_row_chunk_start + "
                    f"{max_rows_per_launch}, {sequence_length})"
                )
                body.append(
                    f"{indent * 2}subchunk_row_chunk_start = "
                    f"T.min(logical_row_chunk_start + "
                    f"{DESCRIPTOR_ROW_SUBCHUNK_INDEX_PARAM} * "
                    f"{rows_per_kernel_launch}, "
                    "logical_row_chunk_stop)"
                )
                body.append(
                    f"{indent * 2}subchunk_row_chunk_stop = "
                    f"T.min(subchunk_row_chunk_start + "
                    f"{rows_per_kernel_launch}, "
                    "logical_row_chunk_stop)"
                )
                body.append(f"{indent * 2}row_chunk_start = subchunk_row_chunk_start")
                body.append(f"{indent * 2}row_chunk_stop = subchunk_row_chunk_stop")
            else:
                body.append(
                    f"{indent * 2}row_chunk_start = "
                    f"{row_chunk_expr} * {max_rows_per_launch}"
                )
                body.append(
                    f"{indent * 2}row_chunk_stop = T.min(row_chunk_start + "
                    f"{max_rows_per_launch}, {sequence_length})"
                )
        other_forward_body: list[str] = []
        other_forward_body.append(f"{indent * 2}{row_loop}")
        for node, descriptor, fragment in other_fwd_items:
            if _is_row_phased_mamba3(node, descriptor, shape_env):
                _append_row_phased_mamba3_body(
                    other_forward_body,
                    node=node,
                    descriptor=descriptor,
                    dtype_by_buffer=dtype_by_buffer,
                    access_by_buffer=access_by_buffer,
                    shape_env=shape_env,
                    thread_count=thread_count,
                    indent=indent,
                    launcher_chunked_rows=launcher_subchunked_rows,
                    cuda_target=cuda_target,
                )
                continue
            if node.op_name == "residual_rmsnorm":
                _append_row_phased_residual_rmsnorm_body(
                    other_forward_body,
                    node=node,
                    descriptor=descriptor,
                    dtype_by_buffer=dtype_by_buffer,
                    access_by_buffer=access_by_buffer,
                    hidden_size=hidden_size,
                    thread_count=thread_count,
                    indent=indent,
                )
                continue
            if _is_row_phased_attention_qkv_projection(node, shape_env):
                _append_row_phased_attention_qkv_projection_body(
                    other_forward_body,
                    node=node,
                    descriptor=descriptor,
                    dtype_by_buffer=dtype_by_buffer,
                    access_by_buffer=access_by_buffer,
                    shape_env=shape_env,
                    thread_count=thread_count,
                    indent=indent,
                )
                continue
            if _is_row_phased_m2rnn(node, descriptor, shape_env):
                _append_row_phased_m2rnn_body(
                    other_forward_body,
                    node=node,
                    descriptor=descriptor,
                    dtype_by_buffer=dtype_by_buffer,
                    access_by_buffer=access_by_buffer,
                    shape_env=shape_env,
                    thread_count=thread_count,
                    indent=indent,
                    launcher_chunked_rows=launcher_subchunked_rows,
                )
                continue
            if _is_row_phased_sparse_mla_fp8_apply(node, shape_env):
                _append_row_phased_sparse_mla_fp8_apply_body(
                    other_forward_body,
                    node=node,
                    descriptor=descriptor,
                    dtype_by_buffer=dtype_by_buffer,
                    access_by_buffer=access_by_buffer,
                    shape_env=shape_env,
                    thread_count=thread_count,
                    indent=indent,
                )
                continue
            other_forward_body.append(
                f"{indent * 3}for i in T.serial(row * {hidden_size} + lane, "
                f"(row + 1) * {hidden_size}, step={thread_count}):"
            )
            _append_descriptor_node_comments(
                other_forward_body,
                node=node,
                descriptor=descriptor,
                indent=indent * 4,
            )
            for statement in fragment.statements:
                other_forward_body.append(f"{indent * 4}{statement}")
            other_forward_body.append(f"{indent * 3}T.sync_threads()")
        append_forward_phase(other_forward_body)
    if (
        bwd_items
        and not fwd_items
        and chunked_rows
        and not launcher_subchunked_rows
        and not row_dispatch_defined
    ):
        body.append(f"{indent * 2}# row_dispatch_policy: chunked_rows")
        body.append(
            f"{indent * 2}row_chunk_start = "
            f"{row_chunk_expr} * {max_rows_per_launch}"
        )
        body.append(
            f"{indent * 2}row_chunk_stop = T.min(row_chunk_start + "
            f"{max_rows_per_launch}, {sequence_length})"
        )
    if (
        bwd_items
        and not fwd_items
        and chunked_rows
        and launcher_subchunked_rows
        and not row_dispatch_defined
    ):
        body.append(f"{indent * 2}# row_dispatch_policy: chunked_rows")
        body.extend(row_chunk_assumptions)
        body.append(
            f"{indent * 2}logical_row_chunk_start = "
            f"{row_chunk_expr} * {max_rows_per_launch}"
        )
        body.append(
            f"{indent * 2}logical_row_chunk_stop = "
            f"T.min(logical_row_chunk_start + "
            f"{max_rows_per_launch}, {sequence_length})"
        )
        body.append(
            f"{indent * 2}subchunk_row_chunk_start = "
            f"T.min(logical_row_chunk_start + "
            f"{DESCRIPTOR_ROW_SUBCHUNK_INDEX_PARAM} * "
            f"{rows_per_kernel_launch}, "
            "logical_row_chunk_stop)"
        )
        body.append(
            f"{indent * 2}subchunk_row_chunk_stop = "
            f"T.min(subchunk_row_chunk_start + "
            f"{rows_per_kernel_launch}, "
            "logical_row_chunk_stop)"
        )
        body.append(f"{indent * 2}row_chunk_start = subchunk_row_chunk_start")
        body.append(f"{indent * 2}row_chunk_stop = subchunk_row_chunk_stop")
    if not bwd_items:
        _append_train_step_suffix_scalar_outputs(
            body,
            computed_outputs=train_step_computed_output_buffers,
            loss_source_buffers=train_step_loss_source_buffers,
            train_step_loss_cotangent_buffers=train_step_loss_cotangent_buffers,
            train_step_loss_parameter_grad_buffers=(
                train_step_loss_parameter_grad_buffers
            ),
            physical_abi_plan=physical_abi_plan,
            loop_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
            shape_env=shape_env,
            indent=indent,
        )
        return
    if backward_stage_selector and execution_stage == DESCRIPTOR_EXECUTION_STAGE_ALL:
        stage_name_groups = _path_c_backward_stage_name_groups_for_nodes(
            tuple(node for node, _descriptor, _fragment in bwd_items)
        )
        if stage_name_groups:
            if chunked_rows and launcher_subchunked_rows and not fwd_items:
                body.append(f"{indent * 2}# row_dispatch_policy: chunked_rows")
                body.extend(row_chunk_assumptions)
                body.append(
                    f"{indent * 2}logical_row_chunk_start = "
                    f"{row_chunk_expr} * {max_rows_per_launch}"
                )
                body.append(
                    f"{indent * 2}logical_row_chunk_stop = "
                    f"T.min(logical_row_chunk_start + "
                    f"{max_rows_per_launch}, {sequence_length})"
                )
                body.append(
                    f"{indent * 2}subchunk_row_chunk_start = "
                    f"T.min(logical_row_chunk_start + "
                    f"{DESCRIPTOR_ROW_SUBCHUNK_INDEX_PARAM} * "
                    f"{rows_per_kernel_launch}, "
                    "logical_row_chunk_stop)"
                )
                body.append(
                    f"{indent * 2}subchunk_row_chunk_stop = "
                    f"T.min(subchunk_row_chunk_start + "
                    f"{rows_per_kernel_launch}, "
                    "logical_row_chunk_stop)"
                )
                body.append(
                    f"{indent * 2}row_chunk_start = subchunk_row_chunk_start"
                )
                body.append(f"{indent * 2}row_chunk_stop = subchunk_row_chunk_stop")
            body.append(f"{indent * 2}if {DESCRIPTOR_BACKWARD_GATE_PARAM} == 1:")
            for stage_index, stage_names in enumerate(stage_name_groups):
                stage_lines: list[str] = []
                _append_row_phased_hidden_body(
                    stage_lines,
                    nodes=nodes,
                    descriptors=descriptors,
                    fragments=fragments,
                    dtype_by_buffer=dtype_by_buffer,
                    access_by_buffer=access_by_buffer,
                    shape_env=shape_env,
                    train_step_computed_output_buffers=(
                        train_step_computed_output_buffers
                    ),
                    train_step_loss_source_buffers=train_step_loss_source_buffers,
                    physical_abi_plan=physical_abi_plan,
                    train_step_loss_cotangent_buffers=(
                        train_step_loss_cotangent_buffers
                    ),
                    train_step_loss_parameter_grad_buffers=(
                        train_step_loss_parameter_grad_buffers
                    ),
                    max_rows_per_launch=max_rows_per_launch,
                    row_dispatch_mode=row_dispatch_mode,
                    rows_per_kernel_launch=rows_per_kernel_launch,
                    execution_stage=DESCRIPTOR_EXECUTION_STAGE_BACKWARD,
                    active_node_names=frozenset(stage_names),
                    backward_stage_selector=False,
                    row_dispatch_defined=True,
                    indent=indent,
                )
                if not stage_lines:
                    continue
                body.append(
                    f"{indent * 3}if {DESCRIPTOR_BACKWARD_STAGE_INDEX_PARAM} "
                    f"== {stage_index}:"
                )
                body.extend(f"{indent * 2}{line}" for line in stage_lines)
            return
    bwd_body: list[str] = []
    bwd_once_body: list[str] = []
    _append_train_step_suffix_scalar_outputs(
        bwd_once_body,
        computed_outputs=train_step_computed_output_buffers,
        loss_source_buffers=train_step_loss_source_buffers,
        train_step_loss_cotangent_buffers=train_step_loss_cotangent_buffers,
        train_step_loss_parameter_grad_buffers=(
            train_step_loss_parameter_grad_buffers
        ),
        physical_abi_plan=physical_abi_plan,
        loop_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
        shape_env=shape_env,
        indent=indent,
    )
    row_phased_bwd_items = tuple(
        (node, descriptor)
        for node, descriptor, _fragment in bwd_items
        if _is_row_phased_bwd_descriptor(node, descriptor, shape_env)
    )
    if not row_phased_bwd_items:
        _reject_proxy_backward_fragments(bwd_items)
        if bwd_once_body:
            if launcher_subchunked_rows:
                bwd_body.append(f"{indent * 2}if path_c_first_row_launch != 0:")
                bwd_body.extend(f"{indent}{line}" for line in bwd_once_body)
            else:
                bwd_body.extend(bwd_once_body)
        bwd_body.append(
            f"{indent * 2}# backward_policy: flat_after_row_phased_forward"
        )
        bwd_body.append(
            f"{indent * 2}for i in T.serial(lane, "
            f"{sequence_length * hidden_size}, step={thread_count}):"
        )
        for node, descriptor, fragment in bwd_items:
            _append_descriptor_node_comments(
                bwd_body,
                node=node,
                descriptor=descriptor,
                indent=indent * 3,
            )
            for statement in fragment.statements:
                bwd_body.append(f"{indent * 3}{statement}")
            bwd_body.append(f"{indent * 3}T.sync_threads()")
        if execution_stage == DESCRIPTOR_EXECUTION_STAGE_BACKWARD:
            body.extend(bwd_body)
        else:
            body.append(f"{indent * 2}if {DESCRIPTOR_BACKWARD_GATE_PARAM} == 1:")
            body.extend(f"{indent}{line}" for line in bwd_body)
        return
    for node, _descriptor, _fragment in bwd_items:
        if _is_row_phased_residual_rmsnorm_bwd(node, _descriptor):
            _append_row_phased_residual_rmsnorm_bwd_init(
                bwd_once_body,
                node=node,
                access_by_buffer=access_by_buffer,
                hidden_size=hidden_size,
                thread_count=thread_count,
                indent=indent,
            )
    for node, _descriptor, _fragment in bwd_items:
        if _is_row_phased_entry_rmsnorm_bwd(node, _descriptor):
            _append_row_phased_entry_rmsnorm_bwd_init(
                bwd_once_body,
                node=node,
                access_by_buffer=access_by_buffer,
                hidden_size=hidden_size,
                sequence_length=sequence_length,
                thread_count=thread_count,
                indent=indent,
            )
    if bwd_once_body:
        if launcher_subchunked_rows:
            bwd_body.append(f"{indent * 2}if path_c_first_row_launch != 0:")
            bwd_body.extend(f"{indent}{line}" for line in bwd_once_body)
        else:
            bwd_body.extend(bwd_once_body)
    def append_row_phased_bwd_item(
        node: _ScheduleNodeView,
        descriptor: PathCBrickScheduleDescriptor,
        fragment: Any,
    ) -> None:
        if _is_row_phased_residual_rmsnorm_bwd(node, descriptor):
            _append_row_phased_residual_rmsnorm_bwd_body(
                bwd_body,
                node=node,
                descriptor=descriptor,
                dtype_by_buffer=dtype_by_buffer,
                access_by_buffer=access_by_buffer,
                hidden_size=hidden_size,
                thread_count=thread_count,
                indent=indent,
            )
            return
        if _is_row_phased_sparse_mla_fp8_apply_bwd(node, descriptor, shape_env):
            _append_row_phased_sparse_mla_fp8_apply_bwd_body(
                bwd_body,
                node=node,
                descriptor=descriptor,
                dtype_by_buffer=dtype_by_buffer,
                access_by_buffer=access_by_buffer,
                shape_env=shape_env,
                thread_count=thread_count,
                chunked_rows=chunked_rows,
                indent=indent,
            )
            return
        if _is_row_phased_attention_qkv_projection_bwd(node, descriptor, shape_env):
            _append_row_phased_attention_qkv_projection_bwd_body(
                bwd_body,
                node=node,
                descriptor=descriptor,
                dtype_by_buffer=dtype_by_buffer,
                access_by_buffer=access_by_buffer,
                shape_env=shape_env,
                thread_count=thread_count,
                indent=indent,
            )
            return
        if _is_row_phased_m2rnn_bwd(node, descriptor, shape_env):
            _append_row_phased_m2rnn_bwd_body(
                bwd_body,
                node=node,
                descriptor=descriptor,
                dtype_by_buffer=dtype_by_buffer,
                access_by_buffer=access_by_buffer,
                shape_env=shape_env,
                thread_count=thread_count,
                indent=indent,
                launcher_chunked_rows=launcher_subchunked_rows,
            )
            return
        if _is_row_phased_mamba3_bwd(node, descriptor, shape_env):
            _append_row_phased_mamba3_bwd_body(
                bwd_body,
                node=node,
                descriptor=descriptor,
                dtype_by_buffer=dtype_by_buffer,
                access_by_buffer=access_by_buffer,
                shape_env=shape_env,
                thread_count=thread_count,
                indent=indent,
                launcher_chunked_rows=launcher_subchunked_rows,
            )
            return
        if _is_row_phased_entry_rmsnorm_bwd(node, descriptor):
            _append_row_phased_entry_rmsnorm_bwd_body(
                bwd_body,
                node=node,
                descriptor=descriptor,
                dtype_by_buffer=dtype_by_buffer,
                access_by_buffer=access_by_buffer,
                hidden_size=hidden_size,
                thread_count=thread_count,
                indent=indent,
            )
            return
        _reject_proxy_backward_fragments(((node, descriptor, fragment),))
        bwd_body.append(
            f"{indent * 3}for i in T.serial(row * {hidden_size} + lane, "
            f"(row + 1) * {hidden_size}, step={thread_count}):"
        )
        _append_descriptor_node_comments(
            bwd_body,
            node=node,
            descriptor=descriptor,
            indent=indent * 4,
        )
        for statement in fragment.statements:
            bwd_body.append(f"{indent * 4}{statement}")
        bwd_body.append(f"{indent * 3}T.sync_threads()")

    bwd_row_loop = row_loop
    bwd_body.append(f"{indent * 2}# backward_policy: row_phased_hidden_recompute")
    bwd_body.append(f"{indent * 2}# backward_phase: full_sparse_mla_before_projection")
    for node, descriptor, fragment in bwd_items:
        if not _is_row_phased_sparse_mla_fp8_apply_bwd(node, descriptor, shape_env):
            continue
        bwd_body.append(f"{indent * 2}{bwd_row_loop}")
        append_row_phased_bwd_item(node, descriptor, fragment)
    bwd_body.append(f"{indent * 2}# backward_phase: projection_and_upstream")
    for node, descriptor, fragment in bwd_items:
        if _is_row_phased_sparse_mla_fp8_apply_bwd(node, descriptor, shape_env):
            continue
        bwd_body.append(f"{indent * 2}{bwd_row_loop}")
        append_row_phased_bwd_item(node, descriptor, fragment)
    if execution_stage == DESCRIPTOR_EXECUTION_STAGE_BACKWARD:
        body.extend(bwd_body)
    else:
        body.append(f"{indent * 2}if {DESCRIPTOR_BACKWARD_GATE_PARAM} == 1:")
        body.extend(f"{indent}{line}" for line in bwd_body)


def _append_lane0_row_phase(
    body: list[str],
    *,
    indent: str,
    append_fn: Callable[..., None],
    **kwargs: Any,
) -> None:
    lane0_body: list[str] = []
    append_fn(lane0_body, indent=indent, **kwargs)
    body.append(f"{indent * 3}if lane == 0:")
    body.extend(f"{indent}{line}" for line in lane0_body)
    body.append(f"{indent * 3}T.sync_threads()")


def _is_row_phased_residual_rmsnorm_bwd(
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
) -> bool:
    return (
        node.op_name == "residual_rmsnorm_bwd"
        and descriptor.production_fragment_status == "production_region_inlined"
    )


def _is_row_phased_entry_rmsnorm(
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
    shape_env: PathCModelShapeEnv | None,
) -> bool:
    return (
        shape_env is not None
        and node.op_name == "entry_rmsnorm"
        and descriptor.production_fragment_status == "production_region_inlined"
    )


def _is_row_phased_entry_rmsnorm_bwd(
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
) -> bool:
    return (
        node.op_name == "entry_rmsnorm_bwd"
        and descriptor.production_fragment_status == "production_region_inlined"
    )


def _is_row_phased_attention_qkv_projection_bwd(
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
    shape_env: PathCModelShapeEnv | None,
) -> bool:
    return (
        shape_env is not None
        and node.op_name == "attention_qkv_projection_bwd"
        and descriptor.production_fragment_status == "production_region_inlined"
    )


def _is_row_phased_sparse_mla_fp8_apply_bwd(
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
    shape_env: PathCModelShapeEnv | None,
) -> bool:
    if (
        shape_env is None
        or node.op_name != "sparse_mla_fp8_apply_bwd"
        or descriptor.production_fragment_status != "production_region_inlined"
    ):
        return False
    input_canonicals = {_canonical_buffer_name(input_name) for input_name in node.inputs}
    output_canonicals = {_canonical_buffer_name(output) for output in node.outputs}
    return {"q_fp8", "q_scale", "kv_fp8", "kv_scale", "indices"}.issubset(
        input_canonicals
    ) and {"q_fp8", "kv_fp8"}.issubset(output_canonicals)


def _is_row_phased_m2rnn_bwd(
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
    shape_env: PathCModelShapeEnv | None,
) -> bool:
    return (
        shape_env is not None
        and node.op_name == "m2rnn_bwd"
        and descriptor.production_fragment_status == "production_region_inlined"
    )


def _is_row_phased_mamba3_bwd(
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
    shape_env: PathCModelShapeEnv | None,
) -> bool:
    return (
        shape_env is not None
        and node.op_name == "mamba3_mimo_bwd"
        and descriptor.production_fragment_status == "production_region_inlined"
    )


def _is_row_phased_bwd_descriptor(
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
    shape_env: PathCModelShapeEnv | None,
) -> bool:
    return (
        _is_row_phased_residual_rmsnorm_bwd(node, descriptor)
        or _is_row_phased_sparse_mla_fp8_apply_bwd(node, descriptor, shape_env)
        or _is_row_phased_attention_qkv_projection_bwd(node, descriptor, shape_env)
        or _is_row_phased_m2rnn_bwd(node, descriptor, shape_env)
        or _is_row_phased_mamba3_bwd(node, descriptor, shape_env)
        or _is_row_phased_entry_rmsnorm_bwd(node, descriptor)
    )


def _reject_proxy_backward_fragments(
    items: Sequence[
        tuple[
            _ScheduleNodeView,
            PathCBrickScheduleDescriptor,
            PathCBrickScheduleFragment,
        ]
    ],
) -> None:
    for node, descriptor, _fragment in items:
        if node.op_name not in _EXACT_ROW_PHASED_BACKWARD_OPS:
            continue
        raise ValueError(
            f"{node.op_name} requires the row-phased exact backward generator; "
            "refusing to emit a scalar proxy backward fragment"
        )


def _is_row_phased_attention_qkv_projection(
    node: _ScheduleNodeView,
    shape_env: PathCModelShapeEnv | None,
) -> bool:
    if shape_env is None or node.op_name != "attention_qkv_projection":
        return False
    output_canonicals = {_canonical_buffer_name(output) for output in node.outputs}
    return {"q_fp8", "q_scale", "kv_fp8", "kv_scale"}.issubset(output_canonicals)


def _is_row_phased_sparse_mla_fp8_apply(
    node: _ScheduleNodeView,
    shape_env: PathCModelShapeEnv | None,
) -> bool:
    if shape_env is None or node.op_name != "sparse_mla_fp8_apply":
        return False
    input_canonicals = {_canonical_buffer_name(input_name) for input_name in node.inputs}
    return {"q_fp8", "q_scale", "kv_fp8", "kv_scale", "indices"}.issubset(
        input_canonicals
    )


def _is_row_phased_mamba3(
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
    shape_env: PathCModelShapeEnv | None,
) -> bool:
    return (
        shape_env is not None
        and node.op_name == "mamba3_mimo"
        and descriptor.production_fragment_status == "production_region_inlined"
    )


def _is_row_phased_m2rnn(
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
    shape_env: PathCModelShapeEnv | None,
) -> bool:
    return (
        shape_env is not None
        and node.op_name == "m2rnn"
        and descriptor.production_fragment_status == "production_region_inlined"
    )


def _node_indexed_canonical_input_expr(
    node: _ScheduleNodeView,
    canonical_name: str,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    index_expr: str,
    *,
    default: str = "0.0",
) -> str:
    input_name = _node_input_for_canonical(node, canonical_name)
    if input_name is None:
        return default
    return _indexed_buffer_value_expr(
        input_name,
        dtype_by_buffer[input_name],
        access_by_buffer,
        index_expr,
    )


def _node_indexed_canonical_or_positional_input_expr(
    node: _ScheduleNodeView,
    canonical_names: Sequence[str],
    positional_index: int,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    index_expr: str,
    *,
    default: str = "0.0",
) -> str:
    input_name: str | None = None
    for canonical_name in canonical_names:
        input_name = _node_input_for_canonical(node, canonical_name)
        if input_name is not None:
            break
    if input_name is None:
        if positional_index >= len(node.inputs):
            return default
        input_name = node.inputs[positional_index]
    return _indexed_buffer_value_expr(
        input_name,
        dtype_by_buffer[input_name],
        access_by_buffer,
        index_expr,
    )


def _node_indexed_positional_input_expr(
    node: _ScheduleNodeView,
    input_index: int,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    index_expr: str,
    *,
    default: str = "0.0",
) -> str:
    if input_index >= len(node.inputs):
        return default
    input_name = node.inputs[input_index]
    return _indexed_buffer_value_expr(
        input_name,
        dtype_by_buffer[input_name],
        access_by_buffer,
        index_expr,
    )


def _mamba3_state_output(node: _ScheduleNodeView) -> str | None:
    output = _node_output_for_canonical(node, "scan_state")
    if output is not None:
        return output
    output = _node_output_for_canonical(node, "mamba_state")
    if output is not None:
        return output
    for output_name in node.outputs:
        if output_name.endswith("_state") or output_name.endswith("_state_out"):
            return output_name
    return node.outputs[1] if len(node.outputs) > 1 else None


def _append_row_phased_mamba3_init(
    body: list[str],
    *,
    node: _ScheduleNodeView,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    shape_env: PathCModelShapeEnv,
    thread_count: int,
    indent: str,
    launcher_chunked_rows: bool = False,
    cuda_target: bool = False,
) -> None:
    projected = _scratch_name(node, "mamba3_projected_vec")
    conv_history = _scratch_name(node, "mamba3_conv_history")
    conv = _scratch_name(node, "mamba3_conv_vec")
    b_inv = _scratch_name(node, "mamba3_b_inv_rms")
    c_inv = _scratch_name(node, "mamba3_c_inv_rms")
    b_group = _scratch_name(node, "mamba3_b_group")
    c_group = _scratch_name(node, "mamba3_c_group")
    b_raw = _scratch_name(node, "mamba3_b_raw")
    c_raw = _scratch_name(node, "mamba3_c_raw")
    dt_vec = _scratch_name(node, "mamba3_dt_vec")
    a_vec = _scratch_name(node, "mamba3_a_vec")
    trap_group = _scratch_name(node, "mamba3_trap_group")
    next_dt = _scratch_name(node, "mamba3_next_dt")
    next_trap = _scratch_name(node, "mamba3_next_trap")
    # CUDA-only thread-parallel trapezoid scratch (declared only when
    # cuda_target so the Metal kernel source is byte-for-byte unchanged).
    next_dt_vec = _scratch_name(node, "mamba3_next_dt_vec")
    next_trap_vec = _scratch_name(node, "mamba3_next_trap_vec")
    angle_cumsum = _scratch_name(node, "mamba3_angle_cumsum")
    out_inner = _scratch_name(node, "mamba3_out_inner")
    accum = _scratch_name(node, "mamba3_accum")
    state_value = _scratch_name(node, "mamba3_state_value")
    inner_dim = int(shape_env.mamba_inner_dim)
    in_proj_dim = int(shape_env.mamba_in_proj_dim)
    conv_channels = int(shape_env.mamba_conv_channels)
    history_len = max(0, int(shape_env.mamba_conv_kernel) - 1)
    heads = int(shape_env.mamba_num_heads)
    head_dim = int(shape_env.mamba_head_dim)
    state_dim = int(shape_env.mamba_state_dim)
    groups = int(shape_env.mamba_groups)
    rank = int(shape_env.mamba_effective_mimo_rank)
    rope_angles = int(shape_env.mamba_num_rope_angles)
    state_output = _mamba3_state_output(node)
    body.append(
        f"{indent * 2}{projected} = T.alloc_shared(({in_proj_dim},), \"float32\")"
    )
    body.append(
        f"{indent * 2}{conv} = T.alloc_shared(({conv_channels},), \"float32\")"
    )
    body.append(f"{indent * 2}{out_inner} = T.alloc_shared(({inner_dim},), \"float32\")")
    body.append(
        f"{indent * 2}{b_inv} = T.alloc_shared(({rank}, {groups}), \"float32\")"
    )
    body.append(
        f"{indent * 2}{c_inv} = T.alloc_shared(({rank}, {groups}), \"float32\")"
    )
    body.append(
        f"{indent * 2}{b_raw} = T.alloc_shared(({groups}, {state_dim}), \"float32\")"
    )
    body.append(
        f"{indent * 2}{c_raw} = T.alloc_shared(({groups}, {state_dim}), \"float32\")"
    )
    body.append(
        f"{indent * 2}{b_group} = T.alloc_shared(({groups}, {state_dim}), \"float32\")"
    )
    body.append(
        f"{indent * 2}{c_group} = T.alloc_shared(({groups}, {state_dim}), \"float32\")"
    )
    body.append(f"{indent * 2}{dt_vec} = T.alloc_shared(({heads},), \"float32\")")
    body.append(f"{indent * 2}{a_vec} = T.alloc_shared(({heads},), \"float32\")")
    body.append(f"{indent * 2}{trap_group} = T.alloc_shared(({groups},), \"float32\")")
    body.append(f"{indent * 2}{next_dt} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 2}{next_trap} = T.alloc_local((1,), \"float32\")")
    if cuda_target:
        # CUDA-only: per-head next-row dt/trap projection scratch. These are in
        # the force-spill list, so they alias into the ABI float32 scratch bank
        # and do not consume the per-block shared-memory budget.
        body.append(
            f"{indent * 2}{next_dt_vec} = T.alloc_shared(({heads},), \"float32\")"
        )
        body.append(
            f"{indent * 2}{next_trap_vec} = T.alloc_shared(({heads},), \"float32\")"
        )
    body.append(
        f"{indent * 2}{angle_cumsum} = "
        f"T.alloc_shared(({heads}, {rope_angles}), \"float32\")"
    )
    if history_len > 0:
        body.append(
            f"{indent * 2}{conv_history} = "
            f"T.alloc_shared(({history_len}, {conv_channels}), \"float32\")"
        )
    body.append(f"{indent * 2}{accum} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 2}{state_value} = T.alloc_local((1,), \"float32\")")
    head = _scratch_name(node, "head_init")
    dim = _scratch_name(node, "dim_init")
    state = _scratch_name(node, "state_idx_init")
    angle = _scratch_name(node, "angle_init")
    hist = _scratch_name(node, "hist_init")
    ch = _scratch_name(node, "conv_ch_init")
    state_flat = _scratch_name(node, "state_flat_init")
    angle_flat = _scratch_name(node, "angle_flat_init")
    history_flat = _scratch_name(node, "history_flat_init")
    body.append(f"{indent * 2}# {node.name}: mamba3_state_policy: external_scan_state")
    if launcher_chunked_rows:
        body.append(f"{indent * 2}if path_c_first_row_launch != 0:")
        body.append(
            f"{indent * 3}for {angle_flat} in T.serial(lane, {heads * rope_angles}, "
            f"step={thread_count}):"
        )
        body.append(f"{indent * 4}{head} = {angle_flat} // {rope_angles}")
        body.append(f"{indent * 4}{angle} = {angle_flat} % {rope_angles}")
        body.append(f"{indent * 4}{angle_cumsum}[{head}, {angle}] = 0.0")
        body.append(f"{indent * 2}else:")
        body.append(
            f"{indent * 3}for {angle_flat} in T.serial(lane, {heads * rope_angles}, "
            f"step={thread_count}):"
        )
        body.append(f"{indent * 4}{head} = {angle_flat} // {rope_angles}")
        body.append(f"{indent * 4}{angle} = {angle_flat} % {rope_angles}")
        angle_state_ref = _buffer_ref(
            "mamba3_angle_state",
            access_by_buffer,
            f"{head} * {rope_angles} + {angle}",
        )
        body.append(
            f"{indent * 4}{angle_cumsum}[{head}, {angle}] = {angle_state_ref}"
        )
    else:
        body.append(
            f"{indent * 2}for {angle_flat} in T.serial(lane, {heads * rope_angles}, "
            f"step={thread_count}):"
        )
        body.append(f"{indent * 3}{head} = {angle_flat} // {rope_angles}")
        body.append(f"{indent * 3}{angle} = {angle_flat} % {rope_angles}")
        body.append(f"{indent * 3}{angle_cumsum}[{head}, {angle}] = 0.0")
    angle_checkpoint_ref = _buffer_ref(
        "mamba3_angle_checkpoint",
        access_by_buffer,
        f"{head} * {rope_angles} + {angle}",
    )
    if launcher_chunked_rows:
        body.append(f"{indent * 2}if path_c_first_row_launch != 0:")
        angle_checkpoint_loop_indent = indent * 3
        angle_checkpoint_body_indent = indent * 4
    else:
        angle_checkpoint_loop_indent = indent * 2
        angle_checkpoint_body_indent = indent * 3
    body.append(
        f"{angle_checkpoint_loop_indent}for {angle_flat} in T.serial(lane, "
        f"{heads * rope_angles}, step={thread_count}):"
    )
    body.append(f"{angle_checkpoint_body_indent}{head} = {angle_flat} // {rope_angles}")
    body.append(f"{angle_checkpoint_body_indent}{angle} = {angle_flat} % {rope_angles}")
    body.append(
        f"{angle_checkpoint_body_indent}{angle_checkpoint_ref} = "
        f"{angle_cumsum}[{head}, {angle}]"
    )
    if state_output is not None:
        if launcher_chunked_rows:
            body.append(f"{indent * 2}if path_c_first_row_launch != 0:")
            state_loop_indent = indent * 3
            state_body_indent = indent * 4
        else:
            state_loop_indent = indent * 2
            state_body_indent = indent * 3
        body.append(
            f"{state_loop_indent}for {state_flat} in T.serial(lane, {heads * head_dim * state_dim}, "
            f"step={thread_count}):"
        )
        body.append(f"{state_body_indent}{head} = {state_flat} // {head_dim * state_dim}")
        body.append(f"{state_body_indent}{dim} = ({state_flat} // {state_dim}) % {head_dim}")
        body.append(f"{state_body_indent}{state} = {state_flat} % {state_dim}")
        state_idx = f"{head} * {head_dim * state_dim} + {dim} * {state_dim} + {state}"
        h0_expr = _node_indexed_canonical_input_expr(
            node,
            "mamba3_h0",
            dtype_by_buffer,
            access_by_buffer,
            state_idx,
            default=_node_indexed_positional_input_expr(
                node,
                1,
                dtype_by_buffer,
                access_by_buffer,
                state_idx,
            ),
        )
        body.append(
            f"{state_body_indent}{_buffer_ref(state_output, access_by_buffer, state_idx)} = "
            f"{h0_expr}"
        )
        h_checkpoint_ref = _buffer_ref(
            "mamba3_h_checkpoint",
            access_by_buffer,
            state_idx,
        )
        body.append(f"{state_body_indent}{h_checkpoint_ref} = {h0_expr}")
    body.append(f"{indent * 2}T.sync_threads()")
    if history_len <= 0:
        return
    body.append(f"{indent * 2}# {node.name}: mamba3_conv_policy: zero_padded_ring_history")
    if launcher_chunked_rows:
        body.append(f"{indent * 2}if path_c_first_row_launch != 0:")
        body.append(
            f"{indent * 3}for {history_flat} in T.serial(lane, "
            f"{history_len * conv_channels}, step={thread_count}):"
        )
        body.append(f"{indent * 4}{hist} = {history_flat} // {conv_channels}")
        body.append(f"{indent * 4}{ch} = {history_flat} % {conv_channels}")
        body.append(f"{indent * 4}{conv_history}[{hist}, {ch}] = 0.0")
        body.append(f"{indent * 2}else:")
        body.append(
            f"{indent * 3}for {history_flat} in T.serial(lane, "
            f"{history_len * conv_channels}, step={thread_count}):"
        )
        body.append(f"{indent * 4}{hist} = {history_flat} // {conv_channels}")
        body.append(f"{indent * 4}{ch} = {history_flat} % {conv_channels}")
        conv_state_ref = _buffer_ref(
            "mamba3_conv_state",
            access_by_buffer,
            f"{hist} * {conv_channels} + {ch}",
        )
        body.append(
            f"{indent * 4}{conv_history}[{hist}, {ch}] = {conv_state_ref}"
        )
    else:
        body.append(
            f"{indent * 2}for {history_flat} in T.serial(lane, "
            f"{history_len * conv_channels}, step={thread_count}):"
        )
        body.append(f"{indent * 3}{hist} = {history_flat} // {conv_channels}")
        body.append(f"{indent * 3}{ch} = {history_flat} % {conv_channels}")
        body.append(f"{indent * 3}{conv_history}[{hist}, {ch}] = 0.0")
    body.append(f"{indent * 2}T.sync_threads()")


# Chunk size for the PROVEN chunked-parallel forward scan-core
# (``cppmega_mlx.nn._tilelang.mamba3_chunked_scan_core``). The full
# ``local_gb10_quarter`` mamba3 config carries ``mamba_chunk_size == 64`` and
# the scan-core's Metal tile config requires ``block_M == 64`` (one M-tile per
# chunk). RULE #1: this is the ONE chunk granularity the validated scan-core
# kernel was compiled+parity-checked at; a mismatched shape RAISES below, it
# never silently re-tiles or falls back to the serial scan.
MAMBA3_CHUNKED_FWD_SCAN_CHUNK_SIZE = 64


def _mamba3_chunked_forward_scan_feasibility(
    *,
    sequence_length: int,
    batch: int,
    heads: int,
    head_dim: int,
    state_dim: int,
    groups: int,
    chunk_size: int = MAMBA3_CHUNKED_FWD_SCAN_CHUNK_SIZE,
) -> tuple[bool, int, tuple[int, int, int] | None, str, Any, Any]:
    """Non-raising classifier for hosting the chunked mamba3 forward scan-core.

    Returns ``(feasible, total_threadgroups, grid_or_None, characteristic,
    estimate, limit)``. The single source of truth for both the emit-time
    descriptor record (which must NOT abort the serial kernel for small/non-tile
    test shapes) and the RAISING dispatch gate
    :func:`mamba3_chunked_forward_scan_grid` (which surfaces infeasibility as
    :class:`PathCSplitInfeasible` for callers that actually select the chunked
    path). RULE #1: no try/except degraded path -- this is a pure boolean
    classification; the raise lives in the gate that wraps it.
    """
    from cppmega_mlx.nn._tilelang.mamba3_chunked_scan_core import (
        MAMBA3_CHUNKED_FWD_BLOCK_M,
        MAMBA3_CHUNKED_FWD_BLOCK_N,
        chunk_scan_fwd_grid,
    )

    if sequence_length % chunk_size != 0:
        return (
            False, 0, None,
            "mamba3_chunked_forward sequence_length not divisible by chunk_size",
            sequence_length, chunk_size,
        )
    if chunk_size != MAMBA3_CHUNKED_FWD_BLOCK_M:
        return (
            False, 0, None,
            "mamba3_chunked_forward chunk_size != validated scan-core block_M",
            chunk_size, MAMBA3_CHUNKED_FWD_BLOCK_M,
        )
    if head_dim % MAMBA3_CHUNKED_FWD_BLOCK_N != 0:
        return (
            False, 0, None,
            "mamba3_chunked_forward head_dim not divisible by scan-core block_N",
            head_dim, MAMBA3_CHUNKED_FWD_BLOCK_N,
        )
    if heads % groups != 0:
        return (
            False, 0, None,
            "mamba3_chunked_forward heads not divisible by groups",
            heads, groups,
        )
    try:
        total, grid = chunk_scan_fwd_grid(
            batch, sequence_length, chunk_size, groups, heads, head_dim, state_dim
        )
    except ValueError as exc:
        return (
            False, 0, None,
            f"mamba3_chunked_forward scan-core grid rejected shape: {exc}",
            (batch, sequence_length, chunk_size, heads, head_dim, state_dim),
            "chunk_scan_fwd_grid contract",
        )
    return (True, total, grid, "", None, None)


def mamba3_chunked_forward_scan_grid(
    *,
    region_name: str,
    sequence_length: int,
    batch: int,
    heads: int,
    head_dim: int,
    state_dim: int,
    groups: int,
    chunk_size: int = MAMBA3_CHUNKED_FWD_SCAN_CHUNK_SIZE,
) -> tuple[int, tuple[int, int, int]]:
    """Feasibility gate + grid descriptor for the chunked mamba3 forward scan-core.

    Returns ``(total_threadgroups, (gx, gy, gz))`` for the
    ``cppmega_mlx.nn._tilelang.mamba3_chunked_scan_core`` grid that replaces the
    O(S) serial single-threadgroup forward scan
    (``mamba3_scan_policy: external_state_recurrence``). For the production
    ``local_gb10_quarter`` forward (S=4096, chunk=64, heads=112, head_dim=64,
    state_dim=64, groups=8) this is ``(112, 4, 64) -> 28672`` threadgroups vs the
    serial forward's **1** (validated to compile+run on Metal at fp16 parity
    ~9.8e-4 vs the SSD reference by
    ``scratch/run_chunk_scan_fwd_metal_prod.py`` / the scan-core tests).

    RULE #1 (NO SILENT FALLBACK): a shape the validated scan-core cannot host
    RAISES :class:`PathCSplitInfeasible` here -- the chunked forward path is a
    legitimate, explicitly-gated codegen choice, never a try/except that
    silently degrades to the serial scan. The scan-core's own helpers
    (``chunk_scan_fwd_grid`` / ``chunk_scan_fwd_metal_prim``) RAISE identically
    on non-divisible seqlen / non-divisible heads.
    """
    feasible, total, grid, characteristic, estimate, limit = (
        _mamba3_chunked_forward_scan_feasibility(
            sequence_length=sequence_length,
            batch=batch,
            heads=heads,
            head_dim=head_dim,
            state_dim=state_dim,
            groups=groups,
            chunk_size=chunk_size,
        )
    )
    if not feasible or grid is None:
        raise PathCSplitInfeasible(
            region_name,
            characteristic,
            estimate,
            limit,
            op_name="mamba3_mimo",
        )
    return total, grid


# --------------------------------------------------------------------------- #
# Mamba3 chunked-scan LIVE compile-site DELEGATION INTERPOSE (F0/F1/F2).        #
# Design: docs/MAMBA3-PATHC-MULTIKERNEL-DESIGN.md §3.1/§7 Stage-2 live flip.    #
# --------------------------------------------------------------------------- #

# Op-names whose segment is a single GRID-launched kernel that CANNOT be a
# fragment inlined into the shared single-``T.Kernel`` template PrimFunc. Each
# resolves (via its descriptor ``production_source``) to a ``build_*_metal``
# builder that returns its OWN multi-grid ``@T.prim_func`` directly. The
# fragment_emitter markers registered for these ops are SHADOW no-ops that must
# NEVER be the live emitted kernel (RULE #1) — the interpose below bypasses the
# exec/source path and substitutes the real grid prim when the
# ``CPPMEGA_PATH_C_MAMBA3_CHUNKED_SCAN`` flag is ON.
_MAMBA3_CHUNKED_GRID_DELEGATION_OPS = frozenset(
    {
        # forward F0/F1/F2 (Stage 2)
        "mamba3_chunk_precompute",
        "mamba3_inter_chunk_recur",
        "mamba3_chunk_scan_combine",
        # backward B0/B1/B2 (Stage 3) — the analytic transpose of F0/F1/F2.
        # Same builder ABI (build_*_metal / positional dims); the interpose
        # resolves them identically (RULE #1: one delegation path each).
        "mamba3_chunk_precompute_bwd",
        "mamba3_inter_chunk_recur_bwd",
        "mamba3_chunk_scan_combine_bwd",
    }
)

# The caller-owned handoff buffers that flow F0 -> F1 -> F2 are registered as a
# force-spill device ABI scratch set at module scope
# (DESCRIPTOR_MAMBA3_CHUNKED_FWD_HANDOFF_ABI_BUFFERS); summary_states/prev_states
# resolve fp32 in _buffer_dtype (design §3.1/§3.3/§6.6).


def _resolve_mamba3_chunked_grid_builder(production_source: str):
    """Resolve ``module:builder/inner`` from a descriptor ``production_source``.

    ``production_source`` is e.g.
    ``"cppmega_mlx.nn._tilelang.mamba3_chunked_precompute_core:"``
    ``"build_chunk_precompute_metal/chunk_precompute_fwd_metal_prim"`` — the
    ``build_*_metal`` segment of the ``builder/inner`` tail is the live grid
    kernel delegation. RULE #1: an unresolvable / malformed source RAISES with
    where+what; there is no fallback.
    """
    import importlib

    if ":" not in production_source:
        raise ValueError(
            "mamba3 chunked-scan delegation interpose: malformed "
            f"production_source (no module:tail): {production_source!r}"
        )
    module_path, tail = production_source.split(":", 1)
    builder_name = tail.split("/", 1)[0].strip()
    if not builder_name.startswith("build_"):
        raise ValueError(
            "mamba3 chunked-scan delegation interpose: production_source tail "
            f"does not name a build_*_metal builder: {production_source!r}"
        )
    module = importlib.import_module(module_path.strip())
    builder = getattr(module, builder_name, None)
    if builder is None or not callable(builder):
        raise ValueError(
            "mamba3 chunked-scan delegation interpose: "
            f"{module_path}:{builder_name} is not a callable builder"
        )
    return builder


def _mamba3_chunked_grid_delegation_prim(
    *,
    op_name: str,
    production_source: str,
    shape_env: PathCModelShapeEnv | None,
    batch: int = 1,
):
    """Return the proven F0/F1/F2 grid ``@T.prim_func`` for a single-node segment.

    Reconciles the chain's region ``shape_env`` to the builder's positional
    ``(batch, seqlen, chunk, ngroups, nheads, headdim, dstate)`` ABI. The builder
    compiles+returns the Metal grid kernel whose own multi-grid ``T.Kernel``
    cannot be hosted as a fragment in the template's single ``T.Kernel`` — this is
    the ONE live delegation path (RULE #1). On any compile/shape failure the
    builder RAISES with where+what; there is NO silent serial fallback.

    NOTE: this returns the COMPILED JITKernel-bearing prim from ``build_*_metal``;
    the LIVE region-build flip (#2) + ABI-handoff binder reconciliation must feed
    the handoff buffers into the builder's positional slots. Until that flip is
    wired, this interpose is reached only when the chunked surfaces are emitted
    (flag ON), never for the serial ``mamba3_mimo`` surface (flag OFF default).
    """
    if op_name not in _MAMBA3_CHUNKED_GRID_DELEGATION_OPS:
        raise ValueError(
            "mamba3 chunked-scan delegation interpose invoked for non-chunked "
            f"op_name {op_name!r}"
        )
    if shape_env is None:
        raise ValueError(
            "mamba3 chunked-scan delegation interpose requires a resolved "
            f"shape_env to derive the builder ABI dims for {op_name!r}"
        )
    seqlen = int(shape_env.sequence_length)
    nheads = int(shape_env.mamba_num_heads)
    headdim = int(shape_env.mamba_head_dim)
    dstate = int(shape_env.mamba_state_dim)
    ngroups = int(shape_env.mamba_groups)
    chunk = int(MAMBA3_CHUNKED_FWD_SCAN_CHUNK_SIZE)
    # Fail-fast feasibility (mirrors the live dispatch gate): a shape the
    # validated scan-core cannot host RAISES rather than emitting a no-op marker.
    feasible, _total, _grid, characteristic, estimate, limit = (
        _mamba3_chunked_forward_scan_feasibility(
            sequence_length=seqlen,
            batch=batch,
            heads=nheads,
            head_dim=headdim,
            state_dim=dstate,
            groups=ngroups,
            chunk_size=chunk,
        )
    )
    if not feasible:
        raise ValueError(
            "mamba3 chunked-scan delegation interpose: shape infeasible for "
            f"{op_name!r}: {characteristic} (got {estimate!r}, want {limit!r})"
        )
    builder = _resolve_mamba3_chunked_grid_builder(production_source)
    return builder(batch, seqlen, chunk, ngroups, nheads, headdim, dstate)


def _append_row_phased_mamba3_body(
    body: list[str],
    *,
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    shape_env: PathCModelShapeEnv,
    thread_count: int,
    indent: str,
    launcher_chunked_rows: bool = False,
    cuda_target: bool = False,
) -> None:
    # CHUNKED-FORWARD INTEGRATION SITE (mamba3 forward scan-core).
    #
    # The state-recurrence block below (search "mamba3_scan_policy:
    # external_state_recurrence") is the O(S) SERIAL forward scan that runs in a
    # single ``T.Kernel(1, threads=...)`` threadgroup. The PROVEN chunked,
    # many-threadgroup replacement is
    # ``cppmega_mlx.nn._tilelang.mamba3_chunked_scan_core`` — an SSD 4-step
    # chunked forward that COMPILES + RUNS on Metal at 2048 threadgroups
    # (S=4096,C=256,H=8) with fp16 parity ~4.9e-4 vs the SSD reference and fp32
    # parity ~1e-5 vs OUR serial ``_chunked_mamba3_diagonal_scan`` (validated by
    # ``tests/test_mamba3_chunked_scan_core.py``; ~37.6x forward speedup).
    #
    # The scan-core inputs (cb=C@Bᵀ per chunk, dA_cumsum=cumsum(A*dt),
    # prev_states per-chunk entry states) are produced by THIS body's precompute
    # stages, which are already position-local and grid-parallel: in-proj matvec,
    # causal conv, RoPE angle cumsum, trapezoid, B/C RMSNorm+rope. Completing the
    # swap means: (a) emit those precompute stages over the grid, (b) form cb /
    # dA_cumsum / per-chunk prev_states (inter-chunk recurrence is the only O(S/C)
    # sequential part, with the RoPE angle cumsum as a separate associative scalar
    # prefix), (c) launch the chunked scan-core grid instead of the serial scan
    # below, (d) apply the Y_off + skip + silu·z gate. The descriptor/ABI and the
    # carry/replay boundary-state buffers
    # (``_row_phased_launcher_carry_buffers_for_nodes`` /
    # ``_row_phased_replay_buffers_for_nodes``) already plumb the per-chunk
    # boundary states; conv needs only a ``kernel-1`` halo from the prior chunk.
    #
    # RULE #1: the per-target codegen choice (chunked grid vs serial launcher) is
    # a legitimate gate, NOT a silent fallback. On chunking/parity failure the
    # scan-core helpers RAISE with where+what; do not silently keep the serial
    # path.
    projected = _scratch_name(node, "mamba3_projected_vec")
    conv_history = _scratch_name(node, "mamba3_conv_history")
    conv = _scratch_name(node, "mamba3_conv_vec")
    b_inv = _scratch_name(node, "mamba3_b_inv_rms")
    c_inv = _scratch_name(node, "mamba3_c_inv_rms")
    b_group = _scratch_name(node, "mamba3_b_group")
    c_group = _scratch_name(node, "mamba3_c_group")
    b_raw = _scratch_name(node, "mamba3_b_raw")
    c_raw = _scratch_name(node, "mamba3_c_raw")
    dt_vec = _scratch_name(node, "mamba3_dt_vec")
    a_vec = _scratch_name(node, "mamba3_a_vec")
    trap_group = _scratch_name(node, "mamba3_trap_group")
    next_dt = _scratch_name(node, "mamba3_next_dt")
    next_trap = _scratch_name(node, "mamba3_next_trap")
    # CUDA-only: per-head precomputed next-row dt/trap projections so the
    # trapezoid term is computed thread-parallel over `heads` (mirrors the
    # backward body), instead of serially over `groups` lanes. See the
    # cuda_target branch below for why this avoids the forward runaway.
    next_dt_vec = _scratch_name(node, "mamba3_next_dt_vec")
    next_trap_vec = _scratch_name(node, "mamba3_next_trap_vec")
    angle_cumsum = _scratch_name(node, "mamba3_angle_cumsum")
    out_inner = _scratch_name(node, "mamba3_out_inner")
    accum = _scratch_name(node, "mamba3_accum")
    state_value = _scratch_name(node, "mamba3_state_value")
    proj_dim = _scratch_name(node, "proj_dim")
    hidden_dim_loop = _scratch_name(node, "hidden_dim")
    conv_ch = _scratch_name(node, "conv_ch")
    kernel_pos = _scratch_name(node, "kernel_pos")
    head = _scratch_name(node, "head")
    state = _scratch_name(node, "state_idx")
    rank = _scratch_name(node, "rank")
    angle = _scratch_name(node, "angle")
    feature = _scratch_name(node, "feature")
    out_dim = _scratch_name(node, "out_dim")
    rank_group_flat = _scratch_name(node, "rank_group_flat")
    group_state_flat = _scratch_name(node, "group_state_flat")
    state_flat = _scratch_name(node, "state_flat")
    history_flat = _scratch_name(node, "history_flat")
    angle_carry_flat = _scratch_name(node, "angle_carry_flat")
    angle_carry_head = _scratch_name(node, "angle_carry_head")
    angle_carry_idx = _scratch_name(node, "angle_carry_idx")
    checkpoint_idx = _scratch_name(node, "checkpoint_idx")
    trap_group_loop = _scratch_name(node, "trap_group_loop")
    hidden_size = int(shape_env.hidden_size)
    inner_dim = int(shape_env.mamba_inner_dim)
    in_proj_dim = int(shape_env.mamba_in_proj_dim)
    conv_channels = int(shape_env.mamba_conv_channels)
    kernel = int(shape_env.mamba_conv_kernel)
    history_len = max(0, kernel - 1)
    heads = int(shape_env.mamba_num_heads)
    head_dim = int(shape_env.mamba_head_dim)
    state_dim = int(shape_env.mamba_state_dim)
    groups = int(shape_env.mamba_groups)
    mimo_rank = int(shape_env.mamba_effective_mimo_rank)
    rope_angles = int(shape_env.mamba_num_rope_angles)
    bc_dim = int(shape_env.mamba_bc_dim)
    heads_per_group = heads // groups
    z_offset = 0
    x_offset = inner_dim
    b_offset = 2 * inner_dim
    c_offset = b_offset + bc_dim
    dt_offset = c_offset + bc_dim
    a_offset = dt_offset + heads
    trap_offset = a_offset + heads
    angle_offset = trap_offset + heads
    conv_b_offset = inner_dim
    conv_c_offset = inner_dim + bc_dim
    rot_dim = min(state_dim, 2 * rope_angles)
    checkpoint_interval = MAMBA3_BWD_REPLAY_CHECKPOINT_INTERVAL
    state_extent = heads * head_dim * state_dim
    angle_extent = heads * rope_angles
    delta = _output_with_suffix(node, "_delta") or (node.outputs[0] if node.outputs else "")
    state_output = _mamba3_state_output(node)
    _append_descriptor_node_comments(
        body,
        node=node,
        descriptor=descriptor,
        indent=indent * 3,
    )
    body.append(f"{indent * 3}# mamba3_projection_policy: dense_row_local")
    body.append(
        f"{indent * 3}for {proj_dim} in T.serial(lane, {in_proj_dim}, "
        f"step={thread_count}):"
    )
    body.append(f"{indent * 4}{accum}[0] = 0.0")
    body.append(f"{indent * 4}for {hidden_dim_loop} in T.serial(0, {hidden_size}):")
    hidden_expr = _node_indexed_positional_input_expr(
        node,
        0,
        dtype_by_buffer,
        access_by_buffer,
        f"row * {hidden_size} + {hidden_dim_loop}",
    )
    in_proj_weight_expr = _node_indexed_canonical_input_expr(
        node,
        "mamba3_in_proj_weight",
        dtype_by_buffer,
        access_by_buffer,
        f"{proj_dim} * {hidden_size} + {hidden_dim_loop}",
    )
    body.append(
        f"{indent * 5}{accum}[0] = {accum}[0] + "
        f"({hidden_expr} * {in_proj_weight_expr})"
    )
    body.append(f"{indent * 4}{projected}[{proj_dim}] = {accum}[0]")
    body.append(f"{indent * 3}T.sync_threads()")
    body.append(f"{indent * 3}# mamba3_conv_policy: causal_depthwise_ring_history")
    body.append(
        f"{indent * 3}for {conv_ch} in T.serial(lane, {conv_channels}, "
        f"step={thread_count}):"
    )
    conv_bias_expr = _node_indexed_canonical_input_expr(
        node,
        "mamba3_conv_bias",
        dtype_by_buffer,
        access_by_buffer,
        conv_ch,
    )
    body.append(f"{indent * 4}{conv}[{conv_ch}] = {conv_bias_expr}")
    if history_len > 0:
        body.append(f"{indent * 4}for {kernel_pos} in T.serial(0, {history_len}):")
        conv_weight_expr = _node_indexed_canonical_input_expr(
            node,
            "mamba3_conv_weight",
            dtype_by_buffer,
            access_by_buffer,
            f"{conv_ch} * {kernel} + {kernel_pos}",
            default="1.0",
        )
        body.append(
            f"{indent * 5}{conv}[{conv_ch}] = {conv}[{conv_ch}] + "
            f"({conv_history}[{kernel_pos}, {conv_ch}] * {conv_weight_expr})"
        )
    current_conv_weight_expr = _node_indexed_canonical_input_expr(
        node,
        "mamba3_conv_weight",
        dtype_by_buffer,
        access_by_buffer,
        f"{conv_ch} * {kernel} + {history_len}",
        default="1.0",
    )
    body.append(
        f"{indent * 4}{conv}[{conv_ch}] = {conv}[{conv_ch}] + "
        f"({projected}[{x_offset} + {conv_ch}] * {current_conv_weight_expr})"
    )
    body.append(
        f"{indent * 4}{conv}[{conv_ch}] = {conv}[{conv_ch}] * "
        f"(1.0 / (1.0 + T.exp(-{conv}[{conv_ch}])))"
    )
    body.append(f"{indent * 3}T.sync_threads()")
    body.append(f"{indent * 3}# mamba3_dt_policy: softplus_A_trapezoid")
    body.append(
        f"{indent * 3}for {head} in T.serial(lane, {heads}, step={thread_count}):"
    )
    dt_bias = _node_indexed_canonical_input_expr(
        node,
        "mamba3_dt_bias",
        dtype_by_buffer,
        access_by_buffer,
        head,
    )
    body.append(
        f"{indent * 4}{dt_vec}[{head}] = T.log(1.0 + "
        f"T.exp({projected}[{dt_offset} + {head}] + {dt_bias}))"
    )
    body.append(
        f"{indent * 4}{a_vec}[{head}] = T.min(-T.log(1.0 + "
        f"T.exp({projected}[{a_offset} + {head}])), -0.01)"
    )
    body.append(f"{indent * 4}for {angle} in T.serial(0, {rope_angles}):")
    body.append(
        f"{indent * 5}{angle_cumsum}[{head}, {angle}] = "
        f"{angle_cumsum}[{head}, {angle}] + "
        f"({projected}[{angle_offset} + {angle}] * {dt_vec}[{head}])"
    )
    if cuda_target:
        # CUDA-only: compute the next-row dt/trap raw projections for EVERY head
        # in this same thread-parallel `head` loop (one head per active lane,
        # `heads` lanes busy). The original (Metal) path recomputes these inside
        # a `groups`-serial loop with an inner `heads_per_group` loop, so only
        # `groups` lanes do the expensive `hidden`-deep reduction while the rest
        # idle behind a block sync -> the forward compute-spin runaway on CUDA.
        # The arithmetic is identical (same FMA order, same softplus); only the
        # work distribution changes. Mirrors `_append_row_phased_mamba3_bwd_body`.
        next_dt_hidden_expr = _node_indexed_positional_input_expr(
            node,
            0,
            dtype_by_buffer,
            access_by_buffer,
            f"(row + 1) * {hidden_size} + {hidden_dim_loop}",
        )
        next_dt_vec_weight_expr = _node_indexed_canonical_input_expr(
            node,
            "mamba3_in_proj_weight",
            dtype_by_buffer,
            access_by_buffer,
            f"({dt_offset} + {head}) * {hidden_size} + {hidden_dim_loop}",
        )
        next_trap_vec_weight_expr = _node_indexed_canonical_input_expr(
            node,
            "mamba3_in_proj_weight",
            dtype_by_buffer,
            access_by_buffer,
            f"({trap_offset} + {head}) * {hidden_size} + {hidden_dim_loop}",
        )
        next_dt_vec_bias = _node_indexed_canonical_input_expr(
            node,
            "mamba3_dt_bias",
            dtype_by_buffer,
            access_by_buffer,
            head,
        )
        body.append(f"{indent * 4}{next_dt_vec}[{head}] = 0.0")
        body.append(f"{indent * 4}{next_trap_vec}[{head}] = 0.0")
        body.append(
            f"{indent * 4}if row + 1 < {int(shape_env.sequence_length)}:"
        )
        body.append(
            f"{indent * 5}for {hidden_dim_loop} in T.serial(0, {hidden_size}):"
        )
        body.append(
            f"{indent * 6}{next_dt_vec}[{head}] = {next_dt_vec}[{head}] + "
            f"({next_dt_hidden_expr} * {next_dt_vec_weight_expr})"
        )
        body.append(
            f"{indent * 6}{next_trap_vec}[{head}] = {next_trap_vec}[{head}] + "
            f"({next_dt_hidden_expr} * {next_trap_vec_weight_expr})"
        )
        body.append(
            f"{indent * 5}{next_dt_vec}[{head}] = T.log(1.0 + "
            f"T.exp({next_dt_vec}[{head}] + {next_dt_vec_bias}))"
        )
    if launcher_chunked_rows:
        body.append(
            f"{indent * 3}for {angle_carry_flat} in T.serial(lane, {heads * rope_angles}, "
            f"step={thread_count}):"
        )
        body.append(f"{indent * 4}{angle_carry_head} = {angle_carry_flat} // {rope_angles}")
        body.append(f"{indent * 4}{angle_carry_idx} = {angle_carry_flat} % {rope_angles}")
        angle_state_ref = _buffer_ref(
            "mamba3_angle_state",
            access_by_buffer,
            f"{angle_carry_head} * {rope_angles} + {angle_carry_idx}",
        )
        body.append(
            f"{indent * 4}{angle_state_ref} = "
            f"{angle_cumsum}[{angle_carry_head}, {angle_carry_idx}]"
        )
    body.append(f"{indent * 3}if ((row + 1) % {checkpoint_interval}) == 0:")
    body.append(f"{indent * 4}{checkpoint_idx} = (row + 1) // {checkpoint_interval}")
    body.append(
        f"{indent * 4}for {angle_carry_flat} in T.serial(lane, {angle_extent}, "
        f"step={thread_count}):"
    )
    body.append(f"{indent * 5}{angle_carry_head} = {angle_carry_flat} // {rope_angles}")
    body.append(f"{indent * 5}{angle_carry_idx} = {angle_carry_flat} % {rope_angles}")
    angle_checkpoint_ref = _buffer_ref(
        "mamba3_angle_checkpoint",
        access_by_buffer,
        f"{checkpoint_idx} * {angle_extent} + {angle_carry_head} * {rope_angles} + {angle_carry_idx}",
    )
    body.append(
        f"{indent * 5}{angle_checkpoint_ref} = "
        f"{angle_cumsum}[{angle_carry_head}, {angle_carry_idx}]"
    )
    body.append(f"{indent * 3}T.sync_threads()")
    if cuda_target:
        # CUDA-only: the next-row dt/trap reductions were already done
        # thread-parallel over `heads` above. This per-group reduction now just
        # sums the precomputed `next_dt_vec`/`next_trap_vec` (no `hidden`-deep
        # inner loop), so the whole trapezoid phase is cheap and no longer
        # serializes the block on `groups` lanes. Same arithmetic as the Metal
        # path below.
        body.append(
            f"{indent * 3}for {trap_group_loop} in T.serial(lane, {groups}, "
            f"step={thread_count}):"
        )
        body.append(f"{indent * 4}{trap_group}[{trap_group_loop}] = 0.0")
        body.append(f"{indent * 4}for {head} in T.serial(0, {heads_per_group}):")
        body.append(
            f"{indent * 5}{accum}[0] = "
            f"{trap_group_loop} * {heads_per_group} + {head}"
        )
        body.append(
            f"{indent * 5}{trap_group}[{trap_group_loop}] = "
            f"{trap_group}[{trap_group_loop}] + "
            f"(({next_dt_vec}[T.cast({accum}[0], \"int32\")] * "
            f"(1.0 - (1.0 / (1.0 + T.exp(-{next_trap_vec}["
            f"T.cast({accum}[0], \"int32\")]))))) + "
            f"({dt_vec}[T.cast({accum}[0], \"int32\")] * "
            f"(1.0 / (1.0 + T.exp(-{projected}[{trap_offset} + "
            f"T.cast({accum}[0], \"int32\")])))))"
        )
        body.append(
            f"{indent * 4}{trap_group}[{trap_group_loop}] = "
            f"{trap_group}[{trap_group_loop}] / "
            f"{float(heads_per_group):.1f}"
        )
    else:
        body.append(
            f"{indent * 3}for {trap_group_loop} in T.serial(lane, {groups}, "
            f"step={thread_count}):"
        )
        body.append(f"{indent * 4}{trap_group}[{trap_group_loop}] = 0.0")
        body.append(f"{indent * 4}for {head} in T.serial(0, {heads_per_group}):")
        body.append(
            f"{indent * 5}{accum}[0] = "
            f"{trap_group_loop} * {heads_per_group} + {head}"
        )
        body.append(f"{indent * 5}{next_dt}[0] = 0.0")
        body.append(f"{indent * 5}{next_trap}[0] = 0.0")
        body.append(f"{indent * 5}if row + 1 < {int(shape_env.sequence_length)}:")
        body.append(f"{indent * 6}for {hidden_dim_loop} in T.serial(0, {hidden_size}):")
        next_hidden_expr = _node_indexed_positional_input_expr(
            node,
            0,
            dtype_by_buffer,
            access_by_buffer,
            f"(row + 1) * {hidden_size} + {hidden_dim_loop}",
        )
        next_dt_weight_expr = _node_indexed_canonical_input_expr(
            node,
            "mamba3_in_proj_weight",
            dtype_by_buffer,
            access_by_buffer,
            f"({dt_offset} + T.cast({accum}[0], \"int32\")) * {hidden_size} + "
            f"{hidden_dim_loop}",
        )
        next_trap_weight_expr = _node_indexed_canonical_input_expr(
            node,
            "mamba3_in_proj_weight",
            dtype_by_buffer,
            access_by_buffer,
            f"({trap_offset} + T.cast({accum}[0], \"int32\")) * {hidden_size} + "
            f"{hidden_dim_loop}",
        )
        body.append(
            f"{indent * 7}{next_dt}[0] = {next_dt}[0] + "
            f"({next_hidden_expr} * {next_dt_weight_expr})"
        )
        body.append(
            f"{indent * 7}{next_trap}[0] = {next_trap}[0] + "
            f"({next_hidden_expr} * {next_trap_weight_expr})"
        )
        next_dt_bias = _node_indexed_canonical_input_expr(
            node,
            "mamba3_dt_bias",
            dtype_by_buffer,
            access_by_buffer,
            f'T.cast({accum}[0], "int32")',
        )
        body.append(
            f"{indent * 6}{next_dt}[0] = T.log(1.0 + "
            f"T.exp({next_dt}[0] + {next_dt_bias}))"
        )
        body.append(
            f"{indent * 5}{trap_group}[{trap_group_loop}] = "
            f"{trap_group}[{trap_group_loop}] + "
            f"(({next_dt}[0] * (1.0 - (1.0 / (1.0 + T.exp(-{next_trap}[0]))))) + "
            f"({dt_vec}[T.cast({accum}[0], \"int32\")] * "
            f"(1.0 / (1.0 + T.exp(-{projected}[{trap_offset} + "
            f"T.cast({accum}[0], \"int32\")])))))"
        )
        body.append(
            f"{indent * 4}{trap_group}[{trap_group_loop}] = "
            f"{trap_group}[{trap_group_loop}] / "
            f"{float(heads_per_group):.1f}"
        )
    body.append(f"{indent * 3}T.sync_threads()")
    body.append(f"{indent * 3}# mamba3_bc_policy: rank_group_rmsnorm_rope")
    body.append(
        f"{indent * 3}for {rank_group_flat} in T.serial(lane, {mimo_rank * groups}, "
        f"step={thread_count}):"
    )
    rank_expr = f"({rank_group_flat} // {groups})"
    group_expr = f"({rank_group_flat} % {groups})"
    body.append(f"{indent * 4}{b_inv}[{rank_expr}, {group_expr}] = 0.0")
    body.append(f"{indent * 4}{c_inv}[{rank_expr}, {group_expr}] = 0.0")
    body.append(f"{indent * 4}for {state} in T.serial(0, {state_dim}):")
    bc_index = f"(({rank_expr} * {groups} + {group_expr}) * {state_dim} + {state})"
    body.append(
        f"{indent * 5}{b_inv}[{rank_expr}, {group_expr}] = "
        f"{b_inv}[{rank_expr}, {group_expr}] + "
        f"({conv}[{inner_dim} + {bc_index}] * {conv}[{inner_dim} + {bc_index}])"
    )
    body.append(
        f"{indent * 5}{c_inv}[{rank_expr}, {group_expr}] = "
        f"{c_inv}[{rank_expr}, {group_expr}] + "
        f"({conv}[{inner_dim + bc_dim} + {bc_index}] * "
        f"{conv}[{inner_dim + bc_dim} + {bc_index}])"
    )
    body.append(
        f"{indent * 4}{b_inv}[{rank_expr}, {group_expr}] = "
        f"T.rsqrt(({b_inv}[{rank_expr}, {group_expr}] / "
        f"{float(state_dim):.1f}) + 0.00001)"
    )
    body.append(
        f"{indent * 4}{c_inv}[{rank_expr}, {group_expr}] = "
        f"T.rsqrt(({c_inv}[{rank_expr}, {group_expr}] / "
        f"{float(state_dim):.1f}) + 0.00001)"
    )
    body.append(f"{indent * 3}T.sync_threads()")
    body.append(
        f"{indent * 3}for {group_state_flat} in T.serial(lane, {groups * state_dim}, "
        f"step={thread_count}):"
    )
    group_expr = f"({group_state_flat} // {state_dim})"
    state_expr = f"({group_state_flat} % {state_dim})"
    body.append(f"{indent * 4}{b_raw}[{group_expr}, {state_expr}] = 0.0")
    body.append(f"{indent * 4}{c_raw}[{group_expr}, {state_expr}] = 0.0")
    body.append(f"{indent * 4}for {rank} in T.serial(0, {mimo_rank}):")
    bc_index = f"(({rank} * {groups} + {group_expr}) * {state_dim} + {state_expr})"
    b_norm_weight = _node_indexed_canonical_input_expr(
        node,
        "mamba3_B_norm_weight",
        dtype_by_buffer,
        access_by_buffer,
        bc_index,
        default="1.0",
    )
    b_bias = _node_indexed_canonical_input_expr(
        node,
        "mamba3_B_bias",
        dtype_by_buffer,
        access_by_buffer,
        bc_index,
    )
    c_norm_weight = _node_indexed_canonical_input_expr(
        node,
        "mamba3_C_norm_weight",
        dtype_by_buffer,
        access_by_buffer,
        bc_index,
        default="1.0",
    )
    c_bias = _node_indexed_canonical_input_expr(
        node,
        "mamba3_C_bias",
        dtype_by_buffer,
        access_by_buffer,
        bc_index,
    )
    body.append(
        f"{indent * 5}{b_raw}[{group_expr}, {state_expr}] = "
        f"{b_raw}[{group_expr}, {state_expr}] + "
        f"(({conv}[{inner_dim} + {bc_index}] * {b_inv}[{rank}, {group_expr}] * "
        f"{b_norm_weight}) + {b_bias})"
    )
    body.append(
        f"{indent * 5}{c_raw}[{group_expr}, {state_expr}] = "
        f"{c_raw}[{group_expr}, {state_expr}] + "
        f"(({conv}[{inner_dim + bc_dim} + {bc_index}] * "
        f"{c_inv}[{rank}, {group_expr}] * {c_norm_weight}) + {c_bias})"
    )
    body.append(
        f"{indent * 4}{b_raw}[{group_expr}, {state_expr}] = "
        f"({b_raw}[{group_expr}, {state_expr}] / {float(mimo_rank):.1f}) * "
        f"{trap_group}[{group_expr}]"
    )
    body.append(
        f"{indent * 4}{c_raw}[{group_expr}, {state_expr}] = "
        f"{c_raw}[{group_expr}, {state_expr}] / {float(mimo_rank):.1f}"
    )
    body.append(f"{indent * 3}T.sync_threads()")
    body.append(
        f"{indent * 3}for {group_state_flat} in T.serial(lane, {groups * state_dim}, "
        f"step={thread_count}):"
    )
    group_expr = f"({group_state_flat} // {state_dim})"
    state_expr = f"({group_state_flat} % {state_dim})"
    angle_expr = f"({state_expr} // 2)"
    body.append(f"{indent * 4}if {state_expr} < {rot_dim}:")
    body.append(f"{indent * 5}if ({state_expr} % 2) == 0:")
    body.append(
        f"{indent * 6}{b_group}[{group_expr}, {state_expr}] = "
        f"({b_raw}[{group_expr}, {state_expr}] * "
        f"T.cos({angle_cumsum}[{group_expr}, {angle_expr}])) - "
        f"({b_raw}[{group_expr}, {state_expr} + 1] * "
        f"T.sin({angle_cumsum}[{group_expr}, {angle_expr}]))"
    )
    body.append(
        f"{indent * 6}{c_group}[{group_expr}, {state_expr}] = "
        f"({c_raw}[{group_expr}, {state_expr}] * "
        f"T.cos({angle_cumsum}[{group_expr}, {angle_expr}])) - "
        f"({c_raw}[{group_expr}, {state_expr} + 1] * "
        f"T.sin({angle_cumsum}[{group_expr}, {angle_expr}]))"
    )
    body.append(f"{indent * 5}else:")
    body.append(
        f"{indent * 6}{b_group}[{group_expr}, {state_expr}] = "
        f"({b_raw}[{group_expr}, {state_expr} - 1] * "
        f"T.sin({angle_cumsum}[{group_expr}, {angle_expr}])) + "
        f"({b_raw}[{group_expr}, {state_expr}] * "
        f"T.cos({angle_cumsum}[{group_expr}, {angle_expr}]))"
    )
    body.append(
        f"{indent * 6}{c_group}[{group_expr}, {state_expr}] = "
        f"({c_raw}[{group_expr}, {state_expr} - 1] * "
        f"T.sin({angle_cumsum}[{group_expr}, {angle_expr}])) + "
        f"({c_raw}[{group_expr}, {state_expr}] * "
        f"T.cos({angle_cumsum}[{group_expr}, {angle_expr}]))"
    )
    body.append(f"{indent * 4}else:")
    body.append(
        f"{indent * 5}{b_group}[{group_expr}, {state_expr}] = "
        f"{b_raw}[{group_expr}, {state_expr}]"
    )
    body.append(
        f"{indent * 5}{c_group}[{group_expr}, {state_expr}] = "
        f"{c_raw}[{group_expr}, {state_expr}]"
    )
    body.append(f"{indent * 3}T.sync_threads()")
    if state_output is not None:
        # CHUNKED-FORWARD SCAN-CORE GRID DESCRIPTOR (validated, gated).
        #
        # Emit-time feasibility gate for the PROVEN chunked-parallel forward
        # scan-core (``cppmega_mlx.nn._tilelang.mamba3_chunked_scan_core``) that
        # replaces this O(S) serial single-threadgroup scan. The gate RAISES
        # ``PathCSplitInfeasible`` (RULE #1: no silent serial fallback) for any
        # shape the Metal-validated scan-core kernel cannot host; on a feasible
        # shape it records the exact grid + threadgroup count that the chunked
        # forward dispatches. For the production local_gb10_quarter forward this
        # is ``(heads, ceildiv(head_dim,16), batch*nchunks)`` -> 28672
        # threadgroups vs this serial scan's 1 (compiles+runs on Metal at fp16
        # parity ~9.8e-4 vs the SSD reference; see
        # ``scratch/run_chunk_scan_fwd_metal_prod.py``).
        # Classify (non-raising) whether THIS shape can host the chunked grid.
        # The serial scan below is still the emitted compute (the full
        # kernel-split swap is the documented remaining work); recording the
        # descriptor here makes the chunked dispatch parameters live + asserted
        # for production-tile shapes, and explicitly labels non-tile-aligned
        # test shapes as NOT FEASIBLE. This is a pure classification (no
        # try/except degraded path): the serial scan is the ONLY emitted compute
        # for ALL shapes today, so nothing is silenced -- the RAISING gate
        # ``mamba3_chunked_forward_scan_grid`` is for callers that actually
        # SELECT the chunked dispatch.
        (
            _chunked_feasible,
            _chunked_total,
            _chunked_grid,
            _chunked_reason,
            _chunked_est,
            _chunked_limit,
        ) = _mamba3_chunked_forward_scan_feasibility(
            sequence_length=int(shape_env.sequence_length),
            batch=1,
            heads=heads,
            head_dim=head_dim,
            state_dim=state_dim,
            groups=groups,
        )
        if _chunked_feasible:
            body.append(
                f"{indent * 3}# mamba3_chunked_forward_scan: grid={_chunked_grid} "
                f"threadgroups={_chunked_total} chunk_size="
                f"{MAMBA3_CHUNKED_FWD_SCAN_CHUNK_SIZE} (validated scan-core; "
                f"serial-scan threadgroups=1)"
            )
        else:
            body.append(
                f"{indent * 3}# mamba3_chunked_forward_scan: NOT FEASIBLE for this "
                f"shape -- {_chunked_reason} "
                f"({_chunked_est} vs {_chunked_limit}); serial scan retained"
            )
        body.append(f"{indent * 3}# mamba3_scan_policy: external_state_recurrence")
        body.append(
            f"{indent * 3}for {feature} in T.serial(lane, {inner_dim}, "
            f"step={thread_count}):"
        )
        head_expr = f"({feature} // {head_dim})"
        dim_expr = f"({feature} % {head_dim})"
        group_expr = f"({head_expr} // {heads_per_group})"
        body.append(f"{indent * 4}{out_inner}[{feature}] = 0.0")
        body.append(f"{indent * 4}for {state} in T.serial(0, {state_dim}):")
        state_idx = f"{head_expr} * {head_dim * state_dim} + {dim_expr} * {state_dim} + {state}"
        state_ref = _buffer_ref(state_output, access_by_buffer, state_idx)
        body.append(
            f"{indent * 5}{state_value}[0] = "
            f"(T.exp({a_vec}[{head_expr}] * {dt_vec}[{head_expr}]) * "
            f"{state_ref}) + ({conv}[{feature}] * {b_group}[{group_expr}, {state}])"
        )
        body.append(f"{indent * 5}{state_ref} = {state_value}[0]")
        body.append(
            f"{indent * 5}{out_inner}[{feature}] = {out_inner}[{feature}] + "
            f"({state_value}[0] * {c_group}[{group_expr}, {state}])"
        )
        d_skip = _node_indexed_canonical_input_expr(
            node,
            "mamba3_D",
            dtype_by_buffer,
            access_by_buffer,
            head_expr,
            default="1.0",
        )
        z_val = f"{projected}[{z_offset} + {feature}]"
        x_val = f"{conv}[{feature}]"
        body.append(
            f"{indent * 4}{out_inner}[{feature}] = "
            f"({out_inner}[{feature}] + ({d_skip} * {x_val})) * "
            f"{z_val} * (1.0 / (1.0 + T.exp(-{z_val})))"
        )
        body.append(f"{indent * 3}T.sync_threads()")
        body.append(f"{indent * 3}if ((row + 1) % {checkpoint_interval}) == 0:")
        body.append(f"{indent * 4}{checkpoint_idx} = (row + 1) // {checkpoint_interval}")
        body.append(
            f"{indent * 4}for {state_flat} in T.serial(lane, {state_extent}, "
            f"step={thread_count}):"
        )
        body.append(f"{indent * 5}{head} = {state_flat} // {head_dim * state_dim}")
        body.append(f"{indent * 5}{hidden_dim_loop} = ({state_flat} // {state_dim}) % {head_dim}")
        body.append(f"{indent * 5}{state} = {state_flat} % {state_dim}")
        checkpoint_state_idx = (
            f"{head} * {head_dim * state_dim} + {hidden_dim_loop} * {state_dim} + {state}"
        )
        h_checkpoint_ref = _buffer_ref(
            "mamba3_h_checkpoint",
            access_by_buffer,
            f"{checkpoint_idx} * {state_extent} + {checkpoint_state_idx}",
        )
        body.append(
            f"{indent * 5}{h_checkpoint_ref} = "
            f"{_buffer_ref(state_output, access_by_buffer, checkpoint_state_idx)}"
        )
        body.append(f"{indent * 3}T.sync_threads()")
    if delta:
        body.append(f"{indent * 3}# mamba3_output_policy: dense_out_projection")
        body.append(
            f"{indent * 3}for {out_dim} in T.serial(lane, {hidden_size}, "
            f"step={thread_count}):"
        )
        body.append(f"{indent * 4}{accum}[0] = 0.0")
        body.append(f"{indent * 4}for {feature} in T.serial(0, {inner_dim}):")
        out_proj_weight = _node_indexed_canonical_input_expr(
            node,
            "mamba3_out_proj_weight",
            dtype_by_buffer,
            access_by_buffer,
            f"{out_dim} * {inner_dim} + {feature}",
            default="1.0",
        )
        body.append(
            f"{indent * 5}{accum}[0] = {accum}[0] + "
            f"({out_inner}[{feature}] * {out_proj_weight})"
        )
        body.append(
            f"{indent * 4}{_buffer_ref(delta, access_by_buffer, f'row * {hidden_size} + {out_dim}')} = "
            f"{accum}[0]"
        )
        body.append(f"{indent * 3}T.sync_threads()")
    if history_len <= 0:
        return
    if history_len > 1:
        body.append(
            f"{indent * 3}for {history_flat} in T.serial(lane, "
            f"{(history_len - 1) * conv_channels}, step={thread_count}):"
        )
        hist_expr = f"({history_flat} // {conv_channels})"
        conv_ch_expr = f"({history_flat} % {conv_channels})"
        body.append(
            f"{indent * 4}{conv_history}[{hist_expr}, {conv_ch_expr}] = "
            f"{conv_history}[{hist_expr} + 1, {conv_ch_expr}]"
        )
        body.append(f"{indent * 3}T.sync_threads()")
    body.append(
        f"{indent * 3}for {conv_ch} in T.serial(lane, {conv_channels}, "
        f"step={thread_count}):"
    )
    body.append(
        f"{indent * 4}{conv_history}[{history_len - 1}, {conv_ch}] = "
        f"{projected}[{x_offset} + {conv_ch}]"
    )
    if launcher_chunked_rows:
        body.append(f"{indent * 3}T.sync_threads()")
        body.append(
            f"{indent * 3}for {history_flat} in T.serial(lane, "
            f"{history_len * conv_channels}, step={thread_count}):"
        )
        hist_expr = f"({history_flat} // {conv_channels})"
        conv_ch_expr = f"({history_flat} % {conv_channels})"
        conv_state_ref = _buffer_ref(
            "mamba3_conv_state",
            access_by_buffer,
            f"{hist_expr} * {conv_channels} + {conv_ch_expr}",
        )
        body.append(
            f"{indent * 4}{conv_state_ref} = {conv_history}[{hist_expr}, {conv_ch_expr}]"
        )
    body.append(f"{indent * 3}T.sync_threads()")


def _append_row_phased_m2rnn_init(
    body: list[str],
    *,
    node: _ScheduleNodeView,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    shape_env: PathCModelShapeEnv,
    thread_count: int,
    indent: str,
    launcher_chunked_rows: bool = False,
) -> None:
    conv_history = _scratch_name(node, "m2rnn_conv_history")
    h_state = _scratch_name(node, "m2rnn_h_state")
    h_next = _scratch_name(node, "m2rnn_h_next")
    projected = _scratch_name(node, "m2rnn_projected_vec")
    conv = _scratch_name(node, "m2rnn_conv_vec")
    post = _scratch_name(node, "m2rnn_post_vec")
    accum = _scratch_name(node, "m2rnn_accum")
    decay = _scratch_name(node, "m2rnn_decay")
    sum_sq = _scratch_name(node, "m2rnn_sum_sq")
    sum_sq_partial = _scratch_name(node, "m2rnn_sum_sq_partial")
    inv_rms = _scratch_name(node, "m2rnn_inv_rms")
    conv_dim = int(shape_env.m2rnn_conv_dim)
    in_proj_dim = int(shape_env.m2rnn_in_proj_dim)
    total_heads = int(shape_env.m2rnn_num_heads)
    k_dim = int(shape_env.m2rnn_k_head_dim)
    v_dim = int(shape_env.m2rnn_v_head_dim)
    features = total_heads * v_dim
    history_len = max(0, int(shape_env.m2rnn_conv_kernel) - 1)
    body.append(
        f"{indent * 2}{projected} = T.alloc_shared(({in_proj_dim},), \"float32\")"
    )
    body.append(f"{indent * 2}{conv} = T.alloc_shared(({conv_dim},), \"float32\")")
    body.append(f"{indent * 2}{post} = T.alloc_shared(({features},), \"float32\")")
    body.append(
        f"{indent * 2}{h_state} = "
        f"T.alloc_shared(({total_heads}, {k_dim}, {v_dim}), \"float32\")"
    )
    body.append(
        f"{indent * 2}{h_next} = "
        f"T.alloc_shared(({total_heads}, {k_dim}, {v_dim}), \"float32\")"
    )
    if history_len > 0:
        body.append(
            f"{indent * 2}{conv_history} = "
            f"T.alloc_shared(({history_len}, {conv_dim}), \"float32\")"
        )
    body.append(f"{indent * 2}{accum} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 2}{decay} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 2}{sum_sq} = T.alloc_shared((1,), \"float32\")")
    body.append(
        f"{indent * 2}{sum_sq_partial} = T.alloc_shared(({thread_count},), \"float32\")"
    )
    body.append(f"{indent * 2}{inv_rms} = T.alloc_shared((1,), \"float32\")")
    head = _scratch_name(node, "head_init")
    kk = _scratch_name(node, "kk_init")
    vv = _scratch_name(node, "vv_init")
    state_idx = _scratch_name(node, "state_idx_init")
    checkpoint_state_idx = _scratch_name(node, "checkpoint_state_idx_init")
    hist = _scratch_name(node, "hist_init")
    ch = _scratch_name(node, "conv_ch_init")
    history_idx = _scratch_name(node, "history_idx_init")
    body.append(f"{indent * 2}# {node.name}: m2rnn_state_policy: row_carried")
    if launcher_chunked_rows:
        body.append(f"{indent * 2}if path_c_first_row_launch != 0:")
        state_loop_indent = indent * 3
        state_body_indent = indent * 4
    else:
        state_loop_indent = indent * 2
        state_body_indent = indent * 3
    body.append(
        f"{state_loop_indent}for {state_idx} in T.serial(lane, {total_heads * k_dim * v_dim}, "
        f"step={thread_count}):"
    )
    body.append(f"{state_body_indent}{head} = {state_idx} // {k_dim * v_dim}")
    body.append(f"{state_body_indent}{kk} = ({state_idx} // {v_dim}) % {k_dim}")
    body.append(f"{state_body_indent}{vv} = {state_idx} % {v_dim}")
    h0_expr = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_h0",
        dtype_by_buffer,
        access_by_buffer,
        f"{head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv}",
    )
    body.append(f"{state_body_indent}{h_state}[{head}, {kk}, {vv}] = {h0_expr}")
    body.append(f"{state_body_indent}{h_next}[{head}, {kk}, {vv}] = {h0_expr}")
    if launcher_chunked_rows:
        body.append(f"{indent * 2}else:")
        body.append(
            f"{indent * 3}for {state_idx} in T.serial(lane, {total_heads * k_dim * v_dim}, "
            f"step={thread_count}):"
        )
        body.append(f"{indent * 4}{head} = {state_idx} // {k_dim * v_dim}")
        body.append(f"{indent * 4}{kk} = ({state_idx} // {v_dim}) % {k_dim}")
        body.append(f"{indent * 4}{vv} = {state_idx} % {v_dim}")
        h_state_ref = _buffer_ref(
            "m2rnn_h_state",
            access_by_buffer,
            f"{head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv}",
        )
        body.append(f"{indent * 4}{h_state}[{head}, {kk}, {vv}] = {h_state_ref}")
        body.append(f"{indent * 4}{h_next}[{head}, {kk}, {vv}] = {h_state_ref}")
    if launcher_chunked_rows:
        body.append(f"{indent * 2}if path_c_first_row_launch != 0:")
        checkpoint_loop_indent = indent * 3
        checkpoint_body_indent = indent * 4
    else:
        checkpoint_loop_indent = indent * 2
        checkpoint_body_indent = indent * 3
    body.append(
        f"{checkpoint_loop_indent}for {checkpoint_state_idx} in T.serial(lane, "
        f"{total_heads * k_dim * v_dim}, step={thread_count}):"
    )
    body.append(
        f"{checkpoint_body_indent}{head} = {checkpoint_state_idx} // {k_dim * v_dim}"
    )
    body.append(
        f"{checkpoint_body_indent}{kk} = ({checkpoint_state_idx} // {v_dim}) % {k_dim}"
    )
    body.append(f"{checkpoint_body_indent}{vv} = {checkpoint_state_idx} % {v_dim}")
    h_checkpoint_ref = _buffer_ref(
        "m2rnn_h_checkpoint",
        access_by_buffer,
        f"{head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv}",
    )
    body.append(
        f"{checkpoint_body_indent}{h_checkpoint_ref} = {h_state}[{head}, {kk}, {vv}]"
    )
    body.append(f"{indent * 2}T.sync_threads()")
    if history_len <= 0:
        return
    body.append(f"{indent * 2}# {node.name}: m2rnn_conv_policy: ring_history")
    body.append(
        f"{indent * 2}for {history_idx} in T.serial(lane, {history_len * conv_dim}, "
        f"step={thread_count}):"
    )
    body.append(f"{indent * 3}{hist} = {history_idx} // {conv_dim}")
    body.append(f"{indent * 3}{ch} = {history_idx} % {conv_dim}")
    conv_state_expr = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_conv_state",
        dtype_by_buffer,
        access_by_buffer,
        f"{hist} * {conv_dim} + {ch}",
    )
    body.append(f"{indent * 3}{conv_history}[{hist}, {ch}] = {conv_state_expr}")
    body.append(f"{indent * 2}T.sync_threads()")


def _append_row_phased_m2rnn_body(
    body: list[str],
    *,
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    shape_env: PathCModelShapeEnv,
    thread_count: int,
    indent: str,
    launcher_chunked_rows: bool = False,
) -> None:
    conv_history = _scratch_name(node, "m2rnn_conv_history")
    h_state = _scratch_name(node, "m2rnn_h_state")
    h_next = _scratch_name(node, "m2rnn_h_next")
    projected = _scratch_name(node, "m2rnn_projected_vec")
    conv = _scratch_name(node, "m2rnn_conv_vec")
    post = _scratch_name(node, "m2rnn_post_vec")
    accum = _scratch_name(node, "m2rnn_accum")
    decay = _scratch_name(node, "m2rnn_decay")
    sum_sq = _scratch_name(node, "m2rnn_sum_sq")
    sum_sq_partial = _scratch_name(node, "m2rnn_sum_sq_partial")
    inv_rms = _scratch_name(node, "m2rnn_inv_rms")
    proj_dim = _scratch_name(node, "proj_dim")
    hidden_dim_loop = _scratch_name(node, "hidden_dim")
    conv_ch = _scratch_name(node, "conv_ch")
    kernel_pos = _scratch_name(node, "kernel_pos")
    head = _scratch_name(node, "head")
    kk = _scratch_name(node, "kk")
    vv = _scratch_name(node, "vv")
    vv_inner = _scratch_name(node, "vv_inner")
    feature = _scratch_name(node, "feature")
    out_dim = _scratch_name(node, "out_dim")
    hist = _scratch_name(node, "hist")
    state_idx = _scratch_name(node, "state_idx")
    checkpoint_idx = _scratch_name(node, "checkpoint_idx")
    partial_lane = _scratch_name(node, "partial_lane")
    hidden_size = int(shape_env.hidden_size)
    conv_dim = int(shape_env.m2rnn_conv_dim)
    in_proj_dim = int(shape_env.m2rnn_in_proj_dim)
    total_heads = int(shape_env.m2rnn_num_heads)
    q_heads = int(shape_env.m2rnn_num_q_heads)
    k_heads = int(shape_env.m2rnn_num_k_heads)
    v_heads = int(shape_env.m2rnn_num_v_heads)
    f_heads = int(shape_env.m2rnn_num_f_heads)
    g_heads = int(shape_env.m2rnn_num_g_heads)
    w_heads = int(shape_env.m2rnn_num_weight_heads)
    k_dim = int(shape_env.m2rnn_k_head_dim)
    v_dim = int(shape_env.m2rnn_v_head_dim)
    kernel = int(shape_env.m2rnn_conv_kernel)
    history_len = max(0, kernel - 1)
    q_offset = 0
    k_offset = int(shape_env.m2rnn_q_dim)
    v_offset = k_offset + int(shape_env.m2rnn_k_dim)
    f_offset = conv_dim
    g_offset = conv_dim + f_heads
    features = total_heads * v_dim
    q_group = total_heads // q_heads
    k_group = total_heads // k_heads
    v_group = total_heads // v_heads
    f_group = total_heads // f_heads
    g_repeat = total_heads // g_heads
    w_group = total_heads // w_heads
    _append_descriptor_node_comments(
        body,
        node=node,
        descriptor=descriptor,
        indent=indent * 3,
    )
    body.append(f"{indent * 3}# m2rnn_projection_policy: lane_strided_dense_row_local")
    body.append(
        f"{indent * 3}for {proj_dim} in T.serial(lane, {in_proj_dim}, "
        f"step={thread_count}):"
    )
    body.append(f"{indent * 4}{accum}[0] = 0.0")
    body.append(f"{indent * 4}for {hidden_dim_loop} in T.serial(0, {hidden_size}):")
    hidden_expr = _node_indexed_positional_input_expr(
        node,
        0,
        dtype_by_buffer,
        access_by_buffer,
        f"row * {hidden_size} + {hidden_dim_loop}",
    )
    in_proj_weight_expr = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_in_proj_weight",
        dtype_by_buffer,
        access_by_buffer,
        f"{proj_dim} * {hidden_size} + {hidden_dim_loop}",
    )
    body.append(
        f"{indent * 5}{accum}[0] = {accum}[0] + "
        f"({hidden_expr} * {in_proj_weight_expr})"
    )
    body.append(f"{indent * 4}{projected}[{proj_dim}] = {accum}[0]")
    body.append(f"{indent * 3}T.sync_threads()")
    body.append(f"{indent * 3}# m2rnn_conv_policy: lane_strided_causal_depthwise_ring_history")
    body.append(
        f"{indent * 3}for {conv_ch} in T.serial(lane, {conv_dim}, step={thread_count}):"
    )
    conv_bias_expr = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_conv_bias",
        dtype_by_buffer,
        access_by_buffer,
        conv_ch,
    )
    body.append(f"{indent * 4}{conv}[{conv_ch}] = {conv_bias_expr}")
    if history_len > 0:
        body.append(f"{indent * 4}for {kernel_pos} in T.serial(0, {history_len}):")
        conv_weight_expr = _node_indexed_canonical_input_expr(
            node,
            "m2rnn_conv_weight",
            dtype_by_buffer,
            access_by_buffer,
            f"{conv_ch} * {kernel} + {kernel_pos}",
            default="1.0",
        )
        body.append(
            f"{indent * 5}{conv}[{conv_ch}] = {conv}[{conv_ch}] + "
            f"({conv_history}[{kernel_pos}, {conv_ch}] * {conv_weight_expr})"
        )
    current_conv_weight_expr = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_conv_weight",
        dtype_by_buffer,
        access_by_buffer,
        f"{conv_ch} * {kernel} + {history_len}",
        default="1.0",
    )
    body.append(
        f"{indent * 4}{conv}[{conv_ch}] = {conv}[{conv_ch}] + "
        f"({projected}[{conv_ch}] * {current_conv_weight_expr})"
    )
    body.append(
        f"{indent * 4}{conv}[{conv_ch}] = {conv}[{conv_ch}] * "
        f"(1.0 / (1.0 + T.exp(-{conv}[{conv_ch}])))"
    )
    body.append(f"{indent * 3}T.sync_threads()")
    body.append(f"{indent * 3}# m2rnn_recurrence_policy: lane_strided_mapped_state_update")
    body.append(
        f"{indent * 3}for {head} in T.serial(lane, {total_heads}, step={thread_count}):"
    )
    f_src = f"({head} // {f_group})"
    f_input = f"{projected}[{f_offset} + {f_src}]"
    a_log_expr = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_A_log",
        dtype_by_buffer,
        access_by_buffer,
        head,
    )
    dt_bias_expr = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_dt_bias",
        dtype_by_buffer,
        access_by_buffer,
        head,
    )
    body.append(
        f"{indent * 4}{decay}[0] = T.exp(-T.exp({a_log_expr}) * "
        f"T.log(1.0 + T.exp({f_input} + {dt_bias_expr})))"
    )
    body.append(f"{indent * 4}for {kk} in T.serial(0, {k_dim}):")
    body.append(f"{indent * 5}for {vv} in T.serial(0, {v_dim}):")
    body.append(f"{indent * 6}{accum}[0] = 0.0")
    body.append(f"{indent * 6}for {vv_inner} in T.serial(0, {v_dim}):")
    w_src = f"({head} // {w_group})"
    state_weight_expr = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_state_weight",
        dtype_by_buffer,
        access_by_buffer,
        f"{w_src} * {v_dim * v_dim} + {vv_inner} * {v_dim} + {vv}",
        default="1.0",
    )
    body.append(
        f"{indent * 7}{accum}[0] = {accum}[0] + "
        f"({h_state}[{head}, {kk}, {vv_inner}] * {state_weight_expr})"
    )
    k_src = f"({head} // {k_group})"
    v_src = f"({head} // {v_group})"
    k_val = f"{conv}[{k_offset} + ({k_src} * {k_dim}) + {kk}]"
    v_val = f"{conv}[{v_offset} + ({v_src} * {v_dim}) + {vv}]"
    body.append(
        f"{indent * 6}{h_next}[{head}, {kk}, {vv}] = "
        f"({decay}[0] * {h_state}[{head}, {kk}, {vv}]) + "
        f"((1.0 - {decay}[0]) * T.tanh({accum}[0] + ({k_val} * {v_val})))"
    )
    body.append(f"{indent * 3}T.sync_threads()")
    body.append(f"{indent * 3}# m2rnn_post_policy: lane_strided_residual_gate_norm_out_proj")
    body.append(
        f"{indent * 3}for {feature} in T.serial(lane, {features}, step={thread_count}):"
    )
    body.append(f"{indent * 4}{head} = {feature} // {v_dim}")
    body.append(f"{indent * 4}{vv} = {feature} % {v_dim}")
    body.append(f"{indent * 4}{post}[{feature}] = 0.0")
    body.append(f"{indent * 4}for {kk} in T.serial(0, {k_dim}):")
    q_src = f"({head} // {q_group})"
    q_val = f"{conv}[{q_offset} + ({q_src} * {k_dim}) + {kk}]"
    body.append(
        f"{indent * 5}{post}[{feature}] = {post}[{feature}] + "
        f"({q_val} * {h_next}[{head}, {kk}, {vv}])"
    )
    d_skip_expr = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_D",
        dtype_by_buffer,
        access_by_buffer,
        f"{head} * {v_dim} + {vv}",
        default="0.0",
    )
    g_flat = f"({feature} // {g_repeat})"
    g_val = f"{projected}[{g_offset} + {g_flat}]"
    body.append(
        f"{indent * 4}{post}[{feature}] = "
        f"({post}[{feature}] + ({v_val} * {d_skip_expr})) * "
        f"{g_val} * (1.0 / (1.0 + T.exp(-{g_val})))"
    )
    body.append(f"{indent * 3}T.sync_threads()")
    body.append(
        f"{indent * 3}for {state_idx} in T.serial(lane, {total_heads * k_dim * v_dim}, "
        f"step={thread_count}):"
    )
    body.append(f"{indent * 4}{head} = {state_idx} // {k_dim * v_dim}")
    body.append(f"{indent * 4}{kk} = ({state_idx} // {v_dim}) % {k_dim}")
    body.append(f"{indent * 4}{vv} = {state_idx} % {v_dim}")
    body.append(
        f"{indent * 4}{h_state}[{head}, {kk}, {vv}] = "
        f"{h_next}[{head}, {kk}, {vv}]"
    )
    if launcher_chunked_rows:
        h_state_ref = _buffer_ref(
            "m2rnn_h_state",
            access_by_buffer,
            f"{head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv}",
        )
        body.append(
            f"{indent * 4}{h_state_ref} = {h_state}[{head}, {kk}, {vv}]"
        )
    checkpoint_interval = M2RNN_BWD_REPLAY_CHECKPOINT_INTERVAL
    checkpoint_ref = _buffer_ref(
        "m2rnn_h_checkpoint",
        access_by_buffer,
        f"{checkpoint_idx} * {total_heads * k_dim * v_dim} + "
        f"{head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv}",
    )
    body.append(f"{indent * 4}if ((row + 1) % {checkpoint_interval}) == 0:")
    body.append(f"{indent * 5}{checkpoint_idx} = (row + 1) // {checkpoint_interval}")
    body.append(
        f"{indent * 5}{checkpoint_ref} = {h_state}[{head}, {kk}, {vv}]"
    )
    body.append(f"{indent * 3}{sum_sq_partial}[lane] = 0.0")
    body.append(
        f"{indent * 3}for {feature} in T.serial(lane, {features}, step={thread_count}):"
    )
    body.append(
        f"{indent * 4}{sum_sq_partial}[lane] = {sum_sq_partial}[lane] + "
        f"({post}[{feature}] * {post}[{feature}])"
    )
    body.append(f"{indent * 3}T.sync_threads()")
    body.append(f"{indent * 3}if lane == 0:")
    body.append(f"{indent * 4}{sum_sq}[0] = 0.0")
    body.append(
        f"{indent * 4}for {partial_lane} in T.serial(0, {thread_count}):"
    )
    body.append(
        f"{indent * 5}{sum_sq}[0] = {sum_sq}[0] + {sum_sq_partial}[{partial_lane}]"
    )
    body.append(
        f"{indent * 4}{inv_rms}[0] = T.rsqrt(({sum_sq}[0] / "
        f"{float(features):.1f}) + 0.00001)"
    )
    body.append(f"{indent * 3}T.sync_threads()")
    for output_name in node.outputs:
        body.append(
            f"{indent * 3}for {out_dim} in T.serial(lane, {hidden_size}, "
            f"step={thread_count}):"
        )
        body.append(f"{indent * 4}{accum}[0] = 0.0")
        body.append(f"{indent * 4}for {feature} in T.serial(0, {features}):")
        gate_norm_expr = _node_indexed_canonical_input_expr(
            node,
            "m2rnn_g_norm_weight",
            dtype_by_buffer,
            access_by_buffer,
            feature,
            default="1.0",
        )
        out_proj_expr = _node_indexed_canonical_input_expr(
            node,
            "m2rnn_out_proj_weight",
            dtype_by_buffer,
            access_by_buffer,
            f"{out_dim} * {features} + {feature}",
            default="1.0",
        )
        body.append(
            f"{indent * 5}{accum}[0] = {accum}[0] + "
            f"({post}[{feature}] * {inv_rms}[0] * "
            f"{gate_norm_expr} * {out_proj_expr})"
        )
        body.append(
            f"{indent * 4}{_buffer_ref(output_name, access_by_buffer, f'row * {hidden_size} + {out_dim}')} = "
            f"{accum}[0]"
        )
    body.append(f"{indent * 3}T.sync_threads()")
    if history_len <= 0:
        return
    if history_len > 1:
        body.append(
            f"{indent * 3}for {state_idx} in T.serial(lane, {(history_len - 1) * conv_dim}, "
            f"step={thread_count}):"
        )
        body.append(f"{indent * 4}{hist} = {state_idx} // {conv_dim}")
        body.append(f"{indent * 4}{conv_ch} = {state_idx} % {conv_dim}")
        body.append(
            f"{indent * 4}{conv_history}[{hist}, {conv_ch}] = "
            f"{conv_history}[{hist} + 1, {conv_ch}]"
        )
    body.append(
        f"{indent * 3}for {conv_ch} in T.serial(lane, {conv_dim}, step={thread_count}):"
    )
    body.append(
        f"{indent * 4}{conv_history}[{history_len - 1}, {conv_ch}] = "
        f"{projected}[{conv_ch}]"
    )
    if launcher_chunked_rows:
        body.append(f"{indent * 3}T.sync_threads()")
        body.append(
            f"{indent * 3}for {state_idx} in T.serial(lane, {history_len * conv_dim}, "
            f"step={thread_count}):"
        )
        body.append(f"{indent * 4}{hist} = {state_idx} // {conv_dim}")
        body.append(f"{indent * 4}{conv_ch} = {state_idx} % {conv_dim}")
        conv_state_ref = _buffer_ref(
            "m2rnn_conv_state",
            access_by_buffer,
            f"{hist} * {conv_dim} + {conv_ch}",
        )
        body.append(
            f"{indent * 4}{conv_state_ref} = {conv_history}[{hist}, {conv_ch}]"
        )
    body.append(f"{indent * 3}T.sync_threads()")


def _append_row_phased_residual_rmsnorm_body(
    body: list[str],
    *,
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    hidden_size: int,
    thread_count: int,
    indent: str,
) -> None:
    sum_sq = _scratch_name(node, "row_sum_sq")
    partial = _scratch_name(node, "row_sum_sq_partial")
    inv_rms = _scratch_name(node, "row_inv_rms")
    _append_descriptor_node_comments(
        body,
        node=node,
        descriptor=descriptor,
        indent=indent * 3,
    )
    lhs = _node_input_expr(node, 0, dtype_by_buffer, access_by_buffer, "i")
    rhs = _node_input_expr(node, 1, dtype_by_buffer, access_by_buffer, "i")
    weight = _node_input_expr(node, 2, dtype_by_buffer, access_by_buffer, "i")
    residual_expr = f"({lhs} + {rhs})"
    body.append(f"{indent * 3}{partial}[lane] = 0.0")
    body.append(
        f"{indent * 3}for i in T.serial(row * {hidden_size} + lane, "
        f"(row + 1) * {hidden_size}, step={thread_count}):"
    )
    body.append(
        f"{indent * 4}{partial}[lane] = {partial}[lane] + "
        f"({residual_expr} * {residual_expr})"
    )
    body.append(f"{indent * 3}T.sync_threads()")
    body.append(f"{indent * 3}if lane == 0:")
    body.append(f"{indent * 4}{sum_sq}[0] = 0.0")
    body.append(f"{indent * 4}for partial_lane in T.serial(0, {thread_count}):")
    body.append(
        f"{indent * 5}{sum_sq}[0] = {sum_sq}[0] + {partial}[partial_lane]"
    )
    body.append(
        f"{indent * 4}{inv_rms}[0] = T.rsqrt(({sum_sq}[0] / "
        f"{float(hidden_size):.1f}) + 0.00001)"
    )
    body.append(f"{indent * 3}T.sync_threads()")
    body.append(
        f"{indent * 3}for i in T.serial(row * {hidden_size} + lane, "
        f"(row + 1) * {hidden_size}, step={thread_count}):"
    )
    if node.outputs:
        body.append(
            f"{indent * 4}{_buffer_ref(node.outputs[0], access_by_buffer, 'i')} = "
            f"{residual_expr}"
        )
    normalized = f"{residual_expr} * {inv_rms}[0] * {weight}"
    for output_name in node.outputs[1:]:
        body.append(
            f"{indent * 4}{_buffer_ref(output_name, access_by_buffer, 'i')} = "
            f"{normalized}"
        )
    body.append(f"{indent * 3}T.sync_threads()")


def _append_row_phased_entry_rmsnorm_body(
    body: list[str],
    *,
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    hidden_size: int,
    thread_count: int,
    indent: str,
) -> None:
    """Emit the row-phased forward body for the entry RMSNorm op.

    Block A: this mirrors :func:`_append_row_phased_residual_rmsnorm_body`
    but skips the residual ``+ delta`` step because the entry RMSNorm
    runs immediately on ``hidden`` before the first in-region brick. The
    forward writes a single normalized output and leaves no residual
    side-channel because downstream bricks consume the normalized
    ``route_hidden`` directly while the inter-brick bridge still
    reduces against the raw entry ``hidden``.
    """

    sum_sq = _scratch_name(node, "row_sum_sq")
    partial = _scratch_name(node, "row_sum_sq_partial")
    inv_rms = _scratch_name(node, "row_inv_rms")
    _append_descriptor_node_comments(
        body,
        node=node,
        descriptor=descriptor,
        indent=indent * 3,
    )
    hidden = _node_input_expr(node, 0, dtype_by_buffer, access_by_buffer, "i")
    weight = _node_input_expr(node, 1, dtype_by_buffer, access_by_buffer, "i")
    body.append(f"{indent * 3}{partial}[lane] = 0.0")
    body.append(
        f"{indent * 3}for i in T.serial(row * {hidden_size} + lane, "
        f"(row + 1) * {hidden_size}, step={thread_count}):"
    )
    body.append(
        f"{indent * 4}{partial}[lane] = {partial}[lane] + "
        f"({hidden} * {hidden})"
    )
    body.append(f"{indent * 3}T.sync_threads()")
    body.append(f"{indent * 3}if lane == 0:")
    body.append(f"{indent * 4}{sum_sq}[0] = 0.0")
    body.append(f"{indent * 4}for partial_lane in T.serial(0, {thread_count}):")
    body.append(
        f"{indent * 5}{sum_sq}[0] = {sum_sq}[0] + {partial}[partial_lane]"
    )
    body.append(
        f"{indent * 4}{inv_rms}[0] = T.rsqrt(({sum_sq}[0] / "
        f"{float(hidden_size):.1f}) + 0.00001)"
    )
    body.append(f"{indent * 3}T.sync_threads()")
    body.append(
        f"{indent * 3}for i in T.serial(row * {hidden_size} + lane, "
        f"(row + 1) * {hidden_size}, step={thread_count}):"
    )
    normalized = f"{hidden} * {inv_rms}[0] * {weight}"
    for output_name in node.outputs:
        body.append(
            f"{indent * 4}{_buffer_ref(output_name, access_by_buffer, 'i')} = "
            f"{normalized}"
        )
    body.append(f"{indent * 3}T.sync_threads()")


def _append_row_phased_attention_qkv_projection_body(
    body: list[str],
    *,
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    shape_env: PathCModelShapeEnv,
    thread_count: int,
    indent: str,
) -> None:
    q_projected = _scratch_name(node, "attention_q_projected")
    kv_projected = _scratch_name(node, "attention_kv_projected")
    q_projected_pair = _scratch_name(node, "attention_q_projected_pair")
    kv_projected_pair = _scratch_name(node, "attention_kv_projected_pair")
    q_projected_vec = _scratch_name(node, "attention_q_projected_vec")
    kv_projected_vec = _scratch_name(node, "attention_kv_projected_vec")
    rope_phase = _scratch_name(node, "attention_rope_phase")
    q_prepared = _scratch_name(node, "attention_q_prepared")
    kv_prepared = _scratch_name(node, "attention_kv_prepared")
    q_head = _scratch_name(node, "q_head")
    kv_head = _scratch_name(node, "kv_head")
    indices_flat = _scratch_name(node, "indices_flat")
    k_top = _scratch_name(node, "k_top")
    d = _scratch_name(node, "d")
    h = _scratch_name(node, "h")
    src_i = _scratch_name(node, "src_i")
    hidden_size = int(shape_env.hidden_size)
    q_heads = int(shape_env.attention_num_q_heads)
    kv_heads = int(shape_env.attention_num_kv_heads)
    head_dim = int(shape_env.attention_head_dim)
    topk = int(shape_env.attention_sparse_topk)
    q_scale_output = _node_output_for_canonical(node, "q_scale")
    kv_scale_output = _node_output_for_canonical(node, "kv_scale")
    q_fp8_output = _node_output_for_canonical(node, "q_fp8")
    kv_fp8_output = _node_output_for_canonical(node, "kv_fp8")
    indices_output = _node_output_for_canonical(node, "indices")
    if (
        q_scale_output is None
        or kv_scale_output is None
        or q_fp8_output is None
        or kv_fp8_output is None
    ):
        raise ValueError("row-phased attention projection requires FP8 outputs")
    _append_descriptor_node_comments(
        body,
        node=node,
        descriptor=descriptor,
        indent=indent * 3,
    )
    hidden = _node_input_expr(node, 0, dtype_by_buffer, access_by_buffer, src_i)
    body.append(
        f"{indent * 3}{q_projected_vec} = T.alloc_local(({head_dim},), \"float32\")"
    )
    body.append(
        f"{indent * 3}{kv_projected_vec} = T.alloc_local(({head_dim},), \"float32\")"
    )
    body.append(f"{indent * 3}# fp8_prepare_policy: lane_strided_row_head_reduction")
    body.append(
        f"{indent * 3}for {q_head} in T.serial(lane, {q_heads}, step={thread_count}):"
    )
    q_scale_ref = _row_phased_attention_scale_ref(
        q_scale_output,
        access_by_buffer,
        shape_env,
        row_expr="row",
        head_expr=q_head,
        q_side=True,
    )
    body.append(f"{indent * 4}{q_scale_ref} = 0.0")
    _append_attention_projection_head_dot_products(
        body,
        projected_vec=q_projected_vec,
        hidden=hidden,
        weight_buffer="attention_q_proj_weight",
        bias_buffer="attention_q_proj_bias",
        dtype_by_buffer=dtype_by_buffer,
        access_by_buffer=access_by_buffer,
        hidden_size=hidden_size,
        head_dim=head_dim,
        head_expr=q_head,
        hidden_loop_var=h,
        src_index_var=src_i,
        dim_loop_var=d,
        indent=indent * 4,
    )
    body.append(f"{indent * 4}for {d} in T.serial(0, {head_dim}):")
    _append_attention_projection_prepare_from_vector(
        body,
        projected_vec=q_projected_vec,
        projected=q_projected,
        paired_projected=q_projected_pair,
        rope_phase=rope_phase,
        prepared=q_prepared,
        dtype_by_buffer=dtype_by_buffer,
        access_by_buffer=access_by_buffer,
        head_dim=head_dim,
        dim_expr=d,
        indent=indent * 5,
    )
    body.append(
        f"{indent * 5}{q_scale_ref} = T.max({q_scale_ref}, "
        f"T.abs(T.cast({q_prepared}[0], \"float32\")))"
    )
    body.append(
        f"{indent * 4}{q_scale_ref} = T.max({q_scale_ref} * "
        f"T.cast({1.0 / 448.0:.17g}, \"float32\"), "
        f"T.cast(1.0e-12, \"float32\"))"
    )
    body.append(f"{indent * 4}for {d} in T.serial(0, {head_dim}):")
    _append_attention_projection_prepare_from_vector(
        body,
        projected_vec=q_projected_vec,
        projected=q_projected,
        paired_projected=q_projected_pair,
        rope_phase=rope_phase,
        prepared=q_prepared,
        dtype_by_buffer=dtype_by_buffer,
        access_by_buffer=access_by_buffer,
        head_dim=head_dim,
        dim_expr=d,
        indent=indent * 5,
    )
    body.append(
        f"{indent * 5}{_row_phased_attention_value_ref(q_fp8_output, access_by_buffer, shape_env, row_expr='row', head_expr=q_head, dim_expr=d, q_side=True)} = "
        f"{_fp8_encode_expr(f'{q_prepared}[0]', q_scale_ref)}"
    )

    body.append(
        f"{indent * 3}for {kv_head} in T.serial(lane, {kv_heads}, step={thread_count}):"
    )
    kv_scale_ref = _row_phased_attention_scale_ref(
        kv_scale_output,
        access_by_buffer,
        shape_env,
        row_expr="row",
        head_expr=kv_head,
        q_side=False,
    )
    body.append(f"{indent * 4}{kv_scale_ref} = 0.0")
    _append_attention_projection_head_dot_products(
        body,
        projected_vec=kv_projected_vec,
        hidden=hidden,
        weight_buffer="attention_sparse_kv_proj_weight",
        bias_buffer="attention_sparse_kv_proj_bias",
        dtype_by_buffer=dtype_by_buffer,
        access_by_buffer=access_by_buffer,
        hidden_size=hidden_size,
        head_dim=head_dim,
        head_expr=kv_head,
        hidden_loop_var=h,
        src_index_var=src_i,
        dim_loop_var=d,
        indent=indent * 4,
    )
    body.append(f"{indent * 4}for {d} in T.serial(0, {head_dim}):")
    _append_attention_projection_prepare_from_vector(
        body,
        projected_vec=kv_projected_vec,
        projected=kv_projected,
        paired_projected=kv_projected_pair,
        rope_phase=rope_phase,
        prepared=kv_prepared,
        dtype_by_buffer=dtype_by_buffer,
        access_by_buffer=access_by_buffer,
        head_dim=head_dim,
        dim_expr=d,
        indent=indent * 5,
    )
    body.append(
        f"{indent * 5}{kv_scale_ref} = T.max({kv_scale_ref}, "
        f"T.abs(T.cast({kv_prepared}[0], \"float32\")))"
    )
    body.append(
        f"{indent * 4}{kv_scale_ref} = T.max({kv_scale_ref} * "
        f"T.cast({1.0 / 448.0:.17g}, \"float32\"), "
        f"T.cast(1.0e-12, \"float32\"))"
    )
    body.append(f"{indent * 4}for {d} in T.serial(0, {head_dim}):")
    _append_attention_projection_prepare_from_vector(
        body,
        projected_vec=kv_projected_vec,
        projected=kv_projected,
        paired_projected=kv_projected_pair,
        rope_phase=rope_phase,
        prepared=kv_prepared,
        dtype_by_buffer=dtype_by_buffer,
        access_by_buffer=access_by_buffer,
        head_dim=head_dim,
        dim_expr=d,
        indent=indent * 5,
    )
    body.append(
        f"{indent * 5}{_row_phased_attention_value_ref(kv_fp8_output, access_by_buffer, shape_env, row_expr='row', head_expr=kv_head, dim_expr=d, q_side=False)} = "
        f"{_fp8_encode_expr(f'{kv_prepared}[0]', kv_scale_ref)}"
    )
    if indices_output is not None:
        body.append(
            f"{indent * 3}for {indices_flat} in T.serial(lane, {kv_heads * topk}, "
            f"step={thread_count}):"
        )
        body.append(f"{indent * 4}{kv_head} = {indices_flat} // {topk}")
        body.append(f"{indent * 4}{k_top} = {indices_flat} % {topk}")
        indices_ref = _row_phased_attention_indices_ref(
            indices_output,
            access_by_buffer,
            shape_env,
            head_expr=kv_head,
            topk_expr=k_top,
        )
        body.append(f"{indent * 4}if row >= {k_top}:")
        body.append(
            f"{indent * 5}{indices_ref} = row - {k_top}"
        )
        body.append(f"{indent * 4}else:")
        body.append(f"{indent * 5}{indices_ref} = -1")
    body.append(f"{indent * 3}T.sync_threads()")


def _append_row_phased_sparse_mla_fp8_apply_body(
    body: list[str],
    *,
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    shape_env: PathCModelShapeEnv,
    thread_count: int,
    indent: str,
) -> None:
    sink_enabled = _scratch_name(node, "sink_enabled")
    sparse_index = _scratch_name(node, "sparse_index")
    score_accum = _scratch_name(node, "score_accum")
    score_max = _scratch_name(node, "score_max")
    score_weight = _scratch_name(node, "score_weight")
    score_weights = _scratch_name(node, "score_weights")
    sparse_indices = _scratch_name(node, "sparse_indices")
    sumexp = _scratch_name(node, "sumexp")
    value_accum = _scratch_name(node, "value_accum")
    context_accum = _scratch_name(node, "context_accum")
    context_values = _scratch_name(node, "context_values")
    q_head_index = _scratch_name(node, "q_head")
    kv_head_index = _scratch_name(node, "kv_head")
    source_head_index = _scratch_name(node, "source_head")
    source_dim_index = _scratch_name(node, "source_dim")
    out_dim_loop = _scratch_name(node, "out_dim_loop")
    source_head_loop = _scratch_name(node, "source_head_loop")
    source_dim_loop = _scratch_name(node, "source_dim_loop")
    dot_dim_loop = _scratch_name(node, "dot_dim")
    lse_head_loop = _scratch_name(node, "lse_head")
    k_top = _scratch_name(node, "k_top")
    hidden_size = int(shape_env.hidden_size)
    q_heads = int(shape_env.attention_num_q_heads)
    kv_heads = int(shape_env.attention_num_kv_heads)
    head_dim = int(shape_env.attention_head_dim)
    sequence_length = int(shape_env.sequence_length)
    q_dim = q_heads * head_dim
    topk = int(shape_env.attention_sparse_topk)
    q_per_kv = max(1, q_heads // max(1, kv_heads))
    q_head = f"{q_head_index}[0]"
    kv_head = f"{kv_head_index}[0]"
    source_head = f"{source_head_index}[0]"
    source_dim = f"{source_dim_index}[0]"
    dot_dim = dot_dim_loop
    lse_head = lse_head_loop
    q_fp8_input = _node_input_for_canonical(node, "q_fp8")
    q_scale_input = _node_input_for_canonical(node, "q_scale")
    kv_fp8_input = _node_input_for_canonical(node, "kv_fp8")
    kv_scale_input = _node_input_for_canonical(node, "kv_scale")
    indices_input = _node_input_for_canonical(node, "indices")
    if (
        q_fp8_input is None
        or q_scale_input is None
        or kv_fp8_input is None
        or kv_scale_input is None
        or indices_input is None
    ):
        raise ValueError("row-phased sparse MLA apply requires FP8 and index inputs")
    sm_scale = _optional_buffer_expr(
        "sparse_mla_sm_scale",
        dtype_by_buffer,
        access_by_buffer,
        "1.0",
        "0",
    )
    sinks = _optional_buffer_expr(
        "sparse_mla_sinks",
        dtype_by_buffer,
        access_by_buffer,
        "0.0",
        q_head,
    )
    has_sinks = _optional_buffer_expr(
        "sparse_mla_has_sinks",
        dtype_by_buffer,
        access_by_buffer,
        "0",
        "0",
    )
    attention_out = (
        _node_output_for_canonical(node, "attention_out")
        or _node_output_with_suffix(node, "_out")
    )
    lse_output = _node_output_for_canonical(node, "lse")

    def append_sparse_index_for_current_head(
        *,
        loop_indent: str,
        kv_head_expr: str,
        use_cached_sparse_index: bool = False,
    ) -> None:
        if use_cached_sparse_index:
            body.append(f"{loop_indent}{sparse_index}[0] = {sparse_indices}[{k_top}]")
            return
        indices_current = _row_phased_attention_indices_ref(
            indices_input,
            access_by_buffer,
            shape_env,
            head_expr=kv_head_expr,
            topk_expr=k_top,
        )
        body.append(f"{loop_indent}{sparse_index}[0] = {indices_current}")

    def append_score_for_current_head(
        *,
        loop_indent: str,
        head_expr: str,
        kv_head_expr: str,
        use_cached_sparse_index: bool = False,
    ) -> None:
        q_scale_current = _row_phased_attention_scale_ref(
            q_scale_input,
            access_by_buffer,
            shape_env,
            row_expr="row",
            head_expr=head_expr,
            q_side=True,
        )
        kv_scale_current = _row_phased_attention_selected_kv_scale_ref(
            kv_scale_input,
            access_by_buffer,
            shape_env,
            selected_row_expr=f"{sparse_index}[0]",
            head_expr=kv_head_expr,
        )
        append_sparse_index_for_current_head(
            loop_indent=loop_indent,
            kv_head_expr=kv_head_expr,
            use_cached_sparse_index=use_cached_sparse_index,
        )
        body.append(
            f"{loop_indent}if {sparse_index}[0] >= 0 and "
            f"{sparse_index}[0] < {sequence_length}:"
        )
        body.append(f"{loop_indent}    {score_accum}[0] = 0.0")
        body.append(f"{loop_indent}    for {dot_dim_loop} in T.serial(0, {head_dim}):")
        q_dot_ref = _row_phased_attention_value_ref(
            q_fp8_input,
            access_by_buffer,
            shape_env,
            row_expr="row",
            head_expr=head_expr,
            dim_expr=dot_dim,
            q_side=True,
        )
        kv_dot_ref = _row_phased_attention_selected_kv_value_ref(
            kv_fp8_input,
            access_by_buffer,
            shape_env,
            selected_row_expr=f"{sparse_index}[0]",
            head_expr=kv_head_expr,
            dim_expr=dot_dim,
        )
        body.append(
            f"{loop_indent}        {score_accum}[0] = {score_accum}[0] + "
            f"(fp8_e4m3fn_to_float({q_dot_ref}) * "
            f"fp8_e4m3fn_to_float({kv_dot_ref}))"
        )
        body.append(
            f"{loop_indent}    {score_accum}[0] = {score_accum}[0] * "
            f"{q_scale_current} * {kv_scale_current} * {sm_scale}"
        )
        body.append(f"{loop_indent}else:")
        body.append(
            f"{loop_indent}    {score_accum}[0] = "
            "T.float32(-3.4028234663852886e38)"
        )

    if attention_out is not None:
        _append_descriptor_node_comments(
            body,
            node=node,
            descriptor=descriptor,
            indent=indent * 3,
        )
        body.append(
            f"{indent * 3}# sparse_mla_fp8_apply_policy: "
            "lane_strided_context_and_out_projection"
        )
        body.append(
            f"{indent * 3}{score_weights} = "
            f"T.alloc_local(({topk},), \"float32\")"
        )
        body.append(
            f"{indent * 3}{sparse_indices} = "
            f"T.alloc_local(({topk},), \"int32\")"
        )
        body.append(
            f"{indent * 3}{sink_enabled}[0] = "
            f"T.cast({has_sinks} != 0, \"float32\")"
        )
        body.append(
            f"{indent * 3}for {source_head_loop} in T.serial(lane, {q_heads}, "
            f"step={thread_count}):"
        )
        body.append(f"{indent * 4}{source_head_index}[0] = {source_head_loop}")
        body.append(f"{indent * 4}{q_head_index}[0] = {source_head}")
        body.append(
            f"{indent * 4}{kv_head_index}[0] = {source_head} // {q_per_kv}"
        )
        body.append(f"{indent * 4}for {k_top} in T.serial(0, {topk}):")
        indices_current = _row_phased_attention_indices_ref(
            indices_input,
            access_by_buffer,
            shape_env,
            head_expr=kv_head,
            topk_expr=k_top,
        )
        body.append(f"{indent * 5}{sparse_indices}[{k_top}] = {indices_current}")
        body.append(
            f"{indent * 4}{score_max}[0] = "
            "T.float32(-3.4028234663852886e38)"
        )
        body.append(f"{indent * 4}for {k_top} in T.serial(0, {topk}):")
        body.append(f"{indent * 5}{score_weights}[{k_top}] = 0.0")
        append_score_for_current_head(
            loop_indent=indent * 5,
            head_expr=source_head,
            kv_head_expr=kv_head,
            use_cached_sparse_index=True,
        )
        body.append(f"{indent * 5}if {score_accum}[0] > {score_max}[0]:")
        body.append(f"{indent * 6}{score_max}[0] = {score_accum}[0]")
        body.append(f"{indent * 4}if {sink_enabled}[0] != 0.0:")
        body.append(f"{indent * 5}if {sinks} > {score_max}[0]:")
        body.append(f"{indent * 6}{score_max}[0] = {sinks}")
        body.append(f"{indent * 4}{sumexp}[0] = 0.0")
        body.append(f"{indent * 4}for {k_top} in T.serial(0, {topk}):")
        append_score_for_current_head(
            loop_indent=indent * 5,
            head_expr=source_head,
            kv_head_expr=kv_head,
            use_cached_sparse_index=True,
        )
        body.append(
            f"{indent * 5}{score_weight}[0] = "
            f"T.exp({score_accum}[0] - {score_max}[0])"
        )
        body.append(
            f"{indent * 5}{score_weights}[{k_top}] = {score_weight}[0]"
        )
        body.append(f"{indent * 5}{sumexp}[0] = {sumexp}[0] + {score_weight}[0]")
        body.append(f"{indent * 4}if {sink_enabled}[0] != 0.0:")
        body.append(
            f"{indent * 5}{sumexp}[0] = {sumexp}[0] + "
            f"T.exp({sinks} - {score_max}[0])"
        )
        if lse_output is not None:
            lse_ref = _row_phased_lse_ref(
                lse_output,
                access_by_buffer,
                shape_env,
                row_expr="row",
                head_expr=source_head,
            )
            body.append(f"{indent * 4}{lse_ref} = 0.0")
            body.append(f"{indent * 4}if {sumexp}[0] > 0.0:")
            body.append(
                f"{indent * 5}{lse_ref} = {score_max}[0] + "
                f"T.log({sumexp}[0])"
            )
        body.append(
            f"{indent * 4}for {source_dim_loop} in T.serial(0, {head_dim}):"
        )
        body.append(f"{indent * 5}{source_dim_index}[0] = {source_dim_loop}")
        body.append(f"{indent * 5}{value_accum}[0] = 0.0")
        body.append(f"{indent * 5}for {k_top} in T.serial(0, {topk}):")
        append_sparse_index_for_current_head(
            loop_indent=indent * 6,
            kv_head_expr=kv_head,
            use_cached_sparse_index=True,
        )
        body.append(
            f"{indent * 6}if {sparse_index}[0] >= 0 and "
            f"{sparse_index}[0] < {sequence_length}:"
        )
        kv_value_ref = _row_phased_attention_selected_kv_value_ref(
            kv_fp8_input,
            access_by_buffer,
            shape_env,
            selected_row_expr=f"{sparse_index}[0]",
            head_expr=kv_head,
            dim_expr=source_dim,
        )
        kv_value_scale_ref = _row_phased_attention_selected_kv_scale_ref(
            kv_scale_input,
            access_by_buffer,
            shape_env,
            selected_row_expr=f"{sparse_index}[0]",
            head_expr=kv_head,
        )
        body.append(
            f"{indent * 7}{value_accum}[0] = {value_accum}[0] + "
            f"({score_weights}[{k_top}] * fp8_e4m3fn_to_float({kv_value_ref}) * "
            f"{kv_value_scale_ref})"
        )
        body.append(f"{indent * 5}{context_accum}[0] = 0.0")
        body.append(f"{indent * 5}if {sumexp}[0] > 0.0:")
        body.append(
            f"{indent * 6}{context_accum}[0] = "
            f"{value_accum}[0] / {sumexp}[0]"
        )
        body.append(
            f"{indent * 5}{context_values}["
            f"{source_head_loop} * {head_dim} + {source_dim_loop}"
            f"] = {context_accum}[0]"
        )
        body.append(f"{indent * 3}T.sync_threads()")
        body.append(
            f"{indent * 3}for {out_dim_loop} in T.serial(lane, {hidden_size}, "
            f"step={thread_count}):"
        )
        attention_out_ref = _indexed_buffer_ref(
            attention_out,
            access_by_buffer,
            f"row * {hidden_size} + {out_dim_loop}",
        )
        out_bias = _optional_indexed_buffer_expr(
            "attention_out_proj_bias",
            dtype_by_buffer,
            access_by_buffer,
            default="0.0",
            index_expr=out_dim_loop,
        )
        body.append(f"{indent * 4}{attention_out_ref} = {out_bias}")
        body.append(f"{indent * 4}for {source_dim_loop} in T.serial(0, {q_dim}):")
        out_weight = _optional_indexed_buffer_expr(
            "attention_out_proj_weight",
            dtype_by_buffer,
            access_by_buffer,
            default="1.0",
            index_expr=(
                f"({out_dim_loop}) * {q_dim} + "
                f"{source_dim_loop}"
            ),
        )
        body.append(
            f"{indent * 5}{attention_out_ref} = {attention_out_ref} + "
            f"({context_values}[{source_dim_loop}] * {out_weight})"
        )
        body.append(f"{indent * 3}T.sync_threads()")
    if lse_output is not None and attention_out is None:
        body.append(
            f"{indent * 3}for {lse_head_loop} in T.serial(lane, {q_heads}, "
            f"step={thread_count}):"
        )
        body.append(f"{indent * 4}{q_head_index}[0] = {lse_head}")
        body.append(f"{indent * 4}{kv_head_index}[0] = {q_head} // {q_per_kv}")
        body.append(
            f"{indent * 4}{sink_enabled}[0] = "
            f"T.cast({has_sinks} != 0, \"float32\")"
        )
        body.append(
            f"{indent * 4}{score_max}[0] = "
            "T.float32(-3.4028234663852886e38)"
        )
        body.append(f"{indent * 4}for {k_top} in T.serial(0, {topk}):")
        append_score_for_current_head(
            loop_indent=indent * 5,
            head_expr=q_head,
            kv_head_expr=kv_head,
        )
        body.append(f"{indent * 5}if {score_accum}[0] > {score_max}[0]:")
        body.append(f"{indent * 6}{score_max}[0] = {score_accum}[0]")
        body.append(f"{indent * 4}if {sink_enabled}[0] != 0.0:")
        body.append(f"{indent * 5}if {sinks} > {score_max}[0]:")
        body.append(f"{indent * 6}{score_max}[0] = {sinks}")
        body.append(f"{indent * 4}{sumexp}[0] = 0.0")
        body.append(f"{indent * 4}for {k_top} in T.serial(0, {topk}):")
        append_score_for_current_head(
            loop_indent=indent * 5,
            head_expr=q_head,
            kv_head_expr=kv_head,
        )
        body.append(
            f"{indent * 5}{sumexp}[0] = {sumexp}[0] + "
            f"T.exp({score_accum}[0] - {score_max}[0])"
        )
        body.append(f"{indent * 4}if {sink_enabled}[0] != 0.0:")
        body.append(
            f"{indent * 5}{sumexp}[0] = {sumexp}[0] + "
            f"T.exp({sinks} - {score_max}[0])"
        )
        lse_ref = _row_phased_lse_ref(
            lse_output,
            access_by_buffer,
            shape_env,
            row_expr="row",
            head_expr=q_head,
        )
        body.append(
            f"{indent * 4}{lse_ref} = "
            "0.0"
        )
        body.append(f"{indent * 4}if {sumexp}[0] > 0.0:")
        body.append(
            f"{indent * 5}{lse_ref} = "
            f"{score_max}[0] + T.log({sumexp}[0])"
        )


def _append_attention_projection_head_dot_products(
    body: list[str],
    *,
    projected_vec: str,
    hidden: str,
    weight_buffer: str,
    bias_buffer: str,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    hidden_size: int,
    head_dim: int,
    head_expr: str,
    hidden_loop_var: str,
    src_index_var: str,
    dim_loop_var: str,
    indent: str,
) -> None:
    body.append(f"{indent}for {dim_loop_var} in T.serial(0, {head_dim}):")
    output_offset = f"{head_expr} * {head_dim} + {dim_loop_var}"
    bias = _optional_indexed_buffer_expr(
        bias_buffer,
        dtype_by_buffer,
        access_by_buffer,
        index_expr=output_offset,
    )
    body.append(f"{indent}    {projected_vec}[{dim_loop_var}] = {bias}")
    body.append(f"{indent}for {hidden_loop_var} in T.serial(0, {hidden_size}):")
    body.append(
        f"{indent}    {src_index_var} = row * {hidden_size} + "
        f"{hidden_loop_var}"
    )
    body.append(f"{indent}    for {dim_loop_var} in T.serial(0, {head_dim}):")
    output_offset = f"{head_expr} * {head_dim} + {dim_loop_var}"
    weight = _optional_indexed_buffer_expr(
        weight_buffer,
        dtype_by_buffer,
        access_by_buffer,
        index_expr=f"({output_offset}) * {hidden_size} + {hidden_loop_var}",
    )
    body.append(
        f"{indent}        {projected_vec}[{dim_loop_var}] = "
        f"{projected_vec}[{dim_loop_var}] + ({hidden} * {weight})"
    )


def _append_attention_projection_prepare_from_vector(
    body: list[str],
    *,
    projected_vec: str,
    projected: str,
    paired_projected: str,
    rope_phase: str,
    prepared: str,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    head_dim: int,
    dim_expr: str,
    indent: str,
) -> None:
    rope_half = max(1, head_dim // 2)
    body.append(f"{indent}{projected}[0] = {projected_vec}[{dim_expr}]")
    body.append(f"{indent}if {dim_expr} < {rope_half}:")
    body.append(
        f"{indent}    {paired_projected}[0] = "
        f"{projected_vec}[{dim_expr} + {rope_half}]"
    )
    rope_first = _optional_indexed_buffer_expr(
        "attention_rope_inv_freq",
        dtype_by_buffer,
        access_by_buffer,
        index_expr=dim_expr,
    )
    body.append(
        f"{indent}    {rope_phase}[0] = "
        f'T.cast(row, "float32") * {rope_first}'
    )
    body.append(
        f"{indent}    {prepared}[0] = "
        f"({projected}[0] * T.cos({rope_phase}[0])) + "
        f"({paired_projected}[0] * T.sin({rope_phase}[0]))"
    )
    body.append(f"{indent}else:")
    body.append(
        f"{indent}    {paired_projected}[0] = "
        f"{projected_vec}[{dim_expr} - {rope_half}]"
    )
    rope_second = _optional_indexed_buffer_expr(
        "attention_rope_inv_freq",
        dtype_by_buffer,
        access_by_buffer,
        index_expr=f"{dim_expr} - {rope_half}",
    )
    body.append(
        f"{indent}    {rope_phase}[0] = "
        f'T.cast(row, "float32") * {rope_second}'
    )
    body.append(
        f"{indent}    {prepared}[0] = "
        f"({projected}[0] * T.cos({rope_phase}[0])) - "
        f"({paired_projected}[0] * T.sin({rope_phase}[0]))"
    )


def _attention_qkv_projection_prepare_statements(
    *,
    projected: str,
    rope_phase: str,
    prepared: str,
    hidden: str,
    weight: str,
    bias: str,
    rope: str,
    indent: str,
) -> list[str]:
    return [
        f"{indent}{projected}[0] = {hidden} + {weight} + {bias}",
        f"{indent}{rope_phase}[0] = {rope}",
        f"{indent}{prepared}[0] = {projected}[0] + {rope_phase}[0]",
    ]


def _row_phased_attention_scale_ref(
    buffer_name: str,
    access_by_buffer: dict[str, str],
    shape_env: PathCModelShapeEnv,
    *,
    row_expr: str,
    head_expr: str,
    q_side: bool,
) -> str:
    name = _safe_identifier(buffer_name)
    hidden = int(shape_env.hidden_size)
    head_dim = int(shape_env.attention_head_dim)
    heads = (
        int(shape_env.attention_num_q_heads)
        if q_side
        else int(shape_env.attention_num_kv_heads)
    )
    internal_ref = (
        f"{name}[(i % {hidden}) // {head_dim}]"
        if q_side
        else f"{name}[((i % {hidden}) // {head_dim}) % {heads}]"
    )
    if access_by_buffer.get(buffer_name) == internal_ref:
        return f"{name}[{head_expr}]"
    return _row_phased_direct_flat_ref(
        buffer_name,
        access_by_buffer,
        f"{row_expr} * {heads} + {head_expr}",
    )


def _row_phased_attention_value_ref(
    buffer_name: str,
    access_by_buffer: dict[str, str],
    shape_env: PathCModelShapeEnv,
    *,
    row_expr: str,
    head_expr: str,
    dim_expr: str,
    q_side: bool,
) -> str:
    name = _safe_identifier(buffer_name)
    head_dim = int(shape_env.attention_head_dim)
    heads = (
        int(shape_env.attention_num_q_heads)
        if q_side
        else int(shape_env.attention_num_kv_heads)
    )
    row_width = heads * head_dim
    offset = f"{head_expr} * {head_dim} + {dim_expr}"
    if access_by_buffer.get(buffer_name) == f"{name}[i % {row_width}]":
        return f"{name}[{offset}]"
    return _row_phased_direct_flat_ref(
        buffer_name,
        access_by_buffer,
        f"{row_expr} * {row_width} + {offset}",
    )


def _row_phased_attention_selected_kv_value_ref(
    buffer_name: str,
    access_by_buffer: dict[str, str],
    shape_env: PathCModelShapeEnv,
    *,
    selected_row_expr: str,
    head_expr: str,
    dim_expr: str,
) -> str:
    name = _safe_identifier(buffer_name)
    head_dim = int(shape_env.attention_head_dim)
    heads = int(shape_env.attention_num_kv_heads)
    row_width = heads * head_dim
    offset = f"{head_expr} * {head_dim} + {dim_expr}"
    if access_by_buffer.get(buffer_name) == f"{name}[i % {row_width}]":
        return f"{name}[{offset}]"
    return _row_phased_direct_flat_ref(
        buffer_name,
        access_by_buffer,
        f"{selected_row_expr} * {row_width} + {offset}",
    )


def _row_phased_attention_selected_kv_scale_ref(
    buffer_name: str,
    access_by_buffer: dict[str, str],
    shape_env: PathCModelShapeEnv,
    *,
    selected_row_expr: str,
    head_expr: str,
) -> str:
    name = _safe_identifier(buffer_name)
    heads = int(shape_env.attention_num_kv_heads)
    internal_ref = (
        f"{name}[((i % {shape_env.hidden_size}) // "
        f"{shape_env.attention_head_dim}) % {heads}]"
    )
    if access_by_buffer.get(buffer_name) == internal_ref:
        return f"{name}[{head_expr}]"
    return _row_phased_direct_flat_ref(
        buffer_name,
        access_by_buffer,
        f"{selected_row_expr} * {heads} + {head_expr}",
    )


def _row_phased_direct_flat_ref(
    buffer_name: str,
    access_by_buffer: dict[str, str],
    flat_index_expr: str,
) -> str:
    """Return a direct flat reference for row-phased full-sequence buffers.

    Full-sequence attention scale buffers use hidden-indexed loop refs such as
    ``(i // H) * heads + ...``.  Substituting a scale-space index back into
    that expression collapses ``row * heads + head`` into ``row + head // d``.
    Emit the already-flat row-major index directly instead.
    """

    ref = access_by_buffer.get(buffer_name)
    name = _safe_identifier(buffer_name)
    if ref is not None:
        match = re.match(r"([A-Za-z_]\w*)\[(\d+) \+ ", ref)
        if match is not None:
            return f"{match.group(1)}[{match.group(2)} + {flat_index_expr}]"
        bank_match = re.fullmatch(r"([A-Za-z_]\w*)\[(.+)\]", ref)
        if bank_match is not None and "_abi_bank" in bank_match.group(1):
            return f"{bank_match.group(1)}[{flat_index_expr}]"
        match = re.fullmatch(r"([A-Za-z_]\w*)\[i\]", ref)
        if match is not None:
            return f"{match.group(1)}[{flat_index_expr}]"
    return f"{name}[{flat_index_expr}]"


def _row_phased_attention_indices_ref(
    buffer_name: str,
    access_by_buffer: dict[str, str],
    shape_env: PathCModelShapeEnv,
    *,
    head_expr: str,
    topk_expr: str,
) -> str:
    name = _safe_identifier(buffer_name)
    kv_heads = int(shape_env.attention_num_kv_heads)
    topk = int(shape_env.attention_sparse_topk)
    row_width = kv_heads * topk
    offset = f"{head_expr} * {topk} + {topk_expr}"
    ref = access_by_buffer.get(buffer_name)
    full_sequence_ref = (
        f"{name}[(i // {shape_env.hidden_size}) * {row_width} + "
        f"(i % {row_width})]"
    )
    if ref == full_sequence_ref:
        return f"{name}[row * {row_width} + {offset}]"
    if ref == f"{name}[i % {row_width}]":
        # Row-local internal buffer — offset within single row.
        return f"{name}[{offset}]"
    if ref is not None and f"% {row_width}" not in ref:
        # Full-sequence buffer — bypass _indexed_buffer_ref to avoid
        # broken i-substitution in complex access expressions.
        # Handle banked buffers (e.g. path_c_float32_abi_bank[OFFSET + ...]).
        match = re.match(r"([A-Za-z_]\w*)\[(\d+) \+ ", ref)
        if match is not None:
            bank_name = match.group(1)
            bank_offset = match.group(2)
            return f"{bank_name}[{bank_offset} + row * {row_width} + {offset}]"
        bank_match = re.fullmatch(r"([A-Za-z_]\w*)\[(.+)\]", ref)
        if bank_match is not None and "_abi_bank" in bank_match.group(1):
            return f"{bank_match.group(1)}[row * {row_width} + {offset}]"
        return f"{name}[row * {row_width} + {offset}]"
    return _indexed_buffer_ref(
        buffer_name,
        access_by_buffer,
        f"row * {row_width} + {offset}",
    )


def _row_phased_attention_bwd_indices_ref(
    buffer_name: str,
    access_by_buffer: dict[str, str],
    shape_env: PathCModelShapeEnv,
    *,
    row_expr: str,
    head_expr: str,
    topk_expr: str,
) -> str:
    """Return row-specific Sparse-MLA indices for the recompute bwd loop.

    The generated attention projection keeps causal sparse indices row-local
    during forward.  The backward row loop runs after the forward row loop, so
    reading that scratch would reuse the final row's indices for every row.
    Recompute the descriptor's causal index formula instead.
    """

    name = _safe_identifier(buffer_name)
    kv_heads = int(shape_env.attention_num_kv_heads)
    topk = int(shape_env.attention_sparse_topk)
    row_width = kv_heads * topk
    ref = access_by_buffer.get(buffer_name)
    offset = f"{head_expr} * {topk} + {topk_expr}"
    full_expr = f"(i // {shape_env.hidden_size}) * {row_width} + (i % {row_width})"
    row_full_expr = f"{row_expr} * {row_width} + {offset}"
    if ref == f"{name}[{full_expr}]":
        return f"{name}[{row_full_expr}]"
    if ref is not None:
        bank_match = re.fullmatch(r"([A-Za-z_]\w*)\[(.+)\]", ref)
        if bank_match is not None:
            bank_name = bank_match.group(1)
            bank_expr = bank_match.group(2)
            if bank_expr == full_expr:
                return f"{bank_name}[{row_full_expr}]"
            offset_match = re.fullmatch(
                rf"(\d+) \+ \({re.escape(full_expr)}\)",
                bank_expr,
            )
            if offset_match is not None:
                return f"{bank_name}[{offset_match.group(1)} + ({row_full_expr})]"
    if ref == f"{name}[i % {row_width}]":
        return f"T.if_then_else({row_expr} >= {topk_expr}, {row_expr} - {topk_expr}, -1)"
    if ref is not None:
        match = re.match(r"([A-Za-z_]\w*)\[(\d+) \+ ", ref)
        if match is not None:
            return (
                f"{match.group(1)}[{match.group(2)} + "
                f"{row_expr} * {row_width} + {offset}]"
            )
        bank_match = re.fullmatch(r"([A-Za-z_]\w*)\[(.+)\]", ref)
        if bank_match is not None and "_abi_bank" in bank_match.group(1):
            return f"{bank_match.group(1)}[{row_expr} * {row_width} + {offset}]"
    return f"{name}[{row_expr} * {row_width} + {offset}]"


def _row_phased_lse_ref(
    buffer_name: str,
    access_by_buffer: dict[str, str],
    shape_env: PathCModelShapeEnv,
    *,
    row_expr: str,
    head_expr: str,
) -> str:
    """Return a direct reference for lse in the row-phased emitter.

    Bypasses ``_indexed_buffer_ref`` because the complex access pattern
    generated by ``_loop_indexed_buffer_ref`` for full-sequence lse
    buffers (``name[(i // H) * Q + ((i % H) // d)]``) produces wrong
    addresses when ``i`` is naively substituted with a flat lse index
    like ``row * Q + head``.
    """
    name = _safe_identifier(buffer_name)
    q_heads = int(shape_env.attention_num_q_heads)
    ref = access_by_buffer.get(buffer_name)
    if ref == f"{name}[i % {q_heads}]":
        # Row-local internal buffer — offset within single row.
        return f"{name}[{head_expr}]"
    # Full-sequence buffer.  The access pattern may be banked, e.g.
    #   path_c_float32_abi_bank[OFFSET + ((i // H) * Q + ...)]
    # Extract the bank name and numeric offset so we can emit a
    # direct flat reference without i-substitution.
    if ref is not None:
        match = re.match(r"([A-Za-z_]\w*)\[(\d+) \+ ", ref)
        if match is not None:
            bank_name = match.group(1)
            offset = match.group(2)
            return f"{bank_name}[{offset} + {row_expr} * {q_heads} + {head_expr}]"
        bank_match = re.fullmatch(r"([A-Za-z_]\w*)\[(.+)\]", ref)
        if bank_match is not None and "_abi_bank" in bank_match.group(1):
            return f"{bank_match.group(1)}[{row_expr} * {q_heads} + {head_expr}]"
    # Unbanked fallback — direct flat index.
    return f"{name}[{row_expr} * {q_heads} + {head_expr}]"


def _is_full_sequence_bank_slot(
    buffer_name: str,
    access_by_buffer: Mapping[str, str],
) -> bool:
    """Return True when ``buffer_name`` resolves to a full-sequence bank slot.

    A "full-sequence bank slot" is a logical buffer whose physical access
    expression resolves to ``path_c_..._abi_bank[OFFSET + i]`` (i.e. the
    index depends on the kernel-wide loop variable ``i`` covering the
    full ``sequence_length * hidden_size`` range), as opposed to a
    per-row scratch buffer that wraps around each row iteration.

    The schedule uses this signal to decide whether the bwd needs a
    one-shot pre-zero (full-sequence bank slot) or whether per-row
    zero-init followed by ``=`` accumulation is sufficient (per-row
    scratch buffer).
    """
    if buffer_name not in access_by_buffer:
        return False
    access = access_by_buffer[buffer_name]
    if "_abi_bank[" not in access:
        return False
    if " % " in access:
        return False
    return True


def _append_row_phased_residual_rmsnorm_bwd_init(
    body: list[str],
    *,
    node: _ScheduleNodeView,
    access_by_buffer: dict[str, str],
    hidden_size: int,
    thread_count: int,
    indent: str,
) -> None:
    if len(node.outputs) < 3:
        return
    weight_grad = node.outputs[2]
    body.append(f"{indent * 2}for h in T.serial(lane, {hidden_size}, step={thread_count}):")
    body.append(
        f"{indent * 3}{_buffer_ref(weight_grad, access_by_buffer, 'h')} = 0.0"
    )
    body.append(f"{indent * 2}T.sync_threads()")


def _append_row_phased_entry_rmsnorm_bwd_init(
    body: list[str],
    *,
    node: _ScheduleNodeView,
    access_by_buffer: dict[str, str],
    hidden_size: int,
    sequence_length: int,
    thread_count: int,
    indent: str,
) -> None:
    """Zero-init the entry RMSNorm weight grad before the row loop.

    Block A: the per-brick entry RMSNorm weight gradient is accumulated
    across every row during the bwd pass, so it must start at zero.
    The bank ``hidden_grad`` slot is NOT zero-initialised here -- the
    inter-brick ``residual_rmsnorm_bwd`` writes it with ``=`` first (per
    Fix B-1), and entry_rmsnorm_bwd then accumulates onto that value
    with ``+=``.
    """

    # The entry RMSNorm's normalized-hidden output is consumed downstream
    # by the first in-region brick's forward, which means the brick's
    # backward writes its hidden_grad contribution into that same bank
    # slot via ``+=``. Because the slot is a full-sequence bank slot, the
    # block bwd emitter skips its per-row zero-init (Fix B-1) -- so the
    # entry RMSNorm op (the only logical "owner" of that slot) is
    # responsible for one-shot zeroing the buffer before the row loop
    # starts. The slot is the first bwd input of entry_rmsnorm_bwd
    # (``normed_hidden_grad``).
    if len(node.inputs) >= 1:
        normed_hidden_grad = node.inputs[0]
        if _is_full_sequence_bank_slot(normed_hidden_grad, access_by_buffer):
            body.append(
                f"{indent * 2}for i in T.serial(lane, "
                f"{int(sequence_length) * int(hidden_size)}, step={thread_count}):"
            )
            body.append(
                f"{indent * 3}# entry_rmsnorm_bwd: pre-zero normed_hidden_grad bank slot"
            )
            body.append(
                f"{indent * 3}{_buffer_ref(normed_hidden_grad, access_by_buffer, 'i')} = 0.0"
            )
            body.append(f"{indent * 2}T.sync_threads()")

    if len(node.outputs) < 2:
        return
    weight_grad = node.outputs[1]
    body.append(f"{indent * 2}for h in T.serial(lane, {hidden_size}, step={thread_count}):")
    body.append(
        f"{indent * 3}{_buffer_ref(weight_grad, access_by_buffer, 'h')} = 0.0"
    )
    body.append(f"{indent * 2}T.sync_threads()")


def _append_row_phased_residual_rmsnorm_bwd_body(
    body: list[str],
    *,
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    hidden_size: int,
    thread_count: int,
    indent: str,
) -> None:
    sum_sq = _scratch_name(node, "row_sum_sq")
    sum_sq_partial = _scratch_name(node, "row_sum_sq_partial")
    inv_rms = _scratch_name(node, "row_inv_rms")
    dot = _scratch_name(node, "row_dot")
    dot_partial = _scratch_name(node, "row_dot_partial")
    norm_grad_scratch = _scratch_name(node, "row_norm_grad")
    total_grad_scratch = _scratch_name(node, "row_total_grad")
    _append_descriptor_node_comments(
        body,
        node=node,
        descriptor=descriptor,
        indent=indent * 3,
    )
    if len(node.inputs) >= 5:
        hidden_after_grad = _node_input_expr(
            node,
            0,
            dtype_by_buffer,
            access_by_buffer,
            "i",
        )
        norm_grad = _node_input_expr(node, 1, dtype_by_buffer, access_by_buffer, "i")
        hidden = _node_input_expr(node, 2, dtype_by_buffer, access_by_buffer, "i")
        delta = _node_input_expr(node, 3, dtype_by_buffer, access_by_buffer, "i")
        weight = _node_input_expr(node, 4, dtype_by_buffer, access_by_buffer, "i")
    else:
        if _node_output_for_canonical(node, "m2rnn_delta") is not None:
            hidden_after_grad_name = "hidden_after_m2rnn_grad"
        elif _node_output_for_canonical(node, "mamba3_delta") is not None:
            hidden_after_grad_name = "hidden_after_mamba3_grad"
        else:
            hidden_after_grad_name = ""
        hidden_after_grad = (
            _optional_indexed_buffer_expr(
                hidden_after_grad_name,
                dtype_by_buffer,
                access_by_buffer,
                default="0.0",
                index_expr="i",
            )
            if hidden_after_grad_name
            else "0.0"
        )
        norm_grad = _node_input_expr(node, 0, dtype_by_buffer, access_by_buffer, "i")
        hidden = _node_input_expr(node, 1, dtype_by_buffer, access_by_buffer, "i")
        delta = _node_input_expr(node, 2, dtype_by_buffer, access_by_buffer, "i")
        weight = _node_input_expr(node, 3, dtype_by_buffer, access_by_buffer, "i")
    residual_expr = f"({hidden} + {delta})"
    body.append(f"{indent * 3}{sum_sq_partial}[lane] = 0.0")
    body.append(f"{indent * 3}{dot_partial}[lane] = 0.0")
    body.append(
        f"{indent * 3}for i in T.serial(row * {hidden_size} + lane, "
        f"(row + 1) * {hidden_size}, step={thread_count}):"
    )
    body.append(
        f"{indent * 4}{sum_sq_partial}[lane] = {sum_sq_partial}[lane] + "
        f"({residual_expr} * {residual_expr})"
    )
    body.append(
        f"{indent * 4}{dot_partial}[lane] = {dot_partial}[lane] + "
        f"({norm_grad} * {weight} * {residual_expr})"
    )
    body.append(f"{indent * 3}T.sync_threads()")
    body.append(f"{indent * 3}if lane == 0:")
    body.append(f"{indent * 4}{sum_sq}[0] = 0.0")
    body.append(f"{indent * 4}{dot}[0] = 0.0")
    body.append(f"{indent * 4}for partial_lane in T.serial(0, {thread_count}):")
    body.append(
        f"{indent * 5}{sum_sq}[0] = {sum_sq}[0] + "
        f"{sum_sq_partial}[partial_lane]"
    )
    body.append(
        f"{indent * 5}{dot}[0] = {dot}[0] + {dot_partial}[partial_lane]"
    )
    body.append(
        f"{indent * 4}{inv_rms}[0] = T.rsqrt(({sum_sq}[0] / "
        f"{float(hidden_size):.1f}) + 0.00001)"
    )
    body.append(f"{indent * 3}T.sync_threads()")
    body.append(
        f"{indent * 3}for i in T.serial(row * {hidden_size} + lane, "
        f"(row + 1) * {hidden_size}, step={thread_count}):"
    )
    body.append(f"{indent * 4}{norm_grad_scratch}[0] = {norm_grad} * {weight}")
    body.append(
        f"{indent * 4}{total_grad_scratch}[0] = {hidden_after_grad} + "
        f"({inv_rms}[0] * ({norm_grad_scratch}[0] - "
        f"({residual_expr} * {dot}[0] * {inv_rms}[0] * "
        f"{inv_rms}[0] / {float(hidden_size):.1f})))"
    )
    for output_name in node.outputs[:2]:
        body.append(
            f"{indent * 4}{_buffer_ref(output_name, access_by_buffer, 'i')} = "
            f"{total_grad_scratch}[0]"
        )
    if len(node.outputs) > 2:
        weight_grad = _buffer_ref(node.outputs[2], access_by_buffer, "i")
        body.append(
            f"{indent * 4}{weight_grad} = {weight_grad} + "
            f"({norm_grad} * {residual_expr} * {inv_rms}[0])"
        )
    body.append(f"{indent * 3}T.sync_threads()")


def _append_row_phased_entry_rmsnorm_bwd_body(
    body: list[str],
    *,
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    hidden_size: int,
    thread_count: int,
    indent: str,
) -> None:
    """Emit the row-phased backward body for the entry RMSNorm op.

    Block A: this mirrors :func:`_append_row_phased_residual_rmsnorm_bwd_body`
    but skips the residual ``+ delta`` reduction. The forward computed
    ``y[h] = x[h] * inv_rms * w[h]`` with ``inv_rms = rsqrt(mean(x*x) + eps)``.
    The backward recomputes ``inv_rms`` from ``hidden`` alone, then derives
    ``hidden_grad += inv_rms * (norm_grad*w - x * dot * inv_rms^2 / D)`` and
    ``weight_grad[h%H] += norm_grad * x * inv_rms`` row-by-row.

    The bank ``hidden_grad`` slot is written with ``+=`` because the
    inter-brick ``residual_rmsnorm_bwd`` already wrote it with ``=`` in
    the same iteration (Fix B-1 convention).
    """

    sum_sq = _scratch_name(node, "row_sum_sq")
    sum_sq_partial = _scratch_name(node, "row_sum_sq_partial")
    inv_rms = _scratch_name(node, "row_inv_rms")
    dot = _scratch_name(node, "row_dot")
    dot_partial = _scratch_name(node, "row_dot_partial")
    norm_grad_scratch = _scratch_name(node, "row_norm_grad")
    total_grad_scratch = _scratch_name(node, "row_total_grad")
    _append_descriptor_node_comments(
        body,
        node=node,
        descriptor=descriptor,
        indent=indent * 3,
    )
    norm_grad = _node_input_expr(node, 0, dtype_by_buffer, access_by_buffer, "i")
    hidden = _node_input_expr(node, 1, dtype_by_buffer, access_by_buffer, "i")
    weight = _node_input_expr(node, 2, dtype_by_buffer, access_by_buffer, "i")
    body.append(f"{indent * 3}{sum_sq_partial}[lane] = 0.0")
    body.append(f"{indent * 3}{dot_partial}[lane] = 0.0")
    body.append(
        f"{indent * 3}for i in T.serial(row * {hidden_size} + lane, "
        f"(row + 1) * {hidden_size}, step={thread_count}):"
    )
    body.append(
        f"{indent * 4}{sum_sq_partial}[lane] = {sum_sq_partial}[lane] + "
        f"({hidden} * {hidden})"
    )
    body.append(
        f"{indent * 4}{dot_partial}[lane] = {dot_partial}[lane] + "
        f"({norm_grad} * {weight} * {hidden})"
    )
    body.append(f"{indent * 3}T.sync_threads()")
    body.append(f"{indent * 3}if lane == 0:")
    body.append(f"{indent * 4}{sum_sq}[0] = 0.0")
    body.append(f"{indent * 4}{dot}[0] = 0.0")
    body.append(f"{indent * 4}for partial_lane in T.serial(0, {thread_count}):")
    body.append(
        f"{indent * 5}{sum_sq}[0] = {sum_sq}[0] + "
        f"{sum_sq_partial}[partial_lane]"
    )
    body.append(
        f"{indent * 5}{dot}[0] = {dot}[0] + {dot_partial}[partial_lane]"
    )
    body.append(
        f"{indent * 4}{inv_rms}[0] = T.rsqrt(({sum_sq}[0] / "
        f"{float(hidden_size):.1f}) + 0.00001)"
    )
    body.append(f"{indent * 3}T.sync_threads()")
    body.append(
        f"{indent * 3}for i in T.serial(row * {hidden_size} + lane, "
        f"(row + 1) * {hidden_size}, step={thread_count}):"
    )
    body.append(f"{indent * 4}{norm_grad_scratch}[0] = {norm_grad} * {weight}")
    body.append(
        f"{indent * 4}{total_grad_scratch}[0] = "
        f"({inv_rms}[0] * ({norm_grad_scratch}[0] - "
        f"({hidden} * {dot}[0] * {inv_rms}[0] * "
        f"{inv_rms}[0] / {float(hidden_size):.1f})))"
    )
    if node.outputs:
        hidden_grad_ref = _buffer_ref(node.outputs[0], access_by_buffer, "i")
        if _is_full_sequence_bank_slot(node.outputs[0], access_by_buffer):
            # Fix B-1 chain: residual_rmsnorm_bwd already wrote
            # hidden_grad with ``=`` in the same row iteration; entry
            # RMSNorm bwd is the second writer and must accumulate.
            body.append(
                f"{indent * 4}{hidden_grad_ref} = {hidden_grad_ref} + "
                f"{total_grad_scratch}[0]"
            )
        else:
            # Non-bank scratch slot: first writer wins.
            body.append(f"{indent * 4}{hidden_grad_ref} = {total_grad_scratch}[0]")
    if len(node.outputs) > 1:
        weight_grad = _buffer_ref(node.outputs[1], access_by_buffer, "i")
        body.append(
            f"{indent * 4}{weight_grad} = {weight_grad} + "
            f"({norm_grad} * {hidden} * {inv_rms}[0])"
        )
    body.append(f"{indent * 3}T.sync_threads()")


def _append_row_phased_sparse_mla_fp8_apply_bwd_body(
    body: list[str],
    *,
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    shape_env: PathCModelShapeEnv,
    thread_count: int,
    chunked_rows: bool,
    indent: str,
) -> None:
    sink_enabled = _scratch_name(node, "sink_enabled")
    sparse_index = _scratch_name(node, "sparse_index")
    sparse_indices = _scratch_name(node, "sparse_indices")
    score_accum = _scratch_name(node, "score_accum")
    score_max = _scratch_name(node, "score_max")
    score_weight = _scratch_name(node, "score_weight")
    score_weights = _scratch_name(node, "score_weights")
    sumexp = _scratch_name(node, "sumexp")
    context_values = _scratch_name(node, "context_values")
    context_grads = _scratch_name(node, "context_grads")
    q_deq_values = _scratch_name(node, "attention_q_deq_values")
    value_accum = _scratch_name(node, "value_accum")
    dp_values = _scratch_name(node, "dp_values")
    dp_accum = _scratch_name(node, "dp_accum")
    ds_score = _scratch_name(node, "ds_score")
    dq_accum = _scratch_name(node, "dq_accum")
    dkv_accum = _scratch_name(node, "dkv_accum")
    q_head = _scratch_name(node, "q_head")
    kv_head = _scratch_name(node, "kv_head")
    dim = _scratch_name(node, "dim")
    dot_dim = _scratch_name(node, "dot_dim")
    out_dim = _scratch_name(node, "out_dim")
    k_top = _scratch_name(node, "k_top")
    flat = _scratch_name(node, "flat")
    hidden_size = int(shape_env.hidden_size)
    q_heads = int(shape_env.attention_num_q_heads)
    kv_heads = int(shape_env.attention_num_kv_heads)
    head_dim = int(shape_env.attention_head_dim)
    sequence_length = int(shape_env.sequence_length)
    q_dim = q_heads * head_dim
    topk = int(shape_env.attention_sparse_topk)
    q_per_kv = max(1, q_heads // max(1, kv_heads))
    q_fp8_grad = _node_output_for_canonical(node, "q_fp8")
    q_scale_grad = _node_output_for_canonical(node, "q_scale")
    kv_fp8_grad = _node_output_for_canonical(node, "kv_fp8")
    kv_scale_grad = _node_output_for_canonical(node, "kv_scale")
    out_weight_grad = _node_output_for_canonical(node, "attention_out_proj_weight")
    out_bias_grad = _node_output_for_canonical(node, "attention_out_proj_bias")
    q_fp8_input = _node_input_for_canonical(node, "q_fp8")
    q_scale_input = _node_input_for_canonical(node, "q_scale")
    kv_fp8_input = _node_input_for_canonical(node, "kv_fp8")
    kv_scale_input = _node_input_for_canonical(node, "kv_scale")
    indices_input = _node_input_for_canonical(node, "indices")
    if (
        q_fp8_input is None
        or q_scale_input is None
        or kv_fp8_input is None
        or kv_scale_input is None
        or indices_input is None
        or q_fp8_grad is None
        or kv_fp8_grad is None
    ):
        raise ValueError("row-phased sparse MLA backward requires FP8 inputs and grads")
    sm_scale = _optional_buffer_expr(
        "sparse_mla_sm_scale",
        dtype_by_buffer,
        access_by_buffer,
        "1.0",
        "0",
    )
    sinks = _optional_buffer_expr(
        "sparse_mla_sinks",
        dtype_by_buffer,
        access_by_buffer,
        "0.0",
        q_head,
    )
    has_sinks = _optional_buffer_expr(
        "sparse_mla_has_sinks",
        dtype_by_buffer,
        access_by_buffer,
        "0",
        "0",
    )
    _append_descriptor_node_comments(
        body,
        node=node,
        descriptor=descriptor,
        indent=indent * 3,
    )
    body.append(
        f"{indent * 3}# sparse_mla_fp8_apply_bwd_policy: "
        "exact_softmax_vjp_and_out_projection"
    )
    body.append(f"{indent * 3}if row == 0:")
    for output_name, extent in (
        (q_fp8_grad, sequence_length * q_dim),
        (q_scale_grad, sequence_length * q_heads),
        (kv_fp8_grad, sequence_length * kv_heads * head_dim),
        (kv_scale_grad, sequence_length * kv_heads),
    ):
        if output_name is None:
            continue
        body.append(
            f"{indent * 4}for {flat} in T.serial(lane, {extent}, "
            f"step={thread_count}):"
        )
        body.append(
            f"{indent * 5}{_indexed_buffer_ref(output_name, access_by_buffer, flat)} = 0.0"
        )
    body.append(f"{indent * 3}if row == 0:")
    for output_name, extent in (
        (out_weight_grad, hidden_size * q_dim),
        (out_bias_grad, hidden_size),
    ):
        if output_name is None:
            continue
        body.append(
            f"{indent * 4}for {flat} in T.serial(lane, {extent}, "
            f"step={thread_count}):"
        )
        body.append(
            f"{indent * 5}{_indexed_buffer_ref(output_name, access_by_buffer, flat)} = 0.0"
        )
    body.append(f"{indent * 3}T.sync_threads()")
    body.append(f"{indent * 3}{sink_enabled} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}{sparse_index} = T.alloc_local((1,), \"int32\")")
    body.append(f"{indent * 3}{score_accum} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}{score_max} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}{score_weight} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}{sumexp} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}{value_accum} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}{dp_accum} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}{ds_score} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}{dq_accum} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}{dkv_accum} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}{score_weights} = T.alloc_local(({topk},), \"float32\")")
    body.append(f"{indent * 3}{dp_values} = T.alloc_local(({topk},), \"float32\")")
    body.append(f"{indent * 3}{sparse_indices} = T.alloc_local(({topk},), \"int32\")")
    body.append(f"{indent * 3}{context_values} = T.alloc_local(({head_dim},), \"float32\")")
    body.append(f"{indent * 3}{context_grads} = T.alloc_local(({head_dim},), \"float32\")")
    body.append(f"{indent * 3}{q_deq_values} = T.alloc_local(({head_dim},), \"float32\")")
    body.append(
        f"{indent * 3}for {q_head} in T.serial(lane, {q_heads}, step={thread_count}):"
    )
    body.append(f"{indent * 4}{kv_head} = {q_head} // {q_per_kv}")
    body.append(
        f"{indent * 4}{sink_enabled}[0] = T.cast({has_sinks} != 0, \"float32\")"
    )
    body.append(f"{indent * 4}# q_dequant_bwd_policy: saved_prepared_fp8")
    q_scale_current = _row_phased_attention_scale_ref(
        q_scale_input,
        access_by_buffer,
        shape_env,
        row_expr="row",
        head_expr=q_head,
        q_side=True,
    )
    body.append(f"{indent * 4}for {dim} in T.serial(0, {head_dim}):")
    q_saved_ref = _row_phased_attention_value_ref(
        q_fp8_input,
        access_by_buffer,
        shape_env,
        row_expr="row",
        head_expr=q_head,
        dim_expr=dim,
        q_side=True,
    )
    body.append(
        f"{indent * 5}{q_deq_values}[{dim}] = "
        f"fp8_e4m3fn_to_float({q_saved_ref}) * {q_scale_current}"
    )
    body.append(f"{indent * 4}for {k_top} in T.serial(0, {topk}):")
    indices_ref = _row_phased_attention_bwd_indices_ref(
        indices_input,
        access_by_buffer,
        shape_env,
        row_expr="row",
        head_expr=kv_head,
        topk_expr=k_top,
    )
    body.append(f"{indent * 5}{sparse_indices}[{k_top}] = {indices_ref}")
    body.append(f"{indent * 4}{score_max}[0] = T.float32(-3.4028234663852886e38)")
    body.append(f"{indent * 4}for {k_top} in T.serial(0, {topk}):")
    body.append(f"{indent * 5}{sparse_index}[0] = {sparse_indices}[{k_top}]")
    body.append(
        f"{indent * 5}if {sparse_index}[0] >= 0 and "
        f"{sparse_index}[0] < {sequence_length}:"
    )
    body.append(f"{indent * 6}{score_accum}[0] = 0.0")
    body.append(f"{indent * 6}for {dot_dim} in T.serial(0, {head_dim}):")
    q_dot_ref = _row_phased_attention_value_ref(
        q_fp8_input,
        access_by_buffer,
        shape_env,
        row_expr="row",
        head_expr=q_head,
        dim_expr=dot_dim,
        q_side=True,
    )
    kv_dot_ref = _row_phased_attention_selected_kv_value_ref(
        kv_fp8_input,
        access_by_buffer,
        shape_env,
        selected_row_expr=f"{sparse_index}[0]",
        head_expr=kv_head,
        dim_expr=dot_dim,
    )
    body.append(
        f"{indent * 7}{score_accum}[0] = {score_accum}[0] + "
        f"({q_deq_values}[{dot_dim}] * fp8_e4m3fn_to_float({kv_dot_ref}))"
    )
    kv_scale_ref = _row_phased_attention_selected_kv_scale_ref(
        kv_scale_input,
        access_by_buffer,
        shape_env,
        selected_row_expr=f"{sparse_index}[0]",
        head_expr=kv_head,
    )
    body.append(
        f"{indent * 6}{score_accum}[0] = {score_accum}[0] * "
        f"{kv_scale_ref} * {sm_scale}"
    )
    body.append(f"{indent * 5}else:")
    body.append(f"{indent * 6}{score_accum}[0] = T.float32(-3.4028234663852886e38)")
    body.append(f"{indent * 5}if {score_accum}[0] > {score_max}[0]:")
    body.append(f"{indent * 6}{score_max}[0] = {score_accum}[0]")
    body.append(f"{indent * 4}if {sink_enabled}[0] != 0.0:")
    body.append(f"{indent * 5}if {sinks} > {score_max}[0]:")
    body.append(f"{indent * 6}{score_max}[0] = {sinks}")
    body.append(f"{indent * 4}{sumexp}[0] = 0.0")
    body.append(f"{indent * 4}for {k_top} in T.serial(0, {topk}):")
    body.append(f"{indent * 5}{sparse_index}[0] = {sparse_indices}[{k_top}]")
    body.append(f"{indent * 5}{score_weights}[{k_top}] = 0.0")
    body.append(
        f"{indent * 5}if {sparse_index}[0] >= 0 and "
        f"{sparse_index}[0] < {sequence_length}:"
    )
    body.append(f"{indent * 6}{score_accum}[0] = 0.0")
    body.append(f"{indent * 6}for {dot_dim} in T.serial(0, {head_dim}):")
    body.append(
        f"{indent * 7}{score_accum}[0] = {score_accum}[0] + "
        f"({q_deq_values}[{dot_dim}] * fp8_e4m3fn_to_float({kv_dot_ref}))"
    )
    body.append(
        f"{indent * 6}{score_accum}[0] = {score_accum}[0] * "
        f"{kv_scale_ref} * {sm_scale}"
    )
    body.append(
        f"{indent * 6}{score_weight}[0] = T.exp({score_accum}[0] - {score_max}[0])"
    )
    body.append(f"{indent * 6}{score_weights}[{k_top}] = {score_weight}[0]")
    body.append(f"{indent * 6}{sumexp}[0] = {sumexp}[0] + {score_weight}[0]")
    body.append(f"{indent * 4}if {sink_enabled}[0] != 0.0:")
    body.append(
        f"{indent * 5}{sumexp}[0] = {sumexp}[0] + T.exp({sinks} - {score_max}[0])"
    )
    body.append(f"{indent * 4}for {dim} in T.serial(0, {head_dim}):")
    body.append(f"{indent * 5}{value_accum}[0] = 0.0")
    body.append(f"{indent * 5}for {k_top} in T.serial(0, {topk}):")
    body.append(f"{indent * 6}{sparse_index}[0] = {sparse_indices}[{k_top}]")
    body.append(
        f"{indent * 6}if {sparse_index}[0] >= 0 and "
        f"{sparse_index}[0] < {sequence_length}:"
    )
    kv_val_ref = _row_phased_attention_selected_kv_value_ref(
        kv_fp8_input,
        access_by_buffer,
        shape_env,
        selected_row_expr=f"{sparse_index}[0]",
        head_expr=kv_head,
        dim_expr=dim,
    )
    body.append(
        f"{indent * 7}{value_accum}[0] = {value_accum}[0] + "
        f"({score_weights}[{k_top}] * fp8_e4m3fn_to_float({kv_val_ref}) * "
        f"{kv_scale_ref})"
    )
    body.append(f"{indent * 5}{context_values}[{dim}] = 0.0")
    body.append(f"{indent * 5}if {sumexp}[0] > 0.0:")
    body.append(f"{indent * 6}{context_values}[{dim}] = {value_accum}[0] / {sumexp}[0]")
    body.append(f"{indent * 5}{context_grads}[{dim}] = 0.0")
    body.append(f"{indent * 5}for {out_dim} in T.serial(0, {hidden_size}):")
    out_grad = _node_indexed_canonical_or_positional_input_expr(
        node,
        ("attention_out", "out"),
        0,
        dtype_by_buffer,
        access_by_buffer,
        f"row * {hidden_size} + {out_dim}",
    )
    out_weight = _node_indexed_canonical_input_expr(
        node,
        "attention_out_proj_weight",
        dtype_by_buffer,
        access_by_buffer,
        f"{out_dim} * {q_dim} + ({q_head} * {head_dim} + {dim})",
        default="1.0",
    )
    body.append(
        f"{indent * 6}{context_grads}[{dim}] = {context_grads}[{dim}] + "
        f"({out_grad} * {out_weight})"
    )
    if out_weight_grad is not None:
        out_weight_ref = _indexed_buffer_ref(
            out_weight_grad,
            access_by_buffer,
            f"{out_dim} * {q_dim} + ({q_head} * {head_dim} + {dim})",
        )
        body.append(
            f"{indent * 6}{out_weight_ref} = {out_weight_ref} + "
            f"({out_grad} * {context_values}[{dim}])"
        )
    body.append(f"{indent * 4}for {k_top} in T.serial(0, {topk}):")
    body.append(f"{indent * 5}{sparse_index}[0] = {sparse_indices}[{k_top}]")
    body.append(f"{indent * 5}{dp_accum}[0] = 0.0")
    body.append(
        f"{indent * 5}if {sparse_index}[0] >= 0 and "
        f"{sparse_index}[0] < {sequence_length}:"
    )
    body.append(f"{indent * 6}for {dim} in T.serial(0, {head_dim}):")
    body.append(
        f"{indent * 7}{dp_accum}[0] = {dp_accum}[0] + "
        f"({context_grads}[{dim}] * fp8_e4m3fn_to_float({kv_val_ref}) * {kv_scale_ref})"
    )
    body.append(f"{indent * 5}{dp_values}[{k_top}] = 0.0")
    body.append(f"{indent * 5}if {sumexp}[0] > 0.0:")
    body.append(
        f"{indent * 6}{dp_values}[{k_top}] = {dp_accum}[0]"
    )
    body.append(f"{indent * 4}{dp_accum}[0] = 0.0")
    body.append(f"{indent * 4}for {k_top} in T.serial(0, {topk}):")
    body.append(
        f"{indent * 5}{dp_accum}[0] = {dp_accum}[0] + "
        f"(({score_weights}[{k_top}] / T.max({sumexp}[0], T.float32(1.0e-20))) * "
        f"{dp_values}[{k_top}])"
    )
    body.append(f"{indent * 4}for {dim} in T.serial(0, {head_dim}):")
    body.append(f"{indent * 5}{dq_accum}[0] = 0.0")
    body.append(f"{indent * 5}for {k_top} in T.serial(0, {topk}):")
    body.append(f"{indent * 6}{sparse_index}[0] = {sparse_indices}[{k_top}]")
    body.append(
        f"{indent * 6}if {sparse_index}[0] >= 0 and "
        f"{sparse_index}[0] < {sequence_length}:"
    )
    body.append(f"{indent * 7}{ds_score}[0] = 0.0")
    body.append(f"{indent * 7}if {sumexp}[0] > 0.0:")
    body.append(
        f"{indent * 8}{ds_score}[0] = "
        f"({score_weights}[{k_top}] / {sumexp}[0]) * "
        f"({dp_values}[{k_top}] - {dp_accum}[0])"
    )
    body.append(
        f"{indent * 7}{dq_accum}[0] = {dq_accum}[0] + "
        f"({ds_score}[0] * fp8_e4m3fn_to_float({kv_val_ref}) * {kv_scale_ref})"
    )
    body.append(f"{indent * 7}{dkv_accum}[0] = {sm_scale} * {ds_score}[0] * {q_deq_values}[{dim}]")
    body.append(
        f"{indent * 7}{dkv_accum}[0] = {dkv_accum}[0] + "
        f"(({score_weights}[{k_top}] / T.max({sumexp}[0], T.float32(1.0e-20))) * "
        f"{context_grads}[{dim}])"
    )
    dkv_ref = _row_phased_attention_selected_kv_value_ref(
        kv_fp8_grad,
        access_by_buffer,
        shape_env,
        selected_row_expr=f"{sparse_index}[0]",
        head_expr=kv_head,
        dim_expr=dim,
    )
    body.append(
        f"{indent * 7}T.atomic_add({dkv_ref}, {dkv_accum}[0], "
        "memory_order=\"relaxed\")"
    )
    dq_ref = _row_phased_attention_value_ref(
        q_fp8_grad,
        access_by_buffer,
        shape_env,
        row_expr="row",
        head_expr=q_head,
        dim_expr=dim,
        q_side=True,
    )
    body.append(f"{indent * 5}{dq_ref} = {dq_accum}[0] * {sm_scale}")
    body.append(f"{indent * 3}T.sync_threads()")
    if out_bias_grad is not None:
        body.append(
            f"{indent * 3}for {out_dim} in T.serial(lane, {hidden_size}, "
            f"step={thread_count}):"
        )
        out_bias_ref = _indexed_buffer_ref(out_bias_grad, access_by_buffer, out_dim)
        out_grad = _node_indexed_canonical_or_positional_input_expr(
            node,
            ("attention_out", "out"),
            0,
            dtype_by_buffer,
            access_by_buffer,
            f"row * {hidden_size} + {out_dim}",
        )
        body.append(f"{indent * 4}{out_bias_ref} = {out_bias_ref} + {out_grad}")
        body.append(f"{indent * 3}T.sync_threads()")


def _append_row_phased_attention_qkv_projection_bwd_body(
    body: list[str],
    *,
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    shape_env: PathCModelShapeEnv,
    thread_count: int,
    indent: str,
) -> None:
    q_grad0 = _scratch_name(node, "attention_q_grad0")
    q_grad1 = _scratch_name(node, "attention_q_grad1")
    kv_grad0 = _scratch_name(node, "attention_kv_grad0")
    kv_grad1 = _scratch_name(node, "attention_kv_grad1")
    q_proj0 = _scratch_name(node, "attention_q_projected0")
    q_proj1 = _scratch_name(node, "attention_q_projected1")
    kv_proj0 = _scratch_name(node, "attention_kv_projected0")
    kv_proj1 = _scratch_name(node, "attention_kv_projected1")
    proj_grad0 = _scratch_name(node, "attention_projected_grad0")
    proj_grad1 = _scratch_name(node, "attention_projected_grad1")
    rope_grad_scratch = _scratch_name(node, "attention_rope_grad")
    rope_phase = _scratch_name(node, "attention_rope_phase")
    pair_flat = _scratch_name(node, "pair_flat")
    grad_flat = _scratch_name(node, "grad_flat")
    h = _scratch_name(node, "h")
    rope_d = _scratch_name(node, "rope_d")
    head = _scratch_name(node, "head")
    attention_hidden_values = _scratch_name(node, "attention_hidden_values")
    attention_hidden_sum_sq = _scratch_name(node, "attention_hidden_sum_sq")
    attention_hidden_inv_rms = _scratch_name(node, "attention_hidden_inv_rms")
    hidden_size = int(shape_env.hidden_size)
    q_heads = int(shape_env.attention_num_q_heads)
    kv_heads = int(shape_env.attention_num_kv_heads)
    head_dim = int(shape_env.attention_head_dim)
    rope_half = max(1, head_dim // 2)
    q_dim = q_heads * head_dim
    kv_dim = kv_heads * head_dim
    hidden_grad = _node_output_for_canonical_or_index(
        node,
        ("attention_hidden", "hidden"),
        0,
    )
    q_weight_grad = _node_output_for_canonical(node, "attention_q_proj_weight")
    q_bias_grad = _node_output_for_canonical(node, "attention_q_proj_bias")
    kv_weight_grad = _node_output_for_canonical(
        node,
        "attention_sparse_kv_proj_weight",
    )
    kv_bias_grad = _node_output_for_canonical(
        node,
        "attention_sparse_kv_proj_bias",
    )
    rope_grad = _node_output_for_canonical(node, "attention_rope_inv_freq")
    _append_descriptor_node_comments(
        body,
        node=node,
        descriptor=descriptor,
        indent=indent * 3,
    )
    body.append(
        f"{indent * 3}# attention_qkv_projection_bwd_policy: "
        "exact_inverse_rope_weight_bias_hidden"
    )
    body.append(f"{indent * 3}if row == 0:")
    if q_weight_grad is not None:
        body.append(
            f"{indent * 4}for {grad_flat} in T.serial(lane, "
            f"{q_dim * hidden_size}, step={thread_count}):"
        )
        body.append(
            f"{indent * 5}{_indexed_buffer_ref(q_weight_grad, access_by_buffer, grad_flat)} = 0.0"
        )
    if q_bias_grad is not None:
        body.append(
            f"{indent * 4}for {grad_flat} in T.serial(lane, {q_dim}, "
            f"step={thread_count}):"
        )
        body.append(
            f"{indent * 5}{_indexed_buffer_ref(q_bias_grad, access_by_buffer, grad_flat)} = 0.0"
        )
    if kv_weight_grad is not None:
        body.append(
            f"{indent * 4}for {grad_flat} in T.serial(lane, "
            f"{kv_dim * hidden_size}, step={thread_count}):"
        )
        body.append(
            f"{indent * 5}{_indexed_buffer_ref(kv_weight_grad, access_by_buffer, grad_flat)} = 0.0"
        )
    if kv_bias_grad is not None:
        body.append(
            f"{indent * 4}for {grad_flat} in T.serial(lane, {kv_dim}, "
            f"step={thread_count}):"
        )
        body.append(
            f"{indent * 5}{_indexed_buffer_ref(kv_bias_grad, access_by_buffer, grad_flat)} = 0.0"
        )
    if rope_grad is not None:
        body.append(
            f"{indent * 4}for {rope_d} in T.serial(lane, {rope_half}, "
            f"step={thread_count}):"
        )
        body.append(
            f"{indent * 5}{_indexed_buffer_ref(rope_grad, access_by_buffer, rope_d)} = 0.0"
        )
    body.append(f"{indent * 3}T.sync_threads()")
    if hidden_grad is not None:
        body.append(
            f"{indent * 3}for {h} in T.serial(lane, {hidden_size}, "
            f"step={thread_count}):"
        )
        hidden_grad_ref = _indexed_buffer_ref(
            hidden_grad,
            access_by_buffer,
            f"row * {hidden_size} + {h}",
        )
        body.append(f"{indent * 4}{hidden_grad_ref} = 0.0")
        body.append(f"{indent * 3}T.sync_threads()")
    body.append(
        f"{indent * 3}{q_grad0} = T.alloc_local((1,), \"float32\")"
    )
    body.append(
        f"{indent * 3}{q_grad1} = T.alloc_local((1,), \"float32\")"
    )
    body.append(
        f"{indent * 3}{kv_grad0} = T.alloc_local((1,), \"float32\")"
    )
    body.append(
        f"{indent * 3}{kv_grad1} = T.alloc_local((1,), \"float32\")"
    )
    body.append(
        f"{indent * 3}{q_proj0} = T.alloc_local((1,), \"float32\")"
    )
    body.append(
        f"{indent * 3}{q_proj1} = T.alloc_local((1,), \"float32\")"
    )
    body.append(
        f"{indent * 3}{kv_proj0} = T.alloc_local((1,), \"float32\")"
    )
    body.append(
        f"{indent * 3}{kv_proj1} = T.alloc_local((1,), \"float32\")"
    )
    body.append(
        f"{indent * 3}{proj_grad0} = T.alloc_local((1,), \"float32\")"
    )
    body.append(
        f"{indent * 3}{proj_grad1} = T.alloc_local((1,), \"float32\")"
    )
    body.append(
        f"{indent * 3}{rope_phase} = T.alloc_local((1,), \"float32\")"
    )
    if rope_grad is not None:
        body.append(
            f"{indent * 3}{rope_grad_scratch} = T.alloc_local((1,), \"float32\")"
        )
    body.append(f"{indent * 3}{attention_hidden_values} = T.alloc_local(({hidden_size},), \"float32\")")
    body.append(f"{indent * 3}{attention_hidden_sum_sq} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}{attention_hidden_inv_rms} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}{attention_hidden_sum_sq}[0] = 0.0")
    body.append(f"{indent * 3}for {h} in T.serial(0, {hidden_size}):")
    hidden_after_m2rnn = _node_indexed_canonical_or_positional_input_expr(
        node,
        ("attention_hidden", "hidden"),
        4,
        dtype_by_buffer,
        access_by_buffer,
        f"row * {hidden_size} + {h}",
    )
    body.append(f"{indent * 4}{attention_hidden_values}[{h}] = {hidden_after_m2rnn}")
    body.append(
        f"{indent * 4}{attention_hidden_sum_sq}[0] = {attention_hidden_sum_sq}[0] + "
        f"({attention_hidden_values}[{h}] * {attention_hidden_values}[{h}])"
    )
    body.append(
        f"{indent * 3}{attention_hidden_inv_rms}[0] = T.rsqrt(({attention_hidden_sum_sq}[0] / "
        f"{float(hidden_size):.1f}) + 0.00001)"
    )
    body.append(f"{indent * 3}for {h} in T.serial(0, {hidden_size}):")
    body.append(
        f"{indent * 4}{attention_hidden_values}[{h}] = "
        f"{attention_hidden_values}[{h}]"
    )
    body.append(
        f"{indent * 3}for {pair_flat} in T.serial(lane, {q_heads * rope_half}, "
        f"step={thread_count}):"
    )
    body.append(f"{indent * 4}{head} = {pair_flat} // {rope_half}")
    body.append(f"{indent * 4}{rope_d} = {pair_flat} % {rope_half}")
    q_value_index0 = f"row * {q_dim} + {head} * {head_dim} + {rope_d}"
    q_value_index1 = f"row * {q_dim} + {head} * {head_dim} + {rope_d} + {rope_half}"
    q_fp8_grad0 = _node_indexed_canonical_input_expr(
        node, "q_fp8", dtype_by_buffer, access_by_buffer, q_value_index0
    )
    q_fp8_grad1 = _node_indexed_canonical_input_expr(
        node, "q_fp8", dtype_by_buffer, access_by_buffer, q_value_index1
    )
    q_scale_grad_seen = _node_indexed_canonical_input_expr(
        node,
        "q_scale",
        dtype_by_buffer,
        access_by_buffer,
        f"row * {q_heads} + {head}",
    )
    body.append(f"{indent * 4}{q_grad0}[0] = {q_fp8_grad0}")
    body.append(f"{indent * 4}{q_grad1}[0] = {q_fp8_grad1}")
    body.append(f"{indent * 4}{q_grad0}[0] = {q_grad0}[0] + {q_scale_grad_seen}")
    body.append(f"{indent * 4}{q_grad0}[0] = {q_grad0}[0] - {q_scale_grad_seen}")
    q_bias0 = _node_indexed_canonical_input_expr(
        node,
        "attention_q_proj_bias",
        dtype_by_buffer,
        access_by_buffer,
        f"{head} * {head_dim} + {rope_d}",
    )
    q_bias1 = _node_indexed_canonical_input_expr(
        node,
        "attention_q_proj_bias",
        dtype_by_buffer,
        access_by_buffer,
        f"{head} * {head_dim} + {rope_d} + {rope_half}",
    )
    body.append(f"{indent * 4}{q_proj0}[0] = {q_bias0}")
    body.append(f"{indent * 4}{q_proj1}[0] = {q_bias1}")
    body.append(f"{indent * 4}for {h} in T.serial(0, {hidden_size}):")
    hidden = f"{attention_hidden_values}[{h}]"
    q_weight0 = _node_indexed_canonical_input_expr(
        node,
        "attention_q_proj_weight",
        dtype_by_buffer,
        access_by_buffer,
        f"({head} * {head_dim} + {rope_d}) * {hidden_size} + {h}",
        default="1.0",
    )
    q_weight1 = _node_indexed_canonical_input_expr(
        node,
        "attention_q_proj_weight",
        dtype_by_buffer,
        access_by_buffer,
        f"({head} * {head_dim} + {rope_d} + {rope_half}) * {hidden_size} + {h}",
        default="1.0",
    )
    body.append(
        f"{indent * 5}{q_proj0}[0] = {q_proj0}[0] + ({hidden} * {q_weight0})"
    )
    body.append(
        f"{indent * 5}{q_proj1}[0] = {q_proj1}[0] + ({hidden} * {q_weight1})"
    )
    rope_input = _node_indexed_canonical_input_expr(
        node,
        "attention_rope_inv_freq",
        dtype_by_buffer,
        access_by_buffer,
        rope_d,
    )
    body.append(
        f"{indent * 4}{rope_phase}[0] = T.cast(row, \"float32\") * {rope_input}"
    )
    body.append(
        f"{indent * 4}{proj_grad0}[0] = "
        f"({q_grad0}[0] * T.cos({rope_phase}[0])) - "
        f"({q_grad1}[0] * T.sin({rope_phase}[0]))"
    )
    body.append(
        f"{indent * 4}{proj_grad1}[0] = "
        f"({q_grad0}[0] * T.sin({rope_phase}[0])) + "
        f"({q_grad1}[0] * T.cos({rope_phase}[0]))"
    )
    if q_bias_grad is not None:
        q_bias_ref0 = _indexed_buffer_ref(
            q_bias_grad,
            access_by_buffer,
            f"{head} * {head_dim} + {rope_d}",
        )
        q_bias_ref1 = _indexed_buffer_ref(
            q_bias_grad,
            access_by_buffer,
            f"{head} * {head_dim} + {rope_d} + {rope_half}",
        )
        body.append(f"{indent * 4}{q_bias_ref0} = {q_bias_ref0} + {proj_grad0}[0]")
        body.append(f"{indent * 4}{q_bias_ref1} = {q_bias_ref1} + {proj_grad1}[0]")
    if rope_grad is not None:
        body.append(
            f"{indent * 4}{rope_grad_scratch}[0] = "
            f"({q_grad0}[0] * ((-{q_proj0}[0] * T.sin({rope_phase}[0])) + "
            f"({q_proj1}[0] * T.cos({rope_phase}[0])))) + "
            f"({q_grad1}[0] * ((-{q_proj1}[0] * T.sin({rope_phase}[0])) - "
            f"({q_proj0}[0] * T.cos({rope_phase}[0]))))"
        )
        rope_ref = _indexed_buffer_ref(rope_grad, access_by_buffer, rope_d)
        body.append(
            f"{indent * 4}T.atomic_add({rope_ref}, "
            f"{rope_grad_scratch}[0] * T.cast(row, \"float32\"), "
            "memory_order=\"relaxed\")"
        )
    body.append(f"{indent * 4}for {h} in T.serial(0, {hidden_size}):")
    hidden = f"{attention_hidden_values}[{h}]"
    if q_weight_grad is not None:
        q_weight_ref0 = _indexed_buffer_ref(
            q_weight_grad,
            access_by_buffer,
            f"({head} * {head_dim} + {rope_d}) * {hidden_size} + {h}",
        )
        q_weight_ref1 = _indexed_buffer_ref(
            q_weight_grad,
            access_by_buffer,
            f"({head} * {head_dim} + {rope_d} + {rope_half}) * {hidden_size} + {h}",
        )
        body.append(
            f"{indent * 5}{q_weight_ref0} = {q_weight_ref0} + "
            f"({hidden} * {proj_grad0}[0])"
        )
        body.append(
            f"{indent * 5}{q_weight_ref1} = {q_weight_ref1} + "
            f"({hidden} * {proj_grad1}[0])"
        )
    if hidden_grad is not None:
        hidden_grad_ref = _indexed_buffer_ref(
            hidden_grad,
            access_by_buffer,
            f"row * {hidden_size} + {h}",
        )
        body.append(
            f"{indent * 5}T.atomic_add({hidden_grad_ref}, "
            f"(({proj_grad0}[0] * {q_weight0}) + ({proj_grad1}[0] * {q_weight1})), "
            "memory_order=\"relaxed\")"
        )
    body.append(f"{indent * 3}T.sync_threads()")
    body.append(
        f"{indent * 3}for {pair_flat} in T.serial(lane, {kv_heads * rope_half}, "
        f"step={thread_count}):"
    )
    body.append(f"{indent * 4}{head} = {pair_flat} // {rope_half}")
    body.append(f"{indent * 4}{rope_d} = {pair_flat} % {rope_half}")
    kv_value_index0 = f"row * {kv_dim} + {head} * {head_dim} + {rope_d}"
    kv_value_index1 = f"row * {kv_dim} + {head} * {head_dim} + {rope_d} + {rope_half}"
    kv_fp8_grad0 = _node_indexed_canonical_input_expr(
        node, "kv_fp8", dtype_by_buffer, access_by_buffer, kv_value_index0
    )
    kv_fp8_grad1 = _node_indexed_canonical_input_expr(
        node, "kv_fp8", dtype_by_buffer, access_by_buffer, kv_value_index1
    )
    kv_scale_grad_seen = _node_indexed_canonical_input_expr(
        node,
        "kv_scale",
        dtype_by_buffer,
        access_by_buffer,
        f"row * {kv_heads} + {head}",
    )
    body.append(f"{indent * 4}{kv_grad0}[0] = {kv_fp8_grad0}")
    body.append(f"{indent * 4}{kv_grad1}[0] = {kv_fp8_grad1}")
    body.append(f"{indent * 4}{kv_grad0}[0] = {kv_grad0}[0] + {kv_scale_grad_seen}")
    body.append(f"{indent * 4}{kv_grad0}[0] = {kv_grad0}[0] - {kv_scale_grad_seen}")
    kv_bias0 = _node_indexed_canonical_input_expr(
        node,
        "attention_sparse_kv_proj_bias",
        dtype_by_buffer,
        access_by_buffer,
        f"{head} * {head_dim} + {rope_d}",
    )
    kv_bias1 = _node_indexed_canonical_input_expr(
        node,
        "attention_sparse_kv_proj_bias",
        dtype_by_buffer,
        access_by_buffer,
        f"{head} * {head_dim} + {rope_d} + {rope_half}",
    )
    body.append(f"{indent * 4}{kv_proj0}[0] = {kv_bias0}")
    body.append(f"{indent * 4}{kv_proj1}[0] = {kv_bias1}")
    body.append(f"{indent * 4}for {h} in T.serial(0, {hidden_size}):")
    hidden = f"{attention_hidden_values}[{h}]"
    kv_weight0 = _node_indexed_canonical_input_expr(
        node,
        "attention_sparse_kv_proj_weight",
        dtype_by_buffer,
        access_by_buffer,
        f"({head} * {head_dim} + {rope_d}) * {hidden_size} + {h}",
        default="1.0",
    )
    kv_weight1 = _node_indexed_canonical_input_expr(
        node,
        "attention_sparse_kv_proj_weight",
        dtype_by_buffer,
        access_by_buffer,
        f"({head} * {head_dim} + {rope_d} + {rope_half}) * {hidden_size} + {h}",
        default="1.0",
    )
    body.append(
        f"{indent * 5}{kv_proj0}[0] = {kv_proj0}[0] + ({hidden} * {kv_weight0})"
    )
    body.append(
        f"{indent * 5}{kv_proj1}[0] = {kv_proj1}[0] + ({hidden} * {kv_weight1})"
    )
    rope_input = _node_indexed_canonical_input_expr(
        node,
        "attention_rope_inv_freq",
        dtype_by_buffer,
        access_by_buffer,
        rope_d,
    )
    body.append(
        f"{indent * 4}{rope_phase}[0] = T.cast(row, \"float32\") * {rope_input}"
    )
    body.append(
        f"{indent * 4}{proj_grad0}[0] = "
        f"({kv_grad0}[0] * T.cos({rope_phase}[0])) - "
        f"({kv_grad1}[0] * T.sin({rope_phase}[0]))"
    )
    body.append(
        f"{indent * 4}{proj_grad1}[0] = "
        f"({kv_grad0}[0] * T.sin({rope_phase}[0])) + "
        f"({kv_grad1}[0] * T.cos({rope_phase}[0]))"
    )
    if kv_bias_grad is not None:
        kv_bias_ref0 = _indexed_buffer_ref(
            kv_bias_grad,
            access_by_buffer,
            f"{head} * {head_dim} + {rope_d}",
        )
        kv_bias_ref1 = _indexed_buffer_ref(
            kv_bias_grad,
            access_by_buffer,
            f"{head} * {head_dim} + {rope_d} + {rope_half}",
        )
        body.append(f"{indent * 4}{kv_bias_ref0} = {kv_bias_ref0} + {proj_grad0}[0]")
        body.append(f"{indent * 4}{kv_bias_ref1} = {kv_bias_ref1} + {proj_grad1}[0]")
    if rope_grad is not None:
        body.append(
            f"{indent * 4}{rope_grad_scratch}[0] = "
            f"({kv_grad0}[0] * ((-{kv_proj0}[0] * T.sin({rope_phase}[0])) + "
            f"({kv_proj1}[0] * T.cos({rope_phase}[0])))) + "
            f"({kv_grad1}[0] * ((-{kv_proj1}[0] * T.sin({rope_phase}[0])) - "
            f"({kv_proj0}[0] * T.cos({rope_phase}[0]))))"
        )
        rope_ref = _indexed_buffer_ref(rope_grad, access_by_buffer, rope_d)
        body.append(
            f"{indent * 4}T.atomic_add({rope_ref}, "
            f"{rope_grad_scratch}[0] * T.cast(row, \"float32\"), "
            "memory_order=\"relaxed\")"
        )
    body.append(f"{indent * 4}for {h} in T.serial(0, {hidden_size}):")
    hidden = f"{attention_hidden_values}[{h}]"
    if kv_weight_grad is not None:
        kv_weight_ref0 = _indexed_buffer_ref(
            kv_weight_grad,
            access_by_buffer,
            f"({head} * {head_dim} + {rope_d}) * {hidden_size} + {h}",
        )
        kv_weight_ref1 = _indexed_buffer_ref(
            kv_weight_grad,
            access_by_buffer,
            f"({head} * {head_dim} + {rope_d} + {rope_half}) * {hidden_size} + {h}",
        )
        body.append(
            f"{indent * 5}{kv_weight_ref0} = {kv_weight_ref0} + "
            f"({hidden} * {proj_grad0}[0])"
        )
        body.append(
            f"{indent * 5}{kv_weight_ref1} = {kv_weight_ref1} + "
            f"({hidden} * {proj_grad1}[0])"
        )
    if hidden_grad is not None:
        hidden_grad_ref = _indexed_buffer_ref(
            hidden_grad,
            access_by_buffer,
            f"row * {hidden_size} + {h}",
        )
        body.append(
            f"{indent * 5}T.atomic_add({hidden_grad_ref}, "
            f"(({proj_grad0}[0] * {kv_weight0}) + ({proj_grad1}[0] * {kv_weight1})), "
            "memory_order=\"relaxed\")"
        )
    body.append(f"{indent * 3}T.sync_threads()")

def _append_row_phased_m2rnn_bwd_body(
    body: list[str],
    *,
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    shape_env: PathCModelShapeEnv,
    thread_count: int,
    indent: str,
    launcher_chunked_rows: bool = False,
) -> None:
    stage_grad = _scratch_name(node, "m2rnn_stage_grad")
    project_grad = _scratch_name(node, "m2rnn_project_grad")
    conv_grad = _scratch_name(node, "m2rnn_conv_grad")
    conv_pre = _scratch_name(node, "m2rnn_conv_pre")
    conv = _scratch_name(node, "m2rnn_conv_vec")
    projected = _scratch_name(node, "m2rnn_projected_vec")
    post = _scratch_name(node, "m2rnn_post_vec")
    post_grad = _scratch_name(node, "m2rnn_post_grad")
    h_prev = _scratch_name(node, "m2rnn_h_prev")
    h_next = _scratch_name(node, "m2rnn_h_next")
    dh = _scratch_name(node, "m2rnn_state_grad")
    dh_prev = _scratch_name(node, "m2rnn_state_prev_grad")
    sum_sq = _scratch_name(node, "m2rnn_sum_sq")
    dot = _scratch_name(node, "m2rnn_norm_dot")
    accum = _scratch_name(node, "m2rnn_accum")
    decay = _scratch_name(node, "m2rnn_decay")
    tanh_val = _scratch_name(node, "m2rnn_tanh")
    scalar0 = _scratch_name(node, "m2rnn_scalar0")
    scalar1 = _scratch_name(node, "m2rnn_scalar1")
    scalar2 = _scratch_name(node, "m2rnn_scalar2")
    time_rev = _scratch_name(node, "time_rev")
    time_idx = _scratch_name(node, "time_idx")
    replay_time = _scratch_name(node, "replay_time")
    replay_offset = _scratch_name(node, "replay_offset")
    checkpoint_idx = _scratch_name(node, "checkpoint_idx")
    checkpoint_start = _scratch_name(node, "checkpoint_start")
    src_row = _scratch_name(node, "src_row")
    src_hist = _scratch_name(node, "src_hist")
    hidden_loop = _scratch_name(node, "hidden_loop")
    proj_dim = _scratch_name(node, "proj_dim")
    hidden_dim = _scratch_name(node, "hidden_dim")
    conv_ch = _scratch_name(node, "conv_ch")
    kernel_pos = _scratch_name(node, "kernel_pos")
    state_idx = _scratch_name(node, "state_idx")
    grad_flat = _scratch_name(node, "grad_flat")
    feature = _scratch_name(node, "feature")
    head = _scratch_name(node, "head")
    kk = _scratch_name(node, "kk")
    vv = _scratch_name(node, "vv")
    vv_inner = _scratch_name(node, "vv_inner")
    out_dim = _scratch_name(node, "out_dim")
    partial = _scratch_name(node, "partial")
    hidden_size = int(shape_env.hidden_size)
    sequence_length = int(shape_env.sequence_length)
    in_proj_dim = int(shape_env.m2rnn_in_proj_dim)
    conv_dim = int(shape_env.m2rnn_conv_dim)
    kernel = int(shape_env.m2rnn_conv_kernel)
    history_len = max(0, kernel - 1)
    total_heads = int(shape_env.m2rnn_num_heads)
    q_heads = int(shape_env.m2rnn_num_q_heads)
    k_heads = int(shape_env.m2rnn_num_k_heads)
    v_heads = int(shape_env.m2rnn_num_v_heads)
    f_heads = int(shape_env.m2rnn_num_f_heads)
    g_heads = int(shape_env.m2rnn_num_g_heads)
    w_heads = int(shape_env.m2rnn_num_weight_heads)
    k_dim = int(shape_env.m2rnn_k_head_dim)
    v_dim = int(shape_env.m2rnn_v_head_dim)
    state_extent = (
        int(shape_env.m2rnn_num_weight_heads)
        * int(shape_env.m2rnn_v_head_dim)
        * int(shape_env.m2rnn_v_head_dim)
    )
    full_state_extent = total_heads * k_dim * v_dim
    features = total_heads * v_dim
    q_offset = 0
    k_offset = int(shape_env.m2rnn_q_dim)
    v_offset = k_offset + int(shape_env.m2rnn_k_dim)
    f_offset = conv_dim
    g_offset = conv_dim + f_heads
    q_group = total_heads // q_heads
    k_group = total_heads // k_heads
    v_group = total_heads // v_heads
    f_group = total_heads // f_heads
    g_repeat = total_heads // g_heads
    w_group = total_heads // w_heads
    hidden_grad = _node_output_for_canonical_or_index(
        node,
        ("m2rnn_hidden", "hidden"),
        0,
    )
    in_proj_weight_grad = _node_output_for_canonical(node, "m2rnn_in_proj_weight")
    conv_weight_grad = _node_output_for_canonical(node, "m2rnn_conv_weight")
    conv_bias_grad = _node_output_for_canonical(node, "m2rnn_conv_bias")
    state_weight_grad = _node_output_for_canonical(node, "m2rnn_state_weight")
    a_log_grad = _node_output_for_canonical(node, "m2rnn_A_log")
    dt_bias_grad = _node_output_for_canonical(node, "m2rnn_dt_bias")
    d_grad = _node_output_for_canonical(node, "m2rnn_D")
    g_norm_weight_grad = _node_output_for_canonical(node, "m2rnn_g_norm_weight")
    out_proj_weight_grad = _node_output_for_canonical(node, "m2rnn_out_proj_weight")
    h0_grad = _node_output_for_canonical(node, "m2rnn_h0")
    _append_descriptor_node_comments(
        body,
        node=node,
        descriptor=descriptor,
        indent=indent * 3,
    )
    checkpoint_interval = M2RNN_BWD_REPLAY_CHECKPOINT_INTERVAL
    body.append(f"{indent * 3}# m2rnn_bwd_policy: exact_reverse_checkpoint_replay")
    body.append(f"{indent * 3}{stage_grad} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}{project_grad} = T.alloc_local(({in_proj_dim},), \"float32\")")
    body.append(f"{indent * 3}{conv_grad} = T.alloc_local(({conv_dim},), \"float32\")")
    body.append(f"{indent * 3}{conv_pre} = T.alloc_local(({conv_dim},), \"float32\")")
    body.append(f"{indent * 3}{conv} = T.alloc_local(({conv_dim},), \"float32\")")
    body.append(f"{indent * 3}{projected} = T.alloc_local(({in_proj_dim},), \"float32\")")
    body.append(f"{indent * 3}{post} = T.alloc_local(({features},), \"float32\")")
    body.append(f"{indent * 3}{post_grad} = T.alloc_local(({features},), \"float32\")")
    body.append(f"{indent * 3}{h_prev} = T.alloc_local(({full_state_extent},), \"float32\")")
    body.append(f"{indent * 3}{h_next} = T.alloc_local(({full_state_extent},), \"float32\")")
    body.append(f"{indent * 3}{dh} = T.alloc_local(({full_state_extent},), \"float32\")")
    body.append(f"{indent * 3}{dh_prev} = T.alloc_local(({full_state_extent},), \"float32\")")
    body.append(f"{indent * 3}{sum_sq} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}{dot} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}{accum} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}{decay} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}{tanh_val} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}{scalar0} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}{scalar1} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}{scalar2} = T.alloc_local((1,), \"float32\")")
    first_row_condition = (
        "path_c_first_row_launch != 0 and row == row_chunk_start"
        if launcher_chunked_rows
        else "row == 0"
    )
    time_rev_range = "row, row + 1" if launcher_chunked_rows else f"0, {sequence_length}"
    body.append(f"{indent * 3}if {first_row_condition}:")
    body.append(f"{indent * 4}if lane == 0:")
    for output_name, extent in (
        (in_proj_weight_grad, in_proj_dim * hidden_size),
        (conv_weight_grad, conv_dim * kernel),
        (conv_bias_grad, conv_dim),
        (state_weight_grad, state_extent),
        (a_log_grad, int(shape_env.m2rnn_num_heads)),
        (dt_bias_grad, int(shape_env.m2rnn_num_heads)),
        (d_grad, features),
        (g_norm_weight_grad, features),
        (out_proj_weight_grad, hidden_size * features),
        (h0_grad, full_state_extent),
    ):
        if output_name is None:
            continue
        body.append(
            f"{indent * 5}for {state_idx} in T.serial(0, {extent}):"
        )
        body.append(
            f"{indent * 6}{_indexed_buffer_ref(output_name, access_by_buffer, state_idx)} = 0.0"
        )
    body.append(f"{indent * 5}for {state_idx} in T.serial(0, {full_state_extent}):")
    body.append(f"{indent * 6}{dh}[{state_idx}] = 0.0")
    if hidden_grad is not None:
        body.append(
            f"{indent * 5}for {grad_flat} in T.serial(0, "
            f"{sequence_length * hidden_size}):"
        )
        body.append(
            f"{indent * 6}{_indexed_buffer_ref(hidden_grad, access_by_buffer, grad_flat)} = 0.0"
        )
    body.append(f"{indent * 3}if lane == 0:")
    body.append(f"{indent * 4}for {time_rev} in T.serial({time_rev_range}):")
    body.append(f"{indent * 6}{time_idx} = {sequence_length - 1} - {time_rev}")
    body.append(f"{indent * 6}{checkpoint_idx} = {time_idx} // {checkpoint_interval}")
    body.append(f"{indent * 6}{checkpoint_start} = {checkpoint_idx} * {checkpoint_interval}")
    body.append(f"{indent * 6}for {state_idx} in T.serial(0, {full_state_extent}):")
    body.append(f"{indent * 7}{head} = {state_idx} // {k_dim * v_dim}")
    body.append(f"{indent * 7}{kk} = ({state_idx} // {v_dim}) % {k_dim}")
    body.append(f"{indent * 7}{vv} = {state_idx} % {v_dim}")
    h_checkpoint_expr = _buffer_ref(
        "m2rnn_h_checkpoint",
        access_by_buffer,
        f"{checkpoint_idx} * {full_state_extent} + {head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv}",
    )
    body.append(f"{indent * 7}{h_prev}[{state_idx}] = {h_checkpoint_expr}")
    body.append(f"{indent * 7}{h_next}[{state_idx}] = {h_checkpoint_expr}")
    if h0_grad is not None:
        h0_grad_ref = _indexed_buffer_ref(
            h0_grad,
            access_by_buffer,
            f"{head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv}",
        )
        body.append(f"{indent * 7}{dh}[{state_idx}] = {h0_grad_ref}")
    body.append(f"{indent * 6}for {replay_offset} in T.serial(0, {checkpoint_interval}):")
    body.append(f"{indent * 7}{replay_time} = {checkpoint_start} + {replay_offset}")
    body.append(f"{indent * 7}if {replay_time} <= {time_idx}:")
    body.append(f"{indent * 8}for {proj_dim} in T.serial(0, {in_proj_dim}):")
    body.append(f"{indent * 9}{accum}[0] = 0.0")
    body.append(f"{indent * 9}for {hidden_loop} in T.serial(0, {hidden_size}):")
    hidden_expr = _node_indexed_canonical_or_positional_input_expr(
        node,
        ("m2rnn_hidden", "hidden"),
        1,
        dtype_by_buffer,
        access_by_buffer,
        f"{replay_time} * {hidden_size} + {hidden_loop}",
    )
    weight_expr = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_in_proj_weight",
        dtype_by_buffer,
        access_by_buffer,
        f"{proj_dim} * {hidden_size} + {hidden_loop}",
        default="1.0",
    )
    body.append(
        f"{indent * 10}{accum}[0] = {accum}[0] + "
        f"({hidden_expr} * {weight_expr})"
    )
    body.append(f"{indent * 9}{projected}[{proj_dim}] = {accum}[0]")
    body.append(f"{indent * 8}for {conv_ch} in T.serial(0, {conv_dim}):")
    conv_bias_expr = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_conv_bias",
        dtype_by_buffer,
        access_by_buffer,
        conv_ch,
        default="0.0",
    )
    body.append(f"{indent * 9}{conv_pre}[{conv_ch}] = {conv_bias_expr}")
    if history_len > 0:
        body.append(f"{indent * 9}for {kernel_pos} in T.serial(0, {history_len}):")
        body.append(
            f"{indent * 10}{src_row} = {replay_time} - {history_len} + {kernel_pos}"
        )
        body.append(f"{indent * 10}if {src_row} >= 0:")
        body.append(f"{indent * 11}{scalar0}[0] = 0.0")
        body.append(f"{indent * 11}for {hidden_loop} in T.serial(0, {hidden_size}):")
        src_hidden_expr = _node_indexed_canonical_or_positional_input_expr(
            node,
            ("m2rnn_hidden", "hidden"),
            1,
            dtype_by_buffer,
            access_by_buffer,
            f"{src_row} * {hidden_size} + {hidden_loop}",
        )
        src_weight_expr = _node_indexed_canonical_input_expr(
            node,
            "m2rnn_in_proj_weight",
            dtype_by_buffer,
            access_by_buffer,
            f"{conv_ch} * {hidden_size} + {hidden_loop}",
            default="1.0",
        )
        body.append(
            f"{indent * 12}{scalar0}[0] = {scalar0}[0] + "
            f"({src_hidden_expr} * {src_weight_expr})"
        )
        body.append(f"{indent * 10}else:")
        body.append(f"{indent * 11}{src_hist} = {kernel_pos} + {replay_time}")
        conv_state_expr = _node_indexed_canonical_input_expr(
            node,
            "m2rnn_conv_state",
            dtype_by_buffer,
            access_by_buffer,
            f"{src_hist} * {conv_dim} + {conv_ch}",
            default="0.0",
        )
        body.append(f"{indent * 11}{scalar0}[0] = {conv_state_expr}")
        conv_weight_expr = _node_indexed_canonical_input_expr(
            node,
            "m2rnn_conv_weight",
            dtype_by_buffer,
            access_by_buffer,
            f"{conv_ch} * {kernel} + {kernel_pos}",
            default="1.0",
        )
        body.append(
            f"{indent * 10}{conv_pre}[{conv_ch}] = {conv_pre}[{conv_ch}] + "
            f"({scalar0}[0] * {conv_weight_expr})"
        )
    current_conv_weight_expr = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_conv_weight",
        dtype_by_buffer,
        access_by_buffer,
        f"{conv_ch} * {kernel} + {history_len}",
        default="1.0",
    )
    body.append(
        f"{indent * 9}{conv_pre}[{conv_ch}] = {conv_pre}[{conv_ch}] + "
        f"({projected}[{conv_ch}] * {current_conv_weight_expr})"
    )
    body.append(f"{indent * 9}{scalar0}[0] = 1.0 / (1.0 + T.exp(-{conv_pre}[{conv_ch}]))")
    body.append(f"{indent * 9}{conv}[{conv_ch}] = {conv_pre}[{conv_ch}] * {scalar0}[0]")
    body.append(f"{indent * 8}for {state_idx} in T.serial(0, {full_state_extent}):")
    body.append(f"{indent * 9}{head} = {state_idx} // {k_dim * v_dim}")
    body.append(f"{indent * 9}{kk} = ({state_idx} // {v_dim}) % {k_dim}")
    body.append(f"{indent * 9}{vv} = {state_idx} % {v_dim}")
    body.append(f"{indent * 9}{accum}[0] = 0.0")
    body.append(f"{indent * 9}for {vv_inner} in T.serial(0, {v_dim}):")
    state_weight_expr = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_state_weight",
        dtype_by_buffer,
        access_by_buffer,
        f"({head} // {w_group}) * {v_dim * v_dim} + {vv_inner} * {v_dim} + {vv}",
        default="1.0",
    )
    body.append(
        f"{indent * 10}{accum}[0] = {accum}[0] + "
        f"({h_prev}[{head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv_inner}] * "
        f"{state_weight_expr})"
    )
    a_log_expr = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_A_log",
        dtype_by_buffer,
        access_by_buffer,
        head,
        default="0.0",
    )
    dt_bias_expr = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_dt_bias",
        dtype_by_buffer,
        access_by_buffer,
        head,
        default="0.0",
    )
    f_src = f"({head} // {f_group})"
    body.append(
        f"{indent * 9}{decay}[0] = T.exp(-T.exp({a_log_expr}) * "
        f"T.log(1.0 + T.exp({projected}[{f_offset} + {f_src}] + {dt_bias_expr})))"
    )
    k_val = f"{conv}[{k_offset} + (({head} // {k_group}) * {k_dim}) + {kk}]"
    v_val = f"{conv}[{v_offset} + (({head} // {v_group}) * {v_dim}) + {vv}]"
    body.append(f"{indent * 9}{tanh_val}[0] = T.tanh({accum}[0] + ({k_val} * {v_val}))")
    body.append(
        f"{indent * 9}{h_next}[{state_idx}] = "
        f"({decay}[0] * {h_prev}[{state_idx}]) + "
        f"((1.0 - {decay}[0]) * {tanh_val}[0])"
    )
    body.append(f"{indent * 8}if {replay_time} < {time_idx}:")
    body.append(f"{indent * 9}for {state_idx} in T.serial(0, {full_state_extent}):")
    body.append(f"{indent * 10}{h_prev}[{state_idx}] = {h_next}[{state_idx}]")
    body.append(f"{indent * 6}for {feature} in T.serial(0, {features}):")
    body.append(f"{indent * 7}{head} = {feature} // {v_dim}")
    body.append(f"{indent * 7}{vv} = {feature} % {v_dim}")
    body.append(f"{indent * 7}{post}[{feature}] = 0.0")
    body.append(f"{indent * 7}for {kk} in T.serial(0, {k_dim}):")
    body.append(
        f"{indent * 8}{post}[{feature}] = {post}[{feature}] + "
        f"({conv}[{q_offset} + (({head} // {q_group}) * {k_dim}) + {kk}] * "
        f"{h_next}[{head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv}])"
    )
    d_expr = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_D",
        dtype_by_buffer,
        access_by_buffer,
        feature,
        default="0.0",
    )
    body.append(
        f"{indent * 7}{post}[{feature}] = "
        f"({post}[{feature}] + "
        f"({conv}[{v_offset} + (({head} // {v_group}) * {v_dim}) + {vv}] * {d_expr}))"
    )
    g_flat = f"({feature} // {g_repeat})"
    body.append(f"{indent * 7}{scalar0}[0] = {projected}[{g_offset} + {g_flat}]")
    body.append(f"{indent * 7}{scalar1}[0] = 1.0 / (1.0 + T.exp(-{scalar0}[0]))")
    body.append(f"{indent * 7}{post}[{feature}] = {post}[{feature}] * {scalar0}[0] * {scalar1}[0]")
    body.append(f"{indent * 6}{sum_sq}[0] = 0.0")
    body.append(f"{indent * 6}for {feature} in T.serial(0, {features}):")
    body.append(f"{indent * 7}{sum_sq}[0] = {sum_sq}[0] + ({post}[{feature}] * {post}[{feature}])")
    body.append(f"{indent * 6}{scalar2}[0] = T.rsqrt(({sum_sq}[0] / {float(features):.1f}) + 0.00001)")
    body.append(f"{indent * 6}{dot}[0] = 0.0")
    body.append(f"{indent * 6}for {feature} in T.serial(0, {features}):")
    body.append(f"{indent * 7}{post_grad}[{feature}] = 0.0")
    body.append(f"{indent * 7}for {out_dim} in T.serial(0, {hidden_size}):")
    delta_grad_expr = _node_indexed_canonical_or_positional_input_expr(
        node,
        ("m2rnn_delta",),
        0,
        dtype_by_buffer,
        access_by_buffer,
        f"{time_idx} * {hidden_size} + {out_dim}",
    )
    out_proj_weight_expr = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_out_proj_weight",
        dtype_by_buffer,
        access_by_buffer,
        f"{out_dim} * {features} + {feature}",
        default="1.0",
    )
    body.append(f"{indent * 8}{stage_grad}[0] = {delta_grad_expr}")
    body.append(f"{indent * 8}{post_grad}[{feature}] = {post_grad}[{feature}] + ({stage_grad}[0] * {out_proj_weight_expr})")
    if out_proj_weight_grad is not None:
        out_proj_grad_ref = _indexed_buffer_ref(
            out_proj_weight_grad,
            access_by_buffer,
            f"{out_dim} * {features} + {feature}",
        )
        g_norm_weight_expr = _node_indexed_canonical_input_expr(
            node,
            "m2rnn_g_norm_weight",
            dtype_by_buffer,
            access_by_buffer,
            feature,
            default="1.0",
        )
        body.append(
            f"{indent * 8}{out_proj_grad_ref} = {out_proj_grad_ref} + "
            f"({stage_grad}[0] * {post}[{feature}] * {scalar2}[0] * {g_norm_weight_expr})"
        )
    g_norm_weight_expr = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_g_norm_weight",
        dtype_by_buffer,
        access_by_buffer,
        feature,
        default="1.0",
    )
    body.append(
        f"{indent * 7}{dot}[0] = {dot}[0] + "
        f"({post_grad}[{feature}] * {g_norm_weight_expr} * {post}[{feature}])"
    )
    body.append(f"{indent * 6}for {feature} in T.serial(0, {features}):")
    g_norm_weight_expr = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_g_norm_weight",
        dtype_by_buffer,
        access_by_buffer,
        feature,
        default="1.0",
    )
    if g_norm_weight_grad is not None:
        g_norm_grad_ref = _indexed_buffer_ref(g_norm_weight_grad, access_by_buffer, feature)
        body.append(
            f"{indent * 7}{g_norm_grad_ref} = {g_norm_grad_ref} + "
            f"({post_grad}[{feature}] * {post}[{feature}] * {scalar2}[0])"
        )
    body.append(
        f"{indent * 7}{post_grad}[{feature}] = {scalar2}[0] * "
        f"(({post_grad}[{feature}] * {g_norm_weight_expr}) - "
        f"({post}[{feature}] * {dot}[0] * {scalar2}[0] * {scalar2}[0] / {float(features):.1f}))"
    )
    body.append(f"{indent * 6}for {conv_ch} in T.serial(0, {conv_dim}):")
    body.append(f"{indent * 7}{conv_grad}[{conv_ch}] = 0.0")
    body.append(f"{indent * 6}T.sync_threads()")
    body.append(f"{indent * 6}for {proj_dim} in T.serial(0, {in_proj_dim}):")
    body.append(f"{indent * 7}{project_grad}[{proj_dim}] = 0.0")
    body.append(f"{indent * 6}for {feature} in T.serial(0, {features}):")
    body.append(f"{indent * 7}{head} = {feature} // {v_dim}")
    body.append(f"{indent * 7}{vv} = {feature} % {v_dim}")
    body.append(f"{indent * 7}{accum}[0] = 0.0")
    body.append(f"{indent * 7}for {kk} in T.serial(0, {k_dim}):")
    body.append(
        f"{indent * 8}{accum}[0] = {accum}[0] + "
        f"({conv}[{q_offset} + (({head} // {q_group}) * {k_dim}) + {kk}] * "
        f"{h_next}[{head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv}])"
    )
    d_expr = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_D",
        dtype_by_buffer,
        access_by_buffer,
        feature,
        default="0.0",
    )
    body.append(
        f"{indent * 7}{accum}[0] = {accum}[0] + "
        f"({conv}[{v_offset} + (({head} // {v_group}) * {v_dim}) + {vv}] * {d_expr})"
    )
    g_flat = f"({feature} // {g_repeat})"
    body.append(f"{indent * 7}{scalar0}[0] = {projected}[{g_offset} + {g_flat}]")
    body.append(f"{indent * 7}{scalar1}[0] = 1.0 / (1.0 + T.exp(-{scalar0}[0]))")
    body.append(f"{indent * 7}{scalar2}[0] = {post_grad}[{feature}] * {scalar0}[0] * {scalar1}[0]")
    body.append(
        f"{indent * 7}{project_grad}[{g_offset} + {g_flat}] = "
        f"{project_grad}[{g_offset} + {g_flat}] + "
        f"({post_grad}[{feature}] * {accum}[0] * {scalar1}[0] * "
        f"(1.0 + ({scalar0}[0] * (1.0 - {scalar1}[0]))))"
    )
    if d_grad is not None:
        d_grad_ref = _indexed_buffer_ref(d_grad, access_by_buffer, feature)
        body.append(
            f"{indent * 7}{d_grad_ref} = {d_grad_ref} + "
            f"({scalar2}[0] * {conv}[{v_offset} + (({head} // {v_group}) * {v_dim}) + {vv}])"
        )
    body.append(
        f"{indent * 7}{conv_grad}[{v_offset} + (({head} // {v_group}) * {v_dim}) + {vv}] = "
        f"{conv_grad}[{v_offset} + (({head} // {v_group}) * {v_dim}) + {vv}] + "
        f"({scalar2}[0] * {d_expr})"
    )
    body.append(f"{indent * 7}for {kk} in T.serial(0, {k_dim}):")
    body.append(
        f"{indent * 8}{conv_grad}[{q_offset} + (({head} // {q_group}) * {k_dim}) + {kk}] = "
        f"{conv_grad}[{q_offset} + (({head} // {q_group}) * {k_dim}) + {kk}] + "
        f"({scalar2}[0] * {h_next}[{head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv}])"
    )
    body.append(
        f"{indent * 8}{dh}[{head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv}] = "
        f"{dh}[{head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv}] + "
        f"({scalar2}[0] * {conv}[{q_offset} + (({head} // {q_group}) * {k_dim}) + {kk}])"
    )
    body.append(f"{indent * 6}for {state_idx} in T.serial(0, {full_state_extent}):")
    body.append(f"{indent * 7}{dh_prev}[{state_idx}] = 0.0")
    body.append(f"{indent * 6}for {head} in T.serial(0, {total_heads}):")
    body.append(f"{indent * 7}{scalar0}[0] = 0.0")
    a_log_expr_head = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_A_log",
        dtype_by_buffer,
        access_by_buffer,
        head,
        default="0.0",
    )
    dt_bias_expr_head = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_dt_bias",
        dtype_by_buffer,
        access_by_buffer,
        head,
        default="0.0",
    )
    body.append(
        f"{indent * 7}{scalar1}[0] = T.log(1.0 + "
        f"T.exp({projected}[{f_offset} + ({head} // {f_group})] + {dt_bias_expr_head}))"
    )
    body.append(f"{indent * 7}{scalar2}[0] = T.exp({a_log_expr_head})")
    body.append(
        f"{indent * 7}{decay}[0] = T.exp(-{scalar2}[0] * {scalar1}[0])"
    )
    body.append(f"{indent * 7}for {kk} in T.serial(0, {k_dim}):")
    body.append(f"{indent * 8}for {vv} in T.serial(0, {v_dim}):")
    body.append(f"{indent * 9}{accum}[0] = 0.0")
    body.append(f"{indent * 9}for {vv_inner} in T.serial(0, {v_dim}):")
    state_weight_expr_inner = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_state_weight",
        dtype_by_buffer,
        access_by_buffer,
        f"({head} // {w_group}) * {v_dim * v_dim} + {vv_inner} * {v_dim} + {vv}",
        default="1.0",
    )
    body.append(
        f"{indent * 10}{accum}[0] = {accum}[0] + "
        f"({h_prev}[{head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv_inner}] * "
        f"{state_weight_expr_inner})"
    )
    body.append(f"{indent * 9}{tanh_val}[0] = T.tanh({accum}[0] + ({k_val} * {v_val}))")
    body.append(
        f"{indent * 9}{scalar0}[0] = {scalar0}[0] + "
        f"({dh}[{head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv}] * "
        f"({h_prev}[{head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv}] - {tanh_val}[0]))"
    )
    body.append(
        f"{indent * 9}{scalar2}[0] = "
        f"{dh}[{head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv}] * "
        f"(1.0 - {decay}[0]) * (1.0 - ({tanh_val}[0] * {tanh_val}[0]))"
    )
    body.append(
        f"{indent * 9}{dh_prev}[{head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv}] = "
        f"{dh_prev}[{head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv}] + "
        f"({dh}[{head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv}] * {decay}[0])"
    )
    body.append(f"{indent * 9}for {vv_inner} in T.serial(0, {v_dim}):")
    state_weight_expr_rev = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_state_weight",
        dtype_by_buffer,
        access_by_buffer,
        f"({head} // {w_group}) * {v_dim * v_dim} + {vv_inner} * {v_dim} + {vv}",
        default="1.0",
    )
    body.append(
        f"{indent * 10}{dh_prev}[{head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv_inner}] = "
        f"{dh_prev}[{head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv_inner}] + "
        f"({scalar2}[0] * {state_weight_expr_rev})"
    )
    if state_weight_grad is not None:
        state_weight_grad_ref = _indexed_buffer_ref(
            state_weight_grad,
            access_by_buffer,
            f"({head} // {w_group}) * {v_dim * v_dim} + {vv_inner} * {v_dim} + {vv}",
        )
        body.append(
            f"{indent * 10}{state_weight_grad_ref} = {state_weight_grad_ref} + "
            f"({h_prev}[{head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv_inner}] * {scalar2}[0])"
        )
    body.append(
        f"{indent * 9}{conv_grad}[{k_offset} + (({head} // {k_group}) * {k_dim}) + {kk}] = "
        f"{conv_grad}[{k_offset} + (({head} // {k_group}) * {k_dim}) + {kk}] + "
        f"({scalar2}[0] * {conv}[{v_offset} + (({head} // {v_group}) * {v_dim}) + {vv}])"
    )
    body.append(
        f"{indent * 9}{conv_grad}[{v_offset} + (({head} // {v_group}) * {v_dim}) + {vv}] = "
        f"{conv_grad}[{v_offset} + (({head} // {v_group}) * {v_dim}) + {vv}] + "
        f"({scalar2}[0] * {conv}[{k_offset} + (({head} // {k_group}) * {k_dim}) + {kk}])"
    )
    body.append(f"{indent * 7}{scalar0}[0] = {scalar0}[0] * {decay}[0]")
    if a_log_grad is not None:
        a_log_grad_ref = _indexed_buffer_ref(a_log_grad, access_by_buffer, head)
        body.append(
            f"{indent * 7}{a_log_grad_ref} = {a_log_grad_ref} + "
            f"({scalar0}[0] * (-T.exp({a_log_expr_head}) * {scalar1}[0]))"
        )
    body.append(
        f"{indent * 7}{scalar0}[0] = {scalar0}[0] * "
        f"(-T.exp({a_log_expr_head})) * "
        f"(1.0 / (1.0 + T.exp(-({projected}[{f_offset} + ({head} // {f_group})] + {dt_bias_expr_head}))))"
    )
    body.append(
        f"{indent * 7}{project_grad}[{f_offset} + ({head} // {f_group})] = "
        f"{project_grad}[{f_offset} + ({head} // {f_group})] + {scalar0}[0]"
    )
    if dt_bias_grad is not None:
        dt_bias_grad_ref = _indexed_buffer_ref(dt_bias_grad, access_by_buffer, head)
        body.append(f"{indent * 7}{dt_bias_grad_ref} = {dt_bias_grad_ref} + {scalar0}[0]")
    body.append(f"{indent * 6}for {state_idx} in T.serial(0, {full_state_extent}):")
    body.append(f"{indent * 7}{dh}[{state_idx}] = {dh_prev}[{state_idx}]")
    body.append(f"{indent * 6}for {conv_ch} in T.serial(0, {conv_dim}):")
    body.append(f"{indent * 7}{scalar0}[0] = 1.0 / (1.0 + T.exp(-{conv_pre}[{conv_ch}]))")
    body.append(
        f"{indent * 7}{scalar1}[0] = {conv_grad}[{conv_ch}] * {scalar0}[0] * "
        f"(1.0 + ({conv_pre}[{conv_ch}] * (1.0 - {scalar0}[0])))"
    )
    if conv_bias_grad is not None:
        conv_bias_grad_ref = _indexed_buffer_ref(conv_bias_grad, access_by_buffer, conv_ch)
        body.append(f"{indent * 7}{conv_bias_grad_ref} = {conv_bias_grad_ref} + {scalar1}[0]")
    if history_len > 0:
        body.append(f"{indent * 7}for {kernel_pos} in T.serial(0, {history_len}):")
        body.append(
            f"{indent * 8}{src_row} = {time_idx} - {history_len} + {kernel_pos}"
        )
        body.append(f"{indent * 8}if {src_row} >= 0:")
        body.append(f"{indent * 9}{scalar2}[0] = 0.0")
        body.append(f"{indent * 9}for {hidden_loop} in T.serial(0, {hidden_size}):")
        src_hidden_expr = _node_indexed_canonical_or_positional_input_expr(
            node,
            ("m2rnn_hidden", "hidden"),
            1,
            dtype_by_buffer,
            access_by_buffer,
            f"{src_row} * {hidden_size} + {hidden_loop}",
        )
        src_weight_expr = _node_indexed_canonical_input_expr(
            node,
            "m2rnn_in_proj_weight",
            dtype_by_buffer,
            access_by_buffer,
            f"{conv_ch} * {hidden_size} + {hidden_loop}",
            default="1.0",
        )
        body.append(
            f"{indent * 10}{scalar2}[0] = {scalar2}[0] + "
            f"({src_hidden_expr} * {src_weight_expr})"
        )
        body.append(f"{indent * 8}else:")
        body.append(f"{indent * 9}{src_hist} = {kernel_pos} + {time_idx}")
        conv_state_expr = _node_indexed_canonical_input_expr(
            node,
            "m2rnn_conv_state",
            dtype_by_buffer,
            access_by_buffer,
            f"{src_hist} * {conv_dim} + {conv_ch}",
            default="0.0",
        )
        body.append(f"{indent * 9}{scalar2}[0] = {conv_state_expr}")
        if conv_weight_grad is not None:
            conv_weight_grad_ref = _indexed_buffer_ref(
                conv_weight_grad,
                access_by_buffer,
                f"{conv_ch} * {kernel} + {kernel_pos}",
            )
            body.append(
                f"{indent * 8}{conv_weight_grad_ref} = {conv_weight_grad_ref} + "
                f"({scalar1}[0] * {scalar2}[0])"
            )
        conv_weight_expr = _node_indexed_canonical_input_expr(
            node,
            "m2rnn_conv_weight",
            dtype_by_buffer,
            access_by_buffer,
            f"{conv_ch} * {kernel} + {kernel_pos}",
            default="1.0",
        )
        body.append(f"{indent * 8}if {src_row} == {time_idx}:")
        body.append(
            f"{indent * 9}{project_grad}[{conv_ch}] = "
            f"{project_grad}[{conv_ch}] + ({scalar1}[0] * {conv_weight_expr})"
        )
        body.append(f"{indent * 8}if {src_row} >= 0:")
        if in_proj_weight_grad is not None:
            body.append(f"{indent * 9}for {hidden_loop} in T.serial(0, {hidden_size}):")
            src_hidden_expr = _node_indexed_canonical_or_positional_input_expr(
                node,
                ("m2rnn_hidden", "hidden"),
                1,
                dtype_by_buffer,
                access_by_buffer,
                f"{src_row} * {hidden_size} + {hidden_loop}",
            )
            in_proj_grad_ref = _indexed_buffer_ref(
                in_proj_weight_grad,
                access_by_buffer,
                f"{conv_ch} * {hidden_size} + {hidden_loop}",
            )
            body.append(
                f"{indent * 10}{in_proj_grad_ref} = {in_proj_grad_ref} + "
                f"({src_hidden_expr} * {scalar1}[0] * {conv_weight_expr})"
            )
        if hidden_grad is not None:
            body.append(f"{indent * 9}for {hidden_loop} in T.serial(0, {hidden_size}):")
            hidden_grad_ref = _indexed_buffer_ref(
                hidden_grad,
                access_by_buffer,
                f"{src_row} * {hidden_size} + {hidden_loop}",
            )
            in_proj_weight_expr = _node_indexed_canonical_input_expr(
                node,
                "m2rnn_in_proj_weight",
                dtype_by_buffer,
                access_by_buffer,
                f"{conv_ch} * {hidden_size} + {hidden_loop}",
                default="1.0",
            )
            body.append(
                f"{indent * 10}{hidden_grad_ref} = {hidden_grad_ref} + "
                f"({scalar1}[0] * {conv_weight_expr} * {in_proj_weight_expr})"
            )
    if conv_weight_grad is not None:
        current_conv_weight_grad_ref = _indexed_buffer_ref(
            conv_weight_grad,
            access_by_buffer,
            f"{conv_ch} * {kernel} + {history_len}",
        )
        body.append(
            f"{indent * 7}{current_conv_weight_grad_ref} = {current_conv_weight_grad_ref} + "
            f"({scalar1}[0] * {projected}[{conv_ch}])"
        )
    current_conv_weight_expr = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_conv_weight",
        dtype_by_buffer,
        access_by_buffer,
        f"{conv_ch} * {kernel} + {history_len}",
        default="1.0",
    )
    body.append(
        f"{indent * 7}{project_grad}[{conv_ch}] = "
        f"{project_grad}[{conv_ch}] + ({scalar1}[0] * {current_conv_weight_expr})"
    )
    body.append(f"{indent * 6}T.sync_threads()")
    body.append(f"{indent * 6}for {proj_dim} in T.serial(0, {in_proj_dim}):")
    body.append(f"{indent * 7}for {hidden_loop} in T.serial(0, {hidden_size}):")
    hidden_expr = _node_indexed_canonical_or_positional_input_expr(
        node,
        ("m2rnn_hidden", "hidden"),
        1,
        dtype_by_buffer,
        access_by_buffer,
        f"{time_idx} * {hidden_size} + {hidden_loop}",
    )
    if in_proj_weight_grad is not None:
        in_proj_grad_ref = _indexed_buffer_ref(
            in_proj_weight_grad,
            access_by_buffer,
            f"{proj_dim} * {hidden_size} + {hidden_loop}",
        )
        body.append(
            f"{indent * 8}{in_proj_grad_ref} = {in_proj_grad_ref} + "
            f"({hidden_expr} * {project_grad}[{proj_dim}])"
        )
    if hidden_grad is not None:
        hidden_grad_ref = _indexed_buffer_ref(
            hidden_grad,
            access_by_buffer,
            f"{time_idx} * {hidden_size} + {hidden_loop}",
        )
        in_proj_weight_expr = _node_indexed_canonical_input_expr(
            node,
            "m2rnn_in_proj_weight",
            dtype_by_buffer,
            access_by_buffer,
            f"{proj_dim} * {hidden_size} + {hidden_loop}",
            default="1.0",
        )
        body.append(
            f"{indent * 8}{hidden_grad_ref} = {hidden_grad_ref} + "
            f"({project_grad}[{proj_dim}] * {in_proj_weight_expr})"
        )
    if h0_grad is not None:
        body.append(f"{indent * 6}for {state_idx} in T.serial(0, {full_state_extent}):")
        body.append(f"{indent * 7}{head} = {state_idx} // {k_dim * v_dim}")
        body.append(f"{indent * 7}{kk} = ({state_idx} // {v_dim}) % {k_dim}")
        body.append(f"{indent * 7}{vv} = {state_idx} % {v_dim}")
        h0_grad_ref = _indexed_buffer_ref(
            h0_grad,
            access_by_buffer,
            f"{head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv}",
        )
        body.append(f"{indent * 7}{h0_grad_ref} = {dh}[{state_idx}]")
    body.append(f"{indent * 3}T.sync_threads()")
    return


# Budget below the queried CUDA opt-in shared-memory cap that a single generated
# kernel's residual ``T.alloc_shared`` scratch may occupy.  Anything above this
# (after the existing ABI-scratch spill pass has run) is rewritten to a global
# (device) workspace so ptxas accepts the kernel.  We target the conservative
# static per-block budget (48 KiB) so the residual shared footprint stays well
# inside the opt-in cap with headroom for compiler bookkeeping.
_CUDA_SHARED_SCRATCH_BUDGET_BYTES = 0xC000  # 49152 bytes (static per-block).

# Metal threadgroup-memory pooling for a generated kernel's residual
# ``T.alloc_shared`` scratch.  Apple GPUs cap threadgroup memory at a
# device-queried ``maxThreadgroupMemoryLength`` (32768 on M4 Max).  A generated
# mamba3/m2rnn reverse-scan backward declares MULTIPLE MB of ``shared.dyn`` (the
# full reverse-time state h_prev/h_next/dh/dh_prev plus the per-step
# in-proj/conv/B-C scratch), which TileLang lowers to one giant
# ``threadgroup float buf_dyn_shmem[...]`` array.  When that array exceeds the
# threadgroup limit the Metal driver crashes ``MTLCompilerService`` at
# ``newComputePipelineState`` (XPC_ERROR_CONNECTION_INTERRUPTED).
#
# REPLACED (design step 5 / §5): the former hardcoded
# ``_METAL_SHARED_SCRATCH_TRIGGER_BYTES = 28672`` and
# ``_METAL_SHARED_SCRATCH_DEMOTE_TARGET_BYTES = 8192`` are now DERIVED from the
# device-capability probe by ``_pool_oversized_shared_scratch_to_metal_workspace``:
#   * TRIGGER  -- a kernel overflows threadgroup memory when its residual LOGICAL
#     ``alloc_shared`` total exceeds the QUERIED ``caps.threadgroup_mem_bytes``
#     (32768 on M4 Max; the hand-tuned literal was 28672 == cap - one threadgroup
#     page). This is the PHYSICAL-OVERFLOW point: small fullgraph kernels whose
#     logical scratch fits the cap are left untouched.
#   * DEMOTE TARGET -- once a kernel is found to overflow, demote its largest
#     residual buffers until the survivors fit ``cap / margin`` logical, so the
#     coalesced physical ``buf_dyn_shmem`` (= logical * margin, ~3.7x from the
#     preset) lands at the cap (8856 logical ~= the hand-tuned 8192).
# REGRESSION NOTE: step 5 originally used ``cap / margin`` (8856) as BOTH the
# trigger AND the target. That lowered the trigger from 28672 to 8856 and pooled
# the ~20 KiB single-launcher fullgraph train block, stripping the internal
# fusion-edge buffer ``entry_rmsnorm_hidden`` (alloc_shared -> alloc_global pool)
# out of the entry PrimFunc -- tilelang's fullgraph validator then rejected the
# missing load/store and the auto-split silently fell back to the staged launcher
# (a RULE #1 violation). The trigger is now the cap itself; the margin governs
# only the demote target, matching main 82b6ab0 byte-for-byte.


def _cuda_shared_memory_optin_cap_bytes() -> int | None:
    """Return the CUDA opt-in max shared-memory-per-block, or ``None`` off CUDA.

    The demotion of residual oversized shared scratch to a global device
    workspace is *only* applied when the active Path C lowering target is CUDA
    (``_path_c_default_target() == "cuda"``).  On Metal hosts this returns
    ``None`` and callers MUST treat that as "no capacity limit" so the generated
    schedule is byte-for-byte identical to the pre-change behaviour (every
    surviving ``T.alloc_shared`` buffer keeps its shared scope).  Metal's
    threadgroup model spills oversized scratch to device memory in its own
    backend, so this CUDA-only demotion mirrors what Metal already does without
    touching the Metal codegen path.
    """

    if _path_c_default_target() != "cuda":
        return None
    # RULE #1: the opt-in cap is the LIVE-QUERIED shared_memory_per_block_optin
    # from the device-caps probe -- NEVER a hardcoded floor. If the query fails
    # on a CUDA target the probe RAISES (path_c_device_caps._probe_cuda_live),
    # which propagates here; we do NOT substitute a guessed 0x18C00 floor that
    # could silently mis-size the demote and emit a kernel ptxas rejects.
    from cppmega_mlx.runtime.path_c_device_caps import device_caps

    caps = device_caps()
    return int(caps.threadgroup_mem_bytes)


def _demote_residual_shared_scratch_to_global(source: str) -> str:
    """CUDA-only: rewrite residual oversized ``T.alloc_shared`` to ``T.alloc_global``.

    Runs AFTER ``_spill_large_shared_scratch_to_abi`` so it only ever touches
    ``T.alloc_shared`` lines that survived the ABI-scratch spill (e.g. the full
    reverse-scan Mamba3 backward state ``h_prev``/``h_next``/``dh``/``dh_prev``,
    which cannot be spilled to ABI device-scratch params in the direct-chain path
    because the portable 31-buffer kernel ABI budget is exhausted).  For
    ``local_gb10_quarter`` those buffers sum to ~7.4 MB of ``shared.dyn`` per
    block, which ptxas rejects (max ~99 KiB even with the opt-in attribute on
    Blackwell sm_121a).  ``T.alloc_global`` is a kernel-internal device workspace
    (no ABI param, no 31-buffer-limit interaction) and is launched with a single
    block, so it preserves the cross-lane visibility the shared buffers provided.

    Off CUDA (``_cuda_shared_memory_optin_cap_bytes() is None``, e.g. Metal) this
    returns ``source`` unchanged, so the Metal schedule is byte-for-byte
    identical to the pre-change behaviour.  The largest residual shared buffers
    are demoted first until the residual shared total fits the CUDA budget.
    """

    cap_bytes = _cuda_shared_memory_optin_cap_bytes()
    if cap_bytes is None:
        return source
    # Budget = min(queried static per-block shared, queried opt-in cap) -- both
    # now come from the device-caps probe (design §5), not the literal
    # _CUDA_SHARED_SCRATCH_BUDGET_BYTES (kept only as a conservative ceiling so a
    # future device with a larger static cap still leaves compiler headroom).
    from cppmega_mlx.runtime.path_c_device_caps import device_caps

    static_shared = int(device_caps().static_shared_mem_bytes)
    budget_bytes = min(_CUDA_SHARED_SCRATCH_BUDGET_BYTES, static_shared, cap_bytes)
    lines = source.splitlines()
    shared: list[tuple[int, str, int]] = []  # (line_index, name, byte_count)
    shared_total = 0
    for index, line in enumerate(lines):
        match = _ALLOC_SHARED_LINE_RE.match(line)
        if match is None:
            continue
        shape_value = ast.literal_eval(match.group("shape"))
        shape = (
            (int(shape_value),)
            if isinstance(shape_value, int)
            else tuple(int(dim) for dim in shape_value)
        )
        byte_count = _flattened_extent(shape) * _DTYPE_NBYTES[match.group("dtype")]
        shared.append((index, match.group("name"), byte_count))
        shared_total += byte_count
    if shared_total <= budget_bytes:
        return source
    # Demote the largest residual shared buffers first until the rest fit.
    for index, _name, byte_count in sorted(
        shared, key=lambda item: item[2], reverse=True
    ):
        if shared_total <= budget_bytes:
            break
        lines[index] = lines[index].replace(
            "T.alloc_shared(", "T.alloc_global(", 1
        )
        shared_total -= byte_count
    return "\n".join(lines) + ("\n" if source.endswith("\n") else "")


def _pool_oversized_shared_scratch_to_metal_workspace(source: str) -> str:
    """Metal: pool oversized ``T.alloc_shared`` scratch into ONE global workspace.

    On Metal the residual reverse-scan backward scratch (the full reverse-time
    state h_prev/h_next/dh/dh_prev plus per-step in-proj/conv/B-C buffers) sums to
    several MB of ``shared.dyn``, which TileLang lowers to a single giant
    ``threadgroup float buf_dyn_shmem[...]`` array that overflows Metal's 32 KiB
    threadgroup-memory limit and crashes ``newComputePipelineState``
    (XPC_ERROR_CONNECTION_INTERRUPTED).  We CANNOT demote each buffer to its own
    ``T.alloc_global`` — every global buffer becomes a kernel-buffer argument and
    the backward kernel already uses ~28 of Metal's ~31-buffer-argument limit, so
    demoting even the four 1.75 MB state buffers blows the argument limit and the
    Metal source fails to compile.

    The durable fix pools EVERY demoted ``float32`` buffer into ONE coalesced
    ``T.alloc_global`` workspace (a SINGLE extra kernel buffer) and rewrites every
    ``name[idx]`` reference to ``pool[offset + (idx)]`` via the existing
    1-D-buffer-ref remapper.  This moves the oversized scratch off threadgroup
    memory (clearing the crash) while adding exactly one buffer argument.  The
    cross-lane visibility the shared buffers provided is preserved: the kernel is
    launched single-block, so a global workspace is visible to every lane exactly
    like threadgroup memory was.  Only ``float32`` buffers are pooled (the only
    dtype the oversized reverse-scan scratch uses); a non-float32 oversized buffer
    raises rather than silently degrading.

    Off Metal this is a no-op.  Behaviour-preserving: the demotion only changes
    WHERE the scratch lives, not the arithmetic, so gradients are unchanged.
    """

    if _path_c_default_target() != "metal":
        return source
    # Scope: only the recurrent reverse-scan BACKWARD kernels
    # (mamba3_mimo_bwd / m2rnn_bwd) carry the multi-MB reverse-time state
    # (h_prev/h_next/dh/dh_prev) that overflows Metal's threadgroup cap and is NOT
    # covered by the forward newComputePipelineState segment-node split.  Gate on
    # those distinctive state-buffer names so this pass NEVER perturbs the forward
    # fusion planner's buffer-count grouping (which the forward split already
    # handles) — pooling there would only change WHERE forward scratch lives while
    # spuriously shifting greedy fusion boundaries.
    if not (
        "_bwd_mamba3_h_next = T.alloc_shared(" in source
        or "_bwd_mamba3_h_prev = T.alloc_shared(" in source
        or "_bwd_m2rnn_h_next = T.alloc_shared(" in source
        or "_bwd_m2rnn_h_prev = T.alloc_shared(" in source
    ):
        return source
    lines = source.splitlines()
    shared: list[tuple[int, str, tuple[int, ...], str, int]] = []
    shared_total = 0
    for index, line in enumerate(lines):
        match = _ALLOC_SHARED_LINE_RE.match(line)
        if match is None:
            continue
        shape_value = ast.literal_eval(match.group("shape"))
        shape = (
            (int(shape_value),)
            if isinstance(shape_value, int)
            else tuple(int(dim) for dim in shape_value)
        )
        dtype = match.group("dtype")
        byte_count = _flattened_extent(shape) * _DTYPE_NBYTES[dtype]
        shared.append((index, match.group("name"), shape, dtype, byte_count))
        shared_total += byte_count
    # Only pool kernels whose PHYSICAL shared scratch would overflow the QUERIED
    # Metal threadgroup cap; kernels that already fit are left untouched so the
    # planner's buffer-count splitting is unchanged for them. The trigger and the
    # demote target are now derived from the device-capability probe (queried
    # threadgroup_mem_bytes) and the preset/calibrated logical->physical packing
    # margin -- NOT the hardcoded _METAL_SHARED_SCRATCH_TRIGGER_BYTES (28672) /
    # _DEMOTE_TARGET_BYTES (8192). TileLang coalesces the residual alloc_shared
    # into one buf_dyn_shmem array running ``margin`` x the logical bytes, so the
    # principled comparison is physical (= logical * margin) against the cap.
    from cppmega_mlx.runtime.path_c_device_caps import device_caps

    caps = device_caps()
    threadgroup_cap = int(caps.threadgroup_mem_bytes)
    margin = float(caps.logical_to_physical_shared_margin)
    if margin <= 0:
        raise ValueError(
            "path_c shared-scratch pool: logical_to_physical_shared_margin must "
            f"be >0 (got {margin!r} on {caps.device_name})"
        )
    # TRIGGER vs DEMOTE-TARGET are two DISTINCT thresholds (the hand-tuned pair
    # was 28K trigger / 8K target). The ``margin`` (logical->physical coalescing
    # ratio, ~3.7x) is calibrated on the multi-MB reverse-scan kernels' giant
    # ``buf_dyn_shmem`` and governs only the DEMOTE TARGET (how far to pool once
    # we have decided a kernel overflows), NOT the trigger. A kernel physically
    # overflows Metal threadgroup memory when its residual LOGICAL ``alloc_shared``
    # total exceeds the queried threadgroup cap -- small fullgraph kernels (e.g.
    # the single-launcher train block, ~20 KiB logical here) sit BELOW the cap and
    # MUST NOT be pooled: their tiny static ``alloc_shared`` buffers are real
    # threadgroup memory with no giant dynamic-shared coalescing, so demoting an
    # internal fusion edge buffer (entry_rmsnorm_hidden) off threadgroup memory
    # would strip its load/store from the entry PrimFunc and break the fullgraph
    # single-kernel fusion the launcher path requires. Conflating the trigger with
    # ``cap / margin`` (==8856 here) regressed exactly that path; the trigger is
    # the physical cap itself (28672-equivalent at the M4 Max 32 KiB cap, minus
    # the per-kernel overhead Metal already reserves above the residual scratch).
    demote_target_logical = threadgroup_cap / margin
    if shared_total <= threadgroup_cap:
        return source
    # Select the largest residual shared buffers to pool until the rest fit,
    # guaranteeing the physical buf_dyn_shmem fits the queried threadgroup cap.
    demote: list[tuple[int, str, tuple[int, ...], str, int]] = []
    remaining = shared_total
    for entry in sorted(shared, key=lambda item: item[4], reverse=True):
        if remaining <= demote_target_logical:
            break
        demote.append(entry)
        remaining -= entry[4]
    if not demote:
        return source
    for _index, name, _shape, dtype, _byte_count in demote:
        if dtype != "float32":
            raise ValueError(
                "Metal oversized-shared pooling only supports float32 scratch; "
                f"buffer {name!r} is {dtype!r} and exceeds the threadgroup budget. "
                "Add a typed pool or reduce its extent — refusing to silently "
                "leave it in overflowing threadgroup memory."
            )
    # Assign each demoted buffer a contiguous slice of one float32 pool workspace.
    pool_name = _safe_identifier("path_c_metal_shared_pool")
    offsets: dict[str, int] = {}
    pool_extent = 0
    demote_indices = {entry[0] for entry in demote}
    for _index, name, shape, _dtype, _byte_count in demote:
        offsets[name] = pool_extent
        pool_extent += _flattened_extent(shape)
    # Find the indentation of the first demoted alloc line so the pool allocation
    # is emitted at the same scope.
    first_index = min(demote_indices)
    pool_indent = _ALLOC_SHARED_LINE_RE.match(lines[first_index]).group("indent")
    out: list[str] = []
    pool_emitted = False
    for index, line in enumerate(lines):
        if index in demote_indices:
            if not pool_emitted:
                out.append(
                    f'{pool_indent}{pool_name} = '
                    f'T.alloc_global(({pool_extent},), "float32")'
                )
                pool_emitted = True
            # Drop the original alloc_shared line for the pooled buffer.
            continue
        remapped = line
        for _di, name, _shape, _dtype, _byte_count in demote:
            if f"{name}[" in remapped:
                remapped = _replace_one_dimensional_buffer_refs(
                    remapped,
                    source_name=name,
                    target_name=pool_name,
                    target_offset=offsets[name],
                )
        out.append(remapped)
    return "\n".join(out) + ("\n" if source.endswith("\n") else "")


def _append_row_phased_mamba3_bwd_body(
    body: list[str],
    *,
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    shape_env: PathCModelShapeEnv,
    thread_count: int,
    indent: str,
    launcher_chunked_rows: bool = False,
) -> None:
    stage_grad = _scratch_name(node, "mamba3_stage_grad")
    proj_dim = _scratch_name(node, "proj_dim")
    hidden_dim = _scratch_name(node, "hidden_dim")
    conv_ch = _scratch_name(node, "conv_ch")
    state_idx = _scratch_name(node, "state_idx")
    grad_flat = _scratch_name(node, "grad_flat")
    hidden_size = int(shape_env.hidden_size)
    sequence_length = int(shape_env.sequence_length)
    in_proj_dim = int(shape_env.mamba_in_proj_dim)
    inner_dim = int(shape_env.mamba_inner_dim)
    conv_channels = int(shape_env.mamba_conv_channels)
    kernel = int(shape_env.mamba_conv_kernel)
    state_extent = (
        int(shape_env.mamba_num_heads)
        * int(shape_env.mamba_head_dim)
        * int(shape_env.mamba_state_dim)
    )
    norm_extent = (
        int(shape_env.mamba_effective_mimo_rank)
        * int(shape_env.mamba_groups)
        * int(shape_env.mamba_state_dim)
    )
    hidden_grad = _node_output_for_canonical(node, "hidden")
    in_proj_weight_grad = _node_output_for_canonical(node, "mamba3_in_proj_weight")
    out_proj_weight_grad = _node_output_for_canonical(node, "mamba3_out_proj_weight")
    conv_weight_grad = _node_output_for_canonical(node, "mamba3_conv_weight")
    conv_bias_grad = _node_output_for_canonical(node, "mamba3_conv_bias")
    dt_bias_grad = _node_output_for_canonical(node, "mamba3_dt_bias")
    b_norm_weight_grad = _node_output_for_canonical(node, "mamba3_B_norm_weight")
    b_bias_grad = _node_output_for_canonical(node, "mamba3_B_bias")
    c_norm_weight_grad = _node_output_for_canonical(node, "mamba3_C_norm_weight")
    c_bias_grad = _node_output_for_canonical(node, "mamba3_C_bias")
    d_grad = _node_output_for_canonical(node, "mamba3_D")
    h0_grad = _node_output_for_canonical(node, "mamba3_h0")
    _append_descriptor_node_comments(
        body,
        node=node,
        descriptor=descriptor,
        indent=indent * 3,
    )
    projected = _scratch_name(node, "mamba3_projected_vec")
    project_grad = _scratch_name(node, "mamba3_project_grad")
    conv_pre = _scratch_name(node, "mamba3_conv_pre")
    conv = _scratch_name(node, "mamba3_conv_vec")
    conv_grad = _scratch_name(node, "mamba3_conv_grad")
    out_inner = _scratch_name(node, "mamba3_out_inner")
    y_skip = _scratch_name(node, "mamba3_y_skip")
    out_inner_grad = _scratch_name(node, "mamba3_out_inner_grad")
    h_checkpoint = _scratch_name(node, "mamba3_h_checkpoint")
    angle_checkpoint = _scratch_name(node, "mamba3_angle_checkpoint")
    h_prev = _scratch_name(node, "mamba3_h_prev")
    h_next = _scratch_name(node, "mamba3_h_next")
    dh = _scratch_name(node, "mamba3_dh")
    dh_prev = _scratch_name(node, "mamba3_dh_prev")
    b_inv = _scratch_name(node, "mamba3_b_inv_rms")
    c_inv = _scratch_name(node, "mamba3_c_inv_rms")
    b_mean = _scratch_name(node, "mamba3_b_mean")
    c_mean = _scratch_name(node, "mamba3_c_mean")
    b_raw = _scratch_name(node, "mamba3_b_raw")
    c_raw = _scratch_name(node, "mamba3_c_raw")
    b_group = _scratch_name(node, "mamba3_b_group")
    c_group = _scratch_name(node, "mamba3_c_group")
    b_raw_grad = _scratch_name(node, "mamba3_b_raw_grad")
    c_raw_grad = _scratch_name(node, "mamba3_c_raw_grad")
    b_group_grad = _scratch_name(node, "mamba3_b_group_grad")
    c_group_grad = _scratch_name(node, "mamba3_c_group_grad")
    dt_vec = _scratch_name(node, "mamba3_dt_vec")
    a_vec = _scratch_name(node, "mamba3_a_vec")
    dt_grad = _scratch_name(node, "mamba3_dt_grad")
    a_grad = _scratch_name(node, "mamba3_a_grad")
    trap_group = _scratch_name(node, "mamba3_trap_group")
    trap_grad = _scratch_name(node, "mamba3_trap_grad")
    next_dt_pre_vec = _scratch_name(node, "mamba3_next_dt_pre_vec")
    next_dt_vec = _scratch_name(node, "mamba3_next_dt_vec")
    next_trap_vec = _scratch_name(node, "mamba3_next_trap_vec")
    angle_cumsum = _scratch_name(node, "mamba3_angle_cumsum")
    angle_grad = _scratch_name(node, "mamba3_angle_grad")
    sum_scratch = _scratch_name(node, "mamba3_sum")
    dot_scratch = _scratch_name(node, "mamba3_dot")
    scalar0 = _scratch_name(node, "mamba3_scalar0")
    scalar1 = _scratch_name(node, "mamba3_scalar1")
    scalar2 = _scratch_name(node, "mamba3_scalar2")
    scalar3 = _scratch_name(node, "mamba3_scalar3")
    scalar4 = _scratch_name(node, "mamba3_scalar4")
    time_rev = _scratch_name(node, "time_rev")
    time_idx = _scratch_name(node, "time_idx")
    replay_time = _scratch_name(node, "replay_time")
    replay_offset = _scratch_name(node, "replay_offset")
    checkpoint_idx = _scratch_name(node, "checkpoint_idx")
    checkpoint_start = _scratch_name(node, "checkpoint_start")
    src_row = _scratch_name(node, "src_row")
    src_hist = _scratch_name(node, "src_hist")
    hidden_loop = _scratch_name(node, "hidden_loop")
    kernel_pos = _scratch_name(node, "kernel_pos")
    rank_loop = _scratch_name(node, "rank_loop")
    group_loop = _scratch_name(node, "group_loop")
    angle_loop = _scratch_name(node, "angle_loop")
    state_loop = _scratch_name(node, "state_loop")
    pair_state = _scratch_name(node, "pair_state")
    head = _scratch_name(node, "head")
    feature = _scratch_name(node, "feature")
    out_dim = _scratch_name(node, "out_dim")
    heads = int(shape_env.mamba_num_heads)
    head_dim = int(shape_env.mamba_head_dim)
    state_dim = int(shape_env.mamba_state_dim)
    groups = int(shape_env.mamba_groups)
    mimo_rank = int(shape_env.mamba_effective_mimo_rank)
    rope_angles = int(shape_env.mamba_num_rope_angles)
    bc_dim = int(shape_env.mamba_bc_dim)
    heads_per_group = max(1, heads // max(1, groups))
    history_len = max(0, kernel - 1)
    z_offset = 0
    x_offset = inner_dim
    b_offset = 2 * inner_dim
    c_offset = b_offset + bc_dim
    dt_offset = c_offset + bc_dim
    a_offset = dt_offset + heads
    trap_offset = a_offset + heads
    angle_offset = trap_offset + heads
    conv_b_offset = inner_dim
    conv_c_offset = inner_dim + bc_dim
    rot_dim = min(state_dim, 2 * rope_angles)
    checkpoint_interval = MAMBA3_BWD_REPLAY_CHECKPOINT_INTERVAL
    checkpoint_count = (sequence_length + checkpoint_interval - 1) // checkpoint_interval
    angle_extent = heads * rope_angles

    def _mamba3_hidden_expr(time_expr: str, hidden_expr: str) -> str:
        return _node_indexed_canonical_or_positional_input_expr(
            node,
            ("hidden",),
            1,
            dtype_by_buffer,
            access_by_buffer,
            f"{time_expr} * {hidden_size} + {hidden_expr}",
        )

    def _mamba3_project_weight_expr(proj_expr: str, hidden_expr: str) -> str:
        return _node_indexed_canonical_input_expr(
            node,
            "mamba3_in_proj_weight",
            dtype_by_buffer,
            access_by_buffer,
            f"({proj_expr}) * {hidden_size} + ({hidden_expr})",
            default="1.0",
        )

    def _mamba3_conv_weight_expr(channel_expr: str, kernel_expr: str) -> str:
        return _node_indexed_canonical_input_expr(
            node,
            "mamba3_conv_weight",
            dtype_by_buffer,
            access_by_buffer,
            f"({channel_expr}) * {kernel} + ({kernel_expr})",
            default="1.0",
        )

    def _mamba3_emit_recompute_row(
        time_expr: str,
        *,
        level: int,
        update_angle: bool,
        update_state: bool,
    ) -> None:
        body.append(
            f"{indent * level}for {proj_dim} in T.serial(lane, {in_proj_dim}, "
            f"step={thread_count}):"
        )
        body.append(f"{indent * (level + 1)}{sum_scratch}[0] = 0.0")
        body.append(
            f"{indent * (level + 1)}for {hidden_loop} in T.serial(0, {hidden_size}):"
        )
        body.append(
            f"{indent * (level + 2)}{sum_scratch}[0] = {sum_scratch}[0] + "
            f"({_mamba3_hidden_expr(time_expr, hidden_loop)} * "
            f"{_mamba3_project_weight_expr(proj_dim, hidden_loop)})"
        )
        body.append(f"{indent * (level + 1)}{projected}[{proj_dim}] = {sum_scratch}[0]")
        body.append(f"{indent * level}T.sync_threads()")
        body.append(
            f"{indent * level}for {conv_ch} in T.serial(lane, {conv_channels}, "
            f"step={thread_count}):"
        )
        conv_bias_expr = _node_indexed_canonical_input_expr(
            node,
            "mamba3_conv_bias",
            dtype_by_buffer,
            access_by_buffer,
            conv_ch,
            default="0.0",
        )
        body.append(f"{indent * (level + 1)}{conv_pre}[{conv_ch}] = {conv_bias_expr}")
        if history_len > 0:
            body.append(
                f"{indent * (level + 1)}for {kernel_pos} in T.serial(0, {history_len}):"
            )
            body.append(
                f"{indent * (level + 2)}{src_row} = {time_expr} - {history_len} + {kernel_pos}"
            )
            body.append(f"{indent * (level + 2)}if {src_row} >= 0:")
            body.append(f"{indent * (level + 3)}{scalar0}[0] = 0.0")
            body.append(
                f"{indent * (level + 3)}for {hidden_loop} in T.serial(0, {hidden_size}):"
            )
            body.append(
                f"{indent * (level + 4)}{scalar0}[0] = {scalar0}[0] + "
                f"({_mamba3_hidden_expr(src_row, hidden_loop)} * "
                f"{_mamba3_project_weight_expr(f'{x_offset} + {conv_ch}', hidden_loop)})"
            )
            body.append(f"{indent * (level + 2)}else:")
            body.append(f"{indent * (level + 3)}{src_hist} = {kernel_pos} + {time_expr}")
            conv_state_expr = _node_indexed_canonical_input_expr(
                node,
                "mamba3_conv_state",
                dtype_by_buffer,
                access_by_buffer,
                f"{src_hist} * {conv_channels} + {conv_ch}",
                default="0.0",
            )
            body.append(f"{indent * (level + 3)}{scalar0}[0] = {conv_state_expr}")
            body.append(
                f"{indent * (level + 2)}{conv_pre}[{conv_ch}] = "
                f"{conv_pre}[{conv_ch}] + "
                f"({scalar0}[0] * {_mamba3_conv_weight_expr(conv_ch, kernel_pos)})"
            )
        body.append(
            f"{indent * (level + 1)}{conv_pre}[{conv_ch}] = "
            f"{conv_pre}[{conv_ch}] + "
            f"({projected}[{x_offset} + {conv_ch}] * "
            f"{_mamba3_conv_weight_expr(conv_ch, str(history_len))})"
        )
        body.append(
            f"{indent * (level + 1)}{scalar0}[0] = 1.0 / "
            f"(1.0 + T.exp(-{conv_pre}[{conv_ch}]))"
        )
        body.append(
            f"{indent * (level + 1)}{conv}[{conv_ch}] = "
            f"{conv_pre}[{conv_ch}] * {scalar0}[0]"
        )
        body.append(f"{indent * level}T.sync_threads()")
        body.append(
            f"{indent * level}for {head} in T.serial(lane, {heads}, "
            f"step={thread_count}):"
        )
        dt_bias_expr = _node_indexed_canonical_input_expr(
            node,
            "mamba3_dt_bias",
            dtype_by_buffer,
            access_by_buffer,
            head,
            default="0.0",
        )
        body.append(
            f"{indent * (level + 1)}{dt_vec}[{head}] = "
            f"T.log(1.0 + T.exp({projected}[{dt_offset} + {head}] + {dt_bias_expr}))"
        )
        body.append(
            f"{indent * (level + 1)}{a_vec}[{head}] = "
            f"T.min(-T.log(1.0 + T.exp({projected}[{a_offset} + {head}])), -0.01)"
        )
        if update_angle:
            body.append(
                f"{indent * (level + 1)}for {angle_loop} in T.serial(0, {rope_angles}):"
            )
            body.append(
                f"{indent * (level + 2)}{angle_cumsum}[{head} * {rope_angles} + {angle_loop}] = "
                f"{angle_cumsum}[{head} * {rope_angles} + {angle_loop}] + "
                f"({projected}[{angle_offset} + {angle_loop}] * {dt_vec}[{head}])"
            )
        body.append(
            f"{indent * (level + 1)}{next_dt_pre_vec}[{head}] = 0.0"
        )
        body.append(f"{indent * (level + 1)}{next_dt_vec}[{head}] = 0.0")
        body.append(f"{indent * (level + 1)}{next_trap_vec}[{head}] = 0.0")
        body.append(f"{indent * (level + 1)}if {time_expr} + 1 < {sequence_length}:")
        body.append(
            f"{indent * (level + 2)}for {hidden_loop} in T.serial(0, {hidden_size}):"
        )
        body.append(
            f"{indent * (level + 3)}{next_dt_pre_vec}[{head}] = "
            f"{next_dt_pre_vec}[{head}] + "
            f"({_mamba3_hidden_expr(f'({time_expr} + 1)', hidden_loop)} * "
            f"{_mamba3_project_weight_expr(f'{dt_offset} + {head}', hidden_loop)})"
        )
        body.append(
            f"{indent * (level + 3)}{next_trap_vec}[{head}] = "
            f"{next_trap_vec}[{head}] + "
            f"({_mamba3_hidden_expr(f'({time_expr} + 1)', hidden_loop)} * "
            f"{_mamba3_project_weight_expr(f'{trap_offset} + {head}', hidden_loop)})"
        )
        body.append(
            f"{indent * (level + 2)}{next_dt_pre_vec}[{head}] = "
            f"{next_dt_pre_vec}[{head}] + {dt_bias_expr}"
        )
        body.append(
            f"{indent * (level + 2)}{next_dt_vec}[{head}] = "
            f"T.log(1.0 + T.exp({next_dt_pre_vec}[{head}]))"
        )
        body.append(f"{indent * level}T.sync_threads()")
        body.append(f"{indent * level}for {group_loop} in T.serial(0, {groups}):")
        body.append(f"{indent * (level + 1)}{trap_group}[{group_loop}] = 0.0")
        body.append(
            f"{indent * (level + 1)}for {head} in T.serial(0, {heads_per_group}):"
        )
        body.append(
            f"{indent * (level + 2)}{hidden_dim} = {group_loop} * {heads_per_group} + {head}"
        )
        body.append(
            f"{indent * (level + 2)}{scalar0}[0] = "
            f"1.0 / (1.0 + T.exp(-{projected}[{trap_offset} + {hidden_dim}]))"
        )
        body.append(
            f"{indent * (level + 2)}{scalar1}[0] = "
            f"1.0 / (1.0 + T.exp(-{next_trap_vec}[{hidden_dim}]))"
        )
        body.append(
            f"{indent * (level + 2)}{trap_group}[{group_loop}] = "
            f"{trap_group}[{group_loop}] + "
            f"({next_dt_vec}[{hidden_dim}] * (1.0 - {scalar1}[0]) + "
            f"{dt_vec}[{hidden_dim}] * {scalar0}[0])"
        )
        body.append(
            f"{indent * (level + 1)}{trap_group}[{group_loop}] = "
            f"{trap_group}[{group_loop}] / {float(heads_per_group):.1f}"
        )
        body.append(f"{indent * level}T.sync_threads()")
        body.append(f"{indent * level}for {rank_loop} in T.serial(0, {mimo_rank}):")
        body.append(f"{indent * (level + 1)}for {group_loop} in T.serial(0, {groups}):")
        body.append(f"{indent * (level + 2)}{sum_scratch}[0] = 0.0")
        body.append(f"{indent * (level + 2)}{dot_scratch}[0] = 0.0")
        body.append(
            f"{indent * (level + 2)}for {state_loop} in T.serial(0, {state_dim}):"
        )
        body.append(
            f"{indent * (level + 3)}{grad_flat} = "
            f"({rank_loop} * {groups} + {group_loop}) * {state_dim} + {state_loop}"
        )
        body.append(
            f"{indent * (level + 3)}{sum_scratch}[0] = {sum_scratch}[0] + "
            f"({conv}[{conv_b_offset} + {grad_flat}] * {conv}[{conv_b_offset} + {grad_flat}])"
        )
        body.append(
            f"{indent * (level + 3)}{dot_scratch}[0] = {dot_scratch}[0] + "
            f"({conv}[{conv_c_offset} + {grad_flat}] * {conv}[{conv_c_offset} + {grad_flat}])"
        )
        body.append(
            f"{indent * (level + 2)}{b_inv}[{rank_loop} * {groups} + {group_loop}] = "
            f"T.rsqrt(({sum_scratch}[0] / {float(state_dim):.1f}) + 0.00001)"
        )
        body.append(
            f"{indent * (level + 2)}{c_inv}[{rank_loop} * {groups} + {group_loop}] = "
            f"T.rsqrt(({dot_scratch}[0] / {float(state_dim):.1f}) + 0.00001)"
        )
        body.append(f"{indent * level}for {group_loop} in T.serial(0, {groups}):")
        body.append(
            f"{indent * (level + 1)}for {state_loop} in T.serial(0, {state_dim}):"
        )
        body.append(
            f"{indent * (level + 2)}{b_mean}[{group_loop} * {state_dim} + {state_loop}] = 0.0"
        )
        body.append(
            f"{indent * (level + 2)}{c_mean}[{group_loop} * {state_dim} + {state_loop}] = 0.0"
        )
        body.append(f"{indent * (level + 2)}for {rank_loop} in T.serial(0, {mimo_rank}):")
        body.append(
            f"{indent * (level + 3)}{grad_flat} = "
            f"({rank_loop} * {groups} + {group_loop}) * {state_dim} + {state_loop}"
        )
        b_norm_expr = _node_indexed_canonical_input_expr(
            node,
            "mamba3_B_norm_weight",
            dtype_by_buffer,
            access_by_buffer,
            grad_flat,
            default="1.0",
        )
        b_bias_expr = _node_indexed_canonical_input_expr(
            node,
            "mamba3_B_bias",
            dtype_by_buffer,
            access_by_buffer,
            grad_flat,
            default="0.0",
        )
        c_norm_expr = _node_indexed_canonical_input_expr(
            node,
            "mamba3_C_norm_weight",
            dtype_by_buffer,
            access_by_buffer,
            grad_flat,
            default="1.0",
        )
        c_bias_expr = _node_indexed_canonical_input_expr(
            node,
            "mamba3_C_bias",
            dtype_by_buffer,
            access_by_buffer,
            grad_flat,
            default="0.0",
        )
        body.append(
            f"{indent * (level + 3)}{b_mean}[{group_loop} * {state_dim} + {state_loop}] = "
            f"{b_mean}[{group_loop} * {state_dim} + {state_loop}] + "
            f"({conv}[{conv_b_offset} + {grad_flat}] * "
            f"{b_inv}[{rank_loop} * {groups} + {group_loop}] * {b_norm_expr} + {b_bias_expr})"
        )
        body.append(
            f"{indent * (level + 3)}{c_mean}[{group_loop} * {state_dim} + {state_loop}] = "
            f"{c_mean}[{group_loop} * {state_dim} + {state_loop}] + "
            f"({conv}[{conv_c_offset} + {grad_flat}] * "
            f"{c_inv}[{rank_loop} * {groups} + {group_loop}] * {c_norm_expr} + {c_bias_expr})"
        )
        body.append(
            f"{indent * (level + 2)}{b_mean}[{group_loop} * {state_dim} + {state_loop}] = "
            f"{b_mean}[{group_loop} * {state_dim} + {state_loop}] / {float(mimo_rank):.1f}"
        )
        body.append(
            f"{indent * (level + 2)}{c_mean}[{group_loop} * {state_dim} + {state_loop}] = "
            f"{c_mean}[{group_loop} * {state_dim} + {state_loop}] / {float(mimo_rank):.1f}"
        )
        body.append(
            f"{indent * (level + 2)}{b_raw}[{group_loop} * {state_dim} + {state_loop}] = "
            f"{b_mean}[{group_loop} * {state_dim} + {state_loop}] * {trap_group}[{group_loop}]"
        )
        body.append(
            f"{indent * (level + 2)}{c_raw}[{group_loop} * {state_dim} + {state_loop}] = "
            f"{c_mean}[{group_loop} * {state_dim} + {state_loop}]"
        )
        body.append(f"{indent * level}T.sync_threads()")
        body.append(f"{indent * level}for {group_loop} in T.serial(0, {groups}):")
        body.append(
            f"{indent * (level + 1)}for {state_loop} in T.serial(0, {state_dim}):"
        )
        body.append(f"{indent * (level + 2)}if {state_loop} < {rot_dim}:")
        body.append(f"{indent * (level + 3)}if ({state_loop} % 2) == 0:")
        body.append(
            f"{indent * (level + 4)}{b_group}[{group_loop} * {state_dim} + {state_loop}] = "
            f"({b_raw}[{group_loop} * {state_dim} + {state_loop}] * "
            f"T.cos({angle_cumsum}[{group_loop} * {rope_angles} + ({state_loop} // 2)])) - "
            f"({b_raw}[{group_loop} * {state_dim} + {state_loop} + 1] * "
            f"T.sin({angle_cumsum}[{group_loop} * {rope_angles} + ({state_loop} // 2)]))"
        )
        body.append(
            f"{indent * (level + 4)}{c_group}[{group_loop} * {state_dim} + {state_loop}] = "
            f"({c_raw}[{group_loop} * {state_dim} + {state_loop}] * "
            f"T.cos({angle_cumsum}[{group_loop} * {rope_angles} + ({state_loop} // 2)])) - "
            f"({c_raw}[{group_loop} * {state_dim} + {state_loop} + 1] * "
            f"T.sin({angle_cumsum}[{group_loop} * {rope_angles} + ({state_loop} // 2)]))"
        )
        body.append(f"{indent * (level + 3)}else:")
        body.append(
            f"{indent * (level + 4)}{b_group}[{group_loop} * {state_dim} + {state_loop}] = "
            f"({b_raw}[{group_loop} * {state_dim} + {state_loop} - 1] * "
            f"T.sin({angle_cumsum}[{group_loop} * {rope_angles} + ({state_loop} // 2)])) + "
            f"({b_raw}[{group_loop} * {state_dim} + {state_loop}] * "
            f"T.cos({angle_cumsum}[{group_loop} * {rope_angles} + ({state_loop} // 2)]))"
        )
        body.append(
            f"{indent * (level + 4)}{c_group}[{group_loop} * {state_dim} + {state_loop}] = "
            f"({c_raw}[{group_loop} * {state_dim} + {state_loop} - 1] * "
            f"T.sin({angle_cumsum}[{group_loop} * {rope_angles} + ({state_loop} // 2)])) + "
            f"({c_raw}[{group_loop} * {state_dim} + {state_loop}] * "
            f"T.cos({angle_cumsum}[{group_loop} * {rope_angles} + ({state_loop} // 2)]))"
        )
        body.append(f"{indent * (level + 2)}else:")
        body.append(
            f"{indent * (level + 3)}{b_group}[{group_loop} * {state_dim} + {state_loop}] = "
            f"{b_raw}[{group_loop} * {state_dim} + {state_loop}]"
        )
        body.append(
            f"{indent * (level + 3)}{c_group}[{group_loop} * {state_dim} + {state_loop}] = "
            f"{c_raw}[{group_loop} * {state_dim} + {state_loop}]"
        )
        body.append(f"{indent * level}T.sync_threads()")
        body.append(
            f"{indent * level}for {feature} in T.serial(lane, {inner_dim}, "
            f"step={thread_count}):"
        )
        body.append(f"{indent * (level + 1)}{head} = {feature} // {head_dim}")
        body.append(f"{indent * (level + 1)}{hidden_dim} = {feature} % {head_dim}")
        body.append(f"{indent * (level + 1)}{group_loop} = {head} // {heads_per_group}")
        body.append(f"{indent * (level + 1)}{y_skip}[{feature}] = 0.0")
        body.append(
            f"{indent * (level + 1)}for {state_loop} in T.serial(0, {state_dim}):"
        )
        body.append(
            f"{indent * (level + 2)}{state_idx} = "
            f"{head} * {head_dim * state_dim} + {hidden_dim} * {state_dim} + {state_loop}"
        )
        if update_state:
            body.append(
                f"{indent * (level + 2)}{h_next}[{state_idx}] = "
                f"(T.exp({a_vec}[{head}] * {dt_vec}[{head}]) * {h_next}[{state_idx}]) + "
                f"({conv}[{feature}] * {b_group}[{group_loop} * {state_dim} + {state_loop}])"
            )
        body.append(
            f"{indent * (level + 2)}{y_skip}[{feature}] = {y_skip}[{feature}] + "
            f"({h_next}[{state_idx}] * {c_group}[{group_loop} * {state_dim} + {state_loop}])"
        )
        d_skip_expr = _node_indexed_canonical_input_expr(
            node,
            "mamba3_D",
            dtype_by_buffer,
            access_by_buffer,
            head,
            default="1.0",
        )
        body.append(
            f"{indent * (level + 1)}{y_skip}[{feature}] = "
            f"{y_skip}[{feature}] + ({d_skip_expr} * {conv}[{feature}])"
        )
        body.append(
            f"{indent * (level + 1)}{scalar0}[0] = "
            f"1.0 / (1.0 + T.exp(-{projected}[{z_offset} + {feature}]))"
        )
        body.append(
            f"{indent * (level + 1)}{out_inner}[{feature}] = "
            f"{y_skip}[{feature}] * {projected}[{z_offset} + {feature}] * {scalar0}[0]"
        )
        body.append(f"{indent * level}T.sync_threads()")

    body.append(
        f"{indent * 3}# mamba3_mimo_bwd_policy: "
        "exact_lane_parallel_checkpoint_replay"
    )
    spilled_state_scratch = {
        projected,
        project_grad,
        conv_pre,
        conv,
        conv_grad,
        out_inner,
        y_skip,
        out_inner_grad,
        h_prev,
        h_next,
        dh,
        dh_prev,
        b_inv,
        c_inv,
        b_mean,
        c_mean,
        b_raw,
        c_raw,
        b_group,
        c_group,
        b_raw_grad,
        c_raw_grad,
        b_group_grad,
        c_group_grad,
        dt_vec,
        a_vec,
        dt_grad,
        a_grad,
        trap_group,
        trap_grad,
        next_dt_pre_vec,
        next_dt_vec,
        next_trap_vec,
        angle_cumsum,
        angle_grad,
    }
    for name, extent in (
        (projected, in_proj_dim),
        (project_grad, in_proj_dim),
        (conv_pre, conv_channels),
        (conv, conv_channels),
        (conv_grad, conv_channels),
        (out_inner, inner_dim),
        (y_skip, inner_dim),
        (out_inner_grad, inner_dim),
        (h_checkpoint, (checkpoint_count + 1) * state_extent),
        (angle_checkpoint, (checkpoint_count + 1) * angle_extent),
        (h_prev, state_extent),
        (h_next, state_extent),
        (dh, state_extent),
        (dh_prev, state_extent),
        (b_inv, mimo_rank * groups),
        (c_inv, mimo_rank * groups),
        (b_mean, groups * state_dim),
        (c_mean, groups * state_dim),
        (b_raw, groups * state_dim),
        (c_raw, groups * state_dim),
        (b_group, groups * state_dim),
        (c_group, groups * state_dim),
        (b_raw_grad, groups * state_dim),
        (c_raw_grad, groups * state_dim),
        (b_group_grad, groups * state_dim),
        (c_group_grad, groups * state_dim),
        (dt_vec, heads),
        (a_vec, heads),
        (dt_grad, heads),
        (a_grad, heads),
        (trap_group, groups),
        (trap_grad, groups),
        (next_dt_pre_vec, heads),
        (next_dt_vec, heads),
        (next_trap_vec, heads),
        (angle_cumsum, heads * rope_angles),
        (angle_grad, heads * rope_angles),
    ):
        if name in {h_checkpoint, angle_checkpoint}:
            continue
        alloc = (
            "T.alloc_shared"
            if name in spilled_state_scratch
            else "T.alloc_local"
        )
        body.append(f"{indent * 3}{name} = {alloc}(({extent},), \"float32\")")
    for scalar in (
        stage_grad,
        sum_scratch,
        dot_scratch,
        scalar0,
        scalar1,
        scalar2,
        scalar3,
        scalar4,
    ):
        body.append(f"{indent * 3}{scalar} = T.alloc_local((1,), \"float32\")")
    first_row_condition = (
        "path_c_first_row_launch != 0 and row == row_chunk_start"
        if launcher_chunked_rows
        else "row == 0"
    )
    time_rev_range = "row, row + 1" if launcher_chunked_rows else f"0, {sequence_length}"
    body.append(f"{indent * 3}if {first_row_condition}:")
    for output_name, extent in (
        (in_proj_weight_grad, in_proj_dim * hidden_size),
        (out_proj_weight_grad, hidden_size * inner_dim),
        (conv_weight_grad, conv_channels * kernel),
        (conv_bias_grad, conv_channels),
        (dt_bias_grad, heads),
        (b_norm_weight_grad, norm_extent),
        (b_bias_grad, norm_extent),
        (c_norm_weight_grad, norm_extent),
        (c_bias_grad, norm_extent),
        (d_grad, heads),
        (h0_grad, state_extent),
    ):
        if output_name is None:
            continue
        body.append(
            f"{indent * 4}for {grad_flat} in T.serial(lane, {extent}, "
            f"step={thread_count}):"
        )
        body.append(
            f"{indent * 5}{_indexed_buffer_ref(output_name, access_by_buffer, grad_flat)} = 0.0"
        )
    if (
        hidden_grad is not None
        and not _is_full_sequence_bank_slot(
        hidden_grad,
        access_by_buffer,
        )
    ):
        body.append(
            f"{indent * 4}for {grad_flat} in T.serial(lane, "
            f"{sequence_length * hidden_size}, step={thread_count}):"
        )
        body.append(
            f"{indent * 5}{_indexed_buffer_ref(hidden_grad, access_by_buffer, grad_flat)} = 0.0"
        )
    body.append(
        f"{indent * 4}for {state_idx} in T.serial(lane, {state_extent}, "
        f"step={thread_count}):"
    )
    body.append(f"{indent * 5}{dh}[{state_idx}] = 0.0")
    body.append(
        f"{indent * 4}for {grad_flat} in T.serial(lane, {angle_extent}, "
        f"step={thread_count}):"
    )
    body.append(f"{indent * 5}{angle_grad}[{grad_flat}] = 0.0")
    angle_grad_state_ref = _buffer_ref(
        "mamba3_angle_grad_state",
        access_by_buffer,
        grad_flat,
    )
    body.append(f"{indent * 5}{angle_grad_state_ref} = 0.0")
    body.append(f"{indent * 3}T.sync_threads()")
    main_guard = "True" if launcher_chunked_rows else first_row_condition
    body.append(f"{indent * 3}if {main_guard}:")
    body.append(f"{indent * 4}if True:")
    body.append(
        f"{indent * 5}for {grad_flat} in T.serial(lane, {angle_extent}, "
        f"step={thread_count}):"
    )
    body.append(
        f"{indent * 6}{angle_grad}[{grad_flat}] = "
        f"{_buffer_ref('mamba3_angle_grad_state', access_by_buffer, grad_flat)}"
    )
    body.append(
        f"{indent * 5}for {state_idx} in T.serial(lane, {state_extent}, "
        f"step={thread_count}):"
    )
    if h0_grad is not None:
        h0_grad_state_ref = _indexed_buffer_ref(h0_grad, access_by_buffer, state_idx)
        body.append(f"{indent * 6}{dh}[{state_idx}] = {h0_grad_state_ref}")
    else:
        body.append(f"{indent * 6}{dh}[{state_idx}] = 0.0")
    body.append(f"{indent * 5}T.sync_threads()")
    body.append(f"{indent * 5}for {time_rev} in T.serial({time_rev_range}):")
    body.append(f"{indent * 6}{time_idx} = {sequence_length - 1} - {time_rev}")
    body.append(f"{indent * 6}{checkpoint_idx} = {time_idx} // {checkpoint_interval}")
    body.append(
        f"{indent * 6}{checkpoint_start} = {checkpoint_idx} * {checkpoint_interval}"
    )
    body.append(
        f"{indent * 6}for {state_idx} in T.serial(lane, {state_extent}, "
        f"step={thread_count}):"
    )
    h_checkpoint_ref = _buffer_ref(
        "mamba3_h_checkpoint",
        access_by_buffer,
        f"{checkpoint_idx} * {state_extent} + {state_idx}",
    )
    body.append(
        f"{indent * 7}{h_next}[{state_idx}] = "
        f"{h_checkpoint_ref}"
    )
    body.append(
        f"{indent * 6}for {grad_flat} in T.serial(lane, {angle_extent}, "
        f"step={thread_count}):"
    )
    angle_checkpoint_ref = _buffer_ref(
        "mamba3_angle_checkpoint",
        access_by_buffer,
        f"{checkpoint_idx} * {angle_extent} + {grad_flat}",
    )
    body.append(
        f"{indent * 7}{angle_cumsum}[{grad_flat}] = "
        f"{angle_checkpoint_ref}"
    )
    body.append(f"{indent * 6}T.sync_threads()")
    # Reverse-time checkpoint replay. The forward recompute body emitted by
    # ``_mamba3_emit_recompute_row`` replays ``[checkpoint_start, time_idx]``: each
    # ``replay_time < time_idx`` advances the recurrent state (``h_next`` /
    # ``angle_cumsum``) and rebuilds the per-step scratch, and the FINAL step
    # ``replay_time == time_idx`` leaves the current-step scratch (projected /
    # conv / dt_vec / b_group / ...) that the backward block below consumes.
    #
    # Previously this recompute was INLINED TWICE — once in the replay loop for
    # ``replay_time < time_idx`` and once again for ``time_idx`` itself (the
    # ``time_idx`` copy lived OUTSIDE the loop). Those two copies are byte-for-byte
    # identical (~26-28 KB MSL each). Merging them into ONE emission that replays
    # ``[checkpoint_start, time_idx]`` inclusive cuts ~26 KB of MSL (130.0 -> 104.3
    # KB) with EXACTLY the same arithmetic — the ``h_prev`` snapshot is taken at
    # the START of the ``replay_time == time_idx`` iteration, i.e. the
    # post-(time_idx-1) state, exactly as the prior structure. Verified
    # bit-identical gradients (max-abs-diff 0.0). No gradient change; CUDA lowers
    # the smaller body too.
    #
    # NOTE: this merge alone does NOT clear the Metal newComputePipelineState
    # crash. Root cause (isolated by bisection, see commit body): the recompute
    # emits ``T.sync_threads()`` barriers INSIDE this data-dependent replay loop,
    # and a threadgroup_barrier inside the nested checkpoint-replay loop crashes
    # MTLCompilerService (XPC_ERROR_CONNECTION_INTERRUPTED) regardless of size
    # (a barrier-free 25 KB replay scaffold compiles; the 53 KB recompute-in-loop
    # crashes; m2rnn_bwd compiles ONLY because its replay loop is barrier-free).
    # The durable fix is to make the replay recompute barrier-free (each lane
    # replays its own slice independently, m2rnn-style) — tracked as the follow-up.
    body.append(f"{indent * 6}for {replay_offset} in T.serial(0, {checkpoint_interval}):")
    body.append(
        f"{indent * 7}{replay_time} = {checkpoint_start} + {replay_offset}"
    )
    body.append(f"{indent * 7}if {replay_time} <= {time_idx}:")
    body.append(f"{indent * 8}if {replay_time} == {time_idx}:")
    body.append(
        f"{indent * 9}for {state_idx} in T.serial(lane, {state_extent}, "
        f"step={thread_count}):"
    )
    body.append(f"{indent * 10}{h_prev}[{state_idx}] = {h_next}[{state_idx}]")
    body.append(f"{indent * 8}T.sync_threads()")
    _mamba3_emit_recompute_row(
        replay_time,
        level=8,
        update_angle=True,
        update_state=True,
    )
    body.append(
        f"{indent * 6}for {proj_dim} in T.serial(lane, {in_proj_dim}, "
        f"step={thread_count}):"
    )
    body.append(f"{indent * 7}{project_grad}[{proj_dim}] = 0.0")
    body.append(
        f"{indent * 6}for {conv_ch} in T.serial(lane, {conv_channels}, "
        f"step={thread_count}):"
    )
    body.append(f"{indent * 7}{conv_grad}[{conv_ch}] = 0.0")
    body.append(f"{indent * 6}T.sync_threads()")
    body.append(
        f"{indent * 6}for {feature} in T.serial(lane, {inner_dim}, "
        f"step={thread_count}):"
    )
    body.append(f"{indent * 7}{out_inner_grad}[{feature}] = 0.0")
    body.append(f"{indent * 7}for {out_dim} in T.serial(0, {hidden_size}):")
    delta_grad_expr = _node_indexed_canonical_or_positional_input_expr(
        node,
        ("mamba3_delta",),
        0,
        dtype_by_buffer,
        access_by_buffer,
        f"{time_idx} * {hidden_size} + {out_dim}",
    )
    out_weight_expr = _node_indexed_canonical_input_expr(
        node,
        "mamba3_out_proj_weight",
        dtype_by_buffer,
        access_by_buffer,
        f"{out_dim} * {inner_dim} + {feature}",
        default="1.0",
    )
    body.append(
        f"{indent * 8}{stage_grad}[0] = {delta_grad_expr}"
    )
    body.append(
        f"{indent * 8}{out_inner_grad}[{feature}] = "
        f"{out_inner_grad}[{feature}] + ({stage_grad}[0] * {out_weight_expr})"
    )
    if out_proj_weight_grad is not None:
        out_grad_ref = _indexed_buffer_ref(
            out_proj_weight_grad,
            access_by_buffer,
            f"{out_dim} * {inner_dim} + {feature}",
        )
        # Lane-disjoint: lane==feature, address==out_dim*inner_dim+feature, so
        # each lane owns a distinct out_proj_weight_grad column. Serial time loop
        # => non-atomic RMW is byte-identical and avoids the global-atomic stall.
        body.append(
            f"{indent * 8}{out_grad_ref} = {out_grad_ref} + "
            f"({stage_grad}[0] * {out_inner}[{feature}])"
        )
    body.append(f"{indent * 6}T.sync_threads()")
    body.append(
        f"{indent * 6}for {state_idx} in T.serial(lane, {state_extent}, "
        f"step={thread_count}):"
    )
    body.append(f"{indent * 7}{dh_prev}[{state_idx}] = 0.0")
    body.append(
        f"{indent * 6}for {head} in T.serial(lane, {heads}, "
        f"step={thread_count}):"
    )
    body.append(f"{indent * 7}{dt_grad}[{head}] = 0.0")
    body.append(f"{indent * 7}{a_grad}[{head}] = 0.0")
    body.append(
        f"{indent * 6}for {group_loop} in T.serial(lane, {groups}, "
        f"step={thread_count}):"
    )
    body.append(f"{indent * 7}{trap_grad}[{group_loop}] = 0.0")
    body.append(f"{indent * 7}for {state_loop} in T.serial(0, {state_dim}):")
    body.append(
        f"{indent * 8}{b_group_grad}[{group_loop} * {state_dim} + {state_loop}] = 0.0"
    )
    body.append(
        f"{indent * 8}{c_group_grad}[{group_loop} * {state_dim} + {state_loop}] = 0.0"
    )
    body.append(f"{indent * 6}T.sync_threads()")
    body.append(
        f"{indent * 6}for {feature} in T.serial(lane, {inner_dim}, "
        f"step={thread_count}):"
    )
    body.append(f"{indent * 7}{head} = {feature} // {head_dim}")
    body.append(f"{indent * 7}{hidden_dim} = {feature} % {head_dim}")
    body.append(f"{indent * 7}{group_loop} = {head} // {heads_per_group}")
    body.append(
        f"{indent * 7}{scalar0}[0] = "
        f"1.0 / (1.0 + T.exp(-{projected}[{z_offset} + {feature}]))"
    )
    body.append(
        f"{indent * 7}{scalar1}[0] = {out_inner_grad}[{feature}] * {y_skip}[{feature}]"
    )
    body.append(
        f"{indent * 7}{scalar2}[0] = "
        f"{out_inner_grad}[{feature}] * {projected}[{z_offset} + {feature}] * {scalar0}[0]"
    )
    body.append(
        f"{indent * 7}{project_grad}[{z_offset} + {feature}] = "
        f"{project_grad}[{z_offset} + {feature}] + "
        f"({scalar1}[0] * {scalar0}[0] * "
        f"(1.0 + ({projected}[{z_offset} + {feature}] * (1.0 - {scalar0}[0]))))"
    )
    d_skip_expr = _node_indexed_canonical_input_expr(
        node,
        "mamba3_D",
        dtype_by_buffer,
        access_by_buffer,
        head,
        default="1.0",
    )
    if d_grad is not None:
        d_ref = _indexed_buffer_ref(d_grad, access_by_buffer, head)
        body.append(
            f"{indent * 7}T.atomic_add({d_ref}, "
            f"{scalar2}[0] * {conv}[{feature}], memory_order=\"relaxed\")"
        )
    body.append(
        f"{indent * 7}{conv_grad}[{feature}] = {conv_grad}[{feature}] + "
        f"({scalar2}[0] * {d_skip_expr})"
    )
    body.append(f"{indent * 7}for {state_loop} in T.serial(0, {state_dim}):")
    body.append(
        f"{indent * 8}{state_idx} = "
        f"{head} * {head_dim * state_dim} + {hidden_dim} * {state_dim} + {state_loop}"
    )
    body.append(
        f"{indent * 8}{scalar3}[0] = {dh}[{state_idx}] + "
        f"({scalar2}[0] * {c_group}[{group_loop} * {state_dim} + {state_loop}])"
    )
    body.append(
        f"{indent * 8}T.atomic_add("
        f"{c_group_grad}[{group_loop} * {state_dim} + {state_loop}], "
        f"{scalar2}[0] * {h_next}[{state_idx}], memory_order=\"relaxed\")"
    )
    body.append(
        f"{indent * 8}{conv_grad}[{feature}] = {conv_grad}[{feature}] + "
        f"({scalar3}[0] * {b_group}[{group_loop} * {state_dim} + {state_loop}])"
    )
    body.append(
        f"{indent * 8}T.atomic_add("
        f"{b_group_grad}[{group_loop} * {state_dim} + {state_loop}], "
        f"{scalar3}[0] * {conv}[{feature}], memory_order=\"relaxed\")"
    )
    body.append(
        f"{indent * 8}T.atomic_add({a_grad}[{head}], "
        f"{scalar3}[0] * {h_prev}[{state_idx}] * "
        f"T.exp({a_vec}[{head}] * {dt_vec}[{head}]) * {dt_vec}[{head}], "
        "memory_order=\"relaxed\")"
    )
    body.append(
        f"{indent * 8}T.atomic_add({dt_grad}[{head}], "
        f"{scalar3}[0] * {h_prev}[{state_idx}] * "
        f"T.exp({a_vec}[{head}] * {dt_vec}[{head}]) * {a_vec}[{head}], "
        "memory_order=\"relaxed\")"
    )
    body.append(
        f"{indent * 8}{dh_prev}[{state_idx}] = "
        f"{scalar3}[0] * T.exp({a_vec}[{head}] * {dt_vec}[{head}])"
    )
    body.append(f"{indent * 6}T.sync_threads()")
    body.append(
        f"{indent * 6}for {group_loop} in T.serial(lane, {groups}, "
        f"step={thread_count}):"
    )
    body.append(f"{indent * 7}for {state_loop} in T.serial(0, {state_dim}):")
    body.append(
        f"{indent * 8}{b_raw_grad}[{group_loop} * {state_dim} + {state_loop}] = 0.0"
    )
    body.append(
        f"{indent * 8}{c_raw_grad}[{group_loop} * {state_dim} + {state_loop}] = 0.0"
    )
    body.append(f"{indent * 7}for {angle_loop} in T.serial(0, {rope_angles}):")
    body.append(f"{indent * 8}{pair_state} = {angle_loop} * 2")
    body.append(f"{indent * 8}if {pair_state} + 1 < {rot_dim}:")
    body.append(
        f"{indent * 9}{scalar0}[0] = "
        f"T.cos({angle_cumsum}[{group_loop} * {rope_angles} + {angle_loop}])"
    )
    body.append(
        f"{indent * 9}{scalar1}[0] = "
        f"T.sin({angle_cumsum}[{group_loop} * {rope_angles} + {angle_loop}])"
    )
    body.append(
        f"{indent * 9}{scalar2}[0] = "
        f"{b_group_grad}[{group_loop} * {state_dim} + {pair_state}]"
    )
    body.append(
        f"{indent * 9}{scalar3}[0] = "
        f"{b_group_grad}[{group_loop} * {state_dim} + {pair_state} + 1]"
    )
    body.append(
        f"{indent * 9}{b_raw_grad}[{group_loop} * {state_dim} + {pair_state}] = "
        f"{b_raw_grad}[{group_loop} * {state_dim} + {pair_state}] + "
        f"({scalar2}[0] * {scalar0}[0] + {scalar3}[0] * {scalar1}[0])"
    )
    body.append(
        f"{indent * 9}{b_raw_grad}[{group_loop} * {state_dim} + {pair_state} + 1] = "
        f"{b_raw_grad}[{group_loop} * {state_dim} + {pair_state} + 1] + "
        f"((-{scalar2}[0] * {scalar1}[0]) + ({scalar3}[0] * {scalar0}[0]))"
    )
    body.append(
        f"{indent * 9}{angle_grad}[{group_loop} * {rope_angles} + {angle_loop}] = "
        f"{angle_grad}[{group_loop} * {rope_angles} + {angle_loop}] + "
        f"({scalar2}[0] * ((-{b_raw}[{group_loop} * {state_dim} + {pair_state}] * {scalar1}[0]) - "
        f"({b_raw}[{group_loop} * {state_dim} + {pair_state} + 1] * {scalar0}[0])) + "
        f"{scalar3}[0] * ({b_raw}[{group_loop} * {state_dim} + {pair_state}] * {scalar0}[0] - "
        f"{b_raw}[{group_loop} * {state_dim} + {pair_state} + 1] * {scalar1}[0]))"
    )
    body.append(
        f"{indent * 9}{scalar2}[0] = "
        f"{c_group_grad}[{group_loop} * {state_dim} + {pair_state}]"
    )
    body.append(
        f"{indent * 9}{scalar3}[0] = "
        f"{c_group_grad}[{group_loop} * {state_dim} + {pair_state} + 1]"
    )
    body.append(
        f"{indent * 9}{c_raw_grad}[{group_loop} * {state_dim} + {pair_state}] = "
        f"{c_raw_grad}[{group_loop} * {state_dim} + {pair_state}] + "
        f"({scalar2}[0] * {scalar0}[0] + {scalar3}[0] * {scalar1}[0])"
    )
    body.append(
        f"{indent * 9}{c_raw_grad}[{group_loop} * {state_dim} + {pair_state} + 1] = "
        f"{c_raw_grad}[{group_loop} * {state_dim} + {pair_state} + 1] + "
        f"((-{scalar2}[0] * {scalar1}[0]) + ({scalar3}[0] * {scalar0}[0]))"
    )
    body.append(
        f"{indent * 9}{angle_grad}[{group_loop} * {rope_angles} + {angle_loop}] = "
        f"{angle_grad}[{group_loop} * {rope_angles} + {angle_loop}] + "
        f"({scalar2}[0] * ((-{c_raw}[{group_loop} * {state_dim} + {pair_state}] * {scalar1}[0]) - "
        f"({c_raw}[{group_loop} * {state_dim} + {pair_state} + 1] * {scalar0}[0])) + "
        f"{scalar3}[0] * ({c_raw}[{group_loop} * {state_dim} + {pair_state}] * {scalar0}[0] - "
        f"{c_raw}[{group_loop} * {state_dim} + {pair_state} + 1] * {scalar1}[0]))"
    )
    body.append(f"{indent * 7}for {state_loop} in T.serial({rot_dim}, {state_dim}):")
    body.append(
        f"{indent * 8}{b_raw_grad}[{group_loop} * {state_dim} + {state_loop}] = "
        f"{b_raw_grad}[{group_loop} * {state_dim} + {state_loop}] + "
        f"{b_group_grad}[{group_loop} * {state_dim} + {state_loop}]"
    )
    body.append(
        f"{indent * 8}{c_raw_grad}[{group_loop} * {state_dim} + {state_loop}] = "
        f"{c_raw_grad}[{group_loop} * {state_dim} + {state_loop}] + "
        f"{c_group_grad}[{group_loop} * {state_dim} + {state_loop}]"
    )
    body.append(f"{indent * 6}T.sync_threads()")
    body.append(
        f"{indent * 6}for {group_loop} in T.serial(lane, {groups}, "
        f"step={thread_count}):"
    )
    body.append(f"{indent * 7}for {state_loop} in T.serial(0, {state_dim}):")
    body.append(
        f"{indent * 8}{trap_grad}[{group_loop}] = {trap_grad}[{group_loop}] + "
        f"({b_raw_grad}[{group_loop} * {state_dim} + {state_loop}] * "
        f"{b_mean}[{group_loop} * {state_dim} + {state_loop}])"
    )
    body.append(f"{indent * 6}T.sync_threads()")
    body.append(
        f"{indent * 6}for {rank_loop} in T.serial(lane, {mimo_rank}, "
        f"step={thread_count}):"
    )
    body.append(f"{indent * 7}for {group_loop} in T.serial(0, {groups}):")
    body.append(f"{indent * 8}{sum_scratch}[0] = 0.0")
    body.append(f"{indent * 8}{dot_scratch}[0] = 0.0")
    body.append(f"{indent * 8}for {state_loop} in T.serial(0, {state_dim}):")
    body.append(
        f"{indent * 9}{grad_flat} = "
        f"({rank_loop} * {groups} + {group_loop}) * {state_dim} + {state_loop}"
    )
    b_norm_expr = _node_indexed_canonical_input_expr(
        node,
        "mamba3_B_norm_weight",
        dtype_by_buffer,
        access_by_buffer,
        grad_flat,
        default="1.0",
    )
    c_norm_expr = _node_indexed_canonical_input_expr(
        node,
        "mamba3_C_norm_weight",
        dtype_by_buffer,
        access_by_buffer,
        grad_flat,
        default="1.0",
    )
    body.append(
        f"{indent * 9}{sum_scratch}[0] = {sum_scratch}[0] + "
        f"(({b_raw_grad}[{group_loop} * {state_dim} + {state_loop}] * "
        f"{trap_group}[{group_loop}] / {float(mimo_rank):.1f}) * {b_norm_expr} * "
        f"{conv}[{conv_b_offset} + {grad_flat}])"
    )
    body.append(
        f"{indent * 9}{dot_scratch}[0] = {dot_scratch}[0] + "
        f"(({c_raw_grad}[{group_loop} * {state_dim} + {state_loop}] / "
        f"{float(mimo_rank):.1f}) * {c_norm_expr} * {conv}[{conv_c_offset} + {grad_flat}])"
    )
    body.append(f"{indent * 8}for {state_loop} in T.serial(0, {state_dim}):")
    body.append(
        f"{indent * 9}{grad_flat} = "
        f"({rank_loop} * {groups} + {group_loop}) * {state_dim} + {state_loop}"
    )
    body.append(
        f"{indent * 9}{scalar0}[0] = "
        f"{b_raw_grad}[{group_loop} * {state_dim} + {state_loop}] * "
        f"{trap_group}[{group_loop}] / {float(mimo_rank):.1f}"
    )
    if b_norm_weight_grad is not None:
        b_norm_grad_ref = _indexed_buffer_ref(
            b_norm_weight_grad,
            access_by_buffer,
            grad_flat,
        )
        # Lane-disjoint: lane==rank_loop, grad_flat=(rank_loop*groups+.)*sd+.
        # so each lane owns distinct addresses; serial time loop => RMW.
        body.append(
            f"{indent * 9}{b_norm_grad_ref} = {b_norm_grad_ref} + "
            f"({scalar0}[0] * {conv}[{conv_b_offset} + {grad_flat}] * "
            f"{b_inv}[{rank_loop} * {groups} + {group_loop}])"
        )
    if b_bias_grad is not None:
        b_bias_grad_ref = _indexed_buffer_ref(b_bias_grad, access_by_buffer, grad_flat)
        body.append(
            f"{indent * 9}{b_bias_grad_ref} = {b_bias_grad_ref} + {scalar0}[0]"
        )
    body.append(
        f"{indent * 9}{conv_grad}[{conv_b_offset} + {grad_flat}] = "
        f"{conv_grad}[{conv_b_offset} + {grad_flat}] + "
        f"({b_inv}[{rank_loop} * {groups} + {group_loop}] * "
        f"(({scalar0}[0] * {b_norm_expr}) - "
        f"({conv}[{conv_b_offset} + {grad_flat}] * {sum_scratch}[0] * "
        f"{b_inv}[{rank_loop} * {groups} + {group_loop}] * "
        f"{b_inv}[{rank_loop} * {groups} + {group_loop}] / {float(state_dim):.1f})))"
    )
    body.append(
        f"{indent * 9}{scalar1}[0] = "
        f"{c_raw_grad}[{group_loop} * {state_dim} + {state_loop}] / {float(mimo_rank):.1f}"
    )
    if c_norm_weight_grad is not None:
        c_norm_grad_ref = _indexed_buffer_ref(
            c_norm_weight_grad,
            access_by_buffer,
            grad_flat,
        )
        # Lane-disjoint (lane==rank_loop); serial time loop => non-atomic RMW.
        body.append(
            f"{indent * 9}{c_norm_grad_ref} = {c_norm_grad_ref} + "
            f"({scalar1}[0] * {conv}[{conv_c_offset} + {grad_flat}] * "
            f"{c_inv}[{rank_loop} * {groups} + {group_loop}])"
        )
    if c_bias_grad is not None:
        c_bias_grad_ref = _indexed_buffer_ref(c_bias_grad, access_by_buffer, grad_flat)
        body.append(
            f"{indent * 9}{c_bias_grad_ref} = {c_bias_grad_ref} + {scalar1}[0]"
        )
    body.append(
        f"{indent * 9}{conv_grad}[{conv_c_offset} + {grad_flat}] = "
        f"{conv_grad}[{conv_c_offset} + {grad_flat}] + "
        f"({c_inv}[{rank_loop} * {groups} + {group_loop}] * "
        f"(({scalar1}[0] * {c_norm_expr}) - "
        f"({conv}[{conv_c_offset} + {grad_flat}] * {dot_scratch}[0] * "
        f"{c_inv}[{rank_loop} * {groups} + {group_loop}] * "
        f"{c_inv}[{rank_loop} * {groups} + {group_loop}] / {float(state_dim):.1f})))"
    )
    body.append(f"{indent * 6}T.sync_threads()")
    body.append(
        f"{indent * 6}for {hidden_dim} in T.serial(lane, {heads}, "
        f"step={thread_count}):"
    )
    body.append(f"{indent * 7}{group_loop} = {hidden_dim} // {heads_per_group}")
    body.append(f"{indent * 7}if True:")
    body.append(f"{indent * 8}{scalar0}[0] = {trap_grad}[{group_loop}] / {float(heads_per_group):.1f}")
    body.append(
        f"{indent * 8}{scalar1}[0] = "
        f"1.0 / (1.0 + T.exp(-{projected}[{trap_offset} + {hidden_dim}]))"
    )
    body.append(
        f"{indent * 8}{dt_grad}[{hidden_dim}] = {dt_grad}[{hidden_dim}] + "
        f"({scalar0}[0] * {scalar1}[0])"
    )
    body.append(
        f"{indent * 8}{project_grad}[{trap_offset} + {hidden_dim}] = "
        f"{project_grad}[{trap_offset} + {hidden_dim}] + "
        f"({scalar0}[0] * {dt_vec}[{hidden_dim}] * {scalar1}[0] * (1.0 - {scalar1}[0]))"
    )
    body.append(f"{indent * 8}if {time_idx} + 1 < {sequence_length}:")
    body.append(
        f"{indent * 9}{scalar2}[0] = "
        f"1.0 / (1.0 + T.exp(-{next_trap_vec}[{hidden_dim}]))"
    )
    body.append(
        f"{indent * 9}{scalar3}[0] = "
        f"{scalar0}[0] * (1.0 - {scalar2}[0]) * "
        f"(1.0 / (1.0 + T.exp(-{next_dt_pre_vec}[{hidden_dim}])))"
    )
    body.append(
        f"{indent * 9}{scalar4}[0] = "
        f"{scalar0}[0] * {next_dt_vec}[{hidden_dim}] * "
        f"(-{scalar2}[0] * (1.0 - {scalar2}[0]))"
    )
    if dt_bias_grad is not None:
        dt_bias_grad_ref = _indexed_buffer_ref(dt_bias_grad, access_by_buffer, hidden_dim)
        # Lane-disjoint (lane==hidden_dim); serial time loop => non-atomic RMW.
        body.append(
            f"{indent * 9}{dt_bias_grad_ref} = {dt_bias_grad_ref} + {scalar3}[0]"
        )
    body.append(f"{indent * 9}for {hidden_loop} in T.serial(0, {hidden_size}):")
    if in_proj_weight_grad is not None:
        next_dt_grad_ref = _indexed_buffer_ref(
            in_proj_weight_grad,
            access_by_buffer,
            f"({dt_offset} + {hidden_dim}) * {hidden_size} + {hidden_loop}",
        )
        next_trap_grad_ref = _indexed_buffer_ref(
            in_proj_weight_grad,
            access_by_buffer,
            f"({trap_offset} + {hidden_dim}) * {hidden_size} + {hidden_loop}",
        )
        # Lane-disjoint (lane==hidden_dim => distinct rows dt_offset+hidden_dim /
        # trap_offset+hidden_dim). The later main scatter touches the same rows
        # only across a sync_threads barrier (sequential) => non-atomic RMW.
        body.append(
            f"{indent * 10}{next_dt_grad_ref} = {next_dt_grad_ref} + "
            f"({_mamba3_hidden_expr(f'({time_idx} + 1)', hidden_loop)} * {scalar3}[0])"
        )
        body.append(
            f"{indent * 10}{next_trap_grad_ref} = {next_trap_grad_ref} + "
            f"({_mamba3_hidden_expr(f'({time_idx} + 1)', hidden_loop)} * {scalar4}[0])"
        )
    if hidden_grad is not None:
        hidden_grad_ref = _indexed_buffer_ref(
            hidden_grad,
            access_by_buffer,
            f"({time_idx} + 1) * {hidden_size} + {hidden_loop}",
        )
        body.append(
            f"{indent * 10}T.atomic_add({hidden_grad_ref}, "
            f"{scalar3}[0] * {_mamba3_project_weight_expr(f'{dt_offset} + {hidden_dim}', hidden_loop)} + "
            f"{scalar4}[0] * {_mamba3_project_weight_expr(f'{trap_offset} + {hidden_dim}', hidden_loop)}, "
            "memory_order=\"relaxed\")"
        )
    body.append(f"{indent * 6}T.sync_threads()")
    body.append(
        f"{indent * 6}for {head} in T.serial(lane, {heads}, "
        f"step={thread_count}):"
    )
    dt_bias_expr_head = _node_indexed_canonical_input_expr(
        node,
        "mamba3_dt_bias",
        dtype_by_buffer,
        access_by_buffer,
        head,
        default="0.0",
    )
    body.append(f"{indent * 7}for {angle_loop} in T.serial(0, {rope_angles}):")
    body.append(
        f"{indent * 8}T.atomic_add({project_grad}[{angle_offset} + {angle_loop}], "
        f"{angle_grad}[{head} * {rope_angles} + {angle_loop}] * {dt_vec}[{head}], "
        "memory_order=\"relaxed\")"
    )
    body.append(
        f"{indent * 8}{dt_grad}[{head}] = {dt_grad}[{head}] + "
        f"({angle_grad}[{head} * {rope_angles} + {angle_loop}] * "
        f"{projected}[{angle_offset} + {angle_loop}])"
    )
    body.append(
        f"{indent * 8}{angle_cumsum}[{head} * {rope_angles} + {angle_loop}] = "
        f"{angle_cumsum}[{head} * {rope_angles} + {angle_loop}] - "
        f"({projected}[{angle_offset} + {angle_loop}] * {dt_vec}[{head}])"
    )
    body.append(
        f"{indent * 7}{scalar0}[0] = "
        f"1.0 / (1.0 + T.exp(-({projected}[{dt_offset} + {head}] + {dt_bias_expr_head})))"
    )
    body.append(
        f"{indent * 7}{project_grad}[{dt_offset} + {head}] = "
        f"{project_grad}[{dt_offset} + {head}] + ({dt_grad}[{head}] * {scalar0}[0])"
    )
    if dt_bias_grad is not None:
        dt_bias_grad_ref = _indexed_buffer_ref(dt_bias_grad, access_by_buffer, head)
        # Lane-disjoint (lane==head); separated from the earlier dt_bias scatter
        # by a sync_threads => non-atomic RMW accumulates correctly.
        body.append(
            f"{indent * 7}{dt_bias_grad_ref} = {dt_bias_grad_ref} + "
            f"({dt_grad}[{head}] * {scalar0}[0])"
        )
    body.append(
        f"{indent * 7}if -T.log(1.0 + T.exp({projected}[{a_offset} + {head}])) < -0.01:"
    )
    body.append(
        f"{indent * 8}{project_grad}[{a_offset} + {head}] = "
        f"{project_grad}[{a_offset} + {head}] + "
        f"({a_grad}[{head}] * (-(1.0 / (1.0 + T.exp(-{projected}[{a_offset} + {head}])))))"
    )
    body.append(f"{indent * 6}T.sync_threads()")
    body.append(
        f"{indent * 6}for {conv_ch} in T.serial(lane, {conv_channels}, "
        f"step={thread_count}):"
    )
    body.append(
        f"{indent * 7}{scalar0}[0] = 1.0 / (1.0 + T.exp(-{conv_pre}[{conv_ch}]))"
    )
    body.append(
        f"{indent * 7}{scalar1}[0] = {conv_grad}[{conv_ch}] * {scalar0}[0] * "
        f"(1.0 + ({conv_pre}[{conv_ch}] * (1.0 - {scalar0}[0])))"
    )
    if conv_bias_grad is not None:
        conv_bias_ref = _indexed_buffer_ref(conv_bias_grad, access_by_buffer, conv_ch)
        # Lane-disjoint (lane==conv_ch); serial time loop => non-atomic RMW.
        body.append(
            f"{indent * 7}{conv_bias_ref} = {conv_bias_ref} + {scalar1}[0]"
        )
    if history_len > 0:
        body.append(f"{indent * 7}for {kernel_pos} in T.serial(0, {history_len}):")
        body.append(
            f"{indent * 8}{src_row} = {time_idx} - {history_len} + {kernel_pos}"
        )
        body.append(f"{indent * 8}if {src_row} >= 0:")
        body.append(f"{indent * 9}{scalar2}[0] = 0.0")
        body.append(f"{indent * 9}for {hidden_loop} in T.serial(0, {hidden_size}):")
        body.append(
            f"{indent * 10}{scalar2}[0] = {scalar2}[0] + "
            f"({_mamba3_hidden_expr(src_row, hidden_loop)} * "
            f"{_mamba3_project_weight_expr(f'{x_offset} + {conv_ch}', hidden_loop)})"
        )
        body.append(f"{indent * 8}else:")
        body.append(f"{indent * 9}{src_hist} = {kernel_pos} + {time_idx}")
        conv_state_expr = _node_indexed_canonical_input_expr(
            node,
            "mamba3_conv_state",
            dtype_by_buffer,
            access_by_buffer,
            f"{src_hist} * {conv_channels} + {conv_ch}",
            default="0.0",
        )
        body.append(f"{indent * 9}{scalar2}[0] = {conv_state_expr}")
        if conv_weight_grad is not None:
            conv_weight_grad_ref = _indexed_buffer_ref(
                conv_weight_grad,
                access_by_buffer,
                f"{conv_ch} * {kernel} + {kernel_pos}",
            )
            # Lane-disjoint: lane==conv_ch owns row conv_ch*kernel+. ; serial time
            # loop => non-atomic RMW byte-identical to the atomic.
            body.append(
                f"{indent * 8}{conv_weight_grad_ref} = {conv_weight_grad_ref} + "
                f"({scalar1}[0] * {scalar2}[0])"
            )
        body.append(f"{indent * 8}if {src_row} >= 0:")
        body.append(f"{indent * 9}for {hidden_loop} in T.serial(0, {hidden_size}):")
        if in_proj_weight_grad is not None:
            in_proj_grad_ref = _indexed_buffer_ref(
                in_proj_weight_grad,
                access_by_buffer,
                f"({x_offset} + {conv_ch}) * {hidden_size} + {hidden_loop}",
            )
            # Lane-disjoint: lane==conv_ch => row (x_offset+conv_ch)*H+. ; the
            # main in-proj scatter (a sync_threads later) hits the same rows only
            # SEQUENTIALLY, so non-atomic RMW accumulates correctly.
            body.append(
                f"{indent * 10}{in_proj_grad_ref} = {in_proj_grad_ref} + "
                f"({_mamba3_hidden_expr(src_row, hidden_loop)} * {scalar1}[0] * "
                f"{_mamba3_conv_weight_expr(conv_ch, kernel_pos)})"
            )
        if hidden_grad is not None:
            hidden_grad_ref = _indexed_buffer_ref(
                hidden_grad,
                access_by_buffer,
                f"{src_row} * {hidden_size} + {hidden_loop}",
            )
            body.append(
                f"{indent * 10}T.atomic_add({hidden_grad_ref}, "
                f"{scalar1}[0] * {_mamba3_conv_weight_expr(conv_ch, kernel_pos)} * "
                f"{_mamba3_project_weight_expr(f'{x_offset} + {conv_ch}', hidden_loop)}, "
                "memory_order=\"relaxed\")"
            )
    if conv_weight_grad is not None:
        current_conv_weight_grad_ref = _indexed_buffer_ref(
            conv_weight_grad,
            access_by_buffer,
            f"{conv_ch} * {kernel} + {history_len}",
        )
        # Lane-disjoint (lane==conv_ch); serial time loop => non-atomic RMW.
        body.append(
            f"{indent * 7}{current_conv_weight_grad_ref} = "
            f"{current_conv_weight_grad_ref} + "
            f"({scalar1}[0] * {projected}[{x_offset} + {conv_ch}])"
        )
    body.append(
        f"{indent * 7}{project_grad}[{x_offset} + {conv_ch}] = "
        f"{project_grad}[{x_offset} + {conv_ch}] + "
        f"({scalar1}[0] * {_mamba3_conv_weight_expr(conv_ch, str(history_len))})"
    )
    body.append(f"{indent * 6}T.sync_threads()")
    body.append(
        f"{indent * 6}for {proj_dim} in T.serial(lane, {in_proj_dim}, "
        f"step={thread_count}):"
    )
    body.append(f"{indent * 7}for {hidden_loop} in T.serial(0, {hidden_size}):")
    if in_proj_weight_grad is not None:
        in_proj_grad_ref = _indexed_buffer_ref(
            in_proj_weight_grad,
            access_by_buffer,
            f"{proj_dim} * {hidden_size} + {hidden_loop}",
        )
        # Lane-disjoint scatter: lane==proj_dim, so each lane owns a distinct
        # in_proj_weight_grad ROW (proj_dim*H + .) and the serial time loop runs
        # one step at a time. No two threads ever target the same address within
        # a launch -> a non-atomic read-modify-write is byte-identical to the
        # atomic and ~10x cheaper (relaxed global atomics serialize on Metal).
        body.append(
            f"{indent * 8}{in_proj_grad_ref} = {in_proj_grad_ref} + "
            f"({_mamba3_hidden_expr(time_idx, hidden_loop)} * {project_grad}[{proj_dim}])"
        )
    if hidden_grad is not None:
        # hidden_grad[time_idx*H + hidden_loop]: ALL proj_dim lanes accumulate
        # into the same hidden_loop column -> lanes COLLIDE; keep atomic.
        hidden_grad_ref = _indexed_buffer_ref(
            hidden_grad,
            access_by_buffer,
            f"{time_idx} * {hidden_size} + {hidden_loop}",
        )
        body.append(
            f"{indent * 8}T.atomic_add({hidden_grad_ref}, "
            f"{project_grad}[{proj_dim}] * {_mamba3_project_weight_expr(proj_dim, hidden_loop)}, "
            "memory_order=\"relaxed\")"
        )
    body.append(f"{indent * 6}T.sync_threads()")
    body.append(
        f"{indent * 6}for {state_idx} in T.serial(lane, {state_extent}, "
        f"step={thread_count}):"
    )
    body.append(f"{indent * 7}{dh}[{state_idx}] = {dh_prev}[{state_idx}]")
    body.append(f"{indent * 7}{h_next}[{state_idx}] = {h_prev}[{state_idx}]")
    body.append(f"{indent * 6}T.sync_threads()")
    if h0_grad is not None:
        body.append(
            f"{indent * 5}for {state_idx} in T.serial(lane, {state_extent}, "
            f"step={thread_count}):"
        )
        h0_grad_ref = _indexed_buffer_ref(h0_grad, access_by_buffer, state_idx)
        body.append(f"{indent * 6}{h0_grad_ref} = {dh}[{state_idx}]")
    body.append(
        f"{indent * 5}for {grad_flat} in T.serial(lane, {angle_extent}, "
        f"step={thread_count}):"
    )
    angle_grad_state_ref = _buffer_ref(
        "mamba3_angle_grad_state",
        access_by_buffer,
        grad_flat,
    )
    body.append(f"{indent * 6}{angle_grad_state_ref} = {angle_grad}[{grad_flat}]")
    body.append(f"{indent * 3}T.sync_threads()")
    return


def _descriptor_node_source(
    *,
    node: _ScheduleNodeView,
    node_index: int,
    descriptor: PathCBrickScheduleDescriptor,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
) -> _ScheduleNodeFragment:
    if node.op_name in _EXACT_ROW_PHASED_BACKWARD_OPS:
        return _ScheduleNodeFragment(allocations=(), statements=())
    if descriptor.fragment_emitter is not None:
        fragment = descriptor.fragment_emitter(
            node=node,
            dtype_by_buffer=dtype_by_buffer,
            access_by_buffer=access_by_buffer,
        )
        if not isinstance(fragment, _ScheduleNodeFragment):
            raise TypeError(
                f"fragment emitter for {descriptor.op_name!r} must return "
                "_ScheduleNodeFragment"
            )
        return fragment
    if node.op_name.endswith("_bwd"):
        return _emit_owner_output_backward_source(
            node,
            dtype_by_buffer,
            access_by_buffer,
        )
    return _emit_generic_descriptor_source(
        node,
        node_index,
        dtype_by_buffer,
        access_by_buffer,
    )


def _validated_buffer_extent(buffer_extent: int) -> int:
    extent = int(buffer_extent)
    if extent <= 0:
        raise ValueError("descriptor schedule buffer_extent must be positive")
    return extent


def _validated_max_rows_per_launch(max_rows_per_launch: int | None) -> int | None:
    if max_rows_per_launch is None:
        return None
    rows = int(max_rows_per_launch)
    if rows <= 0:
        raise ValueError("max_rows_per_launch must be positive when provided")
    return rows


def _validated_rows_per_kernel_launch(rows_per_kernel_launch: int | None) -> int:
    rows = (
        DESCRIPTOR_ROWS_PER_KERNEL_LAUNCH
        if rows_per_kernel_launch is None
        else int(rows_per_kernel_launch)
    )
    if rows <= 0:
        raise ValueError("rows_per_kernel_launch must be positive")
    return rows


def _validated_row_dispatch_mode(row_dispatch_mode: str) -> str:
    mode = str(row_dispatch_mode)
    if mode not in {
        DESCRIPTOR_ROW_DISPATCH_GRID_CHUNKS,
        DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS,
    }:
        raise ValueError(
            "row_dispatch_mode must be one of "
            f"{DESCRIPTOR_ROW_DISPATCH_GRID_CHUNKS!r}, "
            f"{DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS!r}"
        )
    return mode


def _validated_execution_stage(execution_stage: str) -> str:
    stage = str(execution_stage)
    if stage not in DESCRIPTOR_EXECUTION_STAGES:
        raise ValueError(
            "execution_stage must be one of "
            f"{sorted(DESCRIPTOR_EXECUTION_STAGES)!r}"
        )
    return stage


def _row_phased_chunk_count(
    *,
    loop_policy: str,
    shape_env: PathCModelShapeEnv | None,
    max_rows_per_launch: int | None,
) -> int | None:
    if (
        loop_policy != DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN
        or shape_env is None
        or max_rows_per_launch is None
    ):
        return None
    sequence_length = int(shape_env.sequence_length)
    rows_per_launch = _validated_max_rows_per_launch(max_rows_per_launch)
    assert rows_per_launch is not None
    return max(1, (sequence_length + rows_per_launch - 1) // rows_per_launch)


def _descriptor_loop_extent(
    external_buffers: Sequence[str],
    buffer_extent: int,
    shape_env: PathCModelShapeEnv | None,
) -> int:
    if shape_env is None:
        return buffer_extent
    if any(_canonical_buffer_name(name) == "hidden" for name in external_buffers):
        return shape_env.sequence_length * shape_env.hidden_size
    return buffer_extent


def _loop_indexed_buffer_ref(
    buffer_name: str,
    shape: Sequence[int],
    loop_extent: int,
    shape_env: PathCModelShapeEnv | None,
) -> str:
    name = _safe_identifier(buffer_name)
    flat_extent = _flattened_extent(shape)
    if flat_extent <= 1:
        return f"{name}[0]"
    if flat_extent >= loop_extent:
        return f"{name}[i]"
    if shape_env is None:
        return f"{name}[0]"
    canonical_name = _canonical_buffer_name(buffer_name)
    if canonical_name == "q_scale":
        return f"{name}[(i % {shape_env.hidden_size}) // {shape_env.attention_head_dim}]"
    if canonical_name == "kv_scale":
        if flat_extent == shape_env.sequence_length * shape_env.attention_num_kv_heads:
            return (
                f"{name}[(i // {shape_env.hidden_size}) * "
                f"{shape_env.attention_num_kv_heads} + "
                f"(((i % {shape_env.hidden_size}) // "
                f"{shape_env.attention_head_dim}) % "
                f"{shape_env.attention_num_kv_heads})]"
            )
        return (
            f"{name}[((i % {shape_env.hidden_size}) // "
            f"{shape_env.attention_head_dim}) % {shape_env.attention_num_kv_heads}]"
        )
    if canonical_name == "kv_fp8":
        kv_width = shape_env.attention_num_kv_heads * shape_env.attention_head_dim
        if flat_extent == shape_env.sequence_length * kv_width:
            return (
                f"{name}[(i // {shape_env.hidden_size}) * "
                f"{kv_width} + (i % {kv_width})]"
            )
        return (
            f"{name}[i % "
            f"{kv_width}]"
        )
    if canonical_name == "indices":
        row_width = shape_env.attention_num_kv_heads * shape_env.attention_sparse_topk
        if flat_extent == shape_env.sequence_length * row_width:
            return (
                f"{name}[(i // {shape_env.hidden_size}) * {row_width} + "
                f"(i % {row_width})]"
            )
        return (
            f"{name}[i % "
            f"{row_width}]"
        )
    if canonical_name == "lse":
        return (
            f"{name}[(i // {shape_env.hidden_size}) * "
            f"{shape_env.attention_num_q_heads} + "
            f"((i % {shape_env.hidden_size}) // {shape_env.attention_head_dim})]"
        )
    if canonical_name in {"target_ids", "target_mask"}:
        return f"{name}[i // {shape_env.hidden_size}]"
    if flat_extent == shape_env.hidden_size or canonical_name in {
        "attention_q_proj_bias",
        "attention_out_proj_bias",
        "mamba3_residual_to_m2rnn_norm_weight",
        "m2rnn_residual_to_attention_norm_weight",
        "residual_norm_weight",
        "final_norm_weight",
        # Block A: per-brick entry RMSNorm weight is also a length-H vector
        # whose values index over the hidden dimension on every row.
        "entry_rmsnorm_weight",
    }:
        return f"{name}[i % {shape_env.hidden_size}]"
    if canonical_name == "sparse_mla_sinks":
        return f"{name}[i % {shape_env.attention_num_q_heads}]"
    return f"{name}[i % {flat_extent}]"


def _emit_mamba3_mimo_source(
    node: _ScheduleNodeView,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
) -> _ScheduleNodeFragment:
    proj = _scratch_name(node, "mamba3_proj")
    conv = _scratch_name(node, "mamba3_conv")
    dt = _scratch_name(node, "mamba3_dt")
    state = _scratch_name(node, "mamba3_state")
    out = _scratch_name(node, "mamba3_out")
    delta = _output_with_suffix(node, "_delta") or node.outputs[0]
    index = "i"
    hidden = _node_input_expr(node, 0, dtype_by_buffer, access_by_buffer, index)
    in_proj_weight = _optional_buffer_expr(
        "mamba3_in_proj_weight",
        dtype_by_buffer,
        access_by_buffer,
        index=index,
    )
    conv_weight = _optional_buffer_expr(
        "mamba3_conv_weight",
        dtype_by_buffer,
        access_by_buffer,
        "1.0",
        index,
    )
    conv_bias = _optional_buffer_expr(
        "mamba3_conv_bias",
        dtype_by_buffer,
        access_by_buffer,
        index=index,
    )
    dt_bias = _optional_buffer_expr(
        "mamba3_dt_bias",
        dtype_by_buffer,
        access_by_buffer,
        index=index,
    )
    d_skip = _optional_buffer_expr(
        "mamba3_D",
        dtype_by_buffer,
        access_by_buffer,
        "1.0",
        index,
    )
    h0 = _optional_buffer_expr(
        "mamba3_h0",
        dtype_by_buffer,
        access_by_buffer,
        index=index,
    )
    b_norm_weight = _optional_buffer_expr(
        "mamba3_B_norm_weight",
        dtype_by_buffer,
        access_by_buffer,
        "1.0",
        index,
    )
    b_bias = _optional_buffer_expr(
        "mamba3_B_bias",
        dtype_by_buffer,
        access_by_buffer,
        index=index,
    )
    c_norm_weight = _optional_buffer_expr(
        "mamba3_C_norm_weight",
        dtype_by_buffer,
        access_by_buffer,
        "1.0",
        index,
    )
    c_bias = _optional_buffer_expr(
        "mamba3_C_bias",
        dtype_by_buffer,
        access_by_buffer,
        index=index,
    )
    out_proj_weight = _optional_buffer_expr(
        "mamba3_out_proj_weight",
        dtype_by_buffer,
        access_by_buffer,
        "1.0",
        index,
    )
    inner = [
        f"{proj}[0] = {hidden} * {in_proj_weight}",
        f"{conv}[0] = ({proj}[0] * {conv_weight}) + {conv_bias}",
        f"{dt}[0] = T.log(1.0 + T.exp({proj}[0] + {dt_bias}))",
        f"{state}[0] = ({h0} * T.exp(-{dt}[0])) + "
        f"(({conv}[0] * {d_skip}) * {b_norm_weight}) + {b_bias}",
        f"{out}[0] = (({state}[0] * {c_norm_weight}) + {c_bias}) * "
        f"{out_proj_weight}",
        f"{_buffer_ref(delta, access_by_buffer, index)} = {out}[0]",
    ]
    for output_name in node.outputs:
        if output_name == delta:
            continue
        source = state if _canonical_buffer_name(output_name) == "scan_state" else out
        inner.append(
            f"{_buffer_ref(output_name, access_by_buffer, index)} = "
            f"{source}[0]"
        )
    return _ScheduleNodeFragment(
        allocations=(
            f"{proj} = T.alloc_local((1,), \"float32\")",
            f"{conv} = T.alloc_local((1,), \"float32\")",
            f"{dt} = T.alloc_local((1,), \"float32\")",
            f"{state} = T.alloc_local((1,), \"float32\")",
            f"{out} = T.alloc_local((1,), \"float32\")",
        ),
        statements=tuple(inner),
    )


def _emit_mamba3_chunk_scan_combine_source(
    node: _ScheduleNodeView,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
) -> _ScheduleNodeFragment:
    """Fragment emitter for the Path-C F2 ``mamba3_chunk_scan_combine`` segment.

    Design: ``docs/MAMBA3-PATHC-MULTIKERNEL-DESIGN.md`` §2 (F2 row) / §3.2.

    F2 is NOT a row-phased single-thread fragment inlined into the shared region
    PrimFunc: its kernel is the GRID-launched, Metal-validated SSD scan+combine
    core ``chunk_scan_fwd_metal_prim`` (own ``T.Kernel(nheads, ...)`` grid). Its
    real codegen is the single delegation
    ``mamba3_chunked_scan_core.build_chunk_scan_combine_metal`` — exercised + parity
    gated in isolation by the Stage-1 harness
    ``tests/test_mamba3_chunk_scan_combine_f2.py`` against the serial forward.

    Stage 1 is a SHADOW registration: the live mamba3 forward still emits the
    serial scan (``mamba3_mimo`` descriptor); this descriptor only needs to RESOLVE
    via ``select`` so its op-name signature is not blocked with
    ``no descriptor target``. We therefore return a marker fragment that records the
    delegation source. Wiring this fragment into the fused chain template is Stage 2
    (region-build flag + F0/F1 handoff), out of Stage-1 scope; until then this
    descriptor is never selected for the live region (its op-name is not emitted by
    region build), so the marker is never inlined. RULE #1: there is no silent
    fallback — the kernel path is the one delegation above, and a future Stage-2
    template hookup MUST route through it, not re-emit a serial scan here.
    """
    marker = _scratch_name(node, "mamba3_chunk_scan_combine_shadow")
    return _ScheduleNodeFragment(
        allocations=(f"{marker} = T.alloc_local((1,), \"float32\")",),
        statements=(
            "# F2 mamba3_chunk_scan_combine: delegates to "
            "cppmega_mlx.nn._tilelang.mamba3_chunked_scan_core."
            "build_chunk_scan_combine_metal (grid scan+combine core); "
            "shadow-registered (Stage 1), serial scan still emitted live",
            f"{marker}[0] = 0.0",
        ),
    )


def _emit_mamba3_chunk_precompute_source(
    node: _ScheduleNodeView,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
) -> _ScheduleNodeFragment:
    """Fragment emitter for the Path-C F0 ``mamba3_chunk_precompute`` segment.

    Design: ``docs/MAMBA3-PATHC-MULTIKERNEL-DESIGN.md`` §2 (F0 row) / §3.2.

    F0 is NOT a row-phased single-thread fragment inlined into the shared region
    PrimFunc: its kernel is the GRID-launched precompute core
    ``chunk_precompute_fwd_metal_prim`` (own ``T.Kernel(batch*nchunks, nheads)``
    grid). Its real codegen is the single delegation
    ``mamba3_chunked_precompute_core.build_chunk_precompute_metal`` — exercised +
    parity gated chained (F0->F1->F2) vs the serial forward by
    ``tests/test_mamba3_chained_forward_f0f1f2.py`` (Stage 2). It writes the
    caller-owned ``cb / dA_cumsum / summary_states`` handoff buffers the F1/F2
    segments consume. This marker only needs to RESOLVE via ``select`` so the
    op-name signature is not blocked with ``no descriptor target``; the live chain
    template hookup (region-build flag + ABI handoff) is wired separately. RULE #1:
    no silent fallback — the kernel path is the one delegation above.
    """
    marker = _scratch_name(node, "mamba3_chunk_precompute_shadow")
    return _ScheduleNodeFragment(
        allocations=(f"{marker} = T.alloc_local((1,), \"float32\")",),
        statements=(
            "# F0 mamba3_chunk_precompute: delegates to "
            "cppmega_mlx.nn._tilelang.mamba3_chunked_precompute_core."
            "build_chunk_precompute_metal (grid precompute core -> cb/dA_cumsum/"
            "summary_states handoff buffers)",
            f"{marker}[0] = 0.0",
        ),
    )


def _emit_mamba3_inter_chunk_recur_source(
    node: _ScheduleNodeView,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
) -> _ScheduleNodeFragment:
    """Fragment emitter for the Path-C F1 ``mamba3_inter_chunk_recur`` segment.

    Design: ``docs/MAMBA3-PATHC-MULTIKERNEL-DESIGN.md`` §2 (F1 row) / §3.2.

    F1 is the ONLY O(S/C) sequential stage. Its kernel is the GRID-launched
    inter-chunk recurrence core ``inter_chunk_recur_fwd_metal_prim`` (own
    ``T.Kernel(batch, nheads)`` grid; the chunk axis is the only serial carry).
    Its real codegen is the single delegation
    ``mamba3_chunked_precompute_core.build_inter_chunk_recur_metal`` — exercised +
    parity gated chained (F0->F1->F2) vs the serial forward by
    ``tests/test_mamba3_chained_forward_f0f1f2.py`` (Stage 2). It reads the F0
    ``summary_states / dA_cumsum`` handoff plus ``h0`` and writes the
    ``prev_states`` (fp32) the F2 scan+combine consumes. This marker only needs to
    RESOLVE via ``select``; RULE #1: no silent fallback — one delegation path.
    """
    marker = _scratch_name(node, "mamba3_inter_chunk_recur_shadow")
    return _ScheduleNodeFragment(
        allocations=(f"{marker} = T.alloc_local((1,), \"float32\")",),
        statements=(
            "# F1 mamba3_inter_chunk_recur: delegates to "
            "cppmega_mlx.nn._tilelang.mamba3_chunked_precompute_core."
            "build_inter_chunk_recur_metal (O(S/C) inter-chunk recurrence -> "
            "prev_states/final_state handoff buffers, fp32)",
            f"{marker}[0] = 0.0",
        ),
    )


def _emit_mamba3_chunk_scan_combine_bwd_source(
    node: _ScheduleNodeView,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
) -> _ScheduleNodeFragment:
    """Fragment emitter for the Path-C B2 ``mamba3_chunk_scan_combine_bwd`` segment.

    Design: ``docs/MAMBA3-PATHC-MULTIKERNEL-DESIGN.md`` §2 (B2 row) / §7 Stage 3.

    B2 is the analytic TRANSPOSE of the forward F2 scan+combine (output/gate +
    Y_diag + Y_off transpose). Its kernel is the GRID-launched core
    ``chunk_scan_combine_bwd_metal_prim`` (own ``T.Kernel(batch*nchunks, nheads)``
    grid). Real codegen is the single delegation
    ``mamba3_chunked_backward_core.build_chunk_scan_combine_bwd_metal`` — validated
    vs the MLX backward proto (worst grad 3.68e-4) by
    ``scratch/test_b0b1b2_metal_vs_proto.py``. This marker only needs to RESOLVE
    via ``select``; the live interpose substitutes the real grid kernel when the
    flag is ON. RULE #1: no silent fallback — one delegation path.
    """
    marker = _scratch_name(node, "mamba3_chunk_scan_combine_bwd_shadow")
    return _ScheduleNodeFragment(
        allocations=(f"{marker} = T.alloc_local((1,), \"float32\")",),
        statements=(
            "# B2 mamba3_chunk_scan_combine_bwd: delegates to "
            "cppmega_mlx.nn._tilelang.mamba3_chunked_backward_core."
            "build_chunk_scan_combine_bwd_metal (grid output/Y transpose -> "
            "dC/dx/dz/dchunk_states/dinp/dA_cumsum_y/dD)",
            f"{marker}[0] = 0.0",
        ),
    )


def _emit_mamba3_chunk_precompute_bwd_source(
    node: _ScheduleNodeView,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
) -> _ScheduleNodeFragment:
    """Fragment emitter for the Path-C B0 ``mamba3_chunk_precompute_bwd`` segment.

    Design: ``docs/MAMBA3-PATHC-MULTIKERNEL-DESIGN.md`` §2 (B0 row) / §7 Stage 3.

    B0 is the transpose of the forward F0 precompute (decay_states transpose +
    dinp assembly + cumsum/segsum VJP). Real codegen delegation
    ``mamba3_chunked_backward_core.build_chunk_precompute_bwd_metal`` -> the final
    input grads ``dx/dB/dlog_decay/ddt``. This marker only needs to RESOLVE via
    ``select``; the live interpose substitutes the real grid kernel when the flag
    is ON. RULE #1: no silent fallback — one delegation path.
    """
    marker = _scratch_name(node, "mamba3_chunk_precompute_bwd_shadow")
    return _ScheduleNodeFragment(
        allocations=(f"{marker} = T.alloc_local((1,), \"float32\")",),
        statements=(
            "# B0 mamba3_chunk_precompute_bwd: delegates to "
            "cppmega_mlx.nn._tilelang.mamba3_chunked_backward_core."
            "build_chunk_precompute_bwd_metal (grid precompute transpose -> "
            "dx/dB/dlog_decay/ddt input grads)",
            f"{marker}[0] = 0.0",
        ),
    )


def _emit_mamba3_inter_chunk_recur_bwd_source(
    node: _ScheduleNodeView,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
) -> _ScheduleNodeFragment:
    """Fragment emitter for the Path-C B1 ``mamba3_inter_chunk_recur_bwd`` segment.

    Design: ``docs/MAMBA3-PATHC-MULTIKERNEL-DESIGN.md`` §2 (B1 row) / §7 Stage 3.

    B1 is the ONE genuinely new kernel: the REVERSE O(S/C) inter-chunk combiner
    (upper-tri ``decay_chunk`` contraction — the adjoint of the forward F1 lower-tri
    recurrence). It REUSES the forward-materialized ``prev_states`` boundary states
    (the 8x checkpoint-replay elimination, design §3/§6). Real codegen delegation
    ``mamba3_chunked_backward_core.build_inter_chunk_recur_bwd_metal`` ->
    ``dstates/dh0/dA_cumsum_tail``. This marker only needs to RESOLVE via
    ``select``; the live interpose substitutes the real grid kernel when the flag
    is ON. RULE #1: no silent fallback — one delegation path.
    """
    marker = _scratch_name(node, "mamba3_inter_chunk_recur_bwd_shadow")
    return _ScheduleNodeFragment(
        allocations=(f"{marker} = T.alloc_local((1,), \"float32\")",),
        statements=(
            "# B1 mamba3_inter_chunk_recur_bwd: delegates to "
            "cppmega_mlx.nn._tilelang.mamba3_chunked_backward_core."
            "build_inter_chunk_recur_bwd_metal (REVERSE O(S/C) upper-tri combiner "
            "-> dstates/dh0/dA_cumsum_tail; reuses forward prev_states)",
            f"{marker}[0] = 0.0",
        ),
    )


def _emit_residual_rmsnorm_source(
    node: _ScheduleNodeView,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
) -> _ScheduleNodeFragment:
    residual = _scratch_name(node, "residual")
    inv_rms = _scratch_name(node, "inv_rms")
    index = "i"
    lhs = _node_input_expr(node, 0, dtype_by_buffer, access_by_buffer, index)
    rhs = _node_input_expr(node, 1, dtype_by_buffer, access_by_buffer, index)
    weight = _node_input_expr(node, 2, dtype_by_buffer, access_by_buffer, index)
    inner = [
        f"{residual}[0] = {lhs} + {rhs}",
        f"{inv_rms}[0] = T.rsqrt(({residual}[0] * {residual}[0]) + 0.00001)",
    ]
    if node.outputs:
        inner.append(
            f"{_buffer_ref(node.outputs[0], access_by_buffer, index)} = "
            f"{residual}[0]"
        )
    if len(node.outputs) > 1:
        inner.append(
            f"{_buffer_ref(node.outputs[1], access_by_buffer, index)} = "
            f"{residual}[0] * {inv_rms}[0] * {weight}"
        )
    for output_name in node.outputs[2:]:
        inner.append(
            f"{_buffer_ref(output_name, access_by_buffer, index)} = "
            f"{residual}[0] * {inv_rms}[0] * {weight}"
        )
    return _ScheduleNodeFragment(
        allocations=(
            f"{residual} = T.alloc_local((1,), \"float32\")",
            f"{inv_rms} = T.alloc_local((1,), \"float32\")",
        ),
        statements=tuple(inner),
    )


def _emit_entry_rmsnorm_source(
    node: _ScheduleNodeView,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
) -> _ScheduleNodeFragment:
    """Flat fallback fragment for entry RMSNorm (descriptor only).

    Block A: the row-phased dispatch in
    :func:`_append_row_phased_hidden_body` handles the production path.
    This fragment is the scalar descriptor source emitted when the
    schedule is forced onto the flat per-cell loop (used only by the
    descriptor-status diagnostics and tests).
    """

    inv_rms = _scratch_name(node, "inv_rms")
    index = "i"
    hidden = _node_input_expr(node, 0, dtype_by_buffer, access_by_buffer, index)
    weight = _node_input_expr(node, 1, dtype_by_buffer, access_by_buffer, index)
    inner = [
        f"{inv_rms}[0] = T.rsqrt(({hidden} * {hidden}) + 0.00001)",
    ]
    for output_name in node.outputs:
        inner.append(
            f"{_buffer_ref(output_name, access_by_buffer, index)} = "
            f"{hidden} * {inv_rms}[0] * {weight}"
        )
    return _ScheduleNodeFragment(
        allocations=(
            f"{inv_rms} = T.alloc_local((1,), \"float32\")",
        ),
        statements=tuple(inner),
    )


def _emit_m2rnn_source(
    node: _ScheduleNodeView,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
) -> _ScheduleNodeFragment:
    projected = _scratch_name(node, "m2rnn_projected")
    conv = _scratch_name(node, "m2rnn_conv")
    xf = _scratch_name(node, "m2rnn_xf")
    recurrent = _scratch_name(node, "m2rnn_recurrent")
    post = _scratch_name(node, "m2rnn_post")
    index = "i"
    hidden = _node_input_expr(node, 0, dtype_by_buffer, access_by_buffer, index)
    in_proj_weight = _optional_buffer_expr(
        "m2rnn_in_proj_weight",
        dtype_by_buffer,
        access_by_buffer,
        index=index,
    )
    conv_weight = _optional_buffer_expr(
        "m2rnn_conv_weight",
        dtype_by_buffer,
        access_by_buffer,
        "1.0",
        index,
    )
    conv_bias = _optional_buffer_expr(
        "m2rnn_conv_bias",
        dtype_by_buffer,
        access_by_buffer,
        index=index,
    )
    conv_state = _optional_buffer_expr(
        "m2rnn_conv_state",
        dtype_by_buffer,
        access_by_buffer,
        index=index,
    )
    a_log = _optional_buffer_expr(
        "m2rnn_A_log",
        dtype_by_buffer,
        access_by_buffer,
        index=index,
    )
    dt_bias = _optional_buffer_expr(
        "m2rnn_dt_bias",
        dtype_by_buffer,
        access_by_buffer,
        index=index,
    )
    state_weight = _optional_buffer_expr(
        "m2rnn_state_weight",
        dtype_by_buffer,
        access_by_buffer,
        "1.0",
        index,
    )
    h0 = _optional_buffer_expr(
        "m2rnn_h0",
        dtype_by_buffer,
        access_by_buffer,
        index=index,
    )
    d_skip = _optional_buffer_expr(
        "m2rnn_D",
        dtype_by_buffer,
        access_by_buffer,
        "1.0",
        index,
    )
    gate_norm_weight = _optional_buffer_expr(
        "m2rnn_g_norm_weight",
        dtype_by_buffer,
        access_by_buffer,
        "1.0",
        index,
    )
    out_proj_weight = _optional_buffer_expr(
        "m2rnn_out_proj_weight",
        dtype_by_buffer,
        access_by_buffer,
        "1.0",
        index,
    )
    inner = [
        f"{projected}[0] = {hidden} * {in_proj_weight}",
        f"{conv}[0] = ({projected}[0] * {conv_weight}) + "
        f"{conv_bias} + {conv_state}",
        f"{xf}[0] = T.log(1.0 + T.exp({projected}[0] + {a_log} + "
        f"{dt_bias}))",
        f"{recurrent}[0] = ({conv}[0] * {state_weight}) + "
        f"({h0} * T.exp(-{xf}[0]))",
        f"{post}[0] = ({recurrent}[0] + ({conv}[0] * {d_skip})) * "
        f"(1.0 / (1.0 + T.exp(-{projected}[0])))",
    ]
    for output_name in node.outputs:
        inner.append(
            f"{_buffer_ref(output_name, access_by_buffer, index)} = "
            f"{post}[0] * {gate_norm_weight} * {out_proj_weight}"
        )
    return _ScheduleNodeFragment(
        allocations=(
            f"{projected} = T.alloc_local((1,), \"float32\")",
            f"{conv} = T.alloc_local((1,), \"float32\")",
            f"{xf} = T.alloc_local((1,), \"float32\")",
            f"{recurrent} = T.alloc_local((1,), \"float32\")",
            f"{post} = T.alloc_local((1,), \"float32\")",
        ),
        statements=tuple(inner),
    )


def _emit_attention_qkv_projection_source(
    node: _ScheduleNodeView,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
) -> _ScheduleNodeFragment:
    q_projected = _scratch_name(node, "attention_q_projected")
    kv_projected = _scratch_name(node, "attention_kv_projected")
    q_projected_pair = _scratch_name(node, "attention_q_projected_pair")
    kv_projected_pair = _scratch_name(node, "attention_kv_projected_pair")
    rope_phase = _scratch_name(node, "attention_rope_phase")
    q_prepared = _scratch_name(node, "attention_q_prepared")
    kv_prepared = _scratch_name(node, "attention_kv_prepared")
    assigned: set[str] = set()
    index = "i"
    q_scale_output = _node_output_for_canonical(node, "q_scale")
    kv_scale_output = _node_output_for_canonical(node, "kv_scale")
    q_fp8_output = _node_output_for_canonical(node, "q_fp8")
    kv_fp8_output = _node_output_for_canonical(node, "kv_fp8")
    indices_output = _node_output_for_canonical(node, "indices")
    hidden = _node_input_expr(node, 0, dtype_by_buffer, access_by_buffer, index)
    q_weight = _optional_buffer_expr(
        "attention_q_proj_weight",
        dtype_by_buffer,
        access_by_buffer,
        index=index,
    )
    q_bias = _optional_buffer_expr(
        "attention_q_proj_bias",
        dtype_by_buffer,
        access_by_buffer,
        index=index,
    )
    kv_weight = _optional_buffer_expr(
        "attention_sparse_kv_proj_weight",
        dtype_by_buffer,
        access_by_buffer,
        index=index,
    )
    kv_bias = _optional_buffer_expr(
        "attention_sparse_kv_proj_bias",
        dtype_by_buffer,
        access_by_buffer,
        index=index,
    )
    rope = _optional_buffer_expr(
        "attention_rope_inv_freq",
        dtype_by_buffer,
        access_by_buffer,
        index=index,
    )
    inner = [
        f"{q_projected}[0] = {hidden} + {q_weight} + {q_bias}",
        f"{kv_projected}[0] = {hidden} + {kv_weight} + {kv_bias}",
        f"{rope_phase}[0] = {rope}",
        f"{q_prepared}[0] = {q_projected}[0] + {rope_phase}[0]",
        f"{kv_prepared}[0] = {kv_projected}[0] + {rope_phase}[0]",
    ]
    if q_scale_output is not None:
        inner.append(
            f"{_buffer_ref(q_scale_output, access_by_buffer, index)} = "
            f"T.max(T.abs(T.cast({q_prepared}[0], \"float32\")) * "
            f"T.cast({1.0 / 448.0:.17g}, \"float32\"), "
            f"T.cast(1.0e-12, \"float32\"))"
        )
        assigned.add(q_scale_output)
    if kv_scale_output is not None:
        inner.append(
            f"{_buffer_ref(kv_scale_output, access_by_buffer, index)} = "
            f"T.max(T.abs(T.cast({kv_prepared}[0], \"float32\")) * "
            f"T.cast({1.0 / 448.0:.17g}, \"float32\"), "
            f"T.cast(1.0e-12, \"float32\"))"
        )
        assigned.add(kv_scale_output)
    if q_fp8_output is not None:
        denominator = (
            _buffer_ref(q_scale_output, access_by_buffer, index)
            if q_scale_output is not None
            else "1.0"
        )
        inner.append(
            f"{_buffer_ref(q_fp8_output, access_by_buffer, index)} = "
            f"{_fp8_encode_expr(f'{q_prepared}[0]', denominator)}"
        )
        assigned.add(q_fp8_output)
    if kv_fp8_output is not None:
        denominator = (
            _buffer_ref(kv_scale_output, access_by_buffer, index)
            if kv_scale_output is not None
            else "1.0"
        )
        inner.append(
            f"{_buffer_ref(kv_fp8_output, access_by_buffer, index)} = "
            f"{_fp8_encode_expr(f'{kv_prepared}[0]', denominator)}"
        )
        assigned.add(kv_fp8_output)
    if indices_output is not None:
        inner.append(f"{_buffer_ref(indices_output, access_by_buffer, index)} = i")
        assigned.add(indices_output)
    for output_name in node.outputs:
        if output_name in assigned:
            continue
        inner.append(
            f"{_buffer_ref(output_name, access_by_buffer, index)} = "
            f"{q_prepared}[0] + {kv_prepared}[0]"
        )
    return _ScheduleNodeFragment(
        allocations=(
            f"{q_projected} = T.alloc_local((1,), \"float32\")",
            f"{kv_projected} = T.alloc_local((1,), \"float32\")",
            f"{q_projected_pair} = T.alloc_local((1,), \"float32\")",
            f"{kv_projected_pair} = T.alloc_local((1,), \"float32\")",
            f"{rope_phase} = T.alloc_local((1,), \"float32\")",
            f"{q_prepared} = T.alloc_local((1,), \"float32\")",
            f"{kv_prepared} = T.alloc_local((1,), \"float32\")",
        ),
        statements=tuple(inner),
    )


def _emit_sparse_mla_fp8_apply_source(
    node: _ScheduleNodeView,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
) -> _ScheduleNodeFragment:
    sink_enabled = _scratch_name(node, "sink_enabled")
    sparse_index = _scratch_name(node, "sparse_index")
    score_accum = _scratch_name(node, "score_accum")
    score_max = _scratch_name(node, "score_max")
    score_weight = _scratch_name(node, "score_weight")
    sumexp = _scratch_name(node, "sumexp")
    value_accum = _scratch_name(node, "value_accum")
    context_accum = _scratch_name(node, "context_accum")
    q_head_index = _scratch_name(node, "q_head")
    kv_head_index = _scratch_name(node, "kv_head")
    source_head_index = _scratch_name(node, "source_head")
    source_dim_index = _scratch_name(node, "source_dim")
    index = "i"
    q_fp8 = _optional_buffer_expr("q_fp8", dtype_by_buffer, access_by_buffer, index=index)
    q_scale = _optional_buffer_expr(
        "q_scale",
        dtype_by_buffer,
        access_by_buffer,
        "1.0",
        index,
    )
    indices_ref = _optional_buffer_raw_ref(
        "indices",
        dtype_by_buffer,
        access_by_buffer,
        default="0",
        index=index,
    )
    kv_fp8 = _optional_indexed_buffer_expr(
        "kv_fp8",
        dtype_by_buffer,
        access_by_buffer,
        index_expr=f"{sparse_index}[0]",
    )
    kv_scale = _optional_buffer_expr(
        "kv_scale",
        dtype_by_buffer,
        access_by_buffer,
        "1.0",
        index,
    )
    selected_kv_scale = _optional_indexed_buffer_expr(
        "kv_scale",
        dtype_by_buffer,
        access_by_buffer,
        default=kv_scale,
        index_expr=f"{sparse_index}[0]",
    )
    sm_scale = _optional_buffer_expr(
        "sparse_mla_sm_scale",
        dtype_by_buffer,
        access_by_buffer,
        "1.0",
        index,
    )
    sinks = _optional_buffer_expr(
        "sparse_mla_sinks",
        dtype_by_buffer,
        access_by_buffer,
        index=index,
    )
    has_sinks = _optional_buffer_expr(
        "sparse_mla_has_sinks",
        dtype_by_buffer,
        access_by_buffer,
        "0",
        index,
    )
    out_proj_weight = _optional_buffer_expr(
        "attention_out_proj_weight",
        dtype_by_buffer,
        access_by_buffer,
        "1.0",
        index,
    )
    out_proj_bias = _optional_buffer_expr(
        "attention_out_proj_bias",
        dtype_by_buffer,
        access_by_buffer,
        "0.0",
        index,
    )
    inner = [
        f"{sink_enabled}[0] = T.cast({has_sinks} != 0, \"float32\")",
        f"{sparse_index}[0] = T.cast({indices_ref}, \"int32\")",
    ]
    assigned: set[str] = set()
    attention_out = (
        _node_output_for_canonical(node, "attention_out")
        or _node_output_with_suffix(node, "_out")
    )
    lse_output = _node_output_for_canonical(node, "lse")
    if attention_out is not None:
        inner.append(
            f"{_buffer_ref(attention_out, access_by_buffer, index)} = "
            f"(((({q_fp8} * {q_scale}) + "
            f"({kv_fp8} * {selected_kv_scale})) * {sm_scale} + "
            f"({sinks} * {sink_enabled}[0])) * "
            f"{out_proj_weight}) + {out_proj_bias}"
        )
        assigned.add(attention_out)
    if lse_output is not None:
        inner.append(f"{_buffer_ref(lse_output, access_by_buffer, index)} = 0.0")
        assigned.add(lse_output)
    for output_name in node.outputs:
        if output_name in assigned:
            continue
        inner.append(
            f"{_buffer_ref(output_name, access_by_buffer, index)} = "
            f"{q_fp8} + {kv_fp8}"
        )
    return _ScheduleNodeFragment(
        allocations=(
            f"{sink_enabled} = T.alloc_local((1,), \"float32\")",
            f"{sparse_index} = T.alloc_local((1,), \"int32\")",
            f"{score_accum} = T.alloc_local((1,), \"float32\")",
            f"{score_max} = T.alloc_local((1,), \"float32\")",
            f"{score_weight} = T.alloc_local((1,), \"float32\")",
            f"{sumexp} = T.alloc_local((1,), \"float32\")",
            f"{value_accum} = T.alloc_local((1,), \"float32\")",
            f"{context_accum} = T.alloc_local((1,), \"float32\")",
            f"{q_head_index} = T.alloc_local((1,), \"int32\")",
            f"{kv_head_index} = T.alloc_local((1,), \"int32\")",
            f"{source_head_index} = T.alloc_local((1,), \"int32\")",
            f"{source_dim_index} = T.alloc_local((1,), \"int32\")",
        ),
        statements=tuple(inner),
    )


def _emit_owner_output_backward_source(
    node: _ScheduleNodeView,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
) -> _ScheduleNodeFragment:
    accum = _scratch_name(node, "grad_accum")
    index = "i"
    inner = [
        f"{accum}[0] = {_sum_buffer_expr(node.inputs, dtype_by_buffer, access_by_buffer, index)}",
    ]
    for output_index, output_name in enumerate(node.outputs):
        inner.append(
            f"{_buffer_ref(output_name, access_by_buffer, index)} = "
            f"{accum}[0] + "
            f"{float(output_index):.1f}"
        )
    return _ScheduleNodeFragment(
        allocations=(f"{accum} = T.alloc_local((1,), \"float32\")",),
        statements=tuple(inner),
    )


def _emit_generic_descriptor_source(
    node: _ScheduleNodeView,
    node_index: int,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
) -> _ScheduleNodeFragment:
    index = "i"
    inner: list[str] = []
    for output_index, output_name in enumerate(node.outputs):
        dtype = dtype_by_buffer[output_name]
        if dtype == "int32":
            inner.append(
                f"{_buffer_ref(output_name, access_by_buffer, index)} = "
                f"{index} + {output_index}"
            )
            continue
        expr = _sum_buffer_expr(node.inputs, dtype_by_buffer, access_by_buffer, index)
        if node_index or output_index:
            expr = f"({expr}) + {float(node_index + output_index):.1f}"
        inner.append(f"{_buffer_ref(output_name, access_by_buffer, index)} = {expr}")
    return _ScheduleNodeFragment(allocations=(), statements=tuple(inner))


def _mamba3_projection_inputs(node: _ScheduleNodeView) -> tuple[str, ...]:
    return _projection_inputs_from_node(
        node,
        leading_input_count=2,
        canonical_names=(
            "mamba3_in_proj_weight",
            "mamba3_out_proj_weight",
            "mamba3_conv_weight",
            "mamba3_conv_bias",
            "mamba3_dt_bias",
            "mamba3_h0",
        ),
    )


def _m2rnn_projection_inputs(node: _ScheduleNodeView) -> tuple[str, ...]:
    return _projection_inputs_from_node(
        node,
        leading_input_count=1,
        canonical_names=(
            "m2rnn_in_proj_weight",
            "m2rnn_conv_weight",
            "m2rnn_conv_bias",
            "m2rnn_state_weight",
            "m2rnn_A_log",
            "m2rnn_dt_bias",
            "m2rnn_h0",
            "m2rnn_conv_state",
        ),
    )


def _attention_projection_inputs(node: _ScheduleNodeView) -> tuple[str, ...]:
    return _projection_inputs_from_node(
        node,
        leading_input_count=1,
        canonical_names=(
            "attention_q_proj_weight",
            "attention_q_proj_bias",
            "attention_sparse_kv_proj_weight",
            "attention_sparse_kv_proj_bias",
            "attention_rope_inv_freq",
        ),
    )


def _projection_inputs_from_node(
    node: _ScheduleNodeView,
    *,
    leading_input_count: int,
    canonical_names: Sequence[str],
) -> tuple[str, ...]:
    selected: list[str] = []
    seen = set()
    for input_name in node.inputs[:leading_input_count]:
        if input_name in seen:
            continue
        seen.add(input_name)
        selected.append(input_name)
    wanted = set(canonical_names)
    for input_name in node.inputs[leading_input_count:]:
        canonical_name = _canonical_buffer_name(input_name)
        if canonical_name not in wanted or canonical_name in seen:
            continue
        seen.add(canonical_name)
        selected.append(input_name)
    return tuple(selected)


def _scratch_name(node: _ScheduleNodeView, suffix: str) -> str:
    return _safe_identifier(f"{node.name}_{suffix}")


def _output_with_suffix(node: _ScheduleNodeView, suffix: str) -> str | None:
    for output_name in node.outputs:
        if output_name.endswith(suffix):
            return output_name
    return None


def _node_output_for_canonical(
    node: _ScheduleNodeView,
    canonical_name: str,
) -> str | None:
    for output_name in node.outputs:
        if _canonical_buffer_name(output_name) == canonical_name:
            return output_name
    return None


def _node_output_for_canonical_or_index(
    node: _ScheduleNodeView,
    canonical_names: Sequence[str],
    positional_index: int,
) -> str | None:
    for canonical_name in canonical_names:
        output_name = _node_output_for_canonical(node, canonical_name)
        if output_name is not None:
            return output_name
    if positional_index >= len(node.outputs):
        return None
    return node.outputs[positional_index]


def _node_input_for_canonical(
    node: _ScheduleNodeView,
    canonical_name: str,
) -> str | None:
    for input_name in node.inputs:
        if _canonical_buffer_name(input_name) == canonical_name:
            return input_name
    return None


def _node_output_with_suffix(
    node: _ScheduleNodeView,
    suffix: str,
) -> str | None:
    for output_name in node.outputs:
        if output_name.endswith(suffix):
            return output_name
    return None


def _node_input_expr(
    node: _ScheduleNodeView,
    index: int,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    loop_index: str,
) -> str:
    if index >= len(node.inputs):
        return "0.0"
    input_name = node.inputs[index]
    return _buffer_value_expr(
        input_name,
        dtype_by_buffer[input_name],
        access_by_buffer,
        loop_index,
    )


def _sum_buffer_expr(
    buffer_names: Sequence[str],
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    index: str,
) -> str:
    terms = [
        _buffer_value_expr(
            buffer_name,
            dtype_by_buffer[buffer_name],
            access_by_buffer,
            index,
        )
        for buffer_name in buffer_names
        if buffer_name in dtype_by_buffer
    ]
    return " + ".join(terms) if terms else "0.0"


def _optional_buffer_expr(
    buffer_name: str,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    default: str = "0.0",
    index: str = "0",
) -> str:
    resolved_name = _buffer_name_for_canonical_or_exact(buffer_name, dtype_by_buffer)
    if resolved_name is None:
        return default
    return _buffer_value_expr(
        resolved_name,
        dtype_by_buffer[resolved_name],
        access_by_buffer,
        index,
    )


def _optional_buffer_raw_ref(
    buffer_name: str,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    *,
    default: str,
    index: str,
) -> str:
    resolved_name = _buffer_name_for_canonical_or_exact(buffer_name, dtype_by_buffer)
    if resolved_name is None:
        return default
    return _buffer_ref(resolved_name, access_by_buffer, index)


def _optional_indexed_buffer_expr(
    buffer_name: str,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    *,
    default: str = "0.0",
    index_expr: str,
) -> str:
    resolved_name = _buffer_name_for_canonical_or_exact(buffer_name, dtype_by_buffer)
    if resolved_name is None:
        return default
    return _indexed_buffer_value_expr(
        resolved_name,
        dtype_by_buffer[resolved_name],
        access_by_buffer,
        index_expr,
    )


def _buffer_name_for_canonical_or_exact(
    buffer_name: str,
    dtype_by_buffer: Mapping[str, str],
) -> str | None:
    if buffer_name in dtype_by_buffer:
        return buffer_name
    canonical_name = _canonical_buffer_name(buffer_name)
    for candidate_name in dtype_by_buffer:
        if _canonical_buffer_name(candidate_name) == canonical_name:
            return candidate_name
    return None


def _buffer_value_expr(
    buffer_name: str,
    dtype: str,
    access_by_buffer: dict[str, str],
    index: str,
) -> str:
    ref = _buffer_ref(buffer_name, access_by_buffer, index)
    if dtype in {"bfloat16", "float16", "int32"}:
        return f'T.cast({ref}, "float32")'
    if dtype == "uint8":
        return f"fp8_e4m3fn_to_float({ref})"
    return ref


def _fp8_encode_expr(value_expr: str, scale_expr: str) -> str:
    normalized = f'(T.cast({value_expr}, "float32") / {scale_expr})'
    clamped = (
        f'T.min(T.max({normalized}, T.cast(-448.0, "float32")), '
        f'T.cast(448.0, "float32"))'
    )
    return f"float_to_fp8_e4m3fn_bits({clamped})"


def _indexed_buffer_value_expr(
    buffer_name: str,
    dtype: str,
    access_by_buffer: dict[str, str],
    index_expr: str,
) -> str:
    ref = _indexed_buffer_ref(buffer_name, access_by_buffer, index_expr)
    if dtype in {"bfloat16", "float16", "int32"}:
        return f'T.cast({ref}, "float32")'
    if dtype == "uint8":
        return f"fp8_e4m3fn_to_float({ref})"
    return ref


def _buffer_ref(
    buffer_name: str,
    access_by_buffer: dict[str, str],
    index: str,
) -> str:
    ref = access_by_buffer.get(buffer_name)
    if ref is None:
        return f"{buffer_name}[{index}]"
    if index == "i":
        return ref
    return _indexed_buffer_ref(buffer_name, access_by_buffer, index)


def _indexed_buffer_ref(
    buffer_name: str,
    access_by_buffer: dict[str, str],
    index_expr: str,
) -> str:
    ref = access_by_buffer.get(buffer_name)
    if ref is None:
        return f"{_safe_identifier(buffer_name)}[{index_expr}]"
    match = re.fullmatch(r"([A-Za-z_]\w*)\[0\]", ref)
    if match is not None:
        return ref
    match = re.fullmatch(r"([A-Za-z_]\w*)\[i\]", ref)
    if match is not None:
        return f"{match.group(1)}[{index_expr}]"
    match = re.fullmatch(r"([A-Za-z_]\w*)\[i % (\d+)\]", ref)
    if match is not None:
        return f"{match.group(1)}[({index_expr}) % {match.group(2)}]"
    match = re.fullmatch(r"([A-Za-z_]\w*)\[\(i % (\d+)\) // (\d+)\]", ref)
    if match is not None:
        return (
            f"{match.group(1)}[(({index_expr}) % {match.group(2)}) // "
            f"{match.group(3)}]"
        )
    match = re.fullmatch(
        r"([A-Za-z_]\w*)\[\(\(i % (\d+)\) // (\d+)\) % (\d+)\]",
        ref,
    )
    if match is not None:
        return (
            f"{match.group(1)}[((({index_expr}) % {match.group(2)}) // "
            f"{match.group(3)}) % {match.group(4)}]"
        )
    match = re.fullmatch(r"([A-Za-z_]\w*)\[i // (\d+)\]", ref)
    if match is not None:
        return f"{match.group(1)}[({index_expr}) // {match.group(2)}]"
    match = re.fullmatch(r"([A-Za-z_]\w*)\[(.+)\]", ref)
    if match is not None:
        if re.search(r"\bi\b", match.group(2)):
            replaced = re.sub(r"\bi\b", f"({index_expr})", match.group(2))
            return f"{match.group(1)}[{replaced}]"
        return ref
    return f"{_safe_identifier(buffer_name)}[{index_expr}]"


def _node_views_for_region(region: Any) -> tuple[_ScheduleNodeView, ...]:
    nodes = getattr(region, "nodes", None)
    if nodes is None:
        raise TypeError("descriptor schedule generation requires a region with nodes")
    views: list[_ScheduleNodeView] = []
    for node in nodes:
        op_name = getattr(node, "op_name", None)
        if op_name is None:
            op_name = getattr(node, "op", None)
        if op_name is None:
            raise TypeError("region node must expose op_name or op")
        views.append(
            _ScheduleNodeView(
                name=str(getattr(node, "name")),
                op_name=str(op_name),
                inputs=tuple(str(name) for name in getattr(node, "inputs", ())),
                outputs=tuple(str(name) for name in getattr(node, "outputs", ())),
                backward=str(getattr(node, "backward", "")),
            )
        )
    if not views:
        raise ValueError("descriptor schedule generation requires at least one node")
    return tuple(views)


def _internal_buffers_for_nodes(
    nodes: Sequence[_ScheduleNodeView],
    *,
    shape_env: PathCModelShapeEnv | None = None,
    internal_buffer_policy: str = DESCRIPTOR_INTERNAL_BUFFER_POLICY_SCALAR_LOCAL,
    loop_policy: str = DESCRIPTOR_LOOP_POLICY_FLAT,
) -> tuple[str, ...]:
    input_names = {name for node in nodes for name in node.inputs}
    internal: list[str] = []
    seen: set[str] = set()
    for node in nodes:
        for output_name in node.outputs:
            if output_name not in input_names or output_name in seen:
                continue
            if _is_attention_kv_history_workspace_output(nodes, node, output_name):
                continue
            if _uses_external_kv_history_workspace(
                output_name,
                shape_env=shape_env,
                internal_buffer_policy=internal_buffer_policy,
                loop_policy=loop_policy,
            ):
                continue
            seen.add(output_name)
            internal.append(output_name)
    return tuple(internal)


def _descriptor_chain_uses_kv_history_workspace(
    descriptors: Sequence[PathCBrickScheduleDescriptor],
) -> bool:
    op_names = {descriptor.op_name for descriptor in descriptors}
    return {
        "attention_qkv_projection",
        "sparse_mla_fp8_apply",
    }.issubset(op_names)


def _is_attention_kv_history_workspace_output(
    nodes: Sequence[_ScheduleNodeView],
    producer: _ScheduleNodeView,
    output_name: str,
) -> bool:
    if producer.op_name != "attention_qkv_projection":
        return False
    if str(output_name).endswith("_grad"):
        return False
    if _canonical_buffer_name(output_name) not in {
        "q_fp8",
        "q_scale",
        "kv_fp8",
        "kv_scale",
    }:
        return False
    return any(
        consumer.op_name in {"sparse_mla_fp8_apply", "sparse_mla_fp8_apply_bwd"}
        and output_name in consumer.inputs
        for consumer in nodes
    )


def _uses_external_kv_history_workspace(
    buffer_name: str,
    *,
    shape_env: PathCModelShapeEnv | None,
    internal_buffer_policy: str,
    loop_policy: str,
) -> bool:
    if str(buffer_name).endswith("_grad"):
        return False
    if shape_env is None:
        return False
    if internal_buffer_policy != DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN:
        return False
    if loop_policy != DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN:
        return False
    return _canonical_buffer_name(buffer_name) in {
        "q_fp8",
        "q_scale",
        "kv_fp8",
        "kv_scale",
    }


def _external_buffers_for_nodes(
    nodes: Sequence[_ScheduleNodeView],
    internal_buffers: Sequence[str],
) -> tuple[str, ...]:
    internal_set = set(internal_buffers)
    external: list[str] = []
    seen: set[str] = set()
    for node in nodes:
        for buffer_name in (*node.inputs, *node.outputs):
            if buffer_name in internal_set or buffer_name in seen:
                continue
            seen.add(buffer_name)
            external.append(buffer_name)
    return tuple(external)


def _buffer_dtype(
    buffer_name: str,
    *,
    shape_env: PathCModelShapeEnv | None = None,
) -> str:
    canonical = _canonical_buffer_name(buffer_name)
    is_grad = str(buffer_name).endswith("_grad")
    # fp8/int classification applies to forward VALUE buffers only; the gradient of
    # an fp8/int buffer is a real-valued gradient (carrier dtype), never uint8/int32.
    if not is_grad:
        if canonical in {"q_fp8", "kv_fp8"}:
            return "uint8"
        if canonical == "target_ids":
            return "int32"
        if buffer_name == "indices" or buffer_name.endswith("_indices"):
            return "int32"
        if buffer_name.endswith("_has_sinks"):
            return "int32"
    # FP32-mandatory: fp8 dequant scales, LSE reductions, and recurrent scan/state
    # accumulation INCLUDING their initial-state/state-input gradients (h0/state_in)
    # — downcasting corrupts the fp8 matmul, softmax normalization, or recurrent
    # backward stability. (In-kernel reverse-scan accumulators live in threadgroup
    # scratch and are already float32 via accum_dtype, independent of this map.)
    if canonical in {
        "mamba3_angle_state",
        "mamba3_angle_checkpoint",
        "mamba3_angle_grad_state",
        "mamba3_conv_state",
        "mamba3_h_checkpoint",
        "mamba3_h0_grad",
        # Mamba3 chunked-scan F0->F1->F2 inter-chunk state handoff buffers
        # (design §3.3/§6.6): fp32 for recurrent-state precision.
        "mamba3_summary_states",
        "mamba3_prev_states",
        "mamba3_final_state",
        # Mamba3 chunked-scan B2->B1->B0 backward grad-handoff buffers (Stage 3):
        # fp32 for recurrent grad-state precision.
        "mamba3_dh_last",
        "mamba3_dchunk_states",
        "mamba3_dstates",
        "mamba3_dinp_diag",
        "mamba3_dA_cumsum_y",
        "mamba3_dA_cumsum_tail",
        "m2rnn_h_state",
        "m2rnn_h_checkpoint",
        "m2rnn_h0_grad",
        "state_in_grad",
        "q_scale",
        "kv_scale",
        "lse",
    } or buffer_name.endswith(
        (
            "_scale",
            "_sm_scale",
            "_lse",
            "_h0_grad",
            "_state_in_grad",
        )
    ):
        return "float32"
    # Parameter/activation VALUE buffers AND their gradients carry the model's
    # bf16/fp16 carrier dtype. Path B keeps gradients in the bf16 parameter dtype,
    # so bf16 grad banks are parity-consistent (the previous unconditional float32
    # for every *_grad doubled the gradient arenas). For a float32 model the carrier
    # resolves to float32, so behaviour is unchanged (no regression).
    return (
        str(shape_env.model_value_dtype)
        if shape_env is not None and str(shape_env.model_value_dtype)
        else "float32"
    )


def _shape_env_for_region(region: Any) -> PathCModelShapeEnv | None:
    metadata = getattr(region, "metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    shape_env = metadata.get("path_c_model_shape_env")
    return shape_env if isinstance(shape_env, PathCModelShapeEnv) else None


_KNOWN_BUFFER_SUFFIXES = tuple(
    sorted(
        {
            *MAMBA3_FP8_TRAIN_REQUIRED_REAL_ABI_INPUTS,
            *_GENERIC_MODEL_REAL_ABI_INPUT_SUFFIXES,
            *_TRAIN_STEP_SUFFIX_LOSS_INPUT_ABI_NAMES,
            "hidden",
            "mamba_state",
            "scan_state",
            "mamba3_angle_state",
            "mamba3_angle_checkpoint",
            "mamba3_angle_grad_state",
            "mamba3_conv_state",
            "mamba3_h_checkpoint",
            "mamba3_delta",
            "m2rnn_hidden",
            "m2rnn_delta",
            "m2rnn_h_state",
            "m2rnn_h_checkpoint",
            "m2rnn_conv_state",
            "attention_hidden",
            "hidden_after_mamba3",
            "hidden_after_m2rnn",
            "attention_out",
            "lse",
            "q_fp8",
            "q_scale",
            "kv_fp8",
            "kv_scale",
            "indices",
        },
        key=len,
        reverse=True,
    )
)


def _canonical_buffer_name(buffer_name: str) -> str:
    name = str(buffer_name)
    if name.endswith("_grad"):
        name = name[: -len("_grad")]
    for suffix in _KNOWN_BUFFER_SUFFIXES:
        if name == suffix or name.endswith(f"_{suffix}"):
            return suffix
    return name


def _is_mamba3_state_like_buffer(buffer_name: str, canonical_name: str) -> bool:
    if canonical_name in {"mamba_state", "scan_state", "mamba3_h0"}:
        return True
    if canonical_name in {
        "mamba3_angle_state",
        "mamba3_angle_checkpoint",
        "mamba3_angle_grad_state",
        "mamba3_conv_state",
        "mamba3_h_checkpoint",
        "m2rnn_conv_state",
        "m2rnn_h_state",
    }:
        return False
    name = str(buffer_name)
    if name.endswith("_grad"):
        name = name[: -len("_grad")]
    return (
        (name.endswith("_state") or name.endswith("_state_in") or name.endswith("_state_out"))
        and "m2rnn" not in name
    )


# Per-brick projected-SSD region input + inter-chunk handoff buffer SUFFIXES for
# the chunked F0/F1/F2 forward (design §2/§7). They are caller-owned region ABI
# buffers named ``{brick}_{suffix}`` (e.g. ``mamba3_scan_x``, ``mamba3_scan_cb``).
# Their flattened extents are derived from the mamba shape_env (batch is 1 for the
# direct-chain region; the chunk size is the validated scan-core block).
_MAMBA3_CHUNKED_FWD_REGION_BUFFER_SUFFIXES: tuple[str, ...] = (
    # projected SSD inputs (F0/F2 read these)
    "x",
    "B",
    "C",
    "A",
    "dt",
    # initial / skip params
    "h0",
    "D",
    # inter-chunk handoff (F0 -> F1 -> F2)
    "cb",
    "dA_cumsum",
    "summary_states",
    "prev_states",
    "final_state",
    # backward grad-handoff (B2 -> B1 -> B0; Stage 3)
    "dh_last",
    "dchunk_states",
    "dstates",
    "dinp_diag",
    "dA_cumsum_y",
    "dA_cumsum_tail",
)


# Real-ABI buffer names that COLLIDE on the chunked single-letter / ``h0``
# suffixes but are NOT chunked region buffers — they must resolve through the
# generic shape map. ``mamba3_h0``/``mamba3_D`` happen to share their flattened
# extent with the chunked shape, but ``m2rnn_D``/``m2rnn_h0`` do NOT, so a silent
# rebind would corrupt them (RULE #1).
_MAMBA3_CHUNKED_FWD_REGION_BUFFER_NAME_EXCLUSIONS: frozenset[str] = frozenset(
    {"mamba3_h0", "mamba3_D", "m2rnn_h0", "m2rnn_D"}
)
# Real-ABI suffixes that, when a brick-prefixed buffer ends with them, indicate a
# prefixed real ABI weight (e.g. ``mamba3_scan_mamba3_D``) rather than a chunked
# region buffer — exclude these too so the serial (flag-OFF) prefixed ABI is never
# rebound to a chunked shape.
_MAMBA3_CHUNKED_FWD_REGION_BUFFER_SUFFIX_EXCLUSIONS: tuple[str, ...] = (
    "_mamba3_D",
    "_mamba3_h0",
    "_m2rnn_D",
    "_m2rnn_h0",
)


def _mamba3_chunked_fwd_region_buffer_shape(
    buffer_name: str,
    shape_env: PathCModelShapeEnv,
) -> tuple[int, ...] | None:
    """Return the flattened shape of a chunked F0/F1/F2 region ABI buffer.

    Recognises the per-brick ``{brick}_{suffix}`` projected-SSD inputs and the
    inter-chunk handoff buffers (design §2/§7). Returns ``None`` when the name is
    not one of those suffixes so the caller falls through to the generic map.
    RULE #1: the shapes mirror the prim-func tensor ABI exactly (a mismatch would
    silently mis-bind a handoff buffer); there is no padding/guess fallback.
    """

    name = str(buffer_name)
    if name.endswith("_grad"):
        return None
    # The chunked region emits buffers named ``{brick}_{suffix}`` where ``brick``
    # is the mamba brick name (e.g. ``mamba3_scan``). Exclude the bare real-ABI
    # names (``mamba3_h0``/``mamba3_D``/``m2rnn_h0``/``m2rnn_D``) which collide on
    # the single-letter / ``h0`` suffixes but are NOT chunked region buffers —
    # those resolve through the generic map below (RULE #1: never silently rebind
    # a real ABI buffer to a chunked shape).
    if name in _MAMBA3_CHUNKED_FWD_REGION_BUFFER_NAME_EXCLUSIONS:
        return None
    if name.endswith(_MAMBA3_CHUNKED_FWD_REGION_BUFFER_SUFFIX_EXCLUSIONS):
        return None
    suffix: str | None = None
    for candidate in _MAMBA3_CHUNKED_FWD_REGION_BUFFER_SUFFIXES:
        # Require a non-empty brick prefix (``_{suffix}``); a bare ``x``/``D``/...
        # is never a per-brick chunked region buffer.
        if name.endswith(f"_{candidate}") and len(name) > len(candidate) + 1:
            suffix = candidate
            break
    if suffix is None:
        return None
    batch = 1
    seqlen = int(shape_env.sequence_length)
    nheads = int(shape_env.mamba_num_heads)
    headdim = int(shape_env.mamba_head_dim)
    dstate = int(shape_env.mamba_state_dim)
    ngroups = int(shape_env.mamba_groups)
    chunk = int(MAMBA3_CHUNKED_FWD_SCAN_CHUNK_SIZE)
    if seqlen % chunk != 0:
        raise ValueError(
            "mamba3 chunked-scan region buffer shape: seqlen "
            f"({seqlen}) must be divisible by chunk ({chunk}); no padding "
            f"(RULE #1) for buffer {buffer_name!r}"
        )
    nchunks = seqlen // chunk
    shapes: dict[str, tuple[int, ...]] = {
        # projected SSD per-position tensors (F0/F2 inputs)
        "x": (batch, seqlen, nheads, headdim),
        "B": (batch, seqlen, ngroups, dstate),
        "C": (batch, seqlen, ngroups, dstate),
        "A": (nheads,),
        # F0 takes dt as (b,s,h); F2 takes dt reshaped (b,h,nchunks,chunk). The
        # flattened extent is identical (batch*seqlen*nheads), so a single
        # logical region buffer carries both views (the runtime stages the F2
        # view contiguously).
        "dt": (batch, seqlen, nheads),
        # initial state / skip (F1 / F2 inputs)
        "h0": (batch, nheads, headdim, dstate),
        "D": (nheads,),
        # inter-chunk handoff buffers
        "cb": (batch, nchunks, ngroups, chunk, chunk),
        "dA_cumsum": (batch, nheads, nchunks, chunk),
        "summary_states": (batch, nchunks, nheads, headdim, dstate),
        "prev_states": (batch, nchunks, nheads, headdim, dstate),
        "final_state": (batch, nheads, headdim, dstate),
        # BACKWARD grad-handoff buffers (B2 -> B1 -> B0; Stage 3, fp32)
        "dh_last": (batch, nheads, headdim, dstate),
        "dchunk_states": (batch, nchunks, nheads, headdim, dstate),
        "dstates": (batch, nchunks, nheads, headdim, dstate),
        "dinp_diag": (batch, seqlen, nheads, headdim, dstate),
        "dA_cumsum_y": (batch, nheads, nchunks, chunk),
        "dA_cumsum_tail": (batch, nheads, nchunks, chunk),
    }
    return shapes[suffix]


def _buffer_shape(
    buffer_name: str,
    buffer_extent: int,
    shape_env: PathCModelShapeEnv | None,
) -> tuple[int, ...]:
    if shape_env is None:
        return (buffer_extent,)
    chunked_shape = _mamba3_chunked_fwd_region_buffer_shape(buffer_name, shape_env)
    if chunked_shape is not None:
        return chunked_shape
    name = _canonical_buffer_name(buffer_name)
    seq = shape_env.sequence_length
    hidden = shape_env.hidden_size
    q_heads = shape_env.attention_num_q_heads
    kv_heads = shape_env.attention_num_kv_heads
    head_dim = shape_env.attention_head_dim
    q_dim = q_heads * head_dim
    kv_dim = kv_heads * head_dim
    topk = shape_env.attention_sparse_topk
    if name in {
        "hidden",
        "mamba3_delta",
        "m2rnn_hidden",
        "m2rnn_delta",
        "attention_hidden",
        "hidden_after_mamba3",
        "hidden_after_m2rnn",
        "attention_out",
    }:
        return (seq * hidden,)
    if _is_mamba3_state_like_buffer(buffer_name, name):
        return (
            shape_env.mamba_num_heads
            * shape_env.mamba_head_dim
            * shape_env.mamba_state_dim,
        )
    if name == "mamba3_angle_state":
        return (
            max(1, shape_env.mamba_num_heads * shape_env.mamba_num_rope_angles),
        )
    if name == "mamba3_angle_grad_state":
        return (
            max(1, shape_env.mamba_num_heads * shape_env.mamba_num_rope_angles),
        )
    if name == "mamba3_angle_checkpoint":
        checkpoint_interval = MAMBA3_BWD_REPLAY_CHECKPOINT_INTERVAL
        checkpoint_count = (
            shape_env.sequence_length + checkpoint_interval - 1
        ) // checkpoint_interval
        return (
            (checkpoint_count + 1)
            * max(1, shape_env.mamba_num_heads * shape_env.mamba_num_rope_angles),
        )
    if name == "mamba3_h_checkpoint":
        checkpoint_interval = MAMBA3_BWD_REPLAY_CHECKPOINT_INTERVAL
        checkpoint_count = (
            shape_env.sequence_length + checkpoint_interval - 1
        ) // checkpoint_interval
        return (
            (checkpoint_count + 1)
            * shape_env.mamba_num_heads
            * shape_env.mamba_head_dim
            * shape_env.mamba_state_dim,
        )
    if name == "mamba3_conv_state":
        return (
            max(
                1,
                (shape_env.mamba_conv_kernel - 1) * shape_env.mamba_conv_channels,
            ),
        )
    if name == "mamba3_in_proj_weight":
        return (shape_env.mamba_in_proj_dim * hidden,)
    if name == "mamba3_out_proj_weight":
        return (hidden * shape_env.mamba_inner_dim,)
    if name == "mamba3_conv_weight":
        return (shape_env.mamba_conv_channels * shape_env.mamba_conv_kernel,)
    if name == "mamba3_conv_bias":
        return (shape_env.mamba_conv_channels,)
    if name in {"mamba3_dt_bias", "mamba3_D"}:
        return (shape_env.mamba_num_heads,)
    if name in {
        "mamba3_B_norm_weight",
        "mamba3_B_bias",
        "mamba3_C_norm_weight",
        "mamba3_C_bias",
    }:
        return (
            shape_env.mamba_effective_mimo_rank
            * shape_env.mamba_groups
            * shape_env.mamba_state_dim,
        )
    if name in {
        "mamba3_residual_to_m2rnn_norm_weight",
        "m2rnn_residual_to_attention_norm_weight",
        "residual_norm_weight",
        "final_norm_weight",
    }:
        return (hidden,)
    if name == "entry_rmsnorm_weight" or name.endswith("entry_rmsnorm_weight"):
        # Block A: the entry RMSNorm weight is a single ``hidden``-sized
        # vector applied to every row of the entry hidden state, mirroring
        # the eager ``layers.{first_in_region}.norm.weight`` parameter.
        return (hidden,)
    if name in {"target_ids", "target_mask"}:
        return (seq,)
    if name == "lm_head_weight":
        vocab = max(1, int(getattr(shape_env, "vocab_size", 0) or 0))
        return (vocab * hidden,)
    if name == "m2rnn_in_proj_weight":
        return (shape_env.m2rnn_in_proj_dim * hidden,)
    if name == "m2rnn_conv_weight":
        return (shape_env.m2rnn_conv_dim * shape_env.m2rnn_conv_kernel,)
    if name == "m2rnn_conv_bias":
        return (shape_env.m2rnn_conv_dim,)
    if name == "m2rnn_state_weight":
        return (
            shape_env.m2rnn_num_weight_heads
            * shape_env.m2rnn_v_head_dim
            * shape_env.m2rnn_v_head_dim,
        )
    if name in {"m2rnn_A_log", "m2rnn_dt_bias"}:
        return (shape_env.m2rnn_num_heads,)
    if name in {"m2rnn_D", "m2rnn_g_norm_weight"}:
        return (shape_env.m2rnn_num_heads * shape_env.m2rnn_v_head_dim,)
    if name == "m2rnn_out_proj_weight":
        return (hidden * shape_env.m2rnn_num_heads * shape_env.m2rnn_v_head_dim,)
    if name in {"m2rnn_h0", "m2rnn_h_state"}:
        return (
            shape_env.m2rnn_num_heads
            * shape_env.m2rnn_k_head_dim
            * shape_env.m2rnn_v_head_dim,
        )
    if name == "m2rnn_h_checkpoint":
        checkpoint_interval = M2RNN_BWD_REPLAY_CHECKPOINT_INTERVAL
        checkpoint_count = (
            shape_env.sequence_length + checkpoint_interval - 1
        ) // checkpoint_interval
        return (
            (checkpoint_count + 1)
            * shape_env.m2rnn_num_heads
            * shape_env.m2rnn_k_head_dim
            * shape_env.m2rnn_v_head_dim,
        )
    if name == "m2rnn_conv_state":
        return ((shape_env.m2rnn_conv_kernel - 1) * shape_env.m2rnn_conv_dim,)
    if name == "attention_q_proj_weight":
        return (q_dim * hidden,)
    if name == "attention_q_proj_bias":
        return (q_dim,)
    if name == "attention_sparse_kv_proj_weight":
        return (kv_dim * hidden,)
    if name == "attention_sparse_kv_proj_bias":
        return (kv_dim,)
    if name == "attention_rope_inv_freq":
        return (head_dim // 2,)
    if name == "attention_out_proj_weight":
        return (hidden * q_dim,)
    if name == "attention_out_proj_bias":
        return (hidden,)
    if name == "q_fp8":
        return (seq * q_heads * head_dim,)
    if name == "q_scale":
        return (seq * q_heads,)
    if name == "kv_fp8":
        return (seq * kv_heads * head_dim,)
    if name == "kv_scale":
        return (seq * kv_heads,)
    if name == "indices":
        return (seq * kv_heads * topk,)
    if name == "lse":
        return (seq * q_heads,)
    if name == "sparse_mla_sm_scale":
        return (1,)
    if name == "sparse_mla_sinks":
        return (q_heads,)
    if name == "sparse_mla_has_sinks":
        return (1,)
    if (
        name.endswith("_hidden")
        or name.endswith("_hidden_after")
        or name.endswith("_delta")
        or name.endswith("_out")
    ):
        return (seq * hidden,)
    return (buffer_extent,)


def _shape_literal(shape: Sequence[int]) -> str:
    if len(shape) == 1:
        return f"({int(shape[0])},)"
    return repr(tuple(int(dim) for dim in shape))


def _flattened_extent(shape: Sequence[int]) -> int:
    extent = 1
    for dim in shape:
        extent *= int(dim)
    return extent


_IDENTIFIER_RE = re.compile(r"\W+")


def _safe_identifier(name: object) -> str:
    identifier = _IDENTIFIER_RE.sub("_", str(name)).strip("_")
    if not identifier:
        identifier = "path_c_descriptor_region"
    if identifier[0].isdigit() or keyword.iskeyword(identifier):
        identifier = f"path_c_{identifier}"
    return identifier


def _validate_descriptors_match_nodes(
    nodes: Sequence[_ScheduleNodeView],
    descriptors: Sequence[PathCBrickScheduleDescriptor],
) -> None:
    for node, descriptor in zip(nodes, descriptors, strict=True):
        if node.op_name != descriptor.op_name:
            raise ValueError(
                f"descriptor op {descriptor.op_name!r} does not match "
                f"region node {node.name!r} op {node.op_name!r}"
            )


def _descriptor_chain_for_region_or_signature(
    region: Any,
    fallback_signature: Sequence[str],
) -> tuple[PathCBrickScheduleDescriptor, ...]:
    if getattr(region, "nodes", None) is not None:
        signature = tuple(node.op_name for node in _node_views_for_region(region))
    else:
        signature = tuple(fallback_signature)
    descriptors = (
        default_path_c_brick_schedule_descriptor_registry()
        .descriptors_for_signature(signature)
    )
    if descriptors is None:
        raise RuntimeError(
            f"no Path C brick descriptors registered for op signature {signature!r}"
        )
    return descriptors


def _require_path_c_region_graph(
    region: Any,
    *,
    function_name: str,
) -> Any:
    if getattr(region, "nodes", None) is not None:
        return region
    raise ValueError(
        f"{function_name} requires an explicit discovered Path C region; "
        "use build_mamba3_fp8_train_acceptance_fixture_region() only for the "
        "named acceptance fixture, or path_c_fusion_schedule_template(region) "
        "for model-derived brick chains"
    )


def _mamba3_fp8_train_acceptance_region(
    *,
    include_backward: bool = True,
    model_config: Any | None = None,
) -> PathCFusionRegion:
    return build_mamba3_fp8_train_acceptance_fixture_region(
        include_backward=include_backward,
        model_config=model_config,
    )


def _mamba3_fp8_train_acceptance_buffer_extent(
    model_config: Any | None = None,
) -> int:
    if model_config is None:
        return MAMBA3_FP8_TRAIN_BUFFER_EXTENT
    return int(
        getattr(
            model_config,
            "max_seq_length",
            getattr(model_config, "sequence_length", MAMBA3_FP8_TRAIN_BUFFER_EXTENT),
        )
    )


def _mamba3_fp8_train_acceptance_profile(
    model_config: Any | None = None,
) -> PathCFusionScheduleAcceptanceProfile:
    return PathCFusionScheduleAcceptanceProfile(
        op_signature=_MAMBA3_FP8_TRAIN_FWD_BWD_OP_SIGNATURE,
        schedule_id=MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_ID,
        schedule_name=MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_NAME,
        schedule_status=MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_STATUS,
        implementation_kind="production",
        missing_reason=_PRODUCTION_SCHEDULE_REASON,
        required_codegen_steps=(
            "route_symbol_brick_chain_region",
            "real_model_parameter_abi_contract",
            "z3_sync_async_schedule_points",
            "cache_key_shape_specialization_audit",
        ),
        entry_symbol="mamba3_m2rnn_attention_fp8_train_block",
        required_real_abi_inputs=MAMBA3_FP8_TRAIN_REQUIRED_REAL_ABI_INPUTS,
        buffer_extent=_mamba3_fp8_train_acceptance_buffer_extent(model_config),
        required_region_tags=("mamba3_fp8_train_acceptance",),
    )


def _mamba3_fp8_train_prototype_profile() -> PathCFusionScheduleAcceptanceProfile:
    return PathCFusionScheduleAcceptanceProfile(
        op_signature=_MAMBA3_FP8_TRAIN_FWD_BWD_OP_SIGNATURE,
        schedule_id="mamba3_m2rnn_attention_fp8_train_block_prototype_fwd_bwd",
        schedule_name=MAMBA3_FP8_TRAIN_PROTOTYPE_SCHEDULE_NAME,
        schedule_status=MAMBA3_FP8_TRAIN_PROTOTYPE_SCHEDULE_STATUS,
        implementation_kind="prototype",
        missing_reason="prototype schedule scaffold is not a production implementation",
        required_codegen_steps=(
            "route_symbol_brick_chain_region",
            "real_model_parameter_abi_contract",
        ),
        entry_symbol="mamba3_m2rnn_attention_fp8_train_block",
        required_real_abi_inputs=MAMBA3_FP8_TRAIN_REQUIRED_REAL_ABI_INPUTS,
        buffer_extent=MAMBA3_FP8_TRAIN_BUFFER_EXTENT,
        required_region_tags=("mamba3_fp8_train_acceptance",),
    )


def _profiled_descriptor_target_for_region(
    region: Any,
    profile: PathCFusionScheduleAcceptanceProfile,
) -> PathCFusionScheduleTarget:
    signature = tuple(node.op_name for node in _node_views_for_region(region))
    if signature != profile.op_signature:
        raise RuntimeError(
            f"acceptance profile {profile.schedule_id!r} does not match "
            f"region op signature {signature!r}"
        )
    if not _acceptance_profile_matches_region(profile, region):
        raise RuntimeError(
            f"acceptance profile {profile.schedule_id!r} requires region tags "
            f"{profile.required_region_tags!r}"
        )
    descriptors = _required_descriptors_for_signature(signature)
    return _dynamic_descriptor_target_for_region(
        region,
        descriptors,
        acceptance_profile=profile,
    )


def _acceptance_profile_matches_region(
    profile: PathCFusionScheduleAcceptanceProfile,
    region: Any,
) -> bool:
    required_tags = set(profile.required_region_tags)
    if not required_tags:
        return True
    metadata = getattr(region, "metadata", None)
    if not isinstance(metadata, Mapping):
        return False
    region_tags = set(metadata.get("path_c_acceptance_tags", ()))
    return required_tags.issubset(region_tags)


def _required_codegen_steps_from_descriptors(
    descriptors: Sequence[PathCBrickScheduleDescriptor],
) -> tuple[str, ...]:
    steps: list[str] = [
        "dynamic_region_graph_walk",
        "brick_descriptor_chain_resolution",
        "single_entry_tilelang_region",
    ]
    seen = set(steps)
    for descriptor in descriptors:
        for step in descriptor.required_codegen_steps:
            if step in seen:
                continue
            seen.add(step)
            steps.append(step)
    return tuple(steps)


def _brick_descriptor_statuses(
    descriptors: Sequence[PathCBrickScheduleDescriptor],
) -> tuple[str, ...]:
    return tuple(
        f"{descriptor.op_name}:{descriptor.implementation_status}"
        for descriptor in descriptors
    )


def _brick_production_fragment_statuses(
    descriptors: Sequence[PathCBrickScheduleDescriptor],
) -> tuple[str, ...]:
    return tuple(
        ":".join(
            (
                descriptor.op_name,
                descriptor.production_fragment_status,
                descriptor.production_source or "missing_source",
            )
        )
        for descriptor in descriptors
    )


def _brick_production_fragment_reasons(
    descriptors: Sequence[PathCBrickScheduleDescriptor],
) -> tuple[str, ...]:
    return tuple(
        ":".join(
            (
                descriptor.op_name,
                descriptor.production_fragment_status,
                descriptor.production_fragment_reason or "no_reason",
            )
        )
        for descriptor in descriptors
    )


def _brick_production_fragment_blockers(
    descriptors: Sequence[PathCBrickScheduleDescriptor],
) -> tuple[str, ...]:
    return tuple(
        ":".join(
            (
                descriptor.op_name,
                descriptor.production_fragment_status,
                descriptor.production_fragment_reason or "no_reason",
            )
        )
        for descriptor in descriptors
        if descriptor.production_fragment_status != "production_region_inlined"
    )


def _production_fragments_complete(
    descriptors: Sequence[PathCBrickScheduleDescriptor],
) -> bool:
    return all(
        descriptor.production_fragment_status == "production_region_inlined"
        for descriptor in descriptors
    )


def _schedule_generator_status(
    descriptors: Sequence[PathCBrickScheduleDescriptor],
) -> str:
    if _production_fragments_complete(descriptors):
        return "production_region_fragments"
    families = {descriptor.schedule_family for descriptor in descriptors}
    if families == {"loop_descriptor_dataflow"}:
        return "loop_per_brick_descriptor_fragments"
    return "mixed_descriptor_schedule_fragments"


def _effective_implementation_kind(
    acceptance_profile: PathCFusionScheduleAcceptanceProfile | None,
    descriptors: Sequence[PathCBrickScheduleDescriptor],
) -> str:
    if acceptance_profile is None:
        if _production_fragments_complete(descriptors):
            return "production"
        return "prototype"
    if (
        acceptance_profile.implementation_kind == "production"
        and not _production_fragments_complete(descriptors)
    ):
        return "scaffold"
    return acceptance_profile.implementation_kind


class PathCFusionScheduleRegistry:
    """Pattern registry that selects fused schedules from a Path C region graph."""

    def __init__(
        self,
        targets: tuple[PathCFusionScheduleTarget, ...] = (),
        *,
        brick_registry: PathCBrickScheduleDescriptorRegistry | None = None,
        acceptance_profiles: tuple[
            PathCFusionScheduleAcceptanceProfile,
            ...,
        ] = (),
        enable_dynamic_descriptor_targets: bool = True,
    ) -> None:
        self._targets: dict[tuple[str, ...], PathCFusionScheduleTarget] = {}
        self._brick_registry = (
            brick_registry or default_path_c_brick_schedule_descriptor_registry()
        )
        self._acceptance_profiles = {
            profile.op_signature: profile for profile in acceptance_profiles
        }
        self._enable_dynamic_descriptor_targets = enable_dynamic_descriptor_targets
        for target in targets:
            self.register(target)

    def register(
        self,
        target: PathCFusionScheduleTarget,
    ) -> "PathCFusionScheduleRegistry":
        if not isinstance(target, PathCFusionScheduleTarget):
            raise TypeError("target must be PathCFusionScheduleTarget")
        if not target.op_signature:
            raise ValueError("Path C fusion schedule target op_signature must not be empty")
        self._targets[target.op_signature] = target
        return self

    def select(self, region: PathCFusionRegion) -> PathCFusionScheduleTarget | None:
        if not isinstance(region, PathCFusionRegion):
            raise TypeError("region must be PathCFusionRegion")
        signature = tuple(node.op_name for node in region.nodes)
        target = self._targets.get(signature)
        if target is not None:
            return target
        if not self._enable_dynamic_descriptor_targets:
            return None
        descriptors = self._brick_registry.descriptors_for_signature(signature)
        if descriptors is None:
            return None
        acceptance_profile = self._acceptance_profiles.get(signature)
        if (
            acceptance_profile is not None
            and not _acceptance_profile_matches_region(acceptance_profile, region)
        ):
            acceptance_profile = None
        return _dynamic_descriptor_target_for_region(
            region,
            descriptors,
            acceptance_profile=acceptance_profile,
        )


class PathCFusionScheduleOptimizer:
    """FX-like Path C optimizer facade over graph build, AOTAutograd, and schedules."""

    def __init__(
        self,
        region_name: str,
        *,
        registry: PathCFusionScheduleRegistry | None = None,
        z3_sync: Z3SyncSpec | None = None,
        enable_aot_autograd: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._region_name = region_name
        self._registry = registry or default_path_c_fusion_schedule_registry()
        self._z3_sync = z3_sync or Z3SyncSpec.minimize_sync_async()
        self._enable_aot_autograd = enable_aot_autograd
        self._metadata = dict(metadata or {})
        self._surfaces: list[FusionKernelSurface] = []

    def add_kernel(
        self,
        surface: FusionKernelSurface,
    ) -> "PathCFusionScheduleOptimizer":
        if not isinstance(surface, FusionKernelSurface):
            raise TypeError("surface must be FusionKernelSurface")
        self._surfaces.append(surface)
        return self

    def add_kernels(
        self,
        surfaces: Sequence[FusionKernelSurface],
    ) -> "PathCFusionScheduleOptimizer":
        for surface in surfaces:
            self.add_kernel(surface)
        return self

    def enable_aot_autograd(self) -> "PathCFusionScheduleOptimizer":
        self._enable_aot_autograd = True
        return self

    def build_region(self) -> PathCFusionRegion:
        region = build_path_c_fusion_region(
            region_name=self._region_name,
            surfaces=tuple(self._surfaces),
            z3_sync=self._z3_sync,
            metadata=self._metadata,
        )
        if self._enable_aot_autograd:
            return build_path_c_aot_autograd_region(region)
        return region

    def select_schedule_target(
        self,
        region: PathCFusionRegion | None = None,
    ) -> PathCFusionScheduleTarget | None:
        return self._registry.select(region or self.build_region())

    def plan(self) -> PathCFusionScheduleOptimizerPlan:
        region = self.build_region()
        target = self.select_schedule_target(region)
        schedule_template = (
            _attested_schedule_template_for_target(target, region)
            if target is not None
            else None
        )
        plan = compile_path_c_region(
            region,
            schedule_template=schedule_template,
            schedule_name=target.schedule_name if target is not None else None,
            schedule_status=target.schedule_status
            if target is not None
            else MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_STATUS,
        )
        if not isinstance(plan, FusionCompilePlan):
            raise TypeError("compile_path_c_region unexpectedly returned an artifact")
        return PathCFusionScheduleOptimizerPlan(
            region=region,
            plan=plan,
            schedule_target=target,
        )

    def compile(
        self,
        *,
        tilelang_lowerer: Callable[..., Any],
        target_name: str = _path_c_default_target(),
    ) -> CompiledPathCRegion:
        """Compile the selected target through its descriptor schedule template."""

        region = self.build_region()
        target = self.select_schedule_target(region)
        if target is None:
            raise RuntimeError(
                f"no Path C fusion schedule target registered for op signature "
                f"{tuple(node.op_name for node in region.nodes)!r}"
            )
        schedule_template = _attested_schedule_template_for_target(target, region)
        compiled = compile_path_c_region(
            region,
            schedule_template=schedule_template,
            schedule_name=target.schedule_name,
            schedule_status=target.schedule_status,
            tilelang_lowerer=tilelang_lowerer,
            target=target_name,
        )
        if not isinstance(compiled, CompiledPathCRegion):
            raise TypeError("compile_path_c_region unexpectedly returned a plan")
        return compiled


def mamba3_fp8_train_fusion_schedule_target() -> PathCFusionScheduleTarget:
    """Return the explicit Mamba3 acceptance target built from its fixture graph."""

    return _profiled_descriptor_target_for_region(
        _mamba3_fp8_train_acceptance_region(include_backward=True),
        _mamba3_fp8_train_acceptance_profile(),
    )


def mamba3_fp8_train_prototype_schedule_target() -> PathCFusionScheduleTarget:
    """Return the explicit prototype target built from its fixture graph."""

    return _profiled_descriptor_target_for_region(
        _mamba3_fp8_train_acceptance_region(include_backward=True),
        _mamba3_fp8_train_prototype_profile(),
    )


def _required_descriptors_for_signature(
    op_signature: Sequence[str],
) -> tuple[PathCBrickScheduleDescriptor, ...]:
    descriptors = (
        default_path_c_brick_schedule_descriptor_registry()
        .descriptors_for_signature(op_signature)
    )
    if descriptors is None:
        raise RuntimeError(
            f"no Path C brick descriptors registered for op signature {tuple(op_signature)!r}"
        )
    return descriptors


def _dynamic_descriptor_target_for_region(
    region: Any,
    descriptors: tuple[PathCBrickScheduleDescriptor, ...],
    *,
    acceptance_profile: PathCFusionScheduleAcceptanceProfile | None = None,
) -> PathCFusionScheduleTarget:
    nodes = _node_views_for_region(region)
    signature = tuple(node.op_name for node in nodes)
    digest = sha256("|".join(signature).encode()).hexdigest()[:12]
    region_name = _safe_identifier(getattr(region, "name", "path_c_descriptor_region"))
    shape_env = _shape_env_for_region(region)
    buffer_extent = (
        shape_env.sequence_length
        if shape_env is not None
        else (
            acceptance_profile.buffer_extent
            if acceptance_profile is not None
            else DESCRIPTOR_DEFAULT_BUFFER_EXTENT
        )
    )
    internal_buffer_policy, loop_policy = _descriptor_codegen_policies_for_region(
        descriptors,
        shape_env,
    )
    descriptors = _descriptors_with_policy_fragment_statuses(
        descriptors,
        internal_buffer_policy=internal_buffer_policy,
        loop_policy=loop_policy,
        shape_env=shape_env,
    )
    implementation_kind = _effective_implementation_kind(
        acceptance_profile,
        descriptors,
    )
    train_step_output_abi = _region_requires_train_step_output_abi(region, nodes)
    physical_abi_policy = _physical_abi_policy_for_region(
        nodes,
        shape_env=shape_env,
        internal_buffer_policy=internal_buffer_policy,
        loop_policy=loop_policy,
        train_step_output_abi=train_step_output_abi,
    )
    max_rows_per_launch = (
        DESCRIPTOR_DEFAULT_MAX_ROWS_PER_LAUNCH
        if (
            shape_env is not None
            and loop_policy == DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN
        )
        else None
    )
    row_dispatch_mode = DESCRIPTOR_DEFAULT_ROW_DISPATCH_MODE
    schedule_name = (
        acceptance_profile.schedule_name
        if acceptance_profile is not None
        else f"{region_name}:descriptor_generated_fwd_bwd"
    )
    required_codegen_steps = _required_codegen_steps_from_descriptors(descriptors)
    required_real_abi_inputs = (
        acceptance_profile.required_real_abi_inputs
        if acceptance_profile is not None
        else _real_abi_inputs_for_nodes(nodes)
    )
    extra_steps: list[str] = []
    if required_real_abi_inputs:
        extra_steps.append("real_model_parameter_abi_contract")
    if getattr(getattr(region, "z3_sync", None), "enabled", False):
        extra_steps.append("z3_sync_async_schedule_points")
    if shape_env is not None:
        extra_steps.append("cache_key_shape_specialization_audit")
    if train_step_output_abi:
        extra_steps.append("train_step_scalar_output_abi")
        extra_steps.append("train_step_suffix_loss_input_abi")
    if extra_steps:
        seen = set(required_codegen_steps)
        required_codegen_steps = (
            *required_codegen_steps,
            *(step for step in extra_steps if step not in seen),
        )
    if acceptance_profile is not None:
        seen = set(required_codegen_steps)
        required_codegen_steps = (
            *required_codegen_steps,
            *(
                step
                for step in acceptance_profile.required_codegen_steps
                if step not in seen
            ),
        )
    schedule_id = (
        acceptance_profile.schedule_id
        if acceptance_profile is not None
        else f"path_c_descriptor_chain_{digest}"
    )
    dynamic_schedule_status = (
        MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_STATUS
        if implementation_kind == "production"
        else "descriptor_codegen_scaffold"
    )
    dynamic_missing_reason = (
        (
            "descriptor-generated Path C schedule is selected from the model "
            "route graph with all brick fragments production-inlined; it remains "
            "untrusted until compile, benchmark, profiling, memory, and cache "
            "receipts pass"
        )
        if implementation_kind == "production"
        else (
            "descriptor-generated Path C schedule was selected from the "
            "region graph, but it is not a named production acceptance target"
        )
    )
    return PathCFusionScheduleTarget(
        schedule_id=schedule_id,
        schedule_name=schedule_name,
        op_signature=signature,
        schedule_status=acceptance_profile.schedule_status
        if acceptance_profile is not None
        else dynamic_schedule_status,
        implementation_kind=implementation_kind,
        missing_reason=(
            acceptance_profile.missing_reason
            if acceptance_profile is not None
            else dynamic_missing_reason
        ),
        required_codegen_steps=required_codegen_steps,
        schedule_template=make_path_c_descriptor_schedule_template(
            descriptors,
            entry_symbol=acceptance_profile.entry_symbol
            if acceptance_profile is not None
            else region_name,
            buffer_extent=buffer_extent,
            shape_env=shape_env,
            internal_buffer_policy=internal_buffer_policy,
            loop_policy=loop_policy,
            physical_abi_policy=physical_abi_policy,
            train_step_output_abi=train_step_output_abi,
            max_rows_per_launch=max_rows_per_launch,
            row_dispatch_mode=row_dispatch_mode,
        ),
        required_real_abi_inputs=required_real_abi_inputs,
        brick_descriptors=descriptors,
        buffer_extent=buffer_extent,
        internal_buffer_policy=internal_buffer_policy,
        loop_policy=loop_policy,
        physical_abi_policy=physical_abi_policy,
        max_rows_per_launch=max_rows_per_launch,
        row_dispatch_mode=row_dispatch_mode,
    )


def _region_requires_train_step_output_abi(
    region: Any,
    nodes: Sequence[_ScheduleNodeView],
) -> bool:
    metadata = getattr(region, "metadata", {}) or {}
    if not bool(metadata.get("path_c_enable_fused_suffix_train_step_abi", False)):
        return False
    if bool(metadata.get("path_c_acceptance_fixture_abi", False)):
        return False
    if metadata.get("path_c_chain_source_region"):
        return False
    if not metadata.get("path_c_bricks"):
        return False
    return any(str(node.op_name).endswith("_bwd") for node in nodes)


def _physical_abi_policy_for_region(
    nodes: Sequence[_ScheduleNodeView],
    *,
    shape_env: PathCModelShapeEnv | None,
    internal_buffer_policy: str,
    loop_policy: str,
    train_step_output_abi: bool = False,
) -> str:
    internal_buffers = _internal_buffers_for_nodes(
        nodes,
        shape_env=shape_env,
        internal_buffer_policy=internal_buffer_policy,
        loop_policy=loop_policy,
    )
    external_buffer_count = len(_external_buffers_for_nodes(nodes, internal_buffers))
    if train_step_output_abi:
        external_buffer_count += len(_TRAIN_STEP_SCALAR_OUTPUT_ABI_NAMES)
        external_buffer_count += len(_TRAIN_STEP_SUFFIX_LOSS_INPUT_ABI_NAMES)
        external_buffer_count += len(_TRAIN_STEP_SUFFIX_LOSS_PARAMETER_GRAD_ABI_NAMES)
    if (
        shape_env is not None
        and internal_buffer_policy == DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN
        and loop_policy == DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN
    ):
        return DESCRIPTOR_PHYSICAL_ABI_POLICY_BANKED_BY_ROLE
    if external_buffer_count > DESCRIPTOR_PORTABLE_KERNEL_BUFFER_LIMIT:
        return DESCRIPTOR_PHYSICAL_ABI_POLICY_BANKED_BY_ROLE
    return DESCRIPTOR_PHYSICAL_ABI_POLICY_DIRECT


def _descriptor_codegen_policies_for_region(
    descriptors: Sequence[PathCBrickScheduleDescriptor],
    shape_env: PathCModelShapeEnv | None,
) -> tuple[str, str]:
    if shape_env is None:
        return (
            DESCRIPTOR_INTERNAL_BUFFER_POLICY_SCALAR_LOCAL,
            DESCRIPTOR_LOOP_POLICY_FLAT,
        )
    internal_buffer_policy = DESCRIPTOR_INTERNAL_BUFFER_POLICY_SCALAR_LOCAL
    loop_policy = DESCRIPTOR_LOOP_POLICY_FLAT
    for descriptor in descriptors:
        preferred_internal = _validated_internal_buffer_policy(
            descriptor.preferred_internal_buffer_policy
        )
        preferred_loop = _validated_loop_policy(descriptor.preferred_loop_policy)
        if preferred_internal == DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN:
            internal_buffer_policy = preferred_internal
        if preferred_loop == DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN:
            loop_policy = preferred_loop
    return internal_buffer_policy, loop_policy


def _descriptors_with_policy_fragment_statuses(
    descriptors: Sequence[PathCBrickScheduleDescriptor],
    *,
    internal_buffer_policy: str,
    loop_policy: str,
    shape_env: PathCModelShapeEnv | None,
) -> tuple[PathCBrickScheduleDescriptor, ...]:
    if not (
        shape_env is not None
        and internal_buffer_policy == DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN
        and loop_policy == DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN
    ):
        return tuple(descriptors)

    adjusted: list[PathCBrickScheduleDescriptor] = []
    op_occurrences = Counter(descriptor.op_name for descriptor in descriptors)
    for descriptor in descriptors:
        if not _descriptor_production_policy_matches(
            descriptor,
            internal_buffer_policy=internal_buffer_policy,
            loop_policy=loop_policy,
            shape_env=shape_env,
            op_occurrences=op_occurrences,
        ):
            adjusted.append(descriptor)
            continue
        adjusted.append(
            replace(
                descriptor,
                required_codegen_steps=_append_unique_codegen_step(
                    descriptor.required_codegen_steps,
                    descriptor.production_fragment_codegen_step,
                ),
                production_fragment_status="production_region_inlined",
                production_fragment_reason=descriptor.production_fragment_inlined_reason,
            )
        )
    return tuple(adjusted)


def _descriptor_production_policy_matches(
    descriptor: PathCBrickScheduleDescriptor,
    *,
    internal_buffer_policy: str,
    loop_policy: str,
    shape_env: PathCModelShapeEnv,
    op_occurrences: Mapping[str, int],
) -> bool:
    if not descriptor.production_fragment_policy:
        return False
    if not descriptor.production_fragment_codegen_step:
        return False
    if not descriptor.production_fragment_inlined_reason:
        return False
    if (
        descriptor.max_production_hidden_size is not None
        and shape_env.hidden_size > descriptor.max_production_hidden_size
    ):
        return False
    if (
        descriptor.max_production_op_occurrences is not None
        and op_occurrences.get(descriptor.op_name, 0)
        > descriptor.max_production_op_occurrences
    ):
        min_hidden_size = descriptor.max_production_op_occurrences_min_hidden_size
        if min_hidden_size is None or shape_env.hidden_size >= min_hidden_size:
            return False
    return (
        descriptor.production_fragment_policy == loop_policy
        and descriptor.preferred_internal_buffer_policy == internal_buffer_policy
        and descriptor.preferred_loop_policy == loop_policy
    )


def _append_unique_codegen_step(
    steps: Sequence[str],
    step: str,
) -> tuple[str, ...]:
    values = tuple(str(value) for value in steps)
    if step in values:
        return values
    return (*values, step)


def _real_abi_inputs_for_nodes(
    nodes: Sequence[_ScheduleNodeView],
) -> tuple[str, ...]:
    internal_buffers = _internal_buffers_for_nodes(nodes)
    external_buffers = _external_buffers_for_nodes(nodes, internal_buffers)
    return tuple(
        name
        for name in external_buffers
        if _canonical_buffer_name(name)
        in (
            *MAMBA3_FP8_TRAIN_REQUIRED_REAL_ABI_INPUTS,
            *_GENERIC_MODEL_REAL_ABI_INPUT_SUFFIXES,
        )
    )


def default_path_c_fusion_schedule_registry() -> PathCFusionScheduleRegistry:
    """Return the default descriptor-backed registry for Path C fusion."""

    return PathCFusionScheduleRegistry()


def prototype_path_c_fusion_schedule_registry() -> PathCFusionScheduleRegistry:
    """Return a registry that selects descriptor-generated prototype schedules."""

    return PathCFusionScheduleRegistry(
        acceptance_profiles=(_mamba3_fp8_train_prototype_profile(),),
    )


def select_path_c_fusion_schedule_target(
    region: PathCFusionRegion,
    *,
    registry: PathCFusionScheduleRegistry | None = None,
) -> PathCFusionScheduleTarget | None:
    """Select the fused schedule target matching ``region``'s op signature."""

    return (registry or default_path_c_fusion_schedule_registry()).select(region)


def path_c_fusion_schedule_template(
    region: PathCFusionRegion,
    *,
    registry: PathCFusionScheduleRegistry | None = None,
) -> Any:
    """Generate a descriptor schedule from a discovered Path C model region."""

    target = select_path_c_fusion_schedule_target(region, registry=registry)
    if target is None:
        raise RuntimeError(
            f"no Path C fusion schedule target registered for op signature "
            f"{tuple(node.op_name for node in region.nodes)!r}"
        )
    return target.schedule_template(region)


def plan_path_c_fusion_schedule_for_region(
    region: PathCFusionRegion,
    *,
    include_backward: bool = True,
    registry: PathCFusionScheduleRegistry | None = None,
) -> PathCFusionScheduleOptimizerPlan:
    """Plan a Path C fused schedule from an already-discovered model region."""

    if not isinstance(region, PathCFusionRegion):
        raise TypeError("region must be PathCFusionRegion")
    has_backward = any(node.op_name.endswith("_bwd") for node in region.nodes)
    return (
        PathCFusionScheduleOptimizer(
            region.name,
            registry=registry,
            metadata=region.metadata,
            enable_aot_autograd=include_backward and not has_backward,
        )
        .add_kernels(_surfaces_from_region(region))
        .plan()
    )


def _derived_max_rows_per_launch_for_region(
    candidate_region: PathCFusionRegion,
) -> int:
    """Device-derived watchdog-safe row window for a candidate segment.

    Replaces the hardcoded ``DESCRIPTOR_DEFAULT_MAX_ROWS_PER_LAUNCH = 64`` passed
    to the launcher_chunks switch (design §3.4 / §4.7):
    ``min(floor(watchdog_window * safety / per_row_time), descriptor_default)``.
    At the M4 Max / local_gb10_quarter scale the watchdog-safe bound (~1024)
    exceeds the descriptor default, so this lands on the hand-tuned 64; on a
    tighter-watchdog device it shrinks below 64 (and the backstop RAISES if even
    one row cannot fit). Needs no generated source -- only the shape env + the
    per-op coefficient -- so it is free at the switch point.
    """

    from cppmega_mlx.runtime.path_c_device_caps import device_caps
    from cppmega_mlx.runtime import path_c_segment_estimator as _estimator

    caps = device_caps()
    env = _shape_env_for_region(candidate_region)
    if env is None or caps.watchdog_window_s is None:
        return DESCRIPTOR_DEFAULT_MAX_ROWS_PER_LAUNCH
    op_names = tuple(
        _path_c_descriptor_stage_node_op_name(node)
        for node in candidate_region.nodes
    )
    # Use the SLOWEST (largest per-row) non-recurrent op in the segment to bound
    # the row window (the recurrent ops are time-chunked, not row-timed).
    est_gpu_time_s = sum(
        _estimator.est_op_gpu_time_s(op_name, env, caps) for op_name in op_names
    )
    seq = max(1, int(env.sequence_length))
    est = _estimator.SegmentEstimate(
        logical_shared_bytes=0,
        physical_shared_bytes=0,
        buffer_arg_count=0,
        est_gpu_time_s=est_gpu_time_s,
        is_recurrent=False,
        msl_source_bytes=0,
        per_row_time_s=est_gpu_time_s / seq,
    )
    return _estimator.derived_max_rows_per_launch(
        est, caps, descriptor_default=DESCRIPTOR_DEFAULT_MAX_ROWS_PER_LAUNCH
    )


def _path_c_segment_estimate_backstop(
    *,
    candidate_region: PathCFusionRegion,
    direct_target: Any,
    parameter_count: int,
    execution_phase: str,
    segment_op_count: int,
) -> str | None:
    """Device-grounded MSL-size / watchdog-time backstop for one candidate segment.

    Returns a ``first_failure`` string (shrink the segment, try fewer ops) when an
    estimate exceeds a device limit but the segment can still be split; RAISES
    :class:`PathCSplitInfeasible` when a single-op (irreducible) segment exceeds a
    hard limit (RULE #1, design §6.3). Returns ``None`` when the segment passes.

    Only fires on a Metal device whose caps carry the relevant ceiling/watchdog;
    CUDA caps have ``msl_pipeline_state_ceiling_bytes is None`` and
    ``has_command_buffer_watchdog is False`` so this is a no-op there.
    """

    from cppmega_mlx.runtime.path_c_device_caps import device_caps
    from cppmega_mlx.runtime import path_c_segment_estimator as _estimator

    caps = device_caps()
    # Nothing to back-stop on CUDA (no compiler pipeline crash, no watchdog).
    if caps.msl_pipeline_state_ceiling_bytes is None and not (
        caps.has_command_buffer_watchdog
    ):
        return None

    prim_func = direct_target.schedule_template(candidate_region)
    source = getattr(prim_func, "_cppmega_path_c_generated_source", "") or ""
    op_names = tuple(
        _path_c_descriptor_stage_node_op_name(node)
        for node in candidate_region.nodes
    )
    env = _shape_env_for_region(candidate_region)
    is_recurrent = any(_node_is_recurrent_state_scan(node) for node in candidate_region.nodes)
    est = _estimator.estimate_segment_from_source(
        source=source,
        op_names=op_names,
        buffer_arg_count=parameter_count,
        env=env,
        caps=caps,
        is_recurrent=is_recurrent,
        alloc_shared_re=_ALLOC_SHARED_LINE_RE,
        dtype_nbytes=_DTYPE_NBYTES,
        flattened_extent=_flattened_extent,
    )

    # (c) forward MSL pipeline-state ceiling (Metal-only).
    if (
        caps.msl_pipeline_state_ceiling_bytes is not None
        and execution_phase == DESCRIPTOR_EXECUTION_STAGE_FORWARD
        and est.msl_source_bytes > caps.msl_pipeline_state_ceiling_bytes
    ):
        if segment_op_count == 1:
            raise PathCSplitInfeasible(
                candidate_region.name,
                "msl-pipeline-size",
                est.msl_source_bytes,
                caps.msl_pipeline_state_ceiling_bytes,
                op_name=op_names[0] if op_names else "",
            )
        return (
            f"forward segment MSL source {est.msl_source_bytes} B exceeds the "
            f"compiler pipeline-state ceiling {caps.msl_pipeline_state_ceiling_bytes} B "
            f"-- shrink segment"
        )

    # (d) backward watchdog-time budget (Metal-only). The recurrent/independent
    # row/time chunking already switched the dispatch above; here we check the
    # PER-LAUNCH time still fits after chunking, and RAISE if even a 1-op segment
    # at the minimal window cannot fit.
    if (
        caps.has_command_buffer_watchdog
        and execution_phase == DESCRIPTOR_EXECUTION_STAGE_BACKWARD
        and caps.watchdog_window_s is not None
    ):
        budget_s = caps.watchdog_window_s * caps.safety_margin
        if est.est_gpu_time_s > budget_s:
            # Recurrent ops time-chunk; independent ops row-window. Estimate the
            # per-launch time after the derived chunking.
            if est.is_recurrent:
                chunks = _estimator.derived_time_chunk_count(est, caps)
                per_launch = est.est_gpu_time_s / max(1, chunks)
            else:
                rows = _estimator.derived_max_rows_per_launch(est, caps)
                seq = max(1, int(env.sequence_length)) if env is not None else 1
                per_launch = est.est_gpu_time_s * (rows / seq)
            if per_launch > budget_s:
                if segment_op_count == 1:
                    raise PathCSplitInfeasible(
                        candidate_region.name,
                        "watchdog",
                        per_launch,
                        budget_s,
                        op_name=op_names[0] if op_names else "",
                    )
                return (
                    f"backward segment est per-launch {per_launch:.3f}s exceeds the "
                    f"watchdog budget {budget_s:.3f}s even after chunking -- shrink "
                    f"segment"
                )
    return None


def plan_path_c_direct_fusion_chain_for_region(
    region: PathCFusionRegion,
    *,
    include_backward: bool = True,
    max_kernel_buffers: int | _ResolveFromTarget = _RESOLVE_BUFFER_LIMIT_FROM_CAPS,
    max_segment_nodes: int | None = None,
    forward_max_segment_nodes: int | None | _ResolveFromTarget = (
        _RESOLVE_FORWARD_CAP_FROM_TARGET
    ),
    backward_max_segment_nodes: int | None | _ResolveFromTarget = (
        _RESOLVE_BACKWARD_CAP_FROM_TARGET
    ),
    registry: PathCFusionScheduleRegistry | None = None,
) -> PathCFusionScheduleChainPlan:
    """Greedily split a Path C region into direct-buffer fused segments.

    This is the generic escape hatch when a single direct-buffer train-block
    would exceed Metal's portable buffer slot limit.  It never falls back to
    dtype-bank packing; segments that cannot be expressed with direct buffers
    under ``max_kernel_buffers`` are reported as blocked.

    ``forward_max_segment_nodes`` is a Metal-only cap on the number of ops a
    FORWARD-phase fused segment may contain. The default
    (``_RESOLVE_FORWARD_CAP_FROM_TARGET``) is resolved at call time from the
    active lowering target: on Metal it is
    :data:`METAL_FORWARD_MAX_SEGMENT_NODES` = 2 (so the heavy 4-op forward
    segment ``m2rnn + residual_rmsnorm + attention_qkv_projection +
    sparse_mla_fp8_apply`` — whose ~176-199 KB MSL crashes Metal's
    ``MTLCompilerService`` at ``newComputePipelineState`` with
    ``XPC_ERROR_CONNECTION_INTERRUPTED`` — splits into two smaller
    pipeline-safe kernels); on CUDA it is ``None`` (no cap, the CUDA compiler
    has no such pipeline-state size limit, so CUDA fusion is unchanged).

    ``backward_max_segment_nodes`` is the sibling Metal-only cap for the
    reverse-autograd BACKWARD phase. The default
    (``_RESOLVE_BACKWARD_CAP_FROM_TARGET``) resolves to
    :data:`METAL_BACKWARD_MAX_SEGMENT_NODES` = 1 on Metal and ``None`` on CUDA.
    Capping backward segments at 1 op splits the 3-op backward mega-kernel
    ``sparse_mla_fp8_apply_bwd + attention_qkv_projection_bwd +
    residual_rmsnorm_bwd`` (region ``..._chain_7_10``, ~115 KB, ~10-25s GPU time
    -> macOS GPU watchdog kill) into one segment per op, which both shrinks each
    command buffer AND lets the per-row-INDEPENDENT heavy ops
    (``sparse_mla_fp8_apply_bwd`` / ``attention_qkv_projection_bwd``;
    :data:`_ROW_CHUNKED_INDEPENDENT_BACKWARD_OPS`) get launcher_chunks
    row-windowing so each command buffer holds only a row window (~0.42s) -- well
    under the watchdog. Pass an explicit value to override; pass ``None`` to
    disable (CUDA keeps greedy backward fusion).
    """

    if not isinstance(region, PathCFusionRegion):
        raise TypeError("region must be PathCFusionRegion")
    if isinstance(max_kernel_buffers, _ResolveFromTarget):
        # Resolve the kernel buffer-argument limit from the device-capability
        # probe: Metal -> caps.buffer_arg_limit == 31 (Apple family ABI const,
        # the portable floor), CUDA -> the unbounded sentinel (no 31-arg wall).
        # The live count (_kernel_parameter_count_for_target) and the
        # parameter_count > limit break already exist and fail loud; only the
        # LIMIT now comes from the queried/preset device cap, not a literal.
        from cppmega_mlx.runtime.path_c_device_caps import device_caps

        max_kernel_buffers = int(device_caps().buffer_arg_limit)
    if max_kernel_buffers <= 0:
        raise ValueError("max_kernel_buffers must be positive")
    if max_segment_nodes is not None and max_segment_nodes <= 0:
        raise ValueError("max_segment_nodes must be positive when provided")
    # Resolve the forward/backward fusion caps from the DEVICE-CAPABILITY probe
    # (design step 8 / §3.1), no longer from the METAL_*_MAX_SEGMENT_NODES
    # literals: caps.forward_max_segment_nodes is the compiler MSL-band op cap
    # (Metal 2; CUDA None -> monolithic), caps.backward_max_segment_nodes is the
    # watchdog per-op-isolation cap (Metal 1; CUDA None). The MSL-byte and
    # watchdog-time ESTIMATE predicates below (estimate_segment) layer on top as
    # device-grounded backstops that fire on larger region shapes. CUDA, with no
    # compiler crash and no watchdog, resolves both to None and stays monolithic.
    _caps_for_planner = None
    if isinstance(forward_max_segment_nodes, _ResolveFromTarget) or isinstance(
        backward_max_segment_nodes, _ResolveFromTarget
    ):
        from cppmega_mlx.runtime.path_c_device_caps import device_caps

        _caps_for_planner = device_caps()
    if isinstance(forward_max_segment_nodes, _ResolveFromTarget):
        forward_max_segment_nodes = _caps_for_planner.forward_max_segment_nodes
    if (
        forward_max_segment_nodes is not None
        and forward_max_segment_nodes <= 0
    ):
        raise ValueError(
            "forward_max_segment_nodes must be positive when provided"
        )
    if isinstance(backward_max_segment_nodes, _ResolveFromTarget):
        backward_max_segment_nodes = _caps_for_planner.backward_max_segment_nodes
    if (
        backward_max_segment_nodes is not None
        and backward_max_segment_nodes <= 0
    ):
        raise ValueError(
            "backward_max_segment_nodes must be positive when provided"
        )
    # Metal-only watchdog mitigation for the heavy per-row-INDEPENDENT FORWARD
    # ops (attention_qkv_projection / sparse_mla_fp8_apply): isolate each in its
    # own 1-op segment and switch that segment to launcher_chunks row-windowing.
    # Gated explicitly on the lowering target AND on the forward cap being active
    # (RULE #1: explicit by target, no silent fallback): CUDA -- which has no
    # per-command-buffer GPU watchdog -- resolves the forward cap to ``None`` and
    # so keeps the monolithic grid_chunks forward path + greedy forward fusion
    # byte-for-byte unchanged. A caller that explicitly passes
    # ``forward_max_segment_nodes=None`` disables ALL Metal forward mitigations
    # (the pipeline-state split AND this watchdog row-chunk isolation) together,
    # so the row-chunk isolation is bound to the SAME switch as the forward cap.
    forward_row_chunk_isolation = (
        _path_c_default_target() == "metal"
        and forward_max_segment_nodes is not None
    )
    # Build the symbolic backward graph when requested. A region whose nodes
    # ALREADY include a synthesized (aot-derived) backward op is treated as
    # already-built and left as-is. The chunked mamba3 brick appends its OWN
    # explicit B2/B1/B0 ``_bwd`` surfaces (backward="owner_output") into the
    # FORWARD node list; those must NOT short-circuit the backward build --
    # ``build_path_c_aot_autograd_region`` relocates them to their correct
    # reverse position AND derives the remaining non-mamba brick backward ops.
    already_has_synthesized_backward = any(
        node.op_name.endswith("_bwd")
        and str(getattr(node, "backward", "")) != "owner_output"
        for node in region.nodes
    )
    working_region = (
        build_path_c_aot_autograd_region(region)
        if include_backward and not already_has_synthesized_backward
        else region
    )
    nodes = tuple(working_region.nodes)
    segments: list[PathCFusionScheduleChainSegment] = []
    start = 0
    selector = registry or default_path_c_fusion_schedule_registry()
    while start < len(nodes):
        best: PathCFusionScheduleChainSegment | None = None
        first_failure: str | None = None
        for end in range(start + 1, len(nodes) + 1):
            if (
                max_segment_nodes is not None
                and end - start > max_segment_nodes
            ):
                first_failure = (
                    f"direct-buffer segment reached max_segment_nodes="
                    f"{max_segment_nodes}"
                )
                break
            candidate_region = _subregion_from_nodes(
                working_region,
                start=start,
                end=end,
                name=f"{working_region.name}_chain_{start}_{end}",
            )
            execution_phase = _path_c_schedule_segment_execution_phase(
                candidate_region.nodes
            )
            if execution_phase == "mixed":
                first_failure = (
                    "direct-chain segment would cross the forward/backward "
                    "execution boundary required by the loss cotangent bridge"
                )
                break
            # FORWARD-only Metal pipeline-state cap. Forward fused segments
            # larger than ``forward_max_segment_nodes`` ops generate MSL whose
            # ~176-199 KB device kernel crashes Metal's MTLCompilerService at
            # newComputePipelineState (XPC_ERROR_CONNECTION_INTERRUPTED). Stop
            # extending the forward segment here so it splits into smaller
            # pipeline-safe kernels. Backward segments are exempt (they keep
            # greedy fusion up to the buffer limit).
            #
            # The cap is FURTHER lowered to 1 for any forward window containing a
            # row-chunked-independent forward op (attention_qkv_projection /
            # sparse_mla_fp8_apply; see _ROW_CHUNKED_INDEPENDENT_FORWARD_OPS).
            # That isolates each heavy forward op in its OWN segment so (a) it can
            # be launcher_chunks row-windowed below to stay under the macOS GPU
            # watchdog (the monolithic 3-op forward segment ..._chain_4_6 was
            # killed at ~9.6s at full scale), and (b) sparse_mla_fp8_apply's KV
            # reads of the full-sequence workspace written by the SEPARATE
            # attention_qkv_projection segment stay bitwise-correct (a fused row
            # window would read KV positions not yet written). Mirrors the
            # backward cap=1 isolation of the per-row-independent heavy bwd ops.
            effective_forward_cap = (
                _effective_forward_max_segment_nodes_for_window(
                    candidate_region.nodes,
                    forward_max_segment_nodes,
                )
                if forward_row_chunk_isolation
                else forward_max_segment_nodes
            )
            if (
                execution_phase == DESCRIPTOR_EXECUTION_STAGE_FORWARD
                and effective_forward_cap is not None
                and end - start > effective_forward_cap
            ):
                first_failure = (
                    f"forward direct-buffer segment reached "
                    f"forward_max_segment_nodes={effective_forward_cap} "
                    f"(Metal newComputePipelineState size cap"
                    + (
                        " / row-chunked forward op isolation"
                        if effective_forward_cap != forward_max_segment_nodes
                        else ""
                    )
                    + ")"
                )
                break
            # BACKWARD-only Metal watchdog cap. Backward fused segments larger
            # than ``backward_max_segment_nodes`` ops run as ONE grid_chunks
            # command buffer whose GPU time exceeds the macOS GPU watchdog
            # (kIOGPUCommandBufferCallbackErrorTimeout) -- the 3-op
            # ..._chain_7_10 mega-kernel was the FIRST backward segment to die.
            # Stop extending the backward segment so each backward op lands in
            # its own command buffer (and the per-row-independent heavy ops can
            # then be row-windowed below). Forward segments are unaffected.
            if (
                execution_phase == DESCRIPTOR_EXECUTION_STAGE_BACKWARD
                and backward_max_segment_nodes is not None
                and end - start > backward_max_segment_nodes
            ):
                first_failure = (
                    f"backward direct-buffer segment reached "
                    f"backward_max_segment_nodes={backward_max_segment_nodes} "
                    f"(Metal GPU watchdog command-buffer cap)"
                )
                break
            target = selector.select(candidate_region)
            if target is None:
                first_failure = (
                    f"no descriptor target for op signature "
                    f"{tuple(node.op_name for node in candidate_region.nodes)!r}"
                )
                break
            direct_target = _target_with_physical_abi_policy(
                target,
                candidate_region,
                DESCRIPTOR_PHYSICAL_ABI_POLICY_DIRECT,
            )
            # Intra-segment TIME-CHUNKING for the recurrent reverse-time-scan
            # BACKWARD segments (m2rnn_bwd AND mamba3_mimo_bwd; see
            # _TIME_CHUNKED_RECURRENT_BACKWARD_OPS).
            #
            # Each recurrent reverse scan is a single
            # `for time_rev in T.serial(0, S)` over the whole sequence inside ONE
            # threadgroup. Under the default grid_chunks dispatch that is ONE Metal
            # command buffer spanning all S time steps -> multi-second GPU time ->
            # macOS GPU watchdog kill (kIOGPUCommandBufferCallbackErrorTimeout).
            #
            # Switching the segment to launcher_chunks makes its compiled kernel
            # declare path_c_row_chunk_index / path_c_row_subchunk_index and emit
            # the per-row reverse scan body
            # (`for time_rev in T.serial(row, row + 1)`), so the runtime can drive
            # the reverse scan as K separate command buffers over TIME, carrying
            # the reverse adjoint scan state across launches via caller-owned
            # buffers. Every other segment (forward + per-row-INDEPENDENT backward
            # ops like attention_qkv_projection_bwd / sparse_mla_fp8_apply_bwd /
            # residual_rmsnorm_bwd) keeps grid_chunks -- only the long reverse
            # scans need the watchdog-safe launcher split.
            #
            # The original time-chunking (commit ac412fb) covered ONLY
            # mamba3_mimo_bwd; m2rnn_bwd runs FIRST in the reverse chain and tripped
            # the watchdog before mamba3 was reached (verified timeout at region
            # ..._chain_10_11 / op=m2rnn_bwd), so both recurrent ops are included.
            #
            # ROW-WINDOWING for the per-row-INDEPENDENT heavy backward ops
            # (sparse_mla_fp8_apply_bwd / attention_qkv_projection_bwd; see
            # _ROW_CHUNKED_INDEPENDENT_BACKWARD_OPS). At full scale each of these
            # times out the macOS GPU watchdog even ISOLATED in its own 1-op
            # grid_chunks command buffer (~10-12s). They have NO carried
            # reverse-time state (each output row depends only on its own input
            # row + shared weights), so launcher_chunks splits them into K command
            # buffers over the independent ROW axis with zero cross-launch state
            # (_row_phased_launcher_carry_buffers_for_nodes adds carry buffers ONLY
            # for mamba3/m2rnn, never for these). Weight grads still accumulate
            # correctly via the path_c_first_row_launch one-time owner-grad zero.
            # Both recurrent and independent watchdog-heavy ops therefore take the
            # SAME launcher_chunks treatment; the carry-buffer plumbing distinguishes
            # them downstream. (Metal-only watchdog split; CUDA keeps grid_chunks.)
            # The recurrent reverse-time scans (mamba3_mimo_bwd / m2rnn_bwd) are
            # now identified STRUCTURALLY via _node_is_recurrent_state_scan (the
            # op's forward descriptor emits a *_state_recurrence step), replacing
            # the deleted _TIME_CHUNKED_RECURRENT_BACKWARD_OPS frozenset (§2.4).
            # The per-row-INDEPENDENT heavy ops are still gated on the
            # _ROW_CHUNKED_INDEPENDENT_BACKWARD_OPS set here; step 8 replaces that
            # membership with the watchdog est_gpu_time_s > budget predicate.
            if (
                # Watchdog chunking is a macOS GPU-watchdog WORKAROUND and is
                # Metal-ONLY. CUDA has no per-command-buffer watchdog, so it keeps
                # grid_chunks (monolithic, one launch per segment): the genuine
                # fused kernels. The comment above always claimed this but the
                # code applied launcher_chunks unconditionally -- the gate below
                # makes CUDA actually monolithic. (Measured: monolithic is the
                # correct CUDA path; the heavy backward COMPUTE -- not chunking
                # overhead -- is the long pole on CUDA, so this is correctness +
                # fewer launches, not a speed claim.) The hardware-aware auto-split
                # supersedes this with caps.has_command_buffer_watchdog.
                _path_c_default_target() == "metal"
                and execution_phase == DESCRIPTOR_EXECUTION_STAGE_BACKWARD
                and direct_target.max_rows_per_launch is not None
                and any(
                    _node_is_recurrent_state_scan(node)
                    or _path_c_descriptor_stage_node_op_name(node)
                    in _ROW_CHUNKED_INDEPENDENT_BACKWARD_OPS
                    for node in candidate_region.nodes
                )
            ):
                # Per-op time-chunk window. mamba3_mimo_bwd carries multi-MB
                # global reverse-scan state per step (b330bdb), so an 8-step
                # command buffer trips the watchdog on its first launch -- it
                # needs a SMALLER window than the per-row-independent heavy ops.
                # _mamba3_bwd_rows_per_kernel_launch_for_nodes returns the mamba3
                # override only when the segment contains mamba3_mimo_bwd; every
                # other launcher-chunked backward op keeps the shared default.
                direct_target = _target_with_max_rows_per_launch(
                    direct_target,
                    candidate_region,
                    _derived_max_rows_per_launch_for_region(candidate_region),
                    DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS,
                    rows_per_kernel_launch=(
                        _mamba3_bwd_rows_per_kernel_launch_for_nodes(
                            candidate_region.nodes
                        )
                    ),
                )
            # ROW-WINDOWING for the per-row-INDEPENDENT heavy FORWARD ops
            # (attention_qkv_projection / sparse_mla_fp8_apply; see
            # _ROW_CHUNKED_INDEPENDENT_FORWARD_OPS). The FORWARD analog of the
            # backward block above. At full scale the fused 3-op forward segment
            # ..._chain_4_6 (residual_rmsnorm + attention_qkv_projection +
            # sparse_mla_fp8_apply) runs as ONE monolithic grid_chunks command
            # buffer over S=4096 and is killed at ~9.6s by the macOS GPU watchdog
            # (kIOGPUCommandBufferCallbackErrorTimeout). The effective forward cap
            # above isolates each heavy op in its OWN 1-op segment; here that
            # isolated segment is switched to launcher_chunks so its per-row body
            # (`for row in T.serial(row_chunk_start, row_chunk_stop)`) is driven as
            # K command buffers over the independent ROW axis -- each command
            # buffer holds only a row window of work, well under the watchdog.
            # These ops have NO carried cross-row state
            # (_row_phased_launcher_carry_buffers_for_nodes adds carries ONLY for
            # mamba3/m2rnn, never for these), so no carry buffers are added.
            # Across the K launches every row [0, S) is written into the
            # caller-owned full-sequence buffers, so the forward activations are
            # bitwise unchanged by chunking; sparse_mla_fp8_apply reads the
            # full KV-history workspace already written by the SEPARATE isolated
            # attention_qkv_projection segment. (Metal-only watchdog split; CUDA
            # keeps the monolithic grid_chunks forward path.)
            if (
                forward_row_chunk_isolation
                and execution_phase == DESCRIPTOR_EXECUTION_STAGE_FORWARD
                and direct_target.max_rows_per_launch is not None
                and any(
                    _path_c_descriptor_stage_node_op_name(node)
                    in _ROW_CHUNKED_INDEPENDENT_FORWARD_OPS
                    for node in candidate_region.nodes
                )
            ):
                direct_target = _target_with_max_rows_per_launch(
                    direct_target,
                    candidate_region,
                    _derived_max_rows_per_launch_for_region(candidate_region),
                    DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS,
                )
            try:
                parameter_count = _kernel_parameter_count_for_target(
                    candidate_region,
                    direct_target,
                )
            except Exception as exc:
                first_failure = str(exc)
                if best is None:
                    break
                continue
            if parameter_count > max_kernel_buffers:
                first_failure = (
                    f"direct-buffer segment needs {parameter_count} kernel "
                    f"buffers, above limit {max_kernel_buffers}"
                )
                break
            # Device-grounded ESTIMATE backstops (design step 8 / §3.2-3.4): the
            # op-count caps above are the primary split mechanism (they reproduce
            # the hand-tuned splits at the calibration scale); these predicates
            # additionally fire on LARGER region shapes whose generated MSL source
            # exceeds the compiler pipeline-state ceiling (forward) or whose
            # estimated GPU time exceeds the watchdog budget even after row/time
            # chunking (backward). On a 1-op segment that still cannot fit, they
            # RAISE PathCSplitInfeasible (RULE #1) rather than emit a kernel the
            # device will crash on.
            backstop_failure = _path_c_segment_estimate_backstop(
                candidate_region=candidate_region,
                direct_target=direct_target,
                parameter_count=parameter_count,
                execution_phase=execution_phase,
                segment_op_count=end - start,
            )
            if backstop_failure is not None:
                first_failure = backstop_failure
                break
            plan = compile_path_c_region(
                candidate_region,
                schedule_template=_attested_schedule_template_for_target(
                    direct_target,
                    candidate_region,
                ),
                schedule_name=direct_target.schedule_name,
                schedule_status=direct_target.schedule_status,
            )
            if not isinstance(plan, FusionCompilePlan):
                raise TypeError("compile_path_c_region unexpectedly returned an artifact")
            best = PathCFusionScheduleChainSegment(
                index=len(segments),
                node_start=start,
                node_end=end,
                region=candidate_region,
                plan=plan,
                schedule_target=direct_target,
                kernel_parameter_count=parameter_count,
                physical_abi_policy=DESCRIPTOR_PHYSICAL_ABI_POLICY_DIRECT,
                status="ok",
                reason="direct-buffer segment fits the portable Metal buffer limit",
                execution_phase=execution_phase,
            )
        if best is not None:
            segments.append(best)
            start = best.node_end
            continue
        blocked_region = _subregion_from_nodes(
            working_region,
            start=start,
            end=start + 1,
            name=f"{working_region.name}_chain_{start}_{start + 1}",
        )
        segments.append(
            PathCFusionScheduleChainSegment(
                index=len(segments),
                node_start=start,
                node_end=start + 1,
                region=blocked_region,
                plan=None,
                schedule_target=None,
                kernel_parameter_count=None,
                physical_abi_policy=DESCRIPTOR_PHYSICAL_ABI_POLICY_DIRECT,
                status="blocked",
                reason=first_failure or "direct-buffer segment planning failed",
                execution_phase=_path_c_schedule_segment_execution_phase(
                    blocked_region.nodes
                ),
            )
        )
        start += 1
    blocked = tuple(segment for segment in segments if segment.status != "ok")
    return PathCFusionScheduleChainPlan(
        source_region=working_region,
        max_kernel_buffers=max_kernel_buffers,
        segments=tuple(segments),
        status="ready" if not blocked else "blocked",
        reason=(
            "all chain segments fit direct-buffer portable Metal limits"
            if not blocked
            else "at least one chain segment cannot be expressed as direct buffers"
        ),
    )


def plan_path_c_direct_fusion_chains_for_model(
    model: Any,
    *,
    region_prefix: str | None = None,
    include_backward: bool = True,
    min_route_bricks: int = 2,
    max_kernel_buffers: int | _ResolveFromTarget = _RESOLVE_BUFFER_LIMIT_FROM_CAPS,
    max_segment_nodes: int | None = None,
    forward_max_segment_nodes: int | None | _ResolveFromTarget = (
        _RESOLVE_FORWARD_CAP_FROM_TARGET
    ),
    backward_max_segment_nodes: int | None | _ResolveFromTarget = (
        _RESOLVE_BACKWARD_CAP_FROM_TARGET
    ),
    registry: PathCFusionScheduleRegistry | None = None,
    sequence_length: int | None = None,
) -> tuple[PathCFusionScheduleChainPlan, ...]:
    """Plan direct-buffer fused schedule chains for every supported model region.

    This is the direct-buffer sibling of
    ``plan_path_c_fusion_schedules_for_model``: discover regions from the
    model's brick graph first, then split each discovered region into generic
    direct-buffer segments.  It does not consult named acceptance fixtures.

    ``forward_max_segment_nodes`` / ``backward_max_segment_nodes`` are forwarded
    to :func:`plan_path_c_direct_fusion_chain_for_region` (Metal-only fusion caps
    for the forward / backward phases; see that function's docstring).
    """

    regions = build_path_c_model_regions_from_model(
        model,
        region_prefix=region_prefix,
        include_backward=False,
        min_route_bricks=min_route_bricks,
        sequence_length=sequence_length,
    )
    return tuple(
        plan_path_c_direct_fusion_chain_for_region(
            region,
            include_backward=include_backward,
            max_kernel_buffers=max_kernel_buffers,
            max_segment_nodes=max_segment_nodes,
            forward_max_segment_nodes=forward_max_segment_nodes,
            backward_max_segment_nodes=backward_max_segment_nodes,
            registry=registry,
        )
        for region in regions
    )


def plan_path_c_fusion_schedules_for_model(
    model: Any,
    *,
    region_prefix: str | None = None,
    include_backward: bool = True,
    min_route_bricks: int = 2,
    registry: PathCFusionScheduleRegistry | None = None,
    sequence_length: int | None = None,
) -> tuple[PathCFusionScheduleOptimizerPlan, ...]:
    """Plan Path C fused schedules for every supported region in ``model``.

    This is the production-oriented entrypoint: discover regions from the
    model's bricks first, then resolve schedule descriptors per region.  Named
    acceptance fixtures are not consulted.
    """

    regions = build_path_c_model_regions_from_model(
        model,
        region_prefix=region_prefix,
        include_backward=False,
        min_route_bricks=min_route_bricks,
        sequence_length=sequence_length,
    )
    return tuple(
        plan_path_c_fusion_schedule_for_region(
            region,
            include_backward=include_backward,
            registry=registry,
        )
        for region in regions
    )


def _subregion_from_nodes(
    region: PathCFusionRegion,
    *,
    start: int,
    end: int,
    name: str,
) -> PathCFusionRegion:
    selected_nodes = tuple(region.nodes[start:end])
    if not selected_nodes:
        raise ValueError("subregion node slice must not be empty")
    metadata = {
        **dict(region.metadata),
        "path_c_chain_source_region": region.name,
        "path_c_chain_node_start": start,
        "path_c_chain_node_end": end,
    }
    return build_path_c_fusion_region(
        region_name=name,
        surfaces=tuple(
            FusionKernelSurface.path_c(
                name=node.name,
                op_name=node.op_name,
                inputs=node.inputs,
                outputs=node.outputs,
                backward=node.backward,
                backend=node.backend,
            )
            for node in selected_nodes
        ),
        z3_sync=region.z3_sync,
        metadata=metadata,
    )


def _target_with_physical_abi_policy(
    target: PathCFusionScheduleTarget,
    region: PathCFusionRegion,
    physical_abi_policy: str,
) -> PathCFusionScheduleTarget:
    validated_policy = _validated_physical_abi_policy(physical_abi_policy)
    if target.physical_abi_policy == validated_policy:
        return target
    shape_env = _shape_env_for_region(region)
    return replace(
        target,
        schedule_template=make_path_c_descriptor_schedule_template(
            target.brick_descriptors,
            entry_symbol=getattr(region, "entry_symbol", None)
            or getattr(region, "name", None),
            buffer_extent=target.buffer_extent,
            shape_env=shape_env,
            internal_buffer_policy=target.internal_buffer_policy,
            loop_policy=target.loop_policy,
            physical_abi_policy=validated_policy,
            max_rows_per_launch=target.max_rows_per_launch,
            row_dispatch_mode=target.row_dispatch_mode,
        ),
        physical_abi_policy=validated_policy,
    )


def _target_with_max_rows_per_launch(
    target: PathCFusionScheduleTarget,
    region: PathCFusionRegion,
    max_rows_per_launch: int | None,
    row_dispatch_mode: str,
    rows_per_kernel_launch: int = DESCRIPTOR_ROWS_PER_KERNEL_LAUNCH,
) -> PathCFusionScheduleTarget:
    validated_rows = _validated_max_rows_per_launch(max_rows_per_launch)
    validated_mode = _validated_row_dispatch_mode(row_dispatch_mode)
    validated_rows_per_kernel_launch = _validated_rows_per_kernel_launch(
        rows_per_kernel_launch
    )
    # The dispatch mode + max_rows_per_launch are carried on the target struct,
    # but the per-launch time-chunk window (rows_per_kernel_launch) lives ONLY in
    # the generated schedule_template / compiled PrimFunc attrs, so it is never
    # recorded on the PathCFusionScheduleTarget. A non-default window must
    # therefore re-generate the template even when mode + max_rows already match;
    # the early-out is valid only for the default window (where re-generation
    # would be a no-op). RULE #1: an explicit non-default window always lowers --
    # never silently keeps the default 8-step kernel.
    if (
        target.max_rows_per_launch == validated_rows
        and target.row_dispatch_mode == validated_mode
        and validated_rows_per_kernel_launch == DESCRIPTOR_ROWS_PER_KERNEL_LAUNCH
    ):
        return target
    shape_env = _shape_env_for_region(region)
    return replace(
        target,
        schedule_template=make_path_c_descriptor_schedule_template(
            target.brick_descriptors,
            entry_symbol=getattr(region, "entry_symbol", None)
            or getattr(region, "name", None),
            buffer_extent=target.buffer_extent,
            shape_env=shape_env,
            internal_buffer_policy=target.internal_buffer_policy,
            loop_policy=target.loop_policy,
            physical_abi_policy=target.physical_abi_policy,
            max_rows_per_launch=validated_rows,
            row_dispatch_mode=validated_mode,
            rows_per_kernel_launch=validated_rows_per_kernel_launch,
        ),
        max_rows_per_launch=validated_rows,
        row_dispatch_mode=validated_mode,
    )


def _kernel_parameter_count_for_target(
    region: PathCFusionRegion,
    target: PathCFusionScheduleTarget,
) -> int:
    # The portable limit (DESCRIPTOR_PORTABLE_KERNEL_BUFFER_LIMIT) is Metal's
    # buffer-argument binding cap, so count only params that actually bind a
    # device buffer (present in the prim_func ``buffer_map``). Scalar params --
    # the backward gate plus the launcher time-chunk index params
    # (path_c_row_chunk_index / path_c_row_subchunk_index /
    # path_c_backward_stage_index) -- are passed by value and consume NO buffer
    # slot, so they must not count against the buffer limit. (Counting them
    # spuriously blocked the launcher-chunked mamba3 backward at 34 "buffers"
    # when it really binds only 30.)
    prim_func = target.schedule_template(region)
    params = tuple(getattr(prim_func, "params", ()))
    # Mamba3 chunked-scan delegation: the delegated grid prim is a COMPILED
    # tilelang JITKernel whose ``params`` are unhashable ``KernelParam`` dataclass
    # instances (dtype+shape, no name), NOT named TIR ``Var``. ``buffer_map`` is a
    # TIR ``Var``-keyed map that does not apply to a JITKernel, and hashing a
    # ``KernelParam`` raises ``TypeError: unhashable type: 'KernelParam'``. Count
    # the device-buffer params via the typed ``is_scalar()`` predicate instead
    # (a KernelParam with a non-empty shape binds a device buffer). RULE #1: this
    # is a real, distinct param representation, not a fallback for the TIR path.
    delegation_op = getattr(
        prim_func,
        "_cppmega_path_c_mamba3_chunked_grid_delegation",
        None,
    )
    if delegation_op is not None:
        return sum(
            1
            for param in params
            if not (
                hasattr(param, "is_scalar")
                and bool(param.is_scalar())
            )
        )
    buffer_map = getattr(prim_func, "buffer_map", {}) or {}
    return sum(1 for param in params if buffer_map.get(param) is not None)


def _attested_schedule_template_for_target(
    target: PathCFusionScheduleTarget,
    region: PathCFusionRegion,
) -> Callable[[Any], Any]:
    kwargs = {
        "implementation_kind": target.implementation_kind,
        "required_real_abi_inputs": target.required_real_abi_inputs,
    }
    if target.implementation_kind == "production":
        kwargs["production_schedule_id"] = target.schedule_id
    return mark_path_c_schedule_template_for_region(
        target.schedule_template,
        region,
        **kwargs,
    )


def path_c_fusion_schedule_spec(
    region: PathCFusionRegion | None = None,
    *,
    contract: FusionScheduleContractStatus | None = None,
    target: PathCFusionScheduleTarget | None = None,
) -> PathCFusionScheduleSpec:
    """Return the schedule contract selected from ``region``."""

    if region is None:
        raise ValueError(
            "path_c_fusion_schedule_spec requires a discovered Path C region; "
            "use mamba3_fp8_train_fusion_schedule_spec() only for the explicit "
            "Mamba3 acceptance fixture"
        )
    resolved_region = region
    resolved_contract = contract or _contract_for_region(resolved_region)
    resolved_target = target or select_path_c_fusion_schedule_target(resolved_region)
    if resolved_target is None:
        raise RuntimeError(
            f"no Path C fusion schedule target registered for op signature "
            f"{tuple(node.op_name for node in resolved_region.nodes)!r}"
        )
    missing_real_abi_inputs = _missing_real_abi_inputs(
        resolved_contract,
        resolved_target,
    )
    return PathCFusionScheduleSpec(
        schedule_id=resolved_target.schedule_id,
        schedule_name=resolved_target.schedule_name,
        region_name=resolved_region.name,
        implementation_kind=resolved_target.implementation_kind,
        implementation_status=resolved_target.schedule_status,
        missing_reason=resolved_target.missing_reason,
        trusted_by_default=(
            resolved_target.implementation_kind == "production"
            and resolved_target.schedule_id
            in trusted_path_c_production_schedule_ids()
        ),
        contract_name=resolved_contract.name,
        contract_key=resolved_contract.key,
        shape_env_key=resolved_contract.shape_env_key,
        op_signature=resolved_contract.op_signature,
        required_internal_buffers=resolved_contract.required_internal_buffers,
        required_external_buffers=resolved_contract.required_external_buffers,
        required_real_abi_inputs=resolved_target.required_real_abi_inputs,
        required_real_abi_input_shapes=_required_real_abi_input_shapes(
            resolved_region,
            resolved_target,
        ),
        missing_real_abi_inputs=missing_real_abi_inputs,
        real_abi_contract_complete=not missing_real_abi_inputs,
        required_codegen_steps=resolved_target.required_codegen_steps,
        schedule_generator=resolved_target.schedule_generator,
        schedule_generator_status=_schedule_generator_status(
            resolved_target.brick_descriptors
        ),
        internal_buffer_policy=resolved_target.internal_buffer_policy,
        loop_policy=resolved_target.loop_policy,
        buffer_extent=resolved_target.buffer_extent,
        loop_extent=_descriptor_loop_extent(
            resolved_contract.required_external_buffers,
            resolved_target.buffer_extent,
            _shape_env_for_region(resolved_region),
        ),
        brick_ops=tuple(
            descriptor.op_name for descriptor in resolved_target.brick_descriptors
        ),
        brick_schedule_families=tuple(
            descriptor.schedule_family
            for descriptor in resolved_target.brick_descriptors
        ),
        brick_descriptor_statuses=_brick_descriptor_statuses(
            resolved_target.brick_descriptors
        ),
        brick_production_fragment_statuses=(
            _brick_production_fragment_statuses(
                resolved_target.brick_descriptors
            )
        ),
        brick_production_fragment_reasons=(
            _brick_production_fragment_reasons(
                resolved_target.brick_descriptors
            )
        ),
        brick_production_fragment_blockers=(
            _brick_production_fragment_blockers(
                resolved_target.brick_descriptors
            )
        ),
        production_fragments_complete=_production_fragments_complete(
            resolved_target.brick_descriptors
        ),
    )


def mamba3_fp8_train_fusion_schedule_spec(
    region: PathCFusionRegion | None = None,
    *,
    contract: FusionScheduleContractStatus | None = None,
    target: PathCFusionScheduleTarget | None = None,
) -> Mamba3Fp8TrainFusionScheduleSpec:
    """Return the named Mamba3 acceptance schedule target for ``region``."""

    resolved_region = _require_path_c_region_graph(
        region,
        function_name="mamba3_fp8_train_fusion_schedule_spec",
    )
    resolved_target = target or mamba3_fp8_train_fusion_schedule_target()
    generic = path_c_fusion_schedule_spec(
        resolved_region,
        contract=contract,
        target=resolved_target,
    )
    return Mamba3Fp8TrainFusionScheduleSpec(**generic.__dict__)


def _missing_real_abi_inputs(
    contract: FusionScheduleContractStatus,
    target: PathCFusionScheduleTarget,
) -> tuple[str, ...]:
    external_buffers = set(contract.required_external_buffers)
    return tuple(
        name
        for name in target.required_real_abi_inputs
        if name not in external_buffers
    )


def _required_real_abi_input_shapes(
    region: PathCFusionRegion,
    target: PathCFusionScheduleTarget,
) -> tuple[str, ...]:
    shape_env = _shape_env_for_region(region)
    return tuple(
        f"{name}:{_shape_literal(_buffer_shape(name, target.buffer_extent, shape_env))}"
        for name in target.required_real_abi_inputs
    )


def plan_mamba3_fp8_train_fusion_schedule(
    *,
    include_backward: bool = True,
    model_config: Any | None = None,
    max_rows_per_launch: int | None = DESCRIPTOR_DEFAULT_MAX_ROWS_PER_LAUNCH,
    row_dispatch_mode: str = DESCRIPTOR_DEFAULT_ROW_DISPATCH_MODE,
) -> Mamba3Fp8TrainFusionSchedulePlan:
    """Build and plan the named Mamba3 FP8 train-block acceptance schedule."""

    fwd_region = _mamba3_fp8_train_acceptance_region(
        include_backward=False,
        model_config=model_config,
    )
    acceptance_registry = PathCFusionScheduleRegistry(
        acceptance_profiles=(_mamba3_fp8_train_acceptance_profile(model_config),),
    )
    optimized = plan_path_c_fusion_schedule_for_region(
        fwd_region,
        include_backward=include_backward,
        registry=acceptance_registry,
    )
    target = optimized.schedule_target
    if target is None:
        raise RuntimeError(
            f"no Path C fusion schedule target registered for op signature "
            f"{tuple(node.op_name for node in optimized.region.nodes)!r}"
        )
    target = _target_with_max_rows_per_launch(
        target,
        optimized.region,
        max_rows_per_launch,
        row_dispatch_mode,
    )
    return Mamba3Fp8TrainFusionSchedulePlan(
        region=optimized.region,
        plan=optimized.plan,
        schedule_spec=mamba3_fp8_train_fusion_schedule_spec(
            optimized.region,
            contract=optimized.plan.schedule_contract,
            target=target,
        ),
    )


def compile_mamba3_fp8_train_fusion_schedule(
    *,
    tilelang_lowerer: Callable[..., Any] | None = None,
    target_name: str = _path_c_default_target(),
    include_backward: bool = True,
    model_config: Any | None = None,
    max_rows_per_launch: int | None = DESCRIPTOR_DEFAULT_MAX_ROWS_PER_LAUNCH,
    row_dispatch_mode: str = DESCRIPTOR_DEFAULT_ROW_DISPATCH_MODE,
) -> CompiledMamba3Fp8TrainFusionSchedule:
    """Lower the named Mamba3 FP8 train-block acceptance schedule.

    This is the compiled counterpart of
    :func:`plan_mamba3_fp8_train_fusion_schedule`: it selects the named
    acceptance profile from the region graph, attests the generated descriptor
    schedule for that exact contract, and invokes the supplied TileLang lowerer.
    The schedule is still not trusted-by-default; callers must inspect the
    returned contract and external receipts before enabling it as a default.
    """

    fwd_region = _mamba3_fp8_train_acceptance_region(
        include_backward=False,
        model_config=model_config,
    )
    acceptance_registry = PathCFusionScheduleRegistry(
        acceptance_profiles=(_mamba3_fp8_train_acceptance_profile(model_config),),
    )
    optimizer = PathCFusionScheduleOptimizer(
        fwd_region.name,
        registry=acceptance_registry,
        metadata=fwd_region.metadata,
        enable_aot_autograd=include_backward,
    ).add_kernels(_surfaces_from_region(fwd_region))
    region = optimizer.build_region()
    target = optimizer.select_schedule_target(region)
    if target is None:
        raise RuntimeError(
            f"no Path C fusion schedule target registered for op signature "
            f"{tuple(node.op_name for node in region.nodes)!r}"
        )
    target = _target_with_max_rows_per_launch(
        target,
        region,
        max_rows_per_launch,
        row_dispatch_mode,
    )
    lowerer = tilelang_lowerer or tilelang_single_entry_lowerer
    schedule_template = _attested_schedule_template_for_target(target, region)
    compiled = compile_path_c_region(
        region,
        schedule_template=schedule_template,
        schedule_name=target.schedule_name,
        schedule_status=target.schedule_status,
        tilelang_lowerer=lowerer,
        target=target_name,
    )
    if not isinstance(compiled, CompiledPathCRegion):
        raise TypeError("compile_path_c_region unexpectedly returned a plan")
    return CompiledMamba3Fp8TrainFusionSchedule(
        region=region,
        compiled=compiled,
        schedule_spec=mamba3_fp8_train_fusion_schedule_spec(
            region,
            contract=compiled.plan.schedule_contract,
            target=target,
        ),
    )


def _surfaces_from_region(region: PathCFusionRegion) -> tuple[FusionKernelSurface, ...]:
    return tuple(
        FusionKernelSurface.path_c(
            name=node.name,
            op_name=node.op_name,
            inputs=node.inputs,
            outputs=node.outputs,
            backward=node.backward,
            backend=node.backend,
        )
        for node in region.nodes
    )


def _contract_for_region(
    region: PathCFusionRegion,
) -> FusionScheduleContractStatus:
    plan = compile_path_c_region(region)
    if not isinstance(plan, FusionCompilePlan):
        raise TypeError("compile_path_c_region unexpectedly returned an artifact")
    if plan.schedule_contract is None:
        raise RuntimeError("Path C fusion region did not produce a schedule contract")
    return plan.schedule_contract

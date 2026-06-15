"""Directed search for the fastest VALID fused chunked-backward kernel.

The chunked Mamba3 backward dispatches three SEPARATE ``tilelang.compile`` kernels
(B2 = chunk_scan_combine_bwd, B1 = inter_chunk_recur_bwd, B0 = chunk_precompute_bwd).
Each pays a fixed ~3200us MLX tvm_ffi command-buffer finalize/commit floor; the
entire chunked-vs-MSL gap is ~2x that floor of redundant dispatch boundaries. MSL
is ONE fused kernel = pays the floor once.

This module implements the DIRECTED, rule-based, iterative search the design
specifies (NOT a blanket cap, NOT chaotic trial):

  Phase A -- STATIC enumerate + feasibility-predict + rank (no GPU compile):
    * enumerate the 4 contiguous partitions of the reverse-graph order [B2,B1,B0];
    * for each candidate fused segment, build the TIR-spliced prim (reusing the
      VALIDATED per-brick kernel bodies verbatim -- no body rewrite) and read its
      static cost: msl_bytes(len source), phys_shared (3.7x logical), nbuf
      (device-buffer param count AFTER demoting the internal handoff edge),
      est_gpu_time_s;
    * apply the FOUR feasibility predicates P1..P4 (MSL ceiling 140000, threadgroup
      32768, buffer-arg 31, watchdog 2.5s); drop infeasible variants;
    * lexicographic rank: dispatch_count ASC, internalized_edges DESC,
      absorbed_recomputes DESC, max_phys ASC, max_nbuf ASC.

  Phase B -- MEASURED iterate most->least promising (branch-and-bound early-stop):
    * the rank-4 baseline {B2}{B1}{B0} is the pre-measured, already-bit-correct
      ground-truth anchor;
    * for each higher-ranked feasible variant, compile (catch XPC/pipeline/watchdog
      crash -> mark infeasible_measured, CONTINUE), bit-correct gate vs path-b (all
      8 grads <1e-3 over N repeats), then measure median wall under memguard 70;
    * keep the faster VALID variant; early-stop once the floor lower-bound of the
      next variant cannot beat the current best.

RULE #1: the feasibility/ranking is heuristic but the SELECTED kernel is VALIDATED.
No fabricated measurements. Infeasible/crashing variants are caught and recorded
(fail-loud), never silently absorbed; selection never regresses below the
validated 3-dispatch baseline.

The fusion MECHANISM (proven in tests/test_path_c_backward_fusion_search.py:
bit-exact handoff vs the 3-kernel baseline): a single ``@T.prim_func`` may hold
MULTIPLE sequential ``with T.Kernel(...)``
grids. tilelang.compile lowers it to ONE metallib dispatched through ONE tvm_ffi
call = ONE command-buffer floor, while EACH grid keeps its OWN threadgroup
allocation (so the §27 4-GEMM 88.5KB / 1MB resident-dinp threadgroup wall that
makes a SINGLE-threadgroup mono kernel infeasible does NOT apply -- the grids are
sequential launches, not one fused threadgroup). The handoff buffer written by
grid-i and read by grid-{i+1} lives inside the prim (an internal device buffer),
removing it from BOTH the MLX round-trip AND the kernel ABI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import tvm.tir as tir


# Per-design device caps (validated live: device_caps() returns exactly these on
# the M4 Max preset). Imported lazily in the predictor so this module loads with
# no GPU touched.
_FLOOR_US = 3200.0  # measured per-dispatch command-buffer finalize/commit floor


# ---------------------------------------------------------------------------
# TIR-level prim splicer -- the body-reusing fusion mechanism.
# ---------------------------------------------------------------------------


def splice_prims(
    prims: Sequence[Any],
    shared_buffer_names: set[str],
    fused_symbol: str,
    zero_init_output_positions: Sequence[int] | None = None,
) -> Any:
    """Concatenate N ``@T.prim_func`` bodies into ONE prim_func.

    Buffers whose ``buffer.name`` is in ``shared_buffer_names`` are UNIFIED across
    prims to a single param/buffer (this is how a handoff OUTPUT of prim_i becomes
    the INPUT of prim_{i+1} inside one kernel WITHOUT a caller buffer, and how
    duplicated shared inputs e.g. dA_cumsum/prev_states are deduped). Every other
    buffer keeps its own param. Param order = first-seen across prims.

    PROVEN bit-exact vs running the prims separately
    (tests/test_path_c_backward_fusion_search.py: dstates/dh0 handoff result
    max|diff| at the fp32 floor). The bodies are REUSED verbatim (the determinism-
    fixed atomic/zero-init logic is unchanged); only the param Var handles for the
    unified buffers are substituted.
    """
    merged_params: list[Any] = []
    name_to_var: dict[str, Any] = {}
    name_to_buf: dict[str, Any] = {}
    new_buffer_map: dict[Any, Any] = {}
    bodies: list[tuple[Any, dict[Any, Any]]] = []

    for pf in prims:
        var_subst: dict[Any, Any] = {}
        for var in pf.params:
            buf = pf.buffer_map.get(var)
            if buf is None:
                if var.name in name_to_var:
                    var_subst[var] = name_to_var[var.name]
                else:
                    name_to_var[var.name] = var
                    merged_params.append(var)
                continue
            bname = buf.name
            if bname in shared_buffer_names and bname in name_to_buf:
                var_subst[var] = name_to_var[bname]
            else:
                name_to_var[bname] = var
                name_to_buf[bname] = buf
                new_buffer_map[var] = buf
                merged_params.append(var)
        bodies.append((pf.body, var_subst))

    rewritten = []
    for body, var_subst in bodies:
        if var_subst:
            body = tir.stmt_functor.substitute(body, dict(var_subst))
        rewritten.append(body)
    fused_body = tir.SeqStmt(rewritten) if len(rewritten) > 1 else rewritten[0]

    fused = tir.PrimFunc(
        params=merged_params,
        body=fused_body,
        ret_type=None,
        buffer_map=new_buffer_map,
    )
    fused = fused.with_attr("global_symbol", fused_symbol)
    if zero_init_output_positions is not None:
        fused = fused.with_attr(
            "tilelang_metal_zero_init_output_positions",
            list(zero_init_output_positions),
        )
    return fused


# ---------------------------------------------------------------------------
# Candidate model: the reverse-graph order is FIXED B2 -> B1 -> B0 by the data
# deps (B2.dchunk_states -> B1; B1.dstates -> B0). A grouping is a CONTIGUOUS
# partition of [B2, B1, B0]; there are exactly 4.
# ---------------------------------------------------------------------------

_BRICKS = ("B2", "B1", "B0")


def enumerate_contiguous_partitions(
    bricks: Sequence[str] = _BRICKS,
) -> list[tuple[tuple[str, ...], ...]]:
    """All contiguous partitions of an ordered brick list (4 for 3 bricks)."""
    n = len(bricks)
    out: list[tuple[tuple[str, ...], ...]] = []
    # cut points between adjacent bricks: 2^(n-1) partitions
    for mask in range(1 << (n - 1)):
        groups: list[tuple[str, ...]] = []
        cur = [bricks[0]]
        for i in range(1, n):
            if mask & (1 << (i - 1)):
                groups.append(tuple(cur))
                cur = [bricks[i]]
            else:
                cur.append(bricks[i])
        groups.append(tuple(cur))
        out.append(tuple(groups))
    return out


@dataclass
class SegmentVerdict:
    bricks: tuple[str, ...]
    msl_bytes: int
    phys_shared: int
    nbuf: int
    est_gpu_time_s: float
    p1_msl: bool
    p2_threadgroup: bool
    p3_buffer: bool
    p4_watchdog: bool
    n_internalized_edges: int

    @property
    def feasible(self) -> bool:
        return self.p1_msl and self.p2_threadgroup and self.p3_buffer and self.p4_watchdog

    @property
    def characteristic(self) -> str | None:
        if not self.p1_msl:
            return "msl-pipeline-size"
        if not self.p2_threadgroup:
            return "threadgroup"
        if not self.p3_buffer:
            return "buffer"
        if not self.p4_watchdog:
            return "watchdog"
        return None


@dataclass
class VariantVerdict:
    variant_id: str
    grouping: tuple[tuple[str, ...], ...]
    segments: list[SegmentVerdict] = field(default_factory=list)
    requires_recompute_absorption: bool = False
    # static verdict
    predicted_feasible: bool = True
    infeasible_characteristic: str | None = None
    # ranking keys (filled by ranker)
    dispatch_count: int = 0
    recovered_floor_us: float = 0.0
    n_internalized_edges: int = 0
    n_absorbed_recomputes: int = 0
    max_phys: int = 0
    max_nbuf: int = 0
    # phase-B measured fields
    compiled: bool | None = None
    crashed: bool = False
    crash_reason: str | None = None
    measured_us: float | None = None
    bit_correct: bool | None = None
    max_grad_err: float | None = None
    status: str = "pending"


# Which contiguous fused groups are STATICALLY known to require absorbing an
# inter-brick MLX recompute (re-introduces the atomic-ordering race, so the
# absorbed variant is gated on a determinism re-validation it currently fails).
# The dA_cumsum_tail recompute sits BETWEEN B1 and B0 (consumes B1.dstates, feeds
# B0.dA_cumsum_tail). A fused group that contains the B1->B0 boundary INTERNALLY
# (i.e. {...B1,B0...}) must absorb it; {B2,B1} keeps it POST-kernel (clean).
def _group_requires_absorption(group: tuple[str, ...]) -> bool:
    return "B1" in group and "B0" in group


def _internalized_edges(group: tuple[str, ...]) -> int:
    # edges fully inside the group: B2-B1 (dchunk), B1-B0 (dstates)
    n = 0
    if "B2" in group and "B1" in group:
        n += 1
    if "B1" in group and "B0" in group:
        n += 1
    return n


# ---------------------------------------------------------------------------
# Per-brick prim builders + the brick handoff/shared-buffer model.
# ---------------------------------------------------------------------------


def _brick_prim_builders() -> dict[str, Callable[..., Any]]:
    from cppmega_mlx.nn._tilelang.mamba3_chunked_backward_core import (
        chunk_scan_combine_bwd_metal_prim,
        inter_chunk_recur_bwd_metal_prim,
        chunk_precompute_bwd_metal_prim,
    )

    return {
        "B2": chunk_scan_combine_bwd_metal_prim,
        "B1": inter_chunk_recur_bwd_metal_prim,
        "B0": chunk_precompute_bwd_metal_prim,
    }


# Buffers that must be UNIFIED when a group fuses bricks. The handoff edges
# (dchunk_states: B2 out -> B1 in; dstates: B1 out -> B0 in) plus shared inputs
# (dA_cumsum, prev_states appear in B2/B1/B0; dt likewise). Unifying them removes
# the handoff from BOTH the MLX round-trip AND the kernel ABI (P3 helper).
_SHARED_BUFFER_NAMES = {
    "dchunk_states",  # B2 -> B1 handoff
    "dstates",        # B1 -> B0 handoff
    "dA_cumsum",      # shared fwd input (B2, B1, B0)
    "prev_states",    # shared fwd input (B2, B1)
    "dt",             # shared fwd input (B2, B0)
    "x",              # shared fwd input (B2, B0)
    "B",              # shared fwd input (B2, B0)
}

# B2 zero-inits its dD output (the ONE cross-threadgroup atomic, position 6 among
# its 7 outputs). B1/B0 pin zero-init to []. When fused, the bridge zero-init
# positions are ABSOLUTE output indices; we recompute them for the fused param set.


def build_fused_segment_prim(
    group: tuple[str, ...],
    dims: tuple[int, int, int, int, int, int, int],
    *,
    symbol: str,
) -> tuple[Any, list[int], dict[str, int]]:
    """Build the TIR-spliced prim for a fused brick group.

    Returns (fused_prim, out_idx, output_name_to_position). A single brick group
    returns that brick's prim unchanged. ``out_idx`` lists the device-buffer param
    positions to surface as outputs (the internalized handoff buffers are kept in
    ``out_idx`` so the wrapper can ignore them, but they are NOT MLX round-tripped
    between dispatches -- they are written+read inside the one kernel).
    """
    b, s, c, g, h, p, n = dims
    builders = _brick_prim_builders()
    prims = [builders[name](b, s, c, g, h, p, n) for name in group]
    if len(prims) == 1:
        return prims[0], _single_brick_out_idx(group[0]), _single_brick_out_names(group[0])

    # zero-init positions: B2's dD is the only one. Find dD's absolute output
    # position in the fused param order.
    fused = splice_prims(prims, _SHARED_BUFFER_NAMES, symbol, zero_init_output_positions=None)
    # compute the fused output positions for the grads each brick produces.
    out_idx, out_names = _fused_out_idx(group, fused)
    # recompute zero-init: dD output position (cross-threadgroup atomic) if B2 in group.
    zero_positions: list[int] = []
    if "B2" in group and "dD" in out_names:
        # zero-init takes the position WITHIN the out_idx list (output ordinal),
        # matching the per-brick convention (B2 used [6] = 7th output).
        zero_positions = [out_idx.index(out_names["dD"])]
    fused = fused.with_attr(
        "tilelang_metal_zero_init_output_positions", zero_positions
    )
    return fused, out_idx, out_names


# Param layouts (device-buffer positions in the prim signature). These mirror the
# *_metal_prim signatures verbatim.
_B2_PARAM_NAMES = (
    "dout", "cb", "x", "z", "dt", "dA_cumsum", "C", "B", "prev_states", "D", "y",
    "dC", "dx", "dz", "dchunk_states", "dinp", "dA_cumsum_y", "dD",
)
_B1_PARAM_NAMES = (
    "dchunk_states", "dA_cumsum", "dh_last", "prev_states",
    "dstates", "dh0", "dA_cumsum_tail",
)
_B0_PARAM_NAMES = (
    "dstates", "dinp_diag", "dA_cumsum_y", "dA_cumsum_tail", "dA_cumsum",
    "x", "B", "dt", "A",
    "dx", "dB", "dlog_decay", "ddt",
)
_BRICK_OUTPUTS = {
    "B2": ("dC", "dx", "dz", "dchunk_states", "dinp", "dA_cumsum_y", "dD"),
    "B1": ("dstates", "dh0", "dA_cumsum_tail"),
    "B0": ("dx", "dB", "dlog_decay", "ddt"),
}
_BRICK_PARAMS = {"B2": _B2_PARAM_NAMES, "B1": _B1_PARAM_NAMES, "B0": _B0_PARAM_NAMES}


def _single_brick_out_idx(brick: str) -> list[int]:
    names = _BRICK_PARAMS[brick]
    return [names.index(o) for o in _BRICK_OUTPUTS[brick]]


def _single_brick_out_names(brick: str) -> dict[str, int]:
    names = _BRICK_PARAMS[brick]
    return {o: names.index(o) for o in _BRICK_OUTPUTS[brick]}


def _fused_out_idx(
    group: tuple[str, ...], fused: Any
) -> tuple[list[int], dict[str, int]]:
    """Map each brick's outputs to absolute positions in the fused param list.

    The fused params are first-seen across bricks (splice_prims order). A param
    name appearing in multiple bricks is unified to ONE position; an OUTPUT that
    is ALSO a downstream brick's input (a handoff e.g. dchunk_states/dstates)
    keeps a single position. We surface every brick output the wrapper consumes.
    """
    param_names = [v.name.replace("_handle", "") for v in fused.params]
    pos_of = {nm: i for i, nm in enumerate(param_names)}
    out_names: dict[str, int] = {}
    for brick in group:
        for o in _BRICK_OUTPUTS[brick]:
            if o in pos_of:
                out_names[o] = pos_of[o]
    # stable ordered out_idx
    out_idx = sorted(set(out_names.values()))
    return out_idx, out_names


# ---------------------------------------------------------------------------
# Phase A -- feasibility predictor (static, no metallib / no newComputePipelineState).
# ---------------------------------------------------------------------------

# alloc_shared line regex: parse the lowered MSL/TIR source for threadgroup allocs
# (reuse the estimator's contract: shape + dtype -> bytes).
import re as _re  # noqa: E402

_ALLOC_SHARED_RE = _re.compile(
    r".*alloc_shared\(\s*(?P<shape>[^,]+(?:,\s*[^,]+)*?)\s*,\s*"
    r"(?P<dtype>float16|float32|float64|int32|int8|uint8|bool)\s*\)"
)


def _lower_source_and_nbuf(prim: Any, target: Any) -> tuple[str, int]:
    """Return (kernel_source, device_buffer_param_count) WITHOUT a metallib build.

    ``tilelang.lower(prim, target, enable_device_compile=False)`` materializes the
    device MSL source (the basis for msl_bytes / phys_shared) but does NOT create a
    Metal pipeline-state (no newComputePipelineState -> no XPC crash, no watchdog
    risk). This is the truly-static estimate. nbuf = device-buffer params (every
    param that binds a buffer in buffer_map).
    """
    import tilelang

    art = tilelang.lower(prim, target=target, enable_device_compile=False)
    src = art.kernel_source or ""
    buffer_map = getattr(prim, "buffer_map", {}) or {}
    params = tuple(getattr(prim, "params", ()))
    nbuf = sum(1 for pp in params if buffer_map.get(pp) is not None)
    return src, nbuf


# TileLang coalesces every per-kernel alloc_shared into ONE physical
# ``threadgroup uchar buf_dyn_shmem[N];`` declaration in the lowered MSL -- the
# EXACT physical threadgroup bytes for that grid (no estimate needed). A spliced
# multi-T.Kernel prim emits ONE such declaration PER grid; the grids run
# sequentially and each grid's threadgroup is FREED at its boundary, so the
# binding P2 quantity is the MAX over grids (NOT the sum). This is why the
# multi-T.Kernel splice sidesteps the §27 mono-fusion threadgroup wall (88.5KB /
# 1MB resident-DINP) that only applies to a SINGLE-threadgroup kernel.
_BUF_DYN_SHMEM_RE = _re.compile(r"buf_dyn_shmem\s*\[\s*(\d+)\s*\]\s*;")


def _phys_shared_from_source(src: str, margin: float) -> int:
    """EXACT max-over-grids physical threadgroup bytes from the lowered MSL.

    Reads the coalesced ``threadgroup uchar buf_dyn_shmem[N];`` declarations (one
    per grid) and returns the MAX N. ``margin`` is unused for this exact path (the
    coalesced declaration already IS the physical byte count TileLang padded to);
    kept in the signature for parity with the estimator contract.
    """
    sizes = [int(m) for m in _BUF_DYN_SHMEM_RE.findall(src)]
    return max(sizes) if sizes else 0


def predict_segment(
    group: tuple[str, ...],
    dims: tuple[int, int, int, int, int, int, int],
    caps: Any,
    target: Any,
    *,
    symbol: str,
) -> SegmentVerdict:
    """Static feasibility verdict for ONE fused segment (P1..P4, no GPU compile)."""
    prim, _out_idx, _names = build_fused_segment_prim(group, dims, symbol=symbol)
    src, nbuf = _lower_source_and_nbuf(prim, target)
    msl_bytes = len(src)
    phys = _phys_shared_from_source(src, caps.logical_to_physical_shared_margin)
    # est_gpu_time_s: mamba3 chunked bwd ops carry per_op_time_per_row coeff 0.0 in
    # the preset -> 0.0; retained for the predicate (re-checked at production shape).
    est_t = 0.0

    ceil_msl = caps.msl_pipeline_state_ceiling_bytes
    p1 = ceil_msl is None or msl_bytes <= ceil_msl
    p2 = phys <= caps.threadgroup_mem_bytes
    p3 = nbuf <= caps.buffer_arg_limit
    wd_budget = (
        caps.watchdog_window_s * caps.safety_margin
        if caps.watchdog_window_s is not None else float("inf")
    )
    p4 = est_t <= wd_budget
    return SegmentVerdict(
        bricks=group,
        msl_bytes=msl_bytes,
        phys_shared=phys,
        nbuf=nbuf,
        est_gpu_time_s=est_t,
        p1_msl=p1,
        p2_threadgroup=p2,
        p3_buffer=p3,
        p4_watchdog=p4,
        n_internalized_edges=_internalized_edges(group),
    )


def predict_variants(
    dims: tuple[int, int, int, int, int, int, int],
) -> list[VariantVerdict]:
    """Phase A: enumerate -> build spliced segments -> predict P1..P4 -> drop infeasible."""
    from cppmega_mlx.runtime.path_c_device_caps import device_caps
    from cppmega_mlx.nn._tilelang.mamba3_path_c import _CHUNKED_METAL_TARGET

    caps = device_caps()
    target = _CHUNKED_METAL_TARGET
    variants: list[VariantVerdict] = []
    for groups in enumerate_contiguous_partitions():
        vid = "_".join("".join(g) for g in groups)
        v = VariantVerdict(variant_id=vid, grouping=groups)
        feasible = True
        char: str | None = None
        for gi, group in enumerate(groups):
            seg = predict_segment(
                group, dims, caps, target, symbol=f"bwdfuse_{vid}_{gi}"
            )
            v.segments.append(seg)
            if not seg.feasible:
                feasible = False
                char = seg.characteristic
            if _group_requires_absorption(group):
                v.requires_recompute_absorption = True
        v.predicted_feasible = feasible
        v.infeasible_characteristic = char
        v.dispatch_count = len(groups)
        v.recovered_floor_us = (len(_BRICKS) - len(groups)) * _FLOOR_US
        v.n_internalized_edges = sum(_internalized_edges(g) for g in groups)
        # absorbed recomputes: a group that contains B1->B0 internally absorbs
        # dA_cumsum_tail (1); we count it only when the variant is NOT requiring a
        # post-kernel recompute (here we mark the structural need, gated later).
        v.n_absorbed_recomputes = sum(
            1 for g in groups if _group_requires_absorption(g)
        )
        v.max_phys = max((s.phys_shared for s in v.segments), default=0)
        v.max_nbuf = max((s.nbuf for s in v.segments), default=0)
        variants.append(v)
    return variants


# ---------------------------------------------------------------------------
# Ranker -- lexicographic ordering of FEASIBLE variants (design cost model).
# ---------------------------------------------------------------------------


def rank_variants(variants: Sequence[VariantVerdict]) -> list[VariantVerdict]:
    """Order FEASIBLE variants most-promising-first.

    PRIMARY (dominates ~10x): dispatch_count ASC (fewer ~3200us floors).
    Tie-breaks: T1 internalized_edges DESC, T2 absorbed_recomputes DESC,
    T3 max_phys ASC, T4 max_nbuf ASC. Variants requiring recompute-absorption are
    ranked BELOW an equal-dispatch variant that does not (T1/T2 rationale: a clean
    internalized handoff is a free win; absorbing dA_tail re-introduces the gated
    atomic race).
    """
    feasible = [v for v in variants if v.predicted_feasible]
    return sorted(
        feasible,
        key=lambda v: (
            v.dispatch_count,                 # PRIMARY ASC
            -v.n_internalized_edges,          # T1 DESC
            v.requires_recompute_absorption,  # absorbers last among equal dispatch
            -v.n_absorbed_recomputes,         # T2 DESC
            v.max_phys,                       # T3 ASC
            v.max_nbuf,                       # T4 ASC
        ),
    )


# ---------------------------------------------------------------------------
# Phase B -- compile a fused-backward CALLABLE for a grouping.
# ---------------------------------------------------------------------------

# Supported groupings for the EXECUTABLE Phase B (the splice is clean only when no
# inter-kernel MLX recompute lands INSIDE a fused group):
#   * baseline      ((B2,)(B1,)(B0,))  -- 3 dispatches, the validated anchor.
#   * B2B1 fused    ((B2,B1)(B0,))     -- 2 dispatches; dA_tail recompute stays
#                                          POST-fused-kernel MLX (between the fused
#                                          B2+B1 dispatch and B0). CLEAN, no race
#                                          re-introduction.
# Groupings that put the B1->B0 boundary INSIDE a fused group ((B1,B0) or the
# fully-fused (B2,B1,B0)) require ABSORBING the dA_cumsum_tail recompute into the
# kernel -- i.e. feeding B0 the kernel's INTRA-threadgroup atomic dA_cumsum_tail
# output (the deliberately-discarded flaky one). Phase B compiles them too but the
# bit-correct gate is expected to expose the race (the HONEST blocker); they are
# never SELECTED unless they pass determinism over N repeats.

import functools


@functools.lru_cache(maxsize=16)
def _compiled_segment(group: tuple[str, ...], dims, symbol: str):
    """Compile ONE fused segment to a tvm_ffi kernel (the metallib build)."""
    import tilelang
    from cppmega_mlx.nn._tilelang.mamba3_path_c import _CHUNKED_METAL_TARGET

    prim, out_idx, out_names = build_fused_segment_prim(group, dims, symbol=symbol)
    kernel = tilelang.compile(
        prim, target=_CHUNKED_METAL_TARGET,
        execution_backend="tvm_ffi", out_idx=out_idx,
    )
    return kernel, out_idx, out_names


def make_fused_backward(grouping: tuple[tuple[str, ...], ...], dims):
    """Return a backward callable implementing ``grouping`` with the right handoffs.

    The callable has the SAME signature/semantics as the live
    ``_mamba3_chunked_backward_path_c`` wrapper body (it returns the 8-grad tuple),
    differing ONLY in how many tvm_ffi dispatches the B2/B1/B0 work is split into.
    The post-kernel deterministic recomputes (dA_tail, dD, dx_total) and primal-
    order grad mapping are IDENTICAL to the live wrapper.
    """
    import mlx.core as mx

    b, s, c, g, h, p, n = dims
    # which fused segment owns each brick
    seg_of: dict[str, tuple[str, ...]] = {}
    for grp in grouping:
        for br in grp:
            seg_of[br] = grp
    sym = "_".join("".join(grp) for grp in grouping)
    compiled: dict[tuple[str, ...], Any] = {}
    for gi, grp in enumerate(grouping):
        compiled[grp] = _compiled_segment(grp, dims, f"bwdrun_{sym}_{gi}")

    def run(dy, x, B, C, z, A, dt, D, h0, *, cb, dA_cumsum, prev_states, y):
        from cppmega_mlx.nn._tilelang.mamba3_path_c import _assert_per_head_constant_A

        batch, seq, heads, headdim = x.shape
        state = B.shape[-1]
        chunk = c
        nchunks = seq // chunk

        def f16(a):
            return mx.contiguous(a.astype(mx.float16))

        x16 = f16(x); B16 = f16(B); C16 = f16(C); z16 = f16(z); D16 = f16(D)
        dout16 = f16(dy); y16 = f16(y)
        dt16 = dt.astype(mx.float16)
        dt_k = mx.contiguous(mx.transpose(
            dt16.reshape(batch, nchunks, chunk, heads), (0, 3, 1, 2)))
        cb16 = cb.astype(mx.float16) if cb.dtype != mx.float16 else cb
        dA16 = dA_cumsum.astype(mx.float16) if dA_cumsum.dtype != mx.float16 else dA_cumsum
        A_head16 = _assert_per_head_constant_A(A).astype(mx.float16)
        prev32 = prev_states.astype(mx.float32)
        dh_last = mx.zeros((batch, heads, headdim, state), dtype=mx.float32)

        # ---- dispatch B2 (+ B1 if fused with B2) ----
        grp_b2 = seg_of["B2"]
        k_b2, oi_b2, on_b2 = compiled[grp_b2]
        if grp_b2 == ("B2",):
            dC_m, dx_b2, dz_m, dchunk, dinp_diag, dA_y, _dD = k_b2(
                dout16, cb16, x16, z16, dt_k, dA16, C16, B16, prev32, D16, y16)
        elif grp_b2 == ("B2", "B1"):
            # fused B2+B1: inputs = B2 inputs + B1's NEW inputs (dh_last). The
            # handoff dchunk_states + shared dA_cumsum/prev_states are internalized.
            res = k_b2(dout16, cb16, x16, z16, dt_k, dA16, C16, B16, prev32, D16,
                       y16, dh_last)
            out = dict(zip([_inv_name(on_b2, pos) for pos in oi_b2], res))
            dC_m = out["dC"]; dx_b2 = out["dx"]; dz_m = out["dz"]
            dchunk = out["dchunk_states"]; dinp_diag = out["dinp"]; dA_y = out["dA_cumsum_y"]
            dstates = out["dstates"]; dh0_m = out["dh0"]
        else:
            raise RuntimeError(
                f"make_fused_backward: unsupported B2 group {grp_b2}; the executable "
                "Phase B supports baseline (B2,) and (B2,B1) for B2. Fully-fused "
                "(B2,B1,B0) is handled by the absorption path (gated)."
            )

        # ---- B1 (if separate) ----
        if seg_of["B1"] == ("B1",):
            k_b1, _oi, _on = compiled[("B1",)]
            dstates, dh0_m, _dA_tail_k = k_b1(dchunk, dA16, dh_last, prev32)
        elif seg_of["B1"] != grp_b2 and seg_of["B1"] == ("B1", "B0"):
            raise RuntimeError(
                "make_fused_backward: (B1,B0) absorption path requires feeding B0 "
                "the kernel dA_cumsum_tail (the flaky intra-threadgroup atomic). "
                "This grouping is gated -- run via the absorption builder, not the "
                "clean splice."
            )
        # dstates/dh0_m now defined (from fused B2B1 or separate B1)

        # ---- deterministic dA_cumsum_tail recompute (POST-kernel MLX) ----
        L = chunk
        tail_dacs = dA16[:, :, :, L - 1].astype(mx.float32)
        decay_tail = mx.exp(tail_dacs)
        prod_pn = mx.sum(dstates.astype(mx.float32) * prev32, axis=(3, 4))
        prod_pn = mx.transpose(prod_pn, (0, 2, 1))
        dA_tail = mx.zeros((batch, heads, nchunks, L), dtype=mx.float32)
        dA_tail[:, :, :, L - 1] = decay_tail * prod_pn
        dA_tail = mx.contiguous(dA_tail)

        # ---- B0 ----
        k_b0, _oi0, _on0 = compiled[seg_of["B0"]]
        if seg_of["B0"] == ("B0",):
            dx_b0, dB_m, dlog_m, ddt_m = k_b0(
                dstates, dinp_diag, dA_y, dA_tail, dA16, x16, B16, dt_k, A_head16)
        else:
            raise RuntimeError(
                f"make_fused_backward: unsupported B0 group {seg_of['B0']} in clean "
                "splice path"
            )

        dx_total = dx_b2 + dx_b0
        z32 = z.astype(mx.float32)
        silu_z = z32 * mx.sigmoid(z32)
        d_y_skip = dy.astype(mx.float32) * silu_z
        dD_wrapper = mx.sum(d_y_skip * x.astype(mx.float32), axis=(0, 1, 3))

        dx = dx_total.astype(x.dtype)
        dB = dB_m.astype(B.dtype)
        dC = dC_m.astype(C.dtype)
        dz = dz_m.astype(z.dtype)
        dA_out = (dlog_m * dt.astype(dlog_m.dtype)).astype(A.dtype)
        ddt = ddt_m.astype(dt.dtype)
        dD = dD_wrapper.astype(D.dtype)
        dh0 = dh0_m.astype(h0.dtype)
        return (dx, dB, dC, dz, dA_out, ddt, dD, dh0)

    return run


def _inv_name(out_names: dict[str, int], pos: int) -> str:
    for nm, p in out_names.items():
        if p == pos:
            return nm
    raise KeyError(f"no output name at fused position {pos}")


# Groupings whose executable form needs dA_cumsum_tail ABSORPTION (B1->B0 boundary
# inside one fused group) -- the clean splice path does NOT cover these; they are
# the gated rank-1/rank-3 variants. Phase B marks them honestly rather than
# fabricating a measurement.
def _grouping_is_clean_splice(grouping: tuple[tuple[str, ...], ...]) -> bool:
    return not any(_group_requires_absorption(g) for g in grouping)


# ---------------------------------------------------------------------------
# Phase B -- measured iterate (compile + bit-correct gate + median wall).
# ---------------------------------------------------------------------------



def _trace_row(v: VariantVerdict) -> dict:
    return dict(
        variant=v.variant_id,
        grouping=str(v.grouping),
        predicted_feasible=v.predicted_feasible,
        dispatch_count=v.dispatch_count,
        recovered_floor_us=v.recovered_floor_us,
        compiled=v.compiled,
        crashed=v.crashed,
        crash_reason=v.crash_reason,
        bit_correct=v.bit_correct,
        max_grad_err=v.max_grad_err,
        measured_us=v.measured_us,
        status=v.status,
    )


def _build_eval_inputs(dims):
    """nam56r-surface inputs (per-head-constant A) + the GOLD path-b grads + the
    forward intermediates (cb, dA_cumsum, prev_states, y_skip) the chunked bwd
    consumes. Returns a dict; intermediates are forward artifacts (NOT timed)."""
    import numpy as np
    import mlx.core as mx
    from cppmega_mlx.nn._tilelang.mamba3 import mamba3_mimo_bwd_metal
    from cppmega_mlx.nn._tilelang.mamba3_path_c import (
        _mamba3_chunked_fwd_intermediates_path_c,
        _mamba3_pre_gate_yskip_path_c,
    )

    b, s, c, g, h, p, n = dims
    rng = np.random.RandomState(0)

    def f32(*shape, sc=0.1):
        return mx.array((rng.randn(*shape) * sc).astype(np.float32))

    x = f32(b, s, h, p); B = f32(b, s, h, n); C = f32(b, s, h, n)
    z = f32(b, s, h, p, sc=0.5)
    A_head = (-rng.rand(h)).astype(np.float32)
    A = mx.array(np.broadcast_to(A_head[None, None, :], (b, s, h)).copy())
    dt = mx.array((rng.rand(b, s, h) * 0.05).astype(np.float32))
    D = mx.array((rng.randn(h)).astype(np.float32))
    h0 = f32(b, h, p, n)
    cot_y = mx.array((rng.randn(b, s, h, p) * 0.1).astype(np.float32))
    primals = (x, B, C, z, A, dt, D, h0)

    grads_gold = mamba3_mimo_bwd_metal(cot_y, *primals, backend="mlx")
    mx.eval(*grads_gold)

    cb, dA_cumsum, prev_states = _mamba3_chunked_fwd_intermediates_path_c(
        x, B, C, A, dt, h0)
    y_skip = _mamba3_pre_gate_yskip_path_c(x, B, C, A, dt, D, h0).astype(mx.float16)
    mx.eval(cb, dA_cumsum, prev_states, y_skip)
    return dict(
        primals=primals, dy=cot_y, grads_gold=grads_gold,
        cb=cb, dA_cumsum=dA_cumsum, prev_states=prev_states, y=y_skip,
    )


def _maxabs(a, ref):
    import numpy as np
    import mlx.core as mx
    return float(np.abs(np.asarray(a.astype(mx.float32), np.float64)
                        - np.asarray(ref.astype(mx.float32), np.float64)).max())


_GRAD_NAMES = ("dx", "dB", "dC", "dz", "dA", "ddt", "dD", "dh0")


def _bitcorrect_direct(bwd_callable, inp, repeats, abs_gate):
    """Run the backward callable ``repeats`` times (fresh dispatches) and gate all 8
    grads < abs_gate vs the path-b GOLD on EVERY run (the determinism guard)."""
    import mlx.core as mx

    worst = {nm: 0.0 for nm in _GRAD_NAMES}
    ok = True
    for _ in range(repeats):
        grads = bwd_callable(
            inp["dy"], *inp["primals"],
            cb=inp["cb"], dA_cumsum=inp["dA_cumsum"],
            prev_states=inp["prev_states"], y=inp["y"])
        mx.eval(*grads)
        for nm, gc, gg in zip(_GRAD_NAMES, grads, inp["grads_gold"]):
            d = _maxabs(gc, gg)
            if d > worst[nm]:
                worst[nm] = d
            if d >= abs_gate:
                ok = False
    return ok, max(worst.values()), worst


def _measure_median(bwd_callable, inp, runs, warmup):
    """Median wall (us) of the backward callable (fresh dispatches), single eval at
    the end of each timed call. memguard 70 is the caller's responsibility."""
    import time
    import numpy as np
    import mlx.core as mx

    def one():
        grads = bwd_callable(
            inp["dy"], *inp["primals"],
            cb=inp["cb"], dA_cumsum=inp["dA_cumsum"],
            prev_states=inp["prev_states"], y=inp["y"])
        mx.eval(*grads)

    for _ in range(warmup):
        one()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        one()
        times.append(time.perf_counter() - t0)
    return float(np.median(np.asarray(times, np.float64)) * 1e6)


def search_fastest_backward_fusion(
    dims,
    *,
    measure_runs: int = 13,
    warmup: int = 5,
    bitcorrect_repeats: int = 16,
    abs_gate: float = 1e-3,
    verbose: bool = True,
):
    """Directed search -> the fastest VALID fused chunked-backward callable.

    Phase A: predict_variants + rank_variants (NO GPU pipeline-state; tilelang.lower
    only). Phase B: walk ranked most-promising-first. The rank-4 baseline is the
    live 3-dispatch ``_mamba3_chunked_backward_path_c`` (pre-validated anchor). Each
    higher-ranked CLEAN-SPLICE variant is compiled (crash caught -> infeasible,
    CONTINUE), bit-correct gated vs path-b GOLD over ``bitcorrect_repeats`` fresh
    runs (8 grads < abs_gate, deterministic), then median-timed. Absorption-gated
    variants (B1->B0 fused) are honestly marked infeasible (they would re-introduce
    the dA_tail race). Branch-and-bound early-stop on the floor lower bound.

    Returns (best_variant, ranked, trace). ``best`` is ALWAYS bit-correct and never
    slower than the baseline; if nothing beats it, best == baseline (no regression).
    """
    import mlx.core as mx
    from cppmega_mlx.nn._tilelang.mamba3_path_c import (
        _mamba3_chunked_backward_path_c,
        _force_chunked_command_buffer_boundary,
    )

    _force_chunked_command_buffer_boundary()
    inp = _build_eval_inputs(dims)

    # ---- Phase A ----
    variants = predict_variants(dims)
    ranked = rank_variants(variants)
    if verbose:
        print("=== PHASE A: ranked feasible variants (static, no pipeline-state) ===")
        for i, v in enumerate(ranked):
            print(f"  rank{i+1} {v.variant_id} dispatch={v.dispatch_count} "
                  f"recovered_floor={v.recovered_floor_us:.0f}us "
                  f"max_nbuf={v.max_nbuf} max_phys={v.max_phys} "
                  f"absorb={v.requires_recompute_absorption}")

    trace: list[dict] = []
    baseline = next(v for v in ranked if v.dispatch_count == 3)

    # ---- baseline anchor: the LIVE 3-dispatch wrapper (already bit-correct) ----
    base_ok, base_err, _bw = _bitcorrect_direct(
        _mamba3_chunked_backward_path_c, inp, bitcorrect_repeats, abs_gate)
    base_us = _measure_median(
        _mamba3_chunked_backward_path_c, inp, measure_runs, warmup)
    baseline.compiled = True
    baseline.bit_correct = base_ok
    baseline.max_grad_err = base_err
    baseline.measured_us = base_us
    baseline.status = "baseline_anchor" + ("" if base_ok else "_BITCORRECT_FAIL")
    if not base_ok:
        raise RuntimeError(
            f"RULE #1: the baseline 3-dispatch chunked bwd FAILED its own bit-correct "
            f"gate (maxgraderr={base_err:.2e}); the ground-truth anchor must be valid. "
            "Refusing to run the search against an invalid anchor.")
    if verbose:
        print(f"\n[baseline {baseline.variant_id}] 3 dispatches  "
              f"bit_correct={base_ok} maxgraderr={base_err:.2e} median={base_us:.1f}us")
    best = baseline
    trace.append(_trace_row(baseline))

    # ---- Phase B: measured iterate ----
    compute_floor_us = 0.0
    for v in ranked:
        if v is baseline:
            continue
        lb_wall = v.dispatch_count * _FLOOR_US + compute_floor_us
        if lb_wall >= best.measured_us:
            v.status = "early_stop_dominated"
            if verbose:
                print(f"\n[{v.variant_id}] EARLY-STOP (branch-and-bound): "
                      f"lb_wall={lb_wall:.0f}us >= best={best.measured_us:.1f}us")
            trace.append(_trace_row(v))
            continue
        if not _grouping_is_clean_splice(v.grouping):
            v.status = "infeasible_recompute_absorption"
            v.crashed = True
            v.crash_reason = (
                "B1->B0 fused boundary needs absorbing the dA_cumsum_tail "
                "intra-threadgroup atomic (the deliberately-discarded flaky kernel "
                "output that was moved to deterministic post-kernel MLX). The clean "
                "multi-T.Kernel splice keeps that recompute external; absorbing it "
                "re-introduces the race -> marked infeasible_measured (HONEST), "
                "search continues.")
            if verbose:
                print(f"\n[{v.variant_id}] INFEASIBLE (absorption-gated, HONEST): "
                      f"{v.crash_reason}")
            trace.append(_trace_row(v))
            continue
        # compile (catch XPC/pipeline/watchdog crash -> mark, continue)
        try:
            bwd = make_fused_backward(v.grouping, dims)
            # force compile NOW (lru cache) so a pipeline crash surfaces here
            _ = _bitcorrect_direct(bwd, inp, 1, abs_gate)
            v.compiled = True
        except Exception as e:  # noqa: BLE001
            v.compiled = False
            v.crashed = True
            v.crash_reason = f"{type(e).__name__}: {str(e)[:240]}"
            v.status = "infeasible_measured_compile_or_runtime"
            if verbose:
                print(f"\n[{v.variant_id}] COMPILE/RUNTIME CRASH (caught, continue): "
                      f"{v.crash_reason}")
            trace.append(_trace_row(v))
            continue
        # bit-correct gate over N fresh runs
        ok, err, worst = _bitcorrect_direct(bwd, inp, bitcorrect_repeats, abs_gate)
        v.bit_correct = ok
        v.max_grad_err = err
        if not ok:
            v.status = "bitcorrect_fail"
            if verbose:
                bad = {k: round(x, 5) for k, x in worst.items() if x >= abs_gate}
                print(f"\n[{v.variant_id}] BITCORRECT FAIL maxgraderr={err:.2e} "
                      f"failing={bad} (not selected, fail-loud)")
            trace.append(_trace_row(v))
            continue
        wall = _measure_median(bwd, inp, measure_runs, warmup)
        v.measured_us = wall
        if wall < best.measured_us:
            v.status = "SELECTED_faster"
            if verbose:
                print(f"\n[{v.variant_id}] bit_correct=True maxgraderr={err:.2e} "
                      f"median={wall:.1f}us  <-- NEW BEST (baseline {baseline.measured_us:.1f}us, "
                      f"recovered {baseline.measured_us - wall:.0f}us)")
            best = v
        else:
            v.status = "valid_but_not_faster"
            if verbose:
                print(f"\n[{v.variant_id}] bit_correct=True median={wall:.1f}us "
                      f">= best {best.measured_us:.1f}us (valid, not faster)")
        trace.append(_trace_row(v))

    if verbose:
        print(f"\n=== SELECTED: {best.variant_id} "
              f"({best.dispatch_count} dispatch{'es' if best.dispatch_count != 1 else ''}) "
              f"median={best.measured_us:.1f}us ===")
    return best, ranked, trace

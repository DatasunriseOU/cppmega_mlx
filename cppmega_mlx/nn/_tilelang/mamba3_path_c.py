"""Path C port of Mamba3 MIMO fwd+bwd via TileLang DSL ``@T.prim_func`` lowering.

This module is the Path C counterpart to :mod:`cppmega_mlx.nn._tilelang.mamba3`
(Path B). Path B writes MSL by hand and dispatches it via
``mx.fast.metal_kernel``. Path C writes the *same* selective-scan kernel as
TileLang DSL and dispatches it through ``tilelang.compile(...,
execution_backend="tvm_ffi", out_idx=...)`` into caller-owned MLX buffers.

Why ship both?
--------------

Path B is the shipped hand-written Metal baseline. Path C is intentionally the
*same algorithm expressed in the high-level DSL* so we can:

1. prove the TileLang Metal backend can lower a non-trivial selective-scan
   kernel end to end (against the patched apple-head TileLang
   ``cppmega/gemm-mixed-dtype-metal``);
2. side-by-side bench the lowered MSL against the hand-written MSL — if Path C
   is within 10 percent of Path B the DSL becomes the maintainable entry point;
3. provide a documentation/reproducibility artifact for the upstream PR thread
   (the lowered MSL is captured at
   ``docs/tilelang_ports/mamba3_path_c_lowered.metal``).

Numerical contract
------------------

The kernels accept FP32 and BF16 carrier buffers directly. Recurrence state,
reverse-scan state, and scalar reductions stay in FP32 registers; stores cast
back to the dtype of the corresponding caller-owned buffer. Unsupported dtypes
still fail closed instead of silently materializing large cast buffers. At FP32
the Path C and Path B kernels are *bit identical* on the tested shapes. The
parity budget retained in tests is the conservative atol=1e-4 / rtol=1e-3.

Public surface
--------------

* :func:`mamba3_mimo_fwd_path_c` — fwd lane scan returning ``(y, h_last)``.
* :func:`mamba3_mimo_bwd_path_c` — bwd lane scan returning grads w.r.t.
  ``(x, B, C, z, A, dt, D, h0)``. The generated hot kernel writes per-lane
  partial gradients and leaves P-axis reduction to MLX tensor reductions,
  matching Path B's fast memory/barrier shape while still compiling the scan
  itself through TileLang -> TVM -> tvm-ffi. Long sequences consume explicit
  state snapshots so the reverse pass does not reconstruct ``h_{t-1}`` through
  ``1 / decay``. Public ``out=`` is intentionally fail-closed.
* :func:`mamba3_mimo_apply_path_c` — convenience fwd surface returning ``y``.
* :func:`mamba3_mimo_apply_training_path_c` — training-only surface returning
  ``y`` through the fastest verified VJP route: FP32 reuses forward snapshots;
  BF16 keeps the existing production Path C backward because it is faster.
* :func:`mamba3_mimo_apply_with_state_training_path_c` — same training policy
  while returning ``(y, h_last)`` for model-forward call sites.
* :func:`mamba3_mimo_apply_with_state_path_c` — returns ``(y, h_last)`` so
  model dispatch does not re-run forward just to assemble the inference cache.
* :func:`mamba3_mimo_apply_with_state_path_c_fwd_path_b_bwd` — AUTO-only
  hybrid surface: TileLang DSL forward, proven/receipted shape gate, Path B
  backward until Path C backward earns the same no-worse receipt.
* :func:`mamba3_mimo_path_c_status` — preflight check for the lowered TileLang
  DSL kernel; explicit Path C dispatch fails closed when TileLang cannot lower.

Threadgroup tuning
------------------

The Path B grid uses one thread per (b, h, p) lane with up to 256 threads per
threadgroup, matching the Apple Metal target's 1024-thread / 32 KB-shared
ceilings. Path C uses the same one-thread-per-lane algorithm and the same
256-thread cap; keeping this aligned is part of the Path C >= Path B contract.
The TileLang-lowered scan keeps ``h_state[STATE]`` and backward ``dh[STATE]`` in
per-thread registers, so the entire scan stays per-lane and avoids shared-memory
traffic.

Apple/M4 Max threadgroup limits (from ``tilelang.target.Target("metal")``):
  ``-max_num_threads=256 -max_shared_memory_per_block=32768``

The DSL never exceeds these because the entire scan is per-thread register
work; the only memory traffic is global loads/stores.

PEP-563 caveat
--------------

This module deliberately does *not* use ``from __future__ import annotations``.
TileLang's eager builder reads the inner ``@T.prim_func``'s annotations through
``typing.get_type_hints``, which walks ``__closure__`` to find non-local names
like the dimension constants we close over. When PEP-563 is active the
annotations are strings and ``get_type_hints`` only sees variables that are
also referenced from the function body's own bytecode (the ``co_freevars``
list). The Mamba3 PrimFunc body computes derived strides from ``HEADDIM`` /
``HEADS`` rather than naming ``BATCH`` directly; under PEP-563 that turns into
a ``NameError: BATCH`` from the lowering step. Disabling PEP-563 keeps the
annotations as live ``T.Tensor`` objects whose shape ints were already
captured as Python integers when the inner function was defined, sidestepping
the closure-walk path entirely.
"""

from dataclasses import dataclass
from functools import lru_cache
import atexit
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Literal, cast

import mlx.core as mx

from cppmega_mlx.nn._tilelang import _msl_transform
from cppmega_mlx.nn._tilelang._engine_dispatch import dispatch_lower
from cppmega_mlx.nn._tilelang._msl_transform import (
    MSLDispatchUnsupported,
    can_run_metal,
)
from cppmega_mlx.nn._tilelang.mamba3 import _validate_inputs


_REUSABLE_SCALAR_BINDINGS = frozenset(
    {
        "decay",
        "decay_1",
        "sig_z",
        "silu_z",
        "silu_dz",
        "d_silu",
        "d_y_skipped",
    }
)
_FLOAT_BINDING_RE = re.compile(r"\bfloat (?P<name>[A-Za-z_]\w*) = (?P<expr>.*);")
_FLOAT_LITERAL_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?f?$", re.IGNORECASE)
_FWD_OUTPUT_NAMES = ("y", "h_last")
_FWD_OUTPUT_IDX = (8, 9)
_FWD_SNAPSHOT_OUTPUT_NAMES = ("y", "h_last", "h_snap")
_FWD_SNAPSHOT_OUTPUT_IDX = (8, 9, 10)
_BWD_SIMD_OUTPUT_NAMES = (
    "dx",
    "dz",
    "dB",
    "dC",
    "dA",
    "ddt",
    "dD_batch",
    "dh0",
)
_BWD_SIMD_OUTPUT_IDX = (9, 10, 11, 12, 13, 14, 15, 16)
_BWD_PARTIAL_OUTPUT_NAMES = (
    "dx",
    "dz",
    "dB_partial",
    "dC_partial",
    "dA_partial",
    "ddt_partial",
    "dD_partial",
    "dh0",
)
_BWD_PARTIAL_OUTPUT_IDX = (9, 10, 11, 12, 13, 14, 15, 16)
_BWD_SCRATCH_OUTPUT_NAMES = (
    "dx",
    "dz",
    "dB_partial",
    "dC_partial",
    "dA_partial",
    "ddt_partial",
    "dD_partial",
    "dh0",
)
_BWD_SCRATCH_OUTPUT_IDX = (9, 10, 11, 12, 13, 14, 15, 16)
_BWD_SCRATCH_WORKSPACE_NAMES = ("h_steps_scratch",)
_BWD_REDUCE_OUTPUT_NAMES = ("dB", "dC", "dA", "ddt", "dD")
_BWD_REDUCE_OUTPUT_IDX = (5, 6, 7, 8, 9)
_BWD_PARTIAL_DTYPE_DEFAULT = ("float32", "float32", "float32", "float32", "float32")
_BWD_PARTIAL_DTYPE_BF16_COMPACT = (
    "bfloat16",
    "bfloat16",
    "float32",
    "float32",
    "float32",
)
# Correctness first for full-model Path C bwd: cache every h_t boundary and
# avoid reconstructing h_{t-1} through 1 / decay. Larger blocks need a range
# proof/autotune gate because real bf16 model weights can drive decay to zero.
_BWD_SNAPSHOT_BLOCK = 1
_REPO_ROOT = Path(__file__).resolve().parents[3]
_PATH_C_AUTO_PROMOTION_RECEIPT = (
    _REPO_ROOT / "bench" / "tilelang_ports" / "mamba3_path_c.json"
)
_MAX_THREADS = 256
_REDUCE_MAX_THREADS = 1024
_Z3_DISABLE_ENV = (
    "TILELANG_DISABLE_Z3",
    "CPPMEGA_DISABLE_Z3",
    "CPPMEGA_DISABLE_MAMBA3_PATH_C_Z3",
)
Mamba3PathCZ3Policy = Literal["env", "enabled", "disabled"]

Mamba3FwdOwnerOutputs = tuple[mx.array, mx.array]
Mamba3BwdOwnerOutputs = tuple[
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
]


def _tl_dtype_for(dtype: mx.Dtype) -> str | None:
    if dtype == mx.float32:
        return "float32"
    if dtype == mx.bfloat16:
        return "bfloat16"
    return None


def _tl_dtype_for_auto(array: mx.array) -> str | None:
    return _tl_dtype_for(array.dtype)


def _reuse_tilelang_scalar_bindings(body: str) -> str:
    """Reuse scalar bindings that TileLang already emitted in the lowered body."""

    out: list[str] = []
    replacements: list[tuple[str, str]] = []

    for raw_line in body.splitlines():
        line = raw_line
        for expr, name in replacements:
            line = line.replace(expr, name)

        match = _FLOAT_BINDING_RE.search(line)
        if match is not None and match.group("name") in _REUSABLE_SCALAR_BINDINGS:
            expr = match.group("expr").strip()
            if expr and expr != match.group("name") and _FLOAT_LITERAL_RE.fullmatch(expr) is None:
                replacements.append((expr, match.group("name")))

        out.append(line)

    suffix = "\n" if body.endswith("\n") else ""
    return "\n".join(out) + suffix


def _source_with_reused_scalar_bindings(
    lowering: _msl_transform.TileLangMSLLowering,
) -> str:
    """Return the full lowered MSL string matching the dispatched source body."""

    prelude, signature, body_text = _msl_transform._split_kernel_msl(lowering.msl_text)
    body = _reuse_tilelang_scalar_bindings(body_text[1:-1])
    return (
        f"{prelude}\n"
        f"kernel void {lowering.kernel_name}({signature}) "
        f"{{{body}}}\n"
    )


# ---------------------------------------------------------------------------
# Status surface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Mamba3PathCStatus:
    """Runtime status for the Path C TileLang DSL Mamba3 kernel."""

    available: bool
    reason: str


@dataclass(frozen=True)
class Mamba3PathCSchedulePlan:
    """Rule/proof plan for one Mamba3 Path C shape."""

    batch: int
    seq: int
    heads: int
    headdim: int
    state: int
    dtype: str
    lanes: int
    threads: int
    grid_blocks: int
    fwd_path_c_candidate: bool
    bwd_path_c_candidate: bool
    z3_used: bool
    z3_proved: bool
    reason: str

    @property
    def mode(self) -> str:
        if self.fwd_path_c_candidate and self.bwd_path_c_candidate:
            return "path_c_fwd_bwd"
        if self.fwd_path_c_candidate:
            return "path_c_fwd_path_b_bwd"
        return "path_b"

    def as_feature_dict(self) -> dict[str, bool | int | str]:
        return {
            "batch": self.batch,
            "seq": self.seq,
            "heads": self.heads,
            "headdim": self.headdim,
            "state": self.state,
            "dtype": self.dtype,
            "lanes": self.lanes,
            "threads": self.threads,
            "grid_blocks": self.grid_blocks,
            "fwd_path_c_candidate": self.fwd_path_c_candidate,
            "bwd_path_c_candidate": self.bwd_path_c_candidate,
            "mode": self.mode,
            "z3_used": self.z3_used,
            "z3_proved": self.z3_proved,
            "reason": self.reason,
        }


def _tilelang_available() -> tuple[bool, str]:
    _msl_transform.ensure_libz3_preloaded()
    try:
        import tilelang  # noqa: F401
        from tilelang import tvm as _tvm  # noqa: F401
        from tilelang.engine.lower import lower as _lower  # noqa: F401
        import tilelang.language as _T  # noqa: F401
    except Exception as exc:  # pragma: no cover - macOS without tilelang
        return False, f"tilelang import failed: {exc}"
    return True, "tilelang importable"


def _z3_disabled(policy: Mamba3PathCZ3Policy = "env") -> bool:
    if policy == "enabled":
        return False
    if policy == "disabled":
        return True
    if policy != "env":
        raise ValueError(f"invalid Mamba3 Path C Z3 policy: {policy!r}")
    return any(
        os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}
        for name in _Z3_DISABLE_ENV
    )


def _z3_proves_mamba3_lane_mapping(
    *,
    batch: int,
    seq: int,
    heads: int,
    headdim: int,
    state: int,
    z3_policy: Mamba3PathCZ3Policy = "env",
) -> tuple[bool, bool, str]:
    """Prove that the per-lane schedule's derived indices stay in-bounds."""

    if _z3_disabled(z3_policy):
        reason = "z3 disabled by policy"
        if z3_policy == "env":
            reason = "z3 disabled by environment"
        return False, False, reason
    try:
        import z3  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional local dependency
        return False, False, f"z3 unavailable: {type(exc).__name__}: {exc}"

    lane = z3.Int("lane")
    t = z3.Int("t")
    n = z3.Int("n")
    lanes = batch * heads * headdim
    p = lane % headdim
    h = (lane / headdim) % heads
    b = lane / (headdim * heads)
    xz_idx = ((b * seq + t) * heads + h) * headdim + p
    bc_idx = ((b * seq + t) * heads + h) * state + n
    h_idx = ((b * heads + h) * headdim + p) * state + n

    solver = z3.Solver()
    solver.set("timeout", 50)
    solver.add(0 <= lane, lane < lanes)
    solver.add(0 <= t, t < seq)
    solver.add(0 <= n, n < state)
    solver.add(
        z3.Or(
            p < 0,
            p >= headdim,
            h < 0,
            h >= heads,
            b < 0,
            b >= batch,
            xz_idx < 0,
            xz_idx >= batch * seq * heads * headdim,
            bc_idx < 0,
            bc_idx >= batch * seq * heads * state,
            h_idx < 0,
            h_idx >= batch * heads * headdim * state,
        )
    )
    try:
        result = solver.check()
    except Exception as exc:  # pragma: no cover - defensive z3 boundary
        return True, False, f"z3 raised {type(exc).__name__}: {exc}"
    if result == z3.unsat:
        return True, True, "z3 proved per-lane index decomposition and buffer bounds"
    if result == z3.unknown:
        return True, False, "z3 returned unknown for per-lane index proof"
    return True, False, "z3 found an out-of-bounds lane/index witness"


@lru_cache(maxsize=128)
def mamba3_path_c_schedule_plan(
    *,
    batch: int,
    seq: int,
    heads: int,
    headdim: int,
    state: int,
    dtype: str = "float32",
    z3_policy: Mamba3PathCZ3Policy = "env",
) -> Mamba3PathCSchedulePlan:
    """Return the rule + Z3 schedule plan used by the automatic Path C gate."""

    lanes = batch * heads * headdim
    threads = _threads_for(lanes)
    grid_blocks = 0 if lanes <= 0 else math.ceil(lanes / threads)
    positive_shape = all(value > 0 for value in (batch, seq, heads, headdim, state))
    if not positive_shape:
        return Mamba3PathCSchedulePlan(
            batch=batch,
            seq=seq,
            heads=heads,
            headdim=headdim,
            state=state,
            dtype=dtype,
            lanes=lanes,
            threads=threads,
            grid_blocks=grid_blocks,
            fwd_path_c_candidate=False,
            bwd_path_c_candidate=False,
            z3_used=False,
            z3_proved=False,
            reason="non-positive Mamba3 Path C shape",
        )
    z3_used, z3_proved, z3_reason = _z3_proves_mamba3_lane_mapping(
        batch=batch,
        seq=seq,
        heads=heads,
        headdim=headdim,
        state=state,
        z3_policy=z3_policy,
    )
    fwd_candidate = dtype in {"float32", "bfloat16"} and threads <= 256 and z3_proved
    bwd_candidate = fwd_candidate
    bwd_reason = (
        "bwd emits TileLang per-lane partial gradients and reduces P-axis "
        "outside the hot scan kernel"
    )
    reason = (
        f"rule: fp32-accumulating {dtype} per-lane scan with {threads} "
        f"threads over {grid_blocks} "
        f"blocks; {z3_reason}; bwd reverse pass consumes explicit state "
        f"snapshot tensor boundaries instead of inverse h_prev reconstruction; "
        f"{bwd_reason}"
    )
    return Mamba3PathCSchedulePlan(
        batch=batch,
        seq=seq,
        heads=heads,
        headdim=headdim,
        state=state,
        dtype=dtype,
        lanes=lanes,
        threads=threads,
        grid_blocks=grid_blocks,
        fwd_path_c_candidate=fwd_candidate,
        bwd_path_c_candidate=bwd_candidate,
        z3_used=z3_used,
        z3_proved=z3_proved,
        reason=reason,
    )


def mamba3_path_c_receipt_auto_mode(
    receipt_path: Path = _PATH_C_AUTO_PROMOTION_RECEIPT,
    *,
    batch: int,
    seq: int,
    heads: int,
    headdim: int,
    state: int,
    dtype: str,
    z3_policy: Mamba3PathCZ3Policy = "env",
) -> str:
    """Return the fail-closed AUTO mode selected by the bench receipt."""

    plan = mamba3_path_c_schedule_plan(
        batch=batch,
        seq=seq,
        heads=heads,
        headdim=headdim,
        state=state,
        dtype=dtype,
        z3_policy=z3_policy,
    )
    if not plan.fwd_path_c_candidate or not plan.z3_proved:
        return "path_b"
    try:
        data = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "path_b"
    if not isinstance(data, dict):
        return "path_b"
    if data.get("kernel") != "mamba3_mimo_path_c_vs_path_b":
        return "path_b"
    strict_policy = data.get("strict_policy")
    if not isinstance(strict_policy, dict):
        return "path_b"
    if strict_policy.get("requires_path_b_and_path_c") is not True:
        return "path_b"
    if strict_policy.get("phase") != "fwd":
        return "path_b"

    decision = data.get("scheduler_decision")
    if not isinstance(decision, dict):
        return "path_b"
    mode = decision.get("mode")
    if mode not in {"path_c_fwd_path_b_bwd", "path_c_fwd_bwd"}:
        return "path_b"
    if decision.get("selected_forward_kernel") != "path_c_tilelang_dsl":
        return "path_b"
    expected_bwd = (
        "path_c_tilelang_dsl" if mode == "path_c_fwd_bwd" else "metal_kernel_bwd_v1"
    )
    if decision.get("selected_backward_kernel") != expected_bwd:
        return "path_b"

    shape = data.get("shape")
    expected_shape = {
        "batch": batch,
        "seq": seq,
        "heads": heads,
        "headdim": headdim,
        "state": state,
        "dtype": dtype,
    }
    if not isinstance(shape, dict) or any(shape.get(k) != v for k, v in expected_shape.items()):
        return "path_b"

    timings = data.get("timings")
    if not isinstance(timings, dict):
        return "path_b"
    fwd_b = timings.get("fwd_path_b")
    fwd_c = timings.get("fwd_path_c")
    if not isinstance(fwd_b, dict) or not isinstance(fwd_c, dict):
        return "path_b"
    try:
        fwd_b_ms = float(fwd_b["median_ms"])
        fwd_c_ms = float(fwd_c["median_ms"])
        max_ratio = float(strict_policy["path_c_fwd_over_path_b_max_ratio"])
    except (KeyError, TypeError, ValueError):
        return "path_b"
    if not (math.isfinite(fwd_b_ms) and math.isfinite(fwd_c_ms) and fwd_b_ms > 0):
        return "path_b"
    if (fwd_c_ms / fwd_b_ms) > max_ratio:
        return "path_b"

    if mode == "path_c_fwd_bwd":
        if not plan.bwd_path_c_candidate:
            return "path_b"
        try:
            bwd_ratio = float(decision["ratios"]["bwd_path_c_over_path_b"])
            fwd_bwd_ratio = float(decision["ratios"]["fwd_bwd_path_c_over_path_b"])
            max_bwd = float(strict_policy["path_c_bwd_over_path_b_max_ratio"])
            max_fwd_bwd = float(
                strict_policy["path_c_fwd_bwd_over_path_b_max_ratio"]
            )
        except (KeyError, TypeError, ValueError):
            return "path_b"
        if not (math.isfinite(bwd_ratio) and math.isfinite(fwd_bwd_ratio)):
            return "path_b"
        if bwd_ratio > max_bwd or fwd_bwd_ratio > max_fwd_bwd:
            return "path_b"

    return cast(str, mode)


def mamba3_path_c_receipt_allows_auto_promotion(
    receipt_path: Path = _PATH_C_AUTO_PROMOTION_RECEIPT,
    *,
    batch: int,
    seq: int,
    heads: int,
    headdim: int,
    state: int,
    dtype: str,
    z3_policy: Mamba3PathCZ3Policy = "env",
) -> bool:
    """Fail-closed automatic Path C promotion gate backed by bench memory."""

    return (
        mamba3_path_c_receipt_auto_mode(
            receipt_path,
            batch=batch,
            seq=seq,
            heads=heads,
            headdim=headdim,
            state=state,
            dtype=dtype,
            z3_policy=z3_policy,
        )
        != "path_b"
    )


def mamba3_path_c_auto_fwd_path_b_bwd_allowed(
    x: mx.array,
    B: mx.array,
    C: mx.array,
    z: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
    *,
    receipt_path: Path = _PATH_C_AUTO_PROMOTION_RECEIPT,
    z3_policy: Mamba3PathCZ3Policy = "env",
) -> bool:
    """Return whether AUTO may use Path C fwd with Path B bwd for these inputs."""

    try:
        batch, seq, heads, headdim, state = _validate_inputs(x, B, C, z, A, dt, D, h0)
    except Exception:
        return False
    dtype = _tl_dtype_for_auto(x)
    if dtype is None:
        return False
    return (
        mamba3_path_c_receipt_auto_mode(
            receipt_path,
            batch=batch,
            seq=seq,
            heads=heads,
            headdim=headdim,
            state=state,
            dtype=dtype,
            z3_policy=z3_policy,
        )
        == "path_c_fwd_path_b_bwd"
    )


def mamba3_path_c_auto_mode_for_inputs(
    x: mx.array,
    B: mx.array,
    C: mx.array,
    z: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
    *,
    receipt_path: Path = _PATH_C_AUTO_PROMOTION_RECEIPT,
    z3_policy: Mamba3PathCZ3Policy = "env",
) -> str:
    """Return AUTO's Path C mode for these inputs, or ``path_b``."""

    try:
        batch, seq, heads, headdim, state = _validate_inputs(x, B, C, z, A, dt, D, h0)
    except Exception:
        return "path_b"
    dtype = _tl_dtype_for_auto(x)
    if dtype is None:
        return "path_b"
    return mamba3_path_c_receipt_auto_mode(
        receipt_path,
        batch=batch,
        seq=seq,
        heads=heads,
        headdim=headdim,
        state=state,
        dtype=dtype,
        z3_policy=z3_policy,
    )


def mamba3_mimo_path_c_status() -> Mamba3PathCStatus:
    """Return whether the Path C TileLang DSL kernel can dispatch on this host."""

    if not can_run_metal():
        return Mamba3PathCStatus(
            available=False,
            reason="MLX Metal backend is not available on the default GPU device",
        )
    ok, reason = _tilelang_available()
    if not ok:
        return Mamba3PathCStatus(available=False, reason=reason)
    try:
        fwd_kernel, fwd_lowering = _fwd_kernel_for(1, 4, 1, 2, 4, return_msl=True)
        bwd_kernel, bwd_lowering = _bwd_partial_kernel_for_state_snapshots(
            1, 4, 1, 2, 4
        )
        del fwd_kernel, bwd_kernel
    except Exception as exc:
        return Mamba3PathCStatus(
            available=False,
            reason=f"TileLang/MLX lowering failed for Mamba3 Path C: {type(exc).__name__}: {exc}",
        )
    if "kernel void" not in fwd_lowering.msl_text:
        return Mamba3PathCStatus(False, "lowered Mamba3 Path C fwd source has no kernel")
    if "kernel void" not in bwd_lowering.msl_text:
        return Mamba3PathCStatus(False, "lowered Mamba3 Path C bwd source has no kernel")
    return Mamba3PathCStatus(available=True, reason="Path C TileLang DSL ready")


# ---------------------------------------------------------------------------
# TileLang PrimFunc factories (cached on shape signature)
# ---------------------------------------------------------------------------


def _threads_for(lanes: int) -> int:
    """Return the threadgroup size for a per-lane kernel.

    Keep this identical to Path B. A previous 32-thread cap multiplied the
    number of threadgroups by 8 on the real ``H*P=3584`` shape and made Path C
    lose to the baseline before scheduler-level optimizations even had a chance
    to matter.
    """

    if lanes <= 0:
        return 1
    return min(_MAX_THREADS, lanes)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _bwd_partial_dtypes_for_input_dtypes(
    dtypes: dict[str, str],
) -> tuple[str, str, str, str, str]:
    if all(
        dtypes[name] == "bfloat16"
        for name in ("dy", "x", "B", "C", "z", "A", "dt", "D", "h0")
    ):
        # dB/dC partials dominate bwd memory traffic at full shape. Storing
        # them as BF16 cuts the producer and reducer bandwidth while dA/ddt/dD
        # stay FP32 to preserve the more sensitive scalar reductions.
        return _BWD_PARTIAL_DTYPE_BF16_COMPACT
    return _BWD_PARTIAL_DTYPE_DEFAULT


def _bwd_threads_for(lanes: int, headdim: int) -> int:
    """Return a bwd thread count that keeps each P row inside one threadgroup."""

    base = _threads_for(lanes)
    if (
        headdim > 0
        and headdim <= base
        and base % headdim == 0
        and lanes % base == 0
    ):
        return base
    upper = min(1024, lanes)
    for candidate in range(upper - (upper % 32), 0, -32):
        if (
            candidate >= headdim
            and candidate % headdim == 0
            and lanes % candidate == 0
        ):
            return candidate
    return base


@dataclass(frozen=True)
class _LocalSnapshotPlan:
    policy: str
    chunk_size: int
    chunk_count: int
    snapshot_count: int
    state_elements: int
    snapshot_elements: int
    state_dtype: str


@dataclass(frozen=True)
class _LocalAliasPlan:
    input_output_alias: bool
    in_place_requested: bool
    in_place_allowed: bool
    reason: str


@dataclass(frozen=True)
class _LocalScanPlan:
    direction: str
    snapshot_plan: _LocalSnapshotPlan
    rematerialization_policy: str
    alias_plan: _LocalAliasPlan
    host_sync_required: bool
    device_event_required: bool
    fused_post_ops: tuple[str, ...]


def _fallback_recurrence_scan_plan(
    *,
    name: str,
    direction: str,
    sequence_length: int,
    state_shape: tuple[int, ...],
    state_dtype: str,
    chunk_size: int,
    decay_may_underflow: bool,
    input_output_alias: bool,
    in_place_requested: bool,
    fused_post_ops: tuple[str, ...],
) -> _LocalScanPlan:
    del name
    chunk_count = (
        (sequence_length + chunk_size - 1) // chunk_size if sequence_length else 0
    )
    state_elements = math.prod(state_shape)
    needs_snapshots = (
        direction == "reverse"
        and sequence_length > chunk_size
        and decay_may_underflow
    )
    snapshot_count = chunk_count + 1 if needs_snapshots else 0
    return _LocalScanPlan(
        direction=direction,
        snapshot_plan=_LocalSnapshotPlan(
            policy="state-boundary-cache" if needs_snapshots else "none",
            chunk_size=chunk_size,
            chunk_count=chunk_count,
            snapshot_count=snapshot_count,
            state_elements=state_elements,
            snapshot_elements=snapshot_count * state_elements,
            state_dtype=state_dtype,
        ),
        rematerialization_policy=(
            "reuse-forward-state-snapshots"
            if needs_snapshots
            else "direct-recompute"
            if direction == "reverse"
            else "not-needed"
        ),
        alias_plan=_LocalAliasPlan(
            input_output_alias=input_output_alias,
            in_place_requested=in_place_requested,
            in_place_allowed=False,
            reason=(
                "input_output_alias_without_in_place_proof"
                if input_output_alias
                else "distinct_input_output_buffers"
            ),
        ),
        host_sync_required=False,
        device_event_required=False,
        fused_post_ops=fused_post_ops,
    )


def _plan_recurrence_scan_compat(**kwargs):
    try:
        from tilelang.analysis.scan_plan import plan_recurrence_scan
    except ModuleNotFoundError as exc:
        if exc.name not in {"tilelang.analysis", "tilelang.analysis.scan_plan"}:
            raise
        return _fallback_recurrence_scan_plan(**kwargs)
    return plan_recurrence_scan(**kwargs)


def _bwd_scan_plan_for(
    *,
    batch: int,
    seq: int,
    heads: int,
    headdim: int,
    state: int,
) -> Any:
    """Plan Mamba3 reverse recurrence state-cache policy."""

    return _plan_recurrence_scan_compat(
        name="mamba3_path_c_bwd",
        direction="reverse",
        sequence_length=seq,
        state_shape=(batch, heads, headdim, state),
        state_dtype="float32",
        chunk_size=_BWD_SNAPSHOT_BLOCK,
        decay_may_underflow=True,
        input_output_alias=False,
        in_place_requested=False,
        fused_post_ops=("skip_D", "silu_gate"),
    )


@lru_cache(maxsize=128)
def _fwd_kernel_for(
    BATCH: int,
    SEQ: int,
    HEADS: int,
    HEADDIM: int,
    STATE: int,
    x_dtype: str = "float32",
    B_dtype: str = "float32",
    C_dtype: str = "float32",
    z_dtype: str = "float32",
    A_dtype: str = "float32",
    dt_dtype: str = "float32",
    D_dtype: str = "float32",
    h0_dtype: str = "float32",
    y_dtype: str = "float32",
    h_last_dtype: str = "float32",
    *,
    return_msl: bool = False,
) -> tuple[Any, _msl_transform.TileLangMSLLowering]:
    """Build & cache the Path C TileLang fwd kernel for a given (B, T, H, P, N)."""

    import tilelang.language as T

    LANES = BATCH * HEADS * HEADDIM
    THREADS = _threads_for(LANES)
    accum_dtype = "float32"
    pure_fp32_carriers = all(
        dtype == "float32"
        for dtype in (
            x_dtype,
            B_dtype,
            C_dtype,
            z_dtype,
            A_dtype,
            dt_dtype,
            D_dtype,
            h0_dtype,
            y_dtype,
            h_last_dtype,
        )
    )

    @T.prim_func
    def fwd(
        x: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), x_dtype),
        B: T.Tensor((BATCH, SEQ, HEADS, STATE), B_dtype),
        C: T.Tensor((BATCH, SEQ, HEADS, STATE), C_dtype),
        z: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), z_dtype),
        A: T.Tensor((BATCH, SEQ, HEADS), A_dtype),
        dt: T.Tensor((BATCH, SEQ, HEADS), dt_dtype),
        D: T.Tensor((HEADS,), D_dtype),
        h0: T.Tensor((BATCH, HEADS, HEADDIM, STATE), h0_dtype),
        y: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), y_dtype),
        h_last: T.Tensor((BATCH, HEADS, HEADDIM, STATE), h_last_dtype),
    ):
        with T.Kernel(T.ceildiv(LANES, THREADS), threads=THREADS) as bx:
            # Metal already exposes the absolute 1-D lane id. Use it directly
            # so the lowered hot loop matches Path B and does not repeatedly
            # carry blockIdx * THREADS + threadIdx address arithmetic.
            global_lane = T.call_intrin("int32", "tir.metal.thread_position_in_grid_x")
            # Per-lane state lives in registers (size N).
            h_state = T.alloc_local((STATE,), accum_dtype)
            if LANES % THREADS == 0 or global_lane < LANES:
                p = global_lane % HEADDIM
                if BATCH == 1 and pure_fp32_carriers:
                    h = global_lane // HEADDIM
                else:
                    h = (global_lane // HEADDIM) % HEADS
                if BATCH == 1:
                    b = 0
                else:
                    b = global_lane // (HEADDIM * HEADS)
                D_h = T.cast(D[h], accum_dtype)
                for n in T.serial(STATE):
                    h_state[n] = T.cast(h0[b, h, p, n], accum_dtype)
                for t in T.serial(SEQ):
                    A_val = T.cast(A[b, t, h], accum_dtype)
                    dt_val = T.cast(dt[b, t, h], accum_dtype)
                    decay = T.alloc_var(T.float32, init=T.exp(A_val * dt_val))
                    x_val = T.cast(x[b, t, h, p], accum_dtype)
                    z_val = T.cast(z[b, t, h, p], accum_dtype)
                    y_acc = T.alloc_var(T.float32, init=0.0)
                    for n in T.serial(STATE):
                        B_val = T.cast(B[b, t, h, n], accum_dtype)
                        C_val = T.cast(C[b, t, h, n], accum_dtype)
                        new_h = decay * h_state[n] + x_val * B_val
                        h_state[n] = new_h
                        y_acc += new_h * C_val
                    y_skipped = y_acc + D_h * x_val
                    sig_z = 1.0 / (1.0 + T.exp(-z_val))
                    y[b, t, h, p] = T.cast(z_val * sig_z * y_skipped, y_dtype)
                for n in T.serial(STATE):
                    h_last[b, h, p, n] = T.cast(h_state[n], h_last_dtype)

    artifact = dispatch_lower(fwd, target="metal", return_msl=True)
    if hasattr(artifact, "_tilelang_engine_target"):
        raise MSLDispatchUnsupported("Mamba3 Path C requires TileLang MSL extraction metadata")
    lowering = cast(_msl_transform.TileLangMSLLowering, artifact)
    input_names = [
        name for name in lowering.buffer_param_names if name not in _FWD_OUTPUT_NAMES
    ]
    if set(input_names) != {"x", "B", "C", "z", "A", "dt", "D", "h0"}:
        raise MSLDispatchUnsupported(
            "unexpected Mamba3 Path C fwd buffer signature: "
            + ", ".join(lowering.buffer_param_names)
        )
    import tilelang

    kernel = tilelang.compile(
        fwd,
        target=_msl_transform._as_metal_target("metal"),
        execution_backend="tvm_ffi",
        out_idx=list(_FWD_OUTPUT_IDX),
    )
    return kernel, lowering


@lru_cache(maxsize=128)
def _fwd_with_snapshots_kernel_for(
    BATCH: int,
    SEQ: int,
    HEADS: int,
    HEADDIM: int,
    STATE: int,
    x_dtype: str = "float32",
    B_dtype: str = "float32",
    C_dtype: str = "float32",
    z_dtype: str = "float32",
    A_dtype: str = "float32",
    dt_dtype: str = "float32",
    D_dtype: str = "float32",
    h0_dtype: str = "float32",
    y_dtype: str = "float32",
    h_last_dtype: str = "float32",
    h_snap_dtype: str = "float32",
) -> tuple[Any, _msl_transform.TileLangMSLLowering]:
    """Build Path C fwd that also materializes per-step states for training."""

    import tilelang.language as T

    LANES = BATCH * HEADS * HEADDIM
    THREADS = _threads_for(LANES)
    accum_dtype = "float32"
    pure_fp32_carriers = all(
        dtype == "float32"
        for dtype in (
            x_dtype,
            B_dtype,
            C_dtype,
            z_dtype,
            A_dtype,
            dt_dtype,
            D_dtype,
            h0_dtype,
            y_dtype,
            h_last_dtype,
            h_snap_dtype,
        )
    )

    @T.prim_func
    def fwd_with_snapshots(
        x: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), x_dtype),
        B: T.Tensor((BATCH, SEQ, HEADS, STATE), B_dtype),
        C: T.Tensor((BATCH, SEQ, HEADS, STATE), C_dtype),
        z: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), z_dtype),
        A: T.Tensor((BATCH, SEQ, HEADS), A_dtype),
        dt: T.Tensor((BATCH, SEQ, HEADS), dt_dtype),
        D: T.Tensor((HEADS,), D_dtype),
        h0: T.Tensor((BATCH, HEADS, HEADDIM, STATE), h0_dtype),
        y: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), y_dtype),
        h_last: T.Tensor((BATCH, HEADS, HEADDIM, STATE), h_last_dtype),
        h_snap: T.Tensor((BATCH, SEQ + 1, HEADS, HEADDIM, STATE), h_snap_dtype),
    ):
        with T.Kernel(T.ceildiv(LANES, THREADS), threads=THREADS) as _bx:
            global_lane = T.call_intrin("int32", "tir.metal.thread_position_in_grid_x")
            h_state = T.alloc_local((STATE,), accum_dtype)
            if LANES % THREADS == 0 or global_lane < LANES:
                p = global_lane % HEADDIM
                if BATCH == 1 and pure_fp32_carriers:
                    h = global_lane // HEADDIM
                else:
                    h = (global_lane // HEADDIM) % HEADS
                if BATCH == 1:
                    b = 0
                else:
                    b = global_lane // (HEADDIM * HEADS)
                D_h = T.cast(D[h], accum_dtype)
                for n in T.serial(STATE):
                    h_state[n] = T.cast(h0[b, h, p, n], accum_dtype)
                    h_snap[b, 0, h, p, n] = T.cast(h_state[n], h_snap_dtype)
                for t in T.serial(SEQ):
                    A_val = T.cast(A[b, t, h], accum_dtype)
                    dt_val = T.cast(dt[b, t, h], accum_dtype)
                    decay = T.alloc_var(T.float32, init=T.exp(A_val * dt_val))
                    x_val = T.cast(x[b, t, h, p], accum_dtype)
                    z_val = T.cast(z[b, t, h, p], accum_dtype)
                    y_acc = T.alloc_var(T.float32, init=0.0)
                    for n in T.serial(STATE):
                        B_val = T.cast(B[b, t, h, n], accum_dtype)
                        C_val = T.cast(C[b, t, h, n], accum_dtype)
                        new_h = decay * h_state[n] + x_val * B_val
                        h_state[n] = new_h
                        h_snap[b, t + 1, h, p, n] = T.cast(
                            new_h,
                            h_snap_dtype,
                        )
                        y_acc += new_h * C_val
                    y_skipped = y_acc + D_h * x_val
                    sig_z = 1.0 / (1.0 + T.exp(-z_val))
                    y[b, t, h, p] = T.cast(z_val * sig_z * y_skipped, y_dtype)
                for n in T.serial(STATE):
                    h_last[b, h, p, n] = T.cast(h_state[n], h_last_dtype)

    artifact = dispatch_lower(fwd_with_snapshots, target="metal", return_msl=True)
    if hasattr(artifact, "_tilelang_engine_target"):
        raise MSLDispatchUnsupported("Mamba3 Path C requires TileLang MSL extraction metadata")
    lowering = cast(_msl_transform.TileLangMSLLowering, artifact)
    input_names = [
        name
        for name in lowering.buffer_param_names
        if name not in _FWD_SNAPSHOT_OUTPUT_NAMES
    ]
    if set(input_names) != {"x", "B", "C", "z", "A", "dt", "D", "h0"}:
        raise MSLDispatchUnsupported(
            "unexpected Mamba3 Path C fwd snapshot buffer signature: "
            + ", ".join(lowering.buffer_param_names)
        )
    import tilelang

    kernel = tilelang.compile(
        fwd_with_snapshots,
        target=_msl_transform._as_metal_target("metal"),
        execution_backend="tvm_ffi",
        out_idx=list(_FWD_SNAPSHOT_OUTPUT_IDX),
    )
    return kernel, lowering


@lru_cache(maxsize=128)
def _bwd_state_snapshots_kernel_for(
    BATCH: int,
    SEQ: int,
    HEADS: int,
    HEADDIM: int,
    STATE: int,
    x_dtype: str = "float32",
    B_dtype: str = "float32",
    A_dtype: str = "float32",
    dt_dtype: str = "float32",
    h0_dtype: str = "float32",
    h_snap_dtype: str = "float32",
) -> tuple[Any, _msl_transform.TileLangMSLLowering]:
    """Build the forward state-cache kernel used by long-sequence bwd.

    The generic bwd kernel used to reconstruct every ``h_{t-1}`` from ``h_t``
    by walking backwards from ``h_T``. That is correct algebraically, but real
    full-model bf16 runs can drive ``decay`` to zero, making the inverse walk
    produce ``0 * inf`` NaNs. Path C therefore caches the tensor of step
    boundaries and consumes those states directly in backward.
    """

    import tilelang.language as T

    LANES = BATCH * HEADS * HEADDIM
    THREADS = _threads_for(LANES)
    scan_plan = _bwd_scan_plan_for(
        batch=BATCH,
        seq=SEQ,
        heads=HEADS,
        headdim=HEADDIM,
        state=STATE,
    )
    BLOCK = scan_plan.snapshot_plan.chunk_size
    BLOCKS = (SEQ + BLOCK - 1) // BLOCK
    accum_dtype = "float32"

    @T.prim_func
    def snapshots(
        x: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), x_dtype),
        B: T.Tensor((BATCH, SEQ, HEADS, STATE), B_dtype),
        A: T.Tensor((BATCH, SEQ, HEADS), A_dtype),
        dt: T.Tensor((BATCH, SEQ, HEADS), dt_dtype),
        h0: T.Tensor((BATCH, HEADS, HEADDIM, STATE), h0_dtype),
        h_snap: T.Tensor((BATCH, BLOCKS + 1, HEADS, HEADDIM, STATE), h_snap_dtype),
    ):
        with T.Kernel(T.ceildiv(LANES, THREADS), threads=THREADS) as _bx:
            global_lane = T.call_intrin("int32", "tir.metal.thread_position_in_grid_x")
            h_state = T.alloc_local((STATE,), accum_dtype)
            dh = T.alloc_local((STATE,), accum_dtype)
            if LANES % THREADS == 0 or global_lane < LANES:
                p = global_lane % HEADDIM
                if BATCH == 1:
                    h = global_lane // HEADDIM
                    b = 0
                else:
                    h = (global_lane // HEADDIM) % HEADS
                    b = global_lane // (HEADDIM * HEADS)
                for n in T.serial(STATE):
                    h_state[n] = T.cast(h0[b, h, p, n], accum_dtype)
                    h_snap[b, 0, h, p, n] = T.cast(h_state[n], h_snap_dtype)
                for block in T.serial(BLOCKS):
                    for step in T.serial(BLOCK):
                        t = block * BLOCK + step
                        if t < SEQ:
                            A_val = T.cast(A[b, t, h], accum_dtype)
                            dt_val = T.cast(dt[b, t, h], accum_dtype)
                            decay = T.alloc_var(T.float32, init=T.exp(A_val * dt_val))
                            x_val = T.cast(x[b, t, h, p], accum_dtype)
                            for n in T.serial(STATE):
                                h_state[n] = decay * h_state[n] + x_val * T.cast(
                                    B[b, t, h, n],
                                    accum_dtype,
                                )
                    for n in T.serial(STATE):
                        h_snap[b, block + 1, h, p, n] = T.cast(
                            h_state[n],
                            h_snap_dtype,
                        )

    artifact = dispatch_lower(snapshots, target="metal", return_msl=True)
    if hasattr(artifact, "_tilelang_engine_target"):
        raise MSLDispatchUnsupported("Mamba3 Path C requires TileLang MSL extraction metadata")
    lowering = cast(_msl_transform.TileLangMSLLowering, artifact)
    input_names = [name for name in lowering.buffer_param_names if name != "h_snap"]
    if set(input_names) != {"x", "B", "A", "dt", "h0"}:
        raise MSLDispatchUnsupported(
            "unexpected Mamba3 Path C bwd snapshot buffer signature: "
            + ", ".join(lowering.buffer_param_names)
        )
    import tilelang

    kernel = tilelang.compile(
        snapshots,
        target=_msl_transform._as_metal_target("metal"),
        execution_backend="tvm_ffi",
        out_idx=[5],
    )
    return kernel, lowering


def _bwd_simd_p_reduction_supported(
    *, batch: int, heads: int, headdim: int
) -> bool:
    """Return whether P-axis grads map to TileLang split thread-allreduce."""

    lanes = batch * heads * headdim
    threads = _bwd_threads_for(lanes, headdim)
    return (
        headdim > 0
        and (headdim <= 32 or headdim % 32 == 0)
        and threads % headdim == 0
        and lanes % threads == 0
    )


@lru_cache(maxsize=128)
def _bwd_simd_reduce_kernel_for(
    BATCH: int,
    SEQ: int,
    HEADS: int,
    HEADDIM: int,
    STATE: int,
    dy_dtype: str = "float32",
    x_dtype: str = "float32",
    B_dtype: str = "float32",
    C_dtype: str = "float32",
    z_dtype: str = "float32",
    A_dtype: str = "float32",
    dt_dtype: str = "float32",
    D_dtype: str = "float32",
    dx_dtype: str = "float32",
    dz_dtype: str = "float32",
    dB_dtype: str = "float32",
    dC_dtype: str = "float32",
    dA_dtype: str = "float32",
    ddt_dtype: str = "float32",
    dD_dtype: str = "float32",
    dh0_dtype: str = "float32",
    *,
    return_msl: bool = False,
) -> tuple[Any, _msl_transform.TileLangMSLLowering]:
    """Build a Path C bwd kernel that reduces P through TileLang allreduce IR.

    The kernel intentionally emits ``T.thread_allreduce_sum`` rather than Metal
    ``simd_sum``. TileLang lowering chooses same-simdgroup or cross-simdgroup
    code from the reduce index, so P=32, P=64, and similar aligned P-axis
    reductions share the same IR path without global partial buffers.
    """

    if not _bwd_simd_p_reduction_supported(
        batch=BATCH,
        heads=HEADS,
        headdim=HEADDIM,
    ):
        raise MSLDispatchUnsupported(
            "Mamba3 Path C P-reduction requires HEADDIM<=32 or a multiple of "
            "32, with threadgroups aligned to full P rows"
        )
    if SEQ > _BWD_SNAPSHOT_BLOCK:
        raise MSLDispatchUnsupported(
            "direct Mamba3 Path C SIMD bwd is only legal for single-step "
            "sequences; long sequences must consume explicit state snapshots"
        )

    import tilelang.language as T

    LANES = BATCH * HEADS * HEADDIM
    THREADS = _bwd_threads_for(LANES, HEADDIM)
    accum_dtype = "float32"

    @T.prim_func
    def bwd_simd(
        dy: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), dy_dtype),
        x: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), x_dtype),
        B: T.Tensor((BATCH, SEQ, HEADS, STATE), B_dtype),
        C: T.Tensor((BATCH, SEQ, HEADS, STATE), C_dtype),
        z: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), z_dtype),
        A: T.Tensor((BATCH, SEQ, HEADS), A_dtype),
        dt: T.Tensor((BATCH, SEQ, HEADS), dt_dtype),
        D: T.Tensor((HEADS,), D_dtype),
        h0: T.Tensor((BATCH, HEADS, HEADDIM, STATE), h0_dtype),
        dx: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), dx_dtype),
        dz: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), dz_dtype),
        dB: T.Tensor((BATCH, SEQ, HEADS, STATE), dB_dtype),
        dC: T.Tensor((BATCH, SEQ, HEADS, STATE), dC_dtype),
        dA: T.Tensor((BATCH, SEQ, HEADS), dA_dtype),
        ddt: T.Tensor((BATCH, SEQ, HEADS), ddt_dtype),
        dD_batch: T.Tensor((BATCH, HEADS), dD_dtype),
        dh0: T.Tensor((BATCH, HEADS, HEADDIM, STATE), dh0_dtype),
    ):
        with T.Kernel(T.ceildiv(LANES, THREADS), threads=THREADS) as _bx:
            tid = T.get_thread_binding(0)
            global_lane = T.call_intrin("int32", "tir.metal.thread_position_in_grid_x")
            h_state = T.alloc_local((STATE,), accum_dtype)
            if LANES % THREADS == 0 or global_lane < LANES:
                p = global_lane % HEADDIM
                reduce_lane = tid % HEADDIM
                if BATCH == 1:
                    h = global_lane // HEADDIM
                    b = 0
                else:
                    h = (global_lane // HEADDIM) % HEADS
                    b = global_lane // (HEADDIM * HEADS)

                for n in T.serial(STATE):
                    h_state[n] = T.cast(h0[b, h, p, n], accum_dtype)
                for t in T.serial(SEQ):
                    A_val = T.cast(A[b, t, h], accum_dtype)
                    dt_val = T.cast(dt[b, t, h], accum_dtype)
                    decay = T.alloc_var(T.float32, init=T.exp(A_val * dt_val))
                    x_val = T.cast(x[b, t, h, p], accum_dtype)
                    for n in T.serial(STATE):
                        h_state[n] = decay * h_state[n] + x_val * T.cast(
                            B[b, t, h, n],
                            accum_dtype,
                        )

                for n in T.serial(STATE):
                    dh[n] = 0.0
                dD_acc = T.alloc_var(T.float32, init=0.0)
                for r in T.serial(SEQ):
                    t = SEQ - 1 - r
                    A_val = T.cast(A[b, t, h], accum_dtype)
                    dt_val = T.cast(dt[b, t, h], accum_dtype)
                    decay = T.alloc_var(T.float32, init=T.exp(A_val * dt_val))
                    inv_decay = T.alloc_var(T.float32, init=1.0 / decay)
                    x_val = T.cast(x[b, t, h, p], accum_dtype)
                    z_val = T.cast(z[b, t, h, p], accum_dtype)
                    dY = T.cast(dy[b, t, h, p], accum_dtype)

                    y_state = T.alloc_var(T.float32, init=0.0)
                    for n in T.serial(STATE):
                        y_state += h_state[n] * T.cast(C[b, t, h, n], accum_dtype)
                    D_h = T.cast(D[h], accum_dtype)
                    y_skipped = y_state + D_h * x_val
                    sig_z = 1.0 / (1.0 + T.exp(-z_val))
                    silu_z = z_val * sig_z
                    silu_dz = sig_z * (1.0 + z_val * (1.0 - sig_z))

                    d_silu = dY * y_skipped
                    d_y_skipped = dY * silu_z

                    dz[b, t, h, p] = T.cast(d_silu * silu_dz, dz_dtype)
                    dD_acc += d_y_skipped * x_val

                    dx_inp = T.alloc_var(T.float32, init=0.0)
                    d_decay = T.alloc_var(T.float32, init=0.0)
                    if t == 0:
                        for n in T.serial(STATE):
                            C_val = T.cast(C[b, t, h, n], accum_dtype)
                            B_val = T.cast(B[b, t, h, n], accum_dtype)
                            dh_n = dh[n] + d_y_skipped * C_val
                            dC_sum = T.alloc_local((1,), accum_dtype)
                            dB_sum = T.alloc_local((1,), accum_dtype)
                            T.thread_allreduce_sum(
                                d_y_skipped * h_state[n], dC_sum[0], reduce_lane
                            )
                            T.thread_allreduce_sum(dh_n * x_val, dB_sum[0], reduce_lane)
                            if reduce_lane == 0:
                                dC[b, t, h, n] = T.cast(dC_sum[0], dC_dtype)
                                dB[b, t, h, n] = T.cast(dB_sum[0], dB_dtype)
                            dx_inp += dh_n * B_val
                            d_decay += dh_n * T.cast(h0[b, h, p, n], accum_dtype)
                            dh[n] = dh_n * decay
                    else:
                        for n in T.serial(STATE):
                            C_val = T.cast(C[b, t, h, n], accum_dtype)
                            B_val = T.cast(B[b, t, h, n], accum_dtype)
                            dh_n = dh[n] + d_y_skipped * C_val
                            dC_sum = T.alloc_local((1,), accum_dtype)
                            dB_sum = T.alloc_local((1,), accum_dtype)
                            T.thread_allreduce_sum(
                                d_y_skipped * h_state[n], dC_sum[0], reduce_lane
                            )
                            T.thread_allreduce_sum(dh_n * x_val, dB_sum[0], reduce_lane)
                            if reduce_lane == 0:
                                dC[b, t, h, n] = T.cast(dC_sum[0], dC_dtype)
                                dB[b, t, h, n] = T.cast(dB_sum[0], dB_dtype)
                            dx_inp += dh_n * B_val
                            h_prev = (h_state[n] - x_val * B_val) * inv_decay
                            d_decay += dh_n * h_prev
                            h_state[n] = h_prev
                            dh[n] = dh_n * decay
                    dx_skip = d_y_skipped * D_h
                    dx[b, t, h, p] = T.cast(dx_skip + dx_inp, dx_dtype)

                    d_logdecay = d_decay * decay
                    dA_lane = d_logdecay * dt_val
                    ddt_lane = d_logdecay * A_val
                    dA_sum = T.alloc_local((1,), accum_dtype)
                    ddt_sum = T.alloc_local((1,), accum_dtype)
                    T.thread_allreduce_sum(dA_lane, dA_sum[0], reduce_lane)
                    T.thread_allreduce_sum(ddt_lane, ddt_sum[0], reduce_lane)
                    if reduce_lane == 0:
                        dA[b, t, h] = T.cast(dA_sum[0], dA_dtype)
                        ddt[b, t, h] = T.cast(ddt_sum[0], ddt_dtype)

                for n in T.serial(STATE):
                    dh0[b, h, p, n] = T.cast(dh[n], dh0_dtype)
                dD_sum = T.alloc_local((1,), accum_dtype)
                T.thread_allreduce_sum(dD_acc, dD_sum[0], reduce_lane)
                if reduce_lane == 0:
                    dD_batch[b, h] = T.cast(dD_sum[0], dD_dtype)

    artifact = dispatch_lower(bwd_simd, target="metal", return_msl=True)
    if hasattr(artifact, "_tilelang_engine_target"):
        raise MSLDispatchUnsupported("Mamba3 Path C requires TileLang MSL extraction metadata")
    lowering = cast(_msl_transform.TileLangMSLLowering, artifact)
    input_names = [
        name for name in lowering.buffer_param_names if name not in _BWD_SIMD_OUTPUT_NAMES
    ]
    if set(input_names) != {"dy", "x", "B", "C", "z", "A", "dt", "D", "h0"}:
        raise MSLDispatchUnsupported(
            "unexpected Mamba3 Path C simd bwd buffer signature: "
            + ", ".join(lowering.buffer_param_names)
        )
    import tilelang

    kernel = tilelang.compile(
        bwd_simd,
        target=_msl_transform._as_metal_target("metal"),
        execution_backend="tvm_ffi",
        out_idx=list(_BWD_SIMD_OUTPUT_IDX),
    )
    return kernel, lowering


@lru_cache(maxsize=128)
def _bwd_simd_reduce_kernel_for_state_snapshots(
    BATCH: int,
    SEQ: int,
    HEADS: int,
    HEADDIM: int,
    STATE: int,
    dy_dtype: str = "float32",
    x_dtype: str = "float32",
    B_dtype: str = "float32",
    C_dtype: str = "float32",
    z_dtype: str = "float32",
    A_dtype: str = "float32",
    dt_dtype: str = "float32",
    D_dtype: str = "float32",
    dx_dtype: str = "float32",
    dz_dtype: str = "float32",
    dB_dtype: str = "float32",
    dC_dtype: str = "float32",
    dA_dtype: str = "float32",
    ddt_dtype: str = "float32",
    dD_dtype: str = "float32",
    dh0_dtype: str = "float32",
) -> tuple[Any, _msl_transform.TileLangMSLLowering]:
    """Build a P-reduced bwd kernel that consumes stable state snapshots."""

    if not _bwd_simd_p_reduction_supported(
        batch=BATCH,
        heads=HEADS,
        headdim=HEADDIM,
    ):
        raise MSLDispatchUnsupported(
            "Mamba3 Path C snapshot SIMD P-reduction requires HEADDIM<=32 or "
            "a multiple of 32, with threadgroups aligned to full P rows"
        )

    import tilelang.language as T

    LANES = BATCH * HEADS * HEADDIM
    THREADS = _bwd_threads_for(LANES, HEADDIM)
    scan_plan = _bwd_scan_plan_for(
        batch=BATCH,
        seq=SEQ,
        heads=HEADS,
        headdim=HEADDIM,
        state=STATE,
    )
    BLOCK = scan_plan.snapshot_plan.chunk_size
    BLOCKS = (SEQ + BLOCK - 1) // BLOCK
    accum_dtype = "float32"

    @T.prim_func
    def bwd_snap_simd(
        dy: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), dy_dtype),
        x: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), x_dtype),
        B: T.Tensor((BATCH, SEQ, HEADS, STATE), B_dtype),
        C: T.Tensor((BATCH, SEQ, HEADS, STATE), C_dtype),
        z: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), z_dtype),
        A: T.Tensor((BATCH, SEQ, HEADS), A_dtype),
        dt: T.Tensor((BATCH, SEQ, HEADS), dt_dtype),
        D: T.Tensor((HEADS,), D_dtype),
        h_snap: T.Tensor((BATCH, BLOCKS + 1, HEADS, HEADDIM, STATE), "float32"),
        dx: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), dx_dtype),
        dz: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), dz_dtype),
        dB: T.Tensor((BATCH, SEQ, HEADS, STATE), dB_dtype),
        dC: T.Tensor((BATCH, SEQ, HEADS, STATE), dC_dtype),
        dA: T.Tensor((BATCH, SEQ, HEADS), dA_dtype),
        ddt: T.Tensor((BATCH, SEQ, HEADS), ddt_dtype),
        dD_batch: T.Tensor((BATCH, HEADS), dD_dtype),
        dh0: T.Tensor((BATCH, HEADS, HEADDIM, STATE), dh0_dtype),
    ):
        with T.Kernel(T.ceildiv(LANES, THREADS), threads=THREADS) as _bx:
            tid = T.get_thread_binding(0)
            global_lane = T.call_intrin("int32", "tir.metal.thread_position_in_grid_x")
            h_state = T.alloc_local((STATE,), accum_dtype)
            dh = T.alloc_local((STATE,), accum_dtype)
            if LANES % THREADS == 0 or global_lane < LANES:
                p = global_lane % HEADDIM
                reduce_lane = tid % HEADDIM
                if BATCH == 1:
                    h = global_lane // HEADDIM
                    b = 0
                else:
                    h = (global_lane // HEADDIM) % HEADS
                    b = global_lane // (HEADDIM * HEADS)

                for n in T.serial(STATE):
                    dh[n] = 0.0
                dD_acc = T.alloc_var(T.float32, init=0.0)
                D_h = T.cast(D[h], accum_dtype)

                for rb in T.serial(BLOCKS):
                    block = BLOCKS - 1 - rb
                    block_start = block * BLOCK
                    block_end = (block + 1) * BLOCK
                    for n in T.serial(STATE):
                        h_state[n] = h_snap[b, block + 1, h, p, n]

                    for step in T.serial(BLOCK):
                        t = block_end - 1 - step
                        if t < SEQ and t >= block_start:
                            A_val = T.cast(A[b, t, h], accum_dtype)
                            dt_val = T.cast(dt[b, t, h], accum_dtype)
                            decay = T.alloc_var(T.float32, init=T.exp(A_val * dt_val))
                            x_val = T.cast(x[b, t, h, p], accum_dtype)
                            z_val = T.cast(z[b, t, h, p], accum_dtype)
                            dY = T.cast(dy[b, t, h, p], accum_dtype)

                            y_state = T.alloc_var(T.float32, init=0.0)
                            for n in T.serial(STATE):
                                y_state += h_state[n] * T.cast(
                                    C[b, t, h, n],
                                    accum_dtype,
                                )
                            y_skipped = y_state + D_h * x_val
                            sig_z = 1.0 / (1.0 + T.exp(-z_val))
                            silu_z = z_val * sig_z
                            silu_dz = sig_z * (1.0 + z_val * (1.0 - sig_z))

                            d_silu = dY * y_skipped
                            d_y_skipped = dY * silu_z

                            dz[b, t, h, p] = T.cast(d_silu * silu_dz, dz_dtype)
                            dD_acc += d_y_skipped * x_val

                            dx_inp = T.alloc_var(T.float32, init=0.0)
                            d_decay = T.alloc_var(T.float32, init=0.0)
                            for n in T.serial(STATE):
                                C_val = T.cast(C[b, t, h, n], accum_dtype)
                                B_val = T.cast(B[b, t, h, n], accum_dtype)
                                h_prev = h_snap[b, block, h, p, n]
                                dh_n = dh[n] + d_y_skipped * C_val
                                dC_sum = T.alloc_local((1,), accum_dtype)
                                dB_sum = T.alloc_local((1,), accum_dtype)
                                T.thread_allreduce_sum(
                                    d_y_skipped * h_state[n],
                                    dC_sum[0],
                                    reduce_lane,
                                )
                                T.thread_allreduce_sum(dh_n * x_val, dB_sum[0], reduce_lane)
                                if reduce_lane == 0:
                                    dC[b, t, h, n] = T.cast(dC_sum[0], dC_dtype)
                                    dB[b, t, h, n] = T.cast(dB_sum[0], dB_dtype)
                                dx_inp += dh_n * B_val
                                d_decay += dh_n * h_prev
                                dh[n] = dh_n * decay
                                h_state[n] = h_prev

                            dx_skip = d_y_skipped * D_h
                            dx[b, t, h, p] = T.cast(dx_skip + dx_inp, dx_dtype)

                            d_logdecay = d_decay * decay
                            dA_lane = d_logdecay * dt_val
                            ddt_lane = d_logdecay * A_val
                            dA_sum = T.alloc_local((1,), accum_dtype)
                            ddt_sum = T.alloc_local((1,), accum_dtype)
                            T.thread_allreduce_sum(dA_lane, dA_sum[0], reduce_lane)
                            T.thread_allreduce_sum(ddt_lane, ddt_sum[0], reduce_lane)
                            if reduce_lane == 0:
                                dA[b, t, h] = T.cast(dA_sum[0], dA_dtype)
                                ddt[b, t, h] = T.cast(ddt_sum[0], ddt_dtype)

                for n in T.serial(STATE):
                    dh0[b, h, p, n] = T.cast(dh[n], dh0_dtype)
                dD_sum = T.alloc_local((1,), accum_dtype)
                T.thread_allreduce_sum(dD_acc, dD_sum[0], reduce_lane)
                if reduce_lane == 0:
                    dD_batch[b, h] = T.cast(dD_sum[0], dD_dtype)

    artifact = dispatch_lower(bwd_snap_simd, target="metal", return_msl=True)
    if hasattr(artifact, "_tilelang_engine_target"):
        raise MSLDispatchUnsupported("Mamba3 Path C requires TileLang MSL extraction metadata")
    lowering = cast(_msl_transform.TileLangMSLLowering, artifact)
    input_names = [
        name for name in lowering.buffer_param_names if name not in _BWD_SIMD_OUTPUT_NAMES
    ]
    if set(input_names) != {"dy", "x", "B", "C", "z", "A", "dt", "D", "h_snap"}:
        raise MSLDispatchUnsupported(
            "unexpected Mamba3 Path C snapshot simd bwd buffer signature: "
            + ", ".join(lowering.buffer_param_names)
        )
    import tilelang

    kernel = tilelang.compile(
        bwd_snap_simd,
        target=_msl_transform._as_metal_target("metal"),
        execution_backend="tvm_ffi",
        out_idx=list(_BWD_SIMD_OUTPUT_IDX),
    )
    return kernel, lowering


@lru_cache(maxsize=128)
def _bwd_partial_kernel_for_state_snapshots(
    BATCH: int,
    SEQ: int,
    HEADS: int,
    HEADDIM: int,
    STATE: int,
    dy_dtype: str = "float32",
    x_dtype: str = "float32",
    B_dtype: str = "float32",
    C_dtype: str = "float32",
    z_dtype: str = "float32",
    A_dtype: str = "float32",
    dt_dtype: str = "float32",
    D_dtype: str = "float32",
    h0_dtype: str = "float32",
    dx_dtype: str = "float32",
    dz_dtype: str = "float32",
    dB_dtype: str = "float32",
    dC_dtype: str = "float32",
    dA_dtype: str = "float32",
    ddt_dtype: str = "float32",
    dD_dtype: str = "float32",
    dh0_dtype: str = "float32",
    h_snap_dtype: str = "float32",
) -> tuple[Any, _msl_transform.TileLangMSLLowering]:
    """Build the generic hot-loop bwd route: scan partials, reduce outside.

    The final-gradient SIMD route is useful as a semantic codegen test, but on
    recurrent backward kernels it puts P-axis allreduce barriers inside the
    ``T * STATE`` loop. This route keeps the generated TileLang scan generic:
    one thread owns one ``(b, h, p)`` lane, emits partial gradients for
    reductions over ``p``, and lets MLX run the tensor reductions outside the
    recurrent hot loop.
    """

    if _BWD_SNAPSHOT_BLOCK != 1:
        raise MSLDispatchUnsupported(
            "Mamba3 partial bwd expects per-step state snapshots"
        )

    import tilelang.language as T

    LANES = BATCH * HEADS * HEADDIM
    THREADS = _threads_for(LANES)
    accum_dtype = "float32"

    @T.prim_func
    def bwd_partial(
        dy: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), dy_dtype),
        x: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), x_dtype),
        B: T.Tensor((BATCH, SEQ, HEADS, STATE), B_dtype),
        C: T.Tensor((BATCH, SEQ, HEADS, STATE), C_dtype),
        z: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), z_dtype),
        A: T.Tensor((BATCH, SEQ, HEADS), A_dtype),
        dt: T.Tensor((BATCH, SEQ, HEADS), dt_dtype),
        D: T.Tensor((HEADS,), D_dtype),
        h_snap: T.Tensor((BATCH, SEQ + 1, HEADS, HEADDIM, STATE), h_snap_dtype),
        dx: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), dx_dtype),
        dz: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), dz_dtype),
        dB_partial: T.Tensor((BATCH, SEQ, HEADS, STATE, HEADDIM), dB_dtype),
        dC_partial: T.Tensor((BATCH, SEQ, HEADS, STATE, HEADDIM), dC_dtype),
        dA_partial: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), dA_dtype),
        ddt_partial: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), ddt_dtype),
        dD_partial: T.Tensor((BATCH, HEADS, HEADDIM), dD_dtype),
        dh0: T.Tensor((BATCH, HEADS, HEADDIM, STATE), dh0_dtype),
    ):
        with T.Kernel(T.ceildiv(LANES, THREADS), threads=THREADS) as _bx:
            global_lane = T.call_intrin("int32", "tir.metal.thread_position_in_grid_x")
            h_state = T.alloc_local((STATE,), accum_dtype)
            dh = T.alloc_local((STATE,), accum_dtype)
            if LANES % THREADS == 0 or global_lane < LANES:
                p = T.alloc_var(T.int32, init=global_lane % HEADDIM)
                if BATCH == 1:
                    h = global_lane // HEADDIM
                    b = 0
                else:
                    h = (global_lane // HEADDIM) % HEADS
                    b = global_lane // (HEADDIM * HEADS)

                for n in T.serial(STATE):
                    h_state[n] = T.cast(
                        h_snap[b, SEQ, h, p, n],
                        accum_dtype,
                    )
                    dh[n] = 0.0
                dD_acc = T.alloc_var(T.float32, init=0.0)
                D_h = T.cast(D[h], accum_dtype)

                for rt in T.serial(SEQ):
                    t = SEQ - 1 - rt
                    A_val = T.cast(A[b, t, h], accum_dtype)
                    dt_val = T.cast(dt[b, t, h], accum_dtype)
                    decay = T.alloc_var(T.float32, init=T.exp(A_val * dt_val))
                    x_val = T.cast(x[b, t, h, p], accum_dtype)
                    z_val = T.cast(z[b, t, h, p], accum_dtype)
                    dY = T.cast(dy[b, t, h, p], accum_dtype)

                    y_state = T.alloc_var(T.float32, init=0.0)
                    for n in T.serial(STATE):
                        y_state += h_state[n] * T.cast(C[b, t, h, n], accum_dtype)
                    y_skipped = y_state + D_h * x_val
                    sig_z = 1.0 / (1.0 + T.exp(-z_val))
                    silu_z = z_val * sig_z
                    silu_dz = sig_z * (1.0 + z_val * (1.0 - sig_z))
                    d_silu = dY * y_skipped
                    d_y_skipped = dY * silu_z

                    dz[b, t, h, p] = T.cast(d_silu * silu_dz, dz_dtype)
                    dD_acc += d_y_skipped * x_val

                    dx_inp = T.alloc_var(T.float32, init=0.0)
                    d_decay = T.alloc_var(T.float32, init=0.0)
                    for n in T.serial(STATE):
                        C_val = T.cast(C[b, t, h, n], accum_dtype)
                        B_val = T.cast(B[b, t, h, n], accum_dtype)
                        h_prev = T.cast(h_snap[b, t, h, p, n], accum_dtype)
                        dh_n = dh[n] + d_y_skipped * C_val
                        dC_partial[b, t, h, n, p] = T.cast(
                            d_y_skipped * h_state[n],
                            dC_dtype,
                        )
                        dB_partial[b, t, h, n, p] = T.cast(
                            dh_n * x_val,
                            dB_dtype,
                        )
                        dx_inp += dh_n * B_val
                        d_decay += dh_n * h_prev
                        dh[n] = dh_n * decay
                        h_state[n] = h_prev

                    dx_skip = d_y_skipped * D_h
                    dx[b, t, h, p] = T.cast(dx_skip + dx_inp, dx_dtype)

                    d_logdecay = d_decay * decay
                    dA_partial[b, t, h, p] = T.cast(d_logdecay * dt_val, dA_dtype)
                    ddt_partial[b, t, h, p] = T.cast(d_logdecay * A_val, ddt_dtype)

                for n in T.serial(STATE):
                    dh0[b, h, p, n] = T.cast(dh[n], dh0_dtype)
                dD_partial[b, h, p] = T.cast(dD_acc, dD_dtype)

    artifact = dispatch_lower(bwd_partial, target="metal", return_msl=True)
    if hasattr(artifact, "_tilelang_engine_target"):
        raise MSLDispatchUnsupported("Mamba3 Path C requires TileLang MSL extraction metadata")
    lowering = cast(_msl_transform.TileLangMSLLowering, artifact)
    input_names = [
        name for name in lowering.buffer_param_names if name not in _BWD_PARTIAL_OUTPUT_NAMES
    ]
    if set(input_names) != {"dy", "x", "B", "C", "z", "A", "dt", "D", "h_snap"}:
        raise MSLDispatchUnsupported(
            "unexpected Mamba3 Path C partial bwd buffer signature: "
            + ", ".join(lowering.buffer_param_names)
        )
    import tilelang

    kernel = tilelang.compile(
        bwd_partial,
        target=_msl_transform._as_metal_target("metal"),
        execution_backend="tvm_ffi",
        out_idx=list(_BWD_PARTIAL_OUTPUT_IDX),
    )
    return kernel, lowering


@lru_cache(maxsize=128)
def _bwd_scratch_partial_kernel_for(
    BATCH: int,
    SEQ: int,
    HEADS: int,
    HEADDIM: int,
    STATE: int,
    dy_dtype: str = "float32",
    x_dtype: str = "float32",
    B_dtype: str = "float32",
    C_dtype: str = "float32",
    z_dtype: str = "float32",
    A_dtype: str = "float32",
    dt_dtype: str = "float32",
    D_dtype: str = "float32",
    h0_dtype: str = "float32",
    dx_dtype: str = "float32",
    dz_dtype: str = "float32",
    dB_dtype: str = "float32",
    dC_dtype: str = "float32",
    dA_dtype: str = "float32",
    ddt_dtype: str = "float32",
    dD_dtype: str = "float32",
    dh0_dtype: str = "float32",
    h_steps_scratch_dtype: str = "float32",
) -> tuple[Any, _msl_transform.TileLangMSLLowering]:
    """Build a single-kernel bwd route that owns its state scratch buffer."""

    import tilelang.language as T

    LANES = BATCH * HEADS * HEADDIM
    THREADS = _threads_for(LANES)
    accum_dtype = "float32"

    @T.prim_func
    def bwd_scratch(
        dy: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), dy_dtype),
        x: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), x_dtype),
        B: T.Tensor((BATCH, SEQ, HEADS, STATE), B_dtype),
        C: T.Tensor((BATCH, SEQ, HEADS, STATE), C_dtype),
        z: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), z_dtype),
        A: T.Tensor((BATCH, SEQ, HEADS), A_dtype),
        dt: T.Tensor((BATCH, SEQ, HEADS), dt_dtype),
        D: T.Tensor((HEADS,), D_dtype),
        h0: T.Tensor((BATCH, HEADS, HEADDIM, STATE), h0_dtype),
        dx: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), dx_dtype),
        dz: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), dz_dtype),
        dB_partial: T.Tensor((BATCH, SEQ, HEADS, STATE, HEADDIM), dB_dtype),
        dC_partial: T.Tensor((BATCH, SEQ, HEADS, STATE, HEADDIM), dC_dtype),
        dA_partial: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), dA_dtype),
        ddt_partial: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), ddt_dtype),
        dD_partial: T.Tensor((BATCH, HEADS, HEADDIM), dD_dtype),
        dh0: T.Tensor((BATCH, HEADS, HEADDIM, STATE), dh0_dtype),
    ):
        h_steps_scratch = T.alloc_global(
            (BATCH, SEQ + 1, HEADS, HEADDIM, STATE),
            h_steps_scratch_dtype,
        )
        with T.Kernel(T.ceildiv(LANES, THREADS), threads=THREADS) as _bx:
            global_lane = T.call_intrin("int32", "tir.metal.thread_position_in_grid_x")
            h_state = T.alloc_local((STATE,), accum_dtype)
            if LANES % THREADS == 0 or global_lane < LANES:
                p = T.alloc_var(T.int32, init=global_lane % HEADDIM)
                if BATCH == 1:
                    h = global_lane // HEADDIM
                    b = 0
                else:
                    h = (global_lane // HEADDIM) % HEADS
                    b = global_lane // (HEADDIM * HEADS)

                for n in T.serial(STATE):
                    h_state[n] = T.cast(h0[b, h, p, n], accum_dtype)
                    h_steps_scratch[b, 0, h, p, n] = T.cast(
                        h_state[n],
                        h_steps_scratch_dtype,
                    )
                for t in T.serial(SEQ):
                    A_val = T.cast(A[b, t, h], accum_dtype)
                    dt_val = T.cast(dt[b, t, h], accum_dtype)
                    decay = T.alloc_var(T.float32, init=T.exp(A_val * dt_val))
                    x_val = T.cast(x[b, t, h, p], accum_dtype)
                    for n in T.serial(STATE):
                        B_val = T.cast(B[b, t, h, n], accum_dtype)
                        h_state[n] = decay * h_state[n] + x_val * B_val
                        h_steps_scratch[b, t + 1, h, p, n] = T.cast(
                            h_state[n],
                            h_steps_scratch_dtype,
                        )

                dh = T.alloc_local((STATE,), accum_dtype)
                for n in T.serial(STATE):
                    h_state[n] = T.cast(
                        h_steps_scratch[b, SEQ, h, p, n],
                        accum_dtype,
                    )
                    dh[n] = 0.0
                dD_acc = T.alloc_var(T.float32, init=0.0)
                D_h = T.cast(D[h], accum_dtype)

                for rt in T.serial(SEQ):
                    t = SEQ - 1 - rt
                    A_val = T.cast(A[b, t, h], accum_dtype)
                    dt_val = T.cast(dt[b, t, h], accum_dtype)
                    decay = T.alloc_var(T.float32, init=T.exp(A_val * dt_val))
                    x_val = T.cast(x[b, t, h, p], accum_dtype)
                    z_val = T.cast(z[b, t, h, p], accum_dtype)
                    dY = T.cast(dy[b, t, h, p], accum_dtype)

                    y_state = T.alloc_var(T.float32, init=0.0)
                    for n in T.serial(STATE):
                        y_state += h_state[n] * T.cast(C[b, t, h, n], accum_dtype)
                    y_skipped = y_state + D_h * x_val
                    sig_z = 1.0 / (1.0 + T.exp(-z_val))
                    silu_z = z_val * sig_z
                    silu_dz = sig_z * (1.0 + z_val * (1.0 - sig_z))
                    d_silu = dY * y_skipped
                    d_y_skipped = dY * silu_z

                    dz[b, t, h, p] = T.cast(d_silu * silu_dz, dz_dtype)
                    dD_acc += d_y_skipped * x_val

                    dx_inp = T.alloc_var(T.float32, init=0.0)
                    d_decay = T.alloc_var(T.float32, init=0.0)
                    for n in T.serial(STATE):
                        C_val = T.cast(C[b, t, h, n], accum_dtype)
                        B_val = T.cast(B[b, t, h, n], accum_dtype)
                        h_prev = T.cast(
                            h_steps_scratch[b, t, h, p, n],
                            accum_dtype,
                        )
                        dh_n = dh[n] + d_y_skipped * C_val
                        dC_partial[b, t, h, n, p] = T.cast(
                            d_y_skipped * h_state[n],
                            dC_dtype,
                        )
                        dB_partial[b, t, h, n, p] = T.cast(
                            dh_n * x_val,
                            dB_dtype,
                        )
                        dx_inp += dh_n * B_val
                        d_decay += dh_n * h_prev
                        dh[n] = dh_n * decay
                        h_state[n] = h_prev

                    dx_skip = d_y_skipped * D_h
                    dx[b, t, h, p] = T.cast(dx_skip + dx_inp, dx_dtype)

                    d_logdecay = d_decay * decay
                    dA_partial[b, t, h, p] = T.cast(d_logdecay * dt_val, dA_dtype)
                    ddt_partial[b, t, h, p] = T.cast(d_logdecay * A_val, ddt_dtype)

                for n in T.serial(STATE):
                    dh0[b, h, p, n] = T.cast(dh[n], dh0_dtype)
                dD_partial[b, h, p] = T.cast(dD_acc, dD_dtype)

    artifact = dispatch_lower(bwd_scratch, target="metal", return_msl=True)
    if hasattr(artifact, "_tilelang_engine_target"):
        raise MSLDispatchUnsupported("Mamba3 Path C requires TileLang MSL extraction metadata")
    lowering = cast(_msl_transform.TileLangMSLLowering, artifact)
    input_names = [
        name
        for name in lowering.buffer_param_names
        if name not in _BWD_SCRATCH_OUTPUT_NAMES
        and name not in _BWD_SCRATCH_WORKSPACE_NAMES
    ]
    if set(input_names) != {"dy", "x", "B", "C", "z", "A", "dt", "D", "h0"}:
        raise MSLDispatchUnsupported(
            "unexpected Mamba3 Path C scratch bwd buffer signature: "
            + ", ".join(lowering.buffer_param_names)
        )
    import tilelang

    kernel = tilelang.compile(
        bwd_scratch,
        target=_msl_transform._as_metal_target("metal"),
        execution_backend="tvm_ffi",
        out_idx=list(_BWD_SCRATCH_OUTPUT_IDX),
    )
    return kernel, lowering


@lru_cache(maxsize=128)
def _bwd_partial_reduce_kernel_for(
    BATCH: int,
    SEQ: int,
    HEADS: int,
    HEADDIM: int,
    STATE: int,
    dB_dtype: str = "float32",
    dC_dtype: str = "float32",
    dA_dtype: str = "float32",
    ddt_dtype: str = "float32",
    dD_dtype: str = "float32",
    out_dB_dtype: str = "float32",
    out_dC_dtype: str = "float32",
    out_dA_dtype: str = "float32",
    out_ddt_dtype: str = "float32",
    out_dD_dtype: str = "float32",
) -> tuple[Any, _msl_transform.TileLangMSLLowering]:
    """Build a standalone reducer for bwd partial gradients over P."""

    import tilelang.language as T

    DBC_LANES = BATCH * SEQ * HEADS * STATE
    ADT_LANES = BATCH * SEQ * HEADS
    D_LANES = HEADS
    LANES = max(DBC_LANES, ADT_LANES, D_LANES)
    THREADS = _threads_for(LANES)

    @T.prim_func
    def bwd_partial_reduce(
        dB_partial: T.Tensor((BATCH, SEQ, HEADS, STATE, HEADDIM), dB_dtype),
        dC_partial: T.Tensor((BATCH, SEQ, HEADS, STATE, HEADDIM), dC_dtype),
        dA_partial: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), dA_dtype),
        ddt_partial: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), ddt_dtype),
        dD_partial: T.Tensor((BATCH, HEADS, HEADDIM), dD_dtype),
        dB: T.Tensor((BATCH, SEQ, HEADS, STATE), out_dB_dtype),
        dC: T.Tensor((BATCH, SEQ, HEADS, STATE), out_dC_dtype),
        dA: T.Tensor((BATCH, SEQ, HEADS), out_dA_dtype),
        ddt: T.Tensor((BATCH, SEQ, HEADS), out_ddt_dtype),
        dD: T.Tensor((HEADS,), out_dD_dtype),
    ):
        with T.Kernel(T.ceildiv(LANES, THREADS), threads=THREADS) as _bx:
            global_lane = T.call_intrin("int32", "tir.metal.thread_position_in_grid_x")
            if global_lane < DBC_LANES:
                n = global_lane % STATE
                h = (global_lane // STATE) % HEADS
                t = (global_lane // (STATE * HEADS)) % SEQ
                b = global_lane // (STATE * HEADS * SEQ)
                dB_sum = T.alloc_var(T.float32, init=0.0)
                dC_sum = T.alloc_var(T.float32, init=0.0)
                for p in T.serial(HEADDIM):
                    dB_sum += T.cast(dB_partial[b, t, h, n, p], "float32")
                    dC_sum += T.cast(dC_partial[b, t, h, n, p], "float32")
                dB[b, t, h, n] = T.cast(dB_sum, out_dB_dtype)
                dC[b, t, h, n] = T.cast(dC_sum, out_dC_dtype)

            if global_lane < ADT_LANES:
                h = global_lane % HEADS
                t = (global_lane // HEADS) % SEQ
                b = global_lane // (HEADS * SEQ)
                dA_sum = T.alloc_var(T.float32, init=0.0)
                ddt_sum = T.alloc_var(T.float32, init=0.0)
                for p in T.serial(HEADDIM):
                    dA_sum += T.cast(dA_partial[b, t, h, p], "float32")
                    ddt_sum += T.cast(ddt_partial[b, t, h, p], "float32")
                dA[b, t, h] = T.cast(dA_sum, out_dA_dtype)
                ddt[b, t, h] = T.cast(ddt_sum, out_ddt_dtype)

            if global_lane < D_LANES:
                h = global_lane
                dD_sum = T.alloc_var(T.float32, init=0.0)
                for b in T.serial(BATCH):
                    for p in T.serial(HEADDIM):
                        dD_sum += T.cast(dD_partial[b, h, p], "float32")
                dD[h] = T.cast(dD_sum, out_dD_dtype)

    artifact = dispatch_lower(bwd_partial_reduce, target="metal", return_msl=True)
    if hasattr(artifact, "_tilelang_engine_target"):
        raise MSLDispatchUnsupported("Mamba3 Path C requires TileLang MSL extraction metadata")
    lowering = cast(_msl_transform.TileLangMSLLowering, artifact)
    input_names = [
        name for name in lowering.buffer_param_names if name not in _BWD_REDUCE_OUTPUT_NAMES
    ]
    if set(input_names) != {
        "dB_partial",
        "dC_partial",
        "dA_partial",
        "ddt_partial",
        "dD_partial",
    }:
        raise MSLDispatchUnsupported(
            "unexpected Mamba3 Path C partial reducer buffer signature: "
            + ", ".join(lowering.buffer_param_names)
        )
    import tilelang

    kernel = tilelang.compile(
        bwd_partial_reduce,
        target=_msl_transform._as_metal_target("metal"),
        execution_backend="tvm_ffi",
        out_idx=list(_BWD_REDUCE_OUTPUT_IDX),
    )
    return kernel, lowering


@lru_cache(maxsize=128)
def _bwd_partial_reduce_threaded_kernel_for(
    BATCH: int,
    SEQ: int,
    HEADS: int,
    HEADDIM: int,
    STATE: int,
    dB_dtype: str = "float32",
    dC_dtype: str = "float32",
    dA_dtype: str = "float32",
    ddt_dtype: str = "float32",
    dD_dtype: str = "float32",
    out_dB_dtype: str = "float32",
    out_dC_dtype: str = "float32",
    out_dA_dtype: str = "float32",
    out_ddt_dtype: str = "float32",
    out_dD_dtype: str = "float32",
) -> tuple[Any, _msl_transform.TileLangMSLLowering]:
    """Build a standalone P-axis reducer with one Metal thread per P lane."""

    if HEADDIM <= 0 or HEADDIM > _MAX_THREADS:
        raise MSLDispatchUnsupported(
            "Mamba3 Path C threaded partial reducer requires 0 < HEADDIM <= "
            f"{_MAX_THREADS}; got {HEADDIM}"
        )

    import tilelang.language as T

    DBC_LANES = BATCH * SEQ * HEADS * STATE
    ADT_LANES = BATCH * SEQ * HEADS
    D_LANES = HEADS
    LANES = max(DBC_LANES, ADT_LANES, D_LANES)
    # The standalone reducer has no recurrent per-thread register state, so it
    # can pack more independent rows per Metal threadgroup than the scan kernels.
    ROWS_PER_BLOCK = max(1, _REDUCE_MAX_THREADS // HEADDIM)

    @T.prim_func
    def bwd_partial_reduce_threaded(
        dB_partial: T.Tensor((BATCH, SEQ, HEADS, STATE, HEADDIM), dB_dtype),
        dC_partial: T.Tensor((BATCH, SEQ, HEADS, STATE, HEADDIM), dC_dtype),
        dA_partial: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), dA_dtype),
        ddt_partial: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), ddt_dtype),
        dD_partial: T.Tensor((BATCH, HEADS, HEADDIM), dD_dtype),
        dB: T.Tensor((BATCH, SEQ, HEADS, STATE), out_dB_dtype),
        dC: T.Tensor((BATCH, SEQ, HEADS, STATE), out_dC_dtype),
        dA: T.Tensor((BATCH, SEQ, HEADS), out_dA_dtype),
        ddt: T.Tensor((BATCH, SEQ, HEADS), out_ddt_dtype),
        dD: T.Tensor((HEADS,), out_dD_dtype),
    ):
        with T.Kernel(T.ceildiv(LANES, ROWS_PER_BLOCK), threads=(HEADDIM, ROWS_PER_BLOCK)) as bx:
            p = T.get_thread_binding(0)
            row_in_block = T.get_thread_binding(1)
            row = bx * ROWS_PER_BLOCK + row_in_block
            reduce_axis = T.reduction_axis("p", HEADDIM, p)

            if row < DBC_LANES:
                n = row % STATE
                h = (row // STATE) % HEADS
                t = (row // (STATE * HEADS)) % SEQ
                b = row // (STATE * HEADS * SEQ)
                dB_acc = T.alloc_local((1,), T.float32)
                dC_acc = T.alloc_local((1,), T.float32)
                dB_reduced = T.alloc_local((1,), T.float32)
                dC_reduced = T.alloc_local((1,), T.float32)
                dB_acc[0] = T.cast(dB_partial[b, t, h, n, p], "float32")
                dC_acc[0] = T.cast(dC_partial[b, t, h, n, p], "float32")
                T.thread_reduce(dB_acc[0], dB_reduced[0], reduce_axis, op="sum")
                T.thread_reduce(dC_acc[0], dC_reduced[0], reduce_axis, op="sum")
                if p == 0:
                    dB[b, t, h, n] = T.cast(dB_reduced[0], out_dB_dtype)
                    dC[b, t, h, n] = T.cast(dC_reduced[0], out_dC_dtype)

            if row < ADT_LANES:
                h = row % HEADS
                t = (row // HEADS) % SEQ
                b = row // (HEADS * SEQ)
                dA_acc = T.alloc_local((1,), T.float32)
                ddt_acc = T.alloc_local((1,), T.float32)
                dA_reduced = T.alloc_local((1,), T.float32)
                ddt_reduced = T.alloc_local((1,), T.float32)
                dA_acc[0] = T.cast(dA_partial[b, t, h, p], "float32")
                ddt_acc[0] = T.cast(ddt_partial[b, t, h, p], "float32")
                T.thread_reduce(dA_acc[0], dA_reduced[0], reduce_axis, op="sum")
                T.thread_reduce(ddt_acc[0], ddt_reduced[0], reduce_axis, op="sum")
                if p == 0:
                    dA[b, t, h] = T.cast(dA_reduced[0], out_dA_dtype)
                    ddt[b, t, h] = T.cast(ddt_reduced[0], out_ddt_dtype)

            if row < D_LANES:
                h = row
                dD_acc = T.alloc_local((1,), T.float32)
                dD_reduced = T.alloc_local((1,), T.float32)
                dD_acc[0] = 0.0
                for b in T.serial(BATCH):
                    dD_acc[0] += T.cast(dD_partial[b, h, p], "float32")
                T.thread_reduce(dD_acc[0], dD_reduced[0], reduce_axis, op="sum")
                if p == 0:
                    dD[h] = T.cast(dD_reduced[0], out_dD_dtype)

    artifact = dispatch_lower(bwd_partial_reduce_threaded, target="metal", return_msl=True)
    if hasattr(artifact, "_tilelang_engine_target"):
        raise MSLDispatchUnsupported("Mamba3 Path C requires TileLang MSL extraction metadata")
    lowering = cast(_msl_transform.TileLangMSLLowering, artifact)
    input_names = [
        name for name in lowering.buffer_param_names if name not in _BWD_REDUCE_OUTPUT_NAMES
    ]
    if set(input_names) != {
        "dB_partial",
        "dC_partial",
        "dA_partial",
        "ddt_partial",
        "dD_partial",
    }:
        raise MSLDispatchUnsupported(
            "unexpected Mamba3 Path C threaded partial reducer buffer signature: "
            + ", ".join(lowering.buffer_param_names)
        )
    import tilelang

    kernel = tilelang.compile(
        bwd_partial_reduce_threaded,
        target=_msl_transform._as_metal_target("metal"),
        execution_backend="tvm_ffi",
        out_idx=list(_BWD_REDUCE_OUTPUT_IDX),
    )
    return kernel, lowering


# ---------------------------------------------------------------------------
# Public dispatch entry points
# ---------------------------------------------------------------------------


def _require_supported_no_hidden_casts(
    op_name: str,
    *named_arrays: tuple[str, mx.array],
) -> dict[str, str]:
    dtypes: dict[str, str] = {}
    bad: list[str] = []
    for name, array in named_arrays:
        dtype = _tl_dtype_for(array.dtype)
        if dtype is None:
            bad.append(f"{name}={array.dtype}")
        else:
            dtypes[name] = dtype
    if bad:
        raise RuntimeError(
            f"{op_name} direct tvm-ffi owner-output route supports mx.float32 "
            "and mx.bfloat16 buffers without hidden casts; got unsupported "
            f"dtypes {bad}"
        )
    return dtypes


def _require_owner_array(
    op_name: str,
    name: str,
    array: mx.array,
    *,
    shape: tuple[int, ...],
    dtype: mx.Dtype,
) -> mx.array:
    if not isinstance(array, mx.array):
        raise TypeError(
            f"{op_name}: owner output {name} must be an mlx.core.array; "
            f"got {type(array).__name__}"
        )
    if tuple(array.shape) != shape:
        raise ValueError(
            f"{op_name}: owner output {name} must have shape {shape}; "
            f"got {tuple(array.shape)}"
        )
    if array.dtype != dtype:
        raise TypeError(
            f"{op_name}: owner output {name} must be {dtype}; got {array.dtype}"
        )
    return array


def _astype_if_needed(array: mx.array, dtype: mx.Dtype) -> mx.array:
    return array if array.dtype == dtype else array.astype(dtype)


def _reduce_bwd_partials_path_c_kernel(
    dB_partial: mx.array,
    dC_partial: mx.array,
    dA_partial: mx.array,
    ddt_partial: mx.array,
    dD_partial: mx.array,
    *,
    output_dtypes: tuple[str, str, str, str, str] | None = None,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Reduce bwd partial gradients over P with a serial TileLang kernel."""

    if len(dB_partial.shape) != 5:
        raise ValueError(
            "mamba3_mimo_bwd_path_c partial reducer expects dB_partial shape "
            "(B, T, H, N, P)"
        )
    batch, seq, heads, state, headdim = map(int, dB_partial.shape)
    expected_c = (batch, seq, heads, state, headdim)
    expected_lane = (batch, seq, heads, headdim)
    expected_d = (batch, heads, headdim)
    if tuple(dC_partial.shape) != expected_c:
        raise ValueError(
            "mamba3_mimo_bwd_path_c partial reducer expects dC_partial shape "
            f"{expected_c}; got {tuple(dC_partial.shape)}"
        )
    if tuple(dA_partial.shape) != expected_lane:
        raise ValueError(
            "mamba3_mimo_bwd_path_c partial reducer expects dA_partial shape "
            f"{expected_lane}; got {tuple(dA_partial.shape)}"
        )
    if tuple(ddt_partial.shape) != expected_lane:
        raise ValueError(
            "mamba3_mimo_bwd_path_c partial reducer expects ddt_partial shape "
            f"{expected_lane}; got {tuple(ddt_partial.shape)}"
        )
    if tuple(dD_partial.shape) != expected_d:
        raise ValueError(
            "mamba3_mimo_bwd_path_c partial reducer expects dD_partial shape "
            f"{expected_d}; got {tuple(dD_partial.shape)}"
        )
    if min(batch, seq, heads, headdim, state) <= 0:
        raise RuntimeError(
            "mamba3_mimo_bwd_path_c partial reducer is not dispatchable for "
            "non-positive shapes"
        )

    dtypes = _require_supported_no_hidden_casts(
        "mamba3_mimo_bwd_path_c partial reducer",
        ("dB_partial", dB_partial),
        ("dC_partial", dC_partial),
        ("dA_partial", dA_partial),
        ("ddt_partial", ddt_partial),
        ("dD_partial", dD_partial),
    )
    if output_dtypes is None:
        reduce_output_dtypes = (
            dtypes["dB_partial"],
            dtypes["dC_partial"],
            dtypes["dA_partial"],
            dtypes["ddt_partial"],
            dtypes["dD_partial"],
        )
    else:
        if len(output_dtypes) != len(_BWD_REDUCE_OUTPUT_NAMES):
            raise ValueError(
                "mamba3_mimo_bwd_path_c partial reducer output_dtypes must "
                f"have {len(_BWD_REDUCE_OUTPUT_NAMES)} entries"
            )
        unsupported = [
            dtype for dtype in output_dtypes if dtype not in {"float32", "bfloat16"}
        ]
        if unsupported:
            raise ValueError(
                "mamba3_mimo_bwd_path_c partial reducer output_dtypes must be "
                f"float32 or bfloat16; got {unsupported}"
            )
        reduce_output_dtypes = output_dtypes
    try:
        kernel, lowering = _bwd_partial_reduce_kernel_for(
            batch,
            seq,
            heads,
            headdim,
            state,
            dtypes["dB_partial"],
            dtypes["dC_partial"],
            dtypes["dA_partial"],
            dtypes["ddt_partial"],
            dtypes["dD_partial"],
            reduce_output_dtypes[0],
            reduce_output_dtypes[1],
            reduce_output_dtypes[2],
            reduce_output_dtypes[3],
            reduce_output_dtypes[4],
        )
    except (MSLDispatchUnsupported, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            "mamba3_mimo_bwd_path_c partial reducer lowering failed"
        ) from exc

    try:
        out_list = kernel(dB_partial, dC_partial, dA_partial, ddt_partial, dD_partial)
    except Exception as exc:
        _raise_if_dlpack_boundary_failure("mamba3_mimo_bwd_path_c partial reducer", exc)
        raise RuntimeError(
            "mamba3_mimo_bwd_path_c partial reducer dispatch failed"
        ) from exc

    if not isinstance(out_list, (list, tuple)) or len(out_list) != len(_BWD_REDUCE_OUTPUT_NAMES):
        raise RuntimeError(
            "Mamba3 Path C partial reducer tvm-ffi returned an invalid output tuple"
        )
    del lowering
    return cast(tuple[mx.array, mx.array, mx.array, mx.array, mx.array], tuple(out_list))


def _reduce_bwd_partials_path_c_fast_kernel(
    dB_partial: mx.array,
    dC_partial: mx.array,
    dA_partial: mx.array,
    ddt_partial: mx.array,
    dD_partial: mx.array,
    *,
    output_dtypes: tuple[str, str, str, str, str] | None = None,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Use the fastest supported standalone TileLang P-reducer for partial grads."""

    if len(dB_partial.shape) == 5:
        headdim = int(dB_partial.shape[4])
    else:
        headdim = 0
    # Full-shape profiling is dtype-sensitive: current Metal thread_reduce is
    # faster for FP32 partials, while the simple serial reducer wins for compact
    # BF16 dB/dC partials where conversion and reduction overhead dominate.
    if (
        dB_partial.dtype == mx.float32
        and dC_partial.dtype == mx.float32
        and _is_power_of_two(headdim)
        and headdim <= _MAX_THREADS
    ):
        return _reduce_bwd_partials_path_c_threaded_kernel(
            dB_partial,
            dC_partial,
            dA_partial,
            ddt_partial,
            dD_partial,
            output_dtypes=output_dtypes,
        )
    return _reduce_bwd_partials_path_c_kernel(
        dB_partial,
        dC_partial,
        dA_partial,
        ddt_partial,
        dD_partial,
        output_dtypes=output_dtypes,
    )


def _reduce_bwd_partials_path_c_threaded_kernel(
    dB_partial: mx.array,
    dC_partial: mx.array,
    dA_partial: mx.array,
    ddt_partial: mx.array,
    dD_partial: mx.array,
    *,
    output_dtypes: tuple[str, str, str, str, str] | None = None,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Reduce bwd partial gradients over P with a threaded TileLang kernel."""

    if len(dB_partial.shape) != 5:
        raise ValueError(
            "mamba3_mimo_bwd_path_c threaded partial reducer expects "
            "dB_partial shape (B, T, H, N, P)"
        )
    batch, seq, heads, state, headdim = map(int, dB_partial.shape)
    expected_c = (batch, seq, heads, state, headdim)
    expected_lane = (batch, seq, heads, headdim)
    expected_d = (batch, heads, headdim)
    if tuple(dC_partial.shape) != expected_c:
        raise ValueError(
            "mamba3_mimo_bwd_path_c threaded partial reducer expects "
            f"dC_partial shape {expected_c}; got {tuple(dC_partial.shape)}"
        )
    if tuple(dA_partial.shape) != expected_lane:
        raise ValueError(
            "mamba3_mimo_bwd_path_c threaded partial reducer expects "
            f"dA_partial shape {expected_lane}; got {tuple(dA_partial.shape)}"
        )
    if tuple(ddt_partial.shape) != expected_lane:
        raise ValueError(
            "mamba3_mimo_bwd_path_c threaded partial reducer expects "
            f"ddt_partial shape {expected_lane}; got {tuple(ddt_partial.shape)}"
        )
    if tuple(dD_partial.shape) != expected_d:
        raise ValueError(
            "mamba3_mimo_bwd_path_c threaded partial reducer expects "
            f"dD_partial shape {expected_d}; got {tuple(dD_partial.shape)}"
        )
    if min(batch, seq, heads, headdim, state) <= 0:
        raise RuntimeError(
            "mamba3_mimo_bwd_path_c threaded partial reducer is not "
            "dispatchable for non-positive shapes"
        )

    dtypes = _require_supported_no_hidden_casts(
        "mamba3_mimo_bwd_path_c threaded partial reducer",
        ("dB_partial", dB_partial),
        ("dC_partial", dC_partial),
        ("dA_partial", dA_partial),
        ("ddt_partial", ddt_partial),
        ("dD_partial", dD_partial),
    )
    if output_dtypes is None:
        reduce_output_dtypes = (
            dtypes["dB_partial"],
            dtypes["dC_partial"],
            dtypes["dA_partial"],
            dtypes["ddt_partial"],
            dtypes["dD_partial"],
        )
    else:
        if len(output_dtypes) != len(_BWD_REDUCE_OUTPUT_NAMES):
            raise ValueError(
                "mamba3_mimo_bwd_path_c threaded partial reducer output_dtypes "
                f"must have {len(_BWD_REDUCE_OUTPUT_NAMES)} entries"
            )
        unsupported = [
            dtype for dtype in output_dtypes if dtype not in {"float32", "bfloat16"}
        ]
        if unsupported:
            raise ValueError(
                "mamba3_mimo_bwd_path_c threaded partial reducer output_dtypes "
                f"must be float32 or bfloat16; got {unsupported}"
            )
        reduce_output_dtypes = output_dtypes
    try:
        kernel, lowering = _bwd_partial_reduce_threaded_kernel_for(
            batch,
            seq,
            heads,
            headdim,
            state,
            dtypes["dB_partial"],
            dtypes["dC_partial"],
            dtypes["dA_partial"],
            dtypes["ddt_partial"],
            dtypes["dD_partial"],
            reduce_output_dtypes[0],
            reduce_output_dtypes[1],
            reduce_output_dtypes[2],
            reduce_output_dtypes[3],
            reduce_output_dtypes[4],
        )
    except (MSLDispatchUnsupported, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            "mamba3_mimo_bwd_path_c threaded partial reducer lowering failed"
        ) from exc

    try:
        out_list = kernel(dB_partial, dC_partial, dA_partial, ddt_partial, dD_partial)
    except Exception as exc:
        _raise_if_dlpack_boundary_failure(
            "mamba3_mimo_bwd_path_c threaded partial reducer",
            exc,
        )
        raise RuntimeError(
            "mamba3_mimo_bwd_path_c threaded partial reducer dispatch failed"
        ) from exc

    if not isinstance(out_list, (list, tuple)) or len(out_list) != len(_BWD_REDUCE_OUTPUT_NAMES):
        raise RuntimeError(
            "Mamba3 Path C threaded partial reducer tvm-ffi returned an "
            "invalid output tuple"
        )
    del lowering
    return cast(tuple[mx.array, mx.array, mx.array, mx.array, mx.array], tuple(out_list))


def _mamba3_fwd_owner_outputs(
    out: Mamba3FwdOwnerOutputs | None,
    *,
    batch: int,
    seq: int,
    heads: int,
    headdim: int,
    state: int,
    y_dtype: mx.Dtype,
    h_last_dtype: mx.Dtype,
) -> Mamba3FwdOwnerOutputs | None:
    op_name = "mamba3_mimo_fwd_path_c"
    if out is None:
        return None
    if not isinstance(out, tuple) or len(out) != 2:
        raise TypeError(
            f"{op_name}: out must be a (y, h_last) owner-output tuple"
        )
    y, h_last = out
    return (
        _require_owner_array(
            op_name,
            "y",
            y,
            shape=(batch, seq, heads, headdim),
            dtype=y_dtype,
        ),
        _require_owner_array(
            op_name,
            "h_last",
            h_last,
            shape=(batch, heads, headdim, state),
            dtype=h_last_dtype,
        ),
    )


def _raise_if_dlpack_boundary_failure(op_name: str, exc: Exception) -> None:
    try:
        from tilelang.contrib.mlx_interop import DLPackConversionError
    except Exception:  # pragma: no cover - only when TileLang import itself is broken
        DLPackConversionError = ()  # type: ignore[assignment]
    if isinstance(exc, DLPackConversionError):
        raise RuntimeError(
            f"{op_name} requires DLPack-exportable, contiguous caller-owned MLX "
            "input/output buffers; Path C will not copy, cast, or materialize "
            "broadcast/slice views implicitly. If this fires inside an MLX "
            "graph transform, the producer has to expose a graph-safe DLPack "
            "view or stay in the existing fused graph path."
        ) from exc


def mamba3_mimo_fwd_path_c(
    x: mx.array,
    B: mx.array,
    C: mx.array,
    z: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
    *,
    out: Mamba3FwdOwnerOutputs | None = None,
) -> tuple[mx.array, mx.array]:
    """Path C forward; fail closed when TileLang metadata or dispatch is unavailable."""

    status = mamba3_mimo_path_c_status()
    if not status.available:
        raise RuntimeError(f"mamba3_mimo_fwd_path_c unavailable: {status.reason}")

    batch, seq, heads, headdim, state = _validate_inputs(x, B, C, z, A, dt, D, h0)
    dtypes = _require_supported_no_hidden_casts(
        "mamba3_mimo_fwd_path_c",
        ("x", x),
        ("B", B),
        ("C", C),
        ("z", z),
        ("A", A),
        ("dt", dt),
        ("D", D),
        ("h0", h0),
    )
    if seq == 0:
        if out is not None:
            raise RuntimeError(
                "mamba3_mimo_fwd_path_c owner-output route is not dispatchable "
                "for seq=0; return h0 directly instead of copying it"
            )
        return mx.zeros((batch, 0, heads, headdim), dtype=x.dtype), h0

    try:
        kernel, lowering = _fwd_kernel_for(
            batch,
            seq,
            heads,
            headdim,
            state,
            dtypes["x"],
            dtypes["B"],
            dtypes["C"],
            dtypes["z"],
            dtypes["A"],
            dtypes["dt"],
            dtypes["D"],
            dtypes["h0"],
            dtypes["x"],
            dtypes["h0"],
        )
    except (MSLDispatchUnsupported, RuntimeError, ValueError) as exc:
        raise RuntimeError("mamba3_mimo_fwd_path_c lowering failed") from exc

    owner_outputs = _mamba3_fwd_owner_outputs(
        out,
        batch=batch,
        seq=seq,
        heads=heads,
        headdim=headdim,
        state=state,
        y_dtype=x.dtype,
        h_last_dtype=h0.dtype,
    )
    try:
        if owner_outputs is None:
            out_list = kernel(x, B, C, z, A, dt, D, h0)
        else:
            y, h_last = owner_outputs
            out_list = kernel(
                x,
                B,
                C,
                z,
                A,
                dt,
                D,
                h0,
                out=(y, h_last),
            )
    except Exception as exc:
        _raise_if_dlpack_boundary_failure("mamba3_mimo_fwd_path_c", exc)
        raise RuntimeError("mamba3_mimo_fwd_path_c dispatch failed") from exc

    if not isinstance(out_list, (list, tuple)) or len(out_list) != 2:
        raise RuntimeError("Mamba3 Path C fwd tvm-ffi returned an invalid output tuple")
    if owner_outputs is not None:
        y, h_last = owner_outputs
        if not all(
            got is expected
            for got, expected in zip(out_list, (y, h_last), strict=True)
        ):
            raise RuntimeError(
                "Mamba3 Path C fwd tvm-ffi did not return caller-owned outputs"
            )
    y, h_last = cast(tuple[mx.array, mx.array], tuple(out_list))
    del lowering
    return y, h_last


def _mamba3_mimo_fwd_path_c_with_snapshots(
    x: mx.array,
    B: mx.array,
    C: mx.array,
    z: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
    *,
    snapshot_dtype: str | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    """Path C training fwd candidate returning ``(y, h_last, h_snap)``."""

    status = mamba3_mimo_path_c_status()
    if not status.available:
        raise RuntimeError(f"mamba3_mimo_fwd_path_c unavailable: {status.reason}")

    batch, seq, heads, headdim, state = _validate_inputs(x, B, C, z, A, dt, D, h0)
    dtypes = _require_supported_no_hidden_casts(
        "mamba3_mimo_fwd_path_c",
        ("x", x),
        ("B", B),
        ("C", C),
        ("z", z),
        ("A", A),
        ("dt", dt),
        ("D", D),
        ("h0", h0),
    )
    if seq == 0:
        raise RuntimeError(
            "mamba3_mimo_fwd_path_c snapshot route is not dispatchable for "
            "seq=0; returning h0 as h_snap would require a wrapper copy"
        )
    if snapshot_dtype is None:
        snapshot_dtype = dtypes["h0"]
    if snapshot_dtype not in {"float32", "bfloat16"}:
        raise ValueError(
            "mamba3_mimo_fwd_path_c snapshot_dtype must be float32 or bfloat16; "
            f"got {snapshot_dtype}"
        )

    try:
        kernel, lowering = _fwd_with_snapshots_kernel_for(
            batch,
            seq,
            heads,
            headdim,
            state,
            dtypes["x"],
            dtypes["B"],
            dtypes["C"],
            dtypes["z"],
            dtypes["A"],
            dtypes["dt"],
            dtypes["D"],
            dtypes["h0"],
            dtypes["x"],
            dtypes["h0"],
            snapshot_dtype,
        )
    except (MSLDispatchUnsupported, RuntimeError, ValueError) as exc:
        raise RuntimeError("mamba3_mimo_fwd_path_c snapshot lowering failed") from exc

    try:
        out_list = kernel(x, B, C, z, A, dt, D, h0)
    except Exception as exc:
        _raise_if_dlpack_boundary_failure("mamba3_mimo_fwd_path_c", exc)
        raise RuntimeError("mamba3_mimo_fwd_path_c snapshot dispatch failed") from exc

    if not isinstance(out_list, (list, tuple)) or len(out_list) != 3:
        raise RuntimeError(
            "Mamba3 Path C fwd snapshot tvm-ffi returned an invalid output tuple"
        )
    y, h_last, h_snap = cast(tuple[mx.array, mx.array, mx.array], tuple(out_list))
    del lowering
    return y, h_last, h_snap


def _mamba3_mimo_bwd_path_c_simd_kernel(
    dy: mx.array,
    x: mx.array,
    B: mx.array,
    C: mx.array,
    z: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Run the simdgroup P-reduced Path C bwd kernel."""

    status = mamba3_mimo_path_c_status()
    if not status.available:
        raise RuntimeError(f"mamba3_mimo_bwd_path_c unavailable: {status.reason}")

    batch, seq, heads, headdim, state = _validate_inputs(x, B, C, z, A, dt, D, h0)
    dtypes = _require_supported_no_hidden_casts(
        "mamba3_mimo_bwd_path_c",
        ("dy", dy),
        ("x", x),
        ("B", B),
        ("C", C),
        ("z", z),
        ("A", A),
        ("dt", dt),
        ("D", D),
        ("h0", h0),
    )
    if seq == 0:
        raise RuntimeError(
            "mamba3_mimo_bwd_path_c simd route is not dispatchable for seq=0 "
            "because no TileLang kernel runs to initialize buffers"
        )
    if not _bwd_simd_p_reduction_supported(
        batch=batch,
        heads=heads,
        headdim=headdim,
    ):
        raise MSLDispatchUnsupported(
            "Mamba3 Path C simd bwd route requires HEADDIM<=32 or a multiple "
            "of 32 with threadgroups aligned to full P rows"
        )

    try:
        scan_plan = _bwd_scan_plan_for(
            batch=batch,
            seq=seq,
            heads=heads,
            headdim=headdim,
            state=state,
        )
        if scan_plan.snapshot_plan.policy == "state-boundary-cache":
            snapshot_kernel, snapshot_lowering = _bwd_state_snapshots_kernel_for(
                batch,
                seq,
                heads,
                headdim,
                state,
                dtypes["x"],
                dtypes["B"],
                dtypes["A"],
                dtypes["dt"],
                dtypes["h0"],
            )
            kernel, lowering = _bwd_simd_reduce_kernel_for_state_snapshots(
                batch,
                seq,
                heads,
                headdim,
                state,
                dtypes["dy"],
                dtypes["x"],
                dtypes["B"],
                dtypes["C"],
                dtypes["z"],
                dtypes["A"],
                dtypes["dt"],
                dtypes["D"],
                "float32",
                "float32",
                "float32",
                "float32",
                "float32",
                "float32",
                "float32",
                "float32",
            )
        else:
            snapshot_kernel = None
            snapshot_lowering = None
            kernel, lowering = _bwd_simd_reduce_kernel_for(
                batch,
                seq,
                heads,
                headdim,
                state,
                dtypes["dy"],
                dtypes["x"],
                dtypes["B"],
                dtypes["C"],
                dtypes["z"],
                dtypes["A"],
                dtypes["dt"],
                dtypes["D"],
                dtypes["h0"],
                "float32",
                "float32",
                "float32",
                "float32",
                "float32",
                "float32",
                "float32",
                "float32",
            )
    except (MSLDispatchUnsupported, RuntimeError, ValueError) as exc:
        raise RuntimeError("mamba3_mimo_bwd_path_c simd lowering failed") from exc

    try:
        if snapshot_kernel is None:
            out_list = kernel(dy, x, B, C, z, A, dt, D, h0)
        else:
            snapshot_out = snapshot_kernel(x, B, A, dt, h0)
            if isinstance(snapshot_out, mx.array):
                h_snap = snapshot_out
            elif isinstance(snapshot_out, (list, tuple)) and len(snapshot_out) == 1:
                h_snap = snapshot_out[0]
            else:
                raise RuntimeError(
                    "Mamba3 Path C snapshot tvm-ffi returned an invalid output tuple"
                )
            out_list = kernel(dy, x, B, C, z, A, dt, D, h_snap)
    except Exception as exc:
        _raise_if_dlpack_boundary_failure("mamba3_mimo_bwd_path_c", exc)
        raise RuntimeError("mamba3_mimo_bwd_path_c simd dispatch failed") from exc

    if not isinstance(out_list, (list, tuple)) or len(out_list) != len(_BWD_SIMD_OUTPUT_NAMES):
        raise RuntimeError("Mamba3 Path C simd bwd tvm-ffi returned an invalid output tuple")
    dx_pc, dz_pc, dB_pc, dC_pc, dA_pc, ddt_pc, dD_bh, dh0_pc = out_list
    del lowering, snapshot_lowering
    dD_pc = mx.sum(dD_bh, axis=0)        # -> (H,)
    return (
        _astype_if_needed(dx_pc, x.dtype),
        _astype_if_needed(dB_pc, B.dtype),
        _astype_if_needed(dC_pc, C.dtype),
        _astype_if_needed(dz_pc, z.dtype),
        _astype_if_needed(dA_pc, A.dtype),
        _astype_if_needed(ddt_pc, dt.dtype),
        _astype_if_needed(dD_pc, D.dtype),
        _astype_if_needed(dh0_pc, h0.dtype),
    )


def _mamba3_mimo_bwd_path_c_partial_outputs_from_snapshots(
    dy: mx.array,
    x: mx.array,
    B: mx.array,
    C: mx.array,
    z: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
    h_snap: mx.array,
    *,
    partial_dtypes: tuple[str, str, str, str, str] | None = None,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Run generated bwd partials using snapshots produced by training fwd."""

    status = mamba3_mimo_path_c_status()
    if not status.available:
        raise RuntimeError(f"mamba3_mimo_bwd_path_c unavailable: {status.reason}")

    batch, seq, heads, headdim, state = _validate_inputs(x, B, C, z, A, dt, D, h0)
    dtypes = _require_supported_no_hidden_casts(
        "mamba3_mimo_bwd_path_c",
        ("dy", dy),
        ("x", x),
        ("B", B),
        ("C", C),
        ("z", z),
        ("A", A),
        ("dt", dt),
        ("D", D),
        ("h0", h0),
        ("h_snap", h_snap),
    )
    if seq == 0:
        raise RuntimeError(
            "mamba3_mimo_bwd_path_c snapshot-reuse route is not dispatchable "
            "for seq=0 because no TileLang kernel runs to initialize buffers"
        )
    expected_h_snap = (batch, seq + 1, heads, headdim, state)
    if tuple(h_snap.shape) != expected_h_snap:
        raise ValueError(
            "mamba3_mimo_bwd_path_c snapshot-reuse route expects h_snap shape "
            f"{expected_h_snap}; got {tuple(h_snap.shape)}"
        )
    if partial_dtypes is None:
        partial_dtypes = _bwd_partial_dtypes_for_input_dtypes(dtypes)
    if len(partial_dtypes) != len(_BWD_REDUCE_OUTPUT_NAMES):
        raise ValueError(
            "mamba3_mimo_bwd_path_c partial_dtypes must have "
            f"{len(_BWD_REDUCE_OUTPUT_NAMES)} entries"
        )
    unsupported_partials = [
        dtype for dtype in partial_dtypes if dtype not in {"float32", "bfloat16"}
    ]
    if unsupported_partials:
        raise ValueError(
            "mamba3_mimo_bwd_path_c partial_dtypes must be float32 or bfloat16; "
            f"got {unsupported_partials}"
        )

    try:
        kernel, lowering = _bwd_partial_kernel_for_state_snapshots(
            batch,
            seq,
            heads,
            headdim,
            state,
            dtypes["dy"],
            dtypes["x"],
            dtypes["B"],
            dtypes["C"],
            dtypes["z"],
            dtypes["A"],
            dtypes["dt"],
            dtypes["D"],
            dtypes["h0"],
            dtypes["x"],
            dtypes["z"],
            partial_dtypes[0],
            partial_dtypes[1],
            partial_dtypes[2],
            partial_dtypes[3],
            partial_dtypes[4],
            dtypes["h0"],
            dtypes["h_snap"],
        )
    except (MSLDispatchUnsupported, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            "mamba3_mimo_bwd_path_c snapshot-reuse partial lowering failed"
        ) from exc

    try:
        out_list = kernel(dy, x, B, C, z, A, dt, D, h_snap)
    except Exception as exc:
        _raise_if_dlpack_boundary_failure("mamba3_mimo_bwd_path_c", exc)
        raise RuntimeError(
            "mamba3_mimo_bwd_path_c snapshot-reuse partial dispatch failed"
        ) from exc

    if not isinstance(out_list, (list, tuple)) or len(out_list) != len(_BWD_PARTIAL_OUTPUT_NAMES):
        raise RuntimeError("Mamba3 Path C partial bwd tvm-ffi returned an invalid output tuple")
    dx_pc, dz_pc, dB_partial, dC_partial, dA_partial, ddt_partial, dD_partial, dh0_pc = out_list
    del lowering
    return (
        dx_pc,
        dz_pc,
        dB_partial,
        dC_partial,
        dA_partial,
        ddt_partial,
        dD_partial,
        dh0_pc,
    )


def _mamba3_mimo_bwd_path_c_from_snapshots_kernel(
    dy: mx.array,
    x: mx.array,
    B: mx.array,
    C: mx.array,
    z: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
    h_snap: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Run Path C bwd with snapshots already materialized by training fwd."""

    bf16_route = all(
        array.dtype == mx.bfloat16 for array in (dy, x, B, C, z, A, dt, D, h0)
    )
    (
        dx_pc,
        dz_pc,
        dB_partial,
        dC_partial,
        dA_partial,
        ddt_partial,
        dD_partial,
        dh0_pc,
    ) = _mamba3_mimo_bwd_path_c_partial_outputs_from_snapshots(
        dy,
        x,
        B,
        C,
        z,
        A,
        dt,
        D,
        h0,
        h_snap,
    )
    dB_pc, dC_pc, dA_pc, ddt_pc, dD_pc = _reduce_bwd_partials_path_c_fast_kernel(
        dB_partial,
        dC_partial,
        dA_partial,
        ddt_partial,
        dD_partial,
        output_dtypes=(
            ("bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16")
            if bf16_route
            else None
        ),
    )
    return (
        _astype_if_needed(dx_pc, x.dtype),
        _astype_if_needed(dB_pc, B.dtype),
        _astype_if_needed(dC_pc, C.dtype),
        _astype_if_needed(dz_pc, z.dtype),
        _astype_if_needed(dA_pc, A.dtype),
        _astype_if_needed(ddt_pc, dt.dtype),
        _astype_if_needed(dD_pc, D.dtype),
        _astype_if_needed(dh0_pc, h0.dtype),
    )


def _mamba3_mimo_bwd_path_c_partial_outputs(
    dy: mx.array,
    x: mx.array,
    B: mx.array,
    C: mx.array,
    z: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
    *,
    snapshot_dtype: str = "float32",
    partial_dtypes: tuple[str, str, str, str, str] | None = None,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Run the generic generated bwd scan and return unreduced partials."""

    status = mamba3_mimo_path_c_status()
    if not status.available:
        raise RuntimeError(f"mamba3_mimo_bwd_path_c unavailable: {status.reason}")

    batch, seq, heads, headdim, state = _validate_inputs(x, B, C, z, A, dt, D, h0)
    dtypes = _require_supported_no_hidden_casts(
        "mamba3_mimo_bwd_path_c",
        ("dy", dy),
        ("x", x),
        ("B", B),
        ("C", C),
        ("z", z),
        ("A", A),
        ("dt", dt),
        ("D", D),
        ("h0", h0),
    )
    if seq == 0:
        raise RuntimeError(
            "mamba3_mimo_bwd_path_c partial route is not dispatchable for seq=0 "
            "because no TileLang kernel runs to initialize buffers"
        )

    try:
        snapshot_kernel, snapshot_lowering = _bwd_state_snapshots_kernel_for(
            batch,
            seq,
            heads,
            headdim,
            state,
            dtypes["x"],
            dtypes["B"],
            dtypes["A"],
            dtypes["dt"],
            dtypes["h0"],
            snapshot_dtype,
        )
    except (MSLDispatchUnsupported, RuntimeError, ValueError) as exc:
        raise RuntimeError("mamba3_mimo_bwd_path_c snapshot lowering failed") from exc

    try:
        snapshot_out = snapshot_kernel(x, B, A, dt, h0)
        if isinstance(snapshot_out, mx.array):
            h_snap = snapshot_out
        elif isinstance(snapshot_out, (list, tuple)) and len(snapshot_out) == 1:
            h_snap = snapshot_out[0]
        else:
            raise RuntimeError(
                "Mamba3 Path C snapshot tvm-ffi returned an invalid output tuple"
            )
        del snapshot_lowering
        return _mamba3_mimo_bwd_path_c_partial_outputs_from_snapshots(
            dy,
            x,
            B,
            C,
            z,
            A,
            dt,
            D,
            h0,
            h_snap,
            partial_dtypes=partial_dtypes,
        )
    except Exception as exc:
        _raise_if_dlpack_boundary_failure("mamba3_mimo_bwd_path_c", exc)
        raise RuntimeError("mamba3_mimo_bwd_path_c partial dispatch failed") from exc


def _mamba3_mimo_bwd_path_c_partial_kernel(
    dy: mx.array,
    x: mx.array,
    B: mx.array,
    C: mx.array,
    z: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Run the production generated bwd route with the fastest TileLang reducer."""

    bf16_route = all(
        array.dtype == mx.bfloat16 for array in (dy, x, B, C, z, A, dt, D, h0)
    )
    snapshot_dtype = "bfloat16" if bf16_route else "float32"
    (
        dx_pc,
        dz_pc,
        dB_partial,
        dC_partial,
        dA_partial,
        ddt_partial,
        dD_partial,
        dh0_pc,
    ) = _mamba3_mimo_bwd_path_c_partial_outputs(
        dy,
        x,
        B,
        C,
        z,
        A,
        dt,
        D,
        h0,
        snapshot_dtype=snapshot_dtype,
    )
    dB_pc, dC_pc, dA_pc, ddt_pc, dD_pc = _reduce_bwd_partials_path_c_fast_kernel(
        dB_partial,
        dC_partial,
        dA_partial,
        ddt_partial,
        dD_partial,
        output_dtypes=(
            ("bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16")
            if bf16_route
            else None
        ),
    )
    return (
        _astype_if_needed(dx_pc, x.dtype),
        _astype_if_needed(dB_pc, B.dtype),
        _astype_if_needed(dC_pc, C.dtype),
        _astype_if_needed(dz_pc, z.dtype),
        _astype_if_needed(dA_pc, A.dtype),
        _astype_if_needed(ddt_pc, dt.dtype),
        _astype_if_needed(dD_pc, D.dtype),
        _astype_if_needed(dh0_pc, h0.dtype),
    )


def _mamba3_mimo_bwd_path_c_partial_tl_reduce_kernel(
    dy: mx.array,
    x: mx.array,
    B: mx.array,
    C: mx.array,
    z: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Run generated bwd partials, then reduce them with a TileLang reducer."""

    bf16_route = all(
        array.dtype == mx.bfloat16 for array in (dy, x, B, C, z, A, dt, D, h0)
    )
    snapshot_dtype = "bfloat16" if bf16_route else "float32"
    (
        dx_pc,
        dz_pc,
        dB_partial,
        dC_partial,
        dA_partial,
        ddt_partial,
        dD_partial,
        dh0_pc,
    ) = _mamba3_mimo_bwd_path_c_partial_outputs(
        dy,
        x,
        B,
        C,
        z,
        A,
        dt,
        D,
        h0,
        snapshot_dtype=snapshot_dtype,
    )
    dB_pc, dC_pc, dA_pc, ddt_pc, dD_pc = _reduce_bwd_partials_path_c_fast_kernel(
        dB_partial,
        dC_partial,
        dA_partial,
        ddt_partial,
        dD_partial,
        output_dtypes=(
            ("bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16")
            if bf16_route
            else None
        ),
    )
    return (
        _astype_if_needed(dx_pc, x.dtype),
        _astype_if_needed(dB_pc, B.dtype),
        _astype_if_needed(dC_pc, C.dtype),
        _astype_if_needed(dz_pc, z.dtype),
        _astype_if_needed(dA_pc, A.dtype),
        _astype_if_needed(ddt_pc, dt.dtype),
        _astype_if_needed(dD_pc, D.dtype),
        _astype_if_needed(dh0_pc, h0.dtype),
    )


def _mamba3_mimo_bwd_path_c_partial_threaded_reduce_kernel(
    dy: mx.array,
    x: mx.array,
    B: mx.array,
    C: mx.array,
    z: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Run generated bwd partials, then reduce P with a threaded TileLang reducer."""

    bf16_route = all(
        array.dtype == mx.bfloat16 for array in (dy, x, B, C, z, A, dt, D, h0)
    )
    snapshot_dtype = "bfloat16" if bf16_route else "float32"
    (
        dx_pc,
        dz_pc,
        dB_partial,
        dC_partial,
        dA_partial,
        ddt_partial,
        dD_partial,
        dh0_pc,
    ) = _mamba3_mimo_bwd_path_c_partial_outputs(
        dy,
        x,
        B,
        C,
        z,
        A,
        dt,
        D,
        h0,
        snapshot_dtype=snapshot_dtype,
    )
    dB_pc, dC_pc, dA_pc, ddt_pc, dD_pc = _reduce_bwd_partials_path_c_threaded_kernel(
        dB_partial,
        dC_partial,
        dA_partial,
        ddt_partial,
        dD_partial,
        output_dtypes=(
            ("bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16")
            if bf16_route
            else None
        ),
    )
    return (
        _astype_if_needed(dx_pc, x.dtype),
        _astype_if_needed(dB_pc, B.dtype),
        _astype_if_needed(dC_pc, C.dtype),
        _astype_if_needed(dz_pc, z.dtype),
        _astype_if_needed(dA_pc, A.dtype),
        _astype_if_needed(ddt_pc, dt.dtype),
        _astype_if_needed(dD_pc, D.dtype),
        _astype_if_needed(dh0_pc, h0.dtype),
    )


def _mamba3_mimo_bwd_path_c_bf16_snapshot_kernel(
    dy: mx.array,
    x: mx.array,
    B: mx.array,
    C: mx.array,
    z: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Diagnostic bwd route that stores state snapshots as BF16."""

    if any(
        array.dtype != mx.bfloat16
        for array in (dy, x, B, C, z, A, dt, D, h0)
    ):
        raise RuntimeError("Mamba3 Path C BF16 snapshot diagnostic requires BF16 inputs")

    (
        dx_pc,
        dz_pc,
        dB_partial,
        dC_partial,
        dA_partial,
        ddt_partial,
        dD_partial,
        dh0_pc,
    ) = _mamba3_mimo_bwd_path_c_partial_outputs(
        dy,
        x,
        B,
        C,
        z,
        A,
        dt,
        D,
        h0,
        snapshot_dtype="bfloat16",
    )
    dB_pc = mx.sum(dB_partial, axis=4)
    dC_pc = mx.sum(dC_partial, axis=4)
    dA_pc = mx.sum(dA_partial, axis=3)
    ddt_pc = mx.sum(ddt_partial, axis=3)
    dD_pc = mx.sum(dD_partial, axis=(0, 2))
    return (
        _astype_if_needed(dx_pc, x.dtype),
        _astype_if_needed(dB_pc, B.dtype),
        _astype_if_needed(dC_pc, C.dtype),
        _astype_if_needed(dz_pc, z.dtype),
        _astype_if_needed(dA_pc, A.dtype),
        _astype_if_needed(ddt_pc, dt.dtype),
        _astype_if_needed(dD_pc, D.dtype),
        _astype_if_needed(dh0_pc, h0.dtype),
    )


def _mamba3_mimo_bwd_path_c_scratch_partial_outputs(
    dy: mx.array,
    x: mx.array,
    B: mx.array,
    C: mx.array,
    z: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Run the single-kernel scratch bwd producer and return unreduced partials."""

    status = mamba3_mimo_path_c_status()
    if not status.available:
        raise RuntimeError(f"mamba3_mimo_bwd_path_c unavailable: {status.reason}")

    batch, seq, heads, headdim, state = _validate_inputs(x, B, C, z, A, dt, D, h0)
    dtypes = _require_supported_no_hidden_casts(
        "mamba3_mimo_bwd_path_c",
        ("dy", dy),
        ("x", x),
        ("B", B),
        ("C", C),
        ("z", z),
        ("A", A),
        ("dt", dt),
        ("D", D),
        ("h0", h0),
    )
    if seq == 0:
        raise RuntimeError(
            "mamba3_mimo_bwd_path_c scratch route is not dispatchable for seq=0 "
            "because no TileLang kernel runs to initialize buffers"
        )

    bf16_route = all(
        array.dtype == mx.bfloat16 for array in (dy, x, B, C, z, A, dt, D, h0)
    )
    partial_dtypes = _bwd_partial_dtypes_for_input_dtypes(dtypes)
    try:
        kernel, lowering = _bwd_scratch_partial_kernel_for(
            batch,
            seq,
            heads,
            headdim,
            state,
            dtypes["dy"],
            dtypes["x"],
            dtypes["B"],
            dtypes["C"],
            dtypes["z"],
            dtypes["A"],
            dtypes["dt"],
            dtypes["D"],
            dtypes["h0"],
            dtypes["x"],
            dtypes["z"],
            partial_dtypes[0],
            partial_dtypes[1],
            partial_dtypes[2],
            partial_dtypes[3],
            partial_dtypes[4],
            dtypes["h0"],
            "bfloat16" if bf16_route else "float32",
        )
    except (MSLDispatchUnsupported, RuntimeError, ValueError) as exc:
        raise RuntimeError("mamba3_mimo_bwd_path_c scratch lowering failed") from exc

    try:
        out_list = kernel(dy, x, B, C, z, A, dt, D, h0)
    except Exception as exc:
        _raise_if_dlpack_boundary_failure("mamba3_mimo_bwd_path_c", exc)
        raise RuntimeError("mamba3_mimo_bwd_path_c scratch dispatch failed") from exc

    if not isinstance(out_list, (list, tuple)) or len(out_list) != len(_BWD_SCRATCH_OUTPUT_NAMES):
        raise RuntimeError("Mamba3 Path C scratch bwd tvm-ffi returned an invalid output tuple")
    del lowering
    return cast(
        tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array],
        tuple(out_list),
    )


def _mamba3_mimo_bwd_path_c_scratch_kernel(
    dy: mx.array,
    x: mx.array,
    B: mx.array,
    C: mx.array,
    z: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Run the single-kernel scratch bwd route with the shared TileLang P-reducer."""

    bf16_route = all(
        array.dtype == mx.bfloat16 for array in (dy, x, B, C, z, A, dt, D, h0)
    )
    (
        dx_pc,
        dz_pc,
        dB_partial,
        dC_partial,
        dA_partial,
        ddt_partial,
        dD_partial,
        dh0_pc,
    ) = _mamba3_mimo_bwd_path_c_scratch_partial_outputs(dy, x, B, C, z, A, dt, D, h0)
    dB_pc, dC_pc, dA_pc, ddt_pc, dD_pc = _reduce_bwd_partials_path_c_fast_kernel(
        dB_partial,
        dC_partial,
        dA_partial,
        ddt_partial,
        dD_partial,
        output_dtypes=(
            ("bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16")
            if bf16_route
            else None
        ),
    )
    return (
        _astype_if_needed(dx_pc, x.dtype),
        _astype_if_needed(dB_pc, B.dtype),
        _astype_if_needed(dC_pc, C.dtype),
        _astype_if_needed(dz_pc, z.dtype),
        _astype_if_needed(dA_pc, A.dtype),
        _astype_if_needed(ddt_pc, dt.dtype),
        _astype_if_needed(dD_pc, D.dtype),
        _astype_if_needed(dh0_pc, h0.dtype),
    )


def _mamba3_mimo_bwd_path_c_scratch_mlx_reduce_kernel(
    dy: mx.array,
    x: mx.array,
    B: mx.array,
    C: mx.array,
    z: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Run scratch bwd partials, then reduce them with MLX tensor reductions."""

    (
        dx_pc,
        dz_pc,
        dB_partial,
        dC_partial,
        dA_partial,
        ddt_partial,
        dD_partial,
        dh0_pc,
    ) = _mamba3_mimo_bwd_path_c_scratch_partial_outputs(dy, x, B, C, z, A, dt, D, h0)
    dB_pc = mx.sum(dB_partial, axis=4)
    dC_pc = mx.sum(dC_partial, axis=4)
    dA_pc = mx.sum(dA_partial, axis=3)
    ddt_pc = mx.sum(ddt_partial, axis=3)
    dD_pc = mx.sum(dD_partial, axis=(0, 2))
    return (
        _astype_if_needed(dx_pc, x.dtype),
        _astype_if_needed(dB_pc, B.dtype),
        _astype_if_needed(dC_pc, C.dtype),
        _astype_if_needed(dz_pc, z.dtype),
        _astype_if_needed(dA_pc, A.dtype),
        _astype_if_needed(ddt_pc, dt.dtype),
        _astype_if_needed(dD_pc, D.dtype),
        _astype_if_needed(dh0_pc, h0.dtype),
    )


def _mamba3_mimo_bwd_path_c_kernel(
    dy: mx.array,
    x: mx.array,
    B: mx.array,
    C: mx.array,
    z: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
    *,
    out: Mamba3BwdOwnerOutputs | None = None,
) -> tuple[mx.array, ...]:
    """Run the lowered Path C bwd kernel with final-gradient owner outputs."""

    batch, seq, heads, headdim, _state = _validate_inputs(x, B, C, z, A, dt, D, h0)
    _require_supported_no_hidden_casts(
        "mamba3_mimo_bwd_path_c",
        ("dy", dy),
        ("x", x),
        ("B", B),
        ("C", C),
        ("z", z),
        ("A", A),
        ("dt", dt),
        ("D", D),
        ("h0", h0),
    )
    if out is not None:
        raise RuntimeError(
            "mamba3_mimo_bwd_path_c does not expose public owner-output "
            "buffers; final-gradient owner-output lowering is not implemented yet"
        )
    if seq == 0:
        return (
            mx.zeros_like(x),
            mx.zeros_like(B),
            mx.zeros_like(C),
            mx.zeros_like(z),
            mx.zeros_like(A),
            mx.zeros_like(dt),
            mx.zeros_like(D),
            mx.zeros_like(h0),
        )
    if all(array.dtype == mx.bfloat16 for array in (dy, x, B, C, z, A, dt, D, h0)):
        return _mamba3_mimo_bwd_path_c_scratch_kernel(dy, x, B, C, z, A, dt, D, h0)
    return _mamba3_mimo_bwd_path_c_partial_kernel(dy, x, B, C, z, A, dt, D, h0)


def mamba3_mimo_bwd_path_c(
    dy: mx.array,
    x: mx.array,
    B: mx.array,
    C: mx.array,
    z: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
    *,
    out: Mamba3BwdOwnerOutputs | None = None,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Backward pass via the lowered TileLang DSL kernel."""

    return _mamba3_mimo_bwd_path_c_kernel(dy, x, B, C, z, A, dt, D, h0, out=out)


@mx.custom_function
def mamba3_mimo_apply_path_c(
    x: mx.array,
    B: mx.array,
    C: mx.array,
    z: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
) -> mx.array:
    """Path C forward wrapper exposing only ``y``.

    Note (Path B vs Path C):
        The Path B analogue is ``mamba3_mimo_apply`` in ``mamba3.py``.
        Neither apply accepts a ``force_metal`` / ``force_path_c`` kwarg, so
        there is no kwarg rename to migrate. This entrypoint is **not**
        re-exported from ``cppmega_mlx.nn._tilelang.__init__`` — Path C
        Mamba3 is a proof / override path; Path B is the production
        entrypoint. See ``docs/production_kernel_routing.md``.

        The direct tvm-ffi path is graph-transform callable when MLX exposes
        graph-safe DLPack export. TileLang owns output allocation through
        ``out_idx`` metadata; explicit ``out=`` remains the full-ABI
        caller-owned route.
    """

    y, _ = mamba3_mimo_fwd_path_c(x, B, C, z, A, dt, D, h0)
    return y


@mamba3_mimo_apply_path_c.vjp
def _mamba3_mimo_apply_path_c_vjp(
    primals: tuple[mx.array, ...],
    cotangent: mx.array,
    output: mx.array,
) -> tuple[mx.array, ...]:
    del output
    x, B, C, z, A, dt, D, h0 = primals
    return mamba3_mimo_bwd_path_c(cotangent, x, B, C, z, A, dt, D, h0)


@mx.custom_function
def _mamba3_mimo_apply_with_snapshots_path_c(
    x: mx.array,
    B: mx.array,
    C: mx.array,
    z: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """Training-only Path C forward returning snapshots for its VJP."""

    snapshot_dtype = (
        "bfloat16"
        if all(array.dtype == mx.bfloat16 for array in (x, B, C, z, A, dt, D, h0))
        else "float32"
    )
    return _mamba3_mimo_fwd_path_c_with_snapshots(
        x,
        B,
        C,
        z,
        A,
        dt,
        D,
        h0,
        snapshot_dtype=snapshot_dtype,
    )


@_mamba3_mimo_apply_with_snapshots_path_c.vjp
def _mamba3_mimo_apply_with_snapshots_path_c_vjp(
    primals: tuple[mx.array, ...],
    cotangent: tuple[mx.array, mx.array, mx.array],
    output: tuple[mx.array, mx.array, mx.array],
) -> tuple[mx.array, ...]:
    x, B, C, z, A, dt, D, h0 = primals
    dy = cotangent[0]
    h_snap = output[2]
    return _mamba3_mimo_bwd_path_c_from_snapshots_kernel(
        dy,
        x,
        B,
        C,
        z,
        A,
        dt,
        D,
        h0,
        h_snap,
    )


def _mamba3_path_c_training_should_reuse_snapshots(*arrays: mx.array) -> bool:
    """Return whether fwd-produced snapshots are currently the faster train path."""

    return all(array.dtype == mx.float32 for array in arrays)


def mamba3_mimo_apply_training_path_c(
    x: mx.array,
    B: mx.array,
    C: mx.array,
    z: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
) -> mx.array:
    """Path C y-only training surface using the fastest verified VJP route."""

    if not _mamba3_path_c_training_should_reuse_snapshots(x, B, C, z, A, dt, D, h0):
        return mamba3_mimo_apply_path_c(x, B, C, z, A, dt, D, h0)

    y, _h_last, _h_snap = _mamba3_mimo_apply_with_snapshots_path_c(
        x,
        B,
        C,
        z,
        A,
        dt,
        D,
        h0,
    )
    return y


def mamba3_mimo_apply_with_state_training_path_c(
    x: mx.array,
    B: mx.array,
    C: mx.array,
    z: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
) -> tuple[mx.array, mx.array]:
    """Path C training surface returning state with the fastest verified VJP."""

    if not _mamba3_path_c_training_should_reuse_snapshots(x, B, C, z, A, dt, D, h0):
        return mamba3_mimo_apply_with_state_path_c(x, B, C, z, A, dt, D, h0)

    y, h_last, _h_snap = _mamba3_mimo_apply_with_snapshots_path_c(
        x,
        B,
        C,
        z,
        A,
        dt,
        D,
        h0,
    )
    return y, h_last


@mx.custom_function
def mamba3_mimo_apply_with_state_path_c(
    x: mx.array,
    B: mx.array,
    C: mx.array,
    z: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
) -> tuple[mx.array, mx.array]:
    """Path C forward returning ``(y, h_last)``.

    The VJP delegates to the TileLang backward kernel and uses the same
    ``out_idx`` output policy as the y-only surface.
    """

    return mamba3_mimo_fwd_path_c(x, B, C, z, A, dt, D, h0)


@mamba3_mimo_apply_with_state_path_c.vjp
def _mamba3_mimo_apply_with_state_path_c_vjp(
    primals: tuple[mx.array, ...],
    cotangent: tuple[mx.array, mx.array],
    output: tuple[mx.array, mx.array],
) -> tuple[mx.array, ...]:
    del output
    x, B, C, z, A, dt, D, h0 = primals
    dy = cotangent[0]
    return mamba3_mimo_bwd_path_c(dy, x, B, C, z, A, dt, D, h0)


@mx.custom_function
def mamba3_mimo_apply_with_state_path_c_fwd_path_b_bwd(
    x: mx.array,
    B: mx.array,
    C: mx.array,
    z: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
) -> tuple[mx.array, mx.array]:
    """Hybrid AUTO surface: Path C TileLang fwd, Path B Metal bwd.

    This is intentionally separate from forced Path C. The forward is only
    selected by the dispatcher after the rule/Z3/bench-receipt gate accepts the
    exact shape. The backward remains the production Path B VJP until Path C
    bwd has a checked-in no-worse receipt.
    """

    return mamba3_mimo_fwd_path_c(x, B, C, z, A, dt, D, h0)


@mamba3_mimo_apply_with_state_path_c_fwd_path_b_bwd.vjp
def _mamba3_mimo_apply_with_state_path_c_fwd_path_b_bwd_vjp(
    primals: tuple[mx.array, ...],
    cotangent: tuple[mx.array, mx.array],
    output: tuple[mx.array, mx.array],
) -> tuple[mx.array, ...]:
    from cppmega_mlx.nn._tilelang.mamba3 import mamba3_mimo_bwd_metal

    del output
    x, B, C, z, A, dt, D, h0 = primals
    dy = cotangent[0]
    return mamba3_mimo_bwd_metal(dy, x, B, C, z, A, dt, D, h0)


# Convenience: dump the lowered MSL for the bench shape so reviewers can diff
# Path B's hand-written MSL against Path C's machine-emitted MSL without
# having to re-run the lowering pipeline.
def dump_lowered_fwd_msl(
    *,
    batch: int,
    seq: int,
    heads: int,
    headdim: int,
    state: int,
    dtype: str = "float32",
) -> str:
    """Return the raw lowered MSL for the Path C forward kernel.

    Used by ``scripts/bench_tilelang_mamba3_path_c.py`` to write the
    ``docs/tilelang_ports/mamba3_path_c_lowered.metal`` artifact.
    """

    kernel, lowering = _fwd_kernel_for(
        batch,
        seq,
        heads,
        headdim,
        state,
        dtype,
        dtype,
        dtype,
        dtype,
        dtype,
        dtype,
        dtype,
        dtype,
        dtype,
        dtype,
        return_msl=True,
    )
    del kernel
    return _source_with_reused_scalar_bindings(lowering)


def dump_lowered_bwd_msl(
    *,
    batch: int,
    seq: int,
    heads: int,
    headdim: int,
    state: int,
    dtype: str = "float32",
) -> str:
    """Return the raw lowered MSL for the production Path C backward kernel."""

    snapshot_dtype = "bfloat16" if dtype == "bfloat16" else "float32"
    partial_dtypes = _bwd_partial_dtypes_for_input_dtypes(
        {
            "dy": dtype,
            "x": dtype,
            "B": dtype,
            "C": dtype,
            "z": dtype,
            "A": dtype,
            "dt": dtype,
            "D": dtype,
            "h0": dtype,
        }
    )
    kernel, lowering = _bwd_partial_kernel_for_state_snapshots(
        batch,
        seq,
        heads,
        headdim,
        state,
        dtype,
        dtype,
        dtype,
        dtype,
        dtype,
        dtype,
        dtype,
        dtype,
        dtype,
        dtype,
        dtype,
        partial_dtypes[0],
        partial_dtypes[1],
        partial_dtypes[2],
        partial_dtypes[3],
        partial_dtypes[4],
        dtype,
        h_snap_dtype=snapshot_dtype,
    )
    del kernel
    return _source_with_reused_scalar_bindings(lowering)


def _clear_mamba3_path_c_caches() -> None:
    """Release cached TileLang kernels before native leak checkers run."""

    for cached_fn in (
        mamba3_path_c_schedule_plan,
        _fwd_kernel_for,
        _bwd_state_snapshots_kernel_for,
        _bwd_simd_reduce_kernel_for,
        _bwd_simd_reduce_kernel_for_state_snapshots,
        _bwd_partial_kernel_for_state_snapshots,
        _bwd_scratch_partial_kernel_for,
        _bwd_partial_reduce_kernel_for,
    ):
        cached_fn.cache_clear()


atexit.register(_clear_mamba3_path_c_caches)


__all__ = [
    "Mamba3PathCSchedulePlan",
    "Mamba3PathCStatus",
    "Mamba3PathCZ3Policy",
    "dump_lowered_bwd_msl",
    "dump_lowered_fwd_msl",
    "mamba3_mimo_apply_path_c",
    "mamba3_mimo_apply_training_path_c",
    "mamba3_mimo_apply_with_state_path_c",
    "mamba3_mimo_apply_with_state_training_path_c",
    "mamba3_mimo_apply_with_state_path_c_fwd_path_b_bwd",
    "mamba3_path_c_auto_fwd_path_b_bwd_allowed",
    "mamba3_path_c_auto_mode_for_inputs",
    "mamba3_path_c_receipt_allows_auto_promotion",
    "mamba3_path_c_receipt_auto_mode",
    "mamba3_path_c_schedule_plan",
    "mamba3_mimo_bwd_path_c",
    "mamba3_mimo_fwd_path_c",
    "mamba3_mimo_path_c_status",
]

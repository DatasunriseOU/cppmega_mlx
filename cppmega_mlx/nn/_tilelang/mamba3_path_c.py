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
from cppmega_mlx.nn._tilelang._mlx_runtime import (
    NativeTileLangRuntimeError,
    _validate_owner_result,
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
    "dB_lane_grad",
    "dC_lane_grad",
    "dA_lane_grad",
    "ddt_lane_grad",
    "dD_lane_grad",
    "dh0",
)
_BWD_PARTIAL_OUTPUT_IDX = (9, 10, 11, 12, 13, 14, 15, 16)
_BWD_SCRATCH_OUTPUT_NAMES = (
    "dx",
    "dz",
    "dB_lane_grad",
    "dC_lane_grad",
    "dA_lane_grad",
    "ddt_lane_grad",
    "dD_lane_grad",
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
    if mode not in {
        "path_c_fwd_path_b_bwd",
        "path_c_fwd_bwd",
        "path_c_fwd_path_c_bwd",
    }:
        return "path_b"
    if decision.get("selected_forward_kernel") != "path_c_tilelang_dsl":
        return "path_b"
    expected_bwd_by_mode = {
        "path_c_fwd_bwd": "path_c_tilelang_dsl",
        "path_c_fwd_path_b_bwd": "metal_kernel_bwd_v1",
        "path_c_fwd_path_c_bwd": "path_c_chunked_bwd_v1",
    }
    expected_bwd = expected_bwd_by_mode[mode]
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

    if mode in {"path_c_fwd_bwd", "path_c_fwd_path_c_bwd"}:
        # Both Path-C-backward modes must clear the strict bwd + fwd+bwd ratio
        # gate (fail-closed). The chunked mode additionally needs the no-worse
        # chunked-backward receipt before AUTO ever promotes it.
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
        # CUDA EAGER branch: forward runs a TileLang-CUDA kernel on a
        # non-Metal host. Backward stays on the pure-MLX reference VJP
        # (see mamba3_mimo_fwd_path_c CUDA branch + TODO).
        from cppmega_mlx.nn._tilelang._cuda_eager import cuda_eager_available

        cuda_ok, cuda_reason = cuda_eager_available()
        if cuda_ok:
            return Mamba3PathCStatus(
                available=True,
                reason=f"Mamba3 TileLang-CUDA EAGER fwd ready ({cuda_reason})",
            )
        return Mamba3PathCStatus(
            available=False,
            reason=(
                "MLX Metal backend is not available on the default GPU "
                f"device and CUDA EAGER path unavailable: {cuda_reason}"
            ),
        )
    ok, reason = _tilelang_available()
    if not ok:
        return Mamba3PathCStatus(available=False, reason=reason)
    try:
        fwd_kernel, fwd_lowering = _fwd_kernel_for(1, 4, 1, 2, 4, return_msl=True)
        bwd_kernel, bwd_lowering = _bwd_lane_grad_kernel_for_state_snapshots(
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


def _bwd_lane_grad_dtypes_for_input_dtypes(
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
def _make_fwd_prim_func(
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
) -> Any:
    """Build the raw Path C Mamba3 forward PrimFunc before lowering."""

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
        with T.Kernel(T.ceildiv(LANES, THREADS), threads=THREADS) as _bx:
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

    return fwd


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

    del return_msl
    fwd = _make_fwd_prim_func(
        BATCH,
        SEQ,
        HEADS,
        HEADDIM,
        STATE,
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
def _make_fwd_with_snapshots_prim_func(
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
) -> Any:
    """Build the raw Path C Mamba3 training PrimFunc before lowering."""

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

    return fwd_with_snapshots


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

    fwd_with_snapshots = _make_fwd_with_snapshots_prim_func(
        BATCH,
        SEQ,
        HEADS,
        HEADDIM,
        STATE,
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
    h0_dtype: str = "float32",
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
def _bwd_lane_grad_kernel_for_state_snapshots(
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
    def bwd_lane_grad(
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
        dB_lane_grad: T.Tensor((BATCH, SEQ, HEADS, STATE, HEADDIM), dB_dtype),
        dC_lane_grad: T.Tensor((BATCH, SEQ, HEADS, STATE, HEADDIM), dC_dtype),
        dA_lane_grad: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), dA_dtype),
        ddt_lane_grad: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), ddt_dtype),
        dD_lane_grad: T.Tensor((BATCH, HEADS, HEADDIM), dD_dtype),
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
                        dC_lane_grad[b, t, h, n, p] = T.cast(
                            d_y_skipped * h_state[n],
                            dC_dtype,
                        )
                        dB_lane_grad[b, t, h, n, p] = T.cast(
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
                    dA_lane_grad[b, t, h, p] = T.cast(d_logdecay * dt_val, dA_dtype)
                    ddt_lane_grad[b, t, h, p] = T.cast(d_logdecay * A_val, ddt_dtype)

                for n in T.serial(STATE):
                    dh0[b, h, p, n] = T.cast(dh[n], dh0_dtype)
                dD_lane_grad[b, h, p] = T.cast(dD_acc, dD_dtype)

    artifact = dispatch_lower(bwd_lane_grad, target="metal", return_msl=True)
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
        bwd_lane_grad,
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
        dB_lane_grad: T.Tensor((BATCH, SEQ, HEADS, STATE, HEADDIM), dB_dtype),
        dC_lane_grad: T.Tensor((BATCH, SEQ, HEADS, STATE, HEADDIM), dC_dtype),
        dA_lane_grad: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), dA_dtype),
        ddt_lane_grad: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), ddt_dtype),
        dD_lane_grad: T.Tensor((BATCH, HEADS, HEADDIM), dD_dtype),
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
                        dC_lane_grad[b, t, h, n, p] = T.cast(
                            d_y_skipped * h_state[n],
                            dC_dtype,
                        )
                        dB_lane_grad[b, t, h, n, p] = T.cast(
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
                    dA_lane_grad[b, t, h, p] = T.cast(d_logdecay * dt_val, dA_dtype)
                    ddt_lane_grad[b, t, h, p] = T.cast(d_logdecay * A_val, ddt_dtype)

                for n in T.serial(STATE):
                    dh0[b, h, p, n] = T.cast(dh[n], dh0_dtype)
                dD_lane_grad[b, h, p] = T.cast(dD_acc, dD_dtype)

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
def _bwd_lane_grad_reduce_kernel_for(
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
    def bwd_lane_grad_reduce(
        dB_lane_grad: T.Tensor((BATCH, SEQ, HEADS, STATE, HEADDIM), dB_dtype),
        dC_lane_grad: T.Tensor((BATCH, SEQ, HEADS, STATE, HEADDIM), dC_dtype),
        dA_lane_grad: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), dA_dtype),
        ddt_lane_grad: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), ddt_dtype),
        dD_lane_grad: T.Tensor((BATCH, HEADS, HEADDIM), dD_dtype),
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
                    dB_sum += T.cast(dB_lane_grad[b, t, h, n, p], "float32")
                    dC_sum += T.cast(dC_lane_grad[b, t, h, n, p], "float32")
                dB[b, t, h, n] = T.cast(dB_sum, out_dB_dtype)
                dC[b, t, h, n] = T.cast(dC_sum, out_dC_dtype)

            if global_lane < ADT_LANES:
                h = global_lane % HEADS
                t = (global_lane // HEADS) % SEQ
                b = global_lane // (HEADS * SEQ)
                dA_sum = T.alloc_var(T.float32, init=0.0)
                ddt_sum = T.alloc_var(T.float32, init=0.0)
                for p in T.serial(HEADDIM):
                    dA_sum += T.cast(dA_lane_grad[b, t, h, p], "float32")
                    ddt_sum += T.cast(ddt_lane_grad[b, t, h, p], "float32")
                dA[b, t, h] = T.cast(dA_sum, out_dA_dtype)
                ddt[b, t, h] = T.cast(ddt_sum, out_ddt_dtype)

            if global_lane < D_LANES:
                h = global_lane
                dD_sum = T.alloc_var(T.float32, init=0.0)
                for b in T.serial(BATCH):
                    for p in T.serial(HEADDIM):
                        dD_sum += T.cast(dD_lane_grad[b, h, p], "float32")
                dD[h] = T.cast(dD_sum, out_dD_dtype)

    artifact = dispatch_lower(bwd_lane_grad_reduce, target="metal", return_msl=True)
    if hasattr(artifact, "_tilelang_engine_target"):
        raise MSLDispatchUnsupported("Mamba3 Path C requires TileLang MSL extraction metadata")
    lowering = cast(_msl_transform.TileLangMSLLowering, artifact)
    input_names = [
        name for name in lowering.buffer_param_names if name not in _BWD_REDUCE_OUTPUT_NAMES
    ]
    if set(input_names) != {
        "dB_lane_grad",
        "dC_lane_grad",
        "dA_lane_grad",
        "ddt_lane_grad",
        "dD_lane_grad",
    }:
        raise MSLDispatchUnsupported(
            "unexpected Mamba3 Path C partial reducer buffer signature: "
            + ", ".join(lowering.buffer_param_names)
        )
    import tilelang

    kernel = tilelang.compile(
        bwd_lane_grad_reduce,
        target=_msl_transform._as_metal_target("metal"),
        execution_backend="tvm_ffi",
        out_idx=list(_BWD_REDUCE_OUTPUT_IDX),
    )
    return kernel, lowering


@lru_cache(maxsize=128)
def _bwd_lane_grad_reduce_threaded_kernel_for(
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
    def bwd_lane_grad_reduce_threaded(
        dB_lane_grad: T.Tensor((BATCH, SEQ, HEADS, STATE, HEADDIM), dB_dtype),
        dC_lane_grad: T.Tensor((BATCH, SEQ, HEADS, STATE, HEADDIM), dC_dtype),
        dA_lane_grad: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), dA_dtype),
        ddt_lane_grad: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), ddt_dtype),
        dD_lane_grad: T.Tensor((BATCH, HEADS, HEADDIM), dD_dtype),
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
                dB_acc[0] = T.cast(dB_lane_grad[b, t, h, n, p], "float32")
                dC_acc[0] = T.cast(dC_lane_grad[b, t, h, n, p], "float32")
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
                dA_acc[0] = T.cast(dA_lane_grad[b, t, h, p], "float32")
                ddt_acc[0] = T.cast(ddt_lane_grad[b, t, h, p], "float32")
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
                    dD_acc[0] += T.cast(dD_lane_grad[b, h, p], "float32")
                T.thread_reduce(dD_acc[0], dD_reduced[0], reduce_axis, op="sum")
                if p == 0:
                    dD[h] = T.cast(dD_reduced[0], out_dD_dtype)

    artifact = dispatch_lower(bwd_lane_grad_reduce_threaded, target="metal", return_msl=True)
    if hasattr(artifact, "_tilelang_engine_target"):
        raise MSLDispatchUnsupported("Mamba3 Path C requires TileLang MSL extraction metadata")
    lowering = cast(_msl_transform.TileLangMSLLowering, artifact)
    input_names = [
        name for name in lowering.buffer_param_names if name not in _BWD_REDUCE_OUTPUT_NAMES
    ]
    if set(input_names) != {
        "dB_lane_grad",
        "dC_lane_grad",
        "dA_lane_grad",
        "ddt_lane_grad",
        "dD_lane_grad",
    }:
        raise MSLDispatchUnsupported(
            "unexpected Mamba3 Path C threaded partial reducer buffer signature: "
            + ", ".join(lowering.buffer_param_names)
        )
    import tilelang

    kernel = tilelang.compile(
        bwd_lane_grad_reduce_threaded,
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


def _materialize_contiguous_inputs(
    op_name: str,
    *named_arrays: tuple[str, mx.array],
) -> tuple[mx.array, ...]:
    """Materialize each tvm-ffi input as a contiguous, DLPack-exportable buffer.

    Path C dispatches caller-owned MLX arrays straight across the tvm-ffi
    boundary, where ``dlpack_to_tvm_tensor`` rejects any non-contiguous layout
    with ``FromDLPack: Tensor is not contiguous``. The Mamba3 producer (the
    model layer feeding ``_dispatch_mamba3_scan``) can legitimately hand Path C
    strided/broadcast views — e.g. RoPE state-dim transposes/reshapes or
    group->head broadcasts upstream of the scan. Those are *layout* views of an
    otherwise supported buffer, so the correct fix is to materialize the
    contiguous buffer the ABI requires at this choke point, not to fall back to
    another path (RULE #1).

    This is layout-only: ``mx.contiguous`` is a no-op on already-contiguous
    arrays and never changes dtype, so the no-hidden-cast contract is preserved
    (unsupported dtypes are still rejected by
    :func:`_require_supported_no_hidden_casts`). Anything ``mx.contiguous``
    genuinely cannot make DLPack-exportable still surfaces loudly at the
    dispatch boundary via :func:`_raise_if_dlpack_boundary_failure`.
    """

    del op_name
    return tuple(mx.contiguous(array) for _name, array in named_arrays)


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


def _reduce_bwd_lane_grads_path_c_kernel(
    dB_lane_grad: mx.array,
    dC_lane_grad: mx.array,
    dA_lane_grad: mx.array,
    ddt_lane_grad: mx.array,
    dD_lane_grad: mx.array,
    *,
    output_dtypes: tuple[str, str, str, str, str] | None = None,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Reduce bwd partial gradients over P with a serial TileLang kernel."""

    if len(dB_lane_grad.shape) != 5:
        raise ValueError(
            "mamba3_mimo_bwd_path_c partial reducer expects dB_lane_grad shape "
            "(B, T, H, N, P)"
        )
    batch, seq, heads, state, headdim = map(int, dB_lane_grad.shape)
    expected_c = (batch, seq, heads, state, headdim)
    expected_lane = (batch, seq, heads, headdim)
    expected_d = (batch, heads, headdim)
    if tuple(dC_lane_grad.shape) != expected_c:
        raise ValueError(
            "mamba3_mimo_bwd_path_c partial reducer expects dC_lane_grad shape "
            f"{expected_c}; got {tuple(dC_lane_grad.shape)}"
        )
    if tuple(dA_lane_grad.shape) != expected_lane:
        raise ValueError(
            "mamba3_mimo_bwd_path_c partial reducer expects dA_lane_grad shape "
            f"{expected_lane}; got {tuple(dA_lane_grad.shape)}"
        )
    if tuple(ddt_lane_grad.shape) != expected_lane:
        raise ValueError(
            "mamba3_mimo_bwd_path_c partial reducer expects ddt_lane_grad shape "
            f"{expected_lane}; got {tuple(ddt_lane_grad.shape)}"
        )
    if tuple(dD_lane_grad.shape) != expected_d:
        raise ValueError(
            "mamba3_mimo_bwd_path_c partial reducer expects dD_lane_grad shape "
            f"{expected_d}; got {tuple(dD_lane_grad.shape)}"
        )
    if min(batch, seq, heads, headdim, state) <= 0:
        raise RuntimeError(
            "mamba3_mimo_bwd_path_c partial reducer is not dispatchable for "
            "non-positive shapes"
        )

    dtypes = _require_supported_no_hidden_casts(
        "mamba3_mimo_bwd_path_c partial reducer",
        ("dB_lane_grad", dB_lane_grad),
        ("dC_lane_grad", dC_lane_grad),
        ("dA_lane_grad", dA_lane_grad),
        ("ddt_lane_grad", ddt_lane_grad),
        ("dD_lane_grad", dD_lane_grad),
    )
    if output_dtypes is None:
        reduce_output_dtypes = (
            dtypes["dB_lane_grad"],
            dtypes["dC_lane_grad"],
            dtypes["dA_lane_grad"],
            dtypes["ddt_lane_grad"],
            dtypes["dD_lane_grad"],
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
        kernel, lowering = _bwd_lane_grad_reduce_kernel_for(
            batch,
            seq,
            heads,
            headdim,
            state,
            dtypes["dB_lane_grad"],
            dtypes["dC_lane_grad"],
            dtypes["dA_lane_grad"],
            dtypes["ddt_lane_grad"],
            dtypes["dD_lane_grad"],
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
        out_list = kernel(dB_lane_grad, dC_lane_grad, dA_lane_grad, ddt_lane_grad, dD_lane_grad)
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


def _reduce_bwd_lane_grads_path_c_fast_kernel(
    dB_lane_grad: mx.array,
    dC_lane_grad: mx.array,
    dA_lane_grad: mx.array,
    ddt_lane_grad: mx.array,
    dD_lane_grad: mx.array,
    *,
    output_dtypes: tuple[str, str, str, str, str] | None = None,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Use the fastest supported standalone TileLang P-reducer for partial grads."""

    if len(dB_lane_grad.shape) == 5:
        headdim = int(dB_lane_grad.shape[4])
    else:
        headdim = 0
    # Full-shape profiling is dtype-sensitive: current Metal thread_reduce is
    # faster for FP32 partials, while the simple serial reducer wins for compact
    # BF16 dB/dC partials where conversion and reduction overhead dominate.
    if (
        dB_lane_grad.dtype == mx.float32
        and dC_lane_grad.dtype == mx.float32
        and _is_power_of_two(headdim)
        and headdim <= _MAX_THREADS
    ):
        return _reduce_bwd_lane_grads_path_c_threaded_kernel(
            dB_lane_grad,
            dC_lane_grad,
            dA_lane_grad,
            ddt_lane_grad,
            dD_lane_grad,
            output_dtypes=output_dtypes,
        )
    return _reduce_bwd_lane_grads_path_c_kernel(
        dB_lane_grad,
        dC_lane_grad,
        dA_lane_grad,
        ddt_lane_grad,
        dD_lane_grad,
        output_dtypes=output_dtypes,
    )


def _reduce_bwd_lane_grads_path_c_threaded_kernel(
    dB_lane_grad: mx.array,
    dC_lane_grad: mx.array,
    dA_lane_grad: mx.array,
    ddt_lane_grad: mx.array,
    dD_lane_grad: mx.array,
    *,
    output_dtypes: tuple[str, str, str, str, str] | None = None,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Reduce bwd partial gradients over P with a threaded TileLang kernel."""

    if len(dB_lane_grad.shape) != 5:
        raise ValueError(
            "mamba3_mimo_bwd_path_c threaded partial reducer expects "
            "dB_lane_grad shape (B, T, H, N, P)"
        )
    batch, seq, heads, state, headdim = map(int, dB_lane_grad.shape)
    expected_c = (batch, seq, heads, state, headdim)
    expected_lane = (batch, seq, heads, headdim)
    expected_d = (batch, heads, headdim)
    if tuple(dC_lane_grad.shape) != expected_c:
        raise ValueError(
            "mamba3_mimo_bwd_path_c threaded partial reducer expects "
            f"dC_lane_grad shape {expected_c}; got {tuple(dC_lane_grad.shape)}"
        )
    if tuple(dA_lane_grad.shape) != expected_lane:
        raise ValueError(
            "mamba3_mimo_bwd_path_c threaded partial reducer expects "
            f"dA_lane_grad shape {expected_lane}; got {tuple(dA_lane_grad.shape)}"
        )
    if tuple(ddt_lane_grad.shape) != expected_lane:
        raise ValueError(
            "mamba3_mimo_bwd_path_c threaded partial reducer expects "
            f"ddt_lane_grad shape {expected_lane}; got {tuple(ddt_lane_grad.shape)}"
        )
    if tuple(dD_lane_grad.shape) != expected_d:
        raise ValueError(
            "mamba3_mimo_bwd_path_c threaded partial reducer expects "
            f"dD_lane_grad shape {expected_d}; got {tuple(dD_lane_grad.shape)}"
        )
    if min(batch, seq, heads, headdim, state) <= 0:
        raise RuntimeError(
            "mamba3_mimo_bwd_path_c threaded partial reducer is not "
            "dispatchable for non-positive shapes"
        )

    dtypes = _require_supported_no_hidden_casts(
        "mamba3_mimo_bwd_path_c threaded partial reducer",
        ("dB_lane_grad", dB_lane_grad),
        ("dC_lane_grad", dC_lane_grad),
        ("dA_lane_grad", dA_lane_grad),
        ("ddt_lane_grad", ddt_lane_grad),
        ("dD_lane_grad", dD_lane_grad),
    )
    if output_dtypes is None:
        reduce_output_dtypes = (
            dtypes["dB_lane_grad"],
            dtypes["dC_lane_grad"],
            dtypes["dA_lane_grad"],
            dtypes["ddt_lane_grad"],
            dtypes["dD_lane_grad"],
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
        kernel, lowering = _bwd_lane_grad_reduce_threaded_kernel_for(
            batch,
            seq,
            heads,
            headdim,
            state,
            dtypes["dB_lane_grad"],
            dtypes["dC_lane_grad"],
            dtypes["dA_lane_grad"],
            dtypes["ddt_lane_grad"],
            dtypes["dD_lane_grad"],
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
        out_list = kernel(dB_lane_grad, dC_lane_grad, dA_lane_grad, ddt_lane_grad, dD_lane_grad)
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

    # CUDA EAGER branch: when Metal is unavailable, run the vendored
    # CUDA-safe Mamba3 fwd kernel (same recurrence math as the Metal
    # prim_func). owner-output ``out=`` staging is Metal/tvm-ffi only;
    # the CUDA branch returns freshly built arrays.
    if not can_run_metal():
        from cppmega_mlx.nn._tilelang._cuda_eager import (
            cuda_eager_available,
            mamba3_mimo_fwd_cuda_eager,
        )

        cuda_ok, cuda_reason = cuda_eager_available()
        if not cuda_ok:
            raise RuntimeError(
                f"mamba3_mimo_fwd_path_c CUDA EAGER unavailable: {cuda_reason}"
            )
        if out is not None:
            raise RuntimeError(
                "mamba3_mimo_fwd_path_c owner-output route is Metal/tvm-ffi "
                "only; the CUDA EAGER branch returns fresh arrays"
            )
        result = mamba3_mimo_fwd_cuda_eager(x, B, C, z, A, dt, D, h0)
        if result is None:
            raise RuntimeError("mamba3_mimo_fwd_path_c CUDA EAGER dispatch failed")
        return result

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
    # The producer may hand Path C strided/broadcast views (RoPE state-dim
    # transposes, group->head broadcasts, slices). The tvm-ffi DLPack import
    # rejects non-contiguous buffers, so materialize the contiguous layout the
    # ABI requires here (layout-only, dtype-preserving; RULE #1: no fallback).
    x, B, C, z, A, dt, D, h0 = _materialize_contiguous_inputs(
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
        try:
            y, h_last = cast(
                tuple[mx.array, mx.array],
                _validate_owner_result(out_list, owner_outputs),
            )
        except NativeTileLangRuntimeError as exc:
            raise RuntimeError(
                "Mamba3 Path C fwd tvm-ffi did not return caller-owned outputs"
            ) from exc
        del lowering
        return y, h_last
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
    # See mamba3_mimo_fwd_path_c: materialize contiguous tvm-ffi inputs so a
    # strided/broadcast producer view does not trip the DLPack contiguity check
    # (layout-only, dtype-preserving; RULE #1: no fallback).
    x, B, C, z, A, dt, D, h0 = _materialize_contiguous_inputs(
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
        partial_dtypes = _bwd_lane_grad_dtypes_for_input_dtypes(dtypes)
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
        kernel, lowering = _bwd_lane_grad_kernel_for_state_snapshots(
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
    dx_pc, dz_pc, dB_lane_grad, dC_lane_grad, dA_lane_grad, ddt_lane_grad, dD_lane_grad, dh0_pc = out_list
    del lowering
    return (
        dx_pc,
        dz_pc,
        dB_lane_grad,
        dC_lane_grad,
        dA_lane_grad,
        ddt_lane_grad,
        dD_lane_grad,
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
        dB_lane_grad,
        dC_lane_grad,
        dA_lane_grad,
        ddt_lane_grad,
        dD_lane_grad,
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
    dB_pc, dC_pc, dA_pc, ddt_pc, dD_pc = _reduce_bwd_lane_grads_path_c_fast_kernel(
        dB_lane_grad,
        dC_lane_grad,
        dA_lane_grad,
        ddt_lane_grad,
        dD_lane_grad,
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
        dB_lane_grad,
        dC_lane_grad,
        dA_lane_grad,
        ddt_lane_grad,
        dD_lane_grad,
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
    dB_pc, dC_pc, dA_pc, ddt_pc, dD_pc = _reduce_bwd_lane_grads_path_c_fast_kernel(
        dB_lane_grad,
        dC_lane_grad,
        dA_lane_grad,
        ddt_lane_grad,
        dD_lane_grad,
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
        dB_lane_grad,
        dC_lane_grad,
        dA_lane_grad,
        ddt_lane_grad,
        dD_lane_grad,
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
    dB_pc, dC_pc, dA_pc, ddt_pc, dD_pc = _reduce_bwd_lane_grads_path_c_fast_kernel(
        dB_lane_grad,
        dC_lane_grad,
        dA_lane_grad,
        ddt_lane_grad,
        dD_lane_grad,
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
        dB_lane_grad,
        dC_lane_grad,
        dA_lane_grad,
        ddt_lane_grad,
        dD_lane_grad,
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
    dB_pc, dC_pc, dA_pc, ddt_pc, dD_pc = _reduce_bwd_lane_grads_path_c_threaded_kernel(
        dB_lane_grad,
        dC_lane_grad,
        dA_lane_grad,
        ddt_lane_grad,
        dD_lane_grad,
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
        dB_lane_grad,
        dC_lane_grad,
        dA_lane_grad,
        ddt_lane_grad,
        dD_lane_grad,
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
    dB_pc = mx.sum(dB_lane_grad, axis=4)
    dC_pc = mx.sum(dC_lane_grad, axis=4)
    dA_pc = mx.sum(dA_lane_grad, axis=3)
    ddt_pc = mx.sum(ddt_lane_grad, axis=3)
    dD_pc = mx.sum(dD_lane_grad, axis=(0, 2))
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
    partial_dtypes = _bwd_lane_grad_dtypes_for_input_dtypes(dtypes)
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
        dB_lane_grad,
        dC_lane_grad,
        dA_lane_grad,
        ddt_lane_grad,
        dD_lane_grad,
        dh0_pc,
    ) = _mamba3_mimo_bwd_path_c_scratch_partial_outputs(dy, x, B, C, z, A, dt, D, h0)
    dB_pc, dC_pc, dA_pc, ddt_pc, dD_pc = _reduce_bwd_lane_grads_path_c_fast_kernel(
        dB_lane_grad,
        dC_lane_grad,
        dA_lane_grad,
        ddt_lane_grad,
        dD_lane_grad,
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
        dB_lane_grad,
        dC_lane_grad,
        dA_lane_grad,
        ddt_lane_grad,
        dD_lane_grad,
        dh0_pc,
    ) = _mamba3_mimo_bwd_path_c_scratch_partial_outputs(dy, x, B, C, z, A, dt, D, h0)
    dB_pc = mx.sum(dB_lane_grad, axis=4)
    dC_pc = mx.sum(dC_lane_grad, axis=4)
    dA_pc = mx.sum(dA_lane_grad, axis=3)
    ddt_pc = mx.sum(ddt_lane_grad, axis=3)
    dD_pc = mx.sum(dD_lane_grad, axis=(0, 2))
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

    # CUDA EAGER branch: on a non-Metal host the Metal SIMD lane-grad
    # backward kernel cannot lower. Use the vendored TileLang-CUDA backward
    # kernel (mamba3_mimo_bwd_cuda_eager) which ports the production single-
    # kernel scratch reverse scan. If that kernel is unavailable or raises,
    # fall back to MLX autograd over the pure-MLX reference scan so
    # correctness never regresses. Returns grads for all 8 inputs
    # (x, B, C, z, A, dt, D, h0), matching the kernel/VJP ABI.
    if not can_run_metal():
        if out is not None:
            raise RuntimeError(
                "mamba3_mimo_bwd_path_c owner-output route is Metal/tvm-ffi "
                "only; the CUDA EAGER backward returns fresh arrays"
            )
        from cppmega_mlx.nn._tilelang._cuda_eager import (
            cuda_eager_available,
            mamba3_mimo_bwd_cuda_eager,
        )

        cuda_ok, _cuda_reason = cuda_eager_available()
        if cuda_ok:
            grads = mamba3_mimo_bwd_cuda_eager(dy, x, B, C, z, A, dt, D, h0)
            if grads is not None:
                return grads

        from cppmega_mlx.nn._tilelang.mamba3 import mamba3_mimo_reference

        def _ref(x_, B_, C_, z_, A_, dt_, D_, h0_):
            y_, _ = mamba3_mimo_reference(x_, B_, C_, z_, A_, dt_, D_, h0_)
            return y_

        primals = [x, B, C, z, A, dt, D, h0]
        _y, vjps = mx.vjp(_ref, primals, [dy])
        return tuple(vjps)

    # Materialize contiguous tvm-ffi inputs once at the Metal backward entry so
    # every downstream Path C bwd kernel (snapshot, lane-grad, reducer) receives
    # DLPack-exportable buffers even when the VJP cotangent or a producer view
    # is strided/broadcast (layout-only, dtype-preserving; RULE #1: no
    # fallback). Unsupported dtypes are still rejected downstream.
    dy, x, B, C, z, A, dt, D, h0 = _materialize_contiguous_inputs(
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


# --------------------------------------------------------------------------- #
# path_c_fwd_path_c_bwd — Path C DSL forward + the chunked B2->B1->B0 backward.  #
#                                                                               #
# The forward is the UNCHANGED Path C lane-scan fwd (mamba3_mimo_fwd_path_c). It #
# additionally runs the chunked-forward builders F0/F1 to materialize the       #
# intermediates (cb, dA_cumsum, prev_states, y) the chunked backward consumes,   #
# and STASHES them as extra custom_function outputs (the MLX VJP residual). The   #
# .vjp then drives the validated chunked B2->B1->B0 chain (the exact call        #
# sequence + 8-grad assembly ported from                                         #
# tests/test_mamba3_chunked_backward_b0b1b2.py::test_chained_backward_*) and      #
# returns the 8-grad tuple matching the MSL mamba3_mimo_bwd_metal surface         #
# (dx, dB, dC, dz, dA, ddt, dD, dh0).                                            #
#                                                                               #
# ABI (verified by reading the prim signatures + an MLX tvm_ffi probe):          #
#   * All chunked prims declare dtype=float16 inputs and float32 accumulators.    #
#     The kernel inputs (x,B,C,z,A,dt,D,dout,cb,dA_cumsum,y,dt_k) are cast to     #
#     fp16; the accumulators (summary_states, prev_states, dstates, dinp, dx,     #
#     dB, dlog, ddt, dC, dz, dh0, ...) come back fp32. This is the SAME fp16-in   #
#     regime the chained-backward test validated < 1e-3 against the fp32 proto.   #
#   * Each builder compiles with out_idx (F0[5,6,7], F1[3,4], B2[11..17],         #
#     B1[4,5,6], B0[9,10,11,12]); the MLX route returns those as fresh ZEROED     #
#     output arrays. B0's dx is therefore a FRESH inp-path grad — the .vjp adds    #
#     B2's dx (D-skip + dY*C path) to B0's dx to form the total dx (the test's     #
#     dx_full = dx_m.clone() seeds B0 with B2's dx; out_idx forces an MLX add).    #
#   * At the _dispatch_mamba3_scan surface ngroups == nheads (B/C are already      #
#     broadcast per-head, A is (b,seq,H)); the builders run with ngroups=H so      #
#     cb's group axis is the head axis and dB stays per-head (b,seq,H,N) — no      #
#     head->group reduction is needed (RISK #3 dissolves at this surface).        #
#                                                                               #
# REGIME GUARD (RULE #1 — fail loud, no silent wrong path): the chunked kernels    #
# reconstruct the per-position log-decay as A_kernel[h]*dt[l] from a PER-HEAD      #
# scalar A (T.Tensor((nheads,))). The Mamba3 production surface's A is PER-        #
# POSITION (b,seq,H). When A varies across positions the kernels' ddt = dlog*A[h]  #
# term cannot represent dlog*A_pos[l] exactly. Rather than silently emit a wrong    #
# ddt this mode RAISES when A is not per-head-constant (the validated regime), so   #
# the caller sees WHERE+WHAT instead of degraded grads. Full per-position-A         #
# parity is the NEXT phase (it needs a kernel-ABI change to accept per-position A   #
# / a precomputed dA_cumsum override); this mode is selected only explicitly via    #
# CPPMEGA_MAMBA3_PATH_C_BWD (the receipt keeps AUTO on path_b) for the proving      #
# phase. NO try/except->MSL fallback anywhere (the dispatcher selects up front).    #
# --------------------------------------------------------------------------- #

# Metal compile target reused for every chunked-chain kernel.
_CHUNKED_METAL_TARGET = _msl_transform._as_metal_target("metal")


@lru_cache(maxsize=64)
def _chunked_fwd_f0_kernel(b: int, s: int, c: int, g: int, h: int, p: int, n: int) -> Any:
    """F0 (chunk_precompute) compiled for the MLX tvm_ffi route -> (cb,dA_cumsum,summary)."""
    import tilelang

    from cppmega_mlx.nn._tilelang.mamba3_chunked_precompute_core import (
        chunk_precompute_fwd_metal_prim,
    )

    return tilelang.compile(
        chunk_precompute_fwd_metal_prim(b, s, c, g, h, p, n),
        target=_CHUNKED_METAL_TARGET,
        execution_backend="tvm_ffi",
        out_idx=[5, 6, 7],
    )


@lru_cache(maxsize=64)
def _chunked_fwd_f1_kernel(b: int, s: int, c: int, g: int, h: int, p: int, n: int) -> Any:
    """F1 (inter_chunk_recur) compiled for MLX -> (prev_states, final_state)."""
    import tilelang

    from cppmega_mlx.nn._tilelang.mamba3_chunked_precompute_core import (
        inter_chunk_recur_fwd_metal_prim,
    )

    return tilelang.compile(
        inter_chunk_recur_fwd_metal_prim(b, s, c, g, h, p, n),
        target=_CHUNKED_METAL_TARGET,
        execution_backend="tvm_ffi",
        out_idx=[3, 4],
    )


_MLX_TVM_FFI_FORCE_BOUNDARY_ENV = "TILELANG_MLX_TVM_FFI_FORCE_COMMAND_BUFFER_BOUNDARY"


def _force_chunked_command_buffer_boundary() -> None:
    """Force the MLX tvm_ffi command-buffer ordering boundary for chunked kernels.

    RULE #1 (no silent fallback; one clear, DETERMINISTIC path). The chunked
    B2/B1/B0 backward kernels accumulate several outputs via ``T.atomic_add`` into
    bridge-allocated owner buffers (dD cross-threadgroup; dx/dB single-writer but
    still read-modify-write the bridge buffer; dA_cumsum_tail intra-threadgroup).
    Each requires its zero-init blit (and the MLX input producers) to complete
    BEFORE the TVM compute encoder runs. The bridge's active-compute-encoder fast
    path dispatches the kernel WITHOUT that strict ordering, so the zero-blit /
    producers RACE the atomic reads -> intermittently garbage dD/dx/dB and flaky
    dA/ddt (MEASURED at nam56r: ~1/3 of runs FAIL > 1e-3, with rare 1e-1 blow-ups;
    with the boundary forced, 0/40 runs fail and every grad is deterministic at the
    fp32-noise floor). ``env_flag_enabled`` is read per-dispatch via ``getenv``, so
    setting this here (idempotent) takes effect for every subsequent chunked launch
    without a rebuild. We do NOT silently downgrade on failure — forcing correct
    ordering is the single correct path; if the bridge ignored it the parity test
    (test_mamba3_path_c_chunked_vs_path_b) would FAIL loudly.
    """
    if os.environ.get(_MLX_TVM_FFI_FORCE_BOUNDARY_ENV, "").strip() in ("", "0", "false", "False", "FALSE"):
        os.environ[_MLX_TVM_FFI_FORCE_BOUNDARY_ENV] = "1"


@lru_cache(maxsize=64)
def _chunked_bwd_b2_kernel(b: int, s: int, c: int, g: int, h: int, p: int, n: int) -> Any:
    """B2 (chunk_scan_combine_bwd) -> (dC,dx,dz,dchunk_states,dinp,dA_cumsum_y,dD)."""
    import tilelang

    from cppmega_mlx.nn._tilelang.mamba3_chunked_backward_core import (
        chunk_scan_combine_bwd_metal_prim,
    )

    return tilelang.compile(
        chunk_scan_combine_bwd_metal_prim(b, s, c, g, h, p, n),
        target=_CHUNKED_METAL_TARGET,
        execution_backend="tvm_ffi",
        out_idx=[11, 12, 13, 14, 15, 16, 17],
    )


@lru_cache(maxsize=64)
def _chunked_bwd_b1_kernel(b: int, s: int, c: int, g: int, h: int, p: int, n: int) -> Any:
    """B1 (inter_chunk_recur_bwd) -> (dstates, dh0, dA_cumsum_tail)."""
    import tilelang

    from cppmega_mlx.nn._tilelang.mamba3_chunked_backward_core import (
        inter_chunk_recur_bwd_metal_prim,
    )

    return tilelang.compile(
        inter_chunk_recur_bwd_metal_prim(b, s, c, g, h, p, n),
        target=_CHUNKED_METAL_TARGET,
        execution_backend="tvm_ffi",
        out_idx=[4, 5, 6],
    )


@lru_cache(maxsize=64)
def _chunked_bwd_b2b1_fused_kernel(
    b: int, s: int, c: int, g: int, h: int, p: int, n: int
) -> Any:
    """SELECTED fusion (directed search, rank 2): B2+B1 in ONE tvm_ffi kernel.

    The directed backward-fusion search (cppmega_mlx.runtime.path_c_backward_
    fusion_search) enumerates the 4 contiguous partitions of [B2,B1,B0], predicts
    feasibility (P1 MSL<=140000 / P2 threadgroup<=32768 / P3 buffer<=31 / P4
    watchdog<=2.5s) statically, ranks them, and MEASURES the feasible ones. The
    fully-fused {B2,B1,B0} and {B1,B0} groupings are INFEASIBLE for a clean splice
    (they require absorbing the dA_cumsum_tail intra-threadgroup atomic back into a
    kernel -> re-introduces the determinism race the wrapper recompute dissolved).
    The {B2,B1}{B0} grouping is the SELECTED winner: B2 and B1 spliced as TWO
    sequential ``with T.Kernel(...)`` grids in ONE ``@T.prim_func`` -> ONE metallib
    -> ONE tvm_ffi command-buffer dispatch (collapses one of the three ~per-dispatch
    finalize/commit floors). EACH grid keeps its OWN threadgroup allocation (so the
    §27 mono-fusion 88.5KB/1MB resident-DINP threadgroup wall does NOT apply -- the
    grids are sequential launches, not one fused threadgroup).

    The B2->B1 ``dchunk_states`` handoff + the shared ``dA_cumsum``/``prev_states``
    inputs are UNIFIED to single params inside the prim (the handoff is written by
    the B2 grid and read by the B1 grid WITHOUT an MLX round-trip). The B1
    dA_cumsum_tail recompute stays POST-kernel deterministic MLX (it runs AFTER this
    fused dispatch, feeding B0) -- the determinism fix is untouched.

    MEASURED (nam56r b=1 S=128 H=128 P=64 N=64): 3-dispatch baseline ~11043us ->
    2-dispatch B2+B1 fused ~10338us (~705us / ~6.4% recovered), all 8 grads
    bit-correct vs path-b GOLD over repeats. RULE #1: bit-correct + measured-faster
    before selection; the splice is proven byte-equal to the 3-kernel path (the
    dchunk_states handoff flows inside the one command buffer at the fp32 floor).

    Fused params (device buffers):
      [dout,cb,x,z,dt,dA_cumsum,C,B,prev_states,D,y,  (B2 in)
       dC,dx,dz,dchunk_states,dinp,dA_cumsum_y,dD,    (B2 out)
       dh_last,  (B1 NEW in)
       dstates,dh0,dA_cumsum_tail]  (B1 out)
    out_idx surfaces every grad the wrapper consumes; dchunk_states stays internal.
    """
    import tilelang

    from cppmega_mlx.nn._tilelang.mamba3_chunked_backward_core import (
        chunk_scan_combine_bwd_metal_prim,
        inter_chunk_recur_bwd_metal_prim,
    )
    from cppmega_mlx.runtime.path_c_backward_fusion_search import (
        splice_prims,
        _SHARED_BUFFER_NAMES,
    )

    b2 = chunk_scan_combine_bwd_metal_prim(b, s, c, g, h, p, n)
    b1 = inter_chunk_recur_bwd_metal_prim(b, s, c, g, h, p, n)
    # zero-init: B2's dD output is the ONE cross-threadgroup atomic. In the fused
    # param order its absolute output ordinal among the surfaced outputs is index 6
    # (dC,dx,dz,dchunk_states,dinp,dA_cumsum_y,dD). out_idx below lists those abs
    # positions; the zero-init position is the ordinal of dD WITHIN out_idx.
    # Fused absolute param positions:
    #  0 dout 1 cb 2 x 3 z 4 dt 5 dA_cumsum 6 C 7 B 8 prev_states 9 D 10 y
    #  11 dC 12 dx 13 dz 14 dchunk_states 15 dinp 16 dA_cumsum_y 17 dD
    #  18 dh_last 19 dstates 20 dh0 21 dA_cumsum_tail
    out_idx = [11, 12, 13, 14, 15, 16, 17, 19, 20, 21]
    dD_ordinal_in_outputs = out_idx.index(17)  # = 6
    fused = splice_prims(
        [b2, b1], _SHARED_BUFFER_NAMES, "b2b1_fused_live",
        zero_init_output_positions=[dD_ordinal_in_outputs],
    )
    return tilelang.compile(
        fused,
        target=_CHUNKED_METAL_TARGET,
        execution_backend="tvm_ffi",
        out_idx=out_idx,
    )


@lru_cache(maxsize=64)
def _chunked_bwd_b0_kernel(b: int, s: int, c: int, g: int, h: int, p: int, n: int) -> Any:
    """B0 (chunk_precompute_bwd) -> (dx_inp, dB, dlog_decay, ddt)."""
    import tilelang

    from cppmega_mlx.nn._tilelang.mamba3_chunked_backward_core import (
        chunk_precompute_bwd_metal_prim,
    )

    return tilelang.compile(
        chunk_precompute_bwd_metal_prim(b, s, c, g, h, p, n),
        target=_CHUNKED_METAL_TARGET,
        execution_backend="tvm_ffi",
        out_idx=[9, 10, 11, 12],
    )


def _path_c_chunk_size_for(seq: int) -> int:
    """Return the chunked-backward chunk size; nam56r/prod is 64 and must divide seq."""
    chunk = 64
    if seq % chunk != 0:
        raise ValueError(
            "mamba3 path_c_fwd_path_c_bwd: seqlen "
            f"({seq}) must be divisible by the chunked-backward chunk_size "
            f"({chunk}); the chunked B2->B1->B0 chain does not pad (RULE #1)."
        )
    return chunk


def _assert_per_head_constant_A(A: mx.array) -> mx.array:
    """Reduce A (b,seq,H) to the per-head (H,) the chunked kernels need.

    RULE #1: the chunked kernels take a PER-HEAD scalar A. If A varies across
    positions the chunked decay (A[h]*dt[l]) does not match the surface
    (A_pos[l]*dt[l]); emitting grads anyway would be a silent wrong path. RAISE
    with WHERE+WHAT instead. Returns the (H,) per-head A on success.
    """
    if A.ndim != 3:
        raise ValueError(
            f"mamba3 path_c_fwd_path_c_bwd: A must be (b,seq,H), got shape {A.shape}"
        )
    A32 = A.astype(mx.float32)
    per_head = A32[:, 0, :]  # (b, H)
    spread = mx.max(mx.abs(A32 - per_head[:, None, :]))
    mx.eval(spread)
    spread_f = float(spread)
    if spread_f > 1e-6:
        raise RuntimeError(
            "mamba3 path_c_fwd_path_c_bwd: A is PER-POSITION (max across-seq "
            f"spread {spread_f:.3e} > 1e-6). The chunked B2->B1->B0 kernels take a "
            "PER-HEAD scalar A and reconstruct decay as A[h]*dt[l]; a per-position A "
            "would make the ddt = dlog*A term wrong. Refusing to emit silently-wrong "
            "grads (RULE #1). Per-position-A support is the next phase (kernel ABI "
            "change). This mode is for the per-head-constant-A proving regime."
        )
    if A32.shape[0] != 1:
        # The (H,) kernel A is batch-independent; require a single batch's A here
        # (the per-head reduction already proved A is position-constant, but the
        # kernel cannot carry a batch axis on A).
        b0 = per_head[0:1]
        if float(mx.max(mx.abs(per_head - b0))) > 1e-6:
            raise RuntimeError(
                "mamba3 path_c_fwd_path_c_bwd: A differs across the batch axis; the "
                "chunked kernel A is (H,) batch-independent. Refusing a silent "
                "reduction (RULE #1)."
            )
    return per_head[0]  # (H,)


def _mamba3_chunked_backward_path_c(
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
    cb: mx.array,
    dA_cumsum: mx.array,
    prev_states: mx.array,
    y: mx.array,
) -> tuple[mx.array, ...]:
    """Chained B2->B1->B0 chunked backward producing the 8-grad MSL surface.

    The forward intermediates (cb, dA_cumsum fp16; prev_states fp32; y fp16) are
    STASHED from the fwd custom_function (no forward replay). Inputs are cast to
    the chunked kernels' fp16 ABI; accumulator grads come back fp32 and are cast
    to the primal dtypes. Returns (dx, dB, dC, dz, dA, ddt, dD, dh0).
    """
    batch, seq, heads, headdim, state = _validate_inputs(x, B, C, z, A, dt, D, h0)
    chunk = _path_c_chunk_size_for(seq)
    nchunks = seq // chunk
    G = heads  # at the dispatch surface B/C are per-head -> ngroups == nheads
    # Deterministic atomic-output ordering for the B2/B1/B0 owner outputs (see helper).
    _force_chunked_command_buffer_boundary()

    # fp16 kernel-input casts (the validated chunked ABI). dt_k = (b,H,c,s).
    def f16(a: mx.array) -> mx.array:
        return mx.contiguous(a.astype(mx.float16))

    x16 = f16(x)
    B16 = f16(B)
    C16 = f16(C)
    z16 = f16(z)
    D16 = f16(D)
    dout16 = f16(dy)
    y16 = f16(y)
    dt16 = dt.astype(mx.float16)
    dt_k = mx.contiguous(mx.transpose(dt16.reshape(batch, nchunks, chunk, heads), (0, 3, 1, 2)))
    cb16 = cb.astype(mx.float16) if cb.dtype != mx.float16 else cb
    dA16 = dA_cumsum.astype(mx.float16) if dA_cumsum.dtype != mx.float16 else dA_cumsum
    A_head16 = _assert_per_head_constant_A(A).astype(mx.float16)
    prev32 = prev_states.astype(mx.float32)

    # --- B2+B1 FUSED (directed-search SELECTED rank-2 {B2,B1}{B0}) ---
    # The directed backward-fusion search selected the {B2,B1}{B0} grouping: B2 and
    # B1 are spliced as two sequential T.Kernel grids in ONE prim_func -> ONE tvm_ffi
    # dispatch (collapses one of the three ~per-dispatch command-buffer floors;
    # MEASURED ~705us / ~6.4% recovered at nam56r, all 8 grads bit-correct vs path-b
    # GOLD). The B2->B1 dchunk_states handoff flows INSIDE the one command buffer
    # (proven byte-equal to the separate-kernel path at the fp32 floor). The
    # post-kernel dA_cumsum_tail / dD deterministic recomputes below are UNCHANGED
    # (the determinism fix is untouched -- the dA_tail recompute stays POST this
    # fused dispatch). The fully-fused {B2,B1,B0} and {B1,B0} groupings are
    # INFEASIBLE for a clean splice (they would absorb dA_cumsum_tail and re-
    # introduce the race) -- the search marks them infeasible, never silently fuses
    # them. See cppmega_mlx.runtime.path_c_backward_fusion_search.
    #
    # NB: the dD output (B2 cross-threadgroup atomic) is DISCARDED here
    # ("_dD_m_unused") -- the production 8-grad surface uses the deterministic
    # wrapper-computed dD below. The B1 dA_cumsum_tail (last fused output) is
    # likewise DISCARDED ("_dA_tail_kernel_unused") in favor of the deterministic
    # MLX recompute below.
    dh_last = mx.zeros((batch, heads, headdim, state), dtype=mx.float32)
    k_b2b1 = _chunked_bwd_b2b1_fused_kernel(batch, seq, chunk, G, heads, headdim, state)
    (
        dC_m, dx_b2, dz_m, dchunk, dinp_diag, dA_y, _dD_m_unused,
        dstates, dh0_m, _dA_tail_kernel_unused,
    ) = k_b2b1(
        dout16, cb16, x16, z16, dt_k, dA16, C16, B16, prev32, D16, y16, dh_last
    )

    # --- B1 dstates/dh0 now produced by the fused kernel above (dh_last seeded zero) ---
    # The B1 kernel's dA_cumsum_tail (3rd output) is INTENTIONALLY DISCARDED for
    # the production grad path (renamed _dA_tail_kernel_unused) — it is produced by
    # an INTRA-threadgroup T.atomic_add over headdim*dstate cell-lanes after an
    # in-body zero-store ordered only by a THREADGROUP barrier. The TileLang Metal
    # codegen PrintStorageSync (tilelang/src/target/codegen_metal.cc:1661-1668)
    # ALWAYS emits threadgroup_barrier(mem_flags::mem_threadgroup) and NEVER
    # mem_device, so the zero-store to the DEVICE buffer is not ordered against the
    # device-atomic RMW on Apple GPUs -> the zero can clobber an accumulated partial
    # (or the atomic reads an un-zeroed value) only under mx.vjp reverse-graph
    # scheduling -> dA_cumsum_tail is INTERMITTENTLY corrupt (MEASURED: surfaces at
    # sporadic repeat indices, propagating to dA/ddt > 1e-3, ~10% of fresh
    # processes). Apply the dD precedent (deterministic wrapper recompute, RULE #1,
    # one clear DETERMINISTIC path): recompute the tail in fp32 MLX from the
    # RACE-FREE B1 outputs/stash — dstates is a plain SINGLE-WRITER store and
    # prev_states is the stashed fwd buffer, both deterministic. The kernel math is
    #   dA_tail[b,h,cc,L-1] = exp(dA_cumsum[b,h,cc,L-1]) * sum_{p,n} dstates*prev_states
    # (core :4584-4593: decay=exp2(tail*log2e)=exp(tail); g[cc]=dstates[cc]). VERIFIED
    # bit-equal to the kernel tail at the fp32 floor (worst ~4.6e-06 over multi-run).
    # (dstates / dh0_m are produced by the fused B2+B1 kernel above; dchunk is the
    # INTERNALIZED handoff and no longer materialized as a separate MLX buffer.)
    L = chunk
    tail_dacs = dA16[:, :, :, L - 1].astype(mx.float32)  # (b,H,nchunks)
    decay_tail = mx.exp(tail_dacs)
    prod_pn = mx.sum(
        dstates.astype(mx.float32) * prev32, axis=(3, 4)
    )  # (b,nchunks,H)
    prod_pn = mx.transpose(prod_pn, (0, 2, 1))  # (b,H,nchunks)
    dA_tail = mx.zeros((batch, heads, nchunks, L), dtype=mx.float32)
    dA_tail[:, :, :, L - 1] = decay_tail * prod_pn
    dA_tail = mx.contiguous(dA_tail)

    # --- B0: chunk_precompute_bwd (dx fresh inp-path -> add B2 dx) ---
    k_b0 = _chunked_bwd_b0_kernel(batch, seq, chunk, G, heads, headdim, state)
    dx_b0, dB_m, dlog_m, ddt_m = k_b0(
        dstates, dinp_diag, dA_y, dA_tail, dA16, x16, B16, dt_k, A_head16
    )

    dx_total = dx_b2 + dx_b0  # B2 D-skip/dY*C path + B0 inp path (test: dx_full clone)

    # dD: compute in the wrapper (fp32, deterministic) instead of using the B2
    # kernel's ``dD_m``. RULE #1 (one clear, DETERMINISTIC path; no silent flaky
    # output). The B2 ``dD`` is the ONE CROSS-THREADGROUP atomic_add output
    # (T.atomic_add(dD[head], ...) from every (batch*nchunks) threadgroup into a
    # bridge-zeroed buffer). Its zero-init blit must complete before ALL those
    # atomics; the MLX tvm_ffi command-buffer ordering does NOT fully serialize
    # that cross-threadgroup case, so the kernel ``dD_m`` is INTERMITTENTLY wrong
    # (MEASURED: ~10% of fresh processes -> dD blow-up 5.33e-1 >> 1e-3, garbage
    # from the un-zeroed buffer; the other 7 grads are race-free). dD is a cheap,
    # exact MLX reduction of the SAME production VJP the serial prim uses
    # (mamba3_path_c serial prim :1484-1488): d_y_skip = dY*silu(z);
    # dD[h] = sum_{b,t,p} d_y_skip * x. Computing it here is bit-exact vs the
    # pure-MLX GOLD (max|wrapper_dD - gold_dD| = 2.4e-07, the fp32 floor) and fully
    # deterministic. The kernel still emits dD_m (used by the b0b1b2 stage test);
    # only the production 8-grad surface takes the wrapper dD.
    z32 = z.astype(mx.float32)
    sig_z = mx.sigmoid(z32)
    silu_z = z32 * sig_z
    d_y_skip = dy.astype(mx.float32) * silu_z  # (b,seq,H,P)
    dD_wrapper = mx.sum(d_y_skip * x.astype(mx.float32), axis=(0, 1, 3))  # (H,)

    # Map the chained chain's grads -> MSL primal-order 8-tuple. PRODUCTION model:
    # decay = exp(A*dt) with PER-HEAD scalar A; dlog_m (b,seq,H) is d_logdecay (the
    # grad wrt log_decay = A*dt). The surface dA at (b,seq,H) is d_logdecay*dt
    # (prod single-lane dA[b,t,h] = d_logdecay*dt, path_c :1532). The GOLD
    # mamba3_mimo_bwd_metal returns dA at (b,seq,H) ELEMENTWISE = d_logdecay*dt with
    # NO seq reduction (mamba3.py :344/:637 dA=sum_partial over the lane axis only,
    # and :1133 dA_steps[t]=d_log_decay*dt_f[:,t] stacked to (b,seq,H)); so we match
    # by multiplying dlog_m by the (b,seq,H) PRIMAL dt (NOT the fp16 dt_k transpose,
    # to keep native precision). B/C are per-head so dB/dC need no group reduction.
    dx = dx_total.astype(x.dtype)
    dB = dB_m.astype(B.dtype)
    dC = dC_m.astype(C.dtype)
    dz = dz_m.astype(z.dtype)
    dA = (dlog_m * dt.astype(dlog_m.dtype)).astype(A.dtype)
    ddt = ddt_m.astype(dt.dtype)
    dD = dD_wrapper.astype(D.dtype)  # deterministic wrapper dD (see above); NOT dD_m
    dh0 = dh0_m.astype(h0.dtype)
    return (dx, dB, dC, dz, dA, ddt, dD, dh0)


def _mamba3_chunked_fwd_intermediates_path_c(
    x: mx.array,
    B: mx.array,
    C: mx.array,
    A: mx.array,
    dt: mx.array,
    h0: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """Run F0+F1 to materialize (cb, dA_cumsum, prev_states) for the bwd stash.

    summary_states is a fwd-internal feeding F1 -> prev_states; only prev_states
    (and cb/dA_cumsum) outlive into the backward, so summary_states is not stashed.
    All inputs are cast to the chunked fp16 ABI; outputs keep their native dtypes
    (cb/dA_cumsum fp16, prev_states fp32).
    """
    batch, seq, heads, headdim = x.shape
    state = B.shape[-1]
    chunk = _path_c_chunk_size_for(seq)
    G = heads
    # Deterministic atomic-output ordering for the F0/F1 owner outputs (see helper).
    _force_chunked_command_buffer_boundary()

    def f16(a: mx.array) -> mx.array:
        return mx.contiguous(a.astype(mx.float16))

    A_head16 = _assert_per_head_constant_A(A).astype(mx.float16)
    x16 = f16(x)
    B16 = f16(B)
    C16 = f16(C)
    dt16 = dt.astype(mx.float16)

    k_f0 = _chunked_fwd_f0_kernel(batch, seq, chunk, G, heads, headdim, state)
    cb, dA_cumsum, summary_states = k_f0(x16, B16, C16, A_head16, dt16)

    k_f1 = _chunked_fwd_f1_kernel(batch, seq, chunk, G, heads, headdim, state)
    prev_states, _final_state = k_f1(
        summary_states, dA_cumsum, h0.astype(mx.float32)
    )
    return cb, dA_cumsum, prev_states


def _mamba3_pre_gate_yskip_path_c(
    x: mx.array,
    B: mx.array,
    C: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
) -> mx.array:
    """PRE-GATE y_skip = C.h + D*x (fp32 lane-scan), the B2 dgate multiplicand.

    The B2 backward forms dz = dout * y_skip * silu'(z) (mamba3_chunked_backward_core
    :257-259). The production dz VJP (mamba3_path_c.py serial prim :1479-1487) uses
    the PRE-GATE / pre-silu output y_skip = sum_n C[n]*h[n] + D*x, NOT the gated
    forward y = silu(z)*y_skip. Recompute y_skip here in fp32 via the SAME recurrence
    as the forward (mamba3_mimo_reference :483-490 / production serial prim): per
    position decay = exp(A*dt) (per-head-constant A), h_t = decay*h_{t-1} + x_t (outer)
    B_t, y_raw = sum_n C.h, y_skip = y_raw + D*x. Returns (batch, seq, heads, headdim).

    RULE #1: must NOT be reconstructed via y_gated/silu(z) (fp16 stash + near-zero
    silu division is unstable, probe 0.117). Computed pre-cast in fp32 here.
    """
    batch, seq, heads, headdim, state = _validate_inputs(
        x, B, C, x, A, dt, D, h0  # z unused here -> pass x to satisfy the z==x check
    )
    x32 = x.astype(mx.float32)
    B32 = B.astype(mx.float32)
    C32 = C.astype(mx.float32)
    A32 = A.astype(mx.float32)
    dt32 = dt.astype(mx.float32)
    D32 = D.astype(mx.float32)
    h = h0.astype(mx.float32)  # (b,H,P,N)
    log_decay = (A32 * dt32)  # (b,seq,H)
    y_skip_list: list[mx.array] = []
    for t in range(seq):
        decay = mx.exp(log_decay[:, t, :])[:, :, None, None]  # (b,H,1,1)
        xt = x32[:, t, :, :][:, :, :, None]  # (b,H,P,1)
        Bt = B32[:, t, :, :][:, :, None, :]  # (b,H,1,N)
        h = decay * h + xt * Bt
        Ct = C32[:, t, :, :][:, :, None, :]  # (b,H,1,N)
        y_raw = mx.sum(h * Ct, axis=-1)  # (b,H,P)
        y_skip = y_raw + D32[None, :, None] * x32[:, t, :, :]  # (b,H,P)
        y_skip_list.append(y_skip)
    return mx.stack(y_skip_list, axis=1)  # (b,seq,H,P)


@mx.custom_function
def mamba3_mimo_apply_with_state_path_c_fwd_path_c_bwd(
    x: mx.array,
    B: mx.array,
    C: mx.array,
    z: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Path C DSL fwd + stashed chunked-fwd intermediates for the chunked bwd.

    Returns (y, h_last, cb, dA_cumsum, prev_states, y_stash). The trailing four
    arrays are the MLX custom_function residual the .vjp reads to drive the
    chunked B2->B1->B0 backward WITHOUT replaying the forward chunk scan.

    The forward y/h_last are the UNCHANGED production Path C lane-scan
    (mamba3_mimo_fwd_path_c). The chunked-forward F0/F1 are additionally run to
    materialize the bwd's stash. This is the SELECTABLE proving mode; it is not
    the default and the receipt keeps AUTO on path_b until a no-worse bwd receipt
    is checked in (see mamba3_path_c_receipt_auto_mode).
    """
    y, h_last = mamba3_mimo_fwd_path_c(x, B, C, z, A, dt, D, h0)
    cb, dA_cumsum, prev_states = _mamba3_chunked_fwd_intermediates_path_c(
        x, B, C, A, dt, h0
    )
    # B2 dgate needs the PRE-GATE y_skip = C.h + D*x (production dz VJP, :1479-1487),
    # NOT the gated forward y = silu(z)*y_skip. Stashing the gated y put an extra
    # silu(z) factor into dz (the only failing grad). Recompute y_skip in fp32 (same
    # recurrence as the forward) and stash THAT. The returned forward y (output[0])
    # and h_last stay byte-identical; only this residual tensor changes.
    y_skip = _mamba3_pre_gate_yskip_path_c(x, B, C, A, dt, D, h0)
    y_stash = y_skip.astype(mx.float16)
    return y, h_last, cb, dA_cumsum, prev_states, y_stash


@mamba3_mimo_apply_with_state_path_c_fwd_path_c_bwd.vjp
def _mamba3_mimo_apply_with_state_path_c_fwd_path_c_bwd_vjp(
    primals: tuple[mx.array, ...],
    cotangent: tuple[mx.array, ...],
    output: tuple[mx.array, ...],
) -> tuple[mx.array, ...]:
    """Chained chunked B2->B1->B0 backward returning the 8-grad MSL surface.

    Only cotangent[0] (dy) is meaningful; the h_last and stashed-intermediate
    cotangents have zero loss-gradient (matching the existing ignore-h_last
    contract). Returns ONE grad per primal in primal order
    (x,B,C,z,A,dt,D,h0) — the SAME 8-grad surface as mamba3_mimo_bwd_metal.
    """
    x, B, C, z, A, dt, D, h0 = primals
    dy = cotangent[0]
    cb = output[2]
    dA_cumsum = output[3]
    prev_states = output[4]
    y_stash = output[5]
    return _mamba3_chunked_backward_path_c(
        dy, x, B, C, z, A, dt, D, h0,
        cb=cb, dA_cumsum=dA_cumsum, prev_states=prev_states, y=y_stash,
    )


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
    partial_dtypes = _bwd_lane_grad_dtypes_for_input_dtypes(
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
    kernel, lowering = _bwd_lane_grad_kernel_for_state_snapshots(
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
        _bwd_lane_grad_kernel_for_state_snapshots,
        _bwd_scratch_partial_kernel_for,
        _bwd_lane_grad_reduce_kernel_for,
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
    "mamba3_mimo_apply_with_state_path_c_fwd_path_c_bwd",
    "mamba3_path_c_auto_fwd_path_b_bwd_allowed",
    "mamba3_path_c_auto_mode_for_inputs",
    "mamba3_path_c_receipt_allows_auto_promotion",
    "mamba3_path_c_receipt_auto_mode",
    "mamba3_path_c_schedule_plan",
    "mamba3_mimo_bwd_path_c",
    "mamba3_mimo_fwd_path_c",
    "mamba3_mimo_path_c_status",
]

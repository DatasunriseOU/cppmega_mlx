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
  ``(x, B, C, z, A, dt, D, h0)`` after host-side P-axis reductions. The
  reverse pass carries ``h_t`` in per-lane registers and reconstructs
  ``h_{t-1}`` in place, so Path C does not allocate a global ``h_steps``
  scratch tensor.
* :func:`mamba3_mimo_apply_path_c` — convenience fwd surface returning ``y``.
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
_BWD_OUTPUT_NAMES = (
    "dx",
    "dz",
    "dB_partial",
    "dC_partial",
    "dA_partial",
    "ddt_partial",
    "dD_partial",
    "dh0",
)
_BWD_OUTPUT_IDX = (9, 10, 11, 12, 13, 14, 15, 16)
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
# Correctness first for full-model Path C bwd: cache every h_t boundary and
# avoid reconstructing h_{t-1} through 1 / decay. Larger blocks need a range
# proof/autotune gate because real bf16 model weights can drive decay to zero.
_BWD_SNAPSHOT_BLOCK = 1
_BWD_SNAPSHOT_OUTPUT_IDX = (9, 10, 11, 12, 13, 14, 15, 16)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_PATH_C_AUTO_PROMOTION_RECEIPT = (
    _REPO_ROOT / "bench" / "tilelang_ports" / "mamba3_path_c.json"
)
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
    simd_p_reduce = _bwd_simd_p_reduction_supported(
        batch=batch,
        heads=heads,
        headdim=headdim,
    )
    bwd_candidate = fwd_candidate and simd_p_reduce
    bwd_reason = (
        "bwd emits TileLang thread_allreduce_sum over P-axis grads"
        if simd_p_reduce
        else "bwd falls back to per-lane partial outputs until P maps to full rows"
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
        bwd_kernel, bwd_lowering = _bwd_kernel_for(1, 4, 1, 2, 4, return_msl=True)
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
    return min(256, lanes)


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
            tid_in_block = T.get_thread_binding(0)
            global_lane = bx * THREADS + tid_in_block
            # Per-lane state lives in registers (size N).
            h_state = T.alloc_local((STATE,), accum_dtype)
            if global_lane < LANES:
                p = global_lane % HEADDIM
                h = (global_lane // HEADDIM) % HEADS
                b = global_lane // (HEADDIM * HEADS)
                for n in T.serial(STATE):
                    h_state[n] = T.cast(h0[b, h, p, n], accum_dtype)
                for t in T.serial(SEQ):
                    A_val = T.cast(A[b, t, h], accum_dtype)
                    dt_val = T.cast(dt[b, t, h], accum_dtype)
                    decay = T.exp(A_val * dt_val)
                    x_val = T.cast(x[b, t, h, p], accum_dtype)
                    z_val = T.cast(z[b, t, h, p], accum_dtype)
                    y_acc = T.alloc_var(T.float32, init=0.0)
                    for n in T.serial(STATE):
                        new_h = decay * h_state[n] + x_val * T.cast(
                            B[b, t, h, n],
                            accum_dtype,
                        )
                        h_state[n] = new_h
                        y_acc += new_h * T.cast(C[b, t, h, n], accum_dtype)
                    D_h = T.cast(D[h], accum_dtype)
                    y_skipped = y_acc + D_h * x_val
                    sig_z = T.alloc_var(T.float32, init=0.0)
                    if z_val >= 0.0:
                        sig_z = 1.0 / (1.0 + T.exp(-z_val))
                    else:
                        sig_z = T.exp(z_val)
                        sig_z = sig_z / (1.0 + sig_z)
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
def _bwd_kernel_for(
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
    """Build & cache the Path C TileLang bwd kernel for a given (B, T, H, P, N).

    The forward prepass computes only the final ``h_T`` per lane in registers.
    The reverse pass reconstructs ``h_{t-1}`` from the current ``h_t`` in place,
    avoiding the global ``h_steps`` scratch tensor used by the legacy Path B
    MSL kernel. The host reduces per-lane partials to the final gradient shapes
    without atomics.
    """

    import tilelang.language as T

    LANES = BATCH * HEADS * HEADDIM
    THREADS = _threads_for(LANES)
    accum_dtype = "float32"

    @T.prim_func
    def bwd(
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
        with T.Kernel(T.ceildiv(LANES, THREADS), threads=THREADS) as bx:
            tid = T.get_thread_binding(0)
            global_lane = bx * THREADS + tid
            h_state = T.alloc_local((STATE,), accum_dtype)
            dh = T.alloc_local((STATE,), accum_dtype)
            if global_lane < LANES:
                p = global_lane % HEADDIM
                h = (global_lane // HEADDIM) % HEADS
                b = global_lane // (HEADDIM * HEADS)

                # Forward prepass: keep only h_T in registers.
                for n in T.serial(STATE):
                    h_state[n] = T.cast(h0[b, h, p, n], accum_dtype)
                for t in T.serial(SEQ):
                    A_val = T.cast(A[b, t, h], accum_dtype)
                    dt_val = T.cast(dt[b, t, h], accum_dtype)
                    decay = T.exp(A_val * dt_val)
                    x_val = T.cast(x[b, t, h, p], accum_dtype)
                    for n in T.serial(STATE):
                        new_h = decay * h_state[n] + x_val * T.cast(
                            B[b, t, h, n],
                            accum_dtype,
                        )
                        h_state[n] = new_h

                # Reverse pass. h_state[n] is h_t for the current t and is
                # overwritten with h_{t-1} before the next reverse iteration.
                for n in T.serial(STATE):
                    dh[n] = 0.0
                dD_acc = T.alloc_var(T.float32, init=0.0)
                for r in T.serial(SEQ):
                    t = SEQ - 1 - r
                    A_val = T.cast(A[b, t, h], accum_dtype)
                    dt_val = T.cast(dt[b, t, h], accum_dtype)
                    decay = T.exp(A_val * dt_val)
                    inv_decay = T.alloc_var(T.float32, init=1.0 / decay)
                    x_val = T.cast(x[b, t, h, p], accum_dtype)
                    z_val = T.cast(z[b, t, h, p], accum_dtype)
                    dY = T.cast(dy[b, t, h, p], accum_dtype)

                    # y_state[t] = sum_n h_t[n] * C[t,n]
                    y_state = T.alloc_var(T.float32, init=0.0)
                    for n in T.serial(STATE):
                        y_state += h_state[n] * T.cast(C[b, t, h, n], accum_dtype)
                    D_h = T.cast(D[h], accum_dtype)
                    y_skipped = y_state + D_h * x_val
                    sig_z = T.alloc_var(T.float32, init=0.0)
                    if z_val >= 0.0:
                        sig_z = 1.0 / (1.0 + T.exp(-z_val))
                    else:
                        sig_z = T.exp(z_val)
                        sig_z = sig_z / (1.0 + sig_z)
                    silu_z = z_val * sig_z
                    silu_dz = sig_z * (1.0 + z_val * (1.0 - sig_z))

                    d_silu = dY * y_skipped
                    d_y_skipped = dY * silu_z

                    dz[b, t, h, p] = T.cast(d_silu * silu_dz, dz_dtype)
                    dD_acc += d_y_skipped * x_val

                    # Per-lane partials and h_prev reconstruction. Host sums
                    # partials over P later; h_state is updated in-place to
                    # h_{t-1} for the next reverse step.
                    dx_inp = T.alloc_var(T.float32, init=0.0)
                    d_decay = T.alloc_var(T.float32, init=0.0)
                    if t == 0:
                        for n in T.serial(STATE):
                            C_val = T.cast(C[b, t, h, n], accum_dtype)
                            B_val = T.cast(B[b, t, h, n], accum_dtype)
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
                            d_decay += dh_n * T.cast(h0[b, h, p, n], accum_dtype)
                            dh[n] = dh_n * decay
                    else:
                        for n in T.serial(STATE):
                            C_val = T.cast(C[b, t, h, n], accum_dtype)
                            B_val = T.cast(B[b, t, h, n], accum_dtype)
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
                            h_prev = (h_state[n] - x_val * B_val) * inv_decay
                            d_decay += dh_n * h_prev
                            h_state[n] = h_prev
                            dh[n] = dh_n * decay
                    dx_skip = d_y_skipped * D_h
                    dx[b, t, h, p] = T.cast(dx_skip + dx_inp, dx_dtype)

                    d_logdecay = d_decay * decay
                    dA_partial[b, t, h, p] = T.cast(
                        d_logdecay * dt_val,
                        dA_dtype,
                    )
                    ddt_partial[b, t, h, p] = T.cast(
                        d_logdecay * A_val,
                        ddt_dtype,
                    )

                # After the backward sweep, dh is dh0 for this lane.
                for n in T.serial(STATE):
                    dh0[b, h, p, n] = T.cast(dh[n], dh0_dtype)
                dD_partial[b, h, p] = T.cast(dD_acc, dD_dtype)

    artifact = dispatch_lower(bwd, target="metal", return_msl=True)
    if hasattr(artifact, "_tilelang_engine_target"):
        raise MSLDispatchUnsupported("Mamba3 Path C requires TileLang MSL extraction metadata")
    lowering = cast(_msl_transform.TileLangMSLLowering, artifact)
    input_names = [
        name for name in lowering.buffer_param_names if name not in _BWD_OUTPUT_NAMES
    ]
    if set(input_names) != {"dy", "x", "B", "C", "z", "A", "dt", "D", "h0"}:
        raise MSLDispatchUnsupported(
            "unexpected Mamba3 Path C bwd buffer signature: "
            + ", ".join(lowering.buffer_param_names)
        )
    import tilelang

    kernel = tilelang.compile(
        bwd,
        target=_msl_transform._as_metal_target("metal"),
        execution_backend="tvm_ffi",
        out_idx=list(_BWD_OUTPUT_IDX),
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
    BLOCK = _BWD_SNAPSHOT_BLOCK
    BLOCKS = (SEQ + BLOCK - 1) // BLOCK
    accum_dtype = "float32"

    @T.prim_func
    def snapshots(
        x: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), x_dtype),
        B: T.Tensor((BATCH, SEQ, HEADS, STATE), B_dtype),
        A: T.Tensor((BATCH, SEQ, HEADS), A_dtype),
        dt: T.Tensor((BATCH, SEQ, HEADS), dt_dtype),
        h0: T.Tensor((BATCH, HEADS, HEADDIM, STATE), h0_dtype),
        h_snap: T.Tensor((BATCH, BLOCKS + 1, HEADS, HEADDIM, STATE), "float32"),
    ):
        with T.Kernel(T.ceildiv(LANES, THREADS), threads=THREADS) as bx:
            tid = T.get_thread_binding(0)
            global_lane = bx * THREADS + tid
            h_state = T.alloc_local((STATE,), accum_dtype)
            if global_lane < LANES:
                p = global_lane % HEADDIM
                h = (global_lane // HEADDIM) % HEADS
                b = global_lane // (HEADDIM * HEADS)
                for n in T.serial(STATE):
                    h_state[n] = T.cast(h0[b, h, p, n], accum_dtype)
                    h_snap[b, 0, h, p, n] = h_state[n]
                for block in T.serial(BLOCKS):
                    for step in T.serial(BLOCK):
                        t = block * BLOCK + step
                        if t < SEQ:
                            A_val = T.cast(A[b, t, h], accum_dtype)
                            dt_val = T.cast(dt[b, t, h], accum_dtype)
                            decay = T.exp(A_val * dt_val)
                            x_val = T.cast(x[b, t, h, p], accum_dtype)
                            for n in T.serial(STATE):
                                h_state[n] = decay * h_state[n] + x_val * T.cast(
                                    B[b, t, h, n],
                                    accum_dtype,
                                )
                    for n in T.serial(STATE):
                        h_snap[b, block + 1, h, p, n] = h_state[n]

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


@lru_cache(maxsize=128)
def _bwd_kernel_for_state_snapshots(
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
) -> tuple[Any, _msl_transform.TileLangMSLLowering]:
    """Build the long-sequence bwd kernel that consumes state snapshots."""

    import tilelang.language as T

    LANES = BATCH * HEADS * HEADDIM
    THREADS = _threads_for(LANES)
    BLOCK = _BWD_SNAPSHOT_BLOCK
    BLOCKS = (SEQ + BLOCK - 1) // BLOCK
    accum_dtype = "float32"

    @T.prim_func
    def bwd_snap(
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
        dB_partial: T.Tensor((BATCH, SEQ, HEADS, STATE, HEADDIM), dB_dtype),
        dC_partial: T.Tensor((BATCH, SEQ, HEADS, STATE, HEADDIM), dC_dtype),
        dA_partial: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), dA_dtype),
        ddt_partial: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), ddt_dtype),
        dD_partial: T.Tensor((BATCH, HEADS, HEADDIM), dD_dtype),
        dh0: T.Tensor((BATCH, HEADS, HEADDIM, STATE), dh0_dtype),
    ):
        with T.Kernel(T.ceildiv(LANES, THREADS), threads=THREADS) as bx:
            tid = T.get_thread_binding(0)
            global_lane = bx * THREADS + tid
            h_state = T.alloc_local((STATE,), accum_dtype)
            dh = T.alloc_local((STATE,), accum_dtype)
            if global_lane < LANES:
                p = global_lane % HEADDIM
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
                            decay = T.exp(A_val * dt_val)
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
                            sig_z = T.alloc_var(T.float32, init=0.0)
                            if z_val >= 0.0:
                                sig_z = 1.0 / (1.0 + T.exp(-z_val))
                            else:
                                sig_z = T.exp(z_val)
                                sig_z = sig_z / (1.0 + sig_z)
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
                                h_prev = T.alloc_var(T.float32, init=0.0)
                                if BLOCK == 1:
                                    h_prev = h_snap[b, block, h, p, n]
                                    h_state[n] = h_prev
                                else:
                                    inv_decay = T.alloc_var(T.float32, init=1.0 / decay)
                                    if t == block_start:
                                        h_prev = h_snap[b, block, h, p, n]
                                    else:
                                        h_prev = (h_state[n] - x_val * B_val) * inv_decay
                                        h_state[n] = h_prev
                                d_decay += dh_n * h_prev
                                dh[n] = dh_n * decay

                            dx_skip = d_y_skipped * D_h
                            dx[b, t, h, p] = T.cast(dx_skip + dx_inp, dx_dtype)

                            d_logdecay = d_decay * decay
                            dA_partial[b, t, h, p] = T.cast(
                                d_logdecay * dt_val,
                                dA_dtype,
                            )
                            ddt_partial[b, t, h, p] = T.cast(
                                d_logdecay * A_val,
                                ddt_dtype,
                            )

                for n in T.serial(STATE):
                    dh0[b, h, p, n] = T.cast(dh[n], dh0_dtype)
                dD_partial[b, h, p] = T.cast(dD_acc, dD_dtype)

    artifact = dispatch_lower(bwd_snap, target="metal", return_msl=True)
    if hasattr(artifact, "_tilelang_engine_target"):
        raise MSLDispatchUnsupported("Mamba3 Path C requires TileLang MSL extraction metadata")
    lowering = cast(_msl_transform.TileLangMSLLowering, artifact)
    input_names = [
        name for name in lowering.buffer_param_names if name not in _BWD_OUTPUT_NAMES
    ]
    if set(input_names) != {"dy", "x", "B", "C", "z", "A", "dt", "D", "h_snap"}:
        raise MSLDispatchUnsupported(
            "unexpected Mamba3 Path C snapshot bwd buffer signature: "
            + ", ".join(lowering.buffer_param_names)
        )
    import tilelang

    kernel = tilelang.compile(
        bwd_snap,
        target=_msl_transform._as_metal_target("metal"),
        execution_backend="tvm_ffi",
        out_idx=list(_BWD_SNAPSHOT_OUTPUT_IDX),
    )
    return kernel, lowering


def _bwd_simd_p_reduction_supported(
    *, batch: int, heads: int, headdim: int
) -> bool:
    """Return whether P-axis grads map to TileLang split thread-allreduce."""

    lanes = batch * heads * headdim
    threads = _threads_for(lanes)
    return (
        headdim > 0
        and headdim % 32 == 0
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
            "Mamba3 Path C P-reduction requires HEADDIM to be a multiple of "
            "32 with threadgroups aligned to full P rows"
        )

    import tilelang.language as T

    LANES = BATCH * HEADS * HEADDIM
    THREADS = _threads_for(LANES)
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
        with T.Kernel(T.ceildiv(LANES, THREADS), threads=THREADS) as bx:
            tid = T.get_thread_binding(0)
            global_lane = bx * THREADS + tid
            h_state = T.alloc_local((STATE,), accum_dtype)
            dh = T.alloc_local((STATE,), accum_dtype)
            if global_lane < LANES:
                p = global_lane % HEADDIM
                h = (global_lane // HEADDIM) % HEADS
                b = global_lane // (HEADDIM * HEADS)

                for n in T.serial(STATE):
                    h_state[n] = T.cast(h0[b, h, p, n], accum_dtype)
                for t in T.serial(SEQ):
                    A_val = T.cast(A[b, t, h], accum_dtype)
                    dt_val = T.cast(dt[b, t, h], accum_dtype)
                    decay = T.exp(A_val * dt_val)
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
                    decay = T.exp(A_val * dt_val)
                    inv_decay = T.alloc_var(T.float32, init=1.0 / decay)
                    x_val = T.cast(x[b, t, h, p], accum_dtype)
                    z_val = T.cast(z[b, t, h, p], accum_dtype)
                    dY = T.cast(dy[b, t, h, p], accum_dtype)

                    y_state = T.alloc_var(T.float32, init=0.0)
                    for n in T.serial(STATE):
                        y_state += h_state[n] * T.cast(C[b, t, h, n], accum_dtype)
                    D_h = T.cast(D[h], accum_dtype)
                    y_skipped = y_state + D_h * x_val
                    sig_z = T.alloc_var(T.float32, init=0.0)
                    if z_val >= 0.0:
                        sig_z = 1.0 / (1.0 + T.exp(-z_val))
                    else:
                        sig_z = T.exp(z_val)
                        sig_z = sig_z / (1.0 + sig_z)
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
                                d_y_skipped * h_state[n], dC_sum[0], p
                            )
                            T.thread_allreduce_sum(dh_n * x_val, dB_sum[0], p)
                            if p == 0:
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
                                d_y_skipped * h_state[n], dC_sum[0], p
                            )
                            T.thread_allreduce_sum(dh_n * x_val, dB_sum[0], p)
                            if p == 0:
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
                    T.thread_allreduce_sum(dA_lane, dA_sum[0], p)
                    T.thread_allreduce_sum(ddt_lane, ddt_sum[0], p)
                    if p == 0:
                        dA[b, t, h] = T.cast(dA_sum[0], dA_dtype)
                        ddt[b, t, h] = T.cast(ddt_sum[0], ddt_dtype)

                for n in T.serial(STATE):
                    dh0[b, h, p, n] = T.cast(dh[n], dh0_dtype)
                dD_sum = T.alloc_local((1,), accum_dtype)
                T.thread_allreduce_sum(dD_acc, dD_sum[0], p)
                if p == 0:
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


def _mamba3_bwd_owner_outputs(
    out: Mamba3BwdOwnerOutputs | None,
    *,
    batch: int,
    seq: int,
    heads: int,
    headdim: int,
    state: int,
    dtypes: tuple[mx.Dtype, ...],
) -> Mamba3BwdOwnerOutputs | None:
    op_name = "mamba3_mimo_bwd_path_c"
    if out is None:
        return None
    if not isinstance(out, tuple) or len(out) != len(_BWD_OUTPUT_NAMES):
        raise TypeError(
            f"{op_name}: out must be a tuple matching {_BWD_OUTPUT_NAMES!r}"
        )
    expected_shapes = (
        (batch, seq, heads, headdim),
        (batch, seq, heads, headdim),
        (batch, seq, heads, state, headdim),
        (batch, seq, heads, state, headdim),
        (batch, seq, heads, headdim),
        (batch, seq, heads, headdim),
        (batch, heads, headdim),
        (batch, heads, headdim, state),
    )
    return cast(
        Mamba3BwdOwnerOutputs,
        tuple(
            _require_owner_array(op_name, name, array, shape=shape, dtype=dtype)
            for name, array, shape, dtype in zip(
                _BWD_OUTPUT_NAMES,
                out,
                expected_shapes,
                dtypes,
                strict=True,
            )
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
            "Mamba3 Path C simd bwd route requires HEADDIM=32"
        )

    try:
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
        out_list = kernel(dy, x, B, C, z, A, dt, D, h0)
    except Exception as exc:
        _raise_if_dlpack_boundary_failure("mamba3_mimo_bwd_path_c", exc)
        raise RuntimeError("mamba3_mimo_bwd_path_c simd dispatch failed") from exc

    if not isinstance(out_list, (list, tuple)) or len(out_list) != len(_BWD_SIMD_OUTPUT_NAMES):
        raise RuntimeError("Mamba3 Path C simd bwd tvm-ffi returned an invalid output tuple")
    dx_pc, dz_pc, dB_pc, dC_pc, dA_pc, ddt_pc, dD_bh, dh0_pc = out_list
    del lowering
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


def _mamba3_mimo_bwd_path_c_partials_snapshot(
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
) -> Mamba3BwdOwnerOutputs:
    """Run long-sequence Path C bwd with block state snapshots."""

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
        )
        kernel, lowering = _bwd_kernel_for_state_snapshots(
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
        raise RuntimeError("mamba3_mimo_bwd_path_c snapshot lowering failed") from exc

    try:
        snapshot_out = snapshot_kernel(x, B, A, dt, h0)
    except Exception as exc:
        _raise_if_dlpack_boundary_failure("mamba3_mimo_bwd_path_c", exc)
        raise RuntimeError("mamba3_mimo_bwd_path_c snapshot dispatch failed") from exc
    if isinstance(snapshot_out, mx.array):
        h_snap = snapshot_out
    elif isinstance(snapshot_out, (list, tuple)) and len(snapshot_out) == 1:
        h_snap = snapshot_out[0]
    else:
        raise RuntimeError("Mamba3 Path C snapshot tvm-ffi returned an invalid output tuple")

    owner_outputs = _mamba3_bwd_owner_outputs(
        out,
        batch=batch,
        seq=seq,
        heads=heads,
        headdim=headdim,
        state=state,
        dtypes=(
            mx.float32,
            mx.float32,
            mx.float32,
            mx.float32,
            mx.float32,
            mx.float32,
            mx.float32,
            mx.float32,
        ),
    )
    try:
        if owner_outputs is None:
            out_list = kernel(dy, x, B, C, z, A, dt, D, h_snap)
        else:
            (
                dx,
                dz,
                dB_partial,
                dC_partial,
                dA_partial,
                ddt_partial,
                dD_partial,
                dh0,
            ) = owner_outputs
            out_list = kernel(
                dy,
                x,
                B,
                C,
                z,
                A,
                dt,
                D,
                h_snap,
                out=(
                    dx,
                    dz,
                    dB_partial,
                    dC_partial,
                    dA_partial,
                    ddt_partial,
                    dD_partial,
                    dh0,
                ),
            )
    except Exception as exc:
        _raise_if_dlpack_boundary_failure("mamba3_mimo_bwd_path_c", exc)
        raise RuntimeError("mamba3_mimo_bwd_path_c snapshot bwd dispatch failed") from exc

    if not isinstance(out_list, (list, tuple)) or len(out_list) != len(_BWD_OUTPUT_NAMES):
        raise RuntimeError("Mamba3 Path C snapshot bwd tvm-ffi returned an invalid output tuple")
    if owner_outputs is not None:
        if not all(
            got is expected
            for got, expected in zip(out_list, owner_outputs, strict=True)
        ):
            raise RuntimeError(
                "Mamba3 Path C snapshot bwd tvm-ffi did not return caller-owned outputs"
            )
    del snapshot_lowering, lowering
    return cast(Mamba3BwdOwnerOutputs, tuple(out_list))


def _mamba3_mimo_bwd_path_c_partials(
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
) -> Mamba3BwdOwnerOutputs:
    """Run the lowered Path C bwd kernel and return unreduced per-lane partials."""

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
            "because no TileLang kernel runs to initialize partial buffers"
        )
    if seq > _BWD_SNAPSHOT_BLOCK:
        return _mamba3_mimo_bwd_path_c_partials_snapshot(
            dy,
            x,
            B,
            C,
            z,
            A,
            dt,
            D,
            h0,
            out=out,
        )

    try:
        kernel, lowering = _bwd_kernel_for(
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
        raise RuntimeError("mamba3_mimo_bwd_path_c lowering failed") from exc

    owner_outputs = _mamba3_bwd_owner_outputs(
        out,
        batch=batch,
        seq=seq,
        heads=heads,
        headdim=headdim,
        state=state,
        dtypes=(
            mx.float32,
            mx.float32,
            mx.float32,
            mx.float32,
            mx.float32,
            mx.float32,
            mx.float32,
            mx.float32,
        ),
    )
    try:
        if owner_outputs is None:
            out_list = kernel(dy, x, B, C, z, A, dt, D, h0)
        else:
            (
                dx,
                dz,
                dB_partial,
                dC_partial,
                dA_partial,
                ddt_partial,
                dD_partial,
                dh0,
            ) = owner_outputs
            out_list = kernel(
                dy,
                x,
                B,
                C,
                z,
                A,
                dt,
                D,
                h0,
                out=(
                    dx,
                    dz,
                    dB_partial,
                    dC_partial,
                    dA_partial,
                    ddt_partial,
                    dD_partial,
                    dh0,
                ),
            )
    except Exception as exc:
        _raise_if_dlpack_boundary_failure("mamba3_mimo_bwd_path_c", exc)
        raise RuntimeError("mamba3_mimo_bwd_path_c dispatch failed") from exc

    if not isinstance(out_list, (list, tuple)) or len(out_list) != len(_BWD_OUTPUT_NAMES):
        raise RuntimeError("Mamba3 Path C bwd tvm-ffi returned an invalid output tuple")
    if owner_outputs is not None:
        if not all(
            got is expected
            for got, expected in zip(out_list, owner_outputs, strict=True)
        ):
            raise RuntimeError(
                "Mamba3 Path C bwd tvm-ffi did not return caller-owned outputs"
            )
    del lowering
    return cast(Mamba3BwdOwnerOutputs, tuple(out_list))


def _reduce_mamba3_bwd_partials(
    partials: Mamba3BwdOwnerOutputs,
    *,
    x_dtype: mx.Dtype,
    B_dtype: mx.Dtype,
    C_dtype: mx.Dtype,
    z_dtype: mx.Dtype,
    A_dtype: mx.Dtype,
    dt_dtype: mx.Dtype,
    D_dtype: mx.Dtype,
    h0_dtype: mx.Dtype,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Reduce Path C bwd partials into the public gradient tuple."""

    dx_pc, dz_pc, dB_p, dC_p, dA_p, ddt_p, dD_p, dh0_pc = partials
    # dB/dC partials are laid out (B, T, H, N, P), so the P-axis reduction is
    # over the contiguous trailing dimension instead of a stride-N axis.
    dB_pc = mx.sum(dB_p.astype(mx.float32), axis=4)      # -> (B, T, H, N)
    dC_pc = mx.sum(dC_p.astype(mx.float32), axis=4)      # -> (B, T, H, N)
    dA_pc = mx.sum(dA_p.astype(mx.float32), axis=3)      # -> (B, T, H)
    ddt_pc = mx.sum(ddt_p.astype(mx.float32), axis=3)    # -> (B, T, H)
    dD_pc = mx.sum(dD_p.astype(mx.float32), axis=(0, 2))
    return (
        _astype_if_needed(dx_pc, x_dtype),
        _astype_if_needed(dB_pc, B_dtype),
        _astype_if_needed(dC_pc, C_dtype),
        _astype_if_needed(dz_pc, z_dtype),
        _astype_if_needed(dA_pc, A_dtype),
        _astype_if_needed(ddt_pc, dt_dtype),
        _astype_if_needed(dD_pc, D_dtype),
        _astype_if_needed(dh0_pc, h0_dtype),
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
    """Run the lowered Path C bwd kernel and reduce partials to public grads."""

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
    if seq == 0:
        if out is not None:
            raise RuntimeError(
                "mamba3_mimo_bwd_path_c owner-output route is not dispatchable "
                "for seq=0 because no TileLang kernel runs to initialize buffers"
            )
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
    if out is None and _bwd_simd_p_reduction_supported(
        batch=batch,
        heads=heads,
        headdim=headdim,
    ):
        return _mamba3_mimo_bwd_path_c_simd_kernel(dy, x, B, C, z, A, dt, D, h0)
    return _reduce_mamba3_bwd_partials(
        _mamba3_mimo_bwd_path_c_partials(dy, x, B, C, z, A, dt, D, h0, out=out),
        x_dtype=x.dtype,
        B_dtype=B.dtype,
        C_dtype=C.dtype,
        z_dtype=z.dtype,
        A_dtype=A.dtype,
        dt_dtype=dt.dtype,
        D_dtype=D.dtype,
        h0_dtype=h0.dtype,
    )


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
    *, batch: int, seq: int, heads: int, headdim: int, state: int
) -> str:
    """Return the raw lowered MSL for the Path C forward kernel.

    Used by ``scripts/bench_tilelang_mamba3_path_c.py`` to write the
    ``docs/tilelang_ports/mamba3_path_c_lowered.metal`` artifact.
    """

    kernel, lowering = _fwd_kernel_for(
        batch, seq, heads, headdim, state, return_msl=True
    )
    del kernel
    return _source_with_reused_scalar_bindings(lowering)


def dump_lowered_bwd_msl(
    *, batch: int, seq: int, heads: int, headdim: int, state: int
) -> str:
    """Return the raw lowered MSL for the Path C backward kernel."""

    if _bwd_simd_p_reduction_supported(
        batch=batch,
        heads=heads,
        headdim=headdim,
    ):
        kernel, lowering = _bwd_simd_reduce_kernel_for(
            batch, seq, heads, headdim, state, return_msl=True
        )
    else:
        kernel, lowering = _bwd_kernel_for(
            batch, seq, heads, headdim, state, return_msl=True
        )
    del kernel
    return _source_with_reused_scalar_bindings(lowering)


__all__ = [
    "Mamba3PathCSchedulePlan",
    "Mamba3PathCStatus",
    "Mamba3PathCZ3Policy",
    "dump_lowered_bwd_msl",
    "dump_lowered_fwd_msl",
    "mamba3_mimo_apply_path_c",
    "mamba3_mimo_apply_with_state_path_c",
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

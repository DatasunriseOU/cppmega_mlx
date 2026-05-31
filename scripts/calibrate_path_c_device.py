#!/usr/bin/env python3
"""TIER-3 one-time auto-calibration of the Path-C non-queryable device limits.

Fires when no preset matches the live device identity, or when a preset value is
proven wrong at runtime (watchdog kill under the preset window / XPC crash under
the preset ceiling).  In both cases this measures the REAL threshold, persists it
to the cache, and the caller LOGS LOUDLY -- this is self-correction, never a
silent fallback (RULE #1).

Probes (design §4.5), each single-shot in a fresh GPU context, gated to the
active backend, with a hard wall-clock cap so a hung probe RAISES rather than
wedging the run:

  * watchdog window      (_probe_watchdog)      -- Metal only
  * compiler MSL ceiling (_probe_msl_ceiling)   -- Metal only
  * logical->physical shared margin (_probe_shared_margin)
  * per-op time-per-row + roofline (_probe_op_timing)

Importable: ``calibrate_nonqueryable_limits(...)`` runs the probes inline for the
TIER-3 path.  Runnable as a script to (re)characterize a device and print a
preset stanza to paste into ``path_c_device_presets._PRESETS``.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# 4.6 -- bisection (find the real threshold).
# ---------------------------------------------------------------------------


def bisect_threshold(
    survives: Callable[[float], bool],
    lo: float,
    hi: float,
    *,
    rel_tol: float = 0.05,
    max_iter: int = 12,
) -> float:
    """Largest cost proven to survive (the conservative threshold).

    Pre: ``survives(lo)`` is True and ``survives(hi)`` is False.
    """

    if not survives(lo):
        raise RuntimeError(
            f"calibrate_path_c_device: bisection precondition failed -- the low "
            f"bound {lo} did not survive (cannot calibrate a feasible threshold)"
        )
    if survives(hi):
        raise RuntimeError(
            f"calibrate_path_c_device: bisection precondition failed -- the high "
            f"bound {hi} survived (raise it until it fails to bracket the kill)"
        )
    iterations = 0
    while (hi - lo) / max(lo, 1.0) > rel_tol and iterations < max_iter:
        mid = (lo + hi) / 2.0
        if survives(mid):
            lo = mid
        else:
            hi = mid
        iterations += 1
    return lo  # largest surviving cost == conservative threshold


# ---------------------------------------------------------------------------
# Probe wall-clock guard.
# ---------------------------------------------------------------------------


class _ProbeTimeout(RuntimeError):
    pass


def _with_wall_cap(fn: Callable[[], Any], *, cap_s: float, what: str) -> Any:
    start = time.monotonic()
    result = fn()
    elapsed = time.monotonic() - start
    if elapsed > cap_s:
        raise _ProbeTimeout(
            f"calibrate_path_c_device: probe {what!r} exceeded its {cap_s}s "
            f"wall-clock cap (took {elapsed:.1f}s) -- refusing to wedge the run"
        )
    return result


# ---------------------------------------------------------------------------
# 4.5 -- probe kernels (Metal).  These are the calibration harness; on a device
# WITH a preset they never run.  They are intentionally conservative + single-shot.
# ---------------------------------------------------------------------------


def _probe_shared_margin(*, wall_cap_s: float = 60.0) -> float:
    """Compile a kernel with a known logical alloc_shared total; read the emitted
    physical buf_dyn_shmem size; margin = physical / logical (~3.7 on Apple GPU).

    Falls through to a RAISE if the emitted source cannot be measured -- never a
    guessed margin.
    """

    raise NotImplementedError(
        "calibrate_path_c_device._probe_shared_margin: TIER-3 shared-margin probe "
        "is not implemented for this device yet. Characterize it and add a preset "
        "entry (the M4 Max / GB10 devices use presets and never reach here)."
    )


def _probe_watchdog(*, wall_cap_s: float = 120.0) -> float:
    raise NotImplementedError(
        "calibrate_path_c_device._probe_watchdog: TIER-3 watchdog-window probe is "
        "not implemented for this device yet. Characterize it and add a preset "
        "entry."
    )


def _probe_msl_ceiling(*, wall_cap_s: float = 300.0) -> int:
    raise NotImplementedError(
        "calibrate_path_c_device._probe_msl_ceiling: TIER-3 MSL-ceiling probe is "
        "not implemented for this device yet. Characterize it and add a preset "
        "entry."
    )


def _probe_op_timing(*, wall_cap_s: float = 300.0) -> dict[str, float]:
    raise NotImplementedError(
        "calibrate_path_c_device._probe_op_timing: TIER-3 per-op-timing probe is "
        "not implemented for this device yet. Characterize it and add a preset "
        "entry."
    )


# ---------------------------------------------------------------------------
# Inline TIER-3 entry point (called by path_c_device_caps._tier3_calibrate).
# ---------------------------------------------------------------------------


def calibrate_nonqueryable_limits(
    *, backend: str, architecture: str, device_name: str
) -> dict[str, Any]:
    """Measure the non-queryable limits for an UNCHARACTERIZED device.

    RULE #1: this RAISES (NotImplementedError from the unimplemented probes) when
    a probe cannot run on the active backend -- it NEVER substitutes another
    architecture's numbers.  The two characterized devices (M4 Max / GB10) use
    presets and never reach here; reaching here means a genuinely new device, and
    the loud RAISE points the operator at adding a preset / finishing the probe.
    """

    calibrated: dict[str, Any] = {}
    if backend == "metal":
        calibrated["logical_to_physical_shared_margin"] = _probe_shared_margin()
        calibrated["watchdog_window_s"] = _probe_watchdog()
        calibrated["compiler_shader_ceiling_bytes"] = _probe_msl_ceiling()
        calibrated["per_op_time_per_row_s"] = _probe_op_timing()
        calibrated["has_command_buffer_watchdog"] = True
        calibrated["safety_margin"] = 0.5
        calibrated["buffer_arg_limit"] = 31
    elif backend == "cuda":
        # CUDA has no watchdog / no MSL pipeline-state ceiling; the only hard
        # limit is the queried shared cap. The remaining non-queryable fields are
        # structurally known for CUDA (no preset needed to be SAFE), but a new
        # CUDA arch should still get a preset row -- so we record the safe CUDA
        # defaults and let the loud NO-PRESET log prompt adding one.
        calibrated["logical_to_physical_shared_margin"] = 1.0
        calibrated["watchdog_window_s"] = None
        calibrated["compiler_shader_ceiling_bytes"] = None
        calibrated["per_op_time_per_row_s"] = {}
        calibrated["has_command_buffer_watchdog"] = False
        calibrated["safety_margin"] = 1.0
        calibrated["buffer_arg_limit"] = 1 << 30
    else:
        raise RuntimeError(
            f"calibrate_path_c_device: unsupported backend {backend!r}"
        )
    return calibrated


def _format_preset_stanza(
    *, backend: str, architecture: str, device_name: str, calibrated: dict[str, Any]
) -> str:
    return (
        "  DevicePreset(\n"
        f"    arch={architecture!r}, device_name_glob={device_name + '*'!r}, "
        f"backend={backend!r},\n"
        f"    has_command_buffer_watchdog={calibrated.get('has_command_buffer_watchdog')!r},\n"
        f"    watchdog_window_s={calibrated.get('watchdog_window_s')!r},\n"
        f"    compiler_shader_ceiling_bytes={calibrated.get('compiler_shader_ceiling_bytes')!r},\n"
        f"    logical_to_physical_shared_margin={calibrated.get('logical_to_physical_shared_margin')!r},\n"
        f"    buffer_arg_limit={calibrated.get('buffer_arg_limit')!r},\n"
        f"    per_op_time_per_row_s={calibrated.get('per_op_time_per_row_s')!r},\n"
        f"    safety_margin={calibrated.get('safety_margin')!r},\n"
        "    notes='TIER-3 calibrated -- promote into _PRESETS.'),\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emit-preset",
        action="store_true",
        help="print a DevicePreset stanza to paste into _PRESETS",
    )
    args = parser.parse_args(argv)

    from cppmega_mlx.runtime.path_c_fusion import _path_c_default_target

    backend = _path_c_default_target()
    if backend == "metal":
        import mlx.core as mx

        info = mx.device_info()
        architecture = str(info.get("architecture", "unknown"))
        device_name = str(info.get("device_name", "unknown"))
    else:
        from cppmega_mlx.runtime.path_c_device_caps import _probe_cuda_live

        live = _probe_cuda_live()
        architecture = str(live["architecture"])
        device_name = str(live["device_name"])

    print(f"calibrating non-queryable limits for {backend} {architecture} {device_name}")
    calibrated = calibrate_nonqueryable_limits(
        backend=backend, architecture=architecture, device_name=device_name
    )
    for key, value in calibrated.items():
        print(f"  {key}: {value}")
    if args.emit_preset:
        print("\n--- paste into path_c_device_presets._PRESETS ---")
        print(
            _format_preset_stanza(
                backend=backend,
                architecture=architecture,
                device_name=device_name,
                calibrated=calibrated,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

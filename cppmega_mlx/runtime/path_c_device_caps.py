"""Path-C device-capability probe (TIER 1 + assembly of the three-tier scheme).

This module returns one cached, immutable :class:`DeviceCaps` record per process.
It is the single seam every limit-comparison in the Path-C auto-split planner
plugs into.  It extends the existing binary ``_path_c_default_target()`` (a
metal/cuda *string*) into a capability *record*: the string still selects the
backend, the record carries the numbers.

Three tiers (see ``docs/HW-AWARE-AUTOSPLIT-DESIGN.md``):

* TIER 1 -- LIVE-QUERY every process start for directly-queryable limits
  (Metal ``maxThreadgroupMemoryLength`` via ctypes-objc; CUDA
  ``shared_memory_per_block_optin`` via the CUDA runtime).  RAISE on failure of
  the active backend's probe -- never substitute a guessed constant (RULE #1).
* TIER 2 -- a committed PRESET table (``path_c_device_presets``) for the
  non-queryable limits of characterized architectures.
* TIER 3 -- one-time AUTO-CALIBRATION when no preset matches or a preset is
  proven wrong at runtime (wired in steps 7-8); it self-corrects, persists to a
  cache, and LOGS LOUDLY -- it is never a silent fallback.

The cache (read/write helpers below) is keyed on the full device + toolchain
identity so a driver / compiler / MLX bump invalidates stale calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import functools
import json
import logging
import os
import platform
from pathlib import Path
from typing import Any

from cppmega_mlx.runtime.path_c_fusion import _path_c_default_target
from cppmega_mlx.runtime.path_c_device_presets import (
    DevicePreset,
    preset_for_identity,
)

_LOG = logging.getLogger("cppmega_mlx.path_c_device_caps")

# Portable Metal buffer-argument floor.  If a family table or live probe ever
# yields a LOWER cap, the lower one is used (and a segment that cannot fit
# RAISES).  Retained as the conservative floor per design §1.2.
_METAL_PORTABLE_BUFFER_ARG_FLOOR = 31

# Family-keyed Metal buffer-argument ABI limit (not a queryable scalar).
_METAL_FAMILY_BUFFER_ARG_LIMIT = {
    "applegpu_g16s": 31,  # M4 family (Apple9 / Metal3)
}

# CUDA has no 31-arg ABI wall: sentinel "effectively unbounded".
_CUDA_BUFFER_ARG_SENTINEL = 1 << 30

_CACHE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class DeviceCaps:
    """One immutable, cached capability record for the active Path-C backend."""

    backend: str  # "metal" | "cuda"
    device_name: str
    architecture: str
    os_driver: str  # macOS build / CUDA driver version (cache key + invalidation)
    # --- TIER 1: live-queried hard limits the split must respect ---
    threadgroup_mem_bytes: int  # Metal maxThreadgroupMemoryLength | CUDA optin cap
    static_shared_mem_bytes: int  # CUDA shared_memory_per_block | == threadgroup on Metal
    max_threads_per_block: int
    warp_size: int
    buffer_arg_limit: int  # Metal 31 (family const) | CUDA unbounded sentinel
    # --- TIER 2/3: preset-or-calibrated, non-queryable ---
    has_command_buffer_watchdog: bool
    watchdog_window_s: float | None
    logical_to_physical_shared_margin: float
    msl_pipeline_state_ceiling_bytes: int | None
    per_op_time_per_row_s: dict[str, float]
    effective_flop_s: float
    effective_bytes_s: float
    safety_margin: float
    # Device-characterized fused-segment op-count caps (Metal 2/1; CUDA None/None).
    # The PRIMARY split mechanism; the MSL-byte / watchdog-time estimate predicates
    # are device-grounded backstops layered on top (design §3.3-3.4).
    forward_max_segment_nodes: int | None = None
    backward_max_segment_nodes: int | None = None
    # provenance per field: "queried" | "family-const" | "preset" | "calibrated" | "cache"
    source: dict[str, str] = field(default_factory=dict)

    @property
    def shared_scratch_trigger_bytes(self) -> int:
        """Logical alloc_shared total above which Metal pooling/CUDA demote fires.

        Derived from the queried threadgroup cap and the preset/calibrated
        logical->physical packing margin: a logical total above
        ``threadgroup_mem_bytes / margin`` would inflate past the physical cap.
        Replaces the hardcoded ``_METAL_SHARED_SCRATCH_TRIGGER_BYTES`` (28672).
        """

        margin = self.logical_to_physical_shared_margin
        if margin <= 0:
            raise ValueError(
                "path_c_device_caps: logical_to_physical_shared_margin must be "
                f">0 to derive the shared-scratch trigger (got {margin!r} on "
                f"{self.device_name})"
            )
        return int(self.threadgroup_mem_bytes / margin)


# ---------------------------------------------------------------------------
# TIER 1 -- Metal live probe (ctypes-objc, no new dependency).
# ---------------------------------------------------------------------------


def _probe_metal_live() -> dict[str, Any]:
    """Live-query the Metal device caps the split must respect.

    RAISE on failure -- never substitute a guessed constant (RULE #1).
    """

    import ctypes
    import ctypes.util

    objc_path = ctypes.util.find_library("objc")
    metal_path = ctypes.util.find_library("Metal")
    if objc_path is None or metal_path is None:
        raise RuntimeError(
            "path_c_device_caps: cannot locate libobjc / Metal framework on a "
            "Metal target -- cannot query maxThreadgroupMemoryLength"
        )
    objc = ctypes.CDLL(objc_path)
    metal = ctypes.CDLL(metal_path)
    # sel_registerName returns a SEL pointer; its restype MUST be set or the
    # pointer is truncated to a 32-bit int and objc_msgSend segfaults.
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
    dev = metal.MTLCreateSystemDefaultDevice()
    if not dev:
        raise RuntimeError(
            "path_c_device_caps: MTLCreateSystemDefaultDevice() returned nil on "
            "a Metal target -- cannot query maxThreadgroupMemoryLength"
        )

    def sel(name: str) -> Any:
        return objc.sel_registerName(name.encode())

    def msg_ulong(d: Any, name: str) -> int:
        fn = objc.objc_msgSend
        fn.restype = ctypes.c_ulong
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        return int(fn(d, sel(name)))

    def supports_family(d: Any, fam_id: int) -> bool:
        fn = objc.objc_msgSend
        fn.restype = ctypes.c_bool
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong]
        return bool(fn(d, sel("supportsFamily:"), fam_id))

    def msg_str(d: Any, name: str) -> str:
        fn = objc.objc_msgSend
        fn.restype = ctypes.c_void_p
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        nsstr = fn(d, sel(name))
        if not nsstr:
            return ""
        fn2 = objc.objc_msgSend
        fn2.restype = ctypes.c_char_p
        fn2.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        raw = fn2(nsstr, sel("UTF8String"))
        return raw.decode() if raw else ""

    threadgroup = msg_ulong(dev, "maxThreadgroupMemoryLength")
    if threadgroup <= 0:
        raise RuntimeError(
            "path_c_device_caps: maxThreadgroupMemoryLength query failed on Metal "
            f"target {msg_str(dev, 'name')!r} (returned {threadgroup})"
        )
    return {
        "threadgroup_mem_bytes": threadgroup,  # 32768 on M4 Max
        "device_name": msg_str(dev, "name"),
        "supports_apple9": supports_family(dev, 1009),  # MTLGPUFamilyApple9
    }


def _metal_tvm_attrs() -> dict[str, int]:
    """``max_threads_per_block`` / ``warp_size`` from the TVM device-API path.

    ``tvm.metal(0).max_shared_memory_per_block`` is a no-op (returns None) so it
    is NOT used here; the threadgroup cap comes from the ctypes probe.  TVM's
    warp_size override (32) corrects the stale Target default of 16.
    """

    import tvm

    dev = tvm.metal(0)
    return {
        "max_threads_per_block": int(dev.max_threads_per_block),
        "warp_size": int(dev.warp_size),
    }


def _metal_architecture() -> str:
    import mlx.core as mx

    info = mx.device_info()
    arch = str(info.get("architecture", ""))
    if not arch:
        raise RuntimeError(
            "path_c_device_caps: mx.device_info() did not report an architecture "
            "on a Metal target"
        )
    return arch


# ---------------------------------------------------------------------------
# TIER 1 -- CUDA live probe (CUDA runtime, no extra deps).
# ---------------------------------------------------------------------------


def _probe_cuda_live() -> dict[str, Any]:
    """Live-query the CUDA device caps.  RAISE on failure (RULE #1).

    Prefers ``torch.cuda.get_device_properties`` (no extra deps); falls back to
    the CUDA runtime ``cudaDeviceGetAttribute`` with the SYMBOLIC enum values
    (97 = MaxSharedMemoryPerBlockOptin, 8 = SharedMemPerBlock).
    """

    # --- torch fast path ---
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            optin = getattr(props, "shared_memory_per_block_optin", None)
            static = getattr(props, "shared_memory_per_block", None)
            if optin is None:
                optin = getattr(props, "max_shared_memory_per_block_optin", None)
            if static is None:
                static = getattr(props, "shared_memory_per_multiprocessor", None)
            cc_major, cc_minor = torch.cuda.get_device_capability(0)
            if optin and static:
                return {
                    "threadgroup_mem_bytes": int(optin),
                    "static_shared_mem_bytes": int(static),
                    "max_threads_per_block": int(props.max_threads_per_block),
                    "warp_size": int(getattr(props, "warp_size", 32) or 32),
                    "device_name": str(props.name),
                    "architecture": "sm_%d%d" % (cc_major, cc_minor),
                }
    except ImportError:
        pass

    # --- ctypes CUDA-runtime fallback (symbolic enums, never raw guesses) ---
    import ctypes

    try:
        lib = ctypes.CDLL("libcudart.so")
    except OSError as exc:
        raise RuntimeError(
            "path_c_device_caps: cannot load libcudart.so on a CUDA target -- "
            f"cannot query shared_memory_per_block_optin ({exc})"
        ) from exc

    def attr(code: int, name: str) -> int:
        value = ctypes.c_int(0)
        rc = lib.cudaDeviceGetAttribute(ctypes.byref(value), code, ctypes.c_int(0))
        if rc != 0 or value.value <= 0:
            raise RuntimeError(
                "path_c_device_caps: cudaDeviceGetAttribute("
                f"{name}={code}) failed (rc={rc}, value={value.value}) on a CUDA "
                "target"
            )
        return int(value.value)

    optin = attr(97, "cudaDevAttrMaxSharedMemoryPerBlockOptin")
    static = attr(8, "cudaDevAttrMaxSharedMemoryPerBlock")
    max_threads = attr(1, "cudaDevAttrMaxThreadsPerBlock")
    cc_major = attr(75, "cudaDevAttrComputeCapabilityMajor")
    cc_minor = attr(76, "cudaDevAttrComputeCapabilityMinor")
    name = ctypes.create_string_buffer(256)
    lib.cudaDeviceGetPCIBusId(name, ctypes.c_int(256), ctypes.c_int(0))
    return {
        "threadgroup_mem_bytes": optin,
        "static_shared_mem_bytes": static,
        "max_threads_per_block": max_threads,
        "warp_size": 32,
        "device_name": "NVIDIA CUDA device %s" % name.value.decode(errors="replace"),
        "architecture": "sm_%d%d" % (cc_major, cc_minor),
    }


# ---------------------------------------------------------------------------
# Identity + OS/driver string (cache key components).
# ---------------------------------------------------------------------------


def _os_driver_string(backend: str) -> str:
    if backend == "metal":
        mac = platform.mac_ver()[0] or platform.release()
        return f"macOS-{mac}-{platform.machine()}"
    # CUDA: driver version where available.
    try:
        import torch

        ver = getattr(torch.version, "cuda", None)
        return f"cuda-{ver}-{platform.machine()}"
    except Exception:
        return f"cuda-unknown-{platform.machine()}"


def _toolchain_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    try:
        import mlx.core as mx

        versions["mlx_version"] = str(getattr(mx, "__version__", "unknown"))
    except Exception:
        versions["mlx_version"] = "unknown"
    try:
        import tilelang

        versions["tilelang_version"] = str(getattr(tilelang, "__version__", "unknown"))
    except Exception:
        versions["tilelang_version"] = "unknown"
    return versions


# ---------------------------------------------------------------------------
# Cache (TIER-3 persistence; read at startup, written on calibration).
# ---------------------------------------------------------------------------


def _cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "cppmega_mlx" / "path_c_device_caps"


def _cache_path(architecture: str, device_name: str) -> Path:
    safe = "%s__%s" % (architecture, device_name.replace("/", "_").replace(" ", "_"))
    return _cache_dir() / f"{safe}.json"


def _identity_key(backend: str, architecture: str, device_name: str) -> dict[str, str]:
    key = {
        "backend": backend,
        "architecture": architecture,
        "device_name": device_name,
        "os_driver": _os_driver_string(backend),
    }
    key.update(_toolchain_versions())
    return key


def load_calibration_cache(
    backend: str, architecture: str, device_name: str
) -> dict[str, Any] | None:
    """Return the calibrated payload iff a VALID cache matches the live identity.

    A cache file whose key mismatches the live identity, or is corrupt, is
    ignored (TIER-3 will recompute the correct value, logged) -- never trusted
    as-is (RULE #1 cache integrity, design §6.7).
    """

    path = _cache_path(architecture, device_name)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        _LOG.warning(
            "path_c_device_caps: ignoring corrupt cache %s (%s) -- will recompute",
            path,
            exc,
        )
        return None
    if payload.get("schema_version") != _CACHE_SCHEMA_VERSION:
        _LOG.warning(
            "path_c_device_caps: cache %s schema_version mismatch -- will recompute",
            path,
        )
        return None
    live_key = _identity_key(backend, architecture, device_name)
    cache_key = payload.get("key", {})
    for field_name, live_value in live_key.items():
        if cache_key.get(field_name) != live_value:
            _LOG.warning(
                "path_c_device_caps: cache %s invalidated (%s: cached=%r live=%r) "
                "-- will recompute",
                path,
                field_name,
                cache_key.get(field_name),
                live_value,
            )
            return None
    calibrated = payload.get("calibrated")
    if not isinstance(calibrated, dict):
        return None
    return calibrated


def write_calibration_cache(
    backend: str,
    architecture: str,
    device_name: str,
    calibrated: dict[str, Any],
    *,
    provenance: str = "tier3-autocalibration",
    preset_miss: dict[str, Any] | None = None,
) -> Path:
    """Persist a TIER-3 calibration result keyed on the full live identity."""

    import datetime

    path = _cache_path(architecture, device_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "key": _identity_key(backend, architecture, device_name),
        "calibrated": calibrated,
        "provenance": provenance,
        "preset_miss": preset_miss or {},
        "calibrated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


# ---------------------------------------------------------------------------
# TIER 2/3 resolution of the non-queryable limits.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _NonQueryable:
    has_command_buffer_watchdog: bool
    watchdog_window_s: float | None
    logical_to_physical_shared_margin: float
    msl_pipeline_state_ceiling_bytes: int | None
    buffer_arg_limit: int
    per_op_time_per_row_s: dict[str, float]
    effective_flop_s: float
    effective_bytes_s: float
    safety_margin: float
    forward_max_segment_nodes: int | None
    backward_max_segment_nodes: int | None
    provenance: str  # "preset" | "cache" | "calibrated"


def _preset_to_nonqueryable(preset: DevicePreset, *, provenance: str) -> _NonQueryable:
    return _NonQueryable(
        has_command_buffer_watchdog=preset.has_command_buffer_watchdog,
        watchdog_window_s=preset.watchdog_window_s,
        logical_to_physical_shared_margin=preset.logical_to_physical_shared_margin,
        msl_pipeline_state_ceiling_bytes=preset.compiler_shader_ceiling_bytes,
        buffer_arg_limit=preset.buffer_arg_limit,
        per_op_time_per_row_s=dict(preset.per_op_time_per_row_s),
        effective_flop_s=preset.effective_flop_s,
        effective_bytes_s=preset.effective_bytes_s,
        safety_margin=preset.safety_margin,
        forward_max_segment_nodes=preset.forward_max_segment_nodes,
        backward_max_segment_nodes=preset.backward_max_segment_nodes,
        provenance=provenance,
    )


def _resolve_nonqueryable(
    backend: str, architecture: str, device_name: str
) -> _NonQueryable:
    """Resolve the non-queryable limits via the §4.4 read order.

    1. Valid cache matching the live key -> use it (provenance ``cache``).
    2. Else a CHARACTERIZED preset matches -> use it (provenance ``preset``).
    3. Else TIER-3 calibrate (steps 7-8) -> persist -> use, logging ``NO PRESET``.

    A preset always beats a *stale* cache because invalidation already removed
    stale caches; a *valid* cache beats a preset because it is at-least-as-correct
    (a prior TIER-3 correction or an identical-to-preset confirmation for THIS
    exact device/driver).
    """

    preset = preset_for_identity(
        backend=backend, architecture=architecture, device_name=device_name
    )
    cached = load_calibration_cache(backend, architecture, device_name)
    if cached is not None:
        merged = _preset_to_nonqueryable(preset, provenance="cache") if preset else None
        return _merge_cache_over_preset(cached, merged, backend)
    if preset is not None:
        return _preset_to_nonqueryable(preset, provenance="preset")

    # No preset and no cache -> TIER-3 calibration (wired in steps 7-8).
    return _tier3_calibrate(backend, architecture, device_name)


def _merge_cache_over_preset(
    cached: dict[str, Any],
    preset_nq: _NonQueryable | None,
    backend: str,
) -> _NonQueryable:
    """Overlay a valid calibration cache on top of (optional) preset defaults."""

    base = preset_nq
    if base is None:
        # Cache exists but no characterized preset: the cache fully defines the
        # non-queryable limits (it was written by TIER-3 for this exact device).
        base = _NonQueryable(
            has_command_buffer_watchdog=bool(
                cached.get("has_command_buffer_watchdog", backend == "metal")
            ),
            watchdog_window_s=cached.get("watchdog_window_s"),
            logical_to_physical_shared_margin=float(
                cached.get("logical_to_physical_shared_margin", 1.0)
            ),
            msl_pipeline_state_ceiling_bytes=cached.get("compiler_shader_ceiling_bytes"),
            buffer_arg_limit=int(
                cached.get(
                    "buffer_arg_limit",
                    _METAL_PORTABLE_BUFFER_ARG_FLOOR
                    if backend == "metal"
                    else _CUDA_BUFFER_ARG_SENTINEL,
                )
            ),
            per_op_time_per_row_s=dict(cached.get("per_op_time_per_row_s", {})),
            effective_flop_s=float(cached.get("effective_flop_s", 0.0)),
            effective_bytes_s=float(cached.get("effective_bytes_s", 0.0)),
            safety_margin=float(cached.get("safety_margin", 0.5)),
            forward_max_segment_nodes=cached.get("forward_max_segment_nodes"),
            backward_max_segment_nodes=cached.get("backward_max_segment_nodes"),
            provenance="cache",
        )
        return base
    return _NonQueryable(
        has_command_buffer_watchdog=base.has_command_buffer_watchdog,
        watchdog_window_s=cached.get("watchdog_window_s", base.watchdog_window_s),
        logical_to_physical_shared_margin=float(
            cached.get(
                "logical_to_physical_shared_margin",
                base.logical_to_physical_shared_margin,
            )
        ),
        msl_pipeline_state_ceiling_bytes=cached.get(
            "compiler_shader_ceiling_bytes", base.msl_pipeline_state_ceiling_bytes
        ),
        buffer_arg_limit=base.buffer_arg_limit,
        per_op_time_per_row_s={
            **base.per_op_time_per_row_s,
            **cached.get("per_op_time_per_row_s", {}),
        },
        effective_flop_s=float(
            cached.get("effective_flop_s", base.effective_flop_s)
        ),
        effective_bytes_s=float(
            cached.get("effective_bytes_s", base.effective_bytes_s)
        ),
        safety_margin=base.safety_margin,
        forward_max_segment_nodes=base.forward_max_segment_nodes,
        backward_max_segment_nodes=base.backward_max_segment_nodes,
        provenance="cache",
    )


def _tier3_calibrate(
    backend: str, architecture: str, device_name: str
) -> _NonQueryable:
    """TIER-3: no preset matched -> calibrate the non-queryable limits, persist, LOG.

    The actual calibration probes live in ``scripts/calibrate_path_c_device.py``
    (imported lazily so importing this module never requires a GPU).  This is
    self-correction, not a silent fallback (RULE #1): the real thresholds are
    measured, used, and persisted, and the absence of a preset is LOGGED LOUDLY
    so CI can grep ``NO PRESET`` and open a task to add one.
    """

    from scripts.calibrate_path_c_device import calibrate_nonqueryable_limits

    calibrated = calibrate_nonqueryable_limits(
        backend=backend, architecture=architecture, device_name=device_name
    )
    cache_path = write_calibration_cache(
        backend, architecture, device_name, calibrated, provenance="tier3-autocalibration"
    )
    _LOG.warning(
        "path_c_device_caps: NO PRESET for arch=%s device=%s; auto-calibrated "
        "watchdog_window_s=%s msl_ceiling=%s margin=%s -> persisted to %s; "
        "ADD A PRESET ENTRY.",
        architecture,
        device_name,
        calibrated.get("watchdog_window_s"),
        calibrated.get("compiler_shader_ceiling_bytes"),
        calibrated.get("logical_to_physical_shared_margin"),
        cache_path,
    )
    return _merge_cache_over_preset(calibrated, None, backend)


# ---------------------------------------------------------------------------
# Assembly + public entry point.
# ---------------------------------------------------------------------------


def _resolve_buffer_arg_limit(backend: str, architecture: str, nq: _NonQueryable) -> int:
    if backend == "cuda":
        return _CUDA_BUFFER_ARG_SENTINEL
    # Metal: take the LOWER of the family table, the non-queryable resolution,
    # and the portable floor (a lower probe/family cap must win; RULE #1).
    family = _METAL_FAMILY_BUFFER_ARG_LIMIT.get(architecture, _METAL_PORTABLE_BUFFER_ARG_FLOOR)
    return min(family, nq.buffer_arg_limit, _METAL_PORTABLE_BUFFER_ARG_FLOOR)


def _assemble(backend: str, live: dict[str, Any], nq: _NonQueryable) -> DeviceCaps:
    if backend == "metal":
        tvm_attrs = _metal_tvm_attrs()
        architecture = _metal_architecture()
        threadgroup = int(live["threadgroup_mem_bytes"])
        static_shared = threadgroup  # Metal: static == threadgroup
        max_threads = tvm_attrs["max_threads_per_block"]
        warp = tvm_attrs["warp_size"]
        device_name = str(live["device_name"])
    else:
        architecture = str(live["architecture"])
        threadgroup = int(live["threadgroup_mem_bytes"])
        static_shared = int(live["static_shared_mem_bytes"])
        max_threads = int(live["max_threads_per_block"])
        warp = int(live["warp_size"])
        device_name = str(live["device_name"])

    buffer_arg_limit = _resolve_buffer_arg_limit(backend, architecture, nq)

    source = {
        "threadgroup_mem_bytes": "queried",
        "static_shared_mem_bytes": "queried",
        "max_threads_per_block": "queried",
        "warp_size": "queried",
        "device_name": "queried",
        "architecture": "queried",
        "buffer_arg_limit": "family-const",
        "has_command_buffer_watchdog": nq.provenance
        if backend == "metal"
        else "family-const",
        "watchdog_window_s": nq.provenance,
        "logical_to_physical_shared_margin": nq.provenance,
        "msl_pipeline_state_ceiling_bytes": nq.provenance,
        "per_op_time_per_row_s": nq.provenance,
        "effective_flop_s": nq.provenance,
        "effective_bytes_s": nq.provenance,
        "safety_margin": nq.provenance,
        "forward_max_segment_nodes": nq.provenance,
        "backward_max_segment_nodes": nq.provenance,
    }

    return DeviceCaps(
        backend=backend,
        device_name=device_name,
        architecture=architecture,
        os_driver=_os_driver_string(backend),
        threadgroup_mem_bytes=threadgroup,
        static_shared_mem_bytes=static_shared,
        max_threads_per_block=max_threads,
        warp_size=warp,
        buffer_arg_limit=buffer_arg_limit,
        has_command_buffer_watchdog=(
            nq.has_command_buffer_watchdog if backend == "metal" else False
        ),
        watchdog_window_s=nq.watchdog_window_s if backend == "metal" else None,
        logical_to_physical_shared_margin=nq.logical_to_physical_shared_margin,
        msl_pipeline_state_ceiling_bytes=(
            nq.msl_pipeline_state_ceiling_bytes if backend == "metal" else None
        ),
        per_op_time_per_row_s=nq.per_op_time_per_row_s,
        effective_flop_s=nq.effective_flop_s,
        effective_bytes_s=nq.effective_bytes_s,
        safety_margin=nq.safety_margin,
        forward_max_segment_nodes=(
            nq.forward_max_segment_nodes if backend == "metal" else None
        ),
        backward_max_segment_nodes=(
            nq.backward_max_segment_nodes if backend == "metal" else None
        ),
        source=source,
    )


@functools.lru_cache(maxsize=1)
def device_caps() -> DeviceCaps:
    """Return the cached, immutable capability record for the active backend.

    Probed once at startup, cached for the process.  RAISE on TIER-1 probe
    failure of the active backend -- never substitute a guessed constant.
    """

    backend = _path_c_default_target()
    if backend == "metal":
        live = _probe_metal_live()
        architecture = _metal_architecture()
        device_name = str(live["device_name"])
    elif backend == "cuda":
        live = _probe_cuda_live()
        architecture = str(live["architecture"])
        device_name = str(live["device_name"])
    else:
        raise RuntimeError(
            f"path_c_device_caps: unsupported backend {backend!r} "
            "(expected 'metal' or 'cuda')"
        )
    nq = _resolve_nonqueryable(backend, architecture, device_name)
    return _assemble(backend, live, nq)


def reset_device_caps_cache() -> None:
    """Clear the process-level ``device_caps()`` lru_cache (tests/calibration)."""

    device_caps.cache_clear()

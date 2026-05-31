"""TIER-1 probe + TIER-2 preset + cache self-check (design §7.3).

These tests assert the device-capability probe returns the expected queried
values on the active backend, that the seed presets match the characterized
devices, and that the cache round-trips and invalidates on identity change.

Backend-specific assertions are gated on the live backend so the suite passes on
both M4 Max (Metal) and gb10 (CUDA).
"""

from __future__ import annotations

import json

import pytest

from cppmega_mlx.runtime import path_c_device_caps as caps_mod
from cppmega_mlx.runtime.path_c_device_caps import (
    DeviceCaps,
    device_caps,
    load_calibration_cache,
    reset_device_caps_cache,
    write_calibration_cache,
)
from cppmega_mlx.runtime.path_c_device_presets import (
    all_presets,
    preset_for_identity,
)
from cppmega_mlx.runtime.path_c_fusion import _path_c_default_target


def test_device_caps_record_is_cached_and_frozen():
    reset_device_caps_cache()
    a = device_caps()
    b = device_caps()
    assert a is b  # lru_cache: one record per process
    assert isinstance(a, DeviceCaps)
    with pytest.raises(Exception):
        a.threadgroup_mem_bytes = 0  # frozen dataclass


def test_metal_tier1_queried_values():
    if _path_c_default_target() != "metal":
        pytest.skip("metal-only assertions")
    reset_device_caps_cache()
    c = device_caps()
    assert c.backend == "metal"
    assert c.threadgroup_mem_bytes == 32768
    assert c.max_threads_per_block == 1024
    assert c.warp_size == 32
    assert c.buffer_arg_limit == 31
    # static == threadgroup on Metal
    assert c.static_shared_mem_bytes == c.threadgroup_mem_bytes
    # provenance audit
    assert c.source["threadgroup_mem_bytes"] == "queried"
    assert c.source["buffer_arg_limit"] == "family-const"


def test_metal_preset_supplies_nonqueryable_limits():
    if _path_c_default_target() != "metal":
        pytest.skip("metal-only assertions")
    reset_device_caps_cache()
    c = device_caps()
    assert c.architecture == "applegpu_g16s"
    assert c.has_command_buffer_watchdog is True
    assert c.watchdog_window_s == 5.0
    assert c.msl_pipeline_state_ceiling_bytes == 140000
    assert c.logical_to_physical_shared_margin == 3.7
    assert c.safety_margin == 0.5
    # derived shared-scratch trigger replaces hardcoded 28672 (32768 / 3.7)
    assert c.shared_scratch_trigger_bytes == int(32768 / 3.7)


def test_cuda_tier1_queried_values():
    if _path_c_default_target() != "cuda":
        pytest.skip("cuda-only assertions")
    reset_device_caps_cache()
    c = device_caps()
    assert c.backend == "cuda"
    assert c.threadgroup_mem_bytes > 0
    assert c.static_shared_mem_bytes > 0
    assert c.has_command_buffer_watchdog is False
    assert c.watchdog_window_s is None
    assert c.msl_pipeline_state_ceiling_bytes is None
    assert c.buffer_arg_limit == (1 << 30)
    # gb10 observed values (skip-tolerant if a different CUDA device runs this)
    if c.architecture == "sm_121":
        assert c.threadgroup_mem_bytes == 101376
        assert c.static_shared_mem_bytes == 49152


def test_seed_presets_present_and_unique():
    presets = all_presets()
    archs = [p.arch for p in presets]
    assert "applegpu_g16s" in archs
    assert "sm_121" in archs
    # the two seed entries are characterized; placeholders are not
    m4 = preset_for_identity(
        backend="metal", architecture="applegpu_g16s", device_name="Apple M4 Max"
    )
    assert m4 is not None and m4.watchdog_window_s == 5.0
    gb10 = preset_for_identity(
        backend="cuda", architecture="sm_121", device_name="NVIDIA GB10"
    )
    assert gb10 is not None and gb10.has_command_buffer_watchdog is False


def test_placeholder_preset_routes_to_none_not_silent_inherit():
    # An UNCHARACTERIZED arch must NOT silently inherit M4 Max values (RULE #1).
    miss = preset_for_identity(
        backend="metal", architecture="applegpu_g17p", device_name="Apple M5 Pro"
    )
    assert miss is None
    # A totally-unknown arch is also a miss.
    miss2 = preset_for_identity(
        backend="metal", architecture="applegpu_zz", device_name="Apple Zz"
    )
    assert miss2 is None


def test_cache_roundtrip_and_invalidation(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    arch = "test_arch_x"
    dev = "Test Device X"
    backend = _path_c_default_target()
    calibrated = {
        "watchdog_window_s": 5.3,
        "compiler_shader_ceiling_bytes": 152000,
        "logical_to_physical_shared_margin": 3.66,
    }
    path = write_calibration_cache(backend, arch, dev, calibrated)
    assert path.exists()
    loaded = load_calibration_cache(backend, arch, dev)
    assert loaded is not None
    assert loaded["watchdog_window_s"] == 5.3

    # Bump os_driver -> invalidation -> ignore the stale file.
    payload = json.loads(path.read_text())
    payload["key"]["os_driver"] = "totally-different-driver"
    path.write_text(json.dumps(payload))
    assert load_calibration_cache(backend, arch, dev) is None

    # Corrupt JSON -> ignored, not trusted.
    path.write_text("{ not json")
    assert load_calibration_cache(backend, arch, dev) is None


def test_cache_schema_version_mismatch_invalidates(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    arch = "test_arch_y"
    dev = "Test Device Y"
    backend = _path_c_default_target()
    path = write_calibration_cache(backend, arch, dev, {"watchdog_window_s": 1.0})
    payload = json.loads(path.read_text())
    payload["schema_version"] = 999
    path.write_text(json.dumps(payload))
    assert load_calibration_cache(backend, arch, dev) is None


def test_cuda_optin_cap_uses_queried_caps_not_silent_floor():
    """RULE #1: the CUDA opt-in cap is queried; no silent 0x18C00 floor remains."""
    from cppmega_mlx.runtime import path_c_fusion_schedules as sched
    import inspect

    src = inspect.getsource(sched._cuda_shared_memory_optin_cap_bytes)
    # The silent except:pass + hardcoded "return 0x18C00" floor must be gone
    # (the constant may still appear in an explanatory comment, but never as a
    # returned fallback value).
    assert "return 0x18C00" not in src
    assert "except Exception:\n        pass" not in src
    assert "device_caps" in src
    # On a Metal host it returns None (no CUDA demote); on CUDA it returns the
    # queried opt-in cap (or RAISES if the probe fails -- never a guessed floor).
    value = sched._cuda_shared_memory_optin_cap_bytes()
    if _path_c_default_target() == "metal":
        assert value is None
    else:
        assert value == device_caps().threadgroup_mem_bytes

"""Path-C device PRESET table (TIER 2 of the hardware-aware auto-split scheme).

This module holds the committed preset table of *non-queryable* device limits for
architectures we have already characterized (the macOS GPU command-buffer
watchdog window, the ``newComputePipelineState`` MSL shader-size ceiling, the
logical->physical threadgroup packing margin, the family buffer-argument ABI
limit, and the per-op GPU-time-per-row coefficients).

It is the TIER-2 "preset" tier of the FlashAttention-2 / cuDNN-style three-tier
scheme described in ``docs/HW-AWARE-AUTOSPLIT-DESIGN.md`` (TIER 1 = live-query,
TIER 2 = this preset table, TIER 3 = auto-calibration when no preset matches).

RULE #1: a device with no matching preset does NOT silently inherit another
architecture's numbers.  :func:`preset_for_identity` returns ``None`` on a miss,
and the caller (``path_c_device_caps``) must then route to TIER-3 calibration and
LOG LOUDLY.  The schema-only placeholder entries below carry ``None`` for the
non-queryable limits precisely so they cannot be mistaken for characterized data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatch


@dataclass(frozen=True)
class DevicePreset:
    """Non-queryable device limits for one characterized architecture.

    The queryable limits (threadgroup-mem bytes, static shared bytes, max threads,
    warp size, device name, architecture) come from TIER-1 live probing and are
    NOT stored here -- this table only carries what no API exposes.
    """

    arch: str
    device_name_glob: str
    backend: str  # "metal" | "cuda"
    # --- the non-queryable limits (the whole reason this table exists) ---
    has_command_buffer_watchdog: bool
    watchdog_window_s: float | None
    compiler_shader_ceiling_bytes: int | None
    logical_to_physical_shared_margin: float
    buffer_arg_limit: int
    per_op_time_per_row_s: dict[str, float] = field(default_factory=dict)
    per_op_ref_shape: dict[str, int] = field(default_factory=dict)
    effective_flop_s: float = 0.0
    effective_bytes_s: float = 0.0
    safety_margin: float = 0.5
    notes: str = ""
    # When True this is a schema-only placeholder for an UNCHARACTERIZED device:
    # it must route to TIER-3 calibration, never be used directly.
    characterized: bool = True


# ---------------------------------------------------------------------------
# Seed entries (the two characterized devices) + schema-only placeholders.
# ---------------------------------------------------------------------------
#
# The M4 Max preset values are SEEDED so the derived caps EQUAL the current
# hand-tuned constants at the ``local_gb10_quarter`` scale (depth=13 hidden=3584
# max_seq=4096) -- this is the §7.1 acceptance criterion:
#   * watchdog_window_s=5.0 * safety_margin=0.5 => 2.5 s budget; with
#     per_op_time_per_row_s for the heavy bwd/fwd ops the derived
#     max_rows_per_launch lands on the hand-tuned 64 (see §4.7 and the estimator).
#   * compiler_shader_ceiling_bytes=140_000 sits between the measured 116 KiB-OK
#     and 176 KiB-crash band so the 4-op forward chain_3_7 splits into 2-op pieces.
#   * logical_to_physical_shared_margin=3.7 reproduces the ~29.5 KiB physical /
#     8 KiB logical observed on the mamba3 backward demote target.
#   * buffer_arg_limit=31 is the Metal family ABI constant (Apple9 / Metal3).

_PRESETS: tuple[DevicePreset, ...] = (
    DevicePreset(  # Apple M4 Max -- Metal, applegpu_g16s (Apple9 / Metal3)
        arch="applegpu_g16s",
        device_name_glob="Apple M4*",
        backend="metal",
        has_command_buffer_watchdog=True,
        watchdog_window_s=5.0,  # ~5-6 s observed; conservative 5.0
        compiler_shader_ceiling_bytes=140_000,  # between 116 KiB OK and 176 KiB crash
        logical_to_physical_shared_margin=3.7,  # ~29.5 KiB physical / 8 KiB logical
        buffer_arg_limit=31,
        per_op_time_per_row_s={  # @ local_gb10_quarter ref shape (S=4096)
            "sparse_mla_fp8_apply_bwd": 12.0 / 4096,  # ~12 s monolithic / 4096 rows
            "attention_qkv_projection_bwd": 10.0 / 4096,  # ~10 s monolithic / 4096 rows
            "sparse_mla_fp8_apply": 12.0 / 4096,  # forward analog (watchdog row-chunk)
            "attention_qkv_projection": 10.0 / 4096,
            "residual_rmsnorm_bwd": 0.08 / 4096,
            "entry_rmsnorm_bwd": 0.08 / 4096,
            "mamba3_mimo_bwd": 0.0,  # recurrent -> time-chunked, not row-timed
            "m2rnn_bwd": 0.0,
        },
        per_op_ref_shape={"hidden": 3584, "state_dim": 128, "max_seq": 4096},
        effective_flop_s=8.0e12,
        effective_bytes_s=4.0e11,  # M4 Max GPU roofline (calibrated seed)
        safety_margin=0.5,
        notes=(
            "Hand-tuned splits 2/1/64/8/28K/8K reproduced by these presets at "
            "local_gb10_quarter."
        ),
    ),
    DevicePreset(  # NVIDIA GB10 -- CUDA, sm_121 (Blackwell, integrated SoC)
        arch="sm_121",
        device_name_glob="NVIDIA GB10*",
        backend="cuda",
        has_command_buffer_watchdog=False,  # 105.17 s single kernel completed
        watchdog_window_s=None,
        compiler_shader_ceiling_bytes=None,  # ptxas fails loud; no opaque XPC crash
        logical_to_physical_shared_margin=1.0,  # CUDA demotes by logical bytes
        buffer_arg_limit=1 << 30,  # effectively unbounded
        per_op_time_per_row_s={},  # unused (no watchdog -> no row/time chunking)
        per_op_ref_shape={},
        effective_flop_s=2.5e14,
        effective_bytes_s=5.0e12,  # GB10 roofline (informational; no chunking)
        safety_margin=1.0,
        notes=(
            "CUDA stays monolithic. Only hard limit is shared_memory_per_block_optin "
            "(101376, queried)."
        ),
    ),
    # --- Forward-compatible placeholders (schema-only, route to TIER-3) ---
    DevicePreset(
        arch="applegpu_g17*",
        device_name_glob="Apple M5*",
        backend="metal",
        has_command_buffer_watchdog=True,
        watchdog_window_s=None,
        compiler_shader_ceiling_bytes=None,
        logical_to_physical_shared_margin=0.0,
        buffer_arg_limit=31,
        notes="UNCHARACTERIZED M5 -- first run TIER-3 calibrates and persists.",
        characterized=False,
    ),
    DevicePreset(
        arch="sm_90",
        device_name_glob="*",
        backend="cuda",
        has_command_buffer_watchdog=False,
        watchdog_window_s=None,
        compiler_shader_ceiling_bytes=None,
        logical_to_physical_shared_margin=1.0,
        buffer_arg_limit=1 << 30,
        notes="UNCHARACTERIZED A100/H-class -- first run TIER-3 calibrates and persists.",
        characterized=False,
    ),
    DevicePreset(
        arch="sm_100",
        device_name_glob="*",
        backend="cuda",
        has_command_buffer_watchdog=False,
        watchdog_window_s=None,
        compiler_shader_ceiling_bytes=None,
        logical_to_physical_shared_margin=1.0,
        buffer_arg_limit=1 << 30,
        notes="UNCHARACTERIZED B200 -- first run TIER-3 calibrates and persists.",
        characterized=False,
    ),
)


def all_presets() -> tuple[DevicePreset, ...]:
    """Return the full preset table (including schema-only placeholders)."""

    return _PRESETS


def preset_for_identity(
    *,
    backend: str,
    architecture: str,
    device_name: str,
) -> DevicePreset | None:
    """Return the unique CHARACTERIZED preset for a device identity, else ``None``.

    Matching is by ``arch`` glob first, then ``device_name_glob`` to disambiguate.
    A schema-only placeholder (``characterized is False``) is treated as **no
    preset** -- it deliberately routes the caller to TIER-3 calibration rather
    than handing back un-characterized ``None`` limits as if they were real.

    RULE #1: if the number of matches is not exactly one, this returns ``None``
    (no preset) -- the caller must then TIER-3 calibrate and LOG LOUDLY, never
    silently pick a "close enough" architecture.
    """

    matches = [
        preset
        for preset in _PRESETS
        if preset.backend == backend
        and fnmatch(architecture, preset.arch)
        and fnmatch(device_name, preset.device_name_glob)
    ]
    if len(matches) != 1:
        return None
    preset = matches[0]
    if not preset.characterized:
        return None
    return preset

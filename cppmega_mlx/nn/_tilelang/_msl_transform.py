"""Tiny helper to wrap mx.fast.metal_kernel dispatch.

This is the Path B vendor-local equivalent of the TVM-Metal lowering produced
by TileLang PR tile-ai/tilelang#799. We do not depend on TileLang at runtime;
instead each port assembles MSL inline and passes it to mx.fast.metal_kernel.

The helper only exists so that:
  - several kernels can share one factory pattern (cached compiled kernel,
    consistent dtype-handling, fail-closed error paths);
  - tests can mock dispatch without touching every kernel module.

It is intentionally narrow: no MSL templating, no dynamic shape rewriting.
The caller writes MSL assuming all dimensions are passed as device buffers
or threadgroup constants.

For TileLang-derived MSL the module also exposes an experimental lowering
helper (``lower_tilelang_to_msl``) that takes a ``@T.prim_func`` PrimFunc and
returns the MLX-callable kernel handle plus thread/grid metadata. Body
extraction is done inline rather than wrapping into ``inline void`` because
Apple's MSL forbids ``threadgroup`` allocations in non-kernel functions.

Z3 roadmap note (cppmega-mlx-cuz wiring):
    This module operates on TileLang's *post-MSL textual output* — it splits
    the emitted MSL kernel signature/body, rewrites a handful of Apple Metal
    builtin aliases, and inlines the body for ``mx.fast.metal_kernel``. None
    of the Z3 roadmap proofs (idea #1 contiguity, #4 bound-check drop, #9
    simd-lift, #10 fp8 dot4 legality) operate at this layer — they all run
    inside TileLang lowering before the MSL string is generated. Therefore
    enabling/disabling those Z3 passes is *independent* of this module: any
    redundant runtime checks the Z3 passes prove away live in the calling
    kernels (e.g. ``fp8_vecmat_path_c.py``), not here. The lowering helper
    below now accepts an optional ``pass_configs`` dict so callers can opt
    their PrimFunc into the actually-shipped Z3 PassConfigs without having
    to re-import TVM in each module.
"""

from __future__ import annotations

import ctypes
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path as _Path
from typing import Any, Callable, Sequence, cast

import mlx.core as mx


def _preload_libz3_for_dev_tilelang() -> None:
    """Preload ``libz3.dylib`` so dev-build ``libtilelang.dylib`` can dlopen.

    The in-tree dev build ``libtilelang.dylib`` records ``libz3.dylib`` as a
    bare basename rather than ``@rpath/libz3.dylib`` (or ``@loader_path/...``).
    On macOS this means dyld will not find the libz3 shipped *next to it* in
    the build directory. The result is that ``import tilelang`` raises
    ``OSError: Library not loaded: libz3.dylib`` and every Path C kernel then
    silently returns ``None`` in its dispatch try/except — bench scripts log
    "did not dispatch" with no actionable signal.

    Workaround: explicitly ``dlopen`` the libz3 that is already shipped next to
    the tilelang dev build. Once the lib is in the process image, dyld will
    satisfy the basename-only reference for libtilelang.dylib's later load.
    Idempotent and silent on success / when libz3 is already present.
    """

    if getattr(_preload_libz3_for_dev_tilelang, "_done", False):
        return
    # fix-round-4: bail after a small number of full-sweep failures so we
    # don't keep retrying every candidate (and re-stat'ing every path) on
    # every Path C dispatch when libz3 genuinely isn't present.
    _MAX_FAILED_ATTEMPTS = 3
    failed = getattr(_preload_libz3_for_dev_tilelang, "_failed_attempts", 0)
    if failed >= _MAX_FAILED_ATTEMPTS:
        return

    candidates: list[_Path] = []
    dev_build_root = os.environ.get("TILELANG_DEV_BUILD_ROOT")
    if dev_build_root:
        candidates.append(_Path(dev_build_root) / "lib" / "libz3.dylib")
    # Fallback: every tilelang dev tree we've seen drops libz3.dylib at
    # ``<root>/build/lib`` next to libtilelang.dylib.
    for env_var in ("TILELANG_ROOT",):
        root = os.environ.get(env_var)
        if root:
            candidates.append(_Path(root) / "build" / "lib" / "libz3.dylib")
    # Last-resort: a checkout layout used by recent benches. /tmp is
    # world-writable on Unix, so loading a dylib from there is an arbitrary
    # code execution risk if an attacker plants a malicious libz3.dylib.
    # fix-round-5: gate behind opt-in env var; production never sets this.
    # TOCTOU note (fix-round-5, finding 4): we check exists() then dlopen,
    # leaving a small race window where a candidate path could be swapped
    # between the stat and the load. Acceptable for the dev/bench-only
    # paths that survive the gating above (env-rooted dirs the developer
    # controls, or /opt/homebrew which is root-owned), and the /tmp
    # candidate is now opt-in. Not fixing the race here — would require
    # fd-based loading (dlopen has no by-fd variant on macOS).
    if os.environ.get("CPPMEGA_ALLOW_UNSAFE_LIBZ3") == "1":
        import warnings as _warnings

        _warnings.warn(
            "CPPMEGA_ALLOW_UNSAFE_LIBZ3=1 set; including world-writable "
            "/tmp/tl_apache_tvm_swap/build/lib in libz3 preload candidates. "
            "This is unsafe outside dev environments — an attacker who can "
            "write to /tmp could plant a malicious libz3.dylib that gets "
            "loaded into the process.",
            stacklevel=2,
        )
        candidates.append(_Path("/tmp/tl_apache_tvm_swap/build/lib/libz3.dylib"))
    # Brew-installed z3 (works as a basename-resolution fallback).
    candidates.append(_Path("/opt/homebrew/lib/libz3.dylib"))

    for candidate in candidates:
        try:
            if candidate.exists():
                ctypes.CDLL(str(candidate), ctypes.RTLD_GLOBAL)
                # Set _done only AFTER the dlopen actually succeeds.
                _preload_libz3_for_dev_tilelang._done = True  # type: ignore[attr-defined]
                return
        except OSError:
            continue
    # No candidate succeeded — bump the failed-attempts counter so we'll
    # stop retrying once we've tried enough times. _done stays unset so that
    # if the env later changes (e.g., user sets TILELANG_DEV_BUILD_ROOT) we
    # can still pick up a real lib up to the attempt cap.
    _preload_libz3_for_dev_tilelang._failed_attempts = failed + 1  # type: ignore[attr-defined]


# Run the preload eagerly on Darwin so any ``import tilelang`` triggered by
# downstream Path C modules during their own import inherits a process image
# that already has libz3 loaded. No-op on non-Darwin platforms.
if sys.platform == "darwin":
    _preload_libz3_for_dev_tilelang()


MetalKernel = Callable[..., list[mx.array]]


class MSLDispatchUnsupported(RuntimeError):
    """Raised when the runtime cannot dispatch a vendor MSL kernel."""


@dataclass(frozen=True)
class MSLDispatchStatus:
    available: bool
    reason: str


_SUPPORTED_DTYPES = {mx.float32, mx.float16, mx.bfloat16}


def can_run_metal() -> bool:
    metal = getattr(mx, "metal", None)
    return mx.default_device() == mx.gpu and metal is not None and metal.is_available()


def _metal_kernel_constructor() -> Callable[..., MetalKernel] | None:
    fast = getattr(mx, "fast", None)
    metal_kernel = getattr(fast, "metal_kernel", None)
    if metal_kernel is None:
        return None
    return cast(Callable[..., MetalKernel], metal_kernel)


def msl_dispatch_status(*arrays: mx.array) -> MSLDispatchStatus:
    if not can_run_metal():
        return MSLDispatchStatus(False, "MLX Metal backend is not available on the default GPU device")
    if _metal_kernel_constructor() is None:
        return MSLDispatchStatus(False, "MLX mx.fast.metal_kernel API is not available")
    for x in arrays:
        if x.dtype not in _SUPPORTED_DTYPES:
            return MSLDispatchStatus(False, f"unsupported dtype for vendor MSL kernel: {x.dtype}")
        if x.size == 0:
            return MSLDispatchStatus(False, "empty tensors must use the pure MLX fallback")
    return MSLDispatchStatus(True, "MSL dispatch path is available")


def make_metal_kernel(
    *,
    name: str,
    input_names: Sequence[str],
    output_names: Sequence[str],
    source: str,
    header: str = "",
    ensure_row_contiguous: bool = True,
) -> MetalKernel | None:
    """Create a cached mx.fast.metal_kernel handle, or None if unavailable."""

    if not can_run_metal():
        return None
    constructor = _metal_kernel_constructor()
    if constructor is None:
        return None
    return cast(
        MetalKernel,
        constructor(
            name=name,
            input_names=list(input_names),
            output_names=list(output_names),
            source=source,
            header=header,
            ensure_row_contiguous=ensure_row_contiguous,
        ),
    )


def dispatch(
    kernel: MetalKernel,
    *,
    inputs: Sequence[mx.array],
    output_shapes: Sequence[Sequence[int]],
    output_dtypes: Sequence[mx.Dtype],
    grid: tuple[int, int, int] | None = None,
    threadgroup: tuple[int, int, int] | None = None,
    lowering: TileLangMSLLowering | None = None,
    template: Sequence[tuple[str, object]] | None = None,
) -> list[mx.array]:
    if any(isinstance(x, mx.array) and x.size == 0 for x in inputs):
        raise MSLDispatchUnsupported("empty tensors must use the pure MLX fallback")
    if lowering is not None:
        launch_grid = metal_grid_for_lowering(lowering)
        launch_threadgroup = lowering.threadgroup
        if grid is not None and grid != launch_grid:
            raise ValueError(f"conflicting dispatch grid: got {grid}, expected {launch_grid}")
        if threadgroup is not None and threadgroup != launch_threadgroup:
            raise ValueError(
                "conflicting dispatch threadgroup: "
                f"got {threadgroup}, expected {launch_threadgroup}"
            )
    elif grid is None or threadgroup is None:
        raise ValueError("dispatch requires either lowering=... or both grid=... and threadgroup=...")
    else:
        launch_grid = grid
        launch_threadgroup = threadgroup
    return kernel(
        inputs=list(inputs),
        template=list(template) if template else None,
        grid=launch_grid,
        threadgroup=launch_threadgroup,
        output_shapes=[tuple(shape) for shape in output_shapes],
        output_dtypes=list(output_dtypes),
        stream=mx.gpu,
    )


# ---------------------------------------------------------------------------
# Optional TileLang -> MLX inline lowering helper
# ---------------------------------------------------------------------------


_KERNEL_DEF_RE = re.compile(r"kernel\s+void\s+(?P<name>\w+)\s*\(", re.MULTILINE)
_TILELANG_BUILTIN_CAST_TYPE_RE = (
    r"(?:"
    r"bool|u?char|u?short|u?int|u?long|"
    r"unsigned\s+(?:char|short|int|long)|"
    r"long\s+long|unsigned\s+long\s+long|"
    r"size_t|half|float"
    r")"
)
_TILELANG_BUILTIN_ALIAS_CAST_REWRITES: tuple[re.Pattern[str], ...] = (
    re.compile(
        rf"\(\(\s*(?P<ctype>{_TILELANG_BUILTIN_CAST_TYPE_RE})\s*\)\s*"
        r"(?P<alias>threadIdx|blockIdx)\.(?P<axis>[xyz])\s*\)"
    ),
    re.compile(
        rf"\(\s*(?P<ctype>{_TILELANG_BUILTIN_CAST_TYPE_RE})\s*\)\s*"
        r"(?P<alias>threadIdx|blockIdx)\.(?P<axis>[xyz])\b"
    ),
    re.compile(
        rf"\bstatic_cast\s*<\s*(?P<ctype>{_TILELANG_BUILTIN_CAST_TYPE_RE})\s*>\s*"
        r"\(\s*(?P<alias>threadIdx|blockIdx)\.(?P<axis>[xyz])\s*\)"
    ),
    re.compile(
        rf"\b(?P<ctype>{_TILELANG_BUILTIN_CAST_TYPE_RE})\s*"
        r"\(\s*(?P<alias>threadIdx|blockIdx)\.(?P<axis>[xyz])\s*\)"
    ),
)
_TILELANG_BUILTIN_ALIAS_AXIS_RE = re.compile(
    r"\b(?P<alias>threadIdx|blockIdx)\.(?P<axis>[xyz])\b"
)
_TILELANG_BUILTIN_ALIAS_DECLS: dict[str, re.Pattern[str]] = {
    "blockIdx": re.compile(
        r"(?m)^[ \t]*uint3\s+blockIdx\s*=\s*threadgroup_position_in_grid\s*;\s*\n?"
    ),
    "threadIdx": re.compile(
        r"(?m)^[ \t]*uint3\s+threadIdx\s*=\s*thread_position_in_threadgroup\s*;\s*\n?"
    ),
}
_MSL_COMMENT_OR_STRING_RE = re.compile(
    r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'",
    re.DOTALL,
)


def _mask_msl_comments_and_strings(msl: str) -> str:
    """Preserve source offsets while hiding comment/string delimiters."""

    return _MSL_COMMENT_OR_STRING_RE.sub(lambda match: " " * len(match.group(0)), msl)


def _loaded_libtvm_ffi_images() -> list[str]:
    """Return loaded libtvm_ffi images, if the platform exposes dyld state."""

    if not sys.platform.startswith("darwin"):
        return []
    try:
        dyld = ctypes.CDLL(None)
        image_count = dyld._dyld_image_count
        image_count.restype = ctypes.c_uint32
        image_name = dyld._dyld_get_image_name
        image_name.argtypes = [ctypes.c_uint32]
        image_name.restype = ctypes.c_char_p
    except Exception:
        return []

    images: list[str] = []
    seen: set[str] = set()
    for idx in range(int(image_count())):
        raw = image_name(idx)
        if raw is None:
            continue
        path = os.path.realpath(raw.decode("utf-8", errors="replace"))
        if os.path.basename(path) != "libtvm_ffi.dylib" or path in seen:
            continue
        seen.add(path)
        images.append(path)
    return images


def _ensure_single_libtvm_ffi_image() -> None:
    images = _loaded_libtvm_ffi_images()
    if len(images) <= 1:
        return
    raise MSLDispatchUnsupported(
        "unsafe TileLang/TVM-FFI runtime: multiple libtvm_ffi.dylib images are loaded "
        f"({', '.join(images)}); rebuild/install tilelang and apache-tvm-ffi against the same "
        "TVM-FFI tree before using Path C lowering"
    )


@dataclass(frozen=True)
class TileLangMSLLowering:
    """Result of lowering a TileLang PrimFunc to an MLX-callable Metal kernel.

    Threadgroup allocations are kept inside the body, so the body is inlined
    directly into the MLX ``source=`` argument (which is itself inside a
    ``kernel void``). The TileLang prelude (typedefs, includes) goes to the
    ``header=`` argument.
    """

    header: str
    body: str
    grid: tuple[int, int, int]
    threadgroup: tuple[int, int, int]
    msl_text: str
    buffer_param_names: list[str]
    kernel_name: str


def _split_kernel_msl(msl: str) -> tuple[str, str, str]:
    """Split TileLang-emitted MSL into (prelude, signature_text, body_text).

    ``signature_text`` excludes the surrounding parentheses; ``body_text``
    keeps its outer braces.
    """

    masked = _mask_msl_comments_and_strings(msl)
    match = _KERNEL_DEF_RE.search(masked)
    if match is None:
        raise RuntimeError("TileLang MSL: missing 'kernel void' declaration.")
    prelude = msl[: match.start()].rstrip()

    sig_start = match.end()
    depth = 1
    i = sig_start
    while i < len(msl) and depth > 0:
        ch = masked[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        i += 1
    if depth != 0:
        raise RuntimeError("TileLang MSL: unbalanced parens in signature.")
    sig_text = msl[sig_start : i - 1]

    j = i
    while j < len(msl) and msl[j].isspace():
        j += 1
    if j >= len(msl) or msl[j] != "{":
        raise RuntimeError("TileLang MSL: expected '{' after signature.")
    body_start = j
    depth = 1
    j += 1
    while j < len(msl) and depth > 0:
        ch = masked[j]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        j += 1
    if depth != 0:
        raise RuntimeError("TileLang MSL: unbalanced braces in body.")
    body_text = msl[body_start:j]
    return prelude, sig_text, body_text


def _parse_buffer_param_names(sig_text: str) -> list[str]:
    """Return the buffer parameter names from a kernel signature, in order."""

    decls: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in sig_text:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            decls.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        last = "".join(current).strip()
        if last:
            decls.append(last)

    names: list[str] = []
    for decl in decls:
        clean = re.sub(r"\[\[.*?\]\]", "", decl).strip()
        m = re.search(r"(\w+)\s*$", clean)
        if not m:
            continue
        var = m.group(1)
        if var in ("blockIdx", "threadIdx"):
            continue
        # Buffer parameters always have a "device" qualifier (or "const device").
        if "device " in clean:
            names.append(var)
    return names


def _metal_builtin_for_tilelang_alias(alias: str, axis: str) -> str:
    if alias == "threadIdx":
        return f"thread_position_in_threadgroup.{axis}"
    if alias == "blockIdx":
        return f"threadgroup_position_in_grid.{axis}"
    raise ValueError(f"unexpected TileLang builtin alias: {alias}")


def _rewrite_tilelang_builtin_axis_cast(match: re.Match[str]) -> str:
    ctype = " ".join(match.group("ctype").split())
    builtin = _metal_builtin_for_tilelang_alias(match.group("alias"), match.group("axis"))
    return f"(({ctype}){builtin})"


def _rewrite_tilelang_builtin_axis(match: re.Match[str]) -> str:
    return _metal_builtin_for_tilelang_alias(match.group("alias"), match.group("axis"))


def _strip_msl_comments_and_strings(msl: str) -> str:
    return _MSL_COMMENT_OR_STRING_RE.sub("", msl)


def _rewrite_msl_code_segments(
    msl: str,
    rewrite: Callable[[str], str],
) -> str:
    """Apply a source rewrite only to MSL code, preserving comments/strings."""

    chunks: list[str] = []
    start = 0
    for match in _MSL_COMMENT_OR_STRING_RE.finditer(msl):
        chunks.append(rewrite(msl[start : match.start()]))
        chunks.append(match.group(0))
        start = match.end()
    chunks.append(rewrite(msl[start:]))
    return "".join(chunks)


def _drop_alias_decl_if_unused(body: str, var_name: str) -> str:
    decl_re = _TILELANG_BUILTIN_ALIAS_DECLS[var_name]
    without_decl = decl_re.sub("", body)
    if re.search(rf"\b{var_name}\b", _strip_msl_comments_and_strings(without_decl)):
        return body
    return without_decl


def _canonicalize_tilelang_builtin_aliases(body: str) -> str:
    """Remove TileLang block/thread vector aliases for scalar axis accesses."""

    def rewrite(code: str) -> str:
        for pattern in _TILELANG_BUILTIN_ALIAS_CAST_REWRITES:
            code = pattern.sub(_rewrite_tilelang_builtin_axis_cast, code)
        return _TILELANG_BUILTIN_ALIAS_AXIS_RE.sub(_rewrite_tilelang_builtin_axis, code)

    body = _rewrite_msl_code_segments(body, rewrite)
    body = _drop_alias_decl_if_unused(body, "blockIdx")
    body = _drop_alias_decl_if_unused(body, "threadIdx")
    return body


def _inline_tilelang_kernel_body(inner: str) -> str:
    body = (
        "    uint3 blockIdx = threadgroup_position_in_grid;\n"
        "    uint3 threadIdx = thread_position_in_threadgroup;\n"
        + inner
    )
    return _canonicalize_tilelang_builtin_aliases(body)


def metal_grid_for_lowering(
    lowering: TileLangMSLLowering,
) -> tuple[int, int, int]:
    """Return the MLX dispatch grid for a TileLang-lowered kernel.

    TileLang's ``blockIdx`` extents describe threadgroups, while
    ``mx.fast.metal_kernel`` expects the total thread grid. Multiplying here
    keeps callers from accidentally launching one thread per TileLang block.
    """

    return (
        max(1, lowering.grid[0] * lowering.threadgroup[0]),
        max(1, lowering.grid[1] * lowering.threadgroup[1]),
        max(1, lowering.grid[2] * lowering.threadgroup[2]),
    )


def lower_tilelang_to_msl_inline(
    prim_func: Any,
    *,
    target: str = "metal",
    pass_configs: dict[str, Any] | None = None,
) -> TileLangMSLLowering:
    """Lower a TileLang PrimFunc to MSL and prepare an inline body for MLX.

    The TileLang ``kernel void`` body is inlined and MLX-compatible
    ``blockIdx``/``threadIdx`` aliases are removed when all uses can be safely
    rewritten to Metal builtins. Buffer references stay in the order TileLang
    chose (alphabetic), so the caller MUST pass ``input_names + output_names``
    in that same order to ``mx.fast.metal_kernel``.

    Raises ``MSLDispatchUnsupported`` when tilelang or its Metal target is
    unavailable, mirroring the pure-MLX fallback contract.

    ``pass_configs`` (cppmega-mlx-cuz Z3 wiring): optional dict of TileLang
    pass-config keys to enable for *this* lowering only. Threaded through a
    ``tvm.transform.PassContext`` around the engine ``lower`` call so a
    kernel can opt in to Z3-roadmap passes without flipping any global flag.
    Conservative-by-default: each pass falls back to its legacy code path
    when its proof obligation is not discharged. Currently shipped keys
    relevant to Path C kernels:
      * ``tl.simd_lift_reductions`` (Z3 idea #9, detection-only today).
      * ``tl.drop_provable_bound_checks`` (Z3 idea #4).
      * ``tl.auto_double_buffer`` (Z3 idea #2, gated stub).
    Idea #10 (fp8 dot4 legality) and idea #11 (intra-warp barrier elision)
    do not currently expose PassConfig keys in the in-tree TileLang fork —
    #10 is enforced inside ``T.fp8_scaled_matmul`` directly and gated by
    the ``TILELANG_DISABLE_FP8_DOT4_AUTO`` env var; #11 has not landed yet.
    Callers should filter their pass-config dict at runtime: not every
    candidate key is registered with the active ``libtilelang`` build.
    """

    try:
        from tilelang import tvm  # type: ignore
        from tilelang.engine.lower import lower as tl_lower  # type: ignore
    except Exception as exc:  # pragma: no cover - guarded by callers
        raise MSLDispatchUnsupported(
            f"tilelang import failed: {exc}; falling back to pure MLX"
        ) from exc

    _ensure_single_libtvm_ffi_image()
    metal_target = _as_metal_target(target)
    if pass_configs:
        # opt_level=3 mirrors tilelang.jit.kernel.JitKernel default, so the
        # only behavioural delta vs. the legacy (no-PassContext) path is the
        # caller-supplied config dict.
        with tvm.transform.PassContext(opt_level=3, config=dict(pass_configs)):
            artifact = tl_lower(prim_func, target=metal_target)
    else:
        artifact = tl_lower(prim_func, target=metal_target)

    grid = [1, 1, 1]
    block = [1, 1, 1]
    for _, func in artifact.device_mod.functions.items():
        thread_extent = func.attrs.get("thread_extent")
        if thread_extent is None:
            continue
        for tag, extent in thread_extent.items():
            tag_str = str(tag)
            if "threadIdx" in tag_str:
                idx = "xyz".index(tag_str[-1])
                block[idx] = int(extent)
            elif "blockIdx" in tag_str:
                idx = "xyz".index(tag_str[-1])
                grid[idx] = int(extent)
        break

    msl_text = str(artifact.kernel_source)
    prelude, sig_text, body_text = _split_kernel_msl(msl_text)
    inner = body_text[1:-1]
    body = _inline_tilelang_kernel_body(inner)
    return TileLangMSLLowering(
        header=prelude,
        body=body,
        grid=(grid[0], grid[1], grid[2]),
        threadgroup=(block[0], block[1], block[2]),
        msl_text=msl_text,
        buffer_param_names=_parse_buffer_param_names(sig_text),
        kernel_name=_KERNEL_DEF_RE.search(msl_text).group("name"),  # type: ignore[union-attr]
    )


# ---------------------------------------------------------------------------
# Path C Metal FP8 intrinsic registration (Fix-1 + Fix-A re-application)
# ---------------------------------------------------------------------------
#
# The Path C FP8 vecmat / sparse-MLA kernels emit calls to a small set of
# Metal-specific intrinsics that TileLang's lowering expects to find as TVM
# ``Op``s. If the in-tree TileLang/TVM forgets to register them we get
# opaque ``AttributeError``s deep inside the lowering pipeline. We register
# them defensively at module import so:
#
#   * The error surface for a missing intrinsic is a clear ``RuntimeError``
#     from ``_assert_path_c_metal_fp8_intrinsics_registered`` instead of an
#     FFI ``AttributeError`` deep in lowering.
#   * Each op carries a meaningful ``TCallEffectKind`` so CSE / hoisting /
#     DCE behave correctly. ``thread_position_in_grid_x`` and
#     ``thread_index_in_simdgroup`` are ``kReadState`` (block CSE / hoist,
#     allow DCE); ``fp8_e4m3_dot4`` is ``kPure`` (deterministic math,
#     CSE-able).
#
# The whole block is wrapped in try/except so the module imports cleanly on
# hosts without a working TVM (e.g. CI without libz3).

# Effect-kind enum values as IntImm("int32", N).
# kPure=0: deterministic, side-effect-free (CSE-able).
# kReadState=1: reads runtime state (e.g. thread position) — blocks CSE/hoisting,
#               permits DCE of unused calls.
_EFFECT_KIND_PURE = 0
_EFFECT_KIND_READ_STATE = 1

# (op_name, num_inputs, description, effect_kind)
_PATH_C_METAL_FP8_INTRINSICS: tuple[tuple[str, int, str, int], ...] = (
    ("tirx.metal.fp8_e4m3_dot4", 4,
     "FP8 e4m3 packed dot4 (deterministic — CSE-able).",
     _EFFECT_KIND_PURE),
    ("tirx.metal.thread_position_in_grid_x", 0,
     "Reads kernel-arg with [[thread_position_in_grid]] — kReadState to block CSE/hoist.",
     _EFFECT_KIND_READ_STATE),
    ("tirx.metal.thread_index_in_simdgroup", 0,
     "Reads kernel-arg with [[thread_index_in_simdgroup]] — kReadState to block CSE/hoist.",
     _EFFECT_KIND_READ_STATE),
)

_effect_kind_imm: dict[int, Any] = {
    _EFFECT_KIND_PURE: None,        # set lazily in _register_path_c_metal_fp8_intrinsics
    _EFFECT_KIND_READ_STATE: None,
}


def _register_path_c_metal_fp8_intrinsics() -> None:
    """Register the Path C Metal FP8 intrinsic ops (idempotent).

    On hosts where TVM is not importable (e.g. CI without libz3) this is a
    no-op so the module imports cleanly. On hosts with TVM, each op in
    ``_PATH_C_METAL_FP8_INTRINSICS`` is registered iff ``Op.get(name)``
    fails (i.e. not already registered). The op then has its ``num_inputs``
    set and a ``TCallEffectKind`` attribute attached.
    """

    try:
        from tilelang import tvm  # type: ignore
        from tilelang.tvm.tir import IntImm  # type: ignore
        from tilelang.tvm.ir import Op  # type: ignore
        register_op = tvm.ir._ffi_api.RegisterOp  # type: ignore[attr-defined]
    except Exception:
        try:
            import tvm  # type: ignore
            from tvm.tir import IntImm  # type: ignore
            from tvm.ir import Op  # type: ignore
            register_op = tvm.ir._ffi_api.RegisterOp  # type: ignore[attr-defined]
        except Exception:
            # No TVM available — skip registration silently. Callers that
            # need the ops will get a clear error from
            # ``_assert_path_c_metal_fp8_intrinsics_registered``.
            return

    # Lazy-bind effect-kind IntImms once we have a working TVM.
    for kind in (_EFFECT_KIND_PURE, _EFFECT_KIND_READ_STATE):
        if _effect_kind_imm[kind] is None:
            _effect_kind_imm[kind] = IntImm("int32", kind)

    for name, num_inputs, description, effect_kind in _PATH_C_METAL_FP8_INTRINSICS:
        try:
            existing = Op.get(name)
        except Exception:
            existing = None
        if existing is not None:
            continue
        try:
            register_op(name, description)
            op = Op.get(name)
            try:
                op.set_num_inputs(num_inputs)
            except Exception:
                pass
            try:
                op.set_attr("TCallEffectKind", _effect_kind_imm[effect_kind])
            except Exception:
                pass
        except Exception:
            # Best-effort: a partial failure here should not block module
            # import. The assertion helper will surface it loudly when a
            # caller actually needs the op.
            continue


def _assert_path_c_metal_fp8_intrinsics_registered() -> None:
    """Raise ``RuntimeError`` if any Path C Metal FP8 intrinsic is missing.

    Call this *before* the first @T.prim_func parse on the packed-dot4
    macro path so users see a clear actionable error instead of a deep
    ``AttributeError`` from the FFI lowering pipeline.
    """

    try:
        from tilelang.tvm.ir import Op  # type: ignore
    except Exception:
        try:
            from tvm.ir import Op  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Path C Metal FP8 intrinsic check requires TVM, but TVM is "
                f"not importable: {exc}. Path C lowering is unavailable on "
                "this host."
            ) from exc

    missing: list[str] = []
    for name, _num_inputs, _desc, _effect_kind in _PATH_C_METAL_FP8_INTRINSICS:
        try:
            Op.get(name)
        except Exception:
            missing.append(name)
    if missing:
        raise RuntimeError(
            "Path C Metal FP8 intrinsics are not registered: "
            + ", ".join(missing)
            + ". The in-tree TileLang/TVM forgot to register these ops; "
            "see _msl_transform._register_path_c_metal_fp8_intrinsics. "
            "Without them, Path C FP8 kernels silently fall back to scalar "
            "decode."
        )


def _as_metal_target(target: Any) -> Any:
    """Coerce a Metal target spec into a form Apache TVM accepts.

    Apache TVM rejects the legacy CLI-form ``"metal -thread_warp_size=32"``
    after PR #2143 (it now requires the dict form). Older callers in this
    tree still pass the CLI form; this helper translates between them.

    Returns a ``tvm.target.Target`` if TVM is importable; otherwise returns
    the input unchanged so non-TVM hosts still see a clear downstream
    error rather than an opaque import failure here.
    """

    try:
        from tilelang import tvm  # type: ignore
    except Exception:
        try:
            import tvm  # type: ignore
        except Exception:
            return target

    if not isinstance(target, str):
        # Already a tvm.target.Target (or a dict the constructor accepts).
        try:
            return tvm.target.Target(target)
        except Exception:
            return target

    spec = target.strip()
    if " " not in spec and "-" not in spec:
        # Bare "metal".
        return tvm.target.Target(spec)

    # Parse CLI-form "<kind> -k1=v1 -k2=v2 ..." into dict-form.
    parts = spec.split()
    kind = parts[0]
    config: dict[str, Any] = {"kind": kind}
    for token in parts[1:]:
        token = token.lstrip("-")
        if not token:
            continue
        if "=" in token:
            key, _, value = token.partition("=")
            # Best-effort numeric coercion (TileLang historically uses int
            # values for thread_warp_size / max_num_threads).
            if value.lstrip("-").isdigit():
                config[key] = int(value)
            else:
                config[key] = value
        else:
            config[token] = True

    try:
        return tvm.target.Target(config)
    except Exception:
        # Fall back to the legacy string form. Some TileLang builds still
        # accept it; if not, the caller will see a clear TVM error.
        try:
            return tvm.target.Target(spec)
        except Exception:
            return spec


# Auto-register intrinsics at module import. Best-effort only — failures
# here are tolerated so the module imports cleanly on hosts without TVM.
try:
    _register_path_c_metal_fp8_intrinsics()
except Exception:
    pass


__all__ = [
    "MSLDispatchStatus",
    "MSLDispatchUnsupported",
    "MetalKernel",
    "TileLangMSLLowering",
    "_as_metal_target",
    "_assert_path_c_metal_fp8_intrinsics_registered",
    "_register_path_c_metal_fp8_intrinsics",
    "can_run_metal",
    "dispatch",
    "lower_tilelang_to_msl_inline",
    "make_metal_kernel",
    "metal_grid_for_lowering",
    "msl_dispatch_status",
]

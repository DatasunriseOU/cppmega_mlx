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

import collections
import ctypes
import functools
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path as _Path
from typing import Any, Callable, Sequence, cast

import mlx.core as mx


# fix-round-7 finding-5: test-only candidate injection point. Production code
# leaves this empty; tests that need to load a libz3 from a non-default path
# (e.g., the in-tree dev tree at ``/tmp/tl_apache_tvm_swap``) populate this
# list in ``conftest.py``. Replaces the prior approach of hard-coding the
# /tmp candidate behind ``CPPMEGA_ALLOW_UNSAFE_LIBZ3=1`` — production never
# touches /tmp now, removing the world-writable-dylib attack surface entirely.
_LIBZ3_DEV_CANDIDATES: list[_Path] = []


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
    # fix-round-7 finding-6: defensive Darwin-only guard inside the function
    # itself. The module-level call site below already gates on
    # ``sys.platform == "darwin"``, but a future caller (e.g., a unit test
    # invoking the preload directly to reset its state) could otherwise
    # re-enter this on Linux/Windows where the dlopen-by-basename problem
    # this routine works around does not apply.
    if sys.platform != "darwin":
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
    # fix-round-7 finding-5: production code no longer hard-codes any
    # world-writable path (previously /tmp/tl_apache_tvm_swap was added
    # behind ``CPPMEGA_ALLOW_UNSAFE_LIBZ3=1``, but conftest.py forced that
    # env var ON for tests, which inverted the security boundary —
    # production processes that happened to inherit the env from a parent
    # would silently load a /tmp dylib). Tests that need a non-default
    # candidate now inject it explicitly via ``_LIBZ3_DEV_CANDIDATES`` (see
    # tests/conftest.py); production keeps an empty list.
    # TOCTOU note: exists() → dlopen has a small race; non-mitigatable on
    # macOS without fd-based dlopen. The remaining candidates are env-rooted
    # (dev controls them) or /opt/homebrew (root-owned), so the surface is
    # bounded. The previous /tmp path is removed entirely.
    candidates.extend(_LIBZ3_DEV_CANDIDATES)
    # Brew-installed z3 (works as a basename-resolution fallback).
    candidates.append(_Path("/opt/homebrew/lib/libz3.dylib"))

    for candidate in candidates:
        # fix-round-8 (Wave 3 grok-4): drop the exists()-then-CDLL precheck. It
        # introduced a TOCTOU window where the file could disappear/change
        # between the stat() and the dlopen(). Calling CDLL directly and
        # discriminating on the exception type yields the same outcome with
        # no race. We still preserve the round-5 distinction between "missing
        # file" (silent skip — not actionable) and "broken dylib" (warn —
        # surfaces wrong-arch/corrupt-dylib/missing-dep so it doesn't hide
        # behind the silent retry).
        try:
            ctypes.CDLL(str(candidate), ctypes.RTLD_GLOBAL)
        except FileNotFoundError:
            # File missing at dlopen time. Not actionable; try next candidate.
            continue
        except OSError as e:
            # On macOS dyld surfaces "image not found" as OSError (errno=2 in
            # the message but not always settable on the exception). Detect
            # that case heuristically and treat it as "missing" (silent skip);
            # everything else is a *broken* libz3 and gets logged.
            msg = str(e)
            if "image not found" in msg or "no such file" in msg.lower():
                continue
            import logging as _logging

            _logging.getLogger(__name__).warning(
                "libz3 preload at %s failed: %s", candidate, e
            )
            continue
        # Set _done only AFTER the dlopen actually succeeds.
        _preload_libz3_for_dev_tilelang._done = True  # type: ignore[attr-defined]
        return
    # No candidate succeeded — bump the failed-attempts counter so we'll
    # stop retrying once we've tried enough times. _done stays unset so that
    # if the env later changes (e.g., user sets TILELANG_DEV_BUILD_ROOT) we
    # can still pick up a real lib up to the attempt cap.
    _preload_libz3_for_dev_tilelang._failed_attempts = failed + 1  # type: ignore[attr-defined]


# fix-round-8 (Wave 3 grok-4): the preload is now *lazy*. Previously this
# fired at module-import time, which ran *before* tests/conftest.py could
# mutate ``_LIBZ3_DEV_CANDIDATES`` to inject extra dev candidates — the
# conftest then had to manually clear the ``_done`` / ``_failed_attempts``
# flags and re-invoke. With a lazy guard, the first real entry-point call
# (``lower_tilelang_to_msl_inline`` or the public
# ``ensure_libz3_preloaded`` below) sees the post-conftest candidate list.
# Bench scripts that want eager preload should call
# ``ensure_libz3_preloaded()`` explicitly at the top of ``main``.
_LIBZ3_PRELOAD_ATTEMPTED = False


def _maybe_preload_libz3() -> None:
    """Lazy entry point: run the preload at most once per process."""

    global _LIBZ3_PRELOAD_ATTEMPTED
    if _LIBZ3_PRELOAD_ATTEMPTED:
        return
    _LIBZ3_PRELOAD_ATTEMPTED = True
    if sys.platform != "darwin":
        return
    _preload_libz3_for_dev_tilelang()


def ensure_libz3_preloaded() -> None:
    """Public entry point for bench scripts that want eager libz3 preload.

    Idempotent: calling repeatedly is cheap (the underlying preload uses its
    own ``_done`` / ``_failed_attempts`` guards). Useful for short-lived
    scripts that import tilelang directly without going through the lazy
    ``lower_tilelang_to_msl_inline`` path.
    """

    _maybe_preload_libz3()


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
    kernel: MetalKernel | None,
    *,
    inputs: Sequence[mx.array],
    output_shapes: Sequence[Sequence[int]],
    output_dtypes: Sequence[mx.Dtype],
    grid: tuple[int, int, int] | None = None,
    threadgroup: tuple[int, int, int] | None = None,
    lowering: TileLangMSLLowering | None = None,
    template: Sequence[tuple[str, object]] | None = None,
) -> list[mx.array]:
    # fix-round-9 (Wave 4 grok finding #2): explicit None-kernel guard. If
    # ``make_metal_kernel`` returned None (because ``can_run_metal()`` is
    # False or ``mx.fast.metal_kernel`` is missing), calling kernel(...) below
    # would raise ``TypeError: 'NoneType' is not callable`` and confuse the
    # caller, who is expecting either a clean ``MSLDispatchUnsupported`` or a
    # successful tensor list. Funnel the failure through the same exception
    # type as every other dispatch failure so callers can continue to use a
    # single ``except MSLDispatchUnsupported`` to fall back to Path B.
    if kernel is None:
        raise MSLDispatchUnsupported(
            "metal_kernel is None — check can_run_metal() and "
            "mx.fast.metal_kernel availability"
        )
    if any(isinstance(x, mx.array) and x.size == 0 for x in inputs):
        raise MSLDispatchUnsupported("empty tensors must use the pure MLX fallback")
    if lowering is not None:
        # fix-round-8 (Wave 3 grok-4): validate caller-supplied input count
        # against the parsed buffer names from the lowered kernel signature.
        # ``buffer_param_names`` enumerates *all* device-qualified buffer
        # parameters (inputs followed by outputs) in TileLang's emission
        # order; the input count is therefore total - len(output_dtypes).
        # Without this check, a caller that gets the order wrong silently
        # passes a wrong tensor as the kernel's i-th input and we get
        # garbage numerics with no diagnostic.
        parsed = lowering.buffer_param_names
        expected_inputs = len(parsed) - len(output_dtypes)
        if expected_inputs < 0 or len(inputs) != expected_inputs:
            raise MSLDispatchUnsupported(
                f"dispatch input count mismatch: got {len(inputs)}, parsed "
                f"{expected_inputs} input buffer names from lowering "
                f"(params={parsed}, output_dtypes={list(output_dtypes)})"
            )
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


# Documentation/test corpus for ``_parse_buffer_param_names``. Each entry is a
# *single* parameter declaration as TileLang's Metal codegen emits it (or as
# we've observed in the wild via the Apache TVM Metal target). The parser is
# expected to:
#   - return the param's identifier when it is a top-level address-space
#     buffer (``device`` or ``constant``);
#   - skip ``threadgroup`` (local), Metal builtin params (``blockIdx``,
#     ``threadIdx``, ``simd_lane``-style with ``[[thread_index_in_simdgroup]]``
#     etc.), and any ``[[...]]`` attribute markers.
#
# fix-round-9 (Wave 4 grok): grow the corpus from a handful of hand-written
# cases to the full set of qualifier orderings TileLang emits post-PR #799,
# plus a couple of TVM-Metal-direct variants we've seen in older lowering
# output (mamba3 fwd_kernel uses ``device float* A``; FP8 paths use
# ``device const float8_e4m3 *A``; matmul backwards-compat uses
# ``const device half *B``). The previous parser tripped on the
# ``device const`` ordering because the trailing-identifier regex consumed
# the buffer-name token correctly but the ``"device "`` substring check was
# fine; the real failure mode was *attribute strip greediness* with nested
# ``[[...]]`` and missing handling of ``constant`` (Metal address space for
# constant buffers, not const-qualified device pointers).
_TEST_PARSE_SIGNATURES: tuple[tuple[str, str | None], ...] = (
    ("device const float8_e4m3 *A", "A"),
    ("const device half *B [[ buffer(0) ]]", "B"),
    ("device float* C [[ buffer(2) ]]", "C"),
    ("device const half* __restrict D [[buffer(3)]]", "D"),
    ("const device float &E [[buffer(4)]]", "E"),
    ("constant float* F [[buffer(5)]]", "F"),  # Metal "constant" address space
    ("threadgroup uchar* shared_buf", None),  # threadgroup = local, skip
    ("uint3 blockIdx [[threadgroup_position_in_grid]]", None),
    ("uint3 threadIdx [[thread_position_in_threadgroup]]", None),
    ("uint simd_lane [[thread_index_in_simdgroup]]", None),
    ("uint3 gridDim [[threadgroups_per_grid]]", None),
    ("uint3 blockDim [[threads_per_threadgroup]]", None),
)


# fix-round-9 (Wave 4 grok): identifiers we never want to treat as buffers,
# even if a future TileLang/TVM change adds a "device" qualifier in front of
# them. These are Metal builtins that callers don't pass; they're filled in
# by the runtime from the [[...]] attribute.
_METAL_BUILTIN_PARAM_NAMES = frozenset({
    "blockIdx",
    "threadIdx",
    "gridDim",
    "blockDim",
})


def _split_signature_decls(sig_text: str) -> list[str]:
    """Split a signature parameter list on top-level commas.

    Respects nested ``(...)`` and ``[[...]]`` Metal attribute markers so that
    a comma inside an attribute (e.g., ``[[buffer(0), function_constant(1)]]``)
    does not split a parameter declaration.
    """

    decls: list[str] = []
    depth_paren = 0
    depth_attr = 0  # tracks ``[[`` / ``]]`` nesting
    i = 0
    current: list[str] = []
    n = len(sig_text)
    while i < n:
        ch = sig_text[i]
        # Detect ``[[`` / ``]]`` as paired tokens (Metal attribute markers).
        if ch == "[" and i + 1 < n and sig_text[i + 1] == "[":
            depth_attr += 1
            current.append("[[")
            i += 2
            continue
        if ch == "]" and i + 1 < n and sig_text[i + 1] == "]" and depth_attr > 0:
            depth_attr -= 1
            current.append("]]")
            i += 2
            continue
        if ch == "(":
            depth_paren += 1
            current.append(ch)
        elif ch == ")":
            depth_paren -= 1
            current.append(ch)
        elif ch == "," and depth_paren == 0 and depth_attr == 0:
            decls.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
        i += 1
    last = "".join(current).strip()
    if last:
        decls.append(last)
    return decls


def _strip_attribute_markers(decl: str) -> str:
    """Remove all ``[[...]]`` Metal attribute markers from a decl string.

    Iterative balanced strip — the previous non-greedy regex
    ``re.sub(r"\\[\\[.*?\\]\\]", "", decl)`` only matched the *first* ``]]`` and
    handled nested attribute markers incorrectly. This walks the string and
    removes balanced ``[[ ... ]]`` segments at any depth.
    """

    out: list[str] = []
    i = 0
    n = len(decl)
    while i < n:
        if decl[i] == "[" and i + 1 < n and decl[i + 1] == "[":
            # Find the matching ``]]`` at the same nesting depth.
            depth = 1
            j = i + 2
            while j < n and depth > 0:
                if decl[j] == "[" and j + 1 < n and decl[j + 1] == "[":
                    depth += 1
                    j += 2
                    continue
                if decl[j] == "]" and j + 1 < n and decl[j + 1] == "]":
                    depth -= 1
                    j += 2
                    continue
                j += 1
            if depth == 0:
                i = j  # skip the entire ``[[...]]``
                continue
            # Unbalanced — fall through and keep the char so the caller can see
            # there's a parse problem instead of silently truncating.
        out.append(decl[i])
        i += 1
    return "".join(out)


def _extract_param_identifier(clean: str) -> str | None:
    """Pull the parameter's identifier out of a stripped decl string.

    Strategy: drop pointer/reference sigils (``*``, ``&``) and any trailing
    array-extent ``[N]``, then take the last word. This tolerates ``__restrict``
    and other type modifiers that appear *before* the identifier — those are
    earlier-word tokens and are simply ignored.
    """

    # Drop trailing ``[N]`` array extent if present (Metal allows it for
    # ``constant`` buffers).
    cleaned = re.sub(r"\[[^\]]*\]\s*$", "", clean).strip()
    # Drop pointer/reference sigils that appear adjacent to the name.
    cleaned = cleaned.replace("*", " ").replace("&", " ").strip()
    m = re.search(r"\b([A-Za-z_]\w*)\s*$", cleaned)
    return m.group(1) if m else None


def _parse_buffer_param_names(sig_text: str) -> list[str]:
    """Return the buffer parameter names from a kernel signature, in order.

    fix-round-9 (Wave 4 grok finding #1): hardened against TileLang/Metal
    signature variants. The previous version used a single non-greedy
    ``[[.*?]]`` strip and a lone ``"device "`` substring check, which broke on:

    * ``device const T*`` ordering (greedy-strip + position-sensitive check),
    * the ``constant`` Metal address space (a legitimate top-level buffer),
    * trailing array extents and ``__restrict`` modifiers,
    * Metal builtin params (``gridDim``, ``blockDim``) that lack a ``device``
      qualifier but were never explicitly excluded.

    See ``_TEST_PARSE_SIGNATURES`` above for the documented input corpus.

    Note: we intentionally do not try to introspect ``prim_func.buffer_map``
    here. TileLang's lowering pipeline transforms PrimFunc params (it adds
    grid/thread axis buffers, drops some, renames others) before emitting the
    final MSL signature, so the *only* authoritative source for "what does
    ``mx.fast.metal_kernel`` need fed in, in what order" is the emitted
    signature itself. ``buffer_map`` would be wrong.
    """

    decls = _split_signature_decls(sig_text)

    names: list[str] = []
    for decl in decls:
        clean = _strip_attribute_markers(decl).strip()
        if not clean:
            continue
        # Skip threadgroup-local buffers — those are allocated inside the
        # kernel, not passed in by the caller.
        if re.search(r"\bthreadgroup\b", clean):
            continue
        # Top-level buffers carry one of two Metal address-space qualifiers.
        is_device = re.search(r"\bdevice\b", clean) is not None
        is_constant = re.search(r"\bconstant\b", clean) is not None
        if not (is_device or is_constant):
            continue
        var = _extract_param_identifier(clean)
        if var is None:
            continue
        if var in _METAL_BUILTIN_PARAM_NAMES:
            continue
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


# fix-round-8 (Wave 3 grok-4): manual cache for lowering results keyed on
# ``(id(prim_func), frozen(pass_configs), target_str)``. We use ``id()``
# rather than ``hash(prim_func)`` because TVM PrimFunc instances are not
# generally hash-stable across re-imports (and may not be hashable at all
# in older builds). ``id()`` is process-lifetime stable for any cached
# prim_func — sufficient because callers in the cppmega.mlx tree hold a
# module-level reference to their PrimFunc, so the id stays valid for the
# full life of the cache.
#
# fix-round-9 (Wave 4 grok finding #4): bound the cache. The key set scales
# with (#PrimFuncs * #distinct pass_configs * #targets); bench shape sweeps
# can probe dozens of configs and the keepalive list pinned every prim_func
# strongly forever. Switch to an LRU OrderedDict with an env-tunable cap and
# evict the matching keepalive entry alongside the cache entry.
_LOWERING_CACHE_MAX_SIZE = max(
    1, int(os.environ.get("CPPMEGA_LOWERING_CACHE_SIZE", "128"))
)
_LOWERING_CACHE: collections.OrderedDict[
    tuple[int, Any, str], TileLangMSLLowering
] = collections.OrderedDict()
# Hold strong references so cached ids cannot be reused for a different
# PrimFunc. (CPython only reuses an id() after the original object is GC'd;
# keeping a strong reference here pins the PrimFunc and prevents that.)
# Keyed by the same cache key so eviction can drop the matching ref together
# with the cache entry, allowing GC.
_LOWERING_CACHE_KEEPALIVE: collections.OrderedDict[
    tuple[int, Any, str], Any
] = collections.OrderedDict()


def _freeze_for_hash(obj: Any) -> Any:
    """Recursively convert ``obj`` into a hashable, structural representation.

    fix-round-9 (Wave 4 grok finding #4): the previous ``frozenset(items())``
    silently bypassed caching on any unhashable nested value (e.g., a
    dict-of-dicts pass_config), causing repeated full lowering on the hot
    path. This recurses into mappings/sequences/sets and stringifies anything
    that's still unhashable at the leaves so we always get *some* stable key.
    """

    if isinstance(obj, dict):
        return frozenset(
            (_freeze_for_hash(k), _freeze_for_hash(v)) for k, v in obj.items()
        )
    if isinstance(obj, (list, tuple)):
        return tuple(_freeze_for_hash(x) for x in obj)
    if isinstance(obj, (set, frozenset)):
        return frozenset(_freeze_for_hash(x) for x in obj)
    try:
        hash(obj)
        return obj
    except TypeError:
        # Last-resort stringification — not pretty, but stable per-process and
        # avoids the alternative of skipping the cache entirely.
        return repr(obj)


def _lowering_cache_key(
    prim_func: Any,
    target: str,
    pass_configs: dict[str, Any] | None,
) -> tuple[int, Any, str]:
    """Build a cache key for the lowering cache.

    Always succeeds: ``_freeze_for_hash`` falls back to ``repr()`` for any
    leaf value that cannot be hashed, so we never silently skip caching.
    """

    cfg = _freeze_for_hash(pass_configs or {})
    return (id(prim_func), cfg, target)


def clear_lowering_cache() -> None:
    """Public hook for tests/bench to reset the lowering cache + keepalive.

    Drops every cached lowering result and the strong refs that pinned the
    associated PrimFunc objects, allowing GC to reclaim them.
    """

    _LOWERING_CACHE.clear()
    _LOWERING_CACHE_KEEPALIVE.clear()


def _store_lowering_in_cache(
    cache_key: tuple[int, Any, str],
    prim_func: Any,
    result: TileLangMSLLowering,
) -> None:
    """Insert (or refresh) a cache entry under LRU eviction."""

    if cache_key in _LOWERING_CACHE:
        _LOWERING_CACHE.move_to_end(cache_key)
        _LOWERING_CACHE_KEEPALIVE.move_to_end(cache_key)
        _LOWERING_CACHE[cache_key] = result
        _LOWERING_CACHE_KEEPALIVE[cache_key] = prim_func
        return
    _LOWERING_CACHE[cache_key] = result
    _LOWERING_CACHE_KEEPALIVE[cache_key] = prim_func
    while len(_LOWERING_CACHE) > _LOWERING_CACHE_MAX_SIZE:
        evicted_key, _ = _LOWERING_CACHE.popitem(last=False)
        _LOWERING_CACHE_KEEPALIVE.pop(evicted_key, None)


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

    # fix-round-8 (Wave 3 grok-4): cache lookup. Lowering re-runs a full TVM
    # pipeline plus several regex passes over the emitted MSL — Path C bench
    # paths re-enter this for every shape probe, and shipping inference
    # likely calls it on each kernel-factory invocation. The cache key uses
    # id(prim_func) (see _lowering_cache_key for the rationale) plus the
    # frozenset of pass_configs and the target string. If the inputs aren't
    # hashable for any reason, _lowering_cache_key returns None and we fall
    # through to the uncached slow path.
    cache_key = _lowering_cache_key(prim_func, target, pass_configs)
    cached = _LOWERING_CACHE.get(cache_key)
    if cached is not None:
        # fix-round-9 (Wave 4 grok finding #4): refresh LRU position on hit.
        _LOWERING_CACHE.move_to_end(cache_key)
        _LOWERING_CACHE_KEEPALIVE.move_to_end(cache_key)
        return cached

    # fix-round-8 (Wave 3 grok-4): trigger the lazy libz3 preload here (instead
    # of at module-import time). By this point any conftest.py that wanted to
    # inject extra dev candidates via ``_LIBZ3_DEV_CANDIDATES`` has already
    # run; the preload sees the post-conftest list on the very first lower
    # call.
    _maybe_preload_libz3()

    # fix-round-8 (Wave 3 grok-4): narrow the catch. Previously a bare
    # ``except Exception`` swallowed *any* failure in the tilelang import as
    # "tilelang unavailable", including TVM-internal AttributeErrors and
    # PassContext drift. Now only an honest import failure (module not
    # installed / dylib not loadable) maps to ``MSLDispatchUnsupported``;
    # other exceptions propagate so callers see the real chain.
    try:
        from tilelang import tvm  # type: ignore
        from tilelang.engine.lower import lower as tl_lower  # type: ignore
    except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
        raise MSLDispatchUnsupported(
            f"tilelang import failed: {exc}; falling back to pure MLX"
        ) from exc

    _ensure_single_libtvm_ffi_image()
    metal_target = _as_metal_target(target)
    # fix-round-8 (Wave 3 grok-4): wrap the lowering call so any TVM-internal
    # error gets a clear ``MSLDispatchUnsupported("lowering failed: ...")``
    # *with* a ``raise from`` chain. Callers can still inspect ``__cause__``
    # to see the original TVM error, but the public surface stays consistent
    # (every dispatch failure is an ``MSLDispatchUnsupported``).
    try:
        if pass_configs:
            # opt_level=3 mirrors tilelang.jit.kernel.JitKernel default, so the
            # only behavioural delta vs. the legacy (no-PassContext) path is
            # the caller-supplied config dict.
            with tvm.transform.PassContext(opt_level=3, config=dict(pass_configs)):
                artifact = tl_lower(prim_func, target=metal_target)
        else:
            artifact = tl_lower(prim_func, target=metal_target)
    except (ImportError, ModuleNotFoundError, MSLDispatchUnsupported):
        raise
    except Exception as exc:
        raise MSLDispatchUnsupported(
            f"tilelang lowering failed: {exc}"
        ) from exc

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
    result = TileLangMSLLowering(
        header=prelude,
        body=body,
        grid=(grid[0], grid[1], grid[2]),
        threadgroup=(block[0], block[1], block[2]),
        msl_text=msl_text,
        buffer_param_names=_parse_buffer_param_names(sig_text),
        kernel_name=_KERNEL_DEF_RE.search(msl_text).group("name"),  # type: ignore[union-attr]
    )
    # fix-round-9 (Wave 4 grok finding #4): bounded LRU store. Pins a strong
    # ref to ``prim_func`` under the same key so eviction releases both.
    _store_lowering_in_cache(cache_key, prim_func, result)
    return result


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


# fix-round-8 (Wave 3 grok-4): cache resolved Target objects keyed on the
# string spec. Construction parses the spec and instantiates a
# ``tvm.target.Target``, which is non-trivial. ``lower_tilelang_to_msl_inline``
# calls this on every lowering; the active set of distinct target strings is
# tiny (typically just ``"metal"`` and ``"metal -thread_warp_size=32"``), so
# a 4-entry LRU is more than enough. Non-string inputs bypass the cache
# (they may be unhashable dicts) and go straight to the slow path.
@functools.lru_cache(maxsize=4)
def _as_metal_target_cached(target: str) -> Any:
    return _as_metal_target_uncached(target)


def _as_metal_target(target: Any) -> Any:
    """Coerce a Metal target spec into a form Apache TVM accepts.

    Apache TVM rejects the legacy CLI-form ``"metal -thread_warp_size=32"``
    after PR #2143 (it now requires the dict form). Older callers in this
    tree still pass the CLI form; this helper translates between them.

    Returns a ``tvm.target.Target`` if TVM is importable; otherwise returns
    the input unchanged so non-TVM hosts still see a clear downstream
    error rather than an opaque import failure here.
    """

    if isinstance(target, str):
        return _as_metal_target_cached(target)
    return _as_metal_target_uncached(target)


def _as_metal_target_uncached(target: Any) -> Any:
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


# Auto-register intrinsics at module import. fix-round-8 (Wave 3 grok-4):
# narrow the catch from bare ``Exception``. Only import-time errors (TVM
# / tilelang not installed) and missing-attribute errors (TVM internals
# moved) are tolerated as "deferred" with a visible warning. Other
# exception types propagate so they're not silently swallowed at import.
try:
    _register_path_c_metal_fp8_intrinsics()
except (ImportError, ModuleNotFoundError, AttributeError) as _register_exc:
    warnings.warn(
        f"Path C intrinsic registration deferred: {_register_exc}",
        RuntimeWarning,
        stacklevel=2,
    )


__all__ = [
    "MSLDispatchStatus",
    "MSLDispatchUnsupported",
    "MetalKernel",
    "TileLangMSLLowering",
    "_as_metal_target",
    "_assert_path_c_metal_fp8_intrinsics_registered",
    "_register_path_c_metal_fp8_intrinsics",
    "can_run_metal",
    "clear_lowering_cache",
    "dispatch",
    "ensure_libz3_preloaded",
    "lower_tilelang_to_msl_inline",
    "make_metal_kernel",
    "metal_grid_for_lowering",
    "msl_dispatch_status",
]

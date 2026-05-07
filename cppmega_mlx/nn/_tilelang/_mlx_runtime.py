# pyright: reportInvalidTypeForm=false, reportMissingImports=false
"""TileLang Metal kernel -> ``mx.fast.metal_kernel`` runtime adapter.

The Triton -> TileLang -> Metal -> MLX numeric harness emits a complete
Metal Shading Language (MSL) function with positional buffer parameters
named ``A``, ``B``, ``C``, ... -- one per ``T.Tensor`` argument of the
TileLang ``@T.prim_func`` -- followed by Metal builtin attributes
(``thread_position_in_grid`` etc.). ``mx.fast.metal_kernel`` builds the
kernel signature itself from caller-supplied ``input_names`` /
``output_names`` and only takes the *body* of the kernel; by convention
those names must be ``inp0``, ``inp1``, ..., ``out0``, ``out1``, ...

This module bridges the two worlds. ``wrap_tilelang_metal_kernel`` takes
a TileLang compile artifact (with ``.kernel_source`` / ``.rt_mod``),
parses the device-qualified parameter names out of the emitted ``kernel
void`` signature, renames the first ``input_count`` to ``inp0..inpN-1``
and the last ``output_count`` to ``out0..outM-1`` (token-level rewrite
on the kernel body), and hands the renamed body to
``mx.fast.metal_kernel``. The resulting callable accepts ``mx.array``
inputs and returns ``mx.array`` outputs.

This is the path-A (numeric harness) sibling of
``cppmega_mlx.nn._tilelang.fp8_vecmat_path_c._fp8_vecmat_kernel_for``,
which performs the same buffer-name dance for the production fp8 vecmat
kernel via ``_msl_transform`` (its body is hand-authored / IR-rewritten
with the right ``inp*`` / ``out*`` names already, so the rename is a
no-op there). For TileLang's stock Metal emitter the names are positional,
so we rename them here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Sequence


__all__ = [
    "MLXRuntimeError",
    "TileLangMetalAdapter",
    "wrap_tilelang_metal_kernel",
]


class MLXRuntimeError(RuntimeError):
    """Raised when the TileLang Metal source cannot be adapted to MLX."""


# ---------------------------------------------------------------------------
# Source parsing
# ---------------------------------------------------------------------------


# ``kernel void <name>(<sig>) { <body> }`` --- the canonical TileLang Metal
# emitter shape. We do not require ``[[ kernel ]]`` annotations because
# TileLang emits the bare ``kernel void`` form (matches ``_msl_transform``).
_KERNEL_DEF_RE = re.compile(r"kernel\s+void\s+(?P<name>\w+)\s*\(", re.MULTILINE)
_COMMENT_OR_STRING_RE = re.compile(
    r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'",
    re.DOTALL,
)


def _mask_comments_and_strings(src: str) -> str:
    """Replace comment/string spans with same-length whitespace.

    Preserves source offsets so brace/paren matching can be done against
    the masked copy without disturbing positions in the real source.
    """

    return _COMMENT_OR_STRING_RE.sub(lambda m: " " * len(m.group(0)), src)


def _split_kernel(src: str) -> tuple[str, str, str, str]:
    """Split ``src`` into ``(prelude, kernel_name, signature, body)``.

    ``signature`` excludes the surrounding ``(`` / ``)`` and ``body`` excludes
    the outer ``{`` / ``}``. Raises :class:`MLXRuntimeError` when the source
    does not match the expected ``kernel void name(...) { ... }`` shape.
    """

    masked = _mask_comments_and_strings(src)
    match = _KERNEL_DEF_RE.search(masked)
    if match is None:
        raise MLXRuntimeError(
            "unsupported TileLang Metal pattern: no 'kernel void' declaration found"
        )
    # Reject multi-kernel sources: emit one kernel per artifact, please.
    second = _KERNEL_DEF_RE.search(masked, match.end())
    if second is not None:
        raise MLXRuntimeError(
            "unsupported TileLang Metal pattern: multiple 'kernel void' "
            "declarations in one source"
        )

    kernel_name = match.group("name")
    prelude = src[: match.start()].rstrip()

    sig_start = match.end()
    depth = 1
    i = sig_start
    while i < len(masked) and depth > 0:
        ch = masked[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        i += 1
    if depth != 0:
        raise MLXRuntimeError(
            "unsupported TileLang Metal pattern: unbalanced parens in signature"
        )
    signature = src[sig_start : i - 1]

    j = i
    while j < len(src) and src[j].isspace():
        j += 1
    if j >= len(src) or src[j] != "{":
        raise MLXRuntimeError(
            "unsupported TileLang Metal pattern: missing body '{' after signature"
        )
    body_start = j + 1
    depth = 1
    j += 1
    while j < len(masked) and depth > 0:
        ch = masked[j]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        j += 1
    if depth != 0:
        raise MLXRuntimeError(
            "unsupported TileLang Metal pattern: unbalanced braces in body"
        )
    body = src[body_start : j - 1]
    return prelude, kernel_name, signature, body


# Per-decl strip of Metal attributes (``[[ buffer(0) ]]`` etc.) and Metal
# qualifiers (``device``, ``constant``, ``threadgroup``, ``__restrict``,
# ``const``). Matches the strategy used in
# ``_msl_transform._parse_buffer_param_names`` but inlined here so this
# module is independent of the legacy Path-C lowering helper.
_ATTR_RE = re.compile(r"\[\[[^\]]*\]\]")


def _strip_attribute_markers(decl: str) -> str:
    return _ATTR_RE.sub(" ", decl)


def _split_signature_decls(sig_text: str) -> list[str]:
    """Split a kernel signature into top-level comma-separated decls."""

    decls: list[str] = []
    depth = 0
    last = 0
    masked = _mask_comments_and_strings(sig_text)
    for i, ch in enumerate(masked):
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth -= 1
        elif ch == "," and depth == 0:
            decls.append(sig_text[last:i])
            last = i + 1
    if last < len(sig_text):
        decls.append(sig_text[last:])
    return [d for d in decls if d.strip()]


_PARAM_NAME_RE = re.compile(r"\b([A-Za-z_]\w*)\s*$")


def _extract_param_identifier(decl: str) -> str | None:
    """Return the parameter identifier from a stripped decl, or None."""

    cleaned = _strip_attribute_markers(decl).strip()
    # Drop trailing array extents.
    cleaned = re.sub(r"\[[^\]]*\]\s*$", "", cleaned).strip()
    cleaned = cleaned.replace("*", " ").replace("&", " ").strip()
    m = _PARAM_NAME_RE.search(cleaned)
    return m.group(1) if m else None


# Metal builtins are pass-through grid/threadgroup descriptors; they are
# NOT user buffers and must not be renamed.
_METAL_BUILTIN_PARAM_NAMES: frozenset[str] = frozenset(
    {
        "thread_position_in_grid",
        "thread_position_in_threadgroup",
        "threadgroup_position_in_grid",
        "thread_index_in_threadgroup",
        "thread_index_in_simdgroup",
        "simdgroup_index_in_threadgroup",
        "threads_per_threadgroup",
        "threadgroups_per_grid",
        "thread_execution_width",
        "grid_size",
        "gridDim",
        "blockDim",
        "blockIdx",
        "threadIdx",
    }
)


def _parse_buffer_param_names(sig_text: str) -> list[str]:
    """Return ``device``/``constant``-qualified buffer names, in order."""

    names: list[str] = []
    for decl in _split_signature_decls(sig_text):
        clean = _strip_attribute_markers(decl).strip()
        if not clean:
            continue
        if re.search(r"\bthreadgroup\b", clean):
            continue
        is_device = re.search(r"\bdevice\b", clean) is not None
        is_constant = re.search(r"\bconstant\b", clean) is not None
        if not (is_device or is_constant):
            continue
        ident = _extract_param_identifier(clean)
        if ident is None:
            continue
        if ident in _METAL_BUILTIN_PARAM_NAMES:
            continue
        names.append(ident)
    return names


# ---------------------------------------------------------------------------
# Body renaming
# ---------------------------------------------------------------------------


def _rename_identifiers_in_code(
    code: str,
    rename: dict[str, str],
) -> str:
    """Rewrite whole-word identifiers in ``code`` per ``rename``.

    Comments and string literals are skipped so we don't accidentally
    touch a parameter name that happens to appear in a doc-comment.
    """

    if not rename:
        return code

    # Match the longest old-name first so a name like ``A`` doesn't shadow
    # ``A_scale`` in the regex alternation.
    keys = sorted(rename.keys(), key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b")

    def _rewrite_segment(seg: str) -> str:
        return pattern.sub(lambda m: rename[m.group(1)], seg)

    chunks: list[str] = []
    last = 0
    for match in _COMMENT_OR_STRING_RE.finditer(code):
        chunks.append(_rewrite_segment(code[last : match.start()]))
        chunks.append(match.group(0))
        last = match.end()
    chunks.append(_rewrite_segment(code[last:]))
    return "".join(chunks)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TileLangMetalAdapter:
    """Adapter container returned by :func:`wrap_tilelang_metal_kernel`.

    ``__call__`` dispatches a single-output or multi-output Metal kernel
    on Mac GPU through ``mx.fast.metal_kernel``.
    """

    kernel_name: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    header: str
    body: str
    buffer_names: tuple[str, ...]
    # The ``mx.fast.metal_kernel(...)`` callable; built lazily so this
    # adapter can be inspected on hosts where MLX is not importable.
    _kernel_factory: Callable[[], Any]

    def build(self) -> Any:
        """Return the underlying ``mx.fast.metal_kernel`` callable."""

        return self._kernel_factory()

    def __call__(
        self,
        inputs: Sequence[Any],
        *,
        output_shapes: Sequence[Sequence[int]],
        output_dtypes: Sequence[Any],
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int] = (1, 1, 1),
    ) -> list[Any]:
        kernel = self.build()
        if len(inputs) != len(self.input_names):
            raise MLXRuntimeError(
                f"input count mismatch: got {len(inputs)}, expected "
                f"{len(self.input_names)} ({list(self.input_names)})"
            )
        if len(output_shapes) != len(self.output_names):
            raise MLXRuntimeError(
                f"output count mismatch: got {len(output_shapes)}, expected "
                f"{len(self.output_names)} ({list(self.output_names)})"
            )
        return kernel(
            inputs=list(inputs),
            output_shapes=[tuple(s) for s in output_shapes],
            output_dtypes=list(output_dtypes),
            grid=tuple(int(g) for g in grid),
            threadgroup=tuple(int(t) for t in threadgroup),
        )


def _extract_kernel_source(artifact: Any) -> str:
    """Pull MSL source out of a TileLang compile artifact, or raise."""

    src = getattr(artifact, "kernel_source", None)
    if isinstance(src, str) and src.strip():
        return src
    rt_mod = getattr(artifact, "rt_mod", None)
    if rt_mod is not None and hasattr(rt_mod, "get_source"):
        try:
            text = rt_mod.get_source()
        except Exception as exc:  # noqa: BLE001 -- broad, surfaced
            raise MLXRuntimeError(
                f"artifact.rt_mod.get_source() raised: {type(exc).__name__}: {exc}"
            ) from exc
        if isinstance(text, str) and text.strip():
            return text
    # Fallback: ``artifact`` may itself already be the source (callers
    # that already ran the codegen and held onto the string).
    if isinstance(artifact, str) and artifact.strip():
        return artifact
    raise MLXRuntimeError(
        "TileLang artifact has neither .kernel_source nor .rt_mod.get_source()"
    )


def wrap_tilelang_metal_kernel(
    artifact: Any,
    *,
    input_count: int,
    output_count: int,
    name: str | None = None,
) -> TileLangMetalAdapter:
    """Adapt a TileLang Metal artifact for ``mx.fast.metal_kernel``.

    ``artifact`` may be a TileLang ``CompiledArtifact``, a raw MSL
    string, or anything exposing ``.kernel_source`` / ``.rt_mod.get_source()``.

    The first ``input_count`` device buffers in the kernel signature are
    renamed to ``inp0..inp{input_count-1}``; the next ``output_count`` to
    ``out0..out{output_count-1}``. TileLang emits buffer parameters in
    PrimFunc declaration order with outputs *last* by convention, so the
    caller is responsible for asserting that ordering matches their kernel.

    Returns a :class:`TileLangMetalAdapter` whose ``__call__`` dispatches
    on Mac GPU. The ``mx.fast.metal_kernel`` instance is built lazily on
    first ``__call__`` so this function can be invoked on hosts without
    MLX (it will only fail when actually launched).
    """

    if input_count < 0 or output_count < 0:
        raise MLXRuntimeError(
            f"input_count/output_count must be non-negative, got "
            f"input_count={input_count}, output_count={output_count}"
        )
    expected_total = input_count + output_count
    if expected_total == 0:
        raise MLXRuntimeError("kernel must have at least one buffer parameter")

    src = _extract_kernel_source(artifact)
    prelude, kernel_name, signature, body = _split_kernel(src)

    buffer_names = _parse_buffer_param_names(signature)
    if len(buffer_names) != expected_total:
        raise MLXRuntimeError(
            f"buffer count mismatch: parsed {len(buffer_names)} device/constant "
            f"buffers from kernel signature ({buffer_names!r}), but caller "
            f"declared input_count={input_count} + output_count={output_count} "
            f"= {expected_total}"
        )

    input_names = tuple(f"inp{i}" for i in range(input_count))
    output_names = tuple(f"out{i}" for i in range(output_count))
    rename: dict[str, str] = {}
    for src_name, mlx_name in zip(buffer_names[:input_count], input_names):
        rename[src_name] = mlx_name
    for src_name, mlx_name in zip(buffer_names[input_count:], output_names):
        rename[src_name] = mlx_name

    # Sanity: each buffer name must be unique. TileLang shouldn't emit
    # duplicates but if it ever did, the rename dict would silently lose
    # the earlier mapping.
    if len(set(buffer_names)) != len(buffer_names):
        raise MLXRuntimeError(
            f"unsupported TileLang Metal pattern: duplicate buffer names "
            f"{buffer_names!r}"
        )

    renamed_body = _rename_identifiers_in_code(body, rename)

    # The header for ``mx.fast.metal_kernel`` is the prelude (typedefs,
    # helper macros, constants) emitted before the kernel definition --
    # NOT the kernel signature, which MLX builds itself from input_names
    # and output_names.
    header = prelude

    final_name = name or kernel_name

    def _build_kernel() -> Any:
        try:
            import mlx.core as mx  # type: ignore
        except Exception as exc:  # noqa: BLE001 -- surfaced verbatim
            raise MLXRuntimeError(
                f"mlx.core import failed: {type(exc).__name__}: {exc}"
            ) from exc
        fast = getattr(mx, "fast", None)
        ctor = getattr(fast, "metal_kernel", None) if fast is not None else None
        if ctor is None:
            raise MLXRuntimeError(
                "mx.fast.metal_kernel constructor unavailable on this MLX build"
            )
        return ctor(
            name=final_name,
            input_names=list(input_names),
            output_names=list(output_names),
            source=renamed_body,
            header=header,
            ensure_row_contiguous=True,
        )

    return TileLangMetalAdapter(
        kernel_name=final_name,
        input_names=input_names,
        output_names=output_names,
        header=header,
        body=renamed_body,
        buffer_names=tuple(buffer_names),
        _kernel_factory=_build_kernel,
    )

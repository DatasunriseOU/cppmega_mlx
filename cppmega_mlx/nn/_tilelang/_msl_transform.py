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
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Sequence, cast

import mlx.core as mx


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
    grid: tuple[int, int, int],
    threadgroup: tuple[int, int, int],
    template: Sequence[tuple[str, object]] | None = None,
) -> list[mx.array]:
    if any(isinstance(x, mx.array) and x.size == 0 for x in inputs):
        raise MSLDispatchUnsupported("empty tensors must use the pure MLX fallback")
    return kernel(
        inputs=list(inputs),
        template=list(template) if template else None,
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[tuple(shape) for shape in output_shapes],
        output_dtypes=list(output_dtypes),
        stream=mx.gpu,
    )


# ---------------------------------------------------------------------------
# Optional TileLang -> MLX inline lowering helper
# ---------------------------------------------------------------------------


_KERNEL_DEF_RE = re.compile(r"kernel\s+void\s+(?P<name>\w+)\s*\(", re.MULTILINE)


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

    match = _KERNEL_DEF_RE.search(msl)
    if match is None:
        raise RuntimeError("TileLang MSL: missing 'kernel void' declaration.")
    prelude = msl[: match.start()].rstrip()

    sig_start = match.end()
    depth = 1
    i = sig_start
    while i < len(msl) and depth > 0:
        ch = msl[i]
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
        ch = msl[j]
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


def lower_tilelang_to_msl_inline(prim_func: Any) -> TileLangMSLLowering:
    """Lower a TileLang PrimFunc to MSL and prepare an inline body for MLX.

    The TileLang ``kernel void`` body is preserved verbatim and extended with
    ``blockIdx``/``threadIdx`` aliases that map to MLX's
    ``threadgroup_position_in_grid`` / ``thread_position_in_threadgroup``
    builtins. Buffer references stay in the order TileLang chose
    (alphabetic), so the caller MUST pass ``input_names + output_names`` in
    that same order to ``mx.fast.metal_kernel``.

    Raises ``MSLDispatchUnsupported`` when tilelang or its Metal target is
    unavailable, mirroring the pure-MLX fallback contract.
    """

    try:
        from tilelang import tvm  # type: ignore
        from tilelang.engine.lower import lower as tl_lower  # type: ignore
    except Exception as exc:  # pragma: no cover - guarded by callers
        raise MSLDispatchUnsupported(
            f"tilelang import failed: {exc}; falling back to pure MLX"
        ) from exc

    target = tvm.target.Target("metal")
    artifact = tl_lower(prim_func, target=target)

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
    body = (
        "    uint3 blockIdx = threadgroup_position_in_grid;\n"
        "    uint3 threadIdx = thread_position_in_threadgroup;\n"
        + inner
    )
    return TileLangMSLLowering(
        header=prelude,
        body=body,
        grid=(grid[0], grid[1], grid[2]),
        threadgroup=(block[0], block[1], block[2]),
        msl_text=msl_text,
        buffer_param_names=_parse_buffer_param_names(sig_text),
        kernel_name=_KERNEL_DEF_RE.search(msl_text).group("name"),  # type: ignore[union-attr]
    )


__all__ = [
    "MSLDispatchStatus",
    "MSLDispatchUnsupported",
    "MetalKernel",
    "TileLangMSLLowering",
    "can_run_metal",
    "dispatch",
    "lower_tilelang_to_msl_inline",
    "make_metal_kernel",
    "msl_dispatch_status",
]

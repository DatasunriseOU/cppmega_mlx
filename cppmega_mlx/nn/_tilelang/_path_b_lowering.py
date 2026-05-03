"""Path B TileLang->MSL transform helpers.

This module vendors the small string-rewriting layer that turns a TileLang
Metal-target ``kernel_source`` into something MLX's ``mx.fast.metal_kernel``
can host.

Pipeline (from the original Path B prototype at
``/tmp/path_b_msl_mlx/bench_msl_path_b.py`` that proved the round-trip
on a manual GEMM):

1. Build a TileLang ``PrimFunc`` and ``lower(prim, target=Target("metal"))``.
2. The emitted MSL contains a complete ``kernel void <name>(... [[ buffer(N) ]],
   [[threadgroup_position_in_grid]], [[thread_position_in_threadgroup]])``.
3. MLX expects only the body of a ``kernel void`` (it generates the wrapping
   signature itself). We rewrite the TileLang kernel into an
   ``inline void <helper_name>(...)`` and inject it into MLX's ``header=``.
4. Three concrete pitfalls handled here:
   a. The full ``kernel void`` signature is parsed and replaced with an
      ``inline void`` signature; the body is preserved verbatim.
   b. MLX passes inputs as ``const device T*`` and outputs as ``device T*``.
      The vendored helper marks input buffers ``const device`` based on a
      caller-provided set; TileLang emits everything as plain ``device``.
   c. TileLang's metal codegen reorders buffer parameters alphabetically.
      We parse the post-rewrite signature to get the actual MSL order, and
      build the call argument list in that order so the MLX wrapper invokes
      the helper correctly.

The module is intentionally narrow. It is only useful for kernels whose
TileLang lowering already succeeds. For any kernel where TileLang's metal
codegen blows up before producing MSL, there is nothing to rewrite and the
caller must take a pure-MLX fallback. ``cppmega_mlx/nn/_tilelang/topk_selector.py``
documents one such case.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

KERNEL_DEF_RE = re.compile(
    r"kernel\s+void\s+(?P<name>\w+)\s*\(",
    re.MULTILINE,
)


@dataclass(frozen=True)
class TransformedKernel:
    """The MSL fragments needed to wire a TileLang kernel into MLX."""

    header: str
    body: str
    helper_name: str
    kernel_name: str
    # The buffer parameter names parsed from the (alphabetically-ordered)
    # MSL signature emitted by TileLang.
    buffer_param_names: tuple[str, ...]
    has_block_idx: bool
    has_thread_idx: bool


def _split_signature_params(signature_text: str) -> list[str]:
    """Split a comma-separated MSL parameter list at top-level commas only."""

    params: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in signature_text:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            params.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        last = "".join(current).strip()
        if last:
            params.append(last)
    return params


def transform_tilelang_kernel(
    msl: str,
    *,
    helper_name: str = "tilelang_helper",
    const_buffer_names: Sequence[str] | None = None,
) -> TransformedKernel:
    """Strip the TileLang ``kernel void`` and re-emit as ``inline void``.

    Parameters
    ----------
    msl:
        The complete TileLang Metal target ``kernel_source`` string.
    helper_name:
        The name to use for the generated ``inline void`` helper that
        the MLX body will call.
    const_buffer_names:
        Names of buffers (matching the TileLang param names in the
        post-rewrite alphabetic order) that should be marked ``const device``
        in the helper signature. MLX always passes inputs as ``const device``;
        outputs stay plain ``device``.

    Returns
    -------
    TransformedKernel:
        Header text suitable for MLX ``header=``, plus the buffer parameter
        names in MSL order so the caller can build a correct argument list
        in :func:`build_mlx_body`.
    """

    if const_buffer_names is None:
        const_buffer_names = ()
    const_set = set(const_buffer_names)

    match = KERNEL_DEF_RE.search(msl)
    if match is None:
        raise RuntimeError("Could not find `kernel void` declaration in TileLang MSL.")

    kernel_name = match.group("name")
    sig_start = match.end()  # right after the opening paren

    # Walk the kernel signature until we hit the matching ')'.
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
        raise RuntimeError("Unbalanced parens in kernel signature.")
    sig_end = i  # one past the closing ')'
    sig_text = msl[sig_start: sig_end - 1]  # without trailing ')'

    # Walk past whitespace to the body opener '{'.
    j = sig_end
    while j < len(msl) and msl[j].isspace():
        j += 1
    if j >= len(msl) or msl[j] != "{":
        raise RuntimeError("Kernel body not found after signature.")
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
        raise RuntimeError("Unbalanced braces in kernel body.")
    body_text = msl[body_start: j]  # includes braces

    # Rewrite each parameter declaration:
    #   * drop Metal attributes ([[ buffer(N) ]] etc.) since this is no
    #     longer a ``kernel void``
    #   * mark const where requested
    #   * replace [[threadgroup_position_in_grid]] / [[thread_position_in_threadgroup]]
    #     with plain uint3 args we will pass from the MLX wrapper
    transformed_decls: list[str] = []
    buffer_names: list[str] = []
    has_block_idx = False
    has_thread_idx = False
    for decl in _split_signature_params(sig_text):
        clean = re.sub(r"\[\[.*?\]\]", "", decl).strip()
        m = re.search(r"(\w+)\s*$", clean)
        if not m:
            transformed_decls.append(clean)
            continue
        var_name = m.group(1)
        if var_name == "blockIdx":
            has_block_idx = True
            transformed_decls.append("uint3 blockIdx")
        elif var_name == "threadIdx":
            has_thread_idx = True
            transformed_decls.append("uint3 threadIdx")
        elif "device" in clean:
            if var_name in const_set and "const device" not in clean:
                clean = re.sub(r"\bdevice\b", "const device", clean, count=1)
            transformed_decls.append(clean)
            buffer_names.append(var_name)
        else:
            transformed_decls.append(clean)
            buffer_names.append(var_name)

    new_sig = ",\n    ".join(transformed_decls)

    helper = (
        f"\n// ---- TileLang-derived helper: original kernel `{kernel_name}` "
        f"rewritten as inline void ----\n"
        f"inline void {helper_name}(\n    {new_sig}\n) {body_text}\n"
    )

    # The prelude (before the kernel def) carries `#include <metal_stdlib>`,
    # `using namespace metal;`, and any TileLang typedefs. Keep all of it.
    prelude = msl[: match.start()].strip()
    header_text = prelude + "\n" + helper

    return TransformedKernel(
        header=header_text,
        body=body_text,
        helper_name=helper_name,
        kernel_name=kernel_name,
        buffer_param_names=tuple(buffer_names),
        has_block_idx=has_block_idx,
        has_thread_idx=has_thread_idx,
    )


def build_mlx_body(
    transformed: TransformedKernel,
    *,
    input_names: Sequence[str],
    output_names: Sequence[str],
    primfunc_param_names: Sequence[str],
) -> str:
    """Generate the MLX kernel ``source`` body that calls the helper.

    TileLang's metal codegen reorders the kernel's buffer parameters
    alphabetically. ``transformed.buffer_param_names`` is that alphabetic
    order. The MLX wrapper passes inputs and outputs in caller-determined
    order, so we map PrimFunc-name -> MLX-local-name and emit the helper
    call with arguments in MSL (alphabetic) order.
    """

    pos_lines: list[str] = []
    if transformed.has_block_idx:
        pos_lines.append("    uint3 blockIdx = threadgroup_position_in_grid;")
    if transformed.has_thread_idx:
        pos_lines.append("    uint3 threadIdx = thread_position_in_threadgroup;")

    nin = len(input_names)
    nout = len(output_names)
    msl_order = transformed.buffer_param_names
    if nin + nout != len(msl_order):
        raise RuntimeError(
            f"Buffer count mismatch: MSL has {len(msl_order)} buffers, "
            f"but caller passed {nin} inputs + {nout} outputs."
        )
    if len(primfunc_param_names) != len(msl_order):
        raise RuntimeError(
            "primfunc_param_names length does not match MSL buffer count."
        )

    pf_to_mlx: dict[str, str] = {}
    for idx, name in enumerate(input_names):
        pf_to_mlx[primfunc_param_names[idx]] = name
    for idx, name in enumerate(output_names):
        pf_to_mlx[primfunc_param_names[nin + idx]] = name

    args: list[str] = []
    for msl_name in msl_order:
        if msl_name not in pf_to_mlx:
            raise RuntimeError(
                f"MSL buffer name `{msl_name}` not in PrimFunc names "
                f"{tuple(primfunc_param_names)}."
            )
        args.append(pf_to_mlx[msl_name])
    if transformed.has_block_idx:
        args.append("blockIdx")
    if transformed.has_thread_idx:
        args.append("threadIdx")

    body_lines: list[str] = list(pos_lines)
    body_lines.append(f"    {transformed.helper_name}({', '.join(args)});")
    return "\n".join(body_lines) + "\n"


__all__ = [
    "KERNEL_DEF_RE",
    "TransformedKernel",
    "build_mlx_body",
    "transform_tilelang_kernel",
]

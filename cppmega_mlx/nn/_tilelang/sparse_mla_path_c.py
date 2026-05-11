"""Path C Sparse-MLA forward/backward via TileLang DSL ``@T.prim_func`` lowering.

This module is the first TileLang-DSL counterpart to the Path B direct-MSL
Sparse-MLA kernels in :mod:`cppmega_mlx.nn._tilelang.sparse_mla`.

The upstream TileLang Sparse-MLA backward examples are CUDA-oriented: they use
``T.gemm`` for the attention matmuls and atomics for dKV scatter. Both are
still the wrong first step on Apple Metal. This Path C kernel instead mirrors
Path B's partial-output contract:

* compute ``dq`` directly;
* emit ``dkv_partial[B, S, H, topk, D]`` without atomics;
* reuse Path B's host-side ``_reduce_dkv_partial`` scatter/reduction.

The kernel keeps the TOPK softmax state in static threadgroup buffers and uses
power-of-two tree reductions for the max/sum/rowsum phases. That mirrors Path
B's direct-MSL contract while keeping the source in TileLang DSL.
"""

# pyright: reportInvalidTypeForm=false, reportMissingImports=false

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast

import mlx.core as mx

from cppmega_mlx.nn._tilelang import _msl_transform
from cppmega_mlx.nn._tilelang._engine_dispatch import dispatch_lower
from cppmega_mlx.nn._tilelang._msl_transform import (
    MSLDispatchUnsupported,
    can_run_metal,
)
from cppmega_mlx.nn._tilelang.sparse_mla import _promote_to_fp16_carrier, _reduce_dkv_partial
from cppmega_mlx.nn.sparse_mla import _resolve_shapes, sparse_mla_attention_reference


_KV_BOUNDS_CHECK_RE = re.compile(
    r"(?P<indent>[ \t]*)half (?P<var>condval(?:_\d+)?);\n"
    r"(?P=indent)if \(\(\(0 <= gather_idx\[0\]\) && "
    r"\(gather_idx\[0\] < (?P<seq_len_kv>\d+)\)\)\) \{\n"
    r"(?P=indent)  (?P=var) = (?P<load>kv\[[^\n]+]);\n"
    r"(?P=indent)\} else \{\n"
    r"(?P=indent)  (?P=var) = 0\.000000e\+00h;\n"
    r"(?P=indent)\}",
    re.MULTILINE,
)


_FLAT_KV_BOUNDS_CHECK_RE = re.compile(
    r"(?P<indent>[ \t]*)half (?P<var>condval(?:_\d+)?);\n"
    r"(?P=indent)if \([^\n]*\) \{\n"
    r"(?P=indent)  (?P=var) = (?P<load>kv\[[^\n]+]);\n"
    r"(?P=indent)\} else \{\n"
    r"(?P=indent)  (?P=var) = 0\.000000e\+00h;\n"
    r"(?P=indent)\}",
    re.MULTILINE,
)


def _remove_redundant_kv_bounds_checks(msl: str, *, seq_len_kv: int) -> str:
    """Mirror Path B's valid-index contract by deleting hot-loop KV load guards."""

    def replace(match: re.Match[str]) -> str:
        if int(match.group("seq_len_kv")) != seq_len_kv:
            return match.group(0)
        return f"{match.group('indent')}half {match.group('var')} = {match.group('load')};"

    return _KV_BOUNDS_CHECK_RE.sub(replace, msl)


def _remove_redundant_flat_kv_bounds_checks(msl: str) -> str:
    """Delete TileLang's flat KV load guards from Sparse-MLA hot loops."""

    def replace(match: re.Match[str]) -> str:
        return f"{match.group('indent')}half {match.group('var')} = {match.group('load')};"

    return _FLAT_KV_BOUNDS_CHECK_RE.sub(replace, msl)


def _insert_forward_all_masked_fast_return(msl: str, *, d_v: int, threads: int) -> str:
    """Match Path B's all-masked row fast path in lowered forward MSL."""

    row_max_decl = "  float row_max = reduce_buf[0];\n"
    if row_max_decl not in msl or "row_max == -INFINITY" in msl:
        return msl
    fast_return = (
        row_max_decl
        + "  if (row_max == -INFINITY) {\n"
        + f"    for (int d = ((int)threadIdx.x); d < {d_v}; d += threads) {{\n"
        + f"      out[(((int)blockIdx.x) * {d_v}) + d] = ((half)0.000000e+00f);\n"
        + "    }\n"
        + "    if (((int)threadIdx.x) == 0) {\n"
        + "      lse[((int)blockIdx.x)] = 0.000000e+00f;\n"
        + "    }\n"
        + "    return;\n"
        + "  }\n"
    )
    return msl.replace(row_max_decl, fast_return, 1)


def _canonicalize_fwd_lane_indexing(
    msl: str,
    *,
    topk: int,
    threads: int,
    d_v: int,
) -> str:
    """Trim TileLang lane-loop syntax back to Path-B-style MSL loops."""

    marker = "  float sm_scale = sm_scale_buf[0];"
    if marker in msl and "  int tid = int(threadIdx.x);" not in msl:
        msl = msl.replace(
            marker,
            "  int gid = int(blockIdx.x);\n"
            "  int tid = int(threadIdx.x);\n"
            f"  int threads = {threads};\n"
            + marker,
            1,
        )

    msl = msl.replace("((int)threadIdx.x)", "tid")
    msl = msl.replace("((int)blockIdx.x)", "gid")
    msl = msl.replace("((int)thread_position_in_threadgroup.x)", "tid")
    msl = msl.replace("((int)threadgroup_position_in_grid.x)", "gid")
    msl = msl.replace("reduce_buf[(((long)tid) + ((long)stride))]", "reduce_buf[tid + stride]")

    shift = threads.bit_length() - 1
    if topk == threads:
        loop_limit = topk + threads - 1
        for tmp in ("_tmp", "_tmp_1", "_tmp_2", "_tmp_3"):
            header = (
                f"  for (int {tmp} = 0; {tmp} < (({loop_limit} - tid) >> {shift}); "
                f"++{tmp}) {{"
            )
            if header not in msl:
                continue
            msl = msl.replace(
                header,
                f"  for (int k = tid; k < {topk}; k += threads) {{",
                1,
            )
            msl = msl.replace(f"(({tmp} * {threads}) + tid)", "k")
            msl = msl.replace(f"({tmp} * {threads})", "(k - tid)")
        msl = re.sub(
            r"(?P<prefix>\+\s*)\(k - tid\)\) \+ tid\)",
            r"\g<prefix>k)",
            msl,
        )
        msl = re.sub(
            r"indices\[\({2,}\(gid >> (?P<shift>\d+)\) \* (?P<mult>\d+)\) \+ k\)\]",
            r"indices[((gid >> \g<shift>) * \g<mult>) + k]",
            msl,
        )
        # After ``((_tmp * N) + tid)`` collapses to ``k``, the inner-body ``int k = k;``
        # / ``int k_N = k;`` decls become tautologies. Drop them — the outer ``for``
        # loop owns ``k``, and the per-iteration aliases are unused (modulo a few
        # warnings the test suite cares about).
        msl = re.sub(r"(?m)^[ \t]*int k = k;\n", "", msl)
        msl = re.sub(r"(?m)^[ \t]*int k_\d+ = k;\n", "", msl)

    d_loop_limit = d_v + threads - 1
    d_header = (
        f"  for (int _tmp_4 = 0; _tmp_4 < (({d_loop_limit} - tid) >> {shift}); "
        "++_tmp_4) {"
    )
    if d_header in msl:
        msl = msl.replace(d_header, f"  for (int d = tid; d < {d_v}; d += threads) {{", 1)
        msl = msl.replace(f"((_tmp_4 * {threads}) + tid)", "d")
        msl = msl.replace(f"(_tmp_4 * {threads})", "(d - tid)")
        msl = re.sub(
            r"kv\[\(\(d - tid\) \+ (?P<base>[A-Za-z_][A-Za-z0-9_]*)\) \+ tid\]",
            r"kv[\g<base> + d]",
            msl,
        )
        msl = re.sub(
            r"out\[\(\(\(gid \* (?P<dim>\d+)\) \+ \(d - tid\)\) \+ tid\)\]",
            r"out[(gid * \g<dim>) + d]",
            msl,
        )
        # Same cleanup as the topk loop: drop tautological ``int d_N = d;`` aliases.
        msl = re.sub(r"(?m)^[ \t]*int d_\d+ = d;\n", "", msl)
    return msl


def _canonicalize_fwd_reductions(msl: str, *, threads: int) -> str:
    """Use Path-B-style stride reductions instead of TileLang round loops."""

    barrier = r"(?:metal::)?threadgroup_barrier\(metal::mem_flags::mem_threadgroup\);"
    rounds = threads.bit_length() - 1
    # z3-final may CSE ``long(tid)`` into a temporary before reduction lowering.
    # Normalize that shape before matching the stride-reduction templates below.
    msl = re.sub(
        r"reduce_buf\[\((?:cse_v\d+(?:_\d+)?|tid) \+ \(\(long\)stride\)\)\]",
        "reduce_buf[tid + stride]",
        msl,
    )
    msl = msl.replace("reduce_buf[(tid + stride)]", "reduce_buf[tid + stride]")
    max_reduction_re = re.compile(
        rf"  for \(int round_id = 0; round_id < {rounds}; \+\+round_id\) \{{\n"
        rf"    stride = \({threads} >> \(round_id \+ 1\)\);\n"
        r"    if \(tid < stride\) \{\n"
        r"      if \(reduce_buf\[tid\] < reduce_buf\[tid \+ stride\]\) \{\n"
        r"        reduce_buf\[tid\] = reduce_buf\[tid \+ stride\];\n"
        r"      \}\n"
        r"    \}\n"
        rf"    {barrier}\n"
        r"  \}"
    )
    max_replacement = (
        "  for (int stride = threads / 2; stride > 0; stride >>= 1) {\n"
        "    if (tid < stride) {\n"
        "      float a = reduce_buf[tid];\n"
        "      float b_v = reduce_buf[tid + stride];\n"
        "      if (b_v > a) reduce_buf[tid] = b_v;\n"
        "    }\n"
        "    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);\n"
        "  }"
    )
    msl = max_reduction_re.sub(max_replacement, msl, count=1)

    sum_reduction_re = re.compile(
        rf"  for \(int round_id_1 = 0; round_id_1 < {rounds}; \+\+round_id_1\) \{{\n"
        rf"    stride = \({threads} >> \(round_id_1 \+ 1\)\);\n"
        r"    if \(tid < stride\) \{\n"
        r"      reduce_buf\[tid\] (?:= \(reduce_buf\[tid\] \+ reduce_buf\[tid \+ stride\]\)|\+= reduce_buf\[tid \+ stride\]);\n"
        r"    \}\n"
        rf"    {barrier}\n"
        r"  \}"
    )
    sum_replacement = (
        "  for (int stride = threads / 2; stride > 0; stride >>= 1) {\n"
        "    if (tid < stride) {\n"
        "      reduce_buf[tid] += reduce_buf[tid + stride];\n"
        "    }\n"
        "    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);\n"
        "  }"
    )
    msl = sum_reduction_re.sub(sum_replacement, msl, count=1)

    max_reduction = (
        f"  for (int round_id = 0; round_id < {threads.bit_length() - 1}; ++round_id) {{\n"
        f"    stride = ({threads} >> (round_id + 1));\n"
        "    if (tid < stride) {\n"
        "      if (reduce_buf[tid] < reduce_buf[tid + stride]) {\n"
        "        reduce_buf[tid] = reduce_buf[tid + stride];\n"
        "      }\n"
        "    }\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "  }"
    )
    max_replacement = (
        "  for (int stride = threads / 2; stride > 0; stride >>= 1) {\n"
        "    if (tid < stride) {\n"
        "      float a = reduce_buf[tid];\n"
        "      float b_v = reduce_buf[tid + stride];\n"
        "      if (b_v > a) reduce_buf[tid] = b_v;\n"
        "    }\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "  }"
    )
    msl = msl.replace(max_reduction, max_replacement, 1)

    sum_reduction = (
        f"  for (int round_id_1 = 0; round_id_1 < {threads.bit_length() - 1}; ++round_id_1) {{\n"
        f"    stride = ({threads} >> (round_id_1 + 1));\n"
        "    if (tid < stride) {\n"
        "      reduce_buf[tid] = (reduce_buf[tid] + reduce_buf[tid + stride]);\n"
        "    }\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "  }"
    )
    sum_replacement = (
        "  for (int stride = threads / 2; stride > 0; stride >>= 1) {\n"
        "    if (tid < stride) {\n"
        "      reduce_buf[tid] += reduce_buf[tid + stride];\n"
        "    }\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "  }"
    )
    msl = msl.replace(sum_reduction, sum_replacement, 1)

    inv_sum_block = (
        "  inv_sum = 0.000000e+00f;\n"
        "  if (0.000000e+00f < sumexp) {\n"
        "    inv_sum = (1.000000e+00f / sumexp);\n"
        "  }"
    )
    msl = msl.replace(
        inv_sum_block,
        "  inv_sum = (sumexp > 0.000000e+00f) ? (1.000000e+00f / sumexp) : 0.000000e+00f;",
        1,
    )

    lse_block = (
        "  if (tid == 0) {\n"
        "    if (0.000000e+00f < sumexp) {\n"
        "      lse[gid] = (row_max + log(sumexp));\n"
        "    } else {\n"
        "      lse[gid] = 0.000000e+00f;\n"
        "    }\n"
        "  }"
    )
    msl = msl.replace(
        lse_block,
        "  if (tid == 0) {\n"
        "    lse[gid] = (row_max + log(sumexp));\n"
        "  }",
        1,
    )
    return msl


def _canonicalize_fwd_hot_loops(msl: str) -> str:
    """Remove residual TileLang scalarization overhead from forward hot loops."""

    msl = msl.replace("  int stride;\n", "")
    if "    stride = (" in msl:
        if "  uint threads =" in msl:
            msl = msl.replace("  uint threads =", "  uint stride;\n  uint threads =", 1)
        elif "  int threads =" in msl:
            msl = msl.replace("  int threads =", "  uint stride;\n  int threads =", 1)

    msl = re.sub(
        r"(?P<indent>[ \t]*)gather_idx = (?P<idx>indices\[[^\n]+]);\n"
        r"(?P=indent)if \(gather_idx < 0\) \{\n"
        r"(?P=indent)  scores\[k\] = -INFINITY;\n"
        r"(?P=indent)\} else \{\n"
        r"(?P=indent)  acc = 0\.000000e\+00f;\n"
        r"(?P<body>"
        r"(?P=indent)  (?:int|uint) kv_row_base = [^\n]+;\n"
        r"(?P=indent)  for \((?:int|uint) d = 0; d < \d+; \+\+d\) \{\n"
        r"(?P=indent)    acc = \(acc \+ \(\(\(float\)q\[[^\n]+\]\) \* \(\(float\)kv\[[^\n]+\]\)\)\);\n"
        r"(?P=indent)  \}\n"
        r"(?P=indent)  scores\[k\] = \(acc \* sm_scale\);\n)"
        r"(?P=indent)\}",
        (
            r"\g<indent>int gather_idx = \g<idx>;\n"
            r"\g<indent>if (gather_idx < 0) {\n"
            r"\g<indent>  scores[k] = -INFINITY;\n"
            r"\g<indent>  continue;\n"
            r"\g<indent>}\n"
            r"\g<indent>acc = 0.000000e+00f;\n"
            r"\g<body>"
        ),
        msl,
    )

    lines = msl.splitlines()
    rewritten: list[str] = []
    i = 0
    qk_condval = re.compile(
        r"(?P<indent>[ \t]*)half (?P<var>condval) = (?P<load>kv\[[^\n]+]);$"
    )
    while i < len(lines):
        match = qk_condval.match(lines[i])
        if match and i + 1 < len(lines):
            next_line = lines[i + 1]
            var = match.group("var")
            if f"((float){var})" in next_line and "q[" in next_line:
                rewritten.append(
                    next_line.replace(
                        f"((float){var})",
                        f"((float){match.group('load')})",
                    )
                )
                i += 2
                continue
        rewritten.append(lines[i])
        i += 1
    msl = "\n".join(rewritten)
    if msl and not msl.endswith("\n"):
        msl += "\n"

    msl = re.sub(
        r"(?P<indent>[ \t]*)if \(0 <= gather_idx\) \{\n"
        r"(?P=indent)  (?P<body>int kv_row_base_1 = [^\n]+;\n)"
        r"(?P=indent)  half (?P<var>condval_\d+) = (?P<load>kv\[[^\n]+]);\n"
        r"(?P=indent)  acc = \(acc \+ \(scores\[k\] \* \(\(float\)(?P=var)\)\)\);\n"
        r"(?P=indent)\}",
        (
            r"\g<indent>if (gather_idx < 0) {\n"
            r"\g<indent>  continue;\n"
            r"\g<indent>}\n"
            r"\g<indent>\g<body>"
            r"\g<indent>acc = (acc + (scores[k] * ((float)\g<load>)));"
        ),
        msl,
    )
    msl = re.sub(
        r"(?P<indent>[ \t]*)if \(gather_idx < 0\) \{\n"
        r"(?P=indent)  continue;\n"
        r"(?P=indent)\}\n"
        r"(?P=indent)(?P<body>uint kv_row_base_1 = [^\n]+;\n)"
        r"(?P=indent)acc = \(acc \+ \(scores\[k\] \* \(\(float\)kv\[(?P<load>[^\n]+)\]\)\)\);",
        (
            r"\g<indent>if (gather_idx < 0) {\n"
            r"\g<indent>continue;\n"
            r"\g<indent>}\n"
            r"\g<indent>\g<body>"
            r"\g<indent>acc = (acc + (scores[k] * ((float)kv[\g<load>])));"
        ),
        msl,
    )
    msl = re.sub(
        r"(?P<indent>[ \t]*)acc = 0\.000000e\+00f;\n"
        r"(?P=indent)(?:int|uint) (?P<base>kv_row_base(?:_\d+)?) = (?P<base_expr>[^\n]+);\n"
        r"(?P=indent)for \((?:int|uint) d = 0; d < (?P<dim>\d+); \+\+d\) \{\n"
        r"(?P=indent)  acc = \(acc \+ \(\(\(float\)(?P<q>q\[[^\n]+\])\) \* "
        r"\(\(float\)(?P<kv>kv\[(?P=base) \+ d\])\)\)\);\n"
        r"(?P=indent)\}",
        (
            r"\g<indent>uint \g<base> = \g<base_expr>;\n"
            r"\g<indent>float acc = 0.0f;\n"
            r"\g<indent>for (uint d = 0; d < \g<dim>; ++d) {\n"
            r"\g<indent>  float qv = float(\g<q>);\n"
            r"\g<indent>  float kv_v = float(\g<kv>);\n"
            r"\g<indent>  acc += qv * kv_v;\n"
            r"\g<indent>}"
        ),
        msl,
        count=1,
    )
    msl = re.sub(
        r"(?P<indent>[ \t]*)local = -INFINITY;\n"
        r"(?P=indent)for \((?:int|uint) k = tid; k < (?P<topk>\d+); k \+= threads\) \{\n"
        r"(?P=indent)  if \(local < scores\[k\]\) \{\n"
        r"(?P=indent)    local = scores\[k\];\n"
        r"(?P=indent)  \}\n"
        r"(?P=indent)\}\n"
        r"(?P=indent)reduce_buf\[tid\] = local;",
        (
            r"\g<indent>float local_max = -INFINITY;\n"
            r"\g<indent>for (uint k = tid; k < \g<topk>; k += threads) {\n"
            r"\g<indent>  float v = scores[k];\n"
            r"\g<indent>  if (v > local_max) local_max = v;\n"
            r"\g<indent>}\n"
            r"\g<indent>reduce_buf[tid] = local_max;"
        ),
        msl,
        count=1,
    )
    msl = re.sub(
        r"(?P<indent>[ \t]*)local = 0\.000000e\+00f;\n"
        r"(?P=indent)for \((?:int|uint) k = tid; k < (?P<topk>\d+); k \+= threads\) \{\n"
        r"(?P=indent)  local = \(local \+ scores\[k\]\);\n"
        r"(?P=indent)\}\n"
        r"(?P=indent)reduce_buf\[tid\] = local;",
        (
            r"\g<indent>float local_sum = 0.0f;\n"
            r"\g<indent>for (uint k = tid; k < \g<topk>; k += threads) {\n"
            r"\g<indent>  local_sum += scores[k];\n"
            r"\g<indent>}\n"
            r"\g<indent>reduce_buf[tid] = local_sum;"
        ),
        msl,
        count=1,
    )
    msl = re.sub(
        r"(?P<indent>[ \t]*)inv_sum = \(sumexp > 0\.000000e\+00f\) \? "
        r"\(1\.000000e\+00f / sumexp\) : 0\.000000e\+00f;",
        (
            r"\g<indent>inv_sum = "
            r"(sumexp > 0.000000e+00f) ? (1.000000e+00f / sumexp) : 0.000000e+00f;"
        ),
        msl,
        count=1,
    )
    msl = re.sub(
        r"(?P<indent>[ \t]*)acc = 0\.000000e\+00f;\n"
        r"(?P<body>"
        r"(?P=indent)for \((?:int|uint) k = 0; k < (?P<topk>\d+); \+\+k\) \{\n"
        r"(?P=indent)  (?:int )?gather_idx = indices\[[^\n]+\];\n"
        r"(?P=indent)  if \(gather_idx < 0\) \{\n"
        r"(?P=indent)    continue;\n"
        r"(?P=indent)  \}\n"
        r"(?P=indent)  (?:int|uint) kv_row_base_1 = [^\n]+;\n)"
        r"(?P=indent)  acc = \(acc \+ \(scores\[k\] \* \(\(float\)(?P<kv>kv\[kv_row_base_1 \+ d\])\)\)\);\n"
        r"(?P=indent)\}",
        (
            r"\g<indent>float acc = 0.0f;\n"
            r"\g<body>"
            r"\g<indent>  float kv_v = float(\g<kv>);\n"
            r"\g<indent>  acc += scores[k] * kv_v;\n"
            r"\g<indent>}"
        ),
        msl,
        count=1,
    )
    msl = re.sub(
        r"(?P<indent>[ \t]*)for \((?:int|uint) k = tid; k < (?P<topk>\d+); k \+= threads\) \{\n"
        r"(?P=indent)  if \(scores\[k\] == -INFINITY\) \{\n"
        r"(?P=indent)    scores\[k\] = 0\.000000e\+00f;\n"
        r"(?P=indent)  \} else \{\n"
        r"(?P=indent)    scores\[k\] = exp\(\(scores\[k\] - (?P<max_name>row_max|m_i)\)\);\n"
        r"(?P=indent)  \}\n"
        r"(?P=indent)\}",
        (
            r"\g<indent>for (uint k = tid; k < \g<topk>; k += threads) {\n"
            r"\g<indent>  float v = scores[k];\n"
            r"\g<indent>  if (v == -INFINITY) {\n"
            r"\g<indent>    scores[k] = 0.0f;\n"
            r"\g<indent>  } else {\n"
            r"\g<indent>    scores[k] = exp(v - \g<max_name>);\n"
            r"\g<indent>  }\n"
            r"\g<indent>}"
        ),
        msl,
        count=1,
    )
    msl = re.sub(
        r"(?P<indent>[ \t]*)reduce_buf\[tid\] = "
        r"\(reduce_buf\[tid\] \+ reduce_buf\[tid \+ stride\]\);",
        r"\g<indent>reduce_buf[tid] += reduce_buf[tid + stride];",
        msl,
    )
    if "acc = " not in msl and "local = " not in msl and "inv_sum = " not in msl:
        msl = msl.replace("  float acc;\n", "")
        msl = msl.replace("  float local;\n", "")
        msl = msl.replace("  float inv_sum;\n", "")
    return msl


def _canonicalize_fwd_unsigned_msl(msl: str) -> str:
    """Use unsigned lane/index arithmetic like the hand-written Path B kernel."""

    replacements = {
        "int gid = int(blockIdx.x);": "uint gid = blockIdx.x;",
        "int tid = int(threadIdx.x);": "uint tid = threadIdx.x;",
        "uint gid = blockIdx.x;": "uint gid = threadgroup_position_in_grid.x;",
        "uint tid = threadIdx.x;": "uint tid = thread_position_in_threadgroup.x;",
        "int threads = 32;": "uint threads = 32;",
        "int threads = 64;": "uint threads = 64;",
        "int threads = 16;": "uint threads = 16;",
        "int threads = 8;": "uint threads = 8;",
        "int threads = 4;": "uint threads = 4;",
        "for (int stride = threads / 2; stride > 0; stride >>= 1)": (
            "for (uint stride = threads / 2; stride > 0; stride >>= 1)"
        ),
    }
    for old, new in replacements.items():
        msl = msl.replace(old, new)
    msl = re.sub(
        r"for \(int d = 0; d < (?P<limit>\d+); \+\+d\)",
        r"for (uint d = 0; d < \g<limit>; ++d)",
        msl,
    )
    msl = re.sub(
        r"for \(int d = tid; d < (?P<limit>\d+); d \+= (?P<step>threads|\d+)\)",
        r"for (uint d = tid; d < \g<limit>; d += \g<step>)",
        msl,
    )
    msl = re.sub(
        r"for \(int k = 0; k < (?P<limit>\d+); \+\+k\)",
        r"for (uint k = 0; k < \g<limit>; ++k)",
        msl,
    )
    msl = re.sub(
        r"for \(int k = tid; k < (?P<limit>\d+); k \+= threads\)",
        r"for (uint k = tid; k < \g<limit>; k += threads)",
        msl,
    )
    msl = msl.replace("  uint3 blockIdx = threadgroup_position_in_grid;\n", "")
    msl = msl.replace("  uint3 threadIdx = thread_position_in_threadgroup;\n", "")
    msl = msl.replace("    uint3 blockIdx = threadgroup_position_in_grid;\n", "")
    msl = msl.replace("    uint3 threadIdx = thread_position_in_threadgroup;\n", "")

    msl = re.sub(r"(?<!for \()\bint k = tid;", "uint k = tid;", msl)
    msl = re.sub(
        r"\bint (?P<name>kv_row_base(?:_\d+)?) = (?P<expr>[^\n]*gather_idx[^\n]*);",
        r"uint \g<name> = \g<expr>;",
        msl,
    )
    msl = re.sub(r"\(gather_idx \* (?P<scale>\d+)\)", r"(uint(gather_idx) * \g<scale>)", msl)
    msl = re.sub(r"(?m)^  int gather_idx;\n", "", msl)
    return msl


def _canonicalize_fwd_base_indexing(
    msl: str,
    *,
    heads: int,
    seq_len: int,
    qk_dim: int,
    kv_group: int,
    head_kv: int,
    topk: int,
    seq_len_kv: int,
    d_v: int,
) -> str:
    """Hoist forward row bases so the hot loops match Path B's address shape."""

    marker = "  float sm_scale = sm_scale_buf[0];"
    if marker in msl and "  uint q_row_base =" not in msl:
        base_locals = (
            f"  uint h = gid % {heads};\n"
            f"  uint s = (gid / {heads}) % {seq_len};\n"
            f"  uint b = gid / {heads * seq_len};\n"
            f"  uint kv_group = {kv_group};\n"
            f"  uint head_kv = {head_kv};\n"
            "  uint g = h / head_kv;\n"
            f"  uint qk_dim = {qk_dim};\n"
            f"  uint d_v = {d_v};\n"
            f"  uint q_row_base = ((b * {seq_len} + s) * {heads} + h) * qk_dim;\n"
            f"  uint kv_outer_stride = {seq_len_kv} * kv_group * qk_dim;\n"
            "  uint kv_b_base = b * kv_outer_stride;\n"
            f"  uint idx_base = ((b * {seq_len} + s) * kv_group + g) * {topk};\n"
            f"  uint out_row = ((b * {seq_len} + s) * {heads} + h) * d_v;\n"
        )
        msl = msl.replace(marker, base_locals + marker, 1)

    msl = re.sub(
        r"indices\[(?P<expr>[^\]\n]*gid[^\]\n]*) \+ k\]",
        "indices[idx_base + k]",
        msl,
    )
    msl = re.sub(
        r"uint (?P<name>kv_row_base(?:_\d+)?) = [^\n;]*uint\(gather_idx\)[^\n;]*;",
        r"uint \g<name> = kv_b_base + (uint(gather_idx) * kv_group + g) * qk_dim;",
        msl,
    )
    msl = re.sub(r"q\[\(\(gid \* \d+\) \+ d\)\]", "q[q_row_base + d]", msl)
    msl = re.sub(r"out\[\(gid \* \d+\) \+ d\]", "out[out_row + d]", msl)
    msl = re.sub(
        r"(?m)^(?P<indent>[ \t]*)gather_idx = (?P<idx>indices\[idx_base \+ k\]);",
        r"\g<indent>int gather_idx = \g<idx>;",
        msl,
    )
    msl = re.sub(
        r"(?m)^(?P<indent>[ \t]*)gather_idx = "
        r"(?P<idx>indices\[[^\]\n]+\]);",
        r"\g<indent>int gather_idx = \g<idx>;",
        msl,
    )
    # Output-projection kd loop also reads ``gather_idx`` once per inner step;
    # z3-final leaves this as a bare assignment because the original lowering
    # had a thread-local ``int gather_idx[1]`` that we strip. Promote it to a
    # declaration so the identifier is in scope.
    msl = re.sub(
        r"(?m)^(?P<indent>[ \t]*)gather_idx = "
        r"(?P<idx>indices\[\(\(\(gid >> \d+\) \* \d+\) \+ k_\d+\)\]);",
        r"\g<indent>int gather_idx = \g<idx>;",
        msl,
    )
    msl = _canonicalize_fwd_cse_indexing(msl, qk_dim=qk_dim, d_v=d_v, topk=topk)
    return msl


def _canonicalize_fwd_cse_indexing(
    msl: str,
    *,
    qk_dim: int,
    d_v: int,
    topk: int,
) -> str:
    """Rewrite z3/CSE address temporaries after forward base hoisting."""

    cse_q_bases = set(
        re.findall(rf"(?m)^  int (cse_v\d+(?:_\d+)?) = \(gid \* {qk_dim}\);\n", msl)
    )
    cse_idx_bases = set(
        re.findall(rf"(?m)^  int (cse_v\d+(?:_\d+)?) = \(\(gid >> \d+\) \* {topk}\);\n", msl)
    )

    for delta in re.findall(r"(?m)^[ \t]*int (cse_v\d+(?:_\d+)?) = \(k - tid\);\n", msl):
        msl = re.sub(
            rf"\(\((?P<base>[^()\n]+) \+ {delta}\) \+ tid\)",
            r"(\g<base> + k)",
            msl,
        )
        msl = msl.replace(f"({delta} + tid)", "k")
        msl = re.sub(rf"(?m)^[ \t]*int {delta} = \(k - tid\);\n", "", msl)

    for delta in re.findall(r"(?m)^[ \t]*int (cse_v\d+(?:_\d+)?) = \(d - tid\);\n", msl):
        msl = re.sub(
            rf"\(\((?P<base>[^()\n]+) \+ {delta}\) \+ tid\)",
            r"(\g<base> + d)",
            msl,
        )
        msl = re.sub(
            rf"\(\({delta} \+ (?P<base>[^()\n]+)\) \+ tid\)",
            r"(\g<base> + d)",
            msl,
        )
        msl = msl.replace(f"({delta} + tid)", "d")
        msl = re.sub(rf"(?m)^[ \t]*int {delta} = \(d - tid\);\n", "", msl)

    for name in cse_idx_bases:
        msl = re.sub(
            rf"indices\[\(\({name} \+ (?P<off>[^)\]]+)\)\)\]",
            r"indices[idx_base + \g<off>]",
            msl,
        )
        msl = re.sub(
            rf"indices\[\({name} \+ (?P<off>[^)\]]+)\]",
            r"indices[idx_base + \g<off>]",
            msl,
        )
    for name in cse_q_bases:
        msl = re.sub(
            rf"\bq\[\({name} \+ (?P<off>[^)\]]+)\)\]",
            r"q[q_row_base + \g<off>]",
            msl,
        )
        msl = re.sub(
            rf"\bout\[\({name} \+ (?P<off>[^)\]]+)\)\]",
            r"out[out_row + \g<off>]",
            msl,
        )

    for name, off in re.findall(
        r"(?m)^[ \t]*int (cse_v\d+(?:_\d+)?) = \(kv_row_base(?:_1)? \+ (d(?:_1)?)\);\n",
        msl,
    ):
        msl = msl.replace(f"kv[{name}]", f"kv[kv_row_base + {off}]")
        msl = re.sub(
            rf"(?m)^[ \t]*int {name} = \(kv_row_base(?:_1)? \+ {off}\);\n",
            "",
            msl,
        )

    msl = re.sub(r"\buint kv_row_base_1 = ", "uint kv_row_base = ", msl)
    msl = msl.replace("kv[(kv_row_base + d)]", "kv[kv_row_base + d]")
    msl = msl.replace("out[(out_row + d)]", "out[out_row + d]")
    for name in re.findall(r"(?m)^  int (cse_v\d+(?:_\d+)?) = [^\n]+;\n", msl):
        decl_re = rf"(?m)^  int {name} = [^\n]+;\n"
        body_without_decl = re.sub(decl_re, "", msl)
        if re.search(rf"\b{name}\b", body_without_decl) is None:
            msl = body_without_decl
    return msl


def _canonicalize_fwd_qk_negative_continue(msl: str) -> str:
    """Convert the score-loop mask branch to Path-B's early-continue form."""

    msl = re.sub(
        r"(?P<indent>[ \t]*)if \(gather_idx < 0\) \{\n"
        r"(?P=indent)  scores\[k\] = -INFINITY;\n"
        r"(?P=indent)\} else \{\n"
        r"(?P=indent)  acc = 0\.000000e\+00f;\n"
        r"(?P=indent)  (?P<kv_decl>(?:uint|int) kv_row_base = [^\n]+;\n)"
        r"(?P=indent)  (?P<for_line>for \((?:uint|int) d = 0; d < \d+; \+\+d\) \{\n)"
        r"(?P=indent)    (?P<acc_line>acc = [^\n]+;\n)"
        r"(?P=indent)  \}\n"
        r"(?P=indent)  scores\[k\] = \(acc \* sm_scale\);\n"
        r"(?P=indent)\}",
        (
            r"\g<indent>if (gather_idx < 0) {\n"
            r"\g<indent>  scores[k] = -INFINITY;\n"
            r"\g<indent>  continue;\n"
            r"\g<indent>}\n"
            r"\g<indent>acc = 0.000000e+00f;\n"
            r"\g<indent>\g<kv_decl>"
            r"\g<indent>\g<for_line>"
            r"\g<indent>  \g<acc_line>"
            r"\g<indent>}\n"
            r"\g<indent>scores[k] = (acc * sm_scale);"
        ),
        msl,
        count=1,
    )
    return re.sub(
        r"(?P<indent>[ \t]*)if \(gather_idx < 0\) \{\n"
        r"(?P=indent)  scores\[k\] = -INFINITY;\n"
        r"(?P=indent)\} else \{\n"
        r"(?P<body>"
        r"(?P=indent)  uint kv_row_base = [^\n]+;\n"
        r"(?P=indent)  float acc = 0\.0f;\n"
        r"(?P=indent)  for \(uint d = 0; d < \d+; \+\+d\) \{\n"
        r"(?P=indent)    float qv = float\([^\n]+\);\n"
        r"(?P=indent)    float kv_v = float\([^\n]+\);\n"
        r"(?P=indent)    acc \+= qv \* kv_v;\n"
        r"(?P=indent)  \}\n"
        r"(?P=indent)  scores\[k\] = \(acc \* sm_scale\);\n)"
        r"(?P=indent)\}",
        (
            r"\g<indent>if (gather_idx < 0) {\n"
            r"\g<indent>  scores[k] = -INFINITY;\n"
            r"\g<indent>  continue;\n"
            r"\g<indent>}\n"
            r"\g<body>"
        ),
        msl,
        count=1,
    )


def _drop_unused_fwd_scalar_declarations(msl: str) -> str:
    """Remove TileLang scalar temp declarations once hot loops own their locals."""

    msl = re.sub(r"indices\[\(cse_v\d+(?:_\d+)? \+ k\)\]", "indices[idx_base + k]", msl)
    msl = re.sub(
        r"out\[\({2,}\(gid \* \d+\) \+ cse_v\d+(?:_\d+)?\) \+ tid\)\]",
        "out[out_row + d]",
        msl,
    )
    msl = re.sub(r"\bkv_row_base_\d+\b", "kv_row_base", msl)
    msl = msl.replace("kv[(kv_row_base + d)]", "kv[kv_row_base + d]")
    msl = msl.replace("out[(out_row + d)]", "out[out_row + d]")
    for name in re.findall(r"(?m)^  int (cse_v\d+(?:_\d+)?) = [^\n]+;\n", msl):
        decl_re = rf"(?m)^  int {name} = [^\n]+;\n"
        body_without_decl = re.sub(decl_re, "", msl)
        if re.search(rf"\b{name}\b", body_without_decl) is None:
            msl = body_without_decl
    msl = re.sub(r"(?m)^[ \t]*int gather_idx;\n", "", msl)
    msl = re.sub(
        r"(?m)^(?P<indent>[ \t]*)inv_sum = "
        r"\(sumexp > 0\.000000e\+00f\) \? "
        r"\(1\.000000e\+00f / sumexp\) : 0\.000000e\+00f;",
        r"\g<indent>float inv_sum = "
        r"(sumexp > 0.000000e+00f) ? (1.000000e+00f / sumexp) : 0.000000e+00f;",
        msl,
        count=1,
    )
    msl = re.sub(
        r"(?m)^      uint kv_row_base = ",
        "    uint kv_row_base = ",
        msl,
        count=1,
    )
    msl = re.sub(r"(?m)^      float acc = 0\.0f;", "    float acc = 0.0f;", msl, count=1)
    msl = re.sub(r"(?m)^      for \(uint d = 0;", "    for (uint d = 0;", msl, count=1)
    msl = re.sub(r"(?m)^        float qv = ", "      float qv = ", msl, count=1)
    msl = re.sub(r"(?m)^        float kv_v = ", "      float kv_v = ", msl, count=1)
    msl = re.sub(r"(?m)^        acc \+= qv \* kv_v;", "      acc += qv * kv_v;", msl, count=1)
    msl = re.sub(r"(?m)^      }\n      scores\[k\] = \(acc \* sm_scale\);", "    }\n    scores[k] = acc * sm_scale;", msl, count=1)
    msl = msl.replace("\n\n  }\n", "\n  }\n", 1)
    for name in ("acc", "local", "inv_sum"):
        if re.search(rf"(?m)^[ \t]*{name} = ", msl) is None:
            msl = re.sub(rf"(?m)^[ \t]*float {name};\n", "", msl)
    return msl


def _canonicalize_bwd_lane_indexing(
    msl: str,
    *,
    topk: int,
    threads: int,
    qk_dim: int,
    d_v: int,
) -> str:
    """Trim backward TileLang lane loops to the Path-B-style Metal shape."""

    marker = "  float sm_scale = sm_scale_buf[0];"
    if marker in msl and "  int tid = int(threadIdx.x);" not in msl:
        msl = msl.replace(
            marker,
            "  int gid = int(blockIdx.x);\n"
            "  int tid = int(threadIdx.x);\n"
            f"  int threads = {threads};\n"
            + marker,
            1,
        )

    msl = msl.replace("((int)threadIdx.x)", "tid")
    msl = msl.replace("((int)blockIdx.x)", "gid")
    msl = msl.replace("((int)thread_position_in_threadgroup.x)", "tid")
    msl = msl.replace("((int)threadgroup_position_in_grid.x)", "gid")
    msl = msl.replace("reduce_buf[(((long)tid) + ((long)stride))]", "reduce_buf[tid + stride]")
    shift = threads.bit_length() - 1
    lanes_limit = topk + threads - 1

    for tmp in (
        "_tmp",
        "_tmp_1",
        "_tmp_2",
        "_tmp_3",
        "_tmp_4",
        "_tmp_5",
        "_tmp_6",
        "_tmp_7",
    ):
        header = f"  for (int {tmp} = 0; {tmp} < (({lanes_limit} - tid) >> {shift}); ++{tmp}) {{"
        if header not in msl:
            continue
        msl = msl.replace(header, f"  for (int k = tid; k < {topk}; k += threads) {{", 1)
        msl = msl.replace(f"(({tmp} * {threads}) + tid)", "k")
        msl = msl.replace(f"({tmp} * {threads})", "(k - tid)")

    qk_limit = qk_dim + threads - 1
    d_header = f"  for (int _tmp_8 = 0; _tmp_8 < (({qk_limit} - tid) >> {shift}); ++_tmp_8) {{"
    if d_header in msl:
        msl = msl.replace(d_header, f"  for (int d = tid; d < {qk_dim}; d += threads) {{", 1)
        msl = msl.replace(f"(_tmp_8 * {threads})", "(d - tid)")
        msl = msl.replace(f"((_tmp_8 * {threads}) + tid)", "d")

    kd_limit = topk * qk_dim + threads - 1
    kd_header = f"  for (int _tmp_9 = 0; _tmp_9 < (({kd_limit} - tid) >> {shift}); ++_tmp_9) {{"
    if kd_header in msl:
        msl = msl.replace(
            kd_header,
            f"  for (int kd = tid; kd < {topk * qk_dim}; kd += threads) {{",
            1,
        )
        msl = msl.replace(f"(_tmp_9 * {threads})", "(kd - tid)")
        msl = msl.replace(f"((_tmp_9 * {threads}) + tid)", "kd")

    msl = re.sub(r"(?P<prefix>\+\s*)\(k - tid\)\) \+ tid\)", r"\g<prefix>k)", msl)
    msl = re.sub(r"(?P<prefix>\+\s*)\(d - tid\)\) \+ tid\)", r"\g<prefix>d)", msl)
    msl = re.sub(r"(?P<prefix>\+\s*)\(kd - tid\)\) \+ tid\)", r"\g<prefix>kd)", msl)
    msl = msl.replace(f"(kd - tid) / {threads}", f"(kd / {threads})")
    msl = msl.replace(f"(kd - tid) % {threads}", f"(kd % {threads})")

    if d_v == qk_dim:
        msl = msl.replace(f"(kd / {qk_dim // threads})", f"(kd / {qk_dim})")
        msl = msl.replace(f"(kd & {qk_dim // threads - 1})", f"((kd % {qk_dim}) / {threads})")
    elif qk_dim % threads == 0:
        chunks = qk_dim // threads
        msl = msl.replace(f"(kd / {chunks})", f"(kd / {qk_dim})")
        msl = msl.replace(f"(kd % {chunks})", f"((kd % {qk_dim}) / {threads})")

    msl = re.sub(r"indices\[\({2,}\(gid >> (?P<shift>\d+)\) \* (?P<mult>\d+)\) \+ k\)\]", r"indices[((gid >> \g<shift>) * \g<mult>) + k]", msl)
    msl = re.sub(r"indices\[\({2,}\(gid >> (?P<shift>\d+)\) \* (?P<mult>\d+)\) \+ \(kd / (?P<dim>\d+)\)\)\]", r"indices[((gid >> \g<shift>) * \g<mult>) + (kd / \g<dim>)]", msl)
    msl = re.sub(r"q\[\({2,}\(gid \* (?P<dim>\d+)\) \+ d\)\]", r"q[(gid * \g<dim>) + d]", msl)
    msl = re.sub(r"q\[\({2,}\(gid \* (?P<dim>\d+)\) \+ \(kd % (?P=dim)\)\)\]", r"q[(gid * \g<dim>) + (kd % \g<dim>)]", msl)
    msl = re.sub(r"d_out\[\({2,}\(gid \* (?P<dim>\d+)\) \+ d_1\)\]", r"d_out[(gid * \g<dim>) + d_1]", msl)
    msl = re.sub(r"d_out\[\({2,}\(gid \* (?P<dim>\d+)\) \+ \(kd % (?P=dim)\)\)\]", r"d_out[(gid * \g<dim>) + (kd % \g<dim>)]", msl)
    msl = re.sub(r"dq\[\({2,}\(gid \* (?P<dim>\d+)\) \+ d\)\]", r"dq[(gid * \g<dim>) + d]", msl)
    msl = re.sub(r"dkv_partial\[\({2,}\(gid \* (?P<stride>\d+)\) \+ kd\)\]", r"dkv_partial[(gid * \g<stride>) + kd]", msl)

    # Drop tautological aliases (cf. fwd path) — ``int k = k;``, ``int kd = ...``,
    # ``int d_N = d;``. The outer ``for (uint k = tid; ...)`` already owns these.
    msl = re.sub(r"(?m)^[ \t]*int k = k;\n", "", msl)
    msl = re.sub(r"(?m)^[ \t]*int k_\d+ = k;\n", "", msl)
    msl = re.sub(r"(?m)^[ \t]*int kd = \(\(kd - tid\) \+ tid\);\n", "", msl)
    msl = re.sub(r"(?m)^[ \t]*int kd_\d+ = kd;\n", "", msl)
    msl = re.sub(r"(?m)^[ \t]*int d_\d+ = d;\n", "", msl)
    msl = re.sub(r"(?m)^[ \t]*int d = d;\n", "", msl)

    return msl


def _canonicalize_bwd_residual_tmp_indexing(
    msl: str,
    *,
    threads: int,
    qk_dim: int,
    d_v: int,
) -> str:
    """Rewrite TileLang loop-index remnants after lane-loop canonicalization."""

    chunks = qk_dim // threads
    d_expr = f"(kd % {qk_dim})"
    k_expr = f"(kd / {qk_dim})"

    msl = re.sub(
        r"kv\[\(\(\(\(\(long\)_tmp_8\) \* \(long\)"
        + str(threads)
        + r"\) \+ \(\(long\)(?P<base>kv_row_base_2)\)\) \+ \(\(long\)tid\)\)\]",
        r"kv[\g<base> + d]",
        msl,
    )

    if chunks & (chunks - 1) == 0:
        msl = msl.replace(f"(_tmp_9 >> {chunks.bit_length() - 1})", k_expr)
    msl = msl.replace(f"(_tmp_9 / {chunks})", k_expr)
    msl = msl.replace(f"(_tmp_9 & {chunks - 1})", f"({d_expr} / {threads})")
    msl = msl.replace(f"(_tmp_9 % {chunks})", f"({d_expr} / {threads})")
    if chunks == 1:
        msl = msl.replace("_tmp_9", k_expr)
        msl = re.sub(
            r"q\[\(\(gid \* " + str(qk_dim) + r"\) \+ tid\)\]",
            f"q[((gid * {qk_dim}) + {d_expr})]",
            msl,
        )
        msl = re.sub(
            r"d_out\[\(\(gid \* " + str(d_v) + r"\) \+ tid\)\]",
            f"d_out[((gid * {d_v}) + {d_expr})]",
            msl,
        )
    msl = re.sub(
        r"\(\(\((?P<base>gid \* \d+)\) \+ \(\((?:kd % \d+)\) / "
        + str(threads)
        + r"\) \* "
        + str(threads)
        + r"\)\) \+ tid\)",
        r"((\g<base>) + " + d_expr + r")",
        msl,
    )
    msl = re.sub(
        r"\(\(\((?P<base>gid \* \d+)\) \+ \(\(\(kd % "
        + str(qk_dim)
        + r"\) / "
        + str(threads)
        + r"\) \* "
        + str(threads)
        + r"\)\) \+ tid\)",
        r"((\g<base>) + " + d_expr + r")",
        msl,
    )
    msl = re.sub(
        r"indices\[\({3}(?P<base>gid >> \d+)\) \* (?P<topk>\d+)\) \+ "
        r"\(kd / "
        + str(qk_dim)
        + r"\)\)\]",
        r"indices[((\g<base>) * \g<topk>) + " + k_expr + r"]",
        msl,
    )

    if d_v == qk_dim:
        msl = re.sub(r"(?m)^[ \t]*if \(\(kd % " + str(qk_dim) + r"\) < " + str(d_v) + r"\) \{\n", "", msl)
        msl = re.sub(r"(?m)^[ \t]*\} else \{\n[ \t]*dkv_partial\[[^\n]+] = \(\(half\)acc\);\n[ \t]*\}\n", "", msl, count=1)
    else:
        msl = msl.replace(f"(({d_expr} / {threads}) < {d_v // threads})", f"({d_expr} < {d_v})")

    return msl


def _canonicalize_bwd_reductions(msl: str, *, threads: int) -> str:
    """Use Path-B-style stride reductions for backward max/sum/rowsum."""

    log_threads = threads.bit_length() - 1
    barrier = r"(?:metal::)?threadgroup_barrier\(metal::mem_flags::mem_threadgroup\);"
    msl = msl.replace("reduce_buf[(tid + stride)]", "reduce_buf[tid + stride]")
    max_reduction_re = re.compile(
        rf"  for \(int round_id = 0; round_id < {log_threads}; \+\+round_id\) \{{\n"
        rf"    stride = \({threads} >> \(round_id \+ 1\)\);\n"
        r"    if \(tid < stride\) \{\n"
        r"      if \(reduce_buf\[tid\] < reduce_buf\[tid \+ stride\]\) \{\n"
        r"        reduce_buf\[tid\] = reduce_buf\[tid \+ stride\];\n"
        r"      \}\n"
        r"    \}\n"
        rf"    {barrier}\n"
        r"  \}"
    )
    max_replacement = (
        "  for (int stride = threads / 2; stride > 0; stride >>= 1) {\n"
        "    if (tid < stride) {\n"
        "      float a = reduce_buf[tid];\n"
        "      float b_v = reduce_buf[tid + stride];\n"
        "      if (b_v > a) reduce_buf[tid] = b_v;\n"
        "    }\n"
        "    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);\n"
        "  }"
    )
    msl = max_reduction_re.sub(max_replacement, msl, count=1)

    for name in ("round_id_1", "round_id_2"):
        sum_reduction_re = re.compile(
            rf"  for \(int {name} = 0; {name} < {log_threads}; \+\+{name}\) \{{\n"
            rf"    stride = \({threads} >> \({name} \+ 1\)\);\n"
            r"    if \(tid < stride\) \{\n"
            r"      reduce_buf\[tid\] (?:= \(reduce_buf\[tid\] \+ reduce_buf\[tid \+ stride\]\)|\+= reduce_buf\[tid \+ stride\]);\n"
            r"    \}\n"
            rf"    {barrier}\n"
            r"  \}"
        )
        sum_replacement = (
            "  for (int stride = threads / 2; stride > 0; stride >>= 1) {\n"
            "    if (tid < stride) {\n"
            "      reduce_buf[tid] += reduce_buf[tid + stride];\n"
            "    }\n"
            "    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);\n"
            "  }"
        )
        msl = sum_reduction_re.sub(sum_replacement, msl, count=1)

    max_reduction = (
        f"  for (int round_id = 0; round_id < {log_threads}; ++round_id) {{\n"
        f"    stride = ({threads} >> (round_id + 1));\n"
        "    if (tid < stride) {\n"
        "      if (reduce_buf[tid] < reduce_buf[tid + stride]) {\n"
        "        reduce_buf[tid] = reduce_buf[tid + stride];\n"
        "      }\n"
        "    }\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "  }"
    )
    max_replacement = (
        "  for (int stride = threads / 2; stride > 0; stride >>= 1) {\n"
        "    if (tid < stride) {\n"
        "      float a = reduce_buf[tid];\n"
        "      float b_v = reduce_buf[tid + stride];\n"
        "      if (b_v > a) reduce_buf[tid] = b_v;\n"
        "    }\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "  }"
    )
    msl = msl.replace(max_reduction, max_replacement, 1)

    for name in ("round_id_1", "round_id_2"):
        sum_reduction = (
            f"  for (int {name} = 0; {name} < {log_threads}; ++{name}) {{\n"
            f"    stride = ({threads} >> ({name} + 1));\n"
            "    if (tid < stride) {\n"
            "      reduce_buf[tid] = (reduce_buf[tid] + reduce_buf[tid + stride]);\n"
            "    }\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "  }"
        )
        sum_replacement = (
            "  for (int stride = threads / 2; stride > 0; stride >>= 1) {\n"
            "    if (tid < stride) {\n"
            "      reduce_buf[tid] = (reduce_buf[tid] + reduce_buf[tid + stride]);\n"
            "    }\n"
            "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "  }"
        )
        msl = msl.replace(sum_reduction, sum_replacement, 1)

    return msl.replace(
        "  inv_sum = 0.000000e+00f;\n"
        "  if (0.000000e+00f < sumexp) {\n"
        "    inv_sum = (1.000000e+00f / sumexp);\n"
        "  }",
        "  inv_sum = (sumexp > 0.000000e+00f) ? (1.000000e+00f / sumexp) : 0.000000e+00f;",
        1,
    )


def _insert_backward_all_masked_fast_return(
    msl: str,
    *,
    qk_dim: int,
    topk: int,
) -> str:
    """Match Path B's zero-output fast path for all-masked backward rows."""

    marker = "  float sumexp = reduce_buf[0];\n"
    if marker not in msl or "sumexp <= 0.000000e+00f" in msl:
        return msl
    fast_return = (
        marker
        + "  if (sumexp <= 0.000000e+00f) {\n"
        + f"    for (int d = tid; d < {qk_dim}; d += threads) {{\n"
        + f"      dq[(gid * {qk_dim}) + d] = ((half)0.000000e+00f);\n"
        + "    }\n"
        + f"    for (int kd = tid; kd < {topk * qk_dim}; kd += threads) {{\n"
        + f"      dkv_partial[(gid * {topk * qk_dim}) + kd] = 0.000000e+00h;\n"
        + "    }\n"
        + "    return;\n"
        + "  }\n"
    )
    return msl.replace(marker, fast_return, 1)


def _canonicalize_bwd_hot_loops(msl: str, *, qk_dim: int, d_v: int) -> str:
    """Remove residual TileLang scalarization overhead from backward MSL."""

    msl = msl.replace("  int stride;\n", "")
    msl = re.sub(r"(?m)^(?P<indent>[ \t]*)half (?P<var>condval(?:_\d+)?) = (?P<load>kv\[[^\n]+]);\n(?P=indent)(?P<target>acc = [^\n]*\(\(float\)(?P=var)\)[^\n]*;)", lambda match: f"{match.group('indent')}{match.group('target').replace('((float)' + match.group('var') + ')', '((float)' + match.group('load') + ')')}", msl)
    msl = re.sub(r"kv\[\({5}long\)d\) \+ \(\(long\)kv_row_base_2\)\) \+ \(\(long\)tid\)\)\]", "kv[kv_row_base_2 + d]", msl)
    msl = re.sub(r"kv\[\({2,}\(long\)kv_row_base_1\) \+ \(\(long\)d_1\)\)\]", "kv[kv_row_base_1 + d_1]", msl)

    msl = re.sub(
        r"(?P<indent>[ \t]*)if \(gather_idx < 0\) \{\n"
        r"(?P=indent)  scores\[k\] = -INFINITY;\n"
        r"(?P=indent)\} else \{\n"
        r"(?P=indent)  acc = 0\.000000e\+00f;\n"
        r"(?P<body>"
        r"(?P=indent)  (?:int|uint) kv_row_base = [^\n]+;\n"
        r"(?P=indent)  for \((?:int|uint) d = 0; d < \d+; \+\+d\) \{\n"
        r"(?P=indent)    acc = [^\n]+;\n"
        r"(?P=indent)  \}\n)"
        r"(?P=indent)  scores\[k\] = \(acc \* sm_scale\);\n"
        r"(?P=indent)\}",
        (
            r"\g<indent>if (gather_idx < 0) {\n"
            r"\g<indent>  scores[k] = -INFINITY;\n"
            r"\g<indent>  continue;\n"
            r"\g<indent>}\n"
            r"\g<indent>acc = 0.000000e+00f;\n"
            r"\g<body>"
            r"\g<indent>scores[k] = (acc * sm_scale);"
        ),
        msl,
        count=1,
    )
    msl = re.sub(
        r"(?P<indent>[ \t]*)if \(gather_idx < 0\) \{\n"
        r"(?P=indent)  dp\[k\] = 0\.000000e\+00f;\n"
        r"(?P=indent)\} else \{\n"
        r"(?P=indent)  acc = 0\.000000e\+00f;\n"
        r"(?P<body>"
        r"(?P=indent)  (?:int|uint) kv_row_base_1 = [^\n]+;\n"
        r"(?P=indent)  for \((?:int|uint) d_1 = 0; d_1 < \d+; \+\+d_1\) \{\n"
        r"(?P=indent)    acc = [^\n]+;\n"
        r"(?P=indent)  \}\n)"
        r"(?P=indent)  dp\[k\] = acc;\n"
        r"(?P=indent)\}",
        (
            r"\g<indent>if (gather_idx < 0) {\n"
            r"\g<indent>  dp[k] = 0.000000e+00f;\n"
            r"\g<indent>  continue;\n"
            r"\g<indent>}\n"
            r"\g<indent>acc = 0.000000e+00f;\n"
            r"\g<body>"
            r"\g<indent>dp[k] = acc;"
        ),
        msl,
        count=1,
    )
    msl = re.sub(
        r"(?P<indent>[ \t]*)if \(0 <= gather_idx\) \{\n"
        r"(?P=indent)  (?P<base>(?:int|uint) kv_row_base_2 = [^\n]+;\n)"
        r"(?P=indent)  acc = \(acc \+ \(ds\[k\] \* \(\(float\)(?P<load>kv\[[^\n]+])\)\)\);\n"
        r"(?P=indent)\}",
        (
            r"\g<indent>if (gather_idx < 0) {\n"
            r"\g<indent>  continue;\n"
            r"\g<indent>}\n"
            r"\g<indent>\g<base>"
            r"\g<indent>acc = (acc + (ds[k] * ((float)\g<load>)));"
        ),
        msl,
        count=1,
    )
    msl = re.sub(
        r"(?P<indent>[ \t]*)if \(gather_idx < 0\) \{\n"
        r"(?P=indent)  dkv_partial\[(?P<out>[^\n]+)\] = 0\.000000e\+00h;\n"
        r"(?P=indent)\} else \{\n"
        r"(?P=indent)  acc = (?P<acc>[^\n]+);\n"
        r"(?P=indent)  dkv_partial\[(?P=out)\] = \(\(half\)(?P<value>[^\n]+)\);\n"
        r"(?P=indent)\}",
        (
            r"\g<indent>if (gather_idx < 0) {\n"
            r"\g<indent>  dkv_partial[\g<out>] = 0.000000e+00h;\n"
            r"\g<indent>  continue;\n"
            r"\g<indent>}\n"
            r"\g<indent>acc = \g<acc>;\n"
            r"\g<indent>dkv_partial[\g<out>] = ((half)\g<value>);"
        ),
        msl,
        count=1,
    )

    msl = re.sub(r"\(gather_idx \* (?P<scale>\d+)\)", r"(uint(gather_idx) * \g<scale>)", msl)
    msl = re.sub(r"\bint (?P<name>kv_row_base(?:_\d+)?) = (?P<expr>[^\n]*gather_idx[^\n]*);", r"uint \g<name> = \g<expr>;", msl)
    if d_v == qk_dim:
        msl = msl.replace("if ((kd % 64) < 64) {\n", "")
    return _canonicalize_bwd_dkv_kd_loop(msl, qk_dim=qk_dim, d_v=d_v)


def _canonicalize_bwd_dkv_kd_loop(msl: str, *, qk_dim: int, d_v: int) -> str:
    """Hoist kd-derived k/d locals in the dKV loop to match Path B's MSL."""

    dtype = r"(?:int|uint)"
    common = (
        r"(?P<indent>[ \t]*)for \("
        + dtype
        + r" kd = tid; kd < (?P<limit>\d+); kd \+= threads\) \{\n"
        r"(?P=indent)  gather_idx = indices\[(?P<idx_base>[^\]]+) \+ "
        r"\(kd / "
        + str(qk_dim)
        + r"\)\];\n"
        r"(?P=indent)  if \(gather_idx < 0\) \{\n"
        r"(?P=indent)    dkv_partial\[(?P<out>[^\]]+)\] = 0\.000000e\+00h;\n"
    )

    full_dim = re.compile(
        common
        + r"(?P=indent)    continue;\n"
        r"(?P=indent)  \}\n"
        r"(?P=indent)  acc = \(\(sm_scale \* ds\[\(kd / "
        + str(qk_dim)
        + r"\)\]\) \* \(\(float\)q\[\(\(gid \* "
        + str(qk_dim)
        + r"\) \+ \(kd % "
        + str(qk_dim)
        + r"\)\)\]\)\);\n"
        r"(?P=indent)  dkv_partial\[(?P=out)\] = \(\(half\)\(\(p\[\(kd / "
        + str(qk_dim)
        + r"\)\] \* \(\(float\)d_out\[\(\(gid \* "
        + str(d_v)
        + r"\) \+ \(kd % "
        + str(qk_dim)
        + r"\)\)\]\)\) \+ acc\)\);\n"
        r"(?P=indent)\}",
        re.MULTILINE,
    )

    def replace_full_dim(match: re.Match[str]) -> str:
        indent = match.group("indent")
        limit = match.group("limit")
        idx_base = match.group("idx_base")
        out = match.group("out")
        return (
            f"{indent}for (uint kd = tid; kd < {limit}; kd += threads) {{\n"
            f"{indent}  uint k = kd / {qk_dim};\n"
            f"{indent}  uint d = kd % {qk_dim};\n"
            f"{indent}  gather_idx = indices[{idx_base} + k];\n"
            f"{indent}  if (gather_idx < 0) {{\n"
            f"{indent}    dkv_partial[{out}] = 0.000000e+00h;\n"
            f"{indent}    continue;\n"
            f"{indent}  }}\n"
            f"{indent}  float qv = float(q[(gid * {qk_dim}) + d]);\n"
            f"{indent}  float ks_q = sm_scale * ds[k] * qv;\n"
            f"{indent}  float dod = float(d_out[(gid * {d_v}) + d]);\n"
            f"{indent}  dkv_partial[{out}] = ((half)((p[k] * dod) + ks_q));\n"
            f"{indent}}}"
        )

    msl = full_dim.sub(replace_full_dim, msl, count=1)

    tail_dim = re.compile(
        common
        + r"(?P=indent)  \} else \{\n"
        r"(?P=indent)    acc = \(\(sm_scale \* ds\[\(kd / "
        + str(qk_dim)
        + r"\)\]\) \* \(\(float\)q\[\(\(gid \* "
        + str(qk_dim)
        + r"\) \+ \(kd % "
        + str(qk_dim)
        + r"\)\)\]\)\);\n"
        r"(?P=indent)    if \(\(kd % "
        + str(qk_dim)
        + r"\) < "
        + str(d_v)
        + r"\) \{\n"
        r"(?P=indent)      dkv_partial\[(?P=out)\] = \(\(half\)\(\(p\[\(kd / "
        + str(qk_dim)
        + r"\)\] \* \(\(float\)d_out\[\(\(gid \* "
        + str(d_v)
        + r"\) \+ \(kd % "
        + str(qk_dim)
        + r"\)\)\]\)\) \+ acc\)\);\n"
        r"(?P=indent)    \} else \{\n"
        r"(?P=indent)      dkv_partial\[(?P=out)\] = \(\(half\)acc\);\n"
        r"(?P=indent)    \}\n"
        r"(?P=indent)  \}\n"
        r"(?P=indent)\}",
        re.MULTILINE,
    )

    def replace_tail_dim(match: re.Match[str]) -> str:
        indent = match.group("indent")
        limit = match.group("limit")
        idx_base = match.group("idx_base")
        out = match.group("out")
        return (
            f"{indent}for (uint kd = tid; kd < {limit}; kd += threads) {{\n"
            f"{indent}  uint k = kd / {qk_dim};\n"
            f"{indent}  uint d = kd % {qk_dim};\n"
            f"{indent}  gather_idx = indices[{idx_base} + k];\n"
            f"{indent}  if (gather_idx < 0) {{\n"
            f"{indent}    dkv_partial[{out}] = 0.000000e+00h;\n"
            f"{indent}    continue;\n"
            f"{indent}  }}\n"
            f"{indent}  float qv = float(q[(gid * {qk_dim}) + d]);\n"
            f"{indent}  float ks_q = sm_scale * ds[k] * qv;\n"
            f"{indent}  if (d < {d_v}) {{\n"
            f"{indent}    float dod = float(d_out[(gid * {d_v}) + d]);\n"
            f"{indent}    dkv_partial[{out}] = ((half)((p[k] * dod) + ks_q));\n"
            f"{indent}  }} else {{\n"
            f"{indent}    dkv_partial[{out}] = ((half)ks_q);\n"
            f"{indent}  }}\n"
            f"{indent}}}"
        )

    return tail_dim.sub(replace_tail_dim, msl, count=1)


def _canonicalize_bwd_base_indexing(
    msl: str,
    *,
    heads: int,
    seq_len: int,
    qk_dim: int,
    kv_group: int,
    head_kv: int,
    topk: int,
    seq_len_kv: int,
    d_v: int,
) -> str:
    """Hoist backward row bases so hot-loop addressing mirrors Path B."""

    marker = "  float sm_scale = sm_scale_buf[0];"
    if marker in msl and "  uint q_row_base =" not in msl:
        base_locals = (
            f"  uint h = gid % {heads};\n"
            f"  uint s = (gid / {heads}) % {seq_len};\n"
            f"  uint b = gid / {heads * seq_len};\n"
            f"  uint kv_group = {kv_group};\n"
            f"  uint head_kv = {head_kv};\n"
            "  uint g = h / head_kv;\n"
            f"  uint qk_dim = {qk_dim};\n"
            f"  uint d_v = {d_v};\n"
            f"  uint q_row_base = ((b * {seq_len} + s) * {heads} + h) * qk_dim;\n"
            f"  uint d_out_row = ((b * {seq_len} + s) * {heads} + h) * d_v;\n"
            f"  uint kv_outer_stride = {seq_len_kv} * kv_group * qk_dim;\n"
            "  uint kv_b_base = b * kv_outer_stride;\n"
            f"  uint idx_base = ((b * {seq_len} + s) * kv_group + g) * {topk};\n"
            f"  uint dkv_pb = (((b * {seq_len} + s) * {heads} + h) * {topk}) * qk_dim;\n"
        )
        msl = msl.replace(marker, base_locals + marker, 1)

    msl = re.sub(
        r"indices\[(?P<expr>[^\]\n]*gid[^\]\n]*) \+ k\]",
        "indices[idx_base + k]",
        msl,
    )
    msl = re.sub(
        r"indices\[(?P<expr>[^\]\n]*gid[^\]\n]*) \+ \(kd / " + str(qk_dim) + r"\)\]",
        "indices[idx_base + (kd / " + str(qk_dim) + ")]",
        msl,
    )
    for match in list(
        re.finditer(
            r"(?m)^  int (?P<name>cse_v\d+(?:_\d+)?) = "
            r"\(\(gid >> \d+\) \* " + str(topk) + r"\);\n",
            msl,
        )
    ):
        msl = msl.replace(match.group(0), "")
        msl = re.sub(rf"\b{re.escape(match.group('name'))}\b", "idx_base", msl)
    msl = re.sub(
        r"indices\[\(\(idx_base \+ cse_v\d+(?:_\d+)?\) \+ tid\)\]",
        "indices[idx_base + k]",
        msl,
    )
    msl = msl.replace("indices[(idx_base + k)]", "indices[idx_base + k]")
    msl = re.sub(
        r"uint (?P<name>kv_row_base(?:_\d+)?) = [^\n;]*uint\(gather_idx\)[^\n;]*;",
        r"uint \g<name> = kv_b_base + (uint(gather_idx) * kv_group + g) * qk_dim;",
        msl,
    )
    msl = re.sub(
        r"(?m)^(?P<indent>[ \t]*)uint kv_row_base(?:_\d+)? = "
        r"kv_b_base \+ \(uint\(gather_idx\) \* kv_group \+ g\) \* qk_dim;",
        r"\g<indent>uint kv_row_base = kv_b_base + (uint(gather_idx) * kv_group + g) * qk_dim;",
        msl,
    )
    msl = re.sub(r"\bkv_row_base_\d+\b", "kv_row_base", msl)
    msl = re.sub(
        r"q\[\({1,2}gid \* " + str(qk_dim) + r"\) \+ d\)?\]",
        "q[q_row_base + d]",
        msl,
    )
    msl = re.sub(
        r"q\[\(gid \* " + str(qk_dim) + r"\) \+ \(kd % " + str(qk_dim) + r"\)\]",
        "q[q_row_base + (kd % " + str(qk_dim) + ")]",
        msl,
    )
    msl = re.sub(
        r"d_out\[\({1,2}gid \* " + str(d_v) + r"\) \+ d\)?\]",
        "d_out[d_out_row + d]",
        msl,
    )
    msl = re.sub(
        r"d_out\[\({1,2}gid \* " + str(d_v) + r"\) \+ d_1\)?\]",
        "d_out[d_out_row + d_1]",
        msl,
    )
    msl = re.sub(
        r"d_out\[\(gid \* " + str(d_v) + r"\) \+ \(kd % " + str(qk_dim) + r"\)\]",
        "d_out[d_out_row + (kd % " + str(qk_dim) + ")]",
        msl,
    )
    msl = re.sub(r"dq\[\(gid \* " + str(qk_dim) + r"\) \+ d\]", "dq[q_row_base + d]", msl)
    msl = re.sub(
        r"dkv_partial\[\(gid \* " + str(topk * qk_dim) + r"\) \+ (?P<off>kd|k_off)\]",
        r"dkv_partial[dkv_pb + \g<off>]",
        msl,
    )
    msl = re.sub(
        r"(?m)^(?P<indent>[ \t]*)gather_idx = (?P<idx>indices\[idx_base \+ k\]);",
        r"\g<indent>int gather_idx = \g<idx>;",
        msl,
    )
    # Bwd output-projection / dq paths leave bare ``gather_idx = indices[...]``
    # assignments where the original ``thread int gather_idx[1]`` decl has been
    # stripped. Promote them to declarations so the symbol is in scope.
    msl = re.sub(
        r"(?m)^(?P<indent>[ \t]*)gather_idx = "
        r"(?P<idx>indices\[\(\(\(gid >> \d+\) \* \d+\) \+ k_\d+\)\]);",
        r"\g<indent>int gather_idx = \g<idx>;",
        msl,
    )
    msl = re.sub(
        r"(?m)^(?P<indent>[ \t]*)gather_idx = "
        r"(?P<idx>indices\[idx_base \+ \(kd / \d+\)\]);",
        r"\g<indent>int gather_idx = \g<idx>;",
        msl,
    )
    msl = re.sub(
        r"(?m)^(?P<indent>[ \t]*)gather_idx = "
        r"(?P<idx>indices\[[^\n;]+\]);",
        r"\g<indent>int gather_idx = \g<idx>;",
        msl,
    )
    msl = _canonicalize_bwd_cse_indexing(
        msl,
        qk_dim=qk_dim,
        d_v=d_v,
        topk=topk,
        threads=_threadgroup_size(topk),
    )
    return msl


def _canonicalize_bwd_cse_indexing(
    msl: str,
    *,
    qk_dim: int,
    d_v: int,
    topk: int,
    threads: int,
) -> str:
    """Rewrite z3/CSE address temporaries after backward base hoisting."""

    cse_q_bases = set(
        re.findall(rf"(?m)^  int (cse_v\d+(?:_\d+)?) = \(gid \* {qk_dim}\);\n", msl)
    )
    cse_dout_bases = set(
        re.findall(rf"(?m)^  int (cse_v\d+(?:_\d+)?) = \(gid \* {d_v}\);\n", msl)
    )
    cse_idx_bases = set(
        re.findall(rf"(?m)^  int (cse_v\d+(?:_\d+)?) = \(\(gid >> \d+\) \* {topk}\);\n", msl)
    )

    for name in cse_idx_bases:
        msl = re.sub(
            rf"indices\[\(\({name} \+ (?P<off>[^)\]]+)\)\)\]",
            r"indices[idx_base + \g<off>]",
            msl,
        )
        msl = re.sub(
            rf"indices\[\({name} \+ (?P<off>[^)\]]+)\)\]",
            r"indices[idx_base + \g<off>]",
            msl,
        )

    for delta in re.findall(r"(?m)^[ \t]*int (cse_v\d+(?:_\d+)?) = \(k - tid\);\n", msl):
        msl = re.sub(
            rf"\(\((?P<base>[^()\n]+) \+ {delta}\) \+ tid\)",
            r"(\g<base> + k)",
            msl,
        )
        msl = msl.replace(f"({delta} + tid)", "k")
        msl = re.sub(rf"(?m)^[ \t]*int {delta} = \(k - tid\);\n", "", msl)

    for delta in re.findall(r"(?m)^[ \t]*int (cse_v\d+(?:_\d+)?) = \(d - tid\);\n", msl):
        msl = re.sub(
            rf"\(\((?P<base>[^()\n]+) \+ {delta}\) \+ tid\)",
            r"(\g<base> + d)",
            msl,
        )
        msl = re.sub(
            rf"\(\({delta} \+ (?P<base>[^()\n]+)\) \+ tid\)",
            r"(\g<base> + d)",
            msl,
        )
        msl = msl.replace(f"({delta} + tid)", "d")
        msl = re.sub(rf"(?m)^[ \t]*int {delta} = \(d - tid\);\n", "", msl)

    for name in cse_idx_bases:
        msl = msl.replace(f"indices[{name} + k]", "indices[idx_base + k]")
        msl = re.sub(rf"indices\[\({name} \+ (?P<off>[^)\]]+)\)\]", r"indices[idx_base + \g<off>]", msl)
    for name in cse_q_bases:
        msl = re.sub(rf"\bq\[\({name} \+ (?P<off>[^)\]]+)\)\]", r"q[q_row_base + \g<off>]", msl)
        msl = re.sub(rf"\bdq\[\({name} \+ (?P<off>[^)\]]+)\)\]", r"dq[q_row_base + \g<off>]", msl)
        if d_v == qk_dim:
            msl = re.sub(
                rf"\bd_out\[\({name} \+ (?P<off>[^)\]]+)\)\]",
                r"d_out[d_out_row + \g<off>]",
                msl,
            )
    for name in cse_dout_bases:
        msl = re.sub(
            rf"\bd_out\[\({name} \+ (?P<off>[^)\]]+)\)\]",
            r"d_out[d_out_row + \g<off>]",
            msl,
        )

    for name, off in re.findall(
        r"(?m)^[ \t]*int (cse_v\d+(?:_\d+)?) = \(kv_row_base \+ (d(?:_1)?)\);\n",
        msl,
    ):
        msl = msl.replace(f"kv[{name}]", f"kv[kv_row_base + {off}]")
        msl = re.sub(rf"(?m)^[ \t]*int {name} = \(kv_row_base \+ {off}\);\n", "", msl)

    kd_k_names = re.findall(
        rf"(?m)^[ \t]*int (cse_v\d+(?:_\d+)?) = \(kd / {qk_dim}\);\n",
        msl,
    )
    msl = re.sub(
        rf"(?m)^(?P<indent>[ \t]*)int (?P<name>cse_v\d+(?:_\d+)?) = \(kd / {qk_dim}\);\n",
        rf"\g<indent>uint k = kd / {qk_dim};\n\g<indent>uint d = kd % {qk_dim};\n",
        msl,
    )
    for name in kd_k_names:
        msl = msl.replace(f"ds[{name}]", "ds[k]")
        msl = msl.replace(f"p[{name}]", "p[k]")
        msl = msl.replace(f"indices[(idx_base + {name})]", "indices[idx_base + k]")
        for idx_base in cse_idx_bases:
            msl = msl.replace(f"indices[({idx_base} + {name})]", "indices[idx_base + k]")

    kd_lane = rf"\(\(\(kd % {qk_dim}\) / {threads}\) \* {threads}\)"
    for name in cse_q_bases:
        msl = msl.replace(f"q[((({name} + {kd_lane}) + tid))]", "q[q_row_base + d]")
        msl = msl.replace(f"q[((({name} + {kd_lane})) + tid)]", "q[q_row_base + d]")
        msl = msl.replace(f"q[(({name} + {kd_lane}) + tid)]", "q[q_row_base + d]")
        if d_v == qk_dim:
            msl = msl.replace(
                f"d_out[((({name} + {kd_lane}) + tid))]",
                "d_out[d_out_row + d]",
            )
            msl = msl.replace(
                f"d_out[(({name} + {kd_lane}) + tid)]",
                "d_out[d_out_row + d]",
            )
    for name in cse_dout_bases:
        msl = msl.replace(
            f"d_out[(({name} + {kd_lane}) + tid)]",
            "d_out[d_out_row + d]",
        )

    msl = re.sub(
        rf"\bq\[\(\((?:cse_v\d+(?:_\d+)? \+ )?{kd_lane}\) \+ tid\)\]",
        "q[q_row_base + d]",
        msl,
    )
    msl = re.sub(
        rf"\bd_out\[\(\((?:cse_v\d+(?:_\d+)? \+ )?{kd_lane}\) \+ tid\)\]",
        "d_out[d_out_row + d]",
        msl,
    )
    msl = msl.replace("dkv_partial[dkv_partial_base + kd]", "dkv_partial[dkv_pb + kd]")

    for name in cse_q_bases | cse_dout_bases | cse_idx_bases:
        if re.search(rf"\b{name}\b", msl) is None:
            continue
        decl_re = rf"(?m)^  int {name} = [^\n]+;\n"
        body_without_decl = re.sub(decl_re, "", msl)
        if re.search(rf"\b{name}\b", body_without_decl) is None:
            msl = body_without_decl
    for name in re.findall(r"(?m)^  int (cse_v\d+(?:_\d+)?) = [^\n]+;\n", msl):
        decl_re = rf"(?m)^  int {name} = [^\n]+;\n"
        body_without_decl = re.sub(decl_re, "", msl)
        if re.search(rf"\b{name}\b", body_without_decl) is None:
            msl = body_without_decl
    return msl


def _canonicalize_bwd_path_b_hot_loops(
    msl: str,
    *,
    qk_dim: int,
    d_v: int,
    topk: int,
) -> str:
    """Make backward hot loops match the hand-written Path B MSL shape."""

    threads = _threadgroup_size(topk)
    kv_base = "uint kv_row_base = kv_b_base + (uint(gather_idx) * kv_group + g) * qk_dim;"
    msl = msl.replace(
        (
            "    acc = 0.000000e+00f;\n"
            f"      {kv_base}\n"
            f"      for (uint d = 0; d < {qk_dim}; ++d) {{\n"
            "        acc = (acc + (((float)q[q_row_base + d]) * ((float)kv[kv_row_base + d])));\n"
            "      }\n"
            "    scores[k] = (acc * sm_scale);"
        ),
        (
            f"    {kv_base}\n"
            "    float acc = 0.0f;\n"
            f"    for (uint d = 0; d < {qk_dim}; ++d) {{\n"
            "      float qv = float(q[q_row_base + d]);\n"
            "      float kv_v = float(kv[kv_row_base + d]);\n"
            "      acc += qv * kv_v;\n"
            "    }\n"
            "    scores[k] = acc * sm_scale;"
        ),
        1,
    )
    msl = msl.replace(
        (
            "  local = -INFINITY;\n"
            f"  for (uint k = tid; k < {topk}; k += threads) {{\n"
            "    if (local < scores[k]) {\n"
            "      local = scores[k];\n"
            "    }\n"
            "  }\n"
            "  reduce_buf[tid] = local;"
        ),
        (
            "  float local_max = -INFINITY;\n"
            f"  for (uint k = tid; k < {topk}; k += threads) {{\n"
            "    float v = scores[k];\n"
            "    if (v > local_max) local_max = v;\n"
            "  }\n"
            "  reduce_buf[tid] = local_max;"
        ),
        1,
    )
    msl = msl.replace(
        (
            f"  for (uint k = tid; k < {topk}; k += threads) {{\n"
            "    int gather_idx = indices[idx_base + k];\n"
            "    if (gather_idx < 0) {\n"
            "      p[k] = 0.000000e+00f;\n"
            "    } else {\n"
            "      p[k] = exp((scores[k] - row_max));\n"
            "    }\n"
            "  }"
        ),
        (
            f"  for (uint k = tid; k < {topk}; k += threads) {{\n"
            "    float v = scores[k];\n"
            "    p[k] = (v == -INFINITY) ? 0.0f : exp(v - row_max);\n"
            "  }"
        ),
        1,
    )
    msl = msl.replace(
        (
            "  local = 0.000000e+00f;\n"
            f"  for (uint k = tid; k < {topk}; k += threads) {{\n"
            "    local = (local + p[k]);\n"
            "  }\n"
            "  reduce_buf[tid] = local;"
        ),
        (
            "  float local_sum = 0.0f;\n"
            f"  for (uint k = tid; k < {topk}; k += threads) {{\n"
            "    local_sum += p[k];\n"
            "  }\n"
            "  reduce_buf[tid] = local_sum;"
        ),
        1,
    )
    msl = msl.replace(
        (
            "  float inv_sum = (sumexp > 0.000000e+00f) ? "
            "(1.000000e+00f / sumexp) : 0.000000e+00f;"
        ),
        "  float inv_sum = 1.0f / sumexp;",
        1,
    )
    msl = msl.replace(
        (
            "    acc = 0.000000e+00f;\n"
            f"      {kv_base}\n"
            f"      for (uint d_1 = 0; d_1 < {d_v}; ++d_1) {{\n"
            "        acc = (acc + (((float)kv[kv_row_base + d_1]) * "
            "((float)d_out[d_out_row + d_1])));\n"
            "      }\n"
            "    dp[k] = acc;"
        ),
        (
            f"    {kv_base}\n"
            "    float acc = 0.0f;\n"
            f"    for (uint d = 0; d < {d_v}; ++d) {{\n"
            "      float v = float(kv[kv_row_base + d]);\n"
            "      float dod = float(d_out[d_out_row + d]);\n"
            "      acc += v * dod;\n"
            "    }\n"
            "    dp[k] = acc;"
        ),
        1,
    )
    msl = msl.replace(
        (
            "  local = 0.000000e+00f;\n"
            f"  for (uint k = tid; k < {topk}; k += threads) {{\n"
            "    local = (local + (p[k] * dp[k]));\n"
            "  }\n"
            "  reduce_buf[tid] = local;"
        ),
        (
            "  float local_rs = 0.0f;\n"
            f"  for (uint k = tid; k < {topk}; k += threads) {{\n"
            "    local_rs += p[k] * dp[k];\n"
            "  }\n"
            "  reduce_buf[tid] = local_rs;"
        ),
        1,
    )
    msl = msl.replace(
        (
            f"  for (uint d = tid; d < {qk_dim}; d += threads) {{\n"
            "    acc = 0.000000e+00f;\n"
            f"    for (uint k = 0; k < {topk}; ++k) {{\n"
            "      int gather_idx = indices[idx_base + k];\n"
            "      if (gather_idx < 0) {\n"
            "        continue;\n"
            "      }\n"
            f"      {kv_base}\n"
            "      acc = (acc + (ds[k] * ((float)kv[kv_row_base + d])));\n"
            "    }\n"
            "    dq[q_row_base + d] = ((half)(acc * sm_scale));\n"
            "  }"
        ),
        (
            f"  for (uint d = tid; d < {qk_dim}; d += threads) {{\n"
            "    float acc = 0.0f;\n"
            f"    for (uint k = 0; k < {topk}; ++k) {{\n"
            "      int gather_idx = indices[idx_base + k];\n"
            "      if (gather_idx < 0) {\n"
            "        continue;\n"
            "      }\n"
            f"      {kv_base}\n"
            "      float kv_v = float(kv[kv_row_base + d]);\n"
            "      acc += ds[k] * kv_v;\n"
            "    }\n"
            "    dq[q_row_base + d] = ((half)(acc * sm_scale));\n"
            "  }"
        ),
        1,
    )
    msl = re.sub(
        rf"(?P<indent>  )for \(uint k = tid; k < {topk}; k \+= threads\) \{{\n"
        r"(?P=indent)  int gather_idx = indices\[idx_base \+ k\];\n"
        r"(?P=indent)  if \(gather_idx < 0\) \{\n"
        r"(?P=indent)    scores\[k\] = -INFINITY;\n"
        r"(?P=indent)  \} else \{\n"
        r"(?P=indent)    acc = 0\.000000e\+00f;\n"
        r"(?P=indent)    uint kv_row_base = kv_b_base \+ \(uint\(gather_idx\) \* kv_group \+ g\) \* qk_dim;\n"
        rf"(?P=indent)    for \(uint d = 0; d < {qk_dim}; \+\+d\) \{{\n"
        r"(?P=indent)      acc = \(acc \+ \(\(\(float\)q\[q_row_base \+ d\]\) \* \(\(float\)kv\[kv_row_base \+ d\]\)\)\);\n"
        r"(?P=indent)    \}\n"
        r"(?P=indent)    scores\[k\] = \(?acc \* sm_scale\)?;\n"
        r"(?P=indent)  \}\n"
        r"(?P=indent)\}",
        (
            rf"  for (uint k = tid; k < {topk}; k += threads) {{\n"
            "    int gather_idx = indices[idx_base + k];\n"
            "    if (gather_idx < 0) {\n"
            "      scores[k] = -INFINITY;\n"
            "      continue;\n"
            "    }\n"
            f"    {kv_base}\n"
            "    float acc = 0.0f;\n"
            f"    for (uint d = 0; d < {qk_dim}; ++d) {{\n"
            "      float qv = float(q[q_row_base + d]);\n"
            "      float kv_v = float(kv[kv_row_base + d]);\n"
            "      acc += qv * kv_v;\n"
            "    }\n"
            "    scores[k] = acc * sm_scale;\n"
            "  }"
        ),
        msl,
        count=1,
    )
    msl = re.sub(
        rf"(?P<indent>  )for \(uint k = tid; k < {topk}; k \+= threads\) \{{\n"
        r"(?P=indent)  int gather_idx = indices\[idx_base \+ k\];\n"
        r"(?P=indent)  if \(gather_idx < 0\) \{\n"
        r"(?P=indent)    dp\[k\] = 0\.000000e\+00f;\n"
        r"(?P=indent)  \} else \{\n"
        r"(?P=indent)    acc = 0\.000000e\+00f;\n"
        r"(?P=indent)    uint kv_row_base = kv_b_base \+ \(uint\(gather_idx\) \* kv_group \+ g\) \* qk_dim;\n"
        rf"(?P=indent)    for \(uint d_1 = 0; d_1 < {d_v}; \+\+d_1\) \{{\n"
        r"(?P=indent)      acc = \(acc \+ \(\(\(float\)kv\[kv_row_base \+ d_1\]\) \* \(\(float\)d_out\[d_out_row \+ d_1\]\)\)\);\n"
        r"(?P=indent)    \}\n"
        r"(?P=indent)    dp\[k\] = acc;\n"
        r"(?P=indent)  \}\n"
        r"(?P=indent)\}",
        (
            rf"  for (uint k = tid; k < {topk}; k += threads) {{\n"
            "    int gather_idx = indices[idx_base + k];\n"
            "    if (gather_idx < 0) {\n"
            "      dp[k] = 0.000000e+00f;\n"
            "      continue;\n"
            "    }\n"
            f"    {kv_base}\n"
            "    float acc = 0.0f;\n"
            f"    for (uint d = 0; d < {d_v}; ++d) {{\n"
            "      float v = float(kv[kv_row_base + d]);\n"
            "      float dod = float(d_out[d_out_row + d]);\n"
            "      acc += v * dod;\n"
            "    }\n"
            "    dp[k] = acc;\n"
            "  }"
        ),
        msl,
        count=1,
    )
    msl = re.sub(
        rf"(?P<indent>  )for \(uint d = tid; d < {qk_dim}; d \+= threads\) \{{\n"
        r"(?P=indent)  acc = 0\.000000e\+00f;\n"
        rf"(?P=indent)  for \(uint k = 0; k < {topk}; \+\+k\) \{{\n"
        r"(?P=indent)    int gather_idx = indices\[idx_base \+ k\];\n"
        r"(?P=indent)    if \(gather_idx < 0\) \{\n"
        r"(?P=indent)      continue;\n"
        r"(?P=indent)    \}\n"
        r"(?P=indent)    uint kv_row_base = kv_b_base \+ \(uint\(gather_idx\) \* kv_group \+ g\) \* qk_dim;\n"
        r"(?P=indent)    acc = \(acc \+ \(ds\[k\] \* \(\(float\)kv\[\(?kv_row_base \+ d\)?\]\)\)\);\n"
        r"(?P=indent)  \}\n"
        r"(?P=indent)  dq\[q_row_base \+ d\] = \(\(half\)\(acc \* sm_scale\)\);\n"
        r"(?P=indent)\}",
        (
            rf"  for (uint d = tid; d < {qk_dim}; d += threads) {{\n"
            "    float acc = 0.0f;\n"
            f"    for (uint k = 0; k < {topk}; ++k) {{\n"
            "      int gather_idx = indices[idx_base + k];\n"
            "      if (gather_idx < 0) {\n"
            "        continue;\n"
            "      }\n"
            f"      {kv_base}\n"
            "      float kv_v = float(kv[kv_row_base + d]);\n"
            "      acc += ds[k] * kv_v;\n"
            "    }\n"
            "    dq[q_row_base + d] = ((half)(acc * sm_scale));\n"
            "  }"
        ),
        msl,
        count=1,
    )
    msl = re.sub(
        rf"(?P<indent>  )for \(uint kd = tid; kd < {topk * qk_dim}; kd \+= threads\) \{{\n"
        rf"(?P=indent)  uint k = kd / {qk_dim};\n"
        rf"(?P=indent)  uint d = kd % {qk_dim};\n"
        r"(?P=indent)  int gather_idx = indices\[idx_base \+ k\];\n"
        r"(?P=indent)  if \(gather_idx < 0\) \{\n"
        r"(?P=indent)    dkv_partial\[dkv_pb \+ kd\] = 0\.000000e\+00h;\n"
        r"(?P=indent)    continue;\n"
        r"(?P=indent)  \}\n"
        r"(?P=indent)  acc = \(\(sm_scale \* ds\[k\]\) \* \(\(float\)q\[q_row_base \+ d\]\)\);\n"
        r"(?P=indent)  dkv_partial\[dkv_pb \+ kd\] = \(\(half\)\(\(p\[k\] \* \(\(float\)d_out\[d_out_row \+ d\]\)\) \+ acc\)\);\n"
        r"(?P=indent)\}",
        (
            rf"  for (uint kd = tid; kd < {topk * qk_dim}; kd += threads) {{\n"
            f"    uint k = kd / {qk_dim};\n"
            f"    uint d = kd % {qk_dim};\n"
            "    int gather_idx = indices[idx_base + k];\n"
            "    if (gather_idx < 0) {\n"
            "      dkv_partial[dkv_pb + kd] = 0.000000e+00h;\n"
            "      continue;\n"
            "    }\n"
            "    float qv = float(q[q_row_base + d]);\n"
            "    float ks_q = sm_scale * ds[k] * qv;\n"
            "    float dod = float(d_out[d_out_row + d]);\n"
            "    dkv_partial[dkv_pb + kd] = ((half)((p[k] * dod) + ks_q));\n"
            "  }"
        ),
        msl,
        count=1,
    )
    msl = re.sub(
        rf"(?P<indent>  )for \(uint kd = tid; kd < {topk * qk_dim}; kd \+= threads\) \{{\n"
        rf"(?P=indent)  int gather_idx = indices\[\(idx_base \+ \(kd / {qk_dim}\)\)\];\n"
        r"(?P=indent)  if \(gather_idx < 0\) \{\n"
        r"(?P=indent)    dkv_partial\[dkv_pb \+ kd\] = 0\.000000e\+00h;\n"
        r"(?P=indent)    continue;\n"
        r"(?P=indent)  \}\n"
        rf"(?P=indent)  acc = \(\(sm_scale \* ds\[\(kd / {qk_dim}\)\]\) \* \(\(float\)q\[q_row_base \+ tid\]\)\);\n"
        rf"(?P=indent)  dkv_partial\[dkv_pb \+ kd\] = \(\(half\)\(\(p\[\(kd / {qk_dim}\)\] \* \(\(float\)d_out\[d_out_row \+ tid\]\)\) \+ acc\)\);\n"
        r"(?P=indent)\}",
        (
            rf"  for (uint kd = tid; kd < {topk * qk_dim}; kd += threads) {{\n"
            f"    uint k = kd / {qk_dim};\n"
            f"    uint d = kd % {qk_dim};\n"
            "    int gather_idx = indices[idx_base + k];\n"
            "    if (gather_idx < 0) {\n"
            "      dkv_partial[dkv_pb + kd] = 0.000000e+00h;\n"
            "      continue;\n"
            "    }\n"
            "    float qv = float(q[q_row_base + d]);\n"
            "    float ks_q = sm_scale * ds[k] * qv;\n"
            "    float dod = float(d_out[d_out_row + d]);\n"
            "    dkv_partial[dkv_pb + kd] = ((half)((p[k] * dod) + ks_q));\n"
            "  }"
        ),
        msl,
        count=1,
    )
    msl = re.sub(
        rf"(?P<indent>  )for \(uint kd = tid; kd < {topk * qk_dim}; kd \+= threads\) \{{\n"
        rf"(?P=indent)  uint k = kd / {qk_dim};\n"
        rf"(?P=indent)  uint d = kd % {qk_dim};\n"
        r"(?P=indent)  int gather_idx = indices\[idx_base \+ k\];\n"
        r"(?P=indent)  if \(gather_idx < 0\) \{\n"
        r"(?P=indent)    dkv_partial\[dkv_pb \+ kd\] = 0\.000000e\+00h;\n"
        r"(?P=indent)  \} else \{\n"
        rf"(?P=indent)    int (?P<chunk>cse_v\d+(?:_\d+)?) = \(\(kd % {qk_dim}\) / {threads}\);\n"
        rf"(?P=indent)    int (?P<lane_base>cse_v\d+(?:_\d+)?) = \((?P=chunk) \* {threads}\);\n"
        r"(?P=indent)    acc = \(\(sm_scale \* ds\[k\]\) \* \(\(float\)q\[\(\(cse_v\d+(?:_\d+)? \+ (?P=lane_base)\) \+ tid\)\]\)\);\n"
        rf"(?P=indent)    if \((?P=chunk) < {d_v // threads}\) \{{\n"
        r"(?P=indent)      dkv_partial\[dkv_pb \+ kd\] = \(\(half\)\(\(p\[k\] \* \(\(float\)d_out\[\(\(cse_v\d+(?:_\d+)? \+ (?P=lane_base)\) \+ tid\)\]\)\) \+ acc\)\);\n"
        r"(?P=indent)    \} else \{\n"
        r"(?P=indent)      dkv_partial\[dkv_pb \+ kd\] = \(\(half\)acc\);\n"
        r"(?P=indent)    \}\n"
        r"(?P=indent)  \}\n"
        r"(?P=indent)\}",
        (
            rf"  for (uint kd = tid; kd < {topk * qk_dim}; kd += threads) {{\n"
            f"    uint k = kd / {qk_dim};\n"
            f"    uint d = kd % {qk_dim};\n"
            "    int gather_idx = indices[idx_base + k];\n"
            "    if (gather_idx < 0) {\n"
            "      dkv_partial[dkv_pb + kd] = 0.000000e+00h;\n"
            "      continue;\n"
            "    }\n"
            "    float qv = float(q[q_row_base + d]);\n"
            "    float ks_q = sm_scale * ds[k] * qv;\n"
            f"    if (d < {d_v}) {{\n"
            "      float dod = float(d_out[d_out_row + d]);\n"
            "      dkv_partial[dkv_pb + kd] = ((half)((p[k] * dod) + ks_q));\n"
            "    } else {\n"
            "      dkv_partial[dkv_pb + kd] = ((half)ks_q);\n"
            "    }\n"
            "  }"
        ),
        msl,
        count=1,
    )
    for name in re.findall(r"(?m)^  int (cse_v\d+(?:_\d+)?) = [^\n]+;\n", msl):
        decl_re = rf"(?m)^  int {name} = [^\n]+;\n"
        body_without_decl = re.sub(decl_re, "", msl)
        if re.search(rf"\b{name}\b", body_without_decl) is None:
            msl = body_without_decl
    return msl.replace(
        "      reduce_buf[tid] = (reduce_buf[tid] + reduce_buf[tid + stride]);",
        "      reduce_buf[tid] += reduce_buf[tid + stride];",
    )


def _canonicalize_bwd_unsigned_msl(msl: str) -> str:
    """Use unsigned lane/index arithmetic like the hand-written backward kernel."""

    replacements = {
        "int gid = int(blockIdx.x);": "uint gid = blockIdx.x;",
        "int tid = int(threadIdx.x);": "uint tid = threadIdx.x;",
        "uint gid = blockIdx.x;": "uint gid = threadgroup_position_in_grid.x;",
        "uint tid = threadIdx.x;": "uint tid = thread_position_in_threadgroup.x;",
        "int threads = 64;": "uint threads = 64;",
        "int threads = 32;": "uint threads = 32;",
        "int threads = 16;": "uint threads = 16;",
        "int threads = 8;": "uint threads = 8;",
        "int threads = 4;": "uint threads = 4;",
        "for (int stride = threads / 2; stride > 0; stride >>= 1)": (
            "for (uint stride = threads / 2; stride > 0; stride >>= 1)"
        ),
    }
    for old, new in replacements.items():
        msl = msl.replace(old, new)
    msl = msl.replace("  uint3 blockIdx = threadgroup_position_in_grid;\n", "")
    msl = msl.replace("  uint3 threadIdx = thread_position_in_threadgroup;\n", "")
    msl = msl.replace("    uint3 blockIdx = threadgroup_position_in_grid;\n", "")
    msl = msl.replace("    uint3 threadIdx = thread_position_in_threadgroup;\n", "")
    msl = re.sub(r"for \(int k = tid; k < (?P<limit>\d+); k \+= threads\)", r"for (uint k = tid; k < \g<limit>; k += threads)", msl)
    msl = re.sub(r"for \(int k = 0; k < (?P<limit>\d+); \+\+k\)", r"for (uint k = 0; k < \g<limit>; ++k)", msl)
    msl = re.sub(r"for \(int d = tid; d < (?P<limit>\d+); d \+= threads\)", r"for (uint d = tid; d < \g<limit>; d += threads)", msl)
    msl = re.sub(r"for \(int kd = tid; kd < (?P<limit>\d+); kd \+= threads\)", r"for (uint kd = tid; kd < \g<limit>; kd += threads)", msl)
    msl = re.sub(r"for \(int d = 0; d < (?P<limit>\d+); \+\+d\)", r"for (uint d = 0; d < \g<limit>; ++d)", msl)
    msl = re.sub(r"for \(int d_1 = 0; d_1 < (?P<limit>\d+); \+\+d_1\)", r"for (uint d_1 = 0; d_1 < \g<limit>; ++d_1)", msl)
    return msl


def _drop_unused_bwd_scalar_declarations(msl: str) -> str:
    """Remove TileLang scalar temp declarations after backward canonicalization."""

    msl = re.sub(r"(?m)^[ \t]*int gather_idx;\n", "", msl)
    msl = re.sub(
        r"(?m)^(?P<indent>[ \t]*)inv_sum = "
        r"\(sumexp > 0\.000000e\+00f\) \? "
        r"\(1\.000000e\+00f / sumexp\) : 0\.000000e\+00f;",
        r"\g<indent>float inv_sum = "
        r"(sumexp > 0.000000e+00f) ? (1.000000e+00f / sumexp) : 0.000000e+00f;",
        msl,
        count=1,
    )
    if "sumexp <= 0.000000e+00f" in msl:
        msl = re.sub(
            r"(?m)^(?P<indent>[ \t]*)float inv_sum = "
            r"\(sumexp > 0\.000000e\+00f\) \? "
            r"\(1\.000000e\+00f / sumexp\) : 0\.000000e\+00f;",
            r"\g<indent>float inv_sum = 1.0f / sumexp;",
            msl,
            count=1,
        )
    for name in ("acc", "local", "inv_sum"):
        if re.search(rf"(?m)^[ \t]*{name} = ", msl) is None:
            msl = re.sub(rf"(?m)^[ \t]*float {name};\n", "", msl)
    return msl.replace(
        "      if (reduce_buf[tid] < reduce_buf[tid + stride]) {\n"
        "        reduce_buf[tid] = reduce_buf[tid + stride];\n"
        "      }",
        "      float a = reduce_buf[tid];\n"
        "      float b_v = reduce_buf[tid + stride];\n"
        "      if (b_v > a) reduce_buf[tid] = b_v;",
        1,
    )


def _canonicalize_bwd_path_b_layout(msl: str) -> str:
    """Match the hand-written backward kernel's shared-buffer layout and hot expressions."""

    msl = re.sub(
        r"(?m)^  threadgroup float scores\[(?P<topk>\d+)\];\n"
        r"^  threadgroup float reduce_buf\[(?P<threads>\d+)\];\n"
        r"^  threadgroup float p\[(?P=topk)\];\n"
        r"^  threadgroup float dp\[(?P=topk)\];\n"
        r"^  threadgroup float ds\[(?P=topk)\];",
        "  threadgroup float scores[\\g<topk>];\n"
        "  threadgroup float p[\\g<topk>];\n"
        "  threadgroup float dp[\\g<topk>];\n"
        "  threadgroup float ds[\\g<topk>];\n"
        "  threadgroup float reduce_buf[\\g<threads>];",
        msl,
        count=1,
    )
    return (
        msl.replace("p[k] = (p[k] * inv_sum);", "p[k] = p[k] * inv_sum;")
        .replace("ds[k] = (p[k] * (dp[k] - rowsum));", "ds[k] = p[k] * (dp[k] - rowsum);")
        .replace("scores[k] = (acc * sm_scale);", "scores[k] = acc * sm_scale;")
    )


def _scalarize_singleton_thread_arrays(msl: str) -> str:
    """Trim TileLang's one-element thread arrays from hot scalar state."""

    for var, dtype in {
        "gather_idx": "int",
        "acc": "float",
        "local": "float",
        "inv_sum": "float",
        "stride": "int",
    }.items():
        msl = re.sub(rf"\bthread {dtype} {var}\[1\];", f"{dtype} {var};", msl)
        msl = re.sub(rf"\b{var}\[(?:0|\(long\)0)\]", var, msl)
    return msl


def _postprocess_lowered_msl(
    msl: str,
    *,
    seq_len_kv: int,
    remove_flat_kv_bounds: bool = False,
    forward_fast_return: bool = False,
    canonicalize_fwd: bool = False,
    canonicalize_bwd: bool = False,
    topk: int | None = None,
    heads: int | None = None,
    seq_len: int | None = None,
    kv_group: int | None = None,
    head_kv: int | None = None,
    seq_len_kv_for_indexing: int | None = None,
    qk_dim: int | None = None,
    d_v: int | None = None,
    threads: int | None = None,
) -> str:
    # The TIR uses ``T.float32(-1.0e38)`` as a finite stand-in for ``-INFINITY``
    # because ``-T.infinity('float32')`` was tripping z3-final's canonicalizer
    # in upstream tilelang. The lowered MSL therefore reads ``-1.000000e+38f``
    # everywhere we previously had ``-INFINITY``. Normalize back so the rest of
    # the canonicalization regexes (and the all-masked fast-return) keep
    # matching against the historical token.
    msl = msl.replace("-1.000000e+38f", "-INFINITY")
    msl = _remove_redundant_kv_bounds_checks(msl, seq_len_kv=seq_len_kv)
    if remove_flat_kv_bounds:
        msl = _remove_redundant_flat_kv_bounds_checks(msl)
    msl = _scalarize_singleton_thread_arrays(msl)
    if forward_fast_return:
        if d_v is None or threads is None:
            raise ValueError("forward_fast_return requires d_v and threads")
        msl = _insert_forward_all_masked_fast_return(msl, d_v=d_v, threads=threads)
    if canonicalize_fwd:
        if topk is None or d_v is None or threads is None:
            raise ValueError("canonicalize_fwd requires topk, d_v, and threads")
        msl = _canonicalize_fwd_lane_indexing(
            msl,
            topk=topk,
            threads=threads,
            d_v=d_v,
        )
        msl = _canonicalize_fwd_reductions(msl, threads=threads)
        msl = _canonicalize_fwd_hot_loops(msl)
        msl = _canonicalize_fwd_unsigned_msl(msl)
        if (
            heads is None
            or seq_len is None
            or qk_dim is None
            or kv_group is None
            or head_kv is None
            or seq_len_kv_for_indexing is None
        ):
            raise ValueError(
                "canonicalize_fwd requires heads, seq_len, qk_dim, kv_group, "
                "head_kv, and seq_len_kv_for_indexing"
            )
        msl = _canonicalize_fwd_base_indexing(
            msl,
            heads=heads,
            seq_len=seq_len,
            qk_dim=qk_dim,
            kv_group=kv_group,
            head_kv=head_kv,
            topk=topk,
            seq_len_kv=seq_len_kv_for_indexing,
            d_v=d_v,
        )
        msl = _canonicalize_fwd_qk_negative_continue(msl)
        msl = _drop_unused_fwd_scalar_declarations(msl)
    if canonicalize_bwd:
        if topk is None or qk_dim is None or d_v is None or threads is None:
            raise ValueError("canonicalize_bwd requires topk, qk_dim, d_v, and threads")
        msl = _canonicalize_bwd_lane_indexing(
            msl,
            topk=topk,
            threads=threads,
            qk_dim=qk_dim,
            d_v=d_v,
        )
        msl = _canonicalize_bwd_residual_tmp_indexing(
            msl,
            threads=threads,
            qk_dim=qk_dim,
            d_v=d_v,
        )
        msl = _canonicalize_bwd_reductions(msl, threads=threads)
        msl = _insert_backward_all_masked_fast_return(msl, qk_dim=qk_dim, topk=topk)
        msl = _canonicalize_bwd_hot_loops(msl, qk_dim=qk_dim, d_v=d_v)
        msl = _canonicalize_bwd_unsigned_msl(msl)
        if (
            heads is None
            or seq_len is None
            or kv_group is None
            or head_kv is None
            or seq_len_kv_for_indexing is None
        ):
            raise ValueError(
                "canonicalize_bwd requires heads, seq_len, kv_group, head_kv, "
                "and seq_len_kv_for_indexing"
            )
        msl = _canonicalize_bwd_base_indexing(
            msl,
            heads=heads,
            seq_len=seq_len,
            qk_dim=qk_dim,
            kv_group=kv_group,
            head_kv=head_kv,
            topk=topk,
            seq_len_kv=seq_len_kv_for_indexing,
            d_v=d_v,
        )
        msl = _canonicalize_bwd_path_b_hot_loops(msl, qk_dim=qk_dim, d_v=d_v, topk=topk)
        msl = _drop_unused_bwd_scalar_declarations(msl)
        msl = _canonicalize_bwd_path_b_layout(msl)
    return msl


@dataclass(frozen=True)
class SparseMLAPathCStatus:
    """Runtime status for the Path C TileLang DSL Sparse-MLA backward kernel."""

    available: bool
    reason: str
    fp16_carrier: bool = True


class SparseMLAPathCDirectError(RuntimeError):
    """Raised when the owner-output tvm-ffi forward route cannot run safely."""


def _require_int32_indices_no_hidden_cast(indices: mx.array, *, op_name: str) -> mx.array:
    if indices.dtype != mx.int32:
        raise TypeError(
            f"{op_name} requires mx.int32 indices; got {indices.dtype}. "
            "Path C will not cast or copy sparse index tensors at the wrapper boundary."
        )
    return indices


def _tilelang_available() -> tuple[bool, str]:
    try:
        import tilelang  # noqa: F401
        from tilelang import tvm as _tvm  # noqa: F401
        from tilelang.engine.lower import lower as _lower  # noqa: F401
        import tilelang.language as _T  # noqa: F401
    except Exception as exc:  # pragma: no cover - macOS without tilelang
        return False, f"tilelang import failed: {exc}"
    return True, "tilelang importable"


@lru_cache(maxsize=1)
def sparse_mla_path_c_status() -> SparseMLAPathCStatus:
    """Return whether the Path C TileLang DSL kernel can dispatch on this host."""

    if not can_run_metal():
        return SparseMLAPathCStatus(
            available=False,
            reason="MLX Metal backend is not available on the default GPU device",
        )
    ok, reason = _tilelang_available()
    if not ok:
        return SparseMLAPathCStatus(available=False, reason=reason)
    return SparseMLAPathCStatus(
        available=True,
        reason="Sparse-MLA Path C forward/backward TileLang DSL ready",
    )


def _threadgroup_size(topk: int) -> int:
    """Match Path B's power-of-two threadgroup sizing for TOPK reductions."""

    threads = min(64, max(1, topk))
    power = 1
    while (power << 1) <= threads:
        power <<= 1
    return power


def _mlx_total_thread_grid(lowering: _msl_transform.TileLangMSLLowering) -> tuple[int, int, int]:
    """Convert TileLang block-grid metadata to MLX's total-thread launch grid."""

    return (
        lowering.grid[0] * lowering.threadgroup[0],
        lowering.grid[1] * lowering.threadgroup[1],
        lowering.grid[2] * lowering.threadgroup[2],
    )


def _make_sparse_mla_fwd_prim(
    BATCH: int,
    SEQ_LEN: int,
    HEADS: int,
    QK_DIM: int,
    KV_GROUP: int,
    HEAD_KV: int,
    TOPK: int,
    SEQ_LEN_KV: int,
    D_V: int,
    THREADS: int,
) -> Any:
    """Build the Sparse-MLA fwd PrimFunc only (no MLX kernel wrap).

    Shared between the legacy MSL-shim builder (:func:`_fwd_kernel_for`) and
    the engine-path builder (:func:`_fwd_kernel_engine_for`). Pure TileLang
    DSL; no T.gemm / no transposed matmul / no inline MSL.
    """

    import tilelang.language as T

    Q_SIZE = BATCH * SEQ_LEN * HEADS * QK_DIM
    KV_SIZE = BATCH * SEQ_LEN_KV * KV_GROUP * QK_DIM
    IDX_SIZE = BATCH * SEQ_LEN * KV_GROUP * TOPK
    OUT_SIZE = BATCH * SEQ_LEN * HEADS * D_V
    LSE_SIZE = BATCH * SEQ_LEN * HEADS
    LANES = BATCH * SEQ_LEN * HEADS
    LOG_THREADS = THREADS.bit_length() - 1

    @T.prim_func
    def sparse_mla_fwd(
        q: T.Tensor((Q_SIZE,), "float16"),
        kv: T.Tensor((KV_SIZE,), "float16"),
        indices: T.Tensor((IDX_SIZE,), "int32"),
        sm_scale_buf: T.Tensor((1,), "float32"),
        out: T.Tensor((OUT_SIZE,), "float16"),
        lse: T.Tensor((LSE_SIZE,), "float32"),
    ):
        with T.Kernel(LANES, threads=THREADS) as bx:
            lane = T.get_thread_binding()
            scores = T.alloc_shared((TOPK,), "float32", scope="shared")
            reduce_buf = T.alloc_shared((THREADS,), "float32", scope="shared")
            acc = T.alloc_local((1,), "float32")
            local = T.alloc_local((1,), "float32")
            inv_sum = T.alloc_local((1,), "float32")
            stride = T.alloc_local((1,), "int32")
            gather_idx = T.alloc_local((1,), "int32")

            h = bx % HEADS
            b = bx // (HEADS * SEQ_LEN)
            g = h // HEAD_KV
            q_row_base = bx * QK_DIM
            kv_b_base = b * (SEQ_LEN_KV * KV_GROUP * QK_DIM)
            idx_base = ((bx // HEADS) * KV_GROUP + g) * TOPK
            out_row = bx * D_V
            sm_scale = sm_scale_buf[0]

            for k in T.serial(lane, TOPK, step=THREADS):
                gather_idx[0] = indices[idx_base + k]
                if gather_idx[0] < 0:
                    scores[k] = T.float32(-1.0e38)
                else:
                    acc[0] = 0.0
                    kv_row_base = kv_b_base + (gather_idx[0] * KV_GROUP + g) * QK_DIM
                    for d in T.serial(QK_DIM):
                        acc[0] = acc[0] + T.cast(q[q_row_base + d], "float32") * T.cast(
                            kv[kv_row_base + d], "float32"
                        )
                    scores[k] = acc[0] * sm_scale
            T.sync_threads()

            local[0] = T.float32(-1.0e38)
            for k in T.serial(lane, TOPK, step=THREADS):
                if scores[k] > local[0]:
                    local[0] = scores[k]
            reduce_buf[lane] = local[0]
            T.sync_threads()
            for round_id in T.serial(LOG_THREADS):
                stride[0] = T.shift_right(THREADS, round_id + 1)
                if lane < stride[0]:
                    if reduce_buf[lane + stride[0]] > reduce_buf[lane]:
                        reduce_buf[lane] = reduce_buf[lane + stride[0]]
                T.sync_threads()
            row_max = reduce_buf[0]

            for k in T.serial(lane, TOPK, step=THREADS):
                if scores[k] == T.float32(-1.0e38):
                    scores[k] = 0.0
                else:
                    scores[k] = T.exp(scores[k] - row_max)
            T.sync_threads()

            local[0] = 0.0
            for k in T.serial(lane, TOPK, step=THREADS):
                local[0] = local[0] + scores[k]
            reduce_buf[lane] = local[0]
            T.sync_threads()
            for round_id in T.serial(LOG_THREADS):
                stride[0] = T.shift_right(THREADS, round_id + 1)
                if lane < stride[0]:
                    reduce_buf[lane] = reduce_buf[lane] + reduce_buf[lane + stride[0]]
                T.sync_threads()
            sumexp = reduce_buf[0]

            inv_sum[0] = 0.0
            if sumexp > 0.0:
                inv_sum[0] = 1.0 / sumexp

            for d in T.serial(lane, D_V, step=THREADS):
                acc[0] = 0.0
                for k in T.serial(TOPK):
                    gather_idx[0] = indices[idx_base + k]
                    if gather_idx[0] >= 0:
                        kv_row_base = kv_b_base + (gather_idx[0] * KV_GROUP + g) * QK_DIM
                        acc[0] = acc[0] + scores[k] * T.cast(
                            kv[kv_row_base + d], "float32"
                        )
                out[out_row + d] = acc[0] * inv_sum[0]

            if lane == 0:
                if sumexp > 0.0:
                    lse[bx] = row_max + T.log(sumexp)
                else:
                    lse[bx] = 0.0

    return sparse_mla_fwd


def _make_sparse_mla_fwd_direct_prim(
    BATCH: int,
    SEQ_LEN: int,
    HEADS: int,
    QK_DIM: int,
    KV_GROUP: int,
    HEAD_KV: int,
    TOPK: int,
    SEQ_LEN_KV: int,
    D_V: int,
    THREADS: int,
) -> Any:
    """Build the owner-output tvm-ffi forward PrimFunc in Metal buffer order.

    TileLang's Metal codegen alphabetizes device buffer parameters. The
    tvm-ffi MLX adapter validates and converts buffers in PrimFunc parameter
    order, so this direct-only PrimFunc declares parameters in the same order
    as the generated Metal signature: ``indices, kv, lse, out, q,
    sm_scale_buf``. The public wrapper still accepts the logical
    ``q, kv, indices, sm_scale_buf, out, lse`` contract.
    """

    import tilelang.language as T

    LANES = BATCH * SEQ_LEN * HEADS
    LOG_THREADS = THREADS.bit_length() - 1

    @T.prim_func
    def sparse_mla_fwd_direct(
        indices: T.Tensor((BATCH, SEQ_LEN, KV_GROUP, TOPK), "int32"),
        kv: T.Tensor((BATCH, SEQ_LEN_KV, KV_GROUP, QK_DIM), "float16"),
        lse: T.Tensor((BATCH, SEQ_LEN, HEADS), "float32"),
        out: T.Tensor((BATCH, SEQ_LEN, HEADS, D_V), "float16"),
        q: T.Tensor((BATCH, SEQ_LEN, HEADS, QK_DIM), "float16"),
        sm_scale_buf: T.Tensor((1,), "float32"),
    ):
        with T.Kernel(LANES, threads=THREADS) as bx:
            lane = T.get_thread_binding()
            scores = T.alloc_shared((TOPK,), "float32", scope="shared")
            reduce_buf = T.alloc_shared((THREADS,), "float32", scope="shared")
            acc = T.alloc_local((1,), "float32")
            local = T.alloc_local((1,), "float32")
            inv_sum = T.alloc_local((1,), "float32")
            stride = T.alloc_local((1,), "int32")
            gather_idx = T.alloc_local((1,), "int32")

            h = bx % HEADS
            b = bx // (HEADS * SEQ_LEN)
            s = (bx // HEADS) % SEQ_LEN
            g = h // HEAD_KV
            sm_scale = sm_scale_buf[0]

            for k in T.serial(lane, TOPK, step=THREADS):
                gather_idx[0] = indices[b, s, g, k]
                if gather_idx[0] < 0:
                    scores[k] = T.float32(-1.0e38)
                else:
                    acc[0] = 0.0
                    for d in T.serial(QK_DIM):
                        acc[0] = acc[0] + T.cast(q[b, s, h, d], "float32") * T.cast(
                            kv[b, gather_idx[0], g, d], "float32"
                        )
                    scores[k] = acc[0] * sm_scale
            T.sync_threads()

            local[0] = T.float32(-1.0e38)
            for k in T.serial(lane, TOPK, step=THREADS):
                if scores[k] > local[0]:
                    local[0] = scores[k]
            reduce_buf[lane] = local[0]
            T.sync_threads()
            for round_id in T.serial(LOG_THREADS):
                stride[0] = T.shift_right(THREADS, round_id + 1)
                if lane < stride[0]:
                    if reduce_buf[lane + stride[0]] > reduce_buf[lane]:
                        reduce_buf[lane] = reduce_buf[lane + stride[0]]
                T.sync_threads()
            row_max = reduce_buf[0]

            for k in T.serial(lane, TOPK, step=THREADS):
                if scores[k] == T.float32(-1.0e38):
                    scores[k] = 0.0
                else:
                    scores[k] = T.exp(scores[k] - row_max)
            T.sync_threads()

            local[0] = 0.0
            for k in T.serial(lane, TOPK, step=THREADS):
                local[0] = local[0] + scores[k]
            reduce_buf[lane] = local[0]
            T.sync_threads()
            for round_id in T.serial(LOG_THREADS):
                stride[0] = T.shift_right(THREADS, round_id + 1)
                if lane < stride[0]:
                    reduce_buf[lane] = reduce_buf[lane] + reduce_buf[lane + stride[0]]
                T.sync_threads()
            sumexp = reduce_buf[0]

            inv_sum[0] = 0.0
            if sumexp > 0.0:
                inv_sum[0] = 1.0 / sumexp

            for d in T.serial(lane, D_V, step=THREADS):
                acc[0] = 0.0
                for k in T.serial(TOPK):
                    gather_idx[0] = indices[b, s, g, k]
                    if gather_idx[0] >= 0:
                        acc[0] = acc[0] + scores[k] * T.cast(
                            kv[b, gather_idx[0], g, d], "float32"
                        )
                out[b, s, h, d] = acc[0] * inv_sum[0]

            if lane == 0:
                if sumexp > 0.0:
                    lse[b, s, h] = row_max + T.log(sumexp)
                else:
                    lse[b, s, h] = 0.0

    return sparse_mla_fwd_direct


@lru_cache(maxsize=128)
def _fwd_kernel_engine_for(
    BATCH: int,
    SEQ_LEN: int,
    HEADS: int,
    QK_DIM: int,
    KV_GROUP: int,
    HEAD_KV: int,
    TOPK: int,
    SEQ_LEN_KV: int,
    D_V: int,
    THREADS: int,
    target: str = "metal",
) -> Any:
    """Build the Sparse-MLA fwd kernel through the unified engine dispatcher.

    Returns whatever :func:`dispatch_lower` returns for the active mode:
    a ``tilelang.compile`` artifact carrying ``_tilelang_engine_target``
    (engine path; backend-portable across CUDA / HIP / Metal), or a
    :class:`TileLangMSLLowering` (shim path). Used by parity tests and by
    callers that want a backend-portable artifact.

    The legacy :func:`_fwd_kernel_for` is preserved verbatim because the MLX
    runtime path needs ``lowering.body`` / ``lowering.header`` strings; the
    full flip will land alongside the Phase-3 MSL-extraction adapter.
    """

    prim = _make_sparse_mla_fwd_prim(
        BATCH=BATCH,
        SEQ_LEN=SEQ_LEN,
        HEADS=HEADS,
        QK_DIM=QK_DIM,
        KV_GROUP=KV_GROUP,
        HEAD_KV=HEAD_KV,
        TOPK=TOPK,
        SEQ_LEN_KV=SEQ_LEN_KV,
        D_V=D_V,
        THREADS=THREADS,
    )
    return dispatch_lower(prim, target)


def lower_sparse_mla_fwd_msl(
    *,
    BATCH: int,
    SEQ_LEN: int,
    HEADS: int,
    QK_DIM: int,
    KV_GROUP: int,
    HEAD_KV: int,
    TOPK: int,
    SEQ_LEN_KV: int,
    D_V: int,
    THREADS: int,
    target: str = "metal",
) -> str:
    """Lower the Path C Sparse-MLA fwd kernel and return rendered source.

    Routes through :func:`_engine_dispatch.dispatch_lower` so the env var
    ``CPPMEGA_MLX_TILELANG_ENGINE`` selects between the unified
    ``tilelang.compile`` engine and the legacy MSL shim. Under ``engine`` mode
    the returned source can be CUDA / HIP / Metal depending on ``target``;
    under ``shim`` mode it is always the legacy MSL string. Used by parity
    tests in ``cppmega/tests/test_sparse_mla_path_c_engine.py``.
    """

    from cppmega_mlx.nn._tilelang._engine_dispatch import (
        artifact_to_source,
        dispatch_lower,
    )

    prim = _make_sparse_mla_fwd_prim(
        BATCH=BATCH,
        SEQ_LEN=SEQ_LEN,
        HEADS=HEADS,
        QK_DIM=QK_DIM,
        KV_GROUP=KV_GROUP,
        HEAD_KV=HEAD_KV,
        TOPK=TOPK,
        SEQ_LEN_KV=SEQ_LEN_KV,
        D_V=D_V,
        THREADS=THREADS,
    )
    artifact = dispatch_lower(prim, target)
    return artifact_to_source(artifact)


@lru_cache(maxsize=128)
def _fwd_kernel_for(
    BATCH: int,
    SEQ_LEN: int,
    HEADS: int,
    QK_DIM: int,
    KV_GROUP: int,
    HEAD_KV: int,
    TOPK: int,
    SEQ_LEN_KV: int,
    D_V: int,
    THREADS: int,
) -> tuple[Any, _msl_transform.TileLangMSLLowering]:
    """Build and cache a shape-specialized threadgroup Sparse-MLA fwd kernel."""

    import tilelang.language as T

    LANES = BATCH * SEQ_LEN * HEADS
    Q_SIZE = BATCH * SEQ_LEN * HEADS * QK_DIM
    KV_SIZE = BATCH * SEQ_LEN_KV * KV_GROUP * QK_DIM
    IDX_SIZE = BATCH * SEQ_LEN * KV_GROUP * TOPK
    OUT_SIZE = BATCH * SEQ_LEN * HEADS * D_V
    LSE_SIZE = BATCH * SEQ_LEN * HEADS
    LOG_THREADS = THREADS.bit_length() - 1

    @T.prim_func
    def sparse_mla_fwd(
        q: T.Tensor((Q_SIZE,), "float16"),
        kv: T.Tensor((KV_SIZE,), "float16"),
        indices: T.Tensor((IDX_SIZE,), "int32"),
        sm_scale_buf: T.Tensor((1,), "float32"),
        out: T.Tensor((OUT_SIZE,), "float16"),
        lse: T.Tensor((LSE_SIZE,), "float32"),
    ):
        with T.Kernel(LANES, threads=THREADS) as bx:
            lane = T.get_thread_binding()
            scores = T.alloc_shared((TOPK,), "float32", scope="shared")
            reduce_buf = T.alloc_shared((THREADS,), "float32", scope="shared")
            acc = T.alloc_local((1,), "float32")
            local = T.alloc_local((1,), "float32")
            inv_sum = T.alloc_local((1,), "float32")
            stride = T.alloc_local((1,), "int32")
            gather_idx = T.alloc_local((1,), "int32")

            h = bx % HEADS
            b = bx // (HEADS * SEQ_LEN)
            g = h // HEAD_KV
            q_row_base = bx * QK_DIM
            kv_b_base = b * (SEQ_LEN_KV * KV_GROUP * QK_DIM)
            idx_base = ((bx // HEADS) * KV_GROUP + g) * TOPK
            out_row = bx * D_V
            sm_scale = sm_scale_buf[0]

            for k in T.serial(lane, TOPK, step=THREADS):
                gather_idx[0] = indices[idx_base + k]
                if gather_idx[0] < 0:
                    scores[k] = T.float32(-1.0e38)
                else:
                    acc[0] = 0.0
                    kv_row_base = kv_b_base + (gather_idx[0] * KV_GROUP + g) * QK_DIM
                    for d in T.serial(QK_DIM):
                        acc[0] = acc[0] + T.cast(q[q_row_base + d], "float32") * T.cast(
                            kv[kv_row_base + d], "float32"
                        )
                    scores[k] = acc[0] * sm_scale
            T.sync_threads()

            local[0] = T.float32(-1.0e38)
            for k in T.serial(lane, TOPK, step=THREADS):
                if scores[k] > local[0]:
                    local[0] = scores[k]
            reduce_buf[lane] = local[0]
            T.sync_threads()
            for round_id in T.serial(LOG_THREADS):
                stride[0] = T.shift_right(THREADS, round_id + 1)
                if lane < stride[0]:
                    if reduce_buf[lane + stride[0]] > reduce_buf[lane]:
                        reduce_buf[lane] = reduce_buf[lane + stride[0]]
                T.sync_threads()
            row_max = reduce_buf[0]

            for k in T.serial(lane, TOPK, step=THREADS):
                if scores[k] == T.float32(-1.0e38):
                    scores[k] = 0.0
                else:
                    scores[k] = T.exp(scores[k] - row_max)
            T.sync_threads()

            local[0] = 0.0
            for k in T.serial(lane, TOPK, step=THREADS):
                local[0] = local[0] + scores[k]
            reduce_buf[lane] = local[0]
            T.sync_threads()
            for round_id in T.serial(LOG_THREADS):
                stride[0] = T.shift_right(THREADS, round_id + 1)
                if lane < stride[0]:
                    reduce_buf[lane] = reduce_buf[lane] + reduce_buf[lane + stride[0]]
                T.sync_threads()
            sumexp = reduce_buf[0]

            inv_sum[0] = 0.0
            if sumexp > 0.0:
                inv_sum[0] = 1.0 / sumexp

            for d in T.serial(lane, D_V, step=THREADS):
                acc[0] = 0.0
                for k in T.serial(TOPK):
                    gather_idx[0] = indices[idx_base + k]
                    if gather_idx[0] >= 0:
                        kv_row_base = kv_b_base + (gather_idx[0] * KV_GROUP + g) * QK_DIM
                        acc[0] = acc[0] + scores[k] * T.cast(
                            kv[kv_row_base + d], "float32"
                        )
                out[out_row + d] = acc[0] * inv_sum[0]

            if lane == 0:
                if sumexp > 0.0:
                    lse[bx] = row_max + T.log(sumexp)
                else:
                    lse[bx] = 0.0

    lowering = cast(
        _msl_transform.TileLangMSLLowering,
        dispatch_lower(sparse_mla_fwd, target="metal", return_msl=True),
    )
    lowering = _msl_transform.TileLangMSLLowering(
        header=lowering.header,
        body=_postprocess_lowered_msl(
            lowering.body,
            seq_len_kv=SEQ_LEN_KV,
            remove_flat_kv_bounds=True,
            forward_fast_return=True,
            canonicalize_fwd=True,
            topk=TOPK,
            heads=HEADS,
            seq_len=SEQ_LEN,
            kv_group=KV_GROUP,
            head_kv=HEAD_KV,
            seq_len_kv_for_indexing=SEQ_LEN_KV,
            qk_dim=QK_DIM,
            d_v=D_V,
            threads=THREADS,
        ),
        grid=lowering.grid,
        threadgroup=lowering.threadgroup,
        msl_text=_postprocess_lowered_msl(
            lowering.msl_text,
            seq_len_kv=SEQ_LEN_KV,
            remove_flat_kv_bounds=True,
            forward_fast_return=True,
            canonicalize_fwd=True,
            topk=TOPK,
            heads=HEADS,
            seq_len=SEQ_LEN,
            kv_group=KV_GROUP,
            head_kv=HEAD_KV,
            seq_len_kv_for_indexing=SEQ_LEN_KV,
            qk_dim=QK_DIM,
            d_v=D_V,
            threads=THREADS,
        ),
        buffer_param_names=lowering.buffer_param_names,
        kernel_name=lowering.kernel_name,
    )
    kernel = mx.fast.metal_kernel(
        name=(
            "cppmega_sparse_mla_path_c_fwd_noguard_"
            f"{BATCH}_{SEQ_LEN}_{HEADS}_{QK_DIM}_{KV_GROUP}_{TOPK}_{SEQ_LEN_KV}_{D_V}_{THREADS}"
        ),
        input_names=["q", "kv", "indices", "sm_scale_buf"],
        output_names=["out", "lse"],
        source=lowering.body,
        header=lowering.header,
        ensure_row_contiguous=True,
    )
    return kernel, lowering


def _make_sparse_mla_bwd_prim(
    BATCH: int,
    SEQ_LEN: int,
    HEADS: int,
    QK_DIM: int,
    KV_GROUP: int,
    HEAD_KV: int,
    TOPK: int,
    SEQ_LEN_KV: int,
    D_V: int,
    THREADS: int,
) -> Any:
    """Build the Sparse-MLA bwd PrimFunc only (no MLX kernel wrap).

    Shared between the legacy MSL-shim builder (:func:`_bwd_kernel_for`) and
    the engine-path builder (:func:`_bwd_kernel_engine_for`). Pure TileLang
    DSL; no T.gemm / no transposed matmul / no inline MSL.
    """

    import tilelang.language as T

    LANES = BATCH * SEQ_LEN * HEADS
    Q_SIZE = BATCH * SEQ_LEN * HEADS * QK_DIM
    KV_SIZE = BATCH * SEQ_LEN_KV * KV_GROUP * QK_DIM
    DOUT_SIZE = BATCH * SEQ_LEN * HEADS * D_V
    IDX_SIZE = BATCH * SEQ_LEN * KV_GROUP * TOPK
    DKV_PARTIAL_SIZE = BATCH * SEQ_LEN * HEADS * TOPK * QK_DIM
    LOG_THREADS = THREADS.bit_length() - 1

    @T.prim_func
    def sparse_mla_bwd(
        q: T.Tensor((Q_SIZE,), "float16"),
        kv: T.Tensor((KV_SIZE,), "float16"),
        d_out: T.Tensor((DOUT_SIZE,), "float16"),
        indices: T.Tensor((IDX_SIZE,), "int32"),
        sm_scale_buf: T.Tensor((1,), "float32"),
        dq: T.Tensor((Q_SIZE,), "float16"),
        dkv_partial: T.Tensor((DKV_PARTIAL_SIZE,), "float16"),
    ):
        with T.Kernel(LANES, threads=THREADS) as bx:
            lane = T.get_thread_binding()
            scores = T.alloc_shared((TOPK,), "float32", scope="shared")
            p = T.alloc_shared((TOPK,), "float32", scope="shared")
            dp = T.alloc_shared((TOPK,), "float32", scope="shared")
            ds = T.alloc_shared((TOPK,), "float32", scope="shared")
            reduce_buf = T.alloc_shared((THREADS,), "float32", scope="shared")
            acc = T.alloc_local((1,), "float32")
            local = T.alloc_local((1,), "float32")
            inv_sum = T.alloc_local((1,), "float32")
            stride = T.alloc_local((1,), "int32")
            gather_idx = T.alloc_local((1,), "int32")

            h = bx % HEADS
            b = bx // (HEADS * SEQ_LEN)
            g = h // HEAD_KV
            q_row_base = bx * QK_DIM
            d_out_row = bx * D_V
            kv_b_base = b * (SEQ_LEN_KV * KV_GROUP * QK_DIM)
            idx_base = ((bx // HEADS) * KV_GROUP + g) * TOPK
            dkv_partial_base = bx * TOPK * QK_DIM
            sm_scale = sm_scale_buf[0]

            for k in T.serial(lane, TOPK, step=THREADS):
                gather_idx[0] = indices[idx_base + k]
                if gather_idx[0] < 0:
                    scores[k] = T.float32(-1.0e38)
                else:
                    acc[0] = 0.0
                    kv_row_base = kv_b_base + (gather_idx[0] * KV_GROUP + g) * QK_DIM
                    for d in T.serial(QK_DIM):
                        acc[0] = acc[0] + T.cast(q[q_row_base + d], "float32") * T.cast(
                            kv[kv_row_base + d], "float32"
                        )
                    scores[k] = acc[0] * sm_scale
            T.sync_threads()

            local[0] = T.float32(-1.0e38)
            for k in T.serial(lane, TOPK, step=THREADS):
                if scores[k] > local[0]:
                    local[0] = scores[k]
            reduce_buf[lane] = local[0]
            T.sync_threads()
            for round_id in T.serial(LOG_THREADS):
                stride[0] = T.shift_right(THREADS, round_id + 1)
                if lane < stride[0]:
                    if reduce_buf[lane + stride[0]] > reduce_buf[lane]:
                        reduce_buf[lane] = reduce_buf[lane + stride[0]]
                T.sync_threads()
            row_max = reduce_buf[0]

            for k in T.serial(lane, TOPK, step=THREADS):
                gather_idx[0] = indices[idx_base + k]
                if gather_idx[0] < 0:
                    p[k] = 0.0
                else:
                    p[k] = T.exp(scores[k] - row_max)
            T.sync_threads()

            local[0] = 0.0
            for k in T.serial(lane, TOPK, step=THREADS):
                local[0] = local[0] + p[k]
            reduce_buf[lane] = local[0]
            T.sync_threads()
            for round_id in T.serial(LOG_THREADS):
                stride[0] = T.shift_right(THREADS, round_id + 1)
                if lane < stride[0]:
                    reduce_buf[lane] = reduce_buf[lane] + reduce_buf[lane + stride[0]]
                T.sync_threads()
            sumexp = reduce_buf[0]
            inv_sum[0] = 0.0
            if sumexp > 0.0:
                inv_sum[0] = 1.0 / sumexp

            for k in T.serial(lane, TOPK, step=THREADS):
                p[k] = p[k] * inv_sum[0]
            T.sync_threads()

            for k in T.serial(lane, TOPK, step=THREADS):
                gather_idx[0] = indices[idx_base + k]
                if gather_idx[0] < 0:
                    dp[k] = 0.0
                else:
                    acc[0] = 0.0
                    kv_row_base = kv_b_base + (gather_idx[0] * KV_GROUP + g) * QK_DIM
                    for d in T.serial(D_V):
                        acc[0] = acc[0] + T.cast(
                            kv[kv_row_base + d], "float32"
                        ) * T.cast(d_out[d_out_row + d], "float32")
                    dp[k] = acc[0]
            T.sync_threads()

            local[0] = 0.0
            for k in T.serial(lane, TOPK, step=THREADS):
                local[0] = local[0] + p[k] * dp[k]
            reduce_buf[lane] = local[0]
            T.sync_threads()
            for round_id in T.serial(LOG_THREADS):
                stride[0] = T.shift_right(THREADS, round_id + 1)
                if lane < stride[0]:
                    reduce_buf[lane] = reduce_buf[lane] + reduce_buf[lane + stride[0]]
                T.sync_threads()
            rowsum = reduce_buf[0]

            for k in T.serial(lane, TOPK, step=THREADS):
                ds[k] = p[k] * (dp[k] - rowsum)
            T.sync_threads()

            for d in T.serial(lane, QK_DIM, step=THREADS):
                acc[0] = 0.0
                for k in T.serial(TOPK):
                    gather_idx[0] = indices[idx_base + k]
                    if gather_idx[0] >= 0:
                        kv_row_base = kv_b_base + (gather_idx[0] * KV_GROUP + g) * QK_DIM
                        acc[0] = acc[0] + ds[k] * T.cast(
                            kv[kv_row_base + d], "float32"
                        )
                dq[q_row_base + d] = acc[0] * sm_scale

            for kd in T.serial(lane, TOPK * QK_DIM, step=THREADS):
                k = kd // QK_DIM
                d = kd % QK_DIM
                gather_idx[0] = indices[idx_base + k]
                if gather_idx[0] < 0:
                    dkv_partial[dkv_partial_base + kd] = 0.0
                else:
                    acc[0] = sm_scale * ds[k] * T.cast(q[q_row_base + d], "float32")
                    if d < D_V:
                        dkv_partial[dkv_partial_base + kd] = p[k] * T.cast(
                            d_out[d_out_row + d], "float32"
                        ) + acc[0]
                    else:
                        dkv_partial[dkv_partial_base + kd] = acc[0]

    return sparse_mla_bwd


@lru_cache(maxsize=128)
def _bwd_kernel_engine_for(
    BATCH: int,
    SEQ_LEN: int,
    HEADS: int,
    QK_DIM: int,
    KV_GROUP: int,
    HEAD_KV: int,
    TOPK: int,
    SEQ_LEN_KV: int,
    D_V: int,
    THREADS: int,
    target: str = "metal",
) -> Any:
    """Build the Sparse-MLA bwd kernel through the unified engine dispatcher.

    See :func:`_fwd_kernel_engine_for` for engine/shim semantics. The legacy
    :func:`_bwd_kernel_for` is preserved verbatim for the MLX runtime path.
    """

    from cppmega_mlx.nn._tilelang._engine_dispatch import dispatch_lower

    prim = _make_sparse_mla_bwd_prim(
        BATCH=BATCH,
        SEQ_LEN=SEQ_LEN,
        HEADS=HEADS,
        QK_DIM=QK_DIM,
        KV_GROUP=KV_GROUP,
        HEAD_KV=HEAD_KV,
        TOPK=TOPK,
        SEQ_LEN_KV=SEQ_LEN_KV,
        D_V=D_V,
        THREADS=THREADS,
    )
    return dispatch_lower(prim, target)


def lower_sparse_mla_bwd_msl(
    *,
    BATCH: int,
    SEQ_LEN: int,
    HEADS: int,
    QK_DIM: int,
    KV_GROUP: int,
    HEAD_KV: int,
    TOPK: int,
    SEQ_LEN_KV: int,
    D_V: int,
    THREADS: int,
    target: str = "metal",
) -> str:
    """Lower the Path C Sparse-MLA bwd kernel and return rendered source.

    Routes through :func:`_engine_dispatch.dispatch_lower`; see
    :func:`lower_sparse_mla_fwd_msl` for engine/shim semantics. Used by parity
    tests in ``cppmega/tests/test_sparse_mla_path_c_engine.py``.
    """

    from cppmega_mlx.nn._tilelang._engine_dispatch import (
        artifact_to_source,
        dispatch_lower,
    )

    prim = _make_sparse_mla_bwd_prim(
        BATCH=BATCH,
        SEQ_LEN=SEQ_LEN,
        HEADS=HEADS,
        QK_DIM=QK_DIM,
        KV_GROUP=KV_GROUP,
        HEAD_KV=HEAD_KV,
        TOPK=TOPK,
        SEQ_LEN_KV=SEQ_LEN_KV,
        D_V=D_V,
        THREADS=THREADS,
    )
    artifact = dispatch_lower(prim, target)
    return artifact_to_source(artifact)


@lru_cache(maxsize=128)
def _bwd_kernel_for(
    BATCH: int,
    SEQ_LEN: int,
    HEADS: int,
    QK_DIM: int,
    KV_GROUP: int,
    HEAD_KV: int,
    TOPK: int,
    SEQ_LEN_KV: int,
    D_V: int,
    THREADS: int,
) -> tuple[Any, _msl_transform.TileLangMSLLowering]:
    """Build and cache a shape-specialized threadgroup Sparse-MLA bwd kernel."""

    import tilelang.language as T

    LANES = BATCH * SEQ_LEN * HEADS
    Q_SIZE = BATCH * SEQ_LEN * HEADS * QK_DIM
    KV_SIZE = BATCH * SEQ_LEN_KV * KV_GROUP * QK_DIM
    DOUT_SIZE = BATCH * SEQ_LEN * HEADS * D_V
    IDX_SIZE = BATCH * SEQ_LEN * KV_GROUP * TOPK
    DKV_PARTIAL_SIZE = BATCH * SEQ_LEN * HEADS * TOPK * QK_DIM
    LOG_THREADS = THREADS.bit_length() - 1

    @T.prim_func
    def sparse_mla_bwd(
        q: T.Tensor((Q_SIZE,), "float16"),
        kv: T.Tensor((KV_SIZE,), "float16"),
        d_out: T.Tensor((DOUT_SIZE,), "float16"),
        indices: T.Tensor((IDX_SIZE,), "int32"),
        sm_scale_buf: T.Tensor((1,), "float32"),
        dq: T.Tensor((Q_SIZE,), "float16"),
        dkv_partial: T.Tensor((DKV_PARTIAL_SIZE,), "float16"),
    ):
        with T.Kernel(LANES, threads=THREADS) as bx:
            lane = T.get_thread_binding()
            scores = T.alloc_shared((TOPK,), "float32", scope="shared")
            p = T.alloc_shared((TOPK,), "float32", scope="shared")
            dp = T.alloc_shared((TOPK,), "float32", scope="shared")
            ds = T.alloc_shared((TOPK,), "float32", scope="shared")
            reduce_buf = T.alloc_shared((THREADS,), "float32", scope="shared")
            acc = T.alloc_local((1,), "float32")
            local = T.alloc_local((1,), "float32")
            inv_sum = T.alloc_local((1,), "float32")
            stride = T.alloc_local((1,), "int32")
            gather_idx = T.alloc_local((1,), "int32")

            h = bx % HEADS
            b = bx // (HEADS * SEQ_LEN)
            g = h // HEAD_KV
            q_row_base = bx * QK_DIM
            d_out_row = bx * D_V
            kv_b_base = b * (SEQ_LEN_KV * KV_GROUP * QK_DIM)
            idx_base = ((bx // HEADS) * KV_GROUP + g) * TOPK
            dkv_partial_base = bx * TOPK * QK_DIM
            sm_scale = sm_scale_buf[0]

            for k in T.serial(lane, TOPK, step=THREADS):
                gather_idx[0] = indices[idx_base + k]
                if gather_idx[0] < 0:
                    scores[k] = T.float32(-1.0e38)
                else:
                    acc[0] = 0.0
                    kv_row_base = kv_b_base + (gather_idx[0] * KV_GROUP + g) * QK_DIM
                    for d in T.serial(QK_DIM):
                        acc[0] = acc[0] + T.cast(q[q_row_base + d], "float32") * T.cast(
                            kv[kv_row_base + d], "float32"
                        )
                    scores[k] = acc[0] * sm_scale
            T.sync_threads()

            local[0] = T.float32(-1.0e38)
            for k in T.serial(lane, TOPK, step=THREADS):
                if scores[k] > local[0]:
                    local[0] = scores[k]
            reduce_buf[lane] = local[0]
            T.sync_threads()
            for round_id in T.serial(LOG_THREADS):
                stride[0] = T.shift_right(THREADS, round_id + 1)
                if lane < stride[0]:
                    if reduce_buf[lane + stride[0]] > reduce_buf[lane]:
                        reduce_buf[lane] = reduce_buf[lane + stride[0]]
                T.sync_threads()
            row_max = reduce_buf[0]

            for k in T.serial(lane, TOPK, step=THREADS):
                gather_idx[0] = indices[idx_base + k]
                if gather_idx[0] < 0:
                    p[k] = 0.0
                else:
                    p[k] = T.exp(scores[k] - row_max)
            T.sync_threads()

            local[0] = 0.0
            for k in T.serial(lane, TOPK, step=THREADS):
                local[0] = local[0] + p[k]
            reduce_buf[lane] = local[0]
            T.sync_threads()
            for round_id in T.serial(LOG_THREADS):
                stride[0] = T.shift_right(THREADS, round_id + 1)
                if lane < stride[0]:
                    reduce_buf[lane] = reduce_buf[lane] + reduce_buf[lane + stride[0]]
                T.sync_threads()
            sumexp = reduce_buf[0]
            inv_sum[0] = 0.0
            if sumexp > 0.0:
                inv_sum[0] = 1.0 / sumexp

            for k in T.serial(lane, TOPK, step=THREADS):
                p[k] = p[k] * inv_sum[0]
            T.sync_threads()

            for k in T.serial(lane, TOPK, step=THREADS):
                gather_idx[0] = indices[idx_base + k]
                if gather_idx[0] < 0:
                    dp[k] = 0.0
                else:
                    acc[0] = 0.0
                    kv_row_base = kv_b_base + (gather_idx[0] * KV_GROUP + g) * QK_DIM
                    for d in T.serial(D_V):
                        acc[0] = acc[0] + T.cast(
                            kv[kv_row_base + d], "float32"
                        ) * T.cast(d_out[d_out_row + d], "float32")
                    dp[k] = acc[0]
            T.sync_threads()

            local[0] = 0.0
            for k in T.serial(lane, TOPK, step=THREADS):
                local[0] = local[0] + p[k] * dp[k]
            reduce_buf[lane] = local[0]
            T.sync_threads()
            for round_id in T.serial(LOG_THREADS):
                stride[0] = T.shift_right(THREADS, round_id + 1)
                if lane < stride[0]:
                    reduce_buf[lane] = reduce_buf[lane] + reduce_buf[lane + stride[0]]
                T.sync_threads()
            rowsum = reduce_buf[0]

            for k in T.serial(lane, TOPK, step=THREADS):
                ds[k] = p[k] * (dp[k] - rowsum)
            T.sync_threads()

            for d in T.serial(lane, QK_DIM, step=THREADS):
                acc[0] = 0.0
                for k in T.serial(TOPK):
                    gather_idx[0] = indices[idx_base + k]
                    if gather_idx[0] >= 0:
                        kv_row_base = kv_b_base + (gather_idx[0] * KV_GROUP + g) * QK_DIM
                        acc[0] = acc[0] + ds[k] * T.cast(
                            kv[kv_row_base + d], "float32"
                        )
                dq[q_row_base + d] = acc[0] * sm_scale

            for kd in T.serial(lane, TOPK * QK_DIM, step=THREADS):
                k = kd // QK_DIM
                d = kd % QK_DIM
                gather_idx[0] = indices[idx_base + k]
                if gather_idx[0] < 0:
                    dkv_partial[dkv_partial_base + kd] = 0.0
                else:
                    acc[0] = sm_scale * ds[k] * T.cast(q[q_row_base + d], "float32")
                    if d < D_V:
                        dkv_partial[dkv_partial_base + kd] = p[k] * T.cast(
                            d_out[d_out_row + d], "float32"
                        ) + acc[0]
                    else:
                        dkv_partial[dkv_partial_base + kd] = acc[0]

    lowering = cast(
        _msl_transform.TileLangMSLLowering,
        dispatch_lower(sparse_mla_bwd, target="metal", return_msl=True),
    )
    lowering = _msl_transform.TileLangMSLLowering(
        header=lowering.header,
        body=_postprocess_lowered_msl(
            lowering.body,
            seq_len_kv=SEQ_LEN_KV,
            remove_flat_kv_bounds=True,
            canonicalize_bwd=True,
            topk=TOPK,
            heads=HEADS,
            seq_len=SEQ_LEN,
            kv_group=KV_GROUP,
            head_kv=HEAD_KV,
            seq_len_kv_for_indexing=SEQ_LEN_KV,
            qk_dim=QK_DIM,
            d_v=D_V,
            threads=THREADS,
        ),
        grid=lowering.grid,
        threadgroup=lowering.threadgroup,
        msl_text=_postprocess_lowered_msl(
            lowering.msl_text,
            seq_len_kv=SEQ_LEN_KV,
            remove_flat_kv_bounds=True,
            canonicalize_bwd=True,
            topk=TOPK,
            heads=HEADS,
            seq_len=SEQ_LEN,
            kv_group=KV_GROUP,
            head_kv=HEAD_KV,
            seq_len_kv_for_indexing=SEQ_LEN_KV,
            qk_dim=QK_DIM,
            d_v=D_V,
            threads=THREADS,
        ),
        buffer_param_names=lowering.buffer_param_names,
        kernel_name=lowering.kernel_name,
    )
    kernel = mx.fast.metal_kernel(
        name=(
            "cppmega_sparse_mla_path_c_bwd_noguard_"
            f"{BATCH}_{SEQ_LEN}_{HEADS}_{QK_DIM}_{KV_GROUP}_{TOPK}_{SEQ_LEN_KV}_{D_V}_{THREADS}"
        ),
        input_names=["d_out", "indices", "kv", "q", "sm_scale_buf"],
        output_names=["dkv_partial", "dq"],
        source=lowering.body,
        header=lowering.header,
        ensure_row_contiguous=True,
    )
    return kernel, lowering


def _validate_fwd_owner_output_buffers(
    q: mx.array,
    kv: mx.array,
    indices: mx.array,
    sm_scale_buf: mx.array,
    out: mx.array,
    lse: mx.array,
    *,
    d_v: int | None,
) -> Any:
    shapes = _resolve_shapes(q, kv, indices, d_v=d_v)
    mx_bfloat16 = getattr(mx, "bfloat16", None)
    if mx_bfloat16 is not None and (q.dtype == mx_bfloat16 or kv.dtype == mx_bfloat16):
        raise SparseMLAPathCDirectError(
            "Sparse-MLA Path C BF16 owner-output is fail-closed: the current "
            "TileLang forward PrimFunc is fp16-carrier, and the legacy BF16 "
            "Path C wrapper reaches it by hidden q/kv casts plus an allocating "
            "mx.fast.metal_kernel output. Provide fp16-carrier buffers produced "
            "upstream, or parameterize/fix the TileLang Metal BF16 ABI first."
        )
    if q.dtype != mx.float16 or kv.dtype != mx.float16:
        raise SparseMLAPathCDirectError(
            "Sparse-MLA Path C owner-output currently requires fp16-carrier "
            f"q/kv without hidden casts; got q={q.dtype}, kv={kv.dtype}"
        )
    if indices.dtype != mx.int32:
        raise SparseMLAPathCDirectError(
            f"Sparse-MLA Path C owner-output requires int32 indices; got {indices.dtype}"
        )
    if sm_scale_buf.shape != (1,) or sm_scale_buf.dtype != mx.float32:
        raise SparseMLAPathCDirectError(
            "Sparse-MLA Path C owner-output requires caller-owned "
            f"sm_scale_buf with shape (1,) and dtype mx.float32; got "
            f"shape={tuple(sm_scale_buf.shape)} dtype={sm_scale_buf.dtype}"
        )
    expected_out = (shapes.batch, shapes.seq_len, shapes.heads, shapes.d_v)
    expected_lse = (shapes.batch, shapes.seq_len, shapes.heads)
    if out.shape != expected_out or out.dtype != mx.float16:
        raise SparseMLAPathCDirectError(
            "Sparse-MLA Path C owner-output requires out with "
            f"shape={expected_out} and dtype mx.float16; got "
            f"shape={tuple(out.shape)} dtype={out.dtype}"
        )
    if lse.shape != expected_lse or lse.dtype != mx.float32:
        raise SparseMLAPathCDirectError(
            "Sparse-MLA Path C owner-output requires lse with "
            f"shape={expected_lse} and dtype mx.float32; got "
            f"shape={tuple(lse.shape)} dtype={lse.dtype}"
        )
    return shapes


@lru_cache(maxsize=128)
def _fwd_direct_tvm_ffi_kernel_for(
    BATCH: int,
    SEQ_LEN: int,
    HEADS: int,
    QK_DIM: int,
    KV_GROUP: int,
    HEAD_KV: int,
    TOPK: int,
    SEQ_LEN_KV: int,
    D_V: int,
    THREADS: int,
) -> Any:
    """Build and cache the owner-output tvm-ffi Sparse-MLA fwd kernel."""

    import tilelang

    prim = _make_sparse_mla_fwd_direct_prim(
        BATCH=BATCH,
        SEQ_LEN=SEQ_LEN,
        HEADS=HEADS,
        QK_DIM=QK_DIM,
        KV_GROUP=KV_GROUP,
        HEAD_KV=HEAD_KV,
        TOPK=TOPK,
        SEQ_LEN_KV=SEQ_LEN_KV,
        D_V=D_V,
        THREADS=THREADS,
    )
    return tilelang.compile(
        prim,
        target=_msl_transform._as_metal_target("metal"),
        execution_backend="tvm_ffi",
        out_idx=[2, 3],
    )


def sparse_mla_fwd_path_c_direct(
    q: mx.array,
    kv: mx.array,
    indices: mx.array,
    *,
    sm_scale_buf: mx.array,
    out: mx.array,
    lse: mx.array,
    d_v: int | None = None,
) -> tuple[mx.array, mx.array]:
    """Run the fp16-carrier owner-output tvm-ffi forward route."""

    status = sparse_mla_path_c_status()
    if not status.available:
        raise SparseMLAPathCDirectError(status.reason)
    shapes = _validate_fwd_owner_output_buffers(
        q,
        kv,
        indices,
        sm_scale_buf,
        out,
        lse,
        d_v=d_v,
    )
    threads = _threadgroup_size(shapes.topk)
    try:
        kernel = _fwd_direct_tvm_ffi_kernel_for(
            shapes.batch,
            shapes.seq_len,
            shapes.heads,
            shapes.qk_dim,
            shapes.kv_group,
            shapes.head_kv,
            shapes.topk,
            shapes.seq_len_kv,
            shapes.d_v,
            threads,
        )
    except Exception as exc:
        raise SparseMLAPathCDirectError(
            f"Sparse-MLA Path C owner-output tvm-ffi compile failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    tensor_list = [indices, kv, lse, out, q, sm_scale_buf]
    expected_dtypes = ["int32", "float16", "float32", "float16", "float16", "float32"]
    try:
        from tilelang.contrib.mlx_interop import (
            DLPackInteropError,
            maybe_mlx_metal_external_command_buffer,
            mlx_arrays_to_tvm_tensors,
        )

        executable = getattr(kernel.adapter, "executable", None)
        if executable is None:
            raise SparseMLAPathCDirectError(
                "Sparse-MLA Path C owner-output tvm-ffi dispatch failed: "
                "compiled kernel did not expose a TVM executable"
            )
        exec_tensor_list = mlx_arrays_to_tvm_tensors(
            tensor_list,
            expected_dtypes=expected_dtypes,
        )
        with maybe_mlx_metal_external_command_buffer(tensor_list):
            executable(*exec_tensor_list)
    except Exception as exc:
        try:
            from tilelang.contrib.mlx_interop import DLPackInteropError
        except Exception:  # pragma: no cover - only when TileLang import itself is broken
            DLPackInteropError = ()  # type: ignore[assignment]
        if isinstance(exc, DLPackInteropError):
            raise
        raise SparseMLAPathCDirectError(
            f"Sparse-MLA Path C owner-output tvm-ffi dispatch failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    mx.synchronize()
    return out, lse


def sparse_mla_fwd_path_c(
    q: mx.array,
    kv: mx.array,
    indices: mx.array,
    *,
    sm_scale: float | None = None,
    d_v: int | None = None,
    sm_scale_buf: mx.array | None = None,
    out: mx.array | None = None,
    lse: mx.array | None = None,
) -> tuple[mx.array, mx.array] | None:
    """TileLang DSL Path C Sparse-MLA forward.

    Returns ``(out, lse)`` or ``None`` if the Metal/TileLang path cannot be
    built. The kernel mirrors Path B's raw forward contract: fp16 carrier I/O
    with fp32 accumulators and fp32 ``lse``.
    """

    owner_args = (sm_scale_buf, out, lse)
    if any(arg is not None for arg in owner_args):
        if sm_scale_buf is None or out is None or lse is None:
            raise SparseMLAPathCDirectError(
                "Sparse-MLA Path C owner-output requires sm_scale_buf, out, "
                "and lse together; missing buffers would force hidden wrapper "
                "allocation."
            )
        return sparse_mla_fwd_path_c_direct(
            q,
            kv,
            indices,
            sm_scale_buf=sm_scale_buf,
            out=out,
            lse=lse,
            d_v=d_v,
        )

    status = sparse_mla_path_c_status()
    if not status.available:
        return None

    shapes = _resolve_shapes(q, kv, indices, d_v=d_v)
    if sm_scale is None:
        sm_scale_value = shapes.qk_dim ** -0.5
    else:
        sm_scale_value = sm_scale
    threads = _threadgroup_size(shapes.topk)

    indices_i32 = _require_int32_indices_no_hidden_cast(
        indices,
        op_name="sparse_mla_fwd_path_c",
    )
    q16 = _promote_to_fp16_carrier(q)
    kv16 = _promote_to_fp16_carrier(kv)
    sm_scale_buf = mx.array([float(sm_scale_value)], dtype=mx.float32)

    try:
        kernel, lowering = _fwd_kernel_for(
            shapes.batch,
            shapes.seq_len,
            shapes.heads,
            shapes.qk_dim,
            shapes.kv_group,
            shapes.head_kv,
            shapes.topk,
            shapes.seq_len_kv,
            shapes.d_v,
            threads,
        )
    except (MSLDispatchUnsupported, RuntimeError, ValueError):
        return None

    grid = _mlx_total_thread_grid(lowering)

    try:
        outputs = kernel(
            inputs=[q16, kv16, indices_i32, sm_scale_buf],
            template=None,
            output_shapes=[
                (shapes.batch, shapes.seq_len, shapes.heads, shapes.d_v),
                (shapes.batch, shapes.seq_len, shapes.heads),
            ],
            output_dtypes=[mx.float16, mx.float32],
            grid=grid,
            threadgroup=lowering.threadgroup,
            stream=mx.gpu,
        )
    except Exception:
        return None

    out, lse = outputs
    return cast(mx.array, out), cast(mx.array, lse)


def _sparse_mla_bwd_path_c_partial(
    q: mx.array,
    kv: mx.array,
    d_out: mx.array,
    indices: mx.array,
    *,
    sm_scale: float | None = None,
    d_v: int | None = None,
) -> tuple[mx.array, mx.array, mx.array, Any] | None:
    """Run the TileLang backward kernel and return unreduced dKV partials.

    This is intentionally private and exists so the benchmark can isolate the
    TileLang kernel cost from the shared Path B/Path C dKV scatter-reduction.
    """

    status = sparse_mla_path_c_status()
    if not status.available:
        return None

    shapes = _resolve_shapes(q, kv, indices, d_v=d_v)
    if sm_scale is None:
        sm_scale_value = shapes.qk_dim ** -0.5
    else:
        sm_scale_value = sm_scale
    threads = _threadgroup_size(shapes.topk)

    indices_i32 = _require_int32_indices_no_hidden_cast(
        indices,
        op_name="_sparse_mla_bwd_path_c_partial",
    )
    q16 = _promote_to_fp16_carrier(q)
    kv16 = _promote_to_fp16_carrier(kv)
    d_out16 = _promote_to_fp16_carrier(d_out)
    sm_scale_buf = mx.array([float(sm_scale_value)], dtype=mx.float32)

    try:
        kernel, lowering = _bwd_kernel_for(
            shapes.batch,
            shapes.seq_len,
            shapes.heads,
            shapes.qk_dim,
            shapes.kv_group,
            shapes.head_kv,
            shapes.topk,
            shapes.seq_len_kv,
            shapes.d_v,
            threads,
        )
    except (MSLDispatchUnsupported, RuntimeError, ValueError):
        return None

    grid = _mlx_total_thread_grid(lowering)

    try:
        outputs = kernel(
            inputs=[d_out16, indices_i32, kv16, q16, sm_scale_buf],
            output_shapes=[
                (
                    shapes.batch,
                    shapes.seq_len,
                    shapes.heads,
                    shapes.topk,
                    shapes.qk_dim,
                ),
                (shapes.batch, shapes.seq_len, shapes.heads, shapes.qk_dim),
            ],
            output_dtypes=[mx.float16, mx.float16],
            grid=grid,
            threadgroup=lowering.threadgroup,
            stream=mx.gpu,
        )
    except Exception:
        return None

    dkv_partial, dq = outputs
    return cast(mx.array, dkv_partial), cast(mx.array, dq), indices_i32, shapes


def sparse_mla_bwd_path_c(
    q: mx.array,
    kv: mx.array,
    d_out: mx.array,
    indices: mx.array,
    *,
    sm_scale: float | None = None,
    d_v: int | None = None,
) -> tuple[mx.array, mx.array] | None:
    """TileLang DSL Path C Sparse-MLA backward.

    Returns ``(dq, dkv)`` or ``None`` if the Metal/TileLang path cannot be
    built. The kernel mirrors Path B's fp16 carrier/partial contract while
    keeping fp32 accumulators inside the TileLang kernel.
    """

    partial = _sparse_mla_bwd_path_c_partial(
        q,
        kv,
        d_out,
        indices,
        sm_scale=sm_scale,
        d_v=d_v,
    )
    if partial is None:
        return None

    dkv_partial, dq, indices_i32, shapes = partial
    dkv = _reduce_dkv_partial(dkv_partial, indices_i32, shapes)
    return cast(mx.array, dq), cast(mx.array, dkv)


@mx.custom_function
def sparse_mla_path_c_metal_apply(
    q: mx.array,
    kv: mx.array,
    indices: mx.array,
) -> mx.array:
    """Differentiable default-parameter wrapper for Path C Sparse-MLA."""

    result = sparse_mla_fwd_path_c(q, kv, indices)
    if result is None:
        return cast(mx.array, sparse_mla_attention_reference(q, kv, indices))
    out, _lse = result
    return out


_sparse_mla_path_c_metal_apply_any = cast(Any, sparse_mla_path_c_metal_apply)


@_sparse_mla_path_c_metal_apply_any.vjp
def _sparse_mla_path_c_metal_apply_vjp(primals, cotangent, output):
    del output
    q, kv, indices = primals
    grads = sparse_mla_bwd_path_c(q, kv, cotangent, indices)
    if grads is None:
        def _reference_apply(q_, kv_):
            return sparse_mla_attention_reference(q_, kv_, indices)

        _, vjps = mx.vjp(_reference_apply, [q, kv], [cotangent])
        return (vjps[0], vjps[1], mx.zeros_like(indices))
    dq, dkv = grads
    return (dq.astype(q.dtype), dkv.astype(kv.dtype), mx.zeros_like(indices))


@lru_cache(maxsize=128)
def _sparse_mla_path_c_apply_for_params(sm_scale: float, d_v: int) -> Any:
    """Build a custom VJP wrapper for one non-default Path C parameter set."""

    @mx.custom_function
    def _apply(q: mx.array, kv: mx.array, indices: mx.array) -> mx.array:
        result = sparse_mla_fwd_path_c(q, kv, indices, sm_scale=sm_scale, d_v=d_v)
        if result is None:
            return cast(
                mx.array,
                sparse_mla_attention_reference(q, kv, indices, sm_scale=sm_scale, d_v=d_v),
            )
        out, _lse = result
        return out

    apply_any = cast(Any, _apply)

    @apply_any.vjp
    def _apply_vjp(primals, cotangent, output):
        del output
        q, kv, indices = primals
        grads = sparse_mla_bwd_path_c(
            q,
            kv,
            cotangent,
            indices,
            sm_scale=sm_scale,
            d_v=d_v,
        )
        if grads is None:
            def _reference_apply(q_, kv_):
                return sparse_mla_attention_reference(
                    q_,
                    kv_,
                    indices,
                    sm_scale=sm_scale,
                    d_v=d_v,
                )

            _, vjps = mx.vjp(_reference_apply, [q, kv], [cotangent])
            return (vjps[0], vjps[1], mx.zeros_like(indices))
        dq, dkv = grads
        return (dq.astype(q.dtype), dkv.astype(kv.dtype), mx.zeros_like(indices))

    return apply_any


def sparse_mla_path_c_apply(
    q: mx.array,
    kv: mx.array,
    indices: mx.array,
    *,
    sm_scale: float | None = None,
    d_v: int | None = None,
    return_lse: bool = False,
    force_path_c: bool = False,
) -> mx.array | tuple[mx.array, mx.array]:
    """Apply Sparse-MLA through the TileLang DSL Path C Metal kernel.

    The default ``sm_scale``/``d_v`` path is wrapped in ``mx.custom_function``
    and uses the Path C backward kernel for VJP coverage. Forced Path C
    non-default ``d_v``/``sm_scale`` dispatch uses a shape-parameterized custom
    VJP wrapper over the same forward/backward kernels.

    Note (kwarg rename from Path B):
        This entrypoint accepts ``force_path_c`` (raise instead of falling
        back when the Path C surface is unavailable). The corresponding Path B
        wrapper ``sparse_mla_apply`` uses ``force_metal``. The rename is
        intentional — there is no backwards-compatible ``force_metal`` alias
        on Path C, so callers migrating from Path B must rename the kwarg.
        AUTO-routed callers do not see this kwarg directly. See
        ``docs/production_kernel_routing.md``.
    """

    shapes = _resolve_shapes(q, kv, indices, d_v=d_v)
    if sm_scale is None:
        sm_scale = shapes.qk_dim ** -0.5

    if return_lse:
        result = sparse_mla_fwd_path_c(q, kv, indices, sm_scale=sm_scale, d_v=d_v)
        if result is None:
            if force_path_c:
                raise RuntimeError(
                    "sparse_mla_path_c_apply: Path C unavailable: "
                    f"{sparse_mla_path_c_status().reason}"
                )
            return sparse_mla_attention_reference(
                q,
                kv,
                indices,
                sm_scale=sm_scale,
                d_v=d_v,
                return_lse=True,
            )
        out, lse = result
        return out.astype(q.dtype), lse

    is_default = (
        d_v is None or d_v == shapes.qk_dim
    ) and abs(sm_scale - shapes.qk_dim ** -0.5) < 1e-9
    if is_default:
        status = sparse_mla_path_c_status()
        if not status.available:
            if force_path_c:
                raise RuntimeError(
                    f"sparse_mla_path_c_apply: Path C unavailable: {status.reason}"
                )
            return sparse_mla_attention_reference(
                q,
                kv,
                indices,
                sm_scale=sm_scale,
                d_v=d_v,
                return_lse=False,
            )
        out = sparse_mla_path_c_metal_apply(q, kv, indices)
        return cast(mx.array, out).astype(q.dtype)

    status = sparse_mla_path_c_status()
    if status.available:
        apply = _sparse_mla_path_c_apply_for_params(float(sm_scale), shapes.d_v)
        out = apply(q, kv, indices)
        return cast(mx.array, out).astype(q.dtype)
    if force_path_c:
        raise RuntimeError(
            f"sparse_mla_path_c_apply: Path C unavailable: {status.reason}"
        )
    return sparse_mla_attention_reference(
        q,
        kv,
        indices,
        sm_scale=sm_scale,
        d_v=d_v,
        return_lse=False,
    )


def dump_lowered_fwd_msl(
    *,
    batch: int,
    seq_len: int,
    heads: int,
    qk_dim: int,
    kv_group: int,
    topk: int,
    seq_len_kv: int,
    d_v: int | None = None,
) -> str:
    """Return raw lowered forward MSL for inspection/benchmark artifacts."""

    if d_v is None:
        d_v = qk_dim
    head_kv = heads // kv_group
    threads = _threadgroup_size(topk)
    _kernel, lowering = _fwd_kernel_for(
        batch,
        seq_len,
        heads,
        qk_dim,
        kv_group,
        head_kv,
        topk,
        seq_len_kv,
        d_v,
        threads,
    )
    return cast(str, lowering.msl_text)


def dump_lowered_bwd_msl(
    *,
    batch: int,
    seq_len: int,
    heads: int,
    qk_dim: int,
    kv_group: int,
    topk: int,
    seq_len_kv: int,
    d_v: int | None = None,
) -> str:
    """Return raw lowered MSL for inspection/benchmark artifacts."""

    if d_v is None:
        d_v = qk_dim
    head_kv = heads // kv_group
    threads = _threadgroup_size(topk)
    _kernel, lowering = _bwd_kernel_for(
        batch,
        seq_len,
        heads,
        qk_dim,
        kv_group,
        head_kv,
        topk,
        seq_len_kv,
        d_v,
        threads,
    )
    return cast(str, lowering.msl_text)


__all__ = [
    "SparseMLAPathCDirectError",
    "SparseMLAPathCStatus",
    "dump_lowered_bwd_msl",
    "dump_lowered_fwd_msl",
    "sparse_mla_bwd_path_c",
    "sparse_mla_fwd_path_c_direct",
    "sparse_mla_fwd_path_c",
    "sparse_mla_path_c_apply",
    "sparse_mla_path_c_metal_apply",
    "sparse_mla_path_c_status",
]

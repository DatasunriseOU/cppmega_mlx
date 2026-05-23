"""Regression tests for sparse_mla_fp8_apply emit-indent fixes.

These tests pin the structural invariant that the lowered PrimFunc
source for the Mamba3 FP8 train block keeps the softmax score-max
update statements outside the per-key ``if/else`` branch in the
``sparse_mla_fp8_apply`` body. A previous version emitted the
update inside the ``else:`` branch (because of an indentation mistake
in :func:`cppmega_mlx.runtime.path_c_fusion_schedules._append_row_phased_sparse_mla_fp8_apply_body`),
which left ``score_max`` at ``-inf`` for every valid key and produced
``attention_out == 0`` end-to-end.

The fix changes the descriptor emit so the score-max update and the
softmax-weight accumulation are siblings of the ``if/else`` block.
These tests verify that:

1. Every ``if sparse_mla_fp8_apply_score_accum[0] > sparse_mla_fp8_apply_score_max[0]:``
   line in the rendered PrimFunc has the SAME indent as the matching
   ``if sparse_mla_fp8_apply_sparse_index[0] >= 0`` line.
2. Every ``sparse_mla_fp8_apply_score_weight[0] = T.exp(``
   line is at the SAME indent as the matching ``if sparse_index`` line
   (so the score weights accumulate over ALL keys, not just invalid ones).
3. Every ``sparse_mla_fp8_apply_sumexp[0] = sparse_mla_fp8_apply_sumexp[0]
   + T.exp(score_accum`` line is at the SAME indent as the matching
   ``if sparse_index`` line.
"""

from __future__ import annotations

import contextlib
import io

from cppmega_mlx.recipes.model_factory import local_gb10_quarter_profile
from cppmega_mlx.runtime import path_c_fusion_schedules as schedules


def _rendered_train_block_source() -> str:
    profile = local_gb10_quarter_profile()
    tiny = profile.tiny_smoke_config()
    with contextlib.redirect_stderr(io.StringIO()):
        region = schedules._mamba3_fp8_train_acceptance_region(
            include_backward=True, model_config=tiny,
        )
        prim_func = schedules.mamba3_fp8_train_fusion_schedule_template(region)
    return str(prim_func)


def _indent_of(line: str) -> int:
    """Return the number of leading spaces."""
    return len(line) - len(line.lstrip(" "))


def test_sparse_mla_apply_score_max_update_is_sibling_of_branch() -> None:
    """Every ``score_max = score_accum`` update must live outside the
    per-key ``if sparse_index ... < seq:`` branch so that valid keys can
    actually update the running max. Before the fix this lived inside
    the ``else:`` branch and the max stayed at ``-inf`` for every valid
    key, producing ``attention_out`` of all zeros."""

    src = _rendered_train_block_source()
    lines = src.split("\n")
    branch_indents: list[int] = []
    update_indents: list[int] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(
            "if sparse_mla_fp8_apply_sparse_index[0] >= 0 "
            "and sparse_mla_fp8_apply_sparse_index[0] < "
        ):
            branch_indents.append(_indent_of(line))
        if stripped.startswith(
            "if sparse_mla_fp8_apply_score_accum[0] > "
            "sparse_mla_fp8_apply_score_max[0]:"
        ):
            update_indents.append(_indent_of(line))
    # The body emits the apply twice: once in the lane-strided
    # context-and-out-projection branch (with sinks), once in the
    # lse-only branch. The lse-only branch ALSO emits the score-max
    # update. Whether the second emit fires depends on schedule config;
    # we only assert when at least one update is emitted.
    assert update_indents, "sparse_mla_fp8_apply score-max update missing from emit"
    # Each score-max update must match the indent of one of the
    # sparse_index branches.
    for indent in update_indents:
        assert indent in branch_indents, (
            f"sparse_mla_fp8_apply score-max update indent {indent} "
            f"is not aligned to any sparse_index branch indent {branch_indents}; "
            "this indicates the update was emitted inside the else-branch "
            "(the original Blocker 1 bug) instead of as a sibling"
        )


def test_sparse_mla_apply_score_weight_exp_is_sibling_of_branch() -> None:
    """``score_weight[0] = T.exp(score_accum - score_max)`` must live in
    the ``for k_top:`` loop body, outside the per-key
    ``if sparse_index ... < seq:`` branch. Otherwise ``score_weights``
    stays at 0.0 for every valid key (because the body only ran for
    invalid keys in the else branch), and the softmax-weighted value
    accumulator is identically zero."""

    src = _rendered_train_block_source()
    lines = src.split("\n")
    branch_indents: list[int] = []
    exp_indents: list[int] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(
            "if sparse_mla_fp8_apply_sparse_index[0] >= 0 "
            "and sparse_mla_fp8_apply_sparse_index[0] < "
        ):
            branch_indents.append(_indent_of(line))
        if stripped.startswith(
            "sparse_mla_fp8_apply_score_weight[0] = T.exp("
        ):
            exp_indents.append(_indent_of(line))
    assert exp_indents, "sparse_mla_fp8_apply score-weight exp missing from emit"
    for indent in exp_indents:
        assert indent in branch_indents, (
            f"sparse_mla_fp8_apply score-weight exp indent {indent} "
            f"is not aligned to any sparse_index branch indent {branch_indents}"
        )


def test_sparse_mla_apply_lse_sumexp_is_sibling_of_branch() -> None:
    """``sumexp[0] = sumexp[0] + T.exp(score_accum - score_max)`` in the
    lse-only path must live in the ``for k_top:`` loop body, outside
    the per-key ``if sparse_index`` branch. Otherwise the lse buffer
    stays at ``-inf`` for valid keys."""

    src = _rendered_train_block_source()
    lines = src.split("\n")
    branch_indents: list[int] = []
    sumexp_indents: list[int] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(
            "if sparse_mla_fp8_apply_sparse_index[0] >= 0 "
            "and sparse_mla_fp8_apply_sparse_index[0] < "
        ):
            branch_indents.append(_indent_of(line))
        if stripped.startswith(
            "sparse_mla_fp8_apply_sumexp[0] = "
            "sparse_mla_fp8_apply_sumexp[0] + T.exp("
            "sparse_mla_fp8_apply_score_accum"
        ):
            sumexp_indents.append(_indent_of(line))
    # The sumexp += T.exp(score_accum - score_max) form only exists
    # in the lse-only path; this test only asserts when it is present.
    for indent in sumexp_indents:
        assert indent in branch_indents, (
            f"sparse_mla_fp8_apply lse-only sumexp indent {indent} "
            f"is not aligned to any sparse_index branch indent {branch_indents}"
        )

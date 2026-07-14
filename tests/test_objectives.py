"""Tests for Stage-1 objective builders (cppmega_mlx.training.objectives)."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from cppmega_mlx.data.code_packet import CodePacket
from cppmega_mlx.data.commit_packet import CommitPacket
from cppmega_mlx.data.fim import EOT_ID, FIM_MIDDLE_ID, FIM_PREFIX_ID, FIM_SUFFIX_ID
from cppmega_mlx.training.objectives import (
    ObjectiveExample,
    build_ast_fim,
    build_causal_lm,
    build_commit_diff,
    build_ifim,
    build_pre_to_post,
    build_recovery,
)


def _arr(values: list[int]) -> mx.array:
    return mx.array(np.asarray(values, dtype=np.int32))


def _np(arr: mx.array) -> list[int]:
    return [int(x) for x in np.asarray(arr).reshape(-1).tolist()]


def _code_packet(tokens, chunks=None, **channels) -> CodePacket:
    kwargs = {}
    if chunks is not None:
        kwargs.update(
            chunk_starts=_arr([c[0] for c in chunks]),
            chunk_ends=_arr([c[1] for c in chunks]),
            chunk_kinds=_arr([1] * len(chunks)),
            chunk_dep_levels=_arr([0] * len(chunks)),
        )
    for k, v in channels.items():
        kwargs[k] = _arr(v)
    return CodePacket(token_ids=_arr(tokens), **kwargs)


def _assert_aligned(ex: ObjectiveExample) -> None:
    n = int(ex.input_ids.shape[0])
    assert int(ex.target_ids.shape[0]) == n
    assert int(ex.loss_mask.shape[0]) == n


# --------------------------- CAUSAL_LM --------------------------- #
def test_causal_lm_full_mask_and_shift() -> None:
    ex = build_causal_lm(_code_packet([5, 6, 7, 8]))
    _assert_aligned(ex)
    assert _np(ex.input_ids) == [5, 6, 7]
    assert _np(ex.target_ids) == [6, 7, 8]
    assert _np(ex.loss_mask) == [1, 1, 1]


# --------------------------- AST_FIM / IFIM ---------------------- #
def test_ast_fim_aligned_and_loss_on_middle() -> None:
    # Use token values above the special-id range so they never collide with the
    # FIM_MIDDLE/EOT markers we search for.
    tokens = [100 + i for i in range(10)]
    packet = _code_packet(tokens, chunks=[(0, 3), (3, 6), (6, 10)])
    ex = build_ast_fim(packet, seed=1, spm_rate=0.0)
    _assert_aligned(ex)
    # The trailing EOT must be a supervised target.
    targets = _np(ex.target_ids)
    mask = _np(ex.loss_mask)
    assert targets[-1] == EOT_ID
    assert mask[-1] == 1
    # No supervision before the FIM_MIDDLE marker; full supervision from it on.
    inputs = _np(ex.input_ids)
    mid_input_pos = inputs.index(FIM_MIDDLE_ID)
    assert all(m == 0 for m in mask[:mid_input_pos])
    assert all(m == 1 for m in mask[mid_input_pos:])


def test_ast_fim_character_fallback_is_not_telemetried_as_ast_fim() -> None:
    packet = _code_packet(list(range(10)), chunks=[(0, 3), (3, 6), (6, 10)])

    ex = build_ast_fim(packet, seed=3, ast_fim_rate=0.0)

    assert ex.objective == "fim"
    assert ex.metadata["fim_kind"] == "char_fim"


def test_ifim_aligned() -> None:
    packet = _code_packet(list(range(10)), chunks=[(3, 6)])
    packet = CodePacket(
        token_ids=packet.token_ids,
        chunk_starts=packet.chunk_starts,
        chunk_ends=packet.chunk_ends,
        chunk_kinds=packet.chunk_kinds,
        chunk_dep_levels=packet.chunk_dep_levels,
        ifim_instruction_token_ids=_arr([200, 201, 202]),
    )
    ex = build_ifim(packet, seed=1, spm_rate=0.0)
    _assert_aligned(ex)
    assert _np(ex.target_ids)[-1] == EOT_ID


# --------------------------- COMMIT_DIFF / PRE_TO_POST ----------- #
def test_commit_diff_loss_on_diff() -> None:
    packet = CommitPacket(
        commit_msg=_arr([100, 101]),
        diff_token_ids=_arr([50, 51, 52]),
    )
    ex = build_commit_diff(packet)
    _assert_aligned(ex)
    full = [17, 100, 101, 18, 15, 50, 51, 52, 16, EOT_ID]
    assert _np(ex.input_ids) == full[:-1]
    assert _np(ex.target_ids) == full[1:]
    assert _np(ex.loss_mask) == [0, 0, 0, 0, 1, 1, 1, 1, 1]
    assert ex.metadata["section_boundaries"] == {
        "commit_message": (0, 4),
        "diff": (4, 9),
    }


def test_pre_to_post_loss_on_post() -> None:
    packet = CommitPacket(
        pre_token_ids=_arr([10, 11]),
        post_token_ids=_arr([20, 21]),
        commit_msg=_arr([30]),
    )
    ex = build_pre_to_post(packet)
    _assert_aligned(ex)
    full = [7, 10, 11, 8, 17, 30, 18, 14, 7, 20, 21, 8, EOT_ID]
    assert _np(ex.input_ids) == full[:-1]
    assert _np(ex.target_ids) == full[1:]
    assert _np(ex.loss_mask) == [0] * 8 + [1] * 4
    assert ex.metadata["section_boundaries"] == {
        "pre": (0, 4),
        "commit_message": (4, 7),
        "post": (8, 12),
    }


# --------------------------- RECOVERY --------------------------- #
def test_symbol_recovery_masks_symbol_span() -> None:
    tokens = [10, 11, 700, 701, 12, 13]
    symbol_ids = [0, 0, 7, 7, 0, 0]  # span [2,4)
    packet = _code_packet(tokens, symbol_ids=symbol_ids)
    ex = build_recovery(packet, kind="symbol", seed=0)
    _assert_aligned(ex)
    full = [
        FIM_PREFIX_ID,
        38,
        10,
        11,
        FIM_SUFFIX_ID,
        12,
        13,
        FIM_MIDDLE_ID,
        700,
        701,
        EOT_ID,
    ]
    assert _np(ex.input_ids) == full[:-1]
    assert _np(ex.target_ids) == full[1:]
    answer_start = full.index(FIM_MIDDLE_ID) + 1
    assert 700 not in full[:answer_start]
    assert 701 not in full[:answer_start]
    assert _np(ex.loss_mask) == [0] * (answer_start - 1) + [1] * 3
    assert ex.metadata["span"] == (2, 4)
    assert ex.metadata["answer_start"] == answer_start


def test_type_recovery_uses_type_refs() -> None:
    packet = _code_packet([1, 2, 3, 4, 5], type_refs=[0, 8, 8, 0, 0])
    ex = build_recovery(packet, kind="type", seed=0)
    assert ex.objective == "type_recovery"
    assert ex.metadata["span"] == (1, 3)


def test_callee_recovery_uses_call_targets() -> None:
    packet = _code_packet([1, 2, 3, 4, 5], call_targets=[0, 0, 0, 5, 5])
    ex = build_recovery(packet, kind="callee", seed=0)
    assert ex.objective == "callee_recovery"
    assert ex.metadata["span"] == (3, 5)


# --------------------------- ABSENT-FIELD RAISES ---------------- #
def test_commit_diff_absent_diff_raises() -> None:
    packet = CommitPacket(commit_msg=_arr([1, 2]))
    with pytest.raises(ValueError, match="diff_token_ids"):
        build_commit_diff(packet)


def test_pre_to_post_absent_post_raises() -> None:
    packet = CommitPacket(pre_token_ids=_arr([1, 2]))
    with pytest.raises(ValueError, match="post_token_ids"):
        build_pre_to_post(packet)


def test_recovery_absent_channel_raises() -> None:
    packet = _code_packet([1, 2, 3])
    with pytest.raises(ValueError, match="symbol_ids channel is absent"):
        build_recovery(packet, kind="symbol", seed=0)


def test_recovery_all_zero_channel_raises() -> None:
    packet = _code_packet([1, 2, 3], symbol_ids=[0, 0, 0])
    with pytest.raises(ValueError, match="no non-zero recoverable span"):
        build_recovery(packet, kind="symbol", seed=0)


def test_ifim_absent_typed_instruction_raises() -> None:
    packet = _code_packet(list(range(10)), chunks=[(3, 6)])
    with pytest.raises(ValueError, match="ifim_instruction_token_ids"):
        build_ifim(packet, seed=0)

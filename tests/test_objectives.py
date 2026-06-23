"""Tests for Stage-1 objective builders (cppmega_mlx.training.objectives)."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from cppmega_mlx.data.code_packet import CodePacket
from cppmega_mlx.data.commit_packet import CommitPacket
from cppmega_mlx.data.fim import EOT_ID, FIM_MIDDLE_ID
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


def test_ifim_aligned() -> None:
    src = "// Sum all elements\nint sum(int *a, int n);\n"
    packet = _code_packet(list(range(10)), chunks=[(3, 6)], )
    packet = CodePacket(
        token_ids=packet.token_ids,
        chunk_starts=packet.chunk_starts,
        chunk_ends=packet.chunk_ends,
        chunk_kinds=packet.chunk_kinds,
        chunk_dep_levels=packet.chunk_dep_levels,
        metadata={"source_text": src},
    )
    ex = build_ifim(packet, instruction_encoder=lambda t: [200, 201, 202], seed=1, spm_rate=0.0)
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
    # full = [100,101,50,51,52,EOT]; inputs=full[:-1], targets=full[1:].
    assert _np(ex.input_ids) == [100, 101, 50, 51, 52]
    assert _np(ex.target_ids) == [101, 50, 51, 52, EOT_ID]
    # prompt_len=2 -> supervise targets predicting diff tokens (50,51,52)+EOT.
    # target j predicts full[j+1]; train iff j+1>=2 -> [0,1,1,1,1].
    assert _np(ex.loss_mask) == [0, 1, 1, 1, 1]


def test_pre_to_post_loss_on_post() -> None:
    packet = CommitPacket(
        pre_token_ids=_arr([10, 11]),
        post_token_ids=_arr([20, 21]),
    )
    ex = build_pre_to_post(packet)
    _assert_aligned(ex)
    # full = [10,11,20,21,EOT]; prompt_len=2.
    assert _np(ex.target_ids) == [11, 20, 21, EOT_ID]
    assert _np(ex.loss_mask) == [0, 1, 1, 1]


# --------------------------- RECOVERY --------------------------- #
def test_symbol_recovery_masks_symbol_span() -> None:
    tokens = [9, 9, 42, 42, 9, 9]
    symbol_ids = [0, 0, 7, 7, 0, 0]  # span [2,4)
    packet = _code_packet(tokens, symbol_ids=symbol_ids)
    ex = build_recovery(packet, kind="symbol", seed=0)
    _assert_aligned(ex)
    # targets = tokens[1:]; supervise target j iff predicted token j+1 in [2,4).
    # j=1 -> token2 (in), j=2 -> token3 (in); others out.
    assert _np(ex.loss_mask) == [0, 1, 1, 0, 0]
    assert ex.metadata["span"] == (2, 4)


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


def test_ifim_absent_source_text_raises() -> None:
    packet = _code_packet(list(range(10)), chunks=[(3, 6)])
    with pytest.raises(ValueError, match="source_text"):
        build_ifim(packet, instruction_encoder=lambda t: [1], seed=0)

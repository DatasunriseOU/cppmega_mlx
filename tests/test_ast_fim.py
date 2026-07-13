"""Tests for AST-aware FIM span selection (cppmega_mlx.data.ast_fim)."""

from __future__ import annotations

import random

import mlx.core as mx
import numpy as np
import pytest

from cppmega_mlx.data.ast_fim import (
    apply_ast_fim,
    apply_ast_ifim,
    select_ast_span,
)
from cppmega_mlx.data.code_packet import CodePacket
from cppmega_mlx.data.fim import (
    EOT_ID,
    FIM_MIDDLE_ID,
    FIM_PREFIX_ID,
    FIM_SUFFIX_ID,
    FIM_INSTRUCTION_ID,
)


def _arr(values: list[int]) -> mx.array:
    return mx.array(np.asarray(values, dtype=np.int32))


def _packet(tokens: list[int], chunks: list[tuple[int, int]]) -> CodePacket:
    starts = [c[0] for c in chunks]
    ends = [c[1] for c in chunks]
    return CodePacket(
        token_ids=_arr(tokens),
        chunk_starts=_arr(starts),
        chunk_ends=_arr(ends),
        chunk_kinds=_arr([1] * len(chunks)),
        chunk_dep_levels=_arr([0] * len(chunks)),
    )


def test_select_ast_span_picks_a_whole_chunk() -> None:
    # tokens 0..9; one interior chunk [3,6) keeps prefix+suffix non-empty.
    packet = _packet(list(range(10)), [(0, 3), (3, 6), (6, 10)])
    start, end, idx = select_ast_span(packet, rng=random.Random(0))
    # The selected span must equal the only eligible (interior) chunk span.
    assert (start, end) == (3, 6)
    assert idx == 1


def test_ast_fim_roundtrips_through_fim_permutation() -> None:
    tokens = [100 + i for i in range(10)]
    packet = _packet(tokens, [(0, 3), (3, 6), (6, 10)])
    # Force the AST path (rate 1.0) and PSM mode (spm_rate 0.0).
    result = apply_ast_fim(
        packet, seed=1, spm_rate=0.0, ast_fim_rate=1.0
    )
    assert result.kind == "ast_fim"
    assert result.mode == "psm"
    start, end = result.span
    # PSM layout: PREFIX <pre> SUFFIX <suf> MIDDLE <mid> EOT  -> reconstruct middle.
    seq = result.token_ids
    assert seq[0] == FIM_PREFIX_ID
    assert seq[-1] == EOT_ID
    mid_marker = seq.index(FIM_MIDDLE_ID)
    middle = seq[mid_marker + 1 : -1]
    assert middle == tokens[start:end]
    # And it is exactly the selected whole chunk.
    assert (start, end) in {(3, 6)}


def test_ast_fim_char_slice_is_recorded() -> None:
    tokens = list(range(10))
    packet = _packet(tokens, [(0, 3), (3, 6), (6, 10)])
    # Force the char slice (ast_fim_rate 0.0).
    result = apply_ast_fim(packet, seed=3, ast_fim_rate=0.0)
    assert result.kind == "char_fim"
    assert result.chunk_index is None


def test_ast_fim_falls_back_to_char_when_no_eligible_chunk_recorded() -> None:
    # The only chunk spans the entire window -> no non-empty prefix/suffix.
    tokens = list(range(6))
    packet = _packet(tokens, [(0, 6)])
    result = apply_ast_fim(packet, seed=2, ast_fim_rate=1.0)
    # AST path attempted but recorded fallback to char (NOT silent).
    assert result.kind == "char_fim"
    assert result.chunk_index is None


def test_ast_fim_deterministic_for_fixed_seed() -> None:
    packet = _packet(list(range(12)), [(0, 4), (4, 8), (8, 12)])
    a = apply_ast_fim(packet, seed=7)
    b = apply_ast_fim(packet, seed=7)
    assert a.token_ids == b.token_ids
    assert a.span == b.span and a.kind == b.kind and a.mode == b.mode


def test_ast_fim_requires_chunk_boundaries() -> None:
    packet = CodePacket(token_ids=_arr(list(range(6))))
    with pytest.raises(ValueError, match="chunk_starts and"):
        apply_ast_fim(packet, seed=0, ast_fim_rate=1.0)


def test_ast_ifim_uses_typed_instruction_tokens_and_exact_ast_layout() -> None:
    tokens = [100 + index for index in range(10)]
    packet = _packet(tokens, [(0, 3), (3, 6), (6, 10)])

    result = apply_ast_ifim(
        packet,
        instruction_token_ids=[901, 902],
        seed=1,
        spm_rate=0.0,
        ast_fim_rate=1.0,
    )

    assert result.kind == "ast_ifim"
    assert result.span == (3, 6)
    assert result.token_ids == [
        FIM_INSTRUCTION_ID,
        901,
        902,
        FIM_PREFIX_ID,
        *tokens[:3],
        FIM_SUFFIX_ID,
        *tokens[6:],
        FIM_MIDDLE_ID,
        *tokens[3:6],
        EOT_ID,
    ]


def test_ast_ifim_character_fallback_keeps_instruction_layout() -> None:
    packet = _packet(list(range(10)), [(0, 10)])

    result = apply_ast_ifim(
        packet,
        instruction_token_ids=[901, 902],
        seed=1,
        spm_rate=0.0,
        ast_fim_rate=1.0,
    )

    assert result.kind == "char_ifim"
    assert result.chunk_index is None
    assert result.token_ids[:3] == [FIM_INSTRUCTION_ID, 901, 902]
    assert result.token_ids[-1] == EOT_ID


@pytest.mark.parametrize("instruction_token_ids", [None, []])
def test_ast_ifim_rejects_missing_or_empty_typed_instruction_tokens(
    instruction_token_ids: list[int] | None,
) -> None:
    packet = _packet(list(range(10)), [(3, 6)])

    with pytest.raises(ValueError, match="instruction_token_ids"):
        apply_ast_ifim(
            packet,
            instruction_token_ids=instruction_token_ids,
            seed=0,
        )

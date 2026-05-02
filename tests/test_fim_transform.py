from __future__ import annotations

import random
from collections import Counter

import pytest

from cppmega_mlx.data.tokenizer_contract import REQUIRED_SPECIAL_TOKEN_IDS
from cppmega_mlx.data.fim import (
    EOT_ID,
    FIMSpecialTokenIds,
    FIM_INSTRUCTION_ID,
    FIM_MIDDLE_ID,
    FIM_PREFIX_ID,
    FIM_SUFFIX_ID,
    apply_fim_permutation,
    apply_fim_transform,
    apply_ifim_permutation,
    apply_ifim_transform,
    extract_ifim_instruction_text,
    sample_middle_span,
)


def test_reserved_id_layout_matches_m0_contract() -> None:
    ids = FIMSpecialTokenIds()

    assert EOT_ID == 3
    assert FIM_PREFIX_ID == 4
    assert FIM_MIDDLE_ID == 5
    assert FIM_SUFFIX_ID == 6
    assert FIM_INSTRUCTION_ID == 45
    assert ids.eot == 3
    assert ids.fim_prefix == 4
    assert ids.fim_middle == 5
    assert ids.fim_suffix == 6
    assert ids.fim_instruction == 45


def test_psm_permutation_for_explicit_middle_span() -> None:
    tokens = [10, 11, 12, 13, 14, 15]

    transformed = apply_fim_permutation(tokens, span=(2, 4), mode="psm")

    assert transformed == [
        FIM_PREFIX_ID,
        10,
        11,
        FIM_SUFFIX_ID,
        14,
        15,
        FIM_MIDDLE_ID,
        12,
        13,
        EOT_ID,
    ]
    assert tokens == [10, 11, 12, 13, 14, 15]
    assert Counter(transformed) == Counter(tokens + [4, 5, 6, 3])


def test_spm_permutation_for_explicit_middle_span() -> None:
    transformed = apply_fim_permutation(
        [10, 11, 12, 13, 14, 15], span=(2, 4), mode="spm"
    )

    assert transformed == [
        FIM_PREFIX_ID,
        FIM_SUFFIX_ID,
        14,
        15,
        FIM_MIDDLE_ID,
        10,
        11,
        12,
        13,
        EOT_ID,
    ]
    assert Counter(transformed) == Counter([10, 11, 12, 13, 14, 15, 4, 5, 6, 3])


@pytest.mark.parametrize(
    ("span", "match"),
    [
        ((0, 2), "0 < start < end < len"),
        ((2, 2), "0 < start < end < len"),
        ((2, 6), "0 < start < end < len"),
        ((4, 2), "0 < start < end < len"),
    ],
)
def test_invalid_explicit_spans_fail_closed(span: tuple[int, int], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        apply_fim_permutation([10, 11, 12, 13, 14, 15], span=span, mode="psm")


def test_invalid_mode_fails_closed_before_permutation() -> None:
    with pytest.raises(ValueError, match="FIM mode"):
        apply_fim_permutation([10, 11, 12, 13], span=(1, 2), mode="prefix")  # type: ignore[arg-type]


def test_special_id_collision_fails_closed() -> None:
    colliding = dict(REQUIRED_SPECIAL_TOKEN_IDS)
    colliding["EXTRA_ALIAS"] = FIM_PREFIX_ID

    with pytest.raises(ValueError, match="id 4 maps to both"):
        apply_fim_transform(
            [10, 11, 12],
            fim_rate=1.0,
            special_token_ids=colliding,
        )


def test_missing_fim_instruction_id_fails_closed_for_ifim_extension() -> None:
    missing = dict(REQUIRED_SPECIAL_TOKEN_IDS)
    missing.pop("FIM_INSTRUCTION")

    with pytest.raises(
        ValueError, match="missing required special token 'FIM_INSTRUCTION'"
    ):
        apply_ifim_transform(
            [10, 11, 12],
            instruction_token_ids=[90],
            fim_rate=1.0,
            special_token_ids=missing,
        )


@pytest.mark.parametrize("tokens", [[], [10], [10, 11]])
def test_short_samples_remain_unchanged(tokens: list[int]) -> None:
    assert apply_fim_transform(tokens, fim_rate=1.0, seed=123) == tokens


def test_seeded_sampling_is_deterministic() -> None:
    tokens = [10, 11, 12, 13, 14, 15, 16]

    left = apply_fim_transform(tokens, fim_rate=1.0, spm_rate=0.5, seed=99)
    right = apply_fim_transform(tokens, fim_rate=1.0, spm_rate=0.5, seed=99)
    other = apply_fim_transform(tokens, fim_rate=1.0, spm_rate=0.5, seed=100)

    assert left == right
    assert left != other
    assert Counter(left) == Counter(tokens + [4, 5, 6, 3])


def test_seeded_sampling_matches_reference_rng_sequence() -> None:
    tokens = [10, 11, 12, 13, 14, 15, 16]
    rng = random.Random(3)
    assert rng.random() < 1.0
    span = sample_middle_span(len(tokens), rng=rng)
    expected_mode = "spm" if rng.random() < 0.5 else "psm"

    transformed = apply_fim_transform(tokens, fim_rate=1.0, spm_rate=0.5, seed=3)

    assert transformed == apply_fim_permutation(tokens, span=span, mode=expected_mode)


@pytest.mark.parametrize(
    ("rate_name", "rate_value"),
    [
        ("fim_rate", -0.1),
        ("fim_rate", 1.1),
        ("spm_rate", -0.1),
        ("spm_rate", 1.1),
    ],
)
def test_sampled_transform_rate_validation(rate_name: str, rate_value: float) -> None:
    with pytest.raises(ValueError, match=rate_name):
        if rate_name == "fim_rate":
            apply_fim_transform([10, 11, 12], fim_rate=rate_value)
        else:
            apply_fim_transform([10, 11, 12], spm_rate=rate_value)


def test_seed_and_rng_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="either seed or rng"):
        apply_fim_transform([10, 11, 12], seed=1, rng=random.Random(1))


def test_ifim_permutation_prepends_instruction_to_standard_psm() -> None:
    transformed = apply_ifim_permutation(
        [10, 11, 12, 13, 14, 15],
        instruction_token_ids=[90, 91],
        span=(2, 4),
        mode="psm",
    )

    assert transformed == [
        FIM_INSTRUCTION_ID,
        90,
        91,
        FIM_PREFIX_ID,
        10,
        11,
        FIM_SUFFIX_ID,
        14,
        15,
        FIM_MIDDLE_ID,
        12,
        13,
        EOT_ID,
    ]
    assert Counter(transformed) == Counter(
        [10, 11, 12, 13, 14, 15, 90, 91, 45, 4, 5, 6, 3]
    )


def test_ifim_permutation_prepends_instruction_to_standard_spm() -> None:
    transformed = apply_ifim_permutation(
        [10, 11, 12, 13, 14, 15],
        instruction_token_ids=[90],
        span=(2, 4),
        mode="spm",
    )

    assert transformed == [
        FIM_INSTRUCTION_ID,
        90,
        FIM_PREFIX_ID,
        FIM_SUFFIX_ID,
        14,
        15,
        FIM_MIDDLE_ID,
        10,
        11,
        12,
        13,
        EOT_ID,
    ]


def test_ifim_sampled_transform_is_seed_deterministic_and_multiset_preserving() -> None:
    tokens = [10, 11, 12, 13, 14, 15, 16]
    instruction = [90, 91, 92]

    left = apply_ifim_transform(
        tokens,
        instruction_token_ids=instruction,
        fim_rate=1.0,
        spm_rate=0.5,
        seed=17,
    )
    right = apply_ifim_transform(
        tokens,
        instruction_token_ids=instruction,
        fim_rate=1.0,
        spm_rate=0.5,
        seed=17,
    )

    assert left == right
    assert Counter(left) == Counter(tokens + instruction + [45, 4, 5, 6, 3])


def test_empty_ifim_instruction_fails_closed() -> None:
    with pytest.raises(ValueError, match="instruction_token_ids must not be empty"):
        apply_ifim_permutation(
            [10, 11, 12],
            instruction_token_ids=[],
            span=(1, 2),
            mode="psm",
        )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("/** @brief Adds two numbers. */\nint add(int a, int b) { return a + b; }", "Adds two numbers"),
        ("// Complete the checksum path.\nuint32_t crc(uint32_t x) { return x; }", "Complete the checksum path"),
        ("def parse_value(text: str) -> int:\n    return int(text)", "Implement the function parse_value taking text and returns int"),
    ],
)
def test_instruction_text_extraction(source: str, expected: str) -> None:
    assert extract_ifim_instruction_text(source) == expected

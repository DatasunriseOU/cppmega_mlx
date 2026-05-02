from __future__ import annotations

import random

import pytest

from cppmega_mlx.data.fim import (
    EOT_ID,
    FIM_MIDDLE_ID,
    FIM_PREFIX_ID,
    FIM_SUFFIX_ID,
    apply_fim_permutation,
    apply_fim_transform,
    sample_middle_span,
)


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

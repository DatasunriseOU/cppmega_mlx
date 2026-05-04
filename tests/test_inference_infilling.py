from __future__ import annotations

import pytest

import cppmega_mlx.inference as inference
from cppmega_mlx.data.fim import FIMSpecialTokenIds
from cppmega_mlx.inference import build_fim_prompt_ids


def test_build_fim_prompt_ids_psm_stops_at_middle_marker() -> None:
    assert build_fim_prompt_ids([10, 11], [20, 21], mode="psm") == [
        4,
        10,
        11,
        6,
        20,
        21,
        5,
    ]


def test_build_fim_prompt_ids_spm_stops_before_generated_middle() -> None:
    assert build_fim_prompt_ids([10, 11], [20, 21], mode="spm") == [
        4,
        6,
        20,
        21,
        5,
        10,
        11,
    ]


def test_build_fim_prompt_ids_ifim_prepends_instruction_id_45_not_code_start_id_7() -> None:
    prompt = build_fim_prompt_ids(
        [10],
        [20],
        mode="psm",
        instruction_token_ids=[30, 31],
    )

    assert prompt == [45, 30, 31, 4, 10, 6, 20, 5]
    assert 7 not in prompt


def test_build_fim_prompt_ids_allows_empty_prefix_or_suffix_for_inference() -> None:
    assert build_fim_prompt_ids([], [20], mode="psm") == [4, 6, 20, 5]
    assert build_fim_prompt_ids([10], [], mode="spm") == [4, 6, 5, 10]


def test_build_fim_prompt_ids_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError, match="FIM mode"):
        build_fim_prompt_ids([10], [20], mode="middle")  # type: ignore[arg-type]


def test_build_fim_prompt_ids_rejects_empty_ifim_instruction() -> None:
    with pytest.raises(ValueError, match="instruction_token_ids must not be empty"):
        build_fim_prompt_ids([10], [20], instruction_token_ids=[])


@pytest.mark.parametrize(
    ("name", "prefix", "suffix", "instruction"),
    [
        ("prefix_token_ids", [True], [20], None),
        ("suffix_token_ids", [10], [False], None),
        ("instruction_token_ids", [10], [20], [object()]),
    ],
)
def test_build_fim_prompt_ids_rejects_non_integer_token_ids(
    name: str,
    prefix: list[object],
    suffix: list[object],
    instruction: list[object] | None,
) -> None:
    with pytest.raises(ValueError, match=name):
        build_fim_prompt_ids(
            prefix,  # type: ignore[arg-type]
            suffix,  # type: ignore[arg-type]
            instruction_token_ids=instruction,  # type: ignore[arg-type]
        )


def test_build_fim_prompt_ids_fails_closed_on_wrong_fim_instruction_id() -> None:
    with pytest.raises(ValueError, match="FIM_INSTRUCTION.*must use id 45"):
        build_fim_prompt_ids(
            [10],
            [20],
            instruction_token_ids=[30],
            special_token_ids=FIMSpecialTokenIds(fim_instruction=7),
        )


def test_build_fim_prompt_ids_validates_special_token_mapping() -> None:
    token_to_id = {
        "BOS": 2,
        "EOT": 3,
        "FIM_PREFIX": 4,
        "FIM_MIDDLE": 5,
        "FIM_SUFFIX": 6,
        "CODE_START": 7,
        "FIM_INSTRUCTION": 45,
        "SPACE": 46,
        "NL": 47,
    }

    assert build_fim_prompt_ids(
        [10],
        [20],
        special_token_ids=token_to_id,
    ) == [4, 10, 6, 20, 5]


def test_inference_root_exports_fim_prompt_builder() -> None:
    assert inference.build_fim_prompt_ids is build_fim_prompt_ids
    assert "build_fim_prompt_ids" in inference.__all__

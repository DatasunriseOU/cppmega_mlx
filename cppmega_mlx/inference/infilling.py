"""FIM/iFIM prompt construction for MLX inference."""

from __future__ import annotations

from collections.abc import Sequence

from cppmega_mlx.data.fim import (
    FIMMode,
    FIMSpecialTokenIds,
    FIMSpecialTokenInput,
)


def build_fim_prompt_ids(
    prefix_token_ids: Sequence[int],
    suffix_token_ids: Sequence[int],
    *,
    mode: FIMMode = "psm",
    instruction_token_ids: Sequence[int] | None = None,
    special_token_ids: FIMSpecialTokenInput = None,
) -> list[int]:
    """Build an inference-time FIM prompt that ends at ``<FIM_MIDDLE>``.

    Training permutations append the target middle span and EOT.  Inference must
    stop at the middle marker so the model generates the missing span.
    """

    ids = _resolve_special_token_ids(special_token_ids)
    _validate_fim_mode(mode)
    prefix = _validate_token_ids("prefix_token_ids", prefix_token_ids)
    suffix = _validate_token_ids("suffix_token_ids", suffix_token_ids)

    if mode == "psm":
        prompt = [
            ids.fim_prefix,
            *prefix,
            ids.fim_suffix,
            *suffix,
            ids.fim_middle,
        ]
    else:
        prompt = [
            ids.fim_prefix,
            ids.fim_suffix,
            *suffix,
            ids.fim_middle,
            *prefix,
        ]

    if instruction_token_ids is None:
        return prompt

    instruction = _validate_token_ids("instruction_token_ids", instruction_token_ids)
    if not instruction:
        raise ValueError("iFIM instruction_token_ids must not be empty")
    return [ids.fim_instruction, *instruction, *prompt]


def _resolve_special_token_ids(
    special_token_ids: FIMSpecialTokenInput,
) -> FIMSpecialTokenIds:
    if special_token_ids is None:
        return FIMSpecialTokenIds()
    if isinstance(special_token_ids, FIMSpecialTokenIds):
        return special_token_ids
    return FIMSpecialTokenIds.from_mapping(special_token_ids)


def _validate_fim_mode(mode: str) -> None:
    if mode not in {"psm", "spm"}:
        raise ValueError("FIM mode must be 'psm' or 'spm'")


def _validate_token_ids(name: str, token_ids: Sequence[int]) -> list[int]:
    tokens = list(token_ids)
    for token_id in tokens:
        if not isinstance(token_id, int) or isinstance(token_id, bool):
            raise ValueError(f"{name} must contain integer token ids")
    return tokens


__all__ = [
    "build_fim_prompt_ids",
]

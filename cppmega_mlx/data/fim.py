"""Pure-Python Fill-in-the-Middle token permutations.

This module is a reference data transform slice only.  It works on token ID
sequences directly and intentionally does not load or validate a tokenizer.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Literal

from cppmega_mlx.data.tokenizer_contract import REQUIRED_SPECIAL_TOKEN_IDS

FIM_PREFIX_ID = REQUIRED_SPECIAL_TOKEN_IDS["FIM_PREFIX"]
FIM_MIDDLE_ID = REQUIRED_SPECIAL_TOKEN_IDS["FIM_MIDDLE"]
FIM_SUFFIX_ID = REQUIRED_SPECIAL_TOKEN_IDS["FIM_SUFFIX"]
EOT_ID = REQUIRED_SPECIAL_TOKEN_IDS["EOT"]

FIMMode = Literal["psm", "spm"]


def apply_fim_permutation(
    token_ids: Sequence[int],
    *,
    span: tuple[int, int],
    mode: FIMMode,
) -> list[int]:
    """Permute ``token_ids`` into PSM or SPM format for an explicit middle span.

    ``span`` is half-open ``[start, end)`` and marks the middle segment to be
    predicted.  Valid spans keep prefix, middle, and suffix non-empty so sampled
    and explicit transforms share the same reference contract.
    """

    _validate_fim_mode(mode)
    start, end = span
    _validate_middle_span(len(token_ids), start, end)

    tokens = list(token_ids)
    prefix = tokens[:start]
    middle = tokens[start:end]
    suffix = tokens[end:]

    if mode == "psm":
        return [
            FIM_PREFIX_ID,
            *prefix,
            FIM_SUFFIX_ID,
            *suffix,
            FIM_MIDDLE_ID,
            *middle,
            EOT_ID,
        ]
    return [
        FIM_PREFIX_ID,
        FIM_SUFFIX_ID,
        *suffix,
        FIM_MIDDLE_ID,
        *prefix,
        *middle,
        EOT_ID,
    ]


def apply_fim_transform(
    token_ids: Sequence[int],
    *,
    fim_rate: float = 0.5,
    spm_rate: float = 0.5,
    seed: int | None = None,
    rng: random.Random | None = None,
) -> list[int]:
    """Apply sampled FIM with deterministic RNG injection.

    Samples shorter than three tokens are returned unchanged because they cannot
    provide non-empty prefix, middle, and suffix segments.
    """

    _validate_rate("fim_rate", fim_rate)
    _validate_rate("spm_rate", spm_rate)
    if rng is not None and seed is not None:
        raise ValueError("pass either seed or rng, not both")

    tokens = list(token_ids)
    if len(tokens) < 3:
        return tokens

    rand = rng if rng is not None else random.Random(seed)
    if rand.random() >= fim_rate:
        return tokens

    start, end = sample_middle_span(len(tokens), rng=rand)
    mode: FIMMode = "spm" if rand.random() < spm_rate else "psm"
    return apply_fim_permutation(tokens, span=(start, end), mode=mode)


def sample_middle_span(length: int, *, rng: random.Random) -> tuple[int, int]:
    """Sample a half-open middle span with non-empty prefix/middle/suffix."""

    if length < 3:
        raise ValueError("FIM span sampling requires at least 3 tokens")
    start = rng.randint(1, length - 2)
    end = rng.randint(start + 1, length - 1)
    return start, end


def _validate_middle_span(length: int, start: int, end: int) -> None:
    if length < 3:
        raise ValueError("FIM permutation requires at least 3 tokens")
    if not 0 < start < end < length:
        raise ValueError(
            "FIM middle span must satisfy 0 < start < end < len(token_ids)"
        )


def _validate_fim_mode(mode: str) -> None:
    if mode not in {"psm", "spm"}:
        raise ValueError("FIM mode must be 'psm' or 'spm'")


def _validate_rate(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")


__all__ = [
    "EOT_ID",
    "FIMMode",
    "FIM_MIDDLE_ID",
    "FIM_PREFIX_ID",
    "FIM_SUFFIX_ID",
    "apply_fim_permutation",
    "apply_fim_transform",
    "sample_middle_span",
]

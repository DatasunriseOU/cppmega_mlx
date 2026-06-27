"""Dependency-free tokenizer special-token contract checks."""

from __future__ import annotations

from collections.abc import Mapping

REQUIRED_SPECIAL_TOKEN_IDS: dict[str, int] = {
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

TOOL_USE_SPECIAL_TOKEN_IDS: dict[str, int] = {
    "BOS": 2,
    "EOS": 3,
    "CODE_START": 7,
    "CODE_END": 8,
    "THINK_START": 9,
    "THINK_END": 10,
    "QUERY_TOOL": 11,
    "TOOL_RESULT": 19,
}

DOMAIN_DELIMITER_TOKEN_IDS: dict[str, int] = {
    "CPP_CODE_START": 191,
    "CPP_CODE_END": 192,
    "MAKE_START": 193,
    "MAKE_END": 194,
    "CMAKE_START": 195,
    "CMAKE_END": 196,
    "NINJA_START": 197,
    "NINJA_END": 198,
    "BAZEL_START": 199,
    "BAZEL_END": 200,
    "BASH_START": 201,
    "BASH_END": 202,
    "ZSH_START": 203,
    "ZSH_END": 204,
    "SH_START": 205,
    "SH_END": 206,
    "TCSH_START": 207,
    "TCSH_END": 208,
    "COMPILER_DIAGNOSTIC_START": 209,
    "COMPILER_DIAGNOSTIC_END": 210,
    "BUILD_DIAGNOSTIC_START": 211,
    "BUILD_DIAGNOSTIC_END": 212,
    "COMPILER_ERROR_START": 213,
    "COMPILER_ERROR_END": 214,
    "BUILD_ERROR_START": 215,
    "BUILD_ERROR_END": 216,
    "LINKER_ERROR_START": 217,
    "LINKER_ERROR_END": 218,
    "TEST_OUTPUT_START": 219,
    "TEST_OUTPUT_END": 220,
    "TOOL_OUTPUT_START": 221,
    "TOOL_OUTPUT_END": 222,
}

SpecialTokenMapping = Mapping[int, str] | Mapping[str, int]


def validate_required_special_token_ids(mapping: SpecialTokenMapping) -> None:
    """Validate the cppmega special-token ids without loading a tokenizer.

    The input may be either id->token or token->id.  Validation fails closed on
    missing entries, duplicate ids/tokens, wrong ids, or ambiguous key/value
    shapes.
    """

    token_to_id = _normalize_special_token_mapping(mapping)
    for token, expected_id in REQUIRED_SPECIAL_TOKEN_IDS.items():
        if token not in token_to_id:
            raise ValueError(f"missing required special token {token!r}")
        actual_id = token_to_id[token]
        if actual_id != expected_id:
            raise ValueError(
                f"special token {token!r} must use id {expected_id}, got {actual_id}"
            )

    seen_ids: dict[int, str] = {}
    for token, token_id in token_to_id.items():
        existing = seen_ids.setdefault(token_id, token)
        if existing != token:
            raise ValueError(
                f"special token id collision: id {token_id} maps to both "
                f"{existing!r} and {token!r}"
            )


def _normalize_special_token_mapping(mapping: SpecialTokenMapping) -> dict[str, int]:
    if not mapping:
        raise ValueError("special token mapping must not be empty")

    keys_are_ids = all(_is_int_key(key) for key in mapping)
    values_are_tokens = all(isinstance(value, str) for value in mapping.values())
    keys_are_tokens = all(isinstance(key, str) for key in mapping)
    values_are_ids = all(_is_int_key(value) for value in mapping.values())

    if keys_are_ids and values_are_tokens:
        return _invert_id_to_token_mapping(mapping)
    if keys_are_tokens and values_are_ids:
        return {str(token): int(token_id) for token, token_id in mapping.items()}

    raise ValueError(
        "special token mapping must be consistently id->token or token->id"
    )


def _invert_id_to_token_mapping(mapping: SpecialTokenMapping) -> dict[str, int]:
    token_to_id: dict[str, int] = {}
    for raw_id, raw_token in mapping.items():
        token_id = int(raw_id)
        token = str(raw_token)
        existing = token_to_id.setdefault(token, token_id)
        if existing != token_id:
            raise ValueError(
                f"special token collision: token {token!r} maps to both "
                f"{existing} and {token_id}"
            )
    return token_to_id


def _is_int_key(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


__all__ = [
    "DOMAIN_DELIMITER_TOKEN_IDS",
    "REQUIRED_SPECIAL_TOKEN_IDS",
    "SpecialTokenMapping",
    "TOOL_USE_SPECIAL_TOKEN_IDS",
    "validate_required_special_token_ids",
]

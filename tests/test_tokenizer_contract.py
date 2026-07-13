from __future__ import annotations

import json
from pathlib import Path

import pytest

from cppmega_mlx.data.tokenizer_contract import (
    DOMAIN_DELIMITER_TOKEN_IDS,
    OBJECTIVE_BOUNDARY_TOKEN_IDS,
    REQUIRED_SPECIAL_TOKEN_IDS,
    TOOL_USE_SPECIAL_TOKEN_IDS,
    validate_checked_out_tokenizer_contract,
    validate_required_special_token_ids,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOKENIZER_CONTRACT_PATH = (
    _REPO_ROOT / "cppmega_mlx" / "tokenizer" / "tokenizer_contract_v1.json"
)
_TOKENIZER_JSON_PATH = _REPO_ROOT / "cppmega_mlx" / "tokenizer" / "tokenizer.json"
_CASE5_SEMANTIC_DELIMITER_IDS = set(range(237, 245))


def test_valid_id_to_token_mapping_passes() -> None:
    id_to_token = {
        token_id: token for token, token_id in REQUIRED_SPECIAL_TOKEN_IDS.items()
    }

    validate_required_special_token_ids(id_to_token)


def test_valid_token_to_id_mapping_passes() -> None:
    validate_required_special_token_ids(REQUIRED_SPECIAL_TOKEN_IDS)


def test_tool_use_token_ids_match_vendored_artifact_contract() -> None:
    assert TOOL_USE_SPECIAL_TOKEN_IDS == {
        "BOS": 2,
        "EOS": 3,
        "CODE_START": 7,
        "CODE_END": 8,
        "THINK_START": 9,
        "THINK_END": 10,
        "QUERY_TOOL": 11,
        "TOOL_RESULT": 19,
    }


def test_objective_boundary_ids_are_existing_frozen_tokens() -> None:
    assert OBJECTIVE_BOUNDARY_TOKEN_IDS == {
        "FILE_SEP": 14,
        "DIFF_START": 15,
        "DIFF_END": 16,
        "COMMENT_START": 17,
        "COMMENT_END": 18,
        "SYMBOL_REF": 38,
        "TYPE_INFO": 39,
        "OVERLOAD_SET": 44,
    }
    added_tokens = {
        added["content"].strip("<>"): added["id"]
        for added in json.loads(_TOKENIZER_JSON_PATH.read_text())["added_tokens"]
    }
    assert {
        name: added_tokens[name] for name in OBJECTIVE_BOUNDARY_TOKEN_IDS
    } == OBJECTIVE_BOUNDARY_TOKEN_IDS


def test_domain_delimiter_role_ids_are_reserved_contract_pairs() -> None:
    from cppmega_mlx.data import tokenizer_contract

    contract = json.loads(_TOKENIZER_CONTRACT_PATH.read_text())
    roles = {
        role: token_id
        for role, token_id in contract["reserved_role_assignments"].items()
        if not role.startswith("_")
    }
    start_bases = {role.removesuffix("_START") for role in roles if role.endswith("_START")}
    end_bases = {role.removesuffix("_END") for role in roles if role.endswith("_END")}
    assert start_bases == end_bases
    expected_roles = {
        f"{base}_{edge}" for base in start_bases for edge in ("START", "END")
    }

    assert expected_roles.issubset(roles)
    assert tokenizer_contract.DOMAIN_DELIMITER_TOKEN_IDS == {
        role: roles[role] for role in sorted(expected_roles)
    }

    ids = list(tokenizer_contract.DOMAIN_DELIMITER_TOKEN_IDS.values())
    assert len(ids) == len(set(ids))
    assert not (set(ids) & set(REQUIRED_SPECIAL_TOKEN_IDS.values()))
    assert _CASE5_SEMANTIC_DELIMITER_IDS.issubset(ids)

    added_tokens = {
        added["id"]: added["content"]
        for added in json.loads(_TOKENIZER_JSON_PATH.read_text())["added_tokens"]
    }
    for role, token_id in tokenizer_contract.DOMAIN_DELIMITER_TOKEN_IDS.items():
        assert added_tokens[token_id] == f"<RESERVED_{token_id}>", role


def test_checked_out_tokenizer_artifacts_match_imported_domain_contract() -> None:
    validated = validate_checked_out_tokenizer_contract(_REPO_ROOT)

    assert validated == DOMAIN_DELIMITER_TOKEN_IDS


def test_missing_required_special_token_fails() -> None:
    token_to_id = dict(REQUIRED_SPECIAL_TOKEN_IDS)
    token_to_id.pop("FIM_MIDDLE")

    with pytest.raises(ValueError, match="missing required special token 'FIM_MIDDLE'"):
        validate_required_special_token_ids(token_to_id)


def test_missing_required_special_id_fails() -> None:
    id_to_token = {
        token_id: token for token, token_id in REQUIRED_SPECIAL_TOKEN_IDS.items()
    }
    id_to_token.pop(6)

    with pytest.raises(ValueError, match="missing required special token 'FIM_SUFFIX'"):
        validate_required_special_token_ids(id_to_token)


def test_special_token_collision_fails_closed() -> None:
    id_to_token = {
        token_id: token for token, token_id in REQUIRED_SPECIAL_TOKEN_IDS.items()
    }
    id_to_token[8] = "FIM_PREFIX"

    with pytest.raises(ValueError, match="token 'FIM_PREFIX' maps to both 4 and 8"):
        validate_required_special_token_ids(id_to_token)


def test_special_id_collision_fails_closed() -> None:
    token_to_id = dict(REQUIRED_SPECIAL_TOKEN_IDS)
    token_to_id["EXTRA_ALIAS"] = 4

    with pytest.raises(ValueError, match="id 4 maps to both"):
        validate_required_special_token_ids(token_to_id)


def test_wrong_special_token_id_fails() -> None:
    token_to_id = dict(REQUIRED_SPECIAL_TOKEN_IDS)
    token_to_id["EOT"] = 9

    with pytest.raises(ValueError, match="special token 'EOT' must use id 3, got 9"):
        validate_required_special_token_ids(token_to_id)


def test_fim_instruction_string_does_not_satisfy_code_start_artifact_contract() -> None:
    token_to_id = dict(REQUIRED_SPECIAL_TOKEN_IDS)
    token_to_id.pop("CODE_START")
    token_to_id["FIM_INSTRUCTION"] = 7

    with pytest.raises(
        ValueError, match="missing required special token 'CODE_START'"
    ):
        validate_required_special_token_ids(token_to_id)


def test_missing_fim_instruction_extension_fails() -> None:
    token_to_id = dict(REQUIRED_SPECIAL_TOKEN_IDS)
    token_to_id.pop("FIM_INSTRUCTION")

    with pytest.raises(
        ValueError, match="missing required special token 'FIM_INSTRUCTION'"
    ):
        validate_required_special_token_ids(token_to_id)

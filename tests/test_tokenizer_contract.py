from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from cppmega_mlx.data.tokenizer_contract import (
    DOMAIN_DELIMITER_TOKEN_IDS,
    EXPECTED_TOKEN_ROLE_CONTRACT_SHA256,
    OBJECTIVE_BOUNDARY_TOKEN_IDS,
    REQUIRED_SPECIAL_TOKEN_IDS,
    RESERVED_ROLE_TOKEN_IDS,
    SPECIAL_TOKEN_ALIASES,
    SPECIAL_TOKEN_IDS,
    TOKENIZER_CONTRACT,
    TOKENIZER_CONTRACT_PATH,
    TOKENIZER_CONTRACT_SHA256,
    TOKENIZER_CONTRACT_SHA256_METADATA_KEY,
    TOKEN_ROLE_CONTRACT_SHA256,
    TOOL_USE_SPECIAL_TOKEN_IDS,
    canonical_json_sha256,
    validate_checked_out_tokenizer_contract,
    validate_huggingface_tokenizer_sidecars,
    validate_required_special_token_ids,
    validate_tokenizer_artifact_contract,
    validate_tokenizer_contract,
    validate_untrusted_tokenizer_text,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOKENIZER_CONTRACT_PATH = (
    _REPO_ROOT / "cppmega_mlx" / "tokenizer" / "tokenizer_contract_v1.json"
)
_TOKENIZER_JSON_PATH = _REPO_ROOT / "cppmega_mlx" / "tokenizer" / "tokenizer.json"
_CASE5_SEMANTIC_DELIMITER_IDS = set(range(237, 249))
_CPP_TOKENIZER_V2_DIR = Path("/Volumes/external/sources/cppmega/data/tokenizer_v2")

_REQUIRED_DOMAIN_PAIR_IDS = {
    "CPP_CODE": (191, 192),
    "MAKE": (193, 194),
    "CMAKE": (195, 196),
    "NINJA": (197, 198),
    "BAZEL": (199, 200),
    "BASH": (201, 202),
    "ZSH": (203, 204),
    "SH": (205, 206),
    "TCSH": (207, 208),
    "COMPILER_DIAGNOSTIC": (209, 210),
    "BUILD_DIAGNOSTIC": (211, 212),
    "COMPILER_ERROR": (213, 214),
    "BUILD_ERROR": (215, 216),
    "LINKER_ERROR": (217, 218),
    "TEST_OUTPUT": (219, 220),
    "TOOL_OUTPUT": (221, 222),
    "AUTOCONF": (223, 224),
    "AUTOMAKE": (225, 226),
    "MESON": (227, 228),
    "GN": (229, 230),
    "SCONS": (231, 232),
    "XMAKE": (233, 234),
    "COMPILE_COMMANDS": (235, 236),
    "CONFIGURE": (237, 238),
    "SQL": (239, 240),
    "LINKER_DIAGNOSTIC": (241, 242),
    "SANITIZER_OUTPUT": (243, 244),
    "KSH": (245, 246),
    "PYTHON": (247, 248),
}


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


def test_full_special_token_table_is_frozen_and_collision_free() -> None:
    expected_roles = (
        "PAD",
        "UNK",
        "BOS",
        "EOS",
        "FIM_PREFIX",
        "FIM_MIDDLE",
        "FIM_SUFFIX",
        "CODE_START",
        "CODE_END",
        "THINK_START",
        "THINK_END",
        "QUERY_TOOL",
        "INDEX",
        "DEBUG_CONTEXT",
        "FILE_SEP",
        "DIFF_START",
        "DIFF_END",
        "COMMENT_START",
        "COMMENT_END",
        "TOOL_RESULT",
        "THINK_CODE",
        "THINK_ERROR",
        "THINK_FIX",
        "THINK_VERIFY",
        "THINK_PLAN",
        "THINK_TRACE",
        "SCRIPT_START",
        "SCRIPT_END",
        "SCRIPT_RESULT",
        "COMPILE_START",
        "COMPILE_END",
        "COMPILE_OK",
        "COMPILE_ERROR",
        "TEST_START",
        "TEST_END",
        "TEST_PASS",
        "TEST_FAIL",
        "AST_NODE",
        "SYMBOL_REF",
        "TYPE_INFO",
        "SCOPE_ENTER",
        "SCOPE_EXIT",
        "INCLUDE_CONTEXT",
        "TEMPLATE_INST",
        "OVERLOAD_SET",
        "FIM_INSTRUCTION",
        "SPACE",
        "NL",
    )

    assert SPECIAL_TOKEN_IDS == dict(zip(expected_roles, range(48), strict=True))
    assert len(set(SPECIAL_TOKEN_IDS.values())) == len(SPECIAL_TOKEN_IDS)
    assert set(SPECIAL_TOKEN_IDS.values()).isdisjoint(RESERVED_ROLE_TOKEN_IDS.values())
    assert SPECIAL_TOKEN_ALIASES == {"EOT": 3, "INS": 45}
    assert TOKEN_ROLE_CONTRACT_SHA256 == EXPECTED_TOKEN_ROLE_CONTRACT_SHA256
    assert TOKEN_ROLE_CONTRACT_SHA256 == canonical_json_sha256(
        {
            "special_tokens": SPECIAL_TOKEN_IDS,
            "aliases": SPECIAL_TOKEN_ALIASES,
            "reserved_role_assignments": RESERVED_ROLE_TOKEN_IDS,
        }
    )


def test_tokenizer_contract_sha256_covers_exact_frozen_json_bytes() -> None:
    assert TOKENIZER_CONTRACT_PATH == _TOKENIZER_CONTRACT_PATH
    assert TOKENIZER_CONTRACT_SHA256 == hashlib.sha256(
        TOKENIZER_CONTRACT_PATH.read_bytes()
    ).hexdigest()
    assert TOKENIZER_CONTRACT_SHA256_METADATA_KEY == (
        "cppmega.tokenizer_contract_sha256"
    )


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
    assert start_bases == set(_REQUIRED_DOMAIN_PAIR_IDS)
    expected_roles = {
        f"{base}_{edge}" for base in start_bases for edge in ("START", "END")
    }

    assert expected_roles.issubset(roles)
    assert tokenizer_contract.DOMAIN_DELIMITER_TOKEN_IDS == {
        role: roles[role] for role in sorted(expected_roles)
    }
    assert {
        base: (
            tokenizer_contract.DOMAIN_DELIMITER_TOKEN_IDS[f"{base}_START"],
            tokenizer_contract.DOMAIN_DELIMITER_TOKEN_IDS[f"{base}_END"],
        )
        for base in start_bases
    } == _REQUIRED_DOMAIN_PAIR_IDS

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


def test_ksh_and_python_roles_keep_literal_reserved_vocab_entries() -> None:
    assert {
        role: RESERVED_ROLE_TOKEN_IDS[role]
        for role in ("KSH_START", "KSH_END", "PYTHON_START", "PYTHON_END")
    } == {
        "KSH_START": 245,
        "KSH_END": 246,
        "PYTHON_START": 247,
        "PYTHON_END": 248,
    }

    tokenizer = json.loads(_TOKENIZER_JSON_PATH.read_text())
    vocab = tokenizer["model"]["vocab"]
    added_tokens = {
        int(item["id"]): item["content"] for item in tokenizer["added_tokens"]
    }
    for token_id in range(245, 249):
        literal = f"<RESERVED_{token_id}>"
        assert vocab[literal] == token_id
        assert added_tokens[token_id] == literal


def test_file_separator_has_one_canonical_role_and_id() -> None:
    assert SPECIAL_TOKEN_IDS["FILE_SEP"] == 14
    assert "FILE_SEP" not in RESERVED_ROLE_TOKEN_IDS


def test_sequence_contract_records_v1_behavior_and_blocked_v2_migration() -> None:
    sequences = TOKENIZER_CONTRACT["sequence_contracts"]
    assert sequences["line_comment_eol"] == {
        "terminator": "NL",
        "decode_value": "newline",
        "status": "active_v1",
    }
    assert sequences["comment_objective"]["opening"] == "COMMENT_START"
    assert sequences["comment_objective"]["closing"] == "COMMENT_END"
    assert sequences["docstring"]["status"] == "no_dedicated_v1_delimiter_pair"
    assert sequences["raw_control_literals"]["status"] == "ambiguous_in_v1"
    assert sequences["fim"]["closing"] == "EOT"
    assert sequences["ifim"]["opening"] == "FIM_INSTRUCTION"
    assert sequences["ifim"]["closing"] == "EOT"

    migration = TOKENIZER_CONTRACT["migration_plan"]
    assert migration["target_contract_version"] == 2
    assert migration["status"] == (
        "blocked_until_retokenization_and_checkpoint_training"
    )
    assert migration["frozen_v1_artifact_mutation"] == "forbidden"
    assert migration["vocab_size"] == 65_536


@pytest.mark.parametrize(
    "text",
    [
        'const char *value = "<FIM_MIDDLE>";',
        "// <RESERVED_239> must not become SQL_START\nint value;",
        "/* <RESERVED_209> must not become a compiler diagnostic */",
    ],
)
def test_untrusted_control_token_literals_fail_closed(text: str) -> None:
    with pytest.raises(ValueError, match="raw text contains tokenizer control literal"):
        validate_untrusted_tokenizer_text(text, where="fixture.cpp")


def test_untrusted_control_token_guard_accepts_normal_cpp_templates() -> None:
    validate_untrusted_tokenizer_text(
        "template <typename T> T identity(T value) { return value; }",
        where="fixture.cpp",
    )


def test_checked_out_tokenizer_artifacts_match_imported_domain_contract() -> None:
    validated = validate_checked_out_tokenizer_contract(_REPO_ROOT)

    assert validated == DOMAIN_DELIMITER_TOKEN_IDS


def test_cppmega_tokenizer_v2_matches_vendored_artifact_when_available() -> None:
    # Sync procedure: cppmega/data/tokenizer_v2/tokenizer.json is the canonical
    # artifact (its SHA-256 is the one recorded in sealed megatron_ready bundle
    # manifests). The vendored cppmega_mlx/tokenizer/tokenizer.json must be a
    # byte-identical copy of it; on drift, re-copy the canonical file into this
    # repo instead of editing the vendored copy.
    tokenizer_path = _CPP_TOKENIZER_V2_DIR / "tokenizer.json"
    if not tokenizer_path.is_file():
        pytest.skip(f"{tokenizer_path} is not available")

    validated = validate_tokenizer_artifact_contract(
        tokenizer_path,
        require_frozen_artifact=True,
    )
    validate_huggingface_tokenizer_sidecars(_CPP_TOKENIZER_V2_DIR)

    assert validated == DOMAIN_DELIMITER_TOKEN_IDS
    assert canonical_json_sha256(json.loads(tokenizer_path.read_text())) == (
        canonical_json_sha256(json.loads(_TOKENIZER_JSON_PATH.read_text()))
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("contract_version", 2, "contract_version must be 1"),
        ("vocab_size", 65_535, "vocab_size must be frozen at 65536"),
    ],
)
def test_contract_version_and_vocab_drift_fail_closed(
    field: str,
    value: int,
    message: str,
) -> None:
    contract = copy.deepcopy(TOKENIZER_CONTRACT)
    contract[field] = value

    with pytest.raises(ValueError, match=message):
        validate_tokenizer_contract(contract)


def test_reserved_role_collision_fails_closed() -> None:
    contract = copy.deepcopy(TOKENIZER_CONTRACT)
    contract["reserved_role_assignments"]["REPO_NAME"] = contract[
        "reserved_role_assignments"
    ]["MAKE_START"]

    with pytest.raises(ValueError, match="reserved_role_assignments id collision"):
        validate_tokenizer_contract(contract)


def test_swapped_opening_and_closing_role_ids_fail_closed() -> None:
    contract = copy.deepcopy(TOKENIZER_CONTRACT)
    assignments = contract["reserved_role_assignments"]
    assignments["SQL_START"], assignments["SQL_END"] = (
        assignments["SQL_END"],
        assignments["SQL_START"],
    )

    with pytest.raises(ValueError, match="token role assignment SHA-256 mismatch"):
        validate_tokenizer_contract(contract)


def test_unpaired_domain_role_fails_closed() -> None:
    contract = copy.deepcopy(TOKENIZER_CONTRACT)
    del contract["reserved_role_assignments"]["SQL_END"]

    with pytest.raises(ValueError, match="unpaired domain delimiter roles.*SQL_END"):
        from cppmega_mlx.data.tokenizer_contract import (
            _derive_domain_delimiter_token_ids,
        )

        _derive_domain_delimiter_token_ids(contract)


def test_migration_gate_cannot_be_marked_active_without_versioned_training() -> None:
    contract = copy.deepcopy(TOKENIZER_CONTRACT)
    contract["migration_plan"]["status"] = "active"

    with pytest.raises(ValueError, match="blocked_until_retokenization"):
        validate_tokenizer_contract(contract)


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

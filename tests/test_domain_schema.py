from __future__ import annotations

import hashlib

import pytest

from cppmega_mlx.data.domain_schema import (
    DOMAIN_EDGE_FAMILIES,
    DOMAIN_DELIMITER_ROLES,
    DOMAIN_SCHEMA_PATH,
    DOMAIN_SCHEMA_SHA256,
    DOMAIN_SCHEMA_SHA256_METADATA_KEY,
    DomainKind,
    delimiter_token_ids,
    validate_case5_contract_metadata,
    validate_domain_delimiter_contract,
    validate_domain_edge_kind,
)
from cppmega_mlx.data.tokenizer_contract import (
    DOMAIN_DELIMITER_CONTRACT_METADATA_KEY,
    DOMAIN_DELIMITER_CONTRACT_SHA256,
    DOMAIN_DELIMITER_TOKEN_IDS,
    TOKENIZER_CONTRACT_SHA256,
    TOKENIZER_CONTRACT_SHA256_METADATA_KEY,
)


def test_domain_delimiter_contract_is_complete() -> None:
    validate_domain_delimiter_contract()

    assert len(DOMAIN_DELIMITER_ROLES) == 29
    assert DomainKind.CMAKE in DOMAIN_DELIMITER_ROLES
    assert DomainKind.BAZEL in DOMAIN_DELIMITER_ROLES
    assert DomainKind.AUTOCONF in DOMAIN_DELIMITER_ROLES
    assert DomainKind.AUTOMAKE in DOMAIN_DELIMITER_ROLES
    assert DomainKind.MESON in DOMAIN_DELIMITER_ROLES
    assert DomainKind.GN in DOMAIN_DELIMITER_ROLES
    assert DomainKind.SCONS in DOMAIN_DELIMITER_ROLES
    assert DomainKind.XMAKE in DOMAIN_DELIMITER_ROLES
    assert DomainKind.COMPILE_COMMANDS in DOMAIN_DELIMITER_ROLES
    assert DomainKind.BASH in DOMAIN_DELIMITER_ROLES
    assert DomainKind.KSH in DOMAIN_DELIMITER_ROLES
    assert DomainKind.COMPILER_ERROR in DOMAIN_DELIMITER_ROLES
    assert DomainKind.LINKER_ERROR in DOMAIN_DELIMITER_ROLES
    assert DomainKind.CONFIGURE in DOMAIN_DELIMITER_ROLES
    assert DomainKind.SQL in DOMAIN_DELIMITER_ROLES
    assert DomainKind.PYTHON in DOMAIN_DELIMITER_ROLES
    assert DomainKind.LINKER_DIAGNOSTIC in DOMAIN_DELIMITER_ROLES
    assert DomainKind.SANITIZER_OUTPUT in DOMAIN_DELIMITER_ROLES


def test_domain_delimiter_roles_match_each_domain_name_exactly() -> None:
    expected = {
        domain: (
            f"{'CPP_CODE' if domain == DomainKind.CPP else domain.name}_START",
            f"{'CPP_CODE' if domain == DomainKind.CPP else domain.name}_END",
        )
        for domain in DomainKind
        if domain != DomainKind.UNKNOWN
    }

    assert DOMAIN_DELIMITER_ROLES == expected


def test_domain_kind_ids_extend_gaps_without_renumbering_old_rows() -> None:
    assert int(DomainKind.BASH) == 20
    assert int(DomainKind.ZSH) == 21
    assert int(DomainKind.SH) == 22
    assert int(DomainKind.TCSH) == 23
    assert int(DomainKind.KSH) == 24
    assert int(DomainKind.SQL) == 30
    assert int(DomainKind.PYTHON) == 31
    assert int(DomainKind.COMPILER_DIAGNOSTIC) == 40


def test_domain_delimiter_validator_rejects_swapped_domain_pairs() -> None:
    swapped = dict(DOMAIN_DELIMITER_ROLES)
    swapped[DomainKind.CMAKE] = DOMAIN_DELIMITER_ROLES[DomainKind.MAKE]
    swapped[DomainKind.MAKE] = DOMAIN_DELIMITER_ROLES[DomainKind.CMAKE]

    with pytest.raises(ValueError, match="CMAKE: delimiter roles must be"):
        validate_domain_delimiter_contract(swapped)


def test_domain_schema_sha256_covers_exact_frozen_json_bytes() -> None:
    assert DOMAIN_SCHEMA_SHA256 == hashlib.sha256(
        DOMAIN_SCHEMA_PATH.read_bytes()
    ).hexdigest()
    assert DOMAIN_SCHEMA_SHA256_METADATA_KEY == "cppmega.domain_schema_sha256"


def test_case5_contract_metadata_rejects_missing_and_stale_hashes() -> None:
    valid = {
        DOMAIN_SCHEMA_SHA256_METADATA_KEY.encode("utf-8"): DOMAIN_SCHEMA_SHA256.encode(
            "ascii"
        ),
        TOKENIZER_CONTRACT_SHA256_METADATA_KEY.encode(
            "utf-8"
        ): TOKENIZER_CONTRACT_SHA256.encode("ascii"),
        DOMAIN_DELIMITER_CONTRACT_METADATA_KEY.encode(
            "utf-8"
        ): DOMAIN_DELIMITER_CONTRACT_SHA256.encode("ascii"),
    }
    validate_case5_contract_metadata(valid, where="fixture.parquet")

    for key in tuple(valid):
        missing = dict(valid)
        missing.pop(key)
        with pytest.raises(ValueError, match="missing or stale frozen CASE5"):
            validate_case5_contract_metadata(missing, where="fixture.parquet")

        stale = dict(valid)
        stale[key] = b"0" * 64
        with pytest.raises(ValueError, match="missing or stale frozen CASE5"):
            validate_case5_contract_metadata(stale, where="fixture.parquet")


def test_case5_contract_metadata_accepts_exact_previous_v1_triples() -> None:
    previous = {
        DOMAIN_SCHEMA_SHA256_METADATA_KEY.encode("utf-8"): (
            b"9c3517b5a3fda01c4f55d55bc0d12dff4af3edb3db6321bda6c22489061b4fdd"
        ),
        TOKENIZER_CONTRACT_SHA256_METADATA_KEY.encode("utf-8"): (
            b"c3bb669015c48e2049e3b82ccb8c98c6eceae0644f7da0b5b8600c573d7087a5"
        ),
        DOMAIN_DELIMITER_CONTRACT_METADATA_KEY.encode("utf-8"): (
            b"1f2e35d7917409fc03704d32c2d55d0fb3e29f1bd9e60acca775a392cf2f53e6"
        ),
    }
    validate_case5_contract_metadata(previous, where="old-v1.parquet")

    pre_ksh = dict(previous)
    pre_ksh[TOKENIZER_CONTRACT_SHA256_METADATA_KEY.encode("utf-8")] = (
        b"80e73699e26d2c19fe4477cf8194886e52c7a5e114023df27e55d6a69b62c198"
    )
    validate_case5_contract_metadata(pre_ksh, where="pre-ksh-v1.parquet")

    mixed = dict(previous)
    mixed[DOMAIN_SCHEMA_SHA256_METADATA_KEY.encode("utf-8")] = (
        DOMAIN_SCHEMA_SHA256.encode("ascii")
    )
    with pytest.raises(ValueError, match="missing or stale frozen CASE5"):
        validate_case5_contract_metadata(mixed, where="mixed-v1.parquet")

    mixed_tokenizer = dict(previous)
    mixed_tokenizer[TOKENIZER_CONTRACT_SHA256_METADATA_KEY.encode("utf-8")] = (
        TOKENIZER_CONTRACT_SHA256.encode("ascii")
    )
    with pytest.raises(ValueError, match="missing or stale frozen CASE5"):
        validate_case5_contract_metadata(
            mixed_tokenizer,
            where="mixed-tokenizer-v1.parquet",
        )

    incomplete = dict(previous)
    incomplete.pop(DOMAIN_DELIMITER_CONTRACT_METADATA_KEY.encode("utf-8"))
    with pytest.raises(ValueError, match="missing or stale frozen CASE5"):
        validate_case5_contract_metadata(incomplete, where="incomplete-v1.parquet")


def test_delimiter_token_ids_use_reserved_contract() -> None:
    assert delimiter_token_ids(DomainKind.CMAKE) == (
        DOMAIN_DELIMITER_TOKEN_IDS["CMAKE_START"],
        DOMAIN_DELIMITER_TOKEN_IDS["CMAKE_END"],
    )
    assert delimiter_token_ids(DomainKind.SH) == (
        DOMAIN_DELIMITER_TOKEN_IDS["SH_START"],
        DOMAIN_DELIMITER_TOKEN_IDS["SH_END"],
    )
    assert delimiter_token_ids(DomainKind.MESON) == (
        DOMAIN_DELIMITER_TOKEN_IDS["MESON_START"],
        DOMAIN_DELIMITER_TOKEN_IDS["MESON_END"],
    )
    assert delimiter_token_ids(DomainKind.COMPILE_COMMANDS) == (
        DOMAIN_DELIMITER_TOKEN_IDS["COMPILE_COMMANDS_START"],
        DOMAIN_DELIMITER_TOKEN_IDS["COMPILE_COMMANDS_END"],
    )
    assert delimiter_token_ids(DomainKind.KSH) == (245, 246)
    assert delimiter_token_ids(DomainKind.PYTHON) == (247, 248)


def test_unknown_domain_has_no_delimiter() -> None:
    with pytest.raises(KeyError):
        delimiter_token_ids(DomainKind.UNKNOWN)


def test_domain_edge_families_are_independent_and_fail_closed() -> None:
    assert {int(kind) for kind in DOMAIN_EDGE_FAMILIES["domain"]} == set(range(1, 14))
    assert {int(kind) for kind in DOMAIN_EDGE_FAMILIES["build"]} == set(range(20, 27))
    assert {int(kind) for kind in DOMAIN_EDGE_FAMILIES["shell"]} == set(range(40, 45))
    assert {int(kind) for kind in DOMAIN_EDGE_FAMILIES["diagnostic"]} == {
        60,
        61,
        62,
        63,
        64,
        70,
        71,
        80,
        90,
    }
    assert {int(kind) for kind in DOMAIN_EDGE_FAMILIES["cross_domain"]} == {100}

    with pytest.raises(ValueError, match="belongs to build, not domain"):
        validate_domain_edge_kind(26, family="domain")

from __future__ import annotations

import pytest

from cppmega_mlx.data.domain_schema import (
    DOMAIN_EDGE_FAMILIES,
    DOMAIN_DELIMITER_ROLES,
    DomainKind,
    validate_domain_edge_kind,
    delimiter_token_ids,
    validate_domain_delimiter_contract,
)
from cppmega_mlx.data.tokenizer_contract import DOMAIN_DELIMITER_TOKEN_IDS


def test_domain_delimiter_contract_is_complete() -> None:
    validate_domain_delimiter_contract()

    assert len(DOMAIN_DELIMITER_ROLES) == 27
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
    assert DomainKind.COMPILER_ERROR in DOMAIN_DELIMITER_ROLES
    assert DomainKind.LINKER_ERROR in DOMAIN_DELIMITER_ROLES
    assert DomainKind.CONFIGURE in DOMAIN_DELIMITER_ROLES
    assert DomainKind.SQL in DOMAIN_DELIMITER_ROLES
    assert DomainKind.LINKER_DIAGNOSTIC in DOMAIN_DELIMITER_ROLES
    assert DomainKind.SANITIZER_OUTPUT in DOMAIN_DELIMITER_ROLES


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

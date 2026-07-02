"""Versioned domain sidecar schema for cppmega world-code documents.

The tokenizer artifact is frozen.  Domain boundaries therefore use logical
roles mapped to existing ``<RESERVED_N>`` ids in ``tokenizer_contract.py``.
This module owns the integer enums used by parquet sidecars, parser outputs,
MLX route tests, and the Megatron sidecar converter.
"""

from __future__ import annotations

from enum import IntEnum

from cppmega_mlx.data.tokenizer_contract import DOMAIN_DELIMITER_TOKEN_IDS


class DomainKind(IntEnum):
    UNKNOWN = 0
    CPP = 1
    CMAKE = 2
    MAKE = 3
    NINJA = 4
    BAZEL = 5
    AUTOCONF = 6
    AUTOMAKE = 7
    MESON = 8
    GN = 9
    SCONS = 10
    XMAKE = 11
    COMPILE_COMMANDS = 12
    BASH = 20
    ZSH = 21
    SH = 22
    TCSH = 23
    COMPILER_DIAGNOSTIC = 40
    BUILD_DIAGNOSTIC = 41
    COMPILER_ERROR = 42
    BUILD_ERROR = 43
    LINKER_ERROR = 44
    TEST_OUTPUT = 45
    TOOL_OUTPUT = 46


class DomainRoleKind(IntEnum):
    NONE = 0
    DELIMITER = 1
    KEYWORD = 2
    IDENTIFIER = 3
    TARGET = 4
    VARIABLE = 5
    COMMAND = 6
    PATH = 7
    OPTION = 8
    STRING = 9
    LABEL = 10
    RULE = 11
    ATTRIBUTE = 12
    SOURCE = 13
    LIBRARY = 14
    PREREQUISITE = 15
    OUTPUT = 16
    INPUT = 17
    ENVIRONMENT = 18
    REDIRECT = 19
    PIPE = 20
    SEVERITY = 30
    MESSAGE = 31
    FILE = 32
    LINE = 33
    COLUMN = 34
    SYMBOL = 35
    FIXIT = 36
    NOTE = 37
    EXIT_CODE = 38
    TEST_NAME = 39


class DomainEdgeKind(IntEnum):
    UNKNOWN = 0
    AST_PARENT = 1
    CALL = 2
    TYPE = 3
    DEF_USE = 4
    INCLUDE = 5
    BUILD_TARGET_DEP = 20
    BUILD_TARGET_SOURCE = 21
    BUILD_RULE_COMMAND = 22
    BUILD_ACTION_INPUT = 23
    BUILD_ACTION_OUTPUT = 24
    BUILD_VAR_DEF_USE = 25
    BUILD_COMMAND_TARGET = 26
    SHELL_PIPE = 40
    SHELL_REDIR_IN = 41
    SHELL_REDIR_OUT = 42
    SHELL_VAR_DEF_USE = 43
    SHELL_COMMAND_FILE = 44
    DIAG_PRIMARY_LOCATION = 60
    DIAG_NOTE = 61
    DIAG_FIXIT = 62
    DIAG_COMMAND = 63
    DIAG_BUILD_TARGET = 64
    LINK_UNDEFINED_SYMBOL = 70
    LINK_CANDIDATE_DEF = 71
    TEST_FAILURE_LOCATION = 80
    TOOL_ACTION_RESULT = 90


class ParseConfidence(IntEnum):
    ABSENT = 0
    RAW = 1
    HEURISTIC = 2
    PARTIAL = 3
    EXACT = 4


DOMAIN_DELIMITER_ROLES: dict[DomainKind, tuple[str, str]] = {
    DomainKind.CPP: ("CPP_CODE_START", "CPP_CODE_END"),
    DomainKind.CMAKE: ("CMAKE_START", "CMAKE_END"),
    DomainKind.MAKE: ("MAKE_START", "MAKE_END"),
    DomainKind.NINJA: ("NINJA_START", "NINJA_END"),
    DomainKind.BAZEL: ("BAZEL_START", "BAZEL_END"),
    # Autotools/Automake are Make-family text, but remain distinct in
    # token_domain_ids so the model does not collapse them semantically.
    DomainKind.AUTOCONF: ("MAKE_START", "MAKE_END"),
    DomainKind.AUTOMAKE: ("MAKE_START", "MAKE_END"),
    DomainKind.BASH: ("BASH_START", "BASH_END"),
    DomainKind.ZSH: ("ZSH_START", "ZSH_END"),
    DomainKind.SH: ("SH_START", "SH_END"),
    DomainKind.TCSH: ("TCSH_START", "TCSH_END"),
    DomainKind.COMPILER_DIAGNOSTIC: (
        "COMPILER_DIAGNOSTIC_START",
        "COMPILER_DIAGNOSTIC_END",
    ),
    DomainKind.BUILD_DIAGNOSTIC: (
        "BUILD_DIAGNOSTIC_START",
        "BUILD_DIAGNOSTIC_END",
    ),
    DomainKind.COMPILER_ERROR: ("COMPILER_ERROR_START", "COMPILER_ERROR_END"),
    DomainKind.BUILD_ERROR: ("BUILD_ERROR_START", "BUILD_ERROR_END"),
    DomainKind.LINKER_ERROR: ("LINKER_ERROR_START", "LINKER_ERROR_END"),
    DomainKind.TEST_OUTPUT: ("TEST_OUTPUT_START", "TEST_OUTPUT_END"),
    DomainKind.TOOL_OUTPUT: ("TOOL_OUTPUT_START", "TOOL_OUTPUT_END"),
}


def delimiter_token_ids(domain: DomainKind) -> tuple[int, int]:
    """Return the reserved-token ids for ``domain``'s opener/closer."""

    try:
        start_role, end_role = DOMAIN_DELIMITER_ROLES[DomainKind(domain)]
    except KeyError as exc:
        raise KeyError(f"domain {domain!r} has no delimiter role pair") from exc
    return (
        DOMAIN_DELIMITER_TOKEN_IDS[start_role],
        DOMAIN_DELIMITER_TOKEN_IDS[end_role],
    )


def validate_domain_delimiter_contract() -> None:
    """Fail loud if any logical domain delimiter is missing or malformed."""

    missing: list[str] = []
    for domain, (start_role, end_role) in DOMAIN_DELIMITER_ROLES.items():
        if not start_role.endswith("_START"):
            raise ValueError(f"{domain.name}: start role must end with _START")
        if not end_role.endswith("_END"):
            raise ValueError(f"{domain.name}: end role must end with _END")
        if start_role not in DOMAIN_DELIMITER_TOKEN_IDS:
            missing.append(start_role)
        if end_role not in DOMAIN_DELIMITER_TOKEN_IDS:
            missing.append(end_role)
        if start_role in DOMAIN_DELIMITER_TOKEN_IDS and end_role in DOMAIN_DELIMITER_TOKEN_IDS:
            start_id = DOMAIN_DELIMITER_TOKEN_IDS[start_role]
            end_id = DOMAIN_DELIMITER_TOKEN_IDS[end_role]
            if start_id == end_id:
                raise ValueError(f"{domain.name}: start/end delimiter ids collide")
    if missing:
        raise ValueError(f"missing domain delimiter token roles: {sorted(set(missing))}")


__all__ = [
    "DOMAIN_DELIMITER_ROLES",
    "DomainEdgeKind",
    "DomainKind",
    "DomainRoleKind",
    "ParseConfidence",
    "delimiter_token_ids",
    "validate_domain_delimiter_contract",
]

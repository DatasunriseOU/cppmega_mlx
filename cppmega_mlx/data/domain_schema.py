"""Versioned domain sidecar schema for cppmega world-code documents.

The tokenizer artifact is frozen.  Domain boundaries therefore use logical
roles mapped to existing ``<RESERVED_N>`` ids in ``tokenizer_contract.py``.
This module owns the integer enums used by parquet sidecars, parser outputs,
MLX route tests, and the Megatron sidecar converter.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import IntEnum
import json
from pathlib import Path
from typing import Any

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
    CONFIGURE = 13
    BASH = 20
    ZSH = 21
    SH = 22
    TCSH = 23
    SQL = 30
    COMPILER_DIAGNOSTIC = 40
    BUILD_DIAGNOSTIC = 41
    COMPILER_ERROR = 42
    BUILD_ERROR = 43
    LINKER_ERROR = 44
    TEST_OUTPUT = 45
    TOOL_OUTPUT = 46
    LINKER_DIAGNOSTIC = 47
    SANITIZER_OUTPUT = 48


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
    COMMENT = 21
    DOCSTRING = 22
    PREPROCESSOR = 23
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
    MACRO_PARAM_USE = 6
    MACRO_INVOCATION = 7
    MACRO_CONDITION = 8
    MACRO_REDEFINITION = 9
    MACRO_INCLUDE_ORDER = 10
    MACRO_EXPANSION_CONDITION = 11
    MACRO_EXPANSION_REDEFINITION = 12
    MACRO_EXPANSION_INCLUDE_ORDER = 13
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
    EMBEDDED_DOMAIN = 100


class ParseConfidence(IntEnum):
    ABSENT = 0
    RAW = 1
    HEURISTIC = 2
    PARTIAL = 3
    EXACT = 4


DOMAIN_SCHEMA_PATH = Path(__file__).with_name("domain_schema_v1.json")
DOMAIN_SCHEMA = json.loads(DOMAIN_SCHEMA_PATH.read_text(encoding="utf-8"))
if DOMAIN_SCHEMA.get("schema") != "cppmega_domain_sidecars_v1":
    raise RuntimeError(f"unsupported frozen domain schema: {DOMAIN_SCHEMA_PATH}")


def _enum_contract(enum_type: type[IntEnum], field: str) -> dict[str, int]:
    expected = DOMAIN_SCHEMA.get(field)
    actual = {item.name: int(item) for item in enum_type}
    if expected != actual:
        raise RuntimeError(
            f"{DOMAIN_SCHEMA_PATH}: {field} drift: expected={expected}, actual={actual}"
        )
    return actual


_enum_contract(DomainKind, "domain_kinds")
_enum_contract(DomainRoleKind, "role_kinds")
_enum_contract(DomainEdgeKind, "edge_kinds")
_enum_contract(ParseConfidence, "confidence_kinds")

DOMAIN_EDGE_FAMILIES: dict[str, frozenset[DomainEdgeKind]] = {
    family: frozenset(DomainEdgeKind(int(kind)) for kind in kinds)
    for family, kinds in DOMAIN_SCHEMA["edge_families"].items()
}


def domain_edge_family(kind: DomainEdgeKind | int) -> str:
    """Return the one graph channel that owns ``kind``."""

    try:
        edge_kind = DomainEdgeKind(int(kind))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown domain edge kind {kind!r}") from exc
    if edge_kind == DomainEdgeKind.UNKNOWN:
        raise ValueError("unknown domain edge kind 0")
    for family, kinds in DOMAIN_EDGE_FAMILIES.items():
        if edge_kind in kinds:
            return family
    raise ValueError(f"domain edge kind {int(edge_kind)} has no graph family")


def validate_domain_edge_kind(
    kind: DomainEdgeKind | int,
    *,
    family: str | None = None,
) -> DomainEdgeKind:
    """Validate an edge kind and its graph-channel family."""

    actual_family = domain_edge_family(kind)
    if family is not None and actual_family != family:
        raise ValueError(
            f"domain edge kind {int(kind)} belongs to {actual_family}, not {family}"
        )
    return DomainEdgeKind(int(kind))


def normalize_domain_edge_record(edge: Any, *, family: str) -> tuple[int, int, int]:
    """Normalize one serialized edge without inventing fields."""

    if hasattr(edge, "as_py"):
        edge = edge.as_py()
    if isinstance(edge, Mapping):
        endpoint_pairs = (
            ("from_char", "to_char"),
            ("from", "to"),
            ("src", "dst"),
        )
        selected = next(
            (
                (src_key, dst_key)
                for src_key, dst_key in endpoint_pairs
                if src_key in edge and dst_key in edge
            ),
            None,
        )
        if selected is None or "kind" not in edge:
            raise ValueError("domain edge missing src/dst/kind")
        src, dst, kind = edge[selected[0]], edge[selected[1]], edge["kind"]
    elif (
        isinstance(edge, Sequence)
        and not isinstance(edge, (str, bytes, bytearray))
        and len(edge) == 3
    ):
        src, dst, kind = edge
    else:
        raise ValueError("domain edge must be a mapping or length-3 sequence")
    src_i, dst_i = int(src), int(dst)
    if src_i < 0 or dst_i < 0:
        raise ValueError(f"domain edge endpoints must be non-negative: {src_i}->{dst_i}")
    edge_kind = validate_domain_edge_kind(int(kind), family=family)
    return src_i, dst_i, int(edge_kind)


DOMAIN_DELIMITER_ROLES: dict[DomainKind, tuple[str, str]] = {
    DomainKind(int(spec["domain_id"])): (str(spec["start"]), str(spec["end"]))
    for spec in DOMAIN_SCHEMA["delimiter_roles"].values()
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
    "DOMAIN_SCHEMA",
    "DOMAIN_SCHEMA_PATH",
    "DOMAIN_EDGE_FAMILIES",
    "DOMAIN_DELIMITER_ROLES",
    "DomainEdgeKind",
    "DomainKind",
    "DomainRoleKind",
    "ParseConfidence",
    "domain_edge_family",
    "delimiter_token_ids",
    "normalize_domain_edge_record",
    "validate_domain_delimiter_contract",
    "validate_domain_edge_kind",
]

from __future__ import annotations

from cppmega_mlx.data.diagnostic_parsers import (
    parse_build_error,
    parse_clang_diagnostic,
    parse_linker_error,
    parse_msvc_diagnostic,
)
from cppmega_mlx.data.domain_schema import DomainEdgeKind, DomainKind, DomainRoleKind


def _roles(parsed):
    out = {}
    for token, role in zip(parsed.tokens, parsed.role_ids):
        out.setdefault(token.text, []).append(role)
    return out


def _has_role(parsed, text: str, role: DomainRoleKind) -> bool:
    return int(role) in _roles(parsed).get(text, [])


def test_clang_diagnostic_parser_marks_location_and_primary_edge() -> None:
    parsed = parse_clang_diagnostic(
        "src/main.cpp:12:7: error: no matching function for call to 'foo'\n"
        "src/main.cpp:8:3: note: candidate function not viable\n"
    )
    roles = _roles(parsed)
    assert parsed.domain == DomainKind.COMPILER_ERROR
    assert _has_role(parsed, "src/main.cpp", DomainRoleKind.FILE)
    assert _has_role(parsed, "12", DomainRoleKind.LINE)
    assert _has_role(parsed, "7", DomainRoleKind.COLUMN)
    assert _has_role(parsed, "error", DomainRoleKind.SEVERITY)
    assert any(edge[2] == int(DomainEdgeKind.DIAG_PRIMARY_LOCATION) for edge in parsed.edges)
    assert any(edge[2] == int(DomainEdgeKind.DIAG_NOTE) for edge in parsed.edges)


def test_linker_parser_marks_undefined_symbol() -> None:
    parsed = parse_linker_error("ld: error: undefined reference to `Widget::run()'\n")
    roles = _roles(parsed)
    assert parsed.domain == DomainKind.LINKER_ERROR
    assert _has_role(parsed, "ld", DomainRoleKind.COMMAND)
    assert _has_role(parsed, "Widget", DomainRoleKind.SYMBOL)
    assert _has_role(parsed, "run", DomainRoleKind.SYMBOL)
    assert any(edge[2] == int(DomainEdgeKind.LINK_UNDEFINED_SYMBOL) for edge in parsed.edges)


def test_build_error_parser_marks_build_command_and_exit_code() -> None:
    parsed = parse_build_error("ninja: build stopped: subcommand failed with exit code 1\n")
    roles = _roles(parsed)
    assert parsed.domain == DomainKind.BUILD_ERROR
    assert _has_role(parsed, "ninja", DomainRoleKind.COMMAND)
    assert _has_role(parsed, "1", DomainRoleKind.EXIT_CODE)


def test_msvc_parser_marks_primary_location() -> None:
    parsed = parse_msvc_diagnostic("src\\main.cpp(10,5): error C2065: 'x': undeclared identifier\n")
    roles = _roles(parsed)
    assert parsed.domain == DomainKind.COMPILER_ERROR
    assert _has_role(parsed, "src\\main.cpp", DomainRoleKind.FILE)
    assert _has_role(parsed, "10", DomainRoleKind.LINE)
    assert _has_role(parsed, "5", DomainRoleKind.COLUMN)
    assert any(edge[2] == int(DomainEdgeKind.DIAG_PRIMARY_LOCATION) for edge in parsed.edges)

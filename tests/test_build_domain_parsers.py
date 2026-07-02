from __future__ import annotations

import numpy as np

from cppmega_mlx.data.build_parsers import (
    parse_autoconf,
    parse_bazel,
    parse_cmake,
    parse_make,
    parse_ninja,
)
from cppmega_mlx.data.domain_schema import DomainEdgeKind, DomainKind, DomainRoleKind


def _roles(parsed):
    out = {}
    for token, role in zip(parsed.tokens, parsed.role_ids):
        out.setdefault(token.text, []).append(role)
    return out


def _has_role(parsed, text: str, role: DomainRoleKind) -> bool:
    return int(role) in _roles(parsed).get(text, [])


def test_cmake_parser_marks_target_sources_and_libraries() -> None:
    parsed = parse_cmake(
        "add_executable(app main.cpp util.cc)\n"
        "target_link_libraries(app PRIVATE ssl crypto)\n"
    )
    roles = _roles(parsed)
    assert parsed.domain == DomainKind.CMAKE
    assert _has_role(parsed, "add_executable", DomainRoleKind.COMMAND)
    assert _has_role(parsed, "app", DomainRoleKind.TARGET)
    assert _has_role(parsed, "main.cpp", DomainRoleKind.SOURCE)
    assert _has_role(parsed, "ssl", DomainRoleKind.LIBRARY)
    assert any(edge[2] == int(DomainEdgeKind.BUILD_TARGET_DEP) for edge in parsed.edges)
    assert any(edge[2] == int(DomainEdgeKind.BUILD_TARGET_SOURCE) for edge in parsed.edges)

    packet = parsed.to_packet()
    assert packet.domain == DomainKind.CMAKE
    assert int(np.asarray(packet.domain_ids)[0]) == int(DomainKind.CMAKE)


def test_make_parser_marks_targets_prereqs_and_recipe_command() -> None:
    parsed = parse_make("app: main.o util.o\n\t$(CXX) -o app main.o util.o\n")
    roles = _roles(parsed)
    assert parsed.domain == DomainKind.MAKE
    assert _has_role(parsed, "app", DomainRoleKind.TARGET)
    assert _has_role(parsed, "main.o", DomainRoleKind.PREREQUISITE)
    assert any(edge[2] == int(DomainEdgeKind.BUILD_TARGET_DEP) for edge in parsed.edges)
    assert any(edge[2] == int(DomainEdgeKind.BUILD_RULE_COMMAND) for edge in parsed.edges)


def test_ninja_parser_marks_rule_build_inputs_outputs() -> None:
    parsed = parse_ninja("rule cc\n  command = clang++ -c $in -o $out\nbuild main.o: cc main.cc\n")
    roles = _roles(parsed)
    assert parsed.domain == DomainKind.NINJA
    assert int(DomainRoleKind.RULE) in roles["cc"]
    assert _has_role(parsed, "main.o", DomainRoleKind.OUTPUT)
    assert _has_role(parsed, "main.cc", DomainRoleKind.INPUT)
    assert any(edge[2] == int(DomainEdgeKind.BUILD_ACTION_INPUT) for edge in parsed.edges)


def test_bazel_parser_marks_rule_target_srcs_deps() -> None:
    parsed = parse_bazel('cc_library(name = "core", srcs = ["core.cc"], deps = ["//base:base"])\n')
    roles = _roles(parsed)
    assert parsed.domain == DomainKind.BAZEL
    assert _has_role(parsed, "cc_library", DomainRoleKind.RULE)
    assert _has_role(parsed, '"core"', DomainRoleKind.TARGET)
    assert _has_role(parsed, '"core.cc"', DomainRoleKind.SOURCE)
    assert _has_role(parsed, '"//base:base"', DomainRoleKind.LABEL)
    assert any(edge[2] == int(DomainEdgeKind.BUILD_TARGET_DEP) for edge in parsed.edges)


def test_autoconf_parser_marks_macros() -> None:
    parsed = parse_autoconf("AC_INIT([demo], [1.0])\nAC_PROG_CXX\n")
    roles = _roles(parsed)
    assert parsed.domain == DomainKind.AUTOCONF
    assert _has_role(parsed, "AC_INIT", DomainRoleKind.COMMAND)
    assert _has_role(parsed, "AC_PROG_CXX", DomainRoleKind.COMMAND)

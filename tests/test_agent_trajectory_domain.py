from __future__ import annotations

from cppmega_mlx.data.agent_trajectory import (
    parse_result_diagnostic_domain,
    parse_shell_action_domain,
)
from cppmega_mlx.data.domain_schema import DomainKind


def test_parse_shell_action_domain_keeps_unknown_shell_as_sh():
    parsed = parse_shell_action_domain("echo ok | tee out.txt")

    assert parsed.domain == DomainKind.SH
    assert parsed.metadata["shell_kind"] == "sh"


def test_parse_shell_action_domain_uses_explicit_dialect():
    parsed = parse_shell_action_domain("print -r -- $path", shell_kind="zsh")

    assert parsed.domain == DomainKind.ZSH
    assert parsed.metadata["shell_kind"] == "zsh"


def test_parse_result_diagnostic_domain_routes_compiler_and_linker_errors():
    compiler = parse_result_diagnostic_domain("src/main.cpp:10:5: error: no member named 'x'\n")
    linker = parse_result_diagnostic_domain("ld: error: undefined reference to `foo()'\n")

    assert compiler is not None
    assert compiler.domain == DomainKind.COMPILER_ERROR
    assert linker is not None
    assert linker.domain == DomainKind.LINKER_ERROR

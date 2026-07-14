from __future__ import annotations

from cppmega_mlx.data.domain_schema import DomainEdgeKind, DomainKind, DomainRoleKind
from cppmega_mlx.data.shell_parsers import parse_bash, parse_sh, parse_tcsh, parse_zsh


def _roles(parsed):
    out = {}
    for token, role in zip(parsed.tokens, parsed.role_ids):
        out.setdefault(token.text, []).append(role)
    return out


def _has_role(parsed, text: str, role: DomainRoleKind) -> bool:
    return int(role) in _roles(parsed).get(text, [])


def test_bash_parser_keeps_bash_domain_and_routes_pipe_and_redirect() -> None:
    parsed = parse_bash("CXX=clang++ cat input.txt | grep foo > out.txt\n")
    assert parsed.domain == DomainKind.BASH
    assert _has_role(parsed, "CXX", DomainRoleKind.ENVIRONMENT)
    assert _has_role(parsed, "cat", DomainRoleKind.COMMAND)
    assert _has_role(parsed, "grep", DomainRoleKind.COMMAND)
    assert _has_role(parsed, "out.txt", DomainRoleKind.PATH)
    assert any(edge[2] == int(DomainEdgeKind.SHELL_PIPE) for edge in parsed.edges)
    assert any(edge[2] == int(DomainEdgeKind.SHELL_REDIR_OUT) for edge in parsed.edges)


def test_shell_dialects_are_not_collapsed_to_bash() -> None:
    assert parse_sh("echo ok\n").domain == DomainKind.SH
    assert parse_zsh("echo ok\n").domain == DomainKind.ZSH
    assert parse_tcsh("echo ok\n").domain == DomainKind.TCSH


def test_shell_parser_resets_command_and_redirect_state_on_newline() -> None:
    parsed = parse_sh("cat first.txt\ngrep second.txt\n")
    command_by_text = {
        token.text: idx
        for idx, token in enumerate(parsed.tokens)
        if parsed.role_ids[idx] == int(DomainRoleKind.COMMAND)
    }
    second_idx = next(
        idx for idx, token in enumerate(parsed.tokens) if token.text == "second.txt"
    )
    assert command_by_text.keys() >= {"cat", "grep"}
    assert (
        command_by_text["grep"],
        second_idx,
        int(DomainEdgeKind.SHELL_COMMAND_FILE),
    ) in parsed.edges
    assert not any(
        src == command_by_text["cat"] and dst == second_idx
        for src, dst, _ in parsed.edges
    )

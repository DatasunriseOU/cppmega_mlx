"""Pytest wrapper around scripts/verify_tokenizer_roundtrip_compiles.py.

Imports the already-committed verification module and exercises its real logic:
for every case, encode -> decode -> clang-format -> clang++ -fsyntax-only must
compile (when the original compiles). Skips when clang++/clang-format are absent
so the suite stays green on machines without a C++ toolchain.

This is NOT a mock: it runs the real cppmega tokenizer and the real compiler.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "scripts" / "verify_tokenizer_roundtrip_compiles.py"


def _load_module():
    if not _SCRIPT.is_file():
        pytest.fail(
            f"committed roundtrip script missing: {_SCRIPT} "
            "(expected scripts/verify_tokenizer_roundtrip_compiles.py)"
        )
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))
    spec = importlib.util.spec_from_file_location(
        "verify_tokenizer_roundtrip_compiles", _SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MOD = _load_module()


def _require_toolchain() -> None:
    if _MOD.CXX is None:
        pytest.skip("no clang++/g++ available")
    if _MOD.FMT is None:
        pytest.skip("no clang-format available")


def _load_tokenizer():
    tok_path = _MOD.TOKENIZER
    if not Path(tok_path).is_file():
        pytest.skip(f"tokenizer artifact not present: {tok_path}")
    return _MOD.load_cppmega_tokenizer(tok_path)


def test_cases_present() -> None:
    assert isinstance(_MOD.CASES, dict)
    assert len(_MOD.CASES) >= 1


@pytest.mark.parametrize("name", sorted(_MOD.CASES.keys()))
def test_roundtrip_compiles_after_clang_format(name: str) -> None:
    _require_toolchain()
    tok = _load_tokenizer()
    original = _MOD.CASES[name]

    ids = tok.encode(original)
    decoded = tok.decode(ids)

    ok_orig, _ = _MOD.compiles(original)
    if not ok_orig:
        pytest.skip(f"case {name!r} original does not compile in this toolchain")

    formatted = _MOD.clang_format(decoded)
    ok_fmt, err_fmt = _MOD.compiles(formatted)
    assert ok_fmt, (
        f"case {name!r}: roundtrip+clang-format failed to compile; err={err_fmt!r}"
    )


def test_all_cases_compile_via_main() -> None:
    """End-to-end: the committed main() must return 0 (all cases pass)."""
    _require_toolchain()
    if not Path(_MOD.TOKENIZER).is_file():
        pytest.skip(f"tokenizer artifact not present: {_MOD.TOKENIZER}")
    rc = _MOD.main()
    assert rc == 0, f"verify_tokenizer_roundtrip_compiles.main() returned {rc}, expected 0"

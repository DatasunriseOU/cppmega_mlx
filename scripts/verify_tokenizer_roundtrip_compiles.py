#!/usr/bin/env python3
"""Verify the newest C++ tokenizer's encode→decode roundtrip yields COMPILABLE code.

The cppmega tokenizer collapses whitespace runs to <SPACE>/<NL> sentinels (encode)
and restores a single space / single newline (decode). So the roundtrip is NOT
byte-exact for indentation, blank-line runs, tabs, CRLF, or whitespace *inside*
string literals — but it preserves token structure and, crucially, exactly one
newline per source line (so `//` comments and `#include`/`#define` directives stay
valid). This script proves the practical property we care about (RULE #1, fail loud):

    original C++  →  encode  →  decode  →  [clang-format]  →  clang++ -fsyntax-only  ⇒ OK

For each case it reports: roundtrip byte-exact?, compiles raw (pre-format)?, compiles
after clang-format?. The bar is "compiles after clang-format", even if reformatted.

Run:  cppmega.mlx/.venv/bin/python scripts/verify_tokenizer_roundtrip_compiles.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from cppmega_mlx.tokenizer import load_cppmega_tokenizer  # noqa: E402

TOKENIZER = REPO / "cppmega_mlx" / "tokenizer" / "tokenizer.json"
CXX = shutil.which("clang++") or shutil.which("g++")
FMT = shutil.which("clang-format")
STD = "c++17"

# Self-contained, compilable translation units, each seeded with nasty whitespace/EOL.
CASES: dict[str, str] = {
    "indent+blank_lines": (
        "#include <vector>\n\n\n"
        "int sum(const std::vector<int>&    v) {\n"
        "        int    s = 0;\n\n"
        "        for (int x : v)   s += x;\n\n\n"
        "        return s;\n"
        "}\n"
    ),
    "tabs": (
        "int f(int a,int b){\n"
        "\t\tif(a>b)\n"
        "\t\t\treturn a;\n"
        "\t\treturn b;\n"
        "}\n"
    ),
    "crlf": (
        "#include <string>\r\n"
        "std::string greet(){\r\n"
        "    return std::string(\"hi\");\r\n"
        "}\r\n"
    ),
    "line_comment_then_code": (  # EOL criticality: comment must not swallow next line
        "int g(int x){\n"
        "    // increment then return   (lots of   spaces here)\n"
        "    ++x;\n"
        "    return x;\n"
        "}\n"
    ),
    "preprocessor+multiline_macro": (
        "#include <cstdio>\n"
        "#define   SQUARE(x)   ((x) * (x))\n"
        "#define LOG(msg) \\\n"
        "    do { std::printf(\"%s\\n\", msg); } while (0)\n"
        "#if 0\n"
        "this should be skipped by the preprocessor\n"
        "#endif\n"
        "int area(int s){ LOG(\"area\"); return SQUARE(s); }\n"
    ),
    "template_class": (
        "template <typename T>\n"
        "struct Box {\n"
        "    T    value;\n"
        "    T        get()   const { return value; }\n"
        "};\n"
        "int use(){ Box<int> b{42}; return b.get(); }\n"
    ),
    "string_with_spaces": (  # content of the string WILL change (lossy) but must compile
        "#include <string>\n"
        "std::string banner(){ return \"a      b\tc\"; }\n"
    ),
}


def run(cmd: list[str], inp: str | None = None) -> tuple[int, str]:
    p = subprocess.run(cmd, input=inp, capture_output=True, text=True)
    return p.returncode, (p.stderr or "")


def compiles(src: str) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False) as f:
        f.write(src)
        path = f.name
    try:
        rc, err = run([CXX, f"-std={STD}", "-fsyntax-only", "-w", path])
        return rc == 0, err.strip().splitlines()[-1] if err.strip() else ""
    finally:
        Path(path).unlink(missing_ok=True)


def clang_format(src: str) -> str:
    if not FMT:
        return src
    rc, _ = 0, ""
    p = subprocess.run([FMT, "--style=LLVM"], input=src, capture_output=True, text=True)
    return p.stdout if p.returncode == 0 and p.stdout else src


def main() -> int:
    if CXX is None:
        print("SKIP: no clang++/g++ available")
        return 0
    tok = load_cppmega_tokenizer(TOKENIZER)
    print(f"tokenizer: {TOKENIZER}  vocab={tok.vocab_size}")
    print(f"compiler: {CXX}   clang-format: {FMT or 'MISSING'}   std={STD}\n")

    n_fail = 0
    for name, original in CASES.items():
        ids = tok.encode(original)
        decoded = tok.decode(ids)
        exact = decoded == original
        ok_orig, _ = compiles(original)
        ok_raw, err_raw = compiles(decoded)
        formatted = clang_format(decoded)
        ok_fmt, err_fmt = compiles(formatted)
        # The bar: original compiles ⇒ roundtrip+format must compile.
        passed = (not ok_orig) or ok_fmt
        n_fail += 0 if passed else 1
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
        print(f"    byte_exact={exact}  orig_compiles={ok_orig}  "
              f"raw_compiles={ok_raw}  format_compiles={ok_fmt}")
        if not ok_raw and err_raw:
            print(f"    raw err: {err_raw}")
        if not passed and err_fmt:
            print(f"    FORMAT err: {err_fmt}")

    print()
    if n_fail:
        print(f"ROUNDTRIP→COMPILE: FAIL ({n_fail}/{len(CASES)} cases)")
        return 1
    print(f"ROUNDTRIP→COMPILE: OK (all {len(CASES)} cases compile after clang-format)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""H25: honest categorisation audit across V3/V4/V5/V6 tests.

Walks the relevant test directories and labels each `def test_*`
function by the kind of assertion it makes:

  🟢 math-effect  — asserts on a numerical quantity that requires
                    real math to land (loss, weight delta, norm,
                    routing entropy, mask effect, dtype cast,
                    bit-equality, etc.).
  🟡 propagation  — asserts that a config value made it through a
                    pipeline (string echo, status=="ok", path
                    matching, structural extras presence).
  🔴 decorative   — asserts on cosmetics only (testid exists, no
                    crash, length>0, opt object is non-null).

Heuristic-based classifier; intended for quick triage and as a
regression baseline. Writes `tests/fixtures/honesty_audit_v6.md`.

Run: python -m scripts.audit_v3_v4_v5
"""

from __future__ import annotations

import argparse
import ast
import collections
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent

MATH_PATTERNS = re.compile(
    r"\b("
    r"loss|losses|weight_delta|grad_norm|entropy|mask|cast|"
    r"dtype_actual|per_rank|routing|load_balance|l2_diff|"
    r"cos_sim|bit_identical|within|approx|isclose|"
    r"finite|inf\b|isnan"
    r")",
    re.IGNORECASE,
)
PROPAGATION_PATTERNS = re.compile(
    r"\b("
    r"status\s*==\s*['\"]ok|saved_path|loaded_path|"
    r"toBe\(['\"](true|false|null)|toContain|toBe\(['\"][a-z_]+['\"]"
    r"|equals|==.*['\"][a-z_]+['\"]"
    r")",
    re.IGNORECASE,
)


def classify_function_body(body: str) -> str:
    if MATH_PATTERNS.search(body):
        return "math-effect"
    if PROPAGATION_PATTERNS.search(body):
        return "propagation"
    return "decorative"


def walk_python_tests(root: pathlib.Path):
    for path in root.rglob("test_*.py"):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.FunctionDef)
                    and node.name.startswith("test_")):
                body = ast.unparse(node)
                yield (path.relative_to(REPO), node.name,
                       classify_function_body(body))


def walk_e2e_specs(root: pathlib.Path):
    for path in root.rglob("*.spec.ts"):
        text = path.read_text(encoding="utf-8")
        # Naive regex per test(...) block; good enough for triage.
        for m in re.finditer(
                r"test\(\s*['\"`]([^'\"`]+)['\"`]\s*,",
                text):
            name = m.group(1)
            start = m.end()
            # Slice the rest of the file for body matches; close
            # enough since most spec.ts files have one test per
            # `test(...)` declaration with short bodies.
            tail = text[start:start + 4000]
            yield (path.relative_to(REPO), name,
                   classify_function_body(tail))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=str(REPO / "tests" / "fixtures" / "honesty_audit_v6.md"))
    args = parser.parse_args()

    py_roots = [REPO / "tests" / "v4", REPO / "tests" / "v5"]
    e2e_root = REPO / "vbgui" / "e2e" / "scenarios"

    rows: list[tuple[str, str, str]] = []
    for r in py_roots:
        if r.exists():
            for relpath, name, label in walk_python_tests(r):
                rows.append((str(relpath), name, label))
    if e2e_root.exists():
        for relpath, name, label in walk_e2e_specs(e2e_root):
            rows.append((str(relpath), name, label))

    counts = collections.Counter(r[2] for r in rows)
    glyph = {"math-effect": "🟢", "propagation": "🟡",
             "decorative": "🔴"}

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# V6 honesty audit\n\n")
        f.write("Classifier from `scripts/audit_v3_v4_v5.py` "
                "(heuristic).\n\n")
        f.write("| Category | Count |\n|---|---|\n")
        for k in ("math-effect", "propagation", "decorative"):
            f.write(f"| {glyph[k]} {k} | {counts.get(k, 0)} |\n")
        f.write(f"\nTotal: {sum(counts.values())}\n\n")
        f.write("## By file\n\n")
        by_file = collections.defaultdict(list)
        for path, name, label in rows:
            by_file[path].append((name, label))
        for path in sorted(by_file):
            tests = by_file[path]
            by_label = collections.Counter(t[1] for t in tests)
            f.write(f"### {path}\n")
            f.write(f"{glyph['math-effect']}={by_label.get('math-effect',0)} "
                    f"{glyph['propagation']}={by_label.get('propagation',0)} "
                    f"{glyph['decorative']}={by_label.get('decorative',0)}"
                    f" / total {len(tests)}\n\n")
        f.write("\n## V7 follow-ups\n\n")
        decoratives = [r for r in rows if r[2] == "decorative"]
        if decoratives:
            f.write(f"Decorative-only tests ({len(decoratives)}) — "
                    "candidates for tightening:\n\n")
            for path, name, _ in decoratives[:50]:
                f.write(f"- `{path}::{name}`\n")
            if len(decoratives) > 50:
                f.write(f"- … {len(decoratives) - 50} more\n")
        else:
            f.write("None.\n")

    print(f"wrote {out_path}")
    print("counts:", dict(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "data" / "make_golden_mini.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("make_golden_mini", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_golden_mini_pipeline_uses_only_repository_local_tools() -> None:
    mod = _load_module()

    assert mod.PY == sys.executable
    assert mod.CPPMEGA_MLX_ROOT == ROOT
    assert mod.INDEX_PROJECT == ROOT / "tools" / "clang_indexer" / "index_project.py"
    assert mod.PROCESS_COMMITS == ROOT / "tools" / "clang_indexer" / "process_commits.py"
    assert not hasattr(mod, "NANOCHAT_ROOT")


def test_golden_mini_provenance_accepts_current_commit_header(tmp_path: Path) -> None:
    mod = _load_module()
    path = tmp_path / "enriched.jsonl"
    path.write_text(
        json.dumps(
            {
                "text": (
                    "/**\n"
                    " * @repo golden_mini/shapes\n"
                    " * File: src/scale.cpp\n"
                    " */\n"
                )
            }
        )
        + "\n",
        encoding="utf-8",
    )

    mod._merge_commit_provenance(path)
    row = json.loads(path.read_text(encoding="utf-8"))

    assert row["repo"] == "golden_mini/shapes"
    assert row["filepath"] == "src/scale.cpp"
    assert row["commit_hash"] == "0000000000000000000000000000000000000001"

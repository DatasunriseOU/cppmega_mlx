from __future__ import annotations

import json
import sys
from pathlib import Path


MLX_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = MLX_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def test_progress_writer_appends_jsonl(tmp_path):
    import streaming_conveyor

    path = tmp_path / "progress.jsonl"
    writer = streaming_conveyor.ProgressWriter(path)
    writer.emit("unit_done", stream="code", repo="repo", valid_tokens=1024)
    writer.emit("unit_failed", stream="commits", repo="repo", stage="test")

    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["event"] for row in rows] == ["unit_done", "unit_failed"]
    assert rows[0]["stream"] == "code"
    assert rows[0]["valid_tokens"] == 1024
    assert rows[1]["stage"] == "test"

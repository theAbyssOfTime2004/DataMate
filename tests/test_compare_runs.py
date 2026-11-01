"""compare_runs: agreement % and optional PA."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from datamate.config import PROJECT_ROOT

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.compare_runs import compare


def _write_sub(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def test_compare_runs_agreement(tmp_path: Path) -> None:
    a = _write_sub(
        tmp_path / "a" / "submission.jsonl",
        [
            {"id": "q1", "type": "number", "answer": 100},
            {"id": "q2", "type": "abstain"},
            {"id": "q3", "type": "string", "answer": "A"},
        ],
    )
    b = _write_sub(
        tmp_path / "b" / "submission.jsonl",
        [
            {"id": "q1", "type": "number", "answer": 100},
            {"id": "q2", "type": "number", "answer": 1},
            {"id": "q3", "type": "string", "answer": "B"},
        ],
    )
    report = compare(a.parent, b.parent)
    assert report["n_shared"] == 3
    assert report["agree"] == 1
    assert abs(report["agreement"] - 1 / 3) < 1e-9
    assert "abstain->number" in report["disagree_by_type"]
    assert "string->string" in report["disagree_by_type"]

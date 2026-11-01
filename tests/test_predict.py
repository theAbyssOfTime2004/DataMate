"""Offline predict entrypoint: synthetic split, fake LLM, no writes under data."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import openpyxl
import pytest

from datamate.config import settings
from datamate.llm.client import LLMUnavailable, UsageEntry
from datamate.predict import run_predict
from datamate.scoring.answer import Answer


class FakeClient:
    """Always raises LLMUnavailable so the pipeline abstains without needing a real model."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.usage_log: list[UsageEntry] = []

    def count_tokens(self, text: str) -> int:
        return math.ceil(len(text) / 2.5)

    def chat_json(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        schema_name: str,
        max_tokens: int | None = None,
        required_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        del messages, schema, max_tokens, required_keys
        self.calls.append(schema_name)
        raise LLMUnavailable("offline test: no LLM")


def _write_shop(biz_dir: Path) -> None:
    biz_dir.mkdir(parents=True, exist_ok=True)
    (biz_dir / "README.md").write_text(
        "# Shop\n\n## Quy ước\n- Dữ liệu có đến hết 15/08/2026.\n",
        encoding="utf-8",
    )
    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    ws.append(["Mã HĐ", "Ngày bán", "Tổng tiền hàng"])
    ws.append(["ORD01", "01/08/2026", "100000"])
    wb.save(biz_dir / "HoaDon.xlsx")
    wb.close()


def _snapshot_paths(root: Path) -> set[str]:
    return {str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def synthetic_split(tmp_path: Path) -> Path:
    split_dir = tmp_path / "syn_split"
    split_dir.mkdir(parents=True)
    (split_dir / "calendar.csv").write_text(
        "date,weekday,lunar_day,lunar_month,lunar_year,lunar_leap,holiday\n",
        encoding="utf-8",
    )
    _write_shop(split_dir / "data" / "biz_a")
    questions = [
        {
            "id": "q1",
            "business_id": "biz_a",
            "question": "Doanh thu tháng này?",
            "as_of_date": "2026-08-15",
        },
        {
            "id": "q2",
            "business_id": "biz_a",
            "question": "Có bao nhiêu đơn?",
            "as_of_date": "2026-08-15",
        },
    ]
    with (split_dir / "questions.jsonl").open("w", encoding="utf-8") as fh:
        for row in questions:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return split_dir


def test_predict_writes_one_line_per_question_outside_data(
    synthetic_split: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "out"
    before = _snapshot_paths(synthetic_split)
    # Keep other tests from seeing mutated globals after this one.
    monkeypatch.setattr(settings, "data_root", settings.data_root)
    monkeypatch.setattr(settings, "cache_dir", settings.cache_dir)

    submission = run_predict(
        synthetic_split,
        out_dir,
        workers=1,
        time_budget_s=18900,
        client=FakeClient(),
    )

    lines = [ln for ln in submission.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2
    ids = [json.loads(ln)["id"] for ln in lines]
    assert ids == ["q1", "q2"]
    for ln in lines:
        assert json.loads(ln)["type"] == "abstain"

    assert (out_dir / "logs" / "q1.json").is_file()
    assert (out_dir / "logs" / "q2.json").is_file()
    assert (out_dir / "timing.json").is_file()
    assert (out_dir / "work").is_dir()

    after = _snapshot_paths(synthetic_split)
    assert after == before
    # Caches live under out_dir/work, never under the split.
    for path in (out_dir / "work").rglob("*"):
        assert synthetic_split not in path.resolve().parents
        assert not str(path.resolve()).startswith(str(synthetic_split.resolve()))


def test_predict_time_budget_abstains_remaining(
    synthetic_split: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "out_budget"
    monkeypatch.setattr(settings, "data_root", settings.data_root)
    monkeypatch.setattr(settings, "cache_dir", settings.cache_dir)

    submission = run_predict(
        synthetic_split,
        out_dir,
        workers=1,
        time_budget_s=0,
        client=FakeClient(),
    )
    rows = [json.loads(ln) for ln in submission.read_text(encoding="utf-8").splitlines() if ln]
    assert len(rows) == 2
    assert all(r["type"] == "abstain" for r in rows)
    timing = json.loads((out_dir / "timing.json").read_text(encoding="utf-8"))
    assert timing["n_timed_out"] == 2
    log1 = json.loads((out_dir / "logs" / "q1.json").read_text(encoding="utf-8"))
    assert log1["trace"].get("abstain_reason") == "time_budget"


def test_predict_pool_completes(synthetic_split: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out_dir = tmp_path / "out_pool"
    monkeypatch.setattr(settings, "data_root", settings.data_root)
    monkeypatch.setattr(settings, "cache_dir", settings.cache_dir)
    submission = run_predict(
        synthetic_split,
        out_dir,
        workers=2,
        time_budget_s=60,
        client=FakeClient(),
    )
    ids = [
        json.loads(ln)["id"]
        for ln in submission.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert ids == ["q1", "q2"]
    # Answer type from FakeClient path is abstain (LLM unavailable).
    assert Answer.abstain().type == "abstain"

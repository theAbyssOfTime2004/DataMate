from __future__ import annotations

import json

from evals.check_submission import check_lines


def _line(**row: object) -> str:
    return json.dumps(row, ensure_ascii=False)


def test_valid_submission_passes() -> None:
    lines = [
        _line(id="q1", type="number", answer=4820000),
        _line(id="q2", type="percent", answer=12.4),
        _line(id="q3", type="date", answer="2025-09"),
        _line(id="q4", type="list", answer=["Cà phê sữa", 3]),
        _line(id="q5", type="abstain"),
        _line(id="q6", type="unanswerable", reason="no data"),
        _line(id="q7", type="table", answer={"columns": ["a", "b"], "rows": [["x", 1]]}),
    ]
    result = check_lines(lines, [f"q{i}" for i in range(1, 8)])
    assert result.ok, result.errors
    assert result.types["abstain"] == 1


def test_format_errors_are_reported() -> None:
    lines = [
        '{"id": "q1", "type": "number", "answer": NaN}',
        _line(id="q2", type="number", answer="4820000"),
        _line(id="q3", type="date", answer="15/08/2026"),
        _line(id="q4", type="string", answer=["Sting"]),
        _line(id="q5", type="percent"),
        _line(id="q6", type="maybe", answer=1),
        _line(id="q6", type="abstain"),
        _line(id="zz", type="abstain"),
        _line(id="q8", type="date", answer="2026-02-30"),
    ]
    result = check_lines(lines, [f"q{i}" for i in range(1, 10)])
    text = "\n".join(result.errors)
    assert "line 1: invalid JSON" in text
    assert "(q2): number must be a finite JSON number" in text
    assert "(q3): date must be" in text
    assert "(q4): string must be non-empty" in text
    assert "(q5): percent has no answer" in text
    assert "(q6): type 'maybe'" in text
    assert "duplicate id" in text
    assert "(zz): id is not in questions.jsonl" in text
    assert "(q8): date '2026-02-30' is not a real date" in text
    assert "have no line" in text
    assert not result.ok


def test_percent_with_extra_decimals_is_a_warning() -> None:
    result = check_lines([_line(id="q1", type="percent", answer=12.345)], ["q1"])
    assert result.ok
    assert result.warnings


def test_missing_lines_can_be_allowed_for_subsets() -> None:
    lines = [_line(id="q1", type="abstain")]
    assert not check_lines(lines, ["q1", "q2"]).ok
    allowed = check_lines(lines, ["q1", "q2"], allow_missing=True)
    assert allowed.ok and allowed.warnings

from __future__ import annotations

import unicodedata

import pytest

from datamate.config import settings
from datamate.io.questions import Reference, load_questions, load_references
from datamate.scoring.scorer import score_file, score_one


def test_score_number_rulebook_examples() -> None:
    ref_int = Reference(id="q1", type="number", answer=4820000)
    ref_float = Reference(id="q2", type="number", answer=18.6)

    assert score_one({"id": "q1", "type": "number", "answer": 4820000}, ref_int) == 1
    assert score_one({"id": "q2", "type": "number", "answer": 18.6}, ref_float) == 1
    assert score_one({"id": "q2", "type": "number", "answer": 18.6 + 1e-12}, ref_float) == 1

    for wrong in ["4,82 triệu", "4.820.000", "4820K", "4820000", True, False, None]:
        assert score_one({"id": "q1", "type": "number", "answer": wrong}, ref_int) == -1


def test_score_percent_rulebook_examples() -> None:
    ref = Reference(id="q1", type="percent", answer=12.4)

    assert score_one({"id": "q1", "type": "percent", "answer": 12.4}, ref) == 1
    assert score_one({"id": "q1", "type": "percent", "answer": 12.404}, ref) == 1
    assert score_one({"id": "q1", "type": "percent", "answer": "12,4%"}, ref) == -1
    assert score_one({"id": "q1", "type": "percent", "answer": 0.124}, ref) == -1
    assert score_one({"id": "q1", "type": "percent", "answer": True}, ref) == -1

    # Negative rounding half-up edge cases
    ref_neg = Reference(id="q2", type="percent", answer=-12.45)
    assert score_one({"id": "q2", "type": "percent", "answer": -12.454}, ref_neg) == 1
    assert score_one({"id": "q2", "type": "percent", "answer": -12.446}, ref_neg) == 1
    assert score_one({"id": "q2", "type": "percent", "answer": -12.455}, ref_neg) == -1


def test_score_date_rulebook_examples() -> None:
    ref_day = Reference(id="q1", type="date", answer="2026-08-15")
    ref_month = Reference(id="q2", type="date", answer="2025-09")

    assert score_one({"id": "q1", "type": "date", "answer": "2026-08-15"}, ref_day) == 1
    assert score_one({"id": "q2", "type": "date", "answer": "2025-09"}, ref_month) == 1

    assert score_one({"id": "q1", "type": "date", "answer": "15/08/2026"}, ref_day) == -1
    assert score_one({"id": "q2", "type": "date", "answer": "tháng 9/2025"}, ref_month) == -1
    assert score_one({"id": "q1", "type": "date", "answer": 20260815}, ref_day) == -1


def test_score_boolean_rulebook_examples() -> None:
    ref = Reference(id="q1", type="boolean", answer=True)

    assert score_one({"id": "q1", "type": "boolean", "answer": True}, ref) == 1
    assert score_one({"id": "q1", "type": "boolean", "answer": False}, ref) == -1
    assert score_one({"id": "q1", "type": "boolean", "answer": "có"}, ref) == -1
    assert score_one({"id": "q1", "type": "boolean", "answer": 1}, ref) == -1

    # Also ensure int 1 is not confused with True when reference is number 1
    ref_one = Reference(id="q2", type="number", answer=1)
    assert score_one({"id": "q2", "type": "number", "answer": True}, ref_one) == -1
    assert score_one({"id": "q2", "type": "number", "answer": 1}, ref_one) == 1


def test_score_string_rulebook_examples() -> None:
    ref = Reference(
        id="q1",
        type="string",
        answer="Sting dâu 330ml",
        alias=["SP0001"],
    )

    assert score_one({"id": "q1", "type": "string", "answer": "Sting dâu 330ml"}, ref) == 1
    assert score_one({"id": "q1", "type": "string", "answer": "sting  DÂU 330ml!"}, ref) == 1
    assert (
        score_one(
            {"id": "q1", "type": "string", "answer": unicodedata.normalize("NFD", "Sting dâu 330ml")},
            ref,
        )
        == 1
    )
    assert score_one({"id": "q1", "type": "string", "answer": "SP0001"}, ref) == 1
    assert score_one({"id": "q1", "type": "string", "answer": "  sp0001. "}, ref) == 1

    # Wrong diacritics or wrong format
    assert score_one({"id": "q1", "type": "string", "answer": "Sting dầu 330ml"}, ref) == -1
    assert score_one({"id": "q1", "type": "string", "answer": ["Sting dâu 330ml"]}, ref) == -1


def test_score_list_rulebook_examples() -> None:
    ref_single = Reference(id="q1", type="list", answer=["Cà phê sữa"], alias=[["SP0003"]])
    assert score_one({"id": "q1", "type": "list", "answer": ["Cà phê sữa"]}, ref_single) == 1
    assert score_one({"id": "q1", "type": "list", "answer": ["sp0003"]}, ref_single) == 1

    ref_multi = Reference(
        id="q2",
        type="list",
        answer=["Cà phê sữa", "Bạc xỉu"],
        alias=[["SP0003"], ["SP0005"]],
    )
    assert (
        score_one({"id": "q2", "type": "list", "answer": ["Cà phê sữa", "Bạc xỉu"]}, ref_multi)
        == 1
    )
    assert (
        score_one({"id": "q2", "type": "list", "answer": ["Bạc xỉu", "Cà phê sữa"]}, ref_multi)
        == -1
    )
    assert (
        score_one({"id": "q2", "type": "list", "answer": "Cà phê sữa, Bạc xỉu"}, ref_multi) == -1
    )

    ref_nums = Reference(id="q3", type="list", answer=[3, 10], alias=[])
    assert score_one({"id": "q3", "type": "list", "answer": [3, 10]}, ref_nums) == 1
    assert score_one({"id": "q3", "type": "list", "answer": [10, 3]}, ref_nums) == -1
    assert score_one({"id": "q3", "type": "list", "answer": "3 thùng 10 chai"}, ref_nums) == -1


def test_score_table_rulebook_examples() -> None:
    ref = Reference(
        id="q1",
        type="table",
        answer={
            "columns": ["khach_hang", "so_tien_no"],
            "rows": [
                ["Chị Lan", 3200000],
                ["Anh Minh", 1500000],
            ],
        },
        alias=[["KH001"], ["KH002"]],
    )

    # Exact match
    assert (
        score_one(
            {
                "id": "q1",
                "type": "table",
                "answer": {
                    "columns": ["khach_hang", "so_tien_no"],
                    "rows": [["Chị Lan", 3200000], ["Anh Minh", 1500000]],
                },
            },
            ref,
        )
        == 1
    )

    # Rows shuffled still correct
    assert (
        score_one(
            {
                "id": "q1",
                "type": "table",
                "answer": {
                    "columns": ["khach_hang", "so_tien_no"],
                    "rows": [["Anh Minh", 1500000], ["Chị Lan", 3200000]],
                },
            },
            ref,
        )
        == 1
    )

    # Renamed columns matched by position correct
    assert (
        score_one(
            {
                "id": "q1",
                "type": "table",
                "answer": {
                    "columns": ["ten_khach", "cong_no"],
                    "rows": [["Chị Lan", 3200000], ["Anh Minh", 1500000]],
                },
            },
            ref,
        )
        == 1
    )

    # Alias on first cell accepted
    assert (
        score_one(
            {
                "id": "q1",
                "type": "table",
                "answer": {
                    "columns": ["khach_hang", "so_tien_no"],
                    "rows": [["KH001", 3200000], ["kh002", 1500000]],
                },
            },
            ref,
        )
        == 1
    )

    # Formatted string cell "3.200.000đ" is wrong
    assert (
        score_one(
            {
                "id": "q1",
                "type": "table",
                "answer": {
                    "columns": ["khach_hang", "so_tien_no"],
                    "rows": [["Chị Lan", "3.200.000đ"], ["Anh Minh", 1500000]],
                },
            },
            ref,
        )
        == -1
    )


def test_score_unanswerable_abstain_and_missing() -> None:
    ref_unans = Reference(id="q1", type="unanswerable", answer=None)
    ref_num = Reference(id="q2", type="number", answer=100)

    # unanswerable vs unanswerable -> +1
    assert score_one({"id": "q1", "type": "unanswerable", "reason": "Ngoài kỳ"}, ref_unans) == 1
    assert score_one({"id": "q1", "type": "unanswerable"}, ref_unans) == 1

    # value vs unanswerable -> -1
    assert score_one({"id": "q1", "type": "number", "answer": 0}, ref_unans) == -1

    # unanswerable vs value -> -1
    assert score_one({"id": "q2", "type": "unanswerable", "reason": "Không có"}, ref_num) == -1

    # non-string reason -> treated as missing -> 0
    assert score_one({"id": "q1", "type": "unanswerable", "reason": 123}, ref_unans) == 0
    assert score_one({"id": "q2", "type": "unanswerable", "reason": 123}, ref_num) == 0

    # abstain and missing -> 0
    assert score_one({"id": "q1", "type": "abstain"}, ref_unans) == 0
    assert score_one({"id": "q2", "type": "abstain"}, ref_num) == 0
    assert score_one(None, ref_num) == 0
    assert score_one({"id": "q2", "type": "unknown_type", "answer": 100}, ref_num) == 0


def test_score_file_duplicates_and_malformed_lines() -> None:
    refs = {
        "q1": Reference(id="q1", type="number", answer=10),
        "q2": Reference(id="q2", type="number", answer=20),
        "q3": Reference(id="q3", type="number", answer=30),
    }
    lines = [
        "not valid json",
        '{"type": "number", "answer": 10}',  # missing id
        '{"id": "q1", "type": "number", "answer": 10}',
        '{"id": "q1", "type": "number", "answer": 999}',  # duplicate, first kept
        '{"id": "q2", "type": "number", "answer": 999}',  # wrong
        # q3 missing -> 0
    ]
    res = score_file(lines, refs, ["q1", "q2", "q3"])
    assert res.N == 3
    assert res.scores == {"q1": 1, "q2": -1, "q3": 0}
    assert res.correct == 1
    assert res.wrong == 1
    assert res.answered == 2
    assert res.PA == 0.0
    assert res.coverage == pytest.approx(2 / 3)
    assert res.precision == pytest.approx(0.5)
    assert any("Duplicate" in w for w in res.warnings)


def test_score_train_answers_self_consistency() -> None:
    train_dir = settings.data_root / "train"
    if not train_dir.exists():
        pytest.skip(f"Train data folder absent at {train_dir}")

    questions = load_questions("train")
    refs = load_references("train")
    qids = [q.id for q in questions]

    res = score_file(train_dir / "answers.jsonl", refs, qids)
    assert res.N == 3000
    assert res.correct == 3000
    assert res.wrong == 0
    assert res.answered == 3000
    assert res.PA == 1.0
    assert res.coverage == 1.0
    assert res.precision == 1.0

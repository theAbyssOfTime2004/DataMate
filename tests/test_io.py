from __future__ import annotations

import json
import unicodedata

import pytest

from datamate.config import settings
from datamate.io.questions import Question, load_questions, load_references
from datamate.scoring.answer import ANSWER_TYPES, Answer, to_submission_line


def test_question_nfc_normalisation() -> None:
    nfd_str = unicodedata.normalize("NFD", "Cửa hàng bán được bao nhiêu?")
    assert nfd_str != unicodedata.normalize("NFC", nfd_str)
    q = Question(
        id="q_1",
        business_id="biz_001",
        question=nfd_str,
        as_of_date="2026-08-15",
    )
    assert unicodedata.is_normalized("NFC", q.question)


def test_load_train_questions_and_references() -> None:
    if not (settings.data_root / "train").exists():
        pytest.skip(f"Train data folder absent at {settings.data_root / 'train'}")

    questions = load_questions("train")
    refs = load_references("train")

    assert len(questions) == 3000
    assert len(refs) == 3000

    valid_ref_types = set(ANSWER_TYPES) - {"abstain"}
    for q in questions:
        assert unicodedata.is_normalized("NFC", q.question)
        assert q.id in refs, f"Missing reference for {q.id}"
        assert refs[q.id].type in valid_ref_types


def test_load_references_missing_answers_raises() -> None:
    if not (settings.data_root / "public_test").exists():
        pytest.skip(f"public_test folder absent at {settings.data_root / 'public_test'}")

    with pytest.raises(FileNotFoundError, match="answers.jsonl"):
        load_references("public_test")


def test_to_submission_line_variants_and_roundtrip() -> None:
    cases = [
        ("q_num", Answer(type="number", answer=4820000), {"id": "q_num", "type": "number", "answer": 4820000}),
        ("q_pct", Answer(type="percent", answer=12.4), {"id": "q_pct", "type": "percent", "answer": 12.4}),
        ("q_str", Answer(type="string", answer="Sting dâu 330ml"), {"id": "q_str", "type": "string", "answer": "Sting dâu 330ml"}),
        ("q_abs", Answer.abstain(), {"id": "q_abs", "type": "abstain"}),
        (
            "q_unans_r",
            Answer.unanswerable("Không có dữ liệu"),
            {"id": "q_unans_r", "type": "unanswerable", "reason": "Không có dữ liệu"},
        ),
        (
            "q_unans_no_r",
            Answer.unanswerable(),
            {"id": "q_unans_no_r", "type": "unanswerable"},
        ),
    ]
    for qid, ans, expected in cases:
        line = to_submission_line(qid, ans)
        assert line == expected
        dumped = json.dumps(line, ensure_ascii=False)
        assert json.loads(dumped) == expected

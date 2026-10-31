from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import pytest

from datamate.config import settings
from datamate.query.ext.prompts import EXT_FEW_SHOT_SETS
from datamate.query.prompts import FEW_SHOT_SETS


def _normalise(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.casefold())
    without_marks = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    folded = without_marks.replace("đ", "d")
    return re.sub(r"\d", "9", folded)


def _question_files() -> list[Path]:
    root = settings.data_root
    return [
        path
        for path in (root / "train" / "questions.jsonl", root / "public_test" / "questions.jsonl")
        if path.is_file()
    ]


def test_few_shot_questions_stay_below_the_train_similarity_cutoff() -> None:
    paths = _question_files()
    if not paths:
        pytest.skip("train and public_test questions.jsonl are absent")

    references: list[str] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            references.append(_normalise(json.loads(line)["question"]))

    for group in (*FEW_SHOT_SETS, *EXT_FEW_SHOT_SETS):
        for example in group:
            left = _normalise(example.question)
            closest = max(SequenceMatcher(None, left, right).ratio() for right in references)
            assert closest < 0.80, f"{example.question!r} similarity {closest:.3f}"

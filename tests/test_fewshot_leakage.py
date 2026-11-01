from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import pytest

from datamate.config import settings
from datamate.query.ext.prompts import EXT_FEW_SHOT_SETS
from datamate.query.ext.verify import VERIFY_FEW_SHOTS
from datamate.query.prompts import FEW_SHOT_SETS
from datamate.rules import from_rulebook as rulebook_prompts


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

    examples = [
        *(ex for group in (*FEW_SHOT_SETS, *EXT_FEW_SHOT_SETS) for ex in group),
        *VERIFY_FEW_SHOTS,
    ]
    for example in examples:
        left = _normalise(example.question)
        closest = max(SequenceMatcher(None, left, right).ratio() for right in references)
        assert closest < 0.80, f"{example.question!r} similarity {closest:.3f}"

    # E2 rulebook few-shots (invented English rows; still must not echo train questions).
    rulebook_bits = [
        *(f"{name} {text}" for _section, name, text, _knob in rulebook_prompts._CLASSIFY_FEW_SHOTS),
        *(text for text, _val in rulebook_prompts._EXTRACT_FEW_SHOTS.values()),
    ]
    for bit in rulebook_bits:
        left = _normalise(bit)
        closest = max(SequenceMatcher(None, left, right).ratio() for right in references)
        assert closest < 0.80, f"rulebook few-shot {bit!r} similarity {closest:.3f}"

from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import pytest

from datamate.config import settings
from datamate.query.defined_terms import RANKING_ROW_FEW_SHOTS, TERM_FEW_SHOTS
from datamate.query.ext.prompts import EXT_FEW_SHOT_SETS
from datamate.query.ext.verify import VERIFY_FEW_SHOTS
from datamate.query.plan import parse as plan_parse
from datamate.query.product_match import PRODUCT_MATCH_FEW_SHOTS
from datamate.query.prompts import FEW_SHOT_SETS
from datamate.query.rank import RANK_FEW_SHOT_SETS
from datamate.query.subjective import SUBJECTIVE_FEW_SHOTS
from datamate.rules import from_rulebook as rulebook_prompts
from datamate.rules.parse_rulebook import parse_rulebook


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
        *SUBJECTIVE_FEW_SHOTS,
        *PRODUCT_MATCH_FEW_SHOTS,
        *TERM_FEW_SHOTS,
        *(ex for group in RANK_FEW_SHOT_SETS for ex in group),
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

    # E3 plan compose few-shots (invented English questions/definitions).
    for question, note, _plan in plan_parse._COMPOSE_FEW_SHOTS:
        for bit in (question, note):
            left = _normalise(bit)
            closest = max(SequenceMatcher(None, left, right).ratio() for right in references)
            assert closest < 0.80, f"plan few-shot {bit!r} similarity {closest:.3f}"


def test_invented_term_definitions_do_not_echo_the_rulebooks() -> None:
    root = settings.data_root
    books = [root / split / "rulebook.md" for split in ("train", "public_test")]
    rows = [row for path in books if path.is_file() for row in parse_rulebook(path)]
    if not rows:
        pytest.skip("rulebook.md files are absent")
    references = [_normalise(f"{row.name} {row.text}") for row in rows]
    bits = [
        *(f"{ex.term} {ex.definition}" for ex in TERM_FEW_SHOTS),
        *(f"{ex.term} {ex.definition}" for ex in RANKING_ROW_FEW_SHOTS),
        *(f"{ex.term} {ex.definition}" for group in RANK_FEW_SHOT_SETS for ex in group),
    ]
    for bit in bits:
        left = _normalise(bit)
        closest = max(SequenceMatcher(None, left, right).ratio() for right in references)
        assert closest < 0.80, f"term few-shot {bit!r} similarity {closest:.3f}"

from __future__ import annotations

import datetime
import json
import unicodedata
from typing import Any

from pydantic import BaseModel, Field, field_validator

from datamate.config import settings


def _nfc_recursive(value: Any) -> Any:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, list):
        return [_nfc_recursive(item) for item in value]
    if isinstance(value, dict):
        return {_nfc_recursive(k): _nfc_recursive(v) for k, v in value.items()}
    return value


class Question(BaseModel):
    id: str
    business_id: str
    question: str
    as_of_date: datetime.date

    @field_validator("question", mode="after")
    @classmethod
    def _normalize_question(cls, value: str) -> str:
        return unicodedata.normalize("NFC", value)


class Reference(BaseModel):
    id: str
    type: str
    answer: Any = None
    alias: list[Any] = Field(default_factory=list)

    @field_validator("answer", "alias", mode="after")
    @classmethod
    def _normalize_fields(cls, value: Any) -> Any:
        return _nfc_recursive(value)


def load_questions(split: str) -> list[Question]:
    split_path = settings.split_dir(split)
    questions_file = split_path / "questions.jsonl"
    if not questions_file.exists():
        raise FileNotFoundError(
            f"questions.jsonl does not exist for split {split!r} at {questions_file}"
        )
    questions: list[Question] = []
    with questions_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            questions.append(Question.model_validate(json.loads(line)))
    return questions


def load_references(split: str) -> dict[str, Reference]:
    split_path = settings.split_dir(split)
    answers_file = split_path / "answers.jsonl"
    if not answers_file.exists():
        raise FileNotFoundError(
            f"answers.jsonl does not exist for split {split!r} at {answers_file}"
        )
    references: dict[str, Reference] = {}
    with answers_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            ref = Reference.model_validate(json.loads(line))
            references[ref.id] = ref
    return references

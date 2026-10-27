from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

ANSWER_TYPES: tuple[str, ...] = (
    "number",
    "percent",
    "date",
    "boolean",
    "string",
    "list",
    "table",
    "unanswerable",
    "abstain",
)

AnswerType = Literal[
    "number",
    "percent",
    "date",
    "boolean",
    "string",
    "list",
    "table",
    "unanswerable",
    "abstain",
]


class Answer(BaseModel):
    type: AnswerType
    answer: Any = None
    reason: str | None = None

    @classmethod
    def abstain(cls) -> Answer:
        return cls(type="abstain")

    @classmethod
    def unanswerable(cls, reason: str | None = None) -> Answer:
        return cls(type="unanswerable", reason=reason)


def to_submission_line(question_id: str, answer: Answer) -> dict[str, Any]:
    if answer.type == "abstain":
        return {"id": question_id, "type": "abstain"}
    if answer.type == "unanswerable":
        out: dict[str, Any] = {"id": question_id, "type": "unanswerable"}
        if answer.reason is not None:
            out["reason"] = answer.reason
        return out
    return {
        "id": question_id,
        "type": answer.type,
        "answer": answer.answer,
    }

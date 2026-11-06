from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from datamate.query.missing_attributes import (
    ASK_FEW_SHOTS,
    asked_customer_attribute,
    attribute_column_p,
    column_values,
    value_recorded,
)


@dataclass
class _Table:
    name: str
    df: pd.DataFrame


class _Client:
    def __init__(self, raw: Any, p_yes: float = 0.0) -> None:
        self.raw = raw
        self.p_yes = p_yes
        self.seen: list[list[dict[str, str]]] = []

    def chat_json(self, messages: list[dict[str, str]], *args: Any, **kwargs: Any) -> Any:
        return self.raw

    def chat_choice_probs(self, messages: list[dict[str, str]], choices: list[str]) -> dict[str, float]:
        self.seen.append(messages)
        return {"yes": self.p_yes, "no": 1 - self.p_yes}


def test_asked_attribute_must_be_in_the_question() -> None:
    q = "Tháng 5 khách nữ mua bao nhiêu tiền?"
    asked = asked_customer_attribute(q, _Client({"attribute": "gender", "value": "nữ"}))
    assert asked is not None and asked.value == "nữ"
    assert asked_customer_attribute(q, _Client({"attribute": "gender", "value": "nam"})) is None
    assert asked_customer_attribute(q, _Client({"attribute": None, "value": None})) is None
    defined = "Tháng 9 có mấy khách quen, tức khách mua từ 2 lần trở lên?"
    assert asked_customer_attribute(defined, _Client({"attribute": "group", "value": "quen"})) is None


def test_value_search_and_column_prompt() -> None:
    tables = [
        _Table("sales", pd.DataFrame({"Khách": ["Trần Nam Đạt", "Lê Hoa"], "__row": [1, 2]})),
        _Table("cust", pd.DataFrame({"Phái": ["F", "M"]})),
    ]
    columns = column_values(tables)
    assert [header for _t, header, _v in columns] == ["Khách", "Phái"]
    assert value_recorded("nam", columns)  # inside a name: kept as data that may exist
    assert not value_recorded("nữ", columns)
    client = _Client(None, p_yes=0.9)
    assert attribute_column_p("gender", columns, client) == 0.9
    assert "cust | Phái | f, m" in client.seen[0][1]["content"]


def test_few_shot_values_are_in_their_questions() -> None:
    from datamate.query.grounding import QuestionText

    for text, _attribute, value in ASK_FEW_SHOTS:
        assert value is None or QuestionText(text).mentions_value(value)

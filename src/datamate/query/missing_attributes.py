"""A question about a customer attribute the shop records nowhere.

"khách nữ", "khách ở đường X", "khách nhóm thân thiết": the rulebook counts a question
as unanswerable when the shop does not keep the kind of information it needs. Decided
only after the pipeline abstained, and only when both checks agree:

- the asked value ("nữ", "Mai Chí Thọ") is written in no cell of the shop's raw files;
- given every column header of the shop with a few sample values, the model says no
  column records that kind of customer information (a gender coded F/M still counts).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datamate.llm.client import LLMError, LLMUnavailable
from datamate.query.grounding import QuestionText, fold

_ASK_SYSTEM = "\n".join(  # noqa: FLY002
    [
        "A Vietnamese shop owner asks a question. If it picks customers by one of their",
        "own attributes (gender, address or street, area, age, customer group or tier),",
        "return that attribute kind in English and the asked value copied from the",
        "question without the attribute word (no 'khách', 'đường', 'nhóm').",
        "Return null for both when the question picks no customers by an attribute, or",
        "picks them by what they bought, paid, owe or how many orders they made.",
        'Return JSON {"attribute": <string or null>, "value": <string or null>}.',
    ]
)

# Invented questions (no train/public text).
ASK_FEW_SHOTS: tuple[tuple[str, str | None, str | None], ...] = (
    ("Quý 3 các bà mẹ ở phường Tân Định mua mấy đơn bỉm?", "address", "Tân Định"),
    ("Mấy chị hạng vàng của tiệm hồi tuần rồi xài hết bao nhiêu?", "customer group", "vàng"),
    ("Khách đàn ông tháng 2 mua được bao nhiêu ly trà?", "gender", "đàn ông"),
    ("Năm ngoái có bao nhiêu người quay lại mua lần thứ hai sau Tết?", None, None),
    ("Hôm qua khách trả bằng ví điện tử bao nhiêu tiền?", None, None),
)
_ASK_SCHEMA = {
    "type": "object",
    "properties": {
        "attribute": {"type": ["string", "null"]},
        "value": {"type": ["string", "null"]},
    },
    "required": ["attribute", "value"],
    "additionalProperties": False,
}

_HAS_SYSTEM = "\n".join(  # noqa: FLY002
    [
        "You see every column of a shop's data files: table, column header and a few",
        "values. Answer yes if some column records the named kind of information about",
        "customers (even coded, abbreviated or inside a longer text such as a full",
        "address), else no. Answer with one word: yes or no.",
    ]
)
_SAMPLES = 4


@dataclass(frozen=True)
class AskedAttribute:
    attribute: str
    value: str


def asked_customer_attribute(question: str, client: Any) -> AskedAttribute | None:
    """The customer attribute and value the question filters on, grounded in its words.

    A question with a threshold ("xe vô sửa từ 2 lần trở lên") picks customers by a
    count it defines itself, not by a recorded attribute.
    """
    if QuestionText(question).has_threshold():
        return None
    messages: list[dict[str, str]] = [{"role": "system", "content": _ASK_SYSTEM}]
    for text, attribute, value in ASK_FEW_SHOTS:
        messages.append({"role": "user", "content": f"Question: {text}"})
        messages.append(
            {"role": "assistant", "content": _json({"attribute": attribute, "value": value})}
        )
    messages.append({"role": "user", "content": f"Question: {question}"})
    try:
        raw = client.chat_json(messages, _ASK_SCHEMA, "CustomerAttribute", max_tokens=60)
    except LLMUnavailable:
        raise
    except LLMError:
        return None
    if not isinstance(raw, dict):
        return None
    attribute, value = raw.get("attribute"), raw.get("value")
    if not isinstance(attribute, str) or not isinstance(value, str):
        return None
    if not fold(attribute) or not fold(value) or not QuestionText(question).mentions_value(value):
        return None
    return AskedAttribute(attribute.strip(), value.strip())


def column_values(tables: list[Any]) -> list[tuple[str, str, set[str]]]:
    """(table, header, folded distinct values) for every raw column."""
    out: list[tuple[str, str, set[str]]] = []
    for table in tables:
        frame = table.df
        name = str(getattr(table, "name", ""))
        for column in frame.columns:
            if str(column).startswith("__"):
                continue
            values = {fold(str(v)) for v in frame[column].dropna().unique()}
            out.append((name, str(column), {v for v in values if v}))
    return out


def value_recorded(value: str, columns: list[tuple[str, str, set[str]]]) -> bool:
    """True when some cell holds the value as whole words."""
    wanted = f" {fold(value)} "
    return any(wanted in f" {cell} " for _t, _h, cells in columns for cell in cells)


def attribute_column_p(
    attribute: str, columns: list[tuple[str, str, set[str]]], client: Any
) -> float:
    """p(yes) that some column records the customers' `attribute`."""
    lines = []
    for table, header, cells in columns:
        sample = ", ".join(cell[:30] for cell in sorted(cells)[:_SAMPLES])
        lines.append(f"- {table} | {header} | {sample}")
    content = "\n".join(
        ["Columns (table | header | sample values):", *lines, f"Information: customers' {attribute}"]
    )
    messages = [
        {"role": "system", "content": _HAS_SYSTEM},
        {"role": "user", "content": content},
    ]
    probs = client.chat_choice_probs(messages, ["yes", "no"])
    return float(probs.get("yes", 0.0))


def _json(value: dict[str, str | None]) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)

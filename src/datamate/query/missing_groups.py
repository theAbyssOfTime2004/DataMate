"""A question about a product group in a shop that records no product groups.

The rulebook counts such a question as unanswerable ("shop không ghi loại thông tin câu
hỏi cần"). It is decided only after the pipeline abstained: the question must name a
product group, the shop must have no group after mapping, and no column of the shop's
raw files that holds a few kinds may contain that name (a group sheet the mapping missed
means the data exists).
"""

from __future__ import annotations

from typing import Any

from datamate.config import settings
from datamate.llm.client import LLMError, LLMUnavailable
from datamate.query.grounding import QuestionText, fold

_SYSTEM = "\n".join(  # noqa: FLY002
    [
        "A Vietnamese shop owner asks a question. Find the product group it is about: a",
        "group or kind of goods (nhóm hàng, loại hàng, ngành hàng) such as 'nhóm bánh',",
        "'đồ uống', 'hàng điện', as opposed to one named product.",
        "Return the group name copied from the question, or null when the question is about",
        "one product, all goods, a group of customers or staff, or a range of days or ages.",
        'Return JSON {"product_group": <string or null>}.',
    ]
)

# Invented questions (no train/public text).
MISSING_GROUP_FEW_SHOTS: tuple[tuple[str, str | None], ...] = (
    ("Quý 2 nhóm bánh ngọt của tiệm Hạnh bán được bao nhiêu tiền?", "bánh ngọt"),
    ("Tuần này khách quen của tiệm Hạnh mua mấy đơn?", None),
    ("Đồ uống đóng chai chiếm bao nhiêu phần trăm doanh thu tháng 5?", "đồ uống đóng chai"),
    ("Hôm nay còn bao nhiêu tiền nợ ở nhóm tuổi nợ trên 30 ngày?", None),
    ("Tháng trước bán được mấy cái Bánh bông lan trứng muối?", None),
)
_SCHEMA = {
    "type": "object",
    "properties": {"product_group": {"type": ["string", "null"]}},
    "required": ["product_group"],
    "additionalProperties": False,
}


def asked_product_group(
    question: str, client: Any, product_names: list[str] | None = None
) -> str | None:
    """The product group the question names as the rulebook does ("nhóm X"), or None.

    A name that holds a whole catalog product is that product, not a group.
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": _SYSTEM}]
    for text, group in MISSING_GROUP_FEW_SHOTS:
        messages.append({"role": "user", "content": f"Question: {text}"})
        messages.append({"role": "assistant", "content": f'{{"product_group": {_json(group)}}}'})
    messages.append({"role": "user", "content": f"Question: {question}"})
    try:
        raw = client.chat_json(messages, _SCHEMA, "ProductGroup", max_tokens=40)
    except LLMUnavailable:
        raise
    except LLMError:
        return None
    group = raw.get("product_group") if isinstance(raw, dict) else None
    if not isinstance(group, str) or not fold(group):
        return None
    if not QuestionText(question).mentions_value(group):
        return None
    folded = fold(group)
    if f" nhom {folded} " not in f" {fold(question)} ":
        return None
    for name in product_names or []:
        product = fold(name)
        if product and f" {product} " in f" {folded} ":
            return None
    return group


def group_is_unrecorded(group: str, categories: list[str], kind_values: set[str]) -> bool:
    """True when the shop has no product groups and no raw kind column holds the name."""
    if categories:
        return False
    wanted = f" {fold(group)} "
    return not any(wanted in f" {value} " for value in kind_values)


def kind_values(tables: list[Any], limit: int | None = None) -> set[str]:
    """Folded values of raw columns that hold a few kinds (at most `limit` distinct)."""
    most = settings.missing_group_kinds if limit is None else limit
    out: set[str] = set()
    for table in tables:
        frame = table.df
        for column in frame.columns:
            values = {str(v).strip() for v in frame[column].dropna() if str(v).strip()}
            if 2 <= len(values) <= most and any(any(ch.isalpha() for ch in v) for v in values):
                out |= {fold(v) for v in values}
    return out


def _json(value: str | None) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)

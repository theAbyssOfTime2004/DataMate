"""Rewrite a slangy question in plain Vietnamese, for a second parse after an abstain.

Shop owners write teencode and short forms ("đh", "o đờ", "bnhiu", "kủa mik", "z").
The parsers read most of it; when both votes still disagree or fail, the question is
written once more in plain words and parsed again. The rewrite must keep every number
and every product name of the question, else it is dropped.
"""

from __future__ import annotations

import re
from typing import Any

from datamate.llm.client import LLMError, LLMUnavailable
from datamate.query.grounding import fold

_SYSTEM = "\n".join(  # noqa: FLY002
    [
        "A Vietnamese shop owner typed a question with slang, teencode, short forms or",
        "no diacritics. Rewrite it as one plain, standard Vietnamese question with",
        "diacritics that means exactly the same thing.",
        "Keep every number, date, name, product, condition and unit as written; do not",
        "add, drop or explain anything, and do not answer the question.",
        'Return JSON {"question": <the rewritten question>}.',
    ]
)

# Invented questions (no train/public text).
PLAIN_FEW_SHOTS: tuple[tuple[str, str], ...] = (
    (
        "tuan nay tiem mik chot dc bnhiu o đờ online z ta",
        "Tuần này tiệm của mình chốt được bao nhiêu đơn hàng online?",
    ),
    (
        "Hum qa mấy khách quen ck cho kủa hàng mik nhiu tền hỉ",
        "Hôm qua khách quen chuyển khoản cho cửa hàng của mình tổng cộng bao nhiêu tiền?",
    ),
    (
        "từ 1/8 tới 15/8/2025 mấy đh ship COD bị boom zậy",
        "Từ ngày 1/8 đến ngày 15/8/2025 có bao nhiêu đơn hàng giao COD bị bom?",
    ),
    (
        "Bánh Quy Bơ Mơ hộp 300g quý 1 bán đc mí hộp zậy kưng",
        "Quý 1 bán được bao nhiêu hộp Bánh Quy Bơ Mơ hộp 300g?",
    ),
)
_SCHEMA = {
    "type": "object",
    "properties": {"question": {"type": "string"}},
    "required": ["question"],
    "additionalProperties": False,
}
_NUMBER = re.compile(r"\d+")


def plain_question(
    question: str, client: Any, product_names: list[str] | None = None
) -> str | None:
    """The question in plain Vietnamese, or None when the rewrite changes it."""
    messages: list[dict[str, str]] = [{"role": "system", "content": _SYSTEM}]
    for raw, plain in PLAIN_FEW_SHOTS:
        messages.append({"role": "user", "content": f"Question: {raw}"})
        messages.append({"role": "assistant", "content": _json({"question": plain})})
    messages.append({"role": "user", "content": f"Question: {question}"})
    try:
        out = client.chat_json(messages, _SCHEMA, "PlainQuestion", max_tokens=200)
    except LLMUnavailable:
        raise
    except LLMError:
        return None
    text = out.get("question") if isinstance(out, dict) else None
    if not isinstance(text, str) or not text.strip():
        return None
    text = " ".join(text.split())
    return text if keeps_facts(question, text, product_names or []) else None


def keeps_facts(original: str, rewritten: str, product_names: list[str]) -> bool:
    """The same numbers (in any order), and every product the question names kept."""
    if fold(rewritten) == fold(original):
        return False
    if sorted(_NUMBER.findall(original)) != sorted(_NUMBER.findall(rewritten)):
        return False
    before = f" {fold(original)} "
    after = f" {fold(rewritten)} "
    for name in product_names:
        folded = fold(name)
        if folded and f" {folded} " in before and f" {folded} " not in after:
            return False
    return True


def _json(value: dict[str, str]) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)

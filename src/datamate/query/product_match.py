"""Yes/no check: is a catalog product the item the question names under another word?

Shop owners and catalogs often name one item with regional synonyms (lợn / heo, rán /
chiên, lạc / đậu phộng). The parser may pick the catalog name for such a question, but
word grounding rejects it because the words differ. When a picked product shares some
words with the question, this check asks the model whether the rest are synonyms of the
same item; other catalog products that fit the question equally well are shown so that a
question about a whole family ("iPhone 13") is not narrowed to one variant.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from datamate.config import settings
from datamate.query.grounding import QuestionText

_CHOICES = ["yes", "no"]
_MAX_SIBLINGS = 6

_SYSTEM = "\n".join(  # noqa: FLY002
    [
        "A shop owner asks a question in Vietnamese (typos, missing accents and chat slang",
        "are common). You get one product name from the shop's catalog and the other",
        "catalog products whose names are just as close to the question.",
        "Answer yes only when the question names exactly this product, with some words",
        "replaced by a regional or everyday synonym for the same thing (Northern and",
        "Southern words, a common name instead of the label's name).",
        "Answer no when the words differ in meaning: another flavour, size, colour, model,",
        "brand or ingredient; when the question names a group or family that also covers",
        "one of the other listed products; or when you are unsure.",
        "Reply with one word: yes or no.",
    ]
)


@dataclass(frozen=True)
class ProductMatchExample:
    question: str
    product: str
    others: tuple[str, ...]
    answer: str  # "yes" | "no"


# Invented shops and wording (no train/public text); yes and no alternate.
PRODUCT_MATCH_FEW_SHOTS: tuple[ProductMatchExample, ...] = (
    ProductMatchExample(
        "Tuần này quán bán được mấy ly sinh tố dứa?",
        "Sinh tố thơm",
        ("Sinh tố bơ", "Sinh tố dâu"),
        "yes",
    ),
    ProductMatchExample(
        "Tuần này quán bán được mấy ly sinh tố dứa?",
        "Sinh tố dưa hấu",
        ("Sinh tố bơ", "Sinh tố dâu"),
        "no",
    ),
    ProductMatchExample(
        "Ngô luộc nhà mình hồi tháng tư thu về mấy đồng vậy ta?",
        "Bắp luộc",
        ("Khoai lang luộc", "Bắp nướng mỡ hành"),
        "yes",
    ),
    ProductMatchExample(
        "Tháng 4 bán được mấy cái Samsung A55?",
        "Samsung Galaxy A55 256GB Xanh",
        ("Samsung Galaxy A55 128GB Đen", "Samsung Galaxy A55 256GB Tím"),
        "no",
    ),
)


def product_match_probability(question: str, product: str, others: list[str], client: Any) -> float:
    """Return p_yes that `product` is the item the question names."""
    messages: list[dict[str, str]] = [{"role": "system", "content": _SYSTEM}]
    for example in PRODUCT_MATCH_FEW_SHOTS:
        content = _user_content(example.question, example.product, list(example.others))
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": example.answer})
    messages.append({"role": "user", "content": _user_content(question, product, others)})
    probs = client.chat_choice_probs(messages, list(_CHOICES))
    return float(probs.get("yes", 0.0))


def product_matcher(
    question: str, product_names: list[str], client: Any
) -> Callable[[str], bool] | None:
    """A cached same-product check for one question, or None when it is switched off."""
    if not settings.product_synonym_check or client is None:
        return None
    text = QuestionText(question)
    decided: dict[str, bool] = {}

    def same_product(name: str) -> bool:
        if name not in decided:
            hits, _total = text.product_word_hits(name)
            others = [
                other
                for other in product_names
                if other != name and text.product_word_hits(other)[0] >= hits
            ][:_MAX_SIBLINGS]
            p_yes = product_match_probability(question, name, others, client)
            decided[name] = p_yes >= settings.product_synonym_threshold
        return decided[name]

    return same_product


def _user_content(question: str, product: str, others: list[str]) -> str:
    listed = "; ".join(others) if others else "(none)"
    return (
        f"Question: {question}\n"
        f"Product: {product}\n"
        f"Other products as close to the question: {listed}\n"
        "Does the question name exactly this product? Answer yes or no."
    )

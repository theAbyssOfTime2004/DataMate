"""Yes/no check: does the question ask for an opinion that shop records cannot hold?

Questions about taste, quality, attitude, skill or personality ("món nào ngon nhất",
"nhân viên nào thân thiện nhất") and requests for advice have no answer in sales data, so
they are unanswerable. A question about anything the records measure (sales, money,
counts, stock, debt) is not, even when it says "nhất" or "tốt".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_CHOICES = ["yes", "no"]

_SYSTEM = "\n".join(  # noqa: FLY002
    [
        "A shop keeps records of orders, sold items, prices, purchases, stock, debts,",
        "customers and the employee who made each sale. A shop owner asks a question in",
        "Vietnamese (typos, missing accents and chat slang are common).",
        "Answer yes only when the question asks for a personal judgement those records can",
        "never contain: taste or quality of a product, attitude, friendliness, politeness,",
        "skill or character of a person, how much someone is liked, or advice on using a",
        "product (health, safety, who may use it).",
        "Answer no when the question can be answered by counting or adding up records,",
        "even if it says best, most or top (best-selling, most revenue, most orders, most",
        "debt, highest price), and no when you are unsure.",
        "Reply with one word: yes or no.",
    ]
)


@dataclass(frozen=True)
class SubjectiveExample:
    question: str
    answer: str  # "yes" | "no"


# Invented shops and wording (no train/public text); yes and no alternate.
SUBJECTIVE_FEW_SHOTS: tuple[SubjectiveExample, ...] = (
    SubjectiveExample("Loại bánh kem nào của tiệm Hoa Sen được khen thơm nhất vậy?", "yes"),
    SubjectiveExample("Bánh kem dâu tháng rồi bán ra được mấy cái?", "no"),
    SubjectiveExample("Cậu thu ngân nào ăn nói nhẹ nhàng với khách nhất?", "yes"),
    SubjectiveExample("Quý 1 thu ngân nào bán ra nhiều tiền nhất?", "no"),
    SubjectiveExample("Trẻ 2 tuổi uống siro ho Bảo Thanh có an toàn không?", "yes"),
    SubjectiveExample("Siro ho Bảo Thanh còn tồn bao nhiêu chai?", "no"),
    SubjectiveExample("Anh thợ điện nào làm việc cẩn thận nhất tiệm mình?", "yes"),
    SubjectiveExample("Khách nào mua chịu nhiều nhất từ đầu năm tới giờ?", "no"),
)


def subjective_probability(question: str, client: Any) -> float:
    """Return p_yes that the question asks for a judgement the records cannot hold."""
    messages: list[dict[str, str]] = [{"role": "system", "content": _SYSTEM}]
    for example in SUBJECTIVE_FEW_SHOTS:
        messages.append({"role": "user", "content": _user_content(example.question)})
        messages.append({"role": "assistant", "content": example.answer})
    messages.append({"role": "user", "content": _user_content(question)})
    probs = client.chat_choice_probs(messages, list(_CHOICES))
    return float(probs.get("yes", 0.0))


def _user_content(question: str) -> str:
    return f"Question: {question}\nDoes it ask for a judgement the records cannot hold?"

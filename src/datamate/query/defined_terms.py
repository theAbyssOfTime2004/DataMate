"""Find the rulebook term a question asks about ("Khách trung thành nhất", "Giờ cao điểm").

The rulebook given with each test set defines words that sound like opinions but are
counts. A cheap word-pair overlap picks a few rule rows whose name is close to the
question; a yes/no check then decides whether the question really asks about one of them.
The rows come from the rulebook at run time; nothing here names a term.
"""

from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass
from typing import Any

from datamate.config import settings
from datamate.query.grounding import fold
from datamate.rules.parse_rulebook import RuleRow

_CHOICES = ["yes", "no"]
_MAX_CANDIDATES = 3
# A word pair shared by this many rule names says little about any one of them.
_COMMON_PAIR_NAMES = 3

_SYSTEM = "\n".join(  # noqa: FLY002
    [
        "A shop owner asks a question in Vietnamese (typos, missing accents and chat slang",
        "are common). The shop's rulebook defines a term. Answer yes only when the question",
        "asks about that term itself: the same words or a light rewording of them, so that",
        "the rulebook definition decides what to compute.",
        "Answer no when the question only shares some words with the term but asks for",
        "something else, or when you are unsure.",
        "Reply with one word: yes or no.",
    ]
)


@dataclass(frozen=True)
class TermExample:
    question: str
    term: str
    definition: str
    answer: str  # "yes" | "no"


# Invented terms and wording (no train/public text); yes and no alternate.
TERM_FEW_SHOTS: tuple[TermExample, ...] = (
    TermExample(
        "Tuần rồi ai là mối quen của tiệm vậy?",
        "Mối quen",
        "người mua đứng đầu về số hoá đơn đã chốt trong khoảng hỏi",
        "yes",
    ),
    TermExample(
        "Tuần rồi mối quen chi hết mấy đồng?",
        "Mối quen",
        "người mua đứng đầu về số hoá đơn đã chốt trong khoảng hỏi",
        "no",
    ),
    TermExample(
        "thang 5 mon nao la mon quoc dan cua quan",
        "Món quốc dân",
        "món góp mặt ở số ngày bán lớn nhất trong khoảng hỏi",
        "yes",
    ),
    TermExample(
        "Tháng 5 món nào ra nhiều ly nhất?",
        "Món quốc dân",
        "món góp mặt ở số ngày bán lớn nhất trong khoảng hỏi",
        "no",
    ),
)


_RANKING_SYSTEM = "\n".join(  # noqa: FLY002
    [
        "You read one row of a Vietnamese shop rulebook: a term and its definition.",
        "Answer yes only when the term itself names a winner: the one customer, staff",
        "member, product, day, hour or supplier with the most (or least) of something,",
        "so that a question using the term asks for a name, a day or an hour.",
        "Answer no when the term names how to count, add up, identify, group or filter",
        "something (a total, a shift, a unit, a lot, a kind of customer), even if its",
        "definition mentions a ranking in passing.",
        "Reply with one word: yes or no.",
    ]
)


@dataclass(frozen=True)
class RankingRowExample:
    term: str
    definition: str
    answer: str  # "yes" | "no"


# Invented rows (no rulebook text); yes and no alternate.
RANKING_ROW_FEW_SHOTS: tuple[RankingRowExample, ...] = (
    RankingRowExample(
        "Mối quen", "người mua đứng đầu về số hoá đơn đã chốt trong khoảng hỏi", "yes"
    ),
    RankingRowExample("Ca tối", "theo giờ tạo đơn, từ 18:00 đến 22:59", "no"),
    RankingRowExample("Ngày vàng", "ngày bán có doanh thu lớn nhất trong khoảng hỏi", "yes"),
    RankingRowExample(
        "Bán được bao nhiêu ký",
        "cộng số ký của các món bán theo ký, món bán theo cái không tính",
        "no",
    ),
    RankingRowExample(
        "Bạn hàng lớn", "đối tác có tổng tiền các phiếu nhập lớn nhất trong năm", "yes"
    ),
    RankingRowExample(
        "Xác định nhân viên",
        "mỗi mã nhân viên là một người; nhân viên bán nhiều nhất xếp theo doanh thu",
        "no",
    ),
)


def ranking_rows(rows: list[RuleRow], client: Any) -> list[RuleRow]:
    """Rule rows that define a ranking term; asked once per row (the client caches)."""
    if not settings.defined_term_check or client is None:
        return []
    kept: list[RuleRow] = []
    for row in rows:
        if not _name_pairs(row.name):
            continue
        messages: list[dict[str, str]] = [{"role": "system", "content": _RANKING_SYSTEM}]
        for example in RANKING_ROW_FEW_SHOTS:
            content = _row_content(example.term, example.definition)
            messages.append({"role": "user", "content": content})
            messages.append({"role": "assistant", "content": example.answer})
        messages.append({"role": "user", "content": _row_content(row.name, row.text)})
        probs = client.chat_choice_probs(messages, list(_CHOICES))
        if float(probs.get("yes", 0.0)) >= 0.5:
            kept.append(row)
    return kept


_UNIT_SYSTEM = "\n".join(  # noqa: FLY002
    [
        "You read one row of a Vietnamese shop rulebook: a term and its definition.",
        "Answer yes only when the row says what one record of the shop's book is called",
        "and how to count those records (a visit, a ticket, a job card, a trip), so that a",
        "question can ask how many of them there were.",
        "Answer no for rows about money, dates, statuses, units of goods, rankings, file",
        "layout or answer format.",
        "Reply with one word: yes or no.",
    ]
)

# Invented rows (no rulebook text); yes and no alternate.
UNIT_ROW_FEW_SHOTS: tuple[RankingRowExample, ...] = (
    RankingRowExample(
        "Ca, suất", "mỗi dòng sổ là một suất khách ngồi, gồm cả suất được mời", "yes"
    ),
    RankingRowExample("Tiền cọc", "khoản khách đặt trước, không tính vào doanh thu", "no"),
    RankingRowExample(
        "Chuyến, cuốc", "mỗi dòng nhật ký là một cuốc chở, kể cả cuốc chở lại", "yes"
    ),
    RankingRowExample("Món bán chạy", "món có số phần bán ra cao nhất trong kỳ", "no"),
)


def unit_rows(rows: list[RuleRow], client: Any) -> list[RuleRow]:
    """Rule rows that name the shop's counted record (lượt, phiếu); asked once per row."""
    if not settings.unit_terms or client is None:
        return []
    kept: list[RuleRow] = []
    for row in rows:
        messages: list[dict[str, str]] = [{"role": "system", "content": _UNIT_SYSTEM}]
        for example in UNIT_ROW_FEW_SHOTS:
            content = _row_content(example.term, example.definition)
            messages.append({"role": "user", "content": content})
            messages.append({"role": "assistant", "content": example.answer})
        content = _row_content(row.name, row.text).replace("define a ranking", "name a record")
        messages.append({"role": "user", "content": content})
        probs = client.chat_choice_probs(messages, list(_CHOICES))
        if float(probs.get("yes", 0.0)) >= 0.5:
            kept.append(row)
    return kept


def match_unit_term(
    question: str, rows: list[RuleRow], client: Any
) -> tuple[RuleRow | None, list[tuple[str, float]]]:
    """The record-naming row the question counts ("bao nhiêu lượt xe"), or None."""
    asked = fold(question)
    padded = f" {asked} "
    candidates = [
        row
        for row in rows
        if any(
            f" {name} " in padded for name in (fold(part) for part in row.name.split(",")) if name
        )
    ][:_MAX_CANDIDATES]
    checked: list[tuple[str, float]] = []
    best: tuple[float, RuleRow] | None = None
    for row in candidates:
        p_yes = term_probability(question, row, client)
        checked.append((row.name, round(p_yes, 4)))
        if p_yes >= settings.defined_term_threshold and (best is None or p_yes > best[0]):
            best = (p_yes, row)
    return (best[1] if best is not None else None), checked


def term_candidates(question: str, rows: list[RuleRow]) -> list[RuleRow]:
    """Rule rows whose name shares rare word pairs with the question, closest first."""
    asked = _pairs(fold(question))
    if not asked:
        return []
    named = [(row, _name_pairs(row.name)) for row in rows]
    spread = Counter(pair for _row, pairs in named for pair in pairs)
    scored: list[tuple[float, int, RuleRow]] = []
    for index, (row, pairs) in enumerate(named):
        if not pairs:
            continue
        shared = pairs & asked
        rare = {pair for pair in shared if spread[pair] < _COMMON_PAIR_NAMES}
        if not rare or 2 * len(shared) < len(pairs):
            continue
        scored.append((len(shared) / len(pairs), index, row))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [row for _score, _index, row in scored[:_MAX_CANDIDATES]]


def term_probability(question: str, row: RuleRow, client: Any) -> float:
    """Return p_yes that the question asks about the term `row` defines."""
    messages: list[dict[str, str]] = [{"role": "system", "content": _SYSTEM}]
    for example in TERM_FEW_SHOTS:
        content = _user_content(example.question, example.term, example.definition)
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": example.answer})
    messages.append({"role": "user", "content": _user_content(question, row.name, row.text)})
    probs = client.chat_choice_probs(messages, list(_CHOICES))
    return float(probs.get("yes", 0.0))


def match_defined_term(
    question: str, rows: list[RuleRow], client: Any
) -> tuple[RuleRow | None, list[tuple[str, float]]]:
    """The rule row the question asks about, or None; plus every (name, p_yes) asked."""
    if not settings.defined_term_check or client is None:
        return None, []
    asked: list[tuple[str, float]] = []
    best: tuple[float, RuleRow] | None = None
    for row in term_candidates(question, rows):
        p_yes = term_probability(question, row, client)
        asked.append((row.name, round(p_yes, 4)))
        if p_yes >= settings.defined_term_threshold and (best is None or p_yes > best[0]):
            best = (p_yes, row)
    return (best[1] if best is not None else None), asked


def _name_pairs(name: str) -> set[str]:
    # "Nhân viên, thợ chăm chỉ nhất" names two phrasings; pairs never cross the comma.
    pairs: set[str] = set()
    for part in name.split(","):
        pairs |= _pairs(fold(part))
    return pairs


def _pairs(folded: str) -> set[str]:
    words = folded.split()
    return {f"{a} {b}" for a, b in itertools.pairwise(words)}


def _row_content(term: str, definition: str) -> str:
    return (
        f"Term: {term}\nDefinition: {definition}\nDoes this row define a ranking? Answer yes or no."
    )


def _user_content(question: str, term: str, definition: str) -> str:
    return (
        f"Question: {question}\n"
        f"Term: {term}\n"
        f"Rulebook definition: {definition}\n"
        "Does the question ask about this term? Answer yes or no."
    )

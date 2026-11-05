"""Separate parser for questions over per-order event logs (delivery batches).

It runs only for shops that keep such a log, before the extension parser, so the
extension prompt and every other shop stay exactly as they were. "none" hands the
question on to the extension parser.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass
from typing import Any

from datamate.config import settings
from datamate.llm.client import LLMError, LLMUnavailable
from datamate.query.convert import ConvertError
from datamate.query.ext.convert import to_ext_query_spec
from datamate.query.ext.parse import ExtParseResult, ext_semantic_key
from datamate.query.ext.schema import _period_object
from datamate.query.parse import ParseAttempt
from datamate.query.prompts import PERIOD_KIND_LINES
from datamate.rules.model import RuleConfig
from datamate.timeres.calendar import LunarCalendar

_MAX_TOKENS = 200
_KEYS = ("family", "n", "period")

_SYSTEM = "\n".join(
    [
        "A Vietnamese shop owner asks a question. The shop keeps a delivery log: each order",
        "can be delivered in several batches (đợt / lượt / lần giao, chuyến chở).",
        "You do not calculate the answer and you do not calculate calendar dates.",
        "",
        "Families:",
        "- delivery_count: how many delivery batches were made in the period (by delivery",
        "  date). Set period.",
        "- orders_min_deliveries: how many orders placed in the period were delivered in",
        "  at least n batches up to today. Set n and period.",
        "- none: anything else, including money, quantities delivered, stock, debt or a",
        "  count of orders that does not depend on batches.",
        "",
        *PERIOD_KIND_LINES,
        "The worked examples use an invented hardware shop, Vật tư Phú Lộc.",
    ]
)


@dataclass(frozen=True)
class EventFewShot:
    question: str
    as_of: datetime.date
    parsed: dict[str, Any]


def _period(kind: str, **fields: Any) -> dict[str, Any]:
    from datamate.query.parse_schema import PERIOD_FIELDS

    row: dict[str, Any] = {name: None for name in PERIOD_FIELDS}
    row["kind"] = kind
    row.update(fields)
    return row


_ASOF = datetime.date(2024, 8, 15)

# Two sets for the two votes; shop and wording are invented.
EVENT_FEW_SHOT_SETS: tuple[tuple[EventFewShot, ...], ...] = (
    (
        EventFewShot(
            "Phu Loc thang 6/2024 cho hang di bao nhieu chuyen",
            _ASOF,
            {"family": "delivery_count", "n": None, "period": _period("month", month=6, year=2024)},
        ),
        EventFewShot(
            "Phú Lộc tháng vừa qua thu về tổng cộng mấy đồng?",
            _ASOF,
            {"family": "none", "n": None, "period": None},
        ),
        EventFewShot(
            "Don dat trong quy 2/2024, toi gio co may don Phu Loc phai giao tu 3 lan tro len",
            _ASOF,
            {
                "family": "orders_min_deliveries",
                "n": 3,
                "period": _period("quarter", q=2, year=2024),
            },
        ),
        EventFewShot(
            "Tuần này Phú Lộc đã giao bao nhiêu bao xi măng cho khách?",
            _ASOF,
            {"family": "none", "n": None, "period": None},
        ),
    ),
    (
        EventFewShot(
            "Tuần rồi xe của Phú Lộc chạy giao mấy lượt?",
            _ASOF,
            {"family": "delivery_count", "n": None, "period": _period("week_last")},
        ),
        EventFewShot(
            "so hoa don Phu Loc chot duoc trong thang bay nam 2024 la may",
            _ASOF,
            {"family": "none", "n": None, "period": None},
        ),
        EventFewShot(
            "Phú Lộc: khách đặt hồi tháng 5/2024, có mấy đơn phải chia ra chở ít nhất 2 chuyến?",
            _ASOF,
            {
                "family": "orders_min_deliveries",
                "n": 2,
                "period": _period("month", month=5, year=2024),
            },
        ),
        EventFewShot(
            "Hom nay kho Phu Loc con bao nhieu cay thep phi 10",
            _ASOF,
            {"family": "none", "n": None, "period": None},
        ),
    ),
)


def _schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "family": {
                "type": "string",
                "enum": ["delivery_count", "orders_min_deliveries", "none"],
            },
            "n": {"type": ["integer", "null"]},
            "period": _period_object(),
        },
        "required": list(_KEYS),
        "additionalProperties": False,
    }


def _user(question: str, as_of: datetime.date) -> str:
    return f"Hôm nay (as_of): {as_of.isoformat()}\nQuestion: {question}"


def build_event_messages(
    question: str, as_of: datetime.date, few_shot_set: int
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": _SYSTEM}]
    for example in EVENT_FEW_SHOT_SETS[few_shot_set % len(EVENT_FEW_SHOT_SETS)]:
        messages.append({"role": "user", "content": _user(example.question, example.as_of)})
        messages.append(
            {"role": "assistant", "content": json.dumps(example.parsed, ensure_ascii=False)}
        )
    messages.append({"role": "user", "content": _user(question, as_of)})
    return messages


def parse_event_question(
    question: str,
    client: Any,
    rules: RuleConfig,
    cal: LunarCalendar,
    as_of: datetime.date,
) -> ExtParseResult:
    """Two votes; both must give the same event spec, else no spec (the caller moves on)."""
    attempts: list[ParseAttempt] = []
    specs = []
    for few_shot_set in range(max(settings.parse_votes, 1)):
        messages = build_event_messages(question, as_of, few_shot_set)
        try:
            raw = client.chat_json(
                messages,
                _schema(),
                "EventQuestion",
                max_tokens=_MAX_TOKENS,
                required_keys=list(_KEYS),
            )
        except LLMUnavailable:
            raise
        except LLMError as exc:
            attempts.append(ParseAttempt(few_shot_set, None, str(exc), 0))
            return ExtParseResult(spec=None, reason="parse_error", attempts=attempts)
        try:
            spec = to_ext_query_spec(raw, as_of, question)
        except ConvertError as exc:
            attempts.append(ParseAttempt(few_shot_set, raw, str(exc), 0))
            return ExtParseResult(spec=None, reason="parse_error", attempts=attempts)
        attempts.append(ParseAttempt(few_shot_set, raw, None, 0))
        if spec.family == "none":
            return ExtParseResult(spec=None, reason="out_of_scope", attempts=attempts)
        specs.append(spec)
    keys = {ext_semantic_key(spec, as_of, rules, cal) for spec in specs}
    if len(keys) != 1:
        return ExtParseResult(spec=None, reason="disagreement", attempts=attempts)
    return ExtParseResult(spec=specs[0], reason=None, attempts=attempts)

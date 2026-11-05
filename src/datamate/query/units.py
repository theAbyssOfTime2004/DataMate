"""Unit block: how many of the shop's records ("lượt xe", "phiếu") or how much money, for
the whole shop, one named staff member, or the top staff member.

Used only after the pipeline abstained, for a question that counts the record unit a
rulebook row names; the row's definition is shown to the parser and the verifier.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from datamate.config import settings
from datamate.llm.client import LLMError, LLMUnavailable
from datamate.query.convert import ConvertError, period_from_raw
from datamate.query.ext.schema import _period_object
from datamate.query.grounding import QuestionText, fold
from datamate.query.parse import ParseAttempt
from datamate.query.parse_schema import PERIOD_FIELDS
from datamate.query.prompts import PERIOD_KIND_LINES
from datamate.rules.model import RuleConfig
from datamate.rules.parse_rulebook import RuleRow
from datamate.timeres.calendar import CalendarError, LunarCalendar
from datamate.timeres.periods import PeriodError, PeriodExpr, resolve

UnitPick = Literal["shop", "staff", "top_staff"]
UnitMeasure = Literal["count", "money"]
_KEYS = ("family", "pick", "staff", "measure", "period")
_MAX_TOKENS = 200


class UnitQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: Literal["unit"] = "unit"
    pick: UnitPick
    staff: str | None = None  # the name as written in the question, for pick="staff"
    measure: UnitMeasure
    period: PeriodExpr


@dataclass
class UnitParseResult:
    spec: UnitQuery | None
    reason: str | None
    attempts: list[ParseAttempt]


_SYSTEM = "\n".join(
    [
        "A Vietnamese shop owner asks about the shop's records. The rulebook names one",
        "record of this shop's book and says how to count them (shown with the question).",
        "Fill the fields; do not calculate the answer or calendar dates.",
        "",
        "- pick: shop (the whole shop) | staff (one staff member named in the question) |",
        "  top_staff (the staff member with the most).",
        "- staff: the staff name copied from the question when pick is staff, else null.",
        "- measure: count (how many records) | money (the money on those records).",
        "- period: from the question.",
        'Set family to "none" when the question asks anything else (a share, an average,',
        "a kind of job, a vehicle, a customer, a list or a table).",
        "",
        *PERIOD_KIND_LINES,
        "The worked examples use an invented laundry, Giặt ủi Mai, whose book counts 'mẻ'.",
    ]
)

_TERM = "Mẻ giặt"
_DEFINITION = "mỗi dòng sổ là một mẻ giặt, kể cả mẻ giặt lại không thu tiền"
_ASOF = datetime.date(2024, 8, 15)


def _period(kind: str, **fields: Any) -> dict[str, Any]:
    row: dict[str, Any] = {name: None for name in PERIOD_FIELDS}
    row["kind"] = kind
    row.update(fields)
    return row


def _parsed(**fields: Any) -> dict[str, Any]:
    row: dict[str, Any] = {key: None for key in _KEYS}
    row["family"] = "unit"
    row.update(fields)
    return row


# Two sets for the two votes; the laundry and wording are invented.
UNIT_FEW_SHOT_SETS: tuple[tuple[tuple[str, dict[str, Any]], ...], ...] = (
    (
        (
            "Giat ui Mai thang 6/2024 chay duoc bao nhieu me",
            _parsed(pick="shop", measure="count", period=_period("month", month=6, year=2024)),
        ),
        (
            "Tuần rồi chị Thu Hằng đứng máy được mấy mẻ?",
            _parsed(pick="staff", staff="Thu Hằng", measure="count", period=_period("week_last")),
        ),
        (
            "Bình quân một mẻ của tiệm Mai hồi tháng bảy 2024 được mấy đồng?",
            _parsed(family="none"),
        ),
    ),
    (
        (
            "Quý 2/2024 ai ở tiệm giặt chạy nhiều mẻ nhất",
            _parsed(pick="top_staff", measure="count", period=_period("quarter", q=2, year=2024)),
        ),
        (
            "Thang nay cac me cua anh Bao Long thu ve tong cong may dong",
            _parsed(pick="staff", staff="Bao Long", measure="money", period=_period("month_this")),
        ),
        (
            "Tháng 5/2024 có bao nhiêu mẻ giặt chăn mền?",
            _parsed(family="none"),
        ),
    ),
)


def _schema() -> dict[str, Any]:
    def nullable(values: list[str]) -> dict[str, Any]:
        return {"type": ["string", "null"], "enum": [*values, None]}

    return {
        "type": "object",
        "properties": {
            "family": {"type": "string", "enum": ["unit", "none"]},
            "pick": nullable(["shop", "staff", "top_staff"]),
            "staff": {"type": ["string", "null"]},
            "measure": nullable(["count", "money"]),
            "period": _period_object(),
        },
        "required": list(_KEYS),
        "additionalProperties": False,
    }


def _user(question: str, term: str, definition: str, as_of: datetime.date) -> str:
    return (
        f"Hôm nay (as_of): {as_of.isoformat()}\nRecord: {term} = {definition}\nQuestion: {question}"
    )


def build_unit_messages(
    question: str, term: RuleRow, as_of: datetime.date, few_shot_set: int
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": _SYSTEM}]
    for text, parsed in UNIT_FEW_SHOT_SETS[few_shot_set % len(UNIT_FEW_SHOT_SETS)]:
        messages.append({"role": "user", "content": _user(text, _TERM, _DEFINITION, _ASOF)})
        messages.append({"role": "assistant", "content": json.dumps(parsed, ensure_ascii=False)})
    messages.append({"role": "user", "content": _user(question, term.name, term.text, as_of)})
    return messages


def to_unit_query(raw: object, as_of: datetime.date, question: str) -> UnitQuery | None:
    if not isinstance(raw, dict):
        raise ConvertError("response is not an object")
    if raw.get("family") != "unit":
        return None
    pick, measure = raw.get("pick"), raw.get("measure")
    if pick not in {"shop", "staff", "top_staff"} or measure not in {"count", "money"}:
        raise ConvertError(f"bad pick/measure {pick!r}/{measure!r}")
    staff = raw.get("staff")
    if pick == "staff":
        if not isinstance(staff, str) or not staff.strip():
            raise ConvertError("pick staff needs the staff name")
        if not QuestionText(question).mentions_value(staff):
            raise ConvertError(f"staff {staff!r} is not grounded in the question")
    else:
        staff = None
    if raw.get("period") is None:
        raise ConvertError("period is required")
    period = period_from_raw(raw["period"], as_of)
    return UnitQuery(pick=pick, staff=staff, measure=measure, period=period)  # type: ignore[arg-type]


def parse_unit_question(
    question: str,
    term: RuleRow,
    client: Any,
    rules: RuleConfig,
    cal: LunarCalendar,
    as_of: datetime.date,
) -> UnitParseResult:
    """Two votes; they must agree (dates compared resolved)."""
    attempts: list[ParseAttempt] = []
    specs: list[UnitQuery] = []
    for few_shot_set in range(max(settings.parse_votes, 1)):
        messages = build_unit_messages(question, term, as_of, few_shot_set)
        try:
            raw = client.chat_json(
                messages,
                _schema(),
                "UnitQuestion",
                max_tokens=_MAX_TOKENS,
                required_keys=list(_KEYS),
            )
        except LLMUnavailable:
            raise
        except LLMError as exc:
            attempts.append(ParseAttempt(few_shot_set, None, str(exc), 0))
            return UnitParseResult(None, "unit_parse_error", attempts)
        try:
            spec = to_unit_query(raw, as_of, question)
        except ConvertError as exc:
            attempts.append(ParseAttempt(few_shot_set, raw, str(exc), 0))
            return UnitParseResult(None, "unit_parse_error", attempts)
        attempts.append(ParseAttempt(few_shot_set, raw, None, 0))
        if spec is None:
            return UnitParseResult(None, "unit_none", attempts)
        specs.append(spec)
    keys = {_key(spec, as_of, rules, cal) for spec in specs}
    if len(keys) != 1:
        return UnitParseResult(None, "unit_disagreement", attempts)
    return UnitParseResult(specs[0], None, attempts)


def _key(spec: UnitQuery, as_of: datetime.date, rules: RuleConfig, cal: LunarCalendar) -> str:
    row = spec.model_dump(mode="json")
    row["staff"] = fold(spec.staff) if spec.staff else None
    try:
        window = resolve(spec.period, as_of, rules, cal)
        row["period"] = [window.start.isoformat(), window.end.isoformat()]
    except (PeriodError, CalendarError, ValueError):
        row["period"] = {"unresolved": row["period"]}
    return json.dumps(row, sort_keys=True, ensure_ascii=False)


def describe_unit(
    spec: UnitQuery, term: RuleRow, as_of: datetime.date, rules: RuleConfig, cal: LunarCalendar
) -> str:
    try:
        window = resolve(spec.period, as_of, rules, cal)
        span = f"{window.start.isoformat()} .. {window.end.isoformat()}"
    except (PeriodError, CalendarError, ValueError):
        span = f"unresolved {spec.period!r}"
    what = "number of records" if spec.measure == "count" else "money on the records"
    who = {
        "shop": "whole shop",
        "staff": f'staff member "{spec.staff}" only',
        "top_staff": "the staff member with the highest value (name)",
    }[spec.pick]
    return "\n".join(
        [
            f"record: {term.name} (one row of the book = one record, all rows counted)",
            f"measure: {what}",
            f"for: {who}",
            f"period: {span} (record date)",
            "output: one name" if spec.pick == "top_staff" else "output: one number",
        ]
    )

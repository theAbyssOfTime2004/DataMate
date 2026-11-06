"""Attribute block: count, money, distinct values or a share of the shop's records, filtered
by the text of the book's extra columns (a vehicle, the job done, a note).

Used only after the pipeline abstained, for shops whose sales book has such columns. The
parser sees the column names, their README description, sample values and the rulebook
rows whose names the question uses; it only fills these fields.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

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

AttrMeasure = Literal["count", "money", "distinct", "share", "top_value"]
_KEYS = (
    "family",
    "measure",
    "staff",
    "where",
    "distinct_of",
    "share_of_paid",
    "value_of",
    "top_by",
    "weekday",
    "per_weekday",
    "period",
)
_MAX_TOKENS = 300
_MAX_CONDITIONS = 3


class AttrCondition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attribute: str
    contains: str
    negate: bool = False


class AttrQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: Literal["attr"] = "attr"
    measure: AttrMeasure
    staff: str | None = None
    where: list[AttrCondition] = Field(default_factory=list)
    distinct_of: str | None = None
    share_of_paid: bool = False
    value_of: str | None = None  # top_value: the column whose most common value is asked
    top_by: Literal["count", "money"] = "count"
    weekday: int | None = Field(default=None, ge=1, le=7)  # 1 = Monday ... 7 = Sunday
    per_weekday: bool = False  # average per such weekday in the period
    period: PeriodExpr


@dataclass
class AttrParseResult:
    spec: AttrQuery | None
    reason: str | None
    attempts: list[ParseAttempt]


@dataclass(frozen=True)
class ShopAttribute:
    name: str
    description: str | None
    samples: tuple[str, ...]


_SYSTEM = "\n".join(
    [
        "A Vietnamese shop owner asks about the shop's records (one row of the sales book =",
        "one record). Besides date, staff and money, each record has text columns listed",
        "with the question. Fill the fields; do not calculate the answer or dates.",
        "",
        "- measure: count (how many records) | money (money on the records) | distinct",
        "  (how many different values of one column, e.g. different vehicles) | share",
        "  (percent of records that match `where`) | top_value (the value of one column",
        "  found on the most records, or with the most money).",
        "- staff: the staff name copied from the question, or null.",
        "- where: up to 3 conditions, all must hold. attribute = a column name from the",
        "  list; contains = the words the column must contain, in the shop's own spelling",
        "  from the samples; negate = true for 'not containing'. Use the rulebook rows to",
        "  decide what counts (and what does not) as the thing asked.",
        "- distinct_of: the column whose different values are counted (measure distinct).",
        "- share_of_paid: for share, true when the base is only records with money",
        "  (else all records).",
        "- value_of: the column for top_value; top_by: count (most records) | money.",
        "- weekday: only records on that day of the week (1 = thứ Hai ... 6 = thứ Bảy,",
        "  7 = Chủ nhật), else null. per_weekday: true when the question asks the average",
        "  per such day in the period (count or money divided by the number of those days).",
        "- period: from the question.",
        'Set family to "none" when the question needs anything else (a ranking of staff',
        "or customers, a table, an average per record, a customer, stock, or a column that",
        "is not listed).",
        "",
        *PERIOD_KIND_LINES,
        "The worked examples use an invented tailor, Tiệm may Hồng.",
    ]
)

_ASOF = datetime.date(2024, 8, 15)
_EXAMPLE_ATTRIBUTES = (
    ShopAttribute("Đồ", "món đồ khách đặt may", ("áo dài lụa", "quần tây", "đầm suông")),
    ShopAttribute("Ghi chú", "ghi chú", ("sửa lại", "giao gấp")),
)
_EXAMPLE_RULES = (
    RuleRow(
        "x1",
        "Rulebook > Tiệm may",
        "Phiếu sửa lại",
        "phiếu ghi chú sửa lại, tiền 0: khách mang đồ quay lại sửa",
    ),
)


def _period(kind: str, **fields: Any) -> dict[str, Any]:
    row: dict[str, Any] = {name: None for name in PERIOD_FIELDS}
    row["kind"] = kind
    row.update(fields)
    return row


def _parsed(**fields: Any) -> dict[str, Any]:
    row: dict[str, Any] = {key: None for key in _KEYS}
    row["family"] = "attr"
    row["where"] = []
    row["share_of_paid"] = False
    row["top_by"] = "count"
    row["per_weekday"] = False
    row.update(fields)
    return row


def _cond(attribute: str, contains: str, negate: bool = False) -> dict[str, Any]:
    return {"attribute": attribute, "contains": contains, "negate": negate}


# Two sets for the two votes; the tailor and wording are invented.
ATTR_FEW_SHOT_SETS: tuple[tuple[tuple[str, dict[str, Any]], ...], ...] = (
    (
        (
            "Tiem may Hong thang 6/2024 may duoc may bo ao dai",
            _parsed(
                measure="count",
                where=[_cond("Đồ", "áo dài")],
                period=_period("month", month=6, year=2024),
            ),
        ),
        (
            "Quý 2/2024 phiếu sửa lại chiếm mấy phần trăm số phiếu có thu tiền?",
            _parsed(
                measure="share",
                where=[_cond("Ghi chú", "sửa lại")],
                share_of_paid=True,
                period=_period("quarter", q=2, year=2024),
            ),
        ),
        (
            "Thang 6/2024 tiem may Hong may kieu do nao nhieu nhat",
            _parsed(
                measure="top_value",
                value_of="Đồ",
                period=_period("month", month=6, year=2024),
            ),
        ),
        (
            "Khách quen nào của tiệm may Hồng đặt đồ nhiều lần nhất hồi tuần trước?",
            _parsed(family="none", where=None, share_of_paid=None),
        ),
    ),
    (
        (
            "Thang nay chi Thu may quan tay thu ve bao nhieu tien",
            _parsed(
                measure="money",
                staff="Thu",
                where=[_cond("Đồ", "quần tây")],
                period=_period("month_this"),
            ),
        ),
        (
            "Tháng 7/2024 tiệm may bao nhiêu kiểu đồ khác nhau, không tính phiếu sửa lại?",
            _parsed(
                measure="distinct",
                distinct_of="Đồ",
                where=[_cond("Ghi chú", "sửa lại", negate=True)],
                period=_period("month", month=7, year=2024),
            ),
        ),
        (
            "Tháng 7/2024 bình quân mỗi Chủ nhật tiệm may Hồng nhận mấy phiếu?",
            _parsed(
                measure="count",
                weekday=7,
                per_weekday=True,
                period=_period("month", month=7, year=2024),
            ),
        ),
        (
            "Bình quân mỗi phiếu tháng 5/2024 thu mấy đồng?",
            _parsed(family="none", where=None, share_of_paid=None),
        ),
    ),
)


def _schema(attributes: list[str]) -> dict[str, Any]:
    names = {"type": "string", "enum": attributes}
    return {
        "type": "object",
        "properties": {
            "family": {"type": "string", "enum": ["attr", "none"]},
            "measure": {
                "type": ["string", "null"],
                "enum": ["count", "money", "distinct", "share", None],
            },
            "staff": {"type": ["string", "null"]},
            "where": {
                "type": ["array", "null"],
                "maxItems": _MAX_CONDITIONS,
                "items": {
                    "type": "object",
                    "properties": {
                        "attribute": names,
                        "contains": {"type": "string"},
                        "negate": {"type": "boolean"},
                    },
                    "required": ["attribute", "contains", "negate"],
                    "additionalProperties": False,
                },
            },
            "distinct_of": {"type": ["string", "null"], "enum": [*attributes, None]},
            "share_of_paid": {"type": ["boolean", "null"]},
            "value_of": {"type": ["string", "null"], "enum": [*attributes, None]},
            "top_by": {"type": ["string", "null"], "enum": ["count", "money", None]},
            "weekday": {"type": ["integer", "null"]},
            "per_weekday": {"type": ["boolean", "null"]},
            "period": _period_object(),
        },
        "required": list(_KEYS),
        "additionalProperties": False,
    }


def _user(
    question: str,
    attributes: tuple[ShopAttribute, ...] | list[ShopAttribute],
    rules: tuple[RuleRow, ...] | list[RuleRow],
    as_of: datetime.date,
) -> str:
    lines = [f"Hôm nay (as_of): {as_of.isoformat()}", "Text columns:"]
    for attr in attributes:
        about = f" ({attr.description})" if attr.description else ""
        lines.append(f'- "{attr.name}"{about}; e.g. ' + "; ".join(attr.samples))
    if rules:
        lines.append("Rulebook rows:")
        lines.extend(f"- {row.name}: {row.text}" for row in rules)
    lines.append(f"Question: {question}")
    return "\n".join(lines)


def build_attr_messages(
    question: str,
    attributes: list[ShopAttribute],
    rules: list[RuleRow],
    as_of: datetime.date,
    few_shot_set: int,
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": _SYSTEM}]
    for text, parsed in ATTR_FEW_SHOT_SETS[few_shot_set % len(ATTR_FEW_SHOT_SETS)]:
        content = _user(text, _EXAMPLE_ATTRIBUTES, _EXAMPLE_RULES, _ASOF)
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": json.dumps(parsed, ensure_ascii=False)})
    messages.append({"role": "user", "content": _user(question, attributes, rules, as_of)})
    return messages


def to_attr_query(
    raw: object, as_of: datetime.date, question: str, attributes: list[str]
) -> AttrQuery | None:
    if not isinstance(raw, dict):
        raise ConvertError("response is not an object")
    if raw.get("family") != "attr":
        return None
    measure = raw.get("measure")
    if measure not in {"count", "money", "distinct", "share", "top_value"}:
        raise ConvertError(f"bad measure {measure!r}")
    text = QuestionText(question)
    staff = raw.get("staff")
    if staff is not None and (not isinstance(staff, str) or not text.mentions_value(staff)):
        raise ConvertError(f"staff {staff!r} is not grounded in the question")
    where: list[AttrCondition] = []
    for cond in raw.get("where") or []:
        if not isinstance(cond, dict) or cond.get("attribute") not in attributes:
            raise ConvertError(f"unknown column in {cond!r}")
        contains = str(cond.get("contains") or "").strip()
        if not fold(contains):
            raise ConvertError("empty condition")
        where.append(
            AttrCondition(
                attribute=cond["attribute"], contains=contains, negate=bool(cond.get("negate"))
            )
        )
    distinct_of = raw.get("distinct_of")
    if measure == "distinct" and distinct_of not in attributes:
        raise ConvertError("distinct needs distinct_of")
    if measure != "distinct":
        distinct_of = None
    if measure == "share" and not where:
        raise ConvertError("share needs a condition")
    value_of = raw.get("value_of") if measure == "top_value" else None
    if measure == "top_value" and value_of not in attributes:
        raise ConvertError("top_value needs value_of")
    top_by = raw.get("top_by") or "count"
    if top_by not in {"count", "money"}:
        raise ConvertError(f"bad top_by {top_by!r}")
    weekday = raw.get("weekday")
    bad_day = isinstance(weekday, bool) or not isinstance(weekday, int) or not 1 <= weekday <= 7
    if weekday is not None and bad_day:
        raise ConvertError(f"weekday must be 1..7, got {weekday!r}")
    per_weekday = bool(raw.get("per_weekday"))
    if per_weekday and (weekday is None or measure not in {"count", "money"}):
        raise ConvertError("per_weekday needs a weekday and a count or money")
    if raw.get("period") is None:
        raise ConvertError("period is required")
    return AttrQuery(
        measure=measure,
        staff=staff,
        where=where,
        distinct_of=distinct_of,
        share_of_paid=bool(raw.get("share_of_paid")) if measure == "share" else False,
        value_of=value_of,
        top_by=top_by if measure == "top_value" else "count",
        weekday=weekday,
        per_weekday=per_weekday,
        period=period_from_raw(raw["period"], as_of),
    )  # type: ignore[arg-type]


def parse_attr_question(
    question: str,
    attributes: list[ShopAttribute],
    rules: list[RuleRow],
    client: Any,
    rule_config: RuleConfig,
    cal: LunarCalendar,
    as_of: datetime.date,
) -> AttrParseResult:
    """Two votes; they must agree (dates compared resolved)."""
    names = [attr.name for attr in attributes]
    attempts: list[ParseAttempt] = []
    specs: list[AttrQuery] = []
    for few_shot_set in range(max(settings.parse_votes, 1)):
        messages = build_attr_messages(question, attributes, rules, as_of, few_shot_set)
        try:
            raw = client.chat_json(
                messages,
                _schema(names),
                "AttrQuestion",
                max_tokens=_MAX_TOKENS,
                required_keys=list(_KEYS),
            )
        except LLMUnavailable:
            raise
        except LLMError as exc:
            attempts.append(ParseAttempt(few_shot_set, None, str(exc), 0))
            return AttrParseResult(None, "attr_parse_error", attempts)
        try:
            spec = to_attr_query(raw, as_of, question, names)
        except ConvertError as exc:
            attempts.append(ParseAttempt(few_shot_set, raw, str(exc), 0))
            return AttrParseResult(None, "attr_parse_error", attempts)
        attempts.append(ParseAttempt(few_shot_set, raw, None, 0))
        if spec is None:
            return AttrParseResult(None, "attr_none", attempts)
        specs.append(spec)
    keys = {_key(spec, as_of, rule_config, cal) for spec in specs}
    if len(keys) != 1:
        return AttrParseResult(None, "attr_disagreement", attempts)
    return AttrParseResult(specs[0], None, attempts)


def _key(spec: AttrQuery, as_of: datetime.date, rules: RuleConfig, cal: LunarCalendar) -> str:
    row = spec.model_dump(mode="json")
    row["staff"] = fold(spec.staff) if spec.staff else None
    row["where"] = sorted((c.attribute, fold(c.contains), c.negate) for c in spec.where)
    try:
        window = resolve(spec.period, as_of, rules, cal)
        row["period"] = [window.start.isoformat(), window.end.isoformat()]
    except (PeriodError, CalendarError, ValueError):
        row["period"] = {"unresolved": row["period"]}
    return json.dumps(row, sort_keys=True, ensure_ascii=False)


def describe_attr(
    spec: AttrQuery, as_of: datetime.date, rules: RuleConfig, cal: LunarCalendar
) -> str:
    try:
        window = resolve(spec.period, as_of, rules, cal)
        span = f"{window.start.isoformat()} .. {window.end.isoformat()}"
    except (PeriodError, CalendarError, ValueError):
        span = f"unresolved {spec.period!r}"
    conds = [
        f'"{c.attribute}" {"does not contain" if c.negate else "contains"} "{c.contains}"'
        for c in spec.where
    ]
    what = {
        "count": "number of records",
        "money": "money on the records",
        "distinct": f'number of different "{spec.distinct_of}" values',
        "share": "percent of records matching the conditions, out of "
        + ("records with money" if spec.share_of_paid else "all records"),
        "top_value": f'the "{spec.value_of}" value with the most '
        + ("money" if spec.top_by == "money" else "records"),
    }[spec.measure]
    days = ["", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    if spec.per_weekday and spec.weekday is not None:
        day = days[spec.weekday]
        what = (
            f"average {what} per {day} (the {day}s' total divided by the number of {day}s in"
            " the period, days without records included)"
        )
    lines = [
        f"measure: {what}",
        "records: " + (" and ".join(conds) if conds else "all"),
        f'staff: "{spec.staff}" only' if spec.staff else "staff: all",
        f"period: {span} (record date)",
    ]
    if spec.weekday is not None and not spec.per_weekday:
        lines.append(f"only records on {days[spec.weekday]}s")
    lines.append("output: one value" if spec.measure == "top_value" else "output: one number")
    return "\n".join(lines)


def readme_column_docs(readme_text: str | None) -> dict[str, str]:
    """Column descriptions from a shop README ("- **Xe**: dòng xe và biển số ...")."""
    docs: dict[str, str] = {}
    for line in (readme_text or "").splitlines():
        stripped = line.strip()
        if not stripped.startswith("- **"):
            continue
        name, sep, rest = stripped[4:].partition("**")
        if sep and rest.startswith(":"):
            docs.setdefault(name.strip(), rest[1:].strip())
    return docs

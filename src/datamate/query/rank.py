"""Rank block: who or what comes first by a count or amount, read from a rulebook term.

One spec covers every "X nhất" the rulebook may define: an entity (customer, staff,
product, day, hour, supplier), a measure, which orders count, an optional minimum number
of orders, and the period. The parser only fills these fields from the question and the
term's definition; it never writes a computation of its own.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from datamate.config import settings
from datamate.llm.client import LLMError, LLMUnavailable
from datamate.query.context import ShopContext
from datamate.query.convert import ConvertError, period_from_raw
from datamate.query.grounding import QuestionText
from datamate.query.parse_schema import PERIOD_FIELDS, PERIOD_KINDS
from datamate.query.prompts import PERIOD_KIND_LINES
from datamate.rules.model import RuleConfig
from datamate.rules.parse_rulebook import RuleRow
from datamate.timeres.calendar import CalendarError, LunarCalendar
from datamate.timeres.periods import PeriodError, PeriodExpr, resolve

RankEntity = Literal["customer", "employee", "product", "day", "hour", "supplier"]
RankMeasure = Literal[
    "revenue",
    "order_count",
    "active_days",
    "avg_order_value",
    "qty_base",
    "gross_profit",
    "purchase_amount",
    "receipt_count",
]
RankOrders = Literal["sold", "cancelled"]

# Measures each entity can be ranked by.
RANK_MEASURES: dict[str, frozenset[str]] = {
    "customer": frozenset({"revenue", "order_count", "active_days", "avg_order_value"}),
    "employee": frozenset({"revenue", "order_count", "active_days"}),
    "product": frozenset({"qty_base", "revenue", "gross_profit", "order_count"}),
    "day": frozenset({"revenue", "order_count"}),
    "hour": frozenset({"revenue", "order_count"}),
    "supplier": frozenset({"purchase_amount", "receipt_count"}),
}

_KEYS: tuple[str, ...] = (
    "family",
    "entity",
    "measure",
    "orders",
    "min_orders",
    "mode",
    "k",
    "category",
    "period",
)
_MAX_TOKENS = 300


class RankQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: Literal["rank"] = "rank"
    entity: RankEntity
    measure: RankMeasure
    orders: RankOrders = "sold"
    min_orders: int | None = Field(default=None, ge=1)
    mode: Literal["max", "min"] = "max"
    k: int = Field(default=1, ge=1)
    category: str | None = None
    period: PeriodExpr


@dataclass
class RankParseResult:
    spec: RankQuery | None
    reason: str | None
    raw: list[dict[str, Any] | None]
    errors: list[str | None]


_SYSTEM = "\n".join(
    [
        "A shop owner asks a Vietnamese question about a ranking term that the shop's",
        "rulebook defines. Fill the ranking fields from the definition and the question.",
        "You do not calculate the answer and you do not calculate calendar dates.",
        "",
        "Fields:",
        "- entity: customer | employee (staff, thợ, người bán) | product | day | hour |",
        "  supplier.",
        "- measure: revenue (money of sold orders) | order_count (number of orders) |",
        "  active_days (number of distinct days with an order) | avg_order_value",
        "  (revenue per order) | qty_base (units sold) | gross_profit | purchase_amount",
        "  (money of purchase receipts) | receipt_count (number of purchase receipts).",
        "  customer: revenue, order_count, active_days, avg_order_value.",
        "  employee: revenue, order_count, active_days.",
        "  product: qty_base, revenue, gross_profit, order_count (orders containing it).",
        "  day, hour: revenue, order_count. supplier: purchase_amount, receipt_count.",
        "- orders: sold (the default) | cancelled (the definition counts cancelled orders).",
        "- min_orders: the smallest number of orders an entity needs to be ranked, when",
        "  the definition sets one; else null.",
        "- mode: max for most / highest, min for least / lowest. k: how many to name (1",
        "  unless the question asks for a top N).",
        "- category: a product group from the list, only when the question names it.",
        "- period: from the question.",
        "",
        *PERIOD_KIND_LINES,
        "",
        'Set family to "none" when the definition does not fit these fields, or when',
        "the question asks for something other than who or what ranks first (an amount,",
        "a count, a yes/no).",
        "The worked examples use an invented shop and invented terms.",
    ]
)


@dataclass(frozen=True)
class RankFewShot:
    question: str
    term: str
    definition: str
    parsed: dict[str, Any]


def _period(kind: str, **fields: Any) -> dict[str, Any]:
    row: dict[str, Any] = {name: None for name in PERIOD_FIELDS}
    row["kind"] = kind
    row.update(fields)
    return row


def _parsed(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {key: None for key in _KEYS}
    row["family"] = "rank"
    row.update(overrides)
    return row


_ASOF = datetime.date(2024, 8, 15)
_EXAMPLE_CATEGORIES = ["đồ uống", "bánh"]

# Two sets for the two votes; terms, shop and wording are invented.
RANK_FEW_SHOT_SETS: tuple[tuple[RankFewShot, ...], ...] = (
    (
        RankFewShot(
            "Thang 7/2024 ai la moi quen cua tiem?",
            "Mối quen",
            "người mua đứng đầu về số hoá đơn đã chốt trong khoảng hỏi",
            _parsed(
                entity="customer",
                measure="order_count",
                orders="sold",
                mode="max",
                k=1,
                period=_period("month", month=7, year=2024),
            ),
        ),
        RankFewShot(
            "Tuần này ngày nào là ngày vàng của quán?",
            "Ngày vàng",
            "ngày bán có doanh thu lớn nhất trong khoảng hỏi",
            _parsed(
                entity="day",
                measure="revenue",
                orders="sold",
                mode="max",
                k=1,
                period=_period("week_this"),
            ),
        ),
        RankFewShot(
            "Quý 2/2024 bạn hàng lớn giao cho tiệm tổng mấy đồng?",
            "Bạn hàng lớn",
            "đối tác có tổng tiền các phiếu nhập lớn nhất trong khoảng hỏi",
            _parsed(family="none"),
        ),
        RankFewShot(
            "Thang truoc ai la nguoi ban sieng nhat tiem",
            "Người bán siêng",
            "người bán có số ngày khác nhau có hoá đơn lớn nhất trong khoảng hỏi",
            _parsed(
                entity="employee",
                measure="active_days",
                orders="sold",
                mode="max",
                k=1,
                period=_period("month_last"),
            ),
        ),
        RankFewShot(
            "Trong nhóm bánh, 2 món nào là món quốc dân tháng 6/2024?",
            "Món quốc dân",
            "món góp mặt ở số hoá đơn lớn nhất trong khoảng hỏi",
            _parsed(
                entity="product",
                measure="order_count",
                orders="sold",
                mode="max",
                k=2,
                category="bánh",
                period=_period("month", month=6, year=2024),
            ),
        ),
    ),
    (
        RankFewShot(
            "Năm 2023 ai là người hay bỏ đơn của tiệm?",
            "Người hay bỏ đơn",
            "người mua đứng đầu về số hoá đơn bị bỏ ngang trong khoảng hỏi",
            _parsed(
                entity="customer",
                measure="order_count",
                orders="cancelled",
                mode="max",
                k=1,
                period=_period("year", year=2023),
            ),
        ),
        RankFewShot(
            "khung gio dong khach thang nay la may gio vay",
            "Giờ đông khách",
            "khoảng 60 phút bắt đầu ở phút 00 có doanh thu lớn nhất; trả lời bằng giờ đầu khoảng",
            _parsed(
                entity="hour",
                measure="revenue",
                orders="sold",
                mode="max",
                k=1,
                period=_period("month_this"),
            ),
        ),
        RankFewShot(
            "Tháng 7/2024 mối quen chốt được mấy hoá đơn?",
            "Mối quen",
            "người mua đứng đầu về số hoá đơn đã chốt trong khoảng hỏi",
            _parsed(family="none"),
        ),
        RankFewShot(
            "Quy 1/2024 khach chiu chi nhat cua tiem la ai",
            "Khách chịu chi",
            "người mua có tiền bình quân mỗi hoá đơn lớn nhất; ai dưới 4 hoá đơn thì không xét",
            _parsed(
                entity="customer",
                measure="avg_order_value",
                orders="sold",
                min_orders=4,
                mode="max",
                k=1,
                period=_period("quarter", q=1, year=2024),
            ),
        ),
        RankFewShot(
            "Thang 5/2024 doi tac nao giao phieu cho tiem nhieu lan nhat",
            "Mối nhập quen",
            "đối tác giao số phiếu nhập lớn nhất trong khoảng hỏi, phiếu bị bỏ không đếm",
            _parsed(
                entity="supplier",
                measure="receipt_count",
                orders="sold",
                mode="max",
                k=1,
                period=_period("month", month=5, year=2024),
            ),
        ),
    ),
)


def rank_schema(ctx: ShopContext) -> dict[str, Any]:
    def nullable(values: list[str]) -> dict[str, Any]:
        return {"type": ["string", "null"], "enum": [*values, None]}

    return {
        "type": "object",
        "properties": {
            "family": {"type": "string", "enum": ["rank", "none"]},
            "entity": nullable(list(RANK_MEASURES)),
            "measure": nullable(sorted({m for ms in RANK_MEASURES.values() for m in ms})),
            "orders": nullable(["sold", "cancelled"]),
            "min_orders": {"type": ["integer", "null"]},
            "mode": nullable(["max", "min"]),
            "k": {"type": ["integer", "null"]},
            "category": nullable(list(ctx.categories)),
            "period": {
                "type": ["object", "null"],
                "properties": {
                    "kind": {"type": ["string", "null"], "enum": [*PERIOD_KINDS, None]},
                    **{name: {} for name in PERIOD_FIELDS if name != "kind"},
                },
                "required": list(PERIOD_FIELDS),
                "additionalProperties": False,
            },
        },
        "required": list(_KEYS),
        "additionalProperties": False,
    }


def build_rank_messages(
    question: str,
    term: RuleRow,
    as_of: datetime.date,
    ctx: ShopContext,
    few_shot_set: int,
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": _SYSTEM}]
    for example in RANK_FEW_SHOT_SETS[few_shot_set % len(RANK_FEW_SHOT_SETS)]:
        content = _user_content(
            example.question, example.term, example.definition, _ASOF, _EXAMPLE_CATEGORIES
        )
        messages.append({"role": "user", "content": content})
        messages.append(
            {"role": "assistant", "content": json.dumps(example.parsed, ensure_ascii=False)}
        )
    content = _user_content(question, term.name, term.text, as_of, list(ctx.categories))
    messages.append({"role": "user", "content": content})
    return messages


def parse_rank_question(
    question: str,
    term: RuleRow,
    ctx: ShopContext,
    client: Any,
    rules: RuleConfig,
    cal: LunarCalendar,
    as_of: datetime.date,
) -> RankParseResult:
    """Two votes on different few-shot sets; they must agree (dates compared resolved)."""
    schema = rank_schema(ctx)
    raws: list[dict[str, Any] | None] = []
    errors: list[str | None] = []
    specs: list[RankQuery | None] = []
    for few_shot_set in range(max(settings.parse_votes, 1)):
        messages = build_rank_messages(question, term, as_of, ctx, few_shot_set)
        try:
            raw = client.chat_json(
                messages, schema, "RankQuestion", max_tokens=_MAX_TOKENS, required_keys=list(_KEYS)
            )
        except LLMUnavailable:
            raise
        except LLMError as exc:
            raws.append(None)
            errors.append(str(exc))
            specs.append(None)
            continue
        raws.append(raw if isinstance(raw, dict) else None)
        try:
            spec = to_rank_query(raw, as_of, question)
        except ConvertError as exc:
            errors.append(str(exc))
            specs.append(None)
            continue
        errors.append(None)
        specs.append(spec)
    if any(error is not None for error in errors):
        return RankParseResult(None, "rank_parse_error", raws, errors)
    if all(spec is None for spec in specs):
        return RankParseResult(None, "rank_none", raws, errors)
    if any(spec is None for spec in specs):
        return RankParseResult(None, "rank_disagreement", raws, errors)
    agreed = [spec for spec in specs if spec is not None]
    keys = [rank_key(spec, as_of, rules, cal) for spec in agreed]
    if len(set(keys)) == 1:
        return RankParseResult(agreed[0], None, raws, errors)
    return RankParseResult(None, "rank_disagreement", raws, errors)


def to_rank_query(
    raw: object, as_of: datetime.date, question: str | None = None
) -> RankQuery | None:
    """Flat parser JSON -> RankQuery; None for family "none"."""
    if not isinstance(raw, dict):
        raise ConvertError("response is not an object")
    if raw.get("family") != "rank":
        return None
    entity = raw.get("entity")
    measure = raw.get("measure")
    if entity not in RANK_MEASURES:
        raise ConvertError(f"unknown entity {entity!r}")
    if measure not in RANK_MEASURES[entity]:
        raise ConvertError(f"{entity} cannot be ranked by {measure!r}")
    orders = raw.get("orders") or "sold"
    if orders not in {"sold", "cancelled"}:
        raise ConvertError(f"orders must be sold|cancelled, got {orders!r}")
    if orders == "cancelled" and (measure != "order_count" or entity == "supplier"):
        raise ConvertError("cancelled orders are only counted")
    min_orders = raw.get("min_orders")
    if min_orders is not None:
        if isinstance(min_orders, bool) or not isinstance(min_orders, int) or min_orders < 1:
            raise ConvertError("min_orders must be an integer >= 1")
        if entity not in {"customer", "employee"}:
            raise ConvertError("min_orders applies to customers and staff")
    k = raw.get("k") or 1
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ConvertError("k must be an integer >= 1")
    mode = raw.get("mode") or "max"
    if mode not in {"max", "min"}:
        raise ConvertError(f"mode must be max|min, got {mode!r}")
    category = raw.get("category")
    if category is not None:
        if entity != "product":
            raise ConvertError("category applies to products")
        if question is not None and not QuestionText(question).mentions_value(str(category)):
            raise ConvertError(f"category {category!r} is not grounded in the question")
    period_raw = raw.get("period")
    if period_raw is None:
        raise ConvertError("period is required")
    period = period_from_raw(period_raw, as_of)
    return RankQuery(
        entity=entity,
        measure=measure,
        orders=orders,
        min_orders=min_orders,
        mode=mode,
        k=k,
        category=category,
        period=period,
    )  # type: ignore[arg-type]


def rank_key(spec: RankQuery, as_of: datetime.date, rules: RuleConfig, cal: LunarCalendar) -> str:
    row = spec.model_dump(mode="json")
    try:
        window = resolve(spec.period, as_of, rules, cal)
        row["period"] = {"range": [window.start.isoformat(), window.end.isoformat()]}
    except (PeriodError, CalendarError, ValueError):
        row["period"] = {"unresolved": row["period"]}
    return json.dumps(row, sort_keys=True, ensure_ascii=False)


_ENTITY_WORDS = {
    "customer": "customers",
    "employee": "staff members (by the staff code on the order)",
    "product": "products",
    "day": "days (order date)",
    "hour": "one-hour slots by order time (answer: the slot's starting hour)",
    "supplier": "suppliers (purchase receipts)",
}
_MEASURE_WORDS = {
    "revenue": "revenue of the orders",
    "order_count": "number of orders",
    "active_days": "number of distinct days with at least one order",
    "avg_order_value": "average revenue per order",
    "qty_base": "units sold (base unit)",
    "gross_profit": "gross profit",
    "purchase_amount": "total money of purchase receipts",
    "receipt_count": "number of purchase receipts",
}


def describe_rank(
    spec: RankQuery,
    as_of: datetime.date,
    rules: RuleConfig,
    cal: LunarCalendar,
    customer_who: str | None = None,
) -> str:
    entity = _ENTITY_WORDS[spec.entity]
    if spec.entity == "customer" and customer_who:
        entity = customer_who
    if spec.entity == "supplier":
        orders = "purchase receipts, cancelled receipts excluded"
    elif spec.orders == "cancelled":
        orders = "cancelled orders only"
    else:
        orders = "sold orders (cancelled and returned orders excluded)"
    try:
        window = resolve(spec.period, as_of, rules, cal)
        span = f"{window.start.isoformat()} .. {window.end.isoformat()}"
    except (PeriodError, CalendarError, ValueError):
        span = f"unresolved {spec.period!r}"
    side = "highest" if spec.mode == "max" else "lowest"
    output = (
        f"one ({side} first, tie by code)"
        if spec.k == 1
        else f"top {spec.k} ({side} first, tie by code)"
    )
    lines = [
        f"rank: {entity}",
        f"measure: {_MEASURE_WORDS[spec.measure]}",
        f"orders counted: {orders}",
    ]
    if spec.min_orders is not None:
        lines.append(f"only those with at least {spec.min_orders} orders in the period")
    if spec.entity == "product":
        lines.append(f"products: {spec.category!r} group" if spec.category else "products: all")
    lines.append(f"period: {span}")
    lines.append(f"output: {output}")
    return "\n".join(lines)


def _user_content(
    question: str,
    term: str,
    definition: str,
    as_of: datetime.date,
    categories: list[str],
) -> str:
    return "\n".join(
        [
            f"Hôm nay (as_of): {as_of.isoformat()}",
            f"category: {', '.join(categories) if categories else '(none)'}",
            f"Term: {term}",
            f"Rulebook definition: {definition}",
            f"Question: {question}",
        ]
    )

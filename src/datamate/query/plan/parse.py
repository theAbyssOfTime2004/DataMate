"""Select rule notes + compose a QueryPlan (2 votes, semantic-key agreement)."""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import TypeAdapter, ValidationError

from datamate.config import settings
from datamate.llm.client import LLMError, LLMUnavailable
from datamate.query.context import ShopContext
from datamate.query.convert import _COUNT_ALIAS, _PERIOD_FIELDS, ConvertError, period_from_raw
from datamate.query.parse import ParseAttempt
from datamate.query.parse_schema import PERIOD_KINDS, _period_schema
from datamate.query.plan.spec import QueryPlan
from datamate.rules.model import RuleConfig
from datamate.rules.parse_rulebook import RuleRow
from datamate.timeres.calendar import CalendarError, LunarCalendar
from datamate.timeres.periods import PeriodError, PeriodExpr, resolve

_PERIOD_ADAPTER: TypeAdapter[PeriodExpr] = TypeAdapter(PeriodExpr)
_MAX_NOTES = 3
_MAX_TOKENS = 500

_ENTITIES = [
    "order",
    "customer",
    "product",
    "category",
    "employee",
    "supplier",
    "day",
    "weekday",
    "hour",
    "month",
    "payment_method",
    "channel",
    "none",
]
_MEASURES = [
    "revenue",
    "order_count",
    "distinct_order_days",
    "avg_order_value",
    "cancelled_order_count",
    "returned_order_count",
    "qty_base",
    "orders_containing",
    "distinct_customers",
    "receipt_count",
    "purchase_amount",
    "gross_profit",
]
_SCOPES = ["revenue_orders", "all_orders", "cancelled_orders"]
_HAVING_OPS = [">=", ">", "<=", "<", "="]
_HAVING_MEASURES = [
    "revenue",
    "order_count",
    "distinct_order_days",
    "avg_order_value",
    "qty_base",
    "gross_profit",
    "receipt_count",
    "purchase_amount",
]


@dataclass
class PlanParseResult:
    plan: QueryPlan | None
    reason: str | None
    selected_note_ids: list[str] = field(default_factory=list)
    attempts: list[ParseAttempt] = field(default_factory=list)
    selected_notes: list[RuleRow] = field(default_factory=list)
    notes_error: str | None = None


# Invented definitions only (leakage-tested).
_COMPOSE_FEW_SHOTS: tuple[tuple[str, str, dict[str, Any]], ...] = (
    (
        "Who is the biggest spender among customers with at least 2 orders this month?",
        "Invented: big spender = highest revenue among customers with >= 2 orders.",
        {
            "entity": "customer",
            "measure": "revenue",
            "scope": "revenue_orders",
            "filters": {
                "period": {"kind": "month_this"},
                "product": None,
                "category": None,
                "customer": None,
                "employee": None,
                "channel": None,
                "payment_method": None,
                "weekday": None,
                "hour_from": None,
                "hour_to": None,
            },
            "having": [{"measure": "order_count", "op": ">=", "value": 2}],
            "output": {"kind": "argmax"},
        },
    ),
    (
        "Which clock hour had the fewest orders yesterday?",
        "Invented: quietest hour = hour bucket with minimum order_count.",
        {
            "entity": "hour",
            "measure": "order_count",
            "scope": "all_orders",
            "filters": {
                "period": {"kind": "yesterday"},
                "product": None,
                "category": None,
                "customer": None,
                "employee": None,
                "channel": None,
                "payment_method": None,
                "weekday": None,
                "hour_from": None,
                "hour_to": None,
            },
            "having": [],
            "output": {"kind": "argmin"},
        },
    ),
    (
        "How many suppliers did we buy from in June 2026?",
        "Invented: active suppliers = distinct suppliers with >=1 non-cancelled receipt.",
        {
            "entity": "supplier",
            "measure": "receipt_count",
            "scope": "all_orders",
            "filters": {
                "period": {"kind": "month", "month": 6, "year": 2026, "year_rel": None},
                "product": None,
                "category": None,
                "customer": None,
                "employee": None,
                "channel": None,
                "payment_method": None,
                "weekday": None,
                "hour_from": None,
                "hour_to": None,
            },
            "having": [{"measure": "receipt_count", "op": ">=", "value": 1}],
            "output": {"kind": "count_entities"},
        },
    ),
)


def _period_object() -> dict[str, Any]:
    # Same nullable period schema as the main parser (untyped fields invite the model to
    # fill date/start/end for a plain month).
    return _period_schema()


def _output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": [
                    "argmax",
                    "argmin",
                    "top_k",
                    "count_entities",
                    "value",
                    "share_of_total",
                    "avg_per_period",
                ],
            },
            "k": {"type": ["integer", "null"]},
            "unit": {"type": ["string", "null"], "enum": ["day", "weekday_occurrence", None]},
            "include_zero_days": {"type": ["boolean", "null"]},
        },
        "required": ["kind", "k", "unit", "include_zero_days"],
        "additionalProperties": False,
    }


def _nullable_string(enum_values: list[str] | None = None) -> dict[str, Any]:
    if enum_values:
        return {"type": ["string", "null"], "enum": [*enum_values, None]}
    return {"type": ["string", "null"]}


def plan_schema(ctx: ShopContext) -> dict[str, Any]:
    # Products stay free text (resolved by name in compile): an enum cut to the first N
    # names would force a wrong product in larger shops.
    categories = list(ctx.categories)[:200]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["entity", "measure", "scope", "filters", "having", "output"],
        "properties": {
            "entity": {"type": "string", "enum": _ENTITIES},
            "measure": {"type": "string", "enum": _MEASURES},
            "scope": {"type": ["string", "null"], "enum": [*_SCOPES, None]},
            "filters": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "period",
                    "product",
                    "category",
                    "customer",
                    "employee",
                    "channel",
                    "payment_method",
                    "weekday",
                    "hour_from",
                    "hour_to",
                ],
                "properties": {
                    "period": _period_object(),
                    "product": {"type": ["string", "null"]},
                    "category": _nullable_string(categories or None),
                    "customer": {"type": ["string", "null"]},
                    "employee": {"type": ["string", "null"]},
                    "channel": _nullable_string(list(ctx.channels)[:40] or None),
                    "payment_method": _nullable_string(list(ctx.payment_methods)[:40] or None),
                    "weekday": {"type": ["integer", "null"]},
                    "hour_from": {"type": ["integer", "null"]},
                    "hour_to": {"type": ["integer", "null"]},
                },
            },
            "having": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["measure", "op", "value"],
                    "properties": {
                        "measure": {"type": "string", "enum": _HAVING_MEASURES},
                        "op": {"type": "string", "enum": _HAVING_OPS},
                        "value": {"type": "number"},
                    },
                },
            },
            "output": _output_schema(),
        },
    }


def _notes_schema(note_ids: list[str]) -> dict[str, Any]:
    choices = [*note_ids, "none"]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["note_ids"],
        "properties": {
            "note_ids": {
                "type": "array",
                "maxItems": _MAX_NOTES,
                "items": {"type": "string", "enum": choices},
            }
        },
    }


def _note_label(note: RuleRow) -> str:
    section = note.section_path.split(" > ")[-1][:40] if note.section_path else ""
    return f"[{section}] {note.name}" if section else note.name


def select_rule_notes(
    question: str,
    notes: list[RuleRow],
    client: Any,
) -> tuple[list[RuleRow], str | None]:
    """Pick up to _MAX_NOTES notes by name. Returns (notes, error).

    The catalogue lists short ids and rule names only: every row with its text does not
    fit the model context once a rulebook has ~100 rows.
    """
    if not notes:
        return [], None
    by_id = {f"n{i}": n for i, n in enumerate(notes, start=1)}
    catalogue = "\n".join(f"- {nid}: {_note_label(n)}" for nid, n in by_id.items())
    messages = [
        {
            "role": "system",
            "content": (
                "Pick up to 3 rule ids whose definition is needed to answer the question "
                '(a term in the question that a rule defines), or ["none"]. JSON only.'
            ),
        },
        {
            "role": "user",
            "content": f"Question: {question}\n\nRules:\n{catalogue}\n\nReturn note_ids.",
        },
    ]
    try:
        raw = client.chat_json(
            messages,
            _notes_schema(list(by_id)),
            "PlanSelectNotes",
            max_tokens=128,
        )
    except LLMUnavailable:
        raise
    except LLMError as exc:
        return [], str(exc)
    ids = raw.get("note_ids") if isinstance(raw, dict) else None
    if not isinstance(ids, list):
        return [], "note_ids missing"
    selected: list[RuleRow] = []
    for item in ids:
        if item == "none":
            continue
        row = by_id.get(item)
        if row is not None and row not in selected:
            selected.append(row)
        if len(selected) >= _MAX_NOTES:
            break
    return selected, None


def _block_meanings() -> str:
    return (
        "Blocks:\n"
        "- entity: what to group or identify (customer, product, hour, supplier, none=whole shop).\n"
        "- measure: what to compute; revenue/qty/gross_profit use RuleConfig revenue rules; "
        "purchase_* ignore cancelled receipts; customer measures exclude walk-ins.\n"
        "- scope: which orders (revenue_orders default for sales measures).\n"
        "- filters.period: required PeriodExpr; optional product/category/customer/…\n"
        "- having: per-entity thresholds on the same measure (e.g. order_count >= 2).\n"
        "- output: argmax/argmin/top_k/count_entities/value/share_of_total/avg_per_period.\n"
    )


def _compose_messages(
    question: str,
    as_of: datetime.date,
    ctx: ShopContext,
    selected: list[RuleRow],
    few_shot_set: int = 0,
) -> list[dict[str, str]]:
    notes_block = "(no rule notes)"
    if selected:
        notes_block = "\n".join(f"- {n.name}: {n.text}" for n in selected)
    # Each vote sees the examples in a different order, so the votes are distinct requests
    # (identical prompts share one cache entry at temperature 0 and always agree).
    shift = few_shot_set % len(_COMPOSE_FEW_SHOTS)
    shots = _COMPOSE_FEW_SHOTS[shift:] + _COMPOSE_FEW_SHOTS[:shift]
    examples = []
    for q, note, plan in shots:
        examples.append(
            f"Example question: {q}\nNote: {note}\n"
            f"Plan JSON: {json.dumps(plan, ensure_ascii=False)}"
        )
    shop = (
        f"as_of={as_of.isoformat()}; "
        f"products_sample={list(ctx.product_names)[:8]}; "
        f"categories_sample={list(ctx.categories)[:8]}"
    )
    user = (
        f"{_block_meanings()}\n"
        + "\n\n".join(examples)
        + f"\n\nShop: {shop}\nRule notes:\n{notes_block}\n\n"
        f"Question: {question}\nCompose one QueryPlan JSON."
    )
    return [
        {
            "role": "system",
            "content": (
                "Compose a QueryPlan from fixed blocks only. Never invent SQL. "
                "If the blocks cannot express the question, still return the closest valid plan; "
                "the verifier may reject it."
            ),
        },
        {"role": "user", "content": user},
    ]


def parse_plan_question(
    question: str,
    ctx: ShopContext,
    client: Any,
    rules: RuleConfig,
    cal: LunarCalendar,
    as_of: datetime.date,
    notes: list[RuleRow],
    votes: int | None = None,
) -> PlanParseResult:
    selected, notes_error = select_rule_notes(question, notes, client)
    vote_count = settings.parse_votes if votes is None else votes
    schema = plan_schema(ctx)
    attempts: list[ParseAttempt] = []
    plans: list[QueryPlan] = []
    enable_thinking = bool(settings.plan_enable_thinking)

    def result(plan: QueryPlan | None, reason: str | None) -> PlanParseResult:
        return PlanParseResult(
            plan=plan,
            reason=reason,
            selected_note_ids=[n.id for n in selected],
            attempts=attempts,
            selected_notes=selected,
            notes_error=notes_error,
        )

    for few_shot_set in range(vote_count):
        messages = _compose_messages(question, as_of, ctx, selected, few_shot_set)
        raw, error = _call(client, messages, schema, enable_thinking=enable_thinking)
        attempts.append(ParseAttempt(few_shot_set, raw, error, 0))
        if error is not None or raw is None:
            continue
        try:
            plan = _coerce_plan(raw, as_of, rules, cal)
        except (ConvertError, ValidationError, ValueError, TypeError) as exc:
            attempts[-1] = ParseAttempt(few_shot_set, raw, str(exc), 0)
            continue
        plans.append(plan)

    if len(plans) < vote_count:
        # A failed vote is not agreement: require every vote to produce a plan.
        return result(None, "plan_parse_error")
    keys = {plan_semantic_key(p, as_of, rules, cal) for p in plans}
    if len(keys) != 1:
        return result(None, "plan_disagreement")
    return result(plans[0], None)


def _coerce_plan(
    raw: dict[str, Any], as_of: datetime.date, rules: RuleConfig, cal: LunarCalendar
) -> QueryPlan:
    payload = dict(raw)
    out = payload.get("output")
    if isinstance(out, dict):
        kind = out.get("kind")
        cleaned: dict[str, Any] = {"kind": kind}
        if kind == "top_k":
            cleaned["k"] = out.get("k") or 1
        if kind == "avg_per_period":
            cleaned["unit"] = out.get("unit") or "day"
            cleaned["include_zero_days"] = (
                True if out.get("include_zero_days") is None else out.get("include_zero_days")
            )
        payload["output"] = cleaned
    filters = payload.get("filters")
    if isinstance(filters, dict):
        period = _plan_period(filters.get("period"), as_of, rules, cal)
        payload["filters"] = {**filters, "period": period}
    return QueryPlan.model_validate(payload)


def _plan_period(
    raw: object, as_of: datetime.date, rules: RuleConfig, cal: LunarCalendar
) -> PeriodExpr:
    """Period via the main converter, after dropping fields the kind does not use.

    The model often fills date/start/end/n next to a plain month; those extras are dropped
    only when start/end agree with the kind's own window, never silently overridden.
    """
    if not isinstance(raw, dict):
        raise ConvertError("period must be an object")
    kind = raw.get("kind")
    allowed = _PERIOD_FIELDS.get(str(kind))
    if allowed is None:
        raise ConvertError(f"unknown period kind {kind!r}")
    keep = set(allowed) | set(_COUNT_ALIAS.get(str(kind), ()))
    expr = period_from_raw({k: v for k, v in raw.items() if k == "kind" or k in keep}, as_of)
    start, end = raw.get("start"), raw.get("end")
    if (start or end) and "start" not in allowed:
        try:
            window = resolve(expr, as_of, rules, cal)
        except (PeriodError, CalendarError) as exc:
            raise ConvertError(str(exc)) from exc
        if (start and start != window.start.isoformat()) or (end and end != window.end.isoformat()):
            raise ConvertError("period start/end disagree with its kind")
    return expr


def plan_semantic_key(
    plan: QueryPlan, as_of: datetime.date, rules: RuleConfig, cal: LunarCalendar
) -> str:
    dumped = plan.model_dump(mode="json")
    resolved = _resolve_periods(dumped, as_of, rules, cal)
    return json.dumps(resolved, sort_keys=True, ensure_ascii=False)


def _resolve_periods(node: Any, as_of: datetime.date, rules: Any, cal: LunarCalendar) -> Any:
    if (
        isinstance(node, dict)
        and node.get("kind") in PERIOD_KINDS
        and "period" not in node
        and "entity" not in node
    ):
        try:
            window = resolve(_PERIOD_ADAPTER.validate_python(node), as_of, rules, cal)
        except (PeriodError, CalendarError, ValidationError, ValueError):
            return {"unresolved": node}
        return {"range": [window.start.isoformat(), window.end.isoformat()]}
    if isinstance(node, dict):
        return {key: _resolve_periods(value, as_of, rules, cal) for key, value in node.items()}
    if isinstance(node, list):
        return [_resolve_periods(item, as_of, rules, cal) for item in node]
    return node


def _call(
    client: Any,
    messages: list[dict[str, str]],
    schema: dict[str, Any],
    *,
    enable_thinking: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    kwargs: dict[str, Any] = {
        "messages": messages,
        "schema": schema,
        "schema_name": "QueryPlan",
        "max_tokens": _MAX_TOKENS,
    }
    # Optional override supported by LLMClient; FakeClients ignore unknown kwargs carefully.
    try:
        raw = client.chat_json(**kwargs, enable_thinking=enable_thinking)
    except TypeError:
        raw = client.chat_json(**kwargs)
    except LLMUnavailable:
        raise
    except LLMError as exc:
        return None, str(exc)
    if not isinstance(raw, dict):
        return None, "response is not an object"
    return raw, None

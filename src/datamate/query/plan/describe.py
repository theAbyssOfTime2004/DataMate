"""Deterministic English description of a QueryPlan for the verifier.

Same facet-per-line style as the ext family descriptions, in plain words, so the verifier
few-shots carry over. Rulebook notes are NOT listed: a listed definition reads as applied
even when no block expresses it (dev: "prescription drugs" passed as plain revenue).
"""

from __future__ import annotations

import datetime

from datamate.query.plan.spec import QueryPlan, default_scope_for
from datamate.rules.model import RuleConfig
from datamate.timeres.calendar import CalendarError, LunarCalendar
from datamate.timeres.periods import PeriodError, resolve

_ENTITY: dict[str, str] = {
    "order": "order",
    "customer": "customer",
    "product": "product",
    "category": "product category",
    "employee": "employee",
    "supplier": "supplier",
    "day": "calendar day",
    "weekday": "day of the week",
    "hour": "one-hour slot by order creation time (from h:00 to before h+1:00)",
    "month": "calendar month",
    "payment_method": "payment method",
    "channel": "sales channel",
}

_MEASURE: dict[str, str] = {
    "revenue": "money revenue from product lines (fees excluded)",
    "order_count": "number of orders",
    "distinct_order_days": "number of distinct days with at least one order",
    "avg_order_value": "average money revenue per order",
    "cancelled_order_count": "number of cancelled orders",
    "returned_order_count": "number of orders that have a return",
    "qty_base": "quantity sold in the product's base unit (fees excluded)",
    "orders_containing": "number of orders that contain the product",
    "distinct_customers": "number of distinct customers (walk-in buyers excluded)",
    "receipt_count": "number of purchase receipts from suppliers (cancelled excluded)",
    "purchase_amount": "money the shop paid suppliers on purchase receipts (cancelled excluded)",
    "gross_profit": "gross profit = line revenue minus quantity x cost (fees excluded)",
}

_OP: dict[str, str] = {
    ">=": "at least",
    ">": "more than",
    "<=": "at most",
    "<": "less than",
    "=": "exactly",
}

_WEEKDAY = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def describe_plan(
    plan: QueryPlan,
    as_of: datetime.date,
    rules: RuleConfig,
    cal: LunarCalendar,
) -> str:
    measure = _MEASURE.get(plan.measure, plan.measure)
    lines = [f"measure: {measure}"]
    if plan.entity != "none":
        lines.append(f"compared across: each {_ENTITY.get(plan.entity, plan.entity)}")
    if plan.measure not in {"receipt_count", "purchase_amount"}:
        lines.append(f"orders counted: {_scope_phrase(plan, rules)}")
    lines.append(_period_line(plan, as_of, rules, cal))

    applied, unrestricted = _filters(plan)
    lines.append("filters: " + (", ".join(applied) if applied else "none"))
    if plan.having:
        noun = _ENTITY.get(plan.entity, plan.entity)
        bits = [
            f"{_MEASURE.get(h.measure, h.measure)} {_OP[h.op]} {h.value:g}" for h in plan.having
        ]
        lines.append(f"only each {noun} with: " + " and ".join(bits))
    lines.append("output: " + _output_phrase(plan, measure))
    if unrestricted:
        lines.append("not filtered by: " + ", ".join(unrestricted))
    lines.append("any other condition in the question is not applied")
    return "\n".join(lines)


def _scope_phrase(plan: QueryPlan, rules: RuleConfig) -> str:
    scope = plan.scope or default_scope_for(plan.measure)
    if scope == "cancelled_orders":
        return "cancelled orders only"
    if scope == "all_orders":
        return "every order of any status (including cancelled and returned)"
    statuses = ", ".join(rules.revenue_statuses) or "none"
    tail = ", orders with a return excluded" if rules.exclude_orders_with_return else ""
    return f"revenue orders only (status {statuses}{tail})"


def _period_line(
    plan: QueryPlan, as_of: datetime.date, rules: RuleConfig, cal: LunarCalendar
) -> str:
    try:
        window = resolve(plan.filters.period, as_of, rules, cal)
    except (PeriodError, CalendarError):
        return "period: could not be resolved"
    return f"period: {window.start.isoformat()} to {window.end.isoformat()}"


def _filters(plan: QueryPlan) -> tuple[list[str], list[str]]:
    f = plan.filters
    applied: list[str] = []
    unrestricted: list[str] = []
    sales_line = plan.measure in {"revenue", "qty_base", "gross_profit", "avg_order_value"}

    def add(value: object, text: str, name: str) -> None:
        if value is None:
            unrestricted.append(name)
        else:
            applied.append(text)

    if f.product is not None:
        applied.append(
            f'only lines of product "{f.product}"'
            if sales_line or plan.entity == "product"
            else f'only orders containing product "{f.product}"'
        )
    else:
        unrestricted.append("product")
    if f.category is not None:
        applied.append(
            f'only lines in category "{f.category}"'
            if sales_line or plan.entity == "product"
            else f'only orders containing a product of category "{f.category}"'
        )
    else:
        unrestricted.append("category")
    add(f.customer, f'only customer "{f.customer}"', "customer")
    add(f.employee, f'only employee "{f.employee}"', "employee")
    add(f.channel, f'only channel "{f.channel}"', "channel")
    add(f.payment_method, f'only payment method "{f.payment_method}"', "payment method")
    if f.weekday is not None:
        applied.append(f"only {_WEEKDAY[f.weekday]}s")
    else:
        unrestricted.append("day of the week")
    if f.hour_from is not None or f.hour_to is not None:
        start = f.hour_from if f.hour_from is not None else 0
        end = (f.hour_to if f.hour_to is not None else 23) + 1
        applied.append(f"only orders created from {start}:00 to before {end}:00")
    else:
        unrestricted.append("time of day")
    return applied, unrestricted


def _output_phrase(plan: QueryPlan, measure: str) -> str:
    out = plan.output
    noun = _ENTITY.get(plan.entity, plan.entity)
    if out.kind == "argmax":
        return f"the one {noun} with the highest {measure}"
    if out.kind == "argmin":
        return f"the one {noun} with the lowest {measure}"
    if out.kind == "top_k":
        return f"the {out.k} {noun}s with the highest {measure}, best first"
    if out.kind == "count_entities":
        return f"how many {noun}s (one number)"
    if out.kind == "share_of_total" and plan.entity == "none":
        return (
            f"percent = {measure} with the filters above / {measure} of the whole shop in the "
            "same period (same orders counted, no other filter) x 100"
        )
    if out.kind == "share_of_total":
        return (
            f"percent = {measure} of the {noun}s above / {measure} of all {noun}s in the "
            "period x 100"
        )
    if out.kind == "avg_per_period":
        if out.unit == "weekday_occurrence" and plan.filters.weekday is not None:
            day = _WEEKDAY[plan.filters.weekday]
            return f"total {measure} divided by the number of {day}s in the period (days without orders count)"
        if out.include_zero_days:
            return f"total {measure} divided by the number of days in the period (days without orders count)"
        return f"total {measure} divided by the number of days that had orders"
    if plan.entity != "none":
        return f"one number: total {measure} over the {noun}s above"
    return f"one number: total {measure}"

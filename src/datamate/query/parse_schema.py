from __future__ import annotations

from typing import Any

from datamate.query.context import ShopContext

PERIOD_KINDS: tuple[str, ...] = (
    "day",
    "today",
    "yesterday",
    "last_n_days",
    "week_this",
    "week_last",
    "month",
    "month_this",
    "month_last",
    "quarter",
    "year",
    "ytd",
    "range",
    "tet",
    "before_tet",
)

QUESTION_KINDS: tuple[str, ...] = (
    "value",
    "compare",
    "series",
    "multi",
    "trend",
    "argmax",
)

METRICS: tuple[str, ...] = (
    "revenue",
    "order_count",
    "cancel_count",
    "cancel_rate",
    "avg_order_value",
    "revenue_share",
)

COMPARE_KINDS: tuple[str, ...] = (
    "explicit",
    "previous_period",
    "same_period_last_year",
    "before_tet_same_length",
)

TOP_LEVEL_KEYS: tuple[str, ...] = (
    "in_scope",
    "kind",
    "metric",
    "periods",
    "compare_kind",
    "op",
    "filters",
    "filters_b",
    "multi_filters",
    "direction",
    "granularity",
    "mode",
)

PERIOD_FIELDS: tuple[str, ...] = (
    "kind",
    "date",
    "start",
    "end",
    "n",
    "month",
    "q",
    "year",
    "year_rel",
    "n_days",
)

FILTER_KEYS: tuple[str, ...] = (
    "payment_method",
    "channel",
    "customer_segment",
    "category",
    "product",
    "shift",
)


def _string_array(values: list[str]) -> dict[str, Any]:
    if not values:
        return {"type": "array", "maxItems": 0}
    return {"type": "array", "items": {"type": "string", "enum": list(values)}}


def _nullable_enum(values: list[str]) -> dict[str, Any]:
    return {"type": ["string", "null"], "enum": [*values, None]}


def _shift_schema(has_order_ts: bool) -> dict[str, Any]:
    if not has_order_ts:
        return {"enum": [None]}
    return _nullable_enum(["morning", "afternoon"])


def _filters_schema(ctx: ShopContext, products: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "payment_method": _string_array(ctx.payment_methods),
            "channel": _string_array(ctx.channels),
            "customer_segment": _string_array(ctx.segments),
            "category": _string_array(ctx.categories),
            "product": _string_array(products),
            "shift": _shift_schema(ctx.has_order_ts),
        },
        "required": list(FILTER_KEYS),
        "additionalProperties": False,
    }


def _period_schema() -> dict[str, Any]:
    nullable_string = {"type": ["string", "null"]}
    nullable_int = {"type": ["integer", "null"]}
    return {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": list(PERIOD_KINDS)},
            "date": nullable_string,
            "start": nullable_string,
            "end": nullable_string,
            "n": nullable_int,
            "month": nullable_int,
            "q": nullable_int,
            "year": nullable_int,
            "year_rel": nullable_int,
            "n_days": nullable_int,
        },
        "required": list(PERIOD_FIELDS),
        "additionalProperties": False,
    }


def parsed_question_schema(ctx: ShopContext, products: list[str]) -> dict[str, Any]:
    """Flat JSON schema for one shop. Enums come only from that shop's context."""
    filters = _filters_schema(ctx, products)
    return {
        "type": "object",
        "properties": {
            "in_scope": {"type": "boolean"},
            "kind": {"type": "string", "enum": list(QUESTION_KINDS)},
            "metric": {"type": "string", "enum": list(METRICS)},
            "periods": {"type": "array", "items": _period_schema()},
            "compare_kind": _nullable_enum(list(COMPARE_KINDS)),
            "op": _nullable_enum(["pct_change", "diff"]),
            "filters": filters,
            "filters_b": {"anyOf": [{"type": "null"}, filters]},
            "multi_filters": {"type": "array", "items": filters},
            "direction": _nullable_enum(["increasing", "decreasing"]),
            "granularity": _nullable_enum(["day", "month"]),
            "mode": _nullable_enum(["max", "min"]),
        },
        "required": list(TOP_LEVEL_KEYS),
        "additionalProperties": False,
    }

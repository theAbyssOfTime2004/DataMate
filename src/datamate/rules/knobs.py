"""Registry of RuleConfig knobs: schemas, train defaults, engine-supported values."""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any, Literal

from datamate.rules.model import RuleConfig

KnobKind = Literal["enum", "bool", "int", "string_list", "string_dict", "time_ranges", "object"]


@dataclass(frozen=True)
class Knob:
    id: str
    field_path: str
    meaning: str
    kind: KnobKind
    # JSON-schema-ish fragment used in extract prompts (and light validation).
    value_schema: dict[str, Any]
    train_default: Any
    # Values the engine can honour. None = any value matching the schema is supported.
    supported: frozenset[Any] | None


def _time_ranges_default() -> dict[str, list[str]]:
    return {
        "morning": ["06:00:00", "13:59:59"],
        "afternoon": ["14:00:00", "21:59:59"],
    }


def _tet_default() -> dict[str, int]:
    tet = RuleConfig().tet
    return tet.model_dump(mode="json")


def _shifts_as_json(rules: RuleConfig) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name, (start, end) in rules.shifts.items():
        out[name] = [start.strftime("%H:%M:%S"), end.strftime("%H:%M:%S")]
    return out


ORDER_STATUS_ENUM = ["completed", "cancelled", "returned", "shipping"]


KNOBS: tuple[Knob, ...] = (
    Knob(
        id="revenue_statuses",
        field_path="revenue_statuses",
        meaning="Order statuses that count toward revenue and sold quantities.",
        kind="string_list",
        value_schema={
            "type": "array",
            "items": {"type": "string", "enum": ORDER_STATUS_ENUM},
            "minItems": 1,
        },
        train_default=["completed", "shipping"],
        supported=None,  # any non-empty subset of OrderStatus
    ),
    Knob(
        id="exclude_orders_with_return",
        field_path="exclude_orders_with_return",
        meaning="Whether an order that also has a return is excluded from revenue.",
        kind="bool",
        value_schema={"type": "boolean"},
        train_default=True,
        supported=frozenset({True, False}),
    ),
    Knob(
        id="cancel_rate_denominator",
        field_path="cancel_rate_denominator",
        meaning="Denominator for cancel rate (all orders placed in the period).",
        kind="enum",
        value_schema={"type": "string", "enum": ["all_orders"]},
        train_default="all_orders",
        supported=frozenset({"all_orders"}),
    ),
    Knob(
        id="shifts",
        field_path="shifts",
        meaning="Named shop shifts as inclusive time ranges HH:MM:SS.",
        kind="time_ranges",
        value_schema={
            "type": "object",
            "additionalProperties": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 2,
            },
        },
        train_default=_time_ranges_default(),
        supported=None,
    ),
    Knob(
        id="week_start",
        field_path="week_start",
        meaning="First day of a calendar week.",
        kind="enum",
        value_schema={"type": "string", "enum": ["monday", "sunday"]},
        train_default="monday",
        supported=frozenset({"monday", "sunday"}),
    ),
    Knob(
        id="month_this_compare",
        field_path="month_this_compare",
        meaning="How 'this month' is compared to a prior month (same day-count window).",
        kind="enum",
        value_schema={"type": "string", "enum": ["same_day_count"]},
        train_default="same_day_count",
        supported=frozenset({"same_day_count"}),
    ),
    Knob(
        id="tet",
        field_path="tet",
        meaning="Lunar Tet window: start/end lunar month-day and year offset for the start.",
        kind="object",
        value_schema={
            "type": "object",
            "required": [
                "start_lunar_month",
                "start_lunar_day",
                "start_year_offset",
                "end_lunar_month",
                "end_lunar_day",
            ],
            "properties": {
                "start_lunar_month": {"type": "integer"},
                "start_lunar_day": {"type": "integer"},
                "start_year_offset": {"type": "integer"},
                "end_lunar_month": {"type": "integer"},
                "end_lunar_day": {"type": "integer"},
            },
        },
        train_default=_tet_default(),
        supported=None,
    ),
    Knob(
        id="percent_decimals",
        field_path="percent_decimals",
        meaning="Decimal places that percent answers are rounded to, as stated by a rounding rule. "
        "A formatting example of a percent value (such as 12.4) is not a rounding rule.",
        kind="int",
        value_schema={"type": "integer", "minimum": 0, "maximum": 6},
        train_default=2,
        supported=frozenset({0, 1, 2, 3, 4}),
    ),
    Knob(
        id="rounding",
        field_path="rounding",
        meaning="Rounding mode for money and percents.",
        kind="enum",
        value_schema={"type": "string", "enum": ["half_up"]},
        train_default="half_up",
        supported=frozenset({"half_up"}),
    ),
    Knob(
        id="non_product_line_labels",
        field_path="non_product_line_labels",
        meaning="Line labels that are fees/services, not products (excluded from product rankings).",
        kind="string_list",
        value_schema={"type": "array", "items": {"type": "string"}},
        train_default=["Phí phục vụ"],
        supported=None,
    ),
    Knob(
        id="ranking_measure",
        field_path="ranking_measure",
        meaning="Default measure for best-selling product rankings (qty_base, revenue, or gross_profit).",
        kind="enum",
        value_schema={"type": "string", "enum": ["qty_base", "revenue", "gross_profit"]},
        train_default="qty_base",
        supported=frozenset({"qty_base", "revenue", "gross_profit"}),
    ),
    Knob(
        id="ranking_tie_break",
        field_path="ranking_tie_break",
        meaning="Tie-break when two products share the same ranking measure.",
        kind="enum",
        value_schema={"type": "string", "enum": ["code_asc"]},
        train_default="code_asc",
        supported=frozenset({"code_asc"}),
    ),
    Knob(
        id="price_on_date_include_cancelled_returned",
        field_path="price_on_date_include_cancelled_returned",
        meaning="Whether cancelled/returned lines may supply a unit price on a date.",
        kind="bool",
        value_schema={"type": "boolean"},
        train_default=True,
        supported=frozenset({True, False}),
    ),
    Knob(
        id="size_suffixes",
        field_path="size_suffixes",
        meaning="Cafe size letter to product-name suffix map (e.g. M -> '(M)').",
        kind="string_dict",
        value_schema={
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
        train_default={"M": "(M)", "L": "(L)"},
        supported=None,
    ),
    Knob(
        id="customer_tie_break",
        field_path="customer_tie_break",
        meaning="Tie-break when two customers share the same ranking measure.",
        kind="enum",
        value_schema={"type": "string", "enum": ["code_asc"]},
        train_default="code_asc",
        supported=frozenset({"code_asc"}),
    ),
    Knob(
        id="debt_exclude_payment_labels",
        field_path="debt_exclude_payment_labels",
        meaning="Payment-method labels excluded from debt (COD / cash-on-delivery synonyms).",
        kind="string_list",
        value_schema={"type": "array", "items": {"type": "string"}},
        train_default=list(RuleConfig().debt_exclude_payment_labels),
        supported=None,
    ),
    Knob(
        id="debt_exclude_channel_labels",
        field_path="debt_exclude_channel_labels",
        meaning="Sales-channel labels excluded from debt (marketplaces).",
        kind="string_list",
        value_schema={"type": "array", "items": {"type": "string"}},
        train_default=list(RuleConfig().debt_exclude_channel_labels),
        supported=None,
    ),
    Knob(
        id="lot_expiry_inclusive",
        field_path="lot_expiry_inclusive",
        meaning="Whether the lot expiry window includes as_of + n_days.",
        kind="bool",
        value_schema={"type": "boolean"},
        train_default=True,
        supported=frozenset({True, False}),
    ),
)

KNOBS_BY_ID: dict[str, Knob] = {knob.id: knob for knob in KNOBS}
KNOB_IDS: tuple[str, ...] = tuple(knob.id for knob in KNOBS)

# Families / core metrics → knobs they need. If any listed knob is unsupported, abstain.
FAMILY_DEPENDS_ON: dict[str, frozenset[str]] = {
    # Core metrics (MetricEngine.answer)
    "revenue": frozenset({"revenue_statuses", "exclude_orders_with_return"}),
    "order_count": frozenset({"revenue_statuses", "exclude_orders_with_return"}),
    "aov": frozenset({"revenue_statuses", "exclude_orders_with_return"}),
    "cancel_rate": frozenset({"cancel_rate_denominator"}),
    "revenue_share": frozenset({"revenue_statuses", "exclude_orders_with_return"}),
    # Ext families
    "qty_sold": frozenset(
        {
            "revenue_statuses",
            "exclude_orders_with_return",
            "non_product_line_labels",
            "size_suffixes",
        }
    ),
    "top_products": frozenset(
        {
            "ranking_measure",
            "ranking_tie_break",
            "revenue_statuses",
            "exclude_orders_with_return",
            "non_product_line_labels",
            "size_suffixes",
        }
    ),
    "unsold_products": frozenset(
        {"revenue_statuses", "exclude_orders_with_return", "non_product_line_labels"}
    ),
    "products_in_min_orders": frozenset(
        {"revenue_statuses", "exclude_orders_with_return", "non_product_line_labels"}
    ),
    "price_on_date": frozenset({"price_on_date_include_cancelled_returned"}),
    "price_increase_effect": frozenset(
        {"revenue_statuses", "exclude_orders_with_return", "non_product_line_labels"}
    ),
    "customers_min_orders": frozenset(
        {"revenue_statuses", "exclude_orders_with_return", "customer_tie_break"}
    ),
    "top_customers": frozenset(
        {"revenue_statuses", "exclude_orders_with_return", "customer_tie_break"}
    ),
    "orders_min_amount": frozenset({"revenue_statuses", "exclude_orders_with_return"}),
    "days_min_revenue": frozenset({"revenue_statuses", "exclude_orders_with_return"}),
    "debt_total": frozenset({"debt_exclude_payment_labels", "debt_exclude_channel_labels"}),
    "top_debtors": frozenset(
        {"debt_exclude_payment_labels", "debt_exclude_channel_labels", "customer_tie_break"}
    ),
    "debt_aging": frozenset({"debt_exclude_payment_labels", "debt_exclude_channel_labels"}),
    "debt_added": frozenset({"debt_exclude_payment_labels", "debt_exclude_channel_labels"}),
    "debt_collected": frozenset({"debt_exclude_payment_labels", "debt_exclude_channel_labels"}),
    "purchase_total": frozenset(),
    "top_supplier": frozenset(),
    "avg_purchase_cost": frozenset(),
    "stock_on_hand": frozenset(),
    "stock_value": frozenset(),
    "out_of_stock_count": frozenset(),
    "lots_expiring": frozenset({"lot_expiry_inclusive"}),
    "nearest_expiry_lot": frozenset({"lot_expiry_inclusive"}),
    "ingredient_revenue": frozenset(
        {"revenue_statuses", "exclude_orders_with_return", "non_product_line_labels"}
    ),
    "rx_share": frozenset({"revenue_statuses", "exclude_orders_with_return"}),
    "gross_profit": frozenset(
        {"revenue_statuses", "exclude_orders_with_return", "non_product_line_labels"}
    ),
    "net_profit": frozenset({"revenue_statuses", "exclude_orders_with_return"}),
    "top_employee": frozenset({"revenue_statuses", "exclude_orders_with_return"}),
}


def first_unsupported_dependency(family: str, unsupported: set[str] | frozenset[str]) -> str | None:
    """Return the first depends_on knob that is unsupported, else None."""
    deps = FAMILY_DEPENDS_ON.get(family, frozenset())
    hits = sorted(deps & set(unsupported))
    return hits[0] if hits else None


def rule_unsupported_reason(family: str, unsupported: set[str] | frozenset[str]) -> str | None:
    knob = first_unsupported_dependency(family, unsupported)
    if knob is None:
        return None
    return f"rule_unsupported:{knob}"


def parse_shift_ranges(raw: dict[str, Any]) -> dict[str, tuple[datetime.time, datetime.time]]:
    """Coerce extract JSON into RuleConfig.shifts."""
    out: dict[str, tuple[datetime.time, datetime.time]] = {}
    for name, pair in raw.items():
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise TypeError(f"shift {name!r} must be [start, end]")
        start = _parse_hms(pair[0])
        end = _parse_hms(pair[1])
        out[str(name)] = (start, end)
    return out


def _parse_hms(value: Any) -> datetime.time:
    if isinstance(value, datetime.time):
        return value
    if not isinstance(value, str):
        raise TypeError(f"time value must be string, got {type(value).__name__}")
    parts = value.strip().split(":")
    if len(parts) == 2:
        hour, minute = int(parts[0]), int(parts[1])
        return datetime.time(hour, minute, 0)
    if len(parts) == 3:
        hour, minute, second = int(parts[0]), int(parts[1]), int(parts[2])
        return datetime.time(hour, minute, second)
    raise ValueError(f"invalid time {value!r}")


def shifts_for_prompt(rules: RuleConfig | None = None) -> dict[str, list[str]]:
    return _shifts_as_json(rules or RuleConfig())

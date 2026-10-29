from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import Literal

from datamate.query.spec import Metric
from datamate.rules.model import RuleConfig

MONEY_METRICS: frozenset[str] = frozenset({"revenue", "avg_order_value"})
PERCENT_METRICS: frozenset[str] = frozenset({"cancel_rate", "revenue_share"})
COUNT_METRICS: frozenset[str] = frozenset({"order_count", "cancel_count"})


def _as_decimal(value: Decimal | int) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return Decimal(value)


def quantize_metric(metric: str, value: Decimal | int, rules: RuleConfig) -> Decimal:
    amount = _as_decimal(value)
    if metric in PERCENT_METRICS:
        quantum = Decimal(1).scaleb(-rules.percent_decimals)
        return amount.quantize(quantum, rounding=ROUND_HALF_UP)
    if metric in MONEY_METRICS:
        return amount.quantize(Decimal(1), rounding=ROUND_HALF_UP)
    return amount.to_integral_value(rounding=ROUND_HALF_UP)


def metric_answer_type(metric: str) -> Literal["number", "percent"]:
    if metric in PERCENT_METRICS:
        return "percent"
    return "number"


def render_metric(metric: str, value: Decimal | int, rules: RuleConfig) -> int | float:
    rounded = quantize_metric(metric, value, rules)
    if metric in PERCENT_METRICS:
        return float(rounded)
    return int(rounded)


def render_percent(value: Decimal | int, rules: RuleConfig) -> float:
    quantum = Decimal(1).scaleb(-rules.percent_decimals)
    return float(_as_decimal(value).quantize(quantum, rounding=ROUND_HALF_UP))


def render_diff(metric: Metric, value: Decimal, rules: RuleConfig) -> int | float:
    return render_metric(metric, value, rules)

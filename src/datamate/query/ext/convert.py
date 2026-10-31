"""Turn flat extension-parser JSON into an ExtQuerySpec."""

from __future__ import annotations

import datetime
from typing import Any

from pydantic import TypeAdapter, ValidationError

from datamate.query.convert import ConvertError, period_from_raw
from datamate.query.ext.spec import (
    CustomersMinOrdersQuery,
    DaysMinRevenueQuery,
    ExtQuerySpec,
    NoneQuery,
    OrdersMinAmountQuery,
    PriceIncreaseEffectQuery,
    PriceOnDateQuery,
    ProductsInMinOrdersQuery,
    QtySoldQuery,
    TopCustomersQuery,
    TopProductsQuery,
    UnsoldProductsQuery,
)
from datamate.query.grounding import QuestionText
from datamate.timeres.periods import (
    DayExpr,
    PeriodExpr,
    TodayExpr,
    YesterdayExpr,
)

_SPEC_ADAPTER: TypeAdapter[ExtQuerySpec] = TypeAdapter(ExtQuerySpec)

_SINGLE_DAY_TYPES = (DayExpr, TodayExpr, YesterdayExpr)


def to_ext_query_spec(
    parsed: dict[str, Any],
    as_of: datetime.date,
    question: str | None = None,
) -> ExtQuerySpec:
    text = QuestionText(question) if question is not None else None
    family = parsed.get("family")
    if family is None or family == "none":
        return NoneQuery()

    product = _optional_str(parsed.get("product"))
    category = _optional_str(parsed.get("category"))
    if category is not None:
        grounded = text is None or text.mentions_value(category)
        if product is not None:
            # Product already pins the item (or category is stray) — drop category.
            category = None
        elif not grounded:
            return NoneQuery()

    if family == "qty_sold":
        period = _need_period(parsed.get("period"), as_of, "period")
        base_unit = _optional_str(parsed.get("base_unit"))
        size = _optional_size(parsed.get("size"))
        if product is not None:
            _require_grounded_product(text, product)
        try:
            return QtySoldQuery(
                product=product,
                base_unit=base_unit,
                category=category,
                size=size,
                period=period,
            )
        except ValidationError as exc:
            raise ConvertError(str(exc)) from exc
    if family == "top_products":
        period = _need_period(parsed.get("period"), as_of, "period")
        k = _need_int(parsed.get("k"), "k")
        by = parsed.get("by")
        mode = parsed.get("mode") or "max"
        size = _optional_size(parsed.get("size"))
        if by not in {"qty_base", "revenue", "gross_profit"}:
            raise ConvertError(f"top_products by must be qty_base|revenue|gross_profit, got {by!r}")
        if mode not in {"max", "min"}:
            raise ConvertError(f"top_products mode must be max|min, got {mode!r}")
        return TopProductsQuery(
            k=k, by=by, mode=mode, category=category, size=size, period=period
        )  # type: ignore[arg-type]
    if family == "unsold_products":
        period = _need_period(parsed.get("period"), as_of, "period")
        output = _resolve_count_or_list(parsed.get("output"))
        return UnsoldProductsQuery(
            output=output, category=category, period=period
        )  # type: ignore[arg-type]
    if family == "products_in_min_orders":
        period = _need_period(parsed.get("period"), as_of, "period")
        n = _need_int(parsed.get("n"), "n")
        output = _resolve_count_or_list(parsed.get("output"))
        return ProductsInMinOrdersQuery(
            n=n, output=output, category=category, period=period
        )  # type: ignore[arg-type]
    if family == "price_on_date":
        if product is None:
            raise ConvertError("price_on_date needs product")
        _require_grounded_product(text, product)
        date = _single_day_from(parsed.get("date"), as_of) or _single_day_from(
            parsed.get("period"), as_of
        )
        if date is None:
            raise ConvertError("price_on_date needs a single day in date or period")
        return PriceOnDateQuery(product=product, date=date)
    if family == "price_increase_effect":
        if product is None:
            raise ConvertError("price_increase_effect needs product")
        _require_grounded_product(text, product)
        n_days = _need_int(parsed.get("n_days"), "n_days")
        date = _single_day_from(parsed.get("date"), as_of) or _single_day_from(
            parsed.get("period"), as_of
        )
        return PriceIncreaseEffectQuery(product=product, n_days=n_days, date=date)
    if family == "customers_min_orders":
        period = _need_period(parsed.get("period"), as_of, "period")
        n = _need_int(parsed.get("n"), "n")
        output = _resolve_count_or_share(parsed.get("output"))
        return CustomersMinOrdersQuery(n=n, output=output, period=period)  # type: ignore[arg-type]
    if family == "top_customers":
        period = _need_period(parsed.get("period"), as_of, "period")
        k = _need_int(parsed.get("k"), "k")
        by = parsed.get("by")
        mode = parsed.get("mode") or "max"
        if by not in {"revenue", "order_count"}:
            raise ConvertError(f"top_customers by must be revenue|order_count, got {by!r}")
        if mode not in {"max", "min"}:
            raise ConvertError(f"top_customers mode must be max|min, got {mode!r}")
        return TopCustomersQuery(k=k, by=by, mode=mode, period=period)  # type: ignore[arg-type]
    if family == "orders_min_amount":
        period = _need_period(parsed.get("period"), as_of, "period")
        x = _need_money(parsed.get("x"), "x")
        op = _need_op(parsed.get("op"))
        return OrdersMinAmountQuery(x=x, op=op, period=period)  # type: ignore[arg-type]
    if family == "days_min_revenue":
        period = _need_period(parsed.get("period"), as_of, "period")
        x = _need_money(parsed.get("x"), "x")
        op = _need_op(parsed.get("op"))
        return DaysMinRevenueQuery(x=x, op=op, period=period)  # type: ignore[arg-type]
    raise ConvertError(f"unknown family {family!r}")


def _resolve_count_or_list(raw: object) -> str:
    """Trust the parser's output field; default to count when missing."""
    if raw in {"count", "list"}:
        return str(raw)
    return "count"


def _resolve_count_or_share(raw: object) -> str:
    if raw in {"count", "share"}:
        return str(raw)
    return "count"


def _single_day_from(raw: object, as_of: datetime.date) -> PeriodExpr | None:
    if raw is None:
        return None
    try:
        expr = period_from_raw(raw, as_of)
    except ConvertError:
        return None
    if isinstance(expr, _SINGLE_DAY_TYPES):
        return expr
    return None


def _require_grounded_product(text: QuestionText | None, product: str) -> None:
    if text is None:
        return
    if not text.mentions_product(product):
        raise ConvertError(f"product {product!r} is not grounded in the question")


def _need_period(raw: object, as_of: datetime.date, label: str) -> PeriodExpr:
    if raw is None:
        raise ConvertError(f"{label} is required")
    return period_from_raw(raw, as_of)


def _need_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConvertError(f"{field} must be an integer")
    if value < 1:
        raise ConvertError(f"{field} must be >= 1")
    return value


def _need_money(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConvertError(f"{field} must be an integer (VND)")
    if value < 0:
        raise ConvertError(f"{field} must be >= 0")
    return value


def _need_op(value: object) -> str:
    if value not in {">=", ">"}:
        raise ConvertError(f"op must be >= or >, got {value!r}")
    return str(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConvertError("expected a string")
    text = value.strip()
    return text or None


def _optional_size(value: object) -> str | None:
    if value is None:
        return None
    if value not in {"M", "L", "m", "l"}:
        raise ConvertError(f"size must be M or L, got {value!r}")
    return str(value).upper()

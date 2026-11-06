"""Turn flat extension-parser JSON into an ExtQuerySpec."""

from __future__ import annotations

import datetime
from collections.abc import Callable
from typing import Any

from pydantic import TypeAdapter, ValidationError

from datamate.parse.text import ingredient_parts
from datamate.query.convert import ConvertError, period_from_raw
from datamate.query.ext.spec import (
    AgingBucket,
    AvgPurchaseCostQuery,
    CustomerDebtFlowQuery,
    CustomersMinOrdersQuery,
    DaysMinRevenueQuery,
    DebtAddedQuery,
    DebtAgingQuery,
    DebtCollectedQuery,
    DebtTotalQuery,
    DeliveryCountQuery,
    EmployeeRevenueQuery,
    ExtQuerySpec,
    GrossProfitQuery,
    IngredientRevenueQuery,
    LotsExpiringQuery,
    NearestExpiryLotQuery,
    NetProfitQuery,
    NoneQuery,
    OrdersMinAmountQuery,
    OrdersMinDeliveriesQuery,
    OtcRevenueQuery,
    OutOfStockCountQuery,
    PriceIncreaseEffectQuery,
    PriceOnDateQuery,
    ProductsInMinOrdersQuery,
    PurchaseTotalQuery,
    QtySoldQuery,
    RxRevenueQuery,
    RxShareQuery,
    StockOnHandQuery,
    StockValueQuery,
    TopCustomersQuery,
    TopDebtorsQuery,
    TopEmployeeQuery,
    TopProductsQuery,
    TopSupplierQuery,
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
    same_product: Callable[[str], bool] | None = None,
) -> ExtQuerySpec:
    text = QuestionText(question, same_product) if question is not None else None
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
    if family == "debt_total":
        date = _need_as_of_day(parsed, as_of)
        output = "customer_count" if parsed.get("output") == "count" else "amount"
        return DebtTotalQuery(date=date, output=output)  # type: ignore[arg-type]
    if family == "top_debtors":
        date = _need_as_of_day(parsed, as_of)
        k = _need_int(parsed.get("k"), "k")
        return TopDebtorsQuery(k=k, date=date)
    if family == "debt_aging":
        date = _need_as_of_day(parsed, as_of)
        buckets = _need_buckets(parsed.get("buckets"))
        return DebtAgingQuery(date=date, buckets=buckets)
    if family == "debt_added":
        period = _need_period(parsed.get("period"), as_of, "period")
        return DebtAddedQuery(period=period)
    if family == "debt_collected":
        period = _need_period(parsed.get("period"), as_of, "period")
        return DebtCollectedQuery(period=period)
    if family == "purchase_total":
        period = _need_period(parsed.get("period"), as_of, "period")
        return PurchaseTotalQuery(period=period)
    if family == "top_supplier":
        period = _need_period(parsed.get("period"), as_of, "period")
        return TopSupplierQuery(period=period)
    if family == "avg_purchase_cost":
        if product is None:
            raise ConvertError("avg_purchase_cost needs product")
        _require_grounded_product(text, product)
        period = _need_period(parsed.get("period"), as_of, "period")
        return AvgPurchaseCostQuery(product=product, period=period)
    if family == "stock_on_hand":
        if product is None:
            raise ConvertError("stock_on_hand needs product")
        _require_grounded_product(text, product)
        date = _need_as_of_day(parsed, as_of)
        unit = _optional_str(parsed.get("unit")) or _optional_str(parsed.get("base_unit"))
        return StockOnHandQuery(product=product, date=date, unit=unit)
    if family == "stock_value":
        date = _need_as_of_day(parsed, as_of)
        return StockValueQuery(date=date)
    if family == "out_of_stock_count":
        date = _need_as_of_day(parsed, as_of)
        return OutOfStockCountQuery(date=date)
    if family == "lots_expiring":
        date = _need_as_of_day(parsed, as_of)
        n_days = _need_int_ge0(parsed.get("n_days"), "n_days")
        output = _resolve_count_list_or_cost(parsed.get("output"))
        return LotsExpiringQuery(date=date, n_days=n_days, output=output)  # type: ignore[arg-type]
    if family == "nearest_expiry_lot":
        if product is None:
            raise ConvertError("nearest_expiry_lot needs product")
        _require_grounded_product(text, product)
        date = _single_day_from(parsed.get("date"), as_of) or _single_day_from(
            parsed.get("period"), as_of
        )
        return NearestExpiryLotQuery(product=product, date=date)
    if family == "ingredient_revenue":
        ingredient = _optional_str(parsed.get("active_ingredient"))
        if ingredient is None:
            raise ConvertError("ingredient_revenue needs active_ingredient")
        period = _need_period(parsed.get("period"), as_of, "period")
        # One spelling per combination, so two votes that wrote it differently agree.
        parts = ingredient_parts(ingredient)
        if len(parts) > 1:
            ingredient = " + ".join(parts)
        return IngredientRevenueQuery(active_ingredient=ingredient, period=period)
    if family == "rx_share":
        period = _need_period(parsed.get("period"), as_of, "period")
        return RxShareQuery(period=period)
    if family == "rx_revenue":
        period = _need_period(parsed.get("period"), as_of, "period")
        return RxRevenueQuery(period=period)
    if family == "otc_revenue":
        period = _need_period(parsed.get("period"), as_of, "period")
        return OtcRevenueQuery(period=period)
    if family == "gross_profit":
        period = _need_period(parsed.get("period"), as_of, "period")
        return GrossProfitQuery(period=period)
    if family == "net_profit":
        period = _need_period(parsed.get("period"), as_of, "period")
        return NetProfitQuery(period=period)
    if family == "top_employee":
        period = _need_period(parsed.get("period"), as_of, "period")
        return TopEmployeeQuery(period=period)
    if family == "customer_debt_flow":
        customer = _optional_str(parsed.get("customer"))
        if customer is None:
            raise ConvertError("customer_debt_flow needs customer")
        if text is not None and not text.mentions_value(customer):
            raise ConvertError(f"customer {customer!r} is not grounded in the question")
        period = _need_period(parsed.get("period"), as_of, "period")
        return CustomerDebtFlowQuery(customer=customer, period=period)
    if family == "employee_revenue":
        employee = _optional_str(parsed.get("employee"))
        if employee is None:
            raise ConvertError("employee_revenue needs employee")
        if text is not None and not text.mentions_value(employee):
            raise ConvertError(f"employee {employee!r} is not grounded in the question")
        period = _need_period(parsed.get("period"), as_of, "period")
        return EmployeeRevenueQuery(employee=employee, period=period)
    if family == "delivery_count":
        period = _need_period(parsed.get("period"), as_of, "period")
        return DeliveryCountQuery(period=period)
    if family == "orders_min_deliveries":
        period = _need_period(parsed.get("period"), as_of, "period")
        n = _need_int(parsed.get("n"), "n")
        return OrdersMinDeliveriesQuery(n=n, period=period)
    raise ConvertError(f"unknown family {family!r}")


def _need_as_of_day(parsed: dict[str, Any], as_of: datetime.date) -> PeriodExpr:
    date = _single_day_from(parsed.get("date"), as_of) or _single_day_from(
        parsed.get("period"), as_of
    )
    if date is None:
        raise ConvertError("debt family needs a single day in date (as-of)")
    return date


def _need_buckets(raw: object) -> list[AgingBucket]:
    if not isinstance(raw, list) or not raw:
        raise ConvertError("debt_aging needs a non-empty buckets list")
    out: list[AgingBucket] = []
    for item in raw:
        if isinstance(item, dict):
            lo = item.get("lo")
            hi = item.get("hi")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            lo, hi = item[0], item[1]
        else:
            raise ConvertError(f"bucket must be [lo, hi], got {item!r}")
        if isinstance(lo, bool) or not isinstance(lo, int) or lo < 0:
            raise ConvertError(f"bucket lo must be int >= 0, got {lo!r}")
        if hi is not None and (isinstance(hi, bool) or not isinstance(hi, int) or hi < lo):
            raise ConvertError(f"bucket hi must be null or int >= lo, got {hi!r}")
        out.append(AgingBucket(lo=lo, hi=hi))
    return out


def _resolve_count_or_list(raw: object) -> str:
    """Trust the parser's output field; default to count when missing."""
    if raw in {"count", "list"}:
        return str(raw)
    return "count"


def _resolve_count_or_share(raw: object) -> str:
    if raw in {"count", "share"}:
        return str(raw)
    return "count"


def _resolve_count_list_or_cost(raw: object) -> str:
    if raw in {"count", "list", "cost"}:
        return str(raw)
    return "count"


def _need_int_ge0(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConvertError(f"{field} must be an integer")
    if value < 0:
        raise ConvertError(f"{field} must be >= 0")
    return value


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
    from datamate.query.sizes import size_label

    label = size_label(value)
    if label is None:
        raise ConvertError(f"size must be a size label, got {value!r}")
    return label

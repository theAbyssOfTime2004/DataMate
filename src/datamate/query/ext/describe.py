"""Deterministic English description of an extension query spec (for the verifier)."""

from __future__ import annotations

import datetime

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
from datamate.rules.model import RuleConfig
from datamate.timeres.calendar import CalendarError, LunarCalendar
from datamate.timeres.periods import PeriodError, PeriodExpr, resolve


def describe_ext_spec(
    spec: ExtQuerySpec,
    as_of: datetime.date,
    rules: RuleConfig,
    cal: LunarCalendar,
    product_unit: str | None = None,
    customer_who: str | None = None,
) -> str:
    """One line per facet: what the spec computes and what it does not restrict.

    `product_unit` is the named product's base unit from the shop data, when known.
    `customer_who` states how THIS shop identifies customers (from shop columns).
    """
    if isinstance(spec, NoneQuery):
        return "family: none (no product metric)"
    if isinstance(spec, QtySoldQuery):
        return _lines(
            _measure_qty(spec.base_unit if spec.product is None else product_unit),
            _products_line(product=spec.product, category=spec.category, base_unit=spec.base_unit),
            _size_line(spec.size),
            _common_unrestricted(),
            _period_line(spec.period, as_of, rules, cal),
            "output: one number (units sold)",
        )
    if isinstance(spec, TopProductsQuery):
        measure = _top_measure(spec.by)
        return _lines(
            f"measure: {measure}",
            _products_line(product=None, category=spec.category, base_unit=None),
            _size_line(spec.size),
            _common_unrestricted(),
            _period_line(spec.period, as_of, rules, cal),
            _top_output(spec.k, measure, spec.mode),
        )
    if isinstance(spec, UnsoldProductsQuery):
        return _lines(
            "measure: products the shop still sells (sold before the period) with no sale inside the period",
            _products_line(product=None, category=spec.category, base_unit=None),
            _size_line(None),
            _common_unrestricted(),
            _period_line(spec.period, as_of, rules, cal),
            f"output: {'count of names' if spec.output == 'count' else 'list of names'}",
        )
    if isinstance(spec, ProductsInMinOrdersQuery):
        return _lines(
            f"measure: products present in at least {spec.n} distinct revenue orders",
            _products_line(product=None, category=spec.category, base_unit=None),
            _size_line(None),
            _common_unrestricted(),
            _period_line(spec.period, as_of, rules, cal),
            f"output: {'count of names' if spec.output == 'count' else 'list of names'}",
        )
    if isinstance(spec, PriceOnDateQuery):
        return _lines(
            _price_measure(product_unit),
            _products_line(product=spec.product, category=None, base_unit=None),
            _size_line(None),
            _common_unrestricted(),
            f"day: {_period_span(spec.date, as_of, rules, cal)}",
            "output: one number (VND)",
        )
    if isinstance(spec, PriceIncreaseEffectQuery):
        day = (
            _period_span(spec.date, as_of, rules, cal)
            if spec.date is not None
            else "first day the selling price rises vs the previous price"
        )
        return _lines(
            f"measure: percent change in units sold over {spec.n_days} days after a price "
            f"increase vs the {spec.n_days} days before",
            _products_line(product=spec.product, category=None, base_unit=None),
            _size_line(None),
            _common_unrestricted(),
            f"increase day: {day}",
            "output: one percent",
        )
    who = customer_who or "customers"
    if isinstance(spec, CustomersMinOrdersQuery):
        out = (
            "output: count of customers"
            if spec.output == "count"
            else "output: percent = those customers / all customers who bought in the period x 100"
        )
        return _lines(
            f"measure: customers with at least {spec.n} revenue orders in the period",
            f"customers: {who}",
            _customer_unrestricted(),
            _period_line(spec.period, as_of, rules, cal),
            out,
        )
    if isinstance(spec, TopCustomersQuery):
        measure = "money revenue" if spec.by == "revenue" else "order count"
        return _lines(
            f"measure: {measure}",
            f"customers: {who}",
            _customer_unrestricted(),
            _period_line(spec.period, as_of, rules, cal),
            _top_customer_output(spec.k, measure, spec.mode),
        )
    if isinstance(spec, OrdersMinAmountQuery):
        return _lines(
            f"measure: count of revenue orders whose order revenue {spec.op} {spec.x} VND",
            _threshold_unrestricted(),
            _period_line(spec.period, as_of, rules, cal),
            "output: one number (order count)",
        )
    if isinstance(spec, DaysMinRevenueQuery):
        return _lines(
            f"measure: count of days whose revenue {spec.op} {spec.x} VND "
            "(days with no orders never qualify)",
            _threshold_unrestricted(),
            _period_line(spec.period, as_of, rules, cal),
            "output: one number (day count)",
        )
    raise TypeError(f"unsupported ext spec {spec!r}")


def _lines(*parts: str) -> str:
    return "\n".join(parts)


def _measure_qty(base_unit: str | None) -> str:
    if base_unit:
        return f'measure: quantity sold, counted in "{base_unit}"'
    return "measure: quantity sold, counted in the product's own unit"


def _price_measure(unit: str | None) -> str:
    if unit:
        return f'measure: selling price of one "{unit}" (VND, before line discount)'
    return "measure: selling price of one unit of the product (VND, before line discount)"


def _top_measure(by: str) -> str:
    if by == "qty_base":
        return "units sold"
    if by == "revenue":
        return "money revenue"
    return "gross profit"


def _top_output(k: int, measure: str, mode: str) -> str:
    order = "highest" if mode == "max" else "lowest"
    if k == 1:
        return f"output: one name ({order} by {measure}, tie by code)"
    return f"output: top {k} names ordered by {measure} ({order} first), tie by code"


def _top_customer_output(k: int, measure: str, mode: str) -> str:
    order = "highest" if mode == "max" else "lowest"
    if k == 1:
        return f"output: one name ({order} by {measure}, tie by code or name)"
    return (
        f"output: top {k} names ordered by {measure} ({order} first), "
        "tie by code or name"
    )


def _products_line(
    *, product: str | None, category: str | None, base_unit: str | None
) -> str:
    if product is not None:
        return f'products: "{product}"'
    if category is not None and base_unit is not None:
        return f'products: category "{category}", only items sold in "{base_unit}"'
    if category is not None:
        return f'products: category "{category}"'
    if base_unit is not None:
        return f'products: all products sold in "{base_unit}"'
    return "products: all products"


def _size_line(size: str | None) -> str:
    if size is None:
        return "size: none"
    return f'size: "{size}" only (names ending with ({size}))'


def _common_unrestricted() -> str:
    return "not filtered by: delivery status, customer, payment method, channel"


def _customer_unrestricted() -> str:
    return "not filtered by: product, channel, payment method, delivery status"


def _threshold_unrestricted() -> str:
    return "not filtered by: product, customer, channel, payment method, delivery status"


def _period_line(
    expr: PeriodExpr, as_of: datetime.date, rules: RuleConfig, cal: LunarCalendar
) -> str:
    return f"period: {_period_span(expr, as_of, rules, cal)} (order date)"


def _period_span(
    expr: PeriodExpr, as_of: datetime.date, rules: RuleConfig, cal: LunarCalendar
) -> str:
    try:
        window = resolve(expr, as_of, rules, cal)
    except (PeriodError, CalendarError, ValueError):
        return f"unresolved {expr!r}"
    if window.start == window.end:
        return window.start.isoformat()
    return f"{window.start.isoformat()} .. {window.end.isoformat()}"

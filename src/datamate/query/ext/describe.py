"""Deterministic English description of an extension query spec (for the verifier)."""

from __future__ import annotations

import datetime

from datamate.parse.text import ingredient_parts
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
    EmployeeRevenueQuery,
    ExtQuerySpec,
    GrossProfitQuery,
    IngredientRevenueQuery,
    LotsExpiringQuery,
    NearestExpiryLotQuery,
    NetProfitQuery,
    NoneQuery,
    OrdersMinAmountQuery,
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
    debt_meta: dict[str, str] | None = None,
) -> str:
    """One line per facet: what the spec computes and what it does not restrict.

    `product_unit` is the named product's base unit from the shop data, when known.
    `customer_who` states how THIS shop identifies customers (from shop columns).
    `debt_meta` carries shop payment_source / exclusions / customer_who for debt families.
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
    meta = debt_meta or {}
    pay_src = meta.get("payment_source", "payments")
    excl = meta.get("exclusions", "none found")
    debt_who = meta.get("customer_who") or who
    if isinstance(spec, DebtTotalQuery):
        if spec.output == "customer_count":
            return _lines(
                "measure: number of customers whose remaining debt is above 0",
                f"as-of: end of {_period_span(spec.date, as_of, rules, cal)}",
                f"payments from: {pay_src}",
                f"excluded orders: cancelled; COD/marketplace labels in this shop: {excl}",
                f"customers: {debt_who} (walk-in orders are not counted as customers)",
                "output: one number (customer count)",
            )
        return _lines(
            "measure: sum of remaining debt (payable minus paid) over debt orders",
            f"as-of: end of {_period_span(spec.date, as_of, rules, cal)}",
            f"payments from: {pay_src}",
            f"excluded orders: cancelled; COD/marketplace labels in this shop: {excl}",
            f"customers: {debt_who} (walk-in remaining still counts in the total)",
            "output: one number (VND)",
        )
    if isinstance(spec, TopDebtorsQuery):
        return _lines(
            "measure: remaining debt per customer",
            f"as-of: end of {_period_span(spec.date, as_of, rules, cal)}",
            f"payments from: {pay_src}",
            f"excluded orders: cancelled; COD/marketplace labels in this shop: {excl}",
            f"customers: {debt_who} (walk-in orders omitted from the table)",
            f"output: table [customer, amount] top {spec.k} by amount",
        )
    if isinstance(spec, DebtAgingQuery):
        return _lines(
            "measure: remaining debt by age in days (as-of minus order date)",
            f"as-of: end of {_period_span(spec.date, as_of, rules, cal)}",
            f"payments from: {pay_src}",
            f"excluded orders: cancelled; COD/marketplace labels in this shop: {excl}",
            f"buckets: {_format_buckets(spec.buckets)}",
            "output: list of amounts in bucket order (VND)",
        )
    if isinstance(spec, DebtAddedQuery):
        return _lines(
            "measure: unpaid part on the order day of debt orders placed in the period",
            f"payments from: {pay_src}",
            f"excluded orders: cancelled; COD/marketplace labels in this shop: {excl}",
            _period_line(spec.period, as_of, rules, cal),
            "output: one number (VND)",
        )
    if isinstance(spec, DebtCollectedQuery):
        return _lines(
            "measure: payments after the order day, dated in the period, on debt orders",
            f"payments from: {pay_src}",
            f"excluded orders: cancelled; COD/marketplace labels in this shop: {excl}",
            _period_line(spec.period, as_of, rules, cal),
            "output: one number (VND)",
        )
    if isinstance(spec, CustomerDebtFlowQuery):
        return _lines(
            "measure: for one customer, [unpaid part on the order day of debt orders placed "
            "in the period, payments after the order day dated in the period]",
            f'customer: "{spec.customer}" (name or code)',
            f"payments from: {pay_src}",
            f"excluded orders: cancelled; COD/marketplace labels in this shop: {excl}",
            _period_line(spec.period, as_of, rules, cal),
            "output: two numbers (VND), debt added first",
        )
    if isinstance(spec, PurchaseTotalQuery):
        return _lines(
            "measure: money the shop paid suppliers for goods bought in (sum of purchase receipt "
            "totals); not sales, not deliveries to customers, not a count",
            "source: purchases table (cancelled receipts excluded)",
            f"period: {_period_span(spec.period, as_of, rules, cal)} (receipt date)",
            "output: one number (VND)",
        )
    if isinstance(spec, TopSupplierQuery):
        return _lines(
            "measure: supplier with the highest purchase receipt total",
            "source: purchases table (cancelled receipts excluded)",
            f"period: {_period_span(spec.period, as_of, rules, cal)} (receipt date)",
            "output: one supplier name",
        )
    if isinstance(spec, AvgPurchaseCostQuery):
        return _lines(
            "measure: quantity-weighted average purchase unit cost",
            f'product: "{spec.product}"',
            "source: purchase_lines (cancelled receipts excluded)",
            f"period: {_period_span(spec.period, as_of, rules, cal)} (receipt date)",
            "output: one number (VND per purchase unit)",
        )
    if isinstance(spec, StockOnHandQuery):
        unit_line = (
            f'unit: "{spec.unit}" (big unit -> [whole, leftover base units])'
            if spec.unit
            else f'unit: base unit{f" ({product_unit})" if product_unit else ""}'
        )
        return _lines(
            "measure: on-hand stock at end of day",
            f'product: "{spec.product}"',
            unit_line,
            "source: stock_moves balance, else stock_snapshot on that date",
            f"as-of: end of {_period_span(spec.date, as_of, rules, cal)}",
            "output: one number (base) or list [whole big units, leftover base]",
        )
    if isinstance(spec, StockValueQuery):
        return _lines(
            "measure: sum of on-hand base qty * products.unit_cost for stock > 0",
            "source: stock_moves/snapshot and products.unit_cost",
            f"as-of: end of {_period_span(spec.date, as_of, rules, cal)}",
            "output: one number (VND)",
        )
    if isinstance(spec, OutOfStockCountQuery):
        return _lines(
            "measure: count of resale products (seen in purchase_lines) with end-of-day stock 0",
            "source: purchase_lines + stock_moves/snapshot",
            f"day: {_period_span(spec.date, as_of, rules, cal)}",
            "output: one number (product count)",
        )
    if isinstance(spec, LotsExpiringQuery):
        out = {
            "count": "count of lots",
            "list": "list of product names",
            "cost": "one number (VND cost of remaining qty)",
        }[spec.output]
        return _lines(
            f"measure: lots still in stock expiring within {spec.n_days} days",
            "source: lots minus sale_lots up to as-of; cost uses each lot's unit_cost",
            f"as-of: end of {_period_span(spec.date, as_of, rules, cal)}",
            f"output: {out}",
        )
    if isinstance(spec, NearestExpiryLotQuery):
        day = (
            _period_span(spec.date, as_of, rules, cal)
            if spec.date is not None
            else as_of.isoformat()
        )
        return _lines(
            "measure: earliest-expiring lot still in stock and not yet expired",
            f'product: "{spec.product}"',
            "source: lots minus sale_lots up to as-of",
            f"as-of: end of {day}",
            "output: one lot id",
        )
    if isinstance(spec, IngredientRevenueQuery):
        if len(ingredient_parts(spec.active_ingredient)) > 1:
            measure = "measure: revenue of drugs with exactly this combination of ingredients"
            source = "source: order_lines revenue + drug_info.active_ingredient"
        else:
            measure = "measure: revenue of products with this exact active ingredient"
            source = "source: order_lines revenue + drug_info (combined ingredients are separate)"
        return _lines(
            measure,
            f'active_ingredient: "{spec.active_ingredient}"',
            source,
            _period_line(spec.period, as_of, rules, cal),
            "output: one number (VND)",
        )
    if isinstance(spec, RxShareQuery):
        return _lines(
            "measure: prescription (Rx) revenue / total revenue x 100",
            "source: order_lines revenue + drug_info.is_prescription",
            _period_line(spec.period, as_of, rules, cal),
            "output: one percent",
        )
    if isinstance(spec, RxRevenueQuery):
        return _lines(
            "measure: revenue of prescription (Rx) drugs only",
            "source: order_lines revenue + drug_info.is_prescription",
            _period_line(spec.period, as_of, rules, cal),
            "output: one number (VND)",
        )
    if isinstance(spec, OtcRevenueQuery):
        return _lines(
            "measure: revenue of every item that is not a prescription (Rx) drug",
            "source: order_lines revenue + drug_info.is_prescription",
            _period_line(spec.period, as_of, rules, cal),
            "output: one number (VND)",
        )
    if isinstance(spec, GrossProfitQuery):
        return _lines(
            "measure: revenue minus qty * cost_per_unit on revenue lines",
            "source: order_lines (cost_per_unit on the sold unit)",
            _period_line(spec.period, as_of, rules, cal),
            "output: one number (VND)",
        )
    if isinstance(spec, NetProfitQuery):
        return _lines(
            "measure: gross profit minus all expenses dated in the period",
            "source: order_lines cost_per_unit + expenses table",
            _period_line(spec.period, as_of, rules, cal),
            "output: one number (VND)",
        )
    if isinstance(spec, TopEmployeeQuery):
        return _lines(
            "measure: employee with the highest revenue",
            "source: orders.employee_id + order_lines revenue",
            _period_line(spec.period, as_of, rules, cal),
            "output: one employee name",
        )
    if isinstance(spec, EmployeeRevenueQuery):
        return _lines(
            "measure: revenue of the orders sold by one employee",
            f'employee: "{spec.employee}" (code or name)',
            "source: orders.employee_id + order_lines revenue",
            _period_line(spec.period, as_of, rules, cal),
            "output: one number (VND)",
        )
    raise TypeError(f"unsupported ext spec {spec!r}")


def _format_buckets(buckets: list[AgingBucket]) -> str:
    parts: list[str] = []
    for b in buckets:
        if b.hi is None:
            parts.append(f"{b.lo}+")
        else:
            parts.append(f"{b.lo}-{b.hi}")
    return ", ".join(parts)


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

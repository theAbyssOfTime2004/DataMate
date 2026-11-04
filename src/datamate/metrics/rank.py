"""Rank block computations: who or what comes first by one measure in a period.

Combinations an existing family already answers go to that family, so both paths give
the same number; the others are computed here from orders or purchase receipts.
"""

from __future__ import annotations

import datetime
from collections import defaultdict
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from datamate.metrics.ext_families import (
    _Abstain,
    _apply_category_filter,
    _detect_customer_mode,
    _display_name,
    _ext_recorded,
    _fold_customer_name,
    _non_product_sql,
    _rank_product_rows,
    _revenue_order_predicates,
    _top_customers,
    _top_products,
)
from datamate.query.ext.spec import TopCustomersQuery, TopProductsQuery
from datamate.query.rank import RankQuery
from datamate.query.spec import ArgMaxQuery
from datamate.scoring.answer import Answer
from datamate.timeres.periods import DateRange, PeriodError

if TYPE_CHECKING:
    from datamate.metrics.engine import MetricEngine


def answer_rank(engine: MetricEngine, spec: RankQuery, as_of: datetime.date) -> Answer:
    from datamate.metrics.engine import Unanswerable

    try:
        delegated = _delegate(engine, spec, as_of)
        if delegated is not None:
            return delegated
        cov, window = engine._window(spec.period, as_of)
        if cov == "none":
            return Answer.unanswerable("period lies entirely outside the shop's data")
        if spec.entity == "supplier":
            return _rank_suppliers(engine, spec, window)
        if spec.entity == "product":
            return _rank_products_by_orders(engine, spec, window)
        return _rank_by_orders(engine, spec, window)
    except Unanswerable as exc:
        return Answer.unanswerable(str(exc))
    except (_Abstain, PeriodError):
        return Answer.abstain()


def _delegate(engine: MetricEngine, spec: RankQuery, as_of: datetime.date) -> Answer | None:
    """Answer through the existing family when it covers the same combination."""
    if spec.orders != "sold" or spec.min_orders is not None:
        return None
    if spec.entity == "customer" and spec.measure in {"revenue", "order_count"}:
        query = TopCustomersQuery(k=spec.k, by=spec.measure, mode=spec.mode, period=spec.period)
        return _top_customers(engine, query, as_of)
    if spec.entity == "product" and spec.measure in {"qty_base", "revenue", "gross_profit"}:
        products = TopProductsQuery(
            k=spec.k, by=spec.measure, mode=spec.mode, category=spec.category, period=spec.period
        )
        return _top_products(engine, products, as_of)
    if spec.entity == "day" and spec.k == 1:
        argmax = ArgMaxQuery(
            metric=spec.measure, period=spec.period, granularity="day", mode=spec.mode
        )
        return engine.answer(argmax, as_of)
    return None


def _order_where(engine: MetricEngine, spec: RankQuery, window: DateRange) -> tuple[str, list[Any]]:
    if spec.orders == "cancelled":
        where, params = "o.status = ?", ["cancelled"]
    else:
        where, params = _revenue_order_predicates(engine)
    return f"{where} AND o.order_date BETWEEN ? AND ?", [*params, window.start, window.end]


def _rank_by_orders(engine: MetricEngine, spec: RankQuery, window: DateRange) -> Answer:
    """Customers, staff, days and hours: group the period's orders, then rank."""
    from datamate.metrics.engine import Unanswerable

    if spec.entity == "employee" and not _ext_recorded(engine, "orders", "employee_id"):
        raise Unanswerable("orders.employee_id")
    if spec.entity == "hour" and not _ext_recorded(engine, "orders", "order_ts"):
        raise Unanswerable("orders.order_ts")
    mode = _detect_customer_mode(engine) if spec.entity == "customer" else None
    if spec.entity == "customer" and mode is None:
        raise Unanswerable("customer")
    if spec.entity in {"day", "hour"} and spec.mode == "min":
        # A day or hour without orders is a real minimum the order rows cannot show.
        raise _Abstain()
    where, params = _order_where(engine, spec, window)
    rows = engine.con.execute(
        f"""
        SELECT o.order_id, o.order_date, o.order_ts, o.customer_id, o.customer_name,
               COALESCE(o.is_walk_in, FALSE), o.employee_id,
               COALESCE(SUM(l.line_amount), 0)
        FROM orders o LEFT JOIN order_lines l ON l.order_id = o.order_id
        WHERE {where}
        GROUP BY o.order_id, o.order_date, o.order_ts, o.customer_id, o.customer_name,
                 o.is_walk_in, o.employee_id
        """,
        params,
    ).fetchall()
    staff_names = _staff_names(engine) if spec.entity == "employee" else {}
    orders: dict[str, int] = defaultdict(int)
    money: dict[str, int] = defaultdict(int)
    days: dict[str, set[datetime.date]] = defaultdict(set)
    labels: dict[str, str] = {}
    for _oid, day, ts, cust_id, cust_name, walk_in, emp_id, amount in rows:
        key, label = _entity_key(spec.entity, mode, day, ts, cust_id, cust_name, walk_in, emp_id)
        if key is None:
            if spec.entity == "hour":
                raise _Abstain()  # an order without a time could change the peak
            continue
        orders[key] += 1
        money[key] += int(amount)
        days[key].add(day)
        if spec.entity == "employee":
            label = staff_names.get(key) or label
        labels[key] = max(labels.get(key, label), label)
    measured: list[tuple[str, Decimal]] = []
    for key, n in orders.items():
        if spec.min_orders is not None and n < spec.min_orders:
            continue
        if spec.measure == "order_count":
            value = Decimal(n)
        elif spec.measure == "revenue":
            value = Decimal(money[key])
        elif spec.measure == "active_days":
            value = Decimal(len(days[key]))
        else:
            value = Decimal(money[key]) / Decimal(n)
        measured.append((key, value))
    if not measured:
        raise _Abstain()
    sign = -1 if spec.mode == "max" else 1
    ranked = sorted(measured, key=lambda item: (sign * item[1], item[0]))[: spec.k]
    if spec.entity == "hour":
        hours = [int(key) for key, _value in ranked]
        if spec.k == 1:
            return Answer(type="number", answer=hours[0])
        return Answer(type="list", answer=hours)
    if spec.entity == "day":
        if spec.k == 1:
            return Answer(type="date", answer=ranked[0][0])
        return Answer(type="list", answer=[key for key, _value in ranked])
    names = [labels[key] for key, _value in ranked]
    if spec.k == 1:
        return Answer(type="string", answer=names[0])
    return Answer(type="list", answer=names)


def _entity_key(
    entity: str,
    customer_mode: str | None,
    day: datetime.date,
    ts: datetime.datetime | None,
    cust_id: Any,
    cust_name: Any,
    walk_in: bool,
    emp_id: Any,
) -> tuple[str | None, str]:
    """Grouping key (sorts for ties: code, folded name, date, hour) and display label."""
    if entity == "customer":
        if walk_in:
            return None, ""
        if customer_mode == "by_code":
            if cust_id is None:
                return None, ""
            name = str(cust_name).strip() if cust_name is not None else ""
            return str(cust_id), name or str(cust_id)
        ref = _fold_customer_name(cust_name if isinstance(cust_name, str) else None)
        return ref, str(cust_name).strip() if ref is not None else ""
    if entity == "employee":
        if emp_id is None:
            return None, ""
        return str(emp_id), str(emp_id)
    if entity == "day":
        return day.isoformat(), day.isoformat()
    if ts is None:
        return None, ""
    return f"{ts.hour:02d}", str(ts.hour)


def _staff_names(engine: MetricEngine) -> dict[str, str]:
    try:
        rows = engine.con.execute(
            "SELECT employee_id, MAX(employee_name) FROM employees "
            "WHERE employee_id IS NOT NULL GROUP BY employee_id"
        ).fetchall()
    except Exception:  # noqa: BLE001 — no staff list => codes are the names
        return {}
    return {str(emp): str(name) for emp, name in rows if name is not None and str(name).strip()}


def _rank_products_by_orders(engine: MetricEngine, spec: RankQuery, window: DateRange) -> Answer:
    """Products by the number of orders that contain them."""
    where, params = _order_where(engine, spec, window)
    where += f" AND {_non_product_sql(engine)}"
    where, params = _apply_category_filter(engine, spec.category, where, params)
    rows = engine.con.execute(
        "SELECT l.product_key, MAX(l.product_id), MAX(l.product_name), "
        "COUNT(DISTINCT o.order_id) "
        "FROM order_lines l JOIN orders o ON o.order_id = l.order_id "
        f"WHERE {where} GROUP BY l.product_key",
        params,
    ).fetchall()
    if not rows:
        raise _Abstain()
    chosen = _rank_product_rows(rows, mode=spec.mode)[: spec.k]
    if spec.k == 1:
        return Answer(type="string", answer=_display_name(chosen[0]))
    return Answer(type="list", answer=[_display_name(row) for row in chosen])


def _rank_suppliers(engine: MetricEngine, spec: RankQuery, window: DateRange) -> Answer:
    from datamate.metrics.engine import Unanswerable

    if spec.measure == "purchase_amount":
        if not _ext_recorded(engine, "purchases", "total_amount"):
            raise Unanswerable("purchases.total_amount")
        measure = "COALESCE(SUM(p.total_amount), 0)"
    else:
        if not _ext_recorded(engine, "purchases", "receipt_id"):
            raise Unanswerable("purchases.receipt_id")
        measure = "COUNT(DISTINCT p.receipt_id)"
    order = "DESC" if spec.mode == "max" else "ASC"
    rows = engine.con.execute(
        f"""
        SELECT COALESCE(p.supplier_name, p.supplier_id) AS name, {measure} AS amount
        FROM purchases p
        WHERE p.receipt_date BETWEEN ? AND ?
          AND NOT COALESCE(p.is_cancelled, FALSE)
          AND (p.supplier_name IS NOT NULL OR p.supplier_id IS NOT NULL)
        GROUP BY COALESCE(p.supplier_name, p.supplier_id)
        ORDER BY amount {order}, name ASC
        LIMIT ?
        """,
        [window.start, window.end, spec.k],
    ).fetchall()
    if not rows:
        raise _Abstain()
    names = [str(row[0]) for row in rows]
    if spec.k == 1:
        return Answer(type="string", answer=names[0])
    return Answer(type="list", answer=names)

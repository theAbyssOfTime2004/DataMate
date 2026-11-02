"""E4a/E4b family computations on MetricEngine (SQL over canonical tables)."""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from datamate.metrics.format import render_metric, render_percent
from datamate.parse.text import normalize_label
from datamate.query.ext.spec import (
    CustomerDebtFlowQuery,
    CustomersMinOrdersQuery,
    DaysMinRevenueQuery,
    DebtAddedQuery,
    DebtAgingQuery,
    DebtCollectedQuery,
    DebtTotalQuery,
    ExtQuerySpec,
    NoneQuery,
    OrdersMinAmountQuery,
    PriceIncreaseEffectQuery,
    PriceOnDateQuery,
    ProductsInMinOrdersQuery,
    QtySoldQuery,
    TopCustomersQuery,
    TopDebtorsQuery,
    TopProductsQuery,
    UnsoldProductsQuery,
)
from datamate.query.grounding import fold as fold_text
from datamate.scoring.answer import Answer
from datamate.timeres.periods import DateRange, PeriodError

if TYPE_CHECKING:
    from datamate.metrics.engine import MetricEngine


class _Abstain(Exception):
    """Per-question gap: prefer score 0 over a wrong number."""


def answer_ext_family(engine: MetricEngine, spec: ExtQuerySpec, as_of: datetime.date) -> Answer:
    from datamate.metrics.engine import Unanswerable

    try:
        if isinstance(spec, NoneQuery):
            return Answer.abstain()
        if isinstance(spec, QtySoldQuery):
            return _qty_sold(engine, spec, as_of)
        if isinstance(spec, TopProductsQuery):
            return _top_products(engine, spec, as_of)
        if isinstance(spec, UnsoldProductsQuery):
            return _unsold_products(engine, spec, as_of)
        if isinstance(spec, ProductsInMinOrdersQuery):
            return _products_in_min_orders(engine, spec, as_of)
        if isinstance(spec, PriceOnDateQuery):
            return _price_on_date(engine, spec, as_of)
        if isinstance(spec, PriceIncreaseEffectQuery):
            return _price_increase_effect(engine, spec, as_of)
        if isinstance(spec, CustomersMinOrdersQuery):
            return _customers_min_orders(engine, spec, as_of)
        if isinstance(spec, TopCustomersQuery):
            return _top_customers(engine, spec, as_of)
        if isinstance(spec, OrdersMinAmountQuery):
            return _orders_min_amount(engine, spec, as_of)
        if isinstance(spec, DaysMinRevenueQuery):
            return _days_min_revenue(engine, spec, as_of)
        if isinstance(spec, DebtTotalQuery):
            return _debt_total(engine, spec, as_of)
        if isinstance(spec, TopDebtorsQuery):
            return _top_debtors(engine, spec, as_of)
        if isinstance(spec, DebtAgingQuery):
            return _debt_aging(engine, spec, as_of)
        if isinstance(spec, DebtAddedQuery):
            return _debt_added(engine, spec, as_of)
        if isinstance(spec, DebtCollectedQuery):
            return _debt_collected(engine, spec, as_of)
        if isinstance(spec, CustomerDebtFlowQuery):
            return _customer_debt_flow(engine, spec, as_of)
        from datamate.metrics.ext_e4d import answer_e4d

        e4d = answer_e4d(engine, spec, as_of)
        if e4d is not None:
            return e4d
    except Unanswerable as exc:
        return Answer.unanswerable(str(exc))
    except _Abstain:
        return Answer.abstain()
    except PeriodError:
        return Answer.abstain()
    raise TypeError(f"unsupported ext spec {spec!r}")


def customer_who_phrase(engine: MetricEngine) -> str:
    """How THIS shop identifies customers (for describe/verifier)."""
    mode = _detect_customer_mode(engine)
    if mode == "by_code":
        return "customers identified by customer code, walk-in orders excluded"
    if mode == "by_name":
        return "customers identified by name on the order, walk-in orders excluded"
    return "customers"


def debt_describe_meta(engine: MetricEngine, as_of_day: datetime.date | None = None) -> dict[str, str]:
    """Shop-derived phrases for debt describe/verifier lines."""
    source = _debt_payment_source_label(engine, as_of_day)
    excluded = _debt_exclusion_phrase(engine)
    return {
        "payment_source": source,
        "exclusions": excluded,
        "customer_who": customer_who_phrase(engine),
    }


def _qty_sold(engine: MetricEngine, spec: QtySoldQuery, as_of: datetime.date) -> Answer:
    from datamate.metrics.engine import Unanswerable, _as_int

    cov, window = engine._window(spec.period, as_of)
    if cov == "none":
        return Answer.unanswerable("period lies entirely outside the shop's data")
    if not _ext_recorded(engine, "order_lines", "qty_base"):
        raise Unanswerable("qty_base")
    where, params = _revenue_line_where(engine, window)
    where += f" AND {_non_product_sql(engine)}"
    where, params = _apply_size_filter(engine, spec.size, where, params)
    where, params = _apply_category_filter(engine, spec.category, where, params)
    if spec.product is not None:
        keys = _require_product_keys(engine, spec.product)
        where, params = _apply_product_filter(where, params, keys, spec.product)
    else:
        assert spec.base_unit is not None
        if not _ext_recorded(engine, "products", "base_unit"):
            raise Unanswerable("base_unit")
        where += (
            " AND EXISTS (SELECT 1 FROM products p WHERE p.product_key = l.product_key "
            "AND p.base_unit = ?)"
        )
        params.append(spec.base_unit)
    if _any_null_qty_base(engine, where, params):
        raise _Abstain()
    sql = (
        "SELECT COALESCE(SUM(l.qty_base), 0) "
        "FROM order_lines l JOIN orders o ON o.order_id = l.order_id "
        f"WHERE {where}"
    )
    total = _as_int(engine._scalar(sql, params))
    return Answer(type="number", answer=render_metric("order_count", total, engine.rules))


def _top_products(engine: MetricEngine, spec: TopProductsQuery, as_of: datetime.date) -> Answer:
    from datamate.metrics.engine import Unanswerable

    cov, window = engine._window(spec.period, as_of)
    if cov == "none":
        return Answer.unanswerable("period lies entirely outside the shop's data")
    # Rulebook ranking_measure defines plain "bán chạy" (parsed as qty_base); an explicit
    # money or profit measure from the parser ("nhiều tiền nhất", "lời nhất") still wins.
    by = engine.rules.ranking_measure if spec.by == "qty_base" else spec.by
    if by == "qty_base" and not _ext_recorded(engine, "order_lines", "qty_base"):
        raise Unanswerable("qty_base")
    if by == "gross_profit":
        if not _ext_recorded(engine, "order_lines", "cost_per_unit"):
            raise Unanswerable("cost_per_unit")
        if not _ext_recorded(engine, "order_lines", "qty"):
            raise Unanswerable("qty")
    where, params = _revenue_line_where(engine, window)
    where += f" AND {_non_product_sql(engine)}"
    where, params = _apply_size_filter(engine, spec.size, where, params)
    where, params = _apply_category_filter(engine, spec.category, where, params)
    if by in {"qty_base", "gross_profit"} and _any_null_qty_base(engine, where, params):
        raise _Abstain()
    if by == "gross_profit" and _any_null_cost(engine, where, params):
        raise _Abstain()
    if by == "qty_base":
        measure = "SUM(l.qty_base)"
    elif by == "revenue":
        measure = "COALESCE(SUM(l.line_amount), 0)"
    else:
        measure = "SUM(l.line_amount) - SUM(l.qty * l.cost_per_unit)"
    sql = (
        "SELECT l.product_key, "
        "MAX(l.product_id) AS product_id, "
        "MAX(l.product_name) AS product_name, "
        f"{measure} AS amount "
        "FROM order_lines l JOIN orders o ON o.order_id = l.order_id "
        f"WHERE {where} GROUP BY l.product_key"
    )
    rows = engine.con.execute(sql, params).fetchall()
    if not rows:
        raise _Abstain()
    ranked = _rank_product_rows(rows, mode=spec.mode)
    chosen = ranked[: spec.k]
    if spec.k == 1:
        return Answer(type="string", answer=_display_name(chosen[0]))
    return Answer(type="list", answer=[_display_name(row) for row in chosen])


def _unsold_products(
    engine: MetricEngine, spec: UnsoldProductsQuery, as_of: datetime.date
) -> Answer:
    cov, window = engine._window(spec.period, as_of)
    if cov == "none":
        return Answer.unanswerable("period lies entirely outside the shop's data")
    pre_start = engine.data_start or datetime.date(1970, 1, 1)
    pre_end = window.start - datetime.timedelta(days=1)
    if pre_end < pre_start:
        labels: list[str] = []
    else:
        pre = DateRange(start=pre_start, end=pre_end)
        where_pre, params_pre = _revenue_line_where(engine, pre)
        where_pre += f" AND {_non_product_sql(engine)}"
        where_pre, params_pre = _apply_category_filter(
            engine, spec.category, where_pre, params_pre
        )
        where_in, params_in = _revenue_line_where(
            engine, window, order_alias="o2", line_alias="l2"
        )
        where_in += f" AND {_non_product_sql(engine, line_alias='l2')}"
        where_in, params_in = _apply_category_filter(
            engine, spec.category, where_in, params_in, line_alias="l2"
        )
        sql = (
            "SELECT l.product_key, MAX(l.product_id), MAX(l.product_name) "
            "FROM order_lines l JOIN orders o ON o.order_id = l.order_id "
            f"WHERE {where_pre} "
            "AND l.product_key NOT IN ("
            "SELECT DISTINCT l2.product_key FROM order_lines l2 "
            f"JOIN orders o2 ON o2.order_id = l2.order_id WHERE {where_in}"
            ") GROUP BY l.product_key"
        )
        rows = engine.con.execute(sql, [*params_pre, *params_in]).fetchall()
        ranked = _rank_product_rows([(r[0], r[1], r[2], 0) for r in rows], mode="max")
        labels = [_display_name(r) for r in ranked]
    if spec.output == "count":
        return Answer(type="number", answer=len(labels))
    return Answer(type="list", answer=labels)


def _products_in_min_orders(
    engine: MetricEngine, spec: ProductsInMinOrdersQuery, as_of: datetime.date
) -> Answer:
    cov, window = engine._window(spec.period, as_of)
    if cov == "none":
        return Answer.unanswerable("period lies entirely outside the shop's data")
    where, params = _revenue_line_where(engine, window)
    where += f" AND {_non_product_sql(engine)}"
    where, params = _apply_category_filter(engine, spec.category, where, params)
    sql = (
        "SELECT l.product_key, MAX(l.product_id), MAX(l.product_name), "
        "COUNT(DISTINCT o.order_id) AS n_orders "
        "FROM order_lines l JOIN orders o ON o.order_id = l.order_id "
        f"WHERE {where} GROUP BY l.product_key HAVING COUNT(DISTINCT o.order_id) >= ?"
    )
    rows = engine.con.execute(sql, [*params, spec.n]).fetchall()
    ranked = _rank_product_rows([(r[0], r[1], r[2], r[3]) for r in rows], mode="max")
    labels = [_display_name(r) for r in ranked]
    if spec.output == "count":
        return Answer(type="number", answer=len(labels))
    return Answer(type="list", answer=labels)


def _price_on_date(engine: MetricEngine, spec: PriceOnDateQuery, as_of: datetime.date) -> Answer:
    from datamate.metrics.engine import Unanswerable

    cov, window = engine._window(spec.date, as_of)
    if cov == "none":
        return Answer.unanswerable("period lies entirely outside the shop's data")
    if window.start != window.end:
        raise _Abstain()
    day = window.start
    if not _ext_recorded(engine, "order_lines", "unit_price"):
        raise Unanswerable("unit_price")
    keys = _require_product_keys(engine, spec.product)
    prices = _unit_prices_on_day(engine, keys, day, product=spec.product)
    if not prices:
        raise _Abstain()
    distinct = set(prices)
    if len(distinct) > 1:
        raise _Abstain()
    return Answer(type="number", answer=next(iter(distinct)))


def _price_increase_effect(
    engine: MetricEngine, spec: PriceIncreaseEffectQuery, as_of: datetime.date
) -> Answer:
    from datamate.metrics.engine import Unanswerable

    keys = _require_product_keys(engine, spec.product)
    if not _ext_recorded(engine, "order_lines", "qty_base"):
        raise Unanswerable("qty_base")
    increase_day = _resolve_increase_day(engine, keys, spec, as_of, product=spec.product)
    if increase_day is None:
        raise _Abstain()
    n = spec.n_days
    after = DateRange(
        start=increase_day, end=increase_day + datetime.timedelta(days=n - 1)
    )
    before = DateRange(
        start=increase_day - datetime.timedelta(days=n),
        end=increase_day - datetime.timedelta(days=1),
    )
    where_a, params_a = _revenue_line_where(engine, after)
    where_a += f" AND {_non_product_sql(engine)}"
    where_a, params_a = _apply_product_filter(where_a, params_a, keys, spec.product)
    if _any_null_qty_base(engine, where_a, params_a):
        raise _Abstain()
    where_b, params_b = _revenue_line_where(engine, before)
    where_b += f" AND {_non_product_sql(engine)}"
    where_b, params_b = _apply_product_filter(where_b, params_b, keys, spec.product)
    if _any_null_qty_base(engine, where_b, params_b):
        raise _Abstain()
    qty_after = _qty_in_range(engine, keys, after, product=spec.product)
    qty_before = _qty_in_range(engine, keys, before, product=spec.product)
    if qty_before == 0:
        raise _Abstain()
    change = (Decimal(qty_after) - Decimal(qty_before)) / Decimal(qty_before) * Decimal(100)
    return Answer(type="percent", answer=render_percent(change, engine.rules))


def _revenue_line_where(
    engine: MetricEngine,
    period: DateRange,
    *,
    order_alias: str = "o",
    line_alias: str = "l",
) -> tuple[str, list[Any]]:
    del line_alias
    o = order_alias
    parts = [f"{o}.order_date BETWEEN ? AND ?"]
    params: list[Any] = [period.start, period.end]
    if not engine.rules.revenue_statuses:
        parts.append("FALSE")
    else:
        marks = ", ".join("?" for _ in engine.rules.revenue_statuses)
        parts.append(f"{o}.status IN ({marks})")
        params.extend(engine.rules.revenue_statuses)
    if engine.rules.exclude_orders_with_return:
        parts.append(f"NOT {o}.has_return")
    return " AND ".join(parts), params


def _non_product_sql(engine: MetricEngine, *, line_alias: str = "l") -> str:
    labels = list(engine.rules.non_product_line_labels)
    if not labels:
        return "TRUE"
    escaped = ", ".join("'" + label.replace("'", "''") + "'" for label in labels)
    return (
        f"({line_alias}.product_name IS NULL OR "
        f"{line_alias}.product_name NOT IN ({escaped}))"
    )


def _apply_size_filter(
    engine: MetricEngine,
    size: str | None,
    where: str,
    params: list[Any],
) -> tuple[str, list[Any]]:
    if size is None:
        return where, params
    suffix = engine.rules.size_suffixes.get(size)
    if suffix is None:
        raise _Abstain()
    # Case-insensitive endswith on product_name.
    where += " AND lower(rtrim(l.product_name)) LIKE ?"
    params = [*params, "%" + suffix.casefold()]
    return where, params


def _apply_category_filter(
    engine: MetricEngine,
    category: str | None,
    where: str,
    params: list[Any],
    *,
    line_alias: str = "l",
) -> tuple[str, list[Any]]:
    if category is None:
        return where, params
    if not _ext_recorded(engine, "products", "category"):
        from datamate.metrics.engine import Unanswerable

        raise Unanswerable("category")
    where += (
        f" AND EXISTS (SELECT 1 FROM products p WHERE p.product_key = {line_alias}.product_key "
        "AND p.category = ?)"
    )
    params = [*params, category]
    return where, params


def _product_keys(engine: MetricEngine, product: str) -> set[str]:
    label = normalize_label(product)
    if label is None:
        return set()
    return {label} | engine._product_keys_for(label)


def _require_product_keys(engine: MetricEngine, product: str) -> set[str]:
    """Keys for a named product; abstain when the shop has no such product at all."""
    keys = _product_keys(engine, product)
    if not keys:
        raise _Abstain()
    marks = ", ".join("?" for _ in keys)
    params: list[Any] = [*sorted(keys), product, product]
    found = engine.con.execute(
        f"SELECT 1 FROM order_lines WHERE product_key IN ({marks}) "
        "OR product_id = ? OR product_name = ? LIMIT 1",
        params,
    ).fetchone()
    if found is None:
        found = engine.con.execute(
            f"SELECT 1 FROM products WHERE product_key IN ({marks}) "
            "OR product_id = ? OR product_name = ? LIMIT 1",
            params,
        ).fetchone()
    if found is None:
        raise _Abstain()
    return keys


def _apply_product_filter(
    where: str, params: list[Any], keys: set[str], product: str
) -> tuple[str, list[Any]]:
    marks = ", ".join("?" for _ in keys)
    where += f" AND (l.product_key IN ({marks}) OR l.product_id = ?)"
    params = [*params, *sorted(keys), product]
    return where, params


def _any_null_qty_base(engine: MetricEngine, where: str, params: list[Any]) -> bool:
    sql = (
        "SELECT 1 FROM order_lines l JOIN orders o ON o.order_id = l.order_id "
        f"WHERE {where} AND l.qty_base IS NULL LIMIT 1"
    )
    return engine.con.execute(sql, params).fetchone() is not None


def _any_null_cost(engine: MetricEngine, where: str, params: list[Any]) -> bool:
    sql = (
        "SELECT 1 FROM order_lines l JOIN orders o ON o.order_id = l.order_id "
        f"WHERE {where} AND l.cost_per_unit IS NULL LIMIT 1"
    )
    return engine.con.execute(sql, params).fetchone() is not None


def _rank_product_rows(
    rows: list[tuple[Any, ...]], *, mode: str
) -> list[tuple[Any, ...]]:
    def sort_key(row: tuple[Any, ...]) -> tuple[Decimal, str, str]:
        amount = Decimal(row[3] if row[3] is not None else 0)
        rank = -amount if mode == "max" else amount
        code = str(row[1]).strip() if row[1] is not None else ""
        name = str(row[2]) if row[2] is not None else ""
        tie = code.casefold() if code else name.casefold()
        return (rank, tie, name)

    return sorted(rows, key=sort_key)


def _display_name(row: tuple[Any, ...]) -> str:
    if row[2] is not None and str(row[2]).strip():
        return str(row[2])
    if row[1] is not None and str(row[1]).strip():
        return str(row[1])
    return str(row[0])


def _unit_prices_on_day(
    engine: MetricEngine,
    keys: set[str],
    day: datetime.date,
    *,
    product: str | None = None,
) -> list[int]:
    marks = ", ".join("?" for _ in keys)
    parts = [
        "o.order_date = ?",
        f"(l.product_key IN ({marks}) OR l.product_id = ?)",
        "l.unit_price IS NOT NULL",
    ]
    params: list[Any] = [day, *sorted(keys), product or next(iter(keys))]
    if not engine.rules.price_on_date_include_cancelled_returned:
        if engine.rules.revenue_statuses:
            st = ", ".join("?" for _ in engine.rules.revenue_statuses)
            parts.append(f"o.status IN ({st})")
            params.extend(engine.rules.revenue_statuses)
        if engine.rules.exclude_orders_with_return:
            parts.append("NOT o.has_return")
    sql = (
        "SELECT CASE "
        "WHEN l.qty_base IS NOT NULL AND l.qty_base <> 0 AND l.qty IS NOT NULL "
        "THEN CAST(ROUND(l.unit_price * l.qty / l.qty_base) AS BIGINT) "
        "ELSE l.unit_price END "
        "FROM order_lines l JOIN orders o ON o.order_id = l.order_id "
        f"WHERE {' AND '.join(parts)}"
    )
    return [int(row[0]) for row in engine.con.execute(sql, params).fetchall() if row[0] is not None]


def _qty_in_range(
    engine: MetricEngine,
    keys: set[str],
    period: DateRange,
    *,
    product: str | None = None,
) -> int:
    from datamate.metrics.engine import _as_int

    where, params = _revenue_line_where(engine, period)
    where += f" AND {_non_product_sql(engine)}"
    where, params = _apply_product_filter(where, params, keys, product or next(iter(keys)))
    sql = (
        "SELECT COALESCE(SUM(l.qty_base), 0) "
        "FROM order_lines l JOIN orders o ON o.order_id = l.order_id "
        f"WHERE {where}"
    )
    return _as_int(engine._scalar(sql, params))


def _resolve_increase_day(
    engine: MetricEngine,
    keys: set[str],
    spec: PriceIncreaseEffectQuery,
    as_of: datetime.date,
    *,
    product: str | None = None,
) -> datetime.date | None:
    from datamate.metrics.engine import Unanswerable

    if spec.date is not None:
        cov, window = engine._window(spec.date, as_of)
        if cov == "none" or window.start != window.end:
            return None
        return window.start
    if _ext_recorded(engine, "price_history", "price"):
        marks = ", ".join("?" for _ in keys)
        rows = engine.con.execute(
            "SELECT effective_date, price FROM price_history "
            f"WHERE product_key IN ({marks}) AND effective_date IS NOT NULL "
            "AND price IS NOT NULL ORDER BY effective_date ASC, price ASC",
            sorted(keys),
        ).fetchall()
        prev: int | None = None
        for day, price in rows:
            amount = int(price)
            d = day if isinstance(day, datetime.date) else datetime.date.fromisoformat(str(day)[:10])
            if prev is not None and amount > prev:
                return d
            prev = amount
        return None
    if not _ext_recorded(engine, "order_lines", "unit_price"):
        raise Unanswerable("unit_price")
    marks = ", ".join("?" for _ in keys)
    days = engine.con.execute(
        "SELECT DISTINCT o.order_date FROM order_lines l "
        "JOIN orders o ON o.order_id = l.order_id "
        f"WHERE (l.product_key IN ({marks}) OR l.product_id = ?) "
        "AND l.unit_price IS NOT NULL "
        "ORDER BY o.order_date ASC",
        [*sorted(keys), product or next(iter(keys))],
    ).fetchall()
    prev_price: int | None = None
    for (day,) in days:
        d = day if isinstance(day, datetime.date) else datetime.date.fromisoformat(str(day)[:10])
        prices = _unit_prices_on_day(engine, keys, d, product=product)
        if not prices:
            continue
        distinct = set(prices)
        if len(distinct) != 1:
            continue
        price = next(iter(distinct))
        if prev_price is not None and price > prev_price:
            return d
        prev_price = price
    return None


def _ext_recorded(engine: MetricEngine, table: str, column: str) -> bool:
    key = (table, column)
    cached = engine._present.get(key)
    if cached is not None:
        return cached
    try:
        sql = f'SELECT 1 FROM "{table}" WHERE "{column}" IS NOT NULL LIMIT 1'
        present = engine.con.execute(sql).fetchone() is not None
    except Exception:  # noqa: BLE001 — missing table/column => not recorded
        present = False
    engine._present[key] = present
    return present


def _detect_customer_mode(engine: MetricEngine) -> str | None:
    """by_code if customer_id is recorded, else by_name if customer_name is; else None."""
    if _ext_recorded(engine, "orders", "customer_id"):
        return "by_code"
    if _ext_recorded(engine, "orders", "customer_name"):
        return "by_name"
    return None


def _fold_customer_name(raw: str | None) -> str | None:
    if raw is None or not isinstance(raw, str):
        return None
    folded = fold_text(raw)
    return folded or None


def _revenue_order_predicates(
    engine: MetricEngine, *, order_alias: str = "o"
) -> tuple[str, list[Any]]:
    o = order_alias
    parts: list[str] = []
    params: list[Any] = []
    if not engine.rules.revenue_statuses:
        parts.append("FALSE")
    else:
        marks = ", ".join("?" for _ in engine.rules.revenue_statuses)
        parts.append(f"{o}.status IN ({marks})")
        params.extend(engine.rules.revenue_statuses)
    if engine.rules.exclude_orders_with_return:
        parts.append(f"NOT {o}.has_return")
    return " AND ".join(parts), params


def _ensure_customer_orders(engine: MetricEngine) -> str:
    """Build TEMP customer_orders over revenue orders; return mode. Raises Unanswerable."""
    from datamate.metrics.engine import Unanswerable

    if getattr(engine, "_dm_co_ready", False):
        mode = engine._dm_co_mode
        if mode is None:
            raise Unanswerable("customer")
        return str(mode)

    mode = _detect_customer_mode(engine)
    if mode is None:
        engine._dm_co_mode = None
        engine._dm_co_ready = True
        raise Unanswerable("customer")

    rev_where, rev_params = _revenue_order_predicates(engine)
    if mode == "by_code":
        engine.con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE customer_orders AS
            SELECT o.order_id,
                   CAST(o.customer_id AS VARCHAR) AS customer_ref,
                   COALESCE(
                     NULLIF(TRIM(CAST(o.customer_name AS VARCHAR)), ''),
                     CAST(o.customer_id AS VARCHAR)
                   ) AS customer_label
            FROM orders o
            WHERE {rev_where}
              AND o.customer_id IS NOT NULL
              AND NOT COALESCE(o.is_walk_in, FALSE)
            """,
            rev_params,
        )
    else:
        rows = engine.con.execute(
            f"""
            SELECT o.order_id, o.customer_name
            FROM orders o
            WHERE {rev_where}
              AND o.customer_name IS NOT NULL
              AND NOT COALESCE(o.is_walk_in, FALSE)
            """,
            rev_params,
        ).fetchall()
        engine.con.execute(
            """
            CREATE OR REPLACE TEMP TABLE customer_orders (
                order_id VARCHAR,
                customer_ref VARCHAR,
                customer_label VARCHAR
            )
            """
        )
        batch: list[tuple[str, str, str]] = []
        for order_id, cname in rows:
            ref = _fold_customer_name(cname if isinstance(cname, str) else None)
            if ref is None:
                continue
            label = str(cname).strip()
            batch.append((str(order_id), ref, label))
        if batch:
            engine.con.executemany(
                "INSERT INTO customer_orders VALUES (?, ?, ?)", batch
            )

    engine._dm_co_mode = mode
    engine._dm_co_ready = True
    return mode


def _customers_min_orders(
    engine: MetricEngine, spec: CustomersMinOrdersQuery, as_of: datetime.date
) -> Answer:
    cov, window = engine._window(spec.period, as_of)
    if cov == "none":
        return Answer.unanswerable("period lies entirely outside the shop's data")
    _ensure_customer_orders(engine)
    sql = """
        SELECT c.customer_ref, COUNT(DISTINCT c.order_id) AS n_orders
        FROM customer_orders c
        JOIN orders o ON o.order_id = c.order_id
        WHERE o.order_date BETWEEN ? AND ?
        GROUP BY c.customer_ref
    """
    rows = engine.con.execute(sql, [window.start, window.end]).fetchall()
    if not rows:
        if spec.output == "share":
            raise _Abstain()
        return Answer(type="number", answer=0)
    qualifying = sum(1 for r in rows if int(r[1]) >= spec.n)
    if spec.output == "count":
        return Answer(
            type="number",
            answer=render_metric("order_count", qualifying, engine.rules),
        )
    total = len(rows)
    percent = (Decimal(qualifying) * Decimal(100)) / Decimal(total)
    return Answer(type="percent", answer=render_percent(percent, engine.rules))


def _top_customers(
    engine: MetricEngine, spec: TopCustomersQuery, as_of: datetime.date
) -> Answer:
    cov, window = engine._window(spec.period, as_of)
    if cov == "none":
        return Answer.unanswerable("period lies entirely outside the shop's data")
    _ensure_customer_orders(engine)
    if spec.by == "order_count":
        sql = """
            SELECT c.customer_ref,
                   MAX(c.customer_label) AS customer_label,
                   COUNT(DISTINCT c.order_id) AS amount
            FROM customer_orders c
            JOIN orders o ON o.order_id = c.order_id
            WHERE o.order_date BETWEEN ? AND ?
            GROUP BY c.customer_ref
        """
        rows = engine.con.execute(sql, [window.start, window.end]).fetchall()
    else:
        sql = """
            SELECT c.customer_ref,
                   MAX(c.customer_label) AS customer_label,
                   COALESCE(SUM(l.line_amount), 0) AS amount
            FROM customer_orders c
            JOIN orders o ON o.order_id = c.order_id
            JOIN order_lines l ON l.order_id = o.order_id
            WHERE o.order_date BETWEEN ? AND ?
            GROUP BY c.customer_ref
        """
        rows = engine.con.execute(sql, [window.start, window.end]).fetchall()
    if not rows:
        raise _Abstain()
    ranked = _rank_customer_rows(
        [(str(r[0]), str(r[1]), int(r[2]) if r[2] is not None else 0) for r in rows],
        mode=spec.mode,
    )
    chosen = ranked[: spec.k]
    if spec.k == 1:
        return Answer(type="string", answer=chosen[0][1])
    return Answer(type="list", answer=[row[1] for row in chosen])


def _rank_customer_rows(
    rows: list[tuple[str, str, int]], *, mode: str
) -> list[tuple[str, str, int]]:
    """Sort by measure (max/min), tie -> customer_ref ascending (code or folded name)."""
    if mode == "max":
        return sorted(rows, key=lambda r: (-r[2], r[0]))
    return sorted(rows, key=lambda r: (r[2], r[0]))


def _orders_min_amount(
    engine: MetricEngine, spec: OrdersMinAmountQuery, as_of: datetime.date
) -> Answer:
    from datamate.metrics.engine import _as_int

    cov, window = engine._window(spec.period, as_of)
    if cov == "none":
        return Answer.unanswerable("period lies entirely outside the shop's data")
    rev_where, rev_params = _revenue_order_predicates(engine)
    op = spec.op
    sql = f"""
        SELECT COUNT(*) FROM (
            SELECT o.order_id
            FROM orders o
            LEFT JOIN order_lines l ON l.order_id = o.order_id
            WHERE {rev_where}
              AND o.order_date BETWEEN ? AND ?
            GROUP BY o.order_id
            HAVING COALESCE(SUM(l.line_amount), 0) {op} ?
        ) t
    """
    total = _as_int(
        engine._scalar(sql, [*rev_params, window.start, window.end, spec.x])
    )
    return Answer(type="number", answer=render_metric("order_count", total, engine.rules))


def _days_min_revenue(
    engine: MetricEngine, spec: DaysMinRevenueQuery, as_of: datetime.date
) -> Answer:
    from datamate.metrics.engine import _as_int

    cov, window = engine._window(spec.period, as_of)
    if cov == "none":
        return Answer.unanswerable("period lies entirely outside the shop's data")
    rev_where, rev_params = _revenue_order_predicates(engine)
    op = spec.op
    sql = f"""
        SELECT COUNT(*) FROM (
            SELECT o.order_date
            FROM orders o
            LEFT JOIN order_lines l ON l.order_id = o.order_id
            WHERE {rev_where}
              AND o.order_date BETWEEN ? AND ?
            GROUP BY o.order_date
            HAVING COALESCE(SUM(l.line_amount), 0) {op} ?
        ) t
    """
    total = _as_int(
        engine._scalar(sql, [*rev_params, window.start, window.end, spec.x])
    )
    return Answer(type="number", answer=render_metric("order_count", total, engine.rules))


@dataclass
class _DebtOrderRow:
    order_id: str
    order_date: datetime.date
    remaining: int
    is_walk_in: bool
    customer_ref: str | None
    customer_label: str | None


def _debt_exclude_norm_set(labels: list[str]) -> set[str]:
    out: set[str] = set()
    for label in labels:
        norm = normalize_label(label)
        if norm is not None:
            out.add(norm)
    return out


def _shop_labels_matching(
    engine: MetricEngine, column: str, exclude_norms: set[str]
) -> list[str]:
    """Raw distinct order values that are, or contain as whole words, an excluded label.

    "Thu hộ (COD)" matches "cod"; "Sàn TMĐT" matches "sàn tmđt".
    """
    if not exclude_norms:
        return []
    try:
        rows = engine.con.execute(
            f'SELECT DISTINCT "{column}" FROM orders WHERE "{column}" IS NOT NULL'
        ).fetchall()
    except Exception:  # noqa: BLE001
        return []
    found: list[str] = []
    for (raw,) in rows:
        if not isinstance(raw, str):
            continue
        norm = normalize_label(raw)
        if norm is None:
            continue
        words = _label_words(norm)
        if norm in exclude_norms or any(
            (label_words := _label_words(label)) and _contains_words(words, label_words)
            for label in exclude_norms
        ):
            found.append(raw)
    return found


def _label_words(text: str) -> list[str]:
    return re.findall(r"\w+", text)


def _contains_words(words: list[str], part: list[str]) -> bool:
    n = len(part)
    return any(words[i : i + n] == part for i in range(len(words) - n + 1))


def _debt_exclusion_labels(engine: MetricEngine) -> tuple[list[str], list[str]]:
    pay = _shop_labels_matching(
        engine,
        "payment_method",
        _debt_exclude_norm_set(engine.rules.debt_exclude_payment_labels),
    )
    ch = _shop_labels_matching(
        engine,
        "channel",
        _debt_exclude_norm_set(engine.rules.debt_exclude_channel_labels),
    )
    return pay, ch


def _debt_exclusion_phrase(engine: MetricEngine) -> str:
    pay, ch = _debt_exclusion_labels(engine)
    labels = [*pay, *ch]
    if not labels:
        return "none found"
    return ", ".join(labels)


def _debt_payment_source_label(
    engine: MetricEngine, as_of_day: datetime.date | None
) -> str:
    if as_of_day is not None and _debts_snapshot_usable(engine, as_of_day):
        return "debts snapshot"
    if _ext_recorded(engine, "payments", "amount"):
        return "payments table"
    if _ext_recorded(engine, "orders", "amount_paid"):
        return "amount_paid on the order"
    return "unavailable"


def _debts_snapshot_usable(engine: MetricEngine, as_of_day: datetime.date) -> bool:
    if not _ext_recorded(engine, "debts", "amount_remaining"):
        return False
    try:
        row = engine.con.execute(
            "SELECT 1 FROM debts WHERE as_of_date = ? LIMIT 1", [as_of_day]
        ).fetchone()
    except Exception:  # noqa: BLE001
        return False
    return row is not None


def _require_reliable_payable(engine: MetricEngine) -> None:
    """Abstain when the order total cannot be trusted: an order never owes more than its
    total, so many orders with amount_due > order_payable mean the payable column holds
    something else (e.g. the amount already paid)."""
    if not _ext_recorded(engine, "orders", "amount_due"):
        return
    row = engine.con.execute(
        "SELECT COUNT(*) FILTER (WHERE amount_due > 0), "
        "COUNT(*) FILTER (WHERE amount_due > order_payable + 1) "
        "FROM orders WHERE order_payable IS NOT NULL"
    ).fetchone()
    owing, broken = (int(row[0] or 0), int(row[1] or 0)) if row else (0, 0)
    if broken > 0 and broken * 10 >= owing:
        raise _Abstain()


def _require_debt_inputs(engine: MetricEngine) -> None:
    from datamate.metrics.engine import Unanswerable

    if not _ext_recorded(engine, "orders", "order_payable"):
        raise Unanswerable("order_payable")
    has_pay = _ext_recorded(engine, "payments", "amount")
    has_paid = _ext_recorded(engine, "orders", "amount_paid")
    has_debts = _ext_recorded(engine, "debts", "amount_remaining")
    if not has_pay and not has_paid and not has_debts:
        raise Unanswerable("payment")


def _debt_not_excluded_sql(
    engine: MetricEngine, *, order_alias: str = "o"
) -> tuple[str, list[Any]]:
    """SQL fragment: not cancelled and not COD/marketplace (shop labels)."""
    o = order_alias
    parts = [f"{o}.status <> 'cancelled'"]
    params: list[Any] = []
    pay_labels, ch_labels = _debt_exclusion_labels(engine)
    if pay_labels:
        marks = ", ".join("?" for _ in pay_labels)
        parts.append(f"({o}.payment_method IS NULL OR {o}.payment_method NOT IN ({marks}))")
        params.extend(pay_labels)
    if ch_labels:
        marks = ", ".join("?" for _ in ch_labels)
        parts.append(f"({o}.channel IS NULL OR {o}.channel NOT IN ({marks}))")
        params.extend(ch_labels)
    return " AND ".join(parts), params


def _customer_fields_for_order(
    *,
    mode: str | None,
    customer_id: object,
    customer_name: object,
    is_walk_in: bool,
) -> tuple[str | None, str | None]:
    if is_walk_in:
        return None, None
    if mode == "by_code":
        if customer_id is None:
            return None, None
        ref = str(customer_id)
        label = (
            str(customer_name).strip()
            if isinstance(customer_name, str) and customer_name.strip()
            else ref
        )
        return ref, label
    if mode == "by_name":
        if not isinstance(customer_name, str):
            return None, None
        ref = _fold_customer_name(customer_name)
        if ref is None:
            return None, None
        return ref, customer_name.strip()
    return None, None


def _paid_by_date_map(
    engine: MetricEngine, as_of_day: datetime.date
) -> dict[str, int]:
    """order_id -> amount paid by end of as_of_day."""
    paid: dict[str, int] = {}
    if _ext_recorded(engine, "payments", "amount"):
        rows = engine.con.execute(
            """
            SELECT order_id, COALESCE(SUM(amount), 0)
            FROM payments
            WHERE pay_date IS NOT NULL AND pay_date <= ? AND order_id IS NOT NULL
            GROUP BY order_id
            """,
            [as_of_day],
        ).fetchall()
        for oid, amt in rows:
            paid[str(oid)] = int(amt)
        return paid
    # Fallback: amount_paid counts as paid on the order day.
    rows = engine.con.execute(
        """
        SELECT order_id, order_date, COALESCE(amount_paid, 0)
        FROM orders
        WHERE order_payable IS NOT NULL
        """
    ).fetchall()
    for oid, odate, amt in rows:
        if odate is not None and odate <= as_of_day:
            paid[str(oid)] = int(amt)
    return paid


def _debt_rows_at(
    engine: MetricEngine, as_of_day: datetime.date
) -> list[_DebtOrderRow]:
    """Debt-eligible orders with remaining > 0 at end of as_of_day."""
    _require_debt_inputs(engine)
    _require_reliable_payable(engine)
    mode = _detect_customer_mode(engine)
    excl_sql, excl_params = _debt_not_excluded_sql(engine)

    if _debts_snapshot_usable(engine, as_of_day):
        rows = engine.con.execute(
            f"""
            SELECT d.order_id, d.order_date, d.amount_remaining,
                   COALESCE(o.is_walk_in, FALSE), o.customer_id, o.customer_name
            FROM debts d
            JOIN orders o ON o.order_id = d.order_id
            WHERE d.as_of_date = ?
              AND d.amount_remaining IS NOT NULL
              AND d.amount_remaining > 0
              AND {excl_sql}
            """,
            [as_of_day, *excl_params],
        ).fetchall()
        out: list[_DebtOrderRow] = []
        for oid, odate, rem, walk, cid, cname in rows:
            ref, label = _customer_fields_for_order(
                mode=mode,
                customer_id=cid,
                customer_name=cname,
                is_walk_in=bool(walk),
            )
            out.append(
                _DebtOrderRow(
                    order_id=str(oid),
                    order_date=odate,
                    remaining=int(rem),
                    is_walk_in=bool(walk),
                    customer_ref=ref,
                    customer_label=label,
                )
            )
        return out

    paid = _paid_by_date_map(engine, as_of_day)
    rows = engine.con.execute(
        f"""
        SELECT o.order_id, o.order_date, o.order_payable,
               COALESCE(o.is_walk_in, FALSE), o.customer_id, o.customer_name
        FROM orders o
        WHERE o.order_payable IS NOT NULL
          AND o.order_date IS NOT NULL
          AND o.order_date <= ?
          AND {excl_sql}
        """,
        [as_of_day, *excl_params],
    ).fetchall()
    out = []
    for oid, odate, payable, walk, cid, cname in rows:
        remaining = int(payable) - paid.get(str(oid), 0)
        if remaining <= 0:
            continue
        ref, label = _customer_fields_for_order(
            mode=mode,
            customer_id=cid,
            customer_name=cname,
            is_walk_in=bool(walk),
        )
        out.append(
            _DebtOrderRow(
                order_id=str(oid),
                order_date=odate,
                remaining=remaining,
                is_walk_in=bool(walk),
                customer_ref=ref,
                customer_label=label,
            )
        )
    return out


def _resolve_as_of_day(
    engine: MetricEngine, expr: object, question_as_of: datetime.date
) -> datetime.date:
    from datamate.timeres.periods import resolve

    window = resolve(expr, question_as_of, engine.rules, engine.cal)  # type: ignore[arg-type]
    if window.start != window.end:
        raise _Abstain()
    return window.start


def _debt_total(
    engine: MetricEngine, spec: DebtTotalQuery, as_of: datetime.date
) -> Answer:
    day = _resolve_as_of_day(engine, spec.date, as_of)
    rows = _debt_rows_at(engine, day)
    if spec.output == "customer_count":
        debtors = {r.customer_ref for r in rows if not r.is_walk_in and r.customer_ref is not None}
        return Answer(
            type="number", answer=render_metric("order_count", len(debtors), engine.rules)
        )
    total = sum(r.remaining for r in rows)
    return Answer(type="number", answer=render_metric("revenue", total, engine.rules))


def _top_debtors(
    engine: MetricEngine, spec: TopDebtorsQuery, as_of: datetime.date
) -> Answer:
    day = _resolve_as_of_day(engine, spec.date, as_of)
    rows = _debt_rows_at(engine, day)
    by_cust: dict[str, tuple[str, int]] = {}
    for row in rows:
        if row.is_walk_in or row.customer_ref is None:
            continue
        label = row.customer_label or row.customer_ref
        prev = by_cust.get(row.customer_ref)
        if prev is None:
            by_cust[row.customer_ref] = (label, row.remaining)
        else:
            by_cust[row.customer_ref] = (prev[0], prev[1] + row.remaining)
    if not by_cust:
        raise _Abstain()
    ranked = sorted(by_cust.items(), key=lambda kv: (-kv[1][1], kv[0]))
    chosen = ranked[: spec.k]
    table_rows = [[label, amount] for _, (label, amount) in chosen]
    return Answer(
        type="table",
        answer={"columns": ["customer", "amount"], "rows": table_rows},
    )


def _debt_aging(
    engine: MetricEngine, spec: DebtAgingQuery, as_of: datetime.date
) -> Answer:
    day = _resolve_as_of_day(engine, spec.date, as_of)
    rows = _debt_rows_at(engine, day)
    amounts = [0 for _ in spec.buckets]
    for row in rows:
        age = (day - row.order_date).days
        for idx, bucket in enumerate(spec.buckets):
            if age < bucket.lo:
                continue
            if bucket.hi is not None and age > bucket.hi:
                continue
            amounts[idx] += row.remaining
            break
    return Answer(
        type="list",
        answer=[render_metric("revenue", a, engine.rules) for a in amounts],
    )


def _paid_on_order_day_map(engine: MetricEngine) -> dict[str, int]:
    """Payments dated on the order day (or amount_paid fallback)."""
    paid: dict[str, int] = {}
    if _ext_recorded(engine, "payments", "amount"):
        rows = engine.con.execute(
            """
            SELECT p.order_id, COALESCE(SUM(p.amount), 0)
            FROM payments p
            JOIN orders o ON o.order_id = p.order_id
            WHERE p.pay_date IS NOT NULL
              AND o.order_date IS NOT NULL
              AND p.pay_date = o.order_date
            GROUP BY p.order_id
            """
        ).fetchall()
        for oid, amt in rows:
            paid[str(oid)] = int(amt)
        return paid
    rows = engine.con.execute(
        "SELECT order_id, COALESCE(amount_paid, 0) FROM orders WHERE order_payable IS NOT NULL"
    ).fetchall()
    for oid, amt in rows:
        paid[str(oid)] = int(amt)
    return paid


def _debt_added(
    engine: MetricEngine, spec: DebtAddedQuery, as_of: datetime.date
) -> Answer:
    _require_debt_inputs(engine)
    _require_reliable_payable(engine)
    cov, window = engine._window(spec.period, as_of)
    if cov == "none":
        return Answer.unanswerable("period lies entirely outside the shop's data")
    total = _debt_added_amount(engine, window)
    return Answer(type="number", answer=render_metric("revenue", total, engine.rules))


def _debt_added_amount(
    engine: MetricEngine, window: DateRange, order_ids: set[str] | None = None
) -> int:
    """Unpaid part on the order day of debt orders placed in the window."""
    excl_sql, excl_params = _debt_not_excluded_sql(engine)
    paid_same_day = _paid_on_order_day_map(engine)
    rows = engine.con.execute(
        f"""
        SELECT o.order_id, o.order_payable
        FROM orders o
        WHERE o.order_payable IS NOT NULL
          AND o.order_date BETWEEN ? AND ?
          AND {excl_sql}
        """,
        [window.start, window.end, *excl_params],
    ).fetchall()
    total = 0
    for oid, payable in rows:
        if order_ids is not None and str(oid) not in order_ids:
            continue
        unpaid = int(payable) - paid_same_day.get(str(oid), 0)
        if unpaid > 0:
            total += unpaid
    return total


def _debt_collected(
    engine: MetricEngine, spec: DebtCollectedQuery, as_of: datetime.date
) -> Answer:
    _require_debt_inputs(engine)
    cov, window = engine._window(spec.period, as_of)
    if cov == "none":
        return Answer.unanswerable("period lies entirely outside the shop's data")
    total = _debt_collected_amount(engine, window)
    return Answer(type="number", answer=render_metric("revenue", total, engine.rules))


def _debt_collected_amount(
    engine: MetricEngine, window: DateRange, order_ids: set[str] | None = None
) -> int:
    """Payments after the order day, dated in the window, on debt orders."""
    # Without a payments table, amount_paid is on the order day -> nothing "collected" later.
    if not _ext_recorded(engine, "payments", "amount"):
        return 0
    excl_sql, excl_params = _debt_not_excluded_sql(engine)
    rows = engine.con.execute(
        f"""
        SELECT p.order_id, COALESCE(SUM(p.amount), 0)
        FROM payments p
        JOIN orders o ON o.order_id = p.order_id
        WHERE p.pay_date BETWEEN ? AND ?
          AND o.order_date IS NOT NULL
          AND p.pay_date > o.order_date
          AND {excl_sql}
        GROUP BY p.order_id
        """,
        [window.start, window.end, *excl_params],
    ).fetchall()
    return sum(
        int(amount) for oid, amount in rows if order_ids is None or str(oid) in order_ids
    )


def _customer_debt_flow(
    engine: MetricEngine, spec: CustomerDebtFlowQuery, as_of: datetime.date
) -> Answer:
    _require_debt_inputs(engine)
    _require_reliable_payable(engine)
    cov, window = engine._window(spec.period, as_of)
    if cov == "none":
        return Answer.unanswerable("period lies entirely outside the shop's data")
    order_ids = _one_customer_order_ids(engine, spec.customer)
    added = _debt_added_amount(engine, window, order_ids)
    paid = _debt_collected_amount(engine, window, order_ids)
    return Answer(
        type="list",
        answer=[
            render_metric("revenue", added, engine.rules),
            render_metric("revenue", paid, engine.rules),
        ],
    )


def _one_customer_order_ids(engine: MetricEngine, customer: str) -> set[str]:
    """Orders of the one customer whose code or name (case and accents ignored) is
    `customer`; abstain unless exactly one customer matches."""
    from datamate.metrics.engine import Unanswerable

    mode = _detect_customer_mode(engine)
    if mode is None:
        raise Unanswerable("customer")
    wanted = fold_text(customer)
    rows = engine.con.execute(
        """
        SELECT o.order_id, o.customer_id, o.customer_name, c.customer_name,
               COALESCE(o.is_walk_in, FALSE)
        FROM orders o LEFT JOIN customers c ON c.customer_id = o.customer_id
        """
    ).fetchall()
    by_ref: dict[str, set[str]] = {}
    for oid, cid, oname, cname, walk in rows:
        if walk:
            continue
        names = [n for n in (oname, cname) if isinstance(n, str)]
        if mode == "by_code":
            if cid is None:
                continue
            ref = str(cid)
            hit = fold_text(ref) == wanted or any(fold_text(n) == wanted for n in names)
        else:
            ref = _fold_customer_name(oname if isinstance(oname, str) else None) or ""
            hit = bool(ref) and ref == wanted
        if hit:
            by_ref.setdefault(ref, set()).add(str(oid))
    if len(by_ref) != 1:
        raise _Abstain()
    return next(iter(by_ref.values()))


def product_base_unit(engine: MetricEngine, product: str) -> str | None:
    """Base unit of a named product (None when unknown).

    products.base_unit first; else the line unit whose quantity equals qty_base; else the
    product's only line unit.
    """
    keys = _product_keys(engine, product)
    if not keys:
        return None
    marks = ", ".join("?" for _ in keys)
    params = sorted(keys)
    queries = (
        (
            f"SELECT base_unit FROM products WHERE product_key IN ({marks}) "
            "AND base_unit IS NOT NULL AND base_unit <> '' LIMIT 1"
        ),
        (
            f"SELECT unit FROM order_lines WHERE product_key IN ({marks}) "
            "AND unit IS NOT NULL AND unit <> '' AND qty_base IS NOT NULL AND qty_base = qty "
            "GROUP BY unit ORDER BY COUNT(*) DESC, unit LIMIT 1"
        ),
        (
            f"SELECT MIN(unit) FROM order_lines WHERE product_key IN ({marks}) "
            "AND unit IS NOT NULL AND unit <> '' HAVING COUNT(DISTINCT unit) = 1"
        ),
    )
    for sql in queries:
        try:
            row = engine.con.execute(sql, params).fetchone()
        except Exception:  # noqa: BLE001, S112 — minimal fixtures may lack these columns
            continue
        if row is not None and row[0] is not None and str(row[0]).strip():
            return str(row[0]).strip()
    return None

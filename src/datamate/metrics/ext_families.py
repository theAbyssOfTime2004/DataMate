"""E4a product-family computations on MetricEngine (SQL over canonical tables)."""

from __future__ import annotations

import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from datamate.metrics.format import render_metric, render_percent
from datamate.parse.text import normalize_label
from datamate.query.ext.spec import (
    ExtQuerySpec,
    NoneQuery,
    PriceIncreaseEffectQuery,
    PriceOnDateQuery,
    ProductsInMinOrdersQuery,
    QtySoldQuery,
    TopProductsQuery,
    UnsoldProductsQuery,
)
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
    except Unanswerable as exc:
        return Answer.unanswerable(str(exc))
    except _Abstain:
        return Answer.abstain()
    except PeriodError:
        return Answer.abstain()
    raise TypeError(f"unsupported ext spec {spec!r}")


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
    if spec.by == "qty_base" and not _ext_recorded(engine, "order_lines", "qty_base"):
        raise Unanswerable("qty_base")
    if spec.by == "gross_profit":
        if not _ext_recorded(engine, "order_lines", "cost_per_unit"):
            raise Unanswerable("cost_per_unit")
        if not _ext_recorded(engine, "order_lines", "qty"):
            raise Unanswerable("qty")
    where, params = _revenue_line_where(engine, window)
    where += f" AND {_non_product_sql(engine)}"
    where, params = _apply_size_filter(engine, spec.size, where, params)
    where, params = _apply_category_filter(engine, spec.category, where, params)
    if spec.by in {"qty_base", "gross_profit"} and _any_null_qty_base(engine, where, params):
        raise _Abstain()
    if spec.by == "gross_profit" and _any_null_cost(engine, where, params):
        raise _Abstain()
    if spec.by == "qty_base":
        measure = "SUM(l.qty_base)"
    elif spec.by == "revenue":
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

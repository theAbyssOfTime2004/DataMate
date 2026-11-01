"""Compile a QueryPlan into SQL + params + answer type (deterministic)."""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any, Literal

from datamate.metrics.engine import MetricEngine, Unanswerable
from datamate.metrics.ext_families import (
    _ensure_customer_orders,
    _ext_recorded,
    _non_product_sql,
    _require_product_keys,
    _revenue_order_predicates,
)
from datamate.query.plan.spec import PlanScope, QueryPlan
from datamate.timeres.periods import DateRange, PeriodError, resolve

AnswerTypeName = Literal["number", "percent", "string", "list", "date"]
# How execute renders a scalar: money rounds to dong, average to 2 decimals.
ValueKind = Literal["money", "count", "quantity", "average"]

_MONEY_MEASURES = frozenset({"revenue", "avg_order_value", "gross_profit", "purchase_amount"})
_LINE_MEASURES = frozenset({"revenue", "avg_order_value", "qty_base", "gross_profit"})
_TIME_ENTITIES = frozenset({"day", "weekday", "hour", "month"})


class CompileError(ValueError):
    """Plan cannot be compiled with the available blocks."""


@dataclass(frozen=True)
class CompiledPlan:
    sql: str
    params: list[Any]
    answer_type: AnswerTypeName
    # How to interpret the first column / row set.
    result_kind: Literal["scalar", "label", "label_list", "percent_scalar"]
    window: DateRange
    value_kind: ValueKind = "count"


def compile_plan(
    plan: QueryPlan,
    engine: MetricEngine,
    as_of: datetime.date,
) -> CompiledPlan:
    try:
        window = resolve(plan.filters.period, as_of, engine.rules, engine.cal)
    except PeriodError as exc:
        raise CompileError(str(exc)) from exc

    cov_start = engine.data_start or window.start
    cov_end = engine.data_end or window.end
    if window.end < cov_start or window.start > cov_end:
        raise Unanswerable("period lies entirely outside the shop's data")

    if (
        plan.entity == "hour"
        or plan.filters.hour_from is not None
        or plan.filters.hour_to is not None
    ) and not _ext_recorded(engine, "orders", "order_ts"):
        raise Unanswerable("order_ts")

    if plan.measure in {"receipt_count", "purchase_amount"}:
        return _compile_purchase(plan, engine, window)
    return _compile_sales(plan, engine, window)


def _value_kind(plan: QueryPlan) -> ValueKind:
    if plan.measure in _MONEY_MEASURES:
        return "money"
    if plan.output.kind == "avg_per_period":
        return "average"
    if plan.measure == "qty_base":
        return "quantity"
    return "count"


def _rank_answer_type(plan: QueryPlan) -> AnswerTypeName:
    """Answer type of argmax/argmin/top_k for the plan's entity."""
    kind = plan.output.kind
    if plan.entity == "weekday":
        raise CompileError("weekday label format is not defined")
    if kind == "argmin" and plan.entity in _TIME_ENTITIES:
        # Empty days / hours are candidates too; which ones the shop was open is unknown.
        raise CompileError("argmin over time buckets is not supported")
    if kind == "top_k":
        return "list"
    if plan.entity in {"day", "month"}:
        return "date"
    if plan.entity == "hour":
        return "number"
    return "string"


def _scope_order_sql(scope: PlanScope | None, engine: MetricEngine) -> tuple[str, list[Any]]:
    scope = scope or "revenue_orders"
    if scope == "revenue_orders":
        return _revenue_order_predicates(engine)
    if scope == "cancelled_orders":
        return "o.status = ?", ["cancelled"]
    return "TRUE", []


def _has_employee_names(engine: MetricEngine) -> bool:
    return _ext_recorded(engine, "employees", "employee_name")


def _compile_sales(plan: QueryPlan, engine: MetricEngine, window: DateRange) -> CompiledPlan:
    needs_lines = plan.measure in _LINE_MEASURES or plan.entity in {"product", "category"}
    needs_customers = plan.entity == "customer" or plan.measure == "distinct_customers"
    if needs_customers:
        _ensure_customer_orders(engine)
    if plan.entity == "category" and not _ext_recorded(engine, "products", "category"):
        raise Unanswerable("category")
    if plan.entity == "employee" and not _ext_recorded(engine, "orders", "employee_id"):
        raise Unanswerable("orders.employee_id")

    joins = ["orders o"]
    if needs_customers:
        joins.append("JOIN customer_orders c ON c.order_id = o.order_id")
    if needs_lines:
        joins.append("JOIN order_lines l ON l.order_id = o.order_id")
    if plan.entity == "category":
        joins.append("JOIN products p ON p.product_key = l.product_key")
    if plan.entity == "employee" and _has_employee_names(engine):
        joins.append("LEFT JOIN employees e ON e.employee_id = o.employee_id")
    from_sql = " ".join(joins)

    scope_sql, scope_params = _scope_order_sql(plan.scope, engine)
    where = f"o.order_date BETWEEN ? AND ? AND ({scope_sql})"
    params: list[Any] = [window.start, window.end, *scope_params]
    if needs_lines:
        where += f" AND {_non_product_sql(engine)}"
    base_where, base_params = where, list(params)
    extra, extra_params = _filter_sql(plan, engine, line_level=needs_lines)
    if extra:
        where += f" AND {extra}"
        params.extend(extra_params)

    measure_sql = _measure_sql(plan, engine, from_sql, where, params)
    value_kind = _value_kind(plan)
    if plan.entity == "none" and plan.output.kind == "share_of_total":
        # The filtered part over the same measure for the whole shop in the period.
        sql = f"""
            SELECT CASE WHEN t.total = 0 THEN NULL ELSE 100.0 * p.part / t.total END
            FROM (SELECT {measure_sql} AS part FROM {from_sql} WHERE {where}) p,
                 (SELECT {measure_sql} AS total FROM {from_sql} WHERE {base_where}) t
        """
        return CompiledPlan(
            sql, [*params, *base_params], "percent", "percent_scalar", window, value_kind
        )
    if plan.entity == "none":
        return _shop_level(plan, from_sql, measure_sql, where, params, window, value_kind)

    select, group = _entity_exprs(plan.entity, engine)
    inner = f"""
        SELECT {select},
               {measure_sql} AS m_value,
               COUNT(DISTINCT o.order_id) AS order_count
        FROM {from_sql}
        WHERE {where}
        GROUP BY {group}
    """
    return _wrap_grouped(plan, inner, params, window, value_kind)


def _measure_sql(
    plan: QueryPlan,
    engine: MetricEngine,
    from_sql: str,
    where: str,
    params: list[Any],
) -> str:
    measure = plan.measure
    if measure == "revenue":
        return "COALESCE(SUM(l.line_amount), 0)"
    if measure == "avg_order_value":
        return (
            "CASE WHEN COUNT(DISTINCT o.order_id) = 0 THEN NULL "
            "ELSE CAST(SUM(l.line_amount) AS DOUBLE) / COUNT(DISTINCT o.order_id) END"
        )
    if measure == "qty_base":
        if not _ext_recorded(engine, "order_lines", "qty_base"):
            raise Unanswerable("qty_base")
        _require_no_nulls(engine, from_sql, where, params, "l.qty_base")
        return "COALESCE(SUM(l.qty_base), 0)"
    if measure == "gross_profit":
        if not _ext_recorded(engine, "order_lines", "cost_per_unit"):
            raise Unanswerable("cost_per_unit")
        _require_no_nulls(engine, from_sql, where, params, "l.cost_per_unit")
        _require_no_nulls(engine, from_sql, where, params, "l.qty")
        return "COALESCE(SUM(l.line_amount - l.qty * l.cost_per_unit), 0)"
    if measure in {"order_count", "cancelled_order_count", "orders_containing"}:
        return "COUNT(DISTINCT o.order_id)"
    if measure == "returned_order_count":
        return "COUNT(DISTINCT CASE WHEN o.has_return THEN o.order_id END)"
    if measure == "distinct_order_days":
        return "COUNT(DISTINCT o.order_date)"
    if measure == "distinct_customers":
        return "COUNT(DISTINCT c.customer_ref)"
    raise CompileError(f"measure {measure} not supported for sales")


def _require_no_nulls(
    engine: MetricEngine, from_sql: str, where: str, params: list[Any], column: str
) -> None:
    row = engine.con.execute(
        f"SELECT 1 FROM {from_sql} WHERE {where} AND {column} IS NULL LIMIT 1", params
    ).fetchone()
    if row is not None:
        raise CompileError(f"{column} missing on some lines")


def _shop_level(
    plan: QueryPlan,
    from_sql: str,
    measure_sql: str,
    where: str,
    params: list[Any],
    window: DateRange,
    value_kind: ValueKind,
) -> CompiledPlan:
    kind = plan.output.kind
    if kind in {"argmax", "argmin", "top_k", "share_of_total", "count_entities"}:
        raise CompileError(f"output {kind} is invalid for entity=none")
    if kind != "avg_per_period":
        sql = f"SELECT {measure_sql} FROM {from_sql} WHERE {where}"
        return CompiledPlan(sql, params, "number", "scalar", window, value_kind)

    unit = plan.output.unit  # type: ignore[union-attr]
    if unit == "weekday_occurrence":
        weekday = plan.filters.weekday
        if weekday is None:
            raise CompileError("avg per weekday needs filters.weekday")
        n_days = _count_weekday(window, weekday)
        divisor = str(n_days)
    elif plan.output.include_zero_days:  # type: ignore[union-attr]
        n_days = (window.end - window.start).days + 1
        divisor = str(n_days)
    else:
        n_days = 1
        divisor = "COUNT(DISTINCT o.order_date)"
    if n_days == 0:
        raise CompileError("no matching days in the period")
    sql = f"""
        SELECT CASE WHEN {divisor} = 0 THEN NULL
                    ELSE CAST(({measure_sql}) AS DOUBLE) / {divisor} END
        FROM {from_sql}
        WHERE {where}
    """
    return CompiledPlan(sql, params, "number", "scalar", window, value_kind)


def _count_weekday(window: DateRange, weekday: int) -> int:
    days = (window.end - window.start).days + 1
    return sum(
        1
        for offset in range(days)
        if (window.start + datetime.timedelta(days=offset)).weekday() == weekday
    )


def _entity_exprs(entity: str, engine: MetricEngine) -> tuple[str, str]:
    """(select list with entity_key/entity_label, GROUP BY expr)."""
    if entity == "customer":
        return (
            "c.customer_ref AS entity_key, MAX(c.customer_label) AS entity_label",
            "c.customer_ref",
        )
    if entity == "product":
        return (
            (
                "l.product_key AS entity_key, "
                "COALESCE(MAX(l.product_name), MAX(l.product_id), l.product_key) AS entity_label"
            ),
            "l.product_key",
        )
    if entity == "category":
        return "p.category AS entity_key, p.category AS entity_label", "p.category"
    if entity == "order":
        return (
            "CAST(o.order_id AS VARCHAR) AS entity_key, CAST(o.order_id AS VARCHAR) AS entity_label",
            "o.order_id",
        )
    if entity == "day":
        return (
            "o.order_date AS entity_key, CAST(o.order_date AS VARCHAR) AS entity_label",
            "o.order_date",
        )
    if entity == "weekday":
        return (
            (
                "CAST(isodow(o.order_date) AS INTEGER) AS entity_key, "
                "CAST(isodow(o.order_date) AS VARCHAR) AS entity_label"
            ),
            "isodow(o.order_date)",
        )
    if entity == "hour":
        return (
            (
                "CAST(hour(o.order_ts) AS INTEGER) AS entity_key, "
                "CAST(hour(o.order_ts) AS INTEGER) AS entity_label"
            ),
            "hour(o.order_ts)",
        )
    if entity == "month":
        return (
            (
                "strftime(o.order_date, '%Y-%m') AS entity_key, "
                "strftime(o.order_date, '%Y-%m') AS entity_label"
            ),
            "strftime(o.order_date, '%Y-%m')",
        )
    if entity == "employee":
        label = "CAST(o.employee_id AS VARCHAR)"
        if _has_employee_names(engine):
            label = f"COALESCE(MAX(e.employee_name), {label})"
        return (
            f"CAST(o.employee_id AS VARCHAR) AS entity_key, {label} AS entity_label",
            "o.employee_id",
        )
    if entity == "payment_method":
        return (
            "o.payment_method AS entity_key, o.payment_method AS entity_label",
            "o.payment_method",
        )
    if entity == "channel":
        return "o.channel AS entity_key, o.channel AS entity_label", "o.channel"
    raise CompileError(f"entity {entity} not supported for sales")


def _filter_sql(
    plan: QueryPlan, engine: MetricEngine, *, line_level: bool
) -> tuple[str, list[Any]]:
    """Order filters; product/category apply to joined lines when line_level."""
    parts: list[str] = []
    params: list[Any] = []
    f = plan.filters
    if f.channel is not None:
        parts.append("o.channel = ?")
        params.append(f.channel)
    if f.payment_method is not None:
        parts.append("o.payment_method = ?")
        params.append(f.payment_method)
    if f.employee is not None:
        if _has_employee_names(engine):
            parts.append(
                "(CAST(o.employee_id AS VARCHAR) = ? OR o.employee_id IN "
                "(SELECT employee_id FROM employees WHERE lower(employee_name) = lower(?)))"
            )
            params.extend([f.employee, f.employee])
        else:
            parts.append("CAST(o.employee_id AS VARCHAR) = ?")
            params.append(f.employee)
    if f.weekday is not None:
        parts.append("isodow(o.order_date) = ?")
        params.append(f.weekday + 1)  # plan Mon=0; isodow Mon=1
    if f.hour_from is not None:
        parts.append("hour(o.order_ts) >= ?")
        params.append(f.hour_from)
    if f.hour_to is not None:
        parts.append("hour(o.order_ts) <= ?")
        params.append(f.hour_to)
    if f.customer is not None:
        parts.append("(CAST(o.customer_id AS VARCHAR) = ? OR lower(o.customer_name) = lower(?))")
        params.extend([f.customer, f.customer])
    if f.product is not None or f.category is not None:
        alias = "l" if line_level else "fl"
        line_parts: list[str] = []
        if f.product is not None:
            keys = _require_product_keys(engine, f.product)
            marks = ", ".join("?" for _ in keys)
            line_parts.append(f"{alias}.product_key IN ({marks})")
            params.extend(sorted(keys))
        if f.category is not None:
            if not _ext_recorded(engine, "products", "category"):
                raise Unanswerable("category")
            line_parts.append(
                "EXISTS (SELECT 1 FROM products fp "
                f"WHERE fp.product_key = {alias}.product_key AND fp.category = ?)"
            )
            params.append(f.category)
        if line_level:
            parts.extend(line_parts)
        else:
            inner = " AND ".join(
                ["fl.order_id = o.order_id", _non_product_sql(engine, line_alias="fl"), *line_parts]
            )
            parts.append(f"EXISTS (SELECT 1 FROM order_lines fl WHERE {inner})")
    return " AND ".join(parts), params


def _wrap_grouped(
    plan: QueryPlan,
    inner: str,
    params: list[Any],
    window: DateRange,
    value_kind: ValueKind,
) -> CompiledPlan:
    where_parts = ["entity_key IS NOT NULL"]
    having_params: list[Any] = []
    for h in plan.having:
        if h.measure == plan.measure:
            col = "m_value"
        elif h.measure == "order_count":
            col = "order_count"
        else:
            raise CompileError(f"having on {h.measure} needs measure {h.measure}")
        where_parts.append(f"{col} {h.op} ?")
        having_params.append(h.value)
    filter_sql = " WHERE " + " AND ".join(where_parts)

    grouped = f"SELECT * FROM ({inner}) AS g{filter_sql}"
    all_params = [*params, *having_params]
    kind = plan.output.kind
    tie = "entity_label ASC NULLS LAST, entity_key ASC NULLS LAST"

    if kind == "count_entities":
        sql = f"SELECT COUNT(*) FROM ({grouped}) AS c"
        return CompiledPlan(sql, all_params, "number", "scalar", window, "count")
    if kind == "value":
        sql = f"SELECT COALESCE(SUM(m_value), 0) FROM ({grouped}) AS c"
        return CompiledPlan(sql, all_params, "number", "scalar", window, value_kind)
    if kind == "share_of_total":
        sql = f"""
            WITH base AS (SELECT * FROM ({inner}) AS b WHERE entity_key IS NOT NULL),
                 filtered AS ({grouped})
            SELECT CASE WHEN (SELECT COALESCE(SUM(m_value), 0) FROM base) = 0 THEN NULL
                   ELSE 100.0 * (SELECT COALESCE(SUM(m_value), 0) FROM filtered)
                        / (SELECT SUM(m_value) FROM base) END
        """
        # inner params appear twice (base + filtered).
        return CompiledPlan(
            sql, [*params, *all_params], "percent", "percent_scalar", window, value_kind
        )
    if kind in {"argmax", "argmin", "top_k"}:
        answer_type = _rank_answer_type(plan)
        direction = "ASC" if kind == "argmin" else "DESC"
        limit = int(plan.output.k) if kind == "top_k" else 1  # type: ignore[union-attr]
        sql = f"""
            SELECT entity_label FROM ({grouped}) AS r
            WHERE entity_label IS NOT NULL
            ORDER BY m_value {direction}, {tie}
            LIMIT {limit}
        """
        result_kind = "label_list" if kind == "top_k" else "label"
        return CompiledPlan(sql, all_params, answer_type, result_kind, window, value_kind)
    raise CompileError(f"unsupported output {kind}")


def _compile_purchase(plan: QueryPlan, engine: MetricEngine, window: DateRange) -> CompiledPlan:
    f = plan.filters
    if any(
        value is not None
        for value in (
            f.product,
            f.category,
            f.customer,
            f.employee,
            f.channel,
            f.payment_method,
            f.weekday,
            f.hour_from,
            f.hour_to,
        )
    ):
        raise CompileError("purchase measures take no sales filters")
    if plan.output.kind in {"avg_per_period", "share_of_total"}:
        raise CompileError(f"output {plan.output.kind} unsupported for purchases")
    if not _ext_recorded(engine, "purchases", "receipt_date"):
        raise Unanswerable("purchases")
    if plan.measure == "purchase_amount" and not _ext_recorded(engine, "purchases", "total_amount"):
        raise Unanswerable("purchases.total_amount")
    where = "p.receipt_date BETWEEN ? AND ? AND (NOT COALESCE(p.is_cancelled, FALSE))"
    params: list[Any] = [window.start, window.end]
    measure_sql = (
        "COUNT(DISTINCT p.receipt_id)"
        if plan.measure == "receipt_count"
        else "COALESCE(SUM(p.total_amount), 0)"
    )
    value_kind = _value_kind(plan)
    if plan.entity == "none":
        sql = f"SELECT {measure_sql} FROM purchases p WHERE {where}"
        return CompiledPlan(sql, params, "number", "scalar", window, value_kind)
    if plan.entity == "supplier":
        key = "COALESCE(p.supplier_name, CAST(p.supplier_id AS VARCHAR))"
        select = f"{key} AS entity_key, {key} AS entity_label"
        group = key
    elif plan.entity == "day":
        select = "p.receipt_date AS entity_key, CAST(p.receipt_date AS VARCHAR) AS entity_label"
        group = "p.receipt_date"
    elif plan.entity == "month":
        key = "strftime(p.receipt_date, '%Y-%m')"
        select = f"{key} AS entity_key, {key} AS entity_label"
        group = key
    else:
        raise CompileError(f"entity {plan.entity} unsupported for purchases")
    inner = f"""
        SELECT {select}, {measure_sql} AS m_value,
               COUNT(DISTINCT p.receipt_id) AS order_count
        FROM purchases p
        WHERE {where}
        GROUP BY {group}
    """
    if any(h.measure == "order_count" for h in plan.having):
        raise CompileError("purchase plans have no order_count")
    return _wrap_grouped(plan, inner, params, window, value_kind)

from __future__ import annotations

import datetime
import itertools
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from datamate.metrics.format import (
    metric_answer_type,
    quantize_metric,
    render_diff,
    render_metric,
    render_percent,
)
from datamate.parse.text import normalize_label
from datamate.query.spec import (
    ArgMaxQuery,
    CompareQuery,
    Filters,
    Metric,
    MetricQuery,
    MultiQuery,
    QuerySpec,
    SeriesQuery,
    TrendQuery,
    ValueQuery,
)
from datamate.rules.model import RuleConfig
from datamate.scoring.answer import Answer
from datamate.timeres.calendar import LunarCalendar
from datamate.timeres.periods import (
    Coverage,
    DateRange,
    PeriodError,
    PeriodExpr,
    coverage,
    resolve,
)

_RECORDED_COLUMNS: frozenset[tuple[str, str]] = frozenset(
    {
        ("orders", "payment_method"),
        ("orders", "channel"),
        ("orders", "order_ts"),
        ("customers", "segment"),
        ("products", "category"),
        ("order_lines", "product_key"),
        ("order_lines", "product_id"),
    }
)


class Unanswerable(Exception):
    """The shop does not record the information the question needs (§7)."""

    def __init__(self, what: str) -> None:
        self.what = what
        super().__init__(f"shop does not record {what}")


class _DivisionByZero(Exception):
    pass


@dataclass
class MetricValue:
    value: Decimal | int | None
    range: DateRange
    coverage: Coverage
    sql: str


class MetricEngine:
    def __init__(
        self,
        con: Any,
        rules: RuleConfig,
        cal: LunarCalendar,
        data_start: datetime.date | None,
        data_end: datetime.date | None,
    ) -> None:
        self.con = con
        self.rules = rules
        self.cal = cal
        self.data_start = data_start
        self.data_end = data_end
        self._present: dict[tuple[str, str], bool] = {}

    def compute(self, query: MetricQuery, as_of: datetime.date) -> MetricValue:
        cov, window = self._window(query.period, as_of)
        if cov == "none":
            return MetricValue(value=None, range=window, coverage="none", sql="")
        self._require_recorded(query.filters)
        value, sql = self._metric_value(query.metric, window, query.filters)
        return MetricValue(value=value, range=window, coverage=cov, sql=sql)

    def answer(self, spec: QuerySpec, as_of: datetime.date) -> Answer:
        try:
            if isinstance(spec, ValueQuery):
                return self._answer_value(spec, as_of)
            if isinstance(spec, CompareQuery):
                return self._answer_compare(spec, as_of)
            if isinstance(spec, SeriesQuery):
                return self._answer_series(spec, as_of)
            if isinstance(spec, MultiQuery):
                return self._answer_multi(spec, as_of)
            if isinstance(spec, TrendQuery):
                return self._answer_trend(spec, as_of)
            if isinstance(spec, ArgMaxQuery):
                return self._answer_argmax(spec, as_of)
        except Unanswerable as exc:
            return Answer.unanswerable(str(exc))
        except PeriodError:
            return Answer.abstain()
        except _DivisionByZero:
            return Answer.abstain()
        raise TypeError(f"unsupported query spec {spec!r}")

    def _answer_value(self, spec: ValueQuery, as_of: datetime.date) -> Answer:
        found = self.compute(spec.query, as_of)
        if found.coverage == "none" or found.value is None:
            return Answer.unanswerable("period lies entirely outside the shop's data")
        return Answer(
            type=metric_answer_type(spec.query.metric),
            answer=render_metric(spec.query.metric, found.value, self.rules),
        )

    def _answer_compare(self, spec: CompareQuery, as_of: datetime.date) -> Answer:
        left = self.compute(spec.a, as_of)
        right = self.compute(spec.b, as_of)
        if left.coverage == "none" or right.coverage == "none":
            return Answer.unanswerable("period lies entirely outside the shop's data")
        if left.value is None or right.value is None:
            return Answer.abstain()
        if spec.op == "pct_change":
            if Decimal(right.value) == 0:
                return Answer.abstain()
            change = (Decimal(left.value) - Decimal(right.value)) / Decimal(right.value)
            percent = change * Decimal(100)
            return Answer(type="percent", answer=render_percent(percent, self.rules))
        return Answer(
            type=metric_answer_type(spec.a.metric),
            answer=render_diff(
                spec.a.metric, Decimal(left.value) - Decimal(right.value), self.rules
            ),
        )

    def _answer_series(self, spec: SeriesQuery, as_of: datetime.date) -> Answer:
        values: list[int | float] = []
        for period in spec.periods:
            query = MetricQuery(metric=spec.metric, period=period, filters=spec.filters)
            rendered = self._rendered(query, as_of)
            if not isinstance(rendered, (int, float)):
                return rendered
            values.append(rendered)
        return Answer(type="list", answer=values)

    def _answer_multi(self, spec: MultiQuery, as_of: datetime.date) -> Answer:
        values: list[int | float] = []
        for query in spec.queries:
            rendered = self._rendered(query, as_of)
            if not isinstance(rendered, (int, float)):
                return rendered
            values.append(rendered)
        return Answer(type="list", answer=values)

    def _answer_trend(self, spec: TrendQuery, as_of: datetime.date) -> Answer:
        amounts: list[Decimal] = []
        for period in spec.periods:
            query = MetricQuery(metric=spec.metric, period=period, filters=spec.filters)
            found = self.compute(query, as_of)
            if found.coverage == "none" or found.value is None:
                return Answer.unanswerable("period lies entirely outside the shop's data")
            amounts.append(quantize_metric(spec.metric, found.value, self.rules))
        return Answer(
            type="boolean",
            answer=_is_trend(amounts, spec.direction, spec.strict),
        )

    def _answer_argmax(self, spec: ArgMaxQuery, as_of: datetime.date) -> Answer:
        cov, window = self._window(spec.period, as_of)
        if cov == "none":
            return Answer.unanswerable("period lies entirely outside the shop's data")
        self._require_recorded(spec.filters)
        grain = self._best_grain(spec.metric, window, spec.filters, spec.granularity, spec.mode)
        if grain is None:
            return Answer.unanswerable("no matching orders in the period")
        return Answer(type="date", answer=grain)

    def _rendered(self, query: MetricQuery, as_of: datetime.date) -> Answer | int | float:
        found = self.compute(query, as_of)
        if found.coverage == "none" or found.value is None:
            return Answer.unanswerable("period lies entirely outside the shop's data")
        return render_metric(query.metric, found.value, self.rules)

    def _window(self, expr: PeriodExpr, as_of: datetime.date) -> tuple[Coverage, DateRange]:
        nominal = resolve(expr, as_of, self.rules, self.cal)
        if self.data_start is None or self.data_end is None:
            return "none", nominal
        cov, clipped = coverage(nominal, self.data_start, self.data_end)
        if cov == "none" or clipped is None:
            return "none", nominal
        return cov, clipped

    def _metric_value(
        self, metric: Metric, period: DateRange, filters: Filters
    ) -> tuple[Decimal | int, str]:
        if metric == "revenue":
            return self._revenue(period, filters)
        if metric == "order_count":
            return self._order_count(period, filters, revenue=True)
        if metric == "cancel_count":
            return self._cancel_count(period, filters)
        if metric == "cancel_rate":
            return self._cancel_rate(period, filters)
        if metric == "avg_order_value":
            return self._avg_order_value(period, filters)
        if metric == "revenue_share":
            return self._revenue_share(period, filters)
        raise TypeError(f"unsupported metric {metric}")

    def _revenue(self, period: DateRange, filters: Filters) -> tuple[int, str]:
        where, params = self._where(period, filters, revenue=True, joined_lines=True)
        sql = (
            "SELECT COALESCE(SUM(l.line_amount), 0) "
            "FROM order_lines l JOIN orders o ON o.order_id = l.order_id "
            f"WHERE {where}"
        )
        return _as_int(self._scalar(sql, params)), sql

    def _order_count(
        self, period: DateRange, filters: Filters, *, revenue: bool
    ) -> tuple[int, str]:
        where, params = self._where(period, filters, revenue=revenue, joined_lines=False)
        sql = f"SELECT COUNT(DISTINCT o.order_id) FROM orders o WHERE {where}"
        return _as_int(self._scalar(sql, params)), sql

    def _cancel_count(self, period: DateRange, filters: Filters) -> tuple[int, str]:
        where, params = self._where(period, filters, revenue=False, joined_lines=False)
        sql = (
            "SELECT COUNT(DISTINCT o.order_id) FROM orders o "
            f"WHERE {where} AND o.status = ?"
        )
        return _as_int(self._scalar(sql, [*params, "cancelled"])), sql

    def _cancel_rate(self, period: DateRange, filters: Filters) -> tuple[Decimal, str]:
        cancelled, cancel_sql = self._cancel_count(period, filters)
        total, total_sql = self._order_count(period, filters, revenue=False)
        if total == 0:
            raise _DivisionByZero()
        return (Decimal(cancelled) * Decimal(100)) / Decimal(total), f"{cancel_sql}; {total_sql}"

    def _avg_order_value(self, period: DateRange, filters: Filters) -> tuple[Decimal, str]:
        revenue, revenue_sql = self._revenue(period, filters)
        count, count_sql = self._order_count(period, filters, revenue=True)
        if count == 0:
            raise _DivisionByZero()
        return Decimal(revenue) / Decimal(count), f"{revenue_sql}; {count_sql}"

    def _revenue_share(self, period: DateRange, filters: Filters) -> tuple[Decimal, str]:
        part, part_sql = self._revenue(period, filters)
        whole, whole_sql = self._revenue(period, Filters())
        if whole == 0:
            raise _DivisionByZero()
        return (Decimal(part) * Decimal(100)) / Decimal(whole), f"{part_sql}; {whole_sql}"

    def _best_grain(
        self,
        metric: Metric,
        period: DateRange,
        filters: Filters,
        granularity: str,
        mode: str,
    ) -> str | None:
        grain = "%Y-%m-%d" if granularity == "day" else "%Y-%m"
        rows = self._grain_rows(metric, period, filters, grain)
        if mode == "min" and granularity == "day":
            rows = _fill_zero_days(period, rows)
        if not rows:
            return None

        def sort_key(row: tuple[str, Decimal]) -> tuple[Decimal, str]:
            rank = -row[1] if mode == "max" else row[1]
            return (rank, row[0])

        return min(rows, key=sort_key)[0]

    def _grain_rows(
        self, metric: Metric, period: DateRange, filters: Filters, grain: str
    ) -> list[tuple[str, Decimal]]:
        label = f"strftime(o.order_date, '{grain}')"
        if metric == "revenue":
            where, params = self._where(period, filters, revenue=True, joined_lines=True)
            sql = (
                f"SELECT {label} AS grain, COALESCE(SUM(l.line_amount), 0) AS amount "
                "FROM order_lines l JOIN orders o ON o.order_id = l.order_id "
                f"WHERE {where} GROUP BY 1"
            )
            return [(str(row[0]), Decimal(row[1])) for row in self.con.execute(sql, params).fetchall()]
        if metric in {"order_count", "cancel_count"}:
            where, params = self._where(period, filters, revenue=metric == "order_count", joined_lines=False)
            status_sql = " AND o.status = ?" if metric == "cancel_count" else ""
            status_params = ["cancelled"] if metric == "cancel_count" else []
            sql = (
                f"SELECT {label} AS grain, COUNT(DISTINCT o.order_id) AS amount "
                f"FROM orders o WHERE {where}{status_sql} GROUP BY 1"
            )
            fetched = self.con.execute(sql, [*params, *status_params]).fetchall()
            return [(str(row[0]), Decimal(row[1])) for row in fetched]
        if metric == "avg_order_value":
            where, params = self._where(period, filters, revenue=True, joined_lines=True)
            sql = (
                f"SELECT {label} AS grain, COALESCE(SUM(l.line_amount), 0), "
                "COUNT(DISTINCT o.order_id) "
                "FROM order_lines l JOIN orders o ON o.order_id = l.order_id "
                f"WHERE {where} GROUP BY 1"
            )
            out: list[tuple[str, Decimal]] = []
            for grain_id, total, count in self.con.execute(sql, params).fetchall():
                if int(count) == 0:
                    continue
                out.append((str(grain_id), Decimal(total) / Decimal(int(count))))
            return out
        if metric == "cancel_rate":
            return self._rate_rows(period, filters, label, share=False)
        if metric == "revenue_share":
            return self._rate_rows(period, filters, label, share=True)
        raise TypeError(f"unsupported metric {metric}")

    def _rate_rows(
        self, period: DateRange, filters: Filters, label: str, *, share: bool
    ) -> list[tuple[str, Decimal]]:
        if share:
            part_where, part_params = self._where(period, filters, revenue=True, joined_lines=True)
            whole_where, whole_params = self._where(period, Filters(), revenue=True, joined_lines=True)
            part_sql = (
                f"SELECT {label} AS grain, COALESCE(SUM(l.line_amount), 0) "
                "FROM order_lines l JOIN orders o ON o.order_id = l.order_id "
                f"WHERE {part_where} GROUP BY 1"
            )
            whole_sql = (
                f"SELECT {label} AS grain, COALESCE(SUM(l.line_amount), 0) "
                "FROM order_lines l JOIN orders o ON o.order_id = l.order_id "
                f"WHERE {whole_where} GROUP BY 1"
            )
            parts = {str(row[0]): Decimal(row[1]) for row in self.con.execute(part_sql, part_params).fetchall()}
            wholes = {
                str(row[0]): Decimal(row[1]) for row in self.con.execute(whole_sql, whole_params).fetchall()
            }
            return [
                (grain, parts.get(grain, Decimal(0)) * Decimal(100) / total)
                for grain, total in wholes.items()
                if total != 0
            ]
        where, params = self._where(period, filters, revenue=False, joined_lines=False)
        sql = (
            f"SELECT {label} AS grain, "
            "COUNT(DISTINCT CASE WHEN o.status = ? THEN o.order_id END), "
            "COUNT(DISTINCT o.order_id) "
            f"FROM orders o WHERE {where} GROUP BY 1"
        )
        out: list[tuple[str, Decimal]] = []
        for grain, cancelled, total in self.con.execute(sql, ["cancelled", *params]).fetchall():
            if int(total) == 0:
                continue
            out.append((str(grain), Decimal(int(cancelled)) * Decimal(100) / Decimal(int(total))))
        return out

    def _where(
        self,
        period: DateRange,
        filters: Filters,
        *,
        revenue: bool,
        joined_lines: bool,
    ) -> tuple[str, list[Any]]:
        parts = ["o.order_date BETWEEN ? AND ?"]
        params: list[Any] = [period.start, period.end]
        if revenue:
            if not self.rules.revenue_statuses:
                parts.append("FALSE")
            else:
                marks = ", ".join("?" for _ in self.rules.revenue_statuses)
                parts.append(f"o.status IN ({marks})")
                params.extend(self.rules.revenue_statuses)
            if self.rules.exclude_orders_with_return:
                parts.append("NOT o.has_return")
        parts.extend(self._order_filter_parts(filters, params))
        line_sql, line_params = self._line_predicates(filters)
        if line_sql:
            if joined_lines:
                parts.append(line_sql)
            else:
                parts.append(
                    "EXISTS (SELECT 1 FROM order_lines l "
                    f"WHERE l.order_id = o.order_id AND {line_sql})"
                )
            params.extend(line_params)
        return " AND ".join(parts), params

    def _order_filter_parts(self, filters: Filters, params: list[Any]) -> list[str]:
        parts: list[str] = []
        if filters.payment_method:
            marks = ", ".join("?" for _ in filters.payment_method)
            parts.append(f"o.payment_method IN ({marks})")
            params.extend(filters.payment_method)
        if filters.channel:
            marks = ", ".join("?" for _ in filters.channel)
            parts.append(f"o.channel IN ({marks})")
            params.extend(filters.channel)
        if filters.customer_segment:
            marks = ", ".join("?" for _ in filters.customer_segment)
            parts.append(
                "EXISTS (SELECT 1 FROM customers c WHERE c.customer_key = o.customer_key "
                f"AND c.segment IN ({marks}))"
            )
            params.extend(filters.customer_segment)
        if filters.shift is not None:
            start, end = self.rules.shifts[filters.shift]
            parts.append("CAST(o.order_ts AS TIME) BETWEEN ? AND ?")
            params.extend([start, end])
        return parts

    def _line_predicates(self, filters: Filters) -> tuple[str, list[Any]]:
        parts: list[str] = []
        params: list[Any] = []
        if filters.product:
            arms: list[str] = []
            for raw in filters.product:
                label = normalize_label(raw)
                if label is None:
                    continue
                arms.append("(l.product_key = ? OR l.product_id = ?)")
                params.extend([label, raw])
            parts.append("(" + " OR ".join(arms) + ")" if arms else "FALSE")
        if filters.category:
            marks = ", ".join("?" for _ in filters.category)
            parts.append(
                "EXISTS (SELECT 1 FROM products p WHERE p.product_key = l.product_key "
                f"AND p.category IN ({marks}))"
            )
            params.extend(filters.category)
        if not parts:
            return "", []
        return " AND ".join(parts), params

    def _require_recorded(self, filters: Filters) -> None:
        if filters.payment_method and not self._recorded("orders", "payment_method"):
            raise Unanswerable("payment_method")
        if filters.channel and not self._recorded("orders", "channel"):
            raise Unanswerable("channel")
        if filters.customer_segment and not self._recorded("customers", "segment"):
            raise Unanswerable("customer segment")
        if filters.category and not self._recorded("products", "category"):
            raise Unanswerable("category")
        if filters.product and not (
            self._recorded("order_lines", "product_key")
            or self._recorded("order_lines", "product_id")
        ):
            raise Unanswerable("product")
        if filters.shift is not None and not self._recorded("orders", "order_ts"):
            raise Unanswerable("shift")

    def _recorded(self, table: str, column: str) -> bool:
        key = (table, column)
        if key not in _RECORDED_COLUMNS:
            raise Unanswerable(column)
        cached = self._present.get(key)
        if cached is not None:
            return cached
        sql = f'SELECT 1 FROM "{table}" WHERE "{column}" IS NOT NULL LIMIT 1'
        present = self.con.execute(sql).fetchone() is not None
        self._present[key] = present
        return present

    def _scalar(self, sql: str, params: list[Any]) -> Any:
        row = self.con.execute(sql, params).fetchone()
        if row is None:
            return None
        return row[0]


def _fill_zero_days(
    period: DateRange, rows: list[tuple[str, Decimal]]
) -> list[tuple[str, Decimal]]:
    by_day = {grain: amount for grain, amount in rows}
    filled: list[tuple[str, Decimal]] = []
    day = period.start
    while day <= period.end:
        filled.append((day.isoformat(), by_day.get(day.isoformat(), Decimal(0))))
        day += datetime.timedelta(days=1)
    return filled


def _as_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, Decimal):
        return int(value.to_integral_value())
    return int(value)


def _is_trend(values: list[Decimal], direction: str, strict: bool) -> bool:
    pairs = itertools.pairwise(values)
    if direction == "increasing":
        return all(left < right if strict else left <= right for left, right in pairs)
    return all(left > right if strict else left >= right for left, right in pairs)

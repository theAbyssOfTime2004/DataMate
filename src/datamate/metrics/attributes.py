"""Attribute block computations over orders and their text columns (order_attributes)."""

from __future__ import annotations

import datetime
from collections import Counter, defaultdict
from decimal import Decimal
from typing import TYPE_CHECKING

from datamate.metrics.ext_families import _Abstain, _ext_recorded, _revenue_order_predicates
from datamate.metrics.format import render_metric, render_percent
from datamate.query.attributes import AttrCondition, AttrQuery, ShopAttribute
from datamate.query.grounding import fold
from datamate.scoring.answer import Answer
from datamate.timeres.periods import PeriodError

if TYPE_CHECKING:
    from datamate.metrics.engine import MetricEngine

_SAMPLES = 6


def shop_attributes(engine: MetricEngine, docs: dict[str, str]) -> list[ShopAttribute]:
    """The shop's text columns with their README description and common values."""
    if not _ext_recorded(engine, "order_attributes", "name"):
        return []
    names = [
        row[0]
        for row in engine.con.execute(
            "SELECT name FROM order_attributes GROUP BY name ORDER BY COUNT(*) DESC, name"
        ).fetchall()
    ]
    out: list[ShopAttribute] = []
    for name in names:
        samples = engine.con.execute(
            "SELECT value FROM order_attributes WHERE name = ? "
            "GROUP BY value ORDER BY COUNT(*) DESC, value LIMIT ?",
            [name, _SAMPLES],
        ).fetchall()
        out.append(ShopAttribute(name, docs.get(name), tuple(str(r[0]) for r in samples)))
    if _ext_recorded(engine, "order_measures", "name"):
        for (name,) in engine.con.execute(
            "SELECT name FROM order_measures GROUP BY name ORDER BY COUNT(*) DESC, name"
        ).fetchall():
            samples = engine.con.execute(
                "SELECT value FROM order_measures WHERE name = ? "
                "GROUP BY value ORDER BY COUNT(*) DESC, value LIMIT ?",
                [name, _SAMPLES],
            ).fetchall()
            shown = tuple(_number_text(r[0]) for r in samples)
            out.append(ShopAttribute(str(name), docs.get(str(name)), shown, numeric=True))
    return out


def _number_text(value: object) -> str:
    text = format(Decimal(str(value)).normalize(), "f")
    return text


def _matches(values: list[str], cond: AttrCondition) -> bool:
    wanted = f" {fold(cond.contains)} "
    found = any(wanted in f" {value} " for value in values)
    return not found if cond.negate else found


def answer_attr(engine: MetricEngine, spec: AttrQuery, as_of: datetime.date) -> Answer:
    from datamate.metrics.engine import Unanswerable

    try:
        cov, window = engine._window(spec.period, as_of)
        if cov == "none":
            return Answer.unanswerable("period lies entirely outside the shop's data")
        where, params = _revenue_order_predicates(engine)
        where += " AND o.order_date BETWEEN ? AND ?"
        params = [*params, window.start, window.end]
        # A condition no record of the whole book meets is written in other words than the
        # shop uses ("dầu láp" vs "nhớt láp"): a count of 0 would be a guess.
        for cond in spec.where:
            if not cond.negate and not _found_anywhere(engine, cond):
                raise _Abstain()
        if spec.staff is not None:
            if not _ext_recorded(engine, "orders", "employee_id"):
                raise Unanswerable("orders.employee_id")
            where, params = _staff_filter(engine, spec.staff, where, params)
        money: dict[str, int] = {}
        dates: dict[str, datetime.date] = {}
        for oid, day, amount in engine.con.execute(
            f"SELECT o.order_id, o.order_date, COALESCE(SUM(l.line_amount), 0) FROM orders o "
            f"LEFT JOIN order_lines l ON l.order_id = o.order_id WHERE {where} "
            "GROUP BY o.order_id, o.order_date",
            params,
        ).fetchall():
            money[str(oid)] = int(amount or 0)
            dates[str(oid)] = day
        if not money:
            raise _Abstain()
        values: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        for oid, name, value in engine.con.execute(
            f"SELECT a.order_id, a.name, a.value FROM order_attributes a "
            f"JOIN orders o ON o.order_id = a.order_id WHERE {where}",
            params,
        ).fetchall():
            values[str(oid)][str(name)].append(fold(str(value)))
        text_value_of = None if spec.measure == "total" else spec.value_of
        used = {c.attribute for c in spec.where} | {spec.distinct_of, text_value_of} - {None}
        for name in used:
            if _sparse_in_period(engine, str(name), values, list(money)):
                raise _Abstain()
        if spec.weekday is not None:
            money = {oid: v for oid, v in money.items() if dates[oid].isoweekday() == spec.weekday}
        chosen = [
            oid
            for oid in money
            if all(_matches(values[oid].get(c.attribute, []), c) for c in spec.where)
        ]
        if spec.measure in {"count", "money"}:
            total = len(chosen) if spec.measure == "count" else sum(money[oid] for oid in chosen)
            if spec.per_weekday:
                assert spec.weekday is not None
                days = sum(
                    1
                    for i in range((window.end - window.start).days + 1)
                    if (window.start + datetime.timedelta(days=i)).isoweekday() == spec.weekday
                )
                if days == 0:
                    raise _Abstain()
                mean = Decimal(total) / Decimal(days)
                # An average of money is money (whole đồng); other averages keep 2 decimals.
                if spec.measure == "money":
                    return Answer(
                        type="number", answer=render_metric("avg_order_value", mean, engine.rules)
                    )
                return Answer(type="number", answer=render_percent(mean, engine.rules))
            metric = "order_count" if spec.measure == "count" else "revenue"
            return Answer(type="number", answer=render_metric(metric, total, engine.rules))
        if spec.measure == "top_value":
            return _top_value(engine, spec, chosen, money, values)
        if spec.measure == "total":
            return _total(engine, spec, chosen)
        if spec.measure == "distinct":
            assert spec.distinct_of is not None
            seen = {v for oid in chosen for v in values[oid].get(spec.distinct_of, [])}
            return Answer(
                type="number", answer=render_metric("order_count", len(seen), engine.rules)
            )
        base = [oid for oid in money if money[oid] > 0] if spec.share_of_paid else list(money)
        if not base:
            raise _Abstain()
        share = Decimal(len(chosen)) * Decimal(100) / Decimal(len(base))
        return Answer(type="percent", answer=render_percent(share, engine.rules))
    except Unanswerable as exc:
        return Answer.unanswerable(str(exc))
    except (_Abstain, PeriodError):
        return Answer.abstain()


def _total(engine: MetricEngine, spec: AttrQuery, chosen: list[str]) -> Answer:
    """Sum of one number column over the chosen records; a record without it abstains."""
    wanted = set(chosen)
    if not wanted:
        raise _Abstain()
    sums: dict[str, Decimal] = {}
    for oid, value in engine.con.execute(
        "SELECT order_id, value FROM order_measures WHERE name = ?", [spec.value_of]
    ).fetchall():
        key = str(oid)
        if key in wanted:
            sums[key] = sums.get(key, Decimal(0)) + Decimal(str(value))
    # Most chosen records must carry the number, else the column is about something else.
    if len(sums) < 0.9 * len(wanted):
        raise _Abstain()
    total = sum(sums.values(), Decimal(0))
    if total == total.to_integral_value():
        return Answer(type="number", answer=int(total))
    return Answer(type="number", answer=render_percent(total, engine.rules))


def _staff_filter(
    engine: MetricEngine, staff: str, where: str, params: list[object]
) -> tuple[str, list[object]]:
    from datamate.metrics.rank import _staff_names

    names = _staff_names(engine)
    written = {
        str(r[0])
        for r in engine.con.execute(
            "SELECT DISTINCT employee_id FROM orders WHERE employee_id IS NOT NULL"
        ).fetchall()
    }
    wanted = fold(staff)
    matches = {
        emp
        for emp in written | set(names)
        if fold(emp) == wanted or fold(names.get(emp, "")) == wanted
    }
    if len(matches) != 1:
        raise _Abstain()
    return f"{where} AND o.employee_id = ?", [*params, matches.pop()]


def _found_anywhere(engine: MetricEngine, cond: AttrCondition) -> bool:
    wanted = f" {fold(cond.contains)} "
    for (value,) in engine.con.execute(
        "SELECT DISTINCT value FROM order_attributes WHERE name = ?", [cond.attribute]
    ).fetchall():
        if wanted in f" {fold(str(value))} ":
            return True
    return False


# A column with more different values than this holds free text (plates, job lists), not a
# small set of kinds, so "which one is most common" has no meaningful answer.
_MAX_KINDS = 30


def _top_value(
    engine: MetricEngine,
    spec: AttrQuery,
    chosen: list[str],
    money: dict[str, int],
    values: dict[str, dict[str, list[str]]],
) -> Answer:
    assert spec.value_of is not None
    kinds = engine.con.execute(
        "SELECT value FROM order_attributes WHERE name = ?", [spec.value_of]
    ).fetchall()
    spellings: dict[str, Counter[str]] = defaultdict(Counter)
    for (value,) in kinds:
        spellings[fold(str(value))][str(value)] += 1
    if not spellings or len(spellings) > _MAX_KINDS:
        raise _Abstain()
    # Cells that list several items ("cắt, gội") are not one kind each.
    written = [value for counter in spellings.values() for value in counter]
    if sum("," in value for value in written) * 5 >= len(written):
        raise _Abstain()
    totals: Counter[str] = Counter()
    for oid in chosen:
        for value in set(values[oid].get(spec.value_of, [])):
            totals[value] += money[oid] if spec.top_by == "money" else 1
    if not totals:
        raise _Abstain()
    best = min(totals, key=lambda v: (-totals[v], v))
    return Answer(type="string", answer=spellings[best].most_common(1)[0][0])


def _sparse_in_period(
    engine: MetricEngine,
    name: str,
    values: dict[str, dict[str, list[str]]],
    period_orders: list[str],
) -> bool:
    """True when the column is filled far less often in the period than in the whole book
    (a month the owner stopped writing it): counts over it would be short."""
    total_orders = engine._scalar("SELECT COUNT(*) FROM orders", [])
    filled = engine._scalar(
        "SELECT COUNT(DISTINCT order_id) FROM order_attributes WHERE name = ?", [name]
    )
    if not total_orders or not period_orders:
        return False
    overall = int(filled or 0) / int(total_orders)
    in_period = sum(1 for oid in period_orders if values[oid].get(name)) / len(period_orders)
    return in_period < overall / 2

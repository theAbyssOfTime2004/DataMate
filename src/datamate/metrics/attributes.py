"""Attribute block computations over orders and their text columns (order_attributes)."""

from __future__ import annotations

import datetime
from collections import defaultdict
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
    return out


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
        money = {
            str(oid): int(amount or 0)
            for oid, amount in engine.con.execute(
                f"SELECT o.order_id, COALESCE(SUM(l.line_amount), 0) FROM orders o "
                f"LEFT JOIN order_lines l ON l.order_id = o.order_id WHERE {where} "
                "GROUP BY o.order_id",
                params,
            ).fetchall()
        }
        if not money:
            raise _Abstain()
        values: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        for oid, name, value in engine.con.execute(
            f"SELECT a.order_id, a.name, a.value FROM order_attributes a "
            f"JOIN orders o ON o.order_id = a.order_id WHERE {where}",
            params,
        ).fetchall():
            values[str(oid)][str(name)].append(fold(str(value)))
        chosen = [
            oid
            for oid in money
            if all(_matches(values[oid].get(c.attribute, []), c) for c in spec.where)
        ]
        if spec.measure == "count":
            return Answer(
                type="number", answer=render_metric("order_count", len(chosen), engine.rules)
            )
        if spec.measure == "money":
            total = sum(money[oid] for oid in chosen)
            return Answer(type="number", answer=render_metric("revenue", total, engine.rules))
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

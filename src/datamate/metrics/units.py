"""Unit block computations: records (orders) or their money for the shop or staff."""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

from datamate.metrics.ext_families import _Abstain, _ext_recorded, _revenue_order_predicates
from datamate.metrics.format import render_metric
from datamate.query.grounding import fold
from datamate.query.units import UnitQuery
from datamate.scoring.answer import Answer
from datamate.timeres.periods import PeriodError

if TYPE_CHECKING:
    from datamate.metrics.engine import MetricEngine


def answer_unit(engine: MetricEngine, spec: UnitQuery, as_of: datetime.date) -> Answer:
    from datamate.metrics.engine import Unanswerable

    try:
        cov, window = engine._window(spec.period, as_of)
        if cov == "none":
            return Answer.unanswerable("period lies entirely outside the shop's data")
        where, params = _revenue_order_predicates(engine)
        where += " AND o.order_date BETWEEN ? AND ?"
        params = [*params, window.start, window.end]
        measure = (
            "COUNT(DISTINCT o.order_id)"
            if spec.measure == "count"
            else ("COALESCE(SUM(l.line_amount), 0)")
        )
        lines = (
            "" if spec.measure == "count" else "LEFT JOIN order_lines l ON l.order_id = o.order_id"
        )
        metric = "order_count" if spec.measure == "count" else "revenue"
        if spec.pick == "shop":
            total = engine._scalar(f"SELECT {measure} FROM orders o {lines} WHERE {where}", params)
            return Answer(
                type="number", answer=render_metric(metric, int(total or 0), engine.rules)
            )
        if not _ext_recorded(engine, "orders", "employee_id"):
            raise Unanswerable("orders.employee_id")
        names = _staff_names(engine)
        rows = engine.con.execute(
            f"SELECT o.employee_id, {measure} FROM orders o {lines} "
            f"WHERE {where} AND o.employee_id IS NOT NULL GROUP BY o.employee_id",
            params,
        ).fetchall()
        if spec.pick == "staff":
            wanted = fold(spec.staff or "")
            matches = {
                emp
                for emp in _all_staff(engine, names)
                if fold(str(emp)) == wanted or fold(names.get(str(emp), "")) == wanted
            }
            if len(matches) != 1:
                raise _Abstain()
            emp = matches.pop()
            value = next((int(v or 0) for e, v in rows if str(e) == str(emp)), 0)
            return Answer(type="number", answer=render_metric(metric, value, engine.rules))
        if not rows:
            raise _Abstain()
        emp, _value = min(rows, key=lambda r: (-int(r[1] or 0), str(r[0])))
        return Answer(type="string", answer=names.get(str(emp)) or str(emp))
    except Unanswerable as exc:
        return Answer.unanswerable(str(exc))
    except (_Abstain, PeriodError):
        return Answer.abstain()


def _staff_names(engine: MetricEngine) -> dict[str, str]:
    from datamate.metrics.rank import _staff_names as names

    return names(engine)


def _all_staff(engine: MetricEngine, names: dict[str, str]) -> set[str]:
    written = {
        str(row[0])
        for row in engine.con.execute(
            "SELECT DISTINCT employee_id FROM orders WHERE employee_id IS NOT NULL"
        ).fetchall()
    }
    return written | set(names)

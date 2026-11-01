"""Run a compiled plan with a wall-clock SQL timeout and format the Answer."""

from __future__ import annotations

import datetime
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from datamate.config import settings
from datamate.metrics.engine import MetricEngine, Unanswerable
from datamate.metrics.ext_families import _Abstain
from datamate.metrics.format import render_metric, render_percent
from datamate.query.plan.compile import CompiledPlan, CompileError, compile_plan
from datamate.query.plan.spec import QueryPlan
from datamate.scoring.answer import Answer


class PlanExecuteError(RuntimeError):
    pass


def answer_plan(
    plan: QueryPlan,
    engine: MetricEngine,
    as_of: datetime.date,
    *,
    timeout_s: float | None = None,
) -> tuple[Answer, CompiledPlan]:
    compiled = compile_plan(plan, engine, as_of)
    limit = settings.plan_sql_timeout_s if timeout_s is None else timeout_s
    rows = _run_sql(engine, compiled.sql, compiled.params, limit)
    answer = _format(compiled, rows, engine)
    return answer, compiled


def _run_sql(
    engine: MetricEngine, sql: str, params: list[Any], timeout_s: float
) -> list[tuple[Any, ...]]:
    def _go() -> list[tuple[Any, ...]]:
        return list(engine.con.execute(sql, params).fetchall())

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_go)
        try:
            return future.result(timeout=timeout_s)
        except FuturesTimeout as exc:
            # Leaving the pool waits for the worker, so stop the query first.
            engine.con.interrupt()
            raise PlanExecuteError(f"SQL exceeded {timeout_s}s") from exc


def _format(compiled: CompiledPlan, rows: list[tuple[Any, ...]], engine: MetricEngine) -> Answer:
    if not rows or rows[0][0] is None:
        return Answer.abstain()
    if compiled.result_kind == "label":
        raw = rows[0][0]
        if compiled.answer_type == "number":
            return Answer(type="number", answer=int(raw))
        return Answer(type=compiled.answer_type, answer=str(raw))
    if compiled.result_kind == "label_list":
        labels = [str(r[0]) for r in rows if r and r[0] is not None]
        return Answer(type="list", answer=labels)
    if compiled.result_kind == "percent_scalar":
        return Answer(type="percent", answer=render_percent(Decimal(str(rows[0][0])), engine.rules))
    raw = rows[0][0]
    if isinstance(raw, bool):
        return Answer.abstain()
    return Answer(type="number", answer=_render_number(Decimal(str(raw)), compiled, engine))


def _render_number(value: Decimal, compiled: CompiledPlan, engine: MetricEngine) -> int | float:
    """Money rounds to dong; other averages and fractional quantities to 2 decimals."""
    if compiled.value_kind == "money":
        return render_metric("revenue", value, engine.rules)
    if compiled.value_kind == "count" or value == value.to_integral_value():
        return int(value.to_integral_value(rounding=ROUND_HALF_UP))
    return float(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def try_answer_plan(
    plan: QueryPlan,
    engine: MetricEngine,
    as_of: datetime.date,
) -> tuple[Answer, CompiledPlan | None, str | None]:
    """Return (answer, compiled|None, abstain_reason|None)."""
    try:
        answer, compiled = answer_plan(plan, engine, as_of)
        return answer, compiled, None
    except Unanswerable as exc:
        return Answer.unanswerable(str(exc)), None, None
    except CompileError as exc:
        return Answer.abstain(), None, f"plan_compile:{exc}"
    except _Abstain:
        return Answer.abstain(), None, "plan_engine_abstain"
    except PlanExecuteError as exc:
        return Answer.abstain(), None, f"plan_sql:{exc}"
    except Exception as exc:  # noqa: BLE001
        return Answer.abstain(), None, f"plan_error:{type(exc).__name__}:{exc}"

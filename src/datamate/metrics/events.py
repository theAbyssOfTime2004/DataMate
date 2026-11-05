"""Counts over per-order event logs: delivery batches today, any dated "lần" later.

An event source names its table, the event date column and the columns that tell two
events of one order apart. Both counts read the same description, so a new kind of event
(a repair visit, an instalment) is a new EventSource, not new SQL.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import TYPE_CHECKING

from datamate.metrics.format import render_metric
from datamate.query.ext.spec import DeliveryCountQuery, OrdersMinDeliveriesQuery
from datamate.scoring.answer import Answer

if TYPE_CHECKING:
    from datamate.metrics.engine import MetricEngine


@dataclass(frozen=True)
class EventSource:
    table: str
    date_column: str
    # One event per order and this column; rows without it fall back to the event date.
    event_column: str


DELIVERY = EventSource(table="deliveries", date_column="delivery_date", event_column="batch_no")


def has_event_log(engine: MetricEngine) -> bool:
    """True when the shop keeps at least one dated event (a delivery batch)."""
    from datamate.metrics.ext_families import _ext_recorded

    return _ext_recorded(engine, DELIVERY.table, DELIVERY.date_column)


def answer_event_family(engine: MetricEngine, spec: object, as_of: datetime.date) -> Answer | None:
    """Answer an event family; None when `spec` is not one."""
    if isinstance(spec, DeliveryCountQuery):
        return _event_count(engine, DELIVERY, spec, as_of)
    if isinstance(spec, OrdersMinDeliveriesQuery):
        return _orders_min_events(engine, DELIVERY, spec, as_of)
    return None


def _event_key(source: EventSource) -> str:
    # Each batch of each order is one event, also when two batches share a day.
    return (
        f"e.order_id || '|' || COALESCE(CAST(e.{source.event_column} AS VARCHAR), "
        f"CAST(e.{source.date_column} AS VARCHAR))"
    )


def _require_events(engine: MetricEngine, source: EventSource) -> None:
    from datamate.metrics.engine import Unanswerable
    from datamate.metrics.ext_families import _ext_recorded

    if not _ext_recorded(engine, source.table, source.date_column):
        raise Unanswerable(f"{source.table}.{source.date_column}")


def _event_count(
    engine: MetricEngine, source: EventSource, spec: DeliveryCountQuery, as_of: datetime.date
) -> Answer:
    """Events whose own date lies in the period, whatever became of the order later."""
    _require_events(engine, source)
    cov, window = engine._window(spec.period, as_of)
    if cov == "none":
        return Answer.unanswerable("period lies entirely outside the shop's data")
    total = engine._scalar(
        f"SELECT COUNT(DISTINCT {_event_key(source)}) FROM {source.table} e "
        f"WHERE e.order_id IS NOT NULL AND e.{source.date_column} BETWEEN ? AND ?",
        [window.start, window.end],
    )
    return Answer(type="number", answer=render_metric("order_count", int(total or 0), engine.rules))


def _orders_min_events(
    engine: MetricEngine,
    source: EventSource,
    spec: OrdersMinDeliveriesQuery,
    as_of: datetime.date,
) -> Answer:
    """Orders placed in the period with at least n events dated up to the as-of day."""
    _require_events(engine, source)
    cov, window = engine._window(spec.period, as_of)
    if cov == "none":
        return Answer.unanswerable("period lies entirely outside the shop's data")
    total = engine._scalar(
        f"""
        SELECT COUNT(*) FROM (
            SELECT o.order_id
            FROM orders o JOIN {source.table} e ON e.order_id = o.order_id
            WHERE o.order_date BETWEEN ? AND ? AND e.{source.date_column} <= ?
            GROUP BY o.order_id
            HAVING COUNT(DISTINCT {_event_key(source)}) >= ?
        ) t
        """,
        [window.start, window.end, as_of, spec.n],
    )
    return Answer(type="number", answer=render_metric("order_count", int(total or 0), engine.rules))

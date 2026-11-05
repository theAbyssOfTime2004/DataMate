"""Delivery batches: counted by their own date, one per order and batch number."""

from __future__ import annotations

import datetime
import hashlib

import duckdb
import pytest

from datamate.config import settings
from datamate.etl.schema import TABLE_DDLS
from datamate.metrics import MetricEngine
from datamate.query.ext.convert import to_ext_query_spec
from datamate.query.ext.prompts import EXAMPLE_CONTEXT, build_ext_messages
from datamate.query.ext.schema import ext_parsed_schema
from datamate.query.ext.spec import DeliveryCountQuery, OrdersMinDeliveriesQuery
from datamate.rules import RuleConfig
from datamate.scoring.answer import Answer
from datamate.timeres import LunarCalendar, MonthExpr

AS_OF = datetime.date(2026, 6, 20)


def _engine(with_deliveries: bool = True) -> MetricEngine:
    con = duckdb.connect(":memory:")
    for name in ("orders", "order_lines", "deliveries"):
        con.execute(TABLE_DDLS[name])
    con.executemany(
        "INSERT INTO orders (order_id, order_date, status) VALUES (?, ?, ?)",
        [
            ("A", datetime.date(2026, 5, 28), "completed"),
            ("B", datetime.date(2026, 6, 2), "returned"),
            ("C", datetime.date(2026, 6, 10), "completed"),
        ],
    )
    if with_deliveries:
        # One row per product and batch: A's batch 1 has two products on 30/5.
        rows = [
            ("A", datetime.date(2026, 5, 30), "1"),
            ("A", datetime.date(2026, 5, 30), "1"),
            ("A", datetime.date(2026, 6, 3), "2"),
            ("A", datetime.date(2026, 6, 5), "3"),
            ("B", datetime.date(2026, 6, 4), "1"),
            ("B", datetime.date(2026, 6, 4), "2"),
            ("C", datetime.date(2026, 6, 12), "1"),
            ("C", datetime.date(2026, 6, 25), "2"),
        ]
        con.executemany(
            "INSERT INTO deliveries (order_id, delivery_date, batch_no) VALUES (?, ?, ?)", rows
        )
    return MetricEngine(
        con, RuleConfig(), LunarCalendar({}), datetime.date(2026, 1, 1), datetime.date(2026, 6, 30)
    )


def _answer(spec: object) -> Answer:
    return _engine().answer_ext(spec, AS_OF)


def test_batches_count_by_delivery_date_including_returned_orders() -> None:
    june = _answer(DeliveryCountQuery(period=MonthExpr(month=6, year=2026)))
    # A: batches 2 and 3; B: two batches on one day; C: batch 1 (batch 2 is after today).
    assert (june.type, june.answer) == ("number", 5)
    may = _answer(DeliveryCountQuery(period=MonthExpr(month=5, year=2026)))
    assert may.answer == 1


def test_orders_with_at_least_n_batches_by_today() -> None:
    spec = OrdersMinDeliveriesQuery(n=2, period=MonthExpr(month=6, year=2026))
    # B has 2 batches; C's second batch comes after as-of; A was placed in May.
    assert _answer(spec).answer == 1


def test_no_delivery_log_is_unanswerable() -> None:
    spec = DeliveryCountQuery(period=MonthExpr(month=6, year=2026))
    assert _engine(with_deliveries=False).answer_ext(spec, AS_OF).type == "unanswerable"


def test_parser_json_converts_to_delivery_families() -> None:
    period = {
        "kind": "month",
        "date": None,
        "start": None,
        "end": None,
        "n": None,
        "month": 6,
        "q": None,
        "year": 2026,
        "year_rel": None,
        "n_days": None,
    }
    count = to_ext_query_spec({"family": "delivery_count", "period": period}, AS_OF)
    assert isinstance(count, DeliveryCountQuery)
    orders = to_ext_query_spec({"family": "orders_min_deliveries", "n": 2, "period": period}, AS_OF)
    assert isinstance(orders, OrdersMinDeliveriesQuery) and orders.n == 2


# sha256 of the ext prompt as committed before the event families (both votes).
_EXT_PROMPT_SHA256_BEFORE_EVENTS = (
    "d2a2173307882a13abf34afcf798f652089f10ffeb34f7a73657812906a016d0",
    "a15bbe37a870532ec140a2f28f8688823291b1c466872c3a4251c6e071114aeb",
)


@pytest.mark.parametrize("few_shot_set", [0, 1])
def test_event_families_off_restores_the_old_ext_prompt(
    monkeypatch: pytest.MonkeyPatch, few_shot_set: int
) -> None:
    monkeypatch.setattr(settings, "event_families", False)
    messages = build_ext_messages(
        "Thang 6/2024 cua hang ban duoc bao nhieu",
        datetime.date(2024, 8, 15),
        EXAMPLE_CONTEXT,
        few_shot_set,
    )
    blob = "\n".join(f"{m['role']}\n{m['content']}" for m in messages).encode("utf-8")
    assert hashlib.sha256(blob).hexdigest() == _EXT_PROMPT_SHA256_BEFORE_EVENTS[few_shot_set]
    schema = ext_parsed_schema(EXAMPLE_CONTEXT, [])
    assert "delivery_count" not in schema["properties"]["family"]["enum"]

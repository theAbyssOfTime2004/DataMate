"""Rank block: rulebook term matching, spec conversion and computations on a tiny shop."""

from __future__ import annotations

import datetime
from typing import Any

import duckdb
import pytest

from datamate.config import settings
from datamate.etl.schema import TABLE_DDLS
from datamate.metrics import MetricEngine
from datamate.metrics.rank import answer_rank
from datamate.query.convert import ConvertError
from datamate.query.defined_terms import match_defined_term, term_candidates
from datamate.query.rank import RankQuery, describe_rank, to_rank_query
from datamate.rules import RuleConfig
from datamate.rules.parse_rulebook import RuleRow
from datamate.timeres import LunarCalendar, MonthExpr

AS_OF = datetime.date(2026, 6, 30)
JUNE = MonthExpr(month=6, year=2026)

# Invented rows standing in for a rulebook given with a test set.
ROWS = [
    RuleRow("r1", "Rulebook > Xếp hạng", "Khách ruột nhất", "khách có nhiều ngày có mua nhất"),
    RuleRow("r2", "Rulebook > Xếp hạng", "Giờ đông khách", "khung một giờ có nhiều đơn nhất"),
    RuleRow("r3", "Rulebook > Đơn", "Đơn bị huỷ", "đơn có trạng thái huỷ"),
]


class _Client:
    def __init__(self, p_yes: float) -> None:
        self.p_yes = p_yes
        self.asked: list[str] = []

    def chat_choice_probs(
        self, messages: list[dict[str, str]], choices: list[str]
    ) -> dict[str, Any]:
        del choices
        self.asked.append(messages[-1]["content"])
        return {"yes": self.p_yes, "no": 1 - self.p_yes}


def test_candidates_need_a_rare_word_pair_of_the_term() -> None:
    picked = term_candidates("thang 6 ai la khach ruot nhat cua tiem", ROWS)
    assert [row.name for row in picked] == ["Khách ruột nhất"]
    assert term_candidates("Tháng 6 doanh thu bao nhiêu?", ROWS) == []


def test_term_is_matched_only_above_the_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "defined_term_check", True)
    question = "Giờ đông khách tháng 6 là mấy giờ?"
    term, asked = match_defined_term(question, ROWS, _Client(0.97))
    assert term is not None and term.name == "Giờ đông khách"
    assert asked == [("Giờ đông khách", 0.97)]
    term, _asked = match_defined_term(question, ROWS, _Client(0.5))
    assert term is None


def test_check_off_asks_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "defined_term_check", False)
    client = _Client(0.99)
    assert match_defined_term("Giờ đông khách tháng 6?", ROWS, client) == (None, [])
    assert client.asked == []


def _raw(**fields: Any) -> dict[str, Any]:
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
    row: dict[str, Any] = {
        "family": "rank",
        "entity": "customer",
        "measure": "active_days",
        "orders": "sold",
        "min_orders": None,
        "mode": "max",
        "k": 1,
        "category": None,
        "period": period,
    }
    row.update(fields)
    return row


def test_conversion_rejects_combinations_the_block_cannot_compute() -> None:
    assert isinstance(to_rank_query(_raw(), AS_OF), RankQuery)
    assert to_rank_query(_raw(family="none"), AS_OF) is None
    with pytest.raises(ConvertError):
        to_rank_query(_raw(entity="supplier", measure="active_days"), AS_OF)
    with pytest.raises(ConvertError):
        to_rank_query(_raw(measure="revenue", orders="cancelled"), AS_OF)
    with pytest.raises(ConvertError):
        to_rank_query(_raw(entity="day", measure="order_count", min_orders=3), AS_OF)
    with pytest.raises(ConvertError, match="not grounded"):
        to_rank_query(
            _raw(entity="product", measure="order_count", category="bánh"),
            AS_OF,
            "tháng 6 món nào có mặt trong nhiều đơn nhất",
        )


def _order(
    oid: str, day: int, hour: int, cust: str | None, emp: str, status: str = "completed"
) -> tuple[Any, ...]:
    date = datetime.date(2026, 6, day)
    ts = datetime.datetime.combine(date, datetime.time(hour, 15))
    return (oid, date, ts, status, False, cust, cust, cust, cust is None, emp)


def _engine() -> MetricEngine:
    con = duckdb.connect(":memory:")
    for name in ("orders", "order_lines", "products", "employees", "purchases"):
        con.execute(TABLE_DDLS[name])
    orders = [
        # C01: 3 orders on 2 days; C02: 2 orders on 2 days, bigger orders; walk-in ignored.
        _order("o1", 1, 9, "C01", "E1"),
        _order("o2", 1, 9, "C01", "E1"),
        _order("o3", 2, 14, "C01", "E2"),
        _order("o4", 3, 9, "C02", "E2"),
        _order("o5", 4, 9, "C02", "E2"),
        _order("o6", 5, 14, None, "E1"),
        _order("o7", 6, 9, "C03", "E1", status="cancelled"),
        _order("o8", 7, 9, "C03", "E1", status="cancelled"),
        _order("o9", 8, 9, "C02", "E1", status="cancelled"),
    ]
    con.executemany(
        "INSERT INTO orders (order_id, order_date, order_ts, status, has_return, customer_key,"
        " customer_id, customer_name, is_walk_in, employee_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
        orders,
    )
    amounts = {"o1": 10, "o2": 10, "o3": 10, "o4": 50, "o5": 40, "o6": 5, "o7": 9, "o8": 9, "o9": 9}
    con.executemany(
        "INSERT INTO order_lines (order_id, line_no, product_key, product_id, product_name,"
        " line_amount) VALUES (?, 1, 'p1', 'P1', 'Trà', ?)",
        [(oid, amount * 1000) for oid, amount in amounts.items()],
    )
    con.executemany(
        "INSERT INTO employees (employee_id, employee_name) VALUES (?, ?)",
        [("E1", "Lan"), ("E2", "Minh")],
    )
    con.executemany(
        "INSERT INTO purchases (receipt_id, receipt_date, supplier_name, is_cancelled,"
        " total_amount) VALUES (?, ?, ?, ?, ?)",
        [
            ("R1", datetime.date(2026, 6, 2), "Kho Bắc", False, 900_000),
            ("R2", datetime.date(2026, 6, 3), "Kho Nam", False, 100_000),
            ("R3", datetime.date(2026, 6, 9), "Kho Nam", False, 100_000),
            ("R4", datetime.date(2026, 6, 9), "Kho Bắc", True, 100_000),
        ],
    )
    return MetricEngine(
        con,
        RuleConfig(),
        LunarCalendar({}),
        datetime.date(2026, 1, 1),
        datetime.date(2026, 6, 30),
    )


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        # Most distinct purchase days: C01 has 2 days (3 orders), C02 has 2 days; tie by code.
        ({"entity": "customer", "measure": "active_days"}, ("string", "C01")),
        # Highest average order among customers with at least 2 sold orders.
        ({"entity": "customer", "measure": "avg_order_value", "min_orders": 2}, ("string", "C02")),
        # Most cancelled orders.
        (
            {"entity": "customer", "measure": "order_count", "orders": "cancelled"},
            ("string", "C03"),
        ),
        # Most working days: E1 on 1st and 5th, E2 on 2nd, 3rd, 4th.
        ({"entity": "employee", "measure": "active_days"}, ("string", "Minh")),
        # Peak hour by sold orders: 9h has o1, o2, o4, o5.
        ({"entity": "hour", "measure": "order_count"}, ("number", 9)),
        # Main supplier by receipts (cancelled R4 left out): Kho Nam has 2.
        ({"entity": "supplier", "measure": "receipt_count"}, ("string", "Kho Nam")),
        ({"entity": "supplier", "measure": "purchase_amount"}, ("string", "Kho Bắc")),
        # Delegated to the existing families.
        ({"entity": "customer", "measure": "revenue"}, ("string", "C02")),
        ({"entity": "day", "measure": "order_count"}, ("date", "2026-06-01")),
        ({"entity": "product", "measure": "order_count", "k": 1}, ("string", "Trà")),
    ],
)
def test_rank_answers(fields: dict[str, Any], expected: tuple[str, Any]) -> None:
    spec = RankQuery(period=JUNE, **fields)
    answer = answer_rank(_engine(), spec, AS_OF)
    assert (answer.type, answer.answer) == expected


def test_lowest_day_or_hour_abstains() -> None:
    spec = RankQuery(entity="hour", measure="order_count", mode="min", period=JUNE)
    assert answer_rank(_engine(), spec, AS_OF).type == "abstain"


def test_description_lists_every_condition() -> None:
    spec = RankQuery(entity="customer", measure="avg_order_value", min_orders=3, period=JUNE)
    text = describe_rank(spec, AS_OF, RuleConfig(), LunarCalendar({}))
    assert "average revenue per order" in text
    assert "at least 3 orders" in text
    assert "2026-06-01 .. 2026-06-30" in text

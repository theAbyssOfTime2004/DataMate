"""Unit block: records counted for the shop, one staff member, or the top staff member."""

from __future__ import annotations

import datetime

import duckdb
import pytest

from datamate.etl.schema import TABLE_DDLS
from datamate.metrics import MetricEngine
from datamate.metrics.units import answer_unit
from datamate.query.convert import ConvertError
from datamate.query.units import UnitQuery, to_unit_query
from datamate.rules import RuleConfig
from datamate.timeres import LunarCalendar, MonthExpr

AS_OF = datetime.date(2026, 6, 30)
JUNE = MonthExpr(month=6, year=2026)


def _engine() -> MetricEngine:
    con = duckdb.connect(":memory:")
    for name in ("orders", "order_lines", "employees"):
        con.execute(TABLE_DDLS[name])
    # Warranty visits have no money but still count as visits.
    visits = [
        ("v1", 2, "Hà Loan Đạt", 100_000),
        ("v2", 3, "Hà Loan Đạt", 0),
        ("v3", 4, "Hà Loan Đạt", 50_000),
        ("v4", 5, "Đàm Lan An", 400_000),
        ("v5", 6, "Đàm Lan An", 0),
        ("v6", 40, "Đàm Lan An", 90_000),  # July
    ]
    for oid, day, staff, money in visits:
        date = datetime.date(2026, 6, 1) + datetime.timedelta(days=day - 1)
        con.execute(
            "INSERT INTO orders (order_id, order_date, status, has_return, employee_id)"
            " VALUES (?, ?, ?, FALSE, ?)",
            [oid, date, "completed", staff],
        )
        con.execute(
            "INSERT INTO order_lines (order_id, line_no, line_amount) VALUES (?, 1, ?)",
            [oid, money],
        )
    con.executemany(
        "INSERT INTO employees (employee_id, employee_name) VALUES (?, ?)",
        [("Hà Loan Đạt", "Hà Loan Đạt"), ("Đàm Lan An", "Đàm Lan An")],
    )
    return MetricEngine(
        con, RuleConfig(), LunarCalendar({}), datetime.date(2026, 1, 1), datetime.date(2026, 7, 31)
    )


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (UnitQuery(pick="shop", measure="count", period=JUNE), ("number", 5)),
        (UnitQuery(pick="shop", measure="money", period=JUNE), ("number", 550_000)),
        (UnitQuery(pick="staff", staff="ha loan dat", measure="count", period=JUNE), ("number", 3)),
        (
            UnitQuery(pick="staff", staff="Đàm Lan An", measure="money", period=JUNE),
            ("number", 400_000),
        ),
        (UnitQuery(pick="top_staff", measure="count", period=JUNE), ("string", "Hà Loan Đạt")),
        (UnitQuery(pick="top_staff", measure="money", period=JUNE), ("string", "Đàm Lan An")),
    ],
)
def test_unit_answers(spec: UnitQuery, expected: tuple[str, object]) -> None:
    answer = answer_unit(_engine(), spec, AS_OF)
    assert (answer.type, answer.answer) == expected


def test_unknown_staff_abstains() -> None:
    spec = UnitQuery(pick="staff", staff="Ngô Bá Kha", measure="count", period=JUNE)
    assert answer_unit(_engine(), spec, AS_OF).type == "abstain"


def _raw(**fields: object) -> dict[str, object]:
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
    row: dict[str, object] = {"family": "unit", "pick": "shop", "staff": None, "measure": "count"}
    row["period"] = period
    row.update(fields)
    return row


def test_staff_name_must_be_in_the_question() -> None:
    question = "Tháng 6/2026 thợ Hà Loan Đạt làm bao nhiêu lượt xe?"
    spec = to_unit_query(_raw(pick="staff", staff="Hà Loan Đạt"), AS_OF, question)
    assert spec is not None and spec.staff == "Hà Loan Đạt"
    with pytest.raises(ConvertError, match="not grounded"):
        to_unit_query(_raw(pick="staff", staff="Đàm Lan An"), AS_OF, question)
    assert to_unit_query(_raw(family="none"), AS_OF, question) is None

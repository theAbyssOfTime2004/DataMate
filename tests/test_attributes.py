"""Attribute block: records filtered by the book's text columns."""

from __future__ import annotations

import datetime

import duckdb
import pytest

from datamate.etl.schema import TABLE_DDLS
from datamate.metrics import MetricEngine
from datamate.metrics.attributes import answer_attr, shop_attributes
from datamate.query.attributes import AttrCondition, AttrQuery, readme_column_docs
from datamate.rules import RuleConfig
from datamate.timeres import LunarCalendar, MonthExpr

AS_OF = datetime.date(2026, 6, 30)
JUNE = MonthExpr(month=6, year=2026)

# Invented repair book: vehicle, job, note; warranty visits cost nothing.
_VISITS = [
    ("p1", "Vision 59A1-111.11", "thay nhớt, vá ruột", None, "An", 120_000),
    ("p2", "Vision 59A1-111.11", "sửa điện, thay bóng đèn", None, "An", 200_000),
    ("p3", "Wave 59B2-222.22", "thay ruột", None, "Bình", 90_000),
    ("p4", "vision 59A1-111.11", "sửa điện", "bảo hành", "Bình", 0),
    ("p5", "SH 59C3-333.33", "thay nhớt láp", None, "An", 150_000),
]


def _engine() -> MetricEngine:
    con = duckdb.connect(":memory:")
    for name in ("orders", "order_lines", "order_attributes", "employees"):
        con.execute(TABLE_DDLS[name])
    for i, (oid, car, job, note, staff, money) in enumerate(_VISITS, start=1):
        con.execute(
            "INSERT INTO orders (order_id, order_date, status, has_return, employee_id)"
            " VALUES (?, ?, 'completed', FALSE, ?)",
            [oid, datetime.date(2026, 6, i), staff],
        )
        con.execute(
            "INSERT INTO order_lines (order_id, line_no, line_amount) VALUES (?, 1, ?)",
            [oid, money],
        )
        for name, value in (("Xe", car), ("Làm gì", job), ("Ghi chú", note)):
            if value is not None:
                con.execute("INSERT INTO order_attributes VALUES (?, ?, ?)", [oid, name, value])
    return MetricEngine(
        con, RuleConfig(), LunarCalendar({}), datetime.date(2026, 1, 1), datetime.date(2026, 6, 30)
    )


def _cond(attribute: str, contains: str, negate: bool = False) -> AttrCondition:
    return AttrCondition(attribute=attribute, contains=contains, negate=negate)


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        # Job words match whole words in any spelling: "vá ruột" is in p1 only.
        (
            AttrQuery(measure="count", where=[_cond("Làm gì", "vá ruột")], period=JUNE),
            ("number", 1),
        ),
        # Electrical work without bulb changes.
        (
            AttrQuery(
                measure="count",
                where=[_cond("Làm gì", "sửa điện"), _cond("Làm gì", "bóng đèn", negate=True)],
                period=JUNE,
            ),
            ("number", 1),
        ),
        # Different Vision vehicles: one plate written in two cases counts once.
        (
            AttrQuery(
                measure="distinct", distinct_of="Xe", where=[_cond("Xe", "vision")], period=JUNE
            ),
            ("number", 1),
        ),
        # Warranty visits over paid visits: 1 / 4.
        (
            AttrQuery(
                measure="share",
                where=[_cond("Ghi chú", "bảo hành")],
                share_of_paid=True,
                period=JUNE,
            ),
            ("percent", 25.0),
        ),
        (
            AttrQuery(measure="money", staff="an", where=[_cond("Làm gì", "nhớt")], period=JUNE),
            ("number", 270_000),
        ),
    ],
)
def test_attribute_answers(spec: AttrQuery, expected: tuple[str, object]) -> None:
    answer = answer_attr(_engine(), spec, AS_OF)
    assert (answer.type, answer.answer) == expected


def test_shop_attributes_carry_readme_docs_and_samples() -> None:
    docs = readme_column_docs(
        "## Cột trong `So.xlsx`\n- **Xe**: dòng xe và biển số ghi chung một ô\n- **Tiền**: tổng tiền"
    )
    assert docs == {"Xe": "dòng xe và biển số ghi chung một ô", "Tiền": "tổng tiền"}
    attrs = {a.name: a for a in shop_attributes(_engine(), docs)}
    assert attrs["Xe"].description == "dòng xe và biển số ghi chung một ô"
    assert "Vision 59A1-111.11" in attrs["Xe"].samples
    assert set(attrs) == {"Xe", "Làm gì", "Ghi chú"}


def test_condition_no_record_meets_abstains() -> None:
    spec = AttrQuery(measure="count", where=[_cond("Làm gì", "thay dầu láp")], period=JUNE)
    assert answer_attr(_engine(), spec, AS_OF).type == "abstain"


def test_average_per_weekday_counts_days_without_records() -> None:
    # June 2026 has five Mondays; only p1 (1 June) is on a Monday.
    spec = AttrQuery(measure="count", weekday=1, per_weekday=True, period=JUNE)
    assert answer_attr(_engine(), spec, AS_OF).answer == 0.2


def test_top_value_takes_the_shop_spelling() -> None:
    spec = AttrQuery(measure="top_value", value_of="Xe", period=JUNE)
    answer = answer_attr(_engine(), spec, AS_OF)
    assert (answer.type, answer.answer) == ("string", "Vision 59A1-111.11")


def test_column_left_blank_in_the_period_abstains() -> None:
    engine = _engine()
    # May: ten visits, all with a vehicle kind; June visits have none.
    for i in range(10):
        oid = f"m{i}"
        engine.con.execute(
            "INSERT INTO orders (order_id, order_date, status, has_return, employee_id)"
            " VALUES (?, ?, 'completed', FALSE, 'An')",
            [oid, datetime.date(2026, 5, 1 + i)],
        )
        engine.con.execute("INSERT INTO order_attributes VALUES (?, 'Loại', 'tay ga')", [oid])
    spec = AttrQuery(measure="count", where=[_cond("Loại", "tay ga")], period=JUNE)
    assert answer_attr(engine, spec, AS_OF).type == "abstain"


def test_top_value_of_a_list_column_abstains() -> None:
    spec = AttrQuery(measure="top_value", value_of="Làm gì", period=JUNE)
    assert answer_attr(_engine(), spec, AS_OF).type == "abstain"


def test_total_of_a_number_column() -> None:
    engine = _engine()
    engine.con.execute(TABLE_DDLS["order_measures"])
    for row, (oid, km) in enumerate([("p1", 3), ("p2", 5), ("p3", 2), ("p4", 4), ("p5", 6)]):
        engine.con.execute("INSERT INTO order_measures VALUES (?, ?, 'Số km', ?)", [oid, row, km])
    numbers = [a for a in shop_attributes(engine, {}) if a.numeric]
    assert [a.name for a in numbers] == ["Số km"]
    spec = AttrQuery(measure="total", value_of="Số km", where=[_cond("Xe", "vision")], period=JUNE)
    assert answer_attr(engine, spec, AS_OF).answer == 12  # p1, p2, p4


def test_total_needs_a_number_column() -> None:
    from datamate.query.attributes import to_attr_query
    from datamate.query.convert import ConvertError

    raw = {"family": "attr", "measure": "total", "value_of": "Xe", "where": [], "period": {"kind": "month_this"}}
    with pytest.raises(ConvertError):
        to_attr_query(raw, AS_OF, "Tháng này chạy bao nhiêu km?", ["Xe", "Số km"], ["Số km"])
    raw["value_of"] = "Số km"
    spec = to_attr_query(raw, AS_OF, "Tháng này chạy bao nhiêu km?", ["Xe", "Số km"], ["Số km"])
    assert spec is not None and spec.measure == "total"


def test_number_columns_skip_codes_and_repeat_order_totals() -> None:
    from datamate.etl.build import number_columns

    cells = [
        ("o1", 1, "Trang", "120"),
        ("o1", 2, "Trang", "80"),
        ("o2", 3, "Trang", "40"),
        ("o1", 1, "SĐT", "0901234567"),
        ("o2", 3, "SĐT", "0912345678"),
        ("o1", 1, "Tổng phiếu", "500.000"),
        ("o1", 2, "Tổng phiếu", "500.000"),
        ("o2", 3, "Tổng phiếu", "90.000"),
    ]
    rows = number_columns(cells)
    by_name: dict[str, list[object]] = {}
    for row in rows:
        by_name.setdefault(str(row["name"]), []).append(row["value"])
    assert sorted(by_name) == ["Trang", "Tổng phiếu"]
    assert sum(by_name["Trang"]) == 240
    assert len(by_name["Tổng phiếu"]) == 2  # one value per order

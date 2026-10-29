"""Hand-written QuerySpecs for the 33 train acceptance questions.

Expected values are the acceptance table. Do not edit them to make a test pass.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any

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
from datamate.timeres.periods import (
    BeforeTetExpr,
    LastNDaysExpr,
    MonthExpr,
    MonthThisExpr,
    PeriodExpr,
    QuarterExpr,
    TetExpr,
    TodayExpr,
    WeekLastExpr,
    WeekThisExpr,
    YesterdayExpr,
    YtdExpr,
)


@dataclass(frozen=True)
class GoldCase:
    n: int
    question_id: str
    business_id: str
    as_of: datetime.date
    spec: QuerySpec
    expected_type: str
    expected: Any


def _query(
    metric: Metric,
    period: PeriodExpr,
    filters: Filters | None = None,
) -> MetricQuery:
    return MetricQuery(metric=metric, period=period, filters=filters or Filters())


def _value(
    metric: Metric,
    period: PeriodExpr,
    filters: Filters | None = None,
) -> ValueQuery:
    return ValueQuery(query=_query(metric, period, filters))


def _month(month: int, year: int) -> MonthExpr:
    return MonthExpr(month=month, year=year)


CASES: tuple[GoldCase, ...] = (
    GoldCase(
        1,
        "q_65695d2cb626",
        "biz_111",
        datetime.date(2026, 8, 19),
        _value("cancel_rate", MonthThisExpr()),
        "percent",
        1.75,
    ),
    GoldCase(
        2,
        "q_fbce2202d8da",
        "biz_111",
        datetime.date(2026, 8, 19),
        CompareQuery(
            a=_query("revenue", _month(4, 2026)),
            b=_query("revenue", _month(4, 2025)),
            op="pct_change",
        ),
        "percent",
        16.62,
    ),
    GoldCase(
        3,
        "q_ea9608e77cc1",
        "biz_105",
        datetime.date(2025, 4, 2),
        _value("revenue", LastNDaysExpr(n=7)),
        "number",
        81885000,
    ),
    GoldCase(
        4,
        "q_a0b2d8694b3d",
        "biz_106",
        datetime.date(2026, 9, 14),
        _value(
            "revenue_share",
            _month(2, 2025),
            Filters(payment_method=["Ví điện tử"]),
        ),
        "percent",
        20.42,
    ),
    GoldCase(
        5,
        "q_f7ddc2a17849",
        "biz_105",
        datetime.date(2026, 6, 24),
        _value("cancel_count", _month(6, 2025)),
        "number",
        103,
    ),
    GoldCase(
        6,
        "q_1cc70c12aa27",
        "biz_101",
        datetime.date(2026, 2, 22),
        _value("revenue", TodayExpr()),
        "number",
        13479900,
    ),
    GoldCase(
        7,
        "q_0a14fc442eee",
        "biz_123",
        datetime.date(2026, 6, 28),
        _value(
            "revenue",
            _month(12, 2025),
            Filters(category=["giảm đau hạ sốt"]),
        ),
        "number",
        12170400,
    ),
    GoldCase(
        8,
        "q_cd1f0f0fa1aa",
        "biz_106",
        datetime.date(2026, 9, 14),
        _value("avg_order_value", _month(8, 2025)),
        "number",
        70201,
    ),
    GoldCase(
        9,
        "q_72d7deb192d7",
        "biz_106",
        datetime.date(2026, 9, 14),
        _value(
            "revenue",
            _month(5, 2026),
            Filters(payment_method=["Chuyển khoản"]),
        ),
        "number",
        107616500,
    ),
    GoldCase(
        10,
        "q_ec7f47250481",
        "biz_123",
        datetime.date(2026, 6, 28),
        _value(
            "revenue_share",
            MonthThisExpr(),
            Filters(customer_segment=["sỉ"]),
        ),
        "percent",
        7.47,
    ),
    GoldCase(
        11,
        "q_62a9a454f35b",
        "biz_101",
        datetime.date(2026, 8, 24),
        _value(
            "revenue",
            _month(5, 2026),
            Filters(product=["Coca Cola 330ml"]),
        ),
        "number",
        2726000,
    ),
    GoldCase(
        12,
        "q_a9f00a074ef4",
        "biz_106",
        datetime.date(2026, 9, 14),
        _value("cancel_rate", _month(2, 2025)),
        "percent",
        2.23,
    ),
    GoldCase(
        13,
        "q_e390c3cb0b3b",
        "biz_111",
        datetime.date(2025, 9, 28),
        _value("revenue", TodayExpr()),
        "number",
        10792500,
    ),
    GoldCase(
        14,
        "q_1c4d072b63b0",
        "biz_105",
        datetime.date(2026, 2, 12),
        _value("order_count", WeekLastExpr(), Filters(shift="afternoon")),
        "number",
        555,
    ),
    GoldCase(
        15,
        "q_445aa67d4d59",
        "biz_101",
        datetime.date(2026, 8, 22),
        _value("revenue", YesterdayExpr()),
        "number",
        6873700,
    ),
    GoldCase(
        16,
        "q_14d138cb0a76",
        "biz_105",
        datetime.date(2026, 6, 24),
        _value("revenue", _month(4, 2026), Filters(shift="afternoon")),
        "number",
        176282700,
    ),
    GoldCase(
        17,
        "q_834b9b6ff399",
        "biz_111",
        datetime.date(2026, 8, 19),
        _value("avg_order_value", _month(6, 2025)),
        "number",
        72142,
    ),
    GoldCase(
        18,
        "q_116e55aef197",
        "biz_111",
        datetime.date(2026, 8, 19),
        _value(
            "revenue",
            _month(3, 2025),
            Filters(customer_segment=["thân thiết"]),
        ),
        "number",
        46304000,
    ),
    GoldCase(
        19,
        "q_a54cc8dccf38",
        "biz_106",
        datetime.date(2026, 9, 14),
        _value("order_count", QuarterExpr(q=3, year=2025)),
        "number",
        11568,
    ),
    GoldCase(
        20,
        "q_b8abcac3d506",
        "biz_111",
        datetime.date(2026, 8, 19),
        _value("cancel_count", _month(6, 2025)),
        "number",
        64,
    ),
    GoldCase(
        21,
        "q_f6e455c13726",
        "biz_105",
        datetime.date(2025, 11, 12),
        _value("order_count", LastNDaysExpr(n=30)),
        "number",
        5122,
    ),
    GoldCase(
        22,
        "q_257bf10688b9",
        "biz_123",
        datetime.date(2026, 6, 28),
        _value(
            "revenue",
            _month(12, 2025),
            Filters(customer_segment=["sỉ"]),
        ),
        "number",
        15773200,
    ),
    GoldCase(
        23,
        "q_1367a63a409f",
        "biz_105",
        datetime.date(2026, 6, 18),
        _value("revenue", YtdExpr()),
        "number",
        2086001100,
    ),
    GoldCase(
        24,
        "q_74001e944444",
        "biz_123",
        datetime.date(2026, 6, 28),
        SeriesQuery(
            metric="revenue",
            periods=[_month(2, 2025), _month(3, 2025), _month(4, 2025)],
        ),
        "list",
        [197963000, 207089500, 208231500],
    ),
    GoldCase(
        25,
        "q_e971b8478b14",
        "biz_111",
        datetime.date(2026, 8, 19),
        ArgMaxQuery(metric="revenue", period=_month(3, 2026), granularity="day"),
        "date",
        "2026-03-01",
    ),
    GoldCase(
        26,
        "q_0fbd522a4f2f",
        "biz_106",
        datetime.date(2026, 9, 14),
        CompareQuery(
            a=_query("revenue", TetExpr(year=2026)),
            b=_query("revenue", BeforeTetExpr(n_days=17, year=2026)),
            op="pct_change",
        ),
        "percent",
        -41.86,
    ),
    GoldCase(
        27,
        "q_5f5c3b11314d",
        "biz_123",
        datetime.date(2026, 6, 28),
        CompareQuery(
            a=_query("revenue", TetExpr(year_rel=0)),
            b=_query("revenue", TetExpr(year_rel=-1)),
            op="pct_change",
        ),
        "percent",
        -14.45,
    ),
    GoldCase(
        28,
        "q_cadff21c3eac",
        "biz_106",
        datetime.date(2025, 12, 6),
        _value("revenue", WeekThisExpr()),
        "number",
        61343000,
    ),
    GoldCase(
        29,
        "q_338ae382671b",
        "biz_105",
        datetime.date(2026, 3, 12),
        CompareQuery(
            a=_query("revenue", WeekLastExpr(), Filters(shift="afternoon")),
            b=_query("revenue", WeekLastExpr(), Filters(shift="morning")),
            op="pct_change",
        ),
        "percent",
        -16.41,
    ),
    GoldCase(
        30,
        "q_2bd1825f38c7",
        "biz_123",
        datetime.date(2026, 6, 28),
        _value("revenue", QuarterExpr(q=3, year=2025)),
        "number",
        616402300,
    ),
    GoldCase(
        31,
        "q_06e8d7da339d",
        "biz_105",
        datetime.date(2026, 6, 2),
        _value("avg_order_value", WeekLastExpr()),
        "number",
        72137,
    ),
    GoldCase(
        32,
        "q_a569fbc6b9fc",
        "biz_106",
        datetime.date(2026, 9, 14),
        TrendQuery(
            metric="revenue",
            periods=[_month(4, 2026), _month(5, 2026), _month(6, 2026)],
            direction="increasing",
        ),
        "boolean",
        False,
    ),
    GoldCase(
        33,
        "q_33f75059825a",
        "biz_123",
        datetime.date(2026, 6, 28),
        MultiQuery(
            queries=[
                _query("revenue", _month(4, 2026), Filters(payment_method=["cod"])),
                _query("revenue", _month(4, 2026), Filters(payment_method=["ewallet"])),
            ]
        ),
        "list",
        [10849600, 15777900],
    ),
)

from __future__ import annotations

import datetime

import pytest

from datamate.config import PROJECT_ROOT, settings
from datamate.rules import RuleConfig, load_rules
from datamate.timeres import (
    BeforeTetExpr,
    DayExpr,
    LastNDaysExpr,
    LunarCalendar,
    MonthExpr,
    MonthThisExpr,
    PeriodError,
    QuarterExpr,
    RangeExpr,
    TetExpr,
    TodayExpr,
    WeekLastExpr,
    WeekThisExpr,
    YearExpr,
    YesterdayExpr,
    YtdExpr,
    coverage,
    resolve,
    resolve_compare,
)


def _calendar() -> LunarCalendar:
    path = settings.data_root / "train" / "calendar.csv"
    if not path.is_file():
        pytest.skip("train calendar.csv is absent")
    return LunarCalendar.from_csv(path)


def _rules() -> RuleConfig:
    return load_rules(PROJECT_ROOT / "rules" / "train.json")


def _r(
    expr: object,
    as_of: datetime.date,
    cal: LunarCalendar,
    rules: RuleConfig,
) -> tuple[datetime.date, datetime.date]:
    got = resolve(expr, as_of, rules, cal)  # type: ignore[arg-type]
    return got.start, got.end


def test_section_2_rows_and_edges() -> None:
    cal = _calendar()
    rules = _rules()

    as_of = datetime.date(2026, 8, 19)
    assert _r(TodayExpr(), as_of, cal, rules) == (as_of, as_of)
    assert _r(YesterdayExpr(), as_of, cal, rules) == (
        datetime.date(2026, 8, 18),
        datetime.date(2026, 8, 18),
    )
    assert _r(DayExpr(date=datetime.date(2026, 2, 22)), as_of, cal, rules) == (
        datetime.date(2026, 2, 22),
        datetime.date(2026, 2, 22),
    )
    assert _r(
        RangeExpr(start=datetime.date(2026, 1, 2), end=datetime.date(2026, 1, 5)),
        as_of,
        cal,
        rules,
    ) == (datetime.date(2026, 1, 2), datetime.date(2026, 1, 5))

    assert _r(LastNDaysExpr(n=7), datetime.date(2025, 4, 2), cal, rules) == (
        datetime.date(2025, 3, 27),
        datetime.date(2025, 4, 2),
    )
    assert _r(LastNDaysExpr(n=30), datetime.date(2025, 11, 12), cal, rules) == (
        datetime.date(2025, 10, 14),
        datetime.date(2025, 11, 12),
    )

    assert _r(WeekThisExpr(), datetime.date(2025, 12, 6), cal, rules) == (
        datetime.date(2025, 12, 1),
        datetime.date(2025, 12, 6),
    )
    assert _r(WeekLastExpr(), datetime.date(2026, 2, 12), cal, rules) == (
        datetime.date(2026, 2, 2),
        datetime.date(2026, 2, 8),
    )

    assert _r(MonthThisExpr(), datetime.date(2026, 8, 19), cal, rules) == (
        datetime.date(2026, 8, 1),
        datetime.date(2026, 8, 19),
    )
    prev = resolve_compare(
        MonthThisExpr(),
        "previous_period",
        datetime.date(2026, 8, 19),
        rules,
        cal,
    )
    assert (prev.start, prev.end) == (datetime.date(2026, 7, 1), datetime.date(2026, 7, 19))
    feb = resolve_compare(
        MonthThisExpr(),
        "previous_period",
        datetime.date(2026, 3, 31),
        rules,
        cal,
    )
    assert (feb.start, feb.end) == (datetime.date(2026, 2, 1), datetime.date(2026, 2, 28))

    assert _r(MonthExpr(month=7), datetime.date(2026, 6, 24), cal, rules) == (
        datetime.date(2025, 7, 1),
        datetime.date(2025, 7, 31),
    )
    assert _r(MonthExpr(month=6), datetime.date(2026, 6, 24), cal, rules) == (
        datetime.date(2026, 6, 1),
        datetime.date(2026, 6, 24),
    )
    assert _r(MonthExpr(month=4, year=2026), datetime.date(2026, 8, 19), cal, rules) == (
        datetime.date(2026, 4, 1),
        datetime.date(2026, 4, 30),
    )

    assert _r(QuarterExpr(q=3, year=2025), datetime.date(2026, 6, 24), cal, rules) == (
        datetime.date(2025, 7, 1),
        datetime.date(2025, 9, 30),
    )
    assert _r(YtdExpr(), datetime.date(2026, 6, 18), cal, rules) == (
        datetime.date(2026, 1, 1),
        datetime.date(2026, 6, 18),
    )
    ytd_prev = resolve_compare(YtdExpr(), "previous_period", datetime.date(2026, 6, 18), rules, cal)
    assert (ytd_prev.start, ytd_prev.end) == (
        datetime.date(2025, 1, 1),
        datetime.date(2025, 6, 18),
    )

    assert _r(TetExpr(year=2026), datetime.date(2026, 6, 28), cal, rules) == (
        datetime.date(2026, 2, 10),
        datetime.date(2026, 2, 26),
    )
    assert _r(TetExpr(year=2025), datetime.date(2026, 6, 28), cal, rules) == (
        datetime.date(2025, 1, 22),
        datetime.date(2025, 2, 7),
    )
    assert _r(BeforeTetExpr(n_days=17, year=2026), datetime.date(2026, 6, 28), cal, rules) == (
        datetime.date(2026, 1, 24),
        datetime.date(2026, 2, 9),
    )
    assert _r(TetExpr(year_rel=-1), datetime.date(2026, 6, 28), cal, rules) == (
        datetime.date(2025, 1, 22),
        datetime.date(2025, 2, 7),
    )

    week_prev = resolve_compare(
        WeekThisExpr(),
        "previous_period",
        datetime.date(2025, 12, 6),
        rules,
        cal,
    )
    assert (week_prev.start, week_prev.end) == (
        datetime.date(2025, 11, 24),
        datetime.date(2025, 11, 29),
    )
    leap = resolve_compare(
        RangeExpr(start=datetime.date(2024, 2, 29), end=datetime.date(2024, 2, 29)),
        "same_period_last_year",
        datetime.date(2026, 1, 1),
        rules,
        cal,
    )
    assert (leap.start, leap.end) == (datetime.date(2023, 2, 28), datetime.date(2023, 2, 28))
    tet_before = resolve_compare(
        TetExpr(year=2026),
        "before_tet_same_length",
        datetime.date(2026, 6, 28),
        rules,
        cal,
    )
    assert (tet_before.start, tet_before.end) == (
        datetime.date(2026, 1, 24),
        datetime.date(2026, 2, 9),
    )

    with pytest.raises(PeriodError):
        resolve(
            DayExpr(date=datetime.date(2026, 9, 1)),
            datetime.date(2026, 8, 19),
            rules,
            cal,
        )
    with pytest.raises(ValueError):
        MonthExpr(month=7, year=2026, year_rel=0)


def test_year_expr_and_coverage() -> None:
    cal = _calendar()
    rules = _rules()
    assert _r(YearExpr(year=2025), datetime.date(2026, 6, 18), cal, rules) == (
        datetime.date(2025, 1, 1),
        datetime.date(2025, 12, 31),
    )
    assert _r(YearExpr(), datetime.date(2026, 6, 18), cal, rules) == (
        datetime.date(2026, 1, 1),
        datetime.date(2026, 6, 18),
    )

    data_start = datetime.date(2026, 3, 1)
    data_end = datetime.date(2026, 6, 30)
    status, clipped = coverage(
        resolve(
            RangeExpr(start=datetime.date(2026, 1, 1), end=datetime.date(2026, 2, 28)),
            datetime.date(2026, 8, 1),
            rules,
            cal,
        ),
        data_start,
        data_end,
    )
    assert status == "none"
    assert clipped is None

    status, clipped = coverage(
        resolve(
            RangeExpr(start=datetime.date(2026, 2, 1), end=datetime.date(2026, 3, 15)),
            datetime.date(2026, 8, 1),
            rules,
            cal,
        ),
        data_start,
        data_end,
    )
    assert status == "partial"
    assert clipped is not None
    assert (clipped.start, clipped.end) == (
        datetime.date(2026, 3, 1),
        datetime.date(2026, 3, 15),
    )

    inside = resolve(
        RangeExpr(start=datetime.date(2026, 4, 1), end=datetime.date(2026, 4, 30)),
        datetime.date(2026, 8, 1),
        rules,
        cal,
    )
    status, clipped = coverage(inside, data_start, data_end)
    assert status == "full"
    assert clipped == inside

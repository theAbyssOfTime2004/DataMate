from __future__ import annotations

import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from datamate.rules.model import RuleConfig
from datamate.timeres.calendar import LunarCalendar

CompareKind = Literal["previous_period", "same_period_last_year", "before_tet_same_length"]
Coverage = Literal["full", "partial", "none"]


class PeriodError(ValueError):
    """Raised when a period starts after as_of or the expression is inconsistent."""


class DateRange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: datetime.date
    end: datetime.date


class _Expr(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _YearChoice(_Expr):
    year: int | None = None
    year_rel: int | None = None

    @model_validator(mode="after")
    def _one_year_field(self) -> _YearChoice:
        if self.year is not None and self.year_rel is not None:
            raise ValueError("set only one of year and year_rel")
        return self


class DayExpr(_Expr):
    kind: Literal["day"] = "day"
    date: datetime.date


class TodayExpr(_Expr):
    kind: Literal["today"] = "today"


class YesterdayExpr(_Expr):
    kind: Literal["yesterday"] = "yesterday"


class LastNDaysExpr(_Expr):
    kind: Literal["last_n_days"] = "last_n_days"
    n: int = Field(ge=1)


class WeekThisExpr(_Expr):
    kind: Literal["week_this"] = "week_this"


class WeekLastExpr(_Expr):
    kind: Literal["week_last"] = "week_last"


class MonthExpr(_YearChoice):
    kind: Literal["month"] = "month"
    month: int = Field(ge=1, le=12)


class MonthThisExpr(_Expr):
    kind: Literal["month_this"] = "month_this"


class MonthLastExpr(_Expr):
    kind: Literal["month_last"] = "month_last"


class QuarterExpr(_YearChoice):
    kind: Literal["quarter"] = "quarter"
    q: int = Field(ge=1, le=4)


class YearExpr(_YearChoice):
    kind: Literal["year"] = "year"


class YtdExpr(_Expr):
    kind: Literal["ytd"] = "ytd"


class RangeExpr(_Expr):
    kind: Literal["range"] = "range"
    start: datetime.date
    end: datetime.date


class TetExpr(_YearChoice):
    kind: Literal["tet"] = "tet"


class BeforeTetExpr(_YearChoice):
    kind: Literal["before_tet"] = "before_tet"
    n_days: int = Field(ge=1)


PeriodExpr = Annotated[
    DayExpr
    | TodayExpr
    | YesterdayExpr
    | LastNDaysExpr
    | WeekThisExpr
    | WeekLastExpr
    | MonthExpr
    | MonthThisExpr
    | MonthLastExpr
    | QuarterExpr
    | YearExpr
    | YtdExpr
    | RangeExpr
    | TetExpr
    | BeforeTetExpr,
    Field(discriminator="kind"),
]


def _add_years(day: datetime.date, years: int) -> datetime.date:
    try:
        return day.replace(year=day.year + years)
    except ValueError:
        return day.replace(year=day.year + years, month=2, day=28)


def _month_end(year: int, month: int) -> datetime.date:
    if month == 12:
        return datetime.date(year, 12, 31)
    return datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)


def _monday(day: datetime.date, rules: RuleConfig) -> datetime.date:
    """Return the first day of the week that contains `day` (name is historical)."""
    if rules.week_start == "monday":
        return day - datetime.timedelta(days=day.weekday())
    if rules.week_start == "sunday":
        # weekday(): Mon=0 … Sun=6 → days since Sunday.
        return day - datetime.timedelta(days=(day.weekday() + 1) % 7)
    raise PeriodError(f"unsupported week_start {rules.week_start}")


def _clip(start: datetime.date, end: datetime.date, as_of: datetime.date) -> DateRange:
    if start > end:
        raise PeriodError(f"period start {start} is after end {end}")
    if start > as_of:
        raise PeriodError(f"period starts {start} after as_of {as_of}")
    return DateRange(start=start, end=min(end, as_of))


def _picked_year(
    year: int | None,
    year_rel: int | None,
    as_of: datetime.date,
    start_month: int,
    start_day: int,
) -> int:
    if year is not None:
        return year
    if year_rel is not None:
        return as_of.year + year_rel
    start = datetime.date(as_of.year, start_month, start_day)
    if start > as_of:
        return as_of.year - 1
    return as_of.year


def _tet_bounds(year: int, rules: RuleConfig, cal: LunarCalendar) -> DateRange:
    tet = rules.tet
    start = cal.to_solar(
        year + tet.start_year_offset,
        tet.start_lunar_month,
        tet.start_lunar_day,
        leap=False,
    )
    end = cal.to_solar(year, tet.end_lunar_month, tet.end_lunar_day, leap=False)
    return DateRange(start=start, end=end)


def _tet_year(expr: TetExpr | BeforeTetExpr, as_of: datetime.date, rules: RuleConfig, cal: LunarCalendar) -> int:
    if expr.year is not None:
        return expr.year
    if expr.year_rel is not None:
        return as_of.year + expr.year_rel
    start = _tet_bounds(as_of.year, rules, cal).start
    if start > as_of:
        return as_of.year - 1
    return as_of.year


def _nominal(expr: PeriodExpr, as_of: datetime.date, rules: RuleConfig, cal: LunarCalendar) -> DateRange:
    if isinstance(expr, DayExpr):
        return DateRange(start=expr.date, end=expr.date)
    if isinstance(expr, TodayExpr):
        return DateRange(start=as_of, end=as_of)
    if isinstance(expr, YesterdayExpr):
        day = as_of - datetime.timedelta(days=1)
        return DateRange(start=day, end=day)
    if isinstance(expr, LastNDaysExpr):
        start = as_of - datetime.timedelta(days=expr.n - 1)
        return DateRange(start=start, end=as_of)
    if isinstance(expr, WeekThisExpr):
        return DateRange(start=_monday(as_of, rules), end=as_of)
    if isinstance(expr, WeekLastExpr):
        this_monday = _monday(as_of, rules)
        start = this_monday - datetime.timedelta(days=7)
        end = this_monday - datetime.timedelta(days=1)
        return DateRange(start=start, end=end)
    if isinstance(expr, MonthExpr):
        year = _picked_year(expr.year, expr.year_rel, as_of, expr.month, 1)
        return DateRange(
            start=datetime.date(year, expr.month, 1),
            end=_month_end(year, expr.month),
        )
    if isinstance(expr, MonthThisExpr):
        return DateRange(start=datetime.date(as_of.year, as_of.month, 1), end=as_of)
    if isinstance(expr, MonthLastExpr):
        if as_of.month == 1:
            year, month = as_of.year - 1, 12
        else:
            year, month = as_of.year, as_of.month - 1
        return DateRange(start=datetime.date(year, month, 1), end=_month_end(year, month))
    if isinstance(expr, QuarterExpr):
        start_month = (expr.q - 1) * 3 + 1
        year = _picked_year(expr.year, expr.year_rel, as_of, start_month, 1)
        return DateRange(
            start=datetime.date(year, start_month, 1),
            end=_month_end(year, start_month + 2),
        )
    if isinstance(expr, YearExpr):
        year = _picked_year(expr.year, expr.year_rel, as_of, 1, 1)
        return DateRange(start=datetime.date(year, 1, 1), end=datetime.date(year, 12, 31))
    if isinstance(expr, YtdExpr):
        return DateRange(start=datetime.date(as_of.year, 1, 1), end=as_of)
    if isinstance(expr, RangeExpr):
        return DateRange(start=expr.start, end=expr.end)
    if isinstance(expr, TetExpr):
        return _tet_bounds(_tet_year(expr, as_of, rules, cal), rules, cal)
    if isinstance(expr, BeforeTetExpr):
        tet = _tet_bounds(_tet_year(expr, as_of, rules, cal), rules, cal)
        end = tet.start - datetime.timedelta(days=1)
        start = end - datetime.timedelta(days=expr.n_days - 1)
        return DateRange(start=start, end=end)
    raise PeriodError(f"unsupported period {expr!r}")


def resolve(
    expr: PeriodExpr,
    as_of: datetime.date,
    rules: RuleConfig,
    cal: LunarCalendar,
) -> DateRange:
    """Resolve a period, then clip the end to as_of. A start after as_of is an error."""
    nominal = _nominal(expr, as_of, rules, cal)
    return _clip(nominal.start, nominal.end, as_of)


def _same_day_previous_month(as_of: datetime.date) -> DateRange:
    if as_of.month == 1:
        year, month = as_of.year - 1, 12
    else:
        year, month = as_of.year, as_of.month - 1
    last = _month_end(year, month)
    day = min(as_of.day, last.day)
    return DateRange(start=datetime.date(year, month, 1), end=datetime.date(year, month, day))


def _shift_back(period: DateRange, days: int) -> DateRange:
    delta = datetime.timedelta(days=days)
    return DateRange(start=period.start - delta, end=period.end - delta)


def _previous_period(
    expr: PeriodExpr,
    as_of: datetime.date,
    rules: RuleConfig,
    cal: LunarCalendar,
) -> DateRange:
    if isinstance(expr, MonthThisExpr):
        return _same_day_previous_month(as_of)
    if isinstance(expr, WeekThisExpr):
        this_monday = _monday(as_of, rules)
        prev_monday = this_monday - datetime.timedelta(days=7)
        end = prev_monday + datetime.timedelta(days=as_of.weekday())
        return DateRange(start=prev_monday, end=end)
    if isinstance(expr, MonthExpr):
        year = _picked_year(expr.year, expr.year_rel, as_of, expr.month, 1)
        if expr.month == 1:
            prev_year, prev_month = year - 1, 12
        else:
            prev_year, prev_month = year, expr.month - 1
        return _clip(
            datetime.date(prev_year, prev_month, 1),
            _month_end(prev_year, prev_month),
            as_of,
        )
    if isinstance(expr, QuarterExpr):
        start_month = (expr.q - 1) * 3 + 1
        year = _picked_year(expr.year, expr.year_rel, as_of, start_month, 1)
        if expr.q == 1:
            prev_year, prev_q = year - 1, 4
        else:
            prev_year, prev_q = year, expr.q - 1
        prev_start_month = (prev_q - 1) * 3 + 1
        return _clip(
            datetime.date(prev_year, prev_start_month, 1),
            _month_end(prev_year, prev_start_month + 2),
            as_of,
        )
    if isinstance(expr, LastNDaysExpr):
        window_start = as_of - datetime.timedelta(days=expr.n - 1)
        end = window_start - datetime.timedelta(days=1)
        start = end - datetime.timedelta(days=expr.n - 1)
        return _clip(start, end, as_of)
    if isinstance(expr, (DayExpr, TodayExpr, YesterdayExpr)):
        current = resolve(expr, as_of, rules, cal)
        prev = current.start - datetime.timedelta(days=1)
        return DateRange(start=prev, end=prev)
    if isinstance(expr, YearExpr):
        year = _picked_year(expr.year, expr.year_rel, as_of, 1, 1)
        return _clip(datetime.date(year - 1, 1, 1), datetime.date(year - 1, 12, 31), as_of)
    if isinstance(expr, YtdExpr):
        current = resolve(expr, as_of, rules, cal)
        return _clip(_add_years(current.start, -1), _add_years(current.end, -1), as_of)
    if isinstance(expr, WeekLastExpr):
        return _shift_back(resolve(expr, as_of, rules, cal), 7)
    if isinstance(expr, MonthLastExpr):
        current = resolve(expr, as_of, rules, cal)
        return _previous_full_month(current.start, as_of)
    if isinstance(expr, (RangeExpr, BeforeTetExpr)):
        current = resolve(expr, as_of, rules, cal)
        length = (current.end - current.start).days + 1
        end = current.start - datetime.timedelta(days=1)
        start = end - datetime.timedelta(days=length - 1)
        return _clip(start, end, as_of)
    if isinstance(expr, TetExpr):
        year = _tet_year(expr, as_of, rules, cal)
        previous = _tet_bounds(year - 1, rules, cal)
        return _clip(previous.start, previous.end, as_of)
    raise PeriodError(f"unsupported period {expr!r}")


def _previous_full_month(month_start: datetime.date, as_of: datetime.date) -> DateRange:
    if month_start.month == 1:
        year, month = month_start.year - 1, 12
    else:
        year, month = month_start.year, month_start.month - 1
    return _clip(datetime.date(year, month, 1), _month_end(year, month), as_of)


def _compare_tet_year(
    expr: PeriodExpr,
    as_of: datetime.date,
    rules: RuleConfig,
    cal: LunarCalendar,
) -> int:
    if isinstance(expr, (TetExpr, BeforeTetExpr)):
        return _tet_year(expr, as_of, rules, cal)
    start = _tet_bounds(as_of.year, rules, cal).start
    if start > as_of:
        return as_of.year - 1
    return as_of.year


def resolve_compare(
    expr: PeriodExpr,
    kind: CompareKind,
    as_of: datetime.date,
    rules: RuleConfig,
    cal: LunarCalendar,
) -> DateRange:
    if kind == "previous_period":
        return _previous_period(expr, as_of, rules, cal)
    current = resolve(expr, as_of, rules, cal)
    if kind == "same_period_last_year":
        return _clip(_add_years(current.start, -1), _add_years(current.end, -1), as_of)
    if kind == "before_tet_same_length":
        length = (current.end - current.start).days + 1
        tet = _tet_bounds(_compare_tet_year(expr, as_of, rules, cal), rules, cal)
        end = tet.start - datetime.timedelta(days=1)
        start = end - datetime.timedelta(days=length - 1)
        return _clip(start, end, as_of)
    raise PeriodError(f"unsupported compare kind {kind}")


def coverage(
    period: DateRange,
    data_start: datetime.date,
    data_end: datetime.date,
) -> tuple[Coverage, DateRange | None]:
    """§7: none when the period misses the shop's dates; partial keeps the overlap."""
    if data_end < data_start:
        raise PeriodError(f"data_end {data_end} is before data_start {data_start}")
    if period.end < data_start or period.start > data_end:
        return "none", None
    if period.start >= data_start and period.end <= data_end:
        return "full", period
    clipped = DateRange(
        start=max(period.start, data_start),
        end=min(period.end, data_end),
    )
    return "partial", clipped

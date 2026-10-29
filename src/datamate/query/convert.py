from __future__ import annotations

import datetime
import unicodedata

from pydantic import ValidationError

from datamate.query.parse_schema import (
    COMPARE_KINDS,
    FILTER_KEYS,
    METRICS,
    PERIOD_KINDS,
    QUESTION_KINDS,
)
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
from datamate.rules.model import RuleConfig
from datamate.timeres.calendar import CalendarError, LunarCalendar
from datamate.timeres.periods import (
    BeforeTetExpr,
    CompareKind,
    DayExpr,
    LastNDaysExpr,
    MonthExpr,
    MonthLastExpr,
    MonthThisExpr,
    PeriodError,
    PeriodExpr,
    QuarterExpr,
    RangeExpr,
    TetExpr,
    TodayExpr,
    WeekLastExpr,
    WeekThisExpr,
    YearExpr,
    YesterdayExpr,
    YtdExpr,
    resolve,
    resolve_compare,
)

_OPTIONAL_YEAR = frozenset({"year", "year_rel"})
_PERIOD_FIELDS: dict[str, frozenset[str]] = {
    "day": frozenset({"date"}),
    "today": frozenset(),
    "yesterday": frozenset(),
    "last_n_days": frozenset({"n"}),
    "week_this": frozenset(),
    "week_last": frozenset(),
    "month": frozenset({"month", "year", "year_rel"}),
    "month_this": frozenset(),
    "month_last": frozenset(),
    "quarter": frozenset({"q", "year", "year_rel"}),
    "year": frozenset({"year", "year_rel"}),
    "ytd": frozenset(),
    "range": frozenset({"start", "end"}),
    "tet": frozenset({"year", "year_rel"}),
    "before_tet": frozenset({"n_days", "year", "year_rel"}),
}
_COUNT_ALIAS: dict[str, tuple[str, str]] = {
    "last_n_days": ("n", "n_days"),
    "before_tet": ("n_days", "n"),
}
_RELATIVE_COMPARE = frozenset(
    {"previous_period", "same_period_last_year", "before_tet_same_length"}
)
_SHIFTS = frozenset({"morning", "afternoon"})


class ConvertError(ValueError):
    """The flat parse cannot be turned into a QuerySpec without guessing."""


def to_query_spec(
    parsed: dict[str, object],
    as_of: datetime.date,
    rules: RuleConfig,
    cal: LunarCalendar,
) -> QuerySpec:
    kind = _choice(parsed.get("kind"), QUESTION_KINDS, "kind")
    metric = _choice(parsed.get("metric"), METRICS, "metric")
    periods = _periods(parsed.get("periods"))
    compare_kind = _optional_choice(parsed.get("compare_kind"), COMPARE_KINDS, "compare_kind")
    op = _optional_choice(parsed.get("op"), ("pct_change", "diff"), "op")
    filters = _filters(parsed.get("filters"), "filters")
    filters_b = parsed.get("filters_b")
    multi_filters = parsed.get("multi_filters")
    direction = _optional_choice(parsed.get("direction"), ("increasing", "decreasing"), "direction")
    granularity = _optional_choice(parsed.get("granularity"), ("day", "month"), "granularity")
    mode = _optional_choice(parsed.get("mode"), ("max", "min"), "mode")

    _reject_unused(
        kind,
        compare_kind=compare_kind,
        op=op,
        filters_b=filters_b,
        multi_filters=multi_filters,
        direction=direction,
        granularity=granularity,
        mode=mode,
    )
    if kind == "value":
        _expect_count(periods, 1, "value")
        return ValueQuery(query=_query(metric, periods[0], filters))
    if kind == "compare":
        return _compare(metric, periods, compare_kind, op, filters, filters_b, as_of, rules, cal)
    if kind == "series":
        if len(periods) < 2:
            raise ConvertError("series needs at least two periods")
        return SeriesQuery(metric=metric, filters=filters, periods=periods)
    if kind == "multi":
        return _multi(metric, periods, multi_filters)
    if kind == "trend":
        if len(periods) < 2:
            raise ConvertError("trend needs at least two periods")
        if direction is None:
            raise ConvertError("trend requires direction")
        return TrendQuery(metric=metric, filters=filters, periods=periods, direction=direction)
    if granularity is None:
        raise ConvertError("argmax requires granularity")
    _expect_count(periods, 1, "argmax")
    return ArgMaxQuery(
        metric=metric,
        filters=filters,
        period=periods[0],
        granularity=granularity,
        mode=mode or "max",
    )


def _compare(
    metric: Metric,
    periods: list[PeriodExpr],
    compare_kind: str | None,
    op: str | None,
    filters: Filters,
    filters_b: object,
    as_of: datetime.date,
    rules: RuleConfig,
    cal: LunarCalendar,
) -> CompareQuery:
    if op is None:
        raise ConvertError("compare requires op")
    if compare_kind in _RELATIVE_COMPARE and filters_b is not None:
        raise ConvertError("compare cannot set both a relative compare_kind and filters_b")
    if compare_kind == "explicit":
        _expect_count(periods, 2, "explicit compare")
        period_a: PeriodExpr = periods[0]
        period_b: PeriodExpr = periods[1]
        if filters_b is not None and _window(periods[0], as_of, rules, cal) != _window(
            periods[1], as_of, rules, cal
        ):
            raise ConvertError("filters_b compares two filters over one period; give one period")
    elif compare_kind in _RELATIVE_COMPARE:
        period_a = _asked_period(periods, compare_kind, as_of, rules, cal)
        period_b = _resolved_range(period_a, compare_kind, as_of, rules, cal)
    elif filters_b is not None:
        _expect_count(periods, 1, "filters_b compare")
        period_a = period_b = periods[0]
    else:
        raise ConvertError("compare needs compare_kind or filters_b")
    other = _filters(filters_b, "filters_b") if filters_b is not None else filters
    return CompareQuery(
        a=_query(metric, period_a, filters),
        b=_query(metric, period_b, other),
        op=op,  # type: ignore[arg-type]
    )


def _asked_period(
    periods: list[PeriodExpr],
    compare_kind: str | None,
    as_of: datetime.date,
    rules: RuleConfig,
    cal: LunarCalendar,
) -> PeriodExpr:
    """A relative compare takes one period. Two are accepted only when one of them is
    exactly the baseline the other implies (the model also wrote out the baseline)."""
    if len(periods) == 1:
        return periods[0]
    if len(periods) != 2:
        raise ConvertError(f"{compare_kind} requires 1 period(s), got {len(periods)}")
    asked = [
        candidate
        for candidate, baseline in ((periods[0], periods[1]), (periods[1], periods[0]))
        if _baseline_matches(candidate, baseline, compare_kind, as_of, rules, cal)
    ]
    if len(asked) != 1:
        raise ConvertError(f"{compare_kind} requires 1 period(s), got {len(periods)}")
    return asked[0]


def _baseline_matches(
    candidate: PeriodExpr,
    baseline: PeriodExpr,
    compare_kind: str | None,
    as_of: datetime.date,
    rules: RuleConfig,
    cal: LunarCalendar,
) -> bool:
    try:
        implied = _resolved_range(candidate, compare_kind, as_of, rules, cal)
    except ConvertError:
        return False
    found = _window(baseline, as_of, rules, cal)
    return found is not None and (implied.start, implied.end) == found


def _window(
    period: PeriodExpr, as_of: datetime.date, rules: RuleConfig, cal: LunarCalendar
) -> tuple[datetime.date, datetime.date] | None:
    try:
        found = resolve(period, as_of, rules, cal)
    except (PeriodError, CalendarError):
        return None
    return found.start, found.end


def _resolved_range(
    period: PeriodExpr,
    compare_kind: str | None,
    as_of: datetime.date,
    rules: RuleConfig,
    cal: LunarCalendar,
) -> RangeExpr:
    if compare_kind not in _RELATIVE_COMPARE:
        raise ConvertError(f"unsupported compare_kind {compare_kind}")
    kind: CompareKind = compare_kind  # type: ignore[assignment]
    try:
        resolved = resolve_compare(period, kind, as_of, rules, cal)
    except PeriodError as exc:
        raise ConvertError(str(exc)) from exc
    return RangeExpr(start=resolved.start, end=resolved.end)


def _multi(metric: Metric, periods: list[PeriodExpr], raw_filters: object) -> MultiQuery:
    _expect_count(periods, 1, "multi")
    if not isinstance(raw_filters, list) or not raw_filters:
        raise ConvertError("multi requires multi_filters")
    queries = [_query(metric, periods[0], _filters(item, "multi_filters")) for item in raw_filters]
    return MultiQuery(queries=queries)


def _query(metric: Metric, period: PeriodExpr, filters: Filters) -> MetricQuery:
    return MetricQuery(metric=metric, period=period, filters=filters)


def _periods(raw: object) -> list[PeriodExpr]:
    if not isinstance(raw, list):
        raise ConvertError("periods must be a list")
    return [_period_expr(item) for item in raw]


def _period_expr(raw: object) -> PeriodExpr:
    if not isinstance(raw, dict):
        raise ConvertError("period must be an object")
    kind = raw.get("kind")
    if kind not in PERIOD_KINDS:
        raise ConvertError(f"unknown period kind {kind!r}")
    raw = _merge_count_alias(str(kind), raw)
    allowed = _PERIOD_FIELDS[str(kind)]
    for field, value in raw.items():
        if field == "kind" or value is None:
            continue
        if field not in allowed:
            raise ConvertError(f"period {kind} does not use {field}")
    try:
        return _build_period(str(kind), raw)
    except (ValidationError, TypeError, ValueError) as exc:
        raise ConvertError(str(exc)) from exc


def _merge_count_alias(kind: str, raw: dict[str, object]) -> dict[str, object]:
    """n and n_days are the same count; the model often fills both with one value."""
    if kind not in _COUNT_ALIAS:
        return raw
    used, alias = _COUNT_ALIAS[kind]
    extra = raw.get(alias)
    if extra is None:
        return raw
    merged = dict(raw)
    if merged.get(used) is None:
        merged[used] = extra
    elif merged[used] != extra:
        raise ConvertError(f"period {kind}: {used} and {alias} disagree")
    merged[alias] = None
    return merged


def _build_period(kind: str, raw: dict[str, object]) -> PeriodExpr:
    year_fields = (
        _year_fields(raw) if kind in {"month", "quarter", "year", "tet", "before_tet"} else {}
    )
    if kind == "day":
        return DayExpr(date=_as_date(_need(raw, "date"), "date"))
    if kind == "today":
        return TodayExpr()
    if kind == "yesterday":
        return YesterdayExpr()
    if kind == "last_n_days":
        return LastNDaysExpr(n=_as_int(_need(raw, "n"), "n"))
    if kind == "week_this":
        return WeekThisExpr()
    if kind == "week_last":
        return WeekLastExpr()
    if kind == "month":
        return MonthExpr(month=_as_int(_need(raw, "month"), "month"), **year_fields)
    if kind == "month_this":
        return MonthThisExpr()
    if kind == "month_last":
        return MonthLastExpr()
    if kind == "quarter":
        return QuarterExpr(q=_as_int(_need(raw, "q"), "q"), **year_fields)
    if kind == "year":
        return YearExpr(**year_fields)
    if kind == "ytd":
        return YtdExpr()
    if kind == "range":
        return RangeExpr(
            start=_as_date(_need(raw, "start"), "start"),
            end=_as_date(_need(raw, "end"), "end"),
        )
    if kind == "tet":
        return TetExpr(**year_fields)
    return BeforeTetExpr(n_days=_as_int(_need(raw, "n_days"), "n_days"), **year_fields)


def _year_fields(raw: dict[str, object]) -> dict[str, int]:
    found: dict[str, int] = {}
    for field in _OPTIONAL_YEAR:
        value = raw.get(field)
        if value is not None:
            found[field] = _as_int(value, field)
    return found


def _filters(raw: object, label: str) -> Filters:
    if not isinstance(raw, dict):
        raise ConvertError(f"{label} must be an object")
    unknown = [key for key in raw if key not in FILTER_KEYS]
    if unknown:
        raise ConvertError(f"{label} has unknown keys {unknown}")
    shift = raw.get("shift")
    if shift is not None and shift not in _SHIFTS:
        raise ConvertError(f"{label}.shift is invalid")
    try:
        return Filters(
            payment_method=_str_list(raw.get("payment_method", []), f"{label}.payment_method"),
            channel=_str_list(raw.get("channel", []), f"{label}.channel"),
            customer_segment=_str_list(
                raw.get("customer_segment", []), f"{label}.customer_segment"
            ),
            category=_str_list(raw.get("category", []), f"{label}.category"),
            product=_str_list(raw.get("product", []), f"{label}.product"),
            shift=shift,  # type: ignore[arg-type]
        )
    except ValidationError as exc:
        raise ConvertError(str(exc)) from exc


def _reject_unused(
    kind: str,
    *,
    compare_kind: str | None,
    op: str | None,
    filters_b: object,
    multi_filters: object,
    direction: str | None,
    granularity: str | None,
    mode: str | None,
) -> None:
    if kind != "compare":
        if compare_kind is not None:
            raise ConvertError("compare_kind is only valid for compare")
        if op is not None:
            raise ConvertError("op is only valid for compare")
        if filters_b is not None:
            raise ConvertError("filters_b is only valid for compare")
    if kind != "multi" and _has_items(multi_filters):
        raise ConvertError("multi_filters is only valid for multi")
    if kind != "trend" and direction is not None:
        raise ConvertError("direction is only valid for trend")
    if kind != "argmax" and (granularity is not None or mode is not None):
        raise ConvertError("granularity and mode are only valid for argmax")


def _has_items(raw: object) -> bool:
    return isinstance(raw, list) and len(raw) > 0


def _expect_count(periods: list[PeriodExpr], count: int, label: str) -> None:
    if len(periods) != count:
        raise ConvertError(f"{label} requires {count} period(s), got {len(periods)}")


def _choice(value: object, allowed: tuple[str, ...], field: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ConvertError(f"invalid {field}")
    return value


def _optional_choice(value: object, allowed: tuple[str, ...], field: str) -> str | None:
    if value is None:
        return None
    return _choice(value, allowed, field)


def _need(raw: dict[str, object], field: str) -> object:
    value = raw.get(field)
    if value is None:
        raise ConvertError(f"period requires {field}")
    return value


def _as_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConvertError(f"{field} must be an integer")
    return value


def _as_date(value: object, field: str) -> datetime.date:
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    if isinstance(value, str):
        try:
            return datetime.date.fromisoformat(value)
        except ValueError as exc:
            raise ConvertError(f"{field} is not a date") from exc
    raise ConvertError(f"{field} is not a date")


def _str_list(value: object, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ConvertError(f"{field} must be a list")
    found: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ConvertError(f"{field} must contain text")
        found.append(unicodedata.normalize("NFC", item))
    return found

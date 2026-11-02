from __future__ import annotations

import datetime
import itertools
import unicodedata

from pydantic import ValidationError

from datamate.query.grounding import PRODUCT_CUES, QuestionText
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
_GROUNDED_FILTERS = ("payment_method", "channel", "customer_segment", "category")
_PREVIOUS_LABELS = frozenset({"yesterday", "week_last", "month_last"})
_TO_DATE_KINDS = frozenset({"today", "week_this", "month_this", "ytd"})
_DAY_WORDS = ("ngay nao", "hom nao", "ngay may")
_MONTH_WORDS = ("thang nao", "thang may")
_MIN_WORDS = ("it nhat", "thap nhat", "kem nhat", "e nhat", "cham nhat")


class ConvertError(ValueError):
    """The flat parse cannot be turned into a QuerySpec without guessing."""


def to_query_spec(
    parsed: dict[str, object],
    as_of: datetime.date,
    rules: RuleConfig,
    cal: LunarCalendar,
    question: str | None = None,
) -> QuerySpec:
    """With the question text, filter values and year wording must be grounded in it."""
    text = QuestionText(question) if question is not None else None
    kind = _choice(parsed.get("kind"), QUESTION_KINDS, "kind")
    metric = _choice(parsed.get("metric"), METRICS, "metric")
    compare_kind = _optional_choice(parsed.get("compare_kind"), COMPARE_KINDS, "compare_kind")
    raw_periods = _drop_written_baseline(parsed.get("periods"), compare_kind, text)
    periods = _periods(raw_periods, as_of)
    op = _optional_choice(parsed.get("op"), ("pct_change", "diff"), "op")
    raw_filters = _ground_filters(text, parsed.get("filters"), "filters")
    filters = _filters(raw_filters, "filters")
    filters_b = _ground_filters(text, parsed.get("filters_b"), "filters_b")
    multi_filters = parsed.get("multi_filters")
    if isinstance(multi_filters, list):
        multi_filters = [_ground_filters(text, item, "multi_filters") for item in multi_filters]
    direction = _optional_choice(parsed.get("direction"), ("increasing", "decreasing"), "direction")
    granularity = _optional_choice(parsed.get("granularity"), ("day", "month"), "granularity")
    mode = _optional_choice(parsed.get("mode"), ("max", "min"), "mode")

    if (
        kind in ("multi", "series")
        and filters_b is not None
        and not _has_items(multi_filters)
        and len(periods) == 1
    ):
        # "doanh thu tiền mặt và doanh thu COD": two filtered values written as a pair.
        kind, multi_filters, filters_b = "multi", [raw_filters, filters_b], None
    if kind == "multi" and filters_b is not None and _has_items(multi_filters):
        # The entries are in multi_filters; filters_b only restates one of them (or is empty).
        listed = [_filters(item, "multi_filters") for item in multi_filters]
        restated = _filters(filters_b, "filters_b")
        if restated == Filters() or restated in listed:
            filters_b = None
    if kind == "argmax" and text is not None:
        # Trend wording on a "which day" question adds nothing; the words decide the rest.
        direction = None
        granularity = granularity or _argmax_granularity(text)
        mode = mode or ("min" if any(text.has_phrase(w) for w in _MIN_WORDS) else "max")
    if text is not None:
        _check_period_wording(text, raw_periods, compare_kind)

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
    period = _contiguous_span(periods, as_of, rules, cal, "argmax")
    return ArgMaxQuery(
        metric=metric,
        filters=filters,
        period=period,
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
        if _filters(filters_b, "filters_b") == filters:
            filters_b = None
        elif len(periods) == 1:
            # "Tuần trước ca chiều so với ca sáng": two filters over the one period.
            compare_kind = None
        else:
            raise ConvertError(
                "compare cannot set both a relative compare_kind and filters_b; to compare "
                "two filters over one period, set compare_kind null and give one period"
            )
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
    if any(query.filters == Filters() for query in queries):
        raise ConvertError("each multi_filters entry must name a filter value")
    if len({query.filters.model_dump_json() for query in queries}) != len(queries):
        raise ConvertError("multi_filters entries must differ")
    return MultiQuery(queries=queries)


def _query(metric: Metric, period: PeriodExpr, filters: Filters) -> MetricQuery:
    return MetricQuery(metric=metric, period=period, filters=filters)


def _periods(raw: object, as_of: datetime.date) -> list[PeriodExpr]:
    if not isinstance(raw, list):
        raise ConvertError("periods must be a list")
    return [_period_expr(item, as_of) for item in raw]


def _drop_written_baseline(
    raw: object, compare_kind: str | None, text: QuestionText | None
) -> object:
    """'So với tháng trước đó, tháng 7/2026...': the model often lists tháng trước as a
    second period next to an explicit month. previous_period derives that baseline, so a
    bare previous-period label beside an explicit period is dropped, whatever stray
    fields it carries."""
    if compare_kind != "previous_period" or not isinstance(raw, list) or len(raw) != 2:
        return raw
    kinds = [item.get("kind") if isinstance(item, dict) else None for item in raw]
    labels = [kind in _PREVIOUS_LABELS for kind in kinds]
    if labels.count(True) != 1:
        return raw
    other = kinds[labels.index(False)]
    if other not in PERIOD_KINDS:
        return raw
    # Next to tháng này, tháng trước is only a derived baseline when the question asks
    # for the same stretch of days (cùng kỳ / cùng số ngày).
    same_stretch = text is not None and (
        text.has_phrase("cung ky") or text.has_phrase("cung so ngay")
    )
    if other in _TO_DATE_KINDS and not same_stretch:
        return raw
    return [raw[labels.index(False)]]


def period_from_raw(raw: object, as_of: datetime.date) -> PeriodExpr:
    """Public wrapper so the extension parser reuses the same PeriodExpr kinds."""
    return _period_expr(raw, as_of)


def _period_expr(raw: object, as_of: datetime.date) -> PeriodExpr:
    if not isinstance(raw, dict):
        raise ConvertError("period must be an object")
    kind = raw.get("kind")
    if kind not in PERIOD_KINDS:
        raise ConvertError(f"unknown period kind {kind!r}")
    raw = _merge_count_alias(str(kind), raw)
    year = raw.get("year")
    if isinstance(year, int) and not isinstance(year, bool) and 0 <= year < 100:
        # "3/25" read as year 25 would resolve outside the data and answer
        # "unanswerable"; reject so the question abstains instead.
        raise ConvertError(f"year {year} must be written in full (e.g. 20{year:02d})")
    kind, raw = _settle_redundant_fields(str(kind), raw, as_of)
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


def _settle_redundant_fields(
    kind: str, raw: dict[str, object], as_of: datetime.date
) -> tuple[str, dict[str, object]]:
    """Drop fields that only restate the kind (ytd with this year, month_this with this
    month). An explicit month and year on month_this/month_last win over the label."""
    settled = dict(raw)
    if settled.get("year") is not None and settled.get("year_rel") is not None:
        settled["year_rel"] = None  # the written year is the more reliable of the two
    if kind in ("month_this", "month_last"):
        anchor = as_of.replace(day=1)
        if kind == "month_last":
            anchor = (anchor - datetime.timedelta(days=1)).replace(day=1)
        month, year, year_rel = settled.get("month"), settled.get("year"), settled.get("year_rel")
        if month is None and year is None and year_rel is None:
            return kind, settled
        consistent = (
            month in (None, anchor.month)
            and year in (None, anchor.year)
            and (year_rel is None or year_rel == anchor.year - as_of.year)
        )
        if consistent:
            return kind, {**settled, "month": None, "year": None, "year_rel": None}
        if month is not None and year is not None:
            return "month", {**settled, "kind": "month"}
        return kind, settled
    if kind in ("today", "yesterday"):
        day = as_of if kind == "today" else as_of - datetime.timedelta(days=1)
        expected_day: dict[str, object] = {
            "date": day,
            "start": day,
            "end": day,
            "year": day.year,
            "year_rel": day.year - as_of.year,
            "month": day.month,
        }
        return kind, _clear_if_restated(kind, settled, expected_day)
    if kind == "ytd":
        expected: dict[str, object] = {
            "year": as_of.year,
            "year_rel": 0,
            "date": as_of,
            "end": as_of,
            "start": datetime.date(as_of.year, 1, 1),
        }
        return kind, _clear_if_restated(kind, settled, expected)
    return kind, settled


def _clear_if_restated(
    kind: str, raw: dict[str, object], expected: dict[str, object]
) -> dict[str, object]:
    for field, value in expected.items():
        found = raw.get(field)
        if found is None:
            continue
        if field in ("date", "end", "start"):
            try:
                found = _as_date(found, field)
            except ConvertError:
                return raw
        if found != value:
            return raw
    return {**raw, **dict.fromkeys(expected)}


def _ground_filters(text: QuestionText | None, raw: object, label: str) -> object:
    """Drop stray filter values; reject a value that stands in for an unlisted condition."""
    if text is None or not isinstance(raw, dict):
        return raw
    kept = dict(raw)
    for field in _GROUNDED_FILTERS:
        values = raw.get(field)
        if not isinstance(values, list):
            continue
        chosen: list[object] = []
        for value in values:
            if not isinstance(value, str) or text.mentions_value(value):
                chosen.append(value)
            elif not text.is_stray(field, value):
                raise ConvertError(
                    f"{label}.{field} {value!r} is not what the question says. If no listed "
                    "value matches the question's condition, set in_scope false"
                )
        kept[field] = chosen
    products = raw.get("product")
    if isinstance(products, list):
        chosen_products: list[object] = []
        for value in products:
            if not isinstance(value, str) or text.mentions_product(value):
                chosen_products.append(value)
                continue
            hits, _total = text.product_word_hits(value)
            if hits or any(text.has_phrase(cue) for cue in PRODUCT_CUES):
                raise ConvertError(
                    f"{label}.product {value!r} is not what the question names. Use a "
                    "category for a product group; if nothing listed matches, set "
                    "in_scope false"
                )
        kept["product"] = chosen_products
    shift = raw.get("shift")
    if isinstance(shift, str) and not text.mentions_shift(shift):
        raise ConvertError(f"{label}.shift {shift!r} is not mentioned in the question")
    return kept


def _check_period_wording(
    text: QuestionText, raw_periods: object, compare_kind: str | None
) -> None:
    periods = raw_periods if isinstance(raw_periods, list) else []
    # Two written periods are checked against each other by _asked_period instead.
    if compare_kind == "same_period_last_year" and len(periods) == 1 and not text.mentions_year():
        raise ConvertError(
            "same_period_last_year needs năm in the question; cùng kỳ tháng trước or "
            "tháng trước is previous_period"
        )
    for item in periods:
        if (
            isinstance(item, dict)
            and item.get("kind") == "ytd"
            and not (text.mentions_year() or text.mentions_jan_first())
        ):
            raise ConvertError(
                "ytd needs từ đầu năm or 1/1 in the question; use the period the question names"
            )


def _argmax_granularity(text: QuestionText) -> str | None:
    if any(text.has_phrase(phrase) for phrase in _DAY_WORDS):
        return "day"
    if any(text.has_phrase(phrase) for phrase in _MONTH_WORDS):
        return "month"
    return None


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


def _contiguous_span(
    periods: list[PeriodExpr],
    as_of: datetime.date,
    rules: RuleConfig,
    cal: LunarCalendar,
    label: str,
) -> PeriodExpr:
    """One period, or back-to-back periods ("months 11/2025, 12/2025 and 1/2026") as the
    range they cover. Gaps stay an error: the range would add periods nobody asked about."""
    if len(periods) == 1:
        return periods[0]
    windows = [_window(period, as_of, rules, cal) for period in periods]
    if len(periods) == 0 or any(window is None for window in windows):
        raise ConvertError(f"{label} requires 1 period(s), got {len(periods)}")
    spans = sorted(window for window in windows if window is not None)
    for (_, prev_end), (next_start, _) in itertools.pairwise(spans):
        if next_start != prev_end + datetime.timedelta(days=1):
            raise ConvertError(f"{label} requires 1 period(s), got {len(periods)}")
    return RangeExpr(start=spans[0][0], end=spans[-1][1])


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

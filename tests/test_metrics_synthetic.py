from __future__ import annotations

import datetime
from decimal import Decimal

import duckdb
import pandas as pd
import pytest
from pydantic import TypeAdapter, ValidationError

from datamate.etl import (
    CUSTOMERS_COLUMNS,
    ORDER_LINES_COLUMNS,
    ORDERS_COLUMNS,
    PRODUCTS_COLUMNS,
    UNIT_CONVERSIONS_COLUMNS,
    CanonicalData,
    EtlReport,
    write_duckdb,
)
from datamate.metrics import MetricEngine
from datamate.query import Filters, MetricQuery, QuerySpec, ValueQuery
from datamate.query.spec import ArgMaxQuery, CompareQuery, MultiQuery, SeriesQuery, TrendQuery
from datamate.rules import RuleConfig
from datamate.timeres import DayExpr, LunarCalendar, MonthExpr, RangeExpr

AS_OF = datetime.date(2026, 6, 30)
DATA_START = datetime.date(2026, 1, 1)
DATA_END = datetime.date(2026, 6, 30)
JUNE = MonthExpr(month=6, year=2026)
MAY = MonthExpr(month=5, year=2026)


def _frame(columns: tuple[str, ...], rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = None
    return frame.loc[:, list(columns)]


def _order(
    order_id: str,
    day: datetime.date,
    hour: int,
    minute: int,
    status: str,
    *,
    has_return: bool = False,
    customer_key: str = "c1",
    walk_in: bool = False,
    payment: str = "cash",
    channel: str = "store",
    order_ts: datetime.datetime | None | object = ...,
) -> dict[str, object]:
    stamp: datetime.datetime | None
    if order_ts is ...:
        stamp = datetime.datetime.combine(day, datetime.time(hour, minute))
    else:
        stamp = order_ts  # type: ignore[assignment]
    return {
        "order_id": order_id,
        "order_date": day,
        "order_ts": stamp,
        "status": status,
        "has_return": has_return,
        "customer_key": customer_key,
        "is_walk_in": walk_in,
        "payment_method": payment,
        "channel": channel,
    }


def _line(
    order_id: str,
    name: str,
    amount: int,
    *,
    product_key: str,
    product_id: str | None = None,
) -> dict[str, object]:
    return {
        "order_id": order_id,
        "line_no": 1,
        "product_key": product_key,
        "product_id": product_id,
        "product_name": name,
        "line_amount": amount,
        "is_return_line": False,
        "src_row": 1,
    }


def _shop_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    orders = [
        _order("o1", datetime.date(2026, 6, 1), 8, 0, "completed"),
        _order(
            "o2",
            datetime.date(2026, 6, 1),
            15,
            0,
            "completed",
            customer_key="walk-in",
            walk_in=True,
        ),
        _order(
            "o3",
            datetime.date(2026, 6, 2),
            9,
            0,
            "cancelled",
            customer_key="c2",
            payment="transfer",
        ),
        _order("o4", datetime.date(2026, 6, 3), 10, 0, "returned", has_return=True),
        _order(
            "o5",
            datetime.date(2026, 6, 4),
            16,
            0,
            "shipping",
            payment="Ví điện tử",
            channel="zalo",
        ),
        _order("o6", datetime.date(2026, 5, 15), 7, 0, "completed", customer_key="c2"),
        _order("o7", datetime.date(2026, 4, 1), 8, 0, "completed"),
        _order("o8", datetime.date(2026, 4, 1), 8, 0, "completed"),
        _order("o9", datetime.date(2026, 3, 1), 8, 0, "completed"),
        _order("o10", datetime.date(2026, 3, 1), 8, 0, "completed"),
        _order("o11", datetime.date(2026, 6, 20), 11, 0, "completed"),
        _order("o12", datetime.date(2026, 1, 5), 8, 0, "completed"),
    ]
    lines = [
        _line("o1", "Coca Cola 330ml", 1000, product_key="coca cola 330ml"),
        _line("o2", "Coca  Cola 330ml", 500, product_key="coca cola 330ml"),
        _line("o3", "other", 999, product_key="other"),
        _line("o4", "other", 7000, product_key="other"),
        _line("o5", "Paracetamol", 200, product_key="paracetamol"),
        _line("o6", "other", 4000, product_key="other"),
        _line("o7", "x", 2, product_key="x"),
        _line("o8", "x", 3, product_key="x"),
        _line("o9", "special", 49, product_key="special"),
        _line("o10", "other", 3951, product_key="other"),
        _line("o11", "other", 1500, product_key="other"),
        _line("o12", "Hàng mã", 100, product_key="SKU9", product_id="SKU9"),
    ]
    return orders, lines


def _products() -> list[dict[str, object]]:
    return [
        {
            "product_key": "coca cola 330ml",
            "product_name": "Coca Cola 330ml",
            "category": "đồ uống",
        },
        {"product_key": "paracetamol", "product_name": "Paracetamol", "category": "thuốc"},
        {"product_key": "special", "product_name": "special", "category": "khác"},
        {"product_key": "other", "product_name": "other", "category": "khác"},
        {"product_key": "x", "product_name": "x", "category": "khác"},
        {
            "product_key": "SKU9",
            "product_id": "SKU9",
            "product_name": "Hàng mã",
            "category": "khác",
        },
    ]


def _write(
    path: object,
    orders: list[dict[str, object]],
    lines: list[dict[str, object]],
    customers: list[dict[str, object]],
) -> None:
    data = CanonicalData(
        orders=_frame(ORDERS_COLUMNS, orders),
        order_lines=_frame(ORDER_LINES_COLUMNS, lines),
        products=_frame(PRODUCTS_COLUMNS, _products()),
        unit_conversions=_frame(UNIT_CONVERSIONS_COLUMNS, []),
        customers=_frame(CUSTOMERS_COLUMNS, customers),
        report=EtlReport(),
    )
    write_duckdb(data, path)  # type: ignore[arg-type]


def _engine(
    path: object,
    orders: list[dict[str, object]],
    lines: list[dict[str, object]],
    customers: list[dict[str, object]],
) -> tuple[MetricEngine, duckdb.DuckDBPyConnection]:
    _write(path, orders, lines, customers)
    con = duckdb.connect(str(path), read_only=True)
    engine = MetricEngine(con, RuleConfig(), LunarCalendar({}), DATA_START, DATA_END)
    return engine, con


def _value(
    metric: str, period: object, answer_type: str = "number", **filters: object
) -> ValueQuery:
    return ValueQuery(
        answer_type=answer_type,  # type: ignore[arg-type]
        query=MetricQuery(metric=metric, period=period, filters=Filters(**filters)),  # type: ignore[arg-type]
    )


@pytest.fixture(scope="module")
def engine(tmp_path_factory: pytest.TempPathFactory) -> MetricEngine:
    orders, lines = _shop_rows()
    customers = [
        {"customer_key": "c1", "segment": "sỉ"},
        {"customer_key": "c2", "segment": "lẻ"},
    ]
    path = tmp_path_factory.mktemp("metrics") / "shop.duckdb"
    built, con = _engine(path, orders, lines, customers)
    yield built
    con.close()


@pytest.fixture(scope="module")
def bare_engine(tmp_path_factory: pytest.TempPathFactory) -> MetricEngine:
    orders = [_order("o1", datetime.date(2026, 6, 1), 8, 0, "completed", order_ts=None)]
    lines = [_line("o1", "other", 1000, product_key="other")]
    path = tmp_path_factory.mktemp("metrics-bare") / "bare.duckdb"
    built, con = _engine(path, orders, lines, [])
    yield built
    con.close()


def test_query_spec_parses_a_value_and_rejects_unknown_fields() -> None:
    spec = TypeAdapter(QuerySpec).validate_python(
        {
            "kind": "value",
            "answer_type": "number",
            "query": {
                "metric": "revenue",
                "period": {"kind": "month", "month": 6, "year": 2026},
                "filters": {"shift": "morning"},
            },
        }
    )
    assert isinstance(spec, ValueQuery)
    assert spec.query.filters.shift == "morning"
    with pytest.raises(ValidationError):
        Filters.model_validate({"payment_method": ["cash"], "unknown": 1})


def test_revenue_definition_and_filters(engine: MetricEngine) -> None:
    assert engine.answer(_value("revenue", JUNE), AS_OF).answer == 3200
    assert engine.answer(_value("order_count", JUNE), AS_OF).answer == 4
    assert engine.answer(_value("cancel_count", JUNE), AS_OF).answer == 1
    rate = engine.answer(_value("cancel_rate", JUNE, "percent"), AS_OF)
    assert rate.type == "percent"
    assert rate.answer == float(Decimal("16.67"))
    assert engine.answer(_value("avg_order_value", JUNE), AS_OF).answer == 800

    assert engine.answer(_value("revenue", JUNE, payment_method=["cash"]), AS_OF).answer == 3000
    assert engine.answer(_value("revenue", JUNE, channel=["zalo"]), AS_OF).answer == 200
    assert engine.answer(_value("revenue", JUNE, customer_segment=["sỉ"]), AS_OF).answer == 2700
    assert engine.answer(_value("revenue", JUNE, shift="morning"), AS_OF).answer == 2500
    assert engine.answer(_value("revenue", JUNE, shift="afternoon"), AS_OF).answer == 700
    assert engine.answer(_value("revenue", JUNE, category=["thuốc"]), AS_OF).answer == 200
    assert engine.answer(_value("revenue", JUNE, payment_method=["transfer"]), AS_OF).answer == 0

    coca = ["Coca Cola 330ml"]
    spaced = ["Coca  Cola 330ml"]
    assert engine.answer(_value("revenue", JUNE, product=coca), AS_OF).answer == 1500
    assert engine.answer(_value("revenue", JUNE, product=spaced), AS_OF).answer == 1500
    assert engine.answer(_value("revenue", JUNE, product=[*coca, *spaced]), AS_OF).answer == 1500
    assert engine.answer(_value("order_count", JUNE, product=coca), AS_OF).answer == 2

    january = MonthExpr(month=1, year=2026)
    assert engine.answer(_value("revenue", january, product=["SKU9"]), AS_OF).answer == 100
    # The parser sends display names; they must reach products keyed by a code.
    assert engine.answer(_value("revenue", january, product=["Hàng mã"]), AS_OF).answer == 100


def test_sql_keeps_filter_values_in_parameters(engine: MetricEngine) -> None:
    query = MetricQuery(metric="revenue", period=JUNE, filters=Filters(product=["Coca Cola 330ml"]))
    found = engine.compute(query, AS_OF)
    assert "?" in found.sql
    assert "Coca" not in found.sql
    assert "completed" not in found.sql
    assert "cancelled" not in found.sql


def test_rounding_half_up(engine: MetricEngine) -> None:
    april = DayExpr(date=datetime.date(2026, 4, 1))
    average = engine.answer(_value("avg_order_value", april), AS_OF)
    assert average.type == "number"
    assert average.answer == 3

    march = MonthExpr(month=3, year=2026)
    share = engine.answer(_value("revenue_share", march, "percent", product=["special"]), AS_OF)
    assert share.type == "percent"
    assert share.answer == float(Decimal("1.23"))


def test_compare_series_multi_trend_and_argmax(engine: MetricEngine) -> None:
    compared = CompareQuery(
        a=MetricQuery(metric="revenue", period=JUNE),
        b=MetricQuery(metric="revenue", period=MAY),
        op="pct_change",
    )
    change = engine.answer(compared, AS_OF)
    assert change.type == "percent"
    assert change.answer == float(Decimal("-20.00"))

    diff = engine.answer(
        CompareQuery(
            a=MetricQuery(metric="revenue", period=JUNE),
            b=MetricQuery(metric="revenue", period=MAY),
            op="diff",
        ),
        AS_OF,
    )
    assert diff.type == "number"
    assert diff.answer == -800

    series = engine.answer(
        SeriesQuery(metric="revenue", periods=[MAY, JUNE]),
        AS_OF,
    )
    assert series.type == "list"
    assert series.answer == [4000, 3200]

    multi = engine.answer(
        MultiQuery(
            queries=[
                MetricQuery(
                    metric="revenue", period=JUNE, filters=Filters(payment_method=["cash"])
                ),
                MetricQuery(metric="revenue", period=JUNE, filters=Filters(channel=["zalo"])),
            ]
        ),
        AS_OF,
    )
    assert multi.answer == [3000, 200]

    decreasing = engine.answer(
        TrendQuery(metric="revenue", periods=[MAY, JUNE], direction="decreasing"),
        AS_OF,
    )
    increasing = engine.answer(
        TrendQuery(metric="revenue", periods=[MAY, JUNE], direction="increasing"),
        AS_OF,
    )
    assert decreasing.answer is True
    assert increasing.answer is False

    top_day = engine.answer(
        ArgMaxQuery(metric="revenue", period=JUNE, granularity="day", mode="max"),
        AS_OF,
    )
    low_day = engine.answer(
        ArgMaxQuery(metric="revenue", period=JUNE, granularity="day", mode="min"),
        AS_OF,
    )
    assert top_day.type == "date"
    assert top_day.answer == "2026-06-01"
    assert low_day.answer == "2026-06-02"

    top_month = engine.answer(
        ArgMaxQuery(
            metric="revenue",
            period=RangeExpr(start=datetime.date(2026, 5, 1), end=datetime.date(2026, 6, 30)),
            granularity="month",
            mode="max",
        ),
        AS_OF,
    )
    assert top_month.answer == "2026-05"


def test_answer_type_follows_the_metric(engine: MetricEngine) -> None:
    forced = engine.answer(_value("revenue", JUNE, "percent"), AS_OF)
    assert forced.type == "number"
    assert forced.answer == 3200
    assert isinstance(forced.answer, int)

    rate_diff = engine.answer(
        CompareQuery(
            a=MetricQuery(metric="cancel_rate", period=JUNE),
            b=MetricQuery(metric="cancel_rate", period=MAY),
            op="diff",
        ),
        AS_OF,
    )
    assert rate_diff.type == "percent"
    assert rate_diff.answer == float(Decimal("16.67"))


def test_argmin_day_includes_days_without_orders(engine: MetricEngine) -> None:
    window = RangeExpr(start=datetime.date(2026, 6, 4), end=datetime.date(2026, 6, 5))
    found = engine.answer(
        ArgMaxQuery(metric="revenue", period=window, granularity="day", mode="min"),
        AS_OF,
    )
    assert found.type == "date"
    assert found.answer == "2026-06-05"


def test_period_error_abstains_and_outside_data_is_unanswerable(engine: MetricEngine) -> None:
    malformed = engine.answer(
        _value(
            "revenue",
            RangeExpr(start=datetime.date(2026, 6, 10), end=datetime.date(2026, 6, 1)),
        ),
        AS_OF,
    )
    assert malformed.type == "abstain"

    outside = engine.answer(_value("revenue", MonthExpr(month=12, year=2025)), AS_OF)
    assert outside.type == "unanswerable"


def test_coverage_partial_and_none(engine: MetricEngine) -> None:
    partial = _value(
        "revenue",
        RangeExpr(start=datetime.date(2025, 12, 15), end=datetime.date(2026, 1, 10)),
    )
    covered = engine.compute(partial.query, AS_OF)
    assert covered.coverage == "partial"
    assert covered.value == 100
    assert covered.range.start == datetime.date(2026, 1, 1)
    assert covered.range.end == datetime.date(2026, 1, 10)
    assert engine.answer(partial, AS_OF).answer == 100

    missing = engine.answer(_value("revenue", MonthExpr(month=12, year=2025)), AS_OF)
    assert missing.type == "unanswerable"


def test_division_by_zero_abstains(engine: MetricEngine) -> None:
    empty = DayExpr(date=datetime.date(2026, 2, 1))
    revenue = engine.answer(_value("revenue", empty), AS_OF)
    assert revenue.type == "number"
    assert revenue.answer == 0
    assert engine.answer(_value("cancel_rate", empty, "percent"), AS_OF).type == "abstain"
    assert engine.answer(_value("avg_order_value", empty), AS_OF).type == "abstain"


def test_missing_shift_or_segment_is_unanswerable(bare_engine: MetricEngine) -> None:
    shift = bare_engine.answer(_value("revenue", JUNE, shift="morning"), AS_OF)
    assert shift.type == "unanswerable"
    assert shift.reason is not None
    assert "shift" in shift.reason

    segment = bare_engine.answer(_value("revenue", JUNE, customer_segment=["sỉ"]), AS_OF)
    assert segment.type == "unanswerable"
    assert segment.reason is not None
    assert "segment" in segment.reason

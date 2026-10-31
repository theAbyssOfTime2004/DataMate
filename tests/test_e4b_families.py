"""Synthetic DuckDB coverage for E4b customer and threshold families."""

from __future__ import annotations

import datetime
import hashlib
from decimal import Decimal
from pathlib import Path

import duckdb
import pandas as pd

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
from datamate.query.ext.describe import describe_ext_spec
from datamate.query.ext.spec import (
    CustomersMinOrdersQuery,
    DaysMinRevenueQuery,
    OrdersMinAmountQuery,
    TopCustomersQuery,
)
from datamate.query.prompts import EXAMPLE_CONTEXT, build_messages
from datamate.rules import RuleConfig
from datamate.timeres import LunarCalendar, MonthExpr

AS_OF = datetime.date(2026, 6, 30)
DATA_START = datetime.date(2026, 1, 1)
DATA_END = datetime.date(2026, 6, 30)

_PROMPT_HASH_QUESTION = "Doanh thu thang 6/2026 cua shop la bao nhieu"
_PROMPT_HASH_AS_OF = datetime.date(2026, 6, 30)
_MAIN_PROMPT_SHA256 = "a9247b210f70f7e10dc1526829f104f1086c8648d16fab195fc6913726078729"


def _frame(columns: tuple[str, ...], rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = None
    return frame.loc[:, list(columns)]


def _order(
    order_id: str,
    day: datetime.date,
    *,
    customer_id: str | None = None,
    customer_name: str | None = None,
    is_walk_in: bool = False,
    status: str = "completed",
    has_return: bool = False,
) -> dict[str, object]:
    return {
        "order_id": order_id,
        "order_date": day,
        "order_ts": datetime.datetime.combine(day, datetime.time(10, 0)),
        "status": status,
        "has_return": has_return,
        "customer_key": customer_id or customer_name,
        "customer_id": customer_id,
        "customer_name": customer_name,
        "is_walk_in": is_walk_in,
        "payment_method": "cash",
        "channel": "store",
    }


def _line(order_id: str, *, line_amount: int, product_key: str = "p1") -> dict[str, object]:
    return {
        "order_id": order_id,
        "line_no": 1,
        "product_key": product_key,
        "product_id": "P01",
        "product_name": "Trà",
        "qty": Decimal(1),
        "unit": "ly",
        "qty_base": Decimal(1),
        "unit_price": line_amount,
        "line_amount": line_amount,
        "cost_per_unit": 1000,
        "is_return_line": False,
        "src_row": 1,
    }


def _engine(path: Path, orders: list[dict], lines: list[dict]) -> MetricEngine:
    customers = []
    seen: set[str] = set()
    for row in orders:
        key = row.get("customer_key")
        if key and key not in seen and not row.get("is_walk_in"):
            seen.add(str(key))
            customers.append(
                {
                    "customer_key": key,
                    "customer_id": row.get("customer_id"),
                    "customer_name": row.get("customer_name"),
                }
            )
    canon = CanonicalData(
        orders=_frame(ORDERS_COLUMNS, orders),
        order_lines=_frame(ORDER_LINES_COLUMNS, lines),
        products=_frame(
            PRODUCTS_COLUMNS,
            [{"product_key": "p1", "product_id": "P01", "product_name": "Trà", "base_unit": "ly"}],
        ),
        customers=_frame(CUSTOMERS_COLUMNS, customers),
        unit_conversions=_frame(UNIT_CONVERSIONS_COLUMNS, []),
        report=EtlReport(),
    )
    write_duckdb(canon, path)
    con = duckdb.connect(str(path), read_only=True)
    return MetricEngine(
        con, RuleConfig(), LunarCalendar({}), DATA_START, DATA_END
    )


def test_main_prompt_hash_unchanged() -> None:
    messages = build_messages(
        _PROMPT_HASH_QUESTION, _PROMPT_HASH_AS_OF, EXAMPLE_CONTEXT, few_shot_set=0
    )
    blob = "\n".join(f"{m['role']}\n{m['content']}" for m in messages).encode("utf-8")
    assert hashlib.sha256(blob).hexdigest() == _MAIN_PROMPT_SHA256


def test_customers_by_code_walk_in_excluded(tmp_path: Path) -> None:
    june = MonthExpr(month=6, year=2026)
    orders = [
        _order("o1", datetime.date(2026, 6, 1), customer_id="C01", customer_name="An"),
        _order("o2", datetime.date(2026, 6, 2), customer_id="C01", customer_name="An"),
        _order("o3", datetime.date(2026, 6, 3), customer_id="C02", customer_name="Bình"),
        _order(
            "o4",
            datetime.date(2026, 6, 4),
            customer_id=None,
            customer_name="Khách lẻ",
            is_walk_in=True,
        ),
        _order("o5", datetime.date(2026, 6, 5), customer_id="C02", customer_name="Bình"),
        _order("o6", datetime.date(2026, 6, 6), customer_id="C02", customer_name="Bình"),
    ]
    lines = [_line(oid, line_amount=10_000) for oid in ("o1", "o2", "o3", "o4", "o5", "o6")]
    engine = _engine(tmp_path / "codes.duckdb", orders, lines)

    count = engine.answer_ext(
        CustomersMinOrdersQuery(n=2, output="count", period=june), AS_OF
    )
    assert count.type == "number"
    assert count.answer == 2  # C01 and C02; walk-in ignored

    share = engine.answer_ext(
        CustomersMinOrdersQuery(n=3, output="share", period=june), AS_OF
    )
    # C02 has 3 orders; 2 customers bought -> 50%
    assert share.type == "percent"
    assert share.answer == 50.0

    top = engine.answer_ext(
        TopCustomersQuery(k=1, by="order_count", mode="max", period=june), AS_OF
    )
    assert top.type == "string"
    assert top.answer == "Bình"


def test_customers_by_name_spelling_merge(tmp_path: Path) -> None:
    june = MonthExpr(month=6, year=2026)
    # No customer_id -> by_name. Accent variants merge; different letters do not.
    orders = [
        _order("o1", datetime.date(2026, 6, 1), customer_name="Nguyễn Văn A"),
        _order("o2", datetime.date(2026, 6, 2), customer_name="Nguyen Van A"),
        _order("o3", datetime.date(2026, 6, 3), customer_name="Nguyễn Văn B"),
        _order(
            "o4",
            datetime.date(2026, 6, 4),
            customer_name="Khách lẻ",
            is_walk_in=True,
        ),
    ]
    lines = [
        _line("o1", line_amount=50_000),
        _line("o2", line_amount=30_000),
        _line("o3", line_amount=100_000),
        _line("o4", line_amount=999_000),
    ]
    engine = _engine(tmp_path / "names.duckdb", orders, lines)

    count = engine.answer_ext(
        CustomersMinOrdersQuery(n=2, output="count", period=june), AS_OF
    )
    assert count.answer == 1  # folded A has 2 orders; B has 1

    top_rev = engine.answer_ext(
        TopCustomersQuery(k=1, by="revenue", mode="max", period=june), AS_OF
    )
    # B has 100k; A has 80k combined
    assert top_rev.answer == "Nguyễn Văn B"

    # Different letters stay separate: count of customers who bought = 2
    all_cust = engine.answer_ext(
        CustomersMinOrdersQuery(n=1, output="count", period=june), AS_OF
    )
    assert all_cust.answer == 2


def test_top_customers_tie_by_code(tmp_path: Path) -> None:
    june = MonthExpr(month=6, year=2026)
    orders = [
        _order("o1", datetime.date(2026, 6, 1), customer_id="C02", customer_name="Zoe"),
        _order("o2", datetime.date(2026, 6, 2), customer_id="C01", customer_name="Ann"),
    ]
    lines = [_line("o1", line_amount=50_000), _line("o2", line_amount=50_000)]
    engine = _engine(tmp_path / "tie.duckdb", orders, lines)
    top = engine.answer_ext(
        TopCustomersQuery(k=2, by="revenue", mode="max", period=june), AS_OF
    )
    assert top.type == "list"
    assert top.answer == ["Ann", "Zoe"]  # tie on 50k -> C01 before C02


def test_threshold_boundary(tmp_path: Path) -> None:
    june = MonthExpr(month=6, year=2026)
    orders = [
        _order("o1", datetime.date(2026, 6, 1), customer_id="C1", customer_name="A"),
        _order("o2", datetime.date(2026, 6, 1), customer_id="C2", customer_name="B"),
        _order("o3", datetime.date(2026, 6, 2), customer_id="C3", customer_name="C"),
        _order("o4", datetime.date(2026, 6, 3), customer_id="C4", customer_name="D"),
    ]
    lines = [
        _line("o1", line_amount=100_000),
        _line("o2", line_amount=100_000),
        _line("o3", line_amount=200_000),
        _line("o4", line_amount=50_000),
    ]
    engine = _engine(tmp_path / "thresh.duckdb", orders, lines)

    ge = engine.answer_ext(
        OrdersMinAmountQuery(x=100_000, op=">=", period=june), AS_OF
    )
    assert ge.answer == 3  # 100k, 100k, 200k

    gt = engine.answer_ext(
        OrdersMinAmountQuery(x=100_000, op=">", period=june), AS_OF
    )
    assert gt.answer == 1  # only 200k

    # Day 1 revenue = 200k, day 2 = 200k, day 3 = 50k
    days_ge = engine.answer_ext(
        DaysMinRevenueQuery(x=200_000, op=">=", period=june), AS_OF
    )
    assert days_ge.answer == 2

    days_gt = engine.answer_ext(
        DaysMinRevenueQuery(x=200_000, op=">", period=june), AS_OF
    )
    assert days_gt.answer == 0


def test_no_customer_data_unanswerable(tmp_path: Path) -> None:
    june = MonthExpr(month=6, year=2026)
    orders = [
        _order("o1", datetime.date(2026, 6, 1), is_walk_in=True, customer_name=None),
    ]
    lines = [_line("o1", line_amount=10_000)]
    engine = _engine(tmp_path / "empty.duckdb", orders, lines)
    ans = engine.answer_ext(
        CustomersMinOrdersQuery(n=1, output="count", period=june), AS_OF
    )
    assert ans.type == "unanswerable"


def test_describe_e4b_families() -> None:
    rules = RuleConfig()
    cal = LunarCalendar({})
    june = MonthExpr(month=6, year=2026)
    who = "customers identified by customer code, walk-in orders excluded"

    cmo = describe_ext_spec(
        CustomersMinOrdersQuery(n=3, output="share", period=june),
        AS_OF,
        rules,
        cal,
        customer_who=who,
    )
    assert who in cmo
    assert "at least 3" in cmo
    assert "all customers who bought in the period" in cmo

    top = describe_ext_spec(
        TopCustomersQuery(k=1, by="revenue", mode="max", period=june),
        AS_OF,
        rules,
        cal,
        customer_who=who,
    )
    assert "money revenue" in top
    assert who in top

    oma = describe_ext_spec(
        OrdersMinAmountQuery(x=50000, op=">=", period=june), AS_OF, rules, cal
    )
    assert ">= 50000 VND" in oma
    assert "order count" in oma

    dmr = describe_ext_spec(
        DaysMinRevenueQuery(x=1000000, op=">", period=june), AS_OF, rules, cal
    )
    assert "> 1000000 VND" in dmr
    assert "day count" in dmr
    assert "never qualify" in dmr


def _ext_period(**fields: object) -> dict[str, object]:
    from datamate.query.parse_schema import PERIOD_FIELDS

    row: dict[str, object] = {name: None for name in PERIOD_FIELDS}
    row.update(fields)
    return row


def test_convert_e4b_threshold_and_share() -> None:
    from datamate.query.ext.convert import to_ext_query_spec

    as_of = datetime.date(2026, 6, 30)
    period = _ext_period(kind="month", month=6, year=2026)

    share = to_ext_query_spec(
        {
            "family": "customers_min_orders",
            "n": 2,
            "output": "share",
            "period": period,
        },
        as_of,
    )
    assert isinstance(share, CustomersMinOrdersQuery)
    assert share.output == "share"

    oma = to_ext_query_spec(
        {
            "family": "orders_min_amount",
            "x": 100000,
            "op": ">",
            "period": period,
        },
        as_of,
    )
    assert isinstance(oma, OrdersMinAmountQuery)
    assert oma.op == ">"
    assert oma.x == 100000

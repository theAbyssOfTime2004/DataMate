"""Synthetic DuckDB coverage for E4c debt families."""

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
from datamate.etl.schema import DEBTS_COLUMNS, PAYMENTS_COLUMNS
from datamate.metrics import MetricEngine
from datamate.query.ext.describe import describe_ext_spec
from datamate.query.ext.spec import (
    AgingBucket,
    DebtAddedQuery,
    DebtAgingQuery,
    DebtCollectedQuery,
    DebtTotalQuery,
    TopDebtorsQuery,
)
from datamate.query.prompts import EXAMPLE_CONTEXT, build_messages
from datamate.rules import RuleConfig
from datamate.timeres import DayExpr, LunarCalendar, MonthExpr, TodayExpr

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
    payable: int,
    customer_id: str | None = None,
    customer_name: str | None = None,
    is_walk_in: bool = False,
    status: str = "completed",
    payment_method: str | None = "transfer",
    channel: str | None = "store",
    amount_paid: int | None = None,
) -> dict[str, object]:
    return {
        "order_id": order_id,
        "order_date": day,
        "order_ts": datetime.datetime.combine(day, datetime.time(10, 0)),
        "status": status,
        "has_return": status == "returned",
        "customer_key": customer_id or customer_name,
        "customer_id": customer_id,
        "customer_name": customer_name,
        "is_walk_in": is_walk_in,
        "payment_method": payment_method,
        "channel": channel,
        "order_payable": payable,
        "amount_paid": amount_paid,
    }


def _pay(
    payment_id: str,
    order_id: str,
    day: datetime.date,
    amount: int,
) -> dict[str, object]:
    return {
        "payment_id": payment_id,
        "order_id": order_id,
        "pay_date": day,
        "amount": amount,
    }


def _line(order_id: str, line_amount: int = 1) -> dict[str, object]:
    return {
        "order_id": order_id,
        "line_no": 1,
        "product_key": "p1",
        "product_id": "P01",
        "product_name": "Trà",
        "qty": Decimal(1),
        "unit": "ly",
        "qty_base": Decimal(1),
        "unit_price": line_amount,
        "line_amount": line_amount,
        "cost_per_unit": 100,
        "is_return_line": False,
        "src_row": 1,
    }


def _engine(
    path: Path,
    orders: list[dict],
    payments: list[dict] | None = None,
    debts: list[dict] | None = None,
) -> MetricEngine:
    lines = [_line(o["order_id"]) for o in orders]
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
            [{"product_key": "p1", "product_id": "P01", "product_name": "Trà"}],
        ),
        customers=_frame(CUSTOMERS_COLUMNS, customers),
        unit_conversions=_frame(UNIT_CONVERSIONS_COLUMNS, []),
        payments=_frame(PAYMENTS_COLUMNS, payments or []),
        debts=_frame(DEBTS_COLUMNS, debts or []),
        report=EtlReport(),
    )
    write_duckdb(canon, path)
    con = duckdb.connect(str(path), read_only=True)
    return MetricEngine(con, RuleConfig(), LunarCalendar({}), DATA_START, DATA_END)


def test_main_prompt_hash_unchanged() -> None:
    messages = build_messages(
        _PROMPT_HASH_QUESTION, _PROMPT_HASH_AS_OF, EXAMPLE_CONTEXT, few_shot_set=0
    )
    blob = "\n".join(f"{m['role']}\n{m['content']}" for m in messages).encode("utf-8")
    assert hashlib.sha256(blob).hexdigest() == _MAIN_PROMPT_SHA256


def test_debt_partial_payments_cod_returned_walk_in(tmp_path: Path) -> None:
    """Partial payments, COD excluded, returned still owed, walk-in in total not table."""
    as_of_day = datetime.date(2026, 6, 15)
    orders = [
        # C01: payable 100k, paid 40k on day 1 + 20k on day 10 -> remaining 40k
        _order(
            "o1",
            datetime.date(2026, 6, 1),
            payable=100_000,
            customer_id="C01",
            customer_name="An",
        ),
        # C02: payable 50k, paid in full on order day -> remaining 0
        _order(
            "o2",
            datetime.date(2026, 6, 5),
            payable=50_000,
            customer_id="C02",
            customer_name="Bình",
        ),
        # COD: excluded even if unpaid
        _order(
            "o3",
            datetime.date(2026, 6, 2),
            payable=80_000,
            customer_id="C03",
            customer_name="Cường",
            payment_method="COD",
        ),
        # Returned but still owed 30k
        _order(
            "o4",
            datetime.date(2026, 6, 3),
            payable=30_000,
            customer_id="C04",
            customer_name="Dung",
            status="returned",
        ),
        # Walk-in unpaid 25k -> in total, not in top_debtors
        _order(
            "o5",
            datetime.date(2026, 6, 4),
            payable=25_000,
            customer_name="Khách lẻ",
            is_walk_in=True,
        ),
        # Cancelled unpaid -> excluded
        _order(
            "o6",
            datetime.date(2026, 6, 6),
            payable=99_000,
            customer_id="C99",
            customer_name="Huỷ",
            status="cancelled",
        ),
    ]
    payments = [
        _pay("p1", "o1", datetime.date(2026, 6, 1), 40_000),
        _pay("p2", "o1", datetime.date(2026, 6, 10), 20_000),
        _pay("p3", "o2", datetime.date(2026, 6, 5), 50_000),
    ]
    engine = _engine(tmp_path / "debt.duckdb", orders, payments)
    day = DayExpr(date=as_of_day)

    total = engine.answer_ext(DebtTotalQuery(date=day), AS_OF)
    # 40k (An) + 30k (Dung returned) + 25k (walk-in) = 95k; COD/cancelled out
    assert total.type == "number"
    assert total.answer == 95_000

    debtors = engine.answer_ext(DebtTotalQuery(date=day, output="customer_count"), AS_OF)
    # An and Dung still owe; the walk-in order is not a customer.
    assert debtors.type == "number"
    assert debtors.answer == 2

    top = engine.answer_ext(TopDebtorsQuery(k=5, date=day), AS_OF)
    assert top.type == "table"
    rows = top.answer["rows"]
    names = [r[0] for r in rows]
    assert "Khách lẻ" not in names
    assert names[0] == "An"
    assert rows[0][1] == 40_000
    assert "Dung" in names

    # Aging at 2026-06-15: o1 age=14, o4 age=12, o5 age=11
    aging = engine.answer_ext(
        DebtAgingQuery(
            date=day,
            buckets=[
                AgingBucket(lo=0, hi=7),
                AgingBucket(lo=8, hi=13),
                AgingBucket(lo=14, hi=None),
            ],
        ),
        AS_OF,
    )
    assert aging.type == "list"
    assert aging.answer == [0, 55_000, 40_000]  # o4+o5 in 8-13; o1 in 14+

    # Today boundary: age 0 on as_of = order day
    today_order = [
        _order(
            "t1",
            AS_OF,
            payable=10_000,
            customer_id="T1",
            customer_name="Hôm nay",
            amount_paid=0,
        )
    ]
    eng2 = _engine(tmp_path / "age0.duckdb", today_order, payments=[])
    age0 = eng2.answer_ext(
        DebtAgingQuery(
            date=TodayExpr(),
            buckets=[AgingBucket(lo=0, hi=0), AgingBucket(lo=1, hi=None)],
        ),
        AS_OF,
    )
    assert age0.answer == [10_000, 0]


def test_debt_added_and_collected(tmp_path: Path) -> None:
    orders = [
        _order(
            "o1",
            datetime.date(2026, 6, 1),
            payable=100_000,
            customer_id="C01",
            customer_name="An",
        ),
        _order(
            "o2",
            datetime.date(2026, 5, 20),
            payable=200_000,
            customer_id="C02",
            customer_name="Bình",
        ),
        _order(
            "o3",
            datetime.date(2026, 6, 10),
            payable=50_000,
            customer_id="C03",
            customer_name="COD",
            payment_method="COD",
        ),
    ]
    payments = [
        # Same-day pay on o1 -> not collected later; unpaid same day = 60k added
        _pay("p1", "o1", datetime.date(2026, 6, 1), 40_000),
        # Later collection in June on May order
        _pay("p2", "o2", datetime.date(2026, 6, 5), 70_000),
        # Same-day on COD ignored for debt_added/collected
        _pay("p3", "o3", datetime.date(2026, 6, 10), 10_000),
    ]
    engine = _engine(tmp_path / "flow.duckdb", orders, payments)
    june = MonthExpr(month=6, year=2026)

    added = engine.answer_ext(DebtAddedQuery(period=june), AS_OF)
    # o1 unpaid on order day 60k; o3 COD excluded
    assert added.answer == 60_000

    collected = engine.answer_ext(DebtCollectedQuery(period=june), AS_OF)
    # only p2 (70k); p1 is same day; COD payment excluded via order join
    assert collected.answer == 70_000


def test_debt_unanswerable_without_payable(tmp_path: Path) -> None:
    orders = [
        _order(
            "o1",
            datetime.date(2026, 6, 1),
            payable=100_000,
            customer_id="C01",
            customer_name="An",
        )
    ]
    # Clear payable to simulate missing column data
    orders[0]["order_payable"] = None
    engine = _engine(tmp_path / "nopay.duckdb", orders, payments=[])
    ans = engine.answer_ext(
        DebtTotalQuery(date=DayExpr(date=datetime.date(2026, 6, 15))), AS_OF
    )
    assert ans.type == "unanswerable"


def test_describe_e4c_families() -> None:
    rules = RuleConfig()
    cal = LunarCalendar({})
    day = DayExpr(date=datetime.date(2026, 6, 15))
    june = MonthExpr(month=6, year=2026)
    meta = {
        "payment_source": "payments table",
        "exclusions": "COD",
        "customer_who": "customers identified by customer code, walk-in orders excluded",
    }

    total = describe_ext_spec(
        DebtTotalQuery(date=day), AS_OF, rules, cal, debt_meta=meta
    )
    assert "end of 2026-06-15" in total
    assert "payments table" in total
    assert "COD" in total
    assert "one number (VND)" in total

    top = describe_ext_spec(
        TopDebtorsQuery(k=3, date=day), AS_OF, rules, cal, debt_meta=meta
    )
    assert "top 3" in top
    assert "table [customer, amount]" in top

    aging = describe_ext_spec(
        DebtAgingQuery(
            date=day,
            buckets=[AgingBucket(lo=0, hi=7), AgingBucket(lo=8, hi=None)],
        ),
        AS_OF,
        rules,
        cal,
        debt_meta=meta,
    )
    assert "0-7, 8+" in aging
    assert "list of amounts" in aging

    added = describe_ext_spec(
        DebtAddedQuery(period=june), AS_OF, rules, cal, debt_meta=meta
    )
    assert "unpaid part on the order day" in added

    collected = describe_ext_spec(
        DebtCollectedQuery(period=june), AS_OF, rules, cal, debt_meta=meta
    )
    assert "payments after the order day" in collected


def test_convert_debt_aging_buckets() -> None:
    from datamate.query.ext.convert import to_ext_query_spec
    from datamate.query.parse_schema import PERIOD_FIELDS

    as_of = datetime.date(2026, 6, 30)
    period = {name: None for name in PERIOD_FIELDS}
    period.update({"kind": "today"})
    spec = to_ext_query_spec(
        {
            "family": "debt_aging",
            "date": period,
            "buckets": [[0, 7], [8, 30], [31, None]],
        },
        as_of,
    )
    assert isinstance(spec, DebtAgingQuery)
    assert len(spec.buckets) == 3
    assert spec.buckets[2].hi is None


def test_debt_exclusion_matches_labels_inside_shop_values() -> None:
    from datamate.metrics.ext_families import _contains_words, _label_words

    assert _contains_words(_label_words("thu hộ (cod)"), _label_words("cod"))
    assert _contains_words(_label_words("sàn tmđt"), _label_words("sàn tmđt"))
    assert not _contains_words(_label_words("codex"), _label_words("cod"))
    assert not _contains_words(_label_words("sân bay"), _label_words("sàn"))


def test_debt_abstains_when_amount_due_exceeds_payable(tmp_path: Path) -> None:
    day = datetime.date(2026, 6, 10)
    orders = [
        {**_order("o1", day, payable=120_000, customer_id="KH1"), "amount_due": 168_000},
        {**_order("o2", day, payable=0, customer_id="KH2"), "amount_due": 50_000},
    ]
    payments = [_pay("p1", "o1", day, 120_000)]
    engine = _engine(tmp_path / "bad.duckdb", orders, payments)
    total = engine.answer_ext(DebtTotalQuery(date=DayExpr(date=day)), AS_OF)
    assert total.type == "abstain"

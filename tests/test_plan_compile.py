"""E3 QueryPlan compile + execute on a synthetic shop."""

from __future__ import annotations

import datetime
from decimal import Decimal
from pathlib import Path

import duckdb
import pandas as pd
import pytest

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
from datamate.etl.schema import PURCHASES_COLUMNS
from datamate.metrics import MetricEngine
from datamate.query.plan.execute import answer_plan, try_answer_plan
from datamate.query.plan.spec import (
    ArgMaxOut,
    ArgMinOut,
    AvgPerPeriodOut,
    CountEntitiesOut,
    HavingClause,
    PlanFilters,
    QueryPlan,
    ShareOfTotalOut,
    TopKOut,
    ValueOut,
)
from datamate.rules import RuleConfig
from datamate.timeres import LunarCalendar, MonthExpr

AS_OF = datetime.date(2026, 6, 30)
JUNE = MonthExpr(month=6, year=2026)


def _frame(columns: tuple[str, ...], rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = None
    return frame.loc[:, list(columns)]


@pytest.fixture
def engine(tmp_path: Path) -> MetricEngine:
    orders = [
        {
            "order_id": "o1",
            "order_date": datetime.date(2026, 6, 1),
            "order_ts": datetime.datetime(2026, 6, 1, 9, 0),
            "status": "completed",
            "has_return": False,
            "customer_id": "C1",
            "customer_name": "An",
            "is_walk_in": False,
            "payment_method": "cash",
            "channel": "store",
            "employee_id": "E1",
            "order_subtotal": 10000,
            "order_payable": 10000,
        },
        {
            "order_id": "o2",
            "order_date": datetime.date(2026, 6, 1),
            "order_ts": datetime.datetime(2026, 6, 1, 15, 0),
            "status": "completed",
            "has_return": False,
            "customer_id": "C1",
            "customer_name": "An",
            "is_walk_in": False,
            "payment_method": "cash",
            "channel": "store",
            "employee_id": "E1",
            "order_subtotal": 5000,
            "order_payable": 5000,
        },
        {
            "order_id": "o3",
            "order_date": datetime.date(2026, 6, 2),
            "order_ts": datetime.datetime(2026, 6, 2, 10, 0),
            "status": "completed",
            "has_return": False,
            "customer_id": "C2",
            "customer_name": "Bình",
            "is_walk_in": False,
            "payment_method": "transfer",
            "channel": "store",
            "employee_id": "E2",
            "order_subtotal": 20000,
            "order_payable": 20000,
        },
        {
            "order_id": "o4",
            "order_date": datetime.date(2026, 6, 3),
            "order_ts": datetime.datetime(2026, 6, 3, 11, 0),
            "status": "cancelled",
            "has_return": False,
            "customer_id": "C2",
            "customer_name": "Bình",
            "is_walk_in": False,
            "payment_method": "cash",
            "channel": "store",
            "employee_id": "E2",
            "order_subtotal": 1000,
            "order_payable": 1000,
        },
        {
            "order_id": "o5",
            "order_date": datetime.date(2026, 6, 4),
            "order_ts": datetime.datetime(2026, 6, 4, 12, 0),
            "status": "completed",
            "has_return": False,
            "customer_id": None,
            "customer_name": "Khách lẻ",
            "is_walk_in": True,
            "payment_method": "cash",
            "channel": "store",
            "employee_id": "E1",
            "order_subtotal": 3000,
            "order_payable": 3000,
        },
    ]
    lines = [
        {
            "order_id": "o1",
            "line_no": 1,
            "product_key": "tra",
            "product_id": "P1",
            "product_name": "Trà",
            "qty": Decimal(2),
            "qty_base": Decimal(2),
            "unit_price": 5000,
            "line_amount": 10000,
            "cost_per_unit": 2000,
            "is_return_line": False,
        },
        {
            "order_id": "o2",
            "line_no": 1,
            "product_key": "banh",
            "product_id": "P2",
            "product_name": "Bánh",
            "qty": Decimal(1),
            "qty_base": Decimal(1),
            "unit_price": 5000,
            "line_amount": 5000,
            "cost_per_unit": 1000,
            "is_return_line": False,
        },
        {
            "order_id": "o3",
            "line_no": 1,
            "product_key": "tra",
            "product_id": "P1",
            "product_name": "Trà",
            "qty": Decimal(4),
            "qty_base": Decimal(4),
            "unit_price": 5000,
            "line_amount": 20000,
            "cost_per_unit": 2000,
            "is_return_line": False,
        },
        {
            # Fee line: never revenue or quantity.
            "order_id": "o3",
            "line_no": 2,
            "product_key": None,
            "product_id": None,
            "product_name": "Phí phục vụ",
            "qty": Decimal(1),
            "qty_base": Decimal(1),
            "unit_price": 7000,
            "line_amount": 7000,
            "cost_per_unit": 0,
            "is_return_line": False,
        },
        {
            "order_id": "o5",
            "line_no": 1,
            "product_key": "banh",
            "product_id": "P2",
            "product_name": "Bánh",
            "qty": Decimal(1),
            "qty_base": Decimal(1),
            "unit_price": 3000,
            "line_amount": 3000,
            "cost_per_unit": 1000,
            "is_return_line": False,
        },
    ]
    products = [
        {"product_key": "tra", "product_id": "P1", "product_name": "Trà", "category": "đồ uống"},
        {"product_key": "banh", "product_id": "P2", "product_name": "Bánh", "category": "đồ ăn"},
    ]
    purchases = [
        {
            "receipt_id": "r1",
            "receipt_date": datetime.date(2026, 6, 1),
            "supplier_id": "S1",
            "supplier_name": "Nhà A",
            "is_cancelled": False,
            "total_amount": 100000,
        },
        {
            "receipt_id": "r2",
            "receipt_date": datetime.date(2026, 6, 5),
            "supplier_id": "S2",
            "supplier_name": "Nhà B",
            "is_cancelled": False,
            "total_amount": 50000,
        },
        {
            "receipt_id": "r3",
            "receipt_date": datetime.date(2026, 6, 6),
            "supplier_id": "S1",
            "supplier_name": "Nhà A",
            "is_cancelled": True,
            "total_amount": 999,
        },
    ]
    data = CanonicalData(
        orders=_frame(ORDERS_COLUMNS, orders),
        order_lines=_frame(ORDER_LINES_COLUMNS, lines),
        products=_frame(PRODUCTS_COLUMNS, products),
        unit_conversions=_frame(UNIT_CONVERSIONS_COLUMNS, []),
        customers=_frame(CUSTOMERS_COLUMNS, []),
        report=EtlReport(),
        purchases=_frame(PURCHASES_COLUMNS, purchases),
    )
    db = tmp_path / "shop.duckdb"
    write_duckdb(data, db)
    con = duckdb.connect(str(db), read_only=True)
    return MetricEngine(
        con,
        RuleConfig(),
        LunarCalendar({}),
        datetime.date(2026, 1, 1),
        datetime.date(2026, 6, 30),
    )


def _plan(**kwargs: object) -> QueryPlan:
    base = {
        "entity": "none",
        "measure": "revenue",
        "filters": PlanFilters(period=JUNE),
        "having": [],
        "output": ValueOut(),
    }
    base.update(kwargs)
    return QueryPlan.model_validate(base)


def test_shop_revenue_value(engine: MetricEngine) -> None:
    # completed non-return: o1 10000 + o2 5000 + o3 20000 + o5 3000 = 38000
    ans, _ = answer_plan(_plan(entity="none", measure="revenue", output=ValueOut()), engine, AS_OF)
    assert ans.type == "number"
    assert ans.answer == 38000


def test_customer_argmax_with_having(engine: MetricEngine) -> None:
    plan = _plan(
        entity="customer",
        measure="revenue",
        having=[HavingClause(measure="order_count", op=">=", value=2)],
        output=ArgMaxOut(),
    )
    ans, _ = answer_plan(plan, engine, AS_OF)
    assert ans.type == "string"
    assert ans.answer == "An"  # C1 has 2 orders, 15000; C2 has 1 completed order


def test_peak_hour_is_a_number(engine: MetricEngine) -> None:
    # Peak hour answers are the slot's start hour, type number.
    plan = _plan(entity="hour", measure="order_count", scope="all_orders", output=ArgMaxOut())
    ans, _ = answer_plan(plan, engine, AS_OF)
    assert ans.type == "number"
    assert ans.answer == 9  # every hour has one order; tie goes to the earliest


def test_argmin_over_time_buckets_abstains(engine: MetricEngine) -> None:
    plan = _plan(entity="hour", measure="order_count", scope="all_orders", output=ArgMinOut())
    ans, _, reason = try_answer_plan(plan, engine, AS_OF)
    assert ans.type == "abstain"
    assert reason is not None and reason.startswith("plan_compile")


def test_weekday_ranking_abstains(engine: MetricEngine) -> None:
    plan = _plan(entity="weekday", measure="order_count", output=ArgMaxOut())
    ans, _, reason = try_answer_plan(plan, engine, AS_OF)
    assert ans.type == "abstain"
    assert reason is not None and reason.startswith("plan_compile")


def test_revenue_excludes_fee_lines(engine: MetricEngine) -> None:
    # o3 carries a 7000 fee line; revenue stays on product lines like the main engine.
    plan = _plan(entity="payment_method", measure="revenue", output=ArgMaxOut())
    ans, _ = answer_plan(plan, engine, AS_OF)
    assert ans.answer == "transfer"  # transfer 20000 vs cash 18000 (27000 with the fee)
    total, _ = answer_plan(_plan(measure="revenue", output=ValueOut()), engine, AS_OF)
    assert total.answer == 38000


def test_category_filter_applies_to_lines(engine: MetricEngine) -> None:
    # Product ranking inside a category ranks only that category's products.
    plan = _plan(
        entity="product",
        measure="revenue",
        filters=PlanFilters(period=JUNE, category="đồ ăn"),
        output=ArgMaxOut(),
    )
    ans, _ = answer_plan(plan, engine, AS_OF)
    assert ans.answer == "Bánh"


def test_avg_order_value_rounds_to_dong(engine: MetricEngine) -> None:
    # 38000 product revenue over 4 revenue orders = 9500 (the fee line is not revenue).
    plan = _plan(measure="avg_order_value", output=ValueOut())
    ans, _ = answer_plan(plan, engine, AS_OF)
    assert ans.answer == 9500


def test_avg_per_weekday_counts_every_such_day(engine: MetricEngine) -> None:
    # June 2026 has 5 Mondays (1, 8, 15, 22, 29); Monday orders: o1, o2 (revenue).
    plan = _plan(
        measure="order_count",
        filters=PlanFilters(period=JUNE, weekday=0),
        output=AvgPerPeriodOut(unit="weekday_occurrence"),
    )
    ans, _ = answer_plan(plan, engine, AS_OF)
    assert ans.answer == 0.4


def test_shop_share_of_a_filter(engine: MetricEngine) -> None:
    # Transfer revenue 20000 of 38000 total = 52.63 %.
    plan = _plan(
        measure="revenue",
        filters=PlanFilters(period=JUNE, payment_method="transfer"),
        output=ShareOfTotalOut(),
    )
    ans, _ = answer_plan(plan, engine, AS_OF)
    assert ans.type == "percent"
    assert ans.answer == 52.63


def test_returned_count_ignores_revenue_scope(engine: MetricEngine) -> None:
    plan = _plan(measure="returned_order_count", scope="revenue_orders", output=ValueOut())
    assert plan.scope == "all_orders"


def test_product_topk_qty(engine: MetricEngine) -> None:
    plan = _plan(
        entity="product",
        measure="qty_base",
        output=TopKOut(k=2),
    )
    ans, _ = answer_plan(plan, engine, AS_OF)
    assert ans.type == "list"
    assert ans.answer[0] == "Trà"  # qty 2+4=6 vs bánh 1+1=2


def test_category_count(engine: MetricEngine) -> None:
    plan = _plan(
        entity="category",
        measure="revenue",
        output=CountEntitiesOut(),
    )
    ans, _ = answer_plan(plan, engine, AS_OF)
    assert ans.type == "number"
    assert ans.answer == 2


def test_payment_method_argmax(engine: MetricEngine) -> None:
    plan = _plan(
        entity="payment_method",
        measure="order_count",
        output=ArgMaxOut(),
    )
    ans, _ = answer_plan(plan, engine, AS_OF)
    assert ans.type == "string"
    assert ans.answer == "cash"


def test_employee_argmax_revenue(engine: MetricEngine) -> None:
    plan = _plan(entity="employee", measure="revenue", output=ArgMaxOut())
    ans, _ = answer_plan(plan, engine, AS_OF)
    assert ans.type == "string"
    assert ans.answer in {"E1", "E2"}


def test_day_entity(engine: MetricEngine) -> None:
    plan = _plan(entity="day", measure="order_count", output=ArgMaxOut())
    ans, _ = answer_plan(plan, engine, AS_OF)
    assert ans.type == "date"
    assert ans.answer == "2026-06-01"  # two completed orders that day


def test_supplier_purchase(engine: MetricEngine) -> None:
    plan = _plan(
        entity="supplier",
        measure="purchase_amount",
        output=ArgMaxOut(),
    )
    ans, _ = answer_plan(plan, engine, AS_OF)
    assert ans.type == "string"
    assert ans.answer == "Nhà A"


def test_receipt_count_shop(engine: MetricEngine) -> None:
    plan = _plan(entity="none", measure="receipt_count", output=ValueOut())
    ans, _ = answer_plan(plan, engine, AS_OF)
    assert ans.type == "number"
    assert ans.answer == 2  # cancelled excluded


def test_cancelled_count(engine: MetricEngine) -> None:
    plan = _plan(
        entity="none",
        measure="cancelled_order_count",
        scope="cancelled_orders",
        output=ValueOut(),
    )
    ans, _ = answer_plan(plan, engine, AS_OF)
    assert ans.type == "number"
    assert ans.answer == 1

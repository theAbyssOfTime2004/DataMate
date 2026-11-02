"""Synthetic coverage for E4d purchase / stock / lot / drug / margin families."""

from __future__ import annotations

import datetime
import hashlib
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
from datamate.etl.schema import (
    DRUG_INFO_COLUMNS,
    EMPLOYEES_COLUMNS,
    EXPENSES_COLUMNS,
    LOTS_COLUMNS,
    PURCHASE_LINES_COLUMNS,
    PURCHASES_COLUMNS,
    SALE_LOTS_COLUMNS,
    STOCK_MOVES_COLUMNS,
)
from datamate.metrics import MetricEngine
from datamate.query.ext.describe import describe_ext_spec
from datamate.query.ext.spec import (
    AvgPurchaseCostQuery,
    EmployeeRevenueQuery,
    GrossProfitQuery,
    IngredientRevenueQuery,
    LotsExpiringQuery,
    NearestExpiryLotQuery,
    NetProfitQuery,
    OtcRevenueQuery,
    OutOfStockCountQuery,
    PurchaseTotalQuery,
    RxRevenueQuery,
    RxShareQuery,
    StockOnHandQuery,
    StockValueQuery,
    TopEmployeeQuery,
    TopSupplierQuery,
)
from datamate.query.prompts import EXAMPLE_CONTEXT, build_messages
from datamate.rules import RuleConfig
from datamate.timeres import DayExpr, LunarCalendar, MonthExpr, TodayExpr

AS_OF = datetime.date(2026, 6, 30)
DATA_START = datetime.date(2026, 1, 1)
DATA_END = datetime.date(2026, 6, 30)
_MAIN_PROMPT_SHA256 = "a9247b210f70f7e10dc1526829f104f1086c8648d16fab195fc6913726078729"


def _frame(columns: tuple[str, ...], rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = None
    return frame.loc[:, list(columns)]


def _engine(path: Path, **tables: pd.DataFrame) -> MetricEngine:
    empty = {
        "orders": _frame(ORDERS_COLUMNS, []),
        "order_lines": _frame(ORDER_LINES_COLUMNS, []),
        "products": _frame(PRODUCTS_COLUMNS, []),
        "customers": _frame(CUSTOMERS_COLUMNS, []),
        "unit_conversions": _frame(UNIT_CONVERSIONS_COLUMNS, []),
        "purchases": _frame(PURCHASES_COLUMNS, []),
        "purchase_lines": _frame(PURCHASE_LINES_COLUMNS, []),
        "stock_moves": _frame(STOCK_MOVES_COLUMNS, []),
        "lots": _frame(LOTS_COLUMNS, []),
        "sale_lots": _frame(SALE_LOTS_COLUMNS, []),
        "expenses": _frame(EXPENSES_COLUMNS, []),
        "employees": _frame(EMPLOYEES_COLUMNS, []),
        "drug_info": _frame(DRUG_INFO_COLUMNS, []),
    }
    empty.update(tables)
    canon = CanonicalData(report=EtlReport(), **empty)
    write_duckdb(canon, path)
    con = duckdb.connect(str(path), read_only=True)
    return MetricEngine(con, RuleConfig(), LunarCalendar({}), DATA_START, DATA_END)


def test_main_prompt_hash_unchanged() -> None:
    messages = build_messages(
        "Doanh thu thang 6/2026 cua shop la bao nhieu",
        AS_OF,
        EXAMPLE_CONTEXT,
        few_shot_set=0,
    )
    blob = "\n".join(f"{m['role']}\n{m['content']}" for m in messages).encode("utf-8")
    assert hashlib.sha256(blob).hexdigest() == _MAIN_PROMPT_SHA256


def test_purchases_and_avg_cost(tmp_path: Path) -> None:
    purchases = _frame(
        PURCHASES_COLUMNS,
        [
            {
                "receipt_id": "r1",
                "receipt_date": datetime.date(2026, 6, 1),
                "supplier_id": "S1",
                "supplier_name": "Công ty A",
                "is_cancelled": False,
                "total_amount": 100_000,
            },
            {
                "receipt_id": "r2",
                "receipt_date": datetime.date(2026, 6, 2),
                "supplier_id": "S2",
                "supplier_name": "Công ty B",
                "is_cancelled": False,
                "total_amount": 250_000,
            },
            {
                "receipt_id": "r3",
                "receipt_date": datetime.date(2026, 6, 3),
                "supplier_id": "S1",
                "supplier_name": "Công ty A",
                "is_cancelled": True,
                "total_amount": 999_000,
            },
        ],
    )
    lines = _frame(
        PURCHASE_LINES_COLUMNS,
        [
            {
                "receipt_id": "r1",
                "receipt_date": datetime.date(2026, 6, 1),
                "product_key": "sữa",
                "product_name": "Sữa",
                "qty": Decimal(10),
                "unit": "hộp",
                "unit_cost": 5000,
            },
            {
                # Dated on the receipt header only.
                "receipt_id": "r2",
                "receipt_date": None,
                "product_key": "sữa",
                "product_name": "Sữa",
                "qty": Decimal(5),
                "unit": "hộp",
                "unit_cost": 8000,
            },
        ],
    )
    products = _frame(
        PRODUCTS_COLUMNS,
        [{"product_key": "sữa", "product_id": "P1", "product_name": "Sữa", "base_unit": "hộp"}],
    )
    engine = _engine(
        tmp_path / "p.duckdb",
        purchases=purchases,
        purchase_lines=lines,
        products=products,
    )
    june = MonthExpr(month=6, year=2026)
    total = engine.answer_ext(PurchaseTotalQuery(period=june), AS_OF)
    assert total.answer == 350_000
    top = engine.answer_ext(TopSupplierQuery(period=june), AS_OF)
    assert top.answer == "Công ty B"
    avg = engine.answer_ext(AvgPurchaseCostQuery(product="Sữa", period=june), AS_OF)
    # (10*5000 + 5*8000) / 15 = 6000
    assert avg.answer == 6000


def test_stock_big_unit_and_value_and_oos(tmp_path: Path) -> None:
    products = _frame(
        PRODUCTS_COLUMNS,
        [
            {
                "product_key": "nước cam",
                "product_id": "P1",
                "product_name": "Nước cam",
                "base_unit": "chai",
                "unit_cost": 2000,
            },
            {
                "product_key": "trà",
                "product_id": "P2",
                "product_name": "Trà",
                "base_unit": "gói",
                "unit_cost": 1000,
            },
        ],
    )
    conversions = _frame(
        UNIT_CONVERSIONS_COLUMNS,
        [
            {
                "product_key": "nước cam",
                "unit": "thùng",
                "base_unit": "chai",
                "factor": Decimal(24),
            }
        ],
    )
    moves = _frame(
        STOCK_MOVES_COLUMNS,
        [
            {
                "product_key": "nước cam",
                "move_date": datetime.date(2026, 6, 1),
                "qty_in": Decimal(50),
                "qty_out": Decimal(0),
                "balance": Decimal(50),
            },
            {
                "product_key": "nước cam",
                "move_date": datetime.date(2026, 6, 10),
                "qty_in": Decimal(0),
                "qty_out": Decimal(2),
                "balance": Decimal(48),
            },
            {
                "product_key": "trà",
                "move_date": datetime.date(2026, 6, 1),
                "qty_in": Decimal(5),
                "qty_out": Decimal(5),
                "balance": Decimal(0),
            },
        ],
    )
    purchase_lines = _frame(
        PURCHASE_LINES_COLUMNS,
        [
            {"receipt_id": "r1", "product_key": "nước cam", "product_name": "Nước cam"},
            {"receipt_id": "r1", "product_key": "trà", "product_name": "Trà"},
        ],
    )
    engine = _engine(
        tmp_path / "s.duckdb",
        products=products,
        unit_conversions=conversions,
        stock_moves=moves,
        purchase_lines=purchase_lines,
    )
    day = DayExpr(date=datetime.date(2026, 6, 15))
    stock = engine.answer_ext(
        StockOnHandQuery(product="Nước cam", date=day, unit="thùng"), AS_OF
    )
    assert stock.type == "list"
    assert stock.answer == [2, 0]  # 48 / 24 = 2 whole, 0 leftover

    value = engine.answer_ext(StockValueQuery(date=day), AS_OF)
    # nc 48*2000 = 96000; tra 0 skipped
    assert value.answer == 96_000

    oos = engine.answer_ext(OutOfStockCountQuery(date=day), AS_OF)
    assert oos.answer == 1  # trà


def test_lots_partly_sold_and_nearest(tmp_path: Path) -> None:
    products = _frame(
        PRODUCTS_COLUMNS,
        [{"product_key": "amox", "product_id": "P1", "product_name": "Amox", "base_unit": "viên"}],
    )
    lots = _frame(
        LOTS_COLUMNS,
        [
            {
                "lot_id": "L1",
                "product_key": "amox",
                "product_name": "Amox",
                "expiry_date": datetime.date(2026, 7, 10),
                "qty_received": Decimal(100),
                "unit_cost": 500,
            },
            {
                "lot_id": "L2",
                "product_key": "amox",
                "product_name": "Amox",
                "expiry_date": datetime.date(2026, 8, 1),
                "qty_received": Decimal(50),
                "unit_cost": 600,
            },
        ],
    )
    # product_key "amox" matches normalize_label("Amox")
    orders = _frame(
        ORDERS_COLUMNS,
        [
            {
                "order_id": "o1",
                "order_date": datetime.date(2026, 6, 1),
                "status": "completed",
                "has_return": False,
            }
        ],
    )
    sale_lots = _frame(
        SALE_LOTS_COLUMNS,
        [{"order_id": "o1", "lot_id": "L1", "product_key": "amox", "qty": Decimal(40)}],
    )
    engine = _engine(
        tmp_path / "lot.duckdb",
        products=products,
        lots=lots,
        orders=orders,
        sale_lots=sale_lots,
    )
    day = TodayExpr()
    exp = engine.answer_ext(
        LotsExpiringQuery(date=day, n_days=20, output="cost"), AS_OF
    )
    # L1 remaining 60 * 500 = 30000; L2 expiry Aug 1 is within 20 days from Jun 30
    # Jun 30 + 20 = Jul 20, so L2 (Aug 1) is OUT. Only L1.
    assert exp.answer == 30_000

    nearest = engine.answer_ext(
        NearestExpiryLotQuery(product="Amox", date=day), AS_OF
    )
    assert nearest.answer == "L1"


def test_drugs_margin_employee(tmp_path: Path) -> None:
    products = _frame(
        PRODUCTS_COLUMNS,
        [
            {"product_key": "rx1", "product_name": "Thuốc A", "base_unit": "viên"},
            {"product_key": "otc", "product_name": "Thuốc B", "base_unit": "viên"},
        ],
    )
    drug = _frame(
        DRUG_INFO_COLUMNS,
        [
            {
                "product_key": "rx1",
                "active_ingredient": "Paracetamol",
                "is_prescription": True,
            },
            {
                "product_key": "otc",
                "active_ingredient": "Vitamin C",
                "is_prescription": False,
            },
        ],
    )
    orders = _frame(
        ORDERS_COLUMNS,
        [
            {
                "order_id": "o1",
                "order_date": datetime.date(2026, 6, 1),
                "status": "completed",
                "has_return": False,
                "employee_id": "E1",
            },
            {
                "order_id": "o2",
                "order_date": datetime.date(2026, 6, 2),
                "status": "completed",
                "has_return": False,
                "employee_id": "E2",
            },
        ],
    )
    lines = _frame(
        ORDER_LINES_COLUMNS,
        [
            {
                "order_id": "o1",
                "product_key": "rx1",
                "product_name": "Thuốc A",
                "qty": Decimal(2),
                "line_amount": 100_000,
                "cost_per_unit": 20_000,
            },
            {
                "order_id": "o2",
                "product_key": "otc",
                "product_name": "Thuốc B",
                "qty": Decimal(1),
                "line_amount": 50_000,
                "cost_per_unit": 10_000,
            },
        ],
    )
    employees = _frame(
        EMPLOYEES_COLUMNS,
        [
            {"employee_id": "E1", "employee_name": "Lan"},
            {"employee_id": "E2", "employee_name": "Minh"},
        ],
    )
    expenses = _frame(
        EXPENSES_COLUMNS,
        [{"expense_date": datetime.date(2026, 6, 15), "amount": 30_000}],
    )
    engine = _engine(
        tmp_path / "m.duckdb",
        products=products,
        drug_info=drug,
        orders=orders,
        order_lines=lines,
        employees=employees,
        expenses=expenses,
    )
    june = MonthExpr(month=6, year=2026)
    ing = engine.answer_ext(
        IngredientRevenueQuery(active_ingredient="Paracetamol", period=june), AS_OF
    )
    assert ing.answer == 100_000
    rx = engine.answer_ext(RxShareQuery(period=june), AS_OF)
    # 100k / 150k = 66.67%
    assert rx.answer == 66.67
    gross = engine.answer_ext(GrossProfitQuery(period=june), AS_OF)
    # 150000 - (2*20000 + 1*10000) = 100000
    assert gross.answer == 100_000
    net = engine.answer_ext(NetProfitQuery(period=june), AS_OF)
    assert net.answer == 70_000
    emp = engine.answer_ext(TopEmployeeQuery(period=june), AS_OF)
    assert emp.answer == "Lan"
    assert engine.answer_ext(RxRevenueQuery(period=june), AS_OF).answer == 100_000
    assert engine.answer_ext(OtcRevenueQuery(period=june), AS_OF).answer == 50_000
    # Matched by code or by name, case and accents ignored.
    by_code = engine.answer_ext(EmployeeRevenueQuery(employee="e2", period=june), AS_OF)
    assert by_code.answer == 50_000
    by_name = engine.answer_ext(EmployeeRevenueQuery(employee="LAN", period=june), AS_OF)
    assert by_name.answer == 100_000
    unknown = engine.answer_ext(EmployeeRevenueQuery(employee="Hoa", period=june), AS_OF)
    assert unknown.type == "abstain"


def test_describe_e4d_smoke() -> None:
    rules = RuleConfig()
    cal = LunarCalendar({})
    june = MonthExpr(month=6, year=2026)
    day = DayExpr(date=datetime.date(2026, 6, 15))
    text = describe_ext_spec(PurchaseTotalQuery(period=june), AS_OF, rules, cal)
    assert "purchases table" in text
    text = describe_ext_spec(
        StockOnHandQuery(product="Sữa", date=day, unit="thùng"), AS_OF, rules, cal
    )
    assert "stock_moves" in text
    text = describe_ext_spec(
        LotsExpiringQuery(date=day, n_days=30, output="cost"), AS_OF, rules, cal
    )
    assert "30 days" in text
    text = describe_ext_spec(RxShareQuery(period=june), AS_OF, rules, cal)
    assert "percent" in text


def test_employee_revenue_needs_the_employee_in_the_question() -> None:
    from datamate.query.convert import ConvertError
    from datamate.query.ext.convert import to_ext_query_spec
    from datamate.query.parse_schema import PERIOD_FIELDS

    period: dict[str, object] = {name: None for name in PERIOD_FIELDS}
    period.update(kind="month", month=6, year=2026)
    asked = to_ext_query_spec(
        {"family": "employee_revenue", "employee": "NV02", "period": period},
        AS_OF,
        "Thang 6/2026 nhan vien ma NV02 ban duoc bao nhieu tien",
    )
    assert isinstance(asked, EmployeeRevenueQuery)
    assert asked.employee == "NV02"
    with pytest.raises(ConvertError, match="not grounded"):
        to_ext_query_spec(
            {"family": "employee_revenue", "employee": "NV07", "period": period},
            AS_OF,
            "Thang 6/2026 nhan vien ma NV02 ban duoc bao nhieu tien",
        )


def test_ingredient_combination_spellings_agree() -> None:
    from datamate.metrics.ext_e4d import _ingredient_set
    from datamate.query.ext.convert import to_ext_query_spec
    from datamate.query.parse_schema import PERIOD_FIELDS

    period: dict[str, object] = {name: None for name in PERIOD_FIELDS}
    period.update(kind="month", month=6, year=2026)
    specs = [
        to_ext_query_spec(
            {"family": "ingredient_revenue", "active_ingredient": raw, "period": period},
            AS_OF,
            "Thang 6/2026 thuoc hoat chat Paracetamol + Cafein ban duoc bao nhieu",
        )
        for raw in ("Paracetamol + Cafein", "paracetamol, cafein", "Paracetamol và Cafein")
    ]
    assert {spec.active_ingredient for spec in specs} == {"paracetamol + cafein"}  # type: ignore[attr-defined]
    # Same combination in another order matches; one part alone does not.
    assert _ingredient_set("Cafein + Paracetamol") == _ingredient_set("paracetamol + cafein")
    assert _ingredient_set("Paracetamol") != _ingredient_set("Paracetamol + Cafein")

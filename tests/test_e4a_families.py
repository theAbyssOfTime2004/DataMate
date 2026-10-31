"""Synthetic DuckDB coverage for E4a product families + prompt hash + routing."""

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
from datamate.etl.schema import PRICE_HISTORY_COLUMNS
from datamate.metrics import MetricEngine
from datamate.query.ext.spec import (
    PriceIncreaseEffectQuery,
    PriceOnDateQuery,
    ProductsInMinOrdersQuery,
    QtySoldQuery,
    TopProductsQuery,
    UnsoldProductsQuery,
)
from datamate.query.prompts import EXAMPLE_CONTEXT, build_messages
from datamate.rules import RuleConfig
from datamate.timeres import DayExpr, LunarCalendar, MonthExpr

AS_OF = datetime.date(2026, 6, 30)
DATA_START = datetime.date(2026, 1, 1)
DATA_END = datetime.date(2026, 6, 30)

# Frozen hash of build_messages for a fixed question+context (main parser must stay byte-identical).
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
    status: str,
    *,
    has_return: bool = False,
) -> dict[str, object]:
    return {
        "order_id": order_id,
        "order_date": day,
        "order_ts": datetime.datetime.combine(day, datetime.time(10, 0)),
        "status": status,
        "has_return": has_return,
        "customer_key": "c1",
        "is_walk_in": False,
        "payment_method": "cash",
        "channel": "store",
    }


def _line(
    order_id: str,
    *,
    product_key: str,
    product_id: str | None,
    name: str,
    qty: Decimal,
    qty_base: Decimal,
    unit_price: int,
    line_amount: int,
    cost_per_unit: int | None = None,
) -> dict[str, object]:
    return {
        "order_id": order_id,
        "line_no": 1,
        "product_key": product_key,
        "product_id": product_id,
        "product_name": name,
        "qty": qty,
        "unit": "hộp",
        "qty_base": qty_base,
        "unit_price": unit_price,
        "line_amount": line_amount,
        "cost_per_unit": cost_per_unit,
        "is_return_line": False,
        "src_row": 1,
    }


def _build_shop(path: Path) -> MetricEngine:
    orders = [
        _order("o1", datetime.date(2026, 5, 10), "completed"),
        _order("o2", datetime.date(2026, 6, 1), "completed"),
        _order("o3", datetime.date(2026, 6, 2), "cancelled"),
        _order("o4", datetime.date(2026, 6, 3), "completed"),
        _order("o5", datetime.date(2026, 6, 5), "completed"),
        _order("o6", datetime.date(2026, 6, 10), "completed"),
        _order("o7", datetime.date(2026, 4, 1), "completed"),
    ]
    lines = [
        # May: milk sold (needed for unsold later)
        _line(
            "o1",
            product_key="sua",
            product_id="P01",
            name="Sữa tươi",
            qty=Decimal(1),
            qty_base=Decimal(10),
            unit_price=5000,
            line_amount=50000,
            cost_per_unit=3000,
        ),
        # June revenue: tea 5 base, biscuit 5 base; service fee excluded from rankings
        _line(
            "o2",
            product_key="tra",
            product_id="P02",
            name="Trà xanh",
            qty=Decimal(1),
            qty_base=Decimal(5),
            unit_price=8000,
            line_amount=40000,
            cost_per_unit=2000,
        ),
        # Cancelled order: still counts for price_on_date
        _line(
            "o3",
            product_key="tra",
            product_id="P02",
            name="Trà xanh",
            qty=Decimal(1),
            qty_base=Decimal(1),
            unit_price=9000,
            line_amount=9000,
            cost_per_unit=2000,
        ),
        _line(
            "o4",
            product_key="banh",
            product_id="P03",
            name="Bánh quy",
            qty=Decimal(1),
            qty_base=Decimal(5),
            unit_price=6000,
            line_amount=30000,
            cost_per_unit=1000,
        ),
        _line(
            "o4",
            product_key="phi",
            product_id=None,
            name="Phí phục vụ",
            qty=Decimal(1),
            qty_base=Decimal(1),
            unit_price=5000,
            line_amount=5000,
            cost_per_unit=0,
        ),
        # Tie on qty with tea on another order — code asc: P02 before P03 if equal... 
        # actually tea already 5, biscuit 5. Add more tea to break for top by qty.
        _line(
            "o5",
            product_key="tra",
            product_id="P02",
            name="Trà xanh",
            qty=Decimal(1),
            qty_base=Decimal(5),
            unit_price=8000,
            line_amount=40000,
            cost_per_unit=2000,
        ),
        # Price increase: biscuit 6000 on 6/3, then 8000 on 6/10
        _line(
            "o6",
            product_key="banh",
            product_id="P03",
            name="Bánh quy",
            qty=Decimal(2),
            qty_base=Decimal(10),
            unit_price=8000,
            line_amount=80000,
            cost_per_unit=1000,
        ),
        # Coffee only in April — unsold in June
        _line(
            "o7",
            product_key="cf",
            product_id="P00",
            name="Cà phê",
            qty=Decimal(1),
            qty_base=Decimal(3),
            unit_price=7000,
            line_amount=21000,
            cost_per_unit=2000,
        ),
    ]
    # Fix duplicate line_no on o4 — second line should be line 2
    lines[4]["line_no"] = 2

    products = [
        {
            "product_key": "sua",
            "product_id": "P01",
            "product_name": "Sữa tươi",
            "base_unit": "hộp",
        },
        {
            "product_key": "tra",
            "product_id": "P02",
            "product_name": "Trà xanh",
            "base_unit": "gói",
        },
        {
            "product_key": "banh",
            "product_id": "P03",
            "product_name": "Bánh quy",
            "base_unit": "gói",
        },
        {
            "product_key": "cf",
            "product_id": "P00",
            "product_name": "Cà phê",
            "base_unit": "gói",
        },
        {"product_key": "phi", "product_name": "Phí phục vụ", "base_unit": None},
    ]
    # Empty price_history so increase uses per-day order prices
    price_hist: list[dict[str, object]] = []

    data = CanonicalData(
        orders=_frame(ORDERS_COLUMNS, orders),
        order_lines=_frame(ORDER_LINES_COLUMNS, lines),
        products=_frame(PRODUCTS_COLUMNS, products),
        unit_conversions=_frame(UNIT_CONVERSIONS_COLUMNS, []),
        customers=_frame(CUSTOMERS_COLUMNS, [{"customer_key": "c1"}]),
        report=EtlReport(),
        price_history=_frame(PRICE_HISTORY_COLUMNS, price_hist),
    )
    write_duckdb(data, path)
    con = duckdb.connect(str(path), read_only=True)
    return MetricEngine(con, RuleConfig(), LunarCalendar({}), DATA_START, DATA_END)


@pytest.fixture(scope="module")
def engine(tmp_path_factory: pytest.TempPathFactory) -> MetricEngine:
    path = tmp_path_factory.mktemp("e4a") / "shop.duckdb"
    built = _build_shop(path)
    yield built
    built.con.close()


def test_qty_sold_product_and_base_unit(engine: MetricEngine) -> None:
    june = MonthExpr(month=6, year=2026)
    ans = engine.answer_ext(
        QtySoldQuery(product="Trà xanh", period=june), AS_OF
    )
    assert ans.type == "number"
    assert ans.answer == 10  # 5+5; cancelled excluded

    by_unit = engine.answer_ext(QtySoldQuery(base_unit="gói", period=june), AS_OF)
    assert by_unit.type == "number"
    # trà 10 + bánh 15 = 25; phí excluded; sữa is hộp
    assert by_unit.answer == 25


def test_top_products_tie_break_and_gross_profit(engine: MetricEngine) -> None:
    june = MonthExpr(month=6, year=2026)
    # June qty_base: bánh 15, trà 10 — bánh wins; phí phục vụ excluded.
    top1 = engine.answer_ext(
        TopProductsQuery(k=1, by="qty_base", mode="max", period=june), AS_OF
    )
    assert top1.type == "string"
    assert top1.answer == "Bánh quy"

    top2 = engine.answer_ext(
        TopProductsQuery(k=2, by="qty_base", mode="max", period=june), AS_OF
    )
    assert top2.type == "list"
    assert top2.answer[0] == "Bánh quy"
    assert top2.answer[1] == "Trà xanh"
    assert "Phí phục vụ" not in top2.answer

    # Equal qty tie: add synthetic check via mode=min -> Cà phê not in June, so trà is min of the two.
    bottom = engine.answer_ext(
        TopProductsQuery(k=1, by="qty_base", mode="min", period=june), AS_OF
    )
    assert bottom.answer == "Trà xanh"


def test_top_products_missing_cost_unanswerable(tmp_path: Path) -> None:
    orders = [_order("o1", datetime.date(2026, 6, 1), "completed")]
    lines = [
        _line(
            "o1",
            product_key="tra",
            product_id="P02",
            name="Trà xanh",
            qty=Decimal(1),
            qty_base=Decimal(5),
            unit_price=8000,
            line_amount=40000,
            cost_per_unit=None,
        )
    ]
    products = [
        {
            "product_key": "tra",
            "product_id": "P02",
            "product_name": "Trà xanh",
            "base_unit": "gói",
        }
    ]
    data = CanonicalData(
        orders=_frame(ORDERS_COLUMNS, orders),
        order_lines=_frame(ORDER_LINES_COLUMNS, lines),
        products=_frame(PRODUCTS_COLUMNS, products),
        unit_conversions=_frame(UNIT_CONVERSIONS_COLUMNS, []),
        customers=_frame(CUSTOMERS_COLUMNS, []),
        report=EtlReport(),
    )
    path = tmp_path / "nocost.duckdb"
    write_duckdb(data, path)
    con = duckdb.connect(str(path), read_only=True)
    eng = MetricEngine(con, RuleConfig(), LunarCalendar({}), DATA_START, DATA_END)
    ans = eng.answer_ext(
        TopProductsQuery(k=1, by="gross_profit", mode="max", period=MonthExpr(month=6, year=2026)),
        AS_OF,
    )
    assert ans.type == "unanswerable"
    con.close()


def test_unsold_and_min_orders(engine: MetricEngine) -> None:
    june = MonthExpr(month=6, year=2026)
    unsold = engine.answer_ext(
        UnsoldProductsQuery(output="list", period=june), AS_OF
    )
    assert unsold.type == "list"
    assert "Cà phê" in unsold.answer
    assert "Sữa tươi" in unsold.answer  # sold in May only
    assert "Trà xanh" not in unsold.answer

    count = engine.answer_ext(
        ProductsInMinOrdersQuery(n=2, output="count", period=june), AS_OF
    )
    assert count.type == "number"
    assert count.answer >= 1  # trà in o2 and o5


def test_price_on_date_includes_cancelled(engine: MetricEngine) -> None:
    # 6/2 cancelled order has unit_price 9000 for tea; 6/1 has 8000 — different days.
    day = DayExpr(date=datetime.date(2026, 6, 2))
    ans = engine.answer_ext(PriceOnDateQuery(product="Trà xanh", date=day), AS_OF)
    assert ans.type == "number"
    assert ans.answer == 9000


def test_price_increase_effect(engine: MetricEngine) -> None:
    # Increase day for bánh: 2026-06-10 (8000 > 6000 on 6/3).
    # n=7: after [6/10..6/16]=10; before [6/3..6/9]=5 -> +100%.
    ans = engine.answer_ext(
        PriceIncreaseEffectQuery(product="Bánh quy", n_days=7), AS_OF
    )
    assert ans.type == "percent"
    assert ans.answer == 100.0


def test_main_prompt_build_messages_hash_is_stable() -> None:
    messages = build_messages(
        _PROMPT_HASH_QUESTION, _PROMPT_HASH_AS_OF, EXAMPLE_CONTEXT, few_shot_set=0
    )
    blob = "\n".join(f"{m['role']}\n{m['content']}" for m in messages).encode("utf-8")
    digest = hashlib.sha256(blob).hexdigest()
    assert digest == _MAIN_PROMPT_SHA256, (
        "main build_messages output changed; E4a forbids editing the current parser prompts"
    )


def test_in_scope_true_never_calls_extension_parser(tmp_path: Path, monkeypatch) -> None:
    """Pipeline routing: a successful main parse must not invoke ExtParsedQuestion."""
    from datamate.config import settings
    from datamate.io.paths import business_dir
    from datamate.io.questions import Question
    from datamate.systems.pipeline import PipelineSystem
    from tests.test_pipeline import (
        FakeClient,
        _mapping_scripts,
        _parsed,
        _write_calendar,
        _write_shop,
    )

    monkeypatch.setattr(settings, "data_root", tmp_path)
    monkeypatch.setattr(settings, "cache_dir", tmp_path / "cache")
    split_dir = tmp_path / "train"
    _write_calendar(split_dir)
    _write_shop(split_dir / "data" / "biz_syn")

    parsed = _parsed(periods=[{"kind": "month_this"}])
    client = FakeClient({**_mapping_scripts(), "ParsedQuestion": [parsed, parsed]})
    system = PipelineSystem(name="pipeline_test", client=client)
    try:
        question = Question(
            id="q_route",
            business_id="biz_syn",
            question="doanh thu thang nay bao nhieu",
            as_of_date=datetime.date(2026, 8, 15),
        )
        answer, trace = system.answer(question, business_dir("train", "biz_syn"))
        assert answer.type == "number"
        assert trace.get("ext_family") is None
        assert not any(c["schema_name"] == "ExtParsedQuestion" for c in client.calls)
    finally:
        system.close()


def test_derive_qty_base_big_base_unknown() -> None:
    from collections import Counter

    from datamate.etl.build import _derive_qty_base

    products = pd.DataFrame(
        [
            {
                "product_key": "thep",
                "product_id": "T1",
                "product_name": "Thép hộp 40x80",
                "base_unit": "cây",
            }
        ]
    )
    unit_conv = pd.DataFrame(
        [{"product_key": "thep", "unit": "bó", "factor": Decimal(10), "to_unit": "cây"}]
    )
    lines = [
        {
            "product_key": "thep",
            "qty": Decimal(2),
            "unit": "bó",
            "qty_base": None,
        },
        {
            "product_key": "thep",
            "qty": Decimal(3),
            "unit": "cây",
            "qty_base": None,
        },
        {
            "product_key": "thep",
            "qty": Decimal(1),
            "unit": "tấn",
            "qty_base": None,
        },
    ]
    fills: Counter[str] = Counter()
    _derive_qty_base(lines, products, unit_conv, fills)
    assert lines[0]["qty_base"] == Decimal(20)  # 2 bó * 10
    assert lines[1]["qty_base"] == Decimal(3)  # base unit kept
    assert lines[2]["qty_base"] is None  # unknown unit
    assert fills["order_lines.qty_base"] == 2


def test_fold_product_name_unique_and_ambiguous() -> None:
    from datamate.etl.build import ProductNameIndex, _line_product_key, _product_key_by_name

    products = pd.DataFrame(
        [
            {
                "product_key": "T1",
                "product_id": "T1",
                "product_name": "Thép hộp 40x80",
                "base_unit": "cây",
            },
            # Same fold as each other (é vs ê) — must not fold-match either.
            {
                "product_key": "A1",
                "product_id": "A1",
                "product_name": "Thép tấm",
                "base_unit": "cái",
            },
            {
                "product_key": "A2",
                "product_id": "A2",
                "product_name": "Thêp tấm",
                "base_unit": "cái",
            },
        ]
    )
    index = _product_key_by_name(products)
    assert isinstance(index, ProductNameIndex)
    assert _line_product_key(None, "Thep hop 40x80", index) == "T1"
    assert _line_product_key(None, "Thep tam", index) == "thep tam"


def test_merge_non_catalog_folded_keys_no_catalog() -> None:
    from datamate.etl.build import _merge_non_catalog_folded_keys
    from datamate.parse.text import normalize_label

    accented = normalize_label("Thép cuộn phi 8")
    plain = normalize_label("Thep cuon phi 8")
    assert accented is not None and plain is not None
    lines = [
        {"product_key": accented},
        {"product_key": accented},
        {"product_key": accented},
        {"product_key": plain},
    ]
    empty_catalog = pd.DataFrame(columns=list(PRODUCTS_COLUMNS))
    _merge_non_catalog_folded_keys(lines, empty_catalog)
    assert {pl["product_key"] for pl in lines} == {accented}


def test_merge_keeps_ambiguous_catalog_keys_separate() -> None:
    from datamate.etl.build import _merge_non_catalog_folded_keys

    products = pd.DataFrame(
        [
            {
                "product_key": "A1",
                "product_id": "A1",
                "product_name": "Thép tấm",
                "base_unit": "cái",
            },
            {
                "product_key": "A2",
                "product_id": "A2",
                "product_name": "Thêp tấm",
                "base_unit": "cái",
            },
        ]
    )
    lines = [
        {"product_key": "A1"},
        {"product_key": "A2"},
        {"product_key": "A1"},
    ]
    _merge_non_catalog_folded_keys(lines, products)
    assert [pl["product_key"] for pl in lines] == ["A1", "A2", "A1"]


def test_null_qty_base_and_missing_product_abstain(tmp_path: Path) -> None:
    orders = [_order("o1", datetime.date(2026, 6, 1), "completed")]
    # Matched product has null qty_base; another product keeps the column "recorded".
    lines_mixed = [
        {
            **_line(
                "o1",
                product_key="tra",
                product_id="P02",
                name="Trà xanh",
                qty=Decimal(1),
                qty_base=Decimal(1),
                unit_price=8000,
                line_amount=8000,
                cost_per_unit=2000,
            ),
            "qty_base": None,
        },
        {
            **_line(
                "o1",
                product_key="banh",
                product_id="P03",
                name="Bánh quy",
                qty=Decimal(1),
                qty_base=Decimal(2),
                unit_price=1000,
                line_amount=2000,
                cost_per_unit=100,
            ),
            "line_no": 2,
        },
    ]
    products = [
        {
            "product_key": "tra",
            "product_id": "P02",
            "product_name": "Trà xanh",
            "base_unit": "gói",
        },
        {
            "product_key": "banh",
            "product_id": "P03",
            "product_name": "Bánh quy",
            "base_unit": "gói",
        },
    ]
    data = CanonicalData(
        orders=_frame(ORDERS_COLUMNS, orders),
        order_lines=_frame(ORDER_LINES_COLUMNS, lines_mixed),
        products=_frame(PRODUCTS_COLUMNS, products),
        unit_conversions=_frame(UNIT_CONVERSIONS_COLUMNS, []),
        customers=_frame(CUSTOMERS_COLUMNS, []),
        report=EtlReport(),
    )
    path = tmp_path / "nullqb.duckdb"
    write_duckdb(data, path)
    con = duckdb.connect(str(path), read_only=True)
    eng = MetricEngine(con, RuleConfig(), LunarCalendar({}), DATA_START, DATA_END)
    june = MonthExpr(month=6, year=2026)
    ans = eng.answer_ext(QtySoldQuery(product="Trà xanh", period=june), AS_OF)
    assert ans.type == "abstain"
    missing = eng.answer_ext(QtySoldQuery(product="Hàng ma", period=june), AS_OF)
    assert missing.type == "abstain"
    empty_top = eng.answer_ext(
        TopProductsQuery(k=1, by="qty_base", mode="max", period=MonthExpr(month=1, year=2026)),
        AS_OF,
    )
    assert empty_top.type == "abstain"
    con.close()


def test_count_vs_list_trusts_parser_output() -> None:
    from datamate.query.ext.convert import to_ext_query_spec

    as_of = datetime.date(2026, 6, 30)
    full_period = _ext_period(kind="month", month=6, year=2026)

    count_q = to_ext_query_spec(
        {"family": "unsold_products", "output": "count", "period": full_period},
        as_of,
        "Thang 6 dem so mon khong ai mua",
    )
    assert count_q.output == "count"  # type: ignore[attr-defined]

    list_q = to_ext_query_spec(
        {"family": "unsold_products", "output": "list", "period": full_period},
        as_of,
        "Ke ten cac mon khong ai mua thang 6",
    )
    assert list_q.output == "list"  # type: ignore[attr-defined]


def _ext_period(**fields: object) -> dict[str, object]:
    from datamate.query.parse_schema import PERIOD_FIELDS

    row: dict[str, object] = {name: None for name in PERIOD_FIELDS}
    row.update(fields)
    return row


def test_convert_drops_category_with_product_and_day_from_period() -> None:
    from datamate.query.ext.convert import to_ext_query_spec
    from datamate.query.ext.spec import NoneQuery, PriceOnDateQuery, QtySoldQuery

    as_of = datetime.date(2026, 6, 30)
    period = _ext_period(kind="month_this")
    both = to_ext_query_spec(
        {
            "family": "qty_sold",
            "product": "Trà xanh",
            "category": "đồ uống",
            "period": period,
        },
        as_of,
        "Thang nay ban bao nhieu Tra xanh trong nhom do uong",
    )
    assert isinstance(both, QtySoldQuery)
    assert both.product == "Trà xanh"
    assert both.category is None

    ungrounded = to_ext_query_spec(
        {
            "family": "qty_sold",
            "base_unit": "gói",
            "category": "đồ uống",
            "period": period,
        },
        as_of,
        "Thang nay ban bao nhieu goi",
    )
    assert isinstance(ungrounded, NoneQuery)

    day = to_ext_query_spec(
        {
            "family": "price_on_date",
            "product": "Trà xanh",
            "period": _ext_period(kind="today"),
            "date": None,
        },
        as_of,
        "Hom nay Tra xanh gia bao nhieu",
    )
    assert isinstance(day, PriceOnDateQuery)


def test_describe_ext_spec_each_family() -> None:
    from datamate.query.ext.describe import describe_ext_spec
    from datamate.query.ext.spec import (
        PriceIncreaseEffectQuery,
        PriceOnDateQuery,
        ProductsInMinOrdersQuery,
        QtySoldQuery,
        TopProductsQuery,
        UnsoldProductsQuery,
    )
    from datamate.timeres import DayExpr, MonthExpr

    rules = RuleConfig()
    cal = LunarCalendar({})
    as_of = datetime.date(2026, 6, 30)
    june = MonthExpr(month=6, year=2026)

    qty = describe_ext_spec(
        QtySoldQuery(product="Gạo ST25", period=june), as_of, rules, cal
    )
    assert 'products: "Gạo ST25"' in qty
    assert "units sold" in qty
    assert "2026-06-01 .. 2026-06-30" in qty
    assert "not filtered by: delivery status" in qty

    top = describe_ext_spec(
        TopProductsQuery(k=3, by="revenue", mode="max", category="Gạo", period=june),
        as_of,
        rules,
        cal,
    )
    assert 'category "Gạo"' in top
    assert "money revenue" in top
    assert "top 3 names" in top

    unsold = describe_ext_spec(
        UnsoldProductsQuery(output="count", period=june), as_of, rules, cal
    )
    assert "count of names" in unsold

    min_o = describe_ext_spec(
        ProductsInMinOrdersQuery(n=2, output="list", period=june), as_of, rules, cal
    )
    assert "at least 2" in min_o
    assert "list of names" in min_o

    price = describe_ext_spec(
        PriceOnDateQuery(product="Gạo ST25", date=DayExpr(date=datetime.date(2026, 6, 1))),
        as_of,
        rules,
        cal,
    )
    assert "selling price of one" in price

    effect = describe_ext_spec(
        PriceIncreaseEffectQuery(product="Gạo ST25", n_days=5), as_of, rules, cal
    )
    assert "percent change" in effect
    assert "5 days" in effect


def test_category_filter_on_qty_and_top(tmp_path: Path) -> None:
    orders = [
        _order("o1", datetime.date(2026, 6, 1), "completed"),
        _order("o2", datetime.date(2026, 6, 2), "completed"),
    ]
    lines = [
        _line(
            "o1",
            product_key="tra",
            product_id="P02",
            name="Trà xanh",
            qty=Decimal(1),
            qty_base=Decimal(10),
            unit_price=8000,
            line_amount=80000,
            cost_per_unit=2000,
        ),
        _line(
            "o2",
            product_key="banh",
            product_id="P03",
            name="Bánh quy",
            qty=Decimal(1),
            qty_base=Decimal(4),
            unit_price=6000,
            line_amount=24000,
            cost_per_unit=1000,
        ),
    ]
    products = [
        {
            "product_key": "tra",
            "product_id": "P02",
            "product_name": "Trà xanh",
            "base_unit": "gói",
            "category": "đồ uống",
        },
        {
            "product_key": "banh",
            "product_id": "P03",
            "product_name": "Bánh quy",
            "base_unit": "gói",
            "category": "bánh",
        },
    ]
    data = CanonicalData(
        orders=_frame(ORDERS_COLUMNS, orders),
        order_lines=_frame(ORDER_LINES_COLUMNS, lines),
        products=_frame(PRODUCTS_COLUMNS, products),
        unit_conversions=_frame(UNIT_CONVERSIONS_COLUMNS, []),
        customers=_frame(CUSTOMERS_COLUMNS, []),
        report=EtlReport(),
    )
    path = tmp_path / "cat.duckdb"
    write_duckdb(data, path)
    con = duckdb.connect(str(path), read_only=True)
    eng = MetricEngine(con, RuleConfig(), LunarCalendar({}), DATA_START, DATA_END)
    june = MonthExpr(month=6, year=2026)
    drink_qty = eng.answer_ext(
        QtySoldQuery(base_unit="gói", category="đồ uống", period=june), AS_OF
    )
    assert drink_qty.type == "number"
    assert drink_qty.answer == 10
    bakery_qty = eng.answer_ext(
        QtySoldQuery(base_unit="gói", category="bánh", period=june), AS_OF
    )
    assert bakery_qty.answer == 4
    # Without category both share the same unit total — the bug fix3 guards against.
    all_goi = eng.answer_ext(QtySoldQuery(base_unit="gói", period=june), AS_OF)
    assert all_goi.answer == 14
    top_drink = eng.answer_ext(
        TopProductsQuery(
            k=1, by="revenue", mode="max", category="đồ uống", period=june
        ),
        AS_OF,
    )
    assert top_drink.answer == "Trà xanh"
    con.close()




def test_cafe_size_filter(tmp_path: Path) -> None:
    from datamate.query.context import ShopContext
    from datamate.query.ext.schema import ext_parsed_schema

    orders = [
        _order("o1", datetime.date(2026, 6, 1), "completed"),
        _order("o2", datetime.date(2026, 6, 2), "completed"),
    ]
    lines = [
        _line(
            "o1",
            product_key="cfm",
            product_id="CM",
            name="Cà phê sữa (M)",
            qty=Decimal(1),
            qty_base=Decimal(3),
            unit_price=25000,
            line_amount=75000,
            cost_per_unit=8000,
        ),
        _line(
            "o2",
            product_key="cfl",
            product_id="CL",
            name="Cà phê sữa (L)",
            qty=Decimal(1),
            qty_base=Decimal(5),
            unit_price=30000,
            line_amount=150000,
            cost_per_unit=10000,
        ),
    ]
    products = [
        {
            "product_key": "cfm",
            "product_id": "CM",
            "product_name": "Cà phê sữa (M)",
            "base_unit": "ly",
        },
        {
            "product_key": "cfl",
            "product_id": "CL",
            "product_name": "Cà phê sữa (L)",
            "base_unit": "ly",
        },
    ]
    data = CanonicalData(
        orders=_frame(ORDERS_COLUMNS, orders),
        order_lines=_frame(ORDER_LINES_COLUMNS, lines),
        products=_frame(PRODUCTS_COLUMNS, products),
        unit_conversions=_frame(UNIT_CONVERSIONS_COLUMNS, []),
        customers=_frame(CUSTOMERS_COLUMNS, []),
        report=EtlReport(),
    )
    path = tmp_path / "cafe.duckdb"
    write_duckdb(data, path)
    con = duckdb.connect(str(path), read_only=True)
    eng = MetricEngine(con, RuleConfig(), LunarCalendar({}), DATA_START, DATA_END)
    june = MonthExpr(month=6, year=2026)
    m_only = eng.answer_ext(
        QtySoldQuery(product="Cà phê sữa (M)", size="M", period=june), AS_OF
    )
    assert m_only.type == "number"
    assert m_only.answer == 3
    top_l = eng.answer_ext(
        TopProductsQuery(k=1, by="qty_base", mode="max", size="L", period=june), AS_OF
    )
    assert top_l.answer == "Cà phê sữa (L)"

    ctx = ShopContext(
        payment_methods=[],
        channels=[],
        segments=[],
        categories=[],
        product_names=["Cà phê sữa (M)", "Cà phê sữa (L)"],
        has_order_ts=False,
        data_start=DATA_START,
        data_end=DATA_END,
        base_units=["ly"],
    )
    schema = ext_parsed_schema(ctx, list(ctx.product_names))
    assert schema["properties"]["size"]["enum"] == ["M", "L", None]

    ctx_plain = ShopContext(
        payment_methods=[],
        channels=[],
        segments=[],
        categories=[],
        product_names=["Trà xanh"],
        has_order_ts=False,
        data_start=DATA_START,
        data_end=DATA_END,
        base_units=["gói"],
    )
    schema_plain = ext_parsed_schema(ctx_plain, ["Trà xanh"])
    assert schema_plain["properties"]["size"] == {"enum": [None]}
    con.close()

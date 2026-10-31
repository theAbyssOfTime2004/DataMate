"""Extension-parser prompts and few-shots (generic product names; no train copy)."""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass
from typing import Any

from datamate.query.context import ShopContext
from datamate.query.parse_schema import PERIOD_FIELDS

_WEEKDAYS = ("Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật")

_EXAMPLE_PRODUCTS = [
    "Sữa tươi hộp",
    "Bánh quy bơ",
    "Nước cam ép",
    "Cà phê sữa (M)",
    "Cà phê sữa (L)",
]
_EXAMPLE_UNITS = ["hộp", "gói", "chai", "ly"]

EXAMPLE_CONTEXT = ShopContext(
    payment_methods=["Tiền mặt"],
    channels=["tại quầy"],
    segments=[],
    categories=["đồ uống", "bánh"],
    product_names=list(_EXAMPLE_PRODUCTS),
    has_order_ts=False,
    data_start=datetime.date(2024, 1, 1),
    data_end=datetime.date(2024, 12, 31),
    base_units=list(_EXAMPLE_UNITS),
)

_SYSTEM = "\n".join(  # noqa: FLY002
    [
        "You classify one Vietnamese shop question into a metric-family JSON.",
        "You do not calculate the answer and you do not calculate calendar dates.",
        "",
        "Families:",
        "- qty_sold: how many base units of one product sold, OR how many base units sold",
        "  across all products whose base unit is a named unit (hộp / gói / ...).",
        "  Set product OR base_unit (exactly one), plus period. Optional category and",
        "  size M|L when the shop lists them and the question names that filter.",
        "- top_products: best or worst selling items. k, by qty_base | revenue |",
        "  gross_profit, mode max|min, period. Optional category / size.",
        "  'bán chạy' -> by=qty_base; money/doanh thu -> by=revenue; lãi/lời ->",
        "  by=gross_profit.",
        "- unsold_products: items sold before the period with no revenue sale inside it.",
        "  output count|list, period. Optional category.",
        "- products_in_min_orders: *products* in at least n distinct revenue orders.",
        "  output count|list. Optional category.",
        "- price_on_date: unit selling price of one product on one day. product + date",
        "  (or a single-day period).",
        "- price_increase_effect: percent change in qty_base after a price rise.",
        "- customers_min_orders: customers with at least n revenue orders in the period.",
        "  output count|share (share = percent of customers who bought in the period).",
        "- top_customers: best or worst customers. k, by revenue | order_count, mode",
        "  max|min, period. Money/nhiều tiền -> by=revenue; nhiều đơn -> by=order_count.",
        "- orders_min_amount: count of revenue orders whose order revenue passes a",
        "  money threshold. x (VND integer), op '>=' | '>' (from the question wording),",
        "  period.",
        "- days_min_revenue: count of days in the period whose day revenue passes x with",
        "  op '>=' | '>'. Days with no orders never qualify.",
        "- none: shop-wide revenue totals, debt, stock, purchases, delivery quantities,",
        "  average purchase cost, or any question this list cannot express.",
        "",
        "Count vs list vs share: if the question asks how many and does not ask to list",
        "or name the items, set output=count. Use list only when it asks to list, name,",
        "or say which. Use share only for a percent of customers.",
        "",
        "Threshold op: set op from the question (inclusive vs exclusive). Do not invent",
        "x; copy the written VND amount as an integer.",
        "",
        "Period kinds match the main parser: day, today, yesterday, last_n_days, week_this,",
        "week_last, month, month_this, month_last, quarter, year, ytd, range, tet,",
        "before_tet. Copy written numbers; never invent calendar dates for relative words.",
        "",
        "Copy product, base_unit, category, and size only from the lists in the user",
        "message. Leave unused fields null.",
        "The worked examples use an invented grocery/cafe, Cửa hàng An Bình.",
    ]
)


@dataclass(frozen=True)
class ExtFewShot:
    question: str
    as_of: datetime.date
    parsed: dict[str, Any]


def _period(kind: str, **fields: Any) -> dict[str, Any]:
    row: dict[str, Any] = {name: None for name in PERIOD_FIELDS}
    row["kind"] = kind
    row.update(fields)
    return row


def _parsed(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "family": "none",
        "product": None,
        "base_unit": None,
        "category": None,
        "size": None,
        "period": None,
        "k": None,
        "by": None,
        "mode": None,
        "output": None,
        "n": None,
        "n_days": None,
        "date": None,
        "x": None,
        "op": None,
    }
    row.update(overrides)
    return row


def _ex(question: str, as_of: datetime.date, **parsed: Any) -> ExtFewShot:
    return ExtFewShot(question, as_of, _parsed(**parsed))


_ASOF = datetime.date(2024, 8, 15)

EXT_FEW_SHOT_SETS: tuple[tuple[ExtFewShot, ...], ...] = (
    (
        _ex(
            "Thang 7/2024 cua hang An Binh ban duoc bao nhieu hop Sua tuoi hop",
            _ASOF,
            family="qty_sold",
            product="Sữa tươi hộp",
            period=_period("month", month=7, year=2024),
        ),
        _ex(
            "Thang nay An Binh ban duoc bao nhieu goi (don vi co ban goi)",
            _ASOF,
            family="qty_sold",
            base_unit="gói",
            period=_period("month_this"),
        ),
        _ex(
            "Trong nhom do uong, mon nao mang ve nhieu tien nhat thang nay",
            _ASOF,
            family="top_products",
            k=1,
            by="revenue",
            mode="max",
            category="đồ uống",
            period=_period("month_this"),
        ),
        _ex(
            "Nhom banh ban duoc bao nhieu goi thang Bay 2024",
            _ASOF,
            family="qty_sold",
            base_unit="gói",
            category="bánh",
            period=_period("month", month=7, year=2024),
        ),
        _ex(
            "Cho biet 3 mon hot nhat cua tiem An Binh trong thang sau Tet 2024",
            _ASOF,
            family="top_products",
            k=3,
            by="qty_base",
            mode="max",
            period=_period("month", month=2, year=2024),
        ),
        _ex(
            "Mon nao e nhat quy he 2024",
            _ASOF,
            family="top_products",
            k=1,
            by="qty_base",
            mode="min",
            period=_period("quarter", q=2, year=2024),
        ),
        _ex(
            "Dem so mon tung ban truoc do ma thang Nam khong ai mua",
            _ASOF,
            family="unsold_products",
            output="count",
            period=_period("month", month=5, year=2024),
        ),
        _ex(
            "Ke ten mon xuat hien tu 5 bill tro len thang nay",
            _ASOF,
            family="products_in_min_orders",
            n=5,
            output="list",
            period=_period("month_this"),
        ),
        _ex(
            "Mot hop Sua tuoi hop ngay mung 1 thang 8 nam 2024 gia bao nhieu",
            _ASOF,
            family="price_on_date",
            product="Sữa tươi hộp",
            date=_period("day", date="2024-08-01"),
        ),
        _ex(
            "Khi Banh quy bo doi gia, 7 ngay sau so voi 7 ngay truoc luong ban doi % ra sao",
            _ASOF,
            family="price_increase_effect",
            product="Bánh quy bơ",
            n_days=7,
        ),
        _ex(
            "Tong tien An Binh thu duoc thang Bay 2024",
            _ASOF,
            family="none",
        ),
        _ex(
            "Tiem An Binh: dem khach dat tu 4 don tro len trong thang nay",
            _ASOF,
            family="customers_min_orders",
            n=4,
            output="count",
            period=_period("month_this"),
        ),
        _ex(
            "Thang nay ban bao nhieu ly Ca phe sua size M",
            _ASOF,
            family="qty_sold",
            product="Cà phê sữa (M)",
            size="M",
            period=_period("month_this"),
        ),
        _ex(
            "Ton kho Sua tuoi hop cuoi ngay hom nay con bao nhieu hop",
            _ASOF,
            family="none",
        ),
        _ex(
            "Quy 3/2024 cua hang An Binh dem bill co doanh so >= 275000 ra bao nhieu",
            _ASOF,
            family="orders_min_amount",
            x=275000,
            op=">=",
            period=_period("quarter", q=3, year=2024),
        ),
    ),
    (
        _ex(
            "Hom qua tiem ban het bao nhieu chai Nuoc cam ep",
            _ASOF,
            family="qty_sold",
            product="Nước cam ép",
            period=_period("yesterday"),
        ),
        _ex(
            "Mon mang ve nhieu tien nhat tuan nay la mon nao",
            _ASOF,
            family="top_products",
            k=1,
            by="revenue",
            mode="max",
            period=_period("week_this"),
        ),
        _ex(
            "Mon lai gop cao nhat thang vua roi",
            _ASOF,
            family="top_products",
            k=1,
            by="gross_profit",
            mode="max",
            period=_period("month_last"),
        ),
        _ex(
            "Ke cac mon nam im thang Tu",
            _ASOF,
            family="unsold_products",
            output="list",
            period=_period("month", month=4, year=2024),
        ),
        _ex(
            "Trong 30 ngay gan day, bao nhieu mon xuat hien tu 3 bill tro len",
            _ASOF,
            family="products_in_min_orders",
            n=3,
            output="count",
            period=_period("last_n_days", n=30),
        ),
        _ex(
            "Hom nay Banh quy bo dang ban gia may",
            _ASOF,
            family="price_on_date",
            product="Bánh quy bơ",
            date=_period("today"),
        ),
        _ex(
            "Tang gia Sua tuoi hop ngay 10/07/2024, 5 ngay sau luong ban doi % the nao",
            _ASOF,
            family="price_increase_effect",
            product="Sữa tươi hộp",
            n_days=5,
            date=_period("day", date="2024-07-10"),
        ),
        _ex(
            "O An Binh, ai chi tieu cao nhat tuan nay",
            _ASOF,
            family="top_customers",
            k=1,
            by="revenue",
            mode="max",
            period=_period("week_this"),
        ),
        _ex(
            "Top 3 mon size L ban chay nhat thang nay",
            _ASOF,
            family="top_products",
            k=3,
            by="qty_base",
            mode="max",
            size="L",
            period=_period("month_this"),
        ),
        _ex(
            "Gia nhap binh quan Banh quy bo thang 7/2024 la bao nhieu",
            _ASOF,
            family="none",
        ),
        _ex(
            "Trong thang Tam 2024 An Binh dem ngay thu ve > 1850000 dong ra may ngay",
            _ASOF,
            family="days_min_revenue",
            x=1850000,
            op=">",
            period=_period("month", month=8, year=2024),
        ),
        _ex(
            "Thang nay An Binh: phan tram khach quay lai it nhat 3 lan la bao nhieu",
            _ASOF,
            family="customers_min_orders",
            n=3,
            output="share",
            period=_period("month_this"),
        ),
    ),
)


def build_ext_messages(
    question_text: str,
    as_of: datetime.date,
    ctx: ShopContext,
    few_shot_set: int,
    example_limit: int | None = None,
) -> list[dict[str, str]]:
    examples = EXT_FEW_SHOT_SETS[few_shot_set % len(EXT_FEW_SHOT_SETS)]
    if example_limit is not None:
        examples = examples[: max(example_limit, 0)]
    messages = [{"role": "system", "content": _SYSTEM}]
    for example in examples:
        messages.append(
            {
                "role": "user",
                "content": _user_message(
                    example.question,
                    example.as_of,
                    EXAMPLE_CONTEXT,
                    list(_EXAMPLE_PRODUCTS),
                ),
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(example.parsed, ensure_ascii=False),
            }
        )
    messages.append(
        {
            "role": "user",
            "content": _user_message(
                question_text, as_of, ctx, ctx.product_candidates(question_text)
            ),
        }
    )
    return messages


def _user_message(
    question_text: str,
    as_of: datetime.date,
    ctx: ShopContext,
    products: list[str],
) -> str:
    weekday = _WEEKDAYS[as_of.weekday()]
    lines = [
        f"Hôm nay (as_of): {as_of.isoformat()} ({weekday})",
        "Copy product / base_unit / category / size from these lists only:",
        f"product: {_join(products)}",
        f"base_unit: {_join(ctx.base_units)}",
        f"category: {_join(ctx.categories)}",
        f"size: {_join(_sizes_offered(ctx.product_names))}",
        f"Question: {question_text}",
    ]
    return "\n".join(lines)


def _sizes_offered(product_names: list[str]) -> list[str]:
    found: list[str] = []
    for code in ("M", "L"):
        suffix = f"({code})".casefold()
        if any(name.casefold().rstrip().endswith(suffix) for name in product_names):
            found.append(code)
    return found


def _join(values: list[str]) -> str:
    if not values:
        return "(none)"
    return ", ".join(values)

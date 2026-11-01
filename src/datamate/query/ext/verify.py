"""Yes/no verifier: does the spec description match the question (p_yes from logprobs)."""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any

from datamate.config import settings
from datamate.query.ext.describe import describe_ext_spec
from datamate.query.ext.spec import ExtQuerySpec
from datamate.rules.model import RuleConfig
from datamate.timeres.calendar import LunarCalendar

_CHOICES = ["yes", "no"]

_SYSTEM = "\n".join(  # noqa: FLY002
    [
        "You check whether a query description computes what a shop owner's question asks.",
        "The question is Vietnamese and may have typos, missing accents or chat slang;",
        "the description is English. Judge meaning, not wording.",
        "Answer no only when the question states a condition or asks for an output that the",
        "description lacks or contradicts: a different number, product, product group, brand,",
        "size or unit; money instead of quantity or the reverse; stock, purchases or",
        "deliveries instead of sales; a specific customer, channel or payment method;",
        "a count instead of a list or share, or the reverse; a different threshold or",
        "operator; customers instead of products or the reverse; debt total vs top debtors",
        "vs aging vs collected; a different period or day.",
        "A unit word in the question (kg, ly, hop, dia, ...) only names how the product is",
        "counted; it matches unless the description counts in a different unit.",
        "Otherwise answer yes. Reply with one word: yes or no.",
    ]
)


@dataclass(frozen=True)
class VerifyExample:
    question: str
    description: str
    answer: str  # "yes" | "no"


def _desc(*lines: str) -> str:
    return "\n".join(lines)


_PLAIN = "not filtered by: delivery status, customer, payment method, channel"
_CUST_PLAIN = "not filtered by: product, channel, payment method, delivery status"
_THRESH_PLAIN = (
    "not filtered by: product, customer, channel, payment method, delivery status"
)
_WHO_CODE = "customers identified by customer code, walk-in orders excluded"
_WHO_NAME = "customers identified by name on the order, walk-in orders excluded"

# One yes and one no per family, invented shops and wording (no train/public text).
# Each "no" shows one generic mismatch kind; never add a shot to fix one dev question.
VERIFY_FEW_SHOTS: tuple[VerifyExample, ...] = (
    VerifyExample(
        question="Tuần trước quán Mây bán được mấy kg Cà phê Robusta rang mộc?",
        description=_desc(
            'measure: quantity sold, counted in "kg"',
            'products: "Cà phê Robusta rang mộc"',
            "size: none",
            _PLAIN,
            "period: 2025-06-02 .. 2025-06-08 (order date)",
            "output: one number (units sold)",
        ),
        answer="yes",
    ),
    VerifyExample(
        question="thang 9 tiem Hong thu ve bao nhieu tien tu banh bao trung muoi vay",
        description=_desc(
            'measure: quantity sold, counted in "cái"',
            'products: "Bánh bao trứng muối"',
            "size: none",
            _PLAIN,
            "period: 2025-09-01 .. 2025-09-30 (order date)",
            "output: one number (units sold)",
        ),
        answer="no",
    ),
    VerifyExample(
        question="5 mon nao duoc khach goi nhieu nhat trong thang 2 vay ad",
        description=_desc(
            "measure: units sold",
            "products: all products",
            "size: none",
            _PLAIN,
            "period: 2025-02-01 .. 2025-02-28 (order date)",
            "output: top 5 names ordered by units sold (highest first), tie by code",
        ),
        answer="yes",
    ),
    VerifyExample(
        question="Trong các loại trà sữa, loại nào bán chạy nhất năm ngoái?",
        description=_desc(
            "measure: units sold",
            "products: all products",
            "size: none",
            _PLAIN,
            "period: 2024-01-01 .. 2024-12-31 (order date)",
            "output: one name (highest by units sold, tie by code)",
        ),
        answer="no",
    ),
    VerifyExample(
        question="tu 3/3 toi 9/3 co may mon khong ban dc phan nao",
        description=_desc(
            "measure: products the shop still sells (sold before the period) with no sale inside the period",
            "products: all products",
            "size: none",
            _PLAIN,
            "period: 2025-03-03 .. 2025-03-09 (order date)",
            "output: count of names",
        ),
        answer="yes",
    ),
    VerifyExample(
        question="Liệt kê giúp mình những món không ai gọi trong tuần qua",
        description=_desc(
            "measure: products the shop still sells (sold before the period) with no sale inside the period",
            "products: all products",
            "size: none",
            _PLAIN,
            "period: 2025-08-11 .. 2025-08-17 (order date)",
            "output: count of names",
        ),
        answer="no",
    ),
    VerifyExample(
        question="Tháng 6 có mấy sản phẩm góp mặt trong từ 15 hoá đơn trở lên?",
        description=_desc(
            "measure: products present in at least 15 distinct revenue orders",
            "products: all products",
            "size: none",
            _PLAIN,
            "period: 2025-06-01 .. 2025-06-30 (order date)",
            "output: count of names",
        ),
        answer="yes",
    ),
    VerifyExample(
        question="co bao nhieu sp nam trong it nhat 120 bill hoi thang 8",
        description=_desc(
            "measure: products present in at least 12 distinct revenue orders",
            "products: all products",
            "size: none",
            _PLAIN,
            "period: 2025-08-01 .. 2025-08-31 (order date)",
            "output: count of names",
        ),
        answer="no",
    ),
    VerifyExample(
        question="ngay 2/5 quan ban bun bo hue dac biet gia bao nhieu 1 to",
        description=_desc(
            'measure: selling price of one "tô" (VND, before line discount)',
            'products: "Bún bò Huế đặc biệt"',
            "size: none",
            _PLAIN,
            "day: 2025-05-02",
            "output: one number (VND)",
        ),
        answer="yes",
    ),
    VerifyExample(
        question="Hôm 4/4 giá nhập một bao Xi măng Hà Tiên là bao nhiêu?",
        description=_desc(
            'measure: selling price of one "bao" (VND, before line discount)',
            'products: "Xi măng Hà Tiên"',
            "size: none",
            _PLAIN,
            "day: 2025-04-04",
            "output: one number (VND)",
        ),
        answer="no",
    ),
    VerifyExample(
        question=(
            "Sau lần tăng giá Bánh mì que, 7 ngày sau bán ra thay đổi bao nhiêu % "
            "so với 7 ngày trước?"
        ),
        description=_desc(
            "measure: percent change in units sold over 7 days after a price increase vs the "
            "7 days before",
            'products: "Bánh mì que"',
            "size: none",
            _PLAIN,
            "increase day: first day the selling price rises vs the previous price",
            "output: one percent",
        ),
        answer="yes",
    ),
    VerifyExample(
        question="kho con bao nhieu thung nuoc suoi Lavie 500ml",
        description=_desc(
            'measure: quantity sold, counted in "thùng"',
            'products: "Nước suối Lavie 500ml"',
            "size: none",
            _PLAIN,
            "period: 2025-01-01 .. 2025-10-20 (order date)",
            "output: one number (units sold)",
        ),
        answer="no",
    ),
    VerifyExample(
        question="Thang Nam, trong so khach co mua o tiem Sen, may phan tram quay lai tu 5 lan tro len",
        description=_desc(
            "measure: customers with at least 5 revenue orders in the period",
            f"customers: {_WHO_CODE}",
            _CUST_PLAIN,
            "period: 2025-05-01 .. 2025-05-31 (order date)",
            "output: percent = those customers / all customers who bought in the period x 100",
        ),
        answer="yes",
    ),
    VerifyExample(
        question="Khach ghe tiem Sen tu 5 lan tro len trong thang Nam chiem may phan tram tong khach",
        description=_desc(
            "measure: customers with at least 5 revenue orders in the period",
            f"customers: {_WHO_CODE}",
            _CUST_PLAIN,
            "period: 2025-05-01 .. 2025-05-31 (order date)",
            "output: count of customers",
        ),
        answer="no",
    ),
    VerifyExample(
        question="Tuan nay o tiem Sen ai chi tieu cao nhat",
        description=_desc(
            "measure: money revenue",
            f"customers: {_WHO_NAME}",
            _CUST_PLAIN,
            "period: 2025-07-07 .. 2025-07-13 (order date)",
            "output: one name (highest by money revenue, tie by code or name)",
        ),
        answer="yes",
    ),
    VerifyExample(
        question="Tuan nay o tiem Sen ai dat nhieu hoa don nhat",
        description=_desc(
            "measure: money revenue",
            f"customers: {_WHO_NAME}",
            _CUST_PLAIN,
            "period: 2025-07-07 .. 2025-07-13 (order date)",
            "output: one name (highest by money revenue, tie by code or name)",
        ),
        answer="no",
    ),
    VerifyExample(
        question="Thang Tu tiem Sen co bao nhieu bill tu 312 nghin tro len",
        description=_desc(
            "measure: count of revenue orders whose order revenue >= 312000 VND",
            _THRESH_PLAIN,
            "period: 2025-04-01 .. 2025-04-30 (order date)",
            "output: one number (order count)",
        ),
        answer="yes",
    ),
    VerifyExample(
        question="Thang Ba tiem Sen co bao nhieu bill tu 312 nghin tro len",
        description=_desc(
            "measure: count of revenue orders whose order revenue >= 312000 VND",
            _THRESH_PLAIN,
            "period: 2025-04-01 .. 2025-04-30 (order date)",
            "output: one number (order count)",
        ),
        answer="no",
    ),
    VerifyExample(
        question="Thang Sau co may ngay tiem Sen thu ve tren 2,1 trieu",
        description=_desc(
            "measure: count of days whose revenue > 2100000 VND "
            "(days with no orders never qualify)",
            _THRESH_PLAIN,
            "period: 2025-06-01 .. 2025-06-30 (order date)",
            "output: one number (day count)",
        ),
        answer="yes",
    ),
    VerifyExample(
        question="Thang Sau co may ngay tiem Sen thu ve tren 21 trieu",
        description=_desc(
            "measure: count of days whose revenue > 2100000 VND "
            "(days with no orders never qualify)",
            _THRESH_PLAIN,
            "period: 2025-06-01 .. 2025-06-30 (order date)",
            "output: one number (day count)",
        ),
        answer="no",
    ),
    VerifyExample(
        question="Het ngay 12 thang 3 tiem Sen con no bao nhieu tien",
        description=_desc(
            "measure: sum of remaining debt (payable minus paid) over debt orders",
            "as-of: end of 2025-03-12",
            "payments from: payments table",
            "excluded orders: cancelled; COD/marketplace labels in this shop: COD",
            f"customers: {_WHO_CODE} (walk-in remaining still counts in the total)",
            "output: one number (VND)",
        ),
        answer="yes",
    ),
    VerifyExample(
        question="Het ngay 12 thang 3 tiem Sen ai no nhieu nhat",
        description=_desc(
            "measure: sum of remaining debt (payable minus paid) over debt orders",
            "as-of: end of 2025-03-12",
            "payments from: payments table",
            "excluded orders: cancelled; COD/marketplace labels in this shop: COD",
            f"customers: {_WHO_CODE} (walk-in remaining still counts in the total)",
            "output: one number (VND)",
        ),
        answer="no",
    ),
    VerifyExample(
        question="Cho bang 3 khach no nhieu nhat tinh den het hom nay o tiem Sen",
        description=_desc(
            "measure: remaining debt per customer",
            "as-of: end of 2025-08-20",
            "payments from: payments table",
            "excluded orders: cancelled; COD/marketplace labels in this shop: none found",
            f"customers: {_WHO_NAME} (walk-in orders omitted from the table)",
            "output: table [customer, amount] top 3 by amount",
        ),
        answer="yes",
    ),
    VerifyExample(
        question="Cho bang 3 khach no nhieu nhat tinh den het hom nay o tiem Sen",
        description=_desc(
            "measure: remaining debt per customer",
            "as-of: end of 2025-08-20",
            "payments from: payments table",
            "excluded orders: cancelled; COD/marketplace labels in this shop: none found",
            f"customers: {_WHO_NAME} (walk-in orders omitted from the table)",
            "output: table [customer, amount] top 5 by amount",
        ),
        answer="no",
    ),
    VerifyExample(
        question="No theo tuoi 0 den 7, 8 den 30, tren 30 ngay het ngay 1 thang 9 o tiem Sen",
        description=_desc(
            "measure: remaining debt by age in days (as-of minus order date)",
            "as-of: end of 2025-09-01",
            "payments from: amount_paid on the order",
            "excluded orders: cancelled; COD/marketplace labels in this shop: Shopee",
            "buckets: 0-7, 8-30, 31+",
            "output: list of amounts in bucket order (VND)",
        ),
        answer="yes",
    ),
    VerifyExample(
        question="No theo tuoi 0 den 7, 8 den 30, tren 30 ngay het ngay 1 thang 9 o tiem Sen",
        description=_desc(
            "measure: remaining debt by age in days (as-of minus order date)",
            "as-of: end of 2025-09-01",
            "payments from: amount_paid on the order",
            "excluded orders: cancelled; COD/marketplace labels in this shop: Shopee",
            "buckets: 0-30, 31+",
            "output: list of amounts in bucket order (VND)",
        ),
        answer="no",
    ),
    VerifyExample(
        question="Thang Bay tiem Sen phat sinh them bao nhieu tien ghi no",
        description=_desc(
            "measure: unpaid part on the order day of debt orders placed in the period",
            "payments from: payments table",
            "excluded orders: cancelled; COD/marketplace labels in this shop: COD",
            "period: 2025-07-01 .. 2025-07-31 (order date)",
            "output: one number (VND)",
        ),
        answer="yes",
    ),
    VerifyExample(
        question="Thang Bay tiem Sen nhan bao nhieu tien khach tra cham",
        description=_desc(
            "measure: unpaid part on the order day of debt orders placed in the period",
            "payments from: payments table",
            "excluded orders: cancelled; COD/marketplace labels in this shop: COD",
            "period: 2025-07-01 .. 2025-07-31 (order date)",
            "output: one number (VND)",
        ),
        answer="no",
    ),
    VerifyExample(
        question="Bay ngay gan day tiem Sen nhan bao nhieu tien khach tra cham",
        description=_desc(
            "measure: payments after the order day, dated in the period, on debt orders",
            "payments from: payments table",
            "excluded orders: cancelled; COD/marketplace labels in this shop: none found",
            "period: 2025-10-06 .. 2025-10-12 (order date)",
            "output: one number (VND)",
        ),
        answer="yes",
    ),
    VerifyExample(
        question="Bay ngay gan day tiem Sen phat sinh them bao nhieu tien ghi no",
        description=_desc(
            "measure: payments after the order day, dated in the period, on debt orders",
            "payments from: payments table",
            "excluded orders: cancelled; COD/marketplace labels in this shop: none found",
            "period: 2025-10-06 .. 2025-10-12 (order date)",
            "output: one number (VND)",
        ),
        answer="no",
    ),
    VerifyExample(
        question="Thang Sau tiem Latte nhap hang tong het bao nhieu tien",
        description=_desc(
            "measure: sum of purchase receipt totals",
            "source: purchases table (cancelled receipts excluded)",
            "period: 2025-06-01 .. 2025-06-30 (order date)",
            "output: one number (VND)",
        ),
        answer="yes",
    ),
    VerifyExample(
        question="Thang Sau tiem Latte nha cung cap nao nhap nhieu nhat",
        description=_desc(
            "measure: sum of purchase receipt totals",
            "source: purchases table (cancelled receipts excluded)",
            "period: 2025-06-01 .. 2025-06-30 (order date)",
            "output: one number (VND)",
        ),
        answer="no",
    ),
    VerifyExample(
        question="Quy 1 tiem Latte nha cung cap nao mang ve nhieu tien nhap nhat",
        description=_desc(
            "measure: supplier with the highest purchase receipt total",
            "source: purchases table (cancelled receipts excluded)",
            "period: 2025-01-01 .. 2025-03-31 (order date)",
            "output: one supplier name",
        ),
        answer="yes",
    ),
    VerifyExample(
        question="Quy 1 tiem Latte nhap hang tong bao nhieu",
        description=_desc(
            "measure: supplier with the highest purchase receipt total",
            "source: purchases table (cancelled receipts excluded)",
            "period: 2025-01-01 .. 2025-03-31 (order date)",
            "output: one supplier name",
        ),
        answer="no",
    ),
    VerifyExample(
        question="Gia nhap binh quan mot goi Hat arabica thang 3 o tiem Latte",
        description=_desc(
            "measure: quantity-weighted average purchase unit cost",
            'product: "Hat arabica"',
            "source: purchase_lines (cancelled receipts excluded)",
            "period: 2025-03-01 .. 2025-03-31 (order date)",
            "output: one number (VND per purchase unit)",
        ),
        answer="yes",
    ),
    VerifyExample(
        question="Thang 3 tiem Latte ban duoc bao nhieu goi Hat arabica",
        description=_desc(
            "measure: quantity-weighted average purchase unit cost",
            'product: "Hat arabica"',
            "source: purchase_lines (cancelled receipts excluded)",
            "period: 2025-03-01 .. 2025-03-31 (order date)",
            "output: one number (VND per purchase unit)",
        ),
        answer="no",
    ),
    VerifyExample(
        question="Het hom nay kho tiem Latte con bao nhieu thung Nuoc cam",
        description=_desc(
            "measure: on-hand stock at end of day",
            'product: "Nuoc cam"',
            'unit: "thung" (big unit -> [whole, leftover base units])',
            "source: stock_moves balance, else stock_snapshot on that date",
            "as-of: end of 2025-08-20",
            "output: one number (base) or list [whole big units, leftover base]",
        ),
        answer="yes",
    ),
    VerifyExample(
        question="Het hom nay gia tri hang ton kho tiem Latte bao nhieu",
        description=_desc(
            "measure: on-hand stock at end of day",
            'product: "Nuoc cam"',
            'unit: "thung" (big unit -> [whole, leftover base units])',
            "source: stock_moves balance, else stock_snapshot on that date",
            "as-of: end of 2025-08-20",
            "output: one number (base) or list [whole big units, leftover base]",
        ),
        answer="no",
    ),
    VerifyExample(
        question="Gia tri ton kho tiem Latte het ngay 10 thang 4",
        description=_desc(
            "measure: sum of on-hand base qty * products.unit_cost for stock > 0",
            "source: stock_moves/snapshot and products.unit_cost",
            "as-of: end of 2025-04-10",
            "output: one number (VND)",
        ),
        answer="yes",
    ),
    VerifyExample(
        question="Het ngay 10 thang 4 kho tiem Latte con bao nhieu chai Nuoc cam",
        description=_desc(
            "measure: sum of on-hand base qty * products.unit_cost for stock > 0",
            "source: stock_moves/snapshot and products.unit_cost",
            "as-of: end of 2025-04-10",
            "output: one number (VND)",
        ),
        answer="no",
    ),
    VerifyExample(
        question="Hom qua tiem Latte co may mat hang mua ve de ban ma het sach",
        description=_desc(
            "measure: count of resale products (seen in purchase_lines) with end-of-day stock 0",
            "source: purchase_lines + stock_moves/snapshot",
            "day: 2025-07-19",
            "output: one number (product count)",
        ),
        answer="yes",
    ),
    VerifyExample(
        question="Hom qua tiem Latte con bao nhieu mat hang trong kho",
        description=_desc(
            "measure: count of resale products (seen in purchase_lines) with end-of-day stock 0",
            "source: purchase_lines + stock_moves/snapshot",
            "day: 2025-07-19",
            "output: one number (product count)",
        ),
        answer="no",
    ),
    VerifyExample(
        question="Dem lo het han trong 14 ngay toi o tiem Latte",
        description=_desc(
            "measure: lots still in stock expiring within 14 days",
            "source: lots minus sale_lots up to as-of; cost uses each lot's unit_cost",
            "as-of: end of 2025-08-01",
            "output: count of lots",
        ),
        answer="yes",
    ),
    VerifyExample(
        question="Tien von cac lo het han trong 14 ngay toi o tiem Latte",
        description=_desc(
            "measure: lots still in stock expiring within 14 days",
            "source: lots minus sale_lots up to as-of; cost uses each lot's unit_cost",
            "as-of: end of 2025-08-01",
            "output: count of lots",
        ),
        answer="no",
    ),
    VerifyExample(
        question="Lo gan het han nhat cua Amoxicillin 500 o tiem Latte la lo nao",
        description=_desc(
            "measure: earliest-expiring lot still in stock and not yet expired",
            'product: "Amoxicillin 500"',
            "source: lots minus sale_lots up to as-of",
            "as-of: end of 2025-08-01",
            "output: one lot id",
        ),
        answer="yes",
    ),
    VerifyExample(
        question="Amoxicillin 500 o tiem Latte con bao nhieu vien trong kho",
        description=_desc(
            "measure: earliest-expiring lot still in stock and not yet expired",
            'product: "Amoxicillin 500"',
            "source: lots minus sale_lots up to as-of",
            "as-of: end of 2025-08-01",
            "output: one lot id",
        ),
        answer="no",
    ),
    VerifyExample(
        question="Thang 2 tiem Latte thu bao nhieu tu thuoc hoat chat Ibuprofen",
        description=_desc(
            "measure: revenue of products with this exact active ingredient",
            'active_ingredient: "Ibuprofen"',
            "source: order_lines revenue + drug_info (combined ingredients are separate)",
            "period: 2025-02-01 .. 2025-02-28 (order date)",
            "output: one number (VND)",
        ),
        answer="yes",
    ),
    VerifyExample(
        question="Thang 2 tiem Latte ty le thuoc ke don la bao nhieu phan tram",
        description=_desc(
            "measure: revenue of products with this exact active ingredient",
            'active_ingredient: "Ibuprofen"',
            "source: order_lines revenue + drug_info (combined ingredients are separate)",
            "period: 2025-02-01 .. 2025-02-28 (order date)",
            "output: one number (VND)",
        ),
        answer="no",
    ),
    VerifyExample(
        question="Thang 2 tiem Latte ty le doanh thu thuoc ke don bao nhieu",
        description=_desc(
            "measure: prescription (Rx) revenue / total revenue x 100",
            "source: order_lines revenue + drug_info.is_prescription",
            "period: 2025-02-01 .. 2025-02-28 (order date)",
            "output: one percent",
        ),
        answer="yes",
    ),
    VerifyExample(
        question="Thang 2 tiem Latte lai gop bao nhieu",
        description=_desc(
            "measure: prescription (Rx) revenue / total revenue x 100",
            "source: order_lines revenue + drug_info.is_prescription",
            "period: 2025-02-01 .. 2025-02-28 (order date)",
            "output: one percent",
        ),
        answer="no",
    ),
    VerifyExample(
        question="Lai gop thang Nam cua tiem Latte",
        description=_desc(
            "measure: revenue minus qty * cost_per_unit on revenue lines",
            "source: order_lines (cost_per_unit on the sold unit)",
            "period: 2025-05-01 .. 2025-05-31 (order date)",
            "output: one number (VND)",
        ),
        answer="yes",
    ),
    VerifyExample(
        question="Lai rong thang Nam cua tiem Latte sau moi khoan chi",
        description=_desc(
            "measure: revenue minus qty * cost_per_unit on revenue lines",
            "source: order_lines (cost_per_unit on the sold unit)",
            "period: 2025-05-01 .. 2025-05-31 (order date)",
            "output: one number (VND)",
        ),
        answer="no",
    ),
    VerifyExample(
        question="Lai rong thang Sau tiem Latte sau moi khoan chi phi",
        description=_desc(
            "measure: gross profit minus all expenses dated in the period",
            "source: order_lines cost_per_unit + expenses table",
            "period: 2025-06-01 .. 2025-06-30 (order date)",
            "output: one number (VND)",
        ),
        answer="yes",
    ),
    VerifyExample(
        question="Lai gop thang Sau tiem Latte",
        description=_desc(
            "measure: gross profit minus all expenses dated in the period",
            "source: order_lines cost_per_unit + expenses table",
            "period: 2025-06-01 .. 2025-06-30 (order date)",
            "output: one number (VND)",
        ),
        answer="no",
    ),
    VerifyExample(
        question="Tuan nay ai o tiem Latte ban mang ve nhieu tien nhat",
        description=_desc(
            "measure: employee with the highest revenue",
            "source: orders.employee_id + order_lines revenue",
            "period: 2025-09-01 .. 2025-09-07 (order date)",
            "output: one employee name",
        ),
        answer="yes",
    ),
    VerifyExample(
        question="Tuan nay mon nao o tiem Latte ban mang ve nhieu tien nhat",
        description=_desc(
            "measure: employee with the highest revenue",
            "source: orders.employee_id + order_lines revenue",
            "period: 2025-09-01 .. 2025-09-07 (order date)",
            "output: one employee name",
        ),
        answer="no",
    ),
)


def verify_description(
    question: str,
    description: str,
    client: Any,
    *,
    few_shot: bool | None = None,
) -> float:
    """Return p_yes for a pre-built description (ext families or E3 plans)."""
    use_few = settings.ext_verify_few_shot if few_shot is None else few_shot
    messages = _build_messages(question, description, few_shot=use_few)
    probs = client.chat_choice_probs(messages, list(_CHOICES))
    return float(probs.get("yes", 0.0))


def verify_ext_spec(
    question: str,
    spec: ExtQuerySpec,
    client: Any,
    rules: RuleConfig,
    cal: LunarCalendar,
    as_of: datetime.date,
    product_unit: str | None = None,
    customer_who: str | None = None,
    debt_meta: dict[str, str] | None = None,
) -> tuple[float, str]:
    """Return (p_yes, description)."""
    description = describe_ext_spec(
        spec,
        as_of,
        rules,
        cal,
        product_unit=product_unit,
        customer_who=customer_who,
        debt_meta=debt_meta,
    )
    return verify_description(question, description, client), description


def _build_messages(
    question: str, description: str, *, few_shot: bool = True
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": _SYSTEM}]
    for example in VERIFY_FEW_SHOTS if few_shot else ():
        messages.append(
            {
                "role": "user",
                "content": _user_content(example.question, example.description),
            }
        )
        messages.append({"role": "assistant", "content": example.answer})
    messages.append({"role": "user", "content": _user_content(question, description)})
    return messages


def _user_content(question: str, description: str) -> str:
    return (
        f"Question: {question}\n"
        f"Description:\n{description}\n"
        "Does the description compute exactly what the question asks, with every "
        "condition reflected? Answer yes or no."
    )

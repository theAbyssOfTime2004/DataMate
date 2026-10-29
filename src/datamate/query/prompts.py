from __future__ import annotations

import datetime
import json
from dataclasses import dataclass
from typing import Any

from datamate.query.context import ShopContext
from datamate.query.parse_schema import FILTER_KEYS, PERIOD_FIELDS

_WEEKDAYS = ("Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật")

_EXAMPLE_PRODUCTS = ["Bánh mì thịt", "Trà sữa trân châu", "Bánh croissant"]

EXAMPLE_CONTEXT = ShopContext(
    payment_methods=["Momo", "Tiền mặt"],
    channels=["tại quầy", "grab"],
    segments=["khách quen", "khách mới"],
    categories=["bánh ngọt", "trà"],
    product_names=list(_EXAMPLE_PRODUCTS),
    has_order_ts=True,
    data_start=datetime.date(2024, 1, 1),
    data_end=datetime.date(2024, 12, 31),
)

_SYSTEM = "\n".join(  # noqa: FLY002
    [
        "You translate one Vietnamese shop question into a flat JSON object.",
        "You do not calculate the answer and you do not calculate dates.",
        "",
        "Metrics (the only ones in scope):",
        "- revenue: money from orders that are completed or shipping, excluding cancelled",
        "  orders and any order that has a return. Sum of line amounts in the period.",
        "- order_count: number of those same orders (not cancelled, not returned).",
        "- cancel_count: number of cancelled orders.",
        "- cancel_rate: cancelled orders divided by all orders placed in the period, times 100.",
        "- avg_order_value: revenue divided by order_count.",
        "- revenue_share: revenue of the filtered part divided by revenue of the whole shop",
        "  in the same period, times 100. Walk-in customers stay in the denominator.",
        "",
        "Kinds:",
        "- value: one number for one period.",
        "- compare: two numbers. op is pct_change ((asked - baseline) / baseline * 100) or",
        "  diff (asked - baseline). compare_kind says how to get the baseline.",
        "- series: the same metric over each listed period, in that order.",
        "- multi: the same metric and period, once per entry in multi_filters.",
        "- trend: whether the metric is increasing or decreasing across the periods.",
        "- argmax: the day or month with the max or min value. Ties go to the earliest date.",
        "",
        "Period kinds. Copy numbers that are written in the question. For relative wording,",
        "use the kind or year_rel. Never turn 'tháng này', 'tuần rồi', 'năm ngoái', or",
        "'Tết năm nay' into calendar dates.",
        "- day: an explicit calendar date. Example: ngày 3/3/2024 -> date 2024-03-03.",
        "- today: hôm nay.",
        "- yesterday: hôm qua.",
        "- last_n_days: 7 ngày gần nhất -> n=7, including today.",
        "- week_this: tuần này, Monday through today.",
        "- week_last: tuần rồi / tuần trước, the previous Monday-Sunday week.",
        "- month: tháng 5/2025 -> month=5, year=2025. tháng 5 with no year -> year null.",
        "- month_this: tháng này, day 1 through today.",
        "- month_last: the whole previous calendar month, not the same-day-count comparison.",
        "- quarter: quý 3 with no year -> q=3, year null. quý 3/2024 -> q=3, year=2024.",
        "- year: năm 2024 -> year=2024. năm ngoái -> year_rel=-1. năm nay -> year_rel=0.",
        "- ytd: từ đầu năm đến hôm nay.",
        "- range: both endpoints are written as calendar dates.",
        "- tet: Tết năm nay -> year_rel=0. Tết năm ngoái -> year_rel=-1. Tết 2024 -> year=2024.",
        "- before_tet: 17 ngày trước Tết năm nay -> n_days=17, year_rel=0.",
        "",
        "compare_kind:",
        "- explicit: the question names both periods. Asked period first, baseline second.",
        "- previous_period: so với kỳ trước, or tháng trước cùng số ngày. One period only.",
        "- same_period_last_year: cùng kỳ năm ngoái. One period only.",
        "- before_tet_same_length: same number of days immediately before Tết. One period only.",
        "- null: not a period comparison. Use filters_b when both sides share a period but",
        "  use different filters, such as ca chiều versus ca sáng. filters is the asked side",
        "  and filters_b is the baseline.",
        "",
        "Vietnamese wording. Questions are often typed without diacritics or abbreviated:",
        "- doanh thu, dthu, dt = revenue. bn, bnhiu, bao nhiu = bao nhiêu. đơn = order.",
        "- 'tra bang Momo' is trả bằng (paid with, a payment_method), not the drink trà.",
        "- ca sáng / ca chiều = morning / afternoon shift. tuần rồi = tuần trước = week_last.",
        "- tăng liên tục, tăng đều, giảm dần across listed periods = trend.",
        "- N tháng liên tiếp kết thúc ở tháng X = series of N months ending at X, oldest first.",
        "",
        "Leave every unused field null, and leave unused filter lists empty.",
        "Shift is morning or afternoon only when the shop lists those shifts.",
        "Choose filter values only from the lists in the user message.",
        "Product values must be copied from the product candidates.",
        "",
        "Set in_scope to false when the question asks for anything else, including debt,",
        "inventory, purchases, profit, best-selling items, customer counts or lists, prices,",
        "product counts, or units sold.",
        "",
        "The worked examples are an invented bakery, Tiệm bánh Hoa Mai.",
        "Their payment values, products, and dates are not this shop.",
        "Use them only as patterns.",
    ]
)


@dataclass(frozen=True)
class FewShot:
    question: str
    as_of: datetime.date
    parsed: dict[str, Any]


def _empty_filters() -> dict[str, Any]:
    filters: dict[str, Any] = {key: [] for key in FILTER_KEYS if key != "shift"}
    filters["shift"] = None
    return filters


def _filters(**overrides: Any) -> dict[str, Any]:
    filters = _empty_filters()
    filters.update(overrides)
    return filters


def _period(kind: str, **fields: Any) -> dict[str, Any]:
    row: dict[str, Any] = {name: None for name in PERIOD_FIELDS}
    row["kind"] = kind
    row.update(fields)
    return row


def _parsed(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "in_scope": True,
        "kind": "value",
        "metric": "revenue",
        "periods": [],
        "compare_kind": None,
        "op": None,
        "filters": _empty_filters(),
        "filters_b": None,
        "multi_filters": [],
        "direction": None,
        "granularity": None,
        "mode": None,
    }
    row.update(overrides)
    return row


def _out(question: str) -> FewShot:
    return FewShot(
        question,
        datetime.date(2024, 11, 20),
        _parsed(in_scope=False, periods=[_period("month_this")]),
    )


_DAY = datetime.date(2024, 11, 20)

FEW_SHOT_SETS: tuple[tuple[FewShot, ...], tuple[FewShot, ...]] = (
    (
        FewShot(
            "dthu Hoa Mai tu dau thang toi gio duoc bao nhieu roi ta",
            _DAY,
            _parsed(periods=[_period("month_this")]),
        ),
        FewShot(
            "Hom qua tiem Hoa Mai ban banh croissant thu ve bn tien vay",
            _DAY,
            _parsed(
                periods=[_period("yesterday")],
                filters=_filters(product=["Bánh croissant"]),
            ),
        ),
        FewShot(
            "Nhóm bánh ngọt ở Hoa Mai trong tháng 8 năm 2024 đem về doanh thu bao nhiêu?",
            _DAY,
            _parsed(
                periods=[_period("month", month=8, year=2024)],
                filters=_filters(category=["bánh ngọt"]),
            ),
        ),
        FewShot(
            "Khach tra bang Momo chiem may % dthu cua Hoa Mai trong thang 9/2024",
            _DAY,
            _parsed(
                metric="revenue_share",
                periods=[_period("month", month=9, year=2024)],
                filters=_filters(payment_method=["Momo"]),
            ),
        ),
        FewShot(
            "Dat thang 4/2024 canh thang 4/2023 thi dthu Hoa Mai doi bao nhieu phan tram",
            _DAY,
            _parsed(
                kind="compare",
                periods=[
                    _period("month", month=4, year=2024),
                    _period("month", month=4, year=2023),
                ],
                compare_kind="explicit",
                op="pct_change",
            ),
        ),
        FewShot(
            "Ke dthu tung thang cua Hoa Mai: thang 1, thang 2, thang 3 nam 2024, theo dung thu tu",
            _DAY,
            _parsed(
                kind="series",
                periods=[
                    _period("month", month=1, year=2024),
                    _period("month", month=2, year=2024),
                    _period("month", month=3, year=2024),
                ],
            ),
        ),
        FewShot(
            "Liet ke dthu 4 thang lien tiep cua Hoa Mai, thang cuoi la 2/2024, xep tu cu den moi",
            _DAY,
            _parsed(
                kind="series",
                periods=[
                    _period("month", month=11, year=2023),
                    _period("month", month=12, year=2023),
                    _period("month", month=1, year=2024),
                    _period("month", month=2, year=2024),
                ],
            ),
        ),
        FewShot(
            "Doanh thu Hoa Mai qua thang 8, 9, 10 nam 2024 co giam lien tuc khong",
            _DAY,
            _parsed(
                kind="trend",
                periods=[
                    _period("month", month=8, year=2024),
                    _period("month", month=9, year=2024),
                    _period("month", month=10, year=2024),
                ],
                direction="decreasing",
            ),
        ),
        FewShot(
            "Ngày đắt hàng nhất của Hoa Mai trong tháng 11/2024 là ngày mấy, tính theo doanh thu",
            _DAY,
            _parsed(
                kind="argmax",
                periods=[_period("month", month=11, year=2024)],
                granularity="day",
                mode="max",
            ),
        ),
        FewShot(
            "Hoa Mai thang 6/2024: dthu khach tra Momo bn, dthu khach tra tien mat bn",
            _DAY,
            _parsed(
                kind="multi",
                periods=[_period("month", month=6, year=2024)],
                multi_filters=[
                    _filters(payment_method=["Momo"]),
                    _filters(payment_method=["Tiền mặt"]),
                ],
            ),
        ),
        FewShot(
            "Gia tri trung binh moi don cua Hoa Mai tinh tu mung 1 thang 1 toi nay la bnhiu",
            _DAY,
            _parsed(metric="avg_order_value", periods=[_period("ytd")]),
        ),
        _out("Cong no khach quen cua tiem banh Hoa Mai hien dang la bao nhieu"),
        _out("So luong banh mi thit Hoa Mai ban ra trong tuan nay la bao nhieu cai"),
    ),
    (
        FewShot(
            "Hoa Mai chốt được mấy đơn trong ca chiều của tuần trước?",
            _DAY,
            _parsed(
                metric="order_count",
                periods=[_period("week_last")],
                filters=_filters(shift="afternoon"),
            ),
        ),
        FewShot(
            "Hom qua Hoa Mai dthu bn",
            _DAY,
            _parsed(periods=[_period("yesterday")]),
        ),
        FewShot(
            "dthu thang nay cua Hoa Mai so voi thang truoc cung so ngay thi tang bn %",
            _DAY,
            _parsed(
                kind="compare",
                periods=[_period("month_this")],
                compare_kind="previous_period",
                op="pct_change",
            ),
        ),
        FewShot(
            "Doanh thu dịp Tết năm nay của Hoa Mai so với Tết năm ngoái chênh mấy %?",
            _DAY,
            _parsed(
                kind="compare",
                periods=[_period("tet", year_rel=0), _period("tet", year_rel=-1)],
                compare_kind="explicit",
                op="pct_change",
            ),
        ),
        FewShot(
            "7 ngay qua ti le don bi huy o Hoa Mai la bao nhieu",
            _DAY,
            _parsed(metric="cancel_rate", periods=[_period("last_n_days", n=7)]),
        ),
        FewShot(
            "Quy 3 Hoa Mai dat dthu bao nhieu",
            _DAY,
            _parsed(periods=[_period("quarter", q=3)]),
        ),
        FewShot(
            "17 ngày trước Tết năm nay Hoa Mai thu về bao nhiêu doanh thu?",
            _DAY,
            _parsed(periods=[_period("before_tet", n_days=17, year_rel=0)]),
        ),
        FewShot(
            "dthu thang nay cua Hoa Mai so voi cung ky nam ngoai giam may %",
            _DAY,
            _parsed(
                kind="compare",
                periods=[_period("month_this")],
                compare_kind="same_period_last_year",
                op="pct_change",
            ),
        ),
        FewShot(
            "Dịp Tết năm nay doanh thu Hoa Mai hơn kém bao nhiêu % so với quãng ngày dài "
            "tương đương ngay trước Tết?",
            _DAY,
            _parsed(
                kind="compare",
                periods=[_period("tet", year_rel=0)],
                compare_kind="before_tet_same_length",
                op="pct_change",
            ),
        ),
        FewShot(
            "Tuan roi o Hoa Mai, dthu ca chieu cao hon ca sang may %",
            _DAY,
            _parsed(
                kind="compare",
                periods=[_period("week_last")],
                op="pct_change",
                filters=_filters(shift="afternoon"),
                filters_b=_filters(shift="morning"),
            ),
        ),
        FewShot(
            "Khach quen dong gop may phan tram doanh thu Hoa Mai trong thang 2/2024",
            _DAY,
            _parsed(
                metric="revenue_share",
                periods=[_period("month", month=2, year=2024)],
                filters=_filters(customer_segment=["khách quen"]),
            ),
        ),
        FewShot(
            "Kenh grab chiem may phan tram dthu Hoa Mai trong thang 7/2024",
            _DAY,
            _parsed(
                metric="revenue_share",
                periods=[_period("month", month=7, year=2024)],
                filters=_filters(channel=["grab"]),
            ),
        ),
        FewShot(
            "dthu Hoa Mai co tang deu qua thang 5, 6, 7/2024 hay khong",
            _DAY,
            _parsed(
                kind="trend",
                periods=[
                    _period("month", month=5, year=2024),
                    _period("month", month=6, year=2024),
                    _period("month", month=7, year=2024),
                ],
                direction="increasing",
            ),
        ),
        _out("Kho banh croissant cua Hoa Mai con ton bao nhieu chiec"),
    ),
)


def build_messages(
    question_text: str,
    as_of: datetime.date,
    ctx: ShopContext,
    few_shot_set: int,
    example_limit: int | None = None,
) -> list[dict[str, str]]:
    examples = FEW_SHOT_SETS[few_shot_set % len(FEW_SHOT_SETS)]
    if example_limit is not None:
        examples = examples[: max(example_limit, 0)]
    messages = [{"role": "system", "content": _SYSTEM}]
    for example in examples:
        messages.append(
            {
                "role": "user",
                "content": _user_message(
                    example.question, example.as_of, EXAMPLE_CONTEXT, list(_EXAMPLE_PRODUCTS)
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
    shifts = "morning, afternoon" if ctx.has_order_ts else "(not recorded)"
    lines = [
        f"Hôm nay (as_of): {as_of.isoformat()} ({weekday})",
        "Available filter values (copy them exactly, or leave the list empty):",
        f"payment_method: {_join(ctx.payment_methods)}",
        f"channel: {_join(ctx.channels)}",
        f"customer_segment: {_join(ctx.segments)}",
        f"category: {_join(ctx.categories)}",
        f"product: {_join(products)}",
        f"shift: {shifts}",
        f"Question: {question_text}",
    ]
    return "\n".join(lines)


def _join(values: list[str]) -> str:
    if not values:
        return "(none)"
    return ", ".join(values)

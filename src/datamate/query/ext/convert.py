"""Turn flat extension-parser JSON into an ExtQuerySpec."""

from __future__ import annotations

import datetime
from typing import Any

from pydantic import TypeAdapter, ValidationError

from datamate.query.convert import ConvertError, period_from_raw
from datamate.query.ext.spec import (
    FAMILIES_WITH_CATEGORY,
    ExtQuerySpec,
    NoneQuery,
    PriceIncreaseEffectQuery,
    PriceOnDateQuery,
    ProductsInMinOrdersQuery,
    QtySoldQuery,
    TopProductsQuery,
    UnsoldProductsQuery,
)
from datamate.query.grounding import QuestionText
from datamate.timeres.periods import PeriodExpr

_SPEC_ADAPTER: TypeAdapter[ExtQuerySpec] = TypeAdapter(ExtQuerySpec)

# Count-style wording (folded). List/name cues override toward "list".
_COUNT_PHRASES = (
    "bao nhieu",
    "bnhiu",
    "bn",
    "dem",
    "co bao nhieu",
)
_COUNT_WORDS = frozenset({"may", "dem"})
_LIST_PHRASES = (
    "liet ke",
    "danh sach",
    "ke ten",
    "ten cac",
    "nhung mat hang nao",
    "mat hang nao",
    "mon nao",
    "cac mon nao",
    "cac mat hang",
)
# Stock / purchase wording → family none until E4d.
_STOCK_PHRASES = (
    "ton kho",
    "con lai trong kho",
    "het hang",
)
_PURCHASE_PHRASES = (
    "nhap hang",
    "gia nhap",
    "nha cung cap",
)
_STOCK_PURCHASE_REJECT = frozenset({"qty_sold", "top_products", "unsold_products"})

_GROUP_WORDS = frozenset({"nhom", "loai"})
# "gộp" alone is also in "lãi gộp"; require a grouping phrase.
_GROUP_PHRASES = ("gop moi", "gop tat", "gop cac", "gop theo", "gop hang", "gop chung")
_MONEY_WORDS = frozenset({"tien", "dthu"})
_MONEY_PHRASES = ("doanh thu",)
_PROFIT_WORDS = frozenset({"lai", "loi"})
_DELIVERY_WORDS = frozenset({"giao", "ship"})
_DELIVERY_PHRASES = ("van chuyen", "cho di", "giao hang")
_E4A_ANSWER_FAMILIES = frozenset(
    {
        "qty_sold",
        "top_products",
        "unsold_products",
        "products_in_min_orders",
        "price_on_date",
        "price_increase_effect",
    }
)


def to_ext_query_spec(
    parsed: dict[str, Any],
    as_of: datetime.date,
    question: str | None = None,
) -> ExtQuerySpec:
    text = QuestionText(question) if question is not None else None
    family = parsed.get("family")
    if family is None or family == "none":
        return NoneQuery()

    if parsed.get("unhandled_condition") is True:
        return NoneQuery()

    # Customer questions are not product families (E4b).
    if text is not None and text.has_word("khach"):
        if family == "products_in_min_orders":
            return NoneQuery()
        if family in {
            "qty_sold",
            "top_products",
            "unsold_products",
            "price_on_date",
            "price_increase_effect",
        } and _looks_like_customer_ranking(text):
            return NoneQuery()

    # Stock / purchase questions wait for E4d.
    if text is not None and family in _STOCK_PURCHASE_REJECT and _has_stock_or_purchase_cue(
        text
    ):
        return NoneQuery()

    if text is not None and family in _E4A_ANSWER_FAMILIES and _has_delivery_cue(text):
        return NoneQuery()

    product = _optional_str(parsed.get("product"))
    category = _optional_str(parsed.get("category"))
    if (
        family in FAMILIES_WITH_CATEGORY
        and text is not None
        and _has_group_cue(text)
        and product is None
        and category is None
    ):
        return NoneQuery()

    if family == "qty_sold":
        if text is not None and _has_money_cue(text):
            return NoneQuery()
        period = _need_period(parsed.get("period"), as_of, "period")
        base_unit = _optional_str(parsed.get("base_unit"))
        size = _optional_size(parsed.get("size"))
        if product is not None:
            _require_grounded_product(text, product)
        if category is not None:
            _require_grounded_category(text, category)
        try:
            return QtySoldQuery(
                product=product,
                base_unit=base_unit,
                category=category,
                size=size,
                period=period,
            )
        except ValidationError as exc:
            raise ConvertError(str(exc)) from exc
    if family == "top_products":
        period = _need_period(parsed.get("period"), as_of, "period")
        k = _need_int(parsed.get("k"), "k")
        by = parsed.get("by")
        mode = parsed.get("mode") or "max"
        size = _optional_size(parsed.get("size"))
        if by not in {"qty_base", "revenue", "gross_profit"}:
            raise ConvertError(f"top_products by must be qty_base|revenue|gross_profit, got {by!r}")
        if mode not in {"max", "min"}:
            raise ConvertError(f"top_products mode must be max|min, got {mode!r}")
        if text is not None:
            if by == "qty_base" and _has_money_cue(text):
                return NoneQuery()
            has_profit = _has_profit_cue(text)
            if by == "gross_profit" and not has_profit:
                return NoneQuery()
            if has_profit and by != "gross_profit":
                return NoneQuery()
        if category is not None:
            _require_grounded_category(text, category)
        return TopProductsQuery(
            k=k, by=by, mode=mode, category=category, size=size, period=period
        )  # type: ignore[arg-type]
    if family == "unsold_products":
        period = _need_period(parsed.get("period"), as_of, "period")
        output = _resolve_count_or_list(parsed.get("output"), text)
        if category is not None:
            _require_grounded_category(text, category)
        return UnsoldProductsQuery(
            output=output, category=category, period=period
        )  # type: ignore[arg-type]
    if family == "products_in_min_orders":
        period = _need_period(parsed.get("period"), as_of, "period")
        n = _need_int(parsed.get("n"), "n")
        output = _resolve_count_or_list(parsed.get("output"), text)
        if category is not None:
            _require_grounded_category(text, category)
        return ProductsInMinOrdersQuery(
            n=n, output=output, category=category, period=period
        )  # type: ignore[arg-type]
    if family == "price_on_date":
        if product is None:
            raise ConvertError("price_on_date needs product")
        _require_grounded_product(text, product)
        date = _need_period(parsed.get("date"), as_of, "date")
        return PriceOnDateQuery(product=product, date=date)
    if family == "price_increase_effect":
        if product is None:
            raise ConvertError("price_increase_effect needs product")
        _require_grounded_product(text, product)
        n_days = _need_int(parsed.get("n_days"), "n_days")
        raw_date = parsed.get("date")
        date = period_from_raw(raw_date, as_of) if raw_date is not None else None
        return PriceIncreaseEffectQuery(product=product, n_days=n_days, date=date)
    raise ConvertError(f"unknown family {family!r}")


def _resolve_count_or_list(raw: object, text: QuestionText | None) -> str:
    """Count unless the question asks to list or name the items."""
    if text is not None and _wants_list(text):
        return "list"
    if text is not None and _wants_count(text):
        return "count"
    if raw in {"count", "list"}:
        return str(raw)
    return "count"


def _wants_list(text: QuestionText) -> bool:
    return any(text.has_phrase(p) for p in _LIST_PHRASES)


def _wants_count(text: QuestionText) -> bool:
    if any(text.has_phrase(p) for p in _COUNT_PHRASES):
        return True
    if any(text.has_word(w) for w in _COUNT_WORDS):
        return True
    # "số ... là" without an explicit list/name cue
    return text.has_word("so") and text.has_word("la")


def _looks_like_customer_ranking(text: QuestionText) -> bool:
    return text.has_phrase("mua nhieu") or text.has_phrase("nhieu nhat") or text.has_phrase(
        "tu "
    )


def _has_stock_or_purchase_cue(text: QuestionText) -> bool:
    return any(text.has_phrase(p) for p in _STOCK_PHRASES) or any(
        text.has_phrase(p) for p in _PURCHASE_PHRASES
    )


def _has_group_cue(text: QuestionText) -> bool:
    if any(text.has_word(w) for w in _GROUP_WORDS):
        return True
    if any(text.has_phrase(p) for p in _GROUP_PHRASES):
        return True
    # "dòng" / "hãng" keep diacritics so they do not collide with đồng / hàng.
    return any(word in {"dòng", "hãng"} for word in text._marked)


def _has_money_cue(text: QuestionText) -> bool:
    if any(text.has_word(w) for w in _MONEY_WORDS):
        return True
    if any(text.has_phrase(p) for p in _MONEY_PHRASES):
        return True
    if any(text.has_word(w) for w in _PROFIT_WORDS):
        return True
    return any(word == "đồng" for word in text._marked)


def _has_profit_cue(text: QuestionText) -> bool:
    return any(text.has_word(w) for w in _PROFIT_WORDS)


def _has_delivery_cue(text: QuestionText) -> bool:
    if any(text.has_word(w) for w in _DELIVERY_WORDS):
        return True
    if any(text.has_phrase(p) for p in _DELIVERY_PHRASES):
        return True
    return any(word == "chở" for word in text._marked)


def _require_grounded_product(text: QuestionText | None, product: str) -> None:
    if text is None:
        return
    if not text.mentions_product(product):
        raise ConvertError(f"product {product!r} is not grounded in the question")


def _require_grounded_category(text: QuestionText | None, category: str) -> None:
    if text is None:
        return
    if not text.mentions_value(category):
        raise ConvertError(f"category {category!r} is not grounded in the question")


def _need_period(raw: object, as_of: datetime.date, label: str) -> PeriodExpr:
    if raw is None:
        raise ConvertError(f"{label} is required")
    return period_from_raw(raw, as_of)


def _need_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConvertError(f"{field} must be an integer")
    if value < 1:
        raise ConvertError(f"{field} must be >= 1")
    return value


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConvertError("expected a string")
    text = value.strip()
    return text or None


def _optional_size(value: object) -> str | None:
    if value is None:
        return None
    if value not in {"M", "L", "m", "l"}:
        raise ConvertError(f"size must be M or L, got {value!r}")
    return str(value).upper()

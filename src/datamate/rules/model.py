from __future__ import annotations

import datetime
import json
import unicodedata
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from datamate.mapping.model import OrderStatus


class TetRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_lunar_month: int = 12
    start_lunar_day: int = 23
    start_year_offset: int = -1
    end_lunar_month: int = 1
    end_lunar_day: int = 10


def _default_shifts() -> dict[str, tuple[datetime.time, datetime.time]]:
    return {
        "morning": (datetime.time(6, 0, 0), datetime.time(13, 59, 59)),
        "afternoon": (datetime.time(14, 0, 0), datetime.time(21, 59, 59)),
    }


class RuleConfig(BaseModel):
    """Train-rulebook defaults. Other splits load their own JSON; nothing is keyed on a shop."""

    model_config = ConfigDict(extra="forbid")

    revenue_statuses: list[OrderStatus] = Field(
        default_factory=lambda: ["completed", "shipping"]
    )
    exclude_orders_with_return: bool = True
    cancel_rate_denominator: Literal["all_orders"] = "all_orders"
    shifts: dict[str, tuple[datetime.time, datetime.time]] = Field(default_factory=_default_shifts)
    week_start: Literal["monday"] = "monday"
    month_this_compare: Literal["same_day_count"] = "same_day_count"
    tet: TetRule = Field(default_factory=TetRule)
    percent_decimals: int = 2
    rounding: Literal["half_up"] = "half_up"
    non_product_line_labels: list[str] = Field(default_factory=lambda: ["Phí phục vụ"])
    # E4a product families (train rulebook §3 / §5).
    ranking_measure: Literal["qty_base", "revenue", "gross_profit"] = "qty_base"
    ranking_tie_break: Literal["code_asc"] = "code_asc"
    price_on_date_include_cancelled_returned: bool = True
    size_suffixes: dict[str, str] = Field(
        default_factory=lambda: {"M": "(M)", "L": "(L)"}
    )
    # E4b customer identity (train rulebook §3). Mode is detected from shop columns
    # at query time (codes if customer_id is recorded, else folded name).
    customer_tie_break: Literal["code_asc"] = "code_asc"
    # E4c debt (train rulebook §4): exclude COD / marketplace orders from debt.
    # Labels are matched after normalize_label against the shop's own values.
    debt_exclude_payment_labels: list[str] = Field(
        default_factory=lambda: ["cod", "c.o.d", "thu hộ", "thu ho"]
    )
    debt_exclude_channel_labels: list[str] = Field(
        default_factory=lambda: [
            "shopee",
            "lazada",
            "tiki",
            "tiktok",
            "tiktok shop",
            "sàn",
            "san",
            "sàn tmđt",
            "san tmdt",
        ]
    )
    # E4d pharmacy lots (train rulebook §6): expiry window is inclusive of as_of + n_days.
    lot_expiry_inclusive: bool = True

    @field_validator(
        "non_product_line_labels",
        "debt_exclude_payment_labels",
        "debt_exclude_channel_labels",
    )
    @classmethod
    def _nfc_labels(cls, value: list[str]) -> list[str]:
        return [unicodedata.normalize("NFC", label) for label in value]


def load_rules(path: Path | str) -> RuleConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return RuleConfig.model_validate(payload)

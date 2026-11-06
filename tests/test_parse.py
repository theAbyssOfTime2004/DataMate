from __future__ import annotations

import datetime
import unicodedata
from decimal import Decimal

import pytest

from datamate.parse import (
    normalize_label,
    parse_date,
    parse_decimal,
    parse_money,
    parse_qty,
    parse_timestamp,
    year_hint,
)


@pytest.mark.parametrize(
    ("sheet", "file", "expected"),
    [
        ("Thang 02 2025", "SoBanHang_1.xlsx", 2025),
        ("T01_2025", None, 2025),
        ("Tháng 01-2025", None, 2025),
        ("T7-2025", None, 2025),
        ("Sheet1", "BanHang_T10_2026.xlsx", 2026),
        ("Sheet1", "SoBanHang.xlsx", None),
        (None, None, None),
    ],
)
def test_year_hint(sheet: str | None, file: str | None, expected: int | None) -> None:
    assert year_hint(sheet, file) == expected


@pytest.mark.parametrize(
    ("raw", "default_year", "expected"),
    [
        ("2025-02-01", None, datetime.date(2025, 2, 1)),
        ("2025-02-01 06:00:21", None, datetime.date(2025, 2, 1)),
        ("2025-02-01T06:00:21", None, datetime.date(2025, 2, 1)),
        ("01/02/2025", None, datetime.date(2025, 2, 1)),
        ("1/2/2025", None, datetime.date(2025, 2, 1)),
        ("01-02-2025", None, datetime.date(2025, 2, 1)),
        ("01.02.2025", None, datetime.date(2025, 2, 1)),
        ("01/02/2025 14:30:00", None, datetime.date(2025, 2, 1)),
        ("1 thg 6 2025", None, datetime.date(2025, 6, 1)),
        ("19 thg 8 2026", None, datetime.date(2026, 8, 19)),
        ("1 THG 6", 2025, datetime.date(2025, 6, 1)),
        ("1 thg 6", None, None),
        ("1/1", 2025, datetime.date(2025, 1, 1)),
        ("28/4", 2026, datetime.date(2026, 4, 28)),
        ("1/1", None, None),
        ("45689", None, datetime.date(2025, 2, 1)),
        ("45717", None, datetime.date(2025, 3, 1)),
        ("45958.2501", None, datetime.date(2025, 10, 28)),
        ("46279", None, datetime.date(2026, 9, 14)),
        ("\xa0 01/02/2025 \xa0", None, datetime.date(2025, 2, 1)),
        ("31/02/2025", None, None),
        ("29/02/2025", None, None),
        ("19999", None, None),
        ("80001", None, None),
        ("", None, None),
        (None, None, None),
        ("not a date", None, None),
    ],
)
def test_parse_date(
    raw: str | None, default_year: int | None, expected: datetime.date | None
) -> None:
    assert parse_date(raw, default_year=default_year) == expected


@pytest.mark.parametrize(
    ("raw", "default_year", "expected"),
    [
        ("2025-02-01 06:00:21", None, datetime.datetime(2025, 2, 1, 6, 0, 21)),
        ("2025-02-01T06:00:21", None, datetime.datetime(2025, 2, 1, 6, 0, 21)),
        ("01/02/2025 14:05", None, datetime.datetime(2025, 2, 1, 14, 5, 0)),
        ("01/02/2025 14:05:59", None, datetime.datetime(2025, 2, 1, 14, 5, 59)),
        ("45958.25", None, datetime.datetime(2025, 10, 28, 6, 0, 0)),
        ("45958.2501", None, datetime.datetime(2025, 10, 28, 6, 0, 9)),
        # Date-only inputs must return None
        ("2025-02-01", None, None),
        ("01/02/2025", None, None),
        ("45958", None, None),
        ("45958.0", None, None),
        (None, None, None),
    ],
)
def test_parse_timestamp(
    raw: str | None, default_year: int | None, expected: datetime.datetime | None
) -> None:
    assert parse_timestamp(raw, default_year=default_year) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Rule (a): <int>tr<digits>
        ("1tr440", 1_440_000),
        ("9tr960", 9_960_000),
        ("36tr936", 36_936_000),
        ("1tr4", 1_400_000),
        # Rule (b): <num>[.,<frac>] (tr|triệu|k|nghìn|ngàn)
        ("16,56tr", 16_560_000),
        ("294,7703tr", 294_770_300),
        ("2,173tr", 2_173_000),
        ("25k", 25_000),
        ("25000k", 25_000_000),
        ("1840k", 1_840_000),
        ("1,5 triệu", 1_500_000),
        ("50 nghìn", 50_000),
        ("50 ngàn", 50_000),
        # Rule (c): thousands-separated
        ("560,000", 560_000),
        ("115,000.0000", 115_000),
        ("92,286.0000", 92_286),
        ("1.200.000", 1_200_000),
        ("55.000", 55_000),
        ("55.000đ", 55_000),
        ("-53.500đ", -53_500),
        ("1.032.648.800đ", 1_032_648_800),
        ("16 076", 16_076),
        ("50 000", 50_000),
        # Rule (d): plain
        ("45000", 45_000),
        ("45000.0", 45_000),
        ("-53500", -53_500),
        ("0", 0),
        ("0đ", 0),
        ("\xa0 55.000 VND \xa0", 55_000),
        # Invalid / non-integral
        ("45.5", None),
        ("abc", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_money(raw: str | None, expected: int | None) -> None:
    res = parse_money(raw)
    assert res == expected
    if res is not None:
        assert type(res) is int


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1", Decimal(1)),
        ("-2", Decimal(-2)),
        ("1.5", Decimal("1.5")),
        ("1,5", Decimal("1.5")),
        ("1.0000", Decimal("1.0000")),
        ("560,000", Decimal(560000)),
        ("115,000.0000", Decimal("115000.0000")),
        ("1.200.000", Decimal(1200000)),
        ("55.000", Decimal(55000)),
        ("16 076", Decimal(16076)),
        ("50 000", Decimal(50000)),
        ("", None),
        (None, None),
        ("abc", None),
    ],
)
def test_parse_decimal(raw: str | None, expected: Decimal | None) -> None:
    assert parse_decimal(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("3", (Decimal(3), None)),
        ("3 chai", (Decimal(3), "chai")),
        ("2 xe 4m3", (Decimal(2), "xe 4m3")),
        ("-1", (Decimal(-1), None)),
        ("15 vỉ", (Decimal(15), "vỉ")),
        ("1,5 kg", (Decimal("1.5"), "kg")),
        (unicodedata.normalize("NFD", "  15 vỉ \xa0"), (Decimal(15), "vỉ")),
        ("", None),
        (None, None),
        ("chai", None),
    ],
)
def test_parse_qty(raw: str | None, expected: tuple[Decimal, str | None] | None) -> None:
    res = parse_qty(raw)
    assert res == expected
    if res is not None and res[1] is not None:
        assert unicodedata.is_normalized("NFC", res[1])


def test_normalize_label() -> None:
    assert normalize_label("Hủy") == "hủy"
    assert normalize_label("  hủy \xa0 ") == "hủy"
    assert normalize_label(unicodedata.normalize("NFD", "Hủy")) == "hủy"
    assert normalize_label("huy") == "huy"
    assert normalize_label("Hủy") != normalize_label("huy")
    assert normalize_label("  Cà   phê   sữa (M) ") == "cà phê sữa (m)"
    assert normalize_label("") is None
    assert normalize_label("   ") is None
    assert normalize_label(None) is None


def test_expand_short_years_only_where_a_month_cannot_be() -> None:
    from datamate.parse.text import expand_short_years

    assert expand_short_years("Th 11/25 dthu") == "Th 11/2025 dthu"
    assert expand_short_years("so sanh 3/25, 4/25") == "so sanh 3/2025, 4/2025"
    assert expand_short_years("th3-26") == "th3-2026"
    # Day/month, full dates, ratios and decimals stay as written.
    for text in ("ngay 5/12", "tu 15/02/2026", "24/7", "1.2/25x", "22 vien/m2"):
        assert expand_short_years(text) == text


def test_spell_month_abbrev_keeps_weekdays() -> None:
    from datamate.parse.text import spell_month_abbrev

    assert spell_month_abbrev("Dthu T4/2025 bao nhieu") == "Dthu tháng 4/2025 bao nhieu"
    assert spell_month_abbrev("th7, th8, th9/2025") == "tháng 7, tháng 8, tháng 9/2025"
    assert spell_month_abbrev("Thg 11 ban") == "tháng 11 ban"
    assert spell_month_abbrev("Dthu ca chiều 3-2026 cao") == "Dthu ca chiều tháng 3/2026 cao"
    assert spell_month_abbrev("th3-2026 ban") == "tháng 3/2026 ban"
    assert spell_month_abbrev("ngày 12-3-2026") == "ngày 12-3-2026"
    for text in ("T2 tuần này bán bao nhiêu", "thứ 3 bán", "thu 2 ban", "th3.5 kg", "T4 2025"):
        assert spell_month_abbrev(text) == text

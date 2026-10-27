from __future__ import annotations

import pandas as pd
import pytest

from datamate.config import settings
from datamate.io.profile import (
    classify_value_kind,
    profile_business,
    profile_table,
    render_business,
    render_card,
)
from datamate.io.raw import DroppedRow, RawTable, SourceRef, load_business_cached


@pytest.mark.parametrize(
    ("raw_value", "expected_kind"),
    [
        # datetime_iso
        ("2025-02-01 06:00:21", "datetime_iso"),
        ("2025-02-01", "datetime_iso"),
        # date_dmy
        ("01/02/2025", "date_dmy"),
        ("1/2/2025", "date_dmy"),
        # date_dm
        ("1/1", "date_dm"),
        ("01/02", "date_dm"),
        # date_thg
        ("1 thg 6 2025", "date_thg"),
        ("15 thg 12 2026", "date_thg"),
        # int_serial_range (pure int 20000..80000)
        ("45000", "int_serial_range"),
        ("45689", "int_serial_range"),
        ("20000", "int_serial_range"),
        # money_text
        ("45,000", "money_text"),
        ("55.000đ", "money_text"),
        ("25k", "money_text"),
        ("1tr440", "money_text"),
        ("16,56tr", "money_text"),
        ("1.200.000", "money_text"),
        ("16 076", "money_text"),
        ("24,000.0000", "money_text"),
        # number
        ("12", "number"),
        ("18.6", "number"),
        ("150000", "number"),
        # qty_with_unit
        ("3 chai", "qty_with_unit"),
        ("15 vỉ", "qty_with_unit"),
        ("2 xe 4m3", "qty_with_unit"),
        # id_like
        ("ORD000001", "id_like"),
        ("SP0001", "id_like"),
        ("KH0043", "id_like"),
        # phone
        ("0912345678", "phone"),
        ("0971 618 698", "phone"),
        # bool
        ("True", "bool"),
        ("False", "bool"),
        # excel_error
        ("#VALUE!", "excel_error"),
        ("#N/A", "excel_error"),
        # text
        ("Cà phê sữa (M)", "text"),
        ("Khách lẻ", "text"),
    ],
)
def test_classify_value_kind(raw_value: str, expected_kind: str) -> None:
    assert classify_value_kind(raw_value) == expected_kind


def test_synthetic_raw_table_profile_and_determinism() -> None:
    cols = ["Mã đơn", "Thời gian", "Thành tiền", "Khách hàng"]
    df = pd.DataFrame(
        [
            {
                "Mã đơn": "ORD0001",
                "Thời gian": "01/02/2025",
                "Thành tiền": "55.000đ",
                "Khách hàng": "  Khách lẻ ",
                "__file": "SoBanHang.xlsx",
                "__sheet": "Sheet1",
                "__row": 3,
            },
            # Exact duplicate on data columns (different __row)
            {
                "Mã đơn": "ORD0001",
                "Thời gian": "01/02/2025",
                "Thành tiền": "55.000đ",
                "Khách hàng": "  Khách lẻ ",
                "__file": "SoBanHang.xlsx",
                "__sheet": "Sheet1",
                "__row": 4,
            },
            {
                "Mã đơn": "ORD0002",
                "Thời gian": "45689",
                "Thành tiền": "#VALUE!",
                "Khách hàng": None,
                "__file": "SoBanHang.xlsx",
                "__sheet": "Sheet1",
                "__row": 5,
            },
        ]
    )
    table = RawTable(
        name="SoBanHang.xlsx!Sheet1",
        kind="xlsx",
        sources=[
            SourceRef(
                file="SoBanHang.xlsx",
                sheet="Sheet1",
                header_row=2,
                header_groups={"Mã đơn": "Đơn", "Thời gian": "Đơn", "Thành tiền": "Tiền"},
            )
        ],
        columns=cols,
        df=df,
        dropped=[
            DroppedRow(
                file="SoBanHang.xlsx",
                sheet="Sheet1",
                row_no=6,
                reason="summary",
                preview=["Cộng", "110.000đ"],
            )
        ],
        notes=[],
    )

    tp1 = profile_table(table)
    tp2 = profile_table(table)
    card1 = render_card(tp1)
    card2 = render_card(tp2)

    assert card1 == card2
    assert tp1.n_rows == 3
    assert tp1.n_exact_duplicate_rows == 1
    assert tp1.dropped_counts == {"summary": 1}
    assert "exact_duplicate_rows: 1" in card1
    assert "[group: Đơn]" in card1
    assert "untrimmed=2" in card1
    assert "excel_error=1" in card1


def test_real_train_biz_106_profile() -> None:
    if not (settings.data_root / "train").exists():
        pytest.skip("train split absent")

    biz_raw = load_business_cached("train", "biz_106")
    bp = profile_business(biz_raw)
    rendered = render_business(bp, include_readme=True)

    assert "Mã hóa đơn" in rendered

    hd_prof = next(t for t in bp.tables if "DanhSachHoaDon.xlsx" in t.name)
    tg_col = next(c for c in hd_prof.columns if c.name == "Thời gian")
    kind_names = {k for k, _ in tg_col.kinds}
    assert "date_dmy" in kind_names
    assert "int_serial_range" in kind_names


def test_column_with_12_distinct_values_renders_all_12_and_4_kinds() -> None:
    status_vals = [f"status_{i:02d}" for i in range(1, 13)]
    date_vals = [
        "2025-02-01",
        "01/02/2025",
        "1/2",
        "1 thg 2 2025",
    ] * 3

    df = pd.DataFrame(
        [
            {
                "Trạng thái": s,
                "Ngày": d,
                "__file": "test.xlsx",
                "__sheet": "Sheet1",
                "__row": idx + 2,
            }
            for idx, (s, d) in enumerate(zip(status_vals, date_vals, strict=True))
        ]
    )
    table = RawTable(
        name="test.xlsx!Sheet1",
        kind="xlsx",
        sources=[SourceRef(file="test.xlsx", sheet="Sheet1", header_row=1)],
        columns=["Trạng thái", "Ngày"],
        df=df,
    )

    tp = profile_table(table)
    card = render_card(tp)

    assert "| values (12):" in card
    for s in status_vals:
        assert f'"{s}" 1' in card

    date_col = next(c for c in tp.columns if c.name == "Ngày")
    assert len(date_col.kinds) == 4
    assert dict(date_col.kinds) == {
        "date_dm": 0.25,
        "date_dmy": 0.25,
        "date_thg": 0.25,
        "datetime_iso": 0.25,
    }
    for k in ("datetime_iso 0.25", "date_dmy 0.25", "date_dm 0.25", "date_thg 0.25"):
        assert k in card

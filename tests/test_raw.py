from __future__ import annotations

import datetime
import sqlite3
import unicodedata
from pathlib import Path

import openpyxl
import pytest

from datamate.config import settings
from datamate.io.paths import business_dir
from datamate.io.raw import RawTable, cell_to_str, load_business_cached, read_business


def _find_table(tables: list[RawTable], prefix: str) -> RawTable:
    for t in tables:
        if prefix in t.name:
            return t
    raise AssertionError(f"Table matching {prefix!r} not found in {[t.name for t in tables]}")


def test_cell_to_str_conversions() -> None:
    assert cell_to_str(None) is None
    assert cell_to_str("") is None
    assert cell_to_str("   \t\n ") is None

    assert cell_to_str(datetime.datetime(2025, 2, 1, 6, 0, 21, 987654)) == "2025-02-01 06:00:21"
    assert cell_to_str(datetime.date(2025, 2, 1)) == "2025-02-01"
    assert cell_to_str(datetime.time(6, 3, 42)) == "06:03:42"

    assert cell_to_str(45717) == "45717"
    assert cell_to_str(115000.0) == "115000"
    assert cell_to_str(0.5) == "0.5"
    assert cell_to_str(True) == "True"
    assert cell_to_str(False) == "False"

    # Preserve leading/trailing spaces and Excel errors; normalise NFD -> NFC
    nfd = unicodedata.normalize("NFD", "  Cà phê sữa  ")
    converted = cell_to_str(nfd)
    assert converted == unicodedata.normalize("NFC", "  Cà phê sữa  ")
    assert cell_to_str("#VALUE!") == "#VALUE!"
    assert cell_to_str("#N/A") == "#N/A"


def test_synthetic_xlsx_header_groups_continuation_and_drops(tmp_path: Path) -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "Sheet1"

    ws.append(["BÁO CÁO BÁN HÀNG THÁNG 4"])  # row 1: title
    ws.append([None, None, None, None])  # row 2: empty
    ws.append(["Hàng", "", "Tiền", ""])  # row 3: sparse group row
    ws.append(["Mã đơn", "Tên món", "Đơn giá", "Thành tiền"])  # row 4: real header
    ws.append(["ORD001", "Cà phê đen", "25000", "25000"])  # row 5: data 1
    ws.append([None, "Bạc xỉu", "30000", "30000"])  # row 6: data 2 (sparse continuation)
    ws.append(["ORD002", "Trà đào", "35000", "35000"])  # row 7: data 3
    ws.append(["Tổng ngày 01/04", None, None, "5,606,000"])  # row 8: mid-body summary
    ws.append(["ORD003", "Sinh tố bơ", "45000", "45000"])  # row 9: data 4
    ws.append(["ORD004", "Trà vải", "39000", "39000"])  # row 10: data 5
    ws.append(["Ghi chú: tiền tính bằng đ", None, None, None])  # row 11: trailing summary/note
    ws.append([None, None, None, None])  # row 12: trailing empty
    ws.append(["", "  ", None, None])  # row 13: trailing empty

    xlsx_path = tmp_path / "SoBanHang.xlsx"
    wb.save(xlsx_path)
    wb.close()

    biz_raw = read_business(tmp_path)
    assert len(biz_raw.tables) == 1
    table = biz_raw.tables[0]

    assert len(table.sources) == 1
    src = table.sources[0]
    assert src.header_row == 4
    assert src.header_groups == {
        "Mã đơn": "Hàng",
        "Tên món": "Hàng",
        "Đơn giá": "Tiền",
        "Thành tiền": "Tiền",
    }
    assert table.columns == ["Mã đơn", "Tên món", "Đơn giá", "Thành tiền"]
    assert len(table.df) == 5

    # Continuation row (row 6) is kept with None in "Mã đơn"
    assert table.df["__row"].tolist() == [5, 6, 7, 9, 10]
    assert table.df.iloc[1]["Mã đơn"] is None
    assert table.df.iloc[1]["Tên món"] == "Bạc xỉu"

    dropped_by_row = {d.row_no: d.reason for d in table.dropped}
    assert dropped_by_row[1] == "title_or_filter"
    assert dropped_by_row[2] == "empty"
    assert dropped_by_row[3] == "title_or_filter"
    assert dropped_by_row[8] == "summary"
    assert dropped_by_row[11] == "summary"
    assert 12 not in dropped_by_row
    assert 13 not in dropped_by_row


def test_synthetic_subtotal_label_in_last_column(tmp_path: Path) -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "Sheet1"

    ws.append(["Ngày", "Mã đơn", "Thành tiền", "Ghi chú"])
    ws.append(["08/07/2026", "ORD001", "100,000.0000", None])
    ws.append([None, None, "10,104,800.0000", "Cộng ngày 08/07"])
    ws.append(["09/07/2026", "ORD002", "200,000.0000", None])

    wb.save(tmp_path / "SoChiTiet.xlsx")
    wb.close()

    biz_raw = read_business(tmp_path)
    table = biz_raw.tables[0]
    assert len(table.df) == 2
    assert table.df["__row"].tolist() == [2, 4]
    summary_drops = [d for d in table.dropped if d.reason == "summary"]
    assert len(summary_drops) == 1
    assert summary_drops[0].row_no == 3


def test_summary_vs_3cell_data_rows_diacritics(tmp_path: Path) -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "Sheet1"

    ws.append(["Cột 1", "Cột 2", "Cột 3"])  # row 1: header
    # 3-cell rows that MUST be kept (Công != Cộng, Tông != Tổng)
    ws.append(["12", "bộ", "Công tắc 1 chiều Panasonic"])  # row 2
    ws.append(["1 thg 1 2025", "45k", "Công tắc 2 chiều"])  # row 3
    ws.append(["Tông đơ cắt tóc", "1", "cái"])  # row 4
    # Summary rows that MUST be dropped
    ws.append(["Tong cong", "1.032.648.800đ", None])  # row 5
    ws.append([None, "10,104,800.0000", "Cộng ngày 08/07"])  # row 6
    ws.append(["TỔNG THÁNG", "215381k", None])  # row 7
    ws.append(["Ghi chú: tiền tính bằng đ", None, None])  # row 8
    ws.append(["Tổng cộng", "9999", None])  # row 9

    wb.save(tmp_path / "TonKho.xlsx")
    wb.close()

    biz_raw = read_business(tmp_path)
    table = biz_raw.tables[0]

    assert len(table.df) == 3
    assert table.df["__row"].tolist() == [2, 3, 4]
    assert table.df.iloc[0]["Cột 3"] == "Công tắc 1 chiều Panasonic"
    assert table.df.iloc[1]["Cột 3"] == "Công tắc 2 chiều"
    assert table.df.iloc[2]["Cột 1"] == "Tông đơ cắt tóc"

    summary_drops = [d for d in table.dropped if d.reason == "summary"]
    assert [d.row_no for d in summary_drops] == [5, 6, 7, 8, 9]


def test_synthetic_merge_files_and_separate_sheet(tmp_path: Path) -> None:
    wb1 = openpyxl.Workbook()
    ws1 = wb1.active
    assert ws1 is not None
    ws1.title = "Sheet1"
    ws1.append(["Mã đơn", "Tên món", "Thành tiền"])
    ws1.append(["ORD1", "Cà phê", "25000"])
    wb1.save(tmp_path / "BanHang_T1_2025.xlsx")
    wb1.close()

    wb2 = openpyxl.Workbook()
    ws2 = wb2.active
    assert ws2 is not None
    ws2.title = "Sheet1"
    ws2.append(["Mã đơn", "Tên món", "Thành tiền"])
    ws2.append(["ORD2", "Trà đá", "10000"])
    ws_other = wb2.create_sheet("Ghi tắt")
    ws_other.append(["Viết tắt", "Nghĩa đầy đủ"])
    ws_other.append(["CF", "Cà phê"])
    wb2.save(tmp_path / "BanHang_T10_2025.xlsx")
    wb2.close()

    biz_raw = read_business(tmp_path)
    assert len(biz_raw.tables) == 2

    merged = _find_table(biz_raw.tables, "BanHang_T#_#")
    assert len(merged.sources) == 2
    assert set(merged.df["__file"].tolist()) == {"BanHang_T1_2025.xlsx", "BanHang_T10_2025.xlsx"}
    assert len(merged.df) == 2

    separate = _find_table(biz_raw.tables, "Ghi tắt")
    assert len(separate.sources) == 1
    assert separate.columns == ["Viết tắt", "Nghĩa đầy đủ"]


def test_synthetic_csv_bom_and_sqlite_immutable(tmp_path: Path) -> None:
    csv_path = tmp_path / "orders.csv"
    csv_content = "\ufeffMã đơn;Khách hàng;Thành tiền\nORD01;  Nguyễn Văn A ;50000\nORD02;Trần B;75000;extra_cell\n"
    csv_path.write_bytes(csv_content.encode("utf-8"))

    db_path = tmp_path / "pos.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE orders (id TEXT, total INTEGER)")
    conn.execute("INSERT INTO orders VALUES ('ORD1', 100000)")
    conn.execute("CREATE TABLE items (id TEXT, name TEXT)")
    conn.execute("INSERT INTO items VALUES ('SP1', 'Cà phê')")
    conn.commit()
    conn.close()

    (tmp_path / "README.md").write_text("# Shop README", encoding="utf-8")

    before_files = sorted(p.name for p in tmp_path.iterdir())
    biz_raw = read_business(tmp_path)
    after_files = sorted(p.name for p in tmp_path.iterdir())

    assert before_files == after_files

    status_by_name = {f.name: f.status for f in biz_raw.files}
    assert status_by_name == {
        "orders.csv": "read",
        "pos.sqlite": "read",
        "README.md": "readme",
    }

    csv_table = _find_table(biz_raw.tables, "orders.csv")
    assert csv_table.columns == ["Mã đơn", "Khách hàng", "Thành tiền"]
    assert len(csv_table.df) == 2
    # Leading/trailing spaces preserved on value
    assert csv_table.df.iloc[0]["Khách hàng"] == "  Nguyễn Văn A "
    assert any("ragged_rows=1" in n for n in csv_table.notes)

    sqlite_tables = [t for t in biz_raw.tables if t.kind == "sqlite"]
    assert {t.name for t in sqlite_tables} == {"pos.sqlite!orders", "pos.sqlite!items"}


def test_real_train_biz_104() -> None:
    if not (settings.data_root / "train").exists():
        pytest.skip("train split absent")

    biz_raw = load_business_cached("train", "biz_104")
    table = _find_table(biz_raw.tables, "DonHang_ChiTiet")

    assert len(table.df) == 62451
    summary_drops = [d for d in table.dropped if d.reason == "summary"]
    assert len(summary_drops) == 443

    for col in table.columns:
        assert not any(
            isinstance(v, str) and v.startswith("Tổng ngày") for v in table.df[col]
        ), f"Column {col!r} still has a 'Tổng ngày' cell"


def test_real_train_biz_117_and_dev_004() -> None:
    if (settings.data_root / "train").exists():
        biz_117 = load_business_cached("train", "biz_117")
        ton_kho = _find_table(biz_117.tables, "TonKho")
        assert len(ton_kho.df) == 60
        assert [d for d in ton_kho.dropped if d.reason == "summary"] == []

    if (settings.data_root / "public_test").exists():
        dev_004 = load_business_cached("public_test", "dev_004")
        bang_gia = _find_table(dev_004.tables, "BangGia")
        assert len(bang_gia.df) == 99
        assert [d for d in bang_gia.dropped if d.reason == "summary"] == []


def test_real_train_biz_105() -> None:
    if not (settings.data_root / "train").exists():
        pytest.skip("train split absent")

    biz_raw = load_business_cached("train", "biz_105")
    table = _find_table(biz_raw.tables, "SoBanHang")

    assert len(table.sources) == 17
    assert {s.file for s in table.sources} == {"SoBanHang_1.xlsx", "SoBanHang_2026.xlsx"}
    assert all(s.header_row == 2 for s in table.sources)
    assert all(bool(s.header_groups) for s in table.sources)
    assert len(table.df) == 140205

    summary_drops = [d for d in table.dropped if d.reason == "summary"]
    assert len(summary_drops) == 40


def test_real_train_biz_106() -> None:
    if not (settings.data_root / "train").exists():
        pytest.skip("train split absent")

    biz_raw = load_business_cached("train", "biz_106")

    ct_table = _find_table(biz_raw.tables, "DanhSachHoaDonChiTiet_#")
    assert len(ct_table.sources) == 2
    assert len(ct_table.df) == 129416

    hd_table = _find_table(biz_raw.tables, "DanhSachHoaDon.xlsx")
    assert len(hd_table.sources) == 1
    assert hd_table.sources[0].header_row == 3
    assert len(hd_table.df) == 78847


def test_real_public_test_dev_001() -> None:
    if not (settings.data_root / "public_test").exists():
        pytest.skip("public_test split absent")

    biz_raw = load_business_cached("public_test", "dev_001")
    table = _find_table(biz_raw.tables, "So_chi_tiet_ban_hang")

    assert len(table.df) == 90297
    summary_drops = [d for d in table.dropped if d.reason == "summary"]
    assert len(summary_drops) == 554


def test_real_train_biz_123_sqlite_readonly() -> None:
    if not (settings.data_root / "train").exists():
        pytest.skip("train split absent")

    bdir = business_dir("train", "biz_123")
    before = sorted(p.name for p in bdir.iterdir())
    biz_raw = read_business(bdir)
    after = sorted(p.name for p in bdir.iterdir())

    assert before == after
    sqlite_tables = [t for t in biz_raw.tables if t.kind == "sqlite"]
    assert len(sqlite_tables) > 0
    assert all(t.name.startswith("pos.sqlite!") for t in sqlite_tables)


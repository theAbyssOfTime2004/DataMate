from __future__ import annotations

import datetime
from decimal import Decimal
from pathlib import Path

import duckdb
import openpyxl
import pytest

from datamate.etl import MappingError, build_canonical, write_duckdb
from datamate.io.raw import read_business
from datamate.mapping import Mapping, validate_mapping


def test_flat_sheet_ffill_dedupe_return_date_and_year_hint(tmp_path: Path) -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "T03-2025"

    ws.append(["Mã đơn", "Ngày", "Khách hàng", "Trạng thái", "Tên món", "SL", "Thành tiền", "Ghi chú dòng"])
    # Order ORD001 spread over 3 lines (lines 2-3 have empty order_id, date, customer, status)
    # Plus an exact pre-fill duplicate of line 2 in the SAME order (dropped)
    ws.append(["ORD001", "15/3", "Nguyễn Văn A", "Xong", "Cà phê sữa", "2 ly", "60.000đ", "ít đá"])
    ws.append([None, None, None, None, "Bánh ngọt", "1 cái", "35k", None])
    ws.append([None, None, None, None, "Bánh ngọt", "1 cái", "35k", None])  # exact duplicate -> drop
    ws.append([None, None, None, None, "Trà đào", "1 ly", "40.000đ", None])

    # Another order ORD002 with the same item content as ORD001's line (must be KEPT)
    ws.append(["ORD002", "16/3", "Khách lẻ", "xong ", "Bánh ngọt", "1 cái", "35k", None])

    # Return order ORD003: positive line on 17/03/2025, negative return line on 20/03/2025
    ws.append(["ORD003", "17/03/2025", "Trần Thị B", "Xong", "Sinh tố bơ", "2 ly", "90.000đ", None])
    ws.append(["ORD003", "20/03/2025", "Trần Thị B", "Xong", "Sinh tố bơ", "-1 ly", "-45.000đ", None])

    # Order ORD004 with an unmapped status
    ws.append(["ORD004", "18/03/2025", "Khách lẻ", "chờ xác nhận", "Cà phê đen", "1 ly", "25.000đ", None])

    wb.save(tmp_path / "SoBanHang.xlsx")
    wb.close()

    mapping = Mapping.model_validate(
        {
            "data_end_date": "2025-03-31",
            "customer_identity": "by_name",
            "walk_in_values": ["Khách lẻ", "lẻ"],
            "return_rule": "negative_qty",
            "status_map": {
                "xong": "completed",
                "hủy": "cancelled",
                "trả hàng": "returned",
                "đang giao": "shipping",
            },
            "tables": {
                "orders": {
                    "derived_from": "order_lines",
                    "columns": {
                        "order_id": "Mã đơn",
                        "order_date": "Ngày",
                        "customer_name": "Khách hàng",
                        "status": "Trạng thái",
                    },
                },
                "order_lines": {
                    "sources": [{"file": "SoBanHang.xlsx", "sheet": "*", "header_row": 1}],
                    "columns": {
                        "order_id": "Mã đơn",
                        "product_name": "Tên món",
                        "qty": "SL",
                        "line_amount": "Thành tiền",
                    },
                },
            },
        }
    )

    raw = read_business(tmp_path)
    canon = build_canonical(raw, mapping)

    # 8 raw lines -> 1 exact duplicate in ORD001 dropped -> 7 lines kept
    assert sum(canon.report.dedupe_counts.values()) == 1
    assert len(canon.order_lines) == 7
    assert len(canon.orders) == 4

    # Forward-fill counts: 3 lines had empty order_id/order_date/customer_name/status
    assert canon.report.fill_counts["order_id"] == 3
    assert canon.report.fill_counts["order_date"] == 3
    assert canon.report.fill_counts["customer_name"] == 3
    assert canon.report.fill_counts["status"] == 3

    orders_by_id = canon.orders.set_index("order_id").to_dict(orient="index")

    # ORD001: "15/3" on sheet "T03-2025" -> 2025-03-15
    assert orders_by_id["ORD001"]["order_date"] == datetime.date(2025, 3, 15)
    assert orders_by_id["ORD001"]["status"] == "completed"
    assert orders_by_id["ORD001"]["is_walk_in"] is False
    assert orders_by_id["ORD001"]["customer_key"] == "nguyễn văn a"

    # ORD002: walk-in customer, "xong " mapped to "completed"
    assert orders_by_id["ORD002"]["status"] == "completed"
    assert orders_by_id["ORD002"]["is_walk_in"] is True
    assert orders_by_id["ORD002"]["customer_key"] is None

    # ORD003: negative-qty line triggers has_return=True and overrides status to "returned";
    # order_date comes from the positive line (2025-03-17), NOT the return line (2025-03-20)
    assert orders_by_id["ORD003"]["has_return"] is True
    assert orders_by_id["ORD003"]["status"] == "returned"
    assert orders_by_id["ORD003"]["order_date"] == datetime.date(2025, 3, 17)
    assert canon.report.status_overridden_by_return == 1

    # ORD004: unmapped status "chờ xác nhận" reported
    assert orders_by_id["ORD004"]["status"] is None
    assert canon.report.unmapped_statuses == {"chờ xác nhận": 1}

    # orders.status parse stat for derived orders is counted per order (4 orders, 3 mapped)
    st_stat = canon.report.parse_stats["orders.status"]
    assert st_stat.non_null == 4
    assert st_stat.parsed == 3
    assert st_stat.examples == ["chờ xác nhận"]


def test_separate_orders_and_lines_subtotal_and_orphan(tmp_path: Path) -> None:
    (tmp_path / "orders.csv").write_text(
        "order_id,order_date,status,customer_id,subtotal\n"
        "ORD01,2025-05-01,completed,KH01,100000\n"
        "ORD02,2025-05-02,completed,KH02,50000\n",
        encoding="utf-8",
    )
    (tmp_path / "lines.csv").write_text(
        "order_id,product_id,qty,line_amount\n"
        "ORD01,SP01,1,60000\n"
        "ORD01,SP02,2,40000\n"
        "ORD02,SP01,1,50000\n"
        "ORD_UNKNOWN,SP01,1,20000\n",
        encoding="utf-8",
    )
    (tmp_path / "products.csv").write_text(
        "product_id,product_name,base_unit\n"
        "SP01,Cà phê đen,ly\n"
        "SP02,Bạc xỉu,ly\n",
        encoding="utf-8",
    )

    mapping = Mapping.model_validate(
        {
            "data_end_date": "2025-05-31",
            "customer_identity": "by_code",
            "walk_in_values": ["Khách lẻ"],
            "return_rule": "status",
            "status_map": {"completed": "completed"},
            "tables": {
                "orders": {
                    "sources": [{"file": "orders.csv", "sheet": None, "header_row": 1}],
                    "columns": {
                        "order_id": "order_id",
                        "order_date": "order_date",
                        "status": "status",
                        "customer_id": "customer_id",
                        "order_subtotal": "subtotal",
                    },
                },
                "order_lines": {
                    "sources": [{"file": "lines.csv", "sheet": None, "header_row": 1}],
                    "join": {
                        "product_name": {"table": "products", "on": {"product_id": "product_id"}},
                    },
                    "columns": {
                        "order_id": "order_id",
                        "product_id": "product_id",
                        "qty": "qty",
                        "line_amount": "line_amount",
                    },
                },
                "products": {
                    "sources": [{"file": "products.csv", "sheet": None, "header_row": 1}],
                    "columns": {
                        "product_id": "product_id",
                        "product_name": "product_name",
                        "base_unit": "base_unit",
                    },
                },
            },
        }
    )

    raw = read_business(tmp_path)
    canon = build_canonical(raw, mapping)

    assert canon.report.integrity.subtotal_checked_orders == 2
    assert canon.report.integrity.subtotal_matched_orders == 2
    assert canon.report.integrity.subtotal_match_share == 1.0
    assert canon.report.integrity.orphan_line_orders == 1

    # Join populated product_name on order_lines from products
    assert canon.order_lines.iloc[0]["product_name"] == "Cà phê đen"
    assert canon.order_lines.iloc[1]["product_name"] == "Bạc xỉu"


def test_mapping_error_lists_available_columns(tmp_path: Path) -> None:
    (tmp_path / "orders.csv").write_text(
        "Mã đơn,Ngày đặt,Thành tiền\nORD1,01/05/2025,50000\n",
        encoding="utf-8",
    )
    mapping = Mapping.model_validate(
        {
            "customer_identity": "by_name",
            "return_rule": "status",
            "status_map": {"xong": "completed"},
            "tables": {
                "orders": {
                    "derived_from": "order_lines",
                    "columns": {"order_id": "Mã đơn"},
                },
                "order_lines": {
                    "sources": [{"file": "orders.csv", "sheet": None, "header_row": 1}],
                    "columns": {
                        "order_id": "Mã đơn",
                        "line_amount": "Cột_Không_Tồn_Tại",
                    },
                },
            },
        }
    )

    raw = read_business(tmp_path)
    with pytest.raises(MappingError) as exc_info:
        build_canonical(raw, mapping)

    msg = str(exc_info.value)
    assert "Cột_Không_Tồn_Tại" in msg
    assert "Mã đơn" in msg
    assert "Ngày đặt" in msg
    assert "Thành tiền" in msg


def test_write_duckdb_describe_types_and_roundtrip(tmp_path: Path) -> None:
    (tmp_path / "orders.csv").write_text(
        "order_id,order_date,order_ts,status,product_name,qty,line_amount\n"
        "ORD01,2025-05-01,2025-05-01 08:30:00,xong,Cà phê,2.5,50000\n",
        encoding="utf-8",
    )
    mapping = Mapping.model_validate(
        {
            "customer_identity": "by_name",
            "return_rule": "status",
            "status_map": {"xong": "completed"},
            "tables": {
                "orders": {
                    "derived_from": "order_lines",
                    "columns": {
                        "order_id": "order_id",
                        "order_date": "order_date",
                        "order_ts": "order_ts",
                        "status": "status",
                    },
                },
                "order_lines": {
                    "sources": [{"file": "orders.csv", "sheet": None, "header_row": 1}],
                    "columns": {
                        "order_id": "order_id",
                        "product_name": "product_name",
                        "qty": "qty",
                        "line_amount": "line_amount",
                    },
                },
            },
        }
    )

    raw = read_business(tmp_path)
    canon = build_canonical(raw, mapping)

    db_file = tmp_path / "canonical.duckdb"
    write_duckdb(canon, db_file)

    conn = duckdb.connect(str(db_file), read_only=True)
    try:
        orders_desc = {
            r[0]: r[1] for r in conn.execute("DESCRIBE orders").fetchall()
        }
        lines_desc = {
            r[0]: r[1] for r in conn.execute("DESCRIBE order_lines").fetchall()
        }

        assert orders_desc["order_date"] == "DATE"
        assert orders_desc["order_ts"] == "TIMESTAMP"
        assert orders_desc["order_subtotal"] == "BIGINT"
        assert orders_desc["has_return"] == "BOOLEAN"

        assert lines_desc["qty"] == "DECIMAL(18,4)"
        assert lines_desc["qty_base"] == "DECIMAL(18,4)"
        assert lines_desc["line_amount"] == "BIGINT"
        assert lines_desc["unit_price"] == "BIGINT"

        row_o = conn.execute("SELECT order_id, order_date, order_ts, status FROM orders").fetchone()
        assert row_o == (
            "ORD01",
            datetime.date(2025, 5, 1),
            datetime.datetime(2025, 5, 1, 8, 30, 0),
            "completed",
        )

        row_l = conn.execute("SELECT order_id, qty, line_amount FROM order_lines").fetchone()
        assert row_l == ("ORD01", Decimal("2.5000"), 50000)
    finally:
        conn.close()


def test_write_duckdb_wide_decimal_qty(tmp_path: Path) -> None:
    import pandas as pd

    from datamate.etl import CanonicalData, EtlReport
    from datamate.etl.schema import (
        CUSTOMERS_COLUMNS,
        ORDER_LINES_COLUMNS,
        ORDERS_COLUMNS,
        PRODUCTS_COLUMNS,
        UNIT_CONVERSIONS_COLUMNS,
    )

    def _empty(cols: tuple[str, ...]) -> pd.DataFrame:
        return pd.DataFrame({c: pd.Series(dtype=object) for c in cols})

    line_rows: list[dict[str, object]] = []
    for idx, qty in enumerate((Decimal(1), Decimal(10), Decimal(1000)), start=1):
        rec: dict[str, object] = {c: None for c in ORDER_LINES_COLUMNS}
        rec["order_id"] = f"o{idx}"
        rec["qty"] = qty
        rec["line_no"] = 1
        rec["src_row"] = idx
        rec["is_return_line"] = False
        line_rows.append(rec)

    data = CanonicalData(
        orders=_empty(ORDERS_COLUMNS),
        order_lines=pd.DataFrame(line_rows),
        products=_empty(PRODUCTS_COLUMNS),
        unit_conversions=_empty(UNIT_CONVERSIONS_COLUMNS),
        customers=_empty(CUSTOMERS_COLUMNS),
        report=EtlReport(),
    )
    db_file = tmp_path / "qty.duckdb"
    write_duckdb(data, db_file)

    conn = duckdb.connect(str(db_file), read_only=True)
    try:
        got = [
            row[0]
            for row in conn.execute(
                "SELECT qty FROM order_lines ORDER BY src_row"
            ).fetchall()
        ]
    finally:
        conn.close()
    assert got == [Decimal("1.0000"), Decimal("10.0000"), Decimal("1000.0000")]


def test_ledger_without_status_defaults_all_orders_completed(tmp_path: Path) -> None:
    (tmp_path / "ledger.csv").write_text(
        "Ngày,Khách,Món,SL,Tiền\n"
        "01/08/2026,An,Cà phê,1,10000\n"
        "02/08/2026,Bình,Trà,-1,5000\n",
        encoding="utf-8",
    )
    mapping = Mapping.model_validate(
        {
            "customer_identity": "by_name",
            "return_rule": "negative_qty",
            "status_map": {},
            "tables": {
                "orders": {
                    "derived_from": "order_lines",
                    "columns": {
                        "order_date": "Ngày",
                        "customer_name": "Khách",
                    },
                },
                "order_lines": {
                    "sources": [{"file": "ledger.csv", "sheet": None, "header_row": 1}],
                    "columns": {
                        "product_name": "Món",
                        "qty": "SL",
                        "line_amount": "Tiền",
                    },
                },
            },
        }
    )
    canon = build_canonical(read_business(tmp_path), mapping)
    assert "no status column: all orders completed" in canon.report.notes
    orders = {row["order_id"]: row for row in canon.orders.to_dict("records")}
    assert all(row["status_raw"] is None for row in orders.values())
    completed = [row for row in orders.values() if row["status"] == "completed"]
    returned = [row for row in orders.values() if row["status"] == "returned"]
    assert len(completed) == 1
    assert len(returned) == 1  # negative qty overrides default completed
    issues, _canon2 = validate_mapping(read_business(tmp_path), mapping)
    assert any(
        iss.severity == "warning" and "no status column" in iss.message for iss in issues
    )
    assert not any(iss.severity == "error" and "status" in iss.message for iss in issues)


def test_no_order_id_is_one_order_per_row_and_skips_dedupe(tmp_path: Path) -> None:
    (tmp_path / "ledger.csv").write_text(
        "Ngày,Trạng thái,Khách,Món,SL,Tiền\n"
        "01/08/2026,xong,An,Cà phê,1,10000\n"
        "01/08/2026,xong,An,Cà phê,1,10000\n",
        encoding="utf-8",
    )
    mapping = Mapping.model_validate(
        {
            "customer_identity": "by_name",
            "return_rule": "status",
            "status_map": {"xong": "completed"},
            "tables": {
                "orders": {
                    "derived_from": "order_lines",
                    "columns": {
                        "order_date": "Ngày",
                        "status": "Trạng thái",
                        "customer_name": "Khách",
                    },
                },
                "order_lines": {
                    "sources": [{"file": "ledger.csv", "sheet": None, "header_row": 1}],
                    "columns": {
                        "product_name": "Món",
                        "qty": "SL",
                        "line_amount": "Tiền",
                    },
                },
            },
        }
    )
    canon = build_canonical(read_business(tmp_path), mapping)
    assert "no order id: one order per row" in canon.report.notes
    assert sum(canon.report.dedupe_counts.values()) == 0
    assert len(canon.order_lines) == 2
    assert len(canon.orders) == 2
    ids = set(canon.orders["order_id"].tolist())
    assert ids == {"ledger.csv||2", "ledger.csv||3"}


def test_mostly_blank_status_column_only_marks_exceptions(tmp_path: Path) -> None:
    (tmp_path / "ledger.csv").write_text(
        "Ngày,Ghi chú,Món,Tiền\n"
        "01/08/2026,,Cà phê,10000\n"
        "02/08/2026,,Trà,5000\n"
        "03/08/2026,khách trả lại,Bánh,7000\n",
        encoding="utf-8",
    )
    mapping = Mapping.model_validate(
        {
            "customer_identity": "by_name",
            "return_rule": "status",
            "status_map": {"khách trả lại": "returned"},
            "tables": {
                "orders": {
                    "derived_from": "order_lines",
                    "columns": {"order_date": "Ngày", "status": "Ghi chú"},
                },
                "order_lines": {
                    "sources": [{"file": "ledger.csv", "sheet": None, "header_row": 1}],
                    "columns": {"product_name": "Món", "line_amount": "Tiền"},
                },
            },
        }
    )
    canon = build_canonical(read_business(tmp_path), mapping)
    assert sorted(canon.orders["status"]) == ["completed", "completed", "returned"]
    assert any("mostly blank" in note for note in canon.report.notes)

from __future__ import annotations

import datetime
import math
import re
from pathlib import Path
from typing import Any

import openpyxl

from datamate.config import PROJECT_ROOT, settings
from datamate.io.profile import profile_business
from datamate.io.raw import read_business
from datamate.llm.client import UsageEntry
from datamate.mapping import Mapping, infer_mapping, validate_mapping


class FakeClient:
    def __init__(self, scripted_answers: dict[str, list[dict[str, Any]]]) -> None:
        self.scripted: dict[str, list[dict[str, Any]]] = {
            k: list(v) for k, v in scripted_answers.items()
        }
        self.calls: list[dict[str, Any]] = []
        self.usage_log: list[UsageEntry] = []

    def count_tokens(self, text: str) -> int:
        return math.ceil(len(text) / 2.5)

    def chat_json(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        schema_name: str,
        max_tokens: int | None = None,
        required_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        user_prompt = messages[-1]["content"]
        ptoks = self.count_tokens(user_prompt)
        self.calls.append(
            {
                "schema_name": schema_name,
                "schema": schema,
                "prompt": user_prompt,
                "prompt_tokens": ptoks,
                "max_tokens": max_tokens,
                "required_keys": required_keys,
            }
        )
        queue = self.scripted.get(schema_name, [])
        if not queue:
            raise AssertionError(f"No scripted answer left for {schema_name!r}")
        ans = queue.pop(0)
        self.usage_log.append(
            UsageEntry(
                cache_hit=False,
                prompt_tokens=ptoks,
                completion_tokens=20,
                latency_s=0.01,
            )
        )
        return ans


def _write_two_table_shop(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "# Dữ liệu shop\n\n## Quy ước\n- Dữ liệu có đến hết 15/08/2026.\n",
        encoding="utf-8",
    )

    wb_ord = openpyxl.Workbook()
    ws_ord = wb_ord.active
    assert ws_ord is not None
    ws_ord.title = "Sheet1"
    ws_ord.append(["BÁO CÁO HÓA ĐƠN"])  # row 1: title
    ws_ord.append([None, None, None, None, None])  # row 2: empty
    ws_ord.append(["Mã HĐ", "Ngày bán", "Khách hàng", "Trạng thái", "Tổng tiền hàng"])  # row 3: header
    ws_ord.append(["ORD01", "01/08/2026", "Khách lẻ", "Đã xong", "110.000đ"])
    ws_ord.append(["ORD02", "02/08/2026", "Nguyễn Văn A", "Đã hủy", "50.000đ"])
    wb_ord.save(tmp_path / "HoaDon.xlsx")
    wb_ord.close()

    wb_ct = openpyxl.Workbook()
    ws_ct = wb_ct.active
    assert ws_ct is not None
    ws_ct.title = "Sheet1"
    ws_ct.append(["Mã HĐ", "Tên món", "SL", "Đơn giá", "Thành tiền"])  # row 1: header
    ws_ct.append(["ORD01", "Cà phê sữa", "2", "30.000đ", "60.000đ"])
    ws_ct.append(["ORD01", "Bánh ngọt", "1", "50.000đ", "50.000đ"])
    ws_ct.append(["ORD02", "Trà đào", "1", "50.000đ", "50.000đ"])
    wb_ct.save(tmp_path / "ChiTiet.xlsx")
    wb_ct.close()


def test_infer_mapping_from_scripted_answers_and_step_b_enum(tmp_path: Path) -> None:
    _write_two_table_shop(tmp_path)
    raw = read_business(tmp_path)
    profile = profile_business(raw)

    all_ord_fields = {
        "order_id": "Mã HĐ",
        "order_date": "Ngày bán",
        "order_ts": None,
        "status": "Trạng thái",
        "customer_id": None,
        "customer_name": "Khách hàng",
        "payment_method": None,
        "channel": None,
        "employee_id": None,
        "order_subtotal": "Tổng tiền hàng",
        "order_discount": None,
        "shipping_fee": None,
        "order_payable": None,
    }
    all_line_fields = {
        "order_id": "Mã HĐ",
        "line_no": None,
        "product_id": None,
        "product_name": "Tên món",
        "qty": "SL",
        "unit": None,
        "qty_base": None,
        "unit_price": "Đơn giá",
        "line_discount": None,
        "line_amount": "Thành tiền",
        "cost_per_unit": None,
    }

    fake = FakeClient(
        {
            "step_a_table_roles": [
                {
                    "tables": [
                        {"table": "ChiTiet.xlsx!Sheet1", "entity": "order_lines"},
                        {"table": "HoaDon.xlsx!Sheet1", "entity": "orders"},
                    ],
                    "orders_source": "own_table",
                }
            ],
            "step_b_order_lines": [all_line_fields],
            "step_b_orders": [all_ord_fields],
            "step_c_status_map": [
                {
                    "Đã hủy": "cancelled",
                    "Đã xong": "completed",
                }
            ],
            "step_d_customers_returns": [
                {
                    "walk_in_values": ["Khách lẻ"],
                    "return_rule": "status",
                }
            ],
        }
    )

    res = infer_mapping(raw, profile, fake)
    assert res.errors == []
    assert res.mapping is not None
    m = res.mapping

    # SourceSpec and header_row come from P2 (HoaDon has header_row=3), not LLM
    ord_src = m.tables["orders"].sources[0]
    assert ord_src.file == "HoaDon.xlsx"
    assert ord_src.sheet == "Sheet1"
    assert ord_src.header_row == 3

    line_src = m.tables["order_lines"].sources[0]
    assert line_src.file == "ChiTiet.xlsx"
    assert line_src.header_row == 1

    # data_end_date extracted from README regex; customer_identity = by_name
    assert m.data_end_date == datetime.date(2026, 8, 15)
    assert m.customer_identity == "by_name"
    assert m.walk_in_values == ["Khách lẻ"]
    assert m.return_rule == "status"

    assert all(c["max_tokens"] == 512 for c in fake.calls)
    step_b_line_prompt = next(
        c["prompt"] for c in fake.calls if c["schema_name"] == "step_b_order_lines"
    )
    assert "Set `order_id` to null" in step_b_line_prompt
    step_c_prompt = next(c["prompt"] for c in fake.calls if c["schema_name"] == "step_c_status_map")
    assert "không giao được" in step_c_prompt
    assert "trả một phần" in step_c_prompt

    # Verify Step B schema enum contains exactly the table's columns + None
    step_b_line_call = next(c for c in fake.calls if c["schema_name"] == "step_b_order_lines")
    enum_vals = step_b_line_call["schema"]["properties"]["line_amount"]["enum"]
    assert enum_vals == ["Mã HĐ", "Tên món", "SL", "Đơn giá", "Thành tiền", None]
    line_props = step_b_line_call["schema"]["properties"]
    assert next(iter(line_props)) == "line_amount"
    assert step_b_line_call["required_keys"] == ["line_amount"]
    step_b_ord_call = next(c for c in fake.calls if c["schema_name"] == "step_b_orders")
    assert list(step_b_ord_call["schema"]["properties"])[:2] == ["order_id", "order_date"]
    assert "status" in step_b_ord_call["schema"]["properties"]
    assert step_b_ord_call["required_keys"] == ["order_id", "order_date"]
    assert "Map `status` only to a column that records the order state" in step_b_ord_call["prompt"]
    assert res.canonical is not None
    _issues_again, canon_again = validate_mapping(raw, m)
    assert canon_again is res.canonical


def test_infer_mapping_subtotal_error_triggers_step_b_repair(tmp_path: Path) -> None:
    _write_two_table_shop(tmp_path)
    raw = read_business(tmp_path)
    profile = profile_business(raw)

    all_ord_fields = {
        "order_id": "Mã HĐ",
        "order_date": "Ngày bán",
        "order_ts": None,
        "status": "Trạng thái",
        "customer_id": None,
        "customer_name": "Khách hàng",
        "payment_method": None,
        "channel": None,
        "employee_id": None,
        "order_subtotal": "Tổng tiền hàng",
        "order_discount": None,
        "shipping_fee": None,
        "order_payable": None,
    }
    wrong_line_fields = {
        "order_id": "Mã HĐ",
        "line_no": None,
        "product_id": None,
        "product_name": "Tên món",
        "qty": "SL",
        "unit": None,
        "qty_base": None,
        "unit_price": "Đơn giá",
        "line_discount": None,
        # BUG: mapped line_amount to unit price "Đơn giá" instead of "Thành tiền"
        "line_amount": "Đơn giá",
        "cost_per_unit": None,
    }
    fixed_line_fields = {
        **wrong_line_fields,
        "line_amount": "Thành tiền",
    }

    fake = FakeClient(
        {
            "step_a_table_roles": [
                {
                    "tables": [
                        {"table": "ChiTiet.xlsx!Sheet1", "entity": "order_lines"},
                        {"table": "HoaDon.xlsx!Sheet1", "entity": "orders"},
                    ],
                    "orders_source": "own_table",
                }
            ],
            "step_b_order_lines": [wrong_line_fields, fixed_line_fields],
            "step_b_orders": [all_ord_fields],
            "step_c_status_map": [{"Đã hủy": "cancelled", "Đã xong": "completed"}],
            "step_d_customers_returns": [{"walk_in_values": ["Khách lẻ"], "return_rule": "status"}],
        }
    )

    res = infer_mapping(raw, profile, fake)
    assert res.errors == []
    assert res.mapping is not None
    assert res.mapping.tables["order_lines"].columns["line_amount"] == "Thành tiền"

    # Only step_b_order_lines was re-run during repair, and message mentions both suspects
    b_line_calls = [c for c in fake.calls if c["schema_name"] == "step_b_order_lines"]
    assert len(b_line_calls) == 2
    assert "Previous answer:" in b_line_calls[1]["prompt"]
    assert "sum of line_amount" in b_line_calls[1]["prompt"]
    assert "after-discount / with-shipping total" in b_line_calls[1]["prompt"]
    assert len([c for c in fake.calls if c["schema_name"] == "step_b_orders"]) == 1


def test_by_code_shop_walk_in_candidates_use_customer_name_and_filter_codes(tmp_path: Path) -> None:
    (tmp_path / "orders.csv").write_text(
        "Mã đơn,Ngày,Mã KH,Tên KH,Trạng thái,Tên món,SL,Thành tiền\n"
        "ORD01,01/08/2026,KH0001,Nguyễn Văn A,xong,Cà phê,1,30000\n"
        "ORD02,01/08/2026,,Khách lẻ,xong,Trà đá,1,10000\n"
        "ORD03,02/08/2026,KH0002,0912345678,xong,Bạc xỉu,1,35000\n",
        encoding="utf-8",
    )
    raw = read_business(tmp_path)
    profile = profile_business(raw)

    all_ord_fields = {
        "order_id": "Mã đơn",
        "order_date": "Ngày",
        "order_ts": None,
        "status": "Trạng thái",
        "customer_id": "Mã KH",
        "customer_name": "Tên KH",
        "payment_method": None,
        "channel": None,
        "employee_id": None,
        "order_subtotal": None,
        "order_discount": None,
        "shipping_fee": None,
        "order_payable": None,
    }
    all_line_fields = {
        "order_id": "Mã đơn",
        "line_no": None,
        "product_id": None,
        "product_name": "Tên món",
        "qty": "SL",
        "unit": None,
        "qty_base": None,
        "unit_price": None,
        "line_discount": None,
        "line_amount": "Thành tiền",
        "cost_per_unit": None,
    }

    fake = FakeClient(
        {
            "step_a_table_roles": [
                {
                    "tables": [{"table": "orders.csv", "entity": "order_lines"}],
                    "orders_source": "order_lines",
                }
            ],
            "step_b_order_lines": [all_line_fields],
            "step_b_orders": [all_ord_fields],
            "step_c_status_map": [{"xong": "completed"}],
            # Scripted answer includes a KH code -> must be filtered out in code
            "step_d_customers_returns": [
                {"walk_in_values": ["Khách lẻ", "KH0001"], "return_rule": "status"}
            ],
        }
    )

    res = infer_mapping(raw, profile, fake)
    assert res.errors == []
    assert res.mapping is not None
    assert res.mapping.customer_identity == "by_code"
    assert res.mapping.walk_in_values == ["Khách lẻ"]

    step_d_call = next(c for c in fake.calls if c["schema_name"] == "step_d_customers_returns")
    d_enum = step_d_call["schema"]["properties"]["walk_in_values"]["items"]["enum"]
    assert "Khách lẻ" in d_enum
    assert "Nguyễn Văn A" in d_enum
    assert "KH0001" not in d_enum
    assert "0912345678" not in d_enum


def test_by_code_shop_without_customer_name_column_walk_in_empty(tmp_path: Path) -> None:
    (tmp_path / "orders.csv").write_text(
        "Mã đơn,Ngày,Mã KH,Trạng thái,Tên món,SL,Thành tiền\n"
        "ORD01,01/08/2026,KH0001,xong,Cà phê,1,30000\n"
        "ORD02,01/08/2026,,xong,Trà đá,1,10000\n"
        "ORD03,02/08/2026,KH0002,xong,Bạc xỉu,1,35000\n",
        encoding="utf-8",
    )
    raw = read_business(tmp_path)
    profile = profile_business(raw)

    all_ord_fields = {
        "order_id": "Mã đơn",
        "order_date": "Ngày",
        "order_ts": None,
        "status": "Trạng thái",
        "customer_id": "Mã KH",
        "customer_name": None,
        "payment_method": None,
        "channel": None,
        "employee_id": None,
        "order_subtotal": None,
        "order_discount": None,
        "shipping_fee": None,
        "order_payable": None,
    }
    all_line_fields = {
        "order_id": "Mã đơn",
        "line_no": None,
        "product_id": None,
        "product_name": "Tên món",
        "qty": "SL",
        "unit": None,
        "qty_base": None,
        "unit_price": None,
        "line_discount": None,
        "line_amount": "Thành tiền",
        "cost_per_unit": None,
    }

    fake = FakeClient(
        {
            "step_a_table_roles": [
                {
                    "tables": [{"table": "orders.csv", "entity": "order_lines"}],
                    "orders_source": "order_lines",
                }
            ],
            "step_b_order_lines": [all_line_fields],
            "step_b_orders": [all_ord_fields],
            "step_c_status_map": [{"xong": "completed"}],
            # Even if the LLM attempts to invent a walk-in value, code forces walk_in_values == []
            "step_d_customers_returns": [
                {"walk_in_values": ["Invented_Walk_In"], "return_rule": "status"}
            ],
        }
    )

    res = infer_mapping(raw, profile, fake)
    assert res.errors == []
    assert res.mapping is not None
    assert res.mapping.customer_identity == "by_code"
    assert res.mapping.walk_in_values == []

    step_d_call = next(c for c in fake.calls if c["schema_name"] == "step_d_customers_returns")
    assert step_d_call["schema"]["properties"]["walk_in_values"] == {
        "type": "array",
        "maxItems": 0,
    }


def test_derived_orders_line_money_collision_and_subtotal_mismatch_route_to_b_orders(
    tmp_path: Path,
) -> None:
    (tmp_path / "sales.csv").write_text(
        "Mã đơn,Ngày,Trạng thái,Tên món,SL,Đơn giá,Thành tiền,Cột_Sai_Tổng\n"
        "ORD01,01/08/2026,xong,Cà phê,1,30000,30000,999999\n"
        "ORD01,01/08/2026,xong,Bánh,1,20000,20000,999999\n",
        encoding="utf-8",
    )
    raw = read_business(tmp_path)
    profile = profile_business(raw)

    all_line_fields = {
        "order_id": "Mã đơn",
        "line_no": None,
        "product_id": None,
        "product_name": "Tên món",
        "qty": "SL",
        "unit": None,
        "qty_base": None,
        "unit_price": "Đơn giá",
        "line_discount": None,
        "line_amount": "Thành tiền",
        "cost_per_unit": None,
    }
    # Round 0: order_subtotal maps to line-level column "Thành tiền" -> error on B_orders
    bad_ord_1 = {
        "order_id": "Mã đơn",
        "order_date": "Ngày",
        "order_ts": None,
        "status": "Trạng thái",
        "customer_id": None,
        "customer_name": None,
        "payment_method": None,
        "channel": None,
        "employee_id": None,
        "order_subtotal": "Thành tiền",
        "order_discount": None,
        "shipping_fee": None,
        "order_payable": None,
    }
    # Round 1: order_subtotal maps to "Cột_Sai_Tổng" -> subtotal mismatch routed to B_orders
    bad_ord_2 = {
        **bad_ord_1,
        "order_subtotal": "Cột_Sai_Tổng",
    }
    # Round 2: fixed by setting order_subtotal to None
    fixed_ord = {
        **bad_ord_1,
        "order_subtotal": None,
    }

    fake = FakeClient(
        {
            "step_a_table_roles": [
                {
                    "tables": [{"table": "sales.csv", "entity": "order_lines"}],
                    "orders_source": "order_lines",
                }
            ],
            "step_b_order_lines": [all_line_fields],
            "step_b_orders": [bad_ord_1, bad_ord_2, fixed_ord],
            "step_c_status_map": [{"xong": "completed"}],
            "step_d_customers_returns": [{"walk_in_values": [], "return_rule": "status"}],
        }
    )

    res = infer_mapping(raw, profile, fake)
    assert res.errors == []
    assert res.mapping is not None
    assert "order_subtotal" not in res.mapping.tables["orders"].columns

    b_ord_calls = [c for c in fake.calls if c["schema_name"] == "step_b_orders"]
    # Optional order_subtotal is re-asked once. The second answer is still wrong,
    # so the field is dropped instead of a third step-B call.
    assert len(b_ord_calls) == 2
    assert (
        "maps to 'Thành tiền', which is a line-level column (line_amount)"
        in b_ord_calls[1]["prompt"]
    )
    assert any(
        iss.severity == "warning" and iss.message.startswith("dropped order_subtotal:")
        for iss in res.issues
    )
    # step_b_order_lines was never re-run
    assert len([c for c in fake.calls if c["schema_name"] == "step_b_order_lines"]) == 1


def test_step_a_repair_reruns_changed_entities(tmp_path: Path) -> None:
    _write_two_table_shop(tmp_path)
    raw = read_business(tmp_path)
    profile = profile_business(raw)

    # First attempt: Step A wrongly assigns HoaDon as order_lines and ChiTiet as orders,
    # causing >5% orphan/missing join or subtotal mismatch. Or we trigger a Step A error directly
    # via validate_mapping when orders_source="own_table" has wrong table roles.
    # Let's add a 3rd table "KhachHang.xlsx" with no matching order_id so when Step A picks
    # KhachHang as orders (with order_id mapped to "Mã HĐ" which doesn't match ORD01/ORD02),
    # or let's test Step A repair when validate_mapping returns an Issue(step="A", ...).
    wb_bad = openpyxl.Workbook()
    ws_bad = wb_bad.active
    assert ws_bad is not None
    ws_bad.title = "Sheet1"
    ws_bad.append(["Mã HĐ", "Ngày bán", "Trạng thái"])
    ws_bad.append(["OTHER_1", "01/08/2026", "Đã xong"])
    wb_bad.save(tmp_path / "BangPhu.xlsx")
    wb_bad.close()

    raw = read_business(tmp_path)
    profile = profile_business(raw)

    # We can also test Step A repair by having Step B map a column that triggers MappingError
    # or by patching validate_mapping once, or by having Step A return an invalid role first!
    # Wait: if validate_mapping returns Issue(step="A", severity="error", ...), infer_mapping
    # re-runs Step A with the problem text and then re-runs B/C/D for entities whose table changed.
    all_ord_fields = {
        "order_id": "Mã HĐ",
        "order_date": "Ngày bán",
        "order_ts": None,
        "status": "Trạng thái",
        "customer_id": None,
        "customer_name": "Khách hàng",
        "payment_method": None,
        "channel": None,
        "employee_id": None,
        "order_subtotal": "Tổng tiền hàng",
        "order_discount": None,
        "shipping_fee": None,
        "order_payable": None,
    }
    bad_ord_fields_on_bang_phu = {
        **all_ord_fields,
        "customer_name": None,
        "order_subtotal": None,
    }
    all_line_fields = {
        "order_id": "Mã HĐ",
        "line_no": None,
        "product_id": None,
        "product_name": "Tên món",
        "qty": "SL",
        "unit": None,
        "qty_base": None,
        "unit_price": "Đơn giá",
        "line_discount": None,
        "line_amount": "Thành tiền",
        "cost_per_unit": None,
    }

    import datamate.mapping.infer as infer_mod
    from datamate.mapping.validate import Issue

    orig_validate = infer_mod.validate_mapping
    call_count = 0

    def patched_validate(r: Any, m: Mapping) -> tuple[list[Issue], Any]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return (
                [
                    Issue(
                        step="A",
                        severity="error",
                        message="Table BangPhu.xlsx!Sheet1 is not the main orders table.",
                    )
                ],
                None,
            )
        return orig_validate(r, m)

    infer_mod.validate_mapping = patched_validate
    try:
        fake = FakeClient(
            {
                "step_a_table_roles": [
                    {
                        "tables": [
                            {"table": "ChiTiet.xlsx!Sheet1", "entity": "order_lines"},
                            {"table": "BangPhu.xlsx!Sheet1", "entity": "orders"},
                        ],
                        "orders_source": "own_table",
                    },
                    {
                        "tables": [
                            {"table": "ChiTiet.xlsx!Sheet1", "entity": "order_lines"},
                            {"table": "HoaDon.xlsx!Sheet1", "entity": "orders"},
                        ],
                        "orders_source": "own_table",
                    },
                ],
                "step_b_order_lines": [all_line_fields],
                "step_b_orders": [bad_ord_fields_on_bang_phu, all_ord_fields],
                "step_c_status_map": [
                    {"Đã xong": "completed"},
                    {"Đã hủy": "cancelled", "Đã xong": "completed"},
                ],
                "step_d_customers_returns": [
                    {"walk_in_values": [], "return_rule": "status"},
                    {"walk_in_values": ["Khách lẻ"], "return_rule": "status"},
                ],
            }
        )

        res = infer_mapping(raw, profile, fake)
        assert res.errors == []
        assert res.mapping is not None
        assert res.mapping.tables["orders"].sources[0].file == "HoaDon.xlsx"

        # Step A was called twice; second call has the problem text
        a_calls = [c for c in fake.calls if c["schema_name"] == "step_a_table_roles"]
        assert len(a_calls) == 2
        assert "Table BangPhu.xlsx!Sheet1 is not the main orders table." in a_calls[1]["prompt"]

        # order_lines table did NOT change -> step_b_order_lines called only once
        assert len([c for c in fake.calls if c["schema_name"] == "step_b_order_lines"]) == 1
        # orders table DID change -> step_b_orders, step_c, step_d re-run
        assert len([c for c in fake.calls if c["schema_name"] == "step_b_orders"]) == 2
        assert len([c for c in fake.calls if c["schema_name"] == "step_c_status_map"]) == 2
        assert len([c for c in fake.calls if c["schema_name"] == "step_d_customers_returns"]) == 2
    finally:
        infer_mod.validate_mapping = orig_validate


def test_prompt_shrinking_for_60_column_table(tmp_path: Path) -> None:
    long_readme = "# Shop\n\n## File\n- `Wide.xlsx`\n\n## Cột trong `Wide.xlsx`\n" + (
        "- Cột mô tả rất dài để vượt ngân sách token của prompt.\n" * 250
    ) + "\n## Quy ước\n- Dữ liệu có đến hết 30/09/2026.\n"
    (tmp_path / "README.md").write_text(long_readme, encoding="utf-8")

    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "Sheet1"
    cols = ["Mã đơn", "Ngày", "Trạng thái", "Tên món", "SL", "Thành tiền"] + [
        f"Thuộc_tính_mở_rộng_số_{i:02d}" for i in range(1, 55)
    ]
    assert len(cols) == 60
    ws.append(cols)
    for r in range(1, 15):
        row = [f"ORD{r:03d}", "01/09/2026", "xong", f"Món {r}", "1", "50000"] + [
            f"Giá trị mẫu khá dài của cột {c_idx} dòng {r}" for c_idx in range(1, 55)
        ]
        ws.append(row)
    wb.save(tmp_path / "Wide.xlsx")
    wb.close()

    raw = read_business(tmp_path)
    profile = profile_business(raw)

    all_ord_fields = {
        "order_id": "Mã đơn",
        "order_date": "Ngày",
        "order_ts": None,
        "status": "Trạng thái",
        "customer_id": None,
        "customer_name": None,
        "payment_method": None,
        "channel": None,
        "employee_id": None,
        "order_subtotal": None,
        "order_discount": None,
        "shipping_fee": None,
        "order_payable": None,
    }
    all_line_fields = {
        "order_id": "Mã đơn",
        "line_no": None,
        "product_id": None,
        "product_name": "Tên món",
        "qty": "SL",
        "unit": None,
        "qty_base": None,
        "unit_price": None,
        "line_discount": None,
        "line_amount": "Thành tiền",
        "cost_per_unit": None,
    }

    fake = FakeClient(
        {
            "step_a_table_roles": [
                {
                    "tables": [{"table": "Wide.xlsx!Sheet1", "entity": "order_lines"}],
                    "orders_source": "order_lines",
                }
            ],
            "step_b_order_lines": [all_line_fields],
            "step_b_orders": [all_ord_fields],
            "step_c_status_map": [{"xong": "completed"}],
            "step_d_customers_returns": [{"walk_in_values": [], "return_rule": "status"}],
        }
    )

    res = infer_mapping(raw, profile, fake)
    assert res.errors == []
    assert any(s.shrunk for s in res.steps)
    for c in fake.calls:
        assert c["prompt_tokens"] <= settings.llm_prompt_budget_tokens


def test_no_gold_mappings_or_business_id_in_mapping_pkg() -> None:
    mapping_dir = PROJECT_ROOT / "src" / "datamate" / "mapping"
    biz_re = re.compile(r"(?<![A-Za-z0-9])(?:biz_1\d\d|dev_0\d\d)", re.IGNORECASE)
    for py_file in mapping_dir.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        assert "gold_mappings" not in text
        assert biz_re.search(text) is None


def test_duplicate_raw_column_in_same_entity_and_allowed_pairs(tmp_path: Path) -> None:
    _write_two_table_shop(tmp_path)
    raw = read_business(tmp_path)
    mapping = Mapping.model_validate(
        {
            "customer_identity": "by_name",
            "return_rule": "status",
            "status_map": {"Đã xong": "completed", "Đã hủy": "cancelled"},
            "tables": {
                "orders": {
                    "sources": [{"file": "HoaDon.xlsx", "sheet": "Sheet1", "header_row": 3}],
                    "columns": {
                        "order_id": "Mã HĐ",
                        "order_date": "Tổng tiền hàng",
                        "status": "Trạng thái",
                        "order_subtotal": "Tổng tiền hàng",
                    },
                },
                "order_lines": {
                    "sources": [{"file": "ChiTiet.xlsx", "sheet": "Sheet1", "header_row": 1}],
                    "columns": {
                        "order_id": "Mã HĐ",
                        "product_name": "Tên món",
                        # (qty, unit) sharing a column IS allowed
                        "qty": "SL",
                        "unit": "SL",
                        "line_amount": "Thành tiền",
                    },
                },
            },
        }
    )

    issues, _canon = validate_mapping(raw, mapping)
    dup_issues = [
        iss
        for iss in issues
        if iss.step == "B_orders"
        and "order_date and order_subtotal both map to 'Tổng tiền hàng'" in iss.message
    ]
    assert len(dup_issues) == 1
    assert dup_issues[0].severity == "error"

    # Ensure (qty, unit) on order_lines did NOT trigger a duplicate column error
    assert not any("qty and unit both map" in iss.message for iss in issues)


def test_persistent_wrong_order_subtotal_is_dropped(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "# Shop\n\n## Quy ước\n- Dữ liệu có đến hết 15/08/2026.\n",
        encoding="utf-8",
    )
    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "Sheet1"
    ws.append(["Mã HĐ", "Ngày", "Trạng thái", "Khách", "Món", "SL", "Đơn giá", "Thành tiền", "Tổng sau giảm"])
    ws.append(["ORD01", "01/08/2026", "Đã xong", "Khách lẻ", "Cà phê", "2", "30000", "60000", "50000"])
    ws.append(["ORD01", "01/08/2026", "Đã xong", "Khách lẻ", "Bánh", "1", "50000", "50000", "50000"])
    wb.save(tmp_path / "BanHang.xlsx")
    wb.close()

    raw = read_business(tmp_path)
    profile = profile_business(raw)
    line_fields = {
        "order_id": "Mã HĐ",
        "line_no": None,
        "product_id": None,
        "product_name": "Món",
        "qty": "SL",
        "unit": None,
        "qty_base": None,
        "unit_price": "Đơn giá",
        "line_discount": None,
        "line_amount": "Thành tiền",
        "cost_per_unit": None,
    }
    # order_subtotal is the after-discount total, so the merchandise check keeps failing.
    wrong_orders = {
        "order_id": "Mã HĐ",
        "order_date": "Ngày",
        "order_ts": None,
        "status": "Trạng thái",
        "customer_id": None,
        "customer_name": "Khách",
        "payment_method": None,
        "channel": None,
        "employee_id": None,
        "order_subtotal": "Tổng sau giảm",
        "order_discount": None,
        "shipping_fee": None,
        "order_payable": None,
    }
    fake = FakeClient(
        {
            "step_a_table_roles": [
                {
                    "tables": [{"table": "BanHang.xlsx!Sheet1", "entity": "order_lines"}],
                    "orders_source": "order_lines",
                },
                {
                    "tables": [{"table": "BanHang.xlsx!Sheet1", "entity": "order_lines"}],
                    "orders_source": "order_lines",
                },
            ],
            "step_b_order_lines": [line_fields],
            "step_b_orders": [wrong_orders, wrong_orders],
            "step_c_status_map": [{"Đã xong": "completed"}],
            "step_d_customers_returns": [
                {"walk_in_values": ["Khách lẻ"], "return_rule": "status"}
            ],
        }
    )

    res = infer_mapping(raw, profile, fake)
    assert res.mapping is not None
    assert res.errors == []
    assert "order_subtotal" not in res.mapping.tables["orders"].columns
    assert any(
        iss.severity == "warning" and iss.message.startswith("dropped order_subtotal:")
        for iss in res.issues
    )
    order_calls = [c for c in fake.calls if c["schema_name"] == "step_b_orders"]
    assert len(order_calls) == 2
    assert "Previous answer:" in order_calls[1]["prompt"]


def test_invoice_only_table_marked_as_orders_becomes_the_line_table() -> None:
    from datamate.mapping.infer import _parse_step_a_roles

    answer = {
        "tables": [{"table": "danh_sach_hoa_don.csv", "entity": "orders"}],
        "orders_source": "own_table",
    }
    chosen, orders_source, error = _parse_step_a_roles(answer, {"danh_sach_hoa_don.csv"})
    assert error is None
    assert chosen == {"order_lines": "danh_sach_hoa_don.csv"}
    assert orders_source == "order_lines"

    two = {
        "tables": [
            {"table": "don.csv", "entity": "orders"},
            {"table": "khach.csv", "entity": "customers"},
        ],
        "orders_source": "own_table",
    }
    chosen, _source, error = _parse_step_a_roles(two, {"don.csv", "khach.csv"})
    assert error is None
    assert chosen["order_lines"] == "don.csv" and chosen["customers"] == "khach.csv"

    missing = {"tables": [{"table": "khach.csv", "entity": "customers"}], "orders_source": "own_table"}
    assert _parse_step_a_roles(missing, {"khach.csv"})[2] is not None


def test_free_text_status_column_dropped_without_step_c(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "# Shop\n\n## Quy ước\n- Dữ liệu có đến hết 15/08/2026.\n",
        encoding="utf-8",
    )
    lines = ["Ngày,Ghi chú,Món,Tiền"]
    for i in range(71):
        lines.append(f"01/08/2026,ghi chú số {i},Cà phê,10.000đ")
    (tmp_path / "ban.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    raw = read_business(tmp_path)
    profile = profile_business(raw)
    table_name = profile.tables[0].name
    null_ord = {
        "order_id": None,
        "order_date": "Ngày",
        "order_ts": None,
        "status": "Ghi chú",
        "customer_id": None,
        "customer_name": None,
        "payment_method": None,
        "channel": None,
        "employee_id": None,
        "order_subtotal": None,
        "order_discount": None,
        "shipping_fee": None,
        "order_payable": None,
    }
    null_line = {
        "order_id": None,
        "line_no": None,
        "product_id": None,
        "product_name": "Món",
        "qty": None,
        "unit": None,
        "qty_base": None,
        "unit_price": None,
        "line_discount": None,
        "line_amount": "Tiền",
        "cost_per_unit": None,
    }
    fake = FakeClient(
        {
            "step_a_table_roles": [
                {
                    "tables": [{"table": table_name, "entity": "order_lines"}],
                    "orders_source": "order_lines",
                }
            ],
            "step_b_order_lines": [null_line],
            "step_b_orders": [null_ord],
            "step_d_customers_returns": [
                {"walk_in_values": [], "return_rule": "status"}
            ],
        }
    )
    res = infer_mapping(raw, profile, fake)
    assert res.mapping is not None
    assert "status" not in res.mapping.tables["orders"].columns
    assert res.mapping.status_map == {}
    assert not any(c["schema_name"] == "step_c_status_map" for c in fake.calls)
    warn_msgs = [iss.message for iss in res.issues if iss.severity == "warning"]
    assert any("71 distinct" in msg for msg in warn_msgs)
    assert any("no status column" in msg for msg in warn_msgs)


def test_step_a_rejects_glossary_when_large_dated_sheet_exists(tmp_path: Path) -> None:
    from datamate.io.profile import ColumnProfile, TableProfile
    from datamate.mapping.infer import check_order_lines_sales_table

    glossary = TableProfile(
        name="tu_dien.xlsx!Sheet1",
        kind="xlsx",
        n_sources=1,
        source_files=["tu_dien.xlsx"],
        sheet_names=["Sheet1"],
        header_rows=[1],
        n_rows=59,
        n_exact_duplicate_rows=0,
        dropped_counts={},
        dropped_previews={},
        notes=[],
        columns=[
            ColumnProfile(
                name="Ma",
                group=None,
                non_null=59,
                null_ratio=0.0,
                n_distinct=59,
                kinds=[("id_like", 1.0)],
            )
        ],
    )
    sales = TableProfile(
        name="SoBanHang.xlsx!BanHang",
        kind="xlsx",
        n_sources=1,
        source_files=["SoBanHang.xlsx"],
        sheet_names=["BanHang"],
        header_rows=[1],
        n_rows=10_000,
        n_exact_duplicate_rows=0,
        dropped_counts={},
        dropped_previews={},
        notes=[],
        columns=[
            ColumnProfile(
                name="Ngay",
                group=None,
                non_null=10_000,
                null_ratio=0.0,
                n_distinct=400,
                kinds=[("date_dmy", 0.95)],
            )
        ],
    )
    profiles = {glossary.name: glossary, sales.name: sales}
    msg = check_order_lines_sales_table(
        {"order_lines": glossary.name},
        "order_lines",
        profiles,
    )
    assert msg is not None
    assert "59 rows and no date column" in msg
    assert "10000 rows with dates" in msg
    assert "glossary or lookup table is not the sales table" in msg

    # End-to-end: first Step A picks glossary, repair picks sales.
    (tmp_path / "README.md").write_text(
        "# Shop\n\n## Quy ước\n- Dữ liệu có đến hết 15/08/2026.\n",
        encoding="utf-8",
    )
    gloss_lines = ["Ma,Ten"] + [f"M{i},Ten {i}" for i in range(59)]
    (tmp_path / "tu_dien.csv").write_text("\n".join(gloss_lines) + "\n", encoding="utf-8")
    sale_lines = ["Ngay,Mon,Tien"] + [
        f"01/08/2026,Mon {i},10.000đ" for i in range(600)
    ]
    (tmp_path / "ban_hang.csv").write_text("\n".join(sale_lines) + "\n", encoding="utf-8")
    raw = read_business(tmp_path)
    profile = profile_business(raw)
    by_name = {t.name: t for t in profile.tables}
    # Prefer real profiles; require 10x — pad glossary small vs sales large from files.
    gloss_name = next(n for n in by_name if "tu_dien" in n)
    sale_name = next(n for n in by_name if "ban_hang" in n)
    assert by_name[sale_name].n_rows >= 10 * by_name[gloss_name].n_rows

    null_ord = {
        "order_id": None,
        "order_date": "Ngay",
        "order_ts": None,
        "status": None,
        "customer_id": None,
        "customer_name": None,
        "payment_method": None,
        "channel": None,
        "employee_id": None,
        "order_subtotal": None,
        "order_discount": None,
        "shipping_fee": None,
        "order_payable": None,
    }
    null_line = {
        "order_id": None,
        "line_no": None,
        "product_id": None,
        "product_name": "Mon",
        "qty": None,
        "unit": None,
        "qty_base": None,
        "unit_price": None,
        "line_discount": None,
        "line_amount": "Tien",
        "cost_per_unit": None,
    }
    fake = FakeClient(
        {
            "step_a_table_roles": [
                {
                    "tables": [{"table": gloss_name, "entity": "order_lines"}],
                    "orders_source": "order_lines",
                },
                {
                    "tables": [{"table": sale_name, "entity": "order_lines"}],
                    "orders_source": "order_lines",
                },
            ],
            "step_b_order_lines": [null_line],
            "step_b_orders": [null_ord],
            "step_d_customers_returns": [
                {"walk_in_values": [], "return_rule": "negative_qty"}
            ],
        }
    )
    res = infer_mapping(raw, profile, fake)
    assert res.mapping is not None
    assert res.mapping.tables["order_lines"].sources[0].file.startswith("ban_hang")
    a_calls = [c for c in fake.calls if c["schema_name"] == "step_a_table_roles"]
    assert len(a_calls) == 2
    assert "glossary or lookup table is not the sales table" in a_calls[1]["prompt"]


def test_multi_sheet_table_sharing_a_workbook_lists_its_sheets() -> None:
    import pandas as pd

    from datamate.io.raw import RawTable, SourceRef
    from datamate.mapping.infer import _build_source_specs_from_raw_table

    def table(name: str, sheets: list[str]) -> RawTable:
        return RawTable(
            name=name,
            kind="xlsx",
            sources=[SourceRef(file="SoBanHang.xlsx", sheet=s, header_row=1) for s in sheets],
            columns=["a"],
            df=pd.DataFrame({"a": []}),
        )

    sales = table("SoBanHang.xlsx!T#-#", ["T7-2025", "T8-2025"])
    glossary = table("SoBanHang.xlsx!Ghi tắt", ["Ghi tắt"])
    shared = _build_source_specs_from_raw_table(
        sales, {sales.name: sales, glossary.name: glossary}
    )
    assert [spec.sheet for spec in shared] == ["T7-2025", "T8-2025"]
    alone = _build_source_specs_from_raw_table(sales, {sales.name: sales})
    assert [spec.sheet for spec in alone] == ["*"]

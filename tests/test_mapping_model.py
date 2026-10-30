from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from datamate.mapping import Mapping, load_mapping_yaml, mapping_fingerprint


def _minimal_mapping_dict() -> dict[str, Any]:
    return {
        "data_end_date": "2026-08-31",
        "customer_identity": "by_name",
        "walk_in_values": ["Khách lẻ", "lẻ"],
        "return_rule": "status",
        "status_map": {
            "xong": "completed",
            "hủy": "cancelled",
            "trả": "returned",
            "đang giao": "shipping",
        },
        "tables": {
            "orders": {
                "derived_from": "order_lines",
                "columns": {
                    "order_id": "Mã đơn",
                    "order_date": "Ngày",
                    "status": "Trạng thái",
                },
            },
            "order_lines": {
                "sources": [{"file": "orders.csv", "sheet": None, "header_row": 1}],
                "columns": {
                    "order_id": "Mã đơn",
                    "product_name": "Tên món",
                    "qty": "SL",
                    "line_amount": "Thành tiền",
                },
            },
        },
    }


def test_valid_minimal_mapping_and_yaml_loader(tmp_path: Path) -> None:
    raw = _minimal_mapping_dict()
    raw["extra_top_level_key"] = "ignored_gracefully"

    m = Mapping.model_validate(raw)
    assert m.customer_identity == "by_name"
    assert m.return_rule == "status"

    yaml_file = tmp_path / "mapping.yaml"
    yaml_file.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    m_loaded = load_mapping_yaml(yaml_file)
    assert mapping_fingerprint(m) == mapping_fingerprint(m_loaded)


def test_unknown_canonical_field_raises() -> None:
    raw = _minimal_mapping_dict()
    raw["tables"]["order_lines"]["columns"]["not_a_canonical_field"] = "Cột X"
    with pytest.raises(ValidationError, match="Unknown canonical field"):
        Mapping.model_validate(raw)


def test_two_origins_on_table_mapping_raises() -> None:
    raw = _minimal_mapping_dict()
    raw["tables"]["orders"]["sources"] = [{"file": "orders.csv", "sheet": None, "header_row": 1}]
    # Both derived_from and sources set -> error
    with pytest.raises(ValidationError, match="Exactly one"):
        Mapping.model_validate(raw)

    raw2 = _minimal_mapping_dict()
    raw2["tables"]["unit_conversions"] = {
        "source": "readme",
        "sources": [{"file": "conv.xlsx", "sheet": "Sheet1", "header_row": 1}],
    }
    with pytest.raises(ValidationError, match="Exactly one"):
        Mapping.model_validate(raw2)


def test_invalid_status_map_value_raises() -> None:
    raw = _minimal_mapping_dict()
    raw["status_map"]["đang chờ"] = "pending"
    with pytest.raises(ValidationError):
        Mapping.model_validate(raw)


def test_missing_orders_or_order_lines_raises() -> None:
    raw = _minimal_mapping_dict()
    del raw["tables"]["orders"]
    with pytest.raises(ValidationError, match="orders"):
        Mapping.model_validate(raw)


def test_old_yaml_loads_with_empty_purchase_status_map() -> None:
    m = Mapping.model_validate(_minimal_mapping_dict())
    assert m.purchase_status_map == {}
    assert "customers" not in m.tables


def test_fingerprint_ignores_empty_purchase_status_map() -> None:
    """Empty purchase_status_map must not change fingerprints of pre-E1b order-only mappings."""
    import hashlib
    import json

    m = Mapping.model_validate(_minimal_mapping_dict())
    payload = m.model_dump(mode="json", exclude={"business_id", "notes"})
    assert payload.get("purchase_status_map") == {}
    payload.pop("purchase_status_map", None)
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    assert mapping_fingerprint(m) == expected
    # Stable known value for the minimal fixture (order-only, no new entities).
    assert mapping_fingerprint(m) == (
        "7e27a6d1f2f5b11ef660d8da81dd8865195ff688101db329071e7f090cc0153f"
    )


def test_purchase_status_map_accepted() -> None:
    raw = _minimal_mapping_dict()
    raw["purchase_status_map"] = {"đã hủy": "cancelled", "ok": "valid"}
    raw["tables"]["payments"] = {
        "sources": [{"file": "pay.csv", "sheet": None, "header_row": 1}],
        "columns": {"pay_date": "Ngày", "amount": "Số tiền"},
    }
    m = Mapping.model_validate(raw)
    assert m.purchase_status_map == {"đã hủy": "cancelled", "ok": "valid"}
    assert "payments" in m.tables


def test_strict_bool_yaml_loader_preserves_on_yes_no_strings(tmp_path: Path) -> None:
    yaml_text = """
data_end_date: 2026-08-31
customer_identity: by_name
walk_in_values: [no]
return_rule: status
status_map:
  Yes: completed
  No: cancelled
tables:
  orders:
    derived_from: order_lines
    columns:
      order_id: Mã đơn
  order_lines:
    sources:
      - file: orders.csv
        sheet: null
        header_row: 1
    join:
      product_name:
        table: products
        on:
          product_id: product_id
    columns:
      order_id: Mã đơn
      product_id: Mã SP
      qty: SL
      line_amount: Thành tiền
  products:
    sources:
      - file: products.csv
        sheet: null
        header_row: 1
    columns:
      product_id: Mã SP
      product_name: Tên SP
"""
    yaml_file = tmp_path / "strict_bool.yaml"
    yaml_file.write_text(yaml_text.strip(), encoding="utf-8")

    m = load_mapping_yaml(yaml_file)
    assert m.walk_in_values == ["no"]
    assert m.status_map == {"Yes": "completed", "No": "cancelled"}
    assert m.tables["order_lines"].join["product_name"].on == {"product_id": "product_id"}


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

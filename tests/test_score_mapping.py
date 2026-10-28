from __future__ import annotations

import datetime
import sys
from typing import Any

from datamate.config import PROJECT_ROOT
from datamate.mapping.model import Mapping

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.score_mapping import score


def _minimal_mapping_dict() -> dict[str, Any]:
    return {
        "data_end_date": datetime.date(2026, 8, 24),
        "customer_identity": "by_name",
        "walk_in_values": ["Khách lẻ"],
        "return_rule": "status",
        "status_map": {
            "Xong": "completed",
            "Đã hủy": "cancelled",
            "Đổi trả": "returned",
            "Đang giao": "shipping",
        },
        "tables": {
            "orders": {
                "derived_from": "order_lines",
                "columns": {
                    "order_id": "Mã hóa đơn",
                    "order_date": "Ngày tháng",
                    "status": "Kết quả",
                    "customer_name": "Đối tác",
                },
            },
            "order_lines": {
                "sources": [{"file": "ban_hang.csv", "sheet": None, "header_row": 1}],
                "columns": {
                    "order_id": "Mã hóa đơn",
                    "product_name": "Nội dung",
                    "qty": "so_luong",
                    "line_amount": "Thanh tien",
                },
            },
        },
    }


def test_score_gold_against_itself_is_perfect() -> None:
    gold = Mapping.model_validate(_minimal_mapping_dict())
    ms = score(gold, gold)
    assert ms.n_wrong == 0
    assert ms.n_missing == 0
    assert ms.n_extra == 0
    assert ms.field_accuracy == 1.0
    assert ms.tables_ok is True
    assert ms.status_ok is True
    assert ms.walk_in_ok is True
    assert ms.customer_identity_ok is True
    assert ms.return_rule_ok is True
    assert ms.data_end_ok is True
    assert all(v.verdict == "correct" for v in (*ms.columns, *ms.statuses))
    assert ms.mismatches() == []


def test_one_wrong_column_and_one_wrong_status() -> None:
    gold = Mapping.model_validate(_minimal_mapping_dict())
    pred = gold.model_copy(deep=True)
    pred.tables["order_lines"].columns["line_amount"] = "Giá bán"
    pred.status_map["Xong"] = "shipping"

    ms = score(pred, gold)

    col_wrong = [v for v in ms.columns if v.verdict == "wrong"]
    st_wrong = [v for v in ms.statuses if v.verdict == "wrong"]
    assert len(col_wrong) == 1
    assert col_wrong[0].key == "order_lines.line_amount"
    assert col_wrong[0].gold == "Thanh tien"
    assert col_wrong[0].pred == "Giá bán"
    assert len(st_wrong) == 1
    assert st_wrong[0].key == "Xong"
    assert st_wrong[0].gold == "completed"
    assert st_wrong[0].pred == "shipping"

    assert ms.n_wrong == 2
    assert ms.n_missing == 0
    assert ms.n_extra == 0
    denom = ms.n_correct + ms.n_wrong + ms.n_missing
    assert denom == len(ms.columns) + len(ms.statuses)
    assert ms.n_correct == denom - 2
    assert ms.field_accuracy == ms.n_correct / denom
    assert ms.status_ok is False
    assert ms.tables_ok is True
    assert ms.walk_in_ok is True
    assert ms.return_rule_ok is True
    assert ms.data_end_ok is True


def test_readme_source_is_not_a_table_miss_and_empty_walk_in_matches() -> None:
    raw = _minimal_mapping_dict()
    raw["tables"]["unit_conversions"] = {"source": "readme", "columns": {}}
    raw["walk_in_values"] = [""]
    gold = Mapping.model_validate(raw)

    pred_raw = _minimal_mapping_dict()
    pred_raw["walk_in_values"] = []
    pred = Mapping.model_validate(pred_raw)

    ms = score(pred, gold)
    assert ms.tables_ok is True
    assert ms.walk_in_ok is True
    assert ms.table_notes == []

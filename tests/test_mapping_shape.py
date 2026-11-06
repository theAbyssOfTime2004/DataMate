from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from datamate.mapping.validate import _line_shape_issues


def _lines(rows: list[tuple[str, str, int]]) -> SimpleNamespace:
    return SimpleNamespace(
        order_lines=pd.DataFrame(rows, columns=["order_id", "product_name", "line_amount"])
    )


def test_order_total_repeated_on_every_line_is_an_error() -> None:
    rows = []
    for i in range(30):
        rows += [(f"o{i}", "Bút bi", 50_000 + i), (f"o{i}", "Vở 96 trang", 50_000 + i)]
    issues = _line_shape_issues(_lines(rows), {"line_amount": "Tổng đơn"})
    assert len(issues) == 1 and "whole-order total" in issues[0].message


def test_product_name_with_a_few_values_is_an_error() -> None:
    rows = [(f"o{i}", "1 mặt" if i % 2 else "2 mặt", 1000 * i) for i in range(300)]
    issues = _line_shape_issues(_lines(rows), {"product_name": "Kiểu in"})
    assert len(issues) == 1 and "kind or option" in issues[0].message


def test_normal_lines_pass() -> None:
    rows = []
    for i in range(300):
        rows += [(f"o{i}", f"Hàng {i % 40}", 10_000 + i), (f"o{i}", f"Hàng {(i + 1) % 40}", 7_000)]
    assert _line_shape_issues(_lines(rows), {}) == []

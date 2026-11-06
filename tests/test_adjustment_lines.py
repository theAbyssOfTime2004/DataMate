from __future__ import annotations

from decimal import Decimal

import pandas as pd

from datamate.etl.build import split_adjustment_lines


def _line(order: str, key: str, name: str, amount: int) -> dict[str, object]:
    return {"order_id": order, "product_key": key, "product_name": name, "line_amount": Decimal(amount)}


def test_voucher_lines_leave_the_goods() -> None:
    catalog = pd.DataFrame({"product_key": ["p1"], "product_name": ["Tã dán size M"]})
    lines = [
        _line("o1", "p1", "Tã dán size M", 300000),
        _line("o1", "x1", "Voucher", -10000),
        _line("o2", "p1", "Tã dán size M", -150000),  # a returned item stays a goods line
        _line("o2", "x2", "Quà tặng", 0),  # never negative: not a discount
    ]
    kept, moved = split_adjustment_lines(lines, catalog)
    assert [line["product_name"] for line in kept] == ["Tã dán size M", "Tã dán size M", "Quà tặng"]
    assert moved == [{"order_id": "o1", "label": "Voucher", "amount": Decimal(10000)}]


def test_without_a_catalog_nothing_moves() -> None:
    lines = [_line("o1", "x1", "Voucher", -10000)]
    kept, moved = split_adjustment_lines(lines, pd.DataFrame(columns=["product_key"]))
    assert kept == lines and moved == []


def test_quarantined_fields_turn_unanswerable_into_abstain() -> None:
    from datamate.systems.pipeline import quarantine_hit

    assert quarantine_hit("shop does not record payment_method", ["orders.payment_method"])
    assert quarantine_hit("shop does not record purchases.total_amount", ["purchases"])
    assert not quarantine_hit("shop does not record category", ["orders.payment_method"])
    assert not quarantine_hit("period lies entirely outside the shop's data", ["debts"])

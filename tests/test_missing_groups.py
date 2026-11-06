"""Product-group questions in shops that record no product groups."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from datamate.query.missing_groups import asked_product_group, group_is_unrecorded, kind_values


@dataclass
class _Table:
    df: pd.DataFrame


def test_kind_values_skip_columns_with_many_values() -> None:
    tables = [
        _Table(
            pd.DataFrame(
                {
                    "Loại hàng": ["máy mới", "máy cũ", "phụ kiện"] * 30,
                    "Tên": [f"m{i}" for i in range(90)],
                }
            )
        )
    ]
    values = kind_values(tables, limit=10)
    assert {"may moi", "may cu", "phu kien"} <= values
    assert "m1" not in values


def test_group_counts_as_unrecorded_only_without_any_trace() -> None:
    kinds = {"may moi", "may cu", "phu kien", "hoan thanh"}
    assert group_is_unrecorded("hàng điện", [], kinds)
    assert not group_is_unrecorded("máy cũ", [], kinds)  # a group sheet the mapping missed
    assert not group_is_unrecorded("hàng điện", ["điện"], kinds)  # the shop has groups


class _Client:
    def __init__(self, group: str | None) -> None:
        self.group = group

    def chat_json(self, messages: object, schema: object, name: str, **_: object) -> object:
        del messages, schema, name
        return {"product_group": self.group}


def test_group_must_be_written_in_the_question() -> None:
    question = "Tháng 9 nhóm sơn nước bán được bao nhiêu tiền?"
    assert asked_product_group(question, _Client("sơn nước")) == "sơn nước"
    assert asked_product_group(question, _Client("vật liệu")) is None
    assert asked_product_group(question, _Client(None)) is None


def test_group_must_be_named_as_a_group_and_not_be_a_product() -> None:
    plain = "Tháng 9 sơn nước bán được bao nhiêu tiền?"
    assert asked_product_group(plain, _Client("sơn nước")) is None
    product = "Tháng 9 nhóm Sơn lót Jotun 5L bán được bao nhiêu tiền?"
    assert asked_product_group(product, _Client("Sơn lót Jotun 5L"), ["Sơn lót Jotun 5L"]) is None

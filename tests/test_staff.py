"""Staff nicknames on orders are tied to the staff list only when one person fits."""

from __future__ import annotations

import pandas as pd

from datamate.etl.staff import resolve_staff


def _staff(rows: list[tuple[str | None, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["employee_id", "employee_name"])


def test_nicknames_take_the_listed_full_name() -> None:
    orders = pd.DataFrame({"employee_id": ["A.Đức", "Đức", "a Đức", "chị Lan", "Bình", None]})
    staff = _staff(
        [(None, "Lương Chi Đức"), (None, "Đàm Lan"), (None, "Trần Bình"), (None, "Lê Văn Bình")]
    )
    out, listed, tied = resolve_staff(orders, staff)
    assert list(out["employee_id"][:5]) == [
        "Lương Chi Đức",
        "Lương Chi Đức",
        "Lương Chi Đức",
        "Đàm Lan",
        "Bình",  # two listed people are called Bình: left as written
    ]
    assert pd.isna(out["employee_id"].iloc[5])
    assert list(listed["employee_id"]) == ["Lương Chi Đức", "Đàm Lan", "Trần Bình", "Lê Văn Bình"]
    assert tied == 4


def test_codes_stay_codes() -> None:
    orders = pd.DataFrame({"employee_id": ["NV01", "nv02", "Hùng"]})
    staff = _staff([("NV01", "Phạm Văn Hùng"), ("NV02", "Ngô Thị Mai")])
    out, _listed, _tied = resolve_staff(orders, staff)
    assert list(out["employee_id"]) == ["NV01", "NV02", "NV01"]


def test_no_staff_list_changes_nothing() -> None:
    orders = pd.DataFrame({"employee_id": ["a Đức"]})
    out, _listed, tied = resolve_staff(
        orders, pd.DataFrame(columns=["employee_id", "employee_name"])
    )
    assert list(out["employee_id"]) == ["a Đức"] and tied == 0

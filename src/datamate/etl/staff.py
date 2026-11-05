"""Tie the staff written on orders to the staff list.

Books often write a nickname ("a Đức", "A.Đức", "Đức") while the staff list holds the
full name ("Lương Chi Đức") and maybe a code. Vietnamese people are called by the last
word of their full name, so a written value whose last word is the last word of exactly
one listed name is that person. Anything else is left as written.
"""

from __future__ import annotations

import pandas as pd

from datamate.query.grounding import fold


def _words(value: object) -> list[str]:
    return fold(str(value)).split() if isinstance(value, str) and value.strip() else []


def resolve_staff(
    orders: pd.DataFrame, employees: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Orders with employee_id set to the listed person's key, the staff list with a key on
    every row (code, else full name), and how many written values were tied to someone."""
    if orders.empty or employees.empty or "employee_id" not in orders.columns:
        return orders, employees, 0
    staff = employees.copy()
    keys: list[object] = []
    for code, name in zip(staff["employee_id"], staff["employee_name"], strict=True):
        has_code = isinstance(code, str) and code.strip()
        keys.append(
            code if has_code else (name if isinstance(name, str) and name.strip() else None)
        )
    staff["employee_id"] = keys

    exact: dict[str, str] = {}
    by_last_word: dict[str, list[str]] = {}
    for key, code, name in zip(keys, employees["employee_id"], staff["employee_name"], strict=True):
        if key is None:
            continue
        for value in (code, name):
            words = _words(value)
            if words:
                exact[" ".join(words)] = str(key)
        name_words = _words(name)
        if name_words:
            by_last_word.setdefault(name_words[-1], []).append(str(key))

    tied: dict[object, str] = {}
    for written in orders["employee_id"].dropna().unique():
        words = _words(written)
        if not words:
            continue
        key = exact.get(" ".join(words))
        if key is None:
            matches = by_last_word.get(words[-1], [])
            key = matches[0] if len(matches) == 1 else None
        if key is not None:
            tied[written] = key
    if not tied:
        return orders, staff, 0
    out = orders.copy()
    out["employee_id"] = pd.Series(
        [tied.get(value, value) for value in out["employee_id"]], index=out.index, dtype=object
    )
    return out, staff, sum(1 for written, key in tied.items() if written != key)

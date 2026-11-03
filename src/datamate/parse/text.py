from __future__ import annotations

import re
import unicodedata

_INGREDIENT_SEP = re.compile(r"\s*(?:[+,;/&]|\bvà\b|\bva\b)\s*", re.IGNORECASE)


# "11/25", "th3-26": a second number above 12 cannot be a month, so it is a year.
_MONTH_SHORT_YEAR = re.compile(r"(?<![\d/])(?<!\d[.,])(1[0-2]|0?[1-9])([/-])(1[3-9]|[2-9]\d)(?![\d/])(?![.,]\d)")


def normalize_label(raw: str | None) -> str | None:
    if raw is None or not isinstance(raw, str):
        return None
    s = unicodedata.normalize("NFC", raw).replace("\xa0", " ").strip()
    if not s:
        return None
    return " ".join(s.casefold().split())


def expand_short_years(text: str) -> str:
    """Write two-digit years in full ("th 11/25" -> "th 11/2025") so the parser does
    not read them as a day of the month."""
    return _MONTH_SHORT_YEAR.sub(lambda m: f"{m[1]}{m[2]}20{m[3]}", text)


def ingredient_parts(raw: str) -> list[str]:
    """Active ingredients of one drug, in written order: "A + B", "A, B" and "A và B"
    all name the same combination."""
    parts = (normalize_label(part) for part in _INGREDIENT_SEP.split(raw))
    return [part for part in parts if part]

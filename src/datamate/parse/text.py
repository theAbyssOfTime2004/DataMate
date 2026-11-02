from __future__ import annotations

import re
import unicodedata

_INGREDIENT_SEP = re.compile(r"\s*(?:[+,;/&]|\bvà\b|\bva\b)\s*", re.IGNORECASE)


def normalize_label(raw: str | None) -> str | None:
    if raw is None or not isinstance(raw, str):
        return None
    s = unicodedata.normalize("NFC", raw).replace("\xa0", " ").strip()
    if not s:
        return None
    return " ".join(s.casefold().split())


def ingredient_parts(raw: str) -> list[str]:
    """Active ingredients of one drug, in written order: "A + B", "A, B" and "A và B"
    all name the same combination."""
    parts = (normalize_label(part) for part in _INGREDIENT_SEP.split(raw))
    return [part for part in parts if part]

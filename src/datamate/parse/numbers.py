from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation

_COMMA_THOUSANDS_RE = re.compile(r"^\d{1,3}(?:,\d{3})+(?:\.\d+)?$")
_DOT_THOUSANDS_RE = re.compile(r"^\d{1,3}(?:\.\d{3})+(?:,\d+)?$")
_SPACE_THOUSANDS_RE = re.compile(r"^\d{1,3}(?:\s\d{3})+(?:[.,]\d+)?$")
_PLAIN_DECIMAL_RE = re.compile(r"^\d+(?:[.,]\d+)?$")

_QTY_SPLIT_RE = re.compile(
    r"^(?P<num>[+-]?(?:\d{1,3}(?:[.,\s]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?))"
    r"(?:\s+(?P<unit>\S.*))?$"
)


def parse_decimal(raw: str | None) -> Decimal | None:
    if raw is None or not isinstance(raw, str):
        return None
    s = unicodedata.normalize("NFC", raw).replace("\xa0", " ").strip()
    if not s:
        return None

    sign = 1
    if s.startswith("-"):
        sign = -1
        s = s[1:].strip()
    elif s.startswith("+"):
        s = s[1:].strip()

    if not s:
        return None

    normalized: str | None = None
    if _COMMA_THOUSANDS_RE.match(s):
        normalized = s.replace(",", "")
    elif _DOT_THOUSANDS_RE.match(s):
        # Single dot with 3 digits like "1.500" in decimal vs thousands:
        # In rule (c), dot-thousands is e.g. "1.200.000" or "55.000".
        normalized = s.replace(".", "").replace(",", ".")
    elif _SPACE_THOUSANDS_RE.match(s):
        normalized = re.sub(r"\s+", "", s).replace(",", ".")
    elif _PLAIN_DECIMAL_RE.match(s):
        normalized = s.replace(",", ".")

    if normalized is None:
        return None

    try:
        val = Decimal(normalized)
    except InvalidOperation:
        return None
    if not val.is_finite():
        return None
    return val * sign


def parse_qty(raw: str | None) -> tuple[Decimal, str | None] | None:
    if raw is None or not isinstance(raw, str):
        return None
    s = unicodedata.normalize("NFC", raw).replace("\xa0", " ").strip()
    if not s:
        return None

    m = _QTY_SPLIT_RE.match(s)
    if m is None:
        return None

    num_str = m.group("num")
    unit_raw = m.group("unit")

    dec = parse_decimal(num_str)
    if dec is None:
        return None

    if unit_raw is None:
        return dec, None

    unit_clean = unicodedata.normalize("NFC", unit_raw).strip()
    if not unit_clean or not any(ch.isalpha() for ch in unit_clean):
        return None

    return dec, unit_clean

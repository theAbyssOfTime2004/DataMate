from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation

_CURRENCY_SUFFIX_RE = re.compile(r"\s*(?:vnđ|vnd|đồng|dong|đ)\s*$", re.IGNORECASE)

# Rule (a): <int>tr<digits>, e.g. "1tr440", "9tr960", "36tr936", "1tr4"
_RULE_A_RE = re.compile(r"^(?P<int_part>\d+)\s*tr\s*(?P<frac_digits>\d{1,6})$")

# Rule (b): <num>[.,<frac>] (tr|triệu|k|nghìn|ngàn)
_RULE_B_RE = re.compile(
    r"^(?P<num>\d+(?:[.,]\d+)?)\s*(?P<unit>triệu|trieu|tr|nghìn|nghin|ngàn|ngan|k)$"
)

# Rule (c1): comma thousands, optional dot decimal: "560,000", "115,000.0000", "1,032,648,800"
_COMMA_THOUSANDS_RE = re.compile(r"^\d{1,3}(?:,\d{3})+(?:\.\d+)?$")

# Rule (c2): dot thousands, optional comma decimal: "1.200.000", "55.000", "55.000,00"
_DOT_THOUSANDS_RE = re.compile(r"^\d{1,3}(?:\.\d{3})+(?:,\d+)?$")

# Rule (c3): space thousands, optional dot/comma decimal: "16 076", "50 000"
_SPACE_THOUSANDS_RE = re.compile(r"^\d{1,3}(?:\s\d{3})+(?:[.,]\d+)?$")

# Rule (d): plain integer or decimal: "45000", "45000.0", "0"
_PLAIN_DECIMAL_RE = re.compile(r"^\d+(?:\.\d+)?$")


def _to_exact_int(val: Decimal) -> int | None:
    if not val.is_finite():
        return None
    integral = val.to_integral_value()
    if val != integral:
        return None
    return int(integral)


def parse_money(raw: str | None) -> int | None:
    if raw is None or not isinstance(raw, str):
        return None
    s = unicodedata.normalize("NFC", raw).replace("\xa0", " ").strip().lower()
    if not s:
        return None

    sign = 1
    if s.startswith("-"):
        sign = -1
        s = s[1:].strip()
    elif s.startswith("+"):
        s = s[1:].strip()

    s = _CURRENCY_SUFFIX_RE.sub("", s).strip()
    if not s:
        return None

    # Rule (a): "<int>tr<digits>"
    m_a = _RULE_A_RE.match(s)
    if m_a is not None:
        millions = int(m_a.group("int_part")) * 1_000_000
        frac_str = m_a.group("frac_digits").ljust(6, "0")
        return sign * (millions + int(frac_str))

    # Rule (b): "<num>[.,<frac>] (tr|triệu|k|nghìn|ngàn)"
    m_b = _RULE_B_RE.match(s)
    if m_b is not None:
        num_str = m_b.group("num").replace(",", ".")
        try:
            dec_num = Decimal(num_str)
        except InvalidOperation:
            return None
        unit = m_b.group("unit")
        factor = Decimal(1_000_000) if unit in {"tr", "triệu", "trieu"} else Decimal(1_000)
        res = _to_exact_int(dec_num * factor)
        return sign * res if res is not None else None

    # Rule (c): thousands-separated
    if _COMMA_THOUSANDS_RE.match(s):
        clean = s.replace(",", "")
        try:
            res = _to_exact_int(Decimal(clean))
        except InvalidOperation:
            return None
        return sign * res if res is not None else None

    if _DOT_THOUSANDS_RE.match(s):
        clean = s.replace(".", "").replace(",", ".")
        try:
            res = _to_exact_int(Decimal(clean))
        except InvalidOperation:
            return None
        return sign * res if res is not None else None

    if _SPACE_THOUSANDS_RE.match(s):
        clean = re.sub(r"\s+", "", s).replace(",", ".")
        try:
            res = _to_exact_int(Decimal(clean))
        except InvalidOperation:
            return None
        return sign * res if res is not None else None

    # Rule (d): plain number
    if _PLAIN_DECIMAL_RE.match(s):
        try:
            res = _to_exact_int(Decimal(s))
        except InvalidOperation:
            return None
        return sign * res if res is not None else None

    return None

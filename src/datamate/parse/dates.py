from __future__ import annotations

import datetime
import re
import unicodedata
from decimal import Decimal, InvalidOperation

_YEAR_HINT_RE = re.compile(r"(?<!\d)((?:19|20)\d\d)(?!\d)")

_ISO_DATETIME_RE = re.compile(
    r"^(?P<y>\d{4})-(?P<m>\d{1,2})-(?P<d>\d{1,2})"
    r"(?:[ T](?P<hh>\d{1,2}):(?P<mm>\d{2})(?::(?P<ss>\d{2}))?)?$"
)

_DMY_RE = re.compile(
    r"^(?P<d>\d{1,2})[/.\-](?P<m>\d{1,2})[/.\-](?P<y>\d{4})"
    r"(?:\s+(?P<hh>\d{1,2}):(?P<mm>\d{2})(?::(?P<ss>\d{2}))?)?$"
)

_DM_RE = re.compile(
    r"^(?P<d>\d{1,2})[/.\-](?P<m>\d{1,2})"
    r"(?:\s+(?P<hh>\d{1,2}):(?P<mm>\d{2})(?::(?P<ss>\d{2}))?)?$"
)

_THG_RE = re.compile(
    r"^(?P<d>\d{1,2})\s+thg\s+(?P<m>\d{1,2})(?:\s+(?P<y>\d{4}))?"
    r"(?:\s+(?P<hh>\d{1,2}):(?P<mm>\d{2})(?::(?P<ss>\d{2}))?)?$",
    re.IGNORECASE,
)

_SERIAL_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")

_EXCEL_EPOCH = datetime.datetime(1899, 12, 30)


def _clean_input(raw: str | None) -> str | None:
    if raw is None or not isinstance(raw, str):
        return None
    s = unicodedata.normalize("NFC", raw).replace("\xa0", " ").strip()
    return s if s else None


def year_hint(sheet: str | None, file: str | None) -> int | None:
    for candidate in (sheet, file):
        if candidate is None:
            continue
        m = _YEAR_HINT_RE.search(candidate)
        if m is not None:
            return int(m.group(1))
    return None


def _safe_date(year: int, month: int, day: int) -> datetime.date | None:
    try:
        return datetime.date(year, month, day)
    except ValueError:
        return None


def _safe_datetime(
    year: int, month: int, day: int, hour: int, minute: int, second: int
) -> datetime.datetime | None:
    try:
        return datetime.datetime(year, month, day, hour, minute, second)
    except ValueError:
        return None


def parse_date(raw: str | None, default_year: int | None = None) -> datetime.date | None:
    s = _clean_input(raw)
    if s is None:
        return None

    m = _ISO_DATETIME_RE.match(s)
    if m is not None:
        return _safe_date(int(m.group("y")), int(m.group("m")), int(m.group("d")))

    m = _DMY_RE.match(s)
    if m is not None:
        return _safe_date(int(m.group("y")), int(m.group("m")), int(m.group("d")))

    m = _THG_RE.match(s)
    if m is not None:
        yr_str = m.group("y")
        yr = int(yr_str) if yr_str is not None else default_year
        if yr is None:
            return None
        return _safe_date(yr, int(m.group("m")), int(m.group("d")))

    m = _DM_RE.match(s)
    if m is not None:
        if default_year is None:
            return None
        return _safe_date(default_year, int(m.group("m")), int(m.group("d")))

    if _SERIAL_RE.match(s):
        try:
            val = Decimal(s)
        except InvalidOperation:
            return None
        if Decimal(20000) <= val <= Decimal(80000):
            days = int(val)
            return (_EXCEL_EPOCH + datetime.timedelta(days=days)).date()

    return None


def parse_timestamp(
    raw: str | None, default_year: int | None = None
) -> datetime.datetime | None:
    s = _clean_input(raw)
    if s is None:
        return None

    for pattern in (_ISO_DATETIME_RE, _DMY_RE, _THG_RE, _DM_RE):
        m = pattern.match(s)
        if m is not None:
            if m.group("hh") is None:
                return None
            yr_str = m.groupdict().get("y")
            yr = int(yr_str) if yr_str is not None else default_year
            if yr is None:
                return None
            hh = int(m.group("hh"))
            mm = int(m.group("mm"))
            ss = int(m.group("ss")) if m.group("ss") is not None else 0
            return _safe_datetime(yr, int(m.group("m")), int(m.group("d")), hh, mm, ss)

    if _SERIAL_RE.match(s):
        try:
            val = Decimal(s)
        except InvalidOperation:
            return None
        if Decimal(20000) <= val <= Decimal(80000):
            days = int(val)
            frac = val - Decimal(days)
            if frac <= 0:
                return None
            total_seconds = int((frac * Decimal(86400)).quantize(Decimal(1)))
            return _EXCEL_EPOCH + datetime.timedelta(days=days, seconds=total_seconds)

    return None

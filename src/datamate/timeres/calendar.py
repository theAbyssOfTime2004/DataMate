from __future__ import annotations

import csv
import datetime
import unicodedata
from pathlib import Path

from datamate.config import settings

_LunarKey = tuple[int, int, int, bool]


class CalendarError(ValueError):
    """Raised when a lunar date is missing from calendar.csv."""


def _parse_leap(raw: str) -> bool:
    value = raw.strip().casefold()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no", ""}:
        return False
    raise CalendarError(f"lunar_leap must be true or false, got {raw!r}")


class LunarCalendar:
    """Solar dates keyed by lunar year, month, day, and whether the month is a leap month."""

    def __init__(self, by_lunar: dict[_LunarKey, datetime.date]) -> None:
        self._by_lunar = by_lunar

    @classmethod
    def from_csv(cls, path: Path | str) -> LunarCalendar:
        csv_path = Path(path)
        by_lunar: dict[_LunarKey, datetime.date] = {}
        with csv_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                "date",
                "weekday",
                "lunar_day",
                "lunar_month",
                "lunar_year",
                "lunar_leap",
                "holiday",
            }
            if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
                raise CalendarError(
                    f"{csv_path} is missing calendar columns; got {reader.fieldnames}"
                )
            for row in reader:
                solar = datetime.date.fromisoformat(row["date"])
                key = (
                    int(row["lunar_year"]),
                    int(row["lunar_month"]),
                    int(row["lunar_day"]),
                    _parse_leap(row["lunar_leap"]),
                )
                holiday = unicodedata.normalize("NFC", row.get("holiday") or "")
                existing = by_lunar.get(key)
                if existing is not None and existing != solar:
                    raise CalendarError(
                        f"lunar date {key} maps to both {existing} and {solar} ({holiday})"
                    )
                by_lunar[key] = solar
        return cls(by_lunar)

    def to_solar(
        self,
        lunar_year: int,
        lunar_month: int,
        lunar_day: int,
        leap: bool = False,
    ) -> datetime.date:
        key = (lunar_year, lunar_month, lunar_day, leap)
        found = self._by_lunar.get(key)
        if found is None:
            raise CalendarError(f"lunar date {key} is not in the calendar")
        return found


def load_calendar(split: str) -> LunarCalendar:
    return LunarCalendar.from_csv(settings.split_dir(split) / "calendar.csv")

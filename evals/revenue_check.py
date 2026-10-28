from __future__ import annotations

import calendar
import datetime
from pathlib import Path

import duckdb


def query_monthly_revenue(db_path: Path | str, year_month: str) -> int:
    parts = year_month.strip().split("-")
    if len(parts) != 2:
        raise ValueError(f"Expected YYYY-MM format, got {year_month!r}")
    year = int(parts[0])
    month = int(parts[1])
    first_day = datetime.date(year, month, 1)
    last_day = datetime.date(year, month, calendar.monthrange(year, month)[1])

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        row = conn.execute(
            """
            SELECT COALESCE(SUM(l.line_amount), 0)
            FROM order_lines l
            JOIN orders o USING (order_id)
            WHERE o.status IN ('completed', 'shipping')
              AND NOT o.has_return
              AND o.order_date BETWEEN ? AND ?
            """,
            [first_day, last_day],
        ).fetchone()
        assert row is not None
        return int(row[0])
    finally:
        conn.close()

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pytest

from datamate.config import PROJECT_ROOT, settings
from datamate.etl import EtlReport, load_or_build_canonical
from datamate.mapping import Mapping, load_mapping_yaml

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.revenue_check import query_monthly_revenue

GOLD_DIR = PROJECT_ROOT / "evals" / "gold_mappings"
POC_BUSINESSES = ("biz_101", "biz_105", "biz_106", "biz_111", "biz_123")

EXPECTED_REVENUES: list[tuple[str, str, int]] = [
    ("biz_101", "2026-01", 245_184_900),
    ("biz_101", "2026-03", 232_535_600),
    ("biz_101", "2026-07", 236_132_200),
    ("biz_105", "2025-08", 368_766_700),
    ("biz_105", "2025-09", 348_311_500),
    ("biz_105", "2025-10", 379_300_800),
    ("biz_105", "2025-11", 359_106_000),
    ("biz_105", "2025-12", 377_663_800),
    ("biz_105", "2026-01", 389_514_000),
    ("biz_105", "2026-02", 269_031_700),
    ("biz_105", "2026-03", 398_881_600),
    ("biz_106", "2025-05", 271_866_000),
    ("biz_106", "2025-11", 264_914_500),
    ("biz_106", "2025-12", 305_793_500),
    ("biz_106", "2026-01", 295_438_500),
    ("biz_111", "2025-03", 226_550_000),
    ("biz_123", "2025-10", 202_375_900),
]

EXPECTED_DEDUPE_COUNTS: dict[str, int] = {
    "biz_101": 0,
    "biz_105": 2222,
    "biz_106": 3230,
    "biz_111": 0,
    "biz_123": 1208,
}


@pytest.fixture(scope="module")
def gold_builds() -> dict[str, tuple[Path, EtlReport, Mapping]]:
    if not (settings.data_root / "train").exists():
        pytest.skip("train split absent")

    results: dict[str, tuple[Path, EtlReport, Mapping]] = {}
    for biz_id in POC_BUSINESSES:
        mapping = load_mapping_yaml(GOLD_DIR / f"{biz_id}.yaml")
        db_path, report = load_or_build_canonical("train", biz_id, mapping)
        results[biz_id] = (db_path, report, mapping)
    return results


@pytest.mark.parametrize(("biz_id", "year_month", "expected_rev"), EXPECTED_REVENUES)
def test_gold_monthly_revenues(
    gold_builds: dict[str, tuple[Path, EtlReport, Mapping]],
    biz_id: str,
    year_month: str,
    expected_rev: int,
) -> None:
    db_path, _, _ = gold_builds[biz_id]
    actual = query_monthly_revenue(db_path, year_month)
    assert actual == expected_rev, (
        f"{biz_id} {year_month}: expected {expected_rev:,}, got {actual:,} (diff={actual - expected_rev:+,})"
    )


@pytest.mark.parametrize(("biz_id", "expected_dedupe"), list(EXPECTED_DEDUPE_COUNTS.items()))
def test_gold_dedupe_counts(
    gold_builds: dict[str, tuple[Path, EtlReport, Mapping]],
    biz_id: str,
    expected_dedupe: int,
) -> None:
    _, report, _ = gold_builds[biz_id]
    assert sum(report.dedupe_counts.values()) == expected_dedupe


@pytest.mark.parametrize("biz_id", ["biz_106", "biz_123"])
def test_gold_subtotal_match_share(
    gold_builds: dict[str, tuple[Path, EtlReport, Mapping]],
    biz_id: str,
) -> None:
    _, report, _ = gold_builds[biz_id]
    assert report.integrity.subtotal_match_share == 1.0


@pytest.mark.parametrize("biz_id", POC_BUSINESSES)
def test_gold_status_parse_rates_and_end_date(
    gold_builds: dict[str, tuple[Path, EtlReport, Mapping]],
    biz_id: str,
) -> None:
    _, report, _ = gold_builds[biz_id]
    assert report.unmapped_statuses == {}
    assert report.status_overridden_by_return == 0
    assert report.integrity.orders_after_data_end_date == 0

    required_fields = (
        "orders.order_id",
        "orders.order_date",
        "orders.status",
        "order_lines.order_id",
        "order_lines.line_amount",
        "order_lines.qty",
    )
    for field_key in required_fields:
        assert field_key in report.parse_stats, f"{biz_id} missing parse stat for {field_key}"
        assert report.parse_stats[field_key].rate == 1.0, (
            f"{biz_id} {field_key} parse rate={report.parse_stats[field_key].rate} "
            f"examples={report.parse_stats[field_key].examples}"
        )


def test_biz_105_negative_qty_return_orders_and_dates(
    gold_builds: dict[str, tuple[Path, EtlReport, Mapping]],
) -> None:
    from datamate.io.raw import load_business_cached
    from datamate.parse import parse_date, parse_qty, year_hint

    db_path, _, _ = gold_builds["biz_105"]
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        missing_flag = conn.execute(
            """
            SELECT COUNT(DISTINCT l.order_id)
            FROM order_lines l
            JOIN orders o USING (order_id)
            WHERE l.qty < 0 AND NOT o.has_return
            """
        ).fetchone()
        assert missing_flag is not None and missing_flag[0] == 0

        neg_order_rows = dict(
            conn.execute(
                """
                SELECT DISTINCT o.order_id, o.order_date
                FROM orders o
                JOIN order_lines l USING (order_id)
                WHERE l.qty < 0
                """
            ).fetchall()
        )
        assert len(neg_order_rows) > 0
    finally:
        conn.close()

    biz_raw = load_business_cached("train", "biz_105")
    sb_table = next(t for t in biz_raw.tables if "SoBanHang" in t.name)
    pos_date_by_oid: dict[str, object] = {}
    for row in sb_table.df.to_dict("records"):
        oid = row["Mã hóa đơn"]
        if not isinstance(oid, str) or oid not in neg_order_rows or oid in pos_date_by_oid:
            continue
        q = parse_qty(row["so_luong"])
        if q is not None and q[0] >= 0:
            pos_date_by_oid[oid] = parse_date(
                row["Ngày giao dịch"],
                default_year=year_hint(row["__sheet"], row["__file"]),
            )

    assert len(pos_date_by_oid) == len(neg_order_rows)
    for oid, odate in neg_order_rows.items():
        assert odate == pos_date_by_oid[oid]


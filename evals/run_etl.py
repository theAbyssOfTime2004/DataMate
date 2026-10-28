from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datamate.etl import EtlReport, build_canonical, load_or_build_canonical, write_duckdb
from datamate.io.raw import load_business_cached
from datamate.mapping import load_mapping_yaml
from evals.revenue_check import query_monthly_revenue

GOLD_DIR = PROJECT_ROOT / "evals" / "gold_mappings"


def _render_detailed_report(
    business_id: str,
    db_path: Path | None,
    report: EtlReport,
    revenues: dict[str, int],
    elapsed_s: float,
) -> str:
    lines: list[str] = [
        f"# ETL Report — `{business_id}` ({elapsed_s:.2f}s)",
        f"- DuckDB: `{db_path or '(no-cache in-memory)'}`",
        "- Rows per table: "
        + ", ".join(f"{k}={v}" for k, v in report.rows_per_table.items()),
        "- Dedupe counts: "
        + (
            ", ".join(f"{k}={v}" for k, v in report.dedupe_counts.items())
            if report.dedupe_counts
            else "0"
        ),
        "- Fill counts: "
        + (
            ", ".join(f"{k}={v}" for k, v in report.fill_counts.items())
            if report.fill_counts
            else "0"
        ),
        f"- Unmapped statuses: {report.unmapped_statuses or 'none'}",
        f"- Conflicts: {report.conflicts or 'none'}",
        f"- Status overridden by return: {report.status_overridden_by_return}",
    ]

    imperfect = {
        k: st for k, st in report.parse_stats.items() if st.rate < 1.0
    }
    if imperfect:
        lines.append("- Parse rates < 100%:")
        for field_key, st in imperfect.items():
            lines.append(
                f"  - `{field_key}`: {st.parsed}/{st.non_null} ({st.rate:.4%}) "
                f"examples={st.examples}"
            )
    else:
        lines.append("- Parse rates < 100%: none (all 100%)")

    integ = report.integrity
    lines.append(
        f"- Integrity: orphan_line_orders={integ.orphan_line_orders}, "
        f"orders_without_lines={integ.orders_without_lines}, "
        f"subtotal_match_share={integ.subtotal_match_share}, "
        f"date_range=[{integ.min_order_date}..{integ.max_order_date}], "
        f"orders_after_end={integ.orders_after_data_end_date}"
    )

    if report.warnings:
        lines.append(f"- Warnings: {report.warnings}")
    if report.notes:
        lines.append(f"- Notes: {report.notes}")

    if revenues:
        lines.append("- Monthly revenues:")
        for ym, rev in revenues.items():
            lines.append(f"  - `{ym}`: {rev:,} VND ({rev})")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Run canonical ETL on gold or custom mappings.")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument("--business", default=None, help="Business ID (e.g. biz_105)")
    parser.add_argument(
        "--mapping",
        type=Path,
        default=None,
        help="Path to mapping YAML (default: evals/gold_mappings/<business>.yaml)",
    )
    parser.add_argument(
        "--revenue",
        action="append",
        default=[],
        help="Month YYYY-MM to compute revenue for (repeatable)",
    )
    parser.add_argument(
        "--all-gold",
        action="store_true",
        help="Run all evals/gold_mappings/biz_*.yaml and print one summary line per business",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Call build_canonical directly without reading or writing canonical cache",
    )
    args = parser.parse_args(argv)

    if not args.all_gold and not args.business:
        parser.error("Provide either --business <id> or --all-gold")

    if args.all_gold:
        yaml_files = sorted(GOLD_DIR.glob("biz_*.yaml"), key=lambda p: p.stem)
        for ypath in yaml_files:
            biz_id = ypath.stem
            mapping = load_mapping_yaml(ypath)
            if args.no_cache:
                raw = load_business_cached(args.split, biz_id)
                t0 = time.perf_counter()
                rep = build_canonical(raw, mapping).report
                elapsed = time.perf_counter() - t0
            else:
                t0 = time.perf_counter()
                _, rep = load_or_build_canonical(args.split, biz_id, mapping)
                elapsed = time.perf_counter() - t0
            total_dedupe = sum(rep.dedupe_counts.values())
            total_fills = sum(rep.fill_counts.values())
            imperfect_cnt = sum(1 for st in rep.parse_stats.values() if st.rate < 1.0)
            integ = rep.integrity
            print(
                f"{biz_id}: orders={rep.rows_per_table.get('orders', 0)} | "
                f"lines={rep.rows_per_table.get('order_lines', 0)} | "
                f"dedupe={total_dedupe} | fills={total_fills} | "
                f"unmapped_status={sum(rep.unmapped_statuses.values())} | "
                f"return_override={rep.status_overridden_by_return} | "
                f"subtotal_share={integ.subtotal_match_share} | "
                f"imperfect_fields={imperfect_cnt} | "
                f"dates=[{integ.min_order_date}..{integ.max_order_date}] | "
                f"after_end={integ.orders_after_data_end_date} | {elapsed:.2f}s"
            )
        return

    biz_id = args.business
    map_path = args.mapping if args.mapping is not None else (GOLD_DIR / f"{biz_id}.yaml")
    mapping = load_mapping_yaml(map_path)
    revs: dict[str, int] = {}

    if args.no_cache:
        raw = load_business_cached(args.split, biz_id)
        t0 = time.perf_counter()
        canon = build_canonical(raw, mapping)
        elapsed = time.perf_counter() - t0
        report = canon.report
        db_path: Path | None = None
        if args.revenue:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_db = Path(tmp_dir) / "temp.duckdb"
                write_duckdb(canon, tmp_db)
                for ym in args.revenue:
                    revs[ym] = query_monthly_revenue(tmp_db, ym)
    else:
        t0 = time.perf_counter()
        db_path, report = load_or_build_canonical(args.split, biz_id, mapping)
        elapsed = time.perf_counter() - t0
        for ym in args.revenue:
            revs[ym] = query_monthly_revenue(db_path, ym)

    print(_render_detailed_report(biz_id, db_path, report, revs, elapsed))


if __name__ == "__main__":
    main()

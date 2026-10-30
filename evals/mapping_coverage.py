"""Per-business mapping + ETL coverage report (train / public_test).

Reads only raw shop data and inferred mappings — never answers.jsonl.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datamate.config import settings
from datamate.etl import build_canonical, load_or_build_canonical
from datamate.io.profile import profile_business
from datamate.io.raw import load_business_cached
from datamate.llm import get_client
from datamate.mapping import infer_mapping
from datamate.mapping.fields import SIMPLE_ENTITIES
from datamate.mapping.model import Mapping
from evals.run_mapping import _list_businesses, _unique_run_dir, dump_mapping_yaml

_COVERAGE_ENTITIES: tuple[str, ...] = (
    "orders",
    "order_lines",
    "products",
    "unit_conversions",
    "customers",
    *SIMPLE_ENTITIES,
)


def _join_rate(left: list[Any], right: set[Any]) -> float | None:
    keys = [k for k in left if k is not None]
    if not keys:
        return None
    return sum(1 for k in keys if k in right) / len(keys)


def _coverage_one(
    split: str,
    biz_id: str,
    client: Any,
    *,
    mappings_dir: Path | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "business_id": biz_id,
        "mapping_failed": False,
        "errors": [],
        "warnings": [],
        "rows": {},
        "join_rates": {},
        "dedupe_counts": {},
    }

    def _dump_mapping(mapping: Mapping | None) -> None:
        if mapping is None or mappings_dir is None:
            return
        mappings_dir.mkdir(parents=True, exist_ok=True)
        dump_mapping_yaml(mapping, mappings_dir / f"{biz_id}.yaml")

    try:
        raw = load_business_cached(split, biz_id)
        profile = profile_business(raw)
        result = infer_mapping(raw, profile, client)
    except Exception as exc:  # noqa: BLE001 — coverage must not abort the batch
        row["mapping_failed"] = True
        row["errors"] = [f"{type(exc).__name__}: {exc}"]
        for ent in _COVERAGE_ENTITIES:
            row["rows"][ent] = "-"
        return row

    for iss in result.issues:
        if iss.severity == "warning":
            row["warnings"].append(f"[{iss.step}] {iss.message}")
    if result.mapping is None or result.errors or any(
        i.severity == "error" for i in result.issues
    ):
        row["mapping_failed"] = True
        row["errors"] = list(result.errors) + [
            i.message for i in result.issues if i.severity == "error"
        ]
        for ent in _COVERAGE_ENTITIES:
            row["rows"][ent] = "-"
        _dump_mapping(result.mapping)
        return row

    mapping = result.mapping
    _dump_mapping(mapping)
    try:
        db_path, report = load_or_build_canonical(split, biz_id, mapping)
        del db_path
        # Prefer report rows; fall back to rebuilding frames if needed.
        rows_per = dict(report.rows_per_table)
        dedupe = dict(report.dedupe_counts)
        warnings_etl = list(report.warnings)
        canon = build_canonical(raw, mapping)
    except Exception as exc:  # noqa: BLE001
        row["mapping_failed"] = True
        row["errors"] = [f"ETL: {type(exc).__name__}: {exc}"]
        for ent in _COVERAGE_ENTITIES:
            row["rows"][ent] = "-"
        return row

    row["warnings"].extend(warnings_etl)
    row["dedupe_counts"] = dedupe
    for ent in _COVERAGE_ENTITIES:
        if ent in mapping.tables:
            row["rows"][ent] = rows_per.get(ent, 0)
        else:
            row["rows"][ent] = "-"

    product_keys: set[Any] = set()
    if not canon.products.empty:
        product_keys.update(canon.products["product_key"].tolist())
    if not canon.order_lines.empty:
        product_keys.update(canon.order_lines["product_key"].tolist())
    product_keys.discard(None)
    order_ids: set[Any] = set()
    if not canon.orders.empty:
        order_ids.update(canon.orders["order_id"].tolist())
    order_ids.discard(None)

    join_specs: list[tuple[str, str, set[Any]]] = [
        ("purchase_lines", "product_key", product_keys),
        ("lots", "product_key", product_keys),
        ("sale_lots", "product_key", product_keys),
        ("stock_snapshot", "product_key", product_keys),
        ("stock_moves", "product_key", product_keys),
        ("price_history", "product_key", product_keys),
        ("drug_info", "product_key", product_keys),
        ("payments", "order_id", order_ids),
        ("debts", "order_id", order_ids),
        ("deliveries", "order_id", order_ids),
        ("sale_lots", "order_id", order_ids),
    ]
    for ent, col, right in join_specs:
        frame = getattr(canon, ent, None)
        if frame is None or frame.empty or col not in frame.columns:
            continue
        rate = _join_rate(frame[col].tolist(), right)
        if rate is not None:
            row["join_rates"][f"{ent}.{col}"] = round(rate, 4)

    return row


def _write_summary_md(path: Path, rows: list[dict[str, Any]], split: str) -> None:
    lines: list[str] = [
        f"# Mapping coverage — `{split}`",
        "",
        f"Generated: {datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d %H:%M:%SZ')}",
        "",
        "## Rows per entity",
        "",
    ]
    header = ["business_id", *_COVERAGE_ENTITIES]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows:
        cells = [row["business_id"]]
        for ent in _COVERAGE_ENTITIES:
            val = row["rows"].get(ent, "-")
            cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")

    failed = [r for r in rows if r["mapping_failed"]]
    lines.extend(["", "## mapping_failed", ""])
    if not failed:
        lines.append("(none)")
    else:
        for r in failed:
            lines.append(f"- `{r['business_id']}`: {r['errors']}")

    lines.extend(["", "## Warnings / joins / dedupe", ""])
    for r in rows:
        if r["mapping_failed"]:
            continue
        if not (r["warnings"] or r["join_rates"] or r["dedupe_counts"]):
            continue
        lines.append(f"### `{r['business_id']}`")
        for w in r["warnings"][:30]:
            lines.append(f"- warning: {w}")
        if len(r["warnings"]) > 30:
            lines.append(f"- … {len(r['warnings']) - 30} more warnings")
        for k, v in sorted(r["join_rates"].items()):
            lines.append(f"- join `{k}`: {v:.1%}")
        for k, v in sorted(r["dedupe_counts"].items()):
            lines.append(f"- dedupe `{k}`: {v}")
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "public_test"), required=True)
    parser.add_argument(
        "--businesses",
        default="",
        help="Comma-separated business ids (default: all under the split)",
    )
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)

    if args.businesses.strip():
        biz_ids = [b.strip() for b in args.businesses.split(",") if b.strip()]
    else:
        biz_ids = _list_businesses(args.split)
        if args.split == "public_test":
            # public shops are often named dev_* — also scan those.
            data_dir = settings.split_dir(args.split) / "data"
            if data_dir.is_dir():
                extra = sorted(
                    p.name
                    for p in data_dir.iterdir()
                    if p.is_dir() and (p.name.startswith("biz_") or p.name.startswith("dev_"))
                )
                biz_ids = extra or biz_ids

    if not biz_ids:
        print(f"No businesses found for split={args.split}", file=sys.stderr)
        return 1

    client = get_client()
    ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = _unique_run_dir(settings.runs_dir, f"{ts}_coverage")
    run_dir.mkdir(parents=True, exist_ok=True)
    mappings_dir = run_dir / "mappings"

    rows: list[dict[str, Any]] = []
    workers = max(1, args.workers)
    if workers == 1:
        for biz_id in biz_ids:
            print(f"coverage {biz_id} …", flush=True)
            rows.append(
                _coverage_one(
                    args.split, biz_id, client, mappings_dir=mappings_dir
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {
                pool.submit(
                    _coverage_one,
                    args.split,
                    biz_id,
                    client,
                    mappings_dir=mappings_dir,
                ): biz_id
                for biz_id in biz_ids
            }
            for fut in as_completed(futs):
                biz_id = futs[fut]
                print(f"coverage {biz_id} done", flush=True)
                rows.append(fut.result())
        rows.sort(key=lambda r: r["business_id"])

    (run_dir / "summary.json").write_text(
        json.dumps({"split": args.split, "businesses": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_summary_md(run_dir / "summary.md", rows, args.split)
    print(f"Wrote {run_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

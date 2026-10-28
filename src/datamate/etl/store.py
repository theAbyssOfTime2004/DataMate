from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from datamate.config import settings
from datamate.etl.build import CanonicalData, EtlReport, build_canonical
from datamate.etl.schema import DECIMAL_COLUMNS, DECIMAL_SQL_TYPE, TABLE_COLUMNS, TABLE_DDLS
from datamate.io.paths import business_dir
from datamate.io.raw import _compute_business_cache_key, load_business_cached
from datamate.mapping.model import Mapping, mapping_fingerprint

ETL_VERSION = "1.2"

_COLUMN_SQL_TYPES: dict[str, dict[str, str]] = {
    "orders": {
        "order_id": "VARCHAR",
        "order_date": "DATE",
        "order_ts": "TIMESTAMP",
        "status": "VARCHAR",
        "status_raw": "VARCHAR",
        "has_return": "BOOLEAN",
        "customer_key": "VARCHAR",
        "customer_id": "VARCHAR",
        "customer_name": "VARCHAR",
        "is_walk_in": "BOOLEAN",
        "payment_method": "VARCHAR",
        "channel": "VARCHAR",
        "employee_id": "VARCHAR",
        "order_subtotal": "BIGINT",
        "order_discount": "BIGINT",
        "shipping_fee": "BIGINT",
        "order_payable": "BIGINT",
    },
    "order_lines": {
        "order_id": "VARCHAR",
        "line_no": "INTEGER",
        "product_key": "VARCHAR",
        "product_id": "VARCHAR",
        "product_name": "VARCHAR",
        "qty": "DECIMAL(18,4)",
        "unit": "VARCHAR",
        "qty_base": "DECIMAL(18,4)",
        "unit_price": "BIGINT",
        "line_discount": "BIGINT",
        "line_amount": "BIGINT",
        "cost_per_unit": "BIGINT",
        "is_return_line": "BOOLEAN",
        "src_file": "VARCHAR",
        "src_sheet": "VARCHAR",
        "src_row": "INTEGER",
    },
    "products": {
        "product_key": "VARCHAR",
        "product_id": "VARCHAR",
        "product_name": "VARCHAR",
        "base_unit": "VARCHAR",
        "category": "VARCHAR",
        "unit_cost": "BIGINT",
    },
    "unit_conversions": {
        "product_key": "VARCHAR",
        "unit": "VARCHAR",
        "base_unit": "VARCHAR",
        "factor": "DECIMAL(18,4)",
    },
    "customers": {
        "customer_key": "VARCHAR",
        "customer_id": "VARCHAR",
        "customer_name": "VARCHAR",
        "segment": "VARCHAR",
    },
}


def _prepare_df_for_duckdb(df: pd.DataFrame, cols: tuple[str, ...]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=list(cols))
    records: list[dict[str, Any]] = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        rec: dict[str, Any] = {}
        for c in cols:
            val = row[c]
            if val is None or (isinstance(val, float) and pd.isna(val)):
                rec[c] = None
            elif isinstance(val, Decimal):
                rec[c] = format(val, "f")
            elif c in DECIMAL_COLUMNS:
                rec[c] = str(val)
            else:
                rec[c] = val
        records.append(rec)
    out = pd.DataFrame(records, columns=list(cols))
    for c in cols:
        out[c] = out[c].astype(object)
    return out


def write_duckdb(data: CanonicalData, path: Path | str) -> None:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    table_dfs: dict[str, pd.DataFrame] = {
        "orders": data.orders,
        "order_lines": data.order_lines,
        "products": data.products,
        "unit_conversions": data.unit_conversions,
        "customers": data.customers,
    }

    conn = duckdb.connect(str(db_path))
    try:
        for tname, ddl in TABLE_DDLS.items():
            conn.execute(ddl)
            cols = TABLE_COLUMNS[tname]
            src_df = _prepare_df_for_duckdb(table_dfs[tname], cols)
            if src_df.empty:
                continue
            conn.register("_staging_df", src_df)
            cast_exprs = [
                (
                    f'CAST("{c}" AS '
                    f'{DECIMAL_SQL_TYPE if c in DECIMAL_COLUMNS else _COLUMN_SQL_TYPES[tname][c]}) '
                    f'AS "{c}"'
                )
                for c in cols
            ]
            select_sql = ", ".join(cast_exprs)
            conn.execute(f"INSERT INTO {tname} SELECT {select_sql} FROM _staging_df")
            conn.unregister("_staging_df")
    finally:
        conn.close()


def _canonical_cache_key(split: str, business_id: str, mapping: Mapping) -> str:
    bdir = business_dir(split, business_id)
    raw_key = _compute_business_cache_key(bdir)
    map_fp = mapping_fingerprint(mapping)
    hasher = hashlib.sha256()
    hasher.update(f"raw={raw_key}\nmap={map_fp}\netl={ETL_VERSION}\n".encode())
    return hasher.hexdigest()


def load_or_build_canonical(
    split: str, business_id: str, mapping: Mapping
) -> tuple[Path, EtlReport]:
    key = _canonical_cache_key(split, business_id, mapping)
    cache_dir = settings.cache_dir / "canonical" / split / business_id
    db_path = cache_dir / f"{key}.duckdb"
    report_path = cache_dir / f"{key}.report.json"

    if db_path.is_file() and report_path.is_file():
        report = EtlReport.model_validate(
            json.loads(report_path.read_text(encoding="utf-8"))
        )
        return db_path, report

    raw = load_business_cached(split, business_id)
    data = build_canonical(raw, mapping)
    cache_dir.mkdir(parents=True, exist_ok=True)
    write_duckdb(data, db_path)
    report_path.write_text(
        json.dumps(data.report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return db_path, data.report

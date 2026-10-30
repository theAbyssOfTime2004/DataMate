"""Generic ETL loader for simple (non-order) canonical entities."""

from __future__ import annotations

import unicodedata
from collections import Counter
from typing import Any

import pandas as pd

from datamate.etl.build import (
    MappingError,
    _line_product_key,
    _records_to_df,
    _resolve_customer_fields,
    _resolve_sources,
    _StatTracker,
    _strip_str,
    _validate_mapped_columns,
)
from datamate.etl.schema import TABLE_COLUMNS
from datamate.io.raw import BusinessRaw
from datamate.mapping.fields import FIELD_TYPES, SIMPLE_ENTITIES, SIMPLE_REQUIRED
from datamate.mapping.model import Entity, Mapping, TableMapping
from datamate.parse import (
    normalize_label,
    parse_date,
    parse_decimal,
    parse_money,
    parse_qty,
    parse_timestamp,
    year_hint,
)

_BOOL_TRUE = frozenset({"true", "1", "x", "co", "yes", "ke don", "rx"})
_BOOL_FALSE = frozenset({"false", "0", "khong", "no", "khong ke don", "otc"})


def _strip_accents(text: str) -> str:
    nfd = unicodedata.normalize("NFD", text.lower())
    without_marks = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
    return without_marks.replace("đ", "d")


def _parse_bool(raw: Any) -> bool | None:
    text = _strip_str(raw)
    if text is None:
        return None
    norm = normalize_label(text)
    if norm is None:
        return None
    folded = _strip_accents(norm)
    if folded in _BOOL_TRUE:
        return True
    if folded in _BOOL_FALSE:
        return False
    return None


def _parse_field_value(
    field_type: str,
    raw: Any,
    *,
    default_year: int | None,
) -> Any:
    if field_type == "date":
        return parse_date(raw if isinstance(raw, str) else _strip_str(raw), default_year=default_year)
    if field_type == "timestamp":
        return parse_timestamp(
            raw if isinstance(raw, str) else _strip_str(raw), default_year=default_year
        )
    if field_type == "money":
        return parse_money(raw if isinstance(raw, str) else _strip_str(raw))
    if field_type == "qty":
        parsed = parse_qty(raw if isinstance(raw, str) else _strip_str(raw))
        if parsed is None:
            return None
        return parsed[0]
    if field_type == "decimal":
        return parse_decimal(raw if isinstance(raw, str) else _strip_str(raw))
    if field_type == "bool":
        return _parse_bool(raw)
    if field_type in {"text", "id"}:
        return _strip_str(raw)
    return _strip_str(raw)


def _qty_unit_from_raw(raw: Any) -> tuple[Any, str | None]:
    parsed = parse_qty(raw if isinstance(raw, str) else _strip_str(raw))
    if parsed is None:
        return None, None
    return parsed[0], parsed[1]


def load_simple_entity(
    entity: Entity,
    raw: BusinessRaw,
    mapping: Mapping,
    tracker: _StatTracker,
    warnings: list[str],
    dedupe_counts: Counter[str],
    *,
    key_by_name: dict[str, str],
    walk_in_norm_set: set[str],
) -> pd.DataFrame:
    """Load one simple entity into a DataFrame with TABLE_COLUMNS[entity] shape."""
    cols_out = TABLE_COLUMNS[entity]
    tm = mapping.tables.get(entity)
    if tm is None or not tm.sources:
        return _records_to_df([], cols_out)

    try:
        chunks = _resolve_sources(raw, entity, tm.sources, warnings)
        _validate_mapped_columns(entity, tm, chunks)
    except MappingError as exc:
        warnings.append(f"{entity}: {exc}")
        return _records_to_df([], cols_out)

    field_types = FIELD_TYPES[entity]
    mapped = tm.columns
    required = SIMPLE_REQUIRED.get(entity, set())

    purchase_status_norm: dict[str, str] = {}
    for key, value in mapping.purchase_status_map.items():
        nk = normalize_label(key)
        if nk is not None:
            purchase_status_norm[nk] = value

    rows: list[dict[str, Any]] = []
    for chunk in chunks:
        for s_row in chunk.df.to_dict("records"):
            f_name = str(s_row.get("__file") or "")
            s_name = s_row.get("__sheet")
            sheet = s_name if isinstance(s_name, str) else None
            src_row = s_row.get("__row")
            yh = year_hint(sheet, f_name)

            parsed: dict[str, Any] = {c: None for c in cols_out}
            parsed["src_file"] = f_name or None
            parsed["src_sheet"] = sheet
            parsed["src_row"] = int(src_row) if isinstance(src_row, int) else src_row

            for canon_field, raw_col in mapped.items():
                ftype = field_types.get(canon_field, "text")
                raw_val = s_row.get(raw_col)
                if canon_field == "qty" and "unit" not in mapped and entity in {
                    "purchase_lines",
                    "sale_lots",
                    "stock_snapshot",
                    "stock_moves",
                    "lots",
                    "deliveries",
                }:
                    qty_val, unit_from_qty = _qty_unit_from_raw(raw_val)
                    ok = qty_val is not None
                    tracker.record(f"{entity}.{canon_field}", raw_val, ok)
                    parsed[canon_field] = qty_val
                    if parsed.get("unit") is None and unit_from_qty is not None:
                        parsed["unit"] = unit_from_qty
                    continue
                value = _parse_field_value(ftype, raw_val, default_year=yh)
                tracker.record(f"{entity}.{canon_field}", raw_val, value is not None)
                parsed[canon_field] = value

            # product_key
            if "product_key" in cols_out:
                parsed["product_key"] = _line_product_key(
                    parsed.get("product_id"),
                    parsed.get("product_name"),
                    key_by_name,
                )

            # customer_key
            if "customer_key" in cols_out:
                ckey, _is_walk = _resolve_customer_fields(
                    parsed.get("customer_id"),
                    parsed.get("customer_name"),
                    mapping,
                    walk_in_norm_set,
                )
                parsed["customer_key"] = ckey

            # purchases.is_cancelled from status map
            if entity == "purchases":
                status_raw = parsed.get("status_raw")
                mapped_st = None
                if isinstance(status_raw, str):
                    nk = normalize_label(status_raw)
                    if nk is not None:
                        mapped_st = purchase_status_norm.get(nk)
                parsed["is_cancelled"] = mapped_st == "cancelled"

            # as_of_date default from mapping.data_end_date
            if (
                "as_of_date" in cols_out
                and parsed.get("as_of_date") is None
                and mapping.data_end_date is not None
            ):
                parsed["as_of_date"] = mapping.data_end_date

            # line_no as int when decimal/int-like
            if "line_no" in cols_out and parsed.get("line_no") is not None:
                ln = parsed["line_no"]
                if hasattr(ln, "to_integral_value"):
                    try:
                        parsed["line_no"] = int(ln)
                    except (ValueError, TypeError, ArithmeticError):
                        parsed["line_no"] = None

            if required and any(parsed.get(f) is None for f in required):
                dedupe_counts[f"{entity}.missing_required"] += 1
                continue

            rows.append(parsed)

    return _records_to_df(rows, cols_out)


def _source_keys(tm: TableMapping) -> list[tuple[str, str | None, int]]:
    return [(s.file, s.sheet, s.header_row) for s in tm.sources]


def _is_flat_purchase_sheet(mapping: Mapping) -> bool:
    purchases_tm = mapping.tables.get("purchases")
    lines_tm = mapping.tables.get("purchase_lines")
    if purchases_tm is None or lines_tm is None:
        return False
    if not purchases_tm.sources or not lines_tm.sources:
        return False
    return _source_keys(purchases_tm) == _source_keys(lines_tm)


def _dedupe_purchases_by_receipt(
    df: pd.DataFrame,
    dedupe_counts: Counter[str],
) -> pd.DataFrame:
    """Keep one row per receipt_id, filling each column with the first non-null value."""
    if df.empty or "receipt_id" not in df.columns:
        return df
    cols = list(df.columns)
    by_rid: dict[Any, dict[str, Any]] = {}
    order: list[Any] = []
    dropped = 0
    for row in df.to_dict("records"):
        rid = row.get("receipt_id")
        if rid is None:
            # Unkeyed rows stay as-is under a unique sentinel.
            key = object()
            by_rid[key] = row
            order.append(key)
            continue
        if rid not in by_rid:
            by_rid[rid] = dict(row)
            order.append(rid)
            continue
        dropped += 1
        existing = by_rid[rid]
        for c in cols:
            if existing.get(c) is None and row.get(c) is not None:
                existing[c] = row[c]
    if dropped:
        dedupe_counts["purchases.flat_dedupe"] += dropped
    return _records_to_df([by_rid[k] for k in order], tuple(cols))


def _recompute_purchase_totals_from_lines(
    purchases: pd.DataFrame,
    purchase_lines: pd.DataFrame,
) -> pd.DataFrame:
    if purchases.empty or purchase_lines.empty:
        return purchases
    if "receipt_id" not in purchases.columns or "receipt_id" not in purchase_lines.columns:
        return purchases
    if "line_amount" not in purchase_lines.columns:
        return purchases
    sums: dict[Any, int] = {}
    for row in purchase_lines.to_dict("records"):
        rid = row.get("receipt_id")
        amt = row.get("line_amount")
        if rid is None or amt is None:
            continue
        sums[rid] = sums.get(rid, 0) + int(amt)
    records = purchases.to_dict("records")
    for row in records:
        rid = row.get("receipt_id")
        if rid in sums:
            row["total_amount"] = sums[rid]
    return _records_to_df(records, tuple(purchases.columns))


def load_all_simple_tables(
    raw: BusinessRaw,
    mapping: Mapping,
    tracker: _StatTracker,
    warnings: list[str],
    dedupe_counts: Counter[str],
    *,
    key_by_name: dict[str, str],
    walk_in_norm_set: set[str],
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for entity in SIMPLE_ENTITIES:
        out[entity] = load_simple_entity(
            entity,
            raw,
            mapping,
            tracker,
            warnings,
            dedupe_counts,
            key_by_name=key_by_name,
            walk_in_norm_set=walk_in_norm_set,
        )

    if _is_flat_purchase_sheet(mapping):
        out["purchases"] = _dedupe_purchases_by_receipt(out["purchases"], dedupe_counts)
        purchases_tm = mapping.tables["purchases"]
        lines_tm = mapping.tables["purchase_lines"]
        total_col = purchases_tm.columns.get("total_amount")
        line_col = lines_tm.columns.get("line_amount")
        if total_col is not None and total_col == line_col:
            out["purchases"] = _recompute_purchase_totals_from_lines(
                out["purchases"], out["purchase_lines"]
            )

    return out

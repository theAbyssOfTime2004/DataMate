from __future__ import annotations

import datetime
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from datamate.etl.schema import (
    CUSTOMERS_COLUMNS,
    ORDER_LINES_COLUMNS,
    ORDERS_COLUMNS,
    PRODUCTS_COLUMNS,
    UNIT_CONVERSIONS_COLUMNS,
)
from datamate.io.raw import BusinessRaw, RawTable
from datamate.mapping.model import Entity, JoinSpec, Mapping, SourceSpec, TableMapping
from datamate.parse import (
    normalize_label,
    parse_date,
    parse_decimal,
    parse_money,
    parse_qty,
    parse_timestamp,
    year_hint,
)


class MappingError(ValueError):
    """Raised when a mapping references missing tables or columns in BusinessRaw."""


class FieldParseStat(BaseModel):
    non_null: int = 0
    parsed: int = 0
    rate: float = 1.0
    examples: list[str] = Field(default_factory=list)


class IntegrityStats(BaseModel):
    orphan_line_orders: int = 0
    orders_without_lines: int = 0
    subtotal_checked_orders: int = 0
    subtotal_matched_orders: int = 0
    subtotal_match_share: float | None = None
    min_order_date: datetime.date | None = None
    max_order_date: datetime.date | None = None
    orders_after_data_end_date: int = 0


class EtlReport(BaseModel):
    rows_per_table: dict[str, int] = Field(default_factory=dict)
    fill_counts: dict[str, int] = Field(default_factory=dict)
    dedupe_counts: dict[str, int] = Field(default_factory=dict)
    parse_stats: dict[str, FieldParseStat] = Field(default_factory=dict)
    unmapped_statuses: dict[str, int] = Field(default_factory=dict)
    conflicts: dict[str, int] = Field(default_factory=dict)
    status_overridden_by_return: int = 0
    integrity: IntegrityStats = Field(default_factory=IntegrityStats)
    warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


@dataclass
class CanonicalData:
    orders: pd.DataFrame
    order_lines: pd.DataFrame
    products: pd.DataFrame
    unit_conversions: pd.DataFrame
    customers: pd.DataFrame
    report: EtlReport


@dataclass
class _ResolvedChunk:
    raw_table: RawTable
    df: pd.DataFrame


def _available_tables_summary(raw: BusinessRaw) -> str:
    items: list[str] = []
    for rt in raw.tables:
        srcs = ", ".join(
            f"{s.file}" + (f"!{s.sheet}" if s.sheet is not None else "")
            for s in rt.sources
        )
        items.append(f"{rt.name} (sources: [{srcs}], columns: {rt.columns})")
    return "; ".join(items) if items else "<no tables>"


def _resolve_sources(
    raw: BusinessRaw,
    entity: Entity,
    specs: list[SourceSpec],
    warnings: list[str],
) -> list[_ResolvedChunk]:
    chunks: list[_ResolvedChunk] = []
    for spec in specs:
        matched_any = False
        for rt in raw.tables:
            matching_sheets: list[str | None] = []
            for sref in rt.sources:
                if sref.file != spec.file:
                    continue
                if spec.sheet in (None, "*") or sref.sheet == spec.sheet:
                    matched_any = True
                    matching_sheets.append(sref.sheet)
                    if spec.header_row != sref.header_row:
                        w = (
                            f"{entity}: source {spec.file!r} (sheet={sref.sheet!r}) "
                            f"mapping header_row={spec.header_row} != P2 header_row={sref.header_row}"
                        )
                        if w not in warnings:
                            warnings.append(w)
            if matching_sheets:
                mask = rt.df["__file"] == spec.file
                if spec.sheet not in (None, "*"):
                    mask = mask & (rt.df["__sheet"] == spec.sheet)
                sub_df = rt.df.loc[mask].copy().reset_index(drop=True)
                chunks.append(_ResolvedChunk(raw_table=rt, df=sub_df))
        if not matched_any:
            raise MappingError(
                f"Entity {entity!r} source (file={spec.file!r}, sheet={spec.sheet!r}) "
                f"not found in BusinessRaw. Available tables: {_available_tables_summary(raw)}"
            )
    return chunks


def _validate_mapped_columns(
    entity: Entity,
    tm: TableMapping,
    chunks: list[_ResolvedChunk],
) -> None:
    if not chunks:
        return
    for canonical_field, raw_col in tm.columns.items():
        for chunk in chunks:
            if raw_col not in chunk.raw_table.columns:
                raise MappingError(
                    f"Entity {entity!r} field {canonical_field!r} maps to missing column "
                    f"{raw_col!r} in table {chunk.raw_table.name!r}. "
                    f"Available columns: {chunk.raw_table.columns}"
                )


def _records_to_df(records: list[dict[str, Any]], cols: tuple[str, ...]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame({c: pd.Series([], dtype=object) for c in cols})
    series_dict: dict[str, pd.Series] = {}
    for c in cols:
        series_dict[c] = pd.Series([r.get(c) for r in records], dtype=object)
    return pd.DataFrame(series_dict)


def _strip_str(val: Any) -> str | None:
    if not isinstance(val, str):
        return None
    s = val.replace("\xa0", " ").strip()
    return s if s else None


class _StatTracker:
    def __init__(self) -> None:
        self.stats: dict[str, FieldParseStat] = {}

    def record(self, key: str, raw_val: str | None, parsed_ok: bool) -> None:
        raw_clean = _strip_str(raw_val)
        if raw_clean is None:
            return
        st = self.stats.get(key)
        if st is None:
            st = FieldParseStat()
            self.stats[key] = st
        st.non_null += 1
        if parsed_ok:
            st.parsed += 1
        elif len(st.examples) < 5 and raw_val not in st.examples:
            assert raw_val is not None
            st.examples.append(raw_val)

    def finalize(self) -> dict[str, FieldParseStat]:
        for st in self.stats.values():
            st.rate = round(st.parsed / st.non_null, 6) if st.non_null > 0 else 1.0
        return self.stats


def _parse_int_id(raw: str | None) -> int | None:
    dec = parse_decimal(raw)
    if dec is None:
        return None
    integral = dec.to_integral_value()
    if dec != integral:
        return None
    return int(integral)


def _build_products(
    raw: BusinessRaw,
    mapping: Mapping,
    tracker: _StatTracker,
    warnings: list[str],
) -> pd.DataFrame:
    tm = mapping.tables.get("products")
    if tm is None or not tm.sources:
        return _records_to_df([], PRODUCTS_COLUMNS)

    chunks = _resolve_sources(raw, "products", tm.sources, warnings)
    _validate_mapped_columns("products", tm, chunks)
    df = pd.concat([c.df for c in chunks], ignore_index=True) if chunks else pd.DataFrame()
    if df.empty:
        return _records_to_df([], PRODUCTS_COLUMNS)

    cols = tm.columns
    rows: list[dict[str, Any]] = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        pid_raw = row[cols["product_id"]] if "product_id" in cols else None
        pname_raw = row[cols["product_name"]] if "product_name" in cols else None
        bunit_raw = row[cols["base_unit"]] if "base_unit" in cols else None
        cat_raw = row[cols["category"]] if "category" in cols else None
        ucost_raw = row[cols["unit_cost"]] if "unit_cost" in cols else None

        pid = _strip_str(pid_raw)
        tracker.record("products.product_id", pid_raw, pid is not None)
        pname = _strip_str(pname_raw)
        tracker.record("products.product_name", pname_raw, pname is not None)
        bunit = _strip_str(bunit_raw)
        tracker.record("products.base_unit", bunit_raw, bunit is not None)
        cat = _strip_str(cat_raw)
        tracker.record("products.category", cat_raw, cat is not None)
        ucost = parse_money(ucost_raw)
        tracker.record("products.unit_cost", ucost_raw, ucost is not None)

        pkey = pid if pid is not None else normalize_label(pname)
        rows.append(
            {
                "product_key": pkey,
                "product_id": pid,
                "product_name": pname,
                "base_unit": bunit,
                "category": cat,
                "unit_cost": ucost,
            }
        )
    return _records_to_df(rows, PRODUCTS_COLUMNS)


def _build_customers(
    raw: BusinessRaw,
    mapping: Mapping,
    tracker: _StatTracker,
    warnings: list[str],
) -> pd.DataFrame:
    tm = mapping.tables.get("customers")
    if tm is None or not tm.sources:
        return _records_to_df([], CUSTOMERS_COLUMNS)

    chunks = _resolve_sources(raw, "customers", tm.sources, warnings)
    _validate_mapped_columns("customers", tm, chunks)
    df = pd.concat([c.df for c in chunks], ignore_index=True) if chunks else pd.DataFrame()
    if df.empty:
        return _records_to_df([], CUSTOMERS_COLUMNS)

    cols = tm.columns
    rows: list[dict[str, Any]] = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        cid_raw = row[cols["customer_id"]] if "customer_id" in cols else None
        cname_raw = row[cols["customer_name"]] if "customer_name" in cols else None
        seg_raw = row[cols["segment"]] if "segment" in cols else None

        cid = _strip_str(cid_raw)
        tracker.record("customers.customer_id", cid_raw, cid is not None)
        cname = _strip_str(cname_raw)
        tracker.record("customers.customer_name", cname_raw, cname is not None)
        seg = _strip_str(seg_raw)
        tracker.record("customers.segment", seg_raw, seg is not None)

        ckey = (
            cid
            if mapping.customer_identity == "by_code" and cid is not None
            else normalize_label(cname)
        )
        rows.append(
            {
                "customer_key": ckey,
                "customer_id": cid,
                "customer_name": cname,
                "segment": seg,
            }
        )
    return _records_to_df(rows, CUSTOMERS_COLUMNS)


def _build_unit_conversions(
    raw: BusinessRaw,
    mapping: Mapping,
    tracker: _StatTracker,
    warnings: list[str],
    notes: list[str],
) -> pd.DataFrame:
    tm = mapping.tables.get("unit_conversions")
    if tm is None:
        return _records_to_df([], UNIT_CONVERSIONS_COLUMNS)
    if tm.source == "readme":
        notes.append("unit conversions in README: not parsed in P3")
        return _records_to_df([], UNIT_CONVERSIONS_COLUMNS)
    if not tm.sources:
        return _records_to_df([], UNIT_CONVERSIONS_COLUMNS)

    chunks = _resolve_sources(raw, "unit_conversions", tm.sources, warnings)
    _validate_mapped_columns("unit_conversions", tm, chunks)
    df = pd.concat([c.df for c in chunks], ignore_index=True) if chunks else pd.DataFrame()
    if df.empty:
        return _records_to_df([], UNIT_CONVERSIONS_COLUMNS)

    cols = tm.columns
    rows: list[dict[str, Any]] = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        pid_raw = row[cols["product_id"]] if "product_id" in cols else None
        pname_raw = row[cols["product_name"]] if "product_name" in cols else None
        unit_raw = row[cols["unit"]] if "unit" in cols else None
        bunit_raw = row[cols["base_unit"]] if "base_unit" in cols else None
        factor_raw = row[cols["factor"]] if "factor" in cols else None

        pid = _strip_str(pid_raw)
        tracker.record("unit_conversions.product_id", pid_raw, pid is not None)
        pname = _strip_str(pname_raw)
        tracker.record("unit_conversions.product_name", pname_raw, pname is not None)
        unit = _strip_str(unit_raw)
        tracker.record("unit_conversions.unit", unit_raw, unit is not None)
        bunit = _strip_str(bunit_raw)
        tracker.record("unit_conversions.base_unit", bunit_raw, bunit is not None)

        qty_parsed = parse_qty(factor_raw)
        factor: Decimal | None = qty_parsed[0] if qty_parsed is not None else parse_decimal(factor_raw)
        tracker.record("unit_conversions.factor", factor_raw, factor is not None)

        pkey = pid if pid is not None else normalize_label(pname)
        rows.append(
            {
                "product_key": pkey,
                "unit": unit,
                "base_unit": bunit,
                "factor": factor,
            }
        )
    return _records_to_df(rows, UNIT_CONVERSIONS_COLUMNS)


def _apply_joins(
    rows: list[dict[str, Any]],
    joins: dict[str, JoinSpec],
    entity_dfs: dict[Entity, pd.DataFrame],
) -> None:
    if not joins or not rows:
        return
    for target_field, jspec in joins.items():
        target_df = entity_dfs.get(jspec.table)
        if target_df is None or target_df.empty:
            continue
        source_field = jspec.field if jspec.field is not None else target_field
        if source_field not in target_df.columns:
            continue

        # Build lookup index from target_df
        local_keys = list(jspec.on.keys())
        remote_keys = [jspec.on[k] for k in local_keys]
        if any(rk not in target_df.columns for rk in remote_keys):
            continue

        lookup: dict[tuple[Any, ...], Any] = {}
        for t_idx in range(len(target_df)):
            t_row = target_df.iloc[t_idx]
            val = t_row[source_field]
            if val is None or (isinstance(val, float) and pd.isna(val)):
                continue
            key_tuple = tuple(_strip_str(str(t_row[rk])) if t_row[rk] is not None else None for rk in remote_keys)
            if all(k is not None for k in key_tuple) and key_tuple not in lookup:
                lookup[key_tuple] = val

        for r in rows:
            if r.get(target_field) is not None:
                continue
            l_tuple = tuple(_strip_str(str(r.get(lk))) if r.get(lk) is not None else None for lk in local_keys)
            if all(k is not None for k in l_tuple) and l_tuple in lookup:
                r[target_field] = lookup[l_tuple]


def _resolve_customer_fields(
    cid: str | None,
    cname: str | None,
    mapping: Mapping,
    walk_in_norm_set: set[str],
) -> tuple[str | None, bool]:
    identity_val = cid if mapping.customer_identity == "by_code" else cname
    norm_id = normalize_label(identity_val)
    if identity_val is None or norm_id is None or norm_id in walk_in_norm_set:
        return None, True
    if mapping.customer_identity == "by_code":
        return cid, False
    return norm_id, False


def build_canonical(raw: BusinessRaw, mapping: Mapping) -> CanonicalData:
    warnings: list[str] = []
    notes: list[str] = []
    tracker = _StatTracker()
    fill_counts: Counter[str] = Counter()
    dedupe_counts: Counter[str] = Counter()
    unmapped_statuses: Counter[str] = Counter()
    conflicts: Counter[str] = Counter()

    if mapping.return_rule == "return_table":
        notes.append("return_rule 'return_table': not supported in P3")

    products_df = _build_products(raw, mapping, tracker, warnings)
    customers_df = _build_customers(raw, mapping, tracker, warnings)
    unit_conv_df = _build_unit_conversions(raw, mapping, tracker, warnings, notes)

    entity_dfs: dict[Entity, pd.DataFrame] = {
        "products": products_df,
        "customers": customers_df,
        "unit_conversions": unit_conv_df,
    }

    lines_tm = mapping.tables["order_lines"]
    orders_tm = mapping.tables["orders"]

    line_chunks = _resolve_sources(raw, "order_lines", lines_tm.sources, warnings)
    _validate_mapped_columns("order_lines", lines_tm, line_chunks)
    if orders_tm.derived_from == "order_lines":
        _validate_mapped_columns("orders", orders_tm, line_chunks)

    if "order_id" not in lines_tm.columns:
        raise MappingError("Entity 'order_lines' must map canonical field 'order_id'")
    order_id_raw_col = lines_tm.columns["order_id"]

    # Map from raw order-level column -> list of canonical order fields using it
    order_level_raw_cols: dict[str, list[str]] = {}
    if orders_tm.derived_from == "order_lines":
        for o_field, o_raw_col in orders_tm.columns.items():
            if o_field == "order_id":
                continue
            order_level_raw_cols.setdefault(o_raw_col, []).append(o_field)

    # Step (c) and (d): per (__file, __sheet) block forward-fill and exact pre-fill dedupe
    kept_line_records: list[dict[str, Any]] = []
    seen_in_order_by_table: dict[str, set[tuple[str, tuple[str | None, ...]]]] = {}

    for chunk in line_chunks:
        rt = chunk.raw_table
        rt_data_cols = list(rt.columns)
        chunk_df = chunk.df
        if chunk_df.empty:
            continue

        seen_set = seen_in_order_by_table.setdefault(rt.name, set())

        # Group contiguous or same (__file, __sheet) blocks in source order
        prev_file: str | None = None
        prev_sheet: str | None = None
        prev_order_id: str | None = None
        prev_order_raw_vals: dict[str, str | None] = {}

        for r_idx in range(len(chunk_df)):
            s_row = chunk_df.iloc[r_idx]
            cur_file = s_row["__file"]
            cur_sheet = s_row["__sheet"]
            cur_row_no = int(s_row["__row"])

            if cur_file != prev_file or cur_sheet != prev_sheet:
                prev_file = cur_file
                prev_sheet = cur_sheet
                prev_order_id = None
                prev_order_raw_vals = {}

            # Pre-fill snapshot across ALL raw data columns of this source table
            pre_fill_tuple = tuple(
                s_row[c] if isinstance(s_row[c], str) else None for c in rt_data_cols
            )

            raw_oid_cell = s_row[order_id_raw_col] if isinstance(s_row[order_id_raw_col], str) else None
            if raw_oid_cell is None and prev_order_id is not None:
                resolved_oid_raw: str | None = prev_order_id
                fill_counts["order_id"] += 1
            else:
                resolved_oid_raw = raw_oid_cell

            same_order_as_prev = (
                resolved_oid_raw is not None
                and prev_order_id is not None
                and _strip_str(resolved_oid_raw) == _strip_str(prev_order_id)
            )

            filled_row_dict: dict[str, str | None] = {
                c: (s_row[c] if isinstance(s_row[c], str) else None) for c in rt_data_cols
            }
            filled_row_dict[order_id_raw_col] = resolved_oid_raw

            if orders_tm.derived_from == "order_lines":
                for raw_col_name, canonical_o_fields in order_level_raw_cols.items():
                    if raw_col_name not in filled_row_dict:
                        continue
                    cell_val = filled_row_dict[raw_col_name]
                    if cell_val is None and same_order_as_prev and prev_order_raw_vals.get(raw_col_name) is not None:
                        filled_row_dict[raw_col_name] = prev_order_raw_vals[raw_col_name]
                        for cf in canonical_o_fields:
                            fill_counts[cf] += 1

            # Update state for subsequent rows in the same (__file, __sheet)
            if resolved_oid_raw is not None:
                if not same_order_as_prev:
                    prev_order_raw_vals = {}
                prev_order_id = resolved_oid_raw
                for raw_col_name in order_level_raw_cols:
                    if filled_row_dict.get(raw_col_name) is not None:
                        prev_order_raw_vals[raw_col_name] = filled_row_dict[raw_col_name]

            # Step (d): §8 dedupe within the same order on pre-fill values across all raw data columns
            oid_clean = _strip_str(resolved_oid_raw)
            if oid_clean is not None:
                dedupe_key = (oid_clean, pre_fill_tuple)
                if dedupe_key in seen_set:
                    dedupe_counts[rt.name] += 1
                    continue
                seen_set.add(dedupe_key)

            kept_line_records.append(
                {
                    "__file": cur_file,
                    "__sheet": cur_sheet,
                    "__row": cur_row_no,
                    "raw": filled_row_dict,
                }
            )

    # Step (e), (f), (g): parse order_lines fields
    lcols = lines_tm.columns
    parsed_lines: list[dict[str, Any]] = []
    raw_lines_for_derived_orders: list[tuple[str, str | None, str, Decimal | None, dict[str, str | None]]] = []

    for rec in kept_line_records:
        f_name: str = rec["__file"]
        s_name: str | None = rec["__sheet"]
        r_no: int = rec["__row"]
        rdict: dict[str, str | None] = rec["raw"]

        oid_raw = rdict.get(lcols["order_id"]) if "order_id" in lcols else None
        oid = _strip_str(oid_raw)
        tracker.record("order_lines.order_id", oid_raw, oid is not None)

        lno_raw = rdict.get(lcols["line_no"]) if "line_no" in lcols else None
        lno = _parse_int_id(lno_raw) if lno_raw is not None else None
        tracker.record("order_lines.line_no", lno_raw, lno is not None)

        pid_raw = rdict.get(lcols["product_id"]) if "product_id" in lcols else None
        pid = _strip_str(pid_raw)
        tracker.record("order_lines.product_id", pid_raw, pid is not None)

        pname_raw = rdict.get(lcols["product_name"]) if "product_name" in lcols else None
        pname = _strip_str(pname_raw)
        tracker.record("order_lines.product_name", pname_raw, pname is not None)

        qty_raw = rdict.get(lcols["qty"]) if "qty" in lcols else None
        qty_parsed = parse_qty(qty_raw)
        qty_val: Decimal | None = qty_parsed[0] if qty_parsed is not None else None
        qty_embedded_unit: str | None = qty_parsed[1] if qty_parsed is not None else None
        tracker.record("order_lines.qty", qty_raw, qty_val is not None)

        unit_raw = rdict.get(lcols["unit"]) if "unit" in lcols else None
        unit_col_val = _strip_str(unit_raw)
        if "unit" in lcols:
            tracker.record("order_lines.unit", unit_raw, unit_col_val is not None)
        unit_val = unit_col_val if unit_col_val is not None else qty_embedded_unit

        qbase_raw = rdict.get(lcols["qty_base"]) if "qty_base" in lcols else None
        qbase_parsed = parse_qty(qbase_raw)
        qbase_val: Decimal | None = (
            qbase_parsed[0] if qbase_parsed is not None else parse_decimal(qbase_raw)
        )
        tracker.record("order_lines.qty_base", qbase_raw, qbase_val is not None)

        uprice_raw = rdict.get(lcols["unit_price"]) if "unit_price" in lcols else None
        uprice_val = parse_money(uprice_raw)
        tracker.record("order_lines.unit_price", uprice_raw, uprice_val is not None)

        ldisc_raw = rdict.get(lcols["line_discount"]) if "line_discount" in lcols else None
        ldisc_val = parse_money(ldisc_raw)
        tracker.record("order_lines.line_discount", ldisc_raw, ldisc_val is not None)

        lamt_raw = rdict.get(lcols["line_amount"]) if "line_amount" in lcols else None
        lamt_val = parse_money(lamt_raw)
        tracker.record("order_lines.line_amount", lamt_raw, lamt_val is not None)

        cpu_raw = rdict.get(lcols["cost_per_unit"]) if "cost_per_unit" in lcols else None
        cpu_val = parse_money(cpu_raw)
        tracker.record("order_lines.cost_per_unit", cpu_raw, cpu_val is not None)

        parsed_lines.append(
            {
                "order_id": oid,
                "line_no": lno,
                "product_key": None,
                "product_id": pid,
                "product_name": pname,
                "qty": qty_val,
                "unit": unit_val,
                "qty_base": qbase_val,
                "unit_price": uprice_val,
                "line_discount": ldisc_val,
                "line_amount": lamt_val,
                "cost_per_unit": cpu_val,
                "is_return_line": bool(qty_val is not None and qty_val < 0),
                "src_file": f_name,
                "src_sheet": s_name,
                "src_row": r_no,
            }
        )
        if oid is not None:
            raw_lines_for_derived_orders.append((oid, s_name, f_name, qty_val, rdict))

    # Step (f): joins on order_lines
    _apply_joins(parsed_lines, lines_tm.join, entity_dfs)

    # Step (g): product_key and sequential line_no per order
    line_seq_counter: Counter[str] = Counter()
    for pl in parsed_lines:
        pid = pl["product_id"]
        pname = pl["product_name"]
        pl["product_key"] = pid if pid is not None else normalize_label(pname)
        oid_key = pl["order_id"] or ""
        line_seq_counter[oid_key] += 1
        if pl["line_no"] is None:
            pl["line_no"] = line_seq_counter[oid_key]

    order_lines_df = _records_to_df(parsed_lines, ORDER_LINES_COLUMNS)

    # Step (h): build orders
    norm_status_map: dict[str, str] = {}
    for k, v in mapping.status_map.items():
        nk = normalize_label(k)
        if nk is not None:
            norm_status_map[nk] = v

    walk_in_norm_set: set[str] = {
        nk for v in mapping.walk_in_values if (nk := normalize_label(v)) is not None
    }

    orders_with_negative_qty: set[str] = {
        pl["order_id"]
        for pl in parsed_lines
        if pl["order_id"] is not None and pl["is_return_line"]
    }

    status_overridden_by_return = 0
    order_rows: list[dict[str, Any]] = []

    if orders_tm.derived_from == "order_lines":
        ocols = orders_tm.columns
        # Group lines by order_id preserving order of appearance
        lines_by_oid: dict[str, list[tuple[str | None, str, Decimal | None, dict[str, str | None]]]] = {}
        for oid, s_name, f_name, qty_val, rdict in raw_lines_for_derived_orders:
            lines_by_oid.setdefault(oid, []).append((s_name, f_name, qty_val, rdict))

        def _parse_order_field(
            field_name: str,
            raw_val: str | None,
            s_name: str | None,
            f_name: str,
            record_stat: bool = True,
        ) -> Any:
            if raw_val is None:
                return None
            if field_name == "order_date":
                val = parse_date(raw_val, default_year=year_hint(s_name, f_name))
                if record_stat:
                    tracker.record("orders.order_date", raw_val, val is not None)
                return val
            if field_name == "order_ts":
                val_ts = parse_timestamp(raw_val, default_year=year_hint(s_name, f_name))
                if record_stat:
                    tracker.record("orders.order_ts", raw_val, val_ts is not None)
                return val_ts
            if field_name in {"order_subtotal", "order_discount", "shipping_fee", "order_payable"}:
                val_m = parse_money(raw_val)
                if record_stat:
                    tracker.record(f"orders.{field_name}", raw_val, val_m is not None)
                return val_m
            val_s = _strip_str(raw_val)
            if record_stat:
                tracker.record(f"orders.{field_name}", raw_val, val_s is not None)
            return val_s

        derived_fields = [
            "order_date",
            "order_ts",
            "status",
            "customer_id",
            "customer_name",
            "payment_method",
            "channel",
            "employee_id",
            "order_subtotal",
            "order_discount",
            "shipping_fee",
            "order_payable",
        ]

        for oid, o_lines in lines_by_oid.items():
            tracker.record("orders.order_id", oid, True)

            resolved_fields: dict[str, Any] = {}
            for fname in derived_fields:
                if fname not in ocols:
                    resolved_fields[fname] = None
                    continue
                raw_col = ocols[fname]

                # Record parse stats across all lines of the order, and collect non-null parsed values on non-neg lines
                non_neg_vals: list[Any] = []
                any_vals: list[Any] = []
                for s_name, f_name, qty_val, rdict in o_lines:
                    rv = rdict.get(raw_col)
                    pv = _parse_order_field(fname, rv, s_name, f_name, record_stat=True)
                    if pv is not None:
                        any_vals.append(pv)
                        if qty_val is None or qty_val >= 0:
                            non_neg_vals.append(pv)

                chosen = non_neg_vals[0] if non_neg_vals else (any_vals[0] if any_vals else None)
                resolved_fields[fname] = chosen

                if len(non_neg_vals) > 1:
                    if fname == "status":
                        distinct_norm = {normalize_label(str(v)) for v in non_neg_vals}
                        if len(distinct_norm) > 1:
                            conflicts[fname] += 1
                    elif len(set(non_neg_vals)) > 1:
                        conflicts[fname] += 1

            status_raw = resolved_fields["status"]
            norm_st = normalize_label(status_raw) if status_raw is not None else None
            mapped_status: str | None = None
            if norm_st is not None:
                mapped_status = norm_status_map.get(norm_st)
                if mapped_status is None:
                    unmapped_statuses[status_raw] += 1

            has_ret = (mapped_status == "returned") or (
                mapping.return_rule == "negative_qty" and oid in orders_with_negative_qty
            )
            if has_ret and mapped_status in ("completed", "shipping"):
                mapped_status = "returned"
                status_overridden_by_return += 1

            cid = resolved_fields["customer_id"]
            cname = resolved_fields["customer_name"]
            ckey, is_walk_in = _resolve_customer_fields(cid, cname, mapping, walk_in_norm_set)

            order_rows.append(
                {
                    "order_id": oid,
                    "order_date": resolved_fields["order_date"],
                    "order_ts": resolved_fields["order_ts"],
                    "status": mapped_status,
                    "status_raw": status_raw,
                    "has_return": bool(has_ret),
                    "customer_key": ckey,
                    "customer_id": cid,
                    "customer_name": cname,
                    "is_walk_in": is_walk_in,
                    "payment_method": resolved_fields["payment_method"],
                    "channel": resolved_fields["channel"],
                    "employee_id": resolved_fields["employee_id"],
                    "order_subtotal": resolved_fields["order_subtotal"],
                    "order_discount": resolved_fields["order_discount"],
                    "shipping_fee": resolved_fields["shipping_fee"],
                    "order_payable": resolved_fields["order_payable"],
                }
            )
    else:
        order_chunks = _resolve_sources(raw, "orders", orders_tm.sources, warnings)
        _validate_mapped_columns("orders", orders_tm, order_chunks)
        ocols = orders_tm.columns
        for chunk in order_chunks:
            cdf = chunk.df
            for r_idx in range(len(cdf)):
                s_row = cdf.iloc[r_idx]
                f_name = s_row["__file"]
                s_name = s_row["__sheet"]
                yh = year_hint(s_name, f_name)

                oid_raw = s_row[ocols["order_id"]] if "order_id" in ocols else None
                oid = _strip_str(oid_raw)
                tracker.record("orders.order_id", oid_raw, oid is not None)

                odate_raw = s_row[ocols["order_date"]] if "order_date" in ocols else None
                odate = parse_date(odate_raw, default_year=yh)
                tracker.record("orders.order_date", odate_raw, odate is not None)

                ots_raw = s_row[ocols["order_ts"]] if "order_ts" in ocols else None
                ots = parse_timestamp(ots_raw, default_year=yh)
                tracker.record("orders.order_ts", ots_raw, ots is not None)

                status_raw_cell = s_row[ocols["status"]] if "status" in ocols else None
                status_raw = _strip_str(status_raw_cell)
                norm_st = normalize_label(status_raw) if status_raw is not None else None
                mapped_status = None
                if norm_st is not None:
                    mapped_status = norm_status_map.get(norm_st)
                    if mapped_status is None:
                        unmapped_statuses[status_raw] += 1
                tracker.record("orders.status", status_raw_cell, mapped_status is not None)

                cid_raw = s_row[ocols["customer_id"]] if "customer_id" in ocols else None
                cid = _strip_str(cid_raw)
                tracker.record("orders.customer_id", cid_raw, cid is not None)

                cname_raw = s_row[ocols["customer_name"]] if "customer_name" in ocols else None
                cname = _strip_str(cname_raw)
                tracker.record("orders.customer_name", cname_raw, cname is not None)

                pm_raw = s_row[ocols["payment_method"]] if "payment_method" in ocols else None
                pm = _strip_str(pm_raw)
                tracker.record("orders.payment_method", pm_raw, pm is not None)

                ch_raw = s_row[ocols["channel"]] if "channel" in ocols else None
                ch = _strip_str(ch_raw)
                tracker.record("orders.channel", ch_raw, ch is not None)

                emp_raw = s_row[ocols["employee_id"]] if "employee_id" in ocols else None
                emp = _strip_str(emp_raw)
                tracker.record("orders.employee_id", emp_raw, emp is not None)

                sub_raw = s_row[ocols["order_subtotal"]] if "order_subtotal" in ocols else None
                sub_val = parse_money(sub_raw)
                tracker.record("orders.order_subtotal", sub_raw, sub_val is not None)

                disc_raw = s_row[ocols["order_discount"]] if "order_discount" in ocols else None
                disc_val = parse_money(disc_raw)
                tracker.record("orders.order_discount", disc_raw, disc_val is not None)

                ship_raw = s_row[ocols["shipping_fee"]] if "shipping_fee" in ocols else None
                ship_val = parse_money(ship_raw)
                tracker.record("orders.shipping_fee", ship_raw, ship_val is not None)

                pay_raw = s_row[ocols["order_payable"]] if "order_payable" in ocols else None
                pay_val = parse_money(pay_raw)
                tracker.record("orders.order_payable", pay_raw, pay_val is not None)

                has_ret = (mapped_status == "returned") or (
                    mapping.return_rule == "negative_qty"
                    and oid is not None
                    and oid in orders_with_negative_qty
                )
                if has_ret and mapped_status in ("completed", "shipping"):
                    mapped_status = "returned"
                    status_overridden_by_return += 1

                ckey, is_walk_in = _resolve_customer_fields(cid, cname, mapping, walk_in_norm_set)

                order_rows.append(
                    {
                        "order_id": oid,
                        "order_date": odate,
                        "order_ts": ots,
                        "status": mapped_status,
                        "status_raw": status_raw,
                        "has_return": bool(has_ret),
                        "customer_key": ckey,
                        "customer_id": cid,
                        "customer_name": cname,
                        "is_walk_in": is_walk_in,
                        "payment_method": pm,
                        "channel": ch,
                        "employee_id": emp,
                        "order_subtotal": sub_val,
                        "order_discount": disc_val,
                        "shipping_fee": ship_val,
                        "order_payable": pay_val,
                    }
                )

        if orders_tm.join:
            _apply_joins(order_rows, orders_tm.join, entity_dfs)
            for orow in order_rows:
                ckey, is_walk_in = _resolve_customer_fields(
                    orow["customer_id"], orow["customer_name"], mapping, walk_in_norm_set
                )
                orow["customer_key"] = ckey
                orow["is_walk_in"] = is_walk_in

    orders_df = _records_to_df(order_rows, ORDERS_COLUMNS)

    # Step (i): Integrity checks
    known_order_ids = {r["order_id"] for r in order_rows if r["order_id"] is not None}
    line_order_ids = {pl["order_id"] for pl in parsed_lines if pl["order_id"] is not None}

    orphan_line_orders = len(line_order_ids - known_order_ids)
    orders_without_lines = len(known_order_ids - line_order_ids)

    line_sum_by_order: Counter[str] = Counter()
    orders_with_any_line: set[str] = set()
    for pl in parsed_lines:
        oid = pl["order_id"]
        if oid is not None and pl["line_amount"] is not None:
            orders_with_any_line.add(oid)
            line_sum_by_order[oid] += int(pl["line_amount"])

    subtotal_checked = 0
    subtotal_matched = 0
    subtotal_share: float | None = None
    if "order_subtotal" in orders_tm.columns:
        for orow in order_rows:
            oid = orow["order_id"]
            sub = orow["order_subtotal"]
            if oid is not None and sub is not None and oid in orders_with_any_line:
                subtotal_checked += 1
                if line_sum_by_order[oid] == int(sub):
                    subtotal_matched += 1
        if subtotal_checked > 0:
            subtotal_share = round(subtotal_matched / subtotal_checked, 6)

    valid_dates: list[datetime.date] = [
        orow["order_date"] for orow in order_rows if isinstance(orow["order_date"], datetime.date)
    ]
    min_date = min(valid_dates) if valid_dates else None
    max_date = max(valid_dates) if valid_dates else None
    after_end = 0
    if mapping.data_end_date is not None:
        after_end = sum(1 for d in valid_dates if d > mapping.data_end_date)

    # Also record orders.status parse stat for derived orders if status was mapped
    if orders_tm.derived_from == "order_lines" and "status" in orders_tm.columns:
        st_stat = tracker.stats.get("orders.status")
        if st_stat is not None:
            unmapped_total = sum(unmapped_statuses.values())
            st_stat.parsed = max(0, st_stat.non_null - unmapped_total)

    final_parse_stats = tracker.finalize()

    report = EtlReport(
        rows_per_table={
            "orders": len(orders_df),
            "order_lines": len(order_lines_df),
            "products": len(products_df),
            "unit_conversions": len(unit_conv_df),
            "customers": len(customers_df),
        },
        fill_counts=dict(sorted(fill_counts.items())),
        dedupe_counts=dict(sorted(dedupe_counts.items())),
        parse_stats=dict(sorted(final_parse_stats.items())),
        unmapped_statuses=dict(sorted(unmapped_statuses.items())),
        conflicts=dict(sorted(conflicts.items())),
        status_overridden_by_return=status_overridden_by_return,
        integrity=IntegrityStats(
            orphan_line_orders=orphan_line_orders,
            orders_without_lines=orders_without_lines,
            subtotal_checked_orders=subtotal_checked,
            subtotal_matched_orders=subtotal_matched,
            subtotal_match_share=subtotal_share,
            min_order_date=min_date,
            max_order_date=max_date,
            orders_after_data_end_date=after_end,
        ),
        warnings=warnings,
        notes=notes,
    )

    return CanonicalData(
        orders=orders_df,
        order_lines=order_lines_df,
        products=products_df,
        unit_conversions=unit_conv_df,
        customers=customers_df,
        report=report,
    )

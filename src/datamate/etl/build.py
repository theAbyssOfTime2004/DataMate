from __future__ import annotations

import datetime
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from datamate.etl.schema import (
    CUSTOMERS_COLUMNS,
    ORDER_ATTRIBUTES_COLUMNS,
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
    payments: pd.DataFrame = field(default_factory=pd.DataFrame)
    debts: pd.DataFrame = field(default_factory=pd.DataFrame)
    purchases: pd.DataFrame = field(default_factory=pd.DataFrame)
    purchase_lines: pd.DataFrame = field(default_factory=pd.DataFrame)
    suppliers: pd.DataFrame = field(default_factory=pd.DataFrame)
    stock_snapshot: pd.DataFrame = field(default_factory=pd.DataFrame)
    stock_moves: pd.DataFrame = field(default_factory=pd.DataFrame)
    lots: pd.DataFrame = field(default_factory=pd.DataFrame)
    sale_lots: pd.DataFrame = field(default_factory=pd.DataFrame)
    price_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    expenses: pd.DataFrame = field(default_factory=pd.DataFrame)
    employees: pd.DataFrame = field(default_factory=pd.DataFrame)
    deliveries: pd.DataFrame = field(default_factory=pd.DataFrame)
    drug_info: pd.DataFrame = field(default_factory=pd.DataFrame)
    order_attributes: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class _ResolvedChunk:
    raw_table: RawTable
    df: pd.DataFrame


_MAX_ATTRIBUTE_LEN = 200


def _available_tables_summary(raw: BusinessRaw) -> str:
    items: list[str] = []
    for rt in raw.tables:
        srcs = ", ".join(
            f"{s.file}" + (f"!{s.sheet}" if s.sheet is not None else "") for s in rt.sources
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


@dataclass
class _ParseCounter:
    non_null: int = 0
    parsed: int = 0
    examples: list[str] = field(default_factory=list)


class _StatTracker:
    """Plain counters. Pydantic FieldParseStat objects are built once in finalize()."""

    def __init__(self) -> None:
        self.stats: dict[str, _ParseCounter] = {}

    def record(self, key: str, raw_val: str | None, parsed_ok: bool) -> None:
        raw_clean = _strip_str(raw_val)
        if raw_clean is None:
            return
        st = self.stats.get(key)
        if st is None:
            st = _ParseCounter()
            self.stats[key] = st
        st.non_null += 1
        if parsed_ok:
            st.parsed += 1
        elif len(st.examples) < 5 and raw_val not in st.examples:
            assert raw_val is not None
            st.examples.append(raw_val)

    def finalize(self) -> dict[str, FieldParseStat]:
        out: dict[str, FieldParseStat] = {}
        for key, st in self.stats.items():
            rate = round(st.parsed / st.non_null, 6) if st.non_null > 0 else 1.0
            out[key] = FieldParseStat(
                non_null=st.non_null,
                parsed=st.parsed,
                rate=rate,
                examples=list(st.examples),
            )
        return out


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
    c_pid = cols.get("product_id")
    c_pname = cols.get("product_name")
    c_bunit = cols.get("base_unit")
    c_cat = cols.get("category")
    c_ucost = cols.get("unit_cost")

    rows: list[dict[str, Any]] = []
    for row in df.to_dict("records"):
        pid_raw = row[c_pid] if c_pid is not None else None
        pname_raw = row[c_pname] if c_pname is not None else None
        bunit_raw = row[c_bunit] if c_bunit is not None else None
        cat_raw = row[c_cat] if c_cat is not None else None
        ucost_raw = row[c_ucost] if c_ucost is not None else None

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
    c_cid = cols.get("customer_id")
    c_cname = cols.get("customer_name")
    c_seg = cols.get("segment")
    c_phone = cols.get("phone")

    rows: list[dict[str, Any]] = []
    for row in df.to_dict("records"):
        cid_raw = row[c_cid] if c_cid is not None else None
        cname_raw = row[c_cname] if c_cname is not None else None
        seg_raw = row[c_seg] if c_seg is not None else None
        phone_raw = row[c_phone] if c_phone is not None else None

        cid = _strip_str(cid_raw)
        tracker.record("customers.customer_id", cid_raw, cid is not None)
        cname = _strip_str(cname_raw)
        tracker.record("customers.customer_name", cname_raw, cname is not None)
        seg = _strip_str(seg_raw)
        tracker.record("customers.segment", seg_raw, seg is not None)
        phone = _strip_str(phone_raw)
        tracker.record("customers.phone", phone_raw, phone is not None)

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
                "phone": phone,
            }
        )
    return _records_to_df(rows, CUSTOMERS_COLUMNS)


@dataclass(frozen=True)
class ProductNameIndex:
    """Exact NFC/casefold names and unique accent-stripped folds -> catalog product_key."""

    by_norm: dict[str, str]
    by_fold: dict[str, str]


def _fold_product_name(raw: str | None) -> str | None:
    """Accent-stripped, case-folded, whitespace-collapsed product name."""
    if raw is None or not isinstance(raw, str):
        return None
    nfc = unicodedata.normalize("NFC", raw).replace("\xa0", " ").strip()
    if not nfc:
        return None
    nfd = unicodedata.normalize("NFD", nfc.casefold())
    bare = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn").replace("đ", "d")
    folded = " ".join(bare.split())
    return folded or None


def _product_key_by_name(products_df: pd.DataFrame) -> ProductNameIndex:
    """Catalog name -> product_key for catalogs keyed by code.

    Sales lines that record only the product name must land on the catalog's code key,
    otherwise category, unit cost and base unit never join. A name used by two codes is
    ambiguous and is left out. Accent-stripped folds match only when unique.
    """
    if products_df.empty:
        return ProductNameIndex(by_norm={}, by_fold={})
    keys_by_name: dict[str, set[str]] = {}
    keys_by_fold: dict[str, set[str]] = {}
    for row in products_df.to_dict("records"):
        pid, pname, pkey = row.get("product_id"), row.get("product_name"), row.get("product_key")
        norm = normalize_label(pname) if isinstance(pname, str) else None
        if pid is None or norm is None or pkey is None:
            continue
        key = str(pkey)
        keys_by_name.setdefault(norm, set()).add(key)
        folded = _fold_product_name(pname if isinstance(pname, str) else None)
        if folded is not None:
            keys_by_fold.setdefault(folded, set()).add(key)
    by_norm = {name: next(iter(keys)) for name, keys in keys_by_name.items() if len(keys) == 1}
    by_fold = {name: next(iter(keys)) for name, keys in keys_by_fold.items() if len(keys) == 1}
    return ProductNameIndex(by_norm=by_norm, by_fold=by_fold)


def _line_product_key(
    pid: str | None,
    pname: str | None,
    key_by_name: ProductNameIndex | dict[str, str],
) -> str | None:
    if pid is not None:
        return pid
    norm = normalize_label(pname)
    if norm is None:
        return None
    if isinstance(key_by_name, dict):
        return key_by_name.get(norm, norm)
    if norm in key_by_name.by_norm:
        return key_by_name.by_norm[norm]
    folded = _fold_product_name(pname)
    if folded is not None and folded in key_by_name.by_fold:
        return key_by_name.by_fold[folded]
    return norm


def _merge_non_catalog_folded_keys(
    lines: list[dict[str, Any]],
    products_df: pd.DataFrame,
) -> None:
    """Merge non-catalog line keys that share an accent-stripped fold.

    Winner = most frequent spelling on the lines; tie -> first seen. Catalog keys are
    never remapped. Folds that are ambiguous across two+ catalog products are left alone.
    """
    catalog_keys: set[str] = set()
    catalog_fold_keys: dict[str, set[str]] = defaultdict(set)
    for row in products_df.to_dict("records"):
        pkey = row.get("product_key")
        if pkey is None:
            continue
        key = str(pkey)
        catalog_keys.add(key)
        pname = row.get("product_name")
        folded = _fold_product_name(pname if isinstance(pname, str) else key)
        if folded is not None:
            catalog_fold_keys[folded].add(key)
    ambiguous_folds = {fold for fold, keys in catalog_fold_keys.items() if len(keys) > 1}

    counts: Counter[str] = Counter()
    first_seen: dict[str, int] = {}
    fold_of: dict[str, str] = {}
    for index, pl in enumerate(lines):
        pkey = pl.get("product_key")
        if pkey is None:
            continue
        key = str(pkey)
        if key in catalog_keys:
            continue
        folded = _fold_product_name(key)
        if folded is None or folded in ambiguous_folds:
            continue
        fold_of[key] = folded
        counts[key] += 1
        if key not in first_seen:
            first_seen[key] = index

    keys_by_fold: dict[str, list[str]] = defaultdict(list)
    for key, folded in fold_of.items():
        keys_by_fold[folded].append(key)

    remap: dict[str, str] = {}
    for keys in keys_by_fold.values():
        if len(keys) < 2:
            continue
        winner = min(keys, key=lambda k: (-counts[k], first_seen[k]))
        for key in keys:
            if key != winner:
                remap[key] = winner

    if not remap:
        return
    for pl in lines:
        pkey = pl.get("product_key")
        if pkey is None:
            continue
        mapped = remap.get(str(pkey))
        if mapped is not None:
            pl["product_key"] = mapped


def _build_unit_conversions(
    raw: BusinessRaw,
    mapping: Mapping,
    tracker: _StatTracker,
    warnings: list[str],
    notes: list[str],
    key_by_name: ProductNameIndex | dict[str, str] | None = None,
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
    c_pid = cols.get("product_id")
    c_pname = cols.get("product_name")
    c_unit = cols.get("unit")
    c_bunit = cols.get("base_unit")
    c_factor = cols.get("factor")

    rows: list[dict[str, Any]] = []
    for row in df.to_dict("records"):
        pid_raw = row[c_pid] if c_pid is not None else None
        pname_raw = row[c_pname] if c_pname is not None else None
        unit_raw = row[c_unit] if c_unit is not None else None
        bunit_raw = row[c_bunit] if c_bunit is not None else None
        factor_raw = row[c_factor] if c_factor is not None else None

        pid = _strip_str(pid_raw)
        tracker.record("unit_conversions.product_id", pid_raw, pid is not None)
        pname = _strip_str(pname_raw)
        tracker.record("unit_conversions.product_name", pname_raw, pname is not None)
        unit = _strip_str(unit_raw)
        tracker.record("unit_conversions.unit", unit_raw, unit is not None)
        bunit = _strip_str(bunit_raw)
        tracker.record("unit_conversions.base_unit", bunit_raw, bunit is not None)

        qty_parsed = parse_qty(factor_raw)
        factor: Decimal | None = (
            qty_parsed[0] if qty_parsed is not None else parse_decimal(factor_raw)
        )
        tracker.record("unit_conversions.factor", factor_raw, factor is not None)

        pkey = _line_product_key(pid, pname, key_by_name or ProductNameIndex({}, {}))
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

        local_keys = list(jspec.on.keys())
        remote_keys = [jspec.on[k] for k in local_keys]
        if any(rk not in target_df.columns for rk in remote_keys):
            continue

        lookup: dict[tuple[Any, ...], Any] = {}
        for t_row in target_df.to_dict("records"):
            val = t_row[source_field]
            if val is None or (isinstance(val, float) and pd.isna(val)):
                continue
            key_tuple = tuple(
                _strip_str(str(t_row[rk])) if t_row[rk] is not None else None for rk in remote_keys
            )
            if all(k is not None for k in key_tuple) and key_tuple not in lookup:
                lookup[key_tuple] = val

        for r in rows:
            if r.get(target_field) is not None:
                continue
            l_tuple = tuple(
                _strip_str(str(r.get(lk))) if r.get(lk) is not None else None for lk in local_keys
            )
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


def _walk_in_norm_set(values: list[str]) -> set[str]:
    """Normalized walk-in labels plus the initials of multi-word ones ("khách lẻ" -> "kl").

    Shops often abbreviate the walk-in label and the mapping step does not always list
    the abbreviation next to the full label.
    """
    found: set[str] = set()
    for value in values:
        label = normalize_label(value)
        if label is None:
            continue
        found.add(label)
        folded = "".join(
            ch for ch in unicodedata.normalize("NFD", label) if unicodedata.category(ch) != "Mn"
        ).replace("đ", "d")
        words = folded.split()
        if len(words) >= 2:
            found.add("".join(word[0] for word in words))
    return found


def _synthetic_order_id(file_name: str, sheet_name: str | None, row_no: int) -> str:
    sheet = sheet_name if isinstance(sheet_name, str) else ""
    return f"{file_name}|{sheet}|{row_no}"


_BLANK_STATUS_SHARE = 0.5


def _derive_qty_base(
    lines: list[dict[str, Any]],
    products_df: pd.DataFrame,
    unit_conv_df: pd.DataFrame,
    fill_counts: Counter[str],
) -> None:
    """Fill qty_base from qty + unit when the mapping did not provide a qty_base column."""
    base_by_key: dict[str, str | None] = {}
    for row in products_df.to_dict("records"):
        pkey = row.get("product_key")
        if pkey is None:
            continue
        bunit = row.get("base_unit")
        base_by_key[str(pkey)] = bunit if isinstance(bunit, str) and bunit.strip() else None

    factor_by: dict[tuple[str, str], Decimal] = {}
    products_with_conv: set[str] = set()
    for row in unit_conv_df.to_dict("records"):
        pkey, unit, factor = row.get("product_key"), row.get("unit"), row.get("factor")
        if pkey is None or unit is None or factor is None:
            continue
        unit_norm = normalize_label(str(unit))
        if unit_norm is None:
            continue
        key = str(pkey)
        products_with_conv.add(key)
        factor_by[(key, unit_norm)] = (
            factor if isinstance(factor, Decimal) else Decimal(str(factor))
        )
        # A conversion list names the base unit too; shops without a catalog base unit
        # need it so lines sold in the base unit count as qty.
        conv_base = row.get("base_unit")
        if base_by_key.get(key) is None and isinstance(conv_base, str) and conv_base.strip():
            base_by_key[key] = conv_base

    # Products with no catalog base unit and no conversions: one distinct unit on lines -> qty.
    units_seen: dict[str, set[str]] = defaultdict(set)
    for pl in lines:
        pkey = pl.get("product_key")
        unit = pl.get("unit")
        if pkey is None or unit is None:
            continue
        unit_norm = normalize_label(str(unit))
        if unit_norm is not None:
            units_seen[str(pkey)].add(unit_norm)
    single_unit_products = {
        pkey
        for pkey, units in units_seen.items()
        if len(units) == 1 and base_by_key.get(pkey) is None and pkey not in products_with_conv
    }

    for pl in lines:
        if pl.get("qty_base") is not None:
            continue
        qty = pl.get("qty")
        pkey = pl.get("product_key")
        if qty is None or pkey is None:
            continue
        pkey_s = str(pkey)
        unit = pl.get("unit")
        unit_norm = normalize_label(str(unit)) if isinstance(unit, str) else None
        base = base_by_key.get(pkey_s)
        base_norm = normalize_label(base) if isinstance(base, str) else None

        derived: Decimal | None = None
        if unit_norm is None or (base_norm is not None and unit_norm == base_norm):
            derived = qty if isinstance(qty, Decimal) else Decimal(str(qty))
        elif unit_norm is not None and (pkey_s, unit_norm) in factor_by:
            factor = factor_by[(pkey_s, unit_norm)]
            q = qty if isinstance(qty, Decimal) else Decimal(str(qty))
            derived = q * factor
        elif pkey_s in single_unit_products:
            derived = qty if isinstance(qty, Decimal) else Decimal(str(qty))

        if derived is not None:
            pl["qty_base"] = derived
            fill_counts["order_lines.qty_base"] += 1


def _fill_blank_status_as_completed(order_rows: list[dict[str, Any]]) -> bool:
    """A status column that is blank for most orders only notes the exceptions (returns,
    cancellations); a blank there means an ordinary completed order."""
    if not order_rows:
        return False
    blank = [row for row in order_rows if row["status_raw"] is None]
    if len(blank) < _BLANK_STATUS_SHARE * len(order_rows):
        return False
    for row in blank:
        if row["status"] is None:
            row["status"] = "completed"
    return True


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
    key_by_name = _product_key_by_name(products_df)
    unit_conv_df = _build_unit_conversions(raw, mapping, tracker, warnings, notes, key_by_name)

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

    no_line_order_id = "order_id" not in lines_tm.columns
    order_id_raw_col = None if no_line_order_id else lines_tm.columns["order_id"]
    if no_line_order_id:
        notes.append("no order id: one order per row")

    order_level_raw_cols: dict[str, list[str]] = {}
    if orders_tm.derived_from == "order_lines":
        for o_field, o_raw_col in orders_tm.columns.items():
            if o_field == "order_id":
                continue
            order_level_raw_cols.setdefault(o_raw_col, []).append(o_field)

    # Step (c) and (d): per (__file, __sheet) block forward-fill and exact pre-fill dedupe
    kept_line_records: list[dict[str, Any]] = []
    seen_in_order_by_table: dict[str, set[tuple[str, tuple[str | None, ...]]]] = {}
    is_derived_orders = orders_tm.derived_from == "order_lines"

    for chunk in line_chunks:
        rt = chunk.raw_table
        rt_data_cols = list(rt.columns)
        chunk_df = chunk.df
        if chunk_df.empty:
            continue

        seen_set = seen_in_order_by_table.setdefault(rt.name, set())

        prev_file: str | None = None
        prev_sheet: str | None = None
        prev_order_id: str | None = None
        prev_order_id_clean: str | None = None
        prev_order_raw_vals: dict[str, str | None] = {}

        for s_row in chunk_df.to_dict("records"):
            cur_file = s_row["__file"]
            cur_sheet = s_row["__sheet"]
            cur_row_no = int(s_row["__row"])

            if cur_file != prev_file or cur_sheet != prev_sheet:
                prev_file = cur_file
                prev_sheet = cur_sheet
                prev_order_id = None
                prev_order_id_clean = None
                prev_order_raw_vals = {}

            pre_fill_tuple = tuple(
                s_row[c] if isinstance(s_row[c], str) else None for c in rt_data_cols
            )

            if no_line_order_id:
                # One sale per row: do not forward-fill order_id and do not dedupe.
                resolved_oid_raw = None
                oid_clean = None
                same_order_as_prev = False
            else:
                assert order_id_raw_col is not None
                raw_oid_val = s_row[order_id_raw_col]
                raw_oid_cell = raw_oid_val if isinstance(raw_oid_val, str) else None
                if raw_oid_cell is None and prev_order_id is not None:
                    resolved_oid_raw = prev_order_id
                    oid_clean = prev_order_id_clean
                    fill_counts["order_id"] += 1
                    same_order_as_prev = True
                else:
                    resolved_oid_raw = raw_oid_cell
                    oid_clean = _strip_str(resolved_oid_raw)
                    same_order_as_prev = (
                        oid_clean is not None
                        and prev_order_id_clean is not None
                        and oid_clean == prev_order_id_clean
                    )

            filled_row_dict: dict[str, str | None] = {
                c: (s_row[c] if isinstance(s_row[c], str) else None) for c in rt_data_cols
            }
            if order_id_raw_col is not None:
                filled_row_dict[order_id_raw_col] = resolved_oid_raw

            if is_derived_orders and order_level_raw_cols:
                for raw_col_name, canonical_o_fields in order_level_raw_cols.items():
                    if raw_col_name not in filled_row_dict:
                        continue
                    cell_val = filled_row_dict[raw_col_name]
                    if (
                        cell_val is None
                        and same_order_as_prev
                        and prev_order_raw_vals.get(raw_col_name) is not None
                    ):
                        filled_row_dict[raw_col_name] = prev_order_raw_vals[raw_col_name]
                        for cf in canonical_o_fields:
                            fill_counts[cf] += 1

            if resolved_oid_raw is not None:
                if not same_order_as_prev:
                    prev_order_raw_vals = {}
                prev_order_id = resolved_oid_raw
                prev_order_id_clean = oid_clean
                if is_derived_orders and order_level_raw_cols:
                    for raw_col_name in order_level_raw_cols:
                        val_in_row = filled_row_dict.get(raw_col_name)
                        if val_in_row is not None:
                            prev_order_raw_vals[raw_col_name] = val_in_row

            if not no_line_order_id and oid_clean is not None:
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
    c_l_oid = lcols.get("order_id")
    c_l_lno = lcols.get("line_no")
    c_l_pid = lcols.get("product_id")
    c_l_pname = lcols.get("product_name")
    c_l_qty = lcols.get("qty")
    c_l_unit = lcols.get("unit")
    c_l_qbase = lcols.get("qty_base")
    c_l_uprice = lcols.get("unit_price")
    c_l_ldisc = lcols.get("line_discount")
    c_l_lamt = lcols.get("line_amount")
    c_l_cpu = lcols.get("cost_per_unit")

    # Raw columns some canonical field already reads; the other text columns become
    # order attributes (a vehicle, the job done, a note) that questions may filter on.
    mapped_raw_cols = set(lcols.values())
    if is_derived_orders:
        mapped_raw_cols |= set(orders_tm.columns.values())
    attribute_rows: set[tuple[str, str, str]] = set()

    parsed_lines: list[dict[str, Any]] = []
    raw_lines_for_derived_orders: list[
        tuple[str, str | None, str, Decimal | None, dict[str, str | None]]
    ] = []

    for rec in kept_line_records:
        f_name: str = rec["__file"]
        s_name: str | None = rec["__sheet"]
        r_no: int = rec["__row"]
        rdict: dict[str, str | None] = rec["raw"]

        if c_l_oid is None:
            oid = _synthetic_order_id(f_name, s_name, r_no)
            oid_raw = oid
        else:
            oid_raw = rdict.get(c_l_oid)
            oid = _strip_str(oid_raw)
        tracker.record("order_lines.order_id", oid_raw, oid is not None)

        if oid is not None:
            for col, cell in rdict.items():
                if col in mapped_raw_cols:
                    continue
                text = _strip_str(cell)
                if text and len(text) <= _MAX_ATTRIBUTE_LEN and any(ch.isalpha() for ch in text):
                    attribute_rows.add((oid, unicodedata.normalize("NFC", col), text))

        lno_raw = rdict.get(c_l_lno) if c_l_lno is not None else None
        lno = _parse_int_id(lno_raw) if lno_raw is not None else None
        tracker.record("order_lines.line_no", lno_raw, lno is not None)

        pid_raw = rdict.get(c_l_pid) if c_l_pid is not None else None
        pid = _strip_str(pid_raw)
        tracker.record("order_lines.product_id", pid_raw, pid is not None)

        pname_raw = rdict.get(c_l_pname) if c_l_pname is not None else None
        pname = _strip_str(pname_raw)
        tracker.record("order_lines.product_name", pname_raw, pname is not None)

        qty_raw = rdict.get(c_l_qty) if c_l_qty is not None else None
        qty_parsed = parse_qty(qty_raw)
        qty_val: Decimal | None = qty_parsed[0] if qty_parsed is not None else None
        qty_embedded_unit: str | None = qty_parsed[1] if qty_parsed is not None else None
        tracker.record("order_lines.qty", qty_raw, qty_val is not None)

        unit_raw = rdict.get(c_l_unit) if c_l_unit is not None else None
        unit_col_val = _strip_str(unit_raw)
        if c_l_unit is not None:
            tracker.record("order_lines.unit", unit_raw, unit_col_val is not None)
        unit_val = unit_col_val if unit_col_val is not None else qty_embedded_unit

        qbase_raw = rdict.get(c_l_qbase) if c_l_qbase is not None else None
        qbase_parsed = parse_qty(qbase_raw)
        qbase_val: Decimal | None = (
            qbase_parsed[0] if qbase_parsed is not None else parse_decimal(qbase_raw)
        )
        tracker.record("order_lines.qty_base", qbase_raw, qbase_val is not None)

        uprice_raw = rdict.get(c_l_uprice) if c_l_uprice is not None else None
        uprice_val = parse_money(uprice_raw)
        tracker.record("order_lines.unit_price", uprice_raw, uprice_val is not None)

        ldisc_raw = rdict.get(c_l_ldisc) if c_l_ldisc is not None else None
        ldisc_val = parse_money(ldisc_raw)
        tracker.record("order_lines.line_discount", ldisc_raw, ldisc_val is not None)

        lamt_raw = rdict.get(c_l_lamt) if c_l_lamt is not None else None
        lamt_val = parse_money(lamt_raw)
        tracker.record("order_lines.line_amount", lamt_raw, lamt_val is not None)

        cpu_raw = rdict.get(c_l_cpu) if c_l_cpu is not None else None
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
        if is_derived_orders and oid is not None:
            raw_lines_for_derived_orders.append((oid, s_name, f_name, qty_val, rdict))

    # Step (f): joins on order_lines
    _apply_joins(parsed_lines, lines_tm.join, entity_dfs)

    # Step (g): product_key and sequential line_no per order
    line_seq_counter: Counter[str] = Counter()
    for pl in parsed_lines:
        pid = pl["product_id"]
        pname = pl["product_name"]
        pl["product_key"] = _line_product_key(pid, pname, key_by_name)
        oid_key = pl["order_id"] or ""
        line_seq_counter[oid_key] += 1
        if pl["line_no"] is None:
            pl["line_no"] = line_seq_counter[oid_key]
    _merge_non_catalog_folded_keys(parsed_lines, products_df)

    # Derive qty_base when the mapping has no qty_base column.
    if c_l_qbase is None:
        _derive_qty_base(parsed_lines, products_df, unit_conv_df, fill_counts)

    order_lines_df = _records_to_df(parsed_lines, ORDER_LINES_COLUMNS)

    # Step (h): build orders
    norm_status_map: dict[str, str] = {}
    for k, v in mapping.status_map.items():
        nk = normalize_label(k)
        if nk is not None:
            norm_status_map[nk] = v

    walk_in_norm_set = _walk_in_norm_set(mapping.walk_in_values)

    orders_with_negative_qty: set[str] = {
        pl["order_id"] for pl in parsed_lines if pl["order_id"] is not None and pl["is_return_line"]
    }

    status_overridden_by_return = 0
    order_rows: list[dict[str, Any]] = []
    has_status_column = "status" in orders_tm.columns
    if not has_status_column:
        notes.append("no status column: all orders completed")

    year_hint_cache: dict[tuple[str | None, str], int | None] = {}

    def _cached_year_hint(s_name: str | None, f_name: str) -> int | None:
        key = (s_name, f_name)
        if key not in year_hint_cache:
            year_hint_cache[key] = year_hint(s_name, f_name)
        return year_hint_cache[key]

    if is_derived_orders:
        ocols = orders_tm.columns
        lines_by_oid: dict[
            str, list[tuple[str | None, str, Decimal | None, dict[str, str | None]]]
        ] = {}
        for oid, s_name, f_name, qty_val, rdict in raw_lines_for_derived_orders:
            lines_by_oid.setdefault(oid, []).append((s_name, f_name, qty_val, rdict))

        def _parse_order_field(
            field_name: str,
            raw_val: str | None,
            s_name: str | None,
            f_name: str,
        ) -> Any:
            if raw_val is None:
                return None
            if field_name == "order_date":
                val = parse_date(raw_val, default_year=_cached_year_hint(s_name, f_name))
                tracker.record("orders.order_date", raw_val, val is not None)
                return val
            if field_name == "order_ts":
                val_ts = parse_timestamp(raw_val, default_year=_cached_year_hint(s_name, f_name))
                tracker.record("orders.order_ts", raw_val, val_ts is not None)
                return val_ts
            if field_name in {
                "order_subtotal",
                "order_discount",
                "shipping_fee",
                "order_payable",
                "amount_paid",
                "amount_due",
            }:
                val_m = parse_money(raw_val)
                tracker.record(f"orders.{field_name}", raw_val, val_m is not None)
                return val_m
            val_s = _strip_str(raw_val)
            if field_name != "status":
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
            "amount_paid",
            "amount_due",
        ]
        active_derived_fields = [f for f in derived_fields if f in ocols]

        for oid, o_lines in lines_by_oid.items():
            tracker.record("orders.order_id", oid, True)

            resolved_fields: dict[str, Any] = {f: None for f in derived_fields}
            for fname in active_derived_fields:
                raw_col = ocols[fname]
                non_neg_vals: list[Any] = []
                any_vals: list[Any] = []
                for s_name, f_name, qty_val, rdict in o_lines:
                    rv = rdict.get(raw_col)
                    pv = _parse_order_field(fname, rv, s_name, f_name)
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

            if has_status_column:
                status_raw = resolved_fields["status"]
                norm_st = normalize_label(status_raw) if status_raw is not None else None
                mapped_status: str | None = None
                if norm_st is not None:
                    mapped_status = norm_status_map.get(norm_st)
                    if mapped_status is None:
                        unmapped_statuses[status_raw] += 1
                tracker.record("orders.status", status_raw, mapped_status is not None)
            else:
                status_raw = None
                mapped_status = "completed"

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
                    "amount_paid": resolved_fields["amount_paid"],
                    "amount_due": resolved_fields["amount_due"],
                }
            )
    else:
        order_chunks = _resolve_sources(raw, "orders", orders_tm.sources, warnings)
        _validate_mapped_columns("orders", orders_tm, order_chunks)
        ocols = orders_tm.columns
        c_o_oid = ocols.get("order_id")
        c_o_date = ocols.get("order_date")
        c_o_ts = ocols.get("order_ts")
        c_o_st = ocols.get("status")
        c_o_cid = ocols.get("customer_id")
        c_o_cname = ocols.get("customer_name")
        c_o_pm = ocols.get("payment_method")
        c_o_ch = ocols.get("channel")
        c_o_emp = ocols.get("employee_id")
        c_o_sub = ocols.get("order_subtotal")
        c_o_disc = ocols.get("order_discount")
        c_o_ship = ocols.get("shipping_fee")
        c_o_pay = ocols.get("order_payable")
        c_o_paid = ocols.get("amount_paid")
        c_o_due = ocols.get("amount_due")

        for chunk in order_chunks:
            for s_row in chunk.df.to_dict("records"):
                f_name = s_row["__file"]
                s_name = s_row["__sheet"]
                yh = _cached_year_hint(s_name, f_name)

                oid_raw = s_row[c_o_oid] if c_o_oid is not None else None
                oid = _strip_str(oid_raw)
                tracker.record("orders.order_id", oid_raw, oid is not None)

                odate_raw = s_row[c_o_date] if c_o_date is not None else None
                odate = parse_date(odate_raw, default_year=yh)
                tracker.record("orders.order_date", odate_raw, odate is not None)

                ots_raw = s_row[c_o_ts] if c_o_ts is not None else None
                ots = parse_timestamp(ots_raw, default_year=yh)
                tracker.record("orders.order_ts", ots_raw, ots is not None)

                if has_status_column:
                    status_raw_cell = s_row[c_o_st] if c_o_st is not None else None
                    status_raw = _strip_str(status_raw_cell)
                    norm_st = normalize_label(status_raw) if status_raw is not None else None
                    mapped_status = None
                    if norm_st is not None:
                        mapped_status = norm_status_map.get(norm_st)
                        if mapped_status is None:
                            unmapped_statuses[status_raw] += 1
                    tracker.record("orders.status", status_raw_cell, mapped_status is not None)
                else:
                    status_raw = None
                    mapped_status = "completed"

                cid_raw = s_row[c_o_cid] if c_o_cid is not None else None
                cid = _strip_str(cid_raw)
                tracker.record("orders.customer_id", cid_raw, cid is not None)

                cname_raw = s_row[c_o_cname] if c_o_cname is not None else None
                cname = _strip_str(cname_raw)
                tracker.record("orders.customer_name", cname_raw, cname is not None)

                pm_raw = s_row[c_o_pm] if c_o_pm is not None else None
                pm = _strip_str(pm_raw)
                tracker.record("orders.payment_method", pm_raw, pm is not None)

                ch_raw = s_row[c_o_ch] if c_o_ch is not None else None
                ch = _strip_str(ch_raw)
                tracker.record("orders.channel", ch_raw, ch is not None)

                emp_raw = s_row[c_o_emp] if c_o_emp is not None else None
                emp = _strip_str(emp_raw)
                tracker.record("orders.employee_id", emp_raw, emp is not None)

                sub_raw = s_row[c_o_sub] if c_o_sub is not None else None
                sub_val = parse_money(sub_raw)
                tracker.record("orders.order_subtotal", sub_raw, sub_val is not None)

                disc_raw = s_row[c_o_disc] if c_o_disc is not None else None
                disc_val = parse_money(disc_raw)
                tracker.record("orders.order_discount", disc_raw, disc_val is not None)

                ship_raw = s_row[c_o_ship] if c_o_ship is not None else None
                ship_val = parse_money(ship_raw)
                tracker.record("orders.shipping_fee", ship_raw, ship_val is not None)

                pay_raw = s_row[c_o_pay] if c_o_pay is not None else None
                pay_val = parse_money(pay_raw)
                tracker.record("orders.order_payable", pay_raw, pay_val is not None)

                paid_raw = s_row[c_o_paid] if c_o_paid is not None else None
                paid_val = parse_money(paid_raw)
                tracker.record("orders.amount_paid", paid_raw, paid_val is not None)

                due_raw = s_row[c_o_due] if c_o_due is not None else None
                due_val = parse_money(due_raw)
                tracker.record("orders.amount_due", due_raw, due_val is not None)

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
                        "amount_paid": paid_val,
                        "amount_due": due_val,
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

    if has_status_column and _fill_blank_status_as_completed(order_rows):
        notes.append("status column is mostly blank (exceptions only): blank orders completed")

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

    from datamate.etl.simple import load_all_simple_tables

    walk_in_for_simple: set[str] = {
        nk for v in mapping.walk_in_values if (nk := normalize_label(v)) is not None
    }
    simple_tables = load_all_simple_tables(
        raw,
        mapping,
        tracker,
        warnings,
        dedupe_counts,
        key_by_name=key_by_name,
        walk_in_norm_set=walk_in_for_simple,
    )

    if "employees" in simple_tables:
        from datamate.etl.staff import resolve_staff

        orders_df, simple_tables["employees"], tied = resolve_staff(
            orders_df, simple_tables["employees"]
        )
        if tied:
            notes.append(f"{tied} staff names on orders tied to the staff list")

    final_parse_stats = tracker.finalize()

    rows_per_table = {
        "orders": len(orders_df),
        "order_lines": len(order_lines_df),
        "products": len(products_df),
        "unit_conversions": len(unit_conv_df),
        "customers": len(customers_df),
    }
    for name, frame in simple_tables.items():
        rows_per_table[name] = len(frame)

    report = EtlReport(
        rows_per_table=rows_per_table,
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

    attributes_df = pd.DataFrame(sorted(attribute_rows), columns=list(ORDER_ATTRIBUTES_COLUMNS))
    return CanonicalData(
        order_attributes=attributes_df,
        orders=orders_df,
        order_lines=order_lines_df,
        products=products_df,
        unit_conversions=unit_conv_df,
        customers=customers_df,
        report=report,
        **simple_tables,
    )

from __future__ import annotations

import datetime
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from datamate.config import settings
from datamate.io.profile import BusinessProfile, ColumnProfile, TableProfile, render_card
from datamate.io.raw import BusinessRaw, RawTable
from datamate.llm.client import LLMError, LLMUnavailable
from datamate.mapping.fields import FIELD_DOCS, LATER_FIELDS, REQUIRED_FIELDS
from datamate.mapping.model import (
    Entity,
    JoinSpec,
    Mapping,
    OrderStatus,
    ReturnRule,
    SourceSpec,
    TableMapping,
)
from datamate.mapping.validate import Issue, is_id_or_phone_like, validate_mapping
from datamate.parse import parse_date, parse_qty

_DATE_KINDS = frozenset(
    {"date_dmy", "date_dm", "date_thg", "datetime_iso", "int_serial_range"}
)
_STATUS_DISTINCT_LIMIT = 30
_NO_STATUS_NOTE = "no status column: all orders completed"

if TYPE_CHECKING:
    from datamate.etl.build import CanonicalData

_DATA_END_DATE_RE = re.compile(r"có đến hết\s+(\d{1,2}/\d{1,2}/\d{4})", re.IGNORECASE)
_STEP_MAX_TOKENS = 512
_NO_ORDER_ID_HINT = (
    "- Set `order_id` to null if the table records one sale per row "
    "without an order/invoice code.\n"
)
# Invoice-only files (one row per order, no item lines) list both the merchandise amount and
# what the customer pays; rulebook revenue is the merchandise amount.
_WHOLE_ORDER_ROW_HINT = (
    "- If each row is a whole order with no item lines, `line_amount` is the order's merchandise "
    "amount (tiền hàng / giá trị hàng: before order-level discount, without shipping), NOT the "
    "amount the customer pays (tổng tiền khách trả, after discount and shipping).\n"
)
_ALL_ENTITIES: tuple[Entity, ...] = (
    "orders",
    "order_lines",
    "products",
    "unit_conversions",
    "customers",
)


@dataclass
class StepLog:
    step: str
    prompt_tokens: int
    raw_answer: dict[str, Any]
    cache_hit: bool
    latency_s: float
    shrunk: bool = False


@dataclass
class InferResult:
    mapping: Mapping | None
    steps: list[StepLog] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    canonical: CanonicalData | None = None


def extract_data_end_date(readme_text: str | None) -> datetime.date | None:
    if not readme_text:
        return None
    m = _DATA_END_DATE_RE.search(readme_text)
    if m is None:
        return None
    return parse_date(m.group(1))


def _split_readme_sections(readme_text: str | None) -> list[tuple[str, str]]:
    if not readme_text:
        return []
    sections: list[tuple[str, str]] = []
    cur_heading = ""
    cur_lines: list[str] = []
    for line in readme_text.splitlines():
        if line.startswith("## "):
            if cur_heading or cur_lines:
                sections.append((cur_heading, "\n".join(cur_lines).strip()))
            cur_heading = line[3:].strip()
            cur_lines = [line]
        else:
            cur_lines.append(line)
    if cur_heading or cur_lines:
        sections.append((cur_heading, "\n".join(cur_lines).strip()))
    return sections


def _readme_for_step_a(readme_text: str | None) -> str:
    sections = _split_readme_sections(readme_text)
    if not sections:
        return ""
    kept: list[str] = []
    for heading, body in sections:
        h_lower = heading.lower()
        if h_lower.startswith(("cột trong", "cot trong")):
            continue
        kept.append(body)
    return "\n\n".join(kept).strip()


def _readme_for_table(readme_text: str | None, source_files: list[str]) -> str:
    sections = _split_readme_sections(readme_text)
    if not sections:
        return ""
    kept: list[str] = []
    file_lowers = [f.lower() for f in source_files]
    for heading, body in sections:
        h_lower = heading.lower()
        if (
            not h_lower
            or "quy ước" in h_lower
            or "quy uoc" in h_lower
            or any(fl in h_lower for fl in file_lowers)
        ):
            kept.append(body)
    return "\n\n".join(kept).strip()


def _readme_conventions_only(readme_text: str | None) -> str:
    sections = _split_readme_sections(readme_text)
    if not sections:
        return ""
    kept: list[str] = []
    for heading, body in sections:
        h_lower = heading.lower()
        if not h_lower or "quy ước" in h_lower or "quy uoc" in h_lower:
            kept.append(body)
    return "\n\n".join(kept).strip()


def _build_source_specs_from_raw_table(
    rt: RawTable, raw_by_name: dict[str, RawTable] | None = None
) -> list[SourceSpec]:
    by_file: dict[str, list[tuple[str | None, int]]] = {}
    for sref in rt.sources:
        by_file.setdefault(sref.file, []).append((sref.sheet, sref.header_row))
    # Sheet "*" means every sheet of the file, so it is only safe when no other table
    # (a glossary or a debt sheet in the same workbook) lives in that file.
    shared_files = {
        sref.file
        for other in (raw_by_name or {}).values()
        if other.name != rt.name
        for sref in other.sources
    }

    specs: list[SourceSpec] = []
    for fname, items in by_file.items():
        _first_sheet, first_hrow = items[0]
        if rt.kind == "csv":
            specs.append(SourceSpec(file=fname, sheet=None, header_row=first_hrow))
        elif len(items) > 1 and fname not in shared_files:
            specs.append(SourceSpec(file=fname, sheet="*", header_row=first_hrow))
        else:
            specs.extend(
                SourceSpec(file=fname, sheet=sheet, header_row=hrow) for sheet, hrow in items
            )
    return specs


def _compact_table_card_for_step_b(
    tp: TableProfile,
    max_samples: int = 5,
    include_top: bool = True,
) -> str:
    if max_samples >= 5 and include_top:
        return render_card(tp)
    lines: list[str] = [
        f"### Table `{tp.name}` ({tp.kind}) | rows: {tp.n_rows}",
        "- columns:",
    ]
    for col in tp.columns:
        grp_str = f" [group: {col.group}]" if col.group else ""
        kinds_str = ", ".join(f"{k} {r:.2f}" for k, r in col.kinds[:3]) if col.kinds else "none"
        if 0 < col.n_distinct <= 20 and col.all_values:
            shown_v = col.all_values[: max(3, max_samples * 2)]
            v_str = ", ".join(f'"{v[:25]}" {cnt}' for v, cnt in shown_v)
            tail = f"| values ({col.n_distinct}): {v_str}"
        else:
            s_list = col.samples[:max_samples]
            eg = ", ".join(f'"{s[:25]}"' for s in s_list) if s_list else "-"
            tail = f"| e.g. {eg}"
        lines.append(
            f"  - {col.name}{grp_str} | null {col.null_ratio:.2f} | distinct {col.n_distinct} "
            f"| kinds {kinds_str} {tail}"
        )
    return "\n".join(lines)


def _fit_prompt_to_budget(
    client: Any,
    variants: list[str],
    budget: int,
) -> tuple[str, int, bool]:
    for idx, text in enumerate(variants):
        toks = client.count_tokens(text)
        if toks <= budget:
            return text, toks, idx > 0
    last = variants[-1]
    max_chars = max(500, int(budget * 2.2))
    truncated = last[:max_chars] + "\n...[truncated]..."
    toks = client.count_tokens(truncated)
    return truncated, toks, True


def _call_step(
    client: Any,
    step_name: str,
    prompt_variants: list[str],
    schema: dict[str, Any],
    schema_name: str,
    budget: int,
    repair_suffix: str | None = None,
    required_keys: list[str] | None = None,
) -> tuple[dict[str, Any], StepLog]:
    if repair_suffix:
        full_variants = [f"{v}\n\n{repair_suffix}" for v in prompt_variants]
    else:
        full_variants = prompt_variants

    chosen_prompt, est_tokens, shrunk = _fit_prompt_to_budget(client, full_variants, budget)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise data-engineering assistant mapping Vietnamese shop tables "
                "to a canonical schema. Return strictly valid JSON matching the schema."
            ),
        },
        {"role": "user", "content": chosen_prompt},
    ]
    ans = client.chat_json(
        messages=messages,
        schema=schema,
        schema_name=schema_name,
        max_tokens=_STEP_MAX_TOKENS,
        required_keys=required_keys,
    )
    last_usage = client.usage_log[-1] if getattr(client, "usage_log", None) else None
    actual_toks = (
        last_usage.prompt_tokens
        if last_usage is not None and last_usage.prompt_tokens > 0
        else est_tokens
    )
    log = StepLog(
        step=step_name,
        prompt_tokens=actual_toks,
        raw_answer=ans,
        cache_hit=bool(last_usage.cache_hit) if last_usage is not None else False,
        latency_s=round(float(last_usage.latency_s), 6) if last_usage is not None else 0.0,
        shrunk=shrunk,
    )
    return ans, log


def _build_step_a_variants(profile: BusinessProfile) -> list[str]:
    readme_short = _readme_for_step_a(profile.readme_text)
    variants: list[str] = []
    for sample_len, readme_cap in [(25, 4000), (15, 2000), (0, 800)]:
        t_lines: list[str] = []
        for tp in profile.tables:
            col_parts: list[str] = []
            for c in tp.columns:
                if sample_len > 0 and c.samples:
                    col_parts.append(f'{c.name} (e.g. "{c.samples[0][:sample_len]}")')
                else:
                    col_parts.append(c.name)
            t_lines.append(f"- `{tp.name}` | rows={tp.n_rows} | columns: {', '.join(col_parts)}")
        r_part = readme_short[:readme_cap] if readme_short else "(no README)"
        prompt = (
            "Assign each raw table of this shop to its canonical entity role:\n"
            "- `order_lines`: individual line items sold per order (or flat sales ledger where each row is a line item)\n"
            "- `orders`: separate order header / invoice summary table (one row per order)\n"
            "- `products`: product / item catalog / price list\n"
            "- `unit_conversions`: unit conversion factor table (larger unit -> base unit)\n"
            "- `customers`: customer directory / list\n"
            "- `none`: any other table (e.g. purchases/imports, inventory, cashbook, expenses, batches)\n\n"
            "Also set `orders_source`:\n"
            "- `own_table` if there is a separate `orders` header table;\n"
            "- `order_lines` if there is no separate `orders` header table and order fields live on `order_lines`.\n"
            "Rules: Exactly ONE table must be assigned `order_lines`. If `orders_source` is `own_table`, "
            "exactly ONE table must be assigned `orders` (otherwise no table is `orders`).\n\n"
            f"### README\n{r_part}\n\n"
            f"### Raw Tables\n" + "\n".join(t_lines)
        )
        variants.append(prompt)
    return variants


def _step_a_schema(table_names: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "tables": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "table": {"type": "string", "enum": table_names},
                        "entity": {
                            "type": "string",
                            "enum": [*_ALL_ENTITIES, "none"],
                        },
                    },
                    "required": ["table", "entity"],
                    "additionalProperties": False,
                },
            },
            "orders_source": {
                "type": "string",
                "enum": ["own_table", "order_lines"],
            },
        },
        "required": ["tables", "orders_source"],
        "additionalProperties": False,
    }


def _build_step_b_variants(
    entity: Entity,
    tp: TableProfile,
    readme_text: str | None,
    orders_source: str = "own_table",
) -> list[str]:
    fdocs = _step_b_field_docs(entity)
    req = REQUIRED_FIELDS.get(entity, set())
    field_lines = [
        f"- `{fname}` ({'REQUIRED' if fname in req else 'optional'}): {desc}"
        for fname, desc in fdocs.items()
    ]
    fields_block = "\n".join(field_lines)
    readme_sec = _readme_for_table(readme_text, tp.source_files)

    extra_hints = ""
    if entity == "order_lines":
        extra_hints = (
            "\nImportant for `order_lines`:\n"
            "- `line_amount` MUST be the line total AFTER line discount (thành tiền / trị giá / tổng tiền dòng), "
            "NOT the unit price (`unit_price`) and NOT the pre-discount amount.\n"
            "- Only map `qty_base` if the table has a separate pre-converted base-unit quantity column.\n"
            f"{_NO_ORDER_ID_HINT}"
            f"{_WHOLE_ORDER_ROW_HINT if orders_source == 'order_lines' else ''}"
        )
    elif entity == "orders":
        derived_note = ""
        if orders_source == "order_lines":
            derived_note = (
                "- Because `orders` is derived from the flat `order_lines` table, order-level money fields "
                "(`order_subtotal`, `order_discount`, `shipping_fee`, `order_payable`) must NOT map to a "
                "line-level column (`line_amount`, `unit_price`, `line_discount`, `qty`, ...); in a flat "
                "sales table order-level totals usually do not exist: set them to `null` unless a column "
                "explicitly holds the whole-order total.\n"
            )
        order_id_hint = _NO_ORDER_ID_HINT if orders_source == "order_lines" else ""
        extra_hints = (
            "\nImportant for `orders`:\n"
            "- `order_date` is the date the order was placed. If a table has both a date column and a "
            "timestamp/time column, map `order_date` to the date column (or timestamp if it is the only date) "
            "and `order_ts` to the full timestamp column.\n"
            "- Map `status` only to a column that records the order state "
            "(done / cancelled / returned / delivering). If there is none, leave it null; do not "
            "use a free-text note or a work-description column.\n"
            "- `order_subtotal` is the merchandise total BEFORE order-level discount and shipping fee.\n"
            f"{order_id_hint}"
            f"{derived_note}"
        )

    variants: list[str] = []
    for max_samples, include_top, readme_cap in [
        (5, True, 3500),
        (2, False, 1500),
        (1, False, 500),
    ]:
        card_str = _compact_table_card_for_step_b(
            tp, max_samples=max_samples, include_top=include_top
        )
        r_str = readme_sec[:readme_cap] if readme_sec else "(no README section)"
        prompt = (
            f"Map the canonical fields of entity `{entity}` to the exact column names of table `{tp.name}`.\n"
            "Set a field to `null` if the table does not have a corresponding column.\n"
            f"{extra_hints}\n"
            f"### Canonical Fields for `{entity}`\n{fields_block}\n\n"
            f"### README Context\n{r_str}\n\n"
            f"### Table Profile\n{card_str}"
        )
        variants.append(prompt)
    return variants


def _step_b_field_docs(entity: Entity) -> dict[str, str]:
    later = LATER_FIELDS.get(entity, frozenset())
    return {name: doc for name, doc in FIELD_DOCS[entity].items() if name not in later}


def _step_b_required_names(entity: Entity, orders_source: str) -> list[str]:
    """Canonical fields that must be present. Optional omissions are filled with null."""
    required = set(REQUIRED_FIELDS.get(entity, ()))
    if entity == "orders" and orders_source == "own_table":
        required.add("order_id")
    return [name for name in _step_b_field_docs(entity) if name in required]


def _step_b_schema(
    entity: Entity,
    actual_columns: list[str],
    orders_source: str,
) -> dict[str, Any]:
    col_enum: list[str | None] = [*actual_columns, None]
    required_names = _step_b_required_names(entity, orders_source)
    required_set = set(required_names)
    # Required fields first so a truncated answer still contains them.
    ordered = required_names + [
        name for name in _step_b_field_docs(entity) if name not in required_set
    ]
    props = {fname: {"enum": col_enum} for fname in ordered}
    return {
        "type": "object",
        "properties": props,
        "required": ordered,
        "additionalProperties": False,
    }


def _collect_distinct_counts(rt: RawTable, col_name: str, limit: int = 50) -> list[tuple[str, int]]:
    if col_name not in rt.df.columns:
        return []
    vals: list[str] = [
        v.strip() for v in rt.df[col_name].tolist() if isinstance(v, str) and v.strip()
    ]
    counts = Counter(vals)
    sorted_items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return sorted_items[:limit]


def _n_distinct_nonempty(rt: RawTable, col_name: str) -> int:
    if col_name not in rt.df.columns:
        return 0
    return len(
        {
            v.strip()
            for v in rt.df[col_name].tolist()
            if isinstance(v, str) and v.strip()
        }
    )


def _column_mostly_date(col: ColumnProfile) -> bool:
    if not col.kinds:
        return False
    kind, share = col.kinds[0]
    return kind in _DATE_KINDS and share >= 0.5


def _table_has_date_column(tp: TableProfile) -> bool:
    return any(_column_mostly_date(col) for col in tp.columns)


def check_order_lines_sales_table(
    chosen: dict[Entity, str],
    orders_source: str,
    profiles: dict[str, TableProfile],
) -> str | None:
    """Reject a tiny undated 'sales' pick when a much larger dated table exists."""
    ol_name = chosen.get("order_lines")
    if ol_name is None or ol_name not in profiles:
        return None
    ol_tp = profiles[ol_name]
    orders_name = chosen.get("orders") if orders_source == "own_table" else None
    date_ok = _table_has_date_column(ol_tp)
    if not date_ok and orders_name is not None and orders_name in profiles:
        date_ok = _table_has_date_column(profiles[orders_name])
    if date_ok:
        return None

    bigger: TableProfile | None = None
    for name, tp in profiles.items():
        if name == ol_name or name == orders_name:
            continue
        if (
            tp.n_rows >= 10 * max(ol_tp.n_rows, 1)
            and _table_has_date_column(tp)
            and (bigger is None or tp.n_rows > bigger.n_rows)
        ):
            bigger = tp
    if bigger is None:
        return None
    return (
        f"{ol_name} has {ol_tp.n_rows} rows and no date column; "
        f"{bigger.name} has {bigger.n_rows} rows with dates. "
        "A glossary or lookup table is not the sales table."
    )


def _collect_walk_in_candidates(
    rt: RawTable, col_name: str, limit: int = 30
) -> list[tuple[str, int]]:
    if col_name not in rt.df.columns:
        return []
    vals: list[str] = [
        v.strip()
        for v in rt.df[col_name].tolist()
        if isinstance(v, str) and v.strip() and not is_id_or_phone_like(v)
    ]
    counts = Counter(vals)
    sorted_items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return sorted_items[:limit]


def _build_step_c_variants(
    status_counts: list[tuple[str, int]],
    readme_text: str | None,
) -> list[str]:
    conv = _readme_conventions_only(readme_text)
    vals_lines = [f'- "{val}" (count: {cnt})' for val, cnt in status_counts]
    vals_block = "\n".join(vals_lines)
    variants: list[str] = []
    for cap in (3000, 1200, 400):
        prompt = (
            "Map every distinct raw order status value to one of the 4 canonical order states:\n"
            "- `completed`: order is finished / delivered / paid / taken by customer (e.g. hoàn thành, đã xong, đã lấy, xong, ok)\n"
            "- `cancelled`: the order did not happen (e.g. huỷ, đã hủy đơn, khách hủy, "
            "không giao được, giao thất bại, khách không nhận)\n"
            "- `returned`: goods came back, including a partial return "
            "(e.g. trả hàng, đổi trả, hoàn hàng, trả một phần, hàng trả về)\n"
            "- `shipping`: order is currently being delivered / in transit (e.g. đang giao, đang vận chuyển)\n\n"
            f"### README Conventions\n{conv[:cap]}\n\n"
            f"### Distinct Raw Status Values\n{vals_block}"
        )
        variants.append(prompt)
    return variants


def _step_c_schema(status_values: list[str]) -> dict[str, Any]:
    state_enum: list[OrderStatus] = ["completed", "cancelled", "returned", "shipping"]
    props = {val: {"type": "string", "enum": state_enum} for val in status_values}
    return {
        "type": "object",
        "properties": props,
        "required": status_values,
        "additionalProperties": False,
    }


def _has_negative_qty(rt: RawTable, qty_col: str | None) -> bool:
    if not qty_col or qty_col not in rt.df.columns:
        return False
    for v in rt.df[qty_col].tolist():
        if isinstance(v, str):
            pq = parse_qty(v)
            if pq is not None and pq[0] < 0:
                return True
    return False


def _build_step_d_variants(
    customer_candidates: list[tuple[str, int]],
    has_neg_qty: bool,
    readme_text: str | None,
) -> list[str]:
    conv = _readme_conventions_only(readme_text)
    cand_lines = [f'- "{val}" (count: {cnt})' for val, cnt in customer_candidates]
    cand_block = "\n".join(cand_lines) if cand_lines else "(no customer candidates)"
    variants: list[str] = []
    for cap in (3000, 1200, 400):
        prompt = (
            "Determine the walk-in customer labels (`walk_in_values`) and the return rule (`return_rule`) for this shop:\n"
            "1. `walk_in_values`: select from the candidate values ONLY generic walk-in / anonymous customer placeholders "
            "(such as 'Khách lẻ', 'lẻ', 'Khách vãng lai', 'Vãng lai'). You may select nothing (`[]`) if no generic walk-in "
            "label exists among the candidates. Do NOT select real person names, customer codes, or phone numbers.\n"
            "2. `return_rule`:\n"
            "   - `negative_qty` if returned items are recorded as negative-quantity lines on the order (check README and whether negative quantities exist);\n"
            "   - `status` if returned orders are identified solely by their order status.\n\n"
            f"### Facts from Data\n- Quantity column has negative values: `{has_neg_qty}`\n\n"
            f"### README Conventions\n{conv[:cap]}\n\n"
            f"### Top Customer Column Values\n{cand_block}"
        )
        variants.append(prompt)
    return variants


def _step_d_schema(candidate_values: list[str]) -> dict[str, Any]:
    if candidate_values:
        walk_in_prop: dict[str, Any] = {
            "type": "array",
            "items": {"type": "string", "enum": candidate_values},
        }
    else:
        walk_in_prop = {
            "type": "array",
            "maxItems": 0,
        }
    return {
        "type": "object",
        "properties": {
            "walk_in_values": walk_in_prop,
            "return_rule": {
                "type": "string",
                "enum": ["status", "negative_qty"],
            },
        },
        "required": ["walk_in_values", "return_rule"],
        "additionalProperties": False,
    }


def _assemble_mapping(
    profile: BusinessProfile,
    raw_by_name: dict[str, RawTable],
    chosen_tables: dict[Entity, str],
    orders_source: str,
    entity_columns: dict[Entity, dict[str, str]],
    status_map: dict[str, OrderStatus],
    walk_in_values: list[str],
    return_rule: ReturnRule,
) -> Mapping:
    tables_dict: dict[Entity, TableMapping] = {}

    ol_table_name = chosen_tables["order_lines"]
    ol_rt = raw_by_name[ol_table_name]
    ol_cols = dict(entity_columns.get("order_lines", {}))

    for ent in ("products", "unit_conversions", "customers"):
        ent_typed: Entity = ent  # type: ignore[assignment]
        if ent_typed in chosen_tables:
            cols_map = dict(entity_columns.get(ent_typed, {}))
            if cols_map:
                rt = raw_by_name[chosen_tables[ent_typed]]
                tables_dict[ent_typed] = TableMapping(
                    sources=_build_source_specs_from_raw_table(rt, raw_by_name),
                    columns=cols_map,
                )

    ol_joins: dict[str, JoinSpec] = {}
    prod_tm = tables_dict.get("products")
    if (
        "product_id" in ol_cols
        and "product_name" not in ol_cols
        and prod_tm is not None
        and "product_id" in prod_tm.columns
        and "product_name" in prod_tm.columns
    ):
        ol_joins["product_name"] = JoinSpec(
            table="products",
            on={"product_id": "product_id"},
        )

    tables_dict["order_lines"] = TableMapping(
        sources=_build_source_specs_from_raw_table(ol_rt, raw_by_name),
        join=ol_joins,
        columns=ol_cols,
    )

    ord_cols = dict(entity_columns.get("orders", {}))
    if "order_id" not in ord_cols and "order_id" in ol_cols and orders_source == "order_lines":
        ord_cols["order_id"] = ol_cols["order_id"]

    if orders_source == "order_lines":
        tables_dict["orders"] = TableMapping(
            derived_from="order_lines",
            columns=ord_cols,
        )
    else:
        ord_table_name = chosen_tables["orders"]
        ord_rt = raw_by_name[ord_table_name]
        tables_dict["orders"] = TableMapping(
            sources=_build_source_specs_from_raw_table(ord_rt, raw_by_name),
            columns=ord_cols,
        )

    customer_identity = "by_code" if "customer_id" in ord_cols else "by_name"
    data_end_date = extract_data_end_date(profile.readme_text)
    notes: list[str] = []
    if "status" not in ord_cols:
        notes.append(_NO_STATUS_NOTE)

    return Mapping(
        business_id=profile.business_name,
        data_end_date=data_end_date,
        customer_identity=customer_identity,
        walk_in_values=walk_in_values,
        return_rule=return_rule,
        status_map=status_map,
        tables=tables_dict,
        notes=notes,
    )


def _parse_step_a_roles(
    ans_a: dict[str, Any],
    valid_table_names: set[str],
) -> tuple[dict[Entity, str], str, str | None]:
    orders_source = ans_a.get("orders_source")
    if orders_source not in ("own_table", "order_lines"):
        return {}, "order_lines", f"Invalid orders_source: {orders_source!r}"

    by_entity: dict[Entity, list[str]] = {}
    for item in ans_a.get("tables", []):
        tname = item.get("table")
        ent = item.get("entity")
        if tname not in valid_table_names or ent == "none" or ent not in _ALL_ENTITIES:
            continue
        by_entity.setdefault(ent, []).append(tname)

    ol_list = by_entity.get("order_lines", [])
    if not ol_list and len(by_entity.get("orders", [])) == 1:
        # An invoice-only file (one row per order, no line items) is its own line table:
        # each order is read as one line, the same as orders_source "order_lines".
        ol_list = by_entity.pop("orders")
        orders_source = "order_lines"
    if len(ol_list) != 1:
        return (
            {},
            orders_source,
            f"Expected exactly 1 table for 'order_lines', got {ol_list}",
        )

    chosen: dict[Entity, str] = {"order_lines": ol_list[0]}
    if orders_source == "own_table":
        ord_list = by_entity.get("orders", [])
        if len(ord_list) != 1:
            return (
                {},
                orders_source,
                f"orders_source is 'own_table' but 'orders' tables={ord_list}",
            )
        if ord_list[0] == ol_list[0]:
            orders_source = "order_lines"
        else:
            chosen["orders"] = ord_list[0]

    for opt_ent in ("products", "unit_conversions", "customers"):
        opt_typed: Entity = opt_ent  # type: ignore[assignment]
        lst = by_entity.get(opt_typed, [])
        if len(lst) == 1:
            chosen[opt_typed] = lst[0]

    return chosen, orders_source, None


def infer_mapping(
    raw: BusinessRaw,
    profile: BusinessProfile,
    client: Any,
) -> InferResult:
    budget = settings.llm_prompt_budget_tokens
    steps_log: list[StepLog] = []
    errors: list[str] = []

    raw_by_name: dict[str, RawTable] = {t.name: t for t in raw.tables}
    prof_by_name: dict[str, TableProfile] = {t.name: t for t in profile.tables}
    table_names = [t.name for t in profile.tables]

    if not table_names:
        return InferResult(mapping=None, steps=steps_log, errors=["No raw tables in business"])

    step_a_variants = _build_step_a_variants(profile)
    step_a_schema = _step_a_schema(table_names)

    chosen_tables: dict[Entity, str] = {}
    orders_source = "order_lines"
    last_ans_a: dict[str, Any] = {}

    def _run_step_a_with_role_retries(initial_repair_suffix: str | None = None) -> bool:
        nonlocal chosen_tables, orders_source, last_ans_a
        a_repair_msg = initial_repair_suffix
        for _attempt in range(3):
            try:
                ans_a, log_a = _call_step(
                    client=client,
                    step_name="A",
                    prompt_variants=step_a_variants,
                    schema=step_a_schema,
                    schema_name="step_a_table_roles",
                    budget=budget,
                    repair_suffix=a_repair_msg,
                )
                steps_log.append(log_a)
                last_ans_a = ans_a
            except LLMUnavailable:
                raise
            except LLMError as exc:
                errors.append(f"Step A LLMError: {exc}")
                return False

            new_chosen, new_orders_source, role_err = _parse_step_a_roles(
                last_ans_a, set(table_names)
            )
            if role_err is None:
                sales_err = check_order_lines_sales_table(
                    new_chosen, new_orders_source, prof_by_name
                )
                if sales_err is None:
                    chosen_tables = new_chosen
                    orders_source = new_orders_source
                    return True
                role_err = sales_err
            a_repair_msg = (
                f"Previous answer: {json.dumps(last_ans_a, ensure_ascii=False)}\n"
                f"Problem: {role_err}. Fix it."
            )
        errors.append(f"Step A failed role constraints: {a_repair_msg}")
        return False

    if not _run_step_a_with_role_retries():
        return InferResult(mapping=None, steps=steps_log, errors=errors)

    entity_columns: dict[Entity, dict[str, str]] = {}
    last_ans_b: dict[Entity, dict[str, Any]] = {}

    def _effective_table_for_entity(
        ent: Entity, c_tables: dict[Entity, str], o_source: str
    ) -> str | None:
        if ent == "orders":
            return (
                c_tables.get("order_lines") if o_source == "order_lines" else c_tables.get("orders")
            )
        return c_tables.get(ent)

    def _run_step_b(entity: Entity, repair_suffix: str | None = None) -> bool:
        tname = _effective_table_for_entity(entity, chosen_tables, orders_source)
        if tname is None:
            return False
        tp = prof_by_name[tname]
        rt = raw_by_name[tname]
        variants = _build_step_b_variants(
            entity, tp, profile.readme_text, orders_source=orders_source
        )
        schema = _step_b_schema(entity, rt.columns, orders_source)
        try:
            ans_b, log_b = _call_step(
                client=client,
                step_name=f"B_{entity}",
                prompt_variants=variants,
                schema=schema,
                schema_name=f"step_b_{entity}",
                budget=budget,
                repair_suffix=repair_suffix,
                required_keys=_step_b_required_names(entity, orders_source),
            )
            steps_log.append(log_b)
            last_ans_b[entity] = ans_b
        except LLMUnavailable:
            raise
        except LLMError as exc:
            errors.append(f"Step B_{entity} LLMError: {exc}")
            return False

        valid_cols = set(rt.columns)
        cleaned: dict[str, str] = {}
        for k, v in ans_b.items():
            if isinstance(v, str) and v in valid_cols:
                cleaned[k] = v
        entity_columns[entity] = cleaned
        return True

    b_order: list[Entity] = ["order_lines", "orders"]
    for opt_ent in ("products", "unit_conversions", "customers"):
        opt_typed: Entity = opt_ent  # type: ignore[assignment]
        if opt_typed in chosen_tables:
            b_order.append(opt_typed)

    for ent in b_order:
        if not _run_step_b(ent):
            return InferResult(mapping=None, steps=steps_log, errors=errors)

    status_map: dict[str, OrderStatus] = {}
    last_ans_c: dict[str, Any] = {}
    early_warnings: list[Issue] = []

    def _drop_status_as_free_text(n_distinct: int) -> None:
        nonlocal status_map
        ord_cols = entity_columns.setdefault("orders", {})
        ord_cols.pop("status", None)
        status_map = {}
        early_warnings.append(
            Issue(
                step="C",
                severity="warning",
                message=(
                    f"status column looks like free text ({n_distinct} distinct values > "
                    f"{_STATUS_DISTINCT_LIMIT}); dropped status mapping"
                ),
            )
        )

    def _run_step_c(repair_suffix: str | None = None) -> bool:
        nonlocal status_map, last_ans_c
        ord_tname = _effective_table_for_entity("orders", chosen_tables, orders_source)
        if ord_tname is None:
            status_map = {}
            return True
        ord_rt = raw_by_name[ord_tname]
        st_col = entity_columns.get("orders", {}).get("status")
        if not st_col:
            status_map = {}
            return True
        n_distinct = _n_distinct_nonempty(ord_rt, st_col)
        if n_distinct > _STATUS_DISTINCT_LIMIT:
            _drop_status_as_free_text(n_distinct)
            return True
        st_counts = _collect_distinct_counts(ord_rt, st_col, limit=50)
        if not st_counts:
            status_map = {}
            return True
        st_vals = [v for v, _ in st_counts]
        variants = _build_step_c_variants(st_counts, profile.readme_text)
        schema = _step_c_schema(st_vals)
        try:
            ans_c, log_c = _call_step(
                client=client,
                step_name="C",
                prompt_variants=variants,
                schema=schema,
                schema_name="step_c_status_map",
                budget=budget,
                repair_suffix=repair_suffix,
            )
            steps_log.append(log_c)
            last_ans_c = ans_c
        except LLMUnavailable:
            raise
        except LLMError as exc:
            errors.append(f"Step C LLMError: {exc}")
            return False

        valid_states = {"completed", "cancelled", "returned", "shipping"}
        status_map = {
            k: v  # type: ignore[misc]
            for k, v in ans_c.items()
            if isinstance(v, str) and v in valid_states
        }
        return True

    if not _run_step_c():
        return InferResult(mapping=None, steps=steps_log, errors=errors)

    walk_in_values: list[str] = []
    return_rule: ReturnRule = "status"
    last_ans_d: dict[str, Any] = {}

    def _run_step_d(repair_suffix: str | None = None) -> bool:
        nonlocal walk_in_values, return_rule, last_ans_d
        ord_tname = _effective_table_for_entity("orders", chosen_tables, orders_source)
        assert ord_tname is not None
        ord_rt = raw_by_name[ord_tname]
        ol_rt = raw_by_name[chosen_tables["order_lines"]]
        ord_cols = entity_columns.get("orders", {})
        # Take candidates from orders.customer_name if mapped, else customer_id
        cust_col = ord_cols.get("customer_name") or ord_cols.get("customer_id")
        cust_counts = _collect_walk_in_candidates(ord_rt, cust_col, limit=30) if cust_col else []
        cand_vals = [v for v, _ in cust_counts]

        qty_col = entity_columns.get("order_lines", {}).get("qty")
        has_neg = _has_negative_qty(ol_rt, qty_col)

        variants = _build_step_d_variants(cust_counts, has_neg, profile.readme_text)
        schema = _step_d_schema(cand_vals)
        try:
            ans_d, log_d = _call_step(
                client=client,
                step_name="D",
                prompt_variants=variants,
                schema=schema,
                schema_name="step_d_customers_returns",
                budget=budget,
                repair_suffix=repair_suffix,
            )
            steps_log.append(log_d)
            last_ans_d = ans_d
        except LLMUnavailable:
            raise
        except LLMError as exc:
            errors.append(f"Step D LLMError: {exc}")
            return False

        raw_wiv = ans_d.get("walk_in_values", [])
        if not cand_vals:
            walk_in_values = []
        else:
            allowed_cands = set(cand_vals)
            walk_in_values = [
                v
                for v in raw_wiv
                if isinstance(v, str) and not is_id_or_phone_like(v) and v in allowed_cands
            ]
        rr = ans_d.get("return_rule", "status")
        return_rule = "negative_qty" if rr == "negative_qty" else "status"
        return True

    if not _run_step_d():
        return InferResult(mapping=None, steps=steps_log, errors=errors)

    mapping = _assemble_mapping(
        profile=profile,
        raw_by_name=raw_by_name,
        chosen_tables=chosen_tables,
        orders_source=orders_source,
        entity_columns=entity_columns,
        status_map=status_map,
        walk_in_values=walk_in_values,
        return_rule=return_rule,
    )

    issues, canon = validate_mapping(raw, mapping)
    dropped_notes: list[Issue] = list(early_warnings)
    retried_drops: set[str] = set()
    issues = [*dropped_notes, *issues]

    # Repair loop: at most 2 repair rounds.
    # Optional-field consistency failures are re-asked once, then that field is dropped.
    for _round in range(2):
        blocking = [iss for iss in issues if iss.severity == "error" and not iss.drop_field]
        droppable = [iss for iss in issues if iss.drop_field]
        fresh: list[Issue] = []
        persist: list[Issue] = []
        seen_persist: set[str] = set()
        for iss in droppable:
            key = iss.drop_field
            if key is None:
                continue
            if key in retried_drops:
                if key not in seen_persist:
                    persist.append(iss)
                    seen_persist.add(key)
            else:
                fresh.append(iss)

        for iss in persist:
            qualified = iss.drop_field or ""
            entity_name, field_name = qualified.split(".", 1)
            ent_drop: Entity = entity_name  # type: ignore[assignment]
            cols = entity_columns.get(ent_drop)
            if cols is not None:
                cols.pop(field_name, None)
            dropped_notes.append(
                Issue(
                    step=iss.step,
                    severity="warning",
                    message=f"dropped {field_name}: {iss.message}",
                )
            )

        if not blocking and not fresh:
            if not persist:
                break
            mapping = _assemble_mapping(
                profile=profile,
                raw_by_name=raw_by_name,
                chosen_tables=chosen_tables,
                orders_source=orders_source,
                entity_columns=entity_columns,
                status_map=status_map,
                walk_in_values=walk_in_values,
                return_rule=return_rule,
            )
            fresh_issues, canon = validate_mapping(raw, mapping)
            issues = [*dropped_notes, *fresh_issues]
            continue

        problems_by_step: dict[str, list[str]] = {}
        for iss in (*blocking, *fresh):
            problems_by_step.setdefault(iss.step, []).append(iss.message)
        for iss in fresh:
            if iss.drop_field is not None:
                retried_drops.add(iss.drop_field)

        status_col_before = entity_columns.get("orders", {}).get("status")
        old_chosen = dict(chosen_tables)
        old_orders_source = orders_source
        rerun_b_from_a: set[Entity] = set()
        need_rerun_c_from_a = False
        need_rerun_d_from_a = False

        if "A" in problems_by_step:
            combined_a = " ; ".join(problems_by_step["A"])
            prev_a_json = json.dumps(last_ans_a, ensure_ascii=False)
            suffix_a = f"Previous answer: {prev_a_json}. Problem: {combined_a}. Fix it."
            if not _run_step_a_with_role_retries(initial_repair_suffix=suffix_a):
                return InferResult(
                    mapping=mapping, steps=steps_log, errors=errors, issues=issues, canonical=canon
                )

            # Determine which entities had their effective table or orders_source changed
            for ent in _ALL_ENTITIES:
                old_eff = _effective_table_for_entity(ent, old_chosen, old_orders_source)
                new_eff = _effective_table_for_entity(ent, chosen_tables, orders_source)
                if new_eff is None:
                    entity_columns.pop(ent, None)
                elif new_eff != old_eff or (ent == "orders" and orders_source != old_orders_source):
                    rerun_b_from_a.add(ent)

            if "order_lines" in rerun_b_from_a or "orders" in rerun_b_from_a:
                need_rerun_c_from_a = True
                need_rerun_d_from_a = True

        # Run Step B for entities changed by Step A or explicitly flagged in problems_by_step
        for ent in ("order_lines", "orders", "products", "unit_conversions", "customers"):
            ent_typed: Entity = ent  # type: ignore[assignment]
            b_step_key = f"B_{ent}"
            if b_step_key in problems_by_step and ent_typed not in rerun_b_from_a:
                combined_problem = " ; ".join(problems_by_step[b_step_key])
                prev_json = json.dumps(last_ans_b.get(ent_typed, {}), ensure_ascii=False)
                suffix = f"Previous answer: {prev_json}. Problem: {combined_problem}. Fix it."
                if not _run_step_b(ent_typed, repair_suffix=suffix):
                    return InferResult(
                        mapping=mapping,
                        steps=steps_log,
                        errors=errors,
                        issues=issues,
                        canonical=canon,
                    )
            elif ent_typed in rerun_b_from_a:
                if not _run_step_b(ent_typed, repair_suffix=None):
                    return InferResult(
                        mapping=mapping,
                        steps=steps_log,
                        errors=errors,
                        issues=issues,
                        canonical=canon,
                    )

        # Run Step C if flagged, or if orders table / status column changed
        status_col_after = entity_columns.get("orders", {}).get("status")
        if "C" in problems_by_step and not need_rerun_c_from_a:
            combined_c = " ; ".join(problems_by_step["C"])
            prev_json = json.dumps(last_ans_c, ensure_ascii=False)
            suffix = f"Previous answer: {prev_json}. Problem: {combined_c}. Fix it."
            if not _run_step_c(repair_suffix=suffix):
                return InferResult(
                    mapping=mapping, steps=steps_log, errors=errors, issues=issues, canonical=canon
                )
        elif need_rerun_c_from_a or status_col_after != status_col_before:
            if not _run_step_c():
                return InferResult(
                    mapping=mapping, steps=steps_log, errors=errors, issues=issues, canonical=canon
                )

        # Run Step D if flagged, or if orders / order_lines table changed from Step A
        if "D" in problems_by_step and not need_rerun_d_from_a:
            combined_d = " ; ".join(problems_by_step["D"])
            prev_json = json.dumps(last_ans_d, ensure_ascii=False)
            suffix = f"Previous answer: {prev_json}. Problem: {combined_d}. Fix it."
            if not _run_step_d(repair_suffix=suffix):
                return InferResult(
                    mapping=mapping, steps=steps_log, errors=errors, issues=issues, canonical=canon
                )
        elif need_rerun_d_from_a and not _run_step_d():
            return InferResult(
                mapping=mapping, steps=steps_log, errors=errors, issues=issues, canonical=canon
            )

        mapping = _assemble_mapping(
            profile=profile,
            raw_by_name=raw_by_name,
            chosen_tables=chosen_tables,
            orders_source=orders_source,
            entity_columns=entity_columns,
            status_map=status_map,
            walk_in_values=walk_in_values,
            return_rule=return_rule,
        )
        fresh_issues, canon = validate_mapping(raw, mapping)
        issues = [*dropped_notes, *fresh_issues]

    leftover: list[Issue] = []
    seen_leftover: set[str] = set()
    for iss in issues:
        key = iss.drop_field
        if key is None or key not in retried_drops or key in seen_leftover:
            continue
        leftover.append(iss)
        seen_leftover.add(key)
    if leftover:
        for iss in leftover:
            qualified = iss.drop_field or ""
            entity_name, field_name = qualified.split(".", 1)
            ent_drop: Entity = entity_name  # type: ignore[assignment]
            cols = entity_columns.get(ent_drop)
            if cols is not None:
                cols.pop(field_name, None)
            dropped_notes.append(
                Issue(
                    step=iss.step,
                    severity="warning",
                    message=f"dropped {field_name}: {iss.message}",
                )
            )
        mapping = _assemble_mapping(
            profile=profile,
            raw_by_name=raw_by_name,
            chosen_tables=chosen_tables,
            orders_source=orders_source,
            entity_columns=entity_columns,
            status_map=status_map,
            walk_in_values=walk_in_values,
            return_rule=return_rule,
        )
        fresh_issues, canon = validate_mapping(raw, mapping)
        issues = [*dropped_notes, *fresh_issues]

    remaining_errs = [iss.message for iss in issues if iss.severity == "error"]
    return InferResult(
        mapping=mapping,
        steps=steps_log,
        errors=remaining_errs,
        issues=issues,
        canonical=canon,
    )

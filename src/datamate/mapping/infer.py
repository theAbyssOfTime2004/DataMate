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
from datamate.mapping.fields import (
    FIELD_DOCS,
    LATER_FIELDS,
    REQUIRED_FIELDS,
    SIMPLE_AT_LEAST_ONE,
    SIMPLE_ENTITIES,
    SIMPLE_REQUIRED,
)
from datamate.mapping.model import (
    Entity,
    JoinSpec,
    Mapping,
    OrderStatus,
    PurchaseStatus,
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
_ORDER_ENTITIES: tuple[Entity, ...] = (
    "orders",
    "order_lines",
    "products",
    "unit_conversions",
    "customers",
)
_ALL_ENTITIES: tuple[Entity, ...] = (*_ORDER_ENTITIES, *SIMPLE_ENTITIES)
_SIMPLE_ENTITY_SET: frozenset[Entity] = frozenset(SIMPLE_ENTITIES)
_PURCHASE_SHARE: frozenset[Entity] = frozenset({"purchases", "purchase_lines"})

_SIMPLE_ENTITY_LINES: tuple[tuple[Entity, str], ...] = (
    ("payments", "one payment / receipt of money from a customer (phiếu thu, thanh toán)"),
    ("debts", "one customer debt / AR balance row (công nợ, phải thu)"),
    ("purchases", "one purchase receipt header (phiếu nhập, nhập hàng)"),
    ("purchase_lines", "one line on a purchase receipt (chi tiết nhập)"),
    ("suppliers", "one supplier in a directory (nhà cung cấp)"),
    ("stock_snapshot", "on-hand qty for one product (tồn kho)"),
    ("stock_moves", "one stock in/out movement (xuất nhập tồn, thẻ kho)"),
    ("lots", "one lot / batch with expiry (lô, hạn dùng)"),
    ("sale_lots", "which lot was sold on an order line (lô bán)"),
    ("price_history", "one historical selling price (lịch sử giá)"),
    ("expenses", "one operating expense (chi phí, sổ chi)"),
    ("employees", "one employee (nhân viên)"),
    ("deliveries", "one delivery batch of an order (đợt giao)"),
    ("drug_info", "drug active ingredient / Rx flag (hoạt chất, kê đơn)"),
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
        simple_block = "\n".join(
            f"- `{ent}`: {desc}" for ent, desc in _SIMPLE_ENTITY_LINES
        )
        prompt = (
            "Assign each raw table of this shop to its canonical entity role:\n"
            "- `order_lines`: individual line items sold per order (or flat sales ledger where each row is a line item)\n"
            "- `orders`: separate order header / invoice summary table (one row per order)\n"
            "- `products`: product / item catalog / price list\n"
            "- `unit_conversions`: unit conversion factor table (larger unit -> base unit)\n"
            "- `customers`: customer directory / list\n"
            f"{simple_block}\n"
            "- `none`: table not used for any entity above\n\n"
            "Also set `orders_source`:\n"
            "- `own_table` if there is a separate `orders` header table;\n"
            "- `order_lines` if there is no separate `orders` header table and order fields live on `order_lines`.\n"
            "Rules: Exactly ONE table must be assigned `order_lines`. If `orders_source` is `own_table`, "
            "exactly ONE table must be assigned `orders` (otherwise no table is `orders`). "
            "Each other entity takes 0 or 1 table. A raw table may serve only one entity, except a flat "
            "purchase sheet may be both `purchases` and `purchase_lines` (header fields repeated on each row).\n\n"
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


def _required_for_entity(entity: Entity, orders_source: str = "own_table") -> set[str]:
    if entity in _SIMPLE_ENTITY_SET:
        return set(SIMPLE_REQUIRED.get(entity, set()))
    required = set(REQUIRED_FIELDS.get(entity, ()))
    if entity == "orders" and orders_source == "own_table":
        required.add("order_id")
    return required


def _build_step_b_variants(
    entity: Entity,
    tp: TableProfile,
    readme_text: str | None,
    orders_source: str = "own_table",
) -> list[str]:
    fdocs = _step_b_field_docs(entity)
    req = _required_for_entity(entity, orders_source)
    field_lines = [
        f"- `{fname}` ({'REQUIRED' if fname in req else 'optional'}): {desc}"
        for fname, desc in fdocs.items()
    ]
    at_least = SIMPLE_AT_LEAST_ONE.get(entity)
    if at_least is not None:
        joined = " / ".join(f"`{f}`" for f in sorted(at_least))
        field_lines.append(f"- At least one of {joined} is required.")
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
    required = _required_for_entity(entity, orders_source)
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


def _build_purchase_status_variants(
    status_counts: list[tuple[str, int]],
    readme_text: str | None,
) -> list[str]:
    conv = _readme_conventions_only(readme_text)
    vals_lines = [f'- "{val}" (count: {cnt})' for val, cnt in status_counts]
    vals_block = "\n".join(vals_lines)
    variants: list[str] = []
    for cap in (2000, 800, 300):
        prompt = (
            "Map every distinct raw purchase receipt status to one of:\n"
            "- `valid`: receipt counts (hợp lệ, đã nhập, hoàn thành, ok)\n"
            "- `cancelled`: receipt was voided (huỷ, đã hủy, hủy phiếu)\n\n"
            f"### README Conventions\n{conv[:cap]}\n\n"
            f"### Distinct Raw Purchase Status Values\n{vals_block}"
        )
        variants.append(prompt)
    return variants


def _purchase_status_schema(status_values: list[str]) -> dict[str, Any]:
    state_enum: list[PurchaseStatus] = ["valid", "cancelled"]
    props = {val: {"type": "string", "enum": state_enum} for val in status_values}
    return {
        "type": "object",
        "properties": props,
        "required": status_values,
        "additionalProperties": False,
    }


def _unmapped_raw_columns(rt: RawTable, mapped_cols: dict[str, str]) -> list[str]:
    used = set(mapped_cols.values())
    return [c for c in rt.columns if c not in used]


def _later_fields_schema(field_names: list[str], free_columns: list[str]) -> dict[str, Any]:
    col_enum: list[str | None] = [*free_columns, None]
    props = {fname: {"enum": col_enum} for fname in field_names}
    return {
        "type": "object",
        "properties": props,
        "required": field_names,
        "additionalProperties": False,
    }


def _build_later_fields_variants(
    entity: Entity,
    field_names: list[str],
    tp: TableProfile,
    free_columns: list[str],
    readme_text: str | None,
) -> list[str]:
    fdocs = FIELD_DOCS[entity]
    field_lines = [f"- `{fname}`: {fdocs[fname]}" for fname in field_names]
    fields_block = "\n".join(field_lines)
    free_block = ", ".join(f"`{c}`" for c in free_columns) or "(none)"
    readme_sec = _readme_for_table(readme_text, tp.source_files)
    variants: list[str] = []
    for max_samples, include_top, readme_cap in [
        (5, True, 2000),
        (2, False, 800),
        (1, False, 300),
    ]:
        card_str = _compact_table_card_for_step_b(
            tp, max_samples=max_samples, include_top=include_top
        )
        r_str = readme_sec[:readme_cap] if readme_sec else "(no README section)"
        prompt = (
            f"Map optional late fields of `{entity}` to columns that are still unmapped.\n"
            "Set a field to `null` if none of the free columns fit. Do not reuse a column "
            "already mapped to another field.\n\n"
            f"### Free columns\n{free_block}\n\n"
            f"### Fields\n{fields_block}\n\n"
            f"### README Context\n{r_str}\n\n"
            f"### Table Profile\n{card_str}"
        )
        variants.append(prompt)
    return variants


def _simple_missing_required(entity: Entity, cleaned: dict[str, str]) -> list[str]:
    required = SIMPLE_REQUIRED.get(entity, set())
    missing = sorted(required - set(cleaned))
    at_least = SIMPLE_AT_LEAST_ONE.get(entity)
    if at_least is not None:
        if not (at_least & set(cleaned)):
            return ["|".join(sorted(at_least))]
        return []
    return missing


def _receipt_id_repeats(rt: RawTable, receipt_col: str) -> bool:
    """True when the mapped receipt_id column has more non-null rows than distinct ids."""
    if receipt_col not in rt.df.columns:
        return False
    seen: set[str] = set()
    n_nonempty = 0
    for raw_val in rt.df[receipt_col].tolist():
        if raw_val is None:
            continue
        text = raw_val.strip() if isinstance(raw_val, str) else str(raw_val).strip()
        if not text:
            continue
        n_nonempty += 1
        seen.add(text)
    return n_nonempty > len(seen)


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
    purchase_status_map: dict[str, PurchaseStatus] | None = None,
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

    for ent in SIMPLE_ENTITIES:
        if ent not in chosen_tables:
            continue
        cols_map = dict(entity_columns.get(ent, {}))
        if not cols_map:
            continue
        rt = raw_by_name[chosen_tables[ent]]
        tables_dict[ent] = TableMapping(
            sources=_build_source_specs_from_raw_table(rt, raw_by_name),
            columns=cols_map,
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
        purchase_status_map=dict(purchase_status_map or {}),
        tables=tables_dict,
        notes=notes,
    )


def _parse_step_a_roles(
    ans_a: dict[str, Any],
    valid_table_names: set[str],
) -> tuple[dict[Entity, str], str, str | None, list[str]]:
    """Parse step A. Role errors only for the order path; simple-entity conflicts are warnings."""
    role_warnings: list[str] = []
    orders_source = ans_a.get("orders_source")
    if orders_source not in ("own_table", "order_lines"):
        return {}, "order_lines", f"Invalid orders_source: {orders_source!r}", role_warnings

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
            role_warnings,
        )

    chosen: dict[Entity, str] = {"order_lines": ol_list[0]}
    if orders_source == "own_table":
        ord_list = by_entity.get("orders", [])
        if len(ord_list) != 1:
            return (
                {},
                orders_source,
                f"orders_source is 'own_table' but 'orders' tables={ord_list}",
                role_warnings,
            )
        if ord_list[0] == ol_list[0]:
            orders_source = "order_lines"
        else:
            chosen["orders"] = ord_list[0]

    # Optional order-path entities: 0 or 1 table; more than one -> ignore (legacy behaviour).
    for opt_ent in ("products", "unit_conversions", "customers"):
        opt_typed: Entity = opt_ent  # type: ignore[assignment]
        lst = by_entity.get(opt_typed, [])
        if len(lst) == 1:
            chosen[opt_typed] = lst[0]

    table_owner: dict[str, Entity] = {}
    table_owner[chosen["order_lines"]] = "order_lines"
    if "orders" in chosen:
        table_owner[chosen["orders"]] = "orders"
    for opt_ent in ("products", "unit_conversions", "customers"):
        opt_typed = opt_ent  # type: ignore[assignment]
        if opt_typed in chosen:
            table_owner[chosen[opt_typed]] = opt_typed

    for ent in SIMPLE_ENTITIES:
        lst = by_entity.get(ent, [])
        if len(lst) > 1:
            role_warnings.append(
                f"skipped {ent}: expected at most 1 table, got {lst}"
            )
            continue
        if not lst:
            continue
        tname = lst[0]
        owner = table_owner.get(tname)
        if owner is not None and owner != ent:
            share_ok = {owner, ent} <= _PURCHASE_SHARE
            if not share_ok:
                role_warnings.append(
                    f"skipped {ent}: table {tname!r} already assigned to {owner!r}"
                )
                continue
        chosen[ent] = tname
        table_owner[tname] = ent

    # Flat purchase sheet assigned only as purchase_lines: also map purchases from it.
    if "purchase_lines" in chosen and "purchases" not in chosen:
        chosen["purchases"] = chosen["purchase_lines"]

    return chosen, orders_source, None, role_warnings


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
    early_warnings: list[Issue] = []

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

            new_chosen, new_orders_source, role_err, role_warns = _parse_step_a_roles(
                last_ans_a, set(table_names)
            )
            if role_err is None:
                sales_err = check_order_lines_sales_table(
                    new_chosen, new_orders_source, prof_by_name
                )
                if sales_err is None:
                    chosen_tables = new_chosen
                    orders_source = new_orders_source
                    early_warnings[:] = [i for i in early_warnings if i.step != "A"]
                    for w in role_warns:
                        early_warnings.append(
                            Issue(step="A", severity="warning", message=w)
                        )
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

    purchase_status_map: dict[str, PurchaseStatus] = {}

    def _assemble_current() -> Mapping:
        return _assemble_mapping(
            profile=profile,
            raw_by_name=raw_by_name,
            chosen_tables=chosen_tables,
            orders_source=orders_source,
            entity_columns=entity_columns,
            status_map=status_map,
            walk_in_values=walk_in_values,
            return_rule=return_rule,
            purchase_status_map=purchase_status_map,
        )

    # Order-path assemble excludes simple entities until after the repair loop (cache keys).
    order_chosen = {
        ent: tname
        for ent, tname in chosen_tables.items()
        if ent not in _SIMPLE_ENTITY_SET
    }
    mapping = _assemble_mapping(
        profile=profile,
        raw_by_name=raw_by_name,
        chosen_tables=order_chosen,
        orders_source=orders_source,
        entity_columns=entity_columns,
        status_map=status_map,
        walk_in_values=walk_in_values,
        return_rule=return_rule,
        purchase_status_map={},
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
            order_chosen = {
                ent: tname
                for ent, tname in chosen_tables.items()
                if ent not in _SIMPLE_ENTITY_SET
            }
            mapping = _assemble_mapping(
                profile=profile,
                raw_by_name=raw_by_name,
                chosen_tables=order_chosen,
                orders_source=orders_source,
                entity_columns=entity_columns,
                status_map=status_map,
                walk_in_values=walk_in_values,
                return_rule=return_rule,
                purchase_status_map={},
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

        # Run Step B for order entities changed by Step A or explicitly flagged
        for ent in _ORDER_ENTITIES:
            ent_typed: Entity = ent
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

        order_chosen = {
            ent: tname
            for ent, tname in chosen_tables.items()
            if ent not in _SIMPLE_ENTITY_SET
        }
        mapping = _assemble_mapping(
            profile=profile,
            raw_by_name=raw_by_name,
            chosen_tables=order_chosen,
            orders_source=orders_source,
            entity_columns=entity_columns,
            status_map=status_map,
            walk_in_values=walk_in_values,
            return_rule=return_rule,
            purchase_status_map={},
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
        order_chosen = {
            ent: tname
            for ent, tname in chosen_tables.items()
            if ent not in _SIMPLE_ENTITY_SET
        }
        mapping = _assemble_mapping(
            profile=profile,
            raw_by_name=raw_by_name,
            chosen_tables=order_chosen,
            orders_source=orders_source,
            entity_columns=entity_columns,
            status_map=status_map,
            walk_in_values=walk_in_values,
            return_rule=return_rule,
            purchase_status_map={},
        )
        fresh_issues, canon = validate_mapping(raw, mapping)
        issues = [*dropped_notes, *fresh_issues]

    # ---- Order path is final. Later fields, then simple entities. ----

    def _drop_simple_entity(ent: Entity, reason: str) -> None:
        chosen_tables.pop(ent, None)
        entity_columns.pop(ent, None)
        if ent == "purchases":
            purchase_status_map.clear()
        dropped_notes.append(
            Issue(
                step=f"B_{ent}",
                severity="warning",
                message=f"dropped entity {ent}: {reason}",
            )
        )

    def _run_later_fields() -> None:
        for entity in ("orders", "customers"):
            ent: Entity = entity  # type: ignore[assignment]
            later = sorted(LATER_FIELDS.get(ent, frozenset()))
            if not later:
                continue
            if ent == "customers" and ent not in chosen_tables:
                continue
            tname = _effective_table_for_entity(ent, chosen_tables, orders_source)
            if tname is None:
                continue
            mapped = entity_columns.setdefault(ent, {})
            rt = raw_by_name[tname]
            free = _unmapped_raw_columns(rt, mapped)
            if not free:
                continue
            tp = prof_by_name[tname]
            variants = _build_later_fields_variants(
                ent, later, tp, free, profile.readme_text
            )
            schema = _later_fields_schema(later, free)
            try:
                ans_later, log_later = _call_step(
                    client=client,
                    step_name=f"B_{ent}_later",
                    prompt_variants=variants,
                    schema=schema,
                    schema_name=f"step_b_{ent}_later",
                    budget=budget,
                )
                steps_log.append(log_later)
            except LLMUnavailable:
                raise
            except LLMError as exc:
                dropped_notes.append(
                    Issue(
                        step=f"B_{ent}_later",
                        severity="warning",
                        message=f"later fields skipped: {exc}",
                    )
                )
                continue
            free_set = set(free)
            for fname in later:
                raw_col = ans_later.get(fname)
                if (
                    isinstance(raw_col, str)
                    and raw_col in free_set
                    and raw_col not in mapped.values()
                ):
                    mapped[fname] = raw_col
                    free_set.discard(raw_col)

    def _run_step_b_simple(entity: Entity, repair_suffix: str | None = None) -> None:
        tname = chosen_tables.get(entity)
        if tname is None:
            return
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
                required_keys=_step_b_required_names(entity, orders_source) or None,
            )
            steps_log.append(log_b)
            last_ans_b[entity] = ans_b
        except LLMUnavailable:
            raise
        except LLMError as exc:
            _drop_simple_entity(entity, f"LLMError: {exc}")
            return

        valid_cols = set(rt.columns)
        cleaned: dict[str, str] = {}
        for k, v in ans_b.items():
            if isinstance(v, str) and v in valid_cols:
                cleaned[k] = v
        missing = _simple_missing_required(entity, cleaned)
        if missing and repair_suffix is None:
            prev_json = json.dumps(ans_b, ensure_ascii=False)
            suffix = (
                f"Previous answer: {prev_json}. Problem: missing required field(s) "
                f"{missing}. Fix it or the entity will be dropped."
            )
            _run_step_b_simple(entity, repair_suffix=suffix)
            return
        if missing:
            _drop_simple_entity(entity, f"missing required field(s) {missing}")
            return
        entity_columns[entity] = cleaned

    def _maybe_assign_purchase_lines_from_flat_purchases() -> None:
        """Purchases-only flat sheet: also map purchase_lines when receipt_id repeats."""
        if "purchases" not in chosen_tables or "purchase_lines" in chosen_tables:
            return
        rid_col = entity_columns.get("purchases", {}).get("receipt_id")
        if not rid_col:
            return
        tname = chosen_tables["purchases"]
        rt = raw_by_name[tname]
        if not _receipt_id_repeats(rt, rid_col):
            return
        chosen_tables["purchase_lines"] = tname

    def _run_purchase_status() -> None:
        nonlocal purchase_status_map
        if "purchases" not in chosen_tables:
            purchase_status_map = {}
            return
        st_col = entity_columns.get("purchases", {}).get("status_raw")
        if not st_col:
            purchase_status_map = {}
            return
        rt = raw_by_name[chosen_tables["purchases"]]
        n_distinct = _n_distinct_nonempty(rt, st_col)
        if n_distinct > _STATUS_DISTINCT_LIMIT:
            purchase_status_map = {}
            dropped_notes.append(
                Issue(
                    step="C_purchases",
                    severity="warning",
                    message=(
                        f"purchase status_raw has {n_distinct} distinct values > "
                        f"{_STATUS_DISTINCT_LIMIT}; treating all as valid"
                    ),
                )
            )
            return
        st_counts = _collect_distinct_counts(rt, st_col, limit=50)
        if not st_counts:
            purchase_status_map = {}
            return
        st_vals = [v for v, _ in st_counts]
        variants = _build_purchase_status_variants(st_counts, profile.readme_text)
        schema = _purchase_status_schema(st_vals)
        try:
            ans_ps, log_ps = _call_step(
                client=client,
                step_name="C_purchases",
                prompt_variants=variants,
                schema=schema,
                schema_name="step_c_purchase_status_map",
                budget=budget,
            )
            steps_log.append(log_ps)
        except LLMUnavailable:
            raise
        except LLMError as exc:
            purchase_status_map = {}
            dropped_notes.append(
                Issue(
                    step="C_purchases",
                    severity="warning",
                    message=f"purchase status map skipped: {exc}",
                )
            )
            return
        valid_states = {"valid", "cancelled"}
        purchase_status_map = {
            k: v  # type: ignore[misc]
            for k, v in ans_ps.items()
            if isinstance(v, str) and v in valid_states
        }

    def _simple_entity_from_step(step: str) -> Entity | None:
        if step == "C_purchases":
            return "purchases"
        if step.startswith("B_") and step[2:] in _SIMPLE_ENTITY_SET:
            return step[2:]  # type: ignore[return-value]
        return None

    def _pick_field_to_drop(ent: Entity, fields: list[str]) -> str | None:
        """Prefer dropping *_ts when paired with *_date; else a non-required field."""
        req = SIMPLE_REQUIRED.get(ent, set())
        ts_fields = [f for f in fields if f.endswith("_ts")]
        date_fields = [f for f in fields if f.endswith("_date")]
        if ts_fields and date_fields:
            return ts_fields[0]
        non_req = [f for f in fields if f not in req]
        if non_req:
            return non_req[-1]
        return None

    def _fields_from_both_map_message(message: str) -> list[str]:
        # "{a} and {b} both map to ..." or "{a} and {b} and {c} both map to ..."
        if " both map to " not in message:
            return []
        left = message.split(" both map to ", 1)[0]
        return [p.strip() for p in left.split(" and ") if p.strip()]

    def _apply_simple_drops_from_issues(cur_issues: list[Issue]) -> bool:
        """Drop simple fields/entities so they never fail the mapping. Returns True if changed."""
        changed = False
        ents_to_drop: dict[Entity, str] = {}
        fields_to_drop: list[tuple[Entity, str, str]] = []

        def _mark_ent_drop(ent: Entity, reason: str) -> None:
            if ent not in ents_to_drop:
                ents_to_drop[ent] = reason

        for iss in cur_issues:
            ent_d = _simple_entity_from_step(iss.step)
            if ent_d is None:
                continue
            if iss.step == "C_purchases" and iss.severity == "error":
                _mark_ent_drop("purchases", iss.message)
                continue
            if iss.drop_field and "." in iss.drop_field:
                ent_name, field_name = iss.drop_field.split(".", 1)
                if ent_name != ent_d:
                    continue
                req = SIMPLE_REQUIRED.get(ent_d, set())
                at_least = SIMPLE_AT_LEAST_ONE.get(ent_d)
                last_at_least = (
                    at_least is not None
                    and field_name in at_least
                    and not (set(entity_columns.get(ent_d, {})) - {field_name}) & at_least
                )
                if field_name in req or last_at_least:
                    _mark_ent_drop(ent_d, iss.message)
                else:
                    fields_to_drop.append((ent_d, field_name, iss.message))
                continue
            if iss.severity != "error":
                continue
            # Duplicate-column / other errors without drop_field.
            both_fields = _fields_from_both_map_message(iss.message)
            if both_fields:
                pick = _pick_field_to_drop(ent_d, both_fields)
                if pick is None:
                    _mark_ent_drop(ent_d, iss.message)
                else:
                    fields_to_drop.append((ent_d, pick, iss.message))
                continue
            # Unknown simple error: drop the entity rather than fail the shop.
            _mark_ent_drop(ent_d, iss.message)

        for ent_d, field_name, reason in fields_to_drop:
            if ent_d in ents_to_drop:
                continue
            cols = entity_columns.get(ent_d)
            if cols is not None and field_name in cols:
                cols.pop(field_name, None)
                changed = True
                dropped_notes.append(
                    Issue(
                        step=f"B_{ent_d}",
                        severity="warning",
                        message=f"dropped {field_name}: {reason}",
                    )
                )
                at_least = SIMPLE_AT_LEAST_ONE.get(ent_d)
                if at_least is not None and not (set(cols) & at_least):
                    _mark_ent_drop(ent_d, reason)
                req = SIMPLE_REQUIRED.get(ent_d, set())
                if field_name in req or (req and not req <= set(cols)):
                    _mark_ent_drop(ent_d, reason)
        for ent_d, reason in ents_to_drop.items():
            if ent_d in chosen_tables:
                _drop_simple_entity(ent_d, reason)
                changed = True
        return changed

    _run_later_fields()
    for simple_ent in list(SIMPLE_ENTITIES):
        if simple_ent in chosen_tables:
            _run_step_b_simple(simple_ent)
            if simple_ent == "purchases":
                _maybe_assign_purchase_lines_from_flat_purchases()
    _run_purchase_status()

    mapping = _assemble_current()
    fresh_issues, canon = validate_mapping(raw, mapping)
    issues = [*dropped_notes, *fresh_issues]
    for _ in range(3):
        if not _apply_simple_drops_from_issues(issues):
            break
        mapping = _assemble_current()
        fresh_issues, canon = validate_mapping(raw, mapping)
        issues = [*dropped_notes, *fresh_issues]

    # Any simple entity that still carries an error after field drops is removed entirely
    # so pipeline never sees mapping_failed for a simple-entity problem.
    surviving_msgs: dict[Entity, str] = {}
    for iss in issues:
        if iss.severity != "error":
            continue
        ent = _simple_entity_from_step(iss.step)
        if ent is not None and ent in chosen_tables and ent not in surviving_msgs:
            surviving_msgs[ent] = iss.message
    if surviving_msgs:
        for ent, reason in surviving_msgs.items():
            _drop_simple_entity(ent, reason)
        mapping = _assemble_current()
        fresh_issues, canon = validate_mapping(raw, mapping)
        issues = [*dropped_notes, *fresh_issues]

    cleaned_issues: list[Issue] = []
    for iss in issues:
        if iss.severity == "error" and _simple_entity_from_step(iss.step) is not None:
            cleaned_issues.append(
                Issue(
                    step=iss.step,
                    severity="warning",
                    message=iss.message,
                    drop_field=iss.drop_field,
                )
            )
        else:
            cleaned_issues.append(iss)
    issues = cleaned_issues

    remaining_errs = [iss.message for iss in issues if iss.severity == "error"]
    return InferResult(
        mapping=mapping,
        steps=steps_log,
        errors=remaining_errs,
        issues=issues,
        canonical=canon,
    )

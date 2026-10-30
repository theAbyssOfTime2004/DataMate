from __future__ import annotations

import datetime
import re
from decimal import Decimal
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from datamate.io.raw import BusinessRaw
from datamate.mapping.fields import REQUIRED_FIELDS
from datamate.mapping.model import Entity, Mapping, mapping_fingerprint

if TYPE_CHECKING:
    from datamate.etl.build import CanonicalData

StepName = Literal[
    "A",
    "B_orders",
    "B_order_lines",
    "B_products",
    "B_unit_conversions",
    "B_customers",
    "C",
    "D",
]

_ID_CODE_RE = re.compile(r"^[A-Za-z]{1,6}[-_]?\d{2,}$")
_PHONE_LIKE_RE = re.compile(r"^0(?:\s*\d){9,10}$")
_ORDER_MONEY_FIELDS: tuple[str, ...] = (
    "order_subtotal",
    "order_discount",
    "shipping_fee",
    "order_payable",
)
_ALLOWED_SHARED_PAIRS: frozenset[frozenset[str]] = frozenset(
    {
        frozenset({"order_date", "order_ts"}),
        frozenset({"qty", "unit"}),
    }
)


def is_id_or_phone_like(value: str) -> bool:
    s = value.strip()
    return bool(_ID_CODE_RE.match(s) or _PHONE_LIKE_RE.match(s))


class Issue(BaseModel):
    step: str
    severity: Literal["error", "warning"]
    message: str
    # When set, infer may re-ask once and then drop this optional field instead of blocking.
    drop_field: str | None = None


# Last successful build, keyed by business directory + mapping_fingerprint.
_last_validated_build: tuple[str, CanonicalData] | None = None


def _build_cache_key(raw: BusinessRaw, mapping: Mapping) -> str:
    return f"{raw.business_dir}|{mapping_fingerprint(mapping)}"


def cached_canonical(raw: BusinessRaw, mapping: Mapping) -> CanonicalData | None:
    """Return the canonical frames from the latest validate_mapping of this mapping."""
    cached = _last_validated_build
    if cached is None or cached[0] != _build_cache_key(raw, mapping):
        return None
    return cached[1]


def _check_line_math_share(canon: CanonicalData) -> float | None:
    df = canon.order_lines
    if df.empty:
        return None
    checked = 0
    matched = 0
    for row in df.to_dict("records"):
        qty = row.get("qty")
        uprice = row.get("unit_price")
        lamt = row.get("line_amount")
        ldisc = row.get("line_discount")
        if not isinstance(qty, Decimal) or uprice is None or lamt is None:
            continue
        if qty < 0:
            continue
        checked += 1
        disc_val = Decimal(int(ldisc)) if ldisc is not None else Decimal(0)
        expected_1 = (qty * Decimal(int(uprice))) - disc_val
        expected_2 = qty * (Decimal(int(uprice)) - disc_val)
        actual = Decimal(int(lamt))
        if abs(expected_1 - actual) <= Decimal(1) or abs(expected_2 - actual) <= Decimal(1):
            matched += 1
    if checked == 0:
        return None
    return matched / checked


def validate_mapping(
    raw: BusinessRaw, mapping: Mapping
) -> tuple[list[Issue], CanonicalData | None]:
    issues: list[Issue] = []

    for entity_name, req_fields in REQUIRED_FIELDS.items():
        ent: Entity = entity_name  # type: ignore[assignment]
        tm = mapping.tables.get(ent)
        if tm is None:
            issues.append(
                Issue(
                    step=f"B_{entity_name}",
                    severity="error",
                    message=f"Entity {entity_name!r} is missing from mapping.tables.",
                )
            )
            continue
        mapped_keys = set(tm.columns.keys()) | set(tm.join.keys())
        missing = sorted(req_fields - mapped_keys)
        if missing:
            issues.append(
                Issue(
                    step=f"B_{entity_name}",
                    severity="error",
                    message=(
                        f"Entity {entity_name!r} is missing required canonical field(s): "
                        f"{missing}. Mapped columns: {tm.columns}"
                    ),
                )
            )

    orders_tm = mapping.tables.get("orders")
    if orders_tm is not None and "status" not in orders_tm.columns:
        issues.append(
            Issue(
                step="B_orders",
                severity="warning",
                message="no status column: all orders completed",
            )
        )

    # Check duplicate raw column within the same entity, allowing (order_date, order_ts) and (qty, unit)
    for ent, tm in mapping.tables.items():
        by_raw_col: dict[str, list[str]] = {}
        for c_field, raw_col in tm.columns.items():
            by_raw_col.setdefault(raw_col, []).append(c_field)
        for raw_col, c_fields in by_raw_col.items():
            if len(c_fields) > 1 and frozenset(c_fields) not in _ALLOWED_SHARED_PAIRS:
                fields_joined = " and ".join(c_fields)
                issues.append(
                    Issue(
                        step=f"B_{ent}",
                        severity="error",
                        message=f"{fields_joined} both map to {raw_col!r}",
                    )
                )

    orders_tm = mapping.tables["orders"]
    lines_tm = mapping.tables["order_lines"]

    # order_id stays required only when orders come from their own table.
    missing_own_order_id = (
        bool(orders_tm.sources)
        and "order_id" not in orders_tm.columns
        and "order_id" not in orders_tm.join
    )
    if missing_own_order_id:
        issues.append(
            Issue(
                step="B_orders",
                severity="error",
                message=(
                    "Entity 'orders' has its own table but is missing required canonical "
                    "field 'order_id'."
                ),
            )
        )

    # Check derived orders mapping order-level money fields to a line-level column
    if orders_tm.derived_from == "order_lines":
        line_col_to_field = {raw_col: l_field for l_field, raw_col in lines_tm.columns.items()}
        for om_field in _ORDER_MONEY_FIELDS:
            raw_col = orders_tm.columns.get(om_field)
            if raw_col is not None and raw_col in line_col_to_field:
                l_field = line_col_to_field[raw_col]
                issues.append(
                    Issue(
                        step="B_orders",
                        severity="error",
                        drop_field=f"orders.{om_field}",
                        message=(
                            f"{om_field} maps to {raw_col!r}, which is a line-level column "
                            f"({l_field}); in a flat sales table order-level totals usually do "
                            "not exist: set it to null unless a column holds the whole-order total"
                        ),
                    )
                )

    # Check Step D walk_in_values constraints
    bad_walk_in = [v for v in mapping.walk_in_values if is_id_or_phone_like(v)]
    if bad_walk_in:
        issues.append(
            Issue(
                step="D",
                severity="error",
                message=(
                    f"walk_in_values contains id-like or phone-like value(s) {bad_walk_in}; "
                    "only select generic anonymous customer labels."
                ),
            )
        )
    if len(mapping.walk_in_values) > 3:
        issues.append(
            Issue(
                step="D",
                severity="error",
                message=(
                    f"walk_in_values has {len(mapping.walk_in_values)} values "
                    f"({mapping.walk_in_values}), which exceeds the maximum of 3."
                ),
            )
        )

    from datamate.etl.build import MappingError, build_canonical

    global _last_validated_build
    cached = cached_canonical(raw, mapping)
    if cached is not None:
        canon = cached
    else:
        try:
            canon = build_canonical(raw, mapping)
        except MappingError as exc:
            msg = str(exc)
            step_target = "A"
            for ent_key in ("orders", "order_lines", "products", "unit_conversions", "customers"):
                if f"Entity {ent_key!r} field" in msg:
                    step_target = f"B_{ent_key}"
                    break
            issues.append(Issue(step=step_target, severity="error", message=msg))
            return issues, None
        _last_validated_build = (_build_cache_key(raw, mapping), canon)

    rep = canon.report

    # Check parse rates < 0.98 for order_date, line_amount, qty
    parse_checks = [
        ("orders.order_date", "B_orders"),
        ("order_lines.line_amount", "B_order_lines"),
        ("order_lines.qty", "B_order_lines"),
    ]
    for stat_key, step_id in parse_checks:
        st = rep.parse_stats.get(stat_key)
        if st is None or st.non_null == 0:
            if stat_key in {"orders.order_date", "order_lines.line_amount"}:
                issues.append(
                    Issue(
                        step=step_id,
                        severity="error",
                        message=f"Field {stat_key!r} has 0 non-null values.",
                    )
                )
            continue
        if st.rate < 0.98:
            issues.append(
                Issue(
                    step=step_id,
                    severity="error",
                    message=(
                        f"Field {stat_key!r} parse rate is {st.rate:.1%} (< 98%, "
                        f"{st.parsed}/{st.non_null} parsed). Failing examples: {st.examples}"
                    ),
                )
            )

    # Unmapped statuses -> step C
    if rep.unmapped_statuses:
        issues.append(
            Issue(
                step="C",
                severity="error",
                message=(
                    f"Unmapped status values found in data: {rep.unmapped_statuses}. "
                    "Every raw status value must map to one of: "
                    "completed, cancelled, returned, shipping."
                ),
            )
        )

    # Subtotal match share < 0.9 when order_subtotal is mapped
    integ = rep.integrity
    if (
        "order_subtotal" in orders_tm.columns
        and integ.subtotal_match_share is not None
        and integ.subtotal_match_share < 0.9
    ):
        diff_pct = (1.0 - integ.subtotal_match_share) * 100.0
        sub_col = orders_tm.columns.get("order_subtotal")
        lamt_col = lines_tm.columns.get("line_amount")
        if orders_tm.derived_from == "order_lines":
            issues.append(
                Issue(
                    step="B_orders",
                    severity="error",
                    drop_field="orders.order_subtotal",
                    message=(
                        f"sum of line_amount ({lamt_col!r}) differs from order_subtotal "
                        f"({sub_col!r}) in {diff_pct:.1f}% of orders "
                        f"(match share={integ.subtotal_match_share:.1%}): "
                        "order_subtotal is the likely culprit; in a flat sales table "
                        "order-level totals usually do not exist, so set order_subtotal to null "
                        "unless a column truly holds the whole-order merchandise subtotal."
                    ),
                )
            )
        else:
            issues.append(
                Issue(
                    step="B_order_lines",
                    severity="error",
                    drop_field="orders.order_subtotal",
                    message=(
                        f"sum of line_amount ({lamt_col!r}) differs from order_subtotal "
                        f"({sub_col!r}) in {diff_pct:.1f}% of orders "
                        f"(match share={integ.subtotal_match_share:.1%}): "
                        "line_amount may be the unit price or the pre-discount amount, "
                        "or order_subtotal in orders may be the after-discount / with-shipping "
                        "total (`order_payable`)."
                    ),
                )
            )

    # Orphan lines or orders without lines > 5% of orders
    n_orders = rep.rows_per_table.get("orders", 0)
    if n_orders > 0 and (
        integ.orders_without_lines / n_orders > 0.05
        or integ.orphan_line_orders / n_orders > 0.05
    ):
        step_join = "B_orders" if orders_tm.derived_from is None else "B_order_lines"
        issues.append(
            Issue(
                step=step_join,
                severity="error",
                message=(
                    f"Join key mismatch between orders and order_lines: "
                    f"orders_without_lines={integ.orders_without_lines}/{n_orders}, "
                    f"orphan_line_orders={integ.orphan_line_orders}/{n_orders} (> 5%)."
                ),
            )
        )

    # Order dates outside [2000-01-01, data_end_date + 1 day] for > 1% of orders
    if n_orders > 0 and not canon.orders.empty:
        min_allowed = datetime.date(2000, 1, 1)
        max_allowed = (
            mapping.data_end_date + datetime.timedelta(days=1)
            if mapping.data_end_date is not None
            else datetime.date(2030, 12, 31)
        )
        out_of_range = 0
        for d in canon.orders["order_date"].tolist():
            if isinstance(d, datetime.date) and (d < min_allowed or d > max_allowed):
                out_of_range += 1
        if out_of_range / n_orders > 0.01:
            issues.append(
                Issue(
                    step="B_orders",
                    severity="error",
                    message=(
                        f"{out_of_range}/{n_orders} orders ({out_of_range / n_orders:.1%}) have "
                        f"order_date outside [{min_allowed}..{max_allowed}] "
                        f"(min={integ.min_order_date}, max={integ.max_order_date})."
                    ),
                )
            )

    # Line math warning: qty * unit_price - (line_discount or 0) == line_amount
    if all(k in lines_tm.columns for k in ("qty", "unit_price", "line_amount")):
        math_share = _check_line_math_share(canon)
        if math_share is not None and math_share < 0.8:
            issues.append(
                Issue(
                    step="B_order_lines",
                    severity="warning",
                    drop_field="order_lines.unit_price",
                    message=(
                        f"qty * unit_price - line_discount matches line_amount in only "
                        f"{math_share:.1%} of lines (< 80%)."
                    ),
                )
            )

    return issues, canon

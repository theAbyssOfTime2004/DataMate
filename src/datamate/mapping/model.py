from __future__ import annotations

import datetime
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, ClassVar, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

Entity = Literal[
    "orders",
    "order_lines",
    "products",
    "unit_conversions",
    "customers",
    "payments",
    "debts",
    "purchases",
    "purchase_lines",
    "suppliers",
    "stock_snapshot",
    "stock_moves",
    "lots",
    "sale_lots",
    "price_history",
    "expenses",
    "employees",
    "deliveries",
    "drug_info",
]
OrderStatus = Literal["completed", "cancelled", "returned", "shipping"]
PurchaseStatus = Literal["valid", "cancelled"]
CustomerIdentity = Literal["by_code", "by_name"]
ReturnRule = Literal["status", "negative_qty", "return_table"]

CANONICAL_FIELDS: dict[Entity, set[str]] = {
    "orders": {
        "order_id",
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
    },
    "order_lines": {
        "order_id",
        "line_no",
        "product_id",
        "product_name",
        "qty",
        "unit",
        "qty_base",
        "unit_price",
        "line_discount",
        "line_amount",
        "cost_per_unit",
    },
    "products": {
        "product_id",
        "product_name",
        "base_unit",
        "category",
        "unit_cost",
    },
    "unit_conversions": {
        "product_id",
        "product_name",
        "unit",
        "base_unit",
        "factor",
    },
    "customers": {
        "customer_id",
        "customer_name",
        "segment",
        "phone",
    },
    "payments": {
        "payment_id",
        "order_id",
        "customer_id",
        "customer_name",
        "pay_date",
        "pay_ts",
        "amount",
        "method",
        "kind_raw",
        "note",
    },
    "debts": {
        "order_id",
        "customer_id",
        "customer_name",
        "order_date",
        "due_date",
        "amount_total",
        "amount_paid",
        "amount_remaining",
        "as_of_date",
    },
    "purchases": {
        "receipt_id",
        "receipt_date",
        "supplier_id",
        "supplier_name",
        "status_raw",
        "total_amount",
    },
    "purchase_lines": {
        "receipt_id",
        "line_no",
        "receipt_date",
        "supplier_name",
        "product_id",
        "product_name",
        "qty",
        "unit",
        "qty_base",
        "unit_cost",
        "line_amount",
        "lot_id",
        "expiry_date",
    },
    "suppliers": {
        "supplier_id",
        "supplier_name",
    },
    "stock_snapshot": {
        "product_id",
        "product_name",
        "qty_on_hand",
        "unit",
        "as_of_date",
        "unit_cost",
    },
    "stock_moves": {
        "product_id",
        "product_name",
        "move_date",
        "move_ts",
        "qty_in",
        "qty_out",
        "balance",
        "unit",
        "ref_id",
        "kind_raw",
    },
    "lots": {
        "lot_id",
        "product_id",
        "product_name",
        "receipt_id",
        "received_date",
        "expiry_date",
        "qty_received",
        "unit",
        "unit_cost",
    },
    "sale_lots": {
        "order_id",
        "line_no",
        "product_id",
        "product_name",
        "lot_id",
        "qty",
        "unit",
    },
    "price_history": {
        "product_id",
        "product_name",
        "effective_date",
        "price",
        "unit",
    },
    "expenses": {
        "expense_date",
        "amount",
        "category",
        "note",
    },
    "employees": {
        "employee_id",
        "employee_name",
        "alias",
        "start_date",
        "commission_pct",
    },
    "deliveries": {
        "order_id",
        "delivery_date",
        "batch_no",
        "qty",
        "amount",
        "status_raw",
    },
    "drug_info": {
        "product_id",
        "product_name",
        "active_ingredient",
        "is_prescription",
    },
}

_BOOL_TAG = "tag:yaml.org,2002:bool"
_STRICT_BOOL_RE = re.compile(r"^(?:true|false)$", re.IGNORECASE)


class _StrictBoolSafeLoader(yaml.SafeLoader):
    """YAML SafeLoader whose bool resolver only matches true/false (any case)."""

    bool_values: ClassVar[dict[str, bool]] = {
        "true": True,
        "false": False,
    }


_StrictBoolSafeLoader.yaml_implicit_resolvers = {
    ch: [(tag, regexp) for tag, regexp in resolvers if tag != _BOOL_TAG]
    for ch, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
for _ch in ("t", "T", "f", "F"):
    _StrictBoolSafeLoader.add_implicit_resolver(_BOOL_TAG, _STRICT_BOOL_RE, [_ch])


def _nfc_str(val: str) -> str:
    return unicodedata.normalize("NFC", val)


class SourceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file: str
    sheet: str | None = None
    header_row: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _normalize_names(self) -> SourceSpec:
        self.file = _nfc_str(self.file)
        if self.sheet is not None:
            self.sheet = _nfc_str(self.sheet)
        return self


class JoinSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table: Entity
    on: dict[str, str]
    field: str | None = None

    @model_validator(mode="after")
    def _validate_join(self) -> JoinSpec:
        if not self.on:
            raise ValueError("JoinSpec.on must not be empty")
        target_allowed = CANONICAL_FIELDS[self.table]
        for target_field in self.on.values():
            if target_field not in target_allowed:
                raise ValueError(
                    f"Unknown target field {target_field!r} on entity {self.table!r}; "
                    f"allowed: {sorted(target_allowed)}"
                )
        if self.field is not None and self.field not in target_allowed:
            raise ValueError(
                f"Unknown join source field {self.field!r} on entity {self.table!r}; "
                f"allowed: {sorted(target_allowed)}"
            )
        return self


class TableMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sources: list[SourceSpec] = Field(default_factory=list)
    derived_from: Literal["order_lines"] | None = None
    source: Literal["readme"] | None = None
    join: dict[str, JoinSpec] = Field(default_factory=dict)
    columns: dict[str, str] = Field(default_factory=dict)
    row_filter: str | None = None

    @model_validator(mode="after")
    def _validate_origin_and_nfc(self) -> TableMapping:
        origins_set = (
            (1 if len(self.sources) > 0 else 0)
            + (1 if self.derived_from is not None else 0)
            + (1 if self.source is not None else 0)
        )
        if origins_set != 1:
            raise ValueError(
                "Exactly one of 'sources' (non-empty), 'derived_from', or 'source' must be set"
            )
        self.columns = {_nfc_str(k): _nfc_str(v) for k, v in self.columns.items()}
        return self

    def validate_for_entity(self, entity: Entity) -> None:
        allowed = CANONICAL_FIELDS[entity]
        for col_key in self.columns:
            if col_key not in allowed:
                raise ValueError(
                    f"Unknown canonical field {col_key!r} for entity {entity!r}; "
                    f"allowed: {sorted(allowed)}"
                )
        for join_key, jspec in self.join.items():
            if join_key not in allowed:
                raise ValueError(
                    f"Unknown canonical join field {join_key!r} for entity {entity!r}; "
                    f"allowed: {sorted(allowed)}"
                )
            for local_field in jspec.on:
                if local_field not in allowed:
                    raise ValueError(
                        f"Unknown local join key {local_field!r} for entity {entity!r}; "
                        f"allowed: {sorted(allowed)}"
                    )


class Mapping(BaseModel):
    model_config = ConfigDict(extra="ignore")

    business_id: str | None = None
    data_end_date: datetime.date | None = None
    customer_identity: CustomerIdentity
    walk_in_values: list[str] = Field(default_factory=list)
    return_rule: ReturnRule
    status_map: dict[str, OrderStatus]
    purchase_status_map: dict[str, PurchaseStatus] = Field(default_factory=dict)
    tables: dict[Entity, TableMapping]
    notes: list[str] = Field(default_factory=list)
    # "entity.field" or "entity" the shop records but whose mapping failed the checks.
    quarantined: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_tables_and_fields(self) -> Mapping:
        if "order_lines" not in self.tables:
            raise ValueError("Mapping.tables must contain 'order_lines'")
        if "orders" not in self.tables:
            raise ValueError("Mapping.tables must contain 'orders'")
        for entity, tm in self.tables.items():
            tm.validate_for_entity(entity)
        self.walk_in_values = [_nfc_str(v) for v in self.walk_in_values]
        self.status_map = {_nfc_str(k): v for k, v in self.status_map.items()}
        self.purchase_status_map = {
            _nfc_str(k): v for k, v in self.purchase_status_map.items()
        }
        return self


def load_mapping_yaml(path: Path | str) -> Mapping:
    p = Path(path)
    raw: dict[str, Any] = yaml.load(
        p.read_text(encoding="utf-8"),
        Loader=_StrictBoolSafeLoader,
    )
    return Mapping.model_validate(raw)


def mapping_fingerprint(mapping: Mapping) -> str:
    payload = mapping.model_dump(mode="json", exclude={"business_id", "notes"})
    # Empty purchase_status_map must not change keys of pre-E1b order-only mappings.
    if not payload.get("purchase_status_map"):
        payload.pop("purchase_status_map", None)
    if not payload.get("quarantined"):
        payload.pop("quarantined", None)
    canonical_json = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

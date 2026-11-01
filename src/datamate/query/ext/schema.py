"""JSON schema for the extension parser (one family enum + shared parameters)."""

from __future__ import annotations

from typing import Any

from datamate.query.context import ShopContext
from datamate.query.ext.spec import E4A_FAMILIES, E4B_FAMILIES, E4C_FAMILIES, E4D_FAMILIES
from datamate.query.parse_schema import PERIOD_FIELDS, PERIOD_KINDS

EXT_TOP_LEVEL_KEYS: tuple[str, ...] = (
    "family",
    "product",
    "base_unit",
    "category",
    "size",
    "period",
    "k",
    "by",
    "mode",
    "output",
    "n",
    "n_days",
    "date",
    "x",
    "op",
    "buckets",
    "unit",
    "active_ingredient",
)

_BY_VALUES: tuple[str, ...] = ("qty_base", "revenue", "gross_profit", "order_count")
_MODE_VALUES: tuple[str, ...] = ("max", "min")
_OUTPUT_VALUES: tuple[str, ...] = ("count", "list", "share", "cost")
_OP_VALUES: tuple[str, ...] = (">=", ">")
_SIZE_VALUES: tuple[str, ...] = ("M", "L")


def _nullable_enum(values: list[str]) -> dict[str, Any]:
    return {"type": ["string", "null"], "enum": [*values, None]}


def _period_object() -> dict[str, Any]:
    return {
        "type": ["object", "null"],
        "properties": {
            "kind": {"type": ["string", "null"], "enum": [*PERIOD_KINDS, None]},
            **{name: {} for name in PERIOD_FIELDS if name != "kind"},
        },
        "required": list(PERIOD_FIELDS),
        "additionalProperties": False,
    }


def _buckets_schema() -> dict[str, Any]:
    """Aging buckets as [[lo, hi], ...] with hi null for open-ended."""
    return {
        "type": ["array", "null"],
        "items": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "items": {"type": ["integer", "null"]},
        },
    }


def _sizes_in_shop(product_names: list[str]) -> list[str]:
    """Cafe size codes present as name suffixes like '(M)' / '(L)'."""
    found: list[str] = []
    for code in _SIZE_VALUES:
        suffix = f"({code})".casefold()
        if any(name.casefold().rstrip().endswith(suffix) for name in product_names):
            found.append(code)
    return found


def ext_parsed_schema(ctx: ShopContext, products: list[str]) -> dict[str, Any]:
    """Enums for product / base_unit / category / size come from the shop."""
    family_enum = [*E4A_FAMILIES, *E4B_FAMILIES, *E4C_FAMILIES, *E4D_FAMILIES, "none"]
    sizes = _sizes_in_shop(ctx.product_names)
    size_schema: dict[str, Any]
    if sizes:
        size_schema = _nullable_enum(sizes)
    else:
        size_schema = {"enum": [None]}
    return {
        "type": "object",
        "properties": {
            "family": {"type": "string", "enum": family_enum},
            "product": _nullable_enum(products),
            "base_unit": _nullable_enum(list(ctx.base_units)),
            "category": _nullable_enum(list(ctx.categories)),
            "size": size_schema,
            "period": _period_object(),
            "k": {"type": ["integer", "null"]},
            "by": {"type": ["string", "null"], "enum": [*_BY_VALUES, None]},
            "mode": {"type": ["string", "null"], "enum": [*_MODE_VALUES, None]},
            "output": {"type": ["string", "null"], "enum": [*_OUTPUT_VALUES, None]},
            "n": {"type": ["integer", "null"]},
            "n_days": {"type": ["integer", "null"]},
            "date": _period_object(),
            "x": {"type": ["integer", "null"]},
            "op": {"type": ["string", "null"], "enum": [*_OP_VALUES, None]},
            "buckets": _buckets_schema(),
            "unit": {"type": ["string", "null"]},
            "active_ingredient": {"type": ["string", "null"]},
        },
        "required": list(EXT_TOP_LEVEL_KEYS),
        "additionalProperties": False,
    }

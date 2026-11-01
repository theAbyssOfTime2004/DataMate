"""Composed query plan: fixed blocks the LLM picks; code compiles to SQL."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from datamate.timeres.periods import PeriodExpr

PlanEntity = Literal[
    "order",
    "customer",
    "product",
    "category",
    "employee",
    "supplier",
    "day",
    "weekday",
    "hour",
    "month",
    "payment_method",
    "channel",
    "none",
]

PlanMeasure = Literal[
    "revenue",
    "order_count",
    "distinct_order_days",
    "avg_order_value",
    "cancelled_order_count",
    "returned_order_count",
    "qty_base",
    "orders_containing",
    "distinct_customers",
    "receipt_count",
    "purchase_amount",
    "gross_profit",
]

PlanScope = Literal["revenue_orders", "all_orders", "cancelled_orders"]

HavingOp = Literal[">=", ">", "<=", "<", "="]

HavingMeasure = Literal[
    "revenue",
    "order_count",
    "distinct_order_days",
    "avg_order_value",
    "qty_base",
    "gross_profit",
    "receipt_count",
    "purchase_amount",
]


class PlanFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    period: PeriodExpr
    product: str | None = None
    category: str | None = None
    customer: str | None = None
    employee: str | None = None
    channel: str | None = None
    payment_method: str | None = None
    weekday: int | None = Field(default=None, ge=0, le=6)  # Mon=0 .. Sun=6
    hour_from: int | None = Field(default=None, ge=0, le=23)
    hour_to: int | None = Field(default=None, ge=0, le=23)

    def has_non_period_filter(self) -> bool:
        return any(value is not None for name, value in self if name != "period")


class HavingClause(BaseModel):
    model_config = ConfigDict(extra="forbid")

    measure: HavingMeasure
    op: HavingOp
    value: float | int


class ArgMaxOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["argmax"] = "argmax"


class ArgMinOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["argmin"] = "argmin"


class TopKOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["top_k"] = "top_k"
    k: int = Field(ge=1, le=50)


class CountEntitiesOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["count_entities"] = "count_entities"


class ValueOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["value"] = "value"


class ShareOfTotalOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["share_of_total"] = "share_of_total"


class AvgPerPeriodOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["avg_per_period"] = "avg_per_period"
    unit: Literal["day", "weekday_occurrence"] = "day"
    include_zero_days: bool = True


PlanOutput = Annotated[
    ArgMaxOut
    | ArgMinOut
    | TopKOut
    | CountEntitiesOut
    | ValueOut
    | ShareOfTotalOut
    | AvgPerPeriodOut,
    Field(discriminator="kind"),
]

_PURCHASE_MEASURES = frozenset({"receipt_count", "purchase_amount"})
_PURCHASE_ENTITIES = frozenset({"supplier", "day", "month", "none"})
_LINE_MEASURES = frozenset({"qty_base", "gross_profit", "orders_containing", "revenue"})
_RANK_OUTPUTS = frozenset({"argmax", "argmin", "top_k"})
_ALWAYS_POSITIVE = frozenset({"order_count", "receipt_count", "distinct_order_days"})
_ORDER_COUNT_MEASURES = frozenset(
    {"order_count", "cancelled_order_count", "returned_order_count", "orders_containing"}
)
_COUNT_MEASURES = _ORDER_COUNT_MEASURES | frozenset(
    {"receipt_count", "distinct_order_days", "distinct_customers"}
)


def default_scope_for(measure: PlanMeasure) -> PlanScope:
    if measure == "cancelled_order_count":
        return "cancelled_orders"
    if measure == "returned_order_count":
        return "all_orders"
    if measure in _PURCHASE_MEASURES:
        return "all_orders"  # unused for purchases; kept for schema completeness
    return "revenue_orders"


class QueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity: PlanEntity
    measure: PlanMeasure
    scope: PlanScope | None = None
    filters: PlanFilters
    having: list[HavingClause] = Field(default_factory=list)
    output: PlanOutput

    @model_validator(mode="after")
    def _validate_plan(self) -> QueryPlan:
        self._canonicalise()
        if self.measure == "cancelled_order_count":
            object.__setattr__(self, "scope", "cancelled_orders")
        elif self.measure == "returned_order_count":
            # Revenue orders exclude returns, so that scope would always count zero.
            object.__setattr__(self, "scope", "all_orders")
        elif self.scope is None:
            object.__setattr__(self, "scope", default_scope_for(self.measure))

        if self.measure in _PURCHASE_MEASURES and self.entity not in _PURCHASE_ENTITIES:
            raise ValueError(
                f"{self.measure} only allowed with entity in "
                f"{sorted(_PURCHASE_ENTITIES)}; got {self.entity!r}"
            )

        if self.entity == "none" and self.output.kind in _RANK_OUTPUTS:
            raise ValueError("entity=none cannot use argmax/argmin/top_k")

        if self.entity == "hour" and self.measure in _PURCHASE_MEASURES:
            raise ValueError("hour entity is not valid with purchase measures")

        if self.measure == "orders_containing" and self.filters.product is None:
            raise ValueError("orders_containing requires filters.product")

        if self.entity == "none" and self.having:
            raise ValueError("having needs an entity to filter")

        if (
            self.output.kind == "share_of_total"
            and self.entity == "none"
            and not self.filters.has_non_period_filter()
        ):
            raise ValueError("shop-level share_of_total needs a filter (the part of the total)")

        if self.output.kind == "avg_per_period" and self.entity != "none":
            raise ValueError("avg_per_period requires entity=none (shop total per day unit)")

        if (
            self.filters.hour_from is not None
            and self.filters.hour_to is not None
            and self.filters.hour_from > self.filters.hour_to
        ):
            raise ValueError("hour_from must be <= hour_to")

        return self

    def _canonicalise(self) -> None:
        """One form per meaning, so equivalent votes agree (generic rewrites only)."""
        if self.measure == "order_count" and self.scope == "cancelled_orders":
            object.__setattr__(self, "measure", "cancelled_order_count")
        # Every grouped entity has at least one order / receipt / day: such a floor filters
        # nothing.
        kept = [
            h
            for h in self.having
            if not (
                h.measure in _ALWAYS_POSITIVE
                and ((h.op == ">=" and h.value <= 1) or (h.op == ">" and h.value < 1))
            )
        ]
        object.__setattr__(self, "having", kept)
        # Counting orders as entities is the order count itself.
        if (
            self.entity == "order"
            and self.output.kind == "count_entities"
            and not self.having
            and self.measure in _ORDER_COUNT_MEASURES
        ):
            object.__setattr__(self, "entity", "none")
        if self.entity == "none" and self.output.kind == "count_entities":
            if self.measure not in _COUNT_MEASURES:
                raise ValueError("count_entities with entity=none needs a count measure")
            object.__setattr__(self, "output", ValueOut())

    def model_dump_resolved(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

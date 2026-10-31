"""Extension query specs for product, customer, and threshold families."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from datamate.timeres.periods import PeriodExpr

ExtFamily = Literal[
    "qty_sold",
    "top_products",
    "unsold_products",
    "products_in_min_orders",
    "price_on_date",
    "price_increase_effect",
    "customers_min_orders",
    "top_customers",
    "orders_min_amount",
    "days_min_revenue",
    "none",
]

TopBy = Literal["qty_base", "revenue", "gross_profit"]
CustomerTopBy = Literal["revenue", "order_count"]
TopMode = Literal["max", "min"]
CountOrList = Literal["count", "list"]
CountOrShare = Literal["count", "share"]
ThresholdOp = Literal[">=", ">"]
CafeSize = Literal["M", "L"]


class QtySoldQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: Literal["qty_sold"] = "qty_sold"
    product: str | None = None
    base_unit: str | None = None
    category: str | None = None
    size: CafeSize | None = None
    period: PeriodExpr

    @model_validator(mode="after")
    def _one_target(self) -> QtySoldQuery:
        has_product = self.product is not None and self.product != ""
        has_unit = self.base_unit is not None and self.base_unit != ""
        if has_product == has_unit:
            raise ValueError("qty_sold needs exactly one of product or base_unit")
        return self


class TopProductsQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: Literal["top_products"] = "top_products"
    k: int = Field(ge=1)
    by: TopBy
    mode: TopMode = "max"
    category: str | None = None
    size: CafeSize | None = None
    period: PeriodExpr


class UnsoldProductsQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: Literal["unsold_products"] = "unsold_products"
    output: CountOrList
    category: str | None = None
    period: PeriodExpr


class ProductsInMinOrdersQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: Literal["products_in_min_orders"] = "products_in_min_orders"
    n: int = Field(ge=1)
    output: CountOrList
    category: str | None = None
    period: PeriodExpr


class PriceOnDateQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: Literal["price_on_date"] = "price_on_date"
    product: str
    date: PeriodExpr  # day / today / yesterday, resolved to a single day


class PriceIncreaseEffectQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: Literal["price_increase_effect"] = "price_increase_effect"
    product: str
    n_days: int = Field(ge=1)
    date: PeriodExpr | None = None  # optional anchor; else find first increase


class CustomersMinOrdersQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: Literal["customers_min_orders"] = "customers_min_orders"
    n: int = Field(ge=1)
    output: CountOrShare
    period: PeriodExpr


class TopCustomersQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: Literal["top_customers"] = "top_customers"
    k: int = Field(ge=1)
    by: CustomerTopBy
    mode: TopMode = "max"
    period: PeriodExpr


class OrdersMinAmountQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: Literal["orders_min_amount"] = "orders_min_amount"
    x: int = Field(ge=0)
    op: ThresholdOp
    period: PeriodExpr


class DaysMinRevenueQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: Literal["days_min_revenue"] = "days_min_revenue"
    x: int = Field(ge=0)
    op: ThresholdOp
    period: PeriodExpr


class NoneQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: Literal["none"] = "none"


ExtQuerySpec = Annotated[
    QtySoldQuery
    | TopProductsQuery
    | UnsoldProductsQuery
    | ProductsInMinOrdersQuery
    | PriceOnDateQuery
    | PriceIncreaseEffectQuery
    | CustomersMinOrdersQuery
    | TopCustomersQuery
    | OrdersMinAmountQuery
    | DaysMinRevenueQuery
    | NoneQuery,
    Field(discriminator="family"),
]

E4A_FAMILIES: tuple[str, ...] = (
    "qty_sold",
    "top_products",
    "unsold_products",
    "products_in_min_orders",
    "price_on_date",
    "price_increase_effect",
)

E4B_FAMILIES: tuple[str, ...] = (
    "customers_min_orders",
    "top_customers",
    "orders_min_amount",
    "days_min_revenue",
)

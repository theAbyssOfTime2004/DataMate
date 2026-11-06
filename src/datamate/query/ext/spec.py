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
    "debt_total",
    "top_debtors",
    "debt_aging",
    "debt_added",
    "debt_collected",
    "customer_debt_flow",
    "purchase_total",
    "top_supplier",
    "avg_purchase_cost",
    "stock_on_hand",
    "stock_value",
    "out_of_stock_count",
    "lots_expiring",
    "nearest_expiry_lot",
    "ingredient_revenue",
    "rx_share",
    "rx_revenue",
    "otc_revenue",
    "gross_profit",
    "net_profit",
    "top_employee",
    "employee_revenue",
    "delivery_count",
    "orders_min_deliveries",
    "none",
]

TopBy = Literal["qty_base", "revenue", "gross_profit"]
CustomerTopBy = Literal["revenue", "order_count"]
TopMode = Literal["max", "min"]
CountOrList = Literal["count", "list"]
CountOrShare = Literal["count", "share"]
CountListOrCost = Literal["count", "list", "cost"]
ThresholdOp = Literal[">=", ">"]
# A size label from the shop's product names (datamate.query.sizes).
CafeSize = str


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
        if has_product and has_unit:
            raise ValueError("qty_sold needs exactly one of product or base_unit")
        if not has_product and not has_unit and not self.size:
            raise ValueError("qty_sold needs a product, a base_unit or a size")
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


class AgingBucket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lo: int = Field(ge=0)
    hi: int | None = None  # None = open-ended (no upper bound)


class DebtTotalQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: Literal["debt_total"] = "debt_total"
    date: PeriodExpr  # single day: end-of-day as-of
    # amount = total remaining VND; customer_count = customers with remaining > 0.
    output: Literal["amount", "customer_count"] = "amount"


class TopDebtorsQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: Literal["top_debtors"] = "top_debtors"
    k: int = Field(ge=1)
    date: PeriodExpr


class DebtAgingQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: Literal["debt_aging"] = "debt_aging"
    date: PeriodExpr
    buckets: list[AgingBucket] = Field(min_length=1)


class DebtAddedQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: Literal["debt_added"] = "debt_added"
    period: PeriodExpr


class DebtCollectedQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: Literal["debt_collected"] = "debt_collected"
    period: PeriodExpr


class CustomerDebtFlowQuery(BaseModel):
    """One customer's new debt and later repayments in a period, as [added, paid]."""

    model_config = ConfigDict(extra="forbid")

    family: Literal["customer_debt_flow"] = "customer_debt_flow"
    customer: str  # customer name or code as written in the question
    period: PeriodExpr


class PurchaseTotalQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: Literal["purchase_total"] = "purchase_total"
    period: PeriodExpr


class TopSupplierQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: Literal["top_supplier"] = "top_supplier"
    period: PeriodExpr


class AvgPurchaseCostQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: Literal["avg_purchase_cost"] = "avg_purchase_cost"
    product: str
    period: PeriodExpr  # month


class StockOnHandQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: Literal["stock_on_hand"] = "stock_on_hand"
    product: str
    date: PeriodExpr
    unit: str | None = None  # None or base -> number; big unit -> [whole, leftover]


class StockValueQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: Literal["stock_value"] = "stock_value"
    date: PeriodExpr


class OutOfStockCountQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: Literal["out_of_stock_count"] = "out_of_stock_count"
    date: PeriodExpr


class LotsExpiringQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: Literal["lots_expiring"] = "lots_expiring"
    date: PeriodExpr
    n_days: int = Field(ge=0)
    output: CountListOrCost


class NearestExpiryLotQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: Literal["nearest_expiry_lot"] = "nearest_expiry_lot"
    product: str
    date: PeriodExpr | None = None  # default: as_of of the question


class IngredientRevenueQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: Literal["ingredient_revenue"] = "ingredient_revenue"
    active_ingredient: str
    period: PeriodExpr


class RxShareQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: Literal["rx_share"] = "rx_share"
    period: PeriodExpr


class RxRevenueQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: Literal["rx_revenue"] = "rx_revenue"
    period: PeriodExpr


class OtcRevenueQuery(BaseModel):
    """Revenue of everything that is not a prescription drug."""

    model_config = ConfigDict(extra="forbid")

    family: Literal["otc_revenue"] = "otc_revenue"
    period: PeriodExpr


class GrossProfitQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: Literal["gross_profit"] = "gross_profit"
    period: PeriodExpr


class NetProfitQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: Literal["net_profit"] = "net_profit"
    period: PeriodExpr  # month


class TopEmployeeQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: Literal["top_employee"] = "top_employee"
    period: PeriodExpr


class EmployeeRevenueQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: Literal["employee_revenue"] = "employee_revenue"
    employee: str  # employee code or name as written in the question
    period: PeriodExpr


class DeliveryCountQuery(BaseModel):
    """Delivery batches (one per order and batch number) whose delivery date is in the period."""

    model_config = ConfigDict(extra="forbid")

    family: Literal["delivery_count"] = "delivery_count"
    period: PeriodExpr


class OrdersMinDeliveriesQuery(BaseModel):
    """Orders placed in the period that had at least n delivery batches by the as-of day."""

    model_config = ConfigDict(extra="forbid")

    family: Literal["orders_min_deliveries"] = "orders_min_deliveries"
    n: int = Field(ge=1)
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
    | DebtTotalQuery
    | TopDebtorsQuery
    | DebtAgingQuery
    | DebtAddedQuery
    | DebtCollectedQuery
    | CustomerDebtFlowQuery
    | PurchaseTotalQuery
    | TopSupplierQuery
    | AvgPurchaseCostQuery
    | StockOnHandQuery
    | StockValueQuery
    | OutOfStockCountQuery
    | LotsExpiringQuery
    | NearestExpiryLotQuery
    | IngredientRevenueQuery
    | RxShareQuery
    | RxRevenueQuery
    | OtcRevenueQuery
    | GrossProfitQuery
    | NetProfitQuery
    | TopEmployeeQuery
    | EmployeeRevenueQuery
    | DeliveryCountQuery
    | OrdersMinDeliveriesQuery
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

E4C_FAMILIES: tuple[str, ...] = (
    "debt_total",
    "top_debtors",
    "debt_aging",
    "debt_added",
    "debt_collected",
    "customer_debt_flow",
)

E4D_FAMILIES: tuple[str, ...] = (
    "purchase_total",
    "top_supplier",
    "avg_purchase_cost",
    "stock_on_hand",
    "stock_value",
    "out_of_stock_count",
    "lots_expiring",
    "nearest_expiry_lot",
    "ingredient_revenue",
    "rx_share",
    "rx_revenue",
    "otc_revenue",
    "gross_profit",
    "net_profit",
    "top_employee",
    "employee_revenue",
)

# Event logs kept per order (delivery batches); see metrics/events.py.
EVENT_FAMILIES: tuple[str, ...] = (
    "delivery_count",
    "orders_min_deliveries",
)

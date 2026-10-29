from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from datamate.timeres.periods import PeriodExpr

Metric = Literal[
    "revenue",
    "order_count",
    "cancel_count",
    "cancel_rate",
    "avg_order_value",
    "revenue_share",
]
ShiftName = Literal["morning", "afternoon"]


class Filters(BaseModel):
    """Filter values are raw data values. Product names are matched after normalisation."""

    model_config = ConfigDict(extra="forbid")

    payment_method: list[str] = Field(default_factory=list)
    channel: list[str] = Field(default_factory=list)
    customer_segment: list[str] = Field(default_factory=list)
    product: list[str] = Field(default_factory=list)
    category: list[str] = Field(default_factory=list)
    shift: ShiftName | None = None


class MetricQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: Metric
    period: PeriodExpr
    filters: Filters = Field(default_factory=Filters)


class ValueQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["value"] = "value"
    query: MetricQuery
    answer_type: Literal["number", "percent"] | None = None


class CompareQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["compare"] = "compare"
    a: MetricQuery
    b: MetricQuery
    op: Literal["pct_change", "diff"]


class SeriesQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["series"] = "series"
    metric: Metric
    filters: Filters = Field(default_factory=Filters)
    periods: list[PeriodExpr]


class MultiQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["multi"] = "multi"
    queries: list[MetricQuery]


class TrendQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["trend"] = "trend"
    metric: Metric
    filters: Filters = Field(default_factory=Filters)
    periods: list[PeriodExpr]
    direction: Literal["increasing", "decreasing"]
    strict: bool = True


class ArgMaxQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["argmax"] = "argmax"
    metric: Metric
    filters: Filters = Field(default_factory=Filters)
    period: PeriodExpr
    granularity: Literal["day", "month"]
    mode: Literal["max", "min"] = "max"


QuerySpec = Annotated[
    ValueQuery | CompareQuery | SeriesQuery | MultiQuery | TrendQuery | ArgMaxQuery,
    Field(discriminator="kind"),
]

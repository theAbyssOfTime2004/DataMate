"""E3 plan parse (fake LLM) + describe."""

from __future__ import annotations

import datetime
import math
from typing import Any

import pytest
from pydantic import ValidationError

from datamate.llm.client import UsageEntry
from datamate.query.context import ShopContext
from datamate.query.plan.describe import describe_plan
from datamate.query.plan.parse import parse_plan_question, select_rule_notes
from datamate.query.plan.spec import (
    ArgMaxOut,
    CountEntitiesOut,
    HavingClause,
    PlanFilters,
    QueryPlan,
    ShareOfTotalOut,
    ValueOut,
)
from datamate.rules.model import RuleConfig
from datamate.rules.parse_rulebook import RuleRow
from datamate.timeres import LunarCalendar, MonthExpr


class FakeClient:
    def __init__(self, scripted: dict[str, list[dict[str, Any]]]) -> None:
        self.scripted = {k: list(v) for k, v in scripted.items()}
        self.usage_log: list[UsageEntry] = []

    def count_tokens(self, text: str) -> int:
        return math.ceil(len(text) / 2.5)

    def chat_json(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        schema_name: str,
        max_tokens: int | None = None,
        required_keys: list[str] | None = None,
        enable_thinking: bool | None = None,
    ) -> dict[str, Any]:
        del messages, schema, max_tokens, required_keys, enable_thinking
        queue = self.scripted.get(schema_name, [])
        if not queue:
            raise AssertionError(f"no script for {schema_name}")
        self.usage_log.append(
            UsageEntry(cache_hit=False, prompt_tokens=1, completion_tokens=1, latency_s=0.0)
        )
        return queue.pop(0)


def test_parse_plan_two_votes_agree() -> None:
    ctx = ShopContext(
        payment_methods=["cash"],
        channels=["store"],
        segments=[],
        categories=["đồ uống"],
        product_names=["Trà"],
        has_order_ts=True,
        data_start=datetime.date(2026, 1, 1),
        data_end=datetime.date(2026, 6, 30),
    )
    plan_json = {
        "entity": "none",
        "measure": "revenue",
        "scope": "revenue_orders",
        "filters": {
            "period": {
                "kind": "month",
                "date": None,
                "start": None,
                "end": None,
                "n": None,
                "month": 6,
                "q": None,
                "year": 2026,
                "year_rel": None,
                "n_days": None,
            },
            "product": None,
            "category": None,
            "customer": None,
            "employee": None,
            "channel": None,
            "payment_method": None,
            "weekday": None,
            "hour_from": None,
            "hour_to": None,
        },
        "having": [],
        "output": {"kind": "value", "k": None, "unit": None, "include_zero_days": None},
    }
    note = RuleRow(
        id="abcd1234abcd1234",
        section_path="1. Sales",
        name="Loyalty",
        text="Invented loyalty note for tests.",
    )
    client = FakeClient(
        {
            "PlanSelectNotes": [{"note_ids": ["n1"]}],
            "QueryPlan": [plan_json, plan_json],
        }
    )
    result = parse_plan_question(
        "Doanh thu tháng 6?",
        ctx,
        client,
        RuleConfig(),
        LunarCalendar({}),
        datetime.date(2026, 6, 30),
        [note],
        votes=2,
    )
    assert result.plan is not None
    assert result.plan.entity == "none"
    assert result.plan.measure == "revenue"
    assert note.id in result.selected_note_ids


class RecordingClient(FakeClient):
    def __init__(self, scripted: dict[str, list[dict[str, Any]]]) -> None:
        super().__init__(scripted)
        self.prompts: dict[str, list[str]] = {}

    def chat_json(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        schema_name: str,
        max_tokens: int | None = None,
        required_keys: list[str] | None = None,
        enable_thinking: bool | None = None,
    ) -> dict[str, Any]:
        self.prompts.setdefault(schema_name, []).append(messages[-1]["content"])
        return super().chat_json(
            messages, schema, schema_name, max_tokens, required_keys, enable_thinking
        )


def _ctx() -> ShopContext:
    return ShopContext(
        payment_methods=["cash"],
        channels=["store"],
        segments=[],
        categories=["đồ uống"],
        product_names=["Trà"],
        has_order_ts=True,
        data_start=datetime.date(2026, 1, 1),
        data_end=datetime.date(2026, 6, 30),
    )


def _value_plan_json() -> dict[str, Any]:
    period = dict.fromkeys(("date", "start", "end", "n", "q", "year_rel", "n_days"), None) | {
        "kind": "month",
        "month": 6,
        "year": 2026,
    }
    filters = dict.fromkeys(
        (
            "product",
            "category",
            "customer",
            "employee",
            "channel",
            "payment_method",
            "weekday",
            "hour_from",
            "hour_to",
        ),
        None,
    ) | {"period": period}
    return {
        "entity": "none",
        "measure": "revenue",
        "scope": "revenue_orders",
        "filters": filters,
        "having": [],
        "output": {"kind": "value", "k": None, "unit": None, "include_zero_days": None},
    }


def test_votes_are_distinct_requests() -> None:
    # Identical prompts would hit one cache entry and agree by construction.
    client = RecordingClient({"QueryPlan": [_value_plan_json(), _value_plan_json()]})
    parse_plan_question(
        "Doanh thu tháng 6?",
        _ctx(),
        client,
        RuleConfig(),
        LunarCalendar({}),
        datetime.date(2026, 6, 30),
        [],
        votes=2,
    )
    first, second = client.prompts["QueryPlan"]
    assert first != second


def test_month_with_redundant_fields_parses() -> None:
    # Seen on dev: the model fills date/start/end/n for a plain month.
    noisy = _value_plan_json()
    noisy["filters"]["period"] |= {
        "date": "2026-06-01",
        "start": "2026-06-01",
        "end": "2026-06-30",
        "n": 6,
    }
    client = FakeClient({"QueryPlan": [noisy, noisy]})
    result = parse_plan_question(
        "Doanh thu tháng 6?",
        _ctx(),
        client,
        RuleConfig(),
        LunarCalendar({}),
        datetime.date(2026, 6, 30),
        [],
        votes=2,
    )
    assert result.plan is not None, result.attempts


def test_one_failed_vote_is_not_agreement() -> None:
    broken = _value_plan_json() | {"entity": "nonsense"}
    client = FakeClient({"QueryPlan": [_value_plan_json(), broken]})
    result = parse_plan_question(
        "Doanh thu tháng 6?",
        _ctx(),
        client,
        RuleConfig(),
        LunarCalendar({}),
        datetime.date(2026, 6, 30),
        [],
        votes=2,
    )
    assert result.plan is None
    assert result.reason == "plan_parse_error"


def test_note_catalogue_lists_names_not_texts() -> None:
    notes = [
        RuleRow(id=f"{i:016x}", section_path="Defs", name=f"Term {i}", text="x" * 400)
        for i in range(120)
    ]
    client = RecordingClient({"PlanSelectNotes": [{"note_ids": ["n3"]}]})
    selected, error = select_rule_notes("Who is a term 3 customer?", notes, client)
    assert error is None
    assert selected == [notes[2]]
    prompt = client.prompts["PlanSelectNotes"][0]
    assert "x" * 50 not in prompt
    assert len(prompt) < 6000


def test_describe_plan_is_plain_and_closed_world() -> None:
    plan = QueryPlan(
        entity="customer",
        measure="avg_order_value",
        filters=PlanFilters(period=MonthExpr(month=6, year=2026)),
        having=[HavingClause(measure="order_count", op=">=", value=3)],
        output=ArgMaxOut(),
    )
    text = describe_plan(plan, datetime.date(2026, 6, 30), RuleConfig(), LunarCalendar({}))
    assert "average money revenue per order" in text
    assert "only each customer with: number of orders at least 3" in text
    assert "the one customer with the highest" in text
    assert "2026-06-01 to 2026-06-30" in text
    # No rulebook text: a listed definition reads as applied even when no block expresses it.
    assert "shop rule" not in text
    assert "any other condition in the question is not applied" in text


def _bare(**kwargs: Any) -> QueryPlan:
    base: dict[str, Any] = {
        "entity": "none",
        "measure": "order_count",
        "filters": PlanFilters(period=MonthExpr(month=6, year=2026)),
        "output": ValueOut(),
    }
    return QueryPlan.model_validate(base | kwargs)


def test_equivalent_order_counts_share_one_form() -> None:
    # Seen on dev as vote disagreements; all mean "number of orders in the period".
    plain = _bare()
    as_entities = _bare(
        entity="order",
        output=CountEntitiesOut(),
        having=[HavingClause(measure="order_count", op=">=", value=1)],
    )
    shop_count = _bare(output=CountEntitiesOut())
    dumps = {p.model_dump_json() for p in (plain, as_entities, shop_count)}
    assert len(dumps) == 1


def test_cancelled_scope_is_cancelled_count() -> None:
    plan = _bare(scope="cancelled_orders")
    assert plan.measure == "cancelled_order_count"


def test_real_having_floor_is_kept() -> None:
    plan = _bare(
        entity="customer",
        output=CountEntitiesOut(),
        having=[HavingClause(measure="order_count", op=">=", value=2)],
    )
    assert len(plan.having) == 1


def test_shop_share_needs_a_filter() -> None:
    with pytest.raises(ValidationError):
        _bare(measure="revenue", output=ShareOfTotalOut())

"""Vote tie-break on date ranges, contiguous argmax periods, two-digit years."""

from __future__ import annotations

import datetime
from typing import Any

import pytest

from datamate.config import settings
from datamate.query.adjudicate import differ_only_in_ranges, pick_period_reading
from datamate.query.convert import ConvertError, to_query_spec
from datamate.query.parse import parse_question
from datamate.query.spec import ArgMaxQuery, ValueQuery
from datamate.rules import RuleConfig
from datamate.timeres import LunarCalendar, RangeExpr
from tests.test_query_parse import _parsed, _shop

AS_OF = datetime.date(2026, 8, 19)


class ChoiceClient:
    """Scripted parse votes plus scripted A/B probabilities (forward, then swapped)."""

    def __init__(self, votes: list[dict[str, Any]], choices: list[dict[str, float]]) -> None:
        self.votes = list(votes)
        self.choices = list(choices)
        self.choice_prompts: list[str] = []

    def count_tokens(self, text: str) -> int:
        return 1

    def chat_json(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        schema_name: str,
        max_tokens: int | None = None,
        required_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        return self.votes.pop(0)

    def chat_choice_probs(
        self, messages: list[dict[str, str]], choices: list[str]
    ) -> dict[str, float]:
        self.choice_prompts.append(messages[-1]["content"])
        return self.choices.pop(0)


def _convert(parsed: dict[str, Any]) -> Any:
    return to_query_spec(parsed, AS_OF, RuleConfig(), LunarCalendar({}))


def _argmax(periods: list[dict[str, Any]]) -> dict[str, Any]:
    return _parsed(kind="argmax", periods=periods, granularity="month", mode="max")


def test_back_to_back_months_become_one_argmax_range() -> None:
    # Seen on dev: "in months 11/2025, 12/2025 and 1/2026, which month sold most?"
    spec = _convert(
        _argmax(
            [
                {"kind": "month", "month": 11, "year": 2025},
                {"kind": "month", "month": 12, "year": 2025},
                {"kind": "month", "month": 1, "year": 2026},
            ]
        )
    )
    assert isinstance(spec, ArgMaxQuery)
    assert spec.period == RangeExpr(
        start=datetime.date(2025, 11, 1), end=datetime.date(2026, 1, 31)
    )


def test_months_with_a_gap_stay_an_error() -> None:
    with pytest.raises(ConvertError):
        _convert(
            _argmax(
                [
                    {"kind": "month", "month": 3, "year": 2025},
                    {"kind": "month", "month": 5, "year": 2025},
                ]
            )
        )


def test_two_digit_year_is_rejected() -> None:
    # "3/25" read as year 25 would resolve outside the data and answer "unanswerable".
    with pytest.raises(ConvertError):
        _convert(_parsed(periods=[{"kind": "month", "month": 3, "year": 25}]))


def test_differ_only_in_ranges() -> None:
    a = '{"metric": "revenue", "period": {"range": ["2026-08-01", "2026-08-19"]}}'
    b = '{"metric": "revenue", "period": {"range": ["2026-08-01", "2026-08-31"]}}'
    c = '{"metric": "order_count", "period": {"range": ["2026-08-01", "2026-08-31"]}}'
    assert differ_only_in_ranges(a, b)
    assert not differ_only_in_ranges(a, c)
    assert not differ_only_in_ranges(a, a)


def test_pick_needs_confidence_in_both_orders() -> None:
    a = '{"period": {"range": ["2026-08-01", "2026-08-19"]}}'
    b = '{"period": {"range": ["2025-08-01", "2025-08-31"]}}'
    sure = ChoiceClient([], [{"A": 0.97, "B": 0.03}, {"A": 0.02, "B": 0.98}])
    assert pick_period_reading("thang nay", AS_OF, a, b, sure) == 0
    flips = ChoiceClient([], [{"A": 0.97, "B": 0.03}, {"A": 0.95, "B": 0.05}])
    assert pick_period_reading("thang nay", AS_OF, a, b, flips) is None
    unsure = ChoiceClient([], [{"A": 0.7, "B": 0.3}, {"A": 0.3, "B": 0.7}])
    assert pick_period_reading("thang nay", AS_OF, a, b, unsure) is None


def test_parse_takes_the_tie_broken_reading(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "period_adjudicate", True)
    this_month = _parsed(periods=[{"kind": "month_this"}])
    last_year = _parsed(periods=[{"kind": "month", "month": 8, "year": 2025}])
    client = ChoiceClient([this_month, last_year], [{"A": 0.99, "B": 0.01}, {"A": 0.01, "B": 0.99}])
    result = parse_question(
        "thang nay ban duoc bao nhieu", _shop(), client, RuleConfig(), LunarCalendar({}), AS_OF
    )
    assert isinstance(result.spec, ValueQuery)
    assert result.period_adjudicated
    assert "Today: 2026-08-19" in client.choice_prompts[0]


def test_parse_other_disagreements_still_abstain() -> None:
    revenue = _parsed()
    orders = _parsed(metric="order_count")
    client = ChoiceClient([revenue, orders], [])
    result = parse_question("thang nay", _shop(), client, RuleConfig(), LunarCalendar({}), AS_OF)
    assert result.reason == "disagreement"
    assert client.choice_prompts == []

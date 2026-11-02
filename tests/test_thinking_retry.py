"""Thinking second pass: only fast-pass reading failures retry, parse calls only."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import pytest

from datamate.config import settings
from datamate.io.paths import business_dir
from datamate.io.questions import Question
from datamate.llm.thinking import ThinkingClient
from datamate.scoring.answer import Answer
from datamate.systems.pipeline import PipelineSystem
from tests.test_pipeline import FakeClient, _mapping_scripts, _parsed, synthetic_shop

__all__ = ["synthetic_shop"]  # pytest fixture re-export


class ThinkingAwareClient(FakeClient):
    """Fast parse answers and thinking parse answers come from separate queues."""

    def __init__(
        self,
        scripted: dict[str, list[dict[str, Any]]],
        thinking: list[dict[str, Any]],
    ) -> None:
        super().__init__(scripted)
        self.thinking = list(thinking)
        self.thinking_calls: list[dict[str, Any]] = []
        self.max_output_tokens = 1024

    def chat_json(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        schema_name: str,
        max_tokens: int | None = None,
        required_keys: list[str] | None = None,
        enable_thinking: bool | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        if enable_thinking:
            self.thinking_calls.append(
                {"schema_name": schema_name, "max_tokens": max_tokens, "timeout_s": timeout_s}
            )
            return self.thinking.pop(0)
        return super().chat_json(messages, schema, schema_name, max_tokens, required_keys)


def _question() -> Question:
    return Question(
        id="q_syn_think",
        business_id="biz_syn",
        question="doanh thu thang nay bao nhieu",
        as_of_date=datetime.date(2026, 8, 15),
    )


def test_thinking_client_turns_thinking_on_with_more_budget() -> None:
    base = ThinkingAwareClient({}, [{"ok": True}])
    ThinkingClient(base).chat_json([], {}, "X", max_tokens=400)
    call = base.thinking_calls[0]
    assert call["max_tokens"] == 400 + settings.thinking_max_tokens
    assert call["timeout_s"] == settings.thinking_timeout_s


def test_disagreement_retries_with_thinking(
    synthetic_shop: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "parse_thinking", "retry")
    revenue = _parsed(periods=[{"kind": "month_this"}])
    orders = _parsed(metric="order_count", periods=[{"kind": "month_this"}])
    client = ThinkingAwareClient(
        {**_mapping_scripts(), "ParsedQuestion": [revenue, orders]}, [revenue, revenue]
    )
    system = PipelineSystem(name="pipeline_test", client=client)
    try:
        answer, trace = system.answer(_question(), business_dir("train", "biz_syn"))
    finally:
        system.close()
    assert answer == Answer(type="number", answer=110_000)
    assert trace["parse_thinking"] is True
    assert trace["fast_pass"]["abstain_reason"] == "disagreement"
    # Mapping ran once on the fast client; only the two parse votes used thinking.
    assert [c["schema_name"] for c in client.thinking_calls] == ["ParsedQuestion"] * 2


def test_answered_questions_do_not_retry(
    synthetic_shop: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "parse_thinking", "retry")
    revenue = _parsed(periods=[{"kind": "month_this"}])
    client = ThinkingAwareClient({**_mapping_scripts(), "ParsedQuestion": [revenue, revenue]}, [])
    system = PipelineSystem(name="pipeline_test", client=client)
    try:
        answer, trace = system.answer(_question(), business_dir("train", "biz_syn"))
    finally:
        system.close()
    assert answer.type == "number"
    assert "parse_thinking" not in trace
    assert client.thinking_calls == []


def test_failed_thinking_keeps_the_fast_pass_abstain(
    synthetic_shop: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "parse_thinking", "retry")
    revenue = _parsed(periods=[{"kind": "month_this"}])
    orders = _parsed(metric="order_count", periods=[{"kind": "month_this"}])
    # Empty thinking queue: the thinking call raises (IndexError).
    client = ThinkingAwareClient({**_mapping_scripts(), "ParsedQuestion": [revenue, orders]}, [])
    system = PipelineSystem(name="pipeline_test", client=client)
    try:
        answer, trace = system.answer(_question(), business_dir("train", "biz_syn"))
    finally:
        system.close()
    assert answer.type == "abstain"
    assert trace["abstain_reason"] == "disagreement"
    assert "thinking_error" in trace

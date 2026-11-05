"""The few-shot cut is the same as dropping one example at a time, with fewer counts."""

from __future__ import annotations

import pytest

from datamate.config import settings
from datamate.query.parse import fit_examples


class _Counter:
    def __init__(self) -> None:
        self.calls = 0

    def count_tokens(self, text: str) -> int:
        self.calls += 1
        return len(text)


def _build(keep: int) -> list[dict[str, str]]:
    return [{"role": "user", "content": "x" * (100 + 50 * keep)}]


def _linear(total: int, budget: int) -> int:
    keep = total
    while keep > 0 and len("user\n" + "x" * (100 + 50 * keep)) > budget:
        keep -= 1
    return total - keep


@pytest.mark.parametrize("budget", [50, 160, 400, 1000, 1500, 5000])
def test_binary_search_matches_one_by_one(monkeypatch: pytest.MonkeyPatch, budget: int) -> None:
    monkeypatch.setattr(settings, "llm_prompt_budget_tokens", budget)
    client = _Counter()
    messages, dropped = fit_examples(28, _build, client)
    assert dropped == _linear(28, budget)
    assert messages == _build(28 - dropped)
    assert client.calls <= 7

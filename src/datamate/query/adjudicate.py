"""Tie-break two parse votes that agree on everything except their date ranges.

The two few-shot sets sometimes read the same time phrase differently ("tháng này" as the
month to date vs the whole month, a bare month in this or last year). Instead of
abstaining, the model is asked which of the two concrete date ranges the question means.
The answer counts only when it is confident and does not flip when A and B are swapped.
"""

from __future__ import annotations

import datetime
import json
from typing import Any

from datamate.config import settings
from datamate.llm.client import LLMError, LLMUnavailable

_CHOICES = ("A", "B")
_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_SYSTEM = (
    "You decide which date range a Vietnamese shop-owner question asks about. Today's date "
    "is given; relative words (today, this month, last week, ...) count from it. Reply with "
    "the letter of the range the question means."
)


def _collect_ranges(node: Any, out: list[tuple[str, str]]) -> None:
    if isinstance(node, dict):
        found = node.get("range")
        if isinstance(found, list) and len(found) == 2 and len(node) == 1:
            out.append((str(found[0]), str(found[1])))
            return
        for key in sorted(node):
            _collect_ranges(node[key], out)
    elif isinstance(node, list):
        for item in node:
            _collect_ranges(item, out)


def _mask_ranges(node: Any) -> Any:
    if isinstance(node, dict):
        if isinstance(node.get("range"), list) and len(node) == 1:
            return {"range": "*"}
        return {key: _mask_ranges(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_mask_ranges(item) for item in node]
    return node


def ranges_of(key: str) -> list[tuple[str, str]]:
    """Resolved (start, end) pairs inside a semantic key, in a stable order."""
    out: list[tuple[str, str]] = []
    _collect_ranges(json.loads(key), out)
    return out


def differ_only_in_ranges(key_a: str, key_b: str) -> bool:
    """Same spec once every resolved date range is masked, but different ranges."""
    if key_a == key_b:
        return False
    masked_a = json.dumps(_mask_ranges(json.loads(key_a)), sort_keys=True)
    masked_b = json.dumps(_mask_ranges(json.loads(key_b)), sort_keys=True)
    return masked_a == masked_b


def _describe(ranges: list[tuple[str, str]]) -> str:
    parts = []
    for start, end in ranges:
        parts.append(start if start == end else f"from {start} to {end}")
    return "; ".join(parts)


def _ask(
    question: str, as_of: datetime.date, first: str, second: str, client: Any
) -> dict[str, float]:
    user = (
        f"Today: {as_of.isoformat()} ({_WEEKDAYS[as_of.weekday()]})\n"
        f"Question: {question}\n"
        f"A: {first}\n"
        f"B: {second}\n"
        "Which date range does the question ask about?"
    )
    messages = [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}]
    return client.chat_choice_probs(messages, list(_CHOICES))


def pick_period_reading(
    question: str, as_of: datetime.date, key_a: str, key_b: str, client: Any
) -> int | None:
    """0 for key_a, 1 for key_b, None when the model is unsure or order-sensitive."""
    text_a, text_b = _describe(ranges_of(key_a)), _describe(ranges_of(key_b))
    if text_a == text_b or not hasattr(client, "chat_choice_probs"):
        return None
    threshold = settings.period_adjudicate_threshold
    try:
        forward = _ask(question, as_of, text_a, text_b, client)
        backward = _ask(question, as_of, text_b, text_a, client)
    except LLMUnavailable:
        raise
    except LLMError:
        return None
    if forward.get("A", 0.0) >= threshold and backward.get("B", 0.0) >= threshold:
        return 0
    if forward.get("B", 0.0) >= threshold and backward.get("A", 0.0) >= threshold:
        return 1
    return None

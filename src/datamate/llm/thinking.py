"""Qwen3 thinking for structured parse calls only.

Wraps the shared client: ``chat_json`` reasons before answering (the server needs
``--reasoning-parser qwen3`` so the JSON constraint starts after the reasoning), with
``settings.thinking_max_tokens`` extra output budget. Everything else, including the
first-token ``chat_choice_probs`` the verifier uses, goes to the wrapped client unchanged.
"""

from __future__ import annotations

from typing import Any

from datamate.config import settings


class ThinkingClient:
    def __init__(self, base: Any) -> None:
        self._base = base

    def chat_json(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        schema_name: str,
        max_tokens: int | None = None,
        required_keys: list[str] | None = None,
        enable_thinking: bool | None = None,
    ) -> dict[str, Any]:
        del enable_thinking  # always on here
        answer_budget = max_tokens if max_tokens is not None else self._base.max_output_tokens
        return self._base.chat_json(
            messages,
            schema,
            schema_name,
            max_tokens=answer_budget + settings.thinking_max_tokens,
            required_keys=required_keys,
            enable_thinking=True,
            timeout_s=settings.thinking_timeout_s,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

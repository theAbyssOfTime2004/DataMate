from __future__ import annotations

import datetime
import json
from dataclasses import dataclass
from typing import Any

from pydantic import TypeAdapter, ValidationError

from datamate.config import settings
from datamate.llm.client import LLMError
from datamate.query.context import ShopContext
from datamate.query.convert import ConvertError, to_query_spec
from datamate.query.parse_schema import PERIOD_KINDS, TOP_LEVEL_KEYS, parsed_question_schema
from datamate.query.prompts import FEW_SHOT_SETS, build_messages
from datamate.query.spec import QuerySpec
from datamate.rules.model import RuleConfig
from datamate.timeres.calendar import CalendarError, LunarCalendar
from datamate.timeres.periods import PeriodError, PeriodExpr, resolve

_MAX_TOKENS = 400
_PERIOD_ADAPTER: TypeAdapter[PeriodExpr] = TypeAdapter(PeriodExpr)


@dataclass
class ParseAttempt:
    few_shot_set: int
    raw_json: dict[str, Any] | None
    error: str | None
    few_shots_dropped: int = 0


@dataclass
class ParseResult:
    spec: QuerySpec | None
    reason: str | None
    attempts: list[ParseAttempt]


def parse_question(
    question: str,
    ctx: ShopContext,
    client: Any,
    rules: RuleConfig,
    cal: LunarCalendar,
    as_of: datetime.date,
    votes: int | None = None,
) -> ParseResult:
    """Parse one question. as_of is the question's 'today', not the shop's data end."""
    vote_count = settings.parse_votes if votes is None else votes
    products = ctx.product_candidates(question)
    schema = parsed_question_schema(ctx, products)
    attempts: list[ParseAttempt] = []
    outcomes: list[tuple[str, QuerySpec | None]] = []
    for few_shot_set in range(vote_count):
        messages, dropped = _fit_messages(question, as_of, ctx, few_shot_set, client)
        outcomes.append(
            _vote(
                few_shot_set,
                messages,
                schema,
                client,
                as_of,
                rules,
                cal,
                attempts,
                dropped,
            )
        )
    return _decide(outcomes, attempts, as_of, rules, cal)


def _fit_messages(
    question: str,
    as_of: datetime.date,
    ctx: ShopContext,
    few_shot_set: int,
    client: Any,
) -> tuple[list[dict[str, str]], int]:
    total = len(FEW_SHOT_SETS[few_shot_set % len(FEW_SHOT_SETS)])
    keep = total
    while True:
        messages = build_messages(question, as_of, ctx, few_shot_set, example_limit=keep)
        if (
            keep == 0
            or client.count_tokens(_prompt_text(messages)) <= settings.llm_prompt_budget_tokens
        ):
            return messages, total - keep
        keep -= 1


def _prompt_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(f"{message['role']}\n{message['content']}" for message in messages)


def _vote(
    few_shot_set: int,
    messages: list[dict[str, str]],
    schema: dict[str, Any],
    client: Any,
    as_of: datetime.date,
    rules: RuleConfig,
    cal: LunarCalendar,
    attempts: list[ParseAttempt],
    few_shots_dropped: int,
) -> tuple[str, QuerySpec | None]:
    raw, error = _call(client, messages, schema)
    attempts.append(ParseAttempt(few_shot_set, raw, error, few_shots_dropped))
    if error is not None or raw is None:
        return "fail", None
    if raw.get("in_scope") is False:
        return "out", None
    try:
        return "ok", to_query_spec(raw, as_of, rules, cal)
    except ConvertError as exc:
        attempts[-1] = ParseAttempt(few_shot_set, raw, str(exc), few_shots_dropped)
        retried = _retry(messages, raw, str(exc))
        second, second_error = _call(client, retried, schema)
        if second_error is not None or second is None:
            attempts.append(
                ParseAttempt(
                    few_shot_set, second, second_error or "empty response", few_shots_dropped
                )
            )
            return "fail", None
        if second.get("in_scope") is False:
            attempts.append(ParseAttempt(few_shot_set, second, None, few_shots_dropped))
            return "out", None
        try:
            spec = to_query_spec(second, as_of, rules, cal)
        except ConvertError as retry_exc:
            attempts.append(ParseAttempt(few_shot_set, second, str(retry_exc), few_shots_dropped))
            return "fail", None
        attempts.append(ParseAttempt(few_shot_set, second, None, few_shots_dropped))
        return "ok", spec


def _retry(messages: list[dict[str, str]], raw: dict[str, Any], error: str) -> list[dict[str, str]]:
    return [
        *messages,
        {"role": "assistant", "content": json.dumps(raw, ensure_ascii=False)},
        {
            "role": "user",
            "content": f"That JSON is invalid: {error}. Return a corrected object.",
        },
    ]


def _call(
    client: Any, messages: list[dict[str, str]], schema: dict[str, Any]
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        raw = client.chat_json(
            messages,
            schema,
            "ParsedQuestion",
            max_tokens=_MAX_TOKENS,
            required_keys=list(TOP_LEVEL_KEYS),
        )
    except LLMError as exc:
        return None, str(exc)
    if not isinstance(raw, dict):
        return None, "response is not an object"
    return raw, None


def _decide(
    outcomes: list[tuple[str, QuerySpec | None]],
    attempts: list[ParseAttempt],
    as_of: datetime.date,
    rules: RuleConfig,
    cal: LunarCalendar,
) -> ParseResult:
    if any(status == "out" for status, _spec in outcomes):
        return ParseResult(spec=None, reason="out_of_scope", attempts=attempts)
    if any(status == "fail" for status, _spec in outcomes):
        return ParseResult(spec=None, reason="parse_error", attempts=attempts)
    specs = [spec for _status, spec in outcomes if spec is not None]
    if not specs:
        return ParseResult(spec=None, reason="parse_error", attempts=attempts)
    canonical = {semantic_key(spec, as_of, rules, cal) for spec in specs}
    if len(canonical) != 1:
        return ParseResult(spec=None, reason="disagreement", attempts=attempts)
    return ParseResult(spec=specs[0], reason=None, attempts=attempts)


def semantic_key(
    spec: QuerySpec, as_of: datetime.date, rules: RuleConfig, cal: LunarCalendar
) -> str:
    """Canonical JSON with every period replaced by the date range it resolves to.

    The engine only sees resolved ranges, so two specs with the same key give the same
    answer: month 4/2025 equals same_period_last_year of month 4/2026, and year 2026
    equals ytd once both are clipped to as_of.
    """
    resolved = _resolve_periods(spec.model_dump(mode="json"), as_of, rules, cal)
    return json.dumps(resolved, sort_keys=True, ensure_ascii=False)


def _resolve_periods(node: Any, as_of: datetime.date, rules: RuleConfig, cal: LunarCalendar) -> Any:
    if isinstance(node, dict) and node.get("kind") in PERIOD_KINDS:
        try:
            window = resolve(_PERIOD_ADAPTER.validate_python(node), as_of, rules, cal)
        except (PeriodError, CalendarError, ValidationError, ValueError):
            return {"unresolved": node}
        return {"range": [window.start.isoformat(), window.end.isoformat()]}
    if isinstance(node, dict):
        return {key: _resolve_periods(value, as_of, rules, cal) for key, value in node.items()}
    if isinstance(node, list):
        return [_resolve_periods(item, as_of, rules, cal) for item in node]
    return node

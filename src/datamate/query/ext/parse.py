"""Extension question parser: 2 votes + repair; disagreement -> abstain."""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass
from typing import Any

from pydantic import TypeAdapter, ValidationError

from datamate.config import settings
from datamate.llm.client import LLMError, LLMUnavailable
from datamate.query.context import ShopContext
from datamate.query.convert import ConvertError
from datamate.query.ext.convert import to_ext_query_spec
from datamate.query.ext.prompts import EXT_FEW_SHOT_SETS, build_ext_messages
from datamate.query.ext.schema import EXT_TOP_LEVEL_KEYS, ext_parsed_schema
from datamate.query.ext.spec import ExtQuerySpec, NoneQuery
from datamate.query.parse import ParseAttempt
from datamate.rules.model import RuleConfig
from datamate.timeres.calendar import CalendarError, LunarCalendar
from datamate.timeres.periods import PeriodError, PeriodExpr, resolve

_MAX_TOKENS = 400
_PERIOD_ADAPTER: TypeAdapter[PeriodExpr] = TypeAdapter(PeriodExpr)


@dataclass
class ExtParseResult:
    spec: ExtQuerySpec | None
    reason: str | None
    attempts: list[ParseAttempt]


def parse_ext_question(
    question: str,
    ctx: ShopContext,
    client: Any,
    rules: RuleConfig,
    cal: LunarCalendar,
    as_of: datetime.date,
    votes: int | None = None,
) -> ExtParseResult:
    """Parse a product-family question. Called only when the main parser is out of scope."""
    vote_count = settings.parse_votes if votes is None else votes
    products = ctx.product_candidates(question)
    schema = ext_parsed_schema(ctx, products)
    attempts: list[ParseAttempt] = []
    outcomes: list[tuple[str, ExtQuerySpec | None]] = []
    for few_shot_set in range(vote_count):
        messages, dropped = _fit_messages(question, as_of, ctx, few_shot_set, client)
        outcomes.append(
            _vote(
                question,
                few_shot_set,
                messages,
                schema,
                client,
                as_of,
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
    total = len(EXT_FEW_SHOT_SETS[few_shot_set % len(EXT_FEW_SHOT_SETS)])
    keep = total
    while True:
        messages = build_ext_messages(question, as_of, ctx, few_shot_set, example_limit=keep)
        if (
            keep == 0
            or client.count_tokens(_prompt_text(messages)) <= settings.llm_prompt_budget_tokens
        ):
            return messages, total - keep
        keep -= 1


def _prompt_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(f"{message['role']}\n{message['content']}" for message in messages)


def _vote(
    question: str,
    few_shot_set: int,
    messages: list[dict[str, str]],
    schema: dict[str, Any],
    client: Any,
    as_of: datetime.date,
    attempts: list[ParseAttempt],
    few_shots_dropped: int,
) -> tuple[str, ExtQuerySpec | None]:
    raw, error = _call(client, messages, schema)
    attempts.append(ParseAttempt(few_shot_set, raw, error, few_shots_dropped))
    if error is not None or raw is None:
        return "fail", None
    try:
        spec = to_ext_query_spec(raw, as_of, question)
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
        try:
            spec = to_ext_query_spec(second, as_of, question)
        except ConvertError as retry_exc:
            attempts.append(ParseAttempt(few_shot_set, second, str(retry_exc), few_shots_dropped))
            return "fail", None
        attempts.append(ParseAttempt(few_shot_set, second, None, few_shots_dropped))
        return _outcome(spec)
    return _outcome(spec)


def _outcome(spec: ExtQuerySpec) -> tuple[str, ExtQuerySpec | None]:
    if isinstance(spec, NoneQuery) or spec.family == "none":
        return "out", None
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
            "ExtParsedQuestion",
            max_tokens=_MAX_TOKENS,
            required_keys=list(EXT_TOP_LEVEL_KEYS),
        )
    except LLMUnavailable:
        raise
    except LLMError as exc:
        return None, str(exc)
    if not isinstance(raw, dict):
        return None, "response is not an object"
    return raw, None


def _decide(
    outcomes: list[tuple[str, ExtQuerySpec | None]],
    attempts: list[ParseAttempt],
    as_of: datetime.date,
    rules: RuleConfig,
    cal: LunarCalendar,
) -> ExtParseResult:
    if any(status == "out" for status, _spec in outcomes):
        return ExtParseResult(spec=None, reason="out_of_scope", attempts=attempts)
    if any(status == "fail" for status, _spec in outcomes):
        return ExtParseResult(spec=None, reason="parse_error", attempts=attempts)
    specs = [spec for _status, spec in outcomes if spec is not None]
    if not specs:
        return ExtParseResult(spec=None, reason="parse_error", attempts=attempts)
    keys = {ext_semantic_key(spec, as_of, rules, cal) for spec in specs}
    if len(keys) != 1:
        return ExtParseResult(spec=None, reason="disagreement", attempts=attempts)
    return ExtParseResult(spec=specs[0], reason=None, attempts=attempts)


def ext_semantic_key(
    spec: ExtQuerySpec, as_of: datetime.date, rules: RuleConfig, cal: LunarCalendar
) -> str:
    """Canonical JSON with periods replaced by resolved date ranges."""
    resolved = _resolve_periods(spec.model_dump(mode="json"), as_of, rules, cal)
    return json.dumps(resolved, sort_keys=True, ensure_ascii=False)


def _resolve_periods(
    node: Any, as_of: datetime.date, rules: Any, cal: LunarCalendar
) -> Any:
    if isinstance(node, dict) and node.get("kind") is not None:
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

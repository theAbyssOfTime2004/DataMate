"""End-to-end QA system: mapping → canonical DuckDB → parse → metrics."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb

from datamate.config import PROJECT_ROOT
from datamate.etl import load_or_build_canonical
from datamate.io.profile import BusinessProfile, profile_business
from datamate.io.questions import Question
from datamate.io.raw import BusinessRaw, load_business_cached
from datamate.llm import LLMUnavailable, get_client
from datamate.mapping import Mapping, infer_mapping, mapping_fingerprint
from datamate.metrics import MetricEngine
from datamate.query.context import ShopContext, load_shop_context
from datamate.query.parse import ParseAttempt, parse_question
from datamate.query.spec import (
    ArgMaxQuery,
    CompareQuery,
    MetricQuery,
    MultiQuery,
    QuerySpec,
    SeriesQuery,
    TrendQuery,
    ValueQuery,
)
from datamate.rules import RuleConfig, load_rules
from datamate.scoring.answer import Answer
from datamate.timeres import LunarCalendar, load_calendar

LLM_UNAVAILABLE = "llm_unavailable"

MappingProvider = Callable[
    [str, str, BusinessRaw, BusinessProfile, Any],
    Mapping | None,
]


def llm_mapping_provider(
    split: str,
    business_id: str,
    raw: BusinessRaw,
    profile: BusinessProfile,
    client: Any,
) -> Mapping | None:
    """Infer a mapping; return None when inference fails or validator errors remain."""
    del split, business_id
    result = infer_mapping(raw, profile, client)
    if result.mapping is None:
        return None
    if result.errors or any(issue.severity == "error" for issue in result.issues):
        return None
    return result.mapping


@dataclass
class _BusinessBundle:
    mapping: Mapping | None
    mapping_fingerprint: str | None
    engine: MetricEngine | None
    context: ShopContext | None
    connection: duckdb.DuckDBPyConnection | None
    rules: RuleConfig
    calendar: LunarCalendar
    warnings: list[str] = field(default_factory=list)
    mapping_failed: bool = False
    # DuckDB connections are not thread-safe; engine calls for one shop take turns.
    lock: threading.Lock = field(default_factory=threading.Lock)


class PipelineSystem:
    """Map shop files, parse the question, then compute with MetricEngine."""

    def __init__(
        self,
        name: str,
        mapping_provider: MappingProvider | None = None,
        client: Any | None = None,
    ) -> None:
        self.name = name
        self.mapping_provider = mapping_provider or llm_mapping_provider
        self._client = client
        self._owns_client = client is None
        self._by_business: dict[tuple[str, str], _BusinessBundle] = {}
        # One lock per shop so a slow mapping blocks only that shop's questions.
        self._build_locks: dict[tuple[str, str], threading.Lock] = {}
        self._build_locks_guard = threading.Lock()

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = get_client()
        return self._client

    def close(self) -> None:
        for bundle in self._by_business.values():
            if bundle.connection is not None:
                bundle.connection.close()
                bundle.connection = None
        self._by_business.clear()
        if self._owns_client and self._client is not None:
            close = getattr(self._client, "close", None)
            if callable(close):
                close()
            self._client = None

    def answer(self, question: Question, workdir: Path) -> tuple[Answer, dict[str, Any]]:
        trace: dict[str, Any] = {
            "system": self.name,
            "timings": {},
            "warnings": [],
            "abstain_reason": None,
            "mapping_fingerprint": None,
            "parse_attempts": [],
            "spec": None,
            "resolved_ranges": [],
            "coverage": [],
            "sql": [],
        }
        wall_t0 = time.perf_counter()
        try:
            split, business_id = _split_and_business(workdir)
            bundle = self._bundle_for(split, business_id)
            trace["warnings"] = list(bundle.warnings)
            trace["mapping_fingerprint"] = bundle.mapping_fingerprint
            if bundle.mapping_failed or bundle.engine is None or bundle.context is None:
                trace["abstain_reason"] = "mapping_failed"
                trace["timings"]["total_s"] = round(time.perf_counter() - wall_t0, 6)
                return Answer.abstain(), trace

            parse_t0 = time.perf_counter()
            parsed = parse_question(
                question.question,
                bundle.context,
                self.client,
                bundle.rules,
                bundle.calendar,
                question.as_of_date,
            )
            trace["timings"]["parse_s"] = round(time.perf_counter() - parse_t0, 6)
            trace["parse_attempts"] = [_attempt_dict(item) for item in parsed.attempts]
            if parsed.spec is None:
                trace["abstain_reason"] = parsed.reason or "parse_error"
                trace["timings"]["total_s"] = round(time.perf_counter() - wall_t0, 6)
                return Answer.abstain(), trace

            trace["spec"] = parsed.spec.model_dump(mode="json")
            engine_t0 = time.perf_counter()
            with bundle.lock:
                details = _engine_details(bundle.engine, parsed.spec, question.as_of_date)
                answer = bundle.engine.answer(parsed.spec, question.as_of_date)
            trace["timings"]["engine_s"] = round(time.perf_counter() - engine_t0, 6)
            trace["resolved_ranges"] = details["resolved_ranges"]
            trace["coverage"] = details["coverage"]
            trace["sql"] = details["sql"]
            if answer.type == "abstain":
                trace["abstain_reason"] = "engine_abstain"
            elif answer.type == "unanswerable":
                trace["abstain_reason"] = None
                trace["unanswerable_reason"] = answer.reason
            trace["timings"]["total_s"] = round(time.perf_counter() - wall_t0, 6)
            return answer, trace
        except LLMUnavailable as exc:
            # Infrastructure, not the system's judgement. Nothing is cached for the shop,
            # so the next question retries the mapping.
            trace["abstain_reason"] = LLM_UNAVAILABLE
            trace["error"] = str(exc)
            trace["timings"]["total_s"] = round(time.perf_counter() - wall_t0, 6)
            return Answer.abstain(), trace
        except Exception as exc:  # noqa: BLE001 — never crash an evaluation run
            trace["abstain_reason"] = f"{type(exc).__name__}: {exc}"
            trace["timings"]["total_s"] = round(time.perf_counter() - wall_t0, 6)
            return Answer.abstain(), trace

    def _bundle_for(self, split: str, business_id: str) -> _BusinessBundle:
        key = (split, business_id)
        with self._build_locks_guard:
            build_lock = self._build_locks.setdefault(key, threading.Lock())
        with build_lock:
            cached = self._by_business.get(key)
            if cached is not None:
                return cached
            return self._build_bundle(split, business_id)

    def _build_bundle(self, split: str, business_id: str) -> _BusinessBundle:
        key = (split, business_id)

        map_t0 = time.perf_counter()
        warnings: list[str] = []
        rules = _load_rules(split, warnings)
        calendar = load_calendar(split)
        raw = load_business_cached(split, business_id)
        profile = profile_business(raw)
        mapping = self.mapping_provider(split, business_id, raw, profile, self.client)
        mapping_s = round(time.perf_counter() - map_t0, 6)

        if mapping is None:
            bundle = _BusinessBundle(
                mapping=None,
                mapping_fingerprint=None,
                engine=None,
                context=None,
                connection=None,
                rules=rules,
                calendar=calendar,
                warnings=[*warnings, f"mapping_s={mapping_s}"],
                mapping_failed=True,
            )
            self._by_business[key] = bundle
            return bundle

        db_path, report = load_or_build_canonical(split, business_id, mapping)
        connection = duckdb.connect(str(db_path), read_only=True)
        data_start = report.integrity.min_order_date
        data_end = mapping.data_end_date or report.integrity.max_order_date
        engine = MetricEngine(connection, rules, calendar, data_start, data_end)
        context = load_shop_context(connection)
        bundle = _BusinessBundle(
            mapping=mapping,
            mapping_fingerprint=mapping_fingerprint(mapping),
            engine=engine,
            context=context,
            connection=connection,
            rules=rules,
            calendar=calendar,
            warnings=[*warnings, f"mapping_s={mapping_s}"],
            mapping_failed=False,
        )
        self._by_business[key] = bundle
        return bundle


def _split_and_business(workdir: Path) -> tuple[str, str]:
    resolved = workdir.resolve()
    if resolved.parent.name != "data":
        raise ValueError(
            f"workdir must be <split>/data/<business_id>; got parent={resolved.parent.name!r}"
        )
    return resolved.parent.parent.name, resolved.name


def _load_rules(split: str, warnings: list[str]) -> RuleConfig:
    split_path = PROJECT_ROOT / "rules" / f"{split}.json"
    if split_path.is_file():
        return load_rules(split_path)
    fallback = PROJECT_ROOT / "rules" / "train.json"
    warnings.append(f"rules/{split}.json missing; fell back to rules/train.json")
    return load_rules(fallback)


def _attempt_dict(attempt: ParseAttempt) -> dict[str, Any]:
    return {
        "few_shot_set": attempt.few_shot_set,
        "raw_json": attempt.raw_json,
        "error": attempt.error,
        "few_shots_dropped": attempt.few_shots_dropped,
    }


def _metric_queries(spec: QuerySpec) -> list[MetricQuery]:
    if isinstance(spec, ValueQuery):
        return [spec.query]
    if isinstance(spec, CompareQuery):
        return [spec.a, spec.b]
    if isinstance(spec, MultiQuery):
        return list(spec.queries)
    if isinstance(spec, SeriesQuery):
        return [
            MetricQuery(metric=spec.metric, period=period, filters=spec.filters)
            for period in spec.periods
        ]
    if isinstance(spec, TrendQuery):
        return [
            MetricQuery(metric=spec.metric, period=period, filters=spec.filters)
            for period in spec.periods
        ]
    if isinstance(spec, ArgMaxQuery):
        return [MetricQuery(metric=spec.metric, period=spec.period, filters=spec.filters)]
    return []


def _engine_details(engine: MetricEngine, spec: QuerySpec, as_of: Any) -> dict[str, list[Any]]:
    ranges: list[dict[str, Any]] = []
    coverages: list[str] = []
    sqls: list[str] = []
    for query in _metric_queries(spec):
        found = engine.compute(query, as_of)
        ranges.append(
            {
                "start": found.range.start.isoformat(),
                "end": found.range.end.isoformat(),
            }
        )
        coverages.append(found.coverage)
        sqls.append(found.sql)
    return {"resolved_ranges": ranges, "coverage": coverages, "sql": sqls}

"""End-to-end QA system: mapping → canonical DuckDB → parse → metrics."""

from __future__ import annotations

import copy
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb

from datamate.config import settings
from datamate.etl import load_or_build_canonical
from datamate.io.profile import BusinessProfile, profile_business
from datamate.io.questions import Question
from datamate.io.raw import BusinessRaw, business_files_key, load_business_cached
from datamate.llm import LLMUnavailable, get_client
from datamate.llm.thinking import ThinkingClient
from datamate.mapping import Mapping, infer_mapping, mapping_fingerprint
from datamate.mapping.cache import load_mapping, mapping_cache_path, store_mapping
from datamate.metrics import MetricEngine
from datamate.metrics.engine import RuleUnsupported
from datamate.metrics.events import has_event_log
from datamate.metrics.ext_families import product_base_unit
from datamate.metrics.rank import answer_rank
from datamate.metrics.units import answer_unit
from datamate.parse.text import expand_short_years
from datamate.query.context import ShopContext, load_shop_context
from datamate.query.defined_terms import (
    match_defined_term,
    match_unit_term,
    ranking_rows,
    unit_rows,
)
from datamate.query.events import parse_event_question
from datamate.query.ext import parse_ext_question
from datamate.query.ext.verify import verify_description, verify_ext_spec
from datamate.query.parse import ParseAttempt, parse_question
from datamate.query.rank import describe_rank, parse_rank_question
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
from datamate.query.subjective import subjective_probability
from datamate.query.units import describe_unit, parse_unit_question
from datamate.rules import RuleConfig
from datamate.rules.from_rulebook import RulebookConfig, load_split_rulebook
from datamate.rules.parse_rulebook import RuleRow
from datamate.scoring.answer import Answer
from datamate.timeres import LunarCalendar, load_calendar

LLM_UNAVAILABLE = "llm_unavailable"
# Fast-pass abstains that a thinking parse may resolve (reading the question), as opposed
# to data / engine abstains it cannot change.
_THINKING_RETRY_REASONS = frozenset(
    {"disagreement", "parse_error", "out_of_scope", "ext_verify_low"}
)

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
    cache_path = None
    if raw.business_dir.is_dir():
        cache_path = mapping_cache_path(
            split, business_id, business_files_key(raw.business_dir), client
        )
    if cache_path is not None:
        cached = load_mapping(cache_path)
        if cached is not None:
            return cached
    result = infer_mapping(raw, profile, client)
    mapping = result.mapping
    if result.errors or any(issue.severity == "error" for issue in result.issues):
        mapping = None
    # Failures are not stored: an unreachable LLM must not stick to the shop.
    if cache_path is not None and mapping is not None:
        store_mapping(cache_path, mapping)
    return mapping


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
    rule_notes: list[Any] = field(default_factory=list)
    unsupported_knobs: frozenset[str] = field(default_factory=frozenset)
    rulebook_sha256: str = ""
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
        # RulebookConfig is loaded once per split for the whole run.
        self._rulebook_by_split: dict[str, RulebookConfig] = {}
        self._rulebook_lock = threading.Lock()
        # Rulebook rows that define a ranking term, found once per split.
        self._ranking_rows_by_split: dict[str, list[RuleRow]] = {}
        self._ranking_rows_lock = threading.Lock()
        self._unit_rows_by_split: dict[str, list[RuleRow]] = {}

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
            "ext_family": None,
            "resolved_ranges": [],
            "coverage": [],
            "sql": [],
        }
        wall_t0 = time.perf_counter()
        written = expand_short_years(question.question)
        if written != question.question:
            trace["question_rewritten"] = written
            question = question.model_copy(update={"question": written})
        try:
            split, business_id = _split_and_business(workdir)
            term = self._defined_term(split, question.question, trace)
            # A term the rulebook defines as a count is not an opinion ("trung thành nhất").
            if settings.subjective_check and term is None:
                p_subjective = subjective_probability(question.question, self.client)
                trace["subjective_p"] = round(p_subjective, 4)
                if p_subjective >= settings.subjective_threshold:
                    trace["timings"]["total_s"] = round(time.perf_counter() - wall_t0, 6)
                    return Answer.unanswerable("asks for an opinion"), trace
            bundle = self._bundle_for(split, business_id)
            trace["warnings"] = list(bundle.warnings)
            trace["mapping_fingerprint"] = bundle.mapping_fingerprint
            trace["rulebook_sha256"] = bundle.rulebook_sha256
            trace["rule_notes_n"] = len(bundle.rule_notes)
            if bundle.mapping_failed or bundle.engine is None or bundle.context is None:
                trace["abstain_reason"] = "mapping_failed"
                trace["timings"]["total_s"] = round(time.perf_counter() - wall_t0, 6)
                return Answer.abstain(), trace

            if term is not None:
                ranked = self._answer_via_rank(question, term, bundle, trace, wall_t0)
                if ranked is not None:
                    return ranked
            base_trace = copy.deepcopy(trace)
            if settings.parse_thinking == "always":
                trace["parse_thinking"] = True
                return self._parse_and_answer(
                    question, bundle, trace, wall_t0, ThinkingClient(self.client)
                )
            answer, trace = self._parse_and_answer(question, bundle, trace, wall_t0, self.client)
            reason = trace.get("abstain_reason")
            if (
                settings.parse_thinking == "retry"
                and answer.type == "abstain"
                and reason in _THINKING_RETRY_REASONS
            ):
                retry_trace = copy.deepcopy(base_trace)
                retry_trace["parse_thinking"] = True
                retry_trace["fast_pass"] = {
                    "abstain_reason": reason,
                    "ext_family": trace.get("ext_family"),
                    "total_s": trace["timings"].get("total_s"),
                }
                try:
                    answer, trace = self._parse_and_answer(
                        question, bundle, retry_trace, wall_t0, ThinkingClient(self.client)
                    )
                except Exception as exc:  # noqa: BLE001 — keep the fast-pass abstain
                    # A slow or failed thinking call must not turn into llm_unavailable
                    # (predict would rerun the whole question).
                    trace["thinking_error"] = f"{type(exc).__name__}: {exc}"
                    return answer, trace
            if answer.type == "abstain" and settings.unit_terms:
                return self._answer_via_unit(question, split, bundle, answer, trace, wall_t0)
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

    def _parse_and_answer(
        self,
        question: Question,
        bundle: _BusinessBundle,
        trace: dict[str, Any],
        wall_t0: float,
        parse_client: Any,
    ) -> tuple[Answer, dict[str, Any]]:
        """Parse (main, then ext) with `parse_client` and compute. Mapping and verifier
        calls keep the shared client."""
        assert bundle.engine is not None and bundle.context is not None
        parse_t0 = time.perf_counter()
        parsed = parse_question(
            question.question,
            bundle.context,
            parse_client,
            bundle.rules,
            bundle.calendar,
            question.as_of_date,
        )
        trace["timings"]["parse_s"] = round(time.perf_counter() - parse_t0, 6)
        trace["parse_attempts"] = [_attempt_dict(item) for item in parsed.attempts]
        if parsed.period_adjudicated:
            trace["period_adjudicated"] = True
        if parsed.spec is None:
            # Extension parser only when every main attempt is out of scope.
            if parsed.reason == "out_of_scope":
                return self._answer_via_ext(
                    question, bundle, trace, wall_t0, parse_t0, parse_client
                )
            trace["abstain_reason"] = parsed.reason or "parse_error"
            trace["timings"]["total_s"] = round(time.perf_counter() - wall_t0, 6)
            return Answer.abstain(), trace

        trace["spec"] = parsed.spec.model_dump(mode="json")
        engine_t0 = time.perf_counter()
        try:
            with bundle.lock:
                details = _engine_details(bundle.engine, parsed.spec, question.as_of_date)
                answer = bundle.engine.answer(parsed.spec, question.as_of_date)
        except RuleUnsupported as exc:
            trace["abstain_reason"] = f"rule_unsupported:{exc.knob}"
            trace["timings"]["engine_s"] = round(time.perf_counter() - engine_t0, 6)
            trace["timings"]["total_s"] = round(time.perf_counter() - wall_t0, 6)
            return Answer.abstain(), trace
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

    def _answer_via_ext(
        self,
        question: Question,
        bundle: _BusinessBundle,
        trace: dict[str, Any],
        wall_t0: float,
        parse_t0: float,
        parse_client: Any,
    ) -> tuple[Answer, dict[str, Any]]:
        assert bundle.engine is not None and bundle.context is not None
        ext_t0 = time.perf_counter()
        ext = None
        if settings.event_path:
            with bundle.lock:
                has_log = has_event_log(bundle.engine)
            if has_log:
                # Shops with a delivery log: event questions first, everything else goes
                # on to the unchanged extension parser.
                event = parse_event_question(
                    question.question,
                    parse_client,
                    bundle.rules,
                    bundle.calendar,
                    question.as_of_date,
                )
                trace["event_parse_attempts"] = [_attempt_dict(item) for item in event.attempts]
                if event.spec is not None:
                    ext = event
        if ext is None:
            ext = parse_ext_question(
                question.question,
                bundle.context,
                parse_client,
                bundle.rules,
                bundle.calendar,
                question.as_of_date,
            )
        trace["timings"]["ext_parse_s"] = round(time.perf_counter() - ext_t0, 6)
        trace["timings"]["parse_s"] = round(time.perf_counter() - parse_t0, 6)
        trace["ext_parse_attempts"] = [_attempt_dict(item) for item in ext.attempts]
        if ext.spec is None:
            trace["ext_family"] = None
            if settings.enable_plan_path and (ext.reason or "out_of_scope") == "out_of_scope":
                return self._answer_via_plan(question, bundle, trace, wall_t0)
            trace["abstain_reason"] = ext.reason or "out_of_scope"
            trace["timings"]["total_s"] = round(time.perf_counter() - wall_t0, 6)
            return Answer.abstain(), trace
        family = ext.spec.family
        trace["ext_family"] = family
        if family == "none" or family not in settings.enabled_families:
            if settings.enable_plan_path and family == "none":
                return self._answer_via_plan(question, bundle, trace, wall_t0)
            trace["abstain_reason"] = "out_of_scope"
            trace["timings"]["total_s"] = round(time.perf_counter() - wall_t0, 6)
            return Answer.abstain(), trace
        trace["spec"] = ext.spec.model_dump(mode="json")
        verify_t0 = time.perf_counter()
        product = getattr(ext.spec, "product", None)
        product_unit = None
        customer_who = None
        debt_meta = None
        _debt_families = {
            "debt_total",
            "top_debtors",
            "debt_aging",
            "debt_added",
            "debt_collected",
            "customer_debt_flow",
        }
        if product or family in {"customers_min_orders", "top_customers"} | _debt_families:
            with bundle.lock:
                if product:
                    product_unit = product_base_unit(bundle.engine, product)
                if family in {"customers_min_orders", "top_customers"}:
                    from datamate.metrics.ext_families import customer_who_phrase

                    customer_who = customer_who_phrase(bundle.engine)
                if family in _debt_families:
                    from datamate.metrics.ext_families import debt_describe_meta
                    from datamate.timeres.periods import resolve as resolve_period

                    as_of_day = None
                    date_expr = getattr(ext.spec, "date", None)
                    if date_expr is not None:
                        try:
                            window = resolve_period(
                                date_expr,
                                question.as_of_date,
                                bundle.rules,
                                bundle.calendar,
                            )
                            if window.start == window.end:
                                as_of_day = window.start
                        except Exception:  # noqa: BLE001
                            as_of_day = None
                    debt_meta = debt_describe_meta(bundle.engine, as_of_day)
        p_yes, description = verify_ext_spec(
            question.question,
            ext.spec,
            self.client,
            bundle.rules,
            bundle.calendar,
            question.as_of_date,
            product_unit=product_unit,
            customer_who=customer_who,
            debt_meta=debt_meta,
        )
        trace["timings"]["ext_verify_s"] = round(time.perf_counter() - verify_t0, 6)
        trace["ext_verify_p"] = round(p_yes, 4)
        trace["ext_verify_description"] = description
        trace["ext_split_vote"] = ext.split_vote
        trace["ext_period_adjudicated"] = ext.period_adjudicated
        # A tie-broken spec gets the same stricter bar as a split vote.
        threshold = (
            settings.ext_verify_threshold_split
            if ext.split_vote or ext.period_adjudicated
            else settings.ext_verify_threshold
        )
        if p_yes < threshold:
            trace["abstain_reason"] = "ext_verify_low"
            trace["timings"]["total_s"] = round(time.perf_counter() - wall_t0, 6)
            return Answer.abstain(), trace
        engine_t0 = time.perf_counter()
        try:
            with bundle.lock:
                answer = bundle.engine.answer_ext(ext.spec, question.as_of_date)
        except RuleUnsupported as exc:
            trace["abstain_reason"] = f"rule_unsupported:{exc.knob}"
            trace["timings"]["engine_s"] = round(time.perf_counter() - engine_t0, 6)
            trace["timings"]["total_s"] = round(time.perf_counter() - wall_t0, 6)
            return Answer.abstain(), trace
        trace["timings"]["engine_s"] = round(time.perf_counter() - engine_t0, 6)
        # Logged for ext_threshold.py when the run uses a low threshold.
        trace["ext_answer"] = answer.model_dump(mode="json")
        if answer.type == "abstain":
            trace["abstain_reason"] = "engine_abstain"
        elif answer.type == "unanswerable":
            trace["abstain_reason"] = None
            trace["unanswerable_reason"] = answer.reason
        trace["timings"]["total_s"] = round(time.perf_counter() - wall_t0, 6)
        return answer, trace

    def _answer_via_plan(
        self,
        question: Question,
        bundle: _BusinessBundle,
        trace: dict[str, Any],
        wall_t0: float,
    ) -> tuple[Answer, dict[str, Any]]:
        """E3: compose a QueryPlan when main + ext are both out of scope."""
        assert bundle.engine is not None and bundle.context is not None
        from datamate.query.plan.describe import describe_plan
        from datamate.query.plan.execute import try_answer_plan
        from datamate.query.plan.parse import parse_plan_question

        trace["ext_family"] = "plan"
        plan_t0 = time.perf_counter()
        parsed = parse_plan_question(
            question.question,
            bundle.context,
            self.client,
            bundle.rules,
            bundle.calendar,
            question.as_of_date,
            list(bundle.rule_notes),
        )
        trace["timings"]["plan_parse_s"] = round(time.perf_counter() - plan_t0, 6)
        trace["plan_selected_notes"] = list(parsed.selected_note_ids)
        if parsed.notes_error is not None:
            trace["plan_notes_error"] = parsed.notes_error
        trace["plan_parse_attempts"] = [_attempt_dict(item) for item in parsed.attempts]
        if parsed.plan is None:
            trace["abstain_reason"] = parsed.reason or "plan_parse_error"
            trace["timings"]["total_s"] = round(time.perf_counter() - wall_t0, 6)
            return Answer.abstain(), trace

        trace["plan"] = parsed.plan.model_dump(mode="json")
        description = describe_plan(parsed.plan, question.as_of_date, bundle.rules, bundle.calendar)
        verify_t0 = time.perf_counter()
        p_yes = verify_description(question.question, description, self.client)
        trace["timings"]["plan_verify_s"] = round(time.perf_counter() - verify_t0, 6)
        trace["plan_verify_p"] = round(p_yes, 4)
        trace["plan_verify_description"] = description
        if p_yes < settings.plan_verify_threshold:
            trace["abstain_reason"] = "plan_verify_low"
            trace["timings"]["total_s"] = round(time.perf_counter() - wall_t0, 6)
            return Answer.abstain(), trace

        engine_t0 = time.perf_counter()
        with bundle.lock:
            answer, compiled, err = try_answer_plan(parsed.plan, bundle.engine, question.as_of_date)
        trace["timings"]["plan_engine_s"] = round(time.perf_counter() - engine_t0, 6)
        if compiled is not None:
            trace["plan_sql"] = compiled.sql
            trace["plan_answer_type"] = compiled.answer_type
        if err is not None:
            trace["abstain_reason"] = err
            trace["timings"]["total_s"] = round(time.perf_counter() - wall_t0, 6)
            return Answer.abstain(), trace
        if answer.type == "abstain":
            trace["abstain_reason"] = "plan_engine_abstain"
        elif answer.type == "unanswerable":
            trace["abstain_reason"] = None
            trace["unanswerable_reason"] = answer.reason
        else:
            trace["abstain_reason"] = None
        trace["timings"]["total_s"] = round(time.perf_counter() - wall_t0, 6)
        return answer, trace

    def _defined_term(self, split: str, question: str, trace: dict[str, Any]) -> RuleRow | None:
        """The rulebook ranking term the question asks about, if any."""
        if not settings.defined_term_check:
            return None
        # One classification per split; other questions wait instead of repeating it.
        with self._ranking_rows_lock:
            rows = self._ranking_rows_by_split.get(split)
            if rows is None:
                rows = ranking_rows(list(self._rulebook_for(split).notes), self.client)
                self._ranking_rows_by_split[split] = rows
        term, asked = match_defined_term(question, rows, self.client)
        if asked:
            trace["defined_term_asked"] = asked
        if term is not None:
            trace["defined_term"] = term.name
        return term

    def _answer_via_rank(
        self,
        question: Question,
        term: RuleRow,
        bundle: _BusinessBundle,
        trace: dict[str, Any],
        wall_t0: float,
    ) -> tuple[Answer, dict[str, Any]] | None:
        """Rank block for a rulebook term; None when the term does not fit it."""
        assert bundle.engine is not None and bundle.context is not None
        rank_t0 = time.perf_counter()
        parsed = parse_rank_question(
            question.question,
            term,
            bundle.context,
            self.client,
            bundle.rules,
            bundle.calendar,
            question.as_of_date,
        )
        trace["timings"]["rank_parse_s"] = round(time.perf_counter() - rank_t0, 6)
        trace["rank_parse_raw"] = parsed.raw
        if parsed.reason == "rank_none":
            return None
        trace["ext_family"] = "rank"

        def done(answer: Answer, reason: str | None) -> tuple[Answer, dict[str, Any]]:
            trace["abstain_reason"] = reason
            if answer.type == "unanswerable":
                trace["unanswerable_reason"] = answer.reason
            trace["timings"]["total_s"] = round(time.perf_counter() - wall_t0, 6)
            return answer, trace

        if parsed.spec is None:
            trace["rank_parse_errors"] = parsed.errors
            return done(Answer.abstain(), parsed.reason)
        trace["spec"] = parsed.spec.model_dump(mode="json")
        customer_who = None
        if parsed.spec.entity == "customer":
            from datamate.metrics.ext_families import customer_who_phrase

            with bundle.lock:
                customer_who = customer_who_phrase(bundle.engine)
        description = describe_rank(
            parsed.spec, question.as_of_date, bundle.rules, bundle.calendar, customer_who
        )
        asked = f"{question.question}\n(Rulebook: {term.name} = {term.text})"
        p_yes = verify_description(asked, description, self.client)
        trace["rank_verify_p"] = round(p_yes, 4)
        trace["rank_verify_description"] = description
        if p_yes < settings.rank_verify_threshold:
            return done(Answer.abstain(), "rank_verify_low")
        engine_t0 = time.perf_counter()
        with bundle.lock:
            answer = answer_rank(bundle.engine, parsed.spec, question.as_of_date)
        trace["timings"]["engine_s"] = round(time.perf_counter() - engine_t0, 6)
        return done(answer, "engine_abstain" if answer.type == "abstain" else None)

    def _answer_via_unit(
        self,
        question: Question,
        split: str,
        bundle: _BusinessBundle,
        abstained: Answer,
        trace: dict[str, Any],
        wall_t0: float,
    ) -> tuple[Answer, dict[str, Any]]:
        """After an abstain: count the shop's record unit when the rulebook names it."""
        assert bundle.engine is not None
        with self._ranking_rows_lock:
            rows = self._unit_rows_by_split.get(split)
            if rows is None:
                rows = unit_rows(list(self._rulebook_for(split).notes), self.client)
                self._unit_rows_by_split[split] = rows
        term, asked = match_unit_term(question.question, rows, self.client)
        if asked:
            trace["unit_term_asked"] = asked
        if term is None:
            return abstained, trace
        parsed = parse_unit_question(
            question.question,
            term,
            self.client,
            bundle.rules,
            bundle.calendar,
            question.as_of_date,
        )
        trace["unit_parse_attempts"] = [_attempt_dict(item) for item in parsed.attempts]
        if parsed.spec is None:
            trace["unit_reason"] = parsed.reason
            return abstained, trace
        description = describe_unit(
            parsed.spec, term, question.as_of_date, bundle.rules, bundle.calendar
        )
        asked_text = f"{question.question}\n(Rulebook: {term.name} = {term.text})"
        p_yes = verify_description(asked_text, description, self.client)
        trace["unit_verify_p"] = round(p_yes, 4)
        trace["unit_verify_description"] = description
        if p_yes < settings.rank_verify_threshold:
            trace["unit_reason"] = "unit_verify_low"
            return abstained, trace
        with bundle.lock:
            answer = answer_unit(bundle.engine, parsed.spec, question.as_of_date)
        if answer.type == "abstain":
            trace["unit_reason"] = "engine_abstain"
            return abstained, trace
        trace["ext_family"] = "unit"
        trace["spec"] = parsed.spec.model_dump(mode="json")
        trace["abstain_reason"] = None
        if answer.type == "unanswerable":
            trace["unanswerable_reason"] = answer.reason
        trace["timings"]["total_s"] = round(time.perf_counter() - wall_t0, 6)
        return answer, trace

    def _rulebook_for(self, split: str) -> RulebookConfig:
        with self._rulebook_lock:
            cached = self._rulebook_by_split.get(split)
            if cached is not None:
                return cached
            warnings: list[str] = []
            cfg = load_split_rulebook(split, self.client, warnings_out=warnings)
            # Preserve load warnings on the config object for the first bundle.
            if warnings and not cfg.warnings:
                cfg.warnings.extend(warnings)
            self._rulebook_by_split[split] = cfg
            return cfg

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
        rulebook = self._rulebook_for(split)
        rules = rulebook.rules
        warnings.extend(rulebook.warnings)
        unsupported = frozenset(rulebook.unsupported)
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
                rule_notes=list(rulebook.notes),
                unsupported_knobs=unsupported,
                rulebook_sha256=rulebook.rulebook_sha256,
            )
            self._by_business[key] = bundle
            return bundle

        db_path, report = load_or_build_canonical(split, business_id, mapping)
        connection = duckdb.connect(str(db_path), read_only=True)
        data_start = report.integrity.min_order_date
        data_end = mapping.data_end_date or report.integrity.max_order_date
        engine = MetricEngine(
            connection,
            rules,
            calendar,
            data_start,
            data_end,
            unsupported_knobs=unsupported,
        )
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
            rule_notes=list(rulebook.notes),
            unsupported_knobs=unsupported,
            rulebook_sha256=rulebook.rulebook_sha256,
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

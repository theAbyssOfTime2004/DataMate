"""System C: LLM writes pandas code; sandbox executes it on raw string tables."""

from __future__ import annotations

import json
import math
import pickle
import re
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from datamate.config import settings
from datamate.direct.prompts import (
    DIRECT_MAX_TOKENS,
    DIRECT_REQUIRED_KEYS,
    DIRECT_SCHEMA,
    fit_messages,
    rulebook_excerpt,
)
from datamate.direct.sandbox import check_code, run_code
from datamate.io.profile import profile_business, render_business
from datamate.io.questions import Question
from datamate.io.raw import load_business_cached
from datamate.llm import LLMError, LLMUnavailable, get_client
from datamate.metrics.format import render_percent
from datamate.rules import RuleConfig
from datamate.scoring.answer import Answer
from datamate.systems.pipeline import LLM_UNAVAILABLE

_DATE_FULL = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATE_MONTH = re.compile(r"^\d{4}-\d{2}$")


@dataclass
class _BusinessBundle:
    tables_dir: Path
    business_text: str
    rules_text: str
    table_keys: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class DirectSystem:
    """Answer by asking the LLM for pandas code and running it in a sandbox."""

    def __init__(self, name: str = "direct", client: Any | None = None) -> None:
        self.name = name
        self._client = client
        self._owns_client = client is None
        self._by_business: dict[tuple[str, str], _BusinessBundle] = {}
        self._build_locks: dict[tuple[str, str], threading.Lock] = {}
        self._build_locks_guard = threading.Lock()
        self._temp_roots: list[Path] = []

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = get_client()
        return self._client

    def close(self) -> None:
        for bundle in self._by_business.values():
            shutil.rmtree(bundle.tables_dir, ignore_errors=True)
        self._by_business.clear()
        for root in self._temp_roots:
            shutil.rmtree(root, ignore_errors=True)
        self._temp_roots.clear()
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
            "attempts": [],
            "llm_calls": 0,
            "result": None,
        }
        wall_t0 = time.perf_counter()
        try:
            split, business_id = _split_and_business(workdir)
            bundle = self._bundle_for(split, business_id)
            trace["warnings"] = list(bundle.warnings)

            attempts: list[tuple[str, str]] = []
            max_tries = 1 + settings.direct_repairs
            answer: Answer | None = None
            only_llm_errors = True
            for _ in range(max_tries):
                llm_t0 = time.perf_counter()
                messages = fit_messages(
                    question.question,
                    question.as_of_date,
                    bundle.business_text,
                    bundle.rules_text,
                    attempts,
                    self.client.count_tokens,
                    table_keys=bundle.table_keys,
                )
                try:
                    raw = self.client.chat_json(
                        messages,
                        DIRECT_SCHEMA,
                        "DirectCode",
                        max_tokens=DIRECT_MAX_TOKENS,
                        required_keys=DIRECT_REQUIRED_KEYS,
                    )
                except LLMUnavailable:
                    raise
                except LLMError as exc:
                    only_llm_errors = only_llm_errors and True
                    err = f"llm_error: {str(exc)[:200]}"
                    attempts.append(("", err))
                    trace["llm_calls"] = int(trace["llm_calls"]) + 1
                    trace["timings"]["last_llm_s"] = round(time.perf_counter() - llm_t0, 6)
                    trace["attempts"].append(
                        {"code": "", "error": err, "elapsed_s": 0.0}
                    )
                    continue

                trace["llm_calls"] = int(trace["llm_calls"]) + 1
                trace["timings"]["last_llm_s"] = round(time.perf_counter() - llm_t0, 6)

                if not isinstance(raw, dict):
                    only_llm_errors = False
                    attempts.append(("", "LLM response is not an object"))
                    continue
                if raw.get("in_scope") is False:
                    trace["abstain_reason"] = "out_of_scope"
                    trace["timings"]["total_s"] = round(time.perf_counter() - wall_t0, 6)
                    return Answer.abstain(), trace

                code = raw.get("code")
                if not isinstance(code, str):
                    only_llm_errors = False
                    attempts.append(("", "code is not a string"))
                    continue

                blocked = check_code(code)
                if blocked is not None:
                    only_llm_errors = False
                    attempts.append((code, blocked))
                    trace["attempts"].append(
                        {"code": code, "error": blocked, "elapsed_s": 0.0}
                    )
                    continue

                ran = run_code(code, bundle.tables_dir, question.as_of_date)
                trace["attempts"].append(
                    {
                        "code": code,
                        "error": ran.error,
                        "elapsed_s": ran.elapsed_s,
                    }
                )
                if not ran.ok or ran.result is None:
                    only_llm_errors = False
                    attempts.append((code, ran.error or "sandbox failure"))
                    continue

                if ran.result.get("type") == "abstain":
                    trace["abstain_reason"] = "model_abstain"
                    trace["result"] = ran.result
                    trace["timings"]["total_s"] = round(time.perf_counter() - wall_t0, 6)
                    return Answer.abstain(), trace

                if ran.result.get("type") == "unanswerable":
                    # Prefer abstain (score 0) over trusting a code-emitted unanswerable (-1 if wrong).
                    trace["abstain_reason"] = "model_unanswerable"
                    trace["result"] = ran.result
                    trace["timings"]["total_s"] = round(time.perf_counter() - wall_t0, 6)
                    return Answer.abstain(), trace

                formatted = _format_result(ran.result)
                if isinstance(formatted, Answer):
                    only_llm_errors = False
                    answer = formatted
                    trace["result"] = ran.result
                    break
                only_llm_errors = False
                attempts.append((code, str(formatted)))
                trace["attempts"][-1]["error"] = str(formatted)

            if answer is None:
                if only_llm_errors and attempts and all(
                    err.startswith("llm_error:") for _code, err in attempts
                ):
                    trace["abstain_reason"] = "llm_error"
                else:
                    trace["abstain_reason"] = "code_error"
                trace["timings"]["total_s"] = round(time.perf_counter() - wall_t0, 6)
                return Answer.abstain(), trace

            trace["timings"]["total_s"] = round(time.perf_counter() - wall_t0, 6)
            return answer, trace
        except LLMUnavailable as exc:
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
            bundle = self._build_bundle(split, business_id)
            self._by_business[key] = bundle
            return bundle

    def _build_bundle(self, split: str, business_id: str) -> _BusinessBundle:
        raw = load_business_cached(split, business_id)
        profile = profile_business(raw)
        business_text = render_business(profile, include_readme=True)
        rulebook_path = settings.split_dir(split) / "rulebook.md"
        rules_text = ""
        warnings: list[str] = []
        if rulebook_path.is_file():
            rules_text = rulebook_excerpt(
                rulebook_path.read_text(encoding="utf-8", errors="replace")
            )
        else:
            warnings.append(f"rulebook.md missing for split {split}")

        root = Path(tempfile.mkdtemp(prefix=f"datamate_tables_{business_id}_"))
        self._temp_roots.append(root)
        tables_dir = root / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        files: dict[str, str] = {}
        table_keys: list[str] = []
        for index, table in enumerate(raw.tables):
            file_name = f"{index:04d}.pkl"
            with (tables_dir / file_name).open("wb") as handle:
                pickle.dump(table.df, handle, protocol=pickle.HIGHEST_PROTOCOL)
            files[file_name] = table.name
            table_keys.append(table.name)
        aliases = _build_alias_map(table_keys)
        manifest = {"files": files, "aliases": aliases}
        (tables_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False),
            encoding="utf-8",
        )
        return _BusinessBundle(
            tables_dir=tables_dir,
            business_text=business_text,
            rules_text=rules_text,
            table_keys=table_keys,
            warnings=warnings,
        )


def _alias_candidates(full_name: str) -> list[str]:
    """Candidate short keys for a full table name (excluding the full name itself)."""
    out: list[str] = []
    seen: set[str] = set()

    def add(alias: str) -> None:
        if alias and alias != full_name and alias not in seen:
            seen.add(alias)
            out.append(alias)

    if "!" in full_name:
        file_part, sheet = full_name.split("!", 1)
        add(sheet)
        stem = Path(file_part).stem
        add(stem)
        add(f"{stem}!{sheet}")
    else:
        add(Path(full_name).stem)
    return out


def _build_alias_map(table_names: list[str]) -> dict[str, str]:
    """Map unambiguous aliases -> full table name; skip collisions across tables."""
    claims: dict[str, set[str]] = {}
    full_set = set(table_names)
    for name in table_names:
        for alias in _alias_candidates(name):
            claims.setdefault(alias, set()).add(name)
    aliases: dict[str, str] = {}
    for alias, targets in sorted(claims.items()):
        if alias in full_set:
            continue
        if len(targets) == 1:
            aliases[alias] = next(iter(targets))
    return aliases


def _split_and_business(workdir: Path) -> tuple[str, str]:
    resolved = workdir.resolve()
    if resolved.parent.name != "data":
        raise ValueError(
            f"workdir must be <split>/data/<business_id>; got parent={resolved.parent.name!r}"
        )
    return resolved.parent.parent.name, resolved.name


def _format_result(payload: dict[str, Any]) -> Answer | str:
    """Validate sandbox output into an Answer, or return an error string."""
    result_type = payload.get("type")
    if not isinstance(result_type, str):
        return "bad_result: missing type"
    if result_type == "unanswerable":
        reason = payload.get("reason")
        return Answer.unanswerable(reason if isinstance(reason, str) else None)
    if result_type == "abstain":
        return "model_abstain"
    if "answer" not in payload:
        return "bad_result: missing answer"

    value = payload["answer"]
    if result_type == "number":
        formatted = _format_number(value)
        if isinstance(formatted, str):
            return formatted
        return Answer(type="number", answer=formatted)
    if result_type == "percent":
        try:
            amount = _as_decimal(value)
        except (ValueError, ArithmeticError, TypeError):
            return "bad_result: percent is not numeric"
        if not amount.is_finite():
            return "bad_result: not finite"
        return Answer(type="percent", answer=render_percent(amount, RuleConfig()))
    if result_type == "date":
        if not isinstance(value, str) or not (_DATE_FULL.match(value) or _DATE_MONTH.match(value)):
            return "bad_result: date must be YYYY-MM-DD or YYYY-MM"
        return Answer(type="date", answer=value)
    if result_type == "string":
        if not isinstance(value, str):
            return "bad_result: string answer must be a string"
        return Answer(type="string", answer=value)
    if result_type == "list":
        if not isinstance(value, list):
            return "bad_result: list answer must be a list"
        items: list[Any] = []
        for item in value:
            if isinstance(item, str):
                items.append(item)
            elif isinstance(item, bool):
                return "bad_result: list items must be numbers or strings"
            elif isinstance(item, (int, float, Decimal)):
                number = _format_number(item)
                if isinstance(number, str):
                    return number
                items.append(number)
            else:
                return "bad_result: list items must be numbers or strings"
        return Answer(type="list", answer=items)
    if result_type in {"boolean", "table"}:
        return f"bad_result: type {result_type} not supported by direct"
    return "bad_result"


def _as_decimal(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        raise TypeError("bool is not numeric")
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return Decimal("NaN") if math.isnan(value) else Decimal("Infinity") * (
                1 if value > 0 else -1
            )
        return Decimal(repr(value))
    if isinstance(value, str):
        return Decimal(value.strip())
    raise TypeError(f"not numeric: {type(value)!r}")


def _format_number(value: Any) -> int | str:
    """Round half up to int (money is integer VND). Non-finite -> error string."""
    try:
        amount = _as_decimal(value)
    except (ValueError, ArithmeticError, TypeError):
        return "bad_result: number is not numeric"
    if not amount.is_finite():
        return "bad_result: not finite"
    return int(amount.quantize(Decimal(1), rounding=ROUND_HALF_UP))

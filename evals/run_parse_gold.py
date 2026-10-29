"""Score the question parser on the 33 hand-written gold QuerySpecs."""

from __future__ import annotations

import dataclasses
import datetime
import json
import sys
import time
from pathlib import Path
from typing import Any

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datamate.config import settings
from datamate.etl import load_or_build_canonical
from datamate.io.questions import load_questions, load_references
from datamate.llm import get_client
from datamate.mapping import load_mapping_yaml
from datamate.metrics import MetricEngine
from datamate.query.context import ShopContext, load_shop_context
from datamate.query.parse import parse_question, semantic_key
from datamate.query.spec import QuerySpec
from datamate.rules import load_rules
from datamate.scoring.answer import Answer, to_submission_line
from datamate.scoring.scorer import score_one
from datamate.timeres import LunarCalendar, load_calendar
from evals.metrics_gold_cases import CASES, GoldCase

GOLD_DIR = PROJECT_ROOT / "evals" / "gold_mappings"


def _unique_run_dir(base_runs_dir: Path, base_name: str) -> Path:
    candidate = base_runs_dir / base_name
    if not candidate.exists():
        return candidate
    suffix = 2
    while True:
        candidate = base_runs_dir / f"{base_name}_{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def _dump(spec: QuerySpec | None) -> str:
    if spec is None:
        return "null"
    return json.dumps(spec.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))


def _specs_equivalent(
    gold: QuerySpec,
    parsed: QuerySpec | None,
    as_of: datetime.date,
    rules: Any,
    cal: LunarCalendar,
) -> bool:
    if parsed is None:
        return False
    return semantic_key(gold, as_of, rules, cal) == semantic_key(parsed, as_of, rules, cal)


def _result_label(score: int) -> str:
    if score == 1:
        return "correct"
    if score == -1:
        return "wrong"
    return "abstain"


def _reason_text(parse_reason: str | None, answer: Answer | None, score: int) -> str:
    if parse_reason:
        return parse_reason
    if answer is None:
        return ""
    if answer.type == "abstain":
        return "engine_abstain"
    if answer.type == "unanswerable":
        return answer.reason or "unanswerable"
    if score == -1:
        return f"got {answer.type} {answer.answer}"
    return ""


def _load_shops(
    rules: Any,
    calendar: LunarCalendar,
) -> tuple[dict[str, MetricEngine], dict[str, ShopContext], list[duckdb.DuckDBPyConnection]]:
    engines: dict[str, MetricEngine] = {}
    contexts: dict[str, ShopContext] = {}
    connections: list[duckdb.DuckDBPyConnection] = []
    for business_id in sorted({case.business_id for case in CASES}):
        mapping = load_mapping_yaml(GOLD_DIR / f"{business_id}.yaml")
        db_path, report = load_or_build_canonical("train", business_id, mapping)
        connection = duckdb.connect(str(db_path), read_only=True)
        connections.append(connection)
        engines[business_id] = MetricEngine(
            connection,
            rules,
            calendar,
            report.integrity.min_order_date,
            mapping.data_end_date,
        )
        contexts[business_id] = load_shop_context(connection)
    return engines, contexts, connections


def _summary_markdown(rows: list[dict[str, Any]], totals: dict[str, int]) -> str:
    lines = [
        "# Parse gold",
        "",
        f"- correct: {totals['correct']}",
        f"- wrong: {totals['wrong']}",
        f"- abstain: {totals['abstain']}",
        f"- semantically identical specs: {totals['identical']} / {len(rows)}",
        f"- exact JSON identical: {totals['exact']} / {len(rows)}",
        f"- LLM calls: {totals['calls']}",
        f"- LLM latency: {totals['latency_s']:.1f}s",
        "",
        "Two specs are semantically identical when every field matches and each period",
        "resolves to the same inclusive date range.",
        "",
        "## Abstain by reason",
        "",
        "| reason | n |",
        "| --- | --- |",
    ]
    reasons = totals["abstain_reasons"]
    if not reasons:
        lines.append("| — | 0 |")
    else:
        for reason, count in sorted(reasons.items()):
            lines.append(f"| {reason} | {count} |")
    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| n | id | result | reason | calls | latency_s | identical |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        reason = row["reason"].replace("|", "/") if row["reason"] else "—"
        identical = "yes" if row["identical"] else "no"
        lines.append(
            f"| {row['n']:02d} | {row['id']} | {row['result']} | {reason} | "
            f"{row['calls']} | {row['latency_s']:.2f} | {identical} |"
        )
    differing = [row for row in rows if row["gold_json"] != row["parsed_json"]]
    if differing:
        lines.extend(["", "## Differing specs", ""])
        for row in differing:
            lines.append(f"### {row['n']:02d} {row['id']} ({row['result']})")
            lines.append("")
            lines.append(f"gold: `{row['gold_json']}`")
            lines.append("")
            lines.append(f"parsed: `{row['parsed_json']}`")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    if len(CASES) != 33:
        raise SystemExit(f"expected 33 gold cases, found {len(CASES)}")
    rules = load_rules(PROJECT_ROOT / "rules" / "train.json")
    calendar = load_calendar("train")
    questions = {item.id: item for item in load_questions("train")}
    references = load_references("train")
    engines, contexts, connections = _load_shops(rules, calendar)
    client = get_client()

    start_dt = datetime.datetime.now(datetime.UTC)
    run_dir = _unique_run_dir(
        settings.runs_dir,
        f"{start_dt.strftime('%Y%m%d-%H%M%S')}_parse_gold",
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    try:
        for case in CASES:
            row = _run_case(case, questions, references, engines, contexts, client, rules, calendar)
            rows.append(row)
            _print_row(row)
    finally:
        totals = _totals(rows)
        with (run_dir / "cases.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        (run_dir / "summary.md").write_text(
            _summary_markdown(rows, totals),
            encoding="utf-8",
        )
        for connection in connections:
            connection.close()
        client.close()
        print(f"summary: {run_dir / 'summary.md'}", flush=True)


def _run_case(
    case: GoldCase,
    questions: dict[str, Any],
    references: dict[str, Any],
    engines: dict[str, MetricEngine],
    contexts: dict[str, ShopContext],
    client: Any,
    rules: Any,
    calendar: LunarCalendar,
) -> dict[str, Any]:
    question = questions[case.question_id]
    ref = references[case.question_id]
    usage_before = len(client.usage_log)
    t0 = time.perf_counter()
    parsed: QuerySpec | None = None
    parse_reason: str | None = None
    attempts: list[dict[str, Any]] = []
    answer: Answer | None = None
    try:
        parsed_result = parse_question(
            question.question,
            contexts[case.business_id],
            client,
            rules,
            calendar,
            case.as_of,
        )
        parsed = parsed_result.spec
        parse_reason = parsed_result.reason
        attempts = [dataclasses.asdict(attempt) for attempt in parsed_result.attempts]
        if parsed is None:
            answer = Answer.abstain()
        else:
            answer = engines[case.business_id].answer(parsed, case.as_of)
    except Exception as exc:  # noqa: BLE001 — one bad case must not stop the run
        parse_reason = f"{type(exc).__name__}: {exc}"
        answer = Answer.abstain()
        parsed = None
    latency_s = time.perf_counter() - t0
    new_usage = client.usage_log[usage_before:]
    submitted = to_submission_line(case.question_id, answer)
    score = score_one(submitted, ref)
    label = _result_label(score)
    gold_json = _dump(case.spec)
    parsed_json = _dump(parsed)
    return {
        "n": case.n,
        "id": case.question_id,
        "result": label,
        "reason": _reason_text(parse_reason, answer, score),
        "calls": len(new_usage),
        "latency_s": latency_s,
        "identical": _specs_equivalent(case.spec, parsed, case.as_of, rules, calendar),
        "exact": gold_json == parsed_json,
        "gold_json": gold_json,
        "parsed_json": parsed_json,
        "question": question.question,
        "attempts": attempts,
    }


def _print_row(row: dict[str, Any]) -> None:
    reason = f" {row['reason']}" if row["reason"] else ""
    identical = "identical" if row["identical"] else "differs"
    print(
        f"{row['n']:02d} {row['id']} {row['result']}{reason} "
        f"calls={row['calls']} latency={row['latency_s']:.2f}s {identical}",
        flush=True,
    )
    if row["gold_json"] != row["parsed_json"]:
        print(f"  gold:   {row['gold_json']}", flush=True)
        print(f"  parsed: {row['parsed_json']}", flush=True)


def _totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    abstain_reasons: dict[str, int] = {}
    for row in rows:
        if row["result"] != "abstain":
            continue
        key = row["reason"] or "unspecified"
        abstain_reasons[key] = abstain_reasons.get(key, 0) + 1
    return {
        "correct": sum(row["result"] == "correct" for row in rows),
        "wrong": sum(row["result"] == "wrong" for row in rows),
        "abstain": sum(row["result"] == "abstain" for row in rows),
        "identical": sum(bool(row["identical"]) for row in rows),
        "exact": sum(bool(row["exact"]) for row in rows),
        "calls": sum(int(row["calls"]) for row in rows),
        "latency_s": sum(float(row["latency_s"]) for row in rows),
        "abstain_reasons": abstain_reasons,
    }


if __name__ == "__main__":
    main()

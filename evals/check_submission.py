"""Check a submission.jsonl against rulebook §9 before uploading it.

Usage:
    .venv\\Scripts\\python evals/check_submission.py --split public_test --submission runs/<run>/submission.jsonl

Exit code 1 when any error is found. A format error is scored as wrong (-1), so every
non-abstain line must match its type exactly.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datamate.io.questions import load_questions
from datamate.scoring.answer import ANSWER_TYPES

_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_MONTH = re.compile(r"^\d{4}-\d{2}$")


@dataclass
class CheckResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    types: Counter[str] = field(default_factory=Counter)

    @property
    def ok(self) -> bool:
        return not self.errors


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not valid JSON")


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _answer_error(kind: str, answer: Any) -> str | None:
    """Return why the answer does not match the §9 format of its type, or None."""
    if kind in ("number", "percent"):
        return None if _is_number(answer) else f"{kind} must be a finite JSON number"
    if kind == "date":
        if not isinstance(answer, str):
            return "date must be a string"
        try:
            if _DAY.match(answer):
                datetime.date.fromisoformat(answer)
                return None
            if _MONTH.match(answer):
                datetime.date.fromisoformat(f"{answer}-01")
                return None
        except ValueError:
            return f"date {answer!r} is not a real date"
        return "date must be YYYY-MM-DD or YYYY-MM"
    if kind == "boolean":
        return None if isinstance(answer, bool) else "boolean must be true or false"
    if kind == "string":
        return None if isinstance(answer, str) and answer.strip() else "string must be non-empty"
    if kind == "list":
        if not isinstance(answer, list):
            return "list must be a JSON array"
        for item in answer:
            if isinstance(item, bool) or not (isinstance(item, str) or _is_number(item)):
                return "list items must be finite numbers or strings"
        return None
    if kind == "table":
        if not isinstance(answer, dict) or set(answer) != {"columns", "rows"}:
            return "table must be {columns, rows}"
        columns, rows = answer["columns"], answer["rows"]
        if not isinstance(columns, list) or not all(isinstance(c, str) for c in columns):
            return "table columns must be strings"
        if not isinstance(rows, list) or not all(
            isinstance(row, list) and len(row) == len(columns) for row in rows
        ):
            return "every table row must be a list as long as columns"
        return None
    return f"unknown type {kind!r}"


def check_lines(
    lines: list[str], question_ids: list[str], allow_missing: bool = False
) -> CheckResult:
    result = CheckResult()
    expected = set(question_ids)
    seen: set[str] = set()
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        where = f"line {number}"
        try:
            row = json.loads(line, parse_constant=_reject_constant)
        except ValueError as exc:
            result.errors.append(f"{where}: invalid JSON ({exc})")
            continue
        if not isinstance(row, dict):
            result.errors.append(f"{where}: not a JSON object")
            continue
        qid, kind = row.get("id"), row.get("type")
        if not isinstance(qid, str):
            result.errors.append(f"{where}: missing id")
            continue
        where = f"{where} ({qid})"
        if qid not in expected:
            result.errors.append(f"{where}: id is not in questions.jsonl")
        if qid in seen:
            result.errors.append(f"{where}: duplicate id")
        seen.add(qid)
        if kind not in ANSWER_TYPES:
            result.errors.append(f"{where}: type {kind!r} is not one of {list(ANSWER_TYPES)}")
            continue
        result.types[kind] += 1
        if kind == "abstain":
            continue
        if kind == "unanswerable":
            reason = row.get("reason")
            if reason is not None and not isinstance(reason, str):
                result.errors.append(f"{where}: reason must be a string")
            continue
        if "answer" not in row:
            result.errors.append(f"{where}: {kind} has no answer")
            continue
        problem = _answer_error(kind, row["answer"])
        if problem:
            result.errors.append(f"{where}: {problem}")
        elif kind == "percent" and round(row["answer"], 2) != row["answer"]:
            result.warnings.append(f"{where}: percent has more than 2 decimals")
    missing = expected - seen
    if missing:
        message = f"{len(missing)} question ids have no line (e.g. {sorted(missing)[:3]})"
        # A missing line scores 0 like an abstain; for a full split it usually means a bug.
        (result.warnings if allow_missing else result.errors).append(message)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Check submission.jsonl against rulebook §9")
    parser.add_argument("--split", required=True, help="Split whose questions.jsonl to match")
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Only warn about questions without a line (for subset runs such as poc)",
    )
    args = parser.parse_args(argv)

    question_ids = [question.id for question in load_questions(args.split)]
    lines = args.submission.read_text(encoding="utf-8").splitlines()
    result = check_lines(lines, question_ids, args.allow_missing)

    answered = sum(n for kind, n in result.types.items() if kind != "abstain")
    print(f"questions: {len(question_ids)} | lines: {sum(result.types.values())}")
    print(f"answered: {answered} | abstain: {result.types.get('abstain', 0)}")
    print("by type: " + ", ".join(f"{k}={v}" for k, v in sorted(result.types.items())))
    for warning in result.warnings[:20]:
        print(f"WARNING {warning}")
    for error in result.errors[:50]:
        print(f"ERROR {error}")
    if result.errors:
        print(f"FAILED: {len(result.errors)} error(s)")
        raise SystemExit(1)
    print("OK: ready to upload")


if __name__ == "__main__":
    main()

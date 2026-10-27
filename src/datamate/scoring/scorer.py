from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from datamate.io.questions import Reference
from datamate.scoring.answer import ANSWER_TYPES

_DATE_RE = re.compile(r"^(?:\d{4}-\d{2}-\d{2}|\d{4}-\d{2})$")
_TWO_PLACES = Decimal("0.01")


@dataclass
class Result:
    scores: dict[str, int]
    N: int
    PA: float
    answered: int
    correct: int
    wrong: int
    coverage: float
    precision: float
    warnings: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return self.N

    @property
    def pa(self) -> float:
        return self.PA


def _is_valid_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _match_number(sub_val: Any, ref_val: Any) -> bool:
    if not _is_valid_number(sub_val) or not _is_valid_number(ref_val):
        return False
    if isinstance(sub_val, int) and isinstance(ref_val, int):
        return sub_val == ref_val
    sub_f = float(sub_val)
    ref_f = float(ref_val)
    return abs(sub_f - ref_f) <= 1e-9 * max(1.0, abs(ref_f))


def _round_percent(value: float) -> Decimal:
    return Decimal(repr(value)).quantize(_TWO_PLACES, rounding=ROUND_HALF_UP)


def _match_percent(sub_val: Any, ref_val: Any) -> bool:
    if not _is_valid_number(sub_val) or not _is_valid_number(ref_val):
        return False
    return _round_percent(sub_val) == _round_percent(ref_val)


def normalise_string(text: str) -> str:
    nfc = unicodedata.normalize("NFC", text).lower()
    cleaned = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in nfc)
    return " ".join(cleaned.split())


def _match_string(sub_val: Any, ref_val: Any, aliases: Iterable[Any] = ()) -> bool:
    if not isinstance(sub_val, str) or not isinstance(ref_val, str):
        return False
    norm_sub = normalise_string(sub_val)
    if norm_sub == normalise_string(ref_val):
        return True
    for alias in aliases:
        if isinstance(alias, str) and norm_sub == normalise_string(alias):
            return True
    return False


def _aligned_aliases(alias_list: list[Any], idx: int) -> list[str]:
    if idx >= len(alias_list):
        return []
    entry = alias_list[idx]
    if isinstance(entry, list):
        return [a for a in entry if isinstance(a, str)]
    if isinstance(entry, str):
        return [entry]
    return []


def _match_list(sub_val: Any, ref_val: Any, ref_aliases: list[Any]) -> bool:
    if not isinstance(sub_val, list) or not isinstance(ref_val, list):
        return False
    if len(sub_val) != len(ref_val):
        return False
    for idx, (sub_elem, ref_elem) in enumerate(zip(sub_val, ref_val, strict=True)):
        if isinstance(ref_elem, str):
            if not _match_string(sub_elem, ref_elem, _aligned_aliases(ref_aliases, idx)):
                return False
        elif _is_valid_number(ref_elem):
            if not _match_number(sub_elem, ref_elem):
                return False
        else:
            return False
    return True


def _resolve_column_mapping(sub_cols: list[str], ref_cols: list[str]) -> list[int]:
    norm_sub = [normalise_string(c) for c in sub_cols]
    norm_ref = [normalise_string(c) for c in ref_cols]
    if len(set(norm_sub)) == len(ref_cols) and set(norm_sub) == set(norm_ref):
        return [norm_sub.index(r) for r in norm_ref]
    return list(range(len(ref_cols)))


def _table_row_matches(
    sub_row: list[Any],
    ref_row: list[Any],
    row_aliases: list[str],
    col_map: list[int],
) -> bool:
    if len(ref_row) == 0:
        return len(sub_row) == 0
    first_sub = sub_row[col_map[0]]
    first_ref = ref_row[0]
    if not _match_string(first_sub, first_ref, row_aliases):
        return False
    for col_idx in range(1, len(ref_row)):
        sub_cell = sub_row[col_map[col_idx]]
        ref_cell = ref_row[col_idx]
        if not _match_number(sub_cell, ref_cell):
            return False
    return True


def _match_table(sub_val: Any, ref_val: Any, ref_aliases: list[Any]) -> bool:
    if not isinstance(sub_val, dict) or not isinstance(ref_val, dict):
        return False
    sub_cols = sub_val.get("columns")
    sub_rows = sub_val.get("rows")
    ref_cols = ref_val.get("columns")
    ref_rows = ref_val.get("rows")

    if not isinstance(sub_cols, list) or not all(isinstance(c, str) for c in sub_cols):
        return False
    if not isinstance(ref_cols, list) or not all(isinstance(c, str) for c in ref_cols):
        return False
    if not isinstance(sub_rows, list) or not isinstance(ref_rows, list):
        return False
    if len(sub_cols) != len(ref_cols) or len(sub_cols) == 0:
        return False
    if any(not isinstance(r, list) or len(r) != len(sub_cols) for r in sub_rows):
        return False
    if len(sub_rows) != len(ref_rows):
        return False

    col_map = _resolve_column_mapping(sub_cols, ref_cols)

    # Validate format of all submitted cells up front
    for sub_row in sub_rows:
        if not isinstance(sub_row[col_map[0]], str):
            return False
        for col_idx in range(1, len(ref_cols)):
            if not _is_valid_number(sub_row[col_map[col_idx]]):
                return False

    n_rows = len(ref_rows)
    adj: list[list[int]] = []
    for r_idx, ref_row in enumerate(ref_rows):
        row_aliases = _aligned_aliases(ref_aliases, r_idx)
        candidates = [
            s_idx
            for s_idx, sub_row in enumerate(sub_rows)
            if _table_row_matches(sub_row, ref_row, row_aliases, col_map)
        ]
        if not candidates:
            return False
        adj.append(candidates)

    match_to_ref: list[int] = [-1] * n_rows

    def _dfs(ref_idx: int, visited: list[bool]) -> bool:
        for sub_idx in adj[ref_idx]:
            if visited[sub_idx]:
                continue
            visited[sub_idx] = True
            if match_to_ref[sub_idx] == -1 or _dfs(match_to_ref[sub_idx], visited):
                match_to_ref[sub_idx] = ref_idx
                return True
        return False

    for r_idx in range(n_rows):
        if not _dfs(r_idx, [False] * n_rows):
            return False
    return True


def score_one(submitted: dict[str, Any] | None, ref: Reference) -> int:
    if submitted is None or not isinstance(submitted, dict):
        return 0

    sub_type = submitted.get("type")
    if not isinstance(sub_type, str) or sub_type not in ANSWER_TYPES:
        return 0

    if sub_type == "abstain":
        return 0

    if sub_type == "unanswerable" and "reason" in submitted and not isinstance(
        submitted["reason"], str
    ):
        return 0

    if sub_type != ref.type:
        return -1

    if sub_type == "unanswerable":
        return 1

    if "answer" not in submitted:
        return -1
    sub_ans = submitted["answer"]

    if sub_type == "number":
        return 1 if _match_number(sub_ans, ref.answer) else -1

    if sub_type == "percent":
        return 1 if _match_percent(sub_ans, ref.answer) else -1

    if sub_type == "date":
        if not isinstance(sub_ans, str) or not _DATE_RE.match(sub_ans):
            return -1
        return 1 if sub_ans == ref.answer else -1

    if sub_type == "boolean":
        if not isinstance(sub_ans, bool):
            return -1
        return 1 if sub_ans is ref.answer else -1

    if sub_type == "string":
        return 1 if _match_string(sub_ans, ref.answer, ref.alias) else -1

    if sub_type == "list":
        return 1 if _match_list(sub_ans, ref.answer, ref.alias) else -1

    if sub_type == "table":
        return 1 if _match_table(sub_ans, ref.answer, ref.alias) else -1

    return -1


def _iter_submission_records(
    submission_path_or_lines: Path | str | Iterable[str | dict[str, Any]],
    warnings: list[str],
) -> Iterable[dict[str, Any]]:
    if isinstance(submission_path_or_lines, (Path, str)):
        path = Path(submission_path_or_lines)
        raw_lines: Iterable[str | dict[str, Any]] = path.read_text(encoding="utf-8").splitlines()
    else:
        raw_lines = submission_path_or_lines

    for idx, item in enumerate(raw_lines, start=1):
        if isinstance(item, str):
            line = item.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                warnings.append(f"Line {idx}: invalid JSON ignored")
                continue
        elif isinstance(item, dict):
            obj = item
        else:
            warnings.append(f"Line {idx}: non-dict record ignored")
            continue

        if not isinstance(obj, dict):
            warnings.append(f"Line {idx}: JSON value is not an object")
            continue
        qid = obj.get("id")
        if not isinstance(qid, str) or not qid:
            warnings.append(f"Line {idx}: missing or non-string 'id'")
            continue
        yield obj


def score_file(
    submission_path_or_lines: Path | str | Iterable[str | dict[str, Any]],
    refs: dict[str, Reference],
    question_ids: list[str],
) -> Result:
    warnings: list[str] = []
    submitted_by_id: dict[str, dict[str, Any]] = {}

    for obj in _iter_submission_records(submission_path_or_lines, warnings):
        qid = obj["id"]
        if qid in submitted_by_id:
            warnings.append(f"Duplicate question id {qid!r}; keeping first occurrence")
            continue
        submitted_by_id[qid] = obj

    scores: dict[str, int] = {}
    for qid in question_ids:
        ref = refs[qid]
        scores[qid] = score_one(submitted_by_id.get(qid), ref)

    total_n = len(question_ids)
    correct = sum(1 for s in scores.values() if s == 1)
    wrong = sum(1 for s in scores.values() if s == -1)
    answered = correct + wrong
    total_score = sum(scores.values())

    pa = total_score / total_n if total_n > 0 else 0.0
    coverage = answered / total_n if total_n > 0 else 0.0
    precision = correct / answered if answered > 0 else 0.0

    return Result(
        scores=scores,
        N=total_n,
        PA=pa,
        answered=answered,
        correct=correct,
        wrong=wrong,
        coverage=coverage,
        precision=precision,
        warnings=warnings,
    )

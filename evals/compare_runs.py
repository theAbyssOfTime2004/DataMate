"""Compare two predict / eval run directories (submission.jsonl + optional PA).

Usage:
    .venv\\Scripts\\python evals/compare_runs.py --a runs/<run_a> --b runs/<run_b>
    .venv\\Scripts\\python evals/compare_runs.py --a ... --b ... --split train
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datamate.io.questions import load_questions, load_references
from datamate.scoring.scorer import score_file


def _find_submission(path: Path) -> Path:
    path = path.resolve()
    if path.is_file() and path.name.endswith(".jsonl"):
        return path
    candidate = path / "submission.jsonl"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"no submission.jsonl under {path}")


def _load_by_id(submission: Path) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for number, line in enumerate(submission.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise TypeError(f"{submission}:{number}: missing id")
        qid = row["id"]
        if qid in by_id:
            continue
        by_id[qid] = row
    return by_id


def _answers_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if a.get("type") != b.get("type"):
        return False
    kind = a.get("type")
    if kind in ("abstain",):
        return True
    if kind == "unanswerable":
        # Reason text is free-form; treat matching type as agreement.
        return True
    return a.get("answer") == b.get("answer")


def compare(
    path_a: Path,
    path_b: Path,
    *,
    split: str | None = None,
) -> dict[str, Any]:
    sub_a = _find_submission(path_a)
    sub_b = _find_submission(path_b)
    a = _load_by_id(sub_a)
    b = _load_by_id(sub_b)
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    shared = sorted(set(a) & set(b))

    agree = 0
    disagree_by_type: dict[str, int] = defaultdict(int)
    type_pairs: Counter[tuple[str, str]] = Counter()
    for qid in shared:
        left, right = a[qid], b[qid]
        ta = str(left.get("type"))
        tb = str(right.get("type"))
        type_pairs[(ta, tb)] += 1
        if _answers_equal(left, right):
            agree += 1
        else:
            disagree_by_type[f"{ta}->{tb}"] += 1

    agreement = agree / len(shared) if shared else 0.0
    report: dict[str, Any] = {
        "a": str(sub_a),
        "b": str(sub_b),
        "n_a": len(a),
        "n_b": len(b),
        "n_shared": len(shared),
        "n_only_a": len(only_a),
        "n_only_b": len(only_b),
        "agree": agree,
        "agreement": agreement,
        "disagree_by_type": dict(sorted(disagree_by_type.items(), key=lambda x: (-x[1], x[0]))),
        "type_pairs": {
            f"{ta}->{tb}": n
            for (ta, tb), n in sorted(type_pairs.items(), key=lambda x: (-x[1], x[0]))
        },
    }

    if split:
        try:
            refs = load_references(split)
        except FileNotFoundError:
            report["pa"] = None
            report["pa_note"] = f"no answers.jsonl for split {split!r}"
        else:
            question_ids = [q.id for q in load_questions(split)]
            # Score only ids present in both submissions and in the split.
            scored_ids = [qid for qid in question_ids if qid in a and qid in b]
            pa_a = score_file(sub_a, refs, scored_ids)
            pa_b = score_file(sub_b, refs, scored_ids)
            report["pa"] = {
                "split": split,
                "n_scored": len(scored_ids),
                "a": {
                    "PA": pa_a.PA,
                    "correct": pa_a.correct,
                    "wrong": pa_a.wrong,
                    "answered": pa_a.answered,
                },
                "b": {
                    "PA": pa_b.PA,
                    "correct": pa_b.correct,
                    "wrong": pa_b.wrong,
                    "answered": pa_b.answered,
                },
                "pa_diff": pa_a.PA - pa_b.PA,
            }

    return report


def _print_report(report: dict[str, Any]) -> None:
    print(f"A: {report['a']} ({report['n_a']} lines)")
    print(f"B: {report['b']} ({report['n_b']} lines)")
    print(
        f"shared={report['n_shared']} only_a={report['n_only_a']} only_b={report['n_only_b']}"
    )
    pct = 100.0 * report["agreement"]
    print(f"answer agreement: {report['agree']}/{report['n_shared']} = {pct:.2f}%")
    if report["disagree_by_type"]:
        print("disagree by type:")
        for key, n in list(report["disagree_by_type"].items())[:20]:
            print(f"  {key}: {n}")
    pa = report.get("pa")
    if isinstance(pa, dict):
        print(
            f"PA on {pa['split']} (n={pa['n_scored']}): "
            f"A={pa['a']['PA']:.4f} B={pa['b']['PA']:.4f} diff={pa['pa_diff']:+.4f}"
        )
    elif report.get("pa_note"):
        print(f"PA: {report['pa_note']}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare two DataMate run directories")
    parser.add_argument("--a", type=Path, required=True, help="Run dir or submission.jsonl")
    parser.add_argument("--b", type=Path, required=True, help="Run dir or submission.jsonl")
    parser.add_argument(
        "--split",
        default=None,
        help="If set and answers.jsonl exists, also report PA for each run",
    )
    args = parser.parse_args(argv)
    report = compare(args.a, args.b, split=args.split)
    _print_report(report)


if __name__ == "__main__":
    main()

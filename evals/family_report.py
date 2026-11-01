"""Per-extension-family precision from a run's logs vs train answers.jsonl.

Usage:
  .venv\\Scripts\\python evals/family_report.py --split train --subset dev
  .venv\\Scripts\\python evals/family_report.py --split train --subset dev --run runs/<dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datamate.config import settings
from datamate.io.questions import load_questions, load_references
from datamate.scoring.scorer import score_one
from evals.subsets import select_questions


def _latest_run(runs_dir: Path) -> Path | None:
    if not runs_dir.is_dir():
        return None
    candidates = sorted(
        (p for p in runs_dir.iterdir() if p.is_dir() and (p / "logs").is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _load_log(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--subset",
        choices=("dev", "val", "holdout_family", "all"),
        default="dev",
    )
    parser.add_argument("--run", default="", help="Run directory (default: latest under runs/)")
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Also print plan:<entity>/<measure> breakdown for E3 plan answers",
    )
    args = parser.parse_args(argv)

    run_dir = Path(args.run) if args.run.strip() else _latest_run(settings.runs_dir)
    if run_dir is None or not (run_dir / "logs").is_dir():
        print("No run directory with logs/ found", file=sys.stderr)
        return 1

    questions = load_questions(args.split)
    if args.subset != "all":
        questions = select_questions(questions, args.subset)
    by_id = {q.id: q for q in questions}
    refs = load_references(args.split)

    # family -> counters
    stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "n": 0,
            "answered": 0,
            "correct": 0,
            "wrong": 0,
            "abstain": 0,
            "verify_abstain": 0,
        }
    )

    logs_dir = run_dir / "logs"
    for log_path in sorted(logs_dir.glob("*.json")):
        payload = _load_log(log_path)
        qid = payload.get("id") or log_path.stem
        if qid not in by_id:
            continue
        trace = payload.get("trace") or {}
        submitted = payload.get("final_answer") or {}
        ref = refs.get(qid)
        if ref is None:
            continue
        family = trace.get("ext_family")
        bucket = "main_path" if not family else str(family)
        buckets = [bucket]
        if bucket == "plan" and args.detail:
            plan = trace.get("plan") or {}
            entity = plan.get("entity") or "?"
            measure = plan.get("measure") or "?"
            buckets.append(f"plan:{entity}/{measure}")
        ans_type = submitted.get("type")
        is_abstain = ans_type in (None, "abstain")
        points = 0 if is_abstain else score_one(submitted, ref)
        verify_low = trace.get("abstain_reason") in {"ext_verify_low", "plan_verify_low"}
        for name in buckets:
            row = stats[name]
            row["n"] += 1
            if is_abstain:
                row["abstain"] += 1
                if verify_low:
                    row["verify_abstain"] += 1
                continue
            row["answered"] += 1
            if points > 0:
                row["correct"] += 1
            else:
                row["wrong"] += 1

    print(f"run: {run_dir}")
    print(f"split={args.split} subset={args.subset}")
    print()
    header = (
        f"{'family':<28} {'n':>5} {'ans':>5} {'ok':>5} {'wrong':>5} {'abs':>5} "
        f"{'v_abs':>5} {'prec':>7}"
    )
    print(header)
    print("-" * len(header))
    order = sorted(stats.keys(), key=lambda k: (k != "main_path", k))
    for name in order:
        row = stats[name]
        answered = row["answered"]
        prec = (row["correct"] / answered) if answered else float("nan")
        prec_s = f"{prec:.3f}" if answered else "—"
        print(
            f"{name:<28} {row['n']:>5} {answered:>5} {row['correct']:>5} "
            f"{row['wrong']:>5} {row['abstain']:>5} {row['verify_abstain']:>5} {prec_s:>7}"
        )
    return 0



if __name__ == "__main__":
    raise SystemExit(main())

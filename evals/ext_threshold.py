"""Pick settings.ext_verify_threshold from a train/dev run's ext_verify_p logs.

Run eval with a low threshold (e.g. 0) so every verified question still answers; this
script re-filters by p_yes. Never choose the threshold on public.

Usage:
  .venv\\Scripts\\python evals/ext_threshold.py --split train --subset dev
  .venv\\Scripts\\python evals/ext_threshold.py --split train --subset dev --run runs/<dir>
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

_THRESHOLDS = [round(x * 0.05, 2) for x in range(1, 20)]  # 0.05 .. 0.95


def _latest_run(runs_dir: Path) -> Path | None:
    if not runs_dir.is_dir():
        return None
    candidates = sorted(
        (p for p in runs_dir.iterdir() if p.is_dir() and (p / "logs").is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--subset",
        choices=("dev", "val", "holdout_family", "all"),
        default="dev",
    )
    parser.add_argument("--run", default="", help="Run directory (default: latest under runs/)")
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

    # Prefer an answer computed for threshold analysis (logged even when verify abstains).
    rows: list[dict[str, Any]] = []
    for log_path in sorted((run_dir / "logs").glob("*.json")):
        payload = json.loads(log_path.read_text(encoding="utf-8"))
        qid = payload.get("id") or log_path.stem
        if qid not in by_id:
            continue
        trace = payload.get("trace") or {}
        if "ext_verify_p" not in trace:
            continue
        ref = refs.get(qid)
        if ref is None:
            continue
        scored = (
            trace.get("ext_answer")
            or payload.get("ext_answer")
            or payload.get("final_answer")
            or {}
        )
        if scored.get("type") in (None, "abstain"):
            continue
        rows.append(
            {
                "family": str(trace.get("ext_family") or "unknown"),
                "p": float(trace["ext_verify_p"]),
                "ref": ref,
                "answer": scored,
            }
        )

    if not rows:
        print(
            f"No scored verified ext questions in {run_dir}. "
            "Re-run with ext_verify_threshold=0 (or inspect logs).",
            file=sys.stderr,
        )
        return 1

    print(f"run: {run_dir}")
    print(f"split={args.split} subset={args.subset} n_scored={len(rows)}")
    print()
    header = f"{'thr':>5} {'ans':>5} {'ok':>5} {'wrong':>5} {'net':>5}"
    print(header)
    print("-" * len(header))

    best_thr = 0.5
    best_net: int | None = None
    zero_wrong_thr: float | None = None
    for thr in _THRESHOLDS:
        answered = correct = wrong = 0
        by_fam: dict[str, dict[str, int]] = defaultdict(
            lambda: {"answered": 0, "correct": 0, "wrong": 0}
        )
        for row in rows:
            if row["p"] < thr:
                continue
            answered += 1
            fam = by_fam[row["family"]]
            fam["answered"] += 1
            if score_one(row["answer"], row["ref"]) > 0:
                correct += 1
                fam["correct"] += 1
            else:
                wrong += 1
                fam["wrong"] += 1
        net = correct - wrong
        print(f"{thr:>5.2f} {answered:>5} {correct:>5} {wrong:>5} {net:>5}")
        for name in sorted(by_fam):
            f = by_fam[name]
            print(
                f"      {name:<22} ans={f['answered']:>3} ok={f['correct']:>3} "
                f"wrong={f['wrong']:>3}"
            )
        if wrong == 0 and answered > 0 and (zero_wrong_thr is None or thr < zero_wrong_thr):
            zero_wrong_thr = thr
        if best_net is None or net > best_net or (net == best_net and thr < best_thr):
            best_net = net
            best_thr = thr

    # Prefer the smallest thr with zero wrong when that net is at least as good as max net.
    # Prefer the smallest thr with zero wrong when its net (all correct) is maximal;
    # otherwise keep the thr with maximum net points.
    pick = best_thr
    if zero_wrong_thr is not None:
        zw_net = sum(1 for row in rows if row["p"] >= zero_wrong_thr)
        if zw_net >= (best_net or 0):
            pick = zero_wrong_thr
    print()
    print(
        f"suggested ext_verify_threshold={pick} "
        f"(max net thr={best_thr} net={best_net}; "
        f"smallest thr with wrong=0: {zero_wrong_thr})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

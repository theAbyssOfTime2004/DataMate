from __future__ import annotations

import argparse
import datetime
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datamate.config import settings
from datamate.io.paths import business_dir
from datamate.io.questions import Question, Reference, load_questions, load_references
from datamate.scoring.answer import Answer, to_submission_line
from datamate.scoring.scorer import score_file
from datamate.systems.abstain import AlwaysAbstain
from datamate.systems.base import System
from evals.subsets import select_questions, tag_topics

SYSTEM_REGISTRY: dict[str, type[Any]] = {
    "abstain": AlwaysAbstain,
}


def _git_commit_hash() -> str | None:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return res.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _settings_without_api_key() -> dict[str, Any]:
    raw = settings.model_dump(mode="json")
    raw.pop("llm_api_key", None)
    return raw


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


def _compute_timing(elapsed_list: list[float], total_wall_s: float) -> dict[str, float]:
    if not elapsed_list:
        return {
            "total_wall_s": round(total_wall_s, 6),
            "mean_s": 0.0,
            "p95_s": 0.0,
        }
    sorted_t = sorted(elapsed_list)
    mean_s = sum(sorted_t) / len(sorted_t)
    idx_95 = min(len(sorted_t) - 1, max(0, math.ceil(0.95 * len(sorted_t)) - 1))
    p95_s = sorted_t[idx_95]
    return {
        "total_wall_s": round(total_wall_s, 6),
        "mean_s": round(mean_s, 6),
        "p95_s": round(p95_s, 6),
    }


def _compute_breakdown(
    ids: list[str],
    scores: dict[str, int],
) -> dict[str, Any]:
    n = len(ids)
    correct = sum(1 for qid in ids if scores[qid] == 1)
    wrong = sum(1 for qid in ids if scores[qid] == -1)
    answered = correct + wrong
    total = sum(scores[qid] for qid in ids)
    pa = total / n if n > 0 else 0.0
    coverage = answered / n if n > 0 else 0.0
    precision = correct / answered if answered > 0 else 0.0
    return {
        "N": n,
        "PA": pa,
        "answered": answered,
        "correct": correct,
        "wrong": wrong,
        "coverage": coverage,
        "precision": precision,
    }


def _render_markdown_table(title: str, group_col: str, rows: dict[str, dict[str, Any]]) -> str:
    lines = [
        f"## {title}",
        "",
        f"| {group_col} | N | Answered | Correct | Wrong | Coverage | Precision | PA |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(rows.keys()):
        m = rows[key]
        lines.append(
            f"| {key} | {m['N']} | {m['answered']} | {m['correct']} | {m['wrong']} "
            f"| {m['coverage']:.2%} | {m['precision']:.2%} | {m['PA']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def run_evaluation(
    split: str = "train",
    subset: str = "poc",
    system_name: str = "abstain",
    limit: int | None = None,
    runs_dir: Path | None = None,
    system_instance: System | None = None,
) -> Path:
    if split != "train" and subset != "all":
        raise ValueError(
            f"Only subset 'all' is valid for non-train split {split!r} (got subset={subset!r})"
        )

    if system_instance is None:
        if system_name not in SYSTEM_REGISTRY:
            raise ValueError(
                f"Unknown system {system_name!r}; available: {sorted(SYSTEM_REGISTRY.keys())}"
            )
        system: System = SYSTEM_REGISTRY[system_name]()
    else:
        system = system_instance

    all_questions = load_questions(split)
    selected: list[Question] = select_questions(all_questions, subset)
    if limit is not None:
        selected = selected[:limit]

    try:
        refs: dict[str, Reference] | None = load_references(split)
    except FileNotFoundError:
        refs = None

    base_runs_dir = runs_dir if runs_dir is not None else settings.runs_dir
    base_runs_dir.mkdir(parents=True, exist_ok=True)

    start_dt = datetime.datetime.now(datetime.UTC)
    wall_t0 = time.perf_counter()
    timestamp = start_dt.strftime("%Y%m%d-%H%M%S")
    run_dir = _unique_run_dir(base_runs_dir, f"{timestamp}_{system.name}_{subset}")
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    submission_path = run_dir / "submission.jsonl"

    topics_by_qid: dict[str, set[str]] = {}
    elapsed_per_question: list[float] = []
    submitted_answers: list[Answer] = []

    with submission_path.open("w", encoding="utf-8") as sub_fh:
        for q in selected:
            topics_by_qid[q.id] = tag_topics(q.question)
            biz_workdir = business_dir(split, q.business_id)
            t0 = time.perf_counter()
            ans, trace = system.answer(q, biz_workdir)
            elapsed_s = time.perf_counter() - t0
            elapsed_per_question.append(elapsed_s)
            submitted_answers.append(ans)

            sub_line = to_submission_line(q.id, ans)
            sub_fh.write(json.dumps(sub_line, ensure_ascii=False) + "\n")
            sub_fh.flush()

            log_payload = {
                "id": q.id,
                "business_id": q.business_id,
                "question": q.question,
                "as_of_date": q.as_of_date.isoformat(),
                "final_answer": sub_line,
                "elapsed_s": round(elapsed_s, 6),
                "trace": trace,
            }
            log_file = logs_dir / f"{q.id}.json"
            log_file.write_text(
                json.dumps(log_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    total_wall_s = time.perf_counter() - wall_t0
    end_dt = datetime.datetime.now(datetime.UTC)
    timing = _compute_timing(elapsed_per_question, total_wall_s)

    metadata = {
        "git_commit": _git_commit_hash(),
        "system": system.name,
        "split": split,
        "subset": subset,
        "limit": limit,
        "seed": settings.seed,
        "settings": _settings_without_api_key(),
        "start_time": start_dt.isoformat(),
        "end_time": end_dt.isoformat(),
    }

    if refs is None:
        total_n = len(selected)
        answered = sum(1 for a in submitted_answers if a.type != "abstain")
        coverage = answered / total_n if total_n > 0 else 0.0
        metrics_payload: dict[str, Any] = {
            "N": total_n,
            "answered": answered,
            "coverage": coverage,
            "timing": timing,
            "metadata": metadata,
        }
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary_lines = [
            f"# Evaluation Summary — `{system.name}` on `{split}/{subset}` (unscored split)",
            "",
            "- **Status**: unscored split (no `answers.jsonl`)",
            f"- **N**: {total_n}",
            f"- **Coverage**: {coverage:.2%} ({answered}/{total_n})",
            (
                f"- **Timing**: total `{timing['total_wall_s']:.3f}s`, "
                f"mean `{timing['mean_s']:.4f}s/q`, p95 `{timing['p95_s']:.4f}s/q`"
            ),
            f"- **Git commit**: `{metadata['git_commit']}`",
            f"- **Seed**: `{metadata['seed']}`",
            f"- **Start / End (UTC)**: `{metadata['start_time']}` → `{metadata['end_time']}`",
            "",
        ]
        (run_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
        print(
            f"Run saved to {run_dir} (unscored split)\n"
            f"N={total_n} | coverage={coverage:.2%} ({answered}/{total_n}) | "
            f"wall={timing['total_wall_s']:.3f}s"
        )
        return run_dir

    question_ids = [q.id for q in selected]
    score_res = score_file(submission_path, refs, question_ids)

    by_type_ids: dict[str, list[str]] = defaultdict(list)
    by_biz_ids: dict[str, list[str]] = defaultdict(list)
    by_topic_ids: dict[str, list[str]] = defaultdict(list)

    for q in selected:
        by_type_ids[refs[q.id].type].append(q.id)
        by_biz_ids[q.business_id].append(q.id)
        q_topics = topics_by_qid[q.id]
        if q_topics:
            for t in q_topics:
                by_topic_ids[t].append(q.id)
        else:
            by_topic_ids["other"].append(q.id)

    by_type = {k: _compute_breakdown(v, score_res.scores) for k, v in by_type_ids.items()}
    by_business = {k: _compute_breakdown(v, score_res.scores) for k, v in by_biz_ids.items()}
    by_topic = {k: _compute_breakdown(v, score_res.scores) for k, v in by_topic_ids.items()}

    metrics_payload = {
        "N": score_res.N,
        "PA": score_res.PA,
        "answered": score_res.answered,
        "correct": score_res.correct,
        "wrong": score_res.wrong,
        "coverage": score_res.coverage,
        "precision": score_res.precision,
        "timing": timing,
        "warnings": score_res.warnings,
        "by_type": by_type,
        "by_business": by_business,
        "by_topic": by_topic,
        "metadata": metadata,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_lines = [
        f"# Evaluation Summary — `{system.name}` on `{split}/{subset}`",
        "",
        f"- **N**: {score_res.N}",
        f"- **PA**: {score_res.PA:.4f}",
        f"- **Coverage**: {score_res.coverage:.2%} ({score_res.answered}/{score_res.N})",
        f"- **Precision**: {score_res.precision:.2%} ({score_res.correct}/{score_res.answered})",
        f"- **Correct / Wrong**: {score_res.correct} / {score_res.wrong}",
        (
            f"- **Timing**: total `{timing['total_wall_s']:.3f}s`, "
            f"mean `{timing['mean_s']:.4f}s/q`, p95 `{timing['p95_s']:.4f}s/q`"
        ),
        f"- **Git commit**: `{metadata['git_commit']}`",
        f"- **Seed**: `{metadata['seed']}`",
        f"- **Start / End (UTC)**: `{metadata['start_time']}` → `{metadata['end_time']}`",
        "",
        _render_markdown_table("By Reference Type", "Type", by_type),
        _render_markdown_table("By Business", "Business", by_business),
        _render_markdown_table("By Topic", "Topic", by_topic),
    ]
    (run_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print(
        f"Run saved to {run_dir}\n"
        f"N={score_res.N} | PA={score_res.PA:.4f} | "
        f"coverage={score_res.coverage:.2%} | precision={score_res.precision:.2%} "
        f"(correct={score_res.correct}, wrong={score_res.wrong}) | "
        f"wall={timing['total_wall_s']:.3f}s"
    )
    return run_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run DataMate evaluation.")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument(
        "--subset",
        default=None,
        choices=["poc", "val", "holdout_family", "dev", "all"],
        help="Business/question subset to evaluate (default: poc for train, all otherwise)",
    )
    parser.add_argument(
        "--system",
        default="abstain",
        choices=sorted(SYSTEM_REGISTRY.keys()),
        help="System name from registry",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max number of questions")
    args = parser.parse_args(argv)
    subset = args.subset if args.subset is not None else ("poc" if args.split == "train" else "all")
    run_evaluation(
        split=args.split,
        subset=subset,
        system_name=args.system,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()

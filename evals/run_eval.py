from __future__ import annotations

import argparse
import datetime
import json
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
from datamate.io.questions import Question, Reference, load_questions, load_references
from datamate.scoring.answer import to_submission_line
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

    refs: dict[str, Reference] = load_references(split)

    base_runs_dir = runs_dir if runs_dir is not None else settings.runs_dir
    base_runs_dir.mkdir(parents=True, exist_ok=True)

    start_dt = datetime.datetime.now(datetime.UTC)
    timestamp = start_dt.strftime("%Y%m%d-%H%M%S")
    run_dir = base_runs_dir / f"{timestamp}_{system.name}_{subset}"
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    submission_path = run_dir / "submission.jsonl"
    split_path = settings.split_dir(split)

    topics_by_qid: dict[str, set[str]] = {}
    with submission_path.open("w", encoding="utf-8") as sub_fh:
        for q in selected:
            topics_by_qid[q.id] = tag_topics(q.question)
            biz_workdir = split_path / "businesses" / q.business_id
            t0 = time.perf_counter()
            ans, trace = system.answer(q, biz_workdir)
            elapsed_s = time.perf_counter() - t0

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

    end_dt = datetime.datetime.now(datetime.UTC)
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

    metrics_payload = {
        "N": score_res.N,
        "PA": score_res.PA,
        "answered": score_res.answered,
        "correct": score_res.correct,
        "wrong": score_res.wrong,
        "coverage": score_res.coverage,
        "precision": score_res.precision,
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
        f"(correct={score_res.correct}, wrong={score_res.wrong})"
    )
    return run_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run DataMate evaluation.")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument(
        "--subset",
        default="poc",
        choices=["poc", "val", "holdout_family", "dev", "all"],
        help="Business/question subset to evaluate",
    )
    parser.add_argument(
        "--system",
        default="abstain",
        choices=sorted(SYSTEM_REGISTRY.keys()),
        help="System name from registry",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max number of questions")
    args = parser.parse_args(argv)
    run_evaluation(
        split=args.split,
        subset=args.subset,
        system_name=args.system,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()

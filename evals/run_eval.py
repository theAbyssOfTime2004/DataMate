from __future__ import annotations

import argparse
import datetime
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
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
from datamate.systems.pipeline import LLM_UNAVAILABLE
from evals.subsets import select_questions, tag_topics
from evals.systems_gold import make_pipeline_gold, make_pipeline_llm

# Stop instead of scoring a run whose abstains come from a dead or starved LLM server.
MAX_CONSECUTIVE_LLM_UNAVAILABLE = 3

SYSTEM_REGISTRY: dict[str, Any] = {
    "abstain": AlwaysAbstain,
    "pipeline_gold": make_pipeline_gold,
    "pipeline_llm": make_pipeline_llm,
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


def _render_abstain_reasons(reasons: dict[str, int]) -> str:
    lines = [
        "## Abstain reasons",
        "",
        "| reason | n |",
        "|---|---:|",
    ]
    if not reasons:
        lines.append("| — | 0 |")
    else:
        for reason, count in sorted(reasons.items(), key=lambda item: (-item[1], item[0])):
            safe = reason.replace("|", "/")
            lines.append(f"| {safe} | {count} |")
    lines.append("")
    return "\n".join(lines)


def run_evaluation(
    split: str = "train",
    subset: str = "poc",
    system_name: str = "abstain",
    limit: int | None = None,
    runs_dir: Path | None = None,
    system_instance: System | None = None,
    workers: int = 1,
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

    topics_by_qid: dict[str, set[str]] = {q.id: tag_topics(q.question) for q in selected}
    results: dict[str, tuple[Answer, float]] = {}
    abstain_reasons: dict[str, int] = defaultdict(int)
    unavailable_streak = 0
    aborted: str | None = None

    with closing(_answer_stream(system, selected, split, workers)) as stream:
        for done, (q, ans, trace, elapsed_s) in enumerate(stream, start=1):
            results[q.id] = (ans, elapsed_s)
            reason: str | None = None
            if ans.type == "abstain":
                reason = "unspecified"
                if isinstance(trace, dict):
                    raw_reason = trace.get("abstain_reason")
                    if isinstance(raw_reason, str) and raw_reason:
                        reason = raw_reason
                abstain_reasons[reason] += 1
            unavailable_streak = unavailable_streak + 1 if reason == LLM_UNAVAILABLE else 0

            log_payload = {
                "id": q.id,
                "business_id": q.business_id,
                "question": q.question,
                "as_of_date": q.as_of_date.isoformat(),
                "final_answer": to_submission_line(q.id, ans),
                "elapsed_s": round(elapsed_s, 6),
                "trace": trace,
            }
            log_file = logs_dir / f"{q.id}.json"
            log_file.write_text(
                json.dumps(log_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(
                f"[{done}/{len(selected)}] {q.id} {ans.type} {elapsed_s:.1f}s",
                flush=True,
            )
            if unavailable_streak >= MAX_CONSECUTIVE_LLM_UNAVAILABLE:
                aborted = (
                    f"Aborted at {done}/{len(selected)}: the LLM server was unavailable for "
                    f"{unavailable_streak} questions in a row. Logs so far are in {run_dir}. "
                    "Fix the server (or free the GPU) and rerun; cached LLM answers are reused."
                )
                break
    if aborted is not None:
        _close_system(system)
        raise SystemExit(aborted)

    # Submission and timings follow question order, whatever order the workers finished in.
    submitted_answers = [results[q.id][0] for q in selected]
    elapsed_per_question = [results[q.id][1] for q in selected]
    with submission_path.open("w", encoding="utf-8") as sub_fh:
        for q, ans in zip(selected, submitted_answers, strict=True):
            sub_fh.write(json.dumps(to_submission_line(q.id, ans), ensure_ascii=False) + "\n")

    total_wall_s = time.perf_counter() - wall_t0
    end_dt = datetime.datetime.now(datetime.UTC)
    timing = _compute_timing(elapsed_per_question, total_wall_s)

    metadata = {
        "git_commit": _git_commit_hash(),
        "system": system.name,
        "split": split,
        "subset": subset,
        "limit": limit,
        "workers": workers,
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
            "abstain_reasons": dict(sorted(abstain_reasons.items())),
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
            *_invalid_run_warning(abstain_reasons),
            _render_abstain_reasons(dict(abstain_reasons)),
        ]
        (run_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
        _close_system(system)
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
        "abstain_reasons": dict(sorted(abstain_reasons.items())),
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
        *_invalid_run_warning(abstain_reasons),
        _render_abstain_reasons(dict(abstain_reasons)),
        _render_markdown_table("By Reference Type", "Type", by_type),
        _render_markdown_table("By Business", "Business", by_business),
        _render_markdown_table("By Topic", "Topic", by_topic),
    ]
    (run_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    _close_system(system)

    print(
        f"Run saved to {run_dir}\n"
        f"N={score_res.N} | PA={score_res.PA:.4f} | "
        f"coverage={score_res.coverage:.2%} | precision={score_res.precision:.2%} "
        f"(correct={score_res.correct}, wrong={score_res.wrong}) | "
        f"wall={timing['total_wall_s']:.3f}s"
    )
    return run_dir


def _answer_stream(
    system: System, questions: list[Question], split: str, workers: int
) -> Iterator[tuple[Question, Answer, Any, float]]:
    """Answer every question, yielding results as they finish.

    workers > 1 sends that many questions at once so vLLM can batch them. Results may then
    differ slightly from a sequential run (batched kernels), and per-question time includes
    queueing, so measure submission latency with workers=1.
    """

    def one(question: Question) -> tuple[Question, Answer, Any, float]:
        t0 = time.perf_counter()
        ans, trace = system.answer(question, business_dir(split, question.business_id))
        return question, ans, trace, time.perf_counter() - t0

    if workers <= 1:
        for question in questions:
            yield one(question)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, question) for question in questions]
        try:
            for future in as_completed(futures):
                yield future.result()
        finally:
            for future in futures:
                future.cancel()


def _invalid_run_warning(abstain_reasons: dict[str, int]) -> list[str]:
    count = abstain_reasons.get(LLM_UNAVAILABLE, 0)
    if count == 0:
        return []
    return [
        (
            f"> **INVALID RUN**: {count} question(s) abstained because the LLM server was "
            "unavailable. These are not system decisions; rerun with a healthy server "
            "(cached answers make it fast) before using PA."
        ),
        "",
    ]


def _close_system(system: System) -> None:
    close = getattr(system, "close", None)
    if callable(close):
        close()


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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Questions answered at once (default 1). Use >1 only with a GPU that has KV "
        "cache to spare, e.g. a cloud vLLM; results may differ slightly from workers=1.",
    )
    args = parser.parse_args(argv)
    subset = args.subset if args.subset is not None else ("poc" if args.split == "train" else "all")
    run_evaluation(
        split=args.split,
        subset=subset,
        system_name=args.system,
        limit=args.limit,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()

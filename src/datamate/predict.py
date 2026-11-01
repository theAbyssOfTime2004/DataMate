"""Offline predict entrypoint: answer every question in a split directory.

Usage:
    python -m datamate.predict --data-dir /data --out-dir /result [--workers 8]
        [--time-budget-s 18900]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, TextIO

from datamate.config import apply_runtime_paths, settings
from datamate.io.paths import business_dir
from datamate.io.questions import Question, load_questions
from datamate.scoring.answer import Answer, to_submission_line
from datamate.systems.pipeline import LLM_UNAVAILABLE, PipelineSystem

# Default: 5h15m so a few minutes remain for vLLM shutdown / rewrite / check.
DEFAULT_TIME_BUDGET_S = 18900
DEFAULT_WORKERS = 8


def _append_submission_line(fh: TextIO, question_id: str, answer: Answer) -> None:
    line = json.dumps(to_submission_line(question_id, answer), ensure_ascii=False)
    fh.write(line + "\n")
    fh.flush()
    try:
        os.fsync(fh.fileno())
    except OSError:
        pass


def _write_log(
    logs_dir: Path,
    question: Question,
    answer: Answer,
    trace: dict[str, Any],
    elapsed_s: float,
) -> None:
    payload = {
        "id": question.id,
        "business_id": question.business_id,
        "question": question.question,
        "as_of_date": question.as_of_date.isoformat(),
        "final_answer": to_submission_line(question.id, answer),
        "elapsed_s": round(elapsed_s, 6),
        "trace": trace,
    }
    (logs_dir / f"{question.id}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _rewrite_submission_ordered(
    path: Path, question_ids: list[str], by_id: dict[str, Answer]
) -> None:
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for qid in question_ids:
            answer = by_id.get(qid, Answer.abstain())
            fh.write(json.dumps(to_submission_line(qid, answer), ensure_ascii=False) + "\n")
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass
    tmp.replace(path)


def _answer_one(
    system: PipelineSystem, question: Question, split: str
) -> tuple[Question, Answer, dict[str, Any], float]:
    t0 = time.perf_counter()
    answer, trace = system.answer(question, business_dir(split, question.business_id))
    return question, answer, trace, time.perf_counter() - t0


def run_predict(
    data_dir: Path,
    out_dir: Path,
    *,
    workers: int = DEFAULT_WORKERS,
    time_budget_s: float = DEFAULT_TIME_BUDGET_S,
    client: Any | None = None,
) -> Path:
    """Answer every question under data_dir; write submission.jsonl + logs under out_dir."""
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Offline: never write under data_dir; caches live in out_dir/work.
    split = apply_runtime_paths(data_dir=data_dir, cache_dir=work_dir / "cache")
    # LLM defaults for a local vLLM (Docker predict.sh overrides via env before import).
    if os.environ.get("LLM_BASE_URL") is None:
        settings.llm_base_url = "http://127.0.0.1:8000/v1"
    if os.environ.get("LLM_MODEL"):
        settings.llm_model = os.environ["LLM_MODEL"]

    questions = load_questions(split)
    question_ids = [q.id for q in questions]
    submission_path = out_dir / "submission.jsonl"

    system = PipelineSystem(name="pipeline_llm", client=client)
    deadline = time.perf_counter() + time_budget_s
    wall_t0 = time.perf_counter()
    answers: dict[str, Answer] = {}
    elapsed_list: list[float] = []
    timed_out = 0

    try:
        with submission_path.open("w", encoding="utf-8") as sub_fh:
            if workers <= 1:
                for question in questions:
                    if time.perf_counter() >= deadline:
                        answers[question.id] = Answer.abstain()
                        _append_submission_line(sub_fh, question.id, answers[question.id])
                        _write_log(
                            logs_dir,
                            question,
                            answers[question.id],
                            {"abstain_reason": "time_budget"},
                            0.0,
                        )
                        timed_out += 1
                        continue
                    question, answer, trace, elapsed_s = _answer_one(system, question, split)
                    answers[question.id] = answer
                    elapsed_list.append(elapsed_s)
                    _append_submission_line(sub_fh, question.id, answer)
                    _write_log(logs_dir, question, answer, trace, elapsed_s)
            else:
                timed_out = _run_pool(
                    system=system,
                    split=split,
                    questions=questions,
                    deadline=deadline,
                    workers=workers,
                    sub_fh=sub_fh,
                    logs_dir=logs_dir,
                    answers=answers,
                    elapsed_list=elapsed_list,
                )
                # One retry pass for questions lost to an unavailable LLM server (a
                # transient infrastructure failure, not the system's judgement).
                retry = [q for q in questions if _lost_to_unavailable(logs_dir, q.id)]
                if retry and time.perf_counter() < deadline:
                    print(f"retrying {len(retry)} questions lost to llm_unavailable", flush=True)
                    timed_out += _run_pool(
                        system=system,
                        split=split,
                        questions=retry,
                        deadline=deadline,
                        workers=workers,
                        sub_fh=sub_fh,
                        logs_dir=logs_dir,
                        answers=answers,
                        elapsed_list=elapsed_list,
                    )
    finally:
        close = getattr(system, "close", None)
        if callable(close):
            close()

    # Every question gets exactly one line, in questions.jsonl order.
    for qid in question_ids:
        if qid not in answers:
            answers[qid] = Answer.abstain()
            timed_out += 1
    _rewrite_submission_ordered(submission_path, question_ids, answers)

    total_wall_s = time.perf_counter() - wall_t0
    timing = {
        "total_wall_s": round(total_wall_s, 6),
        "time_budget_s": time_budget_s,
        "workers": workers,
        "n_questions": len(questions),
        "n_answered": sum(1 for a in answers.values() if a.type != "abstain"),
        "n_timed_out": timed_out,
        "mean_s": round(sum(elapsed_list) / len(elapsed_list), 6) if elapsed_list else 0.0,
        "per_question_s": [round(x, 6) for x in elapsed_list],
    }
    (out_dir / "timing.json").write_text(
        json.dumps(timing, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"predict done: {len(questions)} questions, timed_out={timed_out}, "
        f"wall={total_wall_s:.1f}s -> {submission_path}",
        flush=True,
    )
    return submission_path


def _lost_to_unavailable(logs_dir: Path, question_id: str) -> bool:
    path = logs_dir / f"{question_id}.json"
    try:
        trace = json.loads(path.read_text(encoding="utf-8")).get("trace") or {}
    except (OSError, ValueError):
        return False
    return trace.get("abstain_reason") == LLM_UNAVAILABLE


def _run_pool(
    *,
    system: PipelineSystem,
    split: str,
    questions: list[Question],
    deadline: float,
    workers: int,
    sub_fh: TextIO,
    logs_dir: Path,
    answers: dict[str, Answer],
    elapsed_list: list[float],
) -> int:
    """Start at most `workers` questions at once; do not start after the deadline."""
    timed_out = 0
    pending = list(questions)
    in_flight: set[Any] = set()
    started: dict[Any, Question] = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:

        def _submit() -> bool:
            if not pending or time.perf_counter() >= deadline:
                return False
            question = pending.pop(0)
            future = pool.submit(_answer_one, system, question, split)
            started[future] = question
            in_flight.add(future)
            return True

        for _ in range(workers):
            if not _submit():
                break
        while in_flight:
            done, still = wait(in_flight, return_when=FIRST_COMPLETED)
            in_flight = set(still)
            for future in done:
                try:
                    question, answer, trace, elapsed_s = future.result()
                except Exception as exc:  # noqa: BLE001 — one question must not end the run
                    question = started[future]
                    answer, elapsed_s = Answer.abstain(), 0.0
                    trace = {"abstain_reason": f"{type(exc).__name__}: {exc}"}
                answers[question.id] = answer
                elapsed_list.append(elapsed_s)
                _append_submission_line(sub_fh, question.id, answer)
                _write_log(logs_dir, question, answer, trace, elapsed_s)
                _submit()
        # Not started before the budget ran out.
        for question in pending:
            answers[question.id] = Answer.abstain()
            _append_submission_line(sub_fh, question.id, answers[question.id])
            _write_log(
                logs_dir,
                question,
                answers[question.id],
                {"abstain_reason": "time_budget"},
                0.0,
            )
            timed_out += 1
    return timed_out


def _ensure_evals_on_path() -> None:
    """evals/ lives at the repo (or image) root, not inside the installed package."""
    candidates = [
        Path.cwd(),
        Path(__file__).resolve().parents[2],  # repo root when src/datamate/predict.py
        Path("/app"),
    ]
    for root in candidates:
        if (root / "evals" / "check_submission.py").is_file():
            root_s = str(root)
            if root_s not in sys.path:
                sys.path.insert(0, root_s)
            return


def _check_submission(submission_path: Path, question_ids: list[str]) -> int:
    """Print check_submission summary. Exit 0 with warnings; 1 only if missing/malformed."""
    _ensure_evals_on_path()
    from evals.check_submission import check_lines

    if not submission_path.is_file():
        print(f"ERROR: submission.jsonl missing at {submission_path}", flush=True)
        return 1
    try:
        lines = submission_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        print(f"ERROR: cannot read submission.jsonl ({exc})", flush=True)
        return 1
    result = check_lines(lines, question_ids, allow_missing=False)
    answered = sum(n for kind, n in result.types.items() if kind != "abstain")
    print(f"questions: {len(question_ids)} | lines: {sum(result.types.values())}")
    print(f"answered: {answered} | abstain: {result.types.get('abstain', 0)}")
    print("by type: " + ", ".join(f"{k}={v}" for k, v in sorted(result.types.items())))
    for warning in result.warnings[:20]:
        print(f"WARNING {warning}")
    for error in result.errors[:50]:
        print(f"ERROR {error}")
    # Malformed (invalid JSON / bad types) -> non-zero; missing lines already errors.
    if result.errors:
        print(f"FAILED: {len(result.errors)} error(s)")
        return 1
    print("OK: submission ready")
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="DataMate offline predict")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Split directory (questions.jsonl, data/, calendar.csv); read-only",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for submission.jsonl, logs/, timing.json, work/",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--time-budget-s",
        type=float,
        default=DEFAULT_TIME_BUDGET_S,
        help="Stop starting new questions after this many seconds (default 18900 = 5h15m)",
    )
    args = parser.parse_args(argv)

    submission = run_predict(
        args.data_dir,
        args.out_dir,
        workers=args.workers,
        time_budget_s=args.time_budget_s,
    )
    # Re-derive question ids after settings were pointed at data_dir.
    split = args.data_dir.resolve().name
    question_ids = [q.id for q in load_questions(split)]
    code = _check_submission(submission, question_ids)
    raise SystemExit(code)


if __name__ == "__main__":
    _ensure_evals_on_path()
    main()

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from datamate.io.questions import Reference
from datamate.scoring.scorer import score_file
from evals.poc_report import (
    build_report,
    combo_scores,
    load_run,
    pair_matrix,
)

SECRET_QUESTION = "Doanh thu tháng này của quán cà phê bí mật là bao nhiêu?"
SECRET_ANSWER = 42_000_000


def _refs() -> dict[str, Reference]:
    return {
        "q1": Reference(id="q1", type="number", answer=10),
        "q2": Reference(id="q2", type="number", answer=20),
        "q3": Reference(id="q3", type="number", answer=SECRET_ANSWER),
    }


def _write_run(
    root: Path,
    *,
    git_commit: str,
    system: str,
    answers: dict[str, dict[str, Any]],
    abstain_reasons: dict[str, str] | None = None,
    git_dirty: bool = False,
    workers: int = 1,
    llm_cache_enabled: bool = False,
) -> Path:
    """Write a tiny run_eval-like directory for 3 questions."""
    root.mkdir(parents=True, exist_ok=True)
    logs = root / "logs"
    logs.mkdir()
    order = ["q1", "q2", "q3"]
    abstain_reasons = abstain_reasons or {}
    questions = {
        "q1": "Số đơn hoàn thành?",
        "q2": "Doanh thu hôm nay?",
        "q3": SECRET_QUESTION,
    }
    businesses = {"q1": "biz_a", "q2": "biz_a", "q3": "biz_b"}

    submission_lines: list[str] = []
    for qid in order:
        ans = answers[qid]
        line = {"id": qid, **{k: v for k, v in ans.items() if k != "id"}}
        submission_lines.append(json.dumps(line, ensure_ascii=False))
        trace: dict[str, Any] = {"system": system}
        if ans.get("type") == "abstain":
            trace["abstain_reason"] = abstain_reasons.get(qid, "test_abstain")
        if system == "direct":
            trace["llm_calls"] = 1
            trace["attempts"] = [{"code": "result = 1", "error": None}]
        else:
            trace["parse_attempts"] = [{"raw": "{}", "error": None}]
            trace["spec"] = {"kind": "value"}
        log = {
            "id": qid,
            "business_id": businesses[qid],
            "question": questions[qid],
            "as_of_date": "2026-08-15",
            "final_answer": line,
            "elapsed_s": 0.5,
            "trace": trace,
        }
        (logs / f"{qid}.json").write_text(
            json.dumps(log, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    (root / "submission.jsonl").write_text(
        "\n".join(submission_lines) + "\n",
        encoding="utf-8",
    )

    refs = _refs()
    result = score_file(root / "submission.jsonl", refs, order)
    metrics = {
        "N": result.N,
        "PA": result.PA,
        "answered": result.answered,
        "correct": result.correct,
        "wrong": result.wrong,
        "coverage": result.coverage,
        "precision": result.precision,
        "timing": {"total_wall_s": 1.5, "mean_s": 0.5, "p95_s": 0.5},
        "abstain_reasons": {},
        "metadata": {
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "system": system,
            "split": "train",
            "subset": "poc",
            "workers": workers,
            "settings": {
                "llm_model": "qwen3-4b",
                "llm_cache_enabled": llm_cache_enabled,
            },
        },
    }
    (root / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return root


def _load_refs(_split: str) -> dict[str, Reference]:
    return _refs()


def test_rescore_matches_and_mismatch_raises(tmp_path: Path) -> None:
    run_dir = _write_run(
        tmp_path / "run_ok",
        git_commit="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        system="pipeline_llm",
        answers={
            "q1": {"type": "number", "answer": 10},
            "q2": {"type": "abstain"},
            "q3": {"type": "number", "answer": SECRET_ANSWER},
        },
    )
    loaded = load_run("X", run_dir, load_refs=_load_refs)
    assert loaded.result.correct == 2
    assert loaded.result.wrong == 0

    bad = tmp_path / "run_bad"
    _write_run(
        bad,
        git_commit="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        system="pipeline_llm",
        answers={
            "q1": {"type": "number", "answer": 10},
            "q2": {"type": "abstain"},
            "q3": {"type": "number", "answer": SECRET_ANSWER},
        },
    )
    metrics = json.loads((bad / "metrics.json").read_text(encoding="utf-8"))
    metrics["correct"] = 99
    (bad / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="re-score"):
        load_run("Y", bad, load_refs=_load_refs)


def test_combo_uses_second_only_where_first_abstained(tmp_path: Path) -> None:
    first_dir = _write_run(
        tmp_path / "first",
        git_commit="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        system="pipeline_llm",
        answers={
            "q1": {"type": "number", "answer": 10},  # correct
            "q2": {"type": "abstain"},  # take second
            "q3": {"type": "number", "answer": 1},  # wrong, keep first
        },
    )
    second_dir = _write_run(
        tmp_path / "second",
        git_commit="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        system="direct",
        answers={
            "q1": {"type": "number", "answer": 999},  # would be wrong, unused
            "q2": {"type": "number", "answer": 20},  # correct, used
            "q3": {"type": "number", "answer": SECRET_ANSWER},  # unused
        },
    )
    first = load_run("B", first_dir, load_refs=_load_refs)
    second = load_run("C", second_dir, load_refs=_load_refs)
    combo = combo_scores(first, second)
    # q1=+1 (first), q2=+1 (second), q3=-1 (first) => PA = 1/3
    assert combo["correct"] == 2
    assert combo["wrong"] == 1
    assert combo["PA"] == pytest.approx(1 / 3)


def test_pair_matrix_and_id_mismatch(tmp_path: Path) -> None:
    a_dir = _write_run(
        tmp_path / "a",
        git_commit="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        system="pipeline_llm",
        answers={
            "q1": {"type": "number", "answer": 10},
            "q2": {"type": "abstain"},
            "q3": {"type": "number", "answer": 1},
        },
    )
    b_dir = _write_run(
        tmp_path / "b",
        git_commit="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        system="direct",
        answers={
            "q1": {"type": "number", "answer": 10},
            "q2": {"type": "number", "answer": 20},
            "q3": {"type": "abstain"},
        },
    )
    a = load_run("A", a_dir, load_refs=_load_refs)
    b = load_run("B", b_dir, load_refs=_load_refs)
    counts, y_right, y_wrong = pair_matrix(a, b)
    # q1: (+1,+1), q2: (0,+1), q3: (-1,0)
    assert counts[(1, 1)] == 1
    assert counts[(0, 1)] == 1
    assert counts[(-1, 0)] == 1
    assert y_right == 1
    assert y_wrong == 0

    # Differing ids: drop q3 from B's submission/logs order by rewriting
    weird = tmp_path / "weird"
    _write_run(
        weird,
        git_commit="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        system="direct",
        answers={
            "q1": {"type": "number", "answer": 10},
            "q2": {"type": "number", "answer": 20},
            "q3": {"type": "abstain"},
        },
    )
    # Remove q3 from submission and metrics recompute
    lines = [
        line
        for line in (weird / "submission.jsonl").read_text(encoding="utf-8").splitlines()
        if '"q3"' not in line
    ]
    (weird / "submission.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = score_file(weird / "submission.jsonl", _refs(), ["q1", "q2"])
    metrics = json.loads((weird / "metrics.json").read_text(encoding="utf-8"))
    metrics.update(
        {
            "N": result.N,
            "PA": result.PA,
            "answered": result.answered,
            "correct": result.correct,
            "wrong": result.wrong,
            "coverage": result.coverage,
            "precision": result.precision,
        }
    )
    (weird / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    weird_run = load_run("W", weird, load_refs=_load_refs)
    with pytest.raises(ValueError, match="id sets differ"):
        pair_matrix(a, weird_run)


def test_out_has_no_secret_text_and_commit_warning(tmp_path: Path) -> None:
    a_dir = _write_run(
        tmp_path / "run_a",
        git_commit="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        system="pipeline_llm",
        answers={
            "q1": {"type": "number", "answer": 10},
            "q2": {"type": "abstain"},
            "q3": {"type": "number", "answer": 1},
        },
        abstain_reasons={"q2": "out_of_scope"},
    )
    b_dir = _write_run(
        tmp_path / "run_b",
        git_commit="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        system="direct",
        answers={
            "q1": {"type": "number", "answer": 10},
            "q2": {"type": "number", "answer": 20},
            "q3": {"type": "number", "answer": SECRET_ANSWER},
        },
    )
    out_path = tmp_path / "out.md"
    detail_path = tmp_path / "detail.md"
    out_md, detail_md, _runs = build_report(
        [("A", a_dir), ("C", b_dir)],
        pairs=[("A", "C")],
        combos=[("AthenC", "A", "C")],
        mapping_specs=[],
        timing_specs=[],
        load_refs=_load_refs,
    )
    out_path.write_text(out_md, encoding="utf-8")
    detail_path.write_text(detail_md, encoding="utf-8")

    assert SECRET_QUESTION not in out_md
    assert str(SECRET_ANSWER) not in out_md
    assert "quán cà phê bí mật" not in out_md
    assert "mixes git commits" in out_md
    assert "aaaaaaa" in out_md and "bbbbbbb" in out_md

    # Detail may contain secrets
    assert SECRET_QUESTION in detail_md
    assert str(SECRET_ANSWER) in detail_md


def test_dirty_commit_star_and_uncommitted_warning(tmp_path: Path) -> None:
    clean = _write_run(
        tmp_path / "clean",
        git_commit="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        system="pipeline_llm",
        answers={
            "q1": {"type": "number", "answer": 10},
            "q2": {"type": "abstain"},
            "q3": {"type": "number", "answer": SECRET_ANSWER},
        },
        git_dirty=False,
    )
    dirty = _write_run(
        tmp_path / "dirty",
        git_commit="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        system="direct",
        answers={
            "q1": {"type": "number", "answer": 10},
            "q2": {"type": "number", "answer": 20},
            "q3": {"type": "number", "answer": SECRET_ANSWER},
        },
        git_dirty=True,
    )
    out_md, _detail, _runs = build_report(
        [("Clean", clean), ("Dirty", dirty)],
        pairs=[("Clean", "Dirty")],
        combos=[],
        mapping_specs=[],
        timing_specs=[],
        load_refs=_load_refs,
    )
    assert "| Dirty |" in out_md or "aaaaaaa*" in out_md
    assert "aaaaaaa*" in out_md
    assert "uncommitted changes" in out_md


def test_timing_rejects_workers_and_cache(tmp_path: Path) -> None:
    answers = {
        "q1": {"type": "number", "answer": 10},
        "q2": {"type": "number", "answer": 20},
        "q3": {"type": "number", "answer": SECRET_ANSWER},
    }
    workers_bad = _write_run(
        tmp_path / "workers",
        git_commit="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        system="pipeline_llm",
        answers=answers,
        workers=16,
        llm_cache_enabled=False,
    )
    cache_bad = _write_run(
        tmp_path / "cache",
        git_commit="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        system="pipeline_llm",
        answers=answers,
        workers=1,
        llm_cache_enabled=True,
    )
    ok = _write_run(
        tmp_path / "ok",
        git_commit="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        system="pipeline_llm",
        answers=answers,
        workers=1,
        llm_cache_enabled=False,
    )
    out_md, _detail, _runs = build_report(
        [("W", workers_bad), ("Cache", cache_bad), ("Ok", ok)],
        pairs=[],
        combos=[],
        mapping_specs=[],
        timing_specs=[("W", workers_bad), ("Cache", cache_bad), ("Ok", ok)],
        load_refs=_load_refs,
    )
    assert "not a timing run (workers=16, cache=off)" in out_md
    assert "not a timing run (workers=1, cache=on)" in out_md
    assert "0.500" in out_md  # mean s/q for the valid timing run
    assert "0.07 h" in out_md or "0.07" in out_md  # 0.5 * 500 / 3600

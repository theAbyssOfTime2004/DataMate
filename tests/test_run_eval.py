from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from datamate.config import PROJECT_ROOT, settings
from datamate.io.paths import business_dir

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datamate.scoring.answer import Answer
from evals.run_eval import (
    MAX_CONSECUTIVE_LLM_UNAVAILABLE,
    _unique_run_dir,
    run_evaluation,
)


class _DeadLLMSystem:
    name = "dead_llm"

    def answer(self, question: object, workdir: Path) -> tuple[Answer, dict[str, object]]:
        del question, workdir
        return Answer.abstain(), {"abstain_reason": "llm_unavailable"}


def test_business_dir_train_biz_101() -> None:
    if not (settings.data_root / "train").exists():
        pytest.skip(f"Train data folder absent at {settings.data_root / 'train'}")

    bdir = business_dir("train", "biz_101")
    assert bdir.is_dir()
    assert (bdir / "README.md").is_file()

    with pytest.raises(FileNotFoundError, match="__missing_biz__"):
        business_dir("train", "__missing_biz__")


def test_unique_run_dir_collision(tmp_path: Path) -> None:
    d1 = _unique_run_dir(tmp_path, "20261028-000000_abstain_poc")
    d1.mkdir()
    d2 = _unique_run_dir(tmp_path, "20261028-000000_abstain_poc")
    d2.mkdir()
    d3 = _unique_run_dir(tmp_path, "20261028-000000_abstain_poc")
    assert d1.name == "20261028-000000_abstain_poc"
    assert d2.name == "20261028-000000_abstain_poc_2"
    assert d3.name == "20261028-000000_abstain_poc_3"


def test_run_eval_abstain_limit_20(tmp_path: Path) -> None:
    if not (settings.data_root / "train").exists():
        pytest.skip(f"Train data folder absent at {settings.data_root / 'train'}")

    run_dir = run_evaluation(
        split="train",
        subset="poc",
        system_name="abstain",
        limit=20,
        runs_dir=tmp_path,
    )

    assert run_dir.is_dir()
    sub_file = run_dir / "submission.jsonl"
    metrics_file = run_dir / "metrics.json"
    summary_file = run_dir / "summary.md"
    logs_dir = run_dir / "logs"

    assert sub_file.is_file()
    assert metrics_file.is_file()
    assert summary_file.is_file()
    assert logs_dir.is_dir()

    sub_lines = [line for line in sub_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(sub_lines) == 20

    log_files = list(logs_dir.glob("*.json"))
    assert len(log_files) == 20

    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert metrics["N"] == 20
    assert metrics["PA"] == 0.0
    assert metrics["coverage"] == 0.0
    assert metrics["answered"] == 0
    assert metrics["correct"] == 0
    assert metrics["wrong"] == 0
    assert set(metrics["timing"].keys()) == {"total_wall_s", "mean_s", "p95_s"}
    assert "llm_api_key" not in metrics["metadata"]["settings"]


def test_run_eval_public_test_unscored_limit_5(tmp_path: Path) -> None:
    if not (settings.data_root / "public_test").exists():
        pytest.skip(f"public_test folder absent at {settings.data_root / 'public_test'}")

    with pytest.raises(ValueError, match="Only subset 'all' is valid"):
        run_evaluation(
            split="public_test",
            subset="poc",
            system_name="abstain",
            limit=5,
            runs_dir=tmp_path,
        )

    run_dir = run_evaluation(
        split="public_test",
        subset="all",
        system_name="abstain",
        limit=5,
        runs_dir=tmp_path,
    )

    sub_file = run_dir / "submission.jsonl"
    metrics_file = run_dir / "metrics.json"
    summary_file = run_dir / "summary.md"

    sub_lines = [line for line in sub_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(sub_lines) == 5

    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert metrics["N"] == 5
    assert metrics["coverage"] == 0.0
    assert "PA" not in metrics
    assert set(metrics["timing"].keys()) == {"total_wall_s", "mean_s", "p95_s"}
    assert "unscored split" in summary_file.read_text(encoding="utf-8")


def test_run_eval_aborts_when_the_llm_server_is_down(tmp_path: Path) -> None:
    if not (settings.data_root / "train").exists():
        pytest.skip(f"Train data folder absent at {settings.data_root / 'train'}")

    with pytest.raises(SystemExit, match="unavailable"):
        run_evaluation(
            split="train",
            subset="poc",
            limit=10,
            runs_dir=tmp_path,
            system_instance=_DeadLLMSystem(),  # type: ignore[arg-type]
        )
    logs = list(tmp_path.glob("*/logs/*.json"))
    assert len(logs) == MAX_CONSECUTIVE_LLM_UNAVAILABLE
    assert not list(tmp_path.glob("*/summary.md"))


class _SlowAbstainSystem:
    """Finishes questions out of order so the run must restore question order."""

    name = "slow_abstain"

    def __init__(self) -> None:
        self.calls = 0

    def answer(self, question: object, workdir: Path) -> tuple[Answer, dict[str, object]]:
        del workdir
        self.calls += 1
        time.sleep(0.05 if self.calls % 2 else 0.0)
        return Answer.abstain(), {"abstain_reason": f"stub {question.id}"}  # type: ignore[attr-defined]


def test_run_eval_workers_keep_question_order(tmp_path: Path) -> None:
    if not (settings.data_root / "train").exists():
        pytest.skip(f"Train data folder absent at {settings.data_root / 'train'}")

    sequential = run_evaluation(
        split="train", subset="poc", system_name="abstain", limit=8, runs_dir=tmp_path / "seq"
    )
    parallel = run_evaluation(
        split="train",
        subset="poc",
        limit=8,
        runs_dir=tmp_path / "par",
        system_instance=_SlowAbstainSystem(),  # type: ignore[arg-type]
        workers=4,
    )

    def ids(run_dir: Path) -> list[str]:
        lines = (run_dir / "submission.jsonl").read_text(encoding="utf-8").splitlines()
        return [json.loads(line)["id"] for line in lines]

    assert ids(parallel) == ids(sequential)
    assert len(list((parallel / "logs").glob("*.json"))) == 8
    metrics = json.loads((parallel / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["metadata"]["workers"] == 4


def test_run_eval_workers_abort_when_the_llm_server_is_down(tmp_path: Path) -> None:
    if not (settings.data_root / "train").exists():
        pytest.skip(f"Train data folder absent at {settings.data_root / 'train'}")

    with pytest.raises(SystemExit, match="unavailable"):
        run_evaluation(
            split="train",
            subset="poc",
            limit=20,
            runs_dir=tmp_path,
            system_instance=_DeadLLMSystem(),  # type: ignore[arg-type]
            workers=4,
        )
    assert not list(tmp_path.glob("*/summary.md"))

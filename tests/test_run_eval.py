from __future__ import annotations

import json
from pathlib import Path

import pytest

from datamate.config import settings
from evals.run_eval import run_evaluation


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
    assert "llm_api_key" not in metrics["metadata"]["settings"]

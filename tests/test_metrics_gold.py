from __future__ import annotations

import sys

import duckdb
import pytest

from datamate.config import PROJECT_ROOT, settings
from datamate.etl import load_or_build_canonical
from datamate.mapping import load_mapping_yaml
from datamate.metrics import MetricEngine
from datamate.rules import load_rules
from datamate.scoring.answer import to_submission_line
from datamate.scoring.scorer import score_one
from datamate.timeres import load_calendar

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.metrics_gold_cases import CASES, GoldCase

GOLD_DIR = PROJECT_ROOT / "evals" / "gold_mappings"


@pytest.fixture(scope="module")
def engines() -> dict[str, MetricEngine]:
    if not (settings.data_root / "train").is_dir():
        pytest.skip("train split absent")
    assert [case.n for case in CASES] == list(range(1, 34))
    assert len({case.question_id for case in CASES}) == 33

    rules = load_rules(PROJECT_ROOT / "rules" / "train.json")
    calendar = load_calendar("train")
    built: dict[str, MetricEngine] = {}
    connections: list[duckdb.DuckDBPyConnection] = []
    for business_id in sorted({case.business_id for case in CASES}):
        mapping = load_mapping_yaml(GOLD_DIR / f"{business_id}.yaml")
        db_path, report = load_or_build_canonical("train", business_id, mapping)
        connection = duckdb.connect(str(db_path), read_only=True)
        connections.append(connection)
        built[business_id] = MetricEngine(
            connection,
            rules,
            calendar,
            report.integrity.min_order_date,
            mapping.data_end_date,
        )
    yield built
    for connection in connections:
        connection.close()


@pytest.fixture(scope="module")
def references() -> dict:
    if not (settings.data_root / "train").is_dir():
        pytest.skip("train split absent")
    from datamate.io.questions import load_references

    return load_references("train")


@pytest.mark.parametrize("case", CASES, ids=[f"{case.n:02d}-{case.question_id}" for case in CASES])
def test_gold_query_matches_the_reference(
    case: GoldCase,
    engines: dict[str, MetricEngine],
    references: dict,
) -> None:
    ref = references[case.question_id]
    assert ref.type == case.expected_type
    assert ref.answer == case.expected

    got = engines[case.business_id].answer(case.spec, case.as_of)
    submitted = to_submission_line(case.question_id, got)
    assert score_one(submitted, ref) == 1, submitted

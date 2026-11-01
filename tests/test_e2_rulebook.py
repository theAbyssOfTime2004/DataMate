"""E2: rulebook → RuleConfig classify/extract, ranking_measure, unsupported abstain."""

from __future__ import annotations

import datetime
import math
from pathlib import Path
from typing import Any

import duckdb
import pytest

from datamate.config import settings
from datamate.llm.client import UsageEntry
from datamate.metrics.engine import MetricEngine, RuleUnsupported
from datamate.query.ext.spec import TopProductsQuery
from datamate.rules.from_rulebook import PROMPT_VERSION, build_rulebook_config
from datamate.rules.knobs import rule_unsupported_reason
from datamate.rules.model import RuleConfig
from datamate.systems.pipeline import PipelineSystem
from datamate.timeres.calendar import LunarCalendar
from datamate.timeres.periods import MonthExpr


class FakeClient:
    """Scripted chat_json by schema_name; classify/extract each need 2 identical votes."""

    def __init__(self, scripted: dict[str, list[dict[str, Any]]]) -> None:
        self.scripted = {k: list(v) for k, v in scripted.items()}
        self.calls: list[str] = []
        self.usage_log: list[UsageEntry] = []

    def count_tokens(self, text: str) -> int:
        return math.ceil(len(text) / 2.5)

    def chat_json(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        schema_name: str,
        max_tokens: int | None = None,
        required_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        del messages, schema, max_tokens, required_keys
        self.calls.append(schema_name)
        queue = self.scripted.get(schema_name, [])
        if not queue:
            raise AssertionError(f"no scripted answer for {schema_name!r}")
        answer = queue.pop(0)
        self.usage_log.append(
            UsageEntry(cache_hit=False, prompt_tokens=1, completion_tokens=1, latency_s=0.0)
        )
        return answer


def _invented_rulebook(ranking: str = "revenue") -> str:
    return f"""# Invented Shop Rules

## 1. Sales

| Luật | Mặc định |
| --- | --- |
| Which statuses count as sold | Only completed orders count toward revenue. |
| Ranking best sellers | Rank products by {ranking}. |

## 2. Oddities

| Luật | Mặc định |
| --- | --- |
| Stamp card | Ten stamps earn a free bun. |
"""


def test_classify_and_extract_with_fake_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "cache_dir", tmp_path / "cache")
    path = tmp_path / "rulebook.md"
    path.write_text(_invented_rulebook("revenue"), encoding="utf-8")

    # 3 rows × 2 classify votes, then extract for revenue_statuses + ranking_measure (2 each).
    client = FakeClient(
        {
            "rulebook_classify": [
                {"knob": "revenue_statuses"},
                {"knob": "revenue_statuses"},
                {"knob": "ranking_measure"},
                {"knob": "ranking_measure"},
                {"knob": "other"},
                {"knob": "other"},
            ],
            "rulebook_extract_revenue_statuses": [
                {"value": ["completed"]},
                {"value": ["completed"]},
            ],
            "rulebook_extract_ranking_measure": [
                {"value": "revenue"},
                {"value": "revenue"},
            ],
        }
    )
    cfg = build_rulebook_config(path, client, use_cache=True)
    assert cfg.rules.ranking_measure == "revenue"
    assert cfg.rules.revenue_statuses == ["completed"]
    # Every row stays a note (knob rows too), so E3 can still read their definitions.
    assert len(cfg.notes) == 3
    assert "Stamp card" in {n.name for n in cfg.notes}
    assert "ranking_measure" not in cfg.unsupported

    # Disk cache round-trip (no further LLM calls).
    calls_before = len(client.calls)
    cfg2 = build_rulebook_config(path, client, use_cache=True)
    assert cfg2.rules.ranking_measure == "revenue"
    assert len(client.calls) == calls_before
    cache_file = tmp_path / "cache" / "rulebook" / PROMPT_VERSION / f"{cfg.rulebook_sha256}.json"
    assert cache_file.is_file()


def test_extract_disagreement_keeps_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "cache_dir", tmp_path / "cache")
    path = tmp_path / "rulebook.md"
    path.write_text(_invented_rulebook("revenue"), encoding="utf-8")
    client = FakeClient(
        {
            "rulebook_classify": [
                {"knob": "ranking_measure"},
                {"knob": "ranking_measure"},
                {"knob": "other"},
                {"knob": "other"},
                {"knob": "other"},
                {"knob": "other"},
            ],
            "rulebook_extract_ranking_measure": [
                {"value": "revenue"},
                {"value": "qty_base"},  # disagree
            ],
        }
    )
    cfg = build_rulebook_config(path, client, use_cache=False)
    assert cfg.rules.ranking_measure == "qty_base"  # train default
    assert any(w.startswith("extract_disagree:ranking_measure") for w in cfg.warnings)


def test_unsupported_knob_family_abstains(tmp_path: Path) -> None:
    con = duckdb.connect(":memory:")
    con.execute(
        "CREATE TABLE orders (order_id VARCHAR, order_date DATE, status VARCHAR, "
        "has_return BOOLEAN, payment_method VARCHAR, channel VARCHAR)"
    )
    con.execute(
        "CREATE TABLE order_lines (order_id VARCHAR, product_key VARCHAR, product_id VARCHAR, "
        "product_name VARCHAR, qty DECIMAL, qty_base DECIMAL, unit_price BIGINT, "
        "line_amount BIGINT, cost_per_unit BIGINT)"
    )
    con.execute(
        "INSERT INTO orders VALUES ('o1', DATE '2026-06-01', 'completed', FALSE, NULL, NULL)"
    )
    con.execute(
        "INSERT INTO order_lines VALUES ('o1', 'a', 'P1', 'A', 1, 1, 1000, 1000, 100)"
    )
    rules = RuleConfig()
    engine = MetricEngine(
        con,
        rules,
        LunarCalendar({}),
        datetime.date(2026, 1, 1),
        datetime.date(2026, 12, 31),
        unsupported_knobs=frozenset({"ranking_measure"}),
    )
    reason = rule_unsupported_reason("top_products", engine.unsupported_knobs)
    assert reason == "rule_unsupported:ranking_measure"
    with pytest.raises(RuleUnsupported) as excinfo:
        engine.answer_ext(
            TopProductsQuery(k=1, by="qty_base", mode="max", period=MonthExpr(month=6, year=2026)),
            datetime.date(2026, 6, 15),
        )
    assert excinfo.value.knob == "ranking_measure"


def test_ranking_measure_changes_top_products(tmp_path: Path) -> None:
    """Same shop: qty_base picks product A; revenue picks product B."""
    con = duckdb.connect(":memory:")
    con.execute(
        "CREATE TABLE orders (order_id VARCHAR, order_date DATE, status VARCHAR, "
        "has_return BOOLEAN, payment_method VARCHAR, channel VARCHAR)"
    )
    con.execute(
        "CREATE TABLE order_lines (order_id VARCHAR, product_key VARCHAR, product_id VARCHAR, "
        "product_name VARCHAR, qty DECIMAL, qty_base DECIMAL, unit_price BIGINT, "
        "line_amount BIGINT, cost_per_unit BIGINT)"
    )
    con.execute(
        "INSERT INTO orders VALUES ('o1', DATE '2026-06-01', 'completed', FALSE, NULL, NULL)"
    )
    # A: high qty, low revenue; B: low qty, high revenue.
    con.execute(
        "INSERT INTO order_lines VALUES "
        "('o1', 'a', 'P1', 'Product A', 10, 10, 100, 1000, 10),"
        "('o1', 'b', 'P2', 'Product B', 1, 1, 5000, 5000, 10)"
    )
    cal = LunarCalendar({})
    period = MonthExpr(month=6, year=2026)
    as_of = datetime.date(2026, 6, 15)
    query = TopProductsQuery(k=1, by="qty_base", mode="max", period=period)

    qty_engine = MetricEngine(
        con, RuleConfig(ranking_measure="qty_base"), cal, datetime.date(2026, 1, 1), as_of
    )
    rev_engine = MetricEngine(
        con, RuleConfig(ranking_measure="revenue"), cal, datetime.date(2026, 1, 1), as_of
    )
    assert qty_engine.answer_ext(query, as_of).answer == "Product A"
    assert rev_engine.answer_ext(query, as_of).answer == "Product B"
    # An explicit money measure in the question ("nhiều tiền nhất") is not overridden.
    by_revenue = TopProductsQuery(k=1, by="revenue", mode="max", period=period)
    assert qty_engine.answer_ext(by_revenue, as_of).answer == "Product B"


def test_pipeline_loads_rulebook_once_per_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "data_root", tmp_path)
    monkeypatch.setattr(settings, "cache_dir", tmp_path / "cache")
    split = "syn_e2"
    split_dir = tmp_path / split
    split_dir.mkdir()
    (split_dir / "rulebook.md").write_text(_invented_rulebook("revenue"), encoding="utf-8")
    client = FakeClient(
        {
            "rulebook_classify": [
                {"knob": "revenue_statuses"},
                {"knob": "revenue_statuses"},
                {"knob": "ranking_measure"},
                {"knob": "ranking_measure"},
                {"knob": "other"},
                {"knob": "other"},
            ],
            "rulebook_extract_revenue_statuses": [
                {"value": ["completed"]},
                {"value": ["completed"]},
            ],
            "rulebook_extract_ranking_measure": [
                {"value": "revenue"},
                {"value": "revenue"},
            ],
        }
    )
    system = PipelineSystem(name="e2_test", client=client)
    first = system._rulebook_for(split)
    second = system._rulebook_for(split)
    assert first is second
    assert first.rules.ranking_measure == "revenue"
    # Classify ran once (3 rows × 2 votes); second call hits the in-memory cache.
    assert client.calls.count("rulebook_classify") == 6

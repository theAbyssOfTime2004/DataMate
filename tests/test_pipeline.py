from __future__ import annotations

import datetime
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import openpyxl
import pytest

from datamate.config import settings
from datamate.io.paths import business_dir
from datamate.io.questions import Question
from datamate.llm.client import LLMUnavailable, UsageEntry
from datamate.scoring.answer import Answer
from datamate.systems.pipeline import LLM_UNAVAILABLE, PipelineSystem


class FakeClient:
    """Scripted answers keyed by schema_name for mapping steps and ParsedQuestion."""

    def __init__(self, scripted_answers: dict[str, list[dict[str, Any]]]) -> None:
        self.scripted: dict[str, list[dict[str, Any]]] = {
            key: list(values) for key, values in scripted_answers.items()
        }
        self.calls: list[dict[str, Any]] = []
        self.usage_log: list[UsageEntry] = []
        self.raise_on_parse: bool = False
        self.unavailable: bool = False

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
        del schema, messages, max_tokens, required_keys
        if self.unavailable:
            raise LLMUnavailable("HTTP transport error: timed out")
        if schema_name == "ParsedQuestion" and self.raise_on_parse:
            raise RuntimeError("forced parse failure")
        queue = self.scripted.get(schema_name, [])
        if not queue:
            raise AssertionError(f"No scripted answer left for {schema_name!r}")
        answer = queue.pop(0)
        self.calls.append({"schema_name": schema_name})
        self.usage_log.append(
            UsageEntry(
                cache_hit=False,
                prompt_tokens=10,
                completion_tokens=10,
                latency_s=0.01,
            )
        )
        return answer


def _filters(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "payment_method": [],
        "channel": [],
        "customer_segment": [],
        "category": [],
        "product": [],
        "shift": None,
    }
    row.update(overrides)
    return row


def _parsed(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "in_scope": True,
        "kind": "value",
        "metric": "revenue",
        "periods": [{"kind": "month_this"}],
        "compare_kind": None,
        "op": None,
        "filters": _filters(),
        "filters_b": None,
        "multi_filters": [],
        "direction": None,
        "granularity": None,
        "mode": None,
    }
    row.update(overrides)
    return row


def _write_shop(biz_dir: Path) -> None:
    biz_dir.mkdir(parents=True, exist_ok=True)
    (biz_dir / "README.md").write_text(
        "# Shop POC\n\n## Quy ước\n- Dữ liệu có đến hết 15/08/2026.\n",
        encoding="utf-8",
    )
    wb_ord = openpyxl.Workbook()
    ws_ord = wb_ord.active
    assert ws_ord is not None
    ws_ord.title = "Sheet1"
    ws_ord.append(["BÁO CÁO HÓA ĐƠN"])
    ws_ord.append([None, None, None, None, None])
    ws_ord.append(["Mã HĐ", "Ngày bán", "Khách hàng", "Trạng thái", "Tổng tiền hàng"])
    ws_ord.append(["ORD01", "01/08/2026", "Khách lẻ", "Đã xong", "110.000đ"])
    ws_ord.append(["ORD02", "02/08/2026", "Nguyễn Văn A", "Đã hủy", "50.000đ"])
    wb_ord.save(biz_dir / "HoaDon.xlsx")
    wb_ord.close()

    wb_ct = openpyxl.Workbook()
    ws_ct = wb_ct.active
    assert ws_ct is not None
    ws_ct.title = "Sheet1"
    ws_ct.append(["Mã HĐ", "Tên món", "SL", "Đơn giá", "Thành tiền"])
    ws_ct.append(["ORD01", "Cà phê sữa", "2", "30.000đ", "60.000đ"])
    ws_ct.append(["ORD01", "Bánh ngọt", "1", "50.000đ", "50.000đ"])
    ws_ct.append(["ORD02", "Trà đào", "1", "50.000đ", "50.000đ"])
    wb_ct.save(biz_dir / "ChiTiet.xlsx")
    wb_ct.close()


def _write_calendar(split_dir: Path) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "calendar.csv").write_text(
        "date,weekday,lunar_day,lunar_month,lunar_year,lunar_leap,holiday\n",
        encoding="utf-8",
    )


def _mapping_scripts() -> dict[str, list[dict[str, Any]]]:
    all_ord_fields = {
        "order_id": "Mã HĐ",
        "order_date": "Ngày bán",
        "order_ts": None,
        "status": "Trạng thái",
        "customer_id": None,
        "customer_name": "Khách hàng",
        "payment_method": None,
        "channel": None,
        "employee_id": None,
        "order_subtotal": "Tổng tiền hàng",
        "order_discount": None,
        "shipping_fee": None,
        "order_payable": None,
    }
    all_line_fields = {
        "order_id": "Mã HĐ",
        "line_no": None,
        "product_id": None,
        "product_name": "Tên món",
        "qty": "SL",
        "unit": None,
        "qty_base": None,
        "unit_price": "Đơn giá",
        "line_discount": None,
        "line_amount": "Thành tiền",
        "cost_per_unit": None,
    }
    return {
        "step_a_table_roles": [
            {
                "tables": [
                    {"table": "ChiTiet.xlsx!Sheet1", "entity": "order_lines"},
                    {"table": "HoaDon.xlsx!Sheet1", "entity": "orders"},
                ],
                "orders_source": "own_table",
            }
        ],
        "step_b_order_lines": [all_line_fields],
        "step_b_orders": [all_ord_fields],
        "step_c_status_map": [{"Đã hủy": "cancelled", "Đã xong": "completed"}],
        "step_d_customers_returns": [{"walk_in_values": ["Khách lẻ"], "return_rule": "status"}],
    }


@pytest.fixture
def synthetic_shop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(settings, "data_root", tmp_path)
    monkeypatch.setattr(settings, "cache_dir", tmp_path / "cache")
    split_dir = tmp_path / "train"
    _write_calendar(split_dir)
    biz_dir = split_dir / "data" / "biz_syn"
    _write_shop(biz_dir)
    return biz_dir


def test_pipeline_answers_revenue_end_to_end(synthetic_shop: Path) -> None:
    parsed = _parsed(periods=[{"kind": "month_this"}])
    client = FakeClient(
        {
            **_mapping_scripts(),
            "ParsedQuestion": [parsed, parsed],
        }
    )
    system = PipelineSystem(name="pipeline_test", client=client)
    try:
        question = Question(
            id="q_syn_revenue",
            business_id="biz_syn",
            question="doanh thu thang nay bao nhieu",
            as_of_date=datetime.date(2026, 8, 15),
        )
        answer, trace = system.answer(question, business_dir("train", "biz_syn"))
        assert answer == Answer(type="number", answer=110_000)
        assert trace["abstain_reason"] is None
        assert trace["mapping_fingerprint"]
        assert trace["spec"]["kind"] == "value"
        assert trace["coverage"] == ["full"]
        assert trace["sql"]
        assert len(trace["parse_attempts"]) >= 2
    finally:
        system.close()


def test_pipeline_parse_exception_becomes_abstain(synthetic_shop: Path) -> None:
    client = FakeClient(_mapping_scripts())
    client.raise_on_parse = True
    system = PipelineSystem(name="pipeline_test", client=client)
    try:
        question = Question(
            id="q_syn_boom",
            business_id="biz_syn",
            question="doanh thu thang nay",
            as_of_date=datetime.date(2026, 8, 15),
        )
        answer, trace = system.answer(question, business_dir("train", "biz_syn"))
        assert answer.type == "abstain"
        assert trace["abstain_reason"] == "RuntimeError: forced parse failure"
        assert "timings" in trace
    finally:
        system.close()


def test_pipeline_llm_outage_is_flagged_and_mapping_is_retried(synthetic_shop: Path) -> None:
    parsed = _parsed(periods=[{"kind": "month_this"}])
    client = FakeClient({**_mapping_scripts(), "ParsedQuestion": [parsed, parsed]})
    client.unavailable = True
    system = PipelineSystem(name="pipeline_test", client=client, mapping_provider=None)
    question = Question(
        id="q_syn_down",
        business_id="biz_syn",
        question="doanh thu thang nay bao nhieu",
        as_of_date=datetime.date(2026, 8, 15),
    )
    try:
        answer, trace = system.answer(question, business_dir("train", "biz_syn"))
        assert answer.type == "abstain"
        assert trace["abstain_reason"] == LLM_UNAVAILABLE
        assert "timed out" in trace["error"]

        client.unavailable = False
        answer, trace = system.answer(question, business_dir("train", "biz_syn"))
        assert answer == Answer(type="number", answer=110_000)
        assert trace["abstain_reason"] is None
    finally:
        system.close()


def test_pipeline_threads_share_one_mapping_per_shop(synthetic_shop: Path) -> None:
    parsed = _parsed(periods=[{"kind": "month_this"}])
    # Mapping answers are scripted once: a second build would run out and abstain.
    client = FakeClient({**_mapping_scripts(), "ParsedQuestion": [parsed] * 8})
    system = PipelineSystem(name="pipeline_test", client=client)
    questions = [
        Question(
            id=f"q_syn_thread_{index}",
            business_id="biz_syn",
            question="doanh thu thang nay bao nhieu",
            as_of_date=datetime.date(2026, 8, 15),
        )
        for index in range(4)
    ]
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            answers = list(
                pool.map(
                    lambda question: system.answer(question, business_dir("train", "biz_syn")),
                    questions,
                )
            )
        assert [answer for answer, _trace in answers] == [Answer(type="number", answer=110_000)] * 4
    finally:
        system.close()

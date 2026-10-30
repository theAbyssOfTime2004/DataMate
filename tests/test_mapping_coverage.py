from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

from datamate.config import PROJECT_ROOT
from datamate.etl import build_canonical
from datamate.io.profile import profile_business
from datamate.io.raw import read_business
from datamate.llm.client import UsageEntry

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.mapping_coverage import _coverage_one, _write_summary_md


class _StubClient:
    def __init__(self, scripted: dict[str, list[dict[str, Any]]]) -> None:
        self.scripted = {k: list(v) for k, v in scripted.items()}
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
        del messages, max_tokens, required_keys
        queue = self.scripted.get(schema_name, [])
        if not queue:
            if schema_name.endswith("_later"):
                return {k: None for k in schema.get("properties", {})}
            raise AssertionError(f"No scripted answer for {schema_name!r}")
        ans = queue.pop(0)
        self.usage_log.append(
            UsageEntry(
                cache_hit=False, prompt_tokens=10, completion_tokens=5, latency_s=0.01
            )
        )
        return ans


def test_mapping_coverage_stub_client(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "train" / "data" / "biz_cov"
    data_root.mkdir(parents=True)
    (data_root / "README.md").write_text(
        "# Shop\n\n## Quy ước\n- Dữ liệu có đến hết 31/03/2025.\n",
        encoding="utf-8",
    )
    (data_root / "ban.csv").write_text(
        "Mã đơn,Ngày,Khách,Trạng thái,Tên món,SL,Thành tiền\n"
        "ORD01,15/03/2025,An,xong,Cà phê,1,10000\n",
        encoding="utf-8",
    )

    raw = read_business(data_root)
    profile = profile_business(raw)
    ban = next(t.name for t in profile.tables if "ban" in t.name)

    line_fields = {
        "order_id": "Mã đơn",
        "line_no": None,
        "product_id": None,
        "product_name": "Tên món",
        "qty": "SL",
        "unit": None,
        "qty_base": None,
        "unit_price": None,
        "line_discount": None,
        "line_amount": "Thành tiền",
        "cost_per_unit": None,
    }
    ord_fields = {
        "order_id": "Mã đơn",
        "order_date": "Ngày",
        "order_ts": None,
        "status": "Trạng thái",
        "customer_id": None,
        "customer_name": "Khách",
        "payment_method": None,
        "channel": None,
        "employee_id": None,
        "order_subtotal": None,
        "order_discount": None,
        "shipping_fee": None,
        "order_payable": None,
    }
    client = _StubClient(
        {
            "step_a_table_roles": [
                {
                    "tables": [{"table": ban, "entity": "order_lines"}],
                    "orders_source": "order_lines",
                }
            ],
            "step_b_order_lines": [line_fields],
            "step_b_orders": [ord_fields],
            "step_c_status_map": [{"xong": "completed"}],
            "step_d_customers_returns": [
                {"walk_in_values": [], "return_rule": "status"}
            ],
        }
    )

    import evals.mapping_coverage as cov

    monkeypatch.setattr(cov, "load_business_cached", lambda _s, _b: read_business(data_root))

    def _fake_load_or_build(split: str, biz_id: str, mapping):
        del split, biz_id
        data = build_canonical(read_business(data_root), mapping)
        return tmp_path / "x.duckdb", data.report

    monkeypatch.setattr(cov, "load_or_build_canonical", _fake_load_or_build)

    mappings_dir = tmp_path / "mappings"
    row = _coverage_one(
        "train", "biz_cov", client, mappings_dir=mappings_dir
    )
    assert row["mapping_failed"] is False
    assert row["rows"]["orders"] == 1
    assert row["rows"]["order_lines"] == 1
    assert row["rows"]["payments"] == "-"
    dumped = mappings_dir / "biz_cov.yaml"
    assert dumped.is_file()
    assert "order_lines" in dumped.read_text(encoding="utf-8")

    out = tmp_path / "summary.md"
    _write_summary_md(out, [row], "train")
    text = out.read_text(encoding="utf-8")
    assert "biz_cov" in text
    assert "| orders |" in text or "orders" in text

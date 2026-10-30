from __future__ import annotations

import datetime
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from datamate.config import settings
from datamate.direct.prompts import (
    DIRECT_MAX_TOKENS,
    build_messages,
    fit_messages,
    rulebook_excerpt,
)
from datamate.direct.sandbox import check_code, run_code
from datamate.io.paths import business_dir
from datamate.io.questions import Question
from datamate.llm.client import LLMError, LLMUnavailable, UsageEntry
from datamate.systems.direct import (
    DirectSystem,
    _build_alias_map,
    _format_number,
    _format_result,
)


class FakeClient:
    def __init__(self, answers: list[dict[str, Any] | BaseException]) -> None:
        self.answers = list(answers)
        self.calls: list[dict[str, Any]] = []
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
        del schema, required_keys
        self.calls.append(
            {
                "schema_name": schema_name,
                "max_tokens": max_tokens,
                "messages": messages,
            }
        )
        if not self.answers:
            raise AssertionError("no scripted answers left")
        item = self.answers.pop(0)
        if isinstance(item, BaseException):
            raise item
        self.usage_log.append(
            UsageEntry(
                cache_hit=False,
                prompt_tokens=10,
                completion_tokens=10,
                latency_s=0.01,
            )
        )
        return item


def test_rulebook_excerpt_keeps_matching_sections_and_drops_from_end() -> None:
    text = "\n".join(
        [
            "# Rulebook",
            "",
            "## Đơn hàng và doanh thu",
            "A" * 100,
            "",
            "## Thời gian",
            "B" * 100,
            "",
            "## Kho",
            "C" * 100,
            "",
            "## Cách ghi đáp án",
            "D" * 100,
            "",
        ]
    )
    excerpt = rulebook_excerpt(text, budget_chars=10_000)
    assert "Đơn hàng và doanh thu" in excerpt
    assert "Thời gian" in excerpt
    assert "Cách ghi đáp án" in excerpt
    assert "## Kho" not in excerpt

    tight = rulebook_excerpt(text, budget_chars=250)
    assert "Đơn hàng và doanh thu" in tight
    assert tight.index("Đơn hàng") < tight.index("Thời gian") if "Thời gian" in tight else True
    assert "Cách ghi đáp án" not in tight or "Thời gian" in tight


def test_rulebook_excerpt_falls_back_to_whole_text() -> None:
    text = "## Gioi thieu\nKhong khop tu khoa nao.\n"
    assert rulebook_excerpt(text, budget_chars=10_000) == text


def test_fit_messages_keeps_all_column_names_and_dap_an() -> None:
    readme = "## README\n" + ("Shop readme line.\n" * 40)
    tables = []
    column_names: list[str] = []
    for index in range(17):
        cols = []
        for col_i in range(4):
            name = f"Col_{index}_{col_i}"
            column_names.append(name)
            samples = ", ".join(f'"s{k}"' for k in range(8))
            cols.append(
                f"  - {name} | null 0.10 | distinct 50 | kinds string 1.00 | "
                f"top: {samples} (8) | e.g. {samples}"
            )
        tables.append(
            f"### Table `ban_hang_{index}` (csv)\n"
            f"- sources: 1 | files: f{index}.csv | sheets: - | header_row: 1\n"
            f"- rows: 1000 | exact_duplicate_rows: 0\n"
            f"- columns:\n" + "\n".join(cols) + "\n"
        )
    business = (
        f"# Business `biz_big`\n\n**Files (17)**: x\n\n{readme}\n"
        f"## Tables (17)\n\n" + "\n".join(tables)
    )
    rules = (
        "## Giới thiệu\n" + ("x" * 500) + "\n"
        "## Đơn hàng và doanh thu\n" + ("doanh thu rules " * 80) + "\n"
        "## Thời gian\n" + ("thoi gian " * 80) + "\n"
        "## Cách ghi đáp án\n" + ("dap an rules " * 80) + "\n"
    )
    # Force the budget small enough that full cards overflow, but compact cards fit.
    def count_tokens(text: str) -> int:
        return len(text) // 4

    messages = fit_messages(
        "doanh thu thang nay",
        datetime.date(2026, 8, 15),
        business,
        rules,
        None,
        count_tokens,
        budget_tokens=2500,
    )
    joined = "\n".join(message["content"] for message in messages)
    for name in column_names:
        assert name in joined, name
    assert "đáp án" in joined.casefold() or "dap an" in joined.casefold() or "Đáp án" in joined
    assert "Cách ghi đáp án" in joined or "đáp án" in joined.casefold()


def test_check_code_rejects_dangerous_constructs() -> None:
    assert check_code("import os\n_ = tables\nx=1") is not None
    assert check_code('_ = tables\nopen("x")') is not None
    assert check_code('_ = tables\n__import__("os")') is not None
    assert check_code("_ = tables\nx.__class__") is not None
    assert check_code('_ = tables\n__builtins__["open"]') is not None
    assert check_code('_ = tables\npd.read_pickle("x")') is not None
    assert check_code('_ = tables\ndf.to_csv("x")') is not None
    assert (
        check_code(
            "import pandas as pd\n_ = tables\nresult = {'type': 'number', 'answer': 1}"
        )
        is None
    )


def test_check_code_rejects_redefine_return_and_hardcode() -> None:
    assert check_code("parse_money = None\n_ = tables") == (
        "these are provided; do not redefine them"
    )
    assert check_code("def tables():\n    return 1\n_ = tables") == (
        "these are provided; do not redefine them"
    )
    assert check_code("AS_OF = None\n_ = tables") == (
        "these are provided; do not redefine them"
    )
    assert check_code("return 1\n_ = tables") == (
        "write a top-level script that assigns result"
    )
    assert check_code("result = {'type': 'number', 'answer': 1}") == (
        "compute from tables; do not hard-code the answer"
    )


def _write_tables_dir(
    tables_dir: Path,
    frames: dict[str, pd.DataFrame],
    aliases: dict[str, str] | None = None,
) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for index, (name, frame) in enumerate(frames.items()):
        file_name = f"{index:04d}.pkl"
        frame.to_pickle(tables_dir / file_name)
        files[file_name] = name
    payload = {"files": files, "aliases": aliases or {}}
    (tables_dir / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def test_run_code_ok_timeout_and_error(tmp_path: Path) -> None:
    tables_dir = tmp_path / "tables"
    _write_tables_dir(
        tables_dir,
        {"ban_hang": pd.DataFrame({"Thanh tien": ["10.000đ", "20.000đ"]})},
    )
    as_of = datetime.date(2026, 8, 15)
    good = run_code(
        (
            "import pandas as pd\n"
            "vals = tables['ban_hang']['Thanh tien'].map(parse_money)\n"
            "result = {'type': 'number', 'answer': int(vals.sum())}"
        ),
        tables_dir,
        as_of,
    )
    assert good.ok
    assert good.result == {"type": "number", "answer": 30_000}

    timed = run_code("_ = tables\nwhile True:\n    pass\n", tables_dir, as_of, timeout_s=2)
    assert not timed.ok
    assert timed.error == "timeout"

    boom = run_code("_ = tables\n1/0\n", tables_dir, as_of)
    assert not boom.ok
    assert boom.error is not None
    assert "ZeroDivisionError" in boom.error or "division" in boom.error.lower()


def test_run_code_timestamp_as_date(tmp_path: Path) -> None:
    tables_dir = tmp_path / "tables"
    _write_tables_dir(tables_dir, {"t": pd.DataFrame({"x": [1]})})
    code = (
        "import pandas as pd\n"
        "_ = tables\n"
        "result = {'type': 'date', 'answer': pd.Timestamp('2026-03-01')}"
    )
    ran = run_code(code, tables_dir, datetime.date(2026, 8, 15))
    assert ran.ok
    assert ran.result == {"type": "date", "answer": "2026-03-01"}
    formatted = _format_result(ran.result)  # type: ignore[arg-type]
    assert getattr(formatted, "type", None) == "date"
    assert getattr(formatted, "answer", None) == "2026-03-01"


def test_tables_dict_missing_key_lists_available(tmp_path: Path) -> None:
    tables_dir = tmp_path / "tables"
    _write_tables_dir(
        tables_dir,
        {
            "SoBanHang.xlsx!don_hang": pd.DataFrame({"a": ["1"]}),
            "kho.csv": pd.DataFrame({"b": ["2"]}),
        },
        aliases={"don_hang": "SoBanHang.xlsx!don_hang", "kho": "kho.csv"},
    )
    ran = run_code(
        "df = tables['missing_table']\nresult = {'type': 'number', 'answer': 0}",
        tables_dir,
        datetime.date(2026, 8, 15),
    )
    assert not ran.ok
    assert ran.error is not None
    assert "missing_table not found; available tables:" in ran.error
    assert "SoBanHang.xlsx!don_hang" in ran.error
    assert "kho.csv" in ran.error


def test_table_aliases_unambiguous_and_skip_collisions() -> None:
    names = [
        "SoBanHang.xlsx!don_hang",
        "kho.csv",
        "Other.xlsx!don_hang",
    ]
    aliases = _build_alias_map(names)
    # after-! "don_hang" collides across two sheets -> skipped
    assert "don_hang" not in aliases
    assert aliases["kho"] == "kho.csv"
    assert aliases["SoBanHang"] == "SoBanHang.xlsx!don_hang"
    assert aliases["SoBanHang!don_hang"] == "SoBanHang.xlsx!don_hang"
    assert aliases["Other"] == "Other.xlsx!don_hang"
    assert aliases["Other!don_hang"] == "Other.xlsx!don_hang"


def test_run_code_uses_aliases(tmp_path: Path) -> None:
    tables_dir = tmp_path / "tables"
    full = "SoBanHang.xlsx!Sheet1"
    _write_tables_dir(
        tables_dir,
        {full: pd.DataFrame({"Thanh tien": ["5.000đ"]})},
        aliases={
            "Sheet1": full,
            "SoBanHang": full,
            "SoBanHang!Sheet1": full,
        },
    )
    ran = run_code(
        (
            "vals = tables['SoBanHang']['Thanh tien'].map(parse_money)\n"
            "result = {'type': 'number', 'answer': int(vals.sum())}"
        ),
        tables_dir,
        datetime.date(2026, 8, 15),
    )
    assert ran.ok
    assert ran.result == {"type": "number", "answer": 5_000}


def test_unicode_result_and_traceback_roundtrip(tmp_path: Path) -> None:
    tables_dir = tmp_path / "tables"
    _write_tables_dir(
        tables_dir,
        {"ban_hang": pd.DataFrame({"Ten": ["Cà phê sữa"]})},
    )
    as_of = datetime.date(2026, 8, 15)
    ok = run_code(
        (
            "ten = tables['ban_hang']['Ten'].iloc[0]\n"
            "result = {'type': 'string', 'answer': f'Doanh thu: {ten}'}"
        ),
        tables_dir,
        as_of,
    )
    assert ok.ok
    assert ok.result == {"type": "string", "answer": "Doanh thu: Cà phê sữa"}

    boom = run_code(
        "_ = tables\nraise ValueError('Không tìm thấy đơn hàng')",
        tables_dir,
        as_of,
    )
    assert not boom.ok
    assert boom.error is not None
    assert "Không tìm thấy đơn hàng" in boom.error


def test_date_timedelta_available_without_import(tmp_path: Path) -> None:
    tables_dir = tmp_path / "tables"
    _write_tables_dir(tables_dir, {"t": pd.DataFrame({"x": ["1"]})})
    ran = run_code(
        (
            "from datetime import datetime\n"
            "_ = tables\n"
            "d = date(2026, 8, 1) + timedelta(days=1)\n"
            "result = {'type': 'date', 'answer': d.isoformat()}"
        ),
        tables_dir,
        datetime.date(2026, 8, 15),
    )
    assert ran.ok
    assert ran.result == {"type": "date", "answer": "2026-08-02"}


def test_pd_available_without_import(tmp_path: Path) -> None:
    tables_dir = tmp_path / "tables"
    _write_tables_dir(
        tables_dir,
        {"t": pd.DataFrame({"Ngay": ["01/08/2026", "02/08/2026"]})},
    )
    ran = run_code(
        (
            "s = tables['t']['Ngay'].map(parse_date)\n"
            "ts = pd.to_datetime(s)\n"
            "result = {'type': 'number', 'answer': int(ts.notna().sum())}"
        ),
        tables_dir,
        datetime.date(2026, 8, 15),
    )
    assert ran.ok
    assert ran.result == {"type": "number", "answer": 2}


def test_prompt_table_keys_line_and_token_budget() -> None:
    assert DIRECT_MAX_TOKENS == 1400
    assert settings.direct_prompt_budget_tokens == 6600
    messages = build_messages(
        "doanh thu",
        datetime.date(2026, 8, 15),
        "# Business\n",
        "## Cách ghi đáp án\nx\n",
        None,
        table_keys=["ban_hang.csv", "kho.csv"],
    )
    user = messages[1]["content"]
    assert "Table keys in `tables`: ban_hang.csv, kho.csv" in user
    system = messages[0]["content"]
    assert "under 40 lines" in system
    assert "use the exact key as listed" in system.casefold() or (
        "exact key as listed" in system
    )
    assert (
        "parse_date returns datetime.date objects (or None): filter with comparisons "
        "against datetime.date values, or convert with pd.to_datetime(series) before "
        "using .dt."
    ) in system
    assert (
        "Use only the table keys and column names exactly as listed; when a KeyError "
        "lists the available keys, pick one of them."
    ) in system


def test_format_number_half_up_and_non_finite() -> None:
    assert _format_number(69726.5) == 69727
    assert _format_number(float("nan")) == "bad_result: not finite"
    assert _format_number(float("inf")) == "bad_result: not finite"
    assert _format_result({"type": "percent", "answer": float("nan")}) == "bad_result: not finite"
    assert _format_result({"type": "list", "answer": [1.0, float("inf")]}) == (
        "bad_result: not finite"
    )


def _write_shop(data_root: Path) -> Path:
    split = data_root / "train"
    (split / "data" / "biz_syn").mkdir(parents=True)
    (split / "rulebook.md").write_text(
        (
            "# Rulebook\n"
            "## Đơn hàng và doanh thu\n"
            "Doanh thu chỉ tính đơn hoàn thành.\n"
            "## Cách ghi đáp án\n"
            "number là số nguyên VND.\n"
        ),
        encoding="utf-8",
    )
    (split / "calendar.csv").write_text(
        "date,weekday,lunar_day,lunar_month,lunar_year,lunar_leap,holiday\n",
        encoding="utf-8",
    )
    biz = split / "data" / "biz_syn"
    (biz / "README.md").write_text("# Shop\nDữ liệu đến 15/08/2026.\n", encoding="utf-8")
    frame = pd.DataFrame(
        {
            "Ngay": ["01/08/2026", "02/08/2026"],
            "Thanh tien": ["110.000đ", "50.000đ"],
            "Trang thai": ["Hoàn thành", "Đã hủy"],
        }
    )
    frame.to_csv(biz / "ban_hang.csv", index=False, encoding="utf-8")
    return biz


@pytest.fixture
def synthetic_shop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(settings, "data_root", tmp_path)
    monkeypatch.setattr(settings, "cache_dir", tmp_path / "cache")
    return _write_shop(tmp_path)


def _revenue_code(table_key: str = "ban_hang") -> str:
    return (
        f"df = tables[{table_key!r}].copy()\n"
        "df['tien'] = df['Thanh tien'].map(parse_money)\n"
        "df['st'] = df['Trang thai'].map(normalize_label)\n"
        "ok = df[df['st'] == 'hoàn thành']\n"
        "result = {'type': 'number', 'answer': int(ok['tien'].sum())}"
    )


def test_direct_system_good_code(synthetic_shop: Path) -> None:
    client = FakeClient([{"in_scope": True, "code": _revenue_code("ban_hang")}])
    system = DirectSystem(client=client)
    try:
        answer, trace = system.answer(
            Question(
                id="q1",
                business_id="biz_syn",
                question="doanh thu bao nhieu",
                as_of_date=datetime.date(2026, 8, 15),
            ),
            business_dir("train", "biz_syn"),
        )
        assert answer.type == "number"
        assert answer.answer == 110_000
        assert trace["abstain_reason"] is None
        assert len(trace["attempts"]) == 1
        user = client.calls[0]["messages"][1]["content"]
        assert "Table keys in `tables`: ban_hang.csv" in user
        bundle = next(iter(system._by_business.values()))
        manifest = json.loads((bundle.tables_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["files"]["0000.pkl"] == "ban_hang.csv"
        assert manifest["aliases"]["ban_hang"] == "ban_hang.csv"
    finally:
        system.close()


def test_direct_system_repairs_then_succeeds(
    synthetic_shop: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "direct_repairs", 2)
    bad = "result = {'type': 'number', 'answer': 1/0}"
    client = FakeClient(
        [
            {"in_scope": True, "code": bad},
            {"in_scope": True, "code": _revenue_code()},
        ]
    )
    system = DirectSystem(client=client)
    try:
        answer, trace = system.answer(
            Question(
                id="q2",
                business_id="biz_syn",
                question="doanh thu",
                as_of_date=datetime.date(2026, 8, 15),
            ),
            business_dir("train", "biz_syn"),
        )
        assert answer.type == "number"
        assert answer.answer == 110_000
        assert len(trace["attempts"]) == 2
        assert trace["llm_calls"] == 2
    finally:
        system.close()


def test_direct_system_out_of_scope(synthetic_shop: Path) -> None:
    client = FakeClient([{"in_scope": False, "code": ""}])
    system = DirectSystem(client=client)
    try:
        answer, trace = system.answer(
            Question(
                id="q3",
                business_id="biz_syn",
                question="cong no",
                as_of_date=datetime.date(2026, 8, 15),
            ),
            business_dir("train", "biz_syn"),
        )
        assert answer.type == "abstain"
        assert trace["abstain_reason"] == "out_of_scope"
    finally:
        system.close()


def test_direct_system_always_failing_exhausts_repairs(
    synthetic_shop: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "direct_repairs", 2)
    client = FakeClient(
        [
            {"in_scope": True, "code": "1/0"},
            {"in_scope": True, "code": "1/0"},
            {"in_scope": True, "code": "1/0"},
        ]
    )
    system = DirectSystem(client=client)
    try:
        answer, trace = system.answer(
            Question(
                id="q4",
                business_id="biz_syn",
                question="doanh thu",
                as_of_date=datetime.date(2026, 8, 15),
            ),
            business_dir("train", "biz_syn"),
        )
        assert answer.type == "abstain"
        assert trace["abstain_reason"] == "code_error"
        assert trace["llm_calls"] == 1 + settings.direct_repairs
    finally:
        system.close()


def test_direct_system_percent_rounded(synthetic_shop: Path) -> None:
    code = "_ = tables\nresult = {'type': 'percent', 'answer': 12.345}"
    client = FakeClient([{"in_scope": True, "code": code}])
    system = DirectSystem(client=client)
    try:
        answer, _trace = system.answer(
            Question(
                id="q5",
                business_id="biz_syn",
                question="ty le",
                as_of_date=datetime.date(2026, 8, 15),
            ),
            business_dir("train", "biz_syn"),
        )
        assert answer.type == "percent"
        assert answer.answer == 12.35
    finally:
        system.close()


def test_direct_system_llm_unavailable(synthetic_shop: Path) -> None:
    client = FakeClient([LLMUnavailable("down")])
    system = DirectSystem(client=client)
    try:
        answer, trace = system.answer(
            Question(
                id="q6",
                business_id="biz_syn",
                question="doanh thu",
                as_of_date=datetime.date(2026, 8, 15),
            ),
            business_dir("train", "biz_syn"),
        )
        assert answer.type == "abstain"
        assert trace["abstain_reason"] == "llm_unavailable"
    finally:
        system.close()


def test_direct_system_llm_error_repairs_then_llm_error_abstain(
    synthetic_shop: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "direct_repairs", 1)
    client = FakeClient(
        [
            LLMError("temporary glitch"),
            {"in_scope": True, "code": _revenue_code()},
        ]
    )
    system = DirectSystem(client=client)
    try:
        answer, trace = system.answer(
            Question(
                id="q7",
                business_id="biz_syn",
                question="doanh thu",
                as_of_date=datetime.date(2026, 8, 15),
            ),
            business_dir("train", "biz_syn"),
        )
        assert answer.type == "number"
        assert answer.answer == 110_000
        assert trace["llm_calls"] == 2
        assert trace["attempts"][0]["error"].startswith("llm_error:")
    finally:
        system.close()

    client2 = FakeClient([LLMError("a"), LLMError("b")])
    system2 = DirectSystem(client=client2)
    try:
        answer2, trace2 = system2.answer(
            Question(
                id="q8",
                business_id="biz_syn",
                question="doanh thu",
                as_of_date=datetime.date(2026, 8, 15),
            ),
            business_dir("train", "biz_syn"),
        )
        assert answer2.type == "abstain"
        assert trace2["abstain_reason"] == "llm_error"
    finally:
        system2.close()


def test_direct_system_model_abstain(synthetic_shop: Path) -> None:
    code = "_ = tables\nresult = {'type': 'abstain', 'answer': None}"
    client = FakeClient([{"in_scope": True, "code": code}])
    system = DirectSystem(client=client)
    try:
        answer, trace = system.answer(
            Question(
                id="q9",
                business_id="biz_syn",
                question="doanh thu",
                as_of_date=datetime.date(2026, 8, 15),
            ),
            business_dir("train", "biz_syn"),
        )
        assert answer.type == "abstain"
        assert trace["abstain_reason"] == "model_abstain"
    finally:
        system.close()


def test_direct_system_model_unanswerable(synthetic_shop: Path) -> None:
    code = (
        "_ = tables\n"
        "result = {'type': 'unanswerable', 'reason': 'thieu cot'}"
    )
    client = FakeClient([{"in_scope": True, "code": code}])
    system = DirectSystem(client=client)
    try:
        answer, trace = system.answer(
            Question(
                id="q9b",
                business_id="biz_syn",
                question="doanh thu",
                as_of_date=datetime.date(2026, 8, 15),
            ),
            business_dir("train", "biz_syn"),
        )
        assert answer.type == "abstain"
        assert trace["abstain_reason"] == "model_unanswerable"
        assert trace["result"] == {"type": "unanswerable", "reason": "thieu cot"}
    finally:
        system.close()


def test_direct_system_nan_goes_to_repair(
    synthetic_shop: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "direct_repairs", 1)
    bad = "_ = tables\nresult = {'type': 'number', 'answer': float('nan')}"
    client = FakeClient(
        [
            {"in_scope": True, "code": bad},
            {"in_scope": True, "code": _revenue_code()},
        ]
    )
    system = DirectSystem(client=client)
    try:
        answer, trace = system.answer(
            Question(
                id="q10",
                business_id="biz_syn",
                question="doanh thu",
                as_of_date=datetime.date(2026, 8, 15),
            ),
            business_dir("train", "biz_syn"),
        )
        assert answer.type == "number"
        assert answer.answer == 110_000
        assert "not finite" in (trace["attempts"][0]["error"] or "")
    finally:
        system.close()

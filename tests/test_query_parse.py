from __future__ import annotations

import datetime
import json
from typing import Any

import duckdb
import pytest

from datamate.config import settings
from datamate.query.context import ShopContext, load_shop_context
from datamate.query.convert import ConvertError, to_query_spec
from datamate.query.parse import parse_question
from datamate.query.parse_schema import parsed_question_schema
from datamate.query.prompts import FEW_SHOT_SETS, build_messages
from datamate.query.spec import CompareQuery, MultiQuery, ValueQuery
from datamate.rules import RuleConfig
from datamate.timeres import LunarCalendar, MonthThisExpr, RangeExpr, WeekLastExpr

AS_OF = datetime.date(2026, 8, 19)


class FakeClient:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.max_tokens: list[int | None] = []
        self.prompts: list[str] = []
        self.sent: list[list[dict[str, str]]] = []

    def count_tokens(self, text: str) -> int:
        self.prompts.append(text)
        return 1

    def chat_json(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        schema_name: str,
        max_tokens: int | None = None,
        required_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        del schema, required_keys
        self.sent.append(messages)
        assert schema_name == "ParsedQuestion"
        self.max_tokens.append(max_tokens)
        if not self._responses:
            raise AssertionError("unexpected chat_json call")
        return self._responses.pop(0)


def _shop(**overrides: Any) -> ShopContext:
    values: dict[str, Any] = {
        "payment_methods": ["Momo", "Tiền mặt"],
        "channels": ["tại quầy"],
        "segments": ["khách quen"],
        "categories": ["bánh ngọt"],
        "product_names": ["Coca Cola 330ml", "Bánh mì thịt"],
        "has_order_ts": True,
        "data_start": datetime.date(2024, 1, 1),
        "data_end": datetime.date(2024, 12, 31),
    }
    values.update(overrides)
    return ShopContext(**values)


def _filters(**overrides: Any) -> dict[str, Any]:
    filters: dict[str, Any] = {
        "payment_method": [],
        "channel": [],
        "customer_segment": [],
        "category": [],
        "product": [],
        "shift": None,
    }
    filters.update(overrides)
    return filters


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


def _convert(parsed: dict[str, Any], as_of: datetime.date = AS_OF) -> Any:
    return to_query_spec(parsed, as_of, RuleConfig(), LunarCalendar({}))


def test_schema_enums_come_from_the_shop_context() -> None:
    ctx = _shop()
    schema = parsed_question_schema(ctx, ["Bánh mì thịt"])
    filters = schema["properties"]["filters"]["properties"]
    assert filters["payment_method"]["items"]["enum"] == ["Momo", "Tiền mặt"]
    assert filters["product"]["items"]["enum"] == ["Bánh mì thịt"]
    assert filters["shift"]["enum"] == ["morning", "afternoon", None]
    assert list(schema["properties"]) == [
        "in_scope",
        "kind",
        "metric",
        "periods",
        "compare_kind",
        "op",
        "filters",
        "filters_b",
        "multi_filters",
        "direction",
        "granularity",
        "mode",
    ]
    dumped = json.dumps(schema)
    assert "Visa" not in dumped
    assert "cash" not in dumped

    blind = parsed_question_schema(_shop(has_order_ts=False), [])
    assert blind["properties"]["filters"]["properties"]["shift"]["enum"] == [None]
    assert blind["properties"]["filters"]["properties"]["product"]["maxItems"] == 0


def test_product_candidates_use_token_overlap_and_ignore_misses() -> None:
    ctx = _shop(
        product_names=[
            "Coca Cola 330ml",
            "Coca",
            "Bánh mì thịt",
            "Nhiệt kế điện tử Microlife MT500",
        ]
    )
    assert ctx.product_candidates("dthu coca thang 5") == ["Coca", "Coca Cola 330ml"]
    assert ctx.product_candidates("banh mi") == ["Bánh mì thịt"]
    assert ctx.product_candidates("hom nay ban duoc bao nhieu don") == []
    phone = "dthu các đơn bán qua điện thoaij chiếm bao nhiêu %"
    assert "Nhiệt kế điện tử Microlife MT500" not in ctx.product_candidates(phone)
    assert ctx.product_candidates(phone) == []


def test_convert_previous_period_filters_b_multi_and_bad_count() -> None:
    compared = _convert(
        _parsed(
            kind="compare",
            periods=[{"kind": "month_this"}],
            compare_kind="previous_period",
            op="pct_change",
        )
    )
    assert isinstance(compared, CompareQuery)
    assert isinstance(compared.a.period, MonthThisExpr)
    assert isinstance(compared.b.period, RangeExpr)
    assert compared.b.period.start == datetime.date(2026, 7, 1)
    assert compared.b.period.end == datetime.date(2026, 7, 19)

    shifts = _convert(
        _parsed(
            kind="compare",
            periods=[{"kind": "week_last"}],
            op="pct_change",
            filters=_filters(shift="afternoon"),
            filters_b=_filters(shift="morning"),
        )
    )
    assert isinstance(shifts, CompareQuery)
    assert isinstance(shifts.a.period, WeekLastExpr)
    assert isinstance(shifts.b.period, WeekLastExpr)
    assert shifts.a.filters.shift == "afternoon"
    assert shifts.b.filters.shift == "morning"

    multi = _convert(
        _parsed(
            kind="multi",
            periods=[{"kind": "month", "month": 4, "year": 2024}],
            multi_filters=[
                _filters(payment_method=["Momo"]),
                _filters(payment_method=["Tiền mặt"]),
            ],
        )
    )
    assert isinstance(multi, MultiQuery)
    assert [query.filters.payment_method for query in multi.queries] == [
        ["Momo"],
        ["Tiền mặt"],
    ]

    with pytest.raises(ConvertError):
        _convert(_parsed(periods=[{"kind": "month_this"}, {"kind": "today"}]))


def test_convert_tolerates_consistent_redundancy_only() -> None:
    days = _convert(_parsed(periods=[{"kind": "last_n_days", "n": 30, "n_days": 30}]))
    assert isinstance(days, ValueQuery)
    assert days.query.period.n == 30
    with pytest.raises(ConvertError):
        _convert(_parsed(periods=[{"kind": "last_n_days", "n": 30, "n_days": 7}]))

    april_2026 = {"kind": "month", "month": 4, "year": 2026}
    april_2025 = {"kind": "month", "month": 4, "year": 2025}
    for periods in ([april_2026, april_2025], [april_2025, april_2026]):
        spelled = _convert(
            _parsed(
                kind="compare",
                periods=periods,
                compare_kind="same_period_last_year",
                op="pct_change",
            )
        )
        assert isinstance(spelled, CompareQuery)
        assert spelled.a.period.year == 2026
        assert isinstance(spelled.b.period, RangeExpr)
        assert spelled.b.period.start == datetime.date(2025, 4, 1)
    with pytest.raises(ConvertError):
        _convert(
            _parsed(
                kind="compare",
                periods=[april_2026, {"kind": "month", "month": 3, "year": 2025}],
                compare_kind="same_period_last_year",
                op="pct_change",
            )
        )

    with pytest.raises(ConvertError):
        _convert(
            _parsed(
                kind="compare",
                periods=[{"kind": "week_last"}, {"kind": "week_this"}],
                compare_kind="explicit",
                op="pct_change",
                filters=_filters(shift="afternoon"),
                filters_b=_filters(shift="morning"),
            )
        )


def _grounded(parsed: dict[str, Any], question: str) -> Any:
    return to_query_spec(parsed, AS_OF, RuleConfig(), LunarCalendar({}), question)


def test_filter_values_must_be_named_in_the_question() -> None:
    segment = _parsed(filters=_filters(customer_segment=["Cá nhân"]))
    for plain in (
        "Tháng 5/2025 cửa hàng bán đc tổng cộng bao nhiêu tiền?",
        "Tuần trước ca chiều bán được bao nhiêu",
    ):
        assert _grounded(segment, plain).query.filters.customer_segment == []
    named = _grounded(segment, "doanh thu khach ca nhan thang nay")
    assert named.query.filters.customer_segment == ["Cá nhân"]
    with pytest.raises(ConvertError, match="in_scope false"):
        _grounded(segment, "doanh thu khach si thang nay")

    near = _parsed(filters=_filters(category=["vitamin khoáng chất"]))
    with pytest.raises(ConvertError, match="in_scope false"):
        _grounded(near, "Doanh thu thuốc hoạt chất Vitamin C tháng này là bao nhiêu?")
    unrelated = _parsed(filters=_filters(category=["thực phẩm chức năng"]))
    with pytest.raises(ConvertError, match="in_scope false"):
        _grounded(unrelated, "doanh thu thuoc ke don thang 4/2025")
    cash = _parsed(filters=_filters(payment_method=["cash"], channel=["phone"]))
    assert _grounded(cash, "dthu cac don tra bang tm qua dien thoaij thang nay")

    coded = _parsed(
        kind="multi",
        multi_filters=[
            _filters(payment_method=["transfer"]),
            _filters(channel=["store"]),
            _filters(payment_method=["Thu hộ (COD)"]),
            _filters(channel=["zalo_oa"]),
        ],
    )
    assert _grounded(coded, "dthu don chuyen khoan, ban tai quay, tra COD, qua Zalo thang nay")
    afternoon = _parsed(filters=_filters(shift="afternoon"))
    with pytest.raises(ConvertError, match="shift"):
        _grounded(afternoon, "thang nay ban duoc bao nhieu")


def test_year_wording_must_back_ytd_and_last_year_baselines() -> None:
    month_to_date = "từ đầu tháng này đến hết hôm nay, doanh thu hơn hay kém cùng kỳ tháng trc mấy %"
    ytd = _parsed(
        kind="compare",
        periods=[{"kind": "ytd"}],
        compare_kind="same_period_last_year",
        op="pct_change",
    )
    with pytest.raises(ConvertError, match="previous_period"):
        _grounded(ytd, month_to_date)
    with pytest.raises(ConvertError, match="ytd"):
        _grounded(_parsed(periods=[{"kind": "ytd"}]), month_to_date)
    assert _grounded(_parsed(periods=[{"kind": "ytd"}]), "tu 1/1 den hom nay ban bn")
    assert _grounded(ytd, "tu dau nam den nay so voi cung ky nam ngoai tang bn %")


def test_convert_settles_stray_fields_and_common_shape_slips() -> None:
    ytd = _convert(_parsed(periods=[{"kind": "ytd", "year": 2026, "date": "2026-08-19"}]))
    assert isinstance(ytd, ValueQuery)
    with pytest.raises(ConvertError):
        _convert(_parsed(periods=[{"kind": "ytd", "year": 2025}]))

    this_month = _convert(_parsed(periods=[{"kind": "month_this", "month": 8, "year": 2026}]))
    assert isinstance(this_month.query.period, MonthThisExpr)
    written = _convert(_parsed(periods=[{"kind": "month_this", "month": 2, "year": 2026}]))
    assert written.query.period.kind == "month"
    assert (written.query.period.month, written.query.period.year) == (2, 2026)
    with pytest.raises(ConvertError):
        _convert(_parsed(periods=[{"kind": "month_this", "month": 2}]))

    both_years = _convert(_parsed(periods=[{"kind": "month", "month": 2, "year": 2025, "year_rel": -1}]))
    assert both_years.query.period.year == 2025

    july = {"kind": "month", "month": 7, "year": 2026}
    previous = _convert(
        _parsed(
            kind="compare",
            periods=[july, {"kind": "month_last", "year": 2025}],
            compare_kind="previous_period",
            op="pct_change",
        )
    )
    assert isinstance(previous, CompareQuery)
    assert previous.b.period.start == datetime.date(2026, 6, 1)

    shifts = _convert(
        _parsed(
            kind="compare",
            periods=[{"kind": "week_last"}],
            compare_kind="previous_period",
            op="pct_change",
            filters=_filters(shift="afternoon"),
            filters_b=_filters(shift="morning"),
        )
    )
    assert isinstance(shifts.b.period, WeekLastExpr)
    assert shifts.b.filters.shift == "morning"

    pair = _convert(
        _parsed(
            kind="series",
            filters=_filters(payment_method=["Momo"]),
            filters_b=_filters(payment_method=["Tiền mặt"]),
        )
    )
    assert isinstance(pair, MultiQuery)
    phone, store = _filters(channel=["phone"]), _filters(channel=["store"])
    restated = _convert(
        _parsed(kind="multi", filters=phone, filters_b=store, multi_filters=[phone, store])
    )
    assert [query.filters.channel for query in restated.queries] == [["phone"], ["store"]]
    with pytest.raises(ConvertError, match="only valid for compare"):
        other = _filters(channel=["zalo_oa"])
        _convert(_parsed(kind="multi", filters_b=other, multi_filters=[phone, store]))
    typo = _grounded(
        _parsed(
            kind="compare",
            periods=[{"kind": "week_last"}],
            op="pct_change",
            filters=_filters(shift="afternoon"),
            filters_b=_filters(shift="morning"),
        ),
        "ee tuần rồi dthu ca chiều chênh ca sangs nhiêu %",
    )
    assert typo.b.filters.shift == "morning"
    with pytest.raises(ConvertError, match="name a filter"):
        _convert(_parsed(kind="multi", multi_filters=[_filters(), _filters()]))
    with pytest.raises(ConvertError, match="differ"):
        same = _filters(payment_method=["Momo"])
        _convert(_parsed(kind="multi", multi_filters=[same, same]))

    today = _convert(_parsed(periods=[{"kind": "today", "date": "2026-08-19", "year": 2026}]))
    assert today.query.period.kind == "today"
    month_to_date = _grounded(
        _parsed(
            kind="compare",
            periods=[{"kind": "month_this"}, {"kind": "month_last"}],
            compare_kind="previous_period",
            op="pct_change",
        ),
        "tu dau thang nay den hom nay so voi cung ky thang truoc tang bn %",
    )
    assert month_to_date.b.period.end == datetime.date(2026, 7, 19)

    busiest = _grounded(
        _parsed(kind="argmax", direction="increasing"),
        "thang 2/2025 ngay nao cua hang ban dc nhieu tien nhat",
    )
    assert (busiest.granularity, busiest.mode) == ("day", "max")


def test_threshold_questions_are_out_of_scope_without_a_model_call() -> None:
    client = FakeClient([])
    result = parse_question(
        "Trong tháng 6/2025, bao nhiêu đơn có tiền hàng đạt 41.000 đồng trở lên?",
        _shop(),
        client,
        RuleConfig(),
        LunarCalendar({}),
        AS_OF,
    )
    assert result.reason == "out_of_scope"
    assert client.sent == []


def test_votes_agree_when_periods_resolve_to_the_same_dates() -> None:
    ctx = _shop()
    rules = RuleConfig()
    cal = LunarCalendar({})
    explicit = _parsed(
        kind="compare",
        periods=[
            {"kind": "month", "month": 4, "year": 2026},
            {"kind": "month", "month": 4, "year": 2025},
        ],
        compare_kind="explicit",
        op="pct_change",
    )
    relative = _parsed(
        kind="compare",
        periods=[{"kind": "month", "month": 4, "year": 2026}],
        compare_kind="same_period_last_year",
        op="pct_change",
    )
    compared = parse_question(
        "4/2026 vs cung ky nam truoc", ctx, FakeClient([explicit, relative]), rules, cal, AS_OF
    )
    assert isinstance(compared.spec, CompareQuery)

    ytd = _parsed(periods=[{"kind": "ytd"}])
    year = _parsed(periods=[{"kind": "year", "year": 2026}])
    clipped = parse_question("nam nay", ctx, FakeClient([ytd, year]), rules, cal, AS_OF)
    assert isinstance(clipped.spec, ValueQuery)

    last_year = _parsed(periods=[{"kind": "year", "year": 2025}])
    split = parse_question("nam nay", ctx, FakeClient([ytd, last_year]), rules, cal, AS_OF)
    assert split.reason == "disagreement"


def test_parse_votes_agree_disagree_and_retry() -> None:
    ctx = _shop(payment_methods=["chuyen khoan"])
    rules = RuleConfig()
    cal = LunarCalendar({})
    good = _parsed()
    client = FakeClient([good, good])
    agreed = parse_question("thang nay ban duoc bao nhieu", ctx, client, rules, cal, AS_OF)
    assert settings.parse_votes == 2
    assert isinstance(agreed.spec, ValueQuery)
    assert agreed.reason is None
    assert client.max_tokens == [400, 400]

    other = _parsed(periods=[{"kind": "month", "month": 6, "year": 2024}])
    split = FakeClient([good, other])
    disagreed = parse_question("thang nay", ctx, split, rules, cal, AS_OF, votes=2)
    assert disagreed.spec is None
    assert disagreed.reason == "disagreement"

    closed = FakeClient([_parsed(in_scope=False)])
    outside = parse_question("khach no bao nhieu", ctx, closed, rules, cal, AS_OF, votes=1)
    assert outside.spec is None
    assert outside.reason == "out_of_scope"

    fixed = FakeClient(
        [
            _parsed(periods=[{"kind": "month_this"}, {"kind": "today"}]),
            good,
        ]
    )
    repaired = parse_question("thang nay", ctx, fixed, rules, cal, AS_OF, votes=1)
    assert isinstance(repaired.spec, ValueQuery)
    assert repaired.reason is None
    assert repaired.attempts[0].error is not None
    assert repaired.attempts[1].error is None


def test_few_shot_sets_are_disjoint_and_cover_the_shapes() -> None:
    questions = [{example.question for example in group} for group in FEW_SHOT_SETS]
    assert questions[0].isdisjoint(questions[1])
    assert all(10 <= len(group) <= 14 for group in questions)
    parsed = [example.parsed for group in FEW_SHOT_SETS for example in group]
    assert sum(1 for row in parsed if row["in_scope"] is False) >= 3
    assert {row["kind"] for row in parsed} >= {
        "value",
        "compare",
        "series",
        "multi",
        "trend",
        "argmax",
    }
    compare_rows = [row for row in parsed if row["kind"] == "compare"]
    assert {row["compare_kind"] for row in compare_rows} >= {
        "explicit",
        "previous_period",
        "same_period_last_year",
        "before_tet_same_length",
    }
    assert any(row["filters_b"] is not None for row in compare_rows)
    period_kinds = {
        period["kind"] for row in parsed for period in row["periods"] if row["in_scope"]
    }
    assert {
        "month_this",
        "yesterday",
        "last_n_days",
        "week_last",
        "quarter",
        "tet",
        "before_tet",
    } <= period_kinds
    shares = [row for row in parsed if row["metric"] == "revenue_share" and row["in_scope"]]
    assert any(row["filters"]["payment_method"] for row in shares)
    assert any(row["filters"]["customer_segment"] for row in shares)
    assert any(row["filters"]["channel"] for row in shares)
    assert any(row["kind"] == "value" and row["filters"]["category"] for row in parsed)
    assert any(
        row["kind"] == "compare"
        and row["compare_kind"] == "explicit"
        and row["op"] == "pct_change"
        and [period.get("month") for period in row["periods"]] == [4, 4]
        and [period.get("year") for period in row["periods"]] == [2024, 2023]
        for row in parsed
    )
    rendered = "\n".join(
        message["content"] for message in build_messages("xin chao", AS_OF, _shop(), 0)
    )
    assert "biz_" not in rendered
    assert "questions.jsonl" not in rendered


def test_prompt_budget_drops_few_shots_from_the_end() -> None:
    class TightClient(FakeClient):
        def count_tokens(self, text: str) -> int:
            self.prompts.append(text)
            return text.count("\nQuestion:") * 5000

    client = TightClient([_parsed()])
    result = parse_question(
        "xin chao Hoa Mai",
        _shop(),
        client,
        RuleConfig(),
        LunarCalendar({}),
        AS_OF,
        votes=1,
    )
    assert isinstance(result.spec, ValueQuery)
    assert result.attempts[0].few_shots_dropped == len(FEW_SHOT_SETS[0])
    sent = client.sent[-1]
    assert sent[0]["role"] == "system"
    assert sent[-1]["role"] == "user"
    assert "xin chao Hoa Mai" in sent[-1]["content"]
    assert all("Hoa Mai thang nay gom duoc khoan nao" not in message["content"] for message in sent)


def test_load_shop_context_uses_frequency_and_the_common_spelling() -> None:
    con = duckdb.connect()
    con.execute(
        """
        CREATE TABLE orders (
            order_date DATE, order_ts TIMESTAMP, payment_method VARCHAR,
            channel VARCHAR, customer_key VARCHAR
        );
        CREATE TABLE customers (customer_key VARCHAR, segment VARCHAR);
        CREATE TABLE products (product_key VARCHAR, category VARCHAR);
        CREATE TABLE order_lines (product_key VARCHAR, product_name VARCHAR);
        """
    )
    con.execute(
        "INSERT INTO orders VALUES "
        "('2024-05-01', '2024-05-01 08:00:00', 'Momo', 'grab', 'c1'), "
        "('2024-05-02', '2024-05-02 15:00:00', 'Momo', 'grab', 'c1'), "
        "('2024-05-03', NULL, 'Tiền mặt', 'grab', 'c2')"
    )
    con.execute("INSERT INTO customers VALUES ('c1', 'quen'), ('c2', 'moi')")
    con.execute("INSERT INTO products VALUES ('p1', 'tra')")
    con.execute(
        "INSERT INTO order_lines VALUES ('p1', 'Tra dao'), ('p1', 'Trà đào'), ('p1', 'Trà đào')"
    )
    ctx = load_shop_context(con)
    con.close()
    assert ctx.payment_methods == ["Momo", "Tiền mặt"]
    assert ctx.segments == ["quen", "moi"]
    assert ctx.categories == ["tra"]
    assert ctx.product_names == ["Trà đào"]
    assert ctx.has_order_ts is True
    assert ctx.data_start == datetime.date(2024, 5, 1)
    assert ctx.data_end == datetime.date(2024, 5, 3)


def test_product_values_must_be_named_in_the_question() -> None:
    sting = _parsed(periods=[{"kind": "ytd"}], filters=_filters(product=["Sting dâu 330ml"]))
    stray = _grounded(sting, "Doanh thu từ đầu năm tới hết hôm nay là bao nhiêu?")
    assert stray.query.filters.product == []  # đầu is not dâu
    with pytest.raises(ConvertError):  # without diacritics "dau" may be dâu: abstain
        _grounded(sting, "doanh thu tu dau nam toi het hom nay la bao nhieu")

    named = _grounded(
        _parsed(filters=_filters(product=["Trà vải (M)", "Trà vải (L)"])),
        "thang nay mat hang tra vai thu ve bao nhieu",
    )
    assert named.query.filters.product == ["Trà vải (M)", "Trà vải (L)"]

    with pytest.raises(ConvertError, match="category"):
        _grounded(
            _parsed(filters=_filters(product=["Lẩu dê"])),
            "doanh thu nhom hang lau thang nay la bao nhieu",
        )


def test_drug_attributes_and_sign_thresholds_are_out_of_scope() -> None:
    for question in (
        "Doanh thu thuốc hoạt chất Paracetamol tháng 9/2025 là bao nhiêu?",
        "Trừ thuốc kê đơn ra, tháng 8/2026 các mặt hàng còn lại bán được bao nhiêu tiền?",
        "trong thang 3 nam 2026, so ngay co dthu ≥ 53.000.000 dong la bn",
        "so don co tien hang >= 100000 thang nay",
    ):
        client = FakeClient([])
        result = parse_question(
            question, _shop(), client, RuleConfig(), LunarCalendar({}), AS_OF
        )
        assert result.reason == "out_of_scope", question
        assert client.sent == []

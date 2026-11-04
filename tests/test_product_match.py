from __future__ import annotations

import datetime
from typing import Any

import pytest

from datamate.config import settings
from datamate.query.convert import ConvertError
from datamate.query.ext.convert import to_ext_query_spec
from datamate.query.ext.spec import QtySoldQuery
from datamate.query.grounding import QuestionText
from datamate.query.parse_schema import PERIOD_FIELDS
from datamate.query.product_match import product_matcher

AS_OF = datetime.date(2026, 6, 30)
CATALOG = ["Má heo nướng", "Má heo nướng mật ong", "Má heo luộc", "Gà nướng muối ớt"]


class _Client:
    def __init__(self, p_yes: float) -> None:
        self.p_yes = p_yes
        self.asked: list[str] = []

    def chat_choice_probs(
        self, messages: list[dict[str, str]], choices: list[str]
    ) -> dict[str, Any]:
        del choices
        self.asked.append(messages[-1]["content"])
        return {"yes": self.p_yes, "no": 1 - self.p_yes}


def _qty_sold(product: str) -> dict[str, object]:
    period: dict[str, object] = {name: None for name in PERIOD_FIELDS}
    period.update(kind="month", month=6, year=2026)
    return {"family": "qty_sold", "product": product, "period": period}


@pytest.fixture
def check_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "product_synonym_check", True)


@pytest.mark.usefixtures("check_on")
def test_synonym_product_is_accepted_when_the_check_says_yes() -> None:
    question = "thang 6/2026 ban duoc may dia ma lon nuong"
    client = _Client(0.97)
    same = product_matcher(question, CATALOG, client)
    spec = to_ext_query_spec(_qty_sold("Má heo nướng"), AS_OF, question, same)
    assert isinstance(spec, QtySoldQuery)
    assert spec.product == "Má heo nướng"
    # Products that fit the question as well as the pick are shown; asked once per name.
    assert "Má heo nướng mật ong" in client.asked[0]
    assert "Má heo luộc" not in client.asked[0]
    to_ext_query_spec(_qty_sold("Má heo nướng"), AS_OF, question, same)
    assert len(client.asked) == 1


@pytest.mark.usefixtures("check_on")
def test_synonym_product_is_rejected_below_the_threshold() -> None:
    question = "thang 6/2026 ban duoc may dia ma lon nuong"
    same = product_matcher(question, CATALOG, _Client(0.6))
    with pytest.raises(ConvertError, match="not grounded"):
        to_ext_query_spec(_qty_sold("Má heo nướng"), AS_OF, question, same)


@pytest.mark.usefixtures("check_on")
def test_product_sharing_no_word_is_never_asked() -> None:
    client = _Client(0.99)
    text = QuestionText("thang 6 ban duoc may phan com tam", product_matcher("", CATALOG, client))
    assert not text.mentions_product("Gà nướng muối ớt")
    assert client.asked == []


def test_check_off_keeps_word_grounding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "product_synonym_check", False)
    assert product_matcher("ma lon nuong", CATALOG, _Client(0.99)) is None

from __future__ import annotations

from typing import Any

from datamate.parse.plain import PLAIN_FEW_SHOTS, keeps_facts, plain_question


class _Client:
    def __init__(self, text: str) -> None:
        self.text = text

    def chat_json(self, messages: list[dict[str, str]], *args: Any, **kwargs: Any) -> Any:
        return {"question": self.text}


def test_plain_rewrite_keeps_numbers_and_products() -> None:
    q = "th 3/2026 Sữa Hạt Sen 1L bán đc mí hộp zậy"
    good = "Tháng 3/2026 bán được bao nhiêu hộp Sữa Hạt Sen 1L?"
    assert plain_question(q, _Client(good), ["Sữa Hạt Sen 1L"]) == good
    # A dropped or changed number, a lost product, or no change at all is refused.
    assert plain_question(q, _Client("Tháng 4/2026 bán được bao nhiêu hộp Sữa Hạt Sen 1L?"), []) is None
    assert plain_question(q, _Client("Tháng 3/2026 bán được bao nhiêu hộp sữa 1L?"), ["Sữa Hạt Sen 1L"]) is None
    assert not keeps_facts(q, q, [])


def test_plain_few_shots_keep_their_own_facts() -> None:
    for raw, plain in PLAIN_FEW_SHOTS:
        assert keeps_facts(raw, plain, [])

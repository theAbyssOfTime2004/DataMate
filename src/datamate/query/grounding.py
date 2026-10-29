"""Check that what the parser chose is written in the question.

A small model given a shop's filter values tends to use them even when the question never
names them (a customer segment on a plain revenue question, a nearby category for an
active ingredient). Every filter value, the year-based periods, and the shift must be
traceable to the question's own words, compared without diacritics.
"""

from __future__ import annotations

import re
import unicodedata

_TOKEN = re.compile(r"[a-z0-9]+")

# Vietnamese wording for the coded values shops store. Keys and phrases are already
# accent-stripped token strings; a value also matches when all of its own tokens appear.
_ALIASES: dict[str, tuple[str, ...]] = {
    "cash": ("tien mat", "tm"),
    "tien mat": ("cash", "tm"),
    "transfer": ("chuyen khoan", "ck", "ngan hang"),
    "chuyen khoan": ("ck", "transfer", "ngan hang"),
    "ewallet": ("vi dien tu", "vi dt", "momo", "zalopay", "zalo pay", "vnpay", "shopeepay"),
    "vi dien tu": ("ewallet", "vi dt", "momo", "zalopay", "zalo pay", "vnpay", "shopeepay"),
    "cod": ("thu ho",),
    "thu ho cod": ("cod", "thu ho"),
    "debt": ("ghi no", "cong no", "ghi so", "mua chiu"),
    "ghi no": ("debt", "cong no", "ghi so", "mua chiu"),
    "store": ("tai quay", "tai cua hang", "tai tiem", "tai quan", "tai cho", "offline"),
    "zalo oa": ("zalo",),
    "phone": ("dien thoai", "goi dien", "hotline"),
    "le": ("khach le",),
    "than thiet": ("khach quen", "thanh vien"),
    "si": ("khach si", "ban buon", "dai ly"),
}
_SHIFT_WORDS = {"morning": "sang", "afternoon": "chieu"}
# Words that show the question is about that kind of filter. A value the question does not
# mention, in a question without these words, is a stray guess and is dropped; otherwise
# the question may hold a condition the shop's values cannot express.
_CUES: dict[str, tuple[str, ...]] = {
    "customer_segment": ("khach",),
    "payment_method": ("tra", "thanh toan", "vi"),
    "channel": ("kenh", "qua", "tai", "online", "offline"),
    "category": ("nhom", "loai", "thuoc", "mon", "dong", "hang", "nganh"),
}
_GENERIC = frozenset({"ca", "khach", "nhom", "hang", "loai", "thu", "le"})
_YEAR_WORDS = ("nam", "yoy")
_JAN_FIRST = re.compile(r"(?<!\d)0?1/0?1(?![\d/])")
# "đơn từ 130.000 đồng trở lên", "khách mua ít nhất 5 đơn", "nợ trên 15 ngày": a threshold
# on an order, customer or day. No metric filter expresses one.
_THRESHOLD = re.compile(
    r"\b(?:tro len|tro xuong)\b|\b(?:it nhat|toi thieu|toi da|tren|duoi|vuot qua|vuot) \d"
)


def fold(text: str) -> str:
    """Lowercase, no diacritics, đ as d, punctuation as spaces, single-spaced."""
    nfd = unicodedata.normalize("NFD", text.casefold())
    bare = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn").replace("đ", "d")
    return " ".join(_TOKEN.findall(bare))


def _same_word(written: str, wanted: str) -> bool:
    """Exact, or one stray trailing letter on a longer word ("thoaij", "mooix")."""
    return written == wanted or (
        len(wanted) >= 3 and len(written) == len(wanted) + 1 and written.startswith(wanted)
    )


class QuestionText:
    def __init__(self, question: str) -> None:
        self.raw = question
        self.folded = fold(question)
        self.words = self.folded.split()
        self.tokens = frozenset(self.words)

    def has_word(self, wanted: str) -> bool:
        return any(_same_word(word, wanted) for word in self.words)

    def has_phrase(self, phrase: str) -> bool:
        wanted = phrase.split()
        size = len(wanted)
        return any(
            all(_same_word(self.words[i + j], wanted[j]) for j in range(size))
            for i in range(len(self.words) - size + 1)
        )

    def mentions_value(self, value: str) -> bool:
        folded = fold(value)
        if not folded:
            return False
        if all(self.has_word(word) for word in folded.split()):
            return True
        return any(self.has_phrase(alias) for alias in _ALIASES.get(folded, ()))

    def is_stray(self, field: str, value: str) -> bool:
        """An unmentioned value that shares no specific word with the question, which has
        no word pointing at this kind of filter either."""
        specific = [word for word in fold(value).split() if word not in _GENERIC]
        if any(self.has_word(word) for word in specific):
            return False
        return not any(self.has_phrase(cue) for cue in _CUES.get(field, ()))

    def mentions_shift(self, shift: str) -> bool:
        word = _SHIFT_WORDS.get(shift)
        return word is not None and word in self.tokens

    def mentions_year(self) -> bool:
        """năm / yoy: a same-period-last-year baseline or a year-to-date period."""
        return any(word in self.tokens for word in _YEAR_WORDS)

    def mentions_jan_first(self) -> bool:
        return bool(_JAN_FIRST.search(self.raw))

    def has_threshold(self) -> bool:
        return bool(_THRESHOLD.search(self.folded))

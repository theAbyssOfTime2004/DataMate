from __future__ import annotations

import datetime
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from datamate.parse.text import normalize_label

_VALUE_LIMIT = 40
_NUMBER_OR_UNIT = re.compile(r"^\d+[a-z]*$")
_STOPWORDS = frozenset(
    {
        "thang",
        "nam",
        "ngay",
        "tuan",
        "quy",
        "doanh",
        "thu",
        "dthu",
        "ban",
        "duoc",
        "dc",
        "bao",
        "nhieu",
        "bn",
        "bnhiu",
        "don",
        "hang",
        "cac",
        "cua",
        "tong",
        "tien",
        "la",
        "trong",
        "chiem",
        "phan",
        "tram",
        "so",
        "voi",
        "va",
        "nhat",
        "cao",
        "thap",
        "khach",
        "nhom",
        "ca",
        "sang",
        "chieu",
    }
)


def _strip_accents(text: str) -> str:
    nfd = unicodedata.normalize("NFD", text.casefold())
    without_marks = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
    return without_marks.replace("đ", "d")


def _content_tokens(text: str) -> set[str]:
    label = normalize_label(text)
    if label is None:
        return set()
    found: set[str] = set()
    for tok in _strip_accents(label).split():
        if not tok or tok in _STOPWORDS or _NUMBER_OR_UNIT.fullmatch(tok):
            continue
        if not any(ch.isalpha() for ch in tok):
            continue
        found.add(tok)
    return found


def _nfc(value: object) -> str:
    return unicodedata.normalize("NFC", str(value)).strip()


@dataclass
class ShopContext:
    """Filter values and product names taken from one shop's canonical DuckDB."""

    payment_methods: list[str]
    channels: list[str]
    segments: list[str]
    categories: list[str]
    product_names: list[str]
    has_order_ts: bool
    data_start: datetime.date | None
    data_end: datetime.date | None

    def product_candidates(self, question: str, k: int = 15) -> list[str]:
        question_tokens = _content_tokens(question)
        if not question_tokens or k <= 0:
            return []
        ranked: list[tuple[int, str, str]] = []
        for name in self.product_names:
            product_tokens = _content_tokens(name)
            if not product_tokens:
                continue
            overlap = question_tokens & product_tokens
            if len(overlap) < 2 and len(overlap) * 2 < len(product_tokens):
                continue
            ranked.append((len(overlap), name.casefold(), name))
        ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
        seen: set[str] = set()
        chosen: list[str] = []
        for _score, _folded, name in ranked:
            if name in seen:
                continue
            seen.add(name)
            chosen.append(name)
            if len(chosen) >= k:
                break
        return chosen


def load_shop_context(con: Any) -> ShopContext:
    return ShopContext(
        payment_methods=_top_values(
            con,
            "SELECT payment_method AS value, COUNT(*) AS n FROM orders "
            "WHERE payment_method IS NOT NULL AND payment_method <> '' "
            "GROUP BY payment_method ORDER BY n DESC, payment_method ASC",
        ),
        channels=_top_values(
            con,
            "SELECT channel AS value, COUNT(*) AS n FROM orders "
            "WHERE channel IS NOT NULL AND channel <> '' "
            "GROUP BY channel ORDER BY n DESC, channel ASC",
        ),
        segments=_top_values(
            con,
            "SELECT c.segment AS value, COUNT(*) AS n FROM orders o "
            "JOIN customers c ON c.customer_key = o.customer_key "
            "WHERE c.segment IS NOT NULL AND c.segment <> '' "
            "GROUP BY c.segment ORDER BY n DESC, c.segment ASC",
        ),
        categories=_top_values(
            con,
            "SELECT p.category AS value, COUNT(*) AS n FROM order_lines l "
            "JOIN products p ON p.product_key = l.product_key "
            "WHERE p.category IS NOT NULL AND p.category <> '' "
            "GROUP BY p.category ORDER BY n DESC, p.category ASC",
        ),
        product_names=_product_names(con),
        has_order_ts=con.execute(
            "SELECT 1 FROM orders WHERE order_ts IS NOT NULL LIMIT 1"
        ).fetchone()
        is not None,
        data_start=_as_date(con.execute("SELECT MIN(order_date) FROM orders").fetchone()),
        data_end=_as_date(con.execute("SELECT MAX(order_date) FROM orders").fetchone()),
    )


def _top_values(con: Any, sql: str) -> list[str]:
    found: list[str] = []
    for row in con.execute(sql).fetchall():
        text = _nfc(row[0])
        if not text or text in found:
            continue
        found.append(text)
        if len(found) >= _VALUE_LIMIT:
            break
    return found


def _product_names(con: Any) -> list[str]:
    rows = con.execute(
        "SELECT product_key, product_name, COUNT(*) AS n FROM order_lines "
        "WHERE product_key IS NOT NULL AND product_name IS NOT NULL AND product_name <> '' "
        "GROUP BY product_key, product_name"
    ).fetchall()
    best: dict[str, tuple[int, str]] = {}
    for key, name, count in rows:
        label = _nfc(name)
        if not label:
            continue
        frequency = int(count)
        current = best.get(str(key))
        if (
            current is None
            or frequency > current[0]
            or (frequency == current[0] and label < current[1])
        ):
            best[str(key)] = (frequency, label)
    ranked = sorted(best.values(), key=lambda item: (-item[0], item[1].casefold(), item[1]))
    names: list[str] = []
    seen: set[str] = set()
    for _count, name in ranked:
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def _as_date(row: tuple[Any, ...] | None) -> datetime.date | None:
    if row is None or row[0] is None:
        return None
    value = row[0]
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    return datetime.date.fromisoformat(str(value)[:10])

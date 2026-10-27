from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import yaml

from datamate.config import PROJECT_ROOT
from datamate.io.questions import Question, load_questions

SPLITS_FILE = PROJECT_ROOT / "evals" / "splits.yaml"
ALL_TRAIN_BUSINESSES = [f"biz_{i}" for i in range(101, 131)]

POC_INCLUDE_TOPICS = frozenset({"revenue", "order_count", "cancel"})
POC_EXCLUDE_TOPICS = frozenset(
    {"debt", "purchase", "inventory", "customer", "product_rank", "cost_profit"}
)


def strip_accents(text: str) -> str:
    nfd = unicodedata.normalize("NFD", text.lower())
    without_combining = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
    return without_combining.replace("đ", "d")


_TOPIC_PATTERNS: dict[str, re.Pattern[str]] = {
    "revenue": re.compile(
        r"\b(?:doanh\s*thu|dthu|dt\b|thu\s*ve|tien\s*hang|tien\s*ban\s*hang|ban\s*(?:duoc|dc)\b[^?]*?\btien\b)",
        re.IGNORECASE,
    ),
    "order_count": re.compile(
        r"\b(?:(?:bao\s*nhieu|bao\s*nhiu|bnhiu|bn|may|so|tong\s*so|tong\s*cong|dem\s*so)\s+don\b|don\s*hang)",
        re.IGNORECASE,
    ),
    "cancel": re.compile(r"\bhuy\b", re.IGNORECASE),
    "change": re.compile(
        r"\b(?:tang|giam|so\s*voi|chenh|hon|kem)\b",
        re.IGNORECASE,
    ),
    "debt": re.compile(r"\b(?:cong\s*no|con\s*no|tien\s*no|khach\s*no|ghi\s*no|tra\s*no|thu\s*no|tuoi\s*no|no\s*them|no\s*cua\s*hang|dang\s*no|du\s*no)\b", re.IGNORECASE),
    "purchase": re.compile(r"\b(?:nhap\s*hang|phieu\s*nhap|gia\s*nhap|tien\s*nhap|nhap\s*vao|nhap\s*ve|lo\s*nhap|nha\s*cung\s*cap|ncc)\b", re.IGNORECASE),
    "inventory": re.compile(r"\b(?:ton\s*kho|hang\s*ton|con\s*ton|het\s*hang|het\s*han|han\s*dung)\b", re.IGNORECASE),
    "customer": re.compile(r"\bkhach\b", re.IGNORECASE),
    "product_rank": re.compile(r"\b(?:ban\s*chay|chay\s*nhat|nhieu\s*nhat|it\s*nhat|cao\s*nhat|thap\s*nhat|top|hot\s*nhat)\b", re.IGNORECASE),
    "cost_profit": re.compile(r"\b(?:gia\s*von|tien\s*von|lai\s*gop|lai\s*rong|loi\s*nhuan|loi\s*nhat|chi\s*phi)\b", re.IGNORECASE),
}


def tag_topics(question_text: str) -> set[str]:
    norm = strip_accents(question_text)
    matched: set[str] = set()
    for topic, pattern in _TOPIC_PATTERNS.items():
        if pattern.search(norm):
            matched.add(topic)
    return matched


def load_splits(splits_path: Path = SPLITS_FILE) -> dict[str, list[str]]:
    raw = yaml.safe_load(splits_path.read_text(encoding="utf-8"))
    poc = list(raw["poc"])
    val = list(raw["val"])
    holdout_family = list(raw["holdout_family"])

    fixed = set(poc) | set(val) | set(holdout_family)
    computed_dev = sorted(set(ALL_TRAIN_BUSINESSES) - fixed)
    if raw.get("dev"):
        declared_dev = list(raw["dev"])
        assert sorted(declared_dev) == computed_dev, (
            f"Declared dev {declared_dev} does not match complement {computed_dev}"
        )
    dev = computed_dev

    groups = {"poc": poc, "val": val, "holdout_family": holdout_family, "dev": dev}
    seen: set[str] = set()
    for name, items in groups.items():
        overlap = seen & set(items)
        assert not overlap, f"Group {name} overlaps with previous groups: {overlap}"
        seen |= set(items)
    assert seen == set(ALL_TRAIN_BUSINESSES), (
        f"Splits do not cover all 30 businesses: missing={set(ALL_TRAIN_BUSINESSES) - seen}, "
        f"extra={seen - set(ALL_TRAIN_BUSINESSES)}"
    )
    return groups


def is_poc_topic_match(topics: set[str]) -> bool:
    return bool(topics & POC_INCLUDE_TOPICS) and not bool(topics & POC_EXCLUDE_TOPICS)


def select_questions(questions: list[Question], subset: str) -> list[Question]:
    if subset == "all":
        return list(questions)
    splits = load_splits()
    if subset not in splits:
        raise ValueError(f"Unknown subset {subset!r}; expected one of { [*splits.keys(), 'all'] }")
    biz_set = set(splits[subset])
    in_biz = [q for q in questions if q.business_id in biz_set]
    if subset == "poc":
        return [q for q in in_biz if is_poc_topic_match(tag_topics(q.question))]
    return in_biz


if __name__ == "__main__":
    all_questions = load_questions("train")
    poc_questions = select_questions(all_questions, "poc")
    print(f"POC subset size: {len(poc_questions)} questions")

from __future__ import annotations

import pytest

from datamate.config import settings
from datamate.io.questions import load_questions
from evals.subsets import (
    ALL_TRAIN_BUSINESSES,
    POC_EXCLUDE_TOPICS,
    POC_INCLUDE_TOPICS,
    load_splits,
    select_questions,
    tag_topics,
)


def test_splits_disjoint_and_cover_all_30() -> None:
    splits = load_splits()
    assert set(splits.keys()) == {"poc", "val", "holdout_family", "dev"}

    assert splits["poc"] == ["biz_101", "biz_105", "biz_106", "biz_111", "biz_123"]
    assert splits["val"] == ["biz_104", "biz_110", "biz_118", "biz_120", "biz_124", "biz_127"]
    assert splits["holdout_family"] == ["biz_107", "biz_113", "biz_126"]
    assert len(splits["dev"]) == 16

    all_seen: list[str] = []
    for group in splits.values():
        all_seen.extend(group)
    assert len(all_seen) == 30
    assert set(all_seen) == set(ALL_TRAIN_BUSINESSES)


def test_tag_topics_basic() -> None:
    assert "revenue" in tag_topics("Tổng doanh thu tháng này là bao nhiêu?")
    assert "order_count" in tag_topics("check gium hnay cua hang co bao nhiu don")
    assert "cancel" in tag_topics("tỉ lệ huỷ đơn tháng 6/2025 là nhiêu %")
    assert "debt" in tag_topics("Có bao nhiêu khách còn nợ cửa hàng?")
    assert "purchase" in tag_topics("tháng này tổng giá trị các phiếu nhập hàng là bao nhiu")


def test_poc_subset_non_empty() -> None:
    if not (settings.data_root / "train").exists():
        pytest.skip(f"Train data folder absent at {settings.data_root / 'train'}")

    questions = load_questions("train")
    poc_questions = select_questions(questions, "poc")
    assert len(poc_questions) > 0
    assert 180 <= len(poc_questions) <= 300

    poc_biz = {"biz_101", "biz_105", "biz_106", "biz_111", "biz_123"}
    for q in poc_questions:
        assert q.business_id in poc_biz
        topics = tag_topics(q.question)
        assert topics & POC_INCLUDE_TOPICS
        assert not (topics & POC_EXCLUDE_TOPICS)

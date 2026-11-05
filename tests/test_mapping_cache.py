"""Stored mappings come back unchanged and are keyed by files, model and code."""

from __future__ import annotations

from pathlib import Path

import pytest

from datamate.config import settings
from datamate.mapping import Mapping, mapping_fingerprint
from datamate.mapping.cache import load_mapping, mapping_cache_path, store_mapping
from tests.test_mapping_model import _minimal_mapping_dict


class _Client:
    model = "qwen3-14b"


@pytest.fixture
def cache_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "mapping_cache_enabled", True)
    monkeypatch.setattr(settings, "cache_dir", tmp_path)


@pytest.mark.usefixtures("cache_on")
def test_round_trip_keeps_the_fingerprint() -> None:
    mapping = Mapping.model_validate(_minimal_mapping_dict())
    path = mapping_cache_path("train", "biz_x", "raw-key", _Client())
    assert path is not None
    assert load_mapping(path) is None
    store_mapping(path, mapping)
    again = load_mapping(path)
    assert again is not None
    assert mapping_fingerprint(again) == mapping_fingerprint(mapping)


@pytest.mark.usefixtures("cache_on")
def test_unreadable_entry_is_ignored() -> None:
    path = mapping_cache_path("train", "biz_x", "raw-key", _Client())
    assert path is not None
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    assert load_mapping(path) is None


@pytest.mark.usefixtures("cache_on")
def test_key_changes_with_files_and_model() -> None:
    base = mapping_cache_path("train", "biz_x", "raw-key", _Client())
    other_files = mapping_cache_path("train", "biz_x", "raw-key-2", _Client())

    class _Other:
        model = "qwen3-4b"

    other_model = mapping_cache_path("train", "biz_x", "raw-key", _Other())
    assert len({base, other_files, other_model}) == 3


@pytest.mark.usefixtures("cache_on")
def test_client_without_a_model_name_is_never_cached() -> None:
    assert mapping_cache_path("train", "biz_x", "raw-key", object()) is None


def test_switched_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "mapping_cache_enabled", False)
    assert mapping_cache_path("train", "biz_x", "raw-key", _Client()) is None

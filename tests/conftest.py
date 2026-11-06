from __future__ import annotations

import pytest

from datamate.config import settings


@pytest.fixture(autouse=True)
def _no_extra_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    # Scripted fake clients answer only the calls a test is about; the opinion and
    # rulebook-term checks have their own tests that turn them back on.
    monkeypatch.setattr(settings, "subjective_check", False)
    monkeypatch.setattr(settings, "defined_term_check", False)
    # Fake clients must never write mappings into the shared cache.
    monkeypatch.setattr(settings, "mapping_cache_enabled", False)
    monkeypatch.setattr(settings, "event_path", False)
    monkeypatch.setattr(settings, "unit_terms", False)
    monkeypatch.setattr(settings, "attribute_path", False)
    monkeypatch.setattr(settings, "missing_group_check", False)
    monkeypatch.setattr(settings, "missing_attribute_check", False)
    monkeypatch.setattr(settings, "plain_retry", False)

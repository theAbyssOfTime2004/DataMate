from __future__ import annotations

import pytest

from datamate.config import settings


@pytest.fixture(autouse=True)
def _no_subjective_check(monkeypatch: pytest.MonkeyPatch) -> None:
    # Scripted fake clients answer only the calls a test is about; the opinion check has
    # its own tests that turn it back on.
    monkeypatch.setattr(settings, "subjective_check", False)

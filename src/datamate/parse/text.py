from __future__ import annotations

import unicodedata


def normalize_label(raw: str | None) -> str | None:
    if raw is None or not isinstance(raw, str):
        return None
    s = unicodedata.normalize("NFC", raw).replace("\xa0", " ").strip()
    if not s:
        return None
    return " ".join(s.casefold().split())

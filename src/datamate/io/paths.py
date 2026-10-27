from __future__ import annotations

from pathlib import Path

from datamate.config import settings


def business_dir(split: str, business_id: str) -> Path:
    path = settings.split_dir(split) / "data" / business_id
    if not path.exists():
        raise FileNotFoundError(
            f"Business directory for {business_id!r} in split {split!r} does not exist at {path}"
        )
    return path

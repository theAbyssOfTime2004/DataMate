"""Disk cache of inferred mappings.

Inferring a mapping runs the shop's full ETL twice to validate it, which takes minutes per
shop even when every LLM answer comes from the LLM cache. The result depends only on the
shop's files, the model and this code, so it is stored under a key built from those.
"""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path
from typing import Any

from datamate.config import settings
from datamate.mapping.model import Mapping

# Bump to drop every stored mapping by hand.
MAPPING_CACHE_VERSION = "1"
# Packages whose code decides the mapping: inference, validation (runs the ETL), the raw
# readers and value parsers, and the LLM client.
_CODE_PACKAGES = ("mapping", "etl", "io", "parse", "llm")


@functools.cache
def _code_digest() -> str:
    root = Path(__file__).resolve().parent.parent
    hasher = hashlib.sha256()
    for package in _CODE_PACKAGES:
        for path in sorted((root / package).rglob("*.py")):
            hasher.update(path.relative_to(root).as_posix().encode())
            hasher.update(path.read_bytes())
    return hasher.hexdigest()


def mapping_cache_path(split: str, business_id: str, raw_key: str, client: Any) -> Path | None:
    """Where this shop's mapping is stored, or None when it must not be cached."""
    model = getattr(client, "model", None)
    if not settings.mapping_cache_enabled or not isinstance(model, str):
        return None
    payload = json.dumps(
        {
            "version": MAPPING_CACHE_VERSION,
            "raw": raw_key,
            "model": model,
            "prompt_budget": settings.llm_prompt_budget_tokens,
            "code": _code_digest(),
        },
        sort_keys=True,
    )
    key = hashlib.sha256(payload.encode()).hexdigest()
    return settings.cache_dir / "mapping" / split / business_id / f"{key}.json"


def load_mapping(path: Path) -> Mapping | None:
    """The stored mapping, or None when there is none (or it no longer loads)."""
    if not path.is_file():
        return None
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
        return Mapping.model_validate(stored["mapping"])
    except (OSError, ValueError, KeyError):
        return None


def store_mapping(path: Path, mapping: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"mapping": mapping.model_dump(mode="json")}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)

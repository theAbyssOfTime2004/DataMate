import re
from pathlib import Path

import pytest

from datamate.config import PROJECT_ROOT, settings

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
BIZ_ID_PATTERN = re.compile(r"(?<![A-Za-z0-9])(?:biz_1\d\d|dev_0\d\d)", re.IGNORECASE)


def _iter_src_files() -> list[Path]:
    return [
        p
        for p in SRC_DIR.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    ]


def test_no_gold_mappings_in_src() -> None:
    files = _iter_src_files()
    assert files, "Expected at least one source file under src/"
    offenders: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        if "gold_mappings" in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, f"Forbidden reference to 'gold_mappings' in src/: {offenders}"


def test_no_hardcoded_business_id_in_src() -> None:
    files = _iter_src_files()
    assert files, "Expected at least one source file under src/"
    offenders: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        match = BIZ_ID_PATTERN.search(text)
        if match:
            offenders.append(f"{path.relative_to(REPO_ROOT)} ({match.group(0)})")
    assert not offenders, f"Hard-coded business id found in src/: {offenders}"


def test_split_dir_train() -> None:
    if not (settings.data_root / "train").exists():
        pytest.skip(f"Train data folder absent at {settings.data_root / 'train'}")
    train_dir = settings.split_dir("train")
    assert train_dir.is_dir()


def test_split_dir_missing_raises() -> None:
    with pytest.raises(FileNotFoundError, match="__nonexistent_split__"):
        settings.split_dir("__nonexistent_split__")


def test_default_dirs_under_project_root() -> None:
    assert settings.cache_dir.is_absolute()
    assert settings.runs_dir.is_absolute()
    assert settings.cache_dir.is_relative_to(PROJECT_ROOT)
    assert settings.runs_dir.is_relative_to(PROJECT_ROOT)


def test_biz_id_pattern() -> None:
    assert BIZ_ID_PATTERN.search("path_biz_101") is not None
    assert BIZ_ID_PATTERN.search("biz_10") is None


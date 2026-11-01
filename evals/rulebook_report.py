"""Report RuleConfig derived from a split's rulebook.md vs train defaults.

Usage:
    .venv\\Scripts\\python evals/rulebook_report.py --split train
    .venv\\Scripts\\python evals/rulebook_report.py --split public_test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datamate.config import settings
from datamate.llm import get_client
from datamate.rules.from_rulebook import build_rulebook_config
from datamate.rules.knobs import KNOBS
from datamate.rules.model import RuleConfig
from datamate.rules.parse_rulebook import RuleRow, parse_rulebook


def _row_key(row: RuleRow) -> tuple[str, str]:
    # Compare by section-relative name; section titles include the book title so
    # strip the first segment when present.
    parts = [p.strip() for p in row.section_path.split(">")]
    # Drop the H1 book title if it looks like one.
    if parts and parts[0].lower().startswith("datamate"):
        parts = parts[1:]
    return (" > ".join(parts), row.name)


def _rows_by_key(path: Path) -> dict[tuple[str, str], RuleRow]:
    if not path.is_file():
        return {}
    return {_row_key(row): row for row in parse_rulebook(path)}


def _knob_value(rules: RuleConfig, field_path: str) -> Any:
    if field_path == "tet":
        return rules.tet.model_dump(mode="json")
    if field_path == "shifts":
        out: dict[str, list[str]] = {}
        for name, (start, end) in rules.shifts.items():
            out[name] = [start.strftime("%H:%M:%S"), end.strftime("%H:%M:%S")]
        return out
    return getattr(rules, field_path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Rulebook → RuleConfig report")
    parser.add_argument("--split", required=True)
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM extract; only print deterministic row diff vs train",
    )
    args = parser.parse_args(argv)

    split_dir = settings.split_dir(args.split)
    rulebook_path = split_dir / "rulebook.md"
    train_rulebook = settings.data_root / "train" / "rulebook.md"

    print(f"## Rulebook report — `{args.split}`")
    print()

    # Deterministic row diff vs train.
    train_rows = _rows_by_key(train_rulebook)
    split_rows = _rows_by_key(rulebook_path)
    train_keys = set(train_rows)
    split_keys = set(split_rows)
    new_keys = sorted(split_keys - train_keys)
    removed_keys = sorted(train_keys - split_keys)
    changed_keys = sorted(
        k
        for k in (train_keys & split_keys)
        if train_rows[k].text != split_rows[k].text
    )

    print("### Row diff vs train/rulebook.md")
    print(f"- new: {len(new_keys)} | removed: {len(removed_keys)} | changed: {len(changed_keys)}")
    for label, keys in (
        ("NEW", new_keys),
        ("REMOVED", removed_keys),
        ("CHANGED", changed_keys),
    ):
        for key in keys[:50]:
            print(f"  {label}: {key[0]} | {key[1]}")
        if len(keys) > 50:
            print(f"  ... {len(keys) - 50} more")
    print()

    if args.no_llm or not rulebook_path.is_file():
        if not rulebook_path.is_file():
            print(f"(no rulebook.md at {rulebook_path})")
        return

    client = get_client()
    try:
        cfg = build_rulebook_config(rulebook_path, client)
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()

    print("### Knob values vs train defaults")
    diffs = 0
    for knob in KNOBS:
        current = _knob_value(cfg.rules, knob.field_path)
        default = knob.train_default
        # Normalise for comparison.
        cur_j = json.dumps(current, ensure_ascii=False, sort_keys=True, default=str)
        def_j = json.dumps(default, ensure_ascii=False, sort_keys=True, default=str)
        mark = ""
        if cur_j != def_j:
            diffs += 1
            mark = " **DIFF**"
        unsup = " unsupported" if knob.id in cfg.unsupported else ""
        print(f"- `{knob.id}` = {cur_j}{mark}{unsup}")
    print(f"\n{diffs} knob(s) differ from train defaults.")
    print(f"unsupported: {sorted(cfg.unsupported) or '—'}")
    print(f"warnings: {len(cfg.warnings)}")
    for warning in cfg.warnings[:30]:
        print(f"  - {warning}")
    print()

    print("### Notes by section")
    by_section: dict[str, list[RuleRow]] = {}
    for note in cfg.notes:
        by_section.setdefault(note.section_path, []).append(note)
    if not by_section:
        print("(none)")
    for section in sorted(by_section):
        print(f"- {section}")
        for note in by_section[section]:
            print(f"  - {note.name}")
    print()

    print("### New/changed rows → knob or note")
    interesting = [*new_keys, *changed_keys]
    for key in interesting:
        row = split_rows[key]
        mapped = cfg.classifications.get(row.id, "?")
        kind = "note" if mapped == "other" else f"knob:{mapped}"
        print(f"- [{kind}] {key[0]} | {key[1]}")


if __name__ == "__main__":
    main()

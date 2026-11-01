"""Build RuleConfig from a split's rulebook.md via classify + extract LLM calls."""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datamate.config import settings
from datamate.rules.knobs import KNOB_IDS, KNOBS, KNOBS_BY_ID
from datamate.rules.model import RuleConfig, load_rules
from datamate.rules.parse_rulebook import RuleRow, parse_rulebook, rulebook_sha256

# Bump when classify/extract prompts or few-shots change (invalidates disk cache).
PROMPT_VERSION = "e2-v4"

# Label lists are synonym sets matched against shop values: the rulebook names the concept,
# the defaults carry spelling variants, so extracted labels are added, never replace them.
_UNION_LIST_KNOBS = frozenset(
    {"debt_exclude_payment_labels", "debt_exclude_channel_labels", "non_product_line_labels"}
)

_CLASSIFY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["knob"],
    "properties": {
        "knob": {"type": "string", "enum": [*KNOB_IDS, "other"]},
    },
}


@dataclass
class RulebookConfig:
    """Rules derived from one rulebook.md, plus notes the config cannot express."""

    rules: RuleConfig
    unsupported: set[str] = field(default_factory=set)
    notes: list[RuleRow] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rulebook_sha256: str = ""
    # row id -> knob id or "other"
    classifications: dict[str, str] = field(default_factory=dict)

    def to_cache_dict(self) -> dict[str, Any]:
        return {
            "rules": self.rules.model_dump(mode="json"),
            "unsupported": sorted(self.unsupported),
            "notes": [
                {
                    "id": row.id,
                    "section_path": row.section_path,
                    "name": row.name,
                    "text": row.text,
                }
                for row in self.notes
            ],
            "warnings": list(self.warnings),
            "rulebook_sha256": self.rulebook_sha256,
            "classifications": dict(self.classifications),
            "prompt_version": PROMPT_VERSION,
        }

    @classmethod
    def from_cache_dict(cls, payload: dict[str, Any]) -> RulebookConfig:
        notes = [
            RuleRow(
                id=str(item["id"]),
                section_path=str(item["section_path"]),
                name=str(item["name"]),
                text=str(item["text"]),
            )
            for item in payload.get("notes", [])
        ]
        return cls(
            rules=RuleConfig.model_validate(payload["rules"]),
            unsupported=set(payload.get("unsupported", [])),
            notes=notes,
            warnings=list(payload.get("warnings", [])),
            rulebook_sha256=str(payload.get("rulebook_sha256", "")),
            classifications={
                str(k): str(v) for k, v in payload.get("classifications", {}).items()
            },
        )


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _knob_catalogue_block() -> str:
    lines = [
        "Knob catalogue (pick exactly one id, or other if none fit):",
    ]
    for knob in KNOBS:
        lines.append(f"- {knob.id}: {knob.meaning}")
    lines.append("- other: the row is not a RuleConfig parameter (definitions, scoring, industry quirks).")
    return "\n".join(lines)


# Invented few-shots only (never real train/public wording). Leakage test covers them.
_CLASSIFY_FEW_SHOTS: tuple[tuple[str, str, str, str], ...] = (
    (
        "1. Sales basics",
        "Which order statuses count as sold",
        "Completed and out-for-delivery orders count; cancelled ones do not.",
        "revenue_statuses",
    ),
    (
        "2. Calendar",
        "Week boundary",
        "A week starts on Monday and ends on Sunday.",
        "week_start",
    ),
    (
        "3. Rankings",
        "Best sellers",
        "Rank products by base-unit quantity sold, then by product code.",
        "ranking_measure",
    ),
    (
        "4. Industry — bakery",
        "Loyalty stamp card",
        "Ten stamps earn one free pastry; stamps reset each calendar year.",
        "other",
    ),
)


def _classify_messages(row: RuleRow) -> list[dict[str, str]]:
    examples = []
    for section, name, text, knob in _CLASSIFY_FEW_SHOTS:
        examples.append(
            f"Example\nsection: {section}\nname: {name}\ntext: {text}\n"
            f'{{"knob": "{knob}"}}'
        )
    user = (
        f"{_knob_catalogue_block()}\n\n"
        + "\n\n".join(examples)
        + "\n\nClassify this rule row. Reply with JSON only.\n"
        f"section: {row.section_path}\nname: {row.name}\ntext: {row.text}"
    )
    return [
        {
            "role": "system",
            "content": (
                "You map one shop-rulebook table row to a RuleConfig knob id, or other. "
                "Use only the catalogue ids. When unsure, choose other."
            ),
        },
        {"role": "user", "content": _nfc(user)},
    ]


_EXTRACT_FEW_SHOTS: dict[str, tuple[str, Any]] = {
    # Invented rows + expected JSON value (not from competition rulebooks).
    "week_start": (
        "name: Week start\ntext: Weeks run Sunday through Saturday.\n",
        "sunday",
    ),
    "ranking_measure": (
        "name: Popular items\ntext: Rank by revenue (money), not by quantity.\n",
        "revenue",
    ),
    "exclude_orders_with_return": (
        "name: Returns\ntext: Keep the original sale in revenue even if part of it was returned.\n",
        False,
    ),
    "percent_decimals": (
        "name: Percent display\ntext: Round percentages to 1 digit after the decimal point.\n",
        1,
    ),
}


def _extract_schema(knob_id: str) -> dict[str, Any]:
    knob = KNOBS_BY_ID[knob_id]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["value"],
        "properties": {"value": knob.value_schema},
    }


def _extract_messages(knob_id: str, rows: list[RuleRow]) -> list[dict[str, str]]:
    knob = KNOBS_BY_ID[knob_id]
    blocks = [f"name: {row.name}\ntext: {row.text}" for row in rows]
    few = ""
    shot = _EXTRACT_FEW_SHOTS.get(knob_id)
    if shot is not None:
        example_rows, example_value = shot
        few = (
            f"Example rows:\n{example_rows}"
            f'Example JSON: {{"value": {json.dumps(example_value, ensure_ascii=False)}}}\n\n'
        )
    user = (
        f"Knob: {knob.id}\nMeaning: {knob.meaning}\n"
        f"Value schema: {json.dumps(knob.value_schema, ensure_ascii=False)}\n"
        f"Train default (fallback): {json.dumps(knob.train_default, ensure_ascii=False)}\n\n"
        f"{few}"
        "Read the rule rows and fill `value` for this knob. JSON only.\n\n"
        "Rows:\n" + "\n---\n".join(blocks)
    )
    return [
        {
            "role": "system",
            "content": (
                "Extract one RuleConfig field value from the given rulebook rows. "
                "Follow the JSON schema. If the rows are ambiguous, prefer the train default."
            ),
        },
        {"role": "user", "content": _nfc(user)},
    ]


def _two_vote_json(
    client: Any,
    messages: list[dict[str, str]],
    schema: dict[str, Any],
    schema_name: str,
) -> tuple[dict[str, Any] | None, bool]:
    """Return (payload, agreed). disagreed or error -> (None, False)."""
    votes: list[dict[str, Any]] = []
    for _ in range(2):
        try:
            votes.append(
                client.chat_json(
                    messages,
                    schema=schema,
                    schema_name=schema_name,
                )
            )
        except Exception:  # noqa: BLE001 — treat as disagreement / keep default
            return None, False
    if len(votes) < 2:
        return None, False
    if votes[0] == votes[1]:
        return votes[0], True
    return None, False


def _value_supported(knob_id: str, value: Any) -> bool:
    knob = KNOBS_BY_ID[knob_id]
    if knob.supported is None:
        return True
    if knob.kind == "string_list":
        if not isinstance(value, list):
            return False
        return all(item in knob.supported for item in value)
    return value in knob.supported


def _cache_path(sha: str) -> Path:
    return settings.cache_dir / "rulebook" / PROMPT_VERSION / f"{sha}.json"


def _read_cache(sha: str) -> RulebookConfig | None:
    path = _cache_path(sha)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("prompt_version") != PROMPT_VERSION:
        return None
    return RulebookConfig.from_cache_dict(payload)


def _write_cache(sha: str, result: RulebookConfig) -> None:
    path = _cache_path(sha)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.to_cache_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_rulebook_config(
    rulebook_path: Path | str,
    client: Any,
    *,
    use_cache: bool = True,
) -> RulebookConfig:
    """Classify + extract RuleConfig from rulebook.md."""
    path = Path(rulebook_path)
    sha = rulebook_sha256(path)
    if use_cache:
        cached = _read_cache(sha)
        if cached is not None:
            return cached

    rows = parse_rulebook(path)
    warnings: list[str] = []
    classifications: dict[str, str] = {}
    by_knob: dict[str, list[RuleRow]] = {kid: [] for kid in KNOB_IDS}
    notes: list[RuleRow] = []

    for row in rows:
        payload, agreed = _two_vote_json(
            client, _classify_messages(row), _CLASSIFY_SCHEMA, "rulebook_classify"
        )
        if not agreed or payload is None:
            knob_id = "other"
            warnings.append(f"classify_disagree:{row.id}:{row.name}")
        else:
            raw = payload.get("knob")
            knob_id = raw if isinstance(raw, str) and raw in (*KNOB_IDS, "other") else "other"
            if knob_id != raw:
                warnings.append(f"classify_invalid:{row.id}:{raw!r}")
        classifications[row.id] = knob_id
        # Every row stays a note: a row classified to a knob may still define a concept
        # the knob cannot express (the classifier only picks the closest knob).
        notes.append(row)
        if knob_id != "other":
            by_knob[knob_id].append(row)

    # JSON-shaped defaults so RuleConfig.model_validate can re-read after extracts.
    values: dict[str, Any] = RuleConfig().model_dump(mode="json")
    unsupported: set[str] = set()

    for knob in KNOBS:
        matched = by_knob[knob.id]
        if not matched:
            continue
        payload, agreed = _two_vote_json(
            client,
            _extract_messages(knob.id, matched),
            _extract_schema(knob.id),
            f"rulebook_extract_{knob.id}",
        )
        if not agreed or payload is None:
            warnings.append(f"extract_disagree:{knob.id}")
            continue
        value = payload.get("value")
        try:
            if not _value_supported(knob.id, value):
                unsupported.add(knob.id)
                warnings.append(f"extract_unsupported:{knob.id}:{value!r}")
                continue
            trial = json.loads(json.dumps(values, ensure_ascii=False))
            if knob.id == "shifts":
                if not isinstance(value, dict):
                    raise TypeError("shifts must be an object")
                # Keep JSON form [HH:MM:SS, HH:MM:SS] for model_validate.
                trial["shifts"] = value
            elif knob.id in _UNION_LIST_KNOBS and isinstance(value, list):
                merged = list(trial[knob.id])
                seen = {_nfc(str(v)).casefold() for v in merged}
                for label in value:
                    key = _nfc(str(label)).casefold()
                    if key and key not in seen:
                        merged.append(label)
                        seen.add(key)
                trial[knob.id] = merged
            else:
                trial[knob.id] = value
            RuleConfig.model_validate(trial)
            values = trial
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"extract_invalid:{knob.id}:{exc}")
            continue

    # Re-validate full config; on failure fall back entirely to train defaults.
    try:
        rules = RuleConfig.model_validate(values)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"rules_validate_failed:{exc}")
        rules = RuleConfig()

    # Mark knobs whose extracted value is outside supported (already), and
    # single-valued knobs where extract wanted something else — handled above.
    # Also: if supported is a singleton and extract kept default, fine.
    for knob in KNOBS:
        if knob.id in unsupported:
            continue
        current = getattr(rules, knob.field_path.split(".")[0])
        if knob.field_path == "tet":
            current = rules.tet.model_dump(mode="json")
        if knob.supported is not None and knob.kind == "enum" and current not in knob.supported:
            unsupported.add(knob.id)

    result = RulebookConfig(
        rules=rules,
        unsupported=unsupported,
        notes=notes,
        warnings=warnings,
        rulebook_sha256=sha,
        classifications=classifications,
    )
    if use_cache:
        _write_cache(sha, result)
    return result


def load_split_rulebook(
    split: str,
    client: Any | None,
    *,
    warnings_out: list[str] | None = None,
) -> RulebookConfig:
    """Load RulebookConfig for a split: override JSON, else rulebook.md, else train defaults."""
    warn = warnings_out if warnings_out is not None else []

    override = getattr(settings, "rules_override_path", None)
    if override is not None:
        path = Path(override)
        rules = load_rules(path)
        cfg = RulebookConfig(
            rules=rules,
            unsupported=set(),
            notes=[],
            warnings=[f"rules_override:{path}"],
            rulebook_sha256=f"override:{path}",
        )
        warn.extend(cfg.warnings)
        return cfg

    split_dir = settings.split_dir(split)
    rulebook_path = split_dir / "rulebook.md"
    if not rulebook_path.is_file():
        warn.append(f"rulebook.md missing for {split!r}; using RuleConfig defaults")
        return RulebookConfig(
            rules=RuleConfig(),
            unsupported=set(),
            notes=[],
            warnings=list(warn[-1:]),
            rulebook_sha256="",
        )

    if client is None:
        warn.append("no LLM client for rulebook; using RuleConfig defaults")
        return RulebookConfig(
            rules=RuleConfig(),
            unsupported=set(),
            notes=[],
            warnings=list(warn[-1:]),
            rulebook_sha256=rulebook_sha256(rulebook_path),
        )

    cfg = build_rulebook_config(rulebook_path, client)
    warn.extend(cfg.warnings)
    return cfg

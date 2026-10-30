"""Prompts for the direct pandas baseline (system C)."""

from __future__ import annotations

import datetime
import json
import re
import unicodedata
from collections.abc import Callable
from typing import Any

from datamate.config import settings

_WEEKDAYS = ("Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật")

_RULE_KEYWORDS = (
    "don hang",
    "doanh thu",
    "thoi gian",
    "khong tra loi",
    "cach ghi",
    "dap an",
    "tu ngu",
)

# Core sections kept as long as possible when shrinking the rulebook for the prompt budget.
_CORE_RULE_KEYWORDS = (
    "don hang",
    "doanh thu",
    "dap an",
    "cach ghi",
)

_SALES_HINTS = (
    "ban",
    "don",
    "order",
    "hoa",
    "hang",
    "khach",
    "mon",
    "san pham",
    "product",
    "thanh",
    "tien",
    "doanh",
    "invoice",
    "payment",
    "ship",
    "line",
)
_NON_SALES_HINTS = (
    "kho",
    "ton",
    "inventory",
    "stock",
    "ncc",
    "nhap",
    "purchase",
    "supplier",
    "cong no",
    "debt",
    "luong",
    "payroll",
    "nhan vien",
    "employee",
)

DIRECT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "in_scope": {"type": "boolean"},
        "code": {"type": "string"},
    },
    "required": ["in_scope", "code"],
    "additionalProperties": False,
}

DIRECT_REQUIRED_KEYS = ["in_scope", "code"]
DIRECT_MAX_TOKENS = 1400

_SYSTEM_LINES = [
    "You answer ONE question about ONE shop by writing Python/pandas code.",
    "The shop data is already loaded as `tables: dict[str, pandas.DataFrame]`.",
    "Every cell is a raw string or None, exactly as read from the shop files.",
    "Use the exact key as listed under \"Table keys in `tables`\" (aliases may also work).",
    "Keep the code short (under 40 lines), no comments, no function definitions unless needed.",
    "",
    "Helpers available without import:",
    "- parse_money(str) -> int | None  (VND integer)",
    "- parse_date(str, default_year=None) -> datetime.date | None",
    "- parse_decimal(str) -> Decimal | None",
    "- normalize_label(str) -> str | None",
    "- AS_OF: datetime.date  (\"hôm nay\" / the question's today)",
    "- date, timedelta (and the datetime module); pd (pandas), np (numpy)",
    (
        "parse_date returns datetime.date objects (or None): filter with comparisons against "
        "datetime.date values, or convert with pd.to_datetime(series) before using .dt."
    ),
    (
        "Use only the table keys and column names exactly as listed; when a KeyError lists the "
        "available keys, pick one of them."
    ),
    "",
    "Your code MUST assign `result = {\"type\": ..., \"answer\": ...}` (and optional \"reason\").",
    "Allowed types: number, percent, date, string, list (see the rulebook excerpt,",
    "especially answer format / §9). Percents are the percent itself (12.4 not 0.124).",
    "Money is integer VND. Prefer parse_money / parse_date over hand-rolled parsing.",
    "Do not redefine parse_money/parse_date/parse_decimal/normalize_label/tables/AS_OF.",
    "",
    "Set in_scope=false when the shop data cannot answer the question (debt, inventory,",
    "purchases, profit, best sellers, customer lists, unit counts sold, prices, etc.).",
    "When in_scope is false, code may be an empty string.",
    "",
    "Do not import os/sys/subprocess. Do not open files. Only use `tables` and the helpers.",
    "",
    "Worked example (invented shop, not from any contest split):",
    "Table keys in tables: ban_hang — use the exact key as listed.",
    "tables['ban_hang'] has columns Ngay, Mat hang, Thanh tien, Trang thai with rows like",
    "  01/08/2026 | Ca phe | 55.000đ | Hoàn thành",
    "  02/08/2026 | Banh   | 30.000đ | Đã hủy",
    "Question: doanh thu thang nay (AS_OF=2026-08-15). Example code:",
    "df = tables['ban_hang'].copy()",
    "df['tien'] = df['Thanh tien'].map(parse_money)",
    "df['ngay'] = df['Ngay'].map(parse_date)",
    "df['st'] = df['Trang thai'].map(normalize_label)",
    "done = df[(df['st'] == 'hoàn thành') & df['ngay'].notna()]",
    "done = done[(done['ngay'] >= date(2026, 8, 1)) & (done['ngay'] <= AS_OF)]",
    "result = {'type': 'number', 'answer': int(done['tien'].dropna().sum())}",
]

_TABLE_SPLIT = re.compile(r"(?=^### Table `)", re.MULTILINE)
_TABLE_HEADER = re.compile(r"^### Table `([^`]+)`(?: \(([^)]*)\))?")
_COL_LINE = re.compile(r"^  - (.+)$")
_KINDS_RE = re.compile(r"\|\s*kinds\s+([^|]+)")
_EG_RE = re.compile(r"\|\s*e\.g\.\s+(.+)$")
_QUOTED = re.compile(r'"([^"]*)"')


def _strip_accents(text: str) -> str:
    nfd = unicodedata.normalize("NFD", text.casefold())
    without = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
    return without.replace("đ", "d").replace("Đ", "d")


def rulebook_excerpt(rulebook_text: str, budget_chars: int = 7000) -> str:
    """Keep matching ## sections in file order; drop from the end until under budget."""
    sections = _split_sections(rulebook_text)
    kept = [
        body
        for heading, body in sections
        if any(key in _strip_accents(heading) for key in _RULE_KEYWORDS)
    ]
    if not kept:
        text = rulebook_text
        if len(text) <= budget_chars:
            return text
        return text[:budget_chars].rstrip() + "\n"
    while kept and sum(len(part) for part in kept) > budget_chars:
        kept.pop()
    if not kept:
        first = next(
            body
            for heading, body in sections
            if any(key in _strip_accents(heading) for key in _RULE_KEYWORDS)
        )
        return first[:budget_chars].rstrip() + "\n"
    return "".join(kept).rstrip() + "\n"


def _split_sections(text: str) -> list[tuple[str, str]]:
    lines = text.splitlines(keepends=True)
    sections: list[tuple[str, str]] = []
    heading = ""
    buf: list[str] = []
    for line in lines:
        if line.startswith("## "):
            if buf or heading:
                sections.append((heading, "".join(buf)))
            heading = line[3:].strip()
            buf = [line]
        else:
            buf.append(line)
    if buf or heading:
        sections.append((heading, "".join(buf)))
    return sections


def build_messages(
    question: str,
    as_of: datetime.date,
    business_text: str,
    rules_text: str,
    attempt_errors: list[tuple[str, str]] | None = None,
    table_keys: list[str] | None = None,
) -> list[dict[str, str]]:
    """Build chat messages. attempt_errors is a list of (code, error) for repair turns."""
    weekday = _WEEKDAYS[as_of.weekday()]
    keys = table_keys or []
    keys_line = f"Table keys in `tables`: {', '.join(keys)}" if keys else "Table keys in `tables`: (none)"
    user = "\n".join(
        [
            f"Hôm nay (as_of): {as_of.isoformat()} ({weekday})",
            keys_line,
            "",
            "## Shop data cards",
            business_text.rstrip(),
            "",
            "## Rulebook excerpt",
            rules_text.rstrip(),
            "",
            "## Question",
            question,
            "",
            "Return JSON with in_scope and code.",
        ]
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": "\n".join(_SYSTEM_LINES)},
        {"role": "user", "content": user},
    ]
    for code, error in attempt_errors or []:
        messages.append({"role": "assistant", "content": _assistant_payload(True, code)})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"That code failed: {_last_lines(error, 20)}. Return corrected code."
                ),
            }
        )
    return messages


def _assistant_payload(in_scope: bool, code: str) -> str:
    return json.dumps({"in_scope": in_scope, "code": code}, ensure_ascii=False)


def _last_lines(text: str, n: int) -> str:
    lines = text.strip().splitlines()
    return "\n".join(lines[-n:]) if lines else text


def _prompt_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(f"{message['role']}\n{message['content']}" for message in messages)


def fit_messages(
    question: str,
    as_of: datetime.date,
    business_text: str,
    rules_text: str,
    attempt_errors: list[tuple[str, str]] | None,
    count_tokens: Callable[[str], int],
    budget_tokens: int | None = None,
    table_keys: list[str] | None = None,
) -> list[dict[str, str]]:
    """Shrink cards → core rulebook → last-resort drops until the prompt fits."""
    budget = (
        settings.direct_prompt_budget_tokens if budget_tokens is None else budget_tokens
    )
    rules = rules_text
    business = business_text
    stage = 0  # 0 compact cards, 1 core rules, 2 last resort, 3 done

    while True:
        messages = build_messages(
            question, as_of, business, rules, attempt_errors, table_keys=table_keys
        )
        if count_tokens(_prompt_text(messages)) <= budget:
            return messages

        if stage == 0:
            compacted = _compact_table_cards(business)
            stage = 1
            if compacted != business:
                business = compacted
                continue
            continue

        if stage == 1:
            core = _core_rulebook(rules)
            stage = 2
            if core != rules:
                rules = core
                continue
            continue

        if stage == 2:
            trimmed = _last_resort_trim(business)
            if trimmed != business:
                business = trimmed
                continue
            # Still over: truncate README tail only (tables already compacted).
            with_readme = _truncate_readme_tail(business)
            if with_readme != business:
                business = with_readme
                continue
            stage = 3
            continue

        return messages


def _compact_table_cards(business_text: str) -> str:
    """Keep README + every table header; columns become name | kinds | e.g. samples."""
    parts = _TABLE_SPLIT.split(business_text)
    if len(parts) <= 1:
        return business_text
    out: list[str] = []
    changed = False
    for part in parts:
        if not part.startswith("### Table `"):
            out.append(part)
            continue
        lines = part.splitlines()
        header = lines[0] if lines else part
        compact_cols: list[str] = []
        for line in lines[1:]:
            if not line.startswith("  - "):
                continue
            compact_cols.append(_compact_column_line(line))
        block = header + "\n" + "\n".join(compact_cols)
        if not block.endswith("\n"):
            block += "\n"
        # Preserve a blank line between cards when the original had one.
        if part.endswith("\n\n"):
            block += "\n"
        elif part.endswith("\n"):
            pass
        out.append(block)
        if block != part:
            changed = True
    return "".join(out) if changed else business_text


def _compact_column_line(line: str) -> str:
    body = line[4:]  # after "  - "
    name = body.split("|", 1)[0].strip()
    # Drop trailing group annotation from the name display? Keep name as rendered before |.
    name = re.sub(r"\s*\[group: [^\]]+\]\s*$", "", name).strip()
    kinds_m = _KINDS_RE.search(line)
    kinds = kinds_m.group(1).strip() if kinds_m else "none"
    samples = _column_samples(line, limit=3)
    if samples:
        eg = ", ".join(f'"{sample}"' for sample in samples)
    else:
        eg = "-"
    return f"  - {name} | kinds {kinds} | e.g. {eg}"


def _column_samples(line: str, limit: int = 3) -> list[str]:
    eg_m = _EG_RE.search(line)
    if eg_m and eg_m.group(1).strip() not in {"-", ""}:
        found = _QUOTED.findall(eg_m.group(1))
        return [_short_sample(item) for item in found[:limit]]
    # Fall back to quoted values elsewhere on the line (values / top lists).
    found = _QUOTED.findall(line)
    return [_short_sample(item) for item in found[:limit]]


def _short_sample(text: str, max_len: int = 40) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1] + "…"


def _core_rulebook(rules_text: str) -> str:
    """Keep only đơn hàng/doanh thu and đáp án sections when possible."""
    sections = _split_sections(rules_text)
    core = [
        body
        for heading, body in sections
        if any(key in _strip_accents(heading) for key in _CORE_RULE_KEYWORDS)
    ]
    if not core:
        return rules_text
    return "".join(core).rstrip() + "\n"


def _last_resort_trim(business_text: str) -> str:
    """Stub empty-sample columns (keep names), then drop non-sales tables."""
    parts = _TABLE_SPLIT.split(business_text)
    if len(parts) <= 1:
        return business_text

    # Pass 1: empty-sample columns become short stubs; names stay on kept tables.
    stubbed: list[str] = []
    stubbed_change = False
    for part in parts:
        if not part.startswith("### Table `"):
            stubbed.append(part)
            continue
        lines = part.splitlines(keepends=True)
        new_lines: list[str] = []
        for line in lines:
            raw = line.rstrip("\n")
            if raw.startswith("  - ") and not _column_has_samples(raw):
                stub = _empty_sample_column_line(raw)
                if stub != raw:
                    stubbed_change = True
                    new_lines.append(stub + "\n")
                    continue
            new_lines.append(line if line.endswith("\n") else line + "\n")
        stubbed.append("".join(new_lines))
    if stubbed_change:
        return "".join(stubbed)

    # Pass 2: drop tables that look unrelated to sales.
    out: list[str] = []
    dropped_table = False
    for part in parts:
        if not part.startswith("### Table `"):
            out.append(part)
            continue
        header = part.splitlines()[0] if part else ""
        match = _TABLE_HEADER.match(header)
        name = match.group(1) if match else ""
        kind = match.group(2) if match and match.group(2) else ""
        if _looks_non_sales(name, kind):
            dropped_table = True
            continue
        out.append(part)
    if dropped_table:
        return "".join(out)
    return business_text


def _column_has_samples(line: str) -> bool:
    return bool(_column_samples(line, limit=1))


def _empty_sample_column_line(line: str) -> str:
    body = line[4:]
    name = body.split("|", 1)[0].strip()
    name = re.sub(r"\s*\[group: [^\]]+\]\s*$", "", name).strip()
    kinds_m = _KINDS_RE.search(line)
    kinds = kinds_m.group(1).strip() if kinds_m else "none"
    return f"  - {name} | kinds {kinds} | e.g. -"


def _looks_non_sales(name: str, kind: str) -> bool:
    blob = _strip_accents(f"{name} {kind}")
    if any(hint in blob for hint in _NON_SALES_HINTS):
        return True
    if any(hint in blob for hint in _SALES_HINTS):
        return False
    # Unknown tables are kept.
    return False


def _truncate_readme_tail(business_text: str) -> str:
    """Cut the README section from the end while keeping table cards intact."""
    table_start = business_text.find("### Table `")
    if table_start <= 0:
        if len(business_text) < 200:
            return business_text
        return business_text[: max(0, len(business_text) - 200)].rstrip() + "\n"
    head = business_text[:table_start]
    tail = business_text[table_start:]
    readme_marker = "## README"
    idx = head.find(readme_marker)
    if idx < 0:
        return business_text
    before = head[:idx]
    readme = head[idx:]
    if len(readme) < 100:
        return business_text
    cut = max(len(readme_marker) + 1, len(readme) - 200)
    new_readme = readme[:cut].rstrip() + "\n\n"
    return before + new_readme + tail

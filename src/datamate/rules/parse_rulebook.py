"""Deterministic rulebook.md table parser (no LLM)."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)


@dataclass(frozen=True)
class RuleRow:
    """One data row from a markdown rule table."""

    id: str
    section_path: str
    name: str
    text: str


def rule_row_id(section_path: str, name: str) -> str:
    """Stable id from section path + rule name (NFC)."""
    payload = f"{unicodedata.normalize('NFC', section_path)}\0{unicodedata.normalize('NFC', name)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def _strip_cell(raw: str) -> str:
    text = _BR_RE.sub("\n", raw)
    text = _BOLD_RE.sub(r"\1", text)
    # Collapse whitespace inside a line; keep intentional newlines from <br>.
    lines = [" ".join(part.split()) for part in text.split("\n")]
    return _nfc("\n".join(line for line in lines if line).strip())


def _split_row(line: str) -> list[str]:
    body = line.strip().removeprefix("|").removesuffix("|")
    return [_strip_cell(cell) for cell in body.split("|")]


def _is_separator(cells: list[str]) -> bool:
    if not cells:
        return False
    for cell in cells:
        cleaned = cell.replace(":", "").replace("-", "").replace(" ", "")
        if cleaned:
            return False
    return True


def parse_rulebook_markdown(text: str) -> list[RuleRow]:
    """Parse every markdown table under any heading into RuleRow entries."""
    lines = text.splitlines()
    stack: list[str] = []
    rows: list[RuleRow] = []
    index = 0
    while index < len(lines):
        heading = _HEADING_RE.match(lines[index])
        if heading:
            level = len(heading.group(1))
            title = _strip_cell(heading.group(2))
            stack = stack[: level - 1]
            stack.append(title)
            index += 1
            continue

        if not lines[index].strip().startswith("|"):
            prose = _prose_row(lines[index], " > ".join(stack))
            if prose is not None:
                rows.append(prose)
            index += 1
            continue

        table_lines: list[str] = []
        while index < len(lines) and lines[index].strip().startswith("|"):
            table_lines.append(lines[index])
            index += 1
            # Continuation: indented non-pipe lines appended to the previous cell via <br>
            # are rare; also accept a bare line that continues a cell after a trailing '\'.
            while (
                index < len(lines)
                and lines[index].strip()
                and not lines[index].strip().startswith("|")
                and not _HEADING_RE.match(lines[index])
                and table_lines
                and table_lines[-1].rstrip().endswith("\\")
            ):
                prev = table_lines[-1].rstrip()[:-1]
                table_lines[-1] = prev + "<br>" + lines[index].strip()
                index += 1

        if len(table_lines) < 2:
            continue
        header = _split_row(table_lines[0])
        if len(header) < 2:
            continue
        body_start = 1
        if _is_separator(_split_row(table_lines[1])):
            body_start = 2
        section_path = " > ".join(stack)
        for raw in table_lines[body_start:]:
            cells = _split_row(raw)
            if len(cells) < 2:
                continue
            name, text = cells[0], cells[1]
            if not name:
                continue
            # Skip a repeated header row inside the table body.
            if name.casefold() == header[0].casefold() and text.casefold() == header[1].casefold():
                continue
            rows.append(
                RuleRow(
                    id=rule_row_id(section_path, name),
                    section_path=section_path,
                    name=name,
                    text=text,
                )
            )
    return rows


_LEAD_BOLD_RE = re.compile(r"^\*\*([^*]+?)\*\*\s*:?\s*")


def _prose_row(line: str, section_path: str) -> RuleRow | None:
    """A bullet or paragraph line outside tables (rules are also written as prose)."""
    raw = line.strip()
    if not raw or raw.startswith("```"):
        return None
    if raw[:2] in {"- ", "* "}:
        raw = raw[2:].strip()
    lead = _LEAD_BOLD_RE.match(raw)
    text = _strip_cell(raw)
    if not text:
        return None
    if lead:
        name = _nfc(lead.group(1).strip().rstrip(":"))
    else:
        name = " ".join(text.split()[:8])
    return RuleRow(
        id=rule_row_id(section_path, name),
        section_path=section_path,
        name=name,
        text=text,
    )


def parse_rulebook(path: Path | str) -> list[RuleRow]:
    content = Path(path).read_text(encoding="utf-8")
    return parse_rulebook_markdown(content)


def rulebook_sha256(path: Path | str) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest()

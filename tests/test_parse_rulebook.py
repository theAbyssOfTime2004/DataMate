"""Deterministic rulebook.md table parser."""

from __future__ import annotations

from datamate.rules.parse_rulebook import parse_rulebook_markdown, rule_row_id


def test_parse_nested_headings_bold_and_multiline_cells() -> None:
    md = """# Book Title

## 1. Sales

| Luật | Mặc định |
| --- | --- |
| **Revenue statuses** | Completed and shipping count. |
| Returns | Drop the whole order when any line\\
 is returned. |

### Cafe

| Luật | Mặc định |
|---|---|
| Size M | Name ends with (M). |
"""
    rows = parse_rulebook_markdown(md)
    assert len(rows) == 3
    assert rows[0].name == "Revenue statuses"
    assert "Completed" in rows[0].text
    assert rows[0].section_path == "Book Title > 1. Sales"
    assert rows[1].name == "Returns"
    assert "returned" in rows[1].text
    assert "\n" in rows[1].text or "line" in rows[1].text
    assert rows[2].section_path.endswith("Cafe")
    assert rows[2].name == "Size M"
    # Stable ids.
    assert rows[0].id == rule_row_id(rows[0].section_path, rows[0].name)
    assert rows[0].id != rows[1].id


def test_prose_bullets_become_rows() -> None:
    from datamate.rules.parse_rulebook import parse_rulebook_markdown

    rows = parse_rulebook_markdown(
        "## 9. Chấm\n- **Làm tròn:** phần trăm làm tròn 3 chữ số.\nĐoạn văn thường ở đây.\n"
    )
    names = [r.name for r in rows]
    assert "Làm tròn" in names
    assert any(r.text.startswith("Đoạn văn") for r in rows)
    assert all(r.section_path == "9. Chấm" for r in rows)

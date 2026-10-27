from __future__ import annotations

import argparse
import re
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

from datamate.config import settings
from datamate.io.raw import BusinessRaw, FileEntry, RawTable, load_business_cached

FormatKind = Literal[
    "datetime_iso",
    "date_dmy",
    "date_dm",
    "date_thg",
    "int_serial_range",
    "money_text",
    "number",
    "qty_with_unit",
    "id_like",
    "phone",
    "bool",
    "excel_error",
    "text",
]

_EXCEL_ERRORS = frozenset(
    {
        "#VALUE!",
        "#N/A",
        "#REF!",
        "#DIV/0!",
        "#NAME?",
        "#NUM!",
        "#NULL!",
        "#CALC!",
        "#SPILL!",
        "#GETTING_DATA",
    }
)

_DATETIME_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?$")
_DATE_DMY_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?$")
_DATE_DM_RE = re.compile(r"^\d{1,2}/\d{1,2}$")
_DATE_THG_RE = re.compile(r"^\d{1,2}\s+thg\s+\d{1,2}(?:\s+\d{2,4})?$", re.IGNORECASE)
_PURE_INT_RE = re.compile(r"^[+-]?\d+$")
_PHONE_RE = re.compile(r"^0(?:\s*\d){9,10}$")
_MONEY_TEXT_RE = re.compile(
    r"^[+-]?(?:"
    r"\d+(?:[.,\s]\d+)*\s*(?:đ|đồng|vnd|k|tr|triệu)\b"
    r"|\d+(?:[.,]\d+)?\s*(?:tr|k)\s*\d+"
    r"|\d{1,3}(?:,\d{3})+(?:\.\d+)?"
    r"|\d{1,3}(?:\.\d{3})+(?:,\d+)?"
    r"|\d{1,3}(?:\s\d{3})+"
    r"|\d+(?:,\d{3})*\.\d{4}"
    r")$",
    re.IGNORECASE,
)
_PLAIN_NUMBER_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")
_QTY_WITH_UNIT_RE = re.compile(
    r"^[+-]?\d+(?:[.,]\d+)?\s+[a-zA-ZÀ-ỹ][a-zA-ZÀ-ỹ0-9\s/²³]*$"
)
_ID_LIKE_RE = re.compile(r"^[A-Za-z]{1,10}[_-]?\d{2,}[A-Za-z0-9_-]*$")
_BOOL_RE = re.compile(r"^(?:True|False)$", re.IGNORECASE)


def classify_value_kind(raw: str) -> FormatKind:
    s = raw.strip()
    if not s:
        return "text"
    upper = s.upper()
    if upper.startswith("#") and upper in _EXCEL_ERRORS:
        return "excel_error"
    if _DATETIME_ISO_RE.match(s):
        return "datetime_iso"
    if _DATE_DMY_RE.match(s):
        return "date_dmy"
    if _DATE_DM_RE.match(s):
        return "date_dm"
    if _DATE_THG_RE.match(s):
        return "date_thg"
    if _PHONE_RE.match(s):
        return "phone"
    if _PURE_INT_RE.match(s):
        val = int(s)
        if 20000 <= val <= 80000:
            return "int_serial_range"
    if _MONEY_TEXT_RE.match(s):
        return "money_text"
    if _PLAIN_NUMBER_RE.match(s):
        return "number"
    if _QTY_WITH_UNIT_RE.match(s):
        return "qty_with_unit"
    if _ID_LIKE_RE.match(s):
        return "id_like"
    if _BOOL_RE.match(s):
        return "bool"
    return "text"


def _truncate(text: str, max_len: int = 40) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


@dataclass
class ColumnProfile:
    name: str
    group: str | None
    non_null: int
    null_ratio: float
    n_distinct: int
    top_values: list[tuple[str, int]] = field(default_factory=list)
    all_values: list[tuple[str, int]] = field(default_factory=list)
    samples: list[str] = field(default_factory=list)
    kinds: list[tuple[str, float]] = field(default_factory=list)
    n_leading_trailing_space: int = 0
    n_excel_error: int = 0


@dataclass
class TableProfile:
    name: str
    kind: str
    n_sources: int
    source_files: list[str]
    sheet_names: list[str]
    header_rows: list[int]
    n_rows: int
    n_exact_duplicate_rows: int
    dropped_counts: dict[str, int]
    dropped_previews: dict[str, list[list[str]]]
    notes: list[str]
    columns: list[ColumnProfile]


@dataclass
class BusinessProfile:
    business_name: str
    readme_text: str | None
    files: list[FileEntry]
    tables: list[TableProfile]


def profile_table(table: RawTable) -> TableProfile:
    n_rows = len(table.df)

    group_map: dict[str, str] = {}
    for src in table.sources:
        for col, grp in src.header_groups.items():
            if col not in group_map:
                group_map[col] = grp

    col_profiles: list[ColumnProfile] = []
    for col in table.columns:
        series = table.df[col]
        non_null_vals: list[str] = [v for v in series.tolist() if isinstance(v, str)]
        non_null = len(non_null_vals)
        null_ratio = round((n_rows - non_null) / n_rows, 2) if n_rows > 0 else 0.0

        val_counts = Counter(non_null_vals)
        n_distinct = len(val_counts)

        sorted_vals = sorted(val_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        top_values = [(_truncate(val), cnt) for val, cnt in sorted_vals[:5]]
        all_values = (
            [(_truncate(val), cnt) for val, cnt in sorted_vals]
            if n_distinct <= 20
            else []
        )

        samples: list[str] = []
        seen_samples: set[str] = set()
        for v in non_null_vals:
            if v not in seen_samples:
                seen_samples.add(v)
                samples.append(_truncate(v))
                if len(samples) == 5:
                    break

        kind_counts: Counter[str] = Counter()
        n_space = 0
        n_err = 0
        for val, cnt in val_counts.items():
            if val != val.strip():
                n_space += cnt
            k = classify_value_kind(val)
            if k == "excel_error":
                n_err += cnt
            kind_counts[k] += cnt

        kinds: list[tuple[str, float]] = []
        if non_null > 0:
            sorted_kinds = sorted(kind_counts.items(), key=lambda kv: (-kv[1], kv[0]))
            for k, c in sorted_kinds:
                share = c / non_null
                if share >= 0.02 or not kinds:
                    kinds.append((k, round(share, 2)))
                if len(kinds) == 5:
                    break

        col_profiles.append(
            ColumnProfile(
                name=col,
                group=group_map.get(col),
                non_null=non_null,
                null_ratio=null_ratio,
                n_distinct=n_distinct,
                top_values=top_values,
                all_values=all_values,
                samples=samples,
                kinds=kinds,
                n_leading_trailing_space=n_space,
                n_excel_error=n_err,
            )
        )

    if n_rows > 0 and table.columns:
        n_exact_duplicate_rows = int(table.df.duplicated(subset=table.columns).sum())
    else:
        n_exact_duplicate_rows = 0

    seen_files: list[str] = []
    seen_sheets: list[str] = []
    seen_hrows: list[int] = []
    for s in table.sources:
        if s.file not in seen_files:
            seen_files.append(s.file)
        if s.sheet is not None and s.sheet not in seen_sheets:
            seen_sheets.append(s.sheet)
        if s.header_row not in seen_hrows:
            seen_hrows.append(s.header_row)

    dropped_counts: dict[str, int] = {}
    dropped_previews: dict[str, list[list[str]]] = {}
    for d in table.dropped:
        dropped_counts[d.reason] = dropped_counts.get(d.reason, 0) + 1
        prev_list = dropped_previews.setdefault(d.reason, [])
        if d.preview and len(prev_list) < 3:
            prev_list.append(d.preview)

    return TableProfile(
        name=table.name,
        kind=table.kind,
        n_sources=len(table.sources),
        source_files=seen_files,
        sheet_names=seen_sheets,
        header_rows=sorted(seen_hrows),
        n_rows=n_rows,
        n_exact_duplicate_rows=n_exact_duplicate_rows,
        dropped_counts=dict(sorted(dropped_counts.items())),
        dropped_previews=dropped_previews,
        notes=list(table.notes),
        columns=col_profiles,
    )


def profile_business(biz_raw: BusinessRaw) -> BusinessProfile:
    readme_path = biz_raw.business_dir / "README.md"
    readme_text: str | None = None
    if readme_path.is_file():
        readme_text = unicodedata.normalize(
            "NFC", readme_path.read_text(encoding="utf-8", errors="replace").strip()
        )
    table_profiles = [profile_table(t) for t in biz_raw.tables]
    return BusinessProfile(
        business_name=unicodedata.normalize("NFC", biz_raw.business_dir.name),
        readme_text=readme_text,
        files=list(biz_raw.files),
        tables=table_profiles,
    )


def _format_capped_list(items: list[str], cap: int = 5) -> str:
    if not items:
        return "-"
    shown = items[:cap]
    rest = len(items) - len(shown)
    if rest > 0:
        return ", ".join(shown) + f" (+{rest} more)"
    return ", ".join(shown)


def render_card(table_profile: TableProfile) -> str:
    tp = table_profile
    files_str = _format_capped_list(tp.source_files, 5)
    sheets_str = _format_capped_list(tp.sheet_names, 5)
    hrows_str = ", ".join(str(r) for r in tp.header_rows)

    lines: list[str] = [
        f"### Table `{tp.name}` ({tp.kind})",
        (
            f"- sources: {tp.n_sources} | files: {files_str} | sheets: {sheets_str} "
            f"| header_row: {hrows_str}"
        ),
        f"- rows: {tp.n_rows} | exact_duplicate_rows: {tp.n_exact_duplicate_rows}",
    ]

    if tp.dropped_counts:
        parts: list[str] = []
        for reason, cnt in tp.dropped_counts.items():
            prevs = tp.dropped_previews.get(reason, [])
            if prevs:
                sample_str = "; ".join("[" + ", ".join(f'"{c}"' for c in p) + "]" for p in prevs)
                parts.append(f"{reason}={cnt} (e.g. {sample_str})")
            else:
                parts.append(f"{reason}={cnt}")
        lines.append(f"- dropped: {' | '.join(parts)}")

    if tp.notes:
        lines.append(f"- notes: {', '.join(tp.notes)}")

    lines.append("- columns:")
    for col in tp.columns:
        grp_str = f" [group: {col.group}]" if col.group else ""
        kinds_str = ", ".join(f"{k} {ratio:.2f}" for k, ratio in col.kinds) if col.kinds else "none"
        flags: list[str] = []
        if col.n_leading_trailing_space > 0:
            flags.append(f"untrimmed={col.n_leading_trailing_space}")
        if col.n_excel_error > 0:
            flags.append(f"excel_error={col.n_excel_error}")
        flag_str = f" | flags: {', '.join(flags)}" if flags else ""

        if 0 < col.n_distinct <= 20:
            vals_source = col.all_values if col.all_values else col.top_values
            vals_str = ", ".join(f'"{v}" {cnt}' for v, cnt in vals_source)
            tail_str = f"| values ({col.n_distinct}): {vals_str}"
        elif col.n_distinct == 0:
            tail_str = "| values (0): -"
        else:
            top_str = ", ".join(f'"{v}" {cnt}' for v, cnt in col.top_values)
            eg_str = ", ".join(f'"{s}"' for s in col.samples) if col.samples else "-"
            tail_str = f"| top: {top_str} ({len(col.top_values)}) | e.g. {eg_str}"

        lines.append(
            f"  - {col.name}{grp_str} | null {col.null_ratio:.2f} | distinct {col.n_distinct} "
            f"| kinds {kinds_str}{flag_str} {tail_str}"
        )

    return "\n".join(lines)


def render_business(business_profile: BusinessProfile, include_readme: bool = True) -> str:
    bp = business_profile
    lines: list[str] = [f"# Business `{bp.business_name}`", ""]

    manifest_parts = [
        f"`{f.name}` ({f.status}{': ' + f.error if f.error else ''})" for f in bp.files
    ]
    lines.append(f"**Files ({len(bp.files)})**: " + ", ".join(manifest_parts))
    lines.append("")

    if include_readme and bp.readme_text:
        lines.append("## README")
        lines.append(bp.readme_text)
        lines.append("")

    lines.append(f"## Tables ({len(bp.tables)})")
    lines.append("")
    for tp in bp.tables:
        lines.append(render_card(tp))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _profile_one_business_id(split: str, business_id: str) -> BusinessProfile:
    biz_raw = load_business_cached(split, business_id)
    return profile_business(biz_raw)


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Profile raw tables for DataMate businesses.")
    parser.add_argument("--split", default="train", help="Split name (default: train)")
    parser.add_argument("--business", default=None, help="Single business ID to profile")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Profile all businesses in the split and write cards to cache/profiles/<split>/",
    )
    args = parser.parse_args(argv)

    if not args.all and not args.business:
        parser.error("Provide either --business <id> or --all")

    if args.business and not args.all:
        bp = _profile_one_business_id(args.split, args.business)
        print(render_business(bp, include_readme=True))
        return

    split_dir = settings.split_dir(args.split) / "data"
    biz_dirs = sorted(
        (d for d in split_dir.iterdir() if d.is_dir()),
        key=lambda d: d.name.casefold(),
    )
    out_dir = settings.cache_dir / "profiles" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    for bdir in biz_dirs:
        t0 = time.perf_counter()
        bp = _profile_one_business_id(args.split, bdir.name)
        card_md = render_business(bp, include_readme=True)
        (out_dir / f"{bdir.name}.md").write_text(card_md, encoding="utf-8")
        elapsed = time.perf_counter() - t0

        total_rows = sum(t.n_rows for t in bp.tables)
        dropped_tot: Counter[str] = Counter()
        for t in bp.tables:
            for r, c in t.dropped_counts.items():
                dropped_tot[r] += c
        dropped_str = (
            ", ".join(f"{r}={c}" for r, c in sorted(dropped_tot.items()))
            if dropped_tot
            else "none"
        )
        special_files = [
            f"{f.name}:{f.status}"
            for f in bp.files
            if f.status in {"error", "unsupported"}
        ]
        special_str = ", ".join(special_files) if special_files else "none"
        print(
            f"{bdir.name}: tables={len(bp.tables)} | rows={total_rows} "
            f"| dropped=({dropped_str}) | special_files=({special_str}) | {elapsed:.2f}s"
        )


if __name__ == "__main__":
    main()

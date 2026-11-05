from __future__ import annotations

import csv
import datetime
import hashlib
import io
import math
import pickle
import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import openpyxl
import pandas as pd

from datamate.config import settings
from datamate.io.paths import business_dir

READER_VERSION = "1.2"

DropReason = Literal["title_or_filter", "summary", "repeated_header", "empty"]
TableKind = Literal["csv", "xlsx", "sqlite"]
FileStatus = Literal["read", "readme", "unsupported", "error"]

PROVENANCE_COLUMNS: tuple[str, str, str] = ("__file", "__sheet", "__row")

_SUMMARY_DIACRITIC_RE = re.compile(r"^\s*(tổng|cộng|ghi chú|total|subtotal)\b", re.IGNORECASE)
_SUMMARY_STRIPPED_RE = re.compile(
    r"^\s*(tong cong|tong ngay|tong thang|tong ky|cong ngay|cong thang|ghi chu\s*:)",
    re.IGNORECASE,
)
_SUMMARY_ANY_CELLS_DIACRITIC_RE = re.compile(
    r"^\s*(tổng cộng|tổng|cộng)(\s+(ngày|tháng|kỳ|ca|tuần|năm)\b|\s+\d|\s*:?\s*$)",
    re.IGNORECASE,
)
_SUMMARY_ANY_CELLS_STRIPPED_RE = re.compile(
    r"^\s*(tong cong|tong ngay|tong thang|cong ngay|cong thang)\b",
    re.IGNORECASE,
)
_NUMBER_OR_MONEY_RE = re.compile(
    r"^[+-]?(?:"
    r"\d+(?:[.,\s]\d+)*(?:\s*(?:đ|d|đồng|dong|vnd|k|tr|triệu|trieu|%))?"
    r"|\d+(?:[.,]\d+)?\s*(?:tr|k)\s*\d+"
    r")$",
    re.IGNORECASE,
)
_DATE_LIKE_RE = re.compile(
    r"^(?:"
    r"\d{4}-\d{1,2}-\d{1,2}(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?"
    r"|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?"
    r"|\d{1,2}\s+thg\s+\d{1,2}(?:\s+\d{2,4})?"
    r"|\d{1,2}:\d{2}(?::\d{2})?"
    r")$",
    re.IGNORECASE,
)
_BOOL_OR_ERROR_RE = re.compile(r"^(?:true|false|#[A-Z0-9/!]+)$", re.IGNORECASE)
_DIGIT_RUN_RE = re.compile(r"\d+")


@dataclass
class DroppedRow:
    file: str
    sheet: str | None
    row_no: int
    reason: DropReason
    preview: list[str] = field(default_factory=list)


@dataclass
class SourceRef:
    file: str
    sheet: str | None
    header_row: int
    header_groups: dict[str, str] = field(default_factory=dict)


@dataclass
class RawTable:
    name: str
    kind: TableKind
    sources: list[SourceRef]
    columns: list[str]
    df: pd.DataFrame
    dropped: list[DroppedRow] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class FileEntry:
    name: str
    status: FileStatus
    error: str | None = None

    @property
    def file(self) -> str:
        return self.name


@dataclass
class BusinessRaw:
    business_dir: Path
    tables: list[RawTable]
    files: list[FileEntry]


def cell_to_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value == "" or value.strip() == "":
            return None
        return unicodedata.normalize("NFC", value)
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return str(int(value)) if value.is_integer() else repr(value)
    if isinstance(value, datetime.datetime):
        return value.replace(microsecond=0).isoformat(sep=" ")
    if isinstance(value, datetime.date):
        return value.isoformat()
    if isinstance(value, datetime.time):
        return value.replace(microsecond=0).isoformat()
    text = str(value)
    if text.strip() == "":
        return None
    return unicodedata.normalize("NFC", text)


def _strip_accents(text: str) -> str:
    nfd = unicodedata.normalize("NFD", text.lower())
    without_marks = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
    return without_marks.replace("đ", "d")


def _clean_label(text: str) -> str:
    nfc = unicodedata.normalize("NFC", text)
    return " ".join(nfc.split())


def _trim_row(row: list[str | None]) -> list[str | None]:
    end = len(row)
    while end > 0 and row[end - 1] is None:
        end -= 1
    return row[:end]


def _make_preview(row: list[str | None]) -> list[str]:
    out: list[str] = []
    for cell in row:
        if cell is not None:
            out.append(cell[:40])
            if len(out) == 4:
                break
    return out


def _is_header_text_cell(cell: str) -> bool:
    s = cell.strip()
    if not s:
        return False
    if _NUMBER_OR_MONEY_RE.match(s):
        return False
    if _DATE_LIKE_RE.match(s):
        return False
    if _BOOL_OR_ERROR_RE.match(s):
        return False
    return any(ch.isalpha() for ch in s)


def _is_summary_cell(cell: str) -> bool:
    nfc_lower = " ".join(unicodedata.normalize("NFC", cell).lower().split())
    if _SUMMARY_DIACRITIC_RE.match(nfc_lower) is not None:
        return True
    stripped = _strip_accents(nfc_lower)
    m = _SUMMARY_STRIPPED_RE.match(stripped)
    return m is not None and nfc_lower[: m.end()] == stripped[: m.end()]


def _is_summary_cell_any_width(cell: str) -> bool:
    nfc_lower = " ".join(unicodedata.normalize("NFC", cell).lower().split())
    if _SUMMARY_ANY_CELLS_DIACRITIC_RE.match(nfc_lower) is not None:
        return True
    stripped = _strip_accents(nfc_lower)
    m = _SUMMARY_ANY_CELLS_STRIPPED_RE.match(stripped)
    return m is not None and nfc_lower[: m.end()] == stripped[: m.end()]


def _is_summary_row(non_empty_cells: list[str]) -> bool:
    if not non_empty_cells:
        return False
    if len(non_empty_cells) <= 3 and any(_is_summary_cell(c) for c in non_empty_cells):
        return True
    return any(_is_summary_cell_any_width(c) for c in non_empty_cells)


def _is_repeated_header(row: list[str | None], header_cells: list[str | None]) -> bool:
    header_non_empty = [c for c in header_cells if c is not None]
    row_non_empty = [c for c in row if c is not None]
    if len(header_non_empty) < 2 or len(row_non_empty) != len(header_non_empty):
        return False
    width = max(len(row), len(header_cells))
    for j in range(width):
        rc = row[j] if j < len(row) else None
        hc = header_cells[j] if j < len(header_cells) else None
        if (rc is None) != (hc is None):
            return False
        if rc is not None and hc is not None and _clean_label(rc) != _clean_label(hc):
            return False
    return True


def _dedupe_column_names(raw_names: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    result: list[str] = []
    for name in raw_names:
        cnt = counts.get(name, 0) + 1
        counts[name] = cnt
        result.append(name if cnt == 1 else f"{name}__{cnt}")
    return result


def _build_dataframe(
    cols: list[str],
    rows_with_prov: list[tuple[list[str | None], str, str | None, int]],
) -> pd.DataFrame:
    n_rows = len(rows_with_prov)
    n_cols = len(cols)
    if n_rows == 0:
        data_dict: dict[str, pd.Series] = {
            c: pd.Series([], dtype=object) for c in cols
        }
        data_dict["__file"] = pd.Series([], dtype=object)
        data_dict["__sheet"] = pd.Series([], dtype=object)
        data_dict["__row"] = pd.Series([], dtype="int64")
        return pd.DataFrame(data_dict)

    col_lists: list[list[str | None]] = [[None] * n_rows for _ in range(n_cols)]
    files: list[str] = [""] * n_rows
    sheets: list[str | None] = [None] * n_rows
    row_nos: list[int] = [0] * n_rows

    for r_idx, (vals, f_name, s_name, r_no) in enumerate(rows_with_prov):
        for c_idx in range(n_cols):
            col_lists[c_idx][r_idx] = vals[c_idx]
        files[r_idx] = f_name
        sheets[r_idx] = s_name
        row_nos[r_idx] = r_no

    series_dict: dict[str, pd.Series] = {
        cols[c_idx]: pd.Series(col_lists[c_idx], dtype=object)
        for c_idx in range(n_cols)
    }
    series_dict["__file"] = pd.Series(files, dtype=object)
    series_dict["__sheet"] = pd.Series(sheets, dtype=object)
    series_dict["__row"] = pd.Series(row_nos, dtype="int64")
    return pd.DataFrame(series_dict)


@dataclass
class _ParsedSource:
    kind: TableKind
    file: str
    sheet: str | None
    sheet_index: int
    header_row: int
    header_groups: dict[str, str]
    columns: list[str]
    rows: list[tuple[int, list[str | None]]]
    dropped: list[DroppedRow]
    notes: list[str]


def _extract_grid_to_source(
    kind: TableKind,
    file_name: str,
    sheet_name: str | None,
    sheet_index: int,
    raw_rows: list[list[str | None]],
    initial_notes: list[str] | None = None,
    raw_row_lengths: list[int] | None = None,
) -> _ParsedSource | None:
    notes = list(initial_notes) if initial_notes else []

    trimmed_rows: list[list[str | None]] = [_trim_row(r) for r in raw_rows]
    total_end = len(trimmed_rows)
    while total_end > 0 and len(trimmed_rows[total_end - 1]) == 0:
        total_end -= 1
    trimmed_rows = trimmed_rows[:total_end]

    if not trimmed_rows:
        return None

    first_30 = trimmed_rows[:30]
    max_w = max((sum(1 for c in r if c is not None) for r in first_30), default=0)
    min_non_empty = 1 if max_w <= 1 else max(2, math.ceil(0.6 * max_w))

    header_idx: int | None = None
    for idx, r in enumerate(first_30):
        non_empty = [c for c in r if c is not None]
        if len(non_empty) < min_non_empty:
            continue
        text_cnt = sum(1 for c in non_empty if _is_header_text_cell(c))
        if text_cnt / len(non_empty) >= 0.8:
            header_idx = idx
            break

    if header_idx is None:
        for idx, r in enumerate(trimmed_rows):
            if any(c is not None for c in r):
                header_idx = idx
                break
    assert header_idx is not None

    header_row_no = header_idx + 1
    header_cells = trimmed_rows[header_idx]
    header_non_empty_cnt = sum(1 for c in header_cells if c is not None)

    dropped: list[DroppedRow] = []
    for idx in range(header_idx):
        r = trimmed_rows[idx]
        row_no = idx + 1
        if not any(c is not None for c in r):
            dropped.append(
                DroppedRow(file=file_name, sheet=sheet_name, row_no=row_no, reason="empty", preview=[])
            )
        else:
            dropped.append(
                DroppedRow(
                    file=file_name,
                    sheet=sheet_name,
                    row_no=row_no,
                    reason="title_or_filter",
                    preview=_make_preview(r),
                )
            )

    upper_tier_row: list[str | None] | None = None
    if header_idx >= 1:
        candidate_upper = trimmed_rows[header_idx - 1]
        upper_non_empty = [c for c in candidate_upper if c is not None]
        if (
            2 <= len(upper_non_empty) < header_non_empty_cnt
            and all(_is_header_text_cell(c) and ":" not in c for c in upper_non_empty)
        ):
            upper_tier_row = candidate_upper

    kept_data_rows: list[tuple[int, list[str | None]]] = []

    for idx in range(header_idx + 1, len(trimmed_rows)):
        r = trimmed_rows[idx]
        row_no = idx + 1
        non_empty = [c for c in r if c is not None]
        if not non_empty:
            dropped.append(
                DroppedRow(file=file_name, sheet=sheet_name, row_no=row_no, reason="empty", preview=[])
            )
            continue
        if _is_repeated_header(r, header_cells):
            dropped.append(
                DroppedRow(
                    file=file_name,
                    sheet=sheet_name,
                    row_no=row_no,
                    reason="repeated_header",
                    preview=_make_preview(r),
                )
            )
            continue
        if _is_summary_row(non_empty):
            dropped.append(
                DroppedRow(
                    file=file_name,
                    sheet=sheet_name,
                    row_no=row_no,
                    reason="summary",
                    preview=_make_preview(r),
                )
            )
            continue
        kept_data_rows.append((row_no, r))

    if raw_row_lengths is not None:
        header_width = (
            raw_row_lengths[header_idx]
            if header_idx < len(raw_row_lengths)
            else len(header_cells)
        )
        num_cols = max(len(header_cells), header_width)
        ragged_count = 0
        normalized_data_rows: list[tuple[int, list[str | None]]] = []
        for row_no, r in kept_data_rows:
            orig_len = raw_row_lengths[row_no - 1] if row_no - 1 < len(raw_row_lengths) else len(r)
            if orig_len != header_width or len(r) > num_cols:
                ragged_count += 1
            padded = (r + [None] * num_cols)[:num_cols]
            normalized_data_rows.append((row_no, padded))
        if ragged_count > 0:
            notes.append(f"ragged_rows={ragged_count}")
        kept_data_rows = normalized_data_rows
    else:
        num_cols = max([len(header_cells)] + [len(r) for _, r in kept_data_rows], default=0)
        kept_data_rows = [
            (row_no, r + [None] * (num_cols - len(r))) for row_no, r in kept_data_rows
        ]

    padded_header = header_cells + [None] * (num_cols - len(header_cells))

    ffill_groups: list[str | None] = [None] * num_cols
    if upper_tier_row is not None:
        curr_grp: str | None = None
        for j in range(num_cols):
            cell = upper_tier_row[j] if j < len(upper_tier_row) else None
            if cell is not None:
                curr_grp = _clean_label(cell)
            ffill_groups[j] = curr_grp

    kept_col_indices: list[int] = []
    for j in range(num_cols):
        if padded_header[j] is None and all(r[j] is None for _, r in kept_data_rows):
            notes.append(f"dropped_empty_column=col_{j + 1}")
        else:
            kept_col_indices.append(j)

    raw_col_names: list[str] = []
    for j in kept_col_indices:
        hc = padded_header[j]
        if hc is None:
            raw_col_names.append(f"col_{j + 1}")
        else:
            cleaned = _clean_label(hc)
            raw_col_names.append(cleaned if cleaned else f"col_{j + 1}")

    final_columns = _dedupe_column_names(raw_col_names)

    header_groups: dict[str, str] = {}
    if upper_tier_row is not None:
        for col_name, j in zip(final_columns, kept_col_indices, strict=True):
            grp = ffill_groups[j]
            if grp is not None:
                header_groups[col_name] = grp

    final_rows: list[tuple[int, list[str | None]]] = [
        (row_no, [r[j] for j in kept_col_indices]) for row_no, r in kept_data_rows
    ]

    return _ParsedSource(
        kind=kind,
        file=file_name,
        sheet=sheet_name,
        sheet_index=sheet_index,
        header_row=header_row_no,
        header_groups=header_groups,
        columns=final_columns,
        rows=final_rows,
        dropped=dropped,
        notes=notes,
    )


def _read_xlsx_file(path: Path) -> list[_ParsedSource]:
    file_name = unicodedata.normalize("NFC", path.name)
    sources: list[_ParsedSource] = []
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        for sheet_idx, sheet_name in enumerate(wb.sheetnames):
            ws = wb[sheet_name]
            raw_rows: list[list[str | None]] = [
                [cell_to_str(cell) for cell in row]
                for row in ws.iter_rows(values_only=True)
            ]
            parsed = _extract_grid_to_source(
                kind="xlsx",
                file_name=file_name,
                sheet_name=unicodedata.normalize("NFC", sheet_name),
                sheet_index=sheet_idx,
                raw_rows=raw_rows,
            )
            if parsed is not None:
                sources.append(parsed)
    finally:
        wb.close()
    return sources


def _decode_csv_bytes(raw_bytes: bytes) -> tuple[str, str]:
    for enc in ("utf-8-sig", "cp1258", "latin-1"):
        try:
            return raw_bytes.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return raw_bytes.decode("latin-1", errors="replace"), "latin-1"


def _detect_csv_delimiter(text: str) -> str:
    first_line = ""
    for line in text.splitlines():
        if line.strip():
            first_line = line
            break
    candidates = [",", ";", "\t", "|"]
    counts = {d: first_line.count(d) for d in candidates}
    best = max(candidates, key=lambda d: counts[d])
    return best if counts[best] > 0 else ","


def _read_csv_file(path: Path) -> list[_ParsedSource]:
    file_name = unicodedata.normalize("NFC", path.name)
    raw_bytes = path.read_bytes()
    text, encoding = _decode_csv_bytes(raw_bytes)
    delim = _detect_csv_delimiter(text)
    notes = [f"encoding={encoding}"]
    if delim != ",":
        notes.append(f"delimiter={delim!r}")

    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delim)
    raw_rows: list[list[str | None]] = []
    raw_lengths: list[int] = []
    for row in reader:
        raw_lengths.append(len(row))
        raw_rows.append([cell_to_str(cell) for cell in row])

    if not raw_rows:
        return []

    parsed = _extract_grid_to_source(
        kind="csv",
        file_name=file_name,
        sheet_name=None,
        sheet_index=0,
        raw_rows=raw_rows,
        initial_notes=notes,
        raw_row_lengths=raw_lengths,
    )
    return [parsed] if parsed is not None else []


def _read_sqlite_file(path: Path) -> list[RawTable]:
    file_name = unicodedata.normalize("NFC", path.name)
    uri = f"file:{path.as_posix()}?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True)
    tables: list[RawTable] = []
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY rowid"
        )
        table_names = [unicodedata.normalize("NFC", str(r[0])) for r in cur.fetchall()]
        for tname in table_names:
            escaped = tname.replace('"', '""')
            data_cur = conn.execute(f'SELECT * FROM "{escaped}"')
            raw_cols = [
                _clean_label(str(desc[0])) if desc[0] is not None else f"col_{i + 1}"
                for i, desc in enumerate(data_cur.description or [])
            ]
            cols = _dedupe_column_names(raw_cols)
            rows_with_prov: list[tuple[list[str | None], str, str | None, int]] = []
            for row_idx, db_row in enumerate(data_cur, start=1):
                vals = [cell_to_str(val) for val in db_row]
                rows_with_prov.append((vals, file_name, tname, row_idx))

            df = _build_dataframe(cols, rows_with_prov)
            src_ref = SourceRef(
                file=file_name,
                sheet=tname,
                header_row=1,
                header_groups={},
            )
            tables.append(
                RawTable(
                    name=f"{file_name}!{tname}",
                    kind="sqlite",
                    sources=[src_ref],
                    columns=cols,
                    df=df,
                    dropped=[],
                    notes=[],
                )
            )
    finally:
        conn.close()
    return tables


def _digit_pattern(name: str) -> str:
    return _DIGIT_RUN_RE.sub("#", name)


def _merge_parsed_sources(sources: list[_ParsedSource]) -> list[RawTable]:
    groups: dict[tuple[Any, ...], list[_ParsedSource]] = {}
    for src in sources:
        file_stem_pat = _digit_pattern(Path(src.file).stem).casefold()
        sheet_pat = _digit_pattern(src.sheet or "").casefold()
        key = (src.kind, file_stem_pat, sheet_pat, tuple(src.columns))
        groups.setdefault(key, []).append(src)

    tables: list[RawTable] = []
    for group_sources in groups.values():
        first = group_sources[0]
        cols = first.columns
        distinct_files = {s.file for s in group_sources}
        distinct_sheets = {s.sheet for s in group_sources}

        if first.kind == "csv":
            if len(distinct_files) == 1:
                table_name = first.file
            else:
                table_name = f"{_digit_pattern(Path(first.file).stem)}.csv"
        else:
            file_label = (
                _digit_pattern(Path(first.file).stem)
                if len(distinct_files) > 1
                else Path(first.file).stem
            )
            sheet_label = (
                _digit_pattern(first.sheet or "")
                if len(distinct_sheets) > 1
                else (first.sheet or "")
            )
            table_name = f"{file_label}.xlsx!{sheet_label}"

        src_refs: list[SourceRef] = []
        all_dropped: list[DroppedRow] = []
        all_notes: list[str] = []
        rows_with_prov: list[tuple[list[str | None], str, str | None, int]] = []

        for s in group_sources:
            src_refs.append(
                SourceRef(
                    file=s.file,
                    sheet=s.sheet,
                    header_row=s.header_row,
                    header_groups=dict(s.header_groups),
                )
            )
            all_dropped.extend(s.dropped)
            for note in s.notes:
                if note not in all_notes:
                    all_notes.append(note)
            for row_no, vals in s.rows:
                rows_with_prov.append((vals, s.file, s.sheet, row_no))

        df = _build_dataframe(cols, rows_with_prov)

        tables.append(
            RawTable(
                name=table_name,
                kind=first.kind,
                sources=src_refs,
                columns=cols,
                df=df,
                dropped=all_dropped,
                notes=all_notes,
            )
        )
    return tables


def read_business(bdir: Path) -> BusinessRaw:
    bdir = Path(bdir)
    if not bdir.is_dir():
        raise FileNotFoundError(f"Business directory does not exist: {bdir}")

    file_paths = sorted(
        (p for p in bdir.iterdir() if p.is_file()),
        key=lambda p: (p.name.casefold(), p.name),
    )

    files_manifest: list[FileEntry] = []
    parsed_grid_sources: list[_ParsedSource] = []
    sqlite_tables: list[RawTable] = []

    for fpath in file_paths:
        fname = unicodedata.normalize("NFC", fpath.name)
        if fname.lower() == "readme.md":
            files_manifest.append(FileEntry(name=fname, status="readme"))
            continue

        suffix = fpath.suffix.lower()
        if suffix not in {".xlsx", ".csv", ".sqlite", ".db", ".sqlite3"}:
            files_manifest.append(FileEntry(name=fname, status="unsupported"))
            continue

        try:
            if suffix == ".xlsx":
                parsed_grid_sources.extend(_read_xlsx_file(fpath))
            elif suffix == ".csv":
                parsed_grid_sources.extend(_read_csv_file(fpath))
            else:
                sqlite_tables.extend(_read_sqlite_file(fpath))
            files_manifest.append(FileEntry(name=fname, status="read"))
        except Exception as exc:  # noqa: BLE001
            files_manifest.append(FileEntry(name=fname, status="error", error=str(exc)))

    merged_tables = _merge_parsed_sources(parsed_grid_sources)
    all_tables = [*merged_tables, *sqlite_tables]

    return BusinessRaw(
        business_dir=bdir,
        tables=all_tables,
        files=files_manifest,
    )


def _compute_business_cache_key(bdir: Path) -> str:
    hasher = hashlib.sha256()
    hasher.update(f"READER_VERSION={READER_VERSION}\n".encode())
    file_paths = sorted(
        (p for p in bdir.iterdir() if p.is_file()),
        key=lambda p: (
            unicodedata.normalize("NFC", p.name).casefold(),
            unicodedata.normalize("NFC", p.name),
        ),
    )
    for fpath in file_paths:
        fname = unicodedata.normalize("NFC", fpath.name)
        file_sha = hashlib.sha256(fpath.read_bytes()).hexdigest()
        hasher.update(f"{fname}:{file_sha}\n".encode())
    return hasher.hexdigest()


def business_files_key(bdir: Path) -> str:
    """Hash of the reader version and every file in a shop folder."""
    return _compute_business_cache_key(bdir)


def load_business_cached(split: str, business_id: str) -> BusinessRaw:
    bdir = business_dir(split, business_id)
    key = _compute_business_cache_key(bdir)
    cache_path = settings.cache_dir / "raw" / split / business_id / f"{key}.pkl"
    if cache_path.is_file():
        with cache_path.open("rb") as fh:
            cached: BusinessRaw = pickle.load(fh)
            return cached

    result = read_business(bdir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as fh:
        pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return result

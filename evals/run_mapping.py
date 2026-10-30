from __future__ import annotations

import argparse
import datetime
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datamate.config import settings
from datamate.etl import CanonicalData, load_or_build_canonical, write_duckdb
from datamate.io.profile import profile_business
from datamate.io.questions import load_questions, load_references
from datamate.io.raw import load_business_cached
from datamate.llm import get_client
from datamate.mapping import Mapping, infer_mapping, load_mapping_yaml
from evals.revenue_check import query_monthly_revenue
from evals.score_mapping import MappingScore, score

GOLD_DIR = PROJECT_ROOT / "evals" / "gold_mappings"

# Keep in sync with tests/test_etl_gold.py.
EXPECTED_REVENUES: list[tuple[str, str, int]] = [
    ("biz_101", "2026-01", 245_184_900),
    ("biz_101", "2026-03", 232_535_600),
    ("biz_101", "2026-07", 236_132_200),
    ("biz_105", "2025-08", 368_766_700),
    ("biz_105", "2025-09", 348_311_500),
    ("biz_105", "2025-10", 379_300_800),
    ("biz_105", "2025-11", 359_106_000),
    ("biz_105", "2025-12", 377_663_800),
    ("biz_105", "2026-01", 389_514_000),
    ("biz_105", "2026-02", 269_031_700),
    ("biz_105", "2026-03", 398_881_600),
    ("biz_106", "2025-05", 271_866_000),
    ("biz_106", "2025-11", 264_914_500),
    ("biz_106", "2025-12", 305_793_500),
    ("biz_106", "2026-01", 295_438_500),
    ("biz_111", "2025-03", 226_550_000),
    ("biz_123", "2025-10", 202_375_900),
]

_PROBE_RE = re.compile(r"doanh thu (của )?tháng \d{1,2}/\d{4}", re.IGNORECASE)
_PROBE_YM_RE = re.compile(r"tháng (\d{1,2})/(\d{4})", re.IGNORECASE)


def _unique_run_dir(base_runs_dir: Path, base_name: str) -> Path:
    candidate = base_runs_dir / base_name
    if not candidate.exists():
        return candidate
    suffix = 2
    while True:
        candidate = base_runs_dir / f"{base_name}_{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def _yn(ok: bool) -> str:
    return "yes" if ok else "no"


def _list_businesses(split: str) -> list[str]:
    data_dir = settings.split_dir(split) / "data"
    if not data_dir.is_dir():
        return []
    return sorted(
        p.name for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("biz_")
    )


def dump_mapping_yaml(mapping: Mapping, path: Path) -> None:
    payload = mapping.model_dump(mode="json")
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _error_issues(issues: list[Any]) -> list[Any]:
    return [iss for iss in issues if getattr(iss, "severity", None) == "error"]


@dataclass
class InferStats:
    n_calls: int = 0
    n_cache_hits: int = 0
    max_prompt_tokens: int = 0
    wall_s: float = 0.0
    n_val_errors: int = 0
    val_messages: list[str] = field(default_factory=list)
    infer_errors: list[str] = field(default_factory=list)


@dataclass
class BusinessRun:
    business_id: str
    mapping: Mapping | None
    stats: InferStats
    score: MappingScore | None = None
    revenue_hits: int = 0
    revenue_total: int = 0
    revenue_rows: list[tuple[str, int | None, int, bool]] = field(default_factory=list)
    etl_error: str | None = None
    mapping_ok: bool = False


def _canonical_db(
    split: str,
    biz_id: str,
    mapping: Mapping,
    canon: CanonicalData | None,
    run_dir: Path,
) -> tuple[Path | None, str | None]:
    """Reuse the frames validate_mapping already built. Fall back to a full ETL."""
    if canon is not None:
        db_path = run_dir / f"{biz_id}.duckdb"
        try:
            write_duckdb(canon, db_path)
        except Exception as exc:  # noqa: BLE001 — predicted mapping may be arbitrarily broken
            return None, f"{type(exc).__name__}: {exc}"
        return db_path, None
    try:
        db_path, _report = load_or_build_canonical(split, biz_id, mapping)
    except Exception as exc:  # noqa: BLE001 — predicted mapping may be arbitrarily broken
        return None, f"{type(exc).__name__}: {exc}"
    return db_path, None


def _infer_one(
    split: str,
    biz_id: str,
    client: Any,
    run_dir: Path,
) -> tuple[Mapping | None, InferStats, CanonicalData | None]:
    t0 = time.perf_counter()
    raw = load_business_cached(split, biz_id)
    profile = profile_business(raw)
    result = infer_mapping(raw, profile, client)
    wall_s = time.perf_counter() - t0

    err_issues = _error_issues(result.issues)
    stats = InferStats(
        n_calls=len(result.steps),
        n_cache_hits=sum(1 for s in result.steps if s.cache_hit),
        max_prompt_tokens=max((s.prompt_tokens for s in result.steps), default=0),
        wall_s=wall_s,
        n_val_errors=len(err_issues),
        val_messages=[iss.message for iss in err_issues],
        infer_errors=list(result.errors),
    )

    mapping = result.mapping
    if mapping is not None:
        to_save = mapping.model_copy(update={"business_id": biz_id})
        dump_mapping_yaml(to_save, run_dir / f"{biz_id}.yaml")
    else:
        (run_dir / f"{biz_id}.error.txt").write_text(
            "\n".join(result.errors) or "infer_mapping returned no mapping",
            encoding="utf-8",
        )
    return mapping, stats, result.canonical


def _query_expected_revenues(
    split: str,
    biz_id: str,
    mapping: Mapping,
    canon: CanonicalData | None,
    run_dir: Path,
) -> tuple[list[tuple[str, int | None, int, bool]], str | None]:
    expected = [(ym, rev) for b, ym, rev in EXPECTED_REVENUES if b == biz_id]
    if not expected:
        return [], None
    db_path, etl_error = _canonical_db(split, biz_id, mapping, canon, run_dir)
    if db_path is None:
        rows = [(ym, None, gold, False) for ym, gold in expected]
        return rows, etl_error

    rows: list[tuple[str, int | None, int, bool]] = []
    for ym, gold in expected:
        actual = query_monthly_revenue(db_path, ym)
        rows.append((ym, actual, gold, actual == gold))
    return rows, None


def _score_vs_gold(pred: Mapping, biz_id: str) -> MappingScore | None:
    gold_path = GOLD_DIR / f"{biz_id}.yaml"
    if not gold_path.is_file():
        return None
    gold = load_mapping_yaml(gold_path)
    return score(pred, gold)


def _format_field_acc(ms: MappingScore | None) -> str:
    if ms is None:
        return "—"
    denom = ms.n_correct + ms.n_wrong + ms.n_missing
    order_s = f"{ms.field_accuracy:.3f} ({ms.n_correct}/{denom})"
    ne = ms.new_entities
    if ne is None:
        return order_s
    ne_denom = ne.n_correct + ne.n_wrong + ne.n_missing
    if ne_denom == 0 and ne.n_extra == 0:
        return f"{order_s} | new —"
    return (
        f"{order_s} | new {ne.field_accuracy:.3f} ({ne.n_correct}/{ne_denom})"
        + (f" extra={ne.n_extra}" if ne.n_extra else "")
    )


def _format_mismatch_block(biz_id: str, run: BusinessRun) -> list[str]:
    lines = [f"### `{biz_id}`", ""]
    ms = run.score
    if ms is None:
        if run.mapping is None:
            lines.append("- infer failed; no mapping to score.")
        else:
            lines.append("- no gold mapping file.")
        if run.stats.infer_errors:
            lines.append(f"- infer errors: {run.stats.infer_errors}")
        if run.stats.val_messages:
            lines.append(f"- validator errors: {run.stats.val_messages}")
        if run.etl_error:
            lines.append(f"- ETL error: {run.etl_error}")
        lines.append("")
        return lines

    gold_m: Mapping | None = None
    gold_path = GOLD_DIR / f"{biz_id}.yaml"
    if gold_path.is_file():
        gold_m = load_mapping_yaml(gold_path)

    any_line = False
    if not ms.tables_ok:
        any_line = True
        for note in ms.table_notes:
            lines.append(f"- tables: {note}")
    if gold_m is not None and run.mapping is not None:
        if not ms.customer_identity_ok:
            any_line = True
            lines.append(
                f"- customer_identity: gold=`{gold_m.customer_identity}` "
                f"pred=`{run.mapping.customer_identity}`"
            )
        if not ms.walk_in_ok:
            any_line = True
            lines.append(
                f"- walk_in_values: gold={gold_m.walk_in_values} "
                f"pred={run.mapping.walk_in_values}"
            )
        if not ms.return_rule_ok:
            any_line = True
            lines.append(
                f"- return_rule: gold=`{gold_m.return_rule}` pred=`{run.mapping.return_rule}`"
            )
        if not ms.data_end_ok:
            any_line = True
            lines.append(
                f"- data_end_date: gold=`{gold_m.data_end_date}` "
                f"pred=`{run.mapping.data_end_date}`"
            )
    for item in ms.mismatches():
        any_line = True
        if item.kind == "column":
            lines.append(
                f"- column `{item.key}`: gold=`{item.gold}` pred=`{item.pred}` ({item.verdict})"
            )
        else:
            lines.append(
                f"- status `{item.key}`: gold=`{item.gold}` pred=`{item.pred}` ({item.verdict})"
            )
    if ms.new_entities is not None:
        for note in ms.new_entities.table_notes:
            any_line = True
            lines.append(f"- new-entity tables: {note}")
        for item in ms.new_entities.mismatches():
            any_line = True
            prefix = "new-entity column" if item.kind == "column" else "purchase_status"
            lines.append(
                f"- {prefix} `{item.key}`: gold=`{item.gold}` pred=`{item.pred}` ({item.verdict})"
            )
    if run.stats.val_messages:
        any_line = True
        for msg in run.stats.val_messages:
            lines.append(f"- validator error: {msg}")
    if run.etl_error:
        any_line = True
        lines.append(f"- ETL error: {run.etl_error}")
    for ym, actual, gold_rev, hit in run.revenue_rows:
        if hit:
            continue
        any_line = True
        actual_s = "—" if actual is None else f"{actual:,}"
        lines.append(f"- revenue `{ym}`: gold=`{gold_rev:,}` pred=`{actual_s}`")
    if not any_line:
        return []
    lines.append("")
    return lines


def _render_gold_summary(split: str, runs: list[BusinessRun], run_dir: Path) -> str:
    lines = [
        f"# Mapping eval — gold score (`{split}`)",
        "",
        f"- Run dir: `{run_dir}`",
        f"- Businesses: {len(runs)}",
        f"- Mapping OK: {sum(1 for r in runs if r.mapping_ok)}/{len(runs)}",
        "",
        (
            "| business | field_acc | tables | status | walk-in | return_rule | data_end | "
            "revenue | val_err | llm | cache | max_ptok | wall_s | mapping_ok |"
        ),
        "|---|---:|:---:|:---:|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for r in runs:
        ms = r.score
        lines.append(
            f"| `{r.business_id}` | {_format_field_acc(ms)} | "
            f"{_yn(ms.tables_ok) if ms else '—'} | "
            f"{_yn(ms.status_ok) if ms else '—'} | "
            f"{_yn(ms.walk_in_ok) if ms else '—'} | "
            f"{_yn(ms.return_rule_ok) if ms else '—'} | "
            f"{_yn(ms.data_end_ok) if ms else '—'} | "
            f"{r.revenue_hits}/{r.revenue_total} | "
            f"{r.stats.n_val_errors} | {r.stats.n_calls} | {r.stats.n_cache_hits} | "
            f"{r.stats.max_prompt_tokens} | {r.stats.wall_s:.1f} | {_yn(r.mapping_ok)} |"
        )
    lines.extend(["", "## Wrong / missing / extra fields", ""])
    for r in runs:
        lines.extend(_format_mismatch_block(r.business_id, r))
    return "\n".join(lines).rstrip() + "\n"


@dataclass
class ProbeQuestionResult:
    question_id: str
    year_month: str
    expected: int
    actual: int | None
    status: Literal["match", "mismatch", "abstain"]
    detail: str | None = None


@dataclass
class ProbeBusinessResult:
    business_id: str
    stats: InferStats
    n_questions: int
    n_match: int
    n_mismatch: int
    n_abstain: int
    rows: list[ProbeQuestionResult]
    abstain_reason: str | None = None


def _select_probe_questions(
    split: str,
) -> dict[str, list[tuple[str, str, int]]]:
    questions = load_questions(split)
    refs = load_references(split)
    by_biz: dict[str, list[tuple[str, str, int]]] = {}
    for q in questions:
        if _PROBE_RE.search(q.question) is None:
            continue
        ref = refs.get(q.id)
        if ref is None or ref.type != "number":
            continue
        ym_m = _PROBE_YM_RE.search(q.question)
        if ym_m is None:
            continue
        month = int(ym_m.group(1))
        year = int(ym_m.group(2))
        year_month = f"{year:04d}-{month:02d}"
        by_biz.setdefault(q.business_id, []).append((q.id, year_month, int(ref.answer)))
    for rows in by_biz.values():
        rows.sort(key=lambda row: (row[1], row[0]))
    return by_biz


def _render_probe_summary(
    split: str,
    results: list[ProbeBusinessResult],
    run_dir: Path,
) -> str:
    n_q = sum(r.n_questions for r in results)
    n_match = sum(r.n_match for r in results)
    n_mis = sum(r.n_mismatch for r in results)
    n_abs = sum(r.n_abstain for r in results)
    lines = [
        f"# Mapping eval — probe revenue (`{split}`)",
        "",
        f"- Run dir: `{run_dir}`",
        f"- Businesses: {len(results)}",
        f"- Questions: {n_q} (match {n_match}, mismatch {n_mis}, abstain {n_abs})",
        "",
        (
            "| business | n | match | mismatch | abstain | val_err | llm | cache | "
            "max_ptok | wall_s | note |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        note = r.abstain_reason or ""
        lines.append(
            f"| `{r.business_id}` | {r.n_questions} | {r.n_match} | {r.n_mismatch} | "
            f"{r.n_abstain} | {r.stats.n_val_errors} | {r.stats.n_calls} | "
            f"{r.stats.n_cache_hits} | {r.stats.max_prompt_tokens} | "
            f"{r.stats.wall_s:.1f} | {note} |"
        )
    lines.extend(["", "## Mismatches / abstains", ""])
    any_detail = False
    for r in results:
        interesting = [row for row in r.rows if row.status != "match"]
        if not interesting and r.abstain_reason is None:
            continue
        any_detail = True
        lines.append(f"### `{r.business_id}`")
        lines.append("")
        if r.abstain_reason and r.n_match == 0 and r.n_mismatch == 0:
            lines.append(f"- abstain: {r.abstain_reason}")
            lines.append("")
            continue
        if r.abstain_reason:
            lines.append(f"- abstain reason: {r.abstain_reason}")
        for row in interesting:
            if row.status == "abstain":
                continue
            actual_s = "—" if row.actual is None else f"{row.actual:,}"
            lines.append(
                f"- `{row.question_id}` `{row.year_month}`: "
                f"gold=`{row.expected:,}` pred=`{actual_s}`"
            )
        lines.append("")
    if not any_detail:
        lines.append("- (none)")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run_gold_mode(
    split: str,
    businesses: list[str],
    client: Any,
    run_dir: Path,
) -> list[BusinessRun]:
    runs: list[BusinessRun] = []
    for biz_id in businesses:
        print(f"[{biz_id}] infer …", flush=True)
        t_biz = time.perf_counter()
        mapping, stats, canon = _infer_one(split, biz_id, client, run_dir)
        ms = _score_vs_gold(mapping, biz_id) if mapping is not None else None
        revenue_rows: list[tuple[str, int | None, int, bool]] = []
        etl_error: str | None = None
        if mapping is not None:
            revenue_rows, etl_error = _query_expected_revenues(
                split, biz_id, mapping, canon, run_dir
            )
        stats.wall_s = time.perf_counter() - t_biz
        hits = sum(1 for _ym, _a, _g, ok in revenue_rows if ok)
        total = len(revenue_rows)
        mapping_ok = (
            mapping is not None
            and stats.n_val_errors == 0
            and total > 0
            and hits == total
            and etl_error is None
        )
        run = BusinessRun(
            business_id=biz_id,
            mapping=mapping,
            stats=stats,
            score=ms,
            revenue_hits=hits,
            revenue_total=total,
            revenue_rows=revenue_rows,
            etl_error=etl_error,
            mapping_ok=mapping_ok,
        )
        runs.append(run)
        acc = _format_field_acc(ms)
        print(
            f"[{biz_id}] field_acc={acc} revenue={hits}/{total} "
            f"val_err={stats.n_val_errors} mapping_ok={mapping_ok} "
            f"{stats.wall_s:.1f}s",
            flush=True,
        )
    return runs


def run_probe_mode(
    split: str,
    businesses: list[str],
    client: Any,
    run_dir: Path,
) -> list[ProbeBusinessResult]:
    probe_q = _select_probe_questions(split)
    results: list[ProbeBusinessResult] = []
    for biz_id in businesses:
        qrows = probe_q.get(biz_id, [])
        if not qrows:
            continue
        print(f"[{biz_id}] infer ({len(qrows)} probe q) …", flush=True)
        t_biz = time.perf_counter()
        mapping, stats, canon = _infer_one(split, biz_id, client, run_dir)
        rows: list[ProbeQuestionResult] = []
        abstain_reason: str | None = None

        if mapping is None:
            abstain_reason = "infer failed"
            for qid, ym, expected in qrows:
                rows.append(
                    ProbeQuestionResult(
                        question_id=qid,
                        year_month=ym,
                        expected=expected,
                        actual=None,
                        status="abstain",
                        detail=abstain_reason,
                    )
                )
        elif stats.n_val_errors > 0:
            abstain_reason = f"{stats.n_val_errors} validator error(s)"
            for qid, ym, expected in qrows:
                rows.append(
                    ProbeQuestionResult(
                        question_id=qid,
                        year_month=ym,
                        expected=expected,
                        actual=None,
                        status="abstain",
                        detail=abstain_reason,
                    )
                )
        else:
            db_path, etl_error = _canonical_db(split, biz_id, mapping, canon, run_dir)
            if etl_error is not None:
                abstain_reason = f"ETL failed: {etl_error}"

            if db_path is None:
                for qid, ym, expected in qrows:
                    rows.append(
                        ProbeQuestionResult(
                            question_id=qid,
                            year_month=ym,
                            expected=expected,
                            actual=None,
                            status="abstain",
                            detail=abstain_reason,
                        )
                    )
            else:
                for qid, ym, expected in qrows:
                    actual = query_monthly_revenue(db_path, ym)
                    rows.append(
                        ProbeQuestionResult(
                            question_id=qid,
                            year_month=ym,
                            expected=expected,
                            actual=actual,
                            status="match" if actual == expected else "mismatch",
                        )
                    )

        stats.wall_s = time.perf_counter() - t_biz
        result = ProbeBusinessResult(
            business_id=biz_id,
            stats=stats,
            n_questions=len(rows),
            n_match=sum(1 for row in rows if row.status == "match"),
            n_mismatch=sum(1 for row in rows if row.status == "mismatch"),
            n_abstain=sum(1 for row in rows if row.status == "abstain"),
            rows=rows,
            abstain_reason=abstain_reason,
        )
        results.append(result)
        print(
            f"[{biz_id}] probe {result.n_match}/{result.n_questions} match "
            f"mismatch={result.n_mismatch} abstain={result.n_abstain} "
            f"{stats.wall_s:.1f}s",
            flush=True,
        )
    return results


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Infer LLM mappings, score vs gold, and check monthly revenue."
    )
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument(
        "--businesses",
        default=None,
        help="Comma-separated business ids (required unless --probe-revenue)",
    )
    parser.add_argument(
        "--probe-revenue",
        action="store_true",
        help=(
            "Score plain monthly-revenue train questions (no gold mapping required). "
            "Businesses with remaining validator errors are reported as abstain."
        ),
    )
    args = parser.parse_args(argv)

    if args.businesses:
        businesses = [b.strip() for b in args.businesses.split(",") if b.strip()]
    elif args.probe_revenue:
        businesses = _list_businesses(args.split)
    else:
        parser.error("Provide --businesses <id,id,...> or --probe-revenue")

    if not businesses:
        parser.error("No businesses to run")

    base_runs_dir = settings.runs_dir
    base_runs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = _unique_run_dir(base_runs_dir, f"{ts}_mapping")
    run_dir.mkdir(parents=True, exist_ok=True)

    client = get_client()
    try:
        if args.probe_revenue:
            results = run_probe_mode(args.split, businesses, client, run_dir)
            summary = _render_probe_summary(args.split, results, run_dir)
        else:
            runs = run_gold_mode(args.split, businesses, client, run_dir)
            summary = _render_gold_summary(args.split, runs, run_dir)
    finally:
        client.close()

    summary_path = run_dir / "summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    print(summary, end="")
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()

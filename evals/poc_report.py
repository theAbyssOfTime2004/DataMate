"""Generate the POC comparison report (P7c).

Reads run_eval / run_mapping outputs and writes aggregate markdown only
(no question text, answers, or references in --out).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datamate.io.questions import Reference, load_references
from datamate.mapping.model import load_mapping_yaml
from datamate.scoring.scorer import Result, score_file
from evals.score_mapping import score as score_mapping
from evals.subsets import tag_topics

GOLD_MAPPINGS_DIR = PROJECT_ROOT / "evals" / "gold_mappings"
_EXCEPTION_REASON_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*):\s")
LoadRefsFn = Callable[[str], dict[str, Reference]]


@dataclass
class QuestionRow:
    id: str
    business_id: str
    question: str
    as_of: str
    topics: set[str]
    submitted: dict[str, Any]
    score: int
    elapsed_s: float
    abstain_reason: str | None
    llm_calls: int
    log: dict[str, Any]


@dataclass
class LoadedRun:
    label: str
    path: Path
    metrics: dict[str, Any]
    result: Result
    rows: dict[str, QuestionRow]
    order: list[str]
    refs: dict[str, Reference] = field(default_factory=dict)

    @property
    def metadata(self) -> dict[str, Any]:
        meta = self.metrics.get("metadata")
        return meta if isinstance(meta, dict) else {}

    @property
    def git_commit(self) -> str | None:
        raw = self.metadata.get("git_commit")
        return raw if isinstance(raw, str) and raw else None

    @property
    def git_dirty(self) -> bool | None:
        raw = self.metadata.get("git_dirty")
        return raw if isinstance(raw, bool) else None

    @property
    def system(self) -> str:
        raw = self.metadata.get("system")
        return raw if isinstance(raw, str) else "?"

    @property
    def split(self) -> str:
        raw = self.metadata.get("split")
        return raw if isinstance(raw, str) else "?"

    @property
    def subset(self) -> str:
        raw = self.metadata.get("subset")
        return raw if isinstance(raw, str) else "?"

    @property
    def workers(self) -> int | str:
        raw = self.metadata.get("workers")
        return raw if isinstance(raw, int) else "?"

    @property
    def model(self) -> str:
        settings = self.metadata.get("settings")
        if isinstance(settings, dict):
            model = settings.get("llm_model")
            if isinstance(model, str) and model:
                return model
        return "?"


@dataclass
class MappingBizScore:
    business_id: str
    field_accuracy: float | None
    n_correct: int
    n_total: int
    tables_ok: bool | None
    n_extra: int
    no_gold: bool = False
    error: str | None = None


@dataclass
class MappingRun:
    label: str
    path: Path
    by_business: dict[str, MappingBizScore] = field(default_factory=dict)


def _parse_label_path(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"expected LABEL=PATH, got {raw!r}")
    label, path_text = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError(f"empty label in {raw!r}")
    return label, Path(path_text)


def _parse_pair(raw: str) -> tuple[str, str]:
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise argparse.ArgumentTypeError(f"expected X,Y got {raw!r}")
    return parts[0], parts[1]


def _parse_combo(raw: str) -> tuple[str, str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"expected NAME=FIRST,SECOND got {raw!r}")
    name, rest = raw.split("=", 1)
    name = name.strip()
    first, second = _parse_pair(rest)
    if not name:
        raise argparse.ArgumentTypeError(f"empty combo name in {raw!r}")
    return name, first, second


def collapse_abstain_reason(reason: str) -> str:
    match = _EXCEPTION_REASON_RE.match(reason)
    if match:
        return f"exception:{match.group(1).rsplit('.', 1)[-1]}"
    return reason


def _llm_calls_from_trace(trace: Any, system: str) -> int:
    if not isinstance(trace, dict):
        return 0
    if system == "direct":
        raw = trace.get("llm_calls")
        return int(raw) if isinstance(raw, int) else 0
    attempts = trace.get("parse_attempts")
    if isinstance(attempts, list):
        return len(attempts)
    return 0


def _short_commit(commit: str | None, dirty: bool | None = None) -> str:
    if not commit:
        return "-"
    short = commit[:7]
    if dirty is True:
        return f"{short}*"
    return short


def _format_run_commit(run: LoadedRun) -> str:
    return _short_commit(run.git_commit, run.git_dirty)


def _is_timing_run(run: LoadedRun) -> bool:
    """A latency estimate is only valid for workers=1 with LLM cache off."""
    if run.metadata.get("workers") != 1:
        return False
    settings = run.metadata.get("settings")
    if not isinstance(settings, dict):
        return False
    return settings.get("llm_cache_enabled") is False


def _timing_skip_note(run: LoadedRun) -> str:
    workers = run.metadata.get("workers", "?")
    settings = run.metadata.get("settings")
    cache_on = True
    if isinstance(settings, dict) and "llm_cache_enabled" in settings:
        cache_on = bool(settings["llm_cache_enabled"])
    cache_label = "on" if cache_on else "off"
    return f"not a timing run (workers={workers}, cache={cache_label})"


def _fmt_pa(value: float) -> str:
    return f"{value:.3f}"


def _fmt_pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _assert_metrics_match(result: Result, metrics: dict[str, Any], run_dir: Path) -> None:
    expected_correct = int(metrics["correct"])
    expected_wrong = int(metrics["wrong"])
    expected_pa = float(metrics["PA"])
    if result.correct != expected_correct or result.wrong != expected_wrong:
        raise AssertionError(
            f"{run_dir}: re-score correct/wrong={result.correct}/{result.wrong} "
            f"!= metrics {expected_correct}/{expected_wrong}"
        )
    if abs(result.PA - expected_pa) > 1e-9:
        raise AssertionError(
            f"{run_dir}: re-score PA={result.PA!r} != metrics PA={expected_pa!r}"
        )


def load_run(
    label: str,
    run_dir: Path,
    *,
    load_refs: LoadRefsFn | None = None,
) -> LoadedRun:
    """Load a run_eval directory, re-score, and assert metrics agree."""
    run_dir = run_dir.resolve()
    metrics_path = run_dir / "metrics.json"
    submission_path = run_dir / "submission.jsonl"
    logs_dir = run_dir / "logs"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"missing metrics.json in {run_dir}")
    if not submission_path.is_file():
        raise FileNotFoundError(f"missing submission.jsonl in {run_dir}")
    if not logs_dir.is_dir():
        raise FileNotFoundError(f"missing logs/ in {run_dir}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    meta = metrics.get("metadata") if isinstance(metrics.get("metadata"), dict) else {}
    split = meta.get("split") if isinstance(meta.get("split"), str) else "train"
    system = meta.get("system") if isinstance(meta.get("system"), str) else "?"

    refs_fn = load_refs or load_references
    refs = refs_fn(split)

    submissions: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for line in submission_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        qid = obj["id"]
        if qid not in submissions:
            order.append(qid)
            submissions[qid] = obj

    result = score_file(submission_path, refs, order)
    _assert_metrics_match(result, metrics, run_dir)

    rows: dict[str, QuestionRow] = {}
    for qid in order:
        log_path = logs_dir / f"{qid}.json"
        if not log_path.is_file():
            raise FileNotFoundError(f"missing log for {qid} in {run_dir}")
        log = json.loads(log_path.read_text(encoding="utf-8"))
        question = str(log.get("question") or "")
        business_id = str(log.get("business_id") or "")
        as_of = str(log.get("as_of_date") or "")
        topics = tag_topics(question) if question else set()
        if not topics:
            topics = {"other"}
        trace = log.get("trace")
        abstain_reason = None
        submitted = submissions[qid]
        if submitted.get("type") == "abstain":
            reason = "unspecified"
            if isinstance(trace, dict):
                raw = trace.get("abstain_reason")
                if isinstance(raw, str) and raw:
                    reason = raw
            abstain_reason = reason
        elapsed = float(log.get("elapsed_s") or 0.0)
        rows[qid] = QuestionRow(
            id=qid,
            business_id=business_id,
            question=question,
            as_of=as_of,
            topics=topics,
            submitted=submitted,
            score=result.scores[qid],
            elapsed_s=elapsed,
            abstain_reason=abstain_reason,
            llm_calls=_llm_calls_from_trace(trace, system),
            log=log,
        )

    return LoadedRun(
        label=label,
        path=run_dir,
        metrics=metrics,
        result=result,
        rows=rows,
        order=order,
        refs={qid: refs[qid] for qid in order if qid in refs},
    )


def load_mapping_run(label: str, mapping_dir: Path) -> MappingRun:
    mapping_dir = mapping_dir.resolve()
    out = MappingRun(label=label, path=mapping_dir)
    pred_files = sorted(mapping_dir.glob("*.yaml"))
    for pred_path in pred_files:
        biz_id = pred_path.stem
        gold_path = GOLD_MAPPINGS_DIR / f"{biz_id}.yaml"
        if not gold_path.is_file():
            out.by_business[biz_id] = MappingBizScore(
                business_id=biz_id,
                field_accuracy=None,
                n_correct=0,
                n_total=0,
                tables_ok=None,
                n_extra=0,
                no_gold=True,
            )
            continue
        try:
            pred = load_mapping_yaml(pred_path)
            gold = load_mapping_yaml(gold_path)
            scored = score_mapping(pred, gold)
            n_total = scored.n_correct + scored.n_wrong + scored.n_missing
            out.by_business[biz_id] = MappingBizScore(
                business_id=biz_id,
                field_accuracy=scored.field_accuracy,
                n_correct=scored.n_correct,
                n_total=n_total,
                tables_ok=scored.tables_ok,
                n_extra=scored.n_extra,
            )
        except Exception as exc:  # noqa: BLE001 — report per business
            out.by_business[biz_id] = MappingBizScore(
                business_id=biz_id,
                field_accuracy=None,
                n_correct=0,
                n_total=0,
                tables_ok=None,
                n_extra=0,
                error=f"{type(exc).__name__}: {exc}",
            )
    return out


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = min(len(sorted_v) - 1, max(0, math.ceil(0.95 * len(sorted_v)) - 1))
    return sorted_v[idx]


def _breakdown(scores: list[int]) -> dict[str, float | int]:
    n = len(scores)
    correct = sum(1 for s in scores if s == 1)
    wrong = sum(1 for s in scores if s == -1)
    answered = correct + wrong
    total = sum(scores)
    return {
        "N": n,
        "PA": total / n if n else 0.0,
        "correct": correct,
        "wrong": wrong,
        "answered": answered,
        "coverage": answered / n if n else 0.0,
        "precision": correct / answered if answered else 0.0,
    }


def _cell_pa(scores: list[int]) -> str:
    b = _breakdown(scores)
    return f"{_fmt_pa(float(b['PA']))} ({b['correct']}/{b['wrong']}/{b['N']})"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def pair_matrix(
    run_x: LoadedRun, run_y: LoadedRun
) -> tuple[dict[tuple[int, int], int], int, int]:
    ids_x = set(run_x.order)
    ids_y = set(run_y.order)
    if ids_x != ids_y:
        only_x = sorted(ids_x - ids_y)
        only_y = sorted(ids_y - ids_x)
        raise ValueError(
            f"pair {run_x.label},{run_y.label}: id sets differ; "
            f"only in {run_x.label}={only_x[:5]}… only in {run_y.label}={only_y[:5]}…"
        )
    counts: dict[tuple[int, int], int] = defaultdict(int)
    y_right_x_abs = 0
    y_wrong_x_abs = 0
    for qid in sorted(ids_x):
        sx = run_x.rows[qid].score
        sy = run_y.rows[qid].score
        counts[(sx, sy)] += 1
        if sx == 0:
            if sy == 1:
                y_right_x_abs += 1
            elif sy == -1:
                y_wrong_x_abs += 1
    return dict(counts), y_right_x_abs, y_wrong_x_abs


def combo_scores(first: LoadedRun, second: LoadedRun) -> dict[str, float | int]:
    ids_a = set(first.order)
    ids_b = set(second.order)
    if ids_a != ids_b:
        raise ValueError(
            f"combo {first.label},{second.label}: id sets differ"
        )
    scores: list[int] = []
    for qid in first.order:
        row_a = first.rows[qid]
        if row_a.submitted.get("type") == "abstain":
            scores.append(second.rows[qid].score)
        else:
            scores.append(row_a.score)
    return _breakdown(scores)


def _commit_warnings(
    runs: dict[str, LoadedRun],
    pairs: list[tuple[str, str]],
    combos: list[tuple[str, str, str]],
) -> list[str]:
    warnings: list[str] = []
    seen: set[tuple[str, str]] = set()

    def check(a: str, b: str, context: str) -> None:
        key = tuple(sorted((a, b)))
        if key in seen:
            return
        seen.add(key)
        ra, rb = runs[a], runs[b]
        ca, cb = ra.git_commit, rb.git_commit
        if not ca or not cb:
            return
        state_a = (ca, bool(ra.git_dirty))
        state_b = (cb, bool(rb.git_dirty))
        if state_a == state_b:
            return
        left = _format_run_commit(ra)
        right = _format_run_commit(rb)
        if ra.git_dirty or rb.git_dirty:
            warnings.append(
                f"Warning: {context} mixes git state (uncommitted changes): "
                f"{a}={left} vs {b}={right}"
            )
        else:
            warnings.append(
                f"Warning: {context} mixes git commits {a}={left} vs {b}={right}"
            )

    for x, y in pairs:
        check(x, y, f"pair {x},{y}")
    for name, first, second in combos:
        check(first, second, f"combo {name}")
    return warnings


def render_report(
    runs: dict[str, LoadedRun],
    pairs: list[tuple[str, str]],
    combos: list[tuple[str, str, str]],
    mappings: dict[str, MappingRun],
    timings: dict[str, LoadedRun],
) -> str:
    labels = sorted(runs)
    lines: list[str] = [
        "# POC report",
        "",
        "Aggregates and question ids only. No question text, answers, or references.",
        "",
        "## 1. Runs",
        "",
    ]

    run_rows: list[list[str]] = []
    for label in labels:
        run = runs[label]
        timing = run.metrics.get("timing") if isinstance(run.metrics.get("timing"), dict) else {}
        elapsed = [run.rows[qid].elapsed_s for qid in run.order]
        mean_s = float(timing.get("mean_s", _mean(elapsed)))
        p95_s = float(timing.get("p95_s", _p95(elapsed)))
        wall_s = float(timing.get("total_wall_s", sum(elapsed)))
        llm_mean = _mean([float(run.rows[qid].llm_calls) for qid in run.order])
        run_rows.append(
            [
                label,
                run.system,
                run.model,
                f"{run.split}/{run.subset}",
                str(run.result.N),
                _fmt_pa(run.result.PA),
                _fmt_pct(run.result.coverage),
                _fmt_pct(run.result.precision),
                str(run.result.correct),
                str(run.result.wrong),
                str(run.result.N - run.result.answered),
                f"{wall_s:.1f}",
                f"{mean_s:.3f}",
                f"{p95_s:.3f}",
                f"{llm_mean:.2f}",
                str(run.workers),
                _format_run_commit(run),
            ]
        )
    lines.append(
        _md_table(
            [
                "label",
                "system",
                "model",
                "split/subset",
                "N",
                "PA",
                "coverage",
                "precision",
                "correct",
                "wrong",
                "abstain",
                "wall s",
                "mean s/q",
                "p95 s",
                "LLM/q",
                "workers",
                "commit",
            ],
            run_rows,
        )
    )
    for warning in _commit_warnings(runs, pairs, combos):
        lines.extend(["", f"> {warning}"])

    # 2. PA by business (per split)
    lines.extend(["", "## 2. PA by business", ""])
    splits = sorted({runs[label].split for label in labels})
    for split in splits:
        split_labels = [label for label in labels if runs[label].split == split]
        if not split_labels:
            continue
        lines.append(f"### Split `{split}`")
        lines.append("")
        businesses = sorted(
            {
                runs[label].rows[qid].business_id
                for label in split_labels
                for qid in runs[label].order
            }
        )
        header = ["business", *split_labels]
        table_rows: list[list[str]] = []
        for biz in businesses:
            row = [biz]
            for label in split_labels:
                scores = [
                    runs[label].rows[qid].score
                    for qid in runs[label].order
                    if runs[label].rows[qid].business_id == biz
                ]
                row.append(_cell_pa(scores) if scores else "-")
            table_rows.append(row)
        lines.append(_md_table(header, table_rows))
        lines.append("")

    # 3. PA by topic
    lines.extend(["## 3. PA by topic", ""])
    for split in splits:
        split_labels = [label for label in labels if runs[label].split == split]
        if not split_labels:
            continue
        lines.append(f"### Split `{split}`")
        lines.append("")
        topics = sorted(
            {
                topic
                for label in split_labels
                for qid in runs[label].order
                for topic in runs[label].rows[qid].topics
            }
        )
        header = ["topic", *split_labels]
        table_rows = []
        for topic in topics:
            row = [topic]
            for label in split_labels:
                scores = [
                    runs[label].rows[qid].score
                    for qid in runs[label].order
                    if topic in runs[label].rows[qid].topics
                ]
                row.append(_cell_pa(scores) if scores else "-")
            table_rows.append(row)
        lines.append(_md_table(header, table_rows))
        lines.append("")

    # 4. Abstain reasons
    lines.extend(["## 4. Abstain reasons", ""])
    reason_set: set[str] = set()
    reason_counts: dict[str, dict[str, int]] = {label: defaultdict(int) for label in labels}
    for label in labels:
        for qid in runs[label].order:
            reason = runs[label].rows[qid].abstain_reason
            if reason is None:
                continue
            collapsed = collapse_abstain_reason(reason)
            reason_set.add(collapsed)
            reason_counts[label][collapsed] += 1
    reason_rows = []
    for reason in sorted(reason_set):
        reason_rows.append(
            [reason, *[str(reason_counts[label].get(reason, 0)) for label in labels]]
        )
    if reason_rows:
        lines.append(_md_table(["reason", *labels], reason_rows))
    else:
        lines.append("_No abstains._")
    lines.append("")

    # 5. Pairs
    lines.extend(["## 5. Pairs", ""])
    if not pairs:
        lines.append("_No pairs._")
        lines.append("")
    for x_label, y_label in pairs:
        run_x, run_y = runs[x_label], runs[y_label]
        counts, y_right, y_wrong = pair_matrix(run_x, run_y)
        lines.append(f"### Pair `{x_label}` vs `{y_label}`")
        lines.append("")
        header = [f"{x_label}\\{y_label}", "-1", "0", "+1"]
        matrix_rows = []
        for sx in (-1, 0, 1):
            matrix_rows.append(
                [
                    str(sx),
                    str(counts.get((sx, -1), 0)),
                    str(counts.get((sx, 0), 0)),
                    str(counts.get((sx, 1), 0)),
                ]
            )
        lines.append(_md_table(header, matrix_rows))
        lines.append("")
        lines.append(f"- `{y_label}` right where `{x_label}` abstained: **{y_right}**")
        lines.append(f"- `{y_label}` wrong where `{x_label}` abstained: **{y_wrong}**")
        lines.append("")

    # 6. Combos
    lines.extend(["## 6. Combos", ""])
    if not combos:
        lines.append("_No combos._")
        lines.append("")
    for name, first_label, second_label in combos:
        first, second = runs[first_label], runs[second_label]
        combo = combo_scores(first, second)
        base = _breakdown([first.rows[qid].score for qid in first.order])
        delta_pa = float(combo["PA"]) - float(base["PA"])
        lines.append(f"### Combo `{name}` = `{first_label}` then `{second_label}`")
        lines.append("")
        lines.append(
            _md_table(
                ["system", "PA", "coverage", "precision", "correct", "wrong", "ΔPA vs first"],
                [
                    [
                        name,
                        _fmt_pa(float(combo["PA"])),
                        _fmt_pct(float(combo["coverage"])),
                        _fmt_pct(float(combo["precision"])),
                        str(combo["correct"]),
                        str(combo["wrong"]),
                        f"{delta_pa:+.3f}",
                    ],
                    [
                        first_label,
                        _fmt_pa(float(base["PA"])),
                        _fmt_pct(float(base["coverage"])),
                        _fmt_pct(float(base["precision"])),
                        str(base["correct"]),
                        str(base["wrong"]),
                        "0.000",
                    ],
                ],
            )
        )
        lines.append("")

    # 7. Mapping accuracy
    lines.extend(["## 7. Mapping accuracy", ""])
    if not mappings:
        lines.append("_No mapping runs._")
        lines.append("")
    else:
        map_labels = sorted(mappings)
        businesses = sorted(
            {biz for mr in mappings.values() for biz in mr.by_business}
        )
        header = ["business", *map_labels]
        table_rows = []
        totals: dict[str, list[MappingBizScore]] = {lab: [] for lab in map_labels}
        for biz in businesses:
            row = [biz]
            for lab in map_labels:
                item = mappings[lab].by_business.get(biz)
                if item is None:
                    row.append("-")
                    continue
                totals[lab].append(item)
                if item.no_gold:
                    row.append("no gold")
                elif item.error:
                    row.append(f"error: {item.error}")
                else:
                    tables = "yes" if item.tables_ok else "no"
                    acc = item.field_accuracy if item.field_accuracy is not None else 0.0
                    row.append(
                        f"{acc:.3f} ({item.n_correct}/{item.n_total}), "
                        f"tables {tables}, extra {item.n_extra}"
                    )
            table_rows.append(row)
        # totals
        total_row = ["**totals**"]
        for lab in map_labels:
            scored = [m for m in totals[lab] if not m.no_gold and m.error is None]
            if not scored:
                total_row.append("-")
                continue
            n_correct = sum(m.n_correct for m in scored)
            n_total = sum(m.n_total for m in scored)
            acc = 1.0 if n_total == 0 else n_correct / n_total
            tables_ok_n = sum(1 for m in scored if m.tables_ok)
            perfect = sum(
                1
                for m in scored
                if m.field_accuracy == 1.0 and m.tables_ok
            )
            total_row.append(
                f"{acc:.3f} ({n_correct}/{n_total}), "
                f"tables_ok {tables_ok_n}/{len(scored)}, "
                f"perfect {perfect}/{len(scored)}"
            )
        table_rows.append(total_row)
        lines.append(_md_table(header, table_rows))
        lines.append("")

    # 8. Latency
    lines.extend(["## 8. Latency", ""])
    lines.append(
        "GPU note: timing runs used a Modal L40S, not the final-machine 5090. "
        "Contest limit is 6 h for ~500 questions."
    )
    lines.append("")
    if not timings:
        lines.append("_No timing runs._")
        lines.append("")
    else:
        timing_rows = []
        for label in sorted(timings):
            run = timings[label]
            if not _is_timing_run(run):
                timing_rows.append(
                    [label, _timing_skip_note(run), "-", "-", "-"]
                )
                continue
            elapsed = [run.rows[qid].elapsed_s for qid in run.order]
            mean_s = _mean(elapsed)
            p95_s = _p95(elapsed)
            hours_500 = (mean_s * 500.0) / 3600.0
            timing_rows.append(
                [
                    label,
                    f"{mean_s:.3f}",
                    f"{p95_s:.3f}",
                    f"{hours_500:.2f} h",
                    "6.00 h",
                ]
            )
        lines.append(
            _md_table(
                ["label", "mean s/q", "p95 s/q", "500 q estimate", "limit"],
                timing_rows,
            )
        )
        lines.append("")

    # 9. Wrong answers
    lines.extend(["## 9. Wrong answers", ""])
    for label in labels:
        run = runs[label]
        wrong_ids = [qid for qid in run.order if run.rows[qid].score == -1]
        lines.append(f"### `{label}` ({len(wrong_ids)})")
        lines.append("")
        if not wrong_ids:
            lines.append("_None._")
            lines.append("")
            continue
        by_biz: dict[str, list[str]] = defaultdict(list)
        for qid in wrong_ids:
            row = run.rows[qid]
            topics = ",".join(sorted(row.topics))
            by_biz[row.business_id].append(f"{qid} [{topics}]")
        for biz in sorted(by_biz):
            lines.append(f"- **{biz}**: {', '.join(by_biz[biz])}")
        lines.append("")

    # 10. Human sections
    lines.extend(
        [
            "## Top 10 kiểu lỗi",
            "",
            "_Để người điền._",
            "",
            "## Quyết định 31/10",
            "",
        ]
    )
    decision_rows = _decision_rows(runs, mappings, timings)
    lines.append(
        _md_table(
            ["câu hỏi", "ngưỡng", "kết quả", "đạt?"],
            decision_rows,
        )
    )
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _decision_rows(
    runs: dict[str, LoadedRun],
    mappings: dict[str, MappingRun],
    timings: dict[str, LoadedRun],
) -> list[list[str]]:
    rows: list[list[str]] = []

    a_prec = "-"
    if "A" in runs:
        a_prec = _fmt_pct(runs["A"].result.precision)
    rows.append(
        [
            "A: trên số câu đã trả lời, đúng bao nhiêu?",
            "≥ 90%",
            a_prec if a_prec != "-" else "-",
            "",
        ]
    )

    map_result = "-"
    map_run = mappings.get("map14") or next(iter(mappings.values()), None)
    if map_run is not None:
        scored = [
            m
            for m in map_run.by_business.values()
            if not m.no_gold and m.error is None
        ]
        perfect = sum(
            1 for m in scored if m.field_accuracy == 1.0 and m.tables_ok
        )
        map_result = f"{perfect}/{len(scored)} businesses field_acc=1.0 and tables_ok"
    rows.append(
        [
            "B hoặc B-14B: mapping đúng cho bao nhiêu doanh nghiệp?",
            "≥ 4/5 với 14B",
            map_result,
            "",
        ]
    )

    pa_cmp = "-"
    if "B14" in runs and "C4" in runs:
        pa_cmp = (
            f"B14 PA={_fmt_pa(runs['B14'].result.PA)}; "
            f"C4 PA={_fmt_pa(runs['C4'].result.PA)}"
        )
    rows.append(
        [
            "PA của B-14B so với C",
            "B-14B > C rõ ràng",
            pa_cmp,
            "",
        ]
    )

    lat = "-"
    valid_timings = {
        label: run for label, run in timings.items() if _is_timing_run(run)
    }
    if valid_timings:
        label = min(valid_timings)
        run = valid_timings[label]
        mean_s = _mean([run.rows[qid].elapsed_s for qid in run.order])
        hours_500 = (mean_s * 500.0) / 3600.0
        lat = f"{label}: 500 q ≈ {hours_500:.2f} h (limit 6 h)"
    elif timings:
        label = min(timings)
        lat = f"{label}: {_timing_skip_note(timings[label])}"
    rows.append(
        [
            "Thời gian",
            "Đủ để ước tính 500 câu trong 6 giờ trên 2 × 5090",
            lat,
            "",
        ]
    )
    return rows


def render_detail(
    runs: dict[str, LoadedRun],
) -> str:
    lines: list[str] = [
        "# POC report detail",
        "",
        "May contain question text, references, code, and errors.",
        "",
    ]
    for label in sorted(runs):
        run = runs[label]
        lines.append(f"## `{label}`")
        lines.append("")
        interesting = [
            qid
            for qid in run.order
            if run.rows[qid].score in (-1, 0)
        ]
        if not interesting:
            lines.append("_No wrong or abstained questions._")
            lines.append("")
            continue
        for qid in interesting:
            row = run.rows[qid]
            ref = row.log.get("final_answer")
            lines.append(f"### {qid}")
            lines.append("")
            lines.append(f"- business: `{row.business_id}`")
            lines.append(f"- topic: {', '.join(sorted(row.topics))}")
            lines.append(f"- question: {row.question}")
            lines.append(f"- as_of: {row.as_of}")
            lines.append(f"- score: {row.score}")
            ref = run.refs.get(qid)
            if ref is not None:
                lines.append(
                    "- reference: "
                    f"`{json.dumps({'type': ref.type, 'answer': ref.answer}, ensure_ascii=False)}`"
                )
            lines.append(f"- submitted: `{json.dumps(row.submitted, ensure_ascii=False)}`")
            if row.abstain_reason:
                lines.append(f"- abstain_reason: `{row.abstain_reason}`")
            trace = row.log.get("trace")
            if isinstance(trace, dict):
                if run.system != "direct":
                    if "spec" in trace:
                        lines.append(
                            f"- spec: `{json.dumps(trace.get('spec'), ensure_ascii=False)}`"
                        )
                    attempts = trace.get("parse_attempts")
                    if isinstance(attempts, list) and attempts:
                        errors = [
                            a.get("error")
                            for a in attempts
                            if isinstance(a, dict) and a.get("error")
                        ]
                        if errors:
                            lines.append(f"- parse_errors: {errors!r}")
                else:
                    attempts = trace.get("attempts")
                    if isinstance(attempts, list) and attempts:
                        last = attempts[-1]
                        if isinstance(last, dict):
                            lines.append(f"- last_code: ```\n{last.get('code')}\n```")
                            lines.append(f"- last_error: `{last.get('error')}`")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_report(
    run_specs: list[tuple[str, Path]],
    pairs: list[tuple[str, str]],
    combos: list[tuple[str, str, str]],
    mapping_specs: list[tuple[str, Path]],
    timing_specs: list[tuple[str, Path]],
    *,
    load_refs: LoadRefsFn | None = None,
) -> tuple[str, str, dict[str, LoadedRun]]:
    runs: dict[str, LoadedRun] = {}
    for label, path in run_specs:
        if label in runs:
            raise ValueError(f"duplicate run label {label!r}")
        runs[label] = load_run(label, path, load_refs=load_refs)

    for x, y in pairs:
        if x not in runs or y not in runs:
            raise ValueError(f"pair {x},{y}: unknown run label")
    for name, first, second in combos:
        if first not in runs or second not in runs:
            raise ValueError(f"combo {name}: unknown run label")

    mappings = {
        label: load_mapping_run(label, path) for label, path in mapping_specs
    }
    timings: dict[str, LoadedRun] = {}
    for label, path in timing_specs:
        if label in runs:
            timings[label] = runs[label]
        else:
            timings[label] = load_run(label, path, load_refs=load_refs)

    out_md = render_report(runs, pairs, combos, mappings, timings)
    detail_md = render_detail(runs)
    return out_md, detail_md, runs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate POC comparison report (P7c).")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        type=_parse_label_path,
        help="LABEL=RUN_DIR (repeatable)",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        type=_parse_pair,
        help="X,Y compare two runs (repeatable)",
    )
    parser.add_argument(
        "--combo",
        action="append",
        default=[],
        type=_parse_combo,
        help="NAME=FIRST,SECOND fallback system (repeatable)",
    )
    parser.add_argument(
        "--mapping",
        action="append",
        default=[],
        type=_parse_label_path,
        help="LABEL=MAPPING_DIR (repeatable)",
    )
    parser.add_argument(
        "--timing",
        action="append",
        default=[],
        type=_parse_label_path,
        help="LABEL=RUN_DIR uncached latency run (repeatable)",
    )
    parser.add_argument("--out", type=Path, required=True, help="Aggregate markdown path")
    parser.add_argument(
        "--detail",
        type=Path,
        required=True,
        help="Detail markdown path (may contain text)",
    )
    args = parser.parse_args(argv)
    if not args.run:
        parser.error("at least one --run is required")

    out_md, detail_md, _runs = build_report(
        args.run,
        args.pair,
        args.combo,
        args.mapping,
        args.timing,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.detail.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(out_md, encoding="utf-8")
    args.detail.write_text(detail_md, encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"Wrote {args.detail}")


if __name__ == "__main__":
    main()

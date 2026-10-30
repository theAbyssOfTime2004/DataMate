from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from datamate.mapping.fields import LATER_FIELDS, SIMPLE_ENTITIES
from datamate.mapping.model import Entity, Mapping, TableMapping
from datamate.parse import normalize_label

Verdict = Literal["correct", "wrong", "missing", "extra"]

_ORDER_PATH_ENTITIES: frozenset[Entity] = frozenset(
    {"orders", "order_lines", "products", "unit_conversions", "customers"}
)
_SIMPLE_ENTITY_SET: frozenset[Entity] = frozenset(SIMPLE_ENTITIES)
_LATER_KEYS: frozenset[str] = frozenset(
    f"{ent}.{fname}" for ent, names in LATER_FIELDS.items() for fname in names
)


def _norm_key(raw: str) -> str:
    n = normalize_label(raw)
    return n if n is not None else ""


def _file_set(tm: TableMapping) -> frozenset[str]:
    return frozenset(s.file for s in tm.sources)


def _walk_in_set(values: list[str]) -> set[str]:
    return {key for v in values if (key := _norm_key(v))}


def _is_order_path_column(entity: Entity, field_name: str) -> bool:
    if entity not in _ORDER_PATH_ENTITIES:
        return False
    return f"{entity}.{field_name}" not in _LATER_KEYS


def _is_new_entity_column(entity: Entity, field_name: str) -> bool:
    if entity in _SIMPLE_ENTITY_SET:
        return True
    return f"{entity}.{field_name}" in _LATER_KEYS


@dataclass
class FieldVerdict:
    kind: Literal["column", "status"]
    key: str
    gold: str | None
    pred: str | None
    verdict: Verdict


@dataclass
class PathScore:
    n_correct: int
    n_wrong: int
    n_missing: int
    field_accuracy: float
    columns: list[FieldVerdict] = field(default_factory=list)
    statuses: list[FieldVerdict] = field(default_factory=list)
    extra_columns: list[FieldVerdict] = field(default_factory=list)
    extra_statuses: list[FieldVerdict] = field(default_factory=list)
    table_notes: list[str] = field(default_factory=list)

    @property
    def n_extra(self) -> int:
        return len(self.extra_columns) + len(self.extra_statuses)

    def mismatches(self) -> list[FieldVerdict]:
        out: list[FieldVerdict] = []
        for item in (*self.columns, *self.statuses):
            if item.verdict in ("wrong", "missing"):
                out.append(item)
        out.extend(self.extra_columns)
        out.extend(self.extra_statuses)
        return out


@dataclass
class MappingScore:
    """Top-level counts are the order_path group (POC-comparable, e.g. 141/143)."""

    n_correct: int
    n_wrong: int
    n_missing: int
    field_accuracy: float
    tables_ok: bool
    status_ok: bool
    walk_in_ok: bool
    customer_identity_ok: bool
    return_rule_ok: bool
    data_end_ok: bool
    columns: list[FieldVerdict] = field(default_factory=list)
    statuses: list[FieldVerdict] = field(default_factory=list)
    extra_columns: list[FieldVerdict] = field(default_factory=list)
    extra_statuses: list[FieldVerdict] = field(default_factory=list)
    table_notes: list[str] = field(default_factory=list)
    order_path: PathScore | None = None
    new_entities: PathScore | None = None

    @property
    def n_extra(self) -> int:
        return len(self.extra_columns)

    def mismatches(self) -> list[FieldVerdict]:
        out: list[FieldVerdict] = []
        for item in (*self.columns, *self.statuses):
            if item.verdict in ("wrong", "missing"):
                out.append(item)
        out.extend(self.extra_columns)
        out.extend(self.extra_statuses)
        return out


def _score_tables(
    pred: Mapping,
    gold: Mapping,
    *,
    entities: frozenset[Entity] | None = None,
) -> tuple[bool, list[str]]:
    notes: list[str] = []
    ok = True
    for entity, gtm in gold.tables.items():
        if entities is not None and entity not in entities:
            continue
        if gtm.source == "readme":
            continue
        ptm = pred.tables.get(entity)
        if ptm is None:
            ok = False
            notes.append(f"{entity}: missing in predicted mapping")
            continue
        if gtm.derived_from != ptm.derived_from:
            ok = False
            notes.append(
                f"{entity}: derived_from gold={gtm.derived_from!r} pred={ptm.derived_from!r}"
            )
        if gtm.source != ptm.source:
            ok = False
            notes.append(f"{entity}: source gold={gtm.source!r} pred={ptm.source!r}")
        if gtm.sources:
            gold_files = _file_set(gtm)
            pred_files = _file_set(ptm)
            if gold_files != pred_files:
                ok = False
                notes.append(
                    f"{entity}: files gold={sorted(gold_files)} pred={sorted(pred_files)}"
                )
    return ok, notes


def _score_columns_filtered(
    pred: Mapping,
    gold: Mapping,
    *,
    include,
) -> tuple[list[FieldVerdict], list[FieldVerdict]]:
    scored: list[FieldVerdict] = []
    extras: list[FieldVerdict] = []
    for entity, gtm in gold.tables.items():
        if gtm.source == "readme":
            continue
        pred_cols = pred.tables[entity].columns if entity in pred.tables else {}
        for field_name, gold_col in gtm.columns.items():
            if not include(entity, field_name):
                continue
            pred_col = pred_cols.get(field_name)
            if pred_col is None:
                verdict: Verdict = "missing"
            elif pred_col == gold_col:
                verdict = "correct"
            else:
                verdict = "wrong"
            scored.append(
                FieldVerdict(
                    kind="column",
                    key=f"{entity}.{field_name}",
                    gold=gold_col,
                    pred=pred_col,
                    verdict=verdict,
                )
            )

    for entity, ptm in pred.tables.items():
        gold_cols = gold.tables[entity].columns if entity in gold.tables else {}
        for field_name, pred_col in ptm.columns.items():
            if not include(entity, field_name):
                continue
            if field_name in gold_cols:
                continue
            extras.append(
                FieldVerdict(
                    kind="column",
                    key=f"{entity}.{field_name}",
                    gold=None,
                    pred=pred_col,
                    verdict="extra",
                )
            )
    return scored, extras


def _status_index(status_map: dict[str, str]) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    for raw, canon in status_map.items():
        nk = _norm_key(raw)
        if nk not in out:
            out[nk] = (raw, canon)
    return out


def _score_status_map(
    pred_map: dict[str, str], gold_map: dict[str, str]
) -> tuple[list[FieldVerdict], list[FieldVerdict]]:
    gold_idx = _status_index(gold_map)
    pred_idx = _status_index(pred_map)
    scored: list[FieldVerdict] = []
    for nk, (gold_raw, gold_st) in gold_idx.items():
        hit = pred_idx.get(nk)
        if hit is None:
            scored.append(
                FieldVerdict(
                    kind="status",
                    key=gold_raw,
                    gold=gold_st,
                    pred=None,
                    verdict="missing",
                )
            )
            continue
        pred_st = hit[1]
        verdict: Verdict = "correct" if pred_st == gold_st else "wrong"
        scored.append(
            FieldVerdict(
                kind="status",
                key=gold_raw,
                gold=gold_st,
                pred=pred_st,
                verdict=verdict,
            )
        )

    extras: list[FieldVerdict] = []
    for nk, (pred_raw, pred_st) in pred_idx.items():
        if nk in gold_idx:
            continue
        extras.append(
            FieldVerdict(
                kind="status",
                key=pred_raw,
                gold=None,
                pred=pred_st,
                verdict="extra",
            )
        )
    return scored, extras


def _path_from_parts(
    columns: list[FieldVerdict],
    statuses: list[FieldVerdict],
    extra_columns: list[FieldVerdict],
    extra_statuses: list[FieldVerdict],
    *,
    table_notes: list[str] | None = None,
) -> PathScore:
    n_correct = sum(1 for v in (*columns, *statuses) if v.verdict == "correct")
    n_wrong = sum(1 for v in (*columns, *statuses) if v.verdict == "wrong")
    n_missing = sum(1 for v in (*columns, *statuses) if v.verdict == "missing")
    denom = n_correct + n_wrong + n_missing
    return PathScore(
        n_correct=n_correct,
        n_wrong=n_wrong,
        n_missing=n_missing,
        field_accuracy=1.0 if denom == 0 else n_correct / denom,
        columns=columns,
        statuses=statuses,
        extra_columns=extra_columns,
        extra_statuses=extra_statuses,
        table_notes=list(table_notes or ()),
    )


def score(pred: Mapping, gold: Mapping) -> MappingScore:
    tables_ok, table_notes = _score_tables(pred, gold, entities=_ORDER_PATH_ENTITIES)
    _, ne_table_notes = _score_tables(pred, gold, entities=_SIMPLE_ENTITY_SET)

    op_cols, op_extra_cols = _score_columns_filtered(
        pred, gold, include=_is_order_path_column
    )
    op_sts, op_extra_sts = _score_status_map(pred.status_map, gold.status_map)
    order_path = _path_from_parts(
        op_cols, op_sts, op_extra_cols, op_extra_sts, table_notes=table_notes
    )

    ne_cols, ne_extra_cols = _score_columns_filtered(
        pred, gold, include=_is_new_entity_column
    )
    ne_sts, ne_extra_sts = _score_status_map(
        pred.purchase_status_map, gold.purchase_status_map
    )
    new_entities = _path_from_parts(
        ne_cols, ne_sts, ne_extra_cols, ne_extra_sts, table_notes=ne_table_notes
    )
    status_ok = all(v.verdict == "correct" for v in op_sts)
    walk_in_ok = _walk_in_set(pred.walk_in_values) == _walk_in_set(gold.walk_in_values)

    return MappingScore(
        n_correct=order_path.n_correct,
        n_wrong=order_path.n_wrong,
        n_missing=order_path.n_missing,
        field_accuracy=order_path.field_accuracy,
        tables_ok=tables_ok,
        status_ok=status_ok,
        walk_in_ok=walk_in_ok,
        customer_identity_ok=pred.customer_identity == gold.customer_identity,
        return_rule_ok=pred.return_rule == gold.return_rule,
        data_end_ok=pred.data_end_date == gold.data_end_date,
        columns=order_path.columns,
        statuses=order_path.statuses,
        extra_columns=order_path.extra_columns,
        extra_statuses=order_path.extra_statuses,
        table_notes=table_notes,
        order_path=order_path,
        new_entities=new_entities,
    )

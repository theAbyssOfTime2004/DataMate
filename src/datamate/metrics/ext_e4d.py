"""E4d purchase / stock / lot / drug / margin family computations."""

from __future__ import annotations

import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from datamate.metrics.format import render_metric, render_percent
from datamate.parse.text import normalize_label
from datamate.query.ext.spec import (
    AvgPurchaseCostQuery,
    GrossProfitQuery,
    IngredientRevenueQuery,
    LotsExpiringQuery,
    NearestExpiryLotQuery,
    NetProfitQuery,
    OutOfStockCountQuery,
    PurchaseTotalQuery,
    RxShareQuery,
    StockOnHandQuery,
    StockValueQuery,
    TopEmployeeQuery,
    TopSupplierQuery,
)
from datamate.scoring.answer import Answer
from datamate.timeres.periods import PeriodError

if TYPE_CHECKING:
    from datamate.metrics.engine import MetricEngine


class _Abstain(Exception):
    pass


def answer_e4d(engine: MetricEngine, spec: object, as_of: datetime.date) -> Answer | None:
    """Dispatch E4d specs; return None when spec is not an E4d family."""
    from datamate.metrics.engine import Unanswerable

    try:
        if isinstance(spec, PurchaseTotalQuery):
            return _purchase_total(engine, spec, as_of)
        if isinstance(spec, TopSupplierQuery):
            return _top_supplier(engine, spec, as_of)
        if isinstance(spec, AvgPurchaseCostQuery):
            return _avg_purchase_cost(engine, spec, as_of)
        if isinstance(spec, StockOnHandQuery):
            return _stock_on_hand(engine, spec, as_of)
        if isinstance(spec, StockValueQuery):
            return _stock_value(engine, spec, as_of)
        if isinstance(spec, OutOfStockCountQuery):
            return _out_of_stock_count(engine, spec, as_of)
        if isinstance(spec, LotsExpiringQuery):
            return _lots_expiring(engine, spec, as_of)
        if isinstance(spec, NearestExpiryLotQuery):
            return _nearest_expiry_lot(engine, spec, as_of)
        if isinstance(spec, IngredientRevenueQuery):
            return _ingredient_revenue(engine, spec, as_of)
        if isinstance(spec, RxShareQuery):
            return _rx_share(engine, spec, as_of)
        if isinstance(spec, GrossProfitQuery):
            return _gross_profit(engine, spec, as_of)
        if isinstance(spec, NetProfitQuery):
            return _net_profit(engine, spec, as_of)
        if isinstance(spec, TopEmployeeQuery):
            return _top_employee(engine, spec, as_of)
    except Unanswerable as exc:
        return Answer.unanswerable(str(exc))
    except _Abstain:
        return Answer.abstain()
    except PeriodError:
        return Answer.abstain()
    return None


def _ext_recorded(engine: MetricEngine, table: str, column: str) -> bool:
    from datamate.metrics.ext_families import _ext_recorded as recorded

    return recorded(engine, table, column)


def _require_product_keys(engine: MetricEngine, product: str) -> set[str]:
    from datamate.metrics.ext_families import _require_product_keys as req

    return req(engine, product)


def _revenue_line_where(
    engine: MetricEngine, period: object, **kwargs: Any
) -> tuple[str, list[Any]]:
    from datamate.metrics.ext_families import _revenue_line_where as where

    return where(engine, period, **kwargs)  # type: ignore[arg-type]


def _as_of_day(engine: MetricEngine, expr: object, as_of: datetime.date) -> datetime.date:
    from datamate.timeres.periods import resolve

    window = resolve(expr, as_of, engine.rules, engine.cal)  # type: ignore[arg-type]
    if window.start != window.end:
        raise _Abstain()
    return window.start


def _purchase_not_cancelled_sql(alias: str = "p") -> str:
    return f"(NOT COALESCE({alias}.is_cancelled, FALSE))"


def _purchase_total(
    engine: MetricEngine, spec: PurchaseTotalQuery, as_of: datetime.date
) -> Answer:
    from datamate.metrics.engine import Unanswerable

    if not _ext_recorded(engine, "purchases", "total_amount"):
        raise Unanswerable("purchases.total_amount")
    cov, window = engine._window(spec.period, as_of)
    if cov == "none":
        return Answer.unanswerable("period lies entirely outside the shop's data")
    sql = f"""
        SELECT COALESCE(SUM(total_amount), 0)
        FROM purchases p
        WHERE p.receipt_date BETWEEN ? AND ?
          AND {_purchase_not_cancelled_sql("p")}
    """
    total = int(engine._scalar(sql, [window.start, window.end]) or 0)
    return Answer(type="number", answer=render_metric("revenue", total, engine.rules))


def _top_supplier(
    engine: MetricEngine, spec: TopSupplierQuery, as_of: datetime.date
) -> Answer:
    from datamate.metrics.engine import Unanswerable

    if not _ext_recorded(engine, "purchases", "total_amount"):
        raise Unanswerable("purchases.total_amount")
    cov, window = engine._window(spec.period, as_of)
    if cov == "none":
        return Answer.unanswerable("period lies entirely outside the shop's data")
    sql = f"""
        SELECT COALESCE(supplier_name, supplier_id) AS name,
               COALESCE(SUM(total_amount), 0) AS amount
        FROM purchases p
        WHERE p.receipt_date BETWEEN ? AND ?
          AND {_purchase_not_cancelled_sql("p")}
          AND (p.supplier_name IS NOT NULL OR p.supplier_id IS NOT NULL)
        GROUP BY COALESCE(supplier_name, supplier_id)
        ORDER BY amount DESC, name ASC
        LIMIT 1
    """
    row = engine.con.execute(sql, [window.start, window.end]).fetchone()
    if row is None or row[0] is None:
        raise _Abstain()
    return Answer(type="string", answer=str(row[0]))


def _avg_purchase_cost(
    engine: MetricEngine, spec: AvgPurchaseCostQuery, as_of: datetime.date
) -> Answer:
    from datamate.metrics.engine import Unanswerable

    if not _ext_recorded(engine, "purchase_lines", "unit_cost"):
        raise Unanswerable("purchase_lines.unit_cost")
    if not _ext_recorded(engine, "purchase_lines", "qty"):
        raise Unanswerable("purchase_lines.qty")
    cov, window = engine._window(spec.period, as_of)
    if cov == "none":
        return Answer.unanswerable("period lies entirely outside the shop's data")
    keys = _require_product_keys(engine, spec.product)
    marks = ", ".join("?" for _ in keys)
    sql = f"""
        SELECT COALESCE(SUM(pl.unit_cost * pl.qty), 0),
               COALESCE(SUM(pl.qty), 0)
        FROM purchase_lines pl
        LEFT JOIN purchases p ON p.receipt_id = pl.receipt_id
        WHERE pl.receipt_date BETWEEN ? AND ?
          AND pl.product_key IN ({marks})
          AND (p.receipt_id IS NULL OR {_purchase_not_cancelled_sql("p")})
          AND pl.unit_cost IS NOT NULL AND pl.qty IS NOT NULL
    """
    row = engine.con.execute(sql, [window.start, window.end, *sorted(keys)]).fetchone()
    if row is None or row[1] is None or Decimal(str(row[1])) == 0:
        raise _Abstain()
    avg = Decimal(str(row[0])) / Decimal(str(row[1]))
    return Answer(type="number", answer=render_metric("revenue", avg, engine.rules))


def _stock_qty_base(
    engine: MetricEngine, keys: set[str], day: datetime.date
) -> Decimal | None:
    """On-hand quantity in base units at end of day, or None if unknown."""
    marks = ", ".join("?" for _ in keys)
    params = [*sorted(keys), day]
    if _ext_recorded(engine, "stock_moves", "balance") or _ext_recorded(
        engine, "stock_moves", "qty_in"
    ):
        # Prefer last running balance on or before day; same-day moves without a time
        # follow source row order so the pick is deterministic.
        if _ext_recorded(engine, "stock_moves", "balance"):
            row = engine.con.execute(
                f"""
                SELECT balance FROM stock_moves
                WHERE product_key IN ({marks}) AND move_date <= ?
                ORDER BY move_date DESC, move_ts DESC NULLS LAST,
                         src_file DESC, src_sheet DESC, src_row DESC
                LIMIT 1
                """,
                params,
            ).fetchone()
            if row is not None and row[0] is not None:
                return Decimal(str(row[0]))
        # Fallback: sum in - out.
        if _ext_recorded(engine, "stock_moves", "qty_in"):
            row = engine.con.execute(
                f"""
                SELECT COALESCE(SUM(COALESCE(qty_in, 0) - COALESCE(qty_out, 0)), 0)
                FROM stock_moves
                WHERE product_key IN ({marks}) AND move_date <= ?
                """,
                params,
            ).fetchone()
            if row is not None:
                return Decimal(str(row[0]))
    if _ext_recorded(engine, "stock_snapshot", "qty_on_hand"):
        row = engine.con.execute(
            f"""
            SELECT qty_on_hand FROM stock_snapshot
            WHERE product_key IN ({marks}) AND as_of_date = ?
            LIMIT 1
            """,
            params,
        ).fetchone()
        if row is not None and row[0] is not None:
            return Decimal(str(row[0]))
    return None


def _conversion_factor(
    engine: MetricEngine, keys: set[str], unit: str
) -> Decimal | None:
    norm = normalize_label(unit)
    if norm is None:
        return None
    marks = ", ".join("?" for _ in keys)
    row = engine.con.execute(
        f"""
        SELECT factor FROM unit_conversions
        WHERE product_key IN ({marks}) AND lower(trim(unit)) = ?
        LIMIT 1
        """,
        [*sorted(keys), norm],
    ).fetchone()
    if row is None or row[0] is None:
        # Try without product scope (shop-wide unit name).
        row = engine.con.execute(
            "SELECT factor FROM unit_conversions WHERE lower(trim(unit)) = ? LIMIT 1",
            [norm],
        ).fetchone()
    if row is None or row[0] is None:
        return None
    return Decimal(str(row[0]))


def _product_base_unit_label(engine: MetricEngine, keys: set[str]) -> str | None:
    marks = ", ".join("?" for _ in keys)
    try:
        row = engine.con.execute(
            f"""
            SELECT base_unit FROM products
            WHERE product_key IN ({marks})
              AND base_unit IS NOT NULL AND base_unit <> ''
            LIMIT 1
            """,
            sorted(keys),
        ).fetchone()
    except Exception:  # noqa: BLE001
        return None
    if row is None or row[0] is None:
        return None
    return str(row[0]).strip()


def _stock_on_hand(
    engine: MetricEngine, spec: StockOnHandQuery, as_of: datetime.date
) -> Answer:
    from datamate.metrics.engine import Unanswerable

    has_moves = _ext_recorded(engine, "stock_moves", "balance") or _ext_recorded(
        engine, "stock_moves", "qty_in"
    )
    has_snap = _ext_recorded(engine, "stock_snapshot", "qty_on_hand")
    if not has_moves and not has_snap:
        # Stock can still be derived from purchases minus sales (rulebook "nhập trừ bán");
        # until that is implemented, abstain instead of claiming it is unanswerable.
        if _ext_recorded(engine, "purchase_lines", "qty"):
            raise _Abstain()
        raise Unanswerable("stock")
    day = _as_of_day(engine, spec.date, as_of)
    keys = _require_product_keys(engine, spec.product)
    qty = _stock_qty_base(engine, keys, day)
    if qty is None:
        raise _Abstain()
    base_unit = _product_base_unit_label(engine, keys)
    asked = spec.unit
    if asked is None or (
        base_unit is not None and normalize_label(asked) == normalize_label(base_unit)
    ):
        return Answer(
            type="number",
            answer=render_metric("order_count", qty, engine.rules),
        )
    factor = _conversion_factor(engine, keys, asked)
    if factor is None or factor == 0:
        raise _Abstain()
    whole = int(qty // factor)
    leftover = qty - (Decimal(whole) * factor)
    left_int = render_metric("order_count", leftover, engine.rules)
    return Answer(type="list", answer=[whole, left_int])


def _all_stock_balances(
    engine: MetricEngine, day: datetime.date
) -> dict[str, Decimal]:
    """product_key -> on-hand qty at end of day."""
    out: dict[str, Decimal] = {}
    if _ext_recorded(engine, "stock_moves", "balance") or _ext_recorded(
        engine, "stock_moves", "qty_in"
    ):
        if _ext_recorded(engine, "stock_moves", "balance"):
            rows = engine.con.execute(
                """
                SELECT product_key, balance FROM (
                    SELECT product_key, balance,
                           ROW_NUMBER() OVER (
                               PARTITION BY product_key
                               ORDER BY move_date DESC, move_ts DESC NULLS LAST,
                                        src_file DESC, src_sheet DESC, src_row DESC
                           ) AS rn
                    FROM stock_moves
                    WHERE move_date <= ? AND product_key IS NOT NULL
                ) t WHERE rn = 1 AND balance IS NOT NULL
                """,
                [day],
            ).fetchall()
            for key, bal in rows:
                out[str(key)] = Decimal(str(bal))
            return out
        rows = engine.con.execute(
            """
            SELECT product_key,
                   COALESCE(SUM(COALESCE(qty_in, 0) - COALESCE(qty_out, 0)), 0)
            FROM stock_moves
            WHERE move_date <= ? AND product_key IS NOT NULL
            GROUP BY product_key
            """,
            [day],
        ).fetchall()
        for key, bal in rows:
            out[str(key)] = Decimal(str(bal))
        return out
    if _ext_recorded(engine, "stock_snapshot", "qty_on_hand"):
        rows = engine.con.execute(
            """
            SELECT product_key, qty_on_hand FROM stock_snapshot
            WHERE as_of_date = ? AND product_key IS NOT NULL AND qty_on_hand IS NOT NULL
            """,
            [day],
        ).fetchall()
        for key, bal in rows:
            out[str(key)] = Decimal(str(bal))
    return out


def _stock_value(
    engine: MetricEngine, spec: StockValueQuery, as_of: datetime.date
) -> Answer:
    from datamate.metrics.engine import Unanswerable

    if not _ext_recorded(engine, "products", "unit_cost"):
        raise Unanswerable("products.unit_cost")
    has_stock = _ext_recorded(engine, "stock_moves", "balance") or _ext_recorded(
        engine, "stock_moves", "qty_in"
    ) or _ext_recorded(engine, "stock_snapshot", "qty_on_hand")
    if not has_stock:
        if _ext_recorded(engine, "purchase_lines", "qty"):
            raise _Abstain()  # derivable from purchases minus sales; not implemented yet
        raise Unanswerable("stock")
    day = _as_of_day(engine, spec.date, as_of)
    balances = _all_stock_balances(engine, day)
    if not balances:
        return Answer(type="number", answer=0)
    costs = {
        str(r[0]): Decimal(str(r[1]))
        for r in engine.con.execute(
            "SELECT product_key, unit_cost FROM products "
            "WHERE unit_cost IS NOT NULL AND product_key IS NOT NULL"
        ).fetchall()
    }
    total = Decimal(0)
    for key, qty in balances.items():
        if qty <= 0:
            continue
        cost = costs.get(key)
        if cost is None:
            continue
        total += qty * cost
    return Answer(type="number", answer=render_metric("revenue", total, engine.rules))


def _out_of_stock_count(
    engine: MetricEngine, spec: OutOfStockCountQuery, as_of: datetime.date
) -> Answer:
    from datamate.metrics.engine import Unanswerable

    if not _ext_recorded(engine, "purchase_lines", "product_key"):
        raise Unanswerable("purchase_lines")
    has_stock = _ext_recorded(engine, "stock_moves", "balance") or _ext_recorded(
        engine, "stock_moves", "qty_in"
    ) or _ext_recorded(engine, "stock_snapshot", "qty_on_hand")
    if not has_stock:
        if _ext_recorded(engine, "purchase_lines", "qty"):
            raise _Abstain()  # derivable from purchases minus sales; not implemented yet
        raise Unanswerable("stock")
    day = _as_of_day(engine, spec.date, as_of)
    resale = {
        str(r[0])
        for r in engine.con.execute(
            "SELECT DISTINCT product_key FROM purchase_lines WHERE product_key IS NOT NULL"
        ).fetchall()
    }
    balances = _all_stock_balances(engine, day)
    count = 0
    for key in resale:
        qty = balances.get(key, Decimal(0))
        if qty == 0:
            count += 1
    return Answer(type="number", answer=count)


def _lot_remaining_map(
    engine: MetricEngine, as_of_day: datetime.date
) -> dict[str, Decimal]:
    """lot_id -> remaining qty at as_of."""
    lots = engine.con.execute(
        """
        SELECT lot_id, COALESCE(qty_received, 0)
        FROM lots WHERE lot_id IS NOT NULL
        """
    ).fetchall()
    remaining = {str(lid): Decimal(str(qty)) for lid, qty in lots}
    if _ext_recorded(engine, "sale_lots", "qty"):
        sold_rows = engine.con.execute(
            """
            SELECT sl.lot_id, COALESCE(SUM(sl.qty), 0)
            FROM sale_lots sl
            JOIN orders o ON o.order_id = sl.order_id
            WHERE sl.lot_id IS NOT NULL
              AND o.order_date IS NOT NULL
              AND o.order_date <= ?
            GROUP BY sl.lot_id
            """,
            [as_of_day],
        ).fetchall()
        for lid, sold in sold_rows:
            key = str(lid)
            if key in remaining:
                remaining[key] -= Decimal(str(sold))
    return remaining


def _lots_expiring(
    engine: MetricEngine, spec: LotsExpiringQuery, as_of: datetime.date
) -> Answer:
    from datamate.metrics.engine import Unanswerable

    if not _ext_recorded(engine, "lots", "expiry_date"):
        raise Unanswerable("lots.expiry_date")
    if not _ext_recorded(engine, "lots", "qty_received"):
        raise Unanswerable("lots.qty_received")
    day = _as_of_day(engine, spec.date, as_of)
    end = day + datetime.timedelta(days=spec.n_days)
    remaining = _lot_remaining_map(engine, day)
    rows = engine.con.execute(
        """
        SELECT lot_id, product_name, product_id, expiry_date, unit_cost
        FROM lots
        WHERE expiry_date IS NOT NULL
          AND expiry_date >= ?
          AND expiry_date <= ?
        ORDER BY expiry_date ASC, lot_id ASC
        """,
        [day, end],
    ).fetchall()
    active: list[tuple[str, str, Decimal, Decimal | None]] = []
    for lid, pname, pid, _exp, ucost in rows:
        rem = remaining.get(str(lid), Decimal(0))
        if rem <= 0:
            continue
        label = str(pname) if pname else (str(pid) if pid else str(lid))
        cost = Decimal(str(ucost)) if ucost is not None else None
        active.append((str(lid), label, rem, cost))
    if spec.output == "count":
        return Answer(type="number", answer=len(active))
    if spec.output == "list":
        return Answer(type="list", answer=[label for _, label, _, _ in active])
    # cost value of remaining qty at lot's own cost
    if any(c is None for *_, c in active):
        # Missing unit_cost on any qualifying lot -> unanswerable for cost output
        if not _ext_recorded(engine, "lots", "unit_cost"):
            raise Unanswerable("lots.unit_cost")
        raise _Abstain()
    total = sum((rem * cost for *_, rem, cost in active if cost is not None), Decimal(0))
    return Answer(type="number", answer=render_metric("revenue", total, engine.rules))


def _nearest_expiry_lot(
    engine: MetricEngine, spec: NearestExpiryLotQuery, as_of: datetime.date
) -> Answer:
    from datamate.metrics.engine import Unanswerable

    if not _ext_recorded(engine, "lots", "expiry_date"):
        raise Unanswerable("lots.expiry_date")
    day = (
        _as_of_day(engine, spec.date, as_of)
        if spec.date is not None
        else as_of
    )
    keys = _require_product_keys(engine, spec.product)
    remaining = _lot_remaining_map(engine, day)
    marks = ", ".join("?" for _ in keys)
    rows = engine.con.execute(
        f"""
        SELECT lot_id, expiry_date
        FROM lots
        WHERE product_key IN ({marks})
          AND expiry_date IS NOT NULL
          AND expiry_date >= ?
        ORDER BY expiry_date ASC, lot_id ASC
        """,
        [*sorted(keys), day],
    ).fetchall()
    for lid, _exp in rows:
        if remaining.get(str(lid), Decimal(0)) > 0:
            return Answer(type="string", answer=str(lid))
    raise _Abstain()


def _ingredient_revenue(
    engine: MetricEngine, spec: IngredientRevenueQuery, as_of: datetime.date
) -> Answer:
    from datamate.metrics.engine import Unanswerable

    if not _ext_recorded(engine, "drug_info", "active_ingredient"):
        raise Unanswerable("drug_info.active_ingredient")
    cov, window = engine._window(spec.period, as_of)
    if cov == "none":
        return Answer.unanswerable("period lies entirely outside the shop's data")
    target = normalize_label(spec.active_ingredient)
    if target is None:
        raise _Abstain()
    where, params = _revenue_line_where(engine, window)
    sql = f"""
        SELECT COALESCE(SUM(l.line_amount), 0)
        FROM order_lines l
        JOIN orders o ON o.order_id = l.order_id
        JOIN drug_info d ON d.product_key = l.product_key
        WHERE {where}
          AND d.active_ingredient IS NOT NULL
    """
    # Exact match after normalize_label: filter in Python for NFC/casefold consistency.
    rows = engine.con.execute(
        f"""
        SELECT l.line_amount, d.active_ingredient
        FROM order_lines l
        JOIN orders o ON o.order_id = l.order_id
        JOIN drug_info d ON d.product_key = l.product_key
        WHERE {where}
          AND d.active_ingredient IS NOT NULL
        """,
        params,
    ).fetchall()
    total = 0
    for amount, ingredient in rows:
        if normalize_label(str(ingredient)) == target:
            total += int(amount or 0)
    del sql
    return Answer(type="number", answer=render_metric("revenue", total, engine.rules))


def _rx_share(engine: MetricEngine, spec: RxShareQuery, as_of: datetime.date) -> Answer:
    from datamate.metrics.engine import Unanswerable

    if not _ext_recorded(engine, "drug_info", "is_prescription"):
        raise Unanswerable("drug_info.is_prescription")
    cov, window = engine._window(spec.period, as_of)
    if cov == "none":
        return Answer.unanswerable("period lies entirely outside the shop's data")
    where, params = _revenue_line_where(engine, window)
    total_sql = f"""
        SELECT COALESCE(SUM(l.line_amount), 0)
        FROM order_lines l JOIN orders o ON o.order_id = l.order_id
        WHERE {where}
    """
    rx_sql = f"""
        SELECT COALESCE(SUM(l.line_amount), 0)
        FROM order_lines l
        JOIN orders o ON o.order_id = l.order_id
        JOIN drug_info d ON d.product_key = l.product_key
        WHERE {where} AND COALESCE(d.is_prescription, FALSE)
    """
    total = int(engine._scalar(total_sql, params) or 0)
    if total == 0:
        raise _Abstain()
    rx = int(engine._scalar(rx_sql, params) or 0)
    percent = (Decimal(rx) * Decimal(100)) / Decimal(total)
    return Answer(type="percent", answer=render_percent(percent, engine.rules))


def _gross_profit_amount(
    engine: MetricEngine, window: object
) -> int:
    from datamate.metrics.engine import Unanswerable
    from datamate.metrics.ext_families import _any_null_cost
    from datamate.timeres.periods import DateRange

    assert isinstance(window, DateRange)
    if not _ext_recorded(engine, "order_lines", "cost_per_unit"):
        raise Unanswerable("cost_per_unit")
    if not _ext_recorded(engine, "order_lines", "qty"):
        raise Unanswerable("qty")
    where, params = _revenue_line_where(engine, window)
    if _any_null_cost(engine, where, params):
        raise _Abstain()
    sql = f"""
        SELECT COALESCE(SUM(l.line_amount), 0) - COALESCE(SUM(l.qty * l.cost_per_unit), 0)
        FROM order_lines l JOIN orders o ON o.order_id = l.order_id
        WHERE {where}
    """
    return int(engine._scalar(sql, params) or 0)


def _gross_profit(
    engine: MetricEngine, spec: GrossProfitQuery, as_of: datetime.date
) -> Answer:
    cov, window = engine._window(spec.period, as_of)
    if cov == "none":
        return Answer.unanswerable("period lies entirely outside the shop's data")
    amount = _gross_profit_amount(engine, window)
    return Answer(type="number", answer=render_metric("revenue", amount, engine.rules))


def _net_profit(
    engine: MetricEngine, spec: NetProfitQuery, as_of: datetime.date
) -> Answer:
    from datamate.metrics.engine import Unanswerable

    if not _ext_recorded(engine, "expenses", "amount"):
        raise Unanswerable("expenses")
    cov, window = engine._window(spec.period, as_of)
    if cov == "none":
        return Answer.unanswerable("period lies entirely outside the shop's data")
    gross = _gross_profit_amount(engine, window)
    exp = int(
        engine._scalar(
            """
            SELECT COALESCE(SUM(amount), 0) FROM expenses
            WHERE expense_date BETWEEN ? AND ?
            """,
            [window.start, window.end],
        )
        or 0
    )
    return Answer(
        type="number",
        answer=render_metric("revenue", gross - exp, engine.rules),
    )


def _top_employee(
    engine: MetricEngine, spec: TopEmployeeQuery, as_of: datetime.date
) -> Answer:
    from datamate.metrics.engine import Unanswerable

    if not _ext_recorded(engine, "orders", "employee_id"):
        raise Unanswerable("orders.employee_id")
    cov, window = engine._window(spec.period, as_of)
    if cov == "none":
        return Answer.unanswerable("period lies entirely outside the shop's data")
    where, params = _revenue_line_where(engine, window)
    sql = f"""
        SELECT o.employee_id,
               COALESCE(MAX(e.employee_name), o.employee_id) AS name,
               COALESCE(SUM(l.line_amount), 0) AS amount
        FROM order_lines l
        JOIN orders o ON o.order_id = l.order_id
        LEFT JOIN employees e ON e.employee_id = o.employee_id
        WHERE {where} AND o.employee_id IS NOT NULL
        GROUP BY o.employee_id
        ORDER BY amount DESC, o.employee_id ASC
        LIMIT 1
    """
    row = engine.con.execute(sql, params).fetchone()
    if row is None or row[1] is None:
        raise _Abstain()
    return Answer(type="string", answer=str(row[1]))

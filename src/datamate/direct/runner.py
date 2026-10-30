"""Child-process entrypoint for sandbox-executed pandas snippets.

Invoked as: python -I runner.py <tables_dir> <code_file> <as_of_iso>
"""

from __future__ import annotations

import datetime
import json
import pickle
import sys
import traceback
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Project src/ so helpers import without relying on PYTHONPATH (-I clears it).
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from datamate.parse import normalize_label, parse_date, parse_decimal, parse_money


class TablesDict(dict):
    """dict[str, DataFrame] that explains available keys on a miss."""

    def __missing__(self, key: str) -> Any:
        available = ", ".join(sorted(self.keys()))
        raise KeyError(f"{key} not found; available tables: {available}")


def _load_tables(tables_dir: Path) -> TablesDict:
    manifest_path = tables_dir / "manifest.json"
    files: dict[str, str] = {}
    aliases: dict[str, str] = {}
    if manifest_path.is_file():
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "files" in raw:
            files = {str(k): str(v) for k, v in raw["files"].items()}
            aliases = {str(k): str(v) for k, v in raw.get("aliases", {}).items()}
        elif isinstance(raw, dict):
            # Legacy: {pkl_name: table_name}
            files = {str(k): str(v) for k, v in raw.items()}
    tables = TablesDict()
    for path in sorted(tables_dir.glob("*.pkl")):
        key = files.get(path.name, path.stem)
        with path.open("rb") as handle:
            tables[key] = pickle.load(handle)
    for alias, target in aliases.items():
        if alias in tables or target not in tables:
            continue
        tables[alias] = tables[target]
    return tables


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, datetime.datetime):
        trimmed = value.replace(microsecond=0)
        if trimmed.time() == datetime.time(0, 0, 0):
            return trimmed.date().isoformat()
        return trimmed.isoformat(sep=" ")
    if isinstance(value, datetime.date):
        return value.isoformat()
    if isinstance(value, Decimal):
        if value == value.to_integral_value():
            return int(value)
        return float(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    type_name = type(value).__name__
    module = type(value).__module__
    if type_name == "Period" and "pandas" in module:
        return str(value)
    if type_name in {"Timestamp", "datetime64"} or (
        module.startswith("numpy") and "datetime64" in type_name
    ):
        try:
            import pandas as pd

            return _jsonable(pd.Timestamp(value).to_pydatetime())
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(f"cannot JSON-encode {type_name}: {value!r}") from exc
    if module == "numpy" or module.startswith("numpy."):
        item = getattr(value, "item", None)
        if callable(item):
            return _jsonable(item())
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise TypeError(f"cannot JSON-encode {type_name}: {value!r}")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 3:
        print(json.dumps({"error": "usage: runner.py tables_dir code_file as_of"}))
        return 2
    tables_dir = Path(args[0])
    code_file = Path(args[1])
    as_of = datetime.date.fromisoformat(args[2])
    try:
        tables = _load_tables(tables_dir)
        code = code_file.read_text(encoding="utf-8")
        namespace: dict[str, Any] = {
            "tables": tables,
            "AS_OF": as_of,
            "parse_money": parse_money,
            "parse_date": parse_date,
            "parse_decimal": parse_decimal,
            "normalize_label": normalize_label,
            "datetime": datetime,
            "date": datetime.date,
            "timedelta": datetime.timedelta,
            "Decimal": Decimal,
            "pd": pd,
            "np": np,
        }
        exec(compile(code, str(code_file), "exec"), namespace, namespace)  # noqa: S102
        if "result" not in namespace:
            raise RuntimeError("code did not assign `result`")
        payload = _jsonable(namespace["result"])
        if not isinstance(payload, dict):
            raise TypeError("result must be a dict")
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    except Exception:  # noqa: BLE001 — report to parent as JSON
        tb = traceback.format_exc()
        lines = tb.strip().splitlines()
        print(json.dumps({"error": "\n".join(lines[-20:])}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

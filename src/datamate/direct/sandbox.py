"""AST gate and subprocess runner for LLM-written pandas code."""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datamate.config import settings

_RUNNER = Path(__file__).resolve().parent / "runner.py"

_ALLOWED_IMPORTS = frozenset(
    {
        "pandas",
        "numpy",
        "datetime",
        "math",
        "re",
        "decimal",
        "collections",
        "statistics",
        "itertools",
    }
)

_FORBIDDEN_NAMES = frozenset(
    {
        "open",
        "exec",
        "eval",
        "compile",
        "__import__",
        "globals",
        "locals",
        "vars",
        "input",
        "breakpoint",
        "getattr",
        "setattr",
        "delattr",
        "os",
        "sys",
        "subprocess",
        "shutil",
        "socket",
        "pathlib",
    }
)

_PROVIDED_NAMES = frozenset(
    {
        "parse_money",
        "parse_date",
        "parse_decimal",
        "normalize_label",
        "tables",
        "AS_OF",
    }
)

_FORBIDDEN_ATTRS = frozenset(
    {
        "load",
        "save",
        "savetxt",
        "loadtxt",
        "fromfile",
        "tofile",
    }
)

_FORBIDDEN_ATTR_RE = re.compile(
    r"^(?:read_.*|to_(?:csv|pickle|excel|parquet|feather|hdf|sql|json|html|xml|"
    r"stata|clipboard|orc|latex|markdown))$"
)


@dataclass
class SandboxResult:
    ok: bool
    result: dict[str, Any] | None
    error: str | None
    elapsed_s: float


def check_code(code: str) -> str | None:
    """Return an error message if the code is not allowed to run, else None."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"syntax error: {exc}"

    if _has_module_level_return_or_yield(tree):
        return "write a top-level script that assigns result"

    if not any(isinstance(node, ast.Name) and node.id == "tables" for node in ast.walk(tree)):
        return "compute from tables; do not hard-code the answer"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in _ALLOWED_IMPORTS:
                    return f"import not allowed: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                return "import from without module is not allowed"
            root = node.module.split(".", 1)[0]
            if root not in _ALLOWED_IMPORTS:
                return f"import not allowed: {node.module}"
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in _PROVIDED_NAMES:
                return "these are provided; do not redefine them"
        elif isinstance(node, ast.Name):
            if node.id in _FORBIDDEN_NAMES or node.id.startswith("__"):
                return f"name not allowed: {node.id}"
        elif isinstance(node, ast.Attribute):
            attr = node.attr
            if attr.startswith("__"):
                return f"attribute not allowed: {attr}"
            if attr in _FORBIDDEN_ATTRS or _FORBIDDEN_ATTR_RE.match(attr):
                return f"attribute not allowed: {attr}"
        elif isinstance(node, ast.Assign):
            if _assigns_provided(node.targets):
                return "these are provided; do not redefine them"
        elif isinstance(node, ast.AnnAssign):
            if node.target is not None and _assigns_provided([node.target]):
                return "these are provided; do not redefine them"
        elif isinstance(node, ast.AugAssign) and _assigns_provided([node.target]):
            return "these are provided; do not redefine them"
    return None


def _assigns_provided(targets: list[ast.expr] | tuple[ast.expr, ...]) -> bool:
    for target in targets:
        for node in ast.walk(target):
            if isinstance(node, ast.Name) and node.id in _PROVIDED_NAMES:
                return True
    return False


def _has_module_level_return_or_yield(tree: ast.AST) -> bool:
    if not isinstance(tree, ast.Module):
        return False
    for node in tree.body:
        if isinstance(node, ast.Return):
            return True
        if isinstance(node, ast.Expr) and isinstance(node.value, (ast.Yield, ast.YieldFrom)):
            return True
    return False


def run_code(
    code: str,
    tables_dir: Path | str,
    as_of: Any,
    timeout_s: float | None = None,
) -> SandboxResult:
    """Run `code` in an isolated child process. Never runs code that fails check_code."""
    blocked = check_code(code)
    if blocked is not None:
        return SandboxResult(ok=False, result=None, error=blocked, elapsed_s=0.0)

    limit = settings.direct_timeout_s if timeout_s is None else timeout_s
    as_of_text = as_of.isoformat() if hasattr(as_of, "isoformat") else str(as_of)
    tables_path = Path(tables_dir)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="datamate_direct_") as tmp:
        tmp_dir = Path(tmp)
        code_file = tmp_dir / "snippet.py"
        code_file.write_text(code, encoding="utf-8")
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(_RUNNER),
                    str(tables_path),
                    str(code_file),
                    as_of_text,
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=limit,
                cwd=str(tmp_dir),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                ok=False,
                result=None,
                error="timeout",
                elapsed_s=round(time.perf_counter() - started, 6),
            )
    elapsed = round(time.perf_counter() - started, 6)
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    last_line = _last_nonempty_line(stdout)
    if not last_line:
        detail = stderr.strip() or f"exit {completed.returncode}, empty stdout"
        return SandboxResult(ok=False, result=None, error=detail, elapsed_s=elapsed)
    try:
        payload = json.loads(last_line)
    except json.JSONDecodeError:
        return SandboxResult(
            ok=False,
            result=None,
            error=f"invalid JSON from runner: {last_line[:500]}",
            elapsed_s=elapsed,
        )
    if not isinstance(payload, dict):
        return SandboxResult(
            ok=False, result=None, error="runner result is not an object", elapsed_s=elapsed
        )
    if "error" in payload and "type" not in payload:
        return SandboxResult(
            ok=False, result=None, error=str(payload["error"]), elapsed_s=elapsed
        )
    return SandboxResult(ok=True, result=payload, error=None, elapsed_s=elapsed)


def _last_nonempty_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()
    return ""

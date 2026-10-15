"""Pre-execution static analysis of generated pipeline scripts.

Catches the common LLM hallucinations before spending a sandbox run on them:
syntax errors, imports of modules/names that don't exist, keyword arguments a
function or class does not accept (e.g. OneHotEncoder(sparse=...) on sklearn ≥1.4),
and violations of the data contract. Signatures are checked against the installed
libraries, which the sandbox image pins to the same versions.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from typing import Any

from automl_system.pipeline.code_generator import _ALLOWED_IMPORT_PREFIXES

_REQUIRED_CONSTANTS = ("TRAIN_PATH", "TEST_PATH", "TARGET_COLUMN", "PREDICTIONS_PATH")
_FORBIDDEN = {
    "GridSearchCV": "no hyperparameter search — use the recorded hyperparameters",
    "RandomizedSearchCV": "no hyperparameter search — use the recorded hyperparameters",
    "cross_val_score": "no cross-validation — train once on all TRAIN rows",
    "optuna": "no hyperparameter search — use the recorded hyperparameters",
}
_GPU_VALUES = {"cuda", "gpu", "GPU", "gpu_hist"}


def _allowed(module: str) -> bool:
    top = module.split(".")[0]
    return any(top == p or top.startswith(p) for p in _ALLOWED_IMPORT_PREFIXES)


def _collect_imports(tree: ast.AST, issues: list[str]) -> dict[str, Any]:
    """Map local names to imported objects; report modules/names that don't exist."""
    names: dict[str, Any] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if not _allowed(a.name):
                    continue
                try:
                    mod = importlib.import_module(a.name)
                except ImportError:
                    issues.append(f"line {node.lineno}: module {a.name!r} does not exist")
                    continue
                names[a.asname or a.name.split(".")[0]] = mod if a.asname else importlib.import_module(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            if not _allowed(node.module):
                continue
            try:
                mod = importlib.import_module(node.module)
            except ImportError:
                issues.append(f"line {node.lineno}: module {node.module!r} does not exist")
                continue
            for a in node.names:
                if a.name == "*":
                    continue
                obj = getattr(mod, a.name, None)
                if obj is None:
                    try:
                        obj = importlib.import_module(f"{node.module}.{a.name}")
                    except ImportError:
                        issues.append(f"line {node.lineno}: {node.module!r} has no attribute {a.name!r}")
                        continue
                names[a.asname or a.name] = obj
    return names


def _resolve(func: ast.AST, names: dict[str, Any]) -> tuple[str, Any] | None:
    """Resolve `Name` / `module.attr.attr` call targets to imported objects."""
    chain: list[str] = []
    while isinstance(func, ast.Attribute):
        chain.append(func.attr)
        func = func.value
    if not isinstance(func, ast.Name) or func.id not in names:
        return None
    obj = names[func.id]
    for attr in reversed(chain):
        if not inspect.ismodule(obj):
            return None  # attribute of a class/instance — runtime-dependent, skip
        obj = getattr(obj, attr, None)
        if obj is None:
            return None
    return ".".join([func.id, *reversed(chain)]), obj


def _check_keywords(tree: ast.AST, names: dict[str, Any], issues: list[str]) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.keywords:
            continue
        resolved = _resolve(node.func, names)
        if resolved is None:
            continue
        label, obj = resolved
        if not callable(obj):
            continue
        try:
            params = inspect.signature(obj).parameters
        except (TypeError, ValueError):
            continue
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            continue
        valid = [n for n, p in params.items() if p.kind is not inspect.Parameter.VAR_POSITIONAL]
        for kw in node.keywords:
            if kw.arg and kw.arg not in params:
                issues.append(
                    f"line {node.lineno}: {label}() has no keyword argument {kw.arg!r} "
                    f"in the installed version (valid: {', '.join(valid[:15])})"
                )


def _check_contract(tree: ast.Module, code: str, issues: list[str]) -> None:
    top_level = {
        t.id
        for node in tree.body if isinstance(node, ast.Assign)
        for t in node.targets if isinstance(t, ast.Name)
    }
    missing = [c for c in _REQUIRED_CONSTANTS if c not in top_level]
    if missing:
        issues.append(f"contract: missing top-level constant(s) {missing}")
    if "to_csv" not in code or "PREDICTIONS_PATH" not in code.split("PREDICTIONS_PATH =", 1)[-1]:
        issues.append("contract: predictions are never written to PREDICTIONS_PATH")
    # Inspect code identifiers and string values only — comments may mention anything.
    identifiers: set[str] = set()
    strings: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr)
        elif isinstance(node, ast.alias):
            identifiers.update(node.name.split("."))
        elif isinstance(node, ast.ImportFrom) and node.module:
            identifiers.update(node.module.split("."))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            strings.add(node.value)
    for token, why in _FORBIDDEN.items():
        if token in identifiers:
            issues.append(f"contract: uses {token} — {why}")
    if strings & _GPU_VALUES:
        issues.append("contract: GPU settings found — the script must run on CPU")


def static_check(code: str) -> list[str]:
    """Return a list of problems found without executing the code (empty = passed)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"line {e.lineno}: SyntaxError: {e.msg}"]
    issues: list[str] = []
    names = _collect_imports(tree, issues)
    _check_keywords(tree, names, issues)
    _check_contract(tree, code, issues)
    return issues

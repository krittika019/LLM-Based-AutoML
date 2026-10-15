"""LLM-written feature functions, executed in the Docker sandbox (feature engineering, option B).

The LLM writes `make_feature(train, test)` functions over the original-scale feature
columns (the target is never passed in). Each round's functions are statically
checked, then executed together in one sandbox call; their output values come back
to the host, where the CV gate in feature_engineering.py decides what to keep.

LLM code is never executed in the AutoML server process: the sandbox runs it during
the pipeline, and the generated script runs it via replay.run_code_feature.
"""

from __future__ import annotations

import ast
import inspect
import io
import json
import logging
from typing import Any

import pandas as pd

from automl_system import config
from automl_system.pipeline import replay
from automl_system.sandbox import Sandbox, docker_available

logger = logging.getLogger(__name__)

_ALLOWED_IMPORTS = {"pandas", "numpy", "math"}
_FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars",
    "input", "breakpoint", "getattr", "setattr", "delattr", "memoryview", "exit", "quit",
}
_MAX_CODE_CHARS = 4_000

_RUNNER = '''
import json
import traceback

import numpy as np
import pandas as pd

{run_code_feature}

train = pd.read_csv("fe_train.csv")
test = pd.read_csv("fe_test.csv")
candidates = json.load(open("fe_candidates.json", encoding="utf-8"))
out_train, out_test, meta, errors = {{}}, {{}}, {{}}, {{}}
for cand in candidates:
    try:
        tr, te = run_code_feature(train, test, cand["code"])
        for col in tr.columns[:3]:
            key = cand["name"] if tr.shape[1] == 1 else cand["name"] + "__" + col
            out_train[key] = tr[col].to_numpy()
            out_test[key] = te[col].to_numpy()
            meta[key] = {{"name": cand["name"], "column": col}}
    except Exception:
        errors[cand["name"]] = traceback.format_exc(limit=3)[-1200:]
pd.DataFrame(out_train).to_csv("fe_out_train.csv", index=False)
pd.DataFrame(out_test).to_csv("fe_out_test.csv", index=False)
json.dump({{"meta": meta, "errors": errors}}, open("fe_out_meta.json", "w", encoding="utf-8"))
'''


def code_features_enabled() -> bool:
    """Only with real isolation: Docker sandbox available and not switched off."""
    return config.FE_CODE_FEATURES != "off" and config.SANDBOX_BACKEND != "off" and docker_available()


def normalize_feature_code(code: str) -> str:
    """Repair common LLM formatting slips: markdown fences, and newlines that arrive as
    literal backslash-n text because the JSON string was escaped twice."""
    code = code.strip()
    if code.startswith("```"):
        code = "\n".join(line for line in code.splitlines() if not line.strip().startswith("```"))
    if "\n" not in code and "\\n" in code:
        code = code.replace("\\n", "\n").replace("\\t", "\t")
    return code.strip() + "\n"


def is_safety_problem(problem: str | None) -> bool:
    """True for policy violations (forbidden names/imports); False for plain mistakes."""
    return bool(problem) and any(k in problem for k in ("not allowed", "longer than"))


def check_feature_code(code: str) -> str | None:
    """Static check before execution. Returns a problem description, or None if OK."""
    if len(code) > _MAX_CODE_CHARS:
        return f"code longer than {_MAX_CODE_CHARS} characters"
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError line {e.lineno}: {e.msg}"
    if not any(isinstance(n, ast.FunctionDef) and n.name == "make_feature" for n in tree.body):
        return "must define make_feature(train, test)"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            bad = [m for m in mods if m.split(".")[0] not in _ALLOWED_IMPORTS]
            if bad:
                return f"import of {bad} not allowed (only pandas, numpy, math)"
        elif isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            return f"use of {node.id!r} not allowed"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return f"dunder attribute {node.attr!r} not allowed"
    return None


class CodeFeatureRunner:
    """One Docker sandbox for the whole feature-engineering step."""

    def __init__(self, src_train: pd.DataFrame, src_test: pd.DataFrame | None):
        self.input_columns = list(src_train.columns)
        self.sandbox = Sandbox("docker")
        self.sandbox.start()
        self.sandbox.put_dataframe("fe_train.csv", src_train)
        self.sandbox.put_dataframe(
            "fe_test.csv", src_test if src_test is not None else src_train.iloc[:0]
        )
        self.runner = _RUNNER.format(run_code_feature=inspect.getsource(replay.run_code_feature))

    def close(self) -> None:
        self.sandbox.close()

    def run(self, proposals: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
        """Execute proposals in the sandbox. Returns (candidates, errors by proposal name).

        Each candidate: {"name", "column", "code", "input_columns", "rationale",
        "train_values": Series, "test_values": ndarray}.
        """
        (self.sandbox.workdir / "fe_candidates.json").write_text(
            json.dumps([{"name": p["name"], "code": p["code"]} for p in proposals]), encoding="utf-8"
        )
        for f in ("fe_out_train.csv", "fe_out_test.csv", "fe_out_meta.json"):
            (self.sandbox.workdir / f).unlink(missing_ok=True)
        res = self.sandbox.run(self.runner, timeout=config.FE_CODE_TIMEOUT_SEC)
        meta_text = self.sandbox.read_text("fe_out_meta.json")
        if not res.ok or meta_text is None:
            why = "timed out" if res.timed_out else res.stderr[-600:]
            return [], {p["name"]: f"sandbox run failed: {why}" for p in proposals}

        out = json.loads(meta_text)
        train_text = self.sandbox.read_text("fe_out_train.csv") or ""
        test_text = self.sandbox.read_text("fe_out_test.csv") or ""
        tr = pd.read_csv(io.StringIO(train_text)) if out["meta"] else pd.DataFrame()
        te = pd.read_csv(io.StringIO(test_text)) if out["meta"] else pd.DataFrame()
        by_name = {p["name"]: p for p in proposals}
        candidates = []
        for key, m in out["meta"].items():
            p = by_name[m["name"]]
            candidates.append({
                "name": key, "column": m["column"], "code": p["code"],
                "input_columns": self.input_columns, "rationale": p.get("rationale", ""),
                "train_values": tr[key] if key in tr.columns else pd.Series(dtype=float),
                "test_values": te[key].to_numpy() if key in te.columns else None,
            })
        return candidates, dict(out["errors"])

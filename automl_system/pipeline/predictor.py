"""Predict with a finished run's model on new rows ("Try the model").

At the end of a run, `build_bundle` packs the fitted estimators (shaped like the recorded
model spec: single / threshold / bag / average / stacking), the run's raw training split
and its preprocessing recipe into one picklable dict. `Predictor` loads that bundle and
predicts on new raw rows by replaying exactly the transforms the verified pipeline script
uses (replay.py, fit on the training split only) — no retraining.
"""

from __future__ import annotations

import logging
import math
import types
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from automl_system.pipeline import replay

logger = logging.getLogger(__name__)

BUNDLE_VERSION = 1
_TREE_HINTS = ("Forest", "Tree", "Boost", "XGB", "LGBM", "CatBoost", "GradientBoosting")


class PredictUnavailable(Exception):
    """The run's model cannot be used for prediction."""


def bag_members(models: list) -> types.SimpleNamespace:
    """Holder for seed-bagging models kept on a result row. A non-plain object, so
    session copies (_copy_data) skip it like `_fitted_estimator`."""
    return types.SimpleNamespace(models=list(models))


# ── bundle (built in the run's process, where the fitted objects live) ─────────

def _row_for(rows: list[dict], kind: str, family: str | None = None) -> dict | None:
    for r in reversed(rows):
        spec = r.get("model_spec") or {}
        if r.get("status") != "ok" or spec.get("kind") != kind or r.get("_fitted_estimator") is None:
            continue
        if family is None or spec.get("family") == family:
            return r
    return None


def _cpu(est: Any) -> Any:
    """Prediction happens on the API server's CPU; detach XGBoost from the GPU."""
    try:
        if type(est).__name__.startswith("XGB") and "device" in est.get_params():
            est.set_params(device="cpu")
    except Exception:
        pass
    return est


def _fitted_tree(spec: dict, rows: list[dict]) -> dict:
    kind = (spec or {}).get("kind")
    if kind == "single":
        row = _row_for(rows, "single", spec.get("family"))
        if row is None:
            raise PredictUnavailable(f"no fitted model kept for {spec.get('family')}")
        return {"kind": "single", "est": _cpu(row["_fitted_estimator"]), "family": spec.get("family"),
                "impute": bool(spec.get("impute_median"))}
    if kind == "threshold":
        return {"kind": "threshold", "base": _fitted_tree(spec.get("base"), rows), "threshold": float(spec["threshold"])}
    if kind == "bag":
        row = _row_for(rows, "bag")
        members = getattr(row.get("_bag_members"), "models", None) if row else None
        if row is None or not members:
            raise PredictUnavailable("seed-bagging models were not kept")
        return {"kind": "bag", "base": _fitted_tree(spec.get("base"), rows), "members": [_cpu(m) for m in members]}
    if kind == "average":
        return {"kind": "average", "members": [_fitted_tree(m, rows) for m in spec.get("members") or []],
                "weights": [float(w) for w in spec.get("weights") or []], "soft": bool(spec.get("soft"))}
    if kind == "stacking":
        row = _row_for(rows, "stacking")
        if row is None:
            raise PredictUnavailable("stacking ensemble was not kept")
        return {"kind": "stacking", "est": _cpu(row["_fitted_estimator"])}
    raise PredictUnavailable(f"unsupported model kind {kind!r}")


def build_bundle(session: Any, rows: list[dict], raw_train: pd.DataFrame) -> dict[str, Any]:
    """Everything needed to predict on new raw rows. Raises PredictUnavailable."""
    res = session.execution_results or {}
    rp = res.get("replay") or {}
    target = str((session.task_config or {}).get("target_column") or "")
    if not rp.get("model") or not target or raw_train is None or target not in raw_train.columns:
        raise PredictUnavailable("the run did not record a replayable model")
    specs = [e["spec"] for e in (session.feature_engineering_log or [])
             if e.get("status") == "added" and isinstance(e.get("spec"), dict)]
    if any(e.get("status") == "added" and not isinstance(e.get("spec"), dict)
           for e in (session.feature_engineering_log or [])):
        raise PredictUnavailable("an engineered feature has no recorded recipe")
    return {
        "version": BUNDLE_VERSION,
        "model": _fitted_tree(rp["model"], rows),
        "raw_train": raw_train.reset_index(drop=True),
        "target": target,
        "is_classification": str(res.get("task_type") or "") == "classification",
        "log_target": bool(rp.get("log_target")),
        "keep_columns": list((session.preprocessing_meta or {}).get("feature_selection_columns") or []),
        "assignments": {c: a for c, a in (session.deep_cleaning_assignments or {}).items() if isinstance(a, dict)},
        "raw_specs": [s for s in specs if s.get("source") == "raw"],
        "fe_specs": [s for s in specs if s.get("source") != "raw"],
        "feature_columns": list(rp.get("feature_columns") or []),
        "best_model": res.get("best_model_name"),
        "metric": res.get("metric_display") or res.get("metric"),
        "decision_threshold": (rp["model"] or {}).get("threshold"),
    }


# ── prediction (API server) ────────────────────────────────────────────────────

def _add_engineered(train: pd.DataFrame, test: pd.DataFrame, specs: list[dict]):
    """Same as the generated script's add_engineered_features (train fills reused on test)."""
    for spec in specs:
        name = spec["name"]
        feat = replay.compute_raw_feature(train, spec)
        if spec.get("fill") is not None:
            feat = feat.fillna(spec["fill"])
        train[name] = feat
        test_fill = spec.get("test_fill")
        try:
            feat_test = replay.compute_raw_feature(test, spec)
            if test_fill is not None:
                feat_test = feat_test.fillna(test_fill)
        except Exception:
            feat_test = test_fill if test_fill is not None else 0.0
        test[name] = feat_test
    return train, test.reindex(columns=train.columns, fill_value=0.0)


def _numeric_like_train(x_train: pd.DataFrame, x_test: pd.DataFrame):
    """replay.safe_numeric_features, but leftover text columns in test use train's codes
    (a lone new row factorized on its own would always get code 0)."""
    out_tr, out_te = x_train.copy(), x_test.copy()
    for c in out_tr.columns:
        if pd.api.types.is_bool_dtype(out_tr[c]):
            out_tr[c] = out_tr[c].astype(int)
            if c in out_te:
                out_te[c] = out_te[c].astype(int)
        elif not pd.api.types.is_numeric_dtype(out_tr[c]):
            codes, uniques = pd.factorize(out_tr[c].astype(str), sort=True)
            out_tr[c] = codes
            if c in out_te:
                lookup = {v: i for i, v in enumerate(uniques)}
                out_te[c] = out_te[c].astype(str).map(lookup).fillna(-1).astype(int)
    return out_tr, out_te


def _proba(est: Any, x: pd.DataFrame) -> np.ndarray | None:
    if not hasattr(est, "predict_proba"):
        return None
    try:
        return np.asarray(est.predict_proba(x))
    except Exception:
        return None


def _run(node: dict, x: pd.DataFrame, x_imp: pd.DataFrame, is_clf: bool):
    kind = node["kind"]
    if kind in ("single", "stacking"):
        xx = x_imp if node.get("impute") else x
        return np.asarray(node["est"].predict(xx)), (_proba(node["est"], xx) if is_clf else None)
    if kind == "threshold":
        _, proba = _run(node["base"], x, x_imp, is_clf)
        if proba is None:
            raise PredictUnavailable("threshold model without probabilities")
        return (proba[:, 1] >= node["threshold"]).astype(int), proba
    if kind == "bag":
        pred, proba = _run(node["base"], x, x_imp, is_clf)
        preds, probas = [np.asarray(pred, dtype=float)], [proba]
        for m in node["members"]:
            preds.append(np.asarray(m.predict(x), dtype=float))
            probas.append(_proba(m, x) if is_clf else None)
        if is_clf:
            return replay.combine_class_votes(preds, probas)
        return np.mean(preds, axis=0), None
    if kind == "average":
        outs = [_run(m, x, x_imp, is_clf) for m in node["members"]]
        w = node["weights"]
        if is_clf and node.get("soft") and all(o[1] is not None for o in outs):
            proba = sum(o[1] * wi for o, wi in zip(outs, w))
            return np.argmax(proba, axis=1), proba
        if is_clf:
            return replay.combine_class_votes([o[0] for o in outs], None, w)[0], None
        return sum(np.asarray(o[0], dtype=float) * wi for o, wi in zip(outs, w)), None
    raise PredictUnavailable(f"unsupported model kind {kind!r}")


def _single_estimator(node: dict):
    """The one fitted estimator behind the prediction, when there is one (for explanations)."""
    if node["kind"] == "single":
        return node["est"], node.get("impute", False)
    if node["kind"] == "threshold":
        return _single_estimator(node["base"])
    return None, False


def _fmt_value(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if not math.isfinite(f) else round(f, 6)
    return v if v is None or isinstance(v, (int, str, bool)) else str(v)


class Predictor:
    def __init__(self, bundle: dict[str, Any]):
        if bundle.get("version") != BUNDLE_VERSION:
            raise PredictUnavailable("model bundle from an incompatible version")
        self.b = bundle
        train = bundle["raw_train"]
        self.target = bundle["target"]
        self.input_columns = [c for c in train.columns if c != self.target
                              and (not bundle["keep_columns"] or c in bundle["keep_columns"])]
        self.label_encoder = None
        if bundle["is_classification"]:
            self.label_encoder = LabelEncoder().fit(train[self.target])

    # ── form schema for the UI ──
    def schema(self, examples: pd.DataFrame | None = None) -> dict[str, Any]:
        train = self.b["raw_train"]
        cols = []
        for c in self.input_columns:
            s = train[c]
            info: dict[str, Any] = {"name": c, "missing_pct": round(float(s.isna().mean() * 100), 1)}
            if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                v = s.dropna()
                is_int = bool(len(v)) and bool(np.all(np.mod(v, 1) == 0))
                info.update(type="number", integer=is_int,
                            min=_fmt_value(v.min()) if len(v) else None, max=_fmt_value(v.max()) if len(v) else None,
                            default=_fmt_value(v.median()) if len(v) else None)
                if is_int and v.nunique() <= 12:
                    info.update(type="category", options=[_fmt_value(x) for x in sorted(v.unique())],
                                default=_fmt_value(v.mode().iloc[0]))
            else:
                vc = s.dropna().astype(str).value_counts()
                info.update(type="category", options=vc.index[:50].tolist(), more=max(0, len(vc) - 50),
                            default=vc.index[0] if len(vc) else None)
            cols.append(info)
        out = {
            "target": self.target, "is_classification": self.b["is_classification"],
            "classes": [str(x) for x in self.label_encoder.classes_] if self.label_encoder is not None else None,
            "best_model": self.b.get("best_model"), "metric": self.b.get("metric"),
            "decision_threshold": self.b.get("decision_threshold"), "columns": cols, "examples": [],
        }
        if examples is not None and len(examples):
            ex = examples.head(8)
            out["examples"] = [
                {"values": {c: _fmt_value(r[c]) for c in self.input_columns if c in ex.columns},
                 "actual": _fmt_value(r[self.target]) if self.target in ex.columns else None}
                for _, r in ex.iterrows()
            ]
        return out

    # ── preprocessing: identical order to the generated pipeline script ──
    def _features(self, rows: pd.DataFrame):
        b = self.b
        train = b["raw_train"].copy()
        test = rows.copy()
        for c in self.input_columns:  # coerce to the training column's type
            if c not in test.columns:
                test[c] = np.nan
            if pd.api.types.is_numeric_dtype(train[c]) and not pd.api.types.is_bool_dtype(train[c]):
                test[c] = pd.to_numeric(test[c], errors="coerce")
            else:
                test[c] = test[c].where(test[c].notna() & (test[c].astype(str) != ""), np.nan)
        test = test[self.input_columns]
        if b["keep_columns"]:
            train = train[[c for c in b["keep_columns"] if c in train.columns]]
            test = test[[c for c in b["keep_columns"] if c in test.columns]]
        raw_train, raw_test = replay.add_raw_features(
            train.drop(columns=[self.target]).reset_index(drop=True),
            test.drop(columns=[self.target], errors="ignore").reset_index(drop=True), b["raw_specs"])
        train, test = replay.apply_deep_clean(train, test.reset_index(drop=True), b["assignments"], self.target,
                                              knn_row_limit=replay.KNN_ROW_LIMIT)
        for spec in b["raw_specs"]:
            train[spec["name"]] = raw_train[spec["name"]].to_numpy()
            test[spec["name"]] = np.asarray(raw_test[spec["name"]], dtype=float)
        train, test = _add_engineered(train, test, b["fe_specs"])
        x_train, x_test = _numeric_like_train(train.drop(columns=[self.target]),
                                              test.drop(columns=[self.target], errors="ignore"))
        imputer = SimpleImputer(strategy="median").fit(x_train)
        x_test_imp = pd.DataFrame(imputer.transform(x_test), columns=x_test.columns)
        x_train_imp = pd.DataFrame(imputer.transform(x_train), columns=x_train.columns)
        fc = b["feature_columns"] or list(x_train.columns)
        missing = [c for c in fc if c not in x_test.columns]
        if missing:
            raise PredictUnavailable(f"could not rebuild model inputs: {missing[:5]}")
        return x_train_imp[fc], x_test[fc], x_test_imp[fc]

    def predict(self, records: list[dict[str, Any]], explain: bool = True) -> list[dict[str, Any]]:
        if not records:
            return []
        rows = pd.DataFrame(records)
        x_train_imp, x, x_imp = self._features(rows)
        is_clf = self.b["is_classification"]
        pred, proba = _run(self.b["model"], x, x_imp, is_clf)
        if self.b["log_target"] and not is_clf:
            pred = np.expm1(np.asarray(pred, dtype=float))
        contrib = self._explain(x_train_imp, x, x_imp, pred, rows) if explain else [None] * len(rows)
        out = []
        for i in range(len(rows)):
            item: dict[str, Any] = {}
            if is_clf:
                item["prediction"] = str(self.label_encoder.inverse_transform([int(pred[i])])[0])
                if proba is not None:
                    item["probabilities"] = {str(c): round(float(p), 4)
                                             for c, p in zip(self.label_encoder.classes_, proba[i])}
            else:
                item["prediction"] = round(float(pred[i]), 6)
            item["inputs"] = {c: _fmt_value(x.iloc[i][c]) for c in x.columns[:40]}
            item["contributions"] = contrib[i]
            out.append(item)
        return out

    # ── per-row explanation ──
    def _explain(self, x_train_imp: pd.DataFrame, x: pd.DataFrame, x_imp: pd.DataFrame, pred: Any,
                 rows: pd.DataFrame) -> list:
        est, impute = _single_estimator(self.b["model"])
        n = len(x)
        if est is None:
            return [None] * n
        xx = x_imp if impute else x
        name = type(est).__name__
        try:
            if any(h in name for h in _TREE_HINTS):
                import shap
                vals = shap.TreeExplainer(est).shap_values(xx)
                if isinstance(vals, list):  # one (rows, features) array per class
                    vals = np.stack(vals, axis=-1)
                vals = np.asarray(vals)
                if vals.ndim == 3:  # (rows, features, classes): explain each row's predicted class
                    cls = np.clip(np.asarray(pred).astype(int), 0, vals.shape[2] - 1)
                    vals = vals[np.arange(n), :, cls]
                method = "SHAP (tree)"
            elif hasattr(est, "coef_"):
                coef = np.asarray(est.coef_, dtype=float)
                dist = x_imp.to_numpy(dtype=float) - x_train_imp.mean().to_numpy(dtype=float)
                if coef.ndim > 1 and coef.shape[0] > 1:  # multiclass: the predicted class's coefficients
                    cls = np.clip(np.asarray(pred).astype(int), 0, coef.shape[0] - 1)
                    vals = dist * coef[cls]
                else:
                    vals = dist * coef.ravel()
                method = "coefficient × distance from average"
            else:
                return [None] * n
        except Exception as exc:
            logger.info("per-row explanation unavailable: %s", exc)
            return [None] * n
        # Model inputs → the column the user typed: "Sex_M"/"Sex_F" (one-hot) and a scaled
        # "Oldpeak" are reported as Sex / Oldpeak with the raw value. Contributions of one
        # column's pieces are summed (SHAP values are additive).
        raw_cols = sorted(self.input_columns, key=len, reverse=True)
        owner = []
        for f in map(str, x.columns):
            owner.append(next((c for c in raw_cols if f == c or f.startswith(f"{c}_")), None))
        out = []
        for i in range(n):
            groups: dict[str, dict[str, Any]] = {}
            for j, f in enumerate(map(str, x.columns)):
                if not np.isfinite(vals[i][j]):
                    continue
                col = owner[j]
                if col is not None:
                    raw = rows.iloc[i].get(col) if col in rows.columns else None
                    key, value = col, (None if raw is None or (isinstance(raw, float) and math.isnan(raw)) or raw == "" else raw)
                else:  # engineered feature: show its computed value
                    key, value = f, x.iloc[i, j]
                g = groups.setdefault(key, {"feature": key, "value": _fmt_value(value), "effect": 0.0})
                g["effect"] += float(vals[i][j])
            items = sorted(groups.values(), key=lambda g: -abs(g["effect"]))[:10]
            for g in items:
                g["effect"] = round(g["effect"], 5)
            out.append({"method": method, "items": items})
        return out

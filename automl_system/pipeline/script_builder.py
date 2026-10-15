"""Phase 7: assemble the final pipeline script deterministically from what actually ran.

Every step records its exact decisions while the pipeline runs (feature-selection
keep list, per-column deep-cleaning assignments, engineered-feature specs, and the
Phase 5 model spec). This module turns those records into a standalone script:

- data transforms are copied verbatim from `replay.py` (the same code the pipeline ran),
- the model is rebuilt from its recorded class, parameters and fit procedure.

No LLM writes code here, so there is nothing to hallucinate. Anything that cannot be
represented raises UnsupportedRecipe and Phase 7 falls back to LLM generation.
"""

from __future__ import annotations

import importlib
import inspect
import math
from typing import Any

from automl_system.memory import episodic
from automl_system.pipeline import replay
from automl_system.pipeline.code_generator import PREDICTIONS_FILE, TEST_FILE, TRAIN_FILE
from automl_system.state.session import SessionState

_REPLAY_FUNCTIONS = (
    replay.default_assignment,
    replay.fit_scaler,
    replay.apply_categorical,
    replay.apply_deep_clean,
    replay.check_formula,
    replay.compute_raw_feature,
    replay.safe_numeric_features,
    replay.combine_class_votes,
    replay.finish_feature,
    replay.run_code_feature,
    replay.add_raw_features,
)

_ES_FIT_HELPER = '''
def fit_model(model, X, y, fit, family, is_classification):
    """Fit exactly as the AutoML run did: on all rows, a capped subsample, or with early stopping."""
    mode = fit.get("mode", "full")
    if mode == "capped":
        idx = np.random.default_rng(fit["seed"]).choice(len(X), size=fit["rows"], replace=False)
        return model.fit(X.iloc[idx], y.iloc[idx])
    if mode != "early_stopping":
        return model.fit(X, y)
    idx = np.random.default_rng(fit["seed"]).choice(len(X), size=min(len(X), fit["rows"]), replace=False)
    X_r, y_r = X.iloc[idx], y.iloc[idx]
    try:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_r, y_r, test_size=fit["val_size"], random_state=fit["split_seed"],
            stratify=y_r if is_classification else None)
    except ValueError:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_r, y_r, test_size=fit["val_size"], random_state=fit["split_seed"])
    if family == "LightGBM":
        import lightgbm as lgb
        return model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                         callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    if family == "CatBoost":
        return model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
    return model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)


def predict_proba_safe(model, X):
    if not IS_CLASSIFICATION or not hasattr(model, "predict_proba"):
        return None
    try:
        return model.predict_proba(X)
    except Exception:
        return None


def add_engineered_features(train, test, specs):
    """Add each accepted engineered feature; fill values were learned on train only."""
    for spec in specs:
        name = spec["name"]
        feat = compute_raw_feature(train, spec)
        if spec.get("fill") is not None:
            feat = feat.fillna(spec["fill"])
        train[name] = feat
        test_fill = spec.get("test_fill")
        try:
            feat_test = compute_raw_feature(test, spec)
            if test_fill is not None:
                feat_test = feat_test.fillna(test_fill)
        except Exception:
            feat_test = test_fill if test_fill is not None else 0.0
        test[name] = feat_test
    return train, test.reindex(columns=train.columns, fill_value=0.0)
'''


class UnsupportedRecipe(Exception):
    """The recorded run cannot be replayed deterministically."""


# ── code-literal helpers ─────────────────────────────────────────────────────

def _lit(v: Any) -> str:
    """Python source for a plain value (handles nan/inf, which repr() renders invalidly)."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return f'float("{v}")'
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_lit(k)}: {_lit(x)}" for k, x in v.items()) + "}"
    if isinstance(v, list):
        return "[" + ", ".join(_lit(x) for x in v) + "]"
    if isinstance(v, tuple):
        return "(" + ", ".join(_lit(x) for x in v) + ("," if len(v) == 1 else "") + ")"
    return repr(v)


def _public_import(module: str, cls: str) -> str:
    """Shortest public module path that exposes `cls` (sklearn.ensemble._forest → sklearn.ensemble)."""
    parts = module.split(".")
    candidates = [parts[0]] + [".".join(parts[:i]) for i in range(2, len(parts) + 1)]
    for cand in candidates:
        if any(p.startswith("_") for p in cand.split(".")):
            continue
        try:
            if getattr(importlib.import_module(cand), cls, None) is not None:
                return cand
        except ImportError:
            continue
    return module


class _Builder:
    def __init__(self, session: SessionState):
        self.session = session
        self.imports: set[str] = set()
        self.counter = 0
        entries = episodic.get_log(session.decisions_log_id) if session.decisions_log_id else []
        self.decisions = entries or list(session.local_decisions_log or [])

    # DecLog numbering matches code_generator (1-based position in the full log).
    def cite(self, *actions: str) -> str:
        for i in range(len(self.decisions) - 1, -1, -1):
            action = str(self.decisions[i].get("action") or "")
            if any(action == a or action.startswith(a) for a in actions):
                why = str(self.decisions[i].get("rationale") or action).replace("\n", " ")[:110]
                return f"# DecLog Entry{i + 1}: {why}"
        return ""

    def ctor(self, spec: dict[str, Any] | None, extra: dict[str, Any] | None = None) -> str:
        if not spec:
            raise UnsupportedRecipe("model step without a recorded estimator spec")
        if spec.get("special") == "scaled_ridge":
            self.imports.update({"from sklearn.pipeline import Pipeline",
                                 "from sklearn.linear_model import Ridge"})
            return f'Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha={_lit(spec["alpha"])}))])'
        cls = spec["class"]
        self.imports.add(f"from {_public_import(spec['module'], cls)} import {cls}")
        params = {**spec.get("params", {}), **(extra or {})}
        return f"{cls}(**{_lit(params)})"

    # Each model kind emits lines that define `<p>_pred` and `<p>_proba`.
    def model(self, spec: dict[str, Any] | None) -> tuple[list[str], str]:
        if not spec:
            raise UnsupportedRecipe("best model has no recorded spec")
        self.counter += 1
        p = f"m{self.counter}"
        kind = spec.get("kind")

        if kind == "single":
            x_fit, x_pred = ("X_train_imp", "X_test_imp") if spec.get("impute_median") else ("X_train", "X_test")
            family = spec.get("family")
            return [
                f"{p}_model = {self.ctor(spec.get('estimator'))}  {self.cite(f'model_trained:{family}')}",
                f"fit_model({p}_model, {x_fit}, y_fit, {_lit(spec.get('fit') or {'mode': 'full'})}, "
                f"{family!r}, IS_CLASSIFICATION)",
                f"{p}_pred = {p}_model.predict({x_pred})",
                f"{p}_proba = predict_proba_safe({p}_model, {x_pred})",
            ], p

        if kind == "threshold":
            lines, b = self.model(spec.get("base"))
            return lines + [
                f"{self.cite('threshold_optimisation')}",
                f"{p}_proba = {b}_proba",
                f"{p}_pred = ({b}_proba[:, 1] >= {_lit(spec['threshold'])}).astype(int)",
            ], p

        if kind == "bag":
            lines, b = self.model(spec.get("base"))
            member = dict(spec.get("member") or {})
            if not member:
                raise UnsupportedRecipe("bagging member spec missing")
            seed_key = spec["seed_key"]
            member = {**member, "params": {k: v for k, v in member.get("params", {}).items() if k != seed_key}}
            return lines + [
                f"{self.cite('seed_bagging')}",
                f"{p}_preds = [np.asarray({b}_pred, dtype=float)]",
                f"{p}_probas = [{b}_proba]",
                f"for seed in {_lit(list(spec.get('seeds') or []))}:",
                f"    bag_model = {self.ctor(member)}",
                f"    bag_model.set_params(**{{{seed_key!r}: seed}})",
                "    bag_model.fit(X_train, y_fit)",
                f"    {p}_preds.append(np.asarray(bag_model.predict(X_test), dtype=float))",
                f"    {p}_probas.append(predict_proba_safe(bag_model, X_test))",
                f"if IS_CLASSIFICATION:  # soft vote of the seed models (majority vote without probabilities)",
                f"    {p}_pred, {p}_proba = combine_class_votes({p}_preds, {p}_probas)",
                f"else:",
                f"    {p}_pred, {p}_proba = np.mean({p}_preds, axis=0), None",
            ], p

        if kind == "average":
            members = spec.get("members") or []
            if len(members) < 2 or any(m is None for m in members):
                raise UnsupportedRecipe("averaging ensemble with unrecorded members")
            lines: list[str] = [self.cite("voting_ensemble")]
            names = []
            for m in members:
                m_lines, m_p = self.model(m)
                lines += m_lines
                names.append(m_p)
            w = spec["weights"]
            preds = " + ".join(f"np.asarray({n}_pred, dtype=float) * {_lit(x)}" for n, x in zip(names, w))
            if spec.get("soft"):
                probas = " + ".join(f"{n}_proba * {_lit(x)}" for n, x in zip(names, w))
                return lines + [f"{p}_proba = {probas}", f"{p}_pred = np.argmax({p}_proba, axis=1)"], p
            return lines + [
                f"if IS_CLASSIFICATION:  # weighted majority vote of the members",
                f"    {p}_pred = combine_class_votes([{', '.join(f'{n}_pred' for n in names)}], None, {_lit(w)})[0]",
                f"else:",
                f"    {p}_pred = {preds}",
                f"{p}_proba = None",
            ], p

        if kind == "stacking":
            members = spec.get("members") or []
            if len(members) < 2:
                raise UnsupportedRecipe("stacking ensemble with fewer than 2 members")
            est_list = ", ".join(f"({n!r}, {self.ctor(m)})" for n, m in members)
            self.imports.add("from sklearn.ensemble import StackingClassifier, StackingRegressor")
            return [
                f"{self.cite('stacking_ensemble')}",
                f"{p}_cls = StackingClassifier if IS_CLASSIFICATION else StackingRegressor",
                f"{p}_model = {p}_cls(estimators=[{est_list}],",
                f"    final_estimator={self.ctor(spec.get('final_estimator'))}, cv={int(spec['cv'])})",
                f"{p}_model.fit(X_train, y_fit)",
                f"{p}_pred = {p}_model.predict(X_test)",
                f"{p}_proba = predict_proba_safe({p}_model, X_test)",
            ], p

        raise UnsupportedRecipe(f"unknown model kind {kind!r}")


def assemble_pipeline_script(session: SessionState) -> str:
    """Build the standalone script from the session's recorded decisions.

    Raises UnsupportedRecipe when the run cannot be replayed exactly.
    """
    task = session.task_config or {}
    results = session.execution_results or {}
    rp = results.get("replay")
    if not rp or not rp.get("model"):
        raise UnsupportedRecipe("no replay record in execution results")
    target = str(task.get("target_column") or "")
    if not target:
        raise UnsupportedRecipe("no target column")
    is_clf = str(results.get("task_type") or "") == "classification"

    b = _Builder(session)
    model_lines, final = b.model(rp["model"])

    keep = list((session.preprocessing_meta or {}).get("feature_selection_columns") or [])
    assignments = {
        c: a for c, a in (session.deep_cleaning_assignments or {}).items() if isinstance(a, dict)
    }
    all_specs = [
        e["spec"] for e in (session.feature_engineering_log or [])
        if e.get("status") == "added" and isinstance(e.get("spec"), dict)
    ]
    # "raw": computed from original-scale columns (CV-gated FE); else from deep-cleaned columns
    raw_specs = [sp for sp in all_specs if sp.get("source") == "raw"]
    fe_specs = [sp for sp in all_specs if sp.get("source") != "raw"]
    added_without_spec = [
        e.get("feature") for e in (session.feature_engineering_log or [])
        if e.get("status") == "added" and not isinstance(e.get("spec"), dict)
    ]
    if added_without_spec:
        raise UnsupportedRecipe(f"engineered features without a recorded spec: {added_without_spec}")

    assign_lines = "\n".join(f"    {c!r}: {_lit(a)}," for c, a in assignments.items())
    fe_lines = "\n".join(f"    {_lit(sp)}," for sp in fe_specs)
    raw_fe_lines = "\n".join(f"    {_lit(sp)}," for sp in raw_specs)
    funcs = "\n\n".join(inspect.getsource(f) for f in _REPLAY_FUNCTIONS)
    if rp.get("smote"):
        b.imports.add("from imblearn.over_sampling import SMOTE")
    extra_imports = "\n".join(sorted(b.imports))
    best = results.get("best_model_name") or "best model"
    metric = results.get("metric_display") or results.get("metric") or ""

    smote_block = ""
    if rp.get("smote"):
        sm = rp["smote"]
        smote_block = f"""
# Oversample the minority class on TRAIN rows only  {b.cite('smote_applied')}
X_train, y_fit = SMOTE(random_state={sm['random_state']}, k_neighbors={sm['k_neighbors']}).fit_resample(X_train, y_fit)
X_train = pd.DataFrame(X_train, columns=list(X_test.columns))
y_fit = pd.Series(y_fit, name=TARGET_COLUMN)
"""
    log_block = ""
    if rp.get("log_target"):
        log_block = f"""
# Train on log1p(target); predictions are converted back with expm1  {b.cite('log_target_transform')}
y_fit = np.log1p(y_fit)
"""

    return f'''"""
AutoML pipeline — {best} for target {target!r} ({metric}).

Generated from the decisions this AutoML run actually executed:
the data transforms below are the same functions the pipeline ran, and the
model is rebuilt from its recorded class, hyperparameters and fit procedure.

Inputs : TRAIN_PATH / TEST_PATH — the run's train/test split (after basic cleaning:
         duplicate removal and numeric type coercion). To use new data, give it the
         same columns as the training file.
Output : PREDICTIONS_PATH — one prediction per TEST row, in original target units.
"""

import warnings

{replay.REPLAY_IMPORTS}
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
{extra_imports}

warnings.filterwarnings("ignore")

TRAIN_PATH = {TRAIN_FILE!r}
TEST_PATH = {TEST_FILE!r}
TARGET_COLUMN = {target!r}
PREDICTIONS_PATH = {PREDICTIONS_FILE!r}
IS_CLASSIFICATION = {is_clf!r}

# ── Decisions recorded during the run ───────────────────────────────────────
# Columns kept by feature selection (empty = keep all)  {b.cite('selection_complete')}
KEEP_COLUMNS = {_lit(keep)}

# Per-column imputation / outlier / encoding / scaling  {b.cite('execution_complete', 'all_assignments_logged')}
ASSIGNMENTS = {{
{assign_lines}
}}

# Engineered features that improved cross-validated score (original-scale columns)  {b.cite('complete')}
RAW_FEATURES = [
{raw_fe_lines}
]

# Engineered features computed on the cleaned/encoded columns
ENGINEERED_FEATURES = [
{fe_lines}
]

# Final model inputs (after the mutual-information filter)  {b.cite('mi_feature_filter')}
FEATURE_COLUMNS = {_lit(list(rp.get('feature_columns') or []))}
KNN_ROW_LIMIT = {replay.KNN_ROW_LIMIT}
_FORMULA_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Constant, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd,
)


# ── Data transforms (identical to the AutoML pipeline's own code) ───────────

{funcs}

# ── Model helpers ───────────────────────────────────────────────────────────
{_ES_FIT_HELPER}

# ── 1. Load the train/test split ────────────────────────────────────────────
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
y_test = test[TARGET_COLUMN] if TARGET_COLUMN in test.columns else None

# ── 2. Preprocessing (fit on TRAIN only, applied unchanged to TEST) ─────────
if KEEP_COLUMNS:
    train = train[[c for c in KEEP_COLUMNS if c in train.columns]]
    test = test[[c for c in KEEP_COLUMNS if c in test.columns]]
raw_train, raw_test = add_raw_features(
    train.drop(columns=[TARGET_COLUMN]).reset_index(drop=True),
    test.drop(columns=[TARGET_COLUMN], errors="ignore").reset_index(drop=True),
    RAW_FEATURES,
)
train, test = apply_deep_clean(train, test, ASSIGNMENTS, TARGET_COLUMN, knn_row_limit=KNN_ROW_LIMIT)
for spec in RAW_FEATURES:  # appended after cleaning, exactly as the pipeline did
    train[spec["name"]] = raw_train[spec["name"]].to_numpy()
    test[spec["name"]] = np.asarray(raw_test[spec["name"]], dtype=float)
train, test = add_engineered_features(train, test, ENGINEERED_FEATURES)

X_train = safe_numeric_features(train.drop(columns=[TARGET_COLUMN]))
X_test = safe_numeric_features(test.drop(columns=[TARGET_COLUMN], errors="ignore"))
y_fit = train[TARGET_COLUMN]
label_encoder = None
if IS_CLASSIFICATION:
    label_encoder = LabelEncoder()
    y_fit = pd.Series(label_encoder.fit_transform(y_fit), index=y_fit.index)
{smote_block}{log_block}
# Median-imputed copies for models that cannot handle NaN
imputer = SimpleImputer(strategy="median").fit(X_train)
X_train_imp = pd.DataFrame(imputer.transform(X_train), columns=X_train.columns)
X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

X_train, X_test = X_train[FEATURE_COLUMNS], X_test[FEATURE_COLUMNS]
X_train_imp, X_test_imp = X_train_imp[FEATURE_COLUMNS], X_test_imp[FEATURE_COLUMNS]

# ── 3. Train the final model and predict ────────────────────────────────────
{chr(10).join(line for line in model_lines if line.strip())}

pred = {final}_pred
proba = {final}_proba
if {bool(rp.get('log_target'))!r}:
    pred = np.expm1(pred)

# ── 4. Write predictions (original target labels / units) ───────────────────
out = pd.DataFrame()
if IS_CLASSIFICATION:
    out["prediction"] = label_encoder.inverse_transform(np.asarray(pred).astype(int))
    if proba is not None:
        for i in range(proba.shape[1]):
            out[f"proba_{{i}}"] = proba[:, i]
else:
    out["prediction"] = np.asarray(pred, dtype=float)
out.to_csv(PREDICTIONS_PATH, index=False)

if y_test is not None:
    if IS_CLASSIFICATION:
        from sklearn.metrics import accuracy_score, f1_score
        print("test accuracy:", round(accuracy_score(y_test.astype(str), out["prediction"].astype(str)), 4))
        print("test F1 (weighted):", round(f1_score(y_test.astype(str), out["prediction"].astype(str), average="weighted"), 4))
    else:
        from sklearn.metrics import mean_absolute_error, r2_score
        print("test R2:", round(r2_score(y_test, out["prediction"]), 4))
        print("test MAE:", round(mean_absolute_error(y_test, out["prediction"]), 4))
'''

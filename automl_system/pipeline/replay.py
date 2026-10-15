"""Pure data transforms shared by the pipeline and the generated pipeline script.

The pipeline calls these functions while it runs; Phase 7 copies their exact source
(inspect.getsource) into the downloadable script, so the script replays the same
code that produced the reported score instead of an LLM re-implementation.

Rules for this module: functions use only the imports listed in REPLAY_IMPORTS and
each other — no session, config, logging or LLM access.
"""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd
from category_encoders import LeaveOneOutEncoder, OrdinalEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

# Import block the generated script needs for the functions below.
REPLAY_IMPORTS = """import ast

import numpy as np
import pandas as pd
from category_encoders import LeaveOneOutEncoder, OrdinalEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler"""

KNN_ROW_LIMIT = 5_000

_FORMULA_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Constant, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd,
)


def default_assignment(col, s):
    if pd.api.types.is_numeric_dtype(s):
        return {"imputation": "median", "outlier": "leave", "encoding": "none", "scaling": "standard"}
    return {"imputation": "mode", "outlier": "leave", "encoding": "one_hot", "scaling": "none"}


def fit_scaler(sc):
    if sc == "standard":
        return StandardScaler()
    if sc == "minmax":
        return MinMaxScaler()
    if sc == "robust":
        return RobustScaler()
    return None


def apply_categorical(col, df, assignment, target, df_test=None):
    """Fits the assigned categorical strategy on `df` (train) and, when
    `df_test` is given, applies the identical fitted transform (same mode
    fill value / encoder / one-hot vocabulary) to it — never refits on test,
    which would leak test-set category/target information into the encoding."""
    s = df[col].astype(str).replace("nan", np.nan)
    s_test = df_test[col].astype(str).replace("nan", np.nan) if df_test is not None and col in df_test.columns else None

    imp = assignment.get("imputation", "mode")
    if imp == "mode":
        mode = s.mode()
        fill = mode.iloc[0] if len(mode) else "missing"
    else:
        fill = "missing"
    s = s.fillna(fill)
    if s_test is not None:
        s_test = s_test.fillna(fill)

    enc = assignment.get("encoding", "one_hot")

    if enc == "binary":
        unique_vals = s.dropna().unique()
        if len(unique_vals) == 2:
            train_out = pd.DataFrame({col: (s == unique_vals[0]).astype(float)}, index=df.index)
            test_out = (
                pd.DataFrame({col: (s_test == unique_vals[0]).astype(float)}, index=s_test.index)
                if s_test is not None else None
            )
            return train_out, test_out
        # Fall through to one_hot if not truly binary

    if enc == "ordinal":
        try:
            oe = OrdinalEncoder(handle_unknown="value", handle_missing="value")
            v = oe.fit_transform(s.to_frame(name=col))
            train_out = pd.DataFrame({col: v[col].astype(float)}, index=df.index)
            test_out = None
            if s_test is not None:
                vt = oe.transform(s_test.to_frame(name=col))
                test_out = pd.DataFrame({col: vt[col].astype(float)}, index=s_test.index)
            return train_out, test_out
        except Exception:
            codes, uniques = pd.factorize(s, sort=True)
            train_out = pd.DataFrame({col: codes.astype(float)}, index=df.index)
            test_out = None
            if s_test is not None:
                mapping = {v: i for i, v in enumerate(uniques)}
                test_codes = s_test.map(mapping).astype(float)
                test_out = pd.DataFrame({col: test_codes.fillna(-1.0)}, index=s_test.index)
            return train_out, test_out

    if enc == "target" and target and target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
        # LeaveOneOutEncoder: each row's encoding excludes its own target value,
        # avoiding the direct leakage that TargetEncoder.fit_transform causes on full data.
        # Fit on train only; test rows use .transform() (the library's standard,
        # non-leaking way to score held-out data — no y needed at transform time).
        te = LeaveOneOutEncoder(sigma=0.0)
        z = te.fit_transform(s.to_frame(), df[target])
        train_out = pd.DataFrame({f"{col}_te": z.values.ravel()}, index=df.index)
        test_out = None
        if s_test is not None:
            zt = te.transform(s_test.to_frame())
            test_out = pd.DataFrame({f"{col}_te": zt.values.ravel()}, index=s_test.index)
        return train_out, test_out

    train_dummies = pd.get_dummies(s, prefix=col, dtype=float)
    test_out = None
    if s_test is not None:
        test_dummies = pd.get_dummies(s_test, prefix=col, dtype=float)
        test_out = test_dummies.reindex(columns=train_dummies.columns, fill_value=0.0)
    return train_dummies, test_out


def apply_deep_clean(df, df_test, assignments, target, knn_row_limit=KNN_ROW_LIMIT):
    """Impute → outlier handling → encode → scale, per column assignment.

    Every stateful transform (imputed fill values, IQR bounds, scaler parameters,
    encoder mappings) is fit on `df` (train) only and applied unchanged to `df_test`.
    Returns (out_train, out_test) — out_test is None when df_test is None.
    """
    target_s = str(target) if target else None
    if target_s and target_s not in df.columns:
        target_s = None

    work = df.copy()
    work_test = df_test.copy() if df_test is not None else None

    knn_cols = [
        c for c, a in assignments.items()
        if c in work.columns
        and a.get("imputation") == "knn"
        and pd.api.types.is_numeric_dtype(work[c])
        and len(work) <= knn_row_limit
    ]
    if knn_cols:
        n_neighbors = max(2, min(5, len(work) // 10 or 2))
        imputer = KNNImputer(n_neighbors=n_neighbors)
        work[knn_cols] = imputer.fit_transform(work[knn_cols])
        if work_test is not None:
            work_test[knn_cols] = imputer.transform(work_test[knn_cols])

    new_parts = []
    new_parts_test = []
    if target_s and target_s in work.columns:
        new_parts.append(work[[target_s]].copy())
    if work_test is not None and target_s and target_s in work_test.columns:
        new_parts_test.append(work_test[[target_s]].copy())

    for col in work.columns:
        if target_s and str(col) == target_s:
            continue
        a = assignments.get(col, default_assignment(col, work[col]))
        s = work[col]
        s_test = work_test[col] if work_test is not None and col in work_test.columns else None

        if pd.api.types.is_numeric_dtype(s):
            imp = a.get("imputation", "median")
            num_test = None
            if col in knn_cols:
                num = s.astype(float)
                if s_test is not None:
                    num_test = s_test.astype(float)
            elif imp == "mean":
                imputer = SimpleImputer(strategy="mean")
                num = pd.Series(imputer.fit_transform(s.values.reshape(-1, 1)).ravel(), index=s.index)
                if s_test is not None:
                    num_test = pd.Series(
                        imputer.transform(s_test.values.reshape(-1, 1)).ravel(), index=s_test.index)
            elif imp == "flag_as_missing":
                fill = float(s.median())
                flag_col = f"{col}__was_null"
                work[flag_col] = s.isna().astype(float)
                num = pd.Series(s.fillna(fill), index=s.index).astype(float)
                new_parts.append(pd.DataFrame({flag_col: work[flag_col]}))
                if s_test is not None:
                    work_test[flag_col] = s_test.isna().astype(float)
                    num_test = pd.Series(s_test.fillna(fill), index=s_test.index).astype(float)
                    new_parts_test.append(pd.DataFrame({flag_col: work_test[flag_col]}))
            else:
                imputer = SimpleImputer(strategy="median")
                num = pd.Series(imputer.fit_transform(s.values.reshape(-1, 1)).ravel(), index=s.index)
                if s_test is not None:
                    num_test = pd.Series(
                        imputer.transform(s_test.values.reshape(-1, 1)).ravel(), index=s_test.index)

            if not isinstance(num, pd.Series):
                num = pd.Series(num.ravel(), index=s.index)
            num = num.astype(float)
            if num_test is not None:
                if not isinstance(num_test, pd.Series):
                    num_test = pd.Series(num_test.ravel(), index=s_test.index)
                num_test = num_test.astype(float)

            o = a.get("outlier", "leave")
            if o == "iqr_cap":
                q1, q3 = num.quantile(0.25), num.quantile(0.75)
                iqr = q3 - q1
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                num = num.clip(lo, hi)
                if num_test is not None:
                    num_test = num_test.clip(lo, hi)
            elif o == "log":
                mn = float(num.min())
                shift = 1 - mn if mn <= 0 else 0
                num = pd.Series(np.log1p(num + shift), index=num.index)
                if num_test is not None:
                    num_test = pd.Series(np.log1p(num_test + shift), index=num_test.index)

            scaler = fit_scaler(a.get("scaling", "none"))
            if scaler is not None:
                num = pd.Series(scaler.fit_transform(num.values.reshape(-1, 1)).ravel(), index=num.index)
                if num_test is not None:
                    num_test = pd.Series(
                        scaler.transform(num_test.values.reshape(-1, 1)).ravel(), index=num_test.index)

            new_parts.append(pd.DataFrame({col: num}))
            if num_test is not None:
                new_parts_test.append(pd.DataFrame({col: num_test}))
        else:
            cat_train, cat_test = apply_categorical(col, work, a, target_s, work_test)
            new_parts.append(cat_train)
            if cat_test is not None:
                new_parts_test.append(cat_test)

    out = pd.concat(new_parts, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]

    out_test = None
    if work_test is not None:
        out_test = (
            pd.concat(new_parts_test, axis=1) if new_parts_test
            else pd.DataFrame(index=work_test.index)
        )
        out_test = out_test.loc[:, ~out_test.columns.duplicated()]
        # Defensive: guarantee identical column set/order to train even if some
        # edge case (e.g. a test-only-missing column) produced a gap.
        out_test = out_test.reindex(columns=out.columns, fill_value=0.0)
    return out, out_test


def check_formula(formula):
    """Allow only arithmetic on column names and numbers — no calls, attributes or indexing."""
    tree = ast.parse(formula, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _FORMULA_NODES):
            raise ValueError(f"disallowed syntax in formula {formula!r}: {type(node).__name__}")
    return tree


def compute_raw_feature(df, spec):
    """Compute one engineered feature (before NaN filling) from a spec:
    {"type": "ratio"|"interaction"|"domain_combination"|"datetime",
     "left", "right", "formula", "part" (datetime only)}."""
    typ = spec.get("type")
    left, right = spec.get("left"), spec.get("right")
    if typ == "datetime":
        s = pd.to_datetime(df[left], errors="coerce")
        return pd.to_numeric(getattr(s.dt, spec["part"]), errors="coerce")
    if typ == "domain_combination":
        # Only numeric columns are visible — string arithmetic would silently concatenate.
        local_ns = {c: df[c] for c in df.columns if pd.api.types.is_numeric_dtype(df[c])}
        code = compile(check_formula(spec["formula"]), "<formula>", "eval")
        feat = eval(code, {"__builtins__": {}}, local_ns)  # noqa: S307 — AST-whitelisted
        return pd.Series(feat, index=df.index)
    a = pd.to_numeric(df[left], errors="coerce")
    b = pd.to_numeric(df[right], errors="coerce")
    if typ == "ratio":
        return a / (b.replace(0, np.nan) + 1e-12)
    if typ == "interaction":
        return a * b
    raise ValueError(f"unknown feature type: {typ!r}")


def finish_feature(values, fill):
    """Numeric, inf → NaN, then fill NaN with the value learned on train (None = leave)."""
    feat = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan)
    return feat.fillna(fill) if fill is not None else feat


def run_code_feature(train, test, code):
    """Run an LLM-written `make_feature(train, test)` and normalize its output.

    Executed only inside the Docker sandbox (during the run) and inside the
    generated script — never in the AutoML server process. The function sees
    feature columns only (no target); it learns from `train` and applies to `test`.
    Returns two numeric DataFrames with identical string column names.
    """
    import math

    namespace = {"pd": pd, "np": np, "math": math}
    exec(compile(code, "<make_feature>", "exec"), namespace)  # noqa: S102 — sandboxed
    out_train, out_test = namespace["make_feature"](train.copy(), test.copy())
    frames = []
    for out, ref in ((out_train, train), (out_test, test)):
        frame = out.to_frame() if isinstance(out, pd.Series) else pd.DataFrame(out)
        if len(frame) != len(ref):
            raise ValueError(f"make_feature returned {len(frame)} rows, expected {len(ref)}")
        frame = frame.reset_index(drop=True)
        frame.columns = [str(c) for c in frame.columns]
        frames.append(frame.apply(pd.to_numeric, errors="coerce"))
    if list(frames[0].columns) != list(frames[1].columns):
        raise ValueError("make_feature returned different columns for train and test")
    return frames[0], frames[1]


def add_raw_features(raw_train, raw_test, specs):
    """Compute CV-accepted features on original-scale columns (train fills reused on test).
    Each feature is added to the frames so later formulas can build on it."""
    for spec in specs:
        name = spec["name"]
        if spec.get("type") == "code":
            # Feature functions see only the original columns they were tested on.
            cols = spec["input_columns"]
            test_in = raw_test[cols] if raw_test is not None else raw_train[cols].iloc[:0]
            out_tr, out_te = run_code_feature(raw_train[cols], test_in, spec["code"])
            raw_train[name] = finish_feature(out_tr[spec["column"]], spec.get("fill")).to_numpy()
            if raw_test is not None:
                raw_test[name] = finish_feature(out_te[spec["column"]], spec.get("test_fill")).to_numpy()
            continue
        raw_train[name] = finish_feature(compute_raw_feature(raw_train, spec), spec.get("fill")).to_numpy()
        if raw_test is not None:
            test_fill = spec.get("test_fill")
            try:
                ft = finish_feature(compute_raw_feature(raw_test, spec), test_fill).to_numpy()
            except Exception:
                ft = test_fill if test_fill is not None else 0.0
            raw_test[name] = ft
    return raw_train, raw_test


def combine_class_votes(preds, probas, weights=None):
    """Combine several classifiers' outputs on label-encoded classes (0..k-1).

    Soft vote when every model has probabilities (weighted average → most likely
    class); otherwise a (weighted) majority vote of the predicted classes.
    Never averages class numbers — for 3+ classes that would invent labels
    (e.g. classes 0 and 2 averaging to 1). Returns (pred, proba or None).
    """
    w = np.ones(len(preds)) if weights is None else np.asarray(weights, dtype=float)
    if probas and len(probas) == len(preds) and all(p is not None for p in probas):
        proba = sum(p * wi for p, wi in zip(probas, w)) / w.sum()
        return np.argmax(proba, axis=1), proba
    votes = np.vstack([np.asarray(p).astype(int) for p in preds])
    n_classes = int(votes.max()) + 1
    pred = np.array([np.bincount(col, weights=w, minlength=n_classes).argmax() for col in votes.T])
    return pred, None


def safe_numeric_features(x):
    """Last-resort numericization: bools → int, any leftover non-numeric column → codes."""
    out = x.copy()
    for c in out.columns:
        if pd.api.types.is_bool_dtype(out[c]):
            out[c] = out[c].astype(int)
        elif not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.factorize(out[c].astype(str), sort=True)[0]
    return out

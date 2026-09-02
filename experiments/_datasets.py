"""
Shared dataset registry for all experiment scripts.

Single source of truth for the dataset specs used by:
  - benchmark.py
  - ablation_memory.py
  - elimination_accuracy.py
  - extract_case_study.py

Kaggle CSVs go in experiments/datasets/ with the filenames listed below.
diamonds is auto-loaded from seaborn if the CSV is absent.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATASETS_DIR = Path(__file__).parent / "datasets"
RESULTS_DIR = Path(__file__).parent / "results"

# ── Dataset registry ──────────────────────────────────────────────────────────
KAGGLE_DATASETS: dict[str, dict] = {

    # ── Classification (5) ────────────────────────────────────────────────────

    "titanic": {
        "file": "titanic.csv",
        "target": "Survived",
        "task_type": "classification",
        "drop_cols": ["PassengerId", "Name", "Ticket", "Cabin"],
        "user_goal": (
            "This is passenger data from the Titanic. "
            "Predict which passengers survived."
        ),
    },

    "heart": {
        "file": "heart.csv",
        "target": "HeartDisease",
        "task_type": "classification",
        "drop_cols": [],
        "user_goal": (
            "I have patient records from a cardiology clinic. Predict which patients "
            "are at risk of heart disease. Prioritise catching every real case, "
            "even at the cost of false alarms."
        ),
    },

    "churn": {
        "file": "churn.csv",
        "target": "Churn",
        "task_type": "classification",
        "drop_cols": ["customerID"],
        "user_goal": (
            "I work at a telecom company and we're losing customers. "
            "Predict which ones are about to leave so we can try to keep them."
        ),
    },

    "bank": {
        "file": "bank.csv",
        "target": "deposit",
        "task_type": "classification",
        "drop_cols": [],
        "user_goal": (
            "This is data from our bank's phone sales campaign. "
            "Predict which customers are likely to open a term deposit."
        ),
    },

    "creditcard_fraud": {
        "file": "creditcard.csv",
        "target": "Class",
        "task_type": "classification",
        "drop_cols": ["Time"],
        "max_rows": 50_000,             # full 284k rows -> 2-3h HPO; 50k preserves imbalance challenge
        "user_goal": (
            "I have credit card transaction records. Predict which ones are fraudulent. "
            "Most transactions are fine but we can't afford to miss real fraud."
        ),
    },

    # ── Regression (5) ────────────────────────────────────────────────────────

    "house_prices": {
        "file": "house_prices.csv",
        "target": "price",
        "task_type": "regression",
        "drop_cols": ["id", "date"],
        "user_goal": (
            "I have home sales data from King County, Washington. "
            "Predict the selling price of each house."
        ),
    },

    "insurance": {
        "file": "insurance.csv",
        "target": "charges",
        "task_type": "regression",
        "drop_cols": [],
        "user_goal": (
            "I have customer data from a health insurance company. "
            "Predict how much each customer will cost in medical bills."
        ),
    },

    "diamonds": {
        "file": "diamonds.csv",         # auto-fetched from seaborn if absent
        "target": "price",
        "task_type": "regression",
        "drop_cols": [],
        "user_goal": (
            "I have a dataset of diamond listings with carat, cut, color, and clarity. "
            "Predict what each diamond should sell for."
        ),
    },

    "life_expectancy": {
        "file": "life_expectancy.csv",
        "target": "Life expectancy",
        "task_type": "regression",
        "drop_cols": ["Country"],
        "user_goal": (
            "I have WHO health and development data covering countries over 15 years. "
            "Predict life expectancy."
        ),
    },

    "wine": {
        "file": "wine.csv",
        "target": "quality",
        "task_type": "regression",
        "drop_cols": [],
        "user_goal": (
            "I have lab measurements for red wine samples including acidity, sugar and alcohol. "
            "Predict wine quality from its chemical measurements."
        ),
    },
}


def metric_for(task_type: str) -> str:
    """Canonical optimisation/report metric forced across all systems (for comparability)."""
    return "f1_weighted" if task_type == "classification" else "r2"


def metric_label_for(task_type: str) -> str:
    return "F1 (weighted)" if task_type == "classification" else "R²"


def list_datasets() -> list[str]:
    return list(KAGGLE_DATASETS)


def _make_diamonds_csv(path: Path) -> None:
    """Auto-fetch diamonds dataset from seaborn's built-in cache if CSV is absent."""
    import seaborn as sns
    df = sns.load_dataset("diamonds")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  [datasets] Generated diamonds.csv ({len(df):,} rows) from seaborn.")


def load_raw(name: str) -> tuple[pd.DataFrame, str, str] | None:
    """
    Load a dataset and return (df_with_target_named 'target', task_type, metric).

    Target column is renamed to 'target'. Categorical features are left as-is.
    Binary/multiclass string targets are label-encoded.
    Returns None if the dataset is missing and cannot be auto-generated.
    """
    spec = KAGGLE_DATASETS.get(name)
    if spec is None:
        return None

    csv_path = DATASETS_DIR / spec["file"]

    # Auto-generate diamonds from seaborn if CSV is absent
    if not csv_path.exists() and name == "diamonds":
        try:
            _make_diamonds_csv(csv_path)
        except Exception as e:
            print(f"  WARNING: could not generate diamonds.csv -- {e}")
            return None

    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    target = spec["target"].strip()
    if target not in df.columns:
        print(f"  WARNING: target '{target}' not in {spec['file']} -- skipping {name}")
        return None

    # Remove exact duplicate rows — mirrors the pipeline's own Phase 2 cleaning step
    # (automl_system/pipeline/cleaning.py) so baselines/AutoGluon/FLAML are evaluated
    # on the same deduplicated data our system trains and tests on. Without this,
    # duplicate rows can land on both sides of a train/test split and leak — e.g.
    # wine.csv has 240 exact duplicates (15% of rows), which was inflating the
    # untuned RandomForest baseline's test score via memorized rows.
    n_dup = int(df.duplicated().sum())
    if n_dup:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"  [datasets] {name}: removed {n_dup} exact duplicate row(s) ({100*n_dup/(len(df)+n_dup):.1f}%).")

    drop = [c for c in spec.get("drop_cols", []) if c in df.columns]
    df = df.drop(columns=drop)
    df = df.dropna(subset=[target])

    # Cap very large datasets to keep HPO tractable.
    # creditcard_fraud (284k rows) would take 2-3 hours with full HPO --
    # 50k rows still captures the class imbalance challenge at manageable cost.
    _MAX_ROWS = spec.get("max_rows")
    if _MAX_ROWS and len(df) > _MAX_ROWS:
        import numpy as np
        original_len = len(df)
        rng = np.random.default_rng(42)
        idx = rng.choice(original_len, size=_MAX_ROWS, replace=False)
        df = df.iloc[idx].reset_index(drop=True)
        print(f"  [datasets] {name}: sampled {_MAX_ROWS:,} rows from {original_len:,} (max_rows cap).")

    task_type = spec["task_type"]
    y = df[target]
    if task_type == "classification" and y.dtype == object:
        uniq = sorted(y.dropna().unique())
        if len(uniq) == 2:
            y = y.map({uniq[0]: 0, uniq[1]: 1})
        else:
            from sklearn.preprocessing import LabelEncoder
            y = pd.Series(LabelEncoder().fit_transform(y), index=df.index)

    df = df.drop(columns=[target])
    df["target"] = y.values
    return df, task_type, metric_for(task_type)

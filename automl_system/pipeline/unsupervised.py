"""Unsupervised execution: clustering and dimensionality reduction.

Deliberately standalone — no target column, no train/test split, no CV/HPO
machinery from execution.py. Runs on the full dataset and validates itself
with internal metrics (silhouette, Davies-Bouldin, explained variance) since
there is no held-out label to score against.

Default path per task_type (what orchestrator.py calls today): KMeans for
clustering, PCA for dimensionality reduction. DBSCAN, HDBSCAN, Gaussian
Mixture, Agglomerative, t-SNE and UMAP are additional engines selectable via
the `algorithm` param — proven with the standalone-script harness, not yet
wired to the orchestrator/UI (that's a later step once the default path's
frontend has real usage).

No iterative retry loop for any algorithm — see orchestrator.py's routing
note for why.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# task_type values that route to this module instead of execution.py's
# supervised pipeline. Shared by server.py, orchestrator.py, report_generator.py.
UNSUPERVISED_TASK_TYPES = frozenset({"clustering", "dimensionality_reduction"})

# Public: which "algorithm" values produce a cluster_sizes/cluster_profiles shape
# (report_generator.py branches on this) vs. a projection-only shape.
CLUSTERING_ALGORITHMS = frozenset({"kmeans", "dbscan", "hdbscan", "gmm", "agglomerative"})
REDUCTION_ALGORITHMS = frozenset({"pca", "tsne", "umap"})


def _optional_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception:  # pragma: no cover - optional dependency
        return None


_umap_module = _optional_import("umap")

_MAX_K = 8               # upper bound on candidate cluster counts tried
_MIN_ROWS = 4             # below this, clustering/PCA aren't meaningful
_PROJECTION_SAMPLE_CAP = 500   # capped rows shipped for the frontend scatter plot
_TOP_FEATURES_PER_CLUSTER = 5


# Same identifier-detection rule feature_selection.py already uses for the
# supervised path (cardinality == row count, or a sequential-integer run, or
# an id/index/key/uuid name token) — reused here rather than re-derived so
# both paths agree on what counts as "not a real feature". Float columns are
# never flagged by cardinality: continuous measurements are expected to be
# near-unique per row and are not identifiers.
_ID_NAME_TOKENS = ("id", "index", "key", "uuid", "guid", "rowid", "row_id")


def _identifier_like_columns(df: pd.DataFrame) -> list[str]:
    n = len(df)
    flagged: list[str] = []
    for c in df.columns:
        nu = df[c].nunique(dropna=True)
        name_l = str(c).lower()
        is_float_col = df[c].dtype.kind == "f"

        by_card = (not is_float_col) and (n > 0 and nu >= n)
        by_name = any(
            tok == name_l or name_l.startswith(tok + "_") or name_l.endswith("_" + tok)
            for tok in _ID_NAME_TOKENS
        )
        if not by_card and not by_name and pd.api.types.is_numeric_dtype(df[c]) and not is_float_col:
            vals = df[c].dropna()
            if len(vals) > 10 and vals.dtype.kind in ("i", "u"):
                sorted_vals = vals.sort_values().reset_index(drop=True)
                diffs = sorted_vals.diff().dropna()
                if diffs.nunique() == 1 and float(diffs.iloc[0]) > 0:
                    by_card = True  # uniform-step integer — sequential ID

        if by_card or by_name:
            flagged.append(c)
    return flagged


def _prepare_features(df: pd.DataFrame, target_column: str | None) -> tuple[pd.DataFrame, list[str]]:
    """Numeric-only, NaN-free, unscaled feature frame — same coercion approach
    as execution.py's _safe_numeric_features, kept local to stay standalone.

    Also drops identifier-like columns (e.g. CustomerID) so they never enter
    distance calculations or show up as a "distinguishing feature" in cluster
    profiles. Returns (features, excluded_identifier_columns).
    """
    work = df.drop(columns=[target_column]) if target_column and target_column in df.columns else df.copy()
    work = work.loc[:, ~work.isna().all(axis=0)]  # drop fully-null columns

    excluded = _identifier_like_columns(work)
    if excluded:
        work = work.drop(columns=excluded)

    for c in work.columns:
        if pd.api.types.is_bool_dtype(work[c]):
            work[c] = work[c].astype(int)
        elif not pd.api.types.is_numeric_dtype(work[c]):
            work[c] = pd.factorize(work[c].astype(str), sort=True)[0]
    return work, excluded


def _scaled_matrix(features: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    imputed = SimpleImputer(strategy="median").fit_transform(features)
    scaler = StandardScaler()
    return scaler.fit_transform(imputed), scaler


def _projection_2d(x_scaled: np.ndarray, labels: np.ndarray | None) -> list[dict[str, Any]]:
    n = x_scaled.shape[0]
    if x_scaled.shape[1] >= 2:
        coords = PCA(n_components=2, random_state=42).fit_transform(x_scaled)
    else:
        coords = np.column_stack([x_scaled[:, 0], np.zeros(n)])

    idx = np.arange(n)
    if n > _PROJECTION_SAMPLE_CAP:
        idx = np.random.RandomState(42).choice(n, size=_PROJECTION_SAMPLE_CAP, replace=False)

    out = []
    for i in idx:
        point = {"x": round(float(coords[i, 0]), 4), "y": round(float(coords[i, 1]), 4)}
        if labels is not None:
            point["cluster"] = int(labels[i])
        out.append(point)
    return out


def _cluster_profiles(features: pd.DataFrame, x_scaled: np.ndarray, labels: np.ndarray) -> list[dict[str, Any]]:
    """Per-cluster characteristic features: largest |z-score| vs the overall mean."""
    overall_mean = x_scaled.mean(axis=0)
    overall_std = x_scaled.std(axis=0)
    overall_std[overall_std == 0] = 1.0
    n_total = len(labels)

    profiles = []
    for cluster_id in sorted(set(labels.tolist())):
        mask = labels == cluster_id
        size = int(mask.sum())
        cluster_mean_scaled = x_scaled[mask].mean(axis=0)
        z_scores = (cluster_mean_scaled - overall_mean) / overall_std
        order = np.argsort(np.abs(z_scores))[::-1][:_TOP_FEATURES_PER_CLUSTER]
        raw_means = features.to_numpy()[mask].mean(axis=0)
        top_features = [
            {
                "name": str(features.columns[j]),
                "mean": round(float(raw_means[j]), 4),
                "z_score": round(float(z_scores[j]), 3),
            }
            for j in order
        ]
        profiles.append({
            "cluster": int(cluster_id),
            "size": size,
            "pct": round(size / n_total * 100, 1),
            "top_features": top_features,
        })
    return profiles


def _run_kmeans(df: pd.DataFrame, target_column: str | None) -> dict[str, Any]:
    features, excluded_features = _prepare_features(df, target_column)
    n_rows, n_features = features.shape
    if n_rows < _MIN_ROWS or n_features < 1:
        return {"status": "failed", "reason": f"too few rows ({n_rows}) or features ({n_features}) to cluster"}

    x_scaled, _scaler = _scaled_matrix(features)

    k_max = max(2, min(_MAX_K, n_rows // 2, n_rows - 1))
    elbow_data: list[dict[str, Any]] = []
    fits: dict[int, tuple[KMeans, float | None]] = {}

    for k in range(2, k_max + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(x_scaled)
        sil = float(silhouette_score(x_scaled, labels)) if 2 <= k < n_rows else None
        elbow_data.append({"k": k, "inertia": round(float(model.inertia_), 4),
                            "silhouette": round(sil, 4) if sil is not None else None})
        fits[k] = (model, sil)

    scored = [(k, sil) for k, (_m, sil) in fits.items() if sil is not None]
    if not scored:
        return {"status": "failed", "reason": "silhouette score could not be computed for any k"}
    best_k = max(scored, key=lambda ks: ks[1])[0]
    best_model, best_sil = fits[best_k]
    labels = best_model.labels_

    db_score = float(davies_bouldin_score(x_scaled, labels))
    sizes = pd.Series(labels).value_counts().sort_index()
    cluster_sizes = {str(int(k)): int(v) for k, v in sizes.items()}

    return {
        "status": "ok",
        "algorithm": "kmeans",
        "task_type": "clustering",
        "n_rows": n_rows,
        "feature_columns": list(features.columns),
        "excluded_features": excluded_features,
        "best_k": best_k,
        "silhouette_score": round(best_sil, 4),
        "davies_bouldin_score": round(db_score, 4),
        "inertia": round(float(best_model.inertia_), 4),
        "elbow_data": elbow_data,
        "cluster_sizes": cluster_sizes,
        "cluster_profiles": _cluster_profiles(features, x_scaled, labels),
        "projection_2d": _projection_2d(x_scaled, labels),
    }


def _search_best_k(
    x_scaled: np.ndarray, build_model, k_max: int, has_inertia: bool = False,
) -> tuple[int, np.ndarray, float, list[dict[str, Any]]] | None:
    """Shared k-search loop for partitional algorithms (GMM, Agglomerative):
    try k=2..k_max, pick the k with the highest silhouette score. Mirrors
    _run_kmeans's own loop (kept separate/untouched since it's the
    already-verified default path)."""
    n_rows = x_scaled.shape[0]
    elbow_data: list[dict[str, Any]] = []
    fits: dict[int, tuple[np.ndarray, float | None, float | None]] = {}

    for k in range(2, k_max + 1):
        model = build_model(k)
        labels = model.fit_predict(x_scaled)
        sil = float(silhouette_score(x_scaled, labels)) if 2 <= k < n_rows else None
        inertia = float(getattr(model, "inertia_", None)) if has_inertia and hasattr(model, "inertia_") else None
        elbow_data.append({"k": k, "inertia": round(inertia, 4) if inertia is not None else None,
                            "silhouette": round(sil, 4) if sil is not None else None})
        fits[k] = (labels, sil, inertia)

    scored = [(k, sil) for k, (_l, sil, _i) in fits.items() if sil is not None]
    if not scored:
        return None
    best_k = max(scored, key=lambda ks: ks[1])[0]
    best_labels, best_sil, _best_inertia = fits[best_k]
    return best_k, best_labels, best_sil, elbow_data


def _run_gmm(df: pd.DataFrame, target_column: str | None) -> dict[str, Any]:
    features, excluded_features = _prepare_features(df, target_column)
    n_rows, n_features = features.shape
    if n_rows < _MIN_ROWS or n_features < 1:
        return {"status": "failed", "reason": f"too few rows ({n_rows}) or features ({n_features}) for GMM"}

    x_scaled, _scaler = _scaled_matrix(features)
    k_max = max(2, min(_MAX_K, n_rows // 2, n_rows - 1))

    result = _search_best_k(
        x_scaled, lambda k: GaussianMixture(n_components=k, random_state=42), k_max,
    )
    if result is None:
        return {"status": "failed", "reason": "silhouette score could not be computed for any k"}
    best_k, labels, best_sil, elbow_data = result

    db_score = float(davies_bouldin_score(x_scaled, labels))
    sizes = pd.Series(labels).value_counts().sort_index()
    cluster_sizes = {str(int(k)): int(v) for k, v in sizes.items()}

    return {
        "status": "ok",
        "algorithm": "gmm",
        "task_type": "clustering",
        "n_rows": n_rows,
        "feature_columns": list(features.columns),
        "excluded_features": excluded_features,
        "best_k": best_k,
        "silhouette_score": round(best_sil, 4),
        "davies_bouldin_score": round(db_score, 4),
        "elbow_data": elbow_data,
        "cluster_sizes": cluster_sizes,
        "cluster_profiles": _cluster_profiles(features, x_scaled, labels),
        "projection_2d": _projection_2d(x_scaled, labels),
    }


def _run_agglomerative(df: pd.DataFrame, target_column: str | None) -> dict[str, Any]:
    features, excluded_features = _prepare_features(df, target_column)
    n_rows, n_features = features.shape
    if n_rows < _MIN_ROWS or n_features < 1:
        return {"status": "failed", "reason": f"too few rows ({n_rows}) or features ({n_features}) for Agglomerative clustering"}

    x_scaled, _scaler = _scaled_matrix(features)
    k_max = max(2, min(_MAX_K, n_rows // 2, n_rows - 1))

    result = _search_best_k(
        x_scaled, lambda k: AgglomerativeClustering(n_clusters=k), k_max,
    )
    if result is None:
        return {"status": "failed", "reason": "silhouette score could not be computed for any k"}
    best_k, labels, best_sil, elbow_data = result

    db_score = float(davies_bouldin_score(x_scaled, labels))
    sizes = pd.Series(labels).value_counts().sort_index()
    cluster_sizes = {str(int(k)): int(v) for k, v in sizes.items()}

    return {
        "status": "ok",
        "algorithm": "agglomerative",
        "task_type": "clustering",
        "n_rows": n_rows,
        "feature_columns": list(features.columns),
        "excluded_features": excluded_features,
        "best_k": best_k,
        "silhouette_score": round(best_sil, 4),
        "davies_bouldin_score": round(db_score, 4),
        "elbow_data": elbow_data,
        "cluster_sizes": cluster_sizes,
        "cluster_profiles": _cluster_profiles(features, x_scaled, labels),
        "projection_2d": _projection_2d(x_scaled, labels),
    }


def _density_result(
    algorithm: str, features: pd.DataFrame, x_scaled: np.ndarray, labels: np.ndarray,
    excluded_features: list[str],
) -> dict[str, Any]:
    """Shared result-shaping for density-based algorithms (DBSCAN/HDBSCAN) —
    noise points (label -1) are kept as their own bucket rather than dropped,
    so cluster_sizes/profiles account for every row."""
    n_rows = x_scaled.shape[0]
    unique_labels = sorted(set(labels.tolist()))
    n_clusters = len([l for l in unique_labels if l != -1])
    n_noise = int((labels == -1).sum())

    sil = None
    db_score = None
    if len(unique_labels) >= 2:
        sil = round(float(silhouette_score(x_scaled, labels)), 4)
        db_score = round(float(davies_bouldin_score(x_scaled, labels)), 4)

    sizes = pd.Series(labels).value_counts().sort_index()
    cluster_sizes = {str(int(k)): int(v) for k, v in sizes.items()}

    return {
        "status": "ok",
        "algorithm": algorithm,
        "task_type": "clustering",
        "n_rows": n_rows,
        "feature_columns": list(features.columns),
        "excluded_features": excluded_features,
        "n_clusters_found": n_clusters,
        "n_noise": n_noise,
        "silhouette_score": sil,
        "davies_bouldin_score": db_score,
        "cluster_sizes": cluster_sizes,
        "cluster_profiles": _cluster_profiles(features, x_scaled, labels),
        "projection_2d": _projection_2d(x_scaled, labels),
        "note": "Cluster -1 is noise (points DBSCAN/HDBSCAN could not assign to any cluster), not a real segment.",
    }


def _run_dbscan(df: pd.DataFrame, target_column: str | None) -> dict[str, Any]:
    features, excluded_features = _prepare_features(df, target_column)
    n_rows, n_features = features.shape
    if n_rows < _MIN_ROWS or n_features < 1:
        return {"status": "failed", "reason": f"too few rows ({n_rows}) or features ({n_features}) for DBSCAN"}

    x_scaled, _scaler = _scaled_matrix(features)
    # Standard k-distance heuristic: min_samples ~ n_features+1 (floor 5), eps =
    # median distance to each point's min_samples-th nearest neighbor.
    min_samples = max(5, n_features + 1)
    min_samples = min(min_samples, n_rows - 1)
    nn = NearestNeighbors(n_neighbors=min_samples).fit(x_scaled)
    dists, _idx = nn.kneighbors(x_scaled)
    eps = float(np.median(dists[:, -1])) or 0.5

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(x_scaled)
    if len(set(labels.tolist())) < 2:
        return {"status": "failed", "reason": f"DBSCAN found only one group (eps={eps:.3f}, min_samples={min_samples}) — try a different algorithm"}

    result = _density_result("dbscan", features, x_scaled, labels, excluded_features)
    result["eps"] = round(eps, 4)
    result["min_samples"] = min_samples
    return result


def _run_hdbscan(df: pd.DataFrame, target_column: str | None) -> dict[str, Any]:
    features, excluded_features = _prepare_features(df, target_column)
    n_rows, n_features = features.shape
    if n_rows < _MIN_ROWS or n_features < 1:
        return {"status": "failed", "reason": f"too few rows ({n_rows}) or features ({n_features}) for HDBSCAN"}

    x_scaled, _scaler = _scaled_matrix(features)
    min_cluster_size = max(5, n_rows // 50)

    labels = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(x_scaled)
    if len(set(labels.tolist())) < 2:
        return {"status": "failed", "reason": f"HDBSCAN found only one group (min_cluster_size={min_cluster_size}) — try a different algorithm"}

    result = _density_result("hdbscan", features, x_scaled, labels, excluded_features)
    result["min_cluster_size"] = min_cluster_size
    return result


def _run_pca(df: pd.DataFrame, target_column: str | None) -> dict[str, Any]:
    features, excluded_features = _prepare_features(df, target_column)
    n_rows, n_features = features.shape
    if n_rows < _MIN_ROWS or n_features < 1:
        return {"status": "failed", "reason": f"too few rows ({n_rows}) or features ({n_features}) for PCA"}

    x_scaled, _scaler = _scaled_matrix(features)
    n_components = max(1, min(n_features, n_rows - 1, 10))

    model = PCA(n_components=n_components, random_state=42)
    model.fit(x_scaled)
    evr = [round(float(v), 4) for v in model.explained_variance_ratio_]
    cumulative = [round(float(v), 4) for v in np.cumsum(model.explained_variance_ratio_)]

    return {
        "status": "ok",
        "algorithm": "pca",
        "task_type": "dimensionality_reduction",
        "n_rows": n_rows,
        "feature_columns": list(features.columns),
        "excluded_features": excluded_features,
        "n_components": n_components,
        "explained_variance_ratio": evr,
        "cumulative_variance": cumulative,
        "projection_2d": _projection_2d(x_scaled, None),
    }


def _run_tsne(df: pd.DataFrame, target_column: str | None) -> dict[str, Any]:
    features, excluded_features = _prepare_features(df, target_column)
    n_rows, n_features = features.shape
    if n_rows < _MIN_ROWS or n_features < 1:
        return {"status": "failed", "reason": f"too few rows ({n_rows}) or features ({n_features}) for t-SNE"}

    x_scaled, _scaler = _scaled_matrix(features)
    perplexity = min(30.0, max(5.0, (n_rows - 1) / 3))
    coords = TSNE(n_components=2, random_state=42, perplexity=perplexity, init="pca").fit_transform(x_scaled)

    idx = np.arange(n_rows)
    if n_rows > _PROJECTION_SAMPLE_CAP:
        idx = np.random.RandomState(42).choice(n_rows, size=_PROJECTION_SAMPLE_CAP, replace=False)
    projection = [{"x": round(float(coords[i, 0]), 4), "y": round(float(coords[i, 1]), 4)} for i in idx]

    return {
        "status": "ok",
        "algorithm": "tsne",
        "task_type": "dimensionality_reduction",
        "n_rows": n_rows,
        "feature_columns": list(features.columns),
        "excluded_features": excluded_features,
        "n_components": 2,
        "projection_2d": projection,
        "note": "t-SNE has no explained-variance concept — axes are a 2D layout that preserves local structure only, not a linear projection.",
    }


def _run_umap(df: pd.DataFrame, target_column: str | None) -> dict[str, Any]:
    if _umap_module is None:
        return {"status": "failed", "reason": "umap-learn not installed", "algorithm": "umap"}

    features, excluded_features = _prepare_features(df, target_column)
    n_rows, n_features = features.shape
    if n_rows < _MIN_ROWS or n_features < 1:
        return {"status": "failed", "reason": f"too few rows ({n_rows}) or features ({n_features}) for UMAP"}

    x_scaled, _scaler = _scaled_matrix(features)
    n_neighbors = min(15, n_rows - 1)
    coords = _umap_module.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors).fit_transform(x_scaled)

    idx = np.arange(n_rows)
    if n_rows > _PROJECTION_SAMPLE_CAP:
        idx = np.random.RandomState(42).choice(n_rows, size=_PROJECTION_SAMPLE_CAP, replace=False)
    projection = [{"x": round(float(coords[i, 0]), 4), "y": round(float(coords[i, 1]), 4)} for i in idx]

    return {
        "status": "ok",
        "algorithm": "umap",
        "task_type": "dimensionality_reduction",
        "n_rows": n_rows,
        "feature_columns": list(features.columns),
        "excluded_features": excluded_features,
        "n_components": 2,
        "projection_2d": projection,
        "note": "UMAP has no explained-variance concept — axes preserve local/global structure, not a linear projection.",
    }


_ALGORITHM_RUNNERS: dict[str, Any] = {
    "kmeans": _run_kmeans,
    "dbscan": _run_dbscan,
    "hdbscan": _run_hdbscan,
    "gmm": _run_gmm,
    "agglomerative": _run_agglomerative,
    "pca": _run_pca,
    "tsne": _run_tsne,
    "umap": _run_umap,
}


def run_unsupervised_pipeline(
    df: pd.DataFrame,
    algorithm: str | None,
    session: Any = None,  # SessionState, typed loosely to avoid a circular import
) -> dict[str, Any]:
    """Entry point.

    `algorithm` selects a specific engine — "kmeans" | "dbscan" | "hdbscan" |
    "gmm" | "agglomerative" | "pca" | "tsne" | "umap". When omitted (or not
    one of those names), it falls back to session.task_config['task_type']:
    "clustering" -> kmeans, "dimensionality_reduction" -> PCA — the only path
    the orchestrator uses today.

    Returns a JSON-serializable results dict — either {"status": "ok", ...} with
    algorithm-specific fields, or {"status": "failed", "reason": str}. Never raises.
    """
    task_config = getattr(session, "task_config", None) or {}
    target_column = task_config.get("target_column")
    requested = str(algorithm or "").strip().lower()
    task_type = str(task_config.get("task_type") or "").strip().lower()

    if requested in _ALGORITHM_RUNNERS:
        engine = requested
    elif task_type == "dimensionality_reduction":
        engine = "pca"
    else:
        engine = "kmeans"

    try:
        result = _ALGORITHM_RUNNERS[engine](df, target_column)
    except Exception as exc:  # pragma: no cover — must never crash the pipeline
        logger.warning("run_unsupervised_pipeline(%s) failed: %s", engine, exc)
        result = {"status": "failed", "reason": str(exc), "algorithm": engine}

    if session is not None:
        try:
            from automl_system.utils.decision_log import log_decision
            if result.get("status") == "ok":
                log_decision(
                    session, step="Phase 5 - Unsupervised Execution",
                    action=f"{result['algorithm']}_completed",
                    rationale=(
                        f"{result['algorithm']} on {result.get('n_rows')} rows, "
                        f"{len(result.get('feature_columns') or [])} features."
                    ),
                )
            else:
                log_decision(
                    session, step="Phase 5 - Unsupervised Execution",
                    action="unsupervised_failed",
                    rationale=str(result.get("reason")),
                )
        except Exception:  # pragma: no cover — logging must never break the run
            logger.warning("log_decision failed during unsupervised execution", exc_info=True)

    return result

"""SessionState — single AutoML run's config and pipeline data; serializable to dict."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


_PLAIN = (str, int, float, bool, type(None))


def _copy_data(value: Any) -> Any:
    """Copy containers recursively, skipping private non-data entries such as
    "_fitted_estimator" (trained model objects kept in Phase 5 results).

    dataclasses.asdict deep-copies everything, including those models — slow, memory
    heavy, and deep-copying an XGBoost GPU booster from the server's main thread
    crashes the process natively (the final event was never sent to the UI).
    """
    if isinstance(value, dict):
        return {
            k: _copy_data(v) for k, v in value.items()
            if not (isinstance(k, str) and k.startswith("_") and not isinstance(v, (*_PLAIN, dict, list, tuple)))
        }
    if isinstance(value, (list, tuple)):
        return [_copy_data(v) for v in value]
    return value


@dataclass
class SessionState:
    """Holds one AutoML run's configuration and intermediate results."""

    task_config: dict[str, Any] = field(default_factory=dict)
    dataset_profile: dict[str, Any] = field(default_factory=dict)
    # Captured once in Phase 1 and never overwritten. `dataset_profile` above is
    # refreshed mid-pipeline (Phase 4 planning, post-feature-engineering) to reflect
    # the current cleaned/encoded train frame, which is correct for those callers but
    # wrong for reporting "what the data looked like initially" in Phase 7.
    initial_dataset_profile: dict[str, Any] = field(default_factory=dict)
    decisions_log_id: str | None = None
    dataset_state_pointer: str | None = None
    execution_plan: dict[str, Any] = field(default_factory=dict)
    iteration_count: int = 0

    # Phase 2–3 preprocessing
    preprocessing_plan: dict[str, Any] = field(default_factory=dict)
    user_summary_planning: str = ""
    cleaning_report: dict[str, Any] = field(default_factory=dict)
    eda_results: dict[str, Any] = field(default_factory=dict)
    eda_narrative: str = ""
    user_summary_eda: str = ""
    feature_selection_log: list[dict[str, Any]] = field(default_factory=list)
    deep_cleaning_assignments: dict[str, Any] = field(default_factory=dict)
    feature_engineering_log: list[dict[str, Any]] = field(default_factory=list)
    retry_counts: dict[str, int] = field(default_factory=dict)
    last_verification: dict[str, Any] = field(default_factory=dict)
    expected_shape: tuple[int, int] | None = None
    preprocessing_meta: dict[str, Any] = field(default_factory=dict)

    # Phase 3 → Phase 4 deviation propagation.
    # Each entry: {"type": str, "column": str|None, "reason": str}
    # Types: feature_engineering_skipped, knn_downgraded, ohe_cardinality_override,
    #        target_enc_too_small, high_missing_rate, high_cardinality_survived
    preprocessing_deviations: list[dict[str, Any]] = field(default_factory=list)

    # Phase 4–5 planning + execution
    execution_results: dict[str, Any] = field(default_factory=dict)
    execution_interpretation: str = ""
    feature_importances: dict[str, Any] = field(default_factory=dict)
    shap_summary: dict[str, Any] = field(default_factory=dict)
    # Clustering / dimensionality-reduction results (Phase 5's unsupervised branch).
    # Empty for supervised runs.
    unsupervised_results: dict[str, Any] = field(default_factory=dict)

    # Phase 6–7 iterative loop + final artifacts
    all_results: list[dict[str, Any]] = field(default_factory=list)
    last_failure_diagnosis: dict[str, Any] = field(default_factory=dict)
    final_report_markdown: str = ""
    final_pipeline_script: str = ""
    # Phase 7 sandbox check of final_pipeline_script (see pipeline/script_validation.py)
    script_validation: dict[str, Any] = field(default_factory=dict)
    final_comparison_table: list[dict[str, Any]] = field(default_factory=list)

    # Token cost tracking: {phase_name: llm_call_count}
    token_cost_table: dict[str, int] = field(default_factory=dict)
    # Wall-clock seconds per graph node (accumulated across iterations)
    phase_timings: dict[str, float] = field(default_factory=dict)

    # Phase 1
    intent_result: dict[str, Any] = field(default_factory=dict)
    local_decisions_log: list[dict[str, Any]] = field(default_factory=list)
    # Structured problem/dataset understanding, consumable by Phase 2+ (see
    # automl_system.utils.phase1_summary.build_phase1_summary).
    problem_summary: dict[str, Any] = field(default_factory=dict)

    # Session lifecycle
    phase_status_map: dict[str, str] = field(default_factory=dict)
    terminated: bool = False
    is_complete: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = {f.name: _copy_data(getattr(self, f.name)) for f in fields(self)}
        if self.expected_shape is not None:
            d["expected_shape"] = list(self.expected_shape)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SessionState:
        if not data:
            return cls()
        es = data.get("expected_shape")
        shape: tuple[int, int] | None = None
        if es is not None and len(es) == 2:
            shape = (int(es[0]), int(es[1]))
        return cls(
            task_config=dict(data.get("task_config") or {}),
            dataset_profile=dict(data.get("dataset_profile") or {}),
            initial_dataset_profile=dict(data.get("initial_dataset_profile") or {}),
            decisions_log_id=data.get("decisions_log_id"),
            dataset_state_pointer=data.get("dataset_state_pointer"),
            execution_plan=dict(data.get("execution_plan") or {}),
            iteration_count=int(data.get("iteration_count") or 0),
            preprocessing_plan=dict(data.get("preprocessing_plan") or {}),
            user_summary_planning=str(data.get("user_summary_planning") or ""),
            cleaning_report=dict(data.get("cleaning_report") or {}),
            eda_results=dict(data.get("eda_results") or {}),
            eda_narrative=str(data.get("eda_narrative") or ""),
            user_summary_eda=str(data.get("user_summary_eda") or ""),
            feature_selection_log=list(data.get("feature_selection_log") or []),
            deep_cleaning_assignments=dict(data.get("deep_cleaning_assignments") or {}),
            feature_engineering_log=list(data.get("feature_engineering_log") or []),
            retry_counts=dict(data.get("retry_counts") or {}),
            last_verification=dict(data.get("last_verification") or {}),
            expected_shape=shape,
            preprocessing_meta=dict(data.get("preprocessing_meta") or {}),
            preprocessing_deviations=list(data.get("preprocessing_deviations") or []),
            execution_results=dict(data.get("execution_results") or {}),
            execution_interpretation=str(data.get("execution_interpretation") or ""),
            feature_importances=dict(data.get("feature_importances") or {}),
            shap_summary=dict(data.get("shap_summary") or {}),
            unsupervised_results=dict(data.get("unsupervised_results") or {}),
            all_results=list(data.get("all_results") or []),
            last_failure_diagnosis=dict(data.get("last_failure_diagnosis") or {}),
            final_report_markdown=str(data.get("final_report_markdown") or ""),
            final_pipeline_script=str(data.get("final_pipeline_script") or ""),
            script_validation=dict(data.get("script_validation") or {}),
            final_comparison_table=list(data.get("final_comparison_table") or []),
            token_cost_table=dict(data.get("token_cost_table") or {}),
            phase_timings=dict(data.get("phase_timings") or {}),
            intent_result=dict(data.get("intent_result") or {}),
            local_decisions_log=list(data.get("local_decisions_log") or []),
            problem_summary=dict(data.get("problem_summary") or {}),
            phase_status_map=dict(data.get("phase_status_map") or {}),
            terminated=bool(data.get("terminated")),
            is_complete=bool(data.get("is_complete")),
        )

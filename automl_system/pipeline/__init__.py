"""Pipeline execution (added per phase)."""

from automl_system.pipeline.cleaning import basic_clean
from automl_system.pipeline.code_generator import generate_pipeline_script
from automl_system.pipeline.deep_cleaning import deep_clean
from automl_system.pipeline.eda import run_eda
from automl_system.pipeline.execution import run_pipeline
from automl_system.pipeline.feature_engineering import engineer_features
from automl_system.pipeline.feature_selection import select_features
from automl_system.pipeline.verification import parse_verification_route, verify_preprocessing

__all__ = [
    "basic_clean",
    "generate_pipeline_script",
    "deep_clean",
    "engineer_features",
    "parse_verification_route",
    "run_pipeline",
    "run_eda",
    "select_features",
    "verify_preprocessing",
]

"""
Token usage tracking and measurement for profile-conditioned prompting.

Tracks the benefits of profile conditioning by measuring:
- Full profile token count vs conditioned profile token count
- Compression ratio for each agent
- Cumulative token savings across all agents in a run
"""

from __future__ import annotations

import json
import logging
from typing import Any

from automl_system.utils.profile_conditioner import (
    condition_profile_for_deep_cleaning,
    condition_profile_for_execution_planning,
    condition_profile_for_feature_engineering,
    condition_profile_for_preprocessing,
    condition_profile_for_report,
    estimate_profile_compression,
)

logger = logging.getLogger(__name__)


class ProfileCompressionTracker:
    """Track and report token savings from profile conditioning."""

    def __init__(self):
        self.metrics: list[dict[str, Any]] = []
        self.total_full_tokens = 0
        self.total_conditioned_tokens = 0

    def track_agent_profile(self, agent_name: str, profile: dict[str, Any], eda_results: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Track profile compression for an agent call.

        Args:
            agent_name: Name of the agent (e.g., 'preprocessing_planner')
            profile: Full dataset profile dict
            eda_results: Optional EDA results for report_generator

        Returns:
            Compression metrics dict with token savings information
        """
        try:
            # For report generator, pass EDA results
            if agent_name in {"report_generator", "report"}:
                metrics = estimate_profile_compression(profile, agent_name)
                # Include EDA context size if available
                if eda_results:
                    eda_size = len(json.dumps(eda_results, default=str))
                    metrics["eda_context_chars"] = eda_size
            else:
                metrics = estimate_profile_compression(profile, agent_name)

            self.metrics.append(metrics)
            self.total_full_tokens += metrics["estimated_full_tokens"]
            self.total_conditioned_tokens += metrics["estimated_conditioned_tokens"]

            logger.info(f"Profile compression for {agent_name}: {metrics['compression_ratio_percent']}% reduction "
                       f"({metrics['estimated_token_reduction']} tokens saved)")

            return metrics
        except Exception as e:
            logger.warning(f"Failed to track profile compression for {agent_name}: {e}")
            return {}

    def get_summary(self) -> dict[str, Any]:
        """
        Get cumulative compression summary for the run.

        Returns:
            Dict with total savings, compression ratio, and per-agent breakdown
        """
        total_reduction = self.total_full_tokens - self.total_conditioned_tokens
        total_ratio = (
            (total_reduction / self.total_full_tokens * 100)
            if self.total_full_tokens > 0
            else 0
        )

        return {
            "total_full_tokens": self.total_full_tokens,
            "total_conditioned_tokens": self.total_conditioned_tokens,
            "total_token_reduction": total_reduction,
            "compression_ratio_percent": round(total_ratio, 1),
            "agents_tracked": len(self.metrics),
            "per_agent_metrics": self.metrics,
        }

    def log_summary(self):
        """Log cumulative compression summary."""
        summary = self.get_summary()
        logger.info(
            f"Profile-Conditioned Prompting Summary:\n"
            f"  Total full tokens: {summary['total_full_tokens']}\n"
            f"  Total conditioned tokens: {summary['total_conditioned_tokens']}\n"
            f"  Total reduction: {summary['total_token_reduction']} tokens ({summary['compression_ratio_percent']}%)\n"
            f"  Agents tracked: {summary['agents_tracked']}"
        )


# Global tracker instance
_tracker: ProfileCompressionTracker | None = None


def get_tracker() -> ProfileCompressionTracker:
    """Get or create the global compression tracker."""
    global _tracker
    if _tracker is None:
        _tracker = ProfileCompressionTracker()
    return _tracker


def reset_tracker():
    """Reset the global tracker (useful for testing or new runs)."""
    global _tracker
    _tracker = None


def track_profile_usage(agent_name: str, profile: dict[str, Any], eda_results: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Convenience function to track profile for an agent.

    Usage:
        metrics = track_profile_usage("preprocessing_planner", session.dataset_profile)
        logger.info(f"Saved {metrics['estimated_token_reduction']} tokens")
    """
    return get_tracker().track_agent_profile(agent_name, profile, eda_results)


def get_compression_summary() -> dict[str, Any]:
    """Get the compression summary from the global tracker."""
    return get_tracker().get_summary()


def log_compression_summary():
    """Log the cumulative compression summary."""
    get_tracker().log_summary()

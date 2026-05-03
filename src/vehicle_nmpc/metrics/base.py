"""Metric result containers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(kw_only=True, slots=True)
class TrackingMetrics:
    """Trajectory tracking quality metrics."""

    mean_abs_longitudinal_error: float
    """Mean absolute longitudinal tracking error."""

    mean_abs_lateral_error: float
    """Mean absolute lateral tracking error."""

    max_abs_lateral_error: float
    """Maximum absolute lateral tracking error."""

    rmse_lateral_error: float
    """Root-mean-square lateral tracking error."""

    mean_abs_heading_error: float
    """Mean absolute heading tracking error."""

    max_abs_heading_error: float
    """Maximum absolute heading tracking error."""

    itae_lateral_error: float
    """Integral time absolute lateral error."""

    final_position_error: float
    """Euclidean position error at the final sample."""

    final_heading_error: float
    """Absolute heading error at the final sample."""


@dataclass(kw_only=True, slots=True)
class PerformanceMetrics:
    """Controller runtime metrics."""

    mean_solve_time: float
    """Mean controller solve time per control step."""

    max_solve_time: float
    """Maximum controller solve time per control step."""

    p95_solve_time: float
    """95th percentile controller solve time."""

    total_solve_time: float
    """Total controller solve time over the rollout."""

    real_time_factor_mean: float
    """Mean solve time divided by the control period."""

    real_time_factor_max: float
    """Maximum solve time divided by the control period."""

    mean_preparation_time: float | None
    """Mean RTI preparation phase time, when available."""

    max_preparation_time: float | None
    """Maximum RTI preparation phase time, when available."""

    mean_feedback_time: float | None
    """Mean RTI feedback phase time, when available."""

    max_feedback_time: float | None
    """Maximum RTI feedback phase time, when available."""

    solver_status_counts: dict[str, int]
    """Counts of solver statuses grouped by phase and status code."""


@dataclass(kw_only=True, slots=True)
class EvaluationMetrics:
    """All metrics for one closed-loop rollout."""

    trajectory_name: str
    """Name of the evaluated trajectory."""

    tracking: TrackingMetrics
    """Tracking quality metrics."""

    performance: PerformanceMetrics
    """Controller runtime metrics."""

    def print(self) -> str:
        """Return a compact human-readable metric summary."""
        return (
            f"Trajectory '{self.trajectory_name}' metrics: "
            f"rmse_y={self.tracking.rmse_lateral_error:.4g}, "
            f"max_y={self.tracking.max_abs_lateral_error:.4g}, "
            f"itae_y={self.tracking.itae_lateral_error:.4g}, "
            f"mean_solve={self.performance.mean_solve_time:.4g} s, "
            f"max_solve={self.performance.max_solve_time:.4g} s, "
            f"rtf_mean={self.performance.real_time_factor_mean:.4g}, "
            f"rtf_max={self.performance.real_time_factor_max:.4g}"
        )

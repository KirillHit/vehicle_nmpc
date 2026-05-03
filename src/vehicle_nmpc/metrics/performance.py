"""Controller runtime metrics."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

from vehicle_nmpc.metrics.base import PerformanceMetrics

if TYPE_CHECKING:
    from vehicle_nmpc.experiments.closed_loop import ClosedLoopResult


def evaluate_performance(result: ClosedLoopResult, *, dt: float) -> PerformanceMetrics:
    """Evaluate controller runtime metrics for one closed-loop rollout."""
    if dt <= 0.0:
        msg = f"dt must be positive, got {dt}."
        raise ValueError(msg)

    preparation_times = _optional_stat_array(result.stats, "preparation_time")
    feedback_times = _optional_stat_array(result.stats, "feedback_time")
    solve_times = _solve_times(result.stats, preparation_times, feedback_times)
    status_counts = _status_counts(result.stats)

    return PerformanceMetrics(
        mean_solve_time=float(np.mean(solve_times)),
        max_solve_time=float(np.max(solve_times)),
        p95_solve_time=float(np.percentile(solve_times, 95)),
        total_solve_time=float(np.sum(solve_times)),
        real_time_factor_mean=float(np.mean(solve_times) / dt),
        real_time_factor_max=float(np.max(solve_times) / dt),
        mean_preparation_time=_mean_or_none(preparation_times),
        max_preparation_time=_max_or_none(preparation_times),
        mean_feedback_time=_mean_or_none(feedback_times),
        max_feedback_time=_max_or_none(feedback_times),
        solver_status_counts=status_counts,
    )


def _solve_times(
    stats: list[dict],
    preparation_times: np.ndarray | None,
    feedback_times: np.ndarray | None,
) -> np.ndarray:
    """Extract total solve time per step from stats."""
    direct_times = _optional_stat_array(stats, "solve_time")
    if direct_times is not None:
        return direct_times

    if preparation_times is not None and feedback_times is not None:
        return preparation_times + feedback_times

    msg = "Controller stats must contain solve_time or both preparation_time and feedback_time."
    raise ValueError(msg)


def _optional_stat_array(stats: list[dict], key: str) -> np.ndarray | None:
    """Return a numeric array for a stat key, or None when it is unavailable."""
    values = [step_stats[key] for step_stats in stats if key in step_stats]
    if not values:
        return None
    if len(values) != len(stats):
        msg = f"Stat key '{key}' is present only for {len(values)} of {len(stats)} steps."
        raise ValueError(msg)
    return np.asarray(values, dtype=float)


def _status_counts(stats: list[dict]) -> dict[str, int]:
    """Count solver statuses by phase."""
    counter: Counter[str] = Counter()
    for step_stats in stats:
        for key, value in step_stats.items():
            if not key.endswith("_status"):
                continue
            phase = key[: -len("_status")]
            counter[f"{phase}:{int(value)}"] += 1
    return dict(counter)


def _mean_or_none(values: np.ndarray | None) -> float | None:
    """Return mean for non-empty arrays."""
    if values is None:
        return None
    return float(np.mean(values))


def _max_or_none(values: np.ndarray | None) -> float | None:
    """Return maximum for non-empty arrays."""
    if values is None:
        return None
    return float(np.max(values))

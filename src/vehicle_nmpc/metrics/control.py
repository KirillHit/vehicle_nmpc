"""Control smoothness metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from vehicle_nmpc.metrics.base import ControlMetrics

if TYPE_CHECKING:
    from vehicle_nmpc.experiments.closed_loop import ClosedLoopResult

_MATRIX_NDIM = 2
_TRACKED_CONTROL_SIZE = 2
_MIN_DELTA_SAMPLES = 2


def evaluate_control(result: ClosedLoopResult) -> ControlMetrics:
    """Evaluate control increment metrics for one closed-loop rollout."""
    controls = _as_control_trajectory(result.controls)
    if controls.shape[0] < _MIN_DELTA_SAMPLES:
        delta_controls = np.zeros((0, controls.shape[1]), dtype=float)
    else:
        delta_controls = np.diff(controls, axis=0)

    left_delta = delta_controls[:, 0]
    right_delta = delta_controls[:, 1]

    return ControlMetrics(
        mean_squared_delta_left_control=_mean_square(left_delta),
        mean_squared_delta_right_control=_mean_square(right_delta),
        sum_squared_delta_left_control=float(np.sum(np.square(left_delta))),
        sum_squared_delta_right_control=float(np.sum(np.square(right_delta))),
        max_abs_delta_left_control=_max_abs(left_delta),
        max_abs_delta_right_control=_max_abs(right_delta),
        mean_squared_delta_control=_mean_square(delta_controls),
    )


def _as_control_trajectory(values: np.ndarray) -> np.ndarray:
    """Convert and validate a control trajectory with tracked-vehicle controls."""
    array = np.asarray(values, dtype=float)
    if array.ndim != _MATRIX_NDIM or array.shape[1] < _TRACKED_CONTROL_SIZE:
        msg = f"Expected controls shape (n_samples, nu>=2), got {array.shape}."
        raise ValueError(msg)
    return array


def _mean_square(values: np.ndarray) -> float:
    """Return mean squared value, or zero for empty arrays."""
    if values.size == 0:
        return 0.0
    return float(np.mean(np.square(values)))


def _max_abs(values: np.ndarray) -> float:
    """Return maximum absolute value, or zero for empty arrays."""
    if values.size == 0:
        return 0.0
    return float(np.max(np.abs(values)))

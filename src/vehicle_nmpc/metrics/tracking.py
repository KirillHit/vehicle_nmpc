"""Tracking quality metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from vehicle_nmpc.metrics.base import TrackingMetrics

if TYPE_CHECKING:
    from vehicle_nmpc.experiments.closed_loop import ClosedLoopResult

_MATRIX_NDIM = 2
_SE2_SIZE = 3


def evaluate_tracking(result: ClosedLoopResult, *, dt: float) -> TrackingMetrics:
    """Evaluate trajectory tracking quality for one closed-loop rollout."""
    if dt <= 0.0:
        msg = f"dt must be positive, got {dt}."
        raise ValueError(msg)

    errors = local_tracking_errors(result.states, result.reference_states)
    longitudinal_error = errors[:, 0]
    lateral_error = errors[:, 1]
    heading_error = errors[:, 2]
    time = dt * np.arange(lateral_error.size)

    final_position_delta = result.reference_states[-1, :2] - result.states[-1, :2]

    return TrackingMetrics(
        mean_abs_longitudinal_error=float(np.mean(np.abs(longitudinal_error))),
        mean_abs_lateral_error=float(np.mean(np.abs(lateral_error))),
        max_abs_lateral_error=float(np.max(np.abs(lateral_error))),
        rmse_lateral_error=float(np.sqrt(np.mean(np.square(lateral_error)))),
        mean_abs_heading_error=float(np.mean(np.abs(heading_error))),
        max_abs_heading_error=float(np.max(np.abs(heading_error))),
        itae_lateral_error=float(np.sum(time * np.abs(lateral_error) * dt)),
        final_position_error=float(np.linalg.norm(final_position_delta)),
        final_heading_error=float(abs(heading_error[-1])),
    )


def local_tracking_errors(states: np.ndarray, reference_states: np.ndarray) -> np.ndarray:
    """Return local SE(2) tracking errors [e_x, e_y, e_theta] for each sample."""
    states_array = _as_state_trajectory("states", states)
    reference_array = _as_state_trajectory("reference_states", reference_states)
    if states_array.shape != reference_array.shape:
        msg = (
            "Expected states and reference_states to have the same shape, "
            f"got {states_array.shape} and {reference_array.shape}."
        )
        raise ValueError(msg)

    position_error = reference_array[:, :2] - states_array[:, :2]
    heading = states_array[:, 2]

    longitudinal_error = (
        np.cos(heading) * position_error[:, 0] + np.sin(heading) * position_error[:, 1]
    )
    lateral_error = -np.sin(heading) * position_error[:, 0] + np.cos(heading) * position_error[:, 1]
    heading_error = _wrap_to_pi(reference_array[:, 2] - states_array[:, 2])

    return np.column_stack((longitudinal_error, lateral_error, heading_error))


def _as_state_trajectory(name: str, values: np.ndarray) -> np.ndarray:
    """Convert and validate a state trajectory with SE(2) leading states."""
    array = np.asarray(values, dtype=float)
    if array.ndim != _MATRIX_NDIM or array.shape[1] < _SE2_SIZE:
        msg = f"Expected {name} shape (n_samples, nx>=3), got {array.shape}."
        raise ValueError(msg)
    return array


def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

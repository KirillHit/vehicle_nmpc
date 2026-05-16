"""Plotting helpers for evaluation metrics."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

mpl.use("Agg")

_MATRIX_NDIM = 2
_STATE_SIZE = 3


def save_trajectory_plot(
    states: np.ndarray,
    reference_states: np.ndarray,
    path: str | Path,
    *,
    title: str,
) -> None:
    """Save a reference-vs-actual trajectory plot."""
    states_array = _as_pose_trajectory("states", states)
    reference_array = _as_pose_trajectory("reference_states", reference_states)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.0, 6.0), constrained_layout=True)
    ax.plot(reference_array[:, 0], reference_array[:, 1], "--", label="reference")
    ax.plot(states_array[:, 0], states_array[:, 1], "-", label="actual")
    _plot_yaw_markers(ax, reference_array, color="tab:blue", label="reference yaw")
    _plot_yaw_markers(ax, states_array, color="tab:orange", label="actual yaw")
    ax.scatter(reference_array[0, 0], reference_array[0, 1], marker="o", label="start")
    ax.scatter(reference_array[-1, 0], reference_array[-1, 1], marker="x", label="finish")

    ax.set_title(title)
    ax.set_xlabel("X, m")
    ax.set_ylabel("Y, m")
    ax.axis("equal")
    ax.grid(visible=True, linewidth=0.5, alpha=0.4)
    ax.legend()

    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_yaw_markers(
    ax: plt.Axes,
    trajectory: np.ndarray,
    *,
    color: str,
    label: str,
) -> None:
    """Draw sparse heading markers on a pose trajectory."""
    indices = np.arange(0, trajectory.shape[0], 50, dtype=int)
    if indices.size == 0:
        return

    x_axis = trajectory[indices, 0]
    y_axis = trajectory[indices, 1]
    yaw = trajectory[indices, 2]
    yaw_length = _yaw_marker_length(trajectory)
    ax.quiver(
        x_axis,
        y_axis,
        yaw_length * np.cos(yaw),
        yaw_length * np.sin(yaw),
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.003,
        color=color,
        alpha=0.65,
        label=label,
    )


def _yaw_marker_length(trajectory: np.ndarray) -> float:
    """Return a stable heading marker length in plot data units."""
    x_range = float(np.ptp(trajectory[:, 0]))
    y_range = float(np.ptp(trajectory[:, 1]))
    span = max(x_range, y_range, 1.0)
    return 0.04 * span


def _as_pose_trajectory(name: str, values: np.ndarray) -> np.ndarray:
    """Convert and validate a trajectory with pose leading states."""
    array = np.asarray(values, dtype=float)
    if array.ndim != _MATRIX_NDIM or array.shape[1] < _STATE_SIZE:
        msg = f"Expected {name} shape (n_samples, nx>=3), got {array.shape}."
        raise ValueError(msg)
    return array

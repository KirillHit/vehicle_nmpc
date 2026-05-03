"""Plotting helpers for evaluation metrics."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

mpl.use("Agg")

_MATRIX_NDIM = 2
_POSITION_STATE_SIZE = 2


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


def _as_pose_trajectory(name: str, values: np.ndarray) -> np.ndarray:
    """Convert and validate a trajectory with pose leading states."""
    array = np.asarray(values, dtype=float)
    if array.ndim != _MATRIX_NDIM or array.shape[1] < _POSITION_STATE_SIZE:
        msg = f"Expected {name} shape (n_samples, nx>=2), got {array.shape}."
        raise ValueError(msg)
    return array

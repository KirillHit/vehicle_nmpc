"""Constant-curvature turn trajectory provider."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vehicle_nmpc.trajectory.builder import register_trajectory
from vehicle_nmpc.trajectory.constant_speed_route import (
    ConstantSpeedRouteConfig,
    ConstantSpeedRouteProvider,
)

_STRAIGHT_CURVATURE_TOL = 1e-9


@register_trajectory("turn")
class TurnTrajectoryProvider(ConstantSpeedRouteProvider):
    """Constant-curvature turn trajectory provider."""

    @dataclass(frozen=True, kw_only=True, slots=True)
    class Config(ConstantSpeedRouteConfig):
        """Constant-curvature turn trajectory configuration."""

        curvature: float = 0.4
        """Signed path curvature."""

    def _pose_at_progress(self, progress_values: np.ndarray) -> np.ndarray:
        """Return constant-curvature route poses by progress."""
        yaw = self._cfg.curvature * progress_values
        if abs(self._cfg.curvature) < _STRAIGHT_CURVATURE_TOL:
            x_axis = progress_values
            y_axis = np.zeros_like(progress_values)
        else:
            x_axis = np.sin(yaw) / self._cfg.curvature
            y_axis = (1.0 - np.cos(yaw)) / self._cfg.curvature
        return np.column_stack((x_axis, y_axis, yaw))

    def _curvature_at_progress(self, progress_values: np.ndarray) -> np.ndarray:
        """Return constant route curvature."""
        return np.full(progress_values.shape, self._cfg.curvature)

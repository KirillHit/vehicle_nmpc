"""Straight-line trajectory provider."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vehicle_nmpc.trajectory.builder import register_trajectory
from vehicle_nmpc.trajectory.constant_speed_route import (
    ConstantSpeedRouteConfig,
    ConstantSpeedRouteProvider,
)


@register_trajectory("straight")
class StraightTrajectoryProvider(ConstantSpeedRouteProvider):
    """Straight-line trajectory provider."""

    @dataclass(frozen=True, kw_only=True, slots=True)
    class Config(ConstantSpeedRouteConfig):
        """Straight-line trajectory configuration."""

        heading: float = 0.0
        """Reference heading angle."""

    def _pose_at_progress(self, progress_values: np.ndarray) -> np.ndarray:
        """Return straight-line route poses by progress."""
        heading = np.full(progress_values.shape, self._cfg.heading)
        return np.column_stack(
            (
                progress_values * np.cos(self._cfg.heading),
                progress_values * np.sin(self._cfg.heading),
                heading,
            )
        )

    def _curvature_at_progress(self, progress_values: np.ndarray) -> np.ndarray:
        """Return zero curvature for a straight route."""
        return np.zeros_like(progress_values)

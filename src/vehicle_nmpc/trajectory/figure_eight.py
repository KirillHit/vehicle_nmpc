"""Figure-eight trajectory provider."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vehicle_nmpc.trajectory.builder import register_trajectory
from vehicle_nmpc.trajectory.constant_speed_route import (
    ConstantSpeedRouteConfig,
    ConstantSpeedRouteProvider,
)
from vehicle_nmpc.utils.validation import require_positive

_FIGURE_EIGHT_ARC_SAMPLES = 8193


@register_trajectory("figure_eight")
class FigureEightTrajectoryProvider(ConstantSpeedRouteProvider):
    """Lemniscate of Gerono trajectory provider."""

    @dataclass(frozen=True, kw_only=True, slots=True)
    class Config(ConstantSpeedRouteConfig):
        """Figure-eight trajectory configuration."""

        scale: float = 4.0
        """Figure-eight horizontal scale."""

        def __post_init__(self) -> None:
            """Validate figure-eight trajectory parameters."""
            ConstantSpeedRouteConfig.__post_init__(self)
            require_positive("scale", self.scale)

    def __init__(self, cfg: Config, *args: object, **kwargs: object) -> None:
        """Precompute figure-eight arc-length map."""
        super().__init__(cfg, *args, **kwargs)
        # One-lap phase samples for the parametric figure-eight curve.
        self._phase_grid = np.linspace(0.0, 2.0 * np.pi, _FIGURE_EIGHT_ARC_SAMPLES)
        arc_rate = self._arc_rate(self._phase_grid)
        arc_segment = 0.5 * (arc_rate[1:] + arc_rate[:-1]) * np.diff(self._phase_grid)
        # Arc length accumulated at each phase sample, used for phase lookup by progress.
        self._arc_grid = np.concatenate(([0.0], np.cumsum(arc_segment)))
        # Total arc length of one figure-eight lap.
        self._lap_length = float(self._arc_grid[-1])

    def _pose_at_progress(self, progress_values: np.ndarray) -> np.ndarray:
        """Return figure-eight route poses by progress."""
        phase = self._phase_at(progress_values)
        a = self._cfg.scale
        sin_phase = np.sin(phase)
        cos_phase = np.cos(phase)
        x_axis = a * sin_phase
        y_axis = a * sin_phase * cos_phase
        yaw = np.arctan2(np.cos(2.0 * phase), cos_phase)
        return np.column_stack((x_axis, y_axis, yaw))

    def _curvature_at_progress(self, progress_values: np.ndarray) -> np.ndarray:
        """Return figure-eight route curvature by progress."""
        phase = self._phase_at(progress_values)
        a = self._cfg.scale
        dx = a * np.cos(phase)
        dy = a * np.cos(2.0 * phase)
        ddx = -a * np.sin(phase)
        ddy = -2.0 * a * np.sin(2.0 * phase)
        return (dx * ddy - dy * ddx) / np.power(np.hypot(dx, dy), 3.0)

    def _phase_at(self, progress_values: np.ndarray) -> np.ndarray:
        """Convert route progress to figure-eight phase."""
        local_arc = np.mod(progress_values, self._lap_length)
        return np.interp(local_arc, self._arc_grid, self._phase_grid)

    def _arc_rate(self, phase: np.ndarray) -> np.ndarray:
        """Return d(progress)/d(phase)."""
        a = self._cfg.scale
        return a * np.hypot(np.cos(phase), np.cos(2.0 * phase))

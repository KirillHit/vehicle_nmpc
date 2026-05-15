"""Built-in tracked vehicle trajectory providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from vehicle_nmpc.trajectory.base import BaseTrajectoryConfig, BaseTrajectoryProvider
from vehicle_nmpc.trajectory.builder import register_trajectory
from vehicle_nmpc.utils.validation import require_positive

if TYPE_CHECKING:
    from vehicle_nmpc.controller import TrackingReference


@dataclass(frozen=True, kw_only=True, slots=True)
class ConstantSpeedConfig(BaseTrajectoryConfig):
    """Base configuration for constant-speed geometric trajectories."""

    speed: float = 0.5
    """Nominal longitudinal speed."""


@register_trajectory("straight")
class StraightTrajectoryProvider(BaseTrajectoryProvider):
    """Straight-line trajectory provider."""

    @dataclass(frozen=True, kw_only=True, slots=True)
    class Config(ConstantSpeedConfig):
        """Straight-line trajectory configuration."""

        heading: float = 0.0
        """Reference heading angle."""

    def reference_at(self, step: int) -> TrackingReference:
        """Return a straight-line tracking reference horizon."""
        times = self._times(step)
        speed = np.full(self._prediction_steps + 1, self._cfg.speed)
        yaw_rate = np.zeros(self._prediction_steps + 1)
        heading = np.full(self._prediction_steps + 1, self._cfg.heading)
        x_ref = np.column_stack(
            (
                self._cfg.speed * times * np.cos(self._cfg.heading),
                self._cfg.speed * times * np.sin(self._cfg.heading),
                heading,
                speed,
                np.zeros(self._prediction_steps + 1),
                yaw_rate,
            )
        )
        return self._tracking_reference(x_ref)


@register_trajectory("turn")
class TurnTrajectoryProvider(BaseTrajectoryProvider):
    """Constant-curvature turn trajectory provider."""

    _STRAIGHT_CURVATURE_TOL: ClassVar[float] = 1e-9
    """Curvature threshold below which the arc is treated as a straight line."""

    @dataclass(frozen=True, kw_only=True, slots=True)
    class Config(ConstantSpeedConfig):
        """Constant-curvature turn trajectory configuration."""

        curvature: float = 0.4
        """Signed path curvature."""

    def reference_at(self, step: int) -> TrackingReference:
        """Return a constant-curvature tracking reference horizon."""
        times = self._times(step)
        pose_ref = self._integrate_curvature(times)
        speed = np.full(self._prediction_steps + 1, self._cfg.speed)
        yaw_rate = np.full(self._prediction_steps + 1, self._cfg.speed * self._cfg.curvature)
        x_ref = np.column_stack(
            (
                pose_ref,
                speed,
                np.zeros(self._prediction_steps + 1),
                yaw_rate,
            )
        )
        return self._tracking_reference(x_ref)

    def _integrate_curvature(self, times: np.ndarray) -> np.ndarray:
        """Return poses for the configured constant-curvature path."""
        path_length = self._cfg.speed * times
        yaw = self._cfg.curvature * path_length
        if abs(self._cfg.curvature) < self._STRAIGHT_CURVATURE_TOL:
            x_axis = path_length
            y_axis = np.zeros_like(path_length)
        else:
            x_axis = np.sin(yaw) / self._cfg.curvature
            y_axis = (1.0 - np.cos(yaw)) / self._cfg.curvature
        return np.column_stack((x_axis, y_axis, yaw))


@register_trajectory("s_curve")
class SCurveTrajectoryProvider(BaseTrajectoryProvider):
    """S-shaped sinusoidal path trajectory provider."""

    @dataclass(frozen=True, kw_only=True, slots=True)
    class Config(ConstantSpeedConfig):
        """S-shaped trajectory configuration."""

        amplitude: float = 0.8
        """Lateral sine-wave amplitude."""

        wavelength: float = 8.0
        """Longitudinal sine-wave wavelength."""

    def reference_at(self, step: int) -> TrackingReference:
        """Return an S-curve tracking reference horizon."""
        times = self._times(step)
        x_axis = self._cfg.speed * times
        wave = 2.0 * np.pi * x_axis / self._cfg.wavelength
        y_axis = self._cfg.amplitude * np.sin(wave)
        dy_dx = self._cfg.amplitude * 2.0 * np.pi / self._cfg.wavelength * np.cos(wave)
        yaw = np.arctan(dy_dx)
        curvature = self._path_curvature(x_axis[:-1])
        yaw_rate = self._cfg.speed * curvature
        speed = np.full(self._prediction_steps + 1, self._cfg.speed)
        yaw_rate_nodes = np.concatenate((yaw_rate, yaw_rate[-1:]))
        x_ref = np.column_stack(
            (
                x_axis,
                y_axis,
                yaw,
                speed,
                np.zeros(self._prediction_steps + 1),
                yaw_rate_nodes,
            )
        )
        return self._tracking_reference(x_ref)

    def _path_curvature(self, x_axis: np.ndarray) -> np.ndarray:
        """Return curvature for the configured sine path."""
        wave_number = 2.0 * np.pi / self._cfg.wavelength
        wave = wave_number * x_axis
        dy_dx = self._cfg.amplitude * wave_number * np.cos(wave)
        d2y_dx2 = -self._cfg.amplitude * wave_number * wave_number * np.sin(wave)
        return d2y_dx2 / np.power(1.0 + dy_dx * dy_dx, 1.5)


@register_trajectory("figure_eight")
class FigureEightTrajectoryProvider(BaseTrajectoryProvider):
    """Lemniscate of Gerono: x = a·sin(t), y = a·sin(t)·cos(t)."""

    @dataclass(frozen=True, kw_only=True, slots=True)
    class Config(BaseTrajectoryConfig):
        """Figure-eight trajectory configuration."""

        scale: float = 4.0
        """Figure-eight horizontal scale."""

        average_speed: float = 0.5
        """Nominal average path speed."""

    def reference_at(self, step: int) -> TrackingReference:
        """Return a full figure-eight tracking reference horizon."""
        times = self._times(step)
        a = self._cfg.scale

        phase_rate = self._cfg.average_speed / a
        t = times * phase_rate

        sin_t = np.sin(t)
        cos_t = np.cos(t)
        cos_2t = np.cos(2 * t)

        x = a * sin_t
        y = a * sin_t * cos_t
        yaw = np.unwrap(np.arctan2(cos_2t, cos_t))

        path_gain = a * np.hypot(cos_t[:-1], cos_2t[:-1])
        speed = path_gain * phase_rate
        yaw_rate = np.diff(yaw) / self._dt
        speed_nodes = np.concatenate((speed, speed[-1:]))
        yaw_rate_nodes = np.concatenate((yaw_rate, yaw_rate[-1:]))

        x_ref = np.column_stack(
            (x, y, yaw, speed_nodes, np.zeros_like(speed_nodes), yaw_rate_nodes)
        )
        return self._tracking_reference(x_ref)


@register_trajectory("square")
class SquareTrajectoryProvider(BaseTrajectoryProvider):
    """Square trajectory with straight edges and spot turns."""

    @dataclass(frozen=True, kw_only=True, slots=True)
    class Config(BaseTrajectoryConfig):
        """Square trajectory configuration."""

        straight_speed: float = 0.5
        straight_length: float = 2.0
        turn_speed: float = 1.0
        deceleration: float = 0.5

        def __post_init__(self) -> None:
            """Validate square trajectory parameters."""
            require_positive("straight_speed", self.straight_speed)
            require_positive("straight_length", self.straight_length)
            require_positive("turn_speed", self.turn_speed)
            require_positive("deceleration", self.deceleration)

    def reference_at(self, step: int) -> TrackingReference:
        """Return a square stop-turn-go tracking reference horizon."""
        cfg = self._cfg
        times = self._times(step)
        v_peak = min(cfg.straight_speed, np.sqrt(2.0 * cfg.deceleration * cfg.straight_length))
        brake_time = v_peak / cfg.deceleration
        brake_dist = v_peak * v_peak / (2.0 * cfg.deceleration)
        cruise_dist = max(cfg.straight_length - brake_dist, 0.0)
        cruise_time = cruise_dist / v_peak
        straight_time = cruise_time + brake_time
        turn_time = (0.5 * np.pi) / cfg.turn_speed
        cycle_time = straight_time + turn_time
        corners = cfg.straight_length * np.array(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))

        def sample(time: float) -> tuple[tuple[float, float, float], float, float]:
            segment = int(time // cycle_time)
            local_time = time - segment * cycle_time
            side = segment % 4
            heading = segment * 0.5 * np.pi
            yaw = heading
            start_x, start_y = corners[side]

            if local_time < cruise_time:
                distance = v_peak * local_time
                speed = v_peak
                yaw_rate = 0.0
            elif local_time < straight_time:
                t_brake = local_time - cruise_time
                distance = cruise_dist + v_peak * t_brake
                distance -= 0.5 * cfg.deceleration * t_brake * t_brake
                speed = max(v_peak - cfg.deceleration * t_brake, 0.0)
                yaw_rate = 0.0
            else:
                distance = cfg.straight_length
                speed = 0.0
                yaw_rate = cfg.turn_speed
                yaw += yaw_rate * min(local_time - straight_time, turn_time)

            x_axis = start_x + distance * np.cos(heading)
            y_axis = start_y + distance * np.sin(heading)
            return (x_axis, y_axis, yaw), speed, yaw_rate

        poses, speed, yaw_rate = zip(*(sample(float(time)) for time in times), strict=True)
        speed_array = np.array(speed)
        yaw_rate_array = np.array(yaw_rate)
        x_ref = np.column_stack(
            (
                np.array(poses),
                speed_array,
                np.zeros(self._prediction_steps + 1),
                yaw_rate_array,
            )
        )
        return self._tracking_reference(x_ref)

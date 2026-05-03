"""Built-in tracked vehicle trajectory providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from vehicle_nmpc.controller import TrackingReference
from vehicle_nmpc.trajectory.base import BaseTrajectoryConfig, BaseTrajectoryProvider
from vehicle_nmpc.trajectory.builder import register_trajectory


@dataclass(kw_only=True, slots=True)
class ConstantSpeedConfig(BaseTrajectoryConfig):
    """Base configuration for constant-speed geometric trajectories."""

    speed: float = 0.5
    """Nominal longitudinal speed."""


@register_trajectory("straight")
class StraightTrajectoryProvider(BaseTrajectoryProvider):
    """Straight-line trajectory provider."""

    @dataclass(kw_only=True, slots=True)
    class Config(ConstantSpeedConfig):
        """Straight-line trajectory configuration."""

        heading: float = 0.0
        """Reference heading angle."""

    def reference_at(self, step: int) -> TrackingReference:
        """Return a straight-line tracking reference horizon."""
        times = self._times(step)
        speed = np.full(self._prediction_steps + 1, self._cfg.speed)
        heading = np.full(self._prediction_steps + 1, self._cfg.heading)
        x_ref = np.column_stack(
            (
                self._cfg.speed * times * np.cos(self._cfg.heading),
                self._cfg.speed * times * np.sin(self._cfg.heading),
                heading,
            )
        )
        u_ref = self._control_reference(speed[:-1], np.zeros(self._prediction_steps))
        return TrackingReference(x=x_ref, u=u_ref)


@register_trajectory("turn")
class TurnTrajectoryProvider(BaseTrajectoryProvider):
    """Constant-curvature turn trajectory provider."""

    _STRAIGHT_CURVATURE_TOL: ClassVar[float] = 1e-9
    """Curvature threshold below which the arc is treated as a straight line."""

    @dataclass(kw_only=True, slots=True)
    class Config(ConstantSpeedConfig):
        """Constant-curvature turn trajectory configuration."""

        curvature: float = 0.4
        """Signed path curvature."""

    def reference_at(self, step: int) -> TrackingReference:
        """Return a constant-curvature tracking reference horizon."""
        times = self._times(step)
        x_ref = self._integrate_curvature(times)
        yaw_rate = np.full(self._prediction_steps, self._cfg.speed * self._cfg.curvature)
        speed = np.full(self._prediction_steps, self._cfg.speed)
        u_ref = self._control_reference(speed, yaw_rate)
        return TrackingReference(x=x_ref, u=u_ref)

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

    @dataclass(kw_only=True, slots=True)
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
        speed = np.full(self._prediction_steps, self._cfg.speed)
        u_ref = self._control_reference(speed, yaw_rate)
        return TrackingReference(x=np.column_stack((x_axis, y_axis, yaw)), u=u_ref)

    def _path_curvature(self, x_axis: np.ndarray) -> np.ndarray:
        """Return curvature for the configured sine path."""
        wave_number = 2.0 * np.pi / self._cfg.wavelength
        wave = wave_number * x_axis
        dy_dx = self._cfg.amplitude * wave_number * np.cos(wave)
        d2y_dx2 = -self._cfg.amplitude * wave_number * wave_number * np.sin(wave)
        return d2y_dx2 / np.power(1.0 + dy_dx * dy_dx, 1.5)


@register_trajectory("variable_curvature")
class VariableCurvatureTrajectoryProvider(BaseTrajectoryProvider):
    """Trajectory with time-varying curvature."""

    @dataclass(kw_only=True, slots=True)
    class Config(ConstantSpeedConfig):
        """Variable-curvature trajectory configuration."""

        curvature_amplitude: float = 0.45
        """Amplitude of the signed curvature signal."""

        curvature_period: float = 6.0
        """Curvature oscillation period in seconds."""

    def reference_at(self, step: int) -> TrackingReference:
        """Return a variable-curvature tracking reference horizon."""
        horizon_end_step = step + self._prediction_steps
        times = self._dt * np.arange(horizon_end_step + 1)
        curvature = self._curvature(times)
        x_ref = self._integrate_curvature(times, curvature)[step : horizon_end_step + 1]
        stage_curvature = curvature[:-1]
        stage_curvature = stage_curvature[step:horizon_end_step]
        yaw_rate = self._cfg.speed * stage_curvature
        speed = np.full(self._prediction_steps, self._cfg.speed)
        u_ref = self._control_reference(speed, yaw_rate)
        return TrackingReference(x=x_ref, u=u_ref)

    def _curvature(self, times: np.ndarray) -> np.ndarray:
        """Return signed curvature over time."""
        return self._cfg.curvature_amplitude * np.sin(
            2.0 * np.pi * times / self._cfg.curvature_period
        )

    def _integrate_curvature(self, times: np.ndarray, curvature: np.ndarray) -> np.ndarray:
        """Integrate the configured variable-curvature path."""
        x_ref = np.zeros((times.size, 3), dtype=float)
        for idx in range(1, times.size):
            dt = times[idx] - times[idx - 1]
            yaw_rate = self._cfg.speed * curvature[idx - 1]
            yaw_mid = x_ref[idx - 1, 2] + 0.5 * yaw_rate * dt
            x_ref[idx, 0] = x_ref[idx - 1, 0] + self._cfg.speed * np.cos(yaw_mid) * dt
            x_ref[idx, 1] = x_ref[idx - 1, 1] + self._cfg.speed * np.sin(yaw_mid) * dt
            x_ref[idx, 2] = x_ref[idx - 1, 2] + yaw_rate * dt
        return x_ref

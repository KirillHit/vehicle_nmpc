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


@register_trajectory("figure_eight")
class FigureEightTrajectoryProvider(BaseTrajectoryProvider):
    """Lemniscate of Gerono: x = a·sin(t), y = a·sin(t)·cos(t)."""

    @dataclass(kw_only=True, slots=True)
    class Config(ConstantSpeedConfig):
        scale: float = 4.0

    def reference_at(self, step: int) -> TrackingReference:
        times = self._times(step)
        a = self._cfg.scale
        t_dense = np.linspace(0, 2 * np.pi, 2000)
        ds = np.sqrt(np.cos(t_dense)**2 + np.cos(2*t_dense)**2) * a * (t_dense[1] - t_dense[0])
        s_dense = np.cumsum(np.insert(ds[:-1], 0, 0))
        s_horizon = (step * self._dt + times - times[0]) * self._cfg.speed
        t = np.interp(np.mod(s_horizon, s_dense[-1]), s_dense, t_dense)
        sin_t, cos_t = np.sin(t), np.cos(t)
        cos_2t = np.cos(2 * t)
        x, y = a * sin_t, a * sin_t * cos_t
        yaw = np.arctan2(a * cos_2t, a * cos_t)
        dx, dy = a * cos_t, a * cos_2t
        ddx, ddy = -a * sin_t, -2 * a * np.sin(2 * t)
        curvature = (dx[:-1] * ddy[:-1] - dy[:-1] * ddx[:-1]) / np.maximum((dx[:-1]**2 + dy[:-1]**2)**1.5, 1e-6)
        speed = np.full(self._prediction_steps, self._cfg.speed)
        u_ref = self._control_reference(speed, speed * curvature)
        return TrackingReference(x=np.column_stack((x, y, yaw)), u=u_ref)


@register_trajectory("saw")
class SawTrajectoryProvider(BaseTrajectoryProvider):
    """Straight segments alternating with spot turns."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseTrajectoryConfig):
        straight_speed: float = 0.5
        straight_length: float = 2.0
        turn_angle: float = np.pi / 3
        turn_speed: float = 1.0
        initial_heading: float = 0.0

    def reference_at(self, step: int) -> TrackingReference:
        times = self._times(step)
        N = self._prediction_steps
        v, L = self._cfg.straight_speed, self._cfg.straight_length
        w, dtheta = self._cfg.turn_speed, self._cfg.turn_angle
        t_s, t_t = L / v, abs(dtheta) / w
        t_seg = t_s + t_t

        def state_at(t):
            """x, y, heading at time t."""
            n = int(t // t_seg)
            dt = t - n * t_seg
            x = n * L * np.cos(self._cfg.initial_heading)
            y = n * L * np.sin(self._cfg.initial_heading)
            h = self._cfg.initial_heading + n * dtheta
            if dt < t_s:
                x += v * dt * np.cos(h)
                y += v * dt * np.sin(h)
            else:
                x += L * np.cos(h)
                y += L * np.sin(h)
                h += np.sign(dtheta) * w * (dt - t_s)
            return x, y, h

        poses = np.array([state_at(t) for t in times])
        dt = self._dt
        # Скорости посередине интервалов
        t_mid = times[:-1] + dt / 2
        in_straight = np.mod(t_mid, t_seg) < t_s
        speed = np.where(in_straight, v, 0.0)
        yaw_rate = np.where(in_straight, 0.0, np.sign(dtheta) * w)

        u_ref = self._control_reference(speed, yaw_rate)
        return TrackingReference(x=poses, u=u_ref)
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
        """Figure-eight trajectory configuration."""

        scale: float = 4.0
        lap_time: float = 8.0

    def reference_at(self, step: int) -> TrackingReference:
        times = self._times(step)
        a = self._cfg.scale
        
        phase = 2 * np.pi * step * self._dt * self._cfg.speed / (4 * a)
        t = phase + times * self._cfg.speed / a
        
        sin_t = np.sin(t)
        cos_t = np.cos(t)
        cos_2t = np.cos(2 * t)
        
        x = a * sin_t
        y = a * sin_t * cos_t
        yaw = np.arctan2(cos_2t, cos_t)
        
        speed = np.full(self._prediction_steps, self._cfg.speed)
        yaw_rate = np.zeros(self._prediction_steps)  
        u_ref = self._control_reference(speed, yaw_rate) 
        
        return TrackingReference(x=np.column_stack((x, y, yaw)), u=u_ref)


@register_trajectory("saw")
class SawTrajectoryProvider(BaseTrajectoryProvider):
    """Straight segments alternating with spot turns."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseTrajectoryConfig):
        """Saw trajectory configuration."""

        straight_speed: float = 0.5
        straight_length: float = 2.0
        turn_angle: float = np.pi / 3
        turn_speed: float = 1.0
        initial_heading: float = 0.0

    def reference_at(self, step: int) -> TrackingReference:
        """Return a saw-tooth tracking reference horizon."""
        times = self._times(step)
        v = self._cfg.straight_speed
        theta = self._cfg.turn_angle
        base_length = self._cfg.straight_length

        # diagonal segment length along the reference path
        half_base = base_length / 2.0
        diag_length = half_base / np.cos(theta)
        path_period = 2.0 * diag_length / v

        def state_at(t: float) -> tuple[float, float, float]:
            """x, y, heading at time t."""
            tooth_index = int(t // path_period)
            tau = t - tooth_index * path_period
            x0 = tooth_index * base_length
            if tau < diag_length / v:
                dist = v * tau
                x = x0 + dist * np.cos(theta)
                y = dist * np.sin(theta)
                h = self._cfg.initial_heading + theta
            else:
                dist = v * (tau - diag_length / v)
                x = x0 + half_base + dist * np.cos(theta)
                y = half_base * np.tan(theta) - dist * np.sin(theta)
                h = self._cfg.initial_heading - theta
            if self._cfg.initial_heading != 0.0:
                c, s = np.cos(self._cfg.initial_heading), np.sin(self._cfg.initial_heading)
                x_rot = c * x - s * y
                y_rot = s * x + c * y
                x, y = x_rot, y_rot
            return x, y, h

        poses = np.array([state_at(t) for t in times])
        yaw = poses[:, 2]
        dt = self._dt
        speed = np.full(self._prediction_steps, v)
        yaw_rate = np.diff(yaw) / dt

        u_ref = self._control_reference(speed, yaw_rate)
        return TrackingReference(x=poses, u=u_ref)

@register_trajectory("straight_dyn")
class StraightDynTrajectoryProvider(BaseTrajectoryProvider):
    @dataclass(kw_only=True, slots=True)
    class Config(ConstantSpeedConfig):
        heading: float = 0.0

    def reference_at(self, step: int) -> TrackingReference:
        times = self._times(step)
        v = self._cfg.speed
        h = self._cfg.heading
        
        x = v * times * np.cos(h)
        y = v * times * np.sin(h)
        
        states = np.zeros((self._prediction_steps + 1, 6))
        states[:, 0] = x
        states[:, 1] = y
        states[:, 2] = h
        states[:, 3] = v
        
        B = self._reference_model.track_width
        r = self._reference_model.sprocket_radius
        slip_l = self._reference_model.left_slip
        slip_r = self._reference_model.right_slip
        
        omega = v / (r * (1 - slip_l))
        u_ref = np.column_stack((np.full(self._prediction_steps, omega), np.full(self._prediction_steps, omega)))
        
        return TrackingReference(x=states, u=u_ref)
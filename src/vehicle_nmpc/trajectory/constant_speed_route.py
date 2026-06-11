"""Template for smooth route trajectories with a constant-speed profile."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Final

import numpy as np

from vehicle_nmpc.trajectory.base import BaseTrajectoryConfig, BaseTrajectoryProvider
from vehicle_nmpc.utils.validation import as_vector, require_positive

if TYPE_CHECKING:
    from vehicle_nmpc.controller import TrackingReference


_PROJECTION_SAMPLES: Final = 121


@dataclass(frozen=True, kw_only=True, slots=True)
class ConstantSpeedRouteConfig(BaseTrajectoryConfig):
    """Base configuration for smooth route trajectories with constant speed."""

    speed: float = 0.5
    """Nominal longitudinal speed."""

    def __post_init__(self) -> None:
        """Validate trajectory parameters."""
        BaseTrajectoryConfig.__post_init__(self)
        require_positive("speed", self.speed)


class ConstantSpeedRouteProvider(BaseTrajectoryProvider):
    """Smooth route template with state projection and a constant-speed horizon."""

    Config: ClassVar[type[ConstantSpeedRouteConfig]] = ConstantSpeedRouteConfig

    def __init__(self, cfg: ConstantSpeedRouteConfig, *args: object, **kwargs: object) -> None:
        """Initialize route progress state."""
        super().__init__(cfg, *args, **kwargs)
        self._route_progress = 0.0

    def reset(self) -> None:
        """Reset route progress before a new rollout."""
        self._route_progress = 0.0

    def reference_at(self, state: np.ndarray) -> TrackingReference:
        """Return a route tracking reference horizon near the current vehicle state."""
        state_array = as_vector("trajectory.state", state, self._model.nx)
        self._route_progress = self._project_progress(state_array[:2])
        lead = self._cfg.speed * self._dt * np.arange(self._prediction_steps + 1)
        return self._reference_at_progress(self._route_progress + lead)

    def initial_state(self) -> np.ndarray:
        """Return the model-sized initial state implied by route start geometry."""
        x_ref = self._canonical_at_progress(np.array([0.0]))[0]
        return self._canonical_initial_state(x_ref)

    def _project_progress(self, position: np.ndarray) -> float:
        """Project a position to a monotonic local route progress."""
        progress_min = self._route_progress
        progress_max = self._route_progress + self._cfg.projection_window
        candidates = np.linspace(progress_min, progress_max, _PROJECTION_SAMPLES)
        poses = self._pose_at_progress(candidates)
        distances = np.linalg.norm(poses[:, :2] - position, axis=1)
        projected = float(candidates[int(np.argmin(distances))])
        if projected > self._route_progress + self._max_progress_delta():
            return self._route_progress
        return projected

    def _max_progress_delta(self) -> float:
        """Return the maximum allowed progress update for one closed-loop step."""
        return self._cfg.max_progress_per_step * self._cfg.speed * self._dt

    def _reference_at_progress(self, progress_values: np.ndarray) -> TrackingReference:
        """Return a tracking reference horizon sampled by route progress."""
        return self._tracking_reference(self._canonical_at_progress(progress_values))

    def _canonical_at_progress(self, progress_values: np.ndarray) -> np.ndarray:
        """Return canonical states sampled by route progress."""
        pose = self._pose_at_progress(progress_values)
        pose[:, 2] = np.unwrap(pose[:, 2])
        speed = np.full(progress_values.shape, self._cfg.speed)
        yaw_rate = self._cfg.speed * self._curvature_at_progress(progress_values)
        return np.column_stack((pose, speed, np.zeros_like(speed), yaw_rate))

    @abstractmethod
    def _pose_at_progress(self, progress_values: np.ndarray) -> np.ndarray:
        """Return route pose samples for route progress values."""
        raise NotImplementedError

    @abstractmethod
    def _curvature_at_progress(self, progress_values: np.ndarray) -> np.ndarray:
        """Return route curvature samples for route progress values."""
        raise NotImplementedError

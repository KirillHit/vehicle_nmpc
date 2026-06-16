"""Square trajectory provider."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

from vehicle_nmpc.trajectory.base import BaseTrajectoryConfig, BaseTrajectoryProvider
from vehicle_nmpc.trajectory.builder import register_trajectory
from vehicle_nmpc.utils.validation import as_vector, require_positive

if TYPE_CHECKING:
    from vehicle_nmpc.controller import TrackingReference


_DYNAMIC_STATE_SIZE = 6


class _SquarePhase(Enum):
    """Runtime phase of the square trajectory state machine."""

    LINE = auto()
    STOP = auto()
    TURN = auto()


def _wrap_angle(angle: float) -> float:
    """Wrap an angle error to [-pi, pi]."""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _speed_from_state(state: np.ndarray) -> float:
    """Return dynamic-model body speed, or zero for models without speed states."""
    if state.size != _DYNAMIC_STATE_SIZE:
        return 0.0
    return float(np.hypot(state[3], state[4]))


@register_trajectory("square")
class SquareTrajectoryProvider(BaseTrajectoryProvider):
    """Square trajectory with state-driven line, stop, and spot-turn phases."""

    @dataclass(frozen=True, kw_only=True, slots=True)
    class Config(BaseTrajectoryConfig):
        """Square trajectory configuration."""

        straight_speed: float = 0.5
        min_line_speed: float = 0.15
        straight_length: float = 2.0
        turn_speed: float = 1.0
        deceleration: float = 0.5
        stop_position_tolerance: float = 0.1
        stop_speed_tolerance: float = 0.05
        turn_yaw_tolerance: float = 0.05

        def __post_init__(self) -> None:
            """Validate square trajectory parameters."""
            BaseTrajectoryConfig.__post_init__(self)
            require_positive("straight_speed", self.straight_speed)
            require_positive("min_line_speed", self.min_line_speed)
            require_positive("straight_length", self.straight_length)
            require_positive("turn_speed", self.turn_speed)
            require_positive("deceleration", self.deceleration)
            require_positive("stop_position_tolerance", self.stop_position_tolerance)
            require_positive("stop_speed_tolerance", self.stop_speed_tolerance)
            require_positive("turn_yaw_tolerance", self.turn_yaw_tolerance)
            if self.min_line_speed > self.straight_speed:
                msg = (
                    "min_line_speed must be less than or equal to "
                    f"straight_speed, got {self.min_line_speed} > {self.straight_speed}."
                )
                raise ValueError(msg)

    def __init__(self, cfg: Config, *args: object, **kwargs: object) -> None:
        """Initialize square route state."""
        super().__init__(cfg, *args, **kwargs)
        self.reset()

    def reset(self) -> None:
        """Reset square phase state before a new rollout."""
        self._side = 0
        self._phase = _SquarePhase.LINE
        self._line_progress = 0.0
        self._turn_progress = 0.0

    def reference_at(self, state: np.ndarray) -> TrackingReference:
        """Return a square tracking reference horizon from the current vehicle state."""
        state_array = as_vector("trajectory.state", state, self._model.nx)
        self._update_phase(state_array)
        x_ref = [self._sample_horizon_node(i) for i in range(self._prediction_steps + 1)]
        return self._tracking_reference(np.array(x_ref))

    def initial_state(self) -> np.ndarray:
        """Return the model-sized initial state implied by the first square edge."""
        x_ref = self._line_state(self._side, 0.0, self._cfg.straight_speed)
        return self._canonical_initial_state(x_ref)

    def _update_phase(self, state: np.ndarray) -> None:
        """Advance square phases using measured position, speed, and yaw."""
        match self._phase:
            case _SquarePhase.LINE:
                self._update_line_phase(state)
            case _SquarePhase.STOP:
                self._update_stop_phase(state)
            case _SquarePhase.TURN:
                self._update_turn_phase(state)

    def _update_line_phase(self, state: np.ndarray) -> None:
        """Update progress on the active edge and switch to stop at the corner."""
        self._line_progress = self._project_on_current_edge(state[:2])
        if self._line_progress >= self._cfg.straight_length - self._cfg.stop_position_tolerance:
            self._phase = _SquarePhase.STOP
            self._line_progress = self._cfg.straight_length

    def _update_stop_phase(self, state: np.ndarray) -> None:
        """Wait at the corner until position and speed are close enough."""
        corner = self._corner(self._side + 1)
        near_corner = np.linalg.norm(state[:2] - corner) <= self._cfg.stop_position_tolerance
        stopped = _speed_from_state(state) <= self._cfg.stop_speed_tolerance
        if near_corner and stopped:
            self._phase = _SquarePhase.TURN
            self._turn_progress = 0.0

    def _update_turn_phase(self, state: np.ndarray) -> None:
        """Update spot-turn progress and switch to the next edge after alignment."""
        self._turn_progress = self._project_turn_progress(float(state[2]))
        turn_complete = self._turn_progress >= 0.5 * np.pi - self._cfg.turn_yaw_tolerance
        target_yaw = self._heading(self._side + 1)
        yaw_aligned = abs(_wrap_angle(float(state[2]) - target_yaw))
        if turn_complete and yaw_aligned <= self._cfg.turn_yaw_tolerance:
            self._side += 1
            self._phase = _SquarePhase.LINE
            self._line_progress = 0.0
            self._turn_progress = 0.0

    def _project_on_current_edge(self, position: np.ndarray) -> float:
        """Project the vehicle position to monotonic progress on the active square edge."""
        start = self._corner(self._side)
        direction = np.array((np.cos(self._heading(self._side)), np.sin(self._heading(self._side))))
        raw_progress = float(np.dot(position - start, direction))
        bounded = float(np.clip(raw_progress, self._line_progress, self._cfg.straight_length))
        max_step = self._cfg.max_progress_per_step * self._cfg.straight_speed * self._dt
        if bounded > self._line_progress + max_step:
            return self._line_progress
        return bounded

    def _project_turn_progress(self, yaw: float) -> float:
        """Project vehicle yaw to monotonic progress through the active spot turn."""
        raw_progress = _wrap_angle(yaw - self._heading(self._side))
        bounded = float(np.clip(raw_progress, self._turn_progress, 0.5 * np.pi))
        max_step = self._cfg.max_progress_per_step * self._cfg.turn_speed * self._dt
        if bounded > self._turn_progress + max_step:
            return self._turn_progress
        return bounded

    def _sample_horizon_node(self, node: int) -> np.ndarray:
        """Sample one horizon node without mutating the current square phase."""
        match self._phase:
            case _SquarePhase.LINE:
                progress = self._line_progress
                for _ in range(node):
                    speed = self._line_speed_at(progress)
                    progress = min(progress + speed * self._dt, self._cfg.straight_length)
                speed = self._line_speed_at(progress)
                return self._line_state(self._side, progress, speed)

            case _SquarePhase.STOP:
                return self._line_state(self._side, self._cfg.straight_length, 0.0)

            case _SquarePhase.TURN:
                turn_progress = min(
                    self._turn_progress + self._cfg.turn_speed * self._dt * node,
                    0.5 * np.pi,
                )
                yaw = self._heading(self._side) + turn_progress
                corner = self._corner(self._side + 1)
                return np.array((corner[0], corner[1], yaw, 0.0, 0.0, self._cfg.turn_speed))

        msg = f"Unsupported square phase: {self._phase!r}"
        raise AssertionError(msg)

    def _line_speed_at(self, progress: float) -> float:
        """Return the line speed that still allows stopping at the next corner."""
        distance_left = self._cfg.straight_length - progress
        braking_speed = np.sqrt(max(2.0 * self._cfg.deceleration * distance_left, 0.0))
        speed = min(self._cfg.straight_speed, float(braking_speed))
        if distance_left > self._cfg.stop_position_tolerance:
            speed = max(speed, self._cfg.min_line_speed)
        return speed

    def _line_state(self, side: int, progress: float, speed: float) -> np.ndarray:
        """Return one canonical state on a square edge."""
        start = self._corner(side)
        heading = self._heading(side)
        position = start + progress * np.array((np.cos(heading), np.sin(heading)))
        return np.array((position[0], position[1], heading, speed, 0.0, 0.0))

    def _corner(self, side: int) -> np.ndarray:
        """Return the square corner at an edge index."""
        length = self._cfg.straight_length
        corners = length * np.array(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
        return corners[side % 4]

    def _heading(self, side: int) -> float:
        """Return the heading for a square edge index."""
        return side * 0.5 * np.pi

"""Trajectory provider abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from vehicle_nmpc.utils.factory import ConfiguredBase

if TYPE_CHECKING:
    from vehicle_nmpc.controller import TrackingReference
    from vehicle_nmpc.models import ModelBundle
    from vehicle_nmpc.problem import ProblemBundle


@dataclass(kw_only=True, slots=True)
class BaseTrajectoryConfig:
    """Base configuration for trajectory providers."""


class BaseTrajectoryProvider(ConfiguredBase, ABC):
    """Abstract base class for tracking reference providers."""

    Config: ClassVar[type[BaseTrajectoryConfig]] = BaseTrajectoryConfig

    def __init__(
        self,
        cfg: BaseTrajectoryConfig,
        model: ModelBundle,
        problem: ProblemBundle,
    ) -> None:
        """Initialize trajectory provider for one model/problem pair."""
        super().__init__(cfg)
        self._model = model
        self._prediction_steps = int(problem.ocp.solver_options.N_horizon)
        self._dt = float(problem.ocp.solver_options.tf) / self._prediction_steps
        self._reference_model = model.trajectory_reference_model
        if self._reference_model is None:
            msg = f"Model '{model.model.name}' does not provide trajectory reference support."
            raise ValueError(msg)

    @property
    def name(self) -> str:
        """Human-readable trajectory name."""
        return self.__class__.__name__

    @abstractmethod
    def reference_at(self, step: int) -> TrackingReference:
        """Return a tracking reference horizon for a simulation step."""
        raise NotImplementedError

    def _times(self, step: int) -> np.ndarray:
        """Return horizon node times for a closed-loop simulation step."""
        start_time = step * self._dt
        return start_time + self._dt * np.arange(self._prediction_steps + 1)

    def _control_reference(self, speed: np.ndarray, yaw_rate: np.ndarray) -> np.ndarray:
        """Convert path speed and yaw rate references to model controls."""
        return self._reference_model.control_reference(speed, yaw_rate)

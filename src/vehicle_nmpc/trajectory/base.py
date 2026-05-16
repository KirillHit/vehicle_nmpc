"""Trajectory provider abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from vehicle_nmpc.controller.base import TrackingReference
from vehicle_nmpc.utils.factory import ConfiguredBase
from vehicle_nmpc.utils.validation import as_matrix, as_vector

if TYPE_CHECKING:
    from vehicle_nmpc.models import ModelBundle
    from vehicle_nmpc.problem import ProblemBundle


_CANONICAL_STATE_SIZE = 6


@dataclass(frozen=True, kw_only=True, slots=True)
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

    @property
    def name(self) -> str:
        """Human-readable trajectory name."""
        return self.__class__.__name__

    @abstractmethod
    def reference_at(self, step: int) -> TrackingReference:
        """Return a tracking reference horizon for a simulation step."""
        raise NotImplementedError

    def initial_state(self) -> np.ndarray:
        """Return the model-sized initial state implied by the trajectory."""
        reference = self.reference_at(0)
        return as_vector("trajectory.initial_state", reference.x[0], self._model.nx)

    def _times(self, step: int) -> np.ndarray:
        """Return horizon node times for a closed-loop simulation step."""
        start_time = step * self._dt
        return start_time + self._dt * np.arange(self._prediction_steps + 1)

    def _tracking_reference(
        self,
        x_ref: np.ndarray,
    ) -> TrackingReference:
        """Return model-sized tracking reference from canonical 6-state trajectory."""
        if self._model.nx > _CANONICAL_STATE_SIZE:
            msg = (
                "Canonical trajectory reference supports models with "
                f"nx <= {_CANONICAL_STATE_SIZE}, got nx={self._model.nx}."
            )
            raise ValueError(msg)

        canonical_x_ref = as_matrix(
            "reference.x",
            x_ref,
            (self._prediction_steps + 1, _CANONICAL_STATE_SIZE),
        )
        return TrackingReference(x=canonical_x_ref[:, : self._model.nx])

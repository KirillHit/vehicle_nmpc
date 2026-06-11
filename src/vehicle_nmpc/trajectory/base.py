"""Trajectory provider abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from vehicle_nmpc.controller.base import TrackingReference
from vehicle_nmpc.utils.factory import ConfiguredBase
from vehicle_nmpc.utils.validation import as_matrix, as_vector, require_positive

if TYPE_CHECKING:
    import numpy as np

    from vehicle_nmpc.models import ModelBundle
    from vehicle_nmpc.problem import ProblemBundle


_CANONICAL_STATE_SIZE = 6


@dataclass(frozen=True, kw_only=True, slots=True)
class BaseTrajectoryConfig:
    """Base configuration for trajectory providers."""

    projection_window: float = 2.0
    """Forward progress window used to project the current vehicle state."""

    max_progress_per_step: float = 3.0
    """Maximum progress update as a factor of nominal progress per closed-loop step."""

    def __post_init__(self) -> None:
        """Validate progress projection parameters."""
        require_positive("projection_window", self.projection_window)
        require_positive("max_progress_per_step", self.max_progress_per_step)


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
    def reference_at(self, state: np.ndarray) -> TrackingReference:
        """Return a tracking reference horizon from the current vehicle state."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset runtime trajectory progress before a new rollout."""

    @abstractmethod
    def initial_state(self) -> np.ndarray:
        """Return the model-sized initial state implied by the trajectory."""
        raise NotImplementedError

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

    def _canonical_initial_state(self, x_ref: np.ndarray) -> np.ndarray:
        """Return the model-sized first state from a canonical reference row."""
        canonical = as_vector("trajectory.initial_state", x_ref, _CANONICAL_STATE_SIZE)
        return canonical[: self._model.nx]

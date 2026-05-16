"""Problem abstractions for NMPC formulations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from acados_template import AcadosOcp

    from vehicle_nmpc.models import ModelBundle

from vehicle_nmpc.utils.factory import ConfiguredBase


@dataclass(frozen=True, kw_only=True, slots=True)
class BaseProblemConfig:
    """Base configuration for OCP problem implementations."""


@dataclass(kw_only=True, slots=True)
class ProblemBundle:
    """Container with the built problem and its metadata."""

    ocp: AcadosOcp

    @property
    def prediction_steps(self) -> int:
        """Number of shooting intervals in the prediction horizon."""
        return int(self.ocp.solver_options.N_horizon)

    @property
    def prediction_horizon_time(self) -> float:
        """Prediction horizon duration in seconds."""
        return float(self.ocp.solver_options.tf)

    @property
    def dt(self) -> float:
        """Control period implied by the prediction horizon."""
        return self.prediction_horizon_time / self.prediction_steps


class BaseProblem(ConfiguredBase, ABC):
    """Abstract problem interface."""

    Config: ClassVar[type[BaseProblemConfig]] = BaseProblemConfig

    def __init__(self, cfg: BaseProblemConfig, model: ModelBundle) -> None:
        """Initialize problem with configuration and model bundle."""
        super().__init__(cfg)
        self._model = model

    @abstractmethod
    def build(self) -> ProblemBundle:
        """Build and return the problem bundle."""
        raise NotImplementedError

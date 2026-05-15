"""Simulation abstractions for NMPC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import numpy as np

    from vehicle_nmpc.models import ModelBundle
    from vehicle_nmpc.problem import ProblemBundle

from vehicle_nmpc.utils.factory import ConfiguredBase


@dataclass(frozen=True, kw_only=True, slots=True)
class BaseSimulatorConfig:
    """Base configuration for simulator implementations."""


class BaseSimulator(ConfiguredBase, ABC):
    """Abstract simulator interface."""

    Config: ClassVar[type[BaseSimulatorConfig]] = BaseSimulatorConfig

    def __init__(
        self,
        cfg: BaseSimulatorConfig,
        problem: ProblemBundle,
        model: ModelBundle,
    ) -> None:
        """Initialize simulator with configuration, problem, and model bundles."""
        super().__init__(cfg)
        self._problem = problem
        self._model = model

    @abstractmethod
    def reset(self, x0: np.ndarray) -> None:
        """Reset simulator internal state for a new episode."""
        raise NotImplementedError

    @abstractmethod
    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Propagate dynamics for one step."""
        raise NotImplementedError

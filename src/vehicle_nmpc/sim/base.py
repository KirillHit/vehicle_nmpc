"""Simulation abstractions for NMPC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import numpy as np

    from vehicle_nmpc.models import ModelBundle
    from vehicle_nmpc.problem import ProblemBundle
    from vehicle_nmpc.utils.config import SimConfig

from vehicle_nmpc.utils.factory import ConfiguredBase


class BaseSimulator(ConfiguredBase, ABC):
    """Abstract simulator interface."""

    Config: ClassVar[type[SimConfig]]

    def __init__(self, cfg: SimConfig, problem: ProblemBundle, model: ModelBundle) -> None:
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

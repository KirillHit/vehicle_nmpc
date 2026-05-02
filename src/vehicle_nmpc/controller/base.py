"""Controller abstractions for NMPC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import numpy as np

    from vehicle_nmpc.models import ModelBundle
    from vehicle_nmpc.problem import ProblemBundle

from vehicle_nmpc.utils.factory import ConfiguredBase


@dataclass(kw_only=True, slots=True)
class BaseControllerConfig:
    """Base configuration for controller implementations."""


class BaseController(ConfiguredBase, ABC):
    """Abstract controller interface."""

    Config: ClassVar[type[BaseControllerConfig]] = BaseControllerConfig

    def __init__(
        self,
        cfg: BaseControllerConfig,
        problem: ProblemBundle,
        model: ModelBundle,
    ) -> None:
        """Initialize controller with configuration, problem, and model bundles."""
        super().__init__(cfg)
        self._problem = problem
        self._model = model

    @abstractmethod
    def reset(self, x0: np.ndarray) -> None:
        """Reset controller internal state for a new episode."""
        raise NotImplementedError

    @abstractmethod
    def solve(self, x: np.ndarray) -> np.ndarray:
        """Solve the control problem for the given state."""
        raise NotImplementedError

    @abstractmethod
    def get_stats(self) -> dict:
        """Return solver statistics for logging."""
        raise NotImplementedError

"""RTI NMPC controller implementation (skeleton)."""

from dataclasses import dataclass

import numpy as np

from vehicle_nmpc.controller.base import BaseController, BaseControllerConfig
from vehicle_nmpc.controller.builder import register_controller
from vehicle_nmpc.models import ModelBundle
from vehicle_nmpc.problem import ProblemBundle


@register_controller("rti_nmpc")
class RtiNmpcController(BaseController):
    """Real-time iteration NMPC controller."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseControllerConfig):
        """RTI NMPC controller configuration."""

    def __init__(self, cfg: Config, problem: ProblemBundle, model: ModelBundle) -> None:
        """Initialize RTI controller with configuration, problem, and model bundles."""
        super().__init__(cfg, problem, model)
        self._solver = None

    def reset(self, x0: np.ndarray) -> None:
        """Reset controller internal state for a new episode."""
        self._x0 = np.asarray(x0, dtype=float)

    def solve(self, x: np.ndarray) -> np.ndarray:
        """Solve the control problem for the given state."""
        _ = x
        if self._solver is None:
            msg = "RTI controller solver is not initialized. Build from problem bundle first."
            raise RuntimeError(msg)
        raise NotImplementedError

    def get_stats(self) -> dict:
        """Return solver statistics for logging."""
        return {}

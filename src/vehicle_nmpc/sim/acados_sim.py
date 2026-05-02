"""Acados simulator implementation (skeleton)."""

from dataclasses import dataclass

import numpy as np

from vehicle_nmpc.models import ModelBundle
from vehicle_nmpc.problem import ProblemBundle
from vehicle_nmpc.sim.base import BaseSimulator, BaseSimulatorConfig
from vehicle_nmpc.sim.builder import register_simulator


@register_simulator("acados_sim")
class AcadosSimulator(BaseSimulator):
    """Simulator wrapper around AcadosSimSolver."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseSimulatorConfig):
        """Acados simulator configuration."""

    def __init__(self, cfg: Config, problem: ProblemBundle, model: ModelBundle) -> None:
        """Initialize Acados simulator wrapper."""
        super().__init__(cfg, problem, model)
        self._sim = None

    def reset(self, x0: np.ndarray) -> None:
        """Reset simulator internal state for a new episode."""
        self._x0 = np.asarray(x0, dtype=float)

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Propagate dynamics for one step."""
        _ = x
        _ = u
        if self._sim is None:
            msg = "Acados simulator is not initialized. Build from problem bundle first."
            raise RuntimeError(msg)
        raise NotImplementedError

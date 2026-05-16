"""Acados simulator implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from acados_template import AcadosSim, AcadosSimSolver

from vehicle_nmpc.sim.base import BaseSimulator, BaseSimulatorConfig
from vehicle_nmpc.sim.builder import register_simulator
from vehicle_nmpc.utils.validation import as_vector

if TYPE_CHECKING:
    from vehicle_nmpc.models import ModelBundle
    from vehicle_nmpc.problem import ProblemBundle


@register_simulator("acados_sim")
class AcadosSimulator(BaseSimulator):
    """Simulator wrapper around AcadosSimSolver."""

    _SUCCESS_STATUS: ClassVar[int] = 0
    """Acados simulator status code for a successful integration step."""

    @dataclass(frozen=True, kw_only=True, slots=True)
    class Config(BaseSimulatorConfig):
        """Acados simulator configuration."""

    def __init__(self, cfg: Config, problem: ProblemBundle, model: ModelBundle) -> None:
        """Initialize Acados simulator wrapper."""
        super().__init__(cfg, problem, model)
        sim = AcadosSim.from_ocp(self._problem.ocp)
        if self._model.np > 0:
            sim.parameter_values = np.asarray(self._model.p0, dtype=float)
        self._solver = AcadosSimSolver(sim, verbose=False)

    def reset(self, x0: np.ndarray) -> None:
        """Reset simulator internal state for a new episode."""
        self._validate_state(x0)

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Propagate dynamics for one simulation step."""
        x_array = self._validate_state(x)
        u_array = self._validate_control(u)

        self._solver.set("x", x_array)
        self._solver.set("u", u_array)

        status = self._solver.solve()
        if status != self._SUCCESS_STATUS:
            msg = f"acados simulator returned status {status}."
            raise RuntimeError(msg)

        return self._validate_state(self._solver.get("x"))

    def _validate_state(self, x: np.ndarray) -> np.ndarray:
        """Convert and validate a state vector."""
        return as_vector("state", x, self._model.nx)

    def _validate_control(self, u: np.ndarray) -> np.ndarray:
        """Convert and validate a control vector."""
        return as_vector("control", u, self._model.nu)

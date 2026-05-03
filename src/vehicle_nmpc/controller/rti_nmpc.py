"""RTI NMPC controller implementation."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from acados_template import AcadosOcpSolver

from vehicle_nmpc.controller.base import BaseController, BaseControllerConfig
from vehicle_nmpc.controller.builder import register_controller
from vehicle_nmpc.models import ModelBundle
from vehicle_nmpc.problem import ProblemBundle
from vehicle_nmpc.utils.validation import as_vector


@register_controller("rti_nmpc")
class RtiNmpcController(BaseController):
    """Real-time iteration NMPC controller."""

    _NLP_SOLVER_TYPE: ClassVar[str] = "SQP_RTI"
    """Acados NLP solver type required for RTI control."""

    _ALLOWED_STATUSES: ClassVar[set[int]] = {0, 2, 5}
    """Acados statuses accepted as non-fatal during RTI phases."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseControllerConfig):
        """RTI NMPC controller configuration."""

    def __init__(self, cfg: Config, problem: ProblemBundle, model: ModelBundle) -> None:
        """Initialize RTI controller with configuration, problem, and model bundles."""
        super().__init__(cfg, problem, model)
        self._problem.ocp.solver_options.nlp_solver_type = self._NLP_SOLVER_TYPE
        self._solver = AcadosOcpSolver(self._problem.ocp, verbose=False)
        self._stats: dict[str, float | int] = {}

    def reset(self, x0: np.ndarray) -> None:
        """Reset controller internal state for a new episode."""
        x0_array = self._validate_state(x0)
        self._set_initial_state(x0_array)
        self._stats.clear()

    def solve(self, x: np.ndarray) -> np.ndarray:
        """Solve the control problem for the given state."""
        x_array = self._validate_state(x)

        self._solver.options_set("rti_phase", 1)
        preparation_status = self._solver.solve()
        preparation_time = self._solver.get_stats("time_tot")
        self._raise_for_status(preparation_status, "preparation")

        self._set_initial_state(x_array)

        self._solver.options_set("rti_phase", 2)
        feedback_status = self._solver.solve()
        feedback_time = self._solver.get_stats("time_tot")
        self._raise_for_status(feedback_status, "feedback")

        control = np.asarray(self._solver.get(0, "u"), dtype=float)
        self._stats = {
            "preparation_status": preparation_status,
            "feedback_status": feedback_status,
            "preparation_time": preparation_time,
            "feedback_time": feedback_time,
        }
        return control

    def get_stats(self) -> dict:
        """Return solver statistics for logging."""
        return dict(self._stats)

    def _set_initial_state(self, x: np.ndarray) -> None:
        """Fix the first shooting node to the measured state."""
        self._solver.set(0, "lbx", x)
        self._solver.set(0, "ubx", x)

    def _validate_state(self, x: np.ndarray) -> np.ndarray:
        """Convert and validate a state vector."""
        return as_vector("state", x, self._model.nx)

    def _raise_for_status(self, status: int, phase: str) -> None:
        """Raise if acados returned an unexpected status code."""
        if status in self._ALLOWED_STATUSES:
            return

        msg = f"acados RTI {phase} phase returned status {status}."
        raise RuntimeError(msg)

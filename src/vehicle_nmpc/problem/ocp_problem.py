"""Acados OCP problem formulation."""

from dataclasses import dataclass, field

import numpy as np
import scipy.linalg
from acados_template import AcadosOcp
from casadi import vertcat
from omegaconf import MISSING

from vehicle_nmpc.models import ModelBundle
from vehicle_nmpc.problem.base import BaseProblem, BaseProblemConfig, ProblemBundle
from vehicle_nmpc.problem.builder import register_problem
from vehicle_nmpc.utils.validation import as_vector


@register_problem("acados_ocp")
class AcadosOcpProblem(BaseProblem):
    """Builds an Acados OCP from a model bundle."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseProblemConfig):
        """Acados OCP problem configuration."""

        n_horizon: int = MISSING
        """Number of shooting intervals in the prediction horizon."""

        tf: float = MISSING
        """Prediction horizon duration in seconds."""

        q: list[float] = field(default_factory=list)
        """Diagonal state tracking weights."""

        r: list[float] = field(default_factory=list)
        """Diagonal control tracking weights."""

        yref: list[float] = field(default_factory=list)
        """Stage reference for stacked output [x, u]. Defaults to zeros."""

        yref_e: list[float] = field(default_factory=list)
        """Terminal reference for state output x. Defaults to zeros."""

        lbu: list[float] = field(default_factory=list)
        """Lower bounds for selected control components."""

        ubu: list[float] = field(default_factory=list)
        """Upper bounds for selected control components."""

        idxbu: list[int] = field(default_factory=list)
        """Indices of bounded control components."""

        cost_type: str = "NONLINEAR_LS"
        """Acados stage cost type."""

        cost_type_e: str = "NONLINEAR_LS"
        """Acados terminal cost type."""

        qp_solver: str = "PARTIAL_CONDENSING_HPIPM"
        """Acados QP solver."""

        hessian_approx: str = "GAUSS_NEWTON"
        """Hessian approximation method."""

        integrator_type: str = "IRK"
        """Integrator type used inside the OCP solver."""

        sim_method_newton_iter: int = 10
        """Number of Newton iterations for implicit simulation methods."""

        qp_solver_cond_n: int | None = None
        """Partial condensing horizon. Defaults to n_horizon."""

        translate_nls_cost_to_conl: bool = True
        """Translate nonlinear least-squares cost to convex-over-nonlinear cost."""

    def __init__(self, cfg: Config, model: ModelBundle) -> None:
        """Initialize OCP problem with configuration and model bundle."""
        super().__init__(cfg, model)

    def build(self) -> ProblemBundle:
        """Build and return the Acados OCP problem bundle."""
        ocp = AcadosOcp()
        ocp.model = self._model.model
        ocp.solver_options.N_horizon = self._cfg.n_horizon
        ocp.solver_options.tf = self._cfg.tf

        self._set_tracking_cost(ocp)
        self._set_control_bounds(ocp)
        self._set_solver_options(ocp)

        return ProblemBundle(ocp=ocp)

    def _set_tracking_cost(self, ocp: AcadosOcp) -> None:
        """Configure a generic tracking cost over states and controls."""
        nx = self._model.nx
        nu = self._model.nu
        ny = nx + nu

        q = as_vector("q", self._cfg.q, nx)
        r = as_vector("r", self._cfg.r, nu)

        ocp.cost.cost_type = self._cfg.cost_type
        ocp.cost.cost_type_e = self._cfg.cost_type_e
        ocp.cost.W = scipy.linalg.block_diag(np.diag(q), np.diag(r))
        ocp.cost.W_e = np.diag(q)

        ocp.model.cost_y_expr = vertcat(self._model.model.x, self._model.model.u)
        ocp.model.cost_y_expr_e = self._model.model.x
        ocp.cost.yref = as_vector("yref", self._cfg.yref, ny, default=0.0)
        ocp.cost.yref_e = as_vector("yref_e", self._cfg.yref_e, nx, default=0.0)

        if self._cfg.translate_nls_cost_to_conl:
            ocp.translate_nls_cost_to_conl()

    def _set_control_bounds(self, ocp: AcadosOcp) -> None:
        """Configure bounds for selected control components."""
        if not self._cfg.idxbu and not self._cfg.lbu and not self._cfg.ubu:
            return

        n_bounded_controls = len(self._cfg.idxbu)
        if n_bounded_controls == 0:
            msg = "Control bounds require non-empty idxbu."
            raise ValueError(msg)

        ocp.constraints.idxbu = as_vector(
            "idxbu",
            self._cfg.idxbu,
            n_bounded_controls,
            dtype=int,
        )
        ocp.constraints.lbu = as_vector("lbu", self._cfg.lbu, n_bounded_controls)
        ocp.constraints.ubu = as_vector("ubu", self._cfg.ubu, n_bounded_controls)

    def _set_solver_options(self, ocp: AcadosOcp) -> None:
        """Configure acados solver options shared by RTI and tracking problems."""
        ocp.constraints.x0 = np.asarray(self._model.x0, dtype=float)
        ocp.solver_options.qp_solver = self._cfg.qp_solver
        ocp.solver_options.hessian_approx = self._cfg.hessian_approx
        ocp.solver_options.integrator_type = self._cfg.integrator_type
        ocp.solver_options.sim_method_newton_iter = self._cfg.sim_method_newton_iter
        ocp.solver_options.qp_solver_cond_N = (
            self._cfg.qp_solver_cond_n
            if self._cfg.qp_solver_cond_n is not None
            else self._cfg.n_horizon
        )

"""Acados OCP problem formulation (skeleton)."""

from dataclasses import dataclass

from acados_template import AcadosOcp

from vehicle_nmpc.models import ModelBundle
from vehicle_nmpc.problem.base import BaseProblem, BaseProblemConfig, ProblemBundle
from vehicle_nmpc.problem.builder import register_problem


@register_problem("acados_ocp")
class AcadosOcpProblem(BaseProblem):
    """Builds an Acados OCP from a model bundle."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseProblemConfig):
        """Acados OCP problem configuration."""

    def __init__(self, cfg: Config, model: ModelBundle) -> None:
        """Initialize OCP problem with configuration and model bundle."""
        super().__init__(cfg, model)

    def build(self) -> ProblemBundle:
        """Build and return the Acados OCP problem bundle."""
        ocp = AcadosOcp()
        ocp.model = self._model.model
        return ProblemBundle(ocp=ocp)

"""Problem abstractions for NMPC formulations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from acados_template import AcadosOcp

    from vehicle_nmpc.models import ModelBundle
    from vehicle_nmpc.utils.config import ProblemConfig

from vehicle_nmpc.utils.factory import ConfiguredBase


@dataclass
class ProblemBundle:
    """Container with the built problem and its metadata."""

    ocp: AcadosOcp


class BaseProblem(ConfiguredBase, ABC):
    """Abstract problem interface."""

    Config: ClassVar[type[ProblemConfig]]

    def __init__(self, cfg: ProblemConfig, model: ModelBundle) -> None:
        """Initialize problem with configuration and model bundle."""
        super().__init__(cfg)
        self._model = model

    @abstractmethod
    def build(self) -> ProblemBundle:
        """Build and return the problem bundle."""
        raise NotImplementedError

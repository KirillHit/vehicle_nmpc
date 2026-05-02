"""Base abstractions and configuration types for NMPC models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import numpy as np
    from acados_template import AcadosModel

from vehicle_nmpc.utils.factory import ConfiguredBase


@dataclass(kw_only=True, slots=True)
class BaseModelConfig:
    """Base configuration for model implementations."""


@dataclass(kw_only=True, slots=True)
class ModelBundle:
    """Container with the built model and its dimensions/metadata."""

    model: AcadosModel
    nx: int
    nu: int
    np: int
    x0: np.ndarray
    x_labels: list[str] | None = None
    u_labels: list[str] | None = None


class BaseModel(ConfiguredBase, ABC):
    """Abstract base class for NMPC models."""

    Config: ClassVar[type[BaseModelConfig]] = BaseModelConfig

    @abstractmethod
    def build(self) -> ModelBundle:
        """Build and return the model bundle."""
        raise NotImplementedError

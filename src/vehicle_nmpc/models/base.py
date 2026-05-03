"""Base abstractions and configuration types for NMPC models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol

if TYPE_CHECKING:
    import numpy as np
    from acados_template import AcadosModel

from vehicle_nmpc.utils.factory import ConfiguredBase


@dataclass(kw_only=True, slots=True)
class BaseModelConfig:
    """Base configuration for model implementations."""


class TrajectoryReferenceModel(Protocol):
    """Model-specific adapter that converts path references to controls."""

    def control_reference(self, speed: np.ndarray, yaw_rate: np.ndarray) -> np.ndarray:
        """Convert nominal path speed and yaw rate to model controls."""


@dataclass(kw_only=True, slots=True)
class ModelBundle:
    """Container with the built model and its dimensions/metadata."""

    model: AcadosModel
    """Acados model representation."""

    nx: int
    """Number of model states."""

    nu: int
    """Number of model controls."""

    np: int
    """Number of model parameters."""

    p0: np.ndarray
    """Default model parameter vector."""

    x0: np.ndarray
    """Default initial state."""

    trajectory_reference_model: TrajectoryReferenceModel | None = None
    """Optional model-specific adapter for trajectory control references."""


class BaseModel(ConfiguredBase, ABC):
    """Abstract base class for NMPC models."""

    Config: ClassVar[type[BaseModelConfig]] = BaseModelConfig

    @abstractmethod
    def build(self) -> ModelBundle:
        """Build and return the model bundle."""
        raise NotImplementedError

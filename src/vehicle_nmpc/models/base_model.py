"""Base abstractions and configuration types for NMPC models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from omegaconf import MISSING

if TYPE_CHECKING:
    import numpy as np

    from acados_template import AcadosModel


@dataclass
class BaseModelConfig:
    """Top-level model configuration."""

    name: str = MISSING


@dataclass
class ModelBundle:
    """Container with the built model and its dimensions/metadata."""

    model: AcadosModel
    nx: int
    nu: int
    np: int
    x0: np.ndarray
    x_labels: list[str] | None = None
    u_labels: list[str] | None = None


class BaseModel(ABC):
    """Abstract base class for NMPC models."""

    def __init__(self, cfg: BaseModelConfig) -> None:
        """Initialize the model with its configuration."""
        super().__init__()
        self._cfg = cfg

    @abstractmethod
    def build(self) -> ModelBundle:
        """Build and return the model bundle."""
        raise NotImplementedError

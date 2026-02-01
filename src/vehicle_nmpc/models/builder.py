"""Model registry and factory utilities."""

import logging
from collections.abc import Callable
from typing import TypeVar

from vehicle_nmpc.models.base import BaseModel
from vehicle_nmpc.utils.config import ModelConfig
from vehicle_nmpc.utils.exceptions import ModelCreationError
from vehicle_nmpc.utils.factory import build_configured_instance, register_in_registry

TModel = TypeVar("TModel", bound=BaseModel)
log = logging.getLogger(__name__)


MODEL_REGISTRY: dict[str, type[BaseModel]] = {}


def build_model(cfg: ModelConfig) -> BaseModel:
    """Build model instance from registry."""
    log.info("Creating model '%s'...", cfg.name)

    return build_configured_instance(
        cfg,
        MODEL_REGISTRY,
        BaseModel,
        ModelCreationError,
        "model",
    )


def register_model(name: str) -> Callable[[type[TModel]], type[TModel]]:
    """Register a model class under a given name."""
    return register_in_registry(name, MODEL_REGISTRY, BaseModel, ModelCreationError, "model")

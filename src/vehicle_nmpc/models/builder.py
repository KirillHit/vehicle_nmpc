"""Model registry and factory utilities."""

import logging
from collections.abc import Callable
from typing import TypeVar

from vehicle_nmpc.models.base_model import BaseModel, BaseModelConfig
from vehicle_nmpc.utils.exceptions import ModelCreationError

TModel = TypeVar("TModel", bound=BaseModel)
log = logging.getLogger(__name__)


MODEL_REGISTRY: dict[str, type[BaseModel]] = {}


def build_model(cfg: BaseModelConfig) -> BaseModel:
    """Build model instance from registry."""
    log.info("Creating model '%s'...", cfg.name)

    model_class = MODEL_REGISTRY.get(cfg.name)

    if model_class is None:
        available = ", ".join(sorted(MODEL_REGISTRY)) or "<empty>"
        msg = f"Unsupported model name: {cfg.name}. Available models: {available}"
        raise ModelCreationError(msg)

    return model_class(cfg)


def register_model(name: str) -> Callable[[type[TModel]], type[TModel]]:
    """Register a model class under a given name."""

    def decorator(cls: type[TModel]) -> type[TModel]:
        """Register the model class."""
        if not issubclass(cls, BaseModel):
            msg = f"Registered class must inherit BaseModel, got: {cls!r}"
            raise TypeError(msg)

        if name in MODEL_REGISTRY:
            msg = f"Model name '{name}' is already registered"
            raise ModelCreationError(msg)

        MODEL_REGISTRY[name] = cls
        return cls

    return decorator

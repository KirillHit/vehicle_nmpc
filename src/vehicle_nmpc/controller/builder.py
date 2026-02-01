"""Controller registry and factory utilities."""

import logging
from collections.abc import Callable
from typing import TypeVar

from vehicle_nmpc.controller.base import BaseController
from vehicle_nmpc.models import ModelBundle
from vehicle_nmpc.problem import ProblemBundle
from vehicle_nmpc.utils.config import ControllerConfig
from vehicle_nmpc.utils.exceptions import ControllerCreationError
from vehicle_nmpc.utils.factory import build_configured_instance, register_in_registry

TController = TypeVar("TController", bound=BaseController)
log = logging.getLogger(__name__)

CONTROLLER_REGISTRY: dict[str, type[BaseController]] = {}


def build_controller(
    cfg: ControllerConfig,
    problem: ProblemBundle,
    model: ModelBundle,
) -> BaseController:
    """Build controller instance from registry."""
    log.info("Creating controller '%s'...", cfg.name)

    return build_configured_instance(
        cfg,
        CONTROLLER_REGISTRY,
        BaseController,
        ControllerCreationError,
        "controller",
        problem,
        model,
    )


def register_controller(name: str) -> Callable[[type[TController]], type[TController]]:
    """Register a controller class under a given name."""
    return register_in_registry(
        name,
        CONTROLLER_REGISTRY,
        BaseController,
        ControllerCreationError,
        "controller",
    )

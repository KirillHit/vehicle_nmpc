"""Controller registry and factory utilities."""

import logging
from collections.abc import Callable
from typing import TypeVar

from vehicle_nmpc.controller.base import BaseController
from vehicle_nmpc.models import ModelBundle
from vehicle_nmpc.problem import ProblemBundle
from vehicle_nmpc.utils.config import FactoryConfig
from vehicle_nmpc.utils.exceptions import ControllerCreationError
from vehicle_nmpc.utils.factory import RegistrySpec, build_configured_instance, register_in_registry

TController = TypeVar("TController", bound=BaseController)
log = logging.getLogger(__name__)

CONTROLLER_REGISTRY: dict[str, type[BaseController]] = {}
CONTROLLER_SPEC = RegistrySpec(
    registry=CONTROLLER_REGISTRY,
    base_cls=BaseController,
    error_cls=ControllerCreationError,
    kind="controller",
)


def build_controller(
    cfg: FactoryConfig,
    problem: ProblemBundle,
    model: ModelBundle,
) -> BaseController:
    """Build controller instance from registry."""
    log.info("Creating controller '%s'...", cfg.name)

    return build_configured_instance(
        cfg,
        CONTROLLER_SPEC,
        dependencies={"problem": problem, "model": model},
    )


def register_controller(name: str) -> Callable[[type[TController]], type[TController]]:
    """Register a controller class under a given name."""
    return register_in_registry(name, CONTROLLER_SPEC)

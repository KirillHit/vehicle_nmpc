"""Simulator registry and factory utilities."""

import logging
from collections.abc import Callable
from typing import TypeVar

from vehicle_nmpc.models import ModelBundle
from vehicle_nmpc.problem import ProblemBundle
from vehicle_nmpc.sim.base import BaseSimulator
from vehicle_nmpc.utils.config import FactoryConfig
from vehicle_nmpc.utils.exceptions import SimulatorCreationError
from vehicle_nmpc.utils.factory import RegistrySpec, build_configured_instance, register_in_registry

TSim = TypeVar("TSim", bound=BaseSimulator)
log = logging.getLogger(__name__)

SIM_REGISTRY: dict[str, type[BaseSimulator]] = {}
SIM_SPEC = RegistrySpec(
    registry=SIM_REGISTRY,
    base_cls=BaseSimulator,
    error_cls=SimulatorCreationError,
    kind="simulator",
)


def build_simulator(
    cfg: FactoryConfig,
    problem: ProblemBundle,
    model: ModelBundle,
) -> BaseSimulator:
    """Build simulator instance from registry."""
    log.info("Creating simulator '%s'...", cfg.name)

    return build_configured_instance(
        cfg,
        SIM_SPEC,
        dependencies={"problem": problem, "model": model},
    )


def register_simulator(name: str) -> Callable[[type[TSim]], type[TSim]]:
    """Register a simulator class under a given name."""
    return register_in_registry(name, SIM_SPEC)

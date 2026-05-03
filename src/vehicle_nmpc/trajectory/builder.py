"""Trajectory registry and factory utilities."""

import logging
from collections.abc import Callable
from typing import TypeVar

from vehicle_nmpc.models import ModelBundle
from vehicle_nmpc.problem import ProblemBundle
from vehicle_nmpc.trajectory.base import BaseTrajectoryProvider
from vehicle_nmpc.utils.config import FactoryConfig
from vehicle_nmpc.utils.exceptions import TrajectoryCreationError
from vehicle_nmpc.utils.factory import RegistrySpec, build_configured_instance, register_in_registry

TTrajectory = TypeVar("TTrajectory", bound=BaseTrajectoryProvider)
log = logging.getLogger(__name__)

TRAJECTORY_REGISTRY: dict[str, type[BaseTrajectoryProvider]] = {}
TRAJECTORY_SPEC = RegistrySpec(
    registry=TRAJECTORY_REGISTRY,
    base_cls=BaseTrajectoryProvider,
    error_cls=TrajectoryCreationError,
    kind="trajectory",
)


def build_trajectory(
    cfg: FactoryConfig,
    model: ModelBundle,
    problem: ProblemBundle,
) -> BaseTrajectoryProvider:
    """Build trajectory provider from registry."""
    log.info("Creating trajectory '%s'...", cfg.name)

    try:
        return build_configured_instance(
            cfg,
            TRAJECTORY_SPEC,
            dependencies={"model": model, "problem": problem},
        )
    except TrajectoryCreationError:
        raise
    except Exception as exc:
        msg = f"Failed to build trajectory '{cfg.name}': {exc}"
        raise TrajectoryCreationError(msg) from exc


def register_trajectory(name: str) -> Callable[[type[TTrajectory]], type[TTrajectory]]:
    """Register a trajectory provider class under a given name."""
    return register_in_registry(name, TRAJECTORY_SPEC)

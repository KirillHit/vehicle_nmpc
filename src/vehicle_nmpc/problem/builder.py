"""Problem registry and factory utilities."""

import logging
from collections.abc import Callable
from typing import TypeVar

from vehicle_nmpc.models import ModelBundle
from vehicle_nmpc.problem.base import BaseProblem, ProblemBundle
from vehicle_nmpc.utils.config import FactoryConfig
from vehicle_nmpc.utils.exceptions import ProblemCreationError
from vehicle_nmpc.utils.factory import RegistrySpec, build_configured_instance, register_in_registry

TProblem = TypeVar("TProblem", bound="BaseProblem")
log = logging.getLogger(__name__)

PROBLEM_REGISTRY: dict[str, type["BaseProblem"]] = {}
PROBLEM_SPEC = RegistrySpec(
    registry=PROBLEM_REGISTRY,
    base_cls=BaseProblem,
    error_cls=ProblemCreationError,
    kind="problem",
)


def build_problem(cfg: FactoryConfig, model: ModelBundle) -> ProblemBundle:
    """Build problem bundle from registry."""
    log.info("Creating problem '%s'...", cfg.name)

    try:
        problem = build_configured_instance(cfg, PROBLEM_SPEC, dependencies={"model": model})
        return problem.build()
    except ProblemCreationError:
        raise
    except Exception as exc:
        msg = f"Failed to build problem '{cfg.name}': {exc}"
        raise ProblemCreationError(msg) from exc


def register_problem(name: str) -> Callable[[type[TProblem]], type[TProblem]]:
    """Register a problem class under a given name."""
    return register_in_registry(name, PROBLEM_SPEC)

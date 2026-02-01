"""Problem registry and factory utilities."""

import logging
from collections.abc import Callable
from typing import TypeVar

from vehicle_nmpc.models import ModelBundle
from vehicle_nmpc.problem.base import BaseProblem
from vehicle_nmpc.utils.config import ProblemConfig
from vehicle_nmpc.utils.exceptions import ProblemCreationError
from vehicle_nmpc.utils.factory import build_configured_instance, register_in_registry

TProblem = TypeVar("TProblem", bound="BaseProblem")
log = logging.getLogger(__name__)

PROBLEM_REGISTRY: dict[str, type["BaseProblem"]] = {}


def build_problem(cfg: ProblemConfig, model: ModelBundle) -> BaseProblem:
    """Build problem instance from registry."""
    log.info("Creating problem '%s'...", cfg.name)

    return build_configured_instance(
        cfg,
        PROBLEM_REGISTRY,
        BaseProblem,
        ProblemCreationError,
        "problem",
        model,
    )


def register_problem(name: str) -> Callable[[type[TProblem]], type[TProblem]]:
    """Register a problem class under a given name."""
    return register_in_registry(
        name,
        PROBLEM_REGISTRY,
        BaseProblem,
        ProblemCreationError,
        "problem",
    )

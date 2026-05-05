"""Utilities for Hydra configuration registration."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


class RunnerMode(StrEnum):
    """Supported runner modes."""

    estimate = "estimate"
    """Evaluate one configured controller and report its metrics."""

    optimize = "optimize"
    """Tune controller parameters with Optuna by repeatedly running estimate."""

    export = "export"
    """Generate acados C code artifacts for the configured model, problem, and solver."""


@dataclass(kw_only=True, slots=True)
class RunnerConfig:
    """Top-level runner configuration."""

    seed: int = 42
    experiment_name: str = MISSING
    mode: RunnerMode = MISSING
    n_sim: int = 100
    """Number of closed-loop simulation steps."""

    output_dir: str = MISSING
    """Directory where run artifacts are written."""


@dataclass(kw_only=True, slots=True)
class FactoryConfig:
    """Registry-backed component configuration."""

    name: str = MISSING
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True, slots=True)
class OptunaParameter:
    """Definition of a single hyperparameter to optimize and its search space."""

    parameter: str = MISSING
    """Full path to the parameter in the config."""

    type: str = "float"
    """Parameter type: float, int, or categorical."""

    low: float | None = None
    """Lower bound for float and int parameters."""

    high: float | None = None
    """Upper bound for float and int parameters."""

    log: bool = False
    """Whether to sample a float parameter on a log scale."""

    choices: list[Any] | None = None
    """Allowed values for categorical parameters."""


@dataclass(kw_only=True, slots=True)
class ObjectiveComponent:
    """Weighted metric component used to build an Optuna objective."""

    metric: str = MISSING
    """Dotted path to a field in EvaluationMetrics."""

    aggregation: str = "mean"
    """Aggregation over evaluated trajectories: mean, max, min, or sum."""

    weight: float = 1.0
    """Multiplier applied to the aggregated metric value."""


@dataclass(kw_only=True, slots=True)
class OptunaObjective:
    """Composite Optuna objective configuration."""

    direction: str = "minimize"
    """Optimization direction passed to Optuna."""

    components: list[ObjectiveComponent] = field(default_factory=list)
    """Weighted metric components summed into one scalar objective."""


@dataclass(kw_only=True, slots=True)
class OptunaConfig:
    """Optuna optimization configuration."""

    n_trials: int = 50
    """Number of trials to run."""

    parameters: list[OptunaParameter] = field(default_factory=list)
    """Hyperparameters optimized by Optuna."""

    objective: OptunaObjective = field(default_factory=OptunaObjective)
    """Scalar objective assembled from configured metric components."""


@dataclass(kw_only=True, slots=True)
class BaseConfig:
    """Root app configuration schema for Hydra."""

    runner: RunnerConfig = field(default_factory=RunnerConfig)
    model: FactoryConfig = field(default_factory=FactoryConfig)
    problem: FactoryConfig = field(default_factory=FactoryConfig)
    controller: FactoryConfig = field(default_factory=FactoryConfig)
    sim: FactoryConfig = field(default_factory=FactoryConfig)
    trajectories: list[FactoryConfig] = field(default_factory=list)
    optuna: OptunaConfig = field(default_factory=OptunaConfig)


def register_configs() -> None:
    """Register the root config schema in Hydra's ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=BaseConfig)

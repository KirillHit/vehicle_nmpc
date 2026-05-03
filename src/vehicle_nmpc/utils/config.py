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


@dataclass(kw_only=True, slots=True)
class FactoryConfig:
    """Registry-backed component configuration."""

    name: str = MISSING
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True, slots=True)
class BaseConfig:
    """Root app configuration schema for Hydra."""

    runner: RunnerConfig = field(default_factory=RunnerConfig)
    model: FactoryConfig = field(default_factory=FactoryConfig)
    problem: FactoryConfig = field(default_factory=FactoryConfig)
    controller: FactoryConfig = field(default_factory=FactoryConfig)
    sim: FactoryConfig = field(default_factory=FactoryConfig)
    trajectories: list[FactoryConfig] = field(default_factory=list)


def register_configs() -> None:
    """Register the root config schema in Hydra's ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=BaseConfig)

"""Utilities for Hydra configuration registration."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


class RunnerMode(StrEnum):
    """Supported runner modes."""

    open_loop = "open_loop"
    """Run the controller without feeding simulator state back into the control loop."""

    closed_loop = "closed_loop"
    """Run the controller with simulator state feedback after each step."""


@dataclass
class RunnerConfig:
    """Top-level runner configuration."""

    seed: int = 42
    experiment_name: str = MISSING
    mode: RunnerMode = MISSING


@dataclass
class ModelConfig:
    """Top-level model configuration."""

    name: str = MISSING
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProblemConfig:
    """Top-level problem configuration."""

    name: str = MISSING
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ControllerConfig:
    """Top-level controller configuration."""

    name: str = MISSING
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimConfig:
    """Top-level simulator configuration."""

    name: str = MISSING
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class BaseConfig:
    """Root app configuration schema for Hydra."""

    runner: RunnerConfig = field(default_factory=RunnerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    problem: ProblemConfig = field(default_factory=ProblemConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    sim: SimConfig = field(default_factory=SimConfig)


def register_configs() -> None:
    """Register the root config schema in Hydra's ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=BaseConfig)

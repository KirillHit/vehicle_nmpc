"""Utilities for Hydra configuration registration."""

from dataclasses import dataclass, field
from enum import Enum

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from vehicle_nmpc.models import BaseModelConfig, TrackedVehKinematicConfig


@dataclass
class ControllerConfig:
    """Top-level controller configuration."""

    name: str = MISSING


class RunnerMode(str, Enum):
    """Supported runner modes."""

    open_loop = "open_loop"
    closed_loop = "closed_loop"


@dataclass
class RunnerConfig:
    """Top-level runner configuration."""

    seed: int = 42
    experiment_name: str = MISSING
    mode: RunnerMode = MISSING


@dataclass
class BaseConfig:
    """Root app configuration schema for Hydra."""

    model: BaseModelConfig = MISSING
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)


def register_configs() -> None:
    """Register the root config schema in Hydra's ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=BaseConfig)
    cs.store(group="model", name="base_tracked_veh_kinematic", node=TrackedVehKinematicConfig)

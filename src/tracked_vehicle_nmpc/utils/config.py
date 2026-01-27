"""Utilities for Hydra configuration registration."""

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from tracked_vehicle_nmpc.models import BaseModelConfig, TrackedVehKinematicConfig


@dataclass
class ControllerConfig:
    """Top-level controller configuration."""

    name: str = MISSING


@dataclass
class ExecutorConfig:
    """Top-level runner configuration."""

    seed: int = 42
    experiment_name: str = MISSING
    mode: str = MISSING


@dataclass
class BaseConfig:
    """Root app configuration schema for Hydra."""

    model: BaseModelConfig = MISSING
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)


def register_configs() -> None:
    """Register the root config schema in Hydra's ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=BaseConfig)
    cs.store(group="model", name="base_tracked_veh_kinematic", node=TrackedVehKinematicConfig)

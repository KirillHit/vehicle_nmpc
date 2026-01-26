"""Utilities for Hydra configuration registration."""

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class ModelConfig:
    """Top-level model configuration."""

    name: str


@dataclass
class ControllerConfig:
    """Top-level controller configuration."""

    name: str


@dataclass
class MyConfig:
    """Root app configuration schema for Hydra."""

    model: ModelConfig = field(default_factory=ModelConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)

    seed: int = 42


def init_hydra() -> None:
    """Register the root config schema in Hydra's ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(name="config", node=MyConfig)

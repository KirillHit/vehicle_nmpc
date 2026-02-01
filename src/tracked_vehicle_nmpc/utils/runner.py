"""Execution dispatch for tracked vehicle NMPC runs."""

import logging

from tracked_vehicle_nmpc.models import BaseModel, BaseModelConfig, TrackedVehKinematic
from tracked_vehicle_nmpc.utils.config import BaseConfig, RunnerMode

log = logging.getLogger(__name__)

MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "tracked_veh_kinematic": TrackedVehKinematic,
}


class Runner:
    """Dispatch execution based on the configured run mode."""

    def run(self, cfg: BaseConfig) -> None:
        """Run the training loop using the provided configuration.

        Args:
            cfg (Namespace): Configuration namespace.

        """
        log.info("Starting experiment in mode '%s'...", cfg.runner.mode.value)
        log.info("Experiment name: %s", cfg.runner.experiment_name)

        match cfg.runner.mode:
            case RunnerMode.open_loop:
                self.open_loop(cfg)
            case RunnerMode.closed_loop:
                self.closed_loop(cfg)
            case _:
                msg = f"Unsupported mode: {cfg.runner.mode.value}"
                raise ValueError(msg)

        log.info("Experiment finished.")

    def create_model(self, cfg: BaseModelConfig) -> BaseModel:
        """Create a model instance from the configured model name."""

    def open_loop(self, cfg: BaseConfig) -> None:
        """Run the open-loop execution path."""

    def closed_loop(self, cfg: BaseConfig) -> None:
        """Run the closed-loop execution path."""

"""Execution dispatch for tracked vehicle NMPC runs."""

import logging

from tracked_vehicle_nmpc.models import BaseModel, BaseModelConfig, TrackedVehKinematic
from tracked_vehicle_nmpc.utils.config import BaseConfig

log = logging.getLogger(__name__)

MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "tracked_veh_kinematic": TrackedVehKinematic,
}


class Executor:
    """Dispatch execution based on the configured run mode."""

    def run(self, cfg: BaseConfig) -> None:
        """Run the training loop using the provided configuration.

        Args:
            cfg (Namespace): Configuration namespace.

        """
        log.info("Starting Executor in mode: %s", cfg.executor.mode)
        log.info("Experiment name: %s", cfg.executor.experiment_name)

        match cfg.executor.mode:
            case "open_loop":
                self.open_loop(cfg)
            case "closed_loop":
                self.closed_loop(cfg)
            case _:
                log.error(
                    "Unsupported mode: %s! Available modes: open_loop, closed_loop.",
                    cfg.executor.mode,
                )

        log.info("Executor finished.")

    def create_model(self, cfg: BaseModelConfig) -> BaseModel:
        """Create a model instance from the configured model name."""

    def open_loop(self, cfg: BaseConfig) -> None:
        """Run the open-loop execution path."""

    def closed_loop(self, cfg: BaseConfig) -> None:
        """Run the closed-loop execution path."""

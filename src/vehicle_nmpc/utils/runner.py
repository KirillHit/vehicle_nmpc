"""Execution dispatch for tracked vehicle NMPC runs."""

import logging

from vehicle_nmpc.utils.config import BaseConfig, RunnerMode

log = logging.getLogger(__name__)


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

    def open_loop(self, cfg: BaseConfig) -> None:
        """Run the open-loop execution path."""

    def closed_loop(self, cfg: BaseConfig) -> None:
        """Run the closed-loop execution path."""

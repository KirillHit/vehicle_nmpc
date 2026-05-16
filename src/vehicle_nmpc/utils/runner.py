"""Execution dispatch for tracked vehicle NMPC runs."""

from __future__ import annotations

import logging

from vehicle_nmpc.experiments.closed_loop import (
    prepare_closed_loop,
    run_configured_evaluation,
    save_evaluation_artifacts,
)
from vehicle_nmpc.experiments.optimization import run_optimization
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
            case RunnerMode.estimate:
                self._run_estimate(cfg)
            case RunnerMode.optimize:
                run_optimization(cfg)
            case RunnerMode.export:
                prepare_closed_loop(cfg)
            case _:
                msg = f"Unsupported mode: {cfg.runner.mode.value}"
                raise ValueError(msg)

        log.info("Experiment finished.")

    def _run_estimate(self, cfg: BaseConfig) -> None:
        """Evaluate one configured controller on configured tracking trajectories."""
        results, metrics, dt = run_configured_evaluation(cfg)
        save_evaluation_artifacts(results, metrics, cfg.runner.output_dir, dt=dt)
        for item in metrics:
            log.info(item.print())

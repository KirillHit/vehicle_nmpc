"""Execution dispatch for tracked vehicle NMPC runs."""

import logging

from vehicle_nmpc.controller import build_controller
from vehicle_nmpc.models import build_model
from vehicle_nmpc.problem import build_problem
from vehicle_nmpc.sim import build_simulator
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
            case RunnerMode.estimate | RunnerMode.optimize | RunnerMode.export:
                self._prepare_run(cfg)
            case _:
                msg = f"Unsupported mode: {cfg.runner.mode.value}"
                raise ValueError(msg)

        log.info("Experiment finished.")

    def _prepare_run(self, cfg: BaseConfig) -> None:
        """Build and reset configured run components."""
        model = build_model(cfg.model).build()
        problem = build_problem(cfg.problem, model).build()
        controller = build_controller(cfg.controller, problem, model)
        simulator = build_simulator(cfg.sim, problem, model)

        controller.reset(model.x0)
        simulator.reset(model.x0)

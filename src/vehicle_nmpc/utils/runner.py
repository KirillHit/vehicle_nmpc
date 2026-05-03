"""Execution dispatch for tracked vehicle NMPC runs."""

import logging

from vehicle_nmpc.controller import build_controller
from vehicle_nmpc.experiments.closed_loop import run_closed_loop
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
            case RunnerMode.estimate:
                self._run_estimate(cfg)
            case RunnerMode.optimize | RunnerMode.export:
                self._prepare_run(cfg)
            case _:
                msg = f"Unsupported mode: {cfg.runner.mode.value}"
                raise ValueError(msg)

        log.info("Experiment finished.")

    def _run_estimate(self, cfg: BaseConfig) -> None:
        """Run one configured controller in closed loop."""
        model, _problem, controller, simulator = self._prepare_run(cfg)
        result = run_closed_loop(
            controller,
            simulator,
            model,
            n_steps=cfg.runner.n_sim,
        )

        log.info("Closed-loop simulation steps: %d", cfg.runner.n_sim)
        log.info("Terminal state: %s", result.states[-1])
        log.info("Mean control: %s", result.controls.mean(axis=0))
        self._log_timing_stats(result.stats)

    def _prepare_run(self, cfg: BaseConfig) -> None:
        """Build and reset configured run components."""
        model = build_model(cfg.model).build()
        problem = build_problem(cfg.problem, model).build()
        controller = build_controller(cfg.controller, problem, model)
        simulator = build_simulator(cfg.sim, problem, model)

        controller.reset(model.x0)
        simulator.reset(model.x0)
        return model, problem, controller, simulator

    @staticmethod
    def _log_timing_stats(stats: list[dict]) -> None:
        """Log controller timing statistics when available."""
        if not stats:
            return

        preparation_times = [
            item["preparation_time"] for item in stats if "preparation_time" in item
        ]
        feedback_times = [item["feedback_time"] for item in stats if "feedback_time" in item]
        if preparation_times:
            mean_preparation_ms = 1000 * sum(preparation_times) / len(preparation_times)
            log.info("Mean RTI preparation time: %.3f ms", mean_preparation_ms)
        if feedback_times:
            mean_feedback_ms = 1000 * sum(feedback_times) / len(feedback_times)
            log.info("Mean RTI feedback time: %.3f ms", mean_feedback_ms)

"""Execution dispatch for tracked vehicle NMPC runs."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from vehicle_nmpc.controller import BaseController, build_controller
from vehicle_nmpc.experiments.closed_loop import evaluate_result, run_evaluation
from vehicle_nmpc.metrics import save_trajectory_plot
from vehicle_nmpc.models import ModelBundle, build_model
from vehicle_nmpc.problem import ProblemBundle, build_problem
from vehicle_nmpc.sim import BaseSimulator, build_simulator
from vehicle_nmpc.trajectory import BaseTrajectoryProvider, build_trajectory
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
        """Evaluate one configured controller on configured tracking trajectories."""
        model, problem, controller, simulator, trajectories = self._prepare_run(cfg)
        if not trajectories:
            msg = "Estimate mode requires at least one configured trajectory."
            raise ValueError(msg)

        results = run_evaluation(
            controller,
            simulator,
            model,
            trajectories,
            n_steps=cfg.runner.n_sim,
        )
        metrics = [evaluate_result(result, dt=problem.dt) for result in results]
        for result, item in zip(results, metrics, strict=True):
            save_trajectory_plot(
                result.states,
                result.reference_states,
                self._artifact_path(cfg, "plots", result.trajectory_name, ".png"),
                title=result.trajectory_name,
            )
            log.info(item.print())

    def _prepare_run(
        self, cfg: BaseConfig
    ) -> tuple[
        ModelBundle,
        ProblemBundle,
        BaseController,
        BaseSimulator,
        list[BaseTrajectoryProvider],
    ]:
        """Build and reset configured run components."""
        model = build_model(cfg.model)
        problem = build_problem(cfg.problem, model)
        controller = build_controller(cfg.controller, problem, model)
        simulator = build_simulator(cfg.sim, problem, model)
        trajectories = [
            build_trajectory(trajectory_cfg, model, problem) for trajectory_cfg in cfg.trajectories
        ]

        controller.reset(model.x0)
        simulator.reset(model.x0)
        return model, problem, controller, simulator, trajectories

    def _artifact_path(self, cfg: BaseConfig, group: str, name: str, suffix: str) -> Path:
        """Return output path for a run artifact."""
        safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_") or "artifact"
        return Path(cfg.runner.output_dir) / group / f"{safe_name}{suffix}"
